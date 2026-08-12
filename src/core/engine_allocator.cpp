#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "spikecorec/core/engine_allocator.h"

namespace spikecorec {

namespace {

// Next multiple of the arena's alignment at or above offset.
u64 align_offset_upwards(u64 offset) {
    const u64 alignment = EngineAllocator::ALLOCATION_ALIGNMENT;
    return (offset + alignment - 1) / alignment * alignment;
}

u64 requested_byte_count(u64 element_bytes, s64 length) {
    if (length <= 0) return 0;
    return element_bytes * static_cast<u64>(length);
}

// Advances cursor past an aligned sub-range of request_bytes and returns where that
// sub-range starts, or throws when the slab has no room left for it.
u64 reserve_sub_range(u64 &cursor, u64 total_bytes, u64 request_bytes, const char *pool_name) {
    const u64 start_offset = align_offset_upwards(cursor);
    const u64 remaining_bytes = start_offset >= total_bytes ? 0 : total_bytes - start_offset;

    if (request_bytes > remaining_bytes) {
        throw std::runtime_error(
                std::string("EngineAllocator: ") + pool_name + " pool exhausted -- requested " +
                std::to_string(request_bytes) + " bytes, " + std::to_string(remaining_bytes) +
                " bytes remaining");
    }

    cursor = start_offset + request_bytes;
    return start_offset;
}

void release_slabs(EngineAllocator &arena) {
    if (arena.pool_cpu_memory != nullptr) {
        ::operator delete(arena.pool_cpu_memory,
                          std::align_val_t(EngineAllocator::ALLOCATION_ALIGNMENT));
        arena.pool_cpu_memory = nullptr;
    }

    // Withdraw the sub-range bindings before the slab goes, so no address outlives the buffer
    // it resolves to.
    for (void *sub_range_address : arena.registered_gpu_sub_range_addresses) {
        unregister_buffer_sub_range(sub_range_address);
    }
    arena.registered_gpu_sub_range_addresses.clear();

    if (arena.pool_gpu_memory.pointer != nullptr) {
        // deallocate() takes its argument by value, so the move leaves the member null.
        deallocate(std::move(arena.pool_gpu_memory));
    }

    arena.cpu_total_bytes = 0;
    arena.gpu_total_bytes = 0;
    arena.cpu_cursor = 0;
    arena.gpu_cursor = 0;
}

} // namespace

EngineAllocator::EngineAllocator() = default;

EngineAllocator::EngineAllocator(u64 cpu_total_bytes, u64 gpu_total_bytes)
    : cpu_total_bytes(cpu_total_bytes), gpu_total_bytes(gpu_total_bytes) {
    if (cpu_total_bytes > 0) {
        pool_cpu_memory = ::operator new(static_cast<usize>(cpu_total_bytes),
                                         std::align_val_t(ALLOCATION_ALIGNMENT));
    }
    if (gpu_total_bytes > 0) {
        pool_gpu_memory = allocate<void>(static_cast<usize>(gpu_total_bytes));
    }
}

EngineAllocator::~EngineAllocator() {
    release_slabs(*this);
}

EngineAllocator::EngineAllocator(EngineAllocator &&other) noexcept
    : cpu_total_bytes(other.cpu_total_bytes),
      gpu_total_bytes(other.gpu_total_bytes),
      cpu_cursor(other.cpu_cursor),
      gpu_cursor(other.gpu_cursor),
      pool_cpu_memory(other.pool_cpu_memory),
      pool_gpu_memory(std::move(other.pool_gpu_memory)),
      registered_gpu_sub_range_addresses(
              std::move(other.registered_gpu_sub_range_addresses)) {
    other.cpu_total_bytes = 0;
    other.gpu_total_bytes = 0;
    other.cpu_cursor = 0;
    other.gpu_cursor = 0;
    other.pool_cpu_memory = nullptr;
    other.registered_gpu_sub_range_addresses.clear();
}

EngineAllocator &EngineAllocator::operator=(EngineAllocator &&other) noexcept {
    if (this != &other) {
        release_slabs(*this);

        cpu_total_bytes = other.cpu_total_bytes;
        gpu_total_bytes = other.gpu_total_bytes;
        cpu_cursor = other.cpu_cursor;
        gpu_cursor = other.gpu_cursor;
        pool_cpu_memory = other.pool_cpu_memory;
        pool_gpu_memory = std::move(other.pool_gpu_memory);
        registered_gpu_sub_range_addresses =
                std::move(other.registered_gpu_sub_range_addresses);

        other.cpu_total_bytes = 0;
        other.gpu_total_bytes = 0;
        other.cpu_cursor = 0;
        other.gpu_cursor = 0;
        other.pool_cpu_memory = nullptr;
        other.registered_gpu_sub_range_addresses.clear();
    }
    return *this;
}

void *EngineAllocator::allocate_cpu(u64 element_bytes, s64 length) {
    const u64 request_bytes = requested_byte_count(element_bytes, length);
    if (request_bytes == 0) return nullptr;

    const u64 start_offset =
            reserve_sub_range(cpu_cursor, cpu_total_bytes, request_bytes, "CPU");
    return static_cast<u8 *>(pool_cpu_memory) + start_offset;
}

GpuPointer<void> EngineAllocator::allocate_gpu(u64 element_bytes, s64 length) {
    GpuPointer<void> sub_range;

    const u64 request_bytes = requested_byte_count(element_bytes, length);
    if (request_bytes == 0) return sub_range;

    const u64 start_offset =
            reserve_sub_range(gpu_cursor, gpu_total_bytes, request_bytes, "GPU");

    // Two spellings of the same range, one per backend -- see the allocate_gpu comment in
    // engine_allocator.h. Either way get_contents() lands on the sub-range's first byte.
#ifdef SPIKECOREC_CUDA
    sub_range.pointer = static_cast<u8 *>(pool_gpu_memory.pointer) + start_offset;
#elif defined(SPIKECOREC_METAL)
    sub_range.pointer = pool_gpu_memory.pointer;
    sub_range.offset_bytes = start_offset;

    // Teach the backend what this address binds to, so passing it to a kernel reaches that
    // sub-range rather than falling through to the pass-a-scalar path.
    void *sub_range_address = sub_range.get_contents();
    register_buffer_sub_range(sub_range_address, pool_gpu_memory.pointer, start_offset);
    registered_gpu_sub_range_addresses.push_back(sub_range_address);
#endif

    return sub_range;
}

} // namespace spikecorec
