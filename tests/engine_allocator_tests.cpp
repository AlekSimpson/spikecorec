#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <gtest/gtest.h>

#include "spikecorec/core/types.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/engine_allocator.h"

using namespace spikecorec;

namespace {

bool is_sixteen_byte_aligned(const void *pointer) {
    return reinterpret_cast<uintptr_t>(pointer) % EngineAllocator::ALLOCATION_ALIGNMENT == 0;
}

} // namespace

// ── arena ────────────────────────────────────────────────────────────────────

TEST(EngineAllocator, cpu_allocations_are_sequential_and_do_not_overlap) {
    EngineAllocator arena(1024, 0);

    u8 *first_range = static_cast<u8 *>(arena.allocate_cpu(sizeof(u8), 100));
    u8 *second_range = static_cast<u8 *>(arena.allocate_cpu(sizeof(u8), 100));
    u8 *third_range = static_cast<u8 *>(arena.allocate_cpu(sizeof(u8), 100));

    ASSERT_NE(first_range, nullptr);
    ASSERT_NE(second_range, nullptr);
    ASSERT_NE(third_range, nullptr);

    EXPECT_GE(second_range, first_range + 100);
    EXPECT_GE(third_range, second_range + 100);

    // Filling each range must leave the others untouched.
    for (s64 index = 0; index < 100; index += 1) {
        first_range[index] = 1;
        second_range[index] = 2;
        third_range[index] = 3;
    }
    for (s64 index = 0; index < 100; index += 1) {
        EXPECT_EQ(first_range[index], 1u);
        EXPECT_EQ(second_range[index], 2u);
        EXPECT_EQ(third_range[index], 3u);
    }
}

TEST(EngineAllocator, cpu_cursor_advances_by_the_requested_size) {
    EngineAllocator arena(1024, 0);
    EXPECT_EQ(arena.cpu_cursor, 0u);

    arena.allocate_cpu(sizeof(f32), 10);
    EXPECT_EQ(arena.cpu_cursor, 40u);

    // The next range starts at the following 16-byte boundary (48), not at 40.
    arena.allocate_cpu(sizeof(u8), 1);
    EXPECT_EQ(arena.cpu_cursor, 49u);

    arena.allocate_cpu(sizeof(s64), 4);
    EXPECT_EQ(arena.cpu_cursor, 64u + 32u);
}

TEST(EngineAllocator, cpu_allocations_are_sixteen_byte_aligned) {
    EngineAllocator arena(1024, 0);

    // Odd sizes leave the cursor unaligned; every returned pointer must still be aligned.
    for (s64 length = 1; length <= 20; length += 1) {
        void *range = arena.allocate_cpu(sizeof(u8), length);
        ASSERT_NE(range, nullptr) << length;
        EXPECT_TRUE(is_sixteen_byte_aligned(range)) << length;
    }
}

TEST(EngineAllocator, cpu_request_beyond_capacity_throws) {
    EngineAllocator arena(64, 0);

    arena.allocate_cpu(sizeof(u8), 48);
    EXPECT_THROW(arena.allocate_cpu(sizeof(u8), 32), std::runtime_error);

    // The failed request must not have consumed anything, so what does fit still fits.
    EXPECT_NO_THROW(arena.allocate_cpu(sizeof(u8), 16));
}

TEST(EngineAllocator, zero_byte_requests_return_null_and_do_not_advance) {
    EngineAllocator arena(1024, 0);

    EXPECT_EQ(arena.allocate_cpu(sizeof(f32), 0), nullptr);
    EXPECT_EQ(arena.cpu_cursor, 0u);

    EXPECT_EQ(arena.allocate_cpu(0, 10), nullptr);
    EXPECT_EQ(arena.cpu_cursor, 0u);
}

TEST(EngineAllocator, zero_sized_arena_constructs_and_destructs_cleanly) {
    {
        EngineAllocator default_arena;
        EXPECT_EQ(default_arena.cpu_total_bytes, 0u);
        EXPECT_EQ(default_arena.gpu_total_bytes, 0u);
        EXPECT_EQ(default_arena.pool_cpu_memory, nullptr);
        EXPECT_EQ(default_arena.pool_gpu_memory.pointer, nullptr);
        EXPECT_EQ(default_arena.allocate_cpu(sizeof(f32), 0), nullptr);
        EXPECT_THROW(default_arena.allocate_cpu(sizeof(f32), 1), std::runtime_error);
    }
    {
        // An explicitly zero-sized arena must not call the backend allocator either.
        EngineAllocator empty_arena(0, 0);
        EXPECT_EQ(empty_arena.pool_cpu_memory, nullptr);
        EXPECT_EQ(empty_arena.pool_gpu_memory.pointer, nullptr);
        EXPECT_THROW(empty_arena.allocate_gpu(sizeof(f32), 1), std::runtime_error);
    }
}

TEST(EngineAllocator, gpu_allocations_advance_the_gpu_cursor) {
    EngineAllocator arena(0, 1024);
    ASSERT_NE(arena.pool_gpu_memory.pointer, nullptr);
    EXPECT_EQ(arena.gpu_cursor, 0u);

    GpuPointer<void> first_range = arena.allocate_gpu(sizeof(f32), 8);
    EXPECT_NE(first_range.pointer, nullptr);
    EXPECT_TRUE(is_sixteen_byte_aligned(first_range.get_contents()));
    EXPECT_EQ(arena.gpu_cursor, 32u);

    GpuPointer<void> second_range = arena.allocate_gpu(sizeof(u8), 1);
    EXPECT_NE(second_range.pointer, nullptr);
    EXPECT_TRUE(is_sixteen_byte_aligned(second_range.get_contents()));
    EXPECT_EQ(arena.gpu_cursor, 33u);

    EXPECT_THROW(arena.allocate_gpu(sizeof(f32), 1024), std::runtime_error);
}

TEST(EngineAllocator, gpu_sub_ranges_start_where_get_contents_says_they_do) {
    EngineAllocator arena(0, 1024);
    const u8 *slab_base = static_cast<const u8 *>(arena.pool_gpu_memory.get_contents());
    ASSERT_NE(slab_base, nullptr);

    GpuPointer<void> first_range = arena.allocate_gpu(sizeof(f32), 8);
    GpuPointer<void> second_range = arena.allocate_gpu(sizeof(f32), 8);

    // Whatever a sub-range handle is made of per backend, the address it reports is the
    // slab base plus the cursor the request was served from.
    EXPECT_EQ(static_cast<const u8 *>(first_range.get_contents()), slab_base);
    EXPECT_EQ(static_cast<const u8 *>(second_range.get_contents()), slab_base + 32);

    // Sub-ranges are distinct memory even though they share one allocation.
    static_cast<u8 *>(first_range.get_contents())[0] = 11;
    static_cast<u8 *>(second_range.get_contents())[0] = 22;
    EXPECT_EQ(static_cast<const u8 *>(first_range.get_contents())[0], 11u);
    EXPECT_EQ(static_cast<const u8 *>(second_range.get_contents())[0], 22u);
}

TEST(EngineAllocator, moving_transfers_the_slabs_and_empties_the_source) {
    EngineAllocator source(256, 0);
    u8 *range = static_cast<u8 *>(source.allocate_cpu(sizeof(u8), 16));
    ASSERT_NE(range, nullptr);
    range[0] = 7;

    EngineAllocator moved_to = std::move(source);
    EXPECT_EQ(moved_to.cpu_total_bytes, 256u);
    EXPECT_EQ(moved_to.cpu_cursor, 16u);
    EXPECT_EQ(range[0], 7u);

    EXPECT_EQ(source.cpu_total_bytes, 0u);       // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(source.cpu_cursor, 0u);
    EXPECT_EQ(source.pool_cpu_memory, nullptr);

    // Assigning over a live arena releases what it held instead of leaking it.
    EngineAllocator assigned_over(128, 0);
    assigned_over.allocate_cpu(sizeof(u8), 64);
    assigned_over = std::move(moved_to);
    EXPECT_EQ(assigned_over.cpu_total_bytes, 256u);
    EXPECT_EQ(assigned_over.cpu_cursor, 16u);
    EXPECT_EQ(range[0], 7u);
}

TEST(EngineAllocator, moving_an_arena_holding_a_gpu_slab_does_not_double_free) {
    // Both arenas go out of scope here: if the move left the source owning the slab too,
    // the second release would fault or corrupt the backend's allocation table.
    EngineAllocator source(0, 512);
    GpuPointer<void> range = source.allocate_gpu(sizeof(f32), 4);
    ASSERT_NE(range.pointer, nullptr);

    EngineAllocator moved_to = std::move(source);
    EXPECT_EQ(moved_to.gpu_total_bytes, 512u);
    EXPECT_EQ(moved_to.gpu_cursor, 16u);
    EXPECT_EQ(source.pool_gpu_memory.pointer, nullptr);  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(source.gpu_total_bytes, 0u);

    EngineAllocator assigned_over(0, 128);
    assigned_over = std::move(moved_to);
    EXPECT_EQ(assigned_over.gpu_total_bytes, 512u);
    EXPECT_EQ(assigned_over.gpu_cursor, 16u);
}

// ── address → buffer binding registry (Metal) ────────────────────────────────
// Metal binds buffer objects, not addresses, so an arena sub-range can only reach a kernel as
// (slab buffer, its own offset). allocate_gpu registers that pair under the sub-range's
// address; resolve_buffer_binding is what metal_dispatch reads it back with. There is nothing
// to resolve on CUDA, where a device pointer is already what the kernel wants.

#ifdef SPIKECOREC_METAL

TEST(BufferBindingRegistry, a_whole_allocation_resolves_to_itself_at_offset_zero) {
    GpuPointer<u8> allocation = allocate<u8>(4096);
    ASSERT_NE(allocation.pointer, nullptr);

    BufferBinding binding = resolve_buffer_binding(allocation.get_contents());
    EXPECT_EQ(binding.platform_handle, static_cast<void *>(allocation.pointer));
    EXPECT_EQ(binding.offset_bytes, 0u);

    // Nothing registered an interior address, so one does not resolve.
    EXPECT_EQ(resolve_buffer_binding(allocation.get_contents() + 64).platform_handle, nullptr);

    deallocate(std::move(allocation));
}

TEST(BufferBindingRegistry, an_unregistered_address_resolves_to_nothing) {
    u8 host_stack_byte = 0;
    EXPECT_EQ(resolve_buffer_binding(&host_stack_byte).platform_handle, nullptr);
    EXPECT_EQ(resolve_buffer_binding(&host_stack_byte).offset_bytes, 0u);
    EXPECT_EQ(resolve_buffer_binding(nullptr).platform_handle, nullptr);
}

TEST(BufferBindingRegistry, a_deallocated_buffer_no_longer_resolves) {
    GpuPointer<u8> allocation = allocate<u8>(4096);
    ASSERT_NE(allocation.pointer, nullptr);

    const u8 *base_address = allocation.get_contents();
    ASSERT_EQ(resolve_buffer_binding(base_address).platform_handle,
              static_cast<void *>(allocation.pointer));

    deallocate(std::move(allocation));

    EXPECT_EQ(resolve_buffer_binding(base_address).platform_handle, nullptr);
}

TEST(BufferBindingRegistry, an_arena_sub_range_resolves_to_the_slab_at_its_own_offset) {
    EngineAllocator arena(0, 4096);
    GpuPointer<void> first_range = arena.allocate_gpu(sizeof(f32), 8);
    GpuPointer<void> second_range = arena.allocate_gpu(sizeof(f32), 8);
    void *slab_handle = static_cast<void *>(arena.pool_gpu_memory.pointer);

    // This is the property metal_dispatch depends on: the address a sub-range reports binds
    // as (slab buffer, that sub-range's offset), not as the slab from byte zero.
    BufferBinding first_binding = resolve_buffer_binding(first_range.get_contents());
    EXPECT_EQ(first_binding.platform_handle, slab_handle);
    EXPECT_EQ(first_binding.offset_bytes, 0u);

    BufferBinding second_binding = resolve_buffer_binding(second_range.get_contents());
    EXPECT_EQ(second_binding.platform_handle, slab_handle);
    EXPECT_EQ(second_binding.offset_bytes, 32u);

    EXPECT_EQ(arena.registered_gpu_sub_range_addresses.size(), 2u);
}

TEST(BufferBindingRegistry, destroying_an_arena_withdraws_its_sub_range_registrations) {
    const u8 *first_address = nullptr;
    const u8 *second_address = nullptr;
    {
        EngineAllocator arena(0, 4096);
        first_address = static_cast<const u8 *>(arena.allocate_gpu(sizeof(f32), 8).get_contents());
        second_address = static_cast<const u8 *>(arena.allocate_gpu(sizeof(f32), 8).get_contents());
        ASSERT_NE(resolve_buffer_binding(second_address).platform_handle, nullptr);
    }

    // The slab is gone; an address that outlived it must not still name a released buffer,
    // which is what a later allocation recycling the address would otherwise be handed.
    EXPECT_EQ(resolve_buffer_binding(first_address).platform_handle, nullptr);
    EXPECT_EQ(resolve_buffer_binding(second_address).platform_handle, nullptr);
}

TEST(BufferBindingRegistry, moving_an_arena_keeps_its_sub_ranges_registered_exactly_once) {
    const u8 *sub_range_address = nullptr;
    {
        EngineAllocator source(0, 4096);
        source.allocate_gpu(sizeof(f32), 8);
        sub_range_address =
                static_cast<const u8 *>(source.allocate_gpu(sizeof(f32), 8).get_contents());

        EngineAllocator moved_to = std::move(source);
        EXPECT_EQ(source.registered_gpu_sub_range_addresses.size(), 0u);  // NOLINT
        EXPECT_EQ(moved_to.registered_gpu_sub_range_addresses.size(), 2u);

        // The moved-from arena releasing nothing means the binding is still live and correct.
        BufferBinding binding = resolve_buffer_binding(sub_range_address);
        EXPECT_EQ(binding.platform_handle, static_cast<void *>(moved_to.pool_gpu_memory.pointer));
        EXPECT_EQ(binding.offset_bytes, 32u);
    }

    // ...and the surviving arena withdrew them when it went out of scope.
    EXPECT_EQ(resolve_buffer_binding(sub_range_address).platform_handle, nullptr);
}

#endif // SPIKECOREC_METAL
