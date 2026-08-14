#pragma once

#include "spikecorec/core/types.h"
#include "spikecorec/core/backend.h"

namespace spikecorec {

// A bump arena over exactly two slabs: one host allocation and one device allocation,
// each sized once by the engine from the parsed model. allocate_cpu/allocate_gpu carve a
// sub-range out of their slab by advancing a cursor, and that is the whole mechanism --
// there is no per-allocation bookkeeping and no free list. Individual sub-ranges are never
// released; only the two slabs are, when the arena dies. Every buffer an engine owns lives
// exactly as long as the engine does, which is what makes that trade sound.
//
// A request that does not fit throws std::runtime_error naming the shortfall, rather than
// returning null for the caller to trip over somewhere further downstream.
struct EngineAllocator {
    // Every sub-range starts on this boundary: GPU buffers and the float4-typed reads in
    // the kernels require at least 16-byte alignment.
    static constexpr u64 ALLOCATION_ALIGNMENT = 16;

    u64 cpu_total_bytes = 0;
    u64 gpu_total_bytes = 0;

    // Byte offsets into their slabs, not allocation counts. u64 rather than a 32-bit type
    // because an arena sized to hold a whole network's cell state passes 2 GB routinely.
    u64 cpu_cursor = 0;
    u64 gpu_cursor = 0;

    void *pool_cpu_memory = nullptr;
    GpuPointer<void> pool_gpu_memory;

    // Metal only, and empty on CUDA. Every sub-range address allocate_gpu has registered with
    // the backend, so the arena can withdraw them all before it releases the slab. Leaving one
    // behind would let a later allocation that happens to land on the same address resolve to
    // the released buffer.
    Vector<void *> registered_gpu_sub_range_addresses;

    // An empty arena owning no slabs: every non-zero request throws. The engine holds one
    // of these until it has parsed a model and knows what to size the real arena to.
    EngineAllocator();

    // Allocates one host slab of cpu_total_bytes and one device slab of gpu_total_bytes.
    // Either size may be zero, in which case that side gets no slab at all and the backend
    // allocator is not called.
    EngineAllocator(u64 cpu_total_bytes, u64 gpu_total_bytes);

    ~EngineAllocator();

    // Movable so the engine can assign a properly sized arena over its empty one once the
    // sizes are known; not copyable, because two arenas sharing one pair of slabs would
    // release them twice.
    EngineAllocator(const EngineAllocator &) = delete;
    EngineAllocator &operator=(const EngineAllocator &) = delete;
    EngineAllocator(EngineAllocator &&other) noexcept;
    EngineAllocator &operator=(EngineAllocator &&other) noexcept;

    // A sub-range of element_bytes * length bytes at the current cursor, 16-byte aligned.
    // Returns nullptr for a zero-byte request. Never free the result: the arena owns it.
    void *allocate_cpu(u64 element_bytes, s64 length);

    // The device counterpart, and the same ownership rule: the returned handle aliases the
    // slab and owns nothing, so it must never be passed to deallocate() -- that would release
    // the whole slab out from under every other sub-range. deallocate() asserts against it for
    // every sub-range but the one at offset 0, which is indistinguishable from the slab handle
    // itself; the arena's destructor is the only code that ever releases a pool.
    //
    // Both backends agree on where the sub-range starts: get_contents() on the returned handle
    // is its first byte, and handing that address to a kernel binds exactly that range and
    // nothing before it. They reach that agreement differently, because a sub-range is not the
    // same kind of thing on each side (see GpuPointer in backend.h):
    //
    //   CUDA  -- `pointer` is the sub-range's own device address; nothing else is needed.
    //   Metal -- `pointer` is the slab's MTL::Buffer, since Metal has no sub-buffer object, and
    //            offset_bytes carries the sub-range's offset within it. The sub-range's address
    //            is also registered with the backend as (slab buffer, offset), so passing it to
    //            a kernel binds via setBuffer(buffer, offset, index) -- Metal binds buffer
    //            objects rather than addresses, so the offset has to arrive that way.
    //
    // The consequence to remember is that on Metal two sub-ranges of one arena compare equal
    // by `.pointer`; only get_contents() distinguishes them.
    GpuPointer<void> allocate_gpu(u64 element_bytes, s64 length);
};

} // namespace spikecorec
