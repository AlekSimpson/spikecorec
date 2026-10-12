#pragma once

#ifdef SPIKECOREC_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include "types.h"

namespace spikecorec {

enum class Engine_Datatype {
    SIGNED32,
    SIGNED64,
    FLOAT32,
    FLOAT64,
    FLOAT32X4,
    UNSIGNED8,
    UNSIGNED16,
    UNSIGNED32,
    UNSIGNED64,
    BOOLEAN,
};

struct Engine_Pointer {
    void *base_pointer = nullptr;
    s64 offset = 0;
    u64 total_bytes = 0;
    u64 alignment = 1;
    bool inline_scalar = false;

    [[nodiscard]] void *get_contents() const {
        if (base_pointer == nullptr) return nullptr;
        if (inline_scalar) return base_pointer;
#ifdef SPIKECOREC_METAL
        return static_cast<u8 *>(static_cast<MTL::Buffer *>(base_pointer)->contents()) + offset;
#else
        return static_cast<u8 *>(base_pointer) + offset;
#endif
    }

    template <typename Element_Type>
    [[nodiscard]] Element_Type *get_contents_as() const {
        return static_cast<Element_Type *>(get_contents());
    }

    [[nodiscard]] bool is_empty() const {
        return base_pointer == nullptr || total_bytes == 0;
    }
};

template <typename Value_Type>
[[nodiscard]] inline Engine_Pointer inline_scalar_argument(const Value_Type &value) {
    return Engine_Pointer{
        const_cast<void *>(static_cast<const void *>(&value)),
        0,
        sizeof(Value_Type),
        alignof(Value_Type),
        true,
    };
}

struct Engine_Function {
#ifdef SPIKECOREC_CUDA
    CUfunction cuda_function{};
    CUmodule cuda_module{};
#elif defined(SPIKECOREC_METAL)
    MTL::ComputePipelineState *pipeline_state = nullptr;
#endif
};

static const Unordered_Map<Engine_Datatype, u64> TYPE_ALIGNMENTS = {
    {Engine_Datatype::SIGNED32, 4},
    {Engine_Datatype::SIGNED64, 8},
    {Engine_Datatype::FLOAT32, 4},
    {Engine_Datatype::FLOAT64, 8},
    {Engine_Datatype::FLOAT32X4, 16},
    {Engine_Datatype::UNSIGNED8, 1},
    {Engine_Datatype::UNSIGNED16, 2},
    {Engine_Datatype::UNSIGNED32, 4},
    {Engine_Datatype::UNSIGNED64, 8},
    {Engine_Datatype::BOOLEAN, 1}
};

#ifdef SPIKECOREC_METAL

// @note:

struct Engine_Backend {
    // Metal wants a 256-byte offset for a buffer bound to a `constant` kernel parameter,
    // and float4 reads need 16.
    static constexpr u64 PARTITION_ALIGNMENT = 256;

    s64 last_offset = 0;
    u64 max_bytes = 0;
    s64 threads_per_block = 256;
    bool memory_has_been_allocated = false;
    Engine_Pointer base_pointer;

    MTL::Device *device = nullptr;
    MTL::CommandQueue *queue = nullptr;
    MTL::Library *default_library = nullptr;
    Vector<MTL::Buffer *> slabs;

    explicit Engine_Backend(s64 threads_per_block_argument = 256);
    ~Engine_Backend();

    Engine_Backend(const Engine_Backend &) = delete;
    Engine_Backend &operator=(const Engine_Backend &) = delete;
    Engine_Backend(Engine_Backend &&) = delete;
    Engine_Backend &operator=(Engine_Backend &&) = delete;

    Optional<Engine_Function> create_function(const String &name, const String &source_code);
    Optional<Engine_Function> load_precompiled_function(const String &name);
    void release_function(Engine_Function &function);
    bool run_function(Engine_Function &function, Vector<Engine_Pointer> &parameters, s64 job_count);

    Engine_Backend &partition(u64 bytes, Engine_Datatype datatype, Vector<Engine_Pointer> &partitions);
    Engine_Pointer allocate(Vector<Engine_Pointer> &partitions);

    // Releases one chunk, named by the handle allocate() returned for it.
    void deallocate_slab(const Engine_Pointer &slab);
    void deallocate_all();

    // Placement hints. No-ops here: Metal buffers already live in host-visible shared
    // storage, so there is nothing to migrate and nothing to replicate. They exist so
    // callers that know their access pattern -- the k^2-tree's arrays are immutable once
    // built, for instance -- can say so once and have it matter on CUDA.
    void advise_read_mostly(const Engine_Pointer &, u64) {}
    void prefetch_to_gpu(const Engine_Pointer &, u64) {}
    void prefetch_to_cpu(const Engine_Pointer &, u64) {}
};

#elif defined(SPIKECOREC_CUDA)

struct Engine_Backend {
    // Metal wants a 256-byte offset for a buffer bound to a `constant` kernel parameter,
    // and float4 reads need 16.
    static constexpr u64 PARTITION_ALIGNMENT = 256;

    s64 last_offset = 0;
    u64 max_bytes = 0;
    s64 threads_per_block = 256;
    bool memory_has_been_allocated = false;
    Engine_Pointer base_pointer;

    CUdevice cuda_gpu_device = 0;
    CUcontext cuda_context = nullptr;
    Vector<void *> slabs;

    explicit Engine_Backend(s64 threads_per_block_argument = 256);
    ~Engine_Backend();

    Engine_Backend(const Engine_Backend &) = delete;
    Engine_Backend &operator=(const Engine_Backend &) = delete;
    Engine_Backend(Engine_Backend &&) = delete;
    Engine_Backend &operator=(Engine_Backend &&) = delete;

    Optional<Engine_Function> create_function(const String &name, const String &source_code);
    Optional<Engine_Function> load_precompiled_function(const String &name);
    void release_function(Engine_Function &function);

    bool run_function(Engine_Function &function, Vector<Engine_Pointer> &parameters, s64 job_count);

    Engine_Backend &partition(u64 bytes, Engine_Datatype datatype, Vector<Engine_Pointer> &partitions);
    Engine_Pointer allocate(Vector<Engine_Pointer> &partitions);

    void deallocate_slab(const Engine_Pointer &slab);
    void deallocate_all();

    // Managed-memory placement hints. read_mostly keeps per-device read-only replicas of
    // a range that is written once and read constantly (the k^2-tree's arrays), instead
    // of migrating it to whichever processor touched it last. The prefetches turn a
    // page-by-page fault-in, driven one touch at a time, into a single bulk copy.
    void advise_read_mostly(const Engine_Pointer &range, u64 byte_count);
    void prefetch_to_gpu(const Engine_Pointer &range, u64 byte_count);
    void prefetch_to_cpu(const Engine_Pointer &range, u64 byte_count);
};

#endif

} // namespace spikecorec
