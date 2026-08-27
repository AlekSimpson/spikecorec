#pragma once

#include <algorithm>
#include <cassert>

#ifdef SPIKECOREC_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include "spikecorec/core/types.h"

namespace spikecorec {

enum class EngineDatatype {
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

struct EnginePointer {
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

    template <typename ElementType>
    [[nodiscard]] ElementType *get_contents_as() const {
        return static_cast<ElementType *>(get_contents());
    }

    [[nodiscard]] bool is_empty() const {
        return base_pointer == nullptr || total_bytes == 0;
    }
};

template <typename ValueType>
[[nodiscard]] inline EnginePointer inline_scalar_argument(const ValueType &value) {
    return EnginePointer{
        const_cast<void *>(static_cast<const void *>(&value)),
        0,
        sizeof(ValueType),
        alignof(ValueType),
        true,
    };
}

struct EngineFunction {
#ifdef SPIKECOREC_CUDA
    CUfunction cuda_function{};
    CUmodule cuda_module{};
#elif defined(SPIKECOREC_METAL)
    MTL::ComputePipelineState *pipeline_state = nullptr;
#endif
};

struct AbstractBackend {
    UnorderedMap<EngineDatatype, u64> type_alignments = {
        {EngineDatatype::SIGNED32, 4},
        {EngineDatatype::SIGNED64, 8},
        {EngineDatatype::FLOAT32, 4},
        {EngineDatatype::FLOAT64, 8},
        {EngineDatatype::FLOAT32X4, 16},
        {EngineDatatype::UNSIGNED8, 1},
        {EngineDatatype::UNSIGNED16, 2},
        {EngineDatatype::UNSIGNED32, 4},
        {EngineDatatype::UNSIGNED64, 8},
        {EngineDatatype::BOOLEAN, 1}
    };

    // Metal wants a 256-byte offset for a buffer bound to a `constant` kernel parameter,
    // and float4 reads need 16.
    static constexpr u64 PARTITION_ALIGNMENT = 256;

    s64 last_offset = 0;
    u64 max_bytes = 0;
    s64 threads_per_block = 256;
    bool memory_has_been_allocated = false;
    EnginePointer base_pointer;
};

#ifdef SPIKECOREC_METAL

struct MetalBackend : AbstractBackend {
    MTL::Device *device = nullptr;
    MTL::CommandQueue *queue = nullptr;
    MTL::Library *default_library = nullptr;
    Vector<MTL::Buffer *> slabs;

    explicit MetalBackend(s64 threads_per_block_argument = 256);
    ~MetalBackend();

    MetalBackend(const MetalBackend &) = delete;
    MetalBackend &operator=(const MetalBackend &) = delete;
    MetalBackend(MetalBackend &&) = delete;
    MetalBackend &operator=(MetalBackend &&) = delete;

    Optional<EngineFunction> create_function(const String &name, const String &source_code);
    Optional<EngineFunction> load_precompiled_function(const String &name);
    void release_function(EngineFunction &function);
    bool run_function(EngineFunction &function, Vector<EnginePointer> &parameters, s64 job_count);

    MetalBackend &partition(u64 bytes, EngineDatatype datatype, Vector<EnginePointer> &partitions);
    EnginePointer allocate(Vector<EnginePointer> &partitions);

    // Releases one chunk, named by the handle allocate() returned for it.
    void deallocate_slab(const EnginePointer &slab);
    void deallocate_all();

    // Placement hints. No-ops here: Metal buffers already live in host-visible shared
    // storage, so there is nothing to migrate and nothing to replicate. They exist so
    // callers that know their access pattern -- the k^2-tree's arrays are immutable once
    // built, for instance -- can say so once and have it matter on CUDA.
    void advise_read_mostly(const EnginePointer &, u64) {}
    void prefetch_to_gpu(const EnginePointer &, u64) {}
    void prefetch_to_cpu(const EnginePointer &, u64) {}
};

using EngineBackend = MetalBackend;

#elif defined(SPIKECOREC_CUDA)

struct CudaBackend : AbstractBackend {
    CUdevice cuda_gpu_device = 0;
    CUcontext cuda_context = nullptr;
    Vector<void *> slabs;

    explicit CudaBackend(s64 threads_per_block_argument = 256);
    ~CudaBackend();

    CudaBackend(const CudaBackend &) = delete;
    CudaBackend &operator=(const CudaBackend &) = delete;
    CudaBackend(CudaBackend &&) = delete;
    CudaBackend &operator=(CudaBackend &&) = delete;

    Optional<EngineFunction> create_function(const String &name, const String &source_code);
    Optional<EngineFunction> load_precompiled_function(const String &name);
    void release_function(EngineFunction &function);

    bool run_function(EngineFunction &function, Vector<EnginePointer> &parameters, s64 job_count);

    CudaBackend &partition(u64 bytes, EngineDatatype datatype, Vector<EnginePointer> &partitions);
    EnginePointer allocate(Vector<EnginePointer> &partitions);

    void deallocate_slab(const EnginePointer &slab);
    void deallocate_all();

    // Managed-memory placement hints. read_mostly keeps per-device read-only replicas of
    // a range that is written once and read constantly (the k^2-tree's arrays), instead
    // of migrating it to whichever processor touched it last. The prefetches turn a
    // page-by-page fault-in, driven one touch at a time, into a single bulk copy.
    void advise_read_mostly(const EnginePointer &range, u64 byte_count);
    void prefetch_to_gpu(const EnginePointer &range, u64 byte_count);
    void prefetch_to_cpu(const EnginePointer &range, u64 byte_count);
};

using EngineBackend = CudaBackend;

#endif

} // namespace spikecorec
