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
    CUmodule cuda_module{};  // null for a precompiled kernel, which the CUDA runtime owns
    // Set for the precompiled kernels, which read every inline argument through a pointer, as their
    // Metal versions read a constant-address-space argument (see cuda/precompiled_kernels.h).
    bool inline_arguments_by_reference = false;
#elif defined(SPIKECOREC_METAL)
    MTL::ComputePipelineState *pipeline_state = nullptr;
#endif
};

// One kernel launch for Engine_Backend::run_functions: the function, its positional arguments and
// its thread count. Inline scalars are read when the launch is encoded, inside run_functions.
struct Engine_Dispatch {
    Engine_Function *function = nullptr;
    Vector<Engine_Pointer> parameters;
    s64 job_count = 0;
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


struct Engine_Backend {
    // Metal needs 256-byte offsets for buffers bound to `constant` parameters; float4 needs 16.
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

    // Bytes held by the live slabs, and the most they have held since the last reset.
    u64 allocated_byte_count = 0;
    u64 peak_allocated_byte_count = 0;
    void reset_peak_allocated_byte_count() { peak_allocated_byte_count = allocated_byte_count; }

    explicit Engine_Backend(s64 threads_per_block_argument = 256);
    ~Engine_Backend();

    Engine_Backend(const Engine_Backend &) = delete;
    Engine_Backend &operator=(const Engine_Backend &) = delete;
    Engine_Backend(Engine_Backend &&) = delete;
    Engine_Backend &operator=(Engine_Backend &&) = delete;

    Optional<Engine_Function> create_function(const String &name, const String &source_code) const;
    Optional<Engine_Function> load_precompiled_function(const String &name) const;
    static void release_function(Engine_Function &function);
    bool run_function(Engine_Function &function, Vector<Engine_Pointer> &parameters, s64 job_count) const;
    // Runs the launches in order in one command buffer and waits once; each sees the writes of
    // those before it. False when the command buffer fails.
    bool run_functions(Vector<Engine_Dispatch> &dispatches) const;

    Engine_Backend &partition(u64 bytes, Engine_Datatype datatype, Vector<Engine_Pointer> &partitions);
    Engine_Pointer allocate(Vector<Engine_Pointer> &partitions);

    // Releases one slab, named by the handle allocate() returned for it.
    void deallocate_slab(const Engine_Pointer &slab);
    void deallocate_all();

    // Placement hints: no-ops, since Metal buffers use shared storage. They take effect on CUDA.
    void advise_read_mostly(const Engine_Pointer &, u64) {}
    void prefetch_to_gpu(const Engine_Pointer &, u64) {}
    void prefetch_to_cpu(const Engine_Pointer &, u64) {}
};

#elif defined(SPIKECOREC_CUDA)

struct Engine_Backend {
    // Matches the Metal backend, so both lay out partitions alike.
    static constexpr u64 PARTITION_ALIGNMENT = 256;

    s64 last_offset = 0;
    u64 max_bytes = 0;
    s64 threads_per_block = 256;
    bool memory_has_been_allocated = false;
    Engine_Pointer base_pointer;

    CUdevice cuda_gpu_device = 0;
    CUcontext cuda_context = nullptr;
    Vector<void *> slabs;
    Vector<u64> slab_byte_counts;  // parallel to slabs

    // Bytes held by the live slabs, and the most they have held since the last reset.
    u64 allocated_byte_count = 0;
    u64 peak_allocated_byte_count = 0;
    void reset_peak_allocated_byte_count() { peak_allocated_byte_count = allocated_byte_count; }

    // For a launch of a kernel with inline_arguments_by_reference: run_function packs its inline
    // arguments into inline_argument_host_bytes, copies them to the device memory at
    // inline_argument_staging, and passes their addresses. The mutex keeps one such launch at a time.
    void *inline_argument_staging = nullptr;
    u64 inline_argument_staging_capacity = 0;
    Vector<u8> inline_argument_host_bytes;
    Mutex inline_argument_mutex;

    explicit Engine_Backend(s64 threads_per_block_argument = 256);
    ~Engine_Backend();

    Engine_Backend(const Engine_Backend &) = delete;
    Engine_Backend &operator=(const Engine_Backend &) = delete;
    Engine_Backend(Engine_Backend &&) = delete;
    Engine_Backend &operator=(Engine_Backend &&) = delete;

    Optional<Engine_Function> create_function(const String &name, const String &source_code);
    // The kernel `name` from the library's precompiled kernels (cuda/precompiled_kernels.h), or
    // nothing when there is none by the name or threads_per_block is not a whole number of warps.
    // The caller then answers on the host.
    Optional<Engine_Function> load_precompiled_function(const String &name);
    void release_function(Engine_Function &function);

    bool run_function(Engine_Function &function, Vector<Engine_Pointer> &parameters, s64 job_count);
    // Runs the launches in order, each through run_function. False at the first that fails.
    bool run_functions(Vector<Engine_Dispatch> &dispatches);

    Engine_Backend &partition(u64 bytes, Engine_Datatype datatype, Vector<Engine_Pointer> &partitions);
    Engine_Pointer allocate(Vector<Engine_Pointer> &partitions);

    void deallocate_slab(const Engine_Pointer &slab);
    void deallocate_all();

    // Managed-memory hints. read_mostly replicates a write-once range (such as the k^2-tree)
    // on each device instead of migrating it; a prefetch replaces page faults with one bulk copy.
    void advise_read_mostly(const Engine_Pointer &range, u64 byte_count);
    void prefetch_to_gpu(const Engine_Pointer &range, u64 byte_count);
    void prefetch_to_cpu(const Engine_Pointer &range, u64 byte_count);
};

#endif

} // namespace spikecorec
