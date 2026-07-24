#pragma once

#include <cassert>
#include "spikecorec/core/types.h"

#ifdef SPIKECOREC_CUDA
#include <cuda.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

namespace spikecorec {
    // Called once at program startup to initialize the GPU context.
    void initialize_gpu_context();

    // Called once at program shutdown to release GPU resources.
    void release_gpu_resources();

    template<typename T>
    struct GpuPointer {
    #ifdef SPIKECOREC_CUDA
        T *pointer = nullptr;

        T *get_contents() {
            return pointer;
        }

        const T *get_contents() const { 
            return pointer; 
        }
    #elif defined(SPIKECOREC_METAL)
        MTL::Buffer *pointer = nullptr;

        T *get_contents() { 
            return static_cast<T *>(pointer->contents()); 
        }

        const T *get_contents() const { 
            return static_cast<const T *>(pointer->contents()); 
        }
    #endif

        GpuPointer() = default;

        GpuPointer(const GpuPointer &) = delete;

        GpuPointer &operator=(const GpuPointer &) = delete;

        GpuPointer(GpuPointer &&other) noexcept {
            pointer = other.pointer;
            other.pointer = nullptr;
        }

        GpuPointer &operator=(GpuPointer &&other) noexcept {
            assert(pointer == nullptr &&
                   "GpuPointer move assignment would leak an existing CUDA allocation — "
                   "call backend::deallocate() before reassigning");

            if (this != &other) {
                pointer = other.pointer;
                other.pointer = nullptr;
            }
            return *this;
        }
    };

    // --- buffer ---
    // non-template bridges — defined in backend.cpp, hide platform types from callers
    void *allocate_bytes(usize byte_size);
    void deallocate_bytes(void *platform_handle);

    template<typename T>
    GpuPointer<T> allocate(usize byte_size) {
        GpuPointer<T> wrapper;
    #ifdef SPIKECOREC_CUDA
        wrapper.pointer = static_cast<T *>(allocate_bytes(byte_size));
    #elif defined(SPIKECOREC_METAL)
        wrapper.pointer = static_cast<MTL::Buffer *>(allocate_bytes(byte_size));
    #endif
        return wrapper;
    }

    template<typename T>
    void deallocate(GpuPointer<T> wrapper) {
        deallocate_bytes(wrapper.pointer);
    }

    // CUDA hint, no-op on Metal
    template<typename T>
    void prefetch_to_gpu(const GpuPointer<T> &wrapper_pointer, usize size_in_bytes) {
    #ifdef SPIKECOREC_CUDA
        cudaMemPrefetchAsync(wrapper_pointer.pointer, size_in_bytes, 0);
    #elif defined(SPIKECOREC_METAL)
        (void)size_in_bytes;
        (void)wrapper_pointer;
    #endif
    }

    // CUDA hint: prefetch managed memory back to the host before the CPU reads
    // it (e.g. before memcpy'ing a GPU-written buffer out to a numpy array or a
    // recorder frame) — turns a synchronous page-by-page fault-in, triggered
    // one touch at a time from the host thread, into a single bulk async copy.
    // No-op on Metal, where buffers already live in host-visible shared storage.
    template<typename T>
    void prefetch_to_cpu(const GpuPointer<T> &wrapper_pointer, usize size_in_bytes) {
    #ifdef SPIKECOREC_CUDA
        cudaMemPrefetchAsync(wrapper_pointer.pointer, size_in_bytes, cudaCpuDeviceId);
    #elif defined(SPIKECOREC_METAL)
        (void)size_in_bytes;
        (void)wrapper_pointer;
    #endif
    }

    // CUDA hint: advise the driver that a managed allocation is read far more
    // than it's written (e.g. the k^2-tree adjacency arrays, immutable once
    // built) so it keeps per-device read-only replicas instead of migrating
    // the whole allocation to whichever processor touches it last.
    // No-op on Metal.
    template<typename T>
    void advise_read_mostly(const GpuPointer<T> &wrapper_pointer, usize size_in_bytes) {
    #ifdef SPIKECOREC_CUDA
        cudaMemAdvise(wrapper_pointer.pointer, size_in_bytes, cudaMemAdviseSetReadMostly, 0);
    #elif defined(SPIKECOREC_METAL)
        (void)size_in_bytes;
        (void)wrapper_pointer;
    #endif
    }

    // --- synchronization ---
    // wait for all in-flight GPU work to complete
    void synchronize_gpu_work();

    // --- kernel lifecycle ---
    // MTLComputePipelineState* on Metal, CUfunction+CUmodule on CUDA. A complete type (not left
    // opaque) so callers besides backend.cpp can hold one by value and cache it across calls —
    // ticket #6's master-kernel assembly compiles each assembled kernel once and reuses the
    // resulting KernelHandle every tick instead of recompiling. Any .cpp file that declares a
    // KernelHandle variable must #include <cuda.h> / <Metal/Metal.hpp> itself first if it does so
    // before this header is included elsewhere in the same translation unit (matches GpuPointer<T>
    // above, which already relies on the same convention for MTL::Buffer).
    struct KernelHandle {
    #ifdef SPIKECOREC_CUDA
        CUfunction cuda_kernel_function{};
        CUmodule   cuda_module{};
    #elif defined(SPIKECOREC_METAL)
        MTL::ComputePipelineState *pipeline_state = nullptr;
    #endif
    };

    KernelHandle compile_kernel(const char *source, const char *function_name);

    void release_kernel(KernelHandle handle);


    // --- command batching ---
    // Batches multiple kernel dispatches into a single command buffer so the caller
    // pays one commit+waitUntilCompleted round trip instead of one per kernel (Metal).
    // No-op on CUDA, where kernel launches are already enqueued async on the default
    // stream — begin/commit are safe to call unconditionally from backend-agnostic code.
    struct MetalCommandBatch;

    MetalCommandBatch *begin_command_batch();

    // Commits the batch's command buffer and blocks until the GPU has finished
    // executing every kernel encoded into it since begin_command_batch().
    void commit_command_batch(MetalCommandBatch *batch);


    // --- dispatch ---
    struct LaunchConfig {
        u32 grid_size;
        u32 block_size;
    };

    // - takes raw pointers, backend resolves to MTLBuffers
    //   internally on Metal
    // - if batch is non-null, encodes into the batch's already-open command buffer
    //   instead of creating/committing/waiting on one of its own (Metal); the caller
    //   is responsible for calling commit_command_batch() once all encodes are queued
    void metal_dispatch(
        KernelHandle handle,
        LaunchConfig config,
        const void *const *args,
        const usize *arg_sizes,
        u32 arg_count,
        MetalCommandBatch *batch = nullptr
    );

    // CUDA analog of metal_dispatch — generic launch of an arbitrary runtime-compiled
    // (NVRTC/compile_kernel) CUfunction, wrapping cuLaunchKernel.
    // - args[i] already points to the argument's storage in exactly the layout
    //   cuLaunchKernel's kernelParams expects, so (unlike Metal) no buffer-object
    //   resolution is needed: a pointer-typed argument's storage already holds a
    //   raw device pointer (see GpuPointer's CUDA branch)
    // - arg_sizes/arg_count are accepted only for signature parity with
    //   metal_dispatch; cuLaunchKernel needs neither (the parameter count/layout
    //   is implicit in the compiled kernel itself)
    // - batch is Metal-specific command-buffer batching; CUDA has no analogous
    //   batching concept in this codebase, so it is accepted only for signature
    //   parity and ignored — every launch is synchronous, matching the
    //   batch == nullptr path of metal_dispatch
    void cuda_dispatch(
        KernelHandle handle,
        LaunchConfig config,
        const void *const *args,
        const usize *arg_sizes,
        u32 arg_count,
        MetalCommandBatch *batch = nullptr
    );

    // --- atomic ops (for backends that need explicit support) ---
    //void atomic_add_f32(f32 *address, f32 value);

    // --- kernel wrappers ---
    // Declared here, implemented per-backend in src/cuda/ and src/metal/.
    // Callers pass raw pointers extracted from GpuPointer::get_contents().

    // Parallel dot products: writes node_count * max_neighbor_count values into output_weights,
    // row-major by source node. Adjacency is resolved via the bit-packed k^2-tree — each thread
    // walks its source node's row in the tree to discover up to max_neighbor_count targets;
    // slots beyond a node's actual neighbor count are sentinel-padded (target -1 -> weight 0).
    // `coefficients` is the shared-basis coefficient vector Ck (rank_float4_stride*4 scalar f32
    // elements — ticket #52/D2): reconstruction is Σ U[i,r]·coefficients[r]·V[j,r], with the
    // WeightMatrix::DEFAULT_MATRIX_INDEX case's all-ones coefficients reducing this to the
    // original dot(U,V).
    void gpu_neighbor_weights(
        const float4 *U,
        const float4 *V,
        const u32    *internal_node_words,
        const u32    *leaf_node_words,
        const u32    *rank_superblock_table,
        const u16    *rank_subblock_table,
        s32           branching_factor,
        s32           superblock_size_words,
        s32           padded_node_count,
        s32           tree_height,
        s32           internal_bit_count,
        s64           node_count,
        s64           max_neighbor_count,
        s64           rank_float4_stride,
        const f32    *coefficients,
        f32          *output_weights
    );

    // In-place element-wise scale of every float4 in U and V by scale_factor.
    void gpu_scale_uv(
        float4 *U,
        float4 *V,
        s64     total_float4_element_count,
        f32     scale_factor
    );

    // Rank-1 Hebbian update of U[source_node] and V[target_node].
    void gpu_weight_update(
        float4 *U,
        float4 *V,
        s64     rank_float4_stride,
        s32     source_node,
        s32     target_node,
        f32     delta,
        f32     learning_rate,
        f32     l2_regularization,
        s32     iterations
    );

    // Apply exponential membrane decay to all neurons from last_tick_updated up to tick.
    // If batch is non-null, encodes into it instead of dispatching standalone (see dispatch()).
    void gpu_decay_all_neurons(
        f32 *membrane_potentials,
        s64 *last_tick_updated,
        s64  neuron_count,
        s64  tick,
        f32  resting_mp,
        f32  decay_rate,
        MetalCommandBatch *batch = nullptr
    );

    // If batch is non-null, encodes into it instead of dispatching standalone (see dispatch()).
    void gpu_add_network_input(
        f32 *membrane_potentials, 
        s32 *input_neuron_indices, 
        const f32 *input_values, 
        s64 element_count, 
        MetalCommandBatch *batch = nullptr
    );

    // Merge override_input_neurons into the current active neuron set for this tick.
    // If batch is non-null, encodes into it instead of dispatching standalone (see dispatch()).
    void gpu_merge_input_neurons(
        s32       *active_neuron_indices,
        s32       *active_neuron_count,
        const s64 *override_input_neurons,
        s64        override_count,
        MetalCommandBatch *batch = nullptr
    );

    // Run one simulation tick: propagate spikes and update membrane potentials.
    // Adjacency is looked up via the bit-packed k^2-tree (replaces the flat
    // neighbor_indices adjacency array) — the kernel walks each spiking neuron's
    // row of the tree to discover its downstream targets.
    // If batch is non-null, encodes into it instead of dispatching standalone (see dispatch()).
    void gpu_step(
        s64           tick,
        s64           next_tick,
        s32           spike_period,
        f32           spike_threshold,
        f32           learning_rate,
        f32           learning_rate_plus,
        f32           decay_rate,
        f32           resting_mp,
        const float4 *U,
        const float4 *V,
        s64           rank_float4_stride,
        f32           constant_weight,
        const u32    *internal_node_words,
        const u32    *leaf_node_words,
        const u32    *rank_superblock_table,
        const u16    *rank_subblock_table,
        s32           branching_factor,
        s32           superblock_size_words,
        s32           padded_node_count,
        s32           tree_height,
        s32           internal_bit_count,
        s64           neuron_count,
        f32          *network_inputs,
        f32          *membrane_potentials,
        s64          *last_spiked,
        s64          *last_tick_updated,
        const s32    *active_neuron_indices,
        const s32    *active_neuron_count,
        s32          *next_active_neuron_indices,
        s32          *next_active_neuron_count,
        s32          *active_generation,
        bool          active_set_optimization_enabled,
        s32           thread_count_per_block,
        s32           block_count,
        MetalCommandBatch *batch = nullptr
    );

    // Build the reservoir feature vector: spike traces, normalised membrane voltages, bias.
    // output_buffer must be pre-allocated to (2 * neuron_count + 1) f32 elements.
    void gpu_reservoir_features(
        s64       neuron_count,
        s64       tick,
        f32       spike_tau,
        f32       voltage_scale,
        f32      *membrane_potentials,
        const s64*last_spiked,
        s64      *last_tick_updated,
        f32       resting_mp,
        f32       decay_rate,
        f32      *output_buffer
    );

    // Batched k2tree adjacency query. Compiles and caches its kernel internally on first use.
    // Writes 0 or 1 into output_buffer[i] for each (source_indices[i], target_indices[i]).
    void gpu_k2tree_adjacent_batch(
        const u32    *internal_node_words,
        const u32    *leaf_node_words,
        const u32    *rank_superblock_table,
        const u16    *rank_subblock_table,
        s32           branching_factor,
        s32           superblock_size_words,
        s32           node_count,
        s32           padded_node_count,
        s32           tree_height,
        s32           internal_bit_count,
        const s32    *source_indices,
        const s32    *target_indices,
        uint8_t      *output_buffer,
        s32           query_count
    );

    // Batched k2tree row enumeration. For each node in source_node_indices, walks its
    // row in the tree and writes up to max_neighbor_count neighbor indices into
    // output_buffer[query_index * max_neighbor_count .. +max_neighbor_count), in
    // tree-traversal order; slots beyond a node's actual neighbor count are
    // sentinel-padded (-1). output_buffer must be pre-sized to
    // query_count * max_neighbor_count elements.
    void gpu_k2tree_get_neighbors_batch(
        const u32    *internal_node_words,
        const u32    *leaf_node_words,
        const u32    *rank_superblock_table,
        const u16    *rank_subblock_table,
        s32           branching_factor,
        s32           superblock_size_words,
        s32           node_count,
        s32           padded_node_count,
        s32           tree_height,
        s32           internal_bit_count,
        const s32    *source_node_indices,
        s32           query_count,
        s32           max_neighbor_count,
        s32          *output_buffer
    );
} // namespace spikecorec
