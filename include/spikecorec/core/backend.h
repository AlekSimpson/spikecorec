#pragma once

#include <cassert>
#include "spikecorec/core/types.h"

namespace spikecorec {
    // Called once at program startup to initialize the GPU context.
    void initialize_gpu_context();

    // Called once at program shutdown to release GPU resources.
    void release_gpu_resources();

    template<typename T>
    struct GpuPointer {
    #ifdef SPIKECOREC_CUDA
        T *pointer = nullptr;

        T*       get_contents()       { return pointer; }
        const T* get_contents() const { return pointer; }
    #elif defined(SPIKECOREC_METAL)
        MTL::Buffer *buffer = nullptr;

        T*       get_contents()       { return static_cast<T*>(buffer->contents()); }
        const T* get_contents() const { return static_cast<const T*>(buffer->contents()); }
    #endif

        GpuPointer() = default;

        GpuPointer(const GpuPointer &) = delete;

        GpuPointer &operator=(const GpuPointer &) = delete;

        GpuPointer(GpuPointer &&other) noexcept {
        #ifdef SPIKECOREC_CUDA
            pointer = other.pointer;
            other.pointer = nullptr;
        #elif defined(SPIKECOREC_METAL)
            buffer = other.buffer;
            other.buffer = nullptr;
        #endif
        }

        GpuPointer &operator=(GpuPointer &&other) noexcept {
            if (this != &other) {
            #ifdef SPIKECOREC_CUDA
                assert(pointer == nullptr &&
                       "GpuPointer move assignment would leak an existing CUDA allocation — "
                       "call backend::deallocate() before reassigning");
                pointer = other.pointer;
                other.pointer = nullptr;
            #elif defined(SPIKECOREC_METAL)
                assert(buffer == nullptr &&
                    "GpuPointer move assignment would leak an existing MTL::Buffer — "
                    "call backend::deallocate() before reassigning");
                buffer = other.buffer;
                other.buffer = nullptr;
            #endif
            }
            return *this;
        }
    };

    // --- buffer ---
    // non-template bridges — defined in backend.cpp, hide platform types from callers
    void* allocate_bytes(usize byte_size);
    void  deallocate_bytes(void* platform_handle);

    template<typename T>
    GpuPointer<T> allocate(usize byte_size) {
        GpuPointer<T> ptr;
    #ifdef SPIKECOREC_CUDA
        ptr.pointer = static_cast<T*>(allocate_bytes(byte_size));
    #elif defined(SPIKECOREC_METAL)
        ptr.buffer  = static_cast<MTL::Buffer*>(allocate_bytes(byte_size));
    #endif
        return ptr;
    }

    template<typename T>
    void deallocate(GpuPointer<T> ptr) {
    #ifdef SPIKECOREC_CUDA
        deallocate_bytes(ptr.pointer);
    #elif defined(SPIKECOREC_METAL)
        deallocate_bytes(ptr.buffer);
    #endif
    }

    // CUDA hint, no-op on Metal
    template<typename T>
    void prefetch_to_gpu(const GpuPointer<T>& ptr, usize size_in_bytes) {
    #ifdef SPIKECOREC_CUDA
        cudaMemPrefetchAsync(ptr.pointer, size_in_bytes, 0);
    #endif
    }

    // --- synchronization ---
    // wait for all in-flight GPU work to complete
    void synchronize_gpu_work();

    // --- kernel lifecycle ---
    // opaque — MTLComputePipelineState* on Metal,
    // CUfunction on CUDA
    struct KernelHandle;

    KernelHandle compile_kernel(
        const char *source, const char *function_name);

    void release_kernel(KernelHandle handle);


    // --- dispatch ---
    struct LaunchConfig {
        u32 grid_size;
        u32 block_size;
    };

    // generic dispatch:
    // - takes raw pointers, backend resolves to MTLBuffers
    //   internally on Metal
    void dispatch(
        KernelHandle handle,
        LaunchConfig config,
        const void *const *args,
        const usize *arg_sizes,
        u32 arg_count
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
    void gpu_decay_all_neurons(
        f32 *membrane_potentials,
        s64 *last_tick_updated,
        s64  neuron_count,
        s64  tick,
        f32  resting_mp,
        f32  decay_rate
    );

    // Element-wise addition of two equal-length vectors: result[i] = a[i] + b[i].
    // result may alias a or b for in-place accumulation.
    // void gpu_vector_add(
    //     f32       *result,
    //     const f32 *a,
    //     const f32 *b,
    //     s64        element_count
    // );
    void gpu_add_network_input(f32 *membrane_potentials, s32 *input_neuron_indices, const f32 *input_values, s64 element_count);

    // Merge override_input_neurons into the current active neuron set for this tick.
    void gpu_merge_input_neurons(
        s32       *active_neuron_indices,
        s32       *active_neuron_count,
        const s64 *override_input_neurons,
        s64        override_count
    );

    // Run one simulation tick: propagate spikes and update membrane potentials.
    // Adjacency is looked up via the bit-packed k^2-tree (replaces the flat
    // neighbor_indices adjacency array) — the kernel walks each spiking neuron's
    // row of the tree to discover its downstream targets.
    void gpu_step(
        s64           tick,
        s64           next_tick,
        s32           spike_period,
        f32           spike_threshold,
        f32           learning_rate,
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
        s32           thread_count_per_block,
        s32           block_count
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
