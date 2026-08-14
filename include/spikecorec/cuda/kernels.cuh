#pragma once

#include "spikecorec/core/types.h"
#include <cuda_runtime.h>

namespace spikecorec::cuda {

// Launch parameters
struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem = 0;
    cudaStream_t stream = nullptr;
};

LaunchConfig default_launch_config(usize n, usize threads_per_block = 256);

// Native, ahead-of-time compiled launchers backing the WeightMatrix GPU kernels.
// All operate on raw device pointers extracted via GpuPointer::get_contents().

// Writes node_count * max_neighbor_count dot products U[source]·V[neighbor] into output_weights,
// row-major by source node. Adjacency is resolved via the bit-packed k^2-tree — each thread
// walks its source node's row in the tree to discover up to max_neighbor_count targets;
// slots beyond a node's actual neighbor count are sentinel-padded (target -1 -> weight 0).
// `coefficients` is the shared-basis Ck vector (rank_float4_stride*4 scalar f32 elements —
// ticket #52/D2): reconstruction is Σ U[i,r]·coefficients[r]·V[j,r]. `coefficients_present`
// selects whether it is read at all; 0 runs the pre-shared-basis dot(U,V) path verbatim.
void launch_neighbor_weights(
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
    s32           coefficients_present,
    f32          *output_weights,
    cudaStream_t  stream = nullptr
);

// In-place element-wise scale of every float4 lane in U and V by scale_factor.
void launch_scale_uv(
    float4      *U,
    float4      *V,
    s64          total_float4_element_count,
    f32          scale_factor,
    cudaStream_t stream = nullptr
);

// Rank-1 Hebbian update of U[source_node] and V[target_node], repeated `iterations` times,
// regularized toward each row's pre-update ("anchor") value.
void launch_weight_update(
    float4      *U,
    float4      *V,
    s64          rank_float4_stride,
    s32          source_node,
    s32          target_node,
    f32          delta,
    f32          learning_rate,
    f32          l2_regularization,
    s32          iterations,
    cudaStream_t stream = nullptr
);

// Run one simulation tick: propagate spikes and update membrane potentials.
// Uncompiled mirror of the Metal `step` kernel — pending verification on CUDA
// hardware (this development machine has no CUDA toolchain).
void launch_step(
    s64           tick,
    s64           next_tick,
    s32           spike_period,
    f32           spike_threshold,
    f32           learning_rate,
    f32           decay_rate,
    f32           resting_mp,
    float4       *U,
    float4       *V,
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
    s32           block_count,
    cudaStream_t  stream = nullptr
);

// Atomically accumulate input_values[i] into membrane_potentials[input_neuron_indices[i]].
void launch_add_network_input(
    f32          *membrane_potentials,
    const s32    *input_neuron_indices,
    const f32    *input_values,
    s64           element_count,
    cudaStream_t  stream = nullptr
);

// Apply exponential membrane decay to all neurons from last_tick_updated up to tick.
void launch_decay_all_neurons(
    f32          *membrane_potentials,
    s64          *last_tick_updated,
    s64           neuron_count,
    s64           tick,
    f32           resting_mp,
    f32           decay_rate,
    cudaStream_t  stream = nullptr
);

// Append override_input_neurons into the current active set, deduping by linear
// scan against active_neuron_indices[0..active_neuron_count[0]).
void launch_merge_input_neurons(
    s32          *active_neuron_indices,
    s32          *active_neuron_count,
    const s64    *override_input_neurons,
    s64           override_count,
    cudaStream_t  stream = nullptr
);

// Build the reservoir feature vector: spike traces, normalised membrane voltages, bias.
// output_buffer must be pre-allocated to (2 * neuron_count + 1) f32 elements.
void launch_reservoir_features(
    s64           neuron_count,
    s64           tick,
    f32           spike_tau,
    f32           voltage_scale,
    f32          *membrane_potentials,
    const s64    *last_spiked,
    s64          *last_tick_updated,
    f32           resting_mp,
    f32           decay_rate,
    f32          *output_buffer,
    cudaStream_t  stream = nullptr
);

// Batched k^2-tree edge-existence queries: writes 0/1 into output_buffer[i] for
// each (source_indices[i], target_indices[i]) pair, mirroring K2Tree::adjacent.
// Tree shape is supplied at runtime, so this single kernel serves every K2Tree
// instance regardless of branching factor or height.
void launch_k2tree_adjacent_batch(
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
    s32           query_count,
    cudaStream_t  stream = nullptr
);

// Batched k2tree row enumeration. For each node in source_node_indices, walks its
// row in the tree and writes up to max_neighbor_count neighbor indices into
// output_buffer[query_index * max_neighbor_count .. +max_neighbor_count), in
// tree-traversal order; slots beyond a node's actual neighbor count are
// sentinel-padded (-1). output_buffer must be pre-sized to
// query_count * max_neighbor_count elements.
void launch_k2tree_get_neighbors_batch(
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
    s32          *output_buffer,
    cudaStream_t  stream = nullptr
);

} // namespace spikecorec::cuda
