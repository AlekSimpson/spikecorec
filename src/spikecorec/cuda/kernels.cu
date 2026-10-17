// CUDA versions of the live kernels in metal/kernels.metal, with the same arguments, in the same
// order. An argument Metal reads from the constant address space is a pointer (see
// precompiled_kernels.h).

#include <cstring>

#include "k2tree_device.cuh"
#include "precompiled_kernels.h"

// ── neighbor_weights ──────────────────────────────────────────────────────────
// One plane's basis value on every edge, one thread per edge ordinal. `coefficients` is the
// plane's Ck row, so the value is the sum over lanes of U[source] * Ck * V[target].
extern "C" __global__ void neighbor_weights_kernel(
    const float4 *basis_u,
    const float4 *basis_v,
    const unsigned int *internal_node_words,
    const unsigned int *leaf_node_words,
    const unsigned int *rank_superblock_table,
    const unsigned short *rank_subblock_table,
    const int *branching_factor,
    const int *superblock_size_words,
    const int *padded_node_count,
    const int *tree_height,
    const int *internal_bit_count,
    const long long *node_count,
    const long long *total_edge_count,
    const long long *rank_float4_stride,
    const float *coefficients,
    const long long *edge_row_offset,
    float *output_weights
) {
    const long long edge_ordinal = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (edge_ordinal >= *total_edge_count) {
        return;
    }
    const long long stride = *rank_float4_stride;

    // The source is the last row whose first edge ordinal is <= edge_ordinal.
    long long low = 0;
    long long high = *node_count;
    while (low < high) {
        const long long middle = (low + high + 1) / 2;
        if (edge_row_offset[middle] <= edge_ordinal) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    const long long source_node = low;
    const long long neighbor_slot = edge_ordinal - edge_row_offset[source_node];

    const unsigned int target_node = k2t_find_nth_neighbor(
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        *branching_factor, *superblock_size_words, static_cast<int>(*node_count), *padded_node_count,
        *tree_height, *internal_bit_count, static_cast<unsigned int>(source_node), static_cast<int>(neighbor_slot));
    if (target_node == K2T_NO_NEIGHBOR) {
        output_weights[edge_ordinal] = 0.0f;
        return;
    }

    const float4 *source_row = basis_u + source_node * stride;
    const float4 *target_row = basis_v + static_cast<long long>(target_node) * stride;

    // Same terms, in the same order, as Weight_Matrix::reconstruct_entry on the host.
    float dot_product = 0.0f;
    for (long long lane = 0; lane < stride; lane += 1) {
        const float4 source_lane = source_row[lane];
        const float4 target_lane = target_row[lane];
        dot_product += source_lane.x * (coefficients[lane * 4 + 0] * target_lane.x) +
                       source_lane.y * (coefficients[lane * 4 + 1] * target_lane.y) +
                       source_lane.z * (coefficients[lane * 4 + 2] * target_lane.z) +
                       source_lane.w * (coefficients[lane * 4 + 3] * target_lane.w);
    }
    output_weights[edge_ordinal] = dot_product;
}

namespace spikecorec::cuda {

const void *kernels_symbol(const char *name) {
    if (std::strcmp(name, "neighbor_weights_kernel") == 0) {
        return reinterpret_cast<const void *>(&neighbor_weights_kernel);
    }
    return nullptr;
}

} // namespace spikecorec::cuda
