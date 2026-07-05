#include "spikecorec/cuda/kernels.cuh"

namespace spikecorec::cuda {

LaunchConfig default_launch_config(usize n, usize threads_per_block) {
    LaunchConfig cfg;
    cfg.block = dim3(static_cast<unsigned>(threads_per_block));
    cfg.grid  = dim3(static_cast<unsigned>((n + threads_per_block - 1) / threads_per_block));
    return cfg;
}

namespace {

// ── k2tree bit-walk helpers ───────────────────────────────────────────────────
// Shared by neighbor_weights_kernel and k2tree_adjacent_batch_kernel.
__device__ __forceinline__ u32 k2t_get_bit(const u32 *words, s32 bit_index) {
    return (words[bit_index >> 5] >> (bit_index & 31)) & 1u;
}

__device__ __forceinline__ s32 k2t_rank1_exclusive(
    const u32 *internal_words,
    const u32 *superblock_table,
    const u16 *subblock_table,
    s32 position,
    s32 superblock_size
) {
    s32 word_index = position >> 5;
    s32 bit_offset = position & 31;
    s32 superblock_index = word_index / superblock_size;
    u32 superblock_base = superblock_table[superblock_index];
    u32 subblock_base = subblock_table[word_index];
    u32 partial_word_mask = (bit_offset == 0) ? 0u : ((1u << bit_offset) - 1u);
    u32 partial_word_popcount = static_cast<u32>(__popc(internal_words[word_index] & partial_word_mask));
    return static_cast<s32>(superblock_base + subblock_base + partial_word_popcount);
}

#define MAX_K2TREE_HEIGHT 32
#define MAX_RANK_FLOAT4_STRIDE 64

// Finds the `target_slot`-th neighbor (0-indexed, in tree-traversal order) of row
// `u`, walking only the subtrees that intersect u's row and have at least one bit
// set (skipping empty regions entirely — the whole point of the k^2-tree). Returns
// the neighbor's node index, or -1 if `u` has fewer than `target_slot + 1` neighbors.
// Iterative DFS with an explicit per-level stack — device code can't recurse
// arbitrarily, and depth is bounded by tree_height (<= MAX_K2TREE_HEIGHT for any
// branching_factor >= 2 and realistic node counts).
__device__ __forceinline__ s32 k2t_find_nth_neighbor(
    const u32 *internal_node_words,
    const u32 *leaf_node_words,
    const u32 *rank_superblock_table,
    const u16 *rank_subblock_table,
    s32 branching_factor,
    s32 superblock_size_words,
    s32 node_count,
    s32 padded_node_count,
    s32 tree_height,
    s32 internal_bit_count,
    s32 u,
    s32 target_slot
) {
    if (tree_height == 0 || u < 0 || u >= node_count || target_slot < 0) return -1;

    s32 branching_factor_squared = branching_factor * branching_factor;

    s32 stack_row_base[MAX_K2TREE_HEIGHT];
    s32 stack_col_base[MAX_K2TREE_HEIGHT];
    s32 stack_block_size[MAX_K2TREE_HEIGHT];
    s32 stack_bit_offset[MAX_K2TREE_HEIGHT];
    s32 stack_next_col[MAX_K2TREE_HEIGHT];

    stack_row_base[0] = 0;
    stack_col_base[0] = 0;
    stack_block_size[0] = padded_node_count;
    stack_bit_offset[0] = 0;
    stack_next_col[0] = 0;

    s32 stack_top = 0;
    s32 neighbors_seen = 0;

    while (stack_top >= 0) {
        s32 level = stack_top;
        s32 col_offset = stack_next_col[level];
        if (col_offset >= branching_factor) {
            stack_top--;
            continue;
        }
        stack_next_col[level] = col_offset + 1;

        s32 row_base = stack_row_base[level];
        s32 col_base = stack_col_base[level];
        s32 block_size = stack_block_size[level];
        s32 level_bit_offset = stack_bit_offset[level];

        s32 child_block_size = block_size / branching_factor;
        s32 row_offset = (u - row_base) / child_block_size;
        s32 child_flat_index = row_offset * branching_factor + col_offset;
        s32 bit_position = level_bit_offset + child_flat_index;

        if (level == tree_height - 1) {
            if (k2t_get_bit(leaf_node_words, bit_position)) {
                s32 v = col_base + col_offset;
                if (v < node_count) {
                    if (neighbors_seen == target_slot) return v;
                    neighbors_seen++;
                }
            }
        } else if (k2t_get_bit(internal_node_words, bit_position)) {
            s32 rank_inclusive = k2t_rank1_exclusive(internal_node_words, rank_superblock_table,
                                                      rank_subblock_table, bit_position, superblock_size_words) + 1;
            s32 child_level = stack_top + 1;
            s32 raw_offset = branching_factor_squared * rank_inclusive;
            stack_row_base[child_level] = row_base + row_offset * child_block_size;
            stack_col_base[child_level] = col_base + col_offset * child_block_size;
            stack_block_size[child_level] = child_block_size;
            stack_bit_offset[child_level] = (child_level == tree_height - 1)
                ? (raw_offset - internal_bit_count)
                : raw_offset;
            stack_next_col[child_level] = 0;
            stack_top = child_level;
        }
    }

    return -1;
}

// Resumes a DFS over a row's neighbors from the caller's stack state, returning
// the next neighbor index each call, or -1 when exhausted. Stack arrays must be
// allocated by the caller (size MAX_K2TREE_HEIGHT each) and initialized once:
//   stack_row_base[0]=0, stack_col_base[0]=0, stack_block_size[0]=padded_node_count,
//   stack_bit_offset[0]=0, stack_next_col[0]=0, stack_top=0 (or -1 to short-circuit).
__device__ __forceinline__ s32 k2t_next_neighbor(
    const u32 *internal_node_words,
    const u32 *leaf_node_words,
    const u32 *rank_superblock_table,
    const u16 *rank_subblock_table,
    s32 branching_factor,
    s32 superblock_size_words,
    s32 node_count,
    s32 tree_height,
    s32 internal_bit_count,
    s32 u,
    s32 *stack_row_base,
    s32 *stack_col_base,
    s32 *stack_block_size,
    s32 *stack_bit_offset,
    s32 *stack_next_col,
    s32 &stack_top
) {
    s32 branching_factor_squared = branching_factor * branching_factor;

    while (stack_top >= 0) {
        s32 level      = stack_top;
        s32 col_offset = stack_next_col[level];
        if (col_offset >= branching_factor) {
            stack_top--;
            continue;
        }
        stack_next_col[level] = col_offset + 1;

        s32 row_base         = stack_row_base[level];
        s32 col_base         = stack_col_base[level];
        s32 block_size       = stack_block_size[level];
        s32 level_bit_offset = stack_bit_offset[level];

        s32 child_block_size = block_size / branching_factor;
        s32 row_offset       = (u - row_base) / child_block_size;
        s32 child_flat_index = row_offset * branching_factor + col_offset;
        s32 bit_position     = level_bit_offset + child_flat_index;

        if (level == tree_height - 1) {
            if (k2t_get_bit(leaf_node_words, bit_position)) {
                s32 v = col_base + col_offset;
                if (v < node_count) return v;
            }
        } else if (k2t_get_bit(internal_node_words, bit_position)) {
            s32 rank_inclusive = k2t_rank1_exclusive(internal_node_words, rank_superblock_table,
                                                      rank_subblock_table, bit_position, superblock_size_words) + 1;
            s32 child_level = stack_top + 1;
            s32 raw_offset  = branching_factor_squared * rank_inclusive;
            stack_row_base[child_level]   = row_base + row_offset * child_block_size;
            stack_col_base[child_level]   = col_base + col_offset * child_block_size;
            stack_block_size[child_level] = child_block_size;
            stack_bit_offset[child_level] = (child_level == tree_height - 1)
                ? (raw_offset - internal_bit_count)
                : raw_offset;
            stack_next_col[child_level] = 0;
            stack_top = child_level;
        }
    }
    return -1;
}

// ── neighbor_weights ──────────────────────────────────────────────────────────
// One thread per (source_node, neighbor_slot) pair; each thread walks its source
// node's row in the k^2-tree (descending only into populated subtrees) to discover
// up to max_neighbor_count targets, then dot-products U[source]·V[target]. Slots
// beyond a node's actual neighbor count are sentinel-padded (target -1 -> weight 0).
// TODO: threads sharing a source_node each independently re-walk that row from the
// root for their one slot, redoing overlapping work. Restructuring to one thread
// (or one block) per source_node — walking the row once and filling all
// max_neighbor_count slots — would eliminate that redundancy.
__global__ void neighbor_weights_kernel(
    const float4 *__restrict__ U,
    const float4 *__restrict__ V,
    const u32    *__restrict__ internal_node_words,
    const u32    *__restrict__ leaf_node_words,
    const u32    *__restrict__ rank_superblock_table,
    const u16    *__restrict__ rank_subblock_table,
    s32 branching_factor,
    s32 superblock_size_words,
    s32 padded_node_count,
    s32 tree_height,
    s32 internal_bit_count,
    s64 node_count,
    s64 max_neighbor_count,
    s64 rank_float4_stride,
    f32 *__restrict__ output_weights
) {
    s64 pair_index = static_cast<s64>(blockIdx.x) * blockDim.x + threadIdx.x;
    s64 total_pairs = node_count * max_neighbor_count;
    if (pair_index >= total_pairs) return;

    s64 source_node = pair_index / max_neighbor_count;
    s64 neighbor_slot = pair_index % max_neighbor_count;
    const float4 *u_row = U + source_node * rank_float4_stride;

    s32 target_node = k2t_find_nth_neighbor(
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, static_cast<s32>(node_count), padded_node_count,
        tree_height, internal_bit_count, static_cast<s32>(source_node), static_cast<s32>(neighbor_slot)
    );
    if (target_node < 0) {
        output_weights[pair_index] = 0.0f;
        return;
    }
    const float4 *v_row = V + static_cast<s64>(target_node) * rank_float4_stride;

    f32 dot_product = 0.0f;
    #pragma unroll 4
    for (s64 lane = 0; lane < rank_float4_stride; ++lane) {
        float4 u4 = u_row[lane];
        float4 v4 = v_row[lane];
        dot_product += u4.x * v4.x + u4.y * v4.y + u4.z * v4.z + u4.w * v4.w;
    }
    output_weights[pair_index] = dot_product;
}

// ── scale_uv ──────────────────────────────────────────────────────────────────
// Pure memory-bound element-wise scale; one thread per float4 lane, fully coalesced.
__global__ void scale_uv_kernel(
    float4 *__restrict__ U,
    float4 *__restrict__ V,
    s64 total_float4_element_count,
    f32 scale_factor
) {
    s64 index = static_cast<s64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total_float4_element_count) return;

    float4 u = U[index];
    u.x *= scale_factor; u.y *= scale_factor; u.z *= scale_factor; u.w *= scale_factor;
    U[index] = u;

    float4 v = V[index];
    v.x *= scale_factor; v.y *= scale_factor; v.z *= scale_factor; v.w *= scale_factor;
    V[index] = v;
}

// ── vector_add ────────────────────────────────────────────────────────────────
// Pure memory-bound element-wise addition; one thread per element, fully coalesced.
// `result` may alias `a` or `b` for in-place accumulation, so the pointers are
// deliberately NOT marked __restrict__ (that would assert no-aliasing to the
// compiler and could miscompile the in-place case).
__global__ void vector_add_kernel(
    f32 *result,
    const f32 *a,
    const f32 *b,
    s64 element_count
) {
    s64 index = static_cast<s64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= element_count) return;
    result[index] = a[index] + b[index];
}

// ── weight_update ─────────────────────────────────────────────────────────────
// Single (source, target) pair, run for `iterations` rounds of proximal Hebbian
// update. Launched as one block of rank_float4_stride threads (capped at 1024;
// rank is always small — e.g. 16 float4 lanes for rank=64). Anchors (the rows'
// values before the first iteration) are cached in shared memory; den_u/den_v
// (the squared-norm denominators) are recomputed each round via a warp-shuffle
// + shared-memory tree reduction since the rows change between rounds.
__global__ void weight_update_kernel(
    float4 *__restrict__ U,
    float4 *__restrict__ V,
    s64 rank_float4_stride,
    s32 source_node,
    s32 target_node,
    f32 delta,
    f32 learning_rate,
    f32 l2_regularization,
    s32 iterations
) {
    extern __shared__ float4 shared_storage[];
    float4 *anchor_u = shared_storage;                          // [rank_float4_stride]
    float4 *anchor_v = shared_storage + rank_float4_stride;     // [rank_float4_stride]

    constexpr int MAX_WARPS = 32; // supports up to 1024 threads per block
    __shared__ f32 warp_partial_u[MAX_WARPS];
    __shared__ f32 warp_partial_v[MAX_WARPS];
    __shared__ f32 den_u, den_v;

    int lane = threadIdx.x;
    bool active = lane < rank_float4_stride;

    float4 *u_row = U + static_cast<s64>(source_node) * rank_float4_stride;
    float4 *v_row = V + static_cast<s64>(target_node) * rank_float4_stride;

    float4 u4 = active ? u_row[lane] : float4{0.0f, 0.0f, 0.0f, 0.0f};
    float4 v4 = active ? v_row[lane] : float4{0.0f, 0.0f, 0.0f, 0.0f};

    if (active) {
        anchor_u[lane] = u4;
        anchor_v[lane] = v4;
    }
    __syncthreads();

    int warp_id = lane >> 5;
    int lane_in_warp = lane & 31;
    int warp_count = (blockDim.x + 31) >> 5;

    for (s32 iteration = 0; iteration < iterations; ++iteration) {
        f32 local_u_sq = u4.x * u4.x + u4.y * u4.y + u4.z * u4.z + u4.w * u4.w;
        f32 local_v_sq = v4.x * v4.x + v4.y * v4.y + v4.z * v4.z + v4.w * v4.w;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_u_sq += __shfl_down_sync(0xffffffffu, local_u_sq, offset);
            local_v_sq += __shfl_down_sync(0xffffffffu, local_v_sq, offset);
        }
        if (lane_in_warp == 0) {
            warp_partial_u[warp_id] = local_u_sq;
            warp_partial_v[warp_id] = local_v_sq;
        }
        __syncthreads();

        if (lane == 0) {
            f32 sum_u = 0.0f, sum_v = 0.0f;
            for (int warp = 0; warp < warp_count; ++warp) {
                sum_u += warp_partial_u[warp];
                sum_v += warp_partial_v[warp];
            }
            den_u = l2_regularization + sum_u;
            den_v = l2_regularization + sum_v;
        }
        __syncthreads();

        if (active) {
            float4 au = anchor_u[lane];
            float4 av = anchor_v[lane];
            f32 inv_den_u = 1.0f / den_u;
            f32 inv_den_v = 1.0f / den_v;

            float4 du, dv;
            du.x = learning_rate * (delta * (v4.x * inv_den_v) - l2_regularization * (u4.x - au.x));
            du.y = learning_rate * (delta * (v4.y * inv_den_v) - l2_regularization * (u4.y - au.y));
            du.z = learning_rate * (delta * (v4.z * inv_den_v) - l2_regularization * (u4.z - au.z));
            du.w = learning_rate * (delta * (v4.w * inv_den_v) - l2_regularization * (u4.w - au.w));

            dv.x = learning_rate * (delta * (u4.x * inv_den_u) - l2_regularization * (v4.x - av.x));
            dv.y = learning_rate * (delta * (u4.y * inv_den_u) - l2_regularization * (v4.y - av.y));
            dv.z = learning_rate * (delta * (u4.z * inv_den_u) - l2_regularization * (v4.z - av.z));
            dv.w = learning_rate * (delta * (u4.w * inv_den_u) - l2_regularization * (v4.w - av.w));

            u4.x += du.x; u4.y += du.y; u4.z += du.z; u4.w += du.w;
            v4.x += dv.x; v4.y += dv.y; v4.z += dv.z; v4.w += dv.w;
        }
        __syncthreads();
    }

    if (active) {
        u_row[lane] = u4;
        v_row[lane] = v4;
    }
}

// ── k2tree_adjacent_batch ─────────────────────────────────────────────────────
// Batched k^2-tree edge-existence queries. One thread per (source, target) pair;
// each thread independently walks the bit-tree from the root, mirroring
// K2Tree::adjacent. Tree shape is supplied at runtime, so a single compiled
// kernel serves every K2Tree instance regardless of branching factor or height.
__global__ void k2tree_adjacent_batch_kernel(
    const u32 *__restrict__ internal_node_words,
    const u32 *__restrict__ leaf_node_words,
    const u32 *__restrict__ rank_superblock_table,
    const u16 *__restrict__ rank_subblock_table,
    s32 branching_factor,
    s32 superblock_size_words,
    s32 node_count,
    s32 padded_node_count,
    s32 tree_height,
    s32 internal_bit_count,
    const s32 *__restrict__ source_indices,
    const s32 *__restrict__ target_indices,
    uint8_t *__restrict__ output_buffer,
    s32 query_count
) {
    s32 query_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_index >= query_count) return;

    s32 u = source_indices[query_index];
    s32 v = target_indices[query_index];

    if (u < 0 || u >= node_count || v < 0 || v >= node_count || tree_height == 0) {
        output_buffer[query_index] = 0;
        return;
    }

    s32 branching_factor_squared = branching_factor * branching_factor;

    if (tree_height == 1) {
        s32 block_size = padded_node_count / branching_factor;
        s32 child_flat_index = (u / block_size) * branching_factor + (v / block_size);
        output_buffer[query_index] = static_cast<uint8_t>(k2t_get_bit(leaf_node_words, child_flat_index));
        return;
    }

    s32 level_bit_offset = 0;
    s32 current_block_size = padded_node_count;
    s32 rank_inclusive = 0;

    for (s32 level = 0; level < tree_height - 1; ++level) {
        s32 block_size = current_block_size / branching_factor;
        s32 row_offset = u / block_size;
        s32 column_offset = v / block_size;
        s32 child_bit_position = level_bit_offset + row_offset * branching_factor + column_offset;

        if (!k2t_get_bit(internal_node_words, child_bit_position)) {
            output_buffer[query_index] = 0;
            return;
        }

        rank_inclusive = k2t_rank1_exclusive(internal_node_words, rank_superblock_table, rank_subblock_table,
                                             child_bit_position, superblock_size_words) + 1;

        u = u % block_size;
        v = v % block_size;

        if (level == tree_height - 2) break;

        level_bit_offset = branching_factor_squared * rank_inclusive;
        current_block_size = block_size;
    }

    s32 leaf_bit_offset = branching_factor_squared * rank_inclusive - internal_bit_count;
    output_buffer[query_index] = static_cast<uint8_t>(
        k2t_get_bit(leaf_node_words, leaf_bit_offset + u * branching_factor + v));
}

// ── k2tree_get_neighbors_batch ────────────────────────────────────────────────
// Batched row enumeration: one thread per (query_index, neighbor_slot) pair;
// each thread independently walks source_node_indices[query_index]'s row in the
// k^2-tree to find its (neighbor_slot)-th neighbor (or -1 if it has fewer).
// Mirrors neighbor_weights_kernel's enumeration but surfaces the discovered
// indices directly instead of consuming them for a dot product.
__global__ void k2tree_get_neighbors_batch_kernel(
    const u32 *__restrict__ internal_node_words,
    const u32 *__restrict__ leaf_node_words,
    const u32 *__restrict__ rank_superblock_table,
    const u16 *__restrict__ rank_subblock_table,
    s32 branching_factor,
    s32 superblock_size_words,
    s32 node_count,
    s32 padded_node_count,
    s32 tree_height,
    s32 internal_bit_count,
    const s32 *__restrict__ source_node_indices,
    s32 query_count,
    s32 max_neighbor_count,
    s32 *__restrict__ output_buffer
) {
    s64 pair_index = static_cast<s64>(blockIdx.x) * blockDim.x + threadIdx.x;
    s64 total_pairs = static_cast<s64>(query_count) * max_neighbor_count;
    if (pair_index >= total_pairs) return;

    s32 query_index = static_cast<s32>(pair_index / max_neighbor_count);
    s32 neighbor_slot = static_cast<s32>(pair_index % max_neighbor_count);
    s32 source_node = source_node_indices[query_index];

    output_buffer[pair_index] = k2t_find_nth_neighbor(
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, node_count, padded_node_count,
        tree_height, internal_bit_count, source_node, neighbor_slot
    );
}

// ── add_network_input ─────────────────────────────────────────────────────────
// Mirrors add_network_input_kernel in src/metal/kernels.metal — one thread per
// input element, atomically accumulating into membrane_potentials. CUDA's native
// atomicAdd(float*, float) replaces Metal's atomic_float cast dance.
__global__ void add_network_input_kernel(
    f32       *membrane_potentials,
    const s32 *input_neuron_indices,
    const f32 *input_values,
    s64        element_count
) {
    s64 index = static_cast<s64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= element_count) return;
    atomicAdd(&membrane_potentials[input_neuron_indices[index]], input_values[index]);
}

// ── apply_decay ───────────────────────────────────────────────────────────────
// Mirrors src/metal/kernels.metal's apply_decay helper. Shared by
// decay_all_neurons_kernel, step_kernel, and reservoir_features_kernel.
__device__ __forceinline__ f32 apply_decay(f32 membrane_potential, f32 resting_potential, f32 decay_rate, int time_delta) {
    if (time_delta <= 0) return membrane_potential;
    if (time_delta == 1) return resting_potential + (membrane_potential - resting_potential) * (1.0f - decay_rate);
    f32 decay = powf(1.0f - decay_rate, (f32)time_delta);
    return resting_potential + (membrane_potential - resting_potential) * decay;
}

// ── decay_all_neurons ─────────────────────────────────────────────────────────
// Mirrors decay_kernel in spikecore/cuda_code/kernels.c — one thread per neuron.
__global__ void decay_all_neurons_kernel(
    f32 *membrane_potentials,
    s64 *last_tick_updated,
    s64  neuron_count,
    s64  tick,
    f32  resting_mp,
    f32  decay_rate
) {
    s64 neuron_index = static_cast<s64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (neuron_index >= neuron_count) return;

    s64 time_delta = tick - last_tick_updated[neuron_index];
    membrane_potentials[neuron_index] = apply_decay(
        membrane_potentials[neuron_index], resting_mp, decay_rate, (int)time_delta);
    last_tick_updated[neuron_index] = tick;
}

// ── merge_input_neurons ───────────────────────────────────────────────────────
// Appends override_input_neurons into the CURRENT active set with linear-scan
// dedup against active_neuron_indices[0..active_neuron_count[0]). Mirrors
// merge_input_neurons_kernel in src/metal/kernels.metal — see the design note
// in backend.h / the parity plan for why this does NOT use generation tags.
__global__ void merge_input_neurons_kernel(
    s32       *active_neuron_indices,
    s32       *active_neuron_count,
    const s64 *override_input_neurons,
    s64        override_count
) {
    s64 override_index = static_cast<s64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (override_index >= override_count) return;

    s64 candidate = override_input_neurons[override_index];
    if (candidate < 0) return;
    s32 neuron = (s32)candidate;

    s32 current_count = active_neuron_count[0];
    for (s32 scan_index = 0; scan_index < current_count; ++scan_index) {
        if (active_neuron_indices[scan_index] == neuron) return;
    }

    s32 position = atomicAdd(active_neuron_count, 1);
    active_neuron_indices[position] = neuron;
}

// ── reservoir_features ────────────────────────────────────────────────────────
// Mirrors reservoir_feature_kernel in spikecore/cuda_code/kernels.c — one thread
// per neuron. Decays membrane potential, writes a spike trace and a normalised
// voltage into output_buffer, and (thread 0 only) the trailing bias term.
// output_buffer must be pre-sized to (2 * neuron_count + 1) elements.
__global__ void reservoir_features_kernel(
    s64        neuron_count,
    s64        tick,
    f32        spike_tau,
    f32        voltage_scale,
    f32       *membrane_potentials,
    const s64 *last_spiked,
    s64       *last_tick_updated,
    f32        resting_mp,
    f32        decay_rate,
    f32       *output_buffer
) {
    s64 neuron_index = static_cast<s64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (neuron_index >= neuron_count) return;

    s64 time_delta = tick - last_tick_updated[neuron_index];
    f32 membrane_potential = apply_decay(
        membrane_potentials[neuron_index], resting_mp, decay_rate, (int)time_delta);
    membrane_potentials[neuron_index] = membrane_potential;
    last_tick_updated[neuron_index] = tick;

    s64 since_spike = tick - last_spiked[neuron_index];
    f32 trace = (last_spiked[neuron_index] > 0) ? expf(-(f32)since_spike / spike_tau) : 0.0f;

    output_buffer[neuron_index] = trace;
    output_buffer[neuron_count + neuron_index] = (membrane_potential - resting_mp) / voltage_scale;

    if (neuron_index == 0) {
        output_buffer[2 * neuron_count] = 1.0f;
    }
}

// ── step_apply_hebbian_update ─────────────────────────────────────────────────
// Mirrors step_apply_hebbian_update in src/metal/kernels.metal — single-pass
// rank-1 Hebbian nudge of U[source_node] (via u_row_accumulator) and V[target_node].
// The reference's l2_regularization * (anchor[d] - anchor[d]) term is always zero
// and is omitted as dead code.
//
// U[source_node] is only ever touched by the one step_kernel thread that owns
// source_node (active_neuron_indices has no duplicates in a given tick), so the
// caller passes it in as a plain register array — accumulated across every edge
// of that neuron with ordinary arithmetic and flushed to global memory once —
// instead of routing every edge's update through atomicAdd. V[target_node], in
// contrast, is genuinely shared across neurons' threads (fan-in), so it stays
// atomic; its anchor snapshot is now taken via atomicAdd(&slot, 0.0f) (CUDA has no
// atomic load for float, but a zero-add is a real atomic RMW and serializes with
// the atomicAdds below) instead of a plain load, removing the previous read/modify
// race against other threads' atomic adds to the same row.
__device__ __forceinline__ void step_apply_hebbian_update(
    float4 *u_row_accumulator,
    float4 *V,
    s64     rank_float4_stride,
    s32     target_node,
    f32     delta,
    f32     learning_rate,
    f32     l2_regularization
) {
    float4 anchor_u[MAX_RANK_FLOAT4_STRIDE];
    float4 anchor_v[MAX_RANK_FLOAT4_STRIDE];

    float4 *v_row = V + (s64)target_node * rank_float4_stride;

    f32 sum_u = l2_regularization;
    f32 sum_v = l2_regularization;
    for (s64 lane = 0; lane < rank_float4_stride; ++lane) {
        float4 u4 = u_row_accumulator[lane];
        float4 *v_slot = v_row + lane;
        float4 v4;
        v4.x = atomicAdd(&v_slot->x, 0.0f);
        v4.y = atomicAdd(&v_slot->y, 0.0f);
        v4.z = atomicAdd(&v_slot->z, 0.0f);
        v4.w = atomicAdd(&v_slot->w, 0.0f);
        anchor_u[lane] = u4;
        anchor_v[lane] = v4;
        sum_u += u4.x * u4.x + u4.y * u4.y + u4.z * u4.z + u4.w * u4.w;
        sum_v += v4.x * v4.x + v4.y * v4.y + v4.z * v4.z + v4.w * v4.w;
    }
    f32 inv_den_u = 1.0f / sum_u;
    f32 inv_den_v = 1.0f / sum_v;

    for (s64 lane = 0; lane < rank_float4_stride; ++lane) {
        f32 scale_u = learning_rate * delta * inv_den_v;
        f32 scale_v = learning_rate * delta * inv_den_u;
        float4 av = anchor_v[lane];
        float4 au = anchor_u[lane];

        u_row_accumulator[lane].x += av.x * scale_u;
        u_row_accumulator[lane].y += av.y * scale_u;
        u_row_accumulator[lane].z += av.z * scale_u;
        u_row_accumulator[lane].w += av.w * scale_u;

        float4 *v_slot = v_row + lane;
        atomicAdd(&v_slot->x, au.x * scale_v);
        atomicAdd(&v_slot->y, au.y * scale_v);
        atomicAdd(&v_slot->z, au.z * scale_v);
        atomicAdd(&v_slot->w, au.w * scale_v);
    }
}

// ── step ──────────────────────────────────────────────────────────────────────
// Uncompiled mirror of the Metal `step` kernel (src/metal/kernels.metal:488-614,
// finalized 2026-06-06) — pending verification on CUDA hardware (this machine has
// no CUDA toolchain). Runs one simulation tick: propagates spikes via k^2-tree
// adjacency and updates membrane potentials. One thread per *active* neuron —
// the global thread index indexes into active_neuron_indices, mirroring
// spikecore's step_kernel (cuda_code/kernels.c). Adjacency is resolved via
// k2t_next_neighbor, walking each spiking neuron's row once (O(D·H)).
__global__ void step_kernel(
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
    s32          *active_generation
) {
    s32 active_count = active_neuron_count[0];
    s32 thread_index = static_cast<s32>(blockIdx.x * blockDim.x + threadIdx.x);
    if (thread_index >= active_count) return;

    s32 neuron_thread_id = active_neuron_indices[thread_index];
    if (neuron_thread_id < 0 || (s64)neuron_thread_id >= neuron_count) return;

    // active_generation stores 32-bit generation tags (no 64-bit atomic exchange
    // on either platform), so next_tick is narrowed just for that. tick/last_spiked/
    // last_tick_updated stay 64-bit and compare directly — mirrors Metal exactly.
    s32 next_tick_i = (s32)next_tick;

    s64 last_updated_tick = last_tick_updated[neuron_thread_id];
    s64 time_since_last_update = tick - last_updated_tick;
    f32 membrane_potential = membrane_potentials[neuron_thread_id];
    membrane_potential = apply_decay(membrane_potential, resting_mp, decay_rate, (int)time_since_last_update);
    membrane_potential += network_inputs[neuron_thread_id];
    network_inputs[neuron_thread_id] = 0.0f;

    s64 time_last_spiked = last_spiked[neuron_thread_id];
    if ((tick - time_last_spiked) == spike_period) {
        membrane_potentials[neuron_thread_id] = resting_mp;
        last_tick_updated[neuron_thread_id] = tick;
        return;
    }

    if (membrane_potential <= spike_threshold) {
        membrane_potentials[neuron_thread_id] = membrane_potential;
        last_tick_updated[neuron_thread_id] = tick;
        return;
    }

    // spike: mark spike time, then propagate to every downstream neighbor —
    // discovered by walking neuron_thread_id's row in the k^2-tree (enumeration
    // stops at the first -1 sentinel; no fixed neighbor-count array).
    if ((tick - time_last_spiked) > spike_period) {
        last_spiked[neuron_thread_id] = tick;
    }

    // U[neuron_thread_id] is exclusively owned by this thread for the whole tick
    // (active_neuron_indices has no duplicates), so it's staged into registers
    // once, accumulated across every outgoing edge below with plain (non-atomic)
    // math, and flushed back to global memory a single time — instead of routing
    // each edge's Hebbian update through 4 atomicAdds per rank lane.
    float4 u_row_accumulator[MAX_RANK_FLOAT4_STRIDE];
    float4 *u_row_global = U + (s64)neuron_thread_id * rank_float4_stride;
    for (s64 lane = 0; lane < rank_float4_stride; ++lane) {
        u_row_accumulator[lane] = u_row_global[lane];
    }

    // Single DFS walk over the row — O(D·H) instead of O(D²·H).
    s32 walk_stack_row_base[MAX_K2TREE_HEIGHT];
    s32 walk_stack_col_base[MAX_K2TREE_HEIGHT];
    s32 walk_stack_block_size[MAX_K2TREE_HEIGHT];
    s32 walk_stack_bit_offset[MAX_K2TREE_HEIGHT];
    s32 walk_stack_next_col[MAX_K2TREE_HEIGHT];
    walk_stack_row_base[0]   = 0;
    walk_stack_col_base[0]   = 0;
    walk_stack_block_size[0] = padded_node_count;
    walk_stack_bit_offset[0] = 0;
    walk_stack_next_col[0]   = 0;
    s32 walk_stack_top = (tree_height > 0 && neuron_thread_id >= 0 &&
                          neuron_thread_id < (s32)neuron_count) ? 0 : -1;

    s32 child;
    while ((child = k2t_next_neighbor(
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, (s32)neuron_count,
        tree_height, internal_bit_count, neuron_thread_id,
        walk_stack_row_base, walk_stack_col_base, walk_stack_block_size,
        walk_stack_bit_offset, walk_stack_next_col, walk_stack_top
    )) >= 0) {

        // STDP Hebbian update — skip neighbors that have never spiked or spiked this tick
        s64 child_last_spiked = last_spiked[child];
        if (learning_rate != 0.0f && !(child_last_spiked == 0 || child_last_spiked == tick)) {
            s64 delta_ticks = tick - child_last_spiked;
            f32 tick_delta = (f32)(delta_ticks < 0 ? -delta_ticks : delta_ticks);
            f32 decay_delta = -learning_rate * powf(tick_delta, -3.0f);
            step_apply_hebbian_update(u_row_accumulator, V, rank_float4_stride, child,
                                      decay_delta, 0.5f, 1.0f);
        }

        // resolve the synaptic weight: constant (if configured) or U[source]·V[target]
        f32 weight = constant_weight;
        if (constant_weight == 0.0f) {
            const float4 *v_row = V + (s64)child * rank_float4_stride;
            f32 dot_product = 0.0f;
            for (s64 lane = 0; lane < rank_float4_stride; ++lane) {
                float4 u4 = u_row_accumulator[lane];
                float4 v4 = v_row[lane];
                dot_product += u4.x * v4.x + u4.y * v4.y + u4.z * v4.z + u4.w * v4.w;
            }
            weight = dot_product;
        }

        atomicAdd(&network_inputs[child], weight);

        // enqueue the neighbor into next tick's active set (once per generation)
        s32 previous_child_generation = atomicExch(&active_generation[child], next_tick_i);
        if (previous_child_generation != next_tick_i) {
            s32 position = atomicAdd(next_active_neuron_count, 1);
            next_active_neuron_indices[position] = child;
        }
    }

    for (s64 lane = 0; lane < rank_float4_stride; ++lane) {
        u_row_global[lane] = u_row_accumulator[lane];
    }

    // re-enqueue the spiking neuron itself for next tick (it may decay/spike again)
    s32 previous_self_generation = atomicExch(&active_generation[neuron_thread_id], next_tick_i);
    if (previous_self_generation != next_tick_i) {
        s32 position = atomicAdd(next_active_neuron_count, 1);
        next_active_neuron_indices[position] = neuron_thread_id;
    }

    membrane_potentials[neuron_thread_id] = membrane_potential;
    last_tick_updated[neuron_thread_id] = tick;
}

} // namespace

void launch_neighbor_weights(
    const float4 *U,
    const float4 *V,
    const u32    *internal_node_words,
    const u32    *leaf_node_words,
    const u32    *rank_superblock_table,
    const u16    *rank_subblock_table,
    s32 branching_factor,
    s32 superblock_size_words,
    s32 padded_node_count,
    s32 tree_height,
    s32 internal_bit_count,
    s64 node_count,
    s64 max_neighbor_count,
    s64 rank_float4_stride,
    f32 *output_weights,
    cudaStream_t stream
) {
    s64 total_pairs = node_count * max_neighbor_count;
    if (total_pairs <= 0) return;
    LaunchConfig cfg = default_launch_config(static_cast<usize>(total_pairs));
    neighbor_weights_kernel<<<cfg.grid, cfg.block, 0, stream>>>(
        U, V,
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, padded_node_count, tree_height, internal_bit_count,
        node_count, max_neighbor_count, rank_float4_stride, output_weights
    );
}

void launch_scale_uv(
    float4 *U,
    float4 *V,
    s64 total_float4_element_count,
    f32 scale_factor,
    cudaStream_t stream
) {
    if (total_float4_element_count <= 0) return;
    LaunchConfig cfg = default_launch_config(static_cast<usize>(total_float4_element_count));
    scale_uv_kernel<<<cfg.grid, cfg.block, 0, stream>>>(
        U, V, total_float4_element_count, scale_factor
    );
}

void launch_weight_update(
    float4 *U,
    float4 *V,
    s64 rank_float4_stride,
    s32 source_node,
    s32 target_node,
    f32 delta,
    f32 learning_rate,
    f32 l2_regularization,
    s32 iterations,
    cudaStream_t stream
) {
    if (rank_float4_stride <= 0 || iterations <= 0) return;

    unsigned threads = 32u;
    while (static_cast<s64>(threads) < rank_float4_stride) threads <<= 1;
    threads = threads > 1024u ? 1024u : threads;

    usize shared_bytes = static_cast<usize>(2 * rank_float4_stride) * sizeof(float4);
    weight_update_kernel<<<1, threads, shared_bytes, stream>>>(
        U, V, rank_float4_stride, source_node, target_node,
        delta, learning_rate, l2_regularization, iterations
    );
}

void launch_vector_add(
    f32 *result,
    const f32 *a,
    const f32 *b,
    s64 element_count,
    cudaStream_t stream
) {
    if (element_count <= 0) return;
    LaunchConfig cfg = default_launch_config(static_cast<usize>(element_count));
    vector_add_kernel<<<cfg.grid, cfg.block, 0, stream>>>(result, a, b, element_count);
}

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
    cudaStream_t  stream
) {
    if (block_count <= 0 || thread_count_per_block <= 0) return;
    step_kernel<<<static_cast<unsigned>(block_count), static_cast<unsigned>(thread_count_per_block), 0, stream>>>(
        tick, next_tick, spike_period, spike_threshold, learning_rate, decay_rate, resting_mp,
        U, V, rank_float4_stride, constant_weight,
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, padded_node_count, tree_height, internal_bit_count,
        neuron_count, network_inputs, membrane_potentials, last_spiked, last_tick_updated,
        active_neuron_indices, active_neuron_count, next_active_neuron_indices, next_active_neuron_count,
        active_generation
    );
}

void launch_add_network_input(
    f32       *membrane_potentials,
    const s32 *input_neuron_indices,
    const f32 *input_values,
    s64        element_count,
    cudaStream_t stream
) {
    if (element_count <= 0) return;
    LaunchConfig cfg = default_launch_config(static_cast<usize>(element_count));
    add_network_input_kernel<<<cfg.grid, cfg.block, 0, stream>>>(
        membrane_potentials, input_neuron_indices, input_values, element_count
    );
}

void launch_decay_all_neurons(
    f32 *membrane_potentials,
    s64 *last_tick_updated,
    s64  neuron_count,
    s64  tick,
    f32  resting_mp,
    f32  decay_rate,
    cudaStream_t stream
) {
    if (neuron_count <= 0) return;
    LaunchConfig cfg = default_launch_config(static_cast<usize>(neuron_count));
    decay_all_neurons_kernel<<<cfg.grid, cfg.block, 0, stream>>>(
        membrane_potentials, last_tick_updated, neuron_count, tick, resting_mp, decay_rate
    );
}

void launch_merge_input_neurons(
    s32       *active_neuron_indices,
    s32       *active_neuron_count,
    const s64 *override_input_neurons,
    s64        override_count,
    cudaStream_t stream
) {
    if (override_count <= 0) return;
    LaunchConfig cfg = default_launch_config(static_cast<usize>(override_count));
    merge_input_neurons_kernel<<<cfg.grid, cfg.block, 0, stream>>>(
        active_neuron_indices, active_neuron_count, override_input_neurons, override_count
    );
}

void launch_reservoir_features(
    s64        neuron_count,
    s64        tick,
    f32        spike_tau,
    f32        voltage_scale,
    f32       *membrane_potentials,
    const s64 *last_spiked,
    s64       *last_tick_updated,
    f32        resting_mp,
    f32        decay_rate,
    f32       *output_buffer,
    cudaStream_t stream
) {
    if (neuron_count <= 0) return;
    LaunchConfig cfg = default_launch_config(static_cast<usize>(neuron_count));
    reservoir_features_kernel<<<cfg.grid, cfg.block, 0, stream>>>(
        neuron_count, tick, spike_tau, voltage_scale, membrane_potentials,
        last_spiked, last_tick_updated, resting_mp, decay_rate, output_buffer
    );
}

void launch_k2tree_adjacent_batch(
    const u32 *internal_node_words,
    const u32 *leaf_node_words,
    const u32 *rank_superblock_table,
    const u16 *rank_subblock_table,
    s32 branching_factor,
    s32 superblock_size_words,
    s32 node_count,
    s32 padded_node_count,
    s32 tree_height,
    s32 internal_bit_count,
    const s32 *source_indices,
    const s32 *target_indices,
    uint8_t *output_buffer,
    s32 query_count,
    cudaStream_t stream
) {
    if (query_count <= 0) return;
    LaunchConfig cfg = default_launch_config(static_cast<usize>(query_count));
    k2tree_adjacent_batch_kernel<<<cfg.grid, cfg.block, 0, stream>>>(
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, node_count, padded_node_count,
        tree_height, internal_bit_count, source_indices, target_indices,
        output_buffer, query_count
    );
}

void launch_k2tree_get_neighbors_batch(
    const u32 *internal_node_words,
    const u32 *leaf_node_words,
    const u32 *rank_superblock_table,
    const u16 *rank_subblock_table,
    s32 branching_factor,
    s32 superblock_size_words,
    s32 node_count,
    s32 padded_node_count,
    s32 tree_height,
    s32 internal_bit_count,
    const s32 *source_node_indices,
    s32 query_count,
    s32 max_neighbor_count,
    s32 *output_buffer,
    cudaStream_t stream
) {
    s64 total_pairs = static_cast<s64>(query_count) * max_neighbor_count;
    if (total_pairs <= 0) return;
    LaunchConfig cfg = default_launch_config(static_cast<usize>(total_pairs));
    k2tree_get_neighbors_batch_kernel<<<cfg.grid, cfg.block, 0, stream>>>(
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, node_count, padded_node_count,
        tree_height, internal_bit_count, source_node_indices, query_count,
        max_neighbor_count, output_buffer
    );
}

} // namespace spikecorec::cuda
