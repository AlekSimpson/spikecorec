#include <metal_stdlib>
using namespace metal;

// ── k2tree bit-walk helpers ───────────────────────────────────────────────────
// Shared by neighbor_weights_kernel and k2tree_adjacent_batch_kernel.
inline uint k2t_get_bit(const device uint *words, int bit_index) {
    return (words[bit_index >> 5] >> (uint)(bit_index & 31)) & 1u;
}

inline int k2t_rank1_exclusive(
    const device uint *internal_words,
    const device uint *superblock_table,
    const device ushort *subblock_table,
    int position,
    int superblock_size
) {
    int word_index = position >> 5;
    int bit_offset = position & 31;
    int superblock_index = word_index / superblock_size;
    uint superblock_base = superblock_table[superblock_index];
    uint subblock_base = (uint)subblock_table[word_index];
    uint partial_word_mask = (bit_offset == 0) ? 0u : ((1u << (uint)bit_offset) - 1u);
    uint partial_word_popcount = popcount(internal_words[word_index] & partial_word_mask);
    return (int)(superblock_base + subblock_base + partial_word_popcount);
}

#define MAX_K2TREE_HEIGHT 32

// Finds the `target_slot`-th neighbor (0-indexed, in tree-traversal order) of row
// `u`, walking only the subtrees that intersect u's row and have at least one bit
// set (skipping empty regions entirely — the whole point of the k^2-tree). Returns
// the neighbor's node index, or -1 if `u` has fewer than `target_slot + 1` neighbors.
// Iterative DFS with an explicit per-level stack — Metal has no recursion, and
// depth is bounded by tree_height (<= MAX_K2TREE_HEIGHT for any branching_factor
// >= 2 and realistic node counts).
inline int k2t_find_nth_neighbor(
    const device uint   *internal_node_words,
    const device uint   *leaf_node_words,
    const device uint   *rank_superblock_table,
    const device ushort *rank_subblock_table,
    int branching_factor,
    int superblock_size_words,
    int node_count,
    int padded_node_count,
    int tree_height,
    int internal_bit_count,
    int u,
    int target_slot
) {
    if (tree_height == 0 || u < 0 || u >= node_count || target_slot < 0) return -1;

    int branching_factor_squared = branching_factor * branching_factor;

    int stack_row_base[MAX_K2TREE_HEIGHT];
    int stack_col_base[MAX_K2TREE_HEIGHT];
    int stack_block_size[MAX_K2TREE_HEIGHT];
    int stack_bit_offset[MAX_K2TREE_HEIGHT];
    int stack_next_col[MAX_K2TREE_HEIGHT];

    stack_row_base[0] = 0;
    stack_col_base[0] = 0;
    stack_block_size[0] = padded_node_count;
    stack_bit_offset[0] = 0;
    stack_next_col[0] = 0;

    int stack_top = 0;
    int neighbors_seen = 0;

    while (stack_top >= 0) {
        int level = stack_top;
        int col_offset = stack_next_col[level];
        if (col_offset >= branching_factor) {
            stack_top--;
            continue;
        }
        stack_next_col[level] = col_offset + 1;

        int row_base = stack_row_base[level];
        int col_base = stack_col_base[level];
        int block_size = stack_block_size[level];
        int level_bit_offset = stack_bit_offset[level];

        int child_block_size = block_size / branching_factor;
        int row_offset = (u - row_base) / child_block_size;
        int child_flat_index = row_offset * branching_factor + col_offset;
        int bit_position = level_bit_offset + child_flat_index;

        if (level == tree_height - 1) {
            if (k2t_get_bit(leaf_node_words, bit_position)) {
                int v = col_base + col_offset;
                if (v < node_count) {
                    if (neighbors_seen == target_slot) return v;
                    neighbors_seen++;
                }
            }
        } else if (k2t_get_bit(internal_node_words, bit_position)) {
            int rank_inclusive = k2t_rank1_exclusive(internal_node_words, rank_superblock_table,
                                                      rank_subblock_table, bit_position, superblock_size_words) + 1;
            int child_level = stack_top + 1;
            int raw_offset = branching_factor_squared * rank_inclusive;
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
inline int k2t_next_neighbor(
    const device uint   *internal_node_words,
    const device uint   *leaf_node_words,
    const device uint   *rank_superblock_table,
    const device ushort *rank_subblock_table,
    int branching_factor,
    int superblock_size_words,
    int node_count,
    int tree_height,
    int internal_bit_count,
    int u,
    thread int *stack_row_base,
    thread int *stack_col_base,
    thread int *stack_block_size,
    thread int *stack_bit_offset,
    thread int *stack_next_col,
    thread int &stack_top
) {
    int branching_factor_squared = branching_factor * branching_factor;

    while (stack_top >= 0) {
        int level      = stack_top;
        int col_offset = stack_next_col[level];
        if (col_offset >= branching_factor) {
            stack_top--;
            continue;
        }
        stack_next_col[level] = col_offset + 1;

        int row_base         = stack_row_base[level];
        int col_base         = stack_col_base[level];
        int block_size       = stack_block_size[level];
        int level_bit_offset = stack_bit_offset[level];

        int child_block_size = block_size / branching_factor;
        int row_offset       = (u - row_base) / child_block_size;
        int child_flat_index = row_offset * branching_factor + col_offset;
        int bit_position     = level_bit_offset + child_flat_index;

        if (level == tree_height - 1) {
            if (k2t_get_bit(leaf_node_words, bit_position)) {
                int v = col_base + col_offset;
                if (v < node_count) return v;
            }
        } else if (k2t_get_bit(internal_node_words, bit_position)) {
            int rank_inclusive = k2t_rank1_exclusive(internal_node_words, rank_superblock_table,
                                                      rank_subblock_table, bit_position, superblock_size_words) + 1;
            int child_level = stack_top + 1;
            int raw_offset  = branching_factor_squared * rank_inclusive;
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
// (or one threadgroup) per source_node — walking the row once and filling all
// max_neighbor_count slots — would eliminate that redundancy.
kernel void neighbor_weights_kernel(
    const device float4 *U                      [[ buffer(0) ]],
    const device float4 *V                      [[ buffer(1) ]],
    const device uint   *internal_node_words    [[ buffer(2) ]],
    const device uint   *leaf_node_words        [[ buffer(3) ]],
    const device uint   *rank_superblock_table  [[ buffer(4) ]],
    const device ushort *rank_subblock_table    [[ buffer(5) ]],
    constant int        &branching_factor       [[ buffer(6) ]],
    constant int        &superblock_size_words  [[ buffer(7) ]],
    constant int        &padded_node_count      [[ buffer(8) ]],
    constant int        &tree_height            [[ buffer(9) ]],
    constant int        &internal_bit_count     [[ buffer(10) ]],
    constant long       &node_count             [[ buffer(11) ]],
    constant long       &max_neighbor_count     [[ buffer(12) ]],
    constant long       &rank_float4_stride     [[ buffer(13) ]],
    device float        *output_weights         [[ buffer(14) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    long pair_index = (long)thread_id;
    long total_pairs = node_count * max_neighbor_count;
    if (pair_index >= total_pairs) return;

    long source_node = pair_index / max_neighbor_count;
    long neighbor_slot = pair_index % max_neighbor_count;
    const device float4 *u_row = U + source_node * rank_float4_stride;

    int target_node = k2t_find_nth_neighbor(
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, (int)node_count, padded_node_count,
        tree_height, internal_bit_count, (int)source_node, (int)neighbor_slot
    );
    if (target_node < 0) {
        output_weights[pair_index] = 0.0f;
        return;
    }
    const device float4 *v_row = V + (long)target_node * rank_float4_stride;

    float dot_product = 0.0f;
    for (long lane = 0; lane < rank_float4_stride; ++lane) {
        dot_product += dot(u_row[lane], v_row[lane]);
    }
    output_weights[pair_index] = dot_product;
}

// ── scale_uv ──────────────────────────────────────────────────────────────────
// Pure memory-bound element-wise scale; one thread per float4 lane, fully coalesced.
kernel void scale_uv_kernel(
    device float4 *U                            [[ buffer(0) ]],
    device float4 *V                            [[ buffer(1) ]],
    constant long  &total_float4_element_count  [[ buffer(2) ]],
    constant float &scale_factor                [[ buffer(3) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    long index = (long)thread_id;
    if (index >= total_float4_element_count) return;
    U[index] *= scale_factor;
    V[index] *= scale_factor;
}

kernel void add_network_input_kernel(
    device float      *membrane_potentials  [[ buffer(0) ]],
    const device int  *input_neuron_indices [[ buffer(1) ]],
    const device float *input_values        [[ buffer(2) ]],
    constant long     &element_count        [[ buffer(3) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    long index = (long)thread_id;
    if (index >= element_count) return;

    int neuron_index = input_neuron_indices[index];
    // atomic: input_neuron_indices is caller-supplied with no uniqueness guarantee —
    // a repeated index would otherwise race as a lost-update read-modify-write.
    device atomic_float *target_slot = (device atomic_float *)(membrane_potentials + neuron_index);
    atomic_fetch_add_explicit(target_slot, input_values[index], memory_order_relaxed);
}

// ── weight_update ─────────────────────────────────────────────────────────────
// Single (source, target) pair, run for `iterations` rounds of proximal Hebbian
// update. Dispatched as one threadgroup of rank_float4_stride threads (rank is
// always small — e.g. 16 float4 lanes for rank=64). Anchors (the rows' values
// before the first iteration) are cached in threadgroup memory; den_u/den_v
// (squared-norm denominators) are recomputed each round via simd_sum + a
// cross-simdgroup tree reduction since the rows change between rounds.
#define MAX_RANK_FLOAT4_STRIDE 64
#define MAX_SIMD_GROUPS 32

kernel void weight_update_kernel(
    device float4 *U                   [[ buffer(0) ]],
    device float4 *V                   [[ buffer(1) ]],
    constant long  &rank_float4_stride [[ buffer(2) ]],
    constant int   &source_node        [[ buffer(3) ]],
    constant int   &target_node        [[ buffer(4) ]],
    constant float &delta              [[ buffer(5) ]],
    constant float &learning_rate      [[ buffer(6) ]],
    constant float &l2_regularization  [[ buffer(7) ]],
    constant int   &iterations         [[ buffer(8) ]],
    uint lane                          [[ thread_position_in_threadgroup ]],
    uint simd_lane                     [[ thread_index_in_simdgroup ]],
    uint simd_group                    [[ simdgroup_index_in_threadgroup ]],
    uint simd_groups_per_threadgroup   [[ simdgroups_per_threadgroup ]]
) {
    threadgroup float4 anchor_u[MAX_RANK_FLOAT4_STRIDE];
    threadgroup float4 anchor_v[MAX_RANK_FLOAT4_STRIDE];
    threadgroup float  warp_partial_u[MAX_SIMD_GROUPS];
    threadgroup float  warp_partial_v[MAX_SIMD_GROUPS];
    threadgroup float  den_u;
    threadgroup float  den_v;

    bool active = (long)lane < rank_float4_stride;

    device float4 *u_row = U + (long)source_node * rank_float4_stride;
    device float4 *v_row = V + (long)target_node * rank_float4_stride;

    float4 u4 = active ? u_row[lane] : float4(0.0f);
    float4 v4 = active ? v_row[lane] : float4(0.0f);

    if (active) {
        anchor_u[lane] = u4;
        anchor_v[lane] = v4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        float group_u = simd_sum(dot(u4, u4));
        float group_v = simd_sum(dot(v4, v4));
        if (simd_lane == 0) {
            warp_partial_u[simd_group] = group_u;
            warp_partial_v[simd_group] = group_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0) {
            float sum_u = 0.0f, sum_v = 0.0f;
            for (uint group = 0; group < simd_groups_per_threadgroup; ++group) {
                sum_u += warp_partial_u[group];
                sum_v += warp_partial_v[group];
            }
            den_u = l2_regularization + sum_u;
            den_v = l2_regularization + sum_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            float4 au = anchor_u[lane];
            float4 av = anchor_v[lane];
            float inv_den_u = 1.0f / den_u;
            float inv_den_v = 1.0f / den_v;

            float4 du = learning_rate * (delta * (v4 * inv_den_v) - l2_regularization * (u4 - au));
            float4 dv = learning_rate * (delta * (u4 * inv_den_u) - l2_regularization * (v4 - av));

            u4 += du;
            v4 += dv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
kernel void k2tree_adjacent_batch_kernel(
    const device uint   *internal_node_words   [[ buffer(0) ]],
    const device uint   *leaf_node_words       [[ buffer(1) ]],
    const device uint   *rank_superblock_table [[ buffer(2) ]],
    const device ushort *rank_subblock_table   [[ buffer(3) ]],
    constant int &branching_factor             [[ buffer(4) ]],
    constant int &superblock_size_words        [[ buffer(5) ]],
    constant int &node_count                   [[ buffer(6) ]],
    constant int &padded_node_count            [[ buffer(7) ]],
    constant int &tree_height                  [[ buffer(8) ]],
    constant int &internal_bit_count           [[ buffer(9) ]],
    const device int  *source_indices          [[ buffer(10) ]],
    const device int  *target_indices          [[ buffer(11) ]],
    device uchar      *output_buffer           [[ buffer(12) ]],
    constant int &query_count                  [[ buffer(13) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    int query_index = (int)thread_id;
    if (query_index >= query_count) return;

    int u = source_indices[query_index];
    int v = target_indices[query_index];

    if (u < 0 || u >= node_count || v < 0 || v >= node_count || tree_height == 0) {
        output_buffer[query_index] = 0;
        return;
    }

    int branching_factor_squared = branching_factor * branching_factor;

    if (tree_height == 1) {
        int block_size = padded_node_count / branching_factor;
        int child_flat_index = (u / block_size) * branching_factor + (v / block_size);
        output_buffer[query_index] = (uchar)k2t_get_bit(leaf_node_words, child_flat_index);
        return;
    }

    int level_bit_offset = 0;
    int current_block_size = padded_node_count;
    int rank_inclusive = 0;

    for (int level = 0; level < tree_height - 1; ++level) {
        int block_size = current_block_size / branching_factor;
        int row_offset = u / block_size;
        int column_offset = v / block_size;
        int child_bit_position = level_bit_offset + row_offset * branching_factor + column_offset;

        if (k2t_get_bit(internal_node_words, child_bit_position) == 0u) {
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

    int leaf_bit_offset = branching_factor_squared * rank_inclusive - internal_bit_count;
    output_buffer[query_index] = (uchar)k2t_get_bit(leaf_node_words, leaf_bit_offset + u * branching_factor + v);
}

// ── k2tree_get_neighbors_batch ────────────────────────────────────────────────
// Batched row enumeration: one thread per (query_index, neighbor_slot) pair;
// each thread independently walks source_node_indices[query_index]'s row in the
// k^2-tree to find its (neighbor_slot)-th neighbor (or -1 if it has fewer).
// Mirrors neighbor_weights_kernel's enumeration but surfaces the discovered
// indices directly instead of consuming them for a dot product.
kernel void k2tree_get_neighbors_batch_kernel(
    const device uint   *internal_node_words   [[ buffer(0) ]],
    const device uint   *leaf_node_words       [[ buffer(1) ]],
    const device uint   *rank_superblock_table [[ buffer(2) ]],
    const device ushort *rank_subblock_table   [[ buffer(3) ]],
    constant int &branching_factor             [[ buffer(4) ]],
    constant int &superblock_size_words        [[ buffer(5) ]],
    constant int &node_count                   [[ buffer(6) ]],
    constant int &padded_node_count            [[ buffer(7) ]],
    constant int &tree_height                  [[ buffer(8) ]],
    constant int &internal_bit_count           [[ buffer(9) ]],
    const device int  *source_node_indices     [[ buffer(10) ]],
    constant int &query_count                  [[ buffer(11) ]],
    constant int &max_neighbor_count           [[ buffer(12) ]],
    device int        *output_buffer           [[ buffer(13) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    long pair_index = (long)thread_id;
    long total_pairs = (long)query_count * (long)max_neighbor_count;
    if (pair_index >= total_pairs) return;

    int query_index = (int)(pair_index / max_neighbor_count);
    int neighbor_slot = (int)(pair_index % max_neighbor_count);
    int source_node = source_node_indices[query_index];

    output_buffer[pair_index] = k2t_find_nth_neighbor(
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, node_count, padded_node_count,
        tree_height, internal_bit_count, source_node, neighbor_slot
    );
}


inline float apply_decay(float membrane_potential, float resting_potential, float decay_rate, int time_delta) {
    if (time_delta <= 0) return membrane_potential;

    if (time_delta == 1) return resting_potential + (membrane_potential - resting_potential) * (1.0f - decay_rate);

    float decay = pow(1.0f - decay_rate, (float)time_delta);
    return resting_potential + (membrane_potential - resting_potential) * decay;
}

// ── step_apply_hebbian_update ─────────────────────────────────────────────────
// Single-pass rank-1 Hebbian nudge of U[source_node] (via u_row_accumulator) and
// V[target_node], mirroring spikecore's update_weight_matrix (cuda_code/kernels.c).
// NOTE: the reference's l2_regularization * (anchor[d] - anchor[d]) term is always
// zero (anchor minus itself), so it's omitted here as dead code — the effective
// update is purely the delta-scaled anchor term, matching reference behavior.
//
// U[source_node] is only ever touched by the one `step` thread that owns
// source_node (active_neuron_indices has no duplicates in a given tick), so the
// caller passes it in as a plain thread-local register array — accumulated across
// every edge of that neuron and flushed to device memory once — instead of routing
// every edge's update through a global atomic. V[target_node], in contrast, is
// genuinely shared across neurons' threads (fan-in), so it remains atomic; its
// anchor snapshot is now read via atomic_load_explicit instead of a plain load,
// removing the previous read/modify race against other threads' atomic adds to
// the same row.
inline void step_apply_hebbian_update(
    thread float4 *u_row_accumulator,
    device float4 *V,
    long rank_float4_stride,
    int target_node,
    float delta,
    float learning_rate,
    float l2_regularization
) {
    thread float4 anchor_u[MAX_RANK_FLOAT4_STRIDE];
    thread float4 anchor_v[MAX_RANK_FLOAT4_STRIDE];

    device float4 *v_row = V + (long)target_node * rank_float4_stride;

    float sum_u = l2_regularization;
    float sum_v = l2_regularization;
    for (long lane = 0; lane < rank_float4_stride; ++lane) {
        device atomic_float *v_components = (device atomic_float *)(v_row + lane);
        float4 u4 = u_row_accumulator[lane];
        float4 v4(
            atomic_load_explicit(&v_components[0], memory_order_relaxed),
            atomic_load_explicit(&v_components[1], memory_order_relaxed),
            atomic_load_explicit(&v_components[2], memory_order_relaxed),
            atomic_load_explicit(&v_components[3], memory_order_relaxed)
        );
        anchor_u[lane] = u4;
        anchor_v[lane] = v4;
        sum_u += dot(u4, u4);
        sum_v += dot(v4, v4);
    }
    float inv_den_u = 1.0f / sum_u;
    float inv_den_v = 1.0f / sum_v;

    for (long lane = 0; lane < rank_float4_stride; ++lane) {
        float4 du = learning_rate * delta * (anchor_v[lane] * inv_den_v);
        float4 dv = learning_rate * delta * (anchor_u[lane] * inv_den_u);

        u_row_accumulator[lane] += du;

        device atomic_float *v_components = (device atomic_float *)(v_row + lane);
        atomic_fetch_add_explicit(&v_components[0], dv.x, memory_order_relaxed);
        atomic_fetch_add_explicit(&v_components[1], dv.y, memory_order_relaxed);
        atomic_fetch_add_explicit(&v_components[2], dv.z, memory_order_relaxed);
        atomic_fetch_add_explicit(&v_components[3], dv.w, memory_order_relaxed);
    }
}


// ── step ──────────────────────────────────────────────────────────────────────
// Runs one simulation tick: propagates spikes via k^2-tree adjacency and updates
// membrane potentials. One thread per *active* neuron — thread_position_in_grid
// indexes into active_neuron_indices, mirroring spikecore's step_kernel
// (cuda_code/kernels.c). Adjacency is resolved via k2t_next_neighbor, walking
// each spiking neuron's row once (O(D·H)). Buffer indices below
// mirror the order gpu_step's Metal wrapper marshals args in (backend.cpp); note
// block_count is not included — the wrapper's arg_count (31) stops just short of
// it, so the kernel has no buffer(31) parameter for it.
kernel void step(
    constant long        &tick                       [[ buffer(0) ]],
    constant long        &next_tick                  [[ buffer(1) ]],
    constant int         &spike_period               [[ buffer(2) ]],
    constant float       &spike_threshold            [[ buffer(3) ]],
    constant float       &learning_rate              [[ buffer(4) ]],
    constant float       &decay_rate                 [[ buffer(5) ]],
    constant float       &resting_mp                 [[ buffer(6) ]],
    device float4        *U                          [[ buffer(7) ]],
    device float4        *V                          [[ buffer(8) ]],
    constant long        &rank_float4_stride         [[ buffer(9) ]],
    constant float       &constant_weight            [[ buffer(10) ]],
    const device uint    *internal_node_words        [[ buffer(11) ]],
    const device uint    *leaf_node_words            [[ buffer(12) ]],
    const device uint    *rank_superblock_table      [[ buffer(13) ]],
    const device ushort  *rank_subblock_table        [[ buffer(14) ]],
    constant int         &branching_factor           [[ buffer(15) ]],
    constant int         &superblock_size_words      [[ buffer(16) ]],
    constant int         &padded_node_count          [[ buffer(17) ]],
    constant int         &tree_height                [[ buffer(18) ]],
    constant int         &internal_bit_count         [[ buffer(19) ]],
    constant long        &neuron_count               [[ buffer(20) ]],
    device float         *network_inputs             [[ buffer(21) ]],
    device float         *membrane_potentials        [[ buffer(22) ]],
    device long          *last_spiked                [[ buffer(23) ]],
    device long          *last_tick_updated          [[ buffer(24) ]],
    const device int     *active_neuron_indices      [[ buffer(25) ]],
    const device int     *active_neuron_count        [[ buffer(26) ]],
    device int           *next_active_neuron_indices [[ buffer(27) ]],
    device int           *next_active_neuron_count   [[ buffer(28) ]],
    device int           *active_generation          [[ buffer(29) ]],
    constant int         &thread_count_per_block     [[ buffer(30) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    int active_count = active_neuron_count[0];
    int thread_index = (int)thread_id;
    if (thread_index >= active_count) return;

    int neuron_thread_id = active_neuron_indices[thread_index];
    if (neuron_thread_id < 0 || (long)neuron_thread_id >= neuron_count) return;

    // active_generation stores 32-bit generation tags (Metal has no 64-bit atomic
    // exchange/fetch_add — only int/uint), so next_tick is narrowed just for that.
    // tick/last_spiked/last_tick_updated stay 64-bit and compare directly.
    int next_tick_i = (int)next_tick;

    long last_updated_tick = last_tick_updated[neuron_thread_id];
    long time_since_last_update = tick - last_updated_tick;
    float membrane_potential = membrane_potentials[neuron_thread_id];
    membrane_potential = apply_decay(membrane_potential, resting_mp, decay_rate, (int)time_since_last_update);
    device atomic_float *self_input_slot = (device atomic_float *)(network_inputs + neuron_thread_id);
    float accumulated_input = atomic_exchange_explicit(self_input_slot, 0.0f, memory_order_relaxed);
    membrane_potential += accumulated_input;

    long time_last_spiked = last_spiked[neuron_thread_id];
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
    // discovered by walking neuron_thread_id's row in the k^2-tree (no fixed
    // neighbor-count array; enumeration stops at the first -1 sentinel).
    if ((tick - time_last_spiked) > spike_period) {
        last_spiked[neuron_thread_id] = tick;
    }

    // U[neuron_thread_id] is exclusively owned by this thread for the whole tick
    // (active_neuron_indices has no duplicates), so it's staged into thread-local
    // registers once, accumulated across every outgoing edge below with plain
    // (non-atomic) math, and flushed back to device memory a single time — instead
    // of routing each edge's Hebbian update through 4 atomics per rank lane.
    thread float4 u_row_accumulator[MAX_RANK_FLOAT4_STRIDE];
    device float4 *u_row_global = U + (long)neuron_thread_id * rank_float4_stride;
    for (long lane = 0; lane < rank_float4_stride; ++lane) {
        u_row_accumulator[lane] = u_row_global[lane];
    }

    // Single DFS walk over the row — O(D·H) instead of O(D²·H).
    thread int walk_stack_row_base[MAX_K2TREE_HEIGHT];
    thread int walk_stack_col_base[MAX_K2TREE_HEIGHT];
    thread int walk_stack_block_size[MAX_K2TREE_HEIGHT];
    thread int walk_stack_bit_offset[MAX_K2TREE_HEIGHT];
    thread int walk_stack_next_col[MAX_K2TREE_HEIGHT];
    walk_stack_row_base[0]   = 0;
    walk_stack_col_base[0]   = 0;
    walk_stack_block_size[0] = padded_node_count;
    walk_stack_bit_offset[0] = 0;
    walk_stack_next_col[0]   = 0;
    int walk_stack_top = (tree_height > 0 && neuron_thread_id >= 0 &&
                          neuron_thread_id < (int)neuron_count) ? 0 : -1;

    int child;
    while ((child = k2t_next_neighbor(
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, (int)neuron_count,
        tree_height, internal_bit_count, neuron_thread_id,
        walk_stack_row_base, walk_stack_col_base, walk_stack_block_size,
        walk_stack_bit_offset, walk_stack_next_col, walk_stack_top
    )) >= 0) {
        // STDP Hebbian update — skip neighbors that have never spiked or spiked this tick
        long child_last_spiked = last_spiked[child];
        if (learning_rate != 0.0f && !(child_last_spiked == 0 || child_last_spiked == tick)) {
            float tick_delta = (float)abs(tick - child_last_spiked);
            float decay_delta = -learning_rate * pow(tick_delta, -3.0f);
            step_apply_hebbian_update(u_row_accumulator, V, rank_float4_stride, child,
                                      decay_delta, 0.5f, 1.0f);
        }

        // resolve the synaptic weight: constant (if configured) or U[source]·V[target]
        float weight = constant_weight;
        if (constant_weight == 0.0f) {
            const device float4 *v_row = V + (long)child * rank_float4_stride;
            float dot_product = 0.0f;
            for (long lane = 0; lane < rank_float4_stride; ++lane) {
                dot_product += dot(u_row_accumulator[lane], v_row[lane]);
            }
            weight = dot_product;
        }

        device atomic_float *input_slot = (device atomic_float *)(network_inputs + child);
        atomic_fetch_add_explicit(input_slot, weight, memory_order_relaxed);

        // enqueue the neighbor into next tick's active set (once per generation)
        device atomic_int *child_generation_slot = (device atomic_int *)(active_generation + child);
        int previous_child_generation = atomic_exchange_explicit(child_generation_slot, next_tick_i, memory_order_relaxed);
        if (previous_child_generation != next_tick_i) {
            device atomic_int *next_count_slot = (device atomic_int *)next_active_neuron_count;
            int position = atomic_fetch_add_explicit(next_count_slot, 1, memory_order_relaxed);
            next_active_neuron_indices[position] = child;
        }
    }

    for (long lane = 0; lane < rank_float4_stride; ++lane) {
        u_row_global[lane] = u_row_accumulator[lane];
    }

    // re-enqueue the spiking neuron itself for next tick (it may decay/spike again)
    device atomic_int *self_generation_slot = (device atomic_int *)(active_generation + neuron_thread_id);
    int previous_self_generation = atomic_exchange_explicit(self_generation_slot, next_tick_i, memory_order_relaxed);
    if (previous_self_generation != next_tick_i) {
        device atomic_int *next_count_slot = (device atomic_int *)next_active_neuron_count;
        int position = atomic_fetch_add_explicit(next_count_slot, 1, memory_order_relaxed);
        next_active_neuron_indices[position] = neuron_thread_id;
    }

    membrane_potentials[neuron_thread_id] = membrane_potential;
    last_tick_updated[neuron_thread_id] = tick;
}

kernel void step_no_active_optimization(
    constant long        &tick                       [[ buffer(0) ]],
    constant long        &next_tick                  [[ buffer(1) ]],
    constant int         &spike_period               [[ buffer(2) ]],
    constant float       &spike_threshold            [[ buffer(3) ]],
    constant float       &learning_rate              [[ buffer(4) ]],
    constant float       &decay_rate                 [[ buffer(5) ]],
    constant float       &resting_mp                 [[ buffer(6) ]],
    device float4        *U                          [[ buffer(7) ]],
    device float4        *V                          [[ buffer(8) ]],
    constant long        &rank_float4_stride         [[ buffer(9) ]],
    constant float       &constant_weight            [[ buffer(10) ]],
    const device uint    *internal_node_words        [[ buffer(11) ]],
    const device uint    *leaf_node_words            [[ buffer(12) ]],
    const device uint    *rank_superblock_table      [[ buffer(13) ]],
    const device ushort  *rank_subblock_table        [[ buffer(14) ]],
    constant int         &branching_factor           [[ buffer(15) ]],
    constant int         &superblock_size_words      [[ buffer(16) ]],
    constant int         &padded_node_count          [[ buffer(17) ]],
    constant int         &tree_height                [[ buffer(18) ]],
    constant int         &internal_bit_count         [[ buffer(19) ]],
    constant long        &neuron_count               [[ buffer(20) ]],
    device float         *network_inputs             [[ buffer(21) ]],
    device float         *membrane_potentials        [[ buffer(22) ]],
    device long          *last_spiked                [[ buffer(23) ]],
    device long          *last_tick_updated          [[ buffer(24) ]],
    constant int         &thread_count_per_block     [[ buffer(25) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    int neuron_thread_id = (int)thread_id;

    if (neuron_thread_id < 0 || (long)neuron_thread_id >= neuron_count) return;

    // active_generation stores 32-bit generation tags (Metal has no 64-bit atomic
    // exchange/fetch_add — only int/uint), so next_tick is narrowed just for that.
    // tick/last_spiked/last_tick_updated stay 64-bit and compare directly.
    int next_tick_i = (int)next_tick;

    long last_updated_tick = last_tick_updated[neuron_thread_id];
    long time_since_last_update = tick - last_updated_tick;
    float membrane_potential = membrane_potentials[neuron_thread_id];
    membrane_potential = apply_decay(membrane_potential, resting_mp, decay_rate, (int)time_since_last_update);
    device atomic_float *self_input_slot = (device atomic_float *)(network_inputs + neuron_thread_id);
    float accumulated_input = atomic_exchange_explicit(self_input_slot, 0.0f, memory_order_relaxed);
    membrane_potential += accumulated_input;

    long time_last_spiked = last_spiked[neuron_thread_id];
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
    // discovered by walking neuron_thread_id's row in the k^2-tree (no fixed
    // neighbor-count array; enumeration stops at the first -1 sentinel).
    if ((tick - time_last_spiked) > spike_period) {
        last_spiked[neuron_thread_id] = tick;
    }

    // U[neuron_thread_id] is exclusively owned by this thread for the whole tick
    // (active_neuron_indices has no duplicates), so it's staged into thread-local
    // registers once, accumulated across every outgoing edge below with plain
    // (non-atomic) math, and flushed back to device memory a single time — instead
    // of routing each edge's Hebbian update through 4 atomics per rank lane.
    thread float4 u_row_accumulator[MAX_RANK_FLOAT4_STRIDE];
    device float4 *u_row_global = U + (long)neuron_thread_id * rank_float4_stride;
    for (long lane = 0; lane < rank_float4_stride; ++lane) {
        u_row_accumulator[lane] = u_row_global[lane];
    }

    // Single DFS walk over the row — O(D·H) instead of O(D²·H).
    thread int walk_stack_row_base[MAX_K2TREE_HEIGHT];
    thread int walk_stack_col_base[MAX_K2TREE_HEIGHT];
    thread int walk_stack_block_size[MAX_K2TREE_HEIGHT];
    thread int walk_stack_bit_offset[MAX_K2TREE_HEIGHT];
    thread int walk_stack_next_col[MAX_K2TREE_HEIGHT];
    walk_stack_row_base[0]   = 0;
    walk_stack_col_base[0]   = 0;
    walk_stack_block_size[0] = padded_node_count;
    walk_stack_bit_offset[0] = 0;
    walk_stack_next_col[0]   = 0;
    int walk_stack_top = (tree_height > 0 && neuron_thread_id >= 0 &&
                          neuron_thread_id < (int)neuron_count) ? 0 : -1;

    int child;
    while ((child = k2t_next_neighbor(
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, (int)neuron_count,
        tree_height, internal_bit_count, neuron_thread_id,
        walk_stack_row_base, walk_stack_col_base, walk_stack_block_size,
        walk_stack_bit_offset, walk_stack_next_col, walk_stack_top
    )) >= 0) {
        // STDP Hebbian update — skip neighbors that have never spiked or spiked this tick
        long child_last_spiked = last_spiked[child];
        if (learning_rate != 0.0f && !(child_last_spiked == 0 || child_last_spiked == tick)) {
            float tick_delta = (float)abs(tick - child_last_spiked);
            float decay_delta = -learning_rate * pow(tick_delta, -3.0f);
            step_apply_hebbian_update(u_row_accumulator, V, rank_float4_stride, child,
                                      decay_delta, 0.5f, 1.0f);
        }

        // resolve the synaptic weight: constant (if configured) or U[source]·V[target]
        float weight = constant_weight;
        if (constant_weight == 0.0f) {
            const device float4 *v_row = V + (long)child * rank_float4_stride;
            float dot_product = 0.0f;
            for (long lane = 0; lane < rank_float4_stride; ++lane) {
                dot_product += dot(u_row_accumulator[lane], v_row[lane]);
            }
            weight = dot_product;
        }

        device atomic_float *input_slot = (device atomic_float *)(network_inputs + child);
        atomic_fetch_add_explicit(input_slot, weight, memory_order_relaxed);
    }

    for (long lane = 0; lane < rank_float4_stride; ++lane) {
        u_row_global[lane] = u_row_accumulator[lane];
    }

    membrane_potentials[neuron_thread_id] = membrane_potential;
    last_tick_updated[neuron_thread_id] = tick;
}

kernel void decay_all_neurons_kernel(
    device float   *membrane_potentials [[ buffer(0) ]],
    device long    *last_tick_updated   [[ buffer(1) ]],
    constant long  &neuron_count        [[ buffer(2) ]],
    constant long  &tick                [[ buffer(3) ]],
    constant float &resting_mp          [[ buffer(4) ]],
    constant float &decay_rate          [[ buffer(5) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    long neuron_index = (long)thread_id;
    if (neuron_index >= neuron_count) return;

    long time_delta = tick - last_tick_updated[neuron_index];
    membrane_potentials[neuron_index] = apply_decay(
        membrane_potentials[neuron_index], resting_mp, decay_rate, (int)time_delta);
    last_tick_updated[neuron_index] = tick;
}

kernel void merge_input_neurons_kernel(
    device int        *active_neuron_indices  [[ buffer(0) ]],
    device int        *active_neuron_count    [[ buffer(1) ]],
    const device long *override_input_neurons [[ buffer(2) ]],
    constant long     &override_count         [[ buffer(3) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    long override_index = (long)thread_id;
    if (override_index >= override_count) return;

    long candidate = override_input_neurons[override_index];
    if (candidate < 0) return;
    int neuron = (int)candidate;

    int current_count = active_neuron_count[0];
    for (int scan_index = 0; scan_index < current_count; ++scan_index) {
        if (active_neuron_indices[scan_index] == neuron) return;
    }

    device atomic_int *count_slot = (device atomic_int *)active_neuron_count;
    int position = atomic_fetch_add_explicit(count_slot, 1, memory_order_relaxed);
    active_neuron_indices[position] = neuron;
}

kernel void reservoir_features_kernel(
    constant long     &neuron_count       [[ buffer(0) ]],
    constant long     &tick               [[ buffer(1) ]],
    constant float    &spike_tau          [[ buffer(2) ]],
    constant float    &voltage_scale      [[ buffer(3) ]],
    device float      *membrane_potentials [[ buffer(4) ]],
    const device long *last_spiked         [[ buffer(5) ]],
    device long       *last_tick_updated   [[ buffer(6) ]],
    constant float    &resting_mp          [[ buffer(7) ]],
    constant float    &decay_rate          [[ buffer(8) ]],
    device float      *output_buffer       [[ buffer(9) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    long neuron_index = (long)thread_id;
    if (neuron_index >= neuron_count) return;

    long time_delta = tick - last_tick_updated[neuron_index];
    float membrane_potential = apply_decay(
        membrane_potentials[neuron_index], resting_mp, decay_rate, (int)time_delta);
    membrane_potentials[neuron_index] = membrane_potential;
    last_tick_updated[neuron_index] = tick;

    long since_spike = tick - last_spiked[neuron_index];
    float trace = (last_spiked[neuron_index] > 0) ? exp(-(float)since_spike / spike_tau) : 0.0f;

    output_buffer[neuron_index] = trace;
    output_buffer[neuron_count + neuron_index] = (membrane_potential - resting_mp) / voltage_scale;

    if (neuron_index == 0) {
        output_buffer[2 * neuron_count] = 1.0f;
    }
}

