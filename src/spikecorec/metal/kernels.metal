#include <metal_stdlib>
using namespace metal;

#include "k2tree_device.metalinc"

// ── neighbor_weights ──────────────────────────────────────────────────────────
// One plane's basis value on every edge, one thread per edge ordinal. `coefficients` is the
// plane's Ck row, so the value is the sum over lanes of U[source] * Ck * V[target].
// TODO: threads sharing a source each re-walk its row from the root to find their slot;
// one thread or threadgroup per source would walk it once.
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
    constant long       &total_edge_count       [[ buffer(12) ]],
    constant long       &rank_float4_stride     [[ buffer(13) ]],
    const device float  *coefficients           [[ buffer(14) ]],
    const device long   *edge_row_offset        [[ buffer(15) ]],
    device float        *output_weights         [[ buffer(16) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    long edge_ordinal = static_cast<long>(thread_id);
    if (edge_ordinal >= total_edge_count) return;

    // The source is the last row whose first edge ordinal is <= edge_ordinal.
    long low = 0;
    long high = node_count;
    while (low < high) {
        long middle = (low + high + 1) / 2;
        if (edge_row_offset[middle] <= edge_ordinal) low = middle;
        else high = middle - 1;
    }
    long source_node = low;
    long neighbor_slot = edge_ordinal - edge_row_offset[source_node];

    uint target_node = k2t_find_nth_neighbor(
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, static_cast<int>(node_count), padded_node_count,
        tree_height, internal_bit_count, static_cast<uint>(source_node), static_cast<int>(neighbor_slot)
    );
    if (target_node == K2T_NO_NEIGHBOR) {
        output_weights[edge_ordinal] = 0.0f;
        return;
    }

    const device float4 *u_row = U + source_node * rank_float4_stride;
    const device float4 *v_row = V + static_cast<long>(target_node) * rank_float4_stride;

    // Same terms, in the same order, as Weight_Matrix::reconstruct_entry on the host.
    float dot_product = 0.0f;
    for (long lane = 0; lane < rank_float4_stride; ++lane) {
        float4 lane_coefficients = float4(
            coefficients[lane * 4 + 0], coefficients[lane * 4 + 1],
            coefficients[lane * 4 + 2], coefficients[lane * 4 + 3]
        );
        dot_product += dot(u_row[lane], lane_coefficients * v_row[lane]);
    }
    output_weights[edge_ordinal] = dot_product;
}

// Unused: nothing loads the kernels below, so they are commented out (resilience review
// finding 49). The live one is neighbor_weights_kernel.
// // ── scale_uv ──────────────────────────────────────────────────────────────────
// // Scales U and V by scale_factor, one thread per float4.
// kernel void scale_uv_kernel(
//     device float4 *U                            [[ buffer(0) ]],
//     device float4 *V                            [[ buffer(1) ]],
//     constant long  &total_float4_element_count  [[ buffer(2) ]],
//     constant float &scale_factor                [[ buffer(3) ]],
//     uint thread_id [[ thread_position_in_grid ]]
// ) {
//     long index = static_cast<long>(thread_id);
//     if (index >= total_float4_element_count) return;
//     U[index] *= scale_factor;
//     V[index] *= scale_factor;
// }
//
// kernel void add_network_input_kernel(
//     device float      *membrane_potentials  [[ buffer(0) ]],
//     const device int  *input_neuron_indices [[ buffer(1) ]],
//     const device float *input_values        [[ buffer(2) ]],
//     constant long     &element_count        [[ buffer(3) ]],
//     uint thread_id [[ thread_position_in_grid ]]
// ) {
//     long index = static_cast<long>(thread_id);
//     if (index >= element_count) return;
//
//     int neuron_index = input_neuron_indices[index];
//     // Atomic because input_neuron_indices may repeat a neuron.
//     device atomic_float *target_slot = (device atomic_float *)(membrane_potentials + neuron_index);
//     atomic_fetch_add_explicit(target_slot, input_values[index], memory_order_relaxed);
// }
//
// // ── weight_update ─────────────────────────────────────────────────────────────
// // `iterations` rounds of proximal Hebbian update on one (source, target) pair's U and V rows.
// // One threadgroup, one thread per float4. The anchors are the rows' starting values; the
// // squared-norm denominators are recomputed each round (simd_sum, then thread 0 sums the groups).
// #define MAX_RANK_FLOAT4_STRIDE 64
// #define MAX_SIMD_GROUPS 32
//
// kernel void weight_update_kernel(
//     device float4 *U                   [[ buffer(0) ]],
//     device float4 *V                   [[ buffer(1) ]],
//     constant long  &rank_float4_stride [[ buffer(2) ]],
//     constant int   &source_node        [[ buffer(3) ]],
//     constant int   &target_node        [[ buffer(4) ]],
//     constant float &delta              [[ buffer(5) ]],
//     constant float &learning_rate      [[ buffer(6) ]],
//     constant float &l2_regularization  [[ buffer(7) ]],
//     constant int   &iterations         [[ buffer(8) ]],
//     uint lane                          [[ thread_position_in_threadgroup ]],
//     uint simd_lane                     [[ thread_index_in_simdgroup ]],
//     uint simd_group                    [[ simdgroup_index_in_threadgroup ]],
//     uint simd_groups_per_threadgroup   [[ simdgroups_per_threadgroup ]]
// ) {
//     threadgroup float4 anchor_u[MAX_RANK_FLOAT4_STRIDE];
//     threadgroup float4 anchor_v[MAX_RANK_FLOAT4_STRIDE];
//     threadgroup float  warp_partial_u[MAX_SIMD_GROUPS];
//     threadgroup float  warp_partial_v[MAX_SIMD_GROUPS];
//     threadgroup float  den_u;
//     threadgroup float  den_v;
//
//     bool active = static_cast<long>(lane) < rank_float4_stride;
//
//     device float4 *u_row = U + static_cast<long>(source_node) * rank_float4_stride;
//     device float4 *v_row = V + static_cast<long>(target_node) * rank_float4_stride;
//
//     float4 u4 = active ? u_row[lane] : float4(0.0f);
//     float4 v4 = active ? v_row[lane] : float4(0.0f);
//
//     if (active) {
//         anchor_u[lane] = u4;
//         anchor_v[lane] = v4;
//     }
//     threadgroup_barrier(mem_flags::mem_threadgroup);
//
//     for (int iteration = 0; iteration < iterations; ++iteration) {
//         float group_u = simd_sum(dot(u4, u4));
//         float group_v = simd_sum(dot(v4, v4));
//         if (simd_lane == 0) {
//             warp_partial_u[simd_group] = group_u;
//             warp_partial_v[simd_group] = group_v;
//         }
//         threadgroup_barrier(mem_flags::mem_threadgroup);
//
//         if (lane == 0) {
//             float sum_u = 0.0f, sum_v = 0.0f;
//             for (uint group = 0; group < simd_groups_per_threadgroup; ++group) {
//                 sum_u += warp_partial_u[group];
//                 sum_v += warp_partial_v[group];
//             }
//             den_u = l2_regularization + sum_u;
//             den_v = l2_regularization + sum_v;
//         }
//         threadgroup_barrier(mem_flags::mem_threadgroup);
//
//         if (active) {
//             float4 au = anchor_u[lane];
//             float4 av = anchor_v[lane];
//             float inv_den_u = 1.0f / den_u;
//             float inv_den_v = 1.0f / den_v;
//
//             float4 du = learning_rate * (delta * (v4 * inv_den_v) - l2_regularization * (u4 - au));
//             float4 dv = learning_rate * (delta * (u4 * inv_den_u) - l2_regularization * (v4 - av));
//
//             u4 += du;
//             v4 += dv;
//         }
//         threadgroup_barrier(mem_flags::mem_threadgroup);
//     }
//
//     if (active) {
//         u_row[lane] = u4;
//         v_row[lane] = v4;
//     }
// }

// Unused: nothing loads the kernels below, so they are commented out (resilience review
// finding 49). The live one is neighbor_weights_kernel.
// // ── k2tree_get_neighbors_batch ────────────────────────────────────────────────
// // Batched row enumeration: one thread per (query, neighbor_slot) pair writes that slot's
// // neighbor, or -1 past the end of the row.
// kernel void k2tree_get_neighbors_batch_kernel(
//     const device uint   *internal_node_words   [[ buffer(0) ]],
//     const device uint   *leaf_node_words       [[ buffer(1) ]],
//     const device uint   *rank_superblock_table [[ buffer(2) ]],
//     const device ushort *rank_subblock_table   [[ buffer(3) ]],
//     constant int &branching_factor             [[ buffer(4) ]],
//     constant int &superblock_size_words        [[ buffer(5) ]],
//     constant int &node_count                   [[ buffer(6) ]],
//     constant int &padded_node_count            [[ buffer(7) ]],
//     constant int &tree_height                  [[ buffer(8) ]],
//     constant int &internal_bit_count           [[ buffer(9) ]],
//     const device int  *source_node_indices     [[ buffer(10) ]],
//     constant int &query_count                  [[ buffer(11) ]],
//     constant int &max_neighbor_count           [[ buffer(12) ]],
//     device int        *output_buffer           [[ buffer(13) ]],
//     uint thread_id [[ thread_position_in_grid ]]
// ) {
//     long pair_index = static_cast<long>(thread_id);
//     long total_pairs = static_cast<long>(query_count) * static_cast<long>(max_neighbor_count);
//     if (pair_index >= total_pairs) return;
//
//     int query_index = static_cast<int>(pair_index / max_neighbor_count);
//     int neighbor_slot = static_cast<int>(pair_index % max_neighbor_count);
//     int source_node = source_node_indices[query_index];
//
//     output_buffer[pair_index] = k2t_find_nth_neighbor(
//         internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
//         branching_factor, superblock_size_words, node_count, padded_node_count,
//         tree_height, internal_bit_count, source_node, neighbor_slot
//     );
// }
//
//
// inline float apply_decay(float membrane_potential, float resting_potential, float decay_rate, int time_delta) {
//     if (time_delta <= 0) return membrane_potential;
//
//     if (time_delta == 1) return resting_potential + (membrane_potential - resting_potential) * (1.0f - decay_rate);
//
//     float decay = pow(1.0f - decay_rate, static_cast<float>(time_delta));
//     return resting_potential + (membrane_potential - resting_potential) * decay;
// }
//
// // ── step_apply_hebbian_update ─────────────────────────────────────────────────
// // Single-pass Hebbian nudge of the caller's U row and V[target_node]. The U row is the calling
// // thread's own accumulator; V[target_node] is shared by many threads (fan-in), so it is read and
// // updated atomically. l2_regularization only pads the denominators.
// inline void step_apply_hebbian_update(
//     thread float4 *u_row_accumulator,
//     device float4 *V,
//     long rank_float4_stride,
//     int target_node,
//     float delta,
//     float learning_rate,
//     float l2_regularization
// ) {
//     thread float4 anchor_u[MAX_RANK_FLOAT4_STRIDE];
//     thread float4 anchor_v[MAX_RANK_FLOAT4_STRIDE];
//
//     device float4 *v_row = V + static_cast<long>(target_node) * rank_float4_stride;
//
//     float sum_u = l2_regularization;
//     float sum_v = l2_regularization;
//     for (long lane = 0; lane < rank_float4_stride; ++lane) {
//         device atomic_float *v_components = (device atomic_float *)(v_row + lane);
//         float4 u4 = u_row_accumulator[lane];
//         float4 v4(
//             atomic_load_explicit(&v_components[0], memory_order_relaxed),
//             atomic_load_explicit(&v_components[1], memory_order_relaxed),
//             atomic_load_explicit(&v_components[2], memory_order_relaxed),
//             atomic_load_explicit(&v_components[3], memory_order_relaxed)
//         );
//         anchor_u[lane] = u4;
//         anchor_v[lane] = v4;
//         sum_u += dot(u4, u4);
//         sum_v += dot(v4, v4);
//     }
//     float inv_den_u = 1.0f / sum_u;
//     float inv_den_v = 1.0f / sum_v;
//
//     for (long lane = 0; lane < rank_float4_stride; ++lane) {
//         float4 du = learning_rate * delta * (anchor_v[lane] * inv_den_v);
//         float4 dv = learning_rate * delta * (anchor_u[lane] * inv_den_u);
//
//         u_row_accumulator[lane] += du;
//
//         device atomic_float *v_components = (device atomic_float *)(v_row + lane);
//         atomic_fetch_add_explicit(&v_components[0], dv.x, memory_order_relaxed);
//         atomic_fetch_add_explicit(&v_components[1], dv.y, memory_order_relaxed);
//         atomic_fetch_add_explicit(&v_components[2], dv.z, memory_order_relaxed);
//         atomic_fetch_add_explicit(&v_components[3], dv.w, memory_order_relaxed);
//     }
// }
//
//
// // ── step ──────────────────────────────────────────────────────────────────────
// // One tick of a fixed leaky integrate-and-fire model, one thread per entry of
// // active_neuron_indices. A spiking neuron walks its k^2-tree row once, scatters each edge's
// // weight into network_inputs and enqueues each target for the next tick.
// kernel void step(
//     constant long        &tick                       [[ buffer(0) ]],
//     constant long        &next_tick                  [[ buffer(1) ]],
//     constant int         &spike_period               [[ buffer(2) ]],
//     constant float       &spike_threshold            [[ buffer(3) ]],
//     constant float       &learning_rate              [[ buffer(4) ]],
//     constant float       &decay_rate                 [[ buffer(5) ]],
//     constant float       &resting_mp                 [[ buffer(6) ]],
//     device float4        *U                          [[ buffer(7) ]],
//     device float4        *V                          [[ buffer(8) ]],
//     constant long        &rank_float4_stride         [[ buffer(9) ]],
//     constant float       &constant_weight            [[ buffer(10) ]],
//     const device uint    *internal_node_words        [[ buffer(11) ]],
//     const device uint    *leaf_node_words            [[ buffer(12) ]],
//     const device uint    *rank_superblock_table      [[ buffer(13) ]],
//     const device ushort  *rank_subblock_table        [[ buffer(14) ]],
//     constant int         &branching_factor           [[ buffer(15) ]],
//     constant int         &superblock_size_words      [[ buffer(16) ]],
//     constant int         &padded_node_count          [[ buffer(17) ]],
//     constant int         &tree_height                [[ buffer(18) ]],
//     constant int         &internal_bit_count         [[ buffer(19) ]],
//     constant long        &neuron_count               [[ buffer(20) ]],
//     device float         *network_inputs             [[ buffer(21) ]],
//     device float         *membrane_potentials        [[ buffer(22) ]],
//     device long          *last_spiked                [[ buffer(23) ]],
//     device long          *last_tick_updated          [[ buffer(24) ]],
//     const device int     *active_neuron_indices      [[ buffer(25) ]],
//     const device int     *active_neuron_count        [[ buffer(26) ]],
//     device int           *next_active_neuron_indices [[ buffer(27) ]],
//     device int           *next_active_neuron_count   [[ buffer(28) ]],
//     device int           *active_generation          [[ buffer(29) ]],
//     constant int         &thread_count_per_block     [[ buffer(30) ]],
//     uint thread_id [[ thread_position_in_grid ]]
// ) {
//     int active_count = active_neuron_count[0];
//     int thread_index = static_cast<int>(thread_id);
//     if (thread_index >= active_count) return;
//
//     int neuron_thread_id = active_neuron_indices[thread_index];
//     if (neuron_thread_id < 0 || static_cast<long>(neuron_thread_id) >= neuron_count) return;
//
//     // Metal has no 64-bit atomic exchange, so active_generation holds next_tick narrowed to 32 bits.
//     int next_tick_i = static_cast<int>(next_tick);
//
//     long last_updated_tick = last_tick_updated[neuron_thread_id];
//     long time_since_last_update = tick - last_updated_tick;
//     float membrane_potential = membrane_potentials[neuron_thread_id];
//     membrane_potential = apply_decay(membrane_potential, resting_mp, decay_rate, static_cast<int>(time_since_last_update));
//     device atomic_float *self_input_slot = (device atomic_float *)(network_inputs + neuron_thread_id);
//     float accumulated_input = atomic_exchange_explicit(self_input_slot, 0.0f, memory_order_relaxed);
//     membrane_potential += accumulated_input;
//
//     long time_last_spiked = last_spiked[neuron_thread_id];
//     if ((tick - time_last_spiked) == spike_period) {
//         membrane_potentials[neuron_thread_id] = resting_mp;
//         last_tick_updated[neuron_thread_id] = tick;
//         return;
//     }
//
//     if (membrane_potential <= spike_threshold) {
//         membrane_potentials[neuron_thread_id] = membrane_potential;
//         last_tick_updated[neuron_thread_id] = tick;
//         return;
//     }
//
//     // Spike: record the tick unless refractory, then propagate along the neuron's k^2-tree row.
//     if ((tick - time_last_spiked) > spike_period) {
//         last_spiked[neuron_thread_id] = tick;
//     }
//
//     // Only this thread writes U[neuron_thread_id] this tick, so it is accumulated in registers
//     // and written back once.
//     thread float4 u_row_accumulator[MAX_RANK_FLOAT4_STRIDE];
//     device float4 *u_row_global = U + static_cast<long>(neuron_thread_id) * rank_float4_stride;
//     for (long lane = 0; lane < rank_float4_stride; ++lane) {
//         u_row_accumulator[lane] = u_row_global[lane];
//     }
//
//     // One DFS walk over the row: O(degree * tree height).
//     thread int walk_stack_row_base[MAX_K2TREE_HEIGHT];
//     thread int walk_stack_col_base[MAX_K2TREE_HEIGHT];
//     thread int walk_stack_block_size[MAX_K2TREE_HEIGHT];
//     thread int walk_stack_bit_offset[MAX_K2TREE_HEIGHT];
//     thread int walk_stack_next_col[MAX_K2TREE_HEIGHT];
//     walk_stack_row_base[0]   = 0;
//     walk_stack_col_base[0]   = 0;
//     walk_stack_block_size[0] = padded_node_count;
//     walk_stack_bit_offset[0] = 0;
//     walk_stack_next_col[0]   = 0;
//     int walk_stack_top = (tree_height > 0 && neuron_thread_id >= 0 &&
//                           neuron_thread_id < static_cast<int>(neuron_count)) ? 0 : -1;
//
//     int child;
//     while ((child = k2t_next_neighbor(
//         internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
//         branching_factor, superblock_size_words, static_cast<int>(neuron_count),
//         tree_height, internal_bit_count, neuron_thread_id,
//         walk_stack_row_base, walk_stack_col_base, walk_stack_block_size,
//         walk_stack_bit_offset, walk_stack_next_col, walk_stack_top
//     )) >= 0) {
//         // STDP update, skipping targets that never spiked (last_spiked < 0) or spiked this tick.
//         long child_last_spiked = last_spiked[child];
//         if (learning_rate != 0.0f && !(child_last_spiked < 0 || child_last_spiked == tick)) {
//             float tick_delta = static_cast<float>(abs(tick - child_last_spiked));
//             float decay_delta = -learning_rate * pow(tick_delta, -3.0f);
//             step_apply_hebbian_update(u_row_accumulator, V, rank_float4_stride, child,
//                                       decay_delta, 0.5f, 1.0f);
//         }
//
//         // The weight: constant_weight when nonzero, otherwise U[source] . V[target].
//         float weight = constant_weight;
//         if (constant_weight == 0.0f) {
//             const device float4 *v_row = V + static_cast<long>(child) * rank_float4_stride;
//             float dot_product = 0.0f;
//             for (long lane = 0; lane < rank_float4_stride; ++lane) {
//                 dot_product += dot(u_row_accumulator[lane], v_row[lane]);
//             }
//             weight = dot_product;
//         }
//
//         device atomic_float *input_slot = (device atomic_float *)(network_inputs + child);
//         atomic_fetch_add_explicit(input_slot, weight, memory_order_relaxed);
//
//         // Enqueue the target into the next tick's active set, once per generation.
//         device atomic_int *child_generation_slot = (device atomic_int *)(active_generation + child);
//         int previous_child_generation = atomic_exchange_explicit(child_generation_slot, next_tick_i, memory_order_relaxed);
//         if (previous_child_generation != next_tick_i) {
//             device atomic_int *next_count_slot = (device atomic_int *)next_active_neuron_count;
//             int position = atomic_fetch_add_explicit(next_count_slot, 1, memory_order_relaxed);
//             next_active_neuron_indices[position] = child;
//         }
//     }
//
//     for (long lane = 0; lane < rank_float4_stride; ++lane) {
//         u_row_global[lane] = u_row_accumulator[lane];
//     }
//
//     // Re-enqueue the spiking neuron itself: it may decay or spike again.
//     device atomic_int *self_generation_slot = (device atomic_int *)(active_generation + neuron_thread_id);
//     int previous_self_generation = atomic_exchange_explicit(self_generation_slot, next_tick_i, memory_order_relaxed);
//     if (previous_self_generation != next_tick_i) {
//         device atomic_int *next_count_slot = (device atomic_int *)next_active_neuron_count;
//         int position = atomic_fetch_add_explicit(next_count_slot, 1, memory_order_relaxed);
//         next_active_neuron_indices[position] = neuron_thread_id;
//     }
//
//     membrane_potentials[neuron_thread_id] = membrane_potential;
//     last_tick_updated[neuron_thread_id] = tick;
// }
//
// kernel void step_no_active_optimization(
//     constant long        &tick                       [[ buffer(0) ]],
//     constant long        &next_tick                  [[ buffer(1) ]],
//     constant int         &spike_period               [[ buffer(2) ]],
//     constant float       &spike_threshold            [[ buffer(3) ]],
//     constant float       &learning_rate              [[ buffer(4) ]],
//     constant float       &decay_rate                 [[ buffer(5) ]],
//     constant float       &resting_mp                 [[ buffer(6) ]],
//     device float4        *U                          [[ buffer(7) ]],
//     device float4        *V                          [[ buffer(8) ]],
//     constant long        &rank_float4_stride         [[ buffer(9) ]],
//     constant float       &constant_weight            [[ buffer(10) ]],
//     const device uint    *internal_node_words        [[ buffer(11) ]],
//     const device uint    *leaf_node_words            [[ buffer(12) ]],
//     const device uint    *rank_superblock_table      [[ buffer(13) ]],
//     const device ushort  *rank_subblock_table        [[ buffer(14) ]],
//     constant int         &branching_factor           [[ buffer(15) ]],
//     constant int         &superblock_size_words      [[ buffer(16) ]],
//     constant int         &padded_node_count          [[ buffer(17) ]],
//     constant int         &tree_height                [[ buffer(18) ]],
//     constant int         &internal_bit_count         [[ buffer(19) ]],
//     constant long        &neuron_count               [[ buffer(20) ]],
//     device float         *network_inputs             [[ buffer(21) ]],
//     device float         *membrane_potentials        [[ buffer(22) ]],
//     device long          *last_spiked                [[ buffer(23) ]],
//     device long          *last_tick_updated          [[ buffer(24) ]],
//     constant int         &thread_count_per_block     [[ buffer(25) ]],
//     uint thread_id [[ thread_position_in_grid ]]
// ) {
//     int neuron_thread_id = static_cast<int>(thread_id);
//
//     if (neuron_thread_id < 0 || static_cast<long>(neuron_thread_id) >= neuron_count) return;
//
//     // Metal has no 64-bit atomic exchange, so active_generation holds next_tick narrowed to 32 bits.
//     int next_tick_i = static_cast<int>(next_tick);
//
//     long last_updated_tick = last_tick_updated[neuron_thread_id];
//     long time_since_last_update = tick - last_updated_tick;
//     float membrane_potential = membrane_potentials[neuron_thread_id];
//     membrane_potential = apply_decay(membrane_potential, resting_mp, decay_rate, static_cast<int>(time_since_last_update));
//     device atomic_float *self_input_slot = (device atomic_float *)(network_inputs + neuron_thread_id);
//     float accumulated_input = atomic_exchange_explicit(self_input_slot, 0.0f, memory_order_relaxed);
//     membrane_potential += accumulated_input;
//
//     long time_last_spiked = last_spiked[neuron_thread_id];
//     if ((tick - time_last_spiked) == spike_period) {
//         membrane_potentials[neuron_thread_id] = resting_mp;
//         last_tick_updated[neuron_thread_id] = tick;
//         return;
//     }
//
//     if (membrane_potential <= spike_threshold) {
//         membrane_potentials[neuron_thread_id] = membrane_potential;
//         last_tick_updated[neuron_thread_id] = tick;
//         return;
//     }
//
//     // Spike: record the tick unless refractory, then propagate along the neuron's k^2-tree row.
//     if ((tick - time_last_spiked) > spike_period) {
//         last_spiked[neuron_thread_id] = tick;
//     }
//
//     // Only this thread writes U[neuron_thread_id] this tick, so it is accumulated in registers
//     // and written back once.
//     thread float4 u_row_accumulator[MAX_RANK_FLOAT4_STRIDE];
//     device float4 *u_row_global = U + static_cast<long>(neuron_thread_id) * rank_float4_stride;
//     for (long lane = 0; lane < rank_float4_stride; ++lane) {
//         u_row_accumulator[lane] = u_row_global[lane];
//     }
//
//     // One DFS walk over the row: O(degree * tree height).
//     thread int walk_stack_row_base[MAX_K2TREE_HEIGHT];
//     thread int walk_stack_col_base[MAX_K2TREE_HEIGHT];
//     thread int walk_stack_block_size[MAX_K2TREE_HEIGHT];
//     thread int walk_stack_bit_offset[MAX_K2TREE_HEIGHT];
//     thread int walk_stack_next_col[MAX_K2TREE_HEIGHT];
//     walk_stack_row_base[0]   = 0;
//     walk_stack_col_base[0]   = 0;
//     walk_stack_block_size[0] = padded_node_count;
//     walk_stack_bit_offset[0] = 0;
//     walk_stack_next_col[0]   = 0;
//     int walk_stack_top = (tree_height > 0 && neuron_thread_id >= 0 &&
//                           neuron_thread_id < static_cast<int>(neuron_count)) ? 0 : -1;
//
//     int child;
//     while ((child = k2t_next_neighbor(
//         internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
//         branching_factor, superblock_size_words, static_cast<int>(neuron_count),
//         tree_height, internal_bit_count, neuron_thread_id,
//         walk_stack_row_base, walk_stack_col_base, walk_stack_block_size,
//         walk_stack_bit_offset, walk_stack_next_col, walk_stack_top
//     )) >= 0) {
//         // STDP update, skipping targets that never spiked (last_spiked < 0) or spiked this tick.
//         long child_last_spiked = last_spiked[child];
//         if (learning_rate != 0.0f && !(child_last_spiked < 0 || child_last_spiked == tick)) {
//             float tick_delta = static_cast<float>(abs(tick - child_last_spiked));
//             float decay_delta = -learning_rate * pow(tick_delta, -3.0f);
//             step_apply_hebbian_update(u_row_accumulator, V, rank_float4_stride, child,
//                                       decay_delta, 0.5f, 1.0f);
//         }
//
//         // The weight: constant_weight when nonzero, otherwise U[source] . V[target].
//         float weight = constant_weight;
//         if (constant_weight == 0.0f) {
//             const device float4 *v_row = V + static_cast<long>(child) * rank_float4_stride;
//             float dot_product = 0.0f;
//             for (long lane = 0; lane < rank_float4_stride; ++lane) {
//                 dot_product += dot(u_row_accumulator[lane], v_row[lane]);
//             }
//             weight = dot_product;
//         }
//
//         device atomic_float *input_slot = (device atomic_float *)(network_inputs + child);
//         atomic_fetch_add_explicit(input_slot, weight, memory_order_relaxed);
//     }
//
//     for (long lane = 0; lane < rank_float4_stride; ++lane) {
//         u_row_global[lane] = u_row_accumulator[lane];
//     }
//
//     membrane_potentials[neuron_thread_id] = membrane_potential;
//     last_tick_updated[neuron_thread_id] = tick;
// }
//
// kernel void decay_all_neurons_kernel(
//     device float   *membrane_potentials [[ buffer(0) ]],
//     device long    *last_tick_updated   [[ buffer(1) ]],
//     constant long  &neuron_count        [[ buffer(2) ]],
//     constant long  &tick                [[ buffer(3) ]],
//     constant float &resting_mp          [[ buffer(4) ]],
//     constant float &decay_rate          [[ buffer(5) ]],
//     uint thread_id [[ thread_position_in_grid ]]
// ) {
//     long neuron_index = static_cast<long>(thread_id);
//     if (neuron_index >= neuron_count) return;
//
//     long time_delta = tick - last_tick_updated[neuron_index];
//     membrane_potentials[neuron_index] = apply_decay(
//         membrane_potentials[neuron_index], resting_mp, decay_rate, static_cast<int>(time_delta));
//     last_tick_updated[neuron_index] = tick;
// }
//
// kernel void merge_input_neurons_kernel(
//     device int        *active_neuron_indices  [[ buffer(0) ]],
//     device int        *active_neuron_count    [[ buffer(1) ]],
//     const device long *override_input_neurons [[ buffer(2) ]],
//     constant long     &override_count         [[ buffer(3) ]],
//     uint thread_id [[ thread_position_in_grid ]]
// ) {
//     long override_index = static_cast<long>(thread_id);
//     if (override_index >= override_count) return;
//
//     long candidate = override_input_neurons[override_index];
//     if (candidate < 0) return;
//     int neuron = static_cast<int>(candidate);
//
//     int current_count = active_neuron_count[0];
//     for (int scan_index = 0; scan_index < current_count; ++scan_index) {
//         if (active_neuron_indices[scan_index] == neuron) return;
//     }
//
//     device atomic_int *count_slot = (device atomic_int *)active_neuron_count;
//     int position = atomic_fetch_add_explicit(count_slot, 1, memory_order_relaxed);
//     active_neuron_indices[position] = neuron;
// }
//
// kernel void reservoir_features_kernel(
//     constant long     &neuron_count       [[ buffer(0) ]],
//     constant long     &tick               [[ buffer(1) ]],
//     constant float    &spike_tau          [[ buffer(2) ]],
//     constant float    &voltage_scale      [[ buffer(3) ]],
//     device float      *membrane_potentials [[ buffer(4) ]],
//     const device long *last_spiked         [[ buffer(5) ]],
//     device long       *last_tick_updated   [[ buffer(6) ]],
//     constant float    &resting_mp          [[ buffer(7) ]],
//     constant float    &decay_rate          [[ buffer(8) ]],
//     device float      *output_buffer       [[ buffer(9) ]],
//     uint thread_id [[ thread_position_in_grid ]]
// ) {
//     long neuron_index = static_cast<long>(thread_id);
//     if (neuron_index >= neuron_count) return;
//
//     long time_delta = tick - last_tick_updated[neuron_index];
//     float membrane_potential = apply_decay(
//         membrane_potentials[neuron_index], resting_mp, decay_rate, static_cast<int>(time_delta));
//     membrane_potentials[neuron_index] = membrane_potential;
//     last_tick_updated[neuron_index] = tick;
//
//     // last_spiked < 0 means never spiked (NEVER_SPIKED_TICK); tick 0 is a real spike tick.
//     long since_spike = tick - last_spiked[neuron_index];
//     float trace = (last_spiked[neuron_index] >= 0) ? exp(-static_cast<float>(since_spike) / spike_tau) : 0.0f;
//
//     output_buffer[neuron_index] = trace;
//     output_buffer[neuron_count + neuron_index] = (membrane_potential - resting_mp) / voltage_scale;
//
//     if (neuron_index == 0) {
//         output_buffer[2 * neuron_count] = 1.0f;
//     }
// }
