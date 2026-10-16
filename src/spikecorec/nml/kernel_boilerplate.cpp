#include "dynamics.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

// The fixed kernel source that create_kernel_root parses. Codegen inserts the population
// dispatch after master_step's `spiked` declaration and the synapse dispatch after the
// propagate walk's `arrived` declaration. Metal takes its k^2-tree helpers from
// k2tree_device.metalinc; the CUDA source carries its own port.
// Bodies of if, else, for and while must be braced: the parser requires it.

const char *METAL_KERNEL_BOILERPLATE = R"METAL(
#include <metal_stdlib>
using namespace metal;

inline float spikecorec_heaviside(float value) {
    return value > 0.0f ? 1.0f : 0.0f;
}

// One edge's basis value, without its update in S: U[source] . (Ck * V[target]).
inline float spikecorec_reconstruct_edge(
    const device float4 *basis_u, const device float4 *basis_v, const device float *coefficients,
    int rank_float4_stride, uint source_node, uint target_node
) {
    const device float4 *u_row = basis_u + (long)source_node * rank_float4_stride;
    const device float4 *v_row = basis_v + (long)target_node * rank_float4_stride;

    float dot_product = 0.0f;
    for (int lane = 0; lane < rank_float4_stride; lane += 1) {
        const float4 lane_coefficients = float4(
            coefficients[lane * 4 + 0], coefficients[lane * 4 + 1],
            coefficients[lane * 4 + 2], coefficients[lane * 4 + 3]
        );
        dot_product += dot(u_row[lane], lane_coefficients * v_row[lane]);
    }
    return dot_product;
}

// Position of one edge's entry in S's pool, or -1: a binary search of the source's CSR row in plane.
inline int spikecorec_edge_delta_index(
    const device int *row_start, const device uint *entry_ordinal,
    int node_count, int plane, uint source_node, long edge_ordinal
) {
    const int row_base = plane * (node_count + 1);
    int low = row_start[row_base + source_node];
    int high = row_start[row_base + source_node + 1];

    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (entry_ordinal[middle] < edge_ordinal) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (low < row_start[row_base + source_node + 1] && entry_ordinal[low] == edge_ordinal) {
        return low;
    }
    return -1;
}

// One edge's value in plane: the basis reconstruction plus the edge's update in S.
inline float spikecorec_load_edge(
    const device float4 *basis_u, const device float4 *basis_v, const device float *edge_coefficients,
    int rank_float4_stride, const device int *row_start, const device uint *entry_ordinal,
    const device float *entry_value, int node_count,
    int plane, uint source_node, uint target_node, long edge_ordinal
) {
    float value = spikecorec_reconstruct_edge(
        basis_u, basis_v, edge_coefficients + (long)plane * rank_float4_stride * 4, rank_float4_stride,
        source_node, target_node);
    const int delta_index = spikecorec_edge_delta_index(
        row_start, entry_ordinal, node_count, plane, source_node, edge_ordinal);
    if (delta_index >= 0) {
        value += entry_value[delta_index];
    }
    return value;
}

// Adds delta to one edge's value in plane. An entry already in S is updated in place (only the
// source's thread walks the edge, so no race), even when it sums to zero; otherwise the update is
// queued for the host merge, in room the walk reserved. So only edges with no S entry queue.
inline void spikecorec_accumulate_edge(
    const device int *row_start, const device uint *entry_ordinal, device float *entry_value,
    device uint *pending_ordinal, device float *pending_value, device uchar *pending_matrix_index,
    device atomic_int *pending_count, long pending_capacity, int node_count,
    int plane, uint source_node, long edge_ordinal, float delta
) {
    if (delta == 0.0f) {
        return;
    }
    const int delta_index = spikecorec_edge_delta_index(
        row_start, entry_ordinal, node_count, plane, source_node, edge_ordinal);
    if (delta_index >= 0) {
        entry_value[delta_index] += delta;
        return;
    }
    const int pending = atomic_fetch_add_explicit(pending_count, 1, memory_order_relaxed);
    if (pending < pending_capacity) {
        pending_ordinal[pending] = edge_ordinal;
        pending_value[pending] = delta;
        pending_matrix_index[pending] = plane;
    }
}

// An edge's synapse prototype: that of the last run starting at or before it.
inline int spikecorec_synapse_prototype(
    const device long *run_first_edge_ordinal, const device int *run_synapse_prototype,
    int run_count, long edge_ordinal
) {
    int low = 0;
    int high = run_count;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (run_first_edge_ordinal[middle] <= edge_ordinal) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low > 0 ? run_synapse_prototype[low - 1] : -1;
}

kernel void master_step(
    constant long         &tick                          [[ buffer(0)  ]],
    constant int          &neuron_count                  [[ buffer(1)  ]],
    constant int          &spike_history_length          [[ buffer(2)  ]],
    constant long         &synapse_active_ticks          [[ buffer(3)  ]],
    constant int          &rank_float4_stride            [[ buffer(4)  ]],
    constant long         &pending_delta_capacity        [[ buffer(6)  ]],
    constant int          &updated_plane_count           [[ buffer(6)  ]],
    constant int          &run_count          [[ buffer(7)  ]],
    constant int          &branching_factor              [[ buffer(8)  ]],
    constant int          &superblock_size_words         [[ buffer(9)  ]],
    constant int          &padded_node_count             [[ buffer(10) ]],
    constant int          &tree_height                   [[ buffer(11) ]],
    constant int          &internal_bit_count            [[ buffer(12) ]],
    device   float        *cell_state                    [[ buffer(13) ]],
    device   float        *network_inputs                [[ buffer(14) ]],
    device   uchar        *spike_history                 [[ buffer(15) ]],
    device   long         *last_spiked                   [[ buffer(16) ]],
    const device uint     *internal_node_words           [[ buffer(17) ]],
    const device uint     *leaf_node_words               [[ buffer(18) ]],
    const device uint     *rank_superblock_table         [[ buffer(19) ]],
    const device ushort   *rank_subblock_table           [[ buffer(20) ]],
    const device float4   *basis_u                       [[ buffer(21) ]],
    const device float4   *basis_v                       [[ buffer(22) ]],
    const device float    *edge_coefficients             [[ buffer(23) ]],
    const device long     *edge_row_offset               [[ buffer(24) ]],
    const device int      *sparse_delta_row_start        [[ buffer(25) ]],
    const device uint     *sparse_delta_edge_ordinal     [[ buffer(26) ]],
    device   float        *sparse_delta_value            [[ buffer(27) ]],
    device   uint         *pending_delta_edge_ordinal    [[ buffer(28) ]],
    device   float        *pending_delta_value           [[ buffer(29) ]],
    device   uchar        *pending_delta_matrix_index    [[ buffer(30) ]],
    device   atomic_int   *pending_delta_control         [[ buffer(31) ]],
    const device long     *run_first_edge_ordinal [[ buffer(32) ]],
    const device int      *run_synapse_prototype  [[ buffer(33) ]],
    const device float    *random_values                 [[ buffer(34) ]],
    device   uint         *event_arrival_count           [[ buffer(35) ]],
    const device long     *input_row_start               [[ buffer(36) ]],
    const device int      *input_entry_prototype         [[ buffer(37) ]],
    device   float        *input_values                  [[ buffer(38) ]],
    const device uchar    *input_spike_counts            [[ buffer(39) ]],
    constant float        &simulation_time               [[ buffer(40) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    // pending_delta_control: [0] updates queued, [1] queue room reserved, [2] this tick's walk pass,
    // [3] neurons this pass walks, [4] neurons it defers, then two lists of deferred neurons of
    // neuron_count each. Pass 0 runs every stage for every neuron; a later pass of the same tick
    // walks only the neurons the pass before deferred (see the reservation in stage 6), once the
    // host has merged the queue into S.
    const int walk_pass = atomic_load_explicit(pending_delta_control + 2, memory_order_relaxed);
    long neuron_index = (long)thread_id;
    if (walk_pass > 0) {
        if ((int)thread_id >= atomic_load_explicit(pending_delta_control + 3, memory_order_relaxed)) {
            return;
        }
        neuron_index = (long)atomic_load_explicit(
            pending_delta_control + 5 + ((walk_pass - 1) % 2) * neuron_count + (int)thread_id, memory_order_relaxed);
    }
    if (neuron_index >= neuron_count) {
        return;
    }

    // Stage 1, deliver: drain this tick's row of network_inputs; scatters go to the other row.
    const int current_row = (int)(tick % 2);
    const int next_row = 1 - current_row;
    const long input_slot = current_row * neuron_count + neuron_index;
    // The cell's inputs add to this before its dynamics read it.
    float network_input = network_inputs[input_slot];
    network_inputs[input_slot] = 0.0f;

    // Stages 2-5: codegen inserts the population dispatch after this declaration.
    bool spiked = false;

    if (spiked) {
        last_spiked[neuron_index] = tick;
    }
    if (walk_pass == 0) {
        spike_history[(tick % spike_history_length) * neuron_count + neuron_index] = spiked ? 1 : 0;
    }

    // Stage 6, propagate: walk this neuron's outgoing edges. Its synapses are at rest once
    // synapse_active_ticks have passed since its last spike, so the walk is skipped then.
    // A negative synapse_active_ticks walks every tick.
    if (synapse_active_ticks >= 0 &&
        (last_spiked[neuron_index] < 0 || tick - last_spiked[neuron_index] > synapse_active_ticks)) {
        return;
    }

    // Reserve queue room for every update the walk can queue: one per updated plane (the planes
    // from 2) for each edge with no entry in that plane of S, since an entry in S changes in place.
    // A walk the queue cannot take now is deferred to the next pass.
    if (updated_plane_count > 0) {
        const int out_degree = (int)(edge_row_offset[neuron_index + 1] - edge_row_offset[neuron_index]);
        int needed = 0;
        for (int plane = 2; plane < 2 + updated_plane_count; plane += 1) {
            const int row_base = plane * (neuron_count + 1) + (int)neuron_index;
            needed += out_degree - (sparse_delta_row_start[row_base + 1] - sparse_delta_row_start[row_base]);
        }
        if (needed > 0) {
            const int reserved = atomic_fetch_add_explicit(pending_delta_control + 1, needed, memory_order_relaxed);
            if (reserved + needed > pending_delta_capacity) {
                atomic_fetch_sub_explicit(pending_delta_control + 1, needed, memory_order_relaxed);
                const int deferred_slot = atomic_fetch_add_explicit(pending_delta_control + 4, 1, memory_order_relaxed);
                atomic_store_explicit(pending_delta_control + 5 + (walk_pass % 2) * neuron_count + deferred_slot,
                                      (int)neuron_index, memory_order_relaxed);
                return;
            }
        }
    }

    // Each edge runs its synapse and scatters its current into the target's next input row.
    thread uint walk_stack_row_base[MAX_K2TREE_HEIGHT];
    thread uint walk_stack_column_base[MAX_K2TREE_HEIGHT];
    thread int walk_stack_block_size[MAX_K2TREE_HEIGHT];
    thread uint walk_stack_bit_offset[MAX_K2TREE_HEIGHT];
    thread int walk_stack_next_column[MAX_K2TREE_HEIGHT];
    walk_stack_row_base[0] = 0;
    walk_stack_column_base[0] = 0;
    walk_stack_block_size[0] = padded_node_count;
    walk_stack_bit_offset[0] = 0;
    walk_stack_next_column[0] = 0;
    int walk_stack_top = tree_height > 0 ? 0 : -1;

    // Read edge_row_offset only after finding an edge: with no connections it is a placeholder.
    int slot = 0;
    while (true) {
        const uint target = k2t_next_neighbor(
            internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
            branching_factor, superblock_size_words, neuron_count, tree_height, internal_bit_count,
            (uint)neuron_index, walk_stack_row_base, walk_stack_column_base, walk_stack_block_size,
            walk_stack_bit_offset, walk_stack_next_column, walk_stack_top);
        if (target == K2T_NO_NEIGHBOR) {
            break;
        }
        const long current_edge = edge_row_offset[neuron_index] + slot;
        if (current_edge >= edge_row_offset[neuron_index + 1]) {
            break;
        }
        slot += 1;

        int delay_ticks = (int)round(spikecorec_load_edge(
            basis_u, basis_v, edge_coefficients, rank_float4_stride, sparse_delta_row_start,
            sparse_delta_edge_ordinal, sparse_delta_value, neuron_count,
            1, (uint)neuron_index, target, current_edge));
        if (delay_ticks < 1) {
            delay_ticks = 1;
        }
        // A fitted delay can pass the longest declared one by the fit error; past the ring it
        // would wrap onto a row written this tick or lately.
        if (delay_ticks > spike_history_length - 1) {
            delay_ticks = spike_history_length - 1;
        }
        const bool arrived = tick >= delay_ticks &&
            spike_history[((tick - delay_ticks) % spike_history_length) * neuron_count + neuron_index] != 0;

        // Codegen inserts the synapse dispatch after the `arrived` declaration.
    }
}
)METAL";

const char *CUDA_KERNEL_BOILERPLATE = R"CUDA(
#define MAX_K2TREE_HEIGHT 32

// Node indices are unsigned, so a walk that finds no node returns this (UINT_MAX) instead of -1.
#define K2T_NO_NEIGHBOR 4294967295u

__device__ inline float spikecorec_heaviside(float value) {
    return value > 0.0f ? 1.0f : 0.0f;
}

// CUDA port of the k^2-tree helpers in k2tree_device.metalinc. Bit positions are unsigned, so a
// tree may hold up to 2^32 bits.
__device__ inline unsigned int k2t_get_bit(const unsigned int *words, unsigned int bit_index) {
    return (words[bit_index >> 5] >> (bit_index & 31u)) & 1u;
}

__device__ inline unsigned int k2t_rank1_exclusive(
    const unsigned int *internal_words,
    const unsigned int *superblock_table,
    const unsigned short *subblock_table,
    unsigned int position,
    int superblock_size
) {
    const unsigned int word_index = position >> 5;
    const unsigned int bit_offset = position & 31u;
    const unsigned int superblock_index = word_index / (unsigned int)superblock_size;
    const unsigned int superblock_base = superblock_table[superblock_index];
    const unsigned int subblock_base = (unsigned int)subblock_table[word_index];
    const unsigned int partial_word_mask = (bit_offset == 0u) ? 0u : ((1u << bit_offset) - 1u);
    const unsigned int partial_word_popcount = __popc(internal_words[word_index] & partial_word_mask);
    return superblock_base + subblock_base + partial_word_popcount;
}

// Returns row_node's next neighbor, resuming the DFS from the caller's stack, or K2T_NO_NEIGHBOR
// when done.
__device__ inline unsigned int k2t_next_neighbor(
    const unsigned int *internal_node_words,
    const unsigned int *leaf_node_words,
    const unsigned int *rank_superblock_table,
    const unsigned short *rank_subblock_table,
    int branching_factor,
    int superblock_size_words,
    int node_count,
    int tree_height,
    int internal_bit_count,
    unsigned int row_node,
    unsigned int *stack_row_base,
    unsigned int *stack_column_base,
    int *stack_block_size,
    unsigned int *stack_bit_offset,
    int *stack_next_column,
    int &stack_top
) {
    const int branching_factor_squared = branching_factor * branching_factor;

    while (stack_top >= 0) {
        const int level = stack_top;
        const int column_offset = stack_next_column[level];
        if (column_offset >= branching_factor) {
            stack_top -= 1;
            continue;
        }
        stack_next_column[level] = column_offset + 1;

        const unsigned int row_base = stack_row_base[level];
        const unsigned int column_base = stack_column_base[level];
        const int block_size = stack_block_size[level];
        const unsigned int level_bit_offset = stack_bit_offset[level];

        const int child_block_size = block_size / branching_factor;
        const int row_offset = (int)((row_node - row_base) / (unsigned int)child_block_size);
        const int child_flat_index = row_offset * branching_factor + column_offset;
        const unsigned int bit_position = level_bit_offset + (unsigned int)child_flat_index;

        if (level == tree_height - 1) {
            if (k2t_get_bit(leaf_node_words, bit_position)) {
                const unsigned int neighbor = column_base + (unsigned int)column_offset;
                if (neighbor < (unsigned int)node_count) {
                    return neighbor;
                }
            }
        } else if (k2t_get_bit(internal_node_words, bit_position)) {
            const unsigned int rank_inclusive = k2t_rank1_exclusive(internal_node_words, rank_superblock_table,
                                                                    rank_subblock_table, bit_position, superblock_size_words) + 1u;
            const int child_level = stack_top + 1;
            const unsigned int raw_offset = (unsigned int)branching_factor_squared * rank_inclusive;
            stack_row_base[child_level] = row_base + (unsigned int)(row_offset * child_block_size);
            stack_column_base[child_level] = column_base + (unsigned int)(column_offset * child_block_size);
            stack_block_size[child_level] = child_block_size;
            stack_bit_offset[child_level] = (child_level == tree_height - 1) ? (raw_offset - (unsigned int)internal_bit_count) : raw_offset;
            stack_next_column[child_level] = 0;
            stack_top = child_level;
        }
    }
    return K2T_NO_NEIGHBOR;
}

// One edge's basis value, without its update in S: U[source] . (Ck * V[target]).
__device__ inline float spikecorec_reconstruct_edge(
    const float4 *basis_u, const float4 *basis_v, const float *coefficients,
    int rank_float4_stride, unsigned int source_node, unsigned int target_node
) {
    const float4 *u_row = basis_u + (long long)source_node * rank_float4_stride;
    const float4 *v_row = basis_v + (long long)target_node * rank_float4_stride;

    float dot_product = 0.0f;
    for (int lane = 0; lane < rank_float4_stride; lane += 1) {
        const float4 u_lane = u_row[lane];
        const float4 v_lane = v_row[lane];
        dot_product += u_lane.x * (coefficients[lane * 4 + 0] * v_lane.x) +
                       u_lane.y * (coefficients[lane * 4 + 1] * v_lane.y) +
                       u_lane.z * (coefficients[lane * 4 + 2] * v_lane.z) +
                       u_lane.w * (coefficients[lane * 4 + 3] * v_lane.w);
    }
    return dot_product;
}

// Position of one edge's entry in S's pool, or -1: a binary search of the source's CSR row in plane.
__device__ inline int spikecorec_edge_delta_index(
    const int *row_start, const unsigned int *entry_ordinal,
    int node_count, int plane, unsigned int source_node, long long edge_ordinal
) {
    const int row_base = plane * (node_count + 1);
    int low = row_start[row_base + source_node];
    int high = row_start[row_base + source_node + 1];

    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (entry_ordinal[middle] < edge_ordinal) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (low < row_start[row_base + source_node + 1] && entry_ordinal[low] == edge_ordinal) {
        return low;
    }
    return -1;
}

// One edge's value in plane: the basis reconstruction plus the edge's update in S.
__device__ inline float spikecorec_load_edge(
    const float4 *basis_u, const float4 *basis_v, const float *edge_coefficients,
    int rank_float4_stride, const int *row_start, const unsigned int *entry_ordinal,
    const float *entry_value, int node_count,
    int plane, unsigned int source_node, unsigned int target_node, long long edge_ordinal
) {
    float value = spikecorec_reconstruct_edge(
        basis_u, basis_v, edge_coefficients + (long long)plane * rank_float4_stride * 4, rank_float4_stride,
        source_node, target_node);
    const int delta_index = spikecorec_edge_delta_index(
        row_start, entry_ordinal, node_count, plane, source_node, edge_ordinal);
    if (delta_index >= 0) {
        value += entry_value[delta_index];
    }
    return value;
}

// Adds delta to one edge's value in plane. An entry already in S is updated in place (only the
// source's thread walks the edge, so no race), even when it sums to zero; otherwise the update is
// queued for the host merge, in room the walk reserved. So only edges with no S entry queue.
__device__ inline void spikecorec_accumulate_edge(
    const int *row_start, const unsigned int *entry_ordinal, float *entry_value,
    unsigned int *pending_ordinal, float *pending_value, unsigned char *pending_matrix_index,
    int *pending_count, long long pending_capacity, int node_count,
    int plane, unsigned int source_node, long long edge_ordinal, float delta
) {
    if (delta == 0.0f) {
        return;
    }
    const int delta_index = spikecorec_edge_delta_index(
        row_start, entry_ordinal, node_count, plane, source_node, edge_ordinal);
    if (delta_index >= 0) {
        entry_value[delta_index] += delta;
        return;
    }
    const int pending = atomicAdd(pending_count, 1);
    if (pending < pending_capacity) {
        pending_ordinal[pending] = edge_ordinal;
        pending_value[pending] = delta;
        pending_matrix_index[pending] = plane;
    }
}

// An edge's synapse prototype: that of the last run starting at or before it.
__device__ inline int spikecorec_synapse_prototype(
    const long long *run_first_edge_ordinal, const int *run_synapse_prototype,
    int run_count, long long edge_ordinal
) {
    int low = 0;
    int high = run_count;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (run_first_edge_ordinal[middle] <= edge_ordinal) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low > 0 ? run_synapse_prototype[low - 1] : -1;
}

extern "C" __global__ void master_step(
    long long tick,
    int neuron_count,
    int spike_history_length,
    long long synapse_active_ticks,
    int rank_float4_stride,
    long long pending_delta_capacity,
    int updated_plane_count,
    int run_count,
    int branching_factor,
    int superblock_size_words,
    int padded_node_count,
    int tree_height,
    int internal_bit_count,
    float *cell_state,
    float *network_inputs,
    unsigned char *spike_history,
    long long *last_spiked,
    const unsigned int *internal_node_words,
    const unsigned int *leaf_node_words,
    const unsigned int *rank_superblock_table,
    const unsigned short *rank_subblock_table,
    const float4 *basis_u,
    const float4 *basis_v,
    const float *edge_coefficients,
    const long long *edge_row_offset,
    const int *sparse_delta_row_start,
    const unsigned int *sparse_delta_edge_ordinal,
    float *sparse_delta_value,
    unsigned int *pending_delta_edge_ordinal,
    float *pending_delta_value,
    unsigned char *pending_delta_matrix_index,
    int *pending_delta_control,
    const long long *run_first_edge_ordinal,
    const int *run_synapse_prototype,
    const float *random_values,
    unsigned int *event_arrival_count,
    const long long *input_row_start,
    const int *input_entry_prototype,
    float *input_values,
    const unsigned char *input_spike_counts,
    double simulation_time
) {
    // pending_delta_control: laid out as in the Metal kernel.
    const int walk_pass = pending_delta_control[2];
    const long long thread_index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long neuron_index = thread_index;
    if (walk_pass > 0) {
        if (thread_index >= pending_delta_control[3]) {
            return;
        }
        neuron_index = pending_delta_control[5 + ((walk_pass - 1) % 2) * neuron_count + (int)thread_index];
    }
    if (neuron_index >= neuron_count) {
        return;
    }

    // Stage 1, deliver: drain this tick's row of network_inputs; scatters go to the other row.
    const int current_row = (int)(tick % 2);
    const int next_row = 1 - current_row;
    const long long input_slot = current_row * neuron_count + neuron_index;
    // The cell's inputs add to this before its dynamics read it.
    float network_input = network_inputs[input_slot];
    network_inputs[input_slot] = 0.0f;

    // Stages 2-5: codegen inserts the population dispatch after this declaration.
    bool spiked = false;

    if (spiked) {
        last_spiked[neuron_index] = tick;
    }
    if (walk_pass == 0) {
        spike_history[(tick % spike_history_length) * neuron_count + neuron_index] = spiked ? 1 : 0;
    }

    // Stage 6, propagate: walk this neuron's outgoing edges. Its synapses are at rest once
    // synapse_active_ticks have passed since its last spike, so the walk is skipped then.
    // A negative synapse_active_ticks walks every tick.
    if (synapse_active_ticks >= 0 &&
        (last_spiked[neuron_index] < 0 || tick - last_spiked[neuron_index] > synapse_active_ticks)) {
        return;
    }

    // Reserve queue room as the Metal kernel does.
    if (updated_plane_count > 0) {
        const int out_degree = (int)(edge_row_offset[neuron_index + 1] - edge_row_offset[neuron_index]);
        int needed = 0;
        for (int plane = 2; plane < 2 + updated_plane_count; plane += 1) {
            const int row_base = plane * (neuron_count + 1) + (int)neuron_index;
            needed += out_degree - (sparse_delta_row_start[row_base + 1] - sparse_delta_row_start[row_base]);
        }
        if (needed > 0) {
            const int reserved = atomicAdd(pending_delta_control + 1, needed);
            if (reserved + needed > pending_delta_capacity) {
                atomicSub(pending_delta_control + 1, needed);
                const int deferred_slot = atomicAdd(pending_delta_control + 4, 1);
                pending_delta_control[5 + (walk_pass % 2) * neuron_count + deferred_slot] = (int)neuron_index;
                return;
            }
        }
    }

    // Each edge runs its synapse and scatters its current into the target's next input row.
    unsigned int walk_stack_row_base[MAX_K2TREE_HEIGHT];
    unsigned int walk_stack_column_base[MAX_K2TREE_HEIGHT];
    int walk_stack_block_size[MAX_K2TREE_HEIGHT];
    unsigned int walk_stack_bit_offset[MAX_K2TREE_HEIGHT];
    int walk_stack_next_column[MAX_K2TREE_HEIGHT];
    walk_stack_row_base[0] = 0;
    walk_stack_column_base[0] = 0;
    walk_stack_block_size[0] = padded_node_count;
    walk_stack_bit_offset[0] = 0;
    walk_stack_next_column[0] = 0;
    int walk_stack_top = tree_height > 0 ? 0 : -1;

    // Read edge_row_offset only after finding an edge: with no connections it is a placeholder.
    int slot = 0;
    while (true) {
        const unsigned int target = k2t_next_neighbor(
            internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
            branching_factor, superblock_size_words, neuron_count, tree_height, internal_bit_count,
            (unsigned int)neuron_index, walk_stack_row_base, walk_stack_column_base, walk_stack_block_size,
            walk_stack_bit_offset, walk_stack_next_column, walk_stack_top);
        if (target == K2T_NO_NEIGHBOR) {
            break;
        }
        const long long current_edge = edge_row_offset[neuron_index] + slot;
        if (current_edge >= edge_row_offset[neuron_index + 1]) {
            break;
        }
        slot += 1;

        int delay_ticks = (int)roundf(spikecorec_load_edge(
            basis_u, basis_v, edge_coefficients, rank_float4_stride, sparse_delta_row_start,
            sparse_delta_edge_ordinal, sparse_delta_value, neuron_count,
            1, (unsigned int)neuron_index, target, current_edge));
        if (delay_ticks < 1) {
            delay_ticks = 1;
        }
        // A fitted delay can pass the longest declared one by the fit error; past the ring it
        // would wrap onto a row written this tick or lately.
        if (delay_ticks > spike_history_length - 1) {
            delay_ticks = spike_history_length - 1;
        }
        const bool arrived = tick >= delay_ticks &&
            spike_history[((tick - delay_ticks) % spike_history_length) * neuron_count + neuron_index] != 0;

        // Codegen inserts the synapse dispatch after the `arrived` declaration.
    }
}
)CUDA";

}
