#include "spikecorec/nml/dynamics.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

// The predefined kernel code, parsed by create_kernel_root. Codegen inserts each
// population's dynamics into master_step right after the spiked declaration, and each
// synapse's dynamics into the propagate walk right after the arrived declaration.
//
// Metal's k^2-tree helpers (k2t_next_neighbor and the rest, plus MAX_K2TREE_HEIGHT) come
// from k2tree_device.metalinc, which create_kernel_root parses in ahead of the first
// function here. The CUDA source carries its own port of them.
//
// Bodies of if, else, for and while must be braced: the parser requires it.

const char *METAL_KERNEL_BOILERPLATE = R"METAL(
#include <metal_stdlib>
using namespace metal;

inline float spikecorec_heaviside(float value) {
    return value > 0.0f ? 1.0f : 0.0f;
}

// One edge's stored value, reconstructed from the shared basis: U[source] . (Ck * V[target]).
inline float spikecorec_reconstruct_edge(
    const device float4 *basis_u, const device float4 *basis_v, const device float *coefficients,
    int rank_float4_stride, int source_node, int target_node
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

// Where one edge's update sits in plane's segment of the delta arrays, or -1. Rows
// are CSR over source nodes, sorted by edge ordinal, so this is a binary search over the row.
inline int spikecorec_edge_delta_index(
    const device int *row_start, const device long *entry_ordinal,
    int node_count, long delta_capacity, int plane, int source_node, long edge_ordinal
) {
    const int row_base = plane * (node_count + 1);
    const device long *plane_ordinal = entry_ordinal + (long)plane * delta_capacity;
    int low = row_start[row_base + source_node];
    int high = row_start[row_base + source_node + 1];

    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (plane_ordinal[middle] < edge_ordinal) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (low < row_start[row_base + source_node + 1] && plane_ordinal[low] == edge_ordinal) {
        return low;
    }
    return -1;
}

// One edge's value in plane: the basis reconstruction plus the edge's update in S.
inline float spikecorec_load_edge(
    const device float4 *basis_u, const device float4 *basis_v, const device float *edge_coefficients,
    int rank_float4_stride, const device int *row_start, const device long *entry_ordinal,
    const device float *entry_value, int node_count, long delta_capacity,
    int plane, int source_node, int target_node, long edge_ordinal
) {
    float value = spikecorec_reconstruct_edge(
        basis_u, basis_v, edge_coefficients + (long)plane * rank_float4_stride * 4, rank_float4_stride,
        source_node, target_node);
    const int delta_index = spikecorec_edge_delta_index(
        row_start, entry_ordinal, node_count, delta_capacity, plane, source_node, edge_ordinal);
    if (delta_index >= 0) {
        value += entry_value[(long)plane * delta_capacity + delta_index];
    }
    return value;
}

// Adds delta to one edge's value in plane. An edge that already holds an update there is
// written in place: only its source neuron's thread walks it, so nothing races. Otherwise the
// update is queued with its plane and becomes readable once the host merges the queue. A
// delta written back to exactly zero queues an empty update, so the merge removes it.
inline void spikecorec_accumulate_edge(
    const device int *row_start, const device long *entry_ordinal, device float *entry_value,
    device long *pending_ordinal, device float *pending_value, device int *pending_matrix_index,
    device atomic_int *pending_count, long pending_capacity, int node_count, long delta_capacity,
    int plane, int source_node, long edge_ordinal, float delta
) {
    if (delta == 0.0f) {
        return;
    }
    const int delta_index = spikecorec_edge_delta_index(
        row_start, entry_ordinal, node_count, delta_capacity, plane, source_node, edge_ordinal);
    if (delta_index >= 0) {
        const long slot = (long)plane * delta_capacity + delta_index;
        entry_value[slot] += delta;
        if (entry_value[slot] != 0.0f) {
            return;
        }
        delta = 0.0f;
    }
    const int pending = atomic_fetch_add_explicit(pending_count, 1, memory_order_relaxed);
    if (pending < pending_capacity) {
        pending_ordinal[pending] = edge_ordinal;
        pending_value[pending] = delta;
        pending_matrix_index[pending] = plane;
    }
}

// The synapse prototype of one edge: the prototype of the last projection run that starts at
// or before it.
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
    constant long         &sparse_delta_capacity         [[ buffer(5)  ]],
    constant long         &pending_delta_capacity        [[ buffer(6)  ]],
    constant int          &projection_run_count          [[ buffer(7)  ]],
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
    const device long     *sparse_delta_edge_ordinal     [[ buffer(26) ]],
    device   float        *sparse_delta_value            [[ buffer(27) ]],
    device   long         *pending_delta_edge_ordinal    [[ buffer(28) ]],
    device   float        *pending_delta_value           [[ buffer(29) ]],
    device   int          *pending_delta_matrix_index    [[ buffer(30) ]],
    device   atomic_int   *pending_delta_count           [[ buffer(31) ]],
    const device long     *projection_first_edge_ordinal [[ buffer(32) ]],
    const device int      *projection_synapse_prototype  [[ buffer(33) ]],
    const device float    *random_values                 [[ buffer(34) ]],
    device   uint         *event_arrival_count           [[ buffer(35) ]],
    const device long     *input_row_start               [[ buffer(36) ]],
    const device int      *input_entry_prototype         [[ buffer(37) ]],
    device   float        *input_values                  [[ buffer(38) ]],
    const device uchar    *input_spike_counts            [[ buffer(39) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    const long neuron_index = (long)thread_id;
    if (neuron_index >= neuron_count) {
        return;
    }

    // Stage 1, deliver. network_inputs is two rows: this tick's arrivals are drained from
    // one while this tick's scatters land in the other.
    const int current_row = (int)(tick % 2);
    const int next_row = 1 - current_row;
    const long input_slot = current_row * neuron_count + neuron_index;
    // The cell's inputs add to this before its dynamics read it.
    float network_input = network_inputs[input_slot];
    network_inputs[input_slot] = 0.0f;

    // Stages 2-5: the population dispatch goes here.
    bool spiked = false;

    if (spiked) {
        last_spiked[neuron_index] = tick;
    }
    spike_history[(tick % spike_history_length) * neuron_count + neuron_index] = spiked ? 1 : 0;

    // Stage 6, propagate: walk this neuron's outgoing edges. Its synapses can only be away
    // from rest from its spike until the longest delay has passed and they have settled
    // after the arrival; outside that window every one is at rest, so the walk is skipped.
    // A negative synapse_active_ticks walks every tick.
    if (synapse_active_ticks >= 0 &&
        (last_spiked[neuron_index] < 0 || tick - last_spiked[neuron_index] > synapse_active_ticks)) {
        return;
    }

    // Each edge runs its synapse and scatters its current into the target's next input row.
    thread int walk_stack_row_base[MAX_K2TREE_HEIGHT];
    thread int walk_stack_column_base[MAX_K2TREE_HEIGHT];
    thread int walk_stack_block_size[MAX_K2TREE_HEIGHT];
    thread int walk_stack_bit_offset[MAX_K2TREE_HEIGHT];
    thread int walk_stack_next_column[MAX_K2TREE_HEIGHT];
    walk_stack_row_base[0] = 0;
    walk_stack_column_base[0] = 0;
    walk_stack_block_size[0] = padded_node_count;
    walk_stack_bit_offset[0] = 0;
    walk_stack_next_column[0] = 0;
    int walk_stack_top = tree_height > 0 ? 0 : -1;

    // edge_row_offset is only read once the walk has found an edge: a model with no
    // connections binds a placeholder there.
    int slot = 0;
    while (true) {
        const int target = k2t_next_neighbor(
            internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
            branching_factor, superblock_size_words, neuron_count, tree_height, internal_bit_count,
            (int)neuron_index, walk_stack_row_base, walk_stack_column_base, walk_stack_block_size,
            walk_stack_bit_offset, walk_stack_next_column, walk_stack_top);
        if (target < 0) {
            break;
        }
        const long current_edge = edge_row_offset[neuron_index] + slot;
        if (current_edge >= edge_row_offset[neuron_index + 1]) {
            break;
        }
        slot += 1;

        int delay_ticks = (int)round(spikecorec_load_edge(
            basis_u, basis_v, edge_coefficients, rank_float4_stride, sparse_delta_row_start,
            sparse_delta_edge_ordinal, sparse_delta_value, neuron_count, sparse_delta_capacity,
            1, (int)neuron_index, target, current_edge));
        if (delay_ticks < 1) {
            delay_ticks = 1;
        }
        const bool arrived = tick >= delay_ticks &&
            spike_history[((tick - delay_ticks) % spike_history_length) * neuron_count + neuron_index] != 0;

        // The synapse dispatch goes here.
    }
}
)METAL";

const char *CUDA_KERNEL_BOILERPLATE = R"CUDA(
#define MAX_K2TREE_HEIGHT 32

__device__ inline float spikecorec_heaviside(float value) {
    return value > 0.0f ? 1.0f : 0.0f;
}

// k^2-tree bit-walk helpers, ported from k2tree_device.metalinc.
__device__ inline unsigned int k2t_get_bit(const unsigned int *words, int bit_index) {
    return (words[bit_index >> 5] >> (unsigned int)(bit_index & 31)) & 1u;
}

__device__ inline int k2t_rank1_exclusive(
    const unsigned int *internal_words,
    const unsigned int *superblock_table,
    const unsigned short *subblock_table,
    int position,
    int superblock_size
) {
    const int word_index = position >> 5;
    const int bit_offset = position & 31;
    const int superblock_index = word_index / superblock_size;
    const unsigned int superblock_base = superblock_table[superblock_index];
    const unsigned int subblock_base = (unsigned int)subblock_table[word_index];
    const unsigned int partial_word_mask = (bit_offset == 0) ? 0u : ((1u << (unsigned int)bit_offset) - 1u);
    const unsigned int partial_word_popcount = __popc(internal_words[word_index] & partial_word_mask);
    return (int)(superblock_base + subblock_base + partial_word_popcount);
}

// Resumes a DFS over row_node's neighbors from the caller's stack state, returning the next
// neighbor each call, or -1 when exhausted.
__device__ inline int k2t_next_neighbor(
    const unsigned int *internal_node_words,
    const unsigned int *leaf_node_words,
    const unsigned int *rank_superblock_table,
    const unsigned short *rank_subblock_table,
    int branching_factor,
    int superblock_size_words,
    int node_count,
    int tree_height,
    int internal_bit_count,
    int row_node,
    int *stack_row_base,
    int *stack_column_base,
    int *stack_block_size,
    int *stack_bit_offset,
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

        const int row_base = stack_row_base[level];
        const int column_base = stack_column_base[level];
        const int block_size = stack_block_size[level];
        const int level_bit_offset = stack_bit_offset[level];

        const int child_block_size = block_size / branching_factor;
        const int row_offset = (row_node - row_base) / child_block_size;
        const int child_flat_index = row_offset * branching_factor + column_offset;
        const int bit_position = level_bit_offset + child_flat_index;

        if (level == tree_height - 1) {
            if (k2t_get_bit(leaf_node_words, bit_position)) {
                const int neighbor = column_base + column_offset;
                if (neighbor < node_count) {
                    return neighbor;
                }
            }
        } else if (k2t_get_bit(internal_node_words, bit_position)) {
            const int rank_inclusive = k2t_rank1_exclusive(internal_node_words, rank_superblock_table,
                                                           rank_subblock_table, bit_position, superblock_size_words) + 1;
            const int child_level = stack_top + 1;
            const int raw_offset = branching_factor_squared * rank_inclusive;
            stack_row_base[child_level] = row_base + row_offset * child_block_size;
            stack_column_base[child_level] = column_base + column_offset * child_block_size;
            stack_block_size[child_level] = child_block_size;
            stack_bit_offset[child_level] = (child_level == tree_height - 1) ? (raw_offset - internal_bit_count) : raw_offset;
            stack_next_column[child_level] = 0;
            stack_top = child_level;
        }
    }
    return -1;
}

// One edge's stored value, reconstructed from the shared basis: U[source] . (Ck * V[target]).
__device__ inline float spikecorec_reconstruct_edge(
    const float4 *basis_u, const float4 *basis_v, const float *coefficients,
    int rank_float4_stride, int source_node, int target_node
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

// Where one edge's update sits in plane's segment of the delta arrays, or -1. Rows
// are CSR over source nodes, sorted by edge ordinal, so this is a binary search over the row.
__device__ inline int spikecorec_edge_delta_index(
    const int *row_start, const long long *entry_ordinal,
    int node_count, long long delta_capacity, int plane, int source_node, long long edge_ordinal
) {
    const int row_base = plane * (node_count + 1);
    const long long *plane_ordinal = entry_ordinal + (long long)plane * delta_capacity;
    int low = row_start[row_base + source_node];
    int high = row_start[row_base + source_node + 1];

    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (plane_ordinal[middle] < edge_ordinal) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (low < row_start[row_base + source_node + 1] && plane_ordinal[low] == edge_ordinal) {
        return low;
    }
    return -1;
}

// One edge's value in plane: the basis reconstruction plus the edge's update in S.
__device__ inline float spikecorec_load_edge(
    const float4 *basis_u, const float4 *basis_v, const float *edge_coefficients,
    int rank_float4_stride, const int *row_start, const long long *entry_ordinal,
    const float *entry_value, int node_count, long long delta_capacity,
    int plane, int source_node, int target_node, long long edge_ordinal
) {
    float value = spikecorec_reconstruct_edge(
        basis_u, basis_v, edge_coefficients + (long long)plane * rank_float4_stride * 4, rank_float4_stride,
        source_node, target_node);
    const int delta_index = spikecorec_edge_delta_index(
        row_start, entry_ordinal, node_count, delta_capacity, plane, source_node, edge_ordinal);
    if (delta_index >= 0) {
        value += entry_value[(long long)plane * delta_capacity + delta_index];
    }
    return value;
}

// Adds delta to one edge's value in plane. An edge that already holds an update there is
// written in place: only its source neuron's thread walks it, so nothing races. Otherwise the
// update is queued with its plane and becomes readable once the host merges the queue. A
// delta written back to exactly zero queues an empty update, so the merge removes it.
__device__ inline void spikecorec_accumulate_edge(
    const int *row_start, const long long *entry_ordinal, float *entry_value,
    long long *pending_ordinal, float *pending_value, int *pending_matrix_index,
    int *pending_count, long long pending_capacity, int node_count, long long delta_capacity,
    int plane, int source_node, long long edge_ordinal, float delta
) {
    if (delta == 0.0f) {
        return;
    }
    const int delta_index = spikecorec_edge_delta_index(
        row_start, entry_ordinal, node_count, delta_capacity, plane, source_node, edge_ordinal);
    if (delta_index >= 0) {
        const long long slot = (long long)plane * delta_capacity + delta_index;
        entry_value[slot] += delta;
        if (entry_value[slot] != 0.0f) {
            return;
        }
        delta = 0.0f;
    }
    const int pending = atomicAdd(pending_count, 1);
    if (pending < pending_capacity) {
        pending_ordinal[pending] = edge_ordinal;
        pending_value[pending] = delta;
        pending_matrix_index[pending] = plane;
    }
}

// The synapse prototype of one edge: the prototype of the last projection run that starts at
// or before it.
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
    long long sparse_delta_capacity,
    long long pending_delta_capacity,
    int projection_run_count,
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
    const long long *sparse_delta_edge_ordinal,
    float *sparse_delta_value,
    long long *pending_delta_edge_ordinal,
    float *pending_delta_value,
    int *pending_delta_matrix_index,
    int *pending_delta_count,
    const long long *projection_first_edge_ordinal,
    const int *projection_synapse_prototype,
    const float *random_values,
    unsigned int *event_arrival_count,
    const long long *input_row_start,
    const int *input_entry_prototype,
    float *input_values,
    const unsigned char *input_spike_counts
) {
    const long long neuron_index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_index >= neuron_count) {
        return;
    }

    // Stage 1, deliver. network_inputs is two rows: this tick's arrivals are drained from
    // one while this tick's scatters land in the other.
    const int current_row = (int)(tick % 2);
    const int next_row = 1 - current_row;
    const long long input_slot = current_row * neuron_count + neuron_index;
    // The cell's inputs add to this before its dynamics read it.
    float network_input = network_inputs[input_slot];
    network_inputs[input_slot] = 0.0f;

    // Stages 2-5: the population dispatch goes here.
    bool spiked = false;

    if (spiked) {
        last_spiked[neuron_index] = tick;
    }
    spike_history[(tick % spike_history_length) * neuron_count + neuron_index] = spiked ? 1 : 0;

    // Stage 6, propagate: walk this neuron's outgoing edges. Its synapses can only be away
    // from rest from its spike until the longest delay has passed and they have settled
    // after the arrival; outside that window every one is at rest, so the walk is skipped.
    // A negative synapse_active_ticks walks every tick.
    if (synapse_active_ticks >= 0 &&
        (last_spiked[neuron_index] < 0 || tick - last_spiked[neuron_index] > synapse_active_ticks)) {
        return;
    }

    // Each edge runs its synapse and scatters its current into the target's next input row.
    int walk_stack_row_base[MAX_K2TREE_HEIGHT];
    int walk_stack_column_base[MAX_K2TREE_HEIGHT];
    int walk_stack_block_size[MAX_K2TREE_HEIGHT];
    int walk_stack_bit_offset[MAX_K2TREE_HEIGHT];
    int walk_stack_next_column[MAX_K2TREE_HEIGHT];
    walk_stack_row_base[0] = 0;
    walk_stack_column_base[0] = 0;
    walk_stack_block_size[0] = padded_node_count;
    walk_stack_bit_offset[0] = 0;
    walk_stack_next_column[0] = 0;
    int walk_stack_top = tree_height > 0 ? 0 : -1;

    // edge_row_offset is only read once the walk has found an edge: a model with no
    // connections binds a placeholder there.
    int slot = 0;
    while (true) {
        const int target = k2t_next_neighbor(
            internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
            branching_factor, superblock_size_words, neuron_count, tree_height, internal_bit_count,
            (int)neuron_index, walk_stack_row_base, walk_stack_column_base, walk_stack_block_size,
            walk_stack_bit_offset, walk_stack_next_column, walk_stack_top);
        if (target < 0) {
            break;
        }
        const long long current_edge = edge_row_offset[neuron_index] + slot;
        if (current_edge >= edge_row_offset[neuron_index + 1]) {
            break;
        }
        slot += 1;

        int delay_ticks = (int)roundf(spikecorec_load_edge(
            basis_u, basis_v, edge_coefficients, rank_float4_stride, sparse_delta_row_start,
            sparse_delta_edge_ordinal, sparse_delta_value, neuron_count, sparse_delta_capacity,
            1, (int)neuron_index, target, current_edge));
        if (delay_ticks < 1) {
            delay_ticks = 1;
        }
        const bool arrived = tick >= delay_ticks &&
            spike_history[((tick - delay_ticks) % spike_history_length) * neuron_count + neuron_index] != 0;

        // The synapse dispatch goes here.
    }
}
)CUDA";

}
