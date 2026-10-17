// CUDA versions of the kernels in metal/weight_fit_kernels.metal, for Weight_Matrix's two
// candidate fits (masked alternating least squares and limited-memory BFGS). Each takes the
// arguments its Metal version takes, in the same order, and computes the same values (up to f32
// rounding: sums are taken in another order). See weight_fit_kernels.metal for what each kernel
// does; the comments here cover only what differs on CUDA.
//
// - An argument Metal reads from the constant address space is a pointer (see
//   precompiled_kernels.h); a kernel reads it once, at the start.
// - A Metal SIMD group is a CUDA warp: 32 threads, the width the kernels assume
//   (FIT_WALK_THREADS_PER_ROW). simd_sum, simd_max, simd_shuffle and simd_any are warp shuffles
//   and votes over the full warp, simdgroup_barrier is __syncwarp, and threadgroup_barrier is
//   __syncthreads. The host launches whole warps: Engine_Backend loads these kernels only when its
//   threads_per_block is a multiple of 32.
// - Metal runs exactly the threads the host asks for, in a last threadgroup that may be partial;
//   CUDA runs whole blocks of threads_per_block. The grid-striding kernels stride by the threads
//   CUDA runs, so every index is still visited once, and a thread past the end adds nothing to a
//   threadgroup's partial. The host reads one partial per block, which is the number of Metal
//   threadgroups for the same thread count.

#include <cstring>

#include "k2tree_device.cuh"
#include "precompiled_kernels.h"

#define FIT_WALK_THREADS_PER_ROW 32u
#define FULL_WARP_MASK 0xffffffffu

// The scalars a walking kernel reads: Weight_Matrix::Fit_Walk_Constants on the host, byte for byte.
// Outside the unnamed namespace, since the kernels take it.
struct Fit_Walk_Constants {
    unsigned int node_count;
    unsigned int branching_factor;
    unsigned int superblock_size_words;
    unsigned int padded_node_count;
    unsigned int tree_height;
    unsigned int internal_bit_count;
    unsigned int basis_lane_count;
    unsigned int target_lane_count;
    float target_lane_sign;
    float inverse_scale;
    unsigned int lane_count;
    float ridge;
    unsigned int walkers_per_row;
    unsigned int connections_listed;
};
static_assert(sizeof(Fit_Walk_Constants) == 14 * 4, "Fit_Walk_Constants must match the host's layout");

namespace {

__device__ inline unsigned int grid_thread_index() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline unsigned int grid_thread_count() {
    return gridDim.x * blockDim.x;
}

__device__ inline unsigned int warp_lane() {
    return threadIdx.x & 31u;
}

// NaN or infinite, by bit pattern.
__device__ inline bool is_not_finite(float value) {
    return (__float_as_uint(value) & 0x7fffffffu) >= 0x7f800000u;
}

__device__ inline float positive_infinity() {
    return __int_as_float(0x7f800000);
}

// An edge's weight in the objective: 1 / max(|target|, 1)^2; a non-finite target gets none.
__device__ inline float edge_weight_of(float target) {
    if (is_not_finite(target)) {
        return 0.0f;
    }
    const float error_scale = fmaxf(fabsf(target), 1.0f);
    return 1.0f / (error_scale * error_scale);
}

// The target the lanes are fitted to: a non-finite one counts as zero, with no weight.
__device__ inline float held_target(float target) {
    return is_not_finite(target) ? 0.0f : target;
}

__device__ inline float warp_sum(float value) {
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1) {
        value += __shfl_xor_sync(FULL_WARP_MASK, value, offset);
    }
    return value;
}

__device__ inline float warp_maximum(float value) {
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(FULL_WARP_MASK, value, offset));
    }
    return value;
}

// The block's sum of one value per thread; valid in the block's first thread. Every thread of the
// block must call it.
__device__ inline float block_sum(float value, float *scratch) {
    const unsigned int lane = warp_lane();
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warp_count = (blockDim.x + 31u) >> 5;
    value = warp_sum(value);
    if (lane == 0u) {
        scratch[warp] = value;
    }
    __syncthreads();
    float total = 0.0f;
    if (warp == 0u) {
        total = (lane < warp_count) ? scratch[lane] : 0.0f;
        total = warp_sum(total);
    }
    __syncthreads();
    return total;
}

// The block's largest value (values are never negative); valid in the first thread.
__device__ inline float block_maximum(float value, float *scratch) {
    const unsigned int lane = warp_lane();
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int warp_count = (blockDim.x + 31u) >> 5;
    value = warp_maximum(value);
    if (lane == 0u) {
        scratch[warp] = value;
    }
    __syncthreads();
    float largest = 0.0f;
    if (warp == 0u) {
        largest = (lane < warp_count) ? scratch[lane] : 0.0f;
        largest = warp_maximum(largest);
    }
    __syncthreads();
    return largest;
}

// ── walking the plane's connections ───────────────────────────────────────────
struct Fit_Walk_Buffers {
    const unsigned int *internal_node_words;
    const unsigned int *leaf_node_words;
    const unsigned int *rank_superblock_table;
    const unsigned short *rank_subblock_table;
    const float *basis_u;
    const float *basis_v;
    const float *coefficients;
    const unsigned int *target_lanes;
    const int *value_row_start;
    const unsigned int *value_target;
    const float *value;
    const int *incoming_offset;
    const unsigned int *incoming_entry;
};

struct Fit_Walk {
    unsigned int row_base[MAX_K2TREE_HEIGHT];
    unsigned int column_base[MAX_K2TREE_HEIGHT];
    unsigned int block_size[MAX_K2TREE_HEIGHT];
    unsigned int bit_offset[MAX_K2TREE_HEIGHT];
    unsigned int next_child[MAX_K2TREE_HEIGHT];
    int top;
    unsigned int node;
    bool along_column;
    unsigned int first;
    unsigned int end;
    bool listed;
    int position;
    int listed_end;
    int entry;
};

__device__ inline Fit_Walk fit_walk_begin(const Fit_Walk_Buffers &buffers, const Fit_Walk_Constants &constants,
                                          unsigned int node, bool along_column, unsigned int walker, bool valid) {
    Fit_Walk walk;
    walk.listed = constants.connections_listed != 0u;
    walk.position = 0;
    walk.listed_end = 0;
    walk.entry = 0;
    if (walk.listed) {
        walk.node = node;
        walk.along_column = along_column;
        walk.top = -1;
        if (valid) {
            const int first = along_column ? buffers.incoming_offset[node] : buffers.value_row_start[node];
            const int end = along_column ? buffers.incoming_offset[node + 1] : buffers.value_row_start[node + 1];
            const int share = (end - first + static_cast<int>(constants.walkers_per_row) - 1) /
                              static_cast<int>(constants.walkers_per_row);
            walk.position = min(first + static_cast<int>(walker) * share, end);
            walk.listed_end = min(walk.position + share, end);
        }
        return walk;
    }
    const unsigned int share = (constants.node_count + constants.walkers_per_row - 1u) / constants.walkers_per_row;
    walk.node = node;
    walk.along_column = along_column;
    walk.first = min(walker * share, constants.node_count);
    walk.end = min(walk.first + share, constants.node_count);
    walk.row_base[0] = 0;
    walk.column_base[0] = 0;
    walk.block_size[0] = constants.padded_node_count;
    walk.bit_offset[0] = 0;
    walk.next_child[0] = 0;
    walk.top = (valid && constants.tree_height > 0u && walk.first < walk.end) ? 0 : -1;
    return walk;
}

__device__ inline unsigned int fit_walk_next(Fit_Walk &walk, const Fit_Walk_Buffers &buffers,
                                             const Fit_Walk_Constants &constants) {
    if (walk.listed) {
        if (walk.position >= walk.listed_end) {
            return K2T_NO_NEIGHBOR;
        }
        if (!walk.along_column) {
            walk.entry = walk.position;
            walk.position += 1;
            return buffers.value_target[walk.entry];
        }
        walk.entry = static_cast<int>(buffers.incoming_entry[walk.position]);
        walk.position += 1;
        // The entry's source: the row whose entries hold it.
        unsigned int low = 0;
        unsigned int high = constants.node_count;
        while (low + 1u < high) {
            const unsigned int middle = low + (high - low) / 2u;
            if (buffers.value_row_start[middle] <= walk.entry) {
                low = middle;
            } else {
                high = middle;
            }
        }
        return low;
    }
    const unsigned int branching_factor = constants.branching_factor;
    const int leaf_level = static_cast<int>(constants.tree_height) - 1;
    while (walk.top >= 0) {
        const int level = walk.top;
        const unsigned int child = walk.next_child[level];
        if (child >= branching_factor) {
            walk.top -= 1;
            continue;
        }
        walk.next_child[level] = child + 1u;

        const unsigned int child_block_size = walk.block_size[level] / branching_factor;
        // The child's first node along the walked end, and the node's place on the fixed end.
        const unsigned int walked_base = walk.along_column ? walk.row_base[level] : walk.column_base[level];
        const unsigned int fixed_base = walk.along_column ? walk.column_base[level] : walk.row_base[level];
        const unsigned int child_first = walked_base + child * child_block_size;
        if (child_first >= walk.end) {
            walk.top -= 1;
            continue;
        }
        if (child_first + child_block_size <= walk.first) {
            continue;
        }
        const unsigned int fixed_offset = (walk.node - fixed_base) / child_block_size;
        const unsigned int row_offset = walk.along_column ? child : fixed_offset;
        const unsigned int column_offset = walk.along_column ? fixed_offset : child;
        const unsigned int bit_position = walk.bit_offset[level] + row_offset * branching_factor + column_offset;

        if (level == leaf_level) {
            if (k2t_get_bit(buffers.leaf_node_words, bit_position) && child_first < constants.node_count) {
                return child_first;
            }
        } else if (k2t_get_bit(buffers.internal_node_words, bit_position)) {
            const unsigned int rank_inclusive =
                k2t_rank1_exclusive(buffers.internal_node_words, buffers.rank_superblock_table,
                                    buffers.rank_subblock_table, bit_position,
                                    static_cast<int>(constants.superblock_size_words)) + 1u;
            const unsigned int raw_offset = branching_factor * branching_factor * rank_inclusive;
            const int child_level = level + 1;
            walk.row_base[child_level] = walk.row_base[level] + row_offset * child_block_size;
            walk.column_base[child_level] = walk.column_base[level] + column_offset * child_block_size;
            walk.block_size[child_level] = child_block_size;
            walk.bit_offset[child_level] = (child_level == leaf_level) ? (raw_offset - constants.internal_bit_count) : raw_offset;
            walk.next_child[child_level] = 0;
            walk.top = child_level;
        }
    }
    return K2T_NO_NEIGHBOR;
}

// Where a row's values for targets from first_target on start: the first entry not below it.
__device__ inline int fit_value_position(const Fit_Walk_Buffers &buffers, unsigned int source, unsigned int first_target) {
    int low = buffers.value_row_start[source];
    int high = buffers.value_row_start[source + 1];
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (buffers.value_target[middle] < first_target) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

// The fit's value for (source, target), or zero when its values hold none.
__device__ inline float fit_value_at(const Fit_Walk_Buffers &buffers, unsigned int source, unsigned int target) {
    const int position = fit_value_position(buffers, source, target);
    return (position < buffers.value_row_start[source + 1] && buffers.value_target[position] == target)
               ? buffers.value[position]
               : 0.0f;
}

struct Fit_Value_Cursor {
    int position;
    int end;
};

__device__ inline Fit_Value_Cursor fit_value_cursor(const Fit_Walk_Buffers &buffers, const Fit_Walk &walk) {
    Fit_Value_Cursor cursor;
    const bool reads_row = !walk.listed && !walk.along_column && walk.top >= 0;
    cursor.position = reads_row ? fit_value_position(buffers, walk.node, walk.first) : 0;
    cursor.end = reads_row ? buffers.value_row_start[walk.node + 1] : 0;
    return cursor;
}

// The fit's value for the walk's connection to other_end.
__device__ inline float fit_value_of(Fit_Value_Cursor &cursor, const Fit_Walk_Buffers &buffers,
                                     const Fit_Walk &walk, unsigned int other_end) {
    if (walk.listed) {
        return buffers.value[walk.entry];
    }
    if (walk.along_column) {
        return fit_value_at(buffers, other_end, walk.node);
    }
    while (cursor.position < cursor.end && buffers.value_target[cursor.position] < other_end) {
        cursor.position += 1;
    }
    return (cursor.position < cursor.end && buffers.value_target[cursor.position] == other_end)
               ? buffers.value[cursor.position]
               : 0.0f;
}

// What the own lanes must hold at (source, target), given the fit's value there.
__device__ inline float fit_target(const Fit_Walk_Buffers &buffers, const Fit_Walk_Constants &constants,
                                   unsigned int source, unsigned int target, float value) {
    float basis_part = 0.0f;
    const float *source_row = buffers.basis_u + static_cast<unsigned long long>(source) * constants.basis_lane_count;
    const float *target_row = buffers.basis_v + static_cast<unsigned long long>(target) * constants.basis_lane_count;
    for (unsigned int index = 0; index < constants.target_lane_count; index += 1) {
        const unsigned int lane = buffers.target_lanes[index];
        basis_part += source_row[lane] * buffers.coefficients[lane] * target_row[lane];
    }
    return (constants.target_lane_sign * basis_part + value) * constants.inverse_scale;
}

// The own lanes' value at (source, target).
__device__ inline float fit_working_value(const float *working_u, const float *working_v, unsigned int lane_count,
                                          unsigned int source, unsigned int target) {
    const float *source_row = working_u + static_cast<unsigned long long>(source) * lane_count;
    const float *target_row = working_v + static_cast<unsigned long long>(target) * lane_count;
    float value = 0.0f;
    for (unsigned int lane = 0; lane < lane_count; lane += 1) {
        value += source_row[lane] * target_row[lane];
    }
    return value;
}

} // namespace

// ── fit_walk_residual ─────────────────────────────────────────────────────────
extern "C" __global__ void fit_walk_residual_kernel(
    const unsigned int *internal_node_words,
    const unsigned int *leaf_node_words,
    const unsigned int *rank_superblock_table,
    const unsigned short *rank_subblock_table,
    const float *basis_u,
    const float *basis_v,
    const float *coefficients,
    const unsigned int *target_lanes,
    const int *value_row_start,
    const unsigned int *value_target,
    const float *value,
    const Fit_Walk_Constants *constants_argument,
    const int *incoming_offset,
    const unsigned int *incoming_entry,
    const float *working_u,
    const float *working_v,
    const unsigned int *rows,
    const unsigned int *first_row_argument,
    const unsigned int *row_count_argument,
    const unsigned int *walk_columns_argument,
    const unsigned int *rows_are_every_node_argument,
    const unsigned char *source_moves,
    float *group_loss,
    float *group_worst_error
) {
    __shared__ float scratch[32];
    const Fit_Walk_Constants constants = *constants_argument;
    const unsigned int first_row = *first_row_argument;
    const unsigned int row_count = *row_count_argument;
    const bool along_column = *walk_columns_argument != 0u;
    const bool rows_are_every_node = *rows_are_every_node_argument != 0u;
    const Fit_Walk_Buffers buffers = {
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        basis_u, basis_v, coefficients, target_lanes, value_row_start, value_target, value,
        incoming_offset, incoming_entry};
    const unsigned int walkers = constants.walkers_per_row;
    const unsigned long long walk_count = static_cast<unsigned long long>(row_count) * walkers;
    float loss = 0.0f;
    float worst_error = 0.0f;
    for (unsigned long long walk_index = grid_thread_index(); walk_index < walk_count; walk_index += grid_thread_count()) {
        const unsigned int slot = static_cast<unsigned int>(walk_index / walkers);
        const unsigned int node = rows_are_every_node ? slot : rows[first_row + slot];
        Fit_Walk walk = fit_walk_begin(buffers, constants, node, along_column,
                                       static_cast<unsigned int>(walk_index % walkers), true);
        Fit_Value_Cursor cursor = fit_value_cursor(buffers, walk);
        while (true) {
            const unsigned int other_end = fit_walk_next(walk, buffers, constants);
            if (other_end == K2T_NO_NEIGHBOR) {
                break;
            }
            const unsigned int source = along_column ? other_end : node;
            const unsigned int target = along_column ? node : other_end;
            if (along_column && source_moves[source] != 0) {
                continue;
            }
            const float target_value = fit_target(buffers, constants, source, target,
                                                  fit_value_of(cursor, buffers, walk, other_end));
            const float residual = fit_working_value(working_u, working_v, constants.lane_count, source, target) -
                                   held_target(target_value);
            const float weight = edge_weight_of(target_value);
            float edge_loss = 0.5f * weight * residual * residual;
            float edge_error = fabsf(residual) * sqrtf(weight);
            if (is_not_finite(edge_loss) || is_not_finite(edge_error)) {
                edge_loss = positive_infinity();
                edge_error = positive_infinity();
            }
            loss += edge_loss;
            worst_error = fmaxf(worst_error, edge_error);
        }
    }

    const float group_total = block_sum(loss, scratch);
    const float group_largest = block_maximum(worst_error, scratch);
    if (threadIdx.x == 0) {
        group_loss[blockIdx.x] = group_total;
        group_worst_error[blockIdx.x] = group_largest;
    }
}

// ── fit_walk_row_solve ────────────────────────────────────────────────────────
// Each warp solves FIT_WALK_THREADS_PER_ROW / walkers_per_row rows, as each Metal SIMD group does.
extern "C" __global__ void fit_walk_row_solve_kernel(
    const unsigned int *internal_node_words,
    const unsigned int *leaf_node_words,
    const unsigned int *rank_superblock_table,
    const unsigned short *rank_subblock_table,
    const float *basis_u,
    const float *basis_v,
    const float *coefficients,
    const unsigned int *target_lanes,
    const int *value_row_start,
    const unsigned int *value_target,
    const float *value,
    const Fit_Walk_Constants *constants_argument,
    const int *incoming_offset,
    const unsigned int *incoming_entry,
    float *solved_side,
    const float *fixed_side,
    const unsigned int *rows,
    const unsigned int *first_row_argument,
    const unsigned int *row_count_argument,
    const unsigned int *solving_targets_argument,
    float *triangle_scratch,
    float *vector_scratch
) {
    const Fit_Walk_Constants constants = *constants_argument;
    const unsigned int first_row = *first_row_argument;
    const unsigned int row_count = *row_count_argument;
    const bool along_column = *solving_targets_argument != 0u;
    const unsigned int thread_id = grid_thread_index();
    const unsigned int walkers = constants.walkers_per_row;
    const unsigned int rows_per_group = FIT_WALK_THREADS_PER_ROW / walkers;
    const unsigned int thread_in_group = thread_id % FIT_WALK_THREADS_PER_ROW;
    const unsigned int group_first_slot = (thread_id / FIT_WALK_THREADS_PER_ROW) * rows_per_group;
    // The same for the whole warp, so the warp returns together.
    if (group_first_slot >= row_count) {
        return;
    }
    const unsigned int group_row_count = min(rows_per_group, row_count - group_first_slot);
    const Fit_Walk_Buffers buffers = {
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        basis_u, basis_v, coefficients, target_lanes, value_row_start, value_target, value,
        incoming_offset, incoming_entry};
    const unsigned int lane_count = constants.lane_count;

    const float root_ridge = sqrtf(constants.ridge);
    for (unsigned int row_in_group = 0; row_in_group < group_row_count; row_in_group += 1) {
        const unsigned long long slot = group_first_slot + row_in_group;
        float *triangle = triangle_scratch + slot * lane_count * lane_count;
        float *projected = vector_scratch + slot * 2u * lane_count;
        for (unsigned int first = 0; first < lane_count; first += 1) {
            for (unsigned int second = thread_in_group; second < lane_count; second += FIT_WALK_THREADS_PER_ROW) {
                triangle[first * lane_count + second] = (first == second) ? root_ridge : 0.0f;
            }
        }
        for (unsigned int lane = thread_in_group; lane < lane_count; lane += FIT_WALK_THREADS_PER_ROW) {
            projected[lane] = 0.0f;
        }
    }
    __syncwarp();

    const unsigned int own_row_in_group = thread_in_group / walkers;
    const bool walks = own_row_in_group < group_row_count;
    const unsigned int own_row = walks ? rows[first_row + group_first_slot + own_row_in_group] : 0u;
    Fit_Walk walk = fit_walk_begin(buffers, constants, own_row, along_column, thread_in_group % walkers, walks);
    Fit_Value_Cursor cursor = fit_value_cursor(buffers, walk);
    bool walking = walks;
    while (__any_sync(FULL_WARP_MASK, walking)) {
        unsigned int round_other_end = K2T_NO_NEIGHBOR;
        float round_root_weight = 0.0f;
        float round_target = 0.0f;
        if (walking) {
            round_other_end = fit_walk_next(walk, buffers, constants);
            if (round_other_end == K2T_NO_NEIGHBOR) {
                walking = false;
            } else {
                const unsigned int source = along_column ? round_other_end : own_row;
                const unsigned int target = along_column ? own_row : round_other_end;
                const float target_value = fit_target(buffers, constants, source, target,
                                                      fit_value_of(cursor, buffers, walk, round_other_end));
                round_root_weight = sqrtf(edge_weight_of(target_value));
                round_target = round_root_weight * held_target(target_value);
            }
        }

        for (unsigned int turn = 0; turn < FIT_WALK_THREADS_PER_ROW; turn += 1) {
            const unsigned int other_end = __shfl_sync(FULL_WARP_MASK, round_other_end, turn);
            const float root_weight = __shfl_sync(FULL_WARP_MASK, round_root_weight, turn);
            float edge_target = __shfl_sync(FULL_WARP_MASK, round_target, turn);
            if (other_end == K2T_NO_NEIGHBOR) {
                continue;
            }
            const unsigned long long slot = group_first_slot + turn / walkers;
            float *triangle = triangle_scratch + slot * lane_count * lane_count;
            float *projected = vector_scratch + slot * 2u * lane_count;   // Q^T b
            float *edge_row = projected + lane_count;
            const float *other_row = fixed_side + static_cast<unsigned long long>(other_end) * lane_count;
            for (unsigned int lane = thread_in_group; lane < lane_count; lane += FIT_WALK_THREADS_PER_ROW) {
                edge_row[lane] = root_weight * other_row[lane];
            }
            __syncwarp();

            // Rotate the connection's row into R one column at a time, zeroing it. Every thread reads
            // the pivot's values before thread 0 writes the pivot's diagonal and Q^T b entry.
            for (unsigned int pivot = 0; pivot < lane_count; pivot += 1) {
                const float incoming = edge_row[pivot];
                if (incoming == 0.0f) {
                    continue;
                }
                float *pivot_row = triangle + pivot * lane_count;
                const float diagonal = pivot_row[pivot];
                const float radius = sqrtf(diagonal * diagonal + incoming * incoming);
                const float cosine = diagonal / radius;
                const float sine = incoming / radius;
                for (unsigned int column = pivot + 1u + thread_in_group; column < lane_count; column += FIT_WALK_THREADS_PER_ROW) {
                    const float kept = pivot_row[column];
                    const float folded = edge_row[column];
                    pivot_row[column] = cosine * kept + sine * folded;
                    edge_row[column] = cosine * folded - sine * kept;
                }
                const float kept_target = projected[pivot];
                const float rotated_target = cosine * kept_target + sine * edge_target;
                edge_target = cosine * edge_target - sine * kept_target;
                __syncwarp();
                if (thread_in_group == 0u) {
                    pivot_row[pivot] = radius;
                    projected[pivot] = rotated_target;
                }
                __syncwarp();
            }
        }
    }

    // R x = Q^T b, in place in projected, for each row: each sum is split over the warp's threads.
    for (unsigned int row_in_group = 0; row_in_group < group_row_count; row_in_group += 1) {
        const unsigned long long slot = group_first_slot + row_in_group;
        float *triangle = triangle_scratch + slot * lane_count * lane_count;
        float *projected = vector_scratch + slot * 2u * lane_count;
        bool solvable = true;
        for (unsigned int countdown = lane_count; countdown > 0u; countdown -= 1) {
            const unsigned int first = countdown - 1u;
            const float diagonal = triangle[first * lane_count + first];
            if (!(diagonal != 0.0f) || is_not_finite(diagonal)) {
                solvable = false;
                break;
            }
            float partial_sum = 0.0f;
            for (unsigned int inner = first + 1u + thread_in_group; inner < lane_count; inner += FIT_WALK_THREADS_PER_ROW) {
                partial_sum += triangle[first * lane_count + inner] * projected[inner];
            }
            const float sum = warp_sum(partial_sum);
            if (thread_in_group == 0u) {
                projected[first] = (projected[first] - sum) / diagonal;
            }
            __syncwarp();
        }
        if (!solvable) {
            continue;
        }
        float *solved_row = solved_side + static_cast<unsigned long long>(rows[first_row + slot]) * lane_count;
        for (unsigned int lane = thread_in_group; lane < lane_count; lane += FIT_WALK_THREADS_PER_ROW) {
            solved_row[lane] = projected[lane];
        }
    }
}

// ── fit_walk_gradient ─────────────────────────────────────────────────────────
// Each warp strides over the rows by the grid's warps, as each Metal SIMD group strides by the
// grid's SIMD groups.
extern "C" __global__ void fit_walk_gradient_kernel(
    const unsigned int *internal_node_words,
    const unsigned int *leaf_node_words,
    const unsigned int *rank_superblock_table,
    const unsigned short *rank_subblock_table,
    const float *basis_u,
    const float *basis_v,
    const float *coefficients,
    const unsigned int *target_lanes,
    const int *value_row_start,
    const unsigned int *value_target,
    const float *value,
    const Fit_Walk_Constants *constants_argument,
    const int *incoming_offset,
    const unsigned int *incoming_entry,
    const float *working_u,
    const float *working_v,
    const unsigned int *moved_rows,
    const unsigned int *moved_source_count_argument,
    const unsigned int *moved_row_count_argument,
    const float *direction,
    float *gradient,
    float *group_ridge_term,
    float *group_slope
) {
    __shared__ float scratch[32];
    const Fit_Walk_Constants constants = *constants_argument;
    const unsigned int moved_source_count = *moved_source_count_argument;
    const unsigned int moved_row_count = *moved_row_count_argument;
    const Fit_Walk_Buffers buffers = {
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        basis_u, basis_v, coefficients, target_lanes, value_row_start, value_target, value,
        incoming_offset, incoming_entry};
    const unsigned int lane_count = constants.lane_count;
    const unsigned int walkers = constants.walkers_per_row;
    const unsigned int rows_per_group = FIT_WALK_THREADS_PER_ROW / walkers;
    const unsigned int group_count = grid_thread_count() / FIT_WALK_THREADS_PER_ROW;
    const unsigned int simd_lane = warp_lane();
    float ridge_term = 0.0f;
    float slope = 0.0f;
    for (unsigned int group_first_slot = (grid_thread_index() / FIT_WALK_THREADS_PER_ROW) * rows_per_group;
         group_first_slot < moved_row_count; group_first_slot += group_count * rows_per_group) {
        const unsigned int group_row_count = min(rows_per_group, moved_row_count - group_first_slot);

        // Each row's sums start at ridge * the row's lane.
        for (unsigned int row_in_group = 0; row_in_group < group_row_count; row_in_group += 1) {
            const unsigned int slot = group_first_slot + row_in_group;
            const float *own_side = (slot >= moved_source_count) ? working_v : working_u;
            const unsigned long long row_start = static_cast<unsigned long long>(moved_rows[slot]) * lane_count;
            for (unsigned int lane = simd_lane; lane < lane_count; lane += FIT_WALK_THREADS_PER_ROW) {
                gradient[static_cast<unsigned long long>(slot) * lane_count + lane] = constants.ridge * own_side[row_start + lane];
            }
        }

        const unsigned int own_row_in_group = simd_lane / walkers;
        const bool walks = own_row_in_group < group_row_count;
        const unsigned int own_slot = group_first_slot + own_row_in_group;
        const unsigned int own_row = walks ? moved_rows[own_slot] : 0u;
        const bool along_column = own_slot >= moved_source_count;
        Fit_Walk walk = fit_walk_begin(buffers, constants, own_row, along_column, simd_lane % walkers, walks);
        Fit_Value_Cursor cursor = fit_value_cursor(buffers, walk);
        bool walking = walks;
        while (__any_sync(FULL_WARP_MASK, walking)) {
            unsigned int round_other_end = K2T_NO_NEIGHBOR;
            float round_weighted_residual = 0.0f;
            if (walking) {
                round_other_end = fit_walk_next(walk, buffers, constants);
                if (round_other_end == K2T_NO_NEIGHBOR) {
                    walking = false;
                } else {
                    const unsigned int source = along_column ? round_other_end : own_row;
                    const unsigned int target = along_column ? own_row : round_other_end;
                    const float target_value = fit_target(buffers, constants, source, target,
                                                          fit_value_of(cursor, buffers, walk, round_other_end));
                    const float residual = fit_working_value(working_u, working_v, lane_count, source, target) -
                                           held_target(target_value);
                    round_weighted_residual = edge_weight_of(target_value) * residual;
                }
            }
            for (unsigned int turn = 0; turn < FIT_WALK_THREADS_PER_ROW; turn += 1) {
                const unsigned int other_end = __shfl_sync(FULL_WARP_MASK, round_other_end, turn);
                const float weighted_residual = __shfl_sync(FULL_WARP_MASK, round_weighted_residual, turn);
                if (other_end == K2T_NO_NEIGHBOR) {
                    continue;
                }
                const unsigned int slot = group_first_slot + turn / walkers;
                const float *other_side = (slot >= moved_source_count) ? working_u : working_v;
                const float *other_row = other_side + static_cast<unsigned long long>(other_end) * lane_count;
                float *sums = gradient + static_cast<unsigned long long>(slot) * lane_count;
                for (unsigned int lane = simd_lane; lane < lane_count; lane += FIT_WALK_THREADS_PER_ROW) {
                    sums[lane] += weighted_residual * other_row[lane];
                }
            }
        }

        for (unsigned int row_in_group = 0; row_in_group < group_row_count; row_in_group += 1) {
            const unsigned int slot = group_first_slot + row_in_group;
            const float *own_side = (slot >= moved_source_count) ? working_v : working_u;
            const unsigned long long row_start = static_cast<unsigned long long>(moved_rows[slot]) * lane_count;
            for (unsigned int lane = simd_lane; lane < lane_count; lane += FIT_WALK_THREADS_PER_ROW) {
                const unsigned long long index = static_cast<unsigned long long>(slot) * lane_count + lane;
                const float row_value = own_side[row_start + lane];
                ridge_term += 0.5f * constants.ridge * row_value * row_value;
                slope += gradient[index] * direction[index];
            }
        }
    }

    const float group_ridge = block_sum(ridge_term, scratch);
    const float group_slope_total = block_sum(slope, scratch);
    if (threadIdx.x == 0) {
        group_ridge_term[blockIdx.x] = group_ridge;
        group_slope[blockIdx.x] = group_slope_total;
    }
}

// ── fit_scatter_point ─────────────────────────────────────────────────────────
// step_length is the host's own value or a buffer's first float; both arrive as a pointer.
extern "C" __global__ void fit_scatter_point_kernel(
    float *working_u,
    float *working_v,
    const unsigned int *moved_rows,
    const unsigned int *moved_source_count_argument,
    const unsigned int *moved_row_count_argument,
    const unsigned int *lane_count_argument,
    const float *point,
    const float *direction,
    const float *step_length_argument
) {
    const unsigned int moved_source_count = *moved_source_count_argument;
    const unsigned int lane_count = *lane_count_argument;
    const float step_length = *step_length_argument;
    const unsigned long long length = static_cast<unsigned long long>(*moved_row_count_argument) * lane_count;
    for (unsigned long long index = grid_thread_index(); index < length; index += grid_thread_count()) {
        const unsigned int slot = static_cast<unsigned int>(index / lane_count);
        const unsigned int lane = static_cast<unsigned int>(index % lane_count);
        float *side = (slot < moved_source_count) ? working_u : working_v;
        const float offset = (step_length != 0.0f) ? step_length * direction[index] : 0.0f;
        side[static_cast<unsigned long long>(moved_rows[slot]) * lane_count + lane] = point[index] + offset;
    }
}

// ── fit_gather_point ──────────────────────────────────────────────────────────
extern "C" __global__ void fit_gather_point_kernel(
    const float *working_u,
    const float *working_v,
    const unsigned int *moved_rows,
    const unsigned int *moved_source_count_argument,
    const unsigned int *moved_row_count_argument,
    const unsigned int *lane_count_argument,
    float *point
) {
    const unsigned int moved_source_count = *moved_source_count_argument;
    const unsigned int lane_count = *lane_count_argument;
    const unsigned long long length = static_cast<unsigned long long>(*moved_row_count_argument) * lane_count;
    for (unsigned long long index = grid_thread_index(); index < length; index += grid_thread_count()) {
        const unsigned int slot = static_cast<unsigned int>(index / lane_count);
        const unsigned int lane = static_cast<unsigned int>(index % lane_count);
        const float *side = (slot < moved_source_count) ? working_u : working_v;
        point[index] = side[static_cast<unsigned long long>(moved_rows[slot]) * lane_count + lane];
    }
}

// ── fit_accept_step ───────────────────────────────────────────────────────────
extern "C" __global__ void fit_accept_step_kernel(
    const float *working_u,
    const float *working_v,
    const unsigned int *moved_rows,
    const unsigned int *moved_source_count_argument,
    const unsigned int *moved_row_count_argument,
    const unsigned int *lane_count_argument,
    float *point,
    const float *trial_gradient,
    float *basis,
    const unsigned int *step_slot_argument,
    const unsigned int *gradient_change_slot_argument,
    const unsigned int *gradient_slot_argument,
    const unsigned int *record_history_argument
) {
    const unsigned int moved_source_count = *moved_source_count_argument;
    const unsigned int lane_count = *lane_count_argument;
    const unsigned long long step_slot = *step_slot_argument;
    const unsigned long long gradient_change_slot = *gradient_change_slot_argument;
    const unsigned long long gradient_slot = *gradient_slot_argument;
    const bool record_history = *record_history_argument != 0u;
    const unsigned long long length = static_cast<unsigned long long>(*moved_row_count_argument) * lane_count;
    for (unsigned long long index = grid_thread_index(); index < length; index += grid_thread_count()) {
        const unsigned int slot = static_cast<unsigned int>(index / lane_count);
        const unsigned int lane = static_cast<unsigned int>(index % lane_count);
        const float *side = (slot < moved_source_count) ? working_u : working_v;
        const float accepted = side[static_cast<unsigned long long>(moved_rows[slot]) * lane_count + lane];
        if (record_history) {
            basis[step_slot * length + index] = accepted - point[index];
            basis[gradient_change_slot * length + index] = trial_gradient[index] - basis[gradient_slot * length + index];
        }
        point[index] = accepted;
        basis[gradient_slot * length + index] = trial_gradient[index];
    }
}

// ── fit_column_square_sums ────────────────────────────────────────────────────
extern "C" __global__ void fit_column_square_sums_kernel(
    const float *working_u,
    const float *working_v,
    const unsigned int *node_count_argument,
    const unsigned int *lane_count_argument,
    const unsigned int *chunk_count_argument,
    float *partial_sums
) {
    const unsigned int node_count = *node_count_argument;
    const unsigned int lane_count = *lane_count_argument;
    const unsigned int chunk_count = *chunk_count_argument;
    const unsigned int thread_id = grid_thread_index();
    const unsigned int lane = thread_id % lane_count;
    const unsigned int chunk = thread_id / lane_count;
    if (chunk >= chunk_count) {
        return;
    }
    float u_sum = 0.0f;
    float v_sum = 0.0f;
    for (unsigned int node = chunk; node < node_count; node += chunk_count) {
        const float u_value = working_u[static_cast<unsigned long long>(node) * lane_count + lane];
        const float v_value = working_v[static_cast<unsigned long long>(node) * lane_count + lane];
        u_sum += u_value * u_value;
        v_sum += v_value * v_value;
    }
    partial_sums[(static_cast<unsigned long long>(chunk) * 2u) * lane_count + lane] = u_sum;
    partial_sums[(static_cast<unsigned long long>(chunk) * 2u + 1u) * lane_count + lane] = v_sum;
}

// ── fit_scale_columns ─────────────────────────────────────────────────────────
extern "C" __global__ void fit_scale_columns_kernel(
    float *working_u,
    float *working_v,
    const unsigned int *node_count_argument,
    const unsigned int *lane_count_argument,
    const float *factors
) {
    const unsigned int lane_count = *lane_count_argument;
    const unsigned long long length = static_cast<unsigned long long>(*node_count_argument) * lane_count;
    for (unsigned long long index = grid_thread_index(); index < length; index += grid_thread_count()) {
        const float factor = factors[index % lane_count];
        working_u[index] *= factor;
        working_v[index] /= factor;
    }
}

// The most basis vectors and left slots fit_dot_products_kernel takes, as in the Metal file.
#define FIT_MAXIMUM_BASIS_VECTOR_COUNT 64
#define FIT_MAXIMUM_LEFT_SLOT_COUNT 3

// ── fit_dot_products ──────────────────────────────────────────────────────────
extern "C" __global__ void fit_dot_products_kernel(
    const float *basis,
    const unsigned long long *length_argument,
    const unsigned int *vector_count_argument,
    const unsigned int *left_slots,
    const unsigned int *left_count_argument,
    float *group_dot_products
) {
    __shared__ float scratch[32];
    const unsigned long long length = *length_argument;
    const unsigned int count = min(*vector_count_argument, static_cast<unsigned int>(FIT_MAXIMUM_BASIS_VECTOR_COUNT));
    const unsigned int lefts = min(*left_count_argument, static_cast<unsigned int>(FIT_MAXIMUM_LEFT_SLOT_COUNT));
    unsigned int left_slot_values[FIT_MAXIMUM_LEFT_SLOT_COUNT];
    for (unsigned int left = 0; left < lefts; left += 1) {
        left_slot_values[left] = left_slots[left];
    }
    float sums[FIT_MAXIMUM_LEFT_SLOT_COUNT * FIT_MAXIMUM_BASIS_VECTOR_COUNT];
    for (unsigned int pair = 0; pair < lefts * count; pair += 1) {
        sums[pair] = 0.0f;
    }
    for (unsigned long long index = grid_thread_index(); index < length; index += grid_thread_count()) {
        float left_values[FIT_MAXIMUM_LEFT_SLOT_COUNT];
        for (unsigned int left = 0; left < lefts; left += 1) {
            left_values[left] = basis[static_cast<unsigned long long>(left_slot_values[left]) * length + index];
        }
        for (unsigned int slot = 0; slot < count; slot += 1) {
            const float basis_value = basis[static_cast<unsigned long long>(slot) * length + index];
            for (unsigned int left = 0; left < lefts; left += 1) {
                sums[left * count + slot] += left_values[left] * basis_value;
            }
        }
    }
    for (unsigned int pair = 0; pair < lefts * count; pair += 1) {
        const float group_total = block_sum(sums[pair], scratch);
        if (threadIdx.x == 0) {
            group_dot_products[static_cast<unsigned long long>(blockIdx.x) * lefts * count + pair] = group_total;
        }
    }
}

// ── fit_reduce_dot_products ───────────────────────────────────────────────────
extern "C" __global__ void fit_reduce_dot_products_kernel(
    const float *group_dot_products,
    const unsigned int *group_count_argument,
    const unsigned int *vector_count_argument,
    const unsigned int *left_slots,
    const unsigned int *left_count_argument,
    float *products
) {
    const unsigned int group_count = *group_count_argument;
    const unsigned int vector_count = *vector_count_argument;
    const unsigned int left_count = *left_count_argument;
    const unsigned int thread_id = grid_thread_index();
    if (thread_id >= left_count * vector_count) {
        return;
    }
    const unsigned int left = thread_id / vector_count;
    const unsigned int slot = thread_id % vector_count;
    float sum = 0.0f;
    for (unsigned int group = 0; group < group_count; group += 1) {
        sum += group_dot_products[(static_cast<unsigned long long>(group) * left_count + left) * vector_count + slot];
    }
    const unsigned int left_slot = left_slots[left];
    products[left_slot * vector_count + slot] = sum;
    bool slot_is_left = false;
    for (unsigned int other = 0; other < left_count; other += 1) {
        slot_is_left = slot_is_left || left_slots[other] == slot;
    }
    if (!slot_is_left) {
        products[slot * vector_count + left_slot] = sum;
    }
}

// The most history pairs fit_quasi_newton_direction_kernel keeps.
#define FIT_MAXIMUM_HISTORY_LENGTH 31

// ── fit_quasi_newton_direction ────────────────────────────────────────────────
extern "C" __global__ void fit_quasi_newton_direction_kernel(
    const float *products,
    unsigned int *history,
    const unsigned int *history_length_argument,
    const unsigned int *new_pair_slot_argument,
    const unsigned int *reset_history_argument,
    float *coefficients,
    float *status,
    float *step_length
) {
    if (grid_thread_index() != 0u) {
        return;
    }
    const unsigned int history_length = *history_length_argument;
    const unsigned int new_pair_slot = *new_pair_slot_argument;
    const unsigned int vector_count = 2u * history_length + 1u;
    const unsigned int gradient_slot = 2u * history_length;
    unsigned int count = (*reset_history_argument != 0u) ? 0u : history[0];
    if (new_pair_slot < history_length) {
        if (count == history_length) {
            for (unsigned int index = 1; index < count; index += 1) {
                history[index] = history[index + 1];
            }
            count -= 1u;
        }
        const float curvature = products[new_pair_slot * vector_count + history_length + new_pair_slot];
        const float change_norm = products[(history_length + new_pair_slot) * vector_count + history_length + new_pair_slot];
        if (!is_not_finite(curvature) && curvature > 1e-10f * change_norm) {
            history[1 + count] = new_pair_slot;
            count += 1u;
        }
    }
    history[0] = count;

    for (unsigned int slot = 0; slot < vector_count; slot += 1) {
        coefficients[slot] = 0.0f;
    }
    coefficients[gradient_slot] = 1.0f;
    float alpha[FIT_MAXIMUM_HISTORY_LENGTH];
    for (unsigned int countdown = count; countdown > 0u; countdown -= 1) {
        const unsigned int step_slot = history[countdown];
        const unsigned int change_slot = history_length + step_slot;
        float dot_product = 0.0f;
        for (unsigned int slot = 0; slot < vector_count; slot += 1) {
            dot_product += coefficients[slot] * products[step_slot * vector_count + slot];
        }
        alpha[countdown - 1] = dot_product / products[step_slot * vector_count + change_slot];
        coefficients[change_slot] -= alpha[countdown - 1];
    }
    if (count > 0u) {
        const unsigned int newest = history[count];
        const unsigned int newest_change = history_length + newest;
        const float gamma = products[newest * vector_count + newest_change] / products[newest_change * vector_count + newest_change];
        for (unsigned int slot = 0; slot < vector_count; slot += 1) {
            coefficients[slot] *= gamma;
        }
    }
    for (unsigned int index = 0; index < count; index += 1) {
        const unsigned int step_slot = history[index + 1];
        const unsigned int change_slot = history_length + step_slot;
        float dot_product = 0.0f;
        for (unsigned int slot = 0; slot < vector_count; slot += 1) {
            dot_product += coefficients[slot] * products[change_slot * vector_count + slot];
        }
        const float beta = dot_product / products[step_slot * vector_count + change_slot];
        coefficients[step_slot] += alpha[index] - beta;
    }

    float slope = 0.0f;
    for (unsigned int slot = 0; slot < vector_count; slot += 1) {
        slope -= coefficients[slot] * products[gradient_slot * vector_count + slot];
    }
    bool steepest = (count == 0u);
    if (!(slope < 0.0f) || is_not_finite(slope)) {
        count = 0;
        history[0] = 0;
        for (unsigned int slot = 0; slot < vector_count; slot += 1) {
            coefficients[slot] = 0.0f;
        }
        coefficients[gradient_slot] = 1.0f;
        slope = -products[gradient_slot * vector_count + gradient_slot];
        steepest = true;
    }
    for (unsigned int slot = 0; slot < vector_count; slot += 1) {
        coefficients[slot] = -coefficients[slot];
    }
    const float gradient_norm_squared = products[gradient_slot * vector_count + gradient_slot];
    const float first_step = steepest ? fminf(1.0f, 1.0f / sqrtf(fmaxf(gradient_norm_squared, 1e-30f))) : 1.0f;
    status[0] = slope;
    status[1] = first_step;
    status[2] = steepest ? 1.0f : 0.0f;
    status[3] = (slope < 0.0f) ? 0.0f : 1.0f;
    step_length[0] = first_step;
}

// ── fit_linear_combination ────────────────────────────────────────────────────
extern "C" __global__ void fit_linear_combination_kernel(
    const float *basis,
    const unsigned long long *length_argument,
    const unsigned int *vector_count_argument,
    const float *coefficients,
    float *output
) {
    const unsigned long long length = *length_argument;
    const unsigned int vector_count = *vector_count_argument;
    for (unsigned long long index = grid_thread_index(); index < length; index += grid_thread_count()) {
        float sum = 0.0f;
        for (unsigned int slot = 0; slot < vector_count; slot += 1) {
            sum += coefficients[slot] * basis[static_cast<unsigned long long>(slot) * length + index];
        }
        output[index] = sum;
    }
}

namespace spikecorec::cuda {

const void *weight_fit_kernels_symbol(const char *name) {
    struct Named_Kernel {
        const char *name;
        const void *symbol;
    };
    static const Named_Kernel KERNELS[] = {
        {"fit_walk_residual_kernel", reinterpret_cast<const void *>(&fit_walk_residual_kernel)},
        {"fit_walk_row_solve_kernel", reinterpret_cast<const void *>(&fit_walk_row_solve_kernel)},
        {"fit_walk_gradient_kernel", reinterpret_cast<const void *>(&fit_walk_gradient_kernel)},
        {"fit_scatter_point_kernel", reinterpret_cast<const void *>(&fit_scatter_point_kernel)},
        {"fit_gather_point_kernel", reinterpret_cast<const void *>(&fit_gather_point_kernel)},
        {"fit_accept_step_kernel", reinterpret_cast<const void *>(&fit_accept_step_kernel)},
        {"fit_column_square_sums_kernel", reinterpret_cast<const void *>(&fit_column_square_sums_kernel)},
        {"fit_scale_columns_kernel", reinterpret_cast<const void *>(&fit_scale_columns_kernel)},
        {"fit_dot_products_kernel", reinterpret_cast<const void *>(&fit_dot_products_kernel)},
        {"fit_reduce_dot_products_kernel", reinterpret_cast<const void *>(&fit_reduce_dot_products_kernel)},
        {"fit_quasi_newton_direction_kernel", reinterpret_cast<const void *>(&fit_quasi_newton_direction_kernel)},
        {"fit_linear_combination_kernel", reinterpret_cast<const void *>(&fit_linear_combination_kernel)},
    };
    for (const Named_Kernel &kernel : KERNELS) {
        if (std::strcmp(name, kernel.name) == 0) {
            return kernel.symbol;
        }
    }
    return nullptr;
}

} // namespace spikecorec::cuda
