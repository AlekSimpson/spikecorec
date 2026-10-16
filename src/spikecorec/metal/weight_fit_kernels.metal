#include <metal_stdlib>
using namespace metal;

#include "k2tree_device.metalinc"

// Kernels for Weight_Matrix's two candidate fits (masked alternating least squares and limited-
// memory BFGS). A fit works on one plane's own lanes, copied out of U and V into dense working
// arrays [node][lane_count].
//
// A fit keeps no array per connection. It walks the k^2-tree, a source's row for the connections
// out of it and a target's column for the connections into it, and computes each connection's
// target where it reads it (fit_target): what the own lanes must hold there, in units of the
// plane's scale. That is the basis over target_lanes, times target_lane_sign, plus the plane's
// value in the fit's values (CSR over source rows, keyed by target node: during a refit the
// plane's updates in S, rekeyed; during construction its starting values). NaN marks a value no
// basis can hold; edge_weight_of(target) scales its squared error.
//
// The rows a fit may move are moved_rows: the sources first (moved_source_count of them), then
// the targets. A packed vector holds one value per moved row and lane, at slot * lane_count + lane.
//
// A walk is a chain of dependent reads, so the walking kernels split each row among
// walkers_per_row threads (walkers), each walking its share of the other ends, enough that the GPU
// has many chains in flight and each walker still finds a few dozen connections. The host launches
// whole SIMD groups of FIT_WALK_THREADS_PER_ROW threads. Reductions write one partial per threadgroup
// and the host adds the partials up; the threads stride by the grid, so the host bounds how many
// partials there are.

#define FIT_WALK_THREADS_PER_ROW 32u

// NaN or infinite, by bit pattern, which fast math cannot fold away.
inline bool is_not_finite(float value) {
    return (as_type<uint>(value) & 0x7fffffffu) >= 0x7f800000u;
}

// An edge's weight in the objective: 1 / max(|target|, 1)^2, the target in units of the plane's
// RMS, so its error counts against max(|value|, RMS); a non-finite target gets none.
inline float edge_weight_of(float target) {
    if (is_not_finite(target)) {
        return 0.0f;
    }
    float error_scale = max(fabs(target), 1.0f);
    return 1.0f / (error_scale * error_scale);
}

// The target the lanes are fitted to: a non-finite one counts as zero, with no weight.
inline float held_target(float target) {
    return is_not_finite(target) ? 0.0f : target;
}

// The threadgroup's sum of one value per thread; valid in the threadgroup's first thread.
inline float threadgroup_sum(float value, threadgroup float *scratch, uint simd_lane, uint simd_group,
                             uint simd_group_count) {
    value = simd_sum(value);
    if (simd_lane == 0) {
        scratch[simd_group] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    if (simd_group == 0) {
        total = (simd_lane < simd_group_count) ? scratch[simd_lane] : 0.0f;
        total = simd_sum(total);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// The threadgroup's largest value (values are never negative); valid in the first thread.
inline float threadgroup_maximum(float value, threadgroup float *scratch, uint simd_lane, uint simd_group,
                                 uint simd_group_count) {
    value = simd_max(value);
    if (simd_lane == 0) {
        scratch[simd_group] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float largest = 0.0f;
    if (simd_group == 0) {
        largest = (simd_lane < simd_group_count) ? scratch[simd_lane] : 0.0f;
        largest = simd_max(largest);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return largest;
}

// ── walking the plane's connections ───────────────────────────────────────────
// The scalars a walking kernel reads; Weight_Matrix::Fit_Walk_Constants on the host.
struct Fit_Walk_Constants {
    uint node_count;
    uint branching_factor;
    uint superblock_size_words;
    uint padded_node_count;
    uint tree_height;
    uint internal_bit_count;
    uint basis_lane_count;   // floats in a row of U or V
    uint target_lane_count;  // entries of target_lanes
    float target_lane_sign;  // +1: the lanes hold the plane's values; -1: shared lanes the own lanes cancel
    float inverse_scale;     // one over the plane's scale, the unit of the working arrays
    uint lane_count;         // working lanes
    float ridge;
    uint walkers_per_row;    // a power of two from 1 to FIT_WALK_THREADS_PER_ROW
};

// The buffers a walking kernel reads, gathered so the helpers below can take them whole.
struct Fit_Walk_Buffers {
    const device uint   *internal_node_words;
    const device uint   *leaf_node_words;
    const device uint   *rank_superblock_table;
    const device ushort *rank_subblock_table;
    const device float  *basis_u;
    const device float  *basis_v;
    const device float  *coefficients;   // the plane's coefficient row
    const device uint   *target_lanes;
    const device int    *value_row_start; // [node_count + 1]
    const device uint   *value_target;
    const device float  *value;
};

// One walker's walk along a node's row (its targets) or column (its sources), limited to the
// walker's share of the other ends, [first, end). A walk keeps the k^2-tree path it is on.
struct Fit_Walk {
    uint row_base[MAX_K2TREE_HEIGHT];
    uint column_base[MAX_K2TREE_HEIGHT];
    uint block_size[MAX_K2TREE_HEIGHT];
    uint bit_offset[MAX_K2TREE_HEIGHT];
    uint next_child[MAX_K2TREE_HEIGHT];
    int top;
    uint node;
    bool along_column;
    uint first;
    uint end;
};

// Walker `walker` of a row's walkers_per_row walks the other ends in its share of the nodes. With
// valid unset the walk is done before it starts.
inline Fit_Walk fit_walk_begin(constant Fit_Walk_Constants &constants, uint node, bool along_column, uint walker,
                               bool valid) {
    Fit_Walk walk;
    const uint share = (constants.node_count + constants.walkers_per_row - 1) / constants.walkers_per_row;
    walk.node = node;
    walk.along_column = along_column;
    walk.first = min(walker * share, constants.node_count);
    walk.end = min(walk.first + share, constants.node_count);
    walk.row_base[0] = 0;
    walk.column_base[0] = 0;
    walk.block_size[0] = constants.padded_node_count;
    walk.bit_offset[0] = 0;
    walk.next_child[0] = 0;
    walk.top = (valid && constants.tree_height > 0 && walk.first < walk.end) ? 0 : -1;
    return walk;
}

// The walk's next other end, ascending, or K2T_NO_NEIGHBOR when it is done. As k2t_next_neighbor
// (along a row) and k2t_next_predecessor (along a column), except that it skips the subtrees whose
// other ends all lie outside [first, end), and stops past end.
inline uint fit_walk_next(thread Fit_Walk &walk, thread const Fit_Walk_Buffers &buffers,
                          constant Fit_Walk_Constants &constants) {
    const uint branching_factor = constants.branching_factor;
    const int leaf_level = static_cast<int>(constants.tree_height) - 1;
    while (walk.top >= 0) {
        int level = walk.top;
        uint child = walk.next_child[level];
        if (child >= branching_factor) {
            walk.top -= 1;
            continue;
        }
        walk.next_child[level] = child + 1;

        uint child_block_size = walk.block_size[level] / branching_factor;
        // The child's first node along the walked end, and the node's place on the fixed end.
        uint walked_base = walk.along_column ? walk.row_base[level] : walk.column_base[level];
        uint fixed_base = walk.along_column ? walk.column_base[level] : walk.row_base[level];
        uint child_first = walked_base + child * child_block_size;
        if (child_first >= walk.end) {
            walk.top -= 1;
            continue;
        }
        if (child_first + child_block_size <= walk.first) {
            continue;
        }
        uint fixed_offset = (walk.node - fixed_base) / child_block_size;
        uint row_offset = walk.along_column ? child : fixed_offset;
        uint column_offset = walk.along_column ? fixed_offset : child;
        uint bit_position = walk.bit_offset[level] + row_offset * branching_factor + column_offset;

        if (level == leaf_level) {
            if (k2t_get_bit(buffers.leaf_node_words, bit_position) && child_first < constants.node_count) {
                return child_first;
            }
        } else if (k2t_get_bit(buffers.internal_node_words, bit_position)) {
            uint rank_inclusive = k2t_rank1_exclusive(buffers.internal_node_words, buffers.rank_superblock_table,
                                                       buffers.rank_subblock_table, bit_position,
                                                       static_cast<int>(constants.superblock_size_words)) + 1u;
            uint raw_offset = branching_factor * branching_factor * rank_inclusive;
            int child_level = level + 1;
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
inline int fit_value_position(thread const Fit_Walk_Buffers &buffers, uint source, uint first_target) {
    int low = buffers.value_row_start[source];
    int high = buffers.value_row_start[source + 1];
    while (low < high) {
        int middle = low + (high - low) / 2;
        if (buffers.value_target[middle] < first_target) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

// The fit's value for (source, target), or zero when its values hold none.
inline float fit_value_at(thread const Fit_Walk_Buffers &buffers, uint source, uint target) {
    int position = fit_value_position(buffers, source, target);
    return (position < buffers.value_row_start[source + 1] && buffers.value_target[position] == target)
               ? buffers.value[position]
               : 0.0f;
}

// A row walk visits its targets in ascending order, the order of the row's values, so it reads
// them with a cursor that starts at the walk's first target.
struct Fit_Value_Cursor {
    int position;
    int end;
};

inline Fit_Value_Cursor fit_value_cursor(thread const Fit_Walk_Buffers &buffers, thread const Fit_Walk &walk) {
    Fit_Value_Cursor cursor;
    const bool reads_row = !walk.along_column && walk.top >= 0;
    cursor.position = reads_row ? fit_value_position(buffers, walk.node, walk.first) : 0;
    cursor.end = reads_row ? buffers.value_row_start[walk.node + 1] : 0;
    return cursor;
}

// The fit's value for the walk's connection to other_end.
inline float fit_value_of(thread Fit_Value_Cursor &cursor, thread const Fit_Walk_Buffers &buffers,
                          thread const Fit_Walk &walk, uint other_end) {
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

// What the own lanes must hold at (source, target), given the fit's value there: the basis over
// target_lanes plus the value, in units of the scale.
inline float fit_target(thread const Fit_Walk_Buffers &buffers, constant Fit_Walk_Constants &constants,
                        uint source, uint target, float value) {
    float basis_part = 0.0f;
    const device float *source_row = buffers.basis_u + static_cast<ulong>(source) * constants.basis_lane_count;
    const device float *target_row = buffers.basis_v + static_cast<ulong>(target) * constants.basis_lane_count;
    for (uint index = 0; index < constants.target_lane_count; ++index) {
        uint lane = buffers.target_lanes[index];
        basis_part += source_row[lane] * buffers.coefficients[lane] * target_row[lane];
    }
    return (constants.target_lane_sign * basis_part + value) * constants.inverse_scale;
}

// The own lanes' value at (source, target).
inline float fit_working_value(const device float *working_u, const device float *working_v, uint lane_count,
                               uint source, uint target) {
    const device float *source_row = working_u + static_cast<ulong>(source) * lane_count;
    const device float *target_row = working_v + static_cast<ulong>(target) * lane_count;
    float value = 0.0f;
    for (uint lane = 0; lane < lane_count; ++lane) {
        value += source_row[lane] * target_row[lane];
    }
    return value;
}

// ── fit_walk_residual ─────────────────────────────────────────────────────────
// The own lanes' residuals against fit_target over the connections of the given rows: each row's
// targets (row walks), or each row's sources (column walks, which skip a connection whose source
// moves, since a row walk counts it). Thread t is walker t % walkers_per_row of row slot
// t / walkers_per_row, striding by the grid. Per threadgroup: the sum of 0.5 * weight * residual^2
// and the worst relative error, |residual| * sqrt(weight). A non-finite residual counts as an
// infinite loss and error. With rows_are_every_node set, row slot s is node s and rows is not read.
kernel void fit_walk_residual_kernel(
    const device uint   *internal_node_words   [[ buffer(0) ]],
    const device uint   *leaf_node_words       [[ buffer(1) ]],
    const device uint   *rank_superblock_table [[ buffer(2) ]],
    const device ushort *rank_subblock_table   [[ buffer(3) ]],
    const device float  *basis_u               [[ buffer(4) ]],
    const device float  *basis_v               [[ buffer(5) ]],
    const device float  *coefficients          [[ buffer(6) ]],
    const device uint   *target_lanes          [[ buffer(7) ]],
    const device int    *value_row_start       [[ buffer(8) ]],
    const device uint   *value_target          [[ buffer(9) ]],
    const device float  *value                 [[ buffer(10) ]],
    constant Fit_Walk_Constants &constants     [[ buffer(11) ]],
    const device float  *working_u             [[ buffer(12) ]],
    const device float  *working_v             [[ buffer(13) ]],
    const device uint   *rows                  [[ buffer(14) ]],
    constant uint       &first_row             [[ buffer(15) ]],
    constant uint       &row_count             [[ buffer(16) ]],
    constant uint       &walk_columns          [[ buffer(17) ]],
    constant uint       &rows_are_every_node   [[ buffer(18) ]],
    const device uchar  *source_moves          [[ buffer(19) ]],
    device float        *group_loss            [[ buffer(20) ]],
    device float        *group_worst_error     [[ buffer(21) ]],
    uint thread_id        [[ thread_position_in_grid ]],
    uint grid_size        [[ threads_per_grid ]],
    uint group_id         [[ threadgroup_position_in_grid ]],
    uint simd_lane        [[ thread_index_in_simdgroup ]],
    uint simd_group       [[ simdgroup_index_in_threadgroup ]],
    uint simd_group_count [[ simdgroups_per_threadgroup ]]
) {
    threadgroup float scratch[32];
    const Fit_Walk_Buffers buffers = {
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        basis_u, basis_v, coefficients, target_lanes, value_row_start, value_target, value};
    const bool along_column = walk_columns != 0u;
    const uint walkers = constants.walkers_per_row;
    const ulong walk_count = static_cast<ulong>(row_count) * walkers;
    float loss = 0.0f;
    float worst_error = 0.0f;
    for (ulong walk_index = thread_id; walk_index < walk_count; walk_index += grid_size) {
        uint slot = static_cast<uint>(walk_index / walkers);
        uint node = (rows_are_every_node != 0u) ? slot : rows[first_row + slot];
        Fit_Walk walk = fit_walk_begin(constants, node, along_column, static_cast<uint>(walk_index % walkers), true);
        Fit_Value_Cursor cursor = fit_value_cursor(buffers, walk);
        while (true) {
            uint other_end = fit_walk_next(walk, buffers, constants);
            if (other_end == K2T_NO_NEIGHBOR) {
                break;
            }
            uint source = along_column ? other_end : node;
            uint target = along_column ? node : other_end;
            if (along_column && source_moves[source] != 0) {
                continue;
            }
            float target_value = fit_target(buffers, constants, source, target, fit_value_of(cursor, buffers, walk, other_end));
            float residual = fit_working_value(working_u, working_v, constants.lane_count, source, target) -
                             held_target(target_value);
            float weight = edge_weight_of(target_value);
            float edge_loss = 0.5f * weight * residual * residual;
            float edge_error = fabs(residual) * sqrt(weight);
            if (is_not_finite(edge_loss) || is_not_finite(edge_error)) {
                edge_loss = INFINITY;
                edge_error = INFINITY;
            }
            loss += edge_loss;
            worst_error = max(worst_error, edge_error);
        }
    }

    float group_total = threadgroup_sum(loss, scratch, simd_lane, simd_group, simd_group_count);
    float group_largest = threadgroup_maximum(worst_error, scratch, simd_lane, simd_group, simd_group_count);
    if (simd_group == 0 && simd_lane == 0) {
        group_loss[group_id] = group_total;
        group_worst_error[group_id] = group_largest;
    }
}

// The row solve and the gradient give a SIMD group FIT_WALK_THREADS_PER_ROW / walkers_per_row rows:
// thread t of the group is walker t % walkers_per_row of the group's row t / walkers_per_row. The
// walks take turns: in each round, every walker whose walk is not done finds its next connection,
// then the group handles the round's connections one at a time, every thread taking its share of the
// lanes of the connection's row. A round's connection is its other end, or K2T_NO_NEIGHBOR when the
// walker is done. A row's lanes are split the same way in every round, so each value is only ever
// written by one thread.

// ── fit_walk_row_solve ────────────────────────────────────────────────────────
// Masked alternating least squares on rows of one side: each row x minimizing its connections'
// weighted squared error plus ridge * |x|^2, given the other side's rows. That is the least squares
// problem [W^(1/2) A; ridge^(1/2) I] x = [W^(1/2) b; 0], with one row of A (the other end's row) and
// of b (the target) per connection. It is solved by QR: R starts as ridge^(1/2) I, and Givens
// rotations fold each connection's row into R and Q^T b in turn; then R x = Q^T b by back
// substitution. QR never forms A^T A, so it keeps its accuracy on the nearly singular rows of
// structured networks, where f32 normal equations lose it.
//
// Every thread of the group computes each rotation, from the same values, and applies it to its
// own columns (column c belongs to thread c % FIT_WALK_THREADS_PER_ROW); the back substitution's
// sums are split the same way. Apple GPUs run SIMD groups of 32 threads.
//
// Row slot s is rows[first_row + s]. Solving sources, its connections are its row's targets (row
// walks); solving targets, its column's sources (column walks). Slot s keeps R (lane_count^2,
// row-major), Q^T b and the connection row being folded (lane_count each) in device scratch at
// s * lane_count^2 and s * 2 * lane_count. A row whose R has a zero on its diagonal is left as it was.
kernel void fit_walk_row_solve_kernel(
    const device uint   *internal_node_words   [[ buffer(0) ]],
    const device uint   *leaf_node_words       [[ buffer(1) ]],
    const device uint   *rank_superblock_table [[ buffer(2) ]],
    const device ushort *rank_subblock_table   [[ buffer(3) ]],
    const device float  *basis_u               [[ buffer(4) ]],
    const device float  *basis_v               [[ buffer(5) ]],
    const device float  *coefficients          [[ buffer(6) ]],
    const device uint   *target_lanes          [[ buffer(7) ]],
    const device int    *value_row_start       [[ buffer(8) ]],
    const device uint   *value_target          [[ buffer(9) ]],
    const device float  *value                 [[ buffer(10) ]],
    constant Fit_Walk_Constants &constants     [[ buffer(11) ]],
    device float        *solved_side           [[ buffer(12) ]],
    const device float  *fixed_side            [[ buffer(13) ]],
    const device uint   *rows                  [[ buffer(14) ]],
    constant uint       &first_row             [[ buffer(15) ]],
    constant uint       &row_count             [[ buffer(16) ]],
    constant uint       &solving_targets       [[ buffer(17) ]],
    device float        *triangle_scratch      [[ buffer(18) ]],
    device float        *vector_scratch        [[ buffer(19) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    const uint walkers = constants.walkers_per_row;
    const uint rows_per_group = FIT_WALK_THREADS_PER_ROW / walkers;
    const uint thread_in_group = thread_id % FIT_WALK_THREADS_PER_ROW;
    const uint group_first_slot = (thread_id / FIT_WALK_THREADS_PER_ROW) * rows_per_group;
    if (group_first_slot >= row_count) {
        return;
    }
    const uint group_row_count = min(rows_per_group, row_count - group_first_slot);
    const Fit_Walk_Buffers buffers = {
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        basis_u, basis_v, coefficients, target_lanes, value_row_start, value_target, value};
    const uint lane_count = constants.lane_count;
    const bool along_column = solving_targets != 0u;

    float root_ridge = sqrt(constants.ridge);
    for (uint row_in_group = 0; row_in_group < group_row_count; ++row_in_group) {
        const ulong slot = group_first_slot + row_in_group;
        device float *triangle = triangle_scratch + slot * lane_count * lane_count;
        device float *projected = vector_scratch + slot * 2 * lane_count;
        for (uint first = 0; first < lane_count; ++first) {
            for (uint second = thread_in_group; second < lane_count; second += FIT_WALK_THREADS_PER_ROW) {
                triangle[first * lane_count + second] = (first == second) ? root_ridge : 0.0f;
            }
        }
        for (uint lane = thread_in_group; lane < lane_count; lane += FIT_WALK_THREADS_PER_ROW) {
            projected[lane] = 0.0f;
        }
    }
    simdgroup_barrier(mem_flags::mem_device);

    const uint own_row_in_group = thread_in_group / walkers;
    const bool walks = own_row_in_group < group_row_count;
    const uint own_row = walks ? rows[first_row + group_first_slot + own_row_in_group] : 0;
    Fit_Walk walk = fit_walk_begin(constants, own_row, along_column, thread_in_group % walkers, walks);
    Fit_Value_Cursor cursor = fit_value_cursor(buffers, walk);
    bool walking = walks;
    while (simd_any(walking)) {
        uint round_other_end = K2T_NO_NEIGHBOR;
        float round_root_weight = 0.0f;
        float round_target = 0.0f;
        if (walking) {
            round_other_end = fit_walk_next(walk, buffers, constants);
            if (round_other_end == K2T_NO_NEIGHBOR) {
                walking = false;
            } else {
                uint source = along_column ? round_other_end : own_row;
                uint target = along_column ? own_row : round_other_end;
                float target_value = fit_target(buffers, constants, source, target,
                                                fit_value_of(cursor, buffers, walk, round_other_end));
                round_root_weight = sqrt(edge_weight_of(target_value));
                round_target = round_root_weight * held_target(target_value);
            }
        }

        for (ushort turn = 0; turn < FIT_WALK_THREADS_PER_ROW; ++turn) {
            uint other_end = simd_shuffle(round_other_end, turn);
            if (other_end == K2T_NO_NEIGHBOR) {
                continue;
            }
            float root_weight = simd_shuffle(round_root_weight, turn);
            float edge_target = simd_shuffle(round_target, turn);
            const ulong slot = group_first_slot + turn / walkers;
            device float *triangle = triangle_scratch + slot * lane_count * lane_count;
            device float *projected = vector_scratch + slot * 2 * lane_count;   // Q^T b
            device float *edge_row = projected + lane_count;
            const device float *other_row = fixed_side + static_cast<ulong>(other_end) * lane_count;
            for (uint lane = thread_in_group; lane < lane_count; lane += FIT_WALK_THREADS_PER_ROW) {
                edge_row[lane] = root_weight * other_row[lane];
            }
            simdgroup_barrier(mem_flags::mem_device);

            // Rotate the connection's row into R one column at a time, zeroing it. Every thread reads
            // the pivot's values before thread 0 writes the pivot's diagonal and Q^T b entry.
            for (uint pivot = 0; pivot < lane_count; ++pivot) {
                float incoming = edge_row[pivot];
                if (incoming == 0.0f) {
                    continue;
                }
                device float *pivot_row = triangle + pivot * lane_count;
                float diagonal = pivot_row[pivot];
                float radius = sqrt(diagonal * diagonal + incoming * incoming);
                float cosine = diagonal / radius;
                float sine = incoming / radius;
                for (uint column = pivot + 1 + thread_in_group; column < lane_count; column += FIT_WALK_THREADS_PER_ROW) {
                    float kept = pivot_row[column];
                    float folded = edge_row[column];
                    pivot_row[column] = cosine * kept + sine * folded;
                    edge_row[column] = cosine * folded - sine * kept;
                }
                float kept_target = projected[pivot];
                float rotated_target = cosine * kept_target + sine * edge_target;
                edge_target = cosine * edge_target - sine * kept_target;
                simdgroup_barrier(mem_flags::mem_device);
                if (thread_in_group == 0) {
                    pivot_row[pivot] = radius;
                    projected[pivot] = rotated_target;
                }
                simdgroup_barrier(mem_flags::mem_device);
            }
        }
    }

    // R x = Q^T b, in place in projected, for each row: each sum is split over the group's threads.
    for (uint row_in_group = 0; row_in_group < group_row_count; ++row_in_group) {
        const ulong slot = group_first_slot + row_in_group;
        device float *triangle = triangle_scratch + slot * lane_count * lane_count;
        device float *projected = vector_scratch + slot * 2 * lane_count;
        bool solvable = true;
        for (uint countdown = lane_count; countdown > 0; --countdown) {
            uint first = countdown - 1;
            float diagonal = triangle[first * lane_count + first];
            if (!(diagonal != 0.0f) || is_not_finite(diagonal)) {
                solvable = false;
                break;
            }
            float partial_sum = 0.0f;
            for (uint inner = first + 1 + thread_in_group; inner < lane_count; inner += FIT_WALK_THREADS_PER_ROW) {
                partial_sum += triangle[first * lane_count + inner] * projected[inner];
            }
            float sum = simd_sum(partial_sum);
            if (thread_in_group == 0) {
                projected[first] = (projected[first] - sum) / diagonal;
            }
            simdgroup_barrier(mem_flags::mem_device);
        }
        if (!solvable) {
            continue;
        }
        device float *solved_row = solved_side + static_cast<ulong>(rows[first_row + slot]) * lane_count;
        for (uint lane = thread_in_group; lane < lane_count; lane += FIT_WALK_THREADS_PER_ROW) {
            solved_row[lane] = projected[lane];
        }
    }
}

// ── fit_walk_gradient ─────────────────────────────────────────────────────────
// The objective's gradient at every moved row and lane, packed: the row's connections' weighted
// residuals times the other end's lane, plus ridge * the row's lane. A moved source walks its row
// and a moved target its column, recomputing each connection's residual; the sums build up in
// gradient itself. Each SIMD group strides by the grid's groups. Per threadgroup: the sum of
// 0.5 * ridge * value^2 (the ridge part of the objective) and of gradient * direction (the slope
// along direction).
kernel void fit_walk_gradient_kernel(
    const device uint   *internal_node_words   [[ buffer(0) ]],
    const device uint   *leaf_node_words       [[ buffer(1) ]],
    const device uint   *rank_superblock_table [[ buffer(2) ]],
    const device ushort *rank_subblock_table   [[ buffer(3) ]],
    const device float  *basis_u               [[ buffer(4) ]],
    const device float  *basis_v               [[ buffer(5) ]],
    const device float  *coefficients          [[ buffer(6) ]],
    const device uint   *target_lanes          [[ buffer(7) ]],
    const device int    *value_row_start       [[ buffer(8) ]],
    const device uint   *value_target          [[ buffer(9) ]],
    const device float  *value                 [[ buffer(10) ]],
    constant Fit_Walk_Constants &constants     [[ buffer(11) ]],
    const device float  *working_u             [[ buffer(12) ]],
    const device float  *working_v             [[ buffer(13) ]],
    const device uint   *moved_rows            [[ buffer(14) ]],
    constant uint       &moved_source_count    [[ buffer(15) ]],
    constant uint       &moved_row_count       [[ buffer(16) ]],
    const device float  *direction             [[ buffer(17) ]],
    device float        *gradient              [[ buffer(18) ]],
    device float        *group_ridge_term      [[ buffer(19) ]],
    device float        *group_slope           [[ buffer(20) ]],
    uint thread_id        [[ thread_position_in_grid ]],
    uint grid_size        [[ threads_per_grid ]],
    uint group_id         [[ threadgroup_position_in_grid ]],
    uint simd_lane        [[ thread_index_in_simdgroup ]],
    uint simd_group       [[ simdgroup_index_in_threadgroup ]],
    uint simd_group_count [[ simdgroups_per_threadgroup ]]
) {
    threadgroup float scratch[32];
    const Fit_Walk_Buffers buffers = {
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        basis_u, basis_v, coefficients, target_lanes, value_row_start, value_target, value};
    const uint lane_count = constants.lane_count;
    const uint walkers = constants.walkers_per_row;
    const uint rows_per_group = FIT_WALK_THREADS_PER_ROW / walkers;
    const uint group_count = grid_size / FIT_WALK_THREADS_PER_ROW;
    float ridge_term = 0.0f;
    float slope = 0.0f;
    for (uint group_first_slot = (thread_id / FIT_WALK_THREADS_PER_ROW) * rows_per_group;
         group_first_slot < moved_row_count; group_first_slot += group_count * rows_per_group) {
        const uint group_row_count = min(rows_per_group, moved_row_count - group_first_slot);

        // Each row's sums start at ridge * the row's lane.
        for (uint row_in_group = 0; row_in_group < group_row_count; ++row_in_group) {
            const uint slot = group_first_slot + row_in_group;
            const device float *own_side = (slot >= moved_source_count) ? working_v : working_u;
            const ulong row_start = static_cast<ulong>(moved_rows[slot]) * lane_count;
            for (uint lane = simd_lane; lane < lane_count; lane += FIT_WALK_THREADS_PER_ROW) {
                gradient[static_cast<ulong>(slot) * lane_count + lane] = constants.ridge * own_side[row_start + lane];
            }
        }

        const uint own_row_in_group = simd_lane / walkers;
        const bool walks = own_row_in_group < group_row_count;
        const uint own_slot = group_first_slot + own_row_in_group;
        const uint own_row = walks ? moved_rows[own_slot] : 0;
        const bool along_column = own_slot >= moved_source_count;
        Fit_Walk walk = fit_walk_begin(constants, own_row, along_column, simd_lane % walkers, walks);
        Fit_Value_Cursor cursor = fit_value_cursor(buffers, walk);
        bool walking = walks;
        while (simd_any(walking)) {
            uint round_other_end = K2T_NO_NEIGHBOR;
            float round_weighted_residual = 0.0f;
            if (walking) {
                round_other_end = fit_walk_next(walk, buffers, constants);
                if (round_other_end == K2T_NO_NEIGHBOR) {
                    walking = false;
                } else {
                    uint source = along_column ? round_other_end : own_row;
                    uint target = along_column ? own_row : round_other_end;
                    float target_value = fit_target(buffers, constants, source, target,
                                                    fit_value_of(cursor, buffers, walk, round_other_end));
                    float residual = fit_working_value(working_u, working_v, lane_count, source, target) -
                                     held_target(target_value);
                    round_weighted_residual = edge_weight_of(target_value) * residual;
                }
            }
            for (ushort turn = 0; turn < FIT_WALK_THREADS_PER_ROW; ++turn) {
                uint other_end = simd_shuffle(round_other_end, turn);
                if (other_end == K2T_NO_NEIGHBOR) {
                    continue;
                }
                float weighted_residual = simd_shuffle(round_weighted_residual, turn);
                const uint slot = group_first_slot + turn / walkers;
                const device float *other_side = (slot >= moved_source_count) ? working_u : working_v;
                const device float *other_row = other_side + static_cast<ulong>(other_end) * lane_count;
                device float *sums = gradient + static_cast<ulong>(slot) * lane_count;
                for (uint lane = simd_lane; lane < lane_count; lane += FIT_WALK_THREADS_PER_ROW) {
                    sums[lane] += weighted_residual * other_row[lane];
                }
            }
        }

        for (uint row_in_group = 0; row_in_group < group_row_count; ++row_in_group) {
            const uint slot = group_first_slot + row_in_group;
            const device float *own_side = (slot >= moved_source_count) ? working_v : working_u;
            const ulong row_start = static_cast<ulong>(moved_rows[slot]) * lane_count;
            for (uint lane = simd_lane; lane < lane_count; lane += FIT_WALK_THREADS_PER_ROW) {
                const ulong index = static_cast<ulong>(slot) * lane_count + lane;
                float row_value = own_side[row_start + lane];
                ridge_term += 0.5f * constants.ridge * row_value * row_value;
                slope += gradient[index] * direction[index];
            }
        }
    }

    float group_ridge = threadgroup_sum(ridge_term, scratch, simd_lane, simd_group, simd_group_count);
    float group_slope_total = threadgroup_sum(slope, scratch, simd_lane, simd_group, simd_group_count);
    if (simd_group == 0 && simd_lane == 0) {
        group_ridge_term[group_id] = group_ridge;
        group_slope[group_id] = group_slope_total;
    }
}

// ── fit_scatter_point ─────────────────────────────────────────────────────────
// Writes point + step_length * direction into the moved rows of the working arrays. A step of
// zero writes point itself, whatever direction holds.
kernel void fit_scatter_point_kernel(
    device float       *working_u          [[ buffer(0) ]],
    device float       *working_v          [[ buffer(1) ]],
    const device uint  *moved_rows         [[ buffer(2) ]],
    constant uint      &moved_source_count [[ buffer(3) ]],
    constant uint      &moved_row_count    [[ buffer(4) ]],
    constant uint      &lane_count         [[ buffer(5) ]],
    const device float *point              [[ buffer(6) ]],
    const device float *direction          [[ buffer(7) ]],
    constant float     &step_length        [[ buffer(8) ]],
    uint thread_id [[ thread_position_in_grid ]],
    uint grid_size [[ threads_per_grid ]]
) {
    ulong length = static_cast<ulong>(moved_row_count) * lane_count;
    for (ulong index = thread_id; index < length; index += grid_size) {
        uint slot = static_cast<uint>(index / lane_count);
        uint lane = static_cast<uint>(index % lane_count);
        device float *side = (slot < moved_source_count) ? working_u : working_v;
        float offset = (step_length != 0.0f) ? step_length * direction[index] : 0.0f;
        side[static_cast<ulong>(moved_rows[slot]) * lane_count + lane] = point[index] + offset;
    }
}

// ── fit_gather_point ──────────────────────────────────────────────────────────
// Packs the moved rows of the working arrays into point.
kernel void fit_gather_point_kernel(
    const device float *working_u          [[ buffer(0) ]],
    const device float *working_v          [[ buffer(1) ]],
    const device uint  *moved_rows         [[ buffer(2) ]],
    constant uint      &moved_source_count [[ buffer(3) ]],
    constant uint      &moved_row_count    [[ buffer(4) ]],
    constant uint      &lane_count         [[ buffer(5) ]],
    device float       *point              [[ buffer(6) ]],
    uint thread_id [[ thread_position_in_grid ]],
    uint grid_size [[ threads_per_grid ]]
) {
    ulong length = static_cast<ulong>(moved_row_count) * lane_count;
    for (ulong index = thread_id; index < length; index += grid_size) {
        uint slot = static_cast<uint>(index / lane_count);
        uint lane = static_cast<uint>(index % lane_count);
        const device float *side = (slot < moved_source_count) ? working_u : working_v;
        point[index] = side[static_cast<ulong>(moved_rows[slot]) * lane_count + lane];
    }
}

// ── fit_accept_step ───────────────────────────────────────────────────────────
// After a line search, the working arrays hold the accepted point. Records the step (new point
// less old) and the gradient change into their history slots of `basis` when record_history is
// set, then makes the accepted point and its gradient (trial_gradient) current. basis holds one
// vector of `length` floats per slot.
kernel void fit_accept_step_kernel(
    const device float *working_u            [[ buffer(0) ]],
    const device float *working_v            [[ buffer(1) ]],
    const device uint  *moved_rows           [[ buffer(2) ]],
    constant uint      &moved_source_count   [[ buffer(3) ]],
    constant uint      &moved_row_count      [[ buffer(4) ]],
    constant uint      &lane_count           [[ buffer(5) ]],
    device float       *point                [[ buffer(6) ]],
    const device float *trial_gradient       [[ buffer(7) ]],
    device float       *basis                [[ buffer(8) ]],
    constant uint      &step_slot            [[ buffer(9) ]],
    constant uint      &gradient_change_slot [[ buffer(10) ]],
    constant uint      &gradient_slot        [[ buffer(11) ]],
    constant uint      &record_history       [[ buffer(12) ]],
    uint thread_id [[ thread_position_in_grid ]],
    uint grid_size [[ threads_per_grid ]]
) {
    ulong length = static_cast<ulong>(moved_row_count) * lane_count;
    for (ulong index = thread_id; index < length; index += grid_size) {
        uint slot = static_cast<uint>(index / lane_count);
        uint lane = static_cast<uint>(index % lane_count);
        const device float *side = (slot < moved_source_count) ? working_u : working_v;
        float accepted = side[static_cast<ulong>(moved_rows[slot]) * lane_count + lane];
        if (record_history != 0u) {
            basis[static_cast<ulong>(step_slot) * length + index] = accepted - point[index];
            basis[static_cast<ulong>(gradient_change_slot) * length + index] =
                trial_gradient[index] - basis[static_cast<ulong>(gradient_slot) * length + index];
        }
        point[index] = accepted;
        basis[static_cast<ulong>(gradient_slot) * length + index] = trial_gradient[index];
    }
}

// ── fit_column_square_sums ────────────────────────────────────────────────────
// The sum of squares of each own lane of U and of V, over every node, in chunk_count partials:
// thread lane + chunk * lane_count adds the nodes chunk, chunk + chunk_count, ... and writes
// partial_sums[(chunk * 2) * lane_count + lane] (U) and [(chunk * 2 + 1) * lane_count + lane] (V).
kernel void fit_column_square_sums_kernel(
    const device float *working_u    [[ buffer(0) ]],
    const device float *working_v    [[ buffer(1) ]],
    constant uint      &node_count   [[ buffer(2) ]],
    constant uint      &lane_count   [[ buffer(3) ]],
    constant uint      &chunk_count  [[ buffer(4) ]],
    device float       *partial_sums [[ buffer(5) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    uint lane = thread_id % lane_count;
    uint chunk = thread_id / lane_count;
    if (chunk >= chunk_count) {
        return;
    }
    float u_sum = 0.0f;
    float v_sum = 0.0f;
    for (uint node = chunk; node < node_count; node += chunk_count) {
        float u_value = working_u[static_cast<ulong>(node) * lane_count + lane];
        float v_value = working_v[static_cast<ulong>(node) * lane_count + lane];
        u_sum += u_value * u_value;
        v_sum += v_value * v_value;
    }
    partial_sums[(static_cast<ulong>(chunk) * 2) * lane_count + lane] = u_sum;
    partial_sums[(static_cast<ulong>(chunk) * 2 + 1) * lane_count + lane] = v_sum;
}

// ── fit_scale_columns ─────────────────────────────────────────────────────────
// Multiplies each own lane of U by factors[lane] and divides that lane of V by it, over every
// node, which leaves every value U * V the same.
kernel void fit_scale_columns_kernel(
    device float   *working_u  [[ buffer(0) ]],
    device float   *working_v  [[ buffer(1) ]],
    constant uint  &node_count [[ buffer(2) ]],
    constant uint  &lane_count [[ buffer(3) ]],
    constant float *factors    [[ buffer(4) ]],
    uint thread_id [[ thread_position_in_grid ]],
    uint grid_size [[ threads_per_grid ]]
) {
    ulong length = static_cast<ulong>(node_count) * lane_count;
    for (ulong index = thread_id; index < length; index += grid_size) {
        float factor = factors[index % lane_count];
        working_u[index] *= factor;
        working_v[index] /= factor;
    }
}

// The most basis vectors and left slots fit_dot_products_kernel takes: 2 * history + 1 vectors
// for the history length Weight_Matrix uses, and the three slots one quasi-Newton step changes.
#define FIT_MAXIMUM_BASIS_VECTOR_COUNT 64
#define FIT_MAXIMUM_LEFT_SLOT_COUNT 3

// ── fit_dot_products ──────────────────────────────────────────────────────────
// The dot product of each of left_count basis slots (left_slots) with each of the first
// vector_count slots, in one pass over the basis. Per threadgroup: one partial per pair, at
// group_id * left_count * vector_count + left * vector_count + slot.
kernel void fit_dot_products_kernel(
    const device float *basis                [[ buffer(0) ]],
    constant ulong     &length               [[ buffer(1) ]],
    constant uint      &vector_count         [[ buffer(2) ]],
    constant uint      *left_slots           [[ buffer(3) ]],
    constant uint      &left_count           [[ buffer(4) ]],
    device float       *group_dot_products   [[ buffer(5) ]],
    uint thread_id        [[ thread_position_in_grid ]],
    uint grid_size        [[ threads_per_grid ]],
    uint group_id         [[ threadgroup_position_in_grid ]],
    uint simd_lane        [[ thread_index_in_simdgroup ]],
    uint simd_group       [[ simdgroup_index_in_threadgroup ]],
    uint simd_group_count [[ simdgroups_per_threadgroup ]]
) {
    threadgroup float scratch[32];
    uint count = min(vector_count, static_cast<uint>(FIT_MAXIMUM_BASIS_VECTOR_COUNT));
    uint lefts = min(left_count, static_cast<uint>(FIT_MAXIMUM_LEFT_SLOT_COUNT));
    float sums[FIT_MAXIMUM_LEFT_SLOT_COUNT * FIT_MAXIMUM_BASIS_VECTOR_COUNT];
    for (uint pair = 0; pair < lefts * count; ++pair) {
        sums[pair] = 0.0f;
    }
    for (ulong index = thread_id; index < length; index += grid_size) {
        float left_values[FIT_MAXIMUM_LEFT_SLOT_COUNT];
        for (uint left = 0; left < lefts; ++left) {
            left_values[left] = basis[static_cast<ulong>(left_slots[left]) * length + index];
        }
        for (uint slot = 0; slot < count; ++slot) {
            float value = basis[static_cast<ulong>(slot) * length + index];
            for (uint left = 0; left < lefts; ++left) {
                sums[left * count + slot] += left_values[left] * value;
            }
        }
    }
    for (uint pair = 0; pair < lefts * count; ++pair) {
        float group_total = threadgroup_sum(sums[pair], scratch, simd_lane, simd_group, simd_group_count);
        if (simd_group == 0 && simd_lane == 0) {
            group_dot_products[static_cast<ulong>(group_id) * lefts * count + pair] = group_total;
        }
    }
}

// ── fit_reduce_dot_products ───────────────────────────────────────────────────
// Adds fit_dot_products_kernel's group partials into the dot-product matrix of the basis slots
// ([vector_count][vector_count]), one thread per (left slot, slot) pair. The pair's transpose is
// written too unless that slot is itself a left slot, whose own thread writes it.
kernel void fit_reduce_dot_products_kernel(
    const device float *group_dot_products [[ buffer(0) ]],
    constant uint      &group_count        [[ buffer(1) ]],
    constant uint      &vector_count       [[ buffer(2) ]],
    constant uint      *left_slots         [[ buffer(3) ]],
    constant uint      &left_count         [[ buffer(4) ]],
    device float       *products           [[ buffer(5) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    if (thread_id >= left_count * vector_count) {
        return;
    }
    uint left = thread_id / vector_count;
    uint slot = thread_id % vector_count;
    float sum = 0.0f;
    for (uint group = 0; group < group_count; ++group) {
        sum += group_dot_products[(static_cast<ulong>(group) * left_count + left) * vector_count + slot];
    }
    uint left_slot = left_slots[left];
    products[left_slot * vector_count + slot] = sum;
    bool slot_is_left = false;
    for (uint other = 0; other < left_count; ++other) {
        slot_is_left = slot_is_left || left_slots[other] == slot;
    }
    if (!slot_is_left) {
        products[slot * vector_count + left_slot] = sum;
    }
}

// The most history pairs fit_quasi_newton_direction_kernel keeps.
#define FIT_MAXIMUM_HISTORY_LENGTH 31

// ── fit_quasi_newton_direction ────────────────────────────────────────────────
// One thread. Updates the history with the pair just written to slot new_pair_slot (dropping the
// oldest pair when the history is full, and keeping the new one only when its curvature is
// positive), then runs the L-BFGS two-loop recursion on the dot products of the basis slots:
// coefficients[slot] is the weight of basis slot `slot` in the direction -H * gradient. Slots
// 0..history_length-1 hold steps, history_length.. their gradient changes, the last the gradient.
// history[0] is the pair count and history[1..] the pairs' step slots, oldest first. When the
// direction does not descend, the history is cleared and the direction is steepest descent.
// status: [0] the slope along the direction, [1] the first step to try, [2] 1 for steepest
// descent, [3] 1 when the gradient is zero. step_length receives status[1].
kernel void fit_quasi_newton_direction_kernel(
    const device float *products        [[ buffer(0) ]],
    device uint        *history         [[ buffer(1) ]],
    constant uint      &history_length  [[ buffer(2) ]],
    constant uint      &new_pair_slot   [[ buffer(3) ]],
    constant uint      &reset_history   [[ buffer(4) ]],
    device float       *coefficients    [[ buffer(5) ]],
    device float       *status          [[ buffer(6) ]],
    device float       *step_length     [[ buffer(7) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    if (thread_id != 0) {
        return;
    }
    uint vector_count = 2 * history_length + 1;
    uint gradient_slot = 2 * history_length;
    uint count = (reset_history != 0u) ? 0u : history[0];
    if (new_pair_slot < history_length) {
        if (count == history_length) {
            for (uint index = 1; index < count; ++index) {
                history[index] = history[index + 1];
            }
            count -= 1;
        }
        float curvature = products[new_pair_slot * vector_count + history_length + new_pair_slot];
        float change_norm = products[(history_length + new_pair_slot) * vector_count + history_length + new_pair_slot];
        if (!is_not_finite(curvature) && curvature > 1e-10f * change_norm) {
            history[1 + count] = new_pair_slot;
            count += 1;
        }
    }
    history[0] = count;

    for (uint slot = 0; slot < vector_count; ++slot) {
        coefficients[slot] = 0.0f;
    }
    coefficients[gradient_slot] = 1.0f;
    float alpha[FIT_MAXIMUM_HISTORY_LENGTH];
    for (uint countdown = count; countdown > 0; --countdown) {
        uint step_slot = history[countdown];
        uint change_slot = history_length + step_slot;
        float dot_product = 0.0f;
        for (uint slot = 0; slot < vector_count; ++slot) {
            dot_product += coefficients[slot] * products[step_slot * vector_count + slot];
        }
        alpha[countdown - 1] = dot_product / products[step_slot * vector_count + change_slot];
        coefficients[change_slot] -= alpha[countdown - 1];
    }
    if (count > 0) {
        uint newest = history[count];
        uint newest_change = history_length + newest;
        float gamma = products[newest * vector_count + newest_change] / products[newest_change * vector_count + newest_change];
        for (uint slot = 0; slot < vector_count; ++slot) {
            coefficients[slot] *= gamma;
        }
    }
    for (uint index = 0; index < count; ++index) {
        uint step_slot = history[index + 1];
        uint change_slot = history_length + step_slot;
        float dot_product = 0.0f;
        for (uint slot = 0; slot < vector_count; ++slot) {
            dot_product += coefficients[slot] * products[change_slot * vector_count + slot];
        }
        float beta = dot_product / products[step_slot * vector_count + change_slot];
        coefficients[step_slot] += alpha[index] - beta;
    }

    float slope = 0.0f;
    for (uint slot = 0; slot < vector_count; ++slot) {
        slope -= coefficients[slot] * products[gradient_slot * vector_count + slot];
    }
    bool steepest = (count == 0u);
    if (!(slope < 0.0f) || is_not_finite(slope)) {
        count = 0;
        history[0] = 0;
        for (uint slot = 0; slot < vector_count; ++slot) {
            coefficients[slot] = 0.0f;
        }
        coefficients[gradient_slot] = 1.0f;
        slope = -products[gradient_slot * vector_count + gradient_slot];
        steepest = true;
    }
    for (uint slot = 0; slot < vector_count; ++slot) {
        coefficients[slot] = -coefficients[slot];
    }
    float gradient_norm_squared = products[gradient_slot * vector_count + gradient_slot];
    float first_step = steepest ? min(1.0f, 1.0f / sqrt(max(gradient_norm_squared, 1e-30f))) : 1.0f;
    status[0] = slope;
    status[1] = first_step;
    status[2] = steepest ? 1.0f : 0.0f;
    status[3] = (slope < 0.0f) ? 0.0f : 1.0f;
    step_length[0] = first_step;
}

// ── fit_linear_combination ────────────────────────────────────────────────────
// output = the sum over the first vector_count basis slots of coefficients[slot] * that slot.
kernel void fit_linear_combination_kernel(
    const device float *basis        [[ buffer(0) ]],
    constant ulong     &length       [[ buffer(1) ]],
    constant uint      &vector_count [[ buffer(2) ]],
    constant float     *coefficients [[ buffer(3) ]],
    device float       *output       [[ buffer(4) ]],
    uint thread_id [[ thread_position_in_grid ]],
    uint grid_size [[ threads_per_grid ]]
) {
    for (ulong index = thread_id; index < length; index += grid_size) {
        float sum = 0.0f;
        for (uint slot = 0; slot < vector_count; ++slot) {
            sum += coefficients[slot] * basis[static_cast<ulong>(slot) * length + index];
        }
        output[index] = sum;
    }
}
