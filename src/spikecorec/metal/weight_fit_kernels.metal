#include <metal_stdlib>
using namespace metal;

// Kernels for Weight_Matrix's two candidate fits (masked alternating least squares and limited-
// memory BFGS). A fit works on one plane's own lanes, copied out of U and V into dense working
// arrays [node][lane_count]. Edge e runs from source_of_edge(e) to edge_target[e]; the plane's own
// lanes must hold residual_target[e] there (NaN for a value no basis can hold), and
// edge_weight_of(residual_target[e]) scales its squared error.
//
// The rows a fit may move are moved_rows: the sources first (moved_source_count of them), then
// the targets. A packed vector holds one value per moved row and lane, at slot * lane_count + lane.
//
// Reductions write one partial per threadgroup and the host adds the partials up. Each thread
// walks its indices with a stride of the grid size, so the host bounds how many partials there are.

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

// The source of an edge: the row whose ordinals hold it, found by binary search of the row
// offsets, since no array of edge sources is kept.
inline uint source_of_edge(const device uint *outgoing_offset, uint node_count, uint edge) {
    uint low = 0;
    uint high = node_count;
    while (low + 1 < high) {
        uint middle = low + (high - low) / 2;
        if (outgoing_offset[middle] <= edge) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return low;
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

// ── fit_edge_residual ─────────────────────────────────────────────────────────
// Per touched edge: the residual of the own lanes against residual_target, stored weighted for the
// gradient. Per threadgroup: the sum of 0.5 * weight * residual^2 and the worst relative error,
// |residual| * sqrt(weight). A non-finite residual counts as an infinite loss and error. When
// every edge is touched, the index is the edge and touched_edges is not read.
kernel void fit_edge_residual_kernel(
    const device float *working_u             [[ buffer(0) ]],
    const device float *working_v             [[ buffer(1) ]],
    const device uint  *outgoing_offset       [[ buffer(2) ]],
    const device uint  *edge_target           [[ buffer(3) ]],
    const device uint  *touched_edges         [[ buffer(4) ]],
    const device float *residual_target       [[ buffer(5) ]],
    constant uint      &touched_edge_count    [[ buffer(6) ]],
    constant uint      &every_edge_is_touched [[ buffer(7) ]],
    constant uint      &lane_count            [[ buffer(8) ]],
    constant uint      &node_count            [[ buffer(9) ]],
    device float       *weighted_residual     [[ buffer(10) ]],
    device float       *group_loss            [[ buffer(11) ]],
    device float       *group_worst_error     [[ buffer(12) ]],
    uint thread_id        [[ thread_position_in_grid ]],
    uint grid_size        [[ threads_per_grid ]],
    uint group_id         [[ threadgroup_position_in_grid ]],
    uint simd_lane        [[ thread_index_in_simdgroup ]],
    uint simd_group       [[ simdgroup_index_in_threadgroup ]],
    uint simd_group_count [[ simdgroups_per_threadgroup ]]
) {
    threadgroup float scratch[32];
    float loss = 0.0f;
    float worst_error = 0.0f;
    for (uint index = thread_id; index < touched_edge_count; index += grid_size) {
        uint edge = (every_edge_is_touched != 0u) ? index : touched_edges[index];
        const device float *source_row =
            working_u + static_cast<ulong>(source_of_edge(outgoing_offset, node_count, edge)) * lane_count;
        const device float *target_row = working_v + static_cast<ulong>(edge_target[edge]) * lane_count;
        float value = 0.0f;
        for (uint lane = 0; lane < lane_count; ++lane) {
            value += source_row[lane] * target_row[lane];
        }
        float target = residual_target[edge];
        float residual = value - held_target(target);
        float weight = edge_weight_of(target);
        weighted_residual[edge] = weight * residual;

        float edge_loss = 0.5f * weight * residual * residual;
        float edge_error = fabs(residual) * sqrt(weight);
        if (is_not_finite(edge_loss) || is_not_finite(edge_error)) {
            edge_loss = INFINITY;
            edge_error = INFINITY;
        }
        loss += edge_loss;
        worst_error = max(worst_error, edge_error);
    }

    float group_total = threadgroup_sum(loss, scratch, simd_lane, simd_group, simd_group_count);
    float group_largest = threadgroup_maximum(worst_error, scratch, simd_lane, simd_group, simd_group_count);
    if (simd_group == 0 && simd_lane == 0) {
        group_loss[group_id] = group_total;
        group_worst_error[group_id] = group_largest;
    }
}

// ── fit_row_solve ─────────────────────────────────────────────────────────────
// Masked alternating least squares, one SIMD group of FIT_ROW_SOLVE_THREADS_PER_ROW threads per row
// of one side: the row x minimizing its edges' weighted squared error plus ridge * |x|^2, given the
// other side's rows. That is the least squares problem [W^(1/2) A; ridge^(1/2) I] x = [W^(1/2) b; 0],
// with one row of A (the other end's row) and of b (the target) per edge. It is solved by QR: R
// starts as ridge^(1/2) I, and Givens rotations fold each edge's row into R and Q^T b in turn; then
// R x = Q^T b by back substitution. QR never forms A^T A, so it keeps its accuracy on the nearly
// singular rows of structured networks, where f32 normal equations lose it.
//
// Every thread of the group computes each rotation, from the same values, and applies it to its
// own columns (column c belongs to thread c % FIT_ROW_SOLVE_THREADS_PER_ROW); the back
// substitution's sums are split the same way. Apple GPUs run SIMD groups of 32 threads, so a group
// is one row.
//
// Row slot s is rows[first_row + s]; its edges are edge_offset[row] up to edge_offset[row + 1], as
// ordinals when edges_are_listed is 0 and as positions in edge_list otherwise; the other end is the
// edge's target for the first and its source for the second. Slot s keeps R
// (lane_count^2, row-major), Q^T b and the edge row being folded (lane_count each) in device
// scratch at s * lane_count^2 and s * 2 * lane_count. A row whose R has a zero on its diagonal is
// left as it was.
constant uint FIT_ROW_SOLVE_THREADS_PER_ROW = 32;

kernel void fit_row_solve_kernel(
    device float       *solved_side         [[ buffer(0) ]],
    const device float *fixed_side          [[ buffer(1) ]],
    const device uint  *rows                [[ buffer(2) ]],
    constant uint      &first_row           [[ buffer(3) ]],
    constant uint      &row_count           [[ buffer(4) ]],
    const device uint  *edge_offset         [[ buffer(5) ]],
    const device uint  *edge_list           [[ buffer(6) ]],
    constant uint      &edges_are_listed    [[ buffer(7) ]],
    const device uint  *edge_target         [[ buffer(8) ]],
    const device uint  *outgoing_offset     [[ buffer(9) ]],
    constant uint      &node_count          [[ buffer(10) ]],
    const device float *residual_target     [[ buffer(11) ]],
    constant uint      &lane_count          [[ buffer(12) ]],
    constant float     &ridge               [[ buffer(13) ]],
    device float       *triangle_scratch    [[ buffer(14) ]],
    device float       *vector_scratch      [[ buffer(15) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
    uint slot = thread_id / FIT_ROW_SOLVE_THREADS_PER_ROW;
    uint column_thread = thread_id % FIT_ROW_SOLVE_THREADS_PER_ROW;
    if (slot >= row_count) {
        return;
    }
    uint row = rows[first_row + slot];
    device float *triangle = triangle_scratch + static_cast<ulong>(slot) * lane_count * lane_count;
    device float *projected = vector_scratch + static_cast<ulong>(slot) * 2 * lane_count;   // Q^T b
    device float *edge_row = projected + lane_count;

    float root_ridge = sqrt(ridge);
    for (uint first = 0; first < lane_count; ++first) {
        for (uint second = column_thread; second < lane_count; second += FIT_ROW_SOLVE_THREADS_PER_ROW) {
            triangle[first * lane_count + second] = (first == second) ? root_ridge : 0.0f;
        }
    }
    for (uint lane = column_thread; lane < lane_count; lane += FIT_ROW_SOLVE_THREADS_PER_ROW) {
        projected[lane] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_device);

    for (uint position = edge_offset[row]; position < edge_offset[row + 1]; ++position) {
        uint edge = (edges_are_listed != 0u) ? edge_list[position] : position;
        uint other_end = (edges_are_listed != 0u) ? source_of_edge(outgoing_offset, node_count, edge) : edge_target[edge];
        const device float *other_row = fixed_side + static_cast<ulong>(other_end) * lane_count;
        float target = residual_target[edge];
        float root_weight = sqrt(edge_weight_of(target));
        for (uint lane = column_thread; lane < lane_count; lane += FIT_ROW_SOLVE_THREADS_PER_ROW) {
            edge_row[lane] = root_weight * other_row[lane];
        }
        float edge_target = root_weight * held_target(target);
        simdgroup_barrier(mem_flags::mem_device);

        // Rotate the edge's row into R one column at a time, zeroing it. Every thread reads the
        // pivot's values before thread 0 writes the pivot's diagonal and Q^T b entry.
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
            for (uint column = pivot + 1 + column_thread; column < lane_count; column += FIT_ROW_SOLVE_THREADS_PER_ROW) {
                float kept = pivot_row[column];
                float folded = edge_row[column];
                pivot_row[column] = cosine * kept + sine * folded;
                edge_row[column] = cosine * folded - sine * kept;
            }
            float kept_target = projected[pivot];
            float rotated_target = cosine * kept_target + sine * edge_target;
            edge_target = cosine * edge_target - sine * kept_target;
            simdgroup_barrier(mem_flags::mem_device);
            if (column_thread == 0) {
                pivot_row[pivot] = radius;
                projected[pivot] = rotated_target;
            }
            simdgroup_barrier(mem_flags::mem_device);
        }
    }

    // R x = Q^T b, in place in projected: each row's sum is split over the group's threads.
    for (uint countdown = lane_count; countdown > 0; --countdown) {
        uint first = countdown - 1;
        float diagonal = triangle[first * lane_count + first];
        if (!(diagonal != 0.0f) || is_not_finite(diagonal)) {
            return;
        }
        float partial_sum = 0.0f;
        for (uint inner = first + 1 + column_thread; inner < lane_count; inner += FIT_ROW_SOLVE_THREADS_PER_ROW) {
            partial_sum += triangle[first * lane_count + inner] * projected[inner];
        }
        float sum = simd_sum(partial_sum);
        if (column_thread == 0) {
            projected[first] = (projected[first] - sum) / diagonal;
        }
        simdgroup_barrier(mem_flags::mem_device);
    }

    device float *solved_row = solved_side + static_cast<ulong>(row) * lane_count;
    for (uint lane = column_thread; lane < lane_count; lane += FIT_ROW_SOLVE_THREADS_PER_ROW) {
        solved_row[lane] = projected[lane];
    }
}

// ── fit_row_gradient ──────────────────────────────────────────────────────────
// The objective's gradient at every moved row and lane, packed: the row's edges' weighted
// residuals times the other end's lane, plus ridge * the row's lane. Per threadgroup: the sum of
// 0.5 * ridge * value^2 (the ridge part of the objective) and of gradient * direction (the slope
// along direction).
kernel void fit_row_gradient_kernel(
    const device float *working_u          [[ buffer(0) ]],
    const device float *working_v          [[ buffer(1) ]],
    const device uint  *moved_rows         [[ buffer(2) ]],
    constant uint      &moved_source_count [[ buffer(3) ]],
    constant uint      &moved_row_count    [[ buffer(4) ]],
    const device uint  *outgoing_offset    [[ buffer(5) ]],
    const device uint  *incoming_offset    [[ buffer(6) ]],
    const device uint  *incoming_edge      [[ buffer(7) ]],
    constant uint      &node_count         [[ buffer(8) ]],
    const device uint  *edge_target        [[ buffer(9) ]],
    const device float *weighted_residual  [[ buffer(10) ]],
    constant uint      &lane_count         [[ buffer(11) ]],
    constant float     &ridge              [[ buffer(12) ]],
    const device float *direction          [[ buffer(13) ]],
    device float       *gradient           [[ buffer(14) ]],
    device float       *group_ridge_term   [[ buffer(15) ]],
    device float       *group_slope        [[ buffer(16) ]],
    uint thread_id        [[ thread_position_in_grid ]],
    uint grid_size        [[ threads_per_grid ]],
    uint group_id         [[ threadgroup_position_in_grid ]],
    uint simd_lane        [[ thread_index_in_simdgroup ]],
    uint simd_group       [[ simdgroup_index_in_threadgroup ]],
    uint simd_group_count [[ simdgroups_per_threadgroup ]]
) {
    threadgroup float scratch[32];
    float ridge_term = 0.0f;
    float slope = 0.0f;
    ulong length = static_cast<ulong>(moved_row_count) * lane_count;
    for (ulong index = thread_id; index < length; index += grid_size) {
        uint slot = static_cast<uint>(index / lane_count);
        uint lane = static_cast<uint>(index % lane_count);
        uint row = moved_rows[slot];
        bool is_source = slot < moved_source_count;
        float value = (is_source ? working_u : working_v)[static_cast<ulong>(row) * lane_count + lane];

        float sum = ridge * value;
        if (is_source) {
            for (uint edge = outgoing_offset[row]; edge < outgoing_offset[row + 1]; ++edge) {
                sum += weighted_residual[edge] * working_v[static_cast<ulong>(edge_target[edge]) * lane_count + lane];
            }
        } else {
            for (uint position = incoming_offset[row]; position < incoming_offset[row + 1]; ++position) {
                uint edge = incoming_edge[position];
                uint source = source_of_edge(outgoing_offset, node_count, edge);
                sum += weighted_residual[edge] * working_u[static_cast<ulong>(source) * lane_count + lane];
            }
        }
        gradient[index] = sum;
        ridge_term += 0.5f * ridge * value * value;
        slope += sum * direction[index];
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
