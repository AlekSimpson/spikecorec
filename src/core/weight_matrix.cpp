//
// Created by Alek Simpson on 5/30/26.
//

#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <limits>
#include <cstdint>
#include <new>

#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;

static constexpr s64 DEFAULT_WEIGHT_RANK = 64;
static constexpr u32 WEIGHT_MATRIX_SAVE_MAGIC = 0x574D5458;

const vector<vector<s32>> &WeightMatrix::validate_network(const vector<vector<s32>> &network) {
    if (network.empty()) {
        log::throw_invalid_argument(log::logger(),
            fmt::format("WeightMatrix: network must have at least one neuron (got {})", network.size()));
    }
    return network;
}

WeightMatrix::WeightMatrix(
    const vector<vector<s32>> &network,
    s64 rank,
    bool check_indexing,
    s64 max_neighbor_count,
    s64 weight_seed
)
    : k2tree(*K2Tree::from_adjacency_list(validate_network(network), (s32)network.size()))
    , node_count((s64)network.size())
    , max_neighbor_count(0)
    , rank(0)
    , rank_float4_stride(0)
    , constant_weight(0.0f)
    , check_indexing(check_indexing)
    , using_constant_weight(false)
{
    this->max_neighbor_count = max_neighbor_count;
    if (max_neighbor_count == -1) {
        s64 longest_row = 0;
        for (s64 node_index = 0; node_index < node_count; node_index++) {
            longest_row = max(longest_row, (s64)network[node_index].size());
        }
        this->max_neighbor_count = longest_row;
    }

    this->rank = (rank > 0) ? rank : min(DEFAULT_WEIGHT_RANK, node_count);
    rank_float4_stride = (this->rank + 3) / 4;

    // Must match MAX_RANK_FLOAT4_STRIDE in kernels.cu / kernels.metal (both == 64).
    // Exceeding it causes out-of-bounds writes into fixed-size kernel arrays (SC-13).
    if (rank_float4_stride > MAX_RANK_FLOAT4_STRIDE) {
        log::throw_invalid_argument(log::logger(),
            fmt::format("WeightMatrix: rank_float4_stride ({}) exceeds the GPU kernel "
                        "limit MAX_RANK_FLOAT4_STRIDE ({}); reduce rank to at most {} "
                        "(got rank={})",
                        rank_float4_stride, MAX_RANK_FLOAT4_STRIDE,
                        MAX_RANK_FLOAT4_STRIDE * 4, this->rank));
    }

    // allocate U and V in unified memory — shape [node_count][rank_float4_stride]
    usize matrix_byte_size = (usize)node_count * (usize)rank_float4_stride * sizeof(float4);
    U_matrix = allocate<float4>(matrix_byte_size);
    V_matrix = allocate<float4>(matrix_byte_size);

    // initialize with independent random normal values (mean=0, std=1).
    // weight_seed >= 0 gives reproducible weights (deterministic runs/tests);
    // weight_seed < 0 falls back to a nondeterministic hardware-entropy seed.
    unsigned resolved_weight_seed = (weight_seed >= 0)
        ? static_cast<unsigned>(weight_seed)
        : random_device{}();
    mt19937 rng(resolved_weight_seed);
    normal_distribution<f32> normal_dist(0.0f, 1.0f);
    float4* u_data = U_matrix.get_contents();
    float4* v_data = V_matrix.get_contents();
    s64 total_float4_element_count = node_count * rank_float4_stride;
    for (s64 element_index = 0; element_index < total_float4_element_count; element_index++) {
        u_data[element_index] = {normal_dist(rng), normal_dist(rng), normal_dist(rng), normal_dist(rng)};
        v_data[element_index] = {normal_dist(rng), normal_dist(rng), normal_dist(rng), normal_dist(rng)};
    }

    prefetch_to_gpu(U_matrix, matrix_byte_size);
    prefetch_to_gpu(V_matrix, matrix_byte_size);

    // DEFAULT_MATRIX_INDEX's coefficient vector: every lane literally 1.0f (see
    // allocate_coefficient_vector — called here with an empty logical_coefficients,
    // so every lane falls into its literal-1.0f branch), so get()/neighbor_weights()
    // stay bit-compatible with the pre-shared-basis dot(U,V) (ticket #52/D2, see §2 of
    // the design memo for why this must be a literal, not a computed value).
    coefficient_vectors.push_back(allocate_coefficient_vector({}));

    // DEFAULT_MATRIX_INDEX's sparse delta buffer (ticket #53/D3): allocated to
    // node_count*max_neighbor_count elements and zero-filled, parallel to the
    // coefficient vector pushed just above — a freshly-constructed WeightMatrix
    // has no pending Sk updates for any matrix.
    sparse_delta_buffers.push_back(allocate_sparse_delta_buffer());
    sparse_delta_touched.push_back(false);

    log::logger().debug("WeightMatrix constructed: node_count={} rank={} rank_float4_stride={} "
                        "max_neighbor_count={} weight_seed={} check_indexing={}",
                        node_count, this->rank, rank_float4_stride, this->max_neighbor_count,
                        resolved_weight_seed, check_indexing);
}

WeightMatrix::~WeightMatrix() {
    deallocate(std::move(U_matrix));
    deallocate(std::move(V_matrix));
    for (auto &coefficient_vector : coefficient_vectors) {
        deallocate(std::move(coefficient_vector));
    }
    for (auto &delta_buffer : sparse_delta_buffers) {
        deallocate(std::move(delta_buffer));
    }
}

WeightMatrix &WeightMatrix::operator=(WeightMatrix &&other) noexcept {
    if (this == &other) {
        return *this;
    }

    // Deallocate this instance's own live GPU buffers before moving the incoming
    // object's state in — see the header comment on this operator for why the
    // implicitly-defaulted version aborts instead.
    deallocate(std::move(U_matrix));
    deallocate(std::move(V_matrix));
    for (auto &coefficient_vector : coefficient_vectors) {
        deallocate(std::move(coefficient_vector));
    }
    coefficient_vectors.clear();
    // sparse_delta_buffers (ticket #53/D3) is now GpuPointer-backed too — same
    // deallocate-then-clear hazard-avoidance as coefficient_vectors above.
    for (auto &delta_buffer : sparse_delta_buffers) {
        deallocate(std::move(delta_buffer));
    }
    sparse_delta_buffers.clear();

    // k2tree is a K2Tree sub-object (not a pointer), and K2Tree's own defaulted
    // move-assignment operator has the identical GpuPointer-assert problem as
    // WeightMatrix's did — assigning into it here would abort just the same.
    // K2Tree's move CONSTRUCTOR has no such issue (GpuPointer's move constructor
    // never asserts), so destroy this instance's k2tree and move-construct the
    // incoming one in its place instead of going through operator=.
    k2tree.~K2Tree();
    new (&k2tree) K2Tree(std::move(other.k2tree));

    U_matrix = std::move(other.U_matrix);
    V_matrix = std::move(other.V_matrix);
    coefficient_vectors = std::move(other.coefficient_vectors);
    sparse_delta_buffers = std::move(other.sparse_delta_buffers);
    // sparse_delta_touched is plain host memory (vector<bool>) — an ordinary
    // vector move-assignment, no GPU-asserting-move hazard to work around here.
    sparse_delta_touched = std::move(other.sparse_delta_touched);
    node_count = other.node_count;
    max_neighbor_count = other.max_neighbor_count;
    rank = other.rank;
    rank_float4_stride = other.rank_float4_stride;
    constant_weight = other.constant_weight;
    check_indexing = other.check_indexing;
    using_constant_weight = other.using_constant_weight;

    return *this;
}

bool spikecorec::can_safely_cast_s64_to_s32(s64 value) {
    return value >= std::numeric_limits<s32>::min() &&
           value <= std::numeric_limits<s32>::max();
}

s64 WeightMatrix::get_neighbors(s64 node_index, s32 *output_buffer) const {
    if (!can_safely_cast_s64_to_s32(node_index) ||
        !check_index_inbounds((s32)node_index)) {
        return 0;
    }

    return k2tree.get_neighbors((s32)node_index, output_buffer, max_neighbor_count);
}

void WeightMatrix::set_constant_weight(f32 value) {
    log::logger().debug("set_constant_weight: value={}", value);
    // U and V are filled across every rank_float4_stride*4 lane below, and
    // get()/neighbor_weights_kernel sum over that same effective lane count (not
    // just the logical `rank` lanes) — so the scale factor must be derived from
    // rank_float4_stride*4 too, or get() overshoots `value` whenever rank isn't a
    // multiple of 4 (bug found by the SC-52/D2 test-hardening pass; fixed here).
    f32 effective_lane_count = (f32)(rank_float4_stride * 4);
    f32 scale = (value != 0.0f) ? sqrtf(fabsf(value) / effective_lane_count) : 0.0f;
    float4 u_fill = {scale, scale, scale, scale};
    f32 v_fill_value = (value >= 0.0f) ? scale : -scale;
    float4 v_fill = {v_fill_value, v_fill_value, v_fill_value, v_fill_value};

    float4 *u_data = U_matrix.get_contents();
    float4 *v_data = V_matrix.get_contents();
    s64 total_float4_element_count = node_count * rank_float4_stride;
    for (s64 index = 0; index < total_float4_element_count; ++index) {
        u_data[index] = u_fill;
        v_data[index] = v_fill;
    }
    constant_weight = value;
    using_constant_weight = true;
}

bool WeightMatrix::check_index_inbounds(s32 node_index) const {
    return (check_indexing &&
            node_index >= 0 && node_index < node_count);
}

bool WeightMatrix::check_index_inbounds(s32 source, s32 target) const {
    return (check_indexing &&
            source >= 0 && source < node_count &&
            target >= 0 && target < node_count);
}

void WeightMatrix::validate_matrix_index(s64 matrix_index) const {
    if (matrix_index < 0 || matrix_index >= (s64)coefficient_vectors.size()) {
        log::throw_invalid_argument(log::logger(),
            fmt::format("WeightMatrix: matrix_index {} out of range [0, {})",
                        matrix_index, coefficient_vectors.size()));
    }
}

GpuPointer<f32> WeightMatrix::allocate_coefficient_vector(const vector<f32> &logical_coefficients) const {
    s64 effective_lane_count = rank_float4_stride * 4;
    GpuPointer<f32> coefficient_vector = allocate<f32>((usize)effective_lane_count * sizeof(f32));
    f32 *coefficient_data = coefficient_vector.get_contents();
    for (s64 lane_index = 0; lane_index < effective_lane_count; ++lane_index) {
        coefficient_data[lane_index] = (lane_index < (s64)logical_coefficients.size())
            ? logical_coefficients[(usize)lane_index]
            : 1.0f;
    }
    return coefficient_vector;
}

// The U*Ck*V reconstruction. For DEFAULT_MATRIX_INDEX, coefficient_values[...] is
// exactly the literal 1.0f in every lane (see allocate_coefficient_vector), so each
// `lane_coefficients[...] * v_row[...].x` below is `1.0f * v.x`, which IEEE-754
// guarantees is bit-identical to `v.x` itself — making this expression, term for
// term and in the same left-to-right accumulation order, bit-identical to the
// pre-D2 `u.x * v.x + u.y * v.y + ...` dot product it replaces (ticket #52/D2,
// see the design memo §2 for why this inline placement — not a separate
// "reweight V, then dot" pass — is what makes that guarantee hold).
f32 WeightMatrix::reconstruct_entry(s32 source_node, s32 target_node, const f32 *coefficient_values) const {
    const float4 *u_row = U_matrix.get_contents() + source_node * rank_float4_stride;
    const float4 *v_row = V_matrix.get_contents() + target_node * rank_float4_stride;
    f32 dot_product = 0.0f;
    for (s64 float4_index = 0; float4_index < rank_float4_stride; ++float4_index) {
        const f32 *lane_coefficients = coefficient_values + float4_index * 4;
        dot_product += u_row[float4_index].x * (lane_coefficients[0] * v_row[float4_index].x)
                     + u_row[float4_index].y * (lane_coefficients[1] * v_row[float4_index].y)
                     + u_row[float4_index].z * (lane_coefficients[2] * v_row[float4_index].z)
                     + u_row[float4_index].w * (lane_coefficients[3] * v_row[float4_index].w);
    }
    return dot_product;
}

GpuPointer<f32> WeightMatrix::allocate_sparse_delta_buffer() const {
    s64 element_count = node_count * max_neighbor_count;
    if (element_count <= 0) {
        // No representable neighbor slots at all (e.g. an edge-free network) —
        // avoid a zero-byte GPU allocation; nothing can ever be accumulated into
        // this matrix's Sk, so a null GpuPointer is never read or deallocated.
        return GpuPointer<f32>();
    }
    GpuPointer<f32> delta_buffer = allocate<f32>((usize)element_count * sizeof(f32));
    // allocate<f32> does not zero-initialize (unlike coefficient_vectors/U/V,
    // which the constructor explicitly fills every lane of) — "untouched" must
    // mean all-zero, so zero it explicitly here.
    memset(delta_buffer.get_contents(), 0, (usize)element_count * sizeof(f32));
    return delta_buffer;
}

optional<s64> WeightMatrix::find_neighbor_slot(s32 source_node, s32 target_node) const {
    vector<s32> neighbor_buffer((usize)max_neighbor_count);
    s64 degree = k2tree.get_neighbors(source_node, neighbor_buffer.data(), max_neighbor_count);
    for (s64 slot = 0; slot < degree; ++slot) {
        if (neighbor_buffer[(usize)slot] == target_node) {
            return slot;
        }
    }
    return nullopt;
}

f32 WeightMatrix::lookup_sparse_delta(s64 matrix_index, s32 source_node, s32 target_node) const {
    if (!sparse_delta_touched[(usize)matrix_index]) {
        // Fast path: no slot search at all for a Sk-free matrix (see arch §4.3
        // "Precise form" — value = U*Ck*V^T + Sk[i,j], and Sk[i,j] is 0 when
        // nothing has ever been accumulated). This is also what keeps
        // get()/get_for_matrix() bit-for-bit unchanged for DEFAULT_MATRIX_INDEX
        // (ticket #52's regression guarantee).
        return 0.0f;
    }
    optional<s64> slot = find_neighbor_slot(source_node, target_node);
    if (!slot.has_value()) {
        return 0.0f;
    }
    const f32 *delta_data = sparse_delta_buffers[(usize)matrix_index].get_contents();
    return delta_data[source_node * max_neighbor_count + *slot];
}

void WeightMatrix::apply_sparse_delta_overlay(f32 *output_weights, s64 matrix_index) const {
    if (!sparse_delta_touched[(usize)matrix_index]) {
        return;
    }

    // sparse_delta_buffers[matrix_index] and output_weights share the exact same
    // node_count * max_neighbor_count, row-major-by-source-node, same-slot-order
    // shape (see the header comment on sparse_delta_buffers), so overlaying Sk is
    // a plain element-wise add — no neighbor walk needed. Slots beyond a node's
    // real degree, and any real edge never accumulated into, are exactly 0.0f.
    const f32 *delta_data = sparse_delta_buffers[(usize)matrix_index].get_contents();
    s64 total_pair_count = node_count * max_neighbor_count;
    for (s64 pair_index = 0; pair_index < total_pair_count; ++pair_index) {
        output_weights[(usize)pair_index] += delta_data[pair_index];
    }
}

f32 WeightMatrix::get(s32 source_node, s32 target_node) const {
    log::logger().trace("get: source_node={} target_node={}", source_node, target_node);
    if (!check_index_inbounds(source_node, target_node)) {
        return 0.0;
    }

    f32 reconstructed = reconstruct_entry(source_node, target_node,
                              coefficient_vectors[(usize)DEFAULT_MATRIX_INDEX].get_contents());
    return reconstructed + lookup_sparse_delta(DEFAULT_MATRIX_INDEX, source_node, target_node);
}

f32 WeightMatrix::get_for_matrix(s32 source_node, s32 target_node, s64 matrix_index) const {
    log::logger().trace("get_for_matrix: source_node={} target_node={} matrix_index={}",
                        source_node, target_node, matrix_index);
    validate_matrix_index(matrix_index);
    if (!check_index_inbounds(source_node, target_node)) {
        return 0.0;
    }

    f32 reconstructed = reconstruct_entry(source_node, target_node,
                              coefficient_vectors[(usize)matrix_index].get_contents());
    return reconstructed + lookup_sparse_delta(matrix_index, source_node, target_node);
}

s64 WeightMatrix::add_coefficient_vector(const vector<f32> &coefficients) {
    if ((s64)coefficients.size() != rank) {
        log::throw_invalid_argument(log::logger(),
            fmt::format("WeightMatrix::add_coefficient_vector: coefficients must have exactly "
                        "rank ({}) elements (got {})", rank, coefficients.size()));
    }

    s64 new_matrix_index = (s64)coefficient_vectors.size();
    coefficient_vectors.push_back(allocate_coefficient_vector(coefficients));
    // Parallel, untouched (all-zero) Sk for the new matrix (ticket #53/D3) — see
    // sparse_delta_buffers' header comment for why this stays indexed identically
    // to coefficient_vectors.
    sparse_delta_buffers.push_back(allocate_sparse_delta_buffer());
    sparse_delta_touched.push_back(false);
    log::logger().debug("add_coefficient_vector: matrix_index={}", new_matrix_index);
    return new_matrix_index;
}

void WeightMatrix::set_coefficient_vector(s64 matrix_index, const vector<f32> &coefficients) {
    validate_matrix_index(matrix_index);
    if ((s64)coefficients.size() != rank) {
        log::throw_invalid_argument(log::logger(),
            fmt::format("WeightMatrix::set_coefficient_vector: coefficients must have exactly "
                        "rank ({}) elements (got {})", rank, coefficients.size()));
    }

    s64 effective_lane_count = rank_float4_stride * 4;
    f32 *coefficient_data = coefficient_vectors[(usize)matrix_index].get_contents();
    for (s64 lane_index = 0; lane_index < effective_lane_count; ++lane_index) {
        coefficient_data[lane_index] = (lane_index < (s64)coefficients.size())
            ? coefficients[(usize)lane_index]
            : 1.0f;
    }
    log::logger().debug("set_coefficient_vector: matrix_index={}", matrix_index);
}

s64 WeightMatrix::matrix_count() const {
    return (s64)coefficient_vectors.size();
}

void WeightMatrix::dispatch_neighbor_weights(f32 *output_weights, const f32 *coefficient_values) const {
    s64 total_pair_count = node_count * max_neighbor_count;
    if (total_pair_count <= 0) return;

    // output_weights is caller-owned host memory (e.g. std::vector::data())
    GpuPointer<f32> device_weights = allocate<f32>((usize)total_pair_count * sizeof(f32));
    gpu_neighbor_weights(
        U_matrix.get_contents(),
        V_matrix.get_contents(),
        k2tree.internal_node_words.get_contents(),
        k2tree.leaf_node_words.get_contents(),
        k2tree.rank_superblock_table.get_contents(),
        k2tree.rank_subblock_table.get_contents(),
        k2tree.branching_factor,
        k2tree.superblock_size_words,
        k2tree.padded_node_count,
        k2tree.tree_height,
        k2tree.internal_bit_count,
        node_count,
        max_neighbor_count,
        rank_float4_stride,
        coefficient_values,
        device_weights.get_contents()
    );
    synchronize_gpu_work();
    prefetch_to_cpu(device_weights, (usize)total_pair_count * sizeof(f32));
    memcpy(output_weights, device_weights.get_contents(), (usize)total_pair_count * sizeof(f32));
    deallocate(std::move(device_weights));
}

void WeightMatrix::neighbor_weights(f32 *output_weights) const {
    dispatch_neighbor_weights(output_weights, coefficient_vectors[(usize)DEFAULT_MATRIX_INDEX].get_contents());
    apply_sparse_delta_overlay(output_weights, DEFAULT_MATRIX_INDEX);
}

void WeightMatrix::neighbor_weights_for_matrix(f32 *output_weights, s64 matrix_index) const {
    validate_matrix_index(matrix_index);
    dispatch_neighbor_weights(output_weights, coefficient_vectors[(usize)matrix_index].get_contents());
    apply_sparse_delta_overlay(output_weights, matrix_index);
}

void WeightMatrix::accumulate_edge_delta(s64 matrix_index, s32 source_node, s32 target_node, f32 delta) {
    log::logger().trace("accumulate_edge_delta: matrix_index={} source_node={} target_node={} delta={}",
                        matrix_index, source_node, target_node, delta);
    validate_matrix_index(matrix_index);

    // Deliberately does not reuse check_index_inbounds()/check_indexing here: those
    // exist to let the hot, edge-unrestricted get()/update() paths skip bounds work
    // when the caller has already guaranteed valid indices, and (as written)
    // check_index_inbounds always reports "out of bounds" whenever check_indexing is
    // false. accumulate_edge_delta is a new, deliberately edge-scoped operation
    // (loadedge/accedge in the IR spec only ever address real edges) with its own
    // contract, independent of that pre-existing knob: a bad (source_node,
    // target_node) here is always a caller bug worth surfacing loudly, not a
    // performance-sensitive check to toggle off.
    bool in_bounds = source_node >= 0 && source_node < node_count &&
                      target_node >= 0 && target_node < node_count;
    if (!in_bounds) {
        log::throw_invalid_argument(log::logger(),
            fmt::format("WeightMatrix::accumulate_edge_delta: (source_node={}, target_node={}) "
                        "out of bounds for node_count={}", source_node, target_node, node_count));
    }
    if (!k2tree.adjacent(source_node, target_node)) {
        log::throw_invalid_argument(log::logger(),
            fmt::format("WeightMatrix::accumulate_edge_delta: (source_node={}, target_node={}) "
                        "is not a real edge — accedge/loadedge are edge-scoped operations",
                        source_node, target_node));
    }

    // Real k^2-tree edges are normally representable within max_neighbor_count
    // (the array is sized to the adjacency's own bound), so a slot is expected to
    // always be found here. The one exception is an explicit max_neighbor_count
    // override smaller than a node's true degree (a deliberate truncation,
    // supported elsewhere in this class) — a real edge beyond that truncation
    // cutoff has no representable slot at all, so this throws rather than write
    // out of bounds.
    optional<s64> slot = find_neighbor_slot(source_node, target_node);
    if (!slot.has_value()) {
        log::throw_invalid_argument(log::logger(),
            fmt::format("WeightMatrix::accumulate_edge_delta: (source_node={}, target_node={}) "
                        "is a real edge but not representable within max_neighbor_count={}",
                        source_node, target_node, max_neighbor_count));
    }

    f32 *delta_data = sparse_delta_buffers[(usize)matrix_index].get_contents();
    delta_data[source_node * max_neighbor_count + *slot] += delta;
    sparse_delta_touched[(usize)matrix_index] = true;
}

WeightStats WeightMatrix::neighbor_weight_stats() const {
    s64 total_pair_count = node_count * max_neighbor_count;
    if (total_pair_count == 0) {
        return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }
    vector<f32> weight_buffer((usize)total_pair_count);
    neighbor_weights(weight_buffer.data());

    f32 weight_sum = 0.0;
    f32 sum_of_squares = 0.0;
    f32 min_weight = weight_buffer[0], max_weight = weight_buffer[0];
    for (s64 pair_index = 0; pair_index < total_pair_count; ++pair_index) {
        f32 weight = weight_buffer[pair_index];
        weight_sum += weight;
        sum_of_squares += weight * weight;
        if (weight < min_weight) min_weight = weight;
        if (weight > max_weight) max_weight = weight;
    }
    f32 mean = weight_sum / (f32)total_pair_count;
    f32 variance = (sum_of_squares / (f32)total_pair_count) - mean * mean;
    f32 standard_deviation = sqrtf(variance > 0.0f ? variance : 0.0f);
    f32 root_mean_square = sqrtf(sum_of_squares / (f32)total_pair_count);

    return {mean, standard_deviation, root_mean_square, min_weight, max_weight};
}

ScaleResult WeightMatrix::scale_neighbor_weights_to_root_mean_square(
    f32 target_root_mean_square,
    f32 epsilon
) {
    if (target_root_mean_square < 0.0f) {
        log::throw_invalid_argument(log::logger(),
            fmt::format("scale_neighbor_weights_to_root_mean_square: target_root_mean_square "
                        "must be non-negative (got {})", target_root_mean_square));
    }

    WeightStats stats_before = neighbor_weight_stats();
    f32 current_root_mean_square = max(stats_before.root_mean_square, epsilon);
    f32 scale_factor = (target_root_mean_square > 0.0f)
        ? sqrtf(target_root_mean_square / current_root_mean_square)
        : 0.0f;

    s64 total_float4_element_count = node_count * rank_float4_stride;
    gpu_scale_uv(U_matrix.get_contents(), V_matrix.get_contents(), total_float4_element_count, scale_factor);
    synchronize_gpu_work();
    constant_weight = 0.0f;
    using_constant_weight = false;

    WeightStats stats_after = neighbor_weight_stats();

    log::logger().debug("scale_neighbor_weights_to_root_mean_square: target_root_mean_square={} "
                        "scale_factor={} rms_before={} rms_after={}",
                        target_root_mean_square, scale_factor,
                        stats_before.root_mean_square, stats_after.root_mean_square);

    return {target_root_mean_square, scale_factor, stats_before, stats_after};
}

void WeightMatrix::update(
    s32 source_node,
    s32 target_node,
    f32 delta,
    f32 learning_rate,
    f32 l2_regularization,
    s32 iterations
) {
    log::logger().trace("update: source_node={} target_node={} delta={} learning_rate={} "
                        "l2_regularization={} iterations={}",
                        source_node, target_node, delta, learning_rate, l2_regularization, iterations);
    if (!check_index_inbounds(source_node, target_node)) {
        return;
    }

    gpu_weight_update(
        U_matrix.get_contents(),
        V_matrix.get_contents(),
        rank_float4_stride,
        source_node,
        target_node,
        delta,
        learning_rate,
        l2_regularization,
        iterations
    );
    synchronize_gpu_work();
}

// save()/load_from_disk() persist only U_matrix/V_matrix — neither the
// coefficient_vectors family (ticket #52/D2) nor sparse_delta_buffers (ticket
// #53/D3, Sk) are written to disk. This is a deliberate extension of #52's own
// choice, for the same reason: there is no production consumer of a
// saved/reloaded multi-matrix family yet, so persisting a per-matrix Sk (which
// only makes sense alongside its matching Ck and basis) would be speculative
// machinery with nothing to exercise it. The periodic refit (#54) is expected
// to fold Sk back into U/V/Ck before any such consumer exists, at which point
// Sk's lifetime is transient between refits anyway rather than something that
// needs to survive a save/load round-trip. If a future ticket needs
// checkpoint/resume across the full matrix family, extend both save() and
// load_from_disk() together, matching whatever coefficient_vectors persistence
// scheme is added then.
void WeightMatrix::save(const char *filepath) const {
    log::logger().debug("WeightMatrix::save: filepath={} node_count={} rank={} rank_float4_stride={}",
                        filepath, node_count, rank, rank_float4_stride);
    ofstream file(filepath, ios::binary);
    file.write(reinterpret_cast<const char*>(&WEIGHT_MATRIX_SAVE_MAGIC), sizeof(u32));
    file.write(reinterpret_cast<const char*>(&node_count), sizeof(s64));
    file.write(reinterpret_cast<const char*>(&rank), sizeof(s64));
    file.write(reinterpret_cast<const char*>(&rank_float4_stride), sizeof(s64));
    const float4* u_data = U_matrix.get_contents();
    const float4* v_data = V_matrix.get_contents();
    file.write(reinterpret_cast<const char*>(u_data),
               (streamsize)(node_count * rank_float4_stride * sizeof(float4)));
    file.write(reinterpret_cast<const char*>(v_data),
               (streamsize)(node_count * rank_float4_stride * sizeof(float4)));
}

void WeightMatrix::load_from_disk(const char *filepath) {
    log::logger().debug("WeightMatrix::load_from_disk: filepath={}", filepath);
    ifstream file(filepath, ios::binary);
    u32 magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(u32));

    s64 saved_node_count, saved_rank, saved_rank_float4_stride;
    file.read(reinterpret_cast<char*>(&saved_node_count), sizeof(s64));
    file.read(reinterpret_cast<char*>(&saved_rank), sizeof(s64));
    file.read(reinterpret_cast<char*>(&saved_rank_float4_stride), sizeof(s64));

    if (saved_node_count != node_count || saved_rank_float4_stride != rank_float4_stride) {
        log::logger().debug("WeightMatrix::load_from_disk: reallocating U/V matrix "
                            "(node_count {} -> {}, rank_float4_stride {} -> {})",
                            node_count, saved_node_count, rank_float4_stride, saved_rank_float4_stride);
        deallocate(std::move(U_matrix));
        deallocate(std::move(V_matrix));
        usize matrix_byte_size = (usize)saved_node_count * (usize)saved_rank_float4_stride * sizeof(float4);
        U_matrix = allocate<float4>(matrix_byte_size);
        V_matrix = allocate<float4>(matrix_byte_size);
        node_count = saved_node_count;
        rank = saved_rank;
        rank_float4_stride = saved_rank_float4_stride;

        // rank_float4_stride changing invalidates every coefficient vector's length
        // (rank_float4_stride*4 lanes) — and any non-default matrix's Ck was defined
        // against the old shared basis anyway, so it's meaningless once U/V above are
        // reallocated. Reset the family back to just the fresh default (all-ones) slot.
        for (auto &coefficient_vector : coefficient_vectors) {
            deallocate(std::move(coefficient_vector));
        }
        coefficient_vectors.clear();
        coefficient_vectors.push_back(allocate_coefficient_vector({}));

        // Sk (ticket #53/D3) is not persisted by save() (see its own comment below),
        // and every existing Sk entry is keyed against edges/coefficients of the old
        // basis anyway — reset in lockstep with coefficient_vectors above so the two
        // families stay parallel-indexed. sparse_delta_buffers is GPU-resident now,
        // so (like coefficient_vectors above) each entry must be deallocated before
        // the vector is cleared.
        for (auto &delta_buffer : sparse_delta_buffers) {
            deallocate(std::move(delta_buffer));
        }
        sparse_delta_buffers.clear();
        sparse_delta_touched.clear();
        sparse_delta_buffers.push_back(allocate_sparse_delta_buffer());
        sparse_delta_touched.push_back(false);
    }

    file.read(reinterpret_cast<char*>(U_matrix.get_contents()),
              (streamsize)(node_count * rank_float4_stride * sizeof(float4)));
    file.read(reinterpret_cast<char*>(V_matrix.get_contents()),
              (streamsize)(node_count * rank_float4_stride * sizeof(float4)));

    log::logger().debug("WeightMatrix::load_from_disk: loaded node_count={} rank={} rank_float4_stride={} magic={:#x}",
                        node_count, rank, rank_float4_stride, magic);
}
