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

f32 WeightMatrix::get(s32 source_node, s32 target_node) const {
    log::logger().trace("get: source_node={} target_node={}", source_node, target_node);
    if (!check_index_inbounds(source_node, target_node)) {
        return 0.0;
    }

    return reconstruct_entry(source_node, target_node,
                              coefficient_vectors[(usize)DEFAULT_MATRIX_INDEX].get_contents());
}

f32 WeightMatrix::get_for_matrix(s32 source_node, s32 target_node, s64 matrix_index) const {
    log::logger().trace("get_for_matrix: source_node={} target_node={} matrix_index={}",
                        source_node, target_node, matrix_index);
    validate_matrix_index(matrix_index);
    if (!check_index_inbounds(source_node, target_node)) {
        return 0.0;
    }

    return reconstruct_entry(source_node, target_node,
                              coefficient_vectors[(usize)matrix_index].get_contents());
}

s64 WeightMatrix::add_coefficient_vector(const vector<f32> &coefficients) {
    if ((s64)coefficients.size() != rank) {
        log::throw_invalid_argument(log::logger(),
            fmt::format("WeightMatrix::add_coefficient_vector: coefficients must have exactly "
                        "rank ({}) elements (got {})", rank, coefficients.size()));
    }

    s64 new_matrix_index = (s64)coefficient_vectors.size();
    coefficient_vectors.push_back(allocate_coefficient_vector(coefficients));
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
}

void WeightMatrix::neighbor_weights_for_matrix(f32 *output_weights, s64 matrix_index) const {
    validate_matrix_index(matrix_index);
    dispatch_neighbor_weights(output_weights, coefficient_vectors[(usize)matrix_index].get_contents());
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
    }

    file.read(reinterpret_cast<char*>(U_matrix.get_contents()),
              (streamsize)(node_count * rank_float4_stride * sizeof(float4)));
    file.read(reinterpret_cast<char*>(V_matrix.get_contents()),
              (streamsize)(node_count * rank_float4_stride * sizeof(float4)));

    log::logger().debug("WeightMatrix::load_from_disk: loaded node_count={} rank={} rank_float4_stride={} magic={:#x}",
                        node_count, rank, rank_float4_stride, magic);
}
