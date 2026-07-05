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

vector<vector<s32>> &WeightMatrix::validate_network(vector<vector<s32>> &network) {
    if (network.empty()) {
        log::throw_invalid_argument(log::logger(),
            fmt::format("WeightMatrix: network must have at least one neuron (got {})", network.size()));
    }
    return network;
}

WeightMatrix::WeightMatrix(
    vector<vector<s32>> &network,
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
    if (max_neighbor_count > 0) {
        this->max_neighbor_count = max_neighbor_count;
    } else {
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
    static constexpr s64 MAX_RANK_FLOAT4_STRIDE = 64;
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

    // U/V are read by every gpu_step/gpu_neighbor_weights/gpu_scale_uv call this
    // run — prefetch them to the device right after the host-side fill instead
    // of letting the first kernel fault every page over one at a time (SC-18).
    prefetch_to_gpu(U_matrix, matrix_byte_size);
    prefetch_to_gpu(V_matrix, matrix_byte_size);

    log::logger().debug("WeightMatrix constructed: node_count={} rank={} rank_float4_stride={} "
                        "max_neighbor_count={} weight_seed={} check_indexing={}",
                        node_count, this->rank, rank_float4_stride, this->max_neighbor_count,
                        resolved_weight_seed, check_indexing);
}

WeightMatrix::~WeightMatrix() {
    deallocate(std::move(U_matrix));
    deallocate(std::move(V_matrix));
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
    // U filled with sqrt(|val|/rank), V filled with ±same based on sign of val
    f32 scale = (value != 0.0f) ? sqrtf(fabsf(value) / (f32)rank) : 0.0f;
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

f32 WeightMatrix::get(s32 source_node, s32 target_node) const {
    log::logger().trace("get: source_node={} target_node={}", source_node, target_node);
    if (!check_index_inbounds(source_node, target_node)) {
        return 0.0;
    }

    const float4 *u_row = U_matrix.get_contents() + source_node * rank_float4_stride;
    const float4 *v_row = V_matrix.get_contents() + target_node * rank_float4_stride;
    f32 dot_product = 0.0f;
    for (s64 float4_index = 0; float4_index < rank_float4_stride; ++float4_index) {
        dot_product += u_row[float4_index].x * v_row[float4_index].x
                     + u_row[float4_index].y * v_row[float4_index].y
                     + u_row[float4_index].z * v_row[float4_index].z
                     + u_row[float4_index].w * v_row[float4_index].w;
    }
    return dot_product;
}

void WeightMatrix::neighbor_weights(f32 *output_weights) const {
    s64 total_pair_count = node_count * max_neighbor_count;
    if (total_pair_count <= 0) return;

    // output_weights is caller-owned host memory (e.g. std::vector::data()) — not
    // GPU-visible — so the kernel writes into a scratch unified-memory buffer and
    // we copy the result back once the device is done.
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
        device_weights.get_contents()
    );
    synchronize_gpu_work();
    prefetch_to_cpu(device_weights, (usize)total_pair_count * sizeof(f32));
    memcpy(output_weights, device_weights.get_contents(), (usize)total_pair_count * sizeof(f32));
    deallocate(std::move(device_weights));
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
    }

    file.read(reinterpret_cast<char*>(U_matrix.get_contents()),
              (streamsize)(node_count * rank_float4_stride * sizeof(float4)));
    file.read(reinterpret_cast<char*>(V_matrix.get_contents()),
              (streamsize)(node_count * rank_float4_stride * sizeof(float4)));

    log::logger().debug("WeightMatrix::load_from_disk: loaded node_count={} rank={} rank_float4_stride={} magic={:#x}",
                        node_count, rank, rank_float4_stride, magic);
}
