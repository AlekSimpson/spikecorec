//
// Created by Alek Simpson on 5/30/26.
//

#include <random>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/core/backend.h"

using namespace std;
using namespace spikecorec;

static constexpr s64 DEFAULT_WEIGHT_RANK = 64;
static constexpr u32 WEIGHT_MATRIX_SAVE_MAGIC = 0x574D5458; // "WMTX"

// ── constructor / destructor ──────────────────────────────────────────────────

WeightMatrix::WeightMatrix(
    vector<vector<s32>>& network,
    s64 rank,
    bool check_indexing
)
    : k2tree(K2Tree::from_adjacency_list(network, (s32)network.size()))
    , node_count((s64)network.size())
    , neighbor_count(0)
    , rank(0)
    , rank_float4_stride(0)
    , constant_weight(0.0f)
    , check_indexing(check_indexing)
    , using_constant_weight(false)
{
    if (node_count <= 0)
        throw invalid_argument("Network must have at least one neuron.");

    // validate uniform neighbor count
    s64 first_node_neighbor_count = (s64)network[0].size();
    for (s64 node_index = 1; node_index < node_count; node_index++) {
        if ((s64)network[node_index].size() != first_node_neighbor_count)
            throw invalid_argument(
                "Inconsistent neighbor count: all neurons must have the same number of neighbors."
            );
    }
    neighbor_count = first_node_neighbor_count;

    this->rank = (rank > 0) ? rank : min(DEFAULT_WEIGHT_RANK, node_count);
    rank_float4_stride = (this->rank + 3) / 4;

    // build flat neighbor index array in unified memory — shape [node_count * neighbor_count]
    neighbor_indices = allocate<s32>((usize)node_count * (usize)neighbor_count * sizeof(s32));
    s32* flat_neighbors = neighbor_indices.get_contents();
    for (s64 source_node = 0; source_node < node_count; source_node++) {
        for (s64 neighbor_slot = 0; neighbor_slot < neighbor_count; neighbor_slot++) {
            flat_neighbors[source_node * neighbor_count + neighbor_slot] =
                network[source_node][neighbor_slot];
        }
    }

    // allocate U and V in unified memory — shape [node_count][rank_float4_stride]
    usize matrix_byte_size = (usize)node_count * (usize)rank_float4_stride * sizeof(float4);
    U_matrix = allocate<float4>(matrix_byte_size);
    V_matrix = allocate<float4>(matrix_byte_size);

    // initialize with independent random normal values (mean=0, std=1)
    mt19937 rng(random_device{}());
    normal_distribution<f32> normal_dist(0.0f, 1.0f);
    float4* u_data = U_matrix.get_contents();
    float4* v_data = V_matrix.get_contents();
    s64 total_float4_element_count = node_count * rank_float4_stride;
    for (s64 element_index = 0; element_index < total_float4_element_count; element_index++) {
        u_data[element_index] = {normal_dist(rng), normal_dist(rng), normal_dist(rng), normal_dist(rng)};
        v_data[element_index] = {normal_dist(rng), normal_dist(rng), normal_dist(rng), normal_dist(rng)};
    }
}

WeightMatrix::~WeightMatrix() {
    deallocate(std::move(neighbor_indices));
    deallocate(std::move(U_matrix));
    deallocate(std::move(V_matrix));
}

// ── neighbor access ───────────────────────────────────────────────────────────

const s32* WeightMatrix::get_neighbors(s64 node_index) const {
    return neighbor_indices.get_contents() + node_index * neighbor_count;
}

// ── weight initialization ─────────────────────────────────────────────────────

void WeightMatrix::set_constant_weight(f32 value) {
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

// ── weight queries ────────────────────────────────────────────────────────────

f32 WeightMatrix::get(s32 source_node, s32 target_node) const {
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
    // TODO: dispatch GPU kernel for parallel dot products over all (source_node, neighbor) pairs
    const float4 *u_data = U_matrix.get_contents();
    const float4 *v_data = V_matrix.get_contents();
    const s32 *flat_neighbors = neighbor_indices.get_contents();
    // for (s64 source_node = 0; source_node < node_count; ++source_node) {
    //     const float4 *u_row = u_data + source_node * rank_float4_stride;
    //     for (s64 neighbor_slot = 0; neighbor_slot < neighbor_count; neighbor_slot++) {
    //         s32 target_node = flat_neighbors[source_node * neighbor_count + neighbor_slot];
    //         const float4* v_row = v_data + (s64)target_node * rank_float4_stride;
    //         f32 dot_product = 0.0f;
    //         for (s64 float4_index = 0; float4_index < rank_float4_stride; float4_index++) {
    //             dot_product += u_row[float4_index].x * v_row[float4_index].x
    //                          + u_row[float4_index].y * v_row[float4_index].y
    //                          + u_row[float4_index].z * v_row[float4_index].z
    //                          + u_row[float4_index].w * v_row[float4_index].w;
    //         }
    //         output_weights[source_node * neighbor_count + neighbor_slot] = dot_product;
    //     }
    // }
}

WeightStats WeightMatrix::neighbor_weight_stats() const {
    s64 total_pair_count = node_count * neighbor_count;
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

// ── weight scaling ────────────────────────────────────────────────────────────

ScaleResult WeightMatrix::scale_neighbor_weights_to_root_mean_square(
    f32 target_root_mean_square,
    f32 epsilon
) {
    if (target_root_mean_square < 0.0f)
        throw invalid_argument("target_root_mean_square must be non-negative.");

    WeightStats stats_before = neighbor_weight_stats();
    f32 current_root_mean_square = max(stats_before.rms, epsilon);
    f32 scale_factor = (target_root_mean_square > 0.0f)
        ? sqrtf(target_root_mean_square / current_root_mean_square)
        : 0.0f;

    // TODO: dispatch GPU kernel for in-place element-wise scaling of U and V
    // float4* u_data = U_matrix.get_contents();
    // float4* v_data = V_matrix.get_contents();
    // s64 total_float4_element_count = node_count * rank_float4_stride;
    // for (s64 index = 0; index < total_float4_element_count; ++index) {
    //     u_data[index].x *= scale_factor;
    //     u_data[index].y *= scale_factor;
    //     u_data[index].z *= scale_factor;
    //     u_data[index].w *= scale_factor;
    //     v_data[index].x *= scale_factor;
    //     v_data[index].y *= scale_factor;
    //     v_data[index].z *= scale_factor;
    //     v_data[index].w *= scale_factor;
    // }
    // constant_weight = 0.0f;
    // using_constant_weight = false;

    WeightStats stats_after = neighbor_weight_stats();
    return {target_root_mean_square, scale_factor, stats_before, stats_after};
}

// ── learning ──────────────────────────────────────────────────────────────────

void WeightMatrix::update(
    s32 source_node,
    s32 target_node,
    f32 delta,
    f32 learning_rate,
    f32 l2_regularization,
    s32 iterations
) {
    // TODO: dispatch GPU kernel (see update_kernel in weights_cuda.py)
    (void)source_node; (void)target_node; (void)delta;
    (void)learning_rate; (void)l2_regularization; (void)iterations;
}

// ── serialization ─────────────────────────────────────────────────────────────

void WeightMatrix::save(const char *filepath) const {
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
    ifstream file(filepath, ios::binary);
    u32 magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(u32));

    s64 saved_node_count, saved_rank, saved_rank_float4_stride;
    file.read(reinterpret_cast<char*>(&saved_node_count), sizeof(s64));
    file.read(reinterpret_cast<char*>(&saved_rank), sizeof(s64));
    file.read(reinterpret_cast<char*>(&saved_rank_float4_stride), sizeof(s64));

    if (saved_node_count != node_count || saved_rank_float4_stride != rank_float4_stride) {
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
}
