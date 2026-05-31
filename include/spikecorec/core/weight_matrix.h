//
// Created by Alek Simpson on 5/30/26.
//
#pragma once

#include <vector>
#include <spikecorec/core/types.h>
#include <spikecorec/core/backend.h>
#include <spikecorec/core/k2tree.h>

using namespace std;

namespace spikecorec {
    struct WeightStats {
        f32 mean;
        f32 std_dev;
        f32 rms;
        f32 min_value;
        f32 max_value;
    };

    struct ScaleResult {
        f32 target_rms;
        f32 scale_factor;
        WeightStats before;
        WeightStats after;
    };

    class WeightMatrix {
    public:
        K2Tree k2tree;
        GpuPointer<s32> neighbor_indices;   // flat [node_count * neighbor_count], row i at i * neighbor_count
        GpuPointer<float4> U_matrix;        // row-major [node_count][rank_float4_stride]
        GpuPointer<float4> V_matrix;
        s64 node_count;
        s64 neighbor_count;                 // neighbors per node (uniform across all nodes)
        s64 rank;                           // latent factor dimensionality
        s64 rank_float4_stride;             // ceil(rank / 4) — float4 elements per row
        f32 constant_weight;
        bool check_indexing;
        bool using_constant_weight;

        WeightMatrix() = delete;

        WeightMatrix(const WeightMatrix &) = delete;

        WeightMatrix &operator=(const WeightMatrix &) = delete;

        WeightMatrix(WeightMatrix &&) = default;

        WeightMatrix &operator=(WeightMatrix &&) = default;

        // network: adjacency list — network[i] is the list of neighbors of node i (must be uniform length)
        // rank:    latent factor dimensionality; -1 → min(64, node_count)
        WeightMatrix(
            vector<vector<s32> > &network,
            s64 rank = -1,
            bool check_indexing = true
        );

        ~WeightMatrix();

        // returns pointer to node_index's neighbor list (neighbor_count elements)
        const s32 *get_neighbors(s64 node_index) const;

        void set_constant_weight(f32 value);

        // writes node_count * neighbor_count dot products into output_weights
        void neighbor_weights(f32 *output_weights) const;

        WeightStats neighbor_weight_stats() const;

        ScaleResult scale_neighbor_weights_to_root_mean_square(
            f32 target_root_mean_square,
            f32 epsilon = 1e-12f
        );

        f32 get(s32 source_node, s32 target_node) const;

        void update(
            s32 source_node,
            s32 target_node,
            f32 delta,
            f32 learning_rate = 0.5f,
            f32 l2_regularization = 1.0f,
            s32 iterations = 1
        );

        void save(const char *filepath) const;

        void load_from_disk(const char *filepath);
    };
} // namespace spikecorec
