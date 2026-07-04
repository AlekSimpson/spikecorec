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
        f32 standard_deviation;
        f32 root_mean_square;
        f32 min_value;
        f32 max_value;
    };

    struct ScaleResult {
        f32 target_root_mean_square;
        f32 scale_factor;
        WeightStats before;
        WeightStats after;
    };

    bool can_safely_cast_s64_to_s32(s64);


    class WeightMatrix {
    public:
        K2Tree k2tree;
        GpuPointer<float4> U_matrix;        // row-major [node_count][rank_float4_stride]
        GpuPointer<float4> V_matrix;
        s64 node_count;
        s64 max_neighbor_count;             // upper bound on neighbors per node — bounds the padded
                                            // [node_count * max_neighbor_count] neighbor_weights output;
                                            // rows for nodes with fewer neighbors are sentinel-padded (-1)
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

        // network:           adjacency list — network[i] is the list of neighbors of node i
        // rank:              latent factor dimensionality; -1 → min(64, node_count)
        // max_neighbor_count: upper bound on neighbors any single node may have; -1 → derived
        //                     from the longest row in `network`
        // weight_seed:       seeds U/V initialization for reproducible weights; -1 → seed
        //                    from std::random_device (non-deterministic)
        WeightMatrix(
            vector<vector<s32> > &network,
            s64 rank = -1,
            bool check_indexing = true,
            s64 max_neighbor_count = -1,
            s64 weight_seed = -1
        );

        ~WeightMatrix();


        bool check_index_inbounds(s32, s32) const;
        bool check_index_inbounds(s32) const;

    private:
        // Fatally exits if `network` is empty. Called from the constructor's
        // initializer list, before k2tree(K2Tree::from_adjacency_list(...)) —
        // k2tree is the first member and does real GPU work, so this must run
        // ahead of it rather than as a body-level check after the fact.
        static vector<vector<s32>> &validate_network(vector<vector<s32>> &network);

    public:

        // writes up to max_neighbor_count neighbor indices of node_index into output_buffer
        // (caller-allocated, at least max_neighbor_count elements); returns the number of
        // neighbors written. Resolved via a k^2-tree row-walk — see K2Tree::get_neighbors.
        [[nodiscard]] s64 get_neighbors(s64 node_index, s32 *output_buffer) const;

        void set_constant_weight(f32 value);

        // writes node_count * max_neighbor_count dot products into output_weights, row-major
        // by source node; slots beyond a node's actual neighbor count are sentinel-padded
        void neighbor_weights(f32 *output_weights) const;

        [[nodiscard]] WeightStats neighbor_weight_stats() const;

        ScaleResult scale_neighbor_weights_to_root_mean_square(
            f32 target_root_mean_square,
            f32 epsilon = 1e-12f
        );

        [[nodiscard]] f32 get(s32 source_node, s32 target_node) const;

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
