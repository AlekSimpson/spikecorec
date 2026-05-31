//
// Created by Alek Simpson on 5/30/26.
//
#pragma once

#include <spikecorec/core/types.h>
#include <spikecorec/core/k2tree.h>

using namespace std;

namespace spikecorec {
    struct WeightStats {
        float mean;
        float std;
        float rms;
        float min;
        float max;
    };

    struct ScaleResult {
        float target_rms;
        float scale_factor;
        WeightStats before;
        WeightStats after;
    };

    class WeightMatrix {
    public:
        vector<vector<s32> > *neighbors;
        K2Tree k2tree;
        float4 *U_matrix;
        float4 *V_matrix;
        s64 neighbor_count;
        s64 size;
        const f32 CONSTANT_WEIGHT;
        bool check_indexing = true;
        bool USING_CONSTANT_WEIGHT = false;

        WeightMatrix(
            vector<vector<s32> > &network,
            s64 rank,
            bool check_indexing = true,
            bool save_network = true,
            bool verify_k2tree = true,
            s64 verify_progress_every_x_steps = 1000);

        WeightMatrix();

        ~WeightMatrix();

        s32 *get_neighbors(s64 i);

        void set_constant_weight(f32 value);

        void neighbor_weights(float *output_weights);

        WeightStats neighbor_weight_stats();

        ScaleResult scale_neighbor_weights_to_rms(
            float target_root_mean_square,
            float epsilon = 1e-12f);

        f32 get(s32 i, s32 j); // __getitem__: dot product U[i]·V[j]
        void update(s32 i, s32 j,
            f32 delta,
            f32 learning_rate = 0.5f,
            f32 l2_regularization = 1.0f,
            s32 iterations = 1);

        void save(const char *filepath);

        void load_from_disk(const char *filepath);
    };
}
