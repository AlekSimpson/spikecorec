//
// Created by Alek Simpson on 5/30/26.
//
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "spikecorec/core/types.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/weight_matrix.h"

using namespace std;

namespace spikecorec {
    struct ScaledReservoirResult {
        ScaleResult weight_scale_result;
        f32 w_accum;
        f32 w_instant;
    };

    class SpikeEngine {
    public:
        WeightMatrix weights;

        GpuPointer<f32> network_inputs; // [neuron_count] — external input accumulator
        GpuPointer<f32> membrane_potentials; // [neuron_count]
        GpuPointer<s64> last_spiked; // [neuron_count] — tick each neuron last fired
        GpuPointer<s64> last_tick_updated; // [neuron_count] — tick each neuron last received input
        GpuPointer<s32> active_neuron_indices; // [neuron_count] — current active set (compacted)
        GpuPointer<s32> next_active_neuron_indices; // [neuron_count] — active set for next tick
        GpuPointer<s32> active_neuron_count; // [1]
        GpuPointer<s32> next_active_neuron_count; // [1]
        GpuPointer<s32> active_generation; // [neuron_count] — generation tag, -1 = inactive
        GpuPointer<s32> input_neuron_indices; // set via set_input_neurons()

        // constexpr (not plain const) so it is implicitly inline in C++17 — it is
        // ODR-used as a default-argument value in the pybind11 bindings
        // (bindings.cpp), which would otherwise require an out-of-line definition.
        static constexpr s64 DEFAULT_MAX_LOG_BYTES = 512 * 1024 * 1024;
        f32 **mp_logs = nullptr;

        s64 lifetime = 0;
        s64 neuron_count;
        s64 input_neuron_count;

        s32 thread_count_per_block;
        s32 block_count;

        f32 resting_membrane_potential;
        f32 decay_rate;
        f32 learning_rate;
        s32 spike_period;
        f32 spike_threshold;

        bool use_constant_weight = false;
        bool running = false;

        SpikeEngine() = delete;

        SpikeEngine(const SpikeEngine &) = delete;

        SpikeEngine &operator=(const SpikeEngine &) = delete;

        SpikeEngine(SpikeEngine &&) = default;

        SpikeEngine &operator=(SpikeEngine &&) = default;

        SpikeEngine(
            vector<vector<s32> > *network,
            const vector<s64> &shape,
            s64 rank = 1,
            f32 resting_mp = 0.1f,
            f32 decay_rate = 0.01f,
            f32 learning_rate = 0.00222f
        );

        ~SpikeEngine();

        void setup_lifetime(int lifetime, bool allocate_logs, s64 max_log_bytes = DEFAULT_MAX_LOG_BYTES);

        void set_input_neurons(const vector<s32> &input_neuron_list);

        void reset_state(s64 last_spiked_value = 0, s32 active_gen_value = -1);

        void step_simulation(
            const vector<f32> &input_values,
            s64 tick,
            const vector<s64> &override_input_neurons = {},
            bool decay_all_neurons = false);

        void start_static_record(
            const vector<vector<f32>> &input_spikes,
            s64 lifetime,
            const string &filename,
            bool record_membrane = true,
            s64 record_stride = 1,
            optional<string> compression = string("auto"),
            optional<int> compression_level = nullopt,
            bool full_decay = true,
            bool compression_async = false,
            usize compression_queue_max = 8,
            usize compression_chunk_bytes = 4 * 1024 * 1024);

        [[nodiscard]] bool is_alive() const;


        [[nodiscard]] pair<f32, f32> estimate_bifurcation_weight(s32 input_period = 1) const;

        void get_reservoir_features_vector(s64 tick, f32 spike_tau, f32 voltage_scale, GpuPointer<f32> output_buffer);

        // first three args are the return values
        void scale_uniform_weights_near_bifurcation(f32 *target_,
                                                   f32 *w_accum_,
                                                   f32 *w_instant_,
                                                   s32 input_period = 1,
                                                   f32 scale = 1.2,
                                                   bool freeze_learning = false,
                                                   const bool *use_constant_weight_ = nullptr);

        ScaledReservoirResult scale_randomized_weights_near_bifurcation(s32 input_period = 1, f32 scale = 1.2, bool freeze_learning = false);

        void shutdown();
    };
} // namespace spikecorec
