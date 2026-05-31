//
// Created by Alek Simpson on 5/30/26.
//
#pragma once

#include <vector>

#include "spikecorec/core/types.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/weight_matrix.h"

using namespace std;

namespace spikecorec {

    class SpikeEngine {
    public:
        WeightMatrix weights;

        GpuPointer<f32> network_inputs;                 // [neuron_count] — external input accumulator
        GpuPointer<f32> membrane_potentials;            // [neuron_count]
        GpuPointer<s32> last_spiked;                    // [neuron_count] — tick each neuron last fired
        GpuPointer<s32> last_tick_updated;              // [neuron_count] — tick each neuron last received input
        GpuPointer<s32> active_neuron_indices;          // [neuron_count] — current active set (compacted)
        GpuPointer<s32> next_active_neuron_indices;     // [neuron_count] — active set for next tick
        GpuPointer<s32> active_neuron_count;            // [1]
        GpuPointer<s32> next_active_neuron_count;       // [1]
        GpuPointer<s32> active_generation;              // [neuron_count] — generation tag, -1 = inactive
        GpuPointer<s32> input_neuron_indices;           // set via set_input_neurons()

        s64 neuron_count;
        s64 input_neuron_count;
        s32 thread_count_per_block;
        s32 block_count;

        f32 resting_membrane_potential;
        f32 decay_rate;
        f32 learning_rate;
        s32 spike_period;
        f32 spike_threshold;

        bool running;

        SpikeEngine() = delete;
        SpikeEngine(const SpikeEngine &) = delete;
        SpikeEngine& operator=(const SpikeEngine &) = delete;
        SpikeEngine(SpikeEngine&&) = default;
        SpikeEngine& operator=(SpikeEngine &&) = default;

        SpikeEngine(
            vector<vector<s32>> *network,
            const vector<s64> &shape,
            s64 rank = 1,
            f32 resting_mp = 0.1f,
            f32 decay_rate = 0.01f,
            f32 learning_rate = 0.00222f
        );

        ~SpikeEngine();

        void set_input_neurons(const vector<s32> &input_neuron_list);

        bool is_alive() const;

        void step(s64 tick);

        void shutdown();
    };

} // namespace spikecorec
