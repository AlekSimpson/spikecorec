#pragma once

#include <vector>

#include "spikecorec/core/types.h"
#include "spikecorec/core/weight_matrix.h"

using namespace std;
namespace spikecorec {

    class SpikeEngine {
    public:
        WeightMatrix weights;

        s64 *last_tick_updated;
        s64 *probe_neuron_ids;
        f64 *network_inputs;
        f32 *membrane_potentials;

        s64 neuron_count;
        s64 compartment_count;
        s64 lifetime;
        s64 RANK = 1;
        const f64 RESTING_MEMBRANE_POTENTIAL = 0.0;
        const f64 DECAY_RATE = 0.01;
        const f64 LEARNING_RATE = 0.00222;
        const s32 SPIKE_PERIOD = 1;
        s32 spike_threshold = 1;

        bool running = false;

        SpikeEngine(vector<vector<s64>> &network, vector<s64> &shape, s64 lifetime, vector<s64> &probe_neuron_ids);
        SpikeEngine();
        ~SpikeEngine();

        bool is_alive();

        void init();

        void step();

        void shutdown();

    };
}

