//
// Created by Alek Simpson on 5/30/26.
//

#include <cstring>

#include "spikecorec/core/engine.h"
#include "spikecorec/core/backend.h"

using namespace std;
using namespace spikecorec;

// ── constructor / destructor ──────────────────────────────────────────────────

SpikeEngine::SpikeEngine(
    vector<vector<s32>> *network,
    const vector<s64> &shape,
    s64 rank,
    f32 resting_mp,
    f32 decay_rate,
    f32 learning_rate
)
    : weights(*network, rank, true)
    , neuron_count(shape[0] * shape[1])
    , input_neuron_count(0)
    , thread_count_per_block(0)
    , block_count(0)
    , resting_membrane_potential(resting_mp)
    , decay_rate(decay_rate)
    , learning_rate(learning_rate)
    , spike_period(1)
    , spike_threshold(1.0f)
    , running(true)
{
    thread_count_per_block = 256;
    block_count = (s32) ((neuron_count + thread_count_per_block - 1) / thread_count_per_block);

    usize neuron_f32_byte_size = (usize) neuron_count * sizeof(f32);
    usize neuron_s32_byte_size = (usize) neuron_count * sizeof(s32);

    // network_inputs [neuron_count] — zero initialized
    network_inputs = allocate<f32>(neuron_f32_byte_size);
    memset(network_inputs.get_contents(), 0, neuron_f32_byte_size);

    // membrane_potentials [neuron_count] — filled with resting_mp
    membrane_potentials = allocate<f32>(neuron_f32_byte_size);
    f32 *mp_data = membrane_potentials.get_contents();
    for (s64 neuron_index = 0; neuron_index < neuron_count; ++neuron_index)
        mp_data[neuron_index] = resting_membrane_potential;

    // last_spiked [neuron_count] — zero initialized
    last_spiked = allocate<s32>(neuron_s32_byte_size);
    memset(last_spiked.get_contents(), 0, neuron_s32_byte_size);

    // last_tick_updated [neuron_count] — zero initialized
    last_tick_updated = allocate<s32>(neuron_s32_byte_size);
    memset(last_tick_updated.get_contents(), 0, neuron_s32_byte_size);

    // active/next_active index buffers [neuron_count] — filled during simulation
    active_neuron_indices = allocate<s32>(neuron_s32_byte_size);
    next_active_neuron_indices = allocate<s32>(neuron_s32_byte_size);

    // single-element count buffers — zero initialized
    active_neuron_count = allocate<s32>(sizeof(s32));
    next_active_neuron_count = allocate<s32>(sizeof(s32));
    active_neuron_count.get_contents()[0] = 0;
    next_active_neuron_count.get_contents()[0] = 0;

    // active_generation [neuron_count] — filled with -1 (all inactive)
    active_generation = allocate<s32>(neuron_s32_byte_size);
    s32 *active_generation_data = active_generation.get_contents();
    for (s64 neuron_index = 0; neuron_index < neuron_count; ++neuron_index)
        active_generation_data[neuron_index] = -1;
}

SpikeEngine::~SpikeEngine() {
    if (running) shutdown();
}

void SpikeEngine::step(s64 tick) {
}

