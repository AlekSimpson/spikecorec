//
// Created by Alek Simpson on 5/30/26.
//

#include <cstring>

#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

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
    usize neuron_s64_byte_size = (usize) neuron_count * sizeof(s64);

    // network_inputs [neuron_count] — zero initialized
    network_inputs = allocate<f32>(neuron_f32_byte_size);
    memset(network_inputs.get_contents(), 0, neuron_f32_byte_size);

    // membrane_potentials [neuron_count] — filled with resting_mp
    membrane_potentials = allocate<f32>(neuron_f32_byte_size);
    std::fill(membrane_potentials.get_contents(),
            membrane_potentials.get_contents() + neuron_count,
            resting_membrane_potential);

    // last_spiked [neuron_count] — zero initialized
    last_spiked = allocate<s64>(neuron_s64_byte_size);
    memset(last_spiked.get_contents(), 0, neuron_s64_byte_size);

    // last_tick_updated [neuron_count] — zero initialized
    last_tick_updated = allocate<s64>(neuron_s64_byte_size);
    memset(last_tick_updated.get_contents(), 0, neuron_s64_byte_size);

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

void SpikeEngine::setup_lifetime(int lifetime_, bool allocate_logs, s64 max_log_bytes) {
    lifetime = lifetime_;
    if (lifetime < 0 || !allocate_logs) return;

    s32 size_of_f32 = 4;
    s64 required_bytes = neuron_count * lifetime * size_of_f32;
    if (max_log_bytes < required_bytes) {
        throw std::runtime_error(
            "Refusing to allocate membrane potential log: " +
            std::to_string(neuron_count) + " neurons x " + std::to_string(lifetime) + " ticks" +
            " requires " + std::to_string(required_bytes) + " bytes, which exceeds the " +
            std::to_string(max_log_bytes) + "-byte budget. Pass a larger max_log_bytes to enable recording."
        );
    }

    mp_logs = new f32*[neuron_count];
    for (s64 i = 0; i < neuron_count; ++i)
        mp_logs[i] = new f32[lifetime];
}

void SpikeEngine::set_input_neurons(const vector<s32> &input_neuron_list) {
    if (input_neuron_list.empty()) return;

    s32 s32_byte_size = 4;
    input_neuron_indices = allocate<s32>(neuron_count * s32_byte_size);
    memcpy(
        input_neuron_indices.get_contents(),
        input_neuron_list.data(),
        input_neuron_list.size() * s32_byte_size
    );
    input_neuron_count = (s64) input_neuron_list.size();
}

void SpikeEngine::reset_state(s64 last_spiked_value, s32 active_gen_value) {
    s32 f32_byte_size = 4;
    usize neuron_s32_byte_size = (usize) neuron_count * sizeof(s32);
    usize neuron_s64_byte_size = (usize) neuron_count * sizeof(s64);

    memset(network_inputs.get_contents(), 0, (usize) neuron_count * f32_byte_size);
    std::fill(membrane_potentials.get_contents(),
            membrane_potentials.get_contents() + neuron_count,
            resting_membrane_potential);
    memset(last_spiked.get_contents(), last_spiked_value, neuron_s64_byte_size);
    memset(last_tick_updated.get_contents(), 0, neuron_s64_byte_size);
    active_neuron_count.get_contents()[0] = 0;
    next_active_neuron_count.get_contents()[0] = 0;
    memset(active_generation.get_contents(), active_gen_value, neuron_s32_byte_size);
}

void SpikeEngine::step_simulation(
    const vector<f32> &input_values,
    s64 tick,
    const vector<s64> &override_input_neurons,
    bool decay_all_neurons
) {
    if (input_values.empty()) throw std::runtime_error("input_values is empty");

    if (decay_all_neurons) {
        gpu_decay_all_neurons(
            membrane_potentials.get_contents(),
            last_tick_updated.get_contents(),
            neuron_count,
            tick,
            resting_membrane_potential,
            decay_rate);
    }

    // input_values/override_input_neurons are host-side vectors — the GPU kernels
    // need them in GPU-visible memory (mirrors the Python reference's per-step
    // cp.asarray(input_values) transfer), so stage transient copies here.
    GpuPointer<f32> staged_input_values = allocate<f32>(input_values.size() * sizeof(f32));
    memcpy(staged_input_values.get_contents(), input_values.data(), input_values.size() * sizeof(f32));

    gpu_add_network_input(
        membrane_potentials.get_contents(),
        input_neuron_indices.get_contents(),
        staged_input_values.get_contents(),
        (s64)input_values.size());

    deallocate(std::move(staged_input_values));

    next_active_neuron_count.get_contents()[0] = 0;

    if (!override_input_neurons.empty()) {
        GpuPointer<s64> staged_override_neurons = allocate<s64>(override_input_neurons.size() * sizeof(s64));
        memcpy(staged_override_neurons.get_contents(), override_input_neurons.data(), override_input_neurons.size() * sizeof(s64));

        gpu_merge_input_neurons(
            active_neuron_indices.get_contents(),
            active_neuron_count.get_contents(),
            staged_override_neurons.get_contents(),
            (s64)override_input_neurons.size());

        deallocate(std::move(staged_override_neurons));
    }

    gpu_step(
        tick,
        tick + 1,
        spike_period,
        spike_threshold,
        learning_rate,
        decay_rate,
        resting_membrane_potential,
        weights.U_matrix.get_contents(),
        weights.V_matrix.get_contents(),
        weights.rank_float4_stride,
        use_constant_weight ? weights.constant_weight : 0.0,
        weights.k2tree.internal_node_words.get_contents(),
        weights.k2tree.leaf_node_words.get_contents(),
        weights.k2tree.rank_superblock_table.get_contents(),
        weights.k2tree.rank_subblock_table.get_contents(),
        weights.k2tree.branching_factor,
        weights.k2tree.superblock_size_words,
        weights.k2tree.padded_node_count,
        weights.k2tree.tree_height,
        weights.k2tree.internal_bit_count,
        neuron_count,
        network_inputs.get_contents(),
        membrane_potentials.get_contents(),
        last_spiked.get_contents(),
        last_tick_updated.get_contents(),
        active_neuron_indices.get_contents(),
        active_neuron_count.get_contents(),
        next_active_neuron_indices.get_contents(),
        next_active_neuron_count.get_contents(),
        active_generation.get_contents(),
        thread_count_per_block,
        block_count);

    std::swap(active_neuron_indices, next_active_neuron_indices);
    std::swap(active_neuron_count, next_active_neuron_count);
}

pair<f32, f32> SpikeEngine::estimate_bifurcation_weight(s32 input_period) const {
    f32 decay_factor = std::pow(1.0f - decay_rate, (f32)input_period);
    f32 w_accum = (spike_threshold - resting_membrane_potential) * (1.0f - decay_factor);
    f32 w_instant = spike_threshold - resting_membrane_potential;
    return make_pair(w_accum, w_instant);
}

void SpikeEngine::scale_uniform_weights_near_bifurcation(
    f32 *target, f32 *w_accum, f32 *w_instant,
    s32 input_period, f32 scale, bool freeze_learning, const bool *use_constant_weight_
) {
    auto [w_accum_, w_instant_] = estimate_bifurcation_weight(input_period);
    *w_accum = w_accum_;
    *w_instant = w_instant_;
    *target = *w_accum * scale;
    weights.set_constant_weight(*target);
    if (freeze_learning) {
        learning_rate = 0.0f;
    }
    use_constant_weight = use_constant_weight_ != nullptr
        ? *use_constant_weight_
        : freeze_learning;
}

ScaledReservoirResult SpikeEngine::scale_randomized_weights_near_bifurcation(s32 input_period, f32 scale, bool freeze_learning) {
    auto [w_accum, w_instant] = estimate_bifurcation_weight(input_period);
    f32 target = abs(w_accum * scale);
    ScaleResult result = weights.scale_neighbor_weights_to_root_mean_square(target);
    use_constant_weight = false;
    if (freeze_learning) {
        learning_rate = 0.0f;
    }
    return ScaledReservoirResult{result,target,w_accum,w_instant};
}

void SpikeEngine::get_reservoir_features_vector(s64 tick, f32 spike_tau, f32 voltage_scale, GpuPointer<f32> output_buffer) {
    if (spike_tau <= 0.0f) {
        return;
    }
    if (voltage_scale <= 0.0f) {
        return;
    }
    gpu_reservoir_features(
        neuron_count,
        tick,
        spike_tau,
        voltage_scale,
        membrane_potentials.get_contents(),
        last_spiked.get_contents(),
        last_tick_updated.get_contents(),
        resting_membrane_potential,
        decay_rate,
        output_buffer.get_contents());
}

bool SpikeEngine::is_alive() const {
    return running;
}

void SpikeEngine::shutdown() {
    if (!running) return;

    if (mp_logs != nullptr) {
        for (s64 i = 0; i < neuron_count; ++i)
            delete[] mp_logs[i];
        delete[] mp_logs;
        mp_logs = nullptr;
    }

    deallocate(std::move(network_inputs));
    deallocate(std::move(membrane_potentials));
    deallocate(std::move(last_spiked));
    deallocate(std::move(last_tick_updated));
    deallocate(std::move(active_neuron_indices));
    deallocate(std::move(next_active_neuron_indices));
    deallocate(std::move(active_neuron_count));
    deallocate(std::move(next_active_neuron_count));
    deallocate(std::move(active_generation));
    deallocate(std::move(input_neuron_indices));

    running = false;
}
























