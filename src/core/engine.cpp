//
// Created by Alek Simpson on 5/30/26.
//

#include <cstring>
#include <cmath>

#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include "spikecorec/core/engine.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/recording.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::log;

// ── constructor / destructor ──────────────────────────────────────────────────

SpikeEngine::SpikeEngine(
    vector<vector<s32>> *network,
    const vector<s64> &shape,
    s64 rank,
    f32 resting_mp,
    f32 decay_rate,
    f32 learning_rate
)
    : logger(make_logger())
    , weights(*network, rank, true)
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

    // input_staging / override_staging [neuron_count] — persistent step_simulation scratch
    // buffers, overwritten in place each tick instead of allocate/copy/free per tick (SC-20)
    input_staging = allocate<f32>(neuron_f32_byte_size);
    override_staging = allocate<s64>(neuron_s64_byte_size);

    // Every buffer above is read and/or written by step_simulation's kernels on
    // every tick for the engine's whole lifetime — prefetch them to the device
    // once here instead of letting the first tick fault each page over
    // one-by-one as the kernels touch it (SC-18).
    prefetch_to_gpu(network_inputs, neuron_f32_byte_size);
    prefetch_to_gpu(membrane_potentials, neuron_f32_byte_size);
    prefetch_to_gpu(last_spiked, neuron_s64_byte_size);
    prefetch_to_gpu(last_tick_updated, neuron_s64_byte_size);
    prefetch_to_gpu(active_neuron_indices, neuron_s32_byte_size);
    prefetch_to_gpu(next_active_neuron_indices, neuron_s32_byte_size);
    prefetch_to_gpu(active_neuron_count, sizeof(s32));
    prefetch_to_gpu(next_active_neuron_count, sizeof(s32));
    prefetch_to_gpu(active_generation, neuron_s32_byte_size);
    prefetch_to_gpu(input_staging, neuron_f32_byte_size);
    prefetch_to_gpu(override_staging, neuron_s64_byte_size);

    logger->debug("SpikeEngine buffers allocated: neuron_f32_byte_size={} neuron_s32_byte_size={} "
                  "neuron_s64_byte_size={} thread_count_per_block={} block_count={}",
                  neuron_f32_byte_size, neuron_s32_byte_size, neuron_s64_byte_size,
                  thread_count_per_block, block_count);
    logger->info("SpikeEngine constructed: neuron_count={} resting_mp={} decay_rate={} learning_rate={}",
                  neuron_count, resting_mp, decay_rate, learning_rate);
}

SpikeEngine::~SpikeEngine() {
    logger->info("Spike Engine shutting down.");
    if (running) shutdown();
}

void SpikeEngine::setup_lifetime(int lifetime_, bool allocate_logs, s64 max_log_bytes) {
    logger->debug("setup_lifetime: lifetime={} allocate_logs={} max_log_bytes={}",
                  lifetime_, allocate_logs, max_log_bytes);
    lifetime = lifetime_;
    if (lifetime < 0 || !allocate_logs) {
        logger->info("Not allocating logs for run data.");
        return;
    }

    s32 size_of_f32 = 4;
    s64 required_bytes = neuron_count * lifetime * size_of_f32;
    if (max_log_bytes < required_bytes) {
        throw_runtime_error(*logger,
            fmt::format("setup_lifetime: refusing to allocate membrane potential log "
                        "({} neurons x {} ticks = {} bytes exceeds {}-byte budget; "
                        "pass a larger max_log_bytes to enable recording)",
                        neuron_count, lifetime, required_bytes, max_log_bytes));
    }

    cell_state_logs = new f32*[neuron_count];
    for (s64 i = 0; i < neuron_count; ++i)
        cell_state_logs[i] = new f32[lifetime];

    logger->debug("setup_lifetime: allocated cell_state_logs for {} neurons x {} ticks", neuron_count, lifetime);
}

void SpikeEngine::set_input_neurons(const vector<s32> &input_neuron_list) {
    logger->debug("set_input_neurons: input_neuron_count={}", input_neuron_list.size());
    if (input_neuron_list.empty()) return;

    s32 s32_byte_size = 4;
    input_neuron_indices = allocate<s32>(neuron_count * s32_byte_size);
    memcpy(
        input_neuron_indices.get_contents(),
        input_neuron_list.data(),
        input_neuron_list.size() * s32_byte_size
    );
    input_neuron_count = (s64) input_neuron_list.size();

    prefetch_to_gpu(input_neuron_indices, (usize)neuron_count * s32_byte_size);
}

void SpikeEngine::reset_state(s64 last_spiked_value, s32 active_gen_value) {
    logger->debug("reset_state: last_spiked_value={} active_gen_value={}", last_spiked_value, active_gen_value);
    s32 f32_byte_size = 4;
    usize neuron_s64_byte_size = (usize) neuron_count * sizeof(s64);

    memset(network_inputs.get_contents(), 0, (usize) neuron_count * f32_byte_size);
    std::fill(membrane_potentials.get_contents(),
            membrane_potentials.get_contents() + neuron_count,
            resting_membrane_potential);
    std::fill(last_spiked.get_contents(), last_spiked.get_contents() + neuron_count, last_spiked_value);
    memset(last_tick_updated.get_contents(), 0, neuron_s64_byte_size);
    active_neuron_count.get_contents()[0] = 0;
    next_active_neuron_count.get_contents()[0] = 0;
    std::fill(active_generation.get_contents(), active_generation.get_contents() + neuron_count, active_gen_value);
}

void SpikeEngine::step_simulation(
    const vector<f32> &input_values,
    s64 tick,
    const vector<s64> &override_input_neurons,
    bool decay_all_neurons
) {
    if (input_values.empty()) {
        log::throw_runtime_error(*logger, fmt::format("step_simulation: input_values is empty (tick={})", tick));
    }

    logger->trace("step_simulation: tick={} input_values.size={} override_input_neurons.size={} decay_all_neurons={}",
                  tick, input_values.size(), override_input_neurons.size(), decay_all_neurons);

    // Batch every kernel this tick needs (decay + add_network_input + merge + step) into
    // a single command buffer instead of committing/waitUntilCompleted per kernel — cuts
    // the tick from up to four CPU/GPU round trips down to one (SC-19). Still waited on
    // before returning: the host swaps active_neuron_indices/active_neuron_count below and
    // the next tick immediately overwrites input_staging/override_staging/next_active_neuron_count,
    // all of which the just-encoded kernels read or write.
    MetalCommandBatch *batch = begin_command_batch();

    if (decay_all_neurons) {
        gpu_decay_all_neurons(
            membrane_potentials.get_contents(),
            last_tick_updated.get_contents(),
            neuron_count,
            tick,
            resting_membrane_potential,
            decay_rate,
            batch);
    }

    // input_values/override_input_neurons are host-side vectors — the GPU kernels need them
    // in GPU-visible memory (mirrors the Python reference's per-step cp.asarray(input_values)
    // transfer), so copy into the persistent staging buffers here (input_staging /
    // override_staging are allocated once in the constructor and overwritten in place).
    memcpy(input_staging.get_contents(), input_values.data(), input_values.size() * sizeof(f32));

    // QUESTION: anyway to combine these two steps? what exactly is the merge_input_neurons for?
    gpu_add_network_input(
        membrane_potentials.get_contents(),
        input_neuron_indices.get_contents(),
        input_staging.get_contents(),
        (s64)input_values.size(),
        batch);

    next_active_neuron_count.get_contents()[0] = 0;

    if (!override_input_neurons.empty()) {
        memcpy(override_staging.get_contents(), override_input_neurons.data(), override_input_neurons.size() * sizeof(s64));

        gpu_merge_input_neurons(
            active_neuron_indices.get_contents(),
            active_neuron_count.get_contents(),
            override_staging.get_contents(),
            (s64)override_input_neurons.size(),
            batch);
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
        block_count,
        batch);

    commit_command_batch(batch);

    std::swap(active_neuron_indices, next_active_neuron_indices);
    std::swap(active_neuron_count, next_active_neuron_count);

    logger->trace("step_simulation: tick={} completed, active_neuron_count={}",
                  tick, active_neuron_count.get_contents()[0]);
}

void SpikeEngine::start_static_record(
    const vector<vector<f32>> &input_spikes,
    s64 lifetime,
    const string &filename,
    bool record_membrane,
    s64 record_stride,
    optional<string> compression,
    optional<int> compression_level,
    bool full_decay,
    bool compression_async,
    usize compression_queue_max,
    usize compression_chunk_bytes
) {
    if (lifetime < 0) {
        log::throw_runtime_error(*logger, fmt::format("start_static_record: lifetime must be >= 0 (got {})", lifetime));
    }
    if (record_stride < 1) {
        log::throw_runtime_error(*logger,
            fmt::format("start_static_record: record_stride must be >= 1 (got {})", record_stride));
    }
    if (input_neuron_count <= 0) {
        log::throw_runtime_error(*logger,
            "start_static_record: no input neurons configured (call set_input_neurons first)");
    }
    if ((s64)input_spikes.size() < lifetime) {
        log::throw_runtime_error(*logger,
            fmt::format("start_static_record: input_spikes has {} ticks but lifetime requires {}",
                        input_spikes.size(), lifetime));
    }

    // Each tick's input row is positionally matched to input_neuron_indices, so
    // it must contain exactly input_neuron_count values. A wider row would make
    // step_simulation's add_network_input kernel read uninitialized
    // input_neuron_indices slots and write out of bounds; an empty/short row
    // would silently under-stimulate or fail mid-loop, leaving a truncated
    // recording. Validate up front (before opening the output file) so bad input
    // produces no partial file.
    for (s64 tick = 0; tick < lifetime; ++tick) {
        if ((s64)input_spikes[(usize)tick].size() != input_neuron_count) {
            log::throw_runtime_error(*logger,
                fmt::format("start_static_record: input_spikes[{}] has {} values but there are {} input neurons",
                            tick, input_spikes[(usize)tick].size(), input_neuron_count));
        }
    }

    logger->info("start_static_record: lifetime={} filename={}", lifetime, filename);

    SimulationRecorder recorder(
        filename, neuron_count, compression, compression_level,
        compression_async, compression_queue_max, compression_chunk_bytes);

    // Forces input neurons into the active set every tick regardless of
    // whether they're already active — mirrors the reference's per-tick
    // _add_active(self.input_neurons, tick) call (spike_engine_cuda.py:344-345).
    vector<s64> override_input_neurons((usize)input_neuron_count);
    const s32 *input_indices = input_neuron_indices.get_contents();
    for (s64 i = 0; i < input_neuron_count; ++i)
        override_input_neurons[(usize)i] = (s64)input_indices[i];

    for (s64 tick = 0; tick < lifetime; ++tick) {
        step_simulation(input_spikes[(usize)tick], tick, override_input_neurons, /*decay_all_neurons=*/false);

        if (record_membrane && tick % record_stride == 0) {
            // _decay_all(tick) runs after step() and only on recorded ticks,
            // immediately before the membrane-potential snapshot — a real,
            // stateful operation (spike_engine_cuda.py:347-352), not a preview.
            if (full_decay) {
                gpu_decay_all_neurons(
                    membrane_potentials.get_contents(),
                    last_tick_updated.get_contents(),
                    neuron_count,
                    tick,
                    resting_membrane_potential,
                    decay_rate);
            }

            synchronize_gpu_work();
            prefetch_to_cpu(membrane_potentials, (usize)neuron_count * sizeof(f32));
            recorder.record_frame(membrane_potentials.get_contents(), neuron_count);
        }
    }

    recorder.finish();
    logger->info("start_static_record: finished");
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

    logger->debug("scale_uniform_weights_near_bifurcation: input_period={} scale={} target={} "
                  "w_accum={} w_instant={} use_constant_weight={}",
                  input_period, scale, *target, *w_accum, *w_instant, use_constant_weight);
}

ScaledReservoirResult SpikeEngine::scale_randomized_weights_near_bifurcation(s32 input_period, f32 scale, bool freeze_learning) {
    auto [w_accum, w_instant] = estimate_bifurcation_weight(input_period);
    f32 target = abs(w_accum * scale);
    ScaleResult result = weights.scale_neighbor_weights_to_root_mean_square(target);
    use_constant_weight = false;
    if (freeze_learning) {
        learning_rate = 0.0f;
    }

    logger->debug("scale_randomized_weights_near_bifurcation: input_period={} scale={} target={} "
                  "w_accum={} w_instant={}",
                  input_period, scale, target, w_accum, w_instant);

    return ScaledReservoirResult{result,w_accum,w_instant};
}

void SpikeEngine::get_reservoir_features_vector(s64 tick, f32 spike_tau, f32 voltage_scale, GpuPointer<f32> output_buffer) {
    logger->trace("get_reservoir_features_vector: tick={} spike_tau={} voltage_scale={}", tick, spike_tau, voltage_scale);
    if (spike_tau <= 0.0f) {
        logger->warn("get_reservoir_features_vector: spike_tau was <= 0.0. Aborting.");
        return;
    }
    if (voltage_scale <= 0.0f) {
        logger->warn("get_reservoir_features_vector: voltage_scale was <= 0.0. Aborting.");
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

    logger->info("shutdown: releasing GPU buffers");

    if (cell_state_logs != nullptr) {
        for (s64 i = 0; i < neuron_count; ++i)
            delete[] cell_state_logs[i];
        delete[] cell_state_logs;
        cell_state_logs = nullptr;
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
    deallocate(std::move(input_staging));
    deallocate(std::move(override_staging));

    running = false;
}
























