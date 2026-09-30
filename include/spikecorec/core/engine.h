//
// Created by Alek Simpson on 5/30/26.
//
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "spikecorec/core/types.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/core/recording.h"
#include "spikecorec/nml/parser.h"

using namespace std;
using namespace spikecorec::nml;

namespace spikecorec {

    // last_spiked holds this for a neuron that has never fired. Negative rather than a
    // large negative magic number: the refractory gate branches on the sign, so "never
    // fired" is a state rather than "fired so long ago the arithmetic works out".
    constexpr s64 NEVER_SPIKED_TICK = -1;

    // One spike, as the model asked for it to be recorded.
    struct RecordedSpike {
        f64 time_seconds = 0.0;
        s64 neuron_index = -1;
    };


    class SpikeEngine {
    public:
        log::SharedPointer<log::EngineLogger> logger;

        // Lives as long as the engine: its instances point into its own component types
        // and document trees, which stay valid until the engine is destroyed.
        NML_Context context;

        EngineBackend gpu;

        WeightMatrix weights;

        EngineFunction kernel_function;

        EnginePointer cell_state;

        // Two rows, alternating by tick parity: a thread drains its slot in one row while
        // this tick's scatters accumulate into the other. That is what makes the synaptic
        // latency exactly one tick rather than one-or-two depending on thread order.
        EnginePointer network_inputs;     // [2][total_neuron_count]

        // Bound wherever the kernel declares a per-edge buffer the model has no plane for.
        // A model with no connections registers none of them
        EnginePointer empty_edge_plane;

        // [spike_history_length][total_neuron_count]. A delayed arrival is answered by
        // asking whether the source spiked `delay` ticks ago, so every spike in flight is
        // remembered, not just the most recent one.
        EnginePointer spike_history;
        EnginePointer last_spiked;        // [total_neuron_count]

        // Host-side stimulus, applied before each dispatch.
        Vector<s64> continuous_injection_targets;
        Vector<f32> continuous_injection_amplitudes;
        Vector<s64> continuous_injection_start_ticks;
        Vector<s64> continuous_injection_end_ticks;

        struct ScheduledSpikeTrain {
            s64 neuron_index = -1;
            f32 magnitude = 0.0f;
            Vector<s32> event_ticks;
            usize cursor = 0;
        };
        Vector<ScheduledSpikeTrain> scheduled_spike_trains;

        Vector<s64> spike_counts_per_neuron;
        Vector<RecordedSpike> recorded_spikes;
        // Row-major [recorded tick][traced quantity], parallel to traced_selections.
        Vector<f32> recorded_traces;
        Vector<RecordingSelection> traced_selections;
        Vector<f64> recorded_trace_times;

        std::unique_ptr<SimulationRecorder> membrane_video_recorder;
        s64 membrane_video_frame_stride = 1;
        // Where each neuron's `v` sits in cell_state, precomputed so a frame is a gather
        // rather than a per-neuron name lookup.
        Vector<s64> membrane_offset_per_neuron;
        Vector<f32> membrane_frame_scratch;

        s64 total_neuron_count = 0;
        s64 lifetime = 0;
        f64 step_dt = 0.0f;
        u64 simulation_seed = 0;

        bool alive = false;

        bool hebbian_plasticity_enabled = false;

        static constexpr s64 DEFAULT_PLASTICITY_DELTA_CAPACITY = 1 << 16;

        f32 correction_ceiling_fraction = 1.0f;

        s64 weight_fit_rank_budget = -1;
        s64 plasticity_fold_every_n_ticks = 64;
        f32 plasticity_learning_rate = 0.01f;
        f32 plasticity_l2_regularization = 1e-6f;
        s32 plasticity_iterations = 1;

        f32 plasticity_target_root_mean_square = -1.0f;

        SpikeEngine() = delete;
        SpikeEngine(const SpikeEngine &) = delete;
        SpikeEngine &operator=(const SpikeEngine &) = delete;
        SpikeEngine(SpikeEngine &&) = delete;
        SpikeEngine &operator=(SpikeEngine &&) = delete;

        explicit SpikeEngine(const String &lems_input_file,
                             bool enable_hebbian_plasticity = false);

        SpikeEngine(const String &lems_input_file,
                    const vector<vector<s32>> &adjacency,
                    const String &synapse_component_id,
                    f64 connection_weight = 1.0,
                    f64 connection_delay_seconds = 0.0,
                    bool enable_hebbian_plasticity = false);

        SpikeEngine(const String &lems_input_file,
                    const vector<vector<s32>> &adjacency,
                    const vector<String> &synapse_component_ids,
                    const vector<f64> &synapse_proportions = {},
                    f64 connection_weight = 1.0,
                    f64 connection_delay_seconds = 0.0,
                    bool enable_hebbian_plasticity = false);

        ~SpikeEngine();

        void run();

        void step_simulation(s64 tick);

        // The state variable named `variable_name` for one neuron, read back from the GPU.
        [[nodiscard]] f32 read_state_variable(s64 neuron_index, const String &variable_name) const;

        // Spikes per neuron per second over the whole run, and the fraction of neurons that
        // spiked at least once. What a demo has to clear to count as alive.
        [[nodiscard]] f64 mean_firing_rate_hertz() const;
        [[nodiscard]] f64 fraction_of_neurons_that_spiked() const;

        void write_recordings();

        void record_membrane_video(const String &path, s64 frame_stride = 1);

        // Every spike of the run as "time<tab>neuron", which is the TIME_ID form an
        // EventOutputFile writes. Here for the same reason record_membrane_video is: LEMS
        // names its recorded cells one EventSelection at a time, and a million of those is
        // a million elements to express "all of them". Call after run().
        void write_spike_file(const String &path) const;

        void shutdown();

    private:
        // Carves every model buffer out of one slab. Runs after the layout is known,
        // because that is what sizes them.
        void allocate_model_buffers();

        // Zeroes and seeds what allocate_model_buffers only reserved. Separate because a
        // slab is uninitialized memory and every one of these is read before it is
        // written on the first tick.
        void initialize_model_buffers();

        void initialize_cell_state();

        void collect_stimulus();

        void log_weight_matrix();

        // Replaces whatever connections the document declared with `adjacency`, all
        // carrying one synapse prototype. Runs before the layout is computed, so
        // everything downstream sees an ordinary parse result.
        void apply_topology(const vector<vector<s32>> &adjacency,
                            const vector<String> &synapse_component_ids,
                            const vector<f64> &synapse_proportions,
                            f64 connection_weight,
                            f64 connection_delay_seconds);

        void apply_stimulus(s64 tick);

        // `plane`, or empty_edge_plane when the model has no such buffer.
        [[nodiscard]] EnginePointer resolve_edge_plane(const EnginePointer &plane) const;


        // The current one event of a spike train injects when the model names no
        // amplitude: enough charge, in a single tick, to carry this neuron from where it
        // starts to where it fires. Derived from the target's own declared quantities, so
        // it follows whatever cell the train is wired to.
        [[nodiscard]] f64 default_spike_amplitude_for(s64 neuron_index) const;

        void record_tick(s64 tick);
    };
} // namespace spikecorec
