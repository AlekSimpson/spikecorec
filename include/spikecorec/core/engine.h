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
#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/dynamics_codegen.h"

using namespace std;
using namespace spikecorec::nml;

namespace spikecorec {

    // constexpr s64 NEVER_SPIKED_TICK = -((s64)1 << 32);

    // One spike, as the model asked for it to be recorded.
    struct RecordedSpike {
        f64 time_seconds = 0.0;
        s64 neuron_index = -1;
    };


    class SpikeEngine {
    public:
        log::SharedPointer<log::EngineLogger> logger;

        EngineBackend gpu;

        WeightMatrix weights;

        NML_ParseResult network_details;
        ModelLayout layout;

        EngineFunction kernel_function;

        // ── model state, carved from the arena ────────────────────────────────────
        EnginePointer cell_state;         // [layout.cell_state_length]
        EnginePointer cell_parameters;    // [layout.cell_parameter_length]
        EnginePointer synapse_parameters; // [layout.synapse_parameter_length]

        // Two rows, alternating by tick parity: a thread drains its slot in one row while
        // this tick's scatters accumulate into the other. That is what makes the synaptic
        // latency exactly one tick rather than one-or-two depending on thread order.
        EnginePointer network_inputs;     // [2][total_neuron_count]

        // [spike_history_length][total_neuron_count]. A delayed arrival is answered by
        // asking whether the source spiked `delay` ticks ago, so every spike in flight is
        // remembered, not just the most recent one.
        // GpuPointer<void> spike_history;
        EnginePointer last_spiked;        // [total_neuron_count]

        // Host-side stimulus, applied before each dispatch.
        Vector<s64> continuous_injection_targets;
        Vector<f32> continuous_injection_amplitudes;
        Vector<s64> continuous_injection_start_ticks;
        Vector<s64> continuous_injection_end_ticks;

        // A scheduled spike train: one entry per (input profile, target) that carries
        // event times, plus a cursor into them. Kept sparse rather than expanded into a
        // dense [target][tick] array, which for a long run is almost all zeros.
        //
        // Each event injects `magnitude` for exactly one tick, so it delivers a charge of
        // magnitude * dt. That is what makes the amplitude a nanoamp-scale number rather
        // than the picoamps a sustained injector uses: to move a 100 pF membrane by 15 mV
        // in one 0.1 ms tick takes 15 nA.
        struct ScheduledSpikeTrain {
            s64 neuron_index = -1;
            f32 magnitude = 0.0f;
            Vector<s32> event_ticks;
            usize cursor = 0;
        };
        Vector<ScheduledSpikeTrain> scheduled_spike_trains;

        // ── recording ─────────────────────────────────────────────────────────────
        // Every neuron's spike count over the run, accumulated tick by tick. Kept
        // separately from recorded_spikes because the aliveness metrics need every
        // neuron's count whether or not the model asked for that neuron to be recorded.
        Vector<s64> spike_counts_per_neuron;
        Vector<RecordedSpike> recorded_spikes;
        // Row-major [recorded tick][traced quantity], parallel to traced_selections.
        Vector<f32> recorded_traces;
        Vector<RecordingSelection> traced_selections;
        Vector<f64> recorded_trace_times;

        // Set up by record_membrane_video(): a .spire recording of every neuron's
        // membrane potential, which is what the video renderer consumes. Separate from the
        // model's own OutputFiles because LEMS has no way to ask for "every neuron, every
        // Nth tick", and pretending one of its elements meant that would be inventing
        // semantics the format does not have.
        // std::unique_ptr<SimulationRecorder> membrane_video_recorder;
        // s64 membrane_video_frame_stride = 1;
        // Where each neuron's `v` sits in cell_state, precomputed so a frame is a gather
        // rather than a per-neuron name lookup.
        Vector<s64> membrane_offset_per_neuron;
        Vector<f32> membrane_frame_scratch;

        s64 total_neuron_count = 0;
        s64 lifetime = 0;
        f32 step_dt = 0.0f;
        u64 simulation_seed = 0;

        bool alive = false;

        SpikeEngine() = delete;
        SpikeEngine(const SpikeEngine &) = delete;
        SpikeEngine &operator=(const SpikeEngine &) = delete;
        SpikeEngine(SpikeEngine &&) = delete;
        SpikeEngine &operator=(SpikeEngine &&) = delete;

        // Parses `lems_input_file`, allocates every buffer the model needs, fills them from
        // the model's starting parameters and OnStart initialisers, builds the weight matrix
        // from the model's connections, and compiles the generated tick kernel. Throws
        // naming the offending ComponentType if the model uses anything Phase 1 does not
        // simulate, rather than loading something it would run incorrectly.
        explicit SpikeEngine(const String &lems_input_file);

        // The same, with the network's connectivity supplied in code instead of in the
        // document. The model still declares the cells, the synapse, the stimulus and the
        // run; `adjacency` says which neuron reaches which, and every edge it describes
        // uses `synapse_component_id` with the given weight and delay.
        //
        // This is how a large network is built: the topology helpers in topologies.h
        // (square_torus and friends) generate millions of edges in a loop, where writing
        // them as <connection> elements would mean millions of lines of XML. `adjacency`
        // must have one row per neuron the model's populations declare.
        SpikeEngine(const String &lems_input_file,
                    const vector<vector<s32>> &adjacency,
                    const String &synapse_component_id,
                    f64 connection_weight = 1.0,
                    f64 connection_delay_seconds = 0.0);

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
        void allocate_model_buffers();
        void initialize_cell_state();
        void build_weight_matrix();
        void collect_stimulus();

        // Replaces whatever connections the document declared with `adjacency`, all
        // carrying one synapse prototype. Runs before the layout is computed, so
        // everything downstream sees an ordinary parse result.
        void apply_topology(const vector<vector<s32>> &adjacency,
                            const String &synapse_component_id,
                            f64 connection_weight,
                            f64 connection_delay_seconds);

        void apply_stimulus(s64 tick);

        void *resolve_edge_plane(s64 matrix_index);

        // The current one event of a spike train injects when the model names no
        // amplitude: enough charge, in a single tick, to carry this neuron from where it
        // starts to where it fires. Derived from the target's own declared quantities, so
        // it follows whatever cell the train is wired to.
        [[nodiscard]] f64 default_spike_amplitude_for(s64 neuron_index) const;

        void record_tick(s64 tick);
    };
} // namespace spikecorec
