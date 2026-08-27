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

        // ── aggregated synapse state ──────────────────────────────────────────────
        // Per-edge synapse state does not exist. Every Phase-1 synapse has linear state
        // dynamics, per-prototype parameters, and a current linear in that state, with
        // `weight` its only per-edge quantity and entering the arrival increment linearly.
        // So the sum of a target's incoming synapse states obeys the same equation each
        // term does, and one accumulator per (target, prototype) reproduces the per-edge
        // computation exactly rather than approximately.
        //
        // That is worth roughly 400 MB per state variable at a million neurons -- and it
        // is faster besides: an edge now touches memory only when a spike actually
        // arrives, instead of loading and storing its state on every tick.
        //
        // Arrivals are double-buffered by tick parity for the same reason network_inputs
        // is: scatters land in the next row while the target drains the current one.
        EnginePointer synapse_arrivals;   // [2][prototype_count][total_neuron_count]
        EnginePointer synapse_state;      // [prototype_count][state_count][total_neuron_count]

        // Bound wherever the kernel declares a per-edge buffer the model has no plane for.
        // A model with no connections registers none of them, and there is no way to bind
        // "nothing" to a `device float *` a kernel declares. One float is enough: the
        // kernel never reads it, because a neuron with no adjacency row leaves the
        // propagate walk before its first load.
        EnginePointer empty_edge_plane;

        // The whole-chunk handle for everything above, released at shutdown.
        EnginePointer model_slab;

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
        std::unique_ptr<SimulationRecorder> membrane_video_recorder;
        s64 membrane_video_frame_stride = 1;
        // Where each neuron's `v` sits in cell_state, precomputed so a frame is a gather
        // rather than a per-neuron name lookup.
        Vector<s64> membrane_offset_per_neuron;
        Vector<f32> membrane_frame_scratch;

        s64 total_neuron_count = 0;
        s64 lifetime = 0;
        f32 step_dt = 0.0f;
        u64 simulation_seed = 0;

        bool alive = false;

        // ── plasticity (opt-in) ───────────────────────────────────────────────────
        bool hebbian_plasticity_enabled = false;

        // How many staged deltas are folded into U/V at once, and how often. Capacity is a
        // fixed budget rather than one slot per edge -- an interval that overflows it loses
        // the excess and logs, which says "fold more often" rather than growing without
        // bound.
        static constexpr s64 DEFAULT_PLASTICITY_DELTA_CAPACITY = 1 << 16;

        // How much of the edge set the weight matrix's corrections may occupy. At 1.0 every
        // model is reproduced exactly, and one with no exploitable structure pays for that
        // in memory rather than in silently wrong weights. Lower it to cap what such a
        // model may spend; a well-structured model needs none of it either way.
        f32 correction_ceiling_fraction = 1.0f;

        // The largest rank the weight matrix's general fit may spend, or -1 to search for
        // the rank whose basis and corrections cost the fewest bytes together. Searching is
        // the default because the cheapest rank is a property of the model: an
        // incompressible field gains nothing from extra rank and wants the smallest, while
        // a structured one is worth spending on.
        s64 weight_fit_rank_budget = -1;
        s64 plasticity_fold_every_n_ticks = 64;
        f32 plasticity_learning_rate = 0.01f;
        f32 plasticity_l2_regularization = 1e-6f;
        s32 plasticity_iterations = 1;

        // Homeostatic scaling, run after each fold. Plain Hebbian only ever strengthens, so
        // without this the weights run away: a potentiated edge fires its target more, which
        // potentiates it further. Rescaling the basis back to the magnitude the model
        // declared keeps the total synaptic drive fixed while letting edges move relative to
        // each other, which is the part the rule is actually for.
        //
        // Captured at construction from the model's own weights, so it is the document's
        // scale being preserved rather than an invented one. Set to a negative value to
        // disable and let the weights grow.
        f32 plasticity_target_root_mean_square = -1.0f;

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
        // enable_hebbian_plasticity switches on the engine's built-in Hebbian rule, which
        // nudges U/V toward a stronger reconstruction for edges whose endpoints fired
        // close together. Off by default: it changes what the simulation computes, and no
        // NeuroML document asks for it.
        //
        // Off costs nothing rather than costing a branch -- the codegen emits no
        // plasticity block at all, and the delta buffers are never allocated.
        explicit SpikeEngine(const String &lems_input_file,
                             bool enable_hebbian_plasticity = false);

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
