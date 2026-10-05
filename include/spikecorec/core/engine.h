//
// Created by Alek Simpson on 5/30/26.
//
#pragma once

#include <functional>
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

    struct RecordedSpike {
        f64 time_seconds = 0.0;
        s64 neuron_index = -1;
    };


    class SpikeEngine {
    public:
        log::SharedPointer<log::EngineLogger> logger;

        NML_Context context;

        EngineBackend gpu;

        WeightMatrix weights;

        EngineFunction kernel_function;

        // master_step's parameters in binding order, and where each one's buffer or value
        // comes from. tick is bound per dispatch.
        Vector<String> kernel_parameter_names;
        UnorderedMap<String, std::function<EnginePointer()>> kernel_argument_sources;
        // The generated master kernel exactly as it was compiled, every constant baked in.
        String master_kernel_source;
        // Ticks after a neuron spikes during which its outgoing synapses run: the longest delay
        // plus the slowest synapse's settle time (Codegen::synapse_active_ticks). -1 when every
        // synapse runs every tick.
        s64 synapse_active_ticks = -1;
        s32 rank_float4_stride_argument = 0;
        // Bound per tick rather than baked: S is resized after a refit when the threshold changed.
        s64 sparse_delta_capacity_argument = 0;
        s32 projection_run_count = 0;

        // The one allocation every model buffer below is carved from.
        EnginePointer model_pointer;

        // The weight matrix's projection runs, on the device, so the kernel can find each
        // edge's synapse prototype. Carved from projection_run_pointer.
        EnginePointer projection_run_pointer;
        EnginePointer projection_first_edge_ordinal;  // s64[run count]
        EnginePointer projection_synapse_prototype;   // s32[run count]

        EnginePointer cell_state;

        // Two rows, alternating by tick parity: a thread drains its slot in one row while
        // this tick's scatters accumulate into the other. That is what makes the synaptic
        // latency exactly one tick rather than one-or-two depending on thread order.
        EnginePointer network_inputs;     // [2][total_neuron_count]

        EnginePointer empty_edge_plane;

        // [spike_history_length][total_neuron_count]. A delayed arrival is answered by
        // asking whether the source spiked `delay` ticks ago, so every spike in flight is
        // remembered, not just the most recent one.
        EnginePointer spike_history;
        EnginePointer last_spiked;        // [total_neuron_count]

        // Draws every random() the run calls, seeded with simulation_seed: OnStart values at
        // construction, then random_values before every tick.
        RandomGenerator random_generator;
        // This tick's uniform draws, one slot per random() call per neuron that runs it
        // (Codegen::random_values_count). Carved from random_values_pointer.
        EnginePointer random_values_pointer;
        EnginePointer random_values;      // f32[max(1, random_values_count)]
        s64 random_values_count = 0;

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
        // row-major [recorded tick][traced quantity]
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
        s64 spike_history_row_count = 0;
        s64 lifetime = 0;
        f64 step_dt = 0.0f;
        u64 simulation_seed = 0;

        bool alive = false;

        bool hebbian_plasticity_enabled = false;

        static constexpr s64 DEFAULT_PLASTICITY_DELTA_CAPACITY = 1 << 16;

        s64 weight_fit_rank_budget = -1;

        // Presets for fit_tolerance: the worst relative error the weight matrix may leave on any
        // per-edge value of a synapse, against the larger of the value and its plane's RMS.
        // Tighter costs rank (memory) and fitting time. A delay stays on its exact tick while
        // tolerance * delay is under half a tick, so each lists the longest delay it keeps exact.
        static constexpr f32 FIT_TOLERANCE_PRECISE = 1e-5f;   // comparisons against reference simulators; 50000 ticks
        static constexpr f32 FIT_TOLERANCE_ACCURATE = 1e-4f;  // validation runs and reported results; 5000 ticks
        static constexpr f32 FIT_TOLERANCE_STANDARD = WeightMatrix::DEFAULT_FIT_TOLERANCE;  // the default; 500 ticks
        static constexpr f32 FIT_TOLERANCE_COMPACT = 1e-2f;   // the largest networks, where memory matters most; 50 ticks

        // Applies to the fit at construction and to every refit after it.
        f32 fit_tolerance = FIT_TOLERANCE_STANDARD;

        // When the weight matrix refits during a run, folding the updates in S into the basis
        // and emptying S. Both are checked after each tick's merge, and 0 disables either:
        // every this many ticks, and once the fullest plane of S holds updates on this
        // fraction of the edge set -- which is also the size S is kept at between refits. Set
        // the threshold with set_refit_occupancy_threshold_fraction, which resizes S to it.
        s64 refit_every_n_ticks = 0;
        f32 refit_occupancy_threshold_fraction = WeightMatrix::DEFAULT_REFIT_OCCUPANCY_THRESHOLD_FRACTION;

        // Refits are at least this many ticks apart, so a busy network whose S refills every
        // tick does not refit every tick; S grows in between instead. 0 allows any spacing.
        s64 minimum_ticks_between_refits = 1000;
        // The construction fit counts as the first.
        s64 last_refit_tick = 0;
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
                             bool enable_hebbian_plasticity = false,
                             f32 fit_tolerance = FIT_TOLERANCE_STANDARD);

        SpikeEngine(const String &lems_input_file,
                    const vector<vector<s32>> &adjacency,
                    const String &synapse_component_id,
                    f64 connection_weight = 1.0,
                    f64 connection_delay_seconds = 0.0,
                    bool enable_hebbian_plasticity = false,
                    f32 fit_tolerance = FIT_TOLERANCE_STANDARD);

        SpikeEngine(const String &lems_input_file,
                    const vector<vector<s32>> &adjacency,
                    const vector<String> &synapse_component_ids,
                    const vector<f64> &synapse_proportions = {},
                    f64 connection_weight = 1.0,
                    f64 connection_delay_seconds = 0.0,
                    bool enable_hebbian_plasticity = false,
                    f32 fit_tolerance = FIT_TOLERANCE_STANDARD);

        ~SpikeEngine();

        void run();

        void step_simulation(s64 tick);

        // Sets the refit threshold, which is also the size S is kept at per plane, and resizes S
        // to it now. S keeps the updates it holds, so it never shrinks below them.
        void set_refit_occupancy_threshold_fraction(f32 fraction);

        // The state variable named `variable_name` for one neuron, read back from the GPU.
        [[nodiscard]] f32 read_state_variable(s64 neuron_index, const String &variable_name) const;

        // The weight-matrix plane that holds `variable_name`, one of the named synapse's LEMS
        // StateVariables, on every edge that synapse carries.
        [[nodiscard]] s64 synapse_state_variable_plane(const String &synapse_component_id,
                                                       const String &variable_name) const;

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
        // Zeroes and seeds what Codegen::allocate_cell_model_memory only reserved. A slab
        // is uninitialized memory and every one of these is read before it is written on
        // the first tick.
        void initialize_model_buffers();

        void collect_stimulus();

        // Builds the k^2-tree and the weight and delay basis from network_data, with
        // matrix_count planes: weight, delay and one per synapse state variable. Also writes
        // the synapse planes' starting state and the device copy of the projection runs.
        void build_weight_matrix(s64 matrix_count, s64 updated_plane_count);

        // Fills kernel_argument_sources, and refuses a master_step parameter the engine has
        // nothing to bind to.
        void register_kernel_arguments();

        // The population neuron_index belongs to, with its first neuron, or nullptr.
        [[nodiscard]] const NML_ComponentInstance *population_of_neuron(s64 neuron_index, s64 &first_neuron) const;

        // Where one neuron's variable sits in cell_state.
        [[nodiscard]] s64 cell_memory_index_of(s64 neuron_index, const String &variable_name) const;

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
