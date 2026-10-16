//
// Created by Alek Simpson on 5/30/26.
//
#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "types.h"
#include "backend.h"
#include "log.h"
#include "weight_matrix.h"
#include "recording.h"
#include "parser.h"

using namespace std;
using namespace spikecorec::nml;

namespace spikecorec {

    // last_spiked of a neuron that never fired. The kernel detects it by sign (last_spiked < 0).
    constexpr s64 NEVER_SPIKED_TICK = -1;

    struct Recorded_Spike {
        f64 time_seconds = 0.0;
        s64 neuron_index = -1;
    };


    class Spike_Engine {
    public:
        Shared_Pointer<log::Engine_Logger> logger;

        NML_Context context;

        Engine_Backend gpu;

        Weight_Matrix weights;

        Engine_Function kernel_function;

        // master_step's parameters in binding order and their sources; tick is bound per dispatch.
        Vector<String> kernel_parameter_names;
        Unordered_Map<String, Function<Engine_Pointer()>> kernel_argument_sources;
        // The master kernel source as compiled, with its constants baked in.
        String master_kernel_source;
        // Ticks after a spike during which a neuron's outgoing synapses run: the longest delay
        // plus the slowest synapse's settle time. -1 runs every synapse every tick.
        s64 synapse_active_ticks = -1;
        s32 rank_float4_stride_argument = 0;
        // Extra walk passes the run has needed because the update queue was full.
        s64 deferred_walk_pass_count = 0;

        // Construction's calibration: planes that start at zero (synapse state) get their lane
        // budgets from values they take when every neuron spikes once, each at its own tick, as in
        // asynchronous activity. The engine runs this many ticks, then restores every state the
        // ticks changed, so the run starts from the model's own starting state. 0 when no plane
        // needed it.
        s64 calibration_tick_count = 0;
        // The most ticks calibration waits for synapses to settle when they never do
        // (synapse_active_ticks of -1).
        static constexpr s64 CALIBRATION_TICK_LIMIT = 2000;
        // Calibration spreads the forced spikes over about one time constant of the slowest
        // synapse: synapse_active_ticks is about the longest delay plus ln(1 / REST_FRACTION),
        // about 14, of those time constants, so the spread is a fourteenth of it.
        static constexpr s64 CALIBRATION_SPIKE_SPREAD_DIVISOR = 14;
        // The tick's time in seconds, computed on the host each tick. f32 on Metal, which has no f64.
#ifdef SPIKECOREC_CUDA
        f64 simulation_time_argument = 0.0;
#else
        f32 simulation_time_argument = 0.0f;
#endif
        s32 run_count = 0;

        // The slab Codegen::allocate_cell_model_memory carves the model buffers from.
        Engine_Pointer model_pointer;

        // The slab holding the device copy of the runs, for the per-edge synapse prototype lookup.
        Engine_Pointer run_pointer;
        Engine_Pointer run_first_edge_ordinal;  // s64[run count]
        Engine_Pointer run_synapse_prototype;   // s32[run count]

        Engine_Pointer cell_state;

        // Two rows alternating by tick parity: a neuron drains one row while this tick's scatters
        // land in the other, so synaptic latency is exactly one tick whatever the thread order.
        Engine_Pointer network_inputs;     // [2][total_neuron_count]

        Engine_Pointer empty_edge_plane;

        // Spikes that reached each cell, for cells with an OnEvent: two rows by tick parity like
        // network_inputs. One slot when no cell type has an OnEvent.
        Engine_Pointer event_arrival_count;  // u32[2][total_neuron_count]

        // [spike_history_length][total_neuron_count] spike flags. A delayed arrival checks
        // whether the source spiked `delay` ticks ago.
        Engine_Pointer spike_history;
        Engine_Pointer last_spiked;        // [total_neuron_count]

        // Seeded with simulation_seed; draws OnStart random() values, then random_values per tick.
        Random_Generator random_generator;
        // Slab for this tick's uniform draws: one per random() call per neuron or input entry.
        Engine_Pointer random_values_pointer;
        Engine_Pointer random_values;      // f32[max(1, random_values_count)]
        s64 random_values_count = 0;

        // Kernel-run inputs, carved from model_pointer. An entry is one input on one target cell.
        Engine_Pointer input_row_start;        // s64[total_neuron_count + 1]: neuron n's first entry
        Engine_Pointer input_entry_prototype;  // s32[entries]: which input each entry runs
        Engine_Pointer input_values;           // f32[entries][input_value_stride]: weight, then state
        Engine_Pointer input_spike_counts;     // u8: each spike train's spikes per tick
        s64 input_entry_count = 0;
        s64 input_value_stride = 1;
        s64 input_spike_count_size = 0;

        // The results below grow during a run; another thread reads them only under results_mutex.
        mutable Mutex results_mutex;
        Vector<s64> spike_counts_per_neuron;
        Vector<Recorded_Spike> recorded_spikes;
        // row-major [recorded tick][traced quantity]
        Vector<f32> recorded_traces;
        Vector<Recording_Selection> traced_selections;
        Vector<f64> recorded_trace_times;

        Unique_Pointer<Simulation_Recorder> membrane_video_recorder;
        s64 membrane_video_frame_stride = 1;
        // Each neuron's `v` index in cell_state, so a frame is a gather, not a name lookup.
        Vector<s64> membrane_offset_per_neuron;
        Vector<f32> membrane_frame_scratch;

        s64 total_neuron_count = 0;
        s64 spike_history_row_count = 0;
        s64 lifetime = 0;
        // Ticks stepped so far, which is also the next tick step_simulation accepts. Written under
        // results_mutex, with the tick's results.
        s64 ticks_stepped = 0;
        f64 step_dt = 0.0f;
        u64 simulation_seed = 0;

        bool alive = false;

        bool hebbian_plasticity_enabled = false;

        static constexpr s64 DEFAULT_PLASTICITY_DELTA_CAPACITY = 1 << 16;

        // Presets for fit_tolerance: the worst relative error a fit may leave on a per-edge value,
        // against the larger of the value and its plane's RMS. Tighter costs rank and fit time.
        // A delay stays exact while tolerance * delay < 0.5, so each lists its longest exact delay.
        static constexpr f32 FIT_TOLERANCE_PRECISE = 1e-5f;   // reference simulators; 50000 ticks
        static constexpr f32 FIT_TOLERANCE_ACCURATE = 1e-4f;  // validation runs; 5000 ticks
        static constexpr f32 FIT_TOLERANCE_STANDARD = Weight_Matrix::DEFAULT_FIT_TOLERANCE;  // default; 500 ticks
        static constexpr f32 FIT_TOLERANCE_COMPACT = 1e-2f;   // the largest networks; 50 ticks

        // Used by the construction fit and every refit.
        f32 fit_tolerance = FIT_TOLERANCE_STANDARD;

        // The candidate fit the construction fit and every refit use.
        Weight_Matrix::Fit_Method fit_method = Weight_Matrix::Fit_Method::MASKED_ALTERNATING_LEAST_SQUARES;

        // The expected fraction of edges whose values change between refits (0 to 1), which sizes
        // every plane's fixed lane budget at construction; see Weight_Matrix.
        f32 expected_active_edge_fraction = Weight_Matrix::DEFAULT_EXPECTED_ACTIVE_EDGE_FRACTION;

        // Refit triggers, checked after each tick's merge; 0 disables either. A refit runs every
        // refit_every_n_ticks, or once the fullest plane of S covers this fraction of the edges,
        // which also sizes S. Set the fraction with set_refit_occupancy_threshold_fraction.
        s64 refit_every_n_ticks = 0;
        f32 refit_occupancy_threshold_fraction = Weight_Matrix::DEFAULT_REFIT_OCCUPANCY_THRESHOLD_FRACTION;

        // Minimum ticks between refits, so an S that refills every tick grows instead of
        // refitting every tick. 0 allows any spacing.
        s64 minimum_ticks_between_refits = 1000;
        // The construction fit counts as a refit at tick 0.
        s64 last_refit_tick = 0;
        s64 plasticity_fold_every_n_ticks = 64;
        f32 plasticity_learning_rate = 0.01f;
        f32 plasticity_l2_regularization = 1e-6f;
        s32 plasticity_iterations = 1;

        f32 plasticity_target_root_mean_square = -1.0f;

        Spike_Engine() = delete;
        Spike_Engine(const Spike_Engine &) = delete;
        Spike_Engine &operator=(const Spike_Engine &) = delete;
        Spike_Engine(Spike_Engine &&) = delete;
        Spike_Engine &operator=(Spike_Engine &&) = delete;

        explicit Spike_Engine(const String &lems_input_file,
                             bool enable_hebbian_plasticity = false,
                             f32 fit_tolerance = FIT_TOLERANCE_STANDARD,
                             Weight_Matrix::Fit_Method fit_method = Weight_Matrix::Fit_Method::MASKED_ALTERNATING_LEAST_SQUARES,
                             f32 expected_active_edge_fraction = Weight_Matrix::DEFAULT_EXPECTED_ACTIVE_EDGE_FRACTION);

        Spike_Engine(const String &lems_input_file,
                    const Vector<Vector<s32>> &adjacency,
                    const String &synapse_component_id,
                    f64 connection_weight = 1.0,
                    f64 connection_delay_seconds = 0.0,
                    bool enable_hebbian_plasticity = false,
                    f32 fit_tolerance = FIT_TOLERANCE_STANDARD,
                    Weight_Matrix::Fit_Method fit_method = Weight_Matrix::Fit_Method::MASKED_ALTERNATING_LEAST_SQUARES,
                    f32 expected_active_edge_fraction = Weight_Matrix::DEFAULT_EXPECTED_ACTIVE_EDGE_FRACTION);

        Spike_Engine(const String &lems_input_file,
                    const Vector<Vector<s32>> &adjacency,
                    const Vector<String> &synapse_component_ids,
                    const Vector<f64> &synapse_proportions = {},
                    f64 connection_weight = 1.0,
                    f64 connection_delay_seconds = 0.0,
                    bool enable_hebbian_plasticity = false,
                    f32 fit_tolerance = FIT_TOLERANCE_STANDARD,
                    Weight_Matrix::Fit_Method fit_method = Weight_Matrix::Fit_Method::MASKED_ALTERNATING_LEAST_SQUARES,
                    f32 expected_active_edge_fraction = Weight_Matrix::DEFAULT_EXPECTED_ACTIVE_EDGE_FRACTION);

        ~Spike_Engine();

        // Steps every tick of the run not yet stepped, so it resumes a stopped run. after_each_tick,
        // when given, is called with each tick once it is done; an exception it throws stops the
        // run. Throws once every tick has run: the engine holds one run's state.
        void run(const Function<void(s64)> &after_each_tick = nullptr);

        // Ticks run in order: tick must be ticks_stepped, below lifetime. The kernel's parity rows,
        // rings and spike trains assume it. Throws otherwise, and after shutdown().
        void step_simulation(s64 tick);

        // Sets the refit threshold and resizes S to match, never below the updates it holds.
        void set_refit_occupancy_threshold_fraction(f32 fraction);

        // The state variable named `variable_name` for one neuron, read back from the GPU.
        [[nodiscard]] f32 read_state_variable(s64 neuron_index, const String &variable_name) const;

        // The weight-matrix plane holding StateVariable `variable_name` of the named synapse.
        [[nodiscard]] s64 synapse_state_variable_plane(const String &synapse_component_id,
                                                       const String &variable_name) const;

        // Spikes per neuron per second over the ticks stepped so far, and the fraction of neurons
        // that spiked at least once.
        [[nodiscard]] f64 mean_firing_rate_hertz() const;
        [[nodiscard]] f64 fraction_of_neurons_that_spiked() const;

        // Writes every OutputFile and EventOutputFile the document declares, a relative fileName
        // relative to the LEMS file. Throws when a file cannot be written.
        void write_recordings();

        void record_membrane_video(const String &path, s64 frame_stride = 1);

        // Writes every spike as "time<tab>neuron" (an EventOutputFile's TIME_ID format) without
        // needing one EventSelection per neuron. Call after run(). Throws when it cannot write.
        void write_spike_file(const String &path) const;

        void shutdown();

    private:
        // The slab starts uninitialized: zeroes the accumulators and spike history, and marks
        // every neuron as never spiked.
        void initialize_model_buffers();

        // Builds the weight matrix from network_data: planes weight, delay, then synapse state
        // variables, fitted from each run's starting values. Also copies the runs to the device.
        void build_weight_matrix(s64 matrix_count, s64 updated_plane_count);

        // Fills kernel_argument_sources; throws on a master_step parameter with no source.
        void register_kernel_arguments();

        // The population neuron_index belongs to, with its first neuron, or nullptr.
        [[nodiscard]] const NML_Component_Instance *population_of_neuron(s64 neuron_index, s64 &first_neuron) const;

        // Where one neuron's variable sits in cell_state.
        [[nodiscard]] s64 cell_memory_index_of(s64 neuron_index, const String &variable_name) const;

        // Replaces the document's connections with `adjacency`, split across the synapses by
        // proportion. Runs before codegen, so everything downstream sees an ordinary parse result.
        void apply_topology(const Vector<Vector<s32>> &adjacency,
                            const Vector<String> &synapse_component_ids,
                            const Vector<f64> &synapse_proportions,
                            f64 connection_weight,
                            f64 connection_delay_seconds);

        // `plane`, or empty_edge_plane when the model has no such buffer.
        [[nodiscard]] Engine_Pointer resolve_edge_plane(const Engine_Pointer &plane) const;

        void record_tick(s64 tick);

        // One tick's kernel work: draws the random values, dispatches master_step, merges the queue
        // into S and runs the deferred walk passes. Neither records nor refits.
        void dispatch_tick(s64 tick, bool plasticity_fold_due);

        // Gives each plane that starts at zero its lane budget from calibration ticks (see
        // calibration_tick_count), and restores the starting state afterwards.
        void calibrate_zero_start_planes();
    };
} // namespace spikecorec
