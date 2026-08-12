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
#include "spikecorec/core/engine_allocator.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/core/recording.h"
#include "spikecorec/nml/nml.h"

using namespace std;
using namespace spikecorec::log;

namespace spikecorec {
    struct ScaledReservoirResult {
        ScaleResult weight_scale_result;
        f32 w_accum;
        f32 w_instant;
    };

    // Transforms a list of event ticks into a per-tick series that is zero everywhere
    // except at the event ticks, where it carries the input's actual magnitude (derived
    // from rate, amplitude and the target's weight). The stream runs from tick 0 through
    // the last event tick inclusive; step_simulation treats every tick past its end as
    // zero rather than requiring every stream to span the whole simulation.
    //
    // `event_ticks` is Vector<s32> to match nml::InputTarget::event_ticks exactly, which is
    // the only thing that ever produces one.
    //
    // A continuous current injector (a pulseGenerator and friends) reports its span as
    // exactly two entries, {start_tick, end_tick}, denoting a window over which current
    // flows continuously rather than two isolated impulses -- so it needs a flag the tick
    // list alone cannot carry. A spikeArray carries real, isolated spike ticks.
    Vector<f64> create_event_stream(
        f64 rate,
        f64 amplitude,
        f64 weight,
        const Vector<s32> &event_ticks,
        bool continuous_current_injection = false
    );

    // One input's per-tick series, already resolved to the neuron it drives.
    struct NeuronInputStream {
        s64 neuron_index = -1;
        Vector<f64> values;
    };

    // One output file, plus everything needed to build its frame without re-deriving the
    // selection each tick. `gathered_indices` are flat cell_state indices for a value
    // recorder and global neuron indices for a spike-event recorder; `frame_values` is the
    // reusable staging buffer the gather writes into, sized once at construction.
    struct RecordingStream {
        unique_ptr<SimulationRecorder> recorder;
        Vector<s64> gathered_indices;
        Vector<f32> frame_values;
        bool gathers_spike_flags = false;
    };

    class SpikeEngine {
    public:
        SharedPointer<EngineLogger> logger;

        // Declared ahead of `weights` deliberately: members initialise in declaration
        // order, and the weight matrix is built out of the parsed model's adjacency.
        nml::NML_ParseResult network_details;

        WeightMatrix weights;

        EngineAllocator allocator;

        // ── model-sized engine buffers ────────────────────────────────────────────
        // Every one of these is an `allocator` sub-range: it aliases the arena's slab and
        // owns nothing, so none of them may ever be passed to deallocate(). The arena
        // releases both slabs when the engine is destroyed.
        //
        // cell_state and cell_parameters are sectioned by cell TYPE -- every neuron of
        // type 0 first, then every neuron of type 1, and so on -- with each neuron
        // occupying state_variable_names.size() / parameter_names.size() floats inside its
        // section. Global neuron indices are never renumbered; cell_state_base,
        // cell_parameter_base and cell_type_index are the scaffolding that resolves a
        // neuron to its slot in one load. This is the layout the generated kernel assumes
        // (see nml/kernel_codegen.h).
        GpuPointer<f32> cell_state;          // [cell_state_element_count]
        GpuPointer<f32> cell_parameters;     // [cell_parameter_element_count]
        GpuPointer<f32> network_inputs;      // [total_neuron_count] — synaptic accumulator
        GpuPointer<s32> cell_state_base;     // [total_neuron_count]
        GpuPointer<s32> cell_parameter_base; // [total_neuron_count]
        GpuPointer<s32> cell_type_index;     // [total_neuron_count]
        GpuPointer<s32> spike_flags;         // [total_neuron_count] — this tick's emissions

        // [total_neuron_count] — tick each neuron last fired. Always allocated: the
        // generated kernel writes it unconditionally on every emission, so it is a kernel
        // output rather than a plasticity-only buffer.
        GpuPointer<s64> last_spiked;

        // [total_neuron_count] — tick each neuron last received input. Allocated only when
        // hebbian learning is enabled, which is the only thing that reads it.
        GpuPointer<s64> last_tick_updated;

        s64 cell_state_element_count = 0;
        s64 cell_parameter_element_count = 0;

        // The compiled generated master kernel. Held through a shared pointer because
        // KernelHandle is opaque outside backend.cpp: a shared pointer type-erases its
        // deleter at construction, where the type is complete, so neither this header nor
        // ~SpikeEngine needs the definition.
        SharedPointer<KernelHandle> tick_kernel;

        // The generated kernel's own argument list. Arguments are bound positionally
        // against this rather than against a hardcoded order, so codegen adding an
        // argument cannot silently misbind the ones already there.
        Vector<String> tick_kernel_argument_names;

        Vector<NeuronInputStream> input_event_streams;

        Vector<RecordingConfig> recording_profiles;
        Vector<RecordingStream> recording_streams;

        // GpuPointer<f32> membrane_potentials; // [neuron_count]
        // GpuPointer<s32> active_neuron_indices; // [neuron_count] — current active set (compacted)
        // GpuPointer<s32> next_active_neuron_indices; // [neuron_count] — active set for next tick
        // GpuPointer<s32> active_neuron_count; // [1]
        // GpuPointer<s32> next_active_neuron_count; // [1]
        // GpuPointer<s32> active_generation; // [neuron_count] — generation tag, -1 = inactive
        // GpuPointer<s32> input_neuron_indices; // set via set_input_neurons()

        // Persistent step_simulation scratch buffers, sized to neuron_count
        // (the existing hard upper bound already relied on by
        // input_neuron_indices / active_neuron_indices)
        // and overwritten in place each tick instead of allocate/copy/free per
        // tick (SC-20).
        // GpuPointer<f32> input_staging; // [neuron_count] — staged input_values
        // [neuron_count] — staged override_input_neurons
        // GpuPointer<s64> override_staging;

        // constexpr (not plain const) so it is implicitly inline in C++17
        // it is ODR-used as a default-argument value in the pybind11 bindings
        // (bindings.cpp), which would otherwise require an out-of-line
        // definition.
        // static constexpr s64 DEFAULT_MAX_LOG_BYTES = 512 * 1024 * 1024;
        // f32 **cell_state_logs = nullptr;

        s64 lifetime = 0;
        s64 total_neuron_count = 0;
        s64 input_neuron_count = 0;

        s32 thread_count_per_block = 0;
        s32 block_count = 0;

        // f32 resting_membrane_potential;
        // f32 decay_rate;
        // f32 learning_rate;
        // s32 spike_period;
        // f32 spike_threshold;

        bool hebbian_learning_enabled = false;
        bool use_constant_weight = false;
        bool alive = false;
        // Off on the NeuroML path: every neuron steps every tick.
        bool active_set_optimization_enabled = false;
        u64 simulation_seed = 0;

        SpikeEngine(const SpikeEngine &) = delete;

        SpikeEngine &operator=(const SpikeEngine &) = delete;

        SpikeEngine(SpikeEngine &&) = default;

        SpikeEngine &operator=(SpikeEngine &&) = default;

        SpikeEngine(String &neuroml_input_file, bool enable_hebbian_learning);

        // ── legacy, pending rework ────────────────────────────────────────────────
        // Every declaration below this line predates the NeuroML path and depends on the
        // hardcoded-LIF members commented out above. Left in place rather than deleted so
        // the owner can adapt them; see src/core/engine.cpp for the matching definitions.
        //
        // SpikeEngine(
        //     vector<vector<s32> > *network,
        //     const vector<s64> &shape,
        //     s64 rank = 1,
        //     f32 resting_mp = 0.1f,
        //     f32 decay_rate = 0.01f,
        //     f32 learning_rate = 0.00222f,
        //     bool plasticity_enabled = true,
        //     bool active_set_optimization_enabled = true
        // );
        //
        // SpikeEngine();

        ~SpikeEngine();

        // void setup_lifetime(
        //     int lifetime,
        //     bool allocate_logs,
        //     s64 max_log_bytes = DEFAULT_MAX_LOG_BYTES
        // );

        // void set_input_neurons(const vector<s32> &input_neuron_list);

        // void reset_state(s64 last_spiked_value = 0, s32 active_gen_value = -1);

        // Advances every neuron by exactly one dt: delivers this tick's external stimulus,
        // runs the generated master kernel, and records.
        void step_simulation(s64 tick);

        // Binds `kernel`'s arguments positionally against `argument_names` and launches it
        // over the whole neuron population. Throws naming the argument when the generated
        // kernel asks for one this engine does not own -- which is what reports that
        // codegen and the engine have drifted apart, rather than binding something
        // arbitrary in its place.
        void dispatch_master_kernel(
            const KernelHandle &kernel,
            const Vector<String> &argument_names,
            s64 tick
        );

        // One frame per recording stream, gathered out of the buffers the kernel just
        // wrote.
        void record_tick_frames();

        // void start_static_record(
        //     const vector<vector<f32>> &input_spikes,
        //     s64 lifetime,
        //     const string &filename,
        //     bool record_membrane = true,
        //     s64 record_stride = 1,
        //     optional<string> compression = string("auto"),
        //     optional<int> compression_level = nullopt,
        //     bool full_decay = true,
        //     bool compression_async = false,
        //     usize compression_queue_max = 8,
        //     usize compression_chunk_bytes = 4 * 1024 * 1024
        // );

        // [[nodiscard]]
        // pair<f32, f32> estimate_bifurcation_weight(s32 input_period = 1) const;

        // bool plasticity_enabled();
        // void enable_plasticity(f32 _learning_rate = 0.00222f);
        // void disable_plasticity();

        // void get_reservoir_features_vector(
        //     s64 tick,
        //     f32 spike_tau,
        //     f32 voltage_scale,
        //     GpuPointer<f32> output_buffer
        // );

        // first three args are the return values
        // void scale_uniform_weights_near_bifurcation(
        //     f32 *target_,
        //     f32 *w_accum_,
        //     f32 *w_instant_,
        //     s32 input_period = 1,
        //     f32 scale = 1.2,
        //     bool freeze_learning = false,
        //     const bool *use_constant_weight_ = nullptr
        // );

        // ScaledReservoirResult scale_randomized_weights_near_bifurcation(
        //     s32 input_period = 1,
        //     f32 scale = 1.2,
        //     bool freeze_learning = false
        // );

        void shutdown();
    };
} // namespace spikecorec
