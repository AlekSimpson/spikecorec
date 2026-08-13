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
    //
    // THE WINDOW IS CLOSED AT BOTH ENDS: the fill below runs `tick <= end_tick`, so
    // end_tick is the last tick current is delivered on and the window is end_tick -
    // start_tick + 1 ticks long. Whatever computes the pair owes this an INCLUSIVE last
    // tick -- nml.cpp's build_input_profile converts LEMS's half-open
    // `t .lt. delay + duration` by subtracting one there, and the two must not drift
    // apart: an exclusive end here would inject one extra tick of charge, which is 10% of
    // a 1ms pulse at a 0.1ms step.
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

    // Every NetworkEdge the model declares between one ordered pair, collapsed into the one
    // slot the k^2-tree adjacency holds for that pair.
    struct AggregatedNetworkEdge {
        s32 source_node = -1;
        s32 target_node = -1;
        f32 summed_weight = 0.0f;
        s32 delay_tick_count = 1;

        // Which synapse prototype the collapsed edges deliver through, -1 when they name
        // none. Like the delay, it cannot be collapsed: the slot runs one lowered synapse
        // program and carries one set of per-edge state.
        s64 synapse_prototype_index = -1;
    };

    // Collapses parallel edges -- two projections between one cell pair, which is an
    // ordinary NeuroML shape -- onto the single adjacency slot that pair has.
    //
    // Weights SUM: two synapses onto one cell both deliver, and the adjacency cannot
    // represent them separately, so their total is the only faithful collapse.
    //
    // Delays cannot be summed or averaged into anything meaningful, and two parallel edges
    // with genuinely different conduction delays are not representable in one slot at all.
    // Rather than pick one silently, a conflict THROWS naming the pair and both delays: a
    // halved or doubled conduction delay changes when every downstream spike lands, and
    // there is no reading of the model under which either answer is right. Edges that agree
    // (the common case -- parallel projections normally share a delay) collapse silently,
    // since nothing is lost. `default_delay_tick_count` is what an edge declaring no delay
    // of its own resolves to, so "declares none" and "declares the default" do not read as a
    // conflict.
    //
    // The SYNAPSE is the same story for the same reason: the slot runs one lowered synapse
    // program over one set of per-edge state, so two parallel edges through different
    // synapses (an AMPA and an NMDA between one cell pair) cannot both be represented, and
    // summing their weights into whichever program came first would run one synapse's
    // dynamics on the other's coupling. That throws too.
    //
    // Returned in first-appearance order, so the wiring it drives is deterministic.
    Vector<AggregatedNetworkEdge> aggregate_network_edges(
        const nml::NML_ParseResult &parse_result,
        s32 default_delay_tick_count,
        EngineLogger &engine_logger
    );

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

        // The compensated-accumulation residual for every cell_state slot: laid out exactly
        // parallel to cell_state, same size and same bases, so one index reaches both.
        //
        // Forward Euler is a running sum, and in f32 each step loses the low bits of an
        // increment that is small beside the total. Over a refractory period of fifty 0.1 ms
        // ticks that costs a whole tick -- the `.geq.` comparison lands one tick late, every
        // time. Metal has no double precision, so the sum is compensated instead: each slot
        // carries the part of its last increment that did not fit, and the next step folds it
        // back in. See "Compensated accumulation" in src/nml/kernel_codegen.cpp.
        GpuPointer<f32> cell_state_residual; // [cell_state_element_count]

        GpuPointer<f32> cell_parameters;     // [cell_parameter_element_count]
        GpuPointer<s32> cell_state_base;     // [total_neuron_count]
        GpuPointer<s32> cell_parameter_base; // [total_neuron_count]
        GpuPointer<s32> cell_type_index;     // [total_neuron_count]
        GpuPointer<s32> spike_flags;         // [total_neuron_count] — this tick's emissions

        // The synaptic delay ring:
        // [network_input_ring_depth * total_neuron_count], one flat allocation indexed as
        // [row][neuron]. Propagation, generated into the cell device functions, adds an
        // edge's delivered current into row (tick + that edge's delay) %
        // network_input_ring_depth; a cell reads row tick % network_input_ring_depth of its
        // own column with a plain load. External stimulus is added into the current tick's
        // row, arriving with no delay.
        //
        // One value per (row, neuron) and no synapse dimension: a spike delivers the whole of
        // the charge its synapse's response to that event carries, in one tick, into one slot
        // -- as the current that delivers it over that tick, so charge Q arrives as Q / dt
        // (see kernel_codegen.h). An edge naming no synapse scatters its raw weight into that
        // same slot.
        //
        // The two are dimensionally incompatible -- amps against a dimensionless weight -- and
        // the slot sums them, so no target may receive both kinds. Codegen refuses a model
        // where one does (see reject_mixed_delivery_targets); onto different targets they are
        // both legal.
        //
        // A row is emptied as a WHOLE ROW by the ring clear kernel dispatched behind the tick
        // kernel, at the end of the tick that read it -- clearing the input this tick consumed
        // while leaving every other row, which holds arrivals not yet due, untouched. Not per
        // slot as each cell reads: a cell type that never reduces over its synapses would then
        // never clear its neurons' slots, and they would accumulate every arrival for the
        // whole run.
        GpuPointer<f32> network_inputs;

        // [EDGE_ATTRIBUTE_PLANE_COUNT * weights.node_count * weights.max_neighbor_count] —
        // three int planes over the per-edge slot convention WeightMatrix addresses an edge by
        // (row-major by source neuron; within a row, the neighbour's position in k^2-tree
        // traversal order). Consolidated into one allocation because they reach the generated
        // kernel as ONE argument: three separate arrays would cost three of Metal's 31 buffer
        // slots where one buffer with baked plane offsets costs one.
        //
        //   plane 0  each edge's delay in whole ticks. Flattened once at construction out of
        //            WeightMatrix::get_edge_delay_ticks, which is where the rounding, the >= 1
        //            floor and the constant-delay fallback are decided; the kernel then needs
        //            one load per edge and no delay logic of its own.
        //   plane 1  which lowered synapse program the edge runs -- its position in
        //            nml::resolve_synapse_programs -- or -1 when the edge's projection names
        //            no synapse, in which case its raw weight IS the delivered current.
        //   plane 2  the tick each edge's synapse state was last advanced through, seeded to
        //            EDGE_NEVER_ADVANCED_TICK. Read and written only under lazy synapse
        //            updates; the eager path advances every edge every tick and never
        //            consults it.
        GpuPointer<s32> edge_attributes;

        // [5] — the k^2-tree shape scalars the generated propagation walk needs, in the order
        // kernel_codegen.h documents: branching_factor, superblock_size_words,
        // padded_node_count, tree_height, internal_bit_count. One buffer rather than five
        // scalar arguments, for the same argument-table reason edge_attributes is one buffer.
        GpuPointer<s32> k2tree_shape;

        // [total_neuron_count] — tick each neuron last fired, or NEURON_NEVER_SPIKED_TICK
        // for one that has not fired at all. Always allocated: the generated kernel writes
        // it unconditionally on every emission, so it is a kernel output rather than a
        // plasticity-only buffer.
        GpuPointer<s64> last_spiked;

        // [total_neuron_count] — tick each neuron last received input. Allocated only when
        // hebbian learning is enabled, which is the only thing that reads it.
        GpuPointer<s64> last_tick_updated;

        // How many int planes edge_attributes packs, and which is which. Named rather than
        // written as bare 0/1/2 at each of the places that index them: a plane mix-up reads
        // as a wrong delay or a wrong synapse and never as a crash. The generated source
        // carries the identical constants (kernel_codegen.cpp's
        // emit_edge_attribute_plane_definitions), so the two must not drift.
        static constexpr s64 EDGE_ATTRIBUTE_DELAY_PLANE = 0;
        static constexpr s64 EDGE_ATTRIBUTE_PROGRAM_PLANE = 1;
        static constexpr s64 EDGE_ATTRIBUTE_UPDATE_TICK_PLANE = 2;
        static constexpr s64 EDGE_ATTRIBUTE_PLANE_COUNT = 3;

        // What edge_attributes' update-tick plane holds for an edge whose synapse state has
        // never been advanced. -1, not 0: an edge whose source spikes at tick 0 must still
        // take the one step the eager path would have taken at tick 0, and a seed of 0 would
        // have it take none -- which is a lazy/eager disagreement from the very first spike.
        static constexpr s32 EDGE_NEVER_ADVANCED_TICK = -1;

        // What an edge_attributes program-plane entry holds when the edge's projection names
        // no synapse: nothing shapes its weight, so its raw weight is the delivered current.
        static constexpr s32 EDGE_WITHOUT_SYNAPSE_PROGRAM = -1;

        // What last_spiked holds for a neuron that has never fired. NEGATIVE, and that is the
        // whole point: tick 0 is a real tick a neuron can fire on, so a zero fill makes "never
        // fired" and "fired on tick 0" the same value and every "has this network fired at
        // all" test reads true on a completely dead network. The generated kernel only ever
        // WRITES last_spiked -- it stores the current tick on an emission, which is never
        // negative -- so the sentinel survives untouched until the neuron's first spike, and
        // `last_spiked < 0` is the never-fired test for every consumer.
        static constexpr s64 NEURON_NEVER_SPIKED_TICK = -1;

        s64 cell_state_element_count = 0;
        s64 cell_parameter_element_count = 0;

        // How many per-edge variable planes the model's synapse state occupies, and therefore
        // how many matrices weights carries in its per-edge variable family. Zero for a model
        // with no wired synapse.
        s64 per_edge_synapse_variable_count = 0;

        // How many tick rows the network_inputs ring holds. Strictly greater than the
        // model's largest per-edge delay, so an arrival scheduled this tick can never land
        // in the row being read this tick, and every row is read (and then cleared) before
        // the ring wraps back onto it.
        s32 network_input_ring_depth = 1;

        // The compiled generated master kernel. Held through a shared pointer so its
        // release_kernel call travels with it: the deleter is installed at compile time and
        // runs on the last reference, wherever that is dropped.
        SharedPointer<KernelHandle> tick_kernel;

        // The generated kernel's own argument list. Arguments are bound positionally
        // against this rather than against a hardcoded order, so codegen adding an
        // argument cannot silently misbind the ones already there.
        Vector<String> tick_kernel_argument_names;

        // Zeroes row tick % network_input_ring_depth of network_inputs, dispatched behind the
        // tick kernel in the same command batch every tick. Held and bound exactly like the
        // tick kernel; its argument list is a subset of the same names.
        SharedPointer<KernelHandle> ring_row_clear_kernel;
        Vector<String> ring_row_clear_kernel_argument_names;

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

        // On by default, and the same KIND of knob active_set_optimization_enabled is: an
        // optimization that is exact for the linear dynamics this engine supports, with the
        // unoptimized path kept as the reference to check it against.
        //
        // On, an edge's synapse state is advanced only when its source spikes, catching up
        // across the whole elapsed interval at that point -- O(spikes x fanout) rather than
        // O(edges) per tick. Off, every edge advances one dt every tick. The two apply the
        // same per-tick step the same number of times, so they agree to floating point rather
        // than approximately; see kernel_codegen.h.
        //
        // Baked into the generated kernel at construction, so changing it afterwards would
        // have no effect and it is not offered as a runtime setter.
        bool use_lazy_synapse_updates = true;

        u64 simulation_seed = 0;

        SpikeEngine(const SpikeEngine &) = delete;

        SpikeEngine &operator=(const SpikeEngine &) = delete;

        // Deleted, not defaulted. A defaulted move leaves the source with `alive` still
        // true and its `logger` null, so the moved-from engine's destructor dereferences a
        // null shared pointer and then runs the whole shutdown body -- a segfault at the
        // end of the scope the move happened in. The defaulted move ASSIGNMENT is worse
        // still: it move-assigns every GpuPointer member, and GpuPointer's move assignment
        // asserts its destination is null, which no live engine's ever is.
        //
        // Deleted rather than hand-written because nothing moves a SpikeEngine: it owns a
        // GPU arena, a compiled pipeline and open recorders, and is constructed in place
        // wherever it is used. Anything needing to relocate one should hold a
        // unique_ptr<SpikeEngine> and move that instead.
        SpikeEngine(SpikeEngine &&) = delete;

        SpikeEngine &operator=(SpikeEngine &&) = delete;

        // `use_lazy_synapse_updates` selects how per-edge synapse state is advanced; see the
        // field of that name. Defaulted so an existing caller keeps the (exact, cheaper) lazy
        // behaviour without a source change, and so the eager reference path is one argument
        // away for anything checking the two against each other.
        SpikeEngine(String &neuroml_input_file, bool enable_hebbian_learning,
                    bool use_lazy_synapse_updates = true);

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
        // runs the generated master kernel and the ring clear behind it, and records.
        void step_simulation(s64 tick);

        // Binds `kernel`'s arguments positionally against `argument_names` and launches it
        // over the whole neuron population. Throws naming the argument when the generated
        // kernel asks for one this engine does not own -- which is what reports that
        // codegen and the engine have drifted apart, rather than binding something
        // arbitrary in its place.
        //
        // With a non-null `batch` the launch is encoded into that batch's command buffer
        // instead of being committed and waited on by itself, which is how the tick kernel
        // and the ring clear behind it reach the GPU as one ordered pair.
        void dispatch_master_kernel(
            const KernelHandle &kernel,
            const Vector<String> &argument_names,
            s64 tick,
            MetalCommandBatch *batch = nullptr
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
