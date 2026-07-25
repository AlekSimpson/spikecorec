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
#include "spikecorec/nml/allocator.h"
#include "spikecorec/nml/ir.h"
#include "spikecorec/nml/model_specification.h"

using namespace std;
using namespace spikecorec::log;

namespace spikecorec {
    struct ScaledReservoirResult {
        ScaleResult weight_scale_result;
        f32 w_accum;
        f32 w_instant;
    };

    class SpikeEngine {
    public:
        SharedPointer<EngineLogger> logger;

        WeightMatrix weights;

        // NML-mode cell state (Stage 1 of folding nml::AssembledModel into SpikeEngine -- only
        // populated by the ModelSpecification constructor below; left default-empty for the two
        // hardcoded-LIF constructors). Public (unlike the rest of the new nml_-prefixed state below)
        // so a caller can read/seed per-neuron state the same way examples/nml_pipeline_support.h's
        // read_membrane_potential/seed_membrane_potentials_from_resting_parameter already do against
        // a bare nml::ModelAllocation.
        nml::ModelAllocation nml_allocation_;

        GpuPointer<f32> network_inputs; // [neuron_count] — external input accumulator
        GpuPointer<f32> membrane_potentials; // [neuron_count]
        GpuPointer<s64> last_spiked; // [neuron_count] — tick each neuron last fired
        GpuPointer<s64> last_tick_updated; // [neuron_count] — tick each neuron last received input
        GpuPointer<s32> active_neuron_indices; // [neuron_count] — current active set (compacted)
        GpuPointer<s32> next_active_neuron_indices; // [neuron_count] — active set for next tick
        GpuPointer<s32> active_neuron_count; // [1]
        GpuPointer<s32> next_active_neuron_count; // [1]
        GpuPointer<s32> active_generation; // [neuron_count] — generation tag, -1 = inactive
        GpuPointer<s32> input_neuron_indices; // set via set_input_neurons()

        // Persistent step_simulation scratch buffers, sized to neuron_count (the existing
        // hard upper bound already relied on by input_neuron_indices / active_neuron_indices)
        // and overwritten in place each tick instead of allocate/copy/free per tick (SC-20).
        GpuPointer<f32> input_staging; // [neuron_count] — staged input_values
        GpuPointer<s64> override_staging; // [neuron_count] — staged override_input_neurons

        static constexpr s64 DEFAULT_MAX_LOG_BYTES = 512 * 1024 * 1024;
        f32 **cell_state_logs = nullptr;

        s64 lifetime = 0;
        s64 neuron_count;
        s64 input_neuron_count;

        s32 thread_count_per_block;
        s32 block_count;

        f32 resting_membrane_potential;
        f32 decay_rate;
        f32 learning_rate;      // minus-side (depression) STDP rate — 0 disables the depression path
        f32 learning_rate_plus; // plus-side (potentiation) STDP rate — 0 disables the potentiation
                                // path; the two are independent so a run can enable either or both
        s32 spike_period;
        f32 spike_threshold;

        bool use_constant_weight = false;
        bool alive = false;
        bool active_set_optimization_enabled = true;

        SpikeEngine(const SpikeEngine &) = delete;

        SpikeEngine &operator=(const SpikeEngine &) = delete;

        // Hand-written (not defaulted): a defaulted move would bitwise-copy nml_drain_kernel_/
        // nml_propagate_kernel_ (KernelHandle has no move semantics of its own -- see backend.h)
        // into the destination WITHOUT clearing them on the moved-from source, and would leave the
        // moved-from source's alive/nml_mode_enabled_ flags unchanged (both true for a live NML-mode
        // engine) -- so its destructor would later call shutdown() again and release/delete the same
        // kernels/buffers the destination still owns. Mirrors WeightMatrix's own hand-written move
        // (see weight_matrix.h) for exactly the same class of hazard. Defined in engine.cpp.
        SpikeEngine(SpikeEngine &&other) noexcept;

        SpikeEngine &operator=(SpikeEngine &&other) noexcept;

        SpikeEngine(
            vector<vector<s32> > *network,
            const vector<s64> &shape,
            s64 rank = 1,
            f32 resting_mp = 0.1f,
            f32 decay_rate = 0.01f,
            f32 learning_rate = 0.00222f,
            bool plasticity_enabled = true, 
            bool active_set_optimization_enabled = true
        );

        SpikeEngine();

        // ── NML-model constructor (Stage 1 of folding nml::AssembledModel into SpikeEngine -- see
        // the "REFACTOR" comments directly above class AssembledModel in
        // include/spikecorec/nml/master_kernel.h). Builds this engine's per-neuron/per-population
        // state directly from a lowered NML ModelSpecification + its per-ComponentType IR programs,
        // instead of the hardcoded LIF cell the two constructors above build. `model` is taken by
        // mutable reference to match nml::allocate_model's own signature (it configures
        // model.adjacency's per-edge variable count in place). `dt_seconds` is this model's own fixed
        // tick duration (the same value every step_tick call's own `dt` argument uses) -- needed here
        // to convert each ConnectionEntry::delay (SI seconds, already unit-resolved by resolve.cpp's
        // own unit_value_to_si) into the whole-tick counts `weights`' delay fields store (mirrors
        // delay_ring.cpp's own, separate, untouched seconds->ticks conversion -- see engine.cpp's
        // constructor body). Defaults to 1e-4f, the established Phase-1 tick duration (examples/
        // nml_pipeline_support.h's own ExampleOptions::dt_seconds default), so every existing caller
        // that only ever runs at that convention keeps compiling and behaving unchanged.
        SpikeEngine(nml::ModelSpecification &model, const Vector<nml::IrProgram> &type_library_ir_programs,
                    f32 dt_seconds = 1e-4f);

        ~SpikeEngine();

        void setup_lifetime(int lifetime, bool allocate_logs, s64 max_log_bytes = DEFAULT_MAX_LOG_BYTES);

        void set_input_neurons(const vector<s32> &input_neuron_list);

        void reset_state(s64 last_spiked_value = 0, s32 active_gen_value = -1);

        void step_simulation(
                s64 tick,
                f32 dt,
                optional<UnorderedMap<s64, f32>> override_network_inputs_this_tick = nullopt)

        // void step_simulation(
        //     const vector<f32> &input_values,
        //     s64 tick,
        //     const vector<s64> &override_input_neurons = {},
        //     bool decay_all_neurons = false);

        // Runs one NML-model tick: stages 2-5 (one dispatch per population with a per-neuron
        // kernel, over its own full neuron range) followed by the fixed deliver-drain and
        // k^2-tree propagate/scatter + active-set-enqueue stages -- mirrors
        // nml::AssembledModel::step_tick's own non-delay-ring branch (see master_kernel.cpp). Only
        // valid on a SpikeEngine constructed via the ModelSpecification constructor above; throws
        // otherwise.
        // void step_tick(f32 dt, s64 tick, s64 next_tick);

        // Sets a single neuron's emit-port flag directly, ahead of calling step_tick for that same
        // tick -- a real, minimal way to host-drive a synthetic/external spike into the fixed
        // propagate stage (stage 6), which reads+clears every nml_emit_port_flags_ entry
        // unconditionally each tick regardless of who set it true. A flag forced true here is
        // therefore dispatched through the exact same k^2-tree scatter/last_spiked/active-set/
        // delay-ring machinery as a spike a population's own `.tick` kernel would have set. Only
        // valid on a SpikeEngine constructed via the ModelSpecification constructor above (throws
        // otherwise), and only for a `port_name` that is one of this model's own known emit ports
        // (nml_emit_port_names_, throws otherwise) and an in-range `neuron_index` (throws
        // otherwise). Deliberately scoped to this single setter rather than exposing
        // nml_emit_port_flags_ itself.
        void force_emit(const String &port_name, s64 neuron_index);

        [[nodiscard]] pair<f32, f32> estimate_bifurcation_weight(s32 input_period = 1) const;

        bool plasticity_enabled();
        // Enables STDP with a minus-side (depression) rate and an optional plus-side (potentiation)
        // rate. The plus-side default is 0.0f so every existing single-argument caller keeps exactly
        // its prior pure-depression behavior (bidirectional potentiation is opt-in). Unchanged
        // no-op-if-already-enabled semantics (guards on `learning_rate > 0.0f`) — see engine.cpp.
        void enable_plasticity(f32 _learning_rate = 0.00222f, f32 _learning_rate_plus = 0.0f);
        void disable_plasticity();

        void get_reservoir_features_vector(s64 tick, f32 spike_tau, f32 voltage_scale, GpuPointer<f32> output_buffer);

        // first three args are the return values
        void scale_uniform_weights_near_bifurcation(f32 *target_,
                                                   f32 *w_accum_,
                                                   f32 *w_instant_,
                                                   s32 input_period = 1,
                                                   f32 scale = 1.2,
                                                   bool freeze_learning = false,
                                                   const bool *use_constant_weight_ = nullptr);

        ScaledReservoirResult scale_randomized_weights_near_bifurcation(s32 input_period = 1, f32 scale = 1.2, bool freeze_learning = false);

        void shutdown();

    private:
        // ── NML-mode runtime state (Stage 1 of folding nml::AssembledModel into SpikeEngine --
        // see the ModelSpecification constructor's own doc comment above). Mirrors what
        // nml::AssembledModel + nml::ModelRuntimeBuffers together held (include/spikecorec/nml/
        // master_kernel.h), flattened directly into SpikeEngine and OWNING what it needs (rather
        // than pointing at externally-supplied buffers the way ModelRuntimeBuffers did). Only used
        // by the ModelSpecification constructor and step_tick() above -- the two hardcoded-LIF
        // constructors never touch any of this, and it stays default/empty for them. Every member
        // here is prefixed `nml_` to keep it visually separate from the original hardcoded-engine
        // fields above, since this is temporary until the fold (CLAUDE.md's NML codegen epic) is
        // fully complete.

        // One resolved dispatch argument for a population's per-neuron `_tick` kernel -- either a
        // per-call scalar substituted at dispatch time (Dt/Tick/PopulationSize), a fixed pointer
        // computed once at construction (every parameter kind that never moves for the engine's
        // lifetime -- emit-port/state/accum/param:dyn/regime/expose/rng_state buffers), or
        // NetworkInputsRingOffset (the delay-ring fold: `network_inputs` is now ring-shaped
        // `[nml_ring_slot_count_ * neuron_count]`, so which slot a population's own `_tick` kernel
        // reads from rotates every tick -- `tick % nml_ring_slot_count_` -- and so cannot be baked
        // as a single fixed pointer the way it could when network_inputs was always flat; only
        // `neuron_index_begin` is cached here, and step_tick recomputes the real pointer each call).
        struct NmlResolvedArgument {
            enum class Kind { Dt, Tick, PopulationSize, FixedPointer, NetworkInputsRingOffset };
            Kind kind = Kind::FixedPointer;
            const void *fixed_pointer = nullptr;
            s32 neuron_index_begin = 0; // only meaningful for Kind::NetworkInputsRingOffset
        };

        // One population's precomputed dispatch: its compiled per-neuron kernel, the neuron count
        // to launch over, and its resolved argument list in parameter order.
        struct NmlPopulationDispatchPlan {
            KernelHandle kernel_handle{};
            s32 population_size = 0;
            Vector<NmlResolvedArgument> arguments;
        };

        // ── delay-ring fold (SpikeEngine-only; see the three REFACTOR comments in delay_ring.h/
        // master_kernel.h/master_kernel.cpp this generalizes) -- network_inputs/
        // next_active_neuron_indices/next_active_neuron_count/active_generation above are all
        // ring-shaped `[nml_ring_slot_count_ * neuron_count]` (next_active_neuron_count is
        // `[nml_ring_slot_count_]`), determined once from `weights`' own delay configuration
        // (constant_delay_ticks/edge_delay_ticks) at NML construction time. `1` is the ordinary
        // "no real per-edge delay beyond the engine's existing implicit one-tick latency" case --
        // every dispatch then always reads/writes slot 0, collapsing to exactly the pre-fold flat
        // behavior. There is only ever ONE compiled drain kernel and ONE compiled propagate kernel
        // (nml_drain_kernel_/nml_propagate_kernel_ below), parameterized by ring_slot_count/
        // current_slot at dispatch time, regardless of this value.
        s64 nml_ring_slot_count_ = 1;

        Vector<NmlPopulationDispatchPlan> nml_population_dispatch_plans_;

        Vector<String> nml_emit_port_names_; // dispatch order for the propagate stage

        KernelHandle nml_drain_kernel_{};
        KernelHandle nml_propagate_kernel_{};

        GpuPointer<u32> nml_rng_state_; // [neuron_count] -- only allocated if some population's
                                         // kernel actually references rand/randn

        // ── Stage 2 of folding nml::AssembledModel into SpikeEngine: real per-edge synapse dispatch
        // (ticket #131) + real STDP (ticket #132), ported from AssembledModel (master_kernel.h/.cpp),
        // extending Stage 1's own precomputed-dispatch-argument-plan pattern to the synapse
        // edge-parallel dispatches. ──────────────────────────────────────────────────────────────────

        Vector<nml::ProjectionEntry> nml_projections_; // copy of model.projections at construction time

        // Minimal per-population info a synapse dispatch needs about its POSTSYNAPTIC population
        // (type_library_index/neuron_index_begin/population_size) -- parallel to model.populations at
        // construction time. nml_population_dispatch_plans_ above is NOT index-parallel to
        // model.populations (Stage 1 only ever kept a plan for a population WITH a per-neuron kernel),
        // so this is stored separately rather than reusing that vector.
        struct NmlPopulationInfo {
            s32 type_library_index = -1;
            s32 neuron_index_begin = 0;
            s32 population_size = 0;
        };
        Vector<NmlPopulationInfo> nml_populations_;

        // Owned copy of the type-library IR programs (parallel to model.type_library), kept ONLY for
        // Stage 2's own lazy, first-step_tick-call topology build (ensure_nml_synapse_dispatch_
        // topology_built below) -- unlike Stage 1's cell-tick argument resolution (fully resolved at
        // construction, see resolve_nml_cell_tick_argument), a synapse dispatch's own argument
        // resolution genuinely cannot happen until this->weights's real k^2-tree is available (only
        // true once the engine is already constructed), so the raw IR that resolution needs must still
        // be around at that later point.
        Vector<nml::IrProgram> nml_type_library_ir_programs_;

        // One compiled onevent/deliver function (`<Type>_deliver_<port>`, ticket #131) -- ported as-is
        // from nml::AssembledModel::DeliverFunctionRuntimeInfo.
        struct DeliverFunctionRuntimeInfo {
            KernelHandle handle{};
            Vector<String> parameter_names_in_order;
        };

        // One entry per DISTINCT synapse_type_library_index actually used by some projection --
        // ported as-is from nml::AssembledModel::SynapseTypeRuntimeInfo (see master_kernel.h).
        struct SynapseTypeRuntimeInfo {
            KernelHandle integrate_edges_handle{};
            Vector<String> integrate_edges_parameter_names_in_order;
            Vector<DeliverFunctionRuntimeInfo> deliver_functions; // in lower_synapse_ir_program_to_gpu_source's own order

            bool matrix_indices_registered = false;
            UnorderedMap<String, s64> matrix_index_by_peredge_name; // populated lazily, see
                                                                     // ensure_nml_synapse_dispatch_topology_built
        };
        UnorderedMap<s32, SynapseTypeRuntimeInfo> nml_synapse_types_by_type_library_index_;

        // One resolved synapse edge-parallel dispatch argument -- Stage 2's own extension of
        // NmlResolvedArgument above to a `<Type>_integrate_edges`/`<Type>_deliver_<port>` function's
        // own parameter list (append_synapse_edge_argument's parameter-kind coverage,
        // master_kernel.cpp). RankFloat4Stride/MaxNeighborCount read live off `this->weights` at
        // dispatch time (like Dt/Tick above), rather than baking a value, since they are themselves
        // already O(1) member reads. EventCount is filled from whichever value applies to the dispatch
        // being built (topology.edge_count for `_integrate_edges`; this tick's own filtered
        // delivered_event_count for a `_deliver_<port>` call) -- both dispatches share this one Kind
        // rather than needing separate ones.
        struct NmlResolvedSynapseArgument {
            enum class Kind { Dt, Tick, EventCount, RankFloat4Stride, MaxNeighborCount, FixedPointer };
            Kind kind = Kind::FixedPointer;
            const void *fixed_pointer = nullptr;
        };

        // One entry per ProjectionEntry (parallel to nml_projections_), built LAZILY on the first
        // step_tick call that has any projections (needs this->weights's own real k^2-tree/
        // max_neighbor_count, only available once the engine is already constructed) -- ported as-is
        // from nml::AssembledModel::ProjectionEdgeTopology, plus this stage's own precomputed resolved
        // argument lists (integrate_edges_arguments/deliver_arguments), populated at the same lazy
        // point the edge arrays themselves are (see ensure_nml_synapse_dispatch_topology_built).
        struct ProjectionEdgeTopology {
            s64 edge_count = 0;
            GpuPointer<s32> source_nodes;  // [edge_count] -- global presynaptic neuron index (Sk row)
            GpuPointer<s32> target_nodes;  // [edge_count] -- global postsynaptic neuron index
            GpuPointer<s32> forward_slots; // [edge_count] -- source's own forward k^2-tree row slot (Sk column)

            // Reused every tick, refilled (memcpy) with THIS tick's fired-source subset of the three
            // arrays above -- see dispatch_nml_synapse_delivery_events.
            GpuPointer<s32> delivery_scratch_source_nodes;
            GpuPointer<s32> delivery_scratch_target_nodes;
            GpuPointer<s32> delivery_scratch_edge_slots;

            // Parallel to this projection's synapse type's integrate_edges_parameter_names_in_order /
            // deliver_functions respectively -- precomputed once here, at topology-build time (the
            // earliest point a synapse dispatch argument CAN be resolved, see this struct's own doc
            // comment above), instead of re-walked every tick.
            Vector<NmlResolvedSynapseArgument> integrate_edges_arguments;
            Vector<Vector<NmlResolvedSynapseArgument>> deliver_arguments;
        };
        Vector<ProjectionEdgeTopology> nml_projection_edge_topology_; // parallel to nml_projections_
        bool nml_synapse_dispatch_topology_built_ = false;

        // Resolves one population `_tick` kernel parameter name to an NmlResolvedArgument -- the
        // SAME parameter-kind resolution nml::AssembledModel's own (anonymous-namespace, so not
        // directly reusable) append_cell_tick_argument performs (src/nml/master_kernel.cpp), just
        // run ONCE here at construction instead of every tick. Throws on a `require`/bare-`param`/
        // k^2-tree-walk parameter kind, matching that function's own established scope boundary.
        // Takes no network_inputs pointer (unlike append_cell_tick_argument) -- the delay-ring fold
        // means "network_inputs" now resolves to Kind::NetworkInputsRingOffset (just
        // neuron_index_begin cached), recomputed against the CURRENT ring slot every tick in
        // step_tick, not baked here.
        NmlResolvedArgument resolve_nml_cell_tick_argument(
            const String &parameter_name, const nml::IrProgram &program, s32 type_library_index,
            s32 neuron_index_begin, s64 cell_state_chunk_base_offset, s32 population_size,
            u32 *rng_state_base, const UnorderedMap<String, GpuPointer<bool>> &emit_port_flags) const;

        // Builds nml_projection_edge_topology_ (from this->weights's real k^2-tree) and registers
        // every used synapse type's peredge variables as real WeightMatrix matrix_index(es), then
        // precomputes every projection's own resolved dispatch-argument list for its synapse type's
        // `_integrate_edges`/`_deliver_<port>` functions -- exactly once, idempotent thereafter. Only
        // callable once this->weights's own k^2-tree is real (i.e. from step_tick, not construction --
        // see master_kernel.h's own doc comment on this same ordering constraint). Throws if a
        // projection's ConnectionEntry is not a real edge in this->weights's k^2-tree (a mismatched
        // adjacency/model.projections pairing).
        void ensure_nml_synapse_dispatch_topology_built();

        // Dispatches every projection's `<Type>_integrate_edges` function over its WHOLE, static edge
        // list (unfiltered), reading+decaying Sk and adding each edge's finished current into
        // network_inputs[target_node] (already drained earlier this same tick).
        void dispatch_nml_synapse_integrate_edges(f32 dt, s64 tick);

        // Filters every projection's static edge list down to this tick's fired sources (reads, but
        // does not clear, nml_emit_port_flags_) and dispatches that projection's synapse type's own
        // `<Type>_deliver_<port>` function(s) against the filtered batch.
        void dispatch_nml_synapse_delivery_events(f32 dt, s64 tick);

        // Resolves one synapse edge-parallel function (`_integrate_edges` or `_deliver_<port>`)
        // parameter name to an NmlResolvedSynapseArgument -- the SAME parameter-kind resolution
        // nml::AssembledModel's own (anonymous-namespace, so not directly reusable)
        // append_synapse_edge_argument performs (src/nml/master_kernel.cpp), just run ONCE per
        // projection at topology-build time instead of every tick.
        NmlResolvedSynapseArgument resolve_nml_synapse_edge_argument(
            const String &parameter_name, const nml::IrProgram &synapse_program,
            const UnorderedMap<String, s64> &matrix_index_by_peredge_name,
            const nml::IrProgram &postsynaptic_cell_program, s32 postsynaptic_population_size,
            s32 postsynaptic_neuron_index_begin, s64 postsynaptic_chunk_base_offset,
            const s32 *source_node_indices_pointer, const s32 *target_node_indices_pointer,
            const s32 *edge_slot_indices_pointer) const;

        // Stage 7 (arch §2): for every neuron whose last_spiked == tick (fired THIS tick -- the fixed
        // propagate dispatch(es) above have already set this for every emit port by the time this
        // runs), applies real bidirectional STDP via WeightMatrix::update() (ticket #132; see
        // master_kernel.h's own "ticket #132" doc comment for the full derivation), the host-side
        // mirror of the legacy GPU `step`/`step_no_active_optimization` kernels' own two-sided rule:
        //   - DEPRESSION (anti-causal): walk the firing neuron's forward k^2-tree neighbors
        //     (get_neighbors); for each already-fired child, nudge the source->child edge DOWN by
        //     -learning_rate * pow(tick_delta, -3).
        //   - POTENTIATION (causal): walk the firing neuron's PREDECESSORS (get_predecessors); for each
        //     predecessor that fired strictly before this tick, nudge the predecessor->source edge UP
        //     by +learning_rate_plus * pow(tick_delta, -3).
        // Reuses SpikeEngine's OWN `learning_rate`/`learning_rate_plus` fields / enable_plasticity() /
        // disable_plasticity() -- each direction is independently gated (depression on learning_rate,
        // potentiation on learning_rate_plus), and the whole method is a no-op when both are 0.
        // "Never fired" is checked as `last_spiked < 0` here (this engine's own -1 seed convention for
        // NML mode, see the ModelSpecification constructor -- unlike AssembledModel's ModelRuntimeBuffers
        // callers, which seed last_spiked to 0 and check against that instead).
        void apply_nml_stdp_plasticity(s64 tick);
    };
} // namespace spikecorec
