#pragma once

#include "spikecorec/core/backend.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/nml/allocator.h"
#include "spikecorec/nml/delay_ring.h"
#include "spikecorec/nml/gpu_source.h"
#include "spikecorec/nml/ir.h"
#include "spikecorec/nml/model_specification.h"

namespace spikecorec::nml {

// ── Master-kernel assembly + compile + cache + dispatch (ticket #6 [C3]; arch §0.4, §4.1, §6;
// IR spec §5) ──────────────────────────────────────────────────────────────────────────────────
//
// Turns every Cell-category type-in-use's IR (ticket #55's per-type GPU source) into the runnable
// step: one compiled kernel per population (dispatched over that population's own neuron range --
// arch §4.1's "cell-type boundary" realized as one dispatch per boundary, rather than a single
// mega-dispatch with a runtime branch, since ticket #55 already emits each type's `_tick` function
// as its own complete, self-contained, independently-compilable kernel -- see gpu_source.h), plus
// two engine-fixed scaffold kernels this ticket adds (deliver-drain and k^2-tree propagate/
// scatter/active-set-enqueue). This mirrors the SAME pattern the engine already uses today:
// SpikeEngine::step_simulation (src/core/engine.cpp) already assembles one tick out of several
// separately-dispatched precompiled kernels (gpu_decay_all_neurons, gpu_add_network_input,
// gpu_merge_input_neurons, gpu_step) rather than one giant fused kernel -- "one master kernel" is
// this ticket's assembled ARTIFACT (one set of jointly-designed, always-compiled-and-cached-
// together kernels), not literally one kernel function.
//
// Composition through `network_inputs` (IR spec §3.5): a population's `_tick` kernel reads
// `network_inputs[neuron]` (whatever the FIXED propagate stage scattered into it on a PRIOR tick);
// the fixed drain stage then zeroes it (now that this tick's per-type kernels have read it); the
// fixed propagate stage scatters THIS tick's freshly-computed spikes into it, to be read at the
// NEXT tick -- preserving the >=1-tick latency ir_spec.md §3.5 requires, with no same-tick
// dependency between a cell's own dynamics and a downstream target's.
//
// ── Scope of this ticket (explicitly NOT solved here) ───────────────────────────────────────────
// - Full synaptic-network wiring (routing a spike through a projection's SYNAPSE ComponentType --
//   its own onevent/deliver handler, ticket #55's `<Type>_deliver_<port>` functions) was NOT wired
//   up by THIS ticket (#6): it needed a "spike-scatter batch construction" subsystem (building the
//   per-tick source/target/edge-slot arrays a deliver function's calling convention expects) that
//   gpu_source.h's own header comment used to flag as not yet built by ANY prior ticket. **That
//   subsystem is now built -- ticket #131, see this header's own "spike-scatter batch-construction
//   subsystem" section below** (`AssembledModel::dispatch_synapse_delivery_events`/
//   `dispatch_synapse_integrate_edges`, `ModelSpecification::projections` driven). This ticket's own
//   fixed propagate stage is UNCHANGED by #131 for a model with no projections (it still reproduces
//   exactly what the CURRENT hardcoded engine's own connectivity does: a spiking neuron's own
//   k^2-tree row, reconstructed via WeightMatrix's shared U/V basis or a constant weight); for a
//   model WITH projections, #131 suppresses this stage's own weight contribution (constant_weight
//   forced to 0 for that dispatch only, `buffers.weights` itself untouched) since real per-edge
//   synapse dispatch now supplies it instead, while this stage still performs `last_spiked`/
//   active-set-enqueue bookkeeping for every spike exactly as before.
// - Per-type parameter resolution (ModelRuntimeBuffers -> a population kernel's actual dispatch
//   arguments) supports the parameter KINDS Phase-1's own GLIF-family cell lowering (cell_lowering.cpp)
//   emits: `dt`/`network_inputs` (reserved), `state`/`accum`/`regime`/`param:dyn`/`expose`-scratch
//   (per-neuron arrays), baked `param` constants (no argument -- already inlined as a literal),
//   `emit_<port>` flags, and the trailing `neuron_count` -- plus, as of ticket #65 [F4],
//   `rng_state` (`rand`/`randn`; ModelRuntimeBuffers::rng_state, caller-allocated and seeded). A
//   per-neuron cell kernel that references `require` or the k^2-tree-walk/shared-basis block
//   (`forall`/`loadedge`/`accedge`) still throws a clear std::runtime_error at
//   dispatch-argument-resolution time -- none of Phase-1's GLIF cells or ticket #65's on-device
//   generators reference either (they are synapse-side or Phase-3 constructs, save for
//   `voltageClamp`'s own `require v`, itself a documented, deliberate exclusion -- see
//   inputs_lowering.h), so this is a documented scope boundary, not a silent gap.
// - Active-set-driven SKIP dispatch (the closed-form multi-tick lazy decay the current hardcoded
//   `step` kernel performs for a neuron with no new input) is NOT implemented: every population's
//   kernel runs over its FULL neuron range every tick, matching the existing engine's own
//   `step_no_active_optimization` path (`active_set_optimization_enabled=false`), which is exactly
//   the comparison basis master_kernel_tests.cpp's LIF-equivalence test uses. The fixed propagate
//   stage still performs the active-set ENQUEUE bookkeeping (next_active_neuron_indices/count,
//   active_generation) the ticket body asks for, so a future ticket can wire in the skip-dispatch
//   fast path (CLAUDE.md's own ticket #62 [F1] is the dedicated, later ticket for the
//   active-set-x-nonlinear-dynamics correctness rule this would need) without this ticket's own
//   data going unpopulated in the meantime.
// - STDP/plasticity (stage 7) is not part of the fixed propagate stage (per CLAUDE.md's own ticket
//   mapping, plasticity wiring is ticket #66 [F5]); a per-type program's OWN `.tick.plasticity` (if
//   declared) is still assembled into its `_tick` function by ticket #55 unchanged.
// - Recording (stage 8, ticket #59 [E2]) is likewise not wired to SimulationRecorder here; a
//   per-type program's own `.tick.record` (if declared) is still compiled into its `_tick` function.

// ── ticket #132: real STDP support on AssembledModel (plasticity + delays + per-edge synapse
// dynamics in one simulation object) ────────────────────────────────────────────────────────────
//
// Ticket #100 [T1]'s own investigation (see plasticity_wiring.h's header comment) found that
// `apply_stdp_wiring`/`SpikeEngine::enable_plasticity` only ever drives the LEGACY hardcoded
// engine's own always-running `step_apply_hebbian_update` call site (src/metal/kernels.metal
// `step`/`step_no_active_optimization`) -- `AssembledModel` had no plasticity mechanism at all, so
// a real NML/GLIF network run through THIS file's own dispatch could never combine cell dynamics +
// delays + active STDP. This closes that gap.
//
// ── Design decision: (a), reusing the EXISTING mechanism, not a new kernel ──
// `step_apply_hebbian_update`'s own math (a rank-1 proximal Hebbian nudge of U[source_node]/
// V[target_node], gated on `learning_rate != 0` and "postsynaptic neuron has spiked before, but not
// THIS tick") is, for the single-iteration case every real call site uses, ALGEBRAICALLY IDENTICAL
// to `WeightMatrix::update()`'s own already-existing, already-tested `gpu_weight_update` dispatch
// (its `l2_regularization * (anchor - anchor)` term is exactly zero at iteration 1, matching
// step_apply_hebbian_update's own doc comment on kernels.metal eliding that same dead term -- see
// weight_matrix.h/.cpp). So "porting the mechanism onto AssembledModel directly" does not mean
// hand-duplicating new MSL/CUDA propagate-kernel math (a real risk of a subtly wrong second
// implementation of the same formula): it means calling the SAME `WeightMatrix::update()` API the
// engine already exposes, host-side, once per (spiking presynaptic neuron, k^2-tree neighbor) pair,
// with the SAME call-site constants (`learning_rate=0.5f`, `l2_regularization=1.0f`,
// `iterations=1`) and the SAME `decay_delta = -stdp_learning_rate * pow(|tick - child_last_spiked|,
// -3)` gate every real call site already uses. `apply_stdp_plasticity` (master_kernel.cpp) is that
// host loop -- stage 7 (arch §2), running immediately after stage 6's propagate dispatch (both the
// flat and ring-mode branches of step_tick), reading `last_spiked` (already fresh for every neuron
// that fired THIS tick, by construction: propagate has finished for every emit port by the time this
// runs) -- deterministically reproducing "skip a child that has never spiked or spiked this same
// tick" with no race, unlike the original fused kernel's own device-side same-tick ambiguity
// (harmless: a caller cannot observe the difference outside that one racy edge case, and nothing
// here depends on it).
//
// Tradeoff, documented rather than hidden: this loops host-side over (spiking neuron x its k^2-tree
// out-degree) with one `WeightMatrix::update()` GPU dispatch (+ sync) per edge, not one fused,
// GPU-parallel-across-edges kernel launch -- slower per tick than the legacy fused engine for a
// large, densely-spiking network. Acceptable for this ticket's scope (a real integration point, not
// a performance rewrite); a future ticket can fuse this into a real batched propagate-kernel variant
// if profiling ever calls for it.
//
// ── Coordinating with ticket #131's per-edge synapse dispatch (the OTHER Ck/Sk writer) ──
// `WeightMatrix::update()` mutates the SHARED `U_matrix`/`V_matrix` basis rows for `source_node`/
// `target_node` directly (weight_matrix.h's own "single-matrix, unit-coefficient special case" of
// the shared basis) -- rows every OTHER registered matrix's entries are ALSO reconstructed from
// (`Σ U[i,r]·Ck[r]·V[j,r]`, arch §4.3). Nudging U/V for the weight's sake therefore also perturbs
// any peredge synapse state (e.g. a conductance `g`) sharing that basis -- real cross-talk ticket
// #54's own periodic `refit()` explicitly re-fits every OTHER matrix's `Ck` to compensate for (see
// weight_matrix.h's own "expected, not a bug" note), but this ticket's own per-tick incremental
// nudge does NOT run any compensating refit. Rather than silently letting STDP corrupt a per-edge
// synapse's own reconstructed state, `enable_plasticity` below throws if this AssembledModel has any
// real per-edge synapse dispatch active (`projections_` non-empty -- i.e. `model.projections` was
// non-empty AND this instance was NOT constructed with `enable_delay_ring=true`, since ring mode
// already forces `projections_` empty regardless of `model.projections`, matching ticket #131's own
// established "not solved here" precedent for the ring/synapse-dispatch combination). This mirrors
// this codebase's own established idiom for an unintegrated combination (document + hard boundary,
// not silent conflation) rather than inventing new Ck-compensation machinery this ticket does not
// need to ship.

// ── ticket #131 [spike-scatter batch-construction subsystem]: real per-edge synapse dispatch ──────
//
// Closes the gap ticket #6's own scope note above used to describe: AssembledModel::step_tick now
// dispatches a projection's own SYNAPSE ComponentType -- not just its Cell-category populations --
// whenever `model.projections` is non-empty, composing with the existing shared-basis WeightMatrix
// (arch §4.3) rather than bypassing it (a peredge variable's `Ck`/`Sk` are real `WeightMatrix`
// storage, registered via `WeightMatrix::add_coefficient_vector` the first time a projection using
// that variable is dispatched). `ModelRuntimeBuffers`/the constructor's own parameter list are
// UNCHANGED -- everything this needs (`model.projections`, each projection's own
// `ConnectionEntry`s, the synapse type's own IrProgram) was already available; a model with no
// projections sees byte-for-byte the same behavior as before this ticket.
//
// ── Why edge-parallel, not a per-neuron `forall neuron_in` walk ──
// gpu_source.h's own `loadedge`/`accedge` doc comment documents a real, pre-existing gap: `forall
// neuron_in`/`neuron_out` both lower identically (walk the CURRENT neuron's own k^2-tree row --
// there is no transposed adjacency anywhere in this codebase), so a per-neuron dispatch of a
// synapse's OWN generated `_tick` function cannot give a postsynaptic neuron its TRUE incoming
// edges without either a real k^2-tree transpose or a second, per-synapse-type tree -- both
// substantial, undecided infrastructure this ticket does not build. Instead,
// `lower_synapse_ir_program_to_gpu_source` (gpu_source.h/.cpp) lowers a synapse's `.tick.integrate`
// (synapse_lowering.cpp's own sole shape: exactly one top-level `forall neuron_in { ... }`) with
// the SAME one-thread-per-edge calling convention its `.tick.deliver` onevent functions already use
// (`source_node_indices`/`target_node_indices`/`edge_slot_indices`/`event_count`) -- summing over a
// postsynaptic neuron's incoming edges by running one thread PER EDGE and atomically adding each
// edge's own finished current into `network_inputs[target_node]`, instead of one thread per neuron
// looping its own (wrong-direction) row. This needs no transpose at all: the edge list a
// projection's own `connections` already give (ModelSpecification, arch §1.4) is reused directly,
// addressed exactly the way WeightMatrix's Sk already is (row = the edge's PRESYNAPTIC node, slot =
// that node's own forward k^2-tree row position -- weight_matrix.h's own convention, unchanged).
//
// ── The two new per-tick dispatches, and why this ordering ──
// Both run at the SAME point in the tick the fixed scalar propagate stage already does (after this
// tick's Cell-category population dispatches + the network_inputs drain), NOT before the cell
// dispatches -- so that network_inputs already reflects this tick's fresh synaptic contribution by
// the time `step_tick` RETURNS, exactly matching the existing scalar-weight path's own OBSERVABLE
// timing (arch §0.2/ir_spec.md §3.5: a write becomes visible right after the tick it is computed
// in, drained+read at the NEXT tick's cell dispatch) -- every existing test/caller checks
// `network_inputs` right after a `step_tick` call returns, so a write that only becomes visible
// transiently DURING a later `step_tick` call (invisible from outside it) would never be observable
// at all. Within this point in the tick, in order:
// - `dispatch_synapse_delivery_events` (`<Type>_deliver_<port>`) filters each projection's static
//   edge list down to the edges whose presynaptic neuron fired THIS tick (host-side, reading --
//   but, per its own established contract, NOT clearing -- `ModelRuntimeBuffers::emit_port_flags`,
//   unioned across every tracked port the same way the fixed propagate stage's own per-port
//   dispatch loop already treats "any tracked port" as "this neuron fired," see below) and
//   dispatches each delivered edge's synapse type's own onevent handler(s) against that filtered
//   batch -- bumping Sk for this tick's fresh spikes.
// - `dispatch_synapse_integrate_edges` (`<Type>_integrate_edges`, one dispatch per PROJECTION,
//   reusing a KernelHandle compiled once per distinct synapse type) runs immediately after, over a
//   projection's WHOLE, static edge list every tick (unfiltered -- decay must run whether or not
//   anything was newly delivered), reading Sk (now including THIS tick's fresh bump above),
//   decaying, and adding each edge's finished current into `network_inputs[target_node]` (already
//   drained earlier this same tick) -- giving exactly the same one-tick latency the constant-weight
//   path already has (a spike delivered/bumped at tick T is first visible, decayed, at tick T+1's
//   cell dispatch), just computed within tick T's OWN `step_tick` call instead of straddling two.
// - Both must run BEFORE the fixed scalar propagate dispatch immediately after them (which still
//   runs, still reads+clears the SAME `emit_port_flags` for its own last_spiked/active-set
//   bookkeeping, but has its own weight contribution to network_inputs forced to zero for a model
//   with projections -- see above) -- delivery specifically needs the flags UNCLEARED when IT reads
//   them.
//
// ── The one new host-side buffer this needs, and why a single dispatch call cannot merge across
//    projections targeting different populations ──
// A synapse's `require <name> from postsynaptic` binding (gpu_source.cpp's `resolve_operand`, this
// ticket's own small extension: `name[target_node]` inside a Deliver-kind/edge-parallel function --
// unambiguous there, unlike the per-neuron-array categories, since target_node IS the postsynaptic
// neuron) needs a real pointer into the postsynaptic CELL's own packed, per-population `cell_state`
// chunk (allocator.h §4.1's SoA layout) -- but `target_node` is a GLOBAL neuron index (it doubles as
// the shared-basis `V[target_node]` lookup, which spans the whole model). Rather than adding a new,
// globally-indexed scratch buffer refreshed every tick (an extra per-tick copy this ticket avoids),
// the pointer FED for that parameter is offset by `chunk_base_offset - neuron_index_begin` (the same
// pointer-arithmetic trick `append_cell_tick_argument` already uses in the other direction for
// `network_inputs`/`emit_<port>`), so indexing it with the GLOBAL `target_node` lands on the right
// LOCAL slot -- valid only when every event in one dispatch call shares the same postsynaptic
// population, which is exactly what dispatching per-PROJECTION (rather than merging every
// projection sharing a synapse type into one call) guarantees for free (`ProjectionEntry` already
// pins one synapse type to one postsynaptic population). `network_inputs`/`emit_<port>` themselves
// need no such offset -- they are already whole-model arrays matching `target_node`'s own global
// indexing directly.
//
// ── Not solved here (documented, not silently dropped) ──
// - `require`-category resolution supports only a name matching a StateDirective on the
//   postsynaptic cell type (every real Phase-1 GLIF cell's `v`) -- a derived-only (Expose-without-
//   State) postsynaptic quantity throws a clear std::runtime_error, matching
//   append_cell_tick_argument's own established scope-boundary style.
// - `enable_delay_ring=true` silently disables this ticket's own synapse dispatch entirely (its
//   `model.projections` are not even read) -- ticket #64's delay ring and this ticket's synapse
//   dispatch have not been integrated with each other, and every existing delay-ring test/example
//   already builds its model with real projections that had zero effect before this ticket, so a
//   hard throw on the combination would regress all of them for no correctness reason; a model
//   built with `enable_delay_ring=true` keeps EXACTLY its pre-#131 behavior (the fixed scalar
//   propagate-ring path), regardless of what its `model.projections` describe.

// ── assembly ─────────────────────────────────────────────────────────────────────────────────

// The names of the two engine-fixed scaffold kernels assemble_master_kernel_source always emits
// (present in every AssembledMasterKernelSource regardless of model content -- harmless to
// compile/dispatch even for a model with zero neurons/edges).
extern const char *const MASTER_KERNEL_DRAIN_NAME;
extern const char *const MASTER_KERNEL_PROPAGATE_NAME;

struct AssembledMasterKernelSource {
    // Parallel to model.populations: population p's cell type's generated GPU source (ticket #55),
    // or a default-constructed (empty-source) GpuSource if that cell type's IR has nothing in any
    // of the 7 per-neuron stages (ticket #55 then generates no `_tick` function at all -- rare in
    // Phase 1, but not assumed away).
    Vector<GpuSource> population_gpu_sources;

    // The two engine-fixed scaffold kernels (see this header's own doc comment above) -- each a
    // single-function GpuSource (one entry in `.functions`, named MASTER_KERNEL_DRAIN_NAME /
    // MASTER_KERNEL_PROPAGATE_NAME respectively).
    GpuSource drain_network_inputs_source;
    GpuSource propagate_source;
};

// Assembles every Cell-category population's generated GPU source (from `type_library_ir_programs`,
// parallel to `model.type_library`, matching allocator.h's own convention) plus the two fixed
// scaffold kernels. Does not compile anything (see AssembledModel below for compile+cache+dispatch).
// Throws std::runtime_error if type_library_ir_programs.size() != model.type_library.size(), or if
// any population's type_library_index is out of range.
AssembledMasterKernelSource assemble_master_kernel_source(
    const ModelSpecification &model, const Vector<IrProgram> &type_library_ir_programs);

// ── ticket #64 [F3]: ring-based deliver-drain/propagate kernel sources ──────────────────────────
//
// The delay-ring generalization of the two engine-fixed scaffold kernels above (see delay_ring.h
// for the ring design). Deliberately NOT added to AssembledMasterKernelSource/
// assemble_master_kernel_source above (that struct/function's own established contract is
// unchanged by this ticket -- only assembled/compiled by AssembledModel's constructor when built
// with enable_delay_ring=true, see below). Exported as free functions purely so tests can genuinely
// compile their exact MSL text through the real Metal toolchain directly, mirroring how
// AssembledMasterKernelSource's own public fields are used for the same purpose.
GpuSource build_drain_ring_kernel_gpu_source();
GpuSource build_propagate_ring_kernel_gpu_source();

// ── compile + cache + dispatch ──────────────────────────────────────────────────────────────────

// Wraps backend::compile_kernel so a compile failure's thrown message carries the generated GPU
// source that failed to compile, plus (where applicable) the IR program it was lowered from --
// ticket #60 [X1] (arch §0.4/§1.3). compile_kernel itself only reports the raw backend compiler
// diagnostic (Metal newLibrary's NSError text / NVRTC's compile log) against a source string the
// caller never sees again once compile_kernel returns -- not enough to debug a bad ComponentType
// lowering without also seeing WHAT was actually emitted. `kernel_label` identifies which kernel
// this is for in the thrown message (e.g. a population id + ComponentType name); `ir_dump` is the
// empty string for a kernel with no per-ComponentType IR to show (the two engine-fixed scaffold
// kernels). Exported (used by AssembledModel's constructor below, and independently testable).
KernelHandle compile_kernel_or_throw_with_source(const String &source_text, const String &function_name,
                                                  const String &kernel_label, const String &ir_dump);

// The engine-owned, implicit per-neuron buffers ir_spec.md §2 says `.alloc` never declares
// (`network_inputs`, `last_spiked`, the active-set arrays, the k^2-tree/shared-basis WeightMatrix --
// arch §0.1) plus the one buffer family this ticket itself introduces and owns the allocation
// contract for (`emit_<port>` flags, gpu_source.h's own "not yet built" placeholder -- ticket #6 is
// what wires it to real storage). None of these are owned by AssembledModel: a caller (e.g. a
// SpikeEngine, or master_kernel_tests.cpp's own fixture) supplies live pointers into whichever
// buffers it already owns, so this ticket's dispatch reuses the SAME engine-owned state rather than
// allocating a second, redundant copy.
struct ModelRuntimeBuffers {
    ModelAllocation *allocation = nullptr;
    WeightMatrix *weights = nullptr;

    f32 *network_inputs = nullptr;         // [total_neuron_count]
    s64 *last_spiked = nullptr;            // [total_neuron_count]
    s32 *next_active_neuron_indices = nullptr; // [total_neuron_count] -- active-set enqueue target
    s32 *next_active_neuron_count = nullptr;   // [1] -- step_tick resets this to 0 at the start of
                                                // every call (matching SpikeEngine::step_simulation's
                                                // own per-tick reset, src/core/engine.cpp), so a
                                                // caller need only zero-initialize it once up front
    s32 *active_generation = nullptr;          // [total_neuron_count]

    // One flag buffer per distinct EventPort name emitted anywhere in the model's Cell-category
    // type-in-use IR (see collect_emit_port_names below), each [total_neuron_count] bool,
    // zero-initialized by the caller before the first tick. A population's `_tick` kernel sets its
    // own port's flag true on firing (ticket #55's `emit <port>` convention); the fixed propagate
    // stage reads+clears it.
    UnorderedMap<String, bool *> emit_port_flags;

    // ── ticket #64 [F3]: spike-delay subsystem ──────────────────────────────────────────────────
    // Non-null activates the ring-based, delay-aware deliver-drain/propagate stages below (only
    // usable if AssembledModel was constructed with enable_delay_ring=true -- step_tick throws on a
    // mismatch either way). When non-null, `network_inputs`/`next_active_neuron_indices`/
    // `next_active_neuron_count`/`active_generation` above are ignored entirely (superseded by
    // `delay_ring`'s own ring-shaped equivalents, see delay_ring.h) -- a caller using the delay ring
    // need not allocate those at all. When null (the default), step_tick's fixed drain/propagate
    // stages behave exactly as they did before this ticket, byte for byte.
    DelayRingAllocation *delay_ring = nullptr;

    // Per-neuron persistent RNG state (ticket #65 [F4]) -- gpu_source.h's own `rng_state` reserved
    // buffer (xorshift32; ticket #55), one `u32` per neuron, model-wide (whole `[total_neuron_count]`,
    // same "+neuron_index_begin" slicing convention as network_inputs/emit_port_flags above). Left
    // null iff no population's kernel in this model actually uses `rand`/`randn` -- ticket #6's
    // AssembledModel only throws asking for this when a dispatched kernel's own parameter list
    // actually needs it, so a model with no on-device generator can leave this unset. The caller
    // seeds every entry to a NONZERO value before the first tick (xorshift32 is stuck at 0 forever
    // once it reaches 0 -- e.g. seed neuron n with `(n+1)*2654435761u | 1u`, or any other
    // per-neuron-distinct nonzero scheme); step_tick only ever reads/advances it, never re-seeds it.
    u32 *rng_state = nullptr; // [total_neuron_count]
};

// Every distinct EventPort name any Cell-category type-in-use's IR fires `emit` on, in first-seen
// order across `model.populations` -- exactly the set of buffers a caller must allocate and set on
// ModelRuntimeBuffers::emit_port_flags before calling AssembledModel::step_tick.
Vector<String> collect_emit_port_names(const ModelSpecification &model, const Vector<IrProgram> &type_library_ir_programs);

// Owns every compiled KernelHandle for one assembled model (one per population with a non-empty
// per-neuron tick kernel, plus the fixed drain/propagate kernels, one propagate dispatch per
// distinct emit-port name) -- compiled once from assemble_master_kernel_source()'s output and
// reused for every subsequent step_tick call (ticket #6's "compile once ... cache, recompile only
// when the model changes": constructing a NEW AssembledModel is the only way to recompile, matching
// how a NEW ModelAllocation/WeightMatrix is how the rest of this pipeline represents "the model
// changed").
class AssembledModel {
public:
    // `enable_delay_ring` (ticket #64 [F3]): when true, also assembles+compiles the ring-based
    // deliver-drain/propagate kernels (delay_ring.h) and step_tick uses THOSE instead of the flat,
    // one-tick-ahead ones -- every step_tick call then requires ModelRuntimeBuffers::delay_ring to
    // be non-null. When false (the default), behavior is exactly what it was before this ticket;
    // ModelRuntimeBuffers::delay_ring must be null (step_tick throws on either mismatch).
    AssembledModel(const ModelSpecification &model, const Vector<IrProgram> &type_library_ir_programs,
                   bool enable_delay_ring = false);
    AssembledModel(const AssembledModel &) = delete;
    AssembledModel &operator=(const AssembledModel &) = delete;
    ~AssembledModel();

    // Runs one tick's cell-type dynamics (stages 2-5, one dispatch per population over its own
    // full neuron range -- arch §4.1's cell-type-boundary dispatch) followed by the fixed
    // deliver-drain and k^2-tree propagate/scatter + active-set-enqueue stages (stages 1's
    // counterpart/6/9) -- see master_kernel.cpp for the exact dispatch order and why it preserves
    // ir_spec.md §3.5's >=1-tick network_inputs latency. `dt`/`tick`/`next_tick` match
    // SpikeEngine::step_simulation's own convention (arch §2's clock-driven tick). Takes no
    // ModelSpecification/IrProgram parameter -- everything needed from them was already captured
    // at construction time (a copy of the type-library IR programs, each population's neuron range
    // and type-library index), matching "recompile only when the model changes": building a new
    // AssembledModel is the only way this cached information changes.
    //
    // Active-set x nonlinear rule (arch §0.5, ticket #62 [F1]): every population's kernel above runs
    // over its FULL neuron range every tick, regardless of population_is_closed_form_advanceable
    // below (see master_kernel.cpp's own header comment on this ticket's still-deliberate scope
    // boundary) -- this is unconditionally correct for BOTH tags. A nonlinear-tagged population must
    // never receive a closed-form multi-tick skip; running its full range every tick trivially
    // satisfies that (no skip ever happens for anyone here). A closed_form_advanceable-tagged
    // population's own SEPARATE closed-form fast-forward is kernels.metal/kernels.cu's pre-existing
    // `apply_decay` + active-set mechanism on the hardcoded SpikeEngine path -- untouched by this
    // ticket. A future skip-dispatch fast path for THIS master-kernel path (deferred by ticket #6,
    // see master_kernel.h's own header comment) MUST gate on population_is_closed_form_advanceable
    // and must never apply a multi-tick skip to a population for which it returns false.
    void step_tick(const ModelRuntimeBuffers &buffers, f32 dt, s64 tick, s64 next_tick);

    // Whether `model.populations[population_index]`'s cell type was tagged
    // closed_form_advanceable (arch §0.5, ticket #62 [F1]) at construction time -- false for a
    // population with no per-neuron kernel (has_kernel == false; e.g. an empty-IR cell type, see
    // AssembledMasterKernelSource's own doc comment), matching IrProgram's own "not applicable"
    // default. Exposed so a caller (or a future skip-dispatch implementer) can consult the tag
    // without re-deriving it from the model/IR programs it already handed to the constructor.
    bool population_is_closed_form_advanceable(usize population_index) const;

    // ── ticket #132: real STDP support -- mirrors SpikeEngine's own SC-11 enable/disable API shape
    // exactly (see this header's own "ticket #132" doc comment above for the full design). Disabled
    // (learning_rate == 0.0f) by default -- byte-for-byte the same propagate/step_tick behavior as
    // before this ticket until a caller opts in.
    bool plasticity_enabled() const;

    // No-op if already enabled (matches SpikeEngine::enable_plasticity). Throws std::runtime_error
    // if this AssembledModel has any real per-edge synapse dispatch active (`projections_`
    // non-empty, ticket #131) -- see this header's own "ticket #132" doc comment for why combining
    // the two is not yet safe (both write the shared U/V basis, uncompensated).
    void enable_plasticity(f32 learning_rate = 0.00222f);

    // No-op if already disabled (matches SpikeEngine::disable_plasticity).
    void disable_plasticity();

private:
    struct PopulationRuntimeInfo {
        bool has_kernel = false;
        KernelHandle handle{};
        Vector<String> parameter_names_in_order;
        s32 type_library_index = -1;
        s32 neuron_index_begin = 0;
        s32 population_size = 0;
    };

    Vector<PopulationRuntimeInfo> populations_;   // parallel to model.populations at construction time
    Vector<IrProgram> type_library_ir_programs_;  // owned copy, indexed by type_library_index
    s64 total_neuron_count_ = 0;

    Vector<String> emit_port_names_; // dispatch order for the propagate stage

    KernelHandle drain_kernel_handle_{};
    Vector<String> drain_parameter_names_;

    KernelHandle propagate_kernel_handle_{};
    Vector<String> propagate_parameter_names_;

    // ── ticket #64 [F3]: ring-based deliver-drain/propagate kernels -- only assembled/compiled
    // when constructed with enable_delay_ring=true (see the constructor's own doc comment above). ──
    bool delay_ring_enabled_ = false;

    KernelHandle drain_ring_kernel_handle_{};
    Vector<String> drain_ring_parameter_names_;

    KernelHandle propagate_ring_kernel_handle_{};
    Vector<String> propagate_ring_parameter_names_;

    // ── ticket #132: real STDP support -- see this header's own "ticket #132" doc comment above.
    // 0.0f == disabled (the default); mirrors SpikeEngine::learning_rate's own "> 0.0f == enabled"
    // convention exactly.
    f32 stdp_learning_rate_ = 0.0f;

    // Stage 7 (arch §2): for every neuron whose `last_spiked == tick` (i.e. fired THIS tick -- the
    // fixed propagate dispatch(es) above have already set this for every emit port by the time this
    // runs), walks its real k^2-tree neighbors and applies the same rank-1 Hebbian nudge
    // step_apply_hebbian_update's own real call sites do, via WeightMatrix::update() (see this
    // header's own "ticket #132" doc comment for the full derivation). No-op when
    // stdp_learning_rate_ == 0.0f. Called from both the flat and ring-mode branches of step_tick,
    // immediately after their own propagate dispatch(es).
    void apply_stdp_plasticity(const ModelRuntimeBuffers &buffers, s64 tick);

    // ── ticket #131: spike-scatter batch-construction subsystem -- see this header's own "spike-
    // scatter batch-construction subsystem" doc comment above for the full design. ──────────────────
    Vector<ProjectionEntry> projections_; // cached copy of model.projections at construction time

    // One compiled onevent/deliver function (`<Type>_deliver_<port>`, ticket #131).
    struct DeliverFunctionRuntimeInfo {
        KernelHandle handle{};
        Vector<String> parameter_names_in_order;
    };

    // One entry per DISTINCT synapse_type_library_index actually used by some projection -- compiled
    // once (constructor) and reused by every projection sharing that synapse type; the WeightMatrix
    // matrix_index registered for each of that type's OWN peredge variables is likewise shared
    // (WeightMatrix's Sk is addressed by (presynaptic node, that node's own forward-row slot),
    // model-wide, independent of which projection an edge belongs to -- see this header's own doc
    // comment above).
    struct SynapseTypeRuntimeInfo {
        KernelHandle integrate_edges_handle{};
        Vector<String> integrate_edges_parameter_names_in_order;
        Vector<DeliverFunctionRuntimeInfo> deliver_functions; // in lower_synapse_ir_program_to_gpu_source's own order

        bool matrix_indices_registered = false;
        UnorderedMap<String, s64> matrix_index_by_peredge_name; // populated lazily, see ensure_synapse_dispatch_topology_built
    };
    UnorderedMap<s32, SynapseTypeRuntimeInfo> synapse_types_by_type_library_index_;

    // One entry per ProjectionEntry (parallel to projections_), built LAZILY on the first step_tick
    // call that has any projections (needs ModelRuntimeBuffers::weights's own real k^2-tree/
    // max_neighbor_count, only available at step_tick time -- ModelSpecification::adjacency is often
    // unset for a hand-built ModelSpecification fixture, see this header's own doc comment above).
    struct ProjectionEdgeTopology {
        s64 edge_count = 0;
        GpuPointer<s32> source_nodes;  // [edge_count] -- global presynaptic neuron index (Sk row)
        GpuPointer<s32> target_nodes;  // [edge_count] -- global postsynaptic neuron index
        GpuPointer<s32> forward_slots; // [edge_count] -- source's own forward k^2-tree row slot (Sk column)

        // Reused every tick, refilled (memcpy) with THIS tick's fired-source subset of the three
        // arrays above -- see dispatch_synapse_delivery_events.
        GpuPointer<s32> delivery_scratch_source_nodes;
        GpuPointer<s32> delivery_scratch_target_nodes;
        GpuPointer<s32> delivery_scratch_edge_slots;
    };
    Vector<ProjectionEdgeTopology> projection_edge_topology_; // parallel to projections_
    bool synapse_dispatch_topology_built_ = false;

    // Builds projection_edge_topology_ (from buffers.weights's real k^2-tree) and registers every
    // used synapse type's peredge variables as real WeightMatrix matrix_index(es) -- exactly once,
    // idempotent thereafter. Throws if a projection's ConnectionEntry is not a real edge in
    // buffers.weights's k^2-tree (a mismatched ModelRuntimeBuffers::weights/model.projections pairing).
    void ensure_synapse_dispatch_topology_built(const ModelRuntimeBuffers &buffers);

    // Dispatches every projection's `<Type>_integrate_edges` function over its WHOLE, static edge
    // list (unfiltered), reading+decaying Sk and adding each edge's finished current into
    // `buffers.network_inputs[target_node]`.
    void dispatch_synapse_integrate_edges(const ModelRuntimeBuffers &buffers, f32 dt, s64 tick);

    // Filters every projection's static edge list down to this tick's fired sources (reads, but
    // does not clear, buffers.emit_port_flags) and dispatches that projection's synapse type's own
    // `<Type>_deliver_<port>` function(s) against the filtered batch.
    void dispatch_synapse_delivery_events(const ModelRuntimeBuffers &buffers, f32 dt, s64 tick);
};

} // namespace spikecorec::nml
