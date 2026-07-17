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
//   its own onevent/deliver handler, ticket #55's `<Type>_deliver_<port>` functions) is NOT wired
//   up: this needs a "spike-scatter batch construction" subsystem (building the per-tick
//   source/target/edge-slot arrays a deliver function's calling convention expects) that
//   gpu_source.h's own header comment already flags as not yet built by ANY prior ticket. This
//   ticket's fixed propagate stage instead reproduces exactly what the CURRENT hardcoded engine's
//   own connectivity already does: a spiking neuron's own k^2-tree row, reconstructed via
//   WeightMatrix's shared U/V basis (or a constant weight) -- the same mechanism arch §5 says Phase
//   1 "reuses... directly." A model with real per-edge synapse ComponentTypes still gets ITS
//   `_tick`/`_deliver_<port>` functions correctly compiled by assemble_master_kernel_source (they
//   are, after all, just more IrPrograms), but this ticket's AssembledModel::step_tick only
//   dispatches Cell-category population kernels + the two fixed stages -- a synapse type's own
//   generated functions are compiled-and-ready, not yet invoked.
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
};

} // namespace spikecorec::nml
