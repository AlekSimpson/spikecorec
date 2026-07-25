#pragma once

#include "spikecorec/core/backend.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/nml/allocator.h"
#include "spikecorec/nml/delay_ring.h"
#include "spikecorec/nml/gpu_source.h"
#include "spikecorec/nml/ir.h"
#include "spikecorec/nml/model_specification.h"

namespace spikecorec::nml {

// REFACTOR: Ya this needs to be totally rewritten or deleted and turned into a set of utiltiy methods in the engine

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
//   subsystem is now built -- ticket #131 (`SpikeEngine`'s own synapse-dispatch machinery,
//   src/core/engine.cpp), `ModelSpecification::projections` driven.** This ticket's own
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

// ── ticket #132 (real STDP support) / ticket #131 (spike-scatter batch-construction subsystem) ──
//
// Both tickets originally added real per-edge synapse dispatch and STDP plasticity directly onto
// `nml::AssembledModel`. That class (and the `ModelRuntimeBuffers`/`PopulationRuntimeInfo` types it
// used) has since been folded into `SpikeEngine` (see include/spikecorec/core/engine.h's own
// "Stage 2 of folding nml::AssembledModel into SpikeEngine" doc comment, and src/core/engine.cpp) --
// the full design rationale for both tickets (why `WeightMatrix::update()` is reused rather than a
// new kernel, the edge-parallel dispatch shape, the delivery/integrate-edges ordering, the
// `require from postsynaptic` pointer-offset trick, etc.) is preserved there, at the call sites that
// actually run today, rather than duplicated here for a class that no longer exists.

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
// scaffold kernels. Does not compile anything (see SpikeEngine's own NML-mode constructor,
// src/core/engine.cpp, for compile+cache+dispatch).
// Throws std::runtime_error if type_library_ir_programs.size() != model.type_library.size(), or if
// any population's type_library_index is out of range.
AssembledMasterKernelSource assemble_master_kernel_source(
    const ModelSpecification &model, const Vector<IrProgram> &type_library_ir_programs);

// ── ticket #64 [F3]: ring-based deliver-drain/propagate kernel sources ──────────────────────────
//
// The delay-ring generalization of the two engine-fixed scaffold kernels above (see delay_ring.h
// for the ring design). Deliberately NOT added to AssembledMasterKernelSource/
// assemble_master_kernel_source above (that struct/function's own established contract is
// unchanged by this ticket -- only assembled/compiled by SpikeEngine's own NML-mode constructor
// when the model needs a delay ring, see src/core/engine.cpp). Exported as free functions purely so tests can genuinely
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
// kernels). Exported (used by SpikeEngine's own NML-mode constructor, src/core/engine.cpp, and
// independently testable).
KernelHandle compile_kernel_or_throw_with_source(const String &source_text, const String &function_name,
                                                  const String &kernel_label, const String &ir_dump);

// Every distinct EventPort name any Cell-category type-in-use's IR fires `emit` on, in first-seen
// order across `model.populations` -- exactly the set of per-population emit-port flag buffers a
// caller (SpikeEngine's own NML-mode constructor, see engine.h/engine.cpp) must allocate and wire up
// before running the assembled model.
Vector<String> collect_emit_port_names(const ModelSpecification &model, const Vector<IrProgram> &type_library_ir_programs);

} // namespace spikecorec::nml
