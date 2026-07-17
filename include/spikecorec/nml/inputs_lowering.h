#pragma once

#include "spikecorec/nml/ir.h"
#include "spikecorec/nml/model_specification.h"

namespace spikecorec::nml {

// ── On-device Inputs (D4 generator) -> IR lowering (ticket #65 [F4]; arch §3.3 D4, §5 Phase 2)
// ──────────────────────────────────────────────────────────────────────────────────────────────
//
// Phase 1 (ticket #58 [E1]) host-precomputes `pulseGenerator` -- a deterministic function of time
// with nothing to evaluate per tick on the GPU. This ticket lowers the Phase-2 on-device
// generators -- the ones that either need genuine per-thread randomness (a Poisson spike source)
// or are simplest expressed as on-device per-tick computation (`sineGenerator`/`rampGenerator`/
// `voltageClamp`) -- into real `IrProgram`s, using ONLY the IR ops the locked spec already provides
// (arithmetic/math/compare/boolean, `rand`, `if/elif/else`; docs/nml_ir_spec.md §3.3-§3.4). No new
// IR op or `.tick` construct is introduced.
//
// Unlike cell_lowering.cpp/synapse_lowering.cpp, this file does NOT generically walk a
// ComponentType's own `<Dynamics>` tree (`TimeDerivative`/`OnCondition`/`OnStart` text) through the
// shared LEMS-expression parser (expression_lowering.h). That parser has no function-call syntax
// (`sin(...)`, `log(...)`, `random(...)`) and no support for a `.and.`-joined compound `OnCondition`
// test (both of which the REAL std-lib `sineGenerator`/`rampGenerator`/`voltageClamp`/
// `SpikeSourcePoisson` XML text uses) -- extending that shared, load-bearing parser with
// arbitrary-function-call/compound-condition grammar just for these four fixed ComponentTypes would
// be a disproportionate, speculative grammar addition serving no other lowering. Instead, each
// supported ComponentType (identified by `TypeLibraryEntry::component_type_name`) gets one
// hand-written IR-builder function below that emits the SAME semantics directly via ir.h's typed
// instruction structs, reading the bound instance's own resolved constants from
// `TypeLibraryEntry::baked_constants` exactly the way cell_lowering.cpp already does for a
// `Parameter`. This is a deliberate, bounded judgment call: the four supported types are a small,
// fixed, well-known set (not arbitrary user-authored dynamics), so template-substituting by
// ComponentType NAME is a direct extension of the IR's own "near-direct template substitution"
// governing principle one level up, and mirrors stimulus_schedule.cpp's own precedent of
// special-casing `pulseGenerator` by name rather than generically interpreting its Dynamics text.
//
// ── Supported ComponentTypes (arch §3.3 D4) ─────────────────────────────────────────────────────
// - `sineGenerator` / `rampGenerator`: a deterministic current, computed on-device from `tick*dt`
//   instead of host-precomputed (Phase 2's own stated alternative to Phase 1's host-precompute
//   path) -- written into `network_inputs` every tick exactly like a synapse's finished current
//   (ir_spec.md §3.5: "there is no special cell-side case").
// - `voltageClamp`: likewise on-device, but additionally needs the postsynaptic membrane potential
//   -- declared via `require v from postsynaptic` (the same construct NMDA's own ir_spec.md §4
//   example uses). KNOWN LIMITATION (not this ticket's gap to close): master_kernel.cpp's
//   per-neuron dispatch-argument builder (ticket #6 [C3]) explicitly throws on any `require`-bound
//   kernel parameter ("out of Phase-1 GLIF-cell scope") -- resolving `v` would mean binding a
//   stimulus's own generated kernel to ANOTHER population's per-neuron cell-state slot at the same
//   neuron index, which is a real, separate wiring gap ticket #6 never built and this ticket does
//   not build either (out of scope: it is not needed by this ticket's own acceptance criteria).
//   `lower_inputs_to_ir` still produces correct, independently-compilable IR/GPU source for
//   `voltageClamp` -- only its AssembledModel dispatch is out of scope.
// - `SpikeSourcePoisson` (PyNN.xml): a genuine on-device Poisson spike source, using the IR's
//   `rand` op (ir_spec.md §3.3/§6: "Stochastic sources ... cover ... SpikeSourcePoisson"). Lowered
//   as a spike-emitting Inputs type exactly like a Cell's own `OnCondition`+`EventOut` (arch §3.2) --
//   it fits the EXISTING per-population dispatch path (master_kernel.cpp iterates `model.populations`
//   generically over whatever TypeLibraryCategory a population's bound type is, arch's own ST2:
//   "population (size,component) -> a contiguous neuron range" of ANY Dynamics ComponentType) with
//   no new master-kernel machinery -- unlike `explicitInput`-bound single-target generators (sine/
//   ramp/voltageClamp's usual real-NML placement), which `model.stimuli` never gets dispatched
//   through at all yet (only host-precomputed `pulseGenerator` has a wired path, via
//   stimulus_schedule.cpp, entirely outside AssembledModel).
//   Simplified from the real ComponentType's own definition (documented, in the spirit of the
//   locked spec's own NMDA "Mg-block elided" precedent): tracks two state variables (`tsince`,
//   `tnext`) rather than the original's four (`tsince`/`tnextIdeal`/`tnextUsed`/`isi`) -- the extra
//   two exist in the original purely to satisfy NEURON's continuous-time WATCH-statement
//   integration, irrelevant to this engine's tick-discretized model; the `H()` Heaviside-based
//   "clamp to a long future time" trick that encodes the `start`/`duration` firing window is
//   likewise replaced by an ordinary boolean AND gate over `tick*dt` against `start`/`start+duration`
//   -- behaviorally equivalent, expressed with existing compare/boolean ops instead of inventing an
//   `H()` op no other ComponentType needs.
//   Known gap: the real ComponentType seeds `tnextUsed` at OnStart as `start - log(random)/rate`;
//   this simplified `tnext` zero-initializes instead (allocate_model never applies a
//   `StateDirective`'s `initial_value` yet -- a pre-existing, documented gap, not introduced here).
//   At `start=0` this costs one benign extra spike per neuron. At `start > 0` it instead fires
//   every neuron once per tick from window onset until `tnext` accumulates past the current time --
//   a burst of roughly `start*rate` spurious spikes concentrated right after onset. On-device
//   Poisson with a nonzero `start` should be treated as unverified until `initial_value` wiring
//   lands; this ticket's own acceptance test only exercises `start=0`.
//
// ── NOT supported: `poissonFiringSynapse` / `compoundInput` (documented judgment call) ──────────
// Both require summing current over an arbitrary, model-specific list of CHILD sub-components
// (`poissonFiringSynapse`'s `ChildInstance` synapse; `compoundInput`'s `Children` list of nested
// `basePointCurrent` generators) -- i.e. `DerivedVariable select="...[*]/i" reduce="add"` over a
// generic child set. The locked IR spec has no construct for this: `forall` only ever iterates a
// neuron's k²-tree in/out edges, and `docs/nml_ir_spec.md` §6 explicitly defers the generalization
// to child/neighbor sets to Phase 3 (ticket #70 [G3]). Lowering either of these types would mean
// inventing that deferred construct early, out of scope for this ticket -- `lower_inputs_to_ir`
// throws a clear, named error for them instead of silently guessing at a shape.

// Lowers one Inputs-category `TypeLibraryEntry` to its `IrProgram`. Dispatches on
// `inputs_entry.component_type_name` (see this header's own doc comment for the exact supported
// set). Throws std::runtime_error if `inputs_entry.category != TypeLibraryCategory::Inputs`, if a
// required baked constant (`TypeLibraryEntry::baked_constants`) is missing for the matched
// ComponentType, or if `component_type_name` names an Inputs ComponentType this file does not
// lower (including, by name, the two documented exclusions above).
IrProgram lower_inputs_to_ir(const TypeLibraryEntry &inputs_entry);

// Convenience driver: lowers every Inputs-category entry in `model`'s type library (skipping
// Cell/Synapse entries), in type-library order -- mirrors
// cell_lowering.h's `lower_all_cell_types_to_ir`/synapse_lowering.h's own analogue.
Vector<IrProgram> lower_all_inputs_to_ir(const ModelSpecification &model);

} // namespace spikecorec::nml
