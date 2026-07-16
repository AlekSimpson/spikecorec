#pragma once

#include "spikecorec/nml/ir.h"
#include "spikecorec/nml/model_specification.h"

namespace spikecorec::nml {

// ── Synapse dynamics -> IR lowering (ticket #51 [B3]; arch §3.3 D3, §4.2-§4.3;
// IR spec §3.3, §3.5, §4's expOne + NMDA examples) ───────────────────────
//
// Lowers one Synapse-category `TypeLibraryEntry`'s flattened `SynapseType`
// Dynamics into a runnable `IrProgram`, dispatching on ticket #7's own
// `is_aggregatable`/`is_conductance_based` classification flags (already
// computed and reviewer-verified in model_specification.cpp -- NOT
// recomputed here) into one of three shapes:
//   - **Aggregatable** (`is_aggregatable == true`): every state variable
//     becomes a per-neuron `accum` slot (arch §4.2); `network_inputs` is
//     read/written by bare name, no per-edge iteration needed.
//   - **Per-edge** (`is_aggregatable == false`): every state variable
//     becomes a `peredge` slot (arch §4.3); `.tick @integrate` wraps a
//     `forall neuron_in { loadedge ...; ... }` loop (IR spec §4's NMDA
//     example) since a peredge variable has no per-neuron bare name to read
//     directly.
// `is_conductance_based` needs no special handling here at all -- IR spec
// §3.5's own invariant ("no special cell-side case") means a conductance
// synapse's `g·(erev−v)` is just its declared `DerivedVariable` exposing
// `i`, lowered through the exact same generic machinery as a current-based
// synapse's trivial `i = g` identity.
//
// Reuses cell_lowering.cpp's shared expression/TimeDerivative-lowering
// machinery via expression_lowering.h (ticket #51's refactor) rather than
// duplicating it -- see that header's doc comment.
//
// Scope (documented, not silently dropped): a synapse's `OnStart` initial
// values are not encoded (Phase-1 default: 0, matching every real D3
// fixture and `AccumDirective`/`PeredgeDirective`'s absent-initial-value
// convention -- unlike `StateDirective`, neither carries an `initial_value`
// field, so a future non-zero-initial-value synapse would need that added,
// out of this ticket's scope). A `DerivedVariable` using `select`/`reduce`
// (Children-based sub-mechanism composition, e.g. `blockingPlasticSynapse`'s
// real `plasticityFactor`/`blockFactor`) throws -- out of Phase-1 scope, per
// the IR spec's own NMDA example eliding the Mg-block. A per-edge
// (`peredge`) state variable's own `TimeDerivative` is not lowered into an
// explicit decay instruction -- arch §4.3's shared low-rank-basis +
// sparse-delta-buffer scheme is pure memory compression (Read
// `U·diag(Ck)·Vᵀ + Sk`, Update `Sk[edge]+=x`, Refit re-fits the plane to
// current values and clears the scratchpad -- none of the three integrates
// an ODE), so it does NOT already provide per-edge decay; a `peredge`
// variable's time-evolution is simply deferred/unspecified in Phase 1,
// matching the provisional IR spec's own NMDA example (which declares
// `param tau` but never references it in `.tick`, and likewise never decays
// its per-edge `g`). A per-edge synapse that actually declares a
// `TimeDerivative` gets a build-time warning (not a throw -- the emitted IR
// is still spec-conformant) so a future real per-edge synapse with genuine
// decay dynamics doesn't lose it silently; see synapse_lowering.cpp.

// Lowers one Synapse-category `TypeLibraryEntry` to its `IrProgram`. Throws
// std::runtime_error if `synapse_entry.category != TypeLibraryCategory::Synapse`,
// if no `DerivedVariable` exposes `i` (the finished current basePointCurrent
// requires), if a `DerivedVariable` uses `select`/`reduce`, if an `OnEvent`
// body contains anything other than an additive self-increment
// `StateAssignment` (`var = var + <increment>`), or if a Dynamics expression
// references an identifier the type never declares.
IrProgram lower_synapse_to_ir(const TypeLibraryEntry &synapse_entry);

// Convenience driver: lowers every Synapse-category entry in `model`'s type
// library, in type-library order.
Vector<IrProgram> lower_all_synapse_types_to_ir(const ModelSpecification &model);

} // namespace spikecorec::nml
