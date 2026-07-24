#pragma once

#include "spikecorec/nml/ir.h"
#include "spikecorec/nml/model_specification.h"

namespace spikecorec::nml {

// ── Synapse dynamics -> IR lowering (ticket #51 [B3]; arch §3.3 D3, §4.3;
// IR spec §3.3, §3.5, §4's NMDA example) ─────────────────────────────────
//
// Lowers one Synapse-category `TypeLibraryEntry`'s flattened `SynapseType`
// Dynamics into a runnable `IrProgram`. Every synapse -- current-based or
// conductance-based, regardless of whether its contributions would
// superpose across converging edges -- lowers uniformly to the SAME
// per-edge shape (arch §4.3's design revision: #52's shared-basis
// WeightMatrix makes per-edge storage cheap enough that the old separate
// per-neuron-accumulator fast path for superposable synapses is gone;
// ticket #57, which would have consumed it, is closed as won't-do): every
// state variable becomes a `peredge` slot; `.tick @integrate` wraps a
// `forall neuron_in { loadedge ...; ... }` loop (IR spec §4's NMDA example)
// since a peredge variable has no per-neuron bare name to read directly.
// `is_conductance_based` needs no special handling beyond that -- IR spec
// §3.5's own invariant ("no special cell-side case") means a conductance
// synapse's `g·(erev−v)` is just its declared `DerivedVariable` exposing
// `i`, lowered inside that same `forall` body through the exact same
// generic machinery as a current-based synapse's trivial `i = g` identity.
//
// Reuses cell_lowering.cpp's shared expression/TimeDerivative-lowering
// machinery via expression_lowering.h (ticket #51's refactor) rather than
// duplicating it -- see that header's doc comment.
//
// Scope (documented, not silently dropped): a synapse's `OnStart` initial
// values are not encoded (Phase-1 default: 0, matching every real D3
// fixture and `PeredgeDirective`'s absent-initial-value convention --
// unlike `StateDirective`, it carries no `initial_value` field, so a future
// non-zero-initial-value synapse would need that added, out of this
// ticket's scope). A `DerivedVariable` using `select`/`reduce`
// (Children-based sub-mechanism composition, e.g. `blockingPlasticSynapse`'s
// real `plasticityFactor`/`blockFactor`) throws -- out of Phase-1 scope, per
// the IR spec's own NMDA example eliding the Mg-block. Every `peredge` state
// variable's own `TimeDerivative` IS lowered: one matching the recognized
// linear-decay shape (`detect_linear_decay_shape`, shared with
// cell_lowering.cpp's own direct-mutation case) uses the closed-form
// `expdecay` path; one that doesn't (e.g. `alphaCurrentSynapse`'s own `I`,
// whose coupled right-hand side references another per-edge state variable,
// `J`) falls back to a general per-edge forward-Euler integration instead.
// Per-edge storage is accumulate-only (arch §4.3: `accedge` is
// `Sk[edge]+=value`; there is no direct "set" op), so neither path can
// mutate a peredge variable in place the way cell-side lowering does --
// both instead read the current value via `loadedge` and `accedge` back
// only the DELTA (the closed-form path's decayed-value-minus-old-value; the
// forward-Euler path's own `dt * rhs`) so the reconstructed read reflects
// the updated value next tick (see synapse_lowering.cpp). A per-edge
// TimeDerivative referencing another per-edge state variable whose OWN
// TimeDerivative was already integrated earlier the same tick (i.e. it
// precedes this one in the type's declared Dynamics) throws rather than
// silently reading its already-updated value -- see synapse_lowering.cpp's
// own header comment on the per-variable integration loop.

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
