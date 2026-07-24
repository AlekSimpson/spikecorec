#pragma once

#include "spikecorec/nml/ir.h"
#include "spikecorec/nml/model_specification.h"

namespace spikecorec::nml {

// ── Cell dynamics -> IR lowering (ticket #50 [B2]; arch §3.1-§3.2, §5 Phase 1;
// IR spec §4's GLIF1 + refractory-regime examples) ──────────────────────
//
// Lowers one Cell-category `TypeLibraryEntry`'s flattened `CellType` Dynamics
// (ticket #49's `ResolvedComponentType`, materialized by ticket #7's
// `ModelSpecification`) into a runnable `IrProgram` (ticket #4's ir.h) -- the
// generic per-cell-type front-end targeted at the full GLIF family (arch §5
// Phase 1). This module does NOT parse/resolve NML (#2/#49 already did that),
// does NOT lower IR to GPU source (#55), and does NOT size engine buffers
// (#5's allocator reads `.alloc`, it isn't produced here) -- see
// cell_lowering.cpp for the per-tag mapping and the minimal LEMS-expression
// lowering it implements.
//
// Bake-vs-parameterize (arch §3.1 `Parameter`): NOT re-decided here. Ticket
// #7's `ModelSpecification` already always bakes a bound instance's Parameter
// values (`TypeLibraryEntry::baked_constants`) for Phase 1 (one population
// binds one component -> every neuron sharing that entry has the same
// resolved value) -- this lowering just reads that map and emits a literal
// `ParamConstantDirective` per `Parameter`. The one exception (ticket #65
// [F4], Phase 2): a `Parameter` name also present in
// `TypeLibraryEntry::heterogeneous_parameter_values` emits a per-neuron
// `param : dyn` array (`ParamDynamicDirective`) instead -- see that field's
// own doc comment (model_specification.h) for how genuine heterogeneity gets
// into that map.

// Lowers one Cell-category `TypeLibraryEntry` to its `IrProgram`. Throws
// std::runtime_error if `cell_entry.category != TypeLibraryCategory::Cell`,
// if a Dynamics expression references an identifier the type never declares
// (a fixture/model authoring error), or if a `Transition`/`OnCondition`
// names a `Regime` the type never declares.
IrProgram lower_cell_to_ir(const TypeLibraryEntry &cell_entry);

// Convenience driver: lowers every Cell-category entry in `model`'s type
// library (skipping Synapse/Inputs entries -- ticket #51's and a future
// ticket's job respectively), in type-library order.
Vector<IrProgram> lower_all_cell_types_to_ir(const ModelSpecification &model);

// ── Regime metadata + active-set x nonlinear-dynamics classification (ticket #62 [F1]; arch §0.5)
// -- declared here (a cleanup of ir.h's own former REFACTOR comment) so
// cell_dynamics_are_closed_form_advanceable is real, directly-callable public API instead of only
// reachable by reading a stored tag back off a constructed IrProgram. ────────────────────────────

// One `TimeDerivative variable="..." value="..."` inside a `<Regime>` body (RegimeDecl::body, arch
// §3.2) -- part of gather_regime_info's own per-regime metadata below.
struct RegimeTimeDerivative {
    String variable_name;
    String value_text;
};

// One `<OnCondition test="...">` inside a `<Regime>` body -- `body_node` is the raw node
// (cell_lowering.cpp's own lower_on_condition_actions walks its StateAssignment/EventOut/Transition
// children, not reproduced here).
struct RegimeOnCondition {
    String test_text;
    const NML_Node *body_node = nullptr;
};

// One `<Regime>`'s gathered metadata (arch §3.2 Regime/Transition/OnEntry; IR spec §4's
// refractory-regime example), as built by gather_regime_info below. The `initial="true"` regime is
// always index 0; every other regime keeps its declaration order after it.
struct RegimeInfo {
    String name;
    s32 index = 0;
    Vector<RegimeTimeDerivative> time_derivatives;
    Vector<RegimeOnCondition> on_conditions;
    Vector<const NML_Node *> on_entry_state_assignments; // flattened <StateAssignment> children of every <OnEntry>
};

// Gathers `cell`'s own `<Regime>` declarations into RegimeInfo metadata, exactly the way
// lower_cell_to_ir gathers them for its own internal use. Exposed so a caller that already has the
// same `cell` lower_cell_to_ir would lower can recompute the SAME `Vector<RegimeInfo>`
// cell_dynamics_are_closed_form_advanceable below expects, without re-deriving this logic itself.
Vector<RegimeInfo> gather_regime_info(const CellType &cell);

// Whether `cell`'s Dynamics (its own top-level TimeDerivatives plus every one of `regimes`'s own) are
// safe for the engine's active-set closed-form multi-tick lazy decay -- true
// ("closed_form_advanceable") iff every TimeDerivative is affine in the cell's own state variables
// (all of GLIF); false ("nonlinear") means the type must integrate exactly one dt per active tick,
// never skipped/fast-forwarded (arch §5 Phase 2's Active-set x nonlinear rule). `regimes` is normally
// `gather_regime_info(cell)`'s own output -- passed in rather than recomputed so this stays a pure
// structural check over data the caller already has (lower_cell_to_ir passes in its own
// already-gathered regimes).
bool cell_dynamics_are_closed_form_advanceable(const CellType &cell, const Vector<RegimeInfo> &regimes);

} // namespace spikecorec::nml
