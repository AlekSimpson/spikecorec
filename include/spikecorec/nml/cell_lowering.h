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

} // namespace spikecorec::nml
