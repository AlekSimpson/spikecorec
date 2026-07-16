#pragma once

#include "spikecorec/core/backend.h"
#include "spikecorec/nml/ir.h"
#include "spikecorec/nml/model_specification.h"

namespace spikecorec::nml {

// ── `.alloc` interpretation -> allocator (ticket #5 [C2]; arch §4.1, §4.2,
// §4.5; IR spec §2) ───────────────────────────────────────────────────────
//
// Interprets every type-in-use's `.alloc` (ticket #4's IrProgram, lowered by
// ticket #50/#51 from ticket #7's ModelSpecification) to size and allocate
// the engine's model-specific `GpuPointer` buffers -- so allocation is
// model-driven, not hardcoded. This module does NOT wire the allocated
// buffers into SpikeEngine's per-tick step loop (that is master-kernel
// assembly, ticket #6 [C3], which depends on this ticket) and does NOT seed
// real per-edge WeightMatrix data (that is tickets #52-54/#57) -- it only
// computes layout and allocates.
//
// Lives in `nml/`, not `core/`: this module's INPUT is entirely NML-derived
// (ModelSpecification, IrProgram), and the existing module boundary already
// has `nml/` depend on `core/` (model_specification.cpp already constructs a
// real `core::WeightMatrix` -- a live GpuPointer-backed allocation -- from a
// ResolvedModel; ir.h/ir.cpp likewise live in `nml/` despite being an
// engine-facing ISA). Putting a second "core depends on nml" edge in for
// just this one file would make the module graph cyclic at the package
// level for no benefit; extending nml's existing "produces real engine
// buffers from model data" precedent keeps the dependency graph a DAG
// (nml -> core only).
//
// `type_library_ir_programs` is a parallel array to `model.type_library`,
// one `IrProgram` per entry, in the same index order -- e.g. built by
// calling `lower_cell_to_ir`/`lower_synapse_to_ir` per entry depending on
// its `TypeLibraryCategory` (see this ticket's tests for exactly that). A
// `TypeLibraryCategory::Inputs` entry has no IR-lowering ticket yet (Phase-1
// stimulus generators are host-precomputed, arch §3.3 D4/§5) -- callers
// should pass an empty placeholder `IrProgram` for those slots; an empty
// `.alloc` correctly contributes nothing here.

// One `require <name> from <scope>` binding, recorded as metadata only (IR
// spec §2: "a binding; storage lives with the owner of the quantity" -- no
// allocation). `type_library_index` identifies which type-in-use declared it.
struct RequireBinding {
    s32 type_library_index = -1;
    String name;
    String scope;
};

// The result of interpreting every type-in-use's `.alloc` for one model
// (ticket #5 [C2]). Owns every model-specific `GpuPointer` this ticket
// allocates -- move-only (`GpuPointer<T>` has no copy constructor), each
// buffer deallocated in the destructor, the same ownership pattern
// `WeightMatrix`/`SpikeEngine` already use.
class ModelAllocation {
public:
    // ── widened cell-state vector, SoA within per-population chunks (arch
    // §4.1) -- `cell_state_element_count` f32 elements total, summed across
    // every population's `size * V_t` (`V_t` = its cell type's `state`
    // directive count). `cell_type_boundaries` is the O(P) boundary array:
    // one element per population, in population order, holding that
    // population's chunk base offset (in elements) into `cell_state`. ──
    GpuPointer<f32> cell_state;
    s64 cell_state_element_count = 0;
    GpuPointer<s64> cell_type_boundaries;
    s64 population_count = 0;

    // ── aggregated per-neuron synapse accumulators (arch §4.2) -- one
    // `GpuPointer<f32>[total_neuron_count]` per (type-in-use, `accum`
    // directive name) pair, keyed by `type_scoped_key()` below. Sized to the
    // model's whole `total_neuron_count` (not just a projection's
    // postsynaptic population), matching the existing engine-owned
    // `network_inputs`'s own whole-model sizing -- simplest safe upper bound,
    // since any neuron could in principle be a target of that synapse type. ──
    UnorderedMap<String, GpuPointer<f32>> accumulators;

    // ── regime index (arch §4.5) -- ONE shared `GpuPointer<s32>[total_neuron_count]`
    // array, present iff at least one type-in-use declares a `regime`
    // directive; a single combined array (not one per type), matching how
    // `last_spiked`/`last_tick_updated`/`active_generation` are already one
    // shared per-neuron array each, interpreted according to whichever cell
    // type a given neuron belongs to. ──
    GpuPointer<s32> regime_indices;
    bool has_regime_index = false;

    // ── `param <name> : dyn <dtype>` per-neuron dynamic parameter arrays --
    // never actually emitted by the current Phase-1 cell/synapse lowering
    // (which always bakes), but part of the locked IR spec's `.alloc`
    // surface (§2), so handled here for whatever future lowering emits it.
    // Keyed by `type_scoped_key()`, sized to `total_neuron_count`. ──
    UnorderedMap<String, GpuPointer<f32>> dynamic_parameter_arrays;

    // ── recorded-derived-exposure scratch (arch §4.1/§4.5: "a recorded
    // derived exposure may need a scratch per-neuron slot") -- allocated
    // only for a Cell-category type's `expose` directive whose name has no
    // matching `state` directive in that same type's `.alloc` (a directly
    // stored StateVariable's own exposure already has a slot and needs no
    // scratch). Restricted to Cell-category entries: Phase-1 recording
    // (`RecordingEntry`, ticket #7) only ever resolves against a population,
    // never a synapse/projection, so a synapse-type `expose` has no
    // per-neuron recording target to scratch for in this ticket's scope.
    // Keyed by `type_scoped_key()`, sized to `total_neuron_count`. ──
    UnorderedMap<String, GpuPointer<f32>> derived_exposure_scratch_buffers;

    // ── baked `param` constants, collected + validated as ready for #55's
    // codegen to inline (no allocation -- a baked constant is a literal in
    // generated code, arch §3.1). Outer key: `type_library_index`; inner
    // map: param name -> its baked value (read from
    // `TypeLibraryEntry::baked_constants`, cross-checked against the
    // `.alloc` `ParamConstantDirective`'s own baked literal). A bare
    // (unbaked) `ParamConstantDirective` -- which Phase-1's own
    // always-bakes rule should never actually produce -- is logged as a
    // warning, not included here, and not fatal. ──
    UnorderedMap<s32, UnorderedMap<String, f64>> baked_param_constants;

    // ── `require <name> from <scope>` bindings, recorded metadata only (no
    // allocation, per IR spec §2). ──
    Vector<RequireBinding> require_bindings;

    // ── WeightMatrix per-edge variable count (arch §4.3) -- the total
    // `peredge` directive count summed across every per-edge synapse
    // type-in-use; also written into `model.adjacency`'s own
    // `per_edge_variable_count` (see `allocate_model`'s `model` parameter)
    // when the model has any adjacency at all. Only the COUNT is set here --
    // seeding real per-edge `Ck`/`Sk` data is tickets #52-54/#57. ──
    s64 per_edge_variable_count = 0;

    ModelAllocation() = default;
    ModelAllocation(const ModelAllocation &) = delete;
    ModelAllocation &operator=(const ModelAllocation &) = delete;
    ModelAllocation(ModelAllocation &&) = default;
    ModelAllocation &operator=(ModelAllocation &&) = default;
    ~ModelAllocation();
};

// Builds the key `accumulators`/`dynamic_parameter_arrays`/
// `derived_exposure_scratch_buffers` are keyed by: one type-in-use's
// (`type_library_index`) own directive `name`, disambiguating two different
// types that happen to declare a same-named quantity (e.g. two synapse
// types both naming their decaying conductance `g`).
String type_scoped_key(s32 type_library_index, const String &name);

// Interprets every entry in `model.type_library`'s corresponding
// `type_library_ir_programs[i].alloc` (parallel arrays, same index order)
// and allocates the model-specific `GpuPointer` buffers described above.
// Also configures `model.adjacency`'s per-edge variable count in place when
// the model has adjacency (arch §4.3) -- hence `model` is taken by mutable
// reference, not const. Throws std::runtime_error if
// `type_library_ir_programs.size() != model.type_library.size()`, if a
// `state`/`accum`/`param:dyn` directive declares a dtype other than `f32`
// (the only dtype Phase-1's cell/synapse lowering ever emits), or if a baked
// `ParamConstantDirective` names a param absent from its
// `TypeLibraryEntry::baked_constants` (a malformed IrProgram/ModelSpecification
// pairing).
ModelAllocation allocate_model(ModelSpecification &model, const Vector<IrProgram> &type_library_ir_programs);

} // namespace spikecorec::nml
