#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cstring>
#include <unordered_set>
#include <utility>

#include "spikecorec/nml/allocator.h"
#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::nml {

namespace {

// Phase-1's cell/synapse lowering (#50/#51) only ever declares f32
// state/accum/param:dyn slots; anything else is out of Phase-1 scope.
void require_f32_dtype(const String &dtype, const String &context) {
    if (dtype != "f32") {
        log::throw_runtime_error(log::logger(),
            "allocator: " + context + " declares dtype '" + dtype + "' -- Phase 1 only supports "
            "f32 (the only dtype the current cell/synapse lowering ever emits)");
    }
}

GpuPointer<f32> allocate_zeroed_f32(s64 element_count) {
    GpuPointer<f32> buffer = allocate<f32>((usize)element_count * sizeof(f32));
    if (element_count > 0) memset(buffer.get_contents(), 0, (usize)element_count * sizeof(f32));
    return buffer;
}

} // namespace

String type_scoped_key(s32 type_library_index, const String &name) {
    return std::to_string(type_library_index) + ":" + name;
}

ModelAllocation::~ModelAllocation() {
    deallocate(std::move(cell_state));
    deallocate(std::move(cell_type_boundaries));
    deallocate(std::move(regime_indices));
    for (auto &entry : accumulators) deallocate(std::move(entry.second));
    for (auto &entry : dynamic_parameter_arrays) deallocate(std::move(entry.second));
    for (auto &entry : derived_exposure_scratch_buffers) deallocate(std::move(entry.second));
}

ModelAllocation allocate_model(ModelSpecification &model, const Vector<IrProgram> &type_library_ir_programs) {
    if (type_library_ir_programs.size() != model.type_library.size()) {
        log::throw_runtime_error(log::logger(),
            "allocator: type_library_ir_programs.size() (" + std::to_string(type_library_ir_programs.size()) +
            ") must match model.type_library.size() (" + std::to_string(model.type_library.size()) + ")");
    }

    ModelAllocation allocation;

    // ── population(s) owning each type-library entry (ticket #65 [F4]) -- needed below to place a
    // heterogeneous `param : dyn` array's real per-neuron values at the right offset within the
    // model-wide `dynamic_parameter_arrays` buffer (that buffer stays whole-model-sized per this
    // header's own existing convention; only the owning population's own neuron range is filled). ──
    UnorderedMap<s32, Vector<const PopulationEntry *>> populations_by_type_library_index;
    for (const auto &population : model.populations) {
        populations_by_type_library_index[population.type_library_index].push_back(&population);
    }

    // ── widened cell-state vector + O(P) boundary array (arch §4.1) ────────
    // Walk populations in order, summing each population's `size * V_t`
    // (`V_t` read from its cell type's own `.alloc` `state` directive count,
    // per this ticket's own charter -- not re-read from
    // `PopulationEntry::state_vector_chunk_base_offset`, which ticket #7
    // already computes the same way from the flattened ComponentType
    // directly, not the IR).
    Vector<s64> boundary_offsets_host(model.populations.size(), 0);
    s64 running_chunk_offset = 0;
    for (usize population_index = 0; population_index < model.populations.size(); ++population_index) {
        const PopulationEntry &population = model.populations[population_index];
        const IrProgram &cell_program = type_library_ir_programs[(usize)population.type_library_index];

        s32 state_variable_count = 0;
        for (const auto &directive : cell_program.alloc) {
            if (!std::holds_alternative<StateDirective>(directive)) continue;
            const auto &state_directive = std::get<StateDirective>(directive);
            require_f32_dtype(state_directive.dtype,
                "cell type '" + cell_program.component_type_name + "' state '" + state_directive.name + "'");
            ++state_variable_count;
        }

        boundary_offsets_host[population_index] = running_chunk_offset;
        running_chunk_offset += (s64)population.size * state_variable_count;
    }

    allocation.cell_state_element_count = running_chunk_offset;
    allocation.cell_state = allocate_zeroed_f32(running_chunk_offset);

    allocation.population_count = (s64)model.populations.size();
    allocation.cell_type_boundaries = allocate<s64>(boundary_offsets_host.size() * sizeof(s64));
    if (!boundary_offsets_host.empty()) {
        memcpy(allocation.cell_type_boundaries.get_contents(), boundary_offsets_host.data(),
               boundary_offsets_host.size() * sizeof(s64));
    }

    // ── walk every type-in-use once for accum/peredge/regime/param/require/
    // expose (arch §4.2, §4.3, §4.5; IR spec §2) ────────────────────────────
    s64 total_peredge_variable_count = 0;

    for (s32 type_library_index = 0; type_library_index < (s32)model.type_library.size(); ++type_library_index) {
        const TypeLibraryEntry &type_entry = model.type_library[(usize)type_library_index];
        const IrProgram &program = type_library_ir_programs[(usize)type_library_index];

        // Names that already have their own per-neuron storage (a `state` or
        // `accum` slot) -- an `expose` directive naming one of these needs no
        // scratch buffer (arch §4.1/§4.5).
        std::unordered_set<String> per_neuron_storage_names;

        for (const auto &directive : program.alloc) {
            if (std::holds_alternative<StateDirective>(directive)) {
                per_neuron_storage_names.insert(std::get<StateDirective>(directive).name);

            } else if (std::holds_alternative<AccumDirective>(directive)) {
                const auto &accum_directive = std::get<AccumDirective>(directive);
                require_f32_dtype(accum_directive.dtype,
                    "type '" + program.component_type_name + "' accum '" + accum_directive.name + "'");
                per_neuron_storage_names.insert(accum_directive.name);
                allocation.accumulators.emplace(
                    type_scoped_key(type_library_index, accum_directive.name),
                    allocate_zeroed_f32(model.total_neuron_count));

            } else if (std::holds_alternative<PeredgeDirective>(directive)) {
                ++total_peredge_variable_count;

            } else if (std::holds_alternative<RegimeDirective>(directive)) {
                allocation.has_regime_index = true;

            } else if (std::holds_alternative<ParamConstantDirective>(directive)) {
                const auto &param_directive = std::get<ParamConstantDirective>(directive);
                if (!param_directive.literal_value.has_value()) {
                    log::logger().warn(
                        "allocator: type '{}' param '{}' has no baked literal_value in its .alloc -- not "
                        "ready for codegen to inline (Phase-1's bake-vs-parameterize rule always bakes a "
                        "bound instance's Parameter value, so this indicates an upstream gap)",
                        program.component_type_name, param_directive.name);
                    continue;
                }
                auto baked_value = type_entry.baked_constants.find(param_directive.name);
                if (baked_value == type_entry.baked_constants.end()) {
                    log::throw_runtime_error(log::logger(),
                        "allocator: type '" + program.component_type_name + "' param '" + param_directive.name +
                        "' has a baked .alloc literal but no matching TypeLibraryEntry::baked_constants value");
                }
                allocation.baked_param_constants[type_library_index][param_directive.name] = baked_value->second;

            } else if (std::holds_alternative<ParamDynamicDirective>(directive)) {
                const auto &param_directive = std::get<ParamDynamicDirective>(directive);
                require_f32_dtype(param_directive.dtype,
                    "type '" + program.component_type_name + "' param:dyn '" + param_directive.name + "'");
                GpuPointer<f32> buffer = allocate_zeroed_f32(model.total_neuron_count);

                // Fill from TypeLibraryEntry::heterogeneous_parameter_values when the lowering that
                // produced this `param : dyn` directive was driven by genuine per-neuron
                // heterogeneity (ticket #65 [F4]) -- left zeroed (with a warning) otherwise, since a
                // `param : dyn` directive with no known per-neuron source has no value to fill it
                // with (e.g. a hand-built IrProgram fixture in a test).
                auto heterogeneous_values = type_entry.heterogeneous_parameter_values.find(param_directive.name);
                if (heterogeneous_values != type_entry.heterogeneous_parameter_values.end()) {
                    const Vector<f64> &values = heterogeneous_values->second;
                    auto owning_populations = populations_by_type_library_index.find(type_library_index);
                    if (owning_populations == populations_by_type_library_index.end()) {
                        log::throw_runtime_error(log::logger(),
                            "allocator: type '" + program.component_type_name + "' has heterogeneous values for "
                            "param:dyn '" + param_directive.name + "' but no population binds it");
                    }
                    for (const PopulationEntry *population : owning_populations->second) {
                        if ((s32)values.size() != population->size) {
                            log::throw_runtime_error(log::logger(),
                                "allocator: type '" + program.component_type_name + "' param:dyn '" +
                                param_directive.name + "' has " + std::to_string(values.size()) +
                                " heterogeneous value(s), but population '" + population->id + "' binding it has "
                                "size " + std::to_string(population->size));
                        }
                        f32 *destination = buffer.get_contents() + population->neuron_index_begin;
                        for (s32 local_index = 0; local_index < population->size; ++local_index) {
                            destination[local_index] = (f32)values[(usize)local_index];
                        }
                    }
                } else {
                    log::logger().warn(
                        "allocator: type '{}' param:dyn '{}' has no matching "
                        "TypeLibraryEntry::heterogeneous_parameter_values -- left zero-filled",
                        program.component_type_name, param_directive.name);
                }

                allocation.dynamic_parameter_arrays.emplace(
                    type_scoped_key(type_library_index, param_directive.name), std::move(buffer));

            } else if (std::holds_alternative<RequireDirective>(directive)) {
                const auto &require_directive = std::get<RequireDirective>(directive);
                allocation.require_bindings.push_back(
                    RequireBinding{type_library_index, require_directive.name, require_directive.scope});
            }
            // ExposeDirective is handled below, once per_neuron_storage_names
            // is fully populated for this type.
        }

        if (type_entry.category != TypeLibraryCategory::Cell) continue; // scratch is Cell-only, see allocator.h
        for (const auto &directive : program.alloc) {
            if (!std::holds_alternative<ExposeDirective>(directive)) continue;
            const auto &expose_directive = std::get<ExposeDirective>(directive);
            if (per_neuron_storage_names.count(expose_directive.name)) continue; // already has a slot
            allocation.derived_exposure_scratch_buffers.emplace(
                type_scoped_key(type_library_index, expose_directive.name),
                allocate_zeroed_f32(model.total_neuron_count));
        }
    }

    if (allocation.has_regime_index) {
        allocation.regime_indices = allocate<s32>((usize)model.total_neuron_count * sizeof(s32));
        memset(allocation.regime_indices.get_contents(), 0, (usize)model.total_neuron_count * sizeof(s32));
    }

    // ── WeightMatrix per-edge variable count (arch §4.3) ────────────────────
    allocation.per_edge_variable_count = total_peredge_variable_count;
    if (model.adjacency.has_value()) {
        model.adjacency->configure_per_edge_variable_count(total_peredge_variable_count);
    }

    return allocation;
}

} // namespace spikecorec::nml
