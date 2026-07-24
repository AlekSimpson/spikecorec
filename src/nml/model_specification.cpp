#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cmath>

#include "spikecorec/nml/model_specification.h"
#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::nml {

namespace {

// ── Small local helpers ──────────────────────────────────────────────────

bool contains(const Vector<String> &values, const String &target) {
    for (const auto &value : values) {
        if (value == target) return true;
    }
    return false;
}

bool chain_contains(const Vector<String> &ancestor_chain, const String &target) {
    return contains(ancestor_chain, target);
}

TypeLibraryCategory category_of(const ComponentTypeEntry &entry) {
    if (std::holds_alternative<CellType>(entry)) return TypeLibraryCategory::Cell;
    if (std::holds_alternative<SynapseType>(entry)) return TypeLibraryCategory::Synapse;
    if (std::holds_alternative<InputsType>(entry)) return TypeLibraryCategory::Inputs;
    log::throw_runtime_error(log::logger(),
        "model_specification: a bound component's type is not a Dynamics ComponentType "
        "(not CellType/SynapseType/InputsType)");
}

s32 state_variable_count_of(const ComponentTypeEntry &entry) {
    if (std::holds_alternative<CellType>(entry)) return (s32)std::get<CellType>(entry).state_variables.size();
    if (std::holds_alternative<SynapseType>(entry)) return (s32)std::get<SynapseType>(entry).state_variables.size();
    if (std::holds_alternative<InputsType>(entry)) return (s32)std::get<InputsType>(entry).state_variables.size();
    return 0;
}

bool synapse_is_conductance_based(const Vector<String> &ancestor_chain) {
    return chain_contains(ancestor_chain, "baseConductanceBasedSynapse") ||
           chain_contains(ancestor_chain, "baseConductanceBasedSynapseTwo");
}

// A resolved "PopName[index]/trailing" or "PopName/index/trailing" path
// (arch §3.1 `Path`/`Text`; connection preCellId/postCellId, explicitInput
// target, OutputColumn/Line quantity, EventSelection select all use one of
// these two forms -- see https://docs.neuroml.org/Userdocs/Paths.html). A
// single leading "../" (a sibling-scope reference, the common form for a
// `connection` inside a `projection` reaching a sibling `population`) is
// stripped first; nothing else about relative-path depth is modeled here,
// which is enough for the single-network-scope models Phase 1 targets.
//
// Both forms REQUIRE the index: in the slash form specifically, the segment
// immediately after the population name must be a well-formed non-negative
// integer -- there is no supported bare "PopName/exposureName" shorthand
// (real NeuroML quantity/select/cell paths always name one specific neuron),
// so a non-numeric segment there (e.g. "ExcPop/v") is rejected as malformed
// input rather than silently misparsed as index 0 with the exposure dropped.
struct ParsedPopulationPath {
    String population_name;
    s32 local_index = -1;
    String trailing; // component id (cell paths) or exposure/port name (recording paths); empty if absent
};

// Validates that `index_text` is a non-empty string of decimal digits (a
// well-formed non-negative integer) before converting it, throwing a clear
// error otherwise. `std::atoi` alone silently yields 0 on a malformed or
// empty segment (e.g. "ExcPop[abc]", a truncated "ExcPop[", or "ExcPop/v"'s
// non-numeric segment) -- that would produce a wrong-but-silent index
// instead of a diagnostic, the same class of bug `looks_like_dimensioned_literal`
// / `unit_value_to_si` guard against elsewhere in the NML front-end.
s32 parse_required_index(const String &index_text, const String &raw_path) {
    bool well_formed = !index_text.empty() &&
        std::all_of(index_text.begin(), index_text.end(),
                     [](unsigned char character) { return std::isdigit(character) != 0; });
    if (!well_formed) {
        log::throw_runtime_error(log::logger(),
            "model_specification: path '" + raw_path + "' has a malformed index segment '" + index_text + "'");
    }
    return std::atoi(index_text.c_str());
}

ParsedPopulationPath parse_population_path(const String &raw_path) {
    ParsedPopulationPath parsed;
    String path = raw_path;
    if (path.rfind("../", 0) == 0) path = path.substr(3);

    auto bracket_open = path.find('[');
    if (bracket_open != String::npos) {
        auto bracket_close = path.find(']', bracket_open);
        if (bracket_close == String::npos) {
            log::throw_runtime_error(log::logger(),
                "model_specification: path '" + raw_path + "' has an unterminated '[' index");
        }
        parsed.population_name = path.substr(0, bracket_open);
        String index_text = path.substr(bracket_open + 1, bracket_close - bracket_open - 1);
        parsed.local_index = parse_required_index(index_text, raw_path);
        if (bracket_close + 1 < path.size() && path[bracket_close + 1] == '/') {
            parsed.trailing = path.substr(bracket_close + 2);
        }
        return parsed;
    }

    auto first_slash = path.find('/');
    if (first_slash == String::npos) {
        parsed.population_name = path;
        return parsed;
    }
    parsed.population_name = path.substr(0, first_slash);

    auto second_slash = path.find('/', first_slash + 1);
    String index_segment = second_slash == String::npos
        ? path.substr(first_slash + 1)
        : path.substr(first_slash + 1, second_slash - first_slash - 1);
    parsed.local_index = parse_required_index(index_segment, raw_path);
    if (second_slash != String::npos) parsed.trailing = path.substr(second_slash + 1);

    return parsed;
}

// Resolves `local_index` within `population` to a global neuron index,
// bounds-checked against the population's actual size. resolve only wires
// IDrefs (S7); it never range-checks a path's numeric index, so nothing
// upstream catches a preCellId/postCellId/target naming an in-range
// population but an out-of-range neuron. Left unchecked, that index feeds
// `adjacency_list[...]` directly (a raw `std::vector::operator[]`) -- a heap
// out-of-bounds write, not just a logic bug -- so this is the single place
// every "population + local index -> global neuron index" site in this file
// routes through.
s32 resolve_required_neuron_index(const PopulationEntry &population, s32 local_index, const String &context) {
    if (local_index < 0 || local_index >= population.size) {
        log::throw_runtime_error(log::logger(),
            "model_specification: " + context + " names index " + std::to_string(local_index) +
            " in population '" + population.id + "' (size " + std::to_string(population.size) + ")");
    }
    return population.neuron_index_begin + local_index;
}

void index_instances_by_symbol(const ResolvedInstance &instance, UnorderedMap<s32, const ResolvedInstance *> &by_symbol) {
    if (instance.symbol_index >= 0) by_symbol[instance.symbol_index] = &instance;
    for (const auto &child : instance.children) {
        index_instances_by_symbol(child, by_symbol);
    }
}

// Recursively collects every instance whose tag matches one of `tags`, in
// document order (a single pass -- collecting each tag with a separate pass
// and concatenating would lose true document order whenever two matched
// tags are interleaved, e.g. a mix of `population` and `populationList`).
void collect_by_tag(const Vector<ResolvedInstance> &instances, const Vector<String> &tags, Vector<const ResolvedInstance *> &matches) {
    for (const auto &instance : instances) {
        if (contains(tags, instance.tag_name)) matches.push_back(&instance);
        collect_by_tag(instance.children, tags, matches);
    }
}

f64 numeric_attribute_or(const ResolvedInstance &instance, const String &name, f64 fallback) {
    auto entry = instance.numeric_attributes.find(name);
    return entry == instance.numeric_attributes.end() ? fallback : entry->second;
}

// Builds (or reuses, if this exact bound instance was already materialized)
// a TypeLibraryEntry for the component instance registered under
// `bound_instance_symbol_index`, appending it to `type_library` and caching
// the result in `type_library_index_by_symbol` (arch §1.4's "cell-type and
// synapse-type library").
s32 get_or_create_type_library_entry(
    s32 bound_instance_symbol_index,
    const ResolvedModel &resolved,
    const UnorderedMap<s32, const ResolvedInstance *> &instances_by_symbol,
    UnorderedMap<s32, s32> &type_library_index_by_symbol,
    Vector<TypeLibraryEntry> &type_library
) {
    auto cached = type_library_index_by_symbol.find(bound_instance_symbol_index);
    if (cached != type_library_index_by_symbol.end()) return cached->second;

    auto instance_iterator = instances_by_symbol.find(bound_instance_symbol_index);
    if (instance_iterator == instances_by_symbol.end()) {
        log::throw_runtime_error(log::logger(),
            "model_specification: a bound component instance's symbol index has no matching instance");
    }
    const ResolvedInstance &bound_instance = *instance_iterator->second;

    // Two distinct failure shapes, given separate messages rather than one collapsed "not a
    // cataloged Dynamics ComponentType": an uncataloged name is a typo/missing-<include>, while a
    // cataloged-but-Structure-bucket name is either a population/projection type used where a
    // Dynamics type is required, or a real ComponentType category this pipeline doesn't classify
    // into Cell/Synapse/Inputs at all (ion channels, morphology, biophysical properties -- Phase 3,
    // not yet supported) -- see NML_Parser::classify_component_type, which leaves exactly those
    // reaching none of its anchors as the bare (Structure-bucketed) ComponentTypeBase.
    auto type_iterator = resolved.types.find(bound_instance.tag_name);
    if (type_iterator == resolved.types.end()) {
        log::throw_runtime_error(log::logger(),
            "model_specification: bound component '" + bound_instance.id + "' names type '" +
            bound_instance.tag_name + "', which is not a cataloged ComponentType at all (check for a "
            "typo, or a missing <include> of the file that declares it)");
    }
    if (type_iterator->second.bucket != ComponentTypeBucket::Dynamics) {
        log::throw_runtime_error(log::logger(),
            "model_specification: bound component '" + bound_instance.id + "' is of type '" +
            bound_instance.tag_name + "', which is cataloged but not as a Cell/Synapse/Inputs "
            "Dynamics ComponentType -- either it is a population/projection (Structure) type "
            "referenced where a Dynamics type is required, or it is a ComponentType category this "
            "pipeline does not yet classify (ion channels, morphology, biophysical properties -- "
            "Phase 3, not yet supported)");
    }
    const ResolvedComponentType &type = type_iterator->second;

    TypeLibraryEntry entry;
    entry.component_type_name = bound_instance.tag_name;
    entry.bound_instance_id = bound_instance.id;
    entry.category = category_of(type.flattened);
    entry.dynamics = type;
    entry.state_variable_count = state_variable_count_of(type.flattened);

    entry.baked_constants = type.fixed_parameter_values;
    for (const auto &[name, value] : bound_instance.numeric_attributes) {
        entry.baked_constants[name] = value; // the bound instance's own value wins over an inherited Fixed pin
    }

    if (entry.category == TypeLibraryCategory::Synapse) {
        entry.is_conductance_based = synapse_is_conductance_based(type.ancestor_chain);
    }

    s32 new_index = (s32)type_library.size();
    type_library.push_back(std::move(entry));
    type_library_index_by_symbol[bound_instance_symbol_index] = new_index;
    return new_index;
}

} // namespace

ModelSpecification build_model_specification(const ResolvedModel &resolved, s64 weight_matrix_rank) {
    ModelSpecification specification;

    // 1. symbol_index -> ResolvedInstance* over the whole instance tree, so
    // an IDref (population `component`, projection `synapse`, explicitInput
    // `input`, ...) can be followed to the instance it names.
    UnorderedMap<s32, const ResolvedInstance *> instances_by_symbol;
    for (const auto &instance : resolved.instances) {
        index_instances_by_symbol(instance, instances_by_symbol);
    }
    UnorderedMap<s32, s32> type_library_index_by_symbol;

    // 2. Populations (arch §1.4/§4.1): walk `population`/`populationList`
    // instances, in document order, assigning contiguous neuron-index ranges
    // and state-vector chunk offsets as they're encountered.
    Vector<const ResolvedInstance *> population_instances;
    collect_by_tag(resolved.instances, {"population", "populationList"}, population_instances);

    UnorderedMap<String, s32> population_index_by_name;
    UnorderedMap<s32, s32> population_index_by_symbol;
    s64 running_chunk_offset = 0;

    for (const auto *population_instance : population_instances) {
        auto component_idref = population_instance->idref_attributes.find("component");
        if (component_idref == population_instance->idref_attributes.end()) {
            log::throw_runtime_error(log::logger(),
                "model_specification: population '" + population_instance->id + "' has no `component` IDref");
        }

        s32 type_library_index = get_or_create_type_library_entry(
            component_idref->second, resolved, instances_by_symbol, type_library_index_by_symbol, specification.type_library);

        PopulationEntry population;
        population.id = population_instance->id;
        population.type_library_index = type_library_index;

        if (population_instance->tag_name == "populationList") {
            s32 instance_count = 0;
            for (const auto &child : population_instance->children) {
                if (child.tag_name == "instance") ++instance_count;
            }
            population.size = instance_count;
        } else {
            population.size = (s32)std::llround(numeric_attribute_or(*population_instance, "size", 0.0));
        }

        population.neuron_index_begin = specification.total_neuron_count;
        population.neuron_index_end = population.neuron_index_begin + population.size;
        population.state_vector_chunk_base_offset = running_chunk_offset;

        specification.total_neuron_count += population.size;
        running_chunk_offset += (s64)population.size * specification.type_library[type_library_index].state_variable_count;

        s32 population_index = (s32)specification.populations.size();
        population_index_by_name[population.id] = population_index;
        if (population_instance->symbol_index >= 0) population_index_by_symbol[population_instance->symbol_index] = population_index;
        specification.populations.push_back(std::move(population));
    }

    // 3. Projection/synapse routing + delay data (arch §1.4, §3.3 ST3): walk
    // `projection` instances and their nested `connection`/`connectionWD`
    // children.
    Vector<const ResolvedInstance *> projection_instances;
    collect_by_tag(resolved.instances, {"projection"}, projection_instances);

    Vector<vector<s32>> adjacency_list(specification.total_neuron_count);

    for (const auto *projection_instance : projection_instances) {
        ProjectionEntry projection;
        projection.id = projection_instance->id;

        auto pre_idref = projection_instance->idref_attributes.find("presynapticPopulation");
        auto post_idref = projection_instance->idref_attributes.find("postsynapticPopulation");
        auto synapse_idref = projection_instance->idref_attributes.find("synapse");
        if (pre_idref == projection_instance->idref_attributes.end() ||
            post_idref == projection_instance->idref_attributes.end() ||
            synapse_idref == projection_instance->idref_attributes.end()) {
            log::throw_runtime_error(log::logger(),
                "model_specification: projection '" + projection_instance->id +
                "' is missing presynapticPopulation/postsynapticPopulation/synapse");
        }

        auto pre_population = population_index_by_symbol.find(pre_idref->second);
        auto post_population = population_index_by_symbol.find(post_idref->second);
        if (pre_population == population_index_by_symbol.end() || post_population == population_index_by_symbol.end()) {
            log::throw_runtime_error(log::logger(),
                "model_specification: projection '" + projection_instance->id +
                "' references a population that was never cataloged");
        }
        projection.presynaptic_population_index = pre_population->second;
        projection.postsynaptic_population_index = post_population->second;
        projection.synapse_type_library_index = get_or_create_type_library_entry(
            synapse_idref->second, resolved, instances_by_symbol, type_library_index_by_symbol, specification.type_library);

        for (const auto &connection_instance : projection_instance->children) {
            if (connection_instance.tag_name != "connection" && connection_instance.tag_name != "connectionWD") continue;

            auto pre_cell_id = connection_instance.string_attributes.find("preCellId");
            auto post_cell_id = connection_instance.string_attributes.find("postCellId");
            if (pre_cell_id == connection_instance.string_attributes.end() ||
                post_cell_id == connection_instance.string_attributes.end()) {
                log::throw_runtime_error(log::logger(),
                    "model_specification: connection in projection '" + projection_instance->id +
                    "' is missing preCellId/postCellId");
            }

            ParsedPopulationPath source_path = parse_population_path(pre_cell_id->second);
            ParsedPopulationPath target_path = parse_population_path(post_cell_id->second);

            auto source_population = population_index_by_name.find(source_path.population_name);
            auto target_population = population_index_by_name.find(target_path.population_name);
            if (source_population == population_index_by_name.end() || target_population == population_index_by_name.end()) {
                log::throw_runtime_error(log::logger(),
                    "model_specification: connection preCellId/postCellId names an uncataloged population "
                    "('" + source_path.population_name + "' / '" + target_path.population_name + "')");
            }

            ConnectionEntry connection;
            connection.source_neuron_index = resolve_required_neuron_index(
                specification.populations[source_population->second], source_path.local_index, "connection preCellId");
            connection.target_neuron_index = resolve_required_neuron_index(
                specification.populations[target_population->second], target_path.local_index, "connection postCellId");
            connection.weight = numeric_attribute_or(connection_instance, "weight", 1.0);
            connection.delay = numeric_attribute_or(connection_instance, "delay", 0.0);

            adjacency_list[(usize)connection.source_neuron_index].push_back(connection.target_neuron_index);
            projection.connections.push_back(connection);
        }

        specification.projections.push_back(std::move(projection));
    }

    bool has_any_edge = false;
    for (const auto &row : adjacency_list) {
        if (!row.empty()) { has_any_edge = true; break; }
    }
    if (has_any_edge) {
        specification.adjacency.emplace(adjacency_list, weight_matrix_rank);
    }

    // 4. Input (stimulus) specs (arch §1.4, §3.3 D4): walk `explicitInput`
    // instances.
    Vector<const ResolvedInstance *> explicit_input_instances;
    collect_by_tag(resolved.instances, {"explicitInput"}, explicit_input_instances);

    for (const auto *explicit_input : explicit_input_instances) {
        auto target = explicit_input->string_attributes.find("target");
        auto input_idref = explicit_input->idref_attributes.find("input");
        if (target == explicit_input->string_attributes.end() || input_idref == explicit_input->idref_attributes.end()) {
            log::throw_runtime_error(log::logger(), "model_specification: explicitInput is missing target/input");
        }

        ParsedPopulationPath target_path = parse_population_path(target->second);
        auto target_population = population_index_by_name.find(target_path.population_name);
        if (target_population == population_index_by_name.end()) {
            log::throw_runtime_error(log::logger(),
                "model_specification: explicitInput target names an uncataloged population '" +
                target_path.population_name + "'");
        }

        StimulusEntry stimulus;
        stimulus.target_neuron_index = resolve_required_neuron_index(
            specification.populations[target_population->second], target_path.local_index, "explicitInput target");
        stimulus.input_type_library_index = get_or_create_type_library_entry(
            input_idref->second, resolved, instances_by_symbol, type_library_index_by_symbol, specification.type_library);
        specification.stimuli.push_back(stimulus);
    }

    // 5. Output (recording) selections (arch §1.4, §4.6): state traces
    // (OutputColumn/Line) and spike records (EventSelection).
    auto add_recording = [&](const ResolvedInstance &recording_instance, const String &path_attribute_name,
                              const String &eventPort_attribute_name, bool is_event_recording) {
        RecordingEntry recording;
        recording.id = recording_instance.id;
        recording.is_event_recording = is_event_recording;

        auto path_attribute = recording_instance.string_attributes.find(path_attribute_name);
        if (path_attribute != recording_instance.string_attributes.end()) {
            recording.quantity_path = path_attribute->second;
            ParsedPopulationPath parsed_path = parse_population_path(recording.quantity_path);
            auto target_population = population_index_by_name.find(parsed_path.population_name);
            // Unlike a connection/stimulus target (bounds-checked and fatal
            // on failure, since those feed indices other code trusts as
            // valid), an unresolved or out-of-range recording target is kept
            // as the documented -1 diagnostic sentinel rather than thrown --
            // recordings are read-only/informational, not something engine
            // memory safety depends on. Left completely silent, though, that
            // sentinel is indistinguishable from a legitimately-unset field
            // to anyone not already reading this source file, so it is also
            // warned here (naming the recording id, its raw path, and why)
            // instead of being a purely silent drop.
            if (target_population != population_index_by_name.end()) {
                const PopulationEntry &population = specification.populations[target_population->second];
                if (parsed_path.local_index >= 0 && parsed_path.local_index < population.size) {
                    recording.target_neuron_index = population.neuron_index_begin + parsed_path.local_index;
                } else {
                    log::logger().warn(
                        "model_specification: recording '{}' path '{}' names index {} in population '{}' "
                        "(size {}) -- out of range, target_neuron_index left unresolved (-1)",
                        recording.id, recording.quantity_path, parsed_path.local_index,
                        parsed_path.population_name, population.size);
                }
            } else {
                log::logger().warn(
                    "model_specification: recording '{}' path '{}' names population '{}', which was "
                    "never cataloged -- target_neuron_index left unresolved (-1)",
                    recording.id, recording.quantity_path, parsed_path.population_name);
            }
            recording.exposure_name = parsed_path.trailing;
        }

        if (is_event_recording) {
            auto event_port = recording_instance.string_attributes.find(eventPort_attribute_name);
            if (event_port != recording_instance.string_attributes.end()) recording.exposure_name = event_port->second;
        }

        specification.recordings.push_back(std::move(recording));
    };

    Vector<const ResolvedInstance *> output_column_instances;
    collect_by_tag(resolved.instances, {"OutputColumn", "Line"}, output_column_instances);
    for (const auto *output_column : output_column_instances) {
        add_recording(*output_column, "quantity", "", false);
    }

    Vector<const ResolvedInstance *> event_selection_instances;
    collect_by_tag(resolved.instances, {"EventSelection"}, event_selection_instances);
    for (const auto *event_selection : event_selection_instances) {
        add_recording(*event_selection, "select", "eventPort", true);
    }

    return specification;
}

WeightMatrix build_weight_matrix_from_projections(const ModelSpecification &model) {
    Vector<Vector<s32>> adjacency((usize)model.total_neuron_count);
    for (const auto &projection : model.projections) {
        for (const auto &connection : projection.connections) {
            adjacency[(usize)connection.source_neuron_index].push_back(connection.target_neuron_index);
        }
    }
    return WeightMatrix(adjacency, /*rank=*/1);
}

} // namespace spikecorec::nml
