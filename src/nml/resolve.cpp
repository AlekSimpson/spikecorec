#include <stdexcept>
#include <variant>

#include "spikecorec/nml/resolve.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/time_utilities.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::nml {

namespace {

// ── Small local helpers (mirrors nml.cpp's own private helpers; not shared
// across translation units since nml.cpp keeps them in its own anonymous
// namespace) ──────────────────────────────────────────────────────────────

String get_attr(const NML_Node &node, const String &name) {
    auto entry = node.attributes.find(name);
    if (entry == node.attributes.end()) return "";
    return std::any_cast<String>(entry->second);
}

bool contains(const Vector<String> &values, const String &target) {
    for (const auto &value : values) {
        if (value == target) return true;
    }
    return false;
}

// Instance-level attributes that are IDrefs (arch §1.2 S7): they must
// resolve through the symbol table to another cataloged id/name, or it is a
// resolve error.
const Vector<String> IDREF_ATTRIBUTE_NAMES = {
    "component", "cell", "presynapticPopulation", "postsynapticPopulation",
    "synapse", "target", "ionChannel"};

// Instance-level attributes that are identity/metadata or structural paths
// (arch §1.2 S6 / §3.1 `Path`/`Text`) rather than dimensioned quantities --
// passed through as raw strings, never unit-converted.
const Vector<String> OPAQUE_STRING_ATTRIBUTE_NAMES = {
    "name", "notes", "preCellId", "postCellId", "destination", "type"};

// `Fixed` (§3.1) lives as a direct child of a ComponentType node, alongside
// `Parameter`/`Constant` -- not nested inside `<Dynamics>`.
Vector<FixedDecl> extract_fixed_decls(const NML_Node &node) {
    Vector<FixedDecl> result;
    for (const auto &child : node.body) {
        if (child.tag_name != "Fixed") continue;
        result.push_back(FixedDecl{get_attr(child, "parameter"), get_attr(child, "value")});
    }
    return result;
}

template <typename T>
void append_all(Vector<T> &destination, const Vector<T> &source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

// Applies every pending `Fixed` pin, converting its value to SI (S1) and
// recording it under the parameter's name. An unconvertible value is a
// resolve error, same as any other dimensioned literal.
void apply_fixed_pins(const Vector<FixedDecl> &pending_fixed, UnorderedMap<String, f64> &fixed_parameter_values) {
    for (const auto &fixed : pending_fixed) {
        try {
            fixed_parameter_values[fixed.parameter] = time::unit_value_to_si(fixed.value);
        } catch (const std::invalid_argument &conversion_error) {
            log::logger().error("resolve: Fixed parameter '{}' has an unconvertible value '{}': {}",
                                 fixed.parameter, fixed.value, conversion_error.what());
            throw;
        }
    }
}

// Walks `name`'s `extends` chain (through `library`, by name) merging every
// ancestor of the SAME ComponentTypeStruct category via `merge_fields`, and
// collecting every `Fixed` seen along the way into `fixed_parameter_values`.
// The chain stops (without error) the moment it reaches an ancestor
// cataloged under a different category (e.g. a CellType chain reaching
// `baseStandalone`, which is a bare ComponentTypeBase) -- that ancestor has
// no ComponentTypeStruct-shaped fields to contribute. An `extends` name that
// isn't cataloged as ANY ComponentType at all is a resolve error (a broken
// IDref).
template <typename ComponentTypeStruct, typename MergeFieldsFunction>
ComponentTypeStruct merge_chain(const String &name, const UnorderedMap<String, ComponentTypeEntry> &library,
                                 const String &category_label, MergeFieldsFunction merge_fields,
                                 UnorderedMap<String, f64> &fixed_parameter_values) {
    auto entry_iterator = library.find(name);
    if (entry_iterator == library.end() || !std::holds_alternative<ComponentTypeStruct>(entry_iterator->second)) {
        log::logger().error("resolve: '{}' is not a cataloged {}", name, category_label);
        throw std::runtime_error("resolve: '" + name + "' is not a cataloged " + category_label);
    }

    ComponentTypeStruct result = std::get<ComponentTypeStruct>(entry_iterator->second);
    Vector<FixedDecl> pending_fixed = extract_fixed_decls(result.raw);

    String parent_name = result.extends;
    for (int hop = 0; !parent_name.empty() && hop < 64; ++hop) {
        auto parent_iterator = library.find(parent_name);
        if (parent_iterator == library.end()) {
            log::logger().error("resolve: unresolved `extends` IDref '{}' in the ancestor chain of '{}'",
                                 parent_name, name);
            throw std::runtime_error("resolve: unresolved `extends` IDref '" + parent_name + "' in the ancestor chain of '" + name + "'");
        }

        if (!std::holds_alternative<ComponentTypeStruct>(parent_iterator->second)) break;

        const ComponentTypeStruct &parent = std::get<ComponentTypeStruct>(parent_iterator->second);
        merge_fields(result, parent);
        append_all(pending_fixed, extract_fixed_decls(parent.raw));

        parent_name = parent.extends;
    }

    apply_fixed_pins(pending_fixed, fixed_parameter_values);
    return result;
}

// ── Per-category field-merge functions (arch §3.1's declaration lists) ──

void merge_cell_fields(CellType &destination, const CellType &parent) {
    append_all(destination.parameters, parent.parameters);
    append_all(destination.state_variables, parent.state_variables);
    append_all(destination.derived_variables, parent.derived_variables);
    append_all(destination.time_derivatives, parent.time_derivatives);
    append_all(destination.on_conditions, parent.on_conditions);
    append_all(destination.exposures, parent.exposures);
    append_all(destination.event_ports, parent.event_ports);
    append_all(destination.regimes, parent.regimes);
    append_all(destination.attachments, parent.attachments);
}

void merge_synapse_fields(SynapseType &destination, const SynapseType &parent) {
    append_all(destination.parameters, parent.parameters);
    append_all(destination.properties, parent.properties);
    append_all(destination.state_variables, parent.state_variables);
    append_all(destination.derived_variables, parent.derived_variables);
    append_all(destination.time_derivatives, parent.time_derivatives);
    append_all(destination.requirements, parent.requirements);
    append_all(destination.exposures, parent.exposures);
    append_all(destination.event_ports, parent.event_ports);
    append_all(destination.on_events, parent.on_events);
    append_all(destination.component_references, parent.component_references);
    append_all(destination.children, parent.children);
}

void merge_inputs_fields(InputsType &destination, const InputsType &parent) {
    append_all(destination.parameters, parent.parameters);
    append_all(destination.properties, parent.properties);
    append_all(destination.state_variables, parent.state_variables);
    append_all(destination.derived_variables, parent.derived_variables);
    append_all(destination.time_derivatives, parent.time_derivatives);
    append_all(destination.on_conditions, parent.on_conditions);
    append_all(destination.on_events, parent.on_events);
    append_all(destination.exposures, parent.exposures);
    append_all(destination.event_ports, parent.event_ports);
    append_all(destination.children, parent.children);
    append_all(destination.component_references, parent.component_references);
}

void merge_population_fields(PopulationType &destination, const PopulationType &parent) {
    append_all(destination.component_references, parent.component_references);
    append_all(destination.parameters, parent.parameters);
    append_all(destination.children, parent.children);
}

void merge_project_fields(ProjectType &destination, const ProjectType &parent) {
    append_all(destination.component_references, parent.component_references);
    append_all(destination.parameters, parent.parameters);
    append_all(destination.path_fields, parent.path_fields);
    append_all(destination.children, parent.children);
}

// Flattens one library entry, dispatching on which of the five categories
// (or the bare, out-of-scope ComponentTypeBase) it holds.
ResolvedComponentType flatten_component_type(const String &name, const UnorderedMap<String, ComponentTypeEntry> &library) {
    const ComponentTypeEntry &entry = library.at(name);
    ResolvedComponentType resolved;

    if (std::holds_alternative<CellType>(entry)) {
        resolved.flattened = merge_chain<CellType>(name, library, "CellType", merge_cell_fields, resolved.fixed_parameter_values);
        resolved.bucket = ComponentTypeBucket::Dynamics;
    } else if (std::holds_alternative<SynapseType>(entry)) {
        resolved.flattened = merge_chain<SynapseType>(name, library, "SynapseType", merge_synapse_fields, resolved.fixed_parameter_values);
        resolved.bucket = ComponentTypeBucket::Dynamics;
    } else if (std::holds_alternative<InputsType>(entry)) {
        resolved.flattened = merge_chain<InputsType>(name, library, "InputsType", merge_inputs_fields, resolved.fixed_parameter_values);
        resolved.bucket = ComponentTypeBucket::Dynamics;
    } else if (std::holds_alternative<PopulationType>(entry)) {
        resolved.flattened = merge_chain<PopulationType>(name, library, "PopulationType", merge_population_fields, resolved.fixed_parameter_values);
        resolved.bucket = ComponentTypeBucket::Structure;
    } else if (std::holds_alternative<ProjectType>(entry)) {
        resolved.flattened = merge_chain<ProjectType>(name, library, "ProjectType", merge_project_fields, resolved.fixed_parameter_values);
        resolved.bucket = ComponentTypeBucket::Structure;
    } else {
        // Bare ComponentTypeBase: an out-of-scope category (ion channels,
        // morphology, biophysical properties, Phase 3, ...). It has no
        // declared field vectors to merge, so it is passed through as-is.
        resolved.flattened = std::get<ComponentTypeBase>(entry);
        resolved.bucket = ComponentTypeBucket::Structure;
    }

    return resolved;
}

// ── Instance-level tree resolution (S1/S2/S7 over root_node) ────────────

void register_instance_ids(const NML_Node &node, SymbolTable &symbols) {
    auto id_attribute = node.attributes.find("id");
    if (id_attribute != node.attributes.end()) {
        symbols.register_symbol(std::any_cast<String>(id_attribute->second));
    }
    for (const auto &child : node.body) {
        register_instance_ids(child, symbols);
    }
}

ResolvedInstance resolve_instance(const NML_Node &node, const SymbolTable &symbols) {
    ResolvedInstance result;
    result.tag_name = node.tag_name;

    for (const auto &[attribute_name, attribute_value] : node.attributes) {
        String value_text = std::any_cast<String>(attribute_value);

        if (attribute_name == "id") {
            result.id = value_text;
            result.symbol_index = symbols.index_of(value_text);
            continue;
        }

        if (contains(IDREF_ATTRIBUTE_NAMES, attribute_name)) {
            if (!symbols.contains(value_text)) {
                log::logger().error("resolve: unresolved IDref '{}' for attribute '{}' on <{}>",
                                     value_text, attribute_name, node.tag_name);
                throw std::runtime_error("resolve: unresolved IDref '" + value_text + "' for attribute '" +
                                          attribute_name + "' on <" + node.tag_name + ">");
            }
            result.idref_attributes[attribute_name] = symbols.index_of(value_text);
            continue;
        }

        if (contains(OPAQUE_STRING_ATTRIBUTE_NAMES, attribute_name)) {
            result.string_attributes[attribute_name] = value_text;
            continue;
        }

        try {
            result.numeric_attributes[attribute_name] = time::unit_value_to_si(value_text);
        } catch (const std::invalid_argument &conversion_error) {
            log::logger().error("resolve: attribute '{}' on <{}> has an unresolvable value '{}': {}",
                                 attribute_name, node.tag_name, value_text, conversion_error.what());
            throw;
        }
    }

    for (const auto &child : node.body) {
        result.children.push_back(resolve_instance(child, symbols));
    }

    return result;
}

} // namespace

s32 SymbolTable::register_symbol(const String &key) {
    auto existing = indices_.find(key);
    if (existing != indices_.end()) return existing->second;

    s32 next_index = static_cast<s32>(indices_.size());
    indices_.emplace(key, next_index);
    return next_index;
}

bool SymbolTable::contains(const String &key) const {
    return indices_.find(key) != indices_.end();
}

s32 SymbolTable::index_of(const String &key) const {
    auto entry = indices_.find(key);
    if (entry == indices_.end()) {
        log::logger().error("resolve: symbol '{}' is not registered in the symbol table", key);
        throw std::runtime_error("resolve: symbol '" + key + "' is not registered in the symbol table");
    }
    return entry->second;
}

ResolvedModel resolve_and_lower(const NML_Parser &parser) {
    ResolvedModel model;

    // 1. Symbol table (S7): every declaration-level ComponentType name, plus
    // every instance-level `id` in the tree, registered up front so IDref
    // resolution never depends on visitation order (same two-phase
    // rationale as NML_Parser::classify_all_cataloged_types).
    for (const auto &[type_name, entry] : parser.library) {
        (void)entry;
        model.symbols.register_symbol(type_name);
    }
    if (parser.root_node) {
        register_instance_ids(*parser.root_node, model.symbols);
    }

    // 2. `extends`/`Fixed` flattening + bucket sort, across the whole
    // library (arch §1.1/§3.1).
    for (const auto &[type_name, entry] : parser.library) {
        (void)entry;
        model.types.emplace(type_name, flatten_component_type(type_name, parser.library));
    }

    // 3. Units → SI + IDref wiring over the instance-level tree (S1/S2/S7).
    if (parser.root_node) {
        for (const auto &child : parser.root_node->body) {
            model.instances.push_back(resolve_instance(child, model.symbols));
        }
    }

    return model;
}

} // namespace spikecorec::nml
