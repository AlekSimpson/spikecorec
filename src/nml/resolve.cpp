#include <cstdlib>
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
// resolve error. `target` is deliberately NOT here -- see below.
const Vector<String> IDREF_ATTRIBUTE_NAMES = {
    "component", "cell", "presynapticPopulation", "postsynapticPopulation",
    "synapse", "input", "ionChannel"};

// Instance-level attributes that are identity/metadata or structural paths
// (arch §1.2 S6 / §3.1 `Path`/`Text`) rather than dimensioned quantities --
// passed through as raw strings, never unit-converted or IDref-resolved.
// `target` is genuinely dual-use in real NeuroML (`Simulation target="net1"`
// is a bare id, but `explicitInput target="Pop0[0]"` is a population-index
// path expression) -- treating it as a strict IDref makes the common path
// form throw a false "unresolved IDref", so it is treated as an opaque path
// here, same as `preCellId`/`postCellId`, at the cost of not validating the
// bare-id (`Simulation target=`) form.
const Vector<String> OPAQUE_STRING_ATTRIBUTE_NAMES = {
    "name", "notes", "preCellId", "postCellId", "destination", "type", "target"};

// Whether `value_text` even looks like a dimensioned literal (a leading
// parsable number, per the same `strtod` rule `unit_value_to_si` itself
// uses) as opposed to an arbitrary metadata/identifier string. Attributes
// outside the hardcoded IDref/opaque allowlists above fall back to this
// check rather than being force-fed to `unit_value_to_si` -- so an unlisted
// string attribute is treated as opaque instead of throwing a misleading
// "unconvertible unit" error (it never was a unit to begin with).
bool looks_like_dimensioned_literal(const String &value_text) {
    const char *text_start = value_text.c_str();
    char *number_end = nullptr;
    std::strtod(text_start, &number_end);
    return number_end != text_start;
}

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

// Merges one ancestor's declarations into `accumulated` (the so-far-merged,
// more-derived set: the type's own declarations, plus every nearer ancestor
// already folded in), keyed by `key_of` (a decl's `name`, or `variable` for
// TimeDerivativeDecl, or `test`/`port` for OnCondition/OnEventDecl). An
// ancestor's declaration for a key `accumulated` already carries is skipped
// -- so the most-derived declaration for a given key always wins (arch
// §3.1 `extends`: inheritance-override semantics) and no key is ever
// double-counted downstream, where these vectors size per-instance state
// (ticket #50 [B2] cell lowering, #5 [C2] allocator).
template <typename T, typename KeyFunction>
void merge_with_override(Vector<T> &accumulated, const Vector<T> &ancestor_declarations, KeyFunction key_of) {
    for (const auto &ancestor_declaration : ancestor_declarations) {
        String key = key_of(ancestor_declaration);
        bool already_declared = false;
        for (const auto &existing : accumulated) {
            if (key_of(existing) == key) {
                already_declared = true;
                break;
            }
        }
        if (!already_declared) accumulated.push_back(ancestor_declaration);
    }
}

// Applies every pending `Fixed` pin, converting its value to SI (S1) and
// recording it under the parameter's name. `pending_fixed` is ordered
// most-derived-first (a type's own `Fixed`s, then its nearest ancestor's,
// and so on -- see merge_chain), so a parameter already pinned is skipped
// here rather than overwritten: the most-derived `Fixed` for a given
// parameter always wins over an ancestor's (arch §3.1 `extends`:
// inheritance-override semantics), without even evaluating an
// already-shadowed ancestor value. An unconvertible value is a resolve
// error, same as any other dimensioned literal.
void apply_fixed_pins(const Vector<FixedDecl> &pending_fixed, UnorderedMap<String, f64> &fixed_parameter_values) {
    for (const auto &fixed : pending_fixed) {
        if (fixed_parameter_values.count(fixed.parameter)) continue;

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

// Key extractors shared across categories -- a generic lambda works for any
// decl struct with a `.name` member (ParameterDecl, StateVariableDecl, ...).
auto by_name = [](const auto &decl) { return decl.name; };
auto by_variable = [](const TimeDerivativeDecl &decl) { return decl.variable; };
auto by_test = [](const OnConditionDecl &decl) { return decl.test; };
auto by_port = [](const OnEventDecl &decl) { return decl.port; };

void merge_cell_fields(CellType &destination, const CellType &parent) {
    merge_with_override(destination.parameters, parent.parameters, by_name);
    merge_with_override(destination.state_variables, parent.state_variables, by_name);
    merge_with_override(destination.derived_variables, parent.derived_variables, by_name);
    merge_with_override(destination.time_derivatives, parent.time_derivatives, by_variable);
    merge_with_override(destination.on_conditions, parent.on_conditions, by_test);
    merge_with_override(destination.exposures, parent.exposures, by_name);
    merge_with_override(destination.event_ports, parent.event_ports, by_name);
    merge_with_override(destination.regimes, parent.regimes, by_name);
    merge_with_override(destination.attachments, parent.attachments, by_name);
    // OnStart has no natural override key (no name/variable identity) --
    // an ancestor's initial-value assignments are simply additional seeds
    // alongside the child's own (arch §3.1 `extends`: "merges ... the
    // entire <Dynamics>").
    append_all(destination.on_starts, parent.on_starts);
}

void merge_synapse_fields(SynapseType &destination, const SynapseType &parent) {
    merge_with_override(destination.parameters, parent.parameters, by_name);
    merge_with_override(destination.properties, parent.properties, by_name);
    merge_with_override(destination.state_variables, parent.state_variables, by_name);
    merge_with_override(destination.derived_variables, parent.derived_variables, by_name);
    merge_with_override(destination.time_derivatives, parent.time_derivatives, by_variable);
    merge_with_override(destination.requirements, parent.requirements, by_name);
    merge_with_override(destination.exposures, parent.exposures, by_name);
    merge_with_override(destination.event_ports, parent.event_ports, by_name);
    merge_with_override(destination.on_events, parent.on_events, by_port);
    merge_with_override(destination.component_references, parent.component_references, by_name);
    merge_with_override(destination.children, parent.children, by_name);
    append_all(destination.on_starts, parent.on_starts);
}

void merge_inputs_fields(InputsType &destination, const InputsType &parent) {
    merge_with_override(destination.parameters, parent.parameters, by_name);
    merge_with_override(destination.properties, parent.properties, by_name);
    merge_with_override(destination.state_variables, parent.state_variables, by_name);
    merge_with_override(destination.derived_variables, parent.derived_variables, by_name);
    merge_with_override(destination.time_derivatives, parent.time_derivatives, by_variable);
    merge_with_override(destination.on_conditions, parent.on_conditions, by_test);
    merge_with_override(destination.on_events, parent.on_events, by_port);
    merge_with_override(destination.exposures, parent.exposures, by_name);
    merge_with_override(destination.event_ports, parent.event_ports, by_name);
    merge_with_override(destination.children, parent.children, by_name);
    merge_with_override(destination.component_references, parent.component_references, by_name);
    append_all(destination.on_starts, parent.on_starts);
}

void merge_population_fields(PopulationType &destination, const PopulationType &parent) {
    merge_with_override(destination.component_references, parent.component_references, by_name);
    merge_with_override(destination.parameters, parent.parameters, by_name);
    merge_with_override(destination.children, parent.children, by_name);
}

void merge_project_fields(ProjectType &destination, const ProjectType &parent) {
    merge_with_override(destination.component_references, parent.component_references, by_name);
    merge_with_override(destination.parameters, parent.parameters, by_name);
    merge_with_override(destination.path_fields, parent.path_fields, by_name);
    merge_with_override(destination.children, parent.children, by_name);
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

        if (contains(OPAQUE_STRING_ATTRIBUTE_NAMES, attribute_name) || !looks_like_dimensioned_literal(value_text)) {
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
