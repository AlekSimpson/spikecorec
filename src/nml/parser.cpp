#include <libxml/parser.h>
#include <libxml/tree.h>

#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"
#include "spikecorec/core/recording.h"

#include "spikecorec/nml/parser.h"
#include "spikecorec/nml/declarations.h"
#include "spikecorec/nml/node.h"
#include "spikecorec/nml/components.h"
#include "spikecorec/nml/dynamics.h"
#include "spikecorec/nml/utilities.h"

using namespace spikecorec;
using namespace std;


namespace spikecorec::nml {

void NML_Parser::bind_instance_data(
    NML_Node *instance_node,
    const ComponentType &component_type,
    ComponentInstance &instance
) {
    Set<NML_DeclarationType> instance_bound_declarations = {
        NML_DeclarationType::Parameter,
        NML_DeclarationType::Property,
        NML_DeclarationType::Text,
        NML_DeclarationType::Path,
        NML_DeclarationType::ComponentReference
    };
    auto is_instance_bound = [instance_bound_declarations](NML_DeclarationType type) -> bool {
        return instance_bound_declrations.contains(type);
    };

    for (const auto &declaration : component_type.declarations.for_all()) {
        if (!is_instance_bound(declaration.tag_type)) continue;
        if (!declaration.has_value("name")) continue;

        String declaration_name = declaration.get_value("name");

        // Absent is not empty: a Fixed-pinned Parameter or a Property with a defaultValue
        // legitimately never appears on the instance tag.
        if (!instance_node->has_attribute(declaration_name)) continue;

        instance.instance_data[declaration_name] =
                instance_node->get_attribute(declaration_name);
    }
}

f64 NML_Parser::resolve_quantity(const String &value) const {
    auto [magnitude, suffix] = split_quantity(value);
    if (suffix.empty()) return magnitude;

    // A <Unit> the document declared wins: it is authoritative for this model, and it is
    // the only source that can express an offset (degC -> K).
    auto declared = declared_units.find(suffix);
    if (declared != declared_units.end()) {
        return magnitude * declared->second.scale + declared->second.offset;
    }

    return magnitude * units::unit_suffix_scale(suffix);
}



void NML_Parser::ingest_document(const String &file_path) {
    std::error_code path_error;
    path canonical = weakly_canonical(path(file_path), path_error);
    String canonical_text = path_error ? file_path : canonical.string();

    if (parsed_neuroml_files.find(canonical_text) != parsed_neuroml_files.end()) return;

    parsed_neuroml_files.insert(canonical_text);

    NML_Node root;
    if (!read_document_root(canonical_text, root)) return;

    path containing_directory = path(canonical_text).parent_path();

    for (NML_Node &child : root.body) {
        if (child.tag_name == "Include" || child.tag_name == "include") {
            String reference = child.has_attribute("file")
                    ? child.get_attribute("file")
                    : child.get_attribute("href");
            if (reference.empty()) continue;

            path included = path(reference);
            if (included.is_relative()) included = containing_directory / included;

            if (!exists(included) && !STANDARD_LIBRARY_PATH.empty()) {
                const path from_standard_library =
                        path(STANDARD_LIBRARY_PATH) / path(reference).filename();
                if (exists(from_standard_library)) included = from_standard_library;
            }

            ingest_document(included.string());
            continue;
        }

        if (child.tag_name == "ComponentType") {
            String type_name = child.get_attribute("name");
            if (type_name.empty()) {
                log::logger().warn("Ignoring <ComponentType> with no name in {}", canonical_text);
                continue;
            }

            auto existing = component_type_source_files.find(type_name);
            if (existing != component_type_source_files.end() &&
                existing->second != canonical_text) {
                throw runtime_error(
                        "Duplicate ComponentType '" + type_name + "' declared in both " +
                        existing->second + " and " + canonical_text);
            }

            component_type_catalogue[type_name] = std::move(child);
            component_type_source_files[type_name] = canonical_text;
            continue;
        }

        if (child.tag_name == "Unit") {
            String symbol = child.get_attribute("symbol");
            if (symbol.empty()) continue;

            // LEMS: si = raw * scale * 10^power + offset. power is folded into scale here
            // so resolution stays a single multiply-add.
            UnitDefinition definition;
            definition.scale = 1.0;
            definition.offset = 0.0;

            if (child.has_attribute("scale")) {
                auto [scale_magnitude, scale_suffix] = split_quantity(child.get_attribute("scale"));
                (void)scale_suffix;
                if (scale_magnitude != 0.0) definition.scale = scale_magnitude;
            }
            if (child.has_attribute("power")) {
                auto [power_magnitude, power_suffix] = split_quantity(child.get_attribute("power"));
                (void)power_suffix;
                definition.scale *= std::pow(10.0, power_magnitude);
            }
            if (child.has_attribute("offset")) {
                auto [offset_magnitude, offset_suffix] =
                        split_quantity(child.get_attribute("offset"));
                (void)offset_suffix;
                definition.offset = offset_magnitude;
            }

            declared_units[symbol] = definition;
            continue;
        }

        if (child.tag_name == "Dimension") continue; // todo?: dimensional analysis is not modelled

        if (child.tag_name == "Constant") {
            String constant_name = child.get_attribute("name");
            if (constant_name.empty()) continue;

            Real resolved;
            resolved.float64 = resolve_quantity(child.get_attribute("value"));
            document_constants[constant_name] = resolved;
            continue;
        }

        if (child.tag_name == "Target") {
            target_component_id = child.get_attribute("component");
            continue;
        }

        // Anything else is a component instance
        document_instance_nodes.push_back(std::move(child));
    }
}

void NML_Parser::extract_all_declarations_nested_in_node(
    NML_Node *node,
    DeclarationList &return_value
) {
    if (!node) return;

    for (const NML_Node &child : node.body) {
        if (child.is_declaration_type()) {
            return_value.insert(child.to_declaration());
            continue;
        }

        extract_all_declarations_nested_in_node(child, return_value);
    }
}

const ComponentType &NML_Parser::resolve_component_type(const String &type_name,
                                                        Vector<String> &resolution_stack) {
    auto already_resolved = declared_component_types.find(type_name);
    if (already_resolved != declared_component_types.end()) return already_resolved->second;

    auto catalogued = component_type_catalogue.find(type_name);
    if (catalogued == component_type_catalogue.end()) {
        String chain;
        for (const String &entry : resolution_stack) chain += entry + " -> ";
        throw runtime_error(
                "Unresolved ComponentType '" + type_name +
                "' referenced from extends chain: " + chain + type_name);
    }

    for (const String &entry : resolution_stack) {
        if (entry != type_name) continue;

        String chain;
        for (const String &stack_entry : resolution_stack) chain += stack_entry + " -> ";
        throw runtime_error("Cyclic ComponentType extends chain: " + chain + type_name);
    }

    resolution_stack.push_back(type_name);

    // A reference, not a copy: resolution only ever writes to declared_component_types,
    // never to the catalogue, so this stays valid across the recursive parent resolve.
    const NML_Node &type_node = catalogued->second;
    String parent_name = type_node.get_attribute("extends");

    DeclarationList declarations;
    if (!parent_name.empty()) {
        const ComponentType &parent = resolve_component_type(parent_name, resolution_stack);
        declarations = parent.declarations;
    }

    DeclarationList own_declarations;
    extract_all_declarations_nested_in_node(type_node, own_declarations);
    for (NML_Declaration &declaration : own_declarations.for_all()) {
        declarations.overlay(declaration)
    }

    /// +16 ???
    // <Fixed parameter="tau" value="10ms"/> pins an inherited Parameter to a constant.
    // Applied after the overlay so it can reach a Parameter this type never declared.
    for (const NML_Declaration &declaration : declarations.for_all()) {
        if (declaration.tag_type != NML_DeclarationType::Fixed) continue;

        String parameter_name = declaration.value_or("parameter");
        if (parameter_name.empty()) continue;

        for (NML_Declaration &target : declarations.declarations) {
            if (target.tag_type != NML_DeclarationType::Parameter) continue;
            if (target.value_or("name") != parameter_name) continue;

            target.datavalues["value"] = declaration.value_or("value");
            break;
        }
    }

    resolution_stack.pop_back();

    ComponentType resolved(type_name, std::move(declarations));
    resolved.extends = parent_name;

    declared_component_types[type_name] = std::move(resolved);
    return declared_component_types[type_name];
}

void NML_Parser::resolve_all_component_types() {
    Vector<String> type_names;
    type_names.reserve(component_type_catalogue.size());
    for (const auto &[type_name, node] : component_type_catalogue) {
        (void)node;
        type_names.push_back(type_name);
    }
    std::sort(type_names.begin(), type_names.end());

    for (const String &type_name : type_names) {
        Vector<String> resolution_stack;
        resolve_component_type(type_name, resolution_stack);
    }
}

void NML_Parser::instantiate_component_type(NML_Node *instance_node) {
    String no_parent;
    instantiate_component_type(instance_node, no_parent);
}

void NML_Parser::instantiate_component_type(NML_Node *instance_node, String &parent_instance_id) {
    if (!instance_node) return;

    String component_type_name = instance_node->tag_name;

    auto type_entry = declared_component_types.find(component_type_name);
    if (type_entry == declared_component_types.end()) {
        log::logger().warn("Ignoring <{}>: no such ComponentType is declared",
                           component_type_name);
        return;
    }
    const ComponentType &component_to_instantiate = type_entry->second;

    String own_id = instance_node->get_attribute("id");
    if (own_id.empty()) {
        // A <connection> inside a projection routinely carries no id. Without a synthetic
        // one every such child collapses onto the same empty key and all but the last is
        // lost.
        usize ordinal = 0;
        if (!parent_instance_id.empty()) {
            auto parent = instance_table.find(parent_instance_id);
            if (parent != instance_table.end()) {
                ordinal = parent->second.structured_instance_data.size();
            }
        }
        own_id = component_type_name + "[" + std::to_string(ordinal) + "]";
    }

    String instance_id = parent_instance_id.empty()
            ? own_id
            : parent_instance_id + "." + own_id;

    if (instance_table.find(instance_id) != instance_table.end()) {
        log::logger().warn("Duplicate component instance id '{}'; the later one wins",
                           instance_id);
    }

    ComponentInstance &instance = instance_table[instance_id];
    instance.id = instance_id;
    instance.component_type_name = component_type_name;
    instance.parent_instance_id = parent_instance_id;
    instance.component_type = &component_to_instantiate;

    bind_instance_data(instance_node, component_to_instantiate, instance);

    // Registered on the parent before recursing, so structured_instance_data lists
    // children in document order.
    if (!parent_instance_id.empty()) {
        instance_table[parent_instance_id].structured_instance_data.push_back(instance_id);
    }

    for (NML_Node &child_instance_node : instance_node->body) {
        instantiate_component_type(&child_instance_node, instance_id);
    }
}

void NML_Parser::parse_lems(const String &lems_main_file) {
    if (!STANDARD_LIBRARY_PATH.empty() && exists(STANDARD_LIBRARY_PATH)) {
        // Sorted: directory_iterator order is unspecified, and ingest order decides which
        // file a duplicate ComponentType is blamed on.
        Vector<String> standard_library_files;
        for (const auto &entry : directory_iterator(STANDARD_LIBRARY_PATH)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".xml" && entry.path().extension() != ".nml") continue;

            standard_library_files.push_back(entry.path().string());
        }
        std::sort(standard_library_files.begin(), standard_library_files.end());

        for (const String &filepath : standard_library_files) {
            ingest_document(filepath);
        }
    } else {
        log::logger().error("NML standard library path does not exist: {}",
                            STANDARD_LIBRARY_PATH);
    }

    ingest_document(lems_main_file);

    resolve_all_component_types();

    std::error_code path_error;
    path canonical = weakly_canonical(path(lems_main_file), path_error);
    read_document_root(path_error ? lems_main_file : canonical.string(), main_document_root);
}

NML_Context NML_Parser::extract_neuroml_context(NML_Node *lems_root) {
    NML_Context context;
    //TODO
    return context;
}

// Accumulates each schema validation error's line number and message (libxml2's messages
// already name the offending element, e.g. "Element 'thisTagDoesNotExistInSchema': No
// matching global declaration available for the validation root.") into `*user_data`,
// joined by " | " -- see validate_against_schema's own comment for why this replaces
// libxml2's default, stderr-only error handler.
// libxml2 2.12 made xmlStructuredErrorFunc take a `const xmlError *`; before that it took
// a mutable xmlErrorPtr. This machine has both 2.9 (pkg-config, what the Makefile picks)
// and 2.13 (xml2-config) installed, so the signature is selected rather than assumed.
#if LIBXML_VERSION >= 21200
void collect_schema_validation_error(void *user_data, const xmlError *error) {
#else
void collect_schema_validation_error(void *user_data, xmlErrorPtr error) {
#endif
    if (!error || !error->message) return;

    String message(error->message);
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) message.pop_back();

    String *destination = static_cast<String *>(user_data);
    if (!destination->empty()) *destination += " | ";
    *destination += "line " + std::to_string(error->line) + ": " + message;
}

bool NML_Parser::validate_lems_schema(const String &lems_filepath) {
    xmlDocPtr document = xmlReadFile(nml_file_path.c_str(), nullptr, 0);
    if (!document) {
        last_schema_validation_errors = "could not read " + nml_file_path;
        return false;
    }

    xmlNodePtr root = xmlDocGetRootElement(document);
    const bool is_neuroml_document =
            root && root->name &&
            String(reinterpret_cast<const char *>(root->name)) == "neuroml";
    xmlFreeDoc(document);

    if (!is_neuroml_document) {
        log::logger().debug("{} is not a NeuroML document root; skipping XSD validation",
                            nml_file_path);
        last_schema_validation_errors.clear();
        return true;
    }

    xmlSchemaParserCtxtPtr parser_context = xmlSchemaNewParserCtxt(NML_SCHEMA_PATH.c_str());
    if (!parser_context) {
        log::logger().error("Could not create XSD parser context for schema {}", NML_SCHEMA_PATH);
        return false;
    }

    xmlSchemaPtr schema = xmlSchemaParse(parser_context);
    xmlSchemaFreeParserCtxt(parser_context);
    if (!schema) {
        log::logger().error("Could not parse XSD schema {}", NML_SCHEMA_PATH);
        return false;
    }

    xmlSchemaValidCtxtPtr valid_context = xmlSchemaNewValidCtxt(schema);
    if (!valid_context) {
        log::logger().error(
                "Could not create XSD validation context for schema {}", NML_SCHEMA_PATH);
        xmlSchemaFree(schema);
        return false;
    }

    last_schema_validation_errors.clear();
    xmlSchemaSetValidStructuredErrors(
            valid_context, collect_schema_validation_error, &last_schema_validation_errors);

    // xmlSchemaValidateFile: 0 = valid, >0 = validation errors (captured above, by element and line
    // number), <0 = internal/API error.
    int result = xmlSchemaValidateFile(valid_context, nml_file_path.c_str(), 0);

    xmlSchemaFreeValidCtxt(valid_context);
    xmlSchemaFree(schema);

    return result == 0;
}

xmlNodePtr NML_Parser::get_xml_root(const String &filepath) {
    xmlDocPtr document = xmlReadFile(filepath.c_str(), nullptr, XML_PARSE_NOBLANKS);
    if (!document) {
        log::logger().error("Could not parse NML standard library file {}", filepath);
        return nullptr;
    }

    xmlNodePtr root = xmlDocGetRootElement(document);
    if (!root) {
        log::logger().error("NML standard library file {} has no root element", filepath);
        xmlFreeDoc(document);
        return nullptr;
    }

    return root;
}









}











