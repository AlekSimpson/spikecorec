#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xmlschemastypes.h>
#include <filesystem>

#include "spikecorec/nml/nml.h"
#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;
using namespace std::filesystem;

namespace spikecorec::nml {

namespace {

// ── ComponentType classification/extraction helpers (ticket #2 [A3]) ────
//
// Pulls the declaration-level (§3.1) and, where present, Dynamics-nested
// (§3.2) tags out of a raw `<ComponentType>` NML_Node into the typed decl
// structs declared in nml.h. Each extractor is a one-to-one mapping from one
// recurring LEMS tag to its decl struct — no semantic interpretation, just
// structural extraction (bake-vs-parameterize, extends-merge, etc. stay
// resolve-pass concerns, ticket #49).

String get_attr(const NML_Node &node, const String &name) {
    auto entry = node.attributes.find(name);
    if (entry == node.attributes.end()) return "";
    return std::any_cast<String>(entry->second);
}

const NML_Node *find_child(const NML_Node &node, const String &tag) {
    for (const auto &child : node.body) {
        if (child.tag_name == tag) return &child;
    }
    return nullptr;
}

Vector<const NML_Node *> find_children(const NML_Node &node, const String &tag) {
    Vector<const NML_Node *> matches;
    for (const auto &child : node.body) {
        if (child.tag_name == tag) matches.push_back(&child);
    }
    return matches;
}

Vector<ParameterDecl> extract_parameters(const NML_Node &node) {
    Vector<ParameterDecl> result;
    for (const auto *child : find_children(node, "Parameter")) {
        result.push_back(ParameterDecl{get_attr(*child, "name"), get_attr(*child, "dimension")});
    }
    return result;
}

Vector<PropertyDecl> extract_properties(const NML_Node &node) {
    Vector<PropertyDecl> result;
    for (const auto *child : find_children(node, "Property")) {
        result.push_back(PropertyDecl{get_attr(*child, "name"), get_attr(*child, "dimension"), get_attr(*child, "defaultValue")});
    }
    return result;
}

Vector<ExposureDecl> extract_exposures(const NML_Node &node) {
    Vector<ExposureDecl> result;
    for (const auto *child : find_children(node, "Exposure")) {
        result.push_back(ExposureDecl{get_attr(*child, "name"), get_attr(*child, "dimension")});
    }
    return result;
}

Vector<EventPortDecl> extract_event_ports(const NML_Node &node) {
    Vector<EventPortDecl> result;
    for (const auto *child : find_children(node, "EventPort")) {
        result.push_back(EventPortDecl{get_attr(*child, "name"), get_attr(*child, "direction")});
    }
    return result;
}

Vector<AttachmentDecl> extract_attachments(const NML_Node &node) {
    Vector<AttachmentDecl> result;
    for (const auto *child : find_children(node, "Attachments")) {
        result.push_back(AttachmentDecl{get_attr(*child, "name"), get_attr(*child, "type")});
    }
    return result;
}

Vector<RequirementDecl> extract_requirements(const NML_Node &node) {
    Vector<RequirementDecl> result;
    for (const auto *child : find_children(node, "Requirement")) {
        result.push_back(RequirementDecl{get_attr(*child, "name"), get_attr(*child, "dimension")});
    }
    return result;
}

// `ComponentReference`/`Link`, plus the electrical/graded-synapse and
// projection-family equivalents (`InstanceRequirement`, `ComponentRequirement`)
// — all "a reference to another component's type," unified into one list.
Vector<ComponentReferenceDecl> extract_component_references(const NML_Node &node) {
    Vector<ComponentReferenceDecl> result;
    for (const char *tag : {"ComponentReference", "Link", "InstanceRequirement", "ComponentRequirement"}) {
        for (const auto *child : find_children(node, tag)) {
            result.push_back(ComponentReferenceDecl{get_attr(*child, "name"), get_attr(*child, "type")});
        }
    }
    return result;
}

Vector<ChildrenDecl> extract_children_decls(const NML_Node &node) {
    Vector<ChildrenDecl> result;
    for (const char *tag : {"Children", "Child"}) {
        for (const auto *child : find_children(node, tag)) {
            result.push_back(ChildrenDecl{get_attr(*child, "name"), get_attr(*child, "type")});
        }
    }
    return result;
}

Vector<PathDecl> extract_path_fields(const NML_Node &node) {
    Vector<PathDecl> result;
    for (const char *tag : {"Path", "Text"}) {
        for (const auto *child : find_children(node, tag)) {
            result.push_back(PathDecl{get_attr(*child, "name")});
        }
    }
    return result;
}

Vector<StateVariableDecl> extract_state_variables(const NML_Node &dynamics) {
    Vector<StateVariableDecl> result;
    for (const auto *child : find_children(dynamics, "StateVariable")) {
        result.push_back(StateVariableDecl{get_attr(*child, "name"), get_attr(*child, "dimension"), get_attr(*child, "exposure")});
    }
    return result;
}

Vector<DerivedVariableDecl> extract_derived_variables(const NML_Node &dynamics) {
    Vector<DerivedVariableDecl> result;
    for (const auto *child : find_children(dynamics, "DerivedVariable")) {
        result.push_back(DerivedVariableDecl{
            get_attr(*child, "name"), get_attr(*child, "dimension"), get_attr(*child, "exposure"),
            get_attr(*child, "value"), get_attr(*child, "select"), get_attr(*child, "reduce")});
    }
    return result;
}

Vector<TimeDerivativeDecl> extract_time_derivatives(const NML_Node &dynamics) {
    Vector<TimeDerivativeDecl> result;
    for (const auto *child : find_children(dynamics, "TimeDerivative")) {
        result.push_back(TimeDerivativeDecl{get_attr(*child, "variable"), get_attr(*child, "value")});
    }
    return result;
}

Vector<OnConditionDecl> extract_on_conditions(const NML_Node &dynamics) {
    Vector<OnConditionDecl> result;
    for (const auto *child : find_children(dynamics, "OnCondition")) {
        result.push_back(OnConditionDecl{get_attr(*child, "test"), *child});
    }
    return result;
}

Vector<OnEventDecl> extract_on_events(const NML_Node &dynamics) {
    Vector<OnEventDecl> result;
    for (const auto *child : find_children(dynamics, "OnEvent")) {
        result.push_back(OnEventDecl{get_attr(*child, "port"), *child});
    }
    return result;
}

Vector<RegimeDecl> extract_regimes(const NML_Node &dynamics) {
    Vector<RegimeDecl> result;
    for (const auto *child : find_children(dynamics, "Regime")) {
        result.push_back(RegimeDecl{get_attr(*child, "name"), get_attr(*child, "initial"), *child});
    }
    return result;
}

ComponentTypeBase make_base(const NML_Node &node, const String &extends) {
    return ComponentTypeBase(get_attr(node, "name"), extends, node);
}

CellType build_cell_type(const NML_Node &node, const String &extends) {
    CellType result(get_attr(node, "name"), extends, node);

    result.parameters = extract_parameters(node);
    result.exposures = extract_exposures(node);
    result.event_ports = extract_event_ports(node);
    result.attachments = extract_attachments(node);

    if (const NML_Node *dynamics = find_child(node, "Dynamics")) {
        result.state_variables = extract_state_variables(*dynamics);
        result.derived_variables = extract_derived_variables(*dynamics);
        result.time_derivatives = extract_time_derivatives(*dynamics);
        result.on_conditions = extract_on_conditions(*dynamics);
        result.regimes = extract_regimes(*dynamics);
    }

    return result;
}

SynapseType build_synapse_type(const NML_Node &node, const String &extends) {
    SynapseType result(get_attr(node, "name"), extends, node);

    result.parameters = extract_parameters(node);
    result.properties = extract_properties(node);
    result.requirements = extract_requirements(node);
    result.exposures = extract_exposures(node);
    result.event_ports = extract_event_ports(node);
    result.component_references = extract_component_references(node);
    result.children = extract_children_decls(node);

    if (const NML_Node *dynamics = find_child(node, "Dynamics")) {
        result.state_variables = extract_state_variables(*dynamics);
        result.derived_variables = extract_derived_variables(*dynamics);
        result.time_derivatives = extract_time_derivatives(*dynamics);
        result.on_events = extract_on_events(*dynamics);
    }

    return result;
}

InputsType build_inputs_type(const NML_Node &node, const String &extends) {
    InputsType result(get_attr(node, "name"), extends, node);

    result.parameters = extract_parameters(node);
    result.properties = extract_properties(node);
    result.exposures = extract_exposures(node);
    result.event_ports = extract_event_ports(node);
    result.children = extract_children_decls(node);
    result.component_references = extract_component_references(node);

    if (const NML_Node *dynamics = find_child(node, "Dynamics")) {
        result.state_variables = extract_state_variables(*dynamics);
        result.derived_variables = extract_derived_variables(*dynamics);
        result.time_derivatives = extract_time_derivatives(*dynamics);
        result.on_conditions = extract_on_conditions(*dynamics);
        result.on_events = extract_on_events(*dynamics);
    }

    return result;
}

PopulationType build_population_type(const NML_Node &node, const String &extends) {
    PopulationType result(get_attr(node, "name"), extends, node);

    result.component_references = extract_component_references(node);
    result.parameters = extract_parameters(node);
    result.children = extract_children_decls(node);

    return result;
}

ProjectType build_project_type(const NML_Node &node, const String &extends) {
    ProjectType result(get_attr(node, "name"), extends, node);

    result.component_references = extract_component_references(node);
    result.parameters = extract_parameters(node);
    result.path_fields = extract_path_fields(node);
    result.children = extract_children_decls(node);

    return result;
}

} // namespace

void NML_Node::add_attribute(String name, Any value) {
    attributes[name] = std::move(value);
}

void NML_Node::nest(NML_Node component) {
    body.push_back(component);
}

void NML_Parser::parse(const String &nml_input_file) {
    bool failed_to_load = load_standard_library();
    if (failed_to_load) {
        log::logger().error("Failed to load NML standard library. It may be missing or malformed.");
        throw runtime_error("Failed to load NML standard library. It may be missing or malformed.");
    }

    ingest_file(nml_input_file, true);
    xmlCleanupParser();
}

// Ingests a single NML/LEMS file into the parser's shared tree, following
// <include href="..."> recursively so the whole include graph ends up merged
// into one NML_Node tree hanging off root_node. Only the top-level file
// passed to parse() is schema-validated: included files are frequently raw
// LEMS (std-lib bundle), which never validates against the NeuroML2 XSD, so
// validation there would be a false positive, not a real error.
void NML_Parser::ingest_file(const String &nml_file_path, bool run_schema_validation) {
    if (run_schema_validation && !validate_against_schema(nml_file_path)) {
        log::logger().error("{} does not validate against the NeuroML2 XSD schema", nml_file_path);
        throw std::runtime_error(nml_file_path + " does not validate against the NeuroML2 XSD schema");
    }

    xmlDocPtr document = xmlReadFile(nml_file_path.c_str(), nullptr, XML_PARSE_NOBLANKS);
    if (!document) {
        log::logger().error("Could not parse NML file {}", nml_file_path);
        return;
    }

    xmlNodePtr root = xmlDocGetRootElement(document);
    if (!root) {
        log::logger().error("NML file {} has no root element", nml_file_path);
        xmlFreeDoc(document);
        return;
    }

    if (!root_node) {
        root_node = new NML_Node(reinterpret_cast<const char *>(root->name));
    }

    String base_directory = path(nml_file_path).parent_path().string();

    for (xmlNodePtr node = root->children; node; node = node->next) {
        if (node->type != XML_ELEMENT_NODE) continue;

        if (xmlStrEqual(node->name, BAD_CAST "include")) {
            xmlChar *href = xmlGetProp(node, BAD_CAST "href");
            if (!href) {
                log::logger().warn("<include> in {} has no href attribute, skipping", nml_file_path);
                continue;
            }

            String include_path = (path(base_directory) / reinterpret_cast<const char *>(href)).string();
            xmlFree(href);

            ingest_file(include_path, false);
            continue;
        }

        NML_Node child = xml_node_to_nml_node(node);

        if (!xmlStrEqual(node->name, BAD_CAST "ComponentType")) {
            root_node->nest(child);
            continue;
        }

        catalog_component_type(child, nml_file_path);
    }

    xmlFreeDoc(document);
}

NML_Node NML_Parser::xml_node_to_nml_node(xmlNodePtr node) {
    String tag_name = reinterpret_cast<const char *>(node->name);
    auto component = NML_Node(tag_name);

    for (xmlAttrPtr attribute = node->properties; attribute; attribute = attribute->next) {
        xmlChar *value = xmlGetProp(node, attribute->name);
        if (value) {
            component.add_attribute(reinterpret_cast<const char *>(attribute->name),
                                    String(reinterpret_cast<const char *>(value)));
            xmlFree(value);
        }
    }

    for (xmlNodePtr child = node->children; child; child = child->next) {
        if (child->type != XML_ELEMENT_NODE) continue;
        component.nest(xml_node_to_nml_node(child));
    }

    return component;
}


bool NML_Parser::load_standard_library() {
    bool library_failed_to_load = false;

    if (!exists(STANDARD_LIBRARY_PATH)) {
        log::logger().error("NML standard library path does not exist: {}", STANDARD_LIBRARY_PATH);
        return true;
    }

    for (const auto &entry : directory_iterator(STANDARD_LIBRARY_PATH)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".xml" && entry.path().extension() != ".nml") continue;

        String file_path = entry.path().string();
        String file_name = entry.path().filename().string();

        xmlDocPtr document = xmlReadFile(file_path.c_str(), nullptr, XML_PARSE_NOBLANKS);
        if (!document) {
            log::logger().error("Could not parse NML standard library file {}", file_path);
            library_failed_to_load = true;
            continue;
        }

        xmlNodePtr root = xmlDocGetRootElement(document);
        if (!root) {
            log::logger().error("NML standard library file {} has no root element", file_path);
            xmlFreeDoc(document);
            library_failed_to_load = true;
            continue;
        }

        for (xmlNodePtr node = root->children; node; node = node->next) {
            if (node->type != XML_ELEMENT_NODE) continue;
            if (!xmlStrEqual(node->name, BAD_CAST "ComponentType")) continue;

            NML_Node component_type = xml_node_to_nml_node(node);

            if (!catalog_component_type(component_type, file_path)) {
                library_failed_to_load = true;
            }
        }

        xmlFreeDoc(document);
    }

    xmlCleanupParser();
    return library_failed_to_load;
}

// Catalogs a parsed `<ComponentType>` node into `library` (classified) and
// `raw_component_types` (raw, for classify_component_type's `extends`-chain
// lookups). Duplicate names are dropped silently (first-cataloged-wins,
// matching the original flat-library behavior) — that is not a failure.
bool NML_Parser::catalog_component_type(const NML_Node &component_type_node, const String &source_path) {
    auto name_attribute = component_type_node.attributes.find("name");
    if (name_attribute == component_type_node.attributes.end()) {
        log::logger().warn("ComponentType in {} has no name attribute, skipping", source_path);
        return false;
    }

    String type_name;
    try {
        type_name = std::any_cast<String>(name_attribute->second);
    } catch (const std::bad_any_cast &) {
        log::logger().error("ComponentType in {} has a non-string name attribute, skipping", source_path);
        return false;
    }

    if (raw_component_types.find(type_name) != raw_component_types.end()) {
        return true; // already cataloged — first-cataloged-wins, not a failure
    }

    auto inserted = raw_component_types.emplace(type_name, component_type_node);
    library.emplace(type_name, classify_component_type(inserted.first->second));
    return true;
}

// Classifies `node` into one of the five ComponentType categories (nml.h)
// by walking its `extends` chain (through `raw_component_types`, by name) up
// to one of the anchor base types the arch doc's §3.3 D1/D3/D4/ST2/ST3
// buckets are grounded on. The chain check includes `node` itself, since
// several Structure-level types (`connection`, `projection`, ...) are
// themselves roots with no `extends` at all. A chain reaching none of the
// anchors (ion channels, morphology, biophysical properties, Phase 3, ...)
// is left unclassified — the bare ComponentTypeBase identity, raw node
// still intact.
ComponentTypeEntry NML_Parser::classify_component_type(const NML_Node &node) {
    static const Vector<String> cell_anchors = {"baseCell"};
    static const Vector<String> synapse_anchors = {"baseSynapse"};
    static const Vector<String> inputs_anchors = {"basePointCurrent", "baseSpikeSource"};
    static const Vector<String> population_anchors = {"basePopulation", "population", "populationList"};
    static const Vector<String> project_anchors = {
        "projection", "connection", "connectionWD", "explicitConnection",
        "synapticConnection", "synapticConnectionWD",
        "electricalConnection", "electricalConnectionInstance", "electricalConnectionInstanceW",
        "electricalProjection",
        "continuousConnection", "continuousConnectionInstance", "continuousConnectionInstanceW",
        "continuousProjection"};

    auto contains = [](const Vector<String> &anchors, const String &value) {
        for (const auto &anchor : anchors) {
            if (anchor == value) return true;
        }
        return false;
    };

    // Walks from `node` up through successive `extends` lookups (bounded to
    // guard against a cyclic chain), returning true the moment any name in
    // the chain (including `node` itself) matches one of `anchors`.
    auto reaches_anchor = [this, &contains](const NML_Node &start, const Vector<String> &anchors) {
        String current_name = get_attr(start, "name");
        const NML_Node *current_node = &start;

        for (int hop = 0; hop < 64 && !current_name.empty(); ++hop) {
            if (contains(anchors, current_name)) return true;

            String parent_name = get_attr(*current_node, "extends");
            if (parent_name.empty()) return false;

            auto parent_entry = raw_component_types.find(parent_name);
            if (parent_entry == raw_component_types.end()) {
                return contains(anchors, parent_name);
            }

            current_name = parent_name;
            current_node = &parent_entry->second;
        }
        return false;
    };

    String extends = get_attr(node, "extends");

    if (reaches_anchor(node, cell_anchors)) return build_cell_type(node, extends);
    if (reaches_anchor(node, synapse_anchors)) return build_synapse_type(node, extends);
    if (reaches_anchor(node, inputs_anchors)) return build_inputs_type(node, extends);
    if (reaches_anchor(node, population_anchors)) return build_population_type(node, extends);
    if (reaches_anchor(node, project_anchors)) return build_project_type(node, extends);

    return make_base(node, extends);
}

ComponentTypeEntry &NML_Parser::get_type_by_name(String name) {
    auto entry = library.find(name);
    if (entry == library.end()) {
        log::logger().error("NML Standard Library type {} could not be found.", name);
        throw std::runtime_error("NML Standard Library type could not be found.");
    }

    return entry->second;
}

bool NML_Parser::validate_against_schema(const String &nml_file_path) {
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
        log::logger().error("Could not create XSD validation context for schema {}", NML_SCHEMA_PATH);
        xmlSchemaFree(schema);
        return false;
    }

    // xmlSchemaValidateFile: 0 = valid, >0 = validation errors (already
    // printed to stderr by libxml2's default handler, located by element and
    // line number), <0 = internal/API error.
    int result = xmlSchemaValidateFile(valid_context, nml_file_path.c_str(), 0);

    xmlSchemaFreeValidCtxt(valid_context);
    xmlSchemaFree(schema);

    return result == 0;
}

}
