#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xmlschemastypes.h>
#include <filesystem>
#include <unordered_set>

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

const NML_Node *find_child(const NML_Node &node, const String &tag) {
    for (const auto &child : node.body) {
        if (child.tag_name == tag) return &child;
    }
    return nullptr;
}

// xmlSchemaSetValidStructuredErrors callback: collects each reported validation error's line number
// and message (libxml2's error messages already name the offending element, e.g. "Element
// 'thisTagDoesNotExistInSchema': No matching global declaration available for the validation
// root.") into `*user_data`, joined by " | " -- see validate_against_schema's own comment for why
// this replaces relying on libxml2's default (stderr-only) error handler.
void collect_schema_validation_error(void *user_data, xmlErrorPtr error) {
    if (!error || !error->message) return;

    String message(error->message);
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) message.pop_back();

    String *destination = static_cast<String *>(user_data);
    if (!destination->empty()) *destination += " | ";
    *destination += "line " + std::to_string(error->line) + ": " + message;
}

// A `<Constant>` (e.g. `hindmarshRose1984Cell`'s `MSEC`, `fitzHughNagumoCell`'s `SEC`) declares a
// name + dimension exactly like a `<Parameter>` does, structurally -- the only difference is that
// its value is a fixed literal in the ComponentType itself (its own `value=` attribute) rather than
// supplied per bound instance. Cataloging it here as an ordinary ParameterDecl (ticket #63 [F2],
// needed to lower these real ComponentTypes at all) makes it a legal identifier everywhere a
// Parameter already is (known_names, `.alloc` ParamConstantDirective emission); resolve.cpp's own
// extract_fixed_decls separately synthesizes the `Fixed` pin that supplies its actual value, so nml.h
// gains no new decl struct for it.

// Every extractor below is a thin call into the shared extract_decls (nml.h) -- one line naming which
// tag(s) mirror this DeclType and how to build one from a matched child's attributes.

Vector<ParameterDecl> extract_parameters(const NML_Node &node) {
    return extract_decls<ParameterDecl>(node, {"Parameter", "Constant"}, [](const NML_Node &child) {
        return ParameterDecl{child.get_attr("name"), child.get_attr("dimension")};
    });
}

Vector<PropertyDecl> extract_properties(const NML_Node &node) {
    return extract_decls<PropertyDecl>(node, {"Property"}, [](const NML_Node &child) {
        return PropertyDecl{child.get_attr("name"), child.get_attr("dimension"), child.get_attr("defaultValue")};
    });
}

Vector<ExposureDecl> extract_exposures(const NML_Node &node) {
    return extract_decls<ExposureDecl>(node, {"Exposure"}, [](const NML_Node &child) {
        return ExposureDecl{child.get_attr("name"), child.get_attr("dimension")};
    });
}

Vector<EventPortDecl> extract_event_ports(const NML_Node &node) {
    return extract_decls<EventPortDecl>(node, {"EventPort"}, [](const NML_Node &child) {
        return EventPortDecl{child.get_attr("name"), child.get_attr("direction")};
    });
}

Vector<AttachmentDecl> extract_attachments(const NML_Node &node) {
    return extract_decls<AttachmentDecl>(node, {"Attachments"}, [](const NML_Node &child) {
        return AttachmentDecl{child.get_attr("name"), child.get_attr("type")};
    });
}

Vector<RequirementDecl> extract_requirements(const NML_Node &node) {
    return extract_decls<RequirementDecl>(node, {"Requirement"}, [](const NML_Node &child) {
        return RequirementDecl{child.get_attr("name"), child.get_attr("dimension")};
    });
}

// `ComponentReference`/`Link`, plus the electrical/graded-synapse and
// projection-family equivalents (`InstanceRequirement`, `ComponentRequirement`)
// — all "a reference to another component's type," unified into one list.
Vector<ComponentReferenceDecl> extract_component_references(const NML_Node &node) {
    return extract_decls<ComponentReferenceDecl>(
        node, {"ComponentReference", "Link", "InstanceRequirement", "ComponentRequirement"}, [](const NML_Node &child) {
            return ComponentReferenceDecl{child.get_attr("name"), child.get_attr("type")};
        });
}

Vector<ChildrenDecl> extract_children_decls(const NML_Node &node) {
    return extract_decls<ChildrenDecl>(node, {"Children", "Child"}, [](const NML_Node &child) {
        return ChildrenDecl{child.get_attr("name"), child.get_attr("type")};
    });
}

Vector<PathDecl> extract_path_fields(const NML_Node &node) {
    return extract_decls<PathDecl>(node, {"Path", "Text"}, [](const NML_Node &child) {
        return PathDecl{child.get_attr("name")};
    });
}

Vector<StateVariableDecl> extract_state_variables(const NML_Node &dynamics) {
    return extract_decls<StateVariableDecl>(dynamics, {"StateVariable"}, [](const NML_Node &child) {
        return StateVariableDecl{child.get_attr("name"), child.get_attr("dimension"), child.get_attr("exposure")};
    });
}

Vector<DerivedVariableDecl> extract_derived_variables(const NML_Node &dynamics) {
    return extract_decls<DerivedVariableDecl>(dynamics, {"DerivedVariable"}, [](const NML_Node &child) {
        return DerivedVariableDecl{
            child.get_attr("name"), child.get_attr("dimension"), child.get_attr("exposure"),
            child.get_attr("value"), child.get_attr("select"), child.get_attr("reduce")};
    });
}

Vector<TimeDerivativeDecl> extract_time_derivatives(const NML_Node &dynamics) {
    return extract_decls<TimeDerivativeDecl>(dynamics, {"TimeDerivative"}, [](const NML_Node &child) {
        return TimeDerivativeDecl{child.get_attr("variable"), child.get_attr("value")};
    });
}

Vector<OnConditionDecl> extract_on_conditions(const NML_Node &dynamics) {
    return extract_decls<OnConditionDecl>(dynamics, {"OnCondition"}, [](const NML_Node &child) {
        return OnConditionDecl{child.get_attr("test"), child};
    });
}

Vector<OnEventDecl> extract_on_events(const NML_Node &dynamics) {
    return extract_decls<OnEventDecl>(dynamics, {"OnEvent"}, [](const NML_Node &child) {
        return OnEventDecl{child.get_attr("port"), child};
    });
}

Vector<RegimeDecl> extract_regimes(const NML_Node &dynamics) {
    return extract_decls<RegimeDecl>(dynamics, {"Regime"}, [](const NML_Node &child) {
        return RegimeDecl{child.get_attr("name"), child.get_attr("initial"), child};
    });
}

Vector<OnStartDecl> extract_on_starts(const NML_Node &dynamics) {
    return extract_decls<OnStartDecl>(dynamics, {"OnStart"}, [](const NML_Node &child) {
        return OnStartDecl{child};
    });
}

ComponentTypeBase make_base(const NML_Node &node, const String &extends) {
    return ComponentTypeBase(node.get_attr("name"), extends, node);
}

CellType build_cell_type(const NML_Node &node, const String &extends) {
    CellType result(node.get_attr("name"), extends, node);

    result.parameters = extract_parameters(node);
    result.exposures = extract_exposures(node);
    result.event_ports = extract_event_ports(node);
    result.attachments = extract_attachments(node);

    if (const NML_Node *dynamics = find_child(node, "Dynamics")) {
        result.state_variables = extract_state_variables(*dynamics);
        result.derived_variables = extract_derived_variables(*dynamics);
        result.time_derivatives = extract_time_derivatives(*dynamics);
        result.on_conditions = extract_on_conditions(*dynamics);
        result.on_starts = extract_on_starts(*dynamics);
        result.regimes = extract_regimes(*dynamics);
    }

    return result;
}

SynapseType build_synapse_type(const NML_Node &node, const String &extends) {
    SynapseType result(node.get_attr("name"), extends, node);

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
        result.on_starts = extract_on_starts(*dynamics);
    }

    return result;
}

InputsType build_inputs_type(const NML_Node &node, const String &extends) {
    InputsType result(node.get_attr("name"), extends, node);

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
        result.on_starts = extract_on_starts(*dynamics);
    }

    return result;
}

PopulationType build_population_type(const NML_Node &node, const String &extends) {
    PopulationType result(node.get_attr("name"), extends, node);

    result.component_references = extract_component_references(node);
    result.parameters = extract_parameters(node);
    result.children = extract_children_decls(node);

    return result;
}

ProjectType build_project_type(const NML_Node &node, const String &extends) {
    ProjectType result(node.get_attr("name"), extends, node);

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

String NML_Node::get_attr(const String &name) const {
    auto entry = attributes.find(name);
    if (entry == attributes.end()) return "";
    return std::any_cast<String>(entry->second);
}

void NML_Parser::parse(const String &nml_input_file) {
    bool failed_to_load = load_standard_library();
    if (failed_to_load) {
        log::logger().error("Failed to load NML standard library. It may be missing or malformed.");
        throw runtime_error("Failed to load NML standard library. It may be missing or malformed.");
    }

    ingest_file(nml_input_file, true);

    // ingest_file's own recursion fully resolves the whole <include> graph
    // before returning, so every ComponentType this run will ever see (std
    // lib + every included file) is now in raw_component_types — safe to
    // classify the newly-ingested user/include types (the std-lib ones are
    // already classified from inside load_standard_library() and are left
    // untouched). Once done, raw_component_types has served its only purpose
    // for this run, so it's dropped rather than left holding a second copy
    // of every node library's entries already own.
    classify_all_cataloged_types();
    raw_component_types.clear();

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
        String detail = last_schema_validation_errors.empty() ? "" : (": " + last_schema_validation_errors);
        log::logger().error("{} does not validate against the NeuroML2 XSD schema{}", nml_file_path, detail);
        throw std::runtime_error(nml_file_path + " does not validate against the NeuroML2 XSD schema" + detail);
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

        catalog_raw_component_type(child, nml_file_path);
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

            if (!catalog_raw_component_type(component_type, file_path)) {
                library_failed_to_load = true;
            }
        }

        xmlFreeDoc(document);
    }

    // The directory scan above (across every std-lib file, in whatever order
    // std::filesystem::directory_iterator happens to visit them) is fully
    // done at this point, so every std-lib ComponentType is now in
    // raw_component_types — safe to classify the whole batch in one pass.
    classify_all_cataloged_types();

    xmlCleanupParser();
    return library_failed_to_load;
}

// Records a parsed `<ComponentType>` node's raw form into
// `raw_component_types`. Duplicate names are dropped silently
// (first-cataloged-wins, matching the original flat-library behavior) —
// that is not a failure. Classification is deferred — see
// classify_all_cataloged_types.
bool NML_Parser::catalog_raw_component_type(const NML_Node &component_type_node, const String &source_path) {
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

    raw_component_types.emplace(type_name, component_type_node);
    return true;
}

// Classifies every node in raw_component_types not yet in `library`. Called
// only at the end of a whole ingestion phase (load_standard_library()'s full
// directory scan; parse()'s whole <include> graph), so every call sees every
// node cataloged during that phase — the fix for the order-dependent
// misclassification eager per-node classification had, since a chain walk
// can now never give up on a parent merely because it hasn't been reached
// yet. Deliberately additive (never clears/rebuilds `library`): a type
// classified correctly on an earlier call (e.g. by load_standard_library(),
// before parse() goes on to ingest more files) stays classified without its
// entry's address changing — re-running classify_component_type on it here
// would just recompute the identical result, so it's skipped instead.
void NML_Parser::classify_all_cataloged_types() {
    for (const auto &[type_name, raw_node] : raw_component_types) {
        if (library.find(type_name) != library.end()) continue;
        library.emplace(type_name, classify_component_type(raw_node));
    }
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
    // Membership-only, unordered lookups -- no anchor set is ever iterated in a particular order or
    // relied on for its declaration order, so an unordered_set (rather than a Vector<String> linearly
    // scanned by hand) is both the right shape and gives reaches_anchor O(1) lookups instead of O(n).
    static const unordered_set<String> cell_anchors = {"baseCell"};
    static const unordered_set<String> synapse_anchors = {"baseSynapse"};
    static const unordered_set<String> inputs_anchors = {"basePointCurrent", "baseSpikeSource"};
    static const unordered_set<String> population_anchors = {"basePopulation", "population", "populationList"};
    static const unordered_set<String> project_anchors = {
        "projection", "connection", "connectionWD", "explicitConnection",
        "synapticConnection", "synapticConnectionWD",
        "electricalConnection", "electricalConnectionInstance", "electricalConnectionInstanceW",
        "electricalProjection",
        "continuousConnection", "continuousConnectionInstance", "continuousConnectionInstanceW",
        "continuousProjection"};

    // Walks from `node` up through successive `extends` lookups (bounded to
    // guard against a cyclic chain), returning true the moment any name in
    // the chain (including `node` itself) matches one of `anchors`.
    auto reaches_anchor = [this](const NML_Node &start, const unordered_set<String> &anchors) {
        String current_name = start.get_attr("name");
        const NML_Node *current_node = &start;

        for (int hop = 0; hop < 64 && !current_name.empty(); ++hop) {
            if (anchors.count(current_name) > 0) return true;

            String parent_name = current_node->get_attr("extends");
            if (parent_name.empty()) return false;

            auto parent_entry = raw_component_types.find(parent_name);
            if (parent_entry == raw_component_types.end()) {
                return anchors.count(parent_name) > 0;
            }

            current_name = parent_name;
            current_node = &parent_entry->second;
        }
        return false;
    };

    String extends = node.get_attr("extends");

    if (reaches_anchor(node, cell_anchors)) return build_cell_type(node, extends);
    if (reaches_anchor(node, synapse_anchors)) return build_synapse_type(node, extends);
    if (reaches_anchor(node, inputs_anchors)) return build_inputs_type(node, extends);
    if (reaches_anchor(node, population_anchors)) return build_population_type(node, extends);
    if (reaches_anchor(node, project_anchors)) return build_project_type(node, extends);

    return make_base(node, extends);
}

ComponentTypeEntry &NML_Parser::get_type_by_name(const String &name) {
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

    // Route every validation error through collect_schema_validation_error instead of libxml2's
    // default handler (which only prints to stderr, invisible to both the caller's exception and
    // this codebase's own logger) -- populates last_schema_validation_errors with each error's line
    // number and message (already naming the offending element) so ingest_file's thrown exception
    // can report exactly what failed and where.
    last_schema_validation_errors.clear();
    xmlSchemaSetValidStructuredErrors(valid_context, collect_schema_validation_error, &last_schema_validation_errors);

    // xmlSchemaValidateFile: 0 = valid, >0 = validation errors (captured above, by element and line
    // number), <0 = internal/API error.
    int result = xmlSchemaValidateFile(valid_context, nml_file_path.c_str(), 0);

    xmlSchemaFreeValidCtxt(valid_context);
    xmlSchemaFree(schema);

    return result == 0;
}

}
