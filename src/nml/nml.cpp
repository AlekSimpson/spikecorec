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

        auto name_attribute = child.attributes.find("name");
        if (name_attribute == child.attributes.end()) {
            log::logger().warn("ComponentType in {} has no name attribute, skipping", nml_file_path);
            continue;
        }

        try {
            String type_name = std::any_cast<String>(name_attribute->second);
            library.emplace(type_name, std::move(child));
        } catch (const std::bad_any_cast &) {
            log::logger().error("ComponentType in {} has a non-string name attribute, skipping", nml_file_path);
        }
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

            auto name_attribute = component_type.attributes.find("name");
            if (name_attribute == component_type.attributes.end()) {
                log::logger().warn("ComponentType in {} has no name attribute, skipping", file_path);
                library_failed_to_load = true;
                continue;
            }

            String type_name;
            try {
                type_name = std::any_cast<String>(name_attribute->second);
            } catch (const std::bad_any_cast &) {
                log::logger().error("ComponentType in {} has a non-string name attribute, skipping", file_path);
                library_failed_to_load = true;
                continue;
            }

            library.emplace(type_name, component_type);

        }

        xmlFreeDoc(document);
    }

    xmlCleanupParser();
    return library_failed_to_load;
}

NML_Node &NML_Parser::get_type_by_name(String name) {
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
