#pragma once

#include <libxml/parser.h>
#include <libxml/tree.h>

#include "spikecorec/core/types.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {
// deliverables:
//  - A lookup API the resolve pass calls to fetch a core ComponentType / include target by name/href.
//  - Document how the bundle is refreshed and where the version is pinned.

struct NML_Node {
    String tag_name;

    NML_Node(String tag_name) : tag_name(tag_name) {};

    NML_Node(const NML_Node &other) : tag_name(other.tag_name), attributes(other.attributes), body(other.body) {};

    UnorderedMap<String, Any> attributes;
    Vector<NML_Node> body;

    void add_attribute(String name, Any value);
    void nest(NML_Node component);
};


#ifndef SPIKECOREC_NML_STD_LIB_DIR
#define SPIKECOREC_NML_STD_LIB_DIR ""
#endif

#ifndef SPIKECOREC_NML_SCHEMA_PATH
#define SPIKECOREC_NML_SCHEMA_PATH ""
#endif

struct NML_Parser {
    UnorderedMap<String, NML_Node> library;
    NML_Node *root_node = nullptr;

    const String STANDARD_LIBRARY_PATH = SPIKECOREC_NML_STD_LIB_DIR;
    const String NML_SCHEMA_PATH = SPIKECOREC_NML_SCHEMA_PATH;

    NML_Parser() {};

    bool validate_against_schema(const String &nml_file_path);

    void parse(const String &nml_input_file);

    void ingest_file(const String &nml_file_path, bool run_schema_validation);

    NML_Node xml_node_to_nml_node(xmlNodePtr node);

    bool load_standard_library();

    NML_Node &get_type_by_name(String name);
};




}
