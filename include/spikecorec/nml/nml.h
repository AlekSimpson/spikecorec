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

struct ComponentType {
    String tag;

    ComponentType(String tag) : tag(tag) {};

    ComponentType(const ComponentType &other) : tag(other.tag), attributes(other.attributes), body(other.body) {};

    UnorderedMap<String, Any> attributes;
    Vector<ComponentType> body;

    void add_attribute(String name, Any value);
    void nest(ComponentType component);
};

ComponentType xml_node_to_component_type(xmlNodePtr node);

#ifndef SPIKECOREC_NML_STD_LIB_DIR
#define SPIKECOREC_NML_STD_LIB_DIR ""
#endif

struct NML_StandardLibrary {
    UnorderedMap<String, ComponentType> standard_library;

    const String STANDARD_LIBRARY_PATH = SPIKECOREC_NML_STD_LIB_DIR;

    NML_StandardLibrary() {};

    bool load_library();

    ComponentType &get_type_by_name(String name);
};

// Acceptance criteria
//     Core types resolve fully offline; version is pinned and recorded.
//     Merge-point API is documented and used by the resolve pass (A4).




}
