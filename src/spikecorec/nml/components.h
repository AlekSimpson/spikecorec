#pragma once

#include <libxml/parser.h>
#include <libxml/tree.h>

#include "dynamics.h"

#include "units.h"
#include "types.h"
#include "recording.h"

#include "node.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {
struct NML_Component_Type {
    String name;
    const NML_Component_Type *extends = nullptr;
    String source_file;

    // The <ComponentType> element in the parser's document tree (not owned).
    const NML_Node *source_node = nullptr;

    Vector<NML_Dynamics_Expression> dynamics;
    // The nearest <Structure> in the extends chain, like dynamics.
    Vector<NML_Structure_Entry> structure;

    // State variable names in slot order, inherited ones first. Filled by parse_component_types.
    Vector<String> state_variable_names;

    NML_Component_Type() = default;
    ~NML_Component_Type() = default;

    NML_Component_Type(const NML_Component_Type &other) = default;
    NML_Component_Type &operator=(const NML_Component_Type &other) = default;

    NML_Component_Type(NML_Component_Type &&other) noexcept = default;
    NML_Component_Type &operator=(NML_Component_Type &&other) noexcept = default;

    NML_Component_Type(String name)
        : name(std::move(name)) {};

    // The declaration namespace_key names (see NML_Tag::namespace_key): the first depth-first
    // match in this type's element, else the nearest ancestor's, so a subtype shadows its
    // ancestors. nullptr when nothing matches or the key is empty (as for OnCondition).
    const NML_Node *find_declaration(const String &namespace_key) const;
};

struct NML_Component_Instance {
    Unordered_Map<String, String> instance_data;
    Vector<String> data_order; // child instance ids, in document order

    const NML_Component_Instance *parent_instance = nullptr;
    const NML_Component_Type *component_type = nullptr;

    String id;

    NML_Component_Instance() = default;
    ~NML_Component_Instance() = default;

    NML_Component_Instance(const NML_Component_Instance &other) = default;
    NML_Component_Instance &operator=(const NML_Component_Instance &other) = default;

    NML_Component_Instance(NML_Component_Instance &&other) noexcept = default;
    NML_Component_Instance &operator=(NML_Component_Instance &&other) noexcept = default;

    String &get_value(String &key);

    bool has_value(const String &key) const;
    String value_or(const String &key, const String &fallback = "") const;
};

}
