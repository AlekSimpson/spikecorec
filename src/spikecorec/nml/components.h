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

    // Per-neuron state slots in order, inherited ones first. Filled by parse_component_types.
    Vector<String> state_variable_names;

    NML_Component_Type() = default;
    ~NML_Component_Type() = default;

    NML_Component_Type(const NML_Component_Type &other) = default;
    NML_Component_Type &operator=(const NML_Component_Type &other) = default;

    NML_Component_Type(NML_Component_Type &&other) noexcept = default;
    NML_Component_Type &operator=(NML_Component_Type &&other) noexcept = default;

    NML_Component_Type(String name)
        : name(std::move(name)) {};

    // The declaration this type resolves `namespace_key` (see NML_Tag::namespace_key)
    // to: the first match in its own element, searched depth-first in document order,
    // otherwise the nearest ancestor's. A subtype's declaration therefore shadows an
    // ancestor's of the same key. nullptr when no type in the chain declares it, and for
    // an empty key, which accumulating tags such as OnCondition have.
    const NML_Node *find_declaration(const String &namespace_key) const;
};

struct NML_Component_Instance {
    Unordered_Map<String, String> instance_data;
    Vector<String> data_order; // list of child instance ids, in source order

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
