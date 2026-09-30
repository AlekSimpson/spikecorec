#pragma once

#include <libxml/parser.h>
#include <libxml/tree.h>

#include "spikecorec/nml/dynamics.h"

#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"
#include "spikecorec/core/recording.h"

#include "spikecorec/nml/node.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {
struct NML_ComponentType {
    String name;
    const NML_ComponentType *extends = nullptr;
    String source_file;

    // The <ComponentType> element in the parser's document tree (not owned).
    const NML_Node *source_node = nullptr;

    Vector<NML_DynamicsExpression> dynamics;

    // Per-neuron state slots in order, inherited ones first. Filled by parse_component_types.
    Vector<String> state_variable_names;

    NML_ComponentType() = default;
    ~NML_ComponentType() = default;

    NML_ComponentType(const NML_ComponentType &other) = default;
    NML_ComponentType &operator=(const NML_ComponentType &other) = default;

    NML_ComponentType(NML_ComponentType &&other) noexcept = default;
    NML_ComponentType &operator=(NML_ComponentType &&other) noexcept = default;

    NML_ComponentType(String name)
        : name(std::move(name)) {};

    // The declaration this type resolves `namespace_key` (see NML_Tag::namespace_key)
    // to: the first match in its own element, searched depth-first in document order,
    // otherwise the nearest ancestor's. A subtype's declaration therefore shadows an
    // ancestor's of the same key. nullptr when no type in the chain declares it, and for
    // an empty key, which accumulating tags such as OnCondition have.
    const NML_Node *find_declaration(const String &namespace_key) const;
};

struct NML_ComponentInstance {
    UnorderedMap<String, String> instance_data;
    Vector<String> data_order; // list of child instance ids, in source order

    const NML_ComponentInstance *parent_instance = nullptr;
    const NML_ComponentType *component_type = nullptr;

    String id;

    NML_ComponentInstance() = default;
    ~NML_ComponentInstance() = default;

    NML_ComponentInstance(const NML_ComponentInstance &other) = default;
    NML_ComponentInstance &operator=(const NML_ComponentInstance &other) = default;

    NML_ComponentInstance(NML_ComponentInstance &&other) noexcept = default;
    NML_ComponentInstance &operator=(NML_ComponentInstance &&other) noexcept = default;

    String &get_value(String &key);

    bool has_value(const String &key) const;
    String value_or(const String &key, const String &fallback = "") const;
};

}
