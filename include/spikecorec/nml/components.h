#pragma once

#include <libxml/parser.h>
#include <libxml/tree.h>

#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"
#include "spikecorec/core/recording.h"

#include "spikecorec/nml/declarations.h"
#include "spikecorec/nml/dynamics.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {
struct NML_ComponentType {
    NML_DeclarationList declarations;
    String name;
    String extends; // parent type name, empty for a root type
    String source_file;
    NML_Node source_node;
    Vector<NML_DynamicsExpresion> dynamics;

    NML_ComponentType() = default;
    ~NML_ComponentType() = default;

    NML_ComponentType(const NML_ComponentType &other) = default;
    NML_ComponentType &operator=(const NML_ComponentType &other) = default;

    NML_ComponentType(NML_ComponentType &&other) noexcept = default;
    NML_ComponentType &operator=(NML_ComponentType &&other) noexcept = default;

    NML_ComponentType(String name)
        : name(std::move(name)) {};
    NML_ComponentType(String name, NML_DeclarationList list)
        : declarations(std::move(list)),
          name(std::move(name)) {};
};

struct NML_ComponentInstance {
    UnorderedMap<String, String> instance_data;
    Vector<String> structured_instance_data; // list of child instance ids, in source order
    String id;
    String parent_instance_id; // empty for a document-scope instance
    const NML_ComponentType *component_type = nullptr;

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
