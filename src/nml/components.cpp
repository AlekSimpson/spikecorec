#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xmlschemastypes.h>
#include <filesystem>

#include "spikecorec/nml/components.h"
#include "spikecorec/nml/dynamics.h"
#include "spikecorec/nml/node.h"
#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"

#include <algorithm>
#include <cmath>

using namespace std;
using namespace spikecorec;
using namespace std::filesystem;

namespace spikecorec::nml {

namespace {

const NML_Node *find_declaration_in_subtree(const NML_Node *node, const String &namespace_key) {
    for (const NML_Node *child : children_of(node)) {
        if (child->body.is_declaration_type() && child->body.namespace_key() == namespace_key) {
            return child;
        }

        const NML_Node *nested = find_declaration_in_subtree(child, namespace_key);
        if (nested) return nested;
    }
    return nullptr;
}

} // namespace

const NML_Node *NML_ComponentType::find_declaration(const String &namespace_key) const {
    if (namespace_key.empty()) return nullptr;

    for (const NML_ComponentType *type = this; type; type = type->extends) {
        const NML_Node *found = find_declaration_in_subtree(type->source_node, namespace_key);
        if (found) return found;
    }
    return nullptr;
}

String &NML_ComponentInstance::get_value(String &key) {
    if (instance_data.find(key) == instance_data.end()) {
        static String missing_value = "";
        return missing_value;
    }

    return instance_data[key];
}

bool NML_ComponentInstance::has_value(const String &key) const {
    return instance_data.find(key) != instance_data.end();
}

String NML_ComponentInstance::value_or(const String &key, const String &fallback) const {
    auto entry = instance_data.find(key);
    if (entry == instance_data.end()) return fallback;
    return entry->second;
}



}
