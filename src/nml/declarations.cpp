#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/xmlschemastypes.h>
#include <filesystem>

#include "spikecorec/nml/declarations.h"
#include "spikecorec/nml/node.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/units.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace std;
using namespace spikecorec;
using namespace std::filesystem;

namespace spikecorec::nml {

// NML Declaration related implementations
bool NML_Declaration::has_value(const String &key) const {
    return datavalues.find(key) != datavalues.end();
}

String NML_Declaration::get_value(const String &key) const {
    auto entry = datavalues.find(key);
    if (entry == datavalues.end()) {
        String message = "Could not find key: " + key + " in neuroml declaration.";
        throw out_of_range(message);
    }

    return entry->second;
}

String NML_Declaration::value_or(const String &key, const String &fallback) const {
    auto entry = datavalues.find(key);
    if (entry == datavalues.end()) return fallback;
    return entry->second;
}

String NML_Declaration::namespace_key() const {
    // The namespaces a ComponentType keeps names unique within, per
    // docs/nml_general_notes.md. Seven tags share `variables`, so a subtype declaring
    // StateVariable "tau" overrides an inherited Parameter "tau".
    switch (tag_type) {
        case NML_DeclarationType::Parameter:
        case NML_DeclarationType::DerivedParameter:
        case NML_DeclarationType::Constant:
        case NML_DeclarationType::Requirement:
        case NML_DeclarationType::Property:
        case NML_DeclarationType::Text:
        case NML_DeclarationType::StateVariable:
        case NML_DeclarationType::DerivedVariable:
        case NML_DeclarationType::ConditionalDerivedVariable:
            return "var:" + value_or("name");

        case NML_DeclarationType::Exposure:
            return "exposure:" + value_or("name");

        case NML_DeclarationType::EventPort:
            return "port:" + value_or("name");

        case NML_DeclarationType::Child:
        case NML_DeclarationType::Children:
        case NML_DeclarationType::Attachment:
        case NML_DeclarationType::ComponentReference:
        case NML_DeclarationType::Link:
        case NML_DeclarationType::Path:
            return "pathsegment:" + value_or("name");

        case NML_DeclarationType::Regime:
            return "regime:" + value_or("name");

        // A subtype restating a derivative for the same variable replaces it rather than
        // integrating the variable twice.
        case NML_DeclarationType::TimeDerivative:
            return "derivative:" + value_or("variable");

        // Everything else accumulates: a subtype adding an OnCondition adds a condition,
        // it does not replace the parent's.
        default:
            return "";
    }
}

void DeclarationList::insert(const NML_Declaration &declaration) {
    auto comparator = [](const NML_Declaration &current, const NML_Declaration &next) {
        return current.tag_type < next.tag_type;
    };
    // upper_bound, not lower_bound: lower_bound returns the first element of an equal
    // tag_type group, so inserting there puts each new declaration ahead of its
    // predecessors and reverses source order within the group. Parameter order is the
    // column order of the exported starting-parameter rows, so that ordering is
    // load-bearing.
    auto index = std::upper_bound(
            declarations.begin(), declarations.end(),
            declaration,
            comparator);

    declarations.insert(index, declaration);
}

void DeclarationList::overlay(const NML_Declaration &declaration) {
    String key = declaration.namespace_key();

    // An empty key means this tag accumulates rather than overrides.
    if (key.empty()) {
        insert(declaration);
        return;
    }

    for (usize index = 0; index < declarations.size(); index += 1) {
        if (declarations[index].namespace_key() != key) continue;

        // Same tag_type keeps the list's grouping intact, so the entry is replaced where
        // it sits and the ancestor's position in declaration order is inherited. A
        // different tag_type (a StateVariable overriding a Parameter) has to move groups.
        if (declarations[index].tag_type == declaration.tag_type) {
            declarations[index] = declaration;
            return;
        }

        declarations.erase(declarations.begin() + static_cast<s64>(index));
        insert(declaration);
        return;
    }

    insert(declaration);
}

const NML_Declaration *DeclarationList::find_first(NML_DeclarationType type,
                                             const String &name) const {
    for (const NML_Declaration &declaration : declarations) {
        if (declaration.tag_type != type) continue;
        if (declaration.value_or("name") == name) return &declaration;
    }
    return nullptr;
}

Vector<const NML_Declaration *> DeclarationList::find_all(NML_DeclarationType type) const {
    Vector<const NML_Declaration *> matches;
    for (const NML_Declaration &declaration : declarations) {
        if (declaration.tag_type != type) continue;
        matches.push_back(&declaration);
    }
    return matches;
}


}
