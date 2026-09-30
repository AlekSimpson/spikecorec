
#include <libxml/parser.h>
#include <libxml/tree.h>

#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"
#include "spikecorec/nml/node.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

NML_DeclarationType string_to_declaration_type(const String &value) {
    static const UnorderedMap<String, NML_DeclarationType> mapping = {
        {"Parameter", NML_DeclarationType::Parameter},
        {"DerivedParameter", NML_DeclarationType::DerivedParameter},
        {"Constant", NML_DeclarationType::Constant},
        {"Requirement", NML_DeclarationType::Requirement},
        {"Exposure", NML_DeclarationType::Exposure},
        {"Property", NML_DeclarationType::Property},
        {"Fixed", NML_DeclarationType::Fixed},
        {"Text", NML_DeclarationType::Text},
        {"EventPort", NML_DeclarationType::EventPort},
        {"Attachments", NML_DeclarationType::Attachment},
        {"ComponentReference", NML_DeclarationType::ComponentReference},
        {"Link", NML_DeclarationType::Link},
        {"Children", NML_DeclarationType::Children},
        {"Child", NML_DeclarationType::Child},
        {"Path", NML_DeclarationType::Path},
        {"StateVariable", NML_DeclarationType::StateVariable},
        {"DerivedVariable", NML_DeclarationType::DerivedVariable},
        {"ConditionalDerivedVariable", NML_DeclarationType::ConditionalDerivedVariable},
        {"Case", NML_DeclarationType::Case},
        {"TimeDerivative", NML_DeclarationType::TimeDerivative},
        {"OnCondition", NML_DeclarationType::OnCondition},
        {"OnEvent", NML_DeclarationType::OnEvent},
        {"OnStart", NML_DeclarationType::OnStart},
        {"OnEntry", NML_DeclarationType::OnEntry},
        {"StateAssignment", NML_DeclarationType::StateAssignment},
        {"EventOut", NML_DeclarationType::EventOut},
        {"Transition", NML_DeclarationType::Transition},
        {"Regime", NML_DeclarationType::Regime},
    };

    auto entry = mapping.find(value);
    if (entry == mapping.end()) return NML_DeclarationType::NOT_A_TYPE;

    return entry->second;
}

// NML_Tag implementation

NML_Tag::NML_Tag(String tag_name)
    : tag_name(std::move(tag_name)),
      tag_type(string_to_declaration_type(this->tag_name)) {}

NML_Tag::NML_Tag(xmlNodePtr xml_node)
    : NML_Tag(String(reinterpret_cast<const char *>(xml_node->name))) {
    for (xmlAttrPtr attribute = xml_node->properties; attribute; attribute = attribute->next) {
        xmlChar *value = xmlGetProp(xml_node, attribute->name);
        if (!value) continue;

        add_attribute(reinterpret_cast<const char *>(attribute->name),
                      String(reinterpret_cast<const char *>(value)));
        xmlFree(value);
    }
}

bool NML_Tag::is_declaration_type() const {
    return tag_type != NML_DeclarationType::NOT_A_TYPE;
}

bool NML_Tag::has_attribute(const String &name) const {
    return attributes.find(name) != attributes.end();
}

void NML_Tag::add_attribute(const String &name, String value) {
    attributes[name] = std::move(value);
}

String NML_Tag::get_attribute(const String &name) const {
    return get_attribute_or(name, "");
}

String NML_Tag::get_attribute_or(const String &name, const String &fallback) const {
    auto entry = attributes.find(name);
    if (entry == attributes.end()) return fallback;
    return entry->second;
}

String NML_Tag::namespace_key() const {
    // The namespaces a ComponentType keeps names unique within, per
    // docs/nml_general_notes.md. Several tags share the variable namespace, so a subtype
    // declaring StateVariable "tau" overrides an inherited Parameter "tau".
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
            return "var:" + get_attribute("name");

        case NML_DeclarationType::Exposure:
            return "exposure:" + get_attribute("name");

        case NML_DeclarationType::EventPort:
            return "port:" + get_attribute("name");

        case NML_DeclarationType::Child:
        case NML_DeclarationType::Children:
        case NML_DeclarationType::Attachment:
        case NML_DeclarationType::ComponentReference:
        case NML_DeclarationType::Link:
        case NML_DeclarationType::Path:
            return "pathsegment:" + get_attribute("name");

        case NML_DeclarationType::Regime:
            return "regime:" + get_attribute("name");

        // A subtype restating a derivative for the same variable replaces it rather than
        // integrating the variable twice.
        case NML_DeclarationType::TimeDerivative:
            return "derivative:" + get_attribute("variable");

        // Everything else accumulates: a subtype adding an OnCondition adds a condition,
        // it does not replace the parent's.
        default:
            return "";
    }
}

// NML tree construction

NML_Node *build_nml_tree(xmlNodePtr xml_node) {
    NML_Tag tag(xml_node);

    bool has_element_children = false;
    for (xmlNodePtr child = xml_node->children; child; child = child->next) {
        if (child->type != XML_ELEMENT_NODE) continue;
        has_element_children = true;
        break;
    }

    if (!has_element_children) return new NML_Node(std::move(tag));

    auto *list_node = new ListNode<NML_Tag>(std::move(tag));
    for (xmlNodePtr child = xml_node->children; child; child = child->next) {
        if (child->type != XML_ELEMENT_NODE) continue;
        list_node->children.push_back(build_nml_tree(child));
    }

    return list_node;
}

}

