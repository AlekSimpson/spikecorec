
#include <libxml/parser.h>
#include <libxml/tree.h>

#include "units.h"
#include "types.h"
#include "node.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

NML_Declaration_Type string_to_declaration_type(const String &value) {
    static const Unordered_Map<String, NML_Declaration_Type> mapping = {
        {"Parameter", NML_Declaration_Type::Parameter},
        {"DerivedParameter", NML_Declaration_Type::DerivedParameter},
        {"Constant", NML_Declaration_Type::Constant},
        {"Requirement", NML_Declaration_Type::Requirement},
        {"Exposure", NML_Declaration_Type::Exposure},
        {"Property", NML_Declaration_Type::Property},
        {"Fixed", NML_Declaration_Type::Fixed},
        {"Text", NML_Declaration_Type::Text},
        {"EventPort", NML_Declaration_Type::EventPort},
        {"Attachments", NML_Declaration_Type::Attachment},
        {"ComponentReference", NML_Declaration_Type::ComponentReference},
        {"Link", NML_Declaration_Type::Link},
        {"Children", NML_Declaration_Type::Children},
        {"Child", NML_Declaration_Type::Child},
        {"Path", NML_Declaration_Type::Path},
        {"StateVariable", NML_Declaration_Type::StateVariable},
        {"DerivedVariable", NML_Declaration_Type::DerivedVariable},
        {"ConditionalDerivedVariable", NML_Declaration_Type::ConditionalDerivedVariable},
        {"Case", NML_Declaration_Type::Case},
        {"TimeDerivative", NML_Declaration_Type::TimeDerivative},
        {"OnCondition", NML_Declaration_Type::OnCondition},
        {"OnEvent", NML_Declaration_Type::OnEvent},
        {"OnStart", NML_Declaration_Type::OnStart},
        {"OnEntry", NML_Declaration_Type::OnEntry},
        {"StateAssignment", NML_Declaration_Type::StateAssignment},
        {"EventOut", NML_Declaration_Type::EventOut},
        {"Transition", NML_Declaration_Type::Transition},
        {"Regime", NML_Declaration_Type::Regime},
        {"ChildInstance", NML_Declaration_Type::ChildInstance},
        {"With", NML_Declaration_Type::With},
        {"EventConnection", NML_Declaration_Type::EventConnection},
    };

    auto entry = mapping.find(value);
    if (entry == mapping.end()) return NML_Declaration_Type::NOT_A_TYPE;

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
    return tag_type != NML_Declaration_Type::NOT_A_TYPE;
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
    // Names are unique per namespace. Several tags share the variable namespace, so a subtype
    // declaring StateVariable "tau" overrides an inherited Parameter "tau".
    switch (tag_type) {
        case NML_Declaration_Type::Parameter:
        case NML_Declaration_Type::DerivedParameter:
        case NML_Declaration_Type::Constant:
        case NML_Declaration_Type::Requirement:
        case NML_Declaration_Type::Property:
        case NML_Declaration_Type::Text:
        case NML_Declaration_Type::StateVariable:
        case NML_Declaration_Type::DerivedVariable:
        case NML_Declaration_Type::ConditionalDerivedVariable:
            return "var:" + get_attribute("name");

        case NML_Declaration_Type::Exposure:
            return "exposure:" + get_attribute("name");

        case NML_Declaration_Type::EventPort:
            return "port:" + get_attribute("name");

        case NML_Declaration_Type::Child:
        case NML_Declaration_Type::Children:
        case NML_Declaration_Type::Attachment:
        case NML_Declaration_Type::ComponentReference:
        case NML_Declaration_Type::Link:
        case NML_Declaration_Type::Path:
            return "pathsegment:" + get_attribute("name");

        case NML_Declaration_Type::Regime:
            return "regime:" + get_attribute("name");

        // A subtype's derivative of the same variable replaces the parent's, not adds to it.
        case NML_Declaration_Type::TimeDerivative:
            return "derivative:" + get_attribute("variable");

        // Everything else accumulates: a subtype's OnCondition adds to the parent's.
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

    auto *list_node = new List_Node<NML_Tag>(std::move(tag));
    for (xmlNodePtr child = xml_node->children; child; child = child->next) {
        if (child->type != XML_ELEMENT_NODE) continue;
        list_node->children.push_back(build_nml_tree(child));
    }

    return list_node;
}

}

