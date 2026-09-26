
#include <libxml/parser.h>
#include <libxml/tree.h>

#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"
#include "spikecorec/nml/declarations.h"

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

// NML_Node implementation

NML_Node::NML_Node(xmlNodePtr xml_node) {
    tag_name = reinterpret_cast<const char *>(node->name);
    
    for (xmlAttrPtr attribute = node->properties; attribute; attribute = attribute->next) {
        xmlChar *value = xmlGetProp(node, attribute->name);
        if (!value) continue;

        add_attribute(reinterpret_cast<const char *>(attribute->name),
                      String(reinterpret_cast<const char *>(value)));
        xmlFree(value);
    }

    for (xmlNodePtr child = node->children; child; child = child->next) {
        if (child->type != XML_ELEMENT_NODE) continue;
        nest(NML_Node(child));
    }
}

bool NML_Node::is_declaration_type() const {
    return string_to_declaration_type(value) != NML_DeclarationType::NOT_A_TYPE;
}

bool NML_Node::has_attribute(const String &name) const {
    return attributes.find(name) != attributes.end();
}

void NML_Node::add_attribute(const String &name, String value) {
    attributes[name] = std::move(value);
}

void NML_Node::nest(NML_Node component) {
    body.push_back(std::move(component));
}

String NML_Node::get_attribute(const String &name) const {
    auto entry = attributes.find(name);
    if (entry == attributes.end()) return "";
    return entry->second;
}

NML_Declaration NML_Node::to_declaration() const {
    NML_Declaration declaration(string_to_declaration_type(tag_name), attributes);

    for (const NML_Node &child : body) {
        if (!child.is_declaration_type()) continue;
        declaration.children.push_back(child.to_declaration());
    }

    return declaration;
}


}

