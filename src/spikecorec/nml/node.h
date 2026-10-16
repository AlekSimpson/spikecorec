#pragma once

#include <type_traits>
#include <utility>

#include <libxml/parser.h>
#include <libxml/tree.h>

#include "units.h"
#include "types.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

enum class NML_Declaration_Type {
    NamedDimension,
    Parameter,
    DerivedParameter,
    Constant,
    Requirement,
    Exposure,
    Property,
    Fixed,
    Text,
    EventPort,
    NamedTypeReference,
    Attachment,
    ComponentReference,
    Link,
    Children,
    Child,
    Path,
    StateVariable,
    DerivedVariable,
    ConditionalDerivedVariable,
    Case,
    TimeDerivative,
    OnCondition,
    OnEvent,
    OnStart,
    OnEntry,
    StateAssignment,
    EventOut,
    Transition,
    Regime,
    ChildInstance,
    With,
    EventConnection,

    NOT_A_TYPE
};

enum class Parse_Node_Type {
    NML_NODE,
    LEMS_NODE,
    KERNEL_NODE,
};

template <typename Syntax_Type, typename Token_Type>
struct Parse_Body {
    Parse_Node_Type parse_type{};
    Syntax_Type syntax_type{}; // the node's kind within parse_type, ex: a Kernel_Node_Type
    Token_Type token{};

    Parse_Body() = default;
    Parse_Body(Parse_Node_Type parse_type, Syntax_Type syntax_type, Token_Type token)
        : parse_type(parse_type), syntax_type(syntax_type), token(std::move(token)) {};
};

template <typename Body>
struct Node {
    Body body;

    Node() = default;
    explicit Node(Body body) : body(std::move(body)) {};
    virtual ~Node() = default;

    Node(const Node &other) = delete;
    Node &operator=(const Node &other) = delete;
};

template <typename Body>
struct Unary_Node : Node<Body> {
    Node<Body> *child = nullptr;

    explicit Unary_Node(Body body) : Node<Body>(std::move(body)) {};
    ~Unary_Node() override { delete child; };
};

template <typename Body>
struct Binary_Node : Node<Body> {
    Node<Body> *left = nullptr;
    Node<Body> *right = nullptr;

    explicit Binary_Node(Body body) : Node<Body>(std::move(body)) {};
    ~Binary_Node() override {
        delete left;
        delete right;
    };
};

template <typename Body>
struct Trinary_Node : Node<Body> {
    Node<Body> *left = nullptr;
    Node<Body> *middle = nullptr;
    Node<Body> *right = nullptr;

    explicit Trinary_Node(Body body) : Node<Body>(std::move(body)) {};
    ~Trinary_Node() override {
        delete left;
        delete middle;
        delete right;
    };
};

template <typename Body>
struct List_Node : Node<Body> {
    Vector<Node<Body> *> children;

    explicit List_Node(Body body) : Node<Body>(std::move(body)) {};
    ~List_Node() override {
        for (Node<Body> *child : children) delete child;
    };
};

// The children of `node`, or an empty list when it is a leaf (not a List_Node) or null.
template <typename Body>
const Vector<Node<Body> *> &children_of(const Node<Body> *node) {
    static const Vector<Node<Body> *> no_children;

    const auto *list_node = dynamic_cast<const List_Node<Body> *>(node);
    if (!list_node) return no_children;
    return list_node->children;
}

struct NML_Tag {
    String tag_name;
    NML_Declaration_Type tag_type = NML_Declaration_Type::NOT_A_TYPE;

    Unordered_Map<String, String> attributes;

    NML_Tag() = default;
    explicit NML_Tag(String tag_name);

    // Reads the element's name and attributes only; build_nml_tree adds the children.
    explicit NML_Tag(xmlNodePtr xml_node);

    bool has_attribute(const String &name) const;
    void add_attribute(const String &name, String value);
    String get_attribute(const String &name) const;
    String get_attribute_or(const String &name, const String &fallback) const;

    // The key a subtype's declaration overrides this one by, ex: "var:tau"; empty if none.
    String namespace_key() const;

    bool is_declaration_type() const;
};

using NML_Node = Node<NML_Tag>;

// Copies an XML element tree. Elements with element children become List_Nodes, others leaves.
NML_Node *build_nml_tree(xmlNodePtr xml_node);



}
