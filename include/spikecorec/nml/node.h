#pragma once

#include <type_traits>
#include <utility>

#include <libxml/parser.h>
#include <libxml/tree.h>

#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

enum class NML_DeclarationType {
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

    NOT_A_TYPE
};

enum class ParseNodeType {
    NML_NODE,
    LEMS_NODE,
    KERNEL_NODE,
};

template <typename SyntaxType, typename TokenType>
struct ParseBody {
    ParseNodeType parse_type{};
    SyntaxType syntax_type{}; // sub type is all the types within the meta "ParseNodeType" set
    TokenType token{};

    ParseBody() = default;
    ParseBody(ParseNodeType parse_type, SyntaxType syntax_type, TokenType token)
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
struct UnaryNode : Node<Body> {
    Node<Body> *child = nullptr;

    explicit UnaryNode(Body body) : Node<Body>(std::move(body)) {};
    ~UnaryNode() override { delete child; };
};

template <typename Body>
struct BinaryNode : Node<Body> {
    Node<Body> *left = nullptr;
    Node<Body> *right = nullptr;

    explicit BinaryNode(Body body) : Node<Body>(std::move(body)) {};
    ~BinaryNode() override {
        delete left;
        delete right;
    };
};

template <typename Body>
struct TrinaryNode : Node<Body> {
    Node<Body> *left = nullptr;
    Node<Body> *middle = nullptr;
    Node<Body> *right = nullptr;

    explicit TrinaryNode(Body body) : Node<Body>(std::move(body)) {};
    ~TrinaryNode() override {
        delete left;
        delete middle;
        delete right;
    };
};

template <typename Body>
struct ListNode : Node<Body> {
    Vector<Node<Body> *> children;

    explicit ListNode(Body body) : Node<Body>(std::move(body)) {};
    ~ListNode() override {
        for (Node<Body> *child : children) delete child;
    };
};

// The children of `node`, or an empty list when it is a leaf (not a ListNode) or null.
template <typename Body>
const Vector<Node<Body> *> &children_of(const Node<Body> *node) {
    static const Vector<Node<Body> *> no_children;

    const auto *list_node = dynamic_cast<const ListNode<Body> *>(node);
    if (!list_node) return no_children;
    return list_node->children;
}

struct NML_Tag {
    String tag_name;
    NML_DeclarationType tag_type = NML_DeclarationType::NOT_A_TYPE;

    UnorderedMap<String, String> attributes;

    NML_Tag() = default;
    explicit NML_Tag(String tag_name);

    // Reads the element's name and attributes only; build_nml_tree adds the children.
    explicit NML_Tag(xmlNodePtr xml_node);

    bool has_attribute(const String &name) const;
    void add_attribute(const String &name, String value);
    String get_attribute(const String &name) const;
    String get_attribute_or(const String &name, const String &fallback) const;

    // Identity of this declaration within its ComponentType, ex: "<namespace>:<name>".
    String namespace_key() const;

    bool is_declaration_type() const;
};

using NML_Node = Node<NML_Tag>;

// Copies an XML element and every element nested in it. An element with element
// children becomes a ListNode<NML_Tag>; any other element is a plain Node<NML_Tag>.
NML_Node *build_nml_tree(xmlNodePtr xml_node);



}
