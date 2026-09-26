#pragma once

#include <type_traits>
#include <utility>

#include <libxml/parser.h>
#include <libxml/tree.h>

#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"
#include "spikecorec/nml/declarations.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

struct AbstractToken {
    String lexeme;

protected:
    ~AbstractToken() = default;

    AbstractToken() = default;
    explicit AbstractToken(String lexeme) : lexeme(std::move(lexeme)) {};

    AbstractToken(const AbstractToken &other) = default;
    AbstractToken &operator=(const AbstractToken &other) = default;

    AbstractToken(AbstractToken &&other) noexcept = default;
    AbstractToken &operator=(AbstractToken &&other) noexcept = default;
};

enum class NodeType {
    NML_NODE,
    LEMS_NODE,
    CUDAC_NODE, 
    MTL_NODE, 
};

template <typename Subtype, typename TokenType>
struct ParseNode {
    static_assert(std::is_base_of_v<AbstractToken, TokenType>,
                  "ParseNode's TokenType must derive from AbstractToken");

    NodeType nodetype;
    Subtype subtype; // sub type is all the types within the meta "NodeType" set
    TokenType token;
    ParseNode **branches = nullptr;
    s32 branch_count = 0;
    s32 first_null_branch_index = 0;

    ParseNode();
    ParseNode(NodeType nodetype, Subtype subtype, const TokenType &token, s32 branch_capacity)
        : nodetype(nodetype), subtype(subtype), token(token), branch_count(branch_capacity) {
        if (branch_capacity == 0) return;

        branches = new ParseNode*[branch_capacity];
        for (s32 index = 0; index < branch_capacity; ++index) {
            branches[index] = nullptr;
        }
    };
    ParseNode(NodeType nodetype, Subtype subtype, s32 branch_capacity)
        : nodetype(nodetype), subtype(subtype), branch_count(branch_capacity) {
        if (branch_capacity == 0) return;

        branches = new ParseNode*[branch_capacity];
        for (s32 index = 0; index < branch_capacity; ++index) {
            branches[index] = nullptr;
        }
    };
    ~ParseNode() {
        if (branches == nullptr) return;
        if (branch_count == 0) return;

        for (s32 index = 0; index < branch_count; ++index) {
            if (branches[index] == nullptr) return;
            delete branches[index];
        }
        delete branches;
    };

    static ParseNode *lems_node(Subtype subtype, const TokenType &token, s32 branch_capacity) {
        return new ParseNode(NodeType::LEMS_NODE, subtype, token, branch_capacity);
    }

    static ParseNode *lems_node(Subtype subtype, s32 branch_capacity) {
        return new ParseNode(NodeType::LEMS_NODE, subtype, branch_capacity);
    }

    void add_branch(ParseNode *branch) {
        if (first_null_branch_index == branch_count) return;

        branches[first_null_branch_index] = branch;
        first_null_branch_index++;
    };
};


struct NML_Node {
    String tag_name;

    UnorderedMap<String, String> attributes;
    Vector<NML_Node> body;

    NML_Node() = default;
    NML_Node(String tag_name) : tag_name(std::move(tag_name)) {};
    NML_Node(xmlNodePtr xml_node);

    NML_Node(const NML_Node &other) = default;
    NML_Node &operator=(const NML_Node &other) = default;

    NML_Node(NML_Node &&other) noexcept = default;
    NML_Node &operator=(NML_Node &&other) noexcept = default;

    bool has_attribute(const String &name) const;
    void add_attribute(const String &name, String value);
    String get_attribute(const String &name) const;
    void nest(NML_Node component);
    NML_Declaration to_declaration() const;
    bool is_declaration_type() const;
};



}
