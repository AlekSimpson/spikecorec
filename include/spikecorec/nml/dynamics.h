#pragma once

#include <type_traits>

// First, so metal-cpp is parsed before any header's `using namespace spikecorec` brings
// String into global lookup.
#include "spikecorec/core/backend.h"
#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"
#include "spikecorec/nml/node.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

enum class KernelNodeType {
    LITERAL,
    IDENTIFIER,
    EXPRESSION,
    BLOCK,
    DEVICE_FUNCTION_IMPL, 
    FUNCTION_CALL,
    KERNEL_FUNCTION_IMPL
};

// A kernel node's token is the source text it emits.
using KernelParseBody = ParseBody<KernelNodeType, String>;
using KernelNode = Node<KernelParseBody>;

struct NML_DynamicsExpression;

// Defined in components.h and parser.h, which include this file first.
struct NML_ComponentType;
struct NML_ComponentInstance;
struct NML_Context;

struct Codegen {
    const NML_Context &context;
    EngineBackend *device;
    Vector<EnginePointer> data_partitions;
    KernelNode *root;

    Codegen(NML_Context &context, EngineBackend *device): context(context), device(device) {};

    void allocate_cell_model_memory();

    String compile();
    String compile(KernelNode *);

    KernelNode *creat_kernel_root();
    KernelNode *translate_kernel_code();
    KernelNode *translate_component_type(NML_ComponentType &component_type);
    KernelNode *translate_component_instance(NML_ComponentInstance &component_instance);
    KernelNode *translate_expression(NML_DynamicsExpression &expression);

    f64 get_starting_parameters(NML_DynamicsExpression &expression);

    // NodeShape picks how many children the node holds: Node (a leaf), UnaryNode,
    // BinaryNode, TrinaryNode or ListNode, e.g. new_node<BinaryNode>(EXPRESSION, "+").
    template <template <typename> class NodeShape = Node>
    NodeShape<KernelParseBody> *new_node(KernelNodeType type, String lexeme) {
        #ifdef SPIKECOREMETALC
            return new NodeShape<KernelParseBody>(
                    KernelParseBody(ParseNodeType::MTL_NODE, type, std::move(lexeme)));
        #else
            return new NodeShape<KernelParseBody>(
                    KernelParseBody(ParseNodeType::CUDAC_NODE, type, std::move(lexeme)));
        #endif
    };
};

struct NML_DynamicsExpression {
    NML_DeclarationType source_tag;

    String target;       // variable written, or exposure / port name
    String expression;   // value= / test= source expression, verbatim
    String regime_name;  // owning Regime, empty when the instruction is regime-free
    String condition;    // expression condition gate

    NML_DynamicsExpression(NML_DeclarationType source_tag)
        : source_tag(source_tag) {};
};

struct LemsExpressionToken {
    enum class Kind {Number, Identifier, Operator, OpenParen, CloseParen, Comma, End};

    String lexeme;
    Kind kind = Kind::End;

    LemsExpressionToken() = default;
    LemsExpressionToken(Kind kind, String lexeme)
        : lexeme(std::move(lexeme)), kind(kind) {};

    LemsExpressionToken(const LemsExpressionToken &other) = default;
    LemsExpressionToken &operator=(const LemsExpressionToken &other) = default;

    LemsExpressionToken(LemsExpressionToken &&other) noexcept = default;
    LemsExpressionToken &operator=(LemsExpressionToken &&other) noexcept = default;

    ~LemsExpressionToken() = default;
};

// does this struct need to be in this file? feels a little out of place in the "dynamics" definitions file
template <typename TokenType>
struct Lexer {
    using Predicate = bool (*)(Lexer &, char);
    using Action = void (*)(Lexer &);

    Vector<Predicate> predicates;
    Vector<Action> actions;
    Vector<TokenType> tokens;
    String source;
    usize position = 0;
    char current_character = '\0';

    Lexer() {};

    Lexer &add_check(Predicate predicate, Action action) {
        predicates.push_back(predicate);
        actions.push_back(action);
        return *this;
    };

    Vector<TokenType> lex(const String &source) {
        tokens.clear();
        this->source = source;
        position = 0;

        while (position < this->source.size()) {
            current_character = this->source[position];

            bool found_unknown_character = true;
            for (usize index = 0; index < predicates.size(); ++index) {
                Predicate current_predicate = predicates[index];
                Action current_action = actions[index];
                if (current_predicate(*this, current_character)) {
                    current_action(*this);
                    found_unknown_character = false;
                    break;
                }
            }

            if (found_unknown_character) {
                throw runtime_error(
                        "Found unexpected character '" + String(1, current_character) +
                        "' in '" + this->source + "' while parsing neuroml.");
            }
        }

        return tokens;
    };
};

using LemsLexer = Lexer<LemsExpressionToken>;

Vector<LemsExpressionToken> tokenize_lems(const String &expression, const String &owner_name);

// What each name inside an NML expression is bound to in the generated kernel: the
// declared NML name maps to the C expression that reads it. "leakConductance" ->
// "cell_parameters[parameter_base + 3]", "v" -> "state_0", "iSyn" -> "network_input".
using SymbolTable = UnorderedMap<String, String>;

enum class LemsNodeSubtype {
    EXPRESSION,
    IDENTIFIER,
    FUNCTION_CALL,
    LIST,
    OPERATOR, 
    FLOAT, 
    INT
};

using LemsParseBody = ParseBody<LemsNodeSubtype, LemsExpressionToken>;
using LemsParseNode = Node<LemsParseBody>;

struct LemsExpressionParser {
    Vector<LemsExpressionToken> tokens;
    SymbolTable symbols;
    String expression;
    String owner_name;
    LemsParseNode *root = nullptr;
    usize token_index = 0;

    [[noreturn]] void fail(const String &reason) const;
    s32 binary_precedence(const LemsExpressionToken &token);
    const LemsExpressionToken &current() const;
    LemsParseNode *resolve_identifier(const LemsExpressionToken &name) const;
    LemsParseNode *parse_primary();
    LemsParseNode *parse_unary();
    LemsParseNode *parse_binary(s32 minimum_precedence);
};

}


