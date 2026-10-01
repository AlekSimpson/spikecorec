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
    LITERAL,              // leaf: token emitted verbatim
    IDENTIFIER,           // leaf: token is the name
    EXPRESSION,           // BinaryNode: token is the operator, "[]" indexes left by right. UnaryNode: prefix operator. TrinaryNode: "?:"
    BLOCK,                // ListNode of statements
    DEVICE_FUNCTION_IMPL, // TrinaryNode: token is the return type, left the name, middle the PARAMETER_LIST, right the body BLOCK
    FUNCTION_CALL,        // ListNode: token is the function name, children the arguments
    KERNEL_FUNCTION_IMPL, // TrinaryNode: same shape as DEVICE_FUNCTION_IMPL
    DECLARATION,          // BinaryNode: token is the type, left the name, right the initial value (nullptr for none)
    ASSIGNMENT,           // BinaryNode: token is the operator ("=", "+="), left the target, right the value
    CONDITIONAL,          // ListNode of CONDITIONAL_BRANCH children, emitted as one if / else if / else chain
    CONDITIONAL_BRANCH,   // BinaryNode: left the test (nullptr for the final else), right the body BLOCK
    CAST,                 // UnaryNode: token is the type, child the value
    FOR,                  // ListNode of exactly four: initializer, test, step (each nullptr for none), body BLOCK
    WHILE,                // BinaryNode: left the test, right the body BLOCK
    JUMP,                 // UnaryNode: token is "return", "break" or "continue", child the returned value (nullptr for none)
    PARAMETER,            // BinaryNode: token is the type, left the name, right an ATTRIBUTE (nullptr for none)
    PARAMETER_LIST,       // ListNode of PARAMETER
    ATTRIBUTE,            // leaf: token is the Metal attribute text, e.g. "thread_position_in_grid"
    PROGRAM               // ListNode: top-level items (verbatim LITERAL lines, DEVICE_FUNCTION_IMPL, KERNEL_FUNCTION_IMPL)
};

// Types are spelled neutrally ("f32", "s64", "u8"), with qualifiers, "*" and "&" as separate
// words, e.g. "const device f32 *". type_to_string maps them to the backend.

enum class KernelBackend { METAL, CUDA };

using KernelParseBody = ParseBody<KernelNodeType, String>;
using KernelNode = Node<KernelParseBody>;
using KernelUnaryNode = UnaryNode<KernelParseBody>;
using KernelBinaryNode = BinaryNode<KernelParseBody>;
using KernelTrinaryNode = TrinaryNode<KernelParseBody>;
using KernelListNode = ListNode<KernelParseBody>;

// Neutral type name -> the backend's spelling.
const UnorderedMap<String, String> &kernel_type_spellings(KernelBackend backend);

// NodeShape picks how many children the node holds: Node (a leaf), UnaryNode,
// BinaryNode, TrinaryNode or ListNode, e.g. new_node<BinaryNode>(EXPRESSION, "+").
template <template <typename> class NodeShape = Node>
NodeShape<KernelParseBody> *new_node(KernelNodeType type, String lexeme) {
    return new NodeShape<KernelParseBody>(
            KernelParseBody(ParseNodeType::KERNEL_NODE, type, std::move(lexeme)));
}

KernelNode *new_declaration(const String &type, const String &name, KernelNode *value);
KernelNode *new_assignment(KernelNode *target, const String &assignment_operator, KernelNode *value);
// Indexing is new_expression("[]", array, index).
KernelNode *new_expression(const String &expression_operator, KernelNode *left, KernelNode *right);
KernelNode *new_cast(const String &type, KernelNode *value);
KernelListNode *new_conditional();
// A null test is the final else.
void add_branch(KernelListNode *conditional, KernelNode *test, KernelNode *body);

struct NML_DynamicsExpression;

// Defined in components.h and parser.h, which include this file first.
struct NML_ComponentType;
struct NML_ComponentInstance;
struct NML_Context;

struct Codegen {
    const NML_Context &context;
    EngineBackend *device;
    Vector<EnginePointer> data_partitions;
    KernelNode *root = nullptr;
#ifdef SPIKECOREC_METAL
    KernelBackend backend = KernelBackend::METAL;
#else
    KernelBackend backend = KernelBackend::CUDA;
#endif
    usize indentation_depth = 0;

    Codegen(NML_Context &context, EngineBackend *device): context(context), device(device) {};

    void allocate_cell_model_memory();

    KernelNode *create_kernel_root();
    KernelNode *translate_kernel_code();
    KernelNode *translate_component_type(NML_ComponentType &component_type);
    KernelNode *translate_component_instance(NML_ComponentInstance &component_instance);
    KernelNode *translate_expression(NML_DynamicsExpression &expression);

    f64 get_starting_parameters(NML_DynamicsExpression &expression);

    String compile();
    String compile(KernelNode *node);
    String compile_or_empty(KernelNode *node);
    String type_to_string(const String &type) const;
    String program_to_string(KernelNode *node);
    String attribute_to_string(KernelNode *node);
    String expression_to_string(KernelNode *node);
    String cast_to_string(KernelNode *node);
    String function_call_to_string(KernelNode *node);
    String block_to_string(KernelNode *node);
    String declaration_to_string(KernelNode *node);
    String assignment_to_string(KernelNode *node);
    String conditional_to_string(KernelNode *node);
    String for_to_string(KernelNode *node);
    String while_to_string(KernelNode *node);
    String jump_to_string(KernelNode *node);
    String parameters_to_string(KernelNode *node, bool number_buffers);
    String device_function_impl_to_string(KernelNode *node);
    String kernel_function_impl_to_string(KernelNode *node);
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

struct KernelToken {
    enum class Kind {Number, Identifier, Keyword, Type, Qualifier, Operator, Punctuation, StringLiteral, Attribute, Preprocessor, End};

    String lexeme;
    Kind kind = Kind::End;

    KernelToken() = default;
    KernelToken(Kind kind, String lexeme)
        : lexeme(std::move(lexeme)), kind(kind) {};
};

using KernelLexer = Lexer<KernelToken>;

// Lexes real Metal or CUDA source; type words and qualifiers are classified per backend.
Vector<KernelToken> tokenize_kernel(const String &source, KernelBackend backend);

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


