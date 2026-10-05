#pragma once

#include <type_traits>

// First, so metal-cpp is parsed before any header's `using namespace spikecorec` brings
// String into global lookup.
#include "spikecorec/core/backend.h"
#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"
#include "spikecorec/nml/node.h"
#include "spikecorec/nml/random_generator.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

enum class KernelNodeType {
    LITERAL,              // leaf: token emitted verbatim
    IDENTIFIER,           // leaf: token is the name
    EXPRESSION,           // BinaryNode: token is the operator, "[]" indexes left by right. UnaryNode: prefix operator. TrinaryNode: "?:"
    BLOCK,                // ListNode of statements
    DEVICE_FUNCTION_IMPL, // ListNode of exactly four: return type, name, PARAMETER_LIST, body BLOCK
    FUNCTION_CALL,        // ListNode: token is the function name, children the arguments
    KERNEL_FUNCTION_IMPL, // ListNode: same shape as DEVICE_FUNCTION_IMPL
    DECLARATION,          // TrinaryNode: left the type, middle the name, right the initial value (nullptr for none)
    ASSIGNMENT,           // BinaryNode: token is the operator ("=", "+="), left the target, right the value
    CONDITIONAL,          // ListNode of CONDITIONAL_BRANCH children, emitted as one if / else if / else chain
    CONDITIONAL_BRANCH,   // BinaryNode: left the test (nullptr for the final else), right the body BLOCK
    CAST,                 // BinaryNode: left the type, right the value
    FOR,                  // ListNode of exactly four: initializer, test, step (each nullptr for none), body BLOCK
    WHILE,                // BinaryNode: left the test, right the body BLOCK
    JUMP,                 // UnaryNode: token is "return", "break" or "continue", child the returned value (nullptr for none)
    PARAMETER,            // TrinaryNode: left the type, middle the name, right an ATTRIBUTE (nullptr for none)
    PARAMETER_LIST,       // ListNode of PARAMETER
    ATTRIBUTE,            // leaf: token is the Metal attribute text, e.g. "thread_position_in_grid"
    PROGRAM,              // ListNode: top-level items
    TYPE,                 // leaf: token is the qualifiers and base type, e.g. "const device f32"
    POINTER,              // UnaryNode: child the pointee type, token the qualifiers on the pointer itself (usually empty)
    REFERENCE,            // UnaryNode: child the referenced type
    INCLUDE,              // leaf: token is the header with its delimiters, e.g. "<metal_stdlib>"
    DEFINE,               // BinaryNode: left the macro name, right its replacement LITERAL (nullptr for none)
    USING_NAMESPACE       // leaf: token is the namespace, e.g. "metal"
};

// A TYPE token spells its base type neutrally ("f32", "s64", "u8") after any qualifiers;
// type_to_string maps it to the backend. Pointers and references are POINTER and
// REFERENCE nodes around the TYPE, never part of its token.

enum class KernelBackend { METAL, CUDA };

using KernelParseBody = ParseBody<KernelNodeType, String>;
using KernelNode = Node<KernelParseBody>;
using KernelUnaryNode = UnaryNode<KernelParseBody>;
using KernelBinaryNode = BinaryNode<KernelParseBody>;
using KernelTrinaryNode = TrinaryNode<KernelParseBody>;
using KernelListNode = ListNode<KernelParseBody>;

// neutral type name -> the backend's spelling.
const UnorderedMap<String, String> &kernel_type_spellings(KernelBackend backend);

// The predefined kernel code, in each backend's own syntax. Metal's k^2-tree helpers are
// in k2tree_device.metalinc instead, which the precompiled kernels share.
extern const char *METAL_KERNEL_BOILERPLATE;
extern const char *CUDA_KERNEL_BOILERPLATE;

// NodeShape picks how many children the node holds: Node (a leaf), UnaryNode,
// BinaryNode, TrinaryNode or ListNode, e.g. new_node<BinaryNode>(EXPRESSION, "+").
template <template <typename> class NodeShape = Node>
NodeShape<KernelParseBody> *new_node(KernelNodeType type, String lexeme) {
    return new NodeShape<KernelParseBody>(
            KernelParseBody(ParseNodeType::KERNEL_NODE, type, std::move(lexeme)));
}

KernelNode *new_type(const String &type);
KernelNode *new_pointer(KernelNode *pointee);
KernelNode *new_declaration(KernelNode *type, const String &name, KernelNode *value);
KernelNode *new_assignment(KernelNode *target, const String &assignment_operator, KernelNode *value);
// Indexing is new_expression("[]", array, index).
KernelNode *new_expression(const String &expression_operator, KernelNode *left, KernelNode *right);
KernelNode *new_unary_expression(const String &expression_operator, KernelNode *operand);
KernelNode *new_cast(KernelNode *type, KernelNode *value);
KernelListNode *new_block();
KernelListNode *new_conditional();
// A null test is the final else.
void add_branch(KernelListNode *conditional, KernelNode *test, KernelNode *body);

// Tree operations.
KernelListNode *find_function(KernelNode *program, const String &name);
// Index of the direct child of block that declares or assigns name, or -1.
s64 find_statement(KernelListNode *block, const String &name);
void insert_statements(KernelListNode *block, usize index, const Vector<KernelNode *> &statements);
KernelNode *clone(const KernelNode *node);
// Replaces every use of name with a clone of replacement. Declared names, parameter names,
// function names and member names are not uses.
void replace_identifier(KernelNode *&node, const String &name, const KernelNode *replacement);
// Every name used under node.
void collect_identifiers(const KernelNode *node, Set<String> &names);
// Removes the parameter from function and puts value in place of every use of it.
void bake_parameter(KernelListNode *function, const String &name, const KernelNode *value);

struct NML_DynamicsExpression {
    NML_DeclarationType source_tag;

    // For Regime: target is the regime name and expression is its initial= value.
    // OnCondition and OnEntry start a group; the StateAssignment, EventOut and
    // Transition entries after them belong to it.
    String target;       // variable written, or exposure / port / regime name
    String expression;   // value= / test= source expression, verbatim
    String regime_name;  // owning Regime, empty when the instruction is regime-free
    String condition;    // expression condition gate
    String select;       // DerivedVariable select=, empty when value= is used
    String reduce;       // DerivedVariable reduce=, with select

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

// Parses real Metal or CUDA source into a PROGRAM. Bodies of if, else, for and while
// must be braced.
struct KernelParser {
    Vector<KernelToken> tokens;
    KernelBackend backend;
    usize token_index = 0;

    KernelParser(Vector<KernelToken> tokens, KernelBackend backend)
        : tokens(std::move(tokens)), backend(backend) {};

    KernelListNode *parse_program();
    KernelNode *parse_preprocessor();
    KernelNode *parse_using_namespace();
    KernelNode *parse_function();
    KernelListNode *parse_parameter_list();
    KernelNode *parse_parameter();
    KernelNode *parse_type();
    KernelListNode *parse_block();
    KernelNode *parse_statement();
    KernelNode *parse_simple_statement();
    KernelNode *parse_declaration();
    KernelListNode *parse_conditional();
    KernelNode *parse_for();
    KernelNode *parse_while();
    KernelNode *parse_jump();
    KernelNode *parse_expression();
    KernelNode *parse_binary(s32 minimum_precedence);
    KernelNode *parse_unary();
    KernelNode *parse_postfix();
    KernelNode *parse_primary();

    const KernelToken &current() const;
    const KernelToken &peek(usize distance) const;
    bool check(KernelToken::Kind kind, const String &lexeme = "") const;
    bool accept(KernelToken::Kind kind, const String &lexeme);
    KernelToken expect(KernelToken::Kind kind, const String &lexeme = "");
    bool starts_type() const;
    [[noreturn]] void fail(const String &reason) const;
};

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

// Parses a whole LEMS expression. The caller owns the result.
LemsParseNode *parse_lems_expression(const String &expression, const String &owner_name);

// Defined in components.h and parser.h, which include this file first.
struct NML_ComponentType;
struct NML_ComponentInstance;
struct NML_Context;

// Host-side value of a LEMS expression; every name it reads must be in values. random()
// draws from random_generator and fails without one.
f64 evaluate_lems(const LemsParseNode *node, const UnorderedMap<String, f64> &values, const String &owner_name,
                  RandomGenerator *random_generator = nullptr);
f64 evaluate_lems(const String &expression, const UnorderedMap<String, f64> &values, const String &owner_name,
                  RandomGenerator *random_generator = nullptr);

// Every Parameter, Constant and DerivedParameter the cell's type resolves, in SI units. An
// unset parameter is left out.
UnorderedMap<String, f64> component_parameter_values(const NML_Context &context, const NML_ComponentInstance &cell);

// The cell's parameter values plus every state variable's starting value: its OnStart
// value, otherwise 0. An OnStart that calls random() draws from random_generator.
UnorderedMap<String, f64> starting_values(const NML_Context &context, const NML_ComponentInstance &cell,
                                          RandomGenerator *random_generator = nullptr);

// The target network's populations, in document order.
Vector<const NML_ComponentInstance *> network_populations(const NML_Context &context);

// The cell a population is made of.
const NML_ComponentInstance &population_cell(const NML_Context &context, const NML_ComponentInstance &population);

// Rows in the spike_history ring.
s64 spike_history_length(const NML_Context &context);

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

    // One translated body per cell type, with LEMS names still in it; populations clone it.
    UnorderedMap<String, KernelListNode *> component_type_templates;

    // The same per synapse type; each synapse prototype clones it.
    UnorderedMap<String, KernelListNode *> synapse_type_templates;

    // random() calls translated so far. Each one reads its own placeholder slot in the type's
    // template until translate_component_instance places it.
    s64 random_call_count = 0;
    // The size of the random_values buffer the kernel reads each tick: one slot per random()
    // call per neuron that runs it.
    s64 random_values_count = 0;

    Codegen(NML_Context &context, EngineBackend *device): context(context), device(device) {};
    ~Codegen();

    // The tree and the templates are owned.
    Codegen(const Codegen &other) = delete;
    Codegen &operator=(const Codegen &other) = delete;

    void allocate_cell_model_memory();
    // Writes every cell's starting state (OnStart values, otherwise 0) into cell_state. An
    // OnStart that calls random() draws for each neuron from random_generator.
    void initialize_cell_state(RandomGenerator &random_generator);

    // Refuses a synapse or input that does not provide the input its target cell reads: an
    // exposure the cell's select names, in the same dimension. Called before anything is built.
    void check_cell_inputs() const;

    KernelNode *create_kernel_root();
    KernelNode *translate_kernel_code();
    KernelListNode *translate_component_type(const NML_ComponentType &component_type);
    KernelListNode *translate_component_instance(const NML_ComponentInstance &population, s64 first_neuron);

    // A synapse type's dynamics for one edge, with LEMS names still in it. Runs on every edge
    // every tick; each state variable lives in its own weight-matrix plane.
    KernelListNode *translate_synapse_type(const NML_ComponentType &component_type);
    // The type's template with every state variable read from its plane, weight from the
    // weight plane and every parameter by its value.
    KernelListNode *translate_synapse_instance(const NML_ComponentInstance &synapse);

    // Weight-matrix planes the kernel uses: weight, delay, then one per state variable of the
    // largest synapse type.
    s64 edge_plane_count() const;
    // How many of those planes the generated kernel writes each tick.
    s64 updated_edge_plane_count() const;
    // Ticks after a neuron spikes during which its outgoing synapses can be away from rest: the
    // longest delay plus the slowest synapse's settle time. -1 when some synapse cannot be left
    // at rest, so every synapse runs every tick.
    s64 synapse_active_ticks() const;
    KernelNode *translate_expression(const LemsParseNode *node);
    KernelNode *translate_lems(const String &expression, const String &owner_name);

    // Replaces master_step's parameter name with value everywhere it is used. For values fixed
    // for the whole run that only the engine knows, such as the weight matrix's shape.
    void bake_kernel_constant(const String &name, s64 value);

    // master_step's parameters in binding order, the thread index excluded.
    Vector<String> kernel_parameter_names() const;

    String compile();
    String compile(KernelNode *node);
    String compile_or_empty(KernelNode *node);
    // An expression without its outermost parentheses, for places that already delimit it.
    String compile_unparenthesized(KernelNode *node);
    String type_to_string(const String &type) const;
    String program_to_string(KernelNode *node);
    String attribute_to_string(KernelNode *node);
    String expression_to_string(KernelNode *node, bool parenthesize = true);
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
    String pointer_to_string(KernelNode *node);
    String define_to_string(KernelNode *node);
};

}
