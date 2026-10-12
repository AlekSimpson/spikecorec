#pragma once

#include <type_traits>

// First, so metal-cpp is parsed before any header's `using namespace spikecorec` brings
// String into global lookup.
#include "backend.h"
#include "units.h"
#include "types.h"
#include "node.h"
#include "random_generator.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

enum class Kernel_Node_Type {
    LITERAL,              // leaf: token emitted verbatim
    IDENTIFIER,           // leaf: token is the name
    EXPRESSION,           // Binary_Node: token is the operator, "[]" indexes left by right. Unary_Node: prefix operator. Trinary_Node: "?:"
    BLOCK,                // List_Node of statements
    DEVICE_FUNCTION_IMPL, // List_Node of exactly four: return type, name, PARAMETER_LIST, body BLOCK
    FUNCTION_CALL,        // List_Node: token is the function name, children the arguments
    KERNEL_FUNCTION_IMPL, // List_Node: same shape as DEVICE_FUNCTION_IMPL
    DECLARATION,          // Trinary_Node: left the type, middle the name, right the initial value (nullptr for none)
    ASSIGNMENT,           // Binary_Node: token is the operator ("=", "+="), left the target, right the value
    CONDITIONAL,          // List_Node of CONDITIONAL_BRANCH children, emitted as one if / else if / else chain
    CONDITIONAL_BRANCH,   // Binary_Node: left the test (nullptr for the final else), right the body BLOCK
    CAST,                 // Binary_Node: left the type, right the value
    FOR,                  // List_Node of exactly four: initializer, test, step (each nullptr for none), body BLOCK
    WHILE,                // Binary_Node: left the test, right the body BLOCK
    JUMP,                 // Unary_Node: token is "return", "break" or "continue", child the returned value (nullptr for none)
    PARAMETER,            // Trinary_Node: left the type, middle the name, right an ATTRIBUTE (nullptr for none)
    PARAMETER_LIST,       // List_Node of PARAMETER
    ATTRIBUTE,            // leaf: token is the Metal attribute text, e.g. "thread_position_in_grid"
    PROGRAM,              // List_Node: top-level items
    TYPE,                 // leaf: token is the qualifiers and base type, e.g. "const device f32"
    POINTER,              // Unary_Node: child the pointee type, token the qualifiers on the pointer itself (usually empty)
    REFERENCE,            // Unary_Node: child the referenced type
    INCLUDE,              // leaf: token is the header with its delimiters, e.g. "<metal_stdlib>"
    DEFINE,               // Binary_Node: left the macro name, right its replacement LITERAL (nullptr for none)
    USING_NAMESPACE       // leaf: token is the namespace, e.g. "metal"
};

// A TYPE token spells its base type neutrally ("f32", "s64", "u8") after any qualifiers;
// type_to_string maps it to the backend. Pointers and references are POINTER and
// REFERENCE nodes around the TYPE, never part of its token.

enum class Kernel_Backend { METAL, CUDA };

using Kernel_Parse_Body = Parse_Body<Kernel_Node_Type, String>;
using Kernel_Node = Node<Kernel_Parse_Body>;
using Kernel_Unary_Node = Unary_Node<Kernel_Parse_Body>;
using Kernel_Binary_Node = Binary_Node<Kernel_Parse_Body>;
using Kernel_Trinary_Node = Trinary_Node<Kernel_Parse_Body>;
using Kernel_List_Node = List_Node<Kernel_Parse_Body>;

// neutral type name -> the backend's spelling.
const Unordered_Map<String, String> &kernel_type_spellings(Kernel_Backend backend);

// The predefined kernel code, in each backend's own syntax. Metal's k^2-tree helpers are
// in k2tree_device.metalinc instead, which the precompiled kernels share.
extern const char *METAL_KERNEL_BOILERPLATE;
extern const char *CUDA_KERNEL_BOILERPLATE;

// Node_Shape picks how many children the node holds: Node (a leaf), Unary_Node,
// Binary_Node, Trinary_Node or List_Node, e.g. new_node<Binary_Node>(EXPRESSION, "+").
template <template <typename> class Node_Shape = Node>
Node_Shape<Kernel_Parse_Body> *new_node(Kernel_Node_Type type, String lexeme) {
    return new Node_Shape<Kernel_Parse_Body>(
            Kernel_Parse_Body(Parse_Node_Type::KERNEL_NODE, type, std::move(lexeme)));
}

Kernel_Node *new_type(const String &type);
Kernel_Node *new_pointer(Kernel_Node *pointee);
Kernel_Node *new_declaration(Kernel_Node *type, const String &name, Kernel_Node *value);
Kernel_Node *new_assignment(Kernel_Node *target, const String &assignment_operator, Kernel_Node *value);
// Indexing is new_expression("[]", array, index).
Kernel_Node *new_expression(const String &expression_operator, Kernel_Node *left, Kernel_Node *right);
Kernel_Node *new_unary_expression(const String &expression_operator, Kernel_Node *operand);
Kernel_Node *new_cast(Kernel_Node *type, Kernel_Node *value);
Kernel_List_Node *new_block();
Kernel_List_Node *new_conditional();
// A null test is the final else.
void add_branch(Kernel_List_Node *conditional, Kernel_Node *test, Kernel_Node *body);

// Tree operations.
Kernel_List_Node *find_function(Kernel_Node *program, const String &name);
// Index of the direct child of block that declares or assigns name, or -1.
s64 find_statement(Kernel_List_Node *block, const String &name);
void insert_statements(Kernel_List_Node *block, usize index, const Vector<Kernel_Node *> &statements);
Kernel_Node *clone(const Kernel_Node *node);
// Replaces every use of name with a clone of replacement. Declared names, parameter names,
// function names and member names are not uses.
void replace_identifier(Kernel_Node *&node, const String &name, const Kernel_Node *replacement);
// Every name used under node.
void collect_identifiers(const Kernel_Node *node, Set<String> &names);
// Replaces every +, -, * and / of float literals alone, and every negated float literal, with the
// literal of its value, worked out in double precision: rounded once instead of per operation.
void fold_constants(Kernel_Node *&node);
// Removes the parameter from function and puts value in place of every use of it.
void bake_parameter(Kernel_List_Node *function, const String &name, const Kernel_Node *value);

struct NML_Dynamics_Expression {
    NML_Declaration_Type source_tag;

    // For Regime: target is the regime name and expression is its initial= value.
    // OnCondition and OnEntry start a group; the StateAssignment, EventOut and
    // Transition entries after them belong to it.
    String target;       // variable written, or exposure / port / regime name
    String expression;   // value= / test= source expression, verbatim
    String regime_name;  // owning Regime, empty when the instruction is regime-free
    String condition;    // expression condition gate
    String select;       // DerivedVariable select=, empty when value= is used
    String reduce;       // DerivedVariable reduce=, with select

    NML_Dynamics_Expression(NML_Declaration_Type source_tag)
        : source_tag(source_tag) {};
};

// One entry of a ComponentType's <Structure>, in document order.
//   ChildInstance:   component names the ComponentReference instantiated once per instance of the type.
//   With:            instance ("this", "parent", or a Path or child name) is bound to alias.
//   EventConnection: source and target are the aliases in its from= and to=; receiver and the
//                    ports are empty when not given.
struct NML_Structure_Entry {
    NML_Declaration_Type source_tag;
    String component;
    String instance;
    String alias;
    String source;
    String target;
    String receiver;
    String source_port;
    String target_port;

    NML_Structure_Entry(NML_Declaration_Type source_tag)
        : source_tag(source_tag) {};
};

struct Lems_Expression_Token {
    enum class Kind {Number, Identifier, Operator, OpenParen, CloseParen, Comma, End};

    String lexeme;
    Kind kind = Kind::End;

    Lems_Expression_Token() = default;
    Lems_Expression_Token(Kind kind, String lexeme)
        : lexeme(std::move(lexeme)), kind(kind) {};

    Lems_Expression_Token(const Lems_Expression_Token &other) = default;
    Lems_Expression_Token &operator=(const Lems_Expression_Token &other) = default;

    Lems_Expression_Token(Lems_Expression_Token &&other) noexcept = default;
    Lems_Expression_Token &operator=(Lems_Expression_Token &&other) noexcept = default;

    ~Lems_Expression_Token() = default;
};

template <typename Token_Type>
struct Lexer {
    using Predicate = bool (*)(Lexer &, char);
    using Action = void (*)(Lexer &);

    Vector<Predicate> predicates;
    Vector<Action> actions;
    Vector<Token_Type> tokens;
    String source;
    usize position = 0;
    char current_character = '\0';

    Lexer() {};

    Lexer &add_check(Predicate predicate, Action action) {
        predicates.push_back(predicate);
        actions.push_back(action);
        return *this;
    };

    Vector<Token_Type> lex(const String &source) {
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
                throw Runtime_Error(
                        "Found unexpected character '" + String(1, current_character) +
                        "' in '" + this->source + "' while parsing neuroml.");
            }
        }

        return tokens;
    };
};

using Lems_Lexer = Lexer<Lems_Expression_Token>;

Vector<Lems_Expression_Token> tokenize_lems(const String &expression, const String &owner_name);

struct Kernel_Token {
    enum class Kind {Number, Identifier, Keyword, Type, Qualifier, Operator, Punctuation, StringLiteral, Attribute, Preprocessor, End};

    String lexeme;
    Kind kind = Kind::End;

    Kernel_Token() = default;
    Kernel_Token(Kind kind, String lexeme)
        : lexeme(std::move(lexeme)), kind(kind) {};
};

using Kernel_Lexer = Lexer<Kernel_Token>;

// Lexes real Metal or CUDA source; type words and qualifiers are classified per backend.
Vector<Kernel_Token> tokenize_kernel(const String &source, Kernel_Backend backend);

// Parses real Metal or CUDA source into a PROGRAM. Bodies of if, else, for and while
// must be braced.
struct Kernel_Parser {
    Vector<Kernel_Token> tokens;
    Kernel_Backend backend;
    usize token_index = 0;

    Kernel_Parser(Vector<Kernel_Token> tokens, Kernel_Backend backend)
        : tokens(std::move(tokens)), backend(backend) {};

    Kernel_List_Node *parse_program();
    Kernel_Node *parse_preprocessor();
    Kernel_Node *parse_using_namespace();
    Kernel_Node *parse_function();
    Kernel_List_Node *parse_parameter_list();
    Kernel_Node *parse_parameter();
    Kernel_Node *parse_type();
    Kernel_List_Node *parse_block();
    Kernel_Node *parse_statement();
    Kernel_Node *parse_simple_statement();
    Kernel_Node *parse_declaration();
    Kernel_List_Node *parse_conditional();
    Kernel_Node *parse_for();
    Kernel_Node *parse_while();
    Kernel_Node *parse_jump();
    Kernel_Node *parse_expression();
    Kernel_Node *parse_binary(s32 minimum_precedence);
    Kernel_Node *parse_unary();
    Kernel_Node *parse_postfix();
    Kernel_Node *parse_primary();

    const Kernel_Token &current() const;
    const Kernel_Token &peek(usize distance) const;
    bool check(Kernel_Token::Kind kind, const String &lexeme = "") const;
    bool accept(Kernel_Token::Kind kind, const String &lexeme);
    Kernel_Token expect(Kernel_Token::Kind kind, const String &lexeme = "");
    bool starts_type() const;
    [[noreturn]] void fail(const String &reason) const;
};

enum class Lems_Node_Subtype {
    EXPRESSION,
    IDENTIFIER,
    FUNCTION_CALL,
    LIST,
    OPERATOR,
    FLOAT,
    INT
};

using Lems_Parse_Body = Parse_Body<Lems_Node_Subtype, Lems_Expression_Token>;
using Lems_Parse_Node = Node<Lems_Parse_Body>;

struct Lems_Expression_Parser {
    Vector<Lems_Expression_Token> tokens;
    String expression;
    String owner_name;
    Lems_Parse_Node *root = nullptr;
    usize token_index = 0;

    [[noreturn]] void fail(const String &reason) const;
    s32 binary_precedence(const Lems_Expression_Token &token);
    const Lems_Expression_Token &current() const;
    Lems_Parse_Node *resolve_identifier(const Lems_Expression_Token &name) const;
    Lems_Parse_Node *parse_primary();
    Lems_Parse_Node *parse_unary();
    Lems_Parse_Node *parse_binary(s32 minimum_precedence);
};

// Parses a whole LEMS expression. The caller owns the result.
Lems_Parse_Node *parse_lems_expression(const String &expression, const String &owner_name);

// Defined in components.h and parser.h, which include this file first.
struct NML_Component_Type;
struct NML_Component_Instance;
struct NML_Context;

// Host-side value of a LEMS expression; every name it reads must be in values. random()
// draws from random_generator and fails without one.
f64 evaluate_lems(const Lems_Parse_Node *node, const Unordered_Map<String, f64> &values, const String &owner_name,
                  Random_Generator *random_generator = nullptr);
f64 evaluate_lems(const String &expression, const Unordered_Map<String, f64> &values, const String &owner_name,
                  Random_Generator *random_generator = nullptr);

// Every Parameter, Constant and DerivedParameter the cell's type resolves, in SI units. An
// unset parameter is left out.
Unordered_Map<String, f64> component_parameter_values(const NML_Context &context, const NML_Component_Instance &cell);

// The cell's parameter values plus every state variable's starting value: its OnStart
// value, otherwise 0. An OnStart that calls random() draws from random_generator.
Unordered_Map<String, f64> starting_values(const NML_Context &context, const NML_Component_Instance &cell,
                                          Random_Generator *random_generator = nullptr);

// The target network's populations, in document order.
Vector<const NML_Component_Instance *> network_populations(const NML_Context &context);

// The cell a population is made of.
const NML_Component_Instance &population_cell(const NML_Context &context, const NML_Component_Instance &population);

// Rows in the spike_history ring.
s64 spike_history_length(const NML_Context &context);

// What one spike from a spike-source input adds to a cell's input for one tick when the input
// gives no amplitude: enough to lift the membrane from the lowest value it starts or resets to,
// to 5% past its threshold, through the cell's capacitance.
f64 default_kick_amplitude(const NML_Context &context, const NML_Component_Instance &cell);

// One part of an input's component tree: the input itself, a child it declares (a Child, or a
// member of a Children collection), or a component it instantiates with ChildInstance, such as
// its own copy of the synapse it references. Every entry (one input on one target cell) has its
// own copy of every part's state.
struct Input_Part {
    const NML_Component_Instance *instance = nullptr;
    s64 parent = -1;            // the part it belongs to; -1 for the input itself
    String role;                // the parent's Child, Children or ComponentReference name it fills
    s64 first_state_slot = 0;   // where its state variables start in an entry's state block
};

// A Children collection of spikes, reduced to how many fire on each tick of the run: 1 on a
// tick a spike fires, 0 otherwise, 2 or more only when spike times round to the same tick.
struct Input_Spike_Train {
    s64 receiver = -1;          // the part their events go to
    Vector<u8> spike_counts;    // [total_tick_count]
};

// An EventOut of one part is delivered to the OnEvents of another.
struct Input_Event_Route {
    s64 source = -1;
    s64 target = -1;
};

struct Input_Tree {
    Vector<Input_Part> parts;    // parts[0] is the input; a parent always comes before its children
    Vector<Input_Spike_Train> spike_trains;
    Vector<Input_Event_Route> routes;
    s64 state_size = 0;         // floats of state per entry
};

// One input on one target cell.
struct Input_Entry {
    s64 neuron_index = -1;
    s32 prototype = -1;         // index into Codegen::input_prototypes
    f32 weight = 1.0f;
};

struct Codegen {
    const NML_Context &context;
    Engine_Backend *device;
    Vector<Engine_Pointer> data_partitions;
    Kernel_Node *root = nullptr;
#ifdef SPIKECOREC_METAL
    Kernel_Backend backend = Kernel_Backend::METAL;
#else
    Kernel_Backend backend = Kernel_Backend::CUDA;
#endif
    usize indentation_depth = 0;

    // One translated body per cell type, with LEMS names still in it; populations clone it.
    Unordered_Map<String, Kernel_List_Node *> component_type_templates;

    // The same per synapse type; each synapse prototype clones it.
    Unordered_Map<String, Kernel_List_Node *> synapse_type_templates;

    // random() calls translated so far. Each one reads its own placeholder slot in the type's
    // template until translate_component_instance places it.
    s64 random_call_count = 0;
    // The size of the random_values buffer the kernel reads each tick: one slot per random()
    // call per neuron that runs it.
    s64 random_values_count = 0;

    Codegen(NML_Context &context, Engine_Backend *device): context(context), device(device) {};
    ~Codegen();

    // The tree and the templates are owned.
    Codegen(const Codegen &other) = delete;
    Codegen &operator=(const Codegen &other) = delete;

    // True when some population's cell type has an OnEvent, so the model needs the
    // event_arrival_count buffer and the kernel counts arrivals into it.
    bool cells_receive_events() const;

    // Inputs. A prototype is one input component, compiled once per population it reaches; an
    // entry is that input on one target cell, with its own weight and its own copy of the
    // input's state. Filled by collect_inputs.
    Vector<const NML_Component_Instance *> input_prototypes;
    Vector<Input_Tree> input_trees;                     // per prototype
    Vector<Vector<s64>> input_spike_train_offsets;     // per prototype, per spike train: where it starts in input_spike_counts
    s64 input_spike_count_size = 0;
    Vector<Input_Entry> input_entries;                  // sorted by neuron_index
    // Floats per entry in input_values: its weight, then the largest prototype's state.
    s64 input_value_stride = 1;

    // The input's component tree as flat parts, with every Structure EventConnection resolved to a
    // route between parts. A Children collection of spikes becomes a spike train, not parts.
    Input_Tree flatten_input_tree(const NML_Component_Instance &input) const;
    // Fills the input prototypes, trees and entries from the document's input profiles.
    void collect_inputs();
    // Writes the input buffers: each neuron's entries, each entry's weight and starting state
    // (OnStart values, otherwise 0), and the spike trains.
    void initialize_input_state(Random_Generator &random_generator);
    // The loop over a neuron's input entries for one population, run before its cell's
    // dynamics; nullptr when no input reaches the population.
    Kernel_List_Node *translate_input_entries(const NML_Component_Instance &population, s64 first_neuron);
    // One prototype's dynamics for one entry on a cell of the population.
    Kernel_List_Node *translate_input_prototype(s32 prototype, const NML_Component_Instance &population, s64 first_neuron);

    void allocate_cell_model_memory();
    // Writes every cell's starting state (OnStart values, otherwise 0) into cell_state. An
    // OnStart that calls random() draws for each neuron from random_generator.
    void initialize_cell_state(Random_Generator &random_generator);

    // Refuses a synapse or input that does not provide the input its target cell reads: an
    // exposure the cell's select names, in the same dimension. A synapse's spikes also count as
    // input to a cell with an OnEvent. Called before anything is built.
    void check_cell_inputs() const;

    Kernel_Node *create_kernel_root();
    Kernel_Node *translate_kernel_code();
    Kernel_List_Node *translate_component_type(const NML_Component_Type &component_type);
    Kernel_List_Node *translate_component_instance(const NML_Component_Instance &population, s64 first_neuron);

    // A synapse type's dynamics for one edge, with LEMS names still in it. Runs on every edge
    // every tick; each state variable lives in its own weight-matrix plane.
    Kernel_List_Node *translate_synapse_type(const NML_Component_Type &component_type);
    // The type's template with every state variable read from its plane, weight from the
    // weight plane and every parameter by its value.
    Kernel_List_Node *translate_synapse_instance(const NML_Component_Instance &synapse);

    // Weight-matrix planes the kernel uses: weight, delay, then one per state variable of the
    // largest synapse type.
    s64 edge_plane_count() const;
    // How many of those planes the generated kernel writes each tick.
    s64 updated_edge_plane_count() const;
    // Ticks after a neuron spikes during which its outgoing synapses can be away from rest: the
    // longest delay plus the slowest synapse's settle time. -1 when some synapse cannot be left
    // at rest, so every synapse runs every tick.
    s64 synapse_active_ticks() const;
    Kernel_Node *translate_expression(const Lems_Parse_Node *node);
    Kernel_Node *translate_lems(const String &expression, const String &owner_name);

    // Replaces master_step's parameter name with value everywhere it is used. For values fixed
    // for the whole run that only the engine knows, such as the weight matrix's shape.
    void bake_kernel_constant(const String &name, s64 value);

    // master_step's parameters in binding order, the thread index excluded.
    Vector<String> kernel_parameter_names() const;

    String compile();
    String compile(Kernel_Node *node);
    String compile_or_empty(Kernel_Node *node);
    // An expression without its outermost parentheses, for places that already delimit it.
    String compile_unparenthesized(Kernel_Node *node);
    String type_to_string(const String &type) const;
    String program_to_string(Kernel_Node *node);
    String attribute_to_string(Kernel_Node *node);
    String expression_to_string(Kernel_Node *node, bool parenthesize = true);
    String cast_to_string(Kernel_Node *node);
    String function_call_to_string(Kernel_Node *node);
    String block_to_string(Kernel_Node *node);
    String declaration_to_string(Kernel_Node *node);
    String assignment_to_string(Kernel_Node *node);
    String conditional_to_string(Kernel_Node *node);
    String for_to_string(Kernel_Node *node);
    String while_to_string(Kernel_Node *node);
    String jump_to_string(Kernel_Node *node);
    String parameters_to_string(Kernel_Node *node, bool number_buffers);
    String device_function_impl_to_string(Kernel_Node *node);
    String kernel_function_impl_to_string(Kernel_Node *node);
    String pointer_to_string(Kernel_Node *node);
    String define_to_string(Kernel_Node *node);
};

}
