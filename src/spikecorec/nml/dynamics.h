#pragma once

#include <type_traits>

// First, so metal-cpp is parsed before any header's `using namespace spikecorec` brings
// String into global lookup.
#include "../core/backend/metal_backend.h"
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
    EXPRESSION,           // token the operator; Binary ("[]" indexes), Unary (prefix) or Trinary
    BLOCK,                // List_Node of statements
    DEVICE_FUNCTION_IMPL, // List_Node: return type, name, PARAMETER_LIST, body BLOCK
    FUNCTION_CALL,        // List_Node: token the function name, children the arguments
    KERNEL_FUNCTION_IMPL, // List_Node: same shape as DEVICE_FUNCTION_IMPL
    DECLARATION,          // Trinary_Node: type, name (or name[size]), initial value or nullptr
    ASSIGNMENT,           // Binary_Node: token the operator ("=", "+="), target, value
    CONDITIONAL,          // List_Node of CONDITIONAL_BRANCH: one if / else if / else chain
    CONDITIONAL_BRANCH,   // Binary_Node: test (nullptr for the final else), body BLOCK
    CAST,                 // Binary_Node: type, value
    FOR,                  // List_Node: initializer, test, step (each may be nullptr), body BLOCK
    WHILE,                // Binary_Node: test, body BLOCK
    JUMP,                 // Unary_Node: token return, break or continue; child the value or nullptr
    PARAMETER,            // Trinary_Node: type, name, ATTRIBUTE or nullptr
    PARAMETER_LIST,       // List_Node of PARAMETER
    ATTRIBUTE,            // leaf: token the Metal attribute text, e.g. "thread_position_in_grid"
    PROGRAM,              // List_Node: top-level items
    TYPE,                 // leaf: token the qualifiers and base type, e.g. "const device f32"
    POINTER,              // Unary_Node: child the pointee type, token any qualifiers after the "*"
    REFERENCE,            // Unary_Node: child the referenced type
    INCLUDE,              // leaf: token the header with its delimiters, e.g. "<metal_stdlib>"
    DEFINE,               // Binary_Node: macro name, replacement LITERAL or nullptr
    USING_NAMESPACE       // leaf: token the namespace, e.g. "metal"
};

// A TYPE token is any qualifiers, then a neutral base type ("f32", "s64", "u8") that
// type_to_string maps per backend. Pointers and references are POINTER / REFERENCE nodes.

enum class Kernel_Backend { METAL, CUDA };

using Kernel_Parse_Body = Parse_Body<Kernel_Node_Type, String>;
using Kernel_Node = Node<Kernel_Parse_Body>;
using Kernel_Unary_Node = Unary_Node<Kernel_Parse_Body>;
using Kernel_Binary_Node = Binary_Node<Kernel_Parse_Body>;
using Kernel_Trinary_Node = Trinary_Node<Kernel_Parse_Body>;
using Kernel_List_Node = List_Node<Kernel_Parse_Body>;

// Neutral type name -> the backend's spelling.
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
// Folds +, -, *, / of two float literals, and negated float literals, into one literal computed
// in double precision.
void fold_constants(Kernel_Node *&node);
// Removes the parameter from function and puts value in place of every use of it.
void bake_parameter(Kernel_List_Node *function, const String &name, const Kernel_Node *value);

struct NML_Dynamics_Expression {
    NML_Declaration_Type source_tag;

    // OnCondition, OnEntry and OnEvent start a group; the StateAssignment, EventOut and
    // Transition entries after them belong to it.
    String target;       // variable written or derived, port, or regime name
    String expression;   // value=, test= or (for a Regime) initial=, verbatim
    String regime_name;  // owning Regime, empty outside any regime
    String condition;    // the owning OnCondition's test, or a Case's condition
    String select;       // DerivedVariable select=, empty when value= is used
    String reduce;       // DerivedVariable reduce=, with select

    NML_Dynamics_Expression(NML_Declaration_Type source_tag)
        : source_tag(source_tag) {};
};

// One entry of a ComponentType's <Structure>, in document order.
//   ChildInstance:   component names the ComponentReference each instance gets its own copy of.
//   With:            instance ("this", "parent", or a Path or child name) is bound to alias.
//   EventConnection: source and target are the from= and to= aliases; unset fields are empty.
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

// A spike-source input's kick when it gives no amplitude: the one-tick current that, through
// the cell's capacitance, lifts the membrane from its lowest start or reset value to 5% past
// threshold.
f64 default_kick_amplitude(const NML_Context &context, const NML_Component_Instance &cell);

// One part of an input's component tree: the input, a Child or Children member, or a
// ChildInstance copy (such as of the synapse it references). Each entry has its own part state.
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

// How one synapse (a prototype) delivers to its targets: what they read of it, and whether that
// needs any state per connection.
struct Synapse_Delivery {
    // The exposure its targets' select reads, e.g. "i"; empty when none of them reads one.
    String read_exposure;
    // The synapse's StateVariable or DerivedVariable behind read_exposure.
    String read_variable;
    // True when read_variable is a StateVariable that only an OnEvent changes, by an increment
    // that reads none of the synapse's state, e.g. accumulated_jump = accumulated_jump + weight * jump.
    // A target's select then reads the sum, over its synapses, of their starting values plus every
    // increment delivered so far, and nothing else of the synapse is observable. So no state is kept
    // per connection: an arrival adds its increment to its target's running sum once, on the tick
    // it arrives, and the synapse does no work between arrivals.
    bool weighted_arrival = false;
    // A weighted arrival's OnEvent value for read_variable, as written in LEMS.
    String arrival_expression;
    // read_variable's OnStart value; each connection adds it to its target's sum at the start.
    f64 starting_value = 0.0;
    // The StateVariables kept per connection, in plane order from FIRST_STATE_VARIABLE_PLANE: all of
    // the type's, or none for a weighted arrival.
    Vector<String> edge_state_variable_names;
    // The cell types of its targets.
    Ordered_Set<String> target_cell_types;
};

// A cell type whose add-select total is read only as `total - tracker`, where tracker is a
// StateVariable whose only assignment is `tracker = total`. Shifting the total and the tracker by
// the same amount changes nothing the cell does, so the kernel keeps only their difference: what
// has reached the cell since the tracker last caught up. The lifetime total, which grows without
// bound and loses its last digits in f32, is never formed. Nothing in the cell reads the tracker
// then, so unless a recording observes it the engine drops it (no cell_state slot, see
// NML_Component_Type::dropped_state_variable_names) and `tracker = total` becomes `difference = 0`.
// A recorded tracker is kept, and `tracker = total` becomes `tracker += difference; difference = 0`,
// so it reads the total caught up to.
struct Tracked_Total {
    String total_variable;    // the add-select DerivedVariable
    String tracker_variable;  // the StateVariable assigned only the total
    bool tracker_kept = false;  // an OutputColumn records the tracker on a cell of the type
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

    // random() calls translated so far. Each reads a placeholder slot until the population or
    // input that runs it is placed.
    s64 random_call_count = 0;
    // Size of the random_values buffer, refilled each tick: one slot per random() call per neuron
    // or input entry that runs it.
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
    Vector<Vector<s64>> input_spike_train_offsets;     // [prototype][train] in input_spike_counts
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

    // Allocates the model buffers. collect_inputs and collect_tracked_totals must have run.
    void allocate_cell_model_memory();
    // Writes every cell's starting state (OnStart values, otherwise 0) into cell_state. An
    // OnStart that calls random() draws for each neuron from random_generator.
    void initialize_cell_state(Random_Generator &random_generator);

    // Refuses a synapse or input that does not provide the input its target cell reads: an
    // exposure the cell's select names, in the same dimension. A synapse's spikes also count as
    // input to a cell with an OnEvent. Called before anything is built.
    void check_cell_inputs() const;

    // Refuses a population whose cell can never fire because its spikes come from child
    // components (a spikeArray's spikes), which the engine runs only for an input. Called before
    // anything is built.
    void check_population_cells() const;

    // network_inputs rows: each tick's input by tick parity (rows 0 and 1). With weighted
    // arrivals or a tracked total, also the increments arriving on a tick, by tick parity (rows 2
    // and 3), and a running sum per neuron (row 4): of the weighted arrivals, which its select reads
    // with row 0 or 1; or, for a cell with a tracked total, the total less its tracker.
    static constexpr s64 WEIGHTED_ARRIVAL_ROW = 2;
    static constexpr s64 RUNNING_SUM_ROW = 4;

    // Per cell type name, the cell types whose totals are tracked (see Tracked_Total): the pattern
    // holds, the tracker starts at 0, and everything reaching their cells delivers increments that
    // start at 0 (weighted arrivals, and inputs whose exposure only accumulates). Filled by
    // collect_tracked_totals, which needs the synapse deliveries and the inputs collected. The engine
    // then drops each tracker that is not kept, before allocate_cell_model_memory sizes cell_state.
    Unordered_Map<String, Tracked_Total> tracked_totals;
    void collect_tracked_totals();

    // Per synapse id. Filled by collect_synapse_deliveries from the connections, so before the
    // engine allocates memory or builds the weight matrix, while it still has them.
    Unordered_Map<String, Synapse_Delivery> synapse_deliveries;
    void collect_synapse_deliveries();
    const Synapse_Delivery &synapse_delivery(const NML_Component_Instance &synapse) const;
    // Whether some synapse delivers weighted arrivals, and whether every synapse with connections does.
    bool has_weighted_arrivals() const;
    bool only_weighted_arrivals() const;
    // Rows in network_inputs: 2, or 5 with weighted arrivals or a tracked total.
    s64 network_input_row_count() const;

    Kernel_Node *create_kernel_root();
    Kernel_Node *translate_kernel_code();
    Kernel_List_Node *translate_component_type(const NML_Component_Type &component_type);
    Kernel_List_Node *translate_component_instance(const NML_Component_Instance &population, s64 first_neuron);

    // A synapse type's dynamics for one edge, with LEMS names still in it. Runs on a neuron's
    // edges while they are active (see synapse_active_ticks); each state variable is a plane.
    // delivered_variable, which its targets read, goes into the target's next input row each tick.
    Kernel_List_Node *translate_synapse_type(const NML_Component_Type &component_type, const String &delivered_variable);
    // The type's template with every state variable read from its plane, weight from the
    // weight plane and every parameter by its value. A weighted arrival's is translate_weighted_arrival.
    Kernel_List_Node *translate_synapse_instance(const NML_Component_Instance &synapse);
    // A weighted arrival on one edge: when a spike reaches the target on the next tick, its OnEvent
    // increment goes into the target's next row of weighted arrivals.
    Kernel_List_Node *translate_weighted_arrival(const NML_Component_Instance &synapse);

    // Weight-matrix planes the kernel uses: weight, delay, then one per StateVariable kept per
    // connection, for the synapse that keeps the most (see Synapse_Delivery).
    s64 edge_plane_count() const;
    // How many of those planes the generated kernel writes: the state-variable planes.
    s64 updated_edge_plane_count() const;
    // Ticks after a neuron spikes during which its outgoing synapses can be away from rest: the
    // longest delay plus the slowest synapse's settle time. -1 when some synapse cannot be left
    // at rest, so every synapse runs every tick. Weighted arrivals have nothing to settle, so when
    // every synapse with connections delivers them it is the longest delay less one, the tick a
    // spike's last arrival is sent.
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
