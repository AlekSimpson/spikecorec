#include "spikecorec/nml/dynamics.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/nml/node.h"
#include "spikecorec/nml/parser.h"

#ifndef SPIKECOREC_METAL_DEVICE_DIR
#define SPIKECOREC_METAL_DEVICE_DIR ""
#endif

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

namespace {

// Names the generated dynamics read from master_step. A LEMS name equal to one of these
// would be captured by the substitutions, so it is refused.
const Set<String> ENGINE_NAMES = {
    "cell_state", "neuron_index", "network_input", "spiked", "refractory", "last_spiked", "tick", "true", "random_values",
    "event_arrivals", "arrival"};

// Names the generated synapse dynamics read from the propagate walk, on top of ENGINE_NAMES.
const Set<String> SYNAPSE_ENGINE_NAMES = {
    "target", "current_edge", "arrived", "next_row", "network_inputs", "neuron_count", "synapse_prototype",
    "basis_u", "basis_v", "edge_coefficients", "rank_float4_stride", "sparse_delta_row_start",
    "sparse_delta_edge_ordinal", "sparse_delta_value", "sparse_delta_capacity", "pending_delta_edge_ordinal",
    "pending_delta_value", "pending_delta_matrix_index", "pending_delta_count", "pending_delta_capacity",
    "projection_first_edge_ordinal", "projection_synapse_prototype", "projection_run_count", "synapse_active_ticks"};

const String DERIVED_PREFIX = "derived_";
const String DERIVATIVE_PREFIX = "derivative_";
// rest_<name> stands for a synapse state variable's OnStart value until the instance fills it in.
const String REST_PREFIX = "rest_";
// random_slot_<n> stands for random() call n's slot in random_values until the population
// that runs it is placed.
const String RANDOM_SLOT_PREFIX = "random_slot_";

// A synapse has settled once every state variable is this close to rest, against the largest
// deviation from rest it reached after the spike.
constexpr f64 REST_FRACTION = 1e-6;

// LEMS function name -> neutral kernel function name. LEMS log is the natural log, like ln.
const UnorderedMap<String, String> LEMS_FUNCTIONS = {
    {"exp", "exp"},   {"ln", "log"},    {"log", "log"},   {"sqrt", "sqrt"},
    {"abs", "fabs"},  {"ceil", "ceil"}, {"floor", "floor"}, {"pow", "pow"},
    {"sin", "sin"},   {"cos", "cos"},   {"tan", "tan"},
    {"sinh", "sinh"}, {"cosh", "cosh"}, {"tanh", "tanh"},
    {"H", "heaviside"},
};

String float_literal(f64 value) {
    if (!std::isfinite(value)) throw runtime_error("Cannot bake the non-finite value " + std::to_string(value) + " into a kernel");
    std::ostringstream text;
    text << std::setprecision(9) << value;
    String literal = text.str();
    if (literal.find_first_of(".e") == String::npos) literal += ".0";
    return literal + "f";
}

KernelNode *identifier(const String &name) {
    return new_node(KernelNodeType::IDENTIFIER, name);
}

KernelNode *literal(const String &text) {
    return new_node(KernelNodeType::LITERAL, text);
}

bool starts_with(const String &text, const String &prefix) {
    return text.rfind(prefix, 0) == 0;
}

KernelListNode *function_call(const String &name, const Vector<KernelNode *> &arguments) {
    KernelListNode *call = new_node<ListNode>(KernelNodeType::FUNCTION_CALL, name);
    call->children = arguments;
    return call;
}

// The walking thread's neuron is the edge's source.
KernelNode *edge_source() {
    return new_cast(new_type("s32"), identifier("neuron_index"));
}

// The current edge's value in plane, read through the weight matrix.
KernelNode *load_edge(s64 plane) {
    return function_call("spikecorec_load_edge", {
        identifier("basis_u"), identifier("basis_v"), identifier("edge_coefficients"), identifier("rank_float4_stride"),
        identifier("sparse_delta_row_start"), identifier("sparse_delta_edge_ordinal"), identifier("sparse_delta_value"),
        identifier("neuron_count"), identifier("sparse_delta_capacity"), literal(std::to_string(plane)),
        edge_source(), identifier("target"), identifier("current_edge")});
}

// Adds delta to the current edge's value in plane.
KernelNode *accumulate_edge(s64 plane, KernelNode *delta) {
    return function_call("spikecorec_accumulate_edge", {
        identifier("sparse_delta_row_start"), identifier("sparse_delta_edge_ordinal"), identifier("sparse_delta_value"),
        identifier("pending_delta_edge_ordinal"), identifier("pending_delta_value"),
        identifier("pending_delta_matrix_index"), identifier("pending_delta_count"), identifier("pending_delta_capacity"),
        identifier("neuron_count"), identifier("sparse_delta_capacity"), literal(std::to_string(plane)),
        edge_source(), identifier("current_edge"), delta});
}

// A name the generated synapse dynamics use themselves.
bool is_synapse_engine_name(const String &name) {
    return ENGINE_NAMES.count(name) || SYNAPSE_ENGINE_NAMES.count(name) || starts_with(name, DERIVED_PREFIX) ||
           starts_with(name, DERIVATIVE_PREFIX) || starts_with(name, REST_PREFIX) || starts_with(name, RANDOM_SLOT_PREFIX);
}

// Every name a LEMS expression reads; function names are not reads.
void collect_lems_identifiers(const LemsParseNode *node, Set<String> &names) {
    if (!node) return;
    if (node->body.syntax_type == LemsNodeSubtype::IDENTIFIER) {
        names.insert(node->body.token.lexeme);
        return;
    }
    if (node->body.syntax_type == LemsNodeSubtype::FUNCTION_CALL) {
        for (const LemsParseNode *argument : children_of(static_cast<const BinaryNode<LemsParseBody> *>(node)->right)) {
            collect_lems_identifiers(argument, names);
        }
        return;
    }
    if (const auto *unary = dynamic_cast<const UnaryNode<LemsParseBody> *>(node)) {
        collect_lems_identifiers(unary->child, names);
    } else if (const auto *binary = dynamic_cast<const BinaryNode<LemsParseBody> *>(node)) {
        collect_lems_identifiers(binary->left, names);
        collect_lems_identifiers(binary->right, names);
    }
}

// Whether a LEMS expression calls function_name anywhere.
bool calls_function(const LemsParseNode *node, const String &function_name) {
    if (!node) return false;
    if (node->body.syntax_type == LemsNodeSubtype::FUNCTION_CALL) {
        if (node->body.token.lexeme == function_name) return true;
        for (const LemsParseNode *argument : children_of(static_cast<const BinaryNode<LemsParseBody> *>(node)->right)) {
            if (calls_function(argument, function_name)) return true;
        }
        return false;
    }
    if (const auto *unary = dynamic_cast<const UnaryNode<LemsParseBody> *>(node)) {
        return calls_function(unary->child, function_name);
    }
    if (const auto *binary = dynamic_cast<const BinaryNode<LemsParseBody> *>(node)) {
        return calls_function(binary->left, function_name) || calls_function(binary->right, function_name);
    }
    return false;
}

} // namespace

// Host-side value of a LEMS expression, for OnStart values and derived parameters.
f64 evaluate_lems(const LemsParseNode *node, const UnorderedMap<String, f64> &values, const String &owner_name,
                  RandomGenerator *random_generator) {
    const String &lexeme = node->body.token.lexeme;
    switch (node->body.syntax_type) {
        case LemsNodeSubtype::FLOAT:
        case LemsNodeSubtype::INT:
            return std::stod(lexeme);

        case LemsNodeSubtype::IDENTIFIER: {
            auto value = values.find(lexeme);
            if (value == values.end()) throw runtime_error("'" + lexeme + "' has no value in " + owner_name);
            return value->second;
        }

        case LemsNodeSubtype::OPERATOR: {
            if (const auto *unary = dynamic_cast<const UnaryNode<LemsParseBody> *>(node)) {
                const f64 operand = evaluate_lems(unary->child, values, owner_name, random_generator);
                return lexeme == "-" ? -operand : operand;
            }
            const auto *binary = static_cast<const BinaryNode<LemsParseBody> *>(node);
            const f64 left = evaluate_lems(binary->left, values, owner_name, random_generator);
            const f64 right = evaluate_lems(binary->right, values, owner_name, random_generator);
            if (lexeme == "+") return left + right;
            if (lexeme == "-") return left - right;
            if (lexeme == "*") return left * right;
            if (lexeme == "/") return left / right;
            if (lexeme == ">") return left > right ? 1.0 : 0.0;
            if (lexeme == "<") return left < right ? 1.0 : 0.0;
            if (lexeme == ">=") return left >= right ? 1.0 : 0.0;
            if (lexeme == "<=") return left <= right ? 1.0 : 0.0;
            if (lexeme == "==") return left == right ? 1.0 : 0.0;
            if (lexeme == "!=") return left != right ? 1.0 : 0.0;
            if (lexeme == "&&") return (left != 0.0 && right != 0.0) ? 1.0 : 0.0;
            if (lexeme == "||") return (left != 0.0 || right != 0.0) ? 1.0 : 0.0;
            throw runtime_error("Operator '" + lexeme + "' cannot be evaluated in " + owner_name);
        }

        case LemsNodeSubtype::FUNCTION_CALL: {
            static const UnorderedMap<String, f64 (*)(f64)> functions = {
                {"exp", [](f64 value) { return std::exp(value); }},
                {"ln", [](f64 value) { return std::log(value); }},
                {"log", [](f64 value) { return std::log(value); }},
                {"sqrt", [](f64 value) { return std::sqrt(value); }},
                {"abs", [](f64 value) { return std::fabs(value); }},
                {"ceil", [](f64 value) { return std::ceil(value); }},
                {"floor", [](f64 value) { return std::floor(value); }},
                {"sin", [](f64 value) { return std::sin(value); }},
                {"cos", [](f64 value) { return std::cos(value); }},
                {"tan", [](f64 value) { return std::tan(value); }},
                {"sinh", [](f64 value) { return std::sinh(value); }},
                {"cosh", [](f64 value) { return std::cosh(value); }},
                {"tanh", [](f64 value) { return std::tanh(value); }},
                {"H", [](f64 value) { return value > 0.0 ? 1.0 : 0.0; }},  // H(0) is 0
            };
            const auto *call = static_cast<const BinaryNode<LemsParseBody> *>(node);
            Vector<f64> arguments;
            for (const LemsParseNode *argument : children_of(call->right)) {
                arguments.push_back(evaluate_lems(argument, values, owner_name, random_generator));
            }
            if (lexeme == "pow" && arguments.size() == 2) return std::pow(arguments[0], arguments[1]);
            if (lexeme == "random" && arguments.size() == 1) {
                if (!random_generator) {
                    throw runtime_error("random() in " + owner_name + " has no random generator to draw from: only a cell's "
                                        "OnStart and dynamics may call it");
                }
                return arguments[0] * random_generator->uniform();
            }
            auto function = functions.find(lexeme);
            if (function == functions.end() || arguments.size() != 1) {
                throw runtime_error("Function '" + lexeme + "' cannot be evaluated in " + owner_name);
            }
            return function->second(arguments[0]);
        }

        default:
            throw runtime_error("Malformed LEMS expression in " + owner_name);
    }
}

f64 evaluate_lems(const String &expression, const UnorderedMap<String, f64> &values, const String &owner_name,
                  RandomGenerator *random_generator) {
    std::unique_ptr<LemsParseNode> tree(parse_lems_expression(expression, owner_name));
    return evaluate_lems(tree.get(), values, owner_name, random_generator);
}

// Every Parameter, Constant and DerivedParameter the cell's type resolves, in SI units. An
// unset parameter is left out; using one fails as an unresolved name.
UnorderedMap<String, f64> component_parameter_values(const NML_Context &context, const NML_ComponentInstance &cell) {
    const NML_ComponentType &component_type = *cell.component_type;
    UnorderedMap<String, f64> values;
    Vector<const NML_Node *> derived_parameters;

    for (const NML_ComponentType *type = &component_type; type; type = type->extends) {
        for (const NML_Node *element : children_of(type->source_node)) {
            const NML_Tag &tag = element->body;
            if (tag.tag_type != NML_DeclarationType::Parameter && tag.tag_type != NML_DeclarationType::Constant &&
                tag.tag_type != NML_DeclarationType::DerivedParameter) {
                continue;
            }
            // A subtype's declaration of the same name shadows this one.
            if (component_type.find_declaration(tag.namespace_key()) != element) continue;

            const String name = tag.get_attribute("name");
            if (tag.tag_type == NML_DeclarationType::Parameter) {
                if (cell.has_value(name)) values[name] = context.resolve_quantity(cell.value_or(name));
            } else if (tag.tag_type == NML_DeclarationType::Constant) {
                values[name] = context.resolve_quantity(tag.get_attribute("value"));
            } else if (tag.has_attribute("value")) {
                derived_parameters.push_back(element);
            }
        }
    }

    // Derived parameters can use each other, so each pass resolves those whose inputs are known.
    bool resolved_one = true;
    while (resolved_one && !derived_parameters.empty()) {
        resolved_one = false;
        for (usize index = 0; index < derived_parameters.size();) {
            const NML_Tag &tag = derived_parameters[index]->body;
            std::unique_ptr<LemsParseNode> tree(parse_lems_expression(tag.get_attribute("value"), component_type.name));

            Set<String> inputs;
            collect_lems_identifiers(tree.get(), inputs);
            const bool inputs_known = std::all_of(inputs.begin(), inputs.end(),
                    [&values](const String &input) { return values.count(input) != 0; });
            if (!inputs_known) {
                index += 1;
                continue;
            }

            values[tag.get_attribute("name")] = evaluate_lems(tree.get(), values, component_type.name);
            derived_parameters.erase(derived_parameters.begin() + index);
            resolved_one = true;
        }
    }
    return values;
}

UnorderedMap<String, f64> starting_values(const NML_Context &context, const NML_ComponentInstance &cell,
                                          RandomGenerator *random_generator) {
    const NML_ComponentType &component_type = *cell.component_type;
    const Vector<String> &variable_names = component_type.state_variable_names;

    UnorderedMap<String, f64> values = component_parameter_values(context, cell);
    for (const String &name : variable_names) values[name] = 0.0;

    // In order, so a value may use the state assigned above it.
    for (const NML_DynamicsExpression &entry : component_type.dynamics) {
        if (entry.source_tag != NML_DeclarationType::OnStart) continue;
        if (std::find(variable_names.begin(), variable_names.end(), entry.target) == variable_names.end()) {
            throw runtime_error("OnStart in " + component_type.name + " assigns '" + entry.target +
                                "', which is not a state variable");
        }
        values[entry.target] = evaluate_lems(entry.expression, values, component_type.name, random_generator);
    }
    return values;
}

// The target network's populations, in document order.
Vector<const NML_ComponentInstance *> network_populations(const NML_Context &context) {
    Vector<const NML_ComponentInstance *> populations;
    const NML_ComponentInstance *network = context.find_instance(context.simulation.target_network_id);
    if (!network) return populations;

    for (const String &child_id : network->data_order) {
        const NML_ComponentInstance *population = context.find_instance(child_id);
        if (context.is_instance_of(population, "population")) populations.push_back(population);
    }
    return populations;
}

// Rows in the spike_history ring. Every delay is at least one tick and a row written this
// tick is never one a delayed arrival reads, so the ring is never shorter than two.
s64 spike_history_length(const NML_Context &context) {
    return std::max<s64>(2, context.simulation.maximum_edge_delay + 1);
}

const NML_ComponentInstance &population_cell(const NML_Context &context, const NML_ComponentInstance &population) {
    const NML_ComponentInstance *cell = context.find_instance(population.value_or("component"));
    if (!cell || !cell->component_type) {
        throw runtime_error("Population '" + population.id + "' references no declared cell");
    }
    return *cell;
}

namespace {

// True when the type reacts to arriving spikes itself, with an OnEvent in its own dynamics.
bool has_on_event(const NML_ComponentType &component_type) {
    for (const NML_DynamicsExpression &entry : component_type.dynamics) {
        if (entry.source_tag == NML_DeclarationType::OnEvent) return true;
    }
    return false;
}

// The child slots of node that hold uses of names. Declared names, parameters, types,
// function names and member names are left out.
Vector<KernelNode **> use_slots(KernelNode *node) {
    Vector<KernelNode **> slots;
    switch (node->body.syntax_type) {
        case KernelNodeType::LITERAL:
        case KernelNodeType::IDENTIFIER:
        case KernelNodeType::PARAMETER:
        case KernelNodeType::PARAMETER_LIST:
        case KernelNodeType::ATTRIBUTE:
        case KernelNodeType::TYPE:
        case KernelNodeType::POINTER:
        case KernelNodeType::REFERENCE:
        case KernelNodeType::INCLUDE:
        case KernelNodeType::DEFINE:
        case KernelNodeType::USING_NAMESPACE:
            return slots;

        case KernelNodeType::DECLARATION: {
            KernelTrinaryNode *declaration = static_cast<KernelTrinaryNode *>(node);
            // An array's size is a use; its name is not.
            if (declaration->middle->body.syntax_type == KernelNodeType::EXPRESSION) {
                slots.push_back(&static_cast<KernelBinaryNode *>(declaration->middle)->right);
            }
            slots.push_back(&declaration->right);
            return slots;
        }

        case KernelNodeType::CAST:
            slots.push_back(&static_cast<KernelBinaryNode *>(node)->right);
            return slots;

        case KernelNodeType::DEVICE_FUNCTION_IMPL:
        case KernelNodeType::KERNEL_FUNCTION_IMPL:
            slots.push_back(&static_cast<KernelListNode *>(node)->children[3]);
            return slots;

        default:
            break;
    }

    if (node->body.syntax_type == KernelNodeType::EXPRESSION && (node->body.token == "." || node->body.token == "->")) {
        slots.push_back(&static_cast<KernelBinaryNode *>(node)->left);
        return slots;
    }

    if (KernelListNode *list = dynamic_cast<KernelListNode *>(node)) {
        for (KernelNode *&child : list->children) slots.push_back(&child);
    } else if (KernelTrinaryNode *trinary = dynamic_cast<KernelTrinaryNode *>(node)) {
        slots.push_back(&trinary->left);
        slots.push_back(&trinary->middle);
        slots.push_back(&trinary->right);
    } else if (KernelBinaryNode *binary = dynamic_cast<KernelBinaryNode *>(node)) {
        slots.push_back(&binary->left);
        slots.push_back(&binary->right);
    } else if (KernelUnaryNode *unary = dynamic_cast<KernelUnaryNode *>(node)) {
        slots.push_back(&unary->child);
    }
    return slots;
}

} // namespace

bool Codegen::cells_receive_events() const {
    for (const NML_ComponentInstance *population : network_populations(context)) {
        if (has_on_event(*population_cell(context, *population).component_type)) return true;
    }
    return false;
}

void Codegen::allocate_cell_model_memory() {
    // Two rows like network_inputs, but only for a model whose cells have an OnEvent; otherwise
    // one slot, so there is still a buffer to bind.
    const s64 event_arrival_slots = cells_receive_events() ? 2 * context.simulation.total_neuron_count : 1;

    device->partition(sizeof(f32) * context.get_cell_state_size(), EngineDatatype::FLOAT32, data_partitions)            // cell_state
          .partition(sizeof(f32) * 2 * context.simulation.total_neuron_count, EngineDatatype::FLOAT32, data_partitions)    // network_inputs
          .partition(sizeof(u8) * spike_history_length(context) * context.simulation.total_neuron_count,
                     EngineDatatype::UNSIGNED8, data_partitions)                                                           // spike_history
          .partition(sizeof(s64) * context.simulation.total_neuron_count, EngineDatatype::SIGNED64, data_partitions)       // last_spiked
          .partition(sizeof(f32), EngineDatatype::FLOAT32, data_partitions)                                                // empty_edge_plane
          .partition(sizeof(u32) * event_arrival_slots, EngineDatatype::UNSIGNED32, data_partitions);                      // event_arrival_count

    EnginePointer slab = device->allocate(data_partitions);
    data_partitions.push_back(slab);

    log::logger().debug("Codegen: {} bytes: cell_state {}, network_inputs {}, spike_history {}, last_spiked {}, "
                        "event_arrival_count {}",
                        slab.total_bytes, context.get_cell_state_size(), 2 * context.simulation.total_neuron_count,
                        spike_history_length(context) * context.simulation.total_neuron_count,
                        context.simulation.total_neuron_count, event_arrival_slots);
}

Codegen::~Codegen() {
    delete root;
    for (auto &[type_name, template_body] : component_type_templates) delete template_body;
    for (auto &[type_name, template_body] : synapse_type_templates) delete template_body;
}

void Codegen::initialize_cell_state(RandomGenerator &random_generator) {
    if (data_partitions.empty()) throw runtime_error("initialize_cell_state: cell memory is not allocated");
    f32 *cell_state = data_partitions[0].get_contents_as<f32>();

    for (const NML_ComponentInstance *population : network_populations(context)) {
        const NML_ComponentInstance &cell = population_cell(context, *population);
        const NML_ComponentType &component_type = *cell.component_type;
        const Vector<String> &variable_names = component_type.state_variable_names;
        const s64 population_base = context.simulation.population_base_indices.at(population->id);
        const s64 population_size = context.get_population_size(population);
        const s64 variable_count = static_cast<s64>(variable_names.size());

        // Every neuron starts the same, unless an OnStart calls random(); then each draws its own.
        bool draws_on_start = false;
        for (const NML_DynamicsExpression &entry : component_type.dynamics) {
            if (entry.source_tag != NML_DeclarationType::OnStart) continue;
            const std::unique_ptr<LemsParseNode> tree(parse_lems_expression(entry.expression, component_type.name));
            draws_on_start = draws_on_start || calls_function(tree.get(), "random");
        }

        Vector<f32> starting_state;
        for (s64 local_index = 0; local_index < population_size; local_index += 1) {
            if (local_index == 0 || draws_on_start) {
                const UnorderedMap<String, f64> values = starting_values(context, cell, &random_generator);
                starting_state.clear();
                for (const String &name : variable_names) starting_state.push_back(static_cast<f32>(values.at(name)));
            }
            std::copy(starting_state.begin(), starting_state.end(),
                      cell_state + population_base + local_index * variable_count);
        }
    }
}

// AST construction

KernelNode *new_type(const String &type) {
    return new_node(KernelNodeType::TYPE, type);
}

KernelNode *new_pointer(KernelNode *pointee) {
    KernelUnaryNode *pointer = new_node<UnaryNode>(KernelNodeType::POINTER, "");
    pointer->child = pointee;
    return pointer;
}

KernelNode *new_declaration(KernelNode *type, const String &name, KernelNode *value) {
    KernelTrinaryNode *declaration = new_node<TrinaryNode>(KernelNodeType::DECLARATION, "");
    declaration->left = type;
    declaration->middle = identifier(name);
    declaration->right = value;
    return declaration;
}

KernelNode *new_assignment(KernelNode *target, const String &assignment_operator, KernelNode *value) {
    KernelBinaryNode *assignment = new_node<BinaryNode>(KernelNodeType::ASSIGNMENT, assignment_operator);
    assignment->left = target;
    assignment->right = value;
    return assignment;
}

KernelNode *new_expression(const String &expression_operator, KernelNode *left, KernelNode *right) {
    KernelBinaryNode *expression = new_node<BinaryNode>(KernelNodeType::EXPRESSION, expression_operator);
    expression->left = left;
    expression->right = right;
    return expression;
}

KernelNode *new_unary_expression(const String &expression_operator, KernelNode *operand) {
    KernelUnaryNode *expression = new_node<UnaryNode>(KernelNodeType::EXPRESSION, expression_operator);
    expression->child = operand;
    return expression;
}

KernelNode *new_cast(KernelNode *type, KernelNode *value) {
    KernelBinaryNode *cast = new_node<BinaryNode>(KernelNodeType::CAST, "");
    cast->left = type;
    cast->right = value;
    return cast;
}

KernelListNode *new_block() {
    return new_node<ListNode>(KernelNodeType::BLOCK, "");
}

KernelListNode *new_conditional() {
    return new_node<ListNode>(KernelNodeType::CONDITIONAL, "");
}

void add_branch(KernelListNode *conditional, KernelNode *test, KernelNode *body) {
    if (body->body.syntax_type != KernelNodeType::BLOCK) throw runtime_error("Conditional branch body must be a BLOCK");
    if (!test && conditional->children.empty()) throw runtime_error("Conditional cannot start with else");
    if (!conditional->children.empty() && !static_cast<KernelBinaryNode *>(conditional->children.back())->left)
        throw runtime_error("Conditional already ends with else");

    KernelBinaryNode *branch = new_node<BinaryNode>(KernelNodeType::CONDITIONAL_BRANCH, "");
    branch->left = test;
    branch->right = body;
    conditional->children.push_back(branch);
}

// Tree operations

KernelListNode *find_function(KernelNode *program, const String &name) {
    for (KernelNode *item : static_cast<KernelListNode *>(program)->children) {
        const KernelNodeType type = item->body.syntax_type;
        if (type != KernelNodeType::DEVICE_FUNCTION_IMPL && type != KernelNodeType::KERNEL_FUNCTION_IMPL) continue;

        KernelListNode *function = static_cast<KernelListNode *>(item);
        if (function->children[1]->body.token == name) return function;
    }
    return nullptr;
}

s64 find_statement(KernelListNode *block, const String &name) {
    for (usize index = 0; index < block->children.size(); index += 1) {
        KernelNode *statement = block->children[index];
        KernelNode *named = nullptr;
        if (statement->body.syntax_type == KernelNodeType::DECLARATION) {
            named = static_cast<KernelTrinaryNode *>(statement)->middle;
            if (named->body.syntax_type == KernelNodeType::EXPRESSION) named = static_cast<KernelBinaryNode *>(named)->left;
        } else if (statement->body.syntax_type == KernelNodeType::ASSIGNMENT) {
            named = static_cast<KernelBinaryNode *>(statement)->left;
        }
        if (named && named->body.syntax_type == KernelNodeType::IDENTIFIER && named->body.token == name) {
            return static_cast<s64>(index);
        }
    }
    return -1;
}

void insert_statements(KernelListNode *block, usize index, const Vector<KernelNode *> &statements) {
    if (index > block->children.size()) throw runtime_error("insert_statements: index past the end of the block");
    block->children.insert(block->children.begin() + index, statements.begin(), statements.end());
}

KernelNode *clone(const KernelNode *node) {
    if (!node) return nullptr;

    if (const KernelListNode *list = dynamic_cast<const KernelListNode *>(node)) {
        KernelListNode *copy = new KernelListNode(list->body);
        for (const KernelNode *child : list->children) copy->children.push_back(clone(child));
        return copy;
    }
    if (const KernelTrinaryNode *trinary = dynamic_cast<const KernelTrinaryNode *>(node)) {
        KernelTrinaryNode *copy = new KernelTrinaryNode(trinary->body);
        copy->left = clone(trinary->left);
        copy->middle = clone(trinary->middle);
        copy->right = clone(trinary->right);
        return copy;
    }
    if (const KernelBinaryNode *binary = dynamic_cast<const KernelBinaryNode *>(node)) {
        KernelBinaryNode *copy = new KernelBinaryNode(binary->body);
        copy->left = clone(binary->left);
        copy->right = clone(binary->right);
        return copy;
    }
    if (const KernelUnaryNode *unary = dynamic_cast<const KernelUnaryNode *>(node)) {
        KernelUnaryNode *copy = new KernelUnaryNode(unary->body);
        copy->child = clone(unary->child);
        return copy;
    }
    return new KernelNode(node->body);
}

void replace_identifier(KernelNode *&node, const String &name, const KernelNode *replacement) {
    if (!node) return;
    if (node->body.syntax_type == KernelNodeType::IDENTIFIER) {
        if (node->body.token != name) return;
        delete node;
        node = clone(replacement);
        return;
    }
    for (KernelNode **slot : use_slots(node)) replace_identifier(*slot, name, replacement);
}

void collect_identifiers(const KernelNode *node, Set<String> &names) {
    if (!node) return;
    if (node->body.syntax_type == KernelNodeType::IDENTIFIER) {
        names.insert(node->body.token);
        return;
    }
    for (KernelNode **slot : use_slots(const_cast<KernelNode *>(node))) collect_identifiers(*slot, names);
}

void bake_parameter(KernelListNode *function, const String &name, const KernelNode *value) {
    KernelListNode *parameters = static_cast<KernelListNode *>(function->children[2]);
    auto parameter = std::find_if(parameters->children.begin(), parameters->children.end(), [&name](KernelNode *candidate) {
        return static_cast<KernelTrinaryNode *>(candidate)->middle->body.token == name;
    });
    if (parameter == parameters->children.end()) {
        throw runtime_error("bake_parameter: " + function->children[1]->body.token + " has no parameter '" + name + "'");
    }
    delete *parameter;
    parameters->children.erase(parameter);
    replace_identifier(function->children[3], name, value);
}

// AST generation

KernelNode *Codegen::create_kernel_root() {
    const char *boilerplate = backend == KernelBackend::METAL ? METAL_KERNEL_BOILERPLATE : CUDA_KERNEL_BOILERPLATE;
    std::unique_ptr<KernelListNode> program(KernelParser(tokenize_kernel(boilerplate, backend), backend).parse_program());

    // Metal's k^2-tree helpers are shared with the precompiled kernels, so they stay in their own
    // file and go in ahead of the first function that calls them.
    if (backend == KernelBackend::METAL) {
        const String helpers_path = String(SPIKECOREC_METAL_DEVICE_DIR) + "/k2tree_device.metalinc";
        std::ifstream helpers_file(helpers_path);
        if (!helpers_file) throw runtime_error("create_kernel_root: cannot read " + helpers_path);
        std::stringstream helpers_text;
        helpers_text << helpers_file.rdbuf();

        std::unique_ptr<KernelListNode> helpers(
                KernelParser(tokenize_kernel(helpers_text.str(), backend), backend).parse_program());

        auto first_function = std::find_if(program->children.begin(), program->children.end(), [](KernelNode *item) {
            return item->body.syntax_type == KernelNodeType::DEVICE_FUNCTION_IMPL ||
                   item->body.syntax_type == KernelNodeType::KERNEL_FUNCTION_IMPL;
        });
        program->children.insert(first_function, helpers->children.begin(), helpers->children.end());
        helpers->children.clear();
    }

    delete root;
    root = program.release();
    return root;
}

KernelNode *Codegen::translate_kernel_code() {
    create_kernel_root();

    KernelListNode *master_step = find_function(root, "master_step");
    if (!master_step) throw runtime_error("translate_kernel_code: the boilerplate has no master_step");
    KernelListNode *body = static_cast<KernelListNode *>(master_step->children[3]);

    // One branch per population, over its range of neurons.
    KernelListNode *dispatch = new_conditional();
    s64 first_neuron = 0;
    try {
        for (const NML_ComponentInstance *population : network_populations(context)) {
            const s64 population_size = context.get_population_size(population);
            if (population_size == 0) continue;

            KernelListNode *population_body = translate_component_instance(*population, first_neuron);
            KernelNode *test = new_expression("<", identifier("neuron_index"),
                                              literal(std::to_string(first_neuron + population_size)));
            add_branch(dispatch, test, population_body);
            first_neuron += population_size;
        }
    } catch (...) {
        delete dispatch;
        throw;
    }

    s64 spiked_index = find_statement(body, "spiked");
    if (spiked_index < 0) {
        delete dispatch;
        throw runtime_error("translate_kernel_code: master_step declares no 'spiked'");
    }

    // Spikes that reached each cell are counted like network_inputs: two rows by tick parity.
    // A neuron takes this tick's count from current_row while this tick's arrivals are counted
    // into next_row.
    const bool receives_events = cells_receive_events();
    if (receives_events) {
        KernelNode *arrival_slot = new_expression("[]", identifier("event_arrival_count"), identifier("input_slot"));
        insert_statements(body, static_cast<usize>(spiked_index),
                          {new_declaration(new_type("const s32"), "event_arrivals", new_cast(new_type("s32"), arrival_slot)),
                           new_assignment(clone(arrival_slot), "=", literal("0"))});
        spiked_index += 2;
    }

    if (dispatch->children.empty()) {
        delete dispatch;
    } else {
        insert_statements(body, static_cast<usize>(spiked_index) + 1, {dispatch});
    }

    // Each edge runs its prototype's synapse, numbered in synapse_instances order as the
    // engine numbers the projection runs.
    const Vector<NML_ComponentInstance> &synapses = context.simulation.synapse_instances;
    if (!synapses.empty()) {
        KernelListNode *walk = nullptr;
        s64 arrived_index = -1;
        for (KernelNode *statement : body->children) {
            if (statement->body.syntax_type != KernelNodeType::WHILE) continue;
            KernelListNode *loop_body = static_cast<KernelListNode *>(static_cast<KernelBinaryNode *>(statement)->right);
            arrived_index = find_statement(loop_body, "arrived");
            if (arrived_index < 0) continue;
            walk = loop_body;
            break;
        }
        if (!walk) throw runtime_error("translate_kernel_code: master_step's propagate walk declares no 'arrived'");

        Vector<KernelNode *> statements;
        if (synapses.size() == 1) {
            KernelListNode *synapse_body = translate_synapse_instance(synapses[0]);
            statements = synapse_body->children;
            synapse_body->children.clear();
            delete synapse_body;
        } else {
            KernelListNode *prototypes = new_conditional();
            try {
                for (usize prototype = 0; prototype < synapses.size(); prototype += 1) {
                    add_branch(prototypes,
                               new_expression("==", identifier("synapse_prototype"), literal(std::to_string(prototype))),
                               translate_synapse_instance(synapses[prototype]));
                }
            } catch (...) {
                delete prototypes;
                throw;
            }
            statements.push_back(new_declaration(
                    new_type("const s32"), "synapse_prototype",
                    function_call("spikecorec_synapse_prototype",
                                  {identifier("projection_first_edge_ordinal"), identifier("projection_synapse_prototype"),
                                   identifier("projection_run_count"), identifier("current_edge")})));
            statements.push_back(prototypes);
        }
        insert_statements(walk, static_cast<usize>(arrived_index) + 1, statements);

        if (receives_events) {
            KernelNode *next_slot = new_expression(
                    "[]", identifier("event_arrival_count"),
                    new_expression("+", new_expression("*", identifier("next_row"), identifier("neuron_count")),
                                   identifier("target")));
            KernelListNode *count_arrival = new_block();
            count_arrival->children.push_back(function_call("atomic_increment", {next_slot}));
            KernelListNode *on_arrival = new_conditional();
            add_branch(on_arrival, identifier("arrived"), count_arrival);
            walk->children.push_back(on_arrival);
        }
    }

    // Constant for the whole run, so baked in rather than bound. Last, so the inserted
    // dynamics' uses are baked too.
    const std::unique_ptr<KernelNode> neuron_count(literal(std::to_string(context.simulation.total_neuron_count)));
    const std::unique_ptr<KernelNode> history_length(literal(std::to_string(spike_history_length(context))));
    bake_parameter(master_step, "neuron_count", neuron_count.get());
    bake_parameter(master_step, "spike_history_length", history_length.get());
    return root;
}

namespace {

// Each derived variable declared once as derived_<name>, placed so a variable comes after
// the ones it reads. A select's value comes from selected_value.
void place_derived_variables(KernelListNode *body, const Vector<String> &derived_names,
                             const UnorderedMap<String, Vector<const NML_DynamicsExpression *>> &derived_entries,
                             const std::function<KernelNode *(const String &)> &translate,
                             const std::function<KernelNode *(const NML_DynamicsExpression &)> &selected_value,
                             const std::function<runtime_error(const String &)> &unsupported,
                             Set<String> &lems_names) {
    UnorderedMap<String, Vector<KernelNode *>> derived_statements;
    UnorderedMap<String, Set<String>> derived_dependencies;
    for (const String &name : derived_names) {
        const Vector<const NML_DynamicsExpression *> &entries = derived_entries.at(name);
        Vector<KernelNode *> &statements = derived_statements[name];
        lems_names.insert(name);

        if (entries.front()->source_tag == NML_DeclarationType::DerivedVariable) {
            const NML_DynamicsExpression &entry = *entries.front();
            KernelNode *value = nullptr;
            if (!entry.select.empty()) {
                value = selected_value(entry);
            } else if (!entry.expression.empty()) {
                value = translate(entry.expression);
            } else {
                throw unsupported("DerivedVariable '" + name + "' has neither value nor select");
            }
            statements.push_back(new_declaration(new_type("const f32"), DERIVED_PREFIX + name, value));
        } else {
            statements.push_back(new_declaration(new_type("f32"), DERIVED_PREFIX + name, literal("0.0f")));
            KernelListNode *cases = new_conditional();
            for (const NML_DynamicsExpression *entry : entries) {
                KernelListNode *case_body = new_block();
                case_body->children.push_back(
                        new_assignment(identifier(DERIVED_PREFIX + name), "=", translate(entry->expression)));
                add_branch(cases, entry->condition.empty() ? nullptr : translate(entry->condition), case_body);
            }
            statements.push_back(cases);
        }

        Set<String> used_names;
        for (const KernelNode *statement : statements) collect_identifiers(statement, used_names);
        for (const String &used_name : used_names) {
            if (used_name != name && derived_entries.count(used_name)) derived_dependencies[name].insert(used_name);
        }
    }

    UnorderedMap<String, s32> visit_state;
    std::function<void(const String &)> place_derived = [&](const String &name) {
        if (visit_state[name] == 2) return;
        if (visit_state[name] == 1) throw unsupported("derived variable '" + name + "' depends on itself");
        visit_state[name] = 1;
        for (const String &dependency : derived_dependencies[name]) place_derived(dependency);
        visit_state[name] = 2;
        for (KernelNode *statement : derived_statements[name]) body->children.push_back(statement);
    };
    for (const String &name : derived_names) place_derived(name);
}

// Ticks a synapse needs after a spike arrives to settle back at rest: every state variable within
// REST_FRACTION of the largest deviation from rest it reached. Simulated on the host the way the
// kernel integrates it, with the given weight. -1 when it cannot be skipped at rest: its OnStart
// values are not a rest state (a derivative or the current is not zero there), or it has not
// settled within the run.
s64 synapse_settle_ticks(const NML_Context &context, const NML_ComponentInstance &synapse, f64 weight) {
    const NML_ComponentType &component_type = *synapse.component_type;
    const String &owner_name = component_type.name;
    const Vector<String> &variable_names = component_type.state_variable_names;
    const f64 step_dt = context.simulation.step_dt;

    struct ParsedEntry {
        String target;
        std::unique_ptr<LemsParseNode> condition;
        std::unique_ptr<LemsParseNode> value;
    };
    auto parse = [&owner_name](const String &expression) {
        return std::unique_ptr<LemsParseNode>(expression.empty() ? nullptr : parse_lems_expression(expression, owner_name));
    };

    Vector<String> derived_names;
    UnorderedMap<String, Vector<ParsedEntry>> derived_entries;
    Vector<ParsedEntry> derivatives;
    Vector<ParsedEntry> event_assignments;
    bool in_event = false;
    for (const NML_DynamicsExpression &entry : component_type.dynamics) {
        switch (entry.source_tag) {
            case NML_DeclarationType::DerivedVariable:
            case NML_DeclarationType::Case:
                if (derived_entries.count(entry.target) == 0) derived_names.push_back(entry.target);
                derived_entries[entry.target].push_back({entry.target, parse(entry.condition), parse(entry.expression)});
                in_event = false;
                break;
            case NML_DeclarationType::TimeDerivative:
                derivatives.push_back({entry.target, nullptr, parse(entry.expression)});
                in_event = false;
                break;
            case NML_DeclarationType::OnEvent:
                in_event = true;
                break;
            case NML_DeclarationType::StateAssignment:
                if (in_event) event_assignments.push_back({entry.target, nullptr, parse(entry.expression)});
                break;
            default:
                in_event = false;
                break;
        }
    }

    // Derived variables in an order where each comes after the ones it reads.
    Vector<String> derived_order;
    while (derived_order.size() < derived_names.size()) {
        const usize placed = derived_order.size();
        for (const String &name : derived_names) {
            if (std::find(derived_order.begin(), derived_order.end(), name) != derived_order.end()) continue;
            Set<String> inputs;
            for (const ParsedEntry &entry : derived_entries.at(name)) {
                if (entry.condition) collect_lems_identifiers(entry.condition.get(), inputs);
                if (entry.value) collect_lems_identifiers(entry.value.get(), inputs);
            }
            const bool inputs_placed = std::all_of(inputs.begin(), inputs.end(), [&](const String &input) {
                return input == name || derived_entries.count(input) == 0 ||
                       std::find(derived_order.begin(), derived_order.end(), input) != derived_order.end();
            });
            if (inputs_placed) derived_order.push_back(name);
        }
        if (derived_order.size() == placed) return -1;
    }

    UnorderedMap<String, f64> values = starting_values(context, synapse);
    values["weight"] = weight;
    values["t"] = 0.0;
    auto evaluate_derived = [&]() {
        for (const String &name : derived_order) {
            f64 value = 0.0;
            for (const ParsedEntry &entry : derived_entries.at(name)) {
                if (entry.condition && evaluate_lems(entry.condition.get(), values, owner_name) == 0.0) continue;
                if (entry.value) value = evaluate_lems(entry.value.get(), values, owner_name);
                break;
            }
            values[name] = value;
        }
    };

    // At rest nothing may change and nothing may reach the target.
    const UnorderedMap<String, f64> rest = values;
    evaluate_derived();
    for (const ParsedEntry &derivative : derivatives) {
        if (evaluate_lems(derivative.value.get(), values, owner_name) != 0.0) return -1;
    }
    if (values.count("i") == 0 || values.at("i") != 0.0) return -1;

    Vector<f64> largest_deviation(variable_names.size(), 0.0);
    const s64 tick_limit = std::max<s64>(1, std::llround(context.simulation.simulation_duration / step_dt));
    for (s64 tick = 0; tick < tick_limit; tick += 1) {
        values["t"] = (f64)tick * step_dt;
        evaluate_derived();
        Vector<f64> changes;
        for (const ParsedEntry &derivative : derivatives) {
            changes.push_back(step_dt * evaluate_lems(derivative.value.get(), values, owner_name));
        }
        for (usize index = 0; index < derivatives.size(); index += 1) values[derivatives[index].target] += changes[index];
        if (tick == 0) {
            for (const ParsedEntry &assignment : event_assignments) {
                values[assignment.target] = evaluate_lems(assignment.value.get(), values, owner_name);
            }
        }

        bool settled = true;
        for (usize index = 0; index < variable_names.size(); index += 1) {
            const f64 deviation = std::fabs(values.at(variable_names[index]) - rest.at(variable_names[index]));
            largest_deviation[index] = std::max(largest_deviation[index], deviation);
            settled = settled && deviation <= REST_FRACTION * largest_deviation[index];
        }
        // The state after this tick is what the next tick starts from.
        if (settled) return tick + 1;
    }
    return -1;
}

} // namespace

// LEMS attaches every synapse on an incoming edge and every current input to the target cell,
// and the cell reads them through its select DerivedVariables (synapses[*]/i). The engine sums
// them all into network_input, so each one has to expose what those selects read, in the
// dimension they read it. A spike train is the engine's own kick into network_input; it only
// needs a cell that reads its input.
void Codegen::check_cell_inputs() const {
    Vector<std::pair<s64, const NML_ComponentType *>> population_ends;
    s64 first_neuron = 0;
    for (const NML_ComponentInstance *population : network_populations(context)) {
        first_neuron += context.get_population_size(population);
        population_ends.push_back({first_neuron, population_cell(context, *population).component_type});
    }
    auto cell_type_of = [&](s64 neuron_index) -> const NML_ComponentType * {
        for (const auto &[population_end, cell_type] : population_ends) {
            if (neuron_index < population_end) return cell_type;
        }
        return nullptr;
    };

    // A synapse's spikes also reach a cell with an OnEvent, so that cell takes a synapse even
    // when it reads no current.
    auto check_attached = [](const NML_ComponentType &cell_type, const NML_ComponentInstance &attached,
                             const String &role, bool must_expose_input, bool delivers_arrivals) {
        const String described = role + " '" + attached.id + "' (" + attached.component_type->name + ") into a '" +
                                 cell_type.name + "'";
        bool reads_input = delivers_arrivals && has_on_event(cell_type);
        for (const NML_DynamicsExpression &entry : cell_type.dynamics) {
            if (entry.select.empty()) continue;
            reads_input = true;
            if (!must_expose_input) continue;

            const String exposure_name = entry.select.substr(entry.select.rfind('/') + 1);
            const NML_Node *selector = cell_type.find_declaration("var:" + entry.target);
            const String read_dimension = selector ? selector->body.get_attribute_or("dimension", "none") : "none";
            const NML_Node *exposure = attached.component_type->find_declaration("exposure:" + exposure_name);
            if (!exposure) {
                throw runtime_error(described + ": it exposes no '" + exposure_name + "', which the cell reads as '" +
                                    entry.target + "'");
            }
            const String exposed_dimension = exposure->body.get_attribute_or("dimension", "none");
            if (exposed_dimension != read_dimension) {
                throw runtime_error(described + ": it exposes '" + exposure_name + "' as " + exposed_dimension +
                                    ", but the cell reads it as " + read_dimension);
            }
        }
        if (!reads_input) throw runtime_error(described + ": the cell reads no input");
    };

    // Each synapse against each cell type it reaches, once.
    UnorderedMap<String, Set<const NML_ComponentType *>> checked_cell_types_per_synapse;
    for (const Vector<NML_NetworkEdge> &row : context.simulation.network_data.list) {
        for (const NML_NetworkEdge &edge : row) {
            const NML_ComponentType *cell_type = cell_type_of(edge.child);
            if (!cell_type || !checked_cell_types_per_synapse[edge.component_id].insert(cell_type).second) continue;

            const NML_ComponentInstance *synapse = context.find_instance(edge.component_id);
            if (!synapse || !synapse->component_type) continue;
            check_attached(*cell_type, *synapse, "Synapse", true, true);
        }
    }

    for (const SimulationInputConfig &profile : context.simulation.input_profiles) {
        const NML_ComponentInstance *input = context.find_instance(profile.input_component_id);
        if (!input || !input->component_type) continue;

        Set<const NML_ComponentType *> checked_cell_types;
        for (const InputTarget &target : profile.targets) {
            const NML_ComponentType *cell_type = cell_type_of(context.neuron_index_of(target.neuron_index));
            if (!cell_type || !checked_cell_types.insert(cell_type).second) continue;
            check_attached(*cell_type, *input, "Input", profile.continuous_current_injection, false);
        }
    }
}

// The type's dynamics with LEMS names still in them: per population, translate_component_instance
// puts each state variable's cell_state slot and each parameter's value in their place.
//
// Order: the OnEvents, once per spike that arrived this tick (event_arrivals), derived
// variables (by dependency), the refractory flag, every derivative into a temporary, the Euler
// steps, then the OnConditions. All derivatives read the state from before the step, and all
// conditions see the state after it.
KernelListNode *Codegen::translate_component_type(const NML_ComponentType &component_type) {
    auto cached = component_type_templates.find(component_type.name);
    if (cached != component_type_templates.end()) return cached->second;

    const String &owner_name = component_type.name;
    auto unsupported = [&owner_name](const String &reason) {
        return runtime_error("ComponentType '" + owner_name + "': " + reason);
    };

    struct EventHandler {
        String regime_name;
        String test;
        Vector<const NML_DynamicsExpression *> assignments;
        bool emits_spike = false;
        String transition;
    };

    // Sort the flat entries by kind.
    Vector<std::pair<String, bool>> regimes;
    Vector<String> derived_names;
    UnorderedMap<String, Vector<const NML_DynamicsExpression *>> derived_entries;
    Vector<const NML_DynamicsExpression *> derivative_entries;
    Vector<EventHandler> handlers;
    // OnEvents: each runs once per spike that reached the cell this tick, whatever its port.
    Vector<EventHandler> arrival_handlers;
    UnorderedMap<String, Vector<const NML_DynamicsExpression *>> entry_assignments;

    enum class Group { NONE, HANDLER, ARRIVAL, ENTRY };
    Group group = Group::NONE;
    for (const NML_DynamicsExpression &entry : component_type.dynamics) {
        switch (entry.source_tag) {
            case NML_DeclarationType::Regime:
                regimes.push_back({entry.target, entry.expression == "true"});
                group = Group::NONE;
                break;
            case NML_DeclarationType::DerivedVariable:
            case NML_DeclarationType::Case:
                if (derived_entries.count(entry.target) == 0) derived_names.push_back(entry.target);
                derived_entries[entry.target].push_back(&entry);
                group = Group::NONE;
                break;
            case NML_DeclarationType::TimeDerivative:
                derivative_entries.push_back(&entry);
                group = Group::NONE;
                break;
            case NML_DeclarationType::OnStart:
                group = Group::NONE;
                break;
            case NML_DeclarationType::OnCondition:
                handlers.push_back({entry.regime_name, entry.expression, {}, false, ""});
                group = Group::HANDLER;
                break;
            case NML_DeclarationType::OnEntry:
                group = Group::ENTRY;
                break;
            case NML_DeclarationType::OnEvent:
                arrival_handlers.push_back({entry.regime_name, "", {}, false, ""});
                group = Group::ARRIVAL;
                break;
            case NML_DeclarationType::StateAssignment:
                if (group == Group::HANDLER) handlers.back().assignments.push_back(&entry);
                else if (group == Group::ARRIVAL) arrival_handlers.back().assignments.push_back(&entry);
                else if (group == Group::ENTRY) entry_assignments[entry.regime_name].push_back(&entry);
                break;
            case NML_DeclarationType::EventOut:
                if (group == Group::HANDLER) handlers.back().emits_spike = true;
                else if (group == Group::ARRIVAL) arrival_handlers.back().emits_spike = true;
                else throw unsupported("EventOut is only supported in an OnCondition or an OnEvent");
                break;
            case NML_DeclarationType::Transition:
                if (group != Group::HANDLER) throw unsupported("Transition is only supported in an OnCondition");
                handlers.back().transition = entry.target;
                break;
            default:
                break;
        }
    }

    // Regimes: only the integrating / refractory pair, lowered to a test on last_spiked. The
    // refractory regime's timer, its OnEntry and its Transitions all collapse into that test.
    String active_regime;
    String refractory_regime;
    String timer_name;
    std::unique_ptr<KernelNode> refractory_duration;
    String refractory_comparison;

    if (regimes.size() > 2) throw unsupported("only an integrating / refractory regime pair is supported");
    if (regimes.size() == 1) active_regime = regimes[0].first;
    if (regimes.size() == 2) {
        if (regimes[0].second == regimes[1].second) throw unsupported("exactly one regime must be initial");
        active_regime = regimes[0].second ? regimes[0].first : regimes[1].first;
        refractory_regime = regimes[0].second ? regimes[1].first : regimes[0].first;

        Vector<const EventHandler *> exits;
        for (const EventHandler &handler : handlers) {
            if (handler.regime_name == refractory_regime) exits.push_back(&handler);
        }
        if (exits.size() != 1 || exits[0]->transition != active_regime || !exits[0]->assignments.empty() ||
            exits[0]->emits_spike) {
            throw unsupported("the refractory regime must have exactly one OnCondition, a plain Transition back");
        }

        // The exit test is timer .geq. duration (or .gt.).
        std::unique_ptr<LemsParseNode> exit_test(parse_lems_expression(exits[0]->test, owner_name));
        const auto *comparison = dynamic_cast<const BinaryNode<LemsParseBody> *>(exit_test.get());
        const String comparison_operator = exit_test->body.token.lexeme;
        if (!comparison || exit_test->body.syntax_type != LemsNodeSubtype::OPERATOR ||
            (comparison_operator != ">=" && comparison_operator != ">") ||
            comparison->left->body.syntax_type != LemsNodeSubtype::IDENTIFIER) {
            throw unsupported("the refractory exit must test a timer against a duration");
        }
        timer_name = comparison->left->body.token.lexeme;
        // The exit is elapsed > duration or elapsed >= duration; refractory is its negation.
        refractory_comparison = comparison_operator == ">" ? "<=" : "<";

        for (const NML_DynamicsExpression *derivative : derivative_entries) {
            if (derivative->regime_name != refractory_regime) continue;
            if (derivative->target != timer_name || evaluate_lems(derivative->expression, {}, owner_name) != 1.0) {
                throw unsupported("'" + derivative->target + "' has a TimeDerivative in the refractory regime; only a "
                                  "refractory timer may");
            }
        }

        // The test has to measure the time since the spike that entered the regime, which is what
        // last_spiked holds. Two forms do:
        //   timer .geq. duration      OnEntry sets timer = 0, TimeDerivative of timer is 1
        //   t .gt. stamp + duration   OnEntry sets stamp = t
        auto entry_assignment_of = [&](const String &variable_name) -> const NML_DynamicsExpression * {
            for (const NML_DynamicsExpression *assignment : entry_assignments[refractory_regime]) {
                if (assignment->target == variable_name) return assignment;
            }
            return nullptr;
        };
        String stamp_name;
        const LemsParseNode *duration = comparison->right;
        if (timer_name == "t") {
            duration = nullptr;
            const auto *sum = dynamic_cast<const BinaryNode<LemsParseBody> *>(comparison->right);
            if (sum && sum->body.syntax_type == LemsNodeSubtype::OPERATOR && sum->body.token.lexeme == "+") {
                for (const auto &[stamp, candidate] : {std::pair{sum->left, sum->right}, std::pair{sum->right, sum->left}}) {
                    if (stamp->body.syntax_type != LemsNodeSubtype::IDENTIFIER) continue;
                    const NML_DynamicsExpression *stamp_entry = entry_assignment_of(stamp->body.token.lexeme);
                    if (!stamp_entry) continue;
                    std::unique_ptr<LemsParseNode> stamp_value(parse_lems_expression(stamp_entry->expression, owner_name));
                    if (stamp_value->body.syntax_type != LemsNodeSubtype::IDENTIFIER || stamp_value->body.token.lexeme != "t") continue;
                    stamp_name = stamp->body.token.lexeme;
                    duration = candidate;
                    break;
                }
            }
            if (!duration) throw unsupported("a refractory exit on t must be t .gt. stamp + duration, with OnEntry setting stamp = t");

            // The stamp has to hold the spike's time for the whole refractory period.
            for (const NML_DynamicsExpression *derivative : derivative_entries) {
                if (derivative->target == stamp_name) throw unsupported("the refractory stamp '" + stamp_name + "' has a TimeDerivative");
            }
            for (const EventHandler &handler : handlers) {
                for (const NML_DynamicsExpression *assignment : handler.assignments) {
                    if (assignment->target == stamp_name) throw unsupported("the refractory stamp '" + stamp_name + "' is assigned in an OnCondition");
                }
            }
        } else {
            const NML_DynamicsExpression *timer_entry = entry_assignment_of(timer_name);
            bool counts_time = false;
            for (const NML_DynamicsExpression *derivative : derivative_entries) {
                if (derivative->regime_name == refractory_regime && derivative->target == timer_name) counts_time = true;
            }
            if (!timer_entry || !counts_time || evaluate_lems(timer_entry->expression, {}, owner_name) != 0.0) {
                throw unsupported("the refractory timer '" + timer_name +
                                  "' must be set to 0 on entry and count time with a TimeDerivative of 1");
            }
        }
        refractory_duration.reset(translate_expression(duration));

        Set<String> duration_names;
        collect_identifiers(refractory_duration.get(), duration_names);
        if (duration_names.count("t") || duration_names.count(timer_name) || duration_names.count(stamp_name)) {
            throw unsupported("the refractory duration cannot depend on time");
        }
        if (!entry_assignments[active_regime].empty()) throw unsupported("OnEntry is only supported in the refractory regime");

        // Entering refractory happens exactly when an active OnCondition transitions into it, so
        // the OnEntry assignments other than the timer's move into those handlers.
        for (EventHandler &handler : handlers) {
            if (handler.regime_name != active_regime) continue;
            if (handler.transition.empty()) continue;
            if (handler.transition != refractory_regime) throw unsupported("unknown Transition '" + handler.transition + "'");
            for (const NML_DynamicsExpression *assignment : entry_assignments[refractory_regime]) {
                if (assignment->target != timer_name) handler.assignments.push_back(assignment);
            }
        }
    }
    for (const EventHandler &handler : handlers) {
        if (refractory_regime.empty() && !handler.transition.empty()) throw unsupported("Transition without a regime pair");
    }

    const bool has_refractory = !refractory_regime.empty();
    auto is_gated = [&](const String &regime_name) { return has_refractory && regime_name == active_regime; };

    Set<String> lems_names;
    auto translate = [&](const String &expression) {
        const std::unique_ptr<LemsParseNode> tree(parse_lems_expression(expression, owner_name));
        collect_lems_identifiers(tree.get(), lems_names);
        return translate_expression(tree.get());
    };
    const String step_dt = float_literal(context.simulation.step_dt);

    // Phase 1 synapses are current-based, so a summed select is the drained input.
    auto selected_value = [&unsupported](const NML_DynamicsExpression &entry) -> KernelNode * {
        if (entry.reduce != "add") throw unsupported("DerivedVariable '" + entry.target + "' selects without reduce=\"add\"");
        return identifier("network_input");
    };
    KernelListNode *body = new_block();

    // refractory = last_spiked >= 0 && (tick - last_spiked) * dt < duration   (<= for a .gt. exit)
    KernelNode *refractory_declaration = nullptr;
    if (has_refractory) {
        KernelNode *last_spike = new_expression("[]", identifier("last_spiked"), identifier("neuron_index"));
        KernelNode *has_spiked = new_expression(">=", last_spike, literal("0"));
        KernelNode *elapsed = new_expression(
                "*", new_cast(new_type("f32"), new_expression("-", identifier("tick"), clone(last_spike))), literal(step_dt));
        KernelNode *within_duration = new_expression(refractory_comparison, elapsed, refractory_duration.release());
        refractory_declaration = new_declaration(
                new_type("const bool"), "refractory", new_expression("&&", has_spiked, within_duration));
    }

    // The OnEvents come first, once per arrival, so the derived variables and derivatives below
    // see what they changed. One inside a regime runs only in it, so the refractory flag moves
    // up ahead of them.
    if (!arrival_handlers.empty()) {
        KernelListNode *per_arrival = new_block();
        bool tests_regime = false;
        for (const EventHandler &handler : arrival_handlers) {
            KernelListNode *handler_body = new_block();
            for (const NML_DynamicsExpression *assignment : handler.assignments) {
                lems_names.insert(assignment->target);
                handler_body->children.push_back(
                        new_assignment(identifier(assignment->target), "=", translate(assignment->expression)));
            }
            if (handler.emits_spike) handler_body->children.push_back(new_assignment(identifier("spiked"), "=", identifier("true")));

            if (has_refractory && !handler.regime_name.empty()) {
                tests_regime = true;
                KernelNode *in_regime = handler.regime_name == refractory_regime
                                                ? identifier("refractory")
                                                : new_unary_expression("!", identifier("refractory"));
                KernelListNode *gate = new_conditional();
                add_branch(gate, in_regime, handler_body);
                per_arrival->children.push_back(gate);
            } else {
                per_arrival->children.push_back(handler_body);
            }
        }
        if (tests_regime) {
            body->children.push_back(refractory_declaration);
            refractory_declaration = nullptr;
        }

        KernelListNode *arrival_loop = new_node<ListNode>(KernelNodeType::FOR, "");
        arrival_loop->children = {new_declaration(new_type("s32"), "arrival", literal("0")),
                                  new_expression("<", identifier("arrival"), identifier("event_arrivals")),
                                  new_assignment(identifier("arrival"), "+=", literal("1")), per_arrival};
        body->children.push_back(arrival_loop);
    }

    place_derived_variables(body, derived_names, derived_entries, translate, selected_value, unsupported, lems_names);
    if (refractory_declaration) body->children.push_back(refractory_declaration);

    // Every derivative from the pre-step state, then every Euler step.
    KernelListNode *gated = new_block();
    Vector<KernelNode *> updates;
    Set<String> integrated_names;
    for (const NML_DynamicsExpression *derivative : derivative_entries) {
        if (has_refractory && derivative->regime_name == refractory_regime) continue;
        if (!integrated_names.insert(derivative->target).second) {
            throw unsupported("'" + derivative->target + "' has more than one TimeDerivative");
        }
        lems_names.insert(derivative->target);

        body->children.push_back(new_declaration(
                new_type("const f32"), DERIVATIVE_PREFIX + derivative->target, translate(derivative->expression)));
        KernelNode *update = new_assignment(
                identifier(derivative->target), "+=",
                new_expression("*", literal(step_dt), identifier(DERIVATIVE_PREFIX + derivative->target)));
        (is_gated(derivative->regime_name) ? gated->children : updates).push_back(update);
    }
    for (KernelNode *update : updates) body->children.push_back(update);

    // OnConditions, after every step.
    Vector<KernelNode *> conditions;
    for (const EventHandler &handler : handlers) {
        if (has_refractory && handler.regime_name == refractory_regime) continue;

        KernelListNode *handler_body = new_block();
        for (const NML_DynamicsExpression *assignment : handler.assignments) {
            lems_names.insert(assignment->target);
            handler_body->children.push_back(
                    new_assignment(identifier(assignment->target), "=", translate(assignment->expression)));
        }
        if (handler.emits_spike) handler_body->children.push_back(new_assignment(identifier("spiked"), "=", identifier("true")));

        KernelListNode *condition = new_conditional();
        add_branch(condition, translate(handler.test), handler_body);
        (is_gated(handler.regime_name) ? gated->children : conditions).push_back(condition);
    }

    if (gated->children.empty()) {
        delete gated;
    } else {
        KernelListNode *integrating = new_conditional();
        add_branch(integrating, new_unary_expression("!", identifier("refractory")), gated);
        body->children.push_back(integrating);
    }
    for (KernelNode *condition : conditions) body->children.push_back(condition);

    for (const String &name : lems_names) {
        if (ENGINE_NAMES.count(name) || starts_with(name, DERIVED_PREFIX) || starts_with(name, DERIVATIVE_PREFIX) ||
            starts_with(name, RANDOM_SLOT_PREFIX)) {
            delete body;
            throw unsupported("the name '" + name + "' is reserved by the generated kernel");
        }
    }

    // Derived variables are read through their declared locals, and t is the tick's time.
    KernelNode *body_node = body;
    for (const String &name : derived_names) {
        const std::unique_ptr<KernelNode> local(identifier(DERIVED_PREFIX + name));
        replace_identifier(body_node, name, local.get());
    }
    const std::unique_ptr<KernelNode> tick_time(new_expression(
            "*", new_cast(new_type("f32"), identifier("tick")), literal(step_dt)));
    replace_identifier(body_node, "t", tick_time.get());

    component_type_templates[component_type.name] = body;
    return body;
}

// The population's dynamics: the type's template with every state variable replaced by its
// slot in cell_state and every parameter by its value.
KernelListNode *Codegen::translate_component_instance(const NML_ComponentInstance &population, s64 first_neuron) {
    const NML_ComponentInstance &cell = population_cell(context, population);
    const NML_ComponentType &component_type = *cell.component_type;
    KernelNode *body = clone(translate_component_type(component_type));

    try {
        // Each random() call gets a slot per neuron of the population, so every neuron draws
        // its own value: neuron n reads random_values[n + slot_start - first_neuron].
        const s64 population_size = context.get_population_size(&population);
        Set<String> template_names;
        collect_identifiers(body, template_names);
        for (const String &name : template_names) {
            if (!starts_with(name, RANDOM_SLOT_PREFIX)) continue;
            const s64 constant_part = random_values_count - first_neuron;
            const std::unique_ptr<KernelNode> slot(new_expression(constant_part < 0 ? "-" : "+", identifier("neuron_index"),
                                                                  literal(std::to_string(std::llabs(constant_part)))));
            replace_identifier(body, name, slot.get());
            random_values_count += population_size;
        }

        // Neuron n's variable at offset is cell_state[n * count + base - first_neuron * count + offset].
        const Vector<String> &variable_names = component_type.state_variable_names;
        const s64 variable_count = static_cast<s64>(variable_names.size());
        const s64 population_base = context.simulation.population_base_indices.at(population.id);
        for (usize offset = 0; offset < variable_names.size(); offset += 1) {
            const s64 constant_part = population_base - first_neuron * variable_count + static_cast<s64>(offset);
            const std::unique_ptr<KernelNode> slot(new_expression(
                    "[]", identifier("cell_state"),
                    new_expression(constant_part < 0 ? "-" : "+",
                                   new_expression("*", identifier("neuron_index"), literal(std::to_string(variable_count))),
                                   literal(std::to_string(std::llabs(constant_part))))));
            replace_identifier(body, variable_names[offset], slot.get());
        }

        for (const auto &[name, value] : component_parameter_values(context, cell)) {
            const std::unique_ptr<KernelNode> value_literal(literal(float_literal(value)));
            replace_identifier(body, name, value_literal.get());
        }

        Set<String> remaining_names;
        collect_identifiers(body, remaining_names);
        for (const String &name : remaining_names) {
            if (ENGINE_NAMES.count(name) || starts_with(name, DERIVED_PREFIX) || starts_with(name, DERIVATIVE_PREFIX)) continue;
            throw runtime_error("ComponentType '" + component_type.name + "' (cell '" + cell.id + "'): '" + name +
                                "' is not a state variable, a set parameter, a constant or a derived variable");
        }
    } catch (...) {
        delete body;
        throw;
    }
    return static_cast<KernelListNode *>(body);
}

// The synapse's dynamics for the current edge, with LEMS names still in them. Each state
// variable lives in its own weight-matrix plane, so a read is a load_edge and a write is an
// accumulate_edge of the change.
//
// It runs on every edge of a neuron whose synapses are in their active window (see
// synapse_active_ticks). Order: derived variables, every derivative into a temporary, the Euler
// steps, the OnEvent on an arrival, then the current into the target. On the window's last tick
// the synapse has settled, and instead of the Euler steps it is put exactly back at rest, where
// it stays while the walk is skipped.
KernelListNode *Codegen::translate_synapse_type(const NML_ComponentType &component_type) {
    auto cached = synapse_type_templates.find(component_type.name);
    if (cached != synapse_type_templates.end()) return cached->second;

    const String &owner_name = component_type.name;
    auto unsupported = [&owner_name](const String &reason) {
        return runtime_error("Synapse ComponentType '" + owner_name + "': " + reason);
    };

    const Vector<String> &variable_names = component_type.state_variable_names;
    auto plane_of = [&](const String &variable_name) -> s64 {
        auto variable = std::find(variable_names.begin(), variable_names.end(), variable_name);
        if (variable == variable_names.end()) throw unsupported("'" + variable_name + "' is not a state variable");
        return WeightMatrix::FIRST_STATE_VARIABLE_PLANE + static_cast<s64>(variable - variable_names.begin());
    };

    // Sort the flat entries by kind.
    Vector<String> derived_names;
    UnorderedMap<String, Vector<const NML_DynamicsExpression *>> derived_entries;
    Vector<const NML_DynamicsExpression *> derivative_entries;
    Vector<const NML_DynamicsExpression *> event_assignments;
    bool receives_spikes = false;
    bool in_event = false;
    for (const NML_DynamicsExpression &entry : component_type.dynamics) {
        switch (entry.source_tag) {
            case NML_DeclarationType::DerivedVariable:
            case NML_DeclarationType::Case:
                if (derived_entries.count(entry.target) == 0) derived_names.push_back(entry.target);
                derived_entries[entry.target].push_back(&entry);
                in_event = false;
                break;
            case NML_DeclarationType::TimeDerivative:
                derivative_entries.push_back(&entry);
                in_event = false;
                break;
            case NML_DeclarationType::OnStart:
                in_event = false;
                break;
            case NML_DeclarationType::OnEvent:
                if (entry.target != "in") throw unsupported("OnEvent on port '" + entry.target + "': a synapse only receives spikes on \"in\"");
                receives_spikes = true;
                in_event = true;
                break;
            case NML_DeclarationType::StateAssignment:
                if (!in_event) throw unsupported("a StateAssignment outside OnStart and OnEvent");
                event_assignments.push_back(&entry);
                break;
            case NML_DeclarationType::Regime:
            case NML_DeclarationType::OnCondition:
            case NML_DeclarationType::OnEntry:
            case NML_DeclarationType::EventOut:
            case NML_DeclarationType::Transition:
                throw unsupported("only OnEvent, TimeDerivative and DerivedVariable are supported in a synapse");
            default:
                break;
        }
    }
    if (!receives_spikes) throw unsupported("no OnEvent port=\"in\", so no spike ever drives it");
    const bool exposes_current = derived_entries.count("i") != 0 ||
                                 std::find(variable_names.begin(), variable_names.end(), "i") != variable_names.end();
    if (!exposes_current) throw unsupported("no current 'i' to deliver to the target");

    Set<String> lems_names;
    auto translate = [&](const String &expression) {
        const std::unique_ptr<LemsParseNode> tree(parse_lems_expression(expression, owner_name));
        if (calls_function(tree.get(), "random")) throw unsupported("random() is not supported in a synapse");
        collect_lems_identifiers(tree.get(), lems_names);
        return translate_expression(tree.get());
    };
    auto selected_value = [&unsupported](const NML_DynamicsExpression &entry) -> KernelNode * {
        throw unsupported("DerivedVariable '" + entry.target + "' selects, which a synapse cannot");
    };
    const String step_dt = float_literal(context.simulation.step_dt);

    KernelListNode *body = new_block();
    place_derived_variables(body, derived_names, derived_entries, translate, selected_value, unsupported, lems_names);

    // Every derivative from the pre-step state, then every Euler step.
    Set<String> integrated_names;
    Vector<KernelNode *> updates;
    for (const NML_DynamicsExpression *derivative : derivative_entries) {
        if (!integrated_names.insert(derivative->target).second) {
            throw unsupported("'" + derivative->target + "' has more than one TimeDerivative");
        }
        lems_names.insert(derivative->target);
        body->children.push_back(new_declaration(
                new_type("const f32"), DERIVATIVE_PREFIX + derivative->target, translate(derivative->expression)));
        updates.push_back(accumulate_edge(
                plane_of(derivative->target),
                new_expression("*", literal(step_dt), identifier(DERIVATIVE_PREFIX + derivative->target))));
    }
    KernelListNode *steps = new_block();
    for (KernelNode *update : updates) steps->children.push_back(update);

    KernelListNode *return_to_rest = new_block();
    for (const String &variable_name : variable_names) {
        lems_names.insert(variable_name);
        return_to_rest->children.push_back(accumulate_edge(
                plane_of(variable_name),
                new_expression("-", identifier(REST_PREFIX + variable_name), identifier(variable_name))));
    }
    KernelNode *last_active_tick = new_expression(
            "==",
            new_expression("-", identifier("tick"), new_expression("[]", identifier("last_spiked"), identifier("neuron_index"))),
            identifier("synapse_active_ticks"));
    KernelListNode *advance = new_conditional();
    add_branch(advance, last_active_tick, return_to_rest);
    add_branch(advance, nullptr, steps);
    body->children.push_back(advance);

    // The OnEvent, as the change each assignment makes.
    KernelListNode *on_arrival = new_block();
    for (const NML_DynamicsExpression *assignment : event_assignments) {
        lems_names.insert(assignment->target);
        KernelNode *increment = new_expression("-", translate(assignment->expression), identifier(assignment->target));
        on_arrival->children.push_back(accumulate_edge(plane_of(assignment->target), increment));
    }
    KernelListNode *arrival = new_conditional();
    add_branch(arrival, identifier("arrived"), on_arrival);
    body->children.push_back(arrival);

    // The current, into the target's next input row.
    KernelNode *input_slot = new_expression(
            "[]", identifier("network_inputs"),
            new_expression("+", new_expression("*", identifier("next_row"), identifier("neuron_count")), identifier("target")));
    body->children.push_back(function_call("atomic_add", {input_slot, identifier("i")}));

    for (const String &name : lems_names) {
        if (is_synapse_engine_name(name)) {
            delete body;
            throw unsupported("the name '" + name + "' is reserved by the generated kernel");
        }
    }

    // Derived variables are read through their declared locals, and t is the tick's time.
    KernelNode *body_node = body;
    for (const String &name : derived_names) {
        const std::unique_ptr<KernelNode> local(identifier(DERIVED_PREFIX + name));
        replace_identifier(body_node, name, local.get());
    }
    const std::unique_ptr<KernelNode> tick_time(new_expression(
            "*", new_cast(new_type("f32"), identifier("tick")), literal(step_dt)));
    replace_identifier(body_node, "t", tick_time.get());

    synapse_type_templates[component_type.name] = body;
    return body;
}

KernelListNode *Codegen::translate_synapse_instance(const NML_ComponentInstance &synapse) {
    if (!synapse.component_type) throw runtime_error("Synapse '" + synapse.id + "' has no ComponentType");
    const NML_ComponentType &component_type = *synapse.component_type;
    KernelNode *body = clone(translate_synapse_type(component_type));

    try {
        for (const auto &[name, value] : component_parameter_values(context, synapse)) {
            const std::unique_ptr<KernelNode> value_literal(literal(float_literal(value)));
            replace_identifier(body, name, value_literal.get());
        }

        // weight is the connection's, stored in the weight plane.
        const std::unique_ptr<KernelNode> edge_weight(load_edge(WeightMatrix::WEIGHT_PLANE));
        replace_identifier(body, "weight", edge_weight.get());

        const Vector<String> &variable_names = component_type.state_variable_names;
        const UnorderedMap<String, f64> rest_values = starting_values(context, synapse);
        for (usize offset = 0; offset < variable_names.size(); offset += 1) {
            const std::unique_ptr<KernelNode> stored(load_edge(WeightMatrix::FIRST_STATE_VARIABLE_PLANE + static_cast<s64>(offset)));
            replace_identifier(body, variable_names[offset], stored.get());
            const std::unique_ptr<KernelNode> rest_value(literal(float_literal(rest_values.at(variable_names[offset]))));
            replace_identifier(body, REST_PREFIX + variable_names[offset], rest_value.get());
        }

        Set<String> remaining_names;
        collect_identifiers(body, remaining_names);
        for (const String &name : remaining_names) {
            if (is_synapse_engine_name(name)) continue;
            throw runtime_error("Synapse ComponentType '" + component_type.name + "' ('" + synapse.id + "'): '" + name +
                                "' is not a state variable, a set parameter, a constant, a derived variable or weight");
        }
    } catch (...) {
        delete body;
        throw;
    }
    return static_cast<KernelListNode *>(body);
}

s64 Codegen::edge_plane_count() const {
    s64 state_variable_count = 0;
    for (const NML_ComponentInstance &synapse : context.simulation.synapse_instances) {
        if (!synapse.component_type) throw runtime_error("Synapse '" + synapse.id + "' has no ComponentType");
        state_variable_count = std::max<s64>(state_variable_count, synapse.component_type->state_variable_names.size());
    }
    return WeightMatrix::FIRST_STATE_VARIABLE_PLANE + state_variable_count;
}

s64 Codegen::updated_edge_plane_count() const {
    // A synapse's dynamics write every one of its StateVariables: the Euler steps, the OnEvent
    // and the return to rest.
    return edge_plane_count() - WeightMatrix::FIRST_STATE_VARIABLE_PLANE;
}

s64 Codegen::synapse_active_ticks() const {
    UnorderedMap<String, f64> largest_weight;
    for (const Vector<NML_NetworkEdge> &row : context.simulation.network_data.list) {
        for (const NML_NetworkEdge &edge : row) {
            f64 &largest = largest_weight[edge.component_id];
            largest = std::max(largest, std::fabs((f64)edge.weight));
        }
    }

    // At least a tick, so no arrival ever lands on the tick that puts a synapse back at rest.
    s64 settle_ticks = 1;
    for (const NML_ComponentInstance &synapse : context.simulation.synapse_instances) {
        auto weight = largest_weight.find(synapse.id);
        if (weight == largest_weight.end()) continue;

        s64 ticks = -1;
        try {
            ticks = synapse_settle_ticks(context, synapse, weight->second);
        } catch (const runtime_error &error) {
            log::logger().warn("Codegen: synapse '{}' cannot be simulated on the host: {}", synapse.id, error.what());
        }
        if (ticks < 0) {
            log::logger().warn("Codegen: synapse '{}' does not settle back to its OnStart values, so every "
                               "synapse runs every tick", synapse.id);
            return -1;
        }
        settle_ticks = std::max(settle_ticks, ticks);
    }

    const s64 active_ticks = std::max<s64>(1, context.simulation.maximum_edge_delay) + settle_ticks;
    log::logger().info("Codegen: a neuron's synapses run for {} ticks after it spikes ({} of delay, {} to settle)",
                       active_ticks, active_ticks - settle_ticks, settle_ticks);
    return active_ticks;
}

KernelNode *Codegen::translate_expression(const LemsParseNode *node) {
    const String &lexeme = node->body.token.lexeme;
    switch (node->body.syntax_type) {
        case LemsNodeSubtype::FLOAT:
        case LemsNodeSubtype::INT:
            // Always a float, so 1/2 is never integer division.
            return literal(float_literal(std::stod(lexeme)));

        case LemsNodeSubtype::IDENTIFIER:
            return identifier(lexeme);

        case LemsNodeSubtype::OPERATOR: {
            if (const auto *unary = dynamic_cast<const UnaryNode<LemsParseBody> *>(node)) {
                return new_unary_expression(lexeme, translate_expression(unary->child));
            }
            const auto *binary = static_cast<const BinaryNode<LemsParseBody> *>(node);
            KernelNode *left = translate_expression(binary->left);
            return new_expression(lexeme, left, translate_expression(binary->right));
        }

        case LemsNodeSubtype::FUNCTION_CALL: {
            // random(x) is x times this tick's uniform draw in the call's own random_values slot.
            if (lexeme == "random") {
                const Vector<LemsParseNode *> &arguments = children_of(static_cast<const BinaryNode<LemsParseBody> *>(node)->right);
                if (arguments.size() != 1) throw runtime_error("LEMS random takes one argument");
                KernelNode *slot = identifier(RANDOM_SLOT_PREFIX + std::to_string(random_call_count));
                random_call_count += 1;
                return new_expression("*", translate_expression(arguments[0]),
                                      new_expression("[]", identifier("random_values"), slot));
            }

            auto function = LEMS_FUNCTIONS.find(lexeme);
            if (function == LEMS_FUNCTIONS.end()) throw runtime_error("LEMS function '" + lexeme + "' has no kernel equivalent");

            KernelListNode *call = new_node<ListNode>(KernelNodeType::FUNCTION_CALL, function->second);
            for (const LemsParseNode *argument : children_of(static_cast<const BinaryNode<LemsParseBody> *>(node)->right)) {
                call->children.push_back(translate_expression(argument));
            }
            return call;
        }

        default:
            throw runtime_error("Malformed LEMS expression '" + lexeme + "'");
    }
}

KernelNode *Codegen::translate_lems(const String &expression, const String &owner_name) {
    std::unique_ptr<LemsParseNode> tree(parse_lems_expression(expression, owner_name));
    return translate_expression(tree.get());
}

void Codegen::bake_kernel_constant(const String &name, s64 value) {
    KernelListNode *master_step = root ? find_function(root, "master_step") : nullptr;
    if (!master_step) throw runtime_error("bake_kernel_constant: there is no master_step until translate_kernel_code runs");
    const std::unique_ptr<KernelNode> value_literal(literal(std::to_string(value)));
    bake_parameter(master_step, name, value_literal.get());
}

Vector<String> Codegen::kernel_parameter_names() const {
    Vector<String> names;
    KernelListNode *master_step = root ? find_function(root, "master_step") : nullptr;
    if (!master_step) return names;

    for (KernelNode *parameter_node : static_cast<KernelListNode *>(master_step->children[2])->children) {
        KernelTrinaryNode *parameter = static_cast<KernelTrinaryNode *>(parameter_node);
        if (parameter->right) continue;
        names.push_back(parameter->middle->body.token);
    }
    return names;
}

// code generation

const UnorderedMap<String, String> &kernel_type_spellings(KernelBackend backend) {
    static const UnorderedMap<String, String> metal_spellings = {
        {"void", "void"}, {"bool", "bool"}, {"u8", "uchar"}, {"u16", "ushort"}, {"s32", "int"},
        {"u32", "uint"}, {"s64", "long"}, {"f32", "float"}};
    static const UnorderedMap<String, String> cuda_spellings = {
        {"void", "void"}, {"bool", "bool"}, {"u8", "unsigned char"}, {"u16", "unsigned short"}, {"s32", "int"},
        {"u32", "unsigned int"}, {"s64", "long long"}, {"f32", "float"}, {"f64", "double"}};
    return backend == KernelBackend::METAL ? metal_spellings : cuda_spellings;
}

String Codegen::compile() {
    return compile(root);
}

String Codegen::compile(KernelNode *node) {
    switch (node->body.syntax_type) {
        case KernelNodeType::LITERAL:              return node->body.token;
        case KernelNodeType::IDENTIFIER:           return node->body.token;
        case KernelNodeType::EXPRESSION:           return expression_to_string(node);
        case KernelNodeType::BLOCK:                return block_to_string(node);
        case KernelNodeType::DEVICE_FUNCTION_IMPL: return device_function_impl_to_string(node);
        case KernelNodeType::FUNCTION_CALL:        return function_call_to_string(node);
        case KernelNodeType::KERNEL_FUNCTION_IMPL: return kernel_function_impl_to_string(node);
        case KernelNodeType::DECLARATION:          return declaration_to_string(node);
        case KernelNodeType::ASSIGNMENT:           return assignment_to_string(node);
        case KernelNodeType::CONDITIONAL:          return conditional_to_string(node);
        case KernelNodeType::CAST:                 return cast_to_string(node);
        case KernelNodeType::FOR:                  return for_to_string(node);
        case KernelNodeType::WHILE:                return while_to_string(node);
        case KernelNodeType::JUMP:                 return jump_to_string(node);
        case KernelNodeType::ATTRIBUTE:            return attribute_to_string(node);
        case KernelNodeType::PROGRAM:              return program_to_string(node);
        case KernelNodeType::TYPE:                 return type_to_string(node->body.token);
        case KernelNodeType::POINTER:              return pointer_to_string(node);
        case KernelNodeType::REFERENCE:            return compile(static_cast<KernelUnaryNode *>(node)->child) + " &";
        case KernelNodeType::INCLUDE:              return "#include " + node->body.token;
        case KernelNodeType::DEFINE:               return define_to_string(node);
        case KernelNodeType::USING_NAMESPACE:      return "using namespace " + node->body.token + ";";
        case KernelNodeType::CONDITIONAL_BRANCH:
        case KernelNodeType::PARAMETER:
        case KernelNodeType::PARAMETER_LIST:       break;
    }
    throw runtime_error("Kernel node cannot be emitted on its own");
}

String Codegen::compile_or_empty(KernelNode *node) {
    return node ? compile_unparenthesized(node) : String();
}

String Codegen::compile_unparenthesized(KernelNode *node) {
    if (node->body.syntax_type == KernelNodeType::EXPRESSION) return expression_to_string(node, false);
    return compile(node);
}

// Words that are neutral type names are mapped; qualifiers and platform-only types pass through.
String Codegen::type_to_string(const String &type) const {
    const UnorderedMap<String, String> &spellings = kernel_type_spellings(backend);
    std::istringstream words(type);
    String word;
    String source;
    while (words >> word) {
        auto spelling = spellings.find(word);
        if (!source.empty()) source += " ";
        source += spelling == spellings.end() ? word : spelling->second;
    }
    return source;
}

String Codegen::pointer_to_string(KernelNode *node) {
    KernelUnaryNode *pointer = static_cast<KernelUnaryNode *>(node);
    String source = compile(pointer->child) + " *";
    if (!pointer->body.token.empty()) source += " " + pointer->body.token;
    return source;
}

String Codegen::define_to_string(KernelNode *node) {
    KernelBinaryNode *define = static_cast<KernelBinaryNode *>(node);
    String source = "#define " + compile(define->left);
    if (define->right) source += " " + compile(define->right);
    return source;
}

// Directives and using lines one per line; functions separated by a blank line.
String Codegen::program_to_string(KernelNode *node) {
    String source;
    for (KernelNode *item : static_cast<KernelListNode *>(node)->children) {
        const KernelNodeType type = item->body.syntax_type;
        const bool is_function = type == KernelNodeType::DEVICE_FUNCTION_IMPL || type == KernelNodeType::KERNEL_FUNCTION_IMPL;
        source += is_function ? "\n" + compile(item) + "\n" : compile(item) + "\n";
    }
    return source;
}

String Codegen::attribute_to_string(KernelNode *node) {
    if (backend != KernelBackend::METAL) throw runtime_error("Attribute '" + node->body.token + "' has no CUDA equivalent");
    return "[[ " + node->body.token + " ]]";
}

// The tree is the precedence, so operators are fully parenthesized unless the place the
// expression sits already delimits it.
String Codegen::expression_to_string(KernelNode *node, bool parenthesize) {
    const String &expression_operator = node->body.token;
    const String open = parenthesize ? "(" : "";
    const String close = parenthesize ? ")" : "";

    if (KernelBinaryNode *binary = dynamic_cast<KernelBinaryNode *>(node)) {
        if (expression_operator == "[]") return compile(binary->left) + "[" + compile_unparenthesized(binary->right) + "]";
        if (expression_operator == "." || expression_operator == "->")
            return compile(binary->left) + expression_operator + compile(binary->right);
        return open + compile(binary->left) + " " + expression_operator + " " + compile(binary->right) + close;
    }
    if (KernelUnaryNode *unary = dynamic_cast<KernelUnaryNode *>(node))
        return open + expression_operator + compile(unary->child) + close;
    if (KernelTrinaryNode *ternary = dynamic_cast<KernelTrinaryNode *>(node))
        return open + compile(ternary->left) + " ? " + compile(ternary->middle) + " : " + compile(ternary->right) + close;
    throw runtime_error("EXPRESSION node has no operands");
}

String Codegen::cast_to_string(KernelNode *node) {
    KernelBinaryNode *cast = static_cast<KernelBinaryNode *>(node);
    return "((" + compile(cast->left) + ")" + compile(cast->right) + ")";
}

// Neutral names that differ per backend are mapped; every other name passes through.
String Codegen::function_call_to_string(KernelNode *node) {
    const Vector<KernelNode *> &arguments = static_cast<KernelListNode *>(node)->children;
    const String &name = node->body.token;

    // atomic_add(target, value): target is the slot itself, not its address.
    if (name == "atomic_add") {
        if (arguments.size() != 2) throw runtime_error("atomic_add takes a target and a value");
        if (backend == KernelBackend::METAL) {
            return "atomic_fetch_add_explicit((device atomic_float *)&(" + compile_unparenthesized(arguments[0]) + "), " +
                   compile_unparenthesized(arguments[1]) + ", memory_order_relaxed)";
        }
        return "atomicAdd(&(" + compile_unparenthesized(arguments[0]) + "), " + compile_unparenthesized(arguments[1]) + ")";
    }

    // atomic_increment(target): adds 1 to a u32 slot.
    if (name == "atomic_increment") {
        if (arguments.size() != 1) throw runtime_error("atomic_increment takes a target");
        if (backend == KernelBackend::METAL) {
            return "atomic_fetch_add_explicit((device atomic_uint *)&(" + compile_unparenthesized(arguments[0]) +
                   "), 1u, memory_order_relaxed)";
        }
        return "atomicAdd(&(" + compile_unparenthesized(arguments[0]) + "), 1u)";
    }

    static const UnorderedMap<String, String> function_spellings = {{"heaviside", "spikecorec_heaviside"}};
    auto spelling = function_spellings.find(name);

    String source = (spelling == function_spellings.end() ? name : spelling->second) + "(";
    for (usize index = 0; index < arguments.size(); index += 1) {
        if (index > 0) source += ", ";
        source += compile_unparenthesized(arguments[index]);
    }
    return source + ")";
}

// The block adds the semicolons, so declarations and assignments can be reused in loop headers.
String Codegen::block_to_string(KernelNode *node) {
    static const Set<KernelNodeType> compound_statements = {
        KernelNodeType::BLOCK, KernelNodeType::CONDITIONAL, KernelNodeType::FOR, KernelNodeType::WHILE,
        KernelNodeType::DEVICE_FUNCTION_IMPL, KernelNodeType::KERNEL_FUNCTION_IMPL};

    String source = "{\n";
    indentation_depth += 1;
    for (KernelNode *statement : static_cast<KernelListNode *>(node)->children) {
        source += String(4 * indentation_depth, ' ') + compile_unparenthesized(statement);
        if (compound_statements.count(statement->body.syntax_type) == 0) source += ";";
        source += "\n";
    }
    indentation_depth -= 1;
    return source + String(4 * indentation_depth, ' ') + "}";
}

String Codegen::declaration_to_string(KernelNode *node) {
    KernelTrinaryNode *declaration = static_cast<KernelTrinaryNode *>(node);
    String source = compile(declaration->left) + " " + compile(declaration->middle);
    if (declaration->right) source += " = " + compile_unparenthesized(declaration->right);
    return source;
}

String Codegen::assignment_to_string(KernelNode *node) {
    KernelBinaryNode *assignment = static_cast<KernelBinaryNode *>(node);
    return compile(assignment->left) + " " + assignment->body.token + " " + compile_unparenthesized(assignment->right);
}

String Codegen::conditional_to_string(KernelNode *node) {
    String source;
    for (KernelNode *branch_node : static_cast<KernelListNode *>(node)->children) {
        KernelBinaryNode *branch = static_cast<KernelBinaryNode *>(branch_node);
        if (!source.empty()) source += " else ";
        if (branch->left) source += "if (" + compile_unparenthesized(branch->left) + ") ";
        source += compile(branch->right);
    }
    return source;
}

String Codegen::for_to_string(KernelNode *node) {
    const Vector<KernelNode *> &parts = static_cast<KernelListNode *>(node)->children;
    return "for (" + compile_or_empty(parts[0]) + "; " + compile_or_empty(parts[1]) + "; " +
           compile_or_empty(parts[2]) + ") " + compile(parts[3]);
}

String Codegen::while_to_string(KernelNode *node) {
    KernelBinaryNode *loop = static_cast<KernelBinaryNode *>(node);
    return "while (" + compile_unparenthesized(loop->left) + ") " + compile(loop->right);
}

String Codegen::jump_to_string(KernelNode *node) {
    KernelUnaryNode *jump = static_cast<KernelUnaryNode *>(node);
    if (!jump->child) return jump->body.token;
    return jump->body.token + " " + compile_unparenthesized(jump->child);
}

// Metal buffer indices come from position, so inserting a parameter never renumbers anything by hand.
String Codegen::parameters_to_string(KernelNode *node, bool number_buffers) {
    String source;
    s64 buffer_index = 0;
    for (KernelNode *parameter_node : static_cast<KernelListNode *>(node)->children) {
        KernelTrinaryNode *parameter = static_cast<KernelTrinaryNode *>(parameter_node);
        if (!source.empty()) source += ",";
        source += "\n    " + compile(parameter->left) + " " + compile(parameter->middle);
        if (parameter->right) {
            source += " " + compile(parameter->right);
        } else if (number_buffers && backend == KernelBackend::METAL) {
            source += " [[ buffer(" + std::to_string(buffer_index) + ") ]]";
            buffer_index += 1;
        }
    }
    return source + "\n";
}

String Codegen::device_function_impl_to_string(KernelNode *node) {
    const Vector<KernelNode *> &parts = static_cast<KernelListNode *>(node)->children;
    const String qualifier = backend == KernelBackend::METAL ? "inline " : "__device__ inline ";
    return qualifier + compile(parts[0]) + " " + compile(parts[1]) +
           "(" + parameters_to_string(parts[2], false) + ") " + compile(parts[3]);
}

String Codegen::kernel_function_impl_to_string(KernelNode *node) {
    const Vector<KernelNode *> &parts = static_cast<KernelListNode *>(node)->children;
    const String qualifier = backend == KernelBackend::METAL ? "kernel " : "extern \"C\" __global__ ";
    return qualifier + compile(parts[0]) + " " + compile(parts[1]) +
           "(" + parameters_to_string(parts[2], true) + ") " + compile(parts[3]);
}

}
