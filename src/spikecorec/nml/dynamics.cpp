#include "dynamics.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <tuple>
#include <sstream>

#include "backend.h"
#include "log.h"
#include "units.h"
#include "types.h"
#include "weight_matrix.h"
#include "node.h"
#include "parser.h"

#ifndef SPIKECOREC_METAL_DEVICE_DIR
#define SPIKECOREC_METAL_DEVICE_DIR ""
#endif

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

namespace {

// Names the generated cell dynamics declare or read in master_step. A LEMS name equal to one
// would be captured, so it is refused.
const Set<String> ENGINE_NAMES = {
    "cell_state", "neuron_index", "network_input", "spiked", "refractory", "last_spiked", "tick", "true", "random_values",
    "walk_pass",
    "event_arrivals", "arrival", "simulation_time", "network_inputs", "neuron_count", "current_row"};

// Names the generated synapse dynamics use in the propagate walk, on top of ENGINE_NAMES.
const Set<String> SYNAPSE_ENGINE_NAMES = {
    "target", "current_edge", "arrived", "next_row", "network_inputs", "neuron_count", "synapse_prototype",
    "basis_u", "basis_v", "edge_coefficients", "rank_float4_stride", "sparse_delta_row_start",
    "sparse_delta_edge_ordinal", "sparse_delta_value", "pending_delta_edge_ordinal",
    "pending_delta_value", "pending_delta_matrix_index", "pending_delta_control", "pending_delta_capacity",
    "updated_plane_count", "walk_pass",
    "run_first_edge_ordinal", "run_synapse_prototype", "run_count", "synapse_active_ticks",
    "delay_ticks", "spike_history", "spike_history_length", "arrival_delay_ticks", "arrives_next_tick", "delivered_value"};

// Names the generated input code declares or reads in master_step. A LEMS name in an input
// equal to one of these, or starting with INPUT_PART_PREFIX, would be captured, so it is refused.
const Set<String> INPUT_ENGINE_NAMES = {
    "cell_state", "neuron_index", "network_input", "tick", "true", "random_values", "input_row_start",
    "input_entry_prototype", "input_values", "input_spike_counts", "input_entry", "input_base", "input_weight",
    "input_prototype", "input_kicks", "train_spikes", "arrival", "simulation_time"};
// input_part<k>_<name> holds part k's <name> for its parent (for part 0, the cell) to read.
const String INPUT_PART_PREFIX = "input_part";

const String DERIVED_PREFIX = "derived_";
const String DERIVATIVE_PREFIX = "derivative_";
// The refractory regime's derivatives, which may share variables with the integrating regime's.
const String REFRACTORY_DERIVATIVE_PREFIX = "refractory_derivative_";
// rest_<name> stands for a synapse state variable's OnStart value until the instance fills it in.
const String REST_PREFIX = "rest_";
// change_<name> sums a synapse state variable's changes in one tick, so S gets one update per plane.
const String CHANGE_PREFIX = "change_";
// random_slot_<n> stands for random() call n's slot in random_values until the population or
// input that runs it is placed.
const String RANDOM_SLOT_PREFIX = "random_slot_";

// A synapse has settled once every state variable is within this fraction of its largest
// deviation from rest since the spike.
constexpr f64 REST_FRACTION = 1e-6;

// LEMS function name -> neutral kernel function name. LEMS log is the natural log, like ln.
const Unordered_Map<String, String> LEMS_FUNCTIONS = {
    {"exp", "exp"},   {"ln", "log"},    {"log", "log"},   {"sqrt", "sqrt"},
    {"abs", "fabs"},  {"ceil", "ceil"}, {"floor", "floor"}, {"pow", "pow"},
    {"sin", "sin"},   {"cos", "cos"},   {"tan", "tan"},
    {"sinh", "sinh"}, {"cosh", "cosh"}, {"tanh", "tanh"},
    {"H", "heaviside"},
};

String float_literal(f64 value) {
    if (!std::isfinite(value)) throw Runtime_Error("Cannot bake the non-finite value " + std::to_string(value) + " into a kernel");
    Output_String_Stream text;
    text << std::setprecision(9) << value;
    String literal = text.str();
    if (literal.find_first_of(".e") == String::npos) literal += ".0";
    return literal + "f";
}

Kernel_Node *identifier(const String &name) {
    return new_node(Kernel_Node_Type::IDENTIFIER, name);
}

Kernel_Node *literal(const String &text) {
    return new_node(Kernel_Node_Type::LITERAL, text);
}

bool starts_with(const String &text, const String &prefix) {
    return text.rfind(prefix, 0) == 0;
}

Kernel_List_Node *function_call(const String &name, const Vector<Kernel_Node *> &arguments);

// ticks * dt as an f32. When 1 / dt is a whole number, divides by it instead: rounded once,
// where f32 ticks * dt is one ulp off on about a third of ticks and can shift t >= delay a tick.
Kernel_Node *time_of_ticks(Kernel_Node *ticks, f64 step_dt) {
    Kernel_Node *tick_count = new_cast(new_type("f32"), ticks);
    const f64 steps_per_second = std::round(1.0 / step_dt);
    if (steps_per_second >= 1.0 && steps_per_second <= 16777216.0 && std::fabs(steps_per_second * step_dt - 1.0) < 1e-12) {
        return function_call("exact_divide", {tick_count, literal(float_literal(steps_per_second))});
    }
    return new_expression("*", tick_count, literal(float_literal(step_dt)));
}

// Reads a float literal's value; false when node is not one.
bool float_literal_value(const Kernel_Node *node, f64 &value) {
    if (!node || node->body.syntax_type != Kernel_Node_Type::LITERAL) return false;
    const String &token = node->body.token;
    if (token.size() < 2 || token.back() != 'f') return false;
    try {
        usize parsed = 0;
        value = std::stod(token.substr(0, token.size() - 1), &parsed);
        return parsed == token.size() - 1;
    } catch (const Exception &) {
        return false;
    }
}

Kernel_List_Node *function_call(const String &name, const Vector<Kernel_Node *> &arguments) {
    Kernel_List_Node *call = new_node<List_Node>(Kernel_Node_Type::FUNCTION_CALL, name);
    call->children = arguments;
    return call;
}

// The walking thread's neuron is the edge's source.
Kernel_Node *edge_source() {
    return new_cast(new_type("u32"), identifier("neuron_index"));
}

// The current edge's value in plane, read through the weight matrix.
Kernel_Node *load_edge(s64 plane) {
    return function_call("spikecorec_load_edge", {
        identifier("basis_u"), identifier("basis_v"), identifier("edge_coefficients"), identifier("rank_float4_stride"),
        identifier("sparse_delta_row_start"), identifier("sparse_delta_edge_ordinal"), identifier("sparse_delta_value"),
        identifier("neuron_count"), literal(std::to_string(plane)),
        edge_source(), identifier("target"), identifier("current_edge")});
}

// Adds delta to the current edge's value in plane.
Kernel_Node *accumulate_edge(s64 plane, Kernel_Node *delta) {
    return function_call("spikecorec_accumulate_edge", {
        identifier("sparse_delta_row_start"), identifier("sparse_delta_edge_ordinal"), identifier("sparse_delta_value"),
        identifier("pending_delta_edge_ordinal"), identifier("pending_delta_value"),
        identifier("pending_delta_matrix_index"), identifier("pending_delta_control"), identifier("pending_delta_capacity"),
        identifier("neuron_count"), literal(std::to_string(plane)),
        edge_source(), identifier("current_edge"), delta});
}

// network_inputs[row * neuron_count + neuron_index]: one of the running thread's own slots.
Kernel_Node *own_input_slot(Kernel_Node *row) {
    return new_expression("[]", identifier("network_inputs"),
                          new_expression("+", new_expression("*", row, identifier("neuron_count")), identifier("neuron_index")));
}

// A name the generated synapse code reserves.
bool is_synapse_engine_name(const String &name) {
    return ENGINE_NAMES.count(name) || SYNAPSE_ENGINE_NAMES.count(name) || starts_with(name, DERIVED_PREFIX) ||
           starts_with(name, DERIVATIVE_PREFIX) || starts_with(name, REST_PREFIX) || starts_with(name, CHANGE_PREFIX) ||
           starts_with(name, RANDOM_SLOT_PREFIX);
}

// Every name a LEMS expression reads; function names are not reads.
void collect_lems_identifiers(const Lems_Parse_Node *node, Set<String> &names) {
    if (!node) return;
    if (node->body.syntax_type == Lems_Node_Subtype::IDENTIFIER) {
        names.insert(node->body.token.lexeme);
        return;
    }
    if (node->body.syntax_type == Lems_Node_Subtype::FUNCTION_CALL) {
        for (const Lems_Parse_Node *argument : children_of(static_cast<const Binary_Node<Lems_Parse_Body> *>(node)->right)) {
            collect_lems_identifiers(argument, names);
        }
        return;
    }
    if (const auto *unary = dynamic_cast<const Unary_Node<Lems_Parse_Body> *>(node)) {
        collect_lems_identifiers(unary->child, names);
    } else if (const auto *binary = dynamic_cast<const Binary_Node<Lems_Parse_Body> *>(node)) {
        collect_lems_identifiers(binary->left, names);
        collect_lems_identifiers(binary->right, names);
    }
}

bool calls_function(const Lems_Parse_Node *node, const String &function_name) {
    if (!node) return false;
    if (node->body.syntax_type == Lems_Node_Subtype::FUNCTION_CALL) {
        if (node->body.token.lexeme == function_name) return true;
        for (const Lems_Parse_Node *argument : children_of(static_cast<const Binary_Node<Lems_Parse_Body> *>(node)->right)) {
            if (calls_function(argument, function_name)) return true;
        }
        return false;
    }
    if (const auto *unary = dynamic_cast<const Unary_Node<Lems_Parse_Body> *>(node)) {
        return calls_function(unary->child, function_name);
    }
    if (const auto *binary = dynamic_cast<const Binary_Node<Lems_Parse_Body> *>(node)) {
        return calls_function(binary->left, function_name) || calls_function(binary->right, function_name);
    }
    return false;
}

} // namespace

f64 evaluate_lems(const Lems_Parse_Node *node, const Unordered_Map<String, f64> &values, const String &owner_name,
                  Random_Generator *random_generator) {
    const String &lexeme = node->body.token.lexeme;
    switch (node->body.syntax_type) {
        case Lems_Node_Subtype::FLOAT:
        case Lems_Node_Subtype::INT:
            return std::stod(lexeme);

        case Lems_Node_Subtype::IDENTIFIER: {
            auto value = values.find(lexeme);
            if (value == values.end()) throw Runtime_Error("'" + lexeme + "' has no value in " + owner_name);
            return value->second;
        }

        case Lems_Node_Subtype::OPERATOR: {
            if (const auto *unary = dynamic_cast<const Unary_Node<Lems_Parse_Body> *>(node)) {
                const f64 operand = evaluate_lems(unary->child, values, owner_name, random_generator);
                return lexeme == "-" ? -operand : operand;
            }
            const auto *binary = static_cast<const Binary_Node<Lems_Parse_Body> *>(node);
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
            throw Runtime_Error("Operator '" + lexeme + "' cannot be evaluated in " + owner_name);
        }

        case Lems_Node_Subtype::FUNCTION_CALL: {
            static const Unordered_Map<String, f64 (*)(f64)> functions = {
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
            const auto *call = static_cast<const Binary_Node<Lems_Parse_Body> *>(node);
            Vector<f64> arguments;
            for (const Lems_Parse_Node *argument : children_of(call->right)) {
                arguments.push_back(evaluate_lems(argument, values, owner_name, random_generator));
            }
            if (lexeme == "pow" && arguments.size() == 2) return std::pow(arguments[0], arguments[1]);
            if (lexeme == "random" && arguments.size() == 1) {
                if (!random_generator) {
                    throw Runtime_Error("random() in " + owner_name + " has no random generator to draw from: only a cell's "
                                        "OnStart and dynamics may call it");
                }
                return arguments[0] * random_generator->uniform();
            }
            auto function = functions.find(lexeme);
            if (function == functions.end() || arguments.size() != 1) {
                throw Runtime_Error("Function '" + lexeme + "' cannot be evaluated in " + owner_name);
            }
            return function->second(arguments[0]);
        }

        default:
            throw Runtime_Error("Malformed LEMS expression in " + owner_name);
    }
}

f64 evaluate_lems(const String &expression, const Unordered_Map<String, f64> &values, const String &owner_name,
                  Random_Generator *random_generator) {
    Unique_Pointer<Lems_Parse_Node> tree(parse_lems_expression(expression, owner_name));
    return evaluate_lems(tree.get(), values, owner_name, random_generator);
}

// An unset parameter is left out, so using one fails as an unresolved name.
Unordered_Map<String, f64> component_parameter_values(const NML_Context &context, const NML_Component_Instance &cell) {
    const NML_Component_Type &component_type = *cell.component_type;
    Unordered_Map<String, f64> values;
    Vector<const NML_Node *> derived_parameters;

    for (const NML_Component_Type *type = &component_type; type; type = type->extends) {
        for (const NML_Node *element : children_of(type->source_node)) {
            const NML_Tag &tag = element->body;
            if (tag.tag_type != NML_Declaration_Type::Parameter && tag.tag_type != NML_Declaration_Type::Constant &&
                tag.tag_type != NML_Declaration_Type::DerivedParameter) {
                continue;
            }
            // A subtype's declaration of the same name shadows this one.
            if (component_type.find_declaration(tag.namespace_key()) != element) continue;

            const String name = tag.get_attribute("name");
            try {
                if (tag.tag_type == NML_Declaration_Type::Parameter) {
                    if (cell.has_value(name)) {
                        values[name] = context.resolve_quantity_of_dimension(cell.value_or(name), tag.get_attribute("dimension"));
                    }
                } else if (tag.tag_type == NML_Declaration_Type::Constant) {
                    values[name] = context.resolve_quantity_of_dimension(tag.get_attribute("value"), tag.get_attribute("dimension"));
                } else if (tag.has_attribute("value")) {
                    derived_parameters.push_back(element);
                }
            } catch (const Runtime_Error &error) {
                throw Runtime_Error("'" + cell.id + "' (" + component_type.name + ") " + name + ": " + error.what());
            }
        }
    }

    // Derived parameters can use each other, so each pass resolves those whose inputs are known.
    bool resolved_one = true;
    while (resolved_one && !derived_parameters.empty()) {
        resolved_one = false;
        for (usize index = 0; index < derived_parameters.size();) {
            const NML_Tag &tag = derived_parameters[index]->body;
            Unique_Pointer<Lems_Parse_Node> tree(parse_lems_expression(tag.get_attribute("value"), component_type.name));

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

Unordered_Map<String, f64> starting_values(const NML_Context &context, const NML_Component_Instance &cell,
                                          Random_Generator *random_generator) {
    const NML_Component_Type &component_type = *cell.component_type;
    const Vector<String> &variable_names = component_type.state_variable_names;

    Unordered_Map<String, f64> values = component_parameter_values(context, cell);
    for (const String &name : variable_names) values[name] = 0.0;

    // In order, so a value may use the state assigned above it. A dropped variable has no value to
    // start: nothing reads it.
    const Vector<String> &dropped_names = component_type.dropped_state_variable_names;
    for (const NML_Dynamics_Expression &entry : component_type.dynamics) {
        if (entry.source_tag != NML_Declaration_Type::OnStart) continue;
        if (std::find(dropped_names.begin(), dropped_names.end(), entry.target) != dropped_names.end()) continue;
        if (std::find(variable_names.begin(), variable_names.end(), entry.target) == variable_names.end()) {
            throw Runtime_Error("OnStart in " + component_type.name + " assigns '" + entry.target +
                                "', which is not a state variable");
        }
        values[entry.target] = evaluate_lems(entry.expression, values, component_type.name, random_generator);
    }
    return values;
}

Vector<const NML_Component_Instance *> network_populations(const NML_Context &context) {
    Vector<const NML_Component_Instance *> populations;
    const NML_Component_Instance *network = context.find_instance(context.simulation.target_network_id);
    if (!network) return populations;

    for (const String &child_id : network->data_order) {
        const NML_Component_Instance *population = context.find_instance(child_id);
        if (context.is_instance_of(population, "population")) populations.push_back(population);
    }
    return populations;
}

// Longest delay + 1 rows, so this tick's row is never one a delayed arrival reads; at least
// two, since every delay is at least one tick.
s64 spike_history_length(const NML_Context &context) {
    return std::max<s64>(2, context.simulation.maximum_edge_delay + 1);
}

f64 default_kick_amplitude(const NML_Context &context, const NML_Component_Instance &cell) {
    constexpr f64 THRESHOLD_OVERSHOOT = 1.05;
    const NML_Component_Type &cell_type = *cell.component_type;
    const Unordered_Map<String, f64> values = starting_values(context, cell);

    f64 capacitance = 0.0;
    for (const NML_Component_Type *type = &cell_type; type; type = type->extends) {
        for (const NML_Node *element : children_of(type->source_node)) {
            const NML_Tag &tag = element->body;
            if (tag.tag_type != NML_Declaration_Type::Parameter) continue;
            if (tag.get_attribute("dimension") != "capacitance") continue;
            if (cell_type.find_declaration(tag.namespace_key()) != element) continue;

            auto value = values.find(tag.get_attribute("name"));
            if (value != values.end()) capacitance = value->second;
        }
    }
    if (capacitance <= 0.0) {
        throw Runtime_Error("A spike input targets a '" + cell_type.name + "', which declares no parameter of "
                            "dimension capacitance, so there is no way to work out what one event should inject. "
                            "Give the input an amplitude");
    }

    // The spiking OnCondition is the one with an EventOut; its test should read
    // "membrane > threshold".
    String spike_test;
    for (const NML_Dynamics_Expression &entry : cell_type.dynamics) {
        if (entry.source_tag != NML_Declaration_Type::EventOut || entry.condition.empty()) continue;
        spike_test = entry.condition;
        break;
    }

    String membrane_name;
    f64 threshold = 0.0;
    bool threshold_found = false;
    if (!spike_test.empty()) {
        Unique_Pointer<Lems_Parse_Node> test(parse_lems_expression(spike_test, cell_type.name));
        const auto *comparison = dynamic_cast<const Binary_Node<Lems_Parse_Body> *>(test.get());
        const String comparison_operator = test->body.token.lexeme;
        if (comparison && test->body.syntax_type == Lems_Node_Subtype::OPERATOR &&
            (comparison_operator == ">" || comparison_operator == ">=") &&
            comparison->left->body.syntax_type == Lems_Node_Subtype::IDENTIFIER) {
            membrane_name = comparison->left->body.token.lexeme;
            try {
                threshold = evaluate_lems(comparison->right, values, cell_type.name);
                threshold_found = true;
            } catch (const Runtime_Error &) {
                // not foldable to a starting value
            }
        }
    }
    if (membrane_name.empty()) {
        throw Runtime_Error("A spike input targets a '" + cell_type.name + "', whose spike condition is not a "
                            "comparison this can read, so there is no threshold to aim at. Give the input an amplitude");
    }

    // The lowest the membrane starts or resets to.
    f64 resting = values.count(membrane_name) ? values.at(membrane_name) : 0.0;
    for (const NML_Dynamics_Expression &entry : cell_type.dynamics) {
        if (entry.source_tag != NML_Declaration_Type::StateAssignment) continue;
        if (entry.condition != spike_test || entry.target != membrane_name) continue;

        try {
            resting = std::min(resting, evaluate_lems(entry.expression, values, cell_type.name));
        } catch (const Runtime_Error &) {
            // not foldable; the OnStart value stands
        }
    }

    if (!threshold_found || threshold <= resting) {
        throw Runtime_Error("A spike input targets a '" + cell_type.name + "', whose threshold resolves to no value "
                            "above its starting membrane potential, so no finite current would make it fire. Give "
                            "the input an amplitude");
    }

    return THRESHOLD_OVERSHOOT * capacitance * (threshold - resting) / context.simulation.step_dt;
}

const NML_Component_Instance &population_cell(const NML_Context &context, const NML_Component_Instance &population) {
    const NML_Component_Instance *cell = context.find_instance(population.value_or("component"));
    if (!cell || !cell->component_type) {
        throw Runtime_Error("Population '" + population.id + "' references no declared cell");
    }
    return *cell;
}

namespace {

// True when the type's dynamics have an OnEvent, so it reacts to arriving spikes.
bool has_on_event(const NML_Component_Type &component_type) {
    for (const NML_Dynamics_Expression &entry : component_type.dynamics) {
        if (entry.source_tag == NML_Declaration_Type::OnEvent) return true;
    }
    return false;
}

// The exposure a cell's select reads ("synapses[*]/i" reads "i"), or "" when it has no select.
String select_exposure(const NML_Component_Type &cell_type) {
    for (const NML_Dynamics_Expression &entry : cell_type.dynamics) {
        if (!entry.select.empty()) return entry.select.substr(entry.select.rfind('/') + 1);
    }
    return "";
}

// The variable a type exposes as exposure_name: the StateVariable or DerivedVariable declared with
// exposure="exposure_name", the nearest type first; exposure_name itself when none is.
String exposed_variable_name(const NML_Component_Type &component_type, const String &exposure_name) {
    Function<const NML_Node *(const NML_Node *)> find_exposed = [&](const NML_Node *node) -> const NML_Node * {
        for (const NML_Node *child : children_of(node)) {
            const NML_Tag &tag = child->body;
            const bool is_variable = tag.tag_type == NML_Declaration_Type::StateVariable ||
                                     tag.tag_type == NML_Declaration_Type::DerivedVariable ||
                                     tag.tag_type == NML_Declaration_Type::ConditionalDerivedVariable;
            if (is_variable && tag.get_attribute_or("exposure", "") == exposure_name) return child;
            if (const NML_Node *nested = find_exposed(child)) return nested;
        }
        return nullptr;
    };
    for (const NML_Component_Type *type = &component_type; type; type = type->extends) {
        if (const NML_Node *declaration = find_exposed(type->source_node)) return declaration->body.get_attribute("name");
    }
    return exposure_name;
}

// The increment of an accumulating value: `variable + increment` or `increment + variable`, or
// `variable - increment` (negated is then set), where increment does not read variable. nullptr
// for any other form.
const Lems_Parse_Node *accumulator_increment(const Lems_Parse_Node *value, const String &variable, bool &negated) {
    negated = false;
    if (!value || value->body.syntax_type != Lems_Node_Subtype::OPERATOR) return nullptr;
    const auto *sum = dynamic_cast<const Binary_Node<Lems_Parse_Body> *>(value);
    if (!sum) return nullptr;

    auto is_variable = [&variable](const Lems_Parse_Node *node) {
        return node->body.syntax_type == Lems_Node_Subtype::IDENTIFIER && node->body.token.lexeme == variable;
    };
    auto reads_variable = [&variable](const Lems_Parse_Node *node) {
        Set<String> names;
        collect_lems_identifiers(node, names);
        return names.count(variable) != 0;
    };

    const String &operation = value->body.token.lexeme;
    const Lems_Parse_Node *increment = nullptr;
    if (operation == "+" && is_variable(sum->left)) increment = sum->right;
    else if (operation == "+" && is_variable(sum->right)) increment = sum->left;
    else if (operation == "-" && is_variable(sum->left)) {
        increment = sum->right;
        negated = true;
    }
    if (!increment || reads_variable(increment)) return nullptr;
    return increment;
}

// Whether variable of a synapse type is a weighted arrival (see Synapse_Delivery): a StateVariable
// with no TimeDerivative, changed only by one StateAssignment in an OnEvent on "in", whose value
// adds an increment reading none of the type's StateVariables or DerivedVariables. Sets
// arrival_expression to that value.
bool is_weighted_arrival(const NML_Component_Type &component_type, const String &variable, String &arrival_expression) {
    const Vector<String> &state_names = component_type.state_variable_names;
    if (std::find(state_names.begin(), state_names.end(), variable) == state_names.end()) return false;

    Set<String> derived_names;
    const NML_Dynamics_Expression *assignment = nullptr;
    bool in_event = false;
    for (const NML_Dynamics_Expression &entry : component_type.dynamics) {
        switch (entry.source_tag) {
            case NML_Declaration_Type::TimeDerivative:
                if (entry.target == variable) return false;
                in_event = false;
                break;
            case NML_Declaration_Type::DerivedVariable:
            case NML_Declaration_Type::Case:
                derived_names.insert(entry.target);
                in_event = false;
                break;
            case NML_Declaration_Type::OnEvent:
                if (entry.target != "in") return false;
                in_event = true;
                break;
            case NML_Declaration_Type::StateAssignment:
                if (entry.target != variable) break;
                if (!in_event || assignment) return false;
                assignment = &entry;
                break;
            case NML_Declaration_Type::OnStart:
                in_event = false;
                break;
            default:
                // Regimes, OnConditions and the like; the synapse's own translation refuses them.
                return false;
        }
    }
    if (!assignment) return false;

    const Unique_Pointer<Lems_Parse_Node> value(parse_lems_expression(assignment->expression, component_type.name));
    bool negated = false;
    const Lems_Parse_Node *increment = accumulator_increment(value.get(), variable, negated);
    if (!increment || calls_function(increment, "random")) return false;
    Set<String> read_names;
    collect_lems_identifiers(increment, read_names);
    for (const String &name : read_names) {
        if (derived_names.count(name)) return false;
        if (std::find(state_names.begin(), state_names.end(), name) != state_names.end()) return false;
    }
    arrival_expression = assignment->expression;
    return true;
}

// Calls visit with each LEMS expression of a type's dynamics and the entry it belongs to: derived
// values, case conditions and values, derivatives, OnCondition tests, StateAssignment and OnStart
// values. A StateAssignment's copy of its OnCondition's test is not visited again.
void for_each_dynamics_expression(const NML_Component_Type &component_type,
                                  const Function<void(const NML_Dynamics_Expression &, const Lems_Parse_Node *)> &visit) {
    for (const NML_Dynamics_Expression &entry : component_type.dynamics) {
        Vector<const String *> texts;
        switch (entry.source_tag) {
            case NML_Declaration_Type::Case:
                texts = {&entry.condition, &entry.expression};
                break;
            case NML_Declaration_Type::DerivedVariable:
            case NML_Declaration_Type::TimeDerivative:
            case NML_Declaration_Type::OnCondition:
            case NML_Declaration_Type::StateAssignment:
            case NML_Declaration_Type::OnStart:
                texts = {&entry.expression};
                break;
            default:
                break;
        }
        for (const String *text : texts) {
            if (text->empty()) continue;
            const Unique_Pointer<Lems_Parse_Node> tree(parse_lems_expression(*text, component_type.name));
            visit(entry, tree.get());
        }
    }
}

bool is_identifier_named(const Lems_Parse_Node *node, const String &name) {
    return node && node->body.syntax_type == Lems_Node_Subtype::IDENTIFIER && node->body.token.lexeme == name;
}

// Counts how node reads total and tracker: each `total - tracker` is a difference use, and any
// other read of either name is an other use.
void count_tracked_uses(const Lems_Parse_Node *node, const String &total, const String &tracker,
                        s64 &difference_uses, s64 &other_uses) {
    if (!node) return;
    if (node->body.syntax_type == Lems_Node_Subtype::IDENTIFIER) {
        if (node->body.token.lexeme == total || node->body.token.lexeme == tracker) other_uses += 1;
        return;
    }
    if (node->body.syntax_type == Lems_Node_Subtype::FUNCTION_CALL) {
        for (const Lems_Parse_Node *argument : children_of(static_cast<const Binary_Node<Lems_Parse_Body> *>(node)->right)) {
            count_tracked_uses(argument, total, tracker, difference_uses, other_uses);
        }
        return;
    }
    if (const auto *unary = dynamic_cast<const Unary_Node<Lems_Parse_Body> *>(node)) {
        count_tracked_uses(unary->child, total, tracker, difference_uses, other_uses);
    } else if (const auto *binary = dynamic_cast<const Binary_Node<Lems_Parse_Body> *>(node)) {
        if (node->body.token.lexeme == "-" && is_identifier_named(binary->left, total) &&
            is_identifier_named(binary->right, tracker)) {
            difference_uses += 1;
            return;
        }
        count_tracked_uses(binary->left, total, tracker, difference_uses, other_uses);
        count_tracked_uses(binary->right, total, tracker, difference_uses, other_uses);
    }
}

// Every name `total - name` subtracts from total in node.
void collect_subtracted_names(const Lems_Parse_Node *node, const String &total, Set<String> &names) {
    if (!node) return;
    if (node->body.syntax_type == Lems_Node_Subtype::FUNCTION_CALL) {
        for (const Lems_Parse_Node *argument : children_of(static_cast<const Binary_Node<Lems_Parse_Body> *>(node)->right)) {
            collect_subtracted_names(argument, total, names);
        }
        return;
    }
    if (const auto *unary = dynamic_cast<const Unary_Node<Lems_Parse_Body> *>(node)) {
        collect_subtracted_names(unary->child, total, names);
    } else if (const auto *binary = dynamic_cast<const Binary_Node<Lems_Parse_Body> *>(node)) {
        if (node->body.syntax_type == Lems_Node_Subtype::OPERATOR && node->body.token.lexeme == "-" &&
            is_identifier_named(binary->left, total) && binary->right->body.syntax_type == Lems_Node_Subtype::IDENTIFIER) {
            names.insert(binary->right->body.token.lexeme);
        }
        collect_subtracted_names(binary->left, total, names);
        collect_subtracted_names(binary->right, total, names);
    }
}

// Whether a cell type reads its add-select total only as `total - tracker` (see Tracked_Total):
// every read of the total or the tracker is inside that difference, except the tracker's own
// assignments, which are all `tracker = total`. The tracker is a StateVariable with no
// TimeDerivative, and nothing else reads it, its OnStart value aside.
bool find_tracked_total(const NML_Component_Type &cell_type, Tracked_Total &tracked) {
    String total;
    for (const NML_Dynamics_Expression &entry : cell_type.dynamics) {
        if (entry.source_tag != NML_Declaration_Type::DerivedVariable || entry.select.empty()) continue;
        if (entry.reduce != "add" || !total.empty()) return false;
        total = entry.target;
    }
    if (total.empty()) return false;

    Set<String> subtracted;
    for_each_dynamics_expression(cell_type, [&](const NML_Dynamics_Expression &, const Lems_Parse_Node *tree) {
        collect_subtracted_names(tree, total, subtracted);
    });
    const Vector<String> &state_names = cell_type.state_variable_names;
    if (subtracted.size() != 1) return false;
    const String tracker = *subtracted.begin();
    if (std::find(state_names.begin(), state_names.end(), tracker) == state_names.end()) return false;

    bool tracked_only = true;
    s64 difference_uses = 0;
    for_each_dynamics_expression(cell_type, [&](const NML_Dynamics_Expression &entry, const Lems_Parse_Node *tree) {
        if (entry.target == tracker) {
            if (entry.source_tag == NML_Declaration_Type::OnStart) return;
            if (entry.source_tag == NML_Declaration_Type::StateAssignment && is_identifier_named(tree, total)) return;
            tracked_only = false;
            return;
        }
        s64 other_uses = 0;
        count_tracked_uses(tree, total, tracker, difference_uses, other_uses);
        tracked_only = tracked_only && other_uses == 0;
    });
    if (!tracked_only || difference_uses == 0) return false;
    tracked.total_variable = total;
    tracked.tracker_variable = tracker;
    return true;
}

// Whether variable of a type only accumulates: a StateVariable with no TimeDerivative, whose every
// StateAssignment adds an increment that does not read it (variable = variable + increment), and
// which nothing else reads. Its value is then the sum of its starting value and the increments.
bool only_accumulates(const NML_Component_Type &component_type, const String &variable) {
    const Vector<String> &state_names = component_type.state_variable_names;
    if (std::find(state_names.begin(), state_names.end(), variable) == state_names.end()) return false;

    bool accumulates = true;
    for_each_dynamics_expression(component_type, [&](const NML_Dynamics_Expression &entry, const Lems_Parse_Node *tree) {
        if (entry.target == variable) {
            if (entry.source_tag == NML_Declaration_Type::OnStart) return;
            bool negated = false;
            if (entry.source_tag == NML_Declaration_Type::StateAssignment && accumulator_increment(tree, variable, negated)) return;
            accumulates = false;
            return;
        }
        Set<String> names;
        collect_lems_identifiers(tree, names);
        accumulates = accumulates && names.count(variable) == 0;
    });
    return accumulates;
}

// A variable's OnStart value for one instance, when it needs no random draw; false otherwise. A
// variable with no OnStart starts at 0.
bool fixed_starting_value(const NML_Context &context, const NML_Component_Instance &instance, const String &variable,
                          f64 &value) {
    value = 0.0;
    for (const NML_Dynamics_Expression &entry : instance.component_type->dynamics) {
        if (entry.source_tag != NML_Declaration_Type::OnStart || entry.target != variable) continue;
        const Unique_Pointer<Lems_Parse_Node> tree(parse_lems_expression(entry.expression, instance.component_type->name));
        if (calls_function(tree.get(), "random")) return false;
        try {
            value = evaluate_lems(tree.get(), component_parameter_values(context, instance), instance.component_type->name);
        } catch (const Runtime_Error &) {
            return false;
        }
    }
    return true;
}

// The child slots of node that hold uses of names; declared names, parameters, types,
// function names and member names are not uses.
Vector<Kernel_Node **> use_slots(Kernel_Node *node) {
    Vector<Kernel_Node **> slots;
    switch (node->body.syntax_type) {
        case Kernel_Node_Type::LITERAL:
        case Kernel_Node_Type::IDENTIFIER:
        case Kernel_Node_Type::PARAMETER:
        case Kernel_Node_Type::PARAMETER_LIST:
        case Kernel_Node_Type::ATTRIBUTE:
        case Kernel_Node_Type::TYPE:
        case Kernel_Node_Type::POINTER:
        case Kernel_Node_Type::REFERENCE:
        case Kernel_Node_Type::INCLUDE:
        case Kernel_Node_Type::DEFINE:
        case Kernel_Node_Type::USING_NAMESPACE:
            return slots;

        case Kernel_Node_Type::DECLARATION: {
            Kernel_Trinary_Node *declaration = static_cast<Kernel_Trinary_Node *>(node);
            // An array's size is a use; its name is not.
            if (declaration->middle->body.syntax_type == Kernel_Node_Type::EXPRESSION) {
                slots.push_back(&static_cast<Kernel_Binary_Node *>(declaration->middle)->right);
            }
            slots.push_back(&declaration->right);
            return slots;
        }

        case Kernel_Node_Type::CAST:
            slots.push_back(&static_cast<Kernel_Binary_Node *>(node)->right);
            return slots;

        case Kernel_Node_Type::DEVICE_FUNCTION_IMPL:
        case Kernel_Node_Type::KERNEL_FUNCTION_IMPL:
            slots.push_back(&static_cast<Kernel_List_Node *>(node)->children[3]);
            return slots;

        default:
            break;
    }

    if (node->body.syntax_type == Kernel_Node_Type::EXPRESSION && (node->body.token == "." || node->body.token == "->")) {
        slots.push_back(&static_cast<Kernel_Binary_Node *>(node)->left);
        return slots;
    }

    if (Kernel_List_Node *list = dynamic_cast<Kernel_List_Node *>(node)) {
        for (Kernel_Node *&child : list->children) slots.push_back(&child);
    } else if (Kernel_Trinary_Node *trinary = dynamic_cast<Kernel_Trinary_Node *>(node)) {
        slots.push_back(&trinary->left);
        slots.push_back(&trinary->middle);
        slots.push_back(&trinary->right);
    } else if (Kernel_Binary_Node *binary = dynamic_cast<Kernel_Binary_Node *>(node)) {
        slots.push_back(&binary->left);
        slots.push_back(&binary->right);
    } else if (Kernel_Unary_Node *unary = dynamic_cast<Kernel_Unary_Node *>(node)) {
        slots.push_back(&unary->child);
    }
    return slots;
}

} // namespace

Input_Tree Codegen::flatten_input_tree(const NML_Component_Instance &input) const {
    Input_Tree tree;
    const s64 tick_count = context.simulation.total_tick_count;

    // A With's instance, seen from one part: "this", "parent", a Path attribute naming a child
    // part ("./syn0"), or a child's role.
    auto resolve_instance = [&](s64 part_index, const String &instance_name) -> s64 {
        const Input_Part &part = tree.parts[static_cast<usize>(part_index)];
        if (instance_name == "this") return part_index;
        if (instance_name == "parent") return part.parent;
        String named = part.instance->value_or(instance_name);
        if (starts_with(named, "./")) named = named.substr(2);
        for (s64 candidate = 0; candidate < static_cast<s64>(tree.parts.size()); candidate += 1) {
            const Input_Part &child = tree.parts[static_cast<usize>(candidate)];
            if (child.parent != part_index) continue;
            if (child.instance->id == named || child.role == instance_name) return candidate;
        }
        throw Runtime_Error("Input '" + input.id + "': '" + part.instance->id + "' names '" + instance_name +
                            "' in its Structure, which is none of its parts");
    };

    Function<void(const NML_Component_Instance &, s64, const String &)> add_part =
            [&](const NML_Component_Instance &instance, s64 parent, const String &role) {
        if (!instance.component_type) throw Runtime_Error("Input part '" + instance.id + "' has no ComponentType");
        const s64 index = static_cast<s64>(tree.parts.size());
        tree.parts.push_back({&instance, parent, role, tree.state_size});
        tree.state_size += static_cast<s64>(instance.component_type->state_variable_names.size());

        // The Child and Children slots its type declares, as (name, type held).
        Vector<Pair<String, String>> slots;
        for (const NML_Component_Type *type = instance.component_type; type; type = type->extends) {
            for (const NML_Node *declaration : children_of(type->source_node)) {
                const NML_Tag &tag = declaration->body;
                if (tag.tag_type != NML_Declaration_Type::Child && tag.tag_type != NML_Declaration_Type::Children) continue;
                slots.push_back({tag.get_attribute("name"), tag.get_attribute("type")});
            }
        }

        Input_Spike_Train spike_train;
        for (const String &child_id : instance.data_order) {
            const NML_Component_Instance *child = context.find_instance(child_id);
            if (!child || !child->component_type) continue;
            String child_role;
            for (const auto &[slot_name, slot_type] : slots) {
                if (!context.is_instance_of(child, slot_type)) continue;
                child_role = slot_name;
                break;
            }
            if (child_role.empty()) continue;

            if (!context.is_instance_of(child, "spike")) {
                add_part(*child, index, child_role);
                continue;
            }
            // A spike fires once, at its time; its own Structure sends the event to its parent.
            if (spike_train.spike_counts.empty()) spike_train.spike_counts.assign(static_cast<usize>(tick_count), 0);
            const Unordered_Map<String, f64> spike_values = component_parameter_values(context, *child);
            auto time = spike_values.find("time");
            if (time == spike_values.end()) throw Runtime_Error("Spike '" + child->id + "' in '" + instance.id + "' has no time");
            const s64 tick = units::seconds_to_ticks(time->second, context.simulation.step_dt);
            if (tick < tick_count) spike_train.spike_counts[static_cast<usize>(tick)] += 1;
        }
        if (!spike_train.spike_counts.empty()) {
            spike_train.receiver = index;
            tree.spike_trains.push_back(std::move(spike_train));
        }

        // Its own copy of each component it instantiates.
        for (const NML_Structure_Entry &entry : instance.component_type->structure) {
            if (entry.source_tag != NML_Declaration_Type::ChildInstance) continue;
            const NML_Component_Instance *referenced = context.find_instance(instance.value_or(entry.component));
            if (!referenced) {
                throw Runtime_Error("Input part '" + instance.id + "': its " + entry.component + " '" +
                                    instance.value_or(entry.component) + "' is not declared");
            }
            add_part(*referenced, index, entry.component);
        }
    };
    add_part(input, -1, "");

    // Each part's EventConnections, between parts.
    for (s64 index = 0; index < static_cast<s64>(tree.parts.size()); index += 1) {
        Unordered_Map<String, s64> aliases;
        for (const NML_Structure_Entry &entry : tree.parts[static_cast<usize>(index)].instance->component_type->structure) {
            if (entry.source_tag == NML_Declaration_Type::With) aliases[entry.alias] = resolve_instance(index, entry.instance);
            if (entry.source_tag != NML_Declaration_Type::EventConnection) continue;
            auto source = aliases.find(entry.source);
            auto target = aliases.find(entry.target);
            if (source == aliases.end() || target == aliases.end()) {
                throw Runtime_Error("Input '" + input.id + "': an EventConnection in '" +
                                    tree.parts[static_cast<usize>(index)].instance->id + "' names an alias no With declares");
            }
            tree.routes.push_back({source->second, target->second});
        }
    }
    return tree;
}

void Codegen::collect_inputs() {
    input_prototypes.clear();
    input_trees.clear();
    input_spike_train_offsets.clear();
    input_entries.clear();
    input_spike_count_size = 0;
    input_value_stride = 1;

    Unordered_Map<String, s32> prototype_of;
    for (const Simulation_Input_Config &profile : context.simulation.input_profiles) {
        const NML_Component_Instance *input = context.find_instance(profile.input_component_id);
        if (!input || !input->component_type) {
            throw Runtime_Error("Input '" + profile.input_component_id + "' is not declared");
        }

        auto known = prototype_of.find(input->id);
        s32 prototype = 0;
        if (known != prototype_of.end()) {
            prototype = known->second;
        } else {
            prototype = static_cast<s32>(input_prototypes.size());
            prototype_of[input->id] = prototype;
            input_prototypes.push_back(input);
            input_trees.push_back(flatten_input_tree(*input));

            Vector<s64> offsets;
            for (const Input_Spike_Train &train : input_trees.back().spike_trains) {
                offsets.push_back(input_spike_count_size);
                input_spike_count_size += static_cast<s64>(train.spike_counts.size());
            }
            input_spike_train_offsets.push_back(std::move(offsets));
            input_value_stride = std::max(input_value_stride, 1 + input_trees.back().state_size);
        }

        for (const Input_Target &target : profile.targets) {
            const s64 neuron_index = context.neuron_index_of(target.neuron_index);
            if (neuron_index < 0) {
                throw Runtime_Error("Input '" + input->id + "' targets cell memory index " +
                                    std::to_string(target.neuron_index) + ", which is in no neuron");
            }
            input_entries.push_back({neuron_index, prototype, static_cast<f32>(target.weight)});
        }
    }
    std::stable_sort(input_entries.begin(), input_entries.end(),
                     [](const Input_Entry &first, const Input_Entry &second) { return first.neuron_index < second.neuron_index; });

    log::logger().info("Codegen: {} inputs on {} cells, {} floats of input state per cell",
                       input_prototypes.size(), input_entries.size(), input_value_stride);
}

bool Codegen::cells_receive_events() const {
    for (const NML_Component_Instance *population : network_populations(context)) {
        if (has_on_event(*population_cell(context, *population).component_type)) return true;
    }
    return false;
}

void Codegen::collect_synapse_deliveries() {
    synapse_deliveries.clear();

    Vector<Pair<s64, const NML_Component_Type *>> population_ends;
    s64 first_neuron = 0;
    for (const NML_Component_Instance *population : network_populations(context)) {
        first_neuron += context.get_population_size(population);
        population_ends.push_back({first_neuron, population_cell(context, *population).component_type});
    }
    auto cell_type_of = [&](s64 neuron_index) -> const NML_Component_Type * {
        for (const auto &[population_end, cell_type] : population_ends) {
            if (neuron_index < population_end) return cell_type;
        }
        return nullptr;
    };

    // The exposures each synapse's target cells read, each cell type once per synapse.
    Unordered_Map<String, Ordered_Set<String>> read_exposures;
    Unordered_Map<String, Set<const NML_Component_Type *>> seen_cell_types;
    auto include = [&](const NML_Network_Edge &edge) {
        const NML_Component_Type *cell_type = cell_type_of(edge.child);
        if (!cell_type || !seen_cell_types[edge.component_id].insert(cell_type).second) return;
        const String exposure = select_exposure(*cell_type);
        if (!exposure.empty()) read_exposures[edge.component_id].insert(exposure);
    };
    for (const Vector<NML_Network_Edge> &row : context.simulation.network_data.list) {
        for (const NML_Network_Edge &edge : row) include(edge);
    }
    for (const NML_Network_Edge &edge : context.simulation.sampled_connections) include(edge);

    for (const NML_Component_Instance &synapse : context.simulation.synapse_instances) {
        if (!synapse.component_type) throw Runtime_Error("Synapse '" + synapse.id + "' has no ComponentType");
        const NML_Component_Type &component_type = *synapse.component_type;

        // The kernel delivers one value of a synapse to all of its targets.
        const Ordered_Set<String> &exposures = read_exposures[synapse.id];
        if (exposures.size() > 1) {
            String names;
            for (const String &exposure : exposures) names += (names.empty() ? "'" : ", '") + exposure + "'";
            throw Runtime_Error("Synapse '" + synapse.id + "' (" + component_type.name + ") reaches cells that read "
                                "different exposures of it, " + names + ", but a synapse delivers one value to all "
                                "of its targets");
        }

        Synapse_Delivery delivery;
        if (!exposures.empty()) {
            delivery.read_exposure = *exposures.begin();
            delivery.read_variable = exposed_variable_name(component_type, delivery.read_exposure);
            delivery.weighted_arrival =
                    is_weighted_arrival(component_type, delivery.read_variable, delivery.arrival_expression);
        }
        if (delivery.weighted_arrival) {
            delivery.starting_value = starting_values(context, synapse).at(delivery.read_variable);
            log::logger().info("Codegen: synapse '{}' delivers '{}' as weighted arrivals: each arrival adds its "
                               "OnEvent increment to its target's sum once, and nothing is kept per connection",
                               synapse.id, delivery.read_exposure);
        } else {
            delivery.edge_state_variable_names = component_type.state_variable_names;
        }
        for (const NML_Component_Type *cell_type : seen_cell_types[synapse.id]) delivery.target_cell_types.insert(cell_type->name);
        synapse_deliveries[synapse.id] = std::move(delivery);
    }
}

const Synapse_Delivery &Codegen::synapse_delivery(const NML_Component_Instance &synapse) const {
    auto delivery = synapse_deliveries.find(synapse.id);
    if (delivery == synapse_deliveries.end()) {
        throw Runtime_Error("Codegen: synapse '" + synapse.id + "' has no delivery; collect_synapse_deliveries "
                            "must run before the kernel and its memory are built");
    }
    return delivery->second;
}

bool Codegen::has_weighted_arrivals() const {
    for (const NML_Component_Instance &synapse : context.simulation.synapse_instances) {
        if (synapse_delivery(synapse).weighted_arrival) return true;
    }
    return false;
}

bool Codegen::only_weighted_arrivals() const {
    Set<String> connected(context.simulation.largest_connection_weight_per_synapse.size());
    for (const auto &[synapse_id, largest_weight] : context.simulation.largest_connection_weight_per_synapse) {
        connected.insert(synapse_id);
    }
    for (const Vector<NML_Network_Edge> &row : context.simulation.network_data.list) {
        for (const NML_Network_Edge &edge : row) connected.insert(edge.component_id);
    }
    for (const NML_Network_Edge &edge : context.simulation.sampled_connections) connected.insert(edge.component_id);

    bool any_connected = false;
    for (const NML_Component_Instance &synapse : context.simulation.synapse_instances) {
        if (!connected.count(synapse.id)) continue;
        if (!synapse_delivery(synapse).weighted_arrival) return false;
        any_connected = true;
    }
    return any_connected;
}

s64 Codegen::network_input_row_count() const {
    return has_weighted_arrivals() || !tracked_totals.empty() ? RUNNING_SUM_ROW + 1 : WEIGHTED_ARRIVAL_ROW;
}

void Codegen::collect_tracked_totals() {
    tracked_totals.clear();

    // Candidates by cell type: the pattern holds and the tracker starts at 0 in every population.
    Unordered_Map<String, Tracked_Total> candidates;
    Set<String> refused;
    Vector<Pair<s64, const NML_Component_Type *>> population_ends;
    s64 first_neuron = 0;
    for (const NML_Component_Instance *population : network_populations(context)) {
        const NML_Component_Instance &cell = population_cell(context, *population);
        const NML_Component_Type &cell_type = *cell.component_type;
        first_neuron += context.get_population_size(population);
        population_ends.push_back({first_neuron, &cell_type});
        if (refused.count(cell_type.name)) continue;

        Tracked_Total tracked;
        f64 tracker_start = 0.0;
        if (!find_tracked_total(cell_type, tracked) ||
            !fixed_starting_value(context, cell, tracked.tracker_variable, tracker_start) || tracker_start != 0.0) {
            refused.insert(cell_type.name);
            candidates.erase(cell_type.name);
            continue;
        }
        candidates[cell_type.name] = tracked;
    }
    if (candidates.empty()) return;

    // Everything that reaches such a cell must deliver increments to a total that starts at 0:
    // synapses as weighted arrivals, and inputs whose exposure only accumulates.
    for (const NML_Component_Instance &synapse : context.simulation.synapse_instances) {
        const Synapse_Delivery &delivery = synapse_delivery(synapse);
        if (delivery.weighted_arrival && delivery.starting_value == 0.0) continue;
        for (const String &cell_type_name : delivery.target_cell_types) candidates.erase(cell_type_name);
    }
    auto cell_type_of = [&](s64 neuron_index) -> const NML_Component_Type * {
        for (const auto &[population_end, cell_type] : population_ends) {
            if (neuron_index < population_end) return cell_type;
        }
        return nullptr;
    };
    Ordered_Set<Pair<s32, String>> checked;
    for (const Input_Entry &entry : input_entries) {
        const NML_Component_Type *cell_type = cell_type_of(entry.neuron_index);
        if (!cell_type || !candidates.count(cell_type->name) || !checked.insert({entry.prototype, cell_type->name}).second) {
            continue;
        }
        const NML_Component_Instance &input = *input_prototypes[static_cast<usize>(entry.prototype)];
        const String variable = exposed_variable_name(*input.component_type, select_exposure(*cell_type));
        f64 input_start = 0.0;
        if (context.is_instance_of(&input, "baseSpikeSource") || !only_accumulates(*input.component_type, variable) ||
            !fixed_starting_value(context, input, variable, input_start) || input_start != 0.0) {
            candidates.erase(cell_type->name);
        }
    }

    // A tracker an OutputColumn records keeps its slot; any other is dropped.
    for (const Recording_Config &profile : context.simulation.recording_profiles) {
        for (const Recording_Selection &selection : profile.selections) {
            if (!selection.event_port.empty() || selection.neuron_index < 0) continue;
            const NML_Component_Type *cell_type = cell_type_of(context.neuron_index_of(selection.neuron_index));
            if (!cell_type) continue;
            auto tracked = candidates.find(cell_type->name);
            if (tracked != candidates.end() && selection.variable_name == tracked->second.tracker_variable) {
                tracked->second.tracker_kept = true;
            }
        }
    }

    for (auto &[cell_type_name, tracked] : candidates) {
        log::logger().info("Codegen: '{}' reads its total '{}' only less '{}', so the kernel keeps their difference "
                           "instead of either; '{}' is {}", cell_type_name, tracked.total_variable,
                           tracked.tracker_variable, tracked.tracker_variable,
                           tracked.tracker_kept ? "recorded, so kept" : "read by nothing else, so dropped");
    }
    tracked_totals = std::move(candidates);
}

void Codegen::check_population_cells() const {
    for (const NML_Component_Instance *population : network_populations(context)) {
        if (context.get_population_size(population) == 0) continue;
        const NML_Component_Instance &cell = population_cell(context, *population);
        if (!context.is_instance_of(&cell, "baseSpikeSource")) continue;

        // A cell's own dynamics are compiled; child components (a spikeArray's spikes) are not.
        bool has_child_components = false;
        for (const String &child_id : cell.data_order) {
            const NML_Component_Instance *child = context.find_instance(child_id);
            has_child_components = has_child_components || (child && child->component_type);
        }
        if (!has_child_components) continue;
        throw Runtime_Error("Population '" + population->id + "' is made of '" + cell.id + "' (" +
                            cell.component_type->name + "), a spike source whose spikes come from its child "
                            "components, which the engine runs only for an input, so its cells would never fire. "
                            "Fire a cell with it instead: an explicitInput or inputList of '" + cell.id +
                            "' onto a population of cells");
    }
}

void Codegen::allocate_cell_model_memory() {
    // Two rows like network_inputs when some cell has an OnEvent; otherwise one slot to bind.
    const s64 event_arrival_slots = cells_receive_events() ? 2 * context.simulation.total_neuron_count : 1;

    // Every input buffer has at least one slot, so there is always one to bind. The inputs and the
    // tracked totals are already collected (see collect_tracked_totals).
    const s64 entry_count = static_cast<s64>(input_entries.size());

    device->partition(sizeof(f32) * context.get_cell_state_size(), Engine_Datatype::FLOAT32, data_partitions)            // cell_state
          .partition(sizeof(f32) * network_input_row_count() * context.simulation.total_neuron_count, Engine_Datatype::FLOAT32,
                     data_partitions)                                                                                      // network_inputs
          .partition(sizeof(u8) * spike_history_length(context) * context.simulation.total_neuron_count,
                     Engine_Datatype::UNSIGNED8, data_partitions)                                                           // spike_history
          .partition(sizeof(s64) * context.simulation.total_neuron_count, Engine_Datatype::SIGNED64, data_partitions)       // last_spiked
          .partition(sizeof(f32), Engine_Datatype::FLOAT32, data_partitions)                                                // empty_edge_plane
          .partition(sizeof(u32) * event_arrival_slots, Engine_Datatype::UNSIGNED32, data_partitions)                       // event_arrival_count
          .partition(sizeof(s64) * (context.simulation.total_neuron_count + 1), Engine_Datatype::SIGNED64, data_partitions) // input_row_start
          .partition(sizeof(s32) * std::max<s64>(1, entry_count), Engine_Datatype::SIGNED32, data_partitions)               // input_entry_prototype
          .partition(sizeof(f32) * std::max<s64>(1, entry_count * input_value_stride), Engine_Datatype::FLOAT32,
                     data_partitions)                                                                                      // input_values
          .partition(sizeof(u8) * std::max<s64>(1, input_spike_count_size), Engine_Datatype::UNSIGNED8, data_partitions); // input_spike_counts

    Engine_Pointer slab = device->allocate(data_partitions);
    data_partitions.push_back(slab);

    log::logger().debug("Codegen: {} bytes: cell_state {}, network_inputs {}, spike_history {}, last_spiked {}, "
                        "event_arrival_count {}",
                        slab.total_bytes, context.get_cell_state_size(), network_input_row_count() * context.simulation.total_neuron_count,
                        spike_history_length(context) * context.simulation.total_neuron_count,
                        context.simulation.total_neuron_count, event_arrival_slots);
}

Codegen::~Codegen() {
    delete root;
    for (auto &[type_name, template_body] : component_type_templates) delete template_body;
    for (auto &[type_name, template_body] : synapse_type_templates) delete template_body;
}

void Codegen::initialize_input_state(Random_Generator &random_generator) {
    if (data_partitions.size() < 10) throw Runtime_Error("initialize_input_state: input memory is not allocated");
    s64 *row_start = data_partitions[6].get_contents_as<s64>();
    s32 *entry_prototype = data_partitions[7].get_contents_as<s32>();
    f32 *values = data_partitions[8].get_contents_as<f32>();
    u8 *spike_counts = data_partitions[9].get_contents_as<u8>();

    // Neuron n's entries are input_row_start[n] up to input_row_start[n + 1].
    const s64 neuron_count = context.simulation.total_neuron_count;
    std::fill(row_start, row_start + neuron_count + 1, 0);
    for (const Input_Entry &entry : input_entries) row_start[entry.neuron_index + 1] += 1;
    for (s64 neuron_index = 0; neuron_index < neuron_count; neuron_index += 1) {
        row_start[neuron_index + 1] += row_start[neuron_index];
    }

    // A part starts every entry the same unless its OnStart calls random().
    auto draws_on_start = [](const NML_Component_Type &component_type) {
        for (const NML_Dynamics_Expression &entry : component_type.dynamics) {
            if (entry.source_tag != NML_Declaration_Type::OnStart) continue;
            const Unique_Pointer<Lems_Parse_Node> tree(parse_lems_expression(entry.expression, component_type.name));
            if (calls_function(tree.get(), "random")) return true;
        }
        return false;
    };
    Ordered_Map<Pair<s32, usize>, Vector<f32>> shared_starting_state;
    for (usize entry_index = 0; entry_index < input_entries.size(); entry_index += 1) {
        const Input_Entry &entry = input_entries[entry_index];
        entry_prototype[entry_index] = entry.prototype;
        f32 *entry_values = values + static_cast<s64>(entry_index) * input_value_stride;
        entry_values[0] = entry.weight;

        const Input_Tree &tree = input_trees[static_cast<usize>(entry.prototype)];
        for (usize part_index = 0; part_index < tree.parts.size(); part_index += 1) {
            const Input_Part &part = tree.parts[part_index];
            const NML_Component_Type &component_type = *part.instance->component_type;
            const Pair<s32, usize> key{entry.prototype, part_index};

            Vector<f32> starting_state;
            auto shared = shared_starting_state.find(key);
            if (shared != shared_starting_state.end()) {
                starting_state = shared->second;
            } else {
                const Unordered_Map<String, f64> part_values = starting_values(context, *part.instance, &random_generator);
                for (const String &name : component_type.state_variable_names) {
                    starting_state.push_back(static_cast<f32>(part_values.at(name)));
                }
                if (!draws_on_start(component_type)) shared_starting_state[key] = starting_state;
            }
            std::copy(starting_state.begin(), starting_state.end(), entry_values + 1 + part.first_state_slot);
        }
    }

    for (usize prototype = 0; prototype < input_trees.size(); prototype += 1) {
        const Vector<Input_Spike_Train> &trains = input_trees[prototype].spike_trains;
        for (usize train = 0; train < trains.size(); train += 1) {
            std::copy(trains[train].spike_counts.begin(), trains[train].spike_counts.end(),
                      spike_counts + input_spike_train_offsets[prototype][train]);
        }
    }
}

void Codegen::initialize_cell_state(Random_Generator &random_generator) {
    if (data_partitions.empty()) throw Runtime_Error("initialize_cell_state: cell memory is not allocated");
    f32 *cell_state = data_partitions[0].get_contents_as<f32>();

    for (const NML_Component_Instance *population : network_populations(context)) {
        const NML_Component_Instance &cell = population_cell(context, *population);
        const NML_Component_Type &component_type = *cell.component_type;
        const Vector<String> &variable_names = component_type.state_variable_names;
        const s64 population_base = context.simulation.population_base_indices.at(population->id);
        const s64 population_size = context.get_population_size(population);
        const s64 slot_count = NML_Context::cell_slot_count(component_type);

        // Every neuron starts the same unless an OnStart calls random().
        bool draws_on_start = false;
        for (const NML_Dynamics_Expression &entry : component_type.dynamics) {
            if (entry.source_tag != NML_Declaration_Type::OnStart) continue;
            const Unique_Pointer<Lems_Parse_Node> tree(parse_lems_expression(entry.expression, component_type.name));
            draws_on_start = draws_on_start || calls_function(tree.get(), "random");
        }

        Vector<f32> starting_state;
        for (s64 local_index = 0; local_index < population_size; local_index += 1) {
            if (local_index == 0 || draws_on_start) {
                const Unordered_Map<String, f64> values = starting_values(context, cell, &random_generator);
                starting_state.clear();
                for (const String &name : variable_names) starting_state.push_back(static_cast<f32>(values.at(name)));
            }
            // A cell with no state variables keeps one slot, which stays zero.
            std::fill_n(cell_state + population_base + local_index * slot_count, slot_count, 0.0f);
            std::copy(starting_state.begin(), starting_state.end(),
                      cell_state + population_base + local_index * slot_count);
        }
    }
}

// AST construction

Kernel_Node *new_type(const String &type) {
    return new_node(Kernel_Node_Type::TYPE, type);
}

Kernel_Node *new_pointer(Kernel_Node *pointee) {
    Kernel_Unary_Node *pointer = new_node<Unary_Node>(Kernel_Node_Type::POINTER, "");
    pointer->child = pointee;
    return pointer;
}

Kernel_Node *new_declaration(Kernel_Node *type, const String &name, Kernel_Node *value) {
    Kernel_Trinary_Node *declaration = new_node<Trinary_Node>(Kernel_Node_Type::DECLARATION, "");
    declaration->left = type;
    declaration->middle = identifier(name);
    declaration->right = value;
    return declaration;
}

Kernel_Node *new_assignment(Kernel_Node *target, const String &assignment_operator, Kernel_Node *value) {
    Kernel_Binary_Node *assignment = new_node<Binary_Node>(Kernel_Node_Type::ASSIGNMENT, assignment_operator);
    assignment->left = target;
    assignment->right = value;
    return assignment;
}

Kernel_Node *new_expression(const String &expression_operator, Kernel_Node *left, Kernel_Node *right) {
    Kernel_Binary_Node *expression = new_node<Binary_Node>(Kernel_Node_Type::EXPRESSION, expression_operator);
    expression->left = left;
    expression->right = right;
    return expression;
}

Kernel_Node *new_unary_expression(const String &expression_operator, Kernel_Node *operand) {
    Kernel_Unary_Node *expression = new_node<Unary_Node>(Kernel_Node_Type::EXPRESSION, expression_operator);
    expression->child = operand;
    return expression;
}

Kernel_Node *new_cast(Kernel_Node *type, Kernel_Node *value) {
    Kernel_Binary_Node *cast = new_node<Binary_Node>(Kernel_Node_Type::CAST, "");
    cast->left = type;
    cast->right = value;
    return cast;
}

Kernel_List_Node *new_block() {
    return new_node<List_Node>(Kernel_Node_Type::BLOCK, "");
}

Kernel_List_Node *new_conditional() {
    return new_node<List_Node>(Kernel_Node_Type::CONDITIONAL, "");
}

void add_branch(Kernel_List_Node *conditional, Kernel_Node *test, Kernel_Node *body) {
    if (body->body.syntax_type != Kernel_Node_Type::BLOCK) throw Runtime_Error("Conditional branch body must be a BLOCK");
    if (!test && conditional->children.empty()) throw Runtime_Error("Conditional cannot start with else");
    if (!conditional->children.empty() && !static_cast<Kernel_Binary_Node *>(conditional->children.back())->left)
        throw Runtime_Error("Conditional already ends with else");

    Kernel_Binary_Node *branch = new_node<Binary_Node>(Kernel_Node_Type::CONDITIONAL_BRANCH, "");
    branch->left = test;
    branch->right = body;
    conditional->children.push_back(branch);
}

// Tree operations

Kernel_List_Node *find_function(Kernel_Node *program, const String &name) {
    for (Kernel_Node *item : static_cast<Kernel_List_Node *>(program)->children) {
        const Kernel_Node_Type type = item->body.syntax_type;
        if (type != Kernel_Node_Type::DEVICE_FUNCTION_IMPL && type != Kernel_Node_Type::KERNEL_FUNCTION_IMPL) continue;

        Kernel_List_Node *function = static_cast<Kernel_List_Node *>(item);
        if (function->children[1]->body.token == name) return function;
    }
    return nullptr;
}

s64 find_statement(Kernel_List_Node *block, const String &name) {
    for (usize index = 0; index < block->children.size(); index += 1) {
        Kernel_Node *statement = block->children[index];
        Kernel_Node *named = nullptr;
        if (statement->body.syntax_type == Kernel_Node_Type::DECLARATION) {
            named = static_cast<Kernel_Trinary_Node *>(statement)->middle;
            if (named->body.syntax_type == Kernel_Node_Type::EXPRESSION) named = static_cast<Kernel_Binary_Node *>(named)->left;
        } else if (statement->body.syntax_type == Kernel_Node_Type::ASSIGNMENT) {
            named = static_cast<Kernel_Binary_Node *>(statement)->left;
        }
        if (named && named->body.syntax_type == Kernel_Node_Type::IDENTIFIER && named->body.token == name) {
            return static_cast<s64>(index);
        }
    }
    return -1;
}

void insert_statements(Kernel_List_Node *block, usize index, const Vector<Kernel_Node *> &statements) {
    if (index > block->children.size()) throw Runtime_Error("insert_statements: index past the end of the block");
    block->children.insert(block->children.begin() + index, statements.begin(), statements.end());
}

Kernel_Node *clone(const Kernel_Node *node) {
    if (!node) return nullptr;

    if (const Kernel_List_Node *list = dynamic_cast<const Kernel_List_Node *>(node)) {
        Kernel_List_Node *copy = new Kernel_List_Node(list->body);
        for (const Kernel_Node *child : list->children) copy->children.push_back(clone(child));
        return copy;
    }
    if (const Kernel_Trinary_Node *trinary = dynamic_cast<const Kernel_Trinary_Node *>(node)) {
        Kernel_Trinary_Node *copy = new Kernel_Trinary_Node(trinary->body);
        copy->left = clone(trinary->left);
        copy->middle = clone(trinary->middle);
        copy->right = clone(trinary->right);
        return copy;
    }
    if (const Kernel_Binary_Node *binary = dynamic_cast<const Kernel_Binary_Node *>(node)) {
        Kernel_Binary_Node *copy = new Kernel_Binary_Node(binary->body);
        copy->left = clone(binary->left);
        copy->right = clone(binary->right);
        return copy;
    }
    if (const Kernel_Unary_Node *unary = dynamic_cast<const Kernel_Unary_Node *>(node)) {
        Kernel_Unary_Node *copy = new Kernel_Unary_Node(unary->body);
        copy->child = clone(unary->child);
        return copy;
    }
    return new Kernel_Node(node->body);
}

void replace_identifier(Kernel_Node *&node, const String &name, const Kernel_Node *replacement) {
    if (!node) return;
    if (node->body.syntax_type == Kernel_Node_Type::IDENTIFIER) {
        if (node->body.token != name) return;
        delete node;
        node = clone(replacement);
        return;
    }
    for (Kernel_Node **slot : use_slots(node)) replace_identifier(*slot, name, replacement);
}

void fold_constants(Kernel_Node *&node) {
    if (!node) return;
    for (Kernel_Node **slot : use_slots(node)) fold_constants(*slot);
    if (node->body.syntax_type != Kernel_Node_Type::EXPRESSION) return;

    const String &expression_operator = node->body.token;
    f64 result = 0.0;
    if (const auto *unary = dynamic_cast<const Kernel_Unary_Node *>(node)) {
        f64 operand = 0.0;
        if (expression_operator != "-" || !float_literal_value(unary->child, operand)) return;
        result = -operand;
    } else if (const auto *binary = dynamic_cast<const Kernel_Binary_Node *>(node)) {
        f64 left = 0.0;
        f64 right = 0.0;
        if (!float_literal_value(binary->left, left) || !float_literal_value(binary->right, right)) return;
        if (expression_operator == "+") result = left + right;
        else if (expression_operator == "-") result = left - right;
        else if (expression_operator == "*") result = left * right;
        else if (expression_operator == "/" && right != 0.0) result = left / right;
        else return;
    } else {
        return;
    }
    if (!std::isfinite(result)) return;
    delete node;
    node = literal(float_literal(result));
}

void collect_identifiers(const Kernel_Node *node, Set<String> &names) {
    if (!node) return;
    if (node->body.syntax_type == Kernel_Node_Type::IDENTIFIER) {
        names.insert(node->body.token);
        return;
    }
    for (Kernel_Node **slot : use_slots(const_cast<Kernel_Node *>(node))) collect_identifiers(*slot, names);
}

void bake_parameter(Kernel_List_Node *function, const String &name, const Kernel_Node *value) {
    Kernel_List_Node *parameters = static_cast<Kernel_List_Node *>(function->children[2]);
    auto parameter = std::find_if(parameters->children.begin(), parameters->children.end(), [&name](Kernel_Node *candidate) {
        return static_cast<Kernel_Trinary_Node *>(candidate)->middle->body.token == name;
    });
    if (parameter == parameters->children.end()) {
        throw Runtime_Error("bake_parameter: " + function->children[1]->body.token + " has no parameter '" + name + "'");
    }
    delete *parameter;
    parameters->children.erase(parameter);
    replace_identifier(function->children[3], name, value);
}

// AST generation

Kernel_Node *Codegen::create_kernel_root() {
    const char *boilerplate = backend == Kernel_Backend::METAL ? METAL_KERNEL_BOILERPLATE : CUDA_KERNEL_BOILERPLATE;
    Unique_Pointer<Kernel_List_Node> program(Kernel_Parser(tokenize_kernel(boilerplate, backend), backend).parse_program());

    // Metal's k^2-tree helpers are shared with the precompiled kernels, so they live in their own
    // file and go in ahead of the boilerplate's first function.
    if (backend == Kernel_Backend::METAL) {
        const String helpers_path = String(SPIKECOREC_METAL_DEVICE_DIR) + "/k2tree_device.metalinc";
        Input_File_Stream helpers_file(helpers_path);
        if (!helpers_file) throw Runtime_Error("create_kernel_root: cannot read " + helpers_path);
        String_Stream helpers_text;
        helpers_text << helpers_file.rdbuf();

        Unique_Pointer<Kernel_List_Node> helpers(
                Kernel_Parser(tokenize_kernel(helpers_text.str(), backend), backend).parse_program());

        auto first_function = std::find_if(program->children.begin(), program->children.end(), [](Kernel_Node *item) {
            return item->body.syntax_type == Kernel_Node_Type::DEVICE_FUNCTION_IMPL ||
                   item->body.syntax_type == Kernel_Node_Type::KERNEL_FUNCTION_IMPL;
        });
        program->children.insert(first_function, helpers->children.begin(), helpers->children.end());
        helpers->children.clear();
    }

    delete root;
    root = program.release();
    return root;
}

Kernel_Node *Codegen::translate_kernel_code() {
    create_kernel_root();

    Kernel_List_Node *master_step = find_function(root, "master_step");
    if (!master_step) throw Runtime_Error("translate_kernel_code: the boilerplate has no master_step");
    Kernel_List_Node *body = static_cast<Kernel_List_Node *>(master_step->children[3]);

    // One branch per population, over its range of neurons. Its inputs run first but are
    // translated after every cell, so the cells' random() slots come first.
    Kernel_List_Node *dispatch = new_conditional();
    s64 first_neuron = 0;
    try {
        Vector<Tuple<const NML_Component_Instance *, s64, Kernel_List_Node *>> population_bodies;
        for (const NML_Component_Instance *population : network_populations(context)) {
            const s64 population_size = context.get_population_size(population);
            if (population_size == 0) continue;

            Kernel_List_Node *population_body = translate_component_instance(*population, first_neuron);
            Kernel_Node *test = new_expression("<", identifier("neuron_index"),
                                              literal(std::to_string(first_neuron + population_size)));
            add_branch(dispatch, test, population_body);
            population_bodies.push_back({population, first_neuron, population_body});
            first_neuron += population_size;
        }
        for (const auto &[population, population_first_neuron, population_body] : population_bodies) {
            Kernel_List_Node *inputs = translate_input_entries(*population, population_first_neuron);
            if (inputs) insert_statements(population_body, 0, {inputs});
        }
    } catch (...) {
        delete dispatch;
        throw;
    }

    s64 spiked_index = find_statement(body, "spiked");
    if (spiked_index < 0) {
        delete dispatch;
        throw Runtime_Error("translate_kernel_code: master_step declares no 'spiked'");
    }

    // Arrivals are counted like network_inputs, in two rows by tick parity: a neuron reads its
    // count from current_row while this tick's arrivals go into next_row.
    const bool receives_events = cells_receive_events();
    if (receives_events) {
        Kernel_Node *arrival_slot = new_expression("[]", identifier("event_arrival_count"), identifier("input_slot"));
        insert_statements(body, static_cast<usize>(spiked_index),
                          {new_declaration(new_type("const s32"), "event_arrivals", new_cast(new_type("s32"), arrival_slot)),
                           new_assignment(clone(arrival_slot), "=", literal("0"))});
        spiked_index += 2;
    }

    // The cells run in a tick's first walk pass only; a later pass only walks deferred neurons' edges.
    if (dispatch->children.empty()) {
        delete dispatch;
    } else {
        Kernel_List_Node *first_pass_body = new_block();
        first_pass_body->children.push_back(dispatch);
        Kernel_List_Node *first_pass_only = new_conditional();
        add_branch(first_pass_only, new_expression("==", identifier("walk_pass"), literal("0")), first_pass_body);
        insert_statements(body, static_cast<usize>(spiked_index) + 1, {first_pass_only});
    }

    // Each edge runs its prototype's synapse; prototypes are numbered in synapse_instances order,
    // as the engine numbers the runs.
    const Vector<NML_Component_Instance> &synapses = context.simulation.synapse_instances;
    if (!synapses.empty()) {
        Kernel_List_Node *walk = nullptr;
        s64 arrived_index = -1;
        for (Kernel_Node *statement : body->children) {
            if (statement->body.syntax_type != Kernel_Node_Type::WHILE) continue;
            Kernel_List_Node *loop_body = static_cast<Kernel_List_Node *>(static_cast<Kernel_Binary_Node *>(statement)->right);
            arrived_index = find_statement(loop_body, "arrived");
            if (arrived_index < 0) continue;
            walk = loop_body;
            break;
        }
        if (!walk) throw Runtime_Error("translate_kernel_code: master_step's propagate walk declares no 'arrived'");

        Vector<Kernel_Node *> statements;
        if (synapses.size() == 1) {
            Kernel_List_Node *synapse_body = translate_synapse_instance(synapses[0]);
            statements = synapse_body->children;
            synapse_body->children.clear();
            delete synapse_body;
        } else {
            Kernel_List_Node *prototypes = new_conditional();
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
                                  {identifier("run_first_edge_ordinal"), identifier("run_synapse_prototype"),
                                   identifier("run_count"), identifier("current_edge")})));
            statements.push_back(prototypes);
        }
        insert_statements(walk, static_cast<usize>(arrived_index) + 1, statements);

        if (receives_events) {
            Kernel_Node *next_slot = new_expression(
                    "[]", identifier("event_arrival_count"),
                    new_expression("+", new_expression("*", identifier("next_row"), identifier("neuron_count")),
                                   identifier("target")));
            Kernel_List_Node *count_arrival = new_block();
            count_arrival->children.push_back(function_call("atomic_increment", {next_slot}));
            Kernel_List_Node *on_arrival = new_conditional();
            add_branch(on_arrival, identifier("arrived"), count_arrival);
            walk->children.push_back(on_arrival);
        }

        // When every synapse delivers weighted arrivals, a neuron's edges have work only on a tick
        // one of its spikes reaches a target on the next tick: one a delay less one tick ago, for
        // some delay in the network's range. Any other tick skips the walk. With a shortest delay
        // of one tick that is every tick of the active window, which master_step already tests.
        const s64 shortest_delay = std::max<s64>(1, context.simulation.minimum_edge_delay);
        const s64 longest_delay = std::max<s64>(1, context.simulation.maximum_edge_delay);
        if (only_weighted_arrivals() && shortest_delay > 1) {
            const s64 walk_stack_index = find_statement(body, "walk_stack_row_base");
            if (walk_stack_index < 0) throw Runtime_Error("translate_kernel_code: master_step declares no 'walk_stack_row_base'");

            Kernel_Node *spike_row = new_expression("%", new_expression("-", identifier("tick"), identifier("elapsed_ticks")),
                                                    identifier("spike_history_length"));
            Kernel_Node *spiked_then = new_expression(
                    "!=",
                    new_expression("[]", identifier("spike_history"),
                                   new_expression("+", new_expression("*", spike_row, identifier("neuron_count")),
                                                  identifier("neuron_index"))),
                    literal("0"));
            Kernel_List_Node *mark_due = new_block();
            mark_due->children.push_back(new_assignment(identifier("arrival_due"), "=", identifier("true")));
            Kernel_List_Node *check_tick = new_conditional();
            add_branch(check_tick,
                       new_expression("&&", new_expression(">=", identifier("tick"), identifier("elapsed_ticks")), spiked_then),
                       mark_due);
            Kernel_List_Node *loop_body = new_block();
            loop_body->children.push_back(check_tick);
            Kernel_List_Node *loop = new_node<List_Node>(Kernel_Node_Type::FOR, "");
            loop->children = {new_declaration(new_type("s32"), "elapsed_ticks", literal(std::to_string(shortest_delay - 1))),
                              new_expression("<=", identifier("elapsed_ticks"), literal(std::to_string(longest_delay - 1))),
                              new_assignment(identifier("elapsed_ticks"), "+=", literal("1")), loop_body};

            Kernel_List_Node *skip_body = new_block();
            skip_body->children.push_back(new_node<Unary_Node>(Kernel_Node_Type::JUMP, "return"));
            Kernel_List_Node *skip_walk = new_conditional();
            add_branch(skip_walk, new_unary_expression("!", identifier("arrival_due")), skip_body);

            insert_statements(body, static_cast<usize>(walk_stack_index),
                              {new_declaration(new_type("bool"), "arrival_due", identifier("false")), loop, skip_walk});
        }
    }

    // Constant for the run, so baked in rather than bound. Last, so the inserted dynamics'
    // uses are baked too.
    const Unique_Pointer<Kernel_Node> neuron_count(literal(std::to_string(context.simulation.total_neuron_count)));
    const Unique_Pointer<Kernel_Node> history_length(literal(std::to_string(spike_history_length(context))));
    bake_parameter(master_step, "neuron_count", neuron_count.get());
    bake_parameter(master_step, "spike_history_length", history_length.get());
    fold_constants(master_step->children[3]);
    return root;
}

namespace {

// Declares each derived variable once as derived_<name>, after the ones it reads. A select's
// value comes from selected_value.
void place_derived_variables(Kernel_List_Node *body, const Vector<String> &derived_names,
                             const Unordered_Map<String, Vector<const NML_Dynamics_Expression *>> &derived_entries,
                             const Function<Kernel_Node *(const String &)> &translate,
                             const Function<Kernel_Node *(const NML_Dynamics_Expression &)> &selected_value,
                             const Function<Runtime_Error(const String &)> &unsupported,
                             Set<String> &lems_names) {
    Unordered_Map<String, Vector<Kernel_Node *>> derived_statements;
    Unordered_Map<String, Set<String>> derived_dependencies;
    for (const String &name : derived_names) {
        const Vector<const NML_Dynamics_Expression *> &entries = derived_entries.at(name);
        Vector<Kernel_Node *> &statements = derived_statements[name];
        lems_names.insert(name);

        if (entries.front()->source_tag == NML_Declaration_Type::DerivedVariable) {
            const NML_Dynamics_Expression &entry = *entries.front();
            Kernel_Node *value = nullptr;
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
            Kernel_List_Node *cases = new_conditional();
            for (const NML_Dynamics_Expression *entry : entries) {
                Kernel_List_Node *case_body = new_block();
                case_body->children.push_back(
                        new_assignment(identifier(DERIVED_PREFIX + name), "=", translate(entry->expression)));
                add_branch(cases, entry->condition.empty() ? nullptr : translate(entry->condition), case_body);
            }
            statements.push_back(cases);
        }

        Set<String> used_names;
        for (const Kernel_Node *statement : statements) collect_identifiers(statement, used_names);
        for (const String &used_name : used_names) {
            if (used_name != name && derived_entries.count(used_name)) derived_dependencies[name].insert(used_name);
        }
    }

    Unordered_Map<String, s32> visit_state;
    Function<void(const String &)> place_derived = [&](const String &name) {
        if (visit_state[name] == 2) return;
        if (visit_state[name] == 1) throw unsupported("derived variable '" + name + "' depends on itself");
        visit_state[name] = 1;
        for (const String &dependency : derived_dependencies[name]) place_derived(dependency);
        visit_state[name] = 2;
        for (Kernel_Node *statement : derived_statements[name]) body->children.push_back(statement);
    };
    for (const String &name : derived_names) place_derived(name);
}

// Ticks after an arrival until every state variable is within REST_FRACTION of its largest
// deviation from rest, simulated on the host as the kernel integrates it. -1 when its OnStart
// values are not a rest state (a nonzero derivative, or a nonzero delivered_variable, which its
// targets read), its derived variables form a cycle, or it does not settle within the run.
s64 synapse_settle_ticks(const NML_Context &context, const NML_Component_Instance &synapse, f64 weight,
                         const String &delivered_variable) {
    const NML_Component_Type &component_type = *synapse.component_type;
    const String &owner_name = component_type.name;
    const Vector<String> &variable_names = component_type.state_variable_names;
    const f64 step_dt = context.simulation.step_dt;

    struct Parsed_Entry {
        String target;
        Unique_Pointer<Lems_Parse_Node> condition;
        Unique_Pointer<Lems_Parse_Node> value;
    };
    auto parse = [&owner_name](const String &expression) {
        return Unique_Pointer<Lems_Parse_Node>(expression.empty() ? nullptr : parse_lems_expression(expression, owner_name));
    };

    Vector<String> derived_names;
    Unordered_Map<String, Vector<Parsed_Entry>> derived_entries;
    Vector<Parsed_Entry> derivatives;
    Vector<Parsed_Entry> event_assignments;
    bool in_event = false;
    for (const NML_Dynamics_Expression &entry : component_type.dynamics) {
        switch (entry.source_tag) {
            case NML_Declaration_Type::DerivedVariable:
            case NML_Declaration_Type::Case:
                if (derived_entries.count(entry.target) == 0) derived_names.push_back(entry.target);
                derived_entries[entry.target].push_back({entry.target, parse(entry.condition), parse(entry.expression)});
                in_event = false;
                break;
            case NML_Declaration_Type::TimeDerivative:
                derivatives.push_back({entry.target, nullptr, parse(entry.expression)});
                in_event = false;
                break;
            case NML_Declaration_Type::OnEvent:
                in_event = true;
                break;
            case NML_Declaration_Type::StateAssignment:
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
            for (const Parsed_Entry &entry : derived_entries.at(name)) {
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

    Unordered_Map<String, f64> values = starting_values(context, synapse);
    values["weight"] = weight;
    values["t"] = 0.0;
    auto evaluate_derived = [&]() {
        for (const String &name : derived_order) {
            f64 value = 0.0;
            for (const Parsed_Entry &entry : derived_entries.at(name)) {
                if (entry.condition && evaluate_lems(entry.condition.get(), values, owner_name) == 0.0) continue;
                if (entry.value) value = evaluate_lems(entry.value.get(), values, owner_name);
                break;
            }
            values[name] = value;
        }
    };

    // At rest nothing may change and nothing may reach the target.
    const Unordered_Map<String, f64> rest = values;
    evaluate_derived();
    for (const Parsed_Entry &derivative : derivatives) {
        if (evaluate_lems(derivative.value.get(), values, owner_name) != 0.0) return -1;
    }
    if (values.count(delivered_variable) == 0 || values.at(delivered_variable) != 0.0) return -1;

    Vector<f64> largest_deviation(variable_names.size(), 0.0);
    const s64 tick_limit = std::max<s64>(1, std::llround(context.simulation.simulation_duration / step_dt));
    for (s64 tick = 0; tick < tick_limit; tick += 1) {
        values["t"] = static_cast<f64>(tick) * step_dt;
        evaluate_derived();
        Vector<f64> changes;
        for (const Parsed_Entry &derivative : derivatives) {
            changes.push_back(step_dt * evaluate_lems(derivative.value.get(), values, owner_name));
        }
        for (usize index = 0; index < derivatives.size(); index += 1) values[derivatives[index].target] += changes[index];
        if (tick == 0) {
            for (const Parsed_Entry &assignment : event_assignments) {
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

bool is_kernel_identifier_named(const Kernel_Node *node, const String &name) {
    return node && node->body.syntax_type == Kernel_Node_Type::IDENTIFIER && node->body.token == name;
}

// Puts a clone of difference in place of every `total - tracker` under node.
void replace_tracked_differences(Kernel_Node *&node, const Tracked_Total &tracked, const Kernel_Node *difference) {
    if (!node) return;
    if (node->body.syntax_type == Kernel_Node_Type::EXPRESSION && node->body.token == "-") {
        const auto *binary = dynamic_cast<const Kernel_Binary_Node *>(node);
        if (binary && is_kernel_identifier_named(binary->left, tracked.total_variable) &&
            is_kernel_identifier_named(binary->right, tracked.tracker_variable)) {
            delete node;
            node = clone(difference);
            return;
        }
    }
    for (Kernel_Node **slot : use_slots(node)) replace_tracked_differences(*slot, tracked, difference);
}

// Turns every `tracker = total` under node into `difference = 0`: the difference, the total less
// the tracker, is then 0. A kept tracker first takes up what it had not yet: `tracker += difference`.
void expand_tracker_assignments(Kernel_Node *node, const Tracked_Total &tracked, const Kernel_Node *difference) {
    if (!node) return;
    if (node->body.syntax_type == Kernel_Node_Type::BLOCK) {
        Vector<Kernel_Node *> &statements = static_cast<Kernel_List_Node *>(node)->children;
        for (usize index = 0; index < statements.size(); index += 1) {
            Kernel_Node *statement = statements[index];
            const auto *assignment = dynamic_cast<const Kernel_Binary_Node *>(statement);
            const bool catches_up = statement->body.syntax_type == Kernel_Node_Type::ASSIGNMENT && statement->body.token == "=" &&
                                    is_kernel_identifier_named(assignment->left, tracked.tracker_variable) &&
                                    is_kernel_identifier_named(assignment->right, tracked.total_variable);
            if (!catches_up) {
                expand_tracker_assignments(statement, tracked, difference);
                continue;
            }
            delete statement;
            statements[index] = new_assignment(clone(difference), "=", literal("0.0f"));
            if (tracked.tracker_kept) {
                statements.insert(statements.begin() + static_cast<s64>(index),
                                  new_assignment(identifier(tracked.tracker_variable), "+=", clone(difference)));
                index += 1;
            }
        }
        return;
    }
    for (Kernel_Node **slot : use_slots(node)) expand_tracker_assignments(*slot, tracked, difference);
}

} // namespace

// The cell reads its synapses and inputs through a select DerivedVariable (synapses[*]/i), and
// the engine sums them all into network_input. So the cell may have only one select, each must
// go to the attachment it reads, and each must expose what it reads, in the same dimension. A
// spike source's kicks only need a cell that reads its input.
void Codegen::check_cell_inputs() const {
    Vector<Pair<s64, const NML_Component_Type *>> population_ends;
    s64 first_neuron = 0;
    for (const NML_Component_Instance *population : network_populations(context)) {
        first_neuron += context.get_population_size(population);
        population_ends.push_back({first_neuron, population_cell(context, *population).component_type});
    }
    auto cell_type_of = [&](s64 neuron_index) -> const NML_Component_Type * {
        for (const auto &[population_end, cell_type] : population_ends) {
            if (neuron_index < population_end) return cell_type;
        }
        return nullptr;
    };

    // A synapse's spikes also reach a cell with an OnEvent, so that cell takes a synapse even
    // when it reads no current.
    auto check_attached = [](const NML_Component_Type &cell_type, const NML_Component_Instance &attached,
                             const String &role, bool must_expose_input, bool delivers_arrivals,
                             const String &destination) {
        const String described = role + " '" + attached.id + "' (" + attached.component_type->name + ") into a '" +
                                 cell_type.name + "'";

        // Every synapse and input lands in one network_input, which only one select can read.
        Vector<const NML_Dynamics_Expression *> selects;
        for (const NML_Dynamics_Expression &entry : cell_type.dynamics) {
            if (!entry.select.empty()) selects.push_back(&entry);
        }
        if (selects.size() > 1) {
            String select_names;
            for (const NML_Dynamics_Expression *entry : selects) {
                select_names += (select_names.empty() ? "" : ", ") + entry->target + " (" + entry->select + ")";
            }
            throw Runtime_Error(described + ": the cell reads its inputs through " + std::to_string(selects.size()) +
                                " selects, " + select_names + ", but the engine sums every synapse and input of a "
                                "cell into one current, so it cannot give each select its own");
        }

        bool reads_input = delivers_arrivals && has_on_event(cell_type);
        for (const NML_Dynamics_Expression *select : selects) {
            const NML_Dynamics_Expression &entry = *select;
            reads_input = true;
            // What goes to an attachment the select does not read never reaches the cell.
            const String attachment = destination.empty() ? "synapses" : destination;
            const String read_attachment = entry.select.substr(0, entry.select.find_first_of("[/"));
            if (read_attachment != attachment) {
                throw Runtime_Error(described + ": it goes to the cell's '" + attachment + "', but the cell reads only '" +
                                    read_attachment + "' (" + entry.target + ")");
            }
            if (!must_expose_input) continue;

            const String exposure_name = entry.select.substr(entry.select.rfind('/') + 1);
            const NML_Node *selector = cell_type.find_declaration("var:" + entry.target);
            const String read_dimension = selector ? selector->body.get_attribute_or("dimension", "none") : "none";
            const NML_Node *exposure = attached.component_type->find_declaration("exposure:" + exposure_name);
            if (!exposure) {
                throw Runtime_Error(described + ": it exposes no '" + exposure_name + "', which the cell reads as '" +
                                    entry.target + "'");
            }
            const String exposed_dimension = exposure->body.get_attribute_or("dimension", "none");
            if (exposed_dimension != read_dimension) {
                throw Runtime_Error(described + ": it exposes '" + exposure_name + "' as " + exposed_dimension +
                                    ", but the cell reads it as " + read_dimension);
            }
        }
        if (!reads_input) throw Runtime_Error(described + ": the cell reads no input");
    };

    // Each synapse against each cell type and destination it reaches, once.
    Unordered_Map<String, Ordered_Set<Pair<const NML_Component_Type *, String>>> checked_per_synapse;
    auto check_connection = [&](const NML_Network_Edge &edge) {
        const NML_Component_Type *cell_type = cell_type_of(edge.child);
        if (!cell_type || !checked_per_synapse[edge.component_id].insert({cell_type, edge.destination}).second) return;

        const NML_Component_Instance *synapse = context.find_instance(edge.component_id);
        if (!synapse || !synapse->component_type) return;
        check_attached(*cell_type, *synapse, "Synapse", true, true, edge.destination);
    };
    for (const Vector<NML_Network_Edge> &row : context.simulation.network_data.list) {
        for (const NML_Network_Edge &edge : row) check_connection(edge);
    }
    for (const NML_Network_Edge &edge : context.simulation.sampled_connections) check_connection(edge);

    for (const Simulation_Input_Config &profile : context.simulation.input_profiles) {
        const NML_Component_Instance *input = context.find_instance(profile.input_component_id);
        if (!input || !input->component_type) continue;

        Ordered_Set<Pair<const NML_Component_Type *, String>> checked;
        for (const Input_Target &target : profile.targets) {
            const NML_Component_Type *cell_type = cell_type_of(context.neuron_index_of(target.neuron_index));
            if (!cell_type || !checked.insert({cell_type, target.destination}).second) continue;
            // A spike source delivers kicks; any other input delivers the exposure the cell reads.
            check_attached(*cell_type, *input, "Input", !context.is_instance_of(input, "baseSpikeSource"), false,
                           target.destination);
        }
    }
}

// The type's dynamics with LEMS names still in them; translate_component_instance fills in each
// population's cell_state slots and parameter values. Order: on leaving refractory, the exit and
// OnEntry assignments; the OnEvents per arrival; derived variables; every derivative from the
// pre-step state; the Euler steps (each regime's only in it); then the OnConditions.
Kernel_List_Node *Codegen::translate_component_type(const NML_Component_Type &component_type) {
    auto cached = component_type_templates.find(component_type.name);
    if (cached != component_type_templates.end()) return cached->second;

    const String &owner_name = component_type.name;
    auto unsupported = [&owner_name](const String &reason) {
        return Runtime_Error("ComponentType '" + owner_name + "': " + reason);
    };

    struct Event_Handler {
        String regime_name;
        String test;
        Vector<const NML_Dynamics_Expression *> assignments;
        bool emits_spike = false;
        String transition;
    };

    // Sort the flat entries by kind.
    Vector<Pair<String, bool>> regimes;
    Vector<String> derived_names;
    Unordered_Map<String, Vector<const NML_Dynamics_Expression *>> derived_entries;
    Vector<const NML_Dynamics_Expression *> derivative_entries;
    Vector<Event_Handler> handlers;
    // OnEvents: each runs once per spike that reached the cell this tick, whatever its port.
    Vector<Event_Handler> arrival_handlers;
    Unordered_Map<String, Vector<const NML_Dynamics_Expression *>> entry_assignments;

    enum class Group { NONE, HANDLER, ARRIVAL, ENTRY };
    Group group = Group::NONE;
    for (const NML_Dynamics_Expression &entry : component_type.dynamics) {
        switch (entry.source_tag) {
            case NML_Declaration_Type::Regime:
                regimes.push_back({entry.target, entry.expression == "true"});
                group = Group::NONE;
                break;
            case NML_Declaration_Type::DerivedVariable:
            case NML_Declaration_Type::Case:
                if (derived_entries.count(entry.target) == 0) derived_names.push_back(entry.target);
                derived_entries[entry.target].push_back(&entry);
                group = Group::NONE;
                break;
            case NML_Declaration_Type::TimeDerivative:
                derivative_entries.push_back(&entry);
                group = Group::NONE;
                break;
            case NML_Declaration_Type::OnStart:
                group = Group::NONE;
                break;
            case NML_Declaration_Type::OnCondition:
                handlers.push_back({entry.regime_name, entry.expression, {}, false, ""});
                group = Group::HANDLER;
                break;
            case NML_Declaration_Type::OnEntry:
                group = Group::ENTRY;
                break;
            case NML_Declaration_Type::OnEvent:
                arrival_handlers.push_back({entry.regime_name, "", {}, false, ""});
                group = Group::ARRIVAL;
                break;
            case NML_Declaration_Type::StateAssignment:
                if (group == Group::HANDLER) handlers.back().assignments.push_back(&entry);
                else if (group == Group::ARRIVAL) arrival_handlers.back().assignments.push_back(&entry);
                else if (group == Group::ENTRY) entry_assignments[entry.regime_name].push_back(&entry);
                break;
            case NML_Declaration_Type::EventOut:
                if (group == Group::HANDLER) handlers.back().emits_spike = true;
                else if (group == Group::ARRIVAL) arrival_handlers.back().emits_spike = true;
                else throw unsupported("EventOut is only supported in an OnCondition or an OnEvent");
                break;
            case NML_Declaration_Type::Transition:
                if (group != Group::HANDLER) throw unsupported("Transition is only supported in an OnCondition");
                handlers.back().transition = entry.target;
                break;
            default:
                break;
        }
    }

    // The integrating / refractory regime pair is lowered to a test on last_spiked, which replaces
    // the timer and both Transitions. Each OnEntry runs with the Transition into its regime, and
    // the refractory regime's own dynamics run while the test holds.
    String active_regime;
    String refractory_regime;
    String timer_name;
    Unique_Pointer<Kernel_Node> refractory_duration;
    String refractory_comparison;
    const Event_Handler *exit_handler = nullptr;

    if (regimes.size() > 2) throw unsupported("only an integrating / refractory regime pair is supported");
    if (regimes.size() == 1) active_regime = regimes[0].first;
    if (regimes.size() == 2) {
        if (regimes[0].second == regimes[1].second) throw unsupported("exactly one regime must be initial");
        active_regime = regimes[0].second ? regimes[0].first : regimes[1].first;
        refractory_regime = regimes[0].second ? regimes[1].first : regimes[0].first;

        // The engine's spike is what enters refractory, so a spike from inside it would restart it.
        Vector<const Event_Handler *> exits;
        for (const Event_Handler &handler : handlers) {
            if (handler.regime_name != refractory_regime) continue;
            if (!handler.transition.empty()) exits.push_back(&handler);
            else if (handler.emits_spike) throw unsupported("an OnCondition in the refractory regime emits a spike");
        }
        if (exits.size() != 1 || exits[0]->transition != active_regime || exits[0]->emits_spike) {
            throw unsupported("the refractory regime must have exactly one OnCondition that transitions back, "
                              "without a spike");
        }
        exit_handler = exits[0];

        // The exit test must compare a timer to a duration with >= or >.
        Unique_Pointer<Lems_Parse_Node> exit_test(parse_lems_expression(exits[0]->test, owner_name));
        const auto *comparison = dynamic_cast<const Binary_Node<Lems_Parse_Body> *>(exit_test.get());
        const String comparison_operator = exit_test->body.token.lexeme;
        if (!comparison || exit_test->body.syntax_type != Lems_Node_Subtype::OPERATOR ||
            (comparison_operator != ">=" && comparison_operator != ">") ||
            comparison->left->body.syntax_type != Lems_Node_Subtype::IDENTIFIER) {
            throw unsupported("the refractory exit must test a timer against a duration");
        }
        timer_name = comparison->left->body.token.lexeme;
        // Refractory is the exit test's negation.
        refractory_comparison = comparison_operator == ">" ? "<=" : "<";

        // The test has to measure the time since the spike that entered the regime, which is what
        // last_spiked holds. Two forms do:
        //   timer .geq. duration      OnEntry sets timer = 0, TimeDerivative of timer is 1
        //   t .gt. stamp + duration   OnEntry sets stamp = t
        auto entry_assignment_of = [&](const String &variable_name) -> const NML_Dynamics_Expression * {
            for (const NML_Dynamics_Expression *assignment : entry_assignments[refractory_regime]) {
                if (assignment->target == variable_name) return assignment;
            }
            return nullptr;
        };
        String stamp_name;
        const Lems_Parse_Node *duration = comparison->right;
        if (timer_name == "t") {
            duration = nullptr;
            const auto *sum = dynamic_cast<const Binary_Node<Lems_Parse_Body> *>(comparison->right);
            if (sum && sum->body.syntax_type == Lems_Node_Subtype::OPERATOR && sum->body.token.lexeme == "+") {
                for (const auto &[stamp, candidate] : {Pair{sum->left, sum->right}, Pair{sum->right, sum->left}}) {
                    if (stamp->body.syntax_type != Lems_Node_Subtype::IDENTIFIER) continue;
                    const NML_Dynamics_Expression *stamp_entry = entry_assignment_of(stamp->body.token.lexeme);
                    if (!stamp_entry) continue;
                    Unique_Pointer<Lems_Parse_Node> stamp_value(parse_lems_expression(stamp_entry->expression, owner_name));
                    if (stamp_value->body.syntax_type != Lems_Node_Subtype::IDENTIFIER || stamp_value->body.token.lexeme != "t") continue;
                    stamp_name = stamp->body.token.lexeme;
                    duration = candidate;
                    break;
                }
            }
            if (!duration) throw unsupported("a refractory exit on t must be t .gt. stamp + duration, with OnEntry setting stamp = t");

            // The stamp has to hold the spike's time for the whole refractory period.
            for (const NML_Dynamics_Expression *derivative : derivative_entries) {
                if (derivative->target == stamp_name) throw unsupported("the refractory stamp '" + stamp_name + "' has a TimeDerivative");
            }
            for (const Event_Handler &handler : handlers) {
                for (const NML_Dynamics_Expression *assignment : handler.assignments) {
                    if (assignment->target == stamp_name) throw unsupported("the refractory stamp '" + stamp_name + "' is assigned in an OnCondition");
                }
            }
        } else {
            const NML_Dynamics_Expression *timer_entry = entry_assignment_of(timer_name);
            bool counts_time = false;
            for (const NML_Dynamics_Expression *derivative : derivative_entries) {
                if (derivative->regime_name == refractory_regime && derivative->target == timer_name &&
                    evaluate_lems(derivative->expression, {}, owner_name) == 1.0) {
                    counts_time = true;
                }
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
        // Entering refractory happens exactly when an active OnCondition transitions into it, so
        // the OnEntry assignments other than the timer's move into those handlers.
        for (Event_Handler &handler : handlers) {
            if (handler.regime_name != active_regime) continue;
            if (handler.transition.empty()) continue;
            if (handler.transition != refractory_regime) throw unsupported("unknown Transition '" + handler.transition + "'");
            for (const NML_Dynamics_Expression *assignment : entry_assignments[refractory_regime]) {
                if (assignment->target != timer_name) handler.assignments.push_back(assignment);
            }
        }
    }
    for (const Event_Handler &handler : handlers) {
        if (refractory_regime.empty() && !handler.transition.empty()) throw unsupported("Transition without a regime pair");
    }

    const bool has_refractory = !refractory_regime.empty();
    auto is_gated = [&](const String &regime_name) { return has_refractory && regime_name == active_regime; };

    // The OnEvents and the refractory exit run before the derived variables are computed, so
    // they cannot read one.
    auto refuse_derived_reads = [&](const Vector<const NML_Dynamics_Expression *> &assignments, const String &where) {
        for (const NML_Dynamics_Expression *assignment : assignments) {
            const Unique_Pointer<Lems_Parse_Node> tree(parse_lems_expression(assignment->expression, owner_name));
            Set<String> read_names;
            collect_lems_identifiers(tree.get(), read_names);
            for (const String &name : read_names) {
                if (derived_entries.count(name) == 0) continue;
                throw unsupported(where + " reads derived variable '" + name + "', which is computed after it runs");
            }
        }
    };
    for (const Event_Handler &handler : arrival_handlers) refuse_derived_reads(handler.assignments, "an OnEvent");
    if (exit_handler) refuse_derived_reads(exit_handler->assignments, "the refractory exit");
    if (has_refractory) refuse_derived_reads(entry_assignments[active_regime], "the OnEntry of '" + active_regime + "'");

    Set<String> lems_names;
    auto translate = [&](const String &expression) {
        const Unique_Pointer<Lems_Parse_Node> tree(parse_lems_expression(expression, owner_name));
        collect_lems_identifiers(tree.get(), lems_names);
        return translate_expression(tree.get());
    };
    const String step_dt = float_literal(context.simulation.step_dt);

    // Synapses and inputs sum their currents into network_input, which an add select reads.
    auto selected_value = [&unsupported](const NML_Dynamics_Expression &entry) -> Kernel_Node * {
        if (entry.reduce != "add") throw unsupported("DerivedVariable '" + entry.target + "' selects without reduce=\"add\"");
        return identifier("network_input");
    };
    Kernel_List_Node *body = new_block();

    // refractory = last_spiked >= 0 && (tick - last_spiked) * dt < duration   (<= for a .gt. exit)
    Kernel_Node *refractory_declaration = nullptr;
    Kernel_List_Node *leaving_refractory = nullptr;
    if (has_refractory) {
        auto elapsed_since_spike = [&](Kernel_Node *now) {
            Kernel_Node *last_spike = new_expression("[]", identifier("last_spiked"), identifier("neuron_index"));
            return time_of_ticks(new_expression("-", now, last_spike), context.simulation.step_dt);
        };
        Kernel_Node *has_spiked = new_expression(
                ">=", new_expression("[]", identifier("last_spiked"), identifier("neuron_index")), literal("0"));
        Kernel_Node *previous_duration = clone(refractory_duration.get());
        Kernel_Node *within_duration =
                new_expression(refractory_comparison, elapsed_since_spike(identifier("tick")), refractory_duration.release());
        refractory_declaration = new_declaration(
                new_type("const bool"), "refractory", new_expression("&&", has_spiked, within_duration));

        // Leaving refractory is the first tick the test fails after it held. The exit
        // OnCondition's assignments and the integrating regime's OnEntry run then.
        Vector<const NML_Dynamics_Expression *> exit_assignments = exit_handler->assignments;
        for (const NML_Dynamics_Expression *assignment : entry_assignments[active_regime]) exit_assignments.push_back(assignment);
        if (exit_assignments.empty()) {
            delete previous_duration;
        } else {
            Kernel_List_Node *exit_body = new_block();
            for (const NML_Dynamics_Expression *assignment : exit_assignments) {
                lems_names.insert(assignment->target);
                exit_body->children.push_back(
                        new_assignment(identifier(assignment->target), "=", translate(assignment->expression)));
            }
            Kernel_Node *previous_tick = new_expression("-", identifier("tick"), literal("1"));
            Kernel_Node *held_before = new_expression(refractory_comparison, elapsed_since_spike(previous_tick), previous_duration);
            Kernel_Node *leaving = new_expression(
                    "&&", new_expression("&&", clone(has_spiked), new_unary_expression("!", identifier("refractory"))),
                    held_before);
            leaving_refractory = new_conditional();
            add_branch(leaving_refractory, leaving, exit_body);
        }
    }

    // The OnEvents run once per arrival. One inside a regime runs only in it.
    Kernel_Node *arrival_loop = nullptr;
    bool arrivals_test_regime = false;
    if (!arrival_handlers.empty()) {
        Kernel_List_Node *per_arrival = new_block();
        for (const Event_Handler &handler : arrival_handlers) {
            Kernel_List_Node *handler_body = new_block();
            for (const NML_Dynamics_Expression *assignment : handler.assignments) {
                lems_names.insert(assignment->target);
                handler_body->children.push_back(
                        new_assignment(identifier(assignment->target), "=", translate(assignment->expression)));
            }
            if (handler.emits_spike) handler_body->children.push_back(new_assignment(identifier("spiked"), "=", identifier("true")));

            if (has_refractory && !handler.regime_name.empty()) {
                arrivals_test_regime = true;
                Kernel_Node *in_regime = handler.regime_name == refractory_regime
                                                ? identifier("refractory")
                                                : new_unary_expression("!", identifier("refractory"));
                Kernel_List_Node *gate = new_conditional();
                add_branch(gate, in_regime, handler_body);
                per_arrival->children.push_back(gate);
            } else {
                per_arrival->children.push_back(handler_body);
            }
        }
        Kernel_List_Node *loop = new_node<List_Node>(Kernel_Node_Type::FOR, "");
        loop->children = {new_declaration(new_type("s32"), "arrival", literal("0")),
                          new_expression("<", identifier("arrival"), identifier("event_arrivals")),
                          new_assignment(identifier("arrival"), "+=", literal("1")), per_arrival};
        arrival_loop = loop;
    }

    // Leaving refractory and the OnEvents come first, so derived variables and derivatives see
    // their changes; the refractory flag moves ahead of them when they test it.
    if (leaving_refractory || arrivals_test_regime) {
        body->children.push_back(refractory_declaration);
        refractory_declaration = nullptr;
    }
    if (leaving_refractory) body->children.push_back(leaving_refractory);
    if (arrival_loop) body->children.push_back(arrival_loop);

    place_derived_variables(body, derived_names, derived_entries, translate, selected_value, unsupported, lems_names);
    if (refractory_declaration) body->children.push_back(refractory_declaration);

    // Every derivative from the pre-step state, then every Euler step. The integrating regime's
    // steps run while the cell is not refractory and the refractory regime's while it is.
    Kernel_List_Node *gated = new_block();
    Kernel_List_Node *held = new_block();
    Vector<Kernel_Node *> updates;
    Set<String> integrated_names;
    Set<String> refractory_integrated_names;
    for (const NML_Dynamics_Expression *derivative : derivative_entries) {
        const bool in_refractory = has_refractory && derivative->regime_name == refractory_regime;
        // The refractory timer collapses into the refractory test.
        if (in_refractory && derivative->target == timer_name) continue;
        if (!(in_refractory ? refractory_integrated_names : integrated_names).insert(derivative->target).second) {
            throw unsupported("'" + derivative->target + "' has more than one TimeDerivative");
        }
        lems_names.insert(derivative->target);

        const String temporary = (in_refractory ? REFRACTORY_DERIVATIVE_PREFIX : DERIVATIVE_PREFIX) + derivative->target;
        body->children.push_back(new_declaration(new_type("const f32"), temporary, translate(derivative->expression)));
        Kernel_Node *update = new_assignment(identifier(derivative->target), "+=",
                                            new_expression("*", literal(step_dt), identifier(temporary)));
        if (in_refractory) held->children.push_back(update);
        else (is_gated(derivative->regime_name) ? gated->children : updates).push_back(update);
    }
    for (Kernel_Node *update : updates) body->children.push_back(update);

    // OnConditions, after every step. The refractory exit already ran on leaving.
    Vector<Kernel_Node *> conditions;
    for (const Event_Handler &handler : handlers) {
        const bool in_refractory = has_refractory && handler.regime_name == refractory_regime;
        if (in_refractory && !handler.transition.empty()) continue;

        Kernel_List_Node *handler_body = new_block();
        for (const NML_Dynamics_Expression *assignment : handler.assignments) {
            lems_names.insert(assignment->target);
            handler_body->children.push_back(
                    new_assignment(identifier(assignment->target), "=", translate(assignment->expression)));
        }
        if (handler.emits_spike) handler_body->children.push_back(new_assignment(identifier("spiked"), "=", identifier("true")));

        Kernel_List_Node *condition = new_conditional();
        add_branch(condition, translate(handler.test), handler_body);
        if (in_refractory) held->children.push_back(condition);
        else (is_gated(handler.regime_name) ? gated->children : conditions).push_back(condition);
    }

    if (gated->children.empty() && held->children.empty()) {
        delete gated;
        delete held;
    } else {
        Kernel_List_Node *by_regime = new_conditional();
        if (gated->children.empty()) {
            delete gated;
            add_branch(by_regime, identifier("refractory"), held);
        } else {
            add_branch(by_regime, new_unary_expression("!", identifier("refractory")), gated);
            if (held->children.empty()) delete held;
            else add_branch(by_regime, nullptr, held);
        }
        body->children.push_back(by_regime);
    }
    for (Kernel_Node *condition : conditions) body->children.push_back(condition);

    for (const String &name : lems_names) {
        if (ENGINE_NAMES.count(name) || starts_with(name, DERIVED_PREFIX) || starts_with(name, DERIVATIVE_PREFIX) ||
            starts_with(name, REFRACTORY_DERIVATIVE_PREFIX) || starts_with(name, RANDOM_SLOT_PREFIX)) {
            delete body;
            throw unsupported("the name '" + name + "' is reserved by the generated kernel");
        }
    }

    // A tracked total is never formed: its difference with the tracker is the neuron's running
    // sum, which translate_component_instance brings up to date before these statements run.
    auto tracked = tracked_totals.find(component_type.name);
    if (tracked != tracked_totals.end()) {
        const s64 total_index = find_statement(body, DERIVED_PREFIX + tracked->second.total_variable);
        if (total_index >= 0) {
            delete body->children[static_cast<usize>(total_index)];
            body->children.erase(body->children.begin() + total_index);
        }
        const Unique_Pointer<Kernel_Node> difference(own_input_slot(literal(std::to_string(RUNNING_SUM_ROW))));
        Kernel_Node *tracked_body = body;
        replace_tracked_differences(tracked_body, tracked->second, difference.get());
        expand_tracker_assignments(body, tracked->second, difference.get());

        Set<String> remaining_names;
        collect_identifiers(body, remaining_names);
        if (remaining_names.count(tracked->second.total_variable) ||
            (!tracked->second.tracker_kept && remaining_names.count(tracked->second.tracker_variable))) {
            delete body;
            throw unsupported("its total '" + tracked->second.total_variable + "' or tracker '" +
                              tracked->second.tracker_variable + "' is still read after the total was tracked");
        }
    }

    // Derived variables are read through their declared locals, and t is the tick's time from the host.
    Kernel_Node *body_node = body;
    for (const String &name : derived_names) {
        const Unique_Pointer<Kernel_Node> local(identifier(DERIVED_PREFIX + name));
        replace_identifier(body_node, name, local.get());
    }
    const Unique_Pointer<Kernel_Node> tick_time(identifier("simulation_time"));
    replace_identifier(body_node, "t", tick_time.get());

    component_type_templates[component_type.name] = body;
    return body;
}

// The type's template with each state variable replaced by its cell_state slot and each
// parameter by its value.
Kernel_List_Node *Codegen::translate_component_instance(const NML_Component_Instance &population, s64 first_neuron) {
    const NML_Component_Instance &cell = population_cell(context, population);
    const NML_Component_Type &component_type = *cell.component_type;
    Kernel_Node *body = clone(translate_component_type(component_type));

    try {
        // Each random() call gets a slot per neuron of the population, so every neuron draws
        // its own value: neuron n reads random_values[n + slot_start - first_neuron].
        const s64 population_size = context.get_population_size(&population);
        Set<String> template_names;
        collect_identifiers(body, template_names);
        for (const String &name : template_names) {
            if (!starts_with(name, RANDOM_SLOT_PREFIX)) continue;
            const s64 constant_part = random_values_count - first_neuron;
            const Unique_Pointer<Kernel_Node> slot(new_expression(constant_part < 0 ? "-" : "+", identifier("neuron_index"),
                                                                  literal(std::to_string(std::llabs(constant_part)))));
            replace_identifier(body, name, slot.get());
            random_values_count += population_size;
        }

        // Neuron n's variable is cell_state[n * count + base - first_neuron * count + offset].
        const Vector<String> &variable_names = component_type.state_variable_names;
        const s64 slot_count = NML_Context::cell_slot_count(component_type);
        const s64 population_base = context.simulation.population_base_indices.at(population.id);
        for (usize offset = 0; offset < variable_names.size(); offset += 1) {
            const s64 constant_part = population_base - first_neuron * slot_count + static_cast<s64>(offset);
            const Unique_Pointer<Kernel_Node> slot(new_expression(
                    "[]", identifier("cell_state"),
                    new_expression(constant_part < 0 ? "-" : "+",
                                   new_expression("*", identifier("neuron_index"), literal(std::to_string(slot_count))),
                                   literal(std::to_string(std::llabs(constant_part))))));
            replace_identifier(body, variable_names[offset], slot.get());
        }

        for (const auto &[name, value] : component_parameter_values(context, cell)) {
            const Unique_Pointer<Kernel_Node> value_literal(literal(float_literal(value)));
            replace_identifier(body, name, value_literal.get());
        }

        Set<String> remaining_names;
        collect_identifiers(body, remaining_names);
        for (const String &name : remaining_names) {
            if (ENGINE_NAMES.count(name) || starts_with(name, DERIVED_PREFIX) || starts_with(name, DERIVATIVE_PREFIX) ||
                starts_with(name, REFRACTORY_DERIVATIVE_PREFIX)) {
                continue;
            }
            throw Runtime_Error("ComponentType '" + component_type.name + "' (cell '" + cell.id + "'): '" + name +
                                "' is not a state variable, a set parameter, a constant or a derived variable");
        }

        // The weighted arrivals that reach the neuron this tick go into its running sum, which its
        // select reads with the tick's other input. For a tracked total the running sum is the
        // total less the tracker: it takes this tick's arrivals and what the inputs added, which is
        // all network_input holds then. The inputs run first, ahead of these.
        Vector<Kernel_Node *> arrivals;
        const bool total_tracked = tracked_totals.count(component_type.name) != 0;
        if (has_weighted_arrivals()) {
            Kernel_Node *arrival_slot = own_input_slot(
                    new_expression("+", literal(std::to_string(WEIGHTED_ARRIVAL_ROW)), identifier("current_row")));
            if (total_tracked) {
                arrivals.push_back(new_assignment(identifier("network_input"), "+=", arrival_slot));
                arrivals.push_back(new_assignment(clone(arrival_slot), "=", literal("0.0f")));
            } else {
                Kernel_Node *sum_slot = own_input_slot(literal(std::to_string(RUNNING_SUM_ROW)));
                arrivals.push_back(new_assignment(sum_slot, "+=", arrival_slot));
                arrivals.push_back(new_assignment(clone(arrival_slot), "=", literal("0.0f")));
                arrivals.push_back(new_assignment(identifier("network_input"), "+=", clone(sum_slot)));
            }
        }
        if (total_tracked) {
            arrivals.push_back(new_assignment(own_input_slot(literal(std::to_string(RUNNING_SUM_ROW))), "+=",
                                              identifier("network_input")));
        }
        insert_statements(static_cast<Kernel_List_Node *>(body), 0, arrivals);
    } catch (...) {
        delete body;
        throw;
    }
    return static_cast<Kernel_List_Node *>(body);
}

// Loops over this neuron's input entries, each running its prototype's code. Runs before the
// cell's dynamics, so the cell reads what its inputs added to network_input this tick.
Kernel_List_Node *Codegen::translate_input_entries(const NML_Component_Instance &population, s64 first_neuron) {
    const s64 population_size = context.get_population_size(&population);
    Vector<bool> reaches_population(input_prototypes.size(), false);
    bool any_entry = false;
    for (const Input_Entry &entry : input_entries) {
        if (entry.neuron_index < first_neuron || entry.neuron_index >= first_neuron + population_size) continue;
        reaches_population[static_cast<usize>(entry.prototype)] = true;
        any_entry = true;
    }
    if (!any_entry) return nullptr;

    Kernel_List_Node *dispatch = new_conditional();
    try {
        for (usize prototype = 0; prototype < input_prototypes.size(); prototype += 1) {
            if (!reaches_population[prototype]) continue;
            add_branch(dispatch, new_expression("==", identifier("input_prototype"), literal(std::to_string(prototype))),
                       translate_input_prototype(static_cast<s32>(prototype), population, first_neuron));
        }
    } catch (...) {
        delete dispatch;
        throw;
    }

    Kernel_List_Node *entry_body = new_block();
    entry_body->children.push_back(new_declaration(
            new_type("const s64"), "input_base",
            new_expression("*", identifier("input_entry"), literal(std::to_string(input_value_stride)))));
    entry_body->children.push_back(new_declaration(
            new_type("const f32"), "input_weight", new_expression("[]", identifier("input_values"), identifier("input_base"))));
    entry_body->children.push_back(new_declaration(
            new_type("const s32"), "input_prototype",
            new_expression("[]", identifier("input_entry_prototype"), identifier("input_entry"))));
    entry_body->children.push_back(dispatch);

    Kernel_List_Node *entry_loop = new_node<List_Node>(Kernel_Node_Type::FOR, "");
    entry_loop->children = {
            new_declaration(new_type("s64"), "input_entry",
                            new_expression("[]", identifier("input_row_start"), identifier("neuron_index"))),
            new_expression("<", identifier("input_entry"),
                           new_expression("[]", identifier("input_row_start"),
                                          new_expression("+", identifier("neuron_index"), literal("1")))),
            new_assignment(identifier("input_entry"), "+=", literal("1")), entry_body};
    Kernel_List_Node *block = new_block();
    block->children.push_back(entry_loop);
    return block;
}

// One input's dynamics for the current entry; each part's state is in the entry's block of
// input_values. Order: spike trains run their receiver's OnEvents once per spike; each part
// steps from the leaves up (derived, derivatives, Euler steps, OnConditions, exports); then the
// input adds its current, or a kick per unrouted spike, to network_input.
Kernel_List_Node *Codegen::translate_input_prototype(s32 prototype, const NML_Component_Instance &population,
                                                   s64 first_neuron) {
    const Input_Tree &tree = input_trees[static_cast<usize>(prototype)];
    const NML_Component_Instance &input = *input_prototypes[static_cast<usize>(prototype)];
    const NML_Component_Instance &cell = population_cell(context, population);
    const NML_Component_Type &cell_type = *cell.component_type;
    const String step_dt = float_literal(context.simulation.step_dt);
    const bool spike_source = context.is_instance_of(&input, "baseSpikeSource");
    auto unsupported = [&input](const String &reason) {
        return Runtime_Error("Input '" + input.id + "' (" + input.component_type->name + "): " + reason);
    };
    auto part_name = [&tree](s64 part_index) {
        const Input_Part &part = tree.parts[static_cast<usize>(part_index)];
        return "'" + part.instance->id + "' (" + part.instance->component_type->name + ")";
    };

    // A state variable of the target cell, in this population's layout.
    auto cell_slot = [&](const String &variable_name) -> Kernel_Node * {
        const Vector<String> &variable_names = cell_type.state_variable_names;
        auto variable = std::find(variable_names.begin(), variable_names.end(), variable_name);
        if (variable == variable_names.end()) return nullptr;
        const s64 slot_count = NML_Context::cell_slot_count(cell_type);
        const s64 constant_part = context.simulation.population_base_indices.at(population.id) -
                                  first_neuron * slot_count + static_cast<s64>(variable - variable_names.begin());
        return new_expression("[]", identifier("cell_state"),
                              new_expression(constant_part < 0 ? "-" : "+",
                                             new_expression("*", identifier("neuron_index"),
                                                            literal(std::to_string(slot_count))),
                                             literal(std::to_string(std::llabs(constant_part)))));
    };

    // What the cell reads from its inputs: the exposure its select names.
    String read_exposure;
    for (const NML_Dynamics_Expression &entry : cell_type.dynamics) {
        if (entry.select.empty()) continue;
        read_exposure = entry.select.substr(entry.select.rfind('/') + 1);
        break;
    }

    // A select is "role[*]/name" or "role/name".
    auto select_role = [](const String &select) { return select.substr(0, select.find_first_of("[/")); };
    auto select_name = [](const String &select) { return select.substr(select.rfind('/') + 1); };
    auto export_local = [](s64 part_index, const String &name) {
        return INPUT_PART_PREFIX + std::to_string(part_index) + "_" + name;
    };

    // The values each part's parent selects from it.
    Vector<Vector<String>> exports(tree.parts.size());
    auto add_export = [&](s64 part_index, const String &name) {
        Vector<String> &names = exports[static_cast<usize>(part_index)];
        if (std::find(names.begin(), names.end(), name) == names.end()) names.push_back(name);
    };
    for (s64 part_index = 0; part_index < static_cast<s64>(tree.parts.size()); part_index += 1) {
        for (const NML_Dynamics_Expression &entry : tree.parts[static_cast<usize>(part_index)].instance->component_type->dynamics) {
            if (entry.select.empty()) continue;
            for (s64 child = 0; child < static_cast<s64>(tree.parts.size()); child += 1) {
                const Input_Part &child_part = tree.parts[static_cast<usize>(child)];
                if (child_part.parent == part_index && child_part.role == select_role(entry.select)) {
                    add_export(child, select_name(entry.select));
                }
            }
        }
    }
    if (!spike_source) {
        if (read_exposure.empty()) throw unsupported("its target, a '" + cell_type.name + "', reads no input");
        add_export(0, read_exposure);
    }

    // A tracked total (see Tracked_Total) takes the input's increments of this tick, not its
    // lifetime value: the exposure only accumulates, so its assignments add to the export, which
    // starts at 0 each tick, and the input's own lifetime copy is never formed.
    String accumulating_variable;
    if (!spike_source && tracked_totals.count(cell_type.name)) {
        accumulating_variable = exposed_variable_name(*input.component_type, read_exposure);
    }

    auto translate = [&](const String &expression, const String &owner_name) {
        const Unique_Pointer<Lems_Parse_Node> tree_node(parse_lems_expression(expression, owner_name));
        return translate_expression(tree_node.get());
    };

    // A part's LEMS names, replaced with what they read for this entry: its state in input_values,
    // its parameters and properties as values (an input's own weight is the entry's), what it
    // requires from the target cell, the tick's time, and a random() slot per entry.
    auto resolve_part_names = [&](Kernel_Node *node, s64 part_index) {
        const Input_Part &part = tree.parts[static_cast<usize>(part_index)];
        const NML_Component_Type &component_type = *part.instance->component_type;
        if (part_index == 0 && !accumulating_variable.empty()) {
            const Unique_Pointer<Kernel_Node> increments(identifier(export_local(0, read_exposure)));
            replace_identifier(node, accumulating_variable, increments.get());
        }
        Set<String> used_names;
        collect_identifiers(node, used_names);

        const Vector<String> &variable_names = component_type.state_variable_names;
        for (usize offset = 0; offset < variable_names.size(); offset += 1) {
            if (!used_names.count(variable_names[offset])) continue;
            const Unique_Pointer<Kernel_Node> slot(new_expression(
                    "[]", identifier("input_values"),
                    new_expression("+", identifier("input_base"),
                                   literal(std::to_string(1 + part.first_state_slot + static_cast<s64>(offset))))));
            replace_identifier(node, variable_names[offset], slot.get());
        }
        for (const auto &[name, value] : component_parameter_values(context, *part.instance)) {
            if (!used_names.count(name)) continue;
            const Unique_Pointer<Kernel_Node> value_literal(literal(float_literal(value)));
            replace_identifier(node, name, value_literal.get());
        }
        for (const NML_Component_Type *type = &component_type; type; type = type->extends) {
            for (const NML_Node *element : children_of(type->source_node)) {
                const NML_Tag &tag = element->body;
                const String name = tag.get_attribute_or("name", "");
                if (name.empty() || !used_names.count(name)) continue;
                if (component_type.find_declaration(tag.namespace_key()) != element) continue;

                if (tag.tag_type == NML_Declaration_Type::Property) {
                    Unique_Pointer<Kernel_Node> value;
                    if (part_index == 0 && name == "weight") {
                        value.reset(identifier("input_weight"));
                    } else {
                        const String text_value = part.instance->has_value(name) ? part.instance->value_or(name)
                                                                                : tag.get_attribute_or("defaultValue", "0");
                        value.reset(literal(float_literal(context.resolve_quantity(text_value))));
                    }
                    replace_identifier(node, name, value.get());
                } else if (tag.tag_type == NML_Declaration_Type::Requirement) {
                    const Unique_Pointer<Kernel_Node> slot(cell_slot(name));
                    if (!slot) {
                        throw unsupported(part_name(part_index) + " needs '" + name + "' from its target, a '" +
                                          cell_type.name + "', which has no such state variable");
                    }
                    replace_identifier(node, name, slot.get());
                }
            }
        }
        if (used_names.count("t")) {
            const Unique_Pointer<Kernel_Node> tick_time(identifier("simulation_time"));
            replace_identifier(node, "t", tick_time.get());
        }
        for (const String &name : used_names) {
            if (!starts_with(name, RANDOM_SLOT_PREFIX)) continue;
            const Unique_Pointer<Kernel_Node> slot(new_expression(
                    "+", identifier("input_entry"), literal(std::to_string(random_values_count))));
            replace_identifier(node, name, slot.get());
            random_values_count += static_cast<s64>(input_entries.size());
        }
    };

    // The LEMS names a part may not use, since the generated code would capture them.
    auto check_part_names = [&](s64 part_index, const Set<String> &names) {
        for (const String &name : names) {
            if (INPUT_ENGINE_NAMES.count(name) || starts_with(name, INPUT_PART_PREFIX) || starts_with(name, DERIVED_PREFIX) ||
                starts_with(name, DERIVATIVE_PREFIX) || starts_with(name, RANDOM_SLOT_PREFIX)) {
                throw unsupported(part_name(part_index) + " uses the name '" + name + "', which the generated kernel reserves");
            }
        }
    };

    // An EventOut from source runs the OnEvents of the parts it is routed to, at once. A spike
    // source's own unrouted spike is a kick.
    Function<Kernel_List_Node *(s64, usize)> deliver_event;
    Function<Kernel_List_Node *(s64, usize)> on_event_body = [&](s64 part_index, usize depth) {
        const NML_Component_Type &component_type = *tree.parts[static_cast<usize>(part_index)].instance->component_type;
        Kernel_List_Node *body = new_block();
        bool in_event = false;
        bool emits_spike = false;
        // The LEMS names the handler uses, taken before translation turns random() into
        // random_values[random_slot_N].
        Set<String> used_names;
        for (const NML_Dynamics_Expression &entry : component_type.dynamics) {
            switch (entry.source_tag) {
                case NML_Declaration_Type::OnEvent:
                    in_event = true;
                    break;
                case NML_Declaration_Type::StateAssignment:
                    if (in_event) {
                        const Unique_Pointer<Lems_Parse_Node> tree_node(parse_lems_expression(entry.expression, component_type.name));
                        used_names.insert(entry.target);
                        collect_lems_identifiers(tree_node.get(), used_names);
                        body->children.push_back(new_assignment(identifier(entry.target), "=",
                                                                translate_expression(tree_node.get())));
                    }
                    break;
                case NML_Declaration_Type::EventOut:
                    if (in_event) emits_spike = true;
                    break;
                case NML_Declaration_Type::Transition:
                    if (in_event) throw unsupported(part_name(part_index) + " has a Transition in an OnEvent");
                    break;
                default:
                    in_event = false;
                    break;
            }
        }
        for (const NML_Dynamics_Expression &entry : component_type.dynamics) {
            if (entry.source_tag != NML_Declaration_Type::DerivedVariable && entry.source_tag != NML_Declaration_Type::Case) continue;
            if (used_names.count(entry.target)) {
                delete body;
                throw unsupported(part_name(part_index) + " reads derived variable '" + entry.target + "' in an OnEvent");
            }
        }
        check_part_names(part_index, used_names);
        resolve_part_names(body, part_index);
        if (emits_spike) {
            Kernel_List_Node *delivered = deliver_event(part_index, depth + 1);
            for (Kernel_Node *statement : delivered->children) body->children.push_back(statement);
            delivered->children.clear();
            delete delivered;
        }
        return body;
    };
    deliver_event = [&](s64 source, usize depth) {
        if (depth > tree.parts.size()) throw unsupported("its parts route events to each other in a loop");
        Kernel_List_Node *block = new_block();
        bool routed = false;
        for (const Input_Event_Route &route : tree.routes) {
            if (route.source != source) continue;
            routed = true;
            block->children.push_back(on_event_body(route.target, depth));
        }
        if (!routed && source == 0 && spike_source) {
            block->children.push_back(new_assignment(identifier("input_kicks"), "+=", literal("1")));
        }
        return block;
    };

    // One part's step for this tick.
    auto part_step = [&](s64 part_index) -> Kernel_List_Node * {
        const NML_Component_Type &component_type = *tree.parts[static_cast<usize>(part_index)].instance->component_type;
        const String &owner_name = component_type.name;

        struct Handler {
            String test;
            Vector<const NML_Dynamics_Expression *> assignments;
            bool emits_spike = false;
        };
        Vector<String> derived_names;
        Unordered_Map<String, Vector<const NML_Dynamics_Expression *>> derived_entries;
        Vector<const NML_Dynamics_Expression *> derivative_entries;
        Vector<Handler> handlers;
        enum class Group { NONE, CONDITION, OTHER };
        Group group = Group::NONE;
        for (const NML_Dynamics_Expression &entry : component_type.dynamics) {
            switch (entry.source_tag) {
                case NML_Declaration_Type::DerivedVariable:
                case NML_Declaration_Type::Case:
                    if (derived_entries.count(entry.target) == 0) derived_names.push_back(entry.target);
                    derived_entries[entry.target].push_back(&entry);
                    group = Group::NONE;
                    break;
                case NML_Declaration_Type::TimeDerivative:
                    derivative_entries.push_back(&entry);
                    group = Group::NONE;
                    break;
                case NML_Declaration_Type::OnCondition:
                    handlers.push_back({entry.expression, {}, false});
                    group = Group::CONDITION;
                    break;
                case NML_Declaration_Type::StateAssignment:
                    if (group == Group::CONDITION) handlers.back().assignments.push_back(&entry);
                    break;
                case NML_Declaration_Type::EventOut:
                    if (group == Group::CONDITION) handlers.back().emits_spike = true;
                    break;
                case NML_Declaration_Type::Regime:
                case NML_Declaration_Type::OnEntry:
                case NML_Declaration_Type::Transition:
                    throw unsupported(part_name(part_index) + " has regimes, which an input part cannot");
                default:
                    group = Group::OTHER;
                    break;
            }
        }

        Set<String> lems_names;
        for (const String &name : component_type.state_variable_names) lems_names.insert(name);
        auto translate_part = [&](const String &expression) {
            const Unique_Pointer<Lems_Parse_Node> tree_node(parse_lems_expression(expression, owner_name));
            collect_lems_identifiers(tree_node.get(), lems_names);
            return translate_expression(tree_node.get());
        };
        // A select reads what the matching child parts export: summed for reduce="add".
        auto selected_value = [&](const NML_Dynamics_Expression &entry) -> Kernel_Node * {
            const String role = select_role(entry.select);
            const String name = select_name(entry.select);
            Vector<s64> children;
            for (s64 child = 0; child < static_cast<s64>(tree.parts.size()); child += 1) {
                if (tree.parts[static_cast<usize>(child)].parent == part_index && tree.parts[static_cast<usize>(child)].role == role) children.push_back(child);
            }
            if (entry.reduce == "add") {
                if (children.empty()) return literal("0.0f");
                Kernel_Node *sum = identifier(export_local(children[0], name));
                for (usize index = 1; index < children.size(); index += 1) {
                    sum = new_expression("+", sum, identifier(export_local(children[index], name)));
                }
                return sum;
            }
            if (!entry.reduce.empty() || children.size() != 1) {
                throw unsupported(part_name(part_index) + " selects '" + entry.select + "', which needs reduce=\"add\" "
                                  "or exactly one part");
            }
            return identifier(export_local(children[0], name));
        };
        auto part_unsupported = [&](const String &reason) { return unsupported(part_name(part_index) + ": " + reason); };

        Kernel_List_Node *body = new_block();
        place_derived_variables(body, derived_names, derived_entries, translate_part, selected_value, part_unsupported,
                                lems_names);

        Vector<Kernel_Node *> updates;
        Set<String> integrated_names;
        for (const NML_Dynamics_Expression *derivative : derivative_entries) {
            if (!integrated_names.insert(derivative->target).second) {
                delete body;
                throw part_unsupported("'" + derivative->target + "' has more than one TimeDerivative");
            }
            body->children.push_back(new_declaration(
                    new_type("const f32"), DERIVATIVE_PREFIX + derivative->target, translate_part(derivative->expression)));
            updates.push_back(new_assignment(
                    identifier(derivative->target), "+=",
                    new_expression("*", literal(step_dt), identifier(DERIVATIVE_PREFIX + derivative->target))));
        }
        for (Kernel_Node *update : updates) body->children.push_back(update);

        for (const Handler &handler : handlers) {
            Kernel_List_Node *handler_body = new_block();
            for (const NML_Dynamics_Expression *assignment : handler.assignments) {
                handler_body->children.push_back(
                        new_assignment(identifier(assignment->target), "=", translate_part(assignment->expression)));
            }
            if (handler.emits_spike) handler_body->children.push_back(deliver_event(part_index, 0));
            Kernel_List_Node *condition = new_conditional();
            add_branch(condition, translate_part(handler.test), handler_body);
            body->children.push_back(condition);
        }

        for (const String &name : exports[static_cast<usize>(part_index)]) {
            lems_names.insert(name);
            // An accumulating exposure's increments already went into its export.
            if (part_index == 0 && name == read_exposure && !accumulating_variable.empty()) continue;
            body->children.push_back(new_assignment(identifier(export_local(part_index, name)), "=", identifier(name)));
        }

        check_part_names(part_index, lems_names);
        Kernel_Node *body_node = body;
        for (const String &name : derived_names) {
            const Unique_Pointer<Kernel_Node> local(identifier(DERIVED_PREFIX + name));
            replace_identifier(body_node, name, local.get());
        }
        resolve_part_names(body, part_index);
        return body;
    };

    Kernel_List_Node *block = new_block();
    try {
        if (spike_source) block->children.push_back(new_declaration(new_type("s32"), "input_kicks", literal("0")));
        for (s64 part_index = 0; part_index < static_cast<s64>(tree.parts.size()); part_index += 1) {
            for (const String &name : exports[static_cast<usize>(part_index)]) {
                block->children.push_back(new_declaration(new_type("f32"), export_local(part_index, name), literal("0.0f")));
            }
        }

        for (usize train = 0; train < tree.spike_trains.size(); train += 1) {
            Kernel_List_Node *train_block = new_block();
            train_block->children.push_back(new_declaration(
                    new_type("const s32"), "train_spikes",
                    new_cast(new_type("s32"),
                             new_expression("[]", identifier("input_spike_counts"),
                                            new_expression("+", literal(std::to_string(input_spike_train_offsets[static_cast<usize>(prototype)][train])),
                                                           identifier("tick"))))));
            Kernel_List_Node *per_spike = on_event_body(tree.spike_trains[train].receiver, 0);
            Kernel_List_Node *spike_loop = new_node<List_Node>(Kernel_Node_Type::FOR, "");
            spike_loop->children = {new_declaration(new_type("s32"), "arrival", literal("0")),
                                    new_expression("<", identifier("arrival"), identifier("train_spikes")),
                                    new_assignment(identifier("arrival"), "+=", literal("1")), per_spike};
            train_block->children.push_back(spike_loop);
            block->children.push_back(train_block);
        }

        for (s64 part_index = static_cast<s64>(tree.parts.size()) - 1; part_index >= 0; part_index -= 1) {
            block->children.push_back(part_step(part_index));
        }

        if (spike_source) {
            const f64 kick = input.has_value("amplitude") ? context.resolve_quantity(input.value_or("amplitude"))
                                                          : default_kick_amplitude(context, cell);
            block->children.push_back(new_assignment(
                    identifier("network_input"), "+=",
                    new_expression("*", new_cast(new_type("f32"), identifier("input_kicks")),
                                   new_expression("*", literal(float_literal(kick)), identifier("input_weight")))));
        } else {
            block->children.push_back(new_assignment(identifier("network_input"), "+=", identifier(export_local(0, read_exposure))));
        }
    } catch (...) {
        delete block;
        throw;
    }

    // Everything left is an engine name, an input local or a part's own local.
    Set<String> remaining_names;
    collect_identifiers(block, remaining_names);
    for (const String &name : remaining_names) {
        if (INPUT_ENGINE_NAMES.count(name) || starts_with(name, INPUT_PART_PREFIX) || starts_with(name, DERIVED_PREFIX) ||
            starts_with(name, DERIVATIVE_PREFIX)) {
            continue;
        }
        delete block;
        throw unsupported("'" + name + "' is not a state variable, a set parameter, a property, a constant or a derived "
                          "variable of any of its parts");
    }
    return block;
}

// The synapse's dynamics for one edge, with LEMS names still in them. Each state variable is a
// plane: reads are load_edge, and the tick's changes to it (its Euler step or return to rest, plus
// the OnEvent on arrival) sum into change_<name>, which one accumulate_edge adds, so an edge
// queues at most one update per plane per tick. Order: derived variables, derivatives, the step,
// the OnEvent on arrival, the updates, the delivered variable (its current) into the target. On the
// active window's last tick it is set exactly to rest instead of stepped, and stays there while skipped.
Kernel_List_Node *Codegen::translate_synapse_type(const NML_Component_Type &component_type,
                                                  const String &delivered_variable) {
    const String template_key = component_type.name + "/" + delivered_variable;
    auto cached = synapse_type_templates.find(template_key);
    if (cached != synapse_type_templates.end()) return cached->second;

    const String &owner_name = component_type.name;
    auto unsupported = [&owner_name](const String &reason) {
        return Runtime_Error("Synapse ComponentType '" + owner_name + "': " + reason);
    };

    const Vector<String> &variable_names = component_type.state_variable_names;
    auto plane_of = [&](const String &variable_name) -> s64 {
        auto variable = std::find(variable_names.begin(), variable_names.end(), variable_name);
        if (variable == variable_names.end()) throw unsupported("'" + variable_name + "' is not a state variable");
        return Weight_Matrix::FIRST_STATE_VARIABLE_PLANE + static_cast<s64>(variable - variable_names.begin());
    };

    // Sort the flat entries by kind.
    Vector<String> derived_names;
    Unordered_Map<String, Vector<const NML_Dynamics_Expression *>> derived_entries;
    Vector<const NML_Dynamics_Expression *> derivative_entries;
    Vector<const NML_Dynamics_Expression *> event_assignments;
    bool receives_spikes = false;
    bool in_event = false;
    for (const NML_Dynamics_Expression &entry : component_type.dynamics) {
        switch (entry.source_tag) {
            case NML_Declaration_Type::DerivedVariable:
            case NML_Declaration_Type::Case:
                if (derived_entries.count(entry.target) == 0) derived_names.push_back(entry.target);
                derived_entries[entry.target].push_back(&entry);
                in_event = false;
                break;
            case NML_Declaration_Type::TimeDerivative:
                derivative_entries.push_back(&entry);
                in_event = false;
                break;
            case NML_Declaration_Type::OnStart:
                in_event = false;
                break;
            case NML_Declaration_Type::OnEvent:
                if (entry.target != "in") throw unsupported("OnEvent on port '" + entry.target + "': a synapse only receives spikes on \"in\"");
                receives_spikes = true;
                in_event = true;
                break;
            case NML_Declaration_Type::StateAssignment:
                if (!in_event) throw unsupported("a StateAssignment outside OnStart and OnEvent");
                event_assignments.push_back(&entry);
                break;
            case NML_Declaration_Type::Regime:
            case NML_Declaration_Type::OnCondition:
            case NML_Declaration_Type::OnEntry:
            case NML_Declaration_Type::EventOut:
            case NML_Declaration_Type::Transition:
                throw unsupported("only OnEvent, TimeDerivative and DerivedVariable are supported in a synapse");
            default:
                break;
        }
    }
    if (!receives_spikes) throw unsupported("no OnEvent port=\"in\", so no spike ever drives it");
    const bool exposes_delivered = derived_entries.count(delivered_variable) != 0 ||
                                   std::find(variable_names.begin(), variable_names.end(), delivered_variable) !=
                                           variable_names.end();
    if (!exposes_delivered) throw unsupported("no '" + delivered_variable + "' to deliver to the target");

    Set<String> lems_names;
    auto translate = [&](const String &expression) {
        const Unique_Pointer<Lems_Parse_Node> tree(parse_lems_expression(expression, owner_name));
        if (calls_function(tree.get(), "random")) throw unsupported("random() is not supported in a synapse");
        collect_lems_identifiers(tree.get(), lems_names);
        return translate_expression(tree.get());
    };
    auto selected_value = [&unsupported](const NML_Dynamics_Expression &entry) -> Kernel_Node * {
        throw unsupported("DerivedVariable '" + entry.target + "' selects, which a synapse cannot");
    };
    const String step_dt = float_literal(context.simulation.step_dt);

    Kernel_List_Node *body = new_block();
    place_derived_variables(body, derived_names, derived_entries, translate, selected_value, unsupported, lems_names);
    // What the targets read, from the pre-step state like every derived variable: a state variable
    // read after its update would see it or not depending on whether the edge has an entry in S.
    body->children.push_back(new_declaration(new_type("const f32"), "delivered_value", identifier(delivered_variable)));

    // Every derivative from the pre-step state, then every Euler step.
    Set<String> integrated_names;
    Vector<Kernel_Node *> updates;
    for (const NML_Dynamics_Expression *derivative : derivative_entries) {
        if (!integrated_names.insert(derivative->target).second) {
            throw unsupported("'" + derivative->target + "' has more than one TimeDerivative");
        }
        lems_names.insert(derivative->target);
        body->children.push_back(new_declaration(
                new_type("const f32"), DERIVATIVE_PREFIX + derivative->target, translate(derivative->expression)));
        updates.push_back(new_assignment(
                identifier(CHANGE_PREFIX + derivative->target), "=",
                new_expression("*", literal(step_dt), identifier(DERIVATIVE_PREFIX + derivative->target))));
    }
    for (const String &variable_name : variable_names) {
        body->children.push_back(new_declaration(new_type("f32"), CHANGE_PREFIX + variable_name, literal(float_literal(0.0))));
    }
    Kernel_List_Node *steps = new_block();
    for (Kernel_Node *update : updates) steps->children.push_back(update);

    Kernel_List_Node *return_to_rest = new_block();
    for (const String &variable_name : variable_names) {
        lems_names.insert(variable_name);
        return_to_rest->children.push_back(new_assignment(
                identifier(CHANGE_PREFIX + variable_name), "=",
                new_expression("-", identifier(REST_PREFIX + variable_name), identifier(variable_name))));
    }
    Kernel_Node *last_active_tick = new_expression(
            "==",
            new_expression("-", identifier("tick"), new_expression("[]", identifier("last_spiked"), identifier("neuron_index"))),
            identifier("synapse_active_ticks"));
    Kernel_List_Node *advance = new_conditional();
    add_branch(advance, last_active_tick, return_to_rest);
    add_branch(advance, nullptr, steps);
    body->children.push_back(advance);

    // The OnEvent, as the change each assignment makes.
    Kernel_List_Node *on_arrival = new_block();
    for (const NML_Dynamics_Expression *assignment : event_assignments) {
        lems_names.insert(assignment->target);
        plane_of(assignment->target);
        Kernel_Node *increment = new_expression("-", translate(assignment->expression), identifier(assignment->target));
        on_arrival->children.push_back(new_assignment(identifier(CHANGE_PREFIX + assignment->target), "+=", increment));
    }
    Kernel_List_Node *arrival = new_conditional();
    add_branch(arrival, identifier("arrived"), on_arrival);
    body->children.push_back(arrival);

    // One update per plane; a change of zero queues nothing.
    for (const String &variable_name : variable_names) {
        body->children.push_back(accumulate_edge(plane_of(variable_name), identifier(CHANGE_PREFIX + variable_name)));
    }

    // What the targets read (the current), into the target's next input row.
    Kernel_Node *input_slot = new_expression(
            "[]", identifier("network_inputs"),
            new_expression("+", new_expression("*", identifier("next_row"), identifier("neuron_count")), identifier("target")));
    body->children.push_back(function_call("atomic_add", {input_slot, identifier("delivered_value")}));

    for (const String &name : lems_names) {
        if (is_synapse_engine_name(name)) {
            delete body;
            throw unsupported("the name '" + name + "' is reserved by the generated kernel");
        }
    }

    // Derived variables are read through their declared locals, and t is the tick's time from the host.
    Kernel_Node *body_node = body;
    for (const String &name : derived_names) {
        const Unique_Pointer<Kernel_Node> local(identifier(DERIVED_PREFIX + name));
        replace_identifier(body_node, name, local.get());
    }
    const Unique_Pointer<Kernel_Node> tick_time(identifier("simulation_time"));
    replace_identifier(body_node, "t", tick_time.get());

    synapse_type_templates[template_key] = body;
    return body;
}

Kernel_List_Node *Codegen::translate_synapse_instance(const NML_Component_Instance &synapse) {
    if (!synapse.component_type) throw Runtime_Error("Synapse '" + synapse.id + "' has no ComponentType");
    const NML_Component_Type &component_type = *synapse.component_type;
    const Synapse_Delivery &delivery = synapse_delivery(synapse);
    if (delivery.weighted_arrival) return translate_weighted_arrival(synapse);

    // A target with no select reads no value of it; its current goes on as before.
    const String delivered_variable = delivery.read_variable.empty() ? "i" : delivery.read_variable;
    Kernel_Node *body = clone(translate_synapse_type(component_type, delivered_variable));

    try {
        for (const auto &[name, value] : component_parameter_values(context, synapse)) {
            const Unique_Pointer<Kernel_Node> value_literal(literal(float_literal(value)));
            replace_identifier(body, name, value_literal.get());
        }

        // weight is the connection's, stored in the weight plane.
        const Unique_Pointer<Kernel_Node> edge_weight(load_edge(Weight_Matrix::WEIGHT_PLANE));
        replace_identifier(body, "weight", edge_weight.get());

        const Vector<String> &variable_names = component_type.state_variable_names;
        const Unordered_Map<String, f64> rest_values = starting_values(context, synapse);
        for (usize offset = 0; offset < variable_names.size(); offset += 1) {
            const Unique_Pointer<Kernel_Node> stored(load_edge(Weight_Matrix::FIRST_STATE_VARIABLE_PLANE + static_cast<s64>(offset)));
            replace_identifier(body, variable_names[offset], stored.get());
            const Unique_Pointer<Kernel_Node> rest_value(literal(float_literal(rest_values.at(variable_names[offset]))));
            replace_identifier(body, REST_PREFIX + variable_names[offset], rest_value.get());
        }

        Set<String> remaining_names;
        collect_identifiers(body, remaining_names);
        for (const String &name : remaining_names) {
            if (is_synapse_engine_name(name)) continue;
            throw Runtime_Error("Synapse ComponentType '" + component_type.name + "' ('" + synapse.id + "'): '" + name +
                                "' is not a state variable, a set parameter, a constant, a derived variable or weight");
        }
    } catch (...) {
        delete body;
        throw;
    }
    return static_cast<Kernel_List_Node *>(body);
}

// The spike that reaches the target on the next tick left the source a delay before it. Delivered
// into the next tick's row, the increment reaches the target's dynamics exactly a delay after the
// spike, as the connection's delay says: the scatter's one tick of latency is part of the delay.
// The delay is held to the network's declared range, which a fitted delay can pass by its fit error.
Kernel_List_Node *Codegen::translate_weighted_arrival(const NML_Component_Instance &synapse) {
    const NML_Component_Type &component_type = *synapse.component_type;
    const Synapse_Delivery &delivery = synapse_delivery(synapse);
    const String &owner_name = component_type.name;
    auto unsupported = [&](const String &reason) {
        return Runtime_Error("Synapse ComponentType '" + owner_name + "' ('" + synapse.id + "'): " + reason);
    };

    const Unique_Pointer<Lems_Parse_Node> value(parse_lems_expression(delivery.arrival_expression, owner_name));
    bool negated = false;
    const Lems_Parse_Node *increment_tree = accumulator_increment(value.get(), delivery.read_variable, negated);
    if (!increment_tree) throw unsupported("'" + delivery.arrival_expression + "' is not an accumulating OnEvent value");
    Set<String> lems_names;
    collect_lems_identifiers(increment_tree, lems_names);
    for (const String &name : lems_names) {
        if (is_synapse_engine_name(name)) throw unsupported("the name '" + name + "' is reserved by the generated kernel");
    }

    Kernel_Node *increment = translate_expression(increment_tree);
    if (negated) increment = new_unary_expression("-", increment);
    Kernel_List_Node *arrival_body = new_block();
    Kernel_List_Node *block = new_block();
    try {
        for (const auto &[name, parameter_value] : component_parameter_values(context, synapse)) {
            const Unique_Pointer<Kernel_Node> value_literal(literal(float_literal(parameter_value)));
            replace_identifier(increment, name, value_literal.get());
        }
        // weight is the connection's, in the weight plane; t is the arrival's, the next tick's time.
        const Unique_Pointer<Kernel_Node> edge_weight(load_edge(Weight_Matrix::WEIGHT_PLANE));
        replace_identifier(increment, "weight", edge_weight.get());
        const Unique_Pointer<Kernel_Node> arrival_time(
                new_expression("+", identifier("simulation_time"), literal(float_literal(context.simulation.step_dt))));
        replace_identifier(increment, "t", arrival_time.get());
        fold_constants(increment);

        Set<String> remaining_names;
        collect_identifiers(increment, remaining_names);
        for (const String &name : remaining_names) {
            if (is_synapse_engine_name(name)) continue;
            throw unsupported("its OnEvent increment reads '" + name + "', which is not a set parameter, a constant or weight");
        }

        const s64 shortest_delay = std::max<s64>(1, context.simulation.minimum_edge_delay);
        const s64 longest_delay = std::max<s64>(1, context.simulation.maximum_edge_delay);
        Kernel_Trinary_Node *held_below_longest = new_node<Trinary_Node>(Kernel_Node_Type::EXPRESSION, "?:");
        held_below_longest->left = new_expression(">", identifier("delay_ticks"), literal(std::to_string(longest_delay)));
        held_below_longest->middle = literal(std::to_string(longest_delay));
        held_below_longest->right = identifier("delay_ticks");
        Kernel_Trinary_Node *held_in_range = new_node<Trinary_Node>(Kernel_Node_Type::EXPRESSION, "?:");
        held_in_range->left = new_expression("<", identifier("delay_ticks"), literal(std::to_string(shortest_delay)));
        held_in_range->middle = literal(std::to_string(shortest_delay));
        held_in_range->right = held_below_longest;
        block->children.push_back(new_declaration(new_type("const s32"), "arrival_delay_ticks", held_in_range));

        // arrives_next_tick = tick + 1 >= delay && spike_history[((tick + 1 - delay) % length) * count + source] != 0
        Kernel_Node *next_tick = new_expression("+", identifier("tick"), literal("1"));
        Kernel_Node *spike_row = new_expression(
                "%", new_expression("-", clone(next_tick), identifier("arrival_delay_ticks")), identifier("spike_history_length"));
        Kernel_Node *spiked_then = new_expression(
                "!=",
                new_expression("[]", identifier("spike_history"),
                               new_expression("+", new_expression("*", spike_row, identifier("neuron_count")),
                                              identifier("neuron_index"))),
                literal("0"));
        block->children.push_back(new_declaration(
                new_type("const bool"), "arrives_next_tick",
                new_expression("&&", new_expression(">=", next_tick, identifier("arrival_delay_ticks")), spiked_then)));

        Kernel_Node *arrival_slot = new_expression(
                "[]", identifier("network_inputs"),
                new_expression("+",
                               new_expression("*",
                                              new_expression("+", literal(std::to_string(WEIGHTED_ARRIVAL_ROW)),
                                                             identifier("next_row")),
                                              identifier("neuron_count")),
                               identifier("target")));
        arrival_body->children.push_back(function_call("atomic_add", {arrival_slot, increment}));
        increment = nullptr;
        Kernel_List_Node *on_arrival = new_conditional();
        add_branch(on_arrival, identifier("arrives_next_tick"), arrival_body);
        arrival_body = nullptr;
        block->children.push_back(on_arrival);
    } catch (...) {
        delete increment;
        delete arrival_body;
        delete block;
        throw;
    }
    return block;
}

s64 Codegen::edge_plane_count() const {
    s64 state_variable_count = 0;
    for (const NML_Component_Instance &synapse : context.simulation.synapse_instances) {
        if (!synapse.component_type) throw Runtime_Error("Synapse '" + synapse.id + "' has no ComponentType");
        state_variable_count = std::max<s64>(state_variable_count, synapse_delivery(synapse).edge_state_variable_names.size());
    }
    return Weight_Matrix::FIRST_STATE_VARIABLE_PLANE + state_variable_count;
}

s64 Codegen::updated_edge_plane_count() const {
    // A synapse's dynamics write every one of its StateVariables kept per connection: the Euler
    // steps, the OnEvent and the return to rest.
    return edge_plane_count() - Weight_Matrix::FIRST_STATE_VARIABLE_PLANE;
}

s64 Codegen::synapse_active_ticks() const {
    Unordered_Map<String, f64> largest_weight = context.simulation.largest_connection_weight_per_synapse;
    const auto include = [&](const NML_Network_Edge &edge) {
        f64 &largest = largest_weight[edge.component_id];
        largest = std::max(largest, std::fabs(static_cast<f64>(edge.weight)));
    };
    for (const Vector<NML_Network_Edge> &row : context.simulation.network_data.list) {
        for (const NML_Network_Edge &edge : row) include(edge);
    }
    for (const NML_Network_Edge &edge : context.simulation.sampled_connections) include(edge);

    // A spike's last weighted arrival is sent the longest delay less one tick after it.
    if (only_weighted_arrivals()) {
        const s64 active_ticks = std::max<s64>(1, context.simulation.maximum_edge_delay) - 1;
        log::logger().info("Codegen: a neuron's synapses run for {} ticks after it spikes, only to send weighted "
                           "arrivals", active_ticks);
        return active_ticks;
    }

    // At least a tick, so no arrival ever lands on the tick that puts a synapse back at rest.
    s64 settle_ticks = 1;
    for (const NML_Component_Instance &synapse : context.simulation.synapse_instances) {
        auto weight = largest_weight.find(synapse.id);
        if (weight == largest_weight.end()) continue;
        // A weighted arrival keeps nothing to settle.
        const Synapse_Delivery &delivery = synapse_delivery(synapse);
        if (delivery.weighted_arrival) continue;

        s64 ticks = -1;
        try {
            ticks = synapse_settle_ticks(context, synapse, weight->second,
                                         delivery.read_variable.empty() ? "i" : delivery.read_variable);
        } catch (const Runtime_Error &error) {
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

Kernel_Node *Codegen::translate_expression(const Lems_Parse_Node *node) {
    const String &lexeme = node->body.token.lexeme;
    switch (node->body.syntax_type) {
        case Lems_Node_Subtype::FLOAT:
        case Lems_Node_Subtype::INT:
            // Always a float, so 1/2 is never integer division.
            return literal(float_literal(std::stod(lexeme)));

        case Lems_Node_Subtype::IDENTIFIER:
            return identifier(lexeme);

        case Lems_Node_Subtype::OPERATOR: {
            if (const auto *unary = dynamic_cast<const Unary_Node<Lems_Parse_Body> *>(node)) {
                return new_unary_expression(lexeme, translate_expression(unary->child));
            }
            const auto *binary = static_cast<const Binary_Node<Lems_Parse_Body> *>(node);
            Kernel_Node *left = translate_expression(binary->left);
            return new_expression(lexeme, left, translate_expression(binary->right));
        }

        case Lems_Node_Subtype::FUNCTION_CALL: {
            // random(x) is x times this tick's uniform draw in the call's own random_values slot.
            if (lexeme == "random") {
                const Vector<Lems_Parse_Node *> &arguments = children_of(static_cast<const Binary_Node<Lems_Parse_Body> *>(node)->right);
                if (arguments.size() != 1) throw Runtime_Error("LEMS random takes one argument");
                Kernel_Node *slot = identifier(RANDOM_SLOT_PREFIX + std::to_string(random_call_count));
                random_call_count += 1;
                return new_expression("*", translate_expression(arguments[0]),
                                      new_expression("[]", identifier("random_values"), slot));
            }

            auto function = LEMS_FUNCTIONS.find(lexeme);
            if (function == LEMS_FUNCTIONS.end()) throw Runtime_Error("LEMS function '" + lexeme + "' has no kernel equivalent");

            Kernel_List_Node *call = new_node<List_Node>(Kernel_Node_Type::FUNCTION_CALL, function->second);
            for (const Lems_Parse_Node *argument : children_of(static_cast<const Binary_Node<Lems_Parse_Body> *>(node)->right)) {
                call->children.push_back(translate_expression(argument));
            }
            return call;
        }

        default:
            throw Runtime_Error("Malformed LEMS expression '" + lexeme + "'");
    }
}

Kernel_Node *Codegen::translate_lems(const String &expression, const String &owner_name) {
    Unique_Pointer<Lems_Parse_Node> tree(parse_lems_expression(expression, owner_name));
    return translate_expression(tree.get());
}

void Codegen::bake_kernel_constant(const String &name, s64 value) {
    Kernel_List_Node *master_step = root ? find_function(root, "master_step") : nullptr;
    if (!master_step) throw Runtime_Error("bake_kernel_constant: there is no master_step until translate_kernel_code runs");
    const Unique_Pointer<Kernel_Node> value_literal(literal(std::to_string(value)));
    bake_parameter(master_step, name, value_literal.get());
}

Vector<String> Codegen::kernel_parameter_names() const {
    Vector<String> names;
    Kernel_List_Node *master_step = root ? find_function(root, "master_step") : nullptr;
    if (!master_step) return names;

    for (Kernel_Node *parameter_node : static_cast<Kernel_List_Node *>(master_step->children[2])->children) {
        Kernel_Trinary_Node *parameter = static_cast<Kernel_Trinary_Node *>(parameter_node);
        if (parameter->right) continue;
        names.push_back(parameter->middle->body.token);
    }
    return names;
}

// Code generation

const Unordered_Map<String, String> &kernel_type_spellings(Kernel_Backend backend) {
    static const Unordered_Map<String, String> metal_spellings = {
        {"void", "void"}, {"bool", "bool"}, {"u8", "uchar"}, {"u16", "ushort"}, {"s32", "int"},
        {"u32", "uint"}, {"s64", "long"}, {"f32", "float"}};
    static const Unordered_Map<String, String> cuda_spellings = {
        {"void", "void"}, {"bool", "bool"}, {"u8", "unsigned char"}, {"u16", "unsigned short"}, {"s32", "int"},
        {"u32", "unsigned int"}, {"s64", "long long"}, {"f32", "float"}, {"f64", "double"}};
    return backend == Kernel_Backend::METAL ? metal_spellings : cuda_spellings;
}

String Codegen::compile() {
    return compile(root);
}

String Codegen::compile(Kernel_Node *node) {
    switch (node->body.syntax_type) {
        case Kernel_Node_Type::LITERAL:              return node->body.token;
        case Kernel_Node_Type::IDENTIFIER:           return node->body.token;
        case Kernel_Node_Type::EXPRESSION:           return expression_to_string(node);
        case Kernel_Node_Type::BLOCK:                return block_to_string(node);
        case Kernel_Node_Type::DEVICE_FUNCTION_IMPL: return device_function_impl_to_string(node);
        case Kernel_Node_Type::FUNCTION_CALL:        return function_call_to_string(node);
        case Kernel_Node_Type::KERNEL_FUNCTION_IMPL: return kernel_function_impl_to_string(node);
        case Kernel_Node_Type::DECLARATION:          return declaration_to_string(node);
        case Kernel_Node_Type::ASSIGNMENT:           return assignment_to_string(node);
        case Kernel_Node_Type::CONDITIONAL:          return conditional_to_string(node);
        case Kernel_Node_Type::CAST:                 return cast_to_string(node);
        case Kernel_Node_Type::FOR:                  return for_to_string(node);
        case Kernel_Node_Type::WHILE:                return while_to_string(node);
        case Kernel_Node_Type::JUMP:                 return jump_to_string(node);
        case Kernel_Node_Type::ATTRIBUTE:            return attribute_to_string(node);
        case Kernel_Node_Type::PROGRAM:              return program_to_string(node);
        case Kernel_Node_Type::TYPE:                 return type_to_string(node->body.token);
        case Kernel_Node_Type::POINTER:              return pointer_to_string(node);
        case Kernel_Node_Type::REFERENCE:            return compile(static_cast<Kernel_Unary_Node *>(node)->child) + " &";
        case Kernel_Node_Type::INCLUDE:              return "#include " + node->body.token;
        case Kernel_Node_Type::DEFINE:               return define_to_string(node);
        case Kernel_Node_Type::USING_NAMESPACE:      return "using namespace " + node->body.token + ";";
        case Kernel_Node_Type::CONDITIONAL_BRANCH:
        case Kernel_Node_Type::PARAMETER:
        case Kernel_Node_Type::PARAMETER_LIST:       break;
    }
    throw Runtime_Error("Kernel node cannot be emitted on its own");
}

String Codegen::compile_or_empty(Kernel_Node *node) {
    return node ? compile_unparenthesized(node) : String();
}

String Codegen::compile_unparenthesized(Kernel_Node *node) {
    if (node->body.syntax_type == Kernel_Node_Type::EXPRESSION) return expression_to_string(node, false);
    return compile(node);
}

// Words that are neutral type names are mapped; qualifiers and platform-only types pass through.
String Codegen::type_to_string(const String &type) const {
    const Unordered_Map<String, String> &spellings = kernel_type_spellings(backend);
    Input_String_Stream words(type);
    String word;
    String source;
    while (words >> word) {
        auto spelling = spellings.find(word);
        if (!source.empty()) source += " ";
        source += spelling == spellings.end() ? word : spelling->second;
    }
    return source;
}

String Codegen::pointer_to_string(Kernel_Node *node) {
    Kernel_Unary_Node *pointer = static_cast<Kernel_Unary_Node *>(node);
    String source = compile(pointer->child) + " *";
    if (!pointer->body.token.empty()) source += " " + pointer->body.token;
    return source;
}

String Codegen::define_to_string(Kernel_Node *node) {
    Kernel_Binary_Node *define = static_cast<Kernel_Binary_Node *>(node);
    String source = "#define " + compile(define->left);
    if (define->right) source += " " + compile(define->right);
    return source;
}

// Directives and using lines one per line; functions separated by a blank line.
String Codegen::program_to_string(Kernel_Node *node) {
    String source;
    for (Kernel_Node *item : static_cast<Kernel_List_Node *>(node)->children) {
        const Kernel_Node_Type type = item->body.syntax_type;
        const bool is_function = type == Kernel_Node_Type::DEVICE_FUNCTION_IMPL || type == Kernel_Node_Type::KERNEL_FUNCTION_IMPL;
        source += is_function ? "\n" + compile(item) + "\n" : compile(item) + "\n";
    }
    return source;
}

String Codegen::attribute_to_string(Kernel_Node *node) {
    if (backend != Kernel_Backend::METAL) throw Runtime_Error("Attribute '" + node->body.token + "' has no CUDA equivalent");
    return "[[ " + node->body.token + " ]]";
}

// The tree is the precedence, so operators are fully parenthesized unless the place the
// expression sits already delimits it.
String Codegen::expression_to_string(Kernel_Node *node, bool parenthesize) {
    const String &expression_operator = node->body.token;
    const String open = parenthesize ? "(" : "";
    const String close = parenthesize ? ")" : "";

    if (Kernel_Binary_Node *binary = dynamic_cast<Kernel_Binary_Node *>(node)) {
        if (expression_operator == "[]") return compile(binary->left) + "[" + compile_unparenthesized(binary->right) + "]";
        if (expression_operator == "." || expression_operator == "->")
            return compile(binary->left) + expression_operator + compile(binary->right);
        return open + compile(binary->left) + " " + expression_operator + " " + compile(binary->right) + close;
    }
    if (Kernel_Unary_Node *unary = dynamic_cast<Kernel_Unary_Node *>(node))
        return open + expression_operator + compile(unary->child) + close;
    if (Kernel_Trinary_Node *ternary = dynamic_cast<Kernel_Trinary_Node *>(node))
        return open + compile(ternary->left) + " ? " + compile(ternary->middle) + " : " + compile(ternary->right) + close;
    throw Runtime_Error("EXPRESSION node has no operands");
}

String Codegen::cast_to_string(Kernel_Node *node) {
    Kernel_Binary_Node *cast = static_cast<Kernel_Binary_Node *>(node);
    return "((" + compile(cast->left) + ")" + compile(cast->right) + ")";
}

// Neutral names that differ per backend are mapped; every other name passes through.
String Codegen::function_call_to_string(Kernel_Node *node) {
    const Vector<Kernel_Node *> &arguments = static_cast<Kernel_List_Node *>(node)->children;
    const String &name = node->body.token;

    // atomic_add(target, value): target is the slot itself, not its address.
    if (name == "atomic_add") {
        if (arguments.size() != 2) throw Runtime_Error("atomic_add takes a target and a value");
        if (backend == Kernel_Backend::METAL) {
            return "atomic_fetch_add_explicit((device atomic_float *)&(" + compile_unparenthesized(arguments[0]) + "), " +
                   compile_unparenthesized(arguments[1]) + ", memory_order_relaxed)";
        }
        return "atomicAdd(&(" + compile_unparenthesized(arguments[0]) + "), " + compile_unparenthesized(arguments[1]) + ")";
    }

    // exact_divide(numerator, denominator): a correctly rounded division even under fast math.
    if (name == "exact_divide") {
        if (arguments.size() != 2) throw Runtime_Error("exact_divide takes a numerator and a denominator");
        const String numerator = compile_unparenthesized(arguments[0]);
        const String denominator = compile_unparenthesized(arguments[1]);
        if (backend == Kernel_Backend::METAL) return "precise::divide(" + numerator + ", " + denominator + ")";
        return "__fdiv_rn(" + numerator + ", " + denominator + ")";
    }

    // atomic_increment(target): adds 1 to a u32 slot.
    if (name == "atomic_increment") {
        if (arguments.size() != 1) throw Runtime_Error("atomic_increment takes a target");
        if (backend == Kernel_Backend::METAL) {
            return "atomic_fetch_add_explicit((device atomic_uint *)&(" + compile_unparenthesized(arguments[0]) +
                   "), 1u, memory_order_relaxed)";
        }
        return "atomicAdd(&(" + compile_unparenthesized(arguments[0]) + "), 1u)";
    }

    static const Unordered_Map<String, String> function_spellings = {{"heaviside", "spikecorec_heaviside"}};
    auto spelling = function_spellings.find(name);

    String source = (spelling == function_spellings.end() ? name : spelling->second) + "(";
    for (usize index = 0; index < arguments.size(); index += 1) {
        if (index > 0) source += ", ";
        source += compile_unparenthesized(arguments[index]);
    }
    return source + ")";
}

// The block adds the semicolons, so declarations and assignments can be reused in loop headers.
String Codegen::block_to_string(Kernel_Node *node) {
    static const Set<Kernel_Node_Type> compound_statements = {
        Kernel_Node_Type::BLOCK, Kernel_Node_Type::CONDITIONAL, Kernel_Node_Type::FOR, Kernel_Node_Type::WHILE,
        Kernel_Node_Type::DEVICE_FUNCTION_IMPL, Kernel_Node_Type::KERNEL_FUNCTION_IMPL};

    String source = "{\n";
    indentation_depth += 1;
    for (Kernel_Node *statement : static_cast<Kernel_List_Node *>(node)->children) {
        source += String(4 * indentation_depth, ' ') + compile_unparenthesized(statement);
        if (compound_statements.count(statement->body.syntax_type) == 0) source += ";";
        source += "\n";
    }
    indentation_depth -= 1;
    return source + String(4 * indentation_depth, ' ') + "}";
}

String Codegen::declaration_to_string(Kernel_Node *node) {
    Kernel_Trinary_Node *declaration = static_cast<Kernel_Trinary_Node *>(node);
    String source = compile(declaration->left) + " " + compile(declaration->middle);
    if (declaration->right) source += " = " + compile_unparenthesized(declaration->right);
    return source;
}

String Codegen::assignment_to_string(Kernel_Node *node) {
    Kernel_Binary_Node *assignment = static_cast<Kernel_Binary_Node *>(node);
    return compile(assignment->left) + " " + assignment->body.token + " " + compile_unparenthesized(assignment->right);
}

String Codegen::conditional_to_string(Kernel_Node *node) {
    String source;
    for (Kernel_Node *branch_node : static_cast<Kernel_List_Node *>(node)->children) {
        Kernel_Binary_Node *branch = static_cast<Kernel_Binary_Node *>(branch_node);
        if (!source.empty()) source += " else ";
        if (branch->left) source += "if (" + compile_unparenthesized(branch->left) + ") ";
        source += compile(branch->right);
    }
    return source;
}

String Codegen::for_to_string(Kernel_Node *node) {
    const Vector<Kernel_Node *> &parts = static_cast<Kernel_List_Node *>(node)->children;
    return "for (" + compile_or_empty(parts[0]) + "; " + compile_or_empty(parts[1]) + "; " +
           compile_or_empty(parts[2]) + ") " + compile(parts[3]);
}

String Codegen::while_to_string(Kernel_Node *node) {
    Kernel_Binary_Node *loop = static_cast<Kernel_Binary_Node *>(node);
    return "while (" + compile_unparenthesized(loop->left) + ") " + compile(loop->right);
}

String Codegen::jump_to_string(Kernel_Node *node) {
    Kernel_Unary_Node *jump = static_cast<Kernel_Unary_Node *>(node);
    if (!jump->child) return jump->body.token;
    return jump->body.token + " " + compile_unparenthesized(jump->child);
}

// Metal buffer indices follow position, so adding a parameter needs no manual renumbering.
String Codegen::parameters_to_string(Kernel_Node *node, bool number_buffers) {
    String source;
    s64 buffer_index = 0;
    for (Kernel_Node *parameter_node : static_cast<Kernel_List_Node *>(node)->children) {
        Kernel_Trinary_Node *parameter = static_cast<Kernel_Trinary_Node *>(parameter_node);
        if (!source.empty()) source += ",";
        source += "\n    " + compile(parameter->left) + " " + compile(parameter->middle);
        if (parameter->right) {
            source += " " + compile(parameter->right);
        } else if (number_buffers && backend == Kernel_Backend::METAL) {
            source += " [[ buffer(" + std::to_string(buffer_index) + ") ]]";
            buffer_index += 1;
        }
    }
    return source + "\n";
}

String Codegen::device_function_impl_to_string(Kernel_Node *node) {
    const Vector<Kernel_Node *> &parts = static_cast<Kernel_List_Node *>(node)->children;
    const String qualifier = backend == Kernel_Backend::METAL ? "inline " : "__device__ inline ";
    return qualifier + compile(parts[0]) + " " + compile(parts[1]) +
           "(" + parameters_to_string(parts[2], false) + ") " + compile(parts[3]);
}

String Codegen::kernel_function_impl_to_string(Kernel_Node *node) {
    const Vector<Kernel_Node *> &parts = static_cast<Kernel_List_Node *>(node)->children;
    const String qualifier = backend == Kernel_Backend::METAL ? "kernel " : "extern \"C\" __global__ ";
    return qualifier + compile(parts[0]) + " " + compile(parts[1]) +
           "(" + parameters_to_string(parts[2], true) + ") " + compile(parts[3]);
}

}
