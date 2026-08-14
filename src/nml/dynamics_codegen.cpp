#include <algorithm>
#include <cctype>
#include <fstream>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "spikecorec/nml/dynamics_codegen.h"
#include "spikecorec/core/log.h"

using namespace std;

namespace spikecorec::nml {

namespace {

// ── tokenizer ────────────────────────────────────────────────────────────────────

// NML spells its comparisons and boolean connectives as dotted words; everything else in
// the expression grammar -- + - * / ( ) , and the function-call form -- is already C.
const UnorderedMap<String, String> DOTTED_OPERATORS = {
    {".gt.", ">"},  {".lt.", "<"},   {".geq.", ">="}, {".leq.", "<="},
    {".eq.", "=="}, {".neq.", "!="}, {".and.", "&&"}, {".or.", "||"},
};

// LEMS function name -> the name emitted into GPU source. Three of these do not map onto
// the same-named C function and getting any of them wrong is silent: LEMS `ln` is natural
// log (C `log`), LEMS `log` is base 10 (C `log10`), and `abs` on a float must be `fabs`
// rather than the integer `abs`.
const UnorderedMap<String, String> FUNCTIONS = {
    {"exp", "exp"},   {"ln", "log"},    {"log", "log10"}, {"sqrt", "sqrt"},
    {"abs", "fabs"},  {"ceil", "ceil"}, {"floor", "floor"},
    {"sin", "sin"},   {"cos", "cos"},   {"tan", "tan"},
    {"sinh", "sinh"}, {"cosh", "cosh"}, {"tanh", "tanh"},
    {"H", "spikecorec_heaviside"},
};

// LEMS `random(x)` needs a per-thread RNG stream that reproduces from the simulation seed;
// that arrives with the on-device generators in Phase 2 (ticket #65/F4). Listing it here
// rather than leaving it out makes the failure say so instead of "unknown function".
const Set<String> PHASE_TWO_FUNCTIONS = {"random"};

struct Token {
    enum class Kind { Number, Identifier, Operator, OpenParen, CloseParen, Comma, End };

    Kind kind = Kind::End;
    String text;
};

bool is_identifier_start(char character) {
    return isalpha(static_cast<unsigned char>(character)) || character == '_';
}

bool is_identifier_character(char character) {
    return isalnum(static_cast<unsigned char>(character)) || character == '_';
}

// A dotted operator and a decimal point both start with '.', so the two are told apart by
// what follows: ".5" is a number, ".gt." an operator.
bool starts_dotted_operator(const String &text, usize position) {
    return position + 1 < text.size() &&
           isalpha(static_cast<unsigned char>(text[position + 1]));
}

Vector<Token> tokenize(const String &expression, const String &owner_name) {
    Vector<Token> tokens;
    usize position = 0;

    while (position < expression.size()) {
        const char character = expression[position];

        if (isspace(static_cast<unsigned char>(character))) {
            position += 1;
            continue;
        }

        if (character == '(') { tokens.push_back({Token::Kind::OpenParen, "("});  position += 1; continue; }
        if (character == ')') { tokens.push_back({Token::Kind::CloseParen, ")"}); position += 1; continue; }
        if (character == ',') { tokens.push_back({Token::Kind::Comma, ","});      position += 1; continue; }

        if (character == '.' && starts_dotted_operator(expression, position)) {
            const usize closing_dot = expression.find('.', position + 1);
            if (closing_dot == String::npos) {
                throw runtime_error(
                        "dynamics_codegen: unterminated dotted operator in '" + expression +
                        "' (" + owner_name + ")");
            }

            const String spelling = expression.substr(position, closing_dot - position + 1);
            auto mapped = DOTTED_OPERATORS.find(spelling);
            if (mapped == DOTTED_OPERATORS.end()) {
                throw runtime_error(
                        "dynamics_codegen: unknown operator '" + spelling + "' in '" +
                        expression + "' (" + owner_name + ")");
            }

            tokens.push_back({Token::Kind::Operator, mapped->second});
            position = closing_dot + 1;
            continue;
        }

        if (isdigit(static_cast<unsigned char>(character)) || character == '.') {
            const usize start = position;
            while (position < expression.size() &&
                   (isdigit(static_cast<unsigned char>(expression[position])) ||
                    expression[position] == '.')) {
                position += 1;
            }
            // An exponent, and the sign that may follow it: 2.7e-5 is one token, not three.
            if (position < expression.size() &&
                (expression[position] == 'e' || expression[position] == 'E')) {
                usize lookahead = position + 1;
                if (lookahead < expression.size() &&
                    (expression[lookahead] == '+' || expression[lookahead] == '-')) {
                    lookahead += 1;
                }
                if (lookahead < expression.size() &&
                    isdigit(static_cast<unsigned char>(expression[lookahead]))) {
                    position = lookahead;
                    while (position < expression.size() &&
                           isdigit(static_cast<unsigned char>(expression[position]))) {
                        position += 1;
                    }
                }
            }
            tokens.push_back({Token::Kind::Number, expression.substr(start, position - start)});
            continue;
        }

        if (is_identifier_start(character)) {
            const usize start = position;
            while (position < expression.size() && is_identifier_character(expression[position])) {
                position += 1;
            }
            tokens.push_back({Token::Kind::Identifier, expression.substr(start, position - start)});
            continue;
        }

        // Two-character comparisons are also legal LEMS spelling alongside the dotted form.
        if (position + 1 < expression.size()) {
            const String pair = expression.substr(position, 2);
            if (pair == "<=" || pair == ">=" || pair == "==" || pair == "!=" ||
                pair == "&&" || pair == "||") {
                tokens.push_back({Token::Kind::Operator, pair});
                position += 2;
                continue;
            }
        }

        if (String("+-*/^<>").find(character) != String::npos) {
            tokens.push_back({Token::Kind::Operator, String(1, character)});
            position += 1;
            continue;
        }

        throw runtime_error(
                "dynamics_codegen: unexpected character '" + String(1, character) + "' in '" +
                expression + "' (" + owner_name + ")");
    }

    tokens.push_back({Token::Kind::End, ""});
    return tokens;
}

// ── parser ───────────────────────────────────────────────────────────────────────
//
// Precedence climbing straight to target text. There is no AST type: the target grammar
// is the source grammar with different spellings, so every parse function returns the
// translated substring and the tree only ever exists as the call stack.

s32 binary_precedence(const String &spelling) {
    if (spelling == "||") return 1;
    if (spelling == "&&") return 2;
    if (spelling == "==" || spelling == "!=") return 3;
    if (spelling == "<" || spelling == ">" || spelling == "<=" || spelling == ">=") return 4;
    if (spelling == "+" || spelling == "-") return 5;
    if (spelling == "*" || spelling == "/") return 6;
    if (spelling == "^") return 7;
    return -1;
}

struct Parser {
    const Vector<Token> &tokens;
    const SymbolTable &symbols;
    const String &expression;
    const String &owner_name;
    usize position = 0;

    const Token &current() const { return tokens[position]; }

    [[noreturn]] void fail(const String &reason) const {
        throw runtime_error("dynamics_codegen: " + reason + " in '" + expression + "' (" +
                            owner_name + ")");
    }

    String resolve_identifier(const String &name) const {
        auto bound = symbols.find(name);
        if (bound == symbols.end()) {
            throw runtime_error(
                    "dynamics_codegen: '" + name + "' in '" + expression + "' (" + owner_name +
                    ") resolves to no parameter, state variable, derived variable, constant "
                    "or engine quantity");
        }
        return bound->second;
    }

    String parse_primary() {
        const Token token = current();

        if (token.kind == Token::Kind::Number) {
            position += 1;
            // A bare integer in NML source would otherwise become integer division in the
            // generated C: `1/tau` must not evaluate to 0.
            const bool already_floating =
                    token.text.find('.') != String::npos ||
                    token.text.find('e') != String::npos ||
                    token.text.find('E') != String::npos;
            return already_floating ? token.text : token.text + ".0";
        }

        if (token.kind == Token::Kind::OpenParen) {
            position += 1;
            const String inner = parse_binary(0);
            if (current().kind != Token::Kind::CloseParen) fail("missing ')'");
            position += 1;

            // No parentheses added back. Everything parse_binary returns is already safe
            // to use as an operand -- either an atom (a name, a literal, a call) or a
            // binary expression it parenthesised itself -- so re-wrapping only produces
            // (((A + B)) * C) where (A + B) * C says the same thing and reads.
            return inner;
        }

        if (token.kind == Token::Kind::Operator && (token.text == "-" || token.text == "+")) {
            position += 1;
            const String operand = parse_unary();
            return token.text == "-" ? "(-" + operand + ")" : operand;
        }

        if (token.kind == Token::Kind::Identifier) {
            const String name = token.text;
            position += 1;

            if (current().kind != Token::Kind::OpenParen) return resolve_identifier(name);

            if (PHASE_TWO_FUNCTIONS.count(name) > 0) {
                throw runtime_error(
                        "dynamics_codegen: '" + name + "' in '" + expression + "' (" +
                        owner_name + ") needs the on-device generators from Phase 2 "
                        "(ticket #65/F4)");
            }

            auto function = FUNCTIONS.find(name);
            if (function == FUNCTIONS.end()) {
                throw runtime_error(
                        "dynamics_codegen: unknown function '" + name + "' in '" + expression +
                        "' (" + owner_name + ")");
            }

            position += 1;
            Vector<String> arguments;
            if (current().kind != Token::Kind::CloseParen) {
                for (;;) {
                    arguments.push_back(parse_binary(0));
                    if (current().kind != Token::Kind::Comma) break;
                    position += 1;
                }
            }
            if (current().kind != Token::Kind::CloseParen) fail("missing ')' after " + name);
            position += 1;

            String rendered = function->second + "(";
            for (usize index = 0; index < arguments.size(); index += 1) {
                if (index > 0) rendered += ", ";
                rendered += arguments[index];
            }
            return rendered + ")";
        }

        fail("expected a value");
    }

    String parse_unary() { return parse_primary(); }

    String parse_binary(s32 minimum_precedence) {
        String left = parse_unary();

        for (;;) {
            const Token token = current();
            if (token.kind != Token::Kind::Operator) break;

            const s32 precedence = binary_precedence(token.text);
            if (precedence < minimum_precedence) break;

            position += 1;

            // `^` is the only right-associative operator, so its right operand is parsed at
            // the same precedence rather than one above: a^b^c is a^(b^c).
            const String right =
                    parse_binary(token.text == "^" ? precedence : precedence + 1);

            left = token.text == "^" ? "pow(" + left + ", " + right + ")"
                                     : "(" + left + " " + token.text + " " + right + ")";
        }

        return left;
    }
};

// ── instruction grouping ─────────────────────────────────────────────────────────

// A DerivedVariable that names a path rather than an arithmetic expression -- iafCell's
// `<DerivedVariable name="iSyn" select="synapses[*]/i" reduce="add"/>` -- is a reduction
// over attached components, not something to translate. The engine has already summed it
// into the cell's input accumulator.
bool is_path_select(const String &expression) {
    return expression.find('/') != String::npos ||
           expression.find('[') != String::npos ||
           expression.find(']') != String::npos;
}

Vector<const DynamicsInstruction *> instructions_in_stage(
        const Vector<DynamicsInstruction> &program,
        DynamicsStage stage,
        NML_DeclarationType source_tag) {
    Vector<const DynamicsInstruction *> selected;
    for (const DynamicsInstruction &instruction : program) {
        if (instruction.stage != stage) continue;
        if (instruction.source_tag != source_tag) continue;
        selected.push_back(&instruction);
    }
    return selected;
}

// The distinct OnCondition tests in the program, in first-appearance order. Reset and Emit
// instructions carry the test that fired them in `condition`, so grouping by that string
// reassembles each OnCondition's body.
Vector<String> ordered_conditions(const Vector<DynamicsInstruction> &program) {
    Vector<String> conditions;
    for (const DynamicsInstruction &instruction : program) {
        if (instruction.stage != DynamicsStage::Reset &&
            instruction.stage != DynamicsStage::Emit) {
            continue;
        }
        if (instruction.condition.empty()) continue;

        bool already_seen = false;
        for (const String &known : conditions) already_seen |= known == instruction.condition;
        if (!already_seen) conditions.push_back(instruction.condition);
    }
    return conditions;
}

void reject_unsupported_constructs(const String &type_name,
                                   const Vector<DynamicsInstruction> &program) {
    for (const DynamicsInstruction &instruction : program) {
        if (!instruction.regime_name.empty()) {
            throw runtime_error(
                    "dynamics_codegen: '" + type_name + "' declares Regime '" +
                    instruction.regime_name + "'; regimes are Phase 2 (ticket #62/#63)");
        }
        if (instruction.source_tag == NML_DeclarationType::ConditionalDerivedVariable ||
            instruction.source_tag == NML_DeclarationType::Case) {
            throw runtime_error(
                    "dynamics_codegen: '" + type_name +
                    "' declares a ConditionalDerivedVariable; that is Phase 2");
        }
    }
}

// The Integrate stage in LEMS order: every DerivedVariable is a function of the current
// state and is evaluated first, then every TimeDerivative reads that state.
//
// The derivatives are all computed into temporaries before any state is written, so a
// system whose variables depend on each other steps simultaneously rather than
// sequentially -- alphaCurrentSynapse's dI/dt = (e*J - I)/tau alongside dJ/dt = -J/tau is
// the standard case, and updating I from an already-advanced J is a silent integration
// error rather than a compile failure.
String emit_integrate_stage(const Vector<DynamicsInstruction> &program,
                            const SymbolTable &symbols,
                            const Vector<String> &state_variable_names,
                            const String &state_reference_prefix,
                            const String &type_name,
                            const String &indent) {
    ostringstream source;

    for (const DynamicsInstruction *derived :
         instructions_in_stage(program, DynamicsStage::Integrate,
                               NML_DeclarationType::DerivedVariable)) {
        if (is_path_select(derived->expression)) continue; // bound to the input accumulator

        source << indent << "const float derived_" << derived->target << " = "
               << translate_expression(derived->expression, symbols, type_name) << ";\n";
    }

    const Vector<const DynamicsInstruction *> derivatives = instructions_in_stage(
            program, DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative);

    for (usize index = 0; index < derivatives.size(); index += 1) {
        source << indent << "const float derivative_" << index << " = "
               << translate_expression(derivatives[index]->expression, symbols, type_name)
               << ";\n";
    }

    for (usize index = 0; index < derivatives.size(); index += 1) {
        const String &written = derivatives[index]->target;

        s64 slot = -1;
        for (usize candidate = 0; candidate < state_variable_names.size(); candidate += 1) {
            if (state_variable_names[candidate] == written) slot = (s64)candidate;
        }
        if (slot < 0) {
            throw runtime_error(
                    "dynamics_codegen: '" + type_name + "' has a TimeDerivative for '" +
                    written + "', which is not one of its StateVariables");
        }

        source << indent << state_reference_prefix << slot << " += step_dt * derivative_"
               << index << ";\n";
    }

    return source.str();
}

// Everything an OnCondition fires: the StateAssignments it performs and the EventOut that
// makes the neuron spike. Emitted as one `if` per distinct test.
String emit_conditional_stages(const Vector<DynamicsInstruction> &program,
                               const SymbolTable &symbols,
                               const Vector<String> &state_variable_names,
                               const String &state_reference_prefix,
                               const String &type_name,
                               const String &spike_flag_name,
                               const String &indent) {
    ostringstream source;

    for (const String &condition : ordered_conditions(program)) {
        source << indent << "if (" << translate_expression(condition, symbols, type_name)
               << ") {\n";

        for (const DynamicsInstruction &instruction : program) {
            if (instruction.condition != condition) continue;

            if (instruction.stage == DynamicsStage::Reset &&
                instruction.source_tag == NML_DeclarationType::StateAssignment) {
                s64 slot = -1;
                for (usize candidate = 0; candidate < state_variable_names.size();
                     candidate += 1) {
                    if (state_variable_names[candidate] == instruction.target) {
                        slot = (s64)candidate;
                    }
                }
                if (slot < 0) {
                    throw runtime_error(
                            "dynamics_codegen: '" + type_name + "' assigns '" +
                            instruction.target +
                            "' in an OnCondition, but it is not one of its StateVariables");
                }

                source << indent << "    " << state_reference_prefix << slot << " = "
                       << translate_expression(instruction.expression, symbols, type_name)
                       << ";\n";
            }

            if (instruction.stage == DynamicsStage::Emit && !spike_flag_name.empty()) {
                source << indent << "    " << spike_flag_name << " = true;\n";
            }
        }

        source << indent << "}\n";
    }

    return source.str();
}

// ── symbol tables ────────────────────────────────────────────────────────────────

// Constants a document declares at its own scope are usable by name inside any
// expression, so they are folded straight into the source as literals.
void bind_global_constants(const NML_ParseResult &parse_result, SymbolTable &symbols) {
    for (const auto &[name, value] : parse_result.global_constants) {
        // A type-namespaced constant ("iafCell.foo") is not a bare identifier and can
        // never be written in an expression, so only the plain names are bound.
        if (name.find('.') != String::npos) continue;

        ostringstream literal;
        literal.precision(17);
        literal << "(" << value.float64 << ")";
        symbols[name] = literal.str();
    }
}

SymbolTable build_cell_symbols(const CellTypeSpecification &cell_type,
                               const NML_ParseResult &parse_result) {
    SymbolTable symbols;
    bind_global_constants(parse_result, symbols);

    for (usize slot = 0; slot < cell_type.state_variable_names.size(); slot += 1) {
        symbols[cell_type.state_variable_names[slot]] = "state_" + to_string(slot);
    }
    for (usize slot = 0; slot < cell_type.parameter_names.size(); slot += 1) {
        symbols[cell_type.parameter_names[slot]] =
                "cell_parameters[parameter_base + " + to_string(slot) + "]";
    }

    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.source_tag != NML_DeclarationType::DerivedVariable) continue;

        // A path-select DerivedVariable is the sum over the cell's attached synapses --
        // iafCell's `iSyn`. The engine has already accumulated exactly that into the
        // neuron's input slot, so the name binds to it rather than being translated.
        symbols[instruction.target] = is_path_select(instruction.expression)
                ? "network_input"
                : "derived_" + instruction.target;
    }

    symbols["t"] = "(step_dt * (float)tick)";
    return symbols;
}

SymbolTable build_synapse_symbols(const SynapseTypeSpecification &synapse_type,
                                  const NML_ParseResult &parse_result) {
    SymbolTable symbols;
    bind_global_constants(parse_result, symbols);

    for (usize slot = 0; slot < synapse_type.state_variable_names.size(); slot += 1) {
        symbols[synapse_type.state_variable_names[slot]] = "state_" + to_string(slot);
    }
    for (usize slot = 0; slot < synapse_type.parameter_names.size(); slot += 1) {
        symbols[synapse_type.parameter_names[slot]] =
                "synapse_parameters[synapse_parameter_base + " + to_string(slot) + "]";
    }

    for (const DynamicsInstruction &instruction : synapse_type.dynamics) {
        if (instruction.source_tag != NML_DeclarationType::DerivedVariable) continue;
        symbols[instruction.target] = "derived_" + instruction.target;
    }

    // `weight` is declared as a Property on the synapse type but supplied per connection
    // (`<connection weight="..."/>`), so it is the edge's stored weight, not the
    // prototype's parameter row. Binding it after the parameter loop is what overrides the
    // slot the loop just wrote.
    symbols["weight"] = "edge_weight";
    symbols["t"] = "(step_dt * (float)tick)";
    return symbols;
}

// ── generated bodies ─────────────────────────────────────────────────────────────

String generate_cell_body(const CellTypeSpecification &cell_type,
                          const NML_ParseResult &parse_result,
                          s64 case_index) {
    reject_unsupported_constructs(cell_type.name, cell_type.dynamics);

    const SymbolTable symbols = build_cell_symbols(cell_type, parse_result);
    const String indent = "            ";
    ostringstream source;

    source << "        case " << case_index << ": { // " << cell_type.name << "\n";

    for (usize slot = 0; slot < cell_type.state_variable_names.size(); slot += 1) {
        source << indent << "float state_" << slot << " = cell_state[state_base + " << slot
               << " * state_stride]; // " << cell_type.state_variable_names[slot] << "\n";
    }
    source << "\n";

    source << emit_integrate_stage(cell_type.dynamics, symbols,
                                   cell_type.state_variable_names, "state_",
                                   cell_type.name, indent);
    source << emit_conditional_stages(cell_type.dynamics, symbols,
                                      cell_type.state_variable_names, "state_",
                                      cell_type.name, "spiked", indent);

    source << "\n";
    for (usize slot = 0; slot < cell_type.state_variable_names.size(); slot += 1) {
        source << indent << "cell_state[state_base + " << slot << " * state_stride] = state_"
               << slot << ";\n";
    }

    source << "        } break;\n";
    return source.str();
}

String generate_synapse_body(const SynapseTypeSpecification &synapse_type,
                             const NML_ParseResult &parse_result,
                             s64 case_index) {
    if (synapse_type.is_conductance_based) {
        throw runtime_error(
                "dynamics_codegen: '" + synapse_type.name +
                "' is conductance-based; Phase 1 simulates current-based synapses only");
    }
    if (synapse_type.requires_per_edge_state) {
        // Every synapse here is already per-edge; the flag means the type composes a
        // plasticity or block mechanism, whose child dynamics are not lowered yet.
        throw runtime_error(
                "dynamics_codegen: '" + synapse_type.name +
                "' composes a plasticity or block mechanism; that is Phase 2 (ticket #66)");
    }
    reject_unsupported_constructs(synapse_type.name, synapse_type.dynamics);

    const SymbolTable symbols = build_synapse_symbols(synapse_type, parse_result);
    const String indent = "                    ";
    ostringstream source;

    source << "                case " << case_index << ": { // " << synapse_type.name << "\n";

    for (usize slot = 0; slot < synapse_type.state_variable_names.size(); slot += 1) {
        source << indent << "float state_" << slot << " = edge_variables[" << (slot + 1)
               << " * edge_plane_stride + edge_base]; // "
               << synapse_type.state_variable_names[slot] << "\n";
    }
    source << "\n";

    // The OnEvent handler fires first, so a spike that arrives this tick is part of the
    // state the derivative step then advances.
    ostringstream arrival;
    for (const DynamicsInstruction &instruction : synapse_type.dynamics) {
        if (instruction.stage != DynamicsStage::Arrival) continue;
        if (instruction.source_tag != NML_DeclarationType::StateAssignment) continue;

        s64 slot = -1;
        for (usize candidate = 0; candidate < synapse_type.state_variable_names.size();
             candidate += 1) {
            if (synapse_type.state_variable_names[candidate] == instruction.target) {
                slot = (s64)candidate;
            }
        }
        if (slot < 0) {
            throw runtime_error(
                    "dynamics_codegen: '" + synapse_type.name + "' assigns '" +
                    instruction.target +
                    "' in an OnEvent, but it is not one of its StateVariables");
        }

        arrival << indent << "    state_" << slot << " = "
                << translate_expression(instruction.expression, symbols, synapse_type.name)
                << ";\n";
    }
    if (!arrival.str().empty()) {
        source << indent << "if (arrived) {\n" << arrival.str() << indent << "}\n";
    }

    source << emit_integrate_stage(synapse_type.dynamics, symbols,
                                   synapse_type.state_variable_names, "state_",
                                   synapse_type.name, indent);

    // The current this synapse delivers is its `i` exposure, which every
    // baseCurrentBasedSynapse declares as a DerivedVariable.
    bool exposes_current = false;
    for (const DynamicsInstruction &instruction : synapse_type.dynamics) {
        if (instruction.source_tag != NML_DeclarationType::DerivedVariable) continue;
        if (instruction.target != "i") continue;
        exposes_current = true;
    }
    if (!exposes_current) {
        throw runtime_error(
                "dynamics_codegen: '" + synapse_type.name +
                "' exposes no current `i`; a current-based synapse must derive one");
    }
    source << indent << "synapse_current = derived_i;\n\n";

    for (usize slot = 0; slot < synapse_type.state_variable_names.size(); slot += 1) {
        source << indent << "edge_variables[" << (slot + 1)
               << " * edge_plane_stride + edge_base] = state_" << slot << ";\n";
    }

    source << "                } break;\n";
    return source.str();
}

// ── baked model tables ───────────────────────────────────────────────────────────

String emit_integer_table(const String &name, const Vector<s64> &values) {
    ostringstream source;
    source << "constant int " << name << "[] = { ";
    for (usize index = 0; index < values.size(); index += 1) {
        if (index > 0) source << ", ";
        source << values[index];
    }
    // A zero-length constant array is not legal MSL, and a model with no synapses reaches
    // exactly that; one unreachable entry keeps the declaration well-formed.
    if (values.empty()) source << "0";
    source << " };\n";
    return source.str();
}

const char *KERNEL_PREAMBLE = R"METAL(
// Generated by spikecorec dynamics_codegen. Do not edit -- regenerate from the model.

inline float spikecorec_heaviside(float value) {
    return value >= 0.0f ? 1.0f : 0.0f;
}
)METAL";

#ifndef SPIKECOREC_METAL_DEVICE_DIR
#define SPIKECOREC_METAL_DEVICE_DIR ""
#endif

// A kernel handed to compile_kernel() is compiled from a bare string with no include path,
// so device code shared with the precompiled shaders has to be pasted in rather than
// included. Reading it keeps one copy of the k^2-tree walk instead of a second that can
// silently drift from the first.
String read_device_include(const String &file_name) {
    const String path = String(SPIKECOREC_METAL_DEVICE_DIR) + "/" + file_name;

    ifstream file(path);
    if (!file) {
        throw runtime_error(
                "dynamics_codegen: cannot read device include '" + path +
                "'; SPIKECOREC_METAL_DEVICE_DIR must point at the source tree's src/metal");
    }

    ostringstream contents;
    contents << file.rdbuf();
    return "\n" + contents.str() + "\n";
}

const char *KERNEL_SIGNATURE = R"METAL(
kernel void master_step(
    constant long        &tick                      [[ buffer(0)  ]],
    constant float       &step_dt                   [[ buffer(1)  ]],
    constant int         &neuron_count              [[ buffer(2)  ]],
    constant int         &spike_history_length      [[ buffer(3)  ]],
    constant int         &max_neighbor_count        [[ buffer(4)  ]],
    device   float       *cell_state                [[ buffer(5)  ]],
    const device float   *cell_parameters           [[ buffer(6)  ]],
    const device float   *synapse_parameters        [[ buffer(7)  ]],
    device   float       *network_inputs            [[ buffer(8)  ]],
    device   uchar       *spike_history             [[ buffer(9)  ]],
    device   long        *last_spiked               [[ buffer(10) ]],
    const device uint    *internal_node_words       [[ buffer(11) ]],
    const device uint    *leaf_node_words           [[ buffer(12) ]],
    const device uint    *rank_superblock_table     [[ buffer(13) ]],
    const device ushort  *rank_subblock_table       [[ buffer(14) ]],
    constant int         &branching_factor          [[ buffer(15) ]],
    constant int         &superblock_size_words     [[ buffer(16) ]],
    constant int         &padded_node_count         [[ buffer(17) ]],
    constant int         &tree_height               [[ buffer(18) ]],
    constant int         &internal_bit_count        [[ buffer(19) ]],
    const device float   *edge_weights              [[ buffer(20) ]],
    const device float   *edge_delays               [[ buffer(21) ]],
    device   float       *edge_variables            [[ buffer(22) ]],
    uint thread_id [[ thread_position_in_grid ]]
) {
)METAL";

const char *KERNEL_PROLOGUE = R"METAL(
    const int neuron_index = (int)thread_id;
    if (neuron_index >= neuron_count) return;

    const int edge_plane_stride = neuron_count * max_neighbor_count;

    // ── stage 1 · deliver ────────────────────────────────────────────────────────
    // network_inputs is two rows: this tick's arrivals are read out of one while this
    // tick's scatters go into the other, so a thread never reads a slot another thread is
    // still writing. That is what makes the one-tick synaptic latency exact rather than
    // "one tick, or two if the scatter happened to land first".
    const int current_row = (int)(tick % 2);
    const int next_row    = 1 - current_row;

    const int input_slot = current_row * neuron_count + neuron_index;
    const float network_input = network_inputs[input_slot];
    network_inputs[input_slot] = 0.0f;

    // Populations occupy contiguous neuron ranges in document order, so the last one
    // starting at or below this neuron is the one it belongs to.
    int population = 0;
    for (int candidate = POPULATION_COUNT - 1; candidate >= 0; --candidate) {
        if (neuron_index >= population_first_neuron[candidate]) { population = candidate; break; }
    }
    const int local_index    = neuron_index - population_first_neuron[population];
    const int state_base     = population_state_base[population] + local_index;
    const int state_stride   = population_neuron_count[population];
    const int parameter_base = population_parameter_base[population];

    // ── stages 2-5 · integrate, detect, emit, reset ──────────────────────────────
    bool spiked = false;
    switch (population_cell_type[population]) {
)METAL";

const char *KERNEL_BETWEEN_CELL_AND_PROPAGATE = R"METAL(    }

    if (spiked) last_spiked[neuron_index] = tick;

    // The history row this tick writes is never a row a delayed arrival reads, because
    // every delay is at least 1 and the ring is one longer than the longest delay.
    spike_history[(int)(tick % spike_history_length) * neuron_count + neuron_index] =
        spiked ? (uchar)1 : (uchar)0;

    // ── stage 6 · propagate ──────────────────────────────────────────────────────
    // Walked every tick, not only on a spike: the per-edge synapse state decays whether or
    // not anything arrived. This thread owns every outgoing edge of its own neuron, so the
    // per-edge slots at (neuron_index, slot) are written by no other thread and need no
    // atomics; only the scatter into another neuron's input slot does.
    thread int walk_stack_row_base[MAX_K2TREE_HEIGHT];
    thread int walk_stack_col_base[MAX_K2TREE_HEIGHT];
    thread int walk_stack_block_size[MAX_K2TREE_HEIGHT];
    thread int walk_stack_bit_offset[MAX_K2TREE_HEIGHT];
    thread int walk_stack_next_col[MAX_K2TREE_HEIGHT];
    walk_stack_row_base[0]   = 0;
    walk_stack_col_base[0]   = 0;
    walk_stack_block_size[0] = padded_node_count;
    walk_stack_bit_offset[0] = 0;
    walk_stack_next_col[0]   = 0;
    int walk_stack_top = tree_height > 0 ? 0 : -1;

    int slot = 0;
    int target;
    while ((target = k2t_next_neighbor(
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, neuron_count,
        tree_height, internal_bit_count, neuron_index,
        walk_stack_row_base, walk_stack_col_base, walk_stack_block_size,
        walk_stack_bit_offset, walk_stack_next_col, walk_stack_top
    )) >= 0) {
        if (slot >= max_neighbor_count) break;

        const int edge_base = neuron_index * max_neighbor_count + slot;
        const float edge_weight = edge_weights[edge_base];

        int delay_ticks = (int)edge_delays[edge_base];
        if (delay_ticks < 1) delay_ticks = 1;

        // Did this neuron spike exactly delay_ticks ago? The history ring is what makes
        // that answerable for every spike rather than only the most recent one.
        bool arrived = false;
        if (tick >= delay_ticks) {
            const int arrival_row = (int)((tick - delay_ticks) % spike_history_length);
            arrived = spike_history[arrival_row * neuron_count + neuron_index] != 0;
        }

        const int synapse_prototype = (int)edge_variables[edge_base];
        const int synapse_parameter_base = synapse_prototype_parameter_base[synapse_prototype];

        float synapse_current = 0.0f;
        switch (synapse_prototype_type[synapse_prototype]) {
)METAL";

const char *KERNEL_EPILOGUE = R"METAL(        }

        device atomic_float *arrival_slot =
            (device atomic_float *)(network_inputs + next_row * neuron_count + target);
        atomic_fetch_add_explicit(arrival_slot, synapse_current, memory_order_relaxed);

        slot += 1;
    }
}
)METAL";

} // namespace

// ── public surface ───────────────────────────────────────────────────────────────

String translate_expression(const String &expression,
                            const SymbolTable &symbols,
                            const String &owner_name) {
    if (expression.empty()) {
        throw runtime_error("dynamics_codegen: empty expression (" + owner_name + ")");
    }

    const Vector<Token> tokens = tokenize(expression, owner_name);
    Parser parser{tokens, symbols, expression, owner_name};

    const String translated = parser.parse_binary(0);
    if (parser.current().kind != Token::Kind::End) {
        throw runtime_error("dynamics_codegen: trailing '" + parser.current().text + "' in '" +
                            expression + "' (" + owner_name + ")");
    }

    return translated;
}

f64 evaluate_initial_value(const String &expression,
                           const Vector<String> &parameter_names,
                           const Vector<Real> &parameter_values,
                           const String &owner_name) {
    String text = expression;
    while (!text.empty() && isspace(static_cast<unsigned char>(text.front()))) text.erase(0, 1);
    while (!text.empty() && isspace(static_cast<unsigned char>(text.back()))) text.pop_back();

    f64 sign = 1.0;
    if (!text.empty() && (text.front() == '-' || text.front() == '+')) {
        if (text.front() == '-') sign = -1.0;
        text.erase(0, 1);
        while (!text.empty() && isspace(static_cast<unsigned char>(text.front()))) text.erase(0, 1);
    }

    for (usize index = 0; index < parameter_names.size(); index += 1) {
        if (parameter_names[index] != text) continue;
        if (index >= parameter_values.size()) break;
        return sign * parameter_values[index].float64;
    }

    try {
        usize consumed = 0;
        const f64 literal = std::stod(text, &consumed);
        if (consumed == text.size()) return sign * literal;
    } catch (const std::exception &) {
        // falls through to the diagnostic below
    }

    throw runtime_error(
            "dynamics_codegen: OnStart value '" + expression + "' in '" + owner_name +
            "' is neither a literal nor one of the type's parameters; Phase 1 folds OnStart "
            "on the host and does not evaluate general expressions there");
}

ModelLayout compute_model_layout(const NML_ParseResult &parse_result) {
    ModelLayout layout;
    layout.total_neuron_count = (s64)parse_result.neurons.size();

    for (const PopulationLayout &population : parse_result.populations) {
        const CellTypeSpecification &cell_type =
                parse_result.cell_types[(usize)population.cell_type_index];

        layout.population_state_base.push_back(layout.cell_state_length);
        layout.cell_state_length +=
                (s64)cell_type.state_variable_names.size() * population.neuron_count;
    }

    for (const ComponentPrototype &prototype : parse_result.cell_prototypes) {
        layout.cell_prototype_parameter_base.push_back(layout.cell_parameter_length);
        layout.cell_parameter_length += (s64)prototype.starting_parameters.size();
    }

    for (const ComponentPrototype &prototype : parse_result.synapse_prototypes) {
        layout.synapse_prototype_parameter_base.push_back(layout.synapse_parameter_length);
        layout.synapse_parameter_length += (s64)prototype.starting_parameters.size();
    }

    for (const SynapseTypeSpecification &synapse_type : parse_result.synapse_types) {
        layout.widest_synapse_state_count = std::max(
                layout.widest_synapse_state_count,
                (s64)synapse_type.state_variable_names.size());
    }
    layout.per_edge_variable_count = 1 + layout.widest_synapse_state_count;

    for (const Neuron &neuron : parse_result.neurons) {
        layout.total_edge_count += (s64)neuron.outgoing_edges.size();
        for (const NetworkEdge &edge : neuron.outgoing_edges) {
            layout.maximum_edge_delay =
                    std::max(layout.maximum_edge_delay, edge.delay_tick_count);
        }
    }
    layout.spike_history_length = std::max<s64>(2, layout.maximum_edge_delay + 1);

    return layout;
}

String generate_master_kernel(const NML_ParseResult &parse_result, const ModelLayout &layout) {
    if (parse_result.populations.empty()) {
        throw runtime_error(
                "dynamics_codegen: the model declares no populations, so there is nothing "
                "to generate a kernel for");
    }

    ostringstream source;
    source << "#include <metal_stdlib>\n"
           << "using namespace metal;\n"
           << read_device_include("k2tree_device.metalinc")
           << KERNEL_PREAMBLE << "\n";

    // ── model tables, baked rather than passed ───────────────────────────────────
    // Populations occupy contiguous neuron ranges and there are a handful of them, so the
    // per-neuron cell type / state base / parameter base every thread needs are derivable
    // from a table small enough to compile into the kernel. That is four per-neuron
    // buffers the engine does not allocate, fill, or bind.
    Vector<s64> first_neuron, neuron_counts, cell_types, state_bases, parameter_bases;
    for (usize index = 0; index < parse_result.populations.size(); index += 1) {
        const PopulationLayout &population = parse_result.populations[index];
        first_neuron.push_back(population.first_neuron_index);
        neuron_counts.push_back(population.neuron_count);
        cell_types.push_back(population.cell_type_index);
        state_bases.push_back(layout.population_state_base[index]);
        parameter_bases.push_back(
                layout.cell_prototype_parameter_base[(usize)population.prototype_index]);
    }

    source << "constant int POPULATION_COUNT = " << parse_result.populations.size() << ";\n"
           << emit_integer_table("population_first_neuron", first_neuron)
           << emit_integer_table("population_neuron_count", neuron_counts)
           << emit_integer_table("population_cell_type", cell_types)
           << emit_integer_table("population_state_base", state_bases)
           << emit_integer_table("population_parameter_base", parameter_bases)
           << "\n";

    Vector<s64> synapse_types, synapse_parameter_bases;
    for (usize index = 0; index < parse_result.synapse_prototypes.size(); index += 1) {
        synapse_types.push_back(parse_result.synapse_prototypes[index].type_index);
        synapse_parameter_bases.push_back(layout.synapse_prototype_parameter_base[index]);
    }
    source << emit_integer_table("synapse_prototype_type", synapse_types)
           << emit_integer_table("synapse_prototype_parameter_base", synapse_parameter_bases)
           << "\n";

    // ── the generated bodies ─────────────────────────────────────────────────────
    ostringstream cell_bodies;
    for (usize index = 0; index < parse_result.cell_types.size(); index += 1) {
        cell_bodies << generate_cell_body(parse_result.cell_types[index], parse_result,
                                          (s64)index);
    }

    ostringstream synapse_bodies;
    for (usize index = 0; index < parse_result.synapse_types.size(); index += 1) {
        synapse_bodies << generate_synapse_body(parse_result.synapse_types[index],
                                                parse_result, (s64)index);
    }

    source << KERNEL_SIGNATURE
           << KERNEL_PROLOGUE
           << cell_bodies.str()
           << KERNEL_BETWEEN_CELL_AND_PROPAGATE
           << synapse_bodies.str()
           << KERNEL_EPILOGUE;

    return source.str();
}

} // namespace spikecorec::nml
