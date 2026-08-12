#include "spikecorec/nml/kernel_codegen.h"

#include <cctype>
#include <iomanip>
#include <sstream>
#include <stdexcept>

using namespace std;
using namespace spikecorec;

namespace spikecorec::nml {

// ── SymbolTable ──────────────────────────────────────────────────────────────

void SymbolTable::define(const String &identifier, const String &read_expression) {
    // emplace, not operator[]: the first definition wins, which is what turns insertion
    // order into resolution precedence.
    identifier_expressions.emplace(identifier, read_expression);
}

bool SymbolTable::contains(const String &identifier) const {
    return identifier_expressions.find(identifier) != identifier_expressions.end();
}

const String &SymbolTable::read_expression_for(const String &identifier) const {
    const auto entry = identifier_expressions.find(identifier);
    if (entry == identifier_expressions.end()) {
        throw runtime_error("kernel_codegen: unknown identifier '" + identifier +
                            "' in ComponentType '" + component_type_name + "'");
    }
    return entry->second;
}

namespace {

// ── Backend selection ────────────────────────────────────────────────────────

enum class KernelBackend { Metal, Cuda };

// Which entry point is being built. The two share a signature, a switch and a preamble;
// only the per-type body and the names differ.
enum class KernelPurpose { Tick, Initialize };

constexpr KernelBackend active_backend() {
#if defined(SPIKECOREC_CUDA) && !defined(SPIKECOREC_METAL)
    return KernelBackend::Cuda;
#else
    return KernelBackend::Metal;
#endif
}

[[noreturn]] void report_error(const String &message, const String &component_type_name) {
    throw runtime_error("kernel_codegen: " + message + " (ComponentType '" +
                        component_type_name + "')");
}

// ── Character classification ─────────────────────────────────────────────────
// Hand-rolled rather than <cctype> at the call sites so a negative char cannot reach
// isalpha and so the notion of "identifier character" is stated once.

bool is_digit_character(char character) {
    return character >= '0' && character <= '9';
}

bool is_identifier_start(char character) {
    return (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
           character == '_';
}

bool is_identifier_character(char character) {
    return is_identifier_start(character) || is_digit_character(character);
}

bool is_whitespace_character(char character) {
    return character == ' ' || character == '\t' || character == '\n' || character == '\r';
}

// ── Tokenizer ────────────────────────────────────────────────────────────────

enum class TokenKind {
    Number,
    Identifier,
    Operator,
    LeftParenthesis,
    RightParenthesis,
    Comma,
    EndOfInput
};

struct Token {
    TokenKind kind = TokenKind::EndOfInput;
    String text;
};

// LEMS spells its comparisons and connectives between dots. The bare C forms also occur in
// NeuroML documents and pass straight through.
const UnorderedMap<String, String> &dotted_operator_table() {
    static const UnorderedMap<String, String> table = {
        {"gt", ">"},   {"lt", "<"},   {"geq", ">="}, {"leq", "<="},
        {"eq", "=="},  {"neq", "!="}, {"and", "&&"}, {"or", "||"},
    };
    return table;
}

// Scans one numeric literal starting at `index`, leaving `index` one past its end.
String scan_number_literal(const String &expression, usize &index) {
    const usize start = index;
    const usize length = expression.length();
    bool has_seen_decimal_point = false;

    while (index < length) {
        const char character = expression[index];

        if (is_digit_character(character)) {
            index++;
            continue;
        }

        if (character == '.' && !has_seen_decimal_point) {
            // A '.' followed by a letter opens a dotted operator, as in "0.gt.1" -- the
            // number has to end before it. This is the case that makes textual
            // substitution unworkable and a tokenizer necessary.
            const char following = (index + 1 < length) ? expression[index + 1] : '\0';
            if (is_identifier_start(following)) break;
            has_seen_decimal_point = true;
            index++;
            continue;
        }

        if (character == 'e' || character == 'E') {
            // Only an exponent when digits really follow, optionally through a sign.
            usize lookahead = index + 1;
            if (lookahead < length &&
                (expression[lookahead] == '+' || expression[lookahead] == '-')) {
                lookahead++;
            }
            if (lookahead < length && is_digit_character(expression[lookahead])) {
                index = lookahead;
                while (index < length && is_digit_character(expression[index])) index++;
            }
            break;
        }

        break;
    }

    return expression.substr(start, index - start);
}

Vector<Token> tokenize_nml_expression(const String &expression,
                                      const String &component_type_name) {
    Vector<Token> tokens;
    const usize length = expression.length();
    usize index = 0;

    while (index < length) {
        const char character = expression[index];

        if (is_whitespace_character(character)) {
            index++;
            continue;
        }

        if (is_digit_character(character) ||
            (character == '.' && index + 1 < length && is_digit_character(expression[index + 1]))) {
            tokens.push_back(Token{TokenKind::Number, scan_number_literal(expression, index)});
            continue;
        }

        if (is_identifier_start(character)) {
            const usize start = index;
            while (index < length && is_identifier_character(expression[index])) index++;
            tokens.push_back(Token{TokenKind::Identifier, expression.substr(start, index - start)});
            continue;
        }

        if (character == '.') {
            const usize word_start = index + 1;
            usize word_end = word_start;
            while (word_end < length && is_identifier_character(expression[word_end])) word_end++;

            if (word_end >= length || expression[word_end] != '.') {
                report_error("malformed dotted operator in expression \"" + expression + "\"",
                             component_type_name);
            }

            const String word = expression.substr(word_start, word_end - word_start);
            const auto entry = dotted_operator_table().find(word);
            if (entry == dotted_operator_table().end()) {
                report_error("unknown dotted operator '." + word + ".' in expression \"" +
                                     expression + "\"",
                             component_type_name);
            }

            tokens.push_back(Token{TokenKind::Operator, entry->second});
            index = word_end + 1;
            continue;
        }

        if (character == '(') {
            tokens.push_back(Token{TokenKind::LeftParenthesis, "("});
            index++;
            continue;
        }

        if (character == ')') {
            tokens.push_back(Token{TokenKind::RightParenthesis, ")"});
            index++;
            continue;
        }

        if (character == ',') {
            tokens.push_back(Token{TokenKind::Comma, ","});
            index++;
            continue;
        }

        // Two-character operators are matched before their one-character prefixes.
        if (index + 1 < length) {
            const String pair = expression.substr(index, 2);
            if (pair == ">=" || pair == "<=" || pair == "==" || pair == "!=" || pair == "&&" ||
                pair == "||") {
                tokens.push_back(Token{TokenKind::Operator, pair});
                index += 2;
                continue;
            }
        }

        if (character == '+' || character == '-' || character == '*' || character == '/' ||
            character == '^' || character == '>' || character == '<') {
            tokens.push_back(Token{TokenKind::Operator, String(1, character)});
            index++;
            continue;
        }

        report_error("unexpected character '" + String(1, character) + "' in expression \"" +
                             expression + "\"",
                     component_type_name);
    }

    tokens.push_back(Token{TokenKind::EndOfInput, ""});
    return tokens;
}

// ── Literal formatting ───────────────────────────────────────────────────────

// Metal rejects the implicit double->float narrowing an unsuffixed literal would cause, so
// every number reaches the shader already a float literal.
String to_float_literal(const String &number_text) {
    const bool is_already_fractional = number_text.find('.') != String::npos ||
                                       number_text.find('e') != String::npos ||
                                       number_text.find('E') != String::npos;
    return is_already_fractional ? number_text + "f" : number_text + ".0f";
}

// Scientific with 9 significant digits: enough to carry a float exactly, and it keeps
// values like 1e-5 (a plausible dt) from rounding to zero the way a fixed-point format
// would.
String format_float_literal(f64 value) {
    ostringstream stream;
    stream << scientific << setprecision(9) << value << "f";
    return stream.str();
}

// ── Expression parser ────────────────────────────────────────────────────────
//
// Recursive descent, lowest precedence outermost:
//   || -> && -> == != -> > < >= <= -> + - -> * / -> unary -> ^ -> primary
// Each binary result is parenthesised, so the emitted C reproduces the parse regardless of
// how C would have grouped it.

class ExpressionParser {
public:
    ExpressionParser(Vector<Token> tokens, const SymbolTable &symbols)
        : tokens_(std::move(tokens)), symbols_(symbols) {}

    String parse_complete_expression() {
        String result = parse_logical_or();
        if (current().kind != TokenKind::EndOfInput) {
            fail("unexpected trailing token '" + current().text + "'");
        }
        return result;
    }

private:
    using ParseFunction = String (ExpressionParser::*)();

    Vector<Token> tokens_;
    const SymbolTable &symbols_;
    usize position_ = 0;

    const Token &current() const { return tokens_[position_]; }

    void advance() {
        if (position_ + 1 < tokens_.size()) position_++;
    }

    [[noreturn]] void fail(const String &message) const {
        report_error(message, symbols_.component_type_name);
    }

    bool current_matches(const Vector<String> &operators) const {
        if (current().kind != TokenKind::Operator) return false;
        for (const String &candidate : operators) {
            if (current().text == candidate) return true;
        }
        return false;
    }

    String parse_left_associative(ParseFunction next_level, const Vector<String> &operators) {
        String left = (this->*next_level)();
        while (current_matches(operators)) {
            const String operator_text = current().text;
            advance();
            const String right = (this->*next_level)();
            left = "(" + left + " " + operator_text + " " + right + ")";
        }
        return left;
    }

    String parse_logical_or() {
        return parse_left_associative(&ExpressionParser::parse_logical_and, {"||"});
    }

    String parse_logical_and() {
        return parse_left_associative(&ExpressionParser::parse_equality, {"&&"});
    }

    String parse_equality() {
        return parse_left_associative(&ExpressionParser::parse_relational, {"==", "!="});
    }

    String parse_relational() {
        return parse_left_associative(&ExpressionParser::parse_additive, {">", "<", ">=", "<="});
    }

    String parse_additive() {
        return parse_left_associative(&ExpressionParser::parse_multiplicative, {"+", "-"});
    }

    String parse_multiplicative() {
        return parse_left_associative(&ExpressionParser::parse_unary, {"*", "/"});
    }

    String parse_unary() {
        if (current_matches({"-", "+"})) {
            const String operator_text = current().text;
            advance();
            const String operand = parse_unary();
            // Unary plus on a float is a no-op, so it is dropped rather than emitted.
            if (operator_text == "+") return operand;
            return "(-" + operand + ")";
        }
        return parse_power();
    }

    // `^` binds tighter than unary minus and is right-associative: `-x^2` is -(x^2) and
    // `a^b^c` is a^(b^c). The exponent re-enters at parse_unary so `2^-1` parses.
    String parse_power() {
        String base = parse_primary();
        if (current_matches({"^"})) {
            advance();
            const String exponent = parse_unary();
            return "pow(" + base + ", " + exponent + ")";
        }
        return base;
    }

    String parse_primary() {
        const Token token = current();

        if (token.kind == TokenKind::Number) {
            advance();
            return to_float_literal(token.text);
        }

        if (token.kind == TokenKind::Identifier) {
            advance();
            if (current().kind == TokenKind::LeftParenthesis) {
                return parse_function_call(token.text);
            }
            return symbols_.read_expression_for(token.text);
        }

        if (token.kind == TokenKind::LeftParenthesis) {
            advance();
            const String inner = parse_logical_or();
            if (current().kind != TokenKind::RightParenthesis) {
                fail("expected ')' in expression");
            }
            advance();
            // No parentheses are added back. A compound result already carries its own,
            // and re-wrapping would nest one pair per group the source happened to write.
            return inner;
        }

        if (token.kind == TokenKind::EndOfInput) fail("expression ended early");
        fail("expected a value but found '" + token.text + "'");
    }

    String parse_function_call(const String &function_name) {
        advance(); // past '('

        Vector<String> arguments;
        if (current().kind != TokenKind::RightParenthesis) {
            arguments.push_back(parse_logical_or());
            while (current().kind == TokenKind::Comma) {
                advance();
                arguments.push_back(parse_logical_or());
            }
        }

        if (current().kind != TokenKind::RightParenthesis) {
            fail("expected ')' closing call to '" + function_name + "'");
        }
        advance();

        return emit_function_call(function_name, arguments);
    }

    String emit_function_call(const String &function_name, const Vector<String> &arguments) const {
        if (function_name == "random") {
            fail("unsupported function 'random': a deterministic per-neuron stream needs a "
                 "seed argument the generated kernels do not take");
        }

        // NeuroML's standard library onto the C-family names. `ln` and `log` are the trap:
        // LEMS's `log` is base 10 and its natural log is `ln`, which is the reverse of C.
        static const UnorderedMap<String, String> single_argument_functions = {
            {"exp", "exp"},   {"ln", "log"},    {"log", "log10"}, {"sin", "sin"},
            {"cos", "cos"},   {"tan", "tan"},   {"sinh", "sinh"}, {"cosh", "cosh"},
            {"tanh", "tanh"}, {"sqrt", "sqrt"}, {"abs", "fabs"},  {"ceil", "ceil"},
            {"floor", "floor"},
        };

        if (function_name == "H") {
            require_argument_count(function_name, arguments, 1);
            return "((" + arguments[0] + ") >= 0.0f ? 1.0f : 0.0f)";
        }

        const auto entry = single_argument_functions.find(function_name);
        if (entry == single_argument_functions.end()) {
            fail("unknown function '" + function_name + "'");
        }

        require_argument_count(function_name, arguments, 1);
        return entry->second + "(" + arguments[0] + ")";
    }

    void require_argument_count(const String &function_name, const Vector<String> &arguments,
                                usize expected_count) const {
        if (arguments.size() != expected_count) {
            fail("function '" + function_name + "' takes " + to_string(expected_count) +
                 " argument(s) but was given " + to_string(arguments.size()));
        }
    }
};

// ── Symbol tables ────────────────────────────────────────────────────────────

SymbolTable build_base_symbol_table(const CellTypeSpecification &cell_type) {
    SymbolTable symbols;
    symbols.component_type_name = cell_type.name;

    for (usize slot = 0; slot < cell_type.state_variable_names.size(); ++slot) {
        symbols.define(cell_type.state_variable_names[slot],
                       "cell_state[state_base + " + to_string(slot) + "]");
    }

    for (usize slot = 0; slot < cell_type.parameter_names.size(); ++slot) {
        symbols.define(cell_type.parameter_names[slot],
                       "cell_parameters[parameter_base + " + to_string(slot) + "]");
    }

    return symbols;
}

// Global constants and built-ins sit below derived locals in the precedence chain, and
// derived locals do not exist yet when the base table is built. Layering the low-priority
// names onto a copy at the point of use keeps the chain ordered however many derived names
// have accumulated by then.
SymbolTable with_fallback_symbols(const SymbolTable &resolved_so_far,
                                  const NML_ParseResult &parse_result) {
    SymbolTable complete = resolved_so_far;

    // A ComponentType's own <Constant>s are keyed "<TypeName>.<name>" so two types cannot
    // collide, while document-scope ones are keyed bare. Expressions write the bare name
    // either way, so the type's own constants are registered first and shadow the
    // document-scope ones.
    const String type_prefix = resolved_so_far.component_type_name + ".";
    for (const auto &constant_entry : parse_result.global_constants) {
        if (constant_entry.first.rfind(type_prefix, 0) == 0) {
            complete.define(constant_entry.first.substr(type_prefix.length()),
                            format_float_literal(constant_entry.second.float64));
        }
    }
    for (const auto &constant_entry : parse_result.global_constants) {
        complete.define(constant_entry.first, format_float_literal(constant_entry.second.float64));
    }

    complete.define("t", "(dt * (float)tick)");
    complete.define("dt", "dt");

    return complete;
}

// ── Emission helpers ─────────────────────────────────────────────────────────

String indent(usize level) {
    return String(level * 4, ' ');
}

// NML names are XML names and may carry characters C identifiers cannot, so anything
// reaching generated source is folded to [A-Za-z0-9_].
String sanitize_identifier(const String &name) {
    String result;
    result.reserve(name.length());
    for (const char character : name) {
        result.push_back(is_identifier_character(character) ? character : '_');
    }
    if (result.empty() || is_digit_character(result[0])) result.insert(result.begin(), '_');
    return result;
}

Optional<usize> find_state_variable_slot(const CellTypeSpecification &cell_type,
                                         const String &name) {
    for (usize slot = 0; slot < cell_type.state_variable_names.size(); ++slot) {
        if (cell_type.state_variable_names[slot] == name) return slot;
    }
    return nullopt;
}

usize require_state_variable_slot(const CellTypeSpecification &cell_type, const String &name,
                                  const String &written_by) {
    const Optional<usize> slot = find_state_variable_slot(cell_type, name);
    if (!slot.has_value()) {
        report_error(written_by + " writes '" + name + "', which is not a StateVariable of this type",
                     cell_type.name);
    }
    return *slot;
}

// Everything this generator refuses to lower, checked in one pass so a model fails at
// generation with the construct named rather than half-generating and running wrong.
void reject_unsupported_instructions(const CellTypeSpecification &cell_type) {
    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        // A StateVariable is a declaration, not an operation: it lands on the RegimeEntry
        // stage only because that is stage_for_declaration's default. Its slot is already
        // known from state_variable_names, so it is skipped rather than rejected.
        if (instruction.source_tag == NML_DeclarationType::StateVariable) continue;

        if (!instruction.regime_name.empty()) {
            report_error("regimes are not supported: an instruction targeting '" +
                                 instruction.target + "' sits inside Regime '" +
                                 instruction.regime_name + "'",
                         cell_type.name);
        }

        switch (instruction.source_tag) {
            case NML_DeclarationType::Regime:
                report_error("Regime '" + instruction.target + "' is not supported", cell_type.name);
            case NML_DeclarationType::Transition:
                report_error("Transition to regime '" + instruction.target + "' is not supported",
                             cell_type.name);
            case NML_DeclarationType::OnEntry:
                report_error("OnEntry is not supported", cell_type.name);
            case NML_DeclarationType::OnEvent:
                report_error("OnEvent on port '" + instruction.target + "' is not supported",
                             cell_type.name);
            case NML_DeclarationType::ConditionalDerivedVariable:
                report_error("ConditionalDerivedVariable '" + instruction.target +
                                     "' is not supported: DynamicsInstruction carries each Case's "
                                     "value but not its condition, so the guards never reach codegen",
                             cell_type.name);
            case NML_DeclarationType::Case:
                report_error("Case is not supported: its condition attribute is not carried by "
                             "DynamicsInstruction",
                             cell_type.name);
            default:
                break;
        }

        if (instruction.stage == DynamicsStage::Arrival) {
            report_error("event arrival handling is not supported (instruction targeting '" +
                                 instruction.target + "')",
                         cell_type.name);
        }
        if (instruction.stage == DynamicsStage::RegimeEntry) {
            report_error("regime entry handling is not supported (instruction targeting '" +
                                 instruction.target + "')",
                         cell_type.name);
        }
    }
}

String emit_state_assignment(const CellTypeSpecification &cell_type,
                             const NML_ParseResult &parse_result, const SymbolTable &symbols,
                             const DynamicsInstruction &instruction, usize indent_level) {
    const usize slot = require_state_variable_slot(cell_type, instruction.target, "StateAssignment");
    return indent(indent_level) + "cell_state[state_base + " + to_string(slot) + "] = " +
           translate_expression(instruction.expression,
                                with_fallback_symbols(symbols, parse_result)) +
           ";\n";
}

String emit_spike(usize indent_level) {
    return indent(indent_level) + "spike_flags[neuron_index] = 1;\n" + indent(indent_level) +
           "last_spiked[neuron_index] = tick;\n";
}

// ── Per-cell-type bodies ─────────────────────────────────────────────────────

String emit_tick_body(const CellTypeSpecification &cell_type, const NML_ParseResult &parse_result) {
    reject_unsupported_instructions(cell_type);

    ostringstream body;
    SymbolTable symbols = build_base_symbol_table(cell_type);

    // Integrate, part 1: DerivedVariables become locals in source order, each visible to
    // the ones after it. A DerivedVariable referencing one declared later resolves to
    // nothing and throws, rather than emitting a forward reference the shader compiler
    // would reject far from its cause.
    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.stage != DynamicsStage::Integrate) continue;
        if (instruction.source_tag != NML_DeclarationType::DerivedVariable) continue;

        const String local_name = "derived_" + sanitize_identifier(instruction.target);
        body << indent(1) << "float " << local_name << " = "
             << translate_expression(instruction.expression,
                                     with_fallback_symbols(symbols, parse_result))
             << ";\n";
        symbols.define(instruction.target, local_name);
    }

    // Integrate, part 2: forward Euler into a temporary per state variable. Nothing is
    // written back until every derivative has been computed, so two variables that
    // reference each other both integrate from the state as it stood at entry.
    Vector<Pair<usize, String>> pending_state_writes;
    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.stage != DynamicsStage::Integrate) continue;
        if (instruction.source_tag != NML_DeclarationType::TimeDerivative) continue;

        const usize slot =
                require_state_variable_slot(cell_type, instruction.target, "TimeDerivative");
        const String temporary_name = "next_" + sanitize_identifier(instruction.target);

        body << indent(1) << "float " << temporary_name << " = cell_state[state_base + " << slot
             << "] + dt * ("
             << translate_expression(instruction.expression,
                                     with_fallback_symbols(symbols, parse_result))
             << ");\n";
        pending_state_writes.push_back({slot, temporary_name});
    }
    for (const auto &pending_write : pending_state_writes) {
        body << indent(1) << "cell_state[state_base + " << pending_write.first
             << "] = " << pending_write.second << ";\n";
    }

    // Anything with no gate at all runs unconditionally. Well-formed LEMS puts every
    // StateAssignment and EventOut inside a handler, so this is close to dead, but
    // dropping such an instruction would be a silent omission.
    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        if (!instruction.condition.empty()) continue;
        if (instruction.stage == DynamicsStage::Reset) {
            body << emit_state_assignment(cell_type, parse_result, symbols, instruction, 1);
        } else if (instruction.stage == DynamicsStage::Emit) {
            body << emit_spike(1);
        }
    }

    // Detect, then the Reset and Emit bodies each OnCondition gates. Reset and Emit
    // instructions carry their OnCondition's test verbatim in `condition`, so that string
    // is the join key back to the condition that fired them.
    Set<String> emitted_condition_tests;
    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.stage != DynamicsStage::Detect) continue;

        // Two OnConditions with identical tests would otherwise each pick up the other's
        // body, duplicating every assignment. They fire together by definition, so the
        // first block already carries both.
        if (!emitted_condition_tests.insert(instruction.expression).second) continue;

        body << indent(1) << "if ("
             << translate_expression(instruction.expression,
                                     with_fallback_symbols(symbols, parse_result))
             << ") {\n";

        for (const DynamicsInstruction &gated : cell_type.dynamics) {
            if (gated.condition != instruction.expression) continue;
            if (gated.stage == DynamicsStage::Reset) {
                body << emit_state_assignment(cell_type, parse_result, symbols, gated, 2);
            } else if (gated.stage == DynamicsStage::Emit) {
                body << emit_spike(2);
            }
        }

        body << indent(1) << "}\n";
    }

    // A gated instruction whose gate never appeared would silently never run.
    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        const bool is_gated_body = instruction.stage == DynamicsStage::Reset ||
                                   instruction.stage == DynamicsStage::Emit;
        if (!is_gated_body || instruction.condition.empty()) continue;
        if (emitted_condition_tests.count(instruction.condition) == 0) {
            report_error("instruction targeting '" + instruction.target + "' is gated by '" +
                                 instruction.condition + "', which matches no OnCondition",
                         cell_type.name);
        }
    }

    return body.str();
}

String emit_initialize_body(const CellTypeSpecification &cell_type,
                            const NML_ParseResult &parse_result) {
    ostringstream body;
    const SymbolTable symbols = build_base_symbol_table(cell_type);

    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.stage != DynamicsStage::Initialize) continue;
        // The OnStart element itself also lands on this stage carrying nothing; only the
        // StateAssignments under it initialise anything.
        if (instruction.source_tag != NML_DeclarationType::StateAssignment) continue;

        body << emit_state_assignment(cell_type, parse_result, symbols, instruction, 1);
    }

    return body.str();
}

// ── Kernel assembly ──────────────────────────────────────────────────────────

const Vector<String> &kernel_argument_names() {
    static const Vector<String> names = {
        "cell_state",      "cell_parameters",     "network_inputs",  "last_spiked",
        "spike_flags",     "cell_state_base",     "cell_parameter_base", "cell_type_index",
        "neuron_count",    "dt",                  "tick",
    };
    return names;
}

String kernel_function_name(KernelPurpose purpose) {
    return purpose == KernelPurpose::Tick ? "simulate_tick" : "initialize_cell_state";
}

String device_function_name(const String &cell_type_name, KernelPurpose purpose) {
    const String prefix =
            purpose == KernelPurpose::Tick ? "cell_type_step_" : "cell_type_initialize_";
    return prefix + sanitize_identifier(cell_type_name);
}

String source_preamble(KernelBackend backend, KernelPurpose purpose) {
    ostringstream preamble;
    preamble << "// Generated by spikecorec::nml::"
             << (purpose == KernelPurpose::Tick ? "generate_tick_kernel"
                                                : "generate_initialize_kernel")
             << " -- do not edit.\n";
    if (backend == KernelBackend::Metal) {
        preamble << "#include <metal_stdlib>\n";
        preamble << "using namespace metal;\n";
    }
    preamble << "\n";
    return preamble.str();
}

String emit_device_function(const String &function_name, const String &body,
                            KernelBackend backend) {
    const bool is_metal = backend == KernelBackend::Metal;
    const String address_space = is_metal ? "device " : "";
    const String tick_type = is_metal ? "long" : "long long";

    ostringstream function;
    function << (is_metal ? "inline void " : "__device__ inline void ") << function_name << "(\n";
    function << indent(2) << address_space << "float *cell_state,\n";
    function << indent(2) << address_space << "const float *cell_parameters,\n";
    function << indent(2) << address_space << "float *network_inputs,\n";
    function << indent(2) << address_space << tick_type << " *last_spiked,\n";
    function << indent(2) << address_space << "int *spike_flags,\n";
    function << indent(2) << "int state_base,\n";
    function << indent(2) << "int parameter_base,\n";
    function << indent(2) << "int neuron_index,\n";
    function << indent(2) << "float dt,\n";
    function << indent(2) << tick_type << " tick\n";
    function << ") {\n";
    function << body;
    function << "}\n\n";
    return function.str();
}

String emit_master_kernel(const Vector<String> &device_function_names, KernelBackend backend,
                          KernelPurpose purpose) {
    ostringstream kernel;

    if (backend == KernelBackend::Metal) {
        kernel << "kernel void " << kernel_function_name(purpose) << "(\n";
        kernel << indent(1) << "device float       *cell_state          [[ buffer(0) ]],\n";
        kernel << indent(1) << "device const float *cell_parameters     [[ buffer(1) ]],\n";
        kernel << indent(1) << "device float       *network_inputs      [[ buffer(2) ]],\n";
        kernel << indent(1) << "device long        *last_spiked         [[ buffer(3) ]],\n";
        kernel << indent(1) << "device int         *spike_flags         [[ buffer(4) ]],\n";
        kernel << indent(1) << "device const int   *cell_state_base     [[ buffer(5) ]],\n";
        kernel << indent(1) << "device const int   *cell_parameter_base [[ buffer(6) ]],\n";
        kernel << indent(1) << "device const int   *cell_type_index     [[ buffer(7) ]],\n";
        kernel << indent(1) << "constant int       &neuron_count        [[ buffer(8) ]],\n";
        kernel << indent(1) << "constant float     &dt                  [[ buffer(9) ]],\n";
        kernel << indent(1) << "constant long      &tick                [[ buffer(10) ]],\n";
        kernel << indent(1) << "uint thread_id [[ thread_position_in_grid ]]\n";
        kernel << ") {\n";
        kernel << indent(1) << "int neuron_index = (int)thread_id;\n";
    } else {
        kernel << "extern \"C\" __global__ void " << kernel_function_name(purpose) << "(\n";
        kernel << indent(1) << "float *cell_state,\n";
        kernel << indent(1) << "const float *cell_parameters,\n";
        kernel << indent(1) << "float *network_inputs,\n";
        kernel << indent(1) << "long long *last_spiked,\n";
        kernel << indent(1) << "int *spike_flags,\n";
        kernel << indent(1) << "const int *cell_state_base,\n";
        kernel << indent(1) << "const int *cell_parameter_base,\n";
        kernel << indent(1) << "const int *cell_type_index,\n";
        kernel << indent(1) << "int neuron_count,\n";
        kernel << indent(1) << "float dt,\n";
        kernel << indent(1) << "long long tick\n";
        kernel << ") {\n";
        kernel << indent(1) << "int neuron_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n";
    }

    kernel << indent(1) << "if (neuron_index >= neuron_count) return;\n\n";
    kernel << indent(1) << "int state_base = cell_state_base[neuron_index];\n";
    kernel << indent(1) << "int parameter_base = cell_parameter_base[neuron_index];\n\n";
    kernel << indent(1) << "switch (cell_type_index[neuron_index]) {\n";

    for (usize type_index = 0; type_index < device_function_names.size(); ++type_index) {
        kernel << indent(2) << "case " << type_index << ":\n";
        kernel << indent(3) << device_function_names[type_index]
               << "(cell_state, cell_parameters, network_inputs, last_spiked, spike_flags, "
                  "state_base, parameter_base, neuron_index, dt, tick);\n";
        kernel << indent(3) << "break;\n";
    }

    kernel << indent(2) << "default:\n";
    kernel << indent(3) << "break;\n";
    kernel << indent(1) << "}\n";
    kernel << "}\n";

    return kernel.str();
}

GeneratedKernel generate_kernel(const NML_ParseResult &parse_result, KernelBackend backend,
                                KernelPurpose purpose) {
    ostringstream source;
    source << source_preamble(backend, purpose);

    Vector<String> device_function_names;
    Set<String> used_function_names;

    for (const CellTypeSpecification &cell_type : parse_result.cell_types) {
        const String function_name = device_function_name(cell_type.name, purpose);

        // Sanitising two distinct type names can land them on one C identifier ("a-b" and
        // "a_b"), which would silently give one type the other's dynamics.
        if (!used_function_names.insert(function_name).second) {
            report_error("cell type name collides with another type's after sanitising to '" +
                                 function_name + "'",
                         cell_type.name);
        }

        const String body = purpose == KernelPurpose::Tick
                                    ? emit_tick_body(cell_type, parse_result)
                                    : emit_initialize_body(cell_type, parse_result);

        source << emit_device_function(function_name, body, backend);
        device_function_names.push_back(function_name);
    }

    source << emit_master_kernel(device_function_names, backend, purpose);

    GeneratedKernel generated;
    generated.source = source.str();
    generated.function_name = kernel_function_name(purpose);
    generated.argument_names = kernel_argument_names();
    return generated;
}

} // namespace

// ── Public interface ─────────────────────────────────────────────────────────

String translate_expression(const String &nml_expression, const SymbolTable &symbols) {
    Vector<Token> tokens = tokenize_nml_expression(nml_expression, symbols.component_type_name);
    if (tokens.size() == 1) {
        report_error("empty expression", symbols.component_type_name);
    }

    ExpressionParser parser(std::move(tokens), symbols);
    return parser.parse_complete_expression();
}

GeneratedKernel generate_tick_kernel(const NML_ParseResult &parse_result) {
    return generate_kernel(parse_result, active_backend(), KernelPurpose::Tick);
}

GeneratedKernel generate_initialize_kernel(const NML_ParseResult &parse_result) {
    return generate_kernel(parse_result, active_backend(), KernelPurpose::Initialize);
}

} // namespace spikecorec::nml
