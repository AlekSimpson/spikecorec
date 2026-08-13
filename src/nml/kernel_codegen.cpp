#include "spikecorec/nml/kernel_codegen.h"

#include <algorithm>
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

// Where one ComponentType's state variables live in generated source. Cells and synapses
// differ only in the array and the base offset they index it by, so everything that reads
// or writes state -- the symbol table, the simultaneous-assignment lowering, the Euler
// write-back -- is written once against this rather than twice against two layouts.
struct StateStorage {
    String component_type_name;
    Vector<String> state_variable_names;
    String array_name; // "cell_state" / "synapse_state"
    String base_name;  // "state_base" / "synapse_state_base"

    String element(usize slot) const {
        return array_name + "[" + base_name + " + " + to_string(slot) + "]";
    }
};

StateStorage cell_state_storage(const CellTypeSpecification &cell_type) {
    return StateStorage{cell_type.name, cell_type.state_variable_names, "cell_state",
                        "state_base"};
}

SymbolTable build_base_symbol_table(const CellTypeSpecification &cell_type) {
    const StateStorage storage = cell_state_storage(cell_type);

    SymbolTable symbols;
    symbols.component_type_name = cell_type.name;

    for (usize slot = 0; slot < cell_type.state_variable_names.size(); ++slot) {
        symbols.define(cell_type.state_variable_names[slot], storage.element(slot));
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

// The type every flat offset into a model-sized buffer is computed in, typedef'd into the
// generated source by source_preamble. 64 bits, matching the s64 the host sizes and indexes
// those same allocations with: in `int` the two disagree silently past INT_MAX and an offset
// wraps negative, which reads another neuron's slot rather than faulting. A ring of depth 64
// over 4 wired synapse prototypes and 8.4M neurons already crosses it.
const String buffer_index_type_name = "SpikecorecBufferIndex";

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

usize require_state_variable_slot(const StateStorage &storage, const String &name,
                                  const String &written_by) {
    for (usize slot = 0; slot < storage.state_variable_names.size(); ++slot) {
        if (storage.state_variable_names[slot] == name) return slot;
    }
    report_error(written_by + " writes '" + name + "', which is not a StateVariable of this type",
                 storage.component_type_name);
}

// ── Regimes ──────────────────────────────────────────────────────────────────
//
// See kernel_codegen.h for what a regime compiles into and why. Everything here answers one
// of three questions: which regimes exist and what index each has, which regime declares a
// TimeDerivative for a given variable, and what a regime's OnEntry body is.

// The local every guard reads. Loaded once, before anything else in the cell body, so a
// Transition taken this tick is observed from the NEXT tick on -- which is what stops the
// regime a cell just moved INTO from also running its own OnCondition in the same tick.
const String active_regime_local_name = "active_regime_index";

// The TimeDerivative `regime_name` declares for `variable_name`, or nullptr when that regime
// declares none -- which is exactly what freezes the variable while the regime is active.
const DynamicsInstruction *find_regime_time_derivative(const CellTypeSpecification &cell_type,
                                                       const String &regime_name,
                                                       const String &variable_name) {
    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.stage != DynamicsStage::Integrate) continue;
        if (instruction.source_tag != NML_DeclarationType::TimeDerivative) continue;
        if (instruction.regime_name != regime_name) continue;
        if (instruction.target != variable_name) continue;
        return &instruction;
    }
    return nullptr;
}

// `regime_name`'s OnEntry StateAssignments, in declaration order. An OnEntry body is located
// by stage rather than by a gate: collect_dynamics_instructions tags what it contains as
// RegimeEntry and clears the condition, which is what separates it from the Reset-stage
// assignments an OnCondition fires.
Vector<const DynamicsInstruction *> regime_entry_assignments(
        const CellTypeSpecification &cell_type, const String &regime_name) {
    Vector<const DynamicsInstruction *> assignments;
    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.stage != DynamicsStage::RegimeEntry) continue;
        if (instruction.source_tag != NML_DeclarationType::StateAssignment) continue;
        if (instruction.regime_name != regime_name) continue;
        assignments.push_back(&instruction);
    }
    return assignments;
}

// Whether any regime declares a TimeDerivative for `variable_name`, which is what decides
// that the variable is integrated through a regime dispatch rather than unconditionally.
bool any_regime_declares_time_derivative(const CellTypeSpecification &cell_type,
                                         const String &variable_name) {
    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.stage != DynamicsStage::Integrate) continue;
        if (instruction.source_tag != NML_DeclarationType::TimeDerivative) continue;
        if (instruction.regime_name.empty()) continue;
        if (instruction.target == variable_name) return true;
    }
    return false;
}

// The regime index, written into and read back out of the cell's own state chunk. Small
// integers are exact in f32, so the round trip is exact and `==` against a literal index is
// a real equality rather than an approximate one.
String regime_state_element(const CellRegimeLayout &regimes) {
    return "cell_state[state_base + " + to_string(regimes.regime_state_slot) + "]";
}

// Everything this generator refuses to lower, checked in one pass so a model fails at
// generation with the construct named rather than half-generating and running wrong.
void reject_unsupported_instructions(const CellTypeSpecification &cell_type,
                                     const CellRegimeLayout &regimes) {
    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        // A StateVariable is a declaration, not an operation: it lands on the RegimeEntry
        // stage only because that is stage_for_declaration's default. Its slot is already
        // known from state_variable_names, so it is skipped rather than rejected.
        if (instruction.source_tag == NML_DeclarationType::StateVariable) continue;

        // A regime-scoped instruction whose regime resolve_cell_regimes never saw cannot
        // happen through the parser -- the name comes from the enclosing Regime -- but it
        // would silently generate a guard against an index no Transition ever writes.
        if (!instruction.regime_name.empty() &&
            regimes.index_of(instruction.regime_name) < 0) {
            report_error("an instruction targeting '" + instruction.target +
                                 "' names Regime '" + instruction.regime_name +
                                 "', which this ComponentType does not declare",
                         cell_type.name);
        }

        switch (instruction.source_tag) {
            case NML_DeclarationType::Regime:
                // LEMS has no nested regimes, and a nested one would need its own index and
                // its own dispatch. Caught here rather than producing a flat chain that
                // silently discards the nesting.
                if (!instruction.regime_name.empty()) {
                    report_error("Regime '" + instruction.target + "' is nested inside Regime '" +
                                         instruction.regime_name + "', which is not supported",
                                 cell_type.name);
                }
                continue;
            case NML_DeclarationType::Transition:
                // A Transition compiles to a store guarded by the OnCondition that fired it.
                // One sitting directly under a Regime has no gate at all, so it would run
                // every tick and the cell would never stay anywhere.
                if (instruction.regime_name.empty() || instruction.condition.empty()) {
                    report_error("Transition to regime '" + instruction.target +
                                         "' is not inside a Regime's OnCondition; it has no "
                                         "gate, so it would fire on every tick",
                                 cell_type.name);
                }
                if (regimes.index_of(instruction.target) < 0) {
                    report_error("Transition names regime '" + instruction.target +
                                         "', which this ComponentType does not declare",
                                 cell_type.name);
                }
                continue;
            case NML_DeclarationType::OnEntry:
                if (instruction.regime_name.empty()) {
                    report_error("OnEntry is not inside a Regime, so there is no entry for it "
                                 "to run on",
                                 cell_type.name);
                }
                continue;
            case NML_DeclarationType::DerivedVariable:
                // A regime-scoped DerivedVariable would be emitted as an ordinary local,
                // visible and evaluated in every regime -- which is not what declaring it
                // inside one means.
                if (!instruction.regime_name.empty()) {
                    report_error("DerivedVariable '" + instruction.target +
                                         "' is declared inside Regime '" +
                                         instruction.regime_name +
                                         "'; only TimeDerivative, OnCondition and OnEntry are "
                                         "lowered inside a regime",
                                 cell_type.name);
                }
                break;
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

        // What is left on the RegimeEntry stage after the tags handled above is an OnEntry
        // body's StateAssignments -- located by stage, since OnEntry clears the condition
        // its children carry. Anything else there is a construct with no lowering.
        if (instruction.stage == DynamicsStage::RegimeEntry) {
            if (instruction.source_tag != NML_DeclarationType::StateAssignment) {
                report_error("regime entry handling is not supported for this construct "
                             "(instruction targeting '" +
                                     instruction.target + "')",
                             cell_type.name);
            }
            if (instruction.regime_name.empty()) {
                report_error("StateAssignment to '" + instruction.target +
                                     "' sits in an OnEntry outside any Regime",
                             cell_type.name);
            }
            continue;
        }

        // A StateAssignment or EventOut directly under a Regime, with no OnCondition around
        // it, has no gate. Emitting it unconditionally would run it in every regime; emitting
        // it under the regime guard would invent a handler the document never wrote.
        const bool is_gated_body = instruction.stage == DynamicsStage::Reset ||
                                   instruction.stage == DynamicsStage::Emit;
        if (is_gated_body && !instruction.regime_name.empty() && instruction.condition.empty()) {
            report_error("an instruction targeting '" + instruction.target +
                                 "' sits directly inside Regime '" + instruction.regime_name +
                                 "' with no OnCondition around it, so nothing gates it",
                         cell_type.name);
        }
    }
}

// One handler's StateAssignments, emitted with LEMS's simultaneous semantics: every
// right-hand side is evaluated against the state as it stood when the handler fired, and
// only then is anything written back. Emitting them sequentially would make "v = u; u = v"
// inside one OnCondition assign u to itself instead of swapping, and the ordering would be
// the document's rather than the model's. This is the same treatment the TimeDerivative path
// already gives its next_* temporaries.
//
// `temporary_name_prefix` separates one handler's temporaries from another's. Two handlers
// can now land in the SAME generated block -- an OnCondition's own assignments and the
// OnEntry assignments of the regime its Transition moves to -- and if both wrote a variable
// the two `float assigned_v` declarations would collide.
String emit_state_assignment_group(const StateStorage &storage,
                                   const NML_ParseResult &parse_result,
                                   const SymbolTable &symbols,
                                   const Vector<const DynamicsInstruction *> &assignments,
                                   usize indent_level,
                                   const String &temporary_name_prefix = "assigned_") {
    if (assignments.empty()) return "";

    const SymbolTable visible_symbols = with_fallback_symbols(symbols, parse_result);

    // A lone assignment cannot observe its own write, so it goes straight to storage. Every
    // GLIF handler has exactly one, and a temporary there would be noise in every generated
    // kernel Phase 1 produces.
    if (assignments.size() == 1) {
        const usize slot = require_state_variable_slot(storage, assignments.front()->target,
                                                       "StateAssignment");
        return indent(indent_level) + storage.element(slot) + " = " +
               translate_expression(assignments.front()->expression, visible_symbols) + ";\n";
    }

    ostringstream group;
    Vector<Pair<usize, String>> pending_state_writes;
    Set<String> declared_temporaries;

    for (const DynamicsInstruction *assignment : assignments) {
        const usize slot =
                require_state_variable_slot(storage, assignment->target, "StateAssignment");
        const String temporary_name =
                temporary_name_prefix + sanitize_identifier(assignment->target);

        // Two assignments to one variable in a single handler is malformed LEMS, but it must
        // not turn into a redeclaration the shader compiler rejects: the second one assigns
        // the existing temporary, so the last writer still wins as it did before.
        const bool is_first_assignment_to_target = declared_temporaries.insert(temporary_name).second;
        group << indent(indent_level) << (is_first_assignment_to_target ? "float " : "")
              << temporary_name << " = "
              << translate_expression(assignment->expression, visible_symbols) << ";\n";

        if (is_first_assignment_to_target) pending_state_writes.push_back({slot, temporary_name});
    }

    for (const auto &pending_write : pending_state_writes) {
        group << indent(indent_level) << storage.element(pending_write.first) << " = "
              << pending_write.second << ";\n";
    }

    return group.str();
}

// Forward Euler over every regime-free TimeDerivative in `dynamics`, into a temporary per
// state variable. Nothing is written back until every derivative has been computed, so two
// variables that reference each other both integrate from the state as it stood at entry.
//
// Regime-scoped derivatives are appended by emit_regime_dispatched_integration below, into
// the same pending list, so the one write-back covers both kinds.
String emit_unconditional_euler_temporaries(
        const StateStorage &storage, const Vector<DynamicsInstruction> &dynamics,
        const NML_ParseResult &parse_result, const SymbolTable &symbols, usize indent_level,
        Vector<Pair<usize, String>> &pending_state_writes) {
    const SymbolTable visible_symbols = with_fallback_symbols(symbols, parse_result);

    ostringstream step;
    Set<usize> already_integrated_slots;

    for (const DynamicsInstruction &instruction : dynamics) {
        if (instruction.stage != DynamicsStage::Integrate) continue;
        if (instruction.source_tag != NML_DeclarationType::TimeDerivative) continue;
        if (!instruction.regime_name.empty()) continue;

        const usize slot =
                require_state_variable_slot(storage, instruction.target, "TimeDerivative");

        // Two TimeDerivatives for one variable is malformed LEMS. Left alone it emits two
        // `float next_v` declarations and fails inside the shader compiler, far from its
        // cause; refused here with the variable named.
        if (!already_integrated_slots.insert(slot).second) {
            report_error("'" + instruction.target +
                                 "' carries more than one TimeDerivative outside any Regime",
                         storage.component_type_name);
        }

        const String temporary_name = "next_" + sanitize_identifier(instruction.target);
        step << indent(indent_level) << "float " << temporary_name << " = "
             << storage.element(slot) << " + dt * ("
             << translate_expression(instruction.expression, visible_symbols) << ");\n";
        pending_state_writes.push_back({slot, temporary_name});
    }

    return step.str();
}

// The regime dispatch, one chain per state variable any regime declares a TimeDerivative
// for, in the type's own state variable order.
//
// The temporary is seeded with the variable's current value, so a regime that declares no
// derivative for it needs to emit NOTHING in its branch: the variable simply keeps what it
// came in with. That absence is the whole mechanism -- GLIF's refractory period is `v`
// having no TimeDerivative in the refractory regime, not a zero derivative and not a flag.
String emit_regime_dispatched_integration(const CellTypeSpecification &cell_type,
                                          const CellRegimeLayout &regimes,
                                          const StateStorage &storage,
                                          const NML_ParseResult &parse_result,
                                          const SymbolTable &symbols, usize indent_level,
                                          Vector<Pair<usize, String>> &pending_state_writes) {
    if (!regimes.has_regimes()) return "";

    const SymbolTable visible_symbols = with_fallback_symbols(symbols, parse_result);

    ostringstream dispatch;
    for (usize slot = 0; slot < storage.state_variable_names.size(); ++slot) {
        const String &variable_name = storage.state_variable_names[slot];
        if (!any_regime_declares_time_derivative(cell_type, variable_name)) continue;

        // Which of the two applies is not decidable from the document, and picking either
        // silently changes how the variable moves in every regime.
        for (const auto &pending_write : pending_state_writes) {
            if (pending_write.first != slot) continue;
            report_error("'" + variable_name +
                                 "' carries both a regime-scoped TimeDerivative and one outside "
                                 "any Regime",
                         cell_type.name);
        }

        const String temporary_name = "next_" + sanitize_identifier(variable_name);
        dispatch << indent(indent_level) << "float " << temporary_name << " = "
                 << storage.element(slot) << ";\n";

        // One regime's body for this variable: the Euler step it declares, or a comment
        // recording that it declares none and the variable is therefore held.
        auto branch_body = [&](const String &regime_name, usize body_indent) -> String {
            const DynamicsInstruction *derivative =
                    find_regime_time_derivative(cell_type, regime_name, variable_name);
            if (derivative == nullptr) {
                return indent(body_indent) + "// Regime '" + regime_name +
                       "' declares no TimeDerivative for '" + variable_name +
                       "', so it holds its value.\n";
            }
            return indent(body_indent) + temporary_name + " = " + storage.element(slot) +
                   " + dt * (" + translate_expression(derivative->expression, visible_symbols) +
                   ");\n";
        };

        // A single regime is always active, so its body needs no guard at all.
        if (regimes.regime_names.size() == 1) {
            dispatch << branch_body(regimes.regime_names[0], indent_level);
            pending_state_writes.push_back({slot, temporary_name});
            continue;
        }

        // The first N-1 regimes get an explicit index comparison; the last is the trailing
        // else, so every index the slot can hold lands in exactly one branch.
        for (usize regime_index = 0; regime_index + 1 < regimes.regime_names.size();
             ++regime_index) {
            dispatch << (regime_index == 0 ? indent(indent_level) : " ") << "if ("
                     << active_regime_local_name << " == " << regime_index << ") {\n"
                     << branch_body(regimes.regime_names[regime_index], indent_level + 1)
                     << indent(indent_level) << "} else";
        }
        dispatch << " {\n" << branch_body(regimes.regime_names.back(), indent_level + 1)
                 << indent(indent_level) << "}\n";

        pending_state_writes.push_back({slot, temporary_name});
    }

    return dispatch.str();
}

String emit_euler_write_back(const StateStorage &storage,
                             const Vector<Pair<usize, String>> &pending_state_writes,
                             usize indent_level) {
    ostringstream write_back;
    for (const auto &pending_write : pending_state_writes) {
        write_back << indent(indent_level) << storage.element(pending_write.first) << " = "
                   << pending_write.second << ";\n";
    }
    return write_back.str();
}

// The Reset-stage StateAssignments the OnCondition `condition`, declared inside regime
// `regime_name`, gates -- in declaration order. Both halves of the key matter: two regimes
// may declare OnConditions with identical tests, and matching on the test alone would give
// each of them the other's assignments as well. An empty pair collects the ungated,
// regime-free ones.
Vector<const DynamicsInstruction *> reset_assignments_gated_by(
        const CellTypeSpecification &cell_type, const String &regime_name,
        const String &condition) {
    Vector<const DynamicsInstruction *> assignments;
    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.stage != DynamicsStage::Reset) continue;
        if (instruction.regime_name != regime_name) continue;
        if (instruction.condition != condition) continue;
        assignments.push_back(&instruction);
    }
    return assignments;
}

String emit_spike(usize indent_level) {
    return indent(indent_level) + "spike_flags[neuron_index] = 1;\n" + indent(indent_level) +
           "last_spiked[neuron_index] = tick;\n";
}

// ── select= paths ────────────────────────────────────────────────────────────
//
// A DerivedVariable carries either an arithmetic `value=` or a `select=` path, and
// nml.cpp's collect_dynamics_instructions folds both attributes into `expression`. Nothing
// downstream records which one it was, so the two are told apart here, by shape, before the
// tokenizer runs -- a path is not an expression and would only report as a malformed one.

// Splits on '/'. A "[ion='ca']" qualifier never contains one, so no escaping is involved.
Vector<String> split_path_segments(const String &path) {
    Vector<String> segments;
    usize segment_start = 0;

    while (true) {
        const usize separator_position = path.find('/', segment_start);
        if (separator_position == String::npos) {
            segments.push_back(path.substr(segment_start));
            return segments;
        }
        segments.push_back(path.substr(segment_start, separator_position - segment_start));
        segment_start = separator_position + 1;
    }
}

// The name a segment selects, with any qualifier dropped: "synapses[*]" -> "synapses".
// nullopt when the text is not shaped like a segment, which is what keeps arithmetic out.
Optional<String> path_segment_name(const String &segment) {
    if (segment.empty() || !is_identifier_start(segment.front())) return nullopt;

    usize index = 0;
    while (index < segment.length() && is_identifier_character(segment[index])) index++;
    if (index == segment.length()) return segment;

    // The only thing that may follow the name is a qualifier closing at the segment's end:
    // "[*]", "[ion='ca']". Anything else -- an operator, whitespace -- is arithmetic.
    if (segment[index] != '[' || segment.back() != ']') return nullopt;
    return segment.substr(0, index);
}

// The head segment's name when `expression` is a select= path, nullopt when it is
// arithmetic to be parsed as usual.
//
// Shape alone does not separate the two: "iMemb/C" is a division and "ionChannel/g" is a
// path, and hindmarshRose1984Cell writes both forms in one Dynamics block. What separates
// them is what the head names. A path's head is a child or collection -- an Attachments or
// Child of the ComponentType, never one of its variables -- so a head that resolves to a
// readable value means the '/' was the division operator. A qualified head ("synapses[*]")
// settles it on its own, being arithmetic under no reading.
Optional<String> select_path_head_name(const String &expression, const SymbolTable &symbols) {
    const Vector<String> segments = split_path_segments(expression);
    if (segments.size() < 2) return nullopt;

    for (const String &segment : segments) {
        if (!path_segment_name(segment).has_value()) return nullopt;
    }

    const String head_name = *path_segment_name(segments.front());
    const bool head_is_qualified = head_name.length() < segments.front().length();
    if (!head_is_qualified && symbols.contains(head_name)) return nullopt;

    return head_name;
}

// The one local every select= path over the attached synapses binds to. The ring row is
// loaded into it once per cell, so a model that reduces over its synapses more than once
// reads one consistent delivered current rather than repeating the load.
const String synaptic_input_local_name = "synaptic_input_accumulator";

// Refuses a select= path that is not a reduction over the attached synapses.
//
// `network_inputs` is the engine's synaptic accumulator: a source scatters its weight into
// its target's slot and the target drains it when the delay has elapsed. A reduction over
// the attached synapses is exactly that quantity, so it lowers to a read of this tick's ring
// row. The selected exposure's name is deliberately not part of the test -- iafCell selects
// "i" and izhikevichCell selects "I" from the same one scalar per neuron -- and neither is
// the `reduce` attribute, which DynamicsInstruction does not carry.
//
// Every other path reaches into a child structure that has no engine buffer behind it, so
// it is refused by name rather than bound to the accumulator, which would run and be wrong.
void require_synaptic_select_path(const String &path, const String &head_name,
                                  const String &component_type_name) {
    if (head_name != "synapses") {
        report_error("unsupported select path \"" + path + "\": '" + head_name +
                             "' names a child structure with no engine buffer behind it. Only a "
                             "selection over 'synapses', which is the per-neuron synaptic input "
                             "accumulator, can be lowered",
                     component_type_name);
    }
}

// ── Stage 6, Propagate ───────────────────────────────────────────────────────
//
// Propagation is generated into every cell device function as a fixed epilogue rather than
// dispatched as a kernel of its own: the supporting engine infrastructure -- the k^2-tree
// adjacency, the shared U/V basis, the delay ring -- is identical for every cell type, so
// the same boilerplate serves all of them.
//
// `network_inputs` is a ring of `ring_depth` rows of `neuron_count` floats. A spiking source
// adds each edge's weight into row (tick + edge delay) % ring_depth of the target's column;
// a cell reads and clears row tick % ring_depth of its own column. `ring_depth` exceeds the
// model's largest per-edge delay (the engine computes it), so an arrival can never land in
// the row being drained this tick, and every row is cleared by its reader before the ring
// wraps back onto it.
//
// The row walk below is the same iterative DFS kernels.metal's k2t_next_neighbor performs,
// and enumerates a row's neighbours in exactly the order K2Tree::get_neighbors does on the
// host (both descend column offsets 0..branching_factor-1 depth first). That is what makes
// `neighbor_slot` address the same per-edge slot the host indexes the sparse delta and
// per-edge delay arrays by.

// Reconstruction reads U/V as plain floats rather than as float4: the buffers are the same
// bytes either way, and scalar lanes make the arithmetic identical to WeightMatrix's own
// host-side reconstruct_entry, which is what `weights.get()` reports. `rank_float4_stride`
// keeps its name because that is the argument the engine binds; a row is
// rank_float4_stride * 4 scalar lanes wide.
String emit_propagation_helpers(KernelBackend backend) {
    const bool is_metal = backend == KernelBackend::Metal;
    const String function_prefix = is_metal ? "inline " : "__device__ inline ";
    const String device_qualifier = is_metal ? "device " : "";
    const String thread_qualifier = is_metal ? "thread " : "";
    const String unsigned_integer_type = is_metal ? "uint" : "unsigned int";
    const String unsigned_short_type = is_metal ? "ushort" : "unsigned short";
    const String population_count_function = is_metal ? "popcount" : "__popc";
    const String tick_type = is_metal ? "long" : "long long";

    ostringstream helpers;

    helpers << "#define SPIKECOREC_MAXIMUM_K2TREE_HEIGHT 32\n\n";

    helpers << function_prefix << unsigned_integer_type << " k2tree_read_bit(\n"
            << indent(2) << device_qualifier << "const " << unsigned_integer_type << " *words,\n"
            << indent(2) << "int bit_index\n"
            << ") {\n"
            << indent(1) << "return (words[bit_index >> 5] >> (" << unsigned_integer_type
            << ")(bit_index & 31)) & 1u;\n"
            << "}\n\n";

    helpers << function_prefix << "int k2tree_rank_one_exclusive(\n"
            << indent(2) << device_qualifier << "const " << unsigned_integer_type
            << " *internal_node_words,\n"
            << indent(2) << device_qualifier << "const " << unsigned_integer_type
            << " *rank_superblock_table,\n"
            << indent(2) << device_qualifier << "const " << unsigned_short_type
            << " *rank_subblock_table,\n"
            << indent(2) << "int bit_position,\n"
            << indent(2) << "int superblock_size_words\n"
            << ") {\n"
            << indent(1) << "int word_index = bit_position >> 5;\n"
            << indent(1) << "int bit_offset = bit_position & 31;\n"
            << indent(1) << "int superblock_index = word_index / superblock_size_words;\n"
            << indent(1) << unsigned_integer_type
            << " superblock_base = rank_superblock_table[superblock_index];\n"
            << indent(1) << unsigned_integer_type << " subblock_base = ("
            << unsigned_integer_type << ")rank_subblock_table[word_index];\n"
            << indent(1) << unsigned_integer_type
            << " partial_word_mask = (bit_offset == 0) ? 0u : ((1u << ("
            << unsigned_integer_type << ")bit_offset) - 1u);\n"
            << indent(1) << unsigned_integer_type << " partial_word_population = "
            << population_count_function
            << "(internal_node_words[word_index] & partial_word_mask);\n"
            << indent(1)
            << "return (int)(superblock_base + subblock_base + partial_word_population);\n"
            << "}\n\n";

    // Resumes the caller's DFS and returns the next neighbour of `source_neuron_index`, or
    // -1 once the row is exhausted. The stack lives in the caller so one walk covers a whole
    // row -- O(degree * height) instead of one root-to-leaf descent per neighbour.
    helpers << function_prefix << "int k2tree_next_neighbor(\n"
            << indent(2) << device_qualifier << "const " << unsigned_integer_type
            << " *internal_node_words,\n"
            << indent(2) << device_qualifier << "const " << unsigned_integer_type
            << " *leaf_node_words,\n"
            << indent(2) << device_qualifier << "const " << unsigned_integer_type
            << " *rank_superblock_table,\n"
            << indent(2) << device_qualifier << "const " << unsigned_short_type
            << " *rank_subblock_table,\n"
            << indent(2) << "int branching_factor,\n"
            << indent(2) << "int superblock_size_words,\n"
            << indent(2) << "int neuron_count,\n"
            << indent(2) << "int tree_height,\n"
            << indent(2) << "int internal_bit_count,\n"
            << indent(2) << "int source_neuron_index,\n"
            << indent(2) << thread_qualifier << "int *stack_row_base,\n"
            << indent(2) << thread_qualifier << "int *stack_column_base,\n"
            << indent(2) << thread_qualifier << "int *stack_block_size,\n"
            << indent(2) << thread_qualifier << "int *stack_bit_offset,\n"
            << indent(2) << thread_qualifier << "int *stack_next_column,\n"
            << indent(2) << thread_qualifier << "int &stack_top\n"
            << ") {\n"
            << indent(1) << "int branching_factor_squared = branching_factor * branching_factor;\n\n"
            << indent(1) << "while (stack_top >= 0) {\n"
            << indent(2) << "int level = stack_top;\n"
            << indent(2) << "int column_offset = stack_next_column[level];\n"
            << indent(2) << "if (column_offset >= branching_factor) {\n"
            << indent(3) << "stack_top -= 1;\n"
            << indent(3) << "continue;\n"
            << indent(2) << "}\n"
            << indent(2) << "stack_next_column[level] = column_offset + 1;\n\n"
            << indent(2) << "int row_base = stack_row_base[level];\n"
            << indent(2) << "int column_base = stack_column_base[level];\n"
            << indent(2) << "int block_size = stack_block_size[level];\n"
            << indent(2) << "int level_bit_offset = stack_bit_offset[level];\n\n"
            << indent(2) << "int child_block_size = block_size / branching_factor;\n"
            << indent(2)
            << "int row_offset = (source_neuron_index - row_base) / child_block_size;\n"
            << indent(2)
            << "int child_flat_index = row_offset * branching_factor + column_offset;\n"
            << indent(2) << "int bit_position = level_bit_offset + child_flat_index;\n\n"
            << indent(2) << "if (level == tree_height - 1) {\n"
            << indent(3) << "if (k2tree_read_bit(leaf_node_words, bit_position)) {\n"
            << indent(4) << "int target_neuron_index = column_base + column_offset;\n"
            << indent(4)
            << "if (target_neuron_index < neuron_count) return target_neuron_index;\n"
            << indent(3) << "}\n"
            << indent(2)
            << "} else if (k2tree_read_bit(internal_node_words, bit_position)) {\n"
            << indent(3) << "int rank_inclusive = k2tree_rank_one_exclusive(\n"
            << indent(5) << "internal_node_words, rank_superblock_table, rank_subblock_table,\n"
            << indent(5) << "bit_position, superblock_size_words) + 1;\n"
            << indent(3) << "int child_level = stack_top + 1;\n"
            << indent(3) << "int raw_offset = branching_factor_squared * rank_inclusive;\n"
            << indent(3)
            << "stack_row_base[child_level] = row_base + row_offset * child_block_size;\n"
            << indent(3) << "stack_column_base[child_level] = column_base + column_offset * "
                            "child_block_size;\n"
            << indent(3) << "stack_block_size[child_level] = child_block_size;\n"
            << indent(3) << "stack_bit_offset[child_level] = (child_level == tree_height - 1)\n"
            << indent(5) << "? (raw_offset - internal_bit_count)\n"
            << indent(5) << ": raw_offset;\n"
            << indent(3) << "stack_next_column[child_level] = 0;\n"
            << indent(3) << "stack_top = child_level;\n"
            << indent(2) << "}\n"
            << indent(1) << "}\n"
            << indent(1) << "return -1;\n"
            << "}\n\n";

    // The ring is one flat allocation indexed as [row][plane][neuron]. `tick_of_arrival` is
    // the tick whose row is wanted -- `tick` for a drain, `tick + delay` for an arrival --
    // and `plane_index` is 0 for the delivered-current plane every cell reads, or 1 + p for
    // wired synapse prototype p's own arrival plane. The plane COUNT is baked rather than
    // passed: the master kernel's argument table is full, and it is a property of the model
    // the source was generated from.
    // The offset is computed in 64 bits, matching the host's own s64 arithmetic over the same
    // allocation (SpikeEngine's network_input_element_count and current_ring_row_base). In
    // `int` the two disagree silently past INT_MAX -- ring_depth 64 over 4 wired prototypes
    // and 8.4M neurons already exceeds it -- and an arrival wraps to a negative index, which
    // lands in another neuron's slot rather than crashing.
    helpers << function_prefix << tick_type << " network_input_ring_index(\n"
            << indent(2) << tick_type << " tick_of_arrival,\n"
            << indent(2) << "int plane_index,\n"
            << indent(2) << "int ring_depth,\n"
            << indent(2) << "int neuron_count,\n"
            << indent(2) << "int neuron_index\n"
            << ") {\n"
            << indent(1) << tick_type << " ring_row = tick_of_arrival % (" << tick_type
            << ")ring_depth;\n"
            << indent(1)
            << "return (ring_row * SPIKECOREC_NETWORK_INPUT_PLANE_COUNT + (" << tick_type
            << ")plane_index) *\n"
            << indent(3) << "(" << tick_type << ")neuron_count + (" << tick_type
            << ")neuron_index;\n"
            << "}\n\n";

    helpers << function_prefix << "void propagate_spike(\n"
            << indent(2) << device_qualifier << "float *network_inputs,\n"
            << indent(2) << device_qualifier << "const " << unsigned_integer_type
            << " *internal_node_words,\n"
            << indent(2) << device_qualifier << "const " << unsigned_integer_type
            << " *leaf_node_words,\n"
            << indent(2) << device_qualifier << "const " << unsigned_integer_type
            << " *rank_superblock_table,\n"
            << indent(2) << device_qualifier << "const " << unsigned_short_type
            << " *rank_subblock_table,\n"
            << indent(2) << device_qualifier << "const float *U_matrix,\n"
            << indent(2) << device_qualifier << "const float *V_matrix,\n"
            << indent(2) << device_qualifier << "const float *edge_weight_coefficients,\n"
            << indent(2) << device_qualifier << "const float *edge_weight_deltas,\n"
            << indent(2) << device_qualifier << "const int *edge_delay_ticks,\n"
            << indent(2) << device_qualifier << "const int *edge_synapse_plane,\n"
            << indent(2) << "int branching_factor,\n"
            << indent(2) << "int superblock_size_words,\n"
            << indent(2) << "int padded_node_count,\n"
            << indent(2) << "int tree_height,\n"
            << indent(2) << "int internal_bit_count,\n"
            << indent(2) << tick_type << " rank_float4_stride,\n"
            << indent(2) << "float constant_weight,\n"
            << indent(2) << "int max_neighbor_count,\n"
            << indent(2) << "int ring_depth,\n"
            << indent(2) << "int neuron_count,\n"
            << indent(2) << "int neuron_index,\n"
            << indent(2) << tick_type << " tick\n"
            << ") {\n"
            << indent(1) << thread_qualifier
            << "int stack_row_base[SPIKECOREC_MAXIMUM_K2TREE_HEIGHT];\n"
            << indent(1) << thread_qualifier
            << "int stack_column_base[SPIKECOREC_MAXIMUM_K2TREE_HEIGHT];\n"
            << indent(1) << thread_qualifier
            << "int stack_block_size[SPIKECOREC_MAXIMUM_K2TREE_HEIGHT];\n"
            << indent(1) << thread_qualifier
            << "int stack_bit_offset[SPIKECOREC_MAXIMUM_K2TREE_HEIGHT];\n"
            << indent(1) << thread_qualifier
            << "int stack_next_column[SPIKECOREC_MAXIMUM_K2TREE_HEIGHT];\n"
            << indent(1) << "stack_row_base[0] = 0;\n"
            << indent(1) << "stack_column_base[0] = 0;\n"
            << indent(1) << "stack_block_size[0] = padded_node_count;\n"
            << indent(1) << "stack_bit_offset[0] = 0;\n"
            << indent(1) << "stack_next_column[0] = 0;\n"
            << indent(1) << "int stack_top = (tree_height > 0 && neuron_index >= 0 &&\n"
            << indent(4) << "neuron_index < neuron_count) ? 0 : -1;\n\n"
            << indent(1) << tick_type << " row_lane_count = rank_float4_stride * 4;\n"
            << indent(1) << tick_type
            << " source_lane_base = (" << tick_type << ")neuron_index * row_lane_count;\n"
            << indent(1) << "int neighbor_slot = 0;\n"
            << indent(1) << "int target_neuron_index = 0;\n\n"
            // A degree above max_neighbor_count has no slot in the per-edge arrays at all
            // (the host cannot address one either), so the walk stops rather than reading
            // into the next source's row.
            << indent(1) << "while (neighbor_slot < max_neighbor_count &&\n"
            << indent(3) << "(target_neuron_index = k2tree_next_neighbor(\n"
            << indent(5) << "internal_node_words, leaf_node_words, rank_superblock_table,\n"
            << indent(5) << "rank_subblock_table, branching_factor, superblock_size_words,\n"
            << indent(5) << "neuron_count, tree_height, internal_bit_count, neuron_index,\n"
            << indent(5) << "stack_row_base, stack_column_base, stack_block_size,\n"
            << indent(5) << "stack_bit_offset, stack_next_column, stack_top)) >= 0) {\n"
            << indent(2) << "int edge_slot = neuron_index * max_neighbor_count + neighbor_slot;\n\n"
            << indent(2) << "float edge_weight = constant_weight;\n"
            << indent(2) << "if (constant_weight == 0.0f) {\n"
            << indent(3) << tick_type << " target_lane_base = (" << tick_type
            << ")target_neuron_index * row_lane_count;\n"
            << indent(3) << "float reconstructed_weight = 0.0f;\n"
            << indent(3) << "for (" << tick_type << " lane = 0; lane < row_lane_count; ++lane) {\n"
            << indent(4) << "reconstructed_weight += U_matrix[source_lane_base + lane] *\n"
            << indent(6) << "(edge_weight_coefficients[lane] * V_matrix[target_lane_base + lane]);\n"
            << indent(3) << "}\n"
            << indent(3) << "edge_weight = reconstructed_weight + edge_weight_deltas[edge_slot];\n"
            << indent(2) << "}\n\n"
            // Which plane the arrival lands in is a property of the EDGE: the synapse
            // prototype its projection names, or plane 0 when it names none -- in which case
            // the raw weight is the delivered current, which is what network_inputs meant
            // before there were any synapse dynamics at all.
            << indent(2) << tick_type << " arrival_index = network_input_ring_index(\n"
            << indent(4) << "tick + (" << tick_type << ")edge_delay_ticks[edge_slot],\n"
            << indent(4) << "edge_synapse_plane[edge_slot], ring_depth,\n"
            << indent(4) << "neuron_count, target_neuron_index);\n";

    // Many sources converge on one target in the same tick, so the arrival has to be atomic.
    if (is_metal) {
        helpers << indent(2) << "device atomic_float *arrival_slot =\n"
                << indent(4) << "(device atomic_float *)(network_inputs + arrival_index);\n"
                << indent(2)
                << "atomic_fetch_add_explicit(arrival_slot, edge_weight, memory_order_relaxed);\n";
    } else {
        helpers << indent(2) << "atomicAdd(network_inputs + arrival_index, edge_weight);\n";
    }

    helpers << indent(2) << "neighbor_slot += 1;\n"
            << indent(1) << "}\n"
            << "}\n\n";

    return helpers.str();
}

// Reads this tick's ring row for this neuron. A PLAIN load, not an exchange: nothing writes
// row `tick % ring_depth` while this tick's kernel is running, because every edge delay is at
// least one tick (the engine asserts that where the delays are flattened), so the row is
// read-only for the duration of the dispatch. The row is emptied afterwards, as a whole row,
// by the clear kernel the engine dispatches behind this one -- see
// generate_ring_row_clear_kernel.
String emit_synaptic_input_read(usize indent_level) {
    ostringstream read;
    read << indent(indent_level) << buffer_index_type_name
         << " synaptic_input_index = network_input_ring_index(\n"
         << indent(indent_level + 2)
         << "tick, 0, ring_depth, neuron_count, neuron_index);\n"
         << indent(indent_level) << "float " << synaptic_input_local_name
         << " = network_inputs[synaptic_input_index];\n";
    return read.str();
}

// The fixed epilogue every cell device function ends with.
String emit_propagation_epilogue(usize indent_level) {
    ostringstream epilogue;
    epilogue << indent(indent_level) << "if (spike_flags[neuron_index] != 0) {\n"
             << indent(indent_level + 1) << "propagate_spike(\n"
             << indent(indent_level + 3)
             << "network_inputs, internal_node_words, leaf_node_words, rank_superblock_table,\n"
             << indent(indent_level + 3)
             << "rank_subblock_table, U_matrix, V_matrix, edge_weight_coefficients,\n"
             << indent(indent_level + 3)
             << "edge_weight_deltas, edge_delay_ticks, edge_synapse_plane, branching_factor,\n"
             << indent(indent_level + 3) << "superblock_size_words,\n"
             << indent(indent_level + 3)
             << "padded_node_count, tree_height, internal_bit_count, rank_float4_stride,\n"
             << indent(indent_level + 3)
             << "constant_weight, max_neighbor_count, ring_depth, neuron_count, neuron_index,\n"
             << indent(indent_level + 3) << "tick);\n"
             << indent(indent_level) << "}\n";
    return epilogue.str();
}

// ── Synapse dynamics ─────────────────────────────────────────────────────────
//
// One wired synapse prototype, resolved once so nothing downstream re-derives an index.
// `state_variable_offset` is in state variables, not floats: the runtime base is
// state_variable_offset * neuron_count + neuron_index * state variable count, and
// neuron_count is only known to the kernel.
struct WiredSynapse {
    const SynapseTypeSpecification *type = nullptr;
    const ComponentPrototype *prototype = nullptr;
    usize plane_index = 0;            // 1 + position in the wired list
    usize state_variable_offset = 0;  // running sum of the preceding prototypes' state widths
};

// The one local an OnEvent handler reads its arriving weight through. Bound to `weight`
// inside the handler and nowhere else -- see build_synapse_symbol_table.
const String synapse_arrival_local_name = "synapse_arrival_weight";

// The exposure a synapse delivers through. LEMS spells it `i` on everything descending from
// basePointCurrent, which every synapse ComponentType in the standard library does, and it is
// the name a postsynaptic cell's own "synapses[*]/i" path selects.
const String synapse_output_variable_name = "i";

// A conductance-based synapse is refused rather than run: see kernel_codegen.h. Detected on
// the parameters it declares, which is what survives into SynapseTypeSpecification --
// mirroring nml.cpp's own commented-out is_conductance_based(), which additionally consulted
// the Requirement on `v` that a SynapseTypeSpecification does not carry.
bool declares_conductance_parameter(const SynapseTypeSpecification &synapse_type) {
    static const Set<String> conductance_parameter_names = {"erev", "gbase", "gbase1", "gbase2"};

    for (const String &parameter_name : synapse_type.parameter_names) {
        if (conductance_parameter_names.count(parameter_name) > 0) return true;
    }
    return false;
}

// The Arrival-stage StateAssignments the OnEvent on `port_name` fires, in declaration order.
Vector<const DynamicsInstruction *> arrival_assignments_on_port(
        const SynapseTypeSpecification &synapse_type, const String &port_name) {
    Vector<const DynamicsInstruction *> assignments;
    for (const DynamicsInstruction &instruction : synapse_type.dynamics) {
        if (instruction.stage != DynamicsStage::Arrival) continue;
        if (instruction.source_tag != NML_DeclarationType::StateAssignment) continue;
        if (instruction.condition != port_name) continue;
        assignments.push_back(&instruction);
    }
    return assignments;
}

// Everything the synapse path refuses, checked in one pass so a model fails at generation
// with the construct named rather than half-generating and running wrong.
void reject_unsupported_synapse(const WiredSynapse &wired) {
    const SynapseTypeSpecification &synapse_type = *wired.type;

    if (declares_conductance_parameter(synapse_type)) {
        report_error("conductance-based synapses are not supported yet. This type declares a "
                     "reversal potential / baseline conductance, so it computes "
                     "i = g * (erev - v) -- a driving force that depends on the postsynaptic "
                     "voltage and reverses sign as v crosses erev. Running it as a "
                     "current-based synapse would be a different model, not an approximation, "
                     "so it is refused. Use a current-based synapse (alphaCurrentSynapse and "
                     "friends), which is what GLIF1-5 use",
                     synapse_type.name);
    }

    if (synapse_type.requires_per_edge_state) {
        report_error("this synapse's state does not superpose across converging edges (it "
                     "carries a plasticity or block mechanism), so it cannot share one "
                     "aggregated state per target neuron. Per-edge synapse state is a "
                     "separate ticket",
                     synapse_type.name);
    }

    for (const DynamicsInstruction &instruction : synapse_type.dynamics) {
        if (instruction.source_tag == NML_DeclarationType::StateVariable) continue;

        if (!instruction.regime_name.empty()) {
            report_error("regimes are not supported: an instruction targeting '" +
                                 instruction.target + "' sits inside Regime '" +
                                 instruction.regime_name + "'",
                         synapse_type.name);
        }

        switch (instruction.source_tag) {
            case NML_DeclarationType::Regime:
            case NML_DeclarationType::Transition:
            case NML_DeclarationType::OnEntry:
                report_error("regimes are not supported on a synapse", synapse_type.name);
            case NML_DeclarationType::ConditionalDerivedVariable:
            case NML_DeclarationType::Case:
                report_error("ConditionalDerivedVariable / Case is not supported: its per-case "
                             "conditions are not carried by DynamicsInstruction",
                             synapse_type.name);
            case NML_DeclarationType::OnCondition:
                report_error("OnCondition is not supported on a synapse: only the arrival "
                             "handler, the time derivatives and the exposed current are lowered",
                             synapse_type.name);
            case NML_DeclarationType::OnEvent:
                // The only incoming port a synapse has. A model declaring another would have
                // its handler silently never run, since nothing routes arrivals to it.
                if (instruction.target != "in") {
                    report_error("OnEvent on port '" + instruction.target +
                                         "' is not supported: spike arrivals are routed to the "
                                         "'in' port only",
                                 synapse_type.name);
                }
                break;
            default:
                break;
        }
    }

    if (arrival_assignments_on_port(synapse_type, "in").empty()) {
        report_error("this synapse declares no OnEvent handler on port 'in', so an arriving "
                     "spike would change nothing about it. A synapse with no arrival handler "
                     "would silently swallow every spike routed through it",
                     synapse_type.name);
    }
}

// State variables read from the aggregated per-(target, prototype) slice; parameters read as
// the prototype's own resolved values, baked as literals because they are constants of the
// prototype rather than of any neuron.
//
// `weight` is the exception and the whole point of the aggregation: inside the OnEvent
// handler it names the SUMMED weight of the arrivals due this tick, so one evaluation of the
// handler stands in for every converging edge's. Outside the handler it keeps the prototype's
// own value, because a TimeDerivative reading "this tick's arrival" would be nonsense.
SymbolTable build_synapse_symbol_table(const WiredSynapse &wired, bool bind_weight_to_arrival) {
    const SynapseTypeSpecification &synapse_type = *wired.type;
    const StateStorage storage{synapse_type.name, synapse_type.state_variable_names,
                               "synapse_state", "synapse_state_base"};

    SymbolTable symbols;
    symbols.component_type_name = synapse_type.name;

    // Defined ahead of everything else because SymbolTable::define is first-wins, and this
    // has to shadow the parameter of the same name.
    if (bind_weight_to_arrival) symbols.define("weight", synapse_arrival_local_name);

    for (usize slot = 0; slot < synapse_type.state_variable_names.size(); ++slot) {
        symbols.define(synapse_type.state_variable_names[slot], storage.element(slot));
    }

    if (wired.prototype->starting_parameters.size() != synapse_type.parameter_names.size()) {
        report_error("synapse prototype '" + wired.prototype->instance_id + "' carries " +
                             to_string(wired.prototype->starting_parameters.size()) +
                             " starting parameters but its type declares " +
                             to_string(synapse_type.parameter_names.size()),
                     synapse_type.name);
    }

    for (usize slot = 0; slot < synapse_type.parameter_names.size(); ++slot) {
        symbols.define(synapse_type.parameter_names[slot],
                       format_float_literal(
                               wired.prototype->starting_parameters[slot].float64));
    }

    return symbols;
}

StateStorage synapse_state_storage(const SynapseTypeSpecification &synapse_type) {
    return StateStorage{synapse_type.name, synapse_type.state_variable_names, "synapse_state",
                        "synapse_state_base"};
}

// Resolves this thread's slice of the aggregated state. Neuron-major within a prototype, so
// one thread's state variables are contiguous. Computed 64-bit for the same reason
// network_input_ring_index is: the host sizes synapse_state in s64.
String emit_synapse_state_base(const WiredSynapse &wired, usize indent_level) {
    return indent(indent_level) + buffer_index_type_name + " synapse_state_base = " +
           to_string(wired.state_variable_offset) + " * (" + buffer_index_type_name +
           ")neuron_count + (" + buffer_index_type_name + ")neuron_index * " +
           to_string(wired.type->state_variable_names.size()) + ";\n";
}

// DerivedVariables as locals, in source order, each visible to the ones after it -- the same
// treatment a cell's get. A select= path is refused: a synapse has no attached children with
// an engine buffer behind them, and the only path shape this generator lowers reduces over a
// CELL's synapses.
String emit_synapse_derived_variables(const WiredSynapse &wired,
                                      const NML_ParseResult &parse_result, SymbolTable &symbols,
                                      Vector<Pair<String, String>> &derived_variable_locals,
                                      usize indent_level) {
    ostringstream derived;

    for (const DynamicsInstruction &instruction : wired.type->dynamics) {
        if (instruction.stage != DynamicsStage::Integrate) continue;
        if (instruction.source_tag != NML_DeclarationType::DerivedVariable) continue;

        const SymbolTable visible_symbols = with_fallback_symbols(symbols, parse_result);
        if (select_path_head_name(instruction.expression, visible_symbols).has_value()) {
            report_error("unsupported select path \"" + instruction.expression +
                                 "\" on a synapse: a synapse has no child structure with an "
                                 "engine buffer behind it",
                         wired.type->name);
        }

        const String read_expression =
                translate_expression(instruction.expression, visible_symbols);
        const String local_name = "derived_" + sanitize_identifier(instruction.target);

        derived << indent(indent_level) << "float " << local_name << " = " << read_expression
                << ";\n";
        derived_variable_locals.push_back({local_name, read_expression});
        symbols.define(instruction.target, local_name);
    }

    return derived.str();
}

// One wired prototype's per-tick body: deliver this tick's arrivals, integrate one dt, then
// add the exposed current into the delivered plane every cell reads.
//
// All three touch only this neuron's own slots, and the thread that runs them is the thread
// that later runs the target cell's dynamics, so the hand-off needs no synchronisation. The
// output add is a PLAIN add for the same reason: within tick T nothing else writes plane 0 of
// row T -- the host's stimulus lands there before the launch, and every propagated arrival
// carries a delay of at least one tick, so it lands in a later row.
String emit_synapse_tick_body(const WiredSynapse &wired, const NML_ParseResult &parse_result) {
    reject_unsupported_synapse(wired);

    const StateStorage storage = synapse_state_storage(*wired.type);

    ostringstream body;
    body << emit_synapse_state_base(wired, 1);

    // ── Stage 1, Deliver ─────────────────────────────────────────────────────
    body << indent(1) << buffer_index_type_name
         << " synapse_arrival_index = network_input_ring_index(\n"
         << indent(3) << "tick, " << wired.plane_index
         << ", ring_depth, neuron_count, neuron_index);\n"
         << indent(1) << "float " << synapse_arrival_local_name
         << " = network_inputs[synapse_arrival_index];\n";

    // Deliberately built without the DerivedVariable locals: the handler runs before they are
    // computed, so one referencing a derived name is reported as an unknown identifier rather
    // than emitted as a forward reference the shader compiler would reject elsewhere.
    const SymbolTable arrival_symbols = build_synapse_symbol_table(wired, true);
    const Vector<const DynamicsInstruction *> arrival_assignments =
            arrival_assignments_on_port(*wired.type, "in");

    // An arrival handler that ignores the arriving weight cannot be aggregated: N converging
    // spikes would be applied as one, silently, and the model would run at a fraction of its
    // declared coupling. Checked on the TRANSLATED expression, where the arrival local appears
    // if and only if `weight` was read.
    const SymbolTable visible_arrival_symbols =
            with_fallback_symbols(arrival_symbols, parse_result);
    for (const DynamicsInstruction *assignment : arrival_assignments) {
        const String translated =
                translate_expression(assignment->expression, visible_arrival_symbols);
        if (translated.find(synapse_arrival_local_name) == String::npos) {
            report_error("the OnEvent handler assigns '" + assignment->target +
                                 "' from an expression that does not read 'weight'. Arrivals "
                                 "converging on one target are summed into a single weight and "
                                 "the handler runs once, so an assignment ignoring that weight "
                                 "would apply many spikes as one",
                         wired.type->name);
        }
    }

    body << indent(1) << "if (" << synapse_arrival_local_name << " != 0.0f) {\n"
         << emit_state_assignment_group(storage, parse_result, arrival_symbols,
                                        arrival_assignments, 2)
         << indent(1) << "}\n";

    // ── Stage 2, Integrate ───────────────────────────────────────────────────
    SymbolTable symbols = build_synapse_symbol_table(wired, false);
    Vector<Pair<String, String>> derived_variable_locals;
    body << emit_synapse_derived_variables(wired, parse_result, symbols, derived_variable_locals,
                                           1);

    // A synapse has no Regimes -- reject_unsupported_synapse refuses one -- so every
    // TimeDerivative it declares integrates unconditionally.
    Vector<Pair<usize, String>> pending_state_writes;
    body << emit_unconditional_euler_temporaries(storage, wired.type->dynamics, parse_result,
                                                 symbols, 1, pending_state_writes);
    body << emit_euler_write_back(storage, pending_state_writes, 1);
    const bool has_integrated_state = !pending_state_writes.empty();

    // The delivered current is this tick's, so the derived locals feeding it are re-evaluated
    // against the state Integrate just wrote -- the same post-integrate reading a cell's own
    // Detect/Reset stages observe.
    if (has_integrated_state) {
        for (const auto &derived_local : derived_variable_locals) {
            body << indent(1) << derived_local.first << " = " << derived_local.second << ";\n";
        }
    }

    // ── Delivery ─────────────────────────────────────────────────────────────
    if (!symbols.contains(synapse_output_variable_name)) {
        report_error("this synapse exposes no '" + synapse_output_variable_name +
                             "', so there is no current to deliver. A synapse's output current "
                             "is the StateVariable or DerivedVariable of that name, which is what "
                             "a postsynaptic cell's \"synapses[*]/i\" path selects",
                     wired.type->name);
    }

    body << indent(1) << buffer_index_type_name
         << " synapse_output_index = network_input_ring_index(\n"
         << indent(3) << "tick, 0, ring_depth, neuron_count, neuron_index);\n"
         << indent(1) << "network_inputs[synapse_output_index] += "
         << symbols.read_expression_for(synapse_output_variable_name) << ";\n";

    return body.str();
}

String emit_synapse_initialize_body(const WiredSynapse &wired,
                                    const NML_ParseResult &parse_result) {
    reject_unsupported_synapse(wired);

    Vector<const DynamicsInstruction *> assignments;
    for (const DynamicsInstruction &instruction : wired.type->dynamics) {
        if (instruction.stage != DynamicsStage::Initialize) continue;
        if (instruction.source_tag != NML_DeclarationType::StateAssignment) continue;
        assignments.push_back(&instruction);
    }
    if (assignments.empty()) return "";

    ostringstream body;
    body << emit_synapse_state_base(wired, 1);
    body << emit_state_assignment_group(synapse_state_storage(*wired.type), parse_result,
                                        build_synapse_symbol_table(wired, false), assignments, 1);
    return body.str();
}

// Resolves every wired prototype against its type, in the order
// wired_synapse_prototype_indices reports -- which is the order that fixes both the plane
// numbering and the synapse_state layout the engine allocates to.
Vector<WiredSynapse> collect_wired_synapses(const NML_ParseResult &parse_result) {
    Vector<WiredSynapse> wired_synapses;
    usize state_variable_offset = 0;

    for (const s64 prototype_index : wired_synapse_prototype_indices(parse_result)) {
        const ComponentPrototype &prototype =
                parse_result.synapse_prototypes[(usize)prototype_index];

        if (prototype.type_index < 0 ||
            prototype.type_index >= (s64)parse_result.synapse_types.size()) {
            throw runtime_error("kernel_codegen: synapse prototype '" + prototype.instance_id +
                                "' names synapse type index " + to_string(prototype.type_index) +
                                ", which no synapse type carries");
        }

        WiredSynapse wired;
        wired.type = &parse_result.synapse_types[(usize)prototype.type_index];
        wired.prototype = &prototype;
        wired.plane_index = wired_synapses.size() + 1;
        wired.state_variable_offset = state_variable_offset;
        state_variable_offset += wired.type->state_variable_names.size();

        wired_synapses.push_back(wired);
    }

    return wired_synapses;
}

String synapse_device_function_name(const WiredSynapse &wired, KernelPurpose purpose) {
    const String prefix =
            purpose == KernelPurpose::Tick ? "synapse_step_" : "synapse_initialize_";
    return prefix + sanitize_identifier(wired.prototype->instance_id);
}

// Emitted with an explicit parameter list rather than through the cell functions' table:
// a synapse touches only the aggregated state and the ring, and reads none of the
// per-neuron cell scaffolding.
String emit_synapse_device_function(const WiredSynapse &wired, const String &body,
                                    KernelBackend backend, KernelPurpose purpose) {
    const bool is_metal = backend == KernelBackend::Metal;
    const String function_prefix = is_metal ? "inline void " : "__device__ inline void ";
    const String device_qualifier = is_metal ? "device " : "";
    const String tick_type = is_metal ? "long" : "long long";

    ostringstream function;
    function << function_prefix << synapse_device_function_name(wired, purpose) << "(\n"
             << indent(2) << device_qualifier << "float *synapse_state,\n";
    if (purpose == KernelPurpose::Tick) {
        function << indent(2) << device_qualifier << "float *network_inputs,\n";
    }
    function << indent(2) << "int neuron_index,\n" << indent(2) << "int neuron_count";
    if (purpose == KernelPurpose::Tick) {
        function << ",\n"
                 << indent(2) << "int ring_depth,\n"
                 << indent(2) << "float dt,\n"
                 << indent(2) << tick_type << " tick";
    }
    function << "\n) {\n" << body << "}\n\n";
    return function.str();
}

String synapse_device_function_call(const WiredSynapse &wired, KernelPurpose purpose,
                                    usize indent_level) {
    const String arguments = purpose == KernelPurpose::Tick
                                     ? "synapse_state, network_inputs, neuron_index, "
                                       "neuron_count, ring_depth, dt, tick"
                                     : "synapse_state, neuron_index, neuron_count";
    return indent(indent_level) + synapse_device_function_name(wired, purpose) + "(" + arguments +
           ");\n";
}

// ── Per-cell-type bodies ─────────────────────────────────────────────────────

String emit_tick_body(const CellTypeSpecification &cell_type,
                      const NML_ParseResult &parse_result) {
    const CellRegimeLayout regimes = resolve_cell_regimes(cell_type);
    reject_unsupported_instructions(cell_type, regimes);

    ostringstream body;
    SymbolTable symbols = build_base_symbol_table(cell_type);
    Vector<Pair<String, String>> derived_variable_locals;

    // Stage 1, Deliver, and Integrate, part 1: DerivedVariables become locals in source
    // order, each visible to the ones after it. A DerivedVariable referencing one declared
    // later resolves to nothing and throws, rather than emitting a forward reference the
    // shader compiler would reject far from its cause.
    //
    // A select= path becomes a local the same way, so whatever it binds to is reached by the
    // name the model gave it and resolves through the ordinary precedence chain. The ring row
    // is loaded once, just before the first path that needs it, and every path in the type
    // binds to that one local -- the load is plain and the row is cleared by the engine's own
    // clear kernel, so this is now consistency rather than correctness.
    bool has_read_synaptic_input = false;

    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.stage != DynamicsStage::Integrate) continue;
        if (instruction.source_tag != NML_DeclarationType::DerivedVariable) continue;

        const SymbolTable visible_symbols = with_fallback_symbols(symbols, parse_result);
        const Optional<String> path_head_name =
                select_path_head_name(instruction.expression, visible_symbols);

        String read_expression;
        if (path_head_name.has_value()) {
            require_synaptic_select_path(instruction.expression, *path_head_name, cell_type.name);
            if (!has_read_synaptic_input) {
                body << emit_synaptic_input_read(1);
                has_read_synaptic_input = true;
            }
            read_expression = synaptic_input_local_name;
        } else {
            read_expression = translate_expression(instruction.expression, visible_symbols);
        }

        const String local_name = "derived_" + sanitize_identifier(instruction.target);
        body << indent(1) << "float " << local_name << " = " << read_expression << ";\n";
        derived_variable_locals.push_back({local_name, read_expression});
        symbols.define(instruction.target, local_name);
    }

    // Integrate, part 2. The regime-free TimeDerivatives first, then the regime-dispatched
    // ones, and only then the single write-back covering both -- so a regime-scoped
    // derivative reading a regime-free variable still sees the state as it stood at entry.
    const StateStorage storage = cell_state_storage(cell_type);
    Vector<Pair<usize, String>> pending_state_writes;
    body << emit_unconditional_euler_temporaries(storage, cell_type.dynamics, parse_result,
                                                 symbols, 1, pending_state_writes);
    body << emit_regime_dispatched_integration(cell_type, regimes, storage, parse_result, symbols,
                                               1, pending_state_writes);
    body << emit_euler_write_back(storage, pending_state_writes, 1);
    const bool has_integrated_state = !pending_state_writes.empty();

    // Stages 3 to 5 -- Detect, Emit, Reset -- all read the state this tick's Integrate just
    // produced, so the DerivedVariable locals they read are re-evaluated here against that
    // same post-integrate state. Without this one half of a comparison would be post-integrate
    // (the cell_state reads inside the test) and the other pre-integrate (a derived local), so
    // a threshold or reset value that is itself a DerivedVariable of a state variable would
    // compare against a value one dt old. See the header for why post-integrate is the reading
    // both halves are made to agree on.
    //
    // Re-evaluated rather than recomputed into fresh names, so every expression written
    // against a derived variable keeps reading the one name the model gave it. The
    // synaptic-input path re-reads the drained local, not the ring row, so the row is still
    // emptied exactly once.
    const bool has_post_integrate_reader =
            any_of(cell_type.dynamics.begin(), cell_type.dynamics.end(),
                   [](const DynamicsInstruction &instruction) {
                       return instruction.stage == DynamicsStage::Detect ||
                              (instruction.stage == DynamicsStage::Reset &&
                               instruction.condition.empty());
                   });
    if (has_integrated_state && has_post_integrate_reader) {
        for (const auto &derived_local : derived_variable_locals) {
            body << indent(1) << derived_local.first << " = " << derived_local.second << ";\n";
        }
    }

    // Detect, then the Reset and Emit bodies each OnCondition gates. Reset and Emit
    // instructions carry their OnCondition's test verbatim in `condition` and their owning
    // regime in `regime_name`, so that PAIR is the join key back to the condition that fired
    // them. The regime half is load-bearing: two regimes routinely declare OnConditions with
    // the same test, and joining on the test alone would give each of them the other's
    // assignments, its EventOut and its Transition.
    auto condition_key = [](const DynamicsInstruction &instruction) {
        return instruction.regime_name + "\x1f" + instruction.condition;
    };

    Set<String> emitted_condition_keys;
    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.stage != DynamicsStage::Detect) continue;

        // Two OnConditions in one regime with identical tests would otherwise each pick up
        // the other's body, duplicating every assignment. They fire together by definition,
        // so the first block already carries both.
        const String detect_key = instruction.regime_name + "\x1f" + instruction.expression;
        if (!emitted_condition_keys.insert(detect_key).second) continue;

        // An OnCondition declared inside a regime only applies while that regime is active,
        // so its own test is ANDed with the regime check rather than replacing it.
        String guard = translate_expression(instruction.expression,
                                            with_fallback_symbols(symbols, parse_result));
        if (!instruction.regime_name.empty()) {
            guard = active_regime_local_name +
                    " == " + to_string(regimes.index_of(instruction.regime_name)) + " && (" +
                    guard + ")";
        }
        body << indent(1) << "if (" << guard << ") {\n";

        // Stage 4 before stage 5, and every StateAssignment this condition gates emitted as
        // one simultaneous group rather than one statement at a time.
        for (const DynamicsInstruction &gated : cell_type.dynamics) {
            if (condition_key(gated) != detect_key) continue;
            if (gated.stage == DynamicsStage::Emit) body << emit_spike(2);
        }
        body << emit_state_assignment_group(
                storage, parse_result, symbols,
                reset_assignments_gated_by(cell_type, instruction.regime_name,
                                           instruction.expression),
                2);

        // Then the Transitions this condition fires. Entering a regime IS the transition, so
        // the target regime's OnEntry body is inlined right here rather than rediscovered by
        // some later "have I just entered" check -- there is no such check anywhere.
        //
        // After the OnCondition's own assignments, and as its own simultaneous group: LEMS
        // runs the handler, then enters the regime, then runs its OnEntry. A separate group
        // also keeps the two handlers' temporaries apart, since both land in this one block.
        for (const DynamicsInstruction &gated : cell_type.dynamics) {
            if (gated.source_tag != NML_DeclarationType::Transition) continue;
            if (condition_key(gated) != detect_key) continue;

            body << indent(2) << regime_state_element(regimes) << " = "
                 << to_string(regimes.index_of(gated.target)) << ".0f;\n";
            body << emit_state_assignment_group(storage, parse_result, symbols,
                                                regime_entry_assignments(cell_type, gated.target),
                                                2, "entered_");
        }

        body << indent(1) << "}\n";
    }

    // Anything with no gate at all runs unconditionally, and runs HERE: stages 4 and 5 follow
    // stage 3, so an ungated reset must be able to override what this tick's Detect blocks
    // wrote rather than be clobbered by them. Well-formed LEMS puts every StateAssignment and
    // EventOut inside a handler, so this is close to dead, but dropping such an instruction
    // would be a silent omission and emitting it early would be a silently wrong order.
    // Anything ungated but regime-scoped was already refused, so these are regime-free.
    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        if (!instruction.condition.empty()) continue;
        if (instruction.stage == DynamicsStage::Emit) body << emit_spike(1);
    }
    body << emit_state_assignment_group(storage, parse_result, symbols,
                                        reset_assignments_gated_by(cell_type, "", ""), 1);

    // A gated instruction whose gate never appeared would silently never run.
    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        const bool is_gated_body = instruction.stage == DynamicsStage::Reset ||
                                   instruction.stage == DynamicsStage::Emit ||
                                   instruction.source_tag == NML_DeclarationType::Transition;
        if (!is_gated_body || instruction.condition.empty()) continue;
        if (emitted_condition_keys.count(condition_key(instruction)) == 0) {
            report_error("instruction targeting '" + instruction.target + "' is gated by '" +
                                 instruction.condition + "', which matches no OnCondition",
                         cell_type.name);
        }
    }

    // Stage 6, Propagate. Unconditional boilerplate: every cell type scatters its spike the
    // same way, so this is emitted whether or not the type declares an EventOut -- a type
    // that never raises the flag simply never enters the block.
    body << emit_propagation_epilogue(1);

    // The active regime, read once and never re-read, prepended so every guard above sees the
    // value the tick STARTED with. That is what keeps the regime a cell has just moved INTO
    // from also running its own OnCondition in the same tick, off state the transition just
    // wrote -- a Transition lands in storage and is observed from the next tick on.
    //
    // Prepended rather than emitted up front so it can be skipped when nothing reads it: a
    // type whose one regime is always active needs no dispatch and no guards, and an unused
    // local is a shader-compiler warning on every kernel the model produces.
    const String tick_body = body.str();
    if (!regimes.has_regimes() ||
        tick_body.find(active_regime_local_name) == String::npos) {
        return tick_body;
    }
    return indent(1) + "int " + active_regime_local_name + " = (int)" +
           regime_state_element(regimes) + ";\n" + tick_body;
}

String emit_initialize_body(const CellTypeSpecification &cell_type,
                            const NML_ParseResult &parse_result) {
    const CellRegimeLayout regimes = resolve_cell_regimes(cell_type);
    const SymbolTable symbols = build_base_symbol_table(cell_type);
    const StateStorage storage = cell_state_storage(cell_type);

    Vector<const DynamicsInstruction *> assignments;
    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.stage != DynamicsStage::Initialize) continue;
        // The OnStart element itself also lands on this stage carrying nothing; only the
        // StateAssignments under it initialise anything.
        if (instruction.source_tag != NML_DeclarationType::StateAssignment) continue;

        assignments.push_back(&instruction);
    }

    ostringstream body;

    // One OnStart body, so the same simultaneous semantics as any other handler.
    body << emit_state_assignment_group(storage, parse_result, symbols, assignments, 1);

    // Then the cell enters its initial regime, which is an entry like any other: the index is
    // stored and that regime's OnEntry runs. After OnStart, because LEMS starts the component
    // and then enters the regime -- an OnEntry writing a variable OnStart also writes is the
    // regime's answer, not the component's.
    if (regimes.has_regimes()) {
        body << indent(1) << regime_state_element(regimes) << " = "
             << to_string(regimes.initial_regime_index) << ".0f;\n";
        body << emit_state_assignment_group(
                storage, parse_result, symbols,
                regime_entry_assignments(cell_type,
                                         regimes.regime_names[(usize)regimes.initial_regime_index]),
                1, "entered_");
    }

    return body.str();
}

// ── Kernel assembly ──────────────────────────────────────────────────────────

// One master-kernel parameter: the name the engine binds it by and how each backend spells
// it. The names the engine sees are derived from this table rather than listed separately,
// so a parameter cannot be declared in the signature under one name and asked for under
// another.
struct KernelArgumentDeclaration {
    String name;
    String metal_declaration;
    String cuda_declaration;
};

const Vector<KernelArgumentDeclaration> &kernel_argument_declarations() {
    // U_matrix and V_matrix are float4 buffers on the host and are declared here as plain
    // floats: identical bytes, and scalar lanes make the reconstruction arithmetic the same
    // one WeightMatrix performs on the CPU. rank_float4_stride keeps its name (it is what
    // the engine binds) and counts float4 elements, so a row is four times that many lanes.
    static const Vector<KernelArgumentDeclaration> declarations = {
        {"cell_state", "device float       *cell_state", "float *cell_state"},
        {"cell_parameters", "device const float *cell_parameters", "const float *cell_parameters"},
        {"network_inputs", "device float       *network_inputs", "float *network_inputs"},
        {"last_spiked", "device long        *last_spiked", "long long *last_spiked"},
        {"spike_flags", "device int         *spike_flags", "int *spike_flags"},
        {"cell_state_base", "device const int   *cell_state_base", "const int *cell_state_base"},
        {"cell_parameter_base", "device const int   *cell_parameter_base",
         "const int *cell_parameter_base"},
        {"cell_type_index", "device const int   *cell_type_index", "const int *cell_type_index"},
        {"neuron_count", "constant int       &neuron_count", "int neuron_count"},
        {"dt", "constant float     &dt", "float dt"},
        {"tick", "constant long      &tick", "long long tick"},
        {"internal_node_words", "device const uint  *internal_node_words",
         "const unsigned int *internal_node_words"},
        {"leaf_node_words", "device const uint  *leaf_node_words",
         "const unsigned int *leaf_node_words"},
        {"rank_superblock_table", "device const uint  *rank_superblock_table",
         "const unsigned int *rank_superblock_table"},
        {"rank_subblock_table", "device const ushort *rank_subblock_table",
         "const unsigned short *rank_subblock_table"},
        {"U_matrix", "device const float *U_matrix", "const float *U_matrix"},
        {"V_matrix", "device const float *V_matrix", "const float *V_matrix"},
        {"edge_weight_coefficients", "device const float *edge_weight_coefficients",
         "const float *edge_weight_coefficients"},
        {"edge_weight_deltas", "device const float *edge_weight_deltas",
         "const float *edge_weight_deltas"},
        {"edge_delay_ticks", "device const int   *edge_delay_ticks",
         "const int *edge_delay_ticks"},
        {"branching_factor", "constant int       &branching_factor", "int branching_factor"},
        {"superblock_size_words", "constant int       &superblock_size_words",
         "int superblock_size_words"},
        {"padded_node_count", "constant int       &padded_node_count", "int padded_node_count"},
        {"tree_height", "constant int       &tree_height", "int tree_height"},
        {"internal_bit_count", "constant int       &internal_bit_count", "int internal_bit_count"},
        {"rank_float4_stride", "constant long      &rank_float4_stride",
         "long long rank_float4_stride"},
        {"constant_weight", "constant float     &constant_weight", "float constant_weight"},
        {"max_neighbor_count", "constant int       &max_neighbor_count", "int max_neighbor_count"},
        {"ring_depth", "constant int       &ring_depth", "int ring_depth"},
        {"synapse_state", "device float       *synapse_state", "float *synapse_state"},
        {"edge_synapse_plane", "device const int   *edge_synapse_plane",
         "const int *edge_synapse_plane"},
    };
    return declarations;
}

const Vector<String> &kernel_argument_names() {
    static const Vector<String> names = [] {
        Vector<String> collected;
        for (const KernelArgumentDeclaration &declaration : kernel_argument_declarations()) {
            collected.push_back(declaration.name);
        }
        return collected;
    }();
    return names;
}

// What a cell device function is handed, in order. The tick entry point carries the whole
// propagation apparatus; the initialize entry point runs OnStart bodies only and takes just
// the storage those touch.
const Vector<String> &device_function_parameter_names(KernelPurpose purpose) {
    static const Vector<String> tick_parameters = {
        "cell_state",       "cell_parameters",          "network_inputs",
        "last_spiked",      "spike_flags",              "internal_node_words",
        "leaf_node_words",  "rank_superblock_table",    "rank_subblock_table",
        "U_matrix",         "V_matrix",                 "edge_weight_coefficients",
        "edge_weight_deltas", "edge_delay_ticks",       "edge_synapse_plane",
        "state_base",       "parameter_base",           "neuron_index",
        "neuron_count",
        "branching_factor", "superblock_size_words",    "padded_node_count",
        "tree_height",      "internal_bit_count",       "rank_float4_stride",
        "constant_weight",  "max_neighbor_count",       "ring_depth",
        "dt",               "tick",
    };
    static const Vector<String> initialize_parameters = {
        "cell_state",     "cell_parameters", "network_inputs", "last_spiked", "spike_flags",
        "state_base",     "parameter_base",  "neuron_index",   "dt",          "tick",
    };
    return purpose == KernelPurpose::Tick ? tick_parameters : initialize_parameters;
}

String kernel_function_name(KernelPurpose purpose) {
    return purpose == KernelPurpose::Tick ? "simulate_tick" : "initialize_cell_state";
}

String device_function_name(const String &cell_type_name, KernelPurpose purpose) {
    const String prefix =
            purpose == KernelPurpose::Tick ? "cell_type_step_" : "cell_type_initialize_";
    return prefix + sanitize_identifier(cell_type_name);
}

String source_preamble(KernelBackend backend, const String &generator_name) {
    ostringstream preamble;
    preamble << "// Generated by spikecorec::nml::" << generator_name << " -- do not edit.\n";
    if (backend == KernelBackend::Metal) {
        preamble << "#include <metal_stdlib>\n";
        preamble << "using namespace metal;\n";
    }
    preamble << "\ntypedef " << (backend == KernelBackend::Metal ? "long" : "long long") << " "
             << buffer_index_type_name << ";\n";
    preamble << "\n";
    return preamble.str();
}

// How many planes wide one network_inputs ring row is: the delivered-current plane every
// cell reads, plus one arrival plane per wired synapse prototype. Baked into the source
// rather than passed, because the argument table is full and it is a property of the model
// the source was generated from.
String network_input_plane_count_definition(const NML_ParseResult &parse_result) {
    return "#define SPIKECOREC_NETWORK_INPUT_PLANE_COUNT " +
           to_string(wired_synapse_prototype_indices(parse_result).size() + 1) + "\n\n";
}

// The one declaration in the argument table that carries `name`.
const KernelArgumentDeclaration &kernel_argument_declaration_for(const String &name) {
    for (const KernelArgumentDeclaration &declaration : kernel_argument_declarations()) {
        if (declaration.name == name) return declaration;
    }
    throw runtime_error("kernel_codegen: '" + name + "' names no kernel argument");
}

// How one device-function parameter is spelled. The three the master kernel resolves itself
// -- the thread's neuron and its two slot offsets -- are plain ints; everything else is
// declared exactly as the corresponding kernel argument, minus the `constant`/reference
// spelling a kernel parameter carries and a Metal address space is added back where the
// argument is a pointer.
String device_function_parameter_declaration(const String &parameter_name, KernelBackend backend) {
    if (parameter_name == "state_base" || parameter_name == "parameter_base" ||
        parameter_name == "neuron_index") {
        return "int " + parameter_name;
    }

    const bool is_metal = backend == KernelBackend::Metal;
    for (const KernelArgumentDeclaration &declaration : kernel_argument_declarations()) {
        if (declaration.name != parameter_name) continue;

        const String kernel_declaration =
                is_metal ? declaration.metal_declaration : declaration.cuda_declaration;

        // A scalar reaches a Metal kernel as `constant T &name`; a device function takes it
        // by value, which is also exactly the CUDA spelling.
        if (is_metal && kernel_declaration.rfind("constant ", 0) == 0) {
            String by_value = kernel_declaration.substr(String("constant ").length());
            const usize ampersand_position = by_value.find('&');
            if (ampersand_position != String::npos) by_value.erase(ampersand_position, 1);
            return by_value;
        }
        return kernel_declaration;
    }

    throw runtime_error("kernel_codegen: device function parameter '" + parameter_name +
                        "' names no kernel argument");
}

String emit_device_function(const String &function_name, const String &body,
                            KernelBackend backend, KernelPurpose purpose) {
    const bool is_metal = backend == KernelBackend::Metal;
    const Vector<String> &parameter_names = device_function_parameter_names(purpose);

    ostringstream function;
    function << (is_metal ? "inline void " : "__device__ inline void ") << function_name << "(\n";
    for (usize position = 0; position < parameter_names.size(); ++position) {
        function << indent(2)
                 << device_function_parameter_declaration(parameter_names[position], backend)
                 << (position + 1 < parameter_names.size() ? ",\n" : "\n");
    }
    function << ") {\n";
    function << body;
    function << "}\n\n";
    return function.str();
}

String emit_master_kernel(const Vector<String> &device_function_names,
                          const Vector<WiredSynapse> &wired_synapses,
                          const Set<String> &emitted_synapse_function_names,
                          KernelBackend backend, KernelPurpose purpose) {
    const bool is_metal = backend == KernelBackend::Metal;
    const Vector<KernelArgumentDeclaration> &declarations = kernel_argument_declarations();

    ostringstream kernel;

    // Both entry points take the identical signature so the engine binds one argument set
    // for both; the initialize kernel simply never reads the propagation half of it.
    kernel << (is_metal ? "kernel void " : "extern \"C\" __global__ void ")
           << kernel_function_name(purpose) << "(\n";
    for (usize position = 0; position < declarations.size(); ++position) {
        kernel << indent(1);
        if (is_metal) {
            kernel << declarations[position].metal_declaration << " [[ buffer(" << position
                   << ") ]],\n";
        } else {
            kernel << declarations[position].cuda_declaration
                   << (position + 1 < declarations.size() ? ",\n" : "\n");
        }
    }
    if (is_metal) kernel << indent(1) << "uint thread_id [[ thread_position_in_grid ]]\n";
    kernel << ") {\n";
    kernel << indent(1)
           << (is_metal ? "int neuron_index = (int)thread_id;\n"
                        : "int neuron_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n");

    kernel << indent(1) << "if (neuron_index >= neuron_count) return;\n\n";

    // Stage 4, Emit, is a per-tick flag: the dynamics raise it and nothing else lowers it, so
    // it is cleared here, on the device, by the one thread that owns the slot. Clearing it
    // from the host between dispatches instead would be a host write between two kernel
    // launches, which needs an explicit synchronisation on CUDA managed memory with
    // concurrentManagedAccess == 0. It cannot be cleared after the dispatch either: the flags
    // are this tick's output, read by the recorder and by callers once step_simulation
    // returns.
    if (purpose == KernelPurpose::Tick) {
        kernel << indent(1) << "spike_flags[neuron_index] = 0;\n\n";
    }

    // Stages 1 and 2 for the synapses converging on THIS neuron, ahead of the cell dynamics
    // and in this same thread. Every one of them writes only this neuron's slots, so the
    // current they deliver reaches the cell below through ordinary program order rather than
    // through any cross-thread synchronisation. Run for every neuron, not only the wired
    // targets: an unwired neuron's arrival slot and state are both zero, so its synapse
    // contributes zero, and knowing which neurons are targets would cost a per-neuron buffer.
    if (!wired_synapses.empty()) {
        for (const WiredSynapse &wired : wired_synapses) {
            if (emitted_synapse_function_names.count(
                        synapse_device_function_name(wired, purpose)) == 0) {
                continue;
            }
            kernel << synapse_device_function_call(wired, purpose, 1);
        }
        kernel << "\n";
    }

    kernel << indent(1) << "int state_base = cell_state_base[neuron_index];\n";
    kernel << indent(1) << "int parameter_base = cell_parameter_base[neuron_index];\n\n";
    kernel << indent(1) << "switch (cell_type_index[neuron_index]) {\n";

    String call_arguments;
    for (const String &parameter_name : device_function_parameter_names(purpose)) {
        if (!call_arguments.empty()) call_arguments += ", ";
        call_arguments += parameter_name;
    }

    for (usize type_index = 0; type_index < device_function_names.size(); ++type_index) {
        kernel << indent(2) << "case " << type_index << ":\n";
        kernel << indent(3) << device_function_names[type_index] << "(" << call_arguments
               << ");\n";
        kernel << indent(3) << "break;\n";
    }

    kernel << indent(2) << "default:\n";
    kernel << indent(3) << "break;\n";
    kernel << indent(1) << "}\n";
    kernel << "}\n";

    return kernel.str();
}

// ── the end-of-tick ring row clear ───────────────────────────────────────────

const String ring_row_clear_function_name = "clear_network_input_ring_row";

const Vector<String> &ring_row_clear_argument_names() {
    static const Vector<String> names = {"network_inputs", "neuron_count", "tick", "ring_depth"};
    return names;
}

GeneratedKernel generate_ring_row_clear(const NML_ParseResult &parse_result,
                                        KernelBackend backend) {
    const bool is_metal = backend == KernelBackend::Metal;
    const Vector<String> &argument_names = ring_row_clear_argument_names();

    ostringstream source;
    source << source_preamble(backend, "generate_ring_row_clear_kernel");
    source << network_input_plane_count_definition(parse_result);
    source << (is_metal ? "kernel void " : "extern \"C\" __global__ void ")
           << ring_row_clear_function_name << "(\n";

    for (usize position = 0; position < argument_names.size(); ++position) {
        const KernelArgumentDeclaration &declaration =
                kernel_argument_declaration_for(argument_names[position]);
        source << indent(1);
        if (is_metal) {
            source << declaration.metal_declaration << " [[ buffer(" << position << ") ]],\n";
        } else {
            source << declaration.cuda_declaration
                   << (position + 1 < argument_names.size() ? ",\n" : "\n");
        }
    }
    if (is_metal) source << indent(1) << "uint thread_id [[ thread_position_in_grid ]]\n";

    source << ") {\n"
           << indent(1)
           << (is_metal ? "int neuron_index = (int)thread_id;\n"
                        : "int neuron_index = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n")
           << indent(1) << "if (neuron_index >= neuron_count) return;\n\n"
           << indent(1) << "int ring_row = (int)(tick % (" << (is_metal ? "long" : "long long")
           << ")ring_depth);\n"
           // Every plane of the row, not just the delivered-current one: an arrival plane
           // this tick's synapse stage has already drained holds a value nothing will consume
           // again, and leaving it would have the ring re-deliver it a full lap later.
           << indent(1)
           << "int plane_base = ring_row * SPIKECOREC_NETWORK_INPUT_PLANE_COUNT;\n"
           << indent(1) << "for (int plane_index = 0;\n"
           << indent(3) << "plane_index < SPIKECOREC_NETWORK_INPUT_PLANE_COUNT; ++plane_index) {\n"
           << indent(2)
           << "network_inputs[(plane_base + plane_index) * neuron_count + neuron_index] = 0.0f;\n"
           << indent(1) << "}\n"
           << "}\n";

    GeneratedKernel generated;
    generated.source = source.str();
    generated.function_name = ring_row_clear_function_name;
    generated.argument_names = argument_names;
    return generated;
}

GeneratedKernel generate_kernel(const NML_ParseResult &parse_result, KernelBackend backend,
                                KernelPurpose purpose) {
    ostringstream source;
    source << source_preamble(backend, purpose == KernelPurpose::Tick
                                               ? "generate_tick_kernel"
                                               : "generate_initialize_kernel");

    // Only the tick entry point propagates, so only it carries the walk and the ring
    // helpers; emitting them into the initialize kernel would leave dead code behind.
    if (purpose == KernelPurpose::Tick) {
        source << network_input_plane_count_definition(parse_result);
        source << emit_propagation_helpers(backend);
    }

    // The synapses ahead of the cells, so a cell device function is free to read anything
    // they declare and so the source reads in the order the tick runs.
    const Vector<WiredSynapse> wired_synapses = collect_wired_synapses(parse_result);
    Set<String> emitted_synapse_function_names;

    for (const WiredSynapse &wired : wired_synapses) {
        const String function_name = synapse_device_function_name(wired, purpose);

        // Two prototype ids can sanitise onto one C identifier ("a-b" and "a_b"), which would
        // silently give one prototype the other's dynamics and parameters.
        if (emitted_synapse_function_names.count(function_name) > 0) {
            report_error("synapse prototype '" + wired.prototype->instance_id +
                                 "' collides with another prototype's after sanitising to '" +
                                 function_name + "'",
                         wired.type->name);
        }

        const String body = purpose == KernelPurpose::Tick
                                    ? emit_synapse_tick_body(wired, parse_result)
                                    : emit_synapse_initialize_body(wired, parse_result);

        // A synapse with no OnStart initialises nothing; the engine zeroes the buffer, so an
        // empty device function would only be dead code the master kernel still called.
        if (body.empty()) continue;

        source << emit_synapse_device_function(wired, body, backend, purpose);
        emitted_synapse_function_names.insert(function_name);
    }

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

        source << emit_device_function(function_name, body, backend, purpose);
        device_function_names.push_back(function_name);
    }

    source << emit_master_kernel(device_function_names, wired_synapses,
                                 emitted_synapse_function_names, backend, purpose);

    GeneratedKernel generated;
    generated.source = source.str();
    generated.function_name = kernel_function_name(purpose);
    generated.argument_names = kernel_argument_names();
    return generated;
}

} // namespace

// ── Public interface ─────────────────────────────────────────────────────────

s64 CellRegimeLayout::index_of(const String &regime_name) const {
    for (usize regime_index = 0; regime_index < regime_names.size(); ++regime_index) {
        if (regime_names[regime_index] == regime_name) return (s64)regime_index;
    }
    return -1;
}

CellRegimeLayout resolve_cell_regimes(const CellTypeSpecification &cell_type) {
    CellRegimeLayout layout;
    layout.regime_state_slot = cell_type.state_variable_names.size();

    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.source_tag != NML_DeclarationType::Regime) continue;

        // Two regimes of one name cannot both be reached: a Transition names a regime by
        // name, so one of them would be unreachable and the other would silently absorb its
        // TimeDerivatives and OnConditions.
        if (layout.index_of(instruction.target) >= 0) {
            report_error("Regime '" + instruction.target + "' is declared more than once",
                         cell_type.name);
        }

        // A Regime carries its initial= attribute in `expression` -- see
        // DynamicsInstruction. Absent or "false" means an ordinary regime.
        const String &initial_attribute = instruction.expression;
        const bool is_initial = initial_attribute == "true" || initial_attribute == "1";
        const bool is_not_initial = initial_attribute.empty() || initial_attribute == "false" ||
                                    initial_attribute == "0";
        if (!is_initial && !is_not_initial) {
            report_error("Regime '" + instruction.target + "' carries initial=\"" +
                                 initial_attribute + "\", which is neither true nor false",
                         cell_type.name);
        }

        if (is_initial) {
            if (layout.initial_regime_index >= 0) {
                report_error("Regime '" + instruction.target +
                                     "' is marked initial, and so is Regime '" +
                                     layout.regime_names[(usize)layout.initial_regime_index] +
                                     "'; a cell can only start in one",
                             cell_type.name);
            }
            layout.initial_regime_index = (s64)layout.regime_names.size();
        }

        layout.regime_names.push_back(instruction.target);
    }

    // Which regime a cell starts in decides whether it begins integrating or begins
    // refractory, which is a different model rather than a different detail. Defaulting to
    // the first declared one would run and be plausible, so it is refused instead.
    if (layout.has_regimes() && layout.initial_regime_index < 0) {
        report_error("this ComponentType declares " + to_string(layout.regime_names.size()) +
                             " Regime(s) but none is marked initial=\"true\", so there is "
                             "nothing to say which one a cell starts in",
                     cell_type.name);
    }

    return layout;
}

usize cell_state_slot_count(const CellTypeSpecification &cell_type) {
    return cell_type.state_variable_names.size() +
           (resolve_cell_regimes(cell_type).has_regimes() ? 1u : 0u);
}

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

GeneratedKernel generate_ring_row_clear_kernel(const NML_ParseResult &parse_result) {
    return generate_ring_row_clear(parse_result, active_backend());
}

Vector<s64> wired_synapse_prototype_indices(const NML_ParseResult &parse_result) {
    Set<s64> referenced_prototype_indices;
    for (const Neuron &neuron : parse_result.neurons) {
        for (const NetworkEdge &edge : neuron.outgoing_edges) {
            if (edge.synapse_prototype_index < 0) continue;
            referenced_prototype_indices.insert(edge.synapse_prototype_index);
        }
    }

    // Walked in prototype order rather than in the set's, so the numbering is a property of
    // the model rather than of the order the edges happened to be visited in.
    Vector<s64> ordered_prototype_indices;
    for (usize prototype_index = 0; prototype_index < parse_result.synapse_prototypes.size();
         ++prototype_index) {
        if (referenced_prototype_indices.count((s64)prototype_index) == 0) continue;
        ordered_prototype_indices.push_back((s64)prototype_index);
    }

    return ordered_prototype_indices;
}

} // namespace spikecorec::nml
