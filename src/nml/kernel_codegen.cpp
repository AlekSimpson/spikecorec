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
// drained exactly once per cell, into this local, so a model that reduces over its synapses
// more than once reads the same delivered current each time instead of finding the slot
// already emptied by its own earlier read.
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

    // The ring is one flat allocation indexed as [row][neuron]; `tick_of_arrival` is the
    // tick whose row is wanted, which is `tick` for a drain and `tick + delay` for an
    // arrival.
    helpers << function_prefix << "int network_input_ring_index(\n"
            << indent(2) << tick_type << " tick_of_arrival,\n"
            << indent(2) << "int ring_depth,\n"
            << indent(2) << "int neuron_count,\n"
            << indent(2) << "int neuron_index\n"
            << ") {\n"
            << indent(1) << "int ring_row = (int)(tick_of_arrival % (" << tick_type
            << ")ring_depth);\n"
            << indent(1) << "return ring_row * neuron_count + neuron_index;\n"
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
            << indent(2) << "int arrival_index = network_input_ring_index(\n"
            << indent(4) << "tick + (" << tick_type << ")edge_delay_ticks[edge_slot], ring_depth,\n"
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

// Reads this tick's ring row for this neuron and clears it in one step, so the row is empty
// when the ring wraps back onto it -- an uncleared row would re-deliver its contents
// ring_depth ticks later, which reads as a plausible oscillation rather than as a bug.
String emit_synaptic_input_drain(KernelBackend backend, usize indent_level) {
    ostringstream drain;
    drain << indent(indent_level) << "int synaptic_input_index = network_input_ring_index(\n"
          << indent(indent_level + 2)
          << "tick, ring_depth, neuron_count, neuron_index);\n";

    if (backend == KernelBackend::Metal) {
        drain << indent(indent_level) << "float " << synaptic_input_local_name
              << " = atomic_exchange_explicit(\n"
              << indent(indent_level + 2)
              << "(device atomic_float *)(network_inputs + synaptic_input_index), 0.0f,\n"
              << indent(indent_level + 2) << "memory_order_relaxed);\n";
    } else {
        drain << indent(indent_level) << "float " << synaptic_input_local_name
              << " = atomicExch(network_inputs + synaptic_input_index, 0.0f);\n";
    }

    return drain.str();
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
             << "edge_weight_deltas, edge_delay_ticks, branching_factor, superblock_size_words,\n"
             << indent(indent_level + 3)
             << "padded_node_count, tree_height, internal_bit_count, rank_float4_stride,\n"
             << indent(indent_level + 3)
             << "constant_weight, max_neighbor_count, ring_depth, neuron_count, neuron_index,\n"
             << indent(indent_level + 3) << "tick);\n"
             << indent(indent_level) << "}\n";
    return epilogue.str();
}

// ── Per-cell-type bodies ─────────────────────────────────────────────────────

String emit_tick_body(const CellTypeSpecification &cell_type, const NML_ParseResult &parse_result,
                      KernelBackend backend) {
    reject_unsupported_instructions(cell_type);

    ostringstream body;
    SymbolTable symbols = build_base_symbol_table(cell_type);

    // Integrate, part 1: DerivedVariables become locals in source order, each visible to
    // the ones after it. A DerivedVariable referencing one declared later resolves to
    // nothing and throws, rather than emitting a forward reference the shader compiler
    // would reject far from its cause.
    //
    // A select= path becomes a local the same way, so whatever it binds to is reached by
    // the name the model gave it and resolves through the ordinary precedence chain. Every
    // such path binds to the ONE drain emitted ahead of them all, so a model reducing over
    // its synapses twice reads the same delivered current twice rather than finding the ring
    // slot already emptied by its own earlier read. The declarations are staged rather than
    // written straight out because whether the drain is needed is only known once they have
    // all been walked.
    ostringstream derived_variable_declarations;
    bool drains_synaptic_input = false;

    for (const DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.stage != DynamicsStage::Integrate) continue;
        if (instruction.source_tag != NML_DeclarationType::DerivedVariable) continue;

        const SymbolTable visible_symbols = with_fallback_symbols(symbols, parse_result);
        const Optional<String> path_head_name =
                select_path_head_name(instruction.expression, visible_symbols);

        String read_expression;
        if (path_head_name.has_value()) {
            require_synaptic_select_path(instruction.expression, *path_head_name, cell_type.name);
            drains_synaptic_input = true;
            read_expression = synaptic_input_local_name;
        } else {
            read_expression = translate_expression(instruction.expression, visible_symbols);
        }

        const String local_name = "derived_" + sanitize_identifier(instruction.target);
        derived_variable_declarations << indent(1) << "float " << local_name << " = "
                                      << read_expression << ";\n";
        symbols.define(instruction.target, local_name);
    }

    // Stage 1, Deliver, for this neuron: its own ring row, read and cleared once.
    if (drains_synaptic_input) body << emit_synaptic_input_drain(backend, 1);
    body << derived_variable_declarations.str();

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

    // Stage 6, Propagate. Unconditional boilerplate: every cell type scatters its spike the
    // same way, so this is emitted whether or not the type declares an EventOut -- a type
    // that never raises the flag simply never enters the block.
    body << emit_propagation_epilogue(1);

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
        "edge_weight_deltas", "edge_delay_ticks",       "state_base",
        "parameter_base",   "neuron_index",             "neuron_count",
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

String emit_master_kernel(const Vector<String> &device_function_names, KernelBackend backend,
                          KernelPurpose purpose) {
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

GeneratedKernel generate_kernel(const NML_ParseResult &parse_result, KernelBackend backend,
                                KernelPurpose purpose) {
    ostringstream source;
    source << source_preamble(backend, purpose);

    // Only the tick entry point propagates, so only it carries the walk and the ring
    // helpers; emitting them into the initialize kernel would leave dead code behind.
    if (purpose == KernelPurpose::Tick) source << emit_propagation_helpers(backend);

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
                                    ? emit_tick_body(cell_type, parse_result, backend)
                                    : emit_initialize_body(cell_type, parse_result);

        source << emit_device_function(function_name, body, backend, purpose);
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
