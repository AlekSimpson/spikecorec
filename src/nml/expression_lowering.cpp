#include <cctype>
#include <cstdio>
#include <unordered_set>

#include "spikecorec/nml/expression_lowering.h"
#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::nml {

String get_attribute_value(const NML_Node &node, const String &attribute_name) {
    auto entry = node.attributes.find(attribute_name);
    if (entry == node.attributes.end()) return "";
    return std::any_cast<String>(entry->second);
}

Vector<const NML_Node *> find_children(const NML_Node &node, const String &tag_name) {
    Vector<const NML_Node *> matches;
    for (const auto &child : node.body) {
        if (child.tag_name == tag_name) matches.push_back(&child);
    }
    return matches;
}

namespace {

// ── AST node factories (module-private -- only the parser below builds
// these; callers only ever see the resulting ExpressionNodePointer) ──────

ExpressionNodePointer make_number_node(String literal_text) {
    auto node = std::make_unique<ExpressionNode>();
    node->kind = ExpressionNodeKind::Number;
    node->text = std::move(literal_text);
    return node;
}

ExpressionNodePointer make_identifier_node(String identifier_name) {
    auto node = std::make_unique<ExpressionNode>();
    node->kind = ExpressionNodeKind::Identifier;
    node->text = std::move(identifier_name);
    return node;
}

ExpressionNodePointer make_negate_node(ExpressionNodePointer operand) {
    auto node = std::make_unique<ExpressionNode>();
    node->kind = ExpressionNodeKind::Negate;
    node->left = std::move(operand);
    return node;
}

ExpressionNodePointer make_binary_node(char operator_character, ExpressionNodePointer left_operand, ExpressionNodePointer right_operand) {
    auto node = std::make_unique<ExpressionNode>();
    node->kind = ExpressionNodeKind::Binary;
    node->binary_operator_character = operator_character;
    node->left = std::move(left_operand);
    node->right = std::move(right_operand);
    return node;
}

enum class TokenKind {
    Number, Identifier, Plus, Minus, Star, Slash, LeftParen, RightParen,
    CompareGt, CompareLt, CompareGe, CompareLe, CompareEq, CompareNe, End
};

struct Token {
    TokenKind kind;
    String text;
};

bool is_comparison_token(TokenKind kind) {
    return kind == TokenKind::CompareGt || kind == TokenKind::CompareLt || kind == TokenKind::CompareGe ||
           kind == TokenKind::CompareLe || kind == TokenKind::CompareEq || kind == TokenKind::CompareNe;
}

// Longest-prefix-first so `.geq.`/`.leq.`/`.neq.` are never mistaken for a
// truncated match of `.eq.` (none of the six is otherwise a prefix of
// another).
const Vector<std::pair<String, TokenKind>> DOT_COMPARISON_OPERATORS = {
    {".geq.", TokenKind::CompareGe}, {".leq.", TokenKind::CompareLe}, {".neq.", TokenKind::CompareNe},
    {".gt.", TokenKind::CompareGt}, {".lt.", TokenKind::CompareLt}, {".eq.", TokenKind::CompareEq},
};

Vector<Token> tokenize_expression(const String &expression_text, const String &context_for_errors) {
    Vector<Token> tokens;
    usize position = 0;

    while (position < expression_text.size()) {
        char character = expression_text[position];
        if (std::isspace(static_cast<unsigned char>(character))) { ++position; continue; }

        if (character == '.') {
            bool matched_operator = false;
            for (const auto &[operator_text, operator_kind] : DOT_COMPARISON_OPERATORS) {
                if (expression_text.compare(position, operator_text.size(), operator_text) == 0) {
                    tokens.push_back(Token{operator_kind, operator_text});
                    position += operator_text.size();
                    matched_operator = true;
                    break;
                }
            }
            if (matched_operator) continue;
            log::throw_runtime_error(log::logger(),
                "expression_lowering: unrecognized '.' operator in expression '" + expression_text + "' (" + context_for_errors + ")");
        }

        if (std::isdigit(static_cast<unsigned char>(character))) {
            usize digit_start = position;
            while (position < expression_text.size() &&
                   (std::isdigit(static_cast<unsigned char>(expression_text[position])) || expression_text[position] == '.')) {
                ++position;
            }
            if (position < expression_text.size() && (expression_text[position] == 'e' || expression_text[position] == 'E')) {
                usize exponent_start = position;
                usize lookahead = position + 1;
                if (lookahead < expression_text.size() && (expression_text[lookahead] == '+' || expression_text[lookahead] == '-')) ++lookahead;
                if (lookahead < expression_text.size() && std::isdigit(static_cast<unsigned char>(expression_text[lookahead]))) {
                    position = lookahead;
                    while (position < expression_text.size() && std::isdigit(static_cast<unsigned char>(expression_text[position]))) ++position;
                } else {
                    position = exponent_start; // the trailing 'e'/'E' wasn't actually an exponent suffix
                }
            }
            tokens.push_back(Token{TokenKind::Number, expression_text.substr(digit_start, position - digit_start)});
            continue;
        }

        if (std::isalpha(static_cast<unsigned char>(character)) || character == '_') {
            usize identifier_start = position;
            while (position < expression_text.size() &&
                   (std::isalnum(static_cast<unsigned char>(expression_text[position])) || expression_text[position] == '_')) {
                ++position;
            }
            tokens.push_back(Token{TokenKind::Identifier, expression_text.substr(identifier_start, position - identifier_start)});
            continue;
        }

        switch (character) {
            case '+': tokens.push_back(Token{TokenKind::Plus, "+"}); break;
            case '-': tokens.push_back(Token{TokenKind::Minus, "-"}); break;
            case '*': tokens.push_back(Token{TokenKind::Star, "*"}); break;
            case '/': tokens.push_back(Token{TokenKind::Slash, "/"}); break;
            case '(': tokens.push_back(Token{TokenKind::LeftParen, "("}); break;
            case ')': tokens.push_back(Token{TokenKind::RightParen, ")"}); break;
            default:
                log::throw_runtime_error(log::logger(),
                    "expression_lowering: unexpected character '" + String(1, character) + "' in expression '" +
                    expression_text + "' (" + context_for_errors + ")");
        }
        ++position;
    }

    tokens.push_back(Token{TokenKind::End, ""});
    return tokens;
}

// Recursive-descent over an already-tokenized (sub)expression -- precedence,
// lowest to highest: additive (`+ -`), multiplicative (`* /`), unary minus,
// primary (number / identifier / parenthesized expression). Comparisons are
// handled one level up, in parse_condition_text, since LEMS forbids nested
// comparisons (a `test=` string has exactly one top-level compare operator).
class ExpressionParser {
public:
    ExpressionParser(const Vector<Token> &tokens, const String &context_for_errors)
        : tokens_(tokens), context_for_errors_(context_for_errors) {}

    ExpressionNodePointer parse_full_expression() {
        ExpressionNodePointer result = parse_additive();
        if (current_token().kind != TokenKind::End) {
            log::throw_runtime_error(log::logger(),
                "expression_lowering: unexpected trailing token '" + current_token().text + "' (" + context_for_errors_ + ")");
        }
        return result;
    }

private:
    const Vector<Token> &tokens_;
    usize position_ = 0;
    const String &context_for_errors_;

    const Token &current_token() const { return tokens_[position_]; }
    Token advance_token() { return tokens_[position_++]; }

    ExpressionNodePointer parse_additive() {
        ExpressionNodePointer left = parse_multiplicative();
        while (current_token().kind == TokenKind::Plus || current_token().kind == TokenKind::Minus) {
            char operator_character = advance_token().kind == TokenKind::Plus ? '+' : '-';
            ExpressionNodePointer right = parse_multiplicative();
            left = make_binary_node(operator_character, std::move(left), std::move(right));
        }
        return left;
    }

    ExpressionNodePointer parse_multiplicative() {
        ExpressionNodePointer left = parse_unary();
        while (current_token().kind == TokenKind::Star || current_token().kind == TokenKind::Slash) {
            char operator_character = advance_token().kind == TokenKind::Star ? '*' : '/';
            ExpressionNodePointer right = parse_unary();
            left = make_binary_node(operator_character, std::move(left), std::move(right));
        }
        return left;
    }

    ExpressionNodePointer parse_unary() {
        if (current_token().kind == TokenKind::Minus) {
            advance_token();
            return make_negate_node(parse_unary());
        }
        return parse_primary();
    }

    ExpressionNodePointer parse_primary() {
        if (current_token().kind == TokenKind::Number) return make_number_node(advance_token().text);
        if (current_token().kind == TokenKind::Identifier) return make_identifier_node(advance_token().text);
        if (current_token().kind == TokenKind::LeftParen) {
            advance_token();
            ExpressionNodePointer inner = parse_additive();
            if (current_token().kind != TokenKind::RightParen) {
                log::throw_runtime_error(log::logger(), "expression_lowering: expected ')' (" + context_for_errors_ + ")");
            }
            advance_token();
            return inner;
        }
        log::throw_runtime_error(log::logger(),
            "expression_lowering: unexpected token '" + current_token().text + "' (" + context_for_errors_ + ")");
    }
};

BinaryOpcode opcode_for_comparison_token(TokenKind kind) {
    switch (kind) {
        case TokenKind::CompareGt: return BinaryOpcode::Gt;
        case TokenKind::CompareLt: return BinaryOpcode::Lt;
        case TokenKind::CompareGe: return BinaryOpcode::Ge;
        case TokenKind::CompareLe: return BinaryOpcode::Le;
        case TokenKind::CompareEq: return BinaryOpcode::Eq;
        case TokenKind::CompareNe: return BinaryOpcode::Ne;
        default:
            log::throw_runtime_error(log::logger(), "expression_lowering: internal error -- not a comparison token");
    }
}

const std::unordered_set<String> RESERVED_ENGINE_NAMES = {"dt", "tick", "network_inputs"};

BinaryOpcode binary_opcode_for(char operator_character) {
    switch (operator_character) {
        case '+': return BinaryOpcode::Add;
        case '-': return BinaryOpcode::Sub;
        case '*': return BinaryOpcode::Mul;
        case '/': return BinaryOpcode::Div;
        default:
            log::throw_runtime_error(log::logger(), "expression_lowering: internal error -- unhandled binary operator");
    }
}

} // namespace

ExpressionNodePointer parse_arithmetic_text(const String &expression_text, const String &context_for_errors) {
    Vector<Token> tokens = tokenize_expression(expression_text, context_for_errors);
    return ExpressionParser(tokens, context_for_errors).parse_full_expression();
}

ParsedCondition parse_condition_text(const String &condition_text) {
    String context_for_errors = "condition '" + condition_text + "'";
    Vector<Token> tokens = tokenize_expression(condition_text, context_for_errors);

    s32 comparison_token_index = -1;
    for (usize token_index = 0; token_index < tokens.size(); ++token_index) {
        if (!is_comparison_token(tokens[token_index].kind)) continue;
        if (comparison_token_index >= 0) {
            log::throw_runtime_error(log::logger(),
                "expression_lowering: condition '" + condition_text + "' has more than one comparison operator");
        }
        comparison_token_index = static_cast<s32>(token_index);
    }
    if (comparison_token_index < 0) {
        log::throw_runtime_error(log::logger(),
            "expression_lowering: condition '" + condition_text + "' has no comparison operator");
    }

    Vector<Token> left_tokens(tokens.begin(), tokens.begin() + comparison_token_index);
    left_tokens.push_back(Token{TokenKind::End, ""});
    Vector<Token> right_tokens(tokens.begin() + comparison_token_index + 1, tokens.end());

    ParsedCondition parsed;
    parsed.opcode = opcode_for_comparison_token(tokens[static_cast<usize>(comparison_token_index)].kind);
    parsed.left = ExpressionParser(left_tokens, context_for_errors).parse_full_expression();
    parsed.right = ExpressionParser(right_tokens, context_for_errors).parse_full_expression();
    return parsed;
}

bool references_identifier(const ExpressionNode &node, const String &identifier_name) {
    switch (node.kind) {
        case ExpressionNodeKind::Identifier: return node.text == identifier_name;
        case ExpressionNodeKind::Number: return false;
        case ExpressionNodeKind::Negate: return references_identifier(*node.left, identifier_name);
        case ExpressionNodeKind::Binary:
            return references_identifier(*node.left, identifier_name) || references_identifier(*node.right, identifier_name);
    }
    return false;
}

LoweringContext::LoweringContext(std::unordered_set<String> known_names, UnorderedMap<String, String> identifier_aliases)
    : known_names_(std::move(known_names)), identifier_aliases_(std::move(identifier_aliases)) {}

String LoweringContext::resolve_identifier(const String &identifier_name) const {
    auto alias_entry = identifier_aliases_.find(identifier_name);
    if (alias_entry != identifier_aliases_.end()) return alias_entry->second;
    if (RESERVED_ENGINE_NAMES.count(identifier_name)) return identifier_name;
    if (known_names_.count(identifier_name)) return identifier_name;
    log::throw_runtime_error(log::logger(),
        "expression_lowering: expression references undeclared identifier '" + identifier_name + "'");
}

String LoweringContext::fresh_temporary() { return "t" + std::to_string(temporary_counter_++); }

void LoweringContext::reset_temporary_counter() { temporary_counter_ = 0; }

// Register-reuse strategy: when combining two operands, a LEAF operand (a
// bare identifier/number) is resolved without emitting an instruction, and a
// COMPOUND operand is computed directly into the combining instruction's own
// destination register (an accumulator), avoiding a second temporary
// whenever only one side needs one -- this reproduces the locked IR spec's
// own GLIF1 example almost exactly (one `t0` reused across the whole
// numerator/denominator chain). Only when BOTH operands are compound does a
// second, independent temporary get allocated (matching the NMDA example's
// `t0`/`t1` pair).
//
// Self-reference guard (fixes a register-aliasing hazard flagged in ticket
// #50's review): a single leaf-leaf combine emits exactly one `dst,a,b`
// instruction, so it is always safe regardless of aliasing -- reads and the
// write happen atomically. The hazard only exists once a COMPOUND operand
// needs its own multi-instruction evaluation and that evaluation is
// accumulated directly into `*preferred_destination`: if
// `*preferred_destination`'s name is read anywhere else in `node` (a plain
// sibling leaf with that same name, or a reference buried inside the OTHER,
// not-yet-evaluated operand), that read would silently observe the compound
// evaluation's clobbered intermediate value instead of the true original --
// e.g. a saturating-synapse-style update `g <- g + gbase*(1-g)`, or
// `v <- v + v*k`, lowered directly into `g`/`v`, would each silently
// miscompute (concretely: `g <- 2*gbase*(1-g)` instead of
// `g <- g + gbase*(1-g)`) with no error thrown. Whenever this is possible,
// the whole node is instead evaluated into an independent fresh temporary --
// a `tN` name the lowering itself introduces and so can never collide with a
// real LEMS identifier, making it unconditionally safe -- and the result is
// copied into the real destination with one final `mov`. This costs one
// extra instruction only in the (narrow) self-referential-and-compound case;
// every other shape (including a self-referential but purely leaf-leaf
// combine like `asc1 <- asc1 + ascAdd1`) is untouched and keeps the
// single-instruction form.
String LoweringContext::emit_expression(const ExpressionNode &node, Vector<TickInstruction> &output,
                       std::optional<String> preferred_destination) {
    if (preferred_destination.has_value() && node.kind == ExpressionNodeKind::Binary) {
        bool left_is_leaf = node.left->kind == ExpressionNodeKind::Number || node.left->kind == ExpressionNodeKind::Identifier;
        bool right_is_leaf = node.right->kind == ExpressionNodeKind::Number || node.right->kind == ExpressionNodeKind::Identifier;
        bool has_a_compound_operand = !left_is_leaf || !right_is_leaf;
        if (has_a_compound_operand && references_identifier(node, *preferred_destination)) {
            String scratch_result = emit_expression(node, output, std::nullopt);
            if (scratch_result != *preferred_destination) {
                output.push_back(MoveInstruction{*preferred_destination, scratch_result});
            }
            return *preferred_destination;
        }
    }

    switch (node.kind) {
        case ExpressionNodeKind::Number: {
            if (!preferred_destination) return node.text;
            output.push_back(MoveInstruction{*preferred_destination, node.text});
            return *preferred_destination;
        }
        case ExpressionNodeKind::Identifier: {
            String resolved_name = resolve_identifier(node.text);
            if (!preferred_destination || *preferred_destination == resolved_name) return resolved_name;
            output.push_back(MoveInstruction{*preferred_destination, resolved_name});
            return *preferred_destination;
        }
        case ExpressionNodeKind::Negate: {
            String operand_name = emit_expression(*node.left, output, std::nullopt);
            String destination_name = preferred_destination.value_or(fresh_temporary());
            output.push_back(UnaryInstruction{UnaryOpcode::Neg, destination_name, operand_name});
            return destination_name;
        }
        case ExpressionNodeKind::Binary: {
            bool left_is_leaf = node.left->kind == ExpressionNodeKind::Number || node.left->kind == ExpressionNodeKind::Identifier;
            bool right_is_leaf = node.right->kind == ExpressionNodeKind::Number || node.right->kind == ExpressionNodeKind::Identifier;

            String destination_name;
            String left_operand_name;
            String right_operand_name;

            if (left_is_leaf && right_is_leaf) {
                left_operand_name = emit_expression(*node.left, output, std::nullopt);
                right_operand_name = emit_expression(*node.right, output, std::nullopt);
                destination_name = preferred_destination.value_or(fresh_temporary());
            } else if (left_is_leaf) {
                left_operand_name = emit_expression(*node.left, output, std::nullopt);
                destination_name = preferred_destination.value_or(fresh_temporary());
                right_operand_name = emit_expression(*node.right, output, destination_name);
            } else if (right_is_leaf) {
                destination_name = preferred_destination.value_or(fresh_temporary());
                left_operand_name = emit_expression(*node.left, output, destination_name);
                right_operand_name = emit_expression(*node.right, output, std::nullopt);
            } else {
                destination_name = preferred_destination.value_or(fresh_temporary());
                left_operand_name = emit_expression(*node.left, output, destination_name);
                right_operand_name = emit_expression(*node.right, output, std::nullopt);
            }

            output.push_back(BinaryInstruction{binary_opcode_for(node.binary_operator_character),
                                                 destination_name, left_operand_name, right_operand_name});
            return destination_name;
        }
    }
    log::throw_runtime_error(log::logger(), "expression_lowering: internal error -- unhandled expression node kind");
}

// ── Integration-method selection (IR spec §3.3 `expdecay dst,a,tau` =
// `a*exp(-dt/tau)`) ───────────────────────────────────────────────────────
//
// `expdecay` has NO built-in "target" term -- it only computes a pure
// multiplicative decay of its own operand toward zero (the expOne synapse
// example's `expdecay g, g, tau` is exactly `dg/dt = -g/tau`). A
// `TimeDerivative` shaped `(target - state)/tau` (target != 0) still decays
// exponentially, but toward `target`, not zero -- its closed form is
// `state' = target + (state-target)*exp(-dt/tau)`, i.e. `(state-target)`
// itself is what decays to zero. So a non-zero target needs a 3-instruction
// sequence (subtract the target out, `expdecay`, add it back); a zero target
// (the after-spike-current case, `-state/tau`) collapses to the
// single-instruction form matching the spec's own example exactly.

// (LinearDecayShape / detect_linear_decay_shape declared in
// expression_lowering.h -- ticket #51 revision exposed them so
// synapse_lowering.cpp's per-edge decay writeback can reuse the same shape
// detection; see the header's doc comment. GLIF1's own `v` dynamics has an
// extra `network_inputs` term and so correctly does NOT match shape 1 below.)
std::optional<LinearDecayShape> detect_linear_decay_shape(const ExpressionNode &right_hand_side, const String &state_variable_name) {
    if (right_hand_side.kind != ExpressionNodeKind::Binary || right_hand_side.binary_operator_character != '/') return std::nullopt;
    const ExpressionNode &numerator = *right_hand_side.left;
    const ExpressionNode &denominator = *right_hand_side.right;

    bool denominator_is_leaf = denominator.kind == ExpressionNodeKind::Identifier || denominator.kind == ExpressionNodeKind::Number;
    if (!denominator_is_leaf) return std::nullopt;
    String time_constant = denominator.text;

    if (numerator.kind == ExpressionNodeKind::Binary && numerator.binary_operator_character == '-' &&
        numerator.right->kind == ExpressionNodeKind::Identifier && numerator.right->text == state_variable_name) {
        const ExpressionNode &target_node = *numerator.left;
        bool target_is_leaf = target_node.kind == ExpressionNodeKind::Identifier || target_node.kind == ExpressionNodeKind::Number;
        if (target_is_leaf && target_node.text != state_variable_name && !references_identifier(target_node, "network_inputs")) {
            return LinearDecayShape{target_node.text, time_constant};
        }
        return std::nullopt;
    }

    if (numerator.kind == ExpressionNodeKind::Negate &&
        numerator.left->kind == ExpressionNodeKind::Identifier && numerator.left->text == state_variable_name) {
        return LinearDecayShape{"0", time_constant};
    }

    return std::nullopt;
}

void lower_time_derivative(const String &state_variable_name, const String &right_hand_side_text,
                            Vector<TickInstruction> &output, LoweringContext &context) {
    context.reset_temporary_counter();
    String context_for_errors = "TimeDerivative '" + state_variable_name + "'";
    ExpressionNodePointer right_hand_side = parse_arithmetic_text(right_hand_side_text, context_for_errors);

    if (auto decay_shape = detect_linear_decay_shape(*right_hand_side, state_variable_name)) {
        if (decay_shape->target == "0") {
            output.push_back(BinaryInstruction{BinaryOpcode::ExpDecay, state_variable_name, state_variable_name, decay_shape->time_constant});
        } else {
            output.push_back(BinaryInstruction{BinaryOpcode::Sub, state_variable_name, state_variable_name, decay_shape->target});
            output.push_back(BinaryInstruction{BinaryOpcode::ExpDecay, state_variable_name, state_variable_name, decay_shape->time_constant});
            output.push_back(BinaryInstruction{BinaryOpcode::Add, state_variable_name, state_variable_name, decay_shape->target});
        }
        return;
    }

    // Forward Euler fallback (matches the locked IR spec's own GLIF1
    // example's instruction shape exactly): accumulate the whole
    // right-hand-side value into one temporary, scale by `dt`, add to state.
    String accumulator_name = context.fresh_temporary();
    context.emit_expression(*right_hand_side, output, accumulator_name);
    output.push_back(BinaryInstruction{BinaryOpcode::Mul, accumulator_name, accumulator_name, "dt"});
    output.push_back(BinaryInstruction{BinaryOpcode::Add, state_variable_name, state_variable_name, accumulator_name});
}

String format_literal(f64 value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    return String(buffer);
}

} // namespace spikecorec::nml
