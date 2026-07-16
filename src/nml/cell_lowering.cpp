#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>

#include "spikecorec/nml/cell_lowering.h"
#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::nml {

namespace {

// ── Small NML_Node helpers (mirrors nml.cpp's own private helpers; not
// shared across translation units since nml.cpp keeps them in its own
// anonymous namespace -- same rationale resolve.cpp already gives for its
// own mirror of them). `RegimeDecl::body`/`OnConditionDecl::body` are the
// raw `<Regime>`/`<OnCondition>` node (ticket #2/#49 keep Dynamics-nested
// control bodies tree-shaped), so this file has to walk them itself. ─────

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

// ── Minimal LEMS math-expression parser (ticket #50's "hard part") ───────
//
// Scoped deliberately to what GLIF1-5 actually need (see cell_lowering.h's
// header comment and this ticket's own scoping instructions): arithmetic
// `+ - * /`, parentheses, unary minus, and the six LEMS dot-operator
// comparisons (`.gt. .lt. .geq. .leq. .eq. .neq.`). No `^`/pow and no
// function-call syntax (`exp(...)` etc.) -- none of the GLIF1-5 fixtures
// below need them: after-spike-current and threshold decay are expressed as
// ODEs lowered via `TimeDerivative` (to `expdecay`, see below), never as a
// closed-form `exp()` call inside the expression text itself. Extending this
// parser for a future ComponentType that genuinely needs pow/exp/trig inside
// an expression is out of this ticket's scope.

enum class ExpressionNodeKind { Number, Identifier, Negate, Binary };

struct ExpressionNode {
    ExpressionNodeKind kind;
    String text;                       // Number literal text, or Identifier name
    char binary_operator_character = '\0'; // '+' '-' '*' '/' for Binary
    std::unique_ptr<ExpressionNode> left;
    std::unique_ptr<ExpressionNode> right; // unused for Negate (operand is `left`)
};

using ExpressionNodePointer = std::unique_ptr<ExpressionNode>;

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
                "cell_lowering: unrecognized '.' operator in expression '" + expression_text + "' (" + context_for_errors + ")");
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
                    "cell_lowering: unexpected character '" + String(1, character) + "' in expression '" +
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
                "cell_lowering: unexpected trailing token '" + current_token().text + "' (" + context_for_errors_ + ")");
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
                log::throw_runtime_error(log::logger(), "cell_lowering: expected ')' (" + context_for_errors_ + ")");
            }
            advance_token();
            return inner;
        }
        log::throw_runtime_error(log::logger(),
            "cell_lowering: unexpected token '" + current_token().text + "' (" + context_for_errors_ + ")");
    }
};

ExpressionNodePointer parse_arithmetic_text(const String &expression_text, const String &context_for_errors) {
    Vector<Token> tokens = tokenize_expression(expression_text, context_for_errors);
    return ExpressionParser(tokens, context_for_errors).parse_full_expression();
}

struct ParsedCondition {
    ExpressionNodePointer left;
    BinaryOpcode opcode;
    ExpressionNodePointer right;
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
            log::throw_runtime_error(log::logger(), "cell_lowering: internal error -- not a comparison token");
    }
}

ParsedCondition parse_condition_text(const String &condition_text) {
    String context_for_errors = "condition '" + condition_text + "'";
    Vector<Token> tokens = tokenize_expression(condition_text, context_for_errors);

    s32 comparison_token_index = -1;
    for (usize token_index = 0; token_index < tokens.size(); ++token_index) {
        if (!is_comparison_token(tokens[token_index].kind)) continue;
        if (comparison_token_index >= 0) {
            log::throw_runtime_error(log::logger(),
                "cell_lowering: condition '" + condition_text + "' has more than one comparison operator");
        }
        comparison_token_index = static_cast<s32>(token_index);
    }
    if (comparison_token_index < 0) {
        log::throw_runtime_error(log::logger(),
            "cell_lowering: condition '" + condition_text + "' has no comparison operator");
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

// ── Identifier resolution + expression codegen ───────────────────────────

const std::unordered_set<String> RESERVED_ENGINE_NAMES = {"dt", "tick", "network_inputs"};

// Whether `identifier_name` is read anywhere inside `node`'s subtree. Used
// both by the integration-method shape detector below and by
// `LoweringContext::emit_expression`'s self-reference guard (a name defined
// ahead of that class since both need it).
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

// Holds everything needed to turn one cell type's parsed expressions into
// `.tick` instructions: which names are legal to reference (state variables,
// parameters, plain-value derived variables), which names are aliases for
// the engine's `network_inputs` (arch §3.2 DerivedVariable: a `select`/
// `reduce="add"` aggregation over attached synapses collapses to one
// `network_inputs` read), and a fresh-temporary counter for `t0, t1, ...`
// intermediates (IR spec §3.1).
class LoweringContext {
public:
    LoweringContext(std::unordered_set<String> known_names, UnorderedMap<String, String> network_input_aliases)
        : known_names_(std::move(known_names)), network_input_aliases_(std::move(network_input_aliases)) {}

    String resolve_identifier(const String &identifier_name) const {
        auto alias_entry = network_input_aliases_.find(identifier_name);
        if (alias_entry != network_input_aliases_.end()) return alias_entry->second;
        if (RESERVED_ENGINE_NAMES.count(identifier_name)) return identifier_name;
        if (known_names_.count(identifier_name)) return identifier_name;
        log::throw_runtime_error(log::logger(),
            "cell_lowering: expression references undeclared identifier '" + identifier_name + "'");
    }

    String fresh_temporary() { return "t" + std::to_string(temporary_counter_++); }

    // Each top-level Dynamics statement (one TimeDerivative, one
    // DerivedVariable, one StateAssignment, one condition test) is fully
    // self-contained -- its own `t0, t1, ...` scratch is dead the moment the
    // statement's final instruction runs -- so the counter resets per
    // statement rather than growing across the whole program (matching the
    // locked IR spec's own examples, which never need more than a couple of
    // reused `tN` names).
    void reset_temporary_counter() { temporary_counter_ = 0; }

    // Lowers `node`'s value into `output`, writing the final result into
    // `preferred_destination` if given (else a fresh `tN` register).
    // Returns the operand name holding the result.
    //
    // Register-reuse strategy: when combining two operands, a LEAF operand
    // (a bare identifier/number) is resolved without emitting an
    // instruction, and a COMPOUND operand is computed directly into the
    // combining instruction's own destination register (an accumulator),
    // avoiding a second temporary whenever only one side needs one -- this
    // reproduces the locked IR spec's own GLIF1 example almost exactly (one
    // `t0` reused across the whole numerator/denominator chain). Only when
    // BOTH operands are compound does a second, independent temporary get
    // allocated (matching the NMDA example's `t0`/`t1` pair).
    //
    // Self-reference guard (fixes a register-aliasing hazard flagged in
    // ticket #50's review): a single leaf-leaf combine emits exactly one
    // `dst,a,b` instruction, so it is always safe regardless of aliasing --
    // reads and the write happen atomically. The hazard only exists once a
    // COMPOUND operand needs its own multi-instruction evaluation and that
    // evaluation is accumulated directly into `*preferred_destination`: if
    // `*preferred_destination`'s name is read anywhere else in `node` (a
    // plain sibling leaf with that same name, or a reference buried inside
    // the OTHER, not-yet-evaluated operand), that read would silently
    // observe the compound evaluation's clobbered intermediate value instead
    // of the true original -- e.g. a saturating-synapse-style update
    // `g <- g + gbase*(1-g)`, or `v <- v + v*k`, lowered directly into `g`/
    // `v`, would each silently miscompute (concretely: `g <- 2*gbase*(1-g)`
    // instead of `g <- g + gbase*(1-g)`) with no error thrown. Whenever this
    // is possible, the whole node is instead evaluated into an independent
    // fresh temporary -- a `tN` name the lowering itself introduces and so
    // can never collide with a real LEMS identifier, making it unconditionally
    // safe -- and the result is copied into the real destination with one
    // final `mov`. This costs one extra instruction only in the (narrow)
    // self-referential-and-compound case; every other shape (including a
    // self-referential but purely leaf-leaf combine like `asc1 <- asc1 +
    // ascAdd1`) is untouched and keeps the single-instruction form.
    String emit_expression(const ExpressionNode &node, Vector<TickInstruction> &output,
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
        log::throw_runtime_error(log::logger(), "cell_lowering: internal error -- unhandled expression node kind");
    }

private:
    std::unordered_set<String> known_names_;
    UnorderedMap<String, String> network_input_aliases_;
    int temporary_counter_ = 0;

    static BinaryOpcode binary_opcode_for(char operator_character) {
        switch (operator_character) {
            case '+': return BinaryOpcode::Add;
            case '-': return BinaryOpcode::Sub;
            case '*': return BinaryOpcode::Mul;
            case '/': return BinaryOpcode::Div;
            default:
                log::throw_runtime_error(log::logger(), "cell_lowering: internal error -- unhandled binary operator");
        }
    }
};

// ── Integration-method selection (ticket #50: "a static property of the
// RHS"; IR spec §3.3 `expdecay dst,a,tau` = `a*exp(-dt/tau)`) ───────────
//
// `expdecay` has NO built-in "target" term -- it only computes a pure
// multiplicative decay of its own operand toward zero (the expOne synapse
// example's `expdecay g, g, tau` is exactly `dg/dt = -g/tau`). A
// `TimeDerivative` shaped `(target - state)/tau` (target != 0) still decays
// exponentially, but toward `target`, not zero -- its closed form is
// `state' = target + (state-target)*exp(-dt/tau)`, i.e. `(state-target)`
// itself is what decays to zero. So a non-zero target needs a 3-instruction
// sequence (subtract the target out, `expdecay`, add it back); a zero
// target (the after-spike-current case, `-state/tau`) collapses to the
// single-instruction form matching the spec's own example exactly.

struct LinearDecayShape {
    String target;
    String time_constant;
};

// Recognizes exactly two shapes (arch §3.2 TimeDerivative; ticket #50's
// "keep it simple and pattern-based" instruction -- not a general symbolic
// linearity prover):
//   1. `(target - state) / tau`, where `target`/`tau` are each a bare
//      identifier or number literal that isn't `state` itself and doesn't
//      reference `network_inputs` (GLIF1's own `v` dynamics has an extra
//      `network_inputs` term and so correctly does NOT match this shape).
//   2. `-state / tau` (equivalent to target 0 -- after-spike-current decay).
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

void lower_state_assignment_into(const NML_Node &state_assignment_node, Vector<TickInstruction> &output, LoweringContext &context) {
    String assigned_variable_name = get_attribute_value(state_assignment_node, "variable");
    String value_text = get_attribute_value(state_assignment_node, "value");
    context.reset_temporary_counter();
    String context_for_errors = "StateAssignment '" + assigned_variable_name + "' = '" + value_text + "'";
    ExpressionNodePointer value_expression = parse_arithmetic_text(value_text, context_for_errors);
    context.emit_expression(*value_expression, output, assigned_variable_name);
}

// ── Regime body parsing (arch §3.2 Regime/Transition/OnEntry; IR spec §4's
// refractory-regime example) -- `RegimeDecl::body` is the raw `<Regime>`
// node, walked the same way `<Dynamics>` itself is walked by nml.cpp's
// (private, not reused here) extractors. ─────────────────────────────────

struct RegimeTimeDerivative {
    String variable_name;
    String value_text;
};

struct RegimeOnCondition {
    String test_text;
    const NML_Node *body_node = nullptr; // the raw <OnCondition> node
};

struct RegimeInfo {
    String name;
    s32 index = 0;
    Vector<RegimeTimeDerivative> time_derivatives;
    Vector<RegimeOnCondition> on_conditions;
    Vector<const NML_Node *> on_entry_state_assignments; // flattened <StateAssignment> children of every <OnEntry>
};

// The `initial="true"` regime becomes index 0; every other regime keeps its
// declaration order after it (if none is marked initial, declaration order
// is used as-is, so the first-declared regime is index 0).
Vector<RegimeInfo> gather_regime_info(const CellType &cell) {
    Vector<const RegimeDecl *> ordered_regimes;
    for (const auto &regime : cell.regimes) ordered_regimes.push_back(&regime);
    std::stable_partition(ordered_regimes.begin(), ordered_regimes.end(),
                          [](const RegimeDecl *regime) { return regime->initial == "true"; });

    Vector<RegimeInfo> regimes;
    for (usize regime_index = 0; regime_index < ordered_regimes.size(); ++regime_index) {
        const RegimeDecl &regime = *ordered_regimes[regime_index];
        RegimeInfo info;
        info.name = regime.name;
        info.index = static_cast<s32>(regime_index);

        for (const auto *time_derivative_node : find_children(regime.body, "TimeDerivative")) {
            info.time_derivatives.push_back(RegimeTimeDerivative{
                get_attribute_value(*time_derivative_node, "variable"), get_attribute_value(*time_derivative_node, "value")});
        }
        for (const auto *on_condition_node : find_children(regime.body, "OnCondition")) {
            info.on_conditions.push_back(RegimeOnCondition{get_attribute_value(*on_condition_node, "test"), on_condition_node});
        }
        for (const auto *on_entry_node : find_children(regime.body, "OnEntry")) {
            for (const auto *state_assignment_node : find_children(*on_entry_node, "StateAssignment")) {
                info.on_entry_state_assignments.push_back(state_assignment_node);
            }
        }

        regimes.push_back(std::move(info));
    }
    return regimes;
}

const String REGIME_VARIABLE_NAME = "r";

// Actions gathered from one `<OnCondition>` body (arch §3.2: `StateAssignment`
// -> 5 Reset, `EventOut` -> 4 Emit, `Transition` -> the regime-index write,
// bundled into 5 Reset per the locked IR spec's own refractory example).
struct OnConditionActions {
    std::optional<String> emit_port_name;
    Vector<TickInstruction> reset_instructions;
};

OnConditionActions lower_on_condition_actions(
    const NML_Node &on_condition_node, LoweringContext &context,
    const UnorderedMap<String, s32> &regime_index_of,
    const UnorderedMap<String, Vector<const NML_Node *>> &on_entry_assignments_of_regime
) {
    OnConditionActions actions;
    for (const auto &child : on_condition_node.body) {
        if (child.tag_name == "StateAssignment") {
            lower_state_assignment_into(child, actions.reset_instructions, context);
        } else if (child.tag_name == "EventOut") {
            if (actions.emit_port_name.has_value()) {
                log::throw_runtime_error(log::logger(), "cell_lowering: an OnCondition with more than one EventOut is not supported");
            }
            actions.emit_port_name = get_attribute_value(child, "port");
        } else if (child.tag_name == "Transition") {
            String target_regime_name = get_attribute_value(child, "regime");
            auto target_regime_index = regime_index_of.find(target_regime_name);
            if (target_regime_index == regime_index_of.end()) {
                log::throw_runtime_error(log::logger(),
                    "cell_lowering: Transition names undeclared regime '" + target_regime_name + "'");
            }
            actions.reset_instructions.push_back(
                SetRegimeInstruction{REGIME_VARIABLE_NAME, std::to_string(target_regime_index->second)});

            auto on_entry_assignments = on_entry_assignments_of_regime.find(target_regime_name);
            if (on_entry_assignments != on_entry_assignments_of_regime.end()) {
                for (const auto *state_assignment_node : on_entry_assignments->second) {
                    lower_state_assignment_into(*state_assignment_node, actions.reset_instructions, context);
                }
            }
        }
        // Any other child tag is not part of arch §3.2's OnCondition body
        // (StateAssignment/EventOut/Transition are the only three) and is
        // ignored here.
    }
    return actions;
}

// Builds the regime-dispatched `@integrate` block for one state variable
// that has an explicit `TimeDerivative` in at least one regime (IR spec §4's
// refractory-regime example's `eq`/`if`/`else` dispatch pattern, generalized
// to N regimes): the first N-1 regimes (in index order) get an explicit
// `eq is_<name>, r, <index>` guard chained as `if`/`elif`; the last regime is
// the trailing `else` (no `eq` needed for it). A regime with no explicit
// `TimeDerivative` for this variable holds it (`mov var, var`) -- GLIF1-5's
// own convention (a spike-triggered reset/OnEntry already sets whatever a
// held variable should read as 0 -- see e.g. `refractoryTimeElapsed`).
void append_regime_dispatch_for_variable(const String &state_variable_name, const Vector<RegimeInfo> &regimes,
                                          Vector<TickInstruction> &output, LoweringContext &context) {
    auto lower_body_for_regime = [&](const RegimeInfo &regime) -> Vector<TickInstruction> {
        for (const auto &time_derivative : regime.time_derivatives) {
            if (time_derivative.variable_name != state_variable_name) continue;
            Vector<TickInstruction> body;
            lower_time_derivative(state_variable_name, time_derivative.value_text, body, context);
            return body;
        }
        return Vector<TickInstruction>{MoveInstruction{state_variable_name, state_variable_name}};
    };

    if (regimes.size() == 1) {
        Vector<TickInstruction> body = lower_body_for_regime(regimes[0]);
        output.insert(output.end(), body.begin(), body.end());
        return;
    }

    IfInstruction dispatch;
    dispatch.condition = "is_" + regimes[0].name;
    output.push_back(BinaryInstruction{BinaryOpcode::Eq, dispatch.condition, REGIME_VARIABLE_NAME, std::to_string(regimes[0].index)});
    dispatch.then_body = lower_body_for_regime(regimes[0]);

    for (usize regime_index = 1; regime_index + 1 < regimes.size(); ++regime_index) {
        ElseIfBranch branch;
        branch.condition = "is_" + regimes[regime_index].name;
        output.push_back(BinaryInstruction{BinaryOpcode::Eq, branch.condition, REGIME_VARIABLE_NAME, std::to_string(regimes[regime_index].index)});
        branch.body = lower_body_for_regime(regimes[regime_index]);
        dispatch.else_if_branches.push_back(std::move(branch));
    }

    dispatch.else_body = lower_body_for_regime(regimes.back());
    output.push_back(dispatch);
}

// Gathers every `OnStart`'s `StateAssignment`s (arch §3.2: seeds state at
// INIT) into a variable-name -> raw-value-text map. `CellType::on_starts`
// is ordered own-declarations-first-then-ancestors (resolve.cpp's
// `merge_cell_fields`), so first-occurrence-wins matches the same
// most-derived-wins convention resolve.cpp already applies to every other
// `extends`-merged declaration.
UnorderedMap<String, String> gather_initial_values(const CellType &cell) {
    UnorderedMap<String, String> initial_value_of_variable;
    for (const auto &on_start : cell.on_starts) {
        for (const auto *state_assignment_node : find_children(on_start.body, "StateAssignment")) {
            String variable_name = get_attribute_value(*state_assignment_node, "variable");
            if (initial_value_of_variable.count(variable_name)) continue;
            initial_value_of_variable[variable_name] = get_attribute_value(*state_assignment_node, "value");
        }
    }
    return initial_value_of_variable;
}

String format_literal(f64 value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    return String(buffer);
}

} // namespace

IrProgram lower_cell_to_ir(const TypeLibraryEntry &cell_entry) {
    if (cell_entry.category != TypeLibraryCategory::Cell || !std::holds_alternative<CellType>(cell_entry.dynamics.flattened)) {
        log::throw_runtime_error(log::logger(),
            "cell_lowering: '" + cell_entry.component_type_name + "' is not a Cell-category TypeLibraryEntry");
    }
    const CellType &cell = std::get<CellType>(cell_entry.dynamics.flattened);

    // ── known-name / network_inputs-alias setup (arch §3.2 DerivedVariable:
    // a `select`/`reduce="add"` aggregation over attached synapses collapses
    // to one `network_inputs` read) ───────────────────────────────────────
    std::unordered_set<String> known_names;
    for (const auto &state_variable : cell.state_variables) known_names.insert(state_variable.name);
    for (const auto &parameter : cell.parameters) known_names.insert(parameter.name);
    for (const auto &derived_variable : cell.derived_variables) known_names.insert(derived_variable.name);

    UnorderedMap<String, String> network_input_aliases;
    for (const auto &derived_variable : cell.derived_variables) {
        if (derived_variable.select.empty()) continue;
        if (derived_variable.reduce != "add") {
            log::throw_runtime_error(log::logger(),
                "cell_lowering: DerivedVariable '" + derived_variable.name + "' uses unsupported reduce '" +
                derived_variable.reduce + "' (Phase 1 only supports reduce=\"add\" over a cell's attached "
                "synapse current, aliased to network_inputs)");
        }
        network_input_aliases[derived_variable.name] = "network_inputs";
    }

    LoweringContext context(known_names, network_input_aliases);

    // ── regime metadata ───────────────────────────────────────────────────
    Vector<RegimeInfo> regimes = gather_regime_info(cell);
    UnorderedMap<String, s32> regime_index_of;
    UnorderedMap<String, Vector<const NML_Node *>> on_entry_assignments_of_regime;
    for (const auto &regime : regimes) {
        regime_index_of[regime.name] = regime.index;
        on_entry_assignments_of_regime[regime.name] = regime.on_entry_state_assignments;
    }

    std::unordered_set<String> regime_scoped_variables;
    for (const auto &regime : regimes) {
        for (const auto &time_derivative : regime.time_derivatives) regime_scoped_variables.insert(time_derivative.variable_name);
    }

    // ── .alloc (arch §4.1 state; §3.1 Parameter bake-vs-parameterize,
    // already decided by ticket #7 -- see cell_lowering.h; §4.5 regime;
    // §4.6 expose) ─────────────────────────────────────────────────────────
    UnorderedMap<String, String> initial_value_of_variable = gather_initial_values(cell);
    Vector<AllocDirective> alloc_directives;

    Vector<String> exposed_names_in_order;
    std::unordered_set<String> exposed_names_seen;
    auto add_exposure = [&](const String &exposure_name) {
        if (exposure_name.empty() || exposed_names_seen.count(exposure_name)) return;
        exposed_names_seen.insert(exposure_name);
        exposed_names_in_order.push_back(exposure_name);
    };

    for (const auto &state_variable : cell.state_variables) {
        std::optional<String> initial_value;
        auto found_initial_value = initial_value_of_variable.find(state_variable.name);
        if (found_initial_value != initial_value_of_variable.end()) initial_value = found_initial_value->second;
        alloc_directives.push_back(StateDirective{state_variable.name, "f32", initial_value});
        add_exposure(state_variable.exposure);
    }
    if (!regimes.empty()) alloc_directives.push_back(RegimeDirective{REGIME_VARIABLE_NAME});
    for (const auto &parameter : cell.parameters) {
        auto baked_value = cell_entry.baked_constants.find(parameter.name);
        std::optional<String> literal_value;
        if (baked_value != cell_entry.baked_constants.end()) literal_value = format_literal(baked_value->second);
        alloc_directives.push_back(ParamConstantDirective{parameter.name, literal_value});
    }
    for (const auto &derived_variable : cell.derived_variables) add_exposure(derived_variable.exposure);
    for (const auto &exposure : cell.exposures) add_exposure(exposure.name);
    for (const auto &exposure_name : exposed_names_in_order) alloc_directives.push_back(ExposeDirective{exposure_name});

    // ── .tick @integrate: plain-value DerivedVariables, then unconditional
    // (non-regime-scoped) TimeDerivatives, then the regime-dispatched ones
    // -- see this file's header note on why derived variables are computed
    // first (they read pre-tick state, before any TimeDerivative advances
    // it) ──────────────────────────────────────────────────────────────────
    TickProgram tick;
    for (const auto &derived_variable : cell.derived_variables) {
        if (!derived_variable.select.empty()) continue; // network_inputs alias -- no instructions needed
        if (derived_variable.value.empty()) {
            log::throw_runtime_error(log::logger(),
                "cell_lowering: DerivedVariable '" + derived_variable.name + "' has neither `value` nor `select`/`reduce`");
        }
        context.reset_temporary_counter();
        ExpressionNodePointer value_expression = parse_arithmetic_text(
            derived_variable.value, "DerivedVariable '" + derived_variable.name + "'");
        context.emit_expression(*value_expression, tick.integrate, derived_variable.name);
    }
    for (const auto &time_derivative : cell.time_derivatives) {
        if (regime_scoped_variables.count(time_derivative.variable)) continue; // overridden per-regime, handled below
        lower_time_derivative(time_derivative.variable, time_derivative.value, tick.integrate, context);
    }
    for (const auto &state_variable : cell.state_variables) {
        if (!regime_scoped_variables.count(state_variable.name)) continue;
        append_regime_dispatch_for_variable(state_variable.name, regimes, tick.integrate, context);
    }

    // ── .tick @detect / @emit / @reset: ungated (top-level) OnConditions
    // first, then every regime's own OnConditions gated by a fresh
    // regime-equality check ANDed with the raw test (IR spec §4's
    // refractory-regime example's `eq`/`gt`/`and` pattern) -- honors
    // detect -> emit -> reset ordering (arch §2/§3.2: the test reads
    // pre-reset state) since the raw test is always evaluated before any of
    // its OnCondition's own actions run. ───────────────────────────────────
    for (usize condition_index = 0; condition_index < cell.on_conditions.size(); ++condition_index) {
        const OnConditionDecl &on_condition = cell.on_conditions[condition_index];
        context.reset_temporary_counter();
        ParsedCondition parsed_condition = parse_condition_text(on_condition.test);
        String left_operand = context.emit_expression(*parsed_condition.left, tick.detect, std::nullopt);
        String right_operand = context.emit_expression(*parsed_condition.right, tick.detect, std::nullopt);
        String condition_name = cell.on_conditions.size() == 1 ? String("spiked") : ("spiked" + std::to_string(condition_index));
        tick.detect.push_back(BinaryInstruction{parsed_condition.opcode, condition_name, left_operand, right_operand});

        OnConditionActions actions = lower_on_condition_actions(
            on_condition.body, context, regime_index_of, on_entry_assignments_of_regime);
        if (actions.emit_port_name.has_value()) {
            tick.emit.push_back(IfInstruction{condition_name, {EmitInstruction{*actions.emit_port_name}}, {}, std::nullopt});
        }
        if (!actions.reset_instructions.empty()) {
            tick.reset.push_back(IfInstruction{condition_name, actions.reset_instructions, {}, std::nullopt});
        }
    }

    for (const auto &regime : regimes) {
        for (usize condition_index = 0; condition_index < regime.on_conditions.size(); ++condition_index) {
            const RegimeOnCondition &on_condition = regime.on_conditions[condition_index];
            context.reset_temporary_counter();

            String is_regime_name = "is_" + regime.name;
            tick.detect.push_back(BinaryInstruction{BinaryOpcode::Eq, is_regime_name, REGIME_VARIABLE_NAME, std::to_string(regime.index)});

            ParsedCondition parsed_condition = parse_condition_text(on_condition.test_text);
            String left_operand = context.emit_expression(*parsed_condition.left, tick.detect, std::nullopt);
            String right_operand = context.emit_expression(*parsed_condition.right, tick.detect, std::nullopt);
            String name_suffix = regime.on_conditions.size() == 1 ? regime.name : (regime.name + std::to_string(condition_index));
            String raw_test_name = "test_" + name_suffix;
            tick.detect.push_back(BinaryInstruction{parsed_condition.opcode, raw_test_name, left_operand, right_operand});

            String fire_name = "fire_" + name_suffix;
            tick.detect.push_back(BinaryInstruction{BinaryOpcode::And, fire_name, is_regime_name, raw_test_name});

            OnConditionActions actions = lower_on_condition_actions(
                *on_condition.body_node, context, regime_index_of, on_entry_assignments_of_regime);
            if (actions.emit_port_name.has_value()) {
                tick.emit.push_back(IfInstruction{fire_name, {EmitInstruction{*actions.emit_port_name}}, {}, std::nullopt});
            }
            if (!actions.reset_instructions.empty()) {
                tick.reset.push_back(IfInstruction{fire_name, actions.reset_instructions, {}, std::nullopt});
            }
        }
    }

    IrProgram program;
    program.component_type_name = cell_entry.component_type_name;
    program.alloc = std::move(alloc_directives);
    program.tick = std::move(tick);
    return program;
}

Vector<IrProgram> lower_all_cell_types_to_ir(const ModelSpecification &model) {
    Vector<IrProgram> programs;
    for (const auto &entry : model.type_library) {
        if (entry.category != TypeLibraryCategory::Cell) continue;
        programs.push_back(lower_cell_to_ir(entry));
    }
    return programs;
}

} // namespace spikecorec::nml
