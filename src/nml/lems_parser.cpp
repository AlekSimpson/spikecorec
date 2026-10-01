#include "spikecorec/nml/dynamics.h"

#include <stdexcept>

#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"
#include "spikecorec/nml/node.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

namespace {

// LEMS function name -> the name emitted into GPU source.
const UnorderedMap<String, String> FUNCTIONS = {
    {"exp", "exp"},   {"ln", "log"},    {"log", "log10"}, {"sqrt", "sqrt"},
    {"abs", "fabs"},  {"ceil", "ceil"}, {"floor", "floor"},
    {"sin", "sin"},   {"cos", "cos"},   {"tan", "tan"},
    {"sinh", "sinh"}, {"cosh", "cosh"}, {"tanh", "tanh"},
    {"H", "spikecorec_heaviside"},
};

LemsParseBody lems_body(LemsNodeSubtype syntax_type, const LemsExpressionToken &token) {
    return LemsParseBody(ParseNodeType::LEMS_NODE, syntax_type, token);
}

} // namespace

s32 LemsExpressionParser::binary_precedence(const LemsExpressionToken &token) {
    if (token.lexeme == "||") return 1;
    if (token.lexeme == "&&") return 2;
    if (token.lexeme == "==" || token.lexeme == "!=") return 3;
    if (token.lexeme == "<" || token.lexeme == ">" || token.lexeme == "<=" || token.lexeme == ">=") return 4;
    if (token.lexeme == "+" || token.lexeme == "-") return 5;
    if (token.lexeme == "*" || token.lexeme == "/") return 6;
    if (token.lexeme == "^") return 7;
    return -1;
}

const LemsExpressionToken &LemsExpressionParser::current() const {
    return tokens[token_index];
}

[[noreturn]] void LemsExpressionParser::fail(const String &reason) const {
    throw runtime_error("lems_parser: " + reason + " in '" + expression + "' (" +
                        owner_name + ")");
}

LemsParseNode *LemsExpressionParser::resolve_identifier(const LemsExpressionToken &name) const {
    auto bound = symbols.find(name.lexeme);
    if (bound == symbols.end()) {
        throw runtime_error(
                "lems_parser: '" + name.lexeme + "' in '" + expression + "' (" + owner_name +
                ") resolves to no parameter, state variable, derived variable, constant "
                "or engine quantity");
    }
    return new LemsParseNode(lems_body(LemsNodeSubtype::IDENTIFIER, name));
}

LemsParseNode *LemsExpressionParser::parse_primary() {
    const LemsExpressionToken token = current();

    if (token.kind == LemsExpressionToken::Kind::Number) {
        token_index += 1;
        auto *node = new LemsParseNode(lems_body(LemsNodeSubtype::FLOAT, token));
        return node;
    }

    if (token.kind == LemsExpressionToken::Kind::OpenParen) {
        token_index += 1;

        auto *inner_expression = parse_binary(0);
        if (current().kind != LemsExpressionToken::Kind::CloseParen) fail("missing ')'");
        token_index += 1;

        return inner_expression;
    }

    if (token.kind == LemsExpressionToken::Kind::Operator && (token.lexeme == "-" || token.lexeme == "+")) {
        token_index += 1;
        auto *unary_operator =
                new UnaryNode<LemsParseBody>(lems_body(LemsNodeSubtype::OPERATOR, token));
        auto *operand = parse_unary();
        unary_operator->child = operand;
        return unary_operator;
    }

    if (token.kind == LemsExpressionToken::Kind::Identifier) {
        LemsExpressionToken function_token = token;
        token_index += 1;

        if (current().kind != LemsExpressionToken::Kind::OpenParen) return resolve_identifier(function_token);

        auto function = FUNCTIONS.find(function_token.lexeme);
        if (function == FUNCTIONS.end()) {
            throw runtime_error(
                    "lems_parser: unknown function '" + function_token.lexeme + "' in '" + expression +
                    "' (" + owner_name + ")");
        }
        token_index += 1; // move past OpenParen 

        Vector<LemsParseNode *> argument_nodes;
        if (current().kind != LemsExpressionToken::Kind::CloseParen) {
            while (true) {
                argument_nodes.push_back(parse_binary(0));
                if (current().kind != LemsExpressionToken::Kind::Comma) break;
                token_index += 1;
            }
        }
        if (current().kind != LemsExpressionToken::Kind::CloseParen) fail("missing ')' after " + function_token.lexeme);
        token_index += 1; // move past CloseParen
        
        auto *function_identifier_node =
                new LemsParseNode(lems_body(LemsNodeSubtype::IDENTIFIER, function_token));
        auto *arguments_node =
                new ListNode<LemsParseBody>(lems_body(LemsNodeSubtype::LIST, LemsExpressionToken()));
        arguments_node->children = std::move(argument_nodes);

        auto *function_node = new BinaryNode<LemsParseBody>(
                lems_body(LemsNodeSubtype::FUNCTION_CALL, function_token));
        function_node->left = function_identifier_node;
        function_node->right = arguments_node;
        return function_node;
    }

    fail("expected a value");
}

LemsParseNode *LemsExpressionParser::parse_unary() { return parse_primary(); }

LemsParseNode *LemsExpressionParser::parse_binary(s32 minimum_precedence) {
    auto *left = parse_unary();

    while (true) {
        const LemsExpressionToken operator_token = current();
        if (operator_token.kind != LemsExpressionToken::Kind::Operator) break;

        const s32 precedence = binary_precedence(operator_token);
        if (precedence < minimum_precedence) break;

        token_index += 1;

        // `^` is the only right-associative operator, so its right operand is parsed at
        // the same precedence rather than one above: a^b^c is a^(b^c).
        auto *right =
                parse_binary(operator_token.lexeme == "^" ? precedence : precedence + 1);

        if (operator_token.lexeme == "^") {
            LemsExpressionToken power_token{LemsExpressionToken::Kind::Identifier, "pow"};

            auto *power_identifier_node =
                    new LemsParseNode(lems_body(LemsNodeSubtype::IDENTIFIER, power_token));

            auto *arguments_node =
                    new ListNode<LemsParseBody>(lems_body(LemsNodeSubtype::LIST, LemsExpressionToken()));
            arguments_node->children.push_back(left);
            arguments_node->children.push_back(right);

            auto *power_node = new BinaryNode<LemsParseBody>(
                    lems_body(LemsNodeSubtype::FUNCTION_CALL, power_token));
            power_node->left = power_identifier_node;
            power_node->right = arguments_node;

            left = power_node;
        } else {
            auto *binary_node = new BinaryNode<LemsParseBody>(
                    lems_body(LemsNodeSubtype::OPERATOR, operator_token));
            binary_node->left = left;
            binary_node->right = right;

            left = binary_node;
        }
    }

    return left;
}



}
