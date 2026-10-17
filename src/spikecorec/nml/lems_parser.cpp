#include "dynamics.h"

#include <stdexcept>

#include "units.h"
#include "types.h"
#include "node.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

namespace {

// LEMS functions an expression may call. Only the keys are read; codegen spells them itself.
const Unordered_Map<String, String> FUNCTIONS = {
    {"exp", "exp"},   {"ln", "log"},    {"log", "log"},   {"sqrt", "sqrt"},
    {"abs", "fabs"},  {"ceil", "ceil"}, {"floor", "floor"},
    {"sin", "sin"},   {"cos", "cos"},   {"tan", "tan"},
    {"sinh", "sinh"}, {"cosh", "cosh"}, {"tanh", "tanh"},
    {"H", "spikecorec_heaviside"},    {"random", "random_values"},
};

Lems_Parse_Body lems_body(Lems_Node_Subtype syntax_type, const Lems_Expression_Token &token) {
    return Lems_Parse_Body(Parse_Node_Type::LEMS_NODE, syntax_type, token);
}

} // namespace

s32 Lems_Expression_Parser::binary_precedence(const Lems_Expression_Token &token) {
    if (token.lexeme == "||") return 1;
    if (token.lexeme == "&&") return 2;
    if (token.lexeme == "==" || token.lexeme == "!=") return 3;
    if (token.lexeme == "<" || token.lexeme == ">" || token.lexeme == "<=" || token.lexeme == ">=") return 4;
    if (token.lexeme == "+" || token.lexeme == "-") return 5;
    if (token.lexeme == "*" || token.lexeme == "/") return 6;
    if (token.lexeme == "^") return 7;
    return -1;
}

const Lems_Expression_Token &Lems_Expression_Parser::current() const {
    return tokens[token_index];
}

[[noreturn]] void Lems_Expression_Parser::fail(const String &reason) const {
    throw Runtime_Error("lems_parser: " + reason + " in '" + expression + "' (" +
                        owner_name + ")");
}

// Names stay LEMS names here; codegen binds them by substitution in the kernel tree.
Lems_Parse_Node *Lems_Expression_Parser::resolve_identifier(const Lems_Expression_Token &name) const {
    return new Lems_Parse_Node(lems_body(Lems_Node_Subtype::IDENTIFIER, name));
}

Lems_Parse_Node *Lems_Expression_Parser::parse_primary() {
    const Lems_Expression_Token token = current();

    if (token.kind == Lems_Expression_Token::Kind::Number) {
        token_index += 1;
        auto *node = new Lems_Parse_Node(lems_body(Lems_Node_Subtype::FLOAT, token));
        return node;
    }

    if (token.kind == Lems_Expression_Token::Kind::OpenParen) {
        token_index += 1;

        auto *inner_expression = parse_binary(0);
        if (current().kind != Lems_Expression_Token::Kind::CloseParen) fail("missing ')'");
        token_index += 1;

        return inner_expression;
    }

    // A sign applies to a whole power, as in jLEMS: -x^2 is -(x^2).
    if (token.kind == Lems_Expression_Token::Kind::Operator && (token.lexeme == "-" || token.lexeme == "+")) {
        token_index += 1;
        auto *unary_operator =
                new Unary_Node<Lems_Parse_Body>(lems_body(Lems_Node_Subtype::OPERATOR, token));
        auto *operand = parse_binary(binary_precedence(Lems_Expression_Token{Lems_Expression_Token::Kind::Operator, "^"}));
        unary_operator->child = operand;
        return unary_operator;
    }

    if (token.kind == Lems_Expression_Token::Kind::Identifier) {
        Lems_Expression_Token function_token = token;
        token_index += 1;

        if (current().kind != Lems_Expression_Token::Kind::OpenParen) return resolve_identifier(function_token);

        auto function = FUNCTIONS.find(function_token.lexeme);
        if (function == FUNCTIONS.end()) {
            throw Runtime_Error(
                    "lems_parser: unknown function '" + function_token.lexeme + "' in '" + expression +
                    "' (" + owner_name + ")");
        }
        token_index += 1; // move past OpenParen 

        Vector<Lems_Parse_Node *> argument_nodes;
        if (current().kind != Lems_Expression_Token::Kind::CloseParen) {
            while (true) {
                argument_nodes.push_back(parse_binary(0));
                if (current().kind != Lems_Expression_Token::Kind::Comma) break;
                token_index += 1;
            }
        }
        if (current().kind != Lems_Expression_Token::Kind::CloseParen) fail("missing ')' after " + function_token.lexeme);
        token_index += 1; // move past CloseParen
        
        auto *function_identifier_node =
                new Lems_Parse_Node(lems_body(Lems_Node_Subtype::IDENTIFIER, function_token));
        auto *arguments_node =
                new List_Node<Lems_Parse_Body>(lems_body(Lems_Node_Subtype::LIST, Lems_Expression_Token()));
        arguments_node->children = std::move(argument_nodes);

        auto *function_node = new Binary_Node<Lems_Parse_Body>(
                lems_body(Lems_Node_Subtype::FUNCTION_CALL, function_token));
        function_node->left = function_identifier_node;
        function_node->right = arguments_node;
        return function_node;
    }

    fail("expected a value");
}

Lems_Parse_Node *Lems_Expression_Parser::parse_unary() { return parse_primary(); }

Lems_Parse_Node *Lems_Expression_Parser::parse_binary(s32 minimum_precedence) {
    auto *left = parse_unary();

    while (true) {
        const Lems_Expression_Token operator_token = current();
        if (operator_token.kind != Lems_Expression_Token::Kind::Operator) break;

        const s32 precedence = binary_precedence(operator_token);
        if (precedence < minimum_precedence) break;

        token_index += 1;

        // Every operator is left-associative, as in jLEMS: a^b^c is (a^b)^c.
        auto *right = parse_binary(precedence + 1);

        if (operator_token.lexeme == "^") {
            Lems_Expression_Token power_token{Lems_Expression_Token::Kind::Identifier, "pow"};

            auto *power_identifier_node =
                    new Lems_Parse_Node(lems_body(Lems_Node_Subtype::IDENTIFIER, power_token));

            auto *arguments_node =
                    new List_Node<Lems_Parse_Body>(lems_body(Lems_Node_Subtype::LIST, Lems_Expression_Token()));
            arguments_node->children.push_back(left);
            arguments_node->children.push_back(right);

            auto *power_node = new Binary_Node<Lems_Parse_Body>(
                    lems_body(Lems_Node_Subtype::FUNCTION_CALL, power_token));
            power_node->left = power_identifier_node;
            power_node->right = arguments_node;

            left = power_node;
        } else {
            auto *binary_node = new Binary_Node<Lems_Parse_Body>(
                    lems_body(Lems_Node_Subtype::OPERATOR, operator_token));
            binary_node->left = left;
            binary_node->right = right;

            left = binary_node;
        }
    }

    return left;
}

Lems_Parse_Node *parse_lems_expression(const String &expression, const String &owner_name) {
    Lems_Expression_Parser parser;
    parser.expression = expression;
    parser.owner_name = owner_name;
    parser.tokens = tokenize_lems(expression, owner_name);

    Lems_Parse_Node *root = parser.parse_binary(0);
    if (parser.current().kind != Lems_Expression_Token::Kind::End) {
        delete root;
        parser.fail("unexpected '" + parser.current().lexeme + "'");
    }
    return root;
}

}
