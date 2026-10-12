#include "dynamics.h"

#include <stdexcept>

#include "types.h"
#include "node.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

namespace {

using Kind = Kernel_Token::Kind;

const Set<String> ASSIGNMENT_OPERATORS = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="};
const Set<String> PREFIX_OPERATORS = {"-", "+", "!", "~", "++", "--", "*", "&"};

// Implied by the node type, so emission puts them back.
const Set<String> KERNEL_QUALIFIERS = {"kernel", "__global__"};
const Set<String> FUNCTION_QUALIFIERS = {"inline", "static", "extern", "__device__", "__forceinline__"};

s32 binary_precedence(const Kernel_Token &token) {
    static const Unordered_Map<String, s32> precedences = {
        {"||", 1}, {"&&", 2}, {"|", 3}, {"^", 4}, {"&", 5}, {"==", 6}, {"!=", 6},
        {"<", 7}, {"<=", 7}, {">", 7}, {">=", 7}, {"<<", 8}, {">>", 8},
        {"+", 9}, {"-", 9}, {"*", 10}, {"/", 10}, {"%", 10}};
    if (token.kind != Kind::Operator) return -1;
    auto found = precedences.find(token.lexeme);
    return found == precedences.end() ? -1 : found->second;
}

String trim(const String &text) {
    const usize first = text.find_first_not_of(" \t\r");
    if (first == String::npos) return "";
    const usize last = text.find_last_not_of(" \t\r");
    return text.substr(first, last - first + 1);
}

} // namespace

const Kernel_Token &Kernel_Parser::current() const {
    return tokens[token_index];
}

const Kernel_Token &Kernel_Parser::peek(usize distance) const {
    return tokens[std::min(token_index + distance, tokens.size() - 1)];
}

bool Kernel_Parser::check(Kind kind, const String &lexeme) const {
    return current().kind == kind && (lexeme.empty() || current().lexeme == lexeme);
}

bool Kernel_Parser::accept(Kind kind, const String &lexeme) {
    if (!check(kind, lexeme)) return false;
    token_index += 1;
    return true;
}

Kernel_Token Kernel_Parser::expect(Kind kind, const String &lexeme) {
    if (!check(kind, lexeme)) fail(lexeme.empty() ? "unexpected token" : "expected '" + lexeme + "'");
    const Kernel_Token token = current();
    token_index += 1;
    return token;
}

bool Kernel_Parser::starts_type() const {
    return check(Kind::Type) || check(Kind::Qualifier);
}

[[noreturn]] void Kernel_Parser::fail(const String &reason) const {
    String context;
    const usize first = token_index >= 8 ? token_index - 8 : 0;
    for (usize index = first; index <= token_index && index < tokens.size(); index += 1) {
        if (!context.empty()) context += " ";
        context += tokens[index].lexeme;
    }
    throw Runtime_Error("kernel_parser: " + reason + " at '" + current().lexeme + "', after: " + context);
}

Kernel_List_Node *Kernel_Parser::parse_program() {
    Kernel_List_Node *program = new_node<List_Node>(Kernel_Node_Type::PROGRAM, "");
    try {
        while (!check(Kind::End)) {
            if (check(Kind::Preprocessor)) program->children.push_back(parse_preprocessor());
            else if (check(Kind::Identifier, "using")) program->children.push_back(parse_using_namespace());
            else program->children.push_back(parse_function());
        }
    } catch (...) {
        delete program;
        throw;
    }
    return program;
}

Kernel_Node *Kernel_Parser::parse_preprocessor() {
    const String line = expect(Kind::Preprocessor).lexeme;
    const String directive_and_rest = trim(line.substr(1));
    const usize directive_end = std::min(directive_and_rest.find_first_of(" \t"), directive_and_rest.size());
    const String directive = directive_and_rest.substr(0, directive_end);
    const String rest = trim(directive_and_rest.substr(directive_end));

    if (directive == "include") return new_node(Kernel_Node_Type::INCLUDE, rest);

    if (directive == "define") {
        const usize name_end = std::min(rest.find_first_of(" \t"), rest.size());
        Kernel_Binary_Node *define = new_node<Binary_Node>(Kernel_Node_Type::DEFINE, "");
        define->left = new_node(Kernel_Node_Type::IDENTIFIER, rest.substr(0, name_end));
        const String replacement = trim(rest.substr(name_end));
        if (!replacement.empty()) define->right = new_node(Kernel_Node_Type::LITERAL, replacement);
        return define;
    }

    token_index -= 1;
    fail("unsupported preprocessor directive '#" + directive + "'");
}

Kernel_Node *Kernel_Parser::parse_using_namespace() {
    expect(Kind::Identifier, "using");
    expect(Kind::Identifier, "namespace");
    const String namespace_name = expect(Kind::Identifier).lexeme;
    expect(Kind::Punctuation, ";");
    return new_node(Kernel_Node_Type::USING_NAMESPACE, namespace_name);
}

Kernel_Node *Kernel_Parser::parse_function() {
    bool is_kernel = false;
    while (true) {
        if (check(Kind::Qualifier) && KERNEL_QUALIFIERS.count(current().lexeme)) {
            is_kernel = true;
            token_index += 1;
        } else if (check(Kind::Qualifier) && FUNCTION_QUALIFIERS.count(current().lexeme)) {
            token_index += 1;
        } else if (check(Kind::StringLiteral, "\"C\"")) {
            token_index += 1;
        } else {
            break;
        }
    }
    if (!starts_type()) fail("expected a function definition");

    Kernel_List_Node *function = new_node<List_Node>(
            is_kernel ? Kernel_Node_Type::KERNEL_FUNCTION_IMPL : Kernel_Node_Type::DEVICE_FUNCTION_IMPL, "");
    try {
        function->children.push_back(parse_type());
        function->children.push_back(new_node(Kernel_Node_Type::IDENTIFIER, expect(Kind::Identifier).lexeme));
        function->children.push_back(parse_parameter_list());
        function->children.push_back(parse_block());
    } catch (...) {
        delete function;
        throw;
    }
    return function;
}

Kernel_List_Node *Kernel_Parser::parse_parameter_list() {
    Kernel_List_Node *parameters = new_node<List_Node>(Kernel_Node_Type::PARAMETER_LIST, "");
    try {
        expect(Kind::Punctuation, "(");
        if (!accept(Kind::Punctuation, ")")) {
            do {
                parameters->children.push_back(parse_parameter());
            } while (accept(Kind::Punctuation, ","));
            expect(Kind::Punctuation, ")");
        }
    } catch (...) {
        delete parameters;
        throw;
    }
    return parameters;
}

// [[ buffer(n) ]] is dropped: emission numbers buffers by position.
Kernel_Node *Kernel_Parser::parse_parameter() {
    Kernel_Trinary_Node *parameter = new_node<Trinary_Node>(Kernel_Node_Type::PARAMETER, "");
    try {
        parameter->left = parse_type();
        parameter->middle = new_node(Kernel_Node_Type::IDENTIFIER, expect(Kind::Identifier).lexeme);
        if (check(Kind::Attribute)) {
            const String attribute = expect(Kind::Attribute).lexeme;
            if (attribute.rfind("buffer(", 0) != 0) parameter->right = new_node(Kernel_Node_Type::ATTRIBUTE, attribute);
        }
    } catch (...) {
        delete parameter;
        throw;
    }
    return parameter;
}

// Qualifiers and base type words, then any "*" and "&". The base type is normalized to its
// neutral name; a type with no neutral name keeps its spelling.
Kernel_Node *Kernel_Parser::parse_type() {
    String qualifiers;
    String base_type;
    while (starts_type()) {
        String &words = check(Kind::Type) ? base_type : qualifiers;
        if (!words.empty()) words += " ";
        words += current().lexeme;
        token_index += 1;
    }
    if (base_type.empty()) fail("expected a type");

    for (const auto &[neutral, spelling] : kernel_type_spellings(backend)) {
        if (spelling == base_type) {
            base_type = neutral;
            break;
        }
    }

    Kernel_Node *type = new_node(Kernel_Node_Type::TYPE, qualifiers.empty() ? base_type : qualifiers + " " + base_type);
    while (true) {
        if (accept(Kind::Operator, "*")) {
            Kernel_Unary_Node *pointer = new_node<Unary_Node>(Kernel_Node_Type::POINTER, "");
            pointer->child = type;
            while (check(Kind::Qualifier)) {
                if (!pointer->body.token.empty()) pointer->body.token += " ";
                pointer->body.token += current().lexeme;
                token_index += 1;
            }
            type = pointer;
        } else if (accept(Kind::Operator, "&")) {
            Kernel_Unary_Node *reference = new_node<Unary_Node>(Kernel_Node_Type::REFERENCE, "");
            reference->child = type;
            type = reference;
        } else {
            break;
        }
    }
    return type;
}

Kernel_List_Node *Kernel_Parser::parse_block() {
    if (!check(Kind::Punctuation, "{")) fail("expected '{': bodies of if, else, for and while must be braced");
    token_index += 1;

    Kernel_List_Node *block = new_block();
    try {
        while (!accept(Kind::Punctuation, "}")) {
            if (check(Kind::End)) fail("unterminated block");
            block->children.push_back(parse_statement());
        }
    } catch (...) {
        delete block;
        throw;
    }
    return block;
}

Kernel_Node *Kernel_Parser::parse_statement() {
    if (check(Kind::Punctuation, "{")) return parse_block();
    if (check(Kind::Keyword, "if")) return parse_conditional();
    if (check(Kind::Keyword, "for")) return parse_for();
    if (check(Kind::Keyword, "while")) return parse_while();
    if (check(Kind::Keyword, "return") || check(Kind::Keyword, "break") || check(Kind::Keyword, "continue")) {
        return parse_jump();
    }
    if (check(Kind::Keyword) || check(Kind::Preprocessor)) fail("unsupported statement");

    Kernel_Node *statement = parse_simple_statement();
    if (!accept(Kind::Punctuation, ";")) {
        delete statement;
        fail("expected ';'");
    }
    return statement;
}

// A declaration, assignment or expression, without its ';' so for headers can use it.
// A statement-level x++ / x-- becomes x += 1 / x -= 1.
Kernel_Node *Kernel_Parser::parse_simple_statement() {
    if (starts_type()) return parse_declaration();

    Kernel_Node *target = parse_expression();
    if (check(Kind::Operator) && ASSIGNMENT_OPERATORS.count(current().lexeme)) {
        const String assignment_operator = expect(Kind::Operator).lexeme;
        Kernel_Node *value = nullptr;
        try {
            value = parse_expression();
        } catch (...) {
            delete target;
            throw;
        }
        return new_assignment(target, assignment_operator, value);
    }
    if (check(Kind::Operator, "++") || check(Kind::Operator, "--")) {
        const String step_operator = expect(Kind::Operator).lexeme == "++" ? "+=" : "-=";
        return new_assignment(target, step_operator, new_node(Kernel_Node_Type::LITERAL, "1"));
    }
    return target;
}

Kernel_Node *Kernel_Parser::parse_declaration() {
    Kernel_Trinary_Node *declaration = new_node<Trinary_Node>(Kernel_Node_Type::DECLARATION, "");
    try {
        declaration->left = parse_type();
        declaration->middle = new_node(Kernel_Node_Type::IDENTIFIER, expect(Kind::Identifier).lexeme);
        if (accept(Kind::Punctuation, "[")) {
            Kernel_Node *size = parse_expression();
            declaration->middle = new_expression("[]", declaration->middle, size);
            expect(Kind::Punctuation, "]");
        }
        if (accept(Kind::Operator, "=")) declaration->right = parse_expression();
        if (check(Kind::Punctuation, ",")) fail("declare one variable per statement");
    } catch (...) {
        delete declaration;
        throw;
    }
    return declaration;
}

Kernel_List_Node *Kernel_Parser::parse_conditional() {
    Kernel_List_Node *conditional = new_conditional();
    try {
        expect(Kind::Keyword, "if");
        do {
            expect(Kind::Punctuation, "(");
            Kernel_Node *test = parse_expression();
            Kernel_Node *body = nullptr;
            try {
                expect(Kind::Punctuation, ")");
                body = parse_block();
            } catch (...) {
                delete test;
                throw;
            }
            add_branch(conditional, test, body);

            if (!accept(Kind::Keyword, "else")) break;
            if (!accept(Kind::Keyword, "if")) {
                add_branch(conditional, nullptr, parse_block());
                break;
            }
        } while (true);
    } catch (...) {
        delete conditional;
        throw;
    }
    return conditional;
}

Kernel_Node *Kernel_Parser::parse_for() {
    Kernel_List_Node *loop = new_node<List_Node>(Kernel_Node_Type::FOR, "");
    loop->children = {nullptr, nullptr, nullptr, nullptr};
    try {
        expect(Kind::Keyword, "for");
        expect(Kind::Punctuation, "(");
        if (!check(Kind::Punctuation, ";")) loop->children[0] = parse_simple_statement();
        expect(Kind::Punctuation, ";");
        if (!check(Kind::Punctuation, ";")) loop->children[1] = parse_expression();
        expect(Kind::Punctuation, ";");
        if (!check(Kind::Punctuation, ")")) loop->children[2] = parse_simple_statement();
        expect(Kind::Punctuation, ")");
        loop->children[3] = parse_block();
    } catch (...) {
        delete loop;
        throw;
    }
    return loop;
}

Kernel_Node *Kernel_Parser::parse_while() {
    Kernel_Binary_Node *loop = new_node<Binary_Node>(Kernel_Node_Type::WHILE, "");
    try {
        expect(Kind::Keyword, "while");
        expect(Kind::Punctuation, "(");
        loop->left = parse_expression();
        expect(Kind::Punctuation, ")");
        loop->right = parse_block();
    } catch (...) {
        delete loop;
        throw;
    }
    return loop;
}

Kernel_Node *Kernel_Parser::parse_jump() {
    Kernel_Unary_Node *jump = new_node<Unary_Node>(Kernel_Node_Type::JUMP, expect(Kind::Keyword).lexeme);
    try {
        if (jump->body.token == "return" && !check(Kind::Punctuation, ";")) jump->child = parse_expression();
        expect(Kind::Punctuation, ";");
    } catch (...) {
        delete jump;
        throw;
    }
    return jump;
}

Kernel_Node *Kernel_Parser::parse_expression() {
    Kernel_Node *test = parse_binary(1);
    if (!accept(Kind::Operator, "?")) return test;

    Kernel_Trinary_Node *ternary = new_node<Trinary_Node>(Kernel_Node_Type::EXPRESSION, "?:");
    ternary->left = test;
    try {
        ternary->middle = parse_expression();
        expect(Kind::Operator, ":");
        ternary->right = parse_expression();
    } catch (...) {
        delete ternary;
        throw;
    }
    return ternary;
}

Kernel_Node *Kernel_Parser::parse_binary(s32 minimum_precedence) {
    Kernel_Node *left = parse_unary();
    while (true) {
        const s32 precedence = binary_precedence(current());
        if (precedence < minimum_precedence) break;

        const String expression_operator = expect(Kind::Operator).lexeme;
        Kernel_Node *right = nullptr;
        try {
            right = parse_binary(precedence + 1);
        } catch (...) {
            delete left;
            throw;
        }
        left = new_expression(expression_operator, left, right);
    }
    return left;
}

// A "(" followed by a type and ")" is a cast; "(float4(...))" is not.
Kernel_Node *Kernel_Parser::parse_unary() {
    if (check(Kind::Operator) && PREFIX_OPERATORS.count(current().lexeme)) {
        const String expression_operator = expect(Kind::Operator).lexeme;
        return new_unary_expression(expression_operator, parse_unary());
    }

    if (check(Kind::Punctuation, "(") && (peek(1).kind == Kind::Type || peek(1).kind == Kind::Qualifier)) {
        const usize open_index = token_index;
        token_index += 1;
        Kernel_Node *type = parse_type();
        if (!accept(Kind::Punctuation, ")")) {
            delete type;
            token_index = open_index;
            return parse_postfix();
        }

        Kernel_Node *value = nullptr;
        try {
            value = parse_unary();
        } catch (...) {
            delete type;
            throw;
        }
        return new_cast(type, value);
    }

    return parse_postfix();
}

Kernel_Node *Kernel_Parser::parse_postfix() {
    Kernel_Node *node = parse_primary();
    try {
        while (true) {
            if (accept(Kind::Punctuation, "[")) {
                Kernel_Node *index = parse_expression();
                node = new_expression("[]", node, index);
                expect(Kind::Punctuation, "]");
            } else if (check(Kind::Punctuation, "(")) {
                if (node->body.syntax_type != Kernel_Node_Type::IDENTIFIER) fail("only named functions can be called");
                token_index += 1;

                Kernel_List_Node *call = new_node<List_Node>(Kernel_Node_Type::FUNCTION_CALL, node->body.token);
                delete node;
                node = call;
                if (!accept(Kind::Punctuation, ")")) {
                    do {
                        call->children.push_back(parse_expression());
                    } while (accept(Kind::Punctuation, ","));
                    expect(Kind::Punctuation, ")");
                }
            } else if (check(Kind::Operator, ".") || check(Kind::Operator, "->")) {
                const String member_operator = expect(Kind::Operator).lexeme;
                node = new_expression(member_operator, node,
                                      new_node(Kernel_Node_Type::IDENTIFIER, expect(Kind::Identifier).lexeme));
            } else {
                break;
            }
        }
    } catch (...) {
        delete node;
        throw;
    }
    return node;
}

// A type name followed by "(" is a constructor call, e.g. float4(...).
Kernel_Node *Kernel_Parser::parse_primary() {
    if (check(Kind::Number)) return new_node(Kernel_Node_Type::LITERAL, expect(Kind::Number).lexeme);
    if (check(Kind::Identifier)) return new_node(Kernel_Node_Type::IDENTIFIER, expect(Kind::Identifier).lexeme);
    if (check(Kind::Type) && peek(1).kind == Kind::Punctuation && peek(1).lexeme == "(") {
        return new_node(Kernel_Node_Type::IDENTIFIER, expect(Kind::Type).lexeme);
    }
    if (accept(Kind::Punctuation, "(")) {
        Kernel_Node *inner = parse_expression();
        if (!accept(Kind::Punctuation, ")")) {
            delete inner;
            fail("expected ')'");
        }
        return inner;
    }
    fail("expected an expression");
}

}
