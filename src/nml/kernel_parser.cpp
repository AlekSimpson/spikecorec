#include "spikecorec/nml/dynamics.h"

#include <stdexcept>

#include "spikecorec/core/types.h"
#include "spikecorec/nml/node.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

namespace {

using Kind = KernelToken::Kind;

const Set<String> ASSIGNMENT_OPERATORS = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="};
const Set<String> PREFIX_OPERATORS = {"-", "+", "!", "~", "++", "--", "*", "&"};

// Implied by the node type, so emission puts them back.
const Set<String> KERNEL_QUALIFIERS = {"kernel", "__global__"};
const Set<String> FUNCTION_QUALIFIERS = {"inline", "static", "extern", "__device__", "__forceinline__"};

s32 binary_precedence(const KernelToken &token) {
    static const UnorderedMap<String, s32> precedences = {
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

const KernelToken &KernelParser::current() const {
    return tokens[token_index];
}

const KernelToken &KernelParser::peek(usize distance) const {
    return tokens[std::min(token_index + distance, tokens.size() - 1)];
}

bool KernelParser::check(Kind kind, const String &lexeme) const {
    return current().kind == kind && (lexeme.empty() || current().lexeme == lexeme);
}

bool KernelParser::accept(Kind kind, const String &lexeme) {
    if (!check(kind, lexeme)) return false;
    token_index += 1;
    return true;
}

KernelToken KernelParser::expect(Kind kind, const String &lexeme) {
    if (!check(kind, lexeme)) fail(lexeme.empty() ? "unexpected token" : "expected '" + lexeme + "'");
    const KernelToken token = current();
    token_index += 1;
    return token;
}

bool KernelParser::starts_type() const {
    return check(Kind::Type) || check(Kind::Qualifier);
}

[[noreturn]] void KernelParser::fail(const String &reason) const {
    String context;
    const usize first = token_index >= 8 ? token_index - 8 : 0;
    for (usize index = first; index <= token_index && index < tokens.size(); index += 1) {
        if (!context.empty()) context += " ";
        context += tokens[index].lexeme;
    }
    throw runtime_error("kernel_parser: " + reason + " at '" + current().lexeme + "', after: " + context);
}

KernelListNode *KernelParser::parse_program() {
    KernelListNode *program = new_node<ListNode>(KernelNodeType::PROGRAM, "");
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

KernelNode *KernelParser::parse_preprocessor() {
    const String line = expect(Kind::Preprocessor).lexeme;
    const String directive_and_rest = trim(line.substr(1));
    const usize directive_end = std::min(directive_and_rest.find_first_of(" \t"), directive_and_rest.size());
    const String directive = directive_and_rest.substr(0, directive_end);
    const String rest = trim(directive_and_rest.substr(directive_end));

    if (directive == "include") return new_node(KernelNodeType::INCLUDE, rest);

    if (directive == "define") {
        const usize name_end = std::min(rest.find_first_of(" \t"), rest.size());
        KernelBinaryNode *define = new_node<BinaryNode>(KernelNodeType::DEFINE, "");
        define->left = new_node(KernelNodeType::IDENTIFIER, rest.substr(0, name_end));
        const String replacement = trim(rest.substr(name_end));
        if (!replacement.empty()) define->right = new_node(KernelNodeType::LITERAL, replacement);
        return define;
    }

    token_index -= 1;
    fail("unsupported preprocessor directive '#" + directive + "'");
}

KernelNode *KernelParser::parse_using_namespace() {
    expect(Kind::Identifier, "using");
    expect(Kind::Identifier, "namespace");
    const String namespace_name = expect(Kind::Identifier).lexeme;
    expect(Kind::Punctuation, ";");
    return new_node(KernelNodeType::USING_NAMESPACE, namespace_name);
}

KernelNode *KernelParser::parse_function() {
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

    KernelListNode *function = new_node<ListNode>(
            is_kernel ? KernelNodeType::KERNEL_FUNCTION_IMPL : KernelNodeType::DEVICE_FUNCTION_IMPL, "");
    try {
        function->children.push_back(parse_type());
        function->children.push_back(new_node(KernelNodeType::IDENTIFIER, expect(Kind::Identifier).lexeme));
        function->children.push_back(parse_parameter_list());
        function->children.push_back(parse_block());
    } catch (...) {
        delete function;
        throw;
    }
    return function;
}

KernelListNode *KernelParser::parse_parameter_list() {
    KernelListNode *parameters = new_node<ListNode>(KernelNodeType::PARAMETER_LIST, "");
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
KernelNode *KernelParser::parse_parameter() {
    KernelTrinaryNode *parameter = new_node<TrinaryNode>(KernelNodeType::PARAMETER, "");
    try {
        parameter->left = parse_type();
        parameter->middle = new_node(KernelNodeType::IDENTIFIER, expect(Kind::Identifier).lexeme);
        if (check(Kind::Attribute)) {
            const String attribute = expect(Kind::Attribute).lexeme;
            if (attribute.rfind("buffer(", 0) != 0) parameter->right = new_node(KernelNodeType::ATTRIBUTE, attribute);
        }
    } catch (...) {
        delete parameter;
        throw;
    }
    return parameter;
}

// Qualifiers and base type words, then any "*" and "&". The base type is normalized to its
// neutral name; a type with no neutral name keeps its spelling.
KernelNode *KernelParser::parse_type() {
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

    KernelNode *type = new_node(KernelNodeType::TYPE, qualifiers.empty() ? base_type : qualifiers + " " + base_type);
    while (true) {
        if (accept(Kind::Operator, "*")) {
            KernelUnaryNode *pointer = new_node<UnaryNode>(KernelNodeType::POINTER, "");
            pointer->child = type;
            while (check(Kind::Qualifier)) {
                if (!pointer->body.token.empty()) pointer->body.token += " ";
                pointer->body.token += current().lexeme;
                token_index += 1;
            }
            type = pointer;
        } else if (accept(Kind::Operator, "&")) {
            KernelUnaryNode *reference = new_node<UnaryNode>(KernelNodeType::REFERENCE, "");
            reference->child = type;
            type = reference;
        } else {
            break;
        }
    }
    return type;
}

KernelListNode *KernelParser::parse_block() {
    if (!check(Kind::Punctuation, "{")) fail("expected '{': bodies of if, else, for and while must be braced");
    token_index += 1;

    KernelListNode *block = new_block();
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

KernelNode *KernelParser::parse_statement() {
    if (check(Kind::Punctuation, "{")) return parse_block();
    if (check(Kind::Keyword, "if")) return parse_conditional();
    if (check(Kind::Keyword, "for")) return parse_for();
    if (check(Kind::Keyword, "while")) return parse_while();
    if (check(Kind::Keyword, "return") || check(Kind::Keyword, "break") || check(Kind::Keyword, "continue")) {
        return parse_jump();
    }
    if (check(Kind::Keyword) || check(Kind::Preprocessor)) fail("unsupported statement");

    KernelNode *statement = parse_simple_statement();
    if (!accept(Kind::Punctuation, ";")) {
        delete statement;
        fail("expected ';'");
    }
    return statement;
}

// A declaration, assignment or expression, without its ';' so for headers can use it.
// A statement-level x++ / x-- becomes x += 1 / x -= 1.
KernelNode *KernelParser::parse_simple_statement() {
    if (starts_type()) return parse_declaration();

    KernelNode *target = parse_expression();
    if (check(Kind::Operator) && ASSIGNMENT_OPERATORS.count(current().lexeme)) {
        const String assignment_operator = expect(Kind::Operator).lexeme;
        KernelNode *value = nullptr;
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
        return new_assignment(target, step_operator, new_node(KernelNodeType::LITERAL, "1"));
    }
    return target;
}

KernelNode *KernelParser::parse_declaration() {
    KernelTrinaryNode *declaration = new_node<TrinaryNode>(KernelNodeType::DECLARATION, "");
    try {
        declaration->left = parse_type();
        declaration->middle = new_node(KernelNodeType::IDENTIFIER, expect(Kind::Identifier).lexeme);
        if (accept(Kind::Punctuation, "[")) {
            KernelNode *size = parse_expression();
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

KernelListNode *KernelParser::parse_conditional() {
    KernelListNode *conditional = new_conditional();
    try {
        expect(Kind::Keyword, "if");
        do {
            expect(Kind::Punctuation, "(");
            KernelNode *test = parse_expression();
            KernelNode *body = nullptr;
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

KernelNode *KernelParser::parse_for() {
    KernelListNode *loop = new_node<ListNode>(KernelNodeType::FOR, "");
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

KernelNode *KernelParser::parse_while() {
    KernelBinaryNode *loop = new_node<BinaryNode>(KernelNodeType::WHILE, "");
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

KernelNode *KernelParser::parse_jump() {
    KernelUnaryNode *jump = new_node<UnaryNode>(KernelNodeType::JUMP, expect(Kind::Keyword).lexeme);
    try {
        if (jump->body.token == "return" && !check(Kind::Punctuation, ";")) jump->child = parse_expression();
        expect(Kind::Punctuation, ";");
    } catch (...) {
        delete jump;
        throw;
    }
    return jump;
}

KernelNode *KernelParser::parse_expression() {
    KernelNode *test = parse_binary(1);
    if (!accept(Kind::Operator, "?")) return test;

    KernelTrinaryNode *ternary = new_node<TrinaryNode>(KernelNodeType::EXPRESSION, "?:");
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

KernelNode *KernelParser::parse_binary(s32 minimum_precedence) {
    KernelNode *left = parse_unary();
    while (true) {
        const s32 precedence = binary_precedence(current());
        if (precedence < minimum_precedence) break;

        const String expression_operator = expect(Kind::Operator).lexeme;
        KernelNode *right = nullptr;
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
KernelNode *KernelParser::parse_unary() {
    if (check(Kind::Operator) && PREFIX_OPERATORS.count(current().lexeme)) {
        const String expression_operator = expect(Kind::Operator).lexeme;
        return new_unary_expression(expression_operator, parse_unary());
    }

    if (check(Kind::Punctuation, "(") && (peek(1).kind == Kind::Type || peek(1).kind == Kind::Qualifier)) {
        const usize open_index = token_index;
        token_index += 1;
        KernelNode *type = parse_type();
        if (!accept(Kind::Punctuation, ")")) {
            delete type;
            token_index = open_index;
            return parse_postfix();
        }

        KernelNode *value = nullptr;
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

KernelNode *KernelParser::parse_postfix() {
    KernelNode *node = parse_primary();
    try {
        while (true) {
            if (accept(Kind::Punctuation, "[")) {
                KernelNode *index = parse_expression();
                node = new_expression("[]", node, index);
                expect(Kind::Punctuation, "]");
            } else if (check(Kind::Punctuation, "(")) {
                if (node->body.syntax_type != KernelNodeType::IDENTIFIER) fail("only named functions can be called");
                token_index += 1;

                KernelListNode *call = new_node<ListNode>(KernelNodeType::FUNCTION_CALL, node->body.token);
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
                                      new_node(KernelNodeType::IDENTIFIER, expect(Kind::Identifier).lexeme));
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
KernelNode *KernelParser::parse_primary() {
    if (check(Kind::Number)) return new_node(KernelNodeType::LITERAL, expect(Kind::Number).lexeme);
    if (check(Kind::Identifier)) return new_node(KernelNodeType::IDENTIFIER, expect(Kind::Identifier).lexeme);
    if (check(Kind::Type) && peek(1).kind == Kind::Punctuation && peek(1).lexeme == "(") {
        return new_node(KernelNodeType::IDENTIFIER, expect(Kind::Type).lexeme);
    }
    if (accept(Kind::Punctuation, "(")) {
        KernelNode *inner = parse_expression();
        if (!accept(Kind::Punctuation, ")")) {
            delete inner;
            fail("expected ')'");
        }
        return inner;
    }
    fail("expected an expression");
}

}
