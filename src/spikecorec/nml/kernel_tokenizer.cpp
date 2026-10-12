#include "dynamics.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "types.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

namespace {

const Set<String> KEYWORDS = {"if", "else", "for", "while", "return", "break", "continue"};

const Set<String> METAL_TYPE_WORDS = {
    "void", "bool", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong", "float", "half",
    "float2", "float4", "int2", "int4", "uint2", "uint4", "atomic_int", "atomic_uint", "atomic_float"};
const Set<String> CUDA_TYPE_WORDS = {
    "void", "bool", "char", "short", "int", "long", "unsigned", "signed", "float", "double",
    "float2", "float4", "int2", "int4", "uint2", "uint4"};

const Set<String> METAL_QUALIFIERS = {
    "const", "constexpr", "static", "inline", "kernel", "device", "constant", "thread", "threadgroup"};
const Set<String> CUDA_QUALIFIERS = {
    "const", "constexpr", "static", "inline", "extern", "__global__", "__device__", "__forceinline__",
    "__restrict__", "__shared__", "__constant__"};

// longest first, so "<<=" is never read as "<<" then "="
const Vector<String> OPERATORS = {
    "<<=", ">>=",
    "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "::",
    "+", "-", "*", "/", "%", "<", ">", "=", "!", "&", "|", "^", "~", "?", ":", "."};

const String OPERATOR_CHARACTERS = "+-*/%<>=!&|^~?:.";
const String PUNCTUATION = "(){}[],;";

bool is_identifier_character(char character) {
    return isalnum(static_cast<u8>(character)) || character == '_';
}

char next_character(const Kernel_Lexer &lexer) {
    return lexer.position + 1 < lexer.source.size() ? lexer.source[lexer.position + 1] : '\0';
}

bool is_whitespace(Kernel_Lexer &, char character) { 
    return isspace(static_cast<u8>(character)); 
}

void skip_whitespace(Kernel_Lexer &lexer) { lexer.position += 1; }

bool starts_comment(Kernel_Lexer &lexer, char character) {
    return character == '/' && (next_character(lexer) == '/' || next_character(lexer) == '*');
}

void skip_comment(Kernel_Lexer &lexer) {
    const bool line_comment = next_character(lexer) == '/';
    const usize end = lexer.source.find(line_comment ? "\n" : "*/", lexer.position + 2);
    if (end == String::npos) lexer.position = lexer.source.size();
    else lexer.position = end + (line_comment ? 1 : 2);
}

bool starts_preprocessor(Kernel_Lexer &, char character) { 
    return character == '#'; 
}

void push_preprocessor(Kernel_Lexer &lexer) {
    const usize end = std::min(lexer.source.find('\n', lexer.position), lexer.source.size());
    lexer.tokens.push_back({Kernel_Token::Kind::Preprocessor, lexer.source.substr(lexer.position, end - lexer.position)});
    lexer.position = end;
}

// metal only: "[[ buffer(5) ]]" is one token, lexeme "buffer(5)"
bool starts_attribute(Kernel_Lexer &lexer, char character) { 
    return character == '[' && next_character(lexer) == '['; 
}

void push_attribute(Kernel_Lexer &lexer) {
    const usize end = lexer.source.find("]]", lexer.position + 2);
    if (end == String::npos) throw Runtime_Error("kernel_tokenizer: unterminated [[ attribute");
    String inner = lexer.source.substr(lexer.position + 2, end - lexer.position - 2);
    inner.erase(0, inner.find_first_not_of(" \t"));
    inner.erase(inner.find_last_not_of(" \t") + 1);
    lexer.tokens.push_back({Kernel_Token::Kind::Attribute, inner});
    lexer.position = end + 2;
}

bool starts_number(Kernel_Lexer &lexer, char character) {
    return isdigit(static_cast<u8>(character)) || (character == '.' && isdigit(static_cast<u8>(next_character(lexer))));
}

void push_number(Kernel_Lexer &lexer) {
    const String &text = lexer.source;
    const usize start = lexer.position;
    usize &position = lexer.position;
    while (position < text.size() && (isdigit(static_cast<u8>(text[position])) || text[position] == '.')) position += 1;
    if (position < text.size() && (text[position] == 'e' || text[position] == 'E')) {
        usize lookahead = position + 1;
        if (lookahead < text.size() && (text[lookahead] == '+' || text[lookahead] == '-')) lookahead += 1;
        if (lookahead < text.size() && isdigit(static_cast<u8>(text[lookahead]))) {
            position = lookahead;
            while (position < text.size() && isdigit(static_cast<u8>(text[position]))) position += 1;
        }
    }
    // Suffixes, e.g. 0.0f, 1u, 20ul
    while (position < text.size() && String("fFuUlL").find(text[position]) != String::npos) position += 1;
    lexer.tokens.push_back({Kernel_Token::Kind::Number, text.substr(start, position - start)});
}

bool starts_string(Kernel_Lexer &, char character) { 
    return character == '"'; 
}

void push_string(Kernel_Lexer &lexer) {
    const usize end = lexer.source.find('"', lexer.position + 1);
    if (end == String::npos) throw Runtime_Error("kernel_tokenizer: unterminated string literal");

    lexer.tokens.push_back({Kernel_Token::Kind::StringLiteral, lexer.source.substr(lexer.position, end - lexer.position + 1)});
    lexer.position = end + 1;
}

bool starts_word(Kernel_Lexer &, char character) { 
    return isalpha(static_cast<u8>(character)) || character == '_'; 
}

void push_word(Kernel_Lexer &lexer, const Set<String> &type_words, const Set<String> &qualifiers) {
    const usize start = lexer.position;
    while (lexer.position < lexer.source.size() && 
           is_identifier_character(lexer.source[lexer.position])) {
        lexer.position += 1;
    }
    const String word = lexer.source.substr(start, lexer.position - start);

    Kernel_Token::Kind kind = Kernel_Token::Kind::Identifier;
    if      (KEYWORDS.count(word))   kind = Kernel_Token::Kind::Keyword;
    else if (type_words.count(word)) kind = Kernel_Token::Kind::Type;
    else if (qualifiers.count(word)) kind = Kernel_Token::Kind::Qualifier;

    lexer.tokens.push_back({kind, word});
}

void push_metal_word(Kernel_Lexer &lexer) { 
    push_word(lexer, METAL_TYPE_WORDS, METAL_QUALIFIERS); 
}

void push_cuda_word(Kernel_Lexer &lexer) { 
    push_word(lexer, CUDA_TYPE_WORDS, CUDA_QUALIFIERS); 
}

// every operator character is itself a one-character operator, so this always advances
bool starts_operator(Kernel_Lexer &, char character) { 
    return OPERATOR_CHARACTERS.find(character) != String::npos; 
}

void push_operator(Kernel_Lexer &lexer) {
    for (const String &spelling : OPERATORS) {
        if (lexer.source.compare(lexer.position, spelling.size(), spelling) != 0) continue;
        lexer.tokens.push_back({Kernel_Token::Kind::Operator, spelling});
        lexer.position += spelling.size();
        return;
    }
}

bool is_punctuation(Kernel_Lexer &, char character) { 
    return PUNCTUATION.find(character) != String::npos; 
}

void push_punctuation(Kernel_Lexer &lexer) {
    lexer.tokens.push_back({Kernel_Token::Kind::Punctuation, String(1, lexer.current_character)});
    lexer.position += 1;
}

} // namespace

Vector<Kernel_Token> tokenize_kernel(const String &source, Kernel_Backend backend) {
    Kernel_Lexer lexer;
    lexer.add_check(is_whitespace, skip_whitespace)
         .add_check(starts_comment, skip_comment)
         .add_check(starts_preprocessor, push_preprocessor);

    if (backend == Kernel_Backend::METAL) lexer.add_check(starts_attribute, push_attribute);

    lexer.add_check(starts_number, push_number)
         .add_check(starts_string, push_string)
         .add_check(starts_word, backend == Kernel_Backend::METAL ? push_metal_word : push_cuda_word)
         .add_check(starts_operator, push_operator)
         .add_check(is_punctuation, push_punctuation);

    Vector<Kernel_Token> tokens = lexer.lex(source);
    tokens.push_back({Kernel_Token::Kind::End, ""});
    return tokens;
}

}
