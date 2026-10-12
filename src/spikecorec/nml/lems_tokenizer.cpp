#include "dynamics.h"

#include <cctype>
#include <stdexcept>

#include "units.h"
#include "types.h"
#include "node.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

namespace {

// Lexer rules are installed as plain function pointers, so every rule lambda below has
// to be captureless. The tables and helpers those lambdas consult therefore live at
// file scope rather than as locals of tokenize_lems.

// NML spells its comparisons and boolean connectives as dotted words; everything else in
// the expression grammar -- + - * / ( ) , and the function-call form -- is already C.
const Unordered_Map<String, String> DOTTED_OPERATORS = {
    {".gt.", ">"},  {".lt.", "<"},   {".geq.", ">="}, {".leq.", "<="},
    {".eq.", "=="}, {".neq.", "!="}, {".and.", "&&"}, {".or.", "||"},
};

bool is_identifier_start(char character) {
    return isalpha(static_cast<unsigned char>(character)) || character == '_';
}

bool is_identifier_character(char character) {
    return isalnum(static_cast<unsigned char>(character)) || character == '_';
}

// A dotted operator and a decimal point both start with '.', so the two are told apart by
// what follows: ".5" is a number, ".gt." an operator.
bool starts_dotted_operator(const String &text, usize position) {
    return (position + 1 < text.size() &&
            isalpha(static_cast<unsigned char>(text[position + 1])));
}

} // namespace

Vector<Lems_Expression_Token> tokenize_lems(const String &expression, const String &owner_name) {
    Lems_Lexer lexer = Lems_Lexer()
        .add_check([](Lems_Lexer &, char character) -> bool {
            return (isspace(static_cast<u8>(character)));
        }, [](Lems_Lexer &lexer) {
            lexer.position += 1;
        })
        .add_check([](Lems_Lexer &, char character) -> bool {
            return (character == '(' || character == ')' || character == ',');
        },
        [](Lems_Lexer &lexer) {
            static const Unordered_Map<char, Lems_Expression_Token::Kind> token_kinds = {
                {'(', Lems_Expression_Token::Kind::OpenParen},
                {')', Lems_Expression_Token::Kind::CloseParen},
                {',', Lems_Expression_Token::Kind::Comma}
            };

            lexer.tokens.push_back(Lems_Expression_Token{
                token_kinds.at(lexer.current_character),
                String(1, lexer.current_character)
            });
            lexer.position += 1;
        })
        .add_check([](Lems_Lexer &lexer, char character) -> bool {
            return (character == '.' && starts_dotted_operator(lexer.source, lexer.position));
        }, [](Lems_Lexer &lexer) {
            const usize closing_dot = lexer.source.find('.', lexer.position + 1);
            if (closing_dot == String::npos) {
                throw Runtime_Error(
                        "lems_tokenizer: unterminated dotted operator in '" + lexer.source);
            }

            const String spelling = lexer.source.substr(lexer.position, closing_dot - lexer.position + 1);
            auto mapped = DOTTED_OPERATORS.find(spelling);
            if (mapped == DOTTED_OPERATORS.end()) {
                throw Runtime_Error(
                        "lems_tokenizer: unknown operator '" + spelling + "' in '" +
                        lexer.source);
            }

            lexer.tokens.push_back({Lems_Expression_Token::Kind::Operator, mapped->second});
            lexer.position = closing_dot + 1;
        })
        .add_check([](Lems_Lexer &, char character) -> bool {
            return (isdigit(static_cast<u8>(character)) || character == '.');
        }, [](Lems_Lexer &lexer) {
            const usize start = lexer.position;
            while (
                lexer.position < lexer.source.size()
             && (isdigit(static_cast<u8>(lexer.source[lexer.position])) ||
                 lexer.source[lexer.position] == '.')
            ) {
                lexer.position += 1;
            }
            // check for an exponent, and the sign that may follow it, ex: 2.7e-5
            if (
                lexer.position < lexer.source.size() &&
                (lexer.source[lexer.position] == 'e' || lexer.source[lexer.position] == 'E')
            ) {
                usize lookahead = lexer.position + 1;
                if (
                    lookahead < lexer.source.size() &&
                    (lexer.source[lookahead] == '+' || lexer.source[lookahead] == '-')
                ) {
                    lookahead += 1;
                }

                if (
                    lookahead < lexer.source.size() &&
                    isdigit(static_cast<u8>(lexer.source[lookahead]))
                ) {
                    lexer.position = lookahead;
                    while (
                        lexer.position < lexer.source.size() &&
                        isdigit(static_cast<u8>(lexer.source[lexer.position]))
                    ) {
                        lexer.position += 1;
                    }
                }
            }
            lexer.tokens.push_back({
                Lems_Expression_Token::Kind::Number,
                lexer.source.substr(start, lexer.position - start)
            });
        })
        .add_check([](Lems_Lexer &, char character) -> bool {
            return is_identifier_start(character);
        }, [](Lems_Lexer &lexer) {
            const usize start = lexer.position;
            while (
                lexer.position < lexer.source.size() &&
                is_identifier_character(lexer.source[lexer.position])
            ) {
                lexer.position += 1;
            }
            lexer.tokens.push_back({
                Lems_Expression_Token::Kind::Identifier,
                lexer.source.substr(start, lexer.position - start)
            });
        })
        .add_check([](Lems_Lexer &lexer, char) -> bool {
            // Two-character comparisons are also legal LEMS spelling alongside the dotted form.
            static const Set<String> pairs = {"<=", ">=", "==", "!=", "&&", "||"};
            return pairs.count(lexer.source.substr(lexer.position, 2)) != 0;
        }, [](Lems_Lexer &lexer) {
            lexer.tokens.push_back({Lems_Expression_Token::Kind::Operator, lexer.source.substr(lexer.position, 2)});
            lexer.position += 2;
        })
        .add_check([](Lems_Lexer &, char character) -> bool {
            return (String("+-*/^<>").find(character) != String::npos);
        }, [](Lems_Lexer &lexer) {
            lexer.tokens.push_back({
                Lems_Expression_Token::Kind::Operator,
                String(1, lexer.current_character)
            });
            lexer.position += 1;
        });

    Vector<Lems_Expression_Token> tokens;
    try {
        tokens = lexer.lex(expression);
    } catch (const Runtime_Error &error) {
        // Lexer is generic and knows nothing about which NML component it was run for, so
        // the owning component is attached to the message here.
        throw Runtime_Error(String(error.what()) + " (" + owner_name + ")");
    }

    tokens.push_back({Lems_Expression_Token::Kind::End, ""});
    return tokens;
}

}
