#include "dynamics.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "units.h"
#include "types.h"
#include "node.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

namespace {

// Lexer rules are plain function pointers, so their lambdas cannot capture; the tables and
// helpers they use live at file scope.

// LEMS dotted comparison and boolean operators, with their C spelling.
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

// '.' followed by a letter starts a dotted operator (".gt."); otherwise a number (".5").
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
            // Exponent with an optional sign, e.g. 2.7e-5.
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
            // "1.2.3" or "1..5" would otherwise read as 1.2 or 1.0.
            const String literal = lexer.source.substr(start, lexer.position - start);
            const usize mantissa_end = literal.find_first_of("eE");
            const String mantissa = literal.substr(0, mantissa_end);
            if (std::count(mantissa.begin(), mantissa.end(), '.') > 1 || mantissa == ".") {
                throw Runtime_Error("lems_tokenizer: malformed number '" + literal + "' in '" + lexer.source + "'");
            }
            lexer.tokens.push_back({Lems_Expression_Token::Kind::Number, literal});
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
            // LEMS also accepts these C operators alongside the dotted forms.
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
        // The generic lexer does not know the owning component, so it is added here.
        throw Runtime_Error(String(error.what()) + " (" + owner_name + ")");
    }

    tokens.push_back({Lems_Expression_Token::Kind::End, ""});
    return tokens;
}

}
