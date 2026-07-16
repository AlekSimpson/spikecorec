#pragma once

#include <memory>
#include <optional>
#include <unordered_set>

#include "spikecorec/nml/ir.h"
#include "spikecorec/nml/nml.h"

namespace spikecorec::nml {

// ── Shared LEMS-expression lowering machinery (ticket #50 [B2]; split out of
// cell_lowering.cpp by ticket #51 [B3] so synapse lowering can reuse it
// without duplicating it) ────────────────────────────────────────────────
//
// Everything a per-ComponentType Dynamics lowering needs to turn a LEMS
// `value=`/`test=` expression string into `.tick` instructions: the minimal
// arithmetic/comparison expression parser (scoped to what GLIF1-5 and the
// D3 synapse family actually need -- see cell_lowering.cpp's original header
// comment for the exact scoping rationale, unchanged by this split),
// `LoweringContext` (identifier resolution + `emit_expression`'s
// register-reuse/register-aliasing-guard codegen), the linear-decay-shape
// detector driving `expdecay` selection, and the `TimeDerivative` lowering
// entry point built on top of it. Cell lowering (#50) and synapse lowering
// (#51) both depend on this; neither owns it.
//
// Deliberately NOT moved here (stayed cell-specific in cell_lowering.cpp,
// since nothing else needs them): Regime/OnCondition/OnStart handling --
// synapses have no Regimes, and Phase-1 synapse OnStart values are always
// the zero default already documented for an absent `StateDirective`/
// `AccumDirective`/`PeredgeDirective` initial value (see synapse_lowering.cpp's
// own header comment).

// `get_attribute_value`/`find_children`: small NML_Node helpers mirroring
// nml.cpp's own private helpers (not reused across translation units for the
// same reason resolve.cpp gives for its own mirror of them -- see
// cell_lowering.cpp's original comment). Generic over any NML_Node, so
// equally usable for a CellType's or a SynapseType's raw Dynamics-nested
// bodies (`RegimeDecl::body`, `OnConditionDecl::body`, `OnEventDecl::body`, ...).
String get_attribute_value(const NML_Node &node, const String &attribute_name);
Vector<const NML_Node *> find_children(const NML_Node &node, const String &tag_name);

// ── Minimal LEMS math-expression parser ──────────────────────────────────
//
// Arithmetic `+ - * /`, parentheses, unary minus, and the six LEMS
// dot-operator comparisons (`.gt. .lt. .geq. .leq. .eq. .neq.`). No `^`/pow
// and no function-call syntax (`exp(...)` etc.) -- see cell_lowering.cpp's
// GLIF1-5 fixtures and this ticket's D3 synapse fixtures, neither of which
// need them.

enum class ExpressionNodeKind { Number, Identifier, Negate, Binary };

struct ExpressionNode {
    ExpressionNodeKind kind;
    String text;                       // Number literal text, or Identifier name
    char binary_operator_character = '\0'; // '+' '-' '*' '/' for Binary
    std::unique_ptr<ExpressionNode> left;
    std::unique_ptr<ExpressionNode> right; // unused for Negate (operand is `left`)
};

using ExpressionNodePointer = std::unique_ptr<ExpressionNode>;

// Parses a bare arithmetic expression (no top-level comparison) into an AST.
// Throws std::runtime_error (via log::throw_runtime_error) on a malformed or
// unrecognized expression. `context_for_errors` is folded into any thrown
// message (e.g. "TimeDerivative 'v'", "condition '...'") so a parse failure
// names where it came from.
ExpressionNodePointer parse_arithmetic_text(const String &expression_text, const String &context_for_errors);

// One parsed LEMS `test=` condition: exactly one top-level dot-comparison
// operator (LEMS forbids nested comparisons), split into its two arithmetic
// operand subtrees plus the comparison opcode.
struct ParsedCondition {
    ExpressionNodePointer left;
    BinaryOpcode opcode;
    ExpressionNodePointer right;
};

// Parses a `test="a .gt. b"`-shaped condition string. Throws
// std::runtime_error if it has zero or more than one top-level comparison
// operator.
ParsedCondition parse_condition_text(const String &condition_text);

// Whether `identifier_name` is read anywhere inside `node`'s subtree. Used
// by the integration-method shape detector below and by
// `LoweringContext::emit_expression`'s self-reference guard.
bool references_identifier(const ExpressionNode &node, const String &identifier_name);

// Holds everything needed to turn one ComponentType's parsed expressions
// into `.tick` instructions: which names are legal to reference (state
// variables, parameters/properties, plain-value derived variables,
// requirements), an alias map for names that actually resolve to some OTHER
// operand (a cell's `select`/`reduce="add"` DerivedVariable collapsing to
// `network_inputs`, arch §3.2; or -- ticket #51 -- a per-edge synapse
// variable resolving to whatever fresh temporary a `loadedge` most recently
// loaded it into), and a fresh-temporary counter for `t0, t1, ...`
// intermediates (IR spec §3.1).
class LoweringContext {
public:
    LoweringContext(std::unordered_set<String> known_names, UnorderedMap<String, String> identifier_aliases);

    String resolve_identifier(const String &identifier_name) const;

    String fresh_temporary();

    // Each top-level Dynamics statement (one TimeDerivative, one
    // DerivedVariable, one StateAssignment, one condition test) is fully
    // self-contained -- its own `t0, t1, ...` scratch is dead the moment the
    // statement's final instruction runs -- so the counter resets per
    // statement rather than growing across the whole program.
    void reset_temporary_counter();

    // Lowers `node`'s value into `output`, writing the final result into
    // `preferred_destination` if given (else a fresh `tN` register).
    // Returns the operand name holding the result. See cell_lowering.cpp's
    // original doc comment (preserved verbatim in expression_lowering.cpp)
    // for the register-reuse strategy and the register-aliasing guard this
    // implements (a review fix on ticket #50, load-bearing for both cell and
    // synapse lowering).
    String emit_expression(const ExpressionNode &node, Vector<TickInstruction> &output,
                           std::optional<String> preferred_destination);

private:
    std::unordered_set<String> known_names_;
    UnorderedMap<String, String> identifier_aliases_;
    int temporary_counter_ = 0;
};

// Lowers one `TimeDerivative variable="<state_variable_name>" value="<right_hand_side_text>"`
// into `output`: the closed-form `expdecay` instruction(s) when the
// right-hand side matches a recognized linear-decay shape, else a
// forward-Euler fallback (accumulate the right-hand side, scale by `dt`, add
// to state). Resets `context`'s temporary counter itself (one top-level
// Dynamics statement, see LoweringContext::reset_temporary_counter).
void lower_time_derivative(const String &state_variable_name, const String &right_hand_side_text,
                            Vector<TickInstruction> &output, LoweringContext &context);

// ── Linear-decay-shape detection (IR spec §3.3 `expdecay dst,a,tau` =
// `a*exp(-dt/tau)`) -- exposed (ticket #51 revision) so synapse_lowering.cpp
// can drive its own per-edge accumulate-only decay writeback (`loadedge` the
// old value, `expdecay`/target-shift to the new one, `accedge` the delta)
// off the exact same recognized shape `lower_time_derivative` uses for the
// direct-mutation cell-side case. ────────────────────────────────────────

// A right-hand side recognized as exponential decay: `target` is `"0"` for
// `-state/tau`, else the identifier/literal a `(target-state)/tau` shape
// decays toward.
struct LinearDecayShape {
    String target;
    String time_constant;
};

// Recognizes exactly two shapes (arch §3.2 TimeDerivative; "keep it simple
// and pattern-based" -- not a general symbolic linearity prover):
//   1. `(target - state) / tau`, where `target`/`tau` are each a bare
//      identifier or number literal that isn't `state` itself and doesn't
//      reference `network_inputs`.
//   2. `-state / tau` (equivalent to target 0 -- after-spike-current decay).
// Returns nullopt for anything else.
std::optional<LinearDecayShape> detect_linear_decay_shape(const ExpressionNode &right_hand_side, const String &state_variable_name);

// Renders an f64 back to a literal token suitable for a `.alloc`
// `ParamConstantDirective`/`.tick` operand (`%.17g`, round-trippable).
String format_literal(f64 value);

} // namespace spikecorec::nml
