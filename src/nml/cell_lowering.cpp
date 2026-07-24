#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <algorithm>
#include <optional>
#include <unordered_set>
#include <utility>

#include "spikecorec/nml/cell_lowering.h"
#include "spikecorec/nml/expression_lowering.h"
#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::nml {

namespace {

// The shared LEMS-expression parser/emitter (tokenizer, ExpressionParser,
// LoweringContext::emit_expression + its register-aliasing guard,
// detect_linear_decay_shape, lower_time_derivative) and the small NML_Node
// helpers (get_attribute_value/find_children) all moved to
// expression_lowering.h/.cpp in ticket #51 [B3], so synapse lowering can
// reuse them without duplicating them -- see that header's own doc comment.
// What stays here is cell-specific: Regime/OnCondition/OnStart handling
// (synapses have no Regimes) and lower_cell_to_ir's own `.alloc`/`.tick`
// assembly.

void lower_state_assignment_into(const NML_Node &state_assignment_node, Vector<TickInstruction> &output, LoweringContext &context) {
    String assigned_variable_name = get_attribute_value(state_assignment_node, "variable");
    String value_text = get_attribute_value(state_assignment_node, "value");
    context.reset_temporary_counter();
    String context_for_errors = "StateAssignment '" + assigned_variable_name + "' = '" + value_text + "'";
    ExpressionNodePointer value_expression = parse_arithmetic_text(value_text, context_for_errors);
    context.emit_expression(*value_expression, output, assigned_variable_name);
}

} // namespace

// ── Regime body parsing (arch §3.2 Regime/Transition/OnEntry; IR spec §4's
// refractory-regime example) -- `RegimeDecl::body` is the raw `<Regime>`
// node, walked the same way `<Dynamics>` itself is walked by nml.cpp's
// (private, not reused here) extractors. ─────────────────────────────────
//
// RegimeTimeDerivative/RegimeOnCondition/RegimeInfo and gather_regime_info now live in
// cell_lowering.h (this cleanup) -- gather_regime_info is real, directly-callable public API rather
// than a private helper only this file's own lower_cell_to_ir could reach.

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

namespace {

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
    const UnorderedMap<String, Vector<const NML_Node *>> &on_entry_assignments_of_regime,
    const String &context_label
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
        } else {
            // Any other child tag is not part of arch §3.2's OnCondition body
            // (StateAssignment/EventOut/Transition are the only three legal ones) -- NOT lowered
            // into `.tick` at all, same class of gap as synapse_lowering.cpp's unrecognized
            // TimeDerivative decay shape warning: a silent drop here would leave a real cell's
            // OnCondition action missing from the generated kernel with no build-time signal, so
            // this at least warns with enough to locate it (which cell, which OnCondition, which
            // child tag) even though lowering it is out of Phase-1 scope.
            log::logger().warn(
                "cell_lowering: {} has an unsupported OnCondition child '<{}>' (only "
                "StateAssignment/EventOut/Transition are lowered -- this child is NOT lowered into '.tick')",
                context_label, child.tag_name);
        }
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

// ── Active-set x nonlinear-dynamics classification (ticket #62 [F1]; arch §0.5) ──────────────────
//
// Whether every TimeDerivative this cell type declares (its own top-level ones, plus every regime's
// own) is AFFINE in the cell's own state variables -- i.e. whether the cell's dynamics, treating
// network_inputs/parameters/constants as external, time-invariant-during-a-skip forcing terms, form
// a linear ODE system with a closed-form solution the engine's active-set optimization can lazily
// fast-forward across skipped ticks (arch §0.5's "linear (all of GLIF)" vs "nonlinear
// (izhikevich/AdEx/HH)" split). This generalizes expression_lowering.h's own detect_linear_decay_shape
// (a narrow single-shape pattern match -- `(target-state)/tau` or `-state/tau` -- used to pick
// expdecay vs forward-Euler for ONE state variable's own TimeDerivative) to a whole-system structural
// check across every state variable a cell declares: a real GLIF fixture's `v` dynamics (e.g.
// `(gL*(EL-v)+iSyn)/C`) is affine in `v` but is NOT the narrow shape detect_linear_decay_shape
// recognizes (it has a `C`/`gL` coefficient and an `iSyn` term), so classifying cells with that
// function directly would misclassify every real GLIF fixture as nonlinear -- the more general
// affine check below is what correctly keeps "all of GLIF" tagged linear.
//
// Referenced identifiers OTHER than the cell's own state variables (parameters, network_inputs) are
// treated as opaque affine leaves -- EXCEPT a plain `value=` DerivedVariable's own name (ticket #63
// [F2]'s own extension over ticket #62's original, narrower version of this check, which treated
// every non-state-variable identifier, DerivedVariable names included, as an opaque leaf): every one
// of ticket #63's real nonlinear cells hides its actual nonlinear term behind exactly this indirection
// -- `izhikevich2007Cell`'s own `v` TimeDerivative is just `iMemb / C`, with the real
// `k*(v-vr)*(v-vt)` product living inside `iMemb`'s own DerivedVariable definition;
// `hindmarshRose1984Cell` nests three levels deep (`iMemb` reads `phi`/`z`, `phi` reads `x`, `x` reads
// `v`) -- so AffineCheckContext below transitively inlines a plain-value DerivedVariable's own
// definition wherever its name is read, closing the exact gap ticket #62's own header comment (see
// its git history) flagged as out of scope at the time ("no Phase-1 GLIF fixture does this... closing
// that gap is out of this ticket's scope"). A `select`/`reduce="add"` alias (e.g. `iSyn`) has no
// entry in `derived_variable_value_by_name` (its own `.value` is empty), so it is still an ordinary
// opaque leaf, exactly as before -- this is a strict extension, not a behavior change for GLIF1-5.

struct AffineCheckContext {
    const std::unordered_set<String> &state_variable_names;
    // Plain `value=` DerivedVariables only (name -> raw value text) -- a select/reduce alias has no
    // entry here, see this section's own header comment.
    const UnorderedMap<String, String> &derived_variable_value_by_name;
    // DFS "currently being resolved" guard (pushed before recursing into a name's own definition,
    // popped after) -- NOT an "ever visited" set, so the SAME DerivedVariable name legitimately read
    // from two independent places (hindmarshRose1984Cell's own `x`, read once each by `phi`/`chi`/
    // `rho`) is still resolved correctly both times; only a genuine cyclic definition (never expected
    // in a real LEMS ComponentType) is conservatively short-circuited.
    std::unordered_set<String> &names_currently_being_resolved;
};

bool expression_depends_on_any_state_variable(const ExpressionNode &node, AffineCheckContext &context);
bool expression_is_affine_in_state_variables(const ExpressionNode &node, AffineCheckContext &context);

bool identifier_transitively_depends_on_state(const String &identifier_name, AffineCheckContext &context) {
    auto definition = context.derived_variable_value_by_name.find(identifier_name);
    if (definition == context.derived_variable_value_by_name.end()) return false; // an ordinary parameter/opaque leaf
    if (!context.names_currently_being_resolved.insert(identifier_name).second) return false; // cyclic reference guard
    ExpressionNodePointer parsed_definition = parse_arithmetic_text(
        definition->second, "DerivedVariable '" + identifier_name + "' dependency classification (ticket #63)");
    bool depends = expression_depends_on_any_state_variable(*parsed_definition, context);
    context.names_currently_being_resolved.erase(identifier_name);
    return depends;
}

bool identifier_transitively_affine(const String &identifier_name, AffineCheckContext &context) {
    auto definition = context.derived_variable_value_by_name.find(identifier_name);
    if (definition == context.derived_variable_value_by_name.end()) return true; // an ordinary parameter/opaque leaf
    if (!context.names_currently_being_resolved.insert(identifier_name).second) return true; // cyclic reference guard
    ExpressionNodePointer parsed_definition = parse_arithmetic_text(
        definition->second, "DerivedVariable '" + identifier_name + "' linearity classification (ticket #63)");
    bool affine = expression_is_affine_in_state_variables(*parsed_definition, context);
    context.names_currently_being_resolved.erase(identifier_name);
    return affine;
}

bool expression_depends_on_any_state_variable(const ExpressionNode &node, AffineCheckContext &context) {
    switch (node.kind) {
        case ExpressionNodeKind::Identifier:
            if (context.state_variable_names.count(node.text)) return true;
            return identifier_transitively_depends_on_state(node.text, context);
        case ExpressionNodeKind::Number: return false;
        case ExpressionNodeKind::Negate: return expression_depends_on_any_state_variable(*node.left, context);
        case ExpressionNodeKind::FunctionCall: return expression_depends_on_any_state_variable(*node.left, context);
        case ExpressionNodeKind::Binary:
            return expression_depends_on_any_state_variable(*node.left, context) ||
                   expression_depends_on_any_state_variable(*node.right, context);
    }
    return false;
}

bool expression_is_affine_in_state_variables(const ExpressionNode &node, AffineCheckContext &context) {
    switch (node.kind) {
        case ExpressionNodeKind::Number:
            return true;
        case ExpressionNodeKind::Identifier:
            if (context.state_variable_names.count(node.text)) return true;
            return identifier_transitively_affine(node.text, context);
        case ExpressionNodeKind::Negate:
            return expression_is_affine_in_state_variables(*node.left, context);
        case ExpressionNodeKind::FunctionCall:
            // exp/log/sin/... of anything that depends on a state variable is never affine; of a
            // pure-parameter/constant argument it is just another opaque (affine) leaf (ticket #63:
            // `adExIaFCell`'s own `exp((v-VT)/delT)`).
            return !expression_depends_on_any_state_variable(*node.left, context);
        case ExpressionNodeKind::Binary:
            if (node.binary_operator_character == '+' || node.binary_operator_character == '-') {
                return expression_is_affine_in_state_variables(*node.left, context) &&
                       expression_is_affine_in_state_variables(*node.right, context);
            }
            if (node.binary_operator_character == '*') {
                bool left_depends = expression_depends_on_any_state_variable(*node.left, context);
                bool right_depends = expression_depends_on_any_state_variable(*node.right, context);
                if (left_depends && right_depends) return false; // a state variable times another -- nonlinear
                return expression_is_affine_in_state_variables(*node.left, context) &&
                       expression_is_affine_in_state_variables(*node.right, context);
            }
            if (node.binary_operator_character == '/') {
                // A state variable in the denominator (e.g. `1/v`) is never affine.
                if (expression_depends_on_any_state_variable(*node.right, context)) return false;
                return expression_is_affine_in_state_variables(*node.left, context);
            }
            return false; // '^' (Pow, ticket #63) and anything else -- never affine
    }
    return false;
}

} // namespace

// cell_dynamics_are_closed_form_advanceable now lives in cell_lowering.h (this cleanup) -- real,
// directly-callable public API instead of only reachable by reading a stored tag back off a
// constructed IrProgram. `regimes` is `lower_cell_to_ir`'s own already-gathered regime metadata
// (gather_regime_info, also declared there) -- passed in rather than recomputed so this stays a pure
// structural check over data the caller already has.
bool cell_dynamics_are_closed_form_advanceable(const CellType &cell, const Vector<RegimeInfo> &regimes) {
    std::unordered_set<String> state_variable_names;
    for (const auto &state_variable : cell.state_variables) state_variable_names.insert(state_variable.name);

    UnorderedMap<String, String> derived_variable_value_by_name;
    for (const auto &derived_variable : cell.derived_variables) {
        if (derived_variable.select.empty() && !derived_variable.value.empty()) {
            derived_variable_value_by_name[derived_variable.name] = derived_variable.value;
        }
    }

    std::unordered_set<String> names_currently_being_resolved;
    AffineCheckContext context{state_variable_names, derived_variable_value_by_name, names_currently_being_resolved};

    auto right_hand_side_is_affine = [&](const String &value_text) {
        ExpressionNodePointer right_hand_side =
            parse_arithmetic_text(value_text, "TimeDerivative linearity classification (ticket #62)");
        return expression_is_affine_in_state_variables(*right_hand_side, context);
    };

    for (const auto &time_derivative : cell.time_derivatives) {
        if (!right_hand_side_is_affine(time_derivative.value)) return false;
    }
    for (const auto &regime : regimes) {
        for (const auto &time_derivative : regime.time_derivatives) {
            if (!right_hand_side_is_affine(time_derivative.value_text)) return false;
        }
    }
    return true;
}

namespace {

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

// Recursively emits `condition`'s boolean value into `output`, returning the operand name holding it
// (ticket #63 [F2]: ParsedCondition can now be a `.and.`/`.or.`-combination of sub-conditions, not
// just one bare comparison -- see expression_lowering.h's own doc comment). A bare-comparison LEAF
// (no `combinator_opcode`) is emitted into `destination_name` directly -- EXACTLY the single
// `BinaryInstruction` this file always emitted before this ticket, so both call sites below are
// byte-for-byte unchanged for every pre-#63 (single-comparison) OnCondition. A combinator node
// recursively emits its own two sub-conditions (each into its OWN fresh temporary, since only the
// outermost condition's name matters to either call site) and ANDs/ORs them together into
// `destination_name`.
String emit_condition_test(const ParsedCondition &condition, LoweringContext &context,
                           Vector<TickInstruction> &output, const String &destination_name) {
    if (!condition.combinator_opcode.has_value()) {
        String left_operand = context.emit_expression(*condition.left, output, std::nullopt);
        String right_operand = context.emit_expression(*condition.right, output, std::nullopt);
        output.push_back(BinaryInstruction{condition.opcode, destination_name, left_operand, right_operand});
        return destination_name;
    }
    String left_name = emit_condition_test(*condition.left_condition, context, output, context.fresh_temporary());
    String right_name = emit_condition_test(*condition.right_condition, context, output, context.fresh_temporary());
    output.push_back(BinaryInstruction{*condition.combinator_opcode, destination_name, left_name, right_name});
    return destination_name;
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
        // Bake-vs-parameterize (arch §3.1; ticket #65 [F4]'s heterogeneous branch): a name present
        // in `heterogeneous_parameter_values` genuinely varies per neuron in this population, so it
        // is emitted as a `param : dyn` array instead of a baked literal, even if `baked_constants`
        // also happens to carry a (population-uniform-case) value for the same name.
        if (cell_entry.heterogeneous_parameter_values.count(parameter.name)) {
            alloc_directives.push_back(ParamDynamicDirective{parameter.name, "f32"});
            continue;
        }
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
        String condition_name = cell.on_conditions.size() == 1 ? String("spiked") : ("spiked" + std::to_string(condition_index));
        emit_condition_test(parsed_condition, context, tick.detect, condition_name);

        OnConditionActions actions = lower_on_condition_actions(
            on_condition.body, context, regime_index_of, on_entry_assignments_of_regime,
            "cell '" + cell_entry.component_type_name + "'s OnCondition '" + on_condition.test + "'");
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
            String name_suffix = regime.on_conditions.size() == 1 ? regime.name : (regime.name + std::to_string(condition_index));
            String raw_test_name = "test_" + name_suffix;
            emit_condition_test(parsed_condition, context, tick.detect, raw_test_name);

            String fire_name = "fire_" + name_suffix;
            tick.detect.push_back(BinaryInstruction{BinaryOpcode::And, fire_name, is_regime_name, raw_test_name});

            OnConditionActions actions = lower_on_condition_actions(
                *on_condition.body_node, context, regime_index_of, on_entry_assignments_of_regime,
                "cell '" + cell_entry.component_type_name + "'s regime '" + regime.name + "' OnCondition '" +
                on_condition.test_text + "'");
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
