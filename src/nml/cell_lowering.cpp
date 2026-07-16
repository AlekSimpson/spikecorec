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
