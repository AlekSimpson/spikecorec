#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <optional>
#include <unordered_set>
#include <utility>

#include "spikecorec/nml/synapse_lowering.h"
#include "spikecorec/nml/expression_lowering.h"
#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::nml {

namespace {

// Every name a synapse's expressions may legally reference: state
// variables, parameters, properties, plain-value derived variables, and
// requirements (arch §3.1's `Requirement` -- near-universally postsynaptic
// `v`).
std::unordered_set<String> gather_known_names(const SynapseType &synapse) {
    std::unordered_set<String> known_names;
    for (const auto &state_variable : synapse.state_variables) known_names.insert(state_variable.name);
    for (const auto &parameter : synapse.parameters) known_names.insert(parameter.name);
    for (const auto &property : synapse.properties) known_names.insert(property.name);
    for (const auto &derived_variable : synapse.derived_variables) known_names.insert(derived_variable.name);
    for (const auto &requirement : synapse.requirements) known_names.insert(requirement.name);
    return known_names;
}

// The `DerivedVariable` whose `exposure="i"` -- basePointCurrent's universal
// "the total current produced by this ComponentType" contract (IR spec
// §3.5's invariant: every synapse computes its finished current and writes
// it to `network_inputs`). Every real D3 fixture (arch §3.3) declares
// exactly one. Throws if none is declared.
const DerivedVariableDecl &find_current_derived_variable(const SynapseType &synapse) {
    for (const auto &derived_variable : synapse.derived_variables) {
        if (derived_variable.exposure == "i") return derived_variable;
    }
    log::throw_runtime_error(log::logger(),
        "synapse_lowering: '" + synapse.name + "' has no DerivedVariable exposing 'i' "
        "(the finished current basePointCurrent's contract requires)");
}

// Lowers every `DerivedVariable` (in declared order, including the
// `exposure="i"` one -- nothing else in this ticket's scope depends on "i"
// until the caller's own final `network_inputs` write, so it needs no
// special ordering) into `output`, each into a local named after its own
// declared name -- same convention cell_lowering.cpp's DerivedVariable
// handling uses. Throws on a `select`/`reduce` DerivedVariable (Children-
// based sub-mechanism composition -- e.g. a real `blockingPlasticSynapse`'s
// `plasticityFactor`/`blockFactor` -- is out of Phase-1 scope; see
// synapse_lowering.h's header comment).
void lower_all_derived_variables(const SynapseType &synapse, Vector<TickInstruction> &output, LoweringContext &context) {
    for (const auto &derived_variable : synapse.derived_variables) {
        if (!derived_variable.select.empty()) {
            log::throw_runtime_error(log::logger(),
                "synapse_lowering: DerivedVariable '" + derived_variable.name + "' uses a select/reduce "
                "sub-component aggregation, which is out of Phase-1 scope (arch §4.3's NMDA example elides "
                "the equivalent Mg-block/plasticity composition the same way)");
        }
        if (derived_variable.value.empty()) {
            log::throw_runtime_error(log::logger(),
                "synapse_lowering: DerivedVariable '" + derived_variable.name + "' has neither `value` nor `select`/`reduce`");
        }
        context.reset_temporary_counter();
        ExpressionNodePointer value_expression = parse_arithmetic_text(
            derived_variable.value, "DerivedVariable '" + derived_variable.name + "'");
        context.emit_expression(*value_expression, output, derived_variable.name);
    }
}

// Lowers one synapse's `onevent <port> { accedge <var>@edge, <increment> }`
// spike-arrival handler(s) (arch §4.3; IR spec §3.4 `onevent`, §3.3
// `accedge`) into `tick.deliver` (IR spec §3.3: `accedge` is `Sk[edge]+=value`
// for a `peredge` destination). Every real D3 fixture this ticket targets
// (expOneSynapse/expTwoSynapse/alphaCurrentSynapse's `g`/`I`/`J`/`A`/`B`
// bumps) shapes each `OnEvent` body as one or more
// additive self-increment `StateAssignment`s (`var = var + <increment>`);
// anything else (a non-additive assignment, or an `EventOut` inside the
// handler -- e.g. a real `blockingPlasticSynapse`'s `relay` port) is out of
// this ticket's scope and throws.
void lower_on_events(const SynapseType &synapse, TickProgram &tick, LoweringContext &context) {
    for (const auto &on_event : synapse.on_events) {
        Vector<TickInstruction> body;
        for (const auto &child : on_event.body.body) {
            if (child.tag_name != "StateAssignment") {
                log::throw_runtime_error(log::logger(),
                    "synapse_lowering: onevent '" + on_event.port + "' has an unsupported child '" +
                    child.tag_name + "' (only an additive self-increment StateAssignment is supported)");
            }

            String variable_name = get_attribute_value(child, "variable");
            String value_text = get_attribute_value(child, "value");
            String context_for_errors = "onevent '" + on_event.port + "' StateAssignment '" + variable_name + "'";

            context.reset_temporary_counter();
            ExpressionNodePointer value_expression = parse_arithmetic_text(value_text, context_for_errors);

            bool is_additive = value_expression->kind == ExpressionNodeKind::Binary &&
                                value_expression->binary_operator_character == '+';
            bool left_is_self = is_additive && value_expression->left->kind == ExpressionNodeKind::Identifier &&
                                 value_expression->left->text == variable_name;
            bool right_is_self = is_additive && value_expression->right->kind == ExpressionNodeKind::Identifier &&
                                  value_expression->right->text == variable_name;
            if (!left_is_self && !right_is_self) {
                log::throw_runtime_error(log::logger(),
                    "synapse_lowering: " + context_for_errors + " is not a supported additive self-increment "
                    "(`" + variable_name + " = " + variable_name + " + <increment>`)");
            }
            const ExpressionNode &increment_node = left_is_self ? *value_expression->right : *value_expression->left;

            String increment_operand = context.emit_expression(increment_node, body, std::nullopt);
            body.push_back(AccumulateEdgeInstruction{variable_name, EdgeSetReference::CurrentEdge, increment_operand});
        }
        tick.deliver.push_back(OnEventInstruction{on_event.port, std::move(body)});
    }
}

} // namespace

IrProgram lower_synapse_to_ir(const TypeLibraryEntry &synapse_entry) {
    if (synapse_entry.category != TypeLibraryCategory::Synapse || !std::holds_alternative<SynapseType>(synapse_entry.dynamics.flattened)) {
        log::throw_runtime_error(log::logger(),
            "synapse_lowering: '" + synapse_entry.component_type_name + "' is not a Synapse-category TypeLibraryEntry");
    }
    const SynapseType &synapse = std::get<SynapseType>(synapse_entry.dynamics.flattened);

    std::unordered_set<String> known_names = gather_known_names(synapse);
    const DerivedVariableDecl &current_derived_variable = find_current_derived_variable(synapse);

    // ── .alloc (arch §4.3 peredge, uniformly -- every synapse's state now
    // uses per-edge storage regardless of superposability, see this file's
    // header comment; requirements; §3.1 Parameter/Property bake-vs-
    // parameterize, same Phase-1 rule cell_lowering.cpp uses; §4.6
    // expose) ────────────────────────────────────────────────────────────────
    Vector<AllocDirective> alloc_directives;

    for (const auto &requirement : synapse.requirements) {
        alloc_directives.push_back(RequireDirective{requirement.name, "postsynaptic"});
    }

    for (const auto &state_variable : synapse.state_variables) {
        alloc_directives.push_back(PeredgeDirective{state_variable.name});
    }

    auto bake_constant_directive = [&](const String &name) {
        auto baked_value = synapse_entry.baked_constants.find(name);
        std::optional<String> literal_value;
        if (baked_value != synapse_entry.baked_constants.end()) literal_value = format_literal(baked_value->second);
        alloc_directives.push_back(ParamConstantDirective{name, literal_value});
    };
    for (const auto &parameter : synapse.parameters) bake_constant_directive(parameter.name);
    for (const auto &property : synapse.properties) bake_constant_directive(property.name);

    Vector<String> exposed_names_in_order;
    std::unordered_set<String> exposed_names_seen;
    auto add_exposure = [&](const String &exposure_name) {
        if (exposure_name.empty() || exposed_names_seen.count(exposure_name)) return;
        exposed_names_seen.insert(exposure_name);
        exposed_names_in_order.push_back(exposure_name);
    };
    for (const auto &state_variable : synapse.state_variables) add_exposure(state_variable.exposure);
    for (const auto &derived_variable : synapse.derived_variables) add_exposure(derived_variable.exposure);
    for (const auto &exposure : synapse.exposures) add_exposure(exposure.name);
    for (const auto &exposure_name : exposed_names_in_order) alloc_directives.push_back(ExposeDirective{exposure_name});

    // ── .tick @deliver: the onevent arrival handler(s) ───────────────────────
    TickProgram tick;
    UnorderedMap<String, String> no_aliases;
    LoweringContext top_level_context(known_names, no_aliases);
    lower_on_events(synapse, tick, top_level_context);

    // ── .tick @integrate: the finished-current computation, arch §3.5's
    // invariant ("every synapse type ... computes its finished total
    // current with .tick instructions and writes it into network_inputs";
    // a conductance synapse's `g·(erev−v)` needs no special cell-side case
    // -- it is just this type's own DerivedVariable exposing "i") -- always
    // `forall neuron_in { loadedge ...; ... }` (IR spec §4's NMDA example),
    // uniformly for every synapse now (see this file's header comment). Each
    // peredge state variable is `loadedge`'d into a name derived from the
    // variable itself (`edge_<name>`, never a `tN` temp) so it stays valid
    // across however many subsequent statements the forall body's own
    // DerivedVariable computations need (this is also what folds a
    // conductance-based synapse's `require v`-driven `g·(erev−v)` into the
    // exact same generic path a current-based synapse's trivial `i = g`
    // identity takes -- `lower_all_derived_variables` below has no
    // conductance-vs-current special case; it just evaluates whichever
    // expression the type's own `i` DerivedVariable declares), without
    // colliding with their OWN per-statement-reset `tN` counter.
    //
    // A state variable whose `TimeDerivative` matches the recognized
    // linear-decay shape (`detect_linear_decay_shape`, shared with
    // cell_lowering.cpp's direct-mutation case via expression_lowering.h) IS
    // now lowered -- per-edge storage is accumulate-only (arch §4.3:
    // `accedge` is `Sk[edge]+=value`, there is no direct "set" op), so decay
    // is expressed as read-decay-writeback-delta: `loadedge` the current
    // value, `expdecay` (or the 3-instruction non-zero-target form) it to
    // what it should decay to, `accedge` the DIFFERENCE back so the next
    // tick's `loadedge` reconstructs the decayed value. The decayed result is
    // written into the same `edge_<name>` register `peredge_aliases` already
    // points every later reference of this variable at, so it decays BEFORE
    // the DerivedVariable computations below read it (mirroring how the old
    // aggregatable path decayed `g` before contributing it). A
    // `TimeDerivative` that does NOT match the recognized shape is still
    // silently dropped from `.tick` (unlike every other unsupported shape in
    // this file, which throws) -- general per-edge forward-Euler integration
    // for an arbitrary right-hand side is out of this ticket's scope -- but a
    // silent drop is still worth a diagnostic so a future synapse with
    // genuine non-decay dynamics doesn't lose it without at least a
    // build-time signal.
    UnorderedMap<String, String> time_derivative_rhs_of_variable;
    for (const auto &time_derivative : synapse.time_derivatives) {
        time_derivative_rhs_of_variable[time_derivative.variable] = time_derivative.value;
    }

    Vector<TickInstruction> forall_body;
    UnorderedMap<String, String> peredge_aliases;
    for (const auto &state_variable : synapse.state_variables) {
        peredge_aliases[state_variable.name] = "edge_" + state_variable.name;
    }
    LoweringContext forall_context(known_names, peredge_aliases);

    for (const auto &state_variable : synapse.state_variables) {
        String edge_local_name = "edge_" + state_variable.name;

        auto time_derivative_entry = time_derivative_rhs_of_variable.find(state_variable.name);
        std::optional<LinearDecayShape> decay_shape;
        if (time_derivative_entry != time_derivative_rhs_of_variable.end()) {
            ExpressionNodePointer right_hand_side = parse_arithmetic_text(
                time_derivative_entry->second, "TimeDerivative '" + state_variable.name + "'");
            decay_shape = detect_linear_decay_shape(*right_hand_side, state_variable.name);
        }

        if (!decay_shape) {
            forall_body.push_back(LoadEdgeInstruction{edge_local_name, state_variable.name, EdgeSetReference::CurrentEdge});
            if (time_derivative_entry != time_derivative_rhs_of_variable.end()) {
                log::logger().warn(
                    "synapse_lowering: '{}' declares a TimeDerivative for '{}' that isn't a recognized "
                    "linear-decay shape ('(target-state)/tau' or '-state/tau') -- its decay is NOT lowered "
                    "into '.tick' (general per-edge forward-Euler integration for an arbitrary right-hand "
                    "side is out of Phase-1 scope)",
                    synapse_entry.component_type_name, state_variable.name);
            }
            continue;
        }

        String old_value_name = edge_local_name + "_old";
        forall_body.push_back(LoadEdgeInstruction{old_value_name, state_variable.name, EdgeSetReference::CurrentEdge});
        if (decay_shape->target == "0") {
            forall_body.push_back(BinaryInstruction{BinaryOpcode::ExpDecay, edge_local_name, old_value_name, decay_shape->time_constant});
        } else {
            forall_body.push_back(BinaryInstruction{BinaryOpcode::Sub, edge_local_name, old_value_name, decay_shape->target});
            forall_body.push_back(BinaryInstruction{BinaryOpcode::ExpDecay, edge_local_name, edge_local_name, decay_shape->time_constant});
            forall_body.push_back(BinaryInstruction{BinaryOpcode::Add, edge_local_name, edge_local_name, decay_shape->target});
        }
        String delta_name = edge_local_name + "_delta";
        forall_body.push_back(BinaryInstruction{BinaryOpcode::Sub, delta_name, edge_local_name, old_value_name});
        forall_body.push_back(AccumulateEdgeInstruction{state_variable.name, EdgeSetReference::CurrentEdge, delta_name});
    }

    lower_all_derived_variables(synapse, forall_body, forall_context);
    forall_body.push_back(BinaryInstruction{BinaryOpcode::Add, "network_inputs", "network_inputs", current_derived_variable.name});

    tick.integrate.push_back(ForAllInstruction{EdgeSetReference::NeuronIn, std::move(forall_body)});

    IrProgram program;
    program.component_type_name = synapse_entry.component_type_name;
    program.alloc = std::move(alloc_directives);
    program.tick = std::move(tick);
    // closed_form_advanceable (ticket #62 [F1]) stays at its default (false) -- the active-set
    // multi-tick skip is a per-neuron/cell concept (arch §4.1), not applicable to a per-edge synapse.
    return program;
}

Vector<IrProgram> lower_all_synapse_types_to_ir(const ModelSpecification &model) {
    Vector<IrProgram> programs;
    for (const auto &entry : model.type_library) {
        if (entry.category != TypeLibraryCategory::Synapse) continue;
        programs.push_back(lower_synapse_to_ir(entry));
    }
    return programs;
}

} // namespace spikecorec::nml
