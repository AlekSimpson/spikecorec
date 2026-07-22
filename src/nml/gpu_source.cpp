#include "spikecorec/nml/gpu_source.h"

#include <optional>
#include <sstream>
#include <unordered_set>
#include <variant>

#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::nml {

namespace {

// ── small local helpers ──────────────────────────────────────────────────

template <class... Alternatives>
struct Overloaded : Alternatives... {
    using Alternatives::operator()...;
};
template <class... Alternatives>
Overloaded(Alternatives...) -> Overloaded<Alternatives...>;

void fail(const String &message) {
    log::throw_runtime_error(log::logger(), "gpu_source: " + message);
}

String indent(int level) {
    return String(static_cast<usize>(level) * 4, ' ');
}

bool is_numeric_literal(const String &token) {
    if (token.empty()) return false;
    usize index = 0;
    if (token[0] == '-') index = 1;
    if (index >= token.length()) return false;
    bool has_digit = false;
    bool has_dot = false;
    for (; index < token.length(); ++index) {
        char character = token[index];
        if (character == '.' && !has_dot) {
            has_dot = true;
            continue;
        }
        if (character < '0' || character > '9') return false;
        has_digit = true;
    }
    return has_digit;
}

enum class Backend { Msl, Cuda };

String dtype_to_backend_type(const String &dtype, Backend backend) {
    if (dtype == "f32") return "float";
    if (dtype == "s32") return "int";
    if (dtype == "s64") return backend == Backend::Msl ? "long" : "long long";
    if (dtype == "u32") return backend == Backend::Msl ? "uint" : "unsigned int";
    if (dtype == "bool") return "bool";
    fail("unsupported .alloc dtype '" + dtype + "'");
    return "";
}

// index type used for the row/other-node casts in the loadedge/accedge Sk index formula --
// matches whichever 64-bit integer keyword this backend uses elsewhere (`long` on Metal,
// `long long` on CUDA, since plain `long` is only 32 bits on some CUDA host platforms).
String index_cast_type(Backend backend) {
    return backend == Backend::Msl ? "long" : "long long";
}

// ── `.alloc` indexing ─────────────────────────────────────────────────────

enum class AllocCategory {
    ParamConstantBaked,
    ParamConstantBare,
    ParamDynamic,
    State,
    Accum,
    Peredge,
    Regime,
    Expose,
    Require
};

struct AllocInfo {
    AllocCategory category;
    String dtype;          // meaningful for array-shaped categories
    String literal_value;  // meaningful only for ParamConstantBaked
};

struct AllocIndex {
    UnorderedMap<String, AllocInfo> by_name;
    Vector<String> peredge_names_in_order;
};

AllocIndex build_alloc_index(const Vector<AllocDirective> &alloc) {
    AllocIndex index;
    for (const auto &directive : alloc) {
        std::visit(Overloaded{
            [&](const ParamConstantDirective &param) {
                if (param.literal_value.has_value()) {
                    index.by_name[param.name] = AllocInfo{AllocCategory::ParamConstantBaked, "f32", *param.literal_value};
                } else {
                    index.by_name[param.name] = AllocInfo{AllocCategory::ParamConstantBare, "f32", ""};
                }
            },
            [&](const ParamDynamicDirective &param) {
                index.by_name[param.name] = AllocInfo{AllocCategory::ParamDynamic, param.dtype, ""};
            },
            [&](const StateDirective &state) {
                index.by_name[state.name] = AllocInfo{AllocCategory::State, state.dtype, ""};
            },
            [&](const AccumDirective &accum) {
                index.by_name[accum.name] = AllocInfo{AllocCategory::Accum, accum.dtype, ""};
            },
            [&](const PeredgeDirective &peredge) {
                index.by_name[peredge.name] = AllocInfo{AllocCategory::Peredge, "f32", ""};
                index.peredge_names_in_order.push_back(peredge.name);
            },
            [&](const RegimeDirective &regime) {
                index.by_name[regime.name] = AllocInfo{AllocCategory::Regime, "s32", ""};
            },
            // `Expose`/`Require` are read/observe-only categories, never real storage of their own
            // (arch §3.1's `Exposure`: "no behavior by itself"; `Requirement`: "a binding... storage
            // lives with the owner"). A name already registered under a storage category above
            // (State/Accum/Peredge/Regime/ParamDynamic/ParamConstant*) in THIS SAME `.alloc` -- true
            // of every real self-exposing StateVariable, e.g. `<StateVariable name="g" .../
            // exposure="g"/>`, which synapse_lowering.cpp always emits as BOTH a `PeredgeDirective`
            // AND (later, since exposures are appended last) an `ExposeDirective` for the identical
            // name -- must keep its real category; only an as-yet-unseen name gets the Expose/
            // Require fallback. Without this, `.alloc`'s own declaration order (storage directives
            // before exposures, for a synapse) would let a later ExposeDirective silently overwrite
            // an earlier PeredgeDirective's entry (ticket #131 first surfaced this: nothing before it
            // ever compiled a REAL, self-exposing conductance-based synapse's GPU source).
            [&](const ExposeDirective &expose) {
                if (index.by_name.count(expose.name) > 0) return;
                index.by_name[expose.name] = AllocInfo{AllocCategory::Expose, "f32", ""};
            },
            [&](const RequireDirective &require) {
                if (index.by_name.count(require.name) > 0) return;
                index.by_name[require.name] = AllocInfo{AllocCategory::Require, "f32", ""};
            },
        }, directive);
    }
    return index;
}

// ── structural scans over a `.tick` instruction list ─────────────────────
//
// These walk `.tick` instructions to collect bookkeeping this ticket's calling convention needs
// (which reserved/`.alloc` names a function touches, whether it needs RNG state, which ports it
// emits on) -- symbol-table-style bookkeeping a compiler always needs before it can emit a function
// signature, not semantic analysis of the dynamics themselves (ir_spec.md line 8's "no analysis"
// is about not reasoning about what the `.tick` code MEANS; this is just seeing what NAMES appear).

// Recurses through `if`/`forall`/`onevent` bodies, handing every leaf instruction (never a control
// construct itself) to `visitor`, in program order.
template <typename Visitor>
void for_each_leaf_instruction(const Vector<TickInstruction> &instructions, const Visitor &visitor) {
    for (const auto &instruction : instructions) {
        std::visit(Overloaded{
            [&](const IfInstruction &control) {
                for_each_leaf_instruction(control.then_body, visitor);
                for (const auto &branch : control.else_if_branches) {
                    for_each_leaf_instruction(branch.body, visitor);
                }
                if (control.else_body.has_value()) {
                    for_each_leaf_instruction(*control.else_body, visitor);
                }
            },
            [&](const ForAllInstruction &control) { for_each_leaf_instruction(control.body, visitor); },
            [&](const OnEventInstruction &control) { for_each_leaf_instruction(control.body, visitor); },
            [&](const auto &leaf) { visitor(leaf); },
        }, instruction.operation);
    }
}

bool contains_random(const Vector<TickInstruction> &instructions) {
    bool found = false;
    for_each_leaf_instruction(instructions, Overloaded{
        [&](const RandomInstruction &) { found = true; },
        [&](const auto &) {},
    });
    return found;
}

void collect_emit_ports(const Vector<TickInstruction> &instructions, Vector<String> &ports_in_order,
                         unordered_set<String> &seen) {
    for_each_leaf_instruction(instructions, Overloaded{
        [&](const EmitInstruction &emit) {
            if (seen.insert(emit.port_name).second) ports_in_order.push_back(emit.port_name);
        },
        [&](const auto &) {},
    });
}

void collect_edge_variable_names(const Vector<TickInstruction> &instructions, unordered_set<String> &names) {
    for_each_leaf_instruction(instructions, Overloaded{
        [&](const LoadEdgeInstruction &load) { names.insert(load.variable_name); },
        [&](const AccumulateEdgeInstruction &accumulate) { names.insert(accumulate.variable_name); },
        [&](const auto &) {},
    });
}

bool contains_whole_set_edge_op(const Vector<TickInstruction> &instructions) {
    bool found = false;
    for_each_leaf_instruction(instructions, Overloaded{
        [&](const LoadEdgeInstruction &load) { found = found || load.edge_reference != EdgeSetReference::CurrentEdge; },
        [&](const AccumulateEdgeInstruction &accumulate) {
            found = found || accumulate.edge_reference != EdgeSetReference::CurrentEdge;
        },
        [&](const auto &) {},
    });
    return found;
}

// Unlike for_each_leaf_instruction, this one is aware of ForAllInstruction itself (needed to decide
// whether a function needs the k^2-tree walk block at all) -- it does not descend into a forall's
// own body (irrelevant once one is found) but does descend into if/onevent bodies looking for one.
bool contains_forall(const Vector<TickInstruction> &instructions) {
    for (const auto &instruction : instructions) {
        bool found = std::visit(Overloaded{
            [](const ForAllInstruction &) { return true; },
            [](const IfInstruction &control) {
                if (contains_forall(control.then_body)) return true;
                for (const auto &branch : control.else_if_branches) {
                    if (contains_forall(branch.body)) return true;
                }
                if (control.else_body.has_value() && contains_forall(*control.else_body)) return true;
                return false;
            },
            [](const OnEventInstruction &control) { return contains_forall(control.body); },
            [](const auto &) { return false; },
        }, instruction.operation);
        if (found) return true;
    }
    return false;
}

// Whether this instruction list contains a `loadedge` anywhere (inside a forall body, or directly
// inside an onevent body -- `emit_loadedge` accepts `@edge` in either lexical context). A
// `loadedge` always needs the shared-basis reconstruction (`U`/`V`/`rank_float4_stride`/
// `coefficients_<name>`), independent of whether the k^2-tree WALK block is needed (an onevent
// body's `loadedge` already has its one, specific edge given directly as parameters -- no walk --
// but still needs the basis to reconstruct that edge's value).
bool contains_loadedge(const Vector<TickInstruction> &instructions) {
    bool found = false;
    for_each_leaf_instruction(instructions, Overloaded{
        [&](const LoadEdgeInstruction &) { found = true; },
        [&](const auto &) {},
    });
    return found;
}

// ── operand scan: which reserved/`.alloc` names a function body touches, plus the ordered,
//    typed list of local (`t0,t1,...` / compare-destination) variables it needs hoisted ────────

enum class ProducedType { Float, Bool };

struct LocalDecl {
    String name;
    bool is_bool;
};

struct ScanResult {
    unordered_set<String> alloc_names_referenced;
    unordered_set<String> reserved_names_referenced; // subset of {"dt", "tick", "network_inputs"}
    Vector<LocalDecl> locals_in_order;
};

// Classifies one operand occurrence and records it into `result`/`locals_seen` accordingly.
// `produced_type_if_destination` is set only when this occurrence is an instruction's destination
// (the one place a fresh local can be declared); reading a name that isn't reserved, isn't a
// `.alloc` name, isn't a numeric literal, and was never declared as a destination is a genuine
// error in the IR (an undeclared local read before assignment).
void record_operand(const String &name, std::optional<ProducedType> produced_type_if_destination,
                     const AllocIndex &alloc_index, ScanResult &result, unordered_set<String> &locals_seen) {
    if (name == "dt" || name == "tick" || name == "network_inputs") {
        result.reserved_names_referenced.insert(name);
        return;
    }
    if (is_numeric_literal(name)) return;

    auto found = alloc_index.by_name.find(name);
    if (found != alloc_index.by_name.end()) {
        if (found->second.category == AllocCategory::Peredge) {
            fail("peredge quantity '" + name + "' used as a plain operand -- only loadedge/accedge may reference a peredge variable");
        }
        result.alloc_names_referenced.insert(name);
        return;
    }

    if (locals_seen.count(name) > 0) return;
    if (!produced_type_if_destination.has_value()) {
        fail("local operand '" + name + "' read before being assigned (not a .alloc name, reserved name, or numeric literal)");
    }
    locals_seen.insert(name);
    result.locals_in_order.push_back(LocalDecl{name, *produced_type_if_destination == ProducedType::Bool});
}

bool binary_opcode_is_bool(BinaryOpcode opcode) {
    switch (opcode) {
        case BinaryOpcode::Gt:
        case BinaryOpcode::Lt:
        case BinaryOpcode::Ge:
        case BinaryOpcode::Le:
        case BinaryOpcode::Eq:
        case BinaryOpcode::Ne:
        case BinaryOpcode::And:
        case BinaryOpcode::Or:
            return true;
        default:
            return false;
    }
}

void scan_operands(const Vector<TickInstruction> &instructions, const AllocIndex &alloc_index, ScanResult &result,
                    unordered_set<String> &locals_seen) {
    for (const auto &instruction : instructions) {
        std::visit(Overloaded{
            [&](const BinaryInstruction &leaf) {
                record_operand(leaf.operand_a, std::nullopt, alloc_index, result, locals_seen);
                record_operand(leaf.operand_b, std::nullopt, alloc_index, result, locals_seen);
                // `expdecay dst,a,tau` -> `a * exp(-dt / tau)` (ir_spec.md §3.3): the snippet
                // template references `dt` implicitly -- it's not one of this instruction's own
                // stored operands, so it must be recorded here or the emitted function ends up
                // referencing an undeclared `dt` (see emit_binary's ExpDecay case).
                if (leaf.opcode == BinaryOpcode::ExpDecay) result.reserved_names_referenced.insert("dt");
                ProducedType type = binary_opcode_is_bool(leaf.opcode) ? ProducedType::Bool : ProducedType::Float;
                record_operand(leaf.destination, type, alloc_index, result, locals_seen);
            },
            [&](const UnaryInstruction &leaf) {
                record_operand(leaf.operand_a, std::nullopt, alloc_index, result, locals_seen);
                ProducedType type = leaf.opcode == UnaryOpcode::Not ? ProducedType::Bool : ProducedType::Float;
                record_operand(leaf.destination, type, alloc_index, result, locals_seen);
            },
            [&](const FusedMultiplyAddInstruction &leaf) {
                record_operand(leaf.operand_a, std::nullopt, alloc_index, result, locals_seen);
                record_operand(leaf.operand_b, std::nullopt, alloc_index, result, locals_seen);
                record_operand(leaf.operand_c, std::nullopt, alloc_index, result, locals_seen);
                record_operand(leaf.destination, ProducedType::Float, alloc_index, result, locals_seen);
            },
            [&](const RandomInstruction &leaf) {
                record_operand(leaf.destination, ProducedType::Float, alloc_index, result, locals_seen);
            },
            [&](const MoveInstruction &leaf) {
                record_operand(leaf.source, std::nullopt, alloc_index, result, locals_seen);
                ProducedType type = ProducedType::Float;
                for (const auto &local : result.locals_in_order) {
                    if (local.name == leaf.source && local.is_bool) {
                        type = ProducedType::Bool;
                        break;
                    }
                }
                record_operand(leaf.destination, type, alloc_index, result, locals_seen);
            },
            [&](const LoadEdgeInstruction &leaf) {
                // leaf.variable_name is edge-scoped (loadedge/accedge only) -- collected separately
                // by collect_edge_variable_names, never through the generic operand path here.
                record_operand(leaf.destination, ProducedType::Float, alloc_index, result, locals_seen);
            },
            [&](const AccumulateEdgeInstruction &leaf) {
                record_operand(leaf.value, std::nullopt, alloc_index, result, locals_seen);
            },
            [&](const EmitInstruction &) { /* port_name is not an operand */ },
            [&](const SetRegimeInstruction &leaf) {
                record_operand(leaf.value, std::nullopt, alloc_index, result, locals_seen);
                record_operand(leaf.regime_name, std::nullopt, alloc_index, result, locals_seen);
            },
            [&](const IfInstruction &control) {
                record_operand(control.condition, std::nullopt, alloc_index, result, locals_seen);
                scan_operands(control.then_body, alloc_index, result, locals_seen);
                for (const auto &branch : control.else_if_branches) {
                    record_operand(branch.condition, std::nullopt, alloc_index, result, locals_seen);
                    scan_operands(branch.body, alloc_index, result, locals_seen);
                }
                if (control.else_body.has_value()) scan_operands(*control.else_body, alloc_index, result, locals_seen);
            },
            [&](const ForAllInstruction &control) { scan_operands(control.body, alloc_index, result, locals_seen); },
            [&](const OnEventInstruction &control) { scan_operands(control.body, alloc_index, result, locals_seen); },
        }, instruction.operation);
    }
}

// ── operand resolution (which function kind is currently being emitted into) ─────────────────

enum class FunctionKind { PerNeuron, Deliver };

// Resolves one `.tick` operand name to the MSL/CUDA expression that reads/writes it -- identical
// text in both backends (the divergence in this ticket lives in leaf-op snippets and the
// loadedge/accedge float4 math, not in operand resolution).
String resolve_operand(const String &name, FunctionKind kind, const AllocIndex &alloc_index) {
    if (name == "dt") return "dt";
    if (name == "tick") return "tick";
    if (name == "network_inputs") {
        return kind == FunctionKind::PerNeuron ? "network_inputs[neuron_index]" : "network_inputs[target_node]";
    }
    if (is_numeric_literal(name)) return name;

    auto found = alloc_index.by_name.find(name);
    if (found != alloc_index.by_name.end()) {
        switch (found->second.category) {
            case AllocCategory::ParamConstantBaked:
            case AllocCategory::ParamConstantBare:
                return name; // scalar -- same reading in either function kind
            case AllocCategory::Peredge:
                fail("peredge quantity '" + name + "' cannot be resolved as a plain operand");
                return "";
            case AllocCategory::ParamDynamic:
            case AllocCategory::State:
            case AllocCategory::Accum:
            case AllocCategory::Regime:
            case AllocCategory::Expose:
                if (kind != FunctionKind::PerNeuron) {
                    fail("per-neuron array quantity '" + name + "' referenced as a plain operand inside a "
                         "deliver/onevent function -- which endpoint (source/target) should index it is "
                         "ambiguous and unsupported by this ticket's lowering");
                }
                return name + "[neuron_index]";
            case AllocCategory::Require:
                if (kind == FunctionKind::PerNeuron) return name + "[neuron_index]";
                // ticket #131: unlike the per-neuron-array categories above, a `require <name> from
                // postsynaptic` binding is UNAMBIGUOUS inside a Deliver-kind (one-thread-per-edge)
                // function -- the postsynaptic neuron IS target_node, whichever endpoint's array this
                // is. No existing onevent body references one (every real fixture's onevent body only
                // ever does `accedge`), so this only WIDENS what a Deliver-kind function can express;
                // it is exactly what a synapse's own edge-parallel `.tick.integrate` gather (this
                // ticket's `<Type>_integrate_edges` function, generated the same way an onevent
                // function is -- see lower_synapse_ir_program_to_gpu_source) needs to read the
                // postsynaptic cell's own exposed state (e.g. `g*(erev-v)`'s `v`).
                return name + "[target_node]";
        }
    }

    return name; // a local variable (t0,t1,... or a compare/boolean destination), verbatim
}

// ── generated-function parameter list ─────────────────────────────────────

enum class ParamKind { Scalar, MutableArray, ConstArray };

struct ParamDescriptor {
    String msl_type;
    String cuda_type;
    String name;
    ParamKind kind;
};

String render_msl_param(const ParamDescriptor &parameter, int buffer_index) {
    String attribute = " [[ buffer(" + std::to_string(buffer_index) + ") ]]";
    switch (parameter.kind) {
        case ParamKind::Scalar: return "    constant " + parameter.msl_type + " &" + parameter.name + attribute;
        case ParamKind::MutableArray: return "    device " + parameter.msl_type + " *" + parameter.name + attribute;
        case ParamKind::ConstArray: return "    const device " + parameter.msl_type + " *" + parameter.name + attribute;
    }
    return "";
}

String render_cuda_param(const ParamDescriptor &parameter) {
    switch (parameter.kind) {
        case ParamKind::Scalar: return parameter.cuda_type + " " + parameter.name;
        case ParamKind::MutableArray: return parameter.cuda_type + " *" + parameter.name;
        case ParamKind::ConstArray: return "const " + parameter.cuda_type + " *" + parameter.name;
    }
    return "";
}

// Builds every parameter this function needs EXCEPT the trailing dispatch-bound parameter(s)
// (`neuron_count` for a per-neuron function; the edge-event arrays + `event_count` for a deliver
// function) -- callers append those themselves, since they differ by function kind.
//
// `needs_tree_walk_block` and `needs_shared_basis_block` are deliberately separate flags (rather
// than one combined "edge-walk" flag): a `forall`/whole-set edge op needs BOTH the k^2-tree walk
// buffers (to discover the neighbor at all) and the shared basis (if it also does a `loadedge`);
// an `onevent`/deliver body's `loadedge` already has its one, specific edge given directly as
// parameters (`source_node`/`target_node`/`edge_slot`, no walk needed) but still needs the shared
// basis to reconstruct that edge's value -- so it sets `needs_shared_basis_block` alone. Both
// flags being true together (the per-neuron function's own case whenever it uses `forall`) is what
// this function still emits identically to before this distinction existed.
Vector<ParamDescriptor> build_common_parameters(const ScanResult &scan,
                                                 const unordered_set<String> &edge_variable_names_referenced,
                                                 bool needs_tree_walk_block, bool needs_shared_basis_block,
                                                 bool needs_rng, const Vector<String> &emit_ports,
                                                 const AllocIndex &alloc_index,
                                                 const Vector<AllocDirective> &alloc_in_order) {
    Vector<ParamDescriptor> parameters;

    if (scan.reserved_names_referenced.count("dt")) parameters.push_back({"float", "float", "dt", ParamKind::Scalar});
    if (scan.reserved_names_referenced.count("tick")) {
        parameters.push_back({"long", "long long", "tick", ParamKind::Scalar});
    }
    if (scan.reserved_names_referenced.count("network_inputs")) {
        parameters.push_back({"float", "float", "network_inputs", ParamKind::MutableArray});
    }

    // A directly-stored StateVariable whose own `exposure=` name equals its own name (arch §3.2's
    // most common shape -- every one of ticket #63's own real nonlinear cells has this, e.g.
    // `<StateVariable name="v" ... exposure="v"/>`) produces BOTH a `state v` AND an `expose v`
    // `.alloc` directive for the exact same underlying quantity (allocator.h's own
    // derived_exposure_scratch_buffers doc comment already documents this precedent: "a directly
    // stored StateVariable's own exposure already has a slot and needs no scratch"). Both directives
    // resolve to the identical operand text (`name + "[neuron_index]"`, resolve_operand) either way,
    // so this set exists purely to stop `alloc_in_order`'s per-directive loop below from declaring
    // TWO kernel parameters of the same name (a real MSL/CUDA compile error -- "redefinition of
    // parameter") for what is really one buffer -- the first directive to reference a given name (in
    // `.alloc` declaration order) wins, matching resolve_operand's own AllocIndex lookup (a plain
    // `unordered_map`, so whichever directive is inserted LAST there actually determines the category
    // resolve_operand sees -- irrelevant here, both categories resolve identically).
    unordered_set<String> parameter_names_already_added;
    auto add_parameter_once = [&](ParamDescriptor descriptor) {
        if (!parameter_names_already_added.insert(descriptor.name).second) return;
        parameters.push_back(std::move(descriptor));
    };

    for (const auto &directive : alloc_in_order) {
        std::visit(Overloaded{
            [&](const ParamConstantDirective &param) {
                if (scan.alloc_names_referenced.count(param.name) == 0) return;
                if (param.literal_value.has_value()) return; // baked as a local const, not a parameter
                add_parameter_once({"float", "float", param.name, ParamKind::Scalar});
            },
            [&](const ParamDynamicDirective &param) {
                if (scan.alloc_names_referenced.count(param.name) == 0) return;
                add_parameter_once({dtype_to_backend_type(param.dtype, Backend::Msl),
                                     dtype_to_backend_type(param.dtype, Backend::Cuda), param.name,
                                     ParamKind::MutableArray});
            },
            [&](const StateDirective &state) {
                if (scan.alloc_names_referenced.count(state.name) == 0) return;
                add_parameter_once({dtype_to_backend_type(state.dtype, Backend::Msl),
                                     dtype_to_backend_type(state.dtype, Backend::Cuda), state.name,
                                     ParamKind::MutableArray});
            },
            [&](const AccumDirective &accum) {
                // accum can be touched two ways: as a plain per-neuron operand (expOne's own
                // @integrate use) or only through accedge's variable_name (expOne's onevent use,
                // which never runs it through record_operand/resolve_operand) -- either needs the
                // underlying buffer declared.
                bool referenced = scan.alloc_names_referenced.count(accum.name) > 0 ||
                                   edge_variable_names_referenced.count(accum.name) > 0;
                if (!referenced) return;
                add_parameter_once({dtype_to_backend_type(accum.dtype, Backend::Msl),
                                     dtype_to_backend_type(accum.dtype, Backend::Cuda), accum.name,
                                     ParamKind::MutableArray});
            },
            [&](const PeredgeDirective &) { /* handled below, alongside the edge-walk block */ },
            [&](const RegimeDirective &regime) {
                if (scan.alloc_names_referenced.count(regime.name) == 0) return;
                add_parameter_once({"int", "int", regime.name, ParamKind::MutableArray});
            },
            [&](const ExposeDirective &expose) {
                if (scan.alloc_names_referenced.count(expose.name) == 0) return;
                add_parameter_once({"float", "float", expose.name, ParamKind::MutableArray});
            },
            [&](const RequireDirective &require) {
                if (scan.alloc_names_referenced.count(require.name) == 0) return;
                add_parameter_once({"float", "float", require.name, ParamKind::MutableArray});
            },
        }, directive);
    }

    if (needs_rng) parameters.push_back({"uint", "unsigned int", "rng_state", ParamKind::MutableArray});
    for (const String &port : emit_ports) {
        parameters.push_back({"bool", "bool", "emit_" + port, ParamKind::MutableArray});
    }

    Vector<String> peredge_names_touched;
    for (const String &name : alloc_index.peredge_names_in_order) {
        if (edge_variable_names_referenced.count(name)) peredge_names_touched.push_back(name);
    }

    // The k^2-tree walk buffers: only needed to DISCOVER a neighbor at all (`forall`/a whole-set
    // edge op) -- an onevent body's `loadedge`/`accedge` already has its one, specific edge given
    // directly as parameters and never walks.
    if (needs_tree_walk_block) {
        parameters.push_back({"uint", "unsigned int", "internal_node_words", ParamKind::ConstArray});
        parameters.push_back({"uint", "unsigned int", "leaf_node_words", ParamKind::ConstArray});
        parameters.push_back({"uint", "unsigned int", "rank_superblock_table", ParamKind::ConstArray});
        parameters.push_back({"ushort", "unsigned short", "rank_subblock_table", ParamKind::ConstArray});
        parameters.push_back({"int", "int", "branching_factor", ParamKind::Scalar});
        parameters.push_back({"int", "int", "superblock_size_words", ParamKind::Scalar});
        parameters.push_back({"int", "int", "padded_node_count", ParamKind::Scalar});
        parameters.push_back({"int", "int", "tree_height", ParamKind::Scalar});
        parameters.push_back({"int", "int", "internal_bit_count", ParamKind::Scalar});
        parameters.push_back({"long", "long long", "node_count", ParamKind::Scalar});
    }

    // Sk's index stride + the per-touched-name sparse delta buffer itself: needed by BOTH
    // `loadedge` (the Sk-overlay term) and `accedge` (Sk[row,slot] += value) on a peredge var,
    // whether or not the shared basis below is also needed (weight_matrix.h's
    // accumulate_edge_delta doc: "Cheap and local -- no basis touched").
    if (needs_tree_walk_block || !peredge_names_touched.empty()) {
        parameters.push_back({"long", "long long", "max_neighbor_count", ParamKind::Scalar});
    }

    // The shared basis: needed whenever this function reconstructs a peredge value at all --
    // `forall` bodies that also `loadedge` (NMDA's `@integrate`), or an onevent body that
    // `loadedge`s the edge it already knows (no walk, but still needs U/V/Ck to reconstruct it).
    if (needs_shared_basis_block) {
        parameters.push_back({"long", "long long", "rank_float4_stride", ParamKind::Scalar});
        parameters.push_back({"float4", "float4", "U", ParamKind::ConstArray});
        parameters.push_back({"float4", "float4", "V", ParamKind::ConstArray});
    }

    for (const String &name : peredge_names_touched) {
        if (needs_shared_basis_block) parameters.push_back({"float", "float", "coefficients_" + name, ParamKind::ConstArray});
        parameters.push_back({"float", "float", "sparse_delta_" + name, ParamKind::MutableArray});
    }

    return parameters;
}

// ── `.tick` instruction emission (the actual template substitution) ──────────────────────────
//
// `@edge` resolves to a (row_node, other_node, slot) triple depending on lexical context --
// see gpu_source.h's doc comment. `row_node`/`other_node`/`slot` are always already-resolved MSL/
// CUDA expressions (a variable name), never re-resolved through resolve_operand (they're this
// ticket's own invented loop/parameter names, not `.tick` operand text).
struct EdgeContext {
    String row_node_expr;
    String other_node_expr;
    String slot_expr;
};

struct EmitContext {
    FunctionKind kind;
    const AllocIndex *alloc_index;
    std::optional<EdgeContext> edge_context;
    int next_forall_id = 0;
};

void emit_instructions(const Vector<TickInstruction> &instructions, Backend backend, EmitContext &context,
                        int indent_level, ostringstream &output);

void emit_binary(const BinaryInstruction &leaf, Backend backend, EmitContext &context, int indent_level,
                  ostringstream &output) {
    // ticket #131: `add network_inputs, network_inputs, <value>` (synapse_lowering.cpp's own sole
    // network_inputs-writing shape) inside an edge-parallel (Deliver-kind) function -- a synapse's
    // own `_integrate_edges`, the only caller that ever reaches this from that function kind --
    // dispatches one GPU thread PER EDGE, so multiple threads may target the SAME postsynaptic
    // neuron's `network_inputs` slot concurrently whenever it has more than one incoming edge (an
    // ordinary, common topology) -- a plain, non-atomic read-modify-write would race and silently
    // drop a contribution. A per-neuron function never shares `network_inputs[neuron_index]`
    // across threads (one thread per neuron), so it keeps the plain form below unchanged. Mirrors
    // the fixed propagate kernel's own identical atomic-add pattern for network_inputs, for the
    // exact same reason (master_kernel.cpp's `build_propagate_kernel_gpu_source`).
    if (leaf.opcode == BinaryOpcode::Add && leaf.destination == "network_inputs" &&
        leaf.operand_a == "network_inputs" && context.kind != FunctionKind::PerNeuron) {
        String value = resolve_operand(leaf.operand_b, context.kind, *context.alloc_index);
        if (backend == Backend::Msl) {
            output << indent(indent_level) << "{\n";
            output << indent(indent_level + 1)
                   << "device atomic_float *network_inputs_atomic_slot = (device atomic_float *)(network_inputs + "
                      "target_node);\n";
            output << indent(indent_level + 1) << "atomic_fetch_add_explicit(network_inputs_atomic_slot, " << value
                   << ", memory_order_relaxed);\n";
            output << indent(indent_level) << "}\n";
        } else {
            output << indent(indent_level) << "atomicAdd(&network_inputs[target_node], " << value << ");\n";
        }
        return;
    }

    String destination = resolve_operand(leaf.destination, context.kind, *context.alloc_index);
    String a = resolve_operand(leaf.operand_a, context.kind, *context.alloc_index);
    String b = resolve_operand(leaf.operand_b, context.kind, *context.alloc_index);
    bool msl = backend == Backend::Msl;
    String expression;
    switch (leaf.opcode) {
        case BinaryOpcode::Add: expression = a + " + " + b; break;
        case BinaryOpcode::Sub: expression = a + " - " + b; break;
        case BinaryOpcode::Mul: expression = a + " * " + b; break;
        case BinaryOpcode::Div: expression = a + " / " + b; break;
        case BinaryOpcode::Mod: expression = (msl ? "fmod(" : "fmodf(") + a + ", " + b + ")"; break;
        case BinaryOpcode::Pow: expression = (msl ? "pow(" : "powf(") + a + ", " + b + ")"; break;
        case BinaryOpcode::Min: expression = (msl ? "min(" : "fminf(") + a + ", " + b + ")"; break;
        case BinaryOpcode::Max: expression = (msl ? "max(" : "fmaxf(") + a + ", " + b + ")"; break;
        case BinaryOpcode::ExpDecay: {
            String dt = resolve_operand("dt", context.kind, *context.alloc_index);
            expression = a + " * " + (msl ? "exp(-" : "expf(-") + dt + " / " + b + ")";
            break;
        }
        case BinaryOpcode::Gt: expression = a + " > " + b; break;
        case BinaryOpcode::Lt: expression = a + " < " + b; break;
        case BinaryOpcode::Ge: expression = a + " >= " + b; break;
        case BinaryOpcode::Le: expression = a + " <= " + b; break;
        case BinaryOpcode::Eq: expression = a + " == " + b; break;
        case BinaryOpcode::Ne: expression = a + " != " + b; break;
        case BinaryOpcode::And: expression = a + " && " + b; break;
        case BinaryOpcode::Or: expression = a + " || " + b; break;
    }
    output << indent(indent_level) << destination << " = " << expression << ";\n";
}

void emit_unary(const UnaryInstruction &leaf, Backend backend, EmitContext &context, int indent_level,
                 ostringstream &output) {
    String destination = resolve_operand(leaf.destination, context.kind, *context.alloc_index);
    String a = resolve_operand(leaf.operand_a, context.kind, *context.alloc_index);
    bool msl = backend == Backend::Msl;
    String expression;
    switch (leaf.opcode) {
        case UnaryOpcode::Neg: expression = "-" + a; break;
        case UnaryOpcode::Exp: expression = (msl ? "exp(" : "expf(") + a + ")"; break;
        case UnaryOpcode::Log: expression = (msl ? "log(" : "logf(") + a + ")"; break;
        case UnaryOpcode::Sqrt: expression = (msl ? "sqrt(" : "sqrtf(") + a + ")"; break;
        case UnaryOpcode::Abs: expression = (msl ? "fabs(" : "fabsf(") + a + ")"; break;
        case UnaryOpcode::Floor: expression = (msl ? "floor(" : "floorf(") + a + ")"; break;
        case UnaryOpcode::Ceil: expression = (msl ? "ceil(" : "ceilf(") + a + ")"; break;
        case UnaryOpcode::Sin: expression = (msl ? "sin(" : "sinf(") + a + ")"; break;
        case UnaryOpcode::Cos: expression = (msl ? "cos(" : "cosf(") + a + ")"; break;
        case UnaryOpcode::Tan: expression = (msl ? "tan(" : "tanf(") + a + ")"; break;
        case UnaryOpcode::Sinh: expression = (msl ? "sinh(" : "sinhf(") + a + ")"; break;
        case UnaryOpcode::Cosh: expression = (msl ? "cosh(" : "coshf(") + a + ")"; break;
        case UnaryOpcode::Tanh: expression = (msl ? "tanh(" : "tanhf(") + a + ")"; break;
        case UnaryOpcode::Not: expression = "!" + a; break;
    }
    output << indent(indent_level) << destination << " = " << expression << ";\n";
}

void emit_fma(const FusedMultiplyAddInstruction &leaf, Backend backend, EmitContext &context, int indent_level,
              ostringstream &output) {
    String destination = resolve_operand(leaf.destination, context.kind, *context.alloc_index);
    String a = resolve_operand(leaf.operand_a, context.kind, *context.alloc_index);
    String b = resolve_operand(leaf.operand_b, context.kind, *context.alloc_index);
    String c = resolve_operand(leaf.operand_c, context.kind, *context.alloc_index);
    String function_name = backend == Backend::Msl ? "fma" : "fmaf";
    output << indent(indent_level) << destination << " = " << function_name << "(" << a << ", " << b << ", " << c
           << ");\n";
}

void emit_random(const RandomInstruction &leaf, Backend, EmitContext &context, int indent_level,
                  ostringstream &output) {
    String destination = resolve_operand(leaf.destination, context.kind, *context.alloc_index);
    String function_name = leaf.opcode == RandomOpcode::Rand ? "spikecorec_rand_uniform" : "spikecorec_randn";
    String index = context.kind == FunctionKind::PerNeuron ? "neuron_index" : "target_node";
    output << indent(indent_level) << destination << " = " << function_name << "(rng_state[" << index << "]);\n";
}

void emit_move(const MoveInstruction &leaf, Backend, EmitContext &context, int indent_level, ostringstream &output) {
    String destination = resolve_operand(leaf.destination, context.kind, *context.alloc_index);
    String source = resolve_operand(leaf.source, context.kind, *context.alloc_index);
    output << indent(indent_level) << destination << " = " << source << ";\n";
}

void emit_emit(const EmitInstruction &leaf, Backend, EmitContext &context, int indent_level, ostringstream &output) {
    String index = context.kind == FunctionKind::PerNeuron ? "neuron_index" : "target_node";
    output << indent(indent_level) << "emit_" << leaf.port_name << "[" << index << "] = true;\n";
}

void emit_set_regime(const SetRegimeInstruction &leaf, Backend, EmitContext &context, int indent_level,
                      ostringstream &output) {
    String destination = resolve_operand(leaf.regime_name, context.kind, *context.alloc_index);
    String value = resolve_operand(leaf.value, context.kind, *context.alloc_index);
    output << indent(indent_level) << destination << " = " << value << ";\n";
}

void emit_if(const IfInstruction &control, Backend backend, EmitContext &context, int indent_level,
             ostringstream &output) {
    String condition = resolve_operand(control.condition, context.kind, *context.alloc_index);
    output << indent(indent_level) << "if (" << condition << ") {\n";
    emit_instructions(control.then_body, backend, context, indent_level + 1, output);
    for (const auto &branch : control.else_if_branches) {
        String branch_condition = resolve_operand(branch.condition, context.kind, *context.alloc_index);
        output << indent(indent_level) << "} else if (" << branch_condition << ") {\n";
        emit_instructions(branch.body, backend, context, indent_level + 1, output);
    }
    if (control.else_body.has_value()) {
        output << indent(indent_level) << "} else {\n";
        emit_instructions(*control.else_body, backend, context, indent_level + 1, output);
    }
    output << indent(indent_level) << "}\n";
}

// Emits the shared for-loop header both `forall` and a whole-set (`@neuron_in`/`@neuron_out`)
// edge op use: `for (<index type> forall_slot_N = 0; ...; ++forall_slot_N) { int
// forall_neighbor_node_N = k2t_find_nth_neighbor(...); if (forall_neighbor_node_N < 0) continue;`.
// The caller still owns
// closing the loop's final `}` once its body is emitted. Only valid in the per-neuron function
// (forall/whole-set walks the current neuron's own k^2-tree row -- see gpu_source.h's documented
// neuron_in/neuron_out limitation); a deliver/onevent function already has its one, specific edge
// given directly as parameters and never needs a walk.
EdgeContext emit_forall_loop_header(Backend backend, EmitContext &context, int indent_level, ostringstream &output) {
    if (context.kind != FunctionKind::PerNeuron) {
        fail("forall / whole-set (neuron_in/neuron_out) edge ops are only supported in per-neuron stages, "
             "not inside a deliver/onevent body");
    }
    int forall_id = context.next_forall_id++;
    String slot_variable = "forall_slot_" + std::to_string(forall_id);
    String neighbor_variable = "forall_neighbor_node_" + std::to_string(forall_id);
    String index_type = index_cast_type(backend);

    output << indent(indent_level) << "for (" << index_type << " " << slot_variable << " = 0; " << slot_variable
           << " < max_neighbor_count; ++" << slot_variable << ") {\n";
    output << indent(indent_level + 1) << "int " << neighbor_variable << " = k2t_find_nth_neighbor(\n";
    output << indent(indent_level + 2) << "internal_node_words, leaf_node_words, rank_superblock_table, "
                                           "rank_subblock_table,\n";
    output << indent(indent_level + 2) << "branching_factor, superblock_size_words, (int)node_count, "
                                           "padded_node_count,\n";
    output << indent(indent_level + 2) << "tree_height, internal_bit_count, (int)neuron_index, (int)"
           << slot_variable << "\n";
    output << indent(indent_level + 1) << ");\n";
    output << indent(indent_level + 1) << "if (" << neighbor_variable << " < 0) continue;\n";

    return EdgeContext{"neuron_index", neighbor_variable, slot_variable};
}

void emit_forall(const ForAllInstruction &control, Backend backend, EmitContext &context, int indent_level,
                  ostringstream &output) {
    std::optional<EdgeContext> saved_edge_context = context.edge_context;
    EdgeContext new_edge_context = emit_forall_loop_header(backend, context, indent_level, output);
    context.edge_context = new_edge_context;
    emit_instructions(control.body, backend, context, indent_level + 1, output);
    context.edge_context = saved_edge_context;
    output << indent(indent_level) << "}\n";
}

// `loadedge dst, <var>@edge`: reconstructs Σ U[row]·Ck[var]·V[other] + Sk[var][row,slot] --
// weight_matrix.h's get_for_matrix()+sparse-delta-overlay math, matching neighbor_weights_kernel's
// own per-lane loop exactly (src/metal/kernels.metal, src/cuda/kernels.cu), MSL/CUDA divergence
// mirrored line-for-line (MSL builds a float4 for the lane's Ck slice and calls dot(); CUDA
// manually expands the four lane products, avoiding float4 literal construction, exactly like
// neighbor_weights_kernel's own CUDA implementation does).
void emit_loadedge(const LoadEdgeInstruction &leaf, Backend backend, EmitContext &context, int indent_level,
                    ostringstream &output) {
    if (leaf.edge_reference != EdgeSetReference::CurrentEdge) {
        fail("loadedge only supports @edge -- a whole-set load has no defined reduction into one destination");
    }
    if (!context.edge_context.has_value()) fail("loadedge '@edge' used outside a forall/onevent body");

    auto found = context.alloc_index->by_name.find(leaf.variable_name);
    if (found == context.alloc_index->by_name.end() || found->second.category != AllocCategory::Peredge) {
        fail("loadedge target '" + leaf.variable_name + "' must be a peredge quantity");
    }

    const EdgeContext &edge = *context.edge_context;
    String destination = resolve_operand(leaf.destination, context.kind, *context.alloc_index);
    String coefficients_name = "coefficients_" + leaf.variable_name;
    String sparse_delta_name = "sparse_delta_" + leaf.variable_name;
    String cast = index_cast_type(backend);
    bool msl = backend == Backend::Msl;

    output << indent(indent_level) << "{\n";
    output << indent(indent_level + 1) << (msl ? "const device float4 *" : "const float4 *")
           << "edge_basis_u_row = U + (" << cast << ")" << edge.row_node_expr << " * rank_float4_stride;\n";
    output << indent(indent_level + 1) << (msl ? "const device float4 *" : "const float4 *")
           << "edge_basis_v_row = V + (" << cast << ")" << edge.other_node_expr << " * rank_float4_stride;\n";
    output << indent(indent_level + 1) << "float edge_reconstruction = 0.0f;\n";
    output << indent(indent_level + 1) << "for (" << cast << " lane = 0; lane < rank_float4_stride; ++lane) {\n";
    if (msl) {
        output << indent(indent_level + 2) << "float4 lane_coefficients = float4(\n";
        output << indent(indent_level + 3) << coefficients_name << "[lane * 4 + 0], " << coefficients_name
               << "[lane * 4 + 1],\n";
        output << indent(indent_level + 3) << coefficients_name << "[lane * 4 + 2], " << coefficients_name
               << "[lane * 4 + 3]\n";
        output << indent(indent_level + 2) << ");\n";
        output << indent(indent_level + 2)
               << "edge_reconstruction += dot(edge_basis_u_row[lane], lane_coefficients * edge_basis_v_row[lane]);\n";
    } else {
        output << indent(indent_level + 2) << "float4 u4 = edge_basis_u_row[lane];\n";
        output << indent(indent_level + 2) << "float4 v4 = edge_basis_v_row[lane];\n";
        output << indent(indent_level + 2) << "const float *lane_coefficients = " << coefficients_name
               << " + lane * 4;\n";
        output << indent(indent_level + 2)
               << "edge_reconstruction += u4.x * (lane_coefficients[0] * v4.x) + u4.y * (lane_coefficients[1] * "
                  "v4.y)\n";
        output << indent(indent_level + 3) << "+ u4.z * (lane_coefficients[2] * v4.z) + u4.w * (lane_coefficients[3] "
                                               "* v4.w);\n";
    }
    output << indent(indent_level + 1) << "}\n";
    output << indent(indent_level + 1) << "edge_reconstruction += " << sparse_delta_name << "[(" << cast << ")"
           << edge.row_node_expr << " * max_neighbor_count + " << edge.slot_expr << "];\n";
    output << indent(indent_level + 1) << destination << " = edge_reconstruction;\n";
    output << indent(indent_level) << "}\n";
}

// `accedge <var>@edge, value`: Sk[var][row,slot] += value for a peredge var (no basis touched --
// weight_matrix.h's accumulate_edge_delta), or <var>[other] += value for an accum var
// (ir_spec.md §4's expOne comment: "aggregated -> g[target] += weight").
void emit_accedge_body(const AccumulateEdgeInstruction &leaf, Backend backend, EmitContext &context,
                        const EdgeContext &edge, int indent_level, ostringstream &output) {
    String value = resolve_operand(leaf.value, context.kind, *context.alloc_index);
    auto found = context.alloc_index->by_name.find(leaf.variable_name);
    if (found == context.alloc_index->by_name.end()) {
        fail("accedge references unknown variable '" + leaf.variable_name + "'");
    }
    if (found->second.category == AllocCategory::Peredge) {
        String cast = index_cast_type(backend);
        output << indent(indent_level) << "sparse_delta_" << leaf.variable_name << "[(" << cast << ")"
               << edge.row_node_expr << " * max_neighbor_count + " << edge.slot_expr << "] += " << value << ";\n";
    } else if (found->second.category == AllocCategory::Accum) {
        output << indent(indent_level) << leaf.variable_name << "[" << edge.other_node_expr << "] += " << value
               << ";\n";
    } else {
        fail("accedge target '" + leaf.variable_name + "' must be a peredge or accum quantity");
    }
}

void emit_accedge(const AccumulateEdgeInstruction &leaf, Backend backend, EmitContext &context, int indent_level,
                   ostringstream &output) {
    if (leaf.edge_reference == EdgeSetReference::CurrentEdge) {
        if (!context.edge_context.has_value()) fail("accedge '@edge' used outside a forall/onevent body");
        emit_accedge_body(leaf, backend, context, *context.edge_context, indent_level, output);
        return;
    }

    // Whole-set scatter/gather form (ir_spec.md §3.4: `accedge g@neuron_out, weight`) -- synthesize
    // the same for-loop `forall` uses, wrapping just this one instruction.
    if (context.edge_context.has_value()) {
        fail("a neuron_in/neuron_out edge-set op nested inside an existing forall/onevent body is not supported");
    }
    EdgeContext new_edge_context = emit_forall_loop_header(backend, context, indent_level, output);
    context.edge_context = new_edge_context;
    emit_accedge_body(leaf, backend, context, new_edge_context, indent_level + 1, output);
    context.edge_context = std::nullopt;
    output << indent(indent_level) << "}\n";
}

void emit_leaf(const TickInstruction &instruction, Backend backend, EmitContext &context, int indent_level,
               ostringstream &output) {
    std::visit(Overloaded{
        [&](const BinaryInstruction &leaf) { emit_binary(leaf, backend, context, indent_level, output); },
        [&](const UnaryInstruction &leaf) { emit_unary(leaf, backend, context, indent_level, output); },
        [&](const FusedMultiplyAddInstruction &leaf) { emit_fma(leaf, backend, context, indent_level, output); },
        [&](const RandomInstruction &leaf) { emit_random(leaf, backend, context, indent_level, output); },
        [&](const MoveInstruction &leaf) { emit_move(leaf, backend, context, indent_level, output); },
        [&](const LoadEdgeInstruction &leaf) { emit_loadedge(leaf, backend, context, indent_level, output); },
        [&](const AccumulateEdgeInstruction &leaf) { emit_accedge(leaf, backend, context, indent_level, output); },
        [&](const EmitInstruction &leaf) { emit_emit(leaf, backend, context, indent_level, output); },
        [&](const SetRegimeInstruction &leaf) { emit_set_regime(leaf, backend, context, indent_level, output); },
        [&](const IfInstruction &control) { emit_if(control, backend, context, indent_level, output); },
        [&](const ForAllInstruction &control) { emit_forall(control, backend, context, indent_level, output); },
        [&](const OnEventInstruction &) { fail("nested onevent blocks are not supported"); },
    }, instruction.operation);
}

void emit_instructions(const Vector<TickInstruction> &instructions, Backend backend, EmitContext &context,
                        int indent_level, ostringstream &output) {
    for (const auto &instruction : instructions) emit_leaf(instruction, backend, context, indent_level, output);
}

String render_locals_declarations(const Vector<LocalDecl> &locals, int indent_level) {
    ostringstream output;
    for (const auto &local : locals) {
        output << indent(indent_level) << (local.is_bool ? "bool " : "float ") << local.name << ";\n";
    }
    return output.str();
}

String render_baked_param_locals(const Vector<AllocDirective> &alloc, const ScanResult &scan, int indent_level) {
    ostringstream output;
    for (const auto &directive : alloc) {
        const auto *param = std::get_if<ParamConstantDirective>(&directive);
        if (param == nullptr || !param->literal_value.has_value()) continue;
        if (scan.alloc_names_referenced.count(param->name) == 0) continue;
        output << indent(indent_level) << "const float " << param->name << " = " << *param->literal_value << ";\n";
    }
    return output.str();
}

// ── shared preambles (emitted once per generated source, matching how
//    k2t_find_nth_neighbor/RNG helpers would be shared by many kernels) ───────────────────────

String k2tree_preamble(Backend backend) {
    if (backend == Backend::Msl) {
        return R"MSL(
// -- k^2-tree bit-walk helpers (mirrors src/metal/kernels.metal's k2t_find_nth_neighbor) --------
inline uint k2t_get_bit(const device uint *words, int bit_index) {
    return (words[bit_index >> 5] >> (uint)(bit_index & 31)) & 1u;
}

inline int k2t_rank1_exclusive(
    const device uint *internal_words,
    const device uint *superblock_table,
    const device ushort *subblock_table,
    int position,
    int superblock_size
) {
    int word_index = position >> 5;
    int bit_offset = position & 31;
    int superblock_index = word_index / superblock_size;
    uint superblock_base = superblock_table[superblock_index];
    uint subblock_base = (uint)subblock_table[word_index];
    uint partial_word_mask = (bit_offset == 0) ? 0u : ((1u << (uint)bit_offset) - 1u);
    uint partial_word_popcount = popcount(internal_words[word_index] & partial_word_mask);
    return (int)(superblock_base + subblock_base + partial_word_popcount);
}

#define MAX_K2TREE_HEIGHT 32

inline int k2t_find_nth_neighbor(
    const device uint   *internal_node_words,
    const device uint   *leaf_node_words,
    const device uint   *rank_superblock_table,
    const device ushort *rank_subblock_table,
    int branching_factor,
    int superblock_size_words,
    int node_count,
    int padded_node_count,
    int tree_height,
    int internal_bit_count,
    int u,
    int target_slot
) {
    if (tree_height == 0 || u < 0 || u >= node_count || target_slot < 0) return -1;

    int branching_factor_squared = branching_factor * branching_factor;

    int stack_row_base[MAX_K2TREE_HEIGHT];
    int stack_col_base[MAX_K2TREE_HEIGHT];
    int stack_block_size[MAX_K2TREE_HEIGHT];
    int stack_bit_offset[MAX_K2TREE_HEIGHT];
    int stack_next_col[MAX_K2TREE_HEIGHT];

    stack_row_base[0] = 0;
    stack_col_base[0] = 0;
    stack_block_size[0] = padded_node_count;
    stack_bit_offset[0] = 0;
    stack_next_col[0] = 0;

    int stack_top = 0;
    int neighbors_seen = 0;

    while (stack_top >= 0) {
        int level = stack_top;
        int col_offset = stack_next_col[level];
        if (col_offset >= branching_factor) {
            stack_top--;
            continue;
        }
        stack_next_col[level] = col_offset + 1;

        int row_base = stack_row_base[level];
        int col_base = stack_col_base[level];
        int block_size = stack_block_size[level];
        int level_bit_offset = stack_bit_offset[level];

        int child_block_size = block_size / branching_factor;
        int row_offset = (u - row_base) / child_block_size;
        int child_flat_index = row_offset * branching_factor + col_offset;
        int bit_position = level_bit_offset + child_flat_index;

        if (level == tree_height - 1) {
            if (k2t_get_bit(leaf_node_words, bit_position)) {
                int v = col_base + col_offset;
                if (v < node_count) {
                    if (neighbors_seen == target_slot) return v;
                    neighbors_seen++;
                }
            }
        } else if (k2t_get_bit(internal_node_words, bit_position)) {
            int rank_inclusive = k2t_rank1_exclusive(internal_node_words, rank_superblock_table,
                                                      rank_subblock_table, bit_position, superblock_size_words) + 1;
            int child_level = stack_top + 1;
            int raw_offset = branching_factor_squared * rank_inclusive;
            stack_row_base[child_level] = row_base + row_offset * child_block_size;
            stack_col_base[child_level] = col_base + col_offset * child_block_size;
            stack_block_size[child_level] = child_block_size;
            stack_bit_offset[child_level] = (child_level == tree_height - 1)
                ? (raw_offset - internal_bit_count)
                : raw_offset;
            stack_next_col[child_level] = 0;
            stack_top = child_level;
        }
    }

    return -1;
}
)MSL";
    }

    return R"CUDA(
// -- k^2-tree bit-walk helpers (mirrors src/cuda/kernels.cu's k2t_find_nth_neighbor) -------------
__device__ __forceinline__ unsigned int k2t_get_bit(const unsigned int *words, int bit_index) {
    return (words[bit_index >> 5] >> (bit_index & 31)) & 1u;
}

__device__ __forceinline__ int k2t_rank1_exclusive(
    const unsigned int *internal_words,
    const unsigned int *superblock_table,
    const unsigned short *subblock_table,
    int position,
    int superblock_size
) {
    int word_index = position >> 5;
    int bit_offset = position & 31;
    int superblock_index = word_index / superblock_size;
    unsigned int superblock_base = superblock_table[superblock_index];
    unsigned int subblock_base = subblock_table[word_index];
    unsigned int partial_word_mask = (bit_offset == 0) ? 0u : ((1u << bit_offset) - 1u);
    unsigned int partial_word_popcount =
        static_cast<unsigned int>(__popc(internal_words[word_index] & partial_word_mask));
    return static_cast<int>(superblock_base + subblock_base + partial_word_popcount);
}

#define MAX_K2TREE_HEIGHT 32

__device__ __forceinline__ int k2t_find_nth_neighbor(
    const unsigned int   *internal_node_words,
    const unsigned int   *leaf_node_words,
    const unsigned int   *rank_superblock_table,
    const unsigned short *rank_subblock_table,
    int branching_factor,
    int superblock_size_words,
    int node_count,
    int padded_node_count,
    int tree_height,
    int internal_bit_count,
    int u,
    int target_slot
) {
    if (tree_height == 0 || u < 0 || u >= node_count || target_slot < 0) return -1;

    int branching_factor_squared = branching_factor * branching_factor;

    int stack_row_base[MAX_K2TREE_HEIGHT];
    int stack_col_base[MAX_K2TREE_HEIGHT];
    int stack_block_size[MAX_K2TREE_HEIGHT];
    int stack_bit_offset[MAX_K2TREE_HEIGHT];
    int stack_next_col[MAX_K2TREE_HEIGHT];

    stack_row_base[0] = 0;
    stack_col_base[0] = 0;
    stack_block_size[0] = padded_node_count;
    stack_bit_offset[0] = 0;
    stack_next_col[0] = 0;

    int stack_top = 0;
    int neighbors_seen = 0;

    while (stack_top >= 0) {
        int level = stack_top;
        int col_offset = stack_next_col[level];
        if (col_offset >= branching_factor) {
            stack_top--;
            continue;
        }
        stack_next_col[level] = col_offset + 1;

        int row_base = stack_row_base[level];
        int col_base = stack_col_base[level];
        int block_size = stack_block_size[level];
        int level_bit_offset = stack_bit_offset[level];

        int child_block_size = block_size / branching_factor;
        int row_offset = (u - row_base) / child_block_size;
        int child_flat_index = row_offset * branching_factor + col_offset;
        int bit_position = level_bit_offset + child_flat_index;

        if (level == tree_height - 1) {
            if (k2t_get_bit(leaf_node_words, bit_position)) {
                int v = col_base + col_offset;
                if (v < node_count) {
                    if (neighbors_seen == target_slot) return v;
                    neighbors_seen++;
                }
            }
        } else if (k2t_get_bit(internal_node_words, bit_position)) {
            int rank_inclusive = k2t_rank1_exclusive(internal_node_words, rank_superblock_table,
                                                      rank_subblock_table, bit_position, superblock_size_words) + 1;
            int child_level = stack_top + 1;
            int raw_offset = branching_factor_squared * rank_inclusive;
            stack_row_base[child_level] = row_base + row_offset * child_block_size;
            stack_col_base[child_level] = col_base + col_offset * child_block_size;
            stack_block_size[child_level] = child_block_size;
            stack_bit_offset[child_level] = (child_level == tree_height - 1)
                ? (raw_offset - internal_bit_count)
                : raw_offset;
            stack_next_col[child_level] = 0;
            stack_top = child_level;
        }
    }

    return -1;
}
)CUDA";
}

// No RNG precedent exists anywhere in src/metal/kernels.metal or src/cuda/kernels.cu -- this is a
// new, documented, minimal choice (gpu_source.h's doc comment): a per-neuron persistent xorshift32
// state, uniform via the top 24 bits of one step, standard-normal via Box-Muller over two draws.
String rng_preamble(Backend backend) {
    if (backend == Backend::Msl) {
        return R"MSL(
// -- per-neuron RNG (xorshift32 + Box-Muller; no prior precedent, see gpu_source.h) --------------
inline uint spikecorec_rand_next_u32(device uint &state) {
    uint x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

inline float spikecorec_rand_uniform(device uint &state) {
    return (float)(spikecorec_rand_next_u32(state) >> 8) * (1.0f / 16777216.0f);
}

inline float spikecorec_randn(device uint &state) {
    float u1 = spikecorec_rand_uniform(state);
    float u2 = spikecorec_rand_uniform(state);
    float radius = sqrt(-2.0f * log(max(u1, 1e-7f)));
    return radius * cos(2.0f * 3.14159265358979323846f * u2);
}
)MSL";
    }

    return R"CUDA(
// -- per-neuron RNG (xorshift32 + Box-Muller; no prior precedent, see gpu_source.h) --------------
__device__ __forceinline__ unsigned int spikecorec_rand_next_u32(unsigned int &state) {
    unsigned int x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

__device__ __forceinline__ float spikecorec_rand_uniform(unsigned int &state) {
    return (float)(spikecorec_rand_next_u32(state) >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float spikecorec_randn(unsigned int &state) {
    float u1 = spikecorec_rand_uniform(state);
    float u2 = spikecorec_rand_uniform(state);
    float radius = sqrtf(-2.0f * logf(fmaxf(u1, 1e-7f)));
    return radius * cosf(2.0f * 3.14159265358979323846f * u2);
}
)CUDA";
}

// ── whole-function generation ──────────────────────────────────────────────────────────────────

struct FunctionResult {
    String msl_text;
    String cuda_text;
    GpuFunctionSignature signature;
};

void render_signature_and_prologue(Backend backend, const String &function_name,
                                    const Vector<ParamDescriptor> &parameters, FunctionKind kind,
                                    const Vector<AllocDirective> &alloc, const ScanResult &scan, ostringstream &output) {
    bool msl = backend == Backend::Msl;
    output << (msl ? "kernel void " : "__global__ void ") << function_name << "(\n";
    for (usize index = 0; index < parameters.size(); ++index) {
        if (msl) {
            output << render_msl_param(parameters[index], static_cast<int>(index)) << ",\n";
        } else {
            output << "    " << render_cuda_param(parameters[index])
                   << (index + 1 < parameters.size() ? ",\n" : "\n");
        }
    }
    if (msl) output << "    uint thread_id [[ thread_position_in_grid ]]\n";
    output << ") {\n";

    if (kind == FunctionKind::PerNeuron) {
        if (msl) {
            output << "    long neuron_index = (long)thread_id;\n";
        } else {
            output << "    long long neuron_index = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n";
        }
        output << "    if (neuron_index >= neuron_count) return;\n";
    } else {
        if (msl) {
            output << "    long delivery_event_index = (long)thread_id;\n";
        } else {
            output << "    long long delivery_event_index = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n";
        }
        output << "    if (delivery_event_index >= event_count) return;\n";
        output << "    int source_node = source_node_indices[delivery_event_index];\n";
        output << "    int target_node = target_node_indices[delivery_event_index];\n";
        output << "    int edge_slot = edge_slot_indices[delivery_event_index];\n";
    }

    output << render_baked_param_locals(alloc, scan, 1);
    output << render_locals_declarations(scan.locals_in_order, 1);
}

// The combined per-neuron function: `@integrate`, `@detect`, `@emit`, `@reset`, `@propagate`,
// `@plasticity`, `@record`, concatenated in that canonical order into ONE function body (see
// gpu_source.h's doc comment for why they must share one local-variable scope). Returns nullopt if
// every one of those 7 stages is empty (nothing to generate).
std::optional<FunctionResult> generate_per_neuron_function(const IrProgram &program, const AllocIndex &alloc_index) {
    Vector<std::pair<String, const Vector<TickInstruction> *>> stages = {
        {"integrate", &program.tick.integrate}, {"detect", &program.tick.detect}, {"emit", &program.tick.emit},
        {"reset", &program.tick.reset},         {"propagate", &program.tick.propagate},
        {"plasticity", &program.tick.plasticity}, {"record", &program.tick.record},
    };

    bool any_non_empty = false;
    for (const auto &stage : stages) any_non_empty = any_non_empty || !stage.second->empty();
    if (!any_non_empty) return std::nullopt;

    ScanResult scan;
    unordered_set<String> locals_seen;
    unordered_set<String> edge_variable_names_referenced;
    bool needs_edge_walk_block = false;
    bool needs_rng = false;
    Vector<String> emit_ports;
    unordered_set<String> emit_ports_seen;

    for (const auto &stage : stages) {
        scan_operands(*stage.second, alloc_index, scan, locals_seen);
        collect_edge_variable_names(*stage.second, edge_variable_names_referenced);
        needs_edge_walk_block = needs_edge_walk_block || contains_forall(*stage.second) ||
                                 contains_whole_set_edge_op(*stage.second);
        needs_rng = needs_rng || contains_random(*stage.second);
        collect_emit_ports(*stage.second, emit_ports, emit_ports_seen);
    }

    // A per-neuron function's only edge context is `forall`/a whole-set edge op (there is no
    // onevent-style "already-known edge" here) -- so the tree-walk block and the shared basis are
    // needed together, whenever either is needed at all.
    Vector<ParamDescriptor> parameters =
        build_common_parameters(scan, edge_variable_names_referenced, needs_edge_walk_block, needs_edge_walk_block,
                                 needs_rng, emit_ports, alloc_index, program.alloc);
    parameters.push_back({"long", "long long", "neuron_count", ParamKind::Scalar});

    String function_name = program.component_type_name + "_tick";

    ostringstream msl;
    render_signature_and_prologue(Backend::Msl, function_name, parameters, FunctionKind::PerNeuron, program.alloc,
                                   scan, msl);
    {
        EmitContext context{FunctionKind::PerNeuron, &alloc_index, std::nullopt, 0};
        for (const auto &stage : stages) {
            if (stage.second->empty()) continue;
            msl << "    // @" << stage.first << "\n";
            emit_instructions(*stage.second, Backend::Msl, context, 1, msl);
        }
    }
    msl << "}\n";

    ostringstream cuda;
    render_signature_and_prologue(Backend::Cuda, function_name, parameters, FunctionKind::PerNeuron, program.alloc,
                                   scan, cuda);
    {
        EmitContext context{FunctionKind::PerNeuron, &alloc_index, std::nullopt, 0};
        for (const auto &stage : stages) {
            if (stage.second->empty()) continue;
            cuda << "    // @" << stage.first << "\n";
            emit_instructions(*stage.second, Backend::Cuda, context, 1, cuda);
        }
    }
    cuda << "}\n";

    Vector<String> parameter_names;
    parameter_names.reserve(parameters.size());
    for (const auto &parameter : parameters) parameter_names.push_back(parameter.name);

    return FunctionResult{msl.str(), cuda.str(), GpuFunctionSignature{function_name, parameter_names}};
}

// Shared machinery behind BOTH kinds of "one thread per edge" function this file generates: an
// onevent/deliver body (below), AND -- ticket #131 -- a Synapse-category program's own
// `.tick.integrate` gather (lower_synapse_ir_program_to_gpu_source), whose single top-level
// `forall neuron_in { ... }` wrapper the caller has already stripped before calling this. `body` is
// already edge-scoped either way (its own `loadedge`/`accedge` reference `@edge` directly, needing
// no walk -- the edge is given as source_node/target_node/edge_slot, see EdgeContext below).
FunctionResult generate_edge_parallel_function(const IrProgram &program, const AllocIndex &alloc_index,
                                                const Vector<TickInstruction> &body, const String &function_name) {
    ScanResult scan;
    unordered_set<String> locals_seen;
    unordered_set<String> edge_variable_names_referenced;
    bool needs_rng = contains_random(body);
    Vector<String> emit_ports;
    unordered_set<String> emit_ports_seen;
    collect_emit_ports(body, emit_ports, emit_ports_seen);
    scan_operands(body, alloc_index, scan, locals_seen);
    collect_edge_variable_names(body, edge_variable_names_referenced);
    if (contains_forall(body) || contains_whole_set_edge_op(body)) {
        fail("forall / whole-set edge ops inside an onevent/deliver/edge-parallel body are not supported by "
             "this ticket's lowering (the edge is already known -- no walk is needed)");
    }
    // The edge is already known (source_node/target_node/edge_slot, no walk) -- but a `loadedge` on a
    // peredge var still needs the shared basis (U/V/rank_float4_stride/coefficients_<name>) to
    // reconstruct that edge's value; `accedge` alone never does (weight_matrix.h: Sk accumulate is
    // O(1), no basis touched).
    bool needs_shared_basis_block = contains_loadedge(body);

    Vector<ParamDescriptor> parameters =
        build_common_parameters(scan, edge_variable_names_referenced, /*needs_tree_walk_block=*/false,
                                 needs_shared_basis_block, needs_rng, emit_ports, alloc_index, program.alloc);
    parameters.push_back({"int", "int", "source_node_indices", ParamKind::ConstArray});
    parameters.push_back({"int", "int", "target_node_indices", ParamKind::ConstArray});
    parameters.push_back({"int", "int", "edge_slot_indices", ParamKind::ConstArray});
    parameters.push_back({"long", "long long", "event_count", ParamKind::Scalar});

    ostringstream msl;
    render_signature_and_prologue(Backend::Msl, function_name, parameters, FunctionKind::Deliver, program.alloc,
                                   scan, msl);
    {
        EmitContext context{FunctionKind::Deliver, &alloc_index,
                             EdgeContext{"source_node", "target_node", "edge_slot"}, 0};
        emit_instructions(body, Backend::Msl, context, 1, msl);
    }
    msl << "}\n";

    ostringstream cuda;
    render_signature_and_prologue(Backend::Cuda, function_name, parameters, FunctionKind::Deliver, program.alloc,
                                   scan, cuda);
    {
        EmitContext context{FunctionKind::Deliver, &alloc_index,
                             EdgeContext{"source_node", "target_node", "edge_slot"}, 0};
        emit_instructions(body, Backend::Cuda, context, 1, cuda);
    }
    cuda << "}\n";

    Vector<String> parameter_names;
    parameter_names.reserve(parameters.size());
    for (const auto &parameter : parameters) parameter_names.push_back(parameter.name);

    return FunctionResult{msl.str(), cuda.str(), GpuFunctionSignature{function_name, parameter_names}};
}

// One function per `onevent <port> { ... }` block found at the top level of `.tick.deliver` (see
// gpu_source.h's doc comment for the batch-of-delivered-edges calling convention).
FunctionResult generate_deliver_function(const IrProgram &program, const AllocIndex &alloc_index,
                                          const OnEventInstruction &onevent, const String &function_name) {
    return generate_edge_parallel_function(program, alloc_index, onevent.body, function_name);
}

// Validates and unwraps a Synapse-category program's `.tick.integrate` -- synapse_lowering.cpp's
// own, sole shape: exactly one top-level `forall neuron_in { ... }` (ir_spec.md §4's expOne/NMDA
// examples). Throws otherwise (a Cell-category or otherwise-unexpected program was passed to
// lower_synapse_ir_program_to_gpu_source).
const Vector<TickInstruction> &unwrap_synapse_integrate_forall_body(const IrProgram &program) {
    if (program.tick.integrate.size() != 1) {
        fail("lower_synapse_ir_program_to_gpu_source: program '" + program.component_type_name +
             "'.tick.integrate must contain exactly one top-level instruction (a `forall neuron_in { ... }`, "
             "synapse_lowering.cpp's own sole shape), found " + std::to_string(program.tick.integrate.size()));
    }
    const auto *forall = std::get_if<ForAllInstruction>(&program.tick.integrate[0].operation);
    if (forall == nullptr) {
        fail("lower_synapse_ir_program_to_gpu_source: program '" + program.component_type_name +
             "'.tick.integrate's one instruction must be a ForAllInstruction (synapse_lowering.cpp's own "
             "sole shape)");
    }
    if (forall->edge_set != EdgeSetReference::NeuronIn && forall->edge_set != EdgeSetReference::NeuronOut) {
        fail("lower_synapse_ir_program_to_gpu_source: program '" + program.component_type_name +
             "'.tick.integrate's forall must iterate NeuronIn/NeuronOut");
    }
    return forall->body;
}

} // namespace

GpuSource lower_ir_program_to_gpu_source(const IrProgram &program) {
    AllocIndex alloc_index = build_alloc_index(program.alloc);

    Vector<FunctionResult> function_results;

    if (std::optional<FunctionResult> per_neuron = generate_per_neuron_function(program, alloc_index);
        per_neuron.has_value()) {
        function_results.push_back(*per_neuron);
    }

    unordered_set<String> deliver_function_names_seen;
    for (const auto &instruction : program.tick.deliver) {
        const auto *onevent = std::get_if<OnEventInstruction>(&instruction.operation);
        if (onevent == nullptr) fail(".tick.deliver may only contain top-level onevent blocks");

        String base_name = program.component_type_name + "_deliver_" + onevent->port_name;
        String function_name = base_name;
        int suffix = 2;
        while (deliver_function_names_seen.count(function_name) > 0) {
            function_name = base_name + "_" + std::to_string(suffix++);
        }
        deliver_function_names_seen.insert(function_name);

        function_results.push_back(generate_deliver_function(program, alloc_index, *onevent, function_name));
    }

    bool needs_edge_walk_anywhere = false;
    bool needs_rng_anywhere = false;
    Vector<const Vector<TickInstruction> *> per_neuron_stage_lists = {
        &program.tick.integrate, &program.tick.detect,     &program.tick.emit,       &program.tick.reset,
        &program.tick.propagate, &program.tick.plasticity, &program.tick.record,
    };
    for (const auto *stage : per_neuron_stage_lists) {
        needs_edge_walk_anywhere =
            needs_edge_walk_anywhere || contains_forall(*stage) || contains_whole_set_edge_op(*stage);
        needs_rng_anywhere = needs_rng_anywhere || contains_random(*stage);
    }
    for (const auto &instruction : program.tick.deliver) {
        if (const auto *onevent = std::get_if<OnEventInstruction>(&instruction.operation)) {
            needs_rng_anywhere = needs_rng_anywhere || contains_random(onevent->body);
        }
    }

    String msl_source = "#include <metal_stdlib>\nusing namespace metal;\n";
    if (needs_edge_walk_anywhere) msl_source += k2tree_preamble(Backend::Msl);
    if (needs_rng_anywhere) msl_source += rng_preamble(Backend::Msl);
    for (const auto &function : function_results) msl_source += "\n" + function.msl_text;

    String cuda_source;
    if (needs_edge_walk_anywhere) cuda_source += "#include <vector_types.h>\n";
    if (needs_edge_walk_anywhere) cuda_source += k2tree_preamble(Backend::Cuda);
    if (needs_rng_anywhere) cuda_source += rng_preamble(Backend::Cuda);
    for (const auto &function : function_results) cuda_source += "\n" + function.cuda_text;

    Vector<GpuFunctionSignature> signatures;
    signatures.reserve(function_results.size());
    for (const auto &function : function_results) signatures.push_back(function.signature);

    return GpuSource{msl_source, cuda_source, signatures};
}

// ticket #131: see gpu_source.h's own doc comment for the full design.
GpuSource lower_synapse_ir_program_to_gpu_source(const IrProgram &program) {
    for (const Vector<TickInstruction> *stage :
         {&program.tick.detect, &program.tick.emit, &program.tick.reset, &program.tick.propagate,
          &program.tick.plasticity, &program.tick.record}) {
        if (!stage->empty()) {
            fail("lower_synapse_ir_program_to_gpu_source: program '" + program.component_type_name +
                 "' has a non-empty @detect/@emit/@reset/@propagate/@plasticity/@record -- only "
                 "@deliver/@integrate are supported for a Synapse-category program (synapse_lowering.cpp "
                 "never populates the others; use lower_ir_program_to_gpu_source for a Cell-category program)");
        }
    }
    const Vector<TickInstruction> &integrate_body = unwrap_synapse_integrate_forall_body(program);

    AllocIndex alloc_index = build_alloc_index(program.alloc);

    // An `ExposeDirective` name with no OTHER backing directive (a plain-value DerivedVariable's own
    // exposure, e.g. expOneSynapse's `i = g*(erev-v)`) carries no real storage for a Synapse-category
    // type: allocator.cpp's own derived_exposure_scratch_buffers is deliberately Cell-only ("scratch
    // is Cell-only"), so a synapse never gets a scratch slot for one. resolve_operand's Expose case
    // only supports FunctionKind::PerNeuron (there is no synapse-side per-neuron function anymore,
    // ticket #131) -- so here, every such name must instead behave as a plain transient local
    // (computed and consumed within the SAME edge-parallel function body, exactly how expOneSynapse's
    // own `i` is used: written once via a DerivedVariable, read back once for `network_inputs +=
    // i`). Removing it from the index BEFORE any function is generated makes record_operand/
    // resolve_operand treat it as an ordinary local uniformly, for every function this lowering
    // produces (onevent AND `_integrate_edges` alike).
    for (auto entry = alloc_index.by_name.begin(); entry != alloc_index.by_name.end();) {
        if (entry->second.category == AllocCategory::Expose) {
            entry = alloc_index.by_name.erase(entry);
        } else {
            ++entry;
        }
    }

    Vector<FunctionResult> function_results;

    unordered_set<String> deliver_function_names_seen;
    for (const auto &instruction : program.tick.deliver) {
        const auto *onevent = std::get_if<OnEventInstruction>(&instruction.operation);
        if (onevent == nullptr) fail("lower_synapse_ir_program_to_gpu_source: .tick.deliver may only contain "
                                      "top-level onevent blocks");

        String base_name = program.component_type_name + "_deliver_" + onevent->port_name;
        String function_name = base_name;
        int suffix = 2;
        while (deliver_function_names_seen.count(function_name) > 0) {
            function_name = base_name + "_" + std::to_string(suffix++);
        }
        deliver_function_names_seen.insert(function_name);

        function_results.push_back(generate_deliver_function(program, alloc_index, *onevent, function_name));
    }

    // The `_integrate_edges` function is always last (AssembledModel::AssembledModel, ticket #131,
    // relies on this exact ordering to tell it apart from the deliver functions above without
    // re-deriving a name-matching scheme).
    String integrate_edges_function_name = program.component_type_name + "_integrate_edges";
    function_results.push_back(
        generate_edge_parallel_function(program, alloc_index, integrate_body, integrate_edges_function_name));

    bool needs_rng_anywhere = contains_random(integrate_body);
    for (const auto &instruction : program.tick.deliver) {
        if (const auto *onevent = std::get_if<OnEventInstruction>(&instruction.operation)) {
            needs_rng_anywhere = needs_rng_anywhere || contains_random(onevent->body);
        }
    }

    // No k^2-tree walk preamble: every function this lowering generates is edge-parallel (the edge
    // is always already known), unlike lower_ir_program_to_gpu_source's own per-neuron function.
    String msl_source = "#include <metal_stdlib>\nusing namespace metal;\n";
    if (needs_rng_anywhere) msl_source += rng_preamble(Backend::Msl);
    for (const auto &function : function_results) msl_source += "\n" + function.msl_text;

    String cuda_source;
    if (needs_rng_anywhere) cuda_source += rng_preamble(Backend::Cuda);
    for (const auto &function : function_results) cuda_source += "\n" + function.cuda_text;

    Vector<GpuFunctionSignature> signatures;
    signatures.reserve(function_results.size());
    for (const auto &function : function_results) signatures.push_back(function.signature);

    return GpuSource{msl_source, cuda_source, signatures};
}

String k2tree_walk_preamble_msl() { return k2tree_preamble(Backend::Msl); }
String k2tree_walk_preamble_cuda() { return k2tree_preamble(Backend::Cuda); }

} // namespace spikecorec::nml
