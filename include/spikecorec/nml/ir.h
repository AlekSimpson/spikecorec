#pragma once

#include <optional>
#include <variant>

#include "spikecorec/core/types.h"

namespace spikecorec::nml {

// ── IR in-engine representation (ticket #4 [B1]; docs/nml_ir_spec.md §2-§3, LOCKED v1.0) ──
//
// A per-ComponentType IR program is `.alloc` (engine-interpreted at init, not
// compiled -- IR spec §2) plus `.tick` (per-tick compute, stage-tagged,
// template-substituted to MSL/CUDA later -- IR spec §3). This header is
// ONLY the in-engine C++ representation of that ISA -- constructible
// programmatically and printable back to the spec's text form (see
// print_ir_program in ir.cpp). It does not lower a ResolvedComponentType
// into this IR (ticket #50/#51) and does not compile it to GPU source
// (ticket #55).

// ── .alloc directives (IR spec §2) ───────────────────────────────────────

// `param <name> = <literal>` (a baked constant) or a bare `param <name>`
// with no literal yet -- IR spec §4's GLIF1 example groups several bare
// names on one `param` line (`param C, gL, EL, vth, vreset`); this
// representation models that as one ParamConstantDirective per name (see
// ir.cpp's printer for the resulting one-per-line normalization).
struct ParamConstantDirective {
    String name;
    std::optional<String> literal_value; // absent => bare form, no `= <literal>`
};

// `param <name> : dyn <dtype>` -- a per-neuron dynamic array (IR spec §2.1).
struct ParamDynamicDirective {
    String name;
    String dtype;
};

// `state <name> : <dtype>` -- a per-neuron cell-state-vector slot (IR spec §2).
//
// `initial_value` (ticket #50 [B2], an additive gap fix, not part of the
// locked v1.0 text): the IR spec's `.alloc` is explicitly "interpreted at
// init" (§2), and arch §3.2 says `OnStart` "seeds the state arrays... at
// INIT" -- but before this ticket nothing in the IR could carry that seed
// anywhere; TickProgram has no INIT-stage hook (arch's stage 9 has none) and
// StateDirective had no value slot. This is the natural, minimal home for it:
// the raw LEMS `OnStart` `StateAssignment` value text (a bare identifier like
// `EL` or a numeric literal like `0`), left uninterpreted here -- resolving it
// to a concrete seed (param lookup vs. literal parse) is the future
// allocator's job (#5), matching `.alloc`'s existing "engine-interpreted, not
// compiled" fate. Absent (nullopt) => no OnStart declared for this state
// variable; the allocator defaults it (Phase-1 convention: 0).
struct StateDirective {
    String name;
    String dtype;
    std::optional<String> initial_value;
};

// `accum <name> : <dtype>` -- a per-neuron synapse accumulator feeding
// `network_inputs` (IR spec §2, arch §4.2).
struct AccumDirective {
    String name;
    String dtype;
};

// `peredge <name>` -- a per-edge WeightMatrix variable (IR spec §2, arch §4.3).
struct PeredgeDirective {
    String name;
};

// `regime <name>` -- a per-neuron regime index (IR spec §2, arch §4.5).
struct RegimeDirective {
    String name;
};

// `expose <name>` -- a recordable quantity (IR spec §2, arch §4.6).
struct ExposeDirective {
    String name;
};

// `require <name> from <scope>` -- a read-only binding from another
// component's scope; no allocation (IR spec §2).
struct RequireDirective {
    String name;
    String scope;
};

using AllocDirective = std::variant<
    ParamConstantDirective,
    ParamDynamicDirective,
    StateDirective,
    AccumDirective,
    PeredgeDirective,
    RegimeDirective,
    ExposeDirective,
    RequireDirective>;

// ── .tick operands (IR spec §3.1) ────────────────────────────────────────
//
// Every operand is a plain name (a model quantity, a `t0,t1,...`
// intermediate, or a reserved engine read `dt`/`tick`/`network_inputs`) --
// represented here simply as String. There are no registers.

// What a `forall`'s `<set>` (§3.4), or a `loadedge`/`accedge` op's `@<edge>`
// (§3.3), refers to. `forall` only ever takes NeuronIn/NeuronOut (the
// current neuron's in-/out-edges); an edge op can additionally use
// CurrentEdge (`@edge` -- the edge bound by an enclosing `forall` body, or,
// inside an `onevent` handler, the specific edge the spike arrived on --
// IR spec §4's expOne/NMDA both write `accedge g@edge, weight` inside
// `onevent`), or take NeuronIn/NeuronOut directly with no `forall` wrapper
// for a whole-set scatter/gather (e.g. `accedge g@neuron_out, weight`).
enum class EdgeSetReference { CurrentEdge, NeuronIn, NeuronOut };

// Two-operand arithmetic/math/compare/boolean ops sharing the `dst,a,b`
// shape (IR spec §3.3): add/sub/mul/div/mod, pow/min/max/expdecay (for
// expdecay, operand_b is the time constant `tau`), gt/lt/ge/le/eq/ne
// (dst is boolean-named), and/or.
enum class BinaryOpcode { Add, Sub, Mul, Div, Mod, Pow, Min, Max, ExpDecay, Gt, Lt, Ge, Le, Eq, Ne, And, Or };

struct BinaryInstruction {
    BinaryOpcode opcode;
    String destination;
    String operand_a;
    String operand_b;
};

// One-operand arithmetic/math/boolean ops sharing the `dst,a` shape (IR
// spec §3.3): neg, exp/log/sqrt/abs/floor/ceil, sin/cos/tan/sinh/cosh/tanh,
// not.
enum class UnaryOpcode { Neg, Exp, Log, Sqrt, Abs, Floor, Ceil, Sin, Cos, Tan, Sinh, Cosh, Tanh, Not };

struct UnaryInstruction {
    UnaryOpcode opcode;
    String destination;
    String operand_a;
};

// `fma dst,a,b,c` (`a*b+c`) -- the one three-operand arithmetic op (IR spec §3.3).
struct FusedMultiplyAddInstruction {
    String destination;
    String operand_a;
    String operand_b;
    String operand_c;
};

// `rand dst` (uniform [0,1)) / `randn dst` (standard normal) -- per-thread
// RNG reads (IR spec §3.3).
enum class RandomOpcode { Rand, Randn };

struct RandomInstruction {
    RandomOpcode opcode;
    String destination;
};

// `mov dst,src` (IR spec §3.3).
struct MoveInstruction {
    String destination;
    String source;
};

// `loadedge dst,<var>@<edge>` -- reads a per-edge variable, reconstructing
// basis+Sk for a `peredge` var (IR spec §3.3).
struct LoadEdgeInstruction {
    String destination;
    String variable_name;
    EdgeSetReference edge_reference = EdgeSetReference::CurrentEdge;
};

// `accedge <var>@<edge>,<value>` -- accumulates into a per-edge (`peredge`)
// or per-neuron (`accum`) variable (IR spec §3.3).
struct AccumulateEdgeInstruction {
    String variable_name;
    EdgeSetReference edge_reference = EdgeSetReference::CurrentEdge;
    String value;
};

// `emit <port>` (IR spec §3.3).
struct EmitInstruction {
    String port_name;
};

// `set_regime <name>,<value>` (IR spec §3.3).
struct SetRegimeInstruction {
    String regime_name;
    String value;
};

// Forward-declared so the control constructs below can hold nested
// instruction lists (Vector<TickInstruction> as an incomplete-type member is
// well-formed as of C++17; TickInstruction itself is defined below, once
// every control construct it can hold is already complete).
struct TickInstruction;

// One `elif <condition> { ... }` branch inside an `if` chain (IR spec §3.4).
struct ElseIfBranch {
    String condition;
    Vector<TickInstruction> body;
};

// `if <condition> { ... } [elif <condition> { ... }]... [else { ... }]` --
// the single branching construct (IR spec §3.4). `condition` is a
// boolean-named operand (the destination of a compare/boolean op).
struct IfInstruction {
    String condition;
    Vector<TickInstruction> then_body;
    Vector<ElseIfBranch> else_if_branches;
    std::optional<Vector<TickInstruction>> else_body; // absent => no `else`
};

// `forall <set> { ... }` -- iterates the current neuron's in- or out-edges;
// `@edge` inside the body refers to the edge under iteration (IR spec §3.4).
struct ForAllInstruction {
    EdgeSetReference edge_set = EdgeSetReference::NeuronIn; // only NeuronIn/NeuronOut are valid here
    Vector<TickInstruction> body;
};

// `onevent <port> { ... }` -- a spike-arrival handler; lives in `@deliver`,
// its body runs when a spike is delivered on `<port>` (IR spec §3.4).
struct OnEventInstruction {
    String port_name;
    Vector<TickInstruction> body;
};

using TickInstructionVariant = std::variant<
    BinaryInstruction,
    UnaryInstruction,
    FusedMultiplyAddInstruction,
    RandomInstruction,
    MoveInstruction,
    LoadEdgeInstruction,
    AccumulateEdgeInstruction,
    EmitInstruction,
    SetRegimeInstruction,
    IfInstruction,
    ForAllInstruction,
    OnEventInstruction>;

// One `.tick` instruction: either a single-op leaf or a control construct,
// per the variant above. Provides a converting constructor so a concrete
// instruction (e.g. `BinaryInstruction{...}`) can be used directly wherever
// a TickInstruction is expected (test/lowering call sites build lists of
// these).
struct TickInstruction {
    TickInstructionVariant operation;

    TickInstruction() = default;

    template <typename ConcreteInstruction>
    TickInstruction(ConcreteInstruction concrete_instruction) : operation(std::move(concrete_instruction)) {}
};

// ── .tick stage tags (IR spec §3.2) ──────────────────────────────────────
//
// A program's `.tick` section, organized by the 8 stage tags a per-type
// program may supply hooks for (stages 1/6/8/9 -- deliver-drain,
// propagate-scatter, record, advance -- are otherwise engine-owned; a
// program only ever supplies the per-type body run within them).
struct TickProgram {
    Vector<TickInstruction> deliver;
    Vector<TickInstruction> integrate;
    Vector<TickInstruction> detect;
    Vector<TickInstruction> emit;
    Vector<TickInstruction> reset;
    Vector<TickInstruction> propagate;
    Vector<TickInstruction> plasticity;
    Vector<TickInstruction> record;
};

// One per-ComponentType IR program: its `.alloc` directives plus its
// `.tick` section (IR spec §1). `component_type_name` is metadata for
// whoever holds a collection of these (e.g. one per cell/synapse type in a
// model) -- it is not part of the printed `.alloc`/`.tick` text itself
// (IR spec §4's examples never print a type name inline).
struct IrProgram {
    String component_type_name;
    Vector<AllocDirective> alloc;
    TickProgram tick;
};

// Renders `program` back to the IR text form shown in the locked spec's §4
// examples: `.alloc` then `.tick`, stage tags in canonical order
// (`@deliver @integrate @detect @emit @reset @propagate @plasticity
// @record`), 2 spaces per nesting level, control-construct bodies always
// expanded one instruction per line (a single, consistent formatting choice
// -- the spec's own examples inline some single/multi-instruction bodies on
// one line, which is a whitespace-only difference from this printer's
// output). A debug/round-trip printer only -- not part of the
// compiled-to-GPU-source path (ticket #55 [C1]).
String print_ir_program(const IrProgram &program);

} // namespace spikecorec::nml
