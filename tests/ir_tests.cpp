#include <gtest/gtest.h>

#include "spikecorec/nml/ir.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── IR in-engine representation tests (ticket #4 [B1]) ──────────────────
//
// Constructs each of docs/nml_ir_spec.md §4's four worked examples via the
// ir.h API exactly as a C++ engineer would (aggregate-initialized structs),
// then asserts print_ir_program reproduces the spec's text -- normalized in
// two clearly-equivalent, judgment-call ways applied consistently
// throughout (see ir.h's print_ir_program doc comment):
//   1. a bare `param a, b, c` line (no `=`/`: dyn`) is printed one name per
//      line rather than comma-joined on one line (whitespace-only).
//   2. every control-construct body (`if`/`elif`/`else`/`forall`/`onevent`)
//      is always expanded one instruction per line, even when the spec's
//      own prose inlines a single- or multi-instruction body on one line
//      (whitespace-only -- same instructions, same order, same nesting).
// Neither normalization changes instruction order, operand names, opcodes,
// or nesting structure.

TEST(Ir, constructs_and_prints_glif1_leaky_integrate_and_fire) {
    IrProgram program;
    program.component_type_name = "GLIF1";
    program.alloc = {
        StateDirective{"v", "f32"},
        ParamConstantDirective{"C", nullopt},
        ParamConstantDirective{"gL", nullopt},
        ParamConstantDirective{"EL", nullopt},
        ParamConstantDirective{"vth", nullopt},
        ParamConstantDirective{"vreset", nullopt},
    };
    program.tick.integrate = {
        BinaryInstruction{BinaryOpcode::Sub, "t0", "EL", "v"},
        BinaryInstruction{BinaryOpcode::Mul, "t0", "gL", "t0"},
        BinaryInstruction{BinaryOpcode::Add, "t0", "network_inputs", "t0"},
        BinaryInstruction{BinaryOpcode::Div, "t0", "t0", "C"},
        BinaryInstruction{BinaryOpcode::Mul, "t0", "t0", "dt"},
        BinaryInstruction{BinaryOpcode::Add, "v", "v", "t0"},
    };
    program.tick.detect = {
        BinaryInstruction{BinaryOpcode::Gt, "spiked", "v", "vth"},
    };
    program.tick.emit = {
        IfInstruction{"spiked", {EmitInstruction{"spike"}}, {}, nullopt},
    };
    program.tick.reset = {
        IfInstruction{"spiked", {MoveInstruction{"v", "vreset"}}, {}, nullopt},
    };

    String expected =
        ".alloc\n"
        "  state v : f32\n"
        "  param C\n"
        "  param gL\n"
        "  param EL\n"
        "  param vth\n"
        "  param vreset\n"
        ".tick\n"
        "  @integrate\n"
        "    sub t0, EL, v\n"
        "    mul t0, gL, t0\n"
        "    add t0, network_inputs, t0\n"
        "    div t0, t0, C\n"
        "    mul t0, t0, dt\n"
        "    add v, v, t0\n"
        "  @detect\n"
        "    gt spiked, v, vth\n"
        "  @emit\n"
        "    if spiked {\n"
        "      emit spike\n"
        "    }\n"
        "  @reset\n"
        "    if spiked {\n"
        "      mov v, vreset\n"
        "    }\n";

    EXPECT_EQ(print_ir_program(program), expected);
}

TEST(Ir, constructs_and_prints_exp_one_current_based_aggregatable_synapse) {
    IrProgram program;
    program.component_type_name = "expOne";
    program.alloc = {
        AccumDirective{"g", "f32"},
        ParamConstantDirective{"tau", nullopt},
        ParamConstantDirective{"weight", nullopt},
    };
    program.tick.deliver = {
        OnEventInstruction{"in", {AccumulateEdgeInstruction{"g", EdgeSetReference::CurrentEdge, "weight"}}},
    };
    program.tick.integrate = {
        BinaryInstruction{BinaryOpcode::ExpDecay, "g", "g", "tau"},
        BinaryInstruction{BinaryOpcode::Add, "network_inputs", "network_inputs", "g"},
    };

    String expected =
        ".alloc\n"
        "  accum g : f32\n"
        "  param tau\n"
        "  param weight\n"
        ".tick\n"
        "  @deliver\n"
        "    onevent in {\n"
        "      accedge g@edge, weight\n"
        "    }\n"
        "  @integrate\n"
        "    expdecay g, g, tau\n"
        "    add network_inputs, network_inputs, g\n";

    EXPECT_EQ(print_ir_program(program), expected);
}

TEST(Ir, constructs_and_prints_nmda_conductance_per_edge_synapse) {
    IrProgram program;
    program.component_type_name = "NMDA";
    program.alloc = {
        PeredgeDirective{"g"},
        RequireDirective{"v", "postsynaptic"},
        ParamConstantDirective{"tau", nullopt},
        ParamConstantDirective{"weight", nullopt},
        ParamConstantDirective{"erev", nullopt},
    };
    program.tick.deliver = {
        OnEventInstruction{"in", {AccumulateEdgeInstruction{"g", EdgeSetReference::CurrentEdge, "weight"}}},
    };
    program.tick.integrate = {
        ForAllInstruction{
            EdgeSetReference::NeuronIn,
            {
                LoadEdgeInstruction{"t0", "g", EdgeSetReference::CurrentEdge},
                BinaryInstruction{BinaryOpcode::Sub, "t1", "erev", "v"},
                BinaryInstruction{BinaryOpcode::Mul, "t0", "t0", "t1"},
                BinaryInstruction{BinaryOpcode::Add, "network_inputs", "network_inputs", "t0"},
            },
        },
    };

    String expected =
        ".alloc\n"
        "  peredge g\n"
        "  require v from postsynaptic\n"
        "  param tau\n"
        "  param weight\n"
        "  param erev\n"
        ".tick\n"
        "  @deliver\n"
        "    onevent in {\n"
        "      accedge g@edge, weight\n"
        "    }\n"
        "  @integrate\n"
        "    forall neuron_in {\n"
        "      loadedge t0, g@edge\n"
        "      sub t1, erev, v\n"
        "      mul t0, t0, t1\n"
        "      add network_inputs, network_inputs, t0\n"
        "    }\n";

    EXPECT_EQ(print_ir_program(program), expected);
}

// docs/nml_ir_spec.md §4's refractory example is explicitly a "sketch": its
// `.alloc` elides some param names as `...`, and its `if is_ref { ... }
// else { ... }` bodies are prose ("hold v, count down" / "normal v
// integration"), not real instructions. Judgment call: the elided params
// are filled in as C/gL/EL (matching GLIF1 -- iafRefCell is LIF+refractory,
// same base params); the refractory branch is modeled literally as "hold v"
// (`mov v, v`, no invented counter state since the sketch's `.alloc` never
// declares one); the normal branch reuses GLIF1's own integration sequence.
// The surrounding skeleton this test actually checks -- regime dispatch via
// eq/if-else, the eq/gt/and detect chain, set_regime on reset -- is exactly
// the spec's.
TEST(Ir, constructs_and_prints_refractory_regime_state_machine) {
    IrProgram program;
    program.component_type_name = "iafRefCell";
    program.alloc = {
        StateDirective{"v", "f32"},
        RegimeDirective{"r"},
        ParamConstantDirective{"C", nullopt},
        ParamConstantDirective{"gL", nullopt},
        ParamConstantDirective{"EL", nullopt},
        ParamConstantDirective{"vth", nullopt},
        ParamConstantDirective{"vreset", nullopt},
        ParamConstantDirective{"t_ref", nullopt},
    };
    program.tick.integrate = {
        BinaryInstruction{BinaryOpcode::Eq, "is_ref", "r", "1"},
        IfInstruction{
            "is_ref",
            {MoveInstruction{"v", "v"}},
            {},
            Vector<TickInstruction>{
                BinaryInstruction{BinaryOpcode::Sub, "t0", "EL", "v"},
                BinaryInstruction{BinaryOpcode::Mul, "t0", "gL", "t0"},
                BinaryInstruction{BinaryOpcode::Add, "t0", "network_inputs", "t0"},
                BinaryInstruction{BinaryOpcode::Div, "t0", "t0", "C"},
                BinaryInstruction{BinaryOpcode::Mul, "t0", "t0", "dt"},
                BinaryInstruction{BinaryOpcode::Add, "v", "v", "t0"},
            },
        },
    };
    program.tick.detect = {
        BinaryInstruction{BinaryOpcode::Eq, "is_int", "r", "0"},
        BinaryInstruction{BinaryOpcode::Gt, "over", "v", "vth"},
        BinaryInstruction{BinaryOpcode::And, "fire", "is_int", "over"},
    };
    program.tick.emit = {
        IfInstruction{"fire", {EmitInstruction{"spike"}}, {}, nullopt},
    };
    program.tick.reset = {
        IfInstruction{"fire", {MoveInstruction{"v", "vreset"}, SetRegimeInstruction{"r", "1"}}, {}, nullopt},
    };

    String expected =
        ".alloc\n"
        "  state v : f32\n"
        "  regime r\n"
        "  param C\n"
        "  param gL\n"
        "  param EL\n"
        "  param vth\n"
        "  param vreset\n"
        "  param t_ref\n"
        ".tick\n"
        "  @integrate\n"
        "    eq is_ref, r, 1\n"
        "    if is_ref {\n"
        "      mov v, v\n"
        "    } else {\n"
        "      sub t0, EL, v\n"
        "      mul t0, gL, t0\n"
        "      add t0, network_inputs, t0\n"
        "      div t0, t0, C\n"
        "      mul t0, t0, dt\n"
        "      add v, v, t0\n"
        "    }\n"
        "  @detect\n"
        "    eq is_int, r, 0\n"
        "    gt over, v, vth\n"
        "    and fire, is_int, over\n"
        "  @emit\n"
        "    if fire {\n"
        "      emit spike\n"
        "    }\n"
        "  @reset\n"
        "    if fire {\n"
        "      mov v, vreset\n"
        "      set_regime r, 1\n"
        "    }\n";

    EXPECT_EQ(print_ir_program(program), expected);
}

// Coverage for op families and alloc directive forms none of the four §4
// examples happen to exercise (fma, rand/randn, floor/ceil/min/max/not,
// elif, `param : dyn`, `expose`) -- one instruction/directive of each, not
// a fifth worked example.
TEST(Ir, prints_every_remaining_leaf_op_and_alloc_directive_form) {
    IrProgram program;
    program.component_type_name = "OpCoverage";
    program.alloc = {
        ParamDynamicDirective{"heterogeneous_scale", "f32"},
        ExposeDirective{"v"},
    };
    program.tick.integrate = {
        FusedMultiplyAddInstruction{"t0", "a", "b", "c"},
        RandomInstruction{RandomOpcode::Rand, "t1"},
        RandomInstruction{RandomOpcode::Randn, "t2"},
        UnaryInstruction{UnaryOpcode::Floor, "t3", "t0"},
        BinaryInstruction{BinaryOpcode::Min, "t4", "t0", "t1"},
        BinaryInstruction{BinaryOpcode::Max, "t5", "t0", "t1"},
        UnaryInstruction{UnaryOpcode::Not, "t6", "is_ref"},
        IfInstruction{
            "t6",
            {EmitInstruction{"spike"}},
            {ElseIfBranch{"t4", {EmitInstruction{"spike"}}}},
            Vector<TickInstruction>{MoveInstruction{"v", "vreset"}},
        },
    };

    String expected =
        ".alloc\n"
        "  param heterogeneous_scale : dyn f32\n"
        "  expose v\n"
        ".tick\n"
        "  @integrate\n"
        "    fma t0, a, b, c\n"
        "    rand t1\n"
        "    randn t2\n"
        "    floor t3, t0\n"
        "    min t4, t0, t1\n"
        "    max t5, t0, t1\n"
        "    not t6, is_ref\n"
        "    if t6 {\n"
        "      emit spike\n"
        "    } elif t4 {\n"
        "      emit spike\n"
        "    } else {\n"
        "      mov v, vreset\n"
        "    }\n";

    EXPECT_EQ(print_ir_program(program), expected);
}
