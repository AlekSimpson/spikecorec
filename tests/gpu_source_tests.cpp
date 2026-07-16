#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>

#include "spikecorec/nml/gpu_source.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── IR `.tick` -> GPU source tests (ticket #55 [C1]) ─────────────────────
//
// The four fixtures below (GLIF1, expOne, NMDA, iafRefCell) are hand-built to match
// docs/nml_ir_spec.md §4's four worked examples EXACTLY as tests/ir_tests.cpp (ticket #4,
// already on this branch) already constructs them -- reusing those already-reviewed,
// self-consistent constructions directly (rather than tests/ir_tests.cpp's own "OpCoverage"
// fixture, whose dummy operands like `a`/`b`/`c`/`is_ref` are never declared -- fine for that
// file's pure print_ir_program round-trip check, but not valid input for this ticket's lowering,
// which validates that every operand actually resolves to something real). Two more fixtures
// (OpCoverageProbe, ScatterProbe) are original to this file, exercising leaf ops/alloc forms/the
// whole-set accedge scatter form none of the four spec examples happen to use.
//
// Every generated MSL source is ACTUALLY compiled via `xcrun -sdk macosx metal -c ... -o
// /dev/null` (mirrors the Makefile's own METALC invocation) -- real, verifiable proof of the
// "emits compiling MSL" acceptance criterion, not just an assertion. The CUDA source is only
// checked structurally (no CUDA toolkit on this machine -- see the ticket's own environment
// constraint); it is verified by careful reading against src/cuda/kernels.cu, not by compiling.

namespace {

// Writes `source` to a temp .metal file and actually compiles it with the real Metal compiler,
// discarding the .air output -- returns true iff compilation succeeded.
bool compiles_as_msl(const String &source, const String &label) {
    String path = "/tmp/spikecorec_gpu_source_test_" + label + ".metal";
    ofstream file(path);
    file << source;
    file.close();
    String command = "xcrun -sdk macosx metal -O2 -c " + path + " -o /dev/null";
    return system(command.c_str()) == 0;
}

IrProgram build_glif1() {
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
    return program;
}

IrProgram build_exp_one() {
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
    return program;
}

IrProgram build_nmda() {
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
    return program;
}

IrProgram build_iaf_ref_cell() {
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
    return program;
}

// Original to this file: exercises every leaf op the four spec examples above don't (neg, fma,
// mod, exp, log, pow, sqrt, abs, floor, ceil, sin/cos/tan/sinh/cosh/tanh, min/max, rand/randn,
// lt/le/ge/ne, not, or, a baked `param = <literal>`, `param : dyn`, `expose`, `require`, and an
// `elif` branch) -- every operand is properly declared/assigned, unlike ir_tests.cpp's
// OpCoverage fixture.
IrProgram build_op_coverage_probe() {
    IrProgram program;
    program.component_type_name = "OpCoverageProbe";
    program.alloc = {
        StateDirective{"v", "f32"},
        ParamConstantDirective{"scale", String("2.0")},
        ParamDynamicDirective{"heterogeneous_offset", "f32"},
        ExposeDirective{"trace"},
        RequireDirective{"neighbor_v", "somewhere"},
    };
    program.tick.integrate = {
        UnaryInstruction{UnaryOpcode::Neg, "t0", "v"},
        FusedMultiplyAddInstruction{"t_fma", "v", "scale", "t0"},
        BinaryInstruction{BinaryOpcode::Mod, "t1", "v", "scale"},
        BinaryInstruction{BinaryOpcode::Pow, "t2", "t1", "2"},
        UnaryInstruction{UnaryOpcode::Sqrt, "t3", "t2"},
        UnaryInstruction{UnaryOpcode::Abs, "t4", "t0"},
        UnaryInstruction{UnaryOpcode::Floor, "t4b", "t4"},
        UnaryInstruction{UnaryOpcode::Ceil, "t5", "t4b"},
        UnaryInstruction{UnaryOpcode::Exp, "t5b", "t5"},
        UnaryInstruction{UnaryOpcode::Log, "t5c", "t5b"},
        UnaryInstruction{UnaryOpcode::Sin, "t6", "t5"},
        UnaryInstruction{UnaryOpcode::Cos, "t7", "t5"},
        UnaryInstruction{UnaryOpcode::Tan, "t8", "t5"},
        UnaryInstruction{UnaryOpcode::Sinh, "t9", "t5"},
        UnaryInstruction{UnaryOpcode::Cosh, "t10", "t5"},
        UnaryInstruction{UnaryOpcode::Tanh, "t11", "t5"},
        BinaryInstruction{BinaryOpcode::Min, "t12", "t3", "heterogeneous_offset"},
        BinaryInstruction{BinaryOpcode::Max, "t13", "t3", "heterogeneous_offset"},
        RandomInstruction{RandomOpcode::Rand, "t14"},
        RandomInstruction{RandomOpcode::Randn, "t15"},
        BinaryInstruction{BinaryOpcode::Lt, "is_low", "t12", "t13"},
        BinaryInstruction{BinaryOpcode::Le, "is_low_or_equal", "t12", "t13"},
        BinaryInstruction{BinaryOpcode::Ge, "is_ge", "t12", "t13"},
        BinaryInstruction{BinaryOpcode::Ne, "is_different", "t12", "t13"},
        UnaryInstruction{UnaryOpcode::Not, "is_not_low", "is_low"},
        BinaryInstruction{BinaryOpcode::Or, "any_flag", "is_low_or_equal", "is_different"},
        MoveInstruction{"trace", "neighbor_v"},
        IfInstruction{
            "is_not_low",
            {MoveInstruction{"v", "t14"}},
            {ElseIfBranch{"any_flag", {MoveInstruction{"v", "t15"}}}},
            Vector<TickInstruction>{MoveInstruction{"v", "is_ge"}},
        },
    };
    return program;
}

// Original to this file: the whole-set `accedge <var>@neuron_out, value` scatter form (ir_spec.md
// §3.4), with no enclosing `forall` -- exercised once for an `accum` target and once for a
// `peredge` target, proving the implicit-forall-wrapping code path compiles for both.
IrProgram build_scatter_probe() {
    IrProgram program;
    program.component_type_name = "ScatterProbe";
    program.alloc = {
        AccumDirective{"downstream_accumulator", "f32"},
        PeredgeDirective{"downstream_trace"},
        ParamConstantDirective{"scatter_value", nullopt},
    };
    program.tick.propagate = {
        AccumulateEdgeInstruction{"downstream_accumulator", EdgeSetReference::NeuronOut, "scatter_value"},
        AccumulateEdgeInstruction{"downstream_trace", EdgeSetReference::NeuronOut, "scatter_value"},
    };
    return program;
}

// Regression fixture (review finding on ticket #55): `loadedge` on a `peredge` variable directly
// inside an `onevent` block, not just `accedge` -- e.g. a Mg-block-style synapse that reads its own
// current value on spike arrival before updating it. `gpu_source.h` documents `onevent` as a valid
// `loadedge` context; this is the case that needs the shared-basis buffers (`U`/`V`/
// `rank_float4_stride`/`coefficients_g`) surfaced on the deliver function even though no `forall`/
// k^2-tree walk is involved (the edge is already known from the delivered spike).
IrProgram build_loadedge_in_deliver_probe() {
    IrProgram program;
    program.component_type_name = "LoadEdgeInDeliverProbe";
    program.alloc = {
        PeredgeDirective{"g"},
        ParamConstantDirective{"weight", nullopt},
    };
    program.tick.deliver = {
        OnEventInstruction{
            "in",
            {
                LoadEdgeInstruction{"t0", "g", EdgeSetReference::CurrentEdge},
                BinaryInstruction{BinaryOpcode::Add, "t1", "t0", "weight"},
                AccumulateEdgeInstruction{"g", EdgeSetReference::CurrentEdge, "t1"},
            },
        },
    };
    return program;
}

} // namespace

TEST(GpuSource, glif1_leaky_integrate_and_fire_compiles) {
    GpuSource source = lower_ir_program_to_gpu_source(build_glif1());
    EXPECT_NE(source.msl_source.find("kernel void GLIF1_tick("), String::npos);
    EXPECT_NE(source.msl_source.find("network_inputs[neuron_index]"), String::npos);
    EXPECT_NE(source.msl_source.find("emit_spike[neuron_index] = true;"), String::npos);
    EXPECT_TRUE(compiles_as_msl(source.msl_source, "glif1"));

    EXPECT_NE(source.cuda_source.find("__global__ void GLIF1_tick("), String::npos);
}

TEST(GpuSource, exp_one_current_based_aggregatable_synapse_compiles) {
    GpuSource source = lower_ir_program_to_gpu_source(build_exp_one());
    EXPECT_NE(source.msl_source.find("kernel void expOne_tick("), String::npos);
    EXPECT_NE(source.msl_source.find("kernel void expOne_deliver_in("), String::npos);
    EXPECT_NE(source.msl_source.find("g[target_node] += weight;"), String::npos);
    EXPECT_TRUE(compiles_as_msl(source.msl_source, "exp_one"));

    EXPECT_NE(source.cuda_source.find("__global__ void expOne_deliver_in("), String::npos);
}

TEST(GpuSource, nmda_conductance_per_edge_synapse_compiles) {
    GpuSource source = lower_ir_program_to_gpu_source(build_nmda());
    EXPECT_NE(source.msl_source.find("kernel void NMDA_tick("), String::npos);
    EXPECT_NE(source.msl_source.find("kernel void NMDA_deliver_in("), String::npos);
    EXPECT_NE(source.msl_source.find("k2t_find_nth_neighbor"), String::npos);
    EXPECT_NE(source.msl_source.find("sparse_delta_g[(long)source_node * max_neighbor_count + edge_slot] += weight;"),
              String::npos);
    EXPECT_NE(source.msl_source.find("edge_reconstruction += dot(edge_basis_u_row[lane], "
                                      "lane_coefficients * edge_basis_v_row[lane]);"),
              String::npos);
    EXPECT_TRUE(compiles_as_msl(source.msl_source, "nmda"));

    EXPECT_NE(source.cuda_source.find("__global__ void NMDA_tick("), String::npos);
    EXPECT_NE(source.cuda_source.find("u4.x * (lane_coefficients[0] * v4.x)"), String::npos);
}

TEST(GpuSource, iaf_ref_cell_refractory_regime_compiles) {
    GpuSource source = lower_ir_program_to_gpu_source(build_iaf_ref_cell());
    EXPECT_NE(source.msl_source.find("kernel void iafRefCell_tick("), String::npos);
    EXPECT_NE(source.msl_source.find("r[neuron_index] = 1;"), String::npos);
    EXPECT_TRUE(compiles_as_msl(source.msl_source, "iaf_ref_cell"));

    EXPECT_NE(source.cuda_source.find("__global__ void iafRefCell_tick("), String::npos);
}

TEST(GpuSource, op_coverage_probe_compiles) {
    GpuSource source = lower_ir_program_to_gpu_source(build_op_coverage_probe());
    EXPECT_NE(source.msl_source.find("kernel void OpCoverageProbe_tick("), String::npos);
    EXPECT_NE(source.msl_source.find("spikecorec_rand_uniform(rng_state[neuron_index]);"), String::npos);
    EXPECT_NE(source.msl_source.find("spikecorec_randn(rng_state[neuron_index]);"), String::npos);
    EXPECT_NE(source.msl_source.find("const float scale = 2.0;"), String::npos);
    EXPECT_TRUE(compiles_as_msl(source.msl_source, "op_coverage_probe"));

    EXPECT_NE(source.cuda_source.find("spikecorec_randn(rng_state[neuron_index]);"), String::npos);
}

TEST(GpuSource, scatter_probe_whole_set_accedge_compiles) {
    GpuSource source = lower_ir_program_to_gpu_source(build_scatter_probe());
    EXPECT_NE(source.msl_source.find("kernel void ScatterProbe_tick("), String::npos);
    EXPECT_NE(source.msl_source.find("downstream_accumulator[forall_neighbor_node_0] += scatter_value;"),
              String::npos);
    EXPECT_NE(source.msl_source.find("sparse_delta_downstream_trace[(long)neuron_index * max_neighbor_count + "
                                      "forall_slot_1] += scatter_value;"),
              String::npos);
    EXPECT_TRUE(compiles_as_msl(source.msl_source, "scatter_probe"));

    EXPECT_NE(source.cuda_source.find("downstream_accumulator[forall_neighbor_node_0] += scatter_value;"),
              String::npos);
}

// Regression test (review finding on ticket #55): a `loadedge` directly inside an `onevent` body
// (no `forall`, no k^2-tree walk -- the edge is already known as source_node/target_node/edge_slot)
// must still get the shared-basis buffers the reconstruction needs. Before the fix,
// generate_deliver_function hardcoded needs_edge_walk_block=false unconditionally, so this would
// have emitted a reference to undeclared U/V/coefficients_g/rank_float4_stride.
TEST(GpuSource, loadedge_inside_onevent_deliver_body_compiles) {
    GpuSource source = lower_ir_program_to_gpu_source(build_loadedge_in_deliver_probe());
    EXPECT_NE(source.msl_source.find("kernel void LoadEdgeInDeliverProbe_deliver_in("), String::npos);
    EXPECT_NE(source.msl_source.find("const device float4 *U"), String::npos);
    EXPECT_NE(source.msl_source.find("const device float4 *V"), String::npos);
    EXPECT_NE(source.msl_source.find("const device float *coefficients_g"), String::npos);
    EXPECT_NE(source.msl_source.find("constant long &rank_float4_stride"), String::npos);
    EXPECT_NE(source.msl_source.find("edge_basis_u_row = U + (long)source_node * rank_float4_stride;"),
              String::npos);
    EXPECT_NE(source.msl_source.find("edge_basis_v_row = V + (long)target_node * rank_float4_stride;"),
              String::npos);
    EXPECT_NE(source.msl_source.find("sparse_delta_g[(long)source_node * max_neighbor_count + edge_slot] += t1;"),
              String::npos);
    EXPECT_TRUE(compiles_as_msl(source.msl_source, "loadedge_in_deliver"));

    EXPECT_NE(source.cuda_source.find("__global__ void LoadEdgeInDeliverProbe_deliver_in("), String::npos);
    EXPECT_NE(source.cuda_source.find("const float4 *U"), String::npos);
    EXPECT_NE(source.cuda_source.find("const float4 *V"), String::npos);
}

TEST(GpuSource, peredge_variable_used_as_plain_operand_is_rejected) {
    IrProgram program;
    program.component_type_name = "Invalid";
    program.alloc = {PeredgeDirective{"g"}};
    program.tick.integrate = {MoveInstruction{"t0", "g"}};

    EXPECT_THROW(lower_ir_program_to_gpu_source(program), std::runtime_error);
}

TEST(GpuSource, undeclared_local_read_before_assignment_is_rejected) {
    IrProgram program;
    program.component_type_name = "Invalid";
    program.tick.integrate = {MoveInstruction{"v", "never_declared"}};

    EXPECT_THROW(lower_ir_program_to_gpu_source(program), std::runtime_error);
}
