#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/engine.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/nml/delay_ring.h"
#include "spikecorec/nml/master_kernel.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── Master-kernel assembly + compile + cache + dispatch tests (ticket #6 [C3]) ───────────────────
//
// Two things this file demonstrates, matching the ticket's own acceptance criteria:
//  1. A resolved model (here, hand-built ModelSpecification/IrProgram fixtures -- the SAME
//     established pattern gpu_source_tests.cpp/ir_tests.cpp/allocator_tests.cpp's own
//     `throws_on_non_f32_state_dtype` fixture already use, rather than going through the full
//     NML/LEMS parse+resolve pipeline) compiles to one runnable master kernel on BOTH backends:
//     every generated MSL fixture is ACTUALLY compiled via `xcrun -sdk macosx metal -c` (mirrors
//     gpu_source_tests.cpp's own `compiles_as_msl` helper); the CUDA text is checked structurally
//     only (no CUDA toolchain on this machine -- the ticket's own documented constraint).
//  2. The assembled master kernel reproduces the current hardcoded LIF cell (src/metal/kernels.metal
//     `step`/`step_no_active_optimization`, dispatched via SpikeEngine::step_simulation) as a
//     special case: a hand-built GLIF1-shaped ComponentType, parameterized so its forward-Euler
//     integrate step is algebraically IDENTICAL to the hardcoded engine's own `apply_decay` +
//     direct-input-add recurrence (C=dt=1, gL=decay_rate, EL=resting_mp, vth=spike_threshold),
//     run through AssembledModel::step_tick side-by-side with the real SpikeEngine over several
//     ticks, comparing membrane-potential trajectory, spike timing, and propagated
//     network_inputs -- an actual behavioral-equivalence test, not just a code-reading argument.
//     `spike_period` is set beyond the test's tick horizon and the IR fixture's own `@reset` stage
//     is left empty to sidestep the hardcoded engine's own idiosyncratic "periodic forced reset
//     every spike_period ticks since the last spike" mechanism (not an instantaneous per-spike
//     reset, and not expressible via the locked IR's OnCondition/StateAssignment semantics without
//     inventing a new construct) -- see master_kernel.h's own header comment for why this is this
//     ticket's scope, not a gap it silently papers over.

namespace {

// Writes `source` to a temp .metal file and actually compiles it with the real Metal compiler,
// discarding the .air output -- returns true iff compilation succeeded. Mirrors
// gpu_source_tests.cpp's own `compiles_as_msl` helper (kept as its own copy here, matching how
// every other *_tests.cpp file in this tree keeps its own fixture/helper code self-contained).
bool compiles_as_msl(const String &source, const String &label) {
    String path = "/tmp/spikecorec_master_kernel_test_" + label + ".metal";
    ofstream file(path);
    file << source;
    file.close();
    String command = "xcrun -sdk macosx metal -O2 -c " + path + " -o /dev/null";
    return system(command.c_str()) == 0;
}

// Formats `value` with enough significant digits to round-trip exactly through a single-precision
// float parse (9 significant digits is always sufficient for f32) -- used so a baked `.alloc`
// literal (embedded as MSL/CUDA source text) reproduces the exact same bit pattern as the host-side
// f32 it was derived from, which the LIF-equivalence test's exact-match assertions depend on.
String precise_float_literal(f32 value) {
    ostringstream stream;
    stream << std::setprecision(9) << value;
    return stream.str();
}

// A minimal, self-contained GLIF1-shaped cell program: `dv/dt = (gL*(EL-v) + network_inputs) / C`
// (forward Euler, one dt per tick), `spike` on `v > vth`, no `@reset` (see this file's own header
// comment for why the hardcoded-LIF-equivalence test deliberately leaves reset empty).
IrProgram build_lif_equivalent_program(const String &type_name, f32 gL, f32 EL, f32 vth) {
    IrProgram program;
    program.component_type_name = type_name;
    program.alloc = {
        StateDirective{"v", "f32", nullopt},
        ParamConstantDirective{"C", String("1.0")},
        ParamConstantDirective{"gL", precise_float_literal(gL)},
        ParamConstantDirective{"EL", precise_float_literal(EL)},
        ParamConstantDirective{"vth", precise_float_literal(vth)},
    };
    program.tick.integrate = {
        BinaryInstruction{BinaryOpcode::Sub, "t0", "EL", "v"},
        BinaryInstruction{BinaryOpcode::Mul, "t0", "gL", "t0"},
        BinaryInstruction{BinaryOpcode::Add, "t0", "network_inputs", "t0"},
        BinaryInstruction{BinaryOpcode::Div, "t0", "t0", "C"},
        BinaryInstruction{BinaryOpcode::Mul, "t0", "t0", "dt"},
        BinaryInstruction{BinaryOpcode::Add, "v", "v", "t0"},
    };
    program.tick.detect = {BinaryInstruction{BinaryOpcode::Gt, "spiked", "v", "vth"}};
    program.tick.emit = {IfInstruction{"spiked", {EmitInstruction{"spike"}}, {}, nullopt}};
    return program;
}

TypeLibraryEntry build_lif_equivalent_type_entry(const String &type_name, const String &bound_instance_id, f32 C,
                                                  f32 gL, f32 EL, f32 vth) {
    TypeLibraryEntry entry;
    entry.component_type_name = type_name;
    entry.bound_instance_id = bound_instance_id;
    entry.category = TypeLibraryCategory::Cell;
    entry.state_variable_count = 1;
    entry.baked_constants = {{"C", (f64)C}, {"gL", (f64)gL}, {"EL", (f64)EL}, {"vth", (f64)vth}};
    return entry;
}

// ── ticket #62 [F1] fixtures: a synthetic, TEST-ONLY nonlinear cell program ──────────────────────
//
// No real Phase-2 nonlinear point-cell ComponentType (izhikevich/AdEx/...) is lowerable yet
// (ticket #63's own job) -- this hand-built IrProgram is this file's own analogue of
// build_lif_equivalent_program above, extended with one `v * v` self-product term in its own
// `@integrate` (a state variable multiplied by itself -- the same shape a real nonlinear cell's
// quadratic term would take, see cell_lowering_tests.cpp's own
// NONLINEAR_TEST_ONLY_COMPONENT_TYPE) -- this file builds IrPrograms by hand rather than through
// lower_cell_to_ir, so nothing computes cell_dynamics_are_closed_form_advanceable's classification
// automatically the way cell_lowering.cpp does for a real lowered type (see this file's own doc
// comment above for why hand-built IrProgram fixtures are this file's established pattern).
IrProgram build_nonlinear_test_only_program(const String &type_name, f32 gL, f32 EL, f32 vth) {
    IrProgram program;
    program.component_type_name = type_name;
    program.alloc = {
        StateDirective{"v", "f32", nullopt},
        ParamConstantDirective{"C", String("1.0")},
        ParamConstantDirective{"gL", precise_float_literal(gL)},
        ParamConstantDirective{"EL", precise_float_literal(EL)},
        ParamConstantDirective{"vth", precise_float_literal(vth)},
    };
    program.tick.integrate = {
        BinaryInstruction{BinaryOpcode::Sub, "t0", "EL", "v"},
        BinaryInstruction{BinaryOpcode::Mul, "t0", "gL", "t0"},
        BinaryInstruction{BinaryOpcode::Add, "t0", "network_inputs", "t0"},
        BinaryInstruction{BinaryOpcode::Mul, "t1", "v", "v"},
        BinaryInstruction{BinaryOpcode::Sub, "t0", "t0", "t1"},
        BinaryInstruction{BinaryOpcode::Div, "t0", "t0", "C"},
        BinaryInstruction{BinaryOpcode::Mul, "t0", "t0", "dt"},
        BinaryInstruction{BinaryOpcode::Add, "v", "v", "t0"},
    };
    program.tick.detect = {BinaryInstruction{BinaryOpcode::Gt, "spiked", "v", "vth"}};
    program.tick.emit = {IfInstruction{"spiked", {EmitInstruction{"spike"}}, {}, nullopt}};
    return program;
}

} // namespace

// ── compile-failure surfacing (ticket #60 [X1]) ──────────────────────────────────────────────────
//
// compile_kernel itself only reports the raw backend compiler diagnostic (Metal newLibrary's
// NSError text) against a source string the caller never sees again once compile_kernel returns --
// not enough to debug a bad ComponentType lowering without also seeing WHAT was actually emitted.
// This deliberately feeds compile_kernel_or_throw_with_source (the wrapper AssembledModel's
// constructor uses for every kernel it compiles) genuinely invalid Metal source, exercising the
// REAL runtime Metal compiler on this machine (no toolchain stub), and asserts the thrown message
// carries the label, the offending generated source, and the IR dump -- not just "compilation
// failed".
TEST(MasterKernel, compile_kernel_or_throw_with_source_surfaces_label_source_and_ir_on_compile_failure) {
    String bogus_source = "kernel void this_is_not_valid_metal_source( { totally not C++ at all !!! }";

    try {
        compile_kernel_or_throw_with_source(bogus_source, "bogus_function_name",
                                             "ComponentType 'TestBrokenComponentType'", "some ir dump text");
        FAIL() << "expected compile_kernel_or_throw_with_source to throw on invalid GPU source";
    } catch (const std::runtime_error &error) {
        String message = error.what();
        EXPECT_NE(message.find("ComponentType 'TestBrokenComponentType'"), String::npos);
        EXPECT_NE(message.find("bogus_function_name"), String::npos);
        EXPECT_NE(message.find(bogus_source), String::npos);
        EXPECT_NE(message.find("some ir dump text"), String::npos);
    }
}

// The same wrapper called with an empty `ir_dump` (the two engine-fixed scaffold kernels' own
// call sites) must still surface the label and source, just without an "IR program" section.
TEST(MasterKernel, compile_kernel_or_throw_with_source_omits_ir_section_when_ir_dump_is_empty) {
    String bogus_source = "kernel void another_invalid_kernel( { still not valid !!! }";

    try {
        compile_kernel_or_throw_with_source(bogus_source, "bogus_function_name_2",
                                             "the engine-fixed deliver-drain kernel", "");
        FAIL() << "expected compile_kernel_or_throw_with_source to throw on invalid GPU source";
    } catch (const std::runtime_error &error) {
        String message = error.what();
        EXPECT_NE(message.find("the engine-fixed deliver-drain kernel"), String::npos);
        EXPECT_NE(message.find(bogus_source), String::npos);
        EXPECT_EQ(message.find("IR program"), String::npos);
    }
}

// ── acceptance criterion 1: a resolved model compiles to one runnable master kernel ─────────────

TEST(MasterKernel, assembles_a_two_population_two_cell_type_model_and_compiles_every_kernel_as_msl) {
    ModelSpecification model;
    model.total_neuron_count = 5;
    model.type_library.push_back(
        build_lif_equivalent_type_entry("ExcitatoryLifCell", "excInstance", 1.0f, 0.1f, 0.0f, 1.0f));
    model.type_library.push_back(
        build_lif_equivalent_type_entry("InhibitoryLifCell", "inhInstance", 1.0f, 0.2f, 0.0f, 1.2f));

    PopulationEntry excitatory_population;
    excitatory_population.id = "ExcPop";
    excitatory_population.type_library_index = 0;
    excitatory_population.size = 3;
    excitatory_population.neuron_index_begin = 0;
    excitatory_population.neuron_index_end = 3;
    model.populations.push_back(excitatory_population);

    PopulationEntry inhibitory_population;
    inhibitory_population.id = "InhPop";
    inhibitory_population.type_library_index = 1;
    inhibitory_population.size = 2;
    inhibitory_population.neuron_index_begin = 3;
    inhibitory_population.neuron_index_end = 5;
    model.populations.push_back(inhibitory_population);

    spikecorec::Vector<IrProgram> programs = {
        build_lif_equivalent_program("ExcitatoryLifCell", 0.1f, 0.0f, 1.0f),
        build_lif_equivalent_program("InhibitoryLifCell", 0.2f, 0.0f, 1.2f),
    };

    AssembledMasterKernelSource assembled = assemble_master_kernel_source(model, programs);
    ASSERT_EQ(assembled.population_gpu_sources.size(), 2u);

    ASSERT_EQ(assembled.population_gpu_sources[0].functions.size(), 1u);
    EXPECT_EQ(assembled.population_gpu_sources[0].functions[0].function_name, "ExcitatoryLifCell_tick");
    ASSERT_EQ(assembled.population_gpu_sources[1].functions.size(), 1u);
    EXPECT_EQ(assembled.population_gpu_sources[1].functions[0].function_name, "InhibitoryLifCell_tick");

    EXPECT_NE(assembled.population_gpu_sources[0].msl_source.find("kernel void ExcitatoryLifCell_tick("), String::npos);
    EXPECT_NE(assembled.population_gpu_sources[1].msl_source.find("kernel void InhibitoryLifCell_tick("), String::npos);
    EXPECT_NE(assembled.drain_network_inputs_source.msl_source.find(String("kernel void ") + MASTER_KERNEL_DRAIN_NAME + "("),
              String::npos);
    EXPECT_NE(
        assembled.propagate_source.msl_source.find(String("kernel void ") + MASTER_KERNEL_PROPAGATE_NAME + "("),
        String::npos);

    EXPECT_TRUE(compiles_as_msl(assembled.population_gpu_sources[0].msl_source, "excitatory_population"));
    EXPECT_TRUE(compiles_as_msl(assembled.population_gpu_sources[1].msl_source, "inhibitory_population"));
    EXPECT_TRUE(compiles_as_msl(assembled.drain_network_inputs_source.msl_source, "drain"));
    EXPECT_TRUE(compiles_as_msl(assembled.propagate_source.msl_source, "propagate"));

    // CUDA: structural check only -- no CUDA toolchain on this machine (the ticket's own
    // documented testing constraint); mirrors gpu_source_tests.cpp's own approach exactly.
    EXPECT_NE(assembled.population_gpu_sources[0].cuda_source.find("__global__ void ExcitatoryLifCell_tick("),
              String::npos);
    EXPECT_NE(assembled.population_gpu_sources[1].cuda_source.find("__global__ void InhibitoryLifCell_tick("),
              String::npos);
    EXPECT_NE(
        assembled.drain_network_inputs_source.cuda_source.find(String("__global__ void ") + MASTER_KERNEL_DRAIN_NAME + "("),
        String::npos);
    EXPECT_NE(
        assembled.propagate_source.cuda_source.find(String("__global__ void ") + MASTER_KERNEL_PROPAGATE_NAME + "("),
        String::npos);

    // compile + cache: constructing an AssembledModel actually calls compile_kernel for every
    // kernel (Metal newLibrary, on this build) without throwing -- the "compile once" half of the
    // ticket, exercised for real, not just assembled as text.
    EXPECT_NO_THROW({
        AssembledModel assembled_model(model, programs);
        (void)assembled_model;
    });
}

TEST(MasterKernel, a_population_whose_cell_type_has_no_per_neuron_tick_content_is_skipped_not_fatal) {
    ModelSpecification model;
    model.total_neuron_count = 2;

    TypeLibraryEntry empty_entry;
    empty_entry.component_type_name = "EmptyCell";
    empty_entry.bound_instance_id = "emptyInstance";
    empty_entry.category = TypeLibraryCategory::Cell;
    empty_entry.state_variable_count = 0;
    model.type_library.push_back(empty_entry);

    PopulationEntry population;
    population.id = "EmptyPop";
    population.type_library_index = 0;
    population.size = 2;
    population.neuron_index_begin = 0;
    population.neuron_index_end = 2;
    model.populations.push_back(population);

    IrProgram empty_program;
    empty_program.component_type_name = "EmptyCell"; // every .tick stage left empty
    spikecorec::Vector<IrProgram> programs{empty_program};

    AssembledMasterKernelSource assembled = assemble_master_kernel_source(model, programs);
    ASSERT_EQ(assembled.population_gpu_sources.size(), 1u);
    EXPECT_TRUE(assembled.population_gpu_sources[0].functions.empty());

    EXPECT_NO_THROW({
        AssembledModel assembled_model(model, programs);
        (void)assembled_model;
    });
}

TEST(MasterKernel, collect_emit_port_names_returns_the_union_across_populations_in_first_seen_order) {
    ModelSpecification model;
    model.total_neuron_count = 2;
    model.type_library.push_back(build_lif_equivalent_type_entry("CellA", "a", 1.0f, 0.1f, 0.0f, 1.0f));
    model.type_library.push_back(build_lif_equivalent_type_entry("CellB", "b", 1.0f, 0.1f, 0.0f, 1.0f));

    PopulationEntry population_a;
    population_a.id = "A";
    population_a.type_library_index = 0;
    population_a.size = 1;
    population_a.neuron_index_begin = 0;
    population_a.neuron_index_end = 1;
    model.populations.push_back(population_a);

    PopulationEntry population_b;
    population_b.id = "B";
    population_b.type_library_index = 1;
    population_b.size = 1;
    population_b.neuron_index_begin = 1;
    population_b.neuron_index_end = 2;
    model.populations.push_back(population_b);

    IrProgram program_a = build_lif_equivalent_program("CellA", 0.1f, 0.0f, 1.0f); // emits "spike"
    IrProgram program_b = build_lif_equivalent_program("CellB", 0.1f, 0.0f, 1.0f);
    program_b.tick.emit = {IfInstruction{"spiked", {EmitInstruction{"burst"}}, {}, nullopt}};
    spikecorec::Vector<IrProgram> programs{program_a, program_b};

    spikecorec::Vector<String> ports = collect_emit_port_names(model, programs);
    ASSERT_EQ(ports.size(), 2u);
    EXPECT_EQ(ports[0], "spike");
    EXPECT_EQ(ports[1], "burst");
}

// ── acceptance criterion 2: reproduces the current hardcoded LIF as a special case ───────────────

TEST(MasterKernel, reproduces_hardcoded_lif_membrane_trajectory_and_spike_timing) {
    const f32 resting_mp = 0.0f;
    const f32 decay_rate = 0.1f;
    const f32 spike_threshold = 1.0f;
    const f32 dt = 1.0f;
    const f32 edge_weight = 0.6f;
    const s64 total_neuron_count = 3;
    const s64 tick_count = 6;

    // ── reference path: the existing hardcoded engine (src/metal/kernels.metal `step_no_active_
    // optimization`, dispatched via SpikeEngine::step_simulation with active-set disabled so every
    // neuron runs every tick -- the same "always dispatch the full range" scheme this ticket's own
    // AssembledModel uses, see master_kernel.h) ──
    vector<vector<s32>> adjacency = {{1}, {}, {}}; // neuron 0 -> neuron 1 only
    SpikeEngine engine(&adjacency, {total_neuron_count, 1}, /*rank=*/1, resting_mp, decay_rate,
                        /*learning_rate=*/0.0f, /*plasticity_enabled=*/false,
                        /*active_set_optimization_enabled=*/false);
    // spike_period=0 keeps the hardcoded engine's own "periodic forced reset" branch unreachable
    // after tick 0 (it only fires when tick - last_spiked == spike_period, and last_spiked is
    // always strictly in the past once a neuron has ever spiked, so that never recurs with
    // spike_period=0) while ALSO making its last_spiked update ("if tick - last_spiked >
    // spike_period") fire on every spiking tick -- matching this ticket's own propagate stage,
    // which writes last_spiked unconditionally on every `emit` per the locked IR's own EventOut
    // semantics (arch §3.2), not gated by the hardcoded engine's spike_period debounce quirk. A
    // large spike_period would leave last_spiked frozen at its very first update in the hardcoded
    // engine but keep updating every tick in this ticket's own propagate stage, a real but
    // orthogonal difference (see master_kernel.h) this test sidesteps by choosing the one
    // spike_period value where both conventions happen to coincide, rather than by weakening the
    // assertion.
    engine.spike_period = 0;
    engine.spike_threshold = spike_threshold;
    engine.weights.set_constant_weight(edge_weight);
    engine.use_constant_weight = true;
    engine.set_input_neurons({0}); // external stimulus targets neuron 0 only

    // ── assembled-model path: a hand-built GLIF1-equivalent ComponentType ──
    ModelSpecification model;
    model.total_neuron_count = (s32)total_neuron_count;
    model.type_library.push_back(
        build_lif_equivalent_type_entry("LifEquivalentCell", "lifEquivalentInstance", 1.0f, decay_rate, resting_mp,
                                         spike_threshold));

    PopulationEntry population;
    population.id = "Pop";
    population.type_library_index = 0;
    population.size = (s32)total_neuron_count;
    population.neuron_index_begin = 0;
    population.neuron_index_end = (s32)total_neuron_count;
    model.populations.push_back(population);

    spikecorec::Vector<IrProgram> programs = {
        build_lif_equivalent_program("LifEquivalentCell", decay_rate, resting_mp, spike_threshold)};

    ModelAllocation allocation = allocate_model(model, programs);
    std::fill(allocation.cell_state.get_contents(), allocation.cell_state.get_contents() + total_neuron_count,
              resting_mp);

    WeightMatrix my_weights(adjacency, /*rank=*/1);
    my_weights.set_constant_weight(edge_weight);

    AssembledModel assembled_model(model, programs);

    GpuPointer<f32> my_network_inputs = allocate<f32>((usize)total_neuron_count * sizeof(f32));
    memset(my_network_inputs.get_contents(), 0, (usize)total_neuron_count * sizeof(f32));
    GpuPointer<s64> my_last_spiked = allocate<s64>((usize)total_neuron_count * sizeof(s64));
    memset(my_last_spiked.get_contents(), 0, (usize)total_neuron_count * sizeof(s64));
    GpuPointer<s32> my_next_active_indices = allocate<s32>((usize)total_neuron_count * sizeof(s32));
    GpuPointer<s32> my_next_active_count = allocate<s32>(sizeof(s32));
    my_next_active_count.get_contents()[0] = 0;
    GpuPointer<s32> my_active_generation = allocate<s32>((usize)total_neuron_count * sizeof(s32));
    std::fill(my_active_generation.get_contents(), my_active_generation.get_contents() + total_neuron_count, -1);
    GpuPointer<bool> my_emit_spike = allocate<bool>((usize)total_neuron_count * sizeof(bool));
    memset(my_emit_spike.get_contents(), 0, (usize)total_neuron_count * sizeof(bool));

    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &my_weights;
    buffers.network_inputs = my_network_inputs.get_contents();
    buffers.last_spiked = my_last_spiked.get_contents();
    buffers.next_active_neuron_indices = my_next_active_indices.get_contents();
    buffers.next_active_neuron_count = my_next_active_count.get_contents();
    buffers.active_generation = my_active_generation.get_contents();
    buffers.emit_port_flags["spike"] = my_emit_spike.get_contents();

    const f32 external_pulse = 1.5f; // raises neuron 0 above threshold on injection tick's own integrate step

    // Tick 0 is deliberately a zero-input "priming" tick for both paths: the hardcoded engine's
    // own lazy-decay optimization special-cases time_since_last_update <= 0 (last_tick_updated
    // starts equal to tick 0 itself, so apply_decay is a no-op on the very first tick -- see
    // src/metal/kernels.metal's own `apply_decay`) -- a transient this ticket's own AssembledModel
    // path has no reason to reproduce (its GLIF-style integrate has no notion of "elapsed ticks",
    // it always applies exactly one dt). Injecting the external pulse one tick later (tick 1, once
    // last_tick_updated genuinely reflects "one tick elapsed" for both paths) sidesteps that
    // tick-0-only transient rather than asserting equivalence through it.
    for (s64 tick = 0; tick < tick_count; ++tick) {
        f32 this_tick_external_input = (tick == 1) ? external_pulse : 0.0f;

        // hardcoded engine: external stimulus goes straight into membrane_potentials (arch §0.2),
        // via the same gpu_add_network_input path step_simulation always uses.
        engine.step_simulation({this_tick_external_input}, tick);

        // assembled model: same external-stimulus convention, added directly into cell_state's
        // "v" slot for neuron 0 before this tick's integrate runs (mirrors gpu_add_network_input's
        // own "external ≠ network_inputs" semantics, arch §0.2).
        allocation.cell_state.get_contents()[0] += this_tick_external_input;
        assembled_model.step_tick(buffers, dt, tick, tick + 1);

        const f32 *engine_membrane_potentials = engine.membrane_potentials.get_contents();
        const f32 *my_v = allocation.cell_state.get_contents();
        for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
            EXPECT_NEAR(engine_membrane_potentials[neuron_index], my_v[neuron_index], 1e-4f)
                << "tick=" << tick << " neuron_index=" << neuron_index;
        }

        const s64 *engine_last_spiked = engine.last_spiked.get_contents();
        const s64 *my_last_spiked_contents = my_last_spiked.get_contents();
        for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
            EXPECT_EQ(engine_last_spiked[neuron_index], my_last_spiked_contents[neuron_index])
                << "tick=" << tick << " neuron_index=" << neuron_index;
        }

        const f32 *engine_network_inputs = engine.network_inputs.get_contents();
        const f32 *my_network_inputs_contents = my_network_inputs.get_contents();
        for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
            EXPECT_NEAR(engine_network_inputs[neuron_index], my_network_inputs_contents[neuron_index], 1e-4f)
                << "tick=" << tick << " neuron_index=" << neuron_index;
        }
    }

    // Sanity: the network actually exercised propagation/spiking within this test horizon (an
    // equivalence test over trajectories that never left resting state would be vacuous).
    bool any_spike_recorded = false;
    for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
        if (engine.last_spiked.get_contents()[neuron_index] != 0) any_spike_recorded = true;
    }
    EXPECT_TRUE(any_spike_recorded);
}

// ── regression: WeightMatrix::refit() must not desync Ck[DEFAULT_MATRIX_INDEX] from the live
// propagate kernel's all-ones assumption (ticket #103) ───────────────────────────────────────────
//
// The propagate kernel above (see its own header comment) reconstructs an edge's weight via a raw
// `dot(u_row, v_row)` with no `coefficient_vectors`/Ck parameter at all -- it hardcodes the
// DEFAULT_MATRIX_INDEX-is-all-ones invariant unconditionally. Before ticket #103's fix,
// WeightMatrix::refit() re-fit Ck[DEFAULT_MATRIX_INDEX] like any other registered matrix, so a
// refit() call could silently desync the host-side WeightMatrix::get() reconstruction (which reads
// whatever Ck[DEFAULT_MATRIX_INDEX] refit() left behind) from this live GPU kernel (which never
// reads it). This constructs a WeightMatrix with real accumulated Sk deltas (via
// accumulate_edge_delta()) on both DEFAULT_MATRIX_INDEX and a second registered matrix, calls
// refit(), then compares WeightMatrix::get()'s reconstruction against a REAL GPU dispatch of the
// propagate stage (via AssembledModel/compile_kernel, not a hand-simulated comparison) -- this
// exact scenario is what ticket #101 hit.

TEST(MasterKernel, refit_default_matrix_reconstruction_stays_synced_with_a_real_propagate_kernel_dispatch) {
    // A small, irregular network -- some nodes with multiple out-edges, one isolated sink -- so the
    // propagate dispatch below exercises more than a single trivial edge.
    vector<vector<s32>> network = {{1, 2}, {2}, {3}, {}};
    const s64 total_neuron_count = 4;
    const s64 rank = 4;

    WeightMatrix weight_matrix(network, rank, /*check_indexing=*/true, -1, /*weight_seed=*/4242);

    // Register a second matrix (ticket #52/D2) sharing the same U/V basis, with its own accumulated
    // Sk -- refit()'s full multi-matrix Ck/U/V sweep must run (not a single-matrix shortcut), and
    // this bug only reproduces when DEFAULT_MATRIX_INDEX itself has real Sk to fit against.
    s64 matrix_b = weight_matrix.add_coefficient_vector({1.0f, -0.5f, 2.0f, 0.25f});

    for (s64 source_node = 0; source_node < total_neuron_count; ++source_node) {
        for (s32 target_node : network[(usize)source_node]) {
            weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, (s32)source_node, target_node, 0.75f);
            weight_matrix.accumulate_edge_delta(matrix_b, (s32)source_node, target_node, -1.25f);
        }
    }

    weight_matrix.refit(/*sweep_count=*/4, /*ridge_regularization=*/1e-4f);

    // Host-side expected reconstruction for every real edge, post-refit -- WeightMatrix::get()
    // reconstructs via DEFAULT_MATRIX_INDEX's Ck (see reconstruct_entry).
    vector<pair<s32, s32>> real_edges;
    vector<f32> expected_edge_weights;
    for (s64 source_node = 0; source_node < total_neuron_count; ++source_node) {
        for (s32 target_node : network[(usize)source_node]) {
            real_edges.emplace_back((s32)source_node, target_node);
            expected_edge_weights.push_back(weight_matrix.get((s32)source_node, target_node));
        }
    }

    // ── a real GPU dispatch of the propagate stage against this SAME WeightMatrix ──
    //
    // A minimal cell type whose @detect is an unconditional literal comparison (0.0 > -1.0, always
    // true) -- every neuron spikes on the very first dispatch, so one step_tick call exercises the
    // propagate kernel's real dot(U,V) scatter for every real edge in the network at once.
    ModelSpecification model;
    model.total_neuron_count = (s32)total_neuron_count;
    TypeLibraryEntry type_entry;
    type_entry.component_type_name = "AlwaysSpikeCell";
    type_entry.bound_instance_id = "alwaysSpikeInstance";
    type_entry.category = TypeLibraryCategory::Cell;
    type_entry.state_variable_count = 1;
    model.type_library.push_back(type_entry);

    PopulationEntry population;
    population.id = "Pop";
    population.type_library_index = 0;
    population.size = (s32)total_neuron_count;
    population.neuron_index_begin = 0;
    population.neuron_index_end = (s32)total_neuron_count;
    model.populations.push_back(population);

    IrProgram program;
    program.component_type_name = "AlwaysSpikeCell";
    program.alloc = {StateDirective{"v", "f32", nullopt}};
    program.tick.detect = {BinaryInstruction{BinaryOpcode::Gt, "spiked", "0.0", "-1.0"}};
    program.tick.emit = {IfInstruction{"spiked", {EmitInstruction{"spike"}}, {}, nullopt}};
    spikecorec::Vector<IrProgram> programs = {program};

    ModelAllocation allocation = allocate_model(model, programs);
    AssembledModel assembled_model(model, programs);

    GpuPointer<f32> network_inputs = allocate<f32>((usize)total_neuron_count * sizeof(f32));
    memset(network_inputs.get_contents(), 0, (usize)total_neuron_count * sizeof(f32));
    GpuPointer<s64> last_spiked = allocate<s64>((usize)total_neuron_count * sizeof(s64));
    memset(last_spiked.get_contents(), 0, (usize)total_neuron_count * sizeof(s64));
    GpuPointer<s32> next_active_indices = allocate<s32>((usize)total_neuron_count * sizeof(s32));
    GpuPointer<s32> next_active_count = allocate<s32>(sizeof(s32));
    next_active_count.get_contents()[0] = 0;
    GpuPointer<s32> active_generation = allocate<s32>((usize)total_neuron_count * sizeof(s32));
    std::fill(active_generation.get_contents(), active_generation.get_contents() + total_neuron_count, -1);
    GpuPointer<bool> emit_spike = allocate<bool>((usize)total_neuron_count * sizeof(bool));
    memset(emit_spike.get_contents(), 0, (usize)total_neuron_count * sizeof(bool));

    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &weight_matrix;
    buffers.network_inputs = network_inputs.get_contents();
    buffers.last_spiked = last_spiked.get_contents();
    buffers.next_active_neuron_indices = next_active_indices.get_contents();
    buffers.next_active_neuron_count = next_active_count.get_contents();
    buffers.active_generation = active_generation.get_contents();
    buffers.emit_port_flags["spike"] = emit_spike.get_contents();

    assembled_model.step_tick(buffers, /*dt=*/1.0f, /*tick=*/0, /*next_tick=*/1);

    vector<f32> expected_network_inputs((usize)total_neuron_count, 0.0f);
    for (usize edge_index = 0; edge_index < real_edges.size(); ++edge_index) {
        expected_network_inputs[(usize)real_edges[edge_index].second] += expected_edge_weights[edge_index];
    }

    const f32 *gpu_network_inputs = network_inputs.get_contents();
    for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
        f32 expected = expected_network_inputs[(usize)neuron_index];
        f32 tolerance = std::fabs(expected) * 1e-3f;
        if (tolerance < 1e-3f) tolerance = 1e-3f;
        EXPECT_NEAR(expected, gpu_network_inputs[neuron_index], tolerance)
            << "neuron_index=" << neuron_index
            << " -- host WeightMatrix::get() reconstruction and a real GPU propagate kernel "
               "dispatch disagree on the default matrix's weight after refit() (ticket #103)";
    }

    // Sanity: every real edge actually contributed something nonzero -- otherwise this test would
    // vacuously pass even with the bug present.
    bool any_nonzero_input = false;
    for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
        if (gpu_network_inputs[neuron_index] != 0.0f) any_nonzero_input = true;
    }
    EXPECT_TRUE(any_nonzero_input);
}

// ── regression: next_active_neuron_count must be reset every tick, not accumulated ────────────────
//
// Isolates AssembledModel::step_tick's own active-set-enqueue bookkeeping (no reference SpikeEngine
// needed here) and asserts the counter's exact value every tick, rather than relying on the
// out-of-bounds write a missing reset causes: a regression back to "never reset" corrupts memory
// past next_active_neuron_indices's own [total_neuron_count] allocation, but Metal buffers are
// page-padded, so that corruption would NOT reliably crash or fail any assertion that doesn't
// directly inspect the counter -- this test inspects it directly, every tick.

TEST(MasterKernel, next_active_neuron_count_is_reset_every_tick_not_accumulated_across_ticks) {
    const f32 resting_mp = 0.0f;
    const f32 decay_rate = 0.1f;
    const f32 spike_threshold = 1.0f;
    const f32 dt = 1.0f;
    const s64 total_neuron_count = 3;

    // No edges at all (unlike the equivalence test above): this isolates the reset property from
    // the k^2-tree scatter/dedup logic entirely -- neuron 0 is the only neuron that ever receives
    // input (the external pulse below), so its own membrane trajectory is the only one that needs
    // hand-verifying, and every spike enqueues exactly itself (no downstream child).
    vector<vector<s32>> adjacency = {{}, {}, {}};

    ModelSpecification model;
    model.total_neuron_count = (s32)total_neuron_count;
    model.type_library.push_back(
        build_lif_equivalent_type_entry("LifEquivalentCell", "lifEquivalentInstance", 1.0f, decay_rate, resting_mp,
                                         spike_threshold));

    PopulationEntry population;
    population.id = "Pop";
    population.type_library_index = 0;
    population.size = (s32)total_neuron_count;
    population.neuron_index_begin = 0;
    population.neuron_index_end = (s32)total_neuron_count;
    model.populations.push_back(population);

    spikecorec::Vector<IrProgram> programs = {
        build_lif_equivalent_program("LifEquivalentCell", decay_rate, resting_mp, spike_threshold)};

    ModelAllocation allocation = allocate_model(model, programs);
    std::fill(allocation.cell_state.get_contents(), allocation.cell_state.get_contents() + total_neuron_count,
              resting_mp);

    WeightMatrix weights(adjacency, /*rank=*/1);

    AssembledModel assembled_model(model, programs);

    GpuPointer<f32> network_inputs = allocate<f32>((usize)total_neuron_count * sizeof(f32));
    memset(network_inputs.get_contents(), 0, (usize)total_neuron_count * sizeof(f32));
    GpuPointer<s64> last_spiked = allocate<s64>((usize)total_neuron_count * sizeof(s64));
    memset(last_spiked.get_contents(), 0, (usize)total_neuron_count * sizeof(s64));
    GpuPointer<s32> next_active_indices = allocate<s32>((usize)total_neuron_count * sizeof(s32));
    GpuPointer<s32> next_active_count = allocate<s32>(sizeof(s32));
    next_active_count.get_contents()[0] = 0;
    GpuPointer<s32> active_generation = allocate<s32>((usize)total_neuron_count * sizeof(s32));
    std::fill(active_generation.get_contents(), active_generation.get_contents() + total_neuron_count, -1);
    GpuPointer<bool> emit_spike = allocate<bool>((usize)total_neuron_count * sizeof(bool));
    memset(emit_spike.get_contents(), 0, (usize)total_neuron_count * sizeof(bool));

    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &weights;
    buffers.network_inputs = network_inputs.get_contents();
    buffers.last_spiked = last_spiked.get_contents();
    buffers.next_active_neuron_indices = next_active_indices.get_contents();
    buffers.next_active_neuron_count = next_active_count.get_contents();
    buffers.active_generation = active_generation.get_contents();
    buffers.emit_port_flags["spike"] = emit_spike.get_contents();

    const f32 external_pulse = 1.5f;

    // Neuron 0's own membrane trajectory (matching the equivalence test's own worked arithmetic
    // above): v = 1.35, 1.215, 1.0935 on ticks 1/2/3 (each > vth = 1.0, so it spikes and enqueues
    // exactly itself -- no children, since this fixture has no edges), then 0.98415, 0.885735 on
    // ticks 4/5 (sub-threshold, no spike, no enqueue).
    const vector<s32> expected_next_active_count_by_tick = {0, 1, 1, 1, 0, 0};

    for (s64 tick = 0; tick < (s64)expected_next_active_count_by_tick.size(); ++tick) {
        f32 this_tick_external_input = (tick == 1) ? external_pulse : 0.0f;
        allocation.cell_state.get_contents()[0] += this_tick_external_input;
        assembled_model.step_tick(buffers, dt, tick, tick + 1);

        EXPECT_EQ(*buffers.next_active_neuron_count, expected_next_active_count_by_tick[(usize)tick])
            << "tick=" << tick;
        // Never exceeds the buffer's own allocated element count regardless of how many neurons
        // spiked this tick -- the property a missing reset violates.
        ASSERT_LE(*buffers.next_active_neuron_count, (s32)total_neuron_count) << "tick=" << tick;
    }
}

// ── regression: a genuine constant weight of exactly 0 must not fall back to the U/V basis ────────
//
// `using_constant_weight` must be threaded through as its own explicit flag rather than overloaded
// onto `constant_weight == 0.0f`. Note that set_constant_weight(0.0f) alone would NOT discriminate
// between the two implementations here: it deliberately fills U/V so their own dot product already
// reconstructs the same constant_weight value (weight_matrix.cpp's own comment on
// set_constant_weight explains this is so get()/neighbor_weights() stay correct too) -- so a
// sentinel-based `constant_weight == 0.0f` check would coincidentally still compute 0 in that exact
// state. This test instead leaves U/V at their construction-time random values (a fixed weight_seed
// for reproducibility) and sets `constant_weight`/`using_constant_weight` directly (both are public
// fields, unlike the set_constant_weight() setter, which would re-sync U/V to match) -- so U.V for
// edge (0,1) is a real nonzero value while constant_weight is genuinely 0, and the propagated
// weight must come out exactly 0.0f (from constant_weight), not that nonzero U.V dot product.

TEST(MasterKernel, a_genuine_zero_constant_weight_propagates_exactly_zero_not_the_uv_basis) {
    const f32 resting_mp = 0.0f;
    const f32 decay_rate = 0.1f;
    const f32 spike_threshold = 1.0f;
    const f32 dt = 1.0f;
    const s64 total_neuron_count = 3;

    vector<vector<s32>> adjacency = {{1}, {}, {}}; // neuron 0 -> neuron 1 only

    ModelSpecification model;
    model.total_neuron_count = (s32)total_neuron_count;
    model.type_library.push_back(
        build_lif_equivalent_type_entry("LifEquivalentCell", "lifEquivalentInstance", 1.0f, decay_rate, resting_mp,
                                         spike_threshold));

    PopulationEntry population;
    population.id = "Pop";
    population.type_library_index = 0;
    population.size = (s32)total_neuron_count;
    population.neuron_index_begin = 0;
    population.neuron_index_end = (s32)total_neuron_count;
    model.populations.push_back(population);

    spikecorec::Vector<IrProgram> programs = {
        build_lif_equivalent_program("LifEquivalentCell", decay_rate, resting_mp, spike_threshold)};

    ModelAllocation allocation = allocate_model(model, programs);
    std::fill(allocation.cell_state.get_contents(), allocation.cell_state.get_contents() + total_neuron_count,
              resting_mp);

    // A fixed weight_seed leaves U/V at deterministic, real (non-zero) random-normal values (see
    // WeightMatrix::WeightMatrix) -- i.e. deliberately NOT calling set_constant_weight(), so U/V
    // for edge (0,1) do not happen to reconstruct 0.
    WeightMatrix weights(adjacency, /*rank=*/1, /*check_indexing=*/true, /*max_neighbor_count=*/-1,
                         /*weight_seed=*/42);
    ASSERT_NE(weights.get(0, 1), 0.0f) << "U/V for edge (0,1) coincidentally reconstruct 0 -- this "
                                           "test's own premise depends on a genuinely nonzero U.V "
                                           "dot product to disagree with constant_weight=0";

    // Both fields are public (unlike set_constant_weight(), which would also re-sync U/V to match
    // this value) -- so U/V stay at the real, nonzero values checked above while constant_weight is
    // genuinely 0.
    weights.constant_weight = 0.0f;
    weights.using_constant_weight = true;

    AssembledModel assembled_model(model, programs);

    GpuPointer<f32> network_inputs = allocate<f32>((usize)total_neuron_count * sizeof(f32));
    memset(network_inputs.get_contents(), 0, (usize)total_neuron_count * sizeof(f32));
    GpuPointer<s64> last_spiked = allocate<s64>((usize)total_neuron_count * sizeof(s64));
    memset(last_spiked.get_contents(), 0, (usize)total_neuron_count * sizeof(s64));
    GpuPointer<s32> next_active_indices = allocate<s32>((usize)total_neuron_count * sizeof(s32));
    GpuPointer<s32> next_active_count = allocate<s32>(sizeof(s32));
    next_active_count.get_contents()[0] = 0;
    GpuPointer<s32> active_generation = allocate<s32>((usize)total_neuron_count * sizeof(s32));
    std::fill(active_generation.get_contents(), active_generation.get_contents() + total_neuron_count, -1);
    GpuPointer<bool> emit_spike = allocate<bool>((usize)total_neuron_count * sizeof(bool));
    memset(emit_spike.get_contents(), 0, (usize)total_neuron_count * sizeof(bool));

    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &weights;
    buffers.network_inputs = network_inputs.get_contents();
    buffers.last_spiked = last_spiked.get_contents();
    buffers.next_active_neuron_indices = next_active_indices.get_contents();
    buffers.next_active_neuron_count = next_active_count.get_contents();
    buffers.active_generation = active_generation.get_contents();
    buffers.emit_port_flags["spike"] = emit_spike.get_contents();

    const f32 external_pulse = 1.5f; // raises neuron 0 above threshold on tick 1's own integrate step

    // Tick 0: zero-input priming tick, no spike. Tick 1: external pulse drives neuron 0 to v=1.35 >
    // vth=1.0, so it spikes and propagate scatters its (genuinely zero) weight into network_inputs[1].
    assembled_model.step_tick(buffers, dt, /*tick=*/0, /*next_tick=*/1);
    allocation.cell_state.get_contents()[0] += external_pulse;
    assembled_model.step_tick(buffers, dt, /*tick=*/1, /*next_tick=*/2);

    EXPECT_TRUE(last_spiked.get_contents()[0] == 1) << "neuron 0 did not spike on tick 1 as expected";
    EXPECT_EQ(network_inputs.get_contents()[1], 0.0f);
}

// ── ticket #62 [F1]: active-set x nonlinear-dynamics correctness rule ────────────────────────────
//
// AssembledModel::step_tick dispatches every population's kernel over its own FULL neuron range
// every tick regardless of closed_form_advanceable (see master_kernel.h/.cpp's own doc comments) --
// so "never fast-forwarded" is a property of ITS OWN per-tick dispatch, testable by comparing the
// assembled model's own trajectory against a plain, tick-by-tick C++ transcription of the exact
// same `.tick.integrate` instructions run once per tick with no batching. This is the master-kernel
// side of this ticket's acceptance criterion; the SEPARATE "a linear cell's skipped neuron still
// fast-forwards via closed-form apply_decay" side is the hardcoded engine's own, pre-existing,
// untouched mechanism -- verified in engine_tests.cpp's own
// `active_set_skip_then_revisit_matches_per_tick_decay_equivalence`, not here (this file has no
// notion of a hardcoded LIF's active-set skip at all -- see this ticket's own report for why).

TEST(MasterKernelActiveSetNonlinearRule, nonlinear_population_never_receives_a_closed_form_multi_tick_jump) {
    const f32 gL = 0.1f;
    const f32 EL = 0.0f;
    const f32 vth = 1000.0f; // never spikes -- isolates the pure integrate step from reset/regime noise
    const f32 dt = 1.0f;
    const s64 total_neuron_count = 2; // neuron 0 is this test's own subject; neuron 1 is an unused peer

    vector<vector<s32>> adjacency = {{}, {}}; // no edges at all -- neuron 0 never receives network_inputs

    ModelSpecification model;
    model.total_neuron_count = (s32)total_neuron_count;
    model.type_library.push_back(
        build_lif_equivalent_type_entry("NonlinearTestOnlyCell", "nonlinearTestOnlyInstance", 1.0f, gL, EL, vth));

    PopulationEntry population;
    population.id = "Pop";
    population.type_library_index = 0;
    population.size = (s32)total_neuron_count;
    population.neuron_index_begin = 0;
    population.neuron_index_end = (s32)total_neuron_count;
    model.populations.push_back(population);

    spikecorec::Vector<IrProgram> programs = {build_nonlinear_test_only_program("NonlinearTestOnlyCell", gL, EL, vth)};

    ModelAllocation allocation = allocate_model(model, programs);
    std::fill(allocation.cell_state.get_contents(), allocation.cell_state.get_contents() + total_neuron_count, EL);

    WeightMatrix weights(adjacency, /*rank=*/1);

    AssembledModel assembled_model(model, programs);

    GpuPointer<f32> network_inputs = allocate<f32>((usize)total_neuron_count * sizeof(f32));
    memset(network_inputs.get_contents(), 0, (usize)total_neuron_count * sizeof(f32));
    GpuPointer<s64> last_spiked = allocate<s64>((usize)total_neuron_count * sizeof(s64));
    memset(last_spiked.get_contents(), 0, (usize)total_neuron_count * sizeof(s64));
    GpuPointer<s32> next_active_indices = allocate<s32>((usize)total_neuron_count * sizeof(s32));
    GpuPointer<s32> next_active_count = allocate<s32>(sizeof(s32));
    next_active_count.get_contents()[0] = 0;
    GpuPointer<s32> active_generation = allocate<s32>((usize)total_neuron_count * sizeof(s32));
    std::fill(active_generation.get_contents(), active_generation.get_contents() + total_neuron_count, -1);
    GpuPointer<bool> emit_spike = allocate<bool>((usize)total_neuron_count * sizeof(bool));
    memset(emit_spike.get_contents(), 0, (usize)total_neuron_count * sizeof(bool));

    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &weights;
    buffers.network_inputs = network_inputs.get_contents();
    buffers.last_spiked = last_spiked.get_contents();
    buffers.next_active_neuron_indices = next_active_indices.get_contents();
    buffers.next_active_neuron_count = next_active_count.get_contents();
    buffers.active_generation = active_generation.get_contents();
    buffers.emit_port_flags["spike"] = emit_spike.get_contents();

    // A one-time pulse on tick 1 raises v off resting; every tick after that (tick 2 onward) neuron
    // 0 receives NEITHER external stimulus NOR any network_inputs (no edges exist at all) -- exactly
    // the "some neurons going quiet (inactive) for multiple ticks" scenario this ticket's acceptance
    // criterion describes. A closed-form multi-tick jump would show up as a DIVERGENCE from the
    // plain per-tick reference computed below the moment more than one quiet tick has elapsed.
    const f32 external_pulse = 0.3f;
    const s64 tick_count = 6;

    f32 reference_v = EL;
    for (s64 tick = 0; tick < tick_count; ++tick) {
        f32 this_tick_external_input = (tick == 1) ? external_pulse : 0.0f;

        allocation.cell_state.get_contents()[0] += this_tick_external_input;
        assembled_model.step_tick(buffers, dt, tick, tick + 1);

        // Plain, unbatched, exactly-one-dt-per-tick transcription of
        // build_nonlinear_test_only_program's own `@integrate` instructions -- network_inputs is
        // always 0.0f here (no edges ever scatter into it).
        reference_v += this_tick_external_input;
        f32 t0 = EL - reference_v;
        t0 = gL * t0;
        t0 = 0.0f + t0;
        f32 t1 = reference_v * reference_v;
        t0 = t0 - t1;
        t0 = t0 / 1.0f; // C
        t0 = t0 * dt;
        reference_v = reference_v + t0;

        EXPECT_NEAR(allocation.cell_state.get_contents()[0], reference_v, 1e-4f) << "tick=" << tick;
    }

    // Sanity: the reference trajectory actually moved (a comparison that never left EL would be
    // vacuous), and neuron 0 never spiked (vth is unreachable here, keeping this test purely about
    // @integrate).
    EXPECT_NE(reference_v, EL);
    EXPECT_EQ(last_spiked.get_contents()[0], 0);
}

// ── ticket #64 [F3]: spike-delay subsystem (delay ring) ──────────────────────────────────────────
//
// Exercises the ring-based deliver-drain/propagate path (AssembledModel constructed with
// enable_delay_ring=true, ModelRuntimeBuffers::delay_ring set) end to end through the real,
// compiled Metal kernels -- both this ticket's own acceptance criteria: (1) a delayed edge delivers
// exactly `delay` ticks later (checked directly against DelayRingAllocation::input_ring, the ring
// generalization of network_inputs the per-population `_tick` kernel actually reads -- IR spec
// §3.5), and (2) the active-set x delay interaction: the target neuron's own pending-active entry
// (DelayRingAllocation::pending_active_neuron_indices/count) appears at exactly the tick its
// delivery lands on, never before. compute_max_delay_ticks()/allocate_delay_ring()'s own host-side
// sizing/defaulting logic is unit-tested in isolation in delay_ring_tests.cpp; this file's own job
// is proving the whole thing wired together through a real AssembledModel::step_tick call.

namespace {

// True iff `neuron_index` appears among the first `count` entries of `indices_base` -- the
// compacted-list membership check every pending-active assertion below needs (order within a slot
// is not itself meaningful, only membership + count).
bool compacted_list_contains(const s32 *indices_base, s32 count, s32 neuron_index) {
    for (s32 position = 0; position < count; ++position) {
        if (indices_base[position] == neuron_index) return true;
    }
    return false;
}

} // namespace

// Genuinely compiles the two new ring kernels' exact MSL text via the real Metal compiler (mirrors
// this file's own `compiles_as_msl` helper / the flat drain+propagate kernels' own compile check in
// assembles_a_two_population_two_cell_type_model_and_compiles_every_kernel_as_msl above) -- on top
// of (not instead of) the real `newLibrary` compile AssembledModel's constructor performs for real
// further down in this section.
TEST(MasterKernelDelayRing, ring_kernel_sources_compile_as_msl) {
    GpuSource drain_ring_source = build_drain_ring_kernel_gpu_source();
    GpuSource propagate_ring_source = build_propagate_ring_kernel_gpu_source();

    EXPECT_NE(drain_ring_source.msl_source.find("kernel void spikecorec_master_drain_network_inputs_ring("),
              String::npos);
    EXPECT_NE(propagate_ring_source.msl_source.find("kernel void spikecorec_master_propagate_ring("), String::npos);

    EXPECT_TRUE(compiles_as_msl(drain_ring_source.msl_source, "drain_ring"));
    EXPECT_TRUE(compiles_as_msl(propagate_ring_source.msl_source, "propagate_ring"));

    // CUDA: structural check only -- no CUDA toolchain on this machine (documented constraint).
    EXPECT_NE(drain_ring_source.cuda_source.find("__global__ void spikecorec_master_drain_network_inputs_ring("),
              String::npos);
    EXPECT_NE(propagate_ring_source.cuda_source.find("__global__ void spikecorec_master_propagate_ring("),
              String::npos);
}

TEST(MasterKernelDelayRing, delivers_exactly_delay_ticks_later_and_wakes_the_active_set_at_the_right_tick) {
    const f32 resting_mp = 0.0f;
    const f32 decay_rate = 0.1f; // gL
    const f32 spike_threshold = 1.0f;
    const f32 dt = 1.0f; // also this test's own dt_seconds -- see allocate_delay_ring call below
    const f32 edge_weight = 0.6f;
    const s64 total_neuron_count = 2; // neuron 0 = source, neuron 1 = target
    const s64 delay_ticks = 5;        // "at least 5 ticks" per this ticket's own acceptance criterion
    const s64 tick_count = 10;

    vector<vector<s32>> adjacency = {{1}, {}}; // neuron 0 -> neuron 1 only

    ModelSpecification model;
    model.total_neuron_count = (s32)total_neuron_count;
    model.type_library.push_back(build_lif_equivalent_type_entry("LifEquivalentCell", "lifEquivalentInstance", 1.0f,
                                                                   decay_rate, resting_mp, spike_threshold));

    PopulationEntry population;
    population.id = "Pop";
    population.type_library_index = 0;
    population.size = (s32)total_neuron_count;
    population.neuron_index_begin = 0;
    population.neuron_index_end = (s32)total_neuron_count;
    model.populations.push_back(population);

    ProjectionEntry projection;
    projection.id = "Proj";
    projection.presynaptic_population_index = 0;
    projection.postsynaptic_population_index = 0;
    ConnectionEntry connection;
    connection.source_neuron_index = 0;
    connection.target_neuron_index = 1;
    connection.weight = (f64)edge_weight;
    connection.delay = (f64)delay_ticks * (f64)dt; // dt doubles as this test's own dt_seconds
    projection.connections.push_back(connection);
    model.projections.push_back(projection);

    spikecorec::Vector<IrProgram> programs = {
        build_lif_equivalent_program("LifEquivalentCell", decay_rate, resting_mp, spike_threshold)};

    ModelAllocation allocation = allocate_model(model, programs);
    std::fill(allocation.cell_state.get_contents(), allocation.cell_state.get_contents() + total_neuron_count,
              resting_mp);

    WeightMatrix weights(adjacency, /*rank=*/1);
    weights.set_constant_weight(edge_weight);
    weights.using_constant_weight = true;

    DelayRingAllocation ring = allocate_delay_ring(model, weights, /*dt_seconds=*/dt);
    ASSERT_EQ(ring.ring_slot_count, delay_ticks + 1); // arch §4.4: max_delay_ticks + 1

    AssembledModel assembled_model(model, programs, /*enable_delay_ring=*/true);

    GpuPointer<s64> last_spiked = allocate<s64>((usize)total_neuron_count * sizeof(s64));
    memset(last_spiked.get_contents(), 0, (usize)total_neuron_count * sizeof(s64));
    GpuPointer<bool> emit_spike = allocate<bool>((usize)total_neuron_count * sizeof(bool));
    memset(emit_spike.get_contents(), 0, (usize)total_neuron_count * sizeof(bool));

    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &weights;
    buffers.last_spiked = last_spiked.get_contents();
    buffers.emit_port_flags["spike"] = emit_spike.get_contents();
    buffers.delay_ring = &ring;

    // A single, deliberately isolated pulse on tick 1: v settles to 0.9*1.2=1.08 > vth=1.0 (one
    // spike, tick 1 only) then decays monotonically toward EL=0 forever after (0.972, 0.8748, ...),
    // never re-crossing vth -- worked out from this fixture's own `@integrate`
    // (dv = dt*(gL*(EL - v) + network_inputs)/C, C=1, applied to v AFTER the external pulse is added
    // directly into cell_state, matching every other test in this file's own external-stimulus
    // convention). This isolates the acceptance criterion to exactly ONE spike -> ONE delivery,
    // rather than the multi-spike burst a larger pulse (e.g. this file's own 1.5 constant) would
    // cause here (there is no @reset in this shared fixture -- see this file's own header comment).
    const f32 external_pulse = 1.2f;

    // Per-tick snapshots of exactly what the target neuron's own `_tick` kernel will read as
    // `network_inputs` THIS tick (ring.input_ring's slot for `tick`, read immediately before
    // step_tick(tick) so nothing has touched it since the last tick's own drain) -- and the matching
    // pending-active bookkeeping for the SAME slot.
    vector<f32> network_input_at_target_by_tick((usize)tick_count, 0.0f);
    vector<s32> pending_count_by_tick((usize)tick_count, 0);
    vector<bool> pending_contains_source_by_tick((usize)tick_count, false);
    vector<bool> pending_contains_target_by_tick((usize)tick_count, false);

    for (s64 tick = 0; tick < tick_count; ++tick) {
        s64 slot = tick % ring.ring_slot_count;
        network_input_at_target_by_tick[(usize)tick] =
            ring.input_ring.get_contents()[slot * total_neuron_count + 1];
        s32 count = ring.pending_active_neuron_count.get_contents()[slot];
        pending_count_by_tick[(usize)tick] = count;
        const s32 *indices_base = ring.pending_active_neuron_indices.get_contents() + slot * total_neuron_count;
        pending_contains_source_by_tick[(usize)tick] = compacted_list_contains(indices_base, count, 0);
        pending_contains_target_by_tick[(usize)tick] = compacted_list_contains(indices_base, count, 1);

        f32 this_tick_external_input = (tick == 1) ? external_pulse : 0.0f;
        allocation.cell_state.get_contents()[0] += this_tick_external_input;
        assembled_model.step_tick(buffers, dt, tick, tick + 1);
    }

    // ── acceptance criterion 1: exact-tick delivery, not before, not after ──────────────────────
    for (s64 tick = 0; tick < tick_count; ++tick) {
        f32 expected = (tick == 1 + delay_ticks) ? edge_weight : 0.0f;
        EXPECT_NEAR(network_input_at_target_by_tick[(usize)tick], expected, 1e-6f)
            << "tick=" << tick << " (spike at tick 1, delay=" << delay_ticks << " ticks)";
    }

    // ── acceptance criterion 2: the active-set x delay interaction -- the target neuron's own
    // pending-active entry appears at exactly the arrival tick (1 + delay_ticks), and the SOURCE
    // neuron's own unconditional one-tick self-reactivation (unaffected by any edge's delay, see
    // master_kernel.cpp's propagate-ring kernel) appears at exactly tick 1 + 1 = 2 -- neither
    // before nor after either tick, and no OTHER tick has any pending entry at all (no other spike
    // ever occurs in this fixture). ──
    for (s64 tick = 0; tick < tick_count; ++tick) {
        if (tick == 1 + delay_ticks) {
            EXPECT_EQ(pending_count_by_tick[(usize)tick], 1) << "tick=" << tick;
            EXPECT_TRUE(pending_contains_target_by_tick[(usize)tick]) << "tick=" << tick;
        } else if (tick == 2) {
            EXPECT_EQ(pending_count_by_tick[(usize)tick], 1) << "tick=" << tick;
            EXPECT_TRUE(pending_contains_source_by_tick[(usize)tick]) << "tick=" << tick;
        } else {
            EXPECT_EQ(pending_count_by_tick[(usize)tick], 0) << "tick=" << tick;
        }
    }

    // Sanity: the source neuron fired exactly once, at tick 1; the target never fired at all (its
    // own single 0.6 pulse at tick 6 settles to v=0.6 < vth=1.0 -- see this test's own header
    // comment's worked arithmetic).
    EXPECT_EQ(last_spiked.get_contents()[0], 1);
    EXPECT_EQ(last_spiked.get_contents()[1], 0);
}

// A model whose longest delay is bigger gets a correspondingly bigger ring -- exercised here through
// the SAME allocate_delay_ring() a real AssembledModel-driven simulation would call, not just
// delay_ring_tests.cpp's own narrower host-side unit tests, tying "sizing scales with the longest
// delay" directly to this ticket's own AssembledModel/ModelRuntimeBuffers integration.
TEST(MasterKernelDelayRing, ring_size_scales_with_the_longest_delay_actually_present_in_the_model) {
    const f32 dt_seconds = 1.0f;
    vector<vector<s32>> adjacency = {{1}, {}};

    auto build_model_with_delay = [&](f64 delay_seconds) {
        ModelSpecification model;
        model.total_neuron_count = 2;
        ProjectionEntry projection;
        ConnectionEntry connection;
        connection.source_neuron_index = 0;
        connection.target_neuron_index = 1;
        connection.weight = 1.0;
        connection.delay = delay_seconds;
        projection.connections.push_back(connection);
        model.projections.push_back(projection);
        return model;
    };

    ModelSpecification short_delay_model = build_model_with_delay(3.0);  // 3 ticks
    ModelSpecification long_delay_model = build_model_with_delay(20.0);  // 20 ticks

    WeightMatrix short_delay_weights(adjacency, /*rank=*/1);
    WeightMatrix long_delay_weights(adjacency, /*rank=*/1);

    DelayRingAllocation short_ring = allocate_delay_ring(short_delay_model, short_delay_weights, dt_seconds);
    DelayRingAllocation long_ring = allocate_delay_ring(long_delay_model, long_delay_weights, dt_seconds);

    EXPECT_EQ(short_ring.ring_slot_count, 4);  // 3 + 1
    EXPECT_EQ(long_ring.ring_slot_count, 21);  // 20 + 1
    EXPECT_GT(long_ring.ring_slot_count, short_ring.ring_slot_count);
}

// Ring mode's own zero/default-delay case (every connection floors to 1 tick, ring_slot_count == 2)
// must reproduce this file's own pre-#64 flat-network_inputs behavior byte for byte -- proving this
// ticket's own claim that it GENERALIZES the existing one-tick latency rather than replacing it.
// Same network/fixture as reproduces_hardcoded_lif_membrane_trajectory_and_spike_timing above,
// driven through the ring path instead of the flat path.
TEST(MasterKernelDelayRing, ring_mode_with_default_delay_reproduces_the_flat_one_tick_latency_behavior) {
    const f32 resting_mp = 0.0f;
    const f32 decay_rate = 0.1f;
    const f32 spike_threshold = 1.0f;
    const f32 dt = 1.0f;
    const f32 edge_weight = 0.6f;
    const s64 total_neuron_count = 3;
    const s64 tick_count = 6;

    // ── reference path: the existing hardcoded engine (identical to the non-ring equivalence test
    // above) ──
    vector<vector<s32>> adjacency = {{1}, {}, {}};
    SpikeEngine engine(&adjacency, {total_neuron_count, 1}, /*rank=*/1, resting_mp, decay_rate,
                        /*learning_rate=*/0.0f, /*plasticity_enabled=*/false,
                        /*active_set_optimization_enabled=*/false);
    engine.spike_period = 0;
    engine.spike_threshold = spike_threshold;
    engine.weights.set_constant_weight(edge_weight);
    engine.use_constant_weight = true;
    engine.set_input_neurons({0});

    // ── assembled-model path, ring mode, no delay attribute anywhere (every connection floors to
    // the implicit 1-tick latency, ring_slot_count = 1 + 1 = 2) ──
    ModelSpecification model;
    model.total_neuron_count = (s32)total_neuron_count;
    model.type_library.push_back(
        build_lif_equivalent_type_entry("LifEquivalentCell", "lifEquivalentInstance", 1.0f, decay_rate, resting_mp,
                                         spike_threshold));

    PopulationEntry population;
    population.id = "Pop";
    population.type_library_index = 0;
    population.size = (s32)total_neuron_count;
    population.neuron_index_begin = 0;
    population.neuron_index_end = (s32)total_neuron_count;
    model.populations.push_back(population);

    ProjectionEntry projection;
    ConnectionEntry connection;
    connection.source_neuron_index = 0;
    connection.target_neuron_index = 1;
    connection.weight = (f64)edge_weight;
    connection.delay = 0.0; // no delay attribute -> floors to 1 tick (compute_max_delay_ticks)
    projection.connections.push_back(connection);
    model.projections.push_back(projection);

    spikecorec::Vector<IrProgram> programs = {
        build_lif_equivalent_program("LifEquivalentCell", decay_rate, resting_mp, spike_threshold)};

    ModelAllocation allocation = allocate_model(model, programs);
    std::fill(allocation.cell_state.get_contents(), allocation.cell_state.get_contents() + total_neuron_count,
              resting_mp);

    WeightMatrix my_weights(adjacency, /*rank=*/1);
    my_weights.set_constant_weight(edge_weight);

    DelayRingAllocation ring = allocate_delay_ring(model, my_weights, /*dt_seconds=*/dt);
    ASSERT_EQ(ring.ring_slot_count, 2); // the pre-#64 flat scheme's own special case

    AssembledModel assembled_model(model, programs, /*enable_delay_ring=*/true);

    GpuPointer<s64> my_last_spiked = allocate<s64>((usize)total_neuron_count * sizeof(s64));
    memset(my_last_spiked.get_contents(), 0, (usize)total_neuron_count * sizeof(s64));
    GpuPointer<bool> my_emit_spike = allocate<bool>((usize)total_neuron_count * sizeof(bool));
    memset(my_emit_spike.get_contents(), 0, (usize)total_neuron_count * sizeof(bool));

    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &my_weights;
    buffers.last_spiked = my_last_spiked.get_contents();
    buffers.emit_port_flags["spike"] = my_emit_spike.get_contents();
    buffers.delay_ring = &ring;

    const f32 external_pulse = 1.5f;

    for (s64 tick = 0; tick < tick_count; ++tick) {
        f32 this_tick_external_input = (tick == 1) ? external_pulse : 0.0f;

        engine.step_simulation({this_tick_external_input}, tick);

        allocation.cell_state.get_contents()[0] += this_tick_external_input;
        assembled_model.step_tick(buffers, dt, tick, tick + 1);

        const f32 *engine_membrane_potentials = engine.membrane_potentials.get_contents();
        const f32 *my_v = allocation.cell_state.get_contents();
        for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
            EXPECT_NEAR(engine_membrane_potentials[neuron_index], my_v[neuron_index], 1e-4f)
                << "tick=" << tick << " neuron_index=" << neuron_index;
        }

        const s64 *engine_last_spiked = engine.last_spiked.get_contents();
        const s64 *my_last_spiked_contents = my_last_spiked.get_contents();
        for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
            EXPECT_EQ(engine_last_spiked[neuron_index], my_last_spiked_contents[neuron_index])
                << "tick=" << tick << " neuron_index=" << neuron_index;
        }

        // engine.network_inputs, checked immediately after this tick's own call, holds whatever THIS
        // tick's own spikes just scattered (to be consumed at tick+1) -- the ring's own equivalent
        // slot for that same "not yet consumed, due next tick" contribution is (tick+1) %
        // ring_slot_count, not tick % ring_slot_count (which is the slot THIS tick's own population
        // dispatch just consumed and drain just zeroed).
        const f32 *engine_network_inputs = engine.network_inputs.get_contents();
        s64 next_slot = (tick + 1) % ring.ring_slot_count;
        const f32 *my_ring_slot_contents = ring.input_ring.get_contents() + next_slot * total_neuron_count;
        for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
            EXPECT_NEAR(engine_network_inputs[neuron_index], my_ring_slot_contents[neuron_index], 1e-4f)
                << "tick=" << tick << " neuron_index=" << neuron_index;
        }
    }

    bool any_spike_recorded = false;
    for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
        if (engine.last_spiked.get_contents()[neuron_index] != 0) any_spike_recorded = true;
    }
    EXPECT_TRUE(any_spike_recorded);
}

// A caller must not mix delay-ring mode between construction and step_tick -- both mismatches throw
// rather than silently reading/writing whichever buffer happens to be set.
TEST(MasterKernelDelayRing, step_tick_throws_on_a_delay_ring_enablement_mismatch) {
    ModelSpecification model;
    model.total_neuron_count = 1;
    model.type_library.push_back(build_lif_equivalent_type_entry("Cell", "instance", 1.0f, 0.1f, 0.0f, 1.0f));

    PopulationEntry population;
    population.id = "Pop";
    population.type_library_index = 0;
    population.size = 1;
    population.neuron_index_begin = 0;
    population.neuron_index_end = 1;
    model.populations.push_back(population);

    spikecorec::Vector<IrProgram> programs = {build_lif_equivalent_program("Cell", 0.1f, 0.0f, 1.0f)};

    ModelAllocation allocation = allocate_model(model, programs);
    vector<vector<s32>> adjacency = {{}};
    WeightMatrix weights(adjacency, /*rank=*/1);

    GpuPointer<f32> network_inputs = allocate<f32>(sizeof(f32));
    memset(network_inputs.get_contents(), 0, sizeof(f32));
    GpuPointer<s64> last_spiked = allocate<s64>(sizeof(s64));
    memset(last_spiked.get_contents(), 0, sizeof(s64));
    GpuPointer<s32> next_active_indices = allocate<s32>(sizeof(s32));
    GpuPointer<s32> next_active_count = allocate<s32>(sizeof(s32));
    next_active_count.get_contents()[0] = 0;
    GpuPointer<s32> active_generation = allocate<s32>(sizeof(s32));
    active_generation.get_contents()[0] = -1;
    GpuPointer<bool> emit_spike = allocate<bool>(sizeof(bool));
    memset(emit_spike.get_contents(), 0, sizeof(bool));

    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &weights;
    buffers.network_inputs = network_inputs.get_contents();
    buffers.last_spiked = last_spiked.get_contents();
    buffers.next_active_neuron_indices = next_active_indices.get_contents();
    buffers.next_active_neuron_count = next_active_count.get_contents();
    buffers.active_generation = active_generation.get_contents();
    buffers.emit_port_flags["spike"] = emit_spike.get_contents();

    // Constructed WITHOUT enable_delay_ring, but buffers.delay_ring is left non-null below.
    AssembledModel assembled_model_flat(model, programs, /*enable_delay_ring=*/false);
    DelayRingAllocation ring = allocate_delay_ring(model, weights, /*dt_seconds=*/1.0f);
    buffers.delay_ring = &ring;
    EXPECT_THROW(assembled_model_flat.step_tick(buffers, 1.0f, 0, 1), std::runtime_error);

    // Constructed WITH enable_delay_ring, but buffers.delay_ring is null below.
    AssembledModel assembled_model_ring(model, programs, /*enable_delay_ring=*/true);
    buffers.delay_ring = nullptr;
    EXPECT_THROW(assembled_model_ring.step_tick(buffers, 1.0f, 0, 1), std::runtime_error);
}

// ── ticket #131: spike-scatter batch-construction subsystem -- real per-edge synapse dynamics ────
//
// The ticket's own acceptance criterion: a live AssembledModel network delivers a spike whose
// downstream effect is measurably derived from the target synapse's own ComponentType parameters --
// two otherwise-identical networks differing only in a synapse's `gbase` must produce measurably
// different postsynaptic trajectories. A hand-built IrProgram fixture (this file's own established
// pattern, see this file's header comment) standing in for a real, resolved conductance-based
// synapse (exactly synapse_lowering.cpp's own generated shape for something like expOneSynapse:
// `require v from postsynaptic`, one `peredge g`, an `onevent` bumping `g` by `gbase`, and
// `.tick.integrate`'s single top-level `forall neuron_in` decaying `g` and adding `g*(erev-v)` into
// `network_inputs`) -- proving the subsystem end-to-end without depending on the real NML/LEMS
// front-end this file's own sibling tests (synapse_lowering_tests.cpp) already cover in isolation.

namespace {

// Mirrors synapse_lowering.cpp's own generated shape for a conductance-based synapse (arch §4.3;
// ir_spec.md §4's expOne example) -- `g` bumped by `gbase` on delivery, decaying with time constant
// `tauDecay`, contributing `g*(erev-v)` into `network_inputs` every tick via `_integrate_edges`
// (lower_synapse_ir_program_to_gpu_source, ticket #131).
IrProgram build_conductance_synapse_program(const String &type_name, f32 gbase, f32 erev, f32 tau_decay) {
    IrProgram program;
    program.component_type_name = type_name;
    program.alloc = {
        RequireDirective{"v", "postsynaptic"},
        PeredgeDirective{"g"},
        ParamConstantDirective{"gbase", precise_float_literal(gbase)},
        ParamConstantDirective{"erev", precise_float_literal(erev)},
        ParamConstantDirective{"tauDecay", precise_float_literal(tau_decay)},
    };
    program.tick.deliver = {
        OnEventInstruction{"in", {AccumulateEdgeInstruction{"g", EdgeSetReference::CurrentEdge, "gbase"}}},
    };
    program.tick.integrate = {
        ForAllInstruction{
            EdgeSetReference::NeuronIn,
            {
                LoadEdgeInstruction{"edge_g_old", "g", EdgeSetReference::CurrentEdge},
                BinaryInstruction{BinaryOpcode::ExpDecay, "edge_g", "edge_g_old", "tauDecay"},
                BinaryInstruction{BinaryOpcode::Sub, "edge_g_delta", "edge_g", "edge_g_old"},
                AccumulateEdgeInstruction{"g", EdgeSetReference::CurrentEdge, "edge_g_delta"},
                BinaryInstruction{BinaryOpcode::Sub, "i", "erev", "v"},
                BinaryInstruction{BinaryOpcode::Mul, "i", "edge_g", "i"},
                BinaryInstruction{BinaryOpcode::Add, "network_inputs", "network_inputs", "i"},
            },
        },
    };
    return program;
}

TypeLibraryEntry build_conductance_synapse_type_entry(const String &type_name, const String &bound_instance_id,
                                                       f32 gbase, f32 erev, f32 tau_decay) {
    TypeLibraryEntry entry;
    entry.component_type_name = type_name;
    entry.bound_instance_id = bound_instance_id;
    entry.category = TypeLibraryCategory::Synapse;
    entry.is_conductance_based = true;
    entry.state_variable_count = 1;
    entry.baked_constants = {{"gbase", (f64)gbase}, {"erev", (f64)erev}, {"tauDecay", (f64)tau_decay}};
    return entry;
}

// One presynaptic cell ("Pop"'s neuron 0) driving one postsynaptic cell (neuron 1) through one
// conductance-based synapse edge, for `tick_count` ticks. Neuron 0 is driven over threshold by a
// single large external pulse on tick 1 (mirroring this file's own `reproduces_hardcoded_lif_
// membrane_trajectory_and_spike_timing` "priming tick" convention); everything else (cell
// parameters, adjacency, dt, the pulse) is IDENTICAL across calls -- only `gbase` varies. Returns
// neuron 1's own membrane-potential trajectory, one sample per tick.
spikecorec::Vector<f32> run_two_neuron_conductance_synapse_network_and_capture_postsynaptic_trajectory(f32 gbase) {
    const f32 resting_mp = 0.0f;
    const f32 decay_rate = 0.1f;
    const f32 spike_threshold = 1.0f;
    const f32 erev = 2.0f;
    const f32 tau_decay = 2.0f;
    const f32 dt = 1.0f;
    const s64 tick_count = 20;

    ModelSpecification model;
    model.total_neuron_count = 2;
    model.type_library.push_back(
        build_lif_equivalent_type_entry("PostCell", "postInstance", 1.0f, decay_rate, resting_mp, spike_threshold));
    model.type_library.push_back(
        build_conductance_synapse_type_entry("CondSynapse", "synInstance", gbase, erev, tau_decay));

    PopulationEntry population;
    population.id = "Pop";
    population.type_library_index = 0;
    population.size = 2;
    population.neuron_index_begin = 0;
    population.neuron_index_end = 2;
    model.populations.push_back(population);

    ProjectionEntry projection;
    projection.id = "Proj";
    projection.presynaptic_population_index = 0;
    projection.postsynaptic_population_index = 0;
    projection.synapse_type_library_index = 1;
    projection.connections.push_back(ConnectionEntry{/*source=*/0, /*target=*/1, /*weight=*/1.0, /*delay=*/0.0});
    model.projections.push_back(projection);

    spikecorec::Vector<IrProgram> programs = {
        build_lif_equivalent_program("PostCell", decay_rate, resting_mp, spike_threshold),
        build_conductance_synapse_program("CondSynapse", gbase, erev, tau_decay),
    };

    ModelAllocation allocation = allocate_model(model, programs);
    std::fill(allocation.cell_state.get_contents(), allocation.cell_state.get_contents() + 2, resting_mp);

    vector<vector<s32>> adjacency = {{1}, {}};
    WeightMatrix weights(adjacency, /*rank=*/1);

    AssembledModel assembled_model(model, programs);

    GpuPointer<f32> network_inputs = allocate<f32>(2 * sizeof(f32));
    memset(network_inputs.get_contents(), 0, 2 * sizeof(f32));
    GpuPointer<s64> last_spiked = allocate<s64>(2 * sizeof(s64));
    memset(last_spiked.get_contents(), 0, 2 * sizeof(s64));
    GpuPointer<s32> next_active_indices = allocate<s32>(2 * sizeof(s32));
    GpuPointer<s32> next_active_count = allocate<s32>(sizeof(s32));
    next_active_count.get_contents()[0] = 0;
    GpuPointer<s32> active_generation = allocate<s32>(2 * sizeof(s32));
    std::fill(active_generation.get_contents(), active_generation.get_contents() + 2, -1);
    GpuPointer<bool> emit_spike = allocate<bool>(2 * sizeof(bool));
    memset(emit_spike.get_contents(), 0, 2 * sizeof(bool));

    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &weights;
    buffers.network_inputs = network_inputs.get_contents();
    buffers.last_spiked = last_spiked.get_contents();
    buffers.next_active_neuron_indices = next_active_indices.get_contents();
    buffers.next_active_neuron_count = next_active_count.get_contents();
    buffers.active_generation = active_generation.get_contents();
    buffers.emit_port_flags["spike"] = emit_spike.get_contents();

    spikecorec::Vector<f32> postsynaptic_trajectory;
    postsynaptic_trajectory.reserve((usize)tick_count);
    for (s64 tick = 0; tick < tick_count; ++tick) {
        if (tick == 1) allocation.cell_state.get_contents()[0] += 5.0f; // drives neuron 0 over threshold
        assembled_model.step_tick(buffers, dt, tick, tick + 1);
        postsynaptic_trajectory.push_back(allocation.cell_state.get_contents()[1]);
    }

    deallocate(std::move(network_inputs));
    deallocate(std::move(last_spiked));
    deallocate(std::move(next_active_indices));
    deallocate(std::move(next_active_count));
    deallocate(std::move(active_generation));
    deallocate(std::move(emit_spike));

    return postsynaptic_trajectory;
}

} // namespace

TEST(MasterKernelSynapseDispatch,
     two_networks_differing_only_in_synapse_gbase_produce_measurably_different_postsynaptic_trajectories) {
    spikecorec::Vector<f32> weak_synapse_trajectory =
        run_two_neuron_conductance_synapse_network_and_capture_postsynaptic_trajectory(/*gbase=*/0.5f);
    spikecorec::Vector<f32> strong_synapse_trajectory =
        run_two_neuron_conductance_synapse_network_and_capture_postsynaptic_trajectory(/*gbase=*/2.0f);
    ASSERT_EQ(weak_synapse_trajectory.size(), strong_synapse_trajectory.size());

    // The postsynaptic neuron must have actually moved away from rest under the strong-gbase
    // network -- proving the synapse's own current genuinely reached it (not merely "technically
    // different by float noise").
    bool postsynaptic_neuron_moved_under_strong_gbase = false;
    for (f32 voltage : strong_synapse_trajectory) {
        if (std::fabs(voltage - 0.0f) > 1e-3f) {
            postsynaptic_neuron_moved_under_strong_gbase = true;
            break;
        }
    }
    EXPECT_TRUE(postsynaptic_neuron_moved_under_strong_gbase)
        << "postsynaptic neuron never moved away from rest -- the synapse's own current never reached it";

    // The two otherwise-identical networks (same cells, same adjacency, same dt, same external
    // pulse) must diverge measurably once gbase differs.
    bool trajectories_differ_measurably = false;
    for (usize tick = 0; tick < weak_synapse_trajectory.size(); ++tick) {
        if (std::fabs(weak_synapse_trajectory[tick] - strong_synapse_trajectory[tick]) > 1e-4f) {
            trajectories_differ_measurably = true;
            break;
        }
    }
    EXPECT_TRUE(trajectories_differ_measurably)
        << "postsynaptic trajectory is identical regardless of gbase -- not actually derived from the "
           "synapse's own parameters";
}
