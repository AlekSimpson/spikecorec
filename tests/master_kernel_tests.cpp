#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/engine.h"
#include "spikecorec/core/weight_matrix.h"
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
