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
#include "spikecorec/nml/delay_ring.h"
#include "spikecorec/nml/master_kernel.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── Master-kernel assembly + compile + cache + dispatch tests (ticket #6 [C3]) ───────────────────
//
// What this file demonstrates, matching the ticket's own acceptance criteria: a resolved model
// (here, hand-built ModelSpecification/IrProgram fixtures -- the SAME established pattern
// gpu_source_tests.cpp/ir_tests.cpp/allocator_tests.cpp's own `throws_on_non_f32_state_dtype`
// fixture already use, rather than going through the full NML/LEMS parse+resolve pipeline) compiles
// to one runnable master kernel on BOTH backends: every generated MSL fixture is ACTUALLY compiled
// via `xcrun -sdk macosx metal -c` (mirrors gpu_source_tests.cpp's own `compiles_as_msl` helper);
// the CUDA text is checked structurally only (no CUDA toolchain on this machine -- the ticket's own
// documented constraint).
//
// This file's own AssembledModel-stateful behavioral-equivalence tests (the assembled master kernel
// reproducing the current hardcoded LIF cell as a special case, real per-edge synapse dispatch, real
// STDP + delay ring, the active-set x nonlinear-dynamics rule, ...) have been ported to SpikeEngine
// as part of folding nml::AssembledModel into SpikeEngine -- see
// tests/spike_engine_nml_construction_tests.cpp's own
// SpikeEngineNmlPropagateDispatch/SpikeEngineNmlActiveSetNonlinearRule/SpikeEngineNmlSynapseDispatch/
// SpikeEngineNmlPlasticity/SpikeEngineNmlDelayRing suites. `nml::AssembledModel` itself has since been
// deleted entirely (master_kernel.h/.cpp). What remains here are the STATELESS free functions
// (compile_kernel_or_throw_with_source/assemble_master_kernel_source/collect_emit_port_
// names/build_drain_ring_kernel_gpu_source/build_propagate_ring_kernel_gpu_source/allocate_delay_
// ring) that are staying in master_kernel.h/.cpp/delay_ring.h/.cpp untouched.

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

    // compile + cache: constructing a SpikeEngine actually calls compile_kernel for every
    // kernel (Metal newLibrary, on this build) without throwing -- the "compile once" half of the
    // ticket, exercised for real, not just assembled as text.
    EXPECT_NO_THROW({ SpikeEngine engine(model, programs); });
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

    EXPECT_NO_THROW({ SpikeEngine engine(model, programs); });
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

// ── ticket #64 [F3]: spike-delay subsystem (delay ring) ──────────────────────────────────────────
//
// The end-to-end AssembledModel-driven ring-delivery/active-set integration test this suite used to
// hold here (delivers_exactly_delay_ticks_later_and_wakes_the_active_set_at_the_right_tick) has been
// ported to SpikeEngineNmlDelayRing.ring_shaped_path_delivers_a_spike_exactly_delay_ticks_ticks_later
// (tests/spike_engine_nml_construction_tests.cpp) as part of folding nml::AssembledModel into
// SpikeEngine (AssembledModel itself has since been deleted entirely -- master_kernel.h/.cpp) -- the
// two tests were already equivalent in intent/assertions. What remains here is genuinely free-function
// coverage (build_drain_ring_kernel_gpu_source/build_propagate_ring_kernel_gpu_source compile as real
// MSL) that has no dependency on AssembledModel at all, so it stays.

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

// A model whose longest delay is bigger gets a correspondingly bigger ring -- exercised here through
// the SAME allocate_delay_ring() a real AssembledModel-driven simulation would call, tying "sizing
// scales with the longest delay" directly to this ticket's own AssembledModel/ModelRuntimeBuffers
// integration. Pure free-function coverage (allocate_delay_ring/WeightMatrix) with no dependency on
// AssembledModel itself -- stays here (tests/delay_ring_tests.cpp's own narrower host-side unit test
// of the same free functions has been retired; see SpikeEngineNmlDelayRing's own per-edge_non_
// uniform_delays_... test, tests/spike_engine_nml_construction_tests.cpp, for the SpikeEngine-side
// integration equivalent of "ring sizes to the longest delay actually present").
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

// ring_mode_with_default_delay_reproduces_the_flat_one_tick_latency_behavior used to live here
// (AssembledModel's own ring-mode-with-a-trivial-delay-projection vs the legacy SpikeEngine
// comparison) -- jointly covered now by SpikeEngineNmlDelayRing.ring_slot_count_one_no_configured_
// delay_matches_the_flat_one_tick_latency (a real projection with delay=0.0, hand-derived expected
// trajectory pattern) and SpikeEngineNmlPropagateDispatch.reproduces_hardcoded_lif_membrane_
// trajectory_and_spike_timing (the legacy-SpikeEngine-vs-NML-mode-SpikeEngine side-by-side
// comparison, no projections at all) -- both in tests/spike_engine_nml_construction_tests.cpp --
// since SpikeEngine's own always-ring-shaped design collapses the old "flat AssembledModel" vs
// "ring-mode AssembledModel with a trivial delay" distinction into one unified mechanism.

// step_tick_throws_on_a_delay_ring_enablement_mismatch used to live here: AssembledModel required a
// caller not to mix delay-ring mode between construction and step_tick (both mismatches threw,
// rather than silently reading/writing whichever buffer happened to be set). This concept does NOT
// apply to SpikeEngine's own design (confirmed by reading engine.h/engine.cpp): there is no separate
// enable flag or two divergent code paths -- SpikeEngine's step_tick always operates on its own
// internally-owned, always ring-shaped (ring_slot_count >= 1) buffers, with no external "buffers"
// parameter to mismatch against a construction-time flag at all. A genuine simplification, not a
// gap -- so, unlike this file's other retired AssembledModel-stateful tests, there is no SpikeEngine
// analogue to port this one to; the behavior it guarded against is categorically impossible in
// SpikeEngine's design, not an uncovered case.

// ── ticket #131: spike-scatter batch-construction subsystem -- real per-edge synapse dynamics ────
//
// MasterKernelSynapseDispatch.two_networks_differing_only_in_synapse_gbase_produce_measurably_
// different_postsynaptic_trajectories used to live here -- already covered verbatim by
// SpikeEngineNmlSynapseDispatch.real_per_edge_synapse_dispatch_measurably_perturbs_postsynaptic_
// membrane_potential (tests/spike_engine_nml_construction_tests.cpp, built independently of this
// task against the SAME conductance-synapse fixture/weak-vs-strong-gbase technique) -- confirmed
// genuinely redundant, retired rather than re-ported.
