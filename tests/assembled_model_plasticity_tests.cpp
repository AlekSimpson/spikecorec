#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/nml/delay_ring.h"
#include "spikecorec/nml/master_kernel.h"
#include "spikecorec/nml/plasticity_wiring.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── Real STDP support on AssembledModel (ticket #132) ────────────────────────────────────────────
//
// See master_kernel.h's own "ticket #132" doc comment for the full design this exercises: STDP
// ported onto AssembledModel by reusing WeightMatrix::update() (mathematically identical, for a
// single iteration, to the legacy engine's own step_apply_hebbian_update -- see plasticity_wiring.h)
// as a stage-7 host loop run right after the fixed propagate stage, in BOTH the flat and ring-mode
// branches of step_tick. This file's own main test is this ticket's acceptance criterion: a single
// AssembledModel-based tick loop combining real GLIF cell dynamics, a non-trivial per-edge delay
// (the delay ring, ticket #64), and active STDP, showing the weight move in the expected (depression)
// direction over the run. The remaining tests cover the smaller API surface this ticket adds
// (enable/disable semantics, the apply_stdp_wiring(ModelSpecification&, AssembledModel&) overload,
// and the explicit incompatibility guard against ticket #131's real per-edge synapse dispatch).
//
// ── What this file deliberately does NOT cover (ticket #129 [T8] stays open) ──
// The 2-neuron GLIF1/LIF-equivalent fixture below proves the INTEGRATION MECHANISM exists (real
// cell dynamics + a non-trivial delay + active STDP, together, in one AssembledModel tick loop) --
// #132's own acceptance criterion. It is NOT ticket #129's own network-scale GLIF3/GLIF5 + STDP test
// (ticket #100's own ~300-500 neuron / 1000-2000 tick anchor, specifically to exercise the
// after-spike-current/adaptation state a 2-neuron GLIF1 fixture has none of), and this ticket adds
// no `examples/glif_stdp_plasticity_example.cpp` either. See master_kernel.h's own "Relationship to
// ticket #129" doc comment for the explicit re-scope: #132 unblocks #129 (a real NML/GLIF network
// can now run with active STDP at all, via the API this file exercises) but does not satisfy it --
// #129 remains open for its own two original deliverables.

namespace {

String precise_float_literal(f32 value) {
    ostringstream stream;
    stream << std::setprecision(9) << value;
    return stream.str();
}

// A minimal, self-contained GLIF1-shaped cell program -- the same fixture shape
// master_kernel_tests.cpp's own build_lif_equivalent_program uses (kept as its own copy here,
// matching this codebase's own established per-file fixture convention): `dv/dt = (gL*(EL-v) +
// network_inputs) / C` (forward Euler, one dt per tick), `spike` on `v > vth`, no `@reset`.
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

// A minimal, self-contained conductance synapse program -- mirrors master_kernel_tests.cpp's own
// build_conductance_synapse_program (kept as its own copy here), used only to build a model with
// real per-edge synapse dispatch active (ticket #131), so this file's own incompatibility-guard test
// can exercise a REAL AssembledModel/ProjectionEntry combination rather than an empty stand-in.
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

// A single-population, no-projections model just big enough to construct a valid AssembledModel --
// used by the tests below that only exercise the plasticity API surface itself, not a real network.
ModelSpecification build_minimal_one_neuron_model(spikecorec::Vector<IrProgram> &out_programs) {
    ModelSpecification model;
    model.total_neuron_count = 1;
    model.type_library.push_back(build_lif_equivalent_type_entry("Cell", "cellInstance", 1.0f, 0.1f, 0.0f, 1.0f));

    PopulationEntry population;
    population.id = "Pop";
    population.type_library_index = 0;
    population.size = 1;
    population.neuron_index_begin = 0;
    population.neuron_index_end = 1;
    model.populations.push_back(population);

    out_programs = {build_lif_equivalent_program("Cell", 0.1f, 0.0f, 1.0f)};
    return model;
}

} // namespace

// ── enable/disable API semantics (mirrors SpikeEngine's own SC-11 behavior) ──────────────────────

TEST(AssembledModelPlasticity, plasticity_enabled_reflects_enable_and_disable_calls) {
    spikecorec::Vector<IrProgram> programs;
    ModelSpecification model = build_minimal_one_neuron_model(programs);
    AssembledModel assembled_model(model, programs);

    EXPECT_FALSE(assembled_model.plasticity_enabled());

    assembled_model.enable_plasticity(0.05f);
    EXPECT_TRUE(assembled_model.plasticity_enabled());

    assembled_model.disable_plasticity();
    EXPECT_FALSE(assembled_model.plasticity_enabled());

    // both a second disable and a redundant enable-while-enabled must not throw (no-op, matching
    // SpikeEngine::enable_plasticity/disable_plasticity's own established semantics)
    EXPECT_NO_THROW(assembled_model.disable_plasticity());
    EXPECT_NO_THROW(assembled_model.enable_plasticity(0.05f));
    EXPECT_NO_THROW(assembled_model.enable_plasticity(0.9f));
    EXPECT_TRUE(assembled_model.plasticity_enabled());
}

// ── coordinating with ticket #131's per-edge synapse dispatch ────────────────────────────────────
//
// enable_plasticity must refuse to run alongside real per-edge synapse dispatch (a second,
// uncoordinated writer of the shared U/V basis a peredge synapse's own Ck also reconstructs from --
// see master_kernel.h's own "ticket #132" doc comment) rather than silently letting STDP corrupt it.
TEST(AssembledModelPlasticity, enable_plasticity_throws_when_real_per_edge_synapse_dispatch_is_active) {
    const f32 resting_mp = 0.0f;
    const f32 decay_rate = 0.1f;
    const f32 spike_threshold = 1.0f;
    const f32 gbase = 1.0f, erev = 2.0f, tau_decay = 2.0f;

    ModelSpecification model;
    model.total_neuron_count = 2;
    model.type_library.push_back(
        build_lif_equivalent_type_entry("PostCell", "postInstance", 1.0f, decay_rate, resting_mp, spike_threshold));
    model.type_library.push_back(build_conductance_synapse_type_entry("CondSynapse", "synInstance", gbase, erev, tau_decay));

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

    AssembledModel assembled_model(model, programs); // enable_delay_ring=false (default): projections_
                                                      // is a real, non-empty copy of model.projections

    EXPECT_THROW(assembled_model.enable_plasticity(0.02f), std::runtime_error);
    EXPECT_FALSE(assembled_model.plasticity_enabled());
}

// The SAME projections, but with enable_delay_ring=true: ticket #131's own established precedent is
// that ring mode forces real per-edge synapse dispatch off entirely regardless of model.projections
// (see master_kernel.h) -- so enable_plasticity must NOT throw here, since there is no second writer
// of the shared U/V basis to coordinate with in this mode.
TEST(AssembledModelPlasticity, enable_plasticity_does_not_throw_under_delay_ring_mode_even_with_projections_present) {
    const f32 resting_mp = 0.0f;
    const f32 decay_rate = 0.1f;
    const f32 spike_threshold = 1.0f;
    const f32 gbase = 1.0f, erev = 2.0f, tau_decay = 2.0f;

    ModelSpecification model;
    model.total_neuron_count = 2;
    model.type_library.push_back(
        build_lif_equivalent_type_entry("PostCell", "postInstance", 1.0f, decay_rate, resting_mp, spike_threshold));
    model.type_library.push_back(build_conductance_synapse_type_entry("CondSynapse", "synInstance", gbase, erev, tau_decay));

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

    AssembledModel assembled_model(model, programs, /*enable_delay_ring=*/true);

    EXPECT_NO_THROW(assembled_model.enable_plasticity(0.02f));
    EXPECT_TRUE(assembled_model.plasticity_enabled());
}

// ── apply_stdp_wiring(ModelSpecification&, AssembledModel&) ──────────────────────────────────────

TEST(AssembledModelPlasticity, apply_stdp_wiring_enables_plasticity_from_a_real_type_library_scan) {
    spikecorec::Vector<IrProgram> programs;
    ModelSpecification model = build_minimal_one_neuron_model(programs);

    // AssembledModel's own constructor requires type_library_ir_programs.size() ==
    // model.type_library.size(), so the assembled model is built BEFORE the extra Synapse-category
    // entry below is appended -- apply_stdp_wiring itself only reads model.type_library at CALL
    // time, so mutating model afterward is exactly what a real caller (scan the resolved model,
    // separately, after the AssembledModel it drives is already built) would do too.
    AssembledModel assembled_model(model, programs);
    ASSERT_FALSE(assembled_model.plasticity_enabled());

    TypeLibraryEntry stdp_synapse_entry;
    stdp_synapse_entry.component_type_name = "TestStdpSynapse";
    stdp_synapse_entry.bound_instance_id = "synInstance";
    stdp_synapse_entry.category = TypeLibraryCategory::Synapse;
    stdp_synapse_entry.baked_constants = {
        {"tauPlus", 0.02}, {"tauMinus", 0.02}, {"aPlus", 0.01}, {"aMinus", 0.03}};
    // Scanned by apply_stdp_wiring (which only reads model.type_library) -- deliberately never
    // referenced by any projection/population, matching this test's own narrow scope.
    model.type_library.push_back(stdp_synapse_entry);

    apply_stdp_wiring(model, assembled_model);

    EXPECT_TRUE(assembled_model.plasticity_enabled());
}

TEST(AssembledModelPlasticity, apply_stdp_wiring_disables_plasticity_when_no_stdp_synapse_present) {
    spikecorec::Vector<IrProgram> programs;
    ModelSpecification model = build_minimal_one_neuron_model(programs);

    AssembledModel assembled_model(model, programs);
    assembled_model.enable_plasticity(0.1f); // simulate an instance constructed with plasticity already live
    ASSERT_TRUE(assembled_model.plasticity_enabled());

    apply_stdp_wiring(model, assembled_model);

    EXPECT_FALSE(assembled_model.plasticity_enabled());
}

// ── ticket #132's own acceptance criterion: real GLIF cell dynamics + a non-trivial per-edge delay
// (the delay ring) + active STDP, all under one AssembledModel-based tick loop ────────────────────
//
// Two neurons, one connection (0 -> 1), each driven to fire exactly once by its own isolated,
// controlled external pulse (the SAME "no @reset" single-spike fixture convention
// master_kernel_tests.cpp's own delay-ring tests use): neuron 1 (postsynaptic) fires first, at
// tick 2; neuron 0 (presynaptic) fires second, at tick 3 -- i.e. postsynaptic-before-presynaptic
// timing, the only case the real Hebbian-update call site ever nudges the weight for (see
// plasticity_wiring.h's own doc comment). The connection's own delay (5 ticks) is deliberately
// decoupled from the STDP-nudged weight: `weights.set_constant_weight`/`using_constant_weight=true`
// makes the DELAY RING's own delivered magnitude a known, fixed, sub-threshold constant (so it can
// never cause a spurious extra spike and desync the fixture's own hand-worked timing), while
// `weights.get(0, 1)` -- which always reads the real U/V reconstruction regardless of
// using_constant_weight (weight_matrix.cpp's own get()) -- is what this test actually asserts STDP
// moved, exactly the same real rank-1 nudge the delay-ring path leaves untouched.
TEST(AssembledModelPlasticity,
     stdp_measurably_depresses_the_weight_over_a_real_glif_run_with_a_non_trivial_delay_ring) {
    const f32 resting_mp = 0.0f;
    const f32 decay_rate = 0.1f; // gL
    const f32 spike_threshold = 1.0f;
    const f32 dt = 1.0f;
    const f32 external_pulse = 1.2f; // isolated single-spike pulse (see this test's own header comment)
    const f32 delivered_constant_weight = 0.1f; // well below spike_threshold, never causes a re-spike
    const s64 total_neuron_count = 2; // 0 = presynaptic (source), 1 = postsynaptic (target)
    const s64 delay_ticks = 5;        // non-trivial per-edge delay
    const s64 postsynaptic_pulse_tick = 2;
    const s64 presynaptic_pulse_tick = 3;
    const s64 tick_count = 12;
    const f32 stdp_learning_rate = 0.02f;

    vector<vector<s32>> adjacency = {{1}, {}}; // neuron 0 -> neuron 1 only

    ModelSpecification model;
    model.total_neuron_count = (s32)total_neuron_count;
    model.type_library.push_back(build_lif_equivalent_type_entry(
        "LifEquivalentCell", "lifEquivalentInstance", 1.0f, decay_rate, resting_mp, spike_threshold));

    PopulationEntry population;
    population.id = "Pop";
    population.type_library_index = 0;
    population.size = (s32)total_neuron_count;
    population.neuron_index_begin = 0;
    population.neuron_index_end = (s32)total_neuron_count;
    model.populations.push_back(population);

    // Read only by allocate_delay_ring's own sizing/defaulting below -- AssembledModel itself
    // ignores model.projections entirely in ring mode (ticket #131's own established precedent, see
    // master_kernel.h), so this is not a second, uncoordinated writer of the shared U/V basis.
    ProjectionEntry projection;
    projection.id = "Proj";
    projection.presynaptic_population_index = 0;
    projection.postsynaptic_population_index = 0;
    ConnectionEntry connection;
    connection.source_neuron_index = 0;
    connection.target_neuron_index = 1;
    connection.weight = 1.0;
    connection.delay = (f64)delay_ticks * (f64)dt;
    projection.connections.push_back(connection);
    model.projections.push_back(projection);

    spikecorec::Vector<IrProgram> programs = {
        build_lif_equivalent_program("LifEquivalentCell", decay_rate, resting_mp, spike_threshold)};

    ModelAllocation allocation = allocate_model(model, programs);
    std::fill(allocation.cell_state.get_contents(), allocation.cell_state.get_contents() + total_neuron_count,
              resting_mp);

    WeightMatrix weights(adjacency, /*rank=*/8, /*check_indexing=*/true, /*max_neighbor_count=*/-1,
                          /*weight_seed=*/7);
    weights.set_constant_weight(delivered_constant_weight);
    weights.using_constant_weight = true;

    DelayRingAllocation ring = allocate_delay_ring(model, weights, /*dt_seconds=*/dt);
    ASSERT_EQ(ring.ring_slot_count, delay_ticks + 1);

    AssembledModel assembled_model(model, programs, /*enable_delay_ring=*/true);
    assembled_model.enable_plasticity(stdp_learning_rate);
    ASSERT_TRUE(assembled_model.plasticity_enabled());

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

    const f32 weight_before = weights.get(0, 1);

    vector<f32> network_input_at_target_by_tick((usize)tick_count, 0.0f);
    for (s64 tick = 0; tick < tick_count; ++tick) {
        s64 slot = tick % ring.ring_slot_count;
        network_input_at_target_by_tick[(usize)tick] =
            ring.input_ring.get_contents()[slot * total_neuron_count + 1];

        if (tick == postsynaptic_pulse_tick) allocation.cell_state.get_contents()[1] += external_pulse;
        if (tick == presynaptic_pulse_tick) allocation.cell_state.get_contents()[0] += external_pulse;

        assembled_model.step_tick(buffers, dt, tick, tick + 1);
    }

    const f32 weight_after = weights.get(0, 1);

    // Sanity: this is real GLIF cell dynamics, not hand-set spike times -- each neuron fired exactly
    // once, at its own controlled tick (the tiny 0.1 delayed synaptic bump neuron 1 receives at
    // tick 8 is far below spike_threshold=1.0 against its own decaying trajectory -- see this test's
    // own header comment).
    EXPECT_EQ(last_spiked.get_contents()[1], postsynaptic_pulse_tick);
    EXPECT_EQ(last_spiked.get_contents()[0], presynaptic_pulse_tick);

    // The non-trivial per-edge delay really is exercised: the presynaptic spike's contribution lands
    // at exactly presynaptic_pulse_tick + delay_ticks, and nowhere else.
    for (s64 tick = 0; tick < tick_count; ++tick) {
        f32 expected = (tick == presynaptic_pulse_tick + delay_ticks) ? delivered_constant_weight : 0.0f;
        EXPECT_NEAR(network_input_at_target_by_tick[(usize)tick], expected, 1e-6f) << "tick=" << tick;
    }

    // ── this ticket's own acceptance criterion: STDP measurably moves the weight, in the only
    // direction the real kernel's own pow(tick_delta, -3) shape ever nudges it (depression) --
    // see plasticity_wiring.h's own investigation into step_apply_hebbian_update's real sign
    // convention, which WeightMatrix::update() (what apply_stdp_plasticity calls) reproduces exactly
    // for a single iteration.
    EXPECT_LT(weight_after, weight_before);
}
