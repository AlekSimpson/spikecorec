#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include "spikecorec/core/engine.h"
#include "spikecorec/nml/cell_lowering.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/stimulus_schedule.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── SpikeEngine's new NML-model constructor / step_tick (Stage 1 of folding
// nml::AssembledModel into SpikeEngine -- see the "REFACTOR" comments above
// class AssembledModel in include/spikecorec/nml/master_kernel.h) ──────────
//
// Nothing calls SpikeEngine's new ModelSpecification constructor yet (that
// migration is a later stage), so this is the first real, end-to-end
// evidence the new code path works: it drives the same GLIF3 single-cell
// fixture tests/exit_model_validation_tests.cpp already validates against a
// real pyneuroml/jNeuroML reference, but through SpikeEngine directly
// instead of nml::AssembledModel -- a genuine smoke test, not just "doesn't
// crash" (asserts a finite trace, real movement away from rest, and a
// spike count in the same qualitative range the existing AssembledModel-
// based exit-model test already established for this exact fixture).

namespace {

String fixture_path(const String &relative_path) {
    return String(SPIKECOREC_TEST_FIXTURES_DIR) + "/" + relative_path;
}

// Mirrors tests/exit_model_validation_tests.cpp's own load_model_from_nml_fixture (a small,
// self-contained copy rather than a shared dependency -- this file's own header comment on why
// every *_tests.cpp/example duplicates this handful of lines instead of factoring it out applies
// here too).
ModelSpecification load_model_from_nml_fixture(const String &fixture_base_name) {
    NML_Parser parser;
    parser.parse(fixture_path("nml/" + fixture_base_name + "_top.nml"));
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

// Mirrors tests/exit_model_validation_tests.cpp's own build_type_library_ir_programs: one IrProgram
// per model.type_library entry, in the same order (an Inputs entry -- glif3_single_cell's own
// pulseGen1 -- gets an empty placeholder, matching Phase-1's host-precomputed pulseGenerator path).
vector<IrProgram> build_type_library_ir_programs(const ModelSpecification &model) {
    vector<IrProgram> programs;
    programs.reserve(model.type_library.size());
    for (const auto &entry : model.type_library) {
        if (entry.category == TypeLibraryCategory::Cell) {
            programs.push_back(lower_cell_to_ir(entry));
        } else {
            IrProgram placeholder;
            placeholder.component_type_name = entry.component_type_name;
            programs.push_back(std::move(placeholder));
        }
    }
    return programs;
}

// "v" is always state slot 0 for GLIF3Cell (declared first in its own <Dynamics>) -- mirrors
// exit_model_validation_tests.cpp's own v_state_index.
s64 v_state_index(const ModelAllocation &allocation, s32 population_index, s32 local_index) {
    return allocation.cell_type_boundaries.get_contents()[population_index] + local_index;
}

// allocate_model zero-initializes cell_state; it does not apply a cell type's own OnStart -- every
// GLIF variant declares `OnStart: v = EL`, seeded here by hand (mirrors
// exit_model_validation_tests.cpp's own seed_initial_membrane_potentials).
void seed_initial_membrane_potentials(ModelAllocation &allocation, const ModelSpecification &model) {
    for (s32 population_index = 0; population_index < (s32)model.populations.size(); ++population_index) {
        const PopulationEntry &population = model.populations[(usize)population_index];
        f64 resting_potential = model.type_library[(usize)population.type_library_index].baked_constants.at("EL");
        for (s32 local_index = 0; local_index < population.size; ++local_index) {
            allocation.cell_state.get_contents()[v_state_index(allocation, population_index, local_index)] =
                (f32)resting_potential;
        }
    }
}

} // namespace

TEST(SpikeEngineNmlConstruction, constructs_without_throwing_from_a_real_glif3_model) {
    ModelSpecification model = load_model_from_nml_fixture("glif3_single_cell");
    vector<IrProgram> programs = build_type_library_ir_programs(model);

    ASSERT_NO_THROW({ SpikeEngine engine(model, programs); });
}

TEST(SpikeEngineNmlConstruction, driven_glif3_single_cell_produces_real_membrane_movement_and_spikes) {
    ModelSpecification model = load_model_from_nml_fixture("glif3_single_cell");
    vector<IrProgram> programs = build_type_library_ir_programs(model);

    SpikeEngine engine(model, programs);
    seed_initial_membrane_potentials(engine.nml_allocation_, model);

    const s64 tick_count = 3500;
    const f32 dt_seconds = 1e-4f;
    StimulusSchedule schedule = build_stimulus_schedule(model, (f64)dt_seconds);

    vector<f32> membrane_trace;
    membrane_trace.reserve((usize)tick_count);
    vector<s64> spike_ticks;

    for (s64 tick = 0; tick < tick_count; ++tick) {
        engine.network_inputs.get_contents()[0] += (f32)schedule.current_at(0, tick);
        engine.step_tick(dt_seconds, tick, tick + 1);

        membrane_trace.push_back(engine.nml_allocation_.cell_state.get_contents()[v_state_index(engine.nml_allocation_, 0, 0)]);
        if (engine.last_spiked.get_contents()[0] == tick) spike_ticks.push_back(tick);
    }

    ASSERT_EQ(membrane_trace.size(), (usize)tick_count);

    f32 minimum_voltage = membrane_trace[0];
    f32 maximum_voltage = membrane_trace[0];
    for (f32 voltage : membrane_trace) {
        ASSERT_TRUE(std::isfinite(voltage));
        ASSERT_LT(std::abs(voltage), 10.0f) << "membrane potential diverged";
        minimum_voltage = std::min(minimum_voltage, voltage);
        maximum_voltage = std::max(maximum_voltage, voltage);
    }

    // Real, non-trivial movement away from rest -- not just "didn't crash".
    EXPECT_GT(maximum_voltage - minimum_voltage, 1e-4f);

    // Same loose range tests/exit_model_validation_tests.cpp's own
    // ExitModelGlif3SingleCell.driven_simulation_produces_spike_frequency_adaptation asserts for
    // this exact fixture/tick_count/dt through nml::AssembledModel -- the real pyneuroml reference
    // fires 13 times under these exact parameters.
    EXPECT_GE(spike_ticks.size(), 5u);
    EXPECT_LE(spike_ticks.size(), 25u);
}

// ── SpikeEngine's Stage 2 (real per-edge synapse dispatch, ticket #131 + real STDP, ticket #132) ──
//
// The fixtures below mirror tests/master_kernel_tests.cpp's own build_lif_equivalent_program/
// build_lif_equivalent_type_entry/build_conductance_synapse_program/build_conductance_synapse_type_
// entry (kept as its own copy here, matching this codebase's established per-file fixture
// convention) -- ported off nml::AssembledModel/ModelRuntimeBuffers onto SpikeEngine's own
// ModelSpecification constructor + step_tick, to prove this stage's actual point: real per-edge
// synapse dispatch (not a placeholder/zeroed weight matrix) genuinely runs through SpikeEngine now.

namespace {

String precise_float_literal(f32 value) {
    ostringstream stream;
    stream << std::setprecision(9) << value;
    return stream.str();
}

// A minimal, self-contained GLIF1-shaped cell program: `dv/dt = (gL*(EL-v) + network_inputs) / C`
// (forward Euler, one dt per tick), `spike` on `v > vth`, no `@reset`.
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

// Mirrors synapse_lowering.cpp's own generated shape for a conductance-based synapse (arch §4.3;
// ir_spec.md §4's expOne example) -- `g` bumped by `gbase` on delivery, decaying with time constant
// `tauDecay`, contributing `g*(erev-v)` into `network_inputs` every tick via `_integrate_edges`.
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
// conductance-based synapse edge, built and run through SpikeEngine's own ModelSpecification
// constructor + step_tick directly -- exercising Stage 2's real per-edge synapse dispatch path (no
// ModelRuntimeBuffers/AssembledModel involved at all). Neuron 0 is driven over threshold by a single
// large external pulse on tick 1; returns neuron 1's own membrane-potential trajectory, one sample
// per tick.
ModelSpecification build_two_neuron_conductance_synapse_model(f32 gbase, vector<IrProgram> &out_programs,
                                                                f32 decay_rate, f32 resting_mp, f32 spike_threshold,
                                                                f32 erev, f32 tau_decay) {
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

    out_programs = {
        build_lif_equivalent_program("PostCell", decay_rate, resting_mp, spike_threshold),
        build_conductance_synapse_program("CondSynapse", gbase, erev, tau_decay),
    };
    return model;
}

vector<f32> run_two_neuron_conductance_synapse_network_through_spike_engine(f32 gbase) {
    const f32 resting_mp = 0.0f;
    const f32 decay_rate = 0.1f;
    const f32 spike_threshold = 1.0f;
    const f32 erev = 2.0f;
    const f32 tau_decay = 2.0f;
    const f32 dt = 1.0f;
    const s64 tick_count = 20;

    vector<IrProgram> programs;
    ModelSpecification model = build_two_neuron_conductance_synapse_model(
        gbase, programs, decay_rate, resting_mp, spike_threshold, erev, tau_decay);

    SpikeEngine engine(model, programs);

    vector<f32> postsynaptic_trajectory;
    postsynaptic_trajectory.reserve((usize)tick_count);
    for (s64 tick = 0; tick < tick_count; ++tick) {
        if (tick == 1) engine.nml_allocation_.cell_state.get_contents()[0] += 5.0f; // drives neuron 0 over threshold
        engine.step_tick(dt, tick, tick + 1);
        postsynaptic_trajectory.push_back(engine.nml_allocation_.cell_state.get_contents()[1]);
    }
    return postsynaptic_trajectory;
}

} // namespace

TEST(SpikeEngineNmlSynapseDispatch,
     real_per_edge_synapse_dispatch_measurably_perturbs_postsynaptic_membrane_potential) {
    vector<f32> weak_synapse_trajectory =
        run_two_neuron_conductance_synapse_network_through_spike_engine(/*gbase=*/0.5f);
    vector<f32> strong_synapse_trajectory =
        run_two_neuron_conductance_synapse_network_through_spike_engine(/*gbase=*/2.0f);
    ASSERT_EQ(weak_synapse_trajectory.size(), strong_synapse_trajectory.size());

    // The postsynaptic neuron must have actually moved away from rest under the strong-gbase
    // network -- proving the synapse's own current genuinely reached it through SpikeEngine's own
    // step_tick, not merely "technically different by float noise".
    bool postsynaptic_neuron_moved_under_strong_gbase = false;
    f32 maximum_deviation_from_rest = 0.0f;
    for (f32 voltage : strong_synapse_trajectory) {
        maximum_deviation_from_rest = std::max(maximum_deviation_from_rest, std::fabs(voltage));
        if (std::fabs(voltage) > 1e-3f) postsynaptic_neuron_moved_under_strong_gbase = true;
    }
    EXPECT_TRUE(postsynaptic_neuron_moved_under_strong_gbase)
        << "postsynaptic neuron never moved away from rest -- the synapse's own current never reached "
           "it through SpikeEngine::step_tick's real per-edge synapse dispatch path";

    // The two otherwise-identical networks (same cells, same adjacency, same dt, same external pulse)
    // must diverge measurably once gbase differs -- ruling out a zeroed/no-op weight-matrix pass-through.
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

    // Report the measured perturbation for the record (not just "didn't crash").
    std::cout << "[SpikeEngineNmlSynapseDispatch] gbase=0.5 postsynaptic trajectory: ";
    for (f32 voltage : weak_synapse_trajectory) std::cout << voltage << " ";
    std::cout << "\n[SpikeEngineNmlSynapseDispatch] gbase=2.0 postsynaptic trajectory: ";
    for (f32 voltage : strong_synapse_trajectory) std::cout << voltage << " ";
    std::cout << "\n[SpikeEngineNmlSynapseDispatch] max |postsynaptic voltage| under gbase=2.0: "
              << maximum_deviation_from_rest << std::endl;
}

// ── ticket #132's guard: enable_plasticity must refuse to run alongside real per-edge synapse
// dispatch -- mirrors AssembledModelPlasticity.enable_plasticity_throws_when_real_per_edge_synapse_
// dispatch_is_active (tests/assembled_model_plasticity_tests.cpp), ported onto SpikeEngine.
TEST(SpikeEngineNmlPlasticity, enable_plasticity_throws_when_real_per_edge_synapse_dispatch_is_active) {
    const f32 resting_mp = 0.0f;
    const f32 decay_rate = 0.1f;
    const f32 spike_threshold = 1.0f;
    const f32 gbase = 1.0f, erev = 2.0f, tau_decay = 2.0f;

    vector<IrProgram> programs;
    ModelSpecification model = build_two_neuron_conductance_synapse_model(
        gbase, programs, decay_rate, resting_mp, spike_threshold, erev, tau_decay);

    SpikeEngine engine(model, programs); // real, non-empty projections

    EXPECT_THROW(engine.enable_plasticity(0.02f), std::runtime_error);
    EXPECT_FALSE(engine.plasticity_enabled());
}

// ── SpikeEngine's hand-written move constructor / move-assignment ────────────────────────────────
//
// Regression coverage: a real, live NML-mode SpikeEngine holds two KernelHandle scalars
// (nml_drain_kernel_/nml_propagate_kernel_ -- see engine.h) that have no move semantics of their
// own. A defaulted move constructor/assignment would bitwise-copy them into the destination
// without clearing the source, and would leave the moved-from source's alive/nml_mode_enabled_
// flags unchanged (both true for a live NML-mode engine) -- so the moved-from object's destructor
// would call shutdown() again and release/delete the same kernels (and, for move-assignment,
// nml_allocation_'s own GPU buffers) the destination still legitimately owns: a double-release/
// double-free. These tests build a genuinely live NML-mode engine (real compiled kernels, real
// per-edge synapse dispatch topology built via step_tick -- exactly the state
// SpikeEngineNmlSynapseDispatch above exercises) specifically so the move has real GPU resources
// to mishandle, then destroy both the moved-from and moved-to engines -- a double-release/
// double-free would show up as a crash (segfault) or an ASan heap-corruption abort here, not a
// clean gtest failure, so "this test doesn't crash" is itself the assertion.

TEST(SpikeEngineNmlMoveSafety, move_construction_does_not_double_release_nml_resources) {
    vector<IrProgram> programs;
    ModelSpecification model = build_two_neuron_conductance_synapse_model(
        /*gbase=*/1.0f, programs, /*decay_rate=*/0.1f, /*resting_mp=*/0.0f, /*spike_threshold=*/1.0f,
        /*erev=*/2.0f, /*tau_decay=*/2.0f);

    SpikeEngine original_engine(model, programs);
    // Drives the lazy synapse-dispatch-topology build (ensure_nml_synapse_dispatch_topology_built)
    // before the move, so the engine being moved genuinely owns real per-edge GPU topology buffers
    // and compiled synapse-type kernels too, not just the two fixed cell-tick kernels.
    original_engine.step_tick(1.0f, 0, 1);

    SpikeEngine moved_engine(std::move(original_engine));

    // The moved-to engine must still be fully functional (proves the move actually transferred
    // ownership, not just avoided crashing).
    ASSERT_NO_THROW({ moved_engine.step_tick(1.0f, 1, 2); });
}

TEST(SpikeEngineNmlMoveSafety, move_assignment_does_not_double_release_nml_resources) {
    vector<IrProgram> programs_one;
    ModelSpecification model_one = build_two_neuron_conductance_synapse_model(
        /*gbase=*/1.0f, programs_one, /*decay_rate=*/0.1f, /*resting_mp=*/0.0f, /*spike_threshold=*/1.0f,
        /*erev=*/2.0f, /*tau_decay=*/2.0f);
    SpikeEngine engine_one(model_one, programs_one);
    engine_one.step_tick(1.0f, 0, 1); // build engine_one's own real synapse-dispatch topology + kernels

    vector<IrProgram> programs_two;
    ModelSpecification model_two = build_two_neuron_conductance_synapse_model(
        /*gbase=*/2.0f, programs_two, /*decay_rate=*/0.1f, /*resting_mp=*/0.0f, /*spike_threshold=*/1.0f,
        /*erev=*/2.0f, /*tau_decay=*/2.0f);
    SpikeEngine engine_two(model_two, programs_two);
    engine_two.step_tick(1.0f, 0, 1); // build engine_two's own real synapse-dispatch topology + kernels
                                      // (its own live resources the move-assignment must release,
                                      // not leak, before absorbing engine_one's state)

    engine_two = std::move(engine_one);

    // The move-assigned-to engine must still be fully functional.
    ASSERT_NO_THROW({ engine_two.step_tick(1.0f, 1, 2); });
}
