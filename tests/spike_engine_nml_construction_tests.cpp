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
#include "spikecorec/nml/delay_ring.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/plasticity_wiring.h"
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

// ── delay-ring fold (SpikeEngine-only) -- see the three REFACTOR comments in delay_ring.h/
// master_kernel.h/master_kernel.cpp this generalizes. network_inputs (and the active-set enqueue
// bookkeeping) is now ring-shaped [ring_slot_count * neuron_count] on SpikeEngine, with
// ring_slot_count derived from WeightMatrix's own constant_delay_ticks/edge_delay_ticks at NML
// construction time -- ring_slot_count == 1 (no real per-edge delay beyond the engine's existing
// implicit one-tick latency) must collapse to byte-identical behavior to the pre-fold flat path;
// ring_slot_count > 1 (a real per-edge delay configured) must deliver a scattered spike EXACTLY
// that many ticks later, not sooner and not later.
//
// The ring_slot_count == 1 case below reuses SpikeEngineNmlSynapseDispatch's own two-neuron
// conductance-synapse fixture (build_two_neuron_conductance_synapse_model/
// run_two_neuron_conductance_synapse_network_through_spike_engine, defined above in this same
// file) rather than a bare, synapse-free projection: engine.cpp's constructor validates every
// ProjectionEntry's synapse_type_library_index whenever nml_ring_slot_count_ == 1 (real per-edge
// synapse dispatch, ticket #131, is only ever force-disabled for a REAL per-edge delay, ring_slot_
// count > 1 -- see engine.cpp's own constructor doc comment), so a projection representable with
// ring_slot_count == 1 needs a real synapse type either way, exactly like every other existing
// projection-bearing fixture in this file already provides. The real per-edge synapse dispatch
// path already carries the SAME >=1-tick network_inputs latency as the plain scalar-weight path
// (ir_spec.md §3.5) -- delivery + integrate both run within the SOURCE neuron's own firing tick,
// so the TARGET neuron cannot observe it until its own NEXT tick's dispatch -- so this is still a
// faithful "ring_slot_count == 1 matches the flat one-tick latency" proof.
//
// The ring_slot_count > 1 case below needs its OWN, separate, synapse-free fixture (real per-edge
// synapse dispatch is force-disabled for that case, so a re-used conductance-synapse model would
// never even exercise the delay ring at all) -- build_two_neuron_delay_ring_model, a plain
// (no real per-edge synapse type) projection relying on the FIXED scalar propagate stage's own
// constant_weight instead. The engine's constructor now converts ConnectionEntry::delay (real SI
// seconds) to whole ticks against its own `dt_seconds` constructor argument (engine.cpp's
// nml_delay_seconds_to_ticks) -- this fixture's own dt=1.0 convention (every step_tick call below
// passes dt=1.0f) means passing dt_seconds=1.0f to the engine's constructor too makes that
// conversion an exact identity, so `connection_delay_ticks` below can still be handed straight to
// ConnectionEntry::delay and read back as that same whole-tick count. Only valid for a delay that
// converts to > 1 tick (forcing nml_ring_slot_count_ > 1, and so nml_projections_ empty, and so its
// own bogus synapse_type_library_index never gets validated) -- NOT a general-purpose "any delay"
// fixture.

namespace {

// A minimal 2-neuron model: neuron 0 -> neuron 1 through a plain (no real per-edge synapse type)
// projection carrying `connection_delay_ticks`. Reuses build_lif_equivalent_program/
// build_lif_equivalent_type_entry (defined above in this same file). Parameters are engineered so
// neuron 0 fires EXACTLY ONCE, at tick 1, from a single external boost: post-boost v=5.0 decays in
// ONE integrate step to 5.0+1.0*(gL*(EL-5.0))/C = 5.0-2.5 = 2.5 (with gL=0.5, EL=0.0, C=1.0,
// dt=1.0) -- still above vth=2.0, so it fires at tick 1 -- then to 2.5-1.25=1.25 the NEXT tick,
// below vth=2.0, so it never fires again. A single, unambiguous source spike pins down this ring's
// own delivery latency exactly (no risk of a second, later spike's own delayed arrival confusing
// the assertion). `connection_delay_ticks` MUST round to > 1 (see this section's own header
// comment above for why).
ModelSpecification build_two_neuron_delay_ring_model(f64 connection_delay_ticks, vector<IrProgram> &out_programs) {
    const f32 decay_rate = 0.5f;      // gL
    const f32 resting_mp = 0.0f;      // EL
    const f32 spike_threshold = 2.0f; // vth

    ModelSpecification model;
    model.total_neuron_count = 2;
    model.type_library.push_back(build_lif_equivalent_type_entry(
        "DelayRingCell", "delayRingInstance", 1.0f, decay_rate, resting_mp, spike_threshold));

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
    projection.connections.push_back(
        ConnectionEntry{/*source=*/0, /*target=*/1, /*weight=*/1.0, /*delay=*/connection_delay_ticks});
    model.projections.push_back(projection);

    out_programs = {build_lif_equivalent_program("DelayRingCell", decay_rate, resting_mp, spike_threshold)};
    return model;
}

// Drives neuron 0 over threshold with a single external boost at tick 1 (see
// build_two_neuron_delay_ring_model's own doc comment) and returns neuron 1's own
// membrane-potential trajectory, one sample per tick, over `tick_count` ticks.
vector<f32> run_two_neuron_delay_ring_network_through_spike_engine(f64 connection_delay_ticks, s64 tick_count,
                                                                    f32 constant_weight) {
    vector<IrProgram> programs;
    ModelSpecification model = build_two_neuron_delay_ring_model(connection_delay_ticks, programs);

    // dt_seconds=1.0f matches this fixture's own dt=1.0 convention (every step_tick call below), so
    // the constructor's real seconds->ticks delay conversion is an identity against
    // `connection_delay_ticks` (see this section's own header comment above).
    SpikeEngine engine(model, programs, 1.0f);
    engine.weights.set_constant_weight(constant_weight);

    vector<f32> postsynaptic_trajectory;
    postsynaptic_trajectory.reserve((usize)tick_count);
    for (s64 tick = 0; tick < tick_count; ++tick) {
        if (tick == 1) engine.nml_allocation_.cell_state.get_contents()[0] += 5.0f; // drives neuron 0 over threshold
        engine.step_tick(1.0f, tick, tick + 1);
        postsynaptic_trajectory.push_back(engine.nml_allocation_.cell_state.get_contents()[1]);
    }
    return postsynaptic_trajectory;
}

} // namespace

TEST(SpikeEngineNmlDelayRing, ring_slot_count_one_no_configured_delay_matches_the_flat_one_tick_latency) {
    // delay=0.0 on this fixture's one connection (build_two_neuron_conductance_synapse_model,
    // defined above in this same file) -- the "no delay attribute given" default -- rounds to <= 1,
    // so this engine's constructor never touches weights' delay config: nml_ring_slot_count_ stays
    // at its own default of 1 (see this section's own header comment for why this reuses that
    // fixture rather than build_two_neuron_delay_ring_model below).
    vector<f32> trajectory = run_two_neuron_conductance_synapse_network_through_spike_engine(/*gbase=*/2.0f);

    // Neuron 0 fires at tick 1 (driven by that fixture's own "+5.0f at tick 1" boost) -- the
    // engine's existing implicit >=1-tick network_inputs latency (CLAUDE.md) means neuron 1 must
    // stay at rest (EL=0.0) through tick 1 and move starting at tick 2, not before.
    ASSERT_GT(trajectory.size(), 2u);
    for (s64 tick = 0; tick <= 1; ++tick) {
        EXPECT_NEAR(trajectory[(usize)tick], 0.0f, 1e-6f)
            << "neuron 1 moved before the expected 1-tick-delayed arrival (tick " << tick << ")";
    }
    EXPECT_GT(std::fabs(trajectory[2]), 1e-3f)
        << "neuron 1 never received neuron 0's spike at the expected 1-tick-delayed arrival (tick 2)";

    std::cout << "[SpikeEngineNmlDelayRing] ring_slot_count==1 postsynaptic trajectory: ";
    for (f32 voltage : trajectory) std::cout << voltage << " ";
    std::cout << std::endl;
}

TEST(SpikeEngineNmlDelayRing, ring_shaped_path_delivers_a_spike_exactly_delay_ticks_ticks_later) {
    const s32 delay_ticks = 5;
    const s64 tick_count = 12;
    vector<f32> trajectory = run_two_neuron_delay_ring_network_through_spike_engine(
        /*connection_delay_ticks=*/(f64)delay_ticks, tick_count, /*constant_weight=*/3.0f);

    // Neuron 0 fires at tick 1 -- the configured 5-tick delay means neuron 1 must stay at rest
    // (EL=0.0) through tick 5 and move starting at tick 6 (1 + 5), not any tick before.
    const s64 expected_arrival_tick = 1 + delay_ticks;
    ASSERT_LT(expected_arrival_tick, tick_count);
    for (s64 tick = 0; tick < expected_arrival_tick; ++tick) {
        EXPECT_NEAR(trajectory[(usize)tick], 0.0f, 1e-6f)
            << "neuron 1 moved BEFORE the configured " << delay_ticks << "-tick delay elapsed (tick " << tick << ")";
    }
    EXPECT_GT(std::fabs(trajectory[(usize)expected_arrival_tick]), 1e-3f)
        << "neuron 1 never received neuron 0's spike at the expected delayed arrival tick ("
        << expected_arrival_tick << ")";

    std::cout << "[SpikeEngineNmlDelayRing] ring_slot_count==" << delay_ticks << " postsynaptic trajectory: ";
    for (f32 voltage : trajectory) std::cout << voltage << " ";
    std::cout << std::endl;
}

// ── real NML per-connection delay -> whole-tick conversion ────────────────────────────────────────
//
// Both fixtures above manually build a ModelSpecification whose ConnectionEntry::delay is either
// exactly 0.0 (build_two_neuron_conductance_synapse_model) or handed to the engine's constructor
// alongside a dt_seconds=1.0f chosen specifically to make the seconds->ticks conversion an identity
// (build_two_neuron_delay_ring_model's own doc comment above) -- neither actually exercises a REAL
// NeuroML `delay` attribute's unit conversion against a real dt_seconds. This test drives an actual
// NeuroML model (tests/fixtures/nml/delayed_coupling_network.nml -- the SAME fixture
// tests/exit_model_validation_tests.cpp's own run_delayed_coupling_network validates against a real
// jLEMS reference) through the real parse/resolve/lower pipeline and SpikeEngine's own NML
// constructor with a real dt_seconds, proving engine.cpp's nml_delay_seconds_to_ticks genuinely
// converts a real `delay="10ms"` NeuroML attribute (already unit-resolved to 0.010 SI seconds by
// resolve.cpp) into exactly 100 whole ticks at this fixture's own 0.1ms step -- not just "didn't
// throw".
TEST(SpikeEngineNmlDelayRing,
     real_nml_connection_delay_converts_to_the_correct_whole_tick_count_and_delivers_on_time) {
    ModelSpecification model = load_model_from_nml_fixture("delayed_coupling_network");
    ASSERT_EQ(model.projections.size(), 1u);
    ASSERT_EQ(model.projections[0].connections.size(), 1u);
    // connectionWD's own delay="10ms", already unit-resolved to SI seconds by resolve.cpp.
    EXPECT_NEAR(model.projections[0].connections[0].delay, 0.010, 1e-9);

    vector<IrProgram> programs = build_type_library_ir_programs(model);

    const f32 dt_seconds = 1e-4f;         // matches this fixture's own <Simulation step="0.1ms">
    const s64 expected_delay_ticks = 100; // 10ms / 0.1ms

    // Cross-check against delay_ring.cpp's own, separate, already-tested seconds->ticks conversion
    // (compute_max_delay_ticks) before even constructing the engine.
    ASSERT_EQ(compute_max_delay_ticks(model, dt_seconds), expected_delay_ticks);

    SpikeEngine engine(model, programs, dt_seconds);
    seed_initial_membrane_potentials(engine.nml_allocation_, model);
    // Arbitrary nonzero placeholder -- this test's own target is delivery TIMING (derived purely
    // from the connection's own delay), not magnitude (mirrors exit_model_validation_tests.cpp's
    // own run_delayed_coupling_network, which uses this exact same fixture/placeholder).
    engine.weights.set_constant_weight(0.6f);

    // Exactly one real connection in this whole model, so the uniform-delay path applies
    // (SpikeEngine's constructor, engine.cpp): weights.set_constant_delay_ticks(...), not the
    // per-edge array.
    ASSERT_TRUE(engine.weights.using_constant_delay_ticks);
    EXPECT_EQ(engine.weights.constant_delay_ticks, expected_delay_ticks)
        << "measured: a real delay=\"10ms\" NeuroML attribute at dt_seconds=1e-4 produced "
        << "constant_delay_ticks=" << engine.weights.constant_delay_ticks
        << " (expected exactly " << expected_delay_ticks << ")";
    // == this engine's own private nml_ring_slot_count_ in the uniform-delay case (see
    // nml_compute_ring_slot_count_from_weight_matrix, engine.cpp).
    const s64 ring_slot_count = engine.weights.constant_delay_ticks;

    // Manual stimulus reconstruction (the front end does not yet recognize inputList/pulseGenerator
    // -- see ExitModelDelayedCouplingNetwork.front_end_does_not_recognize_inputList_yet_documented_
    // gap, exit_model_validation_tests.cpp): delayed_coupling_network.nml's own <pulseGenerator
    // id="pulseGen1" delay="10ms" duration="6ms" amplitude="0.5nA"/>, applied to SourcePop's neuron 0
    // (global neuron index 0, declared first).
    const s64 tick_count = 600;             // 60ms / 0.1ms, matching this fixture's own
                                             // <Simulation length="60ms">
    const s64 stimulus_delay_ticks = 100;   // pulseGen1 delay="10ms"
    const s64 stimulus_duration_ticks = 60; // pulseGen1 duration="6ms"
    const f32 amplitude_amperes = 0.5e-9f;
    const s32 source_neuron_index = 0; // SourcePop, declared first
    const s32 target_neuron_index = 1; // TargetPop, declared second
    const f32 delivery_epsilon = 1e-9f;

    vector<s64> source_spike_ticks;
    vector<s64> target_delivery_ticks;
    for (s64 tick = 0; tick < tick_count; ++tick) {
        s64 current_ring_slot = tick % ring_slot_count;
        // This tick's own ring slot, read BEFORE this tick's own stimulus/step_tick call so it
        // reflects only whatever a PRIOR tick's propagate stage already scattered into it.
        f32 target_network_input_this_tick =
            engine.network_inputs.get_contents()[current_ring_slot * engine.neuron_count + target_neuron_index];
        if (std::fabs(target_network_input_this_tick) > delivery_epsilon) target_delivery_ticks.push_back(tick);

        if (tick >= stimulus_delay_ticks && tick < stimulus_delay_ticks + stimulus_duration_ticks) {
            engine.network_inputs.get_contents()[current_ring_slot * engine.neuron_count + source_neuron_index] +=
                amplitude_amperes;
        }

        engine.step_tick(dt_seconds, tick, tick + 1);
        if (engine.last_spiked.get_contents()[source_neuron_index] == tick) source_spike_ticks.push_back(tick);
    }

    ASSERT_EQ(source_spike_ticks.size(), 1u);
    ASSERT_EQ(target_delivery_ticks.size(), 1u);
    EXPECT_EQ(target_delivery_ticks[0], source_spike_ticks[0] + expected_delay_ticks)
        << "measured source spike tick=" << source_spike_ticks[0]
        << " measured target delivery tick=" << target_delivery_ticks[0]
        << " (expected delivery exactly " << expected_delay_ticks << " ticks later)";

    std::cout << "[SpikeEngineNmlDelayRing] real NML delay=\"10ms\" at dt_seconds=1e-4 -> "
              << "constant_delay_ticks=" << engine.weights.constant_delay_ticks
              << ", source_spike_tick=" << source_spike_ticks[0]
              << ", target_delivery_tick=" << target_delivery_ticks[0] << std::endl;
}

// ── Ported from tests/master_kernel_tests.cpp's own MasterKernel suite (folding nml::AssembledModel
// into SpikeEngine, see master_kernel.h's own "REFACTOR" comments above class AssembledModel) --
// these four tests exercised AssembledModel's STATEFUL dispatch behavior; same test intent/
// assertions as the originals, just driving SpikeEngine's own ModelSpecification constructor +
// step_tick instead of AssembledModel + a hand-built ModelRuntimeBuffers. master_kernel_tests.cpp's
// own MasterKernel.compile_kernel_or_throw_with_source_.../assembles_a_two_population_.../
// a_population_whose_cell_type_.../collect_emit_port_names_... tests are NOT re-ported here: they
// exercise STATELESS FREE FUNCTIONS (compile_kernel_or_throw_with_source/assemble_master_kernel_
// source/collect_emit_port_names) that are staying in master_kernel.h/.cpp untouched -- only the
// STATEFUL AssembledModel class itself is going away.

TEST(SpikeEngineNmlPropagateDispatch, reproduces_hardcoded_lif_membrane_trajectory_and_spike_timing) {
    const f32 resting_mp = 0.0f;
    const f32 decay_rate = 0.1f;
    const f32 spike_threshold = 1.0f;
    const f32 dt = 1.0f;
    const f32 edge_weight = 0.6f;
    const s64 total_neuron_count = 3;
    const s64 tick_count = 6;

    // ── reference path: the existing hardcoded engine (src/metal/kernels.metal `step_no_active_
    // optimization`, dispatched via SpikeEngine::step_simulation with active-set disabled so every
    // neuron runs every tick) -- SpikeEngine's own legacy (non-NML) constructor. ──
    vector<vector<s32>> adjacency = {{1}, {}, {}}; // neuron 0 -> neuron 1 only
    SpikeEngine legacy_engine(&adjacency, {total_neuron_count, 1}, /*rank=*/1, resting_mp, decay_rate,
                              /*learning_rate=*/0.0f, /*plasticity_enabled=*/false,
                              /*active_set_optimization_enabled=*/false);
    // spike_period=0 keeps the hardcoded engine's own "periodic forced reset" branch unreachable
    // after tick 0 while ALSO making its last_spiked update fire on every spiking tick -- matching
    // the NML-mode propagate stage below, which writes last_spiked unconditionally on every `emit`
    // per the locked IR's own EventOut semantics (arch §3.2), not gated by the hardcoded engine's
    // spike_period debounce quirk. This is the one spike_period value where both conventions happen
    // to coincide (see master_kernel_tests.cpp's own original doc comment on this same fixture for
    // the full derivation -- unchanged by this port).
    legacy_engine.spike_period = 0;
    legacy_engine.spike_threshold = spike_threshold;
    legacy_engine.weights.set_constant_weight(edge_weight);
    legacy_engine.use_constant_weight = true;
    legacy_engine.set_input_neurons({0}); // external stimulus targets neuron 0 only

    // ── NML-mode path: a hand-built GLIF1-equivalent ComponentType, driven through SpikeEngine's OWN
    // ModelSpecification constructor + step_tick directly (no AssembledModel/ModelRuntimeBuffers) ──
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

    vector<IrProgram> programs = {
        build_lif_equivalent_program("LifEquivalentCell", decay_rate, resting_mp, spike_threshold)};

    SpikeEngine nml_engine(model, programs, /*dt_seconds=*/1.0f); // no projections at all -- ring_
                                                                   // slot_count stays 1 (the flat,
                                                                   // pre-fold one-tick-latency case)
    // Replaces this engine's own (edge-free, since `model` above has no projections) weights with
    // the SAME adjacency the legacy reference engine above uses -- mirrors the original master_
    // kernel_tests.cpp test's own standalone `WeightMatrix my_weights(adjacency, rank=1)` passed
    // separately via ModelRuntimeBuffers (replacing engine.weights post-construction is safe, see
    // refit_default_matrix_reconstruction_stays_synced_with_a_real_propagate_kernel_dispatch below
    // for the full reasoning).
    nml_engine.weights = WeightMatrix(adjacency, /*rank=*/1);
    nml_engine.weights.set_constant_weight(edge_weight);

    std::fill(nml_engine.nml_allocation_.cell_state.get_contents(),
              nml_engine.nml_allocation_.cell_state.get_contents() + total_neuron_count, resting_mp);

    const f32 external_pulse = 1.5f; // raises neuron 0 above threshold on injection tick's own integrate step

    // Tick 0 is deliberately a zero-input "priming" tick for both paths: the hardcoded engine's own
    // lazy-decay optimization special-cases time_since_last_update <= 0 (a transient the NML path has
    // no reason to reproduce -- it always applies exactly one dt, no notion of "elapsed ticks").
    // Injecting the external pulse one tick later (tick 1) sidesteps that tick-0-only transient
    // rather than asserting equivalence through it.
    for (s64 tick = 0; tick < tick_count; ++tick) {
        f32 this_tick_external_input = (tick == 1) ? external_pulse : 0.0f;

        // legacy engine: external stimulus goes straight into membrane_potentials (arch §0.2), via
        // the same gpu_add_network_input path step_simulation always uses.
        legacy_engine.step_simulation({this_tick_external_input}, tick);

        // NML-mode engine: same external-stimulus convention, added directly into cell_state's "v"
        // slot for neuron 0 before this tick's integrate runs (mirrors gpu_add_network_input's own
        // "external != network_inputs" semantics, arch §0.2).
        nml_engine.nml_allocation_.cell_state.get_contents()[0] += this_tick_external_input;
        nml_engine.step_tick(dt, tick, tick + 1);

        const f32 *legacy_membrane_potentials = legacy_engine.membrane_potentials.get_contents();
        const f32 *nml_v = nml_engine.nml_allocation_.cell_state.get_contents();
        for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
            EXPECT_NEAR(legacy_membrane_potentials[neuron_index], nml_v[neuron_index], 1e-4f)
                << "tick=" << tick << " neuron_index=" << neuron_index;
        }

        // Compared as "fired exactly THIS tick" from tick 1 onward, not as raw last_spiked values:
        // the legacy engine's own "never fired" seed is 0 (engine.cpp's legacy constructor), while
        // SpikeEngine's own NML-mode seed is -1 (engine.cpp's ModelSpecification constructor,
        // engine.h's own doc comment on apply_nml_stdp_plasticity) -- a real, deliberate convention
        // difference this port must account for, not a bug. tick 0 is skipped here specifically
        // because the legacy engine's own "never fired" sentinel (0) coincides with the tick-0 tick
        // number itself, which would otherwise spuriously read as "fired at tick 0" for every neuron
        // regardless of whether it actually did (neither engine genuinely spikes at tick 0 in this
        // fixture -- this tick's own external input is 0, see this test's own "priming tick" comment
        // above); no such ambiguity exists from tick 1 onward, since 0 (legacy's sentinel) can never
        // equal a `tick >= 1` comparison target.
        const s64 *legacy_last_spiked = legacy_engine.last_spiked.get_contents();
        const s64 *nml_last_spiked = nml_engine.last_spiked.get_contents();
        if (tick > 0) {
            for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
                bool legacy_fired_this_tick = (legacy_last_spiked[neuron_index] == tick);
                bool nml_fired_this_tick = (nml_last_spiked[neuron_index] == tick);
                EXPECT_EQ(legacy_fired_this_tick, nml_fired_this_tick)
                    << "tick=" << tick << " neuron_index=" << neuron_index;
            }
        }

        // ring_slot_count == 1 here (no projections at all), so nml_engine.network_inputs addresses
        // exactly like the legacy engine's own flat buffer -- direct element-wise comparison.
        const f32 *legacy_network_inputs = legacy_engine.network_inputs.get_contents();
        const f32 *nml_network_inputs = nml_engine.network_inputs.get_contents();
        for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
            EXPECT_NEAR(legacy_network_inputs[neuron_index], nml_network_inputs[neuron_index], 1e-4f)
                << "tick=" << tick << " neuron_index=" << neuron_index;
        }
    }

    // Sanity: the network actually exercised propagation/spiking within this test horizon (an
    // equivalence test over trajectories that never left resting state would be vacuous).
    bool any_spike_recorded = false;
    for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
        if (legacy_engine.last_spiked.get_contents()[neuron_index] != 0) any_spike_recorded = true;
    }
    EXPECT_TRUE(any_spike_recorded);
}

// ── regression: WeightMatrix::refit() must not desync Ck[DEFAULT_MATRIX_INDEX] from the live
// propagate kernel's all-ones assumption (ticket #103) -- ported from master_kernel_tests.cpp's own
// MasterKernel.refit_default_matrix_reconstruction_stays_synced_with_a_real_propagate_kernel_
// dispatch. The propagate kernel reconstructs an edge's weight via a raw `dot(u_row, v_row)` with no
// coefficient_vectors/Ck parameter at all -- it hardcodes the DEFAULT_MATRIX_INDEX-is-all-ones
// invariant unconditionally. This constructs a SpikeEngine, then REPLACES its own `weights` member
// (public, see engine.h) with a real, irregular network carrying real accumulated Sk deltas (via
// accumulate_edge_delta()) on both DEFAULT_MATRIX_INDEX and a second registered matrix, calls
// refit(), then compares WeightMatrix::get()'s reconstruction against a REAL GPU dispatch of the
// propagate stage (via a real SpikeEngine::step_tick call) -- this exact scenario is what ticket
// #101 hit. Replacing engine.weights post-construction is safe: WeightMatrix's own hand-written
// move-assignment (weight_matrix.h) deallocates the destination's existing buffers first, and
// nothing else this engine already built (the per-neuron cell-tick kernel/dispatch plan) references
// weights' own pointers -- only the FIXED propagate stage (step_tick) reads `this->weights` live,
// every tick, not from any cached/precomputed argument.
TEST(SpikeEngineNmlPropagateDispatch,
     refit_default_matrix_reconstruction_stays_synced_with_a_real_propagate_kernel_dispatch) {
    // A small, irregular network -- some nodes with multiple out-edges, one isolated sink -- so the
    // propagate dispatch below exercises more than a single trivial edge.
    vector<vector<s32>> network = {{1, 2}, {2}, {3}, {}};
    const s64 total_neuron_count = 4;
    const s64 rank = 4;

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
    vector<IrProgram> programs = {program};

    SpikeEngine engine(model, programs); // no projections at all -- weights start edge-free

    // Replaces this engine's own (edge-free) weights with the real, irregular network above.
    engine.weights = WeightMatrix(network, rank, /*check_indexing=*/true, -1, /*weight_seed=*/4242);

    // Register a second matrix (ticket #52/D2) sharing the same U/V basis, with its own accumulated
    // Sk -- refit()'s full multi-matrix Ck/U/V sweep must run (not a single-matrix shortcut), and
    // this bug only reproduces when DEFAULT_MATRIX_INDEX itself has real Sk to fit against.
    s64 matrix_b = engine.weights.add_coefficient_vector({1.0f, -0.5f, 2.0f, 0.25f});

    for (s64 source_node = 0; source_node < total_neuron_count; ++source_node) {
        for (s32 target_node : network[(usize)source_node]) {
            engine.weights.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, (s32)source_node, target_node,
                                                  0.75f);
            engine.weights.accumulate_edge_delta(matrix_b, (s32)source_node, target_node, -1.25f);
        }
    }

    engine.weights.refit(/*sweep_count=*/4, /*ridge_regularization=*/1e-4f);

    // Host-side expected reconstruction for every real edge, post-refit -- WeightMatrix::get()
    // reconstructs via DEFAULT_MATRIX_INDEX's Ck (see reconstruct_entry, weight_matrix.cpp).
    vector<s32> real_edge_sources;
    vector<s32> real_edge_targets;
    vector<f32> expected_edge_weights;
    for (s64 source_node = 0; source_node < total_neuron_count; ++source_node) {
        for (s32 target_node : network[(usize)source_node]) {
            real_edge_sources.push_back((s32)source_node);
            real_edge_targets.push_back(target_node);
            expected_edge_weights.push_back(engine.weights.get((s32)source_node, target_node));
        }
    }

    engine.step_tick(/*dt=*/1.0f, /*tick=*/0, /*next_tick=*/1);

    vector<f32> expected_network_inputs((usize)total_neuron_count, 0.0f);
    for (usize edge_index = 0; edge_index < real_edge_sources.size(); ++edge_index) {
        expected_network_inputs[(usize)real_edge_targets[edge_index]] += expected_edge_weights[edge_index];
    }

    // ring_slot_count == 1 here (no projections at all), so this is addressed exactly like the
    // pre-fold flat buffer.
    const f32 *gpu_network_inputs = engine.network_inputs.get_contents();
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

// ── regression: next_active_neuron_count must be reset every tick, not accumulated -- ported from
// master_kernel_tests.cpp's own MasterKernel.next_active_neuron_count_is_reset_every_tick_not_
// accumulated_across_ticks. Isolates SpikeEngine::step_tick's own active-set-enqueue bookkeeping,
// asserting the counter's exact value every tick rather than relying on the out-of-bounds write a
// missing reset causes (Metal buffers are page-padded, so that corruption would not reliably crash).
TEST(SpikeEngineNmlPropagateDispatch, next_active_neuron_count_is_reset_every_tick_not_accumulated_across_ticks) {
    const f32 resting_mp = 0.0f;
    const f32 decay_rate = 0.1f;
    const f32 spike_threshold = 1.0f;
    const f32 dt = 1.0f;
    const s64 total_neuron_count = 3;

    // No edges/projections at all: this isolates the reset property from the k^2-tree scatter/dedup
    // logic entirely -- neuron 0 is the only neuron that ever receives input (the external pulse
    // below), so its own membrane trajectory is the only one that needs hand-verifying, and every
    // spike enqueues exactly itself (no downstream child).
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

    vector<IrProgram> programs = {
        build_lif_equivalent_program("LifEquivalentCell", decay_rate, resting_mp, spike_threshold)};

    SpikeEngine engine(model, programs); // no projections at all -- ring_slot_count stays 1

    const f32 external_pulse = 1.5f;

    // Neuron 0's own membrane trajectory (matching the original equivalence test's own worked
    // arithmetic): v = 1.35, 1.215, 1.0935 on ticks 1/2/3 (each > vth = 1.0, so it spikes and
    // enqueues exactly itself -- no children, since this fixture has no edges), then 0.98415,
    // 0.885735 on ticks 4/5 (sub-threshold, no spike, no enqueue).
    const vector<s32> expected_next_active_count_by_tick = {0, 1, 1, 1, 0, 0};

    for (s64 tick = 0; tick < (s64)expected_next_active_count_by_tick.size(); ++tick) {
        f32 this_tick_external_input = (tick == 1) ? external_pulse : 0.0f;
        engine.nml_allocation_.cell_state.get_contents()[0] += this_tick_external_input;
        engine.step_tick(dt, tick, tick + 1);

        // ring slot 0 always -- ring_slot_count == 1 (no projections at all in this fixture)
        s32 next_active_count_this_tick = engine.next_active_neuron_count.get_contents()[0];
        EXPECT_EQ(next_active_count_this_tick, expected_next_active_count_by_tick[(usize)tick]) << "tick=" << tick;
        // Never exceeds the buffer's own allocated element count regardless of how many neurons
        // spiked this tick -- the property a missing reset violates.
        ASSERT_LE(next_active_count_this_tick, (s32)total_neuron_count) << "tick=" << tick;
    }
}

// ── regression: a genuine constant weight of exactly 0 must not fall back to the U/V basis -- ported
// from master_kernel_tests.cpp's own MasterKernel.a_genuine_zero_constant_weight_propagates_exactly_
// zero_not_the_uv_basis. `using_constant_weight` must be threaded through as its own explicit flag
// rather than overloaded onto `constant_weight == 0.0f`. Note that set_constant_weight(0.0f) alone
// would NOT discriminate between the two implementations here: it deliberately fills U/V so their
// own dot product already reconstructs the same constant_weight value -- so a sentinel-based
// `constant_weight == 0.0f` check would coincidentally still compute 0 in that exact state. This
// test instead leaves U/V at their construction-time random values (a fixed weight_seed for
// reproducibility) and sets `constant_weight`/`using_constant_weight` directly (both are public
// fields, unlike the set_constant_weight() setter, which would re-sync U/V to match) -- so U.V for
// edge (0,1) is a real nonzero value while constant_weight is genuinely 0, and the propagated weight
// must come out exactly 0.0f (from constant_weight), not that nonzero U.V dot product. Preserves the
// original test's own anti-placeholder-test technique exactly, per this task's own migration notes.
TEST(SpikeEngineNmlPropagateDispatch, a_genuine_zero_constant_weight_propagates_exactly_zero_not_the_uv_basis) {
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

    vector<IrProgram> programs = {
        build_lif_equivalent_program("LifEquivalentCell", decay_rate, resting_mp, spike_threshold)};

    SpikeEngine engine(model, programs); // no projections at all -- weights start edge-free

    // A fixed weight_seed leaves U/V at deterministic, real (non-zero) random-normal values --
    // deliberately NOT calling set_constant_weight(), so U/V for edge (0,1) do not happen to
    // reconstruct 0. Replaces this engine's own (edge-free) weights with this real adjacency (see
    // refit_default_matrix_reconstruction_stays_synced_with_a_real_propagate_kernel_dispatch above
    // for why replacing engine.weights post-construction is safe).
    engine.weights = WeightMatrix(adjacency, /*rank=*/1, /*check_indexing=*/true, /*max_neighbor_count=*/-1,
                                  /*weight_seed=*/42);
    ASSERT_NE(engine.weights.get(0, 1), 0.0f) << "U/V for edge (0,1) coincidentally reconstruct 0 -- this "
                                                  "test's own premise depends on a genuinely nonzero U.V "
                                                  "dot product to disagree with constant_weight=0";

    // Both fields are public (unlike set_constant_weight(), which would also re-sync U/V to match
    // this value) -- so U/V stay at the real, nonzero values checked above while constant_weight is
    // genuinely 0.
    engine.weights.constant_weight = 0.0f;
    engine.weights.using_constant_weight = true;

    const f32 external_pulse = 1.5f; // raises neuron 0 above threshold on tick 1's own integrate step

    // Tick 0: zero-input priming tick, no spike. Tick 1: external pulse drives neuron 0 to v=1.35 >
    // vth=1.0, so it spikes and propagate scatters its (genuinely zero) weight into network_inputs[1].
    engine.step_tick(dt, /*tick=*/0, /*next_tick=*/1);
    engine.nml_allocation_.cell_state.get_contents()[0] += external_pulse;
    engine.step_tick(dt, /*tick=*/1, /*next_tick=*/2);

    EXPECT_TRUE(engine.last_spiked.get_contents()[0] == 1) << "neuron 0 did not spike on tick 1 as expected";
    EXPECT_EQ(engine.network_inputs.get_contents()[1], 0.0f); // ring_slot_count == 1 -- slot 0 always
}

// ── ticket #62 [F1]: active-set x nonlinear-dynamics correctness rule -- ported from master_kernel_
// tests.cpp's own MasterKernelActiveSetNonlinearRule.nonlinear_population_never_receives_a_closed_
// form_multi_tick_jump. SpikeEngine::step_tick dispatches every population's kernel over its own FULL
// neuron range every tick regardless of any closed-form-advanceable classification -- so "never
// fast-forwarded" is testable by comparing the NML-mode engine's own trajectory against a plain,
// tick-by-tick C++ transcription of the exact same `.tick.integrate` instructions run once per tick
// with no batching. This is the master-kernel side of this ticket's acceptance criterion; the
// SEPARATE "a linear cell's skipped neuron still fast-forwards via closed-form apply_decay" side is
// the hardcoded engine's own, pre-existing, untouched mechanism -- verified in engine_tests.cpp's own
// `active_set_skip_then_revisit_matches_per_tick_decay_equivalence`, not here.

namespace {

// A synthetic, TEST-ONLY nonlinear cell program -- mirrors master_kernel_tests.cpp's own build_
// nonlinear_test_only_program (kept as its own copy here, matching this codebase's established
// per-file fixture convention): the same shape build_lif_equivalent_program uses, extended with one
// `v * v` self-product term in its own `@integrate` -- the same shape a real nonlinear cell's
// quadratic term would take (see cell_lowering_tests.cpp's own NONLINEAR_TEST_ONLY_COMPONENT_TYPE).
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

TEST(SpikeEngineNmlActiveSetNonlinearRule, nonlinear_population_never_receives_a_closed_form_multi_tick_jump) {
    const f32 gL = 0.1f;
    const f32 EL = 0.0f;
    const f32 vth = 1000.0f; // never spikes -- isolates the pure integrate step from reset/regime noise
    const f32 dt = 1.0f;
    const s64 total_neuron_count = 2; // neuron 0 is this test's own subject; neuron 1 is an unused peer

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

    vector<IrProgram> programs = {build_nonlinear_test_only_program("NonlinearTestOnlyCell", gL, EL, vth)};

    SpikeEngine engine(model, programs); // no edges/projections at all -- neuron 0 never receives
                                          // network_inputs from anywhere but its own external pulse

    // A one-time pulse on tick 1 raises v off resting; every tick after that (tick 2 onward) neuron 0
    // receives NEITHER external stimulus NOR any network_inputs -- exactly the "some neurons going
    // quiet (inactive) for multiple ticks" scenario this ticket's acceptance criterion describes. A
    // closed-form multi-tick jump would show up as a DIVERGENCE from the plain per-tick reference
    // computed below the moment more than one quiet tick has elapsed.
    const f32 external_pulse = 0.3f;
    const s64 tick_count = 6;

    f32 reference_v = EL;
    for (s64 tick = 0; tick < tick_count; ++tick) {
        f32 this_tick_external_input = (tick == 1) ? external_pulse : 0.0f;

        engine.nml_allocation_.cell_state.get_contents()[0] += this_tick_external_input;
        engine.step_tick(dt, tick, tick + 1);

        // Plain, unbatched, exactly-one-dt-per-tick transcription of build_nonlinear_test_only_
        // program's own `@integrate` instructions -- network_inputs is always 0.0f here (no edges
        // ever scatter into it).
        reference_v += this_tick_external_input;
        f32 t0 = EL - reference_v;
        t0 = gL * t0;
        t0 = 0.0f + t0;
        f32 t1 = reference_v * reference_v;
        t0 = t0 - t1;
        t0 = t0 / 1.0f; // C
        t0 = t0 * dt;
        reference_v = reference_v + t0;

        EXPECT_NEAR(engine.nml_allocation_.cell_state.get_contents()[0], reference_v, 1e-4f) << "tick=" << tick;
    }

    // Sanity: the reference trajectory actually moved (a comparison that never left EL would be
    // vacuous), and neuron 0 never spiked (vth is unreachable here, keeping this test purely about
    // @integrate).
    EXPECT_NE(reference_v, EL);
    // SpikeEngine's own NML-mode "never fired" seed convention is -1 (engine.cpp's ModelSpecification
    // constructor), NOT AssembledModel/ModelRuntimeBuffers' own 0 -- see engine.h's own doc comment
    // on apply_nml_stdp_plasticity for this exact, deliberate distinction.
    EXPECT_EQ(engine.last_spiked.get_contents()[0], -1);
}

// ── Ported from tests/assembled_model_plasticity_tests.cpp's own AssembledModelPlasticity suite
// (folding nml::AssembledModel into SpikeEngine). AssembledModelPlasticity.enable_plasticity_throws_
// when_real_per_edge_synapse_dispatch_is_active is NOT re-ported here: it is already covered
// verbatim by SpikeEngineNmlPlasticity.enable_plasticity_throws_when_real_per_edge_synapse_dispatch_
// is_active above (same conductance-synapse fixture, same assertion).

namespace {

// A single-population, no-projections model just big enough to construct a valid SpikeEngine --
// used by the tests below that only exercise the plasticity API surface itself, not a real network.
// Mirrors assembled_model_plasticity_tests.cpp's own build_minimal_one_neuron_model (kept as its own
// copy here, matching this codebase's established per-file fixture convention).
ModelSpecification build_minimal_one_neuron_model(vector<IrProgram> &out_programs) {
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

TEST(SpikeEngineNmlPlasticity, plasticity_enabled_reflects_enable_and_disable_calls) {
    vector<IrProgram> programs;
    ModelSpecification model = build_minimal_one_neuron_model(programs);
    SpikeEngine engine(model, programs);

    EXPECT_FALSE(engine.plasticity_enabled());

    engine.enable_plasticity(0.05f);
    EXPECT_TRUE(engine.plasticity_enabled());

    engine.disable_plasticity();
    EXPECT_FALSE(engine.plasticity_enabled());

    // both a second disable and a redundant enable-while-enabled must not throw (no-op, matching
    // SpikeEngine::enable_plasticity/disable_plasticity's own established semantics, SC-11)
    EXPECT_NO_THROW(engine.disable_plasticity());
    EXPECT_NO_THROW(engine.enable_plasticity(0.05f));
    EXPECT_NO_THROW(engine.enable_plasticity(0.9f));
    EXPECT_TRUE(engine.plasticity_enabled());
}

// The SAME projections as SpikeEngineNmlSynapseDispatch's own conductance-synapse fixture above, but
// with a per-connection delay that converts to > 1 tick: ticket #131's own established precedent is
// that ring mode (nml_ring_slot_count_ > 1) forces real per-edge synapse dispatch off entirely
// regardless of model.projections (this engine's own nml_projections_ is forced empty at construction
// time -- see engine.cpp's own constructor doc comment) -- so enable_plasticity must NOT throw here,
// since there is no second writer of the shared U/V basis to coordinate with in this mode. Mirrors
// AssembledModelPlasticity.enable_plasticity_does_not_throw_under_delay_ring_mode_even_with_
// projections_present (tests/assembled_model_plasticity_tests.cpp).
TEST(SpikeEngineNmlPlasticity, enable_plasticity_does_not_throw_under_delay_ring_mode_even_with_projections_present) {
    const f32 resting_mp = 0.0f;
    const f32 decay_rate = 0.1f;
    const f32 spike_threshold = 1.0f;
    const f32 gbase = 1.0f, erev = 2.0f, tau_decay = 2.0f;
    const f64 delay_ticks = 5.0; // > 1 tick -- forces nml_ring_slot_count_ > 1

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
    projection.connections.push_back(
        ConnectionEntry{/*source=*/0, /*target=*/1, /*weight=*/1.0, /*delay=*/delay_ticks});
    model.projections.push_back(projection);

    vector<IrProgram> programs = {
        build_lif_equivalent_program("PostCell", decay_rate, resting_mp, spike_threshold),
        build_conductance_synapse_program("CondSynapse", gbase, erev, tau_decay),
    };

    // dt_seconds=1.0f makes delay_ticks read back as an identity whole-tick count (matches this
    // file's own established build_two_neuron_delay_ring_model convention below).
    SpikeEngine engine(model, programs, /*dt_seconds=*/1.0f); // real, non-empty model.projections, but
                                                               // ring_slot_count > 1 forces this
                                                               // engine's own nml_projections_ empty
    ASSERT_TRUE(engine.weights.using_constant_delay_ticks);
    ASSERT_GT(engine.weights.constant_delay_ticks, 1); // confirms ring mode really is active

    EXPECT_NO_THROW(engine.enable_plasticity(0.02f));
    EXPECT_TRUE(engine.plasticity_enabled());
}

TEST(SpikeEngineNmlPlasticity, apply_stdp_wiring_enables_plasticity_from_a_real_type_library_scan) {
    vector<IrProgram> programs;
    ModelSpecification model = build_minimal_one_neuron_model(programs);

    // SpikeEngine's own ModelSpecification constructor requires type_library_ir_programs.size() ==
    // model.type_library.size(), so the engine is built BEFORE the extra Synapse-category entry
    // below is appended -- apply_stdp_wiring itself only reads model.type_library at CALL time, so
    // mutating model afterward is exactly what a real caller (scan the resolved model, separately,
    // after the SpikeEngine it drives is already built) would do too.
    SpikeEngine engine(model, programs);
    ASSERT_FALSE(engine.plasticity_enabled());

    TypeLibraryEntry stdp_synapse_entry;
    stdp_synapse_entry.component_type_name = "TestStdpSynapse";
    stdp_synapse_entry.bound_instance_id = "synInstance";
    stdp_synapse_entry.category = TypeLibraryCategory::Synapse;
    stdp_synapse_entry.baked_constants = {
        {"tauPlus", 0.02}, {"tauMinus", 0.02}, {"aPlus", 0.01}, {"aMinus", 0.03}};
    // Scanned by apply_stdp_wiring (which only reads model.type_library) -- deliberately never
    // referenced by any projection/population, matching this test's own narrow scope.
    model.type_library.push_back(stdp_synapse_entry);

    // The EXISTING apply_stdp_wiring(const ModelSpecification&, SpikeEngine&) overload (ticket #66,
    // predating ticket #132's AssembledModel&-overload) -- SpikeEngine::enable_plasticity/disable_
    // plasticity are the SAME unified API for both hardcoded-LIF and NML modes now, so this needed
    // zero new code (confirmed here, not just asserted).
    apply_stdp_wiring(model, engine);

    EXPECT_TRUE(engine.plasticity_enabled());
}

TEST(SpikeEngineNmlPlasticity, apply_stdp_wiring_disables_plasticity_when_no_stdp_synapse_present) {
    vector<IrProgram> programs;
    ModelSpecification model = build_minimal_one_neuron_model(programs);

    SpikeEngine engine(model, programs);
    engine.enable_plasticity(0.1f); // simulate an instance constructed with plasticity already live
    ASSERT_TRUE(engine.plasticity_enabled());

    apply_stdp_wiring(model, engine);

    EXPECT_FALSE(engine.plasticity_enabled());
}

// ── Ported from tests/assembled_model_plasticity_tests.cpp's own AssembledModelPlasticity suite:
// ticket #132's own acceptance criterion -- real GLIF cell dynamics + a non-trivial per-edge delay
// (the delay ring) + active STDP, all under one SpikeEngine-based tick loop. Reuses build_two_
// neuron_delay_ring_model (SpikeEngineNmlDelayRing's own fixture, defined above in this same file)
// rather than redeclaring a third copy of the same 2-neuron "fires exactly once" cell fixture --
// placed here, after that helper's own definition, even though it belongs to the
// SpikeEngineNmlPlasticity suite (gtest suite membership is by name, not by file position).
TEST(SpikeEngineNmlPlasticity,
     stdp_measurably_depresses_the_weight_over_a_real_glif_run_with_a_non_trivial_delay_ring) {
    const s64 delay_ticks = 5;
    const f32 delivered_constant_weight = 0.1f; // well below this fixture's own vth=2.0 -- never
                                                 // causes a re-spike
    const f32 external_pulse = 5.0f;
    const s64 postsynaptic_pulse_tick = 2; // neuron 1 fires first
    const s64 presynaptic_pulse_tick = 3;  // neuron 0 fires second -- the only ordering apply_nml_
                                            // stdp_plasticity's own depression-only update ever
                                            // nudges the weight for (see engine.cpp)
    const s64 tick_count = 12;
    const f32 stdp_learning_rate = 0.02f;

    vector<IrProgram> programs;
    ModelSpecification model = build_two_neuron_delay_ring_model((f64)delay_ticks, programs);

    SpikeEngine engine(model, programs, /*dt_seconds=*/1.0f);
    engine.weights.set_constant_weight(delivered_constant_weight);
    engine.enable_plasticity(stdp_learning_rate);
    ASSERT_TRUE(engine.plasticity_enabled());

    const f32 weight_before = engine.weights.get(0, 1);

    vector<f32> network_input_at_target_by_tick((usize)tick_count, 0.0f);
    for (s64 tick = 0; tick < tick_count; ++tick) {
        s64 current_ring_slot = tick % delay_ticks; // ring_slot_count == delay_ticks in the uniform-
                                                     // delay case, confirmed against engine.weights.
                                                     // constant_delay_ticks below
        network_input_at_target_by_tick[(usize)tick] =
            engine.network_inputs.get_contents()[current_ring_slot * engine.neuron_count + 1];

        if (tick == postsynaptic_pulse_tick) engine.nml_allocation_.cell_state.get_contents()[1] += external_pulse;
        if (tick == presynaptic_pulse_tick) engine.nml_allocation_.cell_state.get_contents()[0] += external_pulse;

        engine.step_tick(1.0f, tick, tick + 1);
    }

    const f32 weight_after = engine.weights.get(0, 1);

    ASSERT_TRUE(engine.weights.using_constant_delay_ticks);
    ASSERT_EQ(engine.weights.constant_delay_ticks, delay_ticks);

    // Sanity: this is real GLIF cell dynamics, not hand-set spike times.
    EXPECT_EQ(engine.last_spiked.get_contents()[1], postsynaptic_pulse_tick);
    EXPECT_EQ(engine.last_spiked.get_contents()[0], presynaptic_pulse_tick);

    // The non-trivial per-edge delay really is exercised: the presynaptic spike's contribution lands
    // at exactly presynaptic_pulse_tick + delay_ticks, and nowhere else.
    for (s64 tick = 0; tick < tick_count; ++tick) {
        f32 expected = (tick == presynaptic_pulse_tick + delay_ticks) ? delivered_constant_weight : 0.0f;
        EXPECT_NEAR(network_input_at_target_by_tick[(usize)tick], expected, 1e-6f) << "tick=" << tick;
    }

    // ── this ticket's own acceptance criterion: STDP measurably moves the weight, in the only
    // direction the real kernel's own pow(tick_delta, -3) shape ever nudges it (depression).
    EXPECT_LT(weight_after, weight_before);
}

// ── Retired from tests/delay_ring_tests.cpp (its own DelayRing suite tested compute_max_delay_ticks/
// allocate_delay_ring, pure host-side functions with NO AssembledModel/GPU dispatch at all). Spike
// Engine's own delay-ring fold (engine.cpp's nml_delay_seconds_to_ticks/nml_compute_ring_slot_count_
// from_weight_matrix) duplicates the exact same "floor to 1 tick absent any real delay, scale to the
// longest delay actually present" logic, but reads delay straight off WeightMatrix's own constant_
// delay_ticks/edge_delay_ticks fields instead of a separate DelayRingAllocation, and is file-local
// (anonymous namespace in engine.cpp) -- not unit-testable in isolation the way delay_ring.cpp's own
// free functions are, so these are SpikeEngine integration tests instead. Per-case disposition
// (delay_ring_tests.cpp's own 9 DelayRing tests):
//   - compute_max_delay_ticks_is_one_with_no_connections_at_all -> zero_projections_default_to_the_
//     flat_one_tick_latency_delay below.
//   - compute_max_delay_ticks_floors_a_zero_or_absent_delay_to_one_tick -> already covered by
//     ring_slot_count_one_no_configured_delay_matches_the_flat_one_tick_latency above.
//   - compute_max_delay_ticks_returns_the_longest_delay_across_every_connection_in_whole_ticks /
//     ring_slot_count_scales_with_the_longest_delay_actually_present -> per_edge_non_uniform_delays_
//     each_deliver_at_their_own_tick_and_the_ring_sizes_to_the_longest below.
//   - allocate_delay_ring_sizes_ring_slot_count_to_max_delay_ticks_plus_one -> the OLD DelayRing
//     Allocation convention pads ring_slot_count to max_delay_ticks + 1; SpikeEngine's own fold
//     deliberately does NOT add that "+1" (nml_compute_ring_slot_count_from_weight_matrix, engine.cpp,
//     returns the max delay itself) -- a genuine, confirmed-safe simplification, not a gap: a ring
//     slot is addressed by `arrival_tick % ring_slot_count` against the ABSOLUTE tick number (not a
//     "slots ahead of now" scheme), so two deliveries can only ever collide on the same physical slot
//     if their absolute arrival ticks differ by a multiple of ring_slot_count -- i.e. at least ring_
//     slot_count ticks apart, exactly how often that slot is drained+read, so no extra safety-margin
//     slot is needed. ring_shaped_path_delivers_a_spike_exactly_delay_ticks_ticks_later above and
//     per_edge_non_uniform_delays_... below both already prove real, correctly-timed delivery under
//     this "no +1" sizing.
//   - allocate_delay_ring_defaults_every_real_edge_to_one_tick_absent_an_explicit_delay -> already
//     covered by ring_slot_count_one_no_configured_delay_matches_the_flat_one_tick_latency above.
//   - allocate_delay_ring_stores_the_real_per_edge_delay_in_whole_ticks -> already covered by
//     ring_shaped_path_delivers_a_spike_exactly_delay_ticks_ticks_later and real_nml_connection_
//     delay_converts_to_the_correct_whole_tick_count_and_delivers_on_time above (the latter via a
//     REAL NeuroML delay attribute + real unit conversion -- strictly stronger coverage).
//   - allocate_delay_ring_skips_a_connection_that_is_not_a_representable_k2tree_neighbor -> DOES NOT
//     APPLY to SpikeEngine's design (confirmed by reading engine.cpp/weight_matrix.cpp): this
//     engine's own `weights` is always built directly (and exclusively) from model.projections via
//     nml::build_weight_matrix_from_projections, with an auto-derived max_neighbor_count (WeightMatrix's
//     own -1 default, sized to the longest real adjacency row) -- so every real ConnectionEntry is
//     GUARANTEED representable in weights' own k^2-tree by construction; the mismatched-adjacency
//     scenario delay_ring_tests.cpp's own test defends against (an externally-supplied WeightMatrix
//     built from a DIFFERENT adjacency than the model's own connections) cannot arise here at all.
//     Separately: WeightMatrix::set_edge_delay_ticks (what this engine's own per-edge delay-seeding
//     loop calls) actually THROWS on a non-representable edge rather than silently skipping it -- the
//     opposite of delay_ring.cpp's own allocate_delay_ring -- moot given the scenario is unreachable.
//   - allocate_delay_ring_zero_initializes_the_input_ring_and_pending_active_bookkeeping ->
//     ring_shaped_buffers_start_fully_zero_initialized_before_any_tick below.

TEST(SpikeEngineNmlDelayRing, zero_projections_default_to_the_flat_one_tick_latency_delay) {
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

    vector<IrProgram> programs = {build_lif_equivalent_program("Cell", 0.1f, 0.0f, 1.0f)};
    ASSERT_TRUE(model.projections.empty());

    SpikeEngine engine(model, programs);

    // No connections at all means this engine's own constructor never touches weights' delay config
    // (engine.cpp's own delay-seeding loop only runs over model.projections' connections), so it
    // stays at WeightMatrix's own "flat one-tick latency" default (weight_matrix.h).
    EXPECT_TRUE(engine.weights.using_constant_delay_ticks);
    EXPECT_EQ(engine.weights.constant_delay_ticks, 1);
}

namespace {

// A 3-neuron model: neuron 0 and neuron 1 each drive neuron 2 through their own plain (no real
// per-edge synapse type) connection, carrying DIFFERENT delays -- proving per-edge (non-uniform)
// delay resolution, and that the ring sizes to the LONGER of the two, not just whichever was seen
// first. Reuses build_lif_equivalent_program/build_lif_equivalent_type_entry (defined above in this
// same file) -- the same gL=0.5/EL=0.0/vth=2.0 "fires exactly once from a single external boost"
// fixture shape build_two_neuron_delay_ring_model's own doc comment already establishes.
ModelSpecification build_three_neuron_mixed_delay_ring_model(f64 short_delay_ticks, f64 long_delay_ticks,
                                                               vector<IrProgram> &out_programs) {
    const f32 decay_rate = 0.5f;      // gL
    const f32 resting_mp = 0.0f;      // EL
    const f32 spike_threshold = 2.0f; // vth

    ModelSpecification model;
    model.total_neuron_count = 3;
    model.type_library.push_back(build_lif_equivalent_type_entry(
        "MixedDelayRingCell", "mixedDelayRingInstance", 1.0f, decay_rate, resting_mp, spike_threshold));

    PopulationEntry population;
    population.id = "Pop";
    population.type_library_index = 0;
    population.size = 3;
    population.neuron_index_begin = 0;
    population.neuron_index_end = 3;
    model.populations.push_back(population);

    ProjectionEntry projection;
    projection.id = "Proj";
    projection.presynaptic_population_index = 0;
    projection.postsynaptic_population_index = 0;
    projection.connections.push_back(
        ConnectionEntry{/*source=*/0, /*target=*/2, /*weight=*/1.0, /*delay=*/short_delay_ticks});
    projection.connections.push_back(
        ConnectionEntry{/*source=*/1, /*target=*/2, /*weight=*/1.0, /*delay=*/long_delay_ticks});
    model.projections.push_back(projection);

    out_programs = {build_lif_equivalent_program("MixedDelayRingCell", decay_rate, resting_mp, spike_threshold)};
    return model;
}

} // namespace

TEST(SpikeEngineNmlDelayRing,
     per_edge_non_uniform_delays_each_deliver_at_their_own_tick_and_the_ring_sizes_to_the_longest) {
    const s64 short_delay_ticks = 3;
    const s64 long_delay_ticks = 5;
    const s64 short_source_pulse_tick = 1; // neuron 0 -> neuron 2, delay=3, fires at tick 1
    const s64 long_source_pulse_tick = 2;  // neuron 1 -> neuron 2, delay=5, fires at tick 2
    const s64 short_expected_arrival_tick = short_source_pulse_tick + short_delay_ticks; // 4
    const s64 long_expected_arrival_tick = long_source_pulse_tick + long_delay_ticks;    // 7
    const s64 tick_count = 10;
    const f32 constant_weight = 3.0f;
    const f32 external_pulse = 5.0f;

    vector<IrProgram> programs;
    ModelSpecification model =
        build_three_neuron_mixed_delay_ring_model((f64)short_delay_ticks, (f64)long_delay_ticks, programs);

    SpikeEngine engine(model, programs, /*dt_seconds=*/1.0f);
    engine.weights.set_constant_weight(constant_weight);

    // The two connections genuinely disagree on delay, so this engine's own constructor takes the
    // per-edge (not uniform) path -- weights.using_constant_delay_ticks is false, and each edge's own
    // slot in weights.edge_delay_ticks carries its own real value; this engine's own private ring
    // size (nml_ring_slot_count_, engine.h) is the LONGER of the two (5), confirmed indirectly below
    // by both deliveries landing on their own exact expected tick.
    ASSERT_FALSE(engine.weights.using_constant_delay_ticks);

    vector<f32> network_input_at_target_by_tick((usize)tick_count, 0.0f);
    for (s64 tick = 0; tick < tick_count; ++tick) {
        s64 current_ring_slot = tick % long_delay_ticks; // ring_slot_count == the longer delay (5)
        network_input_at_target_by_tick[(usize)tick] =
            engine.network_inputs.get_contents()[current_ring_slot * engine.neuron_count + 2];

        if (tick == short_source_pulse_tick) engine.nml_allocation_.cell_state.get_contents()[0] += external_pulse;
        if (tick == long_source_pulse_tick) engine.nml_allocation_.cell_state.get_contents()[1] += external_pulse;

        engine.step_tick(1.0f, tick, tick + 1);
    }

    for (s64 tick = 0; tick < tick_count; ++tick) {
        f32 expected = 0.0f;
        if (tick == short_expected_arrival_tick || tick == long_expected_arrival_tick) expected = constant_weight;
        EXPECT_NEAR(network_input_at_target_by_tick[(usize)tick], expected, 1e-6f)
            << "tick=" << tick << " (short delay=" << short_delay_ticks << " arrives at "
            << short_expected_arrival_tick << ", long delay=" << long_delay_ticks << " arrives at "
            << long_expected_arrival_tick << ")";
    }

    EXPECT_EQ(engine.last_spiked.get_contents()[0], short_source_pulse_tick);
    EXPECT_EQ(engine.last_spiked.get_contents()[1], long_source_pulse_tick);
}

TEST(SpikeEngineNmlDelayRing, ring_shaped_buffers_start_fully_zero_initialized_before_any_tick) {
    const s64 delay_ticks = 5;
    vector<IrProgram> programs;
    ModelSpecification model = build_two_neuron_delay_ring_model((f64)delay_ticks, programs);

    // dt_seconds=1.0f matches this fixture's own dt=1.0 convention (see build_two_neuron_delay_ring_
    // model's own doc comment above), making the delay's seconds->ticks conversion an identity.
    SpikeEngine engine(model, programs, 1.0f);

    ASSERT_TRUE(engine.weights.using_constant_delay_ticks);
    ASSERT_EQ(engine.weights.constant_delay_ticks, delay_ticks);
    const s64 ring_slot_count = engine.weights.constant_delay_ticks;

    // SpikeEngine's own ring-shaped network_inputs/active_generation/next_active_neuron_count
    // (engine.h's own doc comment on nml_ring_slot_count_) are the direct generalization of
    // DelayRingAllocation's own input_ring/pending_active_generation/pending_active_neuron_count --
    // every slot must start zero (network_inputs)/-1 (active_generation, "never enqueued")/0 (next_
    // active_neuron_count) BEFORE any step_tick call, not just eventually.
    const f32 *network_inputs = engine.network_inputs.get_contents();
    for (s64 index = 0; index < ring_slot_count * engine.neuron_count; ++index) {
        EXPECT_EQ(network_inputs[index], 0.0f) << "index=" << index;
    }

    const s32 *active_generation = engine.active_generation.get_contents();
    for (s64 index = 0; index < ring_slot_count * engine.neuron_count; ++index) {
        EXPECT_EQ(active_generation[index], -1) << "index=" << index;
    }

    const s32 *next_active_neuron_count = engine.next_active_neuron_count.get_contents();
    for (s64 slot = 0; slot < ring_slot_count; ++slot) {
        EXPECT_EQ(next_active_neuron_count[slot], 0) << "slot=" << slot;
    }
}
