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
