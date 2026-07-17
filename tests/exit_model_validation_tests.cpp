#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/nml/allocator.h"
#include "spikecorec/nml/cell_lowering.h"
#include "spikecorec/nml/master_kernel.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/stimulus_schedule.h"
#include "spikecorec/nml/synapse_lowering.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── Phase-1 validation / exit models (ticket #61 [H1]; arch §5) ──────────────
//
// The two arch §5 Phase-1 exit models: a GLIF3 single cell reproducing
// after-spike-current dynamics and spike-frequency adaptation under a current
// step (tests/fixtures/nml/glif3_single_cell.nml), and a current- and
// conductance-based GLIF E/I network including an NMDA synapse producing a
// spike raster (tests/fixtures/nml/glif_ei_network.nml). Both fixtures are
// checked-in, standalone NeuroML/LEMS files (not the usual inline C++ string
// literal every other *_tests.cpp file in this tree uses) specifically so the
// SAME file drives both pyneuroml/jNeuroML (the real reference simulator) and
// spikecorec's own pipeline -- see each fixture's own header comment for the
// exact ComponentTypes/topology and why (reused verbatim from
// cell_lowering_tests.cpp's GLIF3 fixture, synapse_lowering_tests.cpp's
// TestNmdaSynapse fixture, and allocator_tests.cpp's GLIF E/I network
// fixture).
//
// Per the ticket's scope clarification (task_master, 2026-07-16): the
// captured reference data under tests/fixtures/reference_data/ is REAL
// output from running both fixtures through pyneuroml/jNeuroML (commands
// documented in each fixture's own LEMS_*.xml header comment) -- not an
// analytic stand-in. Everything in this file EXCEPT the two
// DISABLED_-prefixed numeric-comparison tests at the bottom runs and passes
// normally; the comparison tests themselves are intentionally disabled (do
// not need to pass in this ticket's PR) per that same scope clarification --
// the user will manually enable them later and verify against this same
// checked-in reference data.
//
// ── Known limitations a reviewer should look at closely ──────────────────
// 1. External stimulus is injected directly into `network_inputs` (read by a
//    GLIF cell's own `.tick`'s `@integrate` stage, arch §3.5/IR spec §3.5),
//    NOT into `membrane_potentials`/cell_state directly the way arch §0.2
//    documents for the ORIGINAL hardcoded (pre-NML) LIF engine and its own
//    ticket #58 stimulus-schedule consumer (master_kernel_tests.cpp). A real
//    SI-unit GLIF cell's injected current (~100s of pA) is many orders of
//    magnitude too small to move its voltage (~10s of mV) by direct addition
//    with no dt/C scaling -- ticket #58's own stimulus_schedule_tests.cpp
//    already anticipates this exact ticket resolving that ("ticket #61's
//    future validation suite exercises this same shape against the live
//    engine once the master kernel is wired") and gives the worked
//    reference integrator (`run_glif1_forward_euler`) this file's own
//    stimulus injection mirrors: the injected current is a `network_inputs`
//    contribution, forward-Euler-integrated by the SAME `.tick` code a real
//    synapse's contribution would be (arch §0.2's own "external stimulus ->
//    membrane_potentials" rule describes the ORIGINAL hardcoded engine's own
//    mechanism specifically, not a constraint on how this validation harness
//    drives a real NML cell's `network_inputs` read).
// 2. The GLIF E/I network's `<inputList>`/`<input>` stimulus (needed --
//    see glif_ei_network.nml's own header comment -- because jLEMS cannot
//    resolve `explicitInput`'s indexed target against a `populationList`
//    population) is NOT a tag spikecorec's own front-end recognizes today
//    (only `explicitInput` is, model_specification.cpp's own
//    `collect_by_tag(..., {"explicitInput"}, ...)`) -- so `model.stimuli` is
//    empty for this fixture, and this file's own network driver
//    reconstructs the identical stimulus window by hand from the .nml's own
//    literal `pulseGenerator` attributes instead of via
//    `build_stimulus_schedule`. Adding `inputList`/`input` parsing is a
//    front-end (ticket #2/#49-territory) feature, out of this validation
//    ticket's own scope.
// 3. AssembledModel's fixed propagate stage (ticket #6) scatters spikes via
//    the k^2-tree/WeightMatrix path (arch's "one shared U/V basis"), NOT by
//    invoking a projection's actual SYNAPSE ComponentType's own per-edge
//    `.tick`/`.deliver_<port>` dynamics (master_kernel.h's own documented
//    scope boundary: routing a spike through a real synapse ComponentType
//    needs a "spike-scatter batch construction" subsystem gpu_source.h's own
//    header comment already flags as "not yet built by ANY prior ticket" --
//    no ticket in CLAUDE.md's own epic ticket graph currently owns building
//    it). This file's own network driver forces the WeightMatrix to a
//    constant weight of EXACTLY ZERO (`set_constant_weight(0.0f)`) rather
//    than reconstructing an arbitrary random low-rank weight for edges whose
//    real conductance/current values are never actually read -- so only the
//    directly-stimulated neuron (ExcPop[0]) is expected to spike in THIS
//    file's own network sanity test; the real jLEMS reference (which DOES
//    run the true expOneSynapse/alphaCurrentSynapse/NMDA dynamics) shows
//    downstream propagation to ExcPop[2]/InhPop[0] that spikecorec's own
//    AssembledModel cannot yet reproduce until that subsystem is built --
//    flagged here, not silently papered over. This file's own recording
//    also bypasses ticket #59's output_recording.h for this one fixture
//    (reads ModelAllocation::cell_state directly instead, the same pattern
//    ticket #6's own master_kernel_tests.cpp uses) since jLEMS's own
//    `populationList`-addressed recording quantities
//    (`population/index/componentId/exposureName`) don't round-trip through
//    spikecorec's own `parse_population_path` the way a plain population's
//    `population/index/exposureName` form does (see glif_ei_network.nml's
//    own header comment) -- the GLIF3 single-cell fixture's plain population
//    has no such issue and isn't affected.

namespace {

String fixture_path(const String &relative_path) {
    return String(SPIKECOREC_TEST_FIXTURES_DIR) + "/" + relative_path;
}

// `fixture_base_name` names the fixture without an extension (e.g.
// "glif3_single_cell") -- loads "<fixture_base_name>_top.nml", a thin
// <include>-only wrapper around the real "<fixture_base_name>.nml" (the same
// file jNeuroML/jLEMS consumes directly). NML_Parser::parse only
// XSD-validates the TOP-LEVEL file it's given (ingest_file's own header
// comment), so parsing through the wrapper -- the same "top.nml includes
// content.nml" convention every other *_tests.cpp fixture in this tree
// already uses -- avoids re-validating these two real, standalone NeuroML
// files as if they were themselves top-level XSD-validated documents.
ModelSpecification load_model_from_nml_fixture(const String &fixture_base_name) {
    NML_Parser parser;
    parser.parse(fixture_path("nml/" + fixture_base_name + "_top.nml"));
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

// Parallel to model.type_library, one IrProgram per entry -- same convention
// allocator_tests.cpp's own build_type_library_ir_programs uses (an Inputs
// entry would get an empty placeholder, but neither fixture in this file
// produces one -- see this file's own header comment on why `inputList`
// leaves no TypeLibraryEntry behind at all).
Vector<IrProgram> build_type_library_ir_programs(const ModelSpecification &model) {
    Vector<IrProgram> programs;
    programs.reserve(model.type_library.size());
    for (const auto &entry : model.type_library) {
        if (entry.category == TypeLibraryCategory::Cell) {
            programs.push_back(lower_cell_to_ir(entry));
        } else if (entry.category == TypeLibraryCategory::Synapse) {
            programs.push_back(lower_synapse_to_ir(entry));
        } else {
            IrProgram placeholder;
            placeholder.component_type_name = entry.component_type_name;
            programs.push_back(std::move(placeholder));
        }
    }
    return programs;
}

// Builds the exact-edge-set WeightMatrix AssembledModel's propagate stage
// needs (arch §0.3) from every projection's connections -- the same
// vector<vector<s32>> construction model_specification.cpp's own adjacency
// building performs internally for model.adjacency, but built directly here
// so a caller has one even for a model with NO connections at all (a
// single-cell model's model.adjacency stays nullopt -- "unset if the model
// has no connections", model_specification.h's own doc comment -- but
// AssembledModel::step_tick's ModelRuntimeBuffers::weights is a required,
// non-null pointer regardless).
WeightMatrix build_weight_matrix(const ModelSpecification &model) {
    vector<vector<s32>> adjacency((usize)model.total_neuron_count);
    for (const auto &projection : model.projections) {
        for (const auto &connection : projection.connections) {
            adjacency[(usize)connection.source_neuron_index].push_back(connection.target_neuron_index);
        }
    }
    return WeightMatrix(adjacency, /*rank=*/1);
}

// One assembled model's live runtime buffers, gathered so a driving loop
// only has to thread one struct through step_tick calls -- same buffer set
// master_kernel_tests.cpp's own fixtures build by hand, just grouped here
// since this file drives two different models over many ticks each rather
// than asserting on one hand-worked trajectory.
struct LiveModelBuffers {
    GpuPointer<f32> network_inputs;
    GpuPointer<s64> last_spiked;
    GpuPointer<s32> next_active_indices;
    GpuPointer<s32> next_active_count;
    GpuPointer<s32> active_generation;
    GpuPointer<bool> emit_spike;
    ModelRuntimeBuffers buffers;
};

LiveModelBuffers make_live_model_buffers(ModelAllocation &allocation, WeightMatrix &weights, s64 total_neuron_count) {
    LiveModelBuffers live;
    live.network_inputs = allocate<f32>((usize)total_neuron_count * sizeof(f32));
    memset(live.network_inputs.get_contents(), 0, (usize)total_neuron_count * sizeof(f32));

    live.last_spiked = allocate<s64>((usize)total_neuron_count * sizeof(s64));
    // -1 (never fired), not the base engine's own zero-initialization
    // convention -- ticket #59's review comment flags that zero-init reads
    // as "fired at tick 0" for a spike-raster sample; -1 sidesteps that
    // entirely for this file's own tick-0-inclusive comparisons.
    std::fill(live.last_spiked.get_contents(), live.last_spiked.get_contents() + total_neuron_count, (s64)-1);

    live.next_active_indices = allocate<s32>((usize)total_neuron_count * sizeof(s32));
    live.next_active_count = allocate<s32>(sizeof(s32));
    live.next_active_count.get_contents()[0] = 0;

    live.active_generation = allocate<s32>((usize)total_neuron_count * sizeof(s32));
    std::fill(live.active_generation.get_contents(), live.active_generation.get_contents() + total_neuron_count, -1);

    live.emit_spike = allocate<bool>((usize)total_neuron_count * sizeof(bool));
    memset(live.emit_spike.get_contents(), 0, (usize)total_neuron_count * sizeof(bool));

    live.buffers.allocation = &allocation;
    live.buffers.weights = &weights;
    live.buffers.network_inputs = live.network_inputs.get_contents();
    live.buffers.last_spiked = live.last_spiked.get_contents();
    live.buffers.next_active_neuron_indices = live.next_active_indices.get_contents();
    live.buffers.next_active_neuron_count = live.next_active_count.get_contents();
    live.buffers.active_generation = live.active_generation.get_contents();
    live.buffers.emit_port_flags["spike"] = live.emit_spike.get_contents();

    return live;
}

// Finds a population's cell type's "v" StateVariable slot (always index 0
// for both GLIF3Cell and GLIF1Cell -- it's declared first in each fixture's
// own <Dynamics>) and returns the absolute cell_state element index for
// neuron `local_index` within `population_index`.
s64 v_state_index(const ModelAllocation &allocation, s32 population_index, s32 local_index) {
    return allocation.cell_type_boundaries.get_contents()[population_index] + local_index;
}

// allocate_model zero-initializes cell_state (allocator_tests.cpp's own
// "Sanity: every cell_state element is zero-initialized" -- it does not
// apply a cell type's own OnStart/initial_value symbolic references at
// allocation time). Every fixture in this file seeds `v` to its own
// population's `EL` (OnStart's own `v = EL`, arch §1.2 S3) by hand instead,
// the same pattern master_kernel_tests.cpp's own fixtures already use
// (`std::fill(allocation.cell_state..., resting_mp)`).
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

// ── GLIF3 single cell driver (deliverable #1) ────────────────────────────

struct Glif3RunResult {
    Vector<f32> membrane_trace; // one sample per tick, tick 0..tick_count-1
    Vector<s64> spike_ticks;    // every tick a spike occurred, in order
};

Glif3RunResult run_glif3_single_cell(s64 tick_count, f32 dt_seconds) {
    ModelSpecification model = load_model_from_nml_fixture("glif3_single_cell");
    Vector<IrProgram> programs = build_type_library_ir_programs(model);

    ModelAllocation allocation = allocate_model(model, programs);
    seed_initial_membrane_potentials(allocation, model);
    WeightMatrix weights = build_weight_matrix(model);
    AssembledModel assembled_model(model, programs);

    LiveModelBuffers live = make_live_model_buffers(allocation, weights, model.total_neuron_count);

    // Real SI-unit stimulus schedule from the fixture's own pulseGenerator +
    // explicitInput (a plain, single-neuron population -- explicitInput
    // parses correctly here, unlike the network fixture, see this file's
    // own header comment).
    StimulusSchedule schedule = build_stimulus_schedule(model, (f64)dt_seconds);

    Glif3RunResult result;
    result.membrane_trace.reserve((usize)tick_count);

    for (s64 tick = 0; tick < tick_count; ++tick) {
        live.buffers.network_inputs[0] += (f32)schedule.current_at(0, tick);
        assembled_model.step_tick(live.buffers, dt_seconds, tick, tick + 1);

        result.membrane_trace.push_back(allocation.cell_state.get_contents()[v_state_index(allocation, 0, 0)]);
        if (live.buffers.last_spiked[0] == tick) result.spike_ticks.push_back(tick);
    }

    return result;
}

// ── GLIF E/I network driver (deliverable #2) ─────────────────────────────

struct NetworkRunResult {
    // Per-neuron (global index) membrane-potential trace and spike ticks.
    Vector<Vector<f32>> membrane_traces; // [neuron_index][tick]
    Vector<Vector<s64>> spike_ticks;     // [neuron_index] -> ticks that neuron spiked
};

NetworkRunResult run_glif_ei_network(s64 tick_count, f32 dt_seconds) {
    ModelSpecification model = load_model_from_nml_fixture("glif_ei_network");
    Vector<IrProgram> programs = build_type_library_ir_programs(model);

    ModelAllocation allocation = allocate_model(model, programs);
    seed_initial_membrane_potentials(allocation, model);
    WeightMatrix weights = build_weight_matrix(model);
    // See this file's own header comment (#3): AssembledModel's fixed
    // propagate stage does not yet invoke a real synapse ComponentType's own
    // per-edge dynamics, so a nonzero (arbitrary, random low-rank
    // reconstructed) scattered weight would not be meaningful either --
    // forcing it to exactly zero makes that explicit rather than silently
    // scattering a number that looks plausible but isn't derived from any
    // of this model's own expOneSynapse/alphaCurrentSynapse/NMDA parameters.
    weights.set_constant_weight(0.0f);

    AssembledModel assembled_model(model, programs);
    LiveModelBuffers live = make_live_model_buffers(allocation, weights, model.total_neuron_count);

    // Manual stimulus reconstruction (see this file's own header comment
    // #2): the fixture's own <pulseGenerator id="pulseGen1" delay="10ms"
    // duration="200ms" amplitude="0.5nA"/>, applied to ExcPop's neuron 0
    // (global neuron index 0, since ExcPop is declared first).
    const f64 seconds_per_tick = (f64)dt_seconds;
    const s64 delay_ticks = (s64)std::round(0.010 / seconds_per_tick);
    const s64 duration_ticks = (s64)std::round(0.200 / seconds_per_tick);
    const f32 amplitude_amperes = 0.5e-9f;
    const s32 stimulus_target_neuron_index = 0;

    NetworkRunResult result;
    result.membrane_traces.resize((usize)model.total_neuron_count);
    result.spike_ticks.resize((usize)model.total_neuron_count);
    for (auto &trace : result.membrane_traces) trace.reserve((usize)tick_count);

    for (s64 tick = 0; tick < tick_count; ++tick) {
        if (tick >= delay_ticks && tick < delay_ticks + duration_ticks) {
            live.buffers.network_inputs[stimulus_target_neuron_index] += amplitude_amperes;
        }
        assembled_model.step_tick(live.buffers, dt_seconds, tick, tick + 1);

        for (s32 population_index = 0; population_index < (s32)model.populations.size(); ++population_index) {
            const PopulationEntry &population = model.populations[(usize)population_index];
            for (s32 local_index = 0; local_index < population.size; ++local_index) {
                s32 neuron_index = population.neuron_index_begin + local_index;
                result.membrane_traces[(usize)neuron_index].push_back(
                    allocation.cell_state.get_contents()[v_state_index(allocation, population_index, local_index)]);
                if (live.buffers.last_spiked[neuron_index] == tick) {
                    result.spike_ticks[(usize)neuron_index].push_back(tick);
                }
            }
        }
    }

    return result;
}

// ── Reference-data loader (jLEMS OutputFile / EventOutputFile TIME_ID format) ─
//
// jLEMS's own plain-text formats (see pyneuroml.pynml.reload_standard_dat_file's
// own doc comment, mirrored here for a self-contained C++ reader): an
// OutputFile row is "<time_seconds> <col0> <col1> ... <colN>" (whitespace
// separated); an EventOutputFile row (format="TIME_ID") is "<time_seconds>
// <selection_id>".

struct ReferenceTrace {
    Vector<f64> time_seconds;
    Vector<Vector<f64>> column_values; // column_values[row_index][column_index]
};

ReferenceTrace load_reference_trace(const String &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("exit_model_validation_tests: could not open reference trace file '" + path + "'");
    }

    ReferenceTrace trace;
    String line;
    while (std::getline(file, line)) {
        if (line.find_first_not_of(" \t\r\n") == String::npos) continue;
        std::istringstream line_stream(line);
        f64 time_value = 0.0;
        line_stream >> time_value;
        trace.time_seconds.push_back(time_value);

        Vector<f64> row_values;
        f64 value = 0.0;
        while (line_stream >> value) row_values.push_back(value);
        trace.column_values.push_back(std::move(row_values));
    }
    return trace;
}

struct ReferenceSpikeRecord {
    f64 time_seconds = 0.0;
    String selection_id;
};

Vector<ReferenceSpikeRecord> load_reference_spikes(const String &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("exit_model_validation_tests: could not open reference spikes file '" + path + "'");
    }

    Vector<ReferenceSpikeRecord> spikes;
    String line;
    while (std::getline(file, line)) {
        if (line.find_first_not_of(" \t\r\n") == String::npos) continue;
        std::istringstream line_stream(line);
        ReferenceSpikeRecord record;
        line_stream >> record.time_seconds >> record.selection_id;
        spikes.push_back(std::move(record));
    }
    return spikes;
}

} // namespace

// ── GLIF3 single cell: front-end + IR + allocation + compile (enabled) ────

TEST(ExitModelGlif3SingleCell, parses_resolves_lowers_allocates_and_compiles_without_throwing) {
    ModelSpecification model = load_model_from_nml_fixture("glif3_single_cell");
    ASSERT_EQ(model.total_neuron_count, 1);
    // GLIF3Cell (Cell) + pulseGen1 (Inputs, bound via explicitInput -- ticket
    // #7's own ModelSpecification catalogs every bound instance a
    // population/projection/explicitInput references, arch §1.4).
    ASSERT_EQ(model.type_library.size(), 2u);

    s32 cell_type_library_index = -1;
    for (s32 index = 0; index < (s32)model.type_library.size(); ++index) {
        if (model.type_library[(usize)index].category == TypeLibraryCategory::Cell) cell_type_library_index = index;
    }
    ASSERT_NE(cell_type_library_index, -1);
    const TypeLibraryEntry &cell_entry = model.type_library[(usize)cell_type_library_index];
    EXPECT_EQ(cell_entry.component_type_name, "GLIF3Cell");
    EXPECT_EQ(cell_entry.state_variable_count, 4); // v, asc1, asc2, refractoryTimeElapsed

    Vector<IrProgram> programs = build_type_library_ir_programs(model);
    ASSERT_EQ(programs.size(), 2u);

    ModelAllocation allocation = allocate_model(model, programs);
    EXPECT_EQ(allocation.cell_state_element_count, 4);

    EXPECT_NO_THROW({
        AssembledModel assembled_model(model, programs);
        (void)assembled_model;
    });
}

TEST(ExitModelGlif3SingleCell, explicit_input_resolves_to_a_real_stimulus_schedule) {
    ModelSpecification model = load_model_from_nml_fixture("glif3_single_cell");
    ASSERT_EQ(model.stimuli.size(), 1u);
    EXPECT_EQ(model.stimuli[0].target_neuron_index, 0);

    StimulusSchedule schedule = build_stimulus_schedule(model, /*seconds_step=*/1e-4);
    ASSERT_EQ(schedule.windows.size(), 1u);
    // delay=20ms, duration=300ms, amplitude=0.6nA (glif3_single_cell.nml).
    EXPECT_EQ(schedule.windows[0].start_tick, 200);
    EXPECT_EQ(schedule.windows[0].end_tick, 3200);
    EXPECT_NEAR(schedule.windows[0].current_value, 0.6e-9, 1e-15);
}

// ── GLIF3 single cell: driven simulation sanity (enabled) ─────────────────
//
// Not a numeric comparison against the pyneuroml reference (that's the
// DISABLED_ test at the bottom of this file) -- just the same class of
// non-vacuous sanity master_kernel_tests.cpp's own equivalence test asserts
// (some spikes actually happened; the trace stays finite), demonstrating the
// assembled model genuinely runs end to end over the fixture's own full
// 350ms/0.1ms horizon (3500 ticks).

TEST(ExitModelGlif3SingleCell, driven_simulation_produces_spike_frequency_adaptation) {
    const s64 tick_count = 3500;
    const f32 dt_seconds = 1e-4f;

    Glif3RunResult result = run_glif3_single_cell(tick_count, dt_seconds);

    ASSERT_EQ(result.membrane_trace.size(), (usize)tick_count);
    for (f32 voltage : result.membrane_trace) {
        ASSERT_TRUE(std::isfinite(voltage));
        ASSERT_LT(std::abs(voltage), 10.0f) << "membrane potential diverged";
    }

    // The real pyneuroml reference fires 13 times under these exact
    // parameters (this file's own checked-in glif3_spikes.dat); a loose
    // range here (not an exact 13) keeps this specific test robust to
    // forward-Euler discretization differences that don't change the
    // qualitative behavior -- the DISABLED_ test below does the exact
    // comparison against the checked-in reference.
    EXPECT_GE(result.spike_ticks.size(), 5u);
    EXPECT_LE(result.spike_ticks.size(), 25u);

    // Spike-frequency adaptation: successive inter-spike intervals grow
    // (the after-spike currents accumulate and increasingly suppress
    // firing) -- checked over the first few spikes, where the effect is
    // clearest before the driving pulse ends.
    ASSERT_GE(result.spike_ticks.size(), 3u);
    s64 first_isi = result.spike_ticks[1] - result.spike_ticks[0];
    s64 second_isi = result.spike_ticks[2] - result.spike_ticks[1];
    EXPECT_GT(second_isi, first_isi);
}

// ── GLIF E/I network: front-end + IR + allocation + compile (enabled) ─────

TEST(ExitModelGlifEiNetwork, parses_resolves_lowers_allocates_and_compiles_without_throwing) {
    ModelSpecification model = load_model_from_nml_fixture("glif_ei_network");
    ASSERT_EQ(model.total_neuron_count, 5); // ExcPop size 3 + InhPop size 2
    ASSERT_EQ(model.populations.size(), 2u);
    ASSERT_EQ(model.projections.size(), 3u);

    // 2 Cell entries (excitatory/inhibitory GLIF1Cell) + 3 Synapse entries
    // (expOneSynapse, alphaCurrentSynapse, TestNmdaSynapse/NMDA).
    ASSERT_EQ(model.type_library.size(), 5u);
    s32 cell_count = 0, synapse_count = 0;
    for (const auto &entry : model.type_library) {
        if (entry.category == TypeLibraryCategory::Cell) ++cell_count;
        if (entry.category == TypeLibraryCategory::Synapse) ++synapse_count;
    }
    EXPECT_EQ(cell_count, 2);
    EXPECT_EQ(synapse_count, 3);

    bool found_conductance_based = false, found_current_based = false;
    for (const auto &entry : model.type_library) {
        if (entry.category != TypeLibraryCategory::Synapse) continue;
        if (entry.is_conductance_based) found_conductance_based = true;
        else found_current_based = true;
    }
    EXPECT_TRUE(found_conductance_based); // expOneSynapse and/or the NMDA synapse
    EXPECT_TRUE(found_current_based);     // alphaCurrentSynapse

    Vector<IrProgram> programs = build_type_library_ir_programs(model);
    ASSERT_EQ(programs.size(), 5u);

    ModelAllocation allocation = allocate_model(model, programs);
    // ExcPop: 3 neurons * V_t 2 (v, refractoryTimeElapsed) = 6.
    // InhPop: 2 neurons * V_t 2 = 4.
    EXPECT_EQ(allocation.cell_state_element_count, 10);

    EXPECT_NO_THROW({
        AssembledModel assembled_model(model, programs);
        (void)assembled_model;
    });
}

TEST(ExitModelGlifEiNetwork, front_end_does_not_recognize_inputList_yet_documented_gap) {
    // See this file's own header comment (#2) -- asserts the documented gap
    // itself stays true, so a future ticket that adds inputList/input
    // parsing doesn't silently leave this file's own manual stimulus
    // reconstruction stale without anyone noticing.
    ModelSpecification model = load_model_from_nml_fixture("glif_ei_network");
    EXPECT_TRUE(model.stimuli.empty());
}

// ── GLIF E/I network: driven simulation sanity (enabled) ──────────────────

TEST(ExitModelGlifEiNetwork, driven_simulation_spikes_the_directly_stimulated_neuron_only) {
    const s64 tick_count = 2500;
    const f32 dt_seconds = 1e-4f;

    NetworkRunResult result = run_glif_ei_network(tick_count, dt_seconds);
    ASSERT_EQ(result.spike_ticks.size(), 5u);

    for (f32 voltage : result.membrane_traces[0]) ASSERT_TRUE(std::isfinite(voltage));

    // ExcPop[0] (global neuron 0) is the only neuron directly driven by this
    // file's own manually-reconstructed stimulus window -- it should spike.
    EXPECT_GE(result.spike_ticks[0].size(), 1u);

    // See this file's own header comment (#3): with the propagate stage's
    // scattered weight forced to exactly zero (no real per-edge synapse
    // dynamics wired in yet), no OTHER neuron receives any input at all, so
    // none of them should spike -- unlike the real jLEMS reference, which
    // genuinely propagates through expOneSynapse/alphaCurrentSynapse/NMDA to
    // ExcPop[2] and InhPop[0] (see glif_ei_network_spikes.dat).
    for (usize neuron_index = 1; neuron_index < result.spike_ticks.size(); ++neuron_index) {
        EXPECT_TRUE(result.spike_ticks[neuron_index].empty())
            << "neuron_index=" << neuron_index << " unexpectedly spiked with zero propagated weight";
    }
}

// ── Reference-data loader (enabled -- loads the checked-in fixture data, does
// NOT call pyneuroml at test time) ─────────────────────────────────────────

TEST(ExitModelReferenceDataLoader, loads_glif3_single_cell_membrane_trace_and_spikes) {
    ReferenceTrace trace = load_reference_trace(
        fixture_path("reference_data/glif3_single_cell/glif3_membrane_trace.dat"));
    // 350ms at 0.1ms steps -> 3501 rows (tick 0 through 3500 inclusive);
    // 3 recorded columns: v, asc1, asc2.
    ASSERT_EQ(trace.time_seconds.size(), 3501u);
    ASSERT_EQ(trace.column_values.size(), 3501u);
    EXPECT_EQ(trace.column_values[0].size(), 3u);
    EXPECT_NEAR(trace.time_seconds[0], 0.0, 1e-12);
    EXPECT_NEAR(trace.time_seconds.back(), 0.35, 1e-9);
    EXPECT_NEAR(trace.column_values[0][0], -0.07, 1e-6); // v starts at EL = -70mV

    Vector<ReferenceSpikeRecord> spikes =
        load_reference_spikes(fixture_path("reference_data/glif3_single_cell/glif3_spikes.dat"));
    // Real captured jLEMS spike-frequency-adaptation raster under this
    // fixture's own current step: 13 spikes with growing inter-spike
    // intervals (see glif3_single_cell.nml's own header comment).
    ASSERT_EQ(spikes.size(), 13u);
    for (const auto &spike : spikes) EXPECT_EQ(spike.selection_id, "sel0");
    for (usize index = 1; index < spikes.size(); ++index) {
        EXPECT_GT(spikes[index].time_seconds, spikes[index - 1].time_seconds);
    }
}

TEST(ExitModelReferenceDataLoader, loads_glif_ei_network_membrane_trace_and_spike_raster) {
    ReferenceTrace trace = load_reference_trace(
        fixture_path("reference_data/glif_ei_network/glif_ei_network_membrane_trace.dat"));
    // 250ms at 0.1ms steps -> 2501 rows; 5 recorded columns (v for every
    // neuron: ExcPop[0..2], InhPop[0..1]).
    ASSERT_EQ(trace.time_seconds.size(), 2501u);
    EXPECT_EQ(trace.column_values[0].size(), 5u);

    Vector<ReferenceSpikeRecord> spikes =
        load_reference_spikes(fixture_path("reference_data/glif_ei_network/glif_ei_network_spikes.dat"));
    ASSERT_EQ(spikes.size(), 192u);

    // Real captured jLEMS spike raster: the directly-driven ExcPop[0] fires,
    // propagates through alphaCurrentSynapse to ExcPop[2], which propagates
    // through the NMDA synapse (and ExcPop[0] through expOneSynapse) to
    // InhPop[0] -- genuine multi-synapse network propagation. ExcPop[1] and
    // InhPop[1] are unconnected and never spike.
    UnorderedMap<String, s64> spike_count_by_selection;
    for (const auto &spike : spikes) spike_count_by_selection[spike.selection_id]++;
    EXPECT_EQ(spike_count_by_selection["sel_exc0"], 28);
    EXPECT_EQ(spike_count_by_selection["sel_exc2"], 56);
    EXPECT_EQ(spike_count_by_selection["sel_inh0"], 108);
    EXPECT_EQ(spike_count_by_selection.count("sel_exc1"), 0u);
    EXPECT_EQ(spike_count_by_selection.count("sel_inh1"), 0u);
}

// ── DISABLED_: spikecorec vs. real pyneuroml/jLEMS reference (per ticket #61's
// clarified scope, these do NOT need to pass in this ticket's PR -- the user
// enables them manually later) ─────────────────────────────────────────────

TEST(ExitModelValidation, DISABLED_glif3_single_cell_matches_pyneuroml_reference) {
    const s64 tick_count = 3500;
    const f32 dt_seconds = 1e-4f;
    const f32 voltage_tolerance = 1e-3f; // 1mV
    const f64 spike_time_tolerance_seconds = 1e-3; // 1ms

    Glif3RunResult own_result = run_glif3_single_cell(tick_count, dt_seconds);
    ReferenceTrace reference_trace = load_reference_trace(
        fixture_path("reference_data/glif3_single_cell/glif3_membrane_trace.dat"));
    Vector<ReferenceSpikeRecord> reference_spikes =
        load_reference_spikes(fixture_path("reference_data/glif3_single_cell/glif3_spikes.dat"));

    ASSERT_EQ(own_result.spike_ticks.size(), reference_spikes.size());
    for (usize index = 0; index < reference_spikes.size(); ++index) {
        f64 own_spike_time_seconds = (f64)own_result.spike_ticks[index] * dt_seconds;
        EXPECT_NEAR(own_spike_time_seconds, reference_spikes[index].time_seconds, spike_time_tolerance_seconds)
            << "spike index " << index;
    }

    ASSERT_EQ(own_result.membrane_trace.size() + 1, reference_trace.time_seconds.size());
    for (s64 tick = 0; tick < tick_count; ++tick) {
        // reference_trace row (tick+1) is this tick's post-integration v
        // (row 0 is the t=0 initial condition, before any integrate step).
        EXPECT_NEAR(own_result.membrane_trace[(usize)tick], reference_trace.column_values[(usize)tick + 1][0],
                    voltage_tolerance)
            << "tick=" << tick;
    }
}

TEST(ExitModelValidation, DISABLED_glif_ei_network_matches_pyneuroml_reference) {
    // See this file's own header comment (#3): this comparison cannot pass
    // today even in principle -- AssembledModel's propagate stage doesn't
    // yet invoke the real per-edge synapse dynamics the jLEMS reference
    // actually ran, so ExcPop[2]/InhPop[0] never spike in spikecorec's own
    // simulation regardless of tolerance. Left DISABLED_ (not merely
    // skipped) so re-enabling it is a deliberate, visible step once that
    // subsystem exists, per this ticket's own clarified scope.
    const s64 tick_count = 2500;
    const f32 dt_seconds = 1e-4f;
    const f64 spike_time_tolerance_seconds = 1e-3;

    NetworkRunResult own_result = run_glif_ei_network(tick_count, dt_seconds);
    Vector<ReferenceSpikeRecord> reference_spikes =
        load_reference_spikes(fixture_path("reference_data/glif_ei_network/glif_ei_network_spikes.dat"));

    UnorderedMap<String, Vector<f64>> reference_spike_times_by_selection;
    for (const auto &spike : reference_spikes) {
        reference_spike_times_by_selection[spike.selection_id].push_back(spike.time_seconds);
    }

    // Global neuron index -> selection id, matching glif_ei_network.nml's
    // own EventSelection declarations (ExcPop 0/1/2, InhPop 3/4).
    static const char *const SELECTION_ID_BY_NEURON_INDEX[] = {
        "sel_exc0", "sel_exc1", "sel_exc2", "sel_inh0", "sel_inh1"};

    for (usize neuron_index = 0; neuron_index < own_result.spike_ticks.size(); ++neuron_index) {
        const Vector<f64> &reference_times = reference_spike_times_by_selection[SELECTION_ID_BY_NEURON_INDEX[neuron_index]];
        ASSERT_EQ(own_result.spike_ticks[neuron_index].size(), reference_times.size())
            << "neuron_index=" << neuron_index;
        for (usize spike_index = 0; spike_index < reference_times.size(); ++spike_index) {
            f64 own_spike_time_seconds = (f64)own_result.spike_ticks[neuron_index][spike_index] * dt_seconds;
            EXPECT_NEAR(own_spike_time_seconds, reference_times[spike_index], spike_time_tolerance_seconds)
                << "neuron_index=" << neuron_index << " spike_index=" << spike_index;
        }
    }
}
