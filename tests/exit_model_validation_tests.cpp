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
#include "spikecorec/nml/delay_ring.h"
#include "spikecorec/nml/inputs_lowering.h"
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
    // ticket #131: AssembledModel now dispatches this model's real expOneSynapse/
    // alphaCurrentSynapse/NMDA per-edge dynamics for a model with real projections (which this one
    // has), forcing this constant-weight placeholder's own scattered contribution to zero
    // regardless (see master_kernel.h) -- left set anyway as an explicit, harmless no-op rather
    // than deleting the call, matching this file's own established "state the placeholder was
    // here" convention, previously documented at this file's own header comment (#3).
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

TEST(ExitModelGlifEiNetwork, driven_simulation_spikes_via_real_per_edge_synapse_propagation) {
    // ticket #131: was `driven_simulation_spikes_the_directly_stimulated_neuron_only`, asserting
    // that NO other neuron ever spiked (run_glif_ei_network's own constant weight was forced to
    // exactly zero, and nothing dispatched any synapse ComponentType's own dynamics, so nothing
    // COULD propagate). AssembledModel now dispatches this model's real expOneSynapse/
    // alphaCurrentSynapse/NMDA per-edge dynamics (run_glif_ei_network's own constant-weight
    // placeholder is now inert -- see master_kernel.h), so real propagation happens exactly as the
    // checked-in jLEMS reference (glif_ei_network_spikes.dat) shows: ExcPop[0] -> InhPop[0] via a
    // real expOneSynapse. ExcPop[0] -> ExcPop[2] (via alphaCurrentSynapse) and the ExcPop[2] ->
    // InhPop[0] NMDA leg do NOT yet reproduce the reference -- alphaCurrentSynapse's own two-
    // state-variable coupled TimeDerivative (`I`/`J`) isn't a recognized linear-decay shape
    // (synapse_lowering.cpp's own documented, separate limitation: "general per-edge forward-Euler
    // integration for an arbitrary right-hand side is out of Phase-1 scope"), so `I` never
    // integrates and ExcPop[2] never receives a nonzero current -- a real, pre-existing, orthogonal
    // gap this ticket does not fix (see DISABLED_glif_ei_network_matches_pyneuroml_reference below,
    // left disabled for exactly this reason).
    const s64 tick_count = 2500;
    const f32 dt_seconds = 1e-4f;

    NetworkRunResult result = run_glif_ei_network(tick_count, dt_seconds);
    ASSERT_EQ(result.spike_ticks.size(), 5u);

    for (f32 voltage : result.membrane_traces[0]) ASSERT_TRUE(std::isfinite(voltage));

    // ExcPop[0] (global neuron 0) is the only neuron directly driven by this
    // file's own manually-reconstructed stimulus window -- it should spike.
    EXPECT_GE(result.spike_ticks[0].size(), 1u);

    // ExcPop[1] and InhPop[1] are genuinely unconnected (no incoming projection at all, matching
    // the reference raster's own "sel_exc1"/"sel_inh1" never appearing) -- they must stay silent
    // regardless of synapse dispatch.
    EXPECT_TRUE(result.spike_ticks[1].empty()) << "ExcPop[1] is unconnected and should never spike";
    EXPECT_TRUE(result.spike_ticks[4].empty()) << "InhPop[1] is unconnected and should never spike";

    // ExcPop[2]: still silent -- see this test's own header comment above (alphaCurrentSynapse's
    // coupled-ODE gap, not a "zero propagated weight" placeholder anymore).
    EXPECT_TRUE(result.spike_ticks[2].empty())
        << "ExcPop[2] should stay silent until alphaCurrentSynapse's coupled TimeDerivative is lowered";

    // InhPop[0]: the ticket #131 acceptance criterion, exercised directly -- ExcPop[0]'s own spike
    // reaches InhPop[0] through expOneSynapse's real, gbase/tauDecay/erev-derived per-edge current
    // (arch §4.3), not a constant-weight placeholder.
    EXPECT_GE(result.spike_ticks[3].size(), 1u)
        << "InhPop[0] should spike via expOneSynapse's real per-edge conductance from ExcPop[0]'s spikes";
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

// ── Phase-2 validation / exit models (ticket #67 [H2]; arch §5 Phase 2) ──────────────────────────
//
// The three arch §5 Phase-2 exit models: an izhikevich network exercising real, jLEMS-verified
// Regular Spiking dynamics through a real (if minimal) synaptic connection
// (tests/fixtures/nml/izhikevich_network.nml); a delayed-coupling network exercising the delay-ring
// subsystem (ticket #64 [F3], include/spikecorec/nml/delay_ring.h) with a real, non-trivial
// `connectionWD` delay (tests/fixtures/nml/delayed_coupling_network.nml); and a Poisson-driven
// population generated on device (ticket #65 [F4]'s own SpikeSourcePoisson generator,
// tests/fixtures/nml/poisson_population.nml). All three fixtures follow ticket #61's own established
// convention exactly: real, checked-in, standalone NeuroML/LEMS files (not inline C++ string
// literals) driving BOTH pyneuroml/jNeuroML (the real reference simulator) and spikecorec's own
// pipeline -- see each fixture's own header comment for the exact ComponentTypes/topology and why.
//
// Per the ticket's scope clarification (task_master, 2026-07-16, same policy as ticket #61's own):
// the captured reference data under tests/fixtures/reference_data/{izhikevich_network,
// delayed_coupling_network,poisson_population}/ is REAL output from running each fixture through
// pyneuroml/jNeuroML (commands documented in each fixture's own LEMS_*.xml header comment) -- not an
// analytic stand-in. Everything in this section EXCEPT the DISABLED_-prefixed numeric-comparison
// tests at the bottom runs and passes normally; those comparison tests are intentionally disabled
// (do not need to pass in this ticket's PR) per that same scope clarification -- the user will
// manually enable them later and verify against this same checked-in reference data.
//
// ── Known limitations a reviewer should look at closely (Phase-2 section) ────────────────────────
// 1. Both network fixtures use `<inputList>`/`<input>` for their own pulseGenerator stimulus (NOT
//    `<explicitInput>`) -- the SAME real, jLEMS-proven-working shape glif_ei_network.nml's own header
//    comment documents for a multi-population network against this exact jLEMS build (bundled with
//    pyneuroml 1.3.22). Per this file's own header comment (#2) above, spikecorec's own front-end
//    does not recognize `inputList` either, so `model.stimuli` is empty for both, and each fixture's
//    own driver below reconstructs the identical stimulus window by hand from the .nml's own literal
//    `pulseGenerator` attributes, exactly mirroring `run_glif_ei_network`'s own established pattern.
// 2. AssembledModel's fixed propagate stage still does not invoke a real per-edge synapse
//    ComponentType's own dynamics (this file's own header comment #3, carried over unchanged from
//    ticket #61 -- no ticket in CLAUDE.md's own epic ticket graph has built the "spike-scatter batch
//    construction" subsystem gpu_source.h's own header comment flags as needed for that yet). Both
//    network fixtures' own drivers below force the WeightMatrix's scattered value to either exactly
//    zero (izhikevich network -- a magnitude comparison would be meaningless without real synapse
//    dynamics, matching glif_ei_network's own precedent) or an arbitrary nonzero placeholder
//    (delayed-coupling network -- this exit model's own validation target is delivery TIMING, which
//    the delay ring derives purely from the connection's own `delay` attribute independent of
//    whatever value gets scattered, so no magnitude comparison is attempted there at all).
// 3. The delayed-coupling network's own DISABLED_ comparison test compares spikecorec's own delay-ring
//    delivery tick against a tick DERIVED from the real captured jLEMS reference trace (the first
//    tick TargetPop's own voltage departs from its resting potential) -- not literally re-deriving
//    the expected tick from the connection's own `delay` attribute a second time (which would be
//    circular and would not actually exercise cross-simulator agreement at all).
// 4. The Poisson population's own comparison is explicitly a STATISTICAL one (aggregate spike count
//    over the recorded window), not spike-for-spike -- see poisson_population.nml's own header
//    comment for why an exact comparison is not meaningful even in principle for this exit model
//    (two structurally different PRNG-driven algorithms, two disjoint PRNG streams).

namespace {

// ── izhikevich network driver (exit model #1) ─────────────────────────────────────────────────────
//
// izhikevich2007Cell's own OnStart (v=v0, u=0) -- allocate_model doesn't apply a StateDirective's own
// initial_value yet (nonlinear_cell_lowering_tests.cpp's own SingleNeuronHarness::seed_state doc
// comment), so izhikevich_network.nml's own two single-neuron populations (both v0=-60mV) are seeded
// by hand here, mirroring seed_initial_membrane_potentials's own GLIF-specific convention above but
// for a 2-state-variable (v, u) cell type instead of GLIF's EL-only v seed. v is state-slot 0 (as for
// every GLIF cell type above), so v_state_index (this file's own ticket #61 helper) is reused
// unchanged to read it back out; u (state-slot 1) is written directly here since no test in this
// file ever needs to read it back.
void seed_izhikevich_initial_state(ModelAllocation &allocation, const ModelSpecification &model) {
    const f32 v0_volts = -0.06f; // izhikevich_network.nml's own v0="-60mV", both populations
    for (s32 population_index = 0; population_index < (s32)model.populations.size(); ++population_index) {
        const PopulationEntry &population = model.populations[(usize)population_index];
        s64 chunk_base = allocation.cell_type_boundaries.get_contents()[population_index];
        for (s32 local_index = 0; local_index < population.size; ++local_index) {
            allocation.cell_state.get_contents()[chunk_base + local_index] = v0_volts;                // v
            allocation.cell_state.get_contents()[chunk_base + population.size + local_index] = 0.0f;  // u
        }
    }
}

struct IzhikevichNetworkRunResult {
    Vector<f32> driven_membrane_trace; // one sample per tick, DrivenPop's own neuron
    Vector<s64> driven_spike_ticks;
    Vector<s64> target_spike_ticks;
};

IzhikevichNetworkRunResult run_izhikevich_network(s64 tick_count, f32 dt_seconds) {
    ModelSpecification model = load_model_from_nml_fixture("izhikevich_network");
    Vector<IrProgram> programs = build_type_library_ir_programs(model);

    ModelAllocation allocation = allocate_model(model, programs);
    seed_izhikevich_initial_state(allocation, model);
    WeightMatrix weights = build_weight_matrix(model);
    // See this file's own header comment (Phase-2 section, #2): forced to exactly zero, matching
    // glif_ei_network's own established precedent -- the driven_simulation... test below only
    // asserts on DrivenPop, never TargetPop, for the same documented reason.
    weights.set_constant_weight(0.0f);

    AssembledModel assembled_model(model, programs);
    LiveModelBuffers live = make_live_model_buffers(allocation, weights, model.total_neuron_count);

    // Manual stimulus reconstruction (see this file's own header comment, Phase-2 section, #1):
    // izhikevich_network.nml's own <pulseGenerator id="pulseGen1" delay="10ms" duration="200ms"
    // amplitude="150pA"/>, applied to DrivenPop's neuron 0 (global neuron index 0, declared first).
    const f64 seconds_per_tick = (f64)dt_seconds;
    const s64 delay_ticks = (s64)std::round(0.010 / seconds_per_tick);
    const s64 duration_ticks = (s64)std::round(0.200 / seconds_per_tick);
    const f32 amplitude_amperes = 150e-12f;

    IzhikevichNetworkRunResult result;
    result.driven_membrane_trace.reserve((usize)tick_count);

    for (s64 tick = 0; tick < tick_count; ++tick) {
        if (tick >= delay_ticks && tick < delay_ticks + duration_ticks) {
            live.buffers.network_inputs[0] += amplitude_amperes;
        }
        assembled_model.step_tick(live.buffers, dt_seconds, tick, tick + 1);

        result.driven_membrane_trace.push_back(allocation.cell_state.get_contents()[v_state_index(allocation, 0, 0)]);
        if (live.buffers.last_spiked[0] == tick) result.driven_spike_ticks.push_back(tick);
        if (live.buffers.last_spiked[1] == tick) result.target_spike_ticks.push_back(tick);
    }

    return result;
}

// ── delayed-coupling network driver (exit model #2) ───────────────────────────────────────────────

struct DelayedCouplingRunResult {
    Vector<s64> source_spike_ticks;
    Vector<s64> target_delivery_ticks; // ticks at which the delay ring's own input_ring shows a
                                        // nonzero contribution reaching TargetPop's own neuron
};

DelayedCouplingRunResult run_delayed_coupling_network(s64 tick_count, f32 dt_seconds) {
    ModelSpecification model = load_model_from_nml_fixture("delayed_coupling_network");
    Vector<IrProgram> programs = build_type_library_ir_programs(model);

    ModelAllocation allocation = allocate_model(model, programs);
    seed_initial_membrane_potentials(allocation, model); // GLIF1Cell -- reuses ticket #61's own EL seed
    WeightMatrix weights = build_weight_matrix(model);
    // An arbitrary nonzero placeholder (see this file's own header comment, Phase-2 section, #2) --
    // this exit model's own validation target is delivery TIMING (arch §4.4), which the delay ring
    // derives purely from the connection's own `delay` attribute, independent of whatever value gets
    // scattered -- unlike the izhikevich network's own zero-weight precedent (a magnitude comparison
    // that WOULD need real per-edge synapse dynamics), no magnitude comparison is made here at all.
    weights.set_constant_weight(0.6f);

    DelayRingAllocation ring = allocate_delay_ring(model, weights, dt_seconds);
    AssembledModel assembled_model(model, programs, /*enable_delay_ring=*/true);

    GpuPointer<s64> last_spiked = allocate<s64>((usize)model.total_neuron_count * sizeof(s64));
    std::fill(last_spiked.get_contents(), last_spiked.get_contents() + model.total_neuron_count, (s64)-1);
    GpuPointer<bool> emit_spike = allocate<bool>((usize)model.total_neuron_count * sizeof(bool));
    memset(emit_spike.get_contents(), 0, (usize)model.total_neuron_count * sizeof(bool));

    // Delay-ring mode: ModelRuntimeBuffers::network_inputs/next_active_neuron_indices/count/
    // active_generation are unused (superseded by `delay_ring`'s own ring-shaped equivalents,
    // master_kernel.h's own documented contract) and are left unallocated/null.
    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &weights;
    buffers.last_spiked = last_spiked.get_contents();
    buffers.emit_port_flags["spike"] = emit_spike.get_contents();
    buffers.delay_ring = &ring;

    // Manual stimulus reconstruction (see this file's own header comment, Phase-2 section, #1):
    // delayed_coupling_network.nml's own <pulseGenerator id="pulseGen1" delay="10ms" duration="6ms"
    // amplitude="0.5nA"/>, applied to SourcePop's neuron 0 (global neuron index 0, declared first).
    const f64 seconds_per_tick = (f64)dt_seconds;
    const s64 stimulus_delay_ticks = (s64)std::round(0.010 / seconds_per_tick);
    const s64 stimulus_duration_ticks = (s64)std::round(0.006 / seconds_per_tick);
    const f32 amplitude_amperes = 0.5e-9f;
    const s32 target_neuron_index = 1; // TargetPop, declared second
    const f32 delivery_epsilon = 1e-9f;

    DelayedCouplingRunResult result;
    for (s64 tick = 0; tick < tick_count; ++tick) {
        s64 current_slot = tick % ring.ring_slot_count;
        // This tick's own ring slot (master_kernel.cpp's own step_tick, ticket #64) -- read BEFORE
        // this tick's own stimulus/step_tick call so it reflects only whatever a PRIOR tick's
        // propagate stage already scattered into it (arch §3.5's >=1-tick latency).
        f32 target_network_input_this_tick =
            ring.input_ring.get_contents()[current_slot * model.total_neuron_count + target_neuron_index];
        if (std::fabs(target_network_input_this_tick) > delivery_epsilon) result.target_delivery_ticks.push_back(tick);

        if (tick >= stimulus_delay_ticks && tick < stimulus_delay_ticks + stimulus_duration_ticks) {
            // Direct injection into the ring's own CURRENT slot -- the same real, SI-unit-current
            // stimulus-injection channel run_glif3_single_cell/run_glif_ei_network use above (this
            // cell's own iSyn DerivedVariable aliases straight to whatever the per-population `_tick`
            // kernel reads as "network_inputs" THIS tick), just ring-indexed rather than flat
            // (ModelRuntimeBuffers::delay_ring's own doc comment: the flat `network_inputs` field is
            // unused/superseded in ring mode, but the per-population `_tick` kernel still reads
            // whatever THIS tick's own ring slot holds under that same reserved parameter name --
            // master_kernel.cpp's own `network_inputs_for_this_tick`).
            ring.input_ring.get_contents()[current_slot * model.total_neuron_count + 0] += amplitude_amperes;
        }

        assembled_model.step_tick(buffers, dt_seconds, tick, tick + 1);
        if (buffers.last_spiked[0] == tick) result.source_spike_ticks.push_back(tick);
    }

    return result;
}

// ── Poisson-driven population driver (exit model #3) ──────────────────────────────────────────────
//
// Mirrors ticket #65's own inputs_lowering_tests.cpp
// poisson_population_spike_count_matches_expected_rate_over_several_thousand_ticks acceptance test
// almost exactly (same trivial-ring-adjacency WeightMatrix workaround, same rng_state seeding
// scheme), driving THIS ticket's own checked-in poisson_population.nml fixture (so the SAME real
// NeuroML file drives both jLEMS and spikecorec, this ticket's own established convention) instead of
// that file's own inline `build_inputs_model` helper, at this exit model's own larger population/
// window scale.

struct PoissonPopulationRunResult {
    s64 total_spike_count = 0;
    s32 neurons_that_fired = 0;
};

PoissonPopulationRunResult run_poisson_population(s32 population_size, s64 tick_count, f32 dt_seconds) {
    ModelSpecification model = load_model_from_nml_fixture("poisson_population");
    Vector<IrProgram> programs{lower_inputs_to_ir(model.type_library[0])};
    ModelAllocation allocation = allocate_model(model, programs);

    // WeightMatrix rejects an edge-free adjacency (ticket #65's own established workaround) -- a
    // trivial ring (neuron n -> (n+1)%population_size, since K2Tree rejects self-loops) purely to
    // satisfy that constructor; this test only cares about spike TIMING (last_spiked), never
    // network_inputs/propagation, so the edge's own weight is irrelevant.
    vector<vector<s32>> adjacency((usize)population_size);
    for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
        adjacency[(usize)neuron_index] = {(neuron_index + 1) % population_size};
    }
    WeightMatrix weights(adjacency, /*rank=*/1);
    weights.set_constant_weight(0.0f);

    AssembledModel assembled_model(model, programs);

    GpuPointer<f32> network_inputs = allocate<f32>((usize)population_size * sizeof(f32));
    memset(network_inputs.get_contents(), 0, (usize)population_size * sizeof(f32));
    GpuPointer<s64> last_spiked = allocate<s64>((usize)population_size * sizeof(s64));
    std::fill(last_spiked.get_contents(), last_spiked.get_contents() + population_size, (s64)-1);
    GpuPointer<s32> next_active_indices = allocate<s32>((usize)population_size * sizeof(s32));
    GpuPointer<s32> next_active_count = allocate<s32>(sizeof(s32));
    next_active_count.get_contents()[0] = 0;
    GpuPointer<s32> active_generation = allocate<s32>((usize)population_size * sizeof(s32));
    std::fill(active_generation.get_contents(), active_generation.get_contents() + population_size, -1);
    GpuPointer<bool> emit_spike = allocate<bool>((usize)population_size * sizeof(bool));
    memset(emit_spike.get_contents(), 0, (usize)population_size * sizeof(bool));
    GpuPointer<u32> rng_state = allocate<u32>((usize)population_size * sizeof(u32));
    for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
        rng_state.get_contents()[neuron_index] = (u32)((neuron_index + 1) * 2654435761u) | 1u; // nonzero seed
    }

    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &weights;
    buffers.network_inputs = network_inputs.get_contents();
    buffers.last_spiked = last_spiked.get_contents();
    buffers.next_active_neuron_indices = next_active_indices.get_contents();
    buffers.next_active_neuron_count = next_active_count.get_contents();
    buffers.active_generation = active_generation.get_contents();
    buffers.emit_port_flags["spike"] = emit_spike.get_contents();
    buffers.rng_state = rng_state.get_contents();

    PoissonPopulationRunResult result;
    for (s64 tick = 0; tick < tick_count; ++tick) {
        assembled_model.step_tick(buffers, dt_seconds, tick, tick + 1);
        const s64 *last_spiked_contents = last_spiked.get_contents();
        for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
            if (last_spiked_contents[neuron_index] == tick) ++result.total_spike_count;
        }
    }
    for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
        if (last_spiked.get_contents()[neuron_index] != -1) ++result.neurons_that_fired;
    }
    return result;
}

} // namespace

// ── izhikevich network: front-end + IR + allocation + compile (enabled) ──────────────────────────

TEST(ExitModelIzhikevichNetwork, parses_resolves_lowers_allocates_and_compiles_without_throwing) {
    ModelSpecification model = load_model_from_nml_fixture("izhikevich_network");
    ASSERT_EQ(model.total_neuron_count, 2);
    ASSERT_EQ(model.populations.size(), 2u);
    ASSERT_EQ(model.projections.size(), 1u);

    // izhikevich2007Cell (Cell, ONE bound instance shared by both populations) + izhCurrSynapse
    // (Synapse).
    ASSERT_EQ(model.type_library.size(), 2u);
    s32 cell_count = 0, synapse_count = 0;
    for (const auto &entry : model.type_library) {
        if (entry.category == TypeLibraryCategory::Cell) ++cell_count;
        if (entry.category == TypeLibraryCategory::Synapse) ++synapse_count;
    }
    EXPECT_EQ(cell_count, 1);
    EXPECT_EQ(synapse_count, 1);

    Vector<IrProgram> programs = build_type_library_ir_programs(model);
    ASSERT_EQ(programs.size(), 2u);

    ModelAllocation allocation = allocate_model(model, programs);
    EXPECT_EQ(allocation.cell_state_element_count, 4); // 2 neurons * (v, u)

    EXPECT_NO_THROW({
        AssembledModel assembled_model(model, programs);
        (void)assembled_model;
    });
}

TEST(ExitModelIzhikevichNetwork, front_end_does_not_recognize_inputList_yet_documented_gap) {
    // See this file's own header comment (Phase-2 section, #1).
    ModelSpecification model = load_model_from_nml_fixture("izhikevich_network");
    EXPECT_TRUE(model.stimuli.empty());
}

// ── izhikevich network: driven simulation sanity (enabled) ───────────────────────────────────────

TEST(ExitModelIzhikevichNetwork, driven_simulation_produces_regular_spiking_pattern) {
    const s64 tick_count = 2300; // 230ms / 0.1ms, matching izhikevich_network.nml's own Simulation
    const f32 dt_seconds = 1e-4f;

    IzhikevichNetworkRunResult result = run_izhikevich_network(tick_count, dt_seconds);

    for (f32 voltage : result.driven_membrane_trace) {
        ASSERT_TRUE(std::isfinite(voltage));
        ASSERT_LT(std::abs(voltage), 1.0f) << "membrane potential diverged";
    }

    // The real pyneuroml reference fires 5 times under these exact parameters (this file's own
    // checked-in izhikevich_network_spikes.dat); a loose range here (not an exact 5) keeps this
    // specific test robust to forward-Euler discretization differences, matching
    // ExitModelGlif3SingleCell's own established convention above -- the DISABLED_ test below does
    // the exact comparison against the checked-in reference.
    EXPECT_GE(result.driven_spike_ticks.size(), 3u);
    EXPECT_LE(result.driven_spike_ticks.size(), 15u);

    // Regular spiking's own settled signature (nonlinear_cell_lowering_tests.cpp's own established
    // check): the LAST two inter-spike intervals stay roughly constant.
    usize spike_count = result.driven_spike_ticks.size();
    ASSERT_GE(spike_count, 3u);
    s64 last_isi = result.driven_spike_ticks[spike_count - 1] - result.driven_spike_ticks[spike_count - 2];
    s64 second_last_isi = result.driven_spike_ticks[spike_count - 2] - result.driven_spike_ticks[spike_count - 3];
    f64 isi_relative_difference =
        std::fabs((f64)(last_isi - second_last_isi)) / (f64)std::max(last_isi, second_last_isi);
    EXPECT_LT(isi_relative_difference, 0.25)
        << "last_isi=" << last_isi << " second_last_isi=" << second_last_isi;
}

// ── delayed-coupling network: front-end + IR + allocation + compile (enabled) ─────────────────────

TEST(ExitModelDelayedCouplingNetwork, parses_resolves_lowers_allocates_and_compiles_without_throwing) {
    ModelSpecification model = load_model_from_nml_fixture("delayed_coupling_network");
    ASSERT_EQ(model.total_neuron_count, 2);
    ASSERT_EQ(model.populations.size(), 2u);
    ASSERT_EQ(model.projections.size(), 1u);
    ASSERT_EQ(model.projections[0].connections.size(), 1u);
    EXPECT_NEAR(model.projections[0].connections[0].delay, 0.010, 1e-9); // connectionWD delay="10ms"

    // GLIF1Cell (Cell) + delaySynapse (Synapse).
    ASSERT_EQ(model.type_library.size(), 2u);

    Vector<IrProgram> programs = build_type_library_ir_programs(model);
    ASSERT_EQ(programs.size(), 2u);

    ModelAllocation allocation = allocate_model(model, programs);
    EXPECT_EQ(allocation.cell_state_element_count, 4); // 2 neurons * (v, refractoryTimeElapsed)

    EXPECT_NO_THROW({
        AssembledModel assembled_model(model, programs, /*enable_delay_ring=*/true);
        (void)assembled_model;
    });
}

TEST(ExitModelDelayedCouplingNetwork, front_end_does_not_recognize_inputList_yet_documented_gap) {
    // See this file's own header comment (Phase-2 section, #1).
    ModelSpecification model = load_model_from_nml_fixture("delayed_coupling_network");
    EXPECT_TRUE(model.stimuli.empty());
}

TEST(ExitModelDelayedCouplingNetwork, delay_ring_sizing_matches_the_connections_own_real_delay) {
    ModelSpecification model = load_model_from_nml_fixture("delayed_coupling_network");
    WeightMatrix weights = build_weight_matrix(model);
    const f32 dt_seconds = 1e-4f;
    EXPECT_EQ(compute_max_delay_ticks(model, dt_seconds), 100); // 10ms / 0.1ms

    DelayRingAllocation ring = allocate_delay_ring(model, weights, dt_seconds);
    EXPECT_EQ(ring.ring_slot_count, 101); // max_delay_ticks (100) + 1
}

// ── delayed-coupling network: driven simulation sanity (enabled) ─────────────────────────────────

TEST(ExitModelDelayedCouplingNetwork, driven_simulation_delivers_after_the_source_spike_at_the_expected_offset) {
    const s64 tick_count = 600; // 60ms / 0.1ms, matching delayed_coupling_network.nml's own Simulation
    const f32 dt_seconds = 1e-4f;
    const s64 expected_delay_ticks = 100; // 10ms / 0.1ms

    DelayedCouplingRunResult result = run_delayed_coupling_network(tick_count, dt_seconds);

    // This fixture's own deliberately isolated pulse (delayed_coupling_network.nml's own header
    // comment) -- exactly one source spike, and exactly one delay-ring delivery, landing EXACTLY
    // `expected_delay_ticks` after it (spikecorec-only, no jLEMS reference needed for this check --
    // arithmetic internal to the delay ring itself).
    ASSERT_EQ(result.source_spike_ticks.size(), 1u);
    ASSERT_EQ(result.target_delivery_ticks.size(), 1u);
    EXPECT_EQ(result.target_delivery_ticks[0], result.source_spike_ticks[0] + expected_delay_ticks);
}

// ── Poisson-driven population: front-end + IR + allocation + compile (enabled) ────────────────────

TEST(ExitModelPoissonPopulation, parses_resolves_lowers_allocates_and_compiles_without_throwing) {
    ModelSpecification model = load_model_from_nml_fixture("poisson_population");
    ASSERT_EQ(model.total_neuron_count, 100);
    ASSERT_EQ(model.populations.size(), 1u);
    ASSERT_EQ(model.type_library.size(), 1u);
    EXPECT_EQ(model.type_library[0].category, TypeLibraryCategory::Inputs);
    EXPECT_EQ(model.type_library[0].component_type_name, "SpikeSourcePoisson");

    IrProgram program = lower_inputs_to_ir(model.type_library[0]);
    Vector<IrProgram> programs{program};
    ModelAllocation allocation = allocate_model(model, programs);
    (void)allocation;

    EXPECT_NO_THROW({
        AssembledModel assembled_model(model, programs);
        (void)assembled_model;
    });
}

// ── Poisson-driven population: driven simulation sanity (enabled) ────────────────────────────────

TEST(ExitModelPoissonPopulation, driven_simulation_matches_expected_poisson_rate_self_consistently) {
    // Mirrors ticket #65's own inputs_lowering_tests.cpp statistical-tolerance acceptance test
    // (population_size=50/rate=20Hz/5s there), at THIS exit model's own larger population/window
    // scale (100 neurons/10Hz/8s, matching poisson_population.nml exactly, so the SAME model drives
    // both this spikecorec-only sanity check and the DISABLED_ jLEMS comparison below).
    const s32 population_size = 100;
    const f64 rate_hz = 10.0;
    const s64 tick_count = 8000;
    const f32 dt_seconds = 0.001f;

    PoissonPopulationRunResult result = run_poisson_population(population_size, tick_count, dt_seconds);

    f64 expected_spike_count = (f64)population_size * rate_hz * ((f64)tick_count * (f64)dt_seconds);
    f64 tolerance = expected_spike_count * 0.25;
    EXPECT_NEAR((f64)result.total_spike_count, expected_spike_count, tolerance)
        << "observed=" << result.total_spike_count << " expected=" << expected_spike_count;
    EXPECT_GT(result.neurons_that_fired, population_size / 2);
}

// ── Reference-data loader (enabled -- loads the checked-in fixture data, does NOT call pyneuroml at
// test time) ───────────────────────────────────────────────────────────────────────────────────────

TEST(ExitModelReferenceDataLoader, loads_izhikevich_network_membrane_trace_and_spike_raster) {
    ReferenceTrace trace = load_reference_trace(
        fixture_path("reference_data/izhikevich_network/izhikevich_network_membrane_trace.dat"));
    // 230ms at 0.1ms steps -> 2301 rows; 3 recorded columns (v_driven, u_driven, v_target).
    ASSERT_EQ(trace.time_seconds.size(), 2301u);
    EXPECT_EQ(trace.column_values[0].size(), 3u);
    EXPECT_NEAR(trace.column_values[0][0], -0.06, 1e-6); // v_driven starts at v0 = -60mV

    Vector<ReferenceSpikeRecord> spikes = load_reference_spikes(
        fixture_path("reference_data/izhikevich_network/izhikevich_network_spikes.dat"));
    // Real captured jLEMS raster: 5 DrivenPop spikes, 5 TargetPop spikes (real network propagation
    // through izhCurrSynapse -- see izhikevich_network.nml's own header comment).
    ASSERT_EQ(spikes.size(), 10u);
    s32 driven_count = 0, target_count = 0;
    for (const auto &spike : spikes) {
        if (spike.selection_id == "sel_driven") ++driven_count;
        else if (spike.selection_id == "sel_target") ++target_count;
    }
    EXPECT_EQ(driven_count, 5);
    EXPECT_EQ(target_count, 5);
}

TEST(ExitModelReferenceDataLoader, loads_delayed_coupling_network_membrane_trace_and_spike_raster) {
    ReferenceTrace trace = load_reference_trace(
        fixture_path("reference_data/delayed_coupling_network/delayed_coupling_network_membrane_trace.dat"));
    // 60ms at 0.1ms steps -> 601 rows; 2 recorded columns (v_source, v_target).
    ASSERT_EQ(trace.time_seconds.size(), 601u);
    EXPECT_EQ(trace.column_values[0].size(), 2u);
    EXPECT_NEAR(trace.column_values[0][0], -0.07, 1e-6); // both start at EL = -70mV

    Vector<ReferenceSpikeRecord> spikes = load_reference_spikes(
        fixture_path("reference_data/delayed_coupling_network/delayed_coupling_network_spikes.dat"));
    // Real captured jLEMS raster: exactly one SourcePop spike (this fixture's own deliberately short
    // pulseGenerator window, see delayed_coupling_network.nml's own header comment); TargetPop never
    // reaches its own threshold (the single delayed delivery is a deliberately weak, subthreshold
    // perturbation).
    ASSERT_EQ(spikes.size(), 1u);
    EXPECT_EQ(spikes[0].selection_id, "sel_source");
}

TEST(ExitModelReferenceDataLoader, loads_poisson_population_spike_raster) {
    Vector<ReferenceSpikeRecord> spikes = load_reference_spikes(
        fixture_path("reference_data/poisson_population/poisson_population_spikes.dat"));
    // Real captured jLEMS raster (seed=17, poisson_population.nml): 100 neurons * 10Hz * 8s ->
    // expected 8000, observed 7929 (a genuine sample of the same rate-10Hz Poisson process).
    ASSERT_EQ(spikes.size(), 7929u);

    UnorderedMap<String, s64> spike_count_by_selection;
    for (const auto &spike : spikes) spike_count_by_selection[spike.selection_id]++;
    EXPECT_EQ(spike_count_by_selection.size(), 100u); // every one of the 100 neurons fired at least once
}

// ── DISABLED_: spikecorec vs. real pyneuroml/jLEMS reference (per ticket #67's clarified scope,
// mirroring ticket #61's own -- these do NOT need to pass in this ticket's PR) ─────────────────────

TEST(ExitModelValidation, DISABLED_izhikevich_network_driven_neuron_matches_pyneuroml_reference) {
    const s64 tick_count = 2300;
    const f32 dt_seconds = 1e-4f;
    const f64 spike_time_tolerance_seconds = 1e-3; // matches ExitModelGlif3SingleCell's own tolerance

    IzhikevichNetworkRunResult own_result = run_izhikevich_network(tick_count, dt_seconds);
    Vector<ReferenceSpikeRecord> reference_spikes = load_reference_spikes(
        fixture_path("reference_data/izhikevich_network/izhikevich_network_spikes.dat"));

    Vector<f64> reference_driven_spike_times;
    for (const auto &spike : reference_spikes) {
        if (spike.selection_id == "sel_driven") reference_driven_spike_times.push_back(spike.time_seconds);
    }

    ASSERT_EQ(own_result.driven_spike_ticks.size(), reference_driven_spike_times.size());
    for (usize index = 0; index < reference_driven_spike_times.size(); ++index) {
        f64 own_spike_time_seconds = (f64)own_result.driven_spike_ticks[index] * dt_seconds;
        EXPECT_NEAR(own_spike_time_seconds, reference_driven_spike_times[index], spike_time_tolerance_seconds)
            << "spike index " << index;
    }
}

TEST(ExitModelValidation, DISABLED_izhikevich_network_target_neuron_does_not_yet_match_pyneuroml_reference) {
    // See izhikevich_network.nml's own header comment / this file's own header comment (Phase-2
    // section, #2): AssembledModel's fixed propagate stage does not yet invoke izhCurrSynapse's own
    // real per-edge dynamics -- with the scattered weight forced to exactly zero, TargetPop in
    // spikecorec's own simulation never receives any input at all and so never spikes, regardless of
    // tolerance, unlike the real jLEMS reference (which genuinely propagates through izhCurrSynapse
    // and shows 5 TargetPop spikes). Left DISABLED_ (not merely skipped) so re-enabling it is a
    // deliberate, visible step once that subsystem exists -- mirrors ticket #61's own
    // DISABLED_glif_ei_network_matches_pyneuroml_reference precedent exactly.
    const s64 tick_count = 2300;
    const f32 dt_seconds = 1e-4f;

    IzhikevichNetworkRunResult own_result = run_izhikevich_network(tick_count, dt_seconds);
    Vector<ReferenceSpikeRecord> reference_spikes = load_reference_spikes(
        fixture_path("reference_data/izhikevich_network/izhikevich_network_spikes.dat"));

    Vector<f64> reference_target_spike_times;
    for (const auto &spike : reference_spikes) {
        if (spike.selection_id == "sel_target") reference_target_spike_times.push_back(spike.time_seconds);
    }

    ASSERT_EQ(own_result.target_spike_ticks.size(), reference_target_spike_times.size());
    for (usize index = 0; index < reference_target_spike_times.size(); ++index) {
        f64 own_spike_time_seconds = (f64)own_result.target_spike_ticks[index] * dt_seconds;
        EXPECT_NEAR(own_spike_time_seconds, reference_target_spike_times[index], 1e-3)
            << "spike index " << index;
    }
}

TEST(ExitModelValidation, DISABLED_delayed_coupling_network_source_spike_matches_pyneuroml_reference) {
    const s64 tick_count = 600;
    const f32 dt_seconds = 1e-4f;
    const f64 spike_time_tolerance_seconds = 1e-3;

    DelayedCouplingRunResult own_result = run_delayed_coupling_network(tick_count, dt_seconds);
    Vector<ReferenceSpikeRecord> reference_spikes = load_reference_spikes(
        fixture_path("reference_data/delayed_coupling_network/delayed_coupling_network_spikes.dat"));

    ASSERT_EQ(own_result.source_spike_ticks.size(), reference_spikes.size());
    for (usize index = 0; index < reference_spikes.size(); ++index) {
        f64 own_spike_time_seconds = (f64)own_result.source_spike_ticks[index] * dt_seconds;
        EXPECT_NEAR(own_spike_time_seconds, reference_spikes[index].time_seconds, spike_time_tolerance_seconds)
            << "spike index " << index;
    }
}

TEST(ExitModelValidation, DISABLED_delayed_coupling_network_delay_ring_matches_pyneuroml_observed_arrival) {
    // The ticket #64/#67 core proof: spikecorec's own delay ring (allocate_delay_ring,
    // AssembledModel's enable_delay_ring=true path) delivers at the SAME tick jLEMS's own real,
    // independently-simulated connection delay does -- NOT re-derived from the connection's own
    // `delay` attribute in this test (that would be circular), but read directly off the real
    // captured jLEMS reference trace: the first tick TargetPop's own membrane potential departs from
    // its resting value EL (delayed_coupling_network.nml's own header comment: TargetPop receives NO
    // stimulus of its own, so any departure is attributable exclusively to the one delayed
    // connection).
    const s64 tick_count = 600;
    const f32 dt_seconds = 1e-4f;
    // jLEMS's own discrete-event bookkeeping does not line up tick-for-tick with a naive
    // forward-Euler comparison (a real, small few-tick offset was observed while capturing this
    // fixture's own reference data -- see this ticket's own final report) -- this is a tolerance
    // band, not exact-tick matching, matching this file's own established 1ms spike-time-tolerance
    // convention above.
    const f64 delivery_time_tolerance_seconds = 1e-3;

    DelayedCouplingRunResult own_result = run_delayed_coupling_network(tick_count, dt_seconds);
    ASSERT_FALSE(own_result.target_delivery_ticks.empty());

    ReferenceTrace reference_trace = load_reference_trace(
        fixture_path("reference_data/delayed_coupling_network/delayed_coupling_network_membrane_trace.dat"));
    const f64 el_volts = -0.07;
    const f64 departure_epsilon = 1e-9;
    s64 reference_arrival_row = -1;
    for (usize row_index = 1; row_index < reference_trace.column_values.size(); ++row_index) {
        if (std::fabs(reference_trace.column_values[row_index][1] - el_volts) > departure_epsilon) {
            reference_arrival_row = (s64)row_index;
            break;
        }
    }
    ASSERT_NE(reference_arrival_row, -1) << "reference TargetPop trace never departs from EL";
    // reference_trace row R is tick (R-1)'s own post-integration value (row 0 is the t=0 initial
    // condition, matching ExitModelValidation's own established row/tick offset convention above).
    f64 reference_arrival_time_seconds = (f64)(reference_arrival_row - 1) * (f64)dt_seconds;

    f64 own_first_delivery_time_seconds = (f64)own_result.target_delivery_ticks[0] * dt_seconds;
    EXPECT_NEAR(own_first_delivery_time_seconds, reference_arrival_time_seconds, delivery_time_tolerance_seconds);
}

TEST(ExitModelValidation, DISABLED_poisson_population_aggregate_spike_count_matches_pyneuroml_reference) {
    // See poisson_population.nml's own header comment / this file's own header comment (Phase-2
    // section, #4): an exact, spike-for-spike comparison is not meaningful here (jLEMS's own
    // renewal-process algorithm and spikecorec's own per-tick Bernoulli-style algorithm draw from two
    // entirely different PRNG streams) -- this compares AGGREGATE spike-count statistics instead, the
    // one genuinely meaningful cross-simulator target for a stochastic generator, at this exit
    // model's own larger (100-neuron, 8s) population scale.
    const s32 population_size = 100;
    const f64 rate_hz = 10.0;
    const s64 tick_count = 8000;
    const f32 dt_seconds = 0.001f;

    PoissonPopulationRunResult own_result = run_poisson_population(population_size, tick_count, dt_seconds);
    Vector<ReferenceSpikeRecord> reference_spikes = load_reference_spikes(
        fixture_path("reference_data/poisson_population/poisson_population_spikes.dat"));

    f64 expected_spike_count = (f64)population_size * rate_hz * ((f64)tick_count * (f64)dt_seconds);
    // Both counts are independent samples of the SAME rate-`rate_hz` Poisson process -- compared to
    // EACH OTHER (not just each independently to the analytic expectation, ExitModelPoissonPopulation
    // above already covers that), with a tolerance sized off ordinary Poisson count-variance
    // (std ~= sqrt(expected_spike_count), matching inputs_lowering_tests.cpp's own established
    // rationale).
    f64 tolerance = std::sqrt(expected_spike_count) * 4.0; // ~4 standard deviations
    EXPECT_NEAR((f64)own_result.total_spike_count, (f64)reference_spikes.size(), tolerance)
        << "own=" << own_result.total_spike_count << " reference=" << reference_spikes.size();
}
