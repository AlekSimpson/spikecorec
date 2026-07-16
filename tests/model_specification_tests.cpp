#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <gtest/gtest.h>

#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/model_specification.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

namespace {

String write_temp_file(const String &filename, const String &contents) {
    String path = (std::filesystem::temp_directory_path() / filename).string();
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

const TypeLibraryEntry &type_library_entry_for(const ModelSpecification &specification, const String &bound_instance_id) {
    for (const auto &entry : specification.type_library) {
        if (entry.bound_instance_id == bound_instance_id) return entry;
    }
    throw std::runtime_error("no type library entry for '" + bound_instance_id + "'");
}

const PopulationEntry &population_named(const ModelSpecification &specification, const String &id) {
    for (const auto &population : specification.populations) {
        if (population.id == id) return population;
    }
    throw std::runtime_error("no population '" + id + "'");
}

const ProjectionEntry &projection_named(const ModelSpecification &specification, const String &id) {
    for (const auto &projection : specification.projections) {
        if (projection.id == id) return projection;
    }
    throw std::runtime_error("no projection '" + id + "'");
}

// A Phase-1 GLIF excitatory/inhibitory network fixture exercising every
// ModelSpecification deliverable (arch §1.4) in one model:
//   - two GLIF1Cell populations (ExcPop size 3, InhPop size 2) bound to
//     DIFFERENT component instances (distinct EL/gL) -- bake-vs-parameterize
//     evidence (§3.1 Parameter): same cell ComponentType, different baked
//     values per population.
//   - three synapse types covering the full §3.3 D3 classification split:
//     expOneSynapse (conductance-based, aggregatable), alphaCurrentSynapse
//     (current-based, aggregatable), blockingPlasticSynapse/NMDA
//     (conductance-based, per-edge/non-aggregatable).
//   - a pulseGenerator driven by an explicitInput (stimulus spec).
//   - an OutputColumn (state recording) and an EventSelection (spike
//     recording).
// Routed entirely through an <include> (like tickets #2/#49's own fixtures):
// Simulation/OutputFile/OutputColumn/EventOutputFile/EventSelection are pure
// LEMS elements, not part of the NeuroML2 XSD, so only the top-level file
// (a bare <include>) is XSD-gated.
String write_model_specification_fixture() {
    write_temp_file("spikecorec_model_spec_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"ModelSpecContent\">"
        "  <ComponentType name=\"GLIF1Cell\" extends=\"baseCell\">"
        "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
        "    <Parameter name=\"gL\" dimension=\"conductance\"/>"
        "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
        "    <Parameter name=\"vth\" dimension=\"voltage\"/>"
        "    <Parameter name=\"vreset\" dimension=\"voltage\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
        "      <TimeDerivative variable=\"v\" value=\"(gL * (EL - v) + iSyn) / C\"/>"
        "      <OnStart>"
        "        <StateAssignment variable=\"v\" value=\"EL\"/>"
        "      </OnStart>"
        "      <OnCondition test=\"v .gt. vth\">"
        "        <EventOut port=\"spike\"/>"
        "        <StateAssignment variable=\"v\" value=\"vreset\"/>"
        "      </OnCondition>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <GLIF1Cell id=\"glif1ExcCellInstance\" C=\"1.0e-10\" gL=\"1.0e-8\" EL=\"-70mV\" vth=\"-50mV\" vreset=\"-70mV\"/>"
        "  <GLIF1Cell id=\"glif1InhCellInstance\" C=\"1.0e-10\" gL=\"1.2e-8\" EL=\"-65mV\" vth=\"-50mV\" vreset=\"-65mV\"/>"
        "  <expOneSynapse id=\"glifExcSynapse\" gbase=\"1nS\" erev=\"0mV\" tauDecay=\"3ms\"/>"
        "  <alphaCurrentSynapse id=\"glifCurrSynapse\" tau=\"2ms\" ibase=\"0.1nA\"/>"
        "  <blockingPlasticSynapse id=\"glifNmdaSynapse\" gbase=\"1nS\" erev=\"0mV\" tauRise=\"1ms\" tauDecay=\"50ms\"/>"
        "  <pulseGenerator id=\"pulseGen1\" delay=\"10ms\" duration=\"100ms\" amplitude=\"1nA\"/>"
        "  <network id=\"GLIFExcInhNet\">"
        "    <population id=\"ExcPop\" component=\"glif1ExcCellInstance\" size=\"3\"/>"
        "    <population id=\"InhPop\" component=\"glif1InhCellInstance\" size=\"2\"/>"
        "    <projection id=\"ExcToInh\" presynapticPopulation=\"ExcPop\" postsynapticPopulation=\"InhPop\" synapse=\"glifExcSynapse\">"
        "      <connectionWD id=\"0\" preCellId=\"../ExcPop/0/glif1ExcCellInstance\" postCellId=\"../InhPop/0/glif1InhCellInstance\" weight=\"1.5\" delay=\"2ms\"/>"
        "      <connection id=\"1\" preCellId=\"../ExcPop/1/glif1ExcCellInstance\" postCellId=\"../InhPop/1/glif1InhCellInstance\"/>"
        "    </projection>"
        "    <projection id=\"ExcRecurrent\" presynapticPopulation=\"ExcPop\" postsynapticPopulation=\"ExcPop\" synapse=\"glifCurrSynapse\">"
        "      <connection id=\"0\" preCellId=\"../ExcPop/0/glif1ExcCellInstance\" postCellId=\"../ExcPop/2/glif1ExcCellInstance\"/>"
        "    </projection>"
        "    <projection id=\"ExcToInhNmda\" presynapticPopulation=\"ExcPop\" postsynapticPopulation=\"InhPop\" synapse=\"glifNmdaSynapse\">"
        "      <connection id=\"0\" preCellId=\"../ExcPop/2/glif1ExcCellInstance\" postCellId=\"../InhPop/0/glif1InhCellInstance\"/>"
        "    </projection>"
        "    <explicitInput target=\"ExcPop[0]\" input=\"pulseGen1\"/>"
        "  </network>"
        "  <Simulation id=\"sim1\" length=\"1000ms\" step=\"0.1ms\" target=\"GLIFExcInhNet\">"
        "    <OutputFile id=\"of0\" fileName=\"results.dat\">"
        "      <OutputColumn id=\"v_exc0\" quantity=\"ExcPop[0]/v\"/>"
        "    </OutputFile>"
        "    <EventOutputFile id=\"spikesFile\" fileName=\"spikes.dat\" format=\"TIME_ID\">"
        "      <EventSelection id=\"sel0\" select=\"InhPop[1]\" eventPort=\"spike\"/>"
        "    </EventOutputFile>"
        "  </Simulation>"
        "</neuroml>");

    return write_temp_file("spikecorec_model_spec_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"ModelSpecTop\">"
        "  <include href=\"spikecorec_model_spec_content.nml\"/>"
        "</neuroml>");
}

ModelSpecification build_test_model_specification() {
    String path = write_model_specification_fixture();
    NML_Parser parser;
    parser.parse(path);
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

// A minimal single-population (size 3) network with one recurrent
// projection, parameterized by the connection's own preCellId/postCellId --
// used to exercise a malformed or out-of-range connection index without
// repeating the full fixture above.
String write_single_population_connection_fixture(const String &pre_cell_id, const String &post_cell_id) {
    write_temp_file("spikecorec_model_spec_bad_connection_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"BadConnectionContent\">"
        "  <ComponentType name=\"GLIF1Cell\" extends=\"baseCell\">"
        "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
        "    <Parameter name=\"gL\" dimension=\"conductance\"/>"
        "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
        "      <TimeDerivative variable=\"v\" value=\"(gL * (EL - v)) / C\"/>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <GLIF1Cell id=\"glifCellInstance\" C=\"1.0e-10\" gL=\"1.0e-8\" EL=\"-70mV\"/>"
        "  <alphaCurrentSynapse id=\"glifCurrSynapse\" tau=\"2ms\" ibase=\"0.1nA\"/>"
        "  <network id=\"BadConnectionNet\">"
        "    <population id=\"ExcPop\" component=\"glifCellInstance\" size=\"3\"/>"
        "    <projection id=\"ExcRecurrent\" presynapticPopulation=\"ExcPop\" postsynapticPopulation=\"ExcPop\" synapse=\"glifCurrSynapse\">"
        "      <connection id=\"0\" preCellId=\"" + pre_cell_id + "\" postCellId=\"" + post_cell_id + "\"/>"
        "    </projection>"
        "  </network>"
        "</neuroml>");

    return write_temp_file("spikecorec_model_spec_bad_connection_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"BadConnectionTop\">"
        "  <include href=\"spikecorec_model_spec_bad_connection_content.nml\"/>"
        "</neuroml>");
}

// A minimal single-population (size 3) network with an OutputColumn whose
// `quantity` path is parameterized -- used to exercise a malformed
// recording path without repeating the full fixture above.
String write_single_population_recording_fixture(const String &quantity_path) {
    write_temp_file("spikecorec_model_spec_bad_quantity_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"BadQuantityContent\">"
        "  <ComponentType name=\"GLIF1Cell\" extends=\"baseCell\">"
        "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <GLIF1Cell id=\"glifCellInstance\" C=\"1.0e-10\"/>"
        "  <network id=\"BadQuantityNet\">"
        "    <population id=\"ExcPop\" component=\"glifCellInstance\" size=\"3\"/>"
        "  </network>"
        "  <Simulation id=\"sim1\" length=\"1000ms\" step=\"0.1ms\" target=\"BadQuantityNet\">"
        "    <OutputFile id=\"of0\" fileName=\"results.dat\">"
        "      <OutputColumn id=\"badCol\" quantity=\"" + quantity_path + "\"/>"
        "    </OutputFile>"
        "  </Simulation>"
        "</neuroml>");

    return write_temp_file("spikecorec_model_spec_bad_quantity_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"BadQuantityTop\">"
        "  <include href=\"spikecorec_model_spec_bad_quantity_content.nml\"/>"
        "</neuroml>");
}

} // namespace

// ── Type library: classification flags (acceptance criterion) ─────────────

TEST(ModelSpecification, classifies_current_based_synapse) {
    ModelSpecification specification = build_test_model_specification();

    const TypeLibraryEntry &alpha_current = type_library_entry_for(specification, "glifCurrSynapse");
    EXPECT_EQ(alpha_current.component_type_name, "alphaCurrentSynapse");
    EXPECT_EQ(alpha_current.category, TypeLibraryCategory::Synapse);
    EXPECT_FALSE(alpha_current.is_conductance_based);
}

TEST(ModelSpecification, classifies_conductance_based_synapse) {
    ModelSpecification specification = build_test_model_specification();

    const TypeLibraryEntry &exp_one = type_library_entry_for(specification, "glifExcSynapse");
    EXPECT_EQ(exp_one.component_type_name, "expOneSynapse");
    EXPECT_EQ(exp_one.category, TypeLibraryCategory::Synapse);
    EXPECT_TRUE(exp_one.is_conductance_based);
}

TEST(ModelSpecification, classifies_conductance_based_nmda_synapse) {
    ModelSpecification specification = build_test_model_specification();

    const TypeLibraryEntry &nmda = type_library_entry_for(specification, "glifNmdaSynapse");
    EXPECT_EQ(nmda.component_type_name, "blockingPlasticSynapse");
    EXPECT_EQ(nmda.category, TypeLibraryCategory::Synapse);
    EXPECT_TRUE(nmda.is_conductance_based);   // extends expTwoSynapse -> baseConductanceBasedSynapse
}

TEST(ModelSpecification, classifies_inputs_type_entry) {
    ModelSpecification specification = build_test_model_specification();

    const TypeLibraryEntry &pulse = type_library_entry_for(specification, "pulseGen1");
    EXPECT_EQ(pulse.component_type_name, "pulseGenerator");
    EXPECT_EQ(pulse.category, TypeLibraryCategory::Inputs);
}

// ── Type library: bake-vs-parameterize (Phase 1 default: bake per bound instance) ──

TEST(ModelSpecification, bakes_distinct_parameter_values_per_population_bound_component) {
    ModelSpecification specification = build_test_model_specification();

    const TypeLibraryEntry &exc_cell = type_library_entry_for(specification, "glif1ExcCellInstance");
    const TypeLibraryEntry &inh_cell = type_library_entry_for(specification, "glif1InhCellInstance");

    EXPECT_EQ(exc_cell.component_type_name, "GLIF1Cell");
    EXPECT_EQ(inh_cell.component_type_name, "GLIF1Cell");

    ASSERT_EQ(exc_cell.baked_constants.count("EL"), 1u);
    ASSERT_EQ(inh_cell.baked_constants.count("EL"), 1u);
    EXPECT_NEAR(exc_cell.baked_constants.at("EL"), -0.070, 1e-9);
    EXPECT_NEAR(inh_cell.baked_constants.at("EL"), -0.065, 1e-9);
    EXPECT_NE(exc_cell.baked_constants.at("EL"), inh_cell.baked_constants.at("EL"));

    ASSERT_EQ(exc_cell.baked_constants.count("gL"), 1u);
    EXPECT_NEAR(exc_cell.baked_constants.at("gL"), 1.0e-8, 1e-15);

    // `v` (a StateVariable, not a Parameter) is never baked -- one state slot
    // per instance (arch §3.1 Parameter: "a StateVariable ... is always
    // per-neuron, never baked").
    EXPECT_EQ(exc_cell.state_variable_count, 1);
}

// ── Populations: index/size/range/offset (acceptance criterion: multi-population network) ──

TEST(ModelSpecification, builds_population_table_with_index_range_size_and_chunk_offset) {
    ModelSpecification specification = build_test_model_specification();

    ASSERT_EQ(specification.populations.size(), 2u);
    const PopulationEntry &exc_pop = population_named(specification, "ExcPop");
    const PopulationEntry &inh_pop = population_named(specification, "InhPop");

    EXPECT_EQ(exc_pop.size, 3);
    EXPECT_EQ(exc_pop.neuron_index_begin, 0);
    EXPECT_EQ(exc_pop.neuron_index_end, 3);
    EXPECT_EQ(exc_pop.state_vector_chunk_base_offset, 0);

    EXPECT_EQ(inh_pop.size, 2);
    EXPECT_EQ(inh_pop.neuron_index_begin, 3);
    EXPECT_EQ(inh_pop.neuron_index_end, 5);
    // ExcPop's chunk is 3 neurons * 1 state variable (GLIF1Cell's `v`) wide.
    EXPECT_EQ(inh_pop.state_vector_chunk_base_offset, 3);

    EXPECT_EQ(specification.total_neuron_count, 5);

    EXPECT_NE(exc_pop.type_library_index, inh_pop.type_library_index);
    EXPECT_EQ(specification.type_library[exc_pop.type_library_index].bound_instance_id, "glif1ExcCellInstance");
    EXPECT_EQ(specification.type_library[inh_pop.type_library_index].bound_instance_id, "glif1InhCellInstance");
}

// ── Adjacency + initial weights (acceptance criterion: populates a WeightMatrix) ──

TEST(ModelSpecification, builds_adjacency_weight_matrix_from_projection_connections) {
    ModelSpecification specification = build_test_model_specification();

    ASSERT_TRUE(specification.adjacency.has_value());
    EXPECT_EQ(specification.adjacency->node_count, specification.total_neuron_count);

    // Edges: ExcPop[0]->InhPop[0] (0->3), ExcPop[1]->InhPop[1] (1->4),
    // ExcPop[0]->ExcPop[2] (0->2), ExcPop[2]->InhPop[0] (2->3).
    vector<s32> buffer((usize)specification.adjacency->max_neighbor_count);

    s64 neighbor_count_0 = specification.adjacency->get_neighbors(0, buffer.data());
    unordered_set<s32> neighbors_0(buffer.begin(), buffer.begin() + neighbor_count_0);
    EXPECT_EQ(neighbors_0, (unordered_set<s32>{3, 2}));

    s64 neighbor_count_2 = specification.adjacency->get_neighbors(2, buffer.data());
    unordered_set<s32> neighbors_2(buffer.begin(), buffer.begin() + neighbor_count_2);
    EXPECT_EQ(neighbors_2, (unordered_set<s32>{3}));
}

// ── Projection routing + delay data (acceptance criterion) ─────────────────

TEST(ModelSpecification, captures_projection_routing_and_per_connection_weight_delay) {
    ModelSpecification specification = build_test_model_specification();

    ASSERT_EQ(specification.projections.size(), 3u);
    const ProjectionEntry &exc_to_inh = projection_named(specification, "ExcToInh");

    EXPECT_EQ(specification.populations[exc_to_inh.presynaptic_population_index].id, "ExcPop");
    EXPECT_EQ(specification.populations[exc_to_inh.postsynaptic_population_index].id, "InhPop");
    EXPECT_EQ(specification.type_library[exc_to_inh.synapse_type_library_index].bound_instance_id, "glifExcSynapse");

    ASSERT_EQ(exc_to_inh.connections.size(), 2u);
    const ConnectionEntry &weighted_delayed = exc_to_inh.connections[0];
    EXPECT_EQ(weighted_delayed.source_neuron_index, 0);
    EXPECT_EQ(weighted_delayed.target_neuron_index, 3);
    EXPECT_NEAR(weighted_delayed.weight, 1.5, 1e-12);
    EXPECT_NEAR(weighted_delayed.delay, 0.002, 1e-12);

    // A plain `connection` (no weight/delay attributes) defaults to weight=1, delay=0.
    const ConnectionEntry &defaulted = exc_to_inh.connections[1];
    EXPECT_EQ(defaulted.source_neuron_index, 1);
    EXPECT_EQ(defaulted.target_neuron_index, 4);
    EXPECT_NEAR(defaulted.weight, 1.0, 1e-12);
    EXPECT_NEAR(defaulted.delay, 0.0, 1e-12);
}

// ── Input (stimulus) and output (recording) specs (acceptance criterion) ───

TEST(ModelSpecification, captures_stimulus_and_recording_specs) {
    ModelSpecification specification = build_test_model_specification();

    ASSERT_EQ(specification.stimuli.size(), 1u);
    const StimulusEntry &stimulus = specification.stimuli[0];
    EXPECT_EQ(stimulus.target_neuron_index, 0); // ExcPop[0]
    EXPECT_EQ(specification.type_library[stimulus.input_type_library_index].bound_instance_id, "pulseGen1");

    ASSERT_EQ(specification.recordings.size(), 2u);
    const RecordingEntry *state_recording = nullptr;
    const RecordingEntry *event_recording = nullptr;
    for (const auto &recording : specification.recordings) {
        if (recording.is_event_recording) event_recording = &recording;
        else state_recording = &recording;
    }
    ASSERT_NE(state_recording, nullptr);
    ASSERT_NE(event_recording, nullptr);

    EXPECT_EQ(state_recording->quantity_path, "ExcPop[0]/v");
    EXPECT_EQ(state_recording->target_neuron_index, 0);
    EXPECT_EQ(state_recording->exposure_name, "v");

    EXPECT_EQ(event_recording->target_neuron_index, 4); // InhPop[1]
    EXPECT_EQ(event_recording->exposure_name, "spike");
}

// ── Regression: path-index safety (review findings #1-#3) ──────────────────

// Finding #1 (memory safety): an out-of-range connection index must throw
// instead of writing out-of-bounds into the adjacency list. ExcPop has size
// 3 (valid local indices 0-2); index 9 is in-range for the population NAME
// but out-of-range for its actual size.
TEST(ModelSpecification, throws_instead_of_out_of_bounds_write_on_out_of_range_connection_index) {
    String path = write_single_population_connection_fixture(
        "../ExcPop/0/glifCellInstance", "../ExcPop/9/glifCellInstance");

    NML_Parser parser;
    parser.parse(path);
    ResolvedModel resolved = resolve_and_lower(parser);

    EXPECT_THROW(build_model_specification(resolved), std::runtime_error);
}

// Finding #2: a malformed (non-numeric) index segment must throw instead of
// silently defaulting to 0.
TEST(ModelSpecification, throws_instead_of_defaulting_to_zero_on_malformed_connection_index) {
    String path = write_single_population_connection_fixture(
        "../ExcPop/0/glifCellInstance", "../ExcPop/abc/glifCellInstance");

    NML_Parser parser;
    parser.parse(path);
    ResolvedModel resolved = resolve_and_lower(parser);

    EXPECT_THROW(build_model_specification(resolved), std::runtime_error);
}

// Finding #3: a slash-form path whose segment after the population name is
// not numeric ("ExcPop/v", no index at all) must be rejected as malformed
// input rather than silently misparsed as index 0 with the "v" exposure
// dropped.
TEST(ModelSpecification, throws_on_slash_form_path_with_non_numeric_trailing_token) {
    String path = write_single_population_recording_fixture("ExcPop/v");

    NML_Parser parser;
    parser.parse(path);
    ResolvedModel resolved = resolve_and_lower(parser);

    EXPECT_THROW(build_model_specification(resolved), std::runtime_error);
}
