#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/plasticity_wiring.h"
#include "spikecorec/core/engine.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── NML STDP -> stage-7 plasticity wiring tests (ticket #66 [F5]) ───────────────────────────────
//
// Two things this file demonstrates, matching the ticket's own acceptance criterion ("An STDP spec
// drives the weight-update path"):
//  1. `find_stdp_spec`/`apply_stdp_wiring` correctly detect an STDP-shaped synapse from a REAL
//     parsed+resolved+lowered NML model (not a hand-built TypeLibraryEntry) and drive
//     SpikeEngine::enable_plasticity/disable_plasticity accordingly -- see
//     plasticity_wiring.h's own doc comment for why `TestStdpSynapse` below is a stand-in for the
//     vendored `stdpSynapse` (which is explicitly marked "NOT YET WORKING" in
//     third_party/neuroml2/std_lib/Synapses.xml and declares no tauPlus/tauMinus/aPlus/aMinus at
//     all), matching this codebase's own established precedent for a real-but-gapped std-lib type
//     (synapse_lowering_tests.cpp's `TestExpTwoSynapse`).
//  2. The mapped learning rate, driven through a REAL SpikeEngine simulation with controlled
//     pre/post spike timing (mirroring engine_tests.cpp's own `SpikeEngine.plasticity_end_to_end`),
//     changes the weight in the direction/shape the ACTUAL kernel code implements (read, not
//     assumed -- see plasticity_wiring.h's doc comment): a one-signed depression, larger in
//     magnitude the more recently the postsynaptic neuron fired.

namespace {

bool approx(f32 first, f32 second, f32 epsilon = 1e-4f) {
    return std::fabs(first - second) <= epsilon * (1.0f + std::fabs(second));
}

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

const String DUMMY_CELL_COMPONENT_TYPE =
    "  <ComponentType name=\"DummyCell\" extends=\"baseCell\">"
    "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
    "      <TimeDerivative variable=\"v\" value=\"network_inputs / C\"/>"
    "    </Dynamics>"
    "  </ComponentType>";

// A stand-in for the vendored (but non-functional, see this file's own header comment) NeuroML
// `stdpSynapse`: a real, self-contained ComponentType baking the standard STDP
// tauPlus/tauMinus/aPlus/aMinus parameter names.
const String TEST_STDP_SYNAPSE_COMPONENT_TYPE =
    "  <ComponentType name=\"TestStdpSynapse\" extends=\"baseConductanceBasedSynapse\">"
    "    <Property name=\"weight\" dimension=\"none\" defaultValue=\"1\"/>"
    "    <Parameter name=\"tau\" dimension=\"time\"/>"
    "    <Parameter name=\"tauPlus\" dimension=\"time\"/>"
    "    <Parameter name=\"tauMinus\" dimension=\"time\"/>"
    "    <Parameter name=\"aPlus\" dimension=\"none\"/>"
    "    <Parameter name=\"aMinus\" dimension=\"none\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"g\" dimension=\"conductance\" exposure=\"g\"/>"
    "      <DerivedVariable name=\"i\" exposure=\"i\" dimension=\"current\" value=\"g * (erev - v)\"/>"
    "      <TimeDerivative variable=\"g\" value=\"-g / tau\"/>"
    "      <OnStart>"
    "        <StateAssignment variable=\"g\" value=\"0\"/>"
    "      </OnStart>"
    "      <OnEvent port=\"in\">"
    "        <StateAssignment variable=\"g\" value=\"g + weight\"/>"
    "      </OnEvent>"
    "    </Dynamics>"
    "  </ComponentType>";

// Wraps one synapse ComponentType (real std-lib tag or custom inline declaration) plus one bound
// instance into a minimal two-neuron network, runs it through parse -> resolve_and_lower ->
// build_model_specification, and returns the resulting ModelSpecification -- mirrors
// synapse_lowering_tests.cpp's own build_synapse_type_library_entry helper (kept as its own copy
// here, matching how every *_tests.cpp file in this tree keeps its own fixture code self-contained).
ModelSpecification build_model_with_one_synapse(
    const String &fixture_id, const String &synapse_component_type_xml,
    const String &synapse_tag, const String &synapse_instance_attributes
) {
    write_temp_file("spikecorec_plasticity_wiring_" + fixture_id + "_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"PlasticityWiring" + fixture_id + "Content\">"
        + DUMMY_CELL_COMPONENT_TYPE +
        "  <DummyCell id=\"dummyCellInstance\" C=\"1.0e-10\"/>"
        + synapse_component_type_xml +
        "  <" + synapse_tag + " id=\"synapseInstance\" " + synapse_instance_attributes + "/>"
        "  <network id=\"Net\">"
        "    <population id=\"Pop\" component=\"dummyCellInstance\" size=\"2\"/>"
        "    <projection id=\"Proj\" presynapticPopulation=\"Pop\" postsynapticPopulation=\"Pop\" synapse=\"synapseInstance\">"
        "      <connection id=\"0\" preCellId=\"Pop/0/dummyCellInstance\" postCellId=\"Pop/1/dummyCellInstance\"/>"
        "    </projection>"
        "  </network>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_plasticity_wiring_" + fixture_id + "_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"PlasticityWiring" + fixture_id + "Top\">"
        "  <include href=\"spikecorec_plasticity_wiring_" + fixture_id + "_content.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

} // namespace

// ── find_stdp_spec ────────────────────────────────────────────────────────────────────────────

TEST(PlasticityWiring, finds_stdp_spec_from_a_real_parsed_and_resolved_nml_model) {
    ModelSpecification model = build_model_with_one_synapse(
        "finds_spec", TEST_STDP_SYNAPSE_COMPONENT_TYPE, "TestStdpSynapse",
        "gbase=\"1nS\" erev=\"0mV\" tau=\"5ms\" tauPlus=\"20ms\" tauMinus=\"20ms\" aPlus=\"0.01\" aMinus=\"0.012\"");
    const TypeLibraryEntry &entry = type_library_entry_for(model, "synapseInstance");

    std::optional<StdpSpec> spec = find_stdp_spec(entry);
    ASSERT_TRUE(spec.has_value());
    EXPECT_TRUE(approx(spec->tau_plus, 0.020f));
    EXPECT_TRUE(approx(spec->tau_minus, 0.020f));
    EXPECT_TRUE(approx(spec->a_plus, 0.01f));
    EXPECT_TRUE(approx(spec->a_minus, 0.012f));

    EXPECT_TRUE(approx(map_stdp_spec_to_learning_rate(*spec), 0.012f));
}

TEST(PlasticityWiring, returns_nullopt_when_synapse_has_no_stdp_parameters) {
    ModelSpecification model = build_model_with_one_synapse(
        "no_spec", "", "expOneSynapse", "gbase=\"1nS\" erev=\"0mV\" tauDecay=\"3ms\"");
    const TypeLibraryEntry &entry = type_library_entry_for(model, "synapseInstance");

    EXPECT_FALSE(find_stdp_spec(entry).has_value());
}

// ── apply_stdp_wiring ─────────────────────────────────────────────────────────────────────────

TEST(PlasticityWiring, apply_stdp_wiring_enables_plasticity_with_the_mapped_rate_when_stdp_spec_present) {
    ModelSpecification model = build_model_with_one_synapse(
        "enable", TEST_STDP_SYNAPSE_COMPONENT_TYPE, "TestStdpSynapse",
        "gbase=\"1nS\" erev=\"0mV\" tau=\"5ms\" tauPlus=\"20ms\" tauMinus=\"20ms\" aPlus=\"0.01\" aMinus=\"0.03\"");

    vector<vector<s32>> network = {{1}, {}};
    SpikeEngine engine(&network, {2, 1}, /*rank=*/4, /*resting_mp=*/0.1f, /*decay_rate=*/0.01f,
                        /*learning_rate=*/0.0f, /*plasticity_enabled=*/true, /*active_set_optimization_enabled=*/true);
    ASSERT_FALSE(engine.plasticity_enabled());

    apply_stdp_wiring(model, engine);

    EXPECT_TRUE(engine.plasticity_enabled());
    EXPECT_TRUE(approx(engine.learning_rate, 0.03f));

    engine.shutdown();
}

TEST(PlasticityWiring, apply_stdp_wiring_disables_plasticity_when_no_stdp_synapse_present) {
    ModelSpecification model = build_model_with_one_synapse(
        "disable", "", "expOneSynapse", "gbase=\"1nS\" erev=\"0mV\" tauDecay=\"3ms\"");

    vector<vector<s32>> network = {{1}, {}};
    SpikeEngine engine(&network, {2, 1}, /*rank=*/4, /*resting_mp=*/0.1f, /*decay_rate=*/0.01f,
                        /*learning_rate=*/0.0f, /*plasticity_enabled=*/true, /*active_set_optimization_enabled=*/true);
    engine.learning_rate = 0.25f; // simulate an engine constructed with plasticity already live
    ASSERT_TRUE(engine.plasticity_enabled());

    apply_stdp_wiring(model, engine);

    EXPECT_FALSE(engine.plasticity_enabled());
    EXPECT_TRUE(approx(engine.learning_rate, 0.0f));

    engine.shutdown();
}

// ── real-simulation STDP direction/shape (acceptance criterion 1's "drive it through a real
// simulation") ───────────────────────────────────────────────────────────────────────────────────
//
// Mirrors engine_tests.cpp's own SpikeEngine.plasticity_end_to_end fixture: a 2-neuron network
// (0 -> 1), neuron 0 driven to spike at a controlled tick, neuron 1's own `last_spiked` seeded to a
// controlled EARLIER tick -- i.e. postsynaptic-before-presynaptic timing, the only case the real
// kernel's `step_apply_hebbian_update` call site ever nudges the weight for (see
// plasticity_wiring.h's doc comment: it is skipped entirely when the postsynaptic neuron has never
// spiked or spiked THIS tick). Two different gaps between the postsynaptic spike and the
// presynaptic spike are compared to confirm the magnitude scales the way `pow(tick_delta, -3)`
// predicts (larger for a smaller gap), not just that direction is right.
TEST(PlasticityWiringDirection, stdp_direction_matches_kernel_sign_convention) {
    auto run_edge_update = [](f32 learning_rate, s64 postsynaptic_last_spiked_tick, s64 presynaptic_spike_tick) -> pair<f32, f32> {
        vector<vector<s32>> network = {{1}, {}};
        SpikeEngine engine(&network, {2, 1}, /*rank=*/8, /*resting_mp=*/0.1f, /*decay_rate=*/0.01f,
                            /*learning_rate=*/0.0f, /*plasticity_enabled=*/true,
                            /*active_set_optimization_enabled=*/true);
        engine.learning_rate = learning_rate;
        engine.use_constant_weight = false;
        engine.set_input_neurons({0});

        s64 *last_spiked = engine.last_spiked.get_contents();
        last_spiked[0] = -1000; // neuron 0 (presynaptic) never spiked before
        last_spiked[1] = postsynaptic_last_spiked_tick;

        f32 before = engine.weights.get(0, 1);
        engine.step_simulation({5.0f}, presynaptic_spike_tick, /*override_input_neurons=*/{0});
        f32 after = engine.weights.get(0, 1);

        engine.shutdown();
        return {before, after};
    };

    StdpSpec spec{/*tau_plus=*/0.02f, /*tau_minus=*/0.02f, /*a_plus=*/0.01f, /*a_minus=*/0.02f};
    f32 mapped_learning_rate = map_stdp_spec_to_learning_rate(spec);

    // Postsynaptic neuron spiked at tick 2, presynaptic spikes (and propagates) at tick 3 --
    // tick_delta = 1 (the smallest possible non-zero gap).
    auto [before_close, after_close] = run_edge_update(mapped_learning_rate, /*postsynaptic_last_spiked_tick=*/2,
                                                        /*presynaptic_spike_tick=*/3);
    // Postsynaptic neuron spiked at tick 2, presynaptic spikes at tick 8 -- tick_delta = 6.
    auto [before_far, after_far] = run_edge_update(mapped_learning_rate, /*postsynaptic_last_spiked_tick=*/2,
                                                    /*presynaptic_spike_tick=*/8);

    // Direction: reading kernels.metal's step_apply_hebbian_update call site, decay_delta =
    // -learning_rate * pow(tick_delta, -3) is always <= 0 for tick_delta > 0 -- i.e. the weight can
    // only ever be nudged DOWN by this path, never up. Confirmed here with real numbers rather than
    // assumed.
    EXPECT_LT(after_close, before_close);
    EXPECT_LT(after_far, before_far);

    // Recency scaling: pow(tick_delta, -3) is larger for a SMALLER tick_delta, so the tick_delta=1
    // case should show a strictly larger-magnitude depression than the tick_delta=6 case.
    f32 magnitude_close = before_close - after_close;
    f32 magnitude_far = before_far - after_far;
    EXPECT_GT(magnitude_close, magnitude_far);

    // learning_rate == 0 (no STDP spec / plasticity disabled) must leave the weight untouched.
    auto [frozen_before, frozen_after] = run_edge_update(/*learning_rate=*/0.0f, /*postsynaptic_last_spiked_tick=*/2,
                                                          /*presynaptic_spike_tick=*/3);
    EXPECT_TRUE(approx(frozen_before, frozen_after, 1e-6f));
}
