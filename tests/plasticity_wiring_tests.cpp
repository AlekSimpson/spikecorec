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

// ── Real bidirectional STDP on the LEGACY engine's GPU kernel path ────────────────────────────────
//
// A genuinely driven simulation (no hand-computed weight deltas — every weight change comes from the
// real step_no_active_optimization GPU kernel) of a 2-neuron network 0 -> 1, run over many repeated
// spike-pair trials with a controlled phase relationship:
//   - CAUSAL (neuron 0 fires one tick BEFORE neuron 1, repeatedly): when neuron 1 fires it walks its
//     PREDECESSORS (the new k2t_next_predecessor traversal), finds neuron 0 fired one tick earlier,
//     and POTENTIATES the 0 -> 1 edge (+learning_rate_plus * pow(1, -3) per pairing). A large
//     inter-trial gap makes the cross-trial anti-causal depression (pow(gap-1, -3), tiny) negligible,
//     so the 0 -> 1 edge must show NET POTENTIATION over the run.
//   - ANTI-CAUSAL (neuron 1 fires one tick before neuron 0, repeatedly): when neuron 0 fires it walks
//     its forward neighbors, finds neuron 1 fired one tick earlier, and DEPRESSES the 0 -> 1 edge
//     (-learning_rate * pow(1, -3) per pairing) — the SAME depression path that shipped before this
//     change, run here with learning_rate_plus = 0 so it is byte-for-byte the pre-change behavior.
//
// Active-set optimization is disabled so every neuron is processed every tick (the
// step_no_active_optimization kernel), which makes firing a clean, deterministic function of the
// injected current and removes active-set carryover from the picture. A small constant scatter weight
// (0.01, well below threshold) means a propagated spike never spuriously fires a neuron — every fire
// is driven by the explicit external pulse, at exactly the intended tick.
namespace {
struct BidirectionalRunResult {
    f32 weight_before = 0.0f;
    f32 weight_after = 0.0f;
    s64 neuron0_fire_count = 0;
    s64 neuron1_fire_count = 0;
};

// Runs the repeated spike-pair protocol described above and returns the 0 -> 1 edge weight before and
// after. `causal == true` fires neuron 0 then neuron 1 each trial; `false` fires neuron 1 then neuron 0.
BidirectionalRunResult run_legacy_bidirectional_pairing(bool causal, f32 learning_rate, f32 learning_rate_plus,
                                                        s32 trial_count, s64 trial_period) {
    vector<vector<s32>> network = {{1}, {}}; // edge 0 -> 1
    SpikeEngine engine(&network, {2, 1}, /*rank=*/8, /*resting_mp=*/0.1f, /*decay_rate=*/0.01f,
                       /*learning_rate=*/0.0f, /*plasticity_enabled=*/true,
                       /*active_set_optimization_enabled=*/false);
    engine.learning_rate = learning_rate;
    engine.learning_rate_plus = learning_rate_plus;
    engine.use_constant_weight = true;
    engine.weights.set_constant_weight(0.01f); // tiny scatter — never spuriously fires a neuron
    engine.set_input_neurons({0, 1});
    engine.reset_state(/*last_spiked_value=*/0, /*active_gen_value=*/-1); // 0 = legacy "never fired"

    const f32 firing_pulse = 5.0f;
    const s64 base_tick = 2; // avoid tick 1, where a last_spiked==0 neuron hits the spike-period reset

    BidirectionalRunResult result;
    result.weight_before = engine.weights.get(0, 1);

    s64 last_tick = base_tick + (s64)(trial_count - 1) * trial_period + 1;
    for (s64 tick = 0; tick <= last_tick; ++tick) {
        // Which neuron (if any) is driven to fire this tick, for the current phase relationship.
        s32 fire_neuron = -1;
        for (s32 trial = 0; trial < trial_count; ++trial) {
            s64 first_tick = base_tick + (s64)trial * trial_period;
            if (tick == first_tick) fire_neuron = causal ? 0 : 1;
            else if (tick == first_tick + 1) fire_neuron = causal ? 1 : 0;
        }

        vector<f32> input = {0.0f, 0.0f};
        if (fire_neuron == 0) input[0] = firing_pulse;
        else if (fire_neuron == 1) input[1] = firing_pulse;

        engine.step_simulation(input, tick);

        // tick > 0 guard: last_spiked is seeded to 0 (the legacy kernel's "never fired" sentinel),
        // which coincides with tick 0 — so a bare `last_spiked == tick` would false-positive there.
        // No neuron is ever driven to fire at tick 0/1, so real fires are all at tick >= 2.
        if (tick > 0 && engine.last_spiked.get_contents()[0] == tick) ++result.neuron0_fire_count;
        if (tick > 0 && engine.last_spiked.get_contents()[1] == tick) ++result.neuron1_fire_count;
    }

    result.weight_after = engine.weights.get(0, 1);
    engine.shutdown();
    return result;
}
} // namespace

TEST(BidirectionalStdpLegacyKernel, causal_pairing_net_potentiates_the_edge_over_a_real_driven_run) {
    const f32 learning_rate = 0.02f;      // minus/depression side
    const f32 learning_rate_plus = 0.02f; // plus/potentiation side
    const s32 trial_count = 12;
    const s64 trial_period = 8; // large gap so cross-trial depression (pow(7,-3)) is negligible

    BidirectionalRunResult causal =
        run_legacy_bidirectional_pairing(/*causal=*/true, learning_rate, learning_rate_plus, trial_count, trial_period);

    // Sanity: this is a REAL driven run — both neurons actually fired, once per trial.
    EXPECT_EQ(causal.neuron0_fire_count, trial_count);
    EXPECT_EQ(causal.neuron1_fire_count, trial_count);

    std::cout << "[BidirectionalStdpLegacyKernel causal] weight(0->1) before=" << causal.weight_before
              << " after=" << causal.weight_after
              << " delta=" << (causal.weight_after - causal.weight_before) << "\n";

    // The causal (pre-before-post) pairing must NET POTENTIATE the 0 -> 1 edge — the whole point of the
    // new predecessor-side potentiation path. A margin well above float noise (the per-pairing
    // potentiation is ~learning_rate_plus, applied trial_count times).
    EXPECT_GT(causal.weight_after, causal.weight_before + 1e-3f);
}

TEST(BidirectionalStdpLegacyKernel, anti_causal_pairing_still_depresses_unchanged_from_before) {
    const f32 learning_rate = 0.02f;
    const s32 trial_count = 12;
    const s64 trial_period = 8;

    // learning_rate_plus = 0 → the potentiation path is fully gated off, so this run is byte-for-byte
    // the depression-only behavior that shipped before this change (a real regression check).
    BidirectionalRunResult anti_causal =
        run_legacy_bidirectional_pairing(/*causal=*/false, learning_rate, /*learning_rate_plus=*/0.0f, trial_count,
                                         trial_period);

    EXPECT_EQ(anti_causal.neuron0_fire_count, trial_count);
    EXPECT_EQ(anti_causal.neuron1_fire_count, trial_count);

    std::cout << "[BidirectionalStdpLegacyKernel anti-causal] weight(0->1) before=" << anti_causal.weight_before
              << " after=" << anti_causal.weight_after
              << " delta=" << (anti_causal.weight_after - anti_causal.weight_before) << "\n";

    // Post-before-pre pairing depresses the 0 -> 1 edge, exactly as the shipped depression path always did.
    EXPECT_LT(anti_causal.weight_after, anti_causal.weight_before - 1e-3f);
}

// The bidirectional-STDP U-row race fix (stage_owned_u_row/flush_owned_u_row) switches a firing
// neuron's OWN U-row stage+flush from a plain register copy/overwrite to an atomic snapshot + atomic
// net-delta ADD whenever potentiation is active (learning_rate_plus != 0). This test pins down that
// the atomic path is NUMERICALLY EQUIVALENT to the plain path for a pure-depression pairing: a SINGLE
// anti-causal pairing (neuron 1 fires at tick 2 when neuron 0 has never fired -> no potentiation lands;
// neuron 0 fires at tick 3 -> depresses 0 -> 1). The ONLY thing that differs between the two runs below
// is whether neuron 0's flush of U[0] at tick 3 goes through the atomic-delta path or the plain path;
// no other thread touches U[0], so the two must agree to well within float noise. (This validates the
// delta-add flush arithmetic deterministically, complementing the genuinely-concurrent hub test below,
// whose exact racing interleaving cannot be forced deterministically.)
TEST(BidirectionalStdpLegacyKernel, atomic_owned_u_row_flush_matches_the_plain_flush_for_pure_depression) {
    BidirectionalRunResult plain_flush =
        run_legacy_bidirectional_pairing(/*causal=*/false, /*learning_rate=*/0.02f, /*learning_rate_plus=*/0.0f,
                                         /*trial_count=*/1, /*trial_period=*/8);
    BidirectionalRunResult atomic_flush =
        run_legacy_bidirectional_pairing(/*causal=*/false, /*learning_rate=*/0.02f, /*learning_rate_plus=*/0.02f,
                                         /*trial_count=*/1, /*trial_period=*/8);

    // Both runs applied exactly one depression to 0 -> 1 and zero potentiations.
    EXPECT_LT(plain_flush.weight_after, plain_flush.weight_before - 1e-4f);
    std::cout << "[BidirectionalStdpLegacyKernel flush-equivalence] plain=" << plain_flush.weight_after
              << " atomic=" << atomic_flush.weight_after << "\n";
    EXPECT_TRUE(approx(atomic_flush.weight_after, plain_flush.weight_after, 1e-6f));
}

// Genuinely-concurrent stress test of the U-row race fix: a hub predecessor (neuron 0) with many
// successors, where every successor fires the SAME tick and simultaneously potentiates its own
// 0 -> successor edge — so U[0] takes `successor_count` concurrent atomic adds per potentiation tick,
// interleaved with neuron 0's own atomic-delta flush on its fire ticks. This is exactly the kind of
// heavy concurrent traffic on one shared U row that the fix has to keep correct; the run must stay
// finite and every edge must net-potentiate. (Note: the precise atomic-vs-plain interleaving the fix
// closes — a predecessor flushing its own U row while a successor potentiates it the SAME tick — is
// only reachable through the nondeterministic last_spiked read-window, since the same-tick exclusion
// otherwise skips it, so no test can force that exact interleaving; this maximizes concurrent U[0]
// atomic contention instead and verifies the outcome is correct/finite under the fix.)
TEST(BidirectionalStdpLegacyKernel, concurrent_potentiation_of_a_shared_hub_predecessor_stays_finite) {
    const s32 successor_count = 12;
    const s32 neuron_count = successor_count + 1; // neuron 0 = hub, 1..successor_count = successors
    vector<vector<s32>> network((usize)neuron_count);
    for (s32 successor = 1; successor <= successor_count; ++successor) network[0].push_back(successor);

    SpikeEngine engine(&network, {neuron_count, 1}, /*rank=*/8, /*resting_mp=*/0.1f, /*decay_rate=*/0.01f,
                       /*learning_rate=*/0.0f, /*plasticity_enabled=*/true,
                       /*active_set_optimization_enabled=*/false);
    engine.learning_rate = 0.02f;
    engine.learning_rate_plus = 0.02f;
    engine.use_constant_weight = true;
    engine.weights.set_constant_weight(0.01f);
    vector<s32> all_neurons;
    for (s32 neuron = 0; neuron < neuron_count; ++neuron) all_neurons.push_back(neuron);
    engine.set_input_neurons(all_neurons);
    engine.reset_state(/*last_spiked_value=*/0, /*active_gen_value=*/-1);

    vector<f32> weight_before((usize)successor_count);
    for (s32 successor = 1; successor <= successor_count; ++successor)
        weight_before[(usize)(successor - 1)] = engine.weights.get(0, successor);

    const f32 firing_pulse = 5.0f;
    const s64 base_tick = 2;
    const s64 trial_period = 8;
    const s32 trial_count = 10;
    s64 last_tick = base_tick + (s64)(trial_count - 1) * trial_period + 1;

    for (s64 tick = 0; tick <= last_tick; ++tick) {
        vector<f32> input((usize)neuron_count, 0.0f);
        for (s32 trial = 0; trial < trial_count; ++trial) {
            s64 first_tick = base_tick + (s64)trial * trial_period;
            if (tick == first_tick) {
                input[0] = firing_pulse; // hub fires
            } else if (tick == first_tick + 1) {
                for (s32 successor = 1; successor <= successor_count; ++successor)
                    input[(usize)successor] = firing_pulse; // every successor fires this same tick
            }
        }
        engine.step_simulation(input, tick);
    }

    for (s32 successor = 1; successor <= successor_count; ++successor) {
        f32 after = engine.weights.get(0, successor);
        EXPECT_TRUE(std::isfinite(after)) << "edge 0->" << successor << " went non-finite under contention";
        EXPECT_GT(after, weight_before[(usize)(successor - 1)] + 1e-3f)
            << "edge 0->" << successor << " did not net-potentiate under concurrent U[0] traffic";
    }
    engine.shutdown();
}
