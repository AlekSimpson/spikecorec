// expCurrSynapse and alphaCurrSynapse: against their closed forms on a pure-integrator target, and
// against jNeuroML on the PyNN cells they were written for.
//
// The integrator is an iafCell with no leak and an unreachable threshold, so its v rises by
// i * dt / C every tick and the total change is the charge the synapse delivered, divided by C.
//
// A refit runs as soon as the first update reaches S, which here is the arrival, and may leave up to
// fit_tolerance of relative error on the state it refits. Runs shorter than the refit spacing are
// exact; the others are held to fit_tolerance.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <gtest/gtest.h>

#include "spikecorec/core/engine.h"
#include "support/test_support.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::test_support;

namespace {

constexpr f64 INTEGRATOR_CAPACITANCE = 100e-12;
constexpr f64 NANOAMPERE = 1e-9;
constexpr f64 SYNAPSE_TIME_CONSTANT = 5e-3;  // tau_syn = 5 (ms)
constexpr f64 EULER_NUMBER = 2.718281828459045;

struct Edge {
    f64 weight = 1.0;
    String delay = "1 ms";
};

// Neuron 0 is a spikeGenerator firing every `period` (first at `period`); neurons 1.. are
// integrators, one per edge, each fed by the generator through `synapse_id`.
String synapse_model(const String &synapse_declaration, const String &synapse_id, const Vector<Edge> &edges,
                     const String &period) {
    std::ostringstream document;
    document << "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"SynapseModel\">\n"
             << "  " << synapse_declaration << "\n"
             << "  <iafCell id=\"integrator\" leakConductance=\"0 nS\" leakReversal=\"-65 mV\" thresh=\"1000 mV\""
                " reset=\"-70 mV\" C=\"100 pF\"/>\n"
             << "  <spikeGenerator id=\"generator\" period=\"" << period << "\"/>\n"
             << "  <network id=\"synapseNetwork\">\n"
             << "    <population id=\"source\" component=\"generator\" size=\"1\"/>\n"
             << "    <population id=\"targets\" component=\"integrator\" size=\"" << edges.size() << "\"/>\n"
             << "    <projection id=\"projection\" presynapticPopulation=\"source\" postsynapticPopulation=\"targets\""
                " synapse=\"" << synapse_id << "\">\n";
    for (usize index = 0; index < edges.size(); index += 1) {
        document << "      <connectionWD id=\"" << index << "\" preCellId=\"../source[0]\" postCellId=\"../targets["
                 << index << "]\" weight=\"" << edges[index].weight << "\" delay=\"" << edges[index].delay << "\"/>\n";
    }
    document << "    </projection>\n  </network>\n</neuroml>\n";
    return document.str();
}

String exp_synapse() { return R"(<expCurrSynapse id="expSynapse" tau_syn="5"/>)"; }
String alpha_synapse() { return R"(<alphaCurrSynapse id="alphaSynapse" tau_syn="5"/>)"; }

unique_ptr<SpikeEngine> make_engine(const TemporaryDirectory &directory, const String &model, const String &length,
                                    const String &step) {
    directory.write("synapse_model.nml", model);
    const String lems = directory.write(
            "LEMS.xml", "<Lems>\n  <Include file=\"Cells.xml\"/>\n  <Include file=\"Synapses.xml\"/>\n"
                        "  <Include file=\"PyNN.xml\"/>\n  <Include file=\"Networks.xml\"/>\n"
                        "  <Include file=\"Simulation.xml\"/>\n  <Include file=\"synapse_model.nml\"/>\n"
                        "  <Simulation id=\"sim1\" length=\"" + length + "\" step=\"" + step +
                        "\" target=\"synapseNetwork\"/>\n  <Target component=\"sim1\"/>\n</Lems>\n");
    return make_unique<SpikeEngine>(lems);
}

// Each tick's value of one synapse state variable on the edge into target_neuron.
Vector<f64> state_per_tick(SpikeEngine &engine, const String &synapse_id, const String &variable, s32 target_neuron) {
    const s64 plane = engine.synapse_state_variable_plane(synapse_id, variable);
    Vector<f64> values;
    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);
        values.push_back(engine.weights.get_for_matrix(0, target_neuron, plane));
    }
    return values;
}

s64 first_nonzero_index(const Vector<f64> &values) {
    const auto found = find_if(values.begin(), values.end(), [](f64 value) { return value != 0.0; });
    return found == values.end() ? -1 : (s64)(found - values.begin());
}

// How far each integrator moved over the run, from where it started.
Vector<f64> integrator_changes(SpikeEngine &engine, s64 target_count) {
    Vector<f64> starts;
    for (s64 neuron = 1; neuron <= target_count; neuron += 1) starts.push_back(engine.read_state_variable(neuron, "v"));
    engine.run();
    Vector<f64> changes;
    for (s64 neuron = 1; neuron <= target_count; neuron += 1) {
        changes.push_back(engine.read_state_variable(neuron, "v") - starts[(usize)(neuron - 1)]);
    }
    return changes;
}

} // namespace

// ── expCurrSynapse ──────────────────────────────────────────────────────────────

// An arrival adds the weight to I; forward Euler then takes I_n = w (1 - dt/tau)^n.
TEST(ExpCurrSynapse, decays_geometrically_from_its_weight) {
    const TemporaryDirectory directory;
    unique_ptr<SpikeEngine> engine =
            make_engine(directory, synapse_model(exp_synapse(), "expSynapse", {{2.5, "1 ms"}}, "30 ms"), "59ms", "0.1ms");
    const Vector<f64> current = state_per_tick(*engine, "expSynapse", "I", 1);

    const s64 arrival = first_nonzero_index(current);
    ASSERT_GE(arrival, 0) << "the arrival never came";
    const f64 ratio = 1.0 - engine->step_dt / SYNAPSE_TIME_CONSTANT;
    for (s64 elapsed = 0; elapsed <= 100; elapsed += 1) {
        EXPECT_NEAR(current[(usize)(arrival + elapsed)], 2.5 * pow(ratio, (f64)elapsed), 1e-5 * 2.5) << "tick " << elapsed;
    }
}

// The whole charge is w nA tau, and half of it has arrived tau ln 2 after the current starts.
TEST(ExpCurrSynapse, delivers_its_charge_with_the_exponential_time_course) {
    const TemporaryDirectory directory;
    unique_ptr<SpikeEngine> engine =
            make_engine(directory, synapse_model(exp_synapse(), "expSynapse", {{1.0, "1 ms"}}, "100 ms"), "199ms", "0.1ms");

    const f64 start = engine->read_state_variable(1, "v");
    Vector<f64> change_per_tick;
    for (s64 tick = 0; tick < engine->lifetime; tick += 1) {
        engine->step_simulation(tick);
        change_per_tick.push_back(engine->read_state_variable(1, "v") - start);
    }
    const f64 total_change = NANOAMPERE * SYNAPSE_TIME_CONSTANT / INTEGRATOR_CAPACITANCE;  // 50 mV
    EXPECT_NEAR(change_per_tick.back(), total_change, engine->weights.fit_tolerance * total_change);

    const s64 onset = first_nonzero_index(change_per_tick);
    ASSERT_GE(onset, 0);
    const auto half = find_if(change_per_tick.begin(), change_per_tick.end(),
                              [&](f64 change) { return change >= 0.5 * total_change; });
    ASSERT_NE(half, change_per_tick.end());
    const f64 half_time = (f64)(half - change_per_tick.begin() - onset + 1) * engine->step_dt;
    EXPECT_NEAR(half_time, SYNAPSE_TIME_CONSTANT * std::log(2.0), engine->step_dt);
}

// The response is linear in the weight, sign included, and successive arrivals add up.
TEST(ExpCurrSynapse, responses_are_linear_in_the_weight_and_superpose) {
    const TemporaryDirectory directory;
    unique_ptr<SpikeEngine> engine = make_engine(
            directory, synapse_model(exp_synapse(), "expSynapse", {{1.0, "1 ms"}, {2.0, "1 ms"}, {-1.0, "1 ms"}}, "40 ms"),
            "199ms", "0.1ms");
    const Vector<f64> changes = integrator_changes(*engine, 3);

    EXPECT_NEAR(changes[1] / changes[0], 2.0, 1e-5);
    EXPECT_NEAR(changes[2] / changes[0], -1.0, 1e-5);
    // Arrivals at about 41, 81, 121 and 161 ms; the last one's tail past 199 ms is e^-7.6 of it.
    const f64 single_arrival = NANOAMPERE * SYNAPSE_TIME_CONSTANT / INTEGRATOR_CAPACITANCE;
    EXPECT_NEAR(changes[0], 4.0 * single_arrival, (1e-3 + 4.0 * engine->weights.fit_tolerance) * single_arrival);
}

// Two edges from one source, with their own weights and delays, keep their own state: the later
// edge is untouched until its own arrival, then follows the same curve scaled by its weight.
TEST(ExpCurrSynapse, each_edge_keeps_its_own_state) {
    const TemporaryDirectory directory;
    unique_ptr<SpikeEngine> engine = make_engine(
            directory, synapse_model(exp_synapse(), "expSynapse", {{1.0, "1 ms"}, {3.0, "4 ms"}}, "30 ms"), "59ms", "0.1ms");
    const s64 plane = engine->synapse_state_variable_plane("expSynapse", "I");
    Vector<f64> first_edge;
    Vector<f64> second_edge;
    for (s64 tick = 0; tick < engine->lifetime; tick += 1) {
        engine->step_simulation(tick);
        first_edge.push_back(engine->weights.get_for_matrix(0, 1, plane));
        second_edge.push_back(engine->weights.get_for_matrix(0, 2, plane));
    }

    const s64 first_arrival = first_nonzero_index(first_edge);
    const s64 second_arrival = first_nonzero_index(second_edge);
    ASSERT_GE(first_arrival, 0);
    EXPECT_EQ(second_arrival - first_arrival, 30) << "3 ms more delay at 0.1 ms";
    for (s64 elapsed = 0; elapsed <= 100; elapsed += 1) {
        EXPECT_NEAR(second_edge[(usize)(second_arrival + elapsed)], 3.0 * first_edge[(usize)(first_arrival + elapsed)], 3e-5)
            << "tick " << elapsed;
    }
}

// ── alphaCurrSynapse ────────────────────────────────────────────────────────────

// A' = -A/tau and I' = (e A - I)/tau: after an arrival adds w to A, I rises to w at t = tau.
TEST(AlphaCurrSynapse, peaks_at_its_weight_one_time_constant_after_arrival) {
    const TemporaryDirectory directory;
    unique_ptr<SpikeEngine> engine = make_engine(
            directory, synapse_model(alpha_synapse(), "alphaSynapse", {{2.0, "1 ms"}}, "40 ms"), "79ms", "0.025ms");
    const s64 current_plane = engine->synapse_state_variable_plane("alphaSynapse", "I");
    const s64 amplitude_plane = engine->synapse_state_variable_plane("alphaSynapse", "A");
    Vector<f64> current;
    Vector<f64> amplitude;
    for (s64 tick = 0; tick < engine->lifetime; tick += 1) {
        engine->step_simulation(tick);
        current.push_back(engine->weights.get_for_matrix(0, 1, current_plane));
        amplitude.push_back(engine->weights.get_for_matrix(0, 1, amplitude_plane));
    }

    const s64 arrival = first_nonzero_index(amplitude);
    ASSERT_GE(arrival, 0);
    EXPECT_NEAR(amplitude[(usize)arrival], 2.0, 2.0 * engine->weights.fit_tolerance);
    const auto peak = max_element(current.begin(), current.end());
    EXPECT_NEAR(*peak, 2.0, 0.01 * 2.0);
    const f64 time_to_peak = (f64)(peak - current.begin() - arrival) * engine->step_dt;
    EXPECT_NEAR(time_to_peak, SYNAPSE_TIME_CONSTANT, 0.01 * SYNAPSE_TIME_CONSTANT);
}

// The alpha function integrates to e w tau; Euler's own recurrence gives exactly the same sum. Sign
// and size scale with the weight.
TEST(AlphaCurrSynapse, delivers_e_times_weight_times_time_constant) {
    const TemporaryDirectory directory;
    unique_ptr<SpikeEngine> engine = make_engine(
            directory, synapse_model(alpha_synapse(), "alphaSynapse", {{0.5, "1 ms"}, {-0.5, "1 ms"}}, "100 ms"),
            "199ms", "0.025ms");
    const Vector<f64> changes = integrator_changes(*engine, 2);

    const f64 total_change = EULER_NUMBER * 0.5 * NANOAMPERE * SYNAPSE_TIME_CONSTANT / INTEGRATOR_CAPACITANCE;
    EXPECT_NEAR(changes[0], total_change, engine->weights.fit_tolerance * total_change);
    EXPECT_NEAR(changes[1], -total_change, engine->weights.fit_tolerance * total_change);
}

// ── against jNeuroML ────────────────────────────────────────────────────────────

namespace {

// tests/fixtures/nml/LEMS_pynn_synapses.xml: neuron 0 is the spikeGenerator, 1 expPsp, 2 alphaPsp,
// 3 expSpiking, 4 alphaSpiking. Reference outputs are from regenerate_references.sh.
struct PynnRun {
    f64 step = 0.0;
    Vector<Vector<f64>> potentials;     // [neuron 1..4][row]; row k is t = k * step
    Vector<Vector<s64>> spike_ticks;    // [neuron 0..4]
};

PynnRun run_engine() {
    SpikeEngine engine(fixture_path("nml/LEMS_pynn_synapses.xml"));
    PynnRun run;
    run.step = engine.step_dt;
    run.potentials.assign(5, {});
    for (s64 neuron = 1; neuron <= 4; neuron += 1) run.potentials[(usize)neuron].push_back(engine.read_state_variable(neuron, "v"));
    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);
        // After tick k the state is at t = (k + 1) * step.
        for (s64 neuron = 1; neuron <= 4; neuron += 1) run.potentials[(usize)neuron].push_back(engine.read_state_variable(neuron, "v"));
    }
    run.spike_ticks.assign(5, {});
    for (const RecordedSpike &spike : engine.recorded_spikes) {
        run.spike_ticks[(usize)spike.neuron_index].push_back(llround(spike.time_seconds / run.step));
    }
    return run;
}

PynnRun read_reference(f64 step) {
    PynnRun run;
    run.step = step;
    run.potentials.assign(5, {});
    std::ifstream potentials(fixture_path("reference/pynn_synapses_v.dat"));
    EXPECT_TRUE(potentials.good()) << "run tests/fixtures/reference/regenerate_references.sh";
    f64 time = 0.0;
    f64 exp_psp = 0.0, alpha_psp = 0.0, exp_spiking = 0.0, alpha_spiking = 0.0;
    while (potentials >> time >> exp_psp >> alpha_psp >> exp_spiking >> alpha_spiking) {
        run.potentials[1].push_back(exp_psp);
        run.potentials[2].push_back(alpha_psp);
        run.potentials[3].push_back(exp_spiking);
        run.potentials[4].push_back(alpha_spiking);
    }
    run.spike_ticks.assign(5, {});
    std::ifstream spikes(fixture_path("reference/pynn_synapses_spikes.dat"));
    f64 spike_time = 0.0;
    s64 neuron = 0;
    while (spikes >> spike_time >> neuron) run.spike_ticks[(usize)neuron].push_back(llround(spike_time / step));
    return run;
}

struct PostsynapticPotential {
    f64 height = 0.0;
    s64 peak_row = 0;
};

// The first postsynaptic potential: its height above rest and the row it peaks on, looking no
// further than the next presynaptic spike.
PostsynapticPotential first_potential(const Vector<f64> &trace, s64 last_row) {
    const auto peak = max_element(trace.begin(), trace.begin() + last_row);
    return {*peak - trace.front(), (s64)(peak - trace.begin())};
}

} // namespace

TEST(Reference, pynn_postsynaptic_potentials_match_jneuroml) {
    const PynnRun engine = run_engine();
    const PynnRun reference = read_reference(engine.step);
    ASSERT_FALSE(reference.potentials[1].empty());
    ASSERT_FALSE(reference.spike_ticks[0].empty());

    // The first presynaptic spike, and the row just before the second one's arrival.
    const s64 presynaptic_tick = reference.spike_ticks[0][0];
    const s64 last_row = reference.spike_ticks[0][1];
    for (s64 neuron : {1, 2}) {
        const PostsynapticPotential expected = first_potential(reference.potentials[(usize)neuron], last_row);
        const PostsynapticPotential measured = first_potential(engine.potentials[(usize)neuron], last_row);
        EXPECT_NEAR(measured.height, expected.height, 0.01 * expected.height) << "neuron " << neuron;
        EXPECT_NEAR((f64)(measured.peak_row - presynaptic_tick), (f64)(expected.peak_row - presynaptic_tick),
                    0.01 * (f64)(expected.peak_row - presynaptic_tick))
            << "neuron " << neuron << ": time to peak after the presynaptic spike";
    }
}

TEST(Reference, pynn_spike_times_match_jneuroml) {
    const PynnRun engine = run_engine();
    const PynnRun reference = read_reference(engine.step);
    for (s64 neuron : {3, 4}) {
        const Vector<s64> &expected = reference.spike_ticks[(usize)neuron];
        const Vector<s64> &measured = engine.spike_ticks[(usize)neuron];
        ASSERT_FALSE(expected.empty()) << "neuron " << neuron;
        EXPECT_NEAR((f64)measured.size(), (f64)expected.size(), 1.0) << "neuron " << neuron;
        for (usize index = 0; index < min(expected.size(), measured.size()); index += 1) {
            EXPECT_NEAR((f64)measured[index], (f64)expected[index], 2.0) << "neuron " << neuron << " spike " << index;
        }
    }
}
