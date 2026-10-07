// Tests for the NeuroML-driven engine: what LEMS expressions mean, what the generated kernel
// computes, and what a simulation does when it runs.
//
// Expected values come from closed forms, from the host evaluator, or from a reference written in
// the test (forward Euler), so a kernel that integrates the wrong equation fails here instead of
// producing a plausible recording nobody can check.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <gtest/gtest.h>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/engine.h"
#include "spikecorec/core/topologies.h"
#include "spikecorec/nml/dynamics.h"
#include "spikecorec/nml/random_generator.h"
#include "support/test_support.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;
using namespace spikecorec::test_support;

namespace {

// One iafCell under a constant supra-rheobase current. tau = C/gL = 20 ms and the rheobase is
// gL*(thresh - leakReversal) = 75 pA, so at 90 pA the cell fires with an interval that has a
// closed form.
String single_cell_model(const String &amplitude = "90 pA") {
    return R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="SingleCell">
  <iafCell id="testCell" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV" reset="-70 mV" C="100 pF"/>
  <pulseGenerator id="drive" delay="0 ms" duration="1000 ms" amplitude=")" + amplitude + R"("/>
  <network id="singleCellNetwork">
    <population id="cellPopulation" component="testCell" size="1"/>
    <explicitInput target="cellPopulation[0]" input="drive"/>
  </network>
</neuroml>
)";
}

// tau * ln((I/gL - (reset - EL)) / (I/gL - (thresh - EL))) for single_cell_model at 90 pA.
f64 analytic_interspike_interval() {
    const f64 membrane_time_constant = 100e-12 / 5e-9;
    const f64 drive_in_volts = 90e-12 / 5e-9;
    return membrane_time_constant *
           std::log((drive_in_volts - (-0.070 - -0.065)) / (drive_in_volts - (-0.050 - -0.065)));
}

// Writes model_contents as model.nml beside a LEMS document that simulates network_id, and returns
// the LEMS document's path.
String write_model(const TemporaryDirectory &directory, const String &model_contents, const String &network_id,
                   const String &length, const String &step, const String &simulation_children = "") {
    directory.write("model.nml", model_contents);
    return directory.write("LEMS.xml", lems_document("model.nml", network_id, length, step, simulation_children));
}

// The include line for the GLIF cell types, which live with the fixtures.
String glif_types_include() {
    return "  <include href=\"" + fixture_path("nml/glif_cell_types.nml") + "\"/>\n";
}

// A cell type whose state variables each integrate one constant expression, so after n ticks each
// one holds n * dt * (the expression's value): what the GPU computed for that expression. The
// arguments are state variables set at OnStart, so the kernel evaluates each function at run time.
struct ProbeExpression {
    String name;
    String expression;
};

const Vector<ProbeExpression> &probe_expressions() {
    static const Vector<ProbeExpression> expressions = {
        {"natural_log", "log(four)"},
        {"ln_alias", "ln(four)"},
        {"exponential", "exp(half)"},
        {"square_root", "sqrt(four)"},
        {"absolute", "abs(negative)"},
        {"heaviside_zero", "H(zero)"},
        {"heaviside_positive", "H(two)"},
        {"heaviside_negative", "H(negative)"},
        {"power", "two ^ three"},
        {"right_associative_power", "two ^ half ^ two"},
        {"floor_probe", "floor(negative)"},
        {"ceiling_probe", "ceil(negative)"},
        {"sine", "sin(half)"},
        {"hyperbolic_tangent", "tanh(half)"},
        {"precedence", "two + three * four"},
        {"left_associative_division", "four / two / two"},
        {"unary_minus", "-three + two"},
    };
    return expressions;
}

const UnorderedMap<String, f64> &probe_inputs() {
    static const UnorderedMap<String, f64> inputs = {
        {"four", 4.0}, {"two", 2.0}, {"three", 3.0}, {"half", 0.5}, {"zero", 0.0}, {"negative", -1.5}};
    return inputs;
}

String probe_cell_model(s64 population_size, const String &extra_dynamics = "") {
    std::ostringstream document;
    document << R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Probe">
  <ComponentType name="probeCell" extends="baseCellMembPot">
    <Dynamics>
      <StateVariable name="v" dimension="voltage" exposure="v"/>
)";
    for (const auto &[name, value] : probe_inputs()) {
        (void)value;
        document << "      <StateVariable name=\"" << name << "\" dimension=\"none\"/>\n";
    }
    for (const ProbeExpression &probe : probe_expressions()) {
        document << "      <StateVariable name=\"" << probe.name << "\" dimension=\"none\"/>\n"
                 << "      <TimeDerivative variable=\"" << probe.name << "\" value=\"" << probe.expression << "\"/>\n";
    }
    document << extra_dynamics << "      <OnStart>\n        <StateAssignment variable=\"v\" value=\"0\"/>\n";
    for (const auto &[name, value] : probe_inputs()) {
        document << "        <StateAssignment variable=\"" << name << "\" value=\"" << value << "\"/>\n";
    }
    document << R"(      </OnStart>
    </Dynamics>
  </ComponentType>
  <probeCell id="probe"/>
  <network id="probeNetwork">
    <population id="probePopulation" component="probe" size=")"
             << population_size << R"("/>
  </network>
</neuroml>
)";
    return document.str();
}

} // namespace

// ── LEMS expressions, on the host ───────────────────────────────────────────────

TEST(Expression, dotted_operators_compare_and_combine) {
    const UnorderedMap<String, f64> none;
    EXPECT_EQ(evaluate_lems("2 .gt. 1", none, "test"), 1.0);
    EXPECT_EQ(evaluate_lems("1 .gt. 1", none, "test"), 0.0);
    EXPECT_EQ(evaluate_lems("1 .geq. 1", none, "test"), 1.0);
    EXPECT_EQ(evaluate_lems("1 .lt. 2", none, "test"), 1.0);
    EXPECT_EQ(evaluate_lems("2 .leq. 1", none, "test"), 0.0);
    EXPECT_EQ(evaluate_lems("3 .eq. 3", none, "test"), 1.0);
    EXPECT_EQ(evaluate_lems("3 .neq. 3", none, "test"), 0.0);
    EXPECT_EQ(evaluate_lems("(1 .lt. 2) .and. (3 .gt. 4)", none, "test"), 0.0);
    EXPECT_EQ(evaluate_lems("(1 .lt. 2) .or. (3 .gt. 4)", none, "test"), 1.0);
}

TEST(Expression, precedence_and_associativity_follow_the_arithmetic) {
    const UnorderedMap<String, f64> none;
    EXPECT_DOUBLE_EQ(evaluate_lems("2 + 3 * 4", none, "test"), 14.0);
    EXPECT_DOUBLE_EQ(evaluate_lems("(2 + 3) * 4", none, "test"), 20.0);
    EXPECT_DOUBLE_EQ(evaluate_lems("10 - 4 - 3", none, "test"), 3.0);
    EXPECT_DOUBLE_EQ(evaluate_lems("12 / 3 / 2", none, "test"), 2.0);
    // ^ is the one right-associative operator: 2^3^2 is 2^9.
    EXPECT_DOUBLE_EQ(evaluate_lems("2 ^ 3 ^ 2", none, "test"), 512.0);
    EXPECT_DOUBLE_EQ(evaluate_lems("2 * 3 ^ 2", none, "test"), 18.0);
}

// LEMS log is the natural logarithm, and H(0) is 0: H(0) = 1 stalled spikeGeneratorPoisson.
TEST(Expression, functions_have_their_lems_meaning) {
    const UnorderedMap<String, f64> values = {{"x", 2.0}};
    EXPECT_NEAR(evaluate_lems("log(2.718281828459045)", values, "test"), 1.0, 1e-15);
    EXPECT_DOUBLE_EQ(evaluate_lems("ln(x)", values, "test"), std::log(2.0));
    EXPECT_DOUBLE_EQ(evaluate_lems("exp(x)", values, "test"), std::exp(2.0));
    EXPECT_DOUBLE_EQ(evaluate_lems("sqrt(x * 8)", values, "test"), 4.0);
    EXPECT_DOUBLE_EQ(evaluate_lems("abs(-x)", values, "test"), 2.0);
    EXPECT_DOUBLE_EQ(evaluate_lems("floor(-1.5)", values, "test"), -2.0);
    EXPECT_DOUBLE_EQ(evaluate_lems("ceil(-1.5)", values, "test"), -1.0);
    EXPECT_DOUBLE_EQ(evaluate_lems("H(0)", values, "test"), 0.0);
    EXPECT_DOUBLE_EQ(evaluate_lems("H(1e-30)", values, "test"), 1.0);
    EXPECT_DOUBLE_EQ(evaluate_lems("H(-1)", values, "test"), 0.0);
    EXPECT_DOUBLE_EQ(evaluate_lems("x ^ 0.5", values, "test"), std::sqrt(2.0));
    EXPECT_DOUBLE_EQ(evaluate_lems("tanh(x)", values, "test"), std::tanh(2.0));
}

TEST(Expression, random_draws_uniformly_up_to_its_argument) {
    RandomGenerator random_generator(42);
    const UnorderedMap<String, f64> none;
    f64 sum = 0.0;
    const s32 draw_count = 20000;
    for (s32 draw = 0; draw < draw_count; draw += 1) {
        const f64 value = evaluate_lems("random(3)", none, "test", &random_generator);
        ASSERT_GT(value, 0.0);
        ASSERT_LT(value, 3.0);
        sum += value;
    }
    // Mean 1.5, standard error 3 / sqrt(12 * 20000) = 0.0061.
    EXPECT_NEAR(sum / draw_count, 1.5, 0.03);
    // Without a generator there is nothing to draw from.
    EXPECT_THROW(evaluate_lems("random(1)", none, "test"), runtime_error);
}

TEST(Expression, malformed_expressions_throw_naming_their_owner) {
    const UnorderedMap<String, f64> none;
    const auto message_of = [&](const String &expression) -> String {
        try {
            evaluate_lems(expression, none, "ownerType");
        } catch (const runtime_error &error) {
            return error.what();
        }
        return "";
    };
    EXPECT_NE(message_of("nosuchfunction(2)").find("nosuchfunction"), String::npos);
    EXPECT_NE(message_of("(2 + 3").find("ownerType"), String::npos);
    EXPECT_NE(message_of("2 $ 3").find("$"), String::npos);
    EXPECT_NE(message_of("unbound + 1").find("unbound"), String::npos);
}

TEST(RandomGenerator, each_distribution_has_its_mean_and_a_seed_reproduces_it) {
    RandomGenerator first(7);
    RandomGenerator second(7);
    const s32 draw_count = 50000;
    f64 uniform_sum = 0.0;
    f64 normal_sum = 0.0;
    f64 normal_square_sum = 0.0;
    f64 exponential_sum = 0.0;
    f64 poisson_sum = 0.0;
    for (s32 draw = 0; draw < draw_count; draw += 1) {
        const f64 uniform = first.uniform();
        ASSERT_GT(uniform, 0.0);
        ASSERT_LT(uniform, 1.0);
        EXPECT_EQ(uniform, second.uniform());
        uniform_sum += uniform;
        const f64 normal = first.normal(2.0, 0.5);
        normal_sum += normal;
        normal_square_sum += (normal - 2.0) * (normal - 2.0);
        exponential_sum += first.exponential(4.0);
        poisson_sum += (f64)first.poisson(3.0);
        second.normal(2.0, 0.5);
        second.exponential(4.0);
        second.poisson(3.0);
    }
    EXPECT_NEAR(uniform_sum / draw_count, 0.5, 0.005);
    EXPECT_NEAR(normal_sum / draw_count, 2.0, 0.01);
    EXPECT_NEAR(sqrt(normal_square_sum / draw_count), 0.5, 0.01);
    EXPECT_NEAR(exponential_sum / draw_count, 0.25, 0.005);
    EXPECT_NEAR(poisson_sum / draw_count, 3.0, 0.03);
}

// ── the generated kernel ────────────────────────────────────────────────────────

// Every LEMS function and operator, computed by the GPU, against the host's value.
TEST(GeneratedKernel, every_function_computes_on_the_gpu_what_the_host_computes) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory, probe_cell_model(2), "probeNetwork", "10ms", "0.1ms"));

    const s64 tick_count = 100;
    for (s64 tick = 0; tick < tick_count; tick += 1) engine.step_simulation(tick);

    const f64 elapsed = (f64)tick_count * engine.step_dt;
    for (const ProbeExpression &probe : probe_expressions()) {
        const f64 expected = evaluate_lems(probe.expression, probe_inputs(), "probeCell") * elapsed;
        for (s64 neuron = 0; neuron < 2; neuron += 1) {
            // f32 accumulation over 100 ticks and Metal's own function precision; H(0) must still be 0.
            EXPECT_NEAR(engine.read_state_variable(neuron, probe.name), expected, 1e-4 * max(fabs(expected), 1e-4))
                << probe.expression << " on neuron " << neuron;
        }
    }
    EXPECT_NE(engine.master_kernel_source.find("master_step"), String::npos);
}

// Each random() call gets one slot of random_values per neuron, drawn afresh every tick.
TEST(GeneratedKernel, random_reads_one_fresh_slot_per_call_per_neuron) {
    const TemporaryDirectory directory;
    const s64 neuron_count = 1000;
    const String drawn_dynamics = R"xml(      <StateVariable name="drawn" dimension="none"/>
      <TimeDerivative variable="drawn" value="random(1) + random(2)"/>
)xml";
    SpikeEngine engine(write_model(directory, probe_cell_model(neuron_count, drawn_dynamics), "probeNetwork",
                                   "100ms", "0.1ms"));
    EXPECT_EQ(engine.random_values_count, 2 * neuron_count);

    const s64 tick_count = 1000;
    for (s64 tick = 0; tick < tick_count; tick += 1) engine.step_simulation(tick);

    // Each tick adds dt * (U(0,1) + U(0,2)): mean 1.5 dt, variance dt^2 * (1 + 4) / 12.
    const f64 step = engine.step_dt;
    Vector<f64> totals;
    for (s64 neuron = 0; neuron < neuron_count; neuron += 1) totals.push_back(engine.read_state_variable(neuron, "drawn"));
    const f64 expected_mean = 1.5 * step * (f64)tick_count;
    const f64 expected_variance = step * step * (5.0 / 12.0) * (f64)tick_count;
    const f64 mean = mean_of(totals);
    f64 variance = 0.0;
    for (f64 total : totals) variance += (total - mean) * (total - mean);
    variance /= (f64)(totals.size() - 1);

    EXPECT_NEAR(mean, expected_mean, 4.0 * sqrt(expected_variance / (f64)neuron_count));
    // A draw reused across ticks or across neurons would make this many times larger, or zero.
    EXPECT_NEAR(variance / expected_variance, 1.0, 0.2);
}

// An OnStart that calls random() draws once per neuron on the host.
TEST(GeneratedKernel, an_on_start_random_draws_once_per_neuron) {
    const TemporaryDirectory directory;
    const s64 neuron_count = 2000;
    const String start_dynamics = R"xml(      <StateVariable name="start" dimension="none"/>
      <OnStart><StateAssignment variable="start" value="random(1)"/></OnStart>
)xml";
    SpikeEngine engine(write_model(directory, probe_cell_model(neuron_count, start_dynamics), "probeNetwork",
                                   "1ms", "0.1ms"));

    Vector<f64> starts;
    for (s64 neuron = 0; neuron < neuron_count; neuron += 1) starts.push_back(engine.read_state_variable(neuron, "start"));
    for (f64 start : starts) {
        ASSERT_GT(start, 0.0);
        ASSERT_LT(start, 1.0);
    }
    EXPECT_NEAR(mean_of(starts), 0.5, 4.0 * sqrt(1.0 / 12.0 / (f64)neuron_count));
    const s64 below_a_quarter = count_if(starts.begin(), starts.end(), [](f64 start) { return start < 0.25; });
    EXPECT_NEAR((f64)below_a_quarter / (f64)neuron_count, 0.25, 0.04);
    EXPECT_GT(set<f64>(starts.begin(), starts.end()).size(), (usize)(neuron_count * 0.99));
}

// The integrating / refractory regime pair lowers to a test on the tick since the last spike, and
// state outside the regimes keeps evolving while the cell is held: GLIF3's after-spike currents
// decay during the refractory period.
TEST(GeneratedKernel, after_spike_currents_decay_through_the_refractory_period) {
    SpikeEngine engine(fixture_path("nml/LEMS_glif_family.xml"));
    EXPECT_NE(engine.master_kernel_source.find("refractory"), String::npos);

    const s64 glif3 = 2;
    s64 first_spike_tick = -1;
    f32 after_spike_current = 0.0f;
    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);
        if (first_spike_tick < 0) {
            for (const RecordedSpike &spike : engine.recorded_spikes) {
                if (spike.neuron_index == glif3) first_spike_tick = tick;
            }
            if (first_spike_tick >= 0) after_spike_current = engine.read_state_variable(glif3, "asc2");
            continue;
        }
        // 4 ms into the 5 ms refractory period: asc2 (tau 10 ms) has decayed by about a third.
        if (tick == first_spike_tick + (s64)llround(4e-3 / engine.step_dt)) {
            const f32 decayed = engine.read_state_variable(glif3, "asc2");
            ASSERT_LT(after_spike_current, 0.0f);
            EXPECT_NEAR(decayed / after_spike_current, std::exp(-4e-3 / 10e-3), 0.02);
            break;
        }
    }
    EXPECT_GE(first_spike_tick, 0) << "GLIF3 never fired";
}

TEST(GeneratedKernel, a_regime_shape_that_is_not_the_refractory_pair_is_refused) {
    const TemporaryDirectory directory;
    const String lems = write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ThreeRegimes">
  <ComponentType name="ThreeRegimeCell" extends="baseCellMembPot">
    <Parameter name="vth" dimension="voltage"/>
    <Dynamics>
      <StateVariable name="v" dimension="voltage" exposure="v"/>
      <Regime name="one" initial="true">
        <TimeDerivative variable="v" value="1"/>
        <OnCondition test="v .gt. vth"><Transition regime="two"/></OnCondition>
      </Regime>
      <Regime name="two"><OnCondition test="v .gt. vth"><Transition regime="three"/></OnCondition></Regime>
      <Regime name="three"><OnCondition test="v .gt. vth"><Transition regime="one"/></OnCondition></Regime>
    </Dynamics>
  </ComponentType>
  <ThreeRegimeCell id="cell" vth="1mV"/>
  <network id="threeRegimeNetwork"><population id="pop" component="cell" size="1"/></network>
</neuroml>
)", "threeRegimeNetwork", "10ms", "0.1ms");
    try {
        SpikeEngine engine(lems);
        FAIL() << "a three-regime state machine must not lower to a refractory gate";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("ThreeRegimeCell"), String::npos) << message;
        EXPECT_NE(message.find("refractory regime pair"), String::npos) << message;
    }
}

// Leaving refractory runs the exit OnCondition's assignments and the integrating regime's OnEntry,
// once, on the first tick the cell is no longer refractory. Its phase climbs at 0.1 per ms, it
// fires at 1 and is refractory while (tick - spike) * dt <= 2.05 ms, so it leaves 21 ticks after
// each spike.
TEST(GeneratedKernel, leaving_refractory_runs_the_exit_and_the_integrating_on_entry) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Leaving">
  <ComponentType name="leavingCell" extends="baseCellMembPot">
    <Parameter name="refract" dimension="time"/>
    <Parameter name="rate" dimension="per_time"/>
    <Dynamics>
      <StateVariable name="v" dimension="voltage" exposure="v"/>
      <StateVariable name="phase" dimension="none"/>
      <StateVariable name="exits" dimension="none"/>
      <StateVariable name="entries" dimension="none"/>
      <StateVariable name="lastSpikeTime" dimension="time"/>
      <Regime name="integrating" initial="true">
        <OnEntry>
          <StateAssignment variable="entries" value="entries + 1"/>
        </OnEntry>
        <TimeDerivative variable="phase" value="rate"/>
        <OnCondition test="phase .gt. 1">
          <StateAssignment variable="phase" value="0"/>
          <EventOut port="spike"/>
          <Transition regime="refractory"/>
        </OnCondition>
      </Regime>
      <Regime name="refractory">
        <OnEntry>
          <StateAssignment variable="lastSpikeTime" value="t"/>
        </OnEntry>
        <OnCondition test="t .gt. lastSpikeTime + refract">
          <StateAssignment variable="exits" value="exits + 1"/>
          <Transition regime="integrating"/>
        </OnCondition>
      </Regime>
    </Dynamics>
  </ComponentType>
  <leavingCell id="cell" refract="2.05ms" rate="0.1per_ms"/>
  <network id="leavingNetwork"><population id="pop" component="cell" size="1"/></network>
</neuroml>
)", "leavingNetwork", "60ms", "0.1ms"));

    Vector<s64> change_ticks;
    f32 previous_exits = 0.0f;
    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);
        const f32 exits = engine.read_state_variable(0, "exits");
        EXPECT_EQ(engine.read_state_variable(0, "entries"), exits) << "tick " << tick;
        if (exits != previous_exits) {
            EXPECT_EQ(exits - previous_exits, 1.0f) << "tick " << tick;
            change_ticks.push_back(tick);
        }
        previous_exits = exits;
    }

    Vector<s64> expected_ticks;
    for (f64 time : spike_times_of(engine, 0)) {
        const s64 leaving_tick = llround(time / engine.step_dt) + 21;
        if (leaving_tick < engine.lifetime) expected_ticks.push_back(leaving_tick);
    }
    ASSERT_GE(expected_ticks.size(), 3u);
    EXPECT_EQ(change_ticks, expected_ticks);
}

// A synapse that reads the target's v (conductance-based) is refused, naming what it reads.
TEST(GeneratedKernel, a_conductance_based_synapse_is_refused_naming_v) {
    const TemporaryDirectory directory;
    const String lems = write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Conductance">
  <expOneSynapse id="conductanceSynapse" gbase="1 nS" erev="0 mV" tauDecay="5 ms"/>
  <iafCell id="c" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV" reset="-70 mV" C="100 pF"/>
  <network id="conductanceNetwork">
    <population id="pop" component="c" size="2"/>
    <projection id="proj" presynapticPopulation="pop" postsynapticPopulation="pop" synapse="conductanceSynapse">
      <connectionWD id="0" preCellId="../pop[0]" postCellId="../pop[1]" weight="1" delay="1 ms"/>
    </projection>
  </network>
</neuroml>
)", "conductanceNetwork", "10ms", "0.1ms");
    try {
        SpikeEngine engine(lems);
        FAIL() << "a conductance-based synapse must not silently lower";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("expOneSynapse"), String::npos) << message;
        EXPECT_NE(message.find("'v'"), String::npos) << message;
    }
}

// ── what a cell's inputs must provide ───────────────────────────────────────────

TEST(CellInputs, an_input_into_a_cell_that_reads_none_is_refused) {
    const TemporaryDirectory directory;
    const String lems = write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="NoInput">
  <iafTauCell id="c" leakReversal="-65 mV" tau="10 ms" thresh="-50 mV" reset="-70 mV"/>
  <pulseGenerator id="drive" delay="0 ms" duration="10 ms" amplitude="1 nA"/>
  <network id="noInputNetwork">
    <population id="pop" component="c" size="1"/>
    <explicitInput target="pop[0]" input="drive"/>
  </network>
</neuroml>
)", "noInputNetwork", "10ms", "0.1ms");
    try {
        SpikeEngine engine(lems);
        FAIL() << "iafTauCell reads no input, so a current into it would be dropped";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("iafTauCell"), String::npos) << message;
        EXPECT_NE(message.find("reads no input"), String::npos) << message;
    }
}

// A synapse must expose what the cell's select reads, in the same dimension.
TEST(CellInputs, a_synapse_exposing_the_wrong_dimension_is_refused) {
    const TemporaryDirectory directory;
    const String lems = write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="WrongDimension">
  <ComponentType name="voltageSynapse" extends="baseSynapse">
    <Exposure name="i" dimension="voltage"/>
    <Dynamics><DerivedVariable name="i" dimension="voltage" exposure="i" value="0"/></Dynamics>
  </ComponentType>
  <voltageSynapse id="wrongSynapse"/>
  <iafCell id="c" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV" reset="-70 mV" C="100 pF"/>
  <network id="wrongNetwork">
    <population id="pop" component="c" size="2"/>
    <projection id="proj" presynapticPopulation="pop" postsynapticPopulation="pop" synapse="wrongSynapse">
      <connection id="0" preCellId="../pop[0]" postCellId="../pop[1]"/>
    </projection>
  </network>
</neuroml>
)", "wrongNetwork", "10ms", "0.1ms");
    try {
        SpikeEngine engine(lems);
        FAIL() << "a synapse exposing i as a voltage must be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("wrongSynapse"), String::npos) << message;
        EXPECT_NE(message.find("voltage"), String::npos) << message;
    }
}

// A spike train's default kick is the charge that carries the cell from reset to threshold, which
// needs the cell's capacitance; a cell that reads no input cannot take one at all.
TEST(CellInputs, a_spike_train_onto_a_cell_that_cannot_take_it_is_refused) {
    const TemporaryDirectory directory;
    const String lems = write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Train">
  <iafTauCell id="c" leakReversal="-65 mV" tau="10 ms" thresh="-50 mV" reset="-70 mV"/>
  <spikeArray id="train"><spike id="0" time="10 ms"/><spike id="1" time="20 ms"/></spikeArray>
  <network id="trainNetwork">
    <population id="pop" component="c" size="1"/>
    <explicitInput target="pop[0]" input="train"/>
  </network>
</neuroml>
)", "trainNetwork", "50ms", "0.1ms");
    try {
        SpikeEngine engine(lems);
        FAIL() << "a spike train onto iafTauCell must be refused";
    } catch (const runtime_error &error) {
        EXPECT_NE(String(error.what()).find("iafTauCell"), String::npos) << error.what();
    }
}

// ── a single cell ───────────────────────────────────────────────────────────────

// Not zero, which is what skipping OnStart would leave, and which is above threshold.
TEST(SingleCell, on_start_puts_the_cell_at_its_leak_reversal) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory, single_cell_model(), "singleCellNetwork", "10ms", "0.05ms"));
    EXPECT_NEAR(engine.read_state_variable(0, "v"), -0.065f, 1e-7f);
}

TEST(SingleCell, the_interspike_interval_matches_the_closed_form) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory, single_cell_model(), "singleCellNetwork", "500ms", "0.05ms"));
    engine.run();

    const Vector<f64> times = spike_times_of(engine, 0);
    ASSERT_GE(times.size(), 4u);
    // Explicit Euler at 0.05 ms plus one tick of threshold-crossing discretisation.
    EXPECT_NEAR(times[3] - times[2], analytic_interspike_interval(), 2e-4);
    EXPECT_NEAR(engine.mean_firing_rate_hertz(), 1.0 / analytic_interspike_interval(), 1.0);
}

// v(t) = EL + (I/gL)(1 - e^(-t/tau)) below threshold: a wrong sign, a missing capacitance or a
// dropped input reaches threshold at another time and also takes another route there.
TEST(SingleCell, the_subthreshold_trajectory_matches_the_closed_form) {
    const TemporaryDirectory directory;
    // 60 pA is below the 75 pA rheobase: the cell charges toward -53 mV and never fires.
    SpikeEngine engine(write_model(directory, single_cell_model("60 pA"), "singleCellNetwork", "100ms", "0.05ms"));

    const f64 membrane_time_constant = 100e-12 / 5e-9;
    const f64 steady_state_offset = 60e-12 / 5e-9;
    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);
        if (tick % 200 != 0) continue;
        const f64 elapsed = (f64)(tick + 1) * engine.step_dt;
        const f64 expected = -0.065 + steady_state_offset * (1.0 - std::exp(-elapsed / membrane_time_constant));
        EXPECT_NEAR(engine.read_state_variable(0, "v"), (f32)expected, 5e-5f) << "at tick " << tick;
    }
    EXPECT_TRUE(engine.recorded_spikes.empty());
}

// A pulse of 1 nA for 2 ms into a 100 pF integrator moves it exactly 20 mV: 20 ticks of 1 mV. One
// tick too many or too few is a 5% error.
TEST(SingleCell, a_pulse_delivers_exactly_its_charge) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Pulse">
  <iafCell id="integrator" leakConductance="0 nS" leakReversal="-65 mV" thresh="1000 mV" reset="-70 mV" C="100 pF"/>
  <pulseGenerator id="pulse" delay="1 ms" duration="2 ms" amplitude="1 nA"/>
  <network id="pulseNetwork">
    <population id="pop" component="integrator" size="1"/>
    <explicitInput target="pop[0]" input="pulse"/>
  </network>
</neuroml>
)", "pulseNetwork", "5ms", "0.1ms"));
    engine.run();
    EXPECT_NEAR(engine.read_state_variable(0, "v"), -0.045f, 1e-6f);
}

// A spike train kicks the cell on exactly the ticks it names, and nothing else drives it.
TEST(SingleCell, a_spike_train_fires_the_cell_on_the_ticks_it_names) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Train">
  <iafCell id="c" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV" reset="-70 mV" C="100 pF"/>
  <spikeArray id="train">
    <spike id="0" time="10 ms"/>
    <spike id="1" time="25 ms"/>
    <spike id="2" time="40 ms"/>
  </spikeArray>
  <network id="trainNetwork">
    <population id="pop" component="c" size="1"/>
    <explicitInput target="pop[0]" input="train"/>
  </network>
</neuroml>
)", "trainNetwork", "60ms", "0.1ms"));
    ASSERT_EQ(engine.scheduled_spike_trains.size(), 1u);
    EXPECT_EQ(engine.scheduled_spike_trains[0].event_ticks, (Vector<s32>{100, 250, 400}));
    // 1.05 * C * (thresh - reset) / dt = 1.05 * 100 pF * 20 mV / 0.1 ms = 21 nA.
    EXPECT_NEAR(engine.scheduled_spike_trains[0].magnitude, 2.1e-8f, 1e-11f);

    engine.run();
    const Vector<f64> times = spike_times_of(engine, 0);
    ASSERT_EQ(times.size(), 3u);
    EXPECT_NEAR(times[0], 0.010, 1.5e-4);
    EXPECT_NEAR(times[1], 0.025, 1.5e-4);
    EXPECT_NEAR(times[2], 0.040, 1.5e-4);
}

// iafRefCell's refractory exit is "t .gt. lastSpikeTime + refract" with OnEntry stamping the time:
// the stamp form. The cell is held at reset for refract, then charges from reset.
TEST(SingleCell, a_time_stamped_refractory_period_holds_the_cell_at_reset) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Refractory">
  <iafRefCell id="c" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV" reset="-70 mV" C="100 pF"
              refract="5 ms"/>
  <pulseGenerator id="drive" delay="0 ms" duration="1000 ms" amplitude="90 pA"/>
  <network id="refractoryNetwork">
    <population id="pop" component="c" size="1"/>
    <explicitInput target="pop[0]" input="drive"/>
  </network>
</neuroml>
)", "refractoryNetwork", "400ms", "0.05ms"));
    engine.run();

    const Vector<f64> times = spike_times_of(engine, 0);
    ASSERT_GE(times.size(), 4u);
    EXPECT_NEAR(times[3] - times[2], 5e-3 + analytic_interspike_interval(), 3e-4);
}

// izhikevich2007Cell against forward Euler of its own equations, written here: the same number of
// spikes and the same mean interval. Spike times are not compared one for one.
TEST(SingleCell, izhikevich_matches_forward_euler) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Izhikevich">
  <izhikevich2007Cell id="c" C="100pF" v0="-60mV" k="0.7nS_per_mV" vr="-60mV" vt="-40mV" vpeak="35mV"
                      a="0.03per_ms" b="-2nS" c="-50mV" d="100pA"/>
  <pulseGenerator id="drive" delay="0 ms" duration="1000 ms" amplitude="200 pA"/>
  <network id="izhikevichNetwork">
    <population id="pop" component="c" size="1"/>
    <explicitInput target="pop[0]" input="drive"/>
  </network>
</neuroml>
)", "izhikevichNetwork", "300ms", "0.01ms"));
    engine.run();

    // v' = (k (v - vr)(v - vt) + I - u) / C, u' = a (b (v - vr) - u); past vpeak: v = c, u += d.
    const f64 capacitance = 100e-12, gain = 0.7e-6, resting = -0.060, threshold = -0.040, peak = 0.035;
    const f64 recovery_rate = 30.0, recovery_coupling = -2e-9, reset = -0.050, recovery_jump = 100e-12;
    const f64 drive = 200e-12, step = engine.step_dt;
    f64 voltage = -0.060, recovery = 0.0;
    Vector<f64> reference_spike_times;
    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        const f64 voltage_derivative = (gain * (voltage - resting) * (voltage - threshold) + drive - recovery) / capacitance;
        const f64 recovery_derivative = recovery_rate * (recovery_coupling * (voltage - resting) - recovery);
        voltage += step * voltage_derivative;
        recovery += step * recovery_derivative;
        if (voltage > peak) {
            voltage = reset;
            recovery += recovery_jump;
            reference_spike_times.push_back((f64)tick * step);
        }
    }

    const Vector<f64> spike_times = spike_times_of(engine, 0);
    ASSERT_GE(reference_spike_times.size(), 5u);
    EXPECT_NEAR((f64)spike_times.size(), (f64)reference_spike_times.size(), 1.0);
    const f64 reference_mean_interval = mean_of(interspike_intervals(reference_spike_times));
    EXPECT_NEAR(mean_of(interspike_intervals(spike_times)), reference_mean_interval, 0.01 * reference_mean_interval);
}

// adExIaFCell's w keeps evolving while the cell is refractory, and its OnEntry resets v and
// jumps w when it enters refractory.
TEST(SingleCell, adaptive_exponential_matches_forward_euler) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="AdEx">
  <adExIaFCell id="c" C="281pF" gL="30nS" EL="-70.6mV" VT="-50.4mV" thresh="-40.4mV" reset="-48.5mV"
               delT="2mV" tauw="144ms" refract="2.05ms" a="4nS" b="0.0805nA"/>
  <pulseGenerator id="drive" delay="0 ms" duration="1000 ms" amplitude="1 nA"/>
  <network id="adexNetwork">
    <population id="pop" component="c" size="1"/>
    <explicitInput target="pop[0]" input="drive"/>
  </network>
</neuroml>
)", "adexNetwork", "300ms", "0.01ms"));
    engine.run();

    // Integrating: v' = (-gL (v - EL) + gL delT exp((v - VT) / delT) - w + I) / C and
    // w' = (a (v - EL) - w) / tauw; past thresh: v = reset, w += b, then refractory for refract,
    // during which only w evolves.
    const f64 capacitance = 281e-12, leak = 30e-9, leak_reversal = -0.0706, slope_threshold = -0.0504;
    const f64 threshold = -0.0404, reset = -0.0485, slope = 0.002, adaptation_time = 0.144;
    const f64 refractory_period = 0.00205, coupling = 4e-9, adaptation_jump = 0.0805e-9, drive = 1e-9;
    const f64 step = engine.step_dt;
    f64 voltage = leak_reversal, adaptation = 0.0;
    s64 last_spike_tick = -1;
    Vector<f64> reference_spike_times;
    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        const bool refractory = last_spike_tick >= 0 && (f64)(tick - last_spike_tick) * step <= refractory_period;
        const f64 adaptation_derivative = (coupling * (voltage - leak_reversal) - adaptation) / adaptation_time;
        if (refractory) {
            adaptation += step * adaptation_derivative;
            continue;
        }
        const f64 voltage_derivative = (-leak * (voltage - leak_reversal) +
                                        leak * slope * std::exp((voltage - slope_threshold) / slope) - adaptation + drive) /
                                       capacitance;
        voltage += step * voltage_derivative;
        adaptation += step * adaptation_derivative;
        if (voltage > threshold) {
            voltage = reset;
            adaptation += adaptation_jump;
            last_spike_tick = tick;
            reference_spike_times.push_back((f64)tick * step);
        }
    }

    const Vector<f64> spike_times = spike_times_of(engine, 0);
    ASSERT_GE(reference_spike_times.size(), 5u);
    EXPECT_NEAR((f64)spike_times.size(), (f64)reference_spike_times.size(), 1.0);
    const f64 reference_mean_interval = mean_of(interspike_intervals(reference_spike_times));
    EXPECT_NEAR(mean_of(interspike_intervals(spike_times)), reference_mean_interval, 0.01 * reference_mean_interval);
    // Adaptation lengthens the intervals, so the run must not be a regular train.
    const Vector<f64> intervals = interspike_intervals(spike_times);
    EXPECT_GT(intervals.back(), intervals.front());
}

TEST(SingleCell, declared_output_files_have_the_shape_the_model_asked_for) {
    const TemporaryDirectory directory;
    const String spike_file = directory.path_of("spikes.dat");
    const String trace_file = directory.path_of("trace.dat");
    std::ostringstream outputs;
    outputs << "    <OutputFile id=\"trace\" fileName=\"" << trace_file << "\">\n"
            << "      <OutputColumn id=\"v\" quantity=\"cellPopulation[0]/v\"/>\n"
            << "    </OutputFile>\n"
            << "    <EventOutputFile id=\"spikes\" fileName=\"" << spike_file << "\" format=\"TIME_ID\">\n"
            << "      <EventSelection id=\"0\" select=\"cellPopulation[0]\" eventPort=\"spike\"/>\n"
            << "    </EventOutputFile>\n";
    SpikeEngine engine(write_model(directory, single_cell_model(), "singleCellNetwork", "200ms", "0.05ms", outputs.str()));
    engine.run();
    engine.write_recordings();

    // One row per tick: the time and v, which stays between reset and threshold.
    std::ifstream trace(trace_file);
    ASSERT_TRUE(trace.good());
    s64 trace_rows = 0;
    String line;
    while (getline(trace, line)) {
        if (line.empty()) continue;
        std::istringstream columns(line);
        f64 time_seconds = 0.0;
        f64 membrane_potential = 0.0;
        ASSERT_TRUE((columns >> time_seconds >> membrane_potential)) << line;
        EXPECT_GE(membrane_potential, -0.075);
        EXPECT_LE(membrane_potential, -0.045);
        trace_rows += 1;
    }
    EXPECT_EQ(trace_rows, engine.lifetime);

    std::ifstream spikes(spike_file);
    ASSERT_TRUE(spikes.good());
    s64 spike_rows = 0;
    while (getline(spikes, line)) spike_rows += line.empty() ? 0 : 1;
    EXPECT_EQ(spike_rows, (s64)engine.recorded_spikes.size());
    EXPECT_GT(spike_rows, 0);
}

// ── connections ─────────────────────────────────────────────────────────────────

// Ticks between a source spike and the first tick its target moves, beyond the connection delay:
//   +1  the arrival's OnEvent runs after that tick's Euler step, so I is still zero after it;
//   +1  an edge sends the current from the start of its tick, before its own step;
//   +1  the engine's input latency: a current sent on one tick is drained by the target the next.
// jNeuroML moves the target 1 tick after the delay on the same model (measured 2026-10-05).
constexpr s64 ARRIVAL_LATENCY_TICKS = 3;

// Cell 1 hears only cell 0, after 3 ms, and first moves 30 + ARRIVAL_LATENCY_TICKS after cell 0 fires.
TEST(Connections, a_delayed_connection_arrives_on_the_tick_it_says) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Delay">
  <alphaCurrentSynapse id="syn" tau="5 ms" ibase="12 pA"/>
  <iafCell id="driven" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV" reset="-70 mV" C="100 pF"/>
  <iafCell id="listener" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV" reset="-70 mV" C="100 pF"/>
  <pulseGenerator id="drive" delay="0 ms" duration="1000 ms" amplitude="90 pA"/>
  <network id="delayNetwork">
    <population id="popDriven" component="driven" size="1"/>
    <population id="popListener" component="listener" size="1"/>
    <projection id="proj" presynapticPopulation="popDriven" postsynapticPopulation="popListener" synapse="syn">
      <connectionWD id="0" preCellId="../popDriven[0]" postCellId="../popListener[0]" weight="1" delay="3 ms"/>
    </projection>
    <explicitInput target="popDriven[0]" input="drive"/>
  </network>
</neuroml>
)", "delayNetwork", "200ms", "0.1ms"));
    // The spike history is sized from the longest connection delay.
    EXPECT_EQ(engine.context.simulation.maximum_edge_delay, 30);
    EXPECT_EQ(engine.spike_history_row_count, 31);

    s64 first_source_spike_tick = -1;
    s64 first_listener_movement_tick = -1;
    for (s64 tick = 0; tick < engine.lifetime && first_listener_movement_tick < 0; tick += 1) {
        engine.step_simulation(tick);
        if (first_source_spike_tick < 0 && !engine.recorded_spikes.empty()) first_source_spike_tick = tick;
        if (first_source_spike_tick >= 0 && fabs(engine.read_state_variable(1, "v") - -0.065f) > 1e-9f) {
            first_listener_movement_tick = tick;
        }
    }
    ASSERT_GE(first_source_spike_tick, 0);
    ASSERT_GE(first_listener_movement_tick, 0);
    EXPECT_EQ(first_listener_movement_tick - first_source_spike_tick, 30 + ARRIVAL_LATENCY_TICKS);
}

// The host fills each edge's values by walking get_neighbors and the kernel finds them by walking
// k2t_next_neighbor; if the walks disagree, every edge silently gets another edge's weight and
// delay. One source, three targets with weights 1/2/3 and delays 1/3/5 ms, one presynaptic spike.
TEST(Connections, each_edge_gets_its_own_weight_and_delay) {
    const TemporaryDirectory directory;
    // The pulse stops at 40 ms, after the first threshold crossing (35.8 ms) and before the second.
    SpikeEngine engine(write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Slots">
  <alphaCurrentSynapse id="syn" tau="5 ms" ibase="12 pA"/>
  <iafCell id="c" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV" reset="-70 mV" C="100 pF"/>
  <pulseGenerator id="drive" delay="0 ms" duration="40 ms" amplitude="90 pA"/>
  <network id="slotNetwork">
    <population id="pop" component="c" size="8"/>
    <projection id="proj" presynapticPopulation="pop" postsynapticPopulation="pop" synapse="syn">
      <connectionWD id="0" preCellId="../pop[0]" postCellId="../pop[1]" weight="1" delay="1 ms"/>
      <connectionWD id="1" preCellId="../pop[0]" postCellId="../pop[4]" weight="2" delay="3 ms"/>
      <connectionWD id="2" preCellId="../pop[0]" postCellId="../pop[7]" weight="3" delay="5 ms"/>
    </projection>
    <explicitInput target="pop[0]" input="drive"/>
  </network>
</neuroml>
)", "slotNetwork", "120ms", "0.1ms"));

    const s64 targets[3] = {1, 4, 7};
    const s64 delay_ticks[3] = {10, 30, 50};
    const f32 resting = -0.065f;
    s64 spike_tick = -1;
    s64 first_movement_tick[3] = {-1, -1, -1};
    f32 peak_deflection[3] = {0.0f, 0.0f, 0.0f};
    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);
        if (spike_tick < 0 && !engine.recorded_spikes.empty()) spike_tick = tick;
        for (s64 index = 0; index < 3; index += 1) {
            const f32 deflection = engine.read_state_variable(targets[index], "v") - resting;
            if (first_movement_tick[index] < 0 && fabs(deflection) > 1e-9f) first_movement_tick[index] = tick;
            peak_deflection[index] = max(peak_deflection[index], deflection);
        }
    }
    ASSERT_EQ(engine.recorded_spikes.size(), 1u) << "the drive should produce one spike";

    for (s64 index = 0; index < 3; index += 1) {
        EXPECT_EQ(first_movement_tick[index] - spike_tick, delay_ticks[index] + ARRIVAL_LATENCY_TICKS)
            << "target " << targets[index];
    }
    // An alpha synapse's response is linear in the weight.
    EXPECT_NEAR(peak_deflection[1] / peak_deflection[0], 2.0, 0.02);
    EXPECT_NEAR(peak_deflection[2] / peak_deflection[0], 3.0, 0.02);
    for (s64 neuron = 1; neuron < 8; neuron += 1) {
        if (neuron == 1 || neuron == 4 || neuron == 7) continue;
        EXPECT_FLOAT_EQ(engine.read_state_variable(neuron, "v"), resting) << "neuron " << neuron;
    }
}

// Two prototypes of one synapse type differing only in the sign of ibase: reading the wrong one
// would flip excitation into inhibition.
TEST(Connections, an_edge_uses_its_own_synapse_prototype) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Prototypes">
  <alphaCurrentSynapse id="excitatory" tau="5 ms" ibase="12 pA"/>
  <alphaCurrentSynapse id="inhibitory" tau="5 ms" ibase="-12 pA"/>
  <iafCell id="c" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV" reset="-70 mV" C="100 pF"/>
  <pulseGenerator id="drive" delay="0 ms" duration="40 ms" amplitude="90 pA"/>
  <network id="prototypeNetwork">
    <population id="pop" component="c" size="4"/>
    <projection id="excitatoryProjection" presynapticPopulation="pop" postsynapticPopulation="pop" synapse="excitatory">
      <connectionWD id="0" preCellId="../pop[0]" postCellId="../pop[1]" weight="1" delay="1 ms"/>
    </projection>
    <projection id="inhibitoryProjection" presynapticPopulation="pop" postsynapticPopulation="pop" synapse="inhibitory">
      <connectionWD id="0" preCellId="../pop[0]" postCellId="../pop[2]" weight="1" delay="1 ms"/>
    </projection>
    <explicitInput target="pop[0]" input="drive"/>
  </network>
</neuroml>
)", "prototypeNetwork", "120ms", "0.1ms"));
    ASSERT_EQ(engine.context.simulation.synapse_instances.size(), 2u);

    const f32 resting = -0.065f;
    f32 excitatory_peak = 0.0f;
    f32 inhibitory_trough = 0.0f;
    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);
        excitatory_peak = max(excitatory_peak, engine.read_state_variable(1, "v") - resting);
        inhibitory_trough = min(inhibitory_trough, engine.read_state_variable(2, "v") - resting);
    }
    ASSERT_EQ(engine.recorded_spikes.size(), 1u);
    EXPECT_GT(excitatory_peak, 0.0f);
    EXPECT_LT(inhibitory_trough, 0.0f);
    EXPECT_NEAR(excitatory_peak, -inhibitory_trough, 1e-6f);
    EXPECT_FLOAT_EQ(engine.read_state_variable(3, "v"), resting);
}

// The spike history wraps every delay + 1 ticks, and a wrong modulo past the first wrap would
// deliver the first spike and drop or misplace later ones. One source firing across ~160 wraps, an
// edge at the worst-case delay, and a pure-integrator target that never fires: every onset of rising
// potential is one arrival, matched one for one with the source's spikes.
TEST(Connections, no_arrival_is_dropped_as_the_spike_history_wraps) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Wrap">
  <alphaCurrentSynapse id="syn" tau="2 ms" ibase="12 pA"/>
  <iafCell id="source" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV" reset="-70 mV" C="100 pF"/>
  <iafCell id="listener" leakConductance="0 nS" leakReversal="-65 mV" thresh="1000 mV" reset="-70 mV" C="100 pF"/>
  <pulseGenerator id="drive" delay="0 ms" duration="1000 ms" amplitude="90 pA"/>
  <network id="wrapNetwork">
    <population id="popSource" component="source" size="1"/>
    <population id="popListener" component="listener" size="1"/>
    <projection id="proj" presynapticPopulation="popSource" postsynapticPopulation="popListener" synapse="syn">
      <connectionWD id="0" preCellId="../popSource[0]" postCellId="../popListener[0]" weight="1" delay="3 ms"/>
    </projection>
    <explicitInput target="popSource[0]" input="drive"/>
  </network>
</neuroml>
)", "wrapNetwork", "500ms", "0.1ms"));
    const s64 delay_ticks = 30;
    ASSERT_EQ(engine.spike_history_row_count, delay_ticks + 1);
    EXPECT_GT(engine.lifetime / engine.spike_history_row_count, 100);

    // An arrival's first tick raises the listener by about 1.6e-6 V; the previous arrival's tail is
    // around 1e-14 V by then. The threshold sits well between.
    const f32 arrival_rise_threshold = 1e-8f;
    Vector<s64> source_spike_ticks;
    Vector<s64> arrival_onset_ticks;
    f32 previous_potential = engine.read_state_variable(1, "v");
    bool was_rising = false;
    usize spikes_seen = 0;
    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);
        while (spikes_seen < engine.recorded_spikes.size()) {
            source_spike_ticks.push_back(tick);
            spikes_seen += 1;
        }
        const f32 potential = engine.read_state_variable(1, "v");
        const bool rising = potential - previous_potential > arrival_rise_threshold;
        if (rising && !was_rising) arrival_onset_ticks.push_back(tick);
        was_rising = rising;
        previous_potential = potential;
    }
    ASSERT_GT(source_spike_ticks.size(), 8u);

    Vector<s64> arrivable_source_ticks;
    for (s64 spike_tick : source_spike_ticks) {
        if (spike_tick <= engine.lifetime - (delay_ticks + ARRIVAL_LATENCY_TICKS)) arrivable_source_ticks.push_back(spike_tick);
    }
    ASSERT_EQ(arrival_onset_ticks.size(), arrivable_source_ticks.size());
    for (usize index = 0; index < arrivable_source_ticks.size(); index += 1) {
        EXPECT_EQ(arrival_onset_ticks[index] - arrivable_source_ticks[index], delay_ticks + ARRIVAL_LATENCY_TICKS)
            << "arrival " << index;
    }
}

// A few distinct (weight, delay) runs fit exactly, and every synapse state variable has a plane.
TEST(Connections, weights_delays_and_synapse_state_live_in_the_weight_matrix) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Weights">
  <alphaCurrentSynapse id="syn" tau="5 ms" ibase="12 pA"/>
  <iafCell id="c" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV" reset="-70 mV" C="100 pF"/>
  <network id="weightNetwork">
    <population id="pop" component="c" size="4"/>
    <projection id="proj" presynapticPopulation="pop" postsynapticPopulation="pop" synapse="syn">
      <connectionWD id="0" preCellId="../pop[0]" postCellId="../pop[1]" weight="0.25" delay="1 ms"/>
      <connectionWD id="1" preCellId="../pop[0]" postCellId="../pop[2]" weight="1.75" delay="2 ms"/>
      <connectionWD id="2" preCellId="../pop[1]" postCellId="../pop[3]" weight="0.001" delay="3 ms"/>
    </projection>
  </network>
</neuroml>
)", "weightNetwork", "10ms", "0.1ms"));
    const WeightMatrix &weights = engine.weights;
    EXPECT_FLOAT_EQ(weights.get(0, 1), 0.25f);
    EXPECT_FLOAT_EQ(weights.get(0, 2), 1.75f);
    EXPECT_FLOAT_EQ(weights.get(1, 3), 0.001f);
    EXPECT_EQ(weights.get_edge_delay_ticks(0, 1), 10);
    EXPECT_EQ(weights.get_edge_delay_ticks(0, 2), 20);
    EXPECT_EQ(weights.get_edge_delay_ticks(1, 3), 30);
    EXPECT_EQ(weights.get_edge_synapse_prototype(0, 1), 0);
    EXPECT_EQ(weights.get_edge_synapse_prototype(3, 0), -1);

    // alphaCurrentSynapse carries I and J, one plane each after weight and delay, both at their
    // OnStart value of zero.
    EXPECT_EQ(weights.matrix_count, WeightMatrix::FIRST_STATE_VARIABLE_PLANE + 2);
    const s64 current_plane = engine.synapse_state_variable_plane("syn", "I");
    const s64 kick_plane = engine.synapse_state_variable_plane("syn", "J");
    EXPECT_NE(current_plane, kick_plane);
    EXPECT_GE(current_plane, WeightMatrix::FIRST_STATE_VARIABLE_PLANE);
    EXPECT_FLOAT_EQ(weights.get_for_matrix(0, 1, current_plane), 0.0f);
    EXPECT_FLOAT_EQ(weights.get_for_matrix(0, 1, kick_plane), 0.0f);
}

// ── spikes into a cell's own OnEvent ────────────────────────────────────────────

namespace {

// A presynaptic spike counts as an arrival on the tick its delay ends, and the target runs its
// OnEvent on the next tick, when it reads that tick's arrival count.
constexpr s64 EVENT_ARRIVAL_LATENCY_TICKS = 1;

// Driven iafCells (source_count of them, all firing together) connected with a 1 ms delay to one
// cell per target component. The target types are declared in cell_types.
String event_cell_model(const String &cell_types, const String &target_components, s64 source_count,
                        const Vector<String> &target_component_ids) {
    std::ostringstream document;
    document << R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="CellEvents">
)" << cell_types << R"(  <alphaCurrentSynapse id="syn" tau="5 ms" ibase="12 pA"/>
  <iafCell id="driven" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV" reset="-70 mV" C="100 pF"/>
)" << target_components << R"(  <pulseGenerator id="drive" delay="0 ms" duration="1000 ms" amplitude="90 pA"/>
  <network id="eventNetwork">
    <population id="sources" component="driven" size=")" << source_count << "\"/>\n";
    for (usize target = 0; target < target_component_ids.size(); target += 1) {
        document << "    <population id=\"targets" << target << "\" component=\"" << target_component_ids[target]
                 << "\" size=\"1\"/>\n";
    }
    for (usize target = 0; target < target_component_ids.size(); target += 1) {
        document << "    <projection id=\"projection" << target << "\" presynapticPopulation=\"sources\" "
                 << "postsynapticPopulation=\"targets" << target << "\" synapse=\"syn\">\n";
        for (s64 source = 0; source < source_count; source += 1) {
            document << "      <connectionWD id=\"" << source << "\" preCellId=\"../sources[" << source
                     << "]\" postCellId=\"../targets" << target << "[0]\" weight=\"1\" delay=\"1 ms\"/>\n";
        }
        document << "    </projection>\n";
    }
    for (s64 source = 0; source < source_count; source += 1) {
        document << "    <explicitInput target=\"sources[" << source << "]\" input=\"drive\"/>\n";
    }
    document << "  </network>\n</neuroml>\n";
    return document.str();
}

// A cell that only counts the spikes reaching it, and optionally relays each one as its own spike.
String counting_cell_type(bool relays_spikes) {
    return String(R"(  <ComponentType name="countingCell" extends="baseCellMembPot">
    <EventPort name="in" direction="in"/>
    <Attachments name="synapses" type="basePointCurrent"/>
    <Dynamics>
      <StateVariable name="v" dimension="voltage" exposure="v"/>
      <StateVariable name="received" dimension="none"/>
      <OnEvent port="in">
        <StateAssignment variable="received" value="received + 1"/>
)") + (relays_spikes ? "        <EventOut port=\"spike\"/>\n" : "") + R"(      </OnEvent>
    </Dynamics>
  </ComponentType>
)";
}

constexpr s64 DELAY_TICKS = 10;

// Each source spike's tick.
Vector<s64> spike_ticks_of(const SpikeEngine &engine, s64 neuron_index) {
    Vector<s64> ticks;
    for (f64 time : spike_times_of(engine, neuron_index)) ticks.push_back(llround(time / engine.step_dt));
    return ticks;
}

} // namespace

TEST(CellEvents, each_arrival_runs_the_on_event_once_on_the_tick_after_its_delay) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory,
                                   event_cell_model(counting_cell_type(false), "  <countingCell id=\"counter\"/>\n", 1, {"counter"}),
                                   "eventNetwork", "200ms", "0.1ms"));
    const s64 counter = 1;

    Vector<s64> count_change_ticks;
    f32 previous_count = 0.0f;
    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);
        const f32 count = engine.read_state_variable(counter, "received");
        if (count != previous_count) {
            EXPECT_EQ(count - previous_count, 1.0f) << "tick " << tick;
            count_change_ticks.push_back(tick);
        }
        previous_count = count;
    }

    Vector<s64> expected_ticks;
    for (s64 spike_tick : spike_ticks_of(engine, 0)) {
        const s64 arrival_tick = spike_tick + DELAY_TICKS + EVENT_ARRIVAL_LATENCY_TICKS;
        if (arrival_tick < engine.lifetime) expected_ticks.push_back(arrival_tick);
    }
    ASSERT_GE(expected_ticks.size(), 3u) << "the drive should fire the source several times";
    EXPECT_EQ(count_change_ticks, expected_ticks);
    EXPECT_FLOAT_EQ(engine.read_state_variable(counter, "v"), 0.0f);
}

// Two sources fire on the same tick, so two spikes reach the target together.
TEST(CellEvents, arrivals_on_the_same_tick_each_run_the_on_event) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory,
                                   event_cell_model(counting_cell_type(false), "  <countingCell id=\"counter\"/>\n", 2, {"counter"}),
                                   "eventNetwork", "200ms", "0.1ms"));
    const s64 counter = 2;

    f32 previous_count = 0.0f;
    s64 change_count = 0;
    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);
        const f32 count = engine.read_state_variable(counter, "received");
        if (count != previous_count) {
            EXPECT_EQ(count - previous_count, 2.0f) << "tick " << tick;
            change_count += 1;
        }
        previous_count = count;
    }
    ASSERT_EQ(spike_ticks_of(engine, 0), spike_ticks_of(engine, 1)) << "the two sources should fire together";
    EXPECT_GE(change_count, 3);
}

TEST(CellEvents, an_event_out_in_an_on_event_relays_each_arrival_as_a_spike) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory,
                                   event_cell_model(counting_cell_type(true), "  <countingCell id=\"relay\"/>\n", 1, {"relay"}),
                                   "eventNetwork", "200ms", "0.1ms"));
    engine.run();

    Vector<s64> expected_ticks;
    for (s64 spike_tick : spike_ticks_of(engine, 0)) {
        const s64 arrival_tick = spike_tick + DELAY_TICKS + EVENT_ARRIVAL_LATENCY_TICKS;
        if (arrival_tick < engine.lifetime) expected_ticks.push_back(arrival_tick);
    }
    ASSERT_GE(expected_ticks.size(), 3u);
    EXPECT_EQ(spike_ticks_of(engine, 1), expected_ticks);
}

// An OnEvent inside the integrating regime does not run while the cell is refractory. "held"
// fires on tick 0 and stays refractory for the whole run; "free" never fires.
TEST(CellEvents, an_on_event_inside_a_regime_runs_only_in_that_regime) {
    const TemporaryDirectory directory;
    const String gated_type = R"(  <ComponentType name="gatedCountingCell" extends="baseCellMembPot">
    <Parameter name="initialPhase" dimension="none"/>
    <Parameter name="refract" dimension="time"/>
    <EventPort name="in" direction="in"/>
    <Attachments name="synapses" type="basePointCurrent"/>
    <Dynamics>
      <StateVariable name="v" dimension="voltage" exposure="v"/>
      <StateVariable name="phase" dimension="none"/>
      <StateVariable name="received" dimension="none"/>
      <StateVariable name="lastSpikeTime" dimension="time"/>
      <OnStart>
        <StateAssignment variable="phase" value="initialPhase"/>
      </OnStart>
      <Regime name="integrating" initial="true">
        <OnCondition test="phase .gt. 1">
          <StateAssignment variable="phase" value="0"/>
          <EventOut port="spike"/>
          <Transition regime="refractory"/>
        </OnCondition>
        <OnEvent port="in">
          <StateAssignment variable="received" value="received + 1"/>
        </OnEvent>
      </Regime>
      <Regime name="refractory">
        <OnEntry>
          <StateAssignment variable="lastSpikeTime" value="t"/>
        </OnEntry>
        <OnCondition test="t .gt. lastSpikeTime + refract">
          <Transition regime="integrating"/>
        </OnCondition>
      </Regime>
    </Dynamics>
  </ComponentType>
)";
    SpikeEngine engine(write_model(directory,
                                   event_cell_model(gated_type,
                                                    "  <gatedCountingCell id=\"held\" initialPhase=\"2\" refract=\"1000 ms\"/>\n"
                                                    "  <gatedCountingCell id=\"free\" initialPhase=\"0\" refract=\"1000 ms\"/>\n",
                                                    1, {"held", "free"}),
                                   "eventNetwork", "200ms", "0.1ms"));
    engine.run();
    const s64 held_cell = 1;
    const s64 free_cell = 2;

    s64 arrivals = 0;
    for (s64 spike_tick : spike_ticks_of(engine, 0)) {
        if (spike_tick + DELAY_TICKS + EVENT_ARRIVAL_LATENCY_TICKS < engine.lifetime) arrivals += 1;
    }
    ASSERT_GE(arrivals, 3);
    EXPECT_EQ(spike_ticks_of(engine, held_cell), Vector<s64>{0});
    EXPECT_TRUE(spike_ticks_of(engine, free_cell).empty());
    EXPECT_FLOAT_EQ(engine.read_state_variable(held_cell, "received"), 0.0f);
    EXPECT_FLOAT_EQ(engine.read_state_variable(free_cell, "received"), (f32)arrivals);
}

// ── more than one cell type ─────────────────────────────────────────────────────

namespace {

// iafCell has one state variable and izhikevich2007Cell two, and their OnStart values differ, so a
// layout overlap shows up as one type's values in the other's slots.
String two_cell_type_model() {
    return R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="TwoTypes">
  <iafCell id="integrateAndFire" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV" reset="-70 mV" C="100 pF"/>
  <izhikevich2007Cell id="izhikevich" C="100pF" v0="-50mV" k="0.7nS_per_mV" vr="-60mV" vt="-40mV" vpeak="35mV"
                      a="0.03per_ms" b="-2nS" c="-50mV" d="100pA"/>
  <pulseGenerator id="iafDrive" delay="0 ms" duration="1000 ms" amplitude="90 pA"/>
  <pulseGenerator id="izhikevichDrive" delay="0 ms" duration="1000 ms" amplitude="200 pA"/>
  <network id="twoTypeNetwork">
    <population id="popIaf" component="integrateAndFire" size="4"/>
    <population id="popIzhikevich" component="izhikevich" size="3"/>
    <inputList id="iafInput" component="iafDrive" population="popIaf">
      <input id="0" target="../popIaf[0]" destination="synapses"/>
      <input id="1" target="../popIaf[1]" destination="synapses"/>
    </inputList>
    <inputList id="izhikevichInput" component="izhikevichDrive" population="popIzhikevich">
      <input id="0" target="../popIzhikevich[0]" destination="synapses"/>
    </inputList>
  </network>
</neuroml>
)";
}

} // namespace

TEST(CellTypes, each_type_gets_its_own_state_slots) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory, two_cell_type_model(), "twoTypeNetwork", "1ms", "0.01ms"));

    // Four one-slot cells then three two-slot cells: ten slots, not a shared maximum's fourteen.
    EXPECT_EQ(engine.context.get_cell_state_size(), 4 * 1 + 3 * 2);
    EXPECT_EQ(engine.context.simulation.population_base_indices.at("twoTypeNetwork/popIaf"), 0);
    EXPECT_EQ(engine.context.simulation.population_base_indices.at("twoTypeNetwork/popIzhikevich"), 4);

    for (s64 neuron = 0; neuron < 4; neuron += 1) {
        EXPECT_NEAR(engine.read_state_variable(neuron, "v"), -0.065f, 1e-7f) << "iafCell " << neuron;
    }
    for (s64 neuron = 4; neuron < 7; neuron += 1) {
        EXPECT_NEAR(engine.read_state_variable(neuron, "v"), -0.050f, 1e-7f) << "izhikevich " << neuron;
        EXPECT_NEAR(engine.read_state_variable(neuron, "u"), 0.0f, 1e-12f) << "izhikevich " << neuron;
    }
    // u belongs to one type only; asking an iafCell for it is an error, not a stray read.
    EXPECT_THROW((void)engine.read_state_variable(0, "u"), runtime_error);
}

TEST(CellTypes, each_type_integrates_its_own_equations) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory, two_cell_type_model(), "twoTypeNetwork", "60ms", "0.01ms"));
    engine.run();

    Vector<s64> spikes_by_population(2, 0);
    for (const RecordedSpike &spike : engine.recorded_spikes) spikes_by_population[spike.neuron_index < 4 ? 0 : 1] += 1;
    EXPECT_GT(spikes_by_population[0], 0) << "no iafCell fired";
    EXPECT_GT(spikes_by_population[1], 0) << "no izhikevich2007Cell fired";

    // An undriven iafCell sits at a fixed point and must not move at all.
    EXPECT_NEAR(engine.read_state_variable(2, "v"), -0.065f, 1e-7f);
    EXPECT_NEAR(engine.read_state_variable(3, "v"), -0.065f, 1e-7f);
    // An undriven izhikevich2007Cell relaxes from v0 = -50 mV toward vr = -60 mV.
    EXPECT_NEAR(engine.read_state_variable(6, "v"), -0.060f, 5e-4f);
    // Two undriven izhikevich cells with identical everything stay identical to the bit.
    EXPECT_FLOAT_EQ(engine.read_state_variable(5, "v"), engine.read_state_variable(6, "v"));
    EXPECT_FLOAT_EQ(engine.read_state_variable(5, "u"), engine.read_state_variable(6, "u"));
    // The driven one's recovery variable moved.
    EXPECT_NE(engine.read_state_variable(4, "u"), 0.0f);
}

// ── the GLIF family ─────────────────────────────────────────────────────────────

// One cell of each GLIF type under the same step. GLIF1's interval has a closed form including its
// refractory timer; the adapting types slow down; no interval anywhere is shorter than t_ref.
TEST(GlifFamily, each_type_produces_its_defining_behaviour) {
    SpikeEngine engine(fixture_path("nml/LEMS_glif_family.xml"));
    ASSERT_EQ(engine.total_neuron_count, 5);
    engine.run();

    // tau = 10 ms, v_inf = EL + I/gL = -20 mV: charging from -70 to -50 mV takes tau ln(50/30), then t_ref.
    const f64 charging_time = 10e-3 * std::log((-0.020 - -0.070) / (-0.020 - -0.050));
    const Vector<f64> glif1 = spike_times_of(engine, 0);
    ASSERT_GE(glif1.size(), 5u);
    EXPECT_NEAR(glif1[3] - glif1[2], charging_time + 5e-3, 2e-4);

    for (s64 cell = 0; cell < 5; cell += 1) {
        const Vector<f64> times = spike_times_of(engine, cell);
        ASSERT_GE(times.size(), 5u) << "cell " << cell;
        for (f64 interval : interspike_intervals(times)) EXPECT_GE(interval, 5e-3 - 1e-4) << "cell " << cell;
    }

    auto adaptation_ratio = [&](s64 cell) {
        const Vector<f64> intervals = interspike_intervals(spike_times_of(engine, cell));
        return intervals.back() / intervals.front();
    };
    EXPECT_NEAR(adaptation_ratio(0), 1.0, 0.05) << "GLIF1 should not adapt";
    EXPECT_NEAR(adaptation_ratio(1), 1.0, 0.05) << "GLIF2 should not adapt";
    EXPECT_GT(adaptation_ratio(2), 1.5) << "GLIF3's after-spike currents should adapt";
    EXPECT_GT(adaptation_ratio(3), 1.2) << "GLIF4's threshold should adapt";
    EXPECT_GT(adaptation_ratio(4), 1.5) << "GLIF5 should adapt";

    EXPECT_LT(engine.read_state_variable(2, "asc1"), 0.0f);
    EXPECT_GT(engine.read_state_variable(3, "theta"), -0.050f);
}

// GLIF2 resets to vreset + resetScale (v - vth). Under a drive that overshoots threshold by ~10 mV
// per tick, a higher resetScale resets nearer threshold and fires more often.
TEST(GlifFamily, glif2_reset_scale_changes_how_often_the_cell_fires) {
    const TemporaryDirectory directory;
    SpikeEngine engine(write_model(directory, R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ResetScale">
)" + glif_types_include() + R"(  <GLIF2Cell id="discardOvershoot" C="100pF" gL="10nS" EL="-70mV" vth="-50mV" vreset="-70mV"
             resetScale="0" t_ref="1ms"/>
  <GLIF2Cell id="carryOvershoot" C="100pF" gL="10nS" EL="-70mV" vth="-50mV" vreset="-70mV"
             resetScale="0.9" t_ref="1ms"/>
  <pulseGenerator id="hardDrive" delay="0ms" duration="200ms" amplitude="10nA"/>
  <network id="resetScaleNetwork">
    <population id="popDiscard" component="discardOvershoot" size="1"/>
    <population id="popCarry" component="carryOvershoot" size="1"/>
    <explicitInput target="popDiscard[0]" input="hardDrive"/>
    <explicitInput target="popCarry[0]" input="hardDrive"/>
  </network>
</neuroml>
)", "resetScaleNetwork", "200ms", "0.1ms"));
    engine.run();

    const Vector<f64> discarding = spike_times_of(engine, 0);
    const Vector<f64> carrying = spike_times_of(engine, 1);
    ASSERT_GE(discarding.size(), 10u);
    ASSERT_GE(carrying.size(), 10u);
    EXPECT_GT(carrying.size(), discarding.size());
    EXPECT_GT((discarding[5] - discarding[4]) - (carrying[5] - carrying[4]), 5e-5);
}

// ── connectivity from a topology ────────────────────────────────────────────────

namespace {

// A GLIF1 sheet whose document declares no connections: they come from a topology helper.
String torus_model(s64 neuron_count) {
    std::ostringstream document;
    document << "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"Torus\">\n" << glif_types_include()
             << R"(  <alphaCurrentSynapse id="torusSynapse" tau="5 ms" ibase="30 pA"/>
  <alphaCurrentSynapse id="otherSynapse" tau="5 ms" ibase="-30 pA"/>
  <GLIF1Cell id="torusCell" C="100pF" gL="10nS" EL="-70mV" vreset="-70mV" t_ref="5ms" vth="-50mV"/>
  <pulseGenerator id="torusDrive" delay="0 ms" duration="1000 ms" amplitude="260 pA"/>
  <network id="torusNetwork">
    <population id="torusPopulation" component="torusCell" size=")"
             << neuron_count << R"("/>
    <inputList id="torusInput" component="torusDrive" population="torusPopulation">
)";
    for (s64 index = 0; index < neuron_count; index += 1) {
        document << "      <input id=\"" << index << "\" target=\"../torusPopulation[" << index
                 << "]\" destination=\"synapses\"/>\n";
    }
    document << "    </inputList>\n  </network>\n</neuroml>\n";
    return document.str();
}

} // namespace

TEST(Topology, connectivity_can_come_from_a_topology_instead_of_the_document) {
    const s64 side = 16;
    const s64 neuron_count = side * side;
    const TemporaryDirectory directory;
    const vector<vector<s32>> adjacency = square_torus(side);
    SpikeEngine engine(write_model(directory, torus_model(neuron_count), "torusNetwork", "200ms", "0.1ms"), adjacency,
                       "torusSynapse", /*connection_weight=*/1.0, /*connection_delay_seconds=*/1e-3);

    EXPECT_EQ(engine.total_neuron_count, neuron_count);
    EXPECT_EQ(engine.weights.total_edge_count, neuron_count * 4);
    for (s64 source = 0; source < neuron_count; source += 1) {
        for (s32 target : adjacency[(usize)source]) {
            EXPECT_FLOAT_EQ(engine.weights.get((s32)source, target), 1.0f) << source << " -> " << target;
            EXPECT_EQ(engine.weights.get_edge_delay_ticks((s32)source, target), 10) << source << " -> " << target;
        }
    }

    engine.run();
    EXPECT_GT(engine.mean_firing_rate_hertz(), 2.0);
    EXPECT_GT(engine.fraction_of_neurons_that_spiked(), 0.9);
}

TEST(Topology, a_topology_that_does_not_match_the_model_is_refused) {
    const TemporaryDirectory directory;
    const String lems = write_model(directory, torus_model(16), "torusNetwork", "10ms", "0.1ms");
    try {
        SpikeEngine engine(lems, square_torus(8), "torusSynapse");
        FAIL() << "a topology larger than the network must be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("64"), String::npos) << message;
        EXPECT_NE(message.find("16"), String::npos) << message;
    }
    try {
        SpikeEngine engine(lems, square_torus(4), "noSuchSynapse");
        FAIL() << "an unknown synapse must be refused";
    } catch (const runtime_error &error) {
        EXPECT_NE(String(error.what()).find("noSuchSynapse"), String::npos) << error.what();
    }
}

// The edges are split, in adjacency order, into one contiguous share per synapse.
TEST(Topology, synapse_shares_split_the_edges_in_order) {
    const TemporaryDirectory directory;
    const String lems = write_model(directory, torus_model(16), "torusNetwork", "10ms", "0.1ms");
    const vector<vector<s32>> adjacency = random_fixed_outdegree(4, 5, 3);  // 80 edges

    const auto prototype_counts = [&](const SpikeEngine &engine) {
        Vector<s64> counts(2, 0);
        s64 last_prototype = 0;
        for (s64 source = 0; source < 16; source += 1) {
            for (s32 target : adjacency[(usize)source]) {
                const s32 prototype = engine.weights.get_edge_synapse_prototype((s32)source, target);
                EXPECT_GE(prototype, last_prototype) << "the shares must be contiguous";
                last_prototype = prototype;
                counts[(usize)prototype] += 1;
            }
        }
        return counts;
    };

    SpikeEngine weighted(lems, adjacency, vector<String>{"torusSynapse", "otherSynapse"}, vector<f64>{3.0, 1.0});
    EXPECT_EQ(prototype_counts(weighted), (Vector<s64>{60, 20}));

    SpikeEngine equal(lems, adjacency, vector<String>{"torusSynapse", "otherSynapse"});
    EXPECT_EQ(prototype_counts(equal), (Vector<s64>{40, 40}));
}

// ── randomness ──────────────────────────────────────────────────────────────────

namespace {

String generator_model() {
    return R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Generators">
  <spikeGeneratorPoisson id="poissonSource" averageRate="50 Hz"/>
  <spikeGeneratorRandom id="uniformSource" minISI="10 ms" maxISI="30 ms"/>
  <network id="generatorNetwork">
    <population id="poissonPopulation" component="poissonSource" size="200"/>
    <population id="uniformPopulation" component="uniformSource" size="200"/>
  </network>
</neuroml>
)";
}

String seeded_generator_lems(const TemporaryDirectory &directory, const String &seed) {
    directory.write("generators.nml", generator_model());
    return directory.write("LEMS_" + seed + ".xml", R"(<Lems>
  <Include file="Cells.xml"/><Include file="Networks.xml"/><Include file="Simulation.xml"/>
  <Include file="generators.nml"/>
  <Simulation id="sim1" length="2s" step="0.1ms" target="generatorNetwork" seed=")" + seed + R"("/>
  <Target component="sim1"/>
</Lems>
)");
}

// Every interspike interval of neurons [first, last).
Vector<f64> intervals_of(const SpikeEngine &engine, s64 first, s64 last) {
    Vector<f64> intervals;
    for (s64 neuron = first; neuron < last; neuron += 1) {
        const Vector<f64> neuron_intervals = interspike_intervals(spike_times_of(engine, neuron));
        intervals.insert(intervals.end(), neuron_intervals.begin(), neuron_intervals.end());
    }
    return intervals;
}

} // namespace

// A Poisson source has exponential intervals: coefficient of variation 1. H(0) = 1 used to stall it
// for about t whenever its next spike time equalled t, which shows as a CV above 1 and long gaps.
TEST(Randomness, a_poisson_generator_fires_at_its_rate_with_poisson_intervals) {
    const TemporaryDirectory directory;
    SpikeEngine engine(seeded_generator_lems(directory, "1234"));
    engine.run();

    const Vector<f64> intervals = intervals_of(engine, 0, 200);
    ASSERT_GT(intervals.size(), 15000u);
    EXPECT_NEAR(1.0 / mean_of(intervals), 50.0, 1.5);
    EXPECT_NEAR(coefficient_of_variation(intervals), 1.0, 0.05);
    EXPECT_LT(*max_element(intervals.begin(), intervals.end()), 0.5) << "a Poisson source stalled";
}

TEST(Randomness, a_uniform_interval_generator_keeps_its_intervals_in_range) {
    const TemporaryDirectory directory;
    SpikeEngine engine(seeded_generator_lems(directory, "1234"));
    engine.run();

    const Vector<f64> intervals = intervals_of(engine, 200, 400);
    ASSERT_GT(intervals.size(), 15000u);
    const f64 tick = engine.step_dt;
    EXPECT_GE(*min_element(intervals.begin(), intervals.end()), 10e-3 - tick);
    EXPECT_LE(*max_element(intervals.begin(), intervals.end()), 30e-3 + tick);
    EXPECT_NEAR(mean_of(intervals), 20e-3, 0.2e-3);
}

TEST(Randomness, the_same_seed_reproduces_a_run_and_another_seed_does_not) {
    const TemporaryDirectory directory;
    const auto spikes_of = [&](const String &seed) {
        SpikeEngine engine(seeded_generator_lems(directory, seed));
        engine.run();
        Vector<Pair<f64, s64>> spikes;
        for (const RecordedSpike &spike : engine.recorded_spikes) spikes.push_back({spike.time_seconds, spike.neuron_index});
        return spikes;
    };
    const Vector<Pair<f64, s64>> first = spikes_of("1234");
    EXPECT_EQ(first, spikes_of("1234"));
    EXPECT_NE(first, spikes_of("99"));
}

// ── recurrent GLIF networks ─────────────────────────────────────────────────────

namespace {

struct NetworkActivity {
    f64 mean_rate = 0.0;
    f64 participation = 0.0;
    f64 peak_synchrony = 0.0;
    f64 active_tick_fraction = 0.0;
    f64 middle_rate = 0.0;
    f64 final_rate = 0.0;
};

NetworkActivity measure_activity(const SpikeEngine &engine, f64 total_seconds) {
    NetworkActivity activity;
    activity.mean_rate = engine.mean_firing_rate_hertz();
    activity.participation = engine.fraction_of_neurons_that_spiked();

    Vector<s64> spikes_per_tick((usize)engine.lifetime, 0);
    for (const RecordedSpike &spike : engine.recorded_spikes) {
        const s64 tick = llround(spike.time_seconds / engine.step_dt);
        if (tick >= 0 && tick < engine.lifetime) spikes_per_tick[(usize)tick] += 1;
    }
    activity.peak_synchrony =
            (f64)*max_element(spikes_per_tick.begin(), spikes_per_tick.end()) / (f64)engine.total_neuron_count;
    activity.active_tick_fraction =
            (f64)count_if(spikes_per_tick.begin(), spikes_per_tick.end(), [](s64 count) { return count > 0; }) /
            (f64)engine.lifetime;

    auto rate_over = [&](f64 from_seconds, f64 to_seconds) {
        const s64 spike_count = count_if(engine.recorded_spikes.begin(), engine.recorded_spikes.end(),
                                         [&](const RecordedSpike &spike) {
                                             return spike.time_seconds >= from_seconds && spike.time_seconds < to_seconds;
                                         });
        return (f64)spike_count / ((f64)engine.total_neuron_count * (to_seconds - from_seconds));
    };
    activity.middle_rate = rate_over(0.4 * total_seconds, 0.6 * total_seconds);
    activity.final_rate = rate_over(0.8 * total_seconds, total_seconds);
    return activity;
}

// C = 100 pF and gL = 10 nS: a 10 ms time constant and a 200 pA rheobase.
String glif_cell_attributes_for(s32 glif_index) {
    const String shared = R"( C="100pF" gL="10nS" EL="-70mV" vreset="-70mV" t_ref="5ms")";
    switch (glif_index) {
        case 1: return shared + R"( vth="-50mV")";
        case 2: return shared + R"( vth="-50mV" resetScale="0.3")";
        case 3: return shared + R"( vth="-50mV" tauAsc1="100ms" tauAsc2="10ms" ascAdd1="-60pA" ascAdd2="-120pA")";
        case 4: return shared + R"( thetaInf="-50mV" tauTheta="50ms" thetaSpikeAdd="3mV")";
        case 5: return shared + R"( thetaInf="-50mV" tauTheta="50ms" thetaSpikeAdd="3mV" tauAsc1="100ms")"
                                R"( tauAsc2="10ms" ascAdd1="-60pA" ascAdd2="-120pA")";
        default: throw runtime_error("glif_cell_attributes_for: index must be 1..5");
    }
}

// 484 cells, 20 random outgoing edges each, split 4:1 excitatory to inhibitory, a fifth of the cells
// driven above rheobase and the rest just below it.
unique_ptr<SpikeEngine> make_glif_network(const TemporaryDirectory &directory, s32 glif_index, f64 simulation_seconds) {
    const s64 side_length = 22;
    const s64 cell_count = side_length * side_length;
    const u64 seed = 20260813;

    set<s64> seeded;
    mt19937_64 generator(seed);
    uniform_int_distribution<s64> anywhere(0, cell_count - 1);
    while ((s64)seeded.size() < cell_count / 5) seeded.insert(anywhere(generator));

    std::ostringstream model;
    model << "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"glifNetwork\">\n" << glif_types_include()
          << "  <alphaCurrentSynapse id=\"excitatorySynapse\" tau=\"5 ms\" ibase=\"16 pA\"/>\n"
          << "  <alphaCurrentSynapse id=\"inhibitorySynapse\" tau=\"5 ms\" ibase=\"-35 pA\"/>\n"
          << "  <GLIF" << glif_index << "Cell id=\"networkCell\"" << glif_cell_attributes_for(glif_index) << "/>\n"
          << "  <pulseGenerator id=\"background\" delay=\"0 s\" duration=\"" << simulation_seconds
          << " s\" amplitude=\"190 pA\"/>\n"
          << "  <pulseGenerator id=\"seedDrive\" delay=\"0 s\" duration=\"" << simulation_seconds
          << " s\" amplitude=\"235 pA\"/>\n"
          << "  <network id=\"network\">\n"
          << "    <population id=\"population\" component=\"networkCell\" size=\"" << cell_count << "\"/>\n";
    for (s64 index = 0; index < cell_count; index += 1) {
        model << "    <explicitInput target=\"population[" << index << "]\" input=\""
              << (seeded.count(index) ? "seedDrive" : "background") << "\"/>\n";
    }
    model << "  </network>\n</neuroml>\n";

    std::ostringstream length;
    length << simulation_seconds << "s";
    const String lems = write_model(directory, model.str(), "network", length.str(), "0.1ms");
    return make_unique<SpikeEngine>(lems, random_fixed_outdegree(side_length, 20, (s64)seed),
                                    vector<String>{"excitatorySynapse", "inhibitorySynapse"}, vector<f64>{0.8, 0.2},
                                    /*connection_weight=*/1.0, /*connection_delay_seconds=*/2e-3);
}

} // namespace

// Alive on the same terms for every GLIF type: a rate in a sensible band, nearly every cell taking
// part, activity spread over time rather than locked into volleys, and still firing at the end.
class GlifNetworkAliveness : public ::testing::TestWithParam<s32> {};

TEST_P(GlifNetworkAliveness, sustains_asynchronous_recurrent_activity) {
    const s32 glif_index = GetParam();
    const f64 simulation_seconds = 2.0;
    const TemporaryDirectory directory;
    unique_ptr<SpikeEngine> engine = make_glif_network(directory, glif_index, simulation_seconds);
    EXPECT_EQ(engine->total_neuron_count, 484);
    EXPECT_EQ(engine->weights.total_edge_count, 484 * 20);
    engine->run();

    const NetworkActivity activity = measure_activity(*engine, simulation_seconds);
    EXPECT_GT(activity.mean_rate, 2.0) << "GLIF" << glif_index << " is barely firing";
    EXPECT_LT(activity.mean_rate, 60.0) << "GLIF" << glif_index << " is saturated";
    EXPECT_GT(activity.participation, 0.85);
    // Lockstep puts every spike on a handful of ticks. The adapting types band at about 10 Hz, which
    // lands near 19% of ticks active, so the floor is 5%.
    EXPECT_GT(activity.active_tick_fraction, 0.05) << "the population is firing in lockstep";
    EXPECT_LT(activity.peak_synchrony, 0.5);
    ASSERT_GT(activity.middle_rate, 0.0);
    EXPECT_GT(activity.final_rate, 0.6 * activity.middle_rate);
    EXPECT_GT(activity.final_rate, 2.0);
}

INSTANTIATE_TEST_SUITE_P(AllFiveTypes, GlifNetworkAliveness, ::testing::Values(1, 2, 3, 4, 5),
                         [](const ::testing::TestParamInfo<s32> &info) { return "GLIF" + to_string(info.param); });

// The types that carry adaptation state fire more slowly in a network under identical drive and
// wiring: the extra state does something once cells are connected.
TEST(GlifNetworks, adapting_types_settle_below_non_adapting_ones) {
    Vector<f64> rate_by_type(6, 0.0);
    for (s32 glif_index : {1, 3, 5}) {
        const TemporaryDirectory directory;
        unique_ptr<SpikeEngine> engine = make_glif_network(directory, glif_index, 1.0);
        engine->run();
        rate_by_type[(usize)glif_index] = engine->mean_firing_rate_hertz();
    }
    EXPECT_LT(rate_by_type[3], 0.75 * rate_by_type[1]) << "GLIF3 " << rate_by_type[3] << " Hz, GLIF1 " << rate_by_type[1];
    EXPECT_LT(rate_by_type[5], 0.75 * rate_by_type[1]) << "GLIF5 " << rate_by_type[5] << " Hz, GLIF1 " << rate_by_type[1];
}
