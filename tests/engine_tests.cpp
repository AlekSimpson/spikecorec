// Tests for the NeuroML-driven engine: expression lowering, the generated kernel, and
// what the simulation actually does when it runs.
//
// The simulation tests are written against quantities that can be derived by hand from the
// model — an analytic interspike interval, an exact arrival tick, a firing-rate band — so
// that a kernel which silently integrates the wrong equation fails here rather than
// producing a plausible-looking recording that nobody can check.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <gtest/gtest.h>

#include "spikecorec/core/engine.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/nml/dynamics_codegen.h"
#include "../examples/balanced_network_model.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

namespace {

class ModelDirectory {
public:
    explicit ModelDirectory(const String &test_name) {
        root_ = filesystem::temp_directory_path() / "spikecorec_engine_tests" / test_name;
        filesystem::remove_all(root_);
        filesystem::create_directories(root_);
    }

    ~ModelDirectory() {
        std::error_code ignored;
        filesystem::remove_all(root_, ignored);
    }

    ModelDirectory(const ModelDirectory &) = delete;
    ModelDirectory &operator=(const ModelDirectory &) = delete;

    String write(const String &name, const String &contents) const {
        const filesystem::path destination = root_ / name;
        ofstream file(destination);
        file << contents;
        file.close();
        return destination.string();
    }

    [[nodiscard]] String path() const { return root_.string(); }

private:
    filesystem::path root_;
};

bool standard_library_available() {
    NML_Parser parser;
    return !parser.STANDARD_LIBRARY_PATH.empty() &&
           filesystem::exists(parser.STANDARD_LIBRARY_PATH);
}

// Everything a Simulation needs around a model document, so each test writes only the
// part it is about.
String lems_wrapper(const String &model_file, const String &network_id,
                    const String &length, const String &step,
                    const String &extra_outputs = "") {
    ostringstream document;
    document << "<Lems>\n"
             << "  <Include file=\"Cells.xml\"/>\n"
             << "  <Include file=\"Synapses.xml\"/>\n"
             << "  <Include file=\"Inputs.xml\"/>\n"
             << "  <Include file=\"Networks.xml\"/>\n"
             << "  <Include file=\"Simulation.xml\"/>\n"
             << "  <Include file=\"" << model_file << "\"/>\n"
             << "  <Simulation id=\"sim1\" length=\"" << length << "\" step=\"" << step
             << "\" target=\"" << network_id << "\">\n"
             << extra_outputs
             << "  </Simulation>\n"
             << "  <Target component=\"sim1\"/>\n"
             << "</Lems>\n";
    return document.str();
}

// One iafCell under a constant supra-rheobase current. tau = C/gL = 20 ms and the
// rheobase is gL*(thresh - leakReversal) = 75 pA, so at 90 pA the cell fires periodically
// with an interval that has a closed form.
String single_cell_model(const String &amplitude = "90 pA") {
    return R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="SingleCell">
  <iafCell id="testCell" leakConductance="5 nS" leakReversal="-65 mV"
           thresh="-50 mV" reset="-70 mV" C="100 pF"/>
  <pulseGenerator id="drive" delay="0 ms" duration="1000 ms" amplitude=")" + amplitude + R"("/>
  <network id="singleCellNetwork">
    <population id="cellPopulation" component="testCell" size="1"/>
    <explicitInput target="cellPopulation[0]" input="drive"/>
  </network>
</neuroml>
)";
}

f64 analytic_interspike_interval() {
    const f64 membrane_time_constant = 100e-12 / 5e-9;
    const f64 drive_in_volts = 90e-12 / 5e-9;
    return membrane_time_constant *
           std::log((drive_in_volts - (-0.070 - -0.065)) /
                    (drive_in_volts - (-0.050 - -0.065)));
}

} // namespace

// ── expression lowering ───────────────────────────────────────────────────────────

TEST(DynamicsCodegen, dotted_operators_lower_to_their_c_spelling) {
    const SymbolTable symbols = {{"v", "state_0"}, {"thresh", "parameter_0"}};

    EXPECT_EQ(translate_expression("v .gt. thresh", symbols, "test"),
              "(state_0 > parameter_0)");
    EXPECT_EQ(translate_expression("v .lt. thresh", symbols, "test"),
              "(state_0 < parameter_0)");
    EXPECT_EQ(translate_expression("v .geq. thresh", symbols, "test"),
              "(state_0 >= parameter_0)");
    EXPECT_EQ(translate_expression("v .leq. thresh", symbols, "test"),
              "(state_0 <= parameter_0)");
    EXPECT_EQ(translate_expression("v .eq. thresh", symbols, "test"),
              "(state_0 == parameter_0)");
    EXPECT_EQ(translate_expression("v .neq. thresh", symbols, "test"),
              "(state_0 != parameter_0)");
    EXPECT_EQ(translate_expression("v .gt. thresh .and. v .lt. thresh", symbols, "test"),
              "((state_0 > parameter_0) && (state_0 < parameter_0))");
    EXPECT_EQ(translate_expression("v .gt. thresh .or. v .lt. thresh", symbols, "test"),
              "((state_0 > parameter_0) || (state_0 < parameter_0))");
}

// LEMS `ln` is natural log and LEMS `log` is base 10, which is the reverse of what the
// names suggest in C. Getting either backwards changes every rate law that uses them and
// nothing about it is a compile error.
TEST(DynamicsCodegen, lems_function_names_map_to_the_right_c_functions) {
    const SymbolTable symbols = {{"x", "state_0"}};

    EXPECT_EQ(translate_expression("ln(x)", symbols, "test"), "log(state_0)");
    EXPECT_EQ(translate_expression("log(x)", symbols, "test"), "log10(state_0)");
    EXPECT_EQ(translate_expression("abs(x)", symbols, "test"), "fabs(state_0)");
    EXPECT_EQ(translate_expression("exp(x)", symbols, "test"), "exp(state_0)");
    EXPECT_EQ(translate_expression("H(x)", symbols, "test"), "spikecorec_heaviside(state_0)");
}

TEST(DynamicsCodegen, precedence_and_associativity_follow_the_arithmetic) {
    const SymbolTable symbols = {{"a", "A"}, {"b", "B"}, {"c", "C"}};

    EXPECT_EQ(translate_expression("a + b * c", symbols, "test"), "(A + (B * C))");
    EXPECT_EQ(translate_expression("(a + b) * c", symbols, "test"), "((A + B) * C)");
    EXPECT_EQ(translate_expression("a - b - c", symbols, "test"), "((A - B) - C)");
    EXPECT_EQ(translate_expression("a * (b + c)", symbols, "test"), "(A * (B + C))");
    EXPECT_EQ(translate_expression("((a))", symbols, "test"), "A");

    // `^` binds tighter than * and is right-associative, so a^b^c is a^(b^c).
    EXPECT_EQ(translate_expression("a ^ b ^ c", symbols, "test"), "pow(A, pow(B, C))");
    EXPECT_EQ(translate_expression("a * b ^ c", symbols, "test"), "(A * pow(B, C))");
}

// An integer literal that reaches the target as an integer turns division into integer
// division: `1/tau` would evaluate to 0 rather than to a rate.
TEST(DynamicsCodegen, integer_literals_are_emitted_as_floating_point) {
    const SymbolTable symbols = {{"tau", "parameter_0"}};

    EXPECT_EQ(translate_expression("1 / tau", symbols, "test"), "(1.0 / parameter_0)");
    EXPECT_EQ(translate_expression("2.5e-3 * tau", symbols, "test"),
              "(2.5e-3 * parameter_0)");
}

TEST(DynamicsCodegen, an_unresolved_name_is_an_error_not_a_pass_through) {
    const SymbolTable symbols = {{"v", "state_0"}};

    EXPECT_THROW(translate_expression("v + mystery", symbols, "someCell"), std::runtime_error);
    EXPECT_THROW(translate_expression("notAFunction(v)", symbols, "someCell"),
                 std::runtime_error);
    EXPECT_THROW(translate_expression("v +", symbols, "someCell"), std::runtime_error);
    EXPECT_THROW(translate_expression("(v", symbols, "someCell"), std::runtime_error);

    try {
        translate_expression("v + mystery", symbols, "someCell");
        FAIL() << "expected a throw";
    } catch (const std::runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("mystery"), String::npos);
        EXPECT_NE(message.find("someCell"), String::npos);
    }
}

TEST(DynamicsCodegen, phase_two_functions_say_so) {
    const SymbolTable symbols = {{"v", "state_0"}};
    try {
        translate_expression("random(v)", symbols, "someCell");
        FAIL() << "expected a throw";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(String(error.what()).find("Phase 2"), String::npos);
    }
}

TEST(DynamicsCodegen, on_start_values_fold_to_literals_or_parameters) {
    const Vector<String> names = {"leakReversal", "thresh"};
    Vector<Real> values(2);
    values[0].float64 = -0.065;
    values[1].float64 = -0.050;

    EXPECT_DOUBLE_EQ(evaluate_initial_value("0", names, values, "cell"), 0.0);
    EXPECT_DOUBLE_EQ(evaluate_initial_value("leakReversal", names, values, "cell"), -0.065);
    EXPECT_DOUBLE_EQ(evaluate_initial_value("-leakReversal", names, values, "cell"), 0.065);
    EXPECT_DOUBLE_EQ(evaluate_initial_value("1.5e-3", names, values, "cell"), 1.5e-3);

    EXPECT_THROW(evaluate_initial_value("thresh - 1", names, values, "cell"),
                 std::runtime_error);
}

// ── the generated kernel ──────────────────────────────────────────────────────────

TEST(DynamicsCodegen, generated_kernel_contains_the_cell_equation_it_was_given) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("generated_source");
    directory.write("model.nml", single_cell_model());
    const String lems_path = directory.write(
            "LEMS.xml", lems_wrapper("model.nml", "singleCellNetwork", "100ms", "0.05ms"));

    NML_Parser parser;
    const NML_ParseResult parsed = parser.parse_lems(lems_path);
    const ModelLayout layout = compute_model_layout(parsed);
    const String source = generate_master_kernel(parsed, layout);

    // iafCell's own declarations, in the order it declares them:
    //   <DerivedVariable name="iSyn" select="synapses[*]/i" reduce="add"/>
    //   <DerivedVariable name="iMemb" value="leakConductance*(leakReversal - v) + iSyn"/>
    //   <TimeDerivative variable="v" value="iMemb / C"/>
    //
    // iSyn is a reduction over attached synapses, which the engine has already summed into
    // this neuron's input accumulator, so it binds straight to `network_input` rather than
    // becoming a temporary of its own. That is why it appears inside iMemb's expression
    // and nowhere on a line of its own.
    const usize memb_line = source.find("const float derived_iMemb =");
    ASSERT_NE(memb_line, String::npos);

    const usize memb_end = source.find('\n', memb_line);
    const String memb_source = source.substr(memb_line, memb_end - memb_line);
    EXPECT_NE(memb_source.find("network_input"), String::npos) << memb_source;

    // C is parameter 0 of iafCell's declared order, so the derivative is iMemb / C.
    EXPECT_NE(source.find("const float derivative_0 = (derived_iMemb / "
                          "cell_parameters[parameter_base + 0]);"), String::npos);
    EXPECT_NE(source.find("state_0 += step_dt * derivative_0;"), String::npos);

    // The OnCondition body: reset the state variable and raise the spike flag.
    EXPECT_NE(source.find("spiked = true;"), String::npos);

    // One case arm per cell type, and the type is named in the source it generated.
    EXPECT_NE(source.find("case 0: { // iafCell"), String::npos);

    // The scaffold's own stages have to be present for the generated bodies to mean
    // anything.
    EXPECT_NE(source.find("kernel void master_step("), String::npos);
    EXPECT_NE(source.find("k2t_next_neighbor("), String::npos);
}

TEST(DynamicsCodegen, a_conductance_based_synapse_is_refused_by_name) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("conductance_refused");
    directory.write("model.nml", R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Cond">
  <iafCell id="c" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV"
           reset="-70 mV" C="100 pF"/>
  <expOneSynapse id="condSyn" gbase="1 nS" erev="0 mV" tauDecay="5 ms"/>
  <network id="condNetwork">
    <population id="pop" component="c" size="2"/>
    <projection id="proj" presynapticPopulation="pop" postsynapticPopulation="pop"
                synapse="condSyn">
      <connectionWD id="0" preCellId="../pop[0]" postCellId="../pop[1]"
                    weight="1" delay="1 ms"/>
    </projection>
  </network>
</neuroml>
)");
    const String lems_path = directory.write(
            "LEMS.xml", lems_wrapper("model.nml", "condNetwork", "10ms", "0.05ms"));

    NML_Parser parser;
    const NML_ParseResult parsed = parser.parse_lems(lems_path);
    const ModelLayout layout = compute_model_layout(parsed);

    try {
        generate_master_kernel(parsed, layout);
        FAIL() << "a conductance-based synapse should not silently lower";
    } catch (const std::runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("expOneSynapse"), String::npos);
        EXPECT_NE(message.find("conductance"), String::npos);
    }
}

// ── running a model ───────────────────────────────────────────────────────────────

TEST(SpikeEngine, on_start_puts_the_cell_at_its_leak_reversal) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("on_start");
    directory.write("model.nml", single_cell_model());
    const String lems_path = directory.write(
            "LEMS.xml", lems_wrapper("model.nml", "singleCellNetwork", "10ms", "0.05ms"));

    SpikeEngine engine(lems_path);

    // Not zero, which is what an engine that skipped OnStart would leave behind — and
    // zero is above the -50 mV threshold, so every neuron would fire on tick 0 and the
    // network would then look permanently dead.
    EXPECT_NEAR(engine.read_state_variable(0, "v"), -0.065f, 1e-7f);
}

TEST(SpikeEngine, a_single_cell_reproduces_its_analytic_interspike_interval) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("analytic_isi");
    directory.write("model.nml", single_cell_model());
    const String lems_path = directory.write(
            "LEMS.xml", lems_wrapper("model.nml", "singleCellNetwork", "500ms", "0.05ms"));

    SpikeEngine engine(lems_path);
    engine.run();

    ASSERT_GE(engine.recorded_spikes.size(), 4u);

    const f64 expected = analytic_interspike_interval();
    const f64 measured = engine.recorded_spikes[3].time_seconds -
                         engine.recorded_spikes[2].time_seconds;

    // Explicit Euler at dt = 0.05 ms plus one tick of threshold-crossing discretisation.
    EXPECT_NEAR(measured, expected, 2e-4);

    // The rate follows from the same interval, so it is a second reading of the same
    // number rather than an independent one — but a rate of zero or of thousands of hertz
    // is the failure this catches.
    EXPECT_NEAR(engine.mean_firing_rate_hertz(), 1.0 / expected, 1.0);
}

// The trajectory between spikes, not just their timing: v(t) = EL + (I/gL)(1 - e^(-t/tau))
// while the cell is charging. A kernel with the wrong sign, a missing capacitance or a
// dropped input term reaches threshold at some other time but also takes a different
// route there, and this checks the route.
TEST(SpikeEngine, the_subthreshold_trajectory_matches_the_closed_form) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("trajectory");
    // 60 pA is below the 75 pA rheobase, so the cell charges toward -53 mV and never
    // fires — the whole run is the closed-form charging curve.
    directory.write("model.nml", single_cell_model("60 pA"));
    const String lems_path = directory.write(
            "LEMS.xml", lems_wrapper("model.nml", "singleCellNetwork", "100ms", "0.05ms"));

    SpikeEngine engine(lems_path);

    const f64 membrane_time_constant = 100e-12 / 5e-9;
    const f64 steady_state_offset = 60e-12 / 5e-9;

    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);

        if (tick % 200 != 0) continue;

        const f64 elapsed = (f64)(tick + 1) * engine.network_details.step_dt;
        const f64 expected = -0.065 + steady_state_offset *
                             (1.0 - std::exp(-elapsed / membrane_time_constant));

        EXPECT_NEAR(engine.read_state_variable(0, "v"), (f32)expected, 5e-5f)
                << "at tick " << tick;
    }

    EXPECT_EQ(engine.recorded_spikes.size(), 0u);
}

// A spike delivered `delay` ticks later must arrive on exactly that tick. The failure this
// guards against is silent: an off-by-one, or a mechanism that only remembers the most
// recent spike and drops earlier ones still in flight.
TEST(SpikeEngine, a_delayed_connection_arrives_on_the_tick_it_says) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("delay");
    // Cell 0 is driven above rheobase; cell 1 receives from it after 3 ms and nothing
    // else, so any movement in cell 1's membrane potential away from rest is that arrival.
    directory.write("model.nml", R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Delay">
  <iafCell id="driven" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV"
           reset="-70 mV" C="100 pF"/>
  <iafCell id="listener" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV"
           reset="-70 mV" C="100 pF"/>
  <alphaCurrentSynapse id="syn" tau="5 ms" ibase="12 pA"/>
  <pulseGenerator id="drive" delay="0 ms" duration="1000 ms" amplitude="90 pA"/>
  <network id="delayNetwork">
    <population id="popDriven" component="driven" size="1"/>
    <population id="popListener" component="listener" size="1"/>
    <projection id="proj" presynapticPopulation="popDriven"
                postsynapticPopulation="popListener" synapse="syn">
      <connectionWD id="0" preCellId="../popDriven[0]" postCellId="../popListener[0]"
                    weight="1" delay="3 ms"/>
    </projection>
    <explicitInput target="popDriven[0]" input="drive"/>
  </network>
</neuroml>
)");
    const String lems_path = directory.write(
            "LEMS.xml", lems_wrapper("model.nml", "delayNetwork", "200ms", "0.1ms"));

    SpikeEngine engine(lems_path);

    // The ring is sized from the connection delay, not from anything about the stimulus.
    EXPECT_EQ(engine.layout.maximum_edge_delay, 30);
    EXPECT_EQ(engine.layout.spike_history_length, 31);

    s64 first_source_spike_tick = -1;
    s64 first_listener_movement_tick = -1;

    const f32 resting = -0.065f;
    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);

        if (first_source_spike_tick < 0 && !engine.recorded_spikes.empty()) {
            first_source_spike_tick = tick;
        }
        if (first_source_spike_tick >= 0 && first_listener_movement_tick < 0) {
            if (std::fabs(engine.read_state_variable(1, "v") - resting) > 1e-9f) {
                first_listener_movement_tick = tick;
            }
        }
        if (first_listener_movement_tick >= 0) break;
    }

    ASSERT_GE(first_source_spike_tick, 0);
    ASSERT_GE(first_listener_movement_tick, 0);

    // Thirty-two ticks, and each one is accounted for:
    //
    //   +30  the connection's own delay. The spike is written into the history ring on the
    //        tick it fires, and the presynaptic thread reads that row 30 ticks later,
    //        which is when the synapse's OnEvent runs and J takes up weight * ibase.
    //   +1   an alpha synapse's current is I, and I is still zero on the tick J jumps --
    //        the alpha response rises from zero rather than stepping. The first non-zero
    //        current is scattered on the following tick.
    //   +1   the engine's input latency: a current scattered on one tick is drained by the
    //        target on the next.
    //
    // Changing any of the three moves this number, which is the point of asserting it
    // exactly rather than as a range.
    EXPECT_EQ(first_listener_movement_tick - first_source_spike_tick, 32);
}

TEST(SpikeEngine, connection_weights_and_delays_survive_the_weight_matrix_exactly) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("exact_weights");
    directory.write("model.nml", R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Weights">
  <iafCell id="c" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV"
           reset="-70 mV" C="100 pF"/>
  <alphaCurrentSynapse id="syn" tau="5 ms" ibase="12 pA"/>
  <network id="weightNetwork">
    <population id="pop" component="c" size="4"/>
    <projection id="proj" presynapticPopulation="pop" postsynapticPopulation="pop"
                synapse="syn">
      <connectionWD id="0" preCellId="../pop[0]" postCellId="../pop[1]"
                    weight="0.25" delay="1 ms"/>
      <connectionWD id="1" preCellId="../pop[0]" postCellId="../pop[2]"
                    weight="1.75" delay="2 ms"/>
      <connectionWD id="2" preCellId="../pop[1]" postCellId="../pop[3]"
                    weight="0.001" delay="3 ms"/>
    </projection>
  </network>
</neuroml>
)");
    const String lems_path = directory.write(
            "LEMS.xml", lems_wrapper("model.nml", "weightNetwork", "10ms", "0.1ms"));

    SpikeEngine engine(lems_path);
    ASSERT_TRUE(engine.weights.has_value());

    // Exactly, not approximately: the default matrix's coefficient vector is pinned to
    // zero, so an edge's weight is the stored value rather than a low-rank reconstruction
    // near it. A weight that came back as 0.2497 would still look plausible in a plot.
    EXPECT_FLOAT_EQ(engine.weights->get(0, 1), 0.25f);
    EXPECT_FLOAT_EQ(engine.weights->get(0, 2), 1.75f);
    EXPECT_FLOAT_EQ(engine.weights->get(1, 3), 0.001f);

    EXPECT_EQ(engine.weights->get_edge_delay_ticks(0, 1), 10);
    EXPECT_EQ(engine.weights->get_edge_delay_ticks(0, 2), 20);
    EXPECT_EQ(engine.weights->get_edge_delay_ticks(1, 3), 30);

    // Plane 0 of the per-edge family carries each edge's synapse prototype, and there is
    // only one prototype in this model.
    EXPECT_FLOAT_EQ(engine.weights->get_edge_variable(0, 0, 1), 0.0f);
    EXPECT_FLOAT_EQ(engine.weights->get_edge_variable(0, 1, 3), 0.0f);

    // alphaCurrentSynapse's OnStart sets both state variables to zero.
    EXPECT_FLOAT_EQ(engine.weights->get_edge_variable(1, 0, 1), 0.0f);
    EXPECT_FLOAT_EQ(engine.weights->get_edge_variable(2, 0, 1), 0.0f);
}

TEST(SpikeEngine, declared_output_files_are_written_with_the_shape_the_model_asked_for) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("recordings");
    directory.write("model.nml", single_cell_model());

    const String spike_file = directory.path() + "/spikes.dat";
    const String trace_file = directory.path() + "/trace.dat";

    ostringstream outputs;
    outputs << "    <OutputFile id=\"trace\" fileName=\"" << trace_file << "\">\n"
            << "      <OutputColumn id=\"v\" quantity=\"cellPopulation[0]/v\"/>\n"
            << "    </OutputFile>\n"
            << "    <EventOutputFile id=\"spikes\" fileName=\"" << spike_file
            << "\" format=\"TIME_ID\">\n"
            << "      <EventSelection id=\"0\" select=\"cellPopulation[0]\" "
               "eventPort=\"spike\"/>\n"
            << "    </EventOutputFile>\n";

    const String lems_path = directory.write(
            "LEMS.xml", lems_wrapper("model.nml", "singleCellNetwork", "200ms", "0.05ms",
                                     outputs.str()));

    SpikeEngine engine(lems_path);
    engine.run();
    engine.write_recordings();

    ASSERT_TRUE(filesystem::exists(trace_file));
    ASSERT_TRUE(filesystem::exists(spike_file));

    // One row per tick, two columns: time and v.
    ifstream trace(trace_file);
    s64 trace_rows = 0;
    String line;
    while (getline(trace, line)) {
        if (line.empty()) continue;
        istringstream columns(line);
        f64 time_seconds = 0.0;
        f64 membrane_potential = 0.0;
        ASSERT_TRUE((columns >> time_seconds >> membrane_potential));
        EXPECT_GE(membrane_potential, -0.075);
        EXPECT_LE(membrane_potential, -0.045);
        trace_rows += 1;
    }
    EXPECT_EQ(trace_rows, engine.lifetime);

    // One line per spike, and as many lines as the run counted.
    ifstream spikes(spike_file);
    s64 spike_rows = 0;
    while (getline(spikes, line)) {
        if (!line.empty()) spike_rows += 1;
    }
    EXPECT_EQ(spike_rows, (s64)engine.recorded_spikes.size());
    EXPECT_GT(spike_rows, 0);
}

// ── the demo is alive ─────────────────────────────────────────────────────────────

// The check that a passing suite is supposed to earn: a network that runs without error,
// records without error and produces almost nothing is the failure mode this whole test
// file exists to make impossible to ship.
TEST(SpikeEngine, the_balanced_network_demo_sustains_asynchronous_activity) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("balanced_network");

    examples::BalancedNetworkParameters parameters;
    const String lems_path =
            examples::write_balanced_network_model(directory.path(), parameters);

    SpikeEngine engine(lems_path);
    EXPECT_EQ(engine.total_neuron_count, 1000);
    EXPECT_EQ(engine.layout.total_edge_count, 20000);

    engine.run();

    // Alive: a population rate in the range a cortical network is usually simulated at.
    EXPECT_GT(engine.mean_firing_rate_hertz(), 5.0);
    EXPECT_LT(engine.mean_firing_rate_hertz(), 40.0);

    // Broadly alive, not a handful of cells carrying the whole rate.
    EXPECT_GT(engine.fraction_of_neurons_that_spiked(), 0.85);

    // Asynchronous: no tick where a large part of the population fires together. A
    // network locked into one repeating volley has a healthy mean rate and a useless
    // raster.
    Vector<s64> spikes_per_tick((usize)engine.lifetime, 0);
    for (const RecordedSpike &spike : engine.recorded_spikes) {
        const s64 tick = (s64)(spike.time_seconds / parameters.step_seconds + 0.5);
        if (tick >= 0 && tick < engine.lifetime) spikes_per_tick[(usize)tick] += 1;
    }
    const s64 busiest = *std::max_element(spikes_per_tick.begin(), spikes_per_tick.end());
    EXPECT_LT((f64)busiest / (f64)engine.total_neuron_count, 0.5);

    // Sustained: the end of the run fires at close to the rate of the middle, rather than
    // ringing once and decaying.
    auto rate_over = [&](f64 from_seconds, f64 to_seconds) {
        s64 count = 0;
        for (const RecordedSpike &spike : engine.recorded_spikes) {
            if (spike.time_seconds >= from_seconds && spike.time_seconds < to_seconds) {
                count += 1;
            }
        }
        return (f64)count / ((f64)engine.total_neuron_count * (to_seconds - from_seconds));
    };

    const f64 total_seconds = parameters.simulation_seconds;
    const f64 middle_rate = rate_over(0.4 * total_seconds, 0.6 * total_seconds);
    const f64 final_rate = rate_over(0.8 * total_seconds, total_seconds);

    ASSERT_GT(middle_rate, 0.0);
    EXPECT_LT(std::fabs(final_rate - middle_rate) / middle_rate, 0.3);
}
