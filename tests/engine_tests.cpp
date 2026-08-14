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
#include "../examples/glif_network_model.h"

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

// Per-edge storage is addressed by (source, slot), where `slot` is the edge's position in
// the k^2-tree row walk. The host fills those slots by walking with get_neighbors(); the
// kernel reads them by walking with k2t_next_neighbor(). Nothing enforces that the two
// walks agree — and if they ever disagree, every edge silently receives another edge's
// weight, delay and synapse state. The network still runs, still spikes, and still looks
// entirely reasonable in a raster.
//
// So: one source, three targets with weights 1x/2x/3x and delays 1/3/5 ms, and a single
// presynaptic spike. Each target's own weight must show up in the size of its response and
// its own delay in the timing of it. A permutation of the slots fails both halves.
TEST(SpikeEngine, each_edge_gets_its_own_weight_and_delay_not_another_edges) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("slot_identity");
    // The pulse stops at 40 ms. The cell first reaches threshold at
    // tau*ln((I/gL)/(I/gL - 15 mV)) = 20 ms * ln(6) = 35.8 ms and the next crossing would
    // be 40.7 ms after that, so the run contains exactly one presynaptic spike and each
    // target sees exactly one postsynaptic response.
    //
    // Targets 1, 4 and 7 rather than 1, 2, 3: non-adjacent columns make the tree walk
    // descend and unwind between neighbours instead of reading one contiguous leaf.
    directory.write("model.nml", R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Slots">
  <iafCell id="c" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV"
           reset="-70 mV" C="100 pF"/>
  <alphaCurrentSynapse id="syn" tau="5 ms" ibase="12 pA"/>
  <pulseGenerator id="drive" delay="0 ms" duration="40 ms" amplitude="90 pA"/>
  <network id="slotNetwork">
    <population id="pop" component="c" size="8"/>
    <projection id="proj" presynapticPopulation="pop" postsynapticPopulation="pop"
                synapse="syn">
      <connectionWD id="0" preCellId="../pop[0]" postCellId="../pop[1]"
                    weight="1" delay="1 ms"/>
      <connectionWD id="1" preCellId="../pop[0]" postCellId="../pop[4]"
                    weight="2" delay="3 ms"/>
      <connectionWD id="2" preCellId="../pop[0]" postCellId="../pop[7]"
                    weight="3" delay="5 ms"/>
    </projection>
    <explicitInput target="pop[0]" input="drive"/>
  </network>
</neuroml>
)");
    const String lems_path = directory.write(
            "LEMS.xml", lems_wrapper("model.nml", "slotNetwork", "120ms", "0.1ms"));

    SpikeEngine engine(lems_path);

    const s64 targets[3] = {1, 4, 7};
    const s64 expected_delay_ticks[3] = {10, 30, 50};

    const f32 resting = -0.065f;
    s64 spike_tick = -1;
    s64 first_movement_tick[3] = {-1, -1, -1};
    f32 peak_deflection[3] = {0.0f, 0.0f, 0.0f};

    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);

        if (spike_tick < 0 && !engine.recorded_spikes.empty()) spike_tick = tick;

        for (s64 index = 0; index < 3; index += 1) {
            const f32 membrane_potential =
                    engine.read_state_variable(targets[index], "v");
            const f32 deflection = membrane_potential - resting;

            if (first_movement_tick[index] < 0 && std::fabs(deflection) > 1e-9f) {
                first_movement_tick[index] = tick;
            }
            peak_deflection[index] = std::max(peak_deflection[index], deflection);
        }
    }

    ASSERT_GE(spike_tick, 0);
    ASSERT_EQ(engine.recorded_spikes.size(), 1u) << "the drive should produce one spike";

    // Timing: each target moves delay + 2 ticks after the presynaptic spike, for its own
    // delay. Slot-permuted delays would give 30/10/50 or some other rearrangement.
    for (s64 index = 0; index < 3; index += 1) {
        ASSERT_GE(first_movement_tick[index], 0) << "target " << targets[index] << " never moved";
        EXPECT_EQ(first_movement_tick[index] - spike_tick, expected_delay_ticks[index] + 2)
                << "target " << targets[index];
    }

    // Size: an alphaCurrentSynapse's response scales linearly in the connection weight, so
    // the three peaks are in 1:2:3. Slot-permuted weights would give 3:2:1 or 2:1:3.
    for (s64 index = 0; index < 3; index += 1) {
        EXPECT_GT(peak_deflection[index], 0.0f) << "target " << targets[index];
    }
    EXPECT_NEAR(peak_deflection[1] / peak_deflection[0], 2.0, 0.02);
    EXPECT_NEAR(peak_deflection[2] / peak_deflection[0], 3.0, 0.02);

    // Nothing reached any other cell: only three of the eight have an incoming edge.
    for (s64 neuron_index = 1; neuron_index < 8; neuron_index += 1) {
        if (neuron_index == 1 || neuron_index == 4 || neuron_index == 7) continue;
        EXPECT_FLOAT_EQ(engine.read_state_variable(neuron_index, "v"), resting)
                << "neuron " << neuron_index << " has no incoming edge and no input";
    }
}

// Each edge names its own synapse prototype, stored in plane 0 of the per-edge family and
// used by the kernel to pick both the generated body and the parameter row. With two
// prototypes of the same type differing only in the sign of ibase, reading the wrong row
// flips excitation into inhibition — which in a balanced network shows up as the whole
// thing dying or running away, but never as an error.
TEST(SpikeEngine, an_edge_uses_the_parameters_of_its_own_synapse_prototype) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("prototype_identity");
    directory.write("model.nml", R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Prototypes">
  <iafCell id="c" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV"
           reset="-70 mV" C="100 pF"/>
  <alphaCurrentSynapse id="excitatory" tau="5 ms" ibase="12 pA"/>
  <alphaCurrentSynapse id="inhibitory" tau="5 ms" ibase="-12 pA"/>
  <pulseGenerator id="drive" delay="0 ms" duration="40 ms" amplitude="90 pA"/>
  <network id="prototypeNetwork">
    <population id="pop" component="c" size="4"/>
    <projection id="excitatoryProjection" presynapticPopulation="pop"
                postsynapticPopulation="pop" synapse="excitatory">
      <connectionWD id="0" preCellId="../pop[0]" postCellId="../pop[1]"
                    weight="1" delay="1 ms"/>
    </projection>
    <projection id="inhibitoryProjection" presynapticPopulation="pop"
                postsynapticPopulation="pop" synapse="inhibitory">
      <connectionWD id="0" preCellId="../pop[0]" postCellId="../pop[2]"
                    weight="1" delay="1 ms"/>
    </projection>
    <explicitInput target="pop[0]" input="drive"/>
  </network>
</neuroml>
)");
    const String lems_path = directory.write(
            "LEMS.xml", lems_wrapper("model.nml", "prototypeNetwork", "120ms", "0.1ms"));

    SpikeEngine engine(lems_path);
    ASSERT_EQ(engine.network_details.synapse_prototypes.size(), 2u);

    const f32 resting = -0.065f;
    f32 excitatory_peak = 0.0f;
    f32 inhibitory_trough = 0.0f;

    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);

        excitatory_peak = std::max(excitatory_peak,
                                   engine.read_state_variable(1, "v") - resting);
        inhibitory_trough = std::min(inhibitory_trough,
                                     engine.read_state_variable(2, "v") - resting);
    }

    ASSERT_EQ(engine.recorded_spikes.size(), 1u);

    // Opposite signs, equal magnitudes: the two prototypes differ only in the sign of
    // ibase, so a swap would show up as both deflections having the same sign.
    EXPECT_GT(excitatory_peak, 0.0f);
    EXPECT_LT(inhibitory_trough, 0.0f);
    EXPECT_NEAR(excitatory_peak, -inhibitory_trough, 1e-6f);

    // Neuron 3 has no incoming edge, so nothing reached it.
    EXPECT_FLOAT_EQ(engine.read_state_variable(3, "v"), resting);
}

// The delayed-arrival test above checks the first arrival and stops. That leaves the part
// that only happens later untested: the spike-history ring wraps every
// spike_history_length ticks, and a modulo that is subtly wrong past the first wrap would
// deliver the first spike correctly and then drop or misplace every one after it. A
// network would still look alive while quietly losing a fraction of its spikes.
//
// So: one source firing repeatedly across ~160 wraps of the ring, a single edge at the
// worst-case delay of ring_length - 1, and a target that never fires. The target is driven
// by nothing else, so between arrivals it only leaks toward its leak reversal -- every run
// of rising membrane potential is one arrival, and they have to match the source's spikes
// one for one, each at exactly the right offset.
TEST(SpikeEngine, no_arrival_is_dropped_as_the_spike_history_ring_wraps) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("ring_wraparound");
    directory.write("model.nml", R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Wrap">
  <iafCell id="source" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV"
           reset="-70 mV" C="100 pF"/>
  <iafCell id="listener" leakConductance="0 nS" leakReversal="-65 mV" thresh="1000 mV"
           reset="-70 mV" C="100 pF"/>
  <alphaCurrentSynapse id="syn" tau="2 ms" ibase="12 pA"/>
  <pulseGenerator id="drive" delay="0 ms" duration="1000 ms" amplitude="90 pA"/>
  <network id="wrapNetwork">
    <population id="popSource" component="source" size="1"/>
    <population id="popListener" component="listener" size="1"/>
    <projection id="proj" presynapticPopulation="popSource"
                postsynapticPopulation="popListener" synapse="syn">
      <connectionWD id="0" preCellId="../popSource[0]" postCellId="../popListener[0]"
                    weight="1" delay="3 ms"/>
    </projection>
    <explicitInput target="popSource[0]" input="drive"/>
  </network>
</neuroml>
)");
    const String lems_path = directory.write(
            "LEMS.xml", lems_wrapper("model.nml", "wrapNetwork", "500ms", "0.1ms"));

    SpikeEngine engine(lems_path);

    // Worst case: the delay is one short of the ring, so the row a delayed arrival reads is
    // the one about to be overwritten.
    const s64 delay_ticks = 30;
    ASSERT_EQ(engine.layout.maximum_edge_delay, delay_ticks);
    ASSERT_EQ(engine.layout.spike_history_length, delay_ticks + 1);

    // The run has to cross the ring many times for this to mean anything.
    EXPECT_GT(engine.lifetime / engine.layout.spike_history_length, 100);

    Vector<s64> source_spike_ticks;
    Vector<s64> arrival_onset_ticks;

    // An arrival's first tick raises the listener by dt * i / C. With tau = 2 ms the
    // synapse's current on that tick is dt*e*(weight*ibase)/tau = 1.6e-12 A, so the step is
    // about 1.6e-6 V. Between arrivals the alpha tail has decayed for a full interspike
    // interval -- twenty time constants -- leaving a per-tick residual around 1e-14 V,
    // which never reaches zero in f32 and is why a plain `v increased` test either counts
    // one endless arrival or, with a leak added, counts the wrong tick. This threshold sits
    // two orders below the step and many above the tail.
    const f32 arrival_rise_threshold = 1e-8f;

    f32 previous_potential = engine.read_state_variable(1, "v");
    bool was_rising = false;
    usize spikes_seen = 0;

    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);

        while (spikes_seen < engine.recorded_spikes.size()) {
            source_spike_ticks.push_back(tick);
            spikes_seen += 1;
        }

        // The listener has no leak conductance and a threshold far out of reach, so it is
        // a pure integrator that never fires: dv/dt is exactly the synaptic current over
        // C. Between arrivals it holds perfectly flat, so the first tick it rises is the
        // tick the arrival reached it -- with a leak, a later arrival has to first
        // overcome the residual decay of the previous one, which moves the measurement.
        const f32 potential = engine.read_state_variable(1, "v");
        const bool rising = (potential - previous_potential) > arrival_rise_threshold;
        if (rising && !was_rising) arrival_onset_ticks.push_back(tick);

        was_rising = rising;
        previous_potential = potential;
    }

    ASSERT_GT(source_spike_ticks.size(), 8u) << "the source barely fired";

    // Arrivals still in flight when the run ends have no onset to match.
    const s64 last_tick_that_can_arrive = engine.lifetime - (delay_ticks + 2);
    Vector<s64> expected_source_ticks;
    for (s64 spike_tick : source_spike_ticks) {
        if (spike_tick <= last_tick_that_can_arrive) expected_source_ticks.push_back(spike_tick);
    }

    ASSERT_EQ(arrival_onset_ticks.size(), expected_source_ticks.size())
            << "the source fired " << expected_source_ticks.size()
            << " times with time to arrive, and the target saw " << arrival_onset_ticks.size()
            << " arrivals";

    // Every one of them lands at the same offset, not just the first: delay, plus one tick
    // for the alpha response to leave zero, plus one for the engine's input latency.
    for (usize index = 0; index < expected_source_ticks.size(); index += 1) {
        EXPECT_EQ(arrival_onset_ticks[index] - expected_source_ticks[index], delay_ticks + 2)
                << "arrival " << index << " of " << expected_source_ticks.size();
    }
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

// A spikeArray declares no amplitude, because it is a spike source rather than a current
// injector — its train becomes current only by going through a synapse. Treating it as a
// current injection would load the model, run every tick, and write a recording with
// nothing in it, which is the failure that is hardest to notice.
TEST(SpikeEngine, a_spike_train_stimulus_is_refused_rather_than_run_silent) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("spike_train_refused");
    directory.write("model.nml", R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Train">
  <iafCell id="c" leakConductance="5 nS" leakReversal="-65 mV" thresh="-50 mV"
           reset="-70 mV" C="100 pF"/>
  <spikeArray id="train">
    <spike id="0" time="10 ms"/>
    <spike id="1" time="20 ms"/>
  </spikeArray>
  <network id="trainNetwork">
    <population id="pop" component="c" size="1"/>
    <explicitInput target="pop[0]" input="train"/>
  </network>
</neuroml>
)");
    const String lems_path = directory.write(
            "LEMS.xml", lems_wrapper("model.nml", "trainNetwork", "50ms", "0.1ms"));

    try {
        SpikeEngine engine(lems_path);
        FAIL() << "a spike-train stimulus should not load as a silent no-op";
    } catch (const std::runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("train"), String::npos) << message;
        EXPECT_NE(message.find("spike train"), String::npos) << message;
    }
}

// ── more than one cell type in a model ────────────────────────────────────────────

// Every other test in this file has exactly one cell type, which leaves the parts of the
// design that exist only for heterogeneity completely unexercised: the generated switch's
// second arm, a cell-state buffer whose populations have different widths, and a parameter
// table whose rows have different lengths.
//
// iafCell has one state variable; izhikevich2007Cell has two and roughly twice the
// parameters. Their OnStart values differ too, which is what makes a layout mistake
// visible rather than merely possible: if the two populations' chunks overlapped, one
// type's starting values would appear in the other's.
namespace {

String two_cell_type_model() {
    return R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="TwoTypes">
  <iafCell id="integrateAndFire" leakConductance="5 nS" leakReversal="-65 mV"
           thresh="-50 mV" reset="-70 mV" C="100 pF"/>
  <izhikevich2007Cell id="izhikevich" C="100pF" v0="-50mV" k="0.7nS_per_mV"
                      vr="-60mV" vt="-40mV" vpeak="35mV" a="0.03per_ms"
                      b="-2nS" c="-50mV" d="100pA"/>
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

// Host side only: this asserts what the layout says and what read_state_variable does with
// it, both of which run on the CPU. It cannot see whether the kernel agrees — the offsets
// reach the kernel as a baked `constant` table, and corrupting that table leaves every
// assertion here passing. both_cell_types_integrate_their_own_equations below is what
// covers the kernel's half, and it was checked against exactly that injected fault.
TEST(SpikeEngine, two_cell_types_get_separate_state_and_parameter_chunks) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("two_cell_types");
    directory.write("model.nml", two_cell_type_model());
    const String lems_path = directory.write(
            "LEMS.xml", lems_wrapper("model.nml", "twoTypeNetwork", "1ms", "0.01ms"));

    SpikeEngine engine(lems_path);

    ASSERT_EQ(engine.network_details.cell_types.size(), 2u);
    ASSERT_EQ(engine.network_details.populations.size(), 2u);

    // Sizes come from the types, not from a shared maximum: four one-variable cells and
    // three two-variable cells is ten slots, not fourteen.
    const CellTypeSpecification &iaf_type = engine.network_details.cell_types[0];
    const CellTypeSpecification &izhikevich_type = engine.network_details.cell_types[1];
    ASSERT_EQ(iaf_type.state_variable_names.size(), 1u);
    ASSERT_EQ(izhikevich_type.state_variable_names.size(), 2u);
    EXPECT_EQ(engine.layout.cell_state_length, 4 * 1 + 3 * 2);

    // The two populations' chunks are disjoint and the second starts where the first ends.
    EXPECT_EQ(engine.layout.population_state_base[0], 0);
    EXPECT_EQ(engine.layout.population_state_base[1], 4);

    // Parameter rows differ in length, so the second prototype's row cannot start at a
    // fixed stride.
    EXPECT_GT(izhikevich_type.parameter_names.size(), iaf_type.parameter_names.size());
    EXPECT_EQ(engine.layout.cell_prototype_parameter_base[0], 0);
    EXPECT_EQ(engine.layout.cell_prototype_parameter_base[1],
              (s64)iaf_type.parameter_names.size());
    EXPECT_EQ(engine.layout.cell_parameter_length,
              (s64)(iaf_type.parameter_names.size() + izhikevich_type.parameter_names.size()));

    // OnStart: iafCell starts at leakReversal, izhikevich2007Cell at v0 with u at zero.
    // Three different starting values across the two chunks, so any overlap shows up here.
    for (s64 neuron_index = 0; neuron_index < 4; neuron_index += 1) {
        EXPECT_NEAR(engine.read_state_variable(neuron_index, "v"), -0.065f, 1e-7f)
                << "iafCell " << neuron_index;
    }
    for (s64 neuron_index = 4; neuron_index < 7; neuron_index += 1) {
        EXPECT_NEAR(engine.read_state_variable(neuron_index, "v"), -0.050f, 1e-7f)
                << "izhikevich " << neuron_index;
        EXPECT_NEAR(engine.read_state_variable(neuron_index, "u"), 0.0f, 1e-12f)
                << "izhikevich " << neuron_index;
    }

    // `u` belongs to one type only, and asking the wrong population for it is an error
    // rather than a read of whatever happens to sit at that offset.
    EXPECT_THROW((void)engine.read_state_variable(0, "u"), std::runtime_error);
}

TEST(SpikeEngine, both_cell_types_integrate_their_own_equations) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("two_cell_types_running");
    directory.write("model.nml", two_cell_type_model());
    const String lems_path = directory.write(
            "LEMS.xml", lems_wrapper("model.nml", "twoTypeNetwork", "60ms", "0.01ms"));

    SpikeEngine engine(lems_path);
    engine.run();

    // Both arms of the generated switch ran: each driven population produced spikes. A
    // switch that fell through to one arm would leave the other population silent.
    Vector<s64> spikes_by_population(2, 0);
    for (const RecordedSpike &spike : engine.recorded_spikes) {
        spikes_by_population[spike.neuron_index < 4 ? 0 : 1] += 1;
    }
    EXPECT_GT(spikes_by_population[0], 0) << "no iafCell fired";
    EXPECT_GT(spikes_by_population[1], 0) << "no izhikevich2007Cell fired";

    // An undriven iafCell sits at a genuine fixed point: at v = leakReversal its whole
    // derivative is gL*(leakReversal - v)/C = 0, so it must not have moved at all. That is
    // the strongest available statement that the driven cells' integration did not reach
    // across the chunk boundary.
    EXPECT_NEAR(engine.read_state_variable(2, "v"), -0.065f, 1e-7f);
    EXPECT_NEAR(engine.read_state_variable(3, "v"), -0.065f, 1e-7f);

    // An undriven izhikevich2007Cell does not stay at v0 — v0 is an initial condition, not
    // a resting potential. At v0 = -50 mV, between vr = -60 mV and vt = -40 mV, the
    // quadratic term k*(v-vr)*(v-vt) is 0.7 nS/mV * 10 mV * -10 mV = -70 pA, so the cell
    // relaxes downward and settles at vr, taking u to zero with it. Asserting it stayed at
    // v0 would be asserting the model is wrong.
    EXPECT_NEAR(engine.read_state_variable(6, "v"), -0.060f, 5e-4f);
    EXPECT_NEAR(engine.read_state_variable(6, "u"), 0.0f, 1e-11f);

    // The two undriven izhikevich cells have identical parameters, identical (zero) input
    // and identical initial state, so they must be identical to the bit. Anything writing
    // outside its own slot — a stride mistake, a chunk overlap — separates them.
    EXPECT_FLOAT_EQ(engine.read_state_variable(5, "v"), engine.read_state_variable(6, "v"));
    EXPECT_FLOAT_EQ(engine.read_state_variable(5, "u"), engine.read_state_variable(6, "u"));

    // Neither undriven izhikevich cell ever reached vpeak.
    for (const RecordedSpike &spike : engine.recorded_spikes) {
        EXPECT_NE(spike.neuron_index, 5);
        EXPECT_NE(spike.neuron_index, 6);
    }

    // izhikevich2007Cell's recovery variable u is driven by its own second
    // TimeDerivative, so a driven one has moved away from zero while the undriven one has
    // not. That is the only place a second state variable per cell is exercised at all.
    EXPECT_NE(engine.read_state_variable(4, "u"), 0.0f);

    // Its OnCondition assigns both v and u (v = c, u = u + d), so a spiking izhikevich
    // cell must sit at or below its reset value rather than above vpeak.
    EXPECT_LE(engine.read_state_variable(4, "v"), 0.035f);
}

// ── the GLIF family ───────────────────────────────────────────────────────────────

namespace {

// Spike times of one neuron, in order.
Vector<f64> spike_times_of(const SpikeEngine &engine, s64 neuron_index) {
    Vector<f64> times;
    for (const RecordedSpike &spike : engine.recorded_spikes) {
        if (spike.neuron_index == neuron_index) times.push_back(spike.time_seconds);
    }
    return times;
}

} // namespace

// All five GLIF types under the same current step. Each one's defining behaviour is
// asserted rather than its trace being eyeballed: GLIF1 and GLIF2 fire at a fixed rate,
// and the three that carry adaptation state fire progressively slower.
TEST(SpikeEngine, the_glif_family_each_produce_their_defining_behaviour) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    SpikeEngine engine("tests/fixtures/nml/LEMS_glif_family.xml");
    ASSERT_EQ(engine.total_neuron_count, 5);
    ASSERT_EQ(engine.network_details.cell_types.size(), 5u);

    engine.run();

    // GLIF1 is a leaky integrator with a hard refractory period, so its interval has a
    // closed form: the time to charge from vreset to vth under the step, plus t_ref.
    //   tau       = C / gL = 100 pF / 10 nS = 10 ms
    //   v_infinity = EL + I/gL = -70 mV + 500 pA / 10 nS = -20 mV
    //   charge    = tau * ln((v_inf - vreset) / (v_inf - vth)) = 10 ms * ln(50/30)
    const f64 membrane_time_constant = 100e-12 / 10e-9;
    const f64 steady_state = -0.070 + 500e-12 / 10e-9;
    const f64 charging_time = membrane_time_constant *
            std::log((steady_state - -0.070) / (steady_state - -0.050));
    const f64 expected_interval = charging_time + 5e-3;

    const Vector<f64> glif1 = spike_times_of(engine, 0);
    ASSERT_GE(glif1.size(), 5u) << "GLIF1 did not fire under 2.5x rheobase";
    EXPECT_NEAR(glif1[3] - glif1[2], expected_interval, 2e-4);

    // Its intervals are all the same: nothing in GLIF1 accumulates across spikes.
    EXPECT_NEAR(glif1[glif1.size() - 1] - glif1[glif1.size() - 2],
                glif1[2] - glif1[1], 2e-4);

    // Every type fires, and the refractory period is respected by all of them: no
    // interval anywhere is shorter than t_ref.
    for (s64 cell = 0; cell < 5; cell += 1) {
        const Vector<f64> times = spike_times_of(engine, cell);
        ASSERT_GE(times.size(), 5u) << "cell " << cell << " fired " << times.size()
                                    << " times";

        for (usize index = 1; index < times.size(); index += 1) {
            EXPECT_GE(times[index] - times[index - 1], 5e-3 - 1e-4)
                    << "cell " << cell << " interval " << index << " is shorter than t_ref";
        }
    }

    // GLIF3 (after-spike currents), GLIF4 (adapting threshold) and GLIF5 (both) slow down
    // over the step; GLIF1 and GLIF2 do not. This is the whole reason those types carry
    // extra state, so it is the thing worth asserting.
    auto adaptation_ratio = [&](s64 cell) {
        const Vector<f64> times = spike_times_of(engine, cell);
        const f64 first = times[1] - times[0];
        const f64 last = times[times.size() - 1] - times[times.size() - 2];
        return last / first;
    };

    EXPECT_NEAR(adaptation_ratio(0), 1.0, 0.05) << "GLIF1 should not adapt";
    EXPECT_NEAR(adaptation_ratio(1), 1.0, 0.05) << "GLIF2 should not adapt";
    EXPECT_GT(adaptation_ratio(2), 1.5) << "GLIF3's after-spike currents should adapt";
    EXPECT_GT(adaptation_ratio(3), 1.2) << "GLIF4's adapting threshold should adapt";
    EXPECT_GT(adaptation_ratio(4), 1.5) << "GLIF5 carries both and should adapt";

    // The adaptation state actually moved, and in the direction that suppresses firing:
    // the after-spike currents are hyperpolarising and the threshold rises.
    EXPECT_LT(engine.read_state_variable(2, "asc1"), 0.0f);
    EXPECT_LT(engine.read_state_variable(2, "asc2"), 0.0f);
    EXPECT_GT(engine.read_state_variable(3, "theta"), -0.050f);
    EXPECT_GT(engine.read_state_variable(4, "theta"), -0.050f);
}

TEST(DynamicsCodegen, a_regime_pair_lowers_to_a_refractory_gate) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    NML_Parser parser;
    const NML_ParseResult parsed =
            parser.parse_lems("tests/fixtures/nml/LEMS_glif_family.xml");
    const ModelLayout layout = compute_model_layout(parsed);
    const String source = generate_master_kernel(parsed, layout);

    // The gate itself, and no trace of a state machine.
    EXPECT_NE(source.find("const float time_since_spike"), String::npos);
    EXPECT_EQ(source.find("set_regime"), String::npos);
    EXPECT_EQ(source.find("neuron_regime"), String::npos);

    // The refractory regime's own timer is never integrated: last_spiked replaces it.
    EXPECT_EQ(source.find("refractoryTimeElapsed +="), String::npos);

    // GLIF3's after-spike currents are declared outside the regimes, so they decay every
    // tick including while the cell is held. If they had been swept into the gate they
    // would freeze during the refractory period and adaptation would be wrong.
    //
    // Scoped to GLIF3's own case arm. Searching the whole kernel compares offsets across
    // five different cell bodies -- GLIF1's gate comes before GLIF3's ascSum simply
    // because GLIF1 is emitted first, which says nothing about either.
    const usize body_start = source.find("case 2: { // GLIF3Cell");
    ASSERT_NE(body_start, String::npos);
    const usize body_end = source.find("} break;", body_start);
    ASSERT_NE(body_end, String::npos);
    const String glif3_body = source.substr(body_start, body_end - body_start);

    const usize gate = glif3_body.find("const float time_since_spike");
    const usize asc_decay = glif3_body.find("derived_ascSum");
    ASSERT_NE(gate, String::npos);
    ASSERT_NE(asc_decay, String::npos);
    EXPECT_LT(asc_decay, gate) << "ascSum must be computed before the refractory gate";

    // And the decay itself is outside the gate, not inside it.
    const usize asc_step = glif3_body.find("state_1 += step_dt");
    ASSERT_NE(asc_step, String::npos);
    EXPECT_LT(asc_step, gate) << "asc1 must decay whether or not the cell is refractory";
}

// GLIF2's only difference from GLIF1 is its reset rule: v = vreset + resetScale*(v - vth),
// which carries part of the threshold overshoot into the next cycle instead of discarding
// it. Under the family fixture's drive the overshoot is a fraction of a millivolt, so
// GLIF2 and GLIF1 fire identically -- which means every other assertion about GLIF2 passes
// just as well if codegen dropped the reset expression entirely.
//
// So: two GLIF2 cells differing only in resetScale, under a drive strong enough that the
// per-tick overshoot is large. A higher resetScale resets closer to threshold and must
// fire more often. If the expression were dropped, both would reset to vreset and the
// counts would match.
TEST(SpikeEngine, glif2_reset_scale_changes_how_often_the_cell_fires) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("reset_scale");
    // 10 nA against a 200 pA leak at threshold is about 98 V/s, so at dt = 0.1 ms the cell
    // overshoots vth by roughly 9.8 mV before it is caught -- a fifth of the 20 mV swing,
    // and enough for resetScale to matter.
    directory.write("model.nml", R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ResetScale">
  <include href="glif_cell_types.nml"/>

  <GLIF2Cell id="discardOvershoot" C="100pF" gL="10nS" EL="-70mV" vth="-50mV"
             vreset="-70mV" resetScale="0" t_ref="1ms"/>
  <GLIF2Cell id="carryOvershoot" C="100pF" gL="10nS" EL="-70mV" vth="-50mV"
             vreset="-70mV" resetScale="0.9" t_ref="1ms"/>

  <pulseGenerator id="hardDrive" delay="0ms" duration="200ms" amplitude="10nA"/>

  <network id="resetScaleNetwork">
    <population id="popDiscard" component="discardOvershoot" size="1"/>
    <population id="popCarry" component="carryOvershoot" size="1"/>
    <explicitInput target="popDiscard[0]" input="hardDrive"/>
    <explicitInput target="popCarry[0]" input="hardDrive"/>
  </network>
</neuroml>
)");
    // The types file is included by relative href, so it has to sit beside the model.
    {
        ifstream source("tests/fixtures/nml/glif_cell_types.nml");
        ASSERT_TRUE(source.good()) << "GLIF cell type fixture missing";
        ostringstream contents;
        contents << source.rdbuf();
        directory.write("glif_cell_types.nml", contents.str());
    }

    const String lems_path = directory.write(
            "LEMS.xml", lems_wrapper("model.nml", "resetScaleNetwork", "200ms", "0.1ms"));

    SpikeEngine engine(lems_path);
    engine.run();

    const Vector<f64> discarding = spike_times_of(engine, 0);
    const Vector<f64> carrying = spike_times_of(engine, 1);

    ASSERT_GE(discarding.size(), 10u);
    ASSERT_GE(carrying.size(), 10u);

    // Resetting nearer to threshold means a shorter climb back, so more spikes in the same
    // window. Equality here is the failure the test exists to catch.
    EXPECT_GT(carrying.size(), discarding.size())
            << "resetScale=0.9 fired " << carrying.size() << " times and resetScale=0 fired "
            << discarding.size() << "; the reset rule is not being applied";

    // And the interval difference is in the direction and rough size the arithmetic says:
    // the carried overshoot removes roughly 9 mV of a 20 mV climb.
    const f64 discarding_interval = discarding[5] - discarding[4];
    const f64 carrying_interval = carrying[5] - carrying[4];
    EXPECT_LT(carrying_interval, discarding_interval);
    EXPECT_GT(discarding_interval - carrying_interval, 5e-5);
}

TEST(DynamicsCodegen, a_regime_shape_that_is_not_the_refractory_pair_is_refused) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    ModelDirectory directory("three_regimes");
    directory.write("model.nml", R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="ThreeRegimes">
  <ComponentType name="ThreeRegimeCell" extends="baseCell">
    <Parameter name="C" dimension="capacitance"/>
    <Parameter name="gL" dimension="conductance"/>
    <Parameter name="EL" dimension="voltage"/>
    <Parameter name="vth" dimension="voltage"/>
    <EventPort name="spike" direction="out"/>
    <Exposure name="v" dimension="voltage"/>
    <Dynamics>
      <StateVariable name="v" dimension="voltage" exposure="v"/>
      <OnStart><StateAssignment variable="v" value="EL"/></OnStart>
      <Regime name="one" initial="true">
        <TimeDerivative variable="v" value="(gL * (EL - v)) / C"/>
        <OnCondition test="v .gt. vth"><Transition regime="two"/></OnCondition>
      </Regime>
      <Regime name="two">
        <OnCondition test="v .gt. vth"><Transition regime="three"/></OnCondition>
      </Regime>
      <Regime name="three">
        <OnCondition test="v .gt. vth"><Transition regime="one"/></OnCondition>
      </Regime>
    </Dynamics>
  </ComponentType>

  <ThreeRegimeCell id="cell" C="100pF" gL="10nS" EL="-70mV" vth="-50mV"/>
  <network id="threeRegimeNetwork">
    <population id="pop" component="cell" size="1"/>
  </network>
</neuroml>
)");
    const String lems_path = directory.write(
            "LEMS.xml", lems_wrapper("model.nml", "threeRegimeNetwork", "10ms", "0.1ms"));

    NML_Parser parser;
    const NML_ParseResult parsed = parser.parse_lems(lems_path);
    const ModelLayout layout = compute_model_layout(parsed);

    try {
        generate_master_kernel(parsed, layout);
        FAIL() << "a three-regime state machine should not lower to a refractory gate";
    } catch (const std::runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("ThreeRegimeCell"), String::npos) << message;
        EXPECT_NE(message.find("3 regimes"), String::npos) << message;
    }
}

// ── the GLIF networks are alive ───────────────────────────────────────────────────

namespace {

struct NetworkActivity {
    f64 mean_rate = 0.0;
    f64 participation = 0.0;
    f64 peak_synchrony = 0.0;
    f64 middle_rate = 0.0;
    f64 final_rate = 0.0;
};

NetworkActivity measure_activity(const SpikeEngine &engine, f64 step_seconds,
                                 f64 total_seconds) {
    NetworkActivity activity;
    activity.mean_rate = engine.mean_firing_rate_hertz();
    activity.participation = engine.fraction_of_neurons_that_spiked();

    Vector<s64> spikes_per_tick((usize)engine.lifetime, 0);
    for (const RecordedSpike &spike : engine.recorded_spikes) {
        const s64 tick = (s64)(spike.time_seconds / step_seconds + 0.5);
        if (tick >= 0 && tick < engine.lifetime) spikes_per_tick[(usize)tick] += 1;
    }
    activity.peak_synchrony =
            (f64)*std::max_element(spikes_per_tick.begin(), spikes_per_tick.end()) /
            (f64)engine.total_neuron_count;

    auto rate_over = [&](f64 from_seconds, f64 to_seconds) {
        s64 count = 0;
        for (const RecordedSpike &spike : engine.recorded_spikes) {
            if (spike.time_seconds >= from_seconds && spike.time_seconds < to_seconds) {
                count += 1;
            }
        }
        return (f64)count / ((f64)engine.total_neuron_count * (to_seconds - from_seconds));
    };
    activity.middle_rate = rate_over(0.4 * total_seconds, 0.6 * total_seconds);
    activity.final_rate = rate_over(0.8 * total_seconds, total_seconds);

    return activity;
}

} // namespace

// One recurrent network per GLIF type, each asserted to be alive on the same terms: a
// population rate in a sensible band, nearly every neuron participating, activity that is
// asynchronous rather than one repeating volley, and a network still firing at the end of
// the run. A network that loads, runs, records and produces almost nothing passes every
// other test in this file and fails this one.
class GlifNetworkAliveness : public ::testing::TestWithParam<s32> {};

TEST_P(GlifNetworkAliveness, sustains_asynchronous_recurrent_activity) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    const s32 glif_index = GetParam();
    ModelDirectory directory("glif" + to_string(glif_index) + "_network");

    examples::GlifNetworkParameters parameters;
    parameters.glif_index = glif_index;
    parameters.cell_types_path = "tests/fixtures/nml/glif_cell_types.nml";

    const String lems_path = examples::write_glif_network_model(directory.path(), parameters);

    SpikeEngine engine(lems_path);
    EXPECT_EQ(engine.total_neuron_count, 500);
    EXPECT_EQ(engine.layout.total_edge_count, 10000);

    engine.run();

    const NetworkActivity activity = measure_activity(
            engine, parameters.step_seconds, parameters.simulation_seconds);

    // Alive, and not saturated. The 5 ms refractory period caps any cell at 200 Hz, so a
    // rate approaching that would mean every cell firing flat out.
    EXPECT_GT(activity.mean_rate, 2.0) << "GLIF" << glif_index << " is barely firing";
    EXPECT_LT(activity.mean_rate, 60.0) << "GLIF" << glif_index << " is saturated";

    // Broadly alive rather than a few cells carrying the whole rate.
    EXPECT_GT(activity.participation, 0.85);

    // Not a lockstep volley: no tick where a large part of the population fires together.
    // This is a ceiling, not a demand for a flat raster -- the adapting types develop a
    // real population rhythm, because after-spike currents recover on a shared time
    // constant and pull the network into bands. That shows up in the raster around 10 Hz
    // for GLIF3 and GLIF5 and is a property of the model, not a defect. What this rules
    // out is every cell firing on the same tick.
    EXPECT_LT(activity.peak_synchrony, 0.5);

    // Still going at the end. A wide floor rather than a tight band, because the types
    // that adapt are genuinely still settling at the end of the run -- GLIF5 carries both
    // after-spike currents and an adapting threshold and drifts down the longest. What
    // this rules out is the network dying, not slow adaptation.
    ASSERT_GT(activity.middle_rate, 0.0);
    EXPECT_GT(activity.final_rate, 0.6 * activity.middle_rate)
            << "GLIF" << glif_index << " decayed from " << activity.middle_rate << " Hz to "
            << activity.final_rate << " Hz";
    EXPECT_GT(activity.final_rate, 2.0);
}

INSTANTIATE_TEST_SUITE_P(AllFiveTypes, GlifNetworkAliveness,
                         ::testing::Values(1, 2, 3, 4, 5),
                         [](const ::testing::TestParamInfo<s32> &info) {
                             return "GLIF" + to_string(info.param);
                         });

// The three types that carry adaptation state fire more slowly in a network than the two
// that do not, under identical drive and identical connectivity. This is the network-level
// counterpart of the single-cell adaptation test: it confirms the extra state is doing
// something once cells are wired together, not just under a clean current step.
TEST(SpikeEngine, adapting_glif_networks_settle_below_non_adapting_ones) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    Vector<f64> rate_by_type(6, 0.0);

    for (s32 glif_index : {1, 3, 5}) {
        ModelDirectory directory("glif_rate_" + to_string(glif_index));

        examples::GlifNetworkParameters parameters;
        parameters.glif_index = glif_index;
        parameters.cell_types_path = "tests/fixtures/nml/glif_cell_types.nml";
        // Half the run of the aliveness tests: the rate separation is fully developed
        // within a second (23 Hz against 5 Hz), and three networks at full length is two
        // minutes of suite time for a comparison that is already clear.
        parameters.simulation_seconds = 1.0;

        SpikeEngine engine(examples::write_glif_network_model(directory.path(), parameters));
        engine.run();
        rate_by_type[(usize)glif_index] = engine.mean_firing_rate_hertz();
    }

    // GLIF3's after-spike currents suppress firing; GLIF5 adds an adapting threshold on
    // top, so it settles at or below GLIF3.
    EXPECT_LT(rate_by_type[3], rate_by_type[1] * 0.75)
            << "GLIF3 " << rate_by_type[3] << " Hz vs GLIF1 " << rate_by_type[1] << " Hz";
    EXPECT_LT(rate_by_type[5], rate_by_type[1] * 0.75)
            << "GLIF5 " << rate_by_type[5] << " Hz vs GLIF1 " << rate_by_type[1] << " Hz";
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
