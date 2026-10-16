// Weighted arrivals: synapse state that only an OnEvent changes, read by the target's add-select,
// as in the Brunel (2000) model A types (fixtures/nml/Brunel2000_ModelA_ComponentTypes.xml). A
// spike sent at t_s adds weight * jump to the target's v at exactly t_s + delay, unless the target
// is refractory then, and the threshold is checked after the jump in the same step.
//
// The single-neuron verification is WORKSPACE_MANIFEST.md section 4, with its scripted spikes sent
// by spikeArray-fired cells through projections. Its expected values are the exact event-driven
// solution on the same 0.01 ms grid; the engine integrates the leak by forward Euler, which differs
// from it by at most (dt / tau)^2 / 2 per step of decay: 1.25e-7 per step, so about 0.001 mV after
// the 1990 steps from the reset to 41.4 ms. The tolerance is twice that.

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

#include "engine.h"
#include "support/test_support.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::test_support;

namespace {

constexpr f64 MILLIVOLT = 1e-3;
constexpr f64 MILLISECOND = 1e-3;
constexpr f64 MEMBRANE_TIME_CONSTANT = 20e-3;
constexpr f64 THRESHOLD = 20e-3;
constexpr f64 RESET = 10e-3;
constexpr f64 REFRACTORY_PERIOD = 2e-3;

// Twice forward Euler's largest difference from the exact solution over the verification's
// first 100 ms (see the top of the file).
constexpr f64 EULER_VOLTAGE_TOLERANCE = 0.002 * MILLIVOLT;

s64 tick_of(f64 time_seconds, f64 step_dt) {
    return static_cast<s64>(std::llround(time_seconds / step_dt));
}

Vector<s64> ticks_of(const Vector<f64> &times, f64 step_dt) {
    Vector<s64> ticks;
    for (f64 time : times) ticks.push_back(tick_of(time, step_dt));
    return ticks;
}

struct Voltage_Jump {
    s64 arrival_tick = 0;
    f64 jump = 0.0;
};

// Neuron 0's v after each tick, the exact event-driven solution of verify_single_neuron.py, on
// whole ticks so no time comparison rounds: on the step ending at a tick's time, v decays by
// exp(-dt / tau) unless refractory, then takes the jumps that have arrived (discarded while
// refractory), then spikes and resets above threshold.
Vector<f64> exact_neuron_0_voltage(s64 tick_count, f64 step_dt, Vector<s64> &spike_ticks) {
    Vector<Voltage_Jump> jumps;
    for (f64 arrival_ms : {11.5, 13.5, 15.5, 17.5, 19.5}) jumps.push_back({tick_of(arrival_ms * MILLISECOND, step_dt), 5.0 * MILLIVOLT});
    for (f64 arrival_ms : {21.0, 41.5}) jumps.push_back({tick_of(arrival_ms * MILLISECOND, step_dt), -3.0 * MILLIVOLT});
    std::sort(jumps.begin(), jumps.end(),
              [](const Voltage_Jump &first, const Voltage_Jump &second) { return first.arrival_tick < second.arrival_tick; });

    Vector<f64> voltage(static_cast<usize>(tick_count), 0.0);
    f64 membrane = 0.0;
    s64 refractory_end_tick = -1;
    usize next_jump = 0;
    for (s64 tick = 1; tick < tick_count; tick += 1) {
        const bool integrating = tick > refractory_end_tick;
        if (integrating) membrane *= std::exp(-step_dt / MEMBRANE_TIME_CONSTANT);
        while (next_jump < jumps.size() && jumps[next_jump].arrival_tick <= tick) {
            if (integrating) membrane += jumps[next_jump].jump;
            next_jump += 1;
        }
        if (membrane > THRESHOLD) {
            spike_ticks.push_back(tick);
            membrane = RESET;
            refractory_end_tick = tick + tick_of(REFRACTORY_PERIOD, step_dt);
        }
        voltage[static_cast<usize>(tick)] = membrane;
    }
    return voltage;
}

f64 mean_and_standard_deviation(const Vector<f64> &values, f64 &standard_deviation) {
    const f64 mean = mean_of(values);
    f64 sum_of_squares = 0.0;
    for (f64 value : values) sum_of_squares += (value - mean) * (value - mean);
    standard_deviation = std::sqrt(sum_of_squares / static_cast<f64>(values.size()));
    return mean;
}

String construction_error(const String &lems) {
    try {
        Spike_Engine engine(lems);
    } catch (const Runtime_Error &error) {
        return error.what();
    }
    return "";
}

} // namespace

// WORKSPACE_MANIFEST.md section 4: every check in its table, against the exact expected values.
TEST(Delta_Synapse, single_neuron_verification_matches_the_exact_solution) {
    Spike_Engine engine(fixture_path("nml/LEMS_brunel_model_a_single_neuron.xml"), false,
                        Spike_Engine::FIT_TOLERANCE_PRECISE);
    const f64 step_dt = engine.step_dt;
    Vector<f64> neuron_0_voltage;
    Vector<f64> neuron_1_voltage;
    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);
        neuron_0_voltage.push_back(engine.read_state_variable(0, "v"));
        neuron_1_voltage.push_back(engine.read_state_variable(1, "v"));
    }
    Vector<s64> expected_spike_ticks;
    const Vector<f64> expected_voltage = exact_neuron_0_voltage(engine.lifetime, step_dt, expected_spike_ticks);

    // The sources fire on their spikes' ticks, so each jump arrives 1.5 ms after its spike.
    EXPECT_EQ(ticks_of(spike_times_of(engine, 2), step_dt), ticks_of({10e-3, 12e-3, 14e-3, 16e-3, 18e-3}, step_dt));
    EXPECT_EQ(ticks_of(spike_times_of(engine, 3), step_dt), ticks_of({19.5e-3, 40e-3}, step_dt));

    // The fifth jump crosses threshold on its own arrival tick, 19.50 ms.
    EXPECT_EQ(expected_spike_ticks, ticks_of({19.5e-3}, step_dt));
    EXPECT_EQ(ticks_of(spike_times_of(engine, 0), step_dt), ticks_of({19.5e-3}, step_dt));
    EXPECT_TRUE(spike_times_of(engine, 1).empty());

    // The table's expected column, and the exact trace at the same ticks.
    const Vector<Pair<f64, f64>> table = {
        {11.6e-3, 4.9751 * MILLIVOLT},   // after the first jump
        {19.4e-3, 15.7521 * MILLIVOLT},  // just before the fifth
        {21.2e-3, 10.0000 * MILLIVOLT},  // refractory, after the discarded inhibitory jump
        {41.4e-3, 3.6972 * MILLIVOLT},   // just before the later inhibitory jump
        {41.6e-3, 0.6754 * MILLIVOLT},   // just after it
    };
    for (const auto &[time, expected] : table) {
        const usize tick = static_cast<usize>(tick_of(time, step_dt));
        EXPECT_NEAR(neuron_0_voltage[tick], expected, EULER_VOLTAGE_TOLERANCE + 0.00005 * MILLIVOLT) << "at " << time * 1e3 << " ms";
        EXPECT_NEAR(neuron_0_voltage[tick], expected_voltage[tick], EULER_VOLTAGE_TOLERANCE) << "at " << time * 1e3 << " ms";
    }

    // Every jump on its arrival tick: a jump a tick early or late would be millivolts off.
    f64 largest_difference = 0.0;
    for (s64 tick = 0; tick <= tick_of(100e-3, step_dt); tick += 1) {
        largest_difference = std::max(largest_difference, std::fabs(neuron_0_voltage[static_cast<usize>(tick)] -
                                                                    expected_voltage[static_cast<usize>(tick)]));
    }
    EXPECT_LT(largest_difference, EULER_VOLTAGE_TOLERANCE);

    // Refractory from the spike until 21.5 ms: held exactly at reset, the jump at 21.0 ms discarded,
    // and consumed, so it is never applied later: nothing is left to apply at the end (the total
    // less the tracker, kept in network_inputs, is 0; the tracker itself is dropped).
    for (s64 tick = tick_of(19.5e-3, step_dt); tick <= tick_of(21.5e-3, step_dt); tick += 1) {
        EXPECT_EQ(static_cast<f32>(neuron_0_voltage[static_cast<usize>(tick)]), static_cast<f32>(RESET)) << "tick " << tick;
    }
    EXPECT_EQ(engine.network_inputs.get_contents_as<f32>()[Codegen::RUNNING_SUM_ROW * engine.total_neuron_count + 0], 0.0f);

    // The jump at 41.5 ms lowers v by exactly 3 mV, after that tick's leak step.
    const usize jump_tick = static_cast<usize>(tick_of(41.5e-3, step_dt));
    const f64 after_leak = neuron_0_voltage[jump_tick - 1] * (1.0 - step_dt / MEMBRANE_TIME_CONSTANT);
    EXPECT_NEAR(neuron_0_voltage[jump_tick] - after_leak, -3.0 * MILLIVOLT, 1e-4 * MILLIVOLT);

    // Neuron 1 under the 1 kHz Poisson drive of 0.1 mV jumps, 200 to 2000 ms: mean rate * jump * tau
    // = 2 mV and SD sqrt(rate * jump^2 * tau / 2) = 0.316 mV. Over T = 1.8 s with correlation time
    // tau, the mean's standard error is sigma sqrt(2 tau / T) = 0.047 mV and the SD's is
    // sigma sqrt(tau / (2 T)) = 0.024 mV; each may be four standard errors off.
    const Vector<f64> steady_state(neuron_1_voltage.begin() + tick_of(200e-3, step_dt), neuron_1_voltage.end());
    f64 standard_deviation = 0.0;
    const f64 mean = mean_and_standard_deviation(steady_state, standard_deviation);
    const f64 expected_mean = 1000.0 * 0.1 * MILLIVOLT * MEMBRANE_TIME_CONSTANT;
    const f64 expected_standard_deviation = std::sqrt(1000.0 * 0.1 * MILLIVOLT * 0.1 * MILLIVOLT * MEMBRANE_TIME_CONSTANT / 2.0);
    const f64 window = static_cast<f64>(steady_state.size()) * step_dt;
    EXPECT_NEAR(mean, expected_mean, 4.0 * expected_standard_deviation * std::sqrt(2.0 * MEMBRANE_TIME_CONSTANT / window));
    EXPECT_NEAR(standard_deviation, expected_standard_deviation,
                4.0 * expected_standard_deviation * std::sqrt(MEMBRANE_TIME_CONSTANT / (2.0 * window)));
}

// effective_delay_test.py: NEST fires the presynaptic neuron at 10.1 ms and the postsynaptic one
// at 11.6 ms, so the spike-to-spike latency through a 1.5 ms synapse is exactly 1.5 ms. So is the
// 0.1 ms one, a single tick: the arrival is sent on the spike's own tick.
TEST(Delta_Synapse, spike_to_spike_latency_is_the_delay) {
    Spike_Engine engine(fixture_path("nml/LEMS_brunel_model_a_effective_delay.xml"));
    engine.run();
    const f64 step_dt = engine.step_dt;

    const Vector<f64> trigger = spike_times_of(engine, 0);
    const Vector<f64> presynaptic = spike_times_of(engine, 1);
    const Vector<f64> postsynaptic = spike_times_of(engine, 2);
    ASSERT_EQ(trigger.size(), 1u);
    ASSERT_EQ(presynaptic.size(), 1u);
    ASSERT_EQ(postsynaptic.size(), 1u);
    EXPECT_EQ(tick_of(trigger[0], step_dt), tick_of(10.0e-3, step_dt));
    EXPECT_EQ(tick_of(presynaptic[0], step_dt), tick_of(10.1e-3, step_dt));
    EXPECT_EQ(tick_of(postsynaptic[0], step_dt), tick_of(11.6e-3, step_dt));
    EXPECT_EQ(tick_of(postsynaptic[0] - presynaptic[0], step_dt), tick_of(1.5e-3, step_dt));
}

// Different delays from one source each arrive exactly: with a shortest delay above one tick the
// kernel walks a neuron's edges only on ticks one of its spikes is due somewhere.
TEST(Delta_Synapse, each_delay_of_a_source_arrives_exactly) {
    const Temporary_Directory directory;
    const String lems = directory.write("LEMS.xml", R"(<Lems>
  <Target component="delay_simulation"/>
  <Include file=")" + fixture_path("nml/Brunel2000_ModelA_Components.xml") + R"("/>
  <deltaVoltageSynapse id="suprathreshold_delta_synapse" jump="25mV"/>
  <spikeArray id="trigger_spikes">
    <spike id="0" time="10ms"/>
    <spike id="1" time="20ms"/>
  </spikeArray>
  <iafCell id="trigger_cell" leakConductance="10nS" leakReversal="-70mV" thresh="-50mV" reset="-70mV" C="200pF"/>
  <network id="delay_network">
    <population id="trigger" component="trigger_cell" size="1"/>
    <population id="targets" component="brunel_model_a_neuron" size="3"/>
    <projection id="delay_projection" presynapticPopulation="trigger" postsynapticPopulation="targets"
                synapse="suprathreshold_delta_synapse">
      <connectionWD id="0" preCellId="../trigger[0]" postCellId="../targets[0]" weight="1" delay="0.5ms"/>
      <connectionWD id="1" preCellId="../trigger[0]" postCellId="../targets[1]" weight="1" delay="1.2ms"/>
      <connectionWD id="2" preCellId="../trigger[0]" postCellId="../targets[2]" weight="1" delay="2ms"/>
    </projection>
    <explicitInput target="trigger[0]" input="trigger_spikes"/>
  </network>
  <Simulation id="delay_simulation" length="30ms" step="0.1ms" target="delay_network" seed="1"/>
</Lems>
)");
    Spike_Engine engine(lems);
    engine.run();
    const f64 step_dt = engine.step_dt;

    EXPECT_EQ(ticks_of(spike_times_of(engine, 0), step_dt), ticks_of({10e-3, 20e-3}, step_dt));
    EXPECT_EQ(ticks_of(spike_times_of(engine, 1), step_dt), ticks_of({10.5e-3, 20.5e-3}, step_dt));
    EXPECT_EQ(ticks_of(spike_times_of(engine, 2), step_dt), ticks_of({11.2e-3, 21.2e-3}, step_dt));
    EXPECT_EQ(ticks_of(spike_times_of(engine, 3), step_dt), ticks_of({12e-3, 22e-3}, step_dt));
}

// A weighted arrival keeps no state per connection, so the weight matrix has only the weight and
// delay planes, the kernel updates none, and a neuron's edges run only until its last arrival is
// sent: the longest delay less a tick. Nothing has to settle.
TEST(Delta_Synapse, keeps_no_state_per_connection) {
    Captured_Log log;
    Spike_Engine engine(fixture_path("nml/LEMS_brunel_model_a_effective_delay.xml"));

    EXPECT_EQ(engine.weights.matrix_count, Weight_Matrix::FIRST_STATE_VARIABLE_PLANE);
    EXPECT_EQ(engine.weights.updated_plane_count, 0);
    EXPECT_EQ(engine.synapse_active_ticks, tick_of(1.5e-3, engine.step_dt) - 1);
    EXPECT_FALSE(log.contains("does not settle"));
    EXPECT_EQ(engine.network_inputs.total_bytes,
              sizeof(f32) * static_cast<u64>((Codegen::RUNNING_SUM_ROW + 1) * engine.total_neuron_count));

    try {
        (void)engine.synapse_state_variable_plane("suprathreshold_delta_synapse", "accumulated_jump");
        ADD_FAILURE() << "synapse_state_variable_plane gave a plane for a weighted arrival";
    } catch (const Runtime_Error &error) {
        EXPECT_NE(String(error.what()).find("keeps no 'accumulated_jump' per connection"), String::npos) << error.what();
    }
}

// The engine runs a spikeArray's spikes only for an input; as a population's cell it would never
// fire, so it is refused.
TEST(Delta_Synapse, a_spike_array_population_is_refused) {
    const Temporary_Directory directory;
    const String model = R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="SpikeArrayPopulation">
  <spikeArray id="scripted">
    <spike id="0" time="10ms"/>
  </spikeArray>
  <iafCell id="target" leakConductance="10nS" leakReversal="-70mV" thresh="-50mV" reset="-70mV" C="200pF"/>
  <expCurrSynapse id="synapse" tau_syn="5"/>
  <network id="spikeArrayNetwork">
    <population id="sources" component="scripted" size="1"/>
    <population id="targets" component="target" size="1"/>
  </network>
</neuroml>
)";
    directory.write("model.nml", model);
    const String lems = directory.write("LEMS.xml", lems_document("model.nml", "spikeArrayNetwork", "20ms", "0.1ms"));
    const String error = construction_error(lems);
    EXPECT_NE(error.find("sources' is made of 'scripted' (spikeArray)"), String::npos) << error;
    EXPECT_NE(error.find("would never fire"), String::npos) << error;
}

// A target reads the exposure its select names. Here a synapse's state x decays between arrivals,
// so it runs every tick of its active window, and x itself (not i, which is 0) reaches the target:
// the target integrates dv/dt = x_total / tau_integration, so v gains the area under x, which
// forward Euler sums as weight * jump * tau_decay / tau_integration.
TEST(Delta_Synapse, a_stepped_synapse_delivers_the_exposure_its_target_reads) {
    const Temporary_Directory directory;
    directory.write("types.xml", R"(<Lems>
  <Include file="Cells.xml"/>
  <Include file="Synapses.xml"/>
  <ComponentType name="decayingVoltageSynapse" extends="baseSynapse">
    <Property name="weight" dimension="none" defaultValue="1"/>
    <Parameter name="jump" dimension="voltage"/>
    <Parameter name="tau_decay" dimension="time"/>
    <Exposure name="x" dimension="voltage"/>
    <Dynamics>
      <StateVariable name="x" dimension="voltage" exposure="x"/>
      <DerivedVariable name="i" dimension="current" exposure="i" value="0"/>
      <TimeDerivative variable="x" value="-x / tau_decay"/>
      <OnEvent port="in">
        <StateAssignment variable="x" value="x + weight * jump"/>
      </OnEvent>
    </Dynamics>
  </ComponentType>
  <ComponentType name="voltageIntegratorCell" extends="baseCellMembPot">
    <Parameter name="tau_integration" dimension="time"/>
    <Attachments name="synapses" type="baseSynapse"/>
    <Dynamics>
      <StateVariable name="v" dimension="voltage" exposure="v"/>
      <DerivedVariable name="x_total" dimension="voltage" select="synapses[*]/x" reduce="add"/>
      <TimeDerivative variable="v" value="x_total / tau_integration"/>
    </Dynamics>
  </ComponentType>
</Lems>
)");
    directory.write("model.nml", R"(<Lems>
  <Include file="types.xml"/>
  <decayingVoltageSynapse id="synapse" jump="1mV" tau_decay="2ms"/>
  <voltageIntegratorCell id="integrator" tau_integration="10ms"/>
  <iafCell id="source_cell" leakConductance="10nS" leakReversal="-70mV" thresh="-50mV" reset="-70mV" C="200pF"/>
  <spikeArray id="trigger_spikes">
    <spike id="0" time="5ms"/>
  </spikeArray>
  <network id="integratorNetwork">
    <population id="source" component="source_cell" size="1"/>
    <population id="target" component="integrator" size="1"/>
    <projection id="projection" presynapticPopulation="source" postsynapticPopulation="target" synapse="synapse">
      <connectionWD id="0" preCellId="../source[0]" postCellId="../target[0]" weight="2" delay="1ms"/>
    </projection>
    <explicitInput target="source[0]" input="trigger_spikes"/>
  </network>
</Lems>
)");
    const String lems = directory.write("LEMS.xml", lems_document("model.nml", "integratorNetwork", "100ms", "0.1ms"));
    Spike_Engine engine(lems, false, Spike_Engine::FIT_TOLERANCE_PRECISE);
    engine.run();

    ASSERT_EQ(spike_times_of(engine, 0).size(), 1u);
    // The geometric sum of x over the ticks after the arrival: weight * jump * tau_decay / dt.
    const f64 expected = 2.0 * 1e-3 * 2e-3 / 10e-3;
    EXPECT_NEAR(engine.read_state_variable(1, "v"), expected, 1e-3 * expected);
}

// ── tracked totals ──────────────────────────────────────────────────────────────
//
// The model A neuron reads its total only as total_jump - applied_jump, and applied_jump is
// assigned only total_jump, so the kernel keeps their difference, not either lifetime value, which
// in f32 rounds every jump to its own spacing once it has grown large.

namespace {

// The model A neuron with no threshold, driven by raising and lowering Poisson inputs of 50 mV at
// 2 kHz each: each input's lifetime total passes 100 V within the second, where f32 spacing is
// 7.6 µV, while v stays near 0.
String unbounded_drive_model() {
    return R"(<Lems>
  <Target component="drive_simulation"/>
  <Include file=")" + fixture_path("nml/Brunel2000_ModelA_Components.xml") + R"("/>
  <iafRefCellDeltaInput id="unbounded_neuron" leak_reversal="0mV" membrane_time_constant="20ms" refractory_period="2ms"
                        thresh="1000V" reset="0mV"/>
  <poissonDeltaVoltageInput id="raising_drive" average_rate="2000Hz" jump="50mV"/>
  <poissonDeltaVoltageInput id="lowering_drive" average_rate="2000Hz" jump="-50mV"/>
  <network id="drive_network">
    <population id="neurons" component="unbounded_neuron" size="1"/>
    <explicitInput target="neurons[0]" input="raising_drive" destination="synapses"/>
    <explicitInput target="neurons[0]" input="lowering_drive" destination="synapses"/>
  </network>
  <Simulation id="drive_simulation" length="1000ms" step="0.1ms" target="drive_network" seed="7"/>
</Lems>
)";
}

// The model A neuron's dynamics with one more DerivedVariable that reads total_jump, which the
// shift of a tracked total would change, so its total is kept whole. It behaves the same.
String total_reading_cell_types() {
    return R"(<Lems>
  <Include file=")" + fixture_path("nml/Brunel2000_ModelA_ComponentTypes.xml") + R"lems("/>
  <ComponentType name="iafRefCellDeltaInputReadingTotal" extends="baseIaf">
    <Parameter name="leak_reversal" dimension="voltage"/>
    <Parameter name="membrane_time_constant" dimension="time"/>
    <Parameter name="refractory_period" dimension="time"/>
    <Attachments name="synapses" type="baseDeltaVoltageSynapse"/>
    <Exposure name="total_jump_seen" dimension="voltage"/>
    <Dynamics>
      <StateVariable name="v" dimension="voltage" exposure="v"/>
      <StateVariable name="applied_jump" dimension="voltage"/>
      <StateVariable name="refractory_end_time" dimension="time"/>
      <DerivedVariable name="total_jump" dimension="voltage" select="synapses[*]/accumulated_jump" reduce="add"/>
      <DerivedVariable name="total_jump_seen" dimension="voltage" exposure="total_jump_seen" value="total_jump"/>
      <DerivedVariable name="pending_jump" dimension="voltage" value="total_jump - applied_jump"/>
      <DerivedVariable name="is_integrating" dimension="none" value="H((t - refractory_end_time) / membrane_time_constant)"/>
      <TimeDerivative variable="v" value="is_integrating * (leak_reversal - v) / membrane_time_constant"/>
      <OnStart>
        <StateAssignment variable="v" value="leak_reversal"/>
        <StateAssignment variable="applied_jump" value="0"/>
        <StateAssignment variable="refractory_end_time" value="-1 * refractory_period"/>
      </OnStart>
      <OnCondition test="pending_jump .neq. 0">
        <StateAssignment variable="v" value="v + is_integrating * pending_jump"/>
        <StateAssignment variable="applied_jump" value="total_jump"/>
      </OnCondition>
      <OnCondition test="v .gt. thresh">
        <EventOut port="spike"/>
        <StateAssignment variable="v" value="reset"/>
        <StateAssignment variable="refractory_end_time" value="t + refractory_period"/>
      </OnCondition>
    </Dynamics>
  </ComponentType>
</Lems>
)lems";
}

} // namespace

// Every jump is a whole number of 50 mV, so v after a tick less v before it, leaked by the step's
// forward Euler, is one too, up to f32 rounding of v itself (v stays within about 1.5 V, where f32
// spacing is 0.12 µV). Lifetime totals of 100 V would leave up to half their 7.6 µV spacing.
TEST(Tracked_Total, keeps_every_jump_exact_however_large_the_totals_grow) {
    Captured_Log log;
    const Temporary_Directory directory;
    Spike_Engine engine(directory.write("LEMS.xml", unbounded_drive_model()));
    EXPECT_TRUE(log.contains("'iafRefCellDeltaInput' reads its total 'total_jump' only less 'applied_jump'"));

    constexpr f64 JUMP = 50e-3;
    f64 largest_residual = 0.0;
    s64 jump_ticks = 0;
    f32 previous_voltage = engine.read_state_variable(0, "v");
    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);
        const f32 voltage = engine.read_state_variable(0, "v");
        // The kernel's leak step: v += dt * (leak_reversal - v) / tau, in f32.
        const f32 leaked = previous_voltage + 0.0001f * ((1.0f * (0.0f - previous_voltage)) / 0.02f);
        const f64 jump = static_cast<f64>(voltage) - static_cast<f64>(leaked);
        if (std::fabs(jump) > 0.5 * JUMP) {
            largest_residual = std::max(largest_residual, std::fabs(jump - JUMP * std::round(jump / JUMP)));
            jump_ticks += 1;
        }
        previous_voltage = voltage;
    }
    EXPECT_GT(jump_ticks, 1000);
    EXPECT_LT(largest_residual, 1e-6);
    // Nothing reads applied_jump any more, and nothing records it, so it has no slot.
    EXPECT_THROW((void)engine.read_state_variable(0, "applied_jump"), Runtime_Error);
}

// A cell type that reads its total anywhere but in total - tracker keeps the lifetime total, and
// behaves as a tracked one: the same scripted jumps, discarded while refractory, give the same v.
TEST(Tracked_Total, a_cell_that_reads_its_total_elsewhere_keeps_it_whole_and_behaves_the_same) {
    Captured_Log log;
    const Temporary_Directory directory;
    directory.write("types.xml", total_reading_cell_types());
    const String lems = directory.write("LEMS.xml", R"(<Lems>
  <Target component="comparison_simulation"/>
  <Include file=")" + fixture_path("nml/Brunel2000_ModelA_Components.xml") + R"("/>
  <Include file="types.xml"/>
  <iafRefCellDeltaInputReadingTotal id="total_reading_neuron" leak_reversal="0mV" membrane_time_constant="20ms"
                                    refractory_period="2ms" thresh="20mV" reset="10mV"/>
  <spikeArray id="excitatory_test_spikes">
    <spike id="0" time="10ms"/> <spike id="1" time="12ms"/> <spike id="2" time="14ms"/>
    <spike id="3" time="16ms"/> <spike id="4" time="18ms"/>
  </spikeArray>
  <spikeArray id="inhibitory_test_spikes">
    <spike id="0" time="19.5ms"/> <spike id="1" time="40ms"/>
  </spikeArray>
  <iafCell id="spike_source_cell" leakConductance="10nS" leakReversal="-70mV" thresh="-50mV" reset="-70mV" C="200pF"/>
  <network id="comparison_network">
    <population id="tracked" component="brunel_model_a_neuron" size="1"/>
    <population id="whole" component="total_reading_neuron" size="1"/>
    <population id="excitatory_source" component="spike_source_cell" size="1"/>
    <population id="inhibitory_source" component="spike_source_cell" size="1"/>
    <projection id="excitatory_projection" presynapticPopulation="excitatory_source" postsynapticPopulation="tracked"
                synapse="excitatory_delta_synapse">
      <connectionWD id="0" preCellId="../excitatory_source[0]" postCellId="../tracked[0]" weight="50" delay="1.5ms"/>
    </projection>
    <projection id="excitatory_projection_whole" presynapticPopulation="excitatory_source" postsynapticPopulation="whole"
                synapse="excitatory_delta_synapse">
      <connectionWD id="0" preCellId="../excitatory_source[0]" postCellId="../whole[0]" weight="50" delay="1.5ms"/>
    </projection>
    <projection id="inhibitory_projection" presynapticPopulation="inhibitory_source" postsynapticPopulation="tracked"
                synapse="inhibitory_delta_synapse">
      <connectionWD id="0" preCellId="../inhibitory_source[0]" postCellId="../tracked[0]" weight="30" delay="1.5ms"/>
    </projection>
    <projection id="inhibitory_projection_whole" presynapticPopulation="inhibitory_source" postsynapticPopulation="whole"
                synapse="inhibitory_delta_synapse">
      <connectionWD id="0" preCellId="../inhibitory_source[0]" postCellId="../whole[0]" weight="30" delay="1.5ms"/>
    </projection>
    <explicitInput target="excitatory_source[0]" input="excitatory_test_spikes"/>
    <explicitInput target="inhibitory_source[0]" input="inhibitory_test_spikes"/>
  </network>
  <Simulation id="comparison_simulation" length="60ms" step="0.01ms" target="comparison_network" seed="1"/>
</Lems>
)");
    Spike_Engine engine(lems, false, Spike_Engine::FIT_TOLERANCE_PRECISE);
    EXPECT_TRUE(log.contains("'iafRefCellDeltaInput' reads its total"));
    EXPECT_FALSE(log.contains("'iafRefCellDeltaInputReadingTotal' reads its total"));

    f64 largest_difference = 0.0;
    for (s64 tick = 0; tick < engine.lifetime; tick += 1) {
        engine.step_simulation(tick);
        largest_difference = std::max(largest_difference, std::fabs(static_cast<f64>(engine.read_state_variable(0, "v")) -
                                                                    static_cast<f64>(engine.read_state_variable(1, "v"))));
    }
    EXPECT_EQ(ticks_of(spike_times_of(engine, 0), engine.step_dt), ticks_of({19.5e-3}, engine.step_dt));
    EXPECT_EQ(ticks_of(spike_times_of(engine, 1), engine.step_dt), ticks_of({19.5e-3}, engine.step_dt));
    // The two neurons get the same jumps through different edges, each fitted only to within the fit
    // tolerance, so each is held to its own edges' fitted weights (neurons 0 tracked, 1 whole, 2 and
    // 3 the excitatory and inhibitory sources; a weight unit is 0.1 mV). Their v then differ by the
    // fitted weights' differences, over the 5 excitatory and 2 inhibitory jumps, and by f32 rounding
    // of v, near 1 nV here. The whole one's applied_jump is every jump's sum, discarded ones
    // included; the tracked one's is dropped.
    const f64 jump_per_weight = 0.1 * MILLIVOLT;
    const f64 excitatory_weight_difference = std::fabs(static_cast<f64>(engine.weights.get(2, 0)) - engine.weights.get(2, 1));
    const f64 inhibitory_weight_difference = std::fabs(static_cast<f64>(engine.weights.get(3, 0)) - engine.weights.get(3, 1));
    EXPECT_LT(largest_difference,
              (5.0 * excitatory_weight_difference + 2.0 * inhibitory_weight_difference) * jump_per_weight + 1e-8);
    EXPECT_NEAR(engine.read_state_variable(1, "applied_jump"),
                (5.0 * engine.weights.get(2, 1) - 2.0 * engine.weights.get(3, 1)) * jump_per_weight, 1e-8);
    EXPECT_THROW((void)engine.read_state_variable(0, "applied_jump"), Runtime_Error);
}

// A tracker an OutputColumn records keeps its slot and reads the total applied so far, here the one
// 25 mV jump each delta neuron took, while both still fire exactly a delay after their input.
TEST(Tracked_Total, a_recorded_tracker_is_kept) {
    Captured_Log log;
    const Temporary_Directory directory;
    const String lems = directory.write("LEMS.xml", R"(<Lems>
  <Target component="delay_simulation"/>
  <Include file=")" + fixture_path("nml/Brunel2000_ModelA_Components.xml") + R"("/>
  <deltaVoltageSynapse id="suprathreshold_delta_synapse" jump="25mV"/>
  <spikeArray id="trigger_spikes">
    <spike id="0" time="10ms"/>
  </spikeArray>
  <iafCell id="trigger_cell" leakConductance="10nS" leakReversal="-70mV" thresh="-50mV" reset="-70mV" C="200pF"/>
  <network id="delay_network">
    <population id="trigger" component="trigger_cell" size="1"/>
    <population id="presynaptic" component="brunel_model_a_neuron" size="1"/>
    <population id="postsynaptic" component="brunel_model_a_neuron" size="1"/>
    <projection id="trigger_projection" presynapticPopulation="trigger" postsynapticPopulation="presynaptic"
                synapse="suprathreshold_delta_synapse">
      <connectionWD id="0" preCellId="../trigger[0]" postCellId="../presynaptic[0]" weight="1" delay="0.1ms"/>
    </projection>
    <projection id="delay_projection" presynapticPopulation="presynaptic" postsynapticPopulation="postsynaptic"
                synapse="suprathreshold_delta_synapse">
      <connectionWD id="0" preCellId="../presynaptic[0]" postCellId="../postsynaptic[0]" weight="1" delay="1.5ms"/>
    </projection>
    <explicitInput target="trigger[0]" input="trigger_spikes"/>
  </network>
  <Simulation id="delay_simulation" length="30ms" step="0.1ms" target="delay_network" seed="1">
    <OutputFile id="applied" fileName="applied.dat">
      <OutputColumn id="presynaptic_applied" quantity="presynaptic[0]/applied_jump"/>
    </OutputFile>
  </Simulation>
</Lems>
)");
    Spike_Engine engine(lems);
    EXPECT_TRUE(log.contains("'applied_jump' is recorded, so kept"));
    engine.run();

    EXPECT_EQ(ticks_of(spike_times_of(engine, 1), engine.step_dt), ticks_of({10.1e-3}, engine.step_dt));
    EXPECT_EQ(ticks_of(spike_times_of(engine, 2), engine.step_dt), ticks_of({11.6e-3}, engine.step_dt));
    EXPECT_NEAR(engine.read_state_variable(1, "applied_jump"), 25.0 * MILLIVOLT, 1e-3 * 25.0 * MILLIVOLT);
    EXPECT_NEAR(engine.read_state_variable(2, "applied_jump"), 25.0 * MILLIVOLT, 1e-3 * 25.0 * MILLIVOLT);
    ASSERT_FALSE(engine.recorded_traces.empty());
    EXPECT_NEAR(engine.recorded_traces.back(), 25.0 * MILLIVOLT, 1e-3 * 25.0 * MILLIVOLT);
}
