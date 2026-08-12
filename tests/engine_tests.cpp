#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "spikecorec/core/types.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/engine.h"
#include "spikecorec/core/recording.h"

using namespace std;
using namespace spikecorec;

namespace {

// Fixtures are written to disk because the engine's unit of work is a file path: the
// parser resolves <Include>s against the including file's own directory, and a
// string-fed engine would exercise none of that. Each test gets its own directory.
class FixtureDirectory {
public:
    explicit FixtureDirectory(const String &test_name) {
        root_ = filesystem::temp_directory_path() / "spikecorec_engine_tests" / test_name;
        filesystem::remove_all(root_);
        filesystem::create_directories(root_);
    }

    ~FixtureDirectory() {
        std::error_code ignored;
        filesystem::remove_all(root_, ignored);
    }

    FixtureDirectory(const FixtureDirectory &) = delete;
    FixtureDirectory &operator=(const FixtureDirectory &) = delete;

    // Returns the absolute path of the written file.
    String write(const String &relative_name, const String &contents) const {
        filesystem::path destination = root_ / relative_name;
        filesystem::create_directories(destination.parent_path());

        ofstream file(destination);
        file << contents;
        file.close();

        return destination.string();
    }

    String path_of(const String &relative_name) const {
        return (root_ / relative_name).string();
    }

private:
    filesystem::path root_;
};

bool standard_library_available() {
    nml::NML_Parser parser;
    return !parser.STANDARD_LIBRARY_PATH.empty() &&
           filesystem::exists(parser.STANDARD_LIBRARY_PATH);
}

// Two cell types, deliberately of different widths, so the type-sectioned cell_state /
// cell_parameters layout is observable rather than degenerate.
//
// The ComponentTypes are declared inline: the vendored standard library has no GLIF type,
// and every standard-library cell that would otherwise fit (iafCell and friends) reduces
// its synaptic input through a `select="synapses[*]/i"` path, which the kernel generator
// does not yet lower.
String two_cell_type_lems_xml(const String &network_file, const String &recording_file) {
    return R"(<Lems>
    <Target component="sim1"/>

    <ComponentType name="simpleLifCell" extends="baseSpikingCell"
                   description="Leaky integrator relaxing towards restingPotential.">
        <Parameter name="tau" dimension="time"/>
        <Parameter name="restingPotential" dimension="voltage"/>
        <Parameter name="threshold" dimension="voltage"/>
        <Parameter name="resetPotential" dimension="voltage"/>
        <Parameter name="startPotential" dimension="voltage"/>

        <Dynamics>
            <StateVariable name="v" dimension="voltage"/>

            <TimeDerivative variable="v" value="(restingPotential - v) / tau"/>

            <OnStart>
                <StateAssignment variable="v" value="startPotential"/>
            </OnStart>

            <OnCondition test="v .gt. threshold">
                <StateAssignment variable="v" value="resetPotential"/>
                <EventOut port="spike"/>
            </OnCondition>
        </Dynamics>
    </ComponentType>

    <ComponentType name="dualStateCell" extends="baseCell"
                   description="Two state variables, so its section is twice as wide.">
        <Parameter name="tau" dimension="time"/>
        <Parameter name="startPotential" dimension="voltage"/>

        <Dynamics>
            <StateVariable name="v" dimension="voltage"/>
            <StateVariable name="w" dimension="voltage"/>

            <TimeDerivative variable="v" value="-v / tau"/>
            <TimeDerivative variable="w" value="v / tau"/>

            <OnStart>
                <StateAssignment variable="v" value="startPotential"/>
                <StateAssignment variable="w" value="0"/>
            </OnStart>
        </Dynamics>
    </ComponentType>

    <Include file=")" + network_file + R"("/>

    <Simulation id="sim1" length="5ms" step="0.1ms" target="net1" seed="4242">
        <OutputFile id="of1" fileName=")" + recording_file + R"(">
            <OutputColumn id="c0" quantity="pop1[0]/v"/>
        </OutputFile>
    </Simulation>
</Lems>
)";
}

// pop1 spikes (tau is short enough that v crosses threshold well inside the run), pop2
// does not. Both prototypes of simpleLifCell differ, so per-prototype parameters are
// exercised too.
String two_cell_type_network_nml() {
    return R"(<neuroml id="enginenet">
    <simpleLifCell id="fastCell" tau="1ms" restingPotential="-50mV" threshold="-55mV"
                   resetPotential="-70mV" startPotential="-70mV"/>
    <dualStateCell id="dualCell" tau="2ms" startPotential="-60mV"/>
    <expOneSynapse id="syn0" gbase="0.5nS" erev="0mV" tauDecay="3ms"/>
    <pulseGenerator id="pg0" delay="0.5ms" duration="1ms" amplitude="0.5nA"/>

    <network id="net1">
        <population id="pop1" component="fastCell" size="2"/>
        <population id="pop2" component="dualCell" size="2"/>

        <projection id="proj1" presynapticPopulation="pop1"
                    postsynapticPopulation="pop2" synapse="syn0">
            <connectionWD id="0" preCellId="../pop1[0]" postCellId="../pop2[1]"
                          weight="2.5" delay="0.3ms"/>
        </projection>

        <explicitInput target="pop1[0]" input="pg0"/>
    </network>
</neuroml>
)";
}

// Writes both fixture files and returns the LEMS main file's path, which is what the
// engine is handed.
String write_two_cell_type_model(const FixtureDirectory &fixture) {
    fixture.write("net.nml", two_cell_type_network_nml());
    return fixture.write("model.xml",
                         two_cell_type_lems_xml("net.nml", fixture.path_of("out.spire")));
}

} // namespace

// ── backend / types ────────────────────────────────────────────────────────────

TEST(Backend, types_layout) {
    EXPECT_EQ(sizeof(u8), 1u);
    EXPECT_EQ(sizeof(u16), 2u);
    EXPECT_EQ(sizeof(u32), 4u);
    EXPECT_EQ(sizeof(u64), 8u);
    EXPECT_EQ(sizeof(s32), 4u);
    EXPECT_EQ(sizeof(s64), 8u);
    EXPECT_EQ(sizeof(f32), 4u);
    EXPECT_EQ(sizeof(f64), 8u);
    EXPECT_EQ(sizeof(spikecorec::float4), 16u);
    EXPECT_EQ(alignof(spikecorec::float4), 16u);
}

TEST(Backend, gpu_pointer_alloc) {
    const usize neuron_count = 32;
    GpuPointer<f32> buffer = allocate<f32>(neuron_count * sizeof(f32));
    f32 *data = buffer.get_contents();
    ASSERT_NE(data, nullptr);

    for (usize index = 0; index < neuron_count; ++index) data[index] = (f32)index * 1.5f;
    for (usize index = 0; index < neuron_count; ++index) EXPECT_EQ(data[index], (f32)index * 1.5f);

    GpuPointer<f32> moved = std::move(buffer);
    EXPECT_EQ(moved.get_contents(), data);
    EXPECT_EQ(buffer.pointer, nullptr);

    deallocate(std::move(moved));
}

// ── create_event_stream ────────────────────────────────────────────────────────

TEST(CreateEventStream, spike_train_is_zero_between_events) {
    const spikecorec::Vector<s32> event_ticks = {2, 5};
    const spikecorec::Vector<f64> stream = create_event_stream(/*rate=*/0.0, /*amplitude=*/0.0,
                                                   /*weight=*/3.0, event_ticks,
                                                   /*continuous_current_injection=*/false);

    ASSERT_EQ(stream.size(), 6u);
    EXPECT_DOUBLE_EQ(stream[0], 0.0);
    EXPECT_DOUBLE_EQ(stream[1], 0.0);
    EXPECT_DOUBLE_EQ(stream[2], 3.0);
    EXPECT_DOUBLE_EQ(stream[3], 0.0);
    EXPECT_DOUBLE_EQ(stream[4], 0.0);
    EXPECT_DOUBLE_EQ(stream[5], 3.0);
}

TEST(CreateEventStream, continuous_injection_fills_its_whole_window) {
    // {start_tick, end_tick}: a window current flows across, not two isolated impulses.
    const spikecorec::Vector<s32> event_ticks = {2, 5};
    const spikecorec::Vector<f64> stream = create_event_stream(/*rate=*/0.0, /*amplitude=*/0.5,
                                                   /*weight=*/2.0, event_ticks,
                                                   /*continuous_current_injection=*/true);

    ASSERT_EQ(stream.size(), 6u);
    EXPECT_DOUBLE_EQ(stream[0], 0.0);
    EXPECT_DOUBLE_EQ(stream[1], 0.0);
    for (usize tick = 2; tick <= 5; ++tick) EXPECT_DOUBLE_EQ(stream[tick], 1.0);
}

TEST(CreateEventStream, magnitude_comes_from_amplitude_then_rate_then_weight) {
    const spikecorec::Vector<s32> event_ticks = {0};

    const spikecorec::Vector<f64> from_amplitude =
            create_event_stream(/*rate=*/7.0, /*amplitude=*/0.25, /*weight=*/2.0, event_ticks);
    const spikecorec::Vector<f64> from_rate =
            create_event_stream(/*rate=*/7.0, /*amplitude=*/0.0, /*weight=*/2.0, event_ticks);
    const spikecorec::Vector<f64> from_weight =
            create_event_stream(/*rate=*/0.0, /*amplitude=*/0.0, /*weight=*/2.0, event_ticks);

    EXPECT_DOUBLE_EQ(from_amplitude[0], 0.5);
    EXPECT_DOUBLE_EQ(from_rate[0], 14.0);
    EXPECT_DOUBLE_EQ(from_weight[0], 2.0);
}

TEST(CreateEventStream, no_events_produces_no_stream) {
    EXPECT_TRUE(create_event_stream(0.0, 0.0, 1.0, {}).empty());
}

// ── construction from a NeuroML/LEMS model ─────────────────────────────────────

TEST(SpikeEngine, neuroml_construction_builds_type_sectioned_layout) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    FixtureDirectory fixture("neuroml_construction");
    String model_path = write_two_cell_type_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    EXPECT_TRUE(engine.alive);
    EXPECT_EQ(engine.total_neuron_count, 4);
    ASSERT_EQ(engine.network_details.cell_types.size(), 2u);
    EXPECT_EQ(engine.network_details.cell_types[0].name, "simpleLifCell");
    EXPECT_EQ(engine.network_details.cell_types[1].name, "dualStateCell");

    // simpleLifCell: 1 state variable, 5 parameters, 2 neurons.
    // dualStateCell: 2 state variables, 2 parameters, 2 neurons.
    EXPECT_EQ(engine.cell_state_element_count, 2 * 1 + 2 * 2);
    EXPECT_EQ(engine.cell_parameter_element_count, 2 * 5 + 2 * 2);

    const s32 *cell_state_base = engine.cell_state_base.get_contents();
    const s32 *cell_parameter_base = engine.cell_parameter_base.get_contents();
    const s32 *cell_type_index = engine.cell_type_index.get_contents();

    EXPECT_EQ(cell_state_base[0], 0);
    EXPECT_EQ(cell_state_base[1], 1);
    EXPECT_EQ(cell_state_base[2], 2);
    EXPECT_EQ(cell_state_base[3], 4);

    EXPECT_EQ(cell_parameter_base[0], 0);
    EXPECT_EQ(cell_parameter_base[1], 5);
    EXPECT_EQ(cell_parameter_base[2], 10);
    EXPECT_EQ(cell_parameter_base[3], 12);

    EXPECT_EQ(cell_type_index[0], 0);
    EXPECT_EQ(cell_type_index[1], 0);
    EXPECT_EQ(cell_type_index[2], 1);
    EXPECT_EQ(cell_type_index[3], 1);

    // Simulation-level bookkeeping: 5ms at 0.1ms per tick.
    EXPECT_EQ(engine.lifetime, 50);
    EXPECT_EQ(engine.simulation_seed, 4242u);
    EXPECT_FALSE(engine.active_set_optimization_enabled);

    engine.shutdown();
    EXPECT_FALSE(engine.alive);
}

TEST(SpikeEngine, neuroml_construction_loads_starting_parameters_and_state) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    FixtureDirectory fixture("neuroml_parameters");
    String model_path = write_two_cell_type_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    const f32 *cell_parameters = engine.cell_parameters.get_contents();
    const s32 *cell_parameter_base = engine.cell_parameter_base.get_contents();
    const f32 *cell_state = engine.cell_state.get_contents();
    const s32 *cell_state_base = engine.cell_state_base.get_contents();

    // Parameters are laid out in the type's declaration order, in SI units.
    const spikecorec::Vector<String> &lif_parameter_names =
            engine.network_details.cell_types[0].parameter_names;
    ASSERT_EQ(lif_parameter_names.size(), 5u);
    EXPECT_EQ(lif_parameter_names[0], "tau");
    EXPECT_EQ(lif_parameter_names[1], "restingPotential");

    EXPECT_NEAR(cell_parameters[cell_parameter_base[0] + 0], 1e-3f, 1e-9f);   // tau = 1ms
    EXPECT_NEAR(cell_parameters[cell_parameter_base[0] + 1], -0.05f, 1e-9f);  // -50mV
    EXPECT_NEAR(cell_parameters[cell_parameter_base[2] + 0], 2e-3f, 1e-9f);   // dual tau = 2ms
    EXPECT_NEAR(cell_parameters[cell_parameter_base[2] + 1], -0.06f, 1e-9f);  // -60mV

    // The OnStart bodies ran: every neuron starts at its own startPotential, and
    // dualStateCell's second state variable starts at zero.
    EXPECT_NEAR(cell_state[cell_state_base[0]], -0.07f, 1e-6f);
    EXPECT_NEAR(cell_state[cell_state_base[1]], -0.07f, 1e-6f);
    EXPECT_NEAR(cell_state[cell_state_base[2] + 0], -0.06f, 1e-6f);
    EXPECT_NEAR(cell_state[cell_state_base[2] + 1], 0.0f, 1e-6f);

    engine.shutdown();
}

TEST(SpikeEngine, neuroml_construction_wires_edges_and_inputs) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    FixtureDirectory fixture("neuroml_wiring");
    String model_path = write_two_cell_type_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    // The one <connectionWD weight="2.5"/> reaches the weight matrix exactly.
    EXPECT_NEAR(engine.weights.get(0, 3), 2.5f, 1e-3f);

    // Its delay="0.3ms" at 0.1ms per tick is pushed through set_edge_delay_ticks, which
    // throws on an edge the k^2-tree does not hold -- so reaching this line at all is what
    // proves the call landed. It is not read back: WeightMatrix exposes no delay getter,
    // and asserting on edge_delay_ticks directly would pin this test to a storage layout
    // that is currently being changed.

    // <explicitInput target="pop1[0]" input="pg0"/> with a pulseGenerator: one continuous
    // window over [delay, delay + duration] = [0.5ms, 1.5ms] = ticks 5..15.
    ASSERT_EQ(engine.input_event_streams.size(), 1u);
    const NeuronInputStream &input_stream = engine.input_event_streams[0];
    EXPECT_EQ(input_stream.neuron_index, 0);
    ASSERT_EQ(input_stream.values.size(), 16u);
    EXPECT_DOUBLE_EQ(input_stream.values[4], 0.0);
    for (usize tick = 5; tick <= 15; ++tick) EXPECT_GT(input_stream.values[tick], 0.0);

    engine.shutdown();
}

// ── step_simulation ────────────────────────────────────────────────────────────

TEST(SpikeEngine, step_simulation_advances_state_and_delivers_input) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    FixtureDirectory fixture("neuroml_stepping");
    String model_path = write_two_cell_type_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    const f32 *cell_state = engine.cell_state.get_contents();
    const s32 *cell_state_base = engine.cell_state_base.get_contents();
    const f32 *network_inputs = engine.network_inputs.get_contents();

    const f32 initial_potential = cell_state[cell_state_base[0]];
    const f32 initial_dual_potential = cell_state[cell_state_base[2]];

    for (s64 tick = 0; tick < 10; ++tick) {
        engine.step_simulation(tick);

        for (s64 element = 0; element < engine.cell_state_element_count; ++element) {
            ASSERT_TRUE(std::isfinite(cell_state[element]))
                    << "cell_state[" << element << "] diverged at tick " << tick;
        }
    }

    // simpleLifCell relaxes from startPotential (-70mV) towards restingPotential (-50mV),
    // so v must have risen; dualStateCell decays towards 0 from -60mV, so its v must have
    // risen too, and by a different amount because the two types differ.
    EXPECT_GT(cell_state[cell_state_base[0]], initial_potential);
    EXPECT_GT(cell_state[cell_state_base[2]], initial_dual_potential);

    // The pulseGenerator's window opens at tick 5, so by tick 9 neuron 0 has accumulated
    // stimulus and no other neuron has.
    EXPECT_GT(network_inputs[0], 0.0f);
    EXPECT_FLOAT_EQ(network_inputs[1], 0.0f);
    EXPECT_FLOAT_EQ(network_inputs[2], 0.0f);
    EXPECT_FLOAT_EQ(network_inputs[3], 0.0f);

    engine.shutdown();
}

TEST(SpikeEngine, step_simulation_detects_and_resets_a_spike) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    FixtureDirectory fixture("neuroml_spiking");
    String model_path = write_two_cell_type_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    const f32 *cell_state = engine.cell_state.get_contents();
    const s32 *cell_state_base = engine.cell_state_base.get_contents();
    const s32 *spike_flags = engine.spike_flags.get_contents();
    const s64 *last_spiked = engine.last_spiked.get_contents();

    // v climbs from -70mV towards -50mV with tau = 1ms, crossing the -55mV threshold
    // around t = 1.4ms, i.e. inside the 50-tick run.
    s64 first_spike_tick = -1;
    for (s64 tick = 0; tick < engine.lifetime; ++tick) {
        engine.step_simulation(tick);
        if (first_spike_tick < 0 && spike_flags[0] != 0) first_spike_tick = tick;
    }

    ASSERT_GE(first_spike_tick, 0) << "the leaky integrator never crossed its threshold";
    EXPECT_LT(first_spike_tick, engine.lifetime);
    EXPECT_EQ(last_spiked[0], last_spiked[1]) << "both neurons share one prototype";

    // Reset drove v back to resetPotential (-70mV) at the tick it fired, and it has been
    // climbing again since, so it is below threshold now.
    EXPECT_LT(cell_state[cell_state_base[0]], -0.055f);

    // dualStateCell emits nothing at all.
    EXPECT_EQ(spike_flags[2], 0);
    EXPECT_EQ(spike_flags[3], 0);

    engine.shutdown();
}

TEST(SpikeEngine, spike_flags_are_cleared_each_tick) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    FixtureDirectory fixture("neuroml_spike_flags");
    String model_path = write_two_cell_type_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    const s32 *spike_flags = engine.spike_flags.get_contents();

    s64 spiking_tick = -1;
    for (s64 tick = 0; tick < engine.lifetime && spiking_tick < 0; ++tick) {
        engine.step_simulation(tick);
        if (spike_flags[0] != 0) spiking_tick = tick;
    }
    ASSERT_GE(spiking_tick, 0);

    // Immediately after firing the neuron is back at its reset potential, so the very next
    // tick cannot fire again — the flag must have been lowered rather than latched.
    engine.step_simulation(spiking_tick + 1);
    EXPECT_EQ(spike_flags[0], 0);

    engine.shutdown();
}

TEST(SpikeEngine, step_simulation_after_shutdown_throws) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    FixtureDirectory fixture("neuroml_shutdown_guard");
    String model_path = write_two_cell_type_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
    engine.shutdown();

    EXPECT_THROW(engine.step_simulation(0), std::runtime_error);
}

// ── kernel argument binding ────────────────────────────────────────────────────

TEST(SpikeEngine, generated_kernel_arguments_are_bound_by_name) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    FixtureDirectory fixture("neuroml_kernel_arguments");
    String model_path = write_two_cell_type_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    // The engine binds whatever the generated kernel names, in the order it names it, so
    // the two cannot drift apart silently. An argument the engine does not own is an
    // error naming that argument, not a silent skip or an arbitrary binding.
    ASSERT_FALSE(engine.tick_kernel_argument_names.empty());
    EXPECT_EQ(engine.tick_kernel_argument_names.front(), "cell_state");

    spikecorec::Vector<String> unknown_argument_names = engine.tick_kernel_argument_names;
    unknown_argument_names.push_back("no_such_kernel_argument");
    EXPECT_THROW(
        engine.dispatch_master_kernel(*engine.tick_kernel, unknown_argument_names, /*tick=*/0),
        std::runtime_error);

    engine.shutdown();
}

// ── hebbian buffers ────────────────────────────────────────────────────────────

TEST(SpikeEngine, last_tick_updated_is_allocated_only_for_hebbian_learning) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    FixtureDirectory plain_fixture("neuroml_no_hebbian");
    String plain_model_path = write_two_cell_type_model(plain_fixture);
    {
        SpikeEngine engine(plain_model_path, /*enable_hebbian_learning=*/false);
        EXPECT_FALSE(engine.hebbian_learning_enabled);
        EXPECT_EQ(engine.last_tick_updated.pointer, nullptr);
        // last_spiked is a kernel output, not a learning buffer, so it exists either way.
        EXPECT_NE(engine.last_spiked.pointer, nullptr);
        engine.shutdown();
    }

    FixtureDirectory hebbian_fixture("neuroml_hebbian");
    String hebbian_model_path = write_two_cell_type_model(hebbian_fixture);
    {
        SpikeEngine engine(hebbian_model_path, /*enable_hebbian_learning=*/true);
        EXPECT_TRUE(engine.hebbian_learning_enabled);
        EXPECT_NE(engine.last_tick_updated.pointer, nullptr);
        EXPECT_NE(engine.last_spiked.pointer, nullptr);
        engine.shutdown();
    }
}

// ── recording ──────────────────────────────────────────────────────────────────

TEST(SpikeEngine, recording_writes_one_frame_per_tick_for_each_selection) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    FixtureDirectory fixture("neuroml_recording");
    String model_path = write_two_cell_type_model(fixture);
    const String recording_path = fixture.path_of("out.spire");

    const s64 recorded_tick_count = 8;
    {
        SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

        // One <OutputFile> with one <OutputColumn quantity="pop1[0]/v"/>.
        ASSERT_EQ(engine.recording_profiles.size(), 1u);
        ASSERT_EQ(engine.recording_streams.size(), 1u);
        EXPECT_EQ(engine.recording_streams[0].frame_values.size(), 1u);
        EXPECT_FALSE(engine.recording_streams[0].gathers_spike_flags);

        for (s64 tick = 0; tick < recorded_tick_count; ++tick) engine.step_simulation(tick);
        engine.shutdown();
    }

    ASSERT_TRUE(filesystem::exists(recording_path));
    SpireRecording recording = read_spire_recording(recording_path);
    EXPECT_EQ(recording.neuron_count, 1);
    EXPECT_EQ(recording.frame_count, recorded_tick_count);

    // The recorded quantity is pop1[0]/v, which is climbing towards restingPotential.
    ASSERT_EQ(recording.frames.size(), (usize)recorded_tick_count);
    EXPECT_GT(recording.frames.back(), recording.frames.front());
}
