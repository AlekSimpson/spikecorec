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
#include <type_traits>
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
// The ComponentTypes are declared inline because the vendored standard library has no GLIF
// type. Neither of these reduces over its synapses, which is deliberate: this fixture is
// about the type-sectioned layout, and the synaptic path has fixtures of its own below
// (synaptic_input_lems_xml).
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
//
// ── Why these projections name no synapse ──────────────────────────────────────
// Every fixture in this section is about the ADJACENCY and the delay ring: which weight
// reaches which neuron on which tick. A projection naming no synapse delivers exactly its
// edge weight into the target's input, which is what these tests assert, so that is the
// shape they are written in. Putting a synapse component on them would insert its own time
// course between the weight and the input and make the assertions read as an approximation
// of something else. The synapse dynamics have fixtures of their own further down
// (alpha_synapse_lems_xml), where the time course is the point.
String two_cell_type_network_nml() {
    return R"(<neuroml id="enginenet">
    <simpleLifCell id="fastCell" tau="1ms" restingPotential="-50mV" threshold="-55mV"
                   resetPotential="-70mV" startPotential="-70mV"/>
    <dualStateCell id="dualCell" tau="2ms" startPotential="-60mV"/>
    <pulseGenerator id="pg0" delay="0.5ms" duration="1ms" amplitude="0.5nA"/>

    <network id="net1">
        <population id="pop1" component="fastCell" size="2"/>
        <population id="pop2" component="dualCell" size="2"/>

        <projection id="proj1" presynapticPopulation="pop1"
                    postsynapticPopulation="pop2">
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

// ── synaptic input fixtures ────────────────────────────────────────────────────
//
// Two cell types built to make the synaptic path observable tick by tick:
//
//  - oneShotCell fires exactly once, at the first tick past its fireTime, so the tick a
//    spike is propagated on is a fact of the model rather than something the test has to
//    infer from an integrator crossing a threshold.
//  - latchCell reduces over its synapses and, on any tick that reduction is non-zero,
//    records the delivered value and counts the delivery. That makes both "how much arrived"
//    and "on how many ticks did anything arrive" plain reads of cell_state.
String synaptic_input_lems_xml(const String &network_file, const String &recording_file) {
    return R"(<Lems>
    <Target component="sim1"/>

    <ComponentType name="oneShotCell" extends="baseSpikingCell"
                   description="Emits exactly one spike, on the first tick past fireTime.">
        <Parameter name="fireTime" dimension="time"/>

        <Dynamics>
            <StateVariable name="hasFired" dimension="none"/>

            <OnStart>
                <StateAssignment variable="hasFired" value="0"/>
            </OnStart>

            <OnCondition test="t .geq. fireTime .and. hasFired .lt. 0.5">
                <StateAssignment variable="hasFired" value="1"/>
                <EventOut port="spike"/>
            </OnCondition>
        </Dynamics>
    </ComponentType>

    <ComponentType name="latchCell" extends="baseCell"
                   description="Drains its synaptic input, keeping the last delivered value
                                and the number of ticks anything was delivered on.">
        <Dynamics>
            <StateVariable name="delivered" dimension="none"/>
            <StateVariable name="deliveryCount" dimension="none"/>

            <DerivedVariable name="iSyn" dimension="none" select="synapses[*]/i" reduce="add"/>

            <OnStart>
                <StateAssignment variable="delivered" value="0"/>
                <StateAssignment variable="deliveryCount" value="0"/>
            </OnStart>

            <OnCondition test="iSyn .neq. 0">
                <StateAssignment variable="delivered" value="iSyn"/>
                <StateAssignment variable="deliveryCount" value="deliveryCount + 1"/>
            </OnCondition>
        </Dynamics>
    </ComponentType>

    <Include file=")" + network_file + R"("/>

    <Simulation id="sim1" length="2ms" step="0.1ms" target="net1">
        <OutputFile id="of1" fileName=")" + recording_file + R"(">
            <OutputColumn id="c0" quantity="popTarget[0]/delivered"/>
        </OutputFile>
    </Simulation>
</Lems>
)";
}

// One source, one target, and `connection_count` parallel connections between them, each
// carrying `connection_weight` and `connection_delay`.
String two_neuron_network_nml(const String &connection_weight, const String &connection_delay,
                              usize connection_count) {
    String projections;
    for (usize connection_index = 0; connection_index < connection_count; ++connection_index) {
        projections += R"(
        <projection id="proj)" + to_string(connection_index) + R"(" presynapticPopulation="popSource"
                    postsynapticPopulation="popTarget">
            <connectionWD id="0" preCellId="../popSource[0]" postCellId="../popTarget[0]"
                          weight=")" + connection_weight + R"(" delay=")" + connection_delay + R"("/>
        </projection>)";
    }

    return R"(<neuroml id="propagationnet">
    <oneShotCell id="source0" fireTime="0.25ms"/>
    <latchCell id="target0"/>

    <network id="net1">
        <population id="popSource" component="source0" size="1"/>
        <population id="popTarget" component="target0" size="1"/>
)" + projections + R"(
    </network>
</neuroml>
)";
}

String write_two_neuron_model(const FixtureDirectory &fixture, const String &connection_weight,
                              const String &connection_delay, usize connection_count = 1) {
    fixture.write("net.nml",
                  two_neuron_network_nml(connection_weight, connection_delay, connection_count));
    return fixture.write("model.xml",
                         synaptic_input_lems_xml("net.nml", fixture.path_of("out.spire")));
}

// A single latchCell driven by a continuous current injector and nothing else. No edges at
// all, so it also covers an edge-free network reaching the propagation apparatus.
String stimulus_only_network_nml() {
    return R"(<neuroml id="stimulusnet">
    <latchCell id="target0"/>
    <pulseGenerator id="pg0" delay="0.2ms" duration="0.5ms" amplitude="0.5nA"/>

    <network id="net1">
        <population id="popTarget" component="target0" size="1"/>

        <explicitInput target="popTarget[0]" input="pg0"/>
    </network>
</neuroml>
)";
}

String write_stimulus_only_model(const FixtureDirectory &fixture) {
    fixture.write("net.nml", stimulus_only_network_nml());
    return fixture.write("model.xml",
                         synaptic_input_lems_xml("net.nml", fixture.path_of("out.spire")));
}

// The same two neurons and the same edge, with a current injector aimed at one of them.
// `stimulus_target` names the population the pulseGenerator drives: "popSource" makes the
// stimulus land on the cell type that never reduces over its synapses, "popTarget" on the
// one that does. The amplitude is deliberately of the same order as the edge weight, so a
// stale re-delivery shows up as a value that is plainly wrong rather than as rounding.
String stimulated_two_neuron_network_nml(const String &stimulus_target,
                                         const String &pulse_delay,
                                         const String &pulse_duration) {
    return R"(<neuroml id="stimulatedpropagationnet">
    <oneShotCell id="source0" fireTime="0.25ms"/>
    <latchCell id="target0"/>
    <pulseGenerator id="pg0" delay=")" + pulse_delay + R"(" duration=")" + pulse_duration +
           R"(" amplitude="1A"/>

    <network id="net1">
        <population id="popSource" component="source0" size="1"/>
        <population id="popTarget" component="target0" size="1"/>

        <projection id="proj0" presynapticPopulation="popSource"
                    postsynapticPopulation="popTarget">
            <connectionWD id="0" preCellId="../popSource[0]" postCellId="../popTarget[0]"
                          weight="2.5" delay="0.3ms"/>
        </projection>

        <explicitInput target=")" + stimulus_target + R"([0]" input="pg0"/>
    </network>
</neuroml>
)";
}

String write_stimulated_two_neuron_model(const FixtureDirectory &fixture,
                                         const String &stimulus_target, const String &pulse_delay,
                                         const String &pulse_duration) {
    fixture.write("net.nml",
                  stimulated_two_neuron_network_nml(stimulus_target, pulse_delay, pulse_duration));
    return fixture.write("model.xml",
                         synaptic_input_lems_xml("net.nml", fixture.path_of("out.spire")));
}

// ── realistic-magnitude weight fixtures ────────────────────────────────────────
//
// The same oneShotCell/latchCell pair as above, plus a relay cell that FIRES off its
// synaptic input rather than merely recording it, so a weight has to survive storage,
// GPU reconstruction and delivery accurately enough to cross a threshold. Every weight
// here is at the SI magnitude NeuroML models actually specify (1e-9), which is where
// storing a weight as a delta against the order-1 random U/V reconstruction rounds it away
// to nothing.
String realistic_weight_lems_xml(const String &network_file, const String &recording_file) {
    return R"(<Lems>
    <Target component="sim1"/>

    <ComponentType name="oneShotCell" extends="baseSpikingCell"
                   description="Emits exactly one spike, on the first tick past fireTime.">
        <Parameter name="fireTime" dimension="time"/>

        <Dynamics>
            <StateVariable name="hasFired" dimension="none"/>

            <OnStart>
                <StateAssignment variable="hasFired" value="0"/>
            </OnStart>

            <OnCondition test="t .geq. fireTime .and. hasFired .lt. 0.5">
                <StateAssignment variable="hasFired" value="1"/>
                <EventOut port="spike"/>
            </OnCondition>
        </Dynamics>
    </ComponentType>

    <ComponentType name="relayCell" extends="baseSpikingCell"
                   description="Fires on any tick its synaptic input exceeds spikeThreshold,
                                keeping the value that crossed it.">
        <Parameter name="spikeThreshold" dimension="none"/>

        <Dynamics>
            <StateVariable name="lastInput" dimension="none"/>

            <DerivedVariable name="iSyn" dimension="none" select="synapses[*]/i" reduce="add"/>

            <OnStart>
                <StateAssignment variable="lastInput" value="0"/>
            </OnStart>

            <OnCondition test="iSyn .gt. spikeThreshold">
                <StateAssignment variable="lastInput" value="iSyn"/>
                <EventOut port="spike"/>
            </OnCondition>
        </Dynamics>
    </ComponentType>

    <ComponentType name="latchCell" extends="baseCell"
                   description="Drains its synaptic input, keeping the last delivered value
                                and the number of ticks anything was delivered on.">
        <Dynamics>
            <StateVariable name="delivered" dimension="none"/>
            <StateVariable name="deliveryCount" dimension="none"/>

            <DerivedVariable name="iSyn" dimension="none" select="synapses[*]/i" reduce="add"/>

            <OnStart>
                <StateAssignment variable="delivered" value="0"/>
                <StateAssignment variable="deliveryCount" value="0"/>
            </OnStart>

            <OnCondition test="iSyn .neq. 0">
                <StateAssignment variable="delivered" value="iSyn"/>
                <StateAssignment variable="deliveryCount" value="deliveryCount + 1"/>
            </OnCondition>
        </Dynamics>
    </ComponentType>

    <Include file=")" + network_file + R"("/>

    <Simulation id="sim1" length="2ms" step="0.1ms" target="net1">
        <OutputFile id="of1" fileName=")" + recording_file + R"(">
            <OutputColumn id="c0" quantity="popRelay[0]/lastInput"/>
        </OutputFile>
    </Simulation>
</Lems>
)";
}

// source -> two relays -> sink. Neuron indices follow population declaration order:
// 0 popSource[0], 1 popRelay[0], 2 popRelay[1], 3 popSink[0].
String realistic_weight_network_nml() {
    return R"(<neuroml id="realisticweightnet">
    <oneShotCell id="source0" fireTime="0.25ms"/>
    <relayCell id="relay0" spikeThreshold="1e-9"/>
    <latchCell id="sink0"/>

    <network id="net1">
        <population id="popSource" component="source0" size="1"/>
        <population id="popRelay" component="relay0" size="2"/>
        <population id="popSink" component="sink0" size="1"/>

        <projection id="proj0" presynapticPopulation="popSource"
                    postsynapticPopulation="popRelay">
            <connectionWD id="0" preCellId="../popSource[0]" postCellId="../popRelay[0]"
                          weight="2.5e-9" delay="0.1ms"/>
            <connectionWD id="1" preCellId="../popSource[0]" postCellId="../popRelay[1]"
                          weight="7.5e-9" delay="0.1ms"/>
        </projection>

        <projection id="proj1" presynapticPopulation="popRelay"
                    postsynapticPopulation="popSink">
            <connectionWD id="0" preCellId="../popRelay[0]" postCellId="../popSink[0]"
                          weight="4e-9" delay="0.1ms"/>
        </projection>
    </network>
</neuroml>
)";
}

String write_realistic_weight_model(const FixtureDirectory &fixture) {
    fixture.write("net.nml", realistic_weight_network_nml());
    return fixture.write("model.xml",
                         realistic_weight_lems_xml("net.nml", fixture.path_of("out.spire")));
}

// ── current-based synapse fixtures ─────────────────────────────────────────────
//
// alphaCurrentSynapse is REAL vendored NeuroML (third_party/neuroml2/std_lib/Synapses.xml),
// not an inline declaration: two coupled state variables integrated every tick, an
// OnEvent on port "in" that bumps one of them by `weight * ibase`, and a DerivedVariable
// `i` exposing the other as the delivered current. That is the whole shape a synapse model
// has, and none of it is exercised by a raw weight added at the arrival tick.
//
// traceCell copies whatever its synapses delivered into a state variable on EVERY tick --
// its OnCondition is `t .geq. 0`, which never fails -- so the delivered current reads back
// as a plain per-tick series rather than only on the ticks something happened to arrive.
// That is what makes a time course distinguishable from an impulse.
String alpha_synapse_lems_xml(const String &network_file, const String &recording_file) {
    return R"(<Lems>
    <Target component="sim1"/>

    <ComponentType name="oneShotCell" extends="baseSpikingCell"
                   description="Emits exactly one spike, on the first tick past fireTime.">
        <Parameter name="fireTime" dimension="time"/>

        <Dynamics>
            <StateVariable name="hasFired" dimension="none"/>

            <OnStart>
                <StateAssignment variable="hasFired" value="0"/>
            </OnStart>

            <OnCondition test="t .geq. fireTime .and. hasFired .lt. 0.5">
                <StateAssignment variable="hasFired" value="1"/>
                <EventOut port="spike"/>
            </OnCondition>
        </Dynamics>
    </ComponentType>

    <ComponentType name="traceCell" extends="baseCell"
                   description="Records what its synapses delivered, on every tick.">
        <Dynamics>
            <StateVariable name="lastInput" dimension="none"/>

            <DerivedVariable name="iSyn" dimension="none" select="synapses[*]/i" reduce="add"/>

            <OnStart>
                <StateAssignment variable="lastInput" value="0"/>
            </OnStart>

            <OnCondition test="t .geq. 0">
                <StateAssignment variable="lastInput" value="iSyn"/>
            </OnCondition>
        </Dynamics>
    </ComponentType>

    <Include file=")" + network_file + R"("/>

    <Simulation id="sim1" length="4ms" step="0.1ms" target="net1">
        <OutputFile id="of1" fileName=")" + recording_file + R"(">
            <OutputColumn id="c0" quantity="popTarget[0]/lastInput"/>
        </OutputFile>
    </Simulation>
</Lems>
)";
}

// One source through one alphaCurrentSynapse onto one target.
String alpha_synapse_network_nml() {
    return R"(<neuroml id="alphasynapsenet">
    <oneShotCell id="source0" fireTime="0.25ms"/>
    <traceCell id="target0"/>
    <alphaCurrentSynapse id="alphaSyn" tau="0.5ms" ibase="1nA"/>

    <network id="net1">
        <population id="popSource" component="source0" size="1"/>
        <population id="popTarget" component="target0" size="1"/>

        <projection id="proj0" presynapticPopulation="popSource"
                    postsynapticPopulation="popTarget" synapse="alphaSyn">
            <connectionWD id="0" preCellId="../popSource[0]" postCellId="../popTarget[0]"
                          weight="1" delay="0.1ms"/>
        </projection>
    </network>
</neuroml>
)";
}

String write_alpha_synapse_model(const FixtureDirectory &fixture) {
    fixture.write("net.nml", alpha_synapse_network_nml());
    return fixture.write("model.xml",
                         alpha_synapse_lems_xml("net.nml", fixture.path_of("out.spire")));
}

// Two sources onto ONE target through two alphaCurrentSynapse prototypes whose only
// difference is `tau`. Their responses decay at different rates, so a single pooled
// accumulator could not represent both: whatever rate it decayed at would be wrong for one
// of them.
String two_tau_network_nml() {
    return R"(<neuroml id="twotaunet">
    <oneShotCell id="source0" fireTime="0.25ms"/>
    <traceCell id="target0"/>
    <alphaCurrentSynapse id="alphaFast" tau="0.5ms" ibase="1nA"/>
    <alphaCurrentSynapse id="alphaSlow" tau="2ms" ibase="1nA"/>

    <network id="net1">
        <population id="popSource" component="source0" size="2"/>
        <population id="popTarget" component="target0" size="1"/>

        <projection id="projFast" presynapticPopulation="popSource"
                    postsynapticPopulation="popTarget" synapse="alphaFast">
            <connectionWD id="0" preCellId="../popSource[0]" postCellId="../popTarget[0]"
                          weight="1" delay="0.1ms"/>
        </projection>
        <projection id="projSlow" presynapticPopulation="popSource"
                    postsynapticPopulation="popTarget" synapse="alphaSlow">
            <connectionWD id="0" preCellId="../popSource[1]" postCellId="../popTarget[0]"
                          weight="1" delay="0.1ms"/>
        </projection>
    </network>
</neuroml>
)";
}

String write_two_tau_model(const FixtureDirectory &fixture) {
    fixture.write("net.nml", two_tau_network_nml());
    return fixture.write("model.xml",
                         alpha_synapse_lems_xml("net.nml", fixture.path_of("out.spire")));
}

// The same one-source-one-target wiring through expOneSynapse, which is conductance-based.
String conductance_synapse_network_nml() {
    return R"(<neuroml id="conductancesynapsenet">
    <oneShotCell id="source0" fireTime="0.25ms"/>
    <traceCell id="target0"/>
    <expOneSynapse id="condSyn" gbase="0.5nS" erev="0mV" tauDecay="3ms"/>

    <network id="net1">
        <population id="popSource" component="source0" size="1"/>
        <population id="popTarget" component="target0" size="1"/>

        <projection id="proj0" presynapticPopulation="popSource"
                    postsynapticPopulation="popTarget" synapse="condSyn">
            <connectionWD id="0" preCellId="../popSource[0]" postCellId="../popTarget[0]"
                          weight="1" delay="0.1ms"/>
        </projection>
    </network>
</neuroml>
)";
}

String write_conductance_synapse_model(const FixtureDirectory &fixture) {
    fixture.write("net.nml", conductance_synapse_network_nml());
    return fixture.write("model.xml",
                         alpha_synapse_lems_xml("net.nml", fixture.path_of("out.spire")));
}

// alphaCurrentSynapse's own dynamics, integrated on the host exactly as the generated
// kernel does: the arrival applied first (stage 1 is Deliver), then one forward-Euler step
// of both state variables from the state as it stood at entry, then `i`, which is `I`, read
// off the state that step just wrote.
//
// Written out rather than compared against the closed-form alpha function because the
// kernel integrates with forward Euler at the model's own dt, and at dt = tau / 5 the two
// differ by percent. The point of the comparison is that the SHAPE and the RATE are the
// synapse's, so the expectation has to be the same discretisation.
struct AlphaSynapseReference {
    f32 tau = 0.0f;
    f32 ibase = 0.0f;
    f32 current_I = 0.0f;
    f32 current_J = 0.0f;

    AlphaSynapseReference(f32 tau, f32 ibase) : tau(tau), ibase(ibase) {}

    // Advances one tick and returns the current delivered on it.
    f32 step(f32 dt, f32 arrival_weight) {
        if (arrival_weight != 0.0f) current_J = current_J + arrival_weight * ibase;

        const f32 next_I = current_I + dt * ((2.7182818284590451f * current_J - current_I) / tau);
        const f32 next_J = current_J + dt * (-current_J / tau);
        current_I = next_I;
        current_J = next_J;

        return current_I;
    }
};

// Where a traceCell's one state variable sits in the flat cell_state array.
f32 trace_cell_last_input(const SpikeEngine &engine, s64 neuron_index) {
    const s64 state_base = engine.cell_state_base.get_contents()[neuron_index];
    return engine.cell_state.get_contents()[state_base];
}

// Where a latchCell's two state variables sit in the flat cell_state array.
struct LatchCellReader {
    const f32 *cell_state = nullptr;
    s64 delivered_index = 0;
    s64 delivery_count_index = 0;

    LatchCellReader(const SpikeEngine &engine, s64 neuron_index) {
        cell_state = engine.cell_state.get_contents();
        const s64 state_base = engine.cell_state_base.get_contents()[neuron_index];
        delivered_index = state_base + 0;
        delivery_count_index = state_base + 1;
    }

    f32 delivered() const { return cell_state[delivered_index]; }
    s32 delivery_count() const { return (s32)cell_state[delivery_count_index]; }
};

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

    // The pulseGenerator has been firing into neuron 0's slot since tick 5, and neither cell
    // type in this model reduces over its synapses. Every one of those slots is nonetheless
    // empty: the ring row is cleared as a whole row at the end of the tick that read it, so
    // input a type never reads is consumed rather than piling up for the rest of the run.
    const s64 ring_element_count =
            (s64)engine.network_input_ring_depth * engine.network_input_plane_count *
            engine.total_neuron_count;
    for (s64 element = 0; element < ring_element_count; ++element) {
        EXPECT_FLOAT_EQ(network_inputs[element], 0.0f)
                << "network_inputs[" << element << "] accumulated input nothing consumed";
    }

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

// ── stage 1, Deliver: draining the ring ────────────────────────────────────────

TEST(SpikeEngine, injected_current_is_delivered_at_constant_amplitude_not_as_a_ramp) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    FixtureDirectory fixture("neuroml_constant_injection");
    String model_path = write_stimulus_only_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
    const LatchCellReader target(engine, /*neuron_index=*/0);

    // The injector's window is [0.2ms, 0.7ms] at 0.1ms per tick, so ticks 2 through 7 each
    // deliver one tick's worth of the same current.
    spikecorec::Vector<f32> delivered_per_tick;
    for (s64 tick = 0; tick < 12; ++tick) {
        engine.step_simulation(tick);
        delivered_per_tick.push_back(target.delivered());
    }

    ASSERT_GT(delivered_per_tick[2], 0.0f) << "the injector's window never opened";

    // Every tick of the window delivers the SAME amount. A slot that is read without being
    // cleared accumulates instead, turning a constant current into a linear ramp -- which
    // still rises, still looks plausible, and is wrong at every steady state.
    for (usize tick = 3; tick <= 7; ++tick) {
        EXPECT_FLOAT_EQ(delivered_per_tick[tick], delivered_per_tick[2])
                << "delivered current changed at tick " << tick
                << "; the ring row is not being cleared after it is read";
    }

    // Delivered on exactly the six ticks of the window and on no others: past the window the
    // row is empty, which is only true if each read cleared it.
    EXPECT_EQ(target.delivery_count(), 6);

    engine.shutdown();
}

// ── stage 6, Propagate ─────────────────────────────────────────────────────────

// Runs the two-neuron model to completion and reports the tick the source fired on and the
// tick its spike was delivered to the target.
struct PropagationObservation {
    s64 source_spike_tick = -1;
    s64 source_spike_count = 0;
    s64 delivery_tick = -1;
    s32 final_delivery_count = 0;
    f32 delivered_value = 0.0f;
};

static PropagationObservation observe_propagation(SpikeEngine &engine, s64 tick_count) {
    const LatchCellReader target(engine, /*neuron_index=*/1);
    const s32 *spike_flags = engine.spike_flags.get_contents();

    PropagationObservation observation;
    for (s64 tick = 0; tick < tick_count; ++tick) {
        const s32 previous_delivery_count = target.delivery_count();
        engine.step_simulation(tick);

        if (spike_flags[0] != 0) {
            if (observation.source_spike_tick < 0) observation.source_spike_tick = tick;
            observation.source_spike_count += 1;
        }
        if (target.delivery_count() != previous_delivery_count &&
            observation.delivery_tick < 0) {
            observation.delivery_tick = tick;
            observation.delivered_value = target.delivered();
        }
    }

    observation.final_delivery_count = target.delivery_count();
    return observation;
}

TEST(SpikeEngine, a_spike_reaches_its_target_one_tick_later_carrying_the_edge_weight) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    FixtureDirectory fixture("neuroml_propagation_one_tick");
    String model_path = write_two_neuron_model(fixture, /*connection_weight=*/"2.5",
                                               /*connection_delay=*/"0.1ms");

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    // One tick of delay needs two ring rows: the row an arrival is scheduled into must not
    // be the row being drained on the tick it is scheduled.
    EXPECT_EQ(engine.network_input_ring_depth, 2);

    const PropagationObservation observation = observe_propagation(engine, /*tick_count=*/20);

    ASSERT_EQ(observation.source_spike_count, 1) << "oneShotCell must fire exactly once";
    ASSERT_GE(observation.delivery_tick, 0) << "the spike was never delivered";
    EXPECT_EQ(observation.delivery_tick, observation.source_spike_tick + 1);

    // What arrived is the edge's own weight, reconstructed on the GPU out of the shared U/V
    // basis and its sparse delta -- the same number WeightMatrix reports on the host.
    EXPECT_NEAR(observation.delivered_value, 2.5f, 1e-3f);
    EXPECT_NEAR(engine.weights.get(0, 1), 2.5f, 1e-3f);

    engine.shutdown();
}

TEST(SpikeEngine, a_delayed_spike_arrives_on_its_own_tick_and_not_before) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    FixtureDirectory fixture("neuroml_propagation_three_ticks");
    String model_path = write_two_neuron_model(fixture, /*connection_weight=*/"2.5",
                                               /*connection_delay=*/"0.3ms");

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
    EXPECT_EQ(engine.network_input_ring_depth, 4) << "one row per delay, plus the row "
                                                     "being drained";

    const LatchCellReader target(engine, /*neuron_index=*/1);
    const s32 *spike_flags = engine.spike_flags.get_contents();

    s64 source_spike_tick = -1;
    s64 delivery_tick = -1;
    for (s64 tick = 0; tick < 20; ++tick) {
        engine.step_simulation(tick);

        if (spike_flags[0] != 0 && source_spike_tick < 0) source_spike_tick = tick;

        if (target.delivery_count() == 0 && source_spike_tick >= 0) {
            // Nothing has arrived yet, so this tick must be inside the delay window.
            EXPECT_LT(tick, source_spike_tick + 3)
                    << "the spike was still undelivered past its arrival tick";
        }
        if (target.delivery_count() != 0 && delivery_tick < 0) delivery_tick = tick;
    }

    ASSERT_GE(source_spike_tick, 0) << "oneShotCell never fired";
    ASSERT_GE(delivery_tick, 0) << "the spike was never delivered";

    // The whole point: three ticks of delay means three ticks of delay. Arriving early is a
    // worse failure than not arriving, because it still produces a plausible-looking run.
    EXPECT_EQ(delivery_tick, source_spike_tick + 3);
    EXPECT_NEAR(target.delivered(), 2.5f, 1e-3f);

    engine.shutdown();
}

TEST(SpikeEngine, a_delivered_spike_is_not_redelivered_when_the_ring_wraps) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    FixtureDirectory fixture("neuroml_ring_wrap");
    String model_path = write_two_neuron_model(fixture, /*connection_weight=*/"2.5",
                                               /*connection_delay=*/"0.3ms");

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
    ASSERT_EQ(engine.network_input_ring_depth, 4);

    // Twenty ticks is five full turns of a four-row ring, so a row left uncleared by its
    // reader would hand the same 2.5 back four more times.
    const PropagationObservation observation = observe_propagation(engine, /*tick_count=*/20);

    ASSERT_EQ(observation.source_spike_count, 1);
    EXPECT_EQ(observation.final_delivery_count, 1)
            << "the spike was delivered more than once: the ring row was not cleared after "
               "being read";

    // Nothing is left pending anywhere in the ring either.
    const f32 *network_inputs = engine.network_inputs.get_contents();
    const s64 ring_element_count =
            (s64)engine.network_input_ring_depth * engine.network_input_plane_count *
            engine.total_neuron_count;
    for (s64 element = 0; element < ring_element_count; ++element) {
        EXPECT_FLOAT_EQ(network_inputs[element], 0.0f)
                << "network_inputs[" << element << "] still holds an undelivered arrival";
    }

    engine.shutdown();
}

TEST(SpikeEngine, input_to_a_type_that_never_reads_its_synapses_does_not_accumulate) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // The stimulus lands on popSource, whose oneShotCell has no `select=` over its synapses
    // and so never reads the ring at all. Under a design where each cell empties its own slot
    // as it reads, nothing would ever empty this one and it would carry the whole run's
    // injected current, growing every tick. The row clear is what makes emptying a slot
    // independent of whether any cell type reads it.
    FixtureDirectory fixture("neuroml_unread_slot");
    String model_path = write_stimulated_two_neuron_model(fixture, /*stimulus_target=*/"popSource",
                                                          /*pulse_delay=*/"0ms",
                                                          /*pulse_duration=*/"1ms");

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
    ASSERT_EQ(engine.network_input_ring_depth, 4);
    ASSERT_EQ(engine.total_neuron_count, 2);

    const f32 *network_inputs = engine.network_inputs.get_contents();
    const LatchCellReader target(engine, /*neuron_index=*/1);

    // Twelve ticks is three full turns of the four-row ring, with the injector firing into
    // the source on every one of them.
    for (s64 tick = 0; tick < 12; ++tick) {
        engine.step_simulation(tick);

        for (s64 row = 0; row < engine.network_input_ring_depth; ++row) {
            const s64 source_slot = row * engine.total_neuron_count + 0;
            EXPECT_FLOAT_EQ(network_inputs[source_slot], 0.0f)
                    << "after tick " << tick << " the source's slot in ring row " << row
                    << " still holds input no cell type reads";
        }
    }

    // The reading type is unaffected: its one spike still arrived, exactly once.
    EXPECT_EQ(target.delivery_count(), 1);

    engine.shutdown();
}

TEST(SpikeEngine, clearing_the_current_row_preserves_a_spike_already_in_flight) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // "Clear the past, preserve the future", stated directly. The target is stimulated on the
    // first three ticks, so the row it reads is non-empty and gets cleared on each of them,
    // while a spike with a three-tick delay is in flight into a LATER row. Clearing more than
    // the row just read would lose that spike; clearing less would re-deliver the stimulus
    // when the four-row ring wrapped back around, four ticks later.
    FixtureDirectory fixture("neuroml_in_flight_across_clears");
    String model_path = write_stimulated_two_neuron_model(fixture, /*stimulus_target=*/"popTarget",
                                                          /*pulse_delay=*/"0ms",
                                                          /*pulse_duration=*/"0.2ms");

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
    ASSERT_EQ(engine.network_input_ring_depth, 4);

    const LatchCellReader target(engine, /*neuron_index=*/1);
    const s32 *spike_flags = engine.spike_flags.get_contents();

    s64 source_spike_tick = -1;
    f32 stimulus_magnitude = 0.0f;
    spikecorec::Vector<f32> delivered_on_tick(12, 0.0f);

    for (s64 tick = 0; tick < 12; ++tick) {
        const s32 previous_delivery_count = target.delivery_count();
        engine.step_simulation(tick);

        if (spike_flags[0] != 0 && source_spike_tick < 0) source_spike_tick = tick;
        if (target.delivery_count() != previous_delivery_count) {
            delivered_on_tick[(usize)tick] = target.delivered();
        }
    }

    ASSERT_GE(source_spike_tick, 0) << "oneShotCell never fired";
    stimulus_magnitude = delivered_on_tick[0];
    ASSERT_GT(stimulus_magnitude, 0.01f)
            << "the injected stimulus is too small for a stale re-delivery to be detectable";

    // Ticks 0 through 2 are the injector's window; ticks 4 and 5 are where the ring wraps
    // back onto rows 0 and 1, which is when an uncleared row would hand its stimulus back.
    EXPECT_GT(delivered_on_tick[1], 0.0f);
    EXPECT_GT(delivered_on_tick[2], 0.0f);
    EXPECT_FLOAT_EQ(delivered_on_tick[4], 0.0f) << "row 0's stimulus was delivered a second time";
    EXPECT_FLOAT_EQ(delivered_on_tick[5], 0.0f) << "row 1's stimulus was delivered a second time";

    // The spike survived every one of those clears and arrived on its own tick, carrying the
    // edge's weight and nothing else -- no stimulus left behind in the row it landed in.
    const usize arrival_tick = (usize)(source_spike_tick + 3);
    ASSERT_LT(arrival_tick, delivered_on_tick.size());
    EXPECT_NEAR(delivered_on_tick[arrival_tick], 2.5f, stimulus_magnitude * 0.1f)
            << "the arrival carried " << delivered_on_tick[arrival_tick]
            << " rather than the edge weight alone";

    engine.shutdown();
}

// ── parallel edges between one ordered pair ────────────────────────────────────

namespace {

nml::NML_ParseResult make_parallel_edge_model(s64 first_delay_tick_count,
                                              s64 second_delay_tick_count,
                                              s64 first_synapse_prototype_index = -1,
                                              s64 second_synapse_prototype_index = -1) {
    nml::NML_ParseResult parse_result;
    parse_result.neurons.resize(3);

    nml::NetworkEdge first_edge;
    first_edge.target_neuron_index = 2;
    first_edge.weight = 1.5;
    first_edge.delay_tick_count = first_delay_tick_count;
    first_edge.synapse_prototype_index = first_synapse_prototype_index;

    nml::NetworkEdge second_edge;
    second_edge.target_neuron_index = 2;
    second_edge.weight = 4.0;
    second_edge.delay_tick_count = second_delay_tick_count;
    second_edge.synapse_prototype_index = second_synapse_prototype_index;

    parse_result.neurons[0].outgoing_edges.push_back(first_edge);
    parse_result.neurons[0].outgoing_edges.push_back(second_edge);
    return parse_result;
}

} // namespace

// ── current-based synapse dynamics ─────────────────────────────────────────────

TEST(SpikeEngine, a_current_based_synapse_delivers_its_time_course_not_an_impulse) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // The one test that separates a real synapse model from adding the raw edge weight at
    // the arrival tick. An alphaCurrentSynapse has two coupled state variables integrated on
    // every tick whether or not anything arrived, so one spike produces a current that RISES
    // over several ticks, peaks around tau, and decays -- not a single non-zero tick.
    FixtureDirectory fixture("neuroml_alpha_synapse_time_course");
    String model_path = write_alpha_synapse_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
    ASSERT_EQ(engine.total_neuron_count, 2);

    // One wired prototype, so the ring is two planes wide: the delivered current every cell
    // reads, and that prototype's own arrivals.
    EXPECT_EQ(engine.network_input_plane_count, 2);
    // Two state variables (I and J) per neuron, for the one prototype.
    EXPECT_EQ(engine.synapse_state_element_count, 2 * engine.total_neuron_count);

    const f32 step_dt = (f32)engine.network_details.step_dt;
    const s64 tick_count = 40;

    // The source fires on the first tick with t >= 0.25ms, and the edge's 0.1ms delay is one
    // tick, so the arrival is due one tick after that.
    const s64 source_spike_tick = 3;
    const s64 arrival_tick = source_spike_tick + 1;

    AlphaSynapseReference reference(/*tau=*/5.0e-4f, /*ibase=*/1.0e-9f);
    spikecorec::Vector<f32> delivered(tick_count, 0.0f);
    spikecorec::Vector<f32> expected(tick_count, 0.0f);

    for (s64 tick = 0; tick < tick_count; ++tick) {
        engine.step_simulation(tick);
        delivered[(usize)tick] = trace_cell_last_input(engine, /*neuron_index=*/1);
        expected[(usize)tick] = reference.step(step_dt, tick == arrival_tick ? 1.0f : 0.0f);
    }

    ASSERT_EQ(engine.spike_flags.get_contents()[0], 0)
            << "the source fired more than once, so the trace is not one spike's response";

    // Nothing before the arrival, and the arrival tick itself is only the start of the rise.
    for (s64 tick = 0; tick < arrival_tick; ++tick) {
        EXPECT_FLOAT_EQ(delivered[(usize)tick], 0.0f)
                << "current was delivered on tick " << tick << ", before the spike arrived";
    }

    // A time course, not an impulse: rising for several ticks after the arrival, falling well
    // afterwards, and non-zero on every tick in between. tau is 0.5ms against a 0.1ms dt, so
    // the peak is around five ticks past the arrival and the tail runs far past that.
    for (s64 tick = arrival_tick; tick < arrival_tick + 3; ++tick) {
        EXPECT_GT(delivered[(usize)tick + 1], delivered[(usize)tick])
                << "the delivered current did not rise from tick " << tick << " to the next";
    }
    for (s64 tick = arrival_tick + 15; tick < tick_count - 1; ++tick) {
        EXPECT_LT(delivered[(usize)tick + 1], delivered[(usize)tick])
                << "the delivered current did not decay from tick " << tick << " to the next";
    }
    for (s64 tick = arrival_tick; tick < tick_count; ++tick) {
        EXPECT_GT(delivered[(usize)tick], 0.0f)
                << "the delivered current was zero on tick " << tick
                << ": a single-tick impulse, not a synapse integrated every tick";
    }

    // And it is the synapse's own trajectory, tick for tick.
    for (s64 tick = 0; tick < tick_count; ++tick) {
        EXPECT_NEAR(delivered[(usize)tick], expected[(usize)tick],
                    std::fabs(expected[(usize)tick]) * 1e-4f + 1e-15f)
                << "tick " << tick << " delivered " << delivered[(usize)tick] << " where the "
                << "synapse's own dynamics give " << expected[(usize)tick];
    }

    engine.shutdown();
}

TEST(SpikeEngine, two_current_based_synapses_on_one_target_decay_at_their_own_rates) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // Two alphaCurrentSynapse prototypes differing only in tau, converging on one neuron.
    // They get a plane and a state slice each and are integrated separately, so the target
    // sees the SUM OF TWO independent responses. Pooling them into one accumulator would
    // have to decay the total at a single rate, which is neither of theirs.
    FixtureDirectory fixture("neuroml_two_tau_synapses");
    String model_path = write_two_tau_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
    ASSERT_EQ(engine.total_neuron_count, 3);

    // Two wired prototypes: three planes, and two state variables each per neuron.
    EXPECT_EQ(engine.network_input_plane_count, 3);
    EXPECT_EQ(engine.synapse_state_element_count, 4 * engine.total_neuron_count);

    const f32 step_dt = (f32)engine.network_details.step_dt;
    const s64 tick_count = 60;
    const s64 arrival_tick = 4;

    AlphaSynapseReference fast_reference(/*tau=*/5.0e-4f, /*ibase=*/1.0e-9f);
    AlphaSynapseReference slow_reference(/*tau=*/2.0e-3f, /*ibase=*/1.0e-9f);
    // What a single pooled state would give: the two arrivals summed into one synapse, which
    // is what the design must NOT do.
    AlphaSynapseReference pooled_reference(/*tau=*/5.0e-4f, /*ibase=*/1.0e-9f);

    for (s64 tick = 0; tick < tick_count; ++tick) {
        engine.step_simulation(tick);

        const f32 arrival_weight = tick == arrival_tick ? 1.0f : 0.0f;
        const f32 expected = fast_reference.step(step_dt, arrival_weight) +
                             slow_reference.step(step_dt, arrival_weight);
        const f32 pooled = pooled_reference.step(step_dt, 2.0f * arrival_weight);
        const f32 delivered = trace_cell_last_input(engine, /*neuron_index=*/2);

        EXPECT_NEAR(delivered, expected, std::fabs(expected) * 1e-4f + 1e-15f)
                << "tick " << tick << ": the two synapses did not each decay at their own tau";

        // Late enough that the fast response has all but gone and the slow one has not: if
        // the two had been pooled at one rate the difference would be plain.
        if (tick >= arrival_tick + 30) {
            EXPECT_GT(delivered, 4.0f * pooled)
                    << "tick " << tick << ": the delivered current decayed like a single "
                    << "pooled synapse rather than like two with different tau";
        }
    }

    engine.shutdown();
}

TEST(SpikeEngine, a_conductance_based_synapse_is_refused_by_name_at_construction) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // expOneSynapse computes i = g * (erev - v): a driving force that depends on the
    // postsynaptic voltage and reverses sign as v crosses erev. Running it as a current-based
    // synapse would be a different model, not an approximation, so the engine refuses to
    // build rather than producing plausible wrong numbers.
    FixtureDirectory fixture("neuroml_conductance_synapse_refused");
    String model_path = write_conductance_synapse_model(fixture);

    try {
        SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
        engine.shutdown();
        FAIL() << "expected a conductance-based synapse to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("expOneSynapse"), String::npos) << message;
        EXPECT_NE(message.find("conductance-based"), String::npos) << message;
        EXPECT_NE(message.find("not supported yet"), String::npos) << message;
    }
}

TEST(AggregateNetworkEdges, parallel_edges_between_one_pair_sum_their_weights) {
    // Two projections between one cell pair -- an AMPA and an NMDA, say -- is an ordinary
    // NeuroML shape, and the adjacency holds one slot for the pair. Both synapses deliver,
    // so the total is the only faithful collapse; keeping the last one declared would drop
    // half the drive with no diagnostic.
    const nml::NML_ParseResult parse_result = make_parallel_edge_model(3, 3);

    const spikecorec::Vector<AggregatedNetworkEdge> aggregated =
            aggregate_network_edges(parse_result, /*default_delay_tick_count=*/1, log::logger());

    ASSERT_EQ(aggregated.size(), 1u);
    EXPECT_EQ(aggregated[0].source_node, 0);
    EXPECT_EQ(aggregated[0].target_node, 2);
    EXPECT_FLOAT_EQ(aggregated[0].summed_weight, 5.5f);
    EXPECT_EQ(aggregated[0].delay_tick_count, 3);
}

TEST(AggregateNetworkEdges, an_undeclared_delay_reads_as_the_default_rather_than_a_conflict) {
    const nml::NML_ParseResult parse_result = make_parallel_edge_model(0, 1);

    const spikecorec::Vector<AggregatedNetworkEdge> aggregated =
            aggregate_network_edges(parse_result, /*default_delay_tick_count=*/1, log::logger());

    ASSERT_EQ(aggregated.size(), 1u);
    EXPECT_FLOAT_EQ(aggregated[0].summed_weight, 5.5f);
    EXPECT_EQ(aggregated[0].delay_tick_count, 1);
}

TEST(AggregateNetworkEdges, parallel_edges_with_different_delays_throw_naming_both) {
    // One slot cannot carry two conduction delays, and picking either one silently moves
    // when every downstream spike lands. There is no right answer to guess, so it is
    // reported instead.
    const nml::NML_ParseResult parse_result = make_parallel_edge_model(3, 7);

    try {
        aggregate_network_edges(parse_result, /*default_delay_tick_count=*/1, log::logger());
        FAIL() << "expected conflicting parallel delays to be rejected";
    } catch (const std::runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("3"), String::npos) << message;
        EXPECT_NE(message.find("7"), String::npos) << message;
    }
}

TEST(AggregateNetworkEdges, parallel_edges_through_one_synapse_still_sum_their_weights) {
    // Same synapse, so their arrivals land in the same plane and superpose there: summing
    // them is exactly what many edges of one prototype converging on a target means.
    const nml::NML_ParseResult parse_result =
            make_parallel_edge_model(3, 3, /*first_synapse_prototype_index=*/1,
                                     /*second_synapse_prototype_index=*/1);

    const spikecorec::Vector<AggregatedNetworkEdge> aggregated =
            aggregate_network_edges(parse_result, /*default_delay_tick_count=*/1, log::logger());

    ASSERT_EQ(aggregated.size(), 1u);
    EXPECT_FLOAT_EQ(aggregated[0].summed_weight, 5.5f);
    EXPECT_EQ(aggregated[0].synapse_prototype_index, 1);
}

TEST(AggregateNetworkEdges, parallel_edges_through_different_synapses_throw_naming_both) {
    // An AMPA and an NMDA between one cell pair need two arrival planes, and the pair has one
    // slot. Summing their weights into whichever plane was seen first would run one synapse's
    // dynamics on the other's coupling -- plausible numbers from the wrong model.
    const nml::NML_ParseResult parse_result =
            make_parallel_edge_model(3, 3, /*first_synapse_prototype_index=*/0,
                                     /*second_synapse_prototype_index=*/1);

    try {
        aggregate_network_edges(parse_result, /*default_delay_tick_count=*/1, log::logger());
        FAIL() << "expected conflicting parallel synapses to be rejected";
    } catch (const std::runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("different synapses"), String::npos) << message;
        EXPECT_NE(message.find("0"), String::npos) << message;
        EXPECT_NE(message.find("1"), String::npos) << message;
    }
}

TEST(SpikeEngine, two_projections_between_one_pair_deliver_their_summed_weight) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    FixtureDirectory fixture("neuroml_parallel_projections");
    String model_path = write_two_neuron_model(fixture, /*connection_weight=*/"2.5",
                                               /*connection_delay=*/"0.3ms",
                                               /*connection_count=*/2);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
    EXPECT_NEAR(engine.weights.get(0, 1), 5.0f, 1e-3f);

    const PropagationObservation observation = observe_propagation(engine, /*tick_count=*/20);

    ASSERT_EQ(observation.final_delivery_count, 1) << "one slot, so one arrival";
    EXPECT_NEAR(observation.delivered_value, 5.0f, 1e-3f);

    engine.shutdown();
}

// ── value semantics ────────────────────────────────────────────────────────────

// The test that would have caught the weight-annihilation bug. Storing a weight as a delta
// against the order-1 random U/V reconstruction rounds any realistic SI weight (1e-9 here)
// to nothing in f32, so every one of these edges used to deliver exactly 0.0f: the relays
// never crossed their threshold, nothing propagated past them, and the failure read as a
// mis-specified model rather than as a rounding error.
TEST(SpikeEngine, weights_at_realistic_si_magnitudes_propagate_and_make_targets_fire) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    FixtureDirectory fixture("neuroml_realistic_weight_magnitudes");
    String model_path = write_realistic_weight_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    // Stored exactly, not approximately -- these are the values the model asked for.
    EXPECT_EQ(engine.weights.get(0, 1), 2.5e-9f);
    EXPECT_EQ(engine.weights.get(0, 2), 7.5e-9f);
    EXPECT_EQ(engine.weights.get(1, 3), 4e-9f);

    const LatchCellReader sink(engine, /*neuron_index=*/3);
    const s32 *spike_flags = engine.spike_flags.get_contents();

    s64 source_spike_tick = -1;
    s64 first_relay_spike_tick = -1;
    s64 second_relay_spike_tick = -1;
    for (s64 tick = 0; tick < 20; ++tick) {
        engine.step_simulation(tick);
        if (spike_flags[0] != 0 && source_spike_tick < 0) source_spike_tick = tick;
        if (spike_flags[1] != 0 && first_relay_spike_tick < 0) first_relay_spike_tick = tick;
        if (spike_flags[2] != 0 && second_relay_spike_tick < 0) second_relay_spike_tick = tick;
    }

    ASSERT_GE(source_spike_tick, 0) << "oneShotCell never fired";
    ASSERT_GE(first_relay_spike_tick, 0)
            << "the 2.5e-9 weight never carried its target over threshold";
    ASSERT_GE(second_relay_spike_tick, 0)
            << "the 7.5e-9 weight never carried its target over threshold";
    EXPECT_EQ(first_relay_spike_tick, source_spike_tick + 1);
    EXPECT_EQ(second_relay_spike_tick, source_spike_tick + 1);

    // A relay's own spike then carries ITS edge's weight onward, so the whole path -- store,
    // reconstruct on the GPU, deliver, fire, propagate again -- ran at 1e-9 magnitudes.
    ASSERT_EQ(sink.delivery_count(), 1) << "the relay's spike never reached the sink";
    EXPECT_NEAR(sink.delivered(), 4e-9f, 1e-12f);

    engine.shutdown();
}

TEST(SpikeEngine, is_neither_copyable_nor_movable) {
    // A defaulted move left the source `alive` with a null logger, so the moved-from
    // engine's destructor dereferenced it and then ran the whole shutdown body; the
    // defaulted move assignment tripped GpuPointer's "destination must be null" assertion
    // once per buffer. Neither operation has a caller, so both are gone rather than
    // hand-written -- anything needing to relocate an engine holds a unique_ptr to one.
    EXPECT_FALSE(std::is_copy_constructible<SpikeEngine>::value);
    EXPECT_FALSE(std::is_copy_assignable<SpikeEngine>::value);
    EXPECT_FALSE(std::is_move_constructible<SpikeEngine>::value);
    EXPECT_FALSE(std::is_move_assignable<SpikeEngine>::value);
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

TEST(SpikeEngine, recording_a_model_with_no_state_variables_is_refused_by_name) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // A ComponentType with no <StateVariable> gives cell_state zero elements, and a
    // zero-element buffer is a null handle rather than an allocation. Gathering a recording
    // frame out of it dereferences that null once per tick, far from the model that caused
    // it, so the mismatch is reported when the stream is built instead.
    FixtureDirectory fixture("neuroml_stateless_recording");
    fixture.write("net.nml", R"(<neuroml id="statelessnet">
    <statelessCell id="cell0" level="1"/>

    <network id="net1">
        <population id="pop1" component="cell0" size="2"/>
    </network>
</neuroml>
)");
    String model_path = fixture.write("model.xml", R"(<Lems>
    <Target component="sim1"/>

    <ComponentType name="statelessCell" extends="baseCell"
                   description="Declares no StateVariable at all, so it has no cell state.">
        <Parameter name="level" dimension="none"/>

        <Dynamics>
            <DerivedVariable name="reading" dimension="none" value="level"/>
        </Dynamics>
    </ComponentType>

    <Include file="net.nml"/>

    <Simulation id="sim1" length="1ms" step="0.1ms" target="net1">
        <OutputFile id="of1" fileName=")" + fixture.path_of("out.spire") + R"(">
            <OutputColumn id="c0" quantity="pop1[0]/reading"/>
        </OutputFile>
    </Simulation>
</Lems>
)");

    try {
        SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
        FAIL() << "expected a model with no state variables to be refused a cell-state recorder";
    } catch (const std::runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("StateVariable"), String::npos) << message;
    }
}
