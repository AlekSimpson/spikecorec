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
#include "spikecorec/nml/kernel_codegen.h"

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

// Where a traceCell's one state variable sits in the flat cell_state array.
f32 trace_cell_last_input(const SpikeEngine &engine, s64 neuron_index) {
    const s64 state_base = engine.cell_state_base.get_contents()[neuron_index];
    return engine.cell_state.get_contents()[state_base];
}

// ── regime fixtures ────────────────────────────────────────────────────────────
//
// A GLIF3-shaped cell carrying the real two-regime refractory pattern, verbatim in the shape
// tests/fixtures/nml/glif3_single_cell.nml uses:
//
//   asc1/asc2   regime-free TimeDerivatives -- they decay through BOTH regimes
//   v           TimeDerivative in `integrating` only. Its ABSENCE from `refractory` is the
//               refractory period: v does not move, so it stays pinned at vreset.
//   refractoryTimeElapsed
//               TimeDerivative in `refractory` only, so the countdown is frozen the rest of
//               the time; `refractory`'s OnEntry zeroes it at the transition in.
//
// The numbers are chosen so the assertions are exact rather than approximate: t_ref = 1ms at
// dt = 0.1ms is ten ticks, the injected current is far enough above rheobase that v crosses
// threshold quickly, and vreset sits below EL so "pinned at vreset" cannot be confused with
// "relaxing towards rest".
String glif3_refractory_lems_xml(const String &network_file, const String &recording_file) {
    return R"(<Lems>
    <Target component="sim1"/>

    <ComponentType name="glif3RefractoryCell" extends="baseSpikingCell"
                   description="GLIF3: two after-spike currents plus a refractory Regime pair.">
        <Parameter name="C" dimension="capacitance"/>
        <Parameter name="gL" dimension="conductance"/>
        <Parameter name="EL" dimension="voltage"/>
        <Parameter name="vth" dimension="voltage"/>
        <Parameter name="vreset" dimension="voltage"/>
        <Parameter name="t_ref" dimension="time"/>
        <Parameter name="tauAsc1" dimension="time"/>
        <Parameter name="tauAsc2" dimension="time"/>
        <Parameter name="ascAdd1" dimension="current"/>
        <Parameter name="ascAdd2" dimension="current"/>

        <Dynamics>
            <StateVariable name="v" dimension="voltage"/>
            <StateVariable name="asc1" dimension="current"/>
            <StateVariable name="asc2" dimension="current"/>
            <StateVariable name="refractoryTimeElapsed" dimension="time"/>

            <DerivedVariable name="iSyn" dimension="current" select="synapses[*]/i" reduce="add"/>
            <DerivedVariable name="ascSum" dimension="current" value="asc1 + asc2"/>

            <TimeDerivative variable="asc1" value="-asc1 / tauAsc1"/>
            <TimeDerivative variable="asc2" value="-asc2 / tauAsc2"/>

            <OnStart>
                <StateAssignment variable="v" value="EL"/>
                <StateAssignment variable="asc1" value="0"/>
                <StateAssignment variable="asc2" value="0"/>
            </OnStart>

            <Regime name="integrating" initial="true">
                <TimeDerivative variable="v" value="(gL * (EL - v) + iSyn + ascSum) / C"/>
                <OnCondition test="v .gt. vth">
                    <EventOut port="spike"/>
                    <StateAssignment variable="v" value="vreset"/>
                    <StateAssignment variable="asc1" value="asc1 + ascAdd1"/>
                    <StateAssignment variable="asc2" value="asc2 + ascAdd2"/>
                    <Transition regime="refractory"/>
                </OnCondition>
            </Regime>
            <Regime name="refractory">
                <OnEntry>
                    <StateAssignment variable="refractoryTimeElapsed" value="0"/>
                </OnEntry>
                <TimeDerivative variable="refractoryTimeElapsed" value="1"/>
                <OnCondition test="refractoryTimeElapsed .geq. t_ref">
                    <Transition regime="integrating"/>
                </OnCondition>
            </Regime>
        </Dynamics>
    </ComponentType>

    <Include file=")" + network_file + R"("/>

    <Simulation id="sim1" length="30ms" step="0.1ms" target="net1">
        <OutputFile id="of1" fileName=")" + recording_file + R"(">
            <OutputColumn id="c0" quantity="popGlif[0]/v"/>
        </OutputFile>
    </Simulation>
</Lems>
)";
}

String glif3_refractory_network_nml() {
    return R"(<neuroml id="glif3refractorynet">
    <glif3RefractoryCell id="glif3Cell" C="100pF" gL="10nS" EL="-70mV" vth="-50mV"
                         vreset="-75mV" t_ref="1ms" tauAsc1="100ms" tauAsc2="10ms"
                         ascAdd1="-100pA" ascAdd2="-200pA"/>
    <pulseGenerator id="pg0" delay="0ms" duration="30ms" amplitude="1nA"/>

    <network id="net1">
        <population id="popGlif" component="glif3Cell" size="1"/>
        <explicitInput target="popGlif[0]" input="pg0"/>
    </network>
</neuroml>
)";
}

// A cell type declaring a Regime and NO StateVariable. Legal LEMS, and the shape that
// defeats a "cell_state is empty" test for "there is nothing to record": the appended regime
// index gives such a type one cell_state slot, so the buffer is not empty and slot 0 is the
// regime number rather than a membrane potential.
String regime_only_lems_xml(const String &network_file, const String &recording_file) {
    return R"(<Lems>
    <Target component="sim1"/>

    <ComponentType name="regimeOnlyCell" extends="baseSpikingCell"
                   description="Declares a Regime and no StateVariable at all.">
        <Parameter name="fireTime" dimension="time"/>

        <Dynamics>
            <Regime name="waiting" initial="true">
                <OnCondition test="t .geq. fireTime">
                    <EventOut port="spike"/>
                    <Transition regime="done"/>
                </OnCondition>
            </Regime>

            <Regime name="done"/>
        </Dynamics>
    </ComponentType>

    <Include file=")" + network_file + R"("/>

    <Simulation id="sim1" length="1ms" step="0.1ms" target="net1">
        <OutputFile id="of1" fileName=")" + recording_file + R"(">
        </OutputFile>
    </Simulation>
</Lems>
)";
}

String regime_only_network_nml() {
    return R"(<neuroml id="regimeonlynet">
    <regimeOnlyCell id="cell0" fireTime="0.25ms"/>

    <network id="net1">
        <population id="popOnly" component="cell0" size="1"/>
    </network>
</neuroml>
)";
}

String write_regime_only_model(const FixtureDirectory &fixture) {
    fixture.write("net.nml", regime_only_network_nml());
    return fixture.write("model.xml",
                         regime_only_lems_xml("net.nml", fixture.path_of("out.spire")));
}

String write_glif3_refractory_model(const FixtureDirectory &fixture) {
    fixture.write("net.nml", glif3_refractory_network_nml());
    return fixture.write("model.xml",
                         glif3_refractory_lems_xml("net.nml", fixture.path_of("out.spire")));
}

// The four state variables plus the appended regime slot, by name, for the one glif3 neuron.
struct Glif3CellReader {
    const f32 *cell_state = nullptr;
    s64 state_base = 0;

    Glif3CellReader(const SpikeEngine &engine, s64 neuron_index) {
        cell_state = engine.cell_state.get_contents();
        state_base = engine.cell_state_base.get_contents()[neuron_index];
    }

    f32 membrane_potential() const { return cell_state[state_base + 0]; }
    f32 after_spike_current_one() const { return cell_state[state_base + 1]; }
    f32 after_spike_current_two() const { return cell_state[state_base + 2]; }
    f32 refractory_time_elapsed() const { return cell_state[state_base + 3]; }
    // Appended after every StateVariable -- see nml::cell_state_slot_count.
    s32 regime_index() const { return (s32)cell_state[state_base + 4]; }
};

// One tick's worth of everything the refractory assertions look at.
struct Glif3Sample {
    s64 tick = 0;
    bool spiked = false;
    f32 membrane_potential = 0.0f;
    f32 after_spike_current_one = 0.0f;
    f32 after_spike_current_two = 0.0f;
    s32 regime_index = 0;
};

// How many ticks the model's own refractory countdown actually takes to reach `t_ref`.
//
// NOT t_ref / dt. The countdown is a forward-Euler integration of `d(refractoryTimeElapsed)
// /dt = 1`, so it is dt added to itself in f32, and repeated addition of 0.1ms accumulates
// slightly SHORT (9.9999993e-4 after ten steps) while 1ms rounds slightly LONG
// (1.0000000e-3). A refractory period written as an exact multiple of dt therefore runs one
// tick longer than the multiple. jLEMS's f64 accumulation falls short in the same direction,
// so this is the reference implementation's behaviour too rather than a divergence from it --
// but it is not what "t_ref / dt ticks" would predict, which is why it is computed here
// instead of assumed.
s64 refractory_window_tick_count(f32 step_dt, f32 refractory_period) {
    f32 accumulated = 0.0f;
    s64 tick_count = 0;
    while (accumulated < refractory_period && tick_count < 1000) {
        accumulated = accumulated + step_dt * 1.0f;
        tick_count += 1;
    }
    return tick_count;
}

spikecorec::Vector<Glif3Sample> run_glif3_refractory(SpikeEngine &engine, s64 tick_count) {
    const Glif3CellReader reader(engine, /*neuron_index=*/0);
    const s32 *spike_flags = engine.spike_flags.get_contents();

    spikecorec::Vector<Glif3Sample> samples;
    for (s64 tick = 0; tick < tick_count; ++tick) {
        engine.step_simulation(tick);
        samples.push_back(Glif3Sample{tick, spike_flags[0] != 0, reader.membrane_potential(),
                                      reader.after_spike_current_one(),
                                      reader.after_spike_current_two(), reader.regime_index()});
    }
    return samples;
}

// ── regime-guard fixture ───────────────────────────────────────────────────────
//
// A cell whose SECOND regime carries an OnCondition that is true on every tick of the whole
// run (`t .geq. 0`) and whose body is observable (a counter). While the cell is in the first
// regime that condition is inactive, so the counter must stay at zero -- and the ONLY thing
// holding it there is the regime guard on the condition.
//
// The GLIF3 fixture cannot show this. Its refractory condition is
// `refractoryTimeElapsed .geq. t_ref`, and the transition into `refractory` zeroes that
// countdown in the same block, so an unguarded version happens to produce the same trace. A
// test built on it would pass with the guard deleted, which is exactly what this one is here
// to rule out.
String regime_guard_lems_xml(const String &network_file, const String &recording_file) {
    return R"(<Lems>
    <Target component="sim1"/>

    <ComponentType name="regimeGuardCell" extends="baseCell"
                   description="Counts, but only once its second regime is entered.">
        <Parameter name="risingRate" dimension="none"/>
        <Parameter name="switchLevel" dimension="none"/>

        <Dynamics>
            <StateVariable name="level" dimension="none"/>
            <StateVariable name="countingRegimeFirings" dimension="none"/>

            <OnStart>
                <StateAssignment variable="level" value="0"/>
                <StateAssignment variable="countingRegimeFirings" value="0"/>
            </OnStart>

            <Regime name="rising" initial="true">
                <TimeDerivative variable="level" value="risingRate"/>
                <OnCondition test="level .gt. switchLevel">
                    <Transition regime="counting"/>
                </OnCondition>
            </Regime>
            <Regime name="counting">
                <OnCondition test="t .geq. 0">
                    <StateAssignment variable="countingRegimeFirings"
                                     value="countingRegimeFirings + 1"/>
                </OnCondition>
            </Regime>
        </Dynamics>
    </ComponentType>

    <Include file=")" + network_file + R"("/>

    <Simulation id="sim1" length="3ms" step="0.1ms" target="net1">
        <OutputFile id="of1" fileName=")" + recording_file + R"(">
            <OutputColumn id="c0" quantity="popGuard[0]/countingRegimeFirings"/>
        </OutputFile>
    </Simulation>
</Lems>
)";
}

String regime_guard_network_nml() {
    return R"(<neuroml id="regimeguardnet">
    <regimeGuardCell id="guardCell" risingRate="1000" switchLevel="1"/>

    <network id="net1">
        <population id="popGuard" component="guardCell" size="1"/>
    </network>
</neuroml>
)";
}

String write_regime_guard_model(const FixtureDirectory &fixture) {
    fixture.write("net.nml", regime_guard_network_nml());
    return fixture.write("model.xml",
                         regime_guard_lems_xml("net.nml", fixture.path_of("out.spire")));
}

// ── unwired synapse fixture ────────────────────────────────────────────────────
//
// Two synapse prototypes DECLARED and no projection naming either, alongside a projection
// that names no synapse at all. One of them is conductance-based, which is refused outright
// when it is wired -- so this also pins down that the refusal is on WIRED prototypes rather
// than on declarations, which is what keeps a document's unused library from failing a model
// that never touches it.
String unwired_synapse_network_nml() {
    return R"(<neuroml id="unwiredsynapsenet">
    <oneShotCell id="source0" fireTime="0.25ms"/>
    <latchCell id="target0"/>
    <alphaCurrentSynapse id="unusedAlpha" tau="0.5ms" ibase="1nA"/>
    <expOneSynapse id="unusedConductance" gbase="0.5nS" erev="0mV" tauDecay="3ms"/>

    <network id="net1">
        <population id="popSource" component="source0" size="1"/>
        <population id="popTarget" component="target0" size="1"/>

        <projection id="proj0" presynapticPopulation="popSource"
                    postsynapticPopulation="popTarget">
            <connectionWD id="0" preCellId="../popSource[0]" postCellId="../popTarget[0]"
                          weight="2.5" delay="0.1ms"/>
        </projection>
    </network>
</neuroml>
)";
}

String write_unwired_synapse_model(const FixtureDirectory &fixture) {
    fixture.write("net.nml", unwired_synapse_network_nml());
    return fixture.write("model.xml",
                         synaptic_input_lems_xml("net.nml", fixture.path_of("out.spire")));
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
            (s64)engine.network_input_ring_depth * engine.total_neuron_count;
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
            (s64)engine.network_input_ring_depth * engine.total_neuron_count;
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
// ── per-edge synapse fixtures ──────────────────────────────────────────────────
//
// expCurrentSynapse is an exponential CURRENT-based synapse, declared inline rather than
// taken from the standard library because the library carries no current-based exponential
// (expOneSynapse and friends are conductance-based, and those are refused). Its shape is the
// one that makes per-edge delivery observable: the OnEvent handler bumps the very variable
// the `i` exposure reads, so the scalar a spike delivers is (this edge's decayed state) plus
// (this spike's own contribution) -- both non-zero, and different on a second spike.
//
// twoShotCell fires twice, which is what separates "the delivered value is computed from
// live state" from "the delivered value was baked at construction".
String per_edge_synapse_lems_xml(const String &network_file, const String &recording_file) {
    return R"(<Lems>
    <Target component="sim1"/>

    <ComponentType name="twoShotCell" extends="baseSpikingCell"
                   description="Emits exactly two spikes, at fireTimeOne and fireTimeTwo.">
        <Parameter name="fireTimeOne" dimension="time"/>
        <Parameter name="fireTimeTwo" dimension="time"/>

        <Dynamics>
            <StateVariable name="firedCount" dimension="none"/>

            <OnStart>
                <StateAssignment variable="firedCount" value="0"/>
            </OnStart>

            <OnCondition test="t .geq. fireTimeOne .and. firedCount .lt. 0.5">
                <StateAssignment variable="firedCount" value="1"/>
                <EventOut port="spike"/>
            </OnCondition>

            <OnCondition test="t .geq. fireTimeTwo .and. firedCount .lt. 1.5">
                <StateAssignment variable="firedCount" value="2"/>
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

    <ComponentType name="expCurrentSynapse" extends="baseCurrentBasedSynapse"
                   description="Exponentially decaying current-based synapse.">
        <Property name="weight" dimension="none" defaultValue="1"/>
        <Parameter name="tau" dimension="time"/>
        <Parameter name="ibase" dimension="current"/>

        <Dynamics>
            <StateVariable name="I" dimension="current"/>

            <DerivedVariable name="i" exposure="i" dimension="current" value="I"/>

            <TimeDerivative variable="I" value="-I / tau"/>

            <OnStart>
                <StateAssignment variable="I" value="0"/>
            </OnStart>

            <OnEvent port="in">
                <StateAssignment variable="I" value="I + weight * ibase"/>
            </OnEvent>
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

// One source firing twice, through one expCurrentSynapse, onto one target.
String per_edge_synapse_network_nml() {
    return R"(<neuroml id="peredgesynapsenet">
    <twoShotCell id="source0" fireTimeOne="0.25ms" fireTimeTwo="0.85ms"/>
    <traceCell id="target0"/>
    <expCurrentSynapse id="expSyn" tau="0.5ms" ibase="1nA"/>

    <network id="net1">
        <population id="popSource" component="source0" size="1"/>
        <population id="popTarget" component="target0" size="1"/>

        <projection id="proj0" presynapticPopulation="popSource"
                    postsynapticPopulation="popTarget" synapse="expSyn">
            <connectionWD id="0" preCellId="../popSource[0]" postCellId="../popTarget[0]"
                          weight="1" delay="0.1ms"/>
        </projection>
    </network>
</neuroml>
)";
}

String write_per_edge_synapse_model(const FixtureDirectory &fixture) {
    fixture.write("net.nml", per_edge_synapse_network_nml());
    return fixture.write("model.xml",
                         per_edge_synapse_lems_xml("net.nml", fixture.path_of("out.spire")));
}

// Two sources firing twice each, onto ONE target, through two expCurrentSynapse prototypes
// whose only difference is `tau`. Every edge carries its own state, so a lazy catch-up that
// used the wrong elapsed count or the wrong edge's slot shows up as a wrong amplitude here
// and nowhere else.
String two_tau_per_edge_network_nml() {
    return R"(<neuroml id="twotauperedgenet">
    <twoShotCell id="source0" fireTimeOne="0.25ms" fireTimeTwo="0.85ms"/>
    <twoShotCell id="source1" fireTimeOne="0.45ms" fireTimeTwo="1.65ms"/>
    <traceCell id="target0"/>
    <expCurrentSynapse id="expFast" tau="0.5ms" ibase="1nA"/>
    <expCurrentSynapse id="expSlow" tau="2ms" ibase="3nA"/>

    <network id="net1">
        <population id="popSourceFast" component="source0" size="1"/>
        <population id="popSourceSlow" component="source1" size="1"/>
        <population id="popTarget" component="target0" size="1"/>

        <projection id="projFast" presynapticPopulation="popSourceFast"
                    postsynapticPopulation="popTarget" synapse="expFast">
            <connectionWD id="0" preCellId="../popSourceFast[0]" postCellId="../popTarget[0]"
                          weight="1" delay="0.1ms"/>
        </projection>
        <projection id="projSlow" presynapticPopulation="popSourceSlow"
                    postsynapticPopulation="popTarget" synapse="expSlow">
            <connectionWD id="0" preCellId="../popSourceSlow[0]" postCellId="../popTarget[0]"
                          weight="1" delay="0.2ms"/>
        </projection>
    </network>
</neuroml>
)";
}

// A synapse whose OnStart leaves its state NON-ZERO, so the state is already moving before
// the first spike ever arrives.
//
// That is the only shape in which the seed of the last-advanced-tick plane is observable at
// all: with an all-zero OnStart and a homogeneous derivative, a state of zero decays to zero
// however many steps are applied, so seeding "never advanced" as tick 0 rather than -1 costs
// exactly one step of nothing and both policies agree anyway. Here the missing step is one
// step of a real decay, and the two disagree by that factor.
String priming_synapse_lems_xml(const String &network_file, const String &recording_file) {
    return R"(<Lems>
    <Target component="sim1"/>

    <ComponentType name="twoShotCell" extends="baseSpikingCell"
                   description="Emits exactly two spikes, at fireTimeOne and fireTimeTwo.">
        <Parameter name="fireTimeOne" dimension="time"/>
        <Parameter name="fireTimeTwo" dimension="time"/>

        <Dynamics>
            <StateVariable name="firedCount" dimension="none"/>

            <OnStart>
                <StateAssignment variable="firedCount" value="0"/>
            </OnStart>

            <OnCondition test="t .geq. fireTimeOne .and. firedCount .lt. 0.5">
                <StateAssignment variable="firedCount" value="1"/>
                <EventOut port="spike"/>
            </OnCondition>

            <OnCondition test="t .geq. fireTimeTwo .and. firedCount .lt. 1.5">
                <StateAssignment variable="firedCount" value="2"/>
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

    <ComponentType name="primedCurrentSynapse" extends="baseCurrentBasedSynapse"
                   description="Exponentially decaying current-based synapse, primed at OnStart.">
        <Property name="weight" dimension="none" defaultValue="1"/>
        <Parameter name="tau" dimension="time"/>
        <Parameter name="ibase" dimension="current"/>

        <Dynamics>
            <StateVariable name="I" dimension="current"/>

            <DerivedVariable name="i" exposure="i" dimension="current" value="I"/>

            <TimeDerivative variable="I" value="-I / tau"/>

            <OnStart>
                <StateAssignment variable="I" value="ibase"/>
            </OnStart>

            <OnEvent port="in">
                <StateAssignment variable="I" value="I + weight * ibase"/>
            </OnEvent>
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

String priming_synapse_network_nml() {
    return R"(<neuroml id="primingsynapsenet">
    <twoShotCell id="source0" fireTimeOne="0.25ms" fireTimeTwo="0.85ms"/>
    <traceCell id="target0"/>
    <primedCurrentSynapse id="primedSyn" tau="0.5ms" ibase="1nA"/>

    <network id="net1">
        <population id="popSource" component="source0" size="1"/>
        <population id="popTarget" component="target0" size="1"/>

        <projection id="proj0" presynapticPopulation="popSource"
                    postsynapticPopulation="popTarget" synapse="primedSyn">
            <connectionWD id="0" preCellId="../popSource[0]" postCellId="../popTarget[0]"
                          weight="1" delay="0.1ms"/>
        </projection>
    </network>
</neuroml>
)";
}

// One source, two targets: one edge through a synapse and one through none at all. The
// per-edge program plane is the only thing that tells them apart, so a slot flattened for
// the wrong edge -- or a "no synapse" edge that picks up a program anyway -- shows up as
// synapse dynamics running on a connection that declared none.
String mixed_synapse_network_nml() {
    return R"(<neuroml id="mixedsynapsenet">
    <twoShotCell id="source0" fireTimeOne="0.25ms" fireTimeTwo="0.85ms"/>
    <traceCell id="target0"/>
    <expCurrentSynapse id="expSyn" tau="0.5ms" ibase="1nA"/>

    <network id="net1">
        <population id="popSource" component="source0" size="1"/>
        <population id="popTarget" component="target0" size="2"/>

        <projection id="projSynapse" presynapticPopulation="popSource"
                    postsynapticPopulation="popTarget" synapse="expSyn">
            <connectionWD id="0" preCellId="../popSource[0]" postCellId="../popTarget[0]"
                          weight="1" delay="0.1ms"/>
        </projection>
        <projection id="projPlain" presynapticPopulation="popSource"
                    postsynapticPopulation="popTarget">
            <connectionWD id="0" preCellId="../popSource[0]" postCellId="../popTarget[1]"
                          weight="2.5" delay="0.1ms"/>
        </projection>
    </network>
</neuroml>
)";
}

String write_mixed_synapse_model(const FixtureDirectory &fixture) {
    fixture.write("net.nml", mixed_synapse_network_nml());
    return fixture.write("model.xml",
                         per_edge_synapse_lems_xml("net.nml", fixture.path_of("out.spire")));
}

String write_priming_synapse_model(const FixtureDirectory &fixture) {
    fixture.write("net.nml", priming_synapse_network_nml());
    return fixture.write("model.xml",
                         priming_synapse_lems_xml("net.nml", fixture.path_of("out.spire")));
}

String write_two_tau_per_edge_model(const FixtureDirectory &fixture) {
    fixture.write("net.nml", two_tau_per_edge_network_nml());
    return fixture.write("model.xml",
                         per_edge_synapse_lems_xml("net.nml", fixture.path_of("out.spire")));
}

// expCurrentSynapse's own dynamics, integrated on the host exactly as the generated kernel
// does under either update policy: one forward-Euler step per elapsed tick, then the arrival
// handler, then `i` read off what that left. Written out rather than compared against
// exp(-t/tau) because the kernel integrates with forward Euler at the model's own dt, and at
// dt = tau / 5 the two differ by percent -- the point of the comparison is the amplitude and
// the rate, so the expectation has to be the same discretisation.
struct ExponentialCurrentSynapseReference {
    f32 tau = 0.0f;
    f32 ibase = 0.0f;
    f32 current_I = 0.0f;
    s64 last_advanced_tick = -1;

    ExponentialCurrentSynapseReference(f32 tau, f32 ibase) : tau(tau), ibase(ibase) {}

    // Advances this edge to `tick`, applies one arrival of `weight`, and returns the scalar
    // the edge delivers.
    f32 deliver(f32 dt, s64 tick, f32 weight) {
        for (s64 elapsed = last_advanced_tick; elapsed < tick; ++elapsed) {
            current_I = current_I + dt * (-current_I / tau);
        }
        last_advanced_tick = tick;
        current_I = current_I + weight * ibase;
        return current_I;
    }
};

// Every non-zero slot of the delay ring, as (flat index, value). One arrival should occupy
// exactly one of them.
struct RingArrival {
    s64 element_index = 0;
    f32 value = 0.0f;
};

spikecorec::Vector<RingArrival> non_zero_ring_arrivals(const SpikeEngine &engine) {
    const f32 *network_inputs = engine.network_inputs.get_contents();
    const s64 element_count =
            (s64)engine.network_input_ring_depth * engine.total_neuron_count;

    spikecorec::Vector<RingArrival> arrivals;
    for (s64 element_index = 0; element_index < element_count; ++element_index) {
        if (network_inputs[element_index] == 0.0f) continue;
        arrivals.push_back(RingArrival{element_index, network_inputs[element_index]});
    }
    return arrivals;
}

// Where an arrival due on `arrival_tick` for `neuron_index` lands in the flat ring.
s64 ring_element_index(const SpikeEngine &engine, s64 arrival_tick, s64 neuron_index) {
    return (arrival_tick % (s64)engine.network_input_ring_depth) * engine.total_neuron_count +
           neuron_index;
}

} // namespace

// ── current-based synapse dynamics ─────────────────────────────────────────────

TEST(SpikeEngine, a_spike_delivers_one_scalar_into_one_ring_slot_from_live_per_edge_state) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // The test that separates per-edge synapse state from a weight baked at construction.
    // When the source spikes, its outgoing edge advances its OWN state to this tick, runs the
    // arrival handler, and writes the resulting scalar into exactly ONE ring slot, at
    // tick + that edge's delay. The second spike therefore delivers a DIFFERENT scalar,
    // because the state moved in between -- which a construction-time weight could not do.
    FixtureDirectory fixture("neuroml_per_edge_synapse");
    String model_path = write_per_edge_synapse_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
    ASSERT_EQ(engine.total_neuron_count, 2);

    // One state variable (I) for the one lowered program, one plane over the edge slots.
    EXPECT_EQ(engine.per_edge_synapse_variable_count, 1);
    EXPECT_EQ(engine.weights.per_edge_variable_count, 1);
    ASSERT_NE(engine.weights.per_edge_variable_values.pointer, nullptr);

    const f32 step_dt = (f32)engine.network_details.step_dt;
    const s64 first_spike_tick = 3;   // first tick with t >= 0.25ms at dt = 0.1ms
    const s64 second_spike_tick = 9;  // first tick with t >= 0.85ms
    const s64 delay_tick_count = 1;   // 0.1ms at dt = 0.1ms

    ExponentialCurrentSynapseReference reference(/*tau=*/5.0e-4f, /*ibase=*/1.0e-9f);
    f32 first_delivered = 0.0f;
    f32 second_delivered = 0.0f;

    for (s64 tick = 0; tick <= second_spike_tick; ++tick) {
        engine.step_simulation(tick);

        if (tick != first_spike_tick && tick != second_spike_tick) continue;

        ASSERT_EQ(engine.spike_flags.get_contents()[0], 1)
                << "the source did not spike on tick " << tick;

        // ONE slot, not a plane and not a per-target accumulation spread over several ticks.
        const spikecorec::Vector<RingArrival> arrivals = non_zero_ring_arrivals(engine);
        ASSERT_EQ(arrivals.size(), 1u)
                << "tick " << tick << ": a spike wrote " << arrivals.size()
                << " ring slots, not one";
        EXPECT_EQ(arrivals[0].element_index,
                  ring_element_index(engine, tick + delay_tick_count, /*neuron_index=*/1))
                << "tick " << tick << ": the arrival did not land at tick + delay for the "
                << "edge's target";

        const f32 expected = reference.deliver(step_dt, tick, /*weight=*/1.0f);
        EXPECT_NEAR(arrivals[0].value, expected, std::fabs(expected) * 1e-4f + 1e-15f)
                << "tick " << tick << ": delivered " << arrivals[0].value << " where the edge's "
                << "own state gives " << expected;

        (tick == first_spike_tick ? first_delivered : second_delivered) = arrivals[0].value;
    }

    // The state was live: the second spike lands on a partly-decayed remainder of the first,
    // so it delivers strictly more. Equal values would mean the edge's state was reset, never
    // written, or baked at construction.
    EXPECT_GT(first_delivered, 0.0f);
    EXPECT_GT(second_delivered, first_delivered)
            << "the second spike delivered the same scalar as the first, so the edge's state "
               "did not evolve between them";

    // And the target reads that same scalar out of the ring on the following tick.
    engine.step_simulation(second_spike_tick + 1);
    EXPECT_NEAR(trace_cell_last_input(engine, /*neuron_index=*/1), second_delivered,
                std::fabs(second_delivered) * 1e-5f + 1e-15f);

    engine.shutdown();
}

TEST(SpikeEngine, per_edge_synapse_state_lives_in_the_weight_matrix_and_evolves_there) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // Where the state lives is not an implementation detail: the generated kernel reads and
    // writes the WeightMatrix's own per-edge variable planes, so the host accessor and the
    // device both see one copy. A mirrored buffer would drift silently.
    FixtureDirectory fixture("neuroml_per_edge_state_storage");
    String model_path = write_per_edge_synapse_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    // Its Ck is pinned to all-zero, which is what makes the stored plane the whole value at
    // any magnitude rather than a delta on top of an order-1 reconstruction.
    const s64 matrix_index = engine.weights.per_edge_variable_matrix_index(0);
    EXPECT_TRUE(engine.weights.is_per_edge_variable_matrix(matrix_index));
    EXPECT_EQ(engine.weights.get_edge_variable(0, /*source_node=*/0, /*target_node=*/1), 0.0f);

    // Run past the first spike; the edge's state is now non-zero, and it is readable through
    // the compressed storage rather than out of a separate engine buffer.
    for (s64 tick = 0; tick <= 4; ++tick) engine.step_simulation(tick);

    const f32 after_first_spike =
            engine.weights.get_edge_variable(0, /*source_node=*/0, /*target_node=*/1);
    EXPECT_GT(after_first_spike, 0.0f)
            << "the edge's synapse state is still zero after a spike travelled down it";
    EXPECT_NEAR(after_first_spike, 1.0e-9f, 1.0e-13f)
            << "the state is not the arrival handler's `weight * ibase` at SI magnitude";

    // The same value read through the generic family accessor, which is what proves this is
    // the shared-basis storage and not a private array wearing its name.
    EXPECT_FLOAT_EQ(engine.weights.get_for_matrix(0, 1, matrix_index), after_first_spike);

    engine.shutdown();
}

TEST(SpikeEngine, lazy_and_eager_synapse_updates_produce_identical_results) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // The equivalence that proves the catch-up is right. Lazy advances an edge only when its
    // source spikes, across the whole elapsed interval at once; eager advances every edge
    // every tick. A wrong elapsed count -- an off-by-one at the seed, a catch-up that stops
    // one tick early -- changes nothing structural and shows up ONLY as a subtly wrong
    // amplitude, which is exactly what comparing the two policies catches.
    //
    // Two edges with different tau, different ibase, different delays and sources that fire
    // at different ticks, so the elapsed intervals differ per edge and a per-edge slot mix-up
    // is visible too.
    FixtureDirectory lazy_fixture("neuroml_lazy_eager_lazy");
    FixtureDirectory eager_fixture("neuroml_lazy_eager_eager");
    String lazy_model_path = write_two_tau_per_edge_model(lazy_fixture);
    String eager_model_path = write_two_tau_per_edge_model(eager_fixture);

    SpikeEngine lazy_engine(lazy_model_path, /*enable_hebbian_learning=*/false,
                            /*use_lazy_synapse_updates=*/true);
    SpikeEngine eager_engine(eager_model_path, /*enable_hebbian_learning=*/false,
                             /*use_lazy_synapse_updates=*/false);

    ASSERT_TRUE(lazy_engine.use_lazy_synapse_updates);
    ASSERT_FALSE(eager_engine.use_lazy_synapse_updates);
    ASSERT_EQ(lazy_engine.total_neuron_count, eager_engine.total_neuron_count);
    ASSERT_EQ(lazy_engine.per_edge_synapse_variable_count, 2);

    const s64 tick_count = 40;
    f32 largest_delivered = 0.0f;
    s64 non_zero_delivery_count = 0;

    // Each edge's own last spike tick, which is the only tick at which the two policies' per
    // edge STATE is comparable: lazy leaves an edge's state as of that edge's last spike,
    // where eager keeps decaying it every tick afterwards. Both are right -- nothing observes
    // the state in between -- so comparing them at the end of the run would compare two
    // different instants and fail for the wrong reason.
    const s64 fast_edge_last_spike_tick = 9;
    const s64 slow_edge_last_spike_tick = 17;
    f32 lazy_state_by_edge[2] = {0.0f, 0.0f};
    f32 eager_state_by_edge[2] = {0.0f, 0.0f};

    for (s64 tick = 0; tick < tick_count; ++tick) {
        lazy_engine.step_simulation(tick);
        eager_engine.step_simulation(tick);

        const f32 lazy_delivered = trace_cell_last_input(lazy_engine, /*neuron_index=*/2);
        const f32 eager_delivered = trace_cell_last_input(eager_engine, /*neuron_index=*/2);

        EXPECT_NEAR(lazy_delivered, eager_delivered,
                    std::fabs(eager_delivered) * 1e-5f + 1e-18f)
                << "tick " << tick << ": lazy delivered " << lazy_delivered << " and eager "
                << eager_delivered;

        largest_delivered = std::max(largest_delivered, std::fabs(eager_delivered));
        if (eager_delivered != 0.0f) non_zero_delivery_count += 1;

        if (tick == fast_edge_last_spike_tick) {
            lazy_state_by_edge[0] = lazy_engine.weights.get_edge_variable(0, 0, 2);
            eager_state_by_edge[0] = eager_engine.weights.get_edge_variable(0, 0, 2);
        }
        if (tick == slow_edge_last_spike_tick) {
            lazy_state_by_edge[1] = lazy_engine.weights.get_edge_variable(1, 1, 2);
            eager_state_by_edge[1] = eager_engine.weights.get_edge_variable(1, 1, 2);
        }
    }

    // A comparison of two all-zero traces would pass for the wrong reason.
    EXPECT_GE(non_zero_delivery_count, 4)
            << "the two policies agreed on a trace that never delivered anything";
    EXPECT_GT(largest_delivered, 1e-10f);

    // The per-edge state each policy left behind agrees too, at each edge's last spike -- so
    // the agreement above is not two different states that happen to deliver the same
    // scalars.
    for (usize edge_position = 0; edge_position < 2; ++edge_position) {
        EXPECT_GT(std::fabs(eager_state_by_edge[edge_position]), 1e-12f)
                << "edge " << edge_position << " carried no state to compare";
        EXPECT_NEAR(lazy_state_by_edge[edge_position], eager_state_by_edge[edge_position],
                    std::fabs(eager_state_by_edge[edge_position]) * 1e-5f + 1e-18f)
                << "edge " << edge_position << " at its own last spike";
    }

    lazy_engine.shutdown();
    eager_engine.shutdown();
}

TEST(SpikeEngine, the_edge_attribute_plane_layout_matches_the_generated_source) {
    // The engine fills these planes and the generated kernel indexes them, each from its own
    // copy of the layout. A disagreement reads as a wrong delay or a wrong synapse program on
    // every edge at once, and never as a crash -- so the two copies are compared directly.
    nml::NML_ParseResult parse_result;
    nml::CellTypeSpecification cell_type;
    cell_type.name = "plainCell";
    cell_type.state_variable_names = {"v"};
    parse_result.cell_types.push_back(cell_type);
    parse_result.neurons.push_back(nml::Neuron{});

    const String tick_source = nml::generate_tick_kernel(parse_result).source;
    EXPECT_NE(tick_source.find("#define SPIKECOREC_EDGE_ATTRIBUTE_DELAY_PLANE " +
                               to_string(SpikeEngine::EDGE_ATTRIBUTE_DELAY_PLANE)),
              String::npos)
            << tick_source;
    EXPECT_NE(tick_source.find("#define SPIKECOREC_EDGE_ATTRIBUTE_PROGRAM_PLANE " +
                               to_string(SpikeEngine::EDGE_ATTRIBUTE_PROGRAM_PLANE)),
              String::npos)
            << tick_source;
    EXPECT_NE(tick_source.find("#define SPIKECOREC_EDGE_ATTRIBUTE_UPDATE_TICK_PLANE " +
                               to_string(SpikeEngine::EDGE_ATTRIBUTE_UPDATE_TICK_PLANE)),
              String::npos)
            << tick_source;

    // And the count covers every plane the generated source names.
    EXPECT_EQ(SpikeEngine::EDGE_ATTRIBUTE_PLANE_COUNT, 3);
}

TEST(SpikeEngine, an_edge_through_no_synapse_delivers_its_raw_weight_beside_one_that_does_not) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // A model mixing the two kinds of edge, which is what makes the per-edge program plane
    // load-bearing rather than decorative: an edge whose projection names no synapse must
    // deliver its raw weight, and an edge that names one must deliver its synapse's scalar,
    // in the same walk of the same source's row. Getting the plane wrong runs synapse
    // dynamics on a plain connection -- a plausible number on a connection that declared no
    // dynamics at all.
    FixtureDirectory fixture("neuroml_mixed_synapse");
    String model_path = write_mixed_synapse_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
    ASSERT_EQ(engine.total_neuron_count, 3);
    EXPECT_EQ(engine.per_edge_synapse_variable_count, 1);

    const s64 first_spike_tick = 3;
    for (s64 tick = 0; tick <= first_spike_tick; ++tick) engine.step_simulation(tick);
    ASSERT_EQ(engine.spike_flags.get_contents()[0], 1);

    // Two arrivals, both in the row for tick + 1, one per target.
    const spikecorec::Vector<RingArrival> arrivals = non_zero_ring_arrivals(engine);
    ASSERT_EQ(arrivals.size(), 2u);

    f32 synapse_target_arrival = 0.0f;
    f32 plain_target_arrival = 0.0f;
    for (const RingArrival &arrival : arrivals) {
        if (arrival.element_index == ring_element_index(engine, first_spike_tick + 1, 1)) {
            synapse_target_arrival = arrival.value;
        }
        if (arrival.element_index == ring_element_index(engine, first_spike_tick + 1, 2)) {
            plain_target_arrival = arrival.value;
        }
    }

    // The plain edge delivers exactly its declared weight, untouched by any synapse.
    EXPECT_FLOAT_EQ(plain_target_arrival, 2.5f);
    // The synapse edge delivers its handler's `weight * ibase` at SI magnitude, which the
    // plain weight would swamp by nine orders of magnitude if the two were confused.
    EXPECT_NEAR(synapse_target_arrival, 1.0e-9f, 1.0e-13f);

    // And the synapse-free edge left no per-edge state behind: nothing ran on it.
    EXPECT_FLOAT_EQ(engine.weights.get_edge_variable(0, /*source_node=*/0, /*target_node=*/2),
                    0.0f);
    EXPECT_NEAR(engine.weights.get_edge_variable(0, /*source_node=*/0, /*target_node=*/1),
                1.0e-9f, 1.0e-13f);

    engine.shutdown();
}

TEST(SpikeEngine, lazy_and_eager_agree_for_a_synapse_whose_state_starts_non_zero) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // What pins down the seed of the last-advanced-tick plane. An edge that has never been
    // advanced is seeded to -1, not 0, because at tick 0 the eager pass takes one step and
    // the catch-up has to take the same one. With an all-zero OnStart that step is a step of
    // nothing and both seeds agree, which is why this needs a synapse whose OnStart leaves
    // real state behind -- its first spike then lands on a value the two policies have
    // decayed a DIFFERENT number of times if the seed is wrong.
    FixtureDirectory lazy_fixture("neuroml_priming_lazy");
    FixtureDirectory eager_fixture("neuroml_priming_eager");
    String lazy_model_path = write_priming_synapse_model(lazy_fixture);
    String eager_model_path = write_priming_synapse_model(eager_fixture);

    SpikeEngine lazy_engine(lazy_model_path, /*enable_hebbian_learning=*/false,
                            /*use_lazy_synapse_updates=*/true);
    SpikeEngine eager_engine(eager_model_path, /*enable_hebbian_learning=*/false,
                             /*use_lazy_synapse_updates=*/false);

    const s64 first_spike_tick = 3;
    const s64 second_spike_tick = 9;

    // The reference the seed decides: I starts at ibase and takes one step per tick from tick
    // 0, so by the first spike it has taken first_spike_tick + 1 of them.
    const f32 step_dt = (f32)lazy_engine.network_details.step_dt;
    const f32 tau = 5.0e-4f;
    f32 expected_state = 1.0e-9f;
    for (s64 tick = 0; tick <= first_spike_tick; ++tick) {
        expected_state = expected_state + step_dt * (-expected_state / tau);
    }
    const f32 expected_first_delivery = expected_state + 1.0e-9f;

    f32 lazy_first_delivery = 0.0f;
    f32 eager_first_delivery = 0.0f;

    for (s64 tick = 0; tick <= second_spike_tick; ++tick) {
        lazy_engine.step_simulation(tick);
        eager_engine.step_simulation(tick);

        if (tick != first_spike_tick && tick != second_spike_tick) continue;

        const spikecorec::Vector<RingArrival> lazy_arrivals =
                non_zero_ring_arrivals(lazy_engine);
        const spikecorec::Vector<RingArrival> eager_arrivals =
                non_zero_ring_arrivals(eager_engine);
        ASSERT_EQ(lazy_arrivals.size(), 1u) << "tick " << tick;
        ASSERT_EQ(eager_arrivals.size(), 1u) << "tick " << tick;

        EXPECT_NEAR(lazy_arrivals[0].value, eager_arrivals[0].value,
                    std::fabs(eager_arrivals[0].value) * 1e-5f + 1e-18f)
                << "tick " << tick << ": the catch-up applied a different number of steps than "
                << "the eager pass";

        if (tick == first_spike_tick) {
            lazy_first_delivery = lazy_arrivals[0].value;
            eager_first_delivery = eager_arrivals[0].value;
        }
    }

    // And both agree with the step count derived above, so this is not two policies agreeing
    // on the same wrong number.
    EXPECT_NEAR(eager_first_delivery, expected_first_delivery,
                std::fabs(expected_first_delivery) * 1e-4f);
    EXPECT_NEAR(lazy_first_delivery, expected_first_delivery,
                std::fabs(expected_first_delivery) * 1e-4f);

    lazy_engine.shutdown();
    eager_engine.shutdown();
}

TEST(SpikeEngine, two_synapses_on_one_target_keep_their_own_per_edge_state) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // Two edges converging on one neuron through prototypes differing in tau and ibase. Each
    // carries its own state planes, so the target sees the sum of two independent responses.
    // Pooling them would have to decay the total at one rate, which is neither of theirs.
    FixtureDirectory fixture("neuroml_two_tau_per_edge");
    String model_path = write_two_tau_per_edge_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
    ASSERT_EQ(engine.total_neuron_count, 3);

    // One state variable per program, two programs.
    EXPECT_EQ(engine.per_edge_synapse_variable_count, 2);

    const f32 step_dt = (f32)engine.network_details.step_dt;
    ExponentialCurrentSynapseReference fast_reference(/*tau=*/5.0e-4f, /*ibase=*/1.0e-9f);
    ExponentialCurrentSynapseReference slow_reference(/*tau=*/2.0e-3f, /*ibase=*/3.0e-9f);

    // Source 0 fires at ticks 3 and 9 through the fast edge (delay 1 tick); source 1 fires at
    // ticks 5 and 17 through the slow edge (delay 2 ticks).
    const f32 first_fast = fast_reference.deliver(step_dt, 3, 1.0f);
    const f32 first_slow = slow_reference.deliver(step_dt, 5, 1.0f);
    const f32 second_fast = fast_reference.deliver(step_dt, 9, 1.0f);
    const f32 second_slow = slow_reference.deliver(step_dt, 17, 1.0f);

    spikecorec::Vector<f32> delivered_by_tick((usize)40, 0.0f);
    for (s64 tick = 0; tick < 40; ++tick) {
        engine.step_simulation(tick);
        delivered_by_tick[(usize)tick] = trace_cell_last_input(engine, /*neuron_index=*/2);
    }

    auto expect_delivery = [&](s64 tick, f32 expected) {
        EXPECT_NEAR(delivered_by_tick[(usize)tick], expected,
                    std::fabs(expected) * 1e-4f + 1e-15f)
                << "tick " << tick;
    };
    expect_delivery(4, first_fast);
    expect_delivery(7, first_slow);
    expect_delivery(10, second_fast);
    expect_delivery(19, second_slow);

    // The two really are on different clocks: the slow prototype's second delivery retains
    // far more of its first than the fast one's does, and the two ibase values keep their
    // amplitudes apart.
    EXPECT_GT(second_slow / first_slow, second_fast / first_fast)
            << "the two edges decayed at the same rate, so their state was pooled";
    EXPECT_GT(first_slow, 2.0f * first_fast) << "the two edges did not keep their own ibase";

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

TEST(SpikeEngine, a_declared_but_unwired_synapse_prototype_is_neither_lowered_nor_refused) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // A document routinely declares more synapse components than its network wires up. An
    // unwired one contributes nothing to any simulation, so it must neither be allocated for
    // nor put through the lowering's refusals -- including the conductance refusal, which
    // would otherwise make a model fail over a component it never uses.
    FixtureDirectory fixture("neuroml_unwired_synapse");
    String model_path = write_unwired_synapse_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    // Both prototypes reached the parse result, so this is not passing by their absence.
    EXPECT_GE(engine.network_details.synapse_prototypes.size(), 2u);

    // Neither is wired, so no per-edge synapse state is allocated at all.
    EXPECT_EQ(engine.per_edge_synapse_variable_count, 0);
    EXPECT_EQ(engine.weights.per_edge_variable_count, 0);

    // And the synapse-free projection still delivers its raw weight, one tick late.
    const LatchCellReader target(engine, /*neuron_index=*/1);
    for (s64 tick = 0; tick < 8; ++tick) engine.step_simulation(tick);
    EXPECT_NEAR(target.delivered(), 2.5f, 1e-5f);
    EXPECT_EQ(target.delivery_count(), 1);

    engine.shutdown();
}

TEST(SpikeEngine, an_alpha_current_synapse_delivers_the_state_its_handler_does_not_touch) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // Documented rather than asserted as desirable. alphaCurrentSynapse exposes i = I while
    // its OnEvent handler bumps J, so under impulsive delivery the scalar a spike forwards is
    // whatever I had decayed to -- which for an isolated first spike is exactly zero, and for
    // later spikes is the tail of the earlier ones rather than the new one's amplitude.
    //
    // This is a consequence of delivering one scalar per spike rather than integrating the
    // synapse's output every tick, and it is recorded here so a change to the ordering shows
    // up as a test failure instead of as a quietly different network.
    FixtureDirectory fixture("neuroml_alpha_impulsive_delivery");
    String model_path = write_alpha_synapse_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    const s64 spike_tick = 3;
    for (s64 tick = 0; tick <= spike_tick; ++tick) engine.step_simulation(tick);

    ASSERT_EQ(engine.spike_flags.get_contents()[0], 1);
    const spikecorec::Vector<RingArrival> arrivals = non_zero_ring_arrivals(engine);
    EXPECT_TRUE(arrivals.empty())
            << "alphaCurrentSynapse's first delivery was non-zero, so the arrival handler now "
               "reaches the exposed variable -- check the delivery ordering";

    // Its state DID move: J carries the arrival even though I, the exposed variable, does not
    // yet. So the synapse is live; it is the exposure that lags.
    EXPECT_FLOAT_EQ(engine.weights.get_edge_variable(0, 0, 1), 0.0f);  // I
    EXPECT_NEAR(engine.weights.get_edge_variable(1, 0, 1), 1.0e-9f, 1.0e-13f);  // J

    engine.shutdown();
}

// ── regimes ────────────────────────────────────────────────────────────────────

TEST(SpikeEngine, recording_a_type_with_a_regime_and_no_state_variable_is_refused) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // The regime index is appended to the cell's own cell_state chunk, so a type declaring a
    // Regime and no StateVariable has cell_state_element_count > 0 while having nothing to
    // record. A refusal keyed on that count therefore stopped firing for exactly this shape,
    // and the recorder would have written out regime numbers as if they were a membrane
    // potential -- plausible values, wrong quantity, no diagnostic.
    FixtureDirectory fixture("neuroml_regime_only_recording");
    String model_path = write_regime_only_model(fixture);

    try {
        SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
        engine.shutdown();
        FAIL() << "expected recording a StateVariable-free cell type to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("regimeOnlyCell"), String::npos) << message;
        EXPECT_NE(message.find("StateVariable"), String::npos) << message;
    }
}

TEST(SpikeEngine, a_regime_bearing_cell_type_widens_its_state_chunk_by_one_slot) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // The regime index is per-neuron state, so it lives in the cell's own cell_state chunk
    // rather than in a kernel argument -- the master kernel's argument table is full. The
    // engine and the generator have to agree on the resulting width or every neuron past the
    // first reads the previous one's state.
    FixtureDirectory fixture("neuroml_regime_layout");
    String model_path = write_glif3_refractory_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    ASSERT_EQ(engine.network_details.cell_types.size(), 1u);
    const nml::CellTypeSpecification &cell_type = engine.network_details.cell_types[0];
    EXPECT_EQ(cell_type.state_variable_names.size(), 4u);
    EXPECT_EQ(nml::cell_state_slot_count(cell_type), 5u);

    // One neuron, so the whole buffer is that one widened chunk.
    EXPECT_EQ(engine.total_neuron_count, 1);
    EXPECT_EQ(engine.cell_state_element_count, 5);

    // The initialize kernel seeded it with `integrating`, which is the regime marked initial.
    const Glif3CellReader reader(engine, /*neuron_index=*/0);
    EXPECT_EQ(reader.regime_index(), 0);

    engine.shutdown();
}

TEST(SpikeEngine, a_glif3_cell_refracts_pinning_v_while_its_after_spike_currents_decay) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // The headline behaviour the whole regime mechanism exists for. `v` has a TimeDerivative
    // in `integrating` and NONE in `refractory`; that absence is the refractory period. The
    // after-spike currents' derivatives sit outside both regimes, so they must keep decaying
    // THROUGH the refractory window -- a GLIF3 that froze them would still emit a spike train,
    // just one with the wrong adaptation.
    FixtureDirectory fixture("neuroml_glif3_refractory");
    String model_path = write_glif3_refractory_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    const f32 reset_potential = -0.075f;
    // t_ref = 1ms at step = 0.1ms -- eleven ticks, not ten; see
    // refractory_window_tick_count.
    const s64 refractory_tick_count =
            refractory_window_tick_count((f32)engine.network_details.step_dt, 1e-3f);
    ASSERT_EQ(refractory_tick_count, 11);

    const spikecorec::Vector<Glif3Sample> samples =
            run_glif3_refractory(engine, engine.lifetime);

    // ── it spikes ────────────────────────────────────────────────────────────
    s64 first_spike_tick = -1;
    s64 spike_count = 0;
    for (const Glif3Sample &sample : samples) {
        if (!sample.spiked) continue;
        if (first_spike_tick < 0) first_spike_tick = sample.tick;
        spike_count += 1;
    }
    ASSERT_GE(first_spike_tick, 0) << "the cell never crossed threshold under a 1nA step";
    ASSERT_GT(spike_count, 1) << "one spike proves nothing about leaving the refractory regime";

    // ── the regime index actually moved ──────────────────────────────────────
    EXPECT_EQ(samples[(usize)first_spike_tick].regime_index, 1)
            << "the Transition did not store the refractory regime's index";
    EXPECT_EQ(samples[(usize)first_spike_tick - 1].regime_index, 0);

    // ── v is pinned at vreset for exactly the refractory window ──────────────
    // The spike tick's own reset writes vreset; every tick of the window then leaves it
    // untouched, because `refractory` declares no TimeDerivative for v. The countdown reaches
    // t_ref on offset `refractory_tick_count`, so the regime flips back at the END of that
    // tick and v is still frozen through it -- it integrates again from the tick after.
    EXPECT_FLOAT_EQ(samples[(usize)first_spike_tick].membrane_potential, reset_potential);
    for (s64 offset = 0; offset <= refractory_tick_count; ++offset) {
        const Glif3Sample &sample = samples[(usize)(first_spike_tick + offset)];
        EXPECT_FLOAT_EQ(sample.membrane_potential, reset_potential)
                << "tick " << sample.tick << " (offset " << offset
                << " into the refractory window): v moved while refractory";
        EXPECT_EQ(sample.regime_index, offset < refractory_tick_count ? 1 : 0)
                << "tick " << sample.tick << ": wrong regime";
    }

    // One tick past the window v is integrating again, and rising: the 1nA step is still on.
    const Glif3Sample &resumed = samples[(usize)(first_spike_tick + refractory_tick_count + 1)];
    EXPECT_EQ(resumed.regime_index, 0);
    EXPECT_GT(resumed.membrane_potential, reset_potential)
            << "v did not resume integrating after the refractory period ended";

    // ── asc1/asc2 keep decaying DURING the refractory window ─────────────────
    // They are bumped by ascAdd1/ascAdd2 (both negative) at the spike, then decay towards
    // zero at tauAsc1 = 100ms and tauAsc2 = 10ms. Both are regime-free, so every refractory
    // tick has to move them.
    const Glif3Sample &at_spike = samples[(usize)first_spike_tick];
    ASSERT_LT(at_spike.after_spike_current_one, 0.0f) << "the spike did not bump asc1";
    ASSERT_LT(at_spike.after_spike_current_two, 0.0f) << "the spike did not bump asc2";

    for (s64 offset = 1; offset <= refractory_tick_count; ++offset) {
        const Glif3Sample &previous = samples[(usize)(first_spike_tick + offset - 1)];
        const Glif3Sample &sample = samples[(usize)(first_spike_tick + offset)];
        EXPECT_GT(sample.after_spike_current_one, previous.after_spike_current_one)
                << "tick " << sample.tick << ": asc1 stopped decaying during the refractory "
                << "period, but its TimeDerivative sits outside both regimes";
        EXPECT_GT(sample.after_spike_current_two, previous.after_spike_current_two)
                << "tick " << sample.tick << ": asc2 stopped decaying during the refractory "
                << "period, but its TimeDerivative sits outside both regimes";
    }

    // asc2 decays ten times faster than asc1, so it has to have recovered further.
    const Glif3Sample &window_end = samples[(usize)(first_spike_tick + refractory_tick_count)];
    EXPECT_LT(std::fabs(window_end.after_spike_current_two - at_spike.after_spike_current_two) /
                      std::fabs(at_spike.after_spike_current_two),
              1.0f);
    EXPECT_GT(std::fabs(window_end.after_spike_current_two - at_spike.after_spike_current_two) /
                      std::fabs(at_spike.after_spike_current_two),
              std::fabs(window_end.after_spike_current_one - at_spike.after_spike_current_one) /
                      std::fabs(at_spike.after_spike_current_one));

    engine.shutdown();
}

TEST(SpikeEngine, a_regime_scoped_on_condition_does_not_fire_while_its_regime_is_inactive) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // `counting`'s OnCondition is `t .geq. 0`, true on every tick of the run, and its body
    // increments an observable counter. The cell starts in `rising`, so the counter must be
    // untouched until the transition -- and the regime guard on the condition is the only
    // thing that makes that so.
    FixtureDirectory fixture("neuroml_regime_guard");
    String model_path = write_regime_guard_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    const f32 *cell_state = engine.cell_state.get_contents();
    const s64 state_base = engine.cell_state_base.get_contents()[0];
    const s64 level_index = state_base + 0;
    const s64 counter_index = state_base + 1;
    const s64 regime_index = state_base + 2;

    s64 transition_tick = -1;
    spikecorec::Vector<f32> counter_by_tick;
    for (s64 tick = 0; tick < engine.lifetime; ++tick) {
        engine.step_simulation(tick);
        if (transition_tick < 0 && (s32)cell_state[regime_index] == 1) transition_tick = tick;
        counter_by_tick.push_back(cell_state[counter_index]);
    }

    // level rises by risingRate * dt = 0.1 a tick, so it passes switchLevel = 1 well inside a
    // 30-tick run. Without a transition the rest of the test proves nothing.
    ASSERT_GE(transition_tick, 0) << "the cell never left its initial regime";
    ASSERT_LT(transition_tick, engine.lifetime - 2);
    EXPECT_GT(cell_state[level_index], 1.0f);

    // Nothing fired while `counting` was inactive -- including on the transition tick itself,
    // where the regime index was read before the Transition stored it.
    for (s64 tick = 0; tick <= transition_tick; ++tick) {
        EXPECT_FLOAT_EQ(counter_by_tick[(usize)tick], 0.0f)
                << "tick " << tick << ": the `counting` regime's OnCondition fired while the "
                << "cell was still in `rising`";
    }

    // And once active it fires on every tick, so the guard is holding the condition off
    // rather than the condition simply never being true.
    for (s64 tick = transition_tick + 1; tick < engine.lifetime; ++tick) {
        EXPECT_FLOAT_EQ(counter_by_tick[(usize)tick], (f32)(tick - transition_tick))
                << "tick " << tick << ": the `counting` regime's OnCondition did not fire once "
                << "per tick after its regime became active";
    }

    engine.shutdown();
}

TEST(SpikeEngine, on_entry_runs_at_the_transition_and_not_on_later_ticks_in_that_regime) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    // `refractory`'s OnEntry zeroes refractoryTimeElapsed. Run once, at the transition, the
    // countdown then advances one dt per refractory tick and the regime ends after t_ref. Run
    // on every refractory tick instead, it would be reset to zero each time and the cell would
    // never leave -- so this is read directly off the countdown's trajectory.
    FixtureDirectory fixture("neuroml_on_entry_once");
    String model_path = write_glif3_refractory_model(fixture);

    SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

    const Glif3CellReader reader(engine, /*neuron_index=*/0);
    const s32 *spike_flags = engine.spike_flags.get_contents();
    const f32 step_dt = (f32)engine.network_details.step_dt;

    s64 first_spike_tick = -1;
    spikecorec::Vector<f32> countdown_after_spike;
    for (s64 tick = 0; tick < engine.lifetime; ++tick) {
        engine.step_simulation(tick);
        if (first_spike_tick < 0 && spike_flags[0] != 0) first_spike_tick = tick;
        if (first_spike_tick >= 0 && tick - first_spike_tick <= 10) {
            countdown_after_spike.push_back(reader.refractory_time_elapsed());
        }
    }
    ASSERT_GE(first_spike_tick, 0);
    ASSERT_EQ(countdown_after_spike.size(), 11u);

    // OnEntry ran, once, on the spike tick itself.
    EXPECT_FLOAT_EQ(countdown_after_spike[0], 0.0f)
            << "the target regime's OnEntry did not run at the transition";

    // And not again: the countdown climbs by exactly one dt per refractory tick.
    for (usize offset = 1; offset < countdown_after_spike.size(); ++offset) {
        EXPECT_NEAR(countdown_after_spike[offset], step_dt * (f32)offset, step_dt * 1e-3f)
                << "offset " << offset
                << " into the refractory window: the countdown is not advancing one dt per "
                << "tick, so OnEntry ran more than once";
    }

    engine.shutdown();
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
