#pragma once

#include <sstream>
#include <string>

#include "example_support.h"

// ── The GLIF family, on a wraparound grid ────────────────────────────────────────────────
//
// One generator shared by the five glif*_torus_network_example programs. Each of them picks
// a variant, hands it a side length and a synaptic weight, and gets back a complete
// NeuroML/LEMS document: side^2 cells on a torus, each wired to its four neighbours, one
// corner driven by a current step, and the whole run recorded to a pair of .spire files.
//
// Generated rather than checked in because an 8x8 torus has 256 connections, and hand-typing
// that five times over is not reasonable. The generated document goes through the identical
// parse path a checked-in one does.
//
// ── the refractory period, and why there is no <Regime> here ─────────────────────────────
//
// The canonical GLIF declarations (tests/fixtures/nml/glif*_single_cell.nml) put the
// refractory period in a <Regime>/<Transition> pair: an `integrating` regime that owns the
// membrane TimeDerivative, and a `refractory` regime it transitions into on a spike. The
// engine's codegen does not support <Regime> yet -- it rejects those models by name -- so
// the declarations below express the SAME refractory behaviour without one:
//
//     <StateVariable name="refractoryTimeRemaining" dimension="time"/>
//     <DerivedVariable name="integrating" value="H(0 - refractoryTimeRemaining)"/>
//     <TimeDerivative variable="refractoryTimeRemaining" value="-1"/>
//     <TimeDerivative variable="v" value="integrating * (...) / C"/>
//
// A spike sets refractoryTimeRemaining to tRef; it counts down at one second per second; and
// `integrating` -- LEMS's Heaviside step, H(x) being 1 for x >= 0 -- gates the membrane
// derivative to zero for exactly as long as it is positive. v therefore holds at vreset
// through the refractory period and resumes integrating after it, which is what the regime
// pair does. This is a real refractory period, not an omission.

namespace spikecorec::examples {

enum class GlifVariant { Glif1, Glif2, Glif3, Glif4, Glif5 };

inline const char *glif_variant_type_name(GlifVariant variant) {
    switch (variant) {
        case GlifVariant::Glif1: return "GLIF1Cell";
        case GlifVariant::Glif2: return "GLIF2Cell";
        case GlifVariant::Glif3: return "GLIF3Cell";
        case GlifVariant::Glif4: return "GLIF4Cell";
        case GlifVariant::Glif5: return "GLIF5Cell";
    }
    return "GLIF1Cell";
}

// What the variant adds to plain leaky integrate-and-fire, for the run's own header line.
inline const char *glif_variant_summary(GlifVariant variant) {
    switch (variant) {
        case GlifVariant::Glif1:
            return "leaky integrate-and-fire: one state variable, a fixed threshold, a flat "
                   "reset";
        case GlifVariant::Glif2:
            return "GLIF1 plus a reset that scales with the threshold overshoot";
        case GlifVariant::Glif3:
            return "GLIF1 plus two after-spike currents (spike-frequency adaptation)";
        case GlifVariant::Glif4:
            return "GLIF1 plus an adaptive threshold";
        case GlifVariant::Glif5:
            return "GLIF3 plus GLIF4: after-spike currents AND an adaptive threshold";
    }
    return "";
}

// ── ComponentType declarations ───────────────────────────────────────────────────────────

inline const char *glif1_component_type() {
    return R"XML(
    <ComponentType name="GLIF1Cell" extends="baseCell"
                   description="Leaky integrate-and-fire with a fixed threshold and a flat
                                reset, gated by a refractory countdown.">
        <Parameter name="C" dimension="capacitance"/>
        <Parameter name="gL" dimension="conductance"/>
        <Parameter name="EL" dimension="voltage"/>
        <Parameter name="vth" dimension="voltage"/>
        <Parameter name="vreset" dimension="voltage"/>
        <Parameter name="tRef" dimension="time"/>
        <Parameter name="synapticCurrentScale" dimension="current"/>
        <Attachments name="synapses" type="basePointCurrent"/>
        <EventPort name="spike" direction="out"/>
        <Exposure name="v" dimension="voltage"/>

        <Dynamics>
            <StateVariable name="v" dimension="voltage" exposure="v"/>
            <StateVariable name="refractoryTimeRemaining" dimension="time"/>

            <DerivedVariable name="synapticWeightSum" dimension="none" select="synapses[*]/i"
                             reduce="add"/>
            <DerivedVariable name="iSyn" dimension="current"
                             value="synapticWeightSum * synapticCurrentScale"/>
            <DerivedVariable name="integrating" dimension="none"
                             value="H(0 - refractoryTimeRemaining)"/>

            <TimeDerivative variable="refractoryTimeRemaining" value="-1"/>
            <TimeDerivative variable="v" value="integrating * (gL * (EL - v) + iSyn) / C"/>

            <OnStart>
                <StateAssignment variable="v" value="EL"/>
                <StateAssignment variable="refractoryTimeRemaining" value="0"/>
            </OnStart>

            <OnCondition test="v .gt. vth">
                <StateAssignment variable="v" value="vreset"/>
                <StateAssignment variable="refractoryTimeRemaining" value="tRef"/>
                <EventOut port="spike"/>
            </OnCondition>
        </Dynamics>
    </ComponentType>
)XML";
}

inline const char *glif2_component_type() {
    return R"XML(
    <ComponentType name="GLIF2Cell" extends="baseCell"
                   description="GLIF1 with a reset that scales with how far past vth the
                                membrane potential overshot on the triggering tick.">
        <Parameter name="C" dimension="capacitance"/>
        <Parameter name="gL" dimension="conductance"/>
        <Parameter name="EL" dimension="voltage"/>
        <Parameter name="vth" dimension="voltage"/>
        <Parameter name="vreset" dimension="voltage"/>
        <Parameter name="tRef" dimension="time"/>
        <Parameter name="resetScale" dimension="none"/>
        <Parameter name="synapticCurrentScale" dimension="current"/>
        <Attachments name="synapses" type="basePointCurrent"/>
        <EventPort name="spike" direction="out"/>
        <Exposure name="v" dimension="voltage"/>

        <Dynamics>
            <StateVariable name="v" dimension="voltage" exposure="v"/>
            <StateVariable name="refractoryTimeRemaining" dimension="time"/>

            <DerivedVariable name="synapticWeightSum" dimension="none" select="synapses[*]/i"
                             reduce="add"/>
            <DerivedVariable name="iSyn" dimension="current"
                             value="synapticWeightSum * synapticCurrentScale"/>
            <DerivedVariable name="integrating" dimension="none"
                             value="H(0 - refractoryTimeRemaining)"/>

            <TimeDerivative variable="refractoryTimeRemaining" value="-1"/>
            <TimeDerivative variable="v" value="integrating * (gL * (EL - v) + iSyn) / C"/>

            <OnStart>
                <StateAssignment variable="v" value="EL"/>
                <StateAssignment variable="refractoryTimeRemaining" value="0"/>
            </OnStart>

            <OnCondition test="v .gt. vth">
                <StateAssignment variable="v" value="vreset + resetScale * (v - vth)"/>
                <StateAssignment variable="refractoryTimeRemaining" value="tRef"/>
                <EventOut port="spike"/>
            </OnCondition>
        </Dynamics>
    </ComponentType>
)XML";
}

inline const char *glif3_component_type() {
    return R"XML(
    <ComponentType name="GLIF3Cell" extends="baseCell"
                   description="GLIF1 plus two after-spike currents, each stepping on every
                                spike and decaying back towards zero.">
        <Parameter name="C" dimension="capacitance"/>
        <Parameter name="gL" dimension="conductance"/>
        <Parameter name="EL" dimension="voltage"/>
        <Parameter name="vth" dimension="voltage"/>
        <Parameter name="vreset" dimension="voltage"/>
        <Parameter name="tRef" dimension="time"/>
        <Parameter name="tauAsc1" dimension="time"/>
        <Parameter name="tauAsc2" dimension="time"/>
        <Parameter name="ascAdd1" dimension="current"/>
        <Parameter name="ascAdd2" dimension="current"/>
        <Parameter name="synapticCurrentScale" dimension="current"/>
        <Attachments name="synapses" type="basePointCurrent"/>
        <EventPort name="spike" direction="out"/>
        <Exposure name="v" dimension="voltage"/>

        <Dynamics>
            <StateVariable name="v" dimension="voltage" exposure="v"/>
            <StateVariable name="asc1" dimension="current"/>
            <StateVariable name="asc2" dimension="current"/>
            <StateVariable name="refractoryTimeRemaining" dimension="time"/>

            <DerivedVariable name="synapticWeightSum" dimension="none" select="synapses[*]/i"
                             reduce="add"/>
            <DerivedVariable name="iSyn" dimension="current"
                             value="synapticWeightSum * synapticCurrentScale"/>
            <DerivedVariable name="ascSum" dimension="current" value="asc1 + asc2"/>
            <DerivedVariable name="integrating" dimension="none"
                             value="H(0 - refractoryTimeRemaining)"/>

            <TimeDerivative variable="refractoryTimeRemaining" value="-1"/>
            <TimeDerivative variable="asc1" value="-asc1 / tauAsc1"/>
            <TimeDerivative variable="asc2" value="-asc2 / tauAsc2"/>
            <TimeDerivative variable="v"
                            value="integrating * (gL * (EL - v) + iSyn + ascSum) / C"/>

            <OnStart>
                <StateAssignment variable="v" value="EL"/>
                <StateAssignment variable="asc1" value="0"/>
                <StateAssignment variable="asc2" value="0"/>
                <StateAssignment variable="refractoryTimeRemaining" value="0"/>
            </OnStart>

            <OnCondition test="v .gt. vth">
                <StateAssignment variable="v" value="vreset"/>
                <StateAssignment variable="asc1" value="asc1 + ascAdd1"/>
                <StateAssignment variable="asc2" value="asc2 + ascAdd2"/>
                <StateAssignment variable="refractoryTimeRemaining" value="tRef"/>
                <EventOut port="spike"/>
            </OnCondition>
        </Dynamics>
    </ComponentType>
)XML";
}

inline const char *glif4_component_type() {
    return R"XML(
    <ComponentType name="GLIF4Cell" extends="baseCell"
                   description="GLIF1 with an adaptive threshold: theta is a state variable
                                that jumps on every spike and relaxes towards thetaInf.">
        <Parameter name="C" dimension="capacitance"/>
        <Parameter name="gL" dimension="conductance"/>
        <Parameter name="EL" dimension="voltage"/>
        <Parameter name="vreset" dimension="voltage"/>
        <Parameter name="tRef" dimension="time"/>
        <Parameter name="thetaInf" dimension="voltage"/>
        <Parameter name="tauTheta" dimension="time"/>
        <Parameter name="thetaSpikeAdd" dimension="voltage"/>
        <Parameter name="synapticCurrentScale" dimension="current"/>
        <Attachments name="synapses" type="basePointCurrent"/>
        <EventPort name="spike" direction="out"/>
        <Exposure name="v" dimension="voltage"/>

        <Dynamics>
            <StateVariable name="v" dimension="voltage" exposure="v"/>
            <StateVariable name="theta" dimension="voltage"/>
            <StateVariable name="refractoryTimeRemaining" dimension="time"/>

            <DerivedVariable name="synapticWeightSum" dimension="none" select="synapses[*]/i"
                             reduce="add"/>
            <DerivedVariable name="iSyn" dimension="current"
                             value="synapticWeightSum * synapticCurrentScale"/>
            <DerivedVariable name="integrating" dimension="none"
                             value="H(0 - refractoryTimeRemaining)"/>

            <TimeDerivative variable="refractoryTimeRemaining" value="-1"/>
            <TimeDerivative variable="theta" value="(thetaInf - theta) / tauTheta"/>
            <TimeDerivative variable="v" value="integrating * (gL * (EL - v) + iSyn) / C"/>

            <OnStart>
                <StateAssignment variable="v" value="EL"/>
                <StateAssignment variable="theta" value="thetaInf"/>
                <StateAssignment variable="refractoryTimeRemaining" value="0"/>
            </OnStart>

            <OnCondition test="v .gt. theta">
                <StateAssignment variable="v" value="vreset"/>
                <StateAssignment variable="theta" value="theta + thetaSpikeAdd"/>
                <StateAssignment variable="refractoryTimeRemaining" value="tRef"/>
                <EventOut port="spike"/>
            </OnCondition>
        </Dynamics>
    </ComponentType>
)XML";
}

inline const char *glif5_component_type() {
    return R"XML(
    <ComponentType name="GLIF5Cell" extends="baseCell"
                   description="GLIF3 plus GLIF4: two after-spike currents AND an adaptive
                                threshold.">
        <Parameter name="C" dimension="capacitance"/>
        <Parameter name="gL" dimension="conductance"/>
        <Parameter name="EL" dimension="voltage"/>
        <Parameter name="vreset" dimension="voltage"/>
        <Parameter name="tRef" dimension="time"/>
        <Parameter name="tauAsc1" dimension="time"/>
        <Parameter name="tauAsc2" dimension="time"/>
        <Parameter name="ascAdd1" dimension="current"/>
        <Parameter name="ascAdd2" dimension="current"/>
        <Parameter name="thetaInf" dimension="voltage"/>
        <Parameter name="tauTheta" dimension="time"/>
        <Parameter name="thetaSpikeAdd" dimension="voltage"/>
        <Parameter name="synapticCurrentScale" dimension="current"/>
        <Attachments name="synapses" type="basePointCurrent"/>
        <EventPort name="spike" direction="out"/>
        <Exposure name="v" dimension="voltage"/>

        <Dynamics>
            <StateVariable name="v" dimension="voltage" exposure="v"/>
            <StateVariable name="theta" dimension="voltage"/>
            <StateVariable name="asc1" dimension="current"/>
            <StateVariable name="asc2" dimension="current"/>
            <StateVariable name="refractoryTimeRemaining" dimension="time"/>

            <DerivedVariable name="synapticWeightSum" dimension="none" select="synapses[*]/i"
                             reduce="add"/>
            <DerivedVariable name="iSyn" dimension="current"
                             value="synapticWeightSum * synapticCurrentScale"/>
            <DerivedVariable name="ascSum" dimension="current" value="asc1 + asc2"/>
            <DerivedVariable name="integrating" dimension="none"
                             value="H(0 - refractoryTimeRemaining)"/>

            <TimeDerivative variable="refractoryTimeRemaining" value="-1"/>
            <TimeDerivative variable="theta" value="(thetaInf - theta) / tauTheta"/>
            <TimeDerivative variable="asc1" value="-asc1 / tauAsc1"/>
            <TimeDerivative variable="asc2" value="-asc2 / tauAsc2"/>
            <TimeDerivative variable="v"
                            value="integrating * (gL * (EL - v) + iSyn + ascSum) / C"/>

            <OnStart>
                <StateAssignment variable="v" value="EL"/>
                <StateAssignment variable="theta" value="thetaInf"/>
                <StateAssignment variable="asc1" value="0"/>
                <StateAssignment variable="asc2" value="0"/>
                <StateAssignment variable="refractoryTimeRemaining" value="0"/>
            </OnStart>

            <OnCondition test="v .gt. theta">
                <StateAssignment variable="v" value="vreset"/>
                <StateAssignment variable="theta" value="theta + thetaSpikeAdd"/>
                <StateAssignment variable="asc1" value="asc1 + ascAdd1"/>
                <StateAssignment variable="asc2" value="asc2 + ascAdd2"/>
                <StateAssignment variable="refractoryTimeRemaining" value="tRef"/>
                <EventOut port="spike"/>
            </OnCondition>
        </Dynamics>
    </ComponentType>
)XML";
}

inline const char *glif_component_type_declaration(GlifVariant variant) {
    switch (variant) {
        case GlifVariant::Glif1: return glif1_component_type();
        case GlifVariant::Glif2: return glif2_component_type();
        case GlifVariant::Glif3: return glif3_component_type();
        case GlifVariant::Glif4: return glif4_component_type();
        case GlifVariant::Glif5: return glif5_component_type();
    }
    return glif1_component_type();
}

// The one prototype every neuron in the torus instantiates. Values are the same membrane
// across all five variants -- 100pF / 10nS / -70mV rest / -50mV threshold / 5ms refractory --
// so a difference between two runs is the variant's own mechanism and nothing else.
//
// synapticCurrentScale is what turns the engine's synaptic accumulator into a current, and
// it is set to 1nA -- so every number that lands in that accumulator, from a connection or
// from the injected stimulus alike, reads as nanoamps.
//
// That indirection is not cosmetic. A NeuroML <connectionWD weight=".."> is DIMENSIONLESS, a
// multiplier on the synapse's own contribution, and the engine delivers that multiplier
// itself (see the projection comment in torus_network_nml). It has to be numerically
// well-conditioned as well: an edge weight is stored as a delta against the WeightMatrix's
// random-initialised U/V reconstruction, whose entries are order 1, so a weight far below
// f32 epsilon relative to that -- 2.5e-8, say -- cancels away to zero when it is read back.
// Working in nanoamps keeps every stored weight order 1..100, where it round-trips exactly.
inline std::string glif_cell_instance(GlifVariant variant, const std::string &instance_id) {
    const std::string membrane =
            R"( C="100pF" gL="10nS" EL="-70mV" vreset="-70mV" tRef="5ms" synapticCurrentScale="1nA")";
    const std::string fixed_threshold = R"( vth="-50mV")";
    const std::string after_spike_currents =
            R"( tauAsc1="100ms" tauAsc2="10ms" ascAdd1="-100pA" ascAdd2="-200pA")";
    const std::string adaptive_threshold =
            R"( thetaInf="-50mV" tauTheta="50ms" thetaSpikeAdd="3mV")";

    std::string attributes = membrane;
    switch (variant) {
        case GlifVariant::Glif1:
            attributes += fixed_threshold;
            break;
        case GlifVariant::Glif2:
            attributes += fixed_threshold + R"( resetScale="0.4")";
            break;
        case GlifVariant::Glif3:
            attributes += fixed_threshold + after_spike_currents;
            break;
        case GlifVariant::Glif4:
            attributes += adaptive_threshold;
            break;
        case GlifVariant::Glif5:
            attributes += after_spike_currents + adaptive_threshold;
            break;
    }

    return std::string("<") + glif_variant_type_name(variant) + " id=\"" + instance_id + "\"" +
           attributes + "/>";
}

// ── the network ──────────────────────────────────────────────────────────────────────────

struct TorusNetworkOptions {
    s64 side_length = 8;

    // The <connectionWD weight="..."> every lateral connection carries. The cell reads the
    // accumulator in nanoamps (synapticCurrentScale="1nA"), so this is nanoamps too: 25 is a
    // 25nA impulse, which over one 0.1ms tick moves a 100pF membrane by 25mV -- just past the
    // 20mV a resting cell needs to reach threshold, so one arriving spike recruits its
    // target. Halve it and a cell needs two neighbours firing together instead.
    f64 connection_weight = 25.0;

    // Conduction delay on every lateral connection. Long enough that one hop is visible as
    // its own step in the first-spike grid rather than the whole torus lighting up at once.
    std::string connection_delay = "1ms";

    // With this off no <projection> is emitted at all, which isolates a cell from its
    // neighbours -- what discrete_spike_input_example wants, so its bit-to-spike mapping is
    // unambiguous.
    bool include_lateral_connections = true;

    std::string simulation_length = "200ms";
    std::string simulation_step = "0.1ms";

    // Absolute paths. Left empty to record nothing.
    std::string membrane_recording_path;
    std::string spike_recording_path;
};

inline s64 torus_neuron_index(s64 row, s64 column, s64 side_length) {
    return row * side_length + column;
}

// Every cell wired to the four neighbours it has on a wraparound grid. The flat index is
// `row * side_length + column`, which is also what render_spire_video.py's --side reshapes a
// recorded frame back into.
inline std::string torus_network_nml(GlifVariant variant, const TorusNetworkOptions &options) {
    const s64 side_length = options.side_length;
    const s64 neuron_count = side_length * side_length;

    std::ostringstream document;
    document << "<neuroml id=\"glifTorusNetwork\">\n    "
             << glif_cell_instance(variant, "torusCell") << "\n"
             << "    <pulseGenerator id=\"cornerDrive\" delay=\"20ms\" duration=\"1000ms\" "
                "amplitude=\"0.6\"/>\n";

    if (options.include_lateral_connections) {
        // NeuroML requires a <projection> to name a synapse instance, so one is declared.
        // Its OWN dynamics are not run: the engine does not lower synapse ComponentTypes
        // yet, so each arriving spike delivers the connection's dimensionless `weight`
        // straight into the target's synaptic accumulator, with no exponential decay shaping
        // it. gbase/erev/tauDecay below are therefore inert, and `weight` (times the target's
        // own synapticCurrentScale) is the whole magnitude. See examples/README.md.
        document << "    <expOneSynapse id=\"lateralSynapse\" gbase=\"10nS\" erev=\"0mV\" "
                    "tauDecay=\"3ms\"/>\n";
    }

    document << "\n    <network id=\"torusNet\">\n"
             << "        <population id=\"torusPop\" component=\"torusCell\" size=\"" << neuron_count
             << "\"/>\n";

    if (options.include_lateral_connections) {
        document << "\n        <projection id=\"lateral\" presynapticPopulation=\"torusPop\"\n"
                 << "                    postsynapticPopulation=\"torusPop\" "
                    "synapse=\"lateralSynapse\">\n";

        const s64 row_offsets[4] = {-1, 1, 0, 0};
        const s64 column_offsets[4] = {0, 0, -1, 1};

        s64 connection_id = 0;
        for (s64 row = 0; row < side_length; ++row) {
            for (s64 column = 0; column < side_length; ++column) {
                for (s64 direction = 0; direction < 4; ++direction) {
                    const s64 neighbor_row =
                            (row + row_offsets[direction] + side_length) % side_length;
                    const s64 neighbor_column =
                            (column + column_offsets[direction] + side_length) % side_length;

                    document << "            <connectionWD id=\"" << connection_id++
                             << "\" preCellId=\"../torusPop["
                             << torus_neuron_index(row, column, side_length)
                             << "]\" postCellId=\"../torusPop["
                             << torus_neuron_index(neighbor_row, neighbor_column, side_length)
                             << "]\"\n                          weight=\""
                             << options.connection_weight << "\" delay=\""
                             << options.connection_delay << "\"/>\n";
                }
            }
        }

        document << "        </projection>\n";
    }

    document << "\n        <explicitInput target=\"torusPop[0]\" input=\"cornerDrive\"/>\n"
             << "    </network>\n</neuroml>\n";

    return document.str();
}

// The LEMS document the engine is actually handed: the variant's ComponentType, the network
// file, and the <Simulation> that fixes dt, the run length and the recordings.
//
// A <Lems> root rather than a <neuroml> one because the ComponentType declarations are raw
// LEMS: NML_Parser XSD-validates a top-level <neuroml> document, and a raw ComponentType does
// not validate against the NeuroML schema. This is the same shape tests/engine_tests.cpp's
// own fixtures use.
inline std::string torus_lems_document(GlifVariant variant, const TorusNetworkOptions &options,
                                       const std::string &network_file_name) {
    std::ostringstream document;
    document << "<Lems>\n    <Target component=\"torusSimulation\"/>\n"
             << glif_component_type_declaration(variant) << "\n    <Include file=\""
             << network_file_name << "\"/>\n\n"
             << "    <Simulation id=\"torusSimulation\" length=\"" << options.simulation_length
             << "\" step=\"" << options.simulation_step << "\" target=\"torusNet\">\n";

    if (!options.membrane_recording_path.empty()) {
        document << "        <OutputFile id=\"membrane\" fileName=\""
                 << options.membrane_recording_path << "\"/>\n";
    }
    if (!options.spike_recording_path.empty()) {
        document << "        <EventOutputFile id=\"spikes\" fileName=\""
                 << options.spike_recording_path << "\" format=\"TIME_ID\"/>\n";
    }

    document << "    </Simulation>\n</Lems>\n";
    return document.str();
}

// Writes both files and returns the LEMS main file's path, which is what SpikeEngine takes.
inline std::string write_torus_model(const GeneratedModelDirectory &model_directory,
                                     GlifVariant variant, const TorusNetworkOptions &options) {
    model_directory.write("torus_network.nml", torus_network_nml(variant, options));
    return model_directory.write("torus_model.xml",
                                 torus_lems_document(variant, options, "torus_network.nml"));
}

// ── the first-spike grid ─────────────────────────────────────────────────────────────────

// Each neuron drawn at its real (row, column) position, shaded by how early it first fired.
// A wavefront spreading from the driven corner in Manhattan distance and meeting itself at
// the antipode is the torus wraparound made visible.
inline void print_first_spike_grid(const SpikeObservation &observation, s64 side_length) {
    s64 earliest_tick = -1;
    s64 latest_tick = -1;
    for (s64 first_tick : observation.first_spike_tick_per_neuron) {
        if (first_tick < 0) continue;
        if (earliest_tick < 0 || first_tick < earliest_tick) earliest_tick = first_tick;
        if (first_tick > latest_tick) latest_tick = first_tick;
    }

    if (earliest_tick < 0) {
        std::cout << "  (no neuron fired -- nothing to plot)\n";
        return;
    }

    std::cout << "  First spike tick   (range " << earliest_tick << " ... " << latest_tick
              << "; '.' = never fired)\n";

    static const char shades[] = "-=+*#%@";
    const s64 shade_count = (s64)sizeof(shades) - 1;
    const s64 tick_span = latest_tick - earliest_tick;

    for (s64 row = 0; row < side_length; ++row) {
        std::cout << "    ";
        for (s64 column = 0; column < side_length; ++column) {
            const s64 first_tick = observation.first_spike_tick_per_neuron[(
                    usize)torus_neuron_index(row, column, side_length)];
            if (first_tick < 0) {
                std::cout << ". ";
                continue;
            }

            // Earliest is the darkest mark, latest the lightest.
            const s64 shade_index =
                    tick_span == 0 ? shade_count - 1
                                   : (shade_count - 1) -
                                             ((first_tick - earliest_tick) * (shade_count - 1)) /
                                                     tick_span;
            std::cout << shades[shade_index] << " ";
        }
        std::cout << "\n";
    }
}

// The shared opening every torus example prints, so the differences between two runs are
// visible at the top.
inline void print_torus_run_header(GlifVariant variant, const TorusNetworkOptions &options) {
    print_heading(std::string(glif_variant_type_name(variant)) + " on a " +
                  std::to_string(options.side_length) + "x" +
                  std::to_string(options.side_length) + " torus");

    std::cout << "  " << glif_variant_summary(variant) << "\n"
              << "  " << options.side_length * options.side_length << " neurons, "
              << (options.include_lateral_connections ? options.side_length * options.side_length * 4
                                                      : 0)
              << " connections, corner neuron 0 driven by a 0.6nA step\n"
              << "  connection weight " << options.connection_weight << " nA, delay "
              << options.connection_delay << "\n";
}

} // namespace spikecorec::examples
