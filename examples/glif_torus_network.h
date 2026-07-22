#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "spikecorec/core/topologies.h"

#include "nml_pipeline_support.h"

// ── Torus-connected GLIF networks, generated as NeuroML ─────────────────────────────────────────
//
// The checked-in fixtures under tests/fixtures/nml/ are small by design — they exist so the SAME
// file can drive both spikecorec and the reference simulator, which means every population and
// connection is hand-typed. That does not scale: a 64-neuron torus has 256 connections.
//
// So this header generates the NeuroML instead. `square_torus(side_length)` (core/topologies.h)
// gives a 4-neighbor wraparound grid — every neuron connects to its right, left, down, and up
// neighbors, with edges wrapping — and each of those edges becomes a `<connection>` element in a
// generated document that goes through the exact same parse → resolve → lower path as any
// hand-written file. Nothing about the pipeline is special-cased for generated input.
//
// The GLIF ComponentType blocks below are reused verbatim (same equations, same Parameter and
// dimension declarations) from tests/cell_lowering_tests.cpp's GLIF fixtures, so a generated network
// runs the same dynamics the lowering tests already cover.
//
// A generated document deliberately omits the `xsi`/`schemaLocation` attributes and the
// `<Simulation>`/`<OutputFile>` blocks the checked-in fixtures carry — those exist purely so the
// same file also drives jLEMS. These networks are spikecorec-only, so only what the front-end
// actually needs is emitted.
//
// A generated neuron's flat index is always `row * side_length + column` — print_torus_grid below
// (and examples/render_spire_video.py, which never sees this header, only a model's total neuron
// count and side length) both rely on that exact convention to place a neuron back on the grid.

namespace spikecorec::examples {

// The GLIF variants this builder emits. GLIF1 is plain leaky integrate-and-fire; GLIF2 adds a
// biologically realistic reset that scales with how far past threshold `v` overshot; GLIF3 adds two
// after-spike currents to a plain LIF; GLIF4 adds an adaptive threshold instead; GLIF5 combines
// GLIF3's after-spike currents with GLIF4's adaptive threshold. See each ComponentType below.
enum class GlifVariant { Glif1 = 1, Glif2 = 2, Glif3 = 3, Glif4 = 4, Glif5 = 5 };

// ── ComponentType declarations ──────────────────────────────────────────────────────────────────
//
// Every GLIF1/GLIF2/GLIF4 block below is reused verbatim (same equations, same Parameter/dimension
// declarations, same declared StateVariable order) from tests/cell_lowering_tests.cpp's own
// GLIF1_COMPONENT_TYPE/GLIF2_COMPONENT_TYPE/GLIF4_COMPONENT_TYPE fixtures (ticket #50), the same
// precedent GLIF3/GLIF5 above already followed.

// GLIF1: plain leaky integrate-and-fire against a FIXED threshold `vth`, with a flat reset to
// `vreset` and a fixed refractory period.
inline const String GLIF1_COMPONENT_TYPE_XML =
    "  <ComponentType name=\"GLIF1Cell\" extends=\"baseCell\">"
    "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
    "    <Parameter name=\"gL\" dimension=\"conductance\"/>"
    "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
    "    <Parameter name=\"vth\" dimension=\"voltage\"/>"
    "    <Parameter name=\"vreset\" dimension=\"voltage\"/>"
    "    <Parameter name=\"t_ref\" dimension=\"time\"/>"
    "    <Attachments name=\"synapses\" type=\"basePointCurrent\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
    "      <StateVariable name=\"refractoryTimeElapsed\" dimension=\"time\"/>"
    "      <DerivedVariable name=\"iSyn\" dimension=\"current\" exposure=\"iSyn\" select=\"synapses[*]/i\" reduce=\"add\"/>"
    "      <OnStart>"
    "        <StateAssignment variable=\"v\" value=\"EL\"/>"
    "        <StateAssignment variable=\"refractoryTimeElapsed\" value=\"0\"/>"
    "      </OnStart>"
    "      <Regime name=\"integrating\" initial=\"true\">"
    "        <TimeDerivative variable=\"v\" value=\"(gL * (EL - v) + iSyn) / C\"/>"
    "        <OnCondition test=\"v .gt. vth\">"
    "          <EventOut port=\"spike\"/>"
    "          <StateAssignment variable=\"v\" value=\"vreset\"/>"
    "          <Transition regime=\"refractory\"/>"
    "        </OnCondition>"
    "      </Regime>"
    "      <Regime name=\"refractory\">"
    "        <OnEntry>"
    "          <StateAssignment variable=\"refractoryTimeElapsed\" value=\"0\"/>"
    "        </OnEntry>"
    "        <TimeDerivative variable=\"refractoryTimeElapsed\" value=\"1\"/>"
    "        <OnCondition test=\"refractoryTimeElapsed .geq. t_ref\">"
    "          <Transition regime=\"integrating\"/>"
    "        </OnCondition>"
    "      </Regime>"
    "    </Dynamics>"
    "  </ComponentType>";

// GLIF2: GLIF1 plus a reset rule that scales with how far past `vth` the membrane potential
// overshot on the triggering tick — `v <- vreset + resetScale * (v - vth)` rather than a flat
// `v <- vreset`. `resetScale=0` degenerates to GLIF1's exact flat reset; a nonzero value lands
// measurably above `vreset`, a more biologically realistic reset rule.
inline const String GLIF2_COMPONENT_TYPE_XML =
    "  <ComponentType name=\"GLIF2Cell\" extends=\"baseCell\">"
    "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
    "    <Parameter name=\"gL\" dimension=\"conductance\"/>"
    "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
    "    <Parameter name=\"vth\" dimension=\"voltage\"/>"
    "    <Parameter name=\"vreset\" dimension=\"voltage\"/>"
    "    <Parameter name=\"resetScale\" dimension=\"none\"/>"
    "    <Parameter name=\"t_ref\" dimension=\"time\"/>"
    "    <Attachments name=\"synapses\" type=\"basePointCurrent\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
    "      <StateVariable name=\"refractoryTimeElapsed\" dimension=\"time\"/>"
    "      <DerivedVariable name=\"iSyn\" dimension=\"current\" exposure=\"iSyn\" select=\"synapses[*]/i\" reduce=\"add\"/>"
    "      <OnStart>"
    "        <StateAssignment variable=\"v\" value=\"EL\"/>"
    "      </OnStart>"
    "      <Regime name=\"integrating\" initial=\"true\">"
    "        <TimeDerivative variable=\"v\" value=\"(gL * (EL - v) + iSyn) / C\"/>"
    "        <OnCondition test=\"v .gt. vth\">"
    "          <EventOut port=\"spike\"/>"
    "          <StateAssignment variable=\"v\" value=\"vreset + resetScale * (v - vth)\"/>"
    "          <Transition regime=\"refractory\"/>"
    "        </OnCondition>"
    "      </Regime>"
    "      <Regime name=\"refractory\">"
    "        <OnEntry>"
    "          <StateAssignment variable=\"refractoryTimeElapsed\" value=\"0\"/>"
    "        </OnEntry>"
    "        <TimeDerivative variable=\"refractoryTimeElapsed\" value=\"1\"/>"
    "        <OnCondition test=\"refractoryTimeElapsed .geq. t_ref\">"
    "          <Transition regime=\"integrating\"/>"
    "        </OnCondition>"
    "      </Regime>"
    "    </Dynamics>"
    "  </ComponentType>";

// GLIF3: leaky integrate-and-fire plus two after-spike currents (`asc1`/`asc2`) that step up on every
// spike and decay between them, against a FIXED threshold `vth`.
inline const String GLIF3_COMPONENT_TYPE_XML =
    "  <ComponentType name=\"GLIF3Cell\" extends=\"baseCell\">"
    "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
    "    <Parameter name=\"gL\" dimension=\"conductance\"/>"
    "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
    "    <Parameter name=\"vth\" dimension=\"voltage\"/>"
    "    <Parameter name=\"vreset\" dimension=\"voltage\"/>"
    "    <Parameter name=\"t_ref\" dimension=\"time\"/>"
    "    <Parameter name=\"tauAsc1\" dimension=\"time\"/>"
    "    <Parameter name=\"tauAsc2\" dimension=\"time\"/>"
    "    <Parameter name=\"ascAdd1\" dimension=\"current\"/>"
    "    <Parameter name=\"ascAdd2\" dimension=\"current\"/>"
    "    <Attachments name=\"synapses\" type=\"basePointCurrent\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
    "      <StateVariable name=\"asc1\" dimension=\"current\" exposure=\"asc1\"/>"
    "      <StateVariable name=\"asc2\" dimension=\"current\" exposure=\"asc2\"/>"
    "      <StateVariable name=\"refractoryTimeElapsed\" dimension=\"time\"/>"
    "      <DerivedVariable name=\"iSyn\" dimension=\"current\" exposure=\"iSyn\" select=\"synapses[*]/i\" reduce=\"add\"/>"
    "      <DerivedVariable name=\"ascSum\" dimension=\"current\" value=\"asc1 + asc2\"/>"
    "      <TimeDerivative variable=\"asc1\" value=\"-asc1 / tauAsc1\"/>"
    "      <TimeDerivative variable=\"asc2\" value=\"-asc2 / tauAsc2\"/>"
    "      <OnStart>"
    "        <StateAssignment variable=\"v\" value=\"EL\"/>"
    "        <StateAssignment variable=\"asc1\" value=\"0\"/>"
    "        <StateAssignment variable=\"asc2\" value=\"0\"/>"
    "      </OnStart>"
    "      <Regime name=\"integrating\" initial=\"true\">"
    "        <TimeDerivative variable=\"v\" value=\"(gL * (EL - v) + iSyn + ascSum) / C\"/>"
    "        <OnCondition test=\"v .gt. vth\">"
    "          <EventOut port=\"spike\"/>"
    "          <StateAssignment variable=\"v\" value=\"vreset\"/>"
    "          <StateAssignment variable=\"asc1\" value=\"asc1 + ascAdd1\"/>"
    "          <StateAssignment variable=\"asc2\" value=\"asc2 + ascAdd2\"/>"
    "          <Transition regime=\"refractory\"/>"
    "        </OnCondition>"
    "      </Regime>"
    "      <Regime name=\"refractory\">"
    "        <OnEntry>"
    "          <StateAssignment variable=\"refractoryTimeElapsed\" value=\"0\"/>"
    "        </OnEntry>"
    "        <TimeDerivative variable=\"refractoryTimeElapsed\" value=\"1\"/>"
    "        <OnCondition test=\"refractoryTimeElapsed .geq. t_ref\">"
    "          <Transition regime=\"integrating\"/>"
    "        </OnCondition>"
    "      </Regime>"
    "    </Dynamics>"
    "  </ComponentType>";

// GLIF4: leaky integrate-and-fire plus an adaptive threshold instead of after-spike currents.
// `theta` is a state variable (not a constant) that relaxes toward `thetaInf` with time constant
// `tauTheta` and jumps by `thetaSpikeAdd` on every spike, so the firing condition is `v > theta`
// rather than `v > vth`.
inline const String GLIF4_COMPONENT_TYPE_XML =
    "  <ComponentType name=\"GLIF4Cell\" extends=\"baseCell\">"
    "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
    "    <Parameter name=\"gL\" dimension=\"conductance\"/>"
    "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
    "    <Parameter name=\"vreset\" dimension=\"voltage\"/>"
    "    <Parameter name=\"t_ref\" dimension=\"time\"/>"
    "    <Parameter name=\"thetaInf\" dimension=\"voltage\"/>"
    "    <Parameter name=\"tauTheta\" dimension=\"time\"/>"
    "    <Parameter name=\"thetaSpikeAdd\" dimension=\"voltage\"/>"
    "    <Attachments name=\"synapses\" type=\"basePointCurrent\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
    "      <StateVariable name=\"theta\" dimension=\"voltage\" exposure=\"theta\"/>"
    "      <StateVariable name=\"refractoryTimeElapsed\" dimension=\"time\"/>"
    "      <DerivedVariable name=\"iSyn\" dimension=\"current\" exposure=\"iSyn\" select=\"synapses[*]/i\" reduce=\"add\"/>"
    "      <TimeDerivative variable=\"theta\" value=\"(thetaInf - theta) / tauTheta\"/>"
    "      <OnStart>"
    "        <StateAssignment variable=\"v\" value=\"EL\"/>"
    "        <StateAssignment variable=\"theta\" value=\"thetaInf\"/>"
    "      </OnStart>"
    "      <Regime name=\"integrating\" initial=\"true\">"
    "        <TimeDerivative variable=\"v\" value=\"(gL * (EL - v) + iSyn) / C\"/>"
    "        <OnCondition test=\"v .gt. theta\">"
    "          <EventOut port=\"spike\"/>"
    "          <StateAssignment variable=\"v\" value=\"vreset\"/>"
    "          <StateAssignment variable=\"theta\" value=\"theta + thetaSpikeAdd\"/>"
    "          <Transition regime=\"refractory\"/>"
    "        </OnCondition>"
    "      </Regime>"
    "      <Regime name=\"refractory\">"
    "        <OnEntry>"
    "          <StateAssignment variable=\"refractoryTimeElapsed\" value=\"0\"/>"
    "        </OnEntry>"
    "        <TimeDerivative variable=\"refractoryTimeElapsed\" value=\"1\"/>"
    "        <OnCondition test=\"refractoryTimeElapsed .geq. t_ref\">"
    "          <Transition regime=\"integrating\"/>"
    "        </OnCondition>"
    "      </Regime>"
    "    </Dynamics>"
    "  </ComponentType>";

// GLIF5: GLIF3's two after-spike currents PLUS an adaptive threshold. `theta` is a state variable
// that relaxes toward `thetaInf` with time constant `tauTheta` and jumps by `thetaSpikeAdd` on every
// spike, so the firing condition is `v > theta` rather than `v > vth` — the threshold itself moves.
inline const String GLIF5_COMPONENT_TYPE_XML =
    "  <ComponentType name=\"GLIF5Cell\" extends=\"baseCell\">"
    "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
    "    <Parameter name=\"gL\" dimension=\"conductance\"/>"
    "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
    "    <Parameter name=\"vreset\" dimension=\"voltage\"/>"
    "    <Parameter name=\"t_ref\" dimension=\"time\"/>"
    "    <Parameter name=\"thetaInf\" dimension=\"voltage\"/>"
    "    <Parameter name=\"tauTheta\" dimension=\"time\"/>"
    "    <Parameter name=\"thetaSpikeAdd\" dimension=\"voltage\"/>"
    "    <Parameter name=\"tauAsc1\" dimension=\"time\"/>"
    "    <Parameter name=\"tauAsc2\" dimension=\"time\"/>"
    "    <Parameter name=\"ascAdd1\" dimension=\"current\"/>"
    "    <Parameter name=\"ascAdd2\" dimension=\"current\"/>"
    "    <Attachments name=\"synapses\" type=\"basePointCurrent\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
    "      <StateVariable name=\"theta\" dimension=\"voltage\" exposure=\"theta\"/>"
    "      <StateVariable name=\"asc1\" dimension=\"current\" exposure=\"asc1\"/>"
    "      <StateVariable name=\"asc2\" dimension=\"current\" exposure=\"asc2\"/>"
    "      <StateVariable name=\"refractoryTimeElapsed\" dimension=\"time\"/>"
    "      <DerivedVariable name=\"iSyn\" dimension=\"current\" exposure=\"iSyn\" select=\"synapses[*]/i\" reduce=\"add\"/>"
    "      <DerivedVariable name=\"ascSum\" dimension=\"current\" value=\"asc1 + asc2\"/>"
    "      <TimeDerivative variable=\"theta\" value=\"(thetaInf - theta) / tauTheta\"/>"
    "      <TimeDerivative variable=\"asc1\" value=\"-asc1 / tauAsc1\"/>"
    "      <TimeDerivative variable=\"asc2\" value=\"-asc2 / tauAsc2\"/>"
    "      <OnStart>"
    "        <StateAssignment variable=\"v\" value=\"EL\"/>"
    "        <StateAssignment variable=\"theta\" value=\"thetaInf\"/>"
    "        <StateAssignment variable=\"asc1\" value=\"0\"/>"
    "        <StateAssignment variable=\"asc2\" value=\"0\"/>"
    "      </OnStart>"
    "      <Regime name=\"integrating\" initial=\"true\">"
    "        <TimeDerivative variable=\"v\" value=\"(gL * (EL - v) + iSyn + ascSum) / C\"/>"
    "        <OnCondition test=\"v .gt. theta\">"
    "          <EventOut port=\"spike\"/>"
    "          <StateAssignment variable=\"v\" value=\"vreset\"/>"
    "          <StateAssignment variable=\"theta\" value=\"theta + thetaSpikeAdd\"/>"
    "          <StateAssignment variable=\"asc1\" value=\"asc1 + ascAdd1\"/>"
    "          <StateAssignment variable=\"asc2\" value=\"asc2 + ascAdd2\"/>"
    "          <Transition regime=\"refractory\"/>"
    "        </OnCondition>"
    "      </Regime>"
    "      <Regime name=\"refractory\">"
    "        <OnEntry>"
    "          <StateAssignment variable=\"refractoryTimeElapsed\" value=\"0\"/>"
    "        </OnEntry>"
    "        <TimeDerivative variable=\"refractoryTimeElapsed\" value=\"1\"/>"
    "        <OnCondition test=\"refractoryTimeElapsed .geq. t_ref\">"
    "          <Transition regime=\"integrating\"/>"
    "        </OnCondition>"
    "      </Regime>"
    "    </Dynamics>"
    "  </ComponentType>";

inline String glif_component_type_name(GlifVariant variant) {
    switch (variant) {
        case GlifVariant::Glif1: return "GLIF1Cell";
        case GlifVariant::Glif2: return "GLIF2Cell";
        case GlifVariant::Glif3: return "GLIF3Cell";
        case GlifVariant::Glif4: return "GLIF4Cell";
        case GlifVariant::Glif5: return "GLIF5Cell";
    }
    throw std::runtime_error("glif_component_type_name: unknown GlifVariant");
}

inline String glif_component_type_xml(GlifVariant variant) {
    switch (variant) {
        case GlifVariant::Glif1: return GLIF1_COMPONENT_TYPE_XML;
        case GlifVariant::Glif2: return GLIF2_COMPONENT_TYPE_XML;
        case GlifVariant::Glif3: return GLIF3_COMPONENT_TYPE_XML;
        case GlifVariant::Glif4: return GLIF4_COMPONENT_TYPE_XML;
        case GlifVariant::Glif5: return GLIF5_COMPONENT_TYPE_XML;
    }
    throw std::runtime_error("glif_component_type_xml: unknown GlifVariant");
}

// Parameter values for one bound cell instance. GLIF3/GLIF5 are verbatim from the real,
// jLEMS-verified parameter set in tests/fixtures/nml/glif3_single_cell.nml (and its GLIF5
// counterpart in the end-to-end tests); GLIF1/GLIF2/GLIF4 are verbatim from the parameter sets
// tests/end_to_end_network_tests.cpp's own anchor tests already validate
// (`glif1_ring_network_current_injection_smallest_anchor`,
// `glif2_ring_network_discrete_spike_array_smallest_anchor`,
// `glif4_ring_network_discrete_spike_array_medium_anchor`). GLIF2's `resetScale=0.4` is that same
// test's "scaled" instance — the more interesting of its two reset rules to demonstrate, since
// `resetScale=0` degenerates to GLIF1's exact flat reset.
inline String glif_cell_instance_attributes(GlifVariant variant) {
    switch (variant) {
        case GlifVariant::Glif1:
            return "C=\"100pF\" gL=\"10nS\" EL=\"-70mV\" vth=\"-50mV\" vreset=\"-70mV\" t_ref=\"2ms\"";
        case GlifVariant::Glif2:
            return "C=\"100pF\" gL=\"10nS\" EL=\"-70mV\" vth=\"-50mV\" vreset=\"-70mV\" resetScale=\"0.4\" "
                   "t_ref=\"2ms\"";
        case GlifVariant::Glif3:
            return "C=\"100pF\" gL=\"10nS\" EL=\"-70mV\" vth=\"-50mV\" vreset=\"-70mV\" t_ref=\"5ms\" "
                   "tauAsc1=\"100ms\" tauAsc2=\"10ms\" ascAdd1=\"-100pA\" ascAdd2=\"-200pA\"";
        case GlifVariant::Glif4:
            return "C=\"100pF\" gL=\"10nS\" EL=\"-70mV\" vreset=\"-70mV\" t_ref=\"2ms\" thetaInf=\"-50mV\" "
                   "tauTheta=\"50ms\" thetaSpikeAdd=\"5mV\"";
        case GlifVariant::Glif5:
            return "C=\"100pF\" gL=\"10nS\" EL=\"-70mV\" vreset=\"-70mV\" t_ref=\"5ms\" thetaInf=\"-50mV\" "
                   "tauTheta=\"50ms\" thetaSpikeAdd=\"5mV\" tauAsc1=\"100ms\" tauAsc2=\"10ms\" "
                   "ascAdd1=\"-100pA\" ascAdd2=\"-200pA\"";
    }
    throw std::runtime_error("glif_cell_instance_attributes: unknown GlifVariant");
}

// The declared-order slot index of `state_variable_name` within `variant`'s `<Dynamics>`.
//
// A population's cell_state chunk is structure-of-arrays, so slot `k`'s [population.size] row sits at
// `cell_type_boundaries[population_index] + k * population.size` — this index is exactly what a
// caller needs to read that variable back out for any neuron.
inline s32 glif_state_variable_slot(GlifVariant variant, const String &state_variable_name) {
    const UnorderedMap<String, s32> glif1_slots = {{"v", 0}, {"refractoryTimeElapsed", 1}};
    const UnorderedMap<String, s32> glif2_slots = {{"v", 0}, {"refractoryTimeElapsed", 1}};
    const UnorderedMap<String, s32> glif3_slots = {{"v", 0}, {"asc1", 1}, {"asc2", 2}, {"refractoryTimeElapsed", 3}};
    const UnorderedMap<String, s32> glif4_slots = {{"v", 0}, {"theta", 1}, {"refractoryTimeElapsed", 2}};
    const UnorderedMap<String, s32> glif5_slots = {
        {"v", 0}, {"theta", 1}, {"asc1", 2}, {"asc2", 3}, {"refractoryTimeElapsed", 4}};

    const UnorderedMap<String, s32> *slots = nullptr;
    switch (variant) {
        case GlifVariant::Glif1: slots = &glif1_slots; break;
        case GlifVariant::Glif2: slots = &glif2_slots; break;
        case GlifVariant::Glif3: slots = &glif3_slots; break;
        case GlifVariant::Glif4: slots = &glif4_slots; break;
        case GlifVariant::Glif5: slots = &glif5_slots; break;
    }
    if (slots == nullptr) throw std::runtime_error("glif_state_variable_slot: unknown GlifVariant");

    auto found = slots->find(state_variable_name);
    if (found == slots->end()) {
        throw std::runtime_error(
            "glif_state_variable_slot: " + glif_component_type_name(variant)
            + " has no state variable named '" + state_variable_name + "'");
    }
    return found->second;
}

// ── Network generation ──────────────────────────────────────────────────────────────────────────

struct TorusNetworkOptions {
    GlifVariant variant = GlifVariant::Glif3;

    // The torus is `side_length × side_length`, so the population holds `side_length²` neurons, each
    // wired to its 4 wraparound grid neighbors.
    s64 side_length = 8;

    // Neurons receiving the external current step, by flat index (row * side_length + column).
    Vector<s32> stimulated_neuron_indices = {0};

    // pulseGenerator attributes, as verbatim unit-suffixed NML literals.
    String stimulus_delay = "10ms";
    String stimulus_duration = "200ms";
    String stimulus_amplitude = "600pA";

    // Non-empty (e.g. "2ms") emits `<connectionWD delay="…"/>` for every edge, which the delay-ring
    // subsystem consumes. Empty emits a plain `<connection>`, floored to the implicit one-tick latency.
    String connection_delay;

    // ── ticket #131 / #138: real per-edge synapse dispatch replaces the old constant-weight
    // scatter placeholder ──────────────────────────────────────────────────────────────────────
    // `include_lateral_connections=false` omits the `<projection>`/`<connection>` block (and the
    // `expOneSynapse` declaration) entirely, leaving a population of otherwise-identical neurons
    // with NO edges at all — every neuron's own trajectory then depends solely on whatever this
    // caller injects into `network_inputs` directly. `include_lateral_connections=true` (the
    // default) emits the full 4-neighbor wraparound `<projection>` wired through a real
    // `expOneSynapse`, whose `gbase` (this struct's own `synapse_gbase`) AssembledModel now
    // dispatches automatically (ticket #131) — see this header's own top-of-file note and each
    // torus example's header comment for why a constant-weight placeholder is no longer the
    // mechanism that makes the torus propagate.
    bool include_lateral_connections = true;

    // expOneSynapse's own `gbase` attribute, as a verbatim unit-suffixed NML literal (e.g. "10nS").
    // Only used when `include_lateral_connections` is true. See
    // parse_torus_example_options's own doc comment for how this value was chosen (the propagation
    // "knee" for this exact parameter set/topology).
    String synapse_gbase = "10nS";
};

inline s64 torus_neuron_count(const TorusNetworkOptions &options) {
    return options.side_length * options.side_length;
}

// Renders a complete NeuroML document: the variant's ComponentType, one bound cell instance,
// (if `include_lateral_connections`) a vendored expOneSynapse for the projection to name, then a
// network holding one population, the explicitInput stimuli, and (again, if
// `include_lateral_connections`) one `<connection>` per torus edge.
inline String generate_glif_torus_network_nml(const TorusNetworkOptions &options) {
    if (options.side_length < 2) {
        throw std::runtime_error("generate_glif_torus_network_nml: side_length must be at least 2");
    }

    const Vector<Vector<s32>> torus_adjacency = square_torus(options.side_length);
    const s64 neuron_count = torus_neuron_count(options);
    const String component_type_name = glif_component_type_name(options.variant);
    const String cell_instance_id = "glifCellInstance";
    const String population_id = "TorusPop";

    std::ostringstream document;
    document << "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"GlifTorusNetwork\">\n"
             << glif_component_type_xml(options.variant) << "\n"
             << "  <" << component_type_name << " id=\"" << cell_instance_id << "\" "
             << glif_cell_instance_attributes(options.variant) << "/>\n";

    if (options.include_lateral_connections) {
        // A real vendored synapse for the projection to reference — AssembledModel now dispatches
        // its actual gbase/tauDecay/erev-derived per-edge conductance (ticket #131), not a
        // constant-weight placeholder (see this header's own top-of-file note).
        //
        // `weight="1"` is set explicitly to its own Property defaultValue (matching
        // tests/fixtures/nml/izhikevich_network.nml's own established precedent, see that fixture's
        // own header comment): model_specification.cpp's baking does not yet fall back to a
        // Property's bare `defaultValue` when the bound instance leaves it unset, and ticket #131's
        // synapse dispatch needs "weight" baked into TypeLibraryEntry::baked_constants to compile
        // this synapse's real per-edge `_deliver_<port>`/`_integrate_edges` functions at all.
        document << "  <expOneSynapse id=\"torusSynapse\" gbase=\"" << options.synapse_gbase
                 << "\" erev=\"0mV\" tauDecay=\"5ms\" weight=\"1\"/>\n";
    }

    for (usize stimulus_index = 0; stimulus_index < options.stimulated_neuron_indices.size(); ++stimulus_index) {
        document << "  <pulseGenerator id=\"pulseGen" << stimulus_index << "\" delay=\"" << options.stimulus_delay
                 << "\" duration=\"" << options.stimulus_duration << "\" amplitude=\"" << options.stimulus_amplitude
                 << "\"/>\n";
    }

    document << "  <network id=\"GlifTorusNet\">\n"
             << "    <population id=\"" << population_id << "\" component=\"" << cell_instance_id
             << "\" size=\"" << neuron_count << "\"/>\n";

    for (usize stimulus_index = 0; stimulus_index < options.stimulated_neuron_indices.size(); ++stimulus_index) {
        document << "    <explicitInput target=\"" << population_id << "["
                 << options.stimulated_neuron_indices[stimulus_index] << "]\" input=\"pulseGen" << stimulus_index
                 << "\"/>\n";
    }

    if (options.include_lateral_connections) {
        document << "    <projection id=\"TorusProj\" presynapticPopulation=\"" << population_id
                 << "\" postsynapticPopulation=\"" << population_id << "\" synapse=\"torusSynapse\">\n";

        s32 connection_id = 0;
        for (s32 source_index = 0; source_index < (s32)neuron_count; ++source_index) {
            for (s32 target_index : torus_adjacency[(usize)source_index]) {
                // square_torus never emits a self-loop for side_length >= 2, but K2Tree rejects them
                // outright, so the guard stays.
                if (target_index == source_index) continue;

                const String source_path =
                    population_id + "/" + std::to_string(source_index) + "/" + cell_instance_id;
                const String target_path =
                    population_id + "/" + std::to_string(target_index) + "/" + cell_instance_id;

                if (options.connection_delay.empty()) {
                    document << "      <connection id=\"" << connection_id << "\" preCellId=\"" << source_path
                             << "\" postCellId=\"" << target_path << "\"/>\n";
                } else {
                    document << "      <connectionWD id=\"" << connection_id << "\" preCellId=\"" << source_path
                             << "\" postCellId=\"" << target_path << "\" weight=\"1\" delay=\""
                             << options.connection_delay << "\"/>\n";
                }
                ++connection_id;
            }
        }

        document << "    </projection>\n";
    }

    document << "  </network>\n"
             << "</neuroml>\n";

    return document.str();
}

// Writes `content_xml` plus a thin `<include>`-only wrapper to the temp directory and runs the pair
// through the front-end.
//
// The split exists for the same reason the checked-in fixtures use it: NML_Parser XSD-validates only
// the top-level file it is handed, and raw LEMS ComponentType declarations do not validate against
// the NeuroML schema.
inline nml::ModelSpecification load_generated_model(const String &document_id, const String &content_xml) {
    const String content_file_name = "spikecorec_example_" + document_id + "_content.nml";
    const std::filesystem::path content_path = std::filesystem::temp_directory_path() / content_file_name;
    const std::filesystem::path top_path =
        std::filesystem::temp_directory_path() / ("spikecorec_example_" + document_id + "_top.nml");

    std::ofstream(content_path) << content_xml;
    std::ofstream(top_path) << "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"" << document_id
                            << "Top\">\n  <include href=\"" << content_file_name << "\"/>\n</neuroml>\n";

    nml::NML_Parser parser;
    parser.parse(top_path.string());
    nml::ResolvedModel resolved = nml::resolve_and_lower(parser);
    return nml::build_model_specification(resolved);
}

// ── Initial state ───────────────────────────────────────────────────────────────────────────────

// Applies the variant's OnStart block by hand: `v = EL` for every variant, plus `theta = thetaInf`
// for GLIF4/GLIF5 (the two variants that declare an adaptive threshold).
//
// allocate_model zero-initializes cell_state and does not evaluate OnStart, so without this a
// GLIF4/GLIF5 network starts with `theta = 0`, which is ABOVE the -50 mV threshold it should have —
// every neuron would sit permanently sub-threshold and the network would never fire.
inline void seed_glif_initial_state(
    nml::ModelAllocation &allocation, const nml::ModelSpecification &model, GlifVariant variant
) {
    seed_membrane_potentials_from_resting_parameter(allocation, model);

    if (variant != GlifVariant::Glif4 && variant != GlifVariant::Glif5) return;

    for (s32 population_index = 0; population_index < (s32)model.populations.size(); ++population_index) {
        const nml::PopulationEntry &population = model.populations[(usize)population_index];
        const nml::TypeLibraryEntry &cell_entry = model.type_library[(usize)population.type_library_index];
        const f32 resting_threshold_volts = (f32)cell_entry.baked_constants.at("thetaInf");
        const s32 theta_slot = glif_state_variable_slot(variant, "theta");

        for (s32 local_index = 0; local_index < population.size; ++local_index) {
            allocation.cell_state.get_contents()[
                state_element_index(allocation, population, population_index, theta_slot, local_index)] =
                resting_threshold_volts;
        }
    }
}

// ── Command-line options ────────────────────────────────────────────────────────────────────────

struct TorusExampleOptions {
    ExampleOptions base;

    s64 side_length = 8;               // side_length² neurons
    String synapse_gbase = "10nS";     // expOneSynapse's real per-edge conductance amplitude
    String record_directory = "recordings"; // where torus examples write their .spire pairs
    bool record = true;                // pass --no-record to skip writing .spire files entirely
};

// Parses the shared flags plus `--side <length>`, `--gbase <siemens literal, e.g. 3nS>`,
// `--record-dir <path>`, and `--no-record`.
//
// `--gbase` deserves explanation. Ticket #131 made AssembledModel dispatch a projection's real
// synapse ComponentType dynamics automatically whenever the model has one — the torus's
// `expOneSynapse` conductance is no longer a stipulated placeholder current, it is `gbase` itself
// (folded through expOneSynapse's own `g' = -g/tauDecay`, `g += gbase` on a delivered spike, and
// `i = g * (erev - v)`). The default below (10nS) is calibrated the same way the old
// `--weight`/`scattered_weight` placeholder used to be, against this exact parameter set/topology
// (8×8 torus, GLIF3/5 as declared above, erev=0mV, tauDecay=5ms) — measured empirically with
// `--ticks 3000`:
//
//     up to 9nS       no propagation at all (only the stimulated corner ever fires — a single GLIF3
//                     spike's own conductance bump decays well below threshold before it reaches a
//                     neighbor, a hard step rather than a gradual falloff, since GLIF's threshold
//                     crossing is itself a hard condition)
//     10nS            the whole torus recruits, with a clean spatial gradient — first-spike-tick
//                     spreads from the driven corner in Manhattan distance and meets itself at the  ← default
//                     antipode, exactly the wavefront glif3_torus_network_example's own README entry
//                     describes
//     20nS and up     recruits much faster; less of a visible spatial gradient
//
// Treat it as a dial for exploring the topology, not as a biologically calibrated conductance.
inline TorusExampleOptions parse_torus_example_options(
    int argument_count, char **argument_values, TorusExampleOptions defaults
) {
    TorusExampleOptions options = defaults;
    for (int argument_index = 1; argument_index < argument_count; ++argument_index) {
        String argument = argument_values[argument_index];
        bool has_value = argument_index + 1 < argument_count;
        if (argument == "--ticks" && has_value) {
            options.base.tick_count = std::strtoll(argument_values[++argument_index], nullptr, 10);
        } else if (argument == "--dt" && has_value) {
            options.base.dt_seconds = std::strtof(argument_values[++argument_index], nullptr);
        } else if (argument == "--side" && has_value) {
            options.side_length = std::strtoll(argument_values[++argument_index], nullptr, 10);
        } else if (argument == "--gbase" && has_value) {
            options.synapse_gbase = argument_values[++argument_index];
        } else if (argument == "--record-dir" && has_value) {
            options.record_directory = argument_values[++argument_index];
        } else if (argument == "--no-record") {
            options.record = false;
        } else if (argument == "--print-ir") {
            options.base.print_ir = true;
        } else if (argument == "--verbose") {
            options.base.verbose = true;
        } else {
            std::cerr << "ignoring unrecognized argument '" << argument << "' (supported: --ticks <count> "
                      << "--dt <seconds> --side <length> --gbase <siemens, e.g. 3nS> "
                      << "--record-dir <path> --no-record --print-ir --verbose)\n";
        }
    }
    configure_example_logging(options.base.verbose);
    return options;
}

// ── Torus-shaped output ─────────────────────────────────────────────────────────────────────────

// Renders a per-neuron value over the torus grid, one character per neuron, using a density ramp.
// A torus wraps in both directions, so the left and right edges are neighbors, as are top and bottom.
//
// A NEGATIVE value means "no data for this neuron" (e.g. a first-spike tick for a neuron that never
// fired) and renders blank. Everything else is normalized across the observed MIN-to-MAX range, not
// 0-to-max: a first-spike-tick grid whose values all sit between 100 and 148 would otherwise collapse
// onto a single ramp character and show nothing.
inline void print_torus_grid(
    const String &label, const Vector<s64> &values_by_neuron, s64 side_length, const String &value_name
) {
    const String density_ramp = ".:-=+*#%@";

    s64 minimum_value = 0;
    s64 maximum_value = 0;
    bool any_value_present = false;
    for (s64 value : values_by_neuron) {
        if (value < 0) continue;
        if (!any_value_present) {
            minimum_value = value;
            maximum_value = value;
            any_value_present = true;
            continue;
        }
        minimum_value = std::min(minimum_value, value);
        maximum_value = std::max(maximum_value, value);
    }

    if (!any_value_present) {
        std::cout << "  " << label << "\n    no neuron has a value — nothing to plot\n";
        return;
    }

    std::cout << "  " << label << "   (" << value_name << " per neuron, range " << minimum_value << " … "
              << maximum_value << ")\n";

    const s64 value_span = std::max<s64>(maximum_value - minimum_value, 1);
    for (s64 row_index = 0; row_index < side_length; ++row_index) {
        std::cout << "    ";
        for (s64 column_index = 0; column_index < side_length; ++column_index) {
            const s64 value = values_by_neuron[(usize)(row_index * side_length + column_index)];
            if (value < 0) {
                std::cout << "  ";
                continue;
            }
            const usize ramp_index =
                (usize)((value - minimum_value) * (s64)(density_ramp.size() - 1) / value_span);
            std::cout << density_ramp[ramp_index] << ' ';
        }
        std::cout << "\n";
    }
    std::cout << "    (ramp: '" << density_ramp << "', low → high; blank = no value; grid wraps on both axes)\n";
}

} // namespace spikecorec::examples
