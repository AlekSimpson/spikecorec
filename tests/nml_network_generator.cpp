#include "nml_network_generator.h"

#include <sstream>
#include <stdexcept>

namespace spikecorec::nml::network_generation {

namespace {

// Every ComponentType text block below is reused verbatim from
// tests/cell_lowering_tests.cpp's own GLIF1_COMPONENT_TYPE..GLIF5_COMPONENT_TYPE fixtures (ticket
// #50) -- same equations, same Parameter/dimension declarations, same declared StateVariable
// order. See that file's own header comment for the modeling rationale of each variant.

const String GLIF1_COMPONENT_TYPE_XML =
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

const String GLIF2_COMPONENT_TYPE_XML =
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

const String GLIF3_COMPONENT_TYPE_XML =
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

const String GLIF4_COMPONENT_TYPE_XML =
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

const String GLIF5_COMPONENT_TYPE_XML =
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

bool contains_variant(const Vector<GlifVariant> &seen, GlifVariant variant) {
    for (GlifVariant existing : seen) {
        if (existing == variant) return true;
    }
    return false;
}

const GeneratedPopulation &population_by_id(const Vector<GeneratedPopulation> &populations, const String &id) {
    for (const auto &population : populations) {
        if (population.population_id == id) return population;
    }
    throw std::runtime_error("nml_network_generator: projection references unknown population '" + id + "'");
}

} // namespace

String glif_component_type_name(GlifVariant variant) {
    switch (variant) {
        case GlifVariant::Glif1: return "GLIF1Cell";
        case GlifVariant::Glif2: return "GLIF2Cell";
        case GlifVariant::Glif3: return "GLIF3Cell";
        case GlifVariant::Glif4: return "GLIF4Cell";
        case GlifVariant::Glif5: return "GLIF5Cell";
    }
    throw std::runtime_error("nml_network_generator: unknown GlifVariant");
}

String glif_component_type_xml(GlifVariant variant) {
    switch (variant) {
        case GlifVariant::Glif1: return GLIF1_COMPONENT_TYPE_XML;
        case GlifVariant::Glif2: return GLIF2_COMPONENT_TYPE_XML;
        case GlifVariant::Glif3: return GLIF3_COMPONENT_TYPE_XML;
        case GlifVariant::Glif4: return GLIF4_COMPONENT_TYPE_XML;
        case GlifVariant::Glif5: return GLIF5_COMPONENT_TYPE_XML;
    }
    throw std::runtime_error("nml_network_generator: unknown GlifVariant");
}

s32 glif_state_variable_count(GlifVariant variant) {
    switch (variant) {
        case GlifVariant::Glif1: return 2;
        case GlifVariant::Glif2: return 2;
        case GlifVariant::Glif3: return 4;
        case GlifVariant::Glif4: return 3;
        case GlifVariant::Glif5: return 5;
    }
    throw std::runtime_error("nml_network_generator: unknown GlifVariant");
}

s32 glif_state_variable_slot(GlifVariant variant, const String &state_variable_name) {
    static const UnorderedMap<String, s32> GLIF1_SLOTS = {{"v", 0}, {"refractoryTimeElapsed", 1}};
    static const UnorderedMap<String, s32> GLIF2_SLOTS = {{"v", 0}, {"refractoryTimeElapsed", 1}};
    static const UnorderedMap<String, s32> GLIF3_SLOTS = {
        {"v", 0}, {"asc1", 1}, {"asc2", 2}, {"refractoryTimeElapsed", 3}};
    static const UnorderedMap<String, s32> GLIF4_SLOTS = {{"v", 0}, {"theta", 1}, {"refractoryTimeElapsed", 2}};
    static const UnorderedMap<String, s32> GLIF5_SLOTS = {
        {"v", 0}, {"theta", 1}, {"asc1", 2}, {"asc2", 3}, {"refractoryTimeElapsed", 4}};

    const UnorderedMap<String, s32> *slots = nullptr;
    switch (variant) {
        case GlifVariant::Glif1: slots = &GLIF1_SLOTS; break;
        case GlifVariant::Glif2: slots = &GLIF2_SLOTS; break;
        case GlifVariant::Glif3: slots = &GLIF3_SLOTS; break;
        case GlifVariant::Glif4: slots = &GLIF4_SLOTS; break;
        case GlifVariant::Glif5: slots = &GLIF5_SLOTS; break;
    }
    if (slots == nullptr) throw std::runtime_error("nml_network_generator: unknown GlifVariant");

    auto found = slots->find(state_variable_name);
    if (found == slots->end()) {
        throw std::runtime_error("nml_network_generator: GLIF variant has no state variable named '" +
                                  state_variable_name + "'");
    }
    return found->second;
}

String generate_network_nml(const String &document_id, const Vector<GeneratedPopulation> &populations,
                             const String &synapse_declarations_xml, const Vector<GeneratedProjection> &projections,
                             const Vector<GeneratedStimulus> &stimuli) {
    std::ostringstream out;
    out << "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"" << document_id << "\">\n";

    Vector<GlifVariant> emitted_variants;
    for (const auto &population : populations) {
        if (contains_variant(emitted_variants, population.variant)) continue;
        emitted_variants.push_back(population.variant);
        out << glif_component_type_xml(population.variant) << "\n";
    }

    for (const auto &population : populations) {
        out << "  <" << glif_component_type_name(population.variant) << " id=\"" << population.bound_instance_id
            << "\" " << population.cell_instance_attributes << "/>\n";
    }

    if (!synapse_declarations_xml.empty()) out << synapse_declarations_xml << "\n";

    for (const auto &stimulus : stimuli) {
        out << "  <pulseGenerator id=\"" << stimulus.pulse_generator_id << "\" delay=\"" << stimulus.delay
            << "\" duration=\"" << stimulus.duration << "\" amplitude=\"" << stimulus.amplitude << "\"/>\n";
    }

    out << "  <network id=\"" << document_id << "Net\">\n";

    for (const auto &population : populations) {
        out << "    <population id=\"" << population.population_id << "\" component=\""
            << population.bound_instance_id << "\" size=\"" << population.neuron_count << "\"/>\n";
    }

    for (const auto &stimulus : stimuli) {
        out << "    <explicitInput target=\"" << stimulus.population_id << "[" << stimulus.local_index
            << "]\" input=\"" << stimulus.pulse_generator_id << "\"/>\n";
    }

    for (const auto &projection : projections) {
        const GeneratedPopulation &presynaptic_population = population_by_id(populations, projection.presynaptic_population_id);
        const GeneratedPopulation &postsynaptic_population = population_by_id(populations, projection.postsynaptic_population_id);

        out << "    <projection id=\"" << projection.projection_id << "\" presynapticPopulation=\""
            << projection.presynaptic_population_id << "\" postsynapticPopulation=\""
            << projection.postsynaptic_population_id << "\" synapse=\"" << projection.synapse_instance_id << "\">\n";

        s32 connection_id = 0;
        for (s32 source_local_index = 0; source_local_index < presynaptic_population.neuron_count; ++source_local_index) {
            for (s32 offset = 1; offset <= projection.out_degree; ++offset) {
                s32 target_local_index = (source_local_index + offset) % postsynaptic_population.neuron_count;
                // K2Tree rejects self-loops -- only reachable for a size-1 self-projection, where
                // every offset wraps back onto the same (only) neuron.
                if (presynaptic_population.population_id == postsynaptic_population.population_id &&
                    target_local_index == source_local_index) {
                    continue;
                }

                String presynaptic_cell_path = projection.presynaptic_population_id + "/" +
                    std::to_string(source_local_index) + "/" + presynaptic_population.bound_instance_id;
                String postsynaptic_cell_path = projection.postsynaptic_population_id + "/" +
                    std::to_string(target_local_index) + "/" + postsynaptic_population.bound_instance_id;

                if (projection.delay_attribute.empty()) {
                    out << "      <connection id=\"" << connection_id << "\" preCellId=\"" << presynaptic_cell_path
                        << "\" postCellId=\"" << postsynaptic_cell_path << "\"/>\n";
                } else {
                    out << "      <connectionWD id=\"" << connection_id << "\" preCellId=\"" << presynaptic_cell_path
                        << "\" postCellId=\"" << postsynaptic_cell_path << "\" weight=\"1\" delay=\""
                        << projection.delay_attribute << "\"/>\n";
                }
                ++connection_id;
            }
        }

        out << "    </projection>\n";
    }

    out << "  </network>\n";
    out << "</neuroml>\n";
    return out.str();
}

} // namespace spikecorec::nml::network_generation
