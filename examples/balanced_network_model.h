#pragma once

// Writes a balanced excitatory/inhibitory network of iafCells as NeuroML, plus the LEMS
// document that drives it. Generated rather than checked in because the interesting part
// is the parameters -- twenty thousand <connection> elements are not something to read --
// and because the same generator backs both the example and the aliveness test.
//
// The parameters below are not arbitrary. With C = 100 pF and leakConductance = 5 nS the
// membrane time constant is 20 ms and the rheobase current is 5 nS * 15 mV = 75 pA. An
// alphaCurrentSynapse's response to one event peaks at `ibase` and carries a total charge
// of e * ibase * tau, so a single excitatory event moves the postsynaptic membrane by
// e * ibase * tau / C -- 1.6 mV at ibase = 12 pA, meaning roughly nine coincident events
// carry a resting cell to threshold.
//
// Inhibition is current-based here, so it is simply a negative ibase. Its magnitude times
// the inhibitory in-degree is set slightly above excitation times the excitatory
// in-degree, which is what keeps the recurrent drive from running away while leaving the
// network sensitive enough to fluctuations to fire irregularly.

#include <fstream>
#include <random>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "spikecorec/core/types.h"

namespace spikecorec::examples {

struct BalancedNetworkParameters {
    s64 excitatory_count = 800;
    s64 inhibitory_count = 200;
    s64 outgoing_edges_per_neuron = 20;

    f64 capacitance_farads = 100e-12;
    f64 leak_conductance_siemens = 5e-9;
    f64 leak_reversal_volts = -0.065;
    f64 threshold_volts = -0.050;
    f64 reset_volts = -0.070;

    f64 excitatory_ibase_amperes = 12e-12;
    f64 inhibitory_ibase_amperes = -54e-12;
    f64 synapse_tau_seconds = 5e-3;

    // Applied to every neuron; below the 75 pA rheobase, so nothing fires on background
    // alone and all activity is synaptically driven.
    f64 background_current_amperes = 74e-12;

    // Applied on top of the background to a fraction of the excitatory population, taking
    // those neurons above rheobase so they seed activity for the rest.
    f64 driver_extra_current_amperes = 25e-12;
    f64 driver_fraction = 0.15;

    f64 minimum_delay_seconds = 1e-3;
    f64 maximum_delay_seconds = 5e-3;

    f64 simulation_seconds = 1.0;
    f64 step_seconds = 1e-4;

    u64 seed = 20260813;

    [[nodiscard]] s64 total_count() const { return excitatory_count + inhibitory_count; }
};

// Writes <directory>/balanced_network.nml and <directory>/LEMS_balanced_network.xml, and
// returns the path of the LEMS document.
inline String write_balanced_network_model(const String &directory,
                                           const BalancedNetworkParameters &parameters) {
    std::mt19937_64 generator(parameters.seed);
    std::uniform_real_distribution<f64> delay_distribution(parameters.minimum_delay_seconds,
                                                           parameters.maximum_delay_seconds);

    struct Edge { s64 source; s64 target; f64 delay; };

    const s64 excitatory_count = parameters.excitatory_count;
    const s64 total_count = parameters.total_count();

    // Buckets: a NeuroML projection names one presynaptic and one postsynaptic population,
    // so the four source/target combinations are four projections.
    Vector<Edge> edges_by_projection[4];

    std::uniform_int_distribution<s64> target_distribution(0, total_count - 1);
    for (s64 source = 0; source < total_count; source += 1) {
        Set<s64> chosen;
        while ((s64)chosen.size() < parameters.outgoing_edges_per_neuron) {
            const s64 target = target_distribution(generator);
            if (target == source) continue;
            chosen.insert(target);
        }

        for (s64 target : chosen) {
            const bool source_is_excitatory = source < excitatory_count;
            const bool target_is_excitatory = target < excitatory_count;
            const s64 bucket = (source_is_excitatory ? 0 : 2) + (target_is_excitatory ? 0 : 1);

            edges_by_projection[bucket].push_back(
                    Edge{source, target, delay_distribution(generator)});
        }
    }

    auto population_name = [&](s64 neuron_index) -> String {
        return neuron_index < excitatory_count ? "popExcitatory" : "popInhibitory";
    };
    auto local_index = [&](s64 neuron_index) -> s64 {
        return neuron_index < excitatory_count ? neuron_index : neuron_index - excitatory_count;
    };

    const String model_path = directory + "/balanced_network.nml";
    const String lems_path = directory + "/LEMS_balanced_network.xml";

    std::ofstream model(model_path);
    model << std::setprecision(12);
    model << "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\"\n"
             "         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n"
             "         id=\"BalancedNetwork\">\n\n";

    model << "  <iafCell id=\"excitatoryCell\""
          << " leakConductance=\"" << parameters.leak_conductance_siemens << " S\""
          << " leakReversal=\"" << parameters.leak_reversal_volts << " V\""
          << " thresh=\"" << parameters.threshold_volts << " V\""
          << " reset=\"" << parameters.reset_volts << " V\""
          << " C=\"" << parameters.capacitance_farads << " F\"/>\n";
    model << "  <iafCell id=\"inhibitoryCell\""
          << " leakConductance=\"" << parameters.leak_conductance_siemens << " S\""
          << " leakReversal=\"" << parameters.leak_reversal_volts << " V\""
          << " thresh=\"" << parameters.threshold_volts << " V\""
          << " reset=\"" << parameters.reset_volts << " V\""
          << " C=\"" << parameters.capacitance_farads << " F\"/>\n\n";

    model << "  <alphaCurrentSynapse id=\"excitatorySynapse\""
          << " tau=\"" << parameters.synapse_tau_seconds << " s\""
          << " ibase=\"" << parameters.excitatory_ibase_amperes << " A\"/>\n";
    model << "  <alphaCurrentSynapse id=\"inhibitorySynapse\""
          << " tau=\"" << parameters.synapse_tau_seconds << " s\""
          << " ibase=\"" << parameters.inhibitory_ibase_amperes << " A\"/>\n\n";

    model << "  <pulseGenerator id=\"backgroundDrive\" delay=\"0 s\" duration=\""
          << parameters.simulation_seconds << " s\" amplitude=\""
          << parameters.background_current_amperes << " A\"/>\n";
    model << "  <pulseGenerator id=\"seedDrive\" delay=\"0 s\" duration=\""
          << parameters.simulation_seconds << " s\" amplitude=\""
          << parameters.driver_extra_current_amperes << " A\"/>\n\n";

    model << "  <network id=\"balancedNetwork\">\n";
    model << "    <population id=\"popExcitatory\" component=\"excitatoryCell\" size=\""
          << excitatory_count << "\"/>\n";
    model << "    <population id=\"popInhibitory\" component=\"inhibitoryCell\" size=\""
          << parameters.inhibitory_count << "\"/>\n\n";

    const char *projection_names[4] = {
        "excitatoryToExcitatory", "excitatoryToInhibitory",
        "inhibitoryToExcitatory", "inhibitoryToInhibitory"
    };
    const char *projection_pre[4] = {
        "popExcitatory", "popExcitatory", "popInhibitory", "popInhibitory"
    };
    const char *projection_post[4] = {
        "popExcitatory", "popInhibitory", "popExcitatory", "popInhibitory"
    };

    for (s64 bucket = 0; bucket < 4; bucket += 1) {
        const char *synapse = bucket < 2 ? "excitatorySynapse" : "inhibitorySynapse";

        model << "    <projection id=\"" << projection_names[bucket]
              << "\" presynapticPopulation=\"" << projection_pre[bucket]
              << "\" postsynapticPopulation=\"" << projection_post[bucket]
              << "\" synapse=\"" << synapse << "\">\n";

        s64 connection_id = 0;
        for (const Edge &edge : edges_by_projection[bucket]) {
            model << "      <connectionWD id=\"" << connection_id
                  << "\" preCellId=\"../" << population_name(edge.source) << "["
                  << local_index(edge.source) << "]\""
                  << " postCellId=\"../" << population_name(edge.target) << "["
                  << local_index(edge.target) << "]\""
                  << " weight=\"1\" delay=\"" << edge.delay << " s\"/>\n";
            connection_id += 1;
        }
        model << "    </projection>\n\n";
    }

    model << "    <inputList id=\"backgroundInput\" component=\"backgroundDrive\" "
             "population=\"popExcitatory\">\n";
    for (s64 neuron_index = 0; neuron_index < total_count; neuron_index += 1) {
        model << "      <input id=\"" << neuron_index << "\" target=\"../"
              << population_name(neuron_index) << "[" << local_index(neuron_index)
              << "]\" destination=\"synapses\"/>\n";
    }
    model << "    </inputList>\n\n";

    const s64 driver_count = (s64)(parameters.driver_fraction * (f64)excitatory_count);
    model << "    <inputList id=\"seedInput\" component=\"seedDrive\" "
             "population=\"popExcitatory\">\n";
    for (s64 neuron_index = 0; neuron_index < driver_count; neuron_index += 1) {
        model << "      <input id=\"" << neuron_index << "\" target=\"../popExcitatory["
              << neuron_index << "]\" destination=\"synapses\"/>\n";
    }
    model << "    </inputList>\n";

    model << "  </network>\n</neuroml>\n";
    model.close();

    std::ofstream lems(lems_path);
    lems << std::setprecision(12);
    lems << "<Lems>\n"
            "    <Include file=\"Cells.xml\"/>\n"
            "    <Include file=\"Synapses.xml\"/>\n"
            "    <Include file=\"Inputs.xml\"/>\n"
            "    <Include file=\"Networks.xml\"/>\n"
            "    <Include file=\"Simulation.xml\"/>\n"
            "    <Include file=\"balanced_network.nml\"/>\n\n";
    lems << "    <Simulation id=\"sim1\" length=\"" << parameters.simulation_seconds
         << "s\" step=\"" << parameters.step_seconds << "s\" target=\"balancedNetwork\">\n";
    lems << "        <EventOutputFile id=\"spikes\" fileName=\"balanced_network_spikes.dat\" "
            "format=\"TIME_ID\">\n";
    for (s64 neuron_index = 0; neuron_index < total_count; neuron_index += 1) {
        lems << "            <EventSelection id=\"" << neuron_index << "\" select=\""
             << population_name(neuron_index) << "[" << local_index(neuron_index)
             << "]\" eventPort=\"spike\"/>\n";
    }
    lems << "        </EventOutputFile>\n";
    lems << "        <OutputFile id=\"traces\" fileName=\"balanced_network_traces.dat\">\n";
    for (s64 traced = 0; traced < 8; traced += 1) {
        lems << "            <OutputColumn id=\"v" << traced << "\" quantity=\"popExcitatory["
             << (traced * 97) << "]/v\"/>\n";
    }
    lems << "        </OutputFile>\n";
    lems << "    </Simulation>\n\n    <Target component=\"sim1\"/>\n</Lems>\n";
    lems.close();

    return lems_path;
}

} // namespace spikecorec::examples
