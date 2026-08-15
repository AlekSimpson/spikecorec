#pragma once

// A balanced excitatory/inhibitory network of GLIF cells with random connectivity, written
// as NeuroML. Shared by the five *_network_demo programs, which differ only in which GLIF
// type the populations instantiate.
//
// Connections are written into the document here rather than generated, because at this
// size (10,000 edges) that is perfectly reasonable and it exercises the parser's own
// projection handling. The torus demos take the other route.
//
// An alphaCurrentSynapse event delivers a charge of e * ibase * tau, so it moves the
// postsynaptic membrane by e * ibase * tau / C. Inhibition is current-based, so it is the
// same synapse with a negative ibase, scaled so recurrent excitation and inhibition
// roughly cancel and the rate is set by the injected drive rather than by runaway feedback.

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

#include "spikecorec/core/types.h"
#include "glif_cell_attributes.h"
#include "demo_support.h"

namespace spikecorec::demos {

struct NetworkDemoParameters {
    s32 glif_index = 1;

    s64 excitatory_count = 400;
    s64 inhibitory_count = 100;
    s64 outgoing_edges_per_neuron = 20;

    f64 excitatory_ibase_amperes = 11e-12;
    f64 inhibitory_ibase_amperes = -50e-12;
    f64 synapse_tau_seconds = 5e-3;

    f64 background_current_amperes = 190e-12;  // below the 200 pA rheobase
    f64 seed_extra_current_amperes = 45e-12;   // takes a seeded cell above it
    f64 seed_fraction = 0.2;

    f64 minimum_delay_seconds = 1e-3;
    f64 maximum_delay_seconds = 4e-3;

    f64 simulation_seconds = 2.0;
    f64 step_seconds = 1e-4;

    u64 seed = 20260813;

    [[nodiscard]] s64 total_count() const { return excitatory_count + inhibitory_count; }
};

// Writes build/demos/<demo_name>.nml and its LEMS document; returns the LEMS path.
inline String write_network_model(const String &demo_name,
                                  const NetworkDemoParameters &parameters,
                                  const String &directory = demo_directory()) {
    std::mt19937_64 generator(parameters.seed);
    std::uniform_real_distribution<f64> delay_distribution(parameters.minimum_delay_seconds,
                                                           parameters.maximum_delay_seconds);

    struct Edge { s64 source; s64 target; f64 delay; };

    const s64 excitatory_count = parameters.excitatory_count;
    const s64 total_count = parameters.total_count();

    // A NeuroML projection names one presynaptic and one postsynaptic population, so the
    // four source/target combinations are four projections.
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
            const s64 bucket = (source < excitatory_count ? 0 : 2) +
                               (target < excitatory_count ? 0 : 1);
            edges_by_projection[bucket].push_back(
                    Edge{source, target, delay_distribution(generator)});
        }
    }

    auto population_of = [&](s64 index) -> String {
        return index < excitatory_count ? "popExcitatory" : "popInhibitory";
    };
    auto local_index = [&](s64 index) -> s64 {
        return index < excitatory_count ? index : index - excitatory_count;
    };

    const String model_path = demo_path(demo_name, ".nml", directory);
    const String lems_path = directory + "/LEMS_" + demo_name + ".xml";
    const String cell_type = glif_cell_type_name(parameters.glif_index);

    std::ofstream model(model_path);
    model << std::setprecision(12);
    model << "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"" << demo_name
          << "\">\n\n"
          << "  <include href=\""
          << std::filesystem::absolute(glif_cell_types_path()).string() << "\"/>\n\n";

    model << "  <" << cell_type << " id=\"excitatoryCell\""
          << glif_cell_attributes(parameters.glif_index) << "/>\n";
    model << "  <" << cell_type << " id=\"inhibitoryCell\""
          << glif_cell_attributes(parameters.glif_index) << "/>\n\n";

    model << "  <alphaCurrentSynapse id=\"excitatorySynapse\" tau=\""
          << parameters.synapse_tau_seconds << " s\" ibase=\""
          << parameters.excitatory_ibase_amperes << " A\"/>\n";
    model << "  <alphaCurrentSynapse id=\"inhibitorySynapse\" tau=\""
          << parameters.synapse_tau_seconds << " s\" ibase=\""
          << parameters.inhibitory_ibase_amperes << " A\"/>\n\n";

    model << "  <pulseGenerator id=\"background\" delay=\"0 s\" duration=\""
          << parameters.simulation_seconds << " s\" amplitude=\""
          << parameters.background_current_amperes << " A\"/>\n";
    model << "  <pulseGenerator id=\"seedDrive\" delay=\"0 s\" duration=\""
          << parameters.simulation_seconds << " s\" amplitude=\""
          << parameters.seed_extra_current_amperes << " A\"/>\n\n";

    model << "  <network id=\"network\">\n";
    model << "    <population id=\"popExcitatory\" component=\"excitatoryCell\" size=\""
          << excitatory_count << "\"/>\n";
    model << "    <population id=\"popInhibitory\" component=\"inhibitoryCell\" size=\""
          << parameters.inhibitory_count << "\"/>\n\n";

    const char *bucket_names[4] = {"excToExc", "excToInh", "inhToExc", "inhToInh"};
    for (s64 bucket = 0; bucket < 4; bucket += 1) {
        const String synapse = bucket < 2 ? "excitatorySynapse" : "inhibitorySynapse";
        const String pre = bucket < 2 ? "popExcitatory" : "popInhibitory";
        const String post = (bucket % 2) == 0 ? "popExcitatory" : "popInhibitory";

        model << "    <projection id=\"" << bucket_names[bucket]
              << "\" presynapticPopulation=\"" << pre << "\" postsynapticPopulation=\""
              << post << "\" synapse=\"" << synapse << "\">\n";

        s64 connection_id = 0;
        for (const Edge &edge : edges_by_projection[bucket]) {
            model << "      <connectionWD id=\"" << connection_id << "\" preCellId=\"../"
                  << population_of(edge.source) << "[" << local_index(edge.source) << "]\""
                  << " postCellId=\"../" << population_of(edge.target) << "["
                  << local_index(edge.target) << "]\" weight=\"1\" delay=\"" << edge.delay
                  << " s\"/>\n";
            connection_id += 1;
        }
        model << "    </projection>\n\n";
    }

    model << "    <inputList id=\"backgroundInput\" component=\"background\" "
             "population=\"popExcitatory\">\n";
    for (s64 index = 0; index < total_count; index += 1) {
        model << "      <input id=\"" << index << "\" target=\"../" << population_of(index)
              << "[" << local_index(index) << "]\" destination=\"synapses\"/>\n";
    }
    model << "    </inputList>\n\n";

    const s64 seeded = (s64)(parameters.seed_fraction * (f64)excitatory_count);
    model << "    <inputList id=\"seedInput\" component=\"seedDrive\" "
             "population=\"popExcitatory\">\n";
    for (s64 index = 0; index < seeded; index += 1) {
        model << "      <input id=\"" << index << "\" target=\"../popExcitatory[" << index
              << "]\" destination=\"synapses\"/>\n";
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
            "    <Include file=\"" << demo_name << ".nml\"/>\n\n";
    lems << "    <Simulation id=\"sim1\" length=\"" << parameters.simulation_seconds
         << "s\" step=\"" << parameters.step_seconds << "s\" target=\"network\">\n";
    lems << "    </Simulation>\n\n    <Target component=\"sim1\"/>\n</Lems>\n";
    lems.close();

    return lems_path;
}

} // namespace spikecorec::demos
