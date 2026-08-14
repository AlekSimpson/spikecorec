#pragma once

// Generates a balanced excitatory/inhibitory recurrent network built from any one of the
// five GLIF cell types, plus the LEMS document that runs it. The same generator backs the
// demo and the tests, so what the tests assert is what the demo shows.
//
// The electrical parameters are shared across all five types, which is what makes the
// networks comparable: C = 100 pF and gL = 10 nS give a 10 ms membrane time constant and a
// rheobase of gL * (vth - EL) = 10 nS * 20 mV = 200 pA, and a 5 ms refractory period caps
// any cell at 200 Hz.
//
// An alphaCurrentSynapse event delivers a charge of e * ibase * tau, so it moves the
// postsynaptic membrane by e * ibase * tau / C: 1.5 mV at ibase = 11 pA, meaning about
// thirteen coincident excitatory events carry a resting cell to threshold. Inhibition is
// current-based, so it is the same synapse with a negative ibase, scaled so that the
// recurrent excitatory and inhibitory drives roughly cancel and the network's rate is set
// by the injected drive rather than by runaway recurrence.

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "spikecorec/core/types.h"

namespace spikecorec::examples {

struct GlifNetworkParameters {
    // 1 through 5. Chooses which GLIF ComponentType the populations instantiate.
    s32 glif_index = 1;

    s64 excitatory_count = 400;
    s64 inhibitory_count = 100;
    s64 outgoing_edges_per_neuron = 20;

    f64 excitatory_ibase_amperes = 11e-12;
    f64 inhibitory_ibase_amperes = -50e-12;
    f64 synapse_tau_seconds = 5e-3;

    // Below the 200 pA rheobase, so nothing fires on background alone.
    f64 background_current_amperes = 190e-12;
    // Takes a fraction of the excitatory population above rheobase to seed activity.
    f64 seed_extra_current_amperes = 45e-12;
    f64 seed_fraction = 0.2;

    f64 minimum_delay_seconds = 1e-3;
    f64 maximum_delay_seconds = 4e-3;

    f64 simulation_seconds = 2.0;
    f64 step_seconds = 1e-4;

    u64 seed = 20260813;

    // Where the GLIF ComponentType declarations live. Emitted into the generated model as
    // an absolute href so the model can be written to any directory -- a test's temporary
    // one included -- without needing the types file copied next to it.
    String cell_types_path = "tests/fixtures/nml/glif_cell_types.nml";

    // How many membrane traces the LEMS document asks for. The video renderer does not use
    // these; they are for eyeballing individual cells.
    s64 traced_cell_count = 6;

    [[nodiscard]] s64 total_count() const { return excitatory_count + inhibitory_count; }
};

// The attributes each GLIF type needs, beyond the shared ones. GLIF4 and GLIF5 have no
// `vth`: their threshold is the state variable theta, which starts at thetaInf.
inline String glif_cell_attributes(s32 glif_index) {
    const String shared = R"( C="100pF" gL="10nS" EL="-70mV" vreset="-70mV" t_ref="5ms")";

    switch (glif_index) {
        case 1: return shared + R"( vth="-50mV")";
        case 2: return shared + R"( vth="-50mV" resetScale="0.3")";
        case 3: return shared + R"( vth="-50mV" tauAsc1="100ms" tauAsc2="10ms")"
                                R"( ascAdd1="-60pA" ascAdd2="-120pA")";
        case 4: return shared + R"( thetaInf="-50mV" tauTheta="50ms" thetaSpikeAdd="3mV")";
        case 5: return shared + R"( thetaInf="-50mV" tauTheta="50ms" thetaSpikeAdd="3mV")"
                                R"( tauAsc1="100ms" tauAsc2="10ms")"
                                R"( ascAdd1="-60pA" ascAdd2="-120pA")";
        default:
            throw std::runtime_error("glif_cell_attributes: GLIF index must be 1..5, got " +
                                     std::to_string(glif_index));
    }
}

// Writes <directory>/glif<N>_network.nml and its LEMS document; returns the LEMS path.
inline String write_glif_network_model(const String &directory,
                                       const GlifNetworkParameters &parameters) {
    const String tag = "glif" + std::to_string(parameters.glif_index);
    const String cell_type = "GLIF" + std::to_string(parameters.glif_index) + "Cell";

    std::mt19937_64 generator(parameters.seed);
    std::uniform_real_distribution<f64> delay_distribution(parameters.minimum_delay_seconds,
                                                           parameters.maximum_delay_seconds);

    struct Edge { s64 source; s64 target; f64 delay; };

    const s64 excitatory_count = parameters.excitatory_count;
    const s64 total_count = parameters.total_count();

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

    auto population_of = [&](s64 neuron_index) -> String {
        return neuron_index < excitatory_count ? tag + "Excitatory" : tag + "Inhibitory";
    };
    auto local_index = [&](s64 neuron_index) -> s64 {
        return neuron_index < excitatory_count ? neuron_index
                                               : neuron_index - excitatory_count;
    };

    const String model_path = directory + "/" + tag + "_network.nml";
    const String lems_path = directory + "/LEMS_" + tag + "_network.xml";

    std::ofstream model(model_path);
    model << std::setprecision(12);
    model << "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"" << tag
          << "Network\">\n\n"
          << "  <include href=\""
          << std::filesystem::absolute(parameters.cell_types_path).string()
          << "\"/>\n\n";

    model << "  <" << cell_type << " id=\"" << tag << "ExcitatoryCell\""
          << glif_cell_attributes(parameters.glif_index) << "/>\n";
    model << "  <" << cell_type << " id=\"" << tag << "InhibitoryCell\""
          << glif_cell_attributes(parameters.glif_index) << "/>\n\n";

    model << "  <alphaCurrentSynapse id=\"" << tag << "ExcitatorySynapse\" tau=\""
          << parameters.synapse_tau_seconds << " s\" ibase=\""
          << parameters.excitatory_ibase_amperes << " A\"/>\n";
    model << "  <alphaCurrentSynapse id=\"" << tag << "InhibitorySynapse\" tau=\""
          << parameters.synapse_tau_seconds << " s\" ibase=\""
          << parameters.inhibitory_ibase_amperes << " A\"/>\n\n";

    model << "  <pulseGenerator id=\"" << tag << "Background\" delay=\"0 s\" duration=\""
          << parameters.simulation_seconds << " s\" amplitude=\""
          << parameters.background_current_amperes << " A\"/>\n";
    model << "  <pulseGenerator id=\"" << tag << "Seed\" delay=\"0 s\" duration=\""
          << parameters.simulation_seconds << " s\" amplitude=\""
          << parameters.seed_extra_current_amperes << " A\"/>\n\n";

    model << "  <network id=\"" << tag << "Network\">\n";
    model << "    <population id=\"" << tag << "Excitatory\" component=\"" << tag
          << "ExcitatoryCell\" size=\"" << excitatory_count << "\"/>\n";
    model << "    <population id=\"" << tag << "Inhibitory\" component=\"" << tag
          << "InhibitoryCell\" size=\"" << parameters.inhibitory_count << "\"/>\n\n";

    const char *bucket_names[4] = {"ExcToExc", "ExcToInh", "InhToExc", "InhToInh"};
    for (s64 bucket = 0; bucket < 4; bucket += 1) {
        const String synapse = tag + (bucket < 2 ? "ExcitatorySynapse" : "InhibitorySynapse");
        const String pre = tag + (bucket < 2 ? "Excitatory" : "Inhibitory");
        const String post = tag + ((bucket % 2) == 0 ? "Excitatory" : "Inhibitory");

        model << "    <projection id=\"" << tag << bucket_names[bucket]
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

    model << "    <inputList id=\"" << tag << "BackgroundInput\" component=\"" << tag
          << "Background\" population=\"" << tag << "Excitatory\">\n";
    for (s64 neuron_index = 0; neuron_index < total_count; neuron_index += 1) {
        model << "      <input id=\"" << neuron_index << "\" target=\"../"
              << population_of(neuron_index) << "[" << local_index(neuron_index)
              << "]\" destination=\"synapses\"/>\n";
    }
    model << "    </inputList>\n\n";

    const s64 seed_count = (s64)(parameters.seed_fraction * (f64)excitatory_count);
    model << "    <inputList id=\"" << tag << "SeedInput\" component=\"" << tag
          << "Seed\" population=\"" << tag << "Excitatory\">\n";
    for (s64 neuron_index = 0; neuron_index < seed_count; neuron_index += 1) {
        model << "      <input id=\"" << neuron_index << "\" target=\"../" << tag
              << "Excitatory[" << neuron_index << "]\" destination=\"synapses\"/>\n";
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
            "    <Include file=\"" << tag << "_network.nml\"/>\n\n";
    lems << "    <Simulation id=\"sim1\" length=\"" << parameters.simulation_seconds
         << "s\" step=\"" << parameters.step_seconds << "s\" target=\"" << tag
         << "Network\">\n";

    lems << "        <EventOutputFile id=\"spikes\" fileName=\"" << directory << "/" << tag
         << "_network_spikes.dat\" format=\"TIME_ID\">\n";
    for (s64 neuron_index = 0; neuron_index < total_count; neuron_index += 1) {
        lems << "            <EventSelection id=\"" << neuron_index << "\" select=\""
             << population_of(neuron_index) << "[" << local_index(neuron_index)
             << "]\" eventPort=\"spike\"/>\n";
    }
    lems << "        </EventOutputFile>\n";

    lems << "        <OutputFile id=\"traces\" fileName=\"" << directory << "/" << tag
         << "_network_traces.dat\">\n";
    for (s64 traced = 0; traced < parameters.traced_cell_count; traced += 1) {
        lems << "            <OutputColumn id=\"v" << traced << "\" quantity=\"" << tag
             << "Excitatory[" << (traced * 37) << "]/v\"/>\n";
    }
    lems << "        </OutputFile>\n";

    lems << "    </Simulation>\n\n    <Target component=\"sim1\"/>\n</Lems>\n";
    lems.close();

    return lems_path;
}

} // namespace spikecorec::examples
