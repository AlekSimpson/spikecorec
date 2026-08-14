#pragma once

// A GLIF network whose connectivity comes from the engine's own topology helpers rather
// than from the document. The LEMS file declares one population, one synapse and the
// stimulus; square_torus() supplies the edges.
//
// That split is what makes size a parameter. A 1,000 x 1,000 torus is a million cells and
// four million edges — as <connection> elements that is four million lines of XML, parsed
// into four million ComponentInstances before a single tick runs. Generated in a loop it
// is a few hundred milliseconds and no document at all.
//
// Electrical parameters are the GLIF family's usual set: C = 100 pF and gL = 10 nS give a
// 10 ms membrane time constant and a rheobase of gL * (vth - EL) = 200 pA, with a 5 ms
// refractory period capping any cell at 200 Hz.

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "spikecorec/core/types.h"
#include "spikecorec/core/topologies.h"

namespace spikecorec::examples {

// How the network is driven. Each pattern injects the same total current on average, so
// what differs between them is where and when it arrives, not how much.
enum class TorusDrivePattern {
    // Every cell held just below rheobase, a fraction of them above it. Activity is
    // sustained by recurrence, and the raster is stationary.
    UniformBackground,
    // Only cells in a disc at the centre of the torus are driven. Activity has to spread
    // outward through the grid, so the raster shows it travelling.
    CentrePatch,
    // The driven fraction is re-drawn every `pattern_period_seconds`, so the network is
    // chased by a moving stimulus rather than settling.
    WanderingPatch,
};

struct GlifTorusParameters {
    s32 glif_index = 1;      // which GLIF ComponentType the population instantiates
    s64 side_length = 32;    // the torus is side_length x side_length cells

    TorusDrivePattern drive_pattern = TorusDrivePattern::UniformBackground;
    f64 pattern_period_seconds = 0.2;
    f64 driven_fraction = 0.2;

    f64 synapse_ibase_amperes = 30e-12;
    f64 synapse_tau_seconds = 5e-3;
    f64 connection_weight = 1.0;
    f64 connection_delay_seconds = 1e-3;

    f64 background_current_amperes = 190e-12;  // below the 200 pA rheobase
    f64 driven_extra_current_amperes = 45e-12; // takes a driven cell above it

    f64 simulation_seconds = 1.0;
    f64 step_seconds = 1e-4;

    String cell_types_path = "tests/fixtures/nml/glif_cell_types.nml";

    [[nodiscard]] s64 neuron_count() const { return side_length * side_length; }
};

// Shared electrical attributes, plus whatever each GLIF type adds. GLIF4 and GLIF5 have no
// `vth`: their threshold is the state variable theta, which starts at thetaInf.
inline String glif_torus_cell_attributes(s32 glif_index) {
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
            throw std::runtime_error("glif_torus_cell_attributes: GLIF index must be 1..5");
    }
}

// Which cells the extra drive lands on, for one pattern at one moment.
inline Vector<s64> driven_cells(const GlifTorusParameters &parameters, s64 pattern_index) {
    const s64 side = parameters.side_length;
    const s64 count = parameters.neuron_count();
    Vector<s64> driven;

    if (parameters.drive_pattern == TorusDrivePattern::UniformBackground) {
        const s64 driven_count = (s64)(parameters.driven_fraction * (f64)count);
        // Spread evenly across the grid rather than taking a contiguous block, so the
        // drive has no spatial structure of its own.
        const s64 stride = driven_count > 0 ? std::max<s64>(1, count / driven_count) : count;
        for (s64 index = 0; index < count; index += stride) driven.push_back(index);
        return driven;
    }

    // Both patch patterns drive a disc; the wandering one moves its centre each period.
    const f64 area_fraction = parameters.driven_fraction;
    const f64 radius = std::sqrt(area_fraction * (f64)count / 3.14159265358979);

    s64 centre_row = side / 2;
    s64 centre_column = side / 2;
    if (parameters.drive_pattern == TorusDrivePattern::WanderingPatch) {
        // A deterministic walk around the torus: a different quadrant each period, so the
        // patch visits the whole grid over a run.
        centre_row = (side / 2 + pattern_index * side / 4) % side;
        centre_column = (side / 2 + pattern_index * side / 3) % side;
    }

    for (s64 row = 0; row < side; row += 1) {
        for (s64 column = 0; column < side; column += 1) {
            // Wraparound distance: the torus has no edges, so the patch wraps too.
            const s64 row_gap = std::min(std::abs(row - centre_row),
                                         side - std::abs(row - centre_row));
            const s64 column_gap = std::min(std::abs(column - centre_column),
                                            side - std::abs(column - centre_column));
            if ((f64)(row_gap * row_gap + column_gap * column_gap) <= radius * radius) {
                driven.push_back(row * side + column);
            }
        }
    }
    return driven;
}

// Writes the model and its LEMS document; returns the LEMS path. The document declares no
// connections at all -- those come from square_torus() through the engine's topology
// constructor.
inline String write_glif_torus_model(const String &directory,
                                     const GlifTorusParameters &parameters) {
    const String tag = "glif" + std::to_string(parameters.glif_index) + "Torus";
    const String cell_type = "GLIF" + std::to_string(parameters.glif_index) + "Cell";

    const String model_path = directory + "/" + tag + ".nml";
    const String lems_path = directory + "/LEMS_" + tag + ".xml";

    std::ofstream model(model_path);
    model << std::setprecision(12);
    model << "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"" << tag
          << "\">\n\n"
          << "  <include href=\""
          << std::filesystem::absolute(parameters.cell_types_path).string() << "\"/>\n\n";

    model << "  <" << cell_type << " id=\"" << tag << "Cell\""
          << glif_torus_cell_attributes(parameters.glif_index) << "/>\n\n";

    model << "  <alphaCurrentSynapse id=\"" << tag << "Synapse\" tau=\""
          << parameters.synapse_tau_seconds << " s\" ibase=\""
          << parameters.synapse_ibase_amperes << " A\"/>\n\n";

    model << "  <pulseGenerator id=\"" << tag << "Background\" delay=\"0 s\" duration=\""
          << parameters.simulation_seconds << " s\" amplitude=\""
          << parameters.background_current_amperes << " A\"/>\n";
    // A wandering patch needs one generator per period, each live only for its own
    // window; every other pattern needs a single one. All of them are document-scope
    // elements -- a pulseGenerator inside <network> is not a component the model declares,
    // and the inputList that names it fails to resolve.
    const s64 period_count = parameters.drive_pattern == TorusDrivePattern::WanderingPatch
            ? std::max<s64>(1, (s64)(parameters.simulation_seconds /
                                     parameters.pattern_period_seconds))
            : 1;

    if (period_count == 1) {
        model << "  <pulseGenerator id=\"" << tag << "Driven\" delay=\"0 s\" duration=\""
              << parameters.simulation_seconds << " s\" amplitude=\""
              << parameters.driven_extra_current_amperes << " A\"/>\n";
    } else {
        for (s64 period = 0; period < period_count; period += 1) {
            model << "  <pulseGenerator id=\"" << tag << "Driven" << period
                  << "\" delay=\"" << (double)period * parameters.pattern_period_seconds
                  << " s\" duration=\"" << parameters.pattern_period_seconds
                  << " s\" amplitude=\"" << parameters.driven_extra_current_amperes
                  << " A\"/>\n";
        }
    }
    model << "\n";

    model << "  <network id=\"" << tag << "Network\">\n";
    model << "    <population id=\"" << tag << "Population\" component=\"" << tag
          << "Cell\" size=\"" << parameters.neuron_count() << "\"/>\n\n";

    // Background on every cell. This is one <input> per neuron, which is the one part of
    // the document that still scales with the population -- inputs have no topology helper
    // the way connections do.
    model << "    <inputList id=\"" << tag << "BackgroundInput\" component=\"" << tag
          << "Background\" population=\"" << tag << "Population\">\n";
    for (s64 index = 0; index < parameters.neuron_count(); index += 1) {
        model << "      <input id=\"" << index << "\" target=\"../" << tag << "Population["
              << index << "]\" destination=\"synapses\"/>\n";
    }
    model << "    </inputList>\n\n";

    // One inputList per period, each naming its own generator, which is how a static
    // document expresses a stimulus that moves.
    for (s64 period = 0; period < period_count; period += 1) {
        const Vector<s64> driven = driven_cells(parameters, period);
        if (driven.empty()) continue;

        model << "    <inputList id=\"" << tag << "DrivenInput" << period
              << "\" component=\"" << tag << "Driven"
              << (period_count > 1 ? std::to_string(period) : String())
              << "\" population=\"" << tag << "Population\">\n";
        for (s64 index : driven) {
            model << "      <input id=\"" << index << "\" target=\"../" << tag
                  << "Population[" << index << "]\" destination=\"synapses\"/>\n";
        }
        model << "    </inputList>\n";
    }

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
            "    <Include file=\"" << tag << ".nml\"/>\n\n";
    lems << "    <Simulation id=\"sim1\" length=\"" << parameters.simulation_seconds
         << "s\" step=\"" << parameters.step_seconds << "s\" target=\"" << tag
         << "Network\">\n";
    lems << "    </Simulation>\n\n    <Target component=\"sim1\"/>\n</Lems>\n";
    lems.close();

    return lems_path;
}

} // namespace spikecorec::examples
