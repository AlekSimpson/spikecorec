#pragma once

// A GLIF sheet on a square torus. The document declares the cells, the synapse and the
// stimulus; the connectivity comes from the engine's own square_torus() through the
// topology constructor, which is what makes the side length a parameter rather than a
// rewrite. A 1,024 x 1,024 torus is four million edges: as <connection> elements that is
// four million lines of XML parsed into four million ComponentInstances before a tick runs.
//
// Shared by the ten torus demos. Each names itself, and that name decides every path it
// writes, so no two demos overwrite each other's model.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

#include "spikecorec/core/types.h"
#include "spikecorec/core/topologies.h"
#include "glif_cell_attributes.h"
#include "demo_support.h"

namespace spikecorec::demos {

// How the sheet is driven. The step patterns hold every cell just below rheobase and take
// a subset above it; the scattered one abandons steps entirely.
enum class TorusDrive {
    // Driven cells spread evenly across the sheet. Stationary once settled.
    UniformBackground,
    // A disc at the centre. Activity has to spread outward through the grid.
    CentrePatch,
    // The disc moves every pattern_period_seconds, so the network is chased.
    WanderingPatch,
    // Every cell driven by its own scattered spike train. Nothing starts at the same
    // instant, so nothing stays in step -- a constant current switched on at t = 0 for
    // every cell of a symmetric grid of identical cells has a symmetric solution, and the
    // network sits in it, which is what a million cells in lockstep looked like. Random
    // arrival times break that, and are also how a real network is driven.
    ScatteredSpikes,
    // Only a handful of cells receive anything from outside. Everything else fires purely
    // on what arrives from its four neighbours, so the sheet behaves as an excitable
    // medium: each seed launches a front that spreads, and fronts annihilate where they
    // meet because the cells between them are refractory.
    //
    // This only works if one presynaptic spike can fire its target. A cell here has four
    // neighbours and no background current, so it sits at EL and needs the whole
    // EL-to-threshold swing from a single arrival -- which makes the synapse strength a
    // load-bearing parameter rather than a taste one. See sparse_seed_ibase_amperes.
    SparseSeeds,
};

struct TorusDemoParameters {
    s32 glif_index = 1;
    s64 side_length = 64;

    TorusDrive drive = TorusDrive::ScatteredSpikes;
    f64 pattern_period_seconds = 0.2;
    f64 driven_fraction = 0.2;

    // How long a cell is held after firing. The default matches every other demo; the
    // sparse-seed one widens it, because a 5 ms tail on a sheet where fronts move a cell
    // per millisecond lets activity re-enter and saturate.
    String refractory_period = "5ms";

    f64 synapse_ibase_amperes = 30e-12;
    f64 synapse_tau_seconds = 5e-3;
    f64 connection_weight = 1.0;
    f64 connection_delay_seconds = 1e-3;

    f64 background_current_amperes = 190e-12;
    f64 driven_extra_current_amperes = 45e-12;

    // Step drives only. Identical cells under an identical step never decorrelate, so the
    // background is fanned across this many generators, assigned by a prime stride. Prime,
    // and not a divisor of a power-of-two side length: 16 variants on a 1024-wide grid puts
    // every variant on the same columns forever and turns decorrelation into stripes.
    s64 background_variant_count = 251;
    f64 background_spread_fraction = 0.06;

    // ScatteredSpikes only. A train declares no amplitude, so each of its events delivers
    // whatever it takes to fire the cell it lands on -- the engine derives that from the
    // target's own capacitance and threshold.
    s64 scattered_train_count = 997;
    f64 scattered_input_rate_hertz = 45.0;
    u64 scattered_seed = 20260814;

    // SparseSeeds only. How many cells hear from outside at all, and how often.
    s64 seed_neuron_count = 10;
    f64 seed_input_rate_hertz = 20.0;

    f64 simulation_seconds = 1.0;
    f64 step_seconds = 1e-4;

    [[nodiscard]] s64 neuron_count() const { return side_length * side_length; }
};

// Which cells the extra drive lands on, for one step pattern at one moment.
inline Vector<s64> torus_driven_cells(const TorusDemoParameters &parameters,
                                      s64 pattern_index) {
    const s64 side = parameters.side_length;
    const s64 count = parameters.neuron_count();
    Vector<s64> driven;

    if (parameters.drive == TorusDrive::UniformBackground) {
        const s64 driven_count = (s64)(parameters.driven_fraction * (f64)count);
        const s64 stride = driven_count > 0 ? std::max<s64>(1, count / driven_count) : count;
        for (s64 index = 0; index < count; index += stride) driven.push_back(index);
        return driven;
    }

    const f64 radius = std::sqrt(parameters.driven_fraction * (f64)count / 3.14159265358979);

    s64 centre_row = side / 2;
    s64 centre_column = side / 2;
    if (parameters.drive == TorusDrive::WanderingPatch) {
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

inline void write_torus_lems(const String &demo_name,
                             const TorusDemoParameters &parameters,
                             const String &directory = demo_directory()) {
    std::ofstream lems(directory + "/LEMS_" + demo_name + ".xml");
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
}

// Writes build/demos/<demo_name>.nml and its LEMS document; returns the LEMS path. The
// document declares no connections at all.
inline String write_torus_model(const String &demo_name,
                                const TorusDemoParameters &parameters,
                                const String &directory = demo_directory()) {
    const String model_path = demo_path(demo_name, ".nml", directory);
    const String lems_path = directory + "/LEMS_" + demo_name + ".xml";
    const String cell_type = glif_cell_type_name(parameters.glif_index);

    std::ofstream model(model_path);
    model << std::setprecision(12);
    model << "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"" << demo_name
          << "\">\n\n"
          << "  <include href=\""
          << std::filesystem::absolute(glif_cell_types_path()).string() << "\"/>\n\n";

    model << "  <" << cell_type << " id=\"torusCell\""
          << glif_cell_attributes(parameters.glif_index, parameters.refractory_period)
          << "/>\n\n";
    model << "  <alphaCurrentSynapse id=\"torusSynapse\" tau=\""
          << parameters.synapse_tau_seconds << " s\" ibase=\""
          << parameters.synapse_ibase_amperes << " A\"/>\n\n";

    if (parameters.drive == TorusDrive::SparseSeeds) {
        std::mt19937_64 generator(parameters.scattered_seed);
        std::uniform_real_distribution<f64> arrival(0.0, parameters.simulation_seconds);
        std::uniform_int_distribution<s64> anywhere(0, parameters.neuron_count() - 1);

        // Distinct cells, drawn anywhere on the sheet.
        Set<s64> seeds;
        while ((s64)seeds.size() < parameters.seed_neuron_count) seeds.insert(anywhere(generator));

        const s64 events_per_seed = std::max<s64>(
                1, (s64)std::llround(parameters.seed_input_rate_hertz *
                                     parameters.simulation_seconds));

        Vector<s64> ordered_seeds(seeds.begin(), seeds.end());
        std::sort(ordered_seeds.begin(), ordered_seeds.end());

        for (usize index = 0; index < ordered_seeds.size(); index += 1) {
            Vector<f64> times;
            for (s64 event = 0; event < events_per_seed; event += 1) {
                times.push_back(arrival(generator));
            }
            std::sort(times.begin(), times.end());

            model << "  <spikeArray id=\"seedTrain" << index << "\">\n";
            for (usize event = 0; event < times.size(); event += 1) {
                model << "    <spike id=\"" << event << "\" time=\"" << times[event]
                      << " s\"/>\n";
            }
            model << "  </spikeArray>\n";
        }

        model << "\n  <network id=\"network\">\n";
        model << "    <population id=\"torusPopulation\" component=\"torusCell\" size=\""
              << parameters.neuron_count() << "\"/>\n\n";

        // One inputList per seed, each naming exactly one cell. Every other cell in the
        // sheet appears in no inputList at all and hears only from its neighbours.
        for (usize index = 0; index < ordered_seeds.size(); index += 1) {
            model << "    <inputList id=\"seedInput" << index << "\" component=\"seedTrain"
                  << index << "\" population=\"torusPopulation\">\n"
                  << "      <input id=\"0\" target=\"../torusPopulation["
                  << ordered_seeds[index] << "]\" destination=\"synapses\"/>\n"
                  << "    </inputList>\n";
        }

        model << "  </network>\n</neuroml>\n";
        model.close();
        write_torus_lems(demo_name, parameters, directory);
        return lems_path;
    }

    if (parameters.drive == TorusDrive::ScatteredSpikes) {
        std::mt19937_64 generator(parameters.scattered_seed);
        std::uniform_real_distribution<f64> arrival(0.0, parameters.simulation_seconds);

        const s64 train_count = std::max<s64>(1, parameters.scattered_train_count);
        const s64 events_per_train = std::max<s64>(
                1, (s64)std::llround(parameters.scattered_input_rate_hertz *
                                     parameters.simulation_seconds));

        for (s64 train = 0; train < train_count; train += 1) {
            Vector<f64> times;
            for (s64 event = 0; event < events_per_train; event += 1) {
                times.push_back(arrival(generator));
            }
            std::sort(times.begin(), times.end());

            model << "  <spikeArray id=\"train" << train << "\">\n";
            for (usize event = 0; event < times.size(); event += 1) {
                model << "    <spike id=\"" << event << "\" time=\"" << times[event]
                      << " s\"/>\n";
            }
            model << "  </spikeArray>\n";
        }
        model << "\n  <network id=\"network\">\n";
        model << "    <population id=\"torusPopulation\" component=\"torusCell\" size=\""
              << parameters.neuron_count() << "\"/>\n\n";

        // A prime stride, so cells sharing a train are scattered across the sheet rather
        // than lined up in a column.
        for (s64 train = 0; train < train_count; train += 1) {
            model << "    <inputList id=\"trainInput" << train << "\" component=\"train"
                  << train << "\" population=\"torusPopulation\">\n";
            for (s64 index = train; index < parameters.neuron_count(); index += train_count) {
                model << "      <input id=\"" << index
                      << "\" target=\"../torusPopulation[" << index
                      << "]\" destination=\"synapses\"/>\n";
            }
            model << "    </inputList>\n";
        }
        model << "  </network>\n</neuroml>\n";
        model.close();
        write_torus_lems(demo_name, parameters, directory);
        return lems_path;
    }

    // ── step drives ──────────────────────────────────────────────────────────────
    const s64 variant_count = std::max<s64>(1, parameters.background_variant_count);
    for (s64 variant = 0; variant < variant_count; variant += 1) {
        const f64 offset = variant_count == 1
                ? 0.0
                : parameters.background_spread_fraction *
                  (2.0 * (f64)variant / (f64)(variant_count - 1) - 1.0);
        model << "  <pulseGenerator id=\"background" << variant
              << "\" delay=\"0 s\" duration=\"" << parameters.simulation_seconds
              << " s\" amplitude=\""
              << parameters.background_current_amperes * (1.0 + offset) << " A\"/>\n";
    }

    const s64 period_count = parameters.drive == TorusDrive::WanderingPatch
            ? std::max<s64>(1, (s64)(parameters.simulation_seconds /
                                     parameters.pattern_period_seconds))
            : 1;

    // Document scope: a pulseGenerator inside <network> is not a component the model
    // declares, and the inputList that names it fails to resolve.
    for (s64 period = 0; period < period_count; period += 1) {
        model << "  <pulseGenerator id=\"driven" << period << "\" delay=\""
              << (f64)period * (period_count > 1 ? parameters.pattern_period_seconds : 0.0)
              << " s\" duration=\""
              << (period_count > 1 ? parameters.pattern_period_seconds
                                   : parameters.simulation_seconds)
              << " s\" amplitude=\"" << parameters.driven_extra_current_amperes
              << " A\"/>\n";
    }
    model << "\n  <network id=\"network\">\n";
    model << "    <population id=\"torusPopulation\" component=\"torusCell\" size=\""
          << parameters.neuron_count() << "\"/>\n\n";

    for (s64 variant = 0; variant < variant_count; variant += 1) {
        model << "    <inputList id=\"backgroundInput" << variant << "\" component=\"background"
              << variant << "\" population=\"torusPopulation\">\n";
        for (s64 index = variant; index < parameters.neuron_count(); index += variant_count) {
            model << "      <input id=\"" << index << "\" target=\"../torusPopulation["
                  << index << "]\" destination=\"synapses\"/>\n";
        }
        model << "    </inputList>\n";
    }

    for (s64 period = 0; period < period_count; period += 1) {
        const Vector<s64> driven = torus_driven_cells(parameters, period);
        if (driven.empty()) continue;

        model << "    <inputList id=\"drivenInput" << period << "\" component=\"driven"
              << period << "\" population=\"torusPopulation\">\n";
        for (s64 index : driven) {
            model << "      <input id=\"" << index << "\" target=\"../torusPopulation["
                  << index << "]\" destination=\"synapses\"/>\n";
        }
        model << "    </inputList>\n";
    }

    model << "  </network>\n</neuroml>\n";
    model.close();
    write_torus_lems(demo_name, parameters, directory);
    return lems_path;
}

} // namespace spikecorec::demos
