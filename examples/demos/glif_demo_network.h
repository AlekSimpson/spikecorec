#pragma once

#include <algorithm>
#include <cmath>

#include "../glif_torus_network.h"

// ── Shared parts of the GLIF demos ──────────────────────────────────────────────────────────
//
// The nine programs in examples/ are each built around one claim and sized to make that claim
// cheaply -- an 8x8 torus, 64 cells, a number printed at the end. A demo has a different job:
// it exists to be WATCHED. So the network here is large enough that a spreading wavefront is
// motion on a screen rather than a handful of cells changing shade, and the run is long enough
// that the wavefront crosses the population, wraps, and meets itself before the recording ends.
//
// Everything else is the examples' own machinery, unchanged: the same GLIF ComponentType
// declarations, the same membrane parameters, the same current-based alphaCurrentSynapse, the
// same torus wiring. A demo is the examples' network at demo scale, not a second model.
//
// What this header adds on top is the reporting a demo needs and an example does not: whether
// the run actually produced activity, spread over the population and spread over time. That
// question is the whole point of rendering a video, and it is answered here in numbers as well,
// so a demo that renders an empty video says so in its own console output.

namespace spikecorec::examples::demos {

// ── the demo network ────────────────────────────────────────────────────────────────────────

// 48 on a side is 2304 neurons and 9216 connections. It is chosen against the video, not against
// the engine: at 48 the wavefront needs about 125 simulated milliseconds to reach the far side,
// which reads as motion, and one neuron is still several pixels wide in render_spire_video.py's
// fixed 600-pixel frame. A 128-wide grid would run just as happily and render as noise.
constexpr s64 DEMO_SIDE_LENGTH = 48;

// Long enough for the wavefront to cross the torus, wrap around the edges and collide with
// itself at the antipode, with a stretch of settled network-wide firing after it.
constexpr const char *DEMO_SIMULATION_LENGTH = "250ms";

// The examples drive neuron 0 -- a corner -- because their first-spike grid reads most clearly
// with the wavefront starting in one. A video reads better the other way round: driven from the
// middle, the wavefront is a diamond expanding symmetrically in all four directions at once,
// and the torus wraparound shows up as four fronts converging on the corners.
inline s64 demo_driven_neuron_index(s64 side_length) {
    return torus_neuron_index(side_length / 2, side_length / 2, side_length);
}

inline TorusNetworkOptions demo_torus_options(s64 side_length, f64 synapse_peak_current) {
    TorusNetworkOptions torus;
    torus.side_length = side_length;
    torus.synapse_peak_current = synapse_peak_current;
    torus.driven_neuron_index = demo_driven_neuron_index(side_length);
    torus.simulation_length = DEMO_SIMULATION_LENGTH;

    // The examples wait 20ms before switching the drive on. At demo scale that is 200 ticks of
    // a completely dead recording at the head of the video, so the demo waits 5ms instead --
    // just enough that a viewer sees the network at rest before anything happens.
    torus.stimulus_delay = "5ms";
    return torus;
}

// ── choosing a variant from the command line ────────────────────────────────────────────────

inline GlifVariant glif_variant_from_name(const std::string &variant_name) {
    if (variant_name == "glif1") return GlifVariant::Glif1;
    if (variant_name == "glif2") return GlifVariant::Glif2;
    if (variant_name == "glif3") return GlifVariant::Glif3;
    if (variant_name == "glif4") return GlifVariant::Glif4;
    if (variant_name == "glif5") return GlifVariant::Glif5;
    throw std::runtime_error("unknown --variant '" + variant_name +
                             "' (expected one of glif1 glif2 glif3 glif4 glif5)");
}

inline std::string glif_variant_short_name(GlifVariant variant) {
    switch (variant) {
        case GlifVariant::Glif1: return "glif1";
        case GlifVariant::Glif2: return "glif2";
        case GlifVariant::Glif3: return "glif3";
        case GlifVariant::Glif4: return "glif4";
        case GlifVariant::Glif5: return "glif5";
    }
    return "glif1";
}

// ── did anything actually happen? ───────────────────────────────────────────────────────────

// A run can produce a video showing nothing in two different ways, and they need separate
// answers: every spike coming from one cell (nothing propagated), and every spike landing in
// one instant (a flash, not activity). Both are read off the same SpikeObservation here, so a
// demo reports them without the caller having to go back to the .spire files.
inline void print_activity_timeline(const SpikeObservation &observation, f64 seconds_per_tick,
                                    s64 bin_count) {
    Vector<s64> spikes_per_bin((usize)bin_count, 0);
    const s64 ticks_per_bin =
            std::max<s64>(1, (observation.tick_count + bin_count - 1) / bin_count);

    for (const Vector<s64> &spike_ticks : observation.spike_ticks_per_neuron) {
        for (s64 tick : spike_ticks) {
            const s64 bin_index = std::min(bin_count - 1, tick / ticks_per_bin);
            spikes_per_bin[(usize)bin_index] += 1;
        }
    }

    s64 busiest_bin_total = 0;
    s64 bins_with_activity = 0;
    for (s64 bin_total : spikes_per_bin) {
        busiest_bin_total = std::max(busiest_bin_total, bin_total);
        if (bin_total > 0) bins_with_activity += 1;
    }

    const f64 milliseconds_per_bin = (f64)ticks_per_bin * seconds_per_tick * 1000.0;
    std::cout << "  Spikes per " << std::fixed << std::setprecision(0) << milliseconds_per_bin
              << "ms of the run (" << bins_with_activity << " of " << bin_count
              << " intervals carry activity)\n";

    const s64 bar_width = 40;
    for (s64 bin_index = 0; bin_index < bin_count; ++bin_index) {
        const s64 bin_total = spikes_per_bin[(usize)bin_index];
        const s64 bar_length =
                busiest_bin_total == 0 ? 0 : (bin_total * bar_width) / busiest_bin_total;

        std::cout << "    " << std::setw(5) << std::fixed << std::setprecision(0)
                  << (f64)bin_index * milliseconds_per_bin << "ms |" << std::string((usize)bar_length, '#')
                  << std::string((usize)(bar_width - bar_length), ' ') << "| " << bin_total << "\n";
    }
}

// The two spatial numbers behind "the wavefront crossed the population": when the first neuron
// fired, and when the LAST neuron to join in fired for the first time. The gap between them is
// how long the front took to reach everything, which is the thing the video shows.
inline void print_wavefront_spread(const SpikeObservation &observation, f64 seconds_per_tick,
                                   s64 total_neuron_count) {
    s64 earliest_first_spike = -1;
    s64 latest_first_spike = -1;
    for (s64 first_tick : observation.first_spike_tick_per_neuron) {
        if (first_tick < 0) continue;
        if (earliest_first_spike < 0 || first_tick < earliest_first_spike) {
            earliest_first_spike = first_tick;
        }
        latest_first_spike = std::max(latest_first_spike, first_tick);
    }

    if (earliest_first_spike < 0) {
        std::cout << "  No neuron fired at all -- this run has nothing to render.\n";
        return;
    }

    const f64 crossing_milliseconds =
            (f64)(latest_first_spike - earliest_first_spike) * seconds_per_tick * 1000.0;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  Wavefront: first spike at tick " << earliest_first_spike << " ("
              << (f64)earliest_first_spike * seconds_per_tick * 1000.0 << "ms); the last of the "
              << observation.spiking_neuron_count() << " neurons to join fired first at tick "
              << latest_first_spike << " (" << crossing_milliseconds
              << "ms later).\n";

    if (observation.spiking_neuron_count() < total_neuron_count) {
        std::cout << "  " << total_neuron_count - observation.spiking_neuron_count()
                  << " neurons never fired.\n";
    }
}

// ── what the variant itself adds ────────────────────────────────────────────────────────────

// Each GLIF variant differs from GLIF1 by exactly one mechanism, and each mechanism leaves its
// own mark on the driven cell. This prints that mark, so a demo says which variant it ran in
// terms of an observed number and not just a type name.
inline void print_variant_signature(GlifVariant variant, const SpikeEngine &engine,
                                    const SpikeObservation &observation, s64 driven_neuron_index,
                                    const Vector<f32> &potential_after_each_spike) {
    const Vector<s64> &driven_spike_ticks =
            observation.spike_ticks_per_neuron[(usize)driven_neuron_index];

    std::cout << "  Driven cell (neuron " << driven_neuron_index << "), "
              << driven_spike_ticks.size() << " spikes\n";
    print_inter_spike_intervals(driven_spike_ticks, engine.network_details.step_dt);

    switch (variant) {
        case GlifVariant::Glif1:
            std::cout << "    GLIF1 carries no adaptation mechanism of any kind, so these settle "
                         "to a\n    constant. Every other variant's intervals are read against "
                         "this one.\n";
            break;

        case GlifVariant::Glif2: {
            std::cout << std::fixed << std::setprecision(1);
            std::cout << "    GLIF2 resets to vreset + resetScale * (v - vth) rather than flat to "
                         "vreset.\n    Microvolts above vreset (-70mV) after each of the first "
                         "twelve resets:\n     ";
            for (usize spike_index = 0;
                 spike_index < potential_after_each_spike.size() && spike_index < 12;
                 ++spike_index) {
                std::cout << " " << (potential_after_each_spike[spike_index] + 0.070f) * 1e6f;
            }
            std::cout << "\n    GLIF1 would print 0.0 every time; every one of these is strictly "
                         "positive.\n";
            break;
        }

        case GlifVariant::Glif3:
            std::cout << std::fixed << std::setprecision(0);
            std::cout << "    GLIF3 adds two after-spike currents. Both step down on every spike "
                         "and\n    decay back towards zero, so each spike makes the next one "
                         "harder to reach.\n    Driven cell at the end of the run (pA): asc1="
                      << read_cell_state(engine, driven_neuron_index, "asc1") * 1e12f
                      << "  asc2="
                      << read_cell_state(engine, driven_neuron_index, "asc2") * 1e12f << "\n";
            break;

        case GlifVariant::Glif4:
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "    GLIF4 makes the threshold itself a state variable: theta jumps on "
                         "every\n    spike and relaxes back towards thetaInf (-50.00mV) in "
                         "between.\n    Driven cell's theta at the end of the run: "
                      << read_cell_state(engine, driven_neuron_index, "theta") * 1000.0f
                      << "mV\n";
            break;

        case GlifVariant::Glif5:
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "    GLIF5 runs GLIF3's after-spike currents AND GLIF4's adaptive "
                         "threshold at\n    once, so both mechanisms push the firing rate the "
                         "same way.\n    Driven cell at the end of the run: theta="
                      << read_cell_state(engine, driven_neuron_index, "theta") * 1000.0f
                      << "mV  asc1=" << std::setprecision(0)
                      << read_cell_state(engine, driven_neuron_index, "asc1") * 1e12f
                      << "pA  asc2="
                      << read_cell_state(engine, driven_neuron_index, "asc2") * 1e12f << "pA\n";
            break;
    }
}

} // namespace spikecorec::examples::demos
