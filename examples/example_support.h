#pragma once

// The backend header has to come first. metal-cpp declares its own NS::String, which
// collides with the engine's `String` alias when <Metal/Metal.hpp> is pulled in later
// through a transitive include. Every test file in tests/ opens the same way.
#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/engine.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/types.h"

// ── Shared scaffolding for the spikecorec examples ──────────────────────────────────────
//
// Every example here drives the same, very short surface:
//
//     SpikeEngine engine(model_file_path, /*enable_hebbian_learning=*/false);
//     for (s64 tick = 0; tick < engine.lifetime; ++tick) engine.step_simulation(tick);
//     engine.shutdown();
//
// The model -- cell dynamics, network wiring, stimulus, recording -- lives entirely in a
// NeuroML/LEMS file. What each example adds on top is only the reading: which state
// variable it watches, what it prints, and what that number means. That part stays inline
// in each example file, because it is the part worth reading.
//
// The helpers below are only what is identical across all of them.

namespace spikecorec::examples {

// One nanoampere, in amperes.
//
// Every quantity in these models is SI, because that is what NeuroML resolves to and what the
// engine holds: a `<pulseGenerator amplitude="0.6nA">` is 6e-10 A by the time it reaches the
// synaptic accumulator, and an `<alphaCurrentSynapse ibase="1nA">` is 1e-9 A when its arrival
// handler steps the synapse. A synaptic current is naturally 1e-9-ish, which is unreadable
// written out, so the examples spell those magnitudes as `1.0 * NANOAMPERE` rather than as
// `1e-9`, and emit them into the model with their unit attached (`1nA`).
//
// Nothing about the storage needs the numbers to be large: WeightMatrix::set_edge_weight
// stores a weight exactly, at any magnitude (see examples/README.md).
constexpr f64 NANOAMPERE = 1e-9;

// A current in amperes, spelled the way NeuroML wants to read it: a magnitude with its unit
// attached, e.g. 1e-9 -> "1nA". Every current these examples put into a model goes through
// here, so a dimensionless bare number can never reach an attribute declaring
// dimension="current".
inline std::string nanoampere_attribute(f64 amperes) {
    std::ostringstream text;
    // The `+ 0.0` normalises a negated zero (IEEE: -0.0 + 0.0 is +0.0), so a switched-off
    // current renders as "0nA" rather than "-0nA".
    text << (amperes / NANOAMPERE + 0.0) << "nA";
    return text.str();
}

// Owns the GPU context for the whole program.
//
// DECLARE THIS BEFORE ANY LOCAL THAT HOLDS A GPU BUFFER. release_gpu_resources() frees every
// buffer the backend handed out, so it has to run AFTER the last object holding one
// (SpikeEngine, WeightMatrix, ...). Locals destruct in reverse declaration order, so
// declaring the guard first makes it destruct last. Calling release_gpu_resources() by hand
// at the end of main() instead frees those buffers while their owners are still alive, and
// the destructors then double-free -- a crash at exit, after a run that otherwise looked
// fine.
//
// The one thing that goes AHEAD of it is configure_logging: initializing the GPU context
// builds the engine's logger singleton, and the singleton takes its level from whoever
// creates it first.
class GpuContextScope {
public:
    GpuContextScope() { initialize_gpu_context(); }
    ~GpuContextScope() { release_gpu_resources(); }

    GpuContextScope(const GpuContextScope &) = delete;
    GpuContextScope &operator=(const GpuContextScope &) = delete;
};

// The flags every example understands. Not all of them are meaningful to every example --
// `torus_side_length` only means something to the torus programs -- but parsing them in one
// place keeps each main() to its own subject.
struct ExampleOptions {
    // Ticks to run. Negative means "whatever the model's own <Simulation length=.../step=...>
    // works out to", which is what engine.lifetime already holds.
    s64 tick_count = -1;

    // Torus edge length; the generated network holds side_length^2 neurons.
    s64 torus_side_length = 8;

    // How hard one arriving spike drives its target, in AMPERES: the `ibase` of the
    // alphaCurrentSynapse every projection here names. The synapse's OWN dynamics are run, so
    // this is the PEAK of the alpha current profile a single arrival injects, reached one
    // `tau` later -- not a single-tick impulse. See the "synaptic strength" note in
    // examples/README.md.
    //
    // `--synapse-current` is typed in nanoamps because 1 is easier to type and read than 1e-9,
    // and parse_example_options converts it here, once.
    f64 synapse_peak_current = 0.0;

    // Where a model's <OutputFile> recordings are written.
    std::string recording_directory = "recordings";

    // The 0/1 bit sequence discrete_spike_input_example drives its cells with. Brackets,
    // commas and spaces are ignored, so both `0001001010` and `[0,0,0,1,0,0,1,0,1,0]` read
    // the same. Empty means "the example's own default".
    std::string input_pattern;

    // Engine log output is quieted to warnings unless this is set.
    bool verbose = false;
};

inline void print_common_option_help(std::ostream &output) {
    output << "  --ticks <count>       run this many ticks instead of the model's own length\n"
              "  --synapse-current <nA>  peak current one arriving spike injects (the "
              "synapse's ibase)\n"
              "  --side <length>       torus edge length; the network holds length^2 neurons\n"
              "  --record-dir <path>   where .spire recordings are written\n"
              "  --pattern <bits>      the 0/1 sequence discrete_spike_input_example drives with\n"
              "  --verbose             show engine log output\n";
}

// Throws on an unrecognised flag rather than ignoring it: a mistyped `--tikcs 100` that
// silently ran the default length would be reported as a result.
//
// `default_synapse_peak_current` is in AMPERES (the caller writes `1.0 * NANOAMPERE`); the
// `--synapse-current` flag is typed in nanoamps and converted here.
inline ExampleOptions parse_example_options(int argument_count, char **argument_values,
                                            f64 default_synapse_peak_current) {
    ExampleOptions options;
    options.synapse_peak_current = default_synapse_peak_current;

    for (int index = 1; index < argument_count; ++index) {
        const std::string flag = argument_values[index];
        const bool has_value = index + 1 < argument_count;

        if (flag == "--verbose") {
            options.verbose = true;
        } else if (flag == "--ticks" && has_value) {
            options.tick_count = std::atoll(argument_values[++index]);
        } else if (flag == "--side" && has_value) {
            options.torus_side_length = std::atoll(argument_values[++index]);
        } else if (flag == "--synapse-current" && has_value) {
            options.synapse_peak_current = std::atof(argument_values[++index]) * NANOAMPERE;
        } else if (flag == "--record-dir" && has_value) {
            options.recording_directory = argument_values[++index];
        } else if (flag == "--pattern" && has_value) {
            options.input_pattern = argument_values[++index];
        } else {
            throw std::runtime_error("unrecognised argument '" + flag + "'");
        }
    }

    return options;
}

// Quiets the engine's own console logging unless --verbose was passed. The logger is a
// process-wide singleton built on first use, so this has to run before the engine is
// constructed to have any effect.
inline void configure_logging(const ExampleOptions &options) {
    log::make_logger(options.verbose ? log::LogLevel::info : log::LogLevel::warn);
}

// A scratch directory holding the model files an example generates.
//
// Several examples build their network rather than checking it in, because a 64-neuron
// torus has 256 connections and hand-typing that is not reasonable. The generated document
// goes through the identical parse path a checked-in one does; nothing is special-cased.
class GeneratedModelDirectory {
public:
    explicit GeneratedModelDirectory(const std::string &example_name) {
        root_ = std::filesystem::temp_directory_path() / "spikecorec_examples" / example_name;
        std::filesystem::remove_all(root_);
        std::filesystem::create_directories(root_);
    }

    ~GeneratedModelDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(root_, ignored);
    }

    GeneratedModelDirectory(const GeneratedModelDirectory &) = delete;
    GeneratedModelDirectory &operator=(const GeneratedModelDirectory &) = delete;

    // Returns the absolute path of the written file.
    std::string write(const std::string &relative_name, const std::string &contents) const {
        const std::filesystem::path destination = root_ / relative_name;
        std::ofstream file(destination);
        file << contents;
        file.close();
        return destination.string();
    }

private:
    std::filesystem::path root_;
};

// ── reading cell state ──────────────────────────────────────────────────────────────────
//
// cell_state is sectioned by cell TYPE, with each neuron holding one float per state
// variable its type declares. cell_state_base resolves a global neuron index to the start of
// its own slot; the variable's position inside that slot is its position in the type's
// declaration order.

inline s64 state_variable_slot(const SpikeEngine &engine, s64 neuron_index,
                               const std::string &variable_name) {
    const s64 type_index = engine.cell_type_index.get_contents()[neuron_index];
    const Vector<String> &state_variable_names =
            engine.network_details.cell_types[(usize)type_index].state_variable_names;

    for (usize slot = 0; slot < state_variable_names.size(); ++slot) {
        if (state_variable_names[slot] == variable_name) return (s64)slot;
    }

    throw std::runtime_error("neuron " + std::to_string(neuron_index) + " has no state variable '" +
                             variable_name + "'");
}

inline f32 read_cell_state(const SpikeEngine &engine, s64 neuron_index,
                           const std::string &variable_name) {
    const s64 state_base = engine.cell_state_base.get_contents()[neuron_index];
    return engine.cell_state
            .get_contents()[state_base + state_variable_slot(engine, neuron_index, variable_name)];
}

// ── running a simulation ────────────────────────────────────────────────────────────────

// Everything a run's spikes say, gathered once so an example can report on them without
// re-running. Sized for the networks here (tens of neurons, thousands of ticks).
struct SpikeObservation {
    Vector<s64> spike_count_per_neuron;
    Vector<s64> first_spike_tick_per_neuron; // -1 for a neuron that never fired
    Vector<Vector<s64>> spike_ticks_per_neuron;
    s64 total_spike_count = 0;
    s64 tick_count = 0;

    s64 spiking_neuron_count() const {
        s64 count = 0;
        for (s64 first_tick : first_spike_tick_per_neuron) {
            if (first_tick >= 0) count += 1;
        }
        return count;
    }
};

// Runs `tick_count` ticks (or the model's own lifetime when `tick_count` is negative),
// recording every emission. `per_tick` runs after each step, for an example that needs to
// watch something the spike record does not carry.
template <typename PerTickObserver>
SpikeObservation run_simulation(SpikeEngine &engine, s64 tick_count,
                                PerTickObserver per_tick) {
    const s64 ticks_to_run = tick_count < 0 ? engine.lifetime : tick_count;

    SpikeObservation observation;
    observation.tick_count = ticks_to_run;
    observation.spike_count_per_neuron.assign((usize)engine.total_neuron_count, 0);
    observation.first_spike_tick_per_neuron.assign((usize)engine.total_neuron_count, -1);
    observation.spike_ticks_per_neuron.resize((usize)engine.total_neuron_count);

    const s32 *spike_flags = engine.spike_flags.get_contents();

    for (s64 tick = 0; tick < ticks_to_run; ++tick) {
        engine.step_simulation(tick);

        for (s64 neuron_index = 0; neuron_index < engine.total_neuron_count; ++neuron_index) {
            if (spike_flags[neuron_index] == 0) continue;

            observation.total_spike_count += 1;
            observation.spike_count_per_neuron[(usize)neuron_index] += 1;
            observation.spike_ticks_per_neuron[(usize)neuron_index].push_back(tick);
            if (observation.first_spike_tick_per_neuron[(usize)neuron_index] < 0) {
                observation.first_spike_tick_per_neuron[(usize)neuron_index] = tick;
            }
        }

        per_tick(tick);
    }

    return observation;
}

inline SpikeObservation run_simulation(SpikeEngine &engine, s64 tick_count) {
    return run_simulation(engine, tick_count, [](s64) {});
}

// ── console output ──────────────────────────────────────────────────────────────────────

inline void print_heading(const std::string &title) {
    std::cout << "\n" << title << "\n" << std::string(title.size(), '=') << "\n";
}

// The intervals between consecutive spikes, in ticks. A lengthening series is spike-frequency
// adaptation; a flat one is a cell with no adaptation mechanism at all.
inline Vector<s64> inter_spike_intervals(const Vector<s64> &spike_ticks) {
    Vector<s64> intervals;
    for (usize slot = 1; slot < spike_ticks.size(); ++slot) {
        intervals.push_back(spike_ticks[slot] - spike_ticks[slot - 1]);
    }
    return intervals;
}

inline void print_inter_spike_intervals(const Vector<s64> &spike_ticks, f64 seconds_per_tick) {
    const Vector<s64> intervals = inter_spike_intervals(spike_ticks);
    if (intervals.empty()) {
        std::cout << "    (fewer than two spikes -- no interval to report)\n";
        return;
    }

    std::cout << "    inter-spike intervals (ms):";
    std::cout << std::fixed << std::setprecision(1);
    for (s64 interval : intervals) std::cout << " " << (f64)interval * seconds_per_tick * 1000.0;
    std::cout << "\n";
}

} // namespace spikecorec::examples
