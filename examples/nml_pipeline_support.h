#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/recording.h"
#include "spikecorec/core/types.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/nml/allocator.h"
#include "spikecorec/nml/cell_lowering.h"
#include "spikecorec/nml/inputs_lowering.h"
#include "spikecorec/nml/ir.h"
#include "spikecorec/nml/master_kernel.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/synapse_lowering.h"

// ── Shared scaffolding for the NeuroML → GPU codegen examples ───────────────────────────────────
//
// Every example in this directory walks the same pipeline:
//
//     NML file → NML_Parser::parse → resolve_and_lower → build_model_specification
//              → lower_<category>_to_ir (one IrProgram per ComponentType in the type library)
//              → allocate_model (interprets each program's `.alloc` section)
//              → AssembledModel (assembles + compiles every `.tick` into one master kernel)
//              → step_tick, once per simulated tick
//
// The helpers below are only the parts that are identical across examples — model loading, buffer
// setup, state addressing, and console output. Each example file keeps its own model-specific
// logic (stimulus, initial state, what it measures) inline and visible, because that is the part
// worth reading.
//
// These helpers are deliberately close to the ones in tests/exit_model_validation_tests.cpp: the
// examples exercise the same pipeline the validation suite does, minus the assertions.
//
// ── ticket #131 note (real per-edge synapse dispatch) ───────────────────────────────────────────
// Every helper below is unchanged by ticket #131: `AssembledModel(model, programs)` now dispatches
// a model's real projection synapse dynamics automatically whenever `model.projections` is
// non-empty (master_kernel.h) — no extra opt-in from a caller. A `WeightMatrix::set_constant_weight`
// call only still matters for a model with NO projections (e.g. poisson_population_example's own
// trivial satisfy-the-constructor ring) or one built with `enable_delay_ring=true` (ticket #64's
// ring bypasses #131's dispatch entirely, see master_kernel.h) — every per-example header comment
// says which case applies.

namespace spikecorec::examples {

// ── GPU context lifetime ────────────────────────────────────────────────────────────────────────

// Brings up the GPU context on construction and tears it down on destruction.
//
// Declare this as the FIRST local in main(). Destruction order matters: release_gpu_resources()
// frees every buffer the backend handed out, so it must run AFTER the last object that owns one
// (ModelAllocation, WeightMatrix, AssembledModel, LiveModelBuffers…). Locals are destroyed in
// reverse declaration order, so declaring the guard first makes it destruct last, which is exactly
// the required order. Calling release_gpu_resources() by hand at the end of main() instead would
// free those buffers while the owners are still alive, and their destructors would then double-free.
struct GpuContextScope {
    GpuContextScope() { initialize_gpu_context(); }
    ~GpuContextScope() { release_gpu_resources(); }

    GpuContextScope(const GpuContextScope &) = delete;
    GpuContextScope &operator=(const GpuContextScope &) = delete;
};

// ── Model loading ───────────────────────────────────────────────────────────────────────────────

// Directory holding the example `.nml` models. Defaults to the checked-in NeuroML fixtures the
// validation suite drives (so an example and its corresponding test read the exact same file);
// override with SPIKECOREC_NML_MODEL_DIR to point the examples at your own models.
inline String nml_model_directory() {
    const char *override_directory = std::getenv("SPIKECOREC_NML_MODEL_DIR");
    if (override_directory != nullptr && override_directory[0] != '\0') return String(override_directory);
    return String(SPIKECOREC_TEST_FIXTURES_DIR) + "/nml";
}

// Parses `<model_base_name>_top.nml`, resolves it (units → SI, IDref wiring, extends/Fixed
// flattening), and lowers it to a flat ModelSpecification.
//
// Every model in this tree is split into a thin `<include>`-only `_top.nml` wrapper plus the real
// `.nml` content file. NML_Parser::parse XSD-validates only the top-level file it is handed, so
// parsing through the wrapper keeps a standalone NeuroML content file (one that jNeuroML consumes
// directly) from being validated as if it were a top-level document.
inline nml::ModelSpecification load_model_specification(const String &model_base_name) {
    nml::NML_Parser parser;
    parser.parse(nml_model_directory() + "/" + model_base_name + "_top.nml");
    nml::ResolvedModel resolved = nml::resolve_and_lower(parser);
    return nml::build_model_specification(resolved);
}

// One IrProgram per `model.type_library` entry, in the same order — the parallel array both
// allocate_model and AssembledModel expect. Each ComponentType is lowered by its own category's
// lowering pass; anything with no on-device program gets an empty placeholder so the arrays stay
// index-aligned.
//
// `lower_inputs_on_device` selects between the two stimulus paths, which are genuinely different:
//   false (default) — Inputs types get a placeholder. This is the Phase-1 `pulseGenerator` path: a
//                     pulse's whole trajectory is three resolved constants, so it is precomputed on
//                     the host by build_stimulus_schedule and never lowered to a kernel at all
//                     (lower_inputs_to_ir throws for pulseGenerator, by design).
//   true            — Inputs types are lowered to real device code. This is the Phase-2 on-device
//                     generator path (sineGenerator / rampGenerator / voltageClamp /
//                     SpikeSourcePoisson), where a population is bound directly to a generator
//                     component and dispatched like any other population.
inline Vector<nml::IrProgram> lower_type_library_to_ir(
    const nml::ModelSpecification &model, bool lower_inputs_on_device = false
) {
    Vector<nml::IrProgram> programs;
    programs.reserve(model.type_library.size());
    for (const auto &entry : model.type_library) {
        if (entry.category == nml::TypeLibraryCategory::Cell) {
            programs.push_back(nml::lower_cell_to_ir(entry));
        } else if (entry.category == nml::TypeLibraryCategory::Synapse) {
            programs.push_back(nml::lower_synapse_to_ir(entry));
        } else if (entry.category == nml::TypeLibraryCategory::Inputs && lower_inputs_on_device) {
            programs.push_back(nml::lower_inputs_to_ir(entry));
        } else {
            nml::IrProgram placeholder;
            placeholder.component_type_name = entry.component_type_name;
            programs.push_back(std::move(placeholder));
        }
    }
    return programs;
}

// The exact-edge-set WeightMatrix the assembled model's propagate stage scatters spikes through,
// built from every projection's connections.
//
// A single-cell model leaves `model.adjacency` unset, but ModelRuntimeBuffers::weights is required
// regardless — hence building the adjacency here rather than reusing the model's own.
//
// ticket #131: for a model WITH projections, AssembledModel now dispatches that projection's real
// synapse ComponentType dynamics automatically and forces this WeightMatrix's own scattered
// contribution to exactly zero for that dispatch (see master_kernel.h) — this function/its
// `set_constant_weight` are still needed (WeightMatrix's own k^2-tree/shared-basis storage backs
// the synapse's per-edge peredge state, arch §4.3), but the constant-weight VALUE itself only
// matters for a model with no projections at all, or one built with `enable_delay_ring=true`.
inline WeightMatrix build_weight_matrix(const nml::ModelSpecification &model) {
    Vector<Vector<s32>> adjacency((usize)model.total_neuron_count);
    for (const auto &projection : model.projections) {
        for (const auto &connection : projection.connections) {
            adjacency[(usize)connection.source_neuron_index].push_back(connection.target_neuron_index);
        }
    }
    return WeightMatrix(adjacency, /*rank=*/1);
}

// ── Runtime buffers ─────────────────────────────────────────────────────────────────────────────

// The engine-owned per-neuron buffers a caller must supply to AssembledModel::step_tick, bundled
// so an example's driving loop threads one struct through instead of nine.
struct LiveModelBuffers {
    GpuPointer<f32> network_inputs;
    GpuPointer<s64> last_spiked;
    GpuPointer<s32> next_active_neuron_indices;
    GpuPointer<s32> next_active_neuron_count;
    GpuPointer<s32> active_generation;
    GpuPointer<bool> emit_spike_flags;
    GpuPointer<u32> rng_state;
    nml::ModelRuntimeBuffers buffers;
};

// Allocates and initializes every buffer above. `seed_rng_state` allocates the per-neuron xorshift32
// state on-device generators need (`rand`/`randn` in a generated kernel); leave it false for a model
// with no on-device generator.
//
// `last_spiked` is seeded to -1 ("never fired") rather than 0, so tick 0 is not misread as a spike.
inline LiveModelBuffers make_live_model_buffers(
    nml::ModelAllocation &allocation, WeightMatrix &weights, s64 total_neuron_count, bool seed_rng_state = false
) {
    const usize neuron_count = (usize)total_neuron_count;
    LiveModelBuffers live;

    live.network_inputs = allocate<f32>(neuron_count * sizeof(f32));
    std::memset(live.network_inputs.get_contents(), 0, neuron_count * sizeof(f32));

    live.last_spiked = allocate<s64>(neuron_count * sizeof(s64));
    std::fill(live.last_spiked.get_contents(), live.last_spiked.get_contents() + total_neuron_count, (s64)-1);

    live.next_active_neuron_indices = allocate<s32>(neuron_count * sizeof(s32));
    live.next_active_neuron_count = allocate<s32>(sizeof(s32));
    live.next_active_neuron_count.get_contents()[0] = 0;

    live.active_generation = allocate<s32>(neuron_count * sizeof(s32));
    std::fill(live.active_generation.get_contents(), live.active_generation.get_contents() + total_neuron_count, -1);

    live.emit_spike_flags = allocate<bool>(neuron_count * sizeof(bool));
    std::memset(live.emit_spike_flags.get_contents(), 0, neuron_count * sizeof(bool));

    live.buffers.allocation = &allocation;
    live.buffers.weights = &weights;
    live.buffers.network_inputs = live.network_inputs.get_contents();
    live.buffers.last_spiked = live.last_spiked.get_contents();
    live.buffers.next_active_neuron_indices = live.next_active_neuron_indices.get_contents();
    live.buffers.next_active_neuron_count = live.next_active_neuron_count.get_contents();
    live.buffers.active_generation = live.active_generation.get_contents();
    live.buffers.emit_port_flags["spike"] = live.emit_spike_flags.get_contents();

    if (seed_rng_state) {
        live.rng_state = allocate<u32>(neuron_count * sizeof(u32));
        // xorshift32 is stuck at zero forever once it reaches zero — every seed must be nonzero and
        // per-neuron distinct.
        for (s64 neuron_index = 0; neuron_index < total_neuron_count; ++neuron_index) {
            live.rng_state.get_contents()[neuron_index] = (u32)((neuron_index + 1) * 2654435761u) | 1u;
        }
        live.buffers.rng_state = live.rng_state.get_contents();
    }

    return live;
}

// ── Cell-state addressing ───────────────────────────────────────────────────────────────────────

// Element index into ModelAllocation::cell_state for one state variable of one neuron.
//
// cell_state is a structure-of-arrays within per-population chunks: state slot `k`'s own
// [population.size] row sits at `cell_type_boundaries[population_index] + k * population.size`, so
// a neuron's value lands at that row's base plus its local index within the population.
inline s64 state_element_index(
    const nml::ModelAllocation &allocation, const nml::PopulationEntry &population,
    s32 population_index, s32 state_slot_index, s32 local_neuron_index
) {
    return allocation.cell_type_boundaries.get_contents()[population_index]
           + (s64)state_slot_index * population.size + local_neuron_index;
}

// Membrane potential ("v", always state slot 0 for every cell type in these examples) of one neuron.
inline f32 read_membrane_potential(
    const nml::ModelAllocation &allocation, const nml::ModelSpecification &model,
    s32 population_index, s32 local_neuron_index
) {
    const nml::PopulationEntry &population = model.populations[(usize)population_index];
    return allocation.cell_state.get_contents()[
        state_element_index(allocation, population, population_index, /*state_slot_index=*/0, local_neuron_index)];
}

// Writes one state variable for every neuron of every population.
inline void seed_state_variable_everywhere(
    nml::ModelAllocation &allocation, const nml::ModelSpecification &model, s32 state_slot_index, f32 value
) {
    for (s32 population_index = 0; population_index < (s32)model.populations.size(); ++population_index) {
        const nml::PopulationEntry &population = model.populations[(usize)population_index];
        for (s32 local_index = 0; local_index < population.size; ++local_index) {
            allocation.cell_state.get_contents()[
                state_element_index(allocation, population, population_index, state_slot_index, local_index)] = value;
        }
    }
}

// Seeds every neuron's membrane potential to its own cell type's resting potential `EL`.
//
// allocate_model zero-initializes cell_state; it does not apply a cell type's OnStart block, so a
// caller seeds initial state itself. Every GLIF variant declares `OnStart: v = EL`, which is what
// this reproduces.
inline void seed_membrane_potentials_from_resting_parameter(
    nml::ModelAllocation &allocation, const nml::ModelSpecification &model
) {
    for (s32 population_index = 0; population_index < (s32)model.populations.size(); ++population_index) {
        const nml::PopulationEntry &population = model.populations[(usize)population_index];
        const nml::TypeLibraryEntry &cell_entry = model.type_library[(usize)population.type_library_index];
        auto resting_potential_iterator = cell_entry.baked_constants.find("EL");
        if (resting_potential_iterator == cell_entry.baked_constants.end()) {
            throw std::runtime_error(
                "seed_membrane_potentials_from_resting_parameter: cell type '" + cell_entry.component_type_name
                + "' declares no EL parameter — seed this model's initial state explicitly instead");
        }
        for (s32 local_index = 0; local_index < population.size; ++local_index) {
            allocation.cell_state.get_contents()[
                state_element_index(allocation, population, population_index, /*state_slot_index=*/0, local_index)] =
                (f32)resting_potential_iterator->second;
        }
    }
}

// ── Command-line options ────────────────────────────────────────────────────────────────────────

struct ExampleOptions {
    s64 tick_count = 5000;
    f32 dt_seconds = 1e-4f;
    bool print_ir = false;
    bool verbose = false;
};

// Quiets the engine logger down to errors unless `--verbose` was passed.
//
// Resolving any model pulls in the vendored NeuroML standard library, which logs a handful of
// warnings about std-lib constants (`J_per_K_per_mol`, `C_per_mol`, …) whose unit symbols this
// build's conversion table doesn't recognize. They are real diagnostics, but they concern
// ComponentTypes these examples never instantiate, and they bury the output. Pass `--verbose` to
// see them and everything else the engine logs.
// The namespace is spelled out in full: core/log.h does `using namespace std;` at global scope, so
// a bare `log::` here resolves against `std::log`/`::log` from <cmath> instead of the engine's
// logging namespace.
inline void configure_example_logging(bool verbose) {
    ::spikecorec::log::logger().set_level(
        verbose ? ::spikecorec::log::LogLevel::info : ::spikecorec::log::LogLevel::err);
}

// Parses `--ticks <count>`, `--dt <seconds>`, `--print-ir`, and `--verbose`; anything else is
// reported and ignored.
inline ExampleOptions parse_example_options(int argument_count, char **argument_values, ExampleOptions defaults) {
    ExampleOptions options = defaults;
    for (int argument_index = 1; argument_index < argument_count; ++argument_index) {
        String argument = argument_values[argument_index];
        bool has_value = argument_index + 1 < argument_count;
        if (argument == "--ticks" && has_value) {
            options.tick_count = std::strtoll(argument_values[++argument_index], nullptr, 10);
        } else if (argument == "--dt" && has_value) {
            options.dt_seconds = std::strtof(argument_values[++argument_index], nullptr);
        } else if (argument == "--print-ir") {
            options.print_ir = true;
        } else if (argument == "--verbose") {
            options.verbose = true;
        } else {
            std::cerr << "ignoring unrecognized argument '" << argument << "' "
                      << "(supported: --ticks <count> --dt <seconds> --print-ir --verbose)\n";
        }
    }
    configure_example_logging(options.verbose);
    return options;
}

// Dumps every lowered program in the IR's own text form — the `.alloc` directives the engine
// interprets at init, then the stage-tagged `.tick` instructions that get template-substituted into
// the master kernel. This is the actual intermediate representation, not a summary of it.
inline void print_ir_programs(const nml::ModelSpecification &model, const Vector<nml::IrProgram> &programs) {
    for (usize program_index = 0; program_index < programs.size(); ++program_index) {
        std::cout << "\n\033[1m// IR for " << model.type_library[program_index].component_type_name
                  << " (instance '" << model.type_library[program_index].bound_instance_id << "')\033[0m\n"
                  << nml::print_ir_program(programs[program_index]) << "\n";
    }
}

// ── Console output ──────────────────────────────────────────────────────────────────────────────

// Total `.tick` instruction count across all nine stage tags.
inline usize count_tick_instructions(const nml::TickProgram &tick) {
    return tick.deliver.size() + tick.integrate.size() + tick.detect.size() + tick.emit.size()
           + tick.reset.size() + tick.propagate.size() + tick.plasticity.size() + tick.record.size();
}

// Per-stage `.tick` instruction counts, e.g. "integrate=6 detect=1 emit=1 reset=3". Stages with no
// instructions are omitted. Stages 1/6/8/9 of the tick loop are engine-owned, so a cell type's
// program typically only populates integrate/detect/emit/reset.
inline String format_tick_stage_breakdown(const nml::TickProgram &tick) {
    const std::pair<const char *, usize> stage_counts[] = {
        {"deliver", tick.deliver.size()},       {"integrate", tick.integrate.size()},
        {"detect", tick.detect.size()},         {"emit", tick.emit.size()},
        {"reset", tick.reset.size()},           {"propagate", tick.propagate.size()},
        {"plasticity", tick.plasticity.size()}, {"record", tick.record.size()},
    };
    String breakdown;
    for (const auto &stage_count : stage_counts) {
        if (stage_count.second == 0) continue;
        if (!breakdown.empty()) breakdown += " ";
        breakdown += String(stage_count.first) + "=" + std::to_string(stage_count.second);
    }
    return breakdown.empty() ? String("(empty)") : breakdown;
}

inline void print_heading(const String &text) {
    std::cout << "\n\033[1m" << text << "\033[0m\n"
              << String(text.size(), '-') << "\n";
}

// Populations, ComponentTypes, projections, and recordings the front-end produced for one model —
// what every example prints before running anything, so the pipeline's output is visible even when
// the simulation itself is uninteresting.
inline void print_model_summary(const nml::ModelSpecification &model, const Vector<nml::IrProgram> &programs) {
    print_heading("Model specification");
    std::cout << "  total neurons     : " << model.total_neuron_count << "\n"
              << "  populations       : " << model.populations.size() << "\n"
              << "  projections       : " << model.projections.size() << "\n"
              << "  recordings        : " << model.recordings.size() << "\n"
              << "  stimuli           : " << model.stimuli.size() << "\n";

    std::cout << "\n  ComponentTypes in use (one lowered IR program each):\n";
    for (usize entry_index = 0; entry_index < model.type_library.size(); ++entry_index) {
        const nml::TypeLibraryEntry &entry = model.type_library[entry_index];
        const nml::IrProgram &program = programs[entry_index];
        std::cout << "    " << std::left << std::setw(26) << entry.component_type_name
                  << " instance=" << std::setw(22) << entry.bound_instance_id
                  << " state_variables=" << entry.state_variable_count
                  << " alloc=" << program.alloc.size()
                  << " tick=" << count_tick_instructions(program.tick)
                  << (entry.category == nml::TypeLibraryCategory::Cell
                          ? (program.closed_form_advanceable ? "  [closed-form advanceable]" : "  [nonlinear]")
                          : "")
                  << "\n"
                  << "      stages: " << format_tick_stage_breakdown(program.tick) << "\n";
    }

    std::cout << "\n  Populations (neuron index ranges):\n";
    for (usize population_index = 0; population_index < model.populations.size(); ++population_index) {
        const nml::PopulationEntry &population = model.populations[population_index];
        std::cout << "    " << std::left << std::setw(16) << population.id
                  << " size=" << std::setw(6) << population.size
                  << " neurons=[" << population.neuron_index_begin << ", " << population.neuron_index_end << ")"
                  << " type=" << model.type_library[(usize)population.type_library_index].component_type_name << "\n";
    }
    std::cout << std::right;
}

inline String format_seconds(f64 seconds) {
    std::ostringstream formatted;
    formatted << std::fixed << std::setprecision(2) << seconds * 1000.0 << "ms";
    return formatted.str();
}

inline void print_spike_times(const String &label, const Vector<s64> &spike_ticks, f32 dt_seconds) {
    std::cout << "  " << std::left << std::setw(20) << label << std::right
              << std::setw(5) << spike_ticks.size() << " spikes";
    if (!spike_ticks.empty()) {
        std::cout << "  at ";
        const usize printed_count = std::min<usize>(spike_ticks.size(), 8);
        for (usize spike_index = 0; spike_index < printed_count; ++spike_index) {
            if (spike_index > 0) std::cout << ", ";
            std::cout << format_seconds((f64)spike_ticks[spike_index] * dt_seconds);
        }
        if (spike_ticks.size() > printed_count) std::cout << ", …";
    }
    std::cout << "\n";
}

// Mean inter-spike interval, and the ratio of the last interval to the first — a value above 1.0
// means the train is slowing down, which is what spike-frequency adaptation looks like.
inline void print_interspike_interval_summary(const Vector<s64> &spike_ticks, f32 dt_seconds) {
    if (spike_ticks.size() < 3) {
        std::cout << "  (fewer than 3 spikes — no interval statistics)\n";
        return;
    }
    Vector<f64> intervals_seconds;
    for (usize spike_index = 1; spike_index < spike_ticks.size(); ++spike_index) {
        intervals_seconds.push_back((f64)(spike_ticks[spike_index] - spike_ticks[spike_index - 1]) * dt_seconds);
    }
    f64 interval_sum = 0.0;
    for (f64 interval : intervals_seconds) interval_sum += interval;

    std::cout << "  first interval      " << format_seconds(intervals_seconds.front()) << "\n"
              << "  last interval       " << format_seconds(intervals_seconds.back()) << "\n"
              << "  mean interval       " << format_seconds(interval_sum / (f64)intervals_seconds.size()) << "\n"
              << "  last / first        " << std::fixed << std::setprecision(2)
              << (intervals_seconds.back() / intervals_seconds.front())
              << (intervals_seconds.back() > intervals_seconds.front() * 1.05
                      ? "   (intervals lengthening — spike-frequency adaptation)\n"
                      : "\n");
    std::cout.unsetf(std::ios::floatfield);
}

// A coarse ASCII plot of a per-tick trace, downsampled to `column_count` columns. Enough to see the
// shape of a state variable without leaving the terminal.
//
// `axis_scale`/`axis_unit` control the y-axis labels only, never the plotted shape — the defaults
// render SI volts as millivolts. A trace in different units needs its own pair (e.g. 1e12f / "pA"
// for a current), otherwise every label rounds to the same degenerate value.
inline void print_trace_plot(
    const String &label, const Vector<f32> &trace, f32 dt_seconds, f32 axis_scale = 1000.0f,
    const String &axis_unit = "mV", s32 row_count = 14, s32 column_count = 92
) {
    if (trace.empty()) {
        std::cout << "  (empty trace)\n";
        return;
    }
    const s32 columns = std::min<s32>(column_count, (s32)trace.size());

    Vector<f32> column_values((usize)columns, 0.0f);
    for (s32 column_index = 0; column_index < columns; ++column_index) {
        // Peak within the column's window, so a brief spike is never averaged away.
        usize window_begin = (usize)((s64)column_index * (s64)trace.size() / columns);
        usize window_end = (usize)((s64)(column_index + 1) * (s64)trace.size() / columns);
        window_end = std::max(window_end, window_begin + 1);
        column_values[(usize)column_index] =
            *std::max_element(trace.begin() + (long)window_begin, trace.begin() + (long)window_end);
    }

    f32 minimum_value = *std::min_element(column_values.begin(), column_values.end());
    f32 maximum_value = *std::max_element(column_values.begin(), column_values.end());
    if (std::fabs(maximum_value - minimum_value) < 1e-12f) maximum_value = minimum_value + 1e-12f;

    std::cout << "  " << label << "   (" << format_seconds(0.0) << " … "
              << format_seconds((f64)trace.size() * dt_seconds) << ")\n";
    for (s32 row_index = row_count - 1; row_index >= 0; --row_index) {
        f32 row_value = minimum_value + (maximum_value - minimum_value) * ((f32)row_index + 0.5f) / (f32)row_count;
        std::cout << "  " << std::setw(9) << std::fixed << std::setprecision(1) << row_value * axis_scale
                  << " " << axis_unit << " │";
        for (s32 column_index = 0; column_index < columns; ++column_index) {
            f32 normalized =
                (column_values[(usize)column_index] - minimum_value) / (maximum_value - minimum_value);
            s32 sample_row = (s32)(normalized * (f32)row_count);
            std::cout << (sample_row >= row_index ? '#' : ' ');
        }
        std::cout << "\n";
    }
    // Pad so the corner lands under the axis bar: 2 leading spaces + 9-wide value + 1 space + unit.
    std::cout << "  " << String(10 + axis_unit.size(), ' ') << "└" << String((usize)columns, '-') << "\n";
    std::cout.unsetf(std::ios::floatfield);
}

// One row per neuron, one column per downsampled time window; '|' marks a window containing a spike.
inline void print_spike_raster(
    const Vector<Vector<s64>> &spike_ticks_by_neuron, s64 tick_count, s32 column_count = 92
) {
    const s32 columns = (s32)std::min<s64>(column_count, std::max<s64>(tick_count, 1));
    for (usize neuron_index = 0; neuron_index < spike_ticks_by_neuron.size(); ++neuron_index) {
        Vector<char> row((usize)columns, ' ');
        for (s64 spike_tick : spike_ticks_by_neuron[neuron_index]) {
            s64 column_index = spike_tick * columns / std::max<s64>(tick_count, 1);
            row[(usize)std::min<s64>(column_index, columns - 1)] = '|';
        }
        std::cout << "  neuron " << std::setw(4) << neuron_index << " │";
        for (char cell : row) std::cout << cell;
        std::cout << "\n";
    }
    std::cout << "  " << String(12, ' ') << "└" << String((usize)columns, '-') << "\n";
}

// ── Recording (ticket #59 [E2]'s SimulationRecorder/`.spire` format already exists end to end --
// see core/recording.h -- but nothing in the engine itself drives one per tick against a running
// AssembledModel, master_kernel.h's own recording doc comment: "not wired to SimulationRecorder ...
// a caller must drive it". This is that caller, added by ticket #138.) ──────────────────────────

// Creates `directory_path` (and any missing parents) if it does not already exist — every example
// that records calls this once before constructing a NetworkActivityRecorder, since SimulationRecorder
// itself (core/recording.cpp's RawSink) just opens an ofstream and throws a plain "failed to open"
// runtime_error if the directory is missing, rather than creating it.
inline void ensure_directory_exists(const String &directory_path) {
    std::filesystem::create_directories(directory_path);
}

// Captures a WHOLE model's activity across a run as two parallel `.spire` streams, one frame (one
// f32 per neuron) per tick each: every neuron's membrane potential (state slot 0 -- `v` for every
// cell type these examples use), and a spike-raster mask (1.0 on the tick a neuron fires, 0.0
// otherwise), matching output_recording.h's own RecordingSourceKind::SpikeRaster convention (a
// spike event encoded as a 1.0/0.0 mask reusing the same per-neuron-f32-array SimulationRecorder
// already handles uniformly for any exposure). Two separate files rather than one interleaved
// stream -- output_recording.h's own established model is "one recorder stream per (population,
// exposure, is_event_recording) triple"; this is that model's direct generalization to a whole
// single-population capture (every torus example's own shape), read back by
// examples/render_spire_video.py.
class NetworkActivityRecorder {
public:
    NetworkActivityRecorder(const String &membrane_potential_path, const String &spike_raster_path, s64 neuron_count)
        : membrane_potential_recorder_(membrane_potential_path, neuron_count),
          spike_raster_recorder_(spike_raster_path, neuron_count),
          membrane_potential_scratch_((usize)neuron_count, 0.0f),
          spike_raster_scratch_((usize)neuron_count, 0.0f) {}

    // Samples every population's neurons for `tick` and appends one frame to each of this
    // recorder's two `.spire` streams. `last_spiked` must be the SAME buffer step_tick was just
    // driven with (the whole-model, global-index array ModelRuntimeBuffers::last_spiked points at).
    void record_tick(
        const nml::ModelAllocation &allocation, const nml::ModelSpecification &model, const s64 *last_spiked,
        s64 tick
    ) {
        for (s32 population_index = 0; population_index < (s32)model.populations.size(); ++population_index) {
            const nml::PopulationEntry &population = model.populations[(usize)population_index];
            for (s32 local_index = 0; local_index < population.size; ++local_index) {
                s32 neuron_index = population.neuron_index_begin + local_index;
                membrane_potential_scratch_[(usize)neuron_index] =
                    read_membrane_potential(allocation, model, population_index, local_index);
                spike_raster_scratch_[(usize)neuron_index] = last_spiked[neuron_index] == tick ? 1.0f : 0.0f;
            }
        }
        membrane_potential_recorder_.record_frame(
            membrane_potential_scratch_.data(), (s64)membrane_potential_scratch_.size());
        spike_raster_recorder_.record_frame(spike_raster_scratch_.data(), (s64)spike_raster_scratch_.size());
    }

    // Flushes and closes both streams. Each SimulationRecorder's own destructor does this too, so
    // calling it explicitly is only needed to print an honest "wrote ..." message afterward (the
    // data may still be sitting in the internal chunk buffer, unflushed, until then).
    void finish() {
        membrane_potential_recorder_.finish();
        spike_raster_recorder_.finish();
    }

private:
    SimulationRecorder membrane_potential_recorder_;
    SimulationRecorder spike_raster_recorder_;
    Vector<f32> membrane_potential_scratch_;
    Vector<f32> spike_raster_scratch_;
};

} // namespace spikecorec::examples
