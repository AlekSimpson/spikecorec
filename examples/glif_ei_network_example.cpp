// ── Example: a GLIF excitatory/inhibitory network ───────────────────────────────────────────────
//
// The multi-population counterpart to glif3_single_cell_example. The same pipeline, but the model
// declares two populations of different GLIF variants wired by three projections carrying three
// different synapse types (current-based, conductance-based, and an NMDA-style synapse), so the
// interesting part is how the front-end lays neurons out and how the k²-tree adjacency is built.
//
// What it demonstrates beyond the single-cell example:
//   1. Several populations sharing one flat neuron index space, and the per-population dispatch
//      that follows from it (arch §4.1's cell-type boundaries)
//   2. Distinct synapse ComponentTypes lowering to their own IR programs alongside the cell types
//   3. Projections → adjacency → the k²-tree/WeightMatrix the propagate stage scatters through
//   4. Driving a network and reading a spike raster back out
//
// ── Read this before drawing conclusions from the raster ────────────────────────────────────────
// The assembled model's propagate stage scatters spikes through the k²-tree/WeightMatrix path; it
// does not yet invoke a projection's actual synapse ComponentType per-edge dynamics. Routing a
// spike through a real synapse needs a spike-scatter batch construction subsystem that no ticket has
// built yet. So this example forces every weight to exactly zero rather than scattering an
// arbitrary reconstructed low-rank value that would look plausible but is not derived from any of
// this model's synapse parameters.
//
// The practical consequence: only the directly stimulated neuron fires. jNeuroML running this same
// file — with the real expOneSynapse/alphaCurrentSynapse/NMDA dynamics — shows downstream
// propagation that this engine cannot reproduce until that subsystem exists. The synapse types
// still parse, resolve, and lower to IR correctly, which is what this example shows.
//
// Run:  ./build/examples/glif_ei_network_example [--ticks 5000] [--dt 0.0001] [--print-ir]

#include <cmath>
#include <iostream>

#include "nml_pipeline_support.h"

using namespace spikecorec;
using namespace spikecorec::nml;
using namespace spikecorec::examples;

int main(int argument_count, char **argument_values) {
    ExampleOptions options = parse_example_options(argument_count, argument_values, ExampleOptions{5000, 1e-4f});
    GpuContextScope gpu_context_scope;

    // ── 1-2. Front-end and lowering ─────────────────────────────────────────────────────────────
    ModelSpecification model = load_model_specification("glif_ei_network");
    Vector<IrProgram> programs = lower_type_library_to_ir(model);
    print_model_summary(model, programs);
    if (options.print_ir) print_ir_programs(model, programs);

    print_heading("Projections");
    for (const ProjectionEntry &projection : model.projections) {
        std::cout << "  " << std::left << std::setw(18) << projection.id << std::right
                  << "  " << model.populations[(usize)projection.presynaptic_population_index].id
                  << " → " << model.populations[(usize)projection.postsynaptic_population_index].id
                  << "  synapse type = "
                  << model.type_library[(usize)projection.synapse_type_library_index].component_type_name
                  << "  connections = " << projection.connections.size() << "\n";
    }

    // ── 3. Allocation and initial state ─────────────────────────────────────────────────────────
    ModelAllocation allocation = allocate_model(model, programs);
    seed_membrane_potentials_from_resting_parameter(allocation, model);

    // ── 4. Adjacency → WeightMatrix ─────────────────────────────────────────────────────────────
    WeightMatrix weights = build_weight_matrix(model);
    weights.set_constant_weight(0.0f); // see this file's header comment on why exactly zero

    AssembledModel assembled_model(model, programs);
    LiveModelBuffers live = make_live_model_buffers(allocation, weights, model.total_neuron_count);

    // ── 5. Stimulus ─────────────────────────────────────────────────────────────────────────────
    // This model drives its pulseGenerator through `<inputList>`/`<input>` rather than
    // `<explicitInput>` — jLEMS cannot resolve explicitInput's indexed target against a
    // populationList population. The front-end only recognizes explicitInput today, so `model.stimuli`
    // is empty here and the window is reconstructed from the file's own literal pulseGenerator
    // attributes. (The single-cell example, whose plain population does parse, uses the real
    // build_stimulus_schedule path instead.)
    const f64 seconds_per_tick = (f64)options.dt_seconds;
    const s64 stimulus_delay_ticks = (s64)std::round(0.010 / seconds_per_tick);
    const s64 stimulus_duration_ticks = (s64)std::round(0.200 / seconds_per_tick);
    const f32 stimulus_amplitude_amperes = 0.5e-9f; // <pulseGenerator ... amplitude="0.5nA"/>
    const s32 stimulus_target_neuron_index = 0;     // ExcPop's neuron 0, declared first

    print_heading("Stimulus (reconstructed from the file's pulseGenerator)");
    std::cout << "  neuron " << stimulus_target_neuron_index
              << "  ticks [" << stimulus_delay_ticks << ", " << stimulus_delay_ticks + stimulus_duration_ticks << ")"
              << "  current " << stimulus_amplitude_amperes * 1e12f << " pA\n";

    // ── 6. Tick loop ────────────────────────────────────────────────────────────────────────────
    print_heading("Simulating");
    std::cout << "  " << options.tick_count << " ticks × " << options.dt_seconds * 1000.0f << "ms = "
              << format_seconds((f64)options.tick_count * options.dt_seconds) << "\n";

    Vector<Vector<f32>> membrane_traces((usize)model.total_neuron_count);
    Vector<Vector<s64>> spike_ticks_by_neuron((usize)model.total_neuron_count);

    for (s64 tick = 0; tick < options.tick_count; ++tick) {
        if (tick >= stimulus_delay_ticks && tick < stimulus_delay_ticks + stimulus_duration_ticks) {
            live.buffers.network_inputs[stimulus_target_neuron_index] += stimulus_amplitude_amperes;
        }

        assembled_model.step_tick(live.buffers, options.dt_seconds, tick, tick + 1);

        for (s32 population_index = 0; population_index < (s32)model.populations.size(); ++population_index) {
            const PopulationEntry &population = model.populations[(usize)population_index];
            for (s32 local_index = 0; local_index < population.size; ++local_index) {
                s32 neuron_index = population.neuron_index_begin + local_index;
                membrane_traces[(usize)neuron_index].push_back(
                    read_membrane_potential(allocation, model, population_index, local_index));
                if (live.buffers.last_spiked[neuron_index] == tick) {
                    spike_ticks_by_neuron[(usize)neuron_index].push_back(tick);
                }
            }
        }
    }

    // ── 7. Results ──────────────────────────────────────────────────────────────────────────────
    print_heading("Spike raster");
    print_spike_raster(spike_ticks_by_neuron, options.tick_count);

    print_heading("Per-neuron spike counts");
    for (s32 population_index = 0; population_index < (s32)model.populations.size(); ++population_index) {
        const PopulationEntry &population = model.populations[(usize)population_index];
        for (s32 local_index = 0; local_index < population.size; ++local_index) {
            s32 neuron_index = population.neuron_index_begin + local_index;
            print_spike_times(
                population.id + "[" + std::to_string(local_index) + "]",
                spike_ticks_by_neuron[(usize)neuron_index], options.dt_seconds);
        }
    }

    print_heading("Membrane potential of the stimulated neuron");
    print_trace_plot("ExcPop[0]/v", membrane_traces[(usize)stimulus_target_neuron_index], options.dt_seconds);

    std::cout << "\nOnly the directly stimulated neuron fires — every edge weight is zero here.\n"
              << "See this file's header comment for why, and tests/exit_model_validation_tests.cpp\n"
              << "for the comparison against jNeuroML's own run of this same file.\n";

    return 0;
}
