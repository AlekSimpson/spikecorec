// ── Example: a GLIF excitatory/inhibitory network ───────────────────────────────────────────────
//
// The multi-population counterpart to glif3_torus_network_example. The same pipeline, but the model
// declares two populations of different GLIF variants wired by three projections carrying three
// different synapse types (current-based, conductance-based, and an NMDA-style synapse), so the
// interesting part is how the front-end lays neurons out and how the k²-tree adjacency is built.
//
// What it demonstrates beyond the torus examples:
//   1. Several populations sharing one flat neuron index space, and the per-population dispatch
//      that follows from it (arch §4.1's cell-type boundaries)
//   2. Distinct synapse ComponentTypes lowering to their own IR programs alongside the cell types
//   3. Projections → adjacency → the k²-tree/WeightMatrix the propagate stage scatters through
//   4. Driving a network and reading a spike raster back out
//
// ── Real per-edge synapse dispatch, and one real, documented gap it still exposes (ticket #131) ──
// Earlier revisions of this example forced every scattered weight to exactly zero, because
// AssembledModel's propagate stage did not yet invoke a projection's synapse ComponentType per-edge
// dynamics. That subsystem now exists (ticket #131): whenever a model declares real projections,
// AssembledModel dispatches every one of them through its own synapse type automatically. Running
// this exact fixture through that real dispatch (see
// tests/exit_model_validation_tests.cpp's own `ExitModelGlifEiNetwork.
// driven_simulation_spikes_via_real_per_edge_synapse_propagation`, which this example mirrors)
// shows:
//
//   ExcPop[0] → InhPop[0]   fires, via a real expOneSynapse conductance derived from ExcPop[0]'s own
//                           spikes — genuine propagation beyond the directly-stimulated cell.
//   ExcPop[0] → ExcPop[2]   stays silent. Not a placeholder gap anymore, but a real, orthogonal, and
//                           SEPARATE lowering limitation: `alphaCurrentSynapse`'s coupled
//                           TimeDerivative (its `I`/`J` state variables satisfy a linear system, not
//                           a single independent exponential decay) isn't a shape
//                           synapse_lowering.cpp recognizes yet ("general per-edge forward-Euler
//                           integration for an arbitrary right-hand side is out of Phase-1 scope",
//                           synapse_lowering.cpp's own documented limitation) — so `I` never
//                           integrates and ExcPop[2] never receives current at all. Left as-is here
//                           (fixing synapse_lowering.cpp is real engine work, out of this ticket's
//                           example/tooling scope) and stated plainly rather than silently omitted.
//
// So this fixture demonstrates propagation genuinely reaching beyond the stimulated cell (ExcPop[0]
// → InhPop[0]), while still being honest about the one synapse shape that does not yet propagate.
// The much larger, cleaner demonstration of whole-network propagation is the torus examples — see
// those (and their recorded `.spire` output, playable with `examples/render_spire_video.py`) for the
// primary "propagation reaches well beyond the input cell" story this suite tells.
//
// Run:  ./build/examples/glif_ei_network_example [--ticks 5000] [--dt 0.0001] [--print-ir]

#include <cmath>
#include <iostream>

#include "spikecorec/core/engine.h"
#include "nml_pipeline_support.h"

using namespace spikecorec;
using namespace spikecorec::nml;
using namespace spikecorec::examples;

// `Vector<...>` is spelled out fully as `spikecorec::Vector<...>` throughout this file, unlike every
// other example prior to the SpikeEngine migration. spikecorec/core/engine.h pulls in a file-scope
// `using namespace spikecorec::log;`, which declares its OWN `Vector` alias template -- ambiguous
// with `spikecorec::Vector` for bare unqualified `Vector<...>` lookup (two alias templates of the
// same name from two using-directives at the same scope, regardless of expanding to the identical
// type). Mirrors what tests/simple_lif_stdp_network_tests.cpp/tests/end_to_end_network_tests.cpp/
// examples/stdp_plasticity_example.cpp already do for the same reason.

int main(int argument_count, char **argument_values) {
    ExampleOptions options = parse_example_options(argument_count, argument_values, ExampleOptions{5000, 1e-4f});
    GpuContextScope gpu_context_scope;

    // ── 1-2. Front-end and lowering ─────────────────────────────────────────────────────────────
    ModelSpecification model = load_model_specification("glif_ei_network");
    spikecorec::Vector<IrProgram> programs = lower_type_library_to_ir(model);
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

    // ── 3-4. SpikeEngine builds its own ModelAllocation + WeightMatrix internally, then every
    // `.tick` section → one master kernel, compiled once. set_constant_weight(0.0f) is left in place
    // as documentation, not because it changes anything: this model has real projections, so
    // SpikeEngine now dispatches expOneSynapse/alphaCurrentSynapse/the NMDA synapse's own real
    // dynamics and forces this WeightMatrix's own scattered contribution to zero regardless (see
    // this file's own header comment). ──────────────────────────────────────────────────────────
    SpikeEngine engine(model, programs, options.dt_seconds);
    seed_membrane_potentials_from_resting_parameter(engine.nml_allocation_, model);
    engine.weights.set_constant_weight(0.0f);

    // ── 5. Stimulus ─────────────────────────────────────────────────────────────────────────────
    // This model drives its pulseGenerator through `<inputList>`/`<input>` rather than
    // `<explicitInput>` — jLEMS cannot resolve explicitInput's indexed target against a
    // populationList population. The front-end now recognizes `<inputList>`/`<input>` too (ticket
    // SC-149), so `model.stimuli` is populated here same as for an equivalent `<explicitInput>` --
    // reconstructed from the file's own literal pulseGenerator attributes below anyway, purely to
    // keep this example's own tick loop self-contained (matching the torus examples' own established
    // style, which build_stimulus_schedule-drives instead).
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

    spikecorec::Vector<spikecorec::Vector<f32>> membrane_traces((usize)model.total_neuron_count);
    spikecorec::Vector<spikecorec::Vector<s64>> spike_ticks_by_neuron((usize)model.total_neuron_count);

    for (s64 tick = 0; tick < options.tick_count; ++tick) {
        if (tick >= stimulus_delay_ticks && tick < stimulus_delay_ticks + stimulus_duration_ticks) {
            engine.network_inputs.get_contents()[stimulus_target_neuron_index] += stimulus_amplitude_amperes;
        }

        engine.step_tick(options.dt_seconds, tick, tick + 1);

        for (s32 population_index = 0; population_index < (s32)model.populations.size(); ++population_index) {
            const PopulationEntry &population = model.populations[(usize)population_index];
            for (s32 local_index = 0; local_index < population.size; ++local_index) {
                s32 neuron_index = population.neuron_index_begin + local_index;
                membrane_traces[(usize)neuron_index].push_back(
                    read_membrane_potential(engine.nml_allocation_, model, population_index, local_index));
                if (engine.last_spiked.get_contents()[neuron_index] == tick) {
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

    const s32 inhibitory_population_index = 1; // InhPop, declared second
    const s32 downstream_neuron_index = model.populations[(usize)inhibitory_population_index].neuron_index_begin;
    bool downstream_neuron_fired = !spike_ticks_by_neuron[(usize)downstream_neuron_index].empty();

    std::cout << "\n" << (downstream_neuron_fired ? "InhPop[0] fired" : "InhPop[0] did NOT fire")
              << " — propagated from ExcPop[0] via a real expOneSynapse per-edge conductance\n"
              << "(ticket #131), not a placeholder. See this file's header comment for ExcPop[2]'s\n"
              << "own remaining, orthogonal gap (alphaCurrentSynapse's coupled TimeDerivative).\n";

    return 0;
}
