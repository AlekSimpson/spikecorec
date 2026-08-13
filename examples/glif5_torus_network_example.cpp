// ── GLIF5 on a torus: the full variant ──────────────────────────────────────────────────
//
// The same 8x8 torus, running GLIF5, which is GLIF3 + GLIF4: it keeps the two after-spike
// currents AND adds the adaptive threshold. Both adaptation mechanisms are live at once, so
// the driven corner's inter-spike intervals lengthen faster than under either alone, and the
// run prints both halves -- the after-spike currents pushing v down and theta lifting the bar
// up -- side by side.
//
// For orientation across the family: GLIF1 is plain leaky integrate-and-fire, GLIF2 adds a
// scaled reset, GLIF3 adds the after-spike currents, GLIF4 adds the adaptive threshold
// instead, and GLIF5 combines the last two. All five are linear in their own state variables.
//
// One thing worth knowing if you write your own GLIF5-shaped model: theta MUST be seeded.
// This model's <OnStart> assigns it thetaInf; leave it at its zero-initialised value instead
// and it sits far ABOVE the -50mV threshold, so no neuron ever fires and the network looks
// silently dead rather than erroring.

#include "glif_torus_network.h"

using namespace spikecorec;
using namespace spikecorec::examples;

int main(int argument_count, char **argument_values) {
    try {
        const ExampleOptions options =
                parse_example_options(argument_count, argument_values,
                                      /*default_synapse_peak_current=*/1.0 * NANOAMPERE);
        configure_logging(options);

        // The first local owning anything GPU-backed, so it destructs last.
        GpuContextScope gpu_context;

        std::filesystem::create_directories(options.recording_directory);

        TorusNetworkOptions torus;
        torus.side_length = options.torus_side_length;
        torus.synapse_peak_current = options.synapse_peak_current;
        torus.membrane_recording_path =
                options.recording_directory + "/glif5_torus_membrane.spire";
        torus.spike_recording_path = options.recording_directory + "/glif5_torus_spikes.spire";

        const GeneratedModelDirectory model_directory("glif5_torus_network");
        String model_path = write_torus_model(model_directory, GlifVariant::Glif5, torus);

        SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
        print_torus_run_header(GlifVariant::Glif5, torus);

        const SpikeObservation observation = run_simulation(engine, options.tick_count);

        std::cout << "\n  " << observation.total_spike_count << " spikes over "
                  << observation.tick_count << " ticks, from "
                  << observation.spiking_neuron_count() << " of " << engine.total_neuron_count
                  << " neurons\n\n";

        print_first_spike_grid(observation, torus.side_length);

        std::cout << "\n  Driven corner (neuron 0), " << observation.spike_count_per_neuron[0]
                  << " spikes\n";
        print_inter_spike_intervals(observation.spike_ticks_per_neuron[0],
                                    engine.network_details.step_dt);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n  Both mechanisms at the end of the run, on the driven corner:\n"
                  << "    after-spike currents  asc1=" << read_cell_state(engine, 0, "asc1") * 1e12f
                  << " pA   asc2=" << read_cell_state(engine, 0, "asc2") * 1e12f << " pA\n"
                  << "    adaptive threshold    theta=" << read_cell_state(engine, 0, "theta") * 1000.0f
                  << " mV  (resting thetaInf is -50.00 mV)\n"
                  << "    v is being pushed down and the threshold lifted up at the same time.\n";

        std::cout << "\n  Recorded to " << torus.membrane_recording_path << " and\n              "
                  << torus.spike_recording_path << "\n"
                  << "  Render either with:  ./examples/render_spire_video.py "
                  << torus.spike_recording_path << " --side " << torus.side_length
                  << " --membrane " << torus.membrane_recording_path << "\n";

        engine.shutdown();
    } catch (const std::exception &error) {
        std::cerr << "glif5_torus_network_example: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
