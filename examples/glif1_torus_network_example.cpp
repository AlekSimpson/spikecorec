// ── GLIF1 on a torus: plain leaky integrate-and-fire ────────────────────────────────────
//
// Start here. This is the whole engine surface in one program: hand SpikeEngine a
// NeuroML/LEMS file, step it, shut it down. Everything else -- the cell dynamics, the
// wiring, the stimulus, the recordings -- is in the model.
//
// GLIF1 is the simplest variant of the family: one state variable (v), a fixed threshold
// (vth), a flat reset to vreset, and a refractory period. There is no adaptation mechanism
// at all, so under a constant current the inter-spike intervals settle to a CONSTANT
// steady-state rate. That flat interval series is the baseline the other four variants'
// adaptation mechanisms are read against -- run this and glif3_torus_network_example back to
// back and compare the two interval lists.
//
// Read the other four in this order afterwards: glif2 (scaled reset), glif3 (after-spike
// currents), glif4 (adaptive threshold), glif5 (the last two combined).

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
                options.recording_directory + "/glif1_torus_membrane.spire";
        torus.spike_recording_path = options.recording_directory + "/glif1_torus_spikes.spire";

        const GeneratedModelDirectory model_directory("glif1_torus_network");
        String model_path = write_torus_model(model_directory, GlifVariant::Glif1, torus);

        SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
        print_torus_run_header(GlifVariant::Glif1, torus);

        const SpikeObservation observation = run_simulation(engine, options.tick_count);

        std::cout << "\n  " << observation.total_spike_count << " spikes over "
                  << observation.tick_count << " ticks, from "
                  << observation.spiking_neuron_count() << " of " << engine.total_neuron_count
                  << " neurons\n\n";

        print_first_spike_grid(observation, torus.side_length);

        // GLIF1's signature: a flat interval series. The driven corner is the only cell under
        // a sustained current, so it is the one whose steady state is worth reading.
        std::cout << "\n  Driven corner (neuron 0), " << observation.spike_count_per_neuron[0]
                  << " spikes\n";
        print_inter_spike_intervals(observation.spike_ticks_per_neuron[0],
                                    engine.network_details.step_dt);
        std::cout << "    GLIF1 has no adaptation mechanism, so these settle to a constant.\n";

        std::cout << "\n  Recorded to " << torus.membrane_recording_path << " and\n              "
                  << torus.spike_recording_path << "\n"
                  << "  Render either with:  ./examples/render_spire_video.py "
                  << torus.spike_recording_path << " --side " << torus.side_length
                  << " --membrane " << torus.membrane_recording_path << "\n";

        engine.shutdown();
    } catch (const std::exception &error) {
        std::cerr << "glif1_torus_network_example: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
