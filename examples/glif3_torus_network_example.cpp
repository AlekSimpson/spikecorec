// ── GLIF3 on a torus: after-spike currents, and adaptation ──────────────────────────────
//
// The same 8x8 torus, running GLIF3: leaky integrate-and-fire plus two after-spike currents,
// asc1 and asc2. Both step DOWN on every spike (ascAdd1/ascAdd2 are negative -- a real
// after-spike current is hyperpolarizing) and decay back towards zero with time constants of
// 100ms and 10ms. Being negative they oppose the drive, so each spike makes the next one
// harder to reach.
//
// That is spike-frequency adaptation, and the run shows it directly: the driven corner's
// inter-spike intervals lengthen monotonically, where glif1_torus_network_example's are flat.
// Nothing in the engine knows what GLIF3 is -- the mechanism comes entirely out of the LEMS
// description below the fold in glif_torus_network.h.
//
// Read glif1_torus_network_example first, then run the two back to back and compare the two
// interval lists.

#include "glif_torus_network.h"

using namespace spikecorec;
using namespace spikecorec::examples;

int main(int argument_count, char **argument_values) {
    try {
        const ExampleOptions options =
                parse_example_options(argument_count, argument_values,
                                      /*default_connection_weight=*/25.0);
        configure_logging(options);

        // The first local owning anything GPU-backed, so it destructs last.
        GpuContextScope gpu_context;

        std::filesystem::create_directories(options.recording_directory);

        TorusNetworkOptions torus;
        torus.side_length = options.torus_side_length;
        torus.connection_weight = options.connection_weight;
        torus.membrane_recording_path =
                options.recording_directory + "/glif3_torus_membrane.spire";
        torus.spike_recording_path = options.recording_directory + "/glif3_torus_spikes.spire";

        const GeneratedModelDirectory model_directory("glif3_torus_network");
        String model_path = write_torus_model(model_directory, GlifVariant::Glif3, torus);

        SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
        print_torus_run_header(GlifVariant::Glif3, torus);

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

        const spikecorec::Vector<s64> intervals =
                inter_spike_intervals(observation.spike_ticks_per_neuron[0]);
        if (intervals.size() >= 2) {
            std::cout << "    These lengthen: " << intervals.front() << " ticks between the "
                      << "first pair, " << intervals.back() << " between the last.\n"
                      << "    That is the two after-spike currents accumulating. GLIF1's are "
                         "flat.\n";
        }

        // asc1 (tau 100ms) outlives asc2 (tau 10ms), so by the end of the run the residual
        // suppression is almost entirely asc1's.
        std::cout << "\n  Corner's after-spike currents at the end of the run (pA): asc1="
                  << read_cell_state(engine, 0, "asc1") * 1e12f
                  << "  asc2=" << read_cell_state(engine, 0, "asc2") * 1e12f << "\n";

        std::cout << "\n  Recorded to " << torus.membrane_recording_path << " and\n              "
                  << torus.spike_recording_path << "\n"
                  << "  Render either with:  ./examples/render_spire_video.py "
                  << torus.spike_recording_path << " --side " << torus.side_length
                  << " --membrane " << torus.membrane_recording_path << "\n";

        engine.shutdown();
    } catch (const std::exception &error) {
        std::cerr << "glif3_torus_network_example: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
