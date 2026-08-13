// ── GLIF4 on a torus: the adaptive threshold ────────────────────────────────────────────
//
// The same 8x8 torus, running GLIF4: leaky integrate-and-fire with an adaptive threshold
// INSTEAD of after-spike currents. `theta` is a state variable, not a constant -- it relaxes
// towards thetaInf with time constant tauTheta, and jumps by thetaSpikeAdd on every spike --
// so the firing condition is `v .gt. theta` and the bar itself rises as the cell fires.
//
// GLIF3 and GLIF4 both produce lengthening inter-spike intervals, by opposite means: GLIF3
// pushes v down, GLIF4 lifts the threshold up. The run prints theta climbing above its
// resting thetaInf under sustained drive, which is the half of the mechanism the interval
// list alone does not show. Read glif5_torus_network_example afterwards for the SAME
// threshold mechanism combined with GLIF3's after-spike currents.

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
                options.recording_directory + "/glif4_torus_membrane.spire";
        torus.spike_recording_path = options.recording_directory + "/glif4_torus_spikes.spire";

        const GeneratedModelDirectory model_directory("glif4_torus_network");
        String model_path = write_torus_model(model_directory, GlifVariant::Glif4, torus);

        SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
        print_torus_run_header(GlifVariant::Glif4, torus);

        const s32 *spike_flags = engine.spike_flags.get_contents();
        spikecorec::Vector<f32> threshold_after_each_corner_spike;

        const SpikeObservation observation =
                run_simulation(engine, options.tick_count, [&](s64) {
                    if (spike_flags[0] != 0) {
                        threshold_after_each_corner_spike.push_back(
                                read_cell_state(engine, /*neuron_index=*/0, "theta"));
                    }
                });

        std::cout << "\n  " << observation.total_spike_count << " spikes over "
                  << observation.tick_count << " ticks, from "
                  << observation.spiking_neuron_count() << " of " << engine.total_neuron_count
                  << " neurons\n\n";

        print_first_spike_grid(observation, torus.side_length);

        std::cout << "\n  Driven corner (neuron 0), " << observation.spike_count_per_neuron[0]
                  << " spikes\n";
        print_inter_spike_intervals(observation.spike_ticks_per_neuron[0],
                                    engine.network_details.step_dt);

        std::cout << "\n  theta after each of the corner's spikes (mV), against a resting "
                     "thetaInf of -50.0:\n   ";
        std::cout << std::fixed << std::setprecision(2);
        for (usize spike_index = 0;
             spike_index < threshold_after_each_corner_spike.size() && spike_index < 12;
             ++spike_index) {
            std::cout << " " << threshold_after_each_corner_spike[spike_index] * 1000.0f;
        }
        std::cout << "\n    Each spike adds thetaSpikeAdd (3mV) and the relaxation towards "
                     "thetaInf\n    only partly undoes it between spikes, so the threshold "
                     "climbs.\n";

        std::cout << "\n  Recorded to " << torus.membrane_recording_path << " and\n              "
                  << torus.spike_recording_path << "\n";

        engine.shutdown();
    } catch (const std::exception &error) {
        std::cerr << "glif4_torus_network_example: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
