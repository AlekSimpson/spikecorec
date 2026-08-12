// ── GLIF2 on a torus: the scaled reset ──────────────────────────────────────────────────
//
// The same 8x8 torus glif1_torus_network_example runs, with one thing changed: the reset
// rule. GLIF1 snaps v back to a flat vreset on every spike. GLIF2 resets it relative to how
// far past vth the membrane overshot on the triggering tick:
//
//     v <- vreset + resetScale * (v - vth)
//
// resetScale is 0.4 here, the interesting non-degenerate case -- at 0 this would be
// bit-exactly GLIF1's own reset. The run prints v immediately after each of the driven
// corner's spikes, which is where the difference is visible: it lands measurably ABOVE
// vreset rather than snapping to it, by 0.4 of whatever overshoot the crossing tick
// happened to produce.
//
// Read glif1_torus_network_example first.

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
                options.recording_directory + "/glif2_torus_membrane.spire";
        torus.spike_recording_path = options.recording_directory + "/glif2_torus_spikes.spire";

        const GeneratedModelDirectory model_directory("glif2_torus_network");
        String model_path = write_torus_model(model_directory, GlifVariant::Glif2, torus);

        SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
        print_torus_run_header(GlifVariant::Glif2, torus);

        // v is read on the tick the spike is emitted, which is the tick the reset has just
        // been applied on -- so this is the post-reset value, not the crossing value.
        const s32 *spike_flags = engine.spike_flags.get_contents();
        spikecorec::Vector<f32> potential_after_each_corner_spike;

        const SpikeObservation observation =
                run_simulation(engine, options.tick_count, [&](s64) {
                    if (spike_flags[0] != 0) {
                        potential_after_each_corner_spike.push_back(
                                read_cell_state(engine, /*neuron_index=*/0, "v"));
                    }
                });

        std::cout << "\n  " << observation.total_spike_count << " spikes over "
                  << observation.tick_count << " ticks, from "
                  << observation.spiking_neuron_count() << " of " << engine.total_neuron_count
                  << " neurons\n\n";

        print_first_spike_grid(observation, torus.side_length);

        std::cout << "\n  How far above vreset (-70mV) the corner landed after each reset, in "
                     "microvolts:\n   ";
        std::cout << std::fixed << std::setprecision(1);
        for (usize spike_index = 0;
             spike_index < potential_after_each_corner_spike.size() && spike_index < 12;
             ++spike_index) {
            std::cout << " "
                      << (potential_after_each_corner_spike[spike_index] + 0.070f) * 1e6f;
        }
        std::cout << "\n    Every one is strictly positive, where GLIF1 would print 0.0 every\n"
                     "    time. Each is resetScale (0.4) times the overshoot past vth on the\n"
                     "    tick that fired -- and one 0.1ms tick of a 0.6nA drive only carries a\n"
                     "    100pF membrane a few hundred microvolts past threshold, so the\n"
                     "    surviving fraction is small. It is the RULE that differs, not its\n"
                     "    size: shorten dt and it shrinks, lengthen dt and it grows.\n";

        std::cout << "\n  Recorded to " << torus.membrane_recording_path << " and\n              "
                  << torus.spike_recording_path << "\n";

        engine.shutdown();
    } catch (const std::exception &error) {
        std::cerr << "glif2_torus_network_example: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
