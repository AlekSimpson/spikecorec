// ── The GLIF family at demo scale, recorded for playback ────────────────────────────────────
//
// Runs one member of the GLIF family on a 48x48 torus -- 2304 cells, 9216 connections -- driven
// from the middle, and records both streams render_spire_video.py needs: an <OutputFile> of
// every cell's membrane potential every tick, and an <EventOutputFile> of every spike. The pair
// plays back as a wavefront leaving the driven cell, expanding across the population, wrapping
// around the torus edges and colliding with itself.
//
// Pick the variant with --variant glif1..glif5. The engine has no idea which one it is running:
// each is a LEMS <ComponentType> in examples/glif_torus_network.h, lowered to GPU code by the
// same codegen path, and what separates them on screen is the mechanism each declares.
//
//     make demo-videos DEMO_PYTHON=<python with numpy+matplotlib>
//
// builds this, runs all five variants and renders all five videos. See examples/demos/README.md.

#include "glif_demo_network.h"

using namespace spikecorec;
using namespace spikecorec::examples;
using namespace spikecorec::examples::demos;

int main(int argument_count, char **argument_values) {
    try {
        const ExampleOptions options =
                parse_example_options(argument_count, argument_values,
                                      /*default_synapse_peak_current=*/1.0 * NANOAMPERE,
                                      /*default_torus_side_length=*/DEMO_SIDE_LENGTH);
        configure_logging(options);

        const GlifVariant variant =
                options.glif_variant_name.empty()
                        ? GlifVariant::Glif1
                        : glif_variant_from_name(options.glif_variant_name);
        const std::string variant_name = glif_variant_short_name(variant);

        // The first local owning anything GPU-backed, so it destructs last.
        GpuContextScope gpu_context;

        std::filesystem::create_directories(options.recording_directory);

        TorusNetworkOptions torus =
                demo_torus_options(options.torus_side_length, options.synapse_peak_current);
        torus.membrane_recording_path =
                options.recording_directory + "/" + variant_name + "_demo_membrane.spire";
        torus.spike_recording_path =
                options.recording_directory + "/" + variant_name + "_demo_spikes.spire";

        const GeneratedModelDirectory model_directory(variant_name + "_demo");
        String model_path = write_torus_model(model_directory, variant, torus);

        SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);
        print_torus_run_header(variant, torus);

        // GLIF2's whole difference from GLIF1 is where the membrane lands after a reset, which
        // is gone by the next tick -- so it has to be read as the run goes, not afterwards.
        // Cheap enough to capture unconditionally: one float per spike of one neuron.
        const s32 *spike_flags = engine.spike_flags.get_contents();
        spikecorec::Vector<f32> potential_after_each_spike;

        const SpikeObservation observation =
                run_simulation(engine, options.tick_count, [&](s64) {
                    if (spike_flags[torus.driven_neuron_index] != 0) {
                        potential_after_each_spike.push_back(
                                read_cell_state(engine, torus.driven_neuron_index, "v"));
                    }
                });

        std::cout << "\n  " << observation.total_spike_count << " spikes over "
                  << observation.tick_count << " ticks, from "
                  << observation.spiking_neuron_count() << " of " << engine.total_neuron_count
                  << " neurons\n\n";

        print_wavefront_spread(observation, engine.network_details.step_dt,
                               engine.total_neuron_count);
        std::cout << "\n";
        print_activity_timeline(observation, engine.network_details.step_dt, /*bin_count=*/12);
        std::cout << "\n";
        print_variant_signature(variant, engine, observation, torus.driven_neuron_index,
                                potential_after_each_spike);

        std::cout << "\n  Recorded to " << torus.membrane_recording_path << " and\n              "
                  << torus.spike_recording_path << "\n"
                  << "  Render with:  ./examples/render_spire_video.py "
                  << torus.spike_recording_path << " --side " << torus.side_length
                  << " --membrane " << torus.membrane_recording_path << " --output "
                  << variant_name << "_demo.gif\n";

        engine.shutdown();
    } catch (const std::exception &error) {
        std::cerr << "glif_family_demo: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
