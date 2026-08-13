// ── A nonlinear cell, from a vendored NeuroML ComponentType ─────────────────────────────
//
// Everything else in this directory generates its model. This one loads a checked-in
// fixture, tests/fixtures/nml/izhikevich_network_top.nml, unmodified -- the same file the
// reference simulator consumes -- and changes nothing about how the engine treats it.
//
// Three things it shows the GLIF examples cannot:
//
//   * NONLINEAR dynamics. izhikevich2007Cell's membrane equation carries a v*v term, so its
//     trajectory has no closed form. The engine's active-set optimization -- which advances a
//     skipped neuron across many ticks in one analytic step -- would be invalid for it; on
//     the NeuroML path that optimization is off entirely and every neuron is stepped every
//     tick, so the question does not currently arise.
//   * a TWO state-variable cell. `v` and the recovery variable `u` share one contiguous slot
//     per neuron, which is what makes cell_state's per-type layout concrete.
//   * a cell type nobody declared. izhikevich2007Cell is real vendored NeuroML from
//     third_party/neuroml2/std_lib/Cells.xml; the parser merges the standard library at load
//     time, so the fixture needs no inline <ComponentType> at all. Compare the GLIF examples,
//     which have to declare theirs because the standard library has no GLIF type.
//
// Note the fixture's <OutputFile fileName="..."> paths are RELATIVE, so its two recordings land
// in whatever directory the process is running in, under the names the fixture chose -- the
// model's decision, not the engine's. Rather than litter wherever it was launched from, this
// example changes into --record-dir before constructing the engine, which is the only lever a
// caller has over a model that names its outputs relatively.

#include "example_support.h"

using namespace spikecorec;
using namespace spikecorec::examples;

int main(int argument_count, char **argument_values) {
    try {
        const ExampleOptions options =
                parse_example_options(argument_count, argument_values,
                                      /*default_synapse_peak_current=*/0.0);
        configure_logging(options);

        // The first local owning anything GPU-backed, so it destructs last.
        GpuContextScope gpu_context;

        String model_path =
                std::string(SPIKECOREC_TEST_FIXTURES_DIR) + "/nml/izhikevich_network_top.nml";

        // The model path is absolute, so only its OUTPUT paths are affected by this. Done
        // before the engine is constructed because the recorders are opened there.
        const std::filesystem::path launch_directory = std::filesystem::current_path();
        std::filesystem::create_directories(options.recording_directory);
        std::filesystem::current_path(options.recording_directory);

        SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

        print_heading("izhikevich2007Cell: two cells, one driven, one recruited");
        std::cout << "  model  " << model_path << "\n"
                  << "  recordings in "
                  << (launch_directory / options.recording_directory).string() << "\n"
                  << "  " << engine.total_neuron_count << " neurons, "
                  << engine.network_details.cell_types.size() << " cell type, "
                  << engine.lifetime << " ticks at "
                  << engine.network_details.step_dt * 1000.0 << " ms\n"
                  << "  state variables per cell:";
        for (const String &state_variable_name :
             engine.network_details.cell_types[0].state_variable_names) {
            std::cout << " " << state_variable_name;
        }
        std::cout << "\n  DrivenPop[0] is neuron 0 and takes a 150pA step; TargetPop[0] is\n"
                     "  neuron 1 and receives only the propagated connection.\n";

        const SpikeObservation observation = run_simulation(engine, options.tick_count);

        std::cout << "\n  neuron 0 (driven)   " << observation.spike_count_per_neuron[0]
                  << " spikes, first at tick " << observation.first_spike_tick_per_neuron[0]
                  << "\n  neuron 1 (target)   " << observation.spike_count_per_neuron[1]
                  << " spikes, first at tick " << observation.first_spike_tick_per_neuron[1]
                  << "\n";

        if (observation.spike_count_per_neuron[1] > 0) {
            std::cout << "\n  The target fires "
                      << observation.first_spike_tick_per_neuron[1] -
                                 observation.first_spike_tick_per_neuron[0]
                      << " tick(s) after the driven cell: nothing stimulates it directly, so\n"
                         "  every one of its spikes arrived over the projection.\n";
        }

        std::cout << "\n  Driven cell, final state:  v="
                  << read_cell_state(engine, 0, "v") * 1000.0f << " mV   u="
                  << read_cell_state(engine, 0, "u") << "\n";
        print_inter_spike_intervals(observation.spike_ticks_per_neuron[0],
                                    engine.network_details.step_dt);
        std::cout << "    Regular spiking: the intervals settle rather than lengthening, "
                     "because\n    izhikevich2007Cell's adaptation lives in u, which reaches a "
                     "steady cycle.\n";

        engine.shutdown();
    } catch (const std::exception &error) {
        std::cerr << "izhikevich_network_example: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
