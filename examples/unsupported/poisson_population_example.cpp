// ── NOT BUILT: a Poisson-driven population ──────────────────────────────────────────────
//
// This program is excluded from `make examples` on purpose. It is here as the shape the
// example will take, and as a precise record of what has to land first.
//
// What it would demonstrate: the second stimulus path. A <pulseGenerator> is three constants,
// so the engine precomputes its whole per-tick series on the host before the run starts
// (SpikeEngine::input_event_streams). A Poisson source cannot be precomputed that way -- it
// has to draw a random number per neuron per tick -- so it has to be lowered to real device
// code and dispatched like any other population, with its own per-neuron RNG state.
//
// ── what is missing, exactly ─────────────────────────────────────────────────────────────
//
// Two independent blockers, both reachable today by pointing SpikeEngine at
// tests/fixtures/nml/poisson_population_top.nml (the checked-in fixture this example would
// load):
//
//  1. `SpikeSourcePoisson` is not classified as a cell, so the model does not even parse.
//     src/nml/nml.cpp classifies it by its ComponentType: it extends `baseSpikeSource`, whose
//     RuntimeCategory is Input rather than Cell, and a <population> whose component is not a
//     Cell throws during parse_lems:
//
//         Population 'PoissonNet.PoissonPop' references component 'poissonInstance' of type
//         'SpikeSourcePoisson', which is not classified as a cell
//
//     A population of generators has to be representable as a population before anything
//     downstream matters.
//
//  2. `random()` is rejected by codegen. SpikeSourcePoisson's dynamics draw the next
//     inter-spike interval with `-1 * log(random(1))/rate`, and src/nml/kernel_codegen.cpp
//     refuses the call outright:
//
//         unsupported function 'random': a deterministic per-neuron stream needs a seed
//         argument the generated kernels do not take
//
//     That is not only a missing function name. There is no per-neuron RNG state buffer in
//     SpikeEngine at all -- no `rng_state` alongside cell_state -- so there is nowhere for a
//     per-neuron stream to live or to be seeded from `simulation_seed`, which the engine does
//     already parse and hold.
//
// Nothing else about this example is blocked: the fixture's <Simulation>, its recordings and
// its network wiring all go through the ordinary path. Delete this comment and move the file
// up into examples/ once both of the above exist.

#include "../example_support.h"

using namespace spikecorec;
using namespace spikecorec::examples;

int main(int argument_count, char **argument_values) {
    try {
        const ExampleOptions options =
                parse_example_options(argument_count, argument_values,
                                      /*default_connection_weight=*/0.0);
        configure_logging(options);

        GpuContextScope gpu_context;

        String model_path =
                std::string(SPIKECOREC_TEST_FIXTURES_DIR) + "/nml/poisson_population_top.nml";

        // Throws today; see blocker 1 above.
        SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

        print_heading("A population of Poisson generators");
        std::cout << "  " << engine.total_neuron_count << " sources, declared rate 10 Hz, seed "
                  << engine.simulation_seed << "\n";

        const SpikeObservation observation = run_simulation(engine, options.tick_count);

        const f64 simulated_seconds =
                (f64)observation.tick_count * engine.network_details.step_dt;
        std::cout << "\n  " << observation.total_spike_count << " spikes over "
                  << simulated_seconds << " s across " << engine.total_neuron_count
                  << " sources\n  observed rate "
                  << (f64)observation.total_spike_count /
                             (simulated_seconds * (f64)engine.total_neuron_count)
                  << " Hz against a declared 10 Hz\n\n"
                  << "  The comparison is necessarily statistical: two PRNG streams never match\n"
                     "  spike for spike, only in distribution.\n";

        engine.shutdown();
    } catch (const std::exception &error) {
        std::cerr << "poisson_population_example: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
