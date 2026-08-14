// A 1,000-cell balanced excitatory/inhibitory network of iafCells, generated as NeuroML,
// compiled to one GPU kernel from its own ComponentTypes, and run for a second.
//
// The numbers it prints are the ones that decide whether the demo is worth looking at: a
// mean population rate in the single-to-low-double-digit hertz, nearly every neuron firing
// at least once, no tick where the whole population fires together, and a rate at the end
// of the run close to the rate in the middle.

#include <algorithm>
#include <cstdio>
#include <iomanip>

#include "spikecorec/core/engine.h"
#include "spikecorec/core/backend.h"
#include "balanced_network_model.h"

using namespace spikecorec;
using namespace spikecorec::examples;

int main(int argument_count, char **arguments) {
    const std::string output_directory = argument_count > 1 ? arguments[1] : "build";

    BalancedNetworkParameters parameters;
    const std::string model_path = write_balanced_network_model(output_directory, parameters);
    std::printf("model written to %s\n", model_path.c_str());

    initialize_gpu_context();

    SpikeEngine engine(model_path);
    engine.run();
    engine.write_recordings();

    // Synchrony: the largest fraction of the population that fired on any single tick. A
    // network locked into one repeating volley reads as a healthy mean rate but is not a
    // useful demo, so it is measured rather than assumed away.
    std::vector<s64> spikes_per_tick((size_t)engine.lifetime, 0);
    for (const RecordedSpike &spike : engine.recorded_spikes) {
        const s64 tick = (s64)(spike.time_seconds / parameters.step_seconds + 0.5);
        if (tick >= 0 && tick < engine.lifetime) spikes_per_tick[(size_t)tick] += 1;
    }
    const s64 busiest_tick =
            *std::max_element(spikes_per_tick.begin(), spikes_per_tick.end());

    // Sustained, or dying out? Compare the last fifth of the run against the middle fifth.
    auto rate_over = [&](f64 from_seconds, f64 to_seconds) {
        s64 count = 0;
        for (const RecordedSpike &spike : engine.recorded_spikes) {
            if (spike.time_seconds >= from_seconds && spike.time_seconds < to_seconds) {
                count += 1;
            }
        }
        return (f64)count / ((f64)engine.total_neuron_count * (to_seconds - from_seconds));
    };

    const f64 total_seconds = parameters.simulation_seconds;
    const f64 middle_rate = rate_over(0.4 * total_seconds, 0.6 * total_seconds);
    const f64 final_rate = rate_over(0.8 * total_seconds, total_seconds);

    std::printf("\n=== balanced network: %lld cells, %lld edges ===\n",
                (long long)engine.total_neuron_count, (long long)engine.layout.total_edge_count);
    std::printf("total spikes             : %zu\n", engine.recorded_spikes.size());
    std::printf("mean firing rate         : %.2f Hz\n", engine.mean_firing_rate_hertz());
    std::printf("neurons that ever spiked : %.1f %%\n",
                100.0 * engine.fraction_of_neurons_that_spiked());
    std::printf("busiest single tick      : %.1f %% of the population\n",
                100.0 * (f64)busiest_tick / (f64)engine.total_neuron_count);
    std::printf("rate at 40-60%% of run    : %.2f Hz\n", middle_rate);
    std::printf("rate at 80-100%% of run   : %.2f Hz\n", final_rate);

    release_gpu_resources();
    return 0;
}
