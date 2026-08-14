// A recurrent balanced excitatory/inhibitory network for each of the five GLIF cell types,
// each run for a second and each recorded to a .spire membrane video alongside its spikes.
//
// What it prints for each network is what decides whether the demo is worth watching: a
// population firing rate, how much of the population ever fires, how synchronous the
// activity is, and whether the rate at the end of the run matches the rate in the middle.
//
//   build/examples/glif_network_example [output_directory]
//
// then render any of them with:
//
//   python3 examples/render_membrane_video.py build/glif3_network_membrane.spire

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "spikecorec/core/engine.h"
#include "spikecorec/core/backend.h"
#include "glif_network_model.h"

using namespace spikecorec;
using namespace spikecorec::examples;

namespace {

struct NetworkSummary {
    s32 glif_index = 0;
    s64 neuron_count = 0;
    s64 edge_count = 0;
    s64 spike_count = 0;
    f64 mean_rate = 0.0;
    f64 participation = 0.0;
    f64 peak_synchrony = 0.0;
    f64 middle_rate = 0.0;
    f64 final_rate = 0.0;
};

f64 rate_over(const SpikeEngine &engine, f64 from_seconds, f64 to_seconds) {
    s64 count = 0;
    for (const RecordedSpike &spike : engine.recorded_spikes) {
        if (spike.time_seconds >= from_seconds && spike.time_seconds < to_seconds) count += 1;
    }
    return (f64)count / ((f64)engine.total_neuron_count * (to_seconds - from_seconds));
}

} // namespace

int main(int argument_count, char **arguments) {
    const std::string output_directory = argument_count > 1 ? arguments[1] : "build";

    initialize_gpu_context();

    std::vector<NetworkSummary> summaries;

    for (s32 glif_index = 1; glif_index <= 5; glif_index += 1) {
        GlifNetworkParameters parameters;
        parameters.glif_index = glif_index;

        const std::string lems_path =
                write_glif_network_model(output_directory, parameters);
        const std::string video_path = output_directory + "/glif" +
                std::to_string(glif_index) + "_network_membrane.spire";

        SpikeEngine engine(lems_path);
        engine.record_membrane_video(video_path, /*frame_stride=*/10);
        engine.run();
        engine.write_recordings();

        NetworkSummary summary;
        summary.glif_index = glif_index;
        summary.neuron_count = engine.total_neuron_count;
        summary.edge_count = engine.layout.total_edge_count;
        summary.spike_count = (s64)engine.recorded_spikes.size();
        summary.mean_rate = engine.mean_firing_rate_hertz();
        summary.participation = engine.fraction_of_neurons_that_spiked();

        std::vector<s64> spikes_per_tick((size_t)engine.lifetime, 0);
        for (const RecordedSpike &spike : engine.recorded_spikes) {
            const s64 tick = (s64)(spike.time_seconds / parameters.step_seconds + 0.5);
            if (tick >= 0 && tick < engine.lifetime) spikes_per_tick[(size_t)tick] += 1;
        }
        summary.peak_synchrony =
                (f64)*std::max_element(spikes_per_tick.begin(), spikes_per_tick.end()) /
                (f64)engine.total_neuron_count;

        const f64 total_seconds = parameters.simulation_seconds;
        summary.middle_rate = rate_over(engine, 0.4 * total_seconds, 0.6 * total_seconds);
        summary.final_rate = rate_over(engine, 0.8 * total_seconds, total_seconds);

        summaries.push_back(summary);
        std::printf("GLIF%d network done: %lld spikes, membrane video -> %s\n",
                    glif_index, (long long)summary.spike_count, video_path.c_str());
    }

    // Every count in this heading comes from the parameters the run actually used. The
    // population sizes and the duration were hardcoded here and went stale the moment the
    // duration changed, which printed "1 s" over a two-second run.
    const GlifNetworkParameters heading_parameters;
    std::printf("\n=== GLIF recurrent networks, %lld excitatory + %lld inhibitory, %g s ===\n",
                (long long)heading_parameters.excitatory_count,
                (long long)heading_parameters.inhibitory_count,
                heading_parameters.simulation_seconds);
    std::printf("%-7s %7s %8s %8s %10s %10s %10s\n",
                "type", "spikes", "rate Hz", "alive %", "peak sync", "mid Hz", "end Hz");
    for (const NetworkSummary &summary : summaries) {
        std::printf("GLIF%-3d %7lld %8.2f %8.1f %9.1f%% %10.2f %10.2f\n",
                    summary.glif_index, (long long)summary.spike_count, summary.mean_rate,
                    100.0 * summary.participation, 100.0 * summary.peak_synchrony,
                    summary.middle_rate, summary.final_rate);
    }
    std::printf("\n%lld edges per network\n", (long long)summaries.front().edge_count);

    release_gpu_resources();
    return 0;
}
