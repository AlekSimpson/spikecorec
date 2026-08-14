// GLIF networks on a square torus, at a range of sizes and under a range of external drive
// patterns. Connectivity comes from the engine's own square_torus(), not from the document,
// which is what lets the size be a parameter rather than a rewrite.
//
//   build/examples/glif_torus_example [output_directory] [max_side_length]
//
// The default sweep tops out at a side length small enough to finish quickly. Pass a larger
// one to push it -- side 1000 is a million cells and four million edges.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "spikecorec/core/engine.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/topologies.h"
#include "glif_torus_model.h"

using namespace spikecorec;
using namespace spikecorec::examples;

namespace {

struct RunSummary {
    String label;
    s64 neuron_count = 0;
    s64 edge_count = 0;
    s64 spike_count = 0;
    f64 mean_rate = 0.0;
    f64 participation = 0.0;
    f64 build_seconds = 0.0;
    f64 run_seconds = 0.0;
};

const char *pattern_name(TorusDrivePattern pattern) {
    switch (pattern) {
        case TorusDrivePattern::UniformBackground: return "uniform";
        case TorusDrivePattern::CentrePatch:       return "centre patch";
        case TorusDrivePattern::WanderingPatch:    return "wandering patch";
    }
    return "?";
}

RunSummary run_one(const String &directory, const GlifTorusParameters &parameters,
                   const String &label, const String &video_path) {
    using Clock = std::chrono::steady_clock;

    const String lems_path = write_glif_torus_model(directory, parameters);

    const auto build_started = Clock::now();
    // The engine's own topology: a 4-neighbour wraparound grid, every cell with the same
    // out-degree, generated rather than written down.
    const std::vector<std::vector<s32>> adjacency = square_torus(parameters.side_length);

    SpikeEngine engine(lems_path, adjacency,
                       "glif" + std::to_string(parameters.glif_index) + "TorusSynapse",
                       parameters.connection_weight, parameters.connection_delay_seconds);
    const f64 build_seconds =
            std::chrono::duration<f64>(Clock::now() - build_started).count();

    if (!video_path.empty()) engine.record_membrane_video(video_path, /*frame_stride=*/10);

    const auto run_started = Clock::now();
    engine.run();
    const f64 run_seconds = std::chrono::duration<f64>(Clock::now() - run_started).count();

    if (!video_path.empty()) engine.write_recordings();

    RunSummary summary;
    summary.label = label;
    summary.neuron_count = engine.total_neuron_count;
    summary.edge_count = engine.layout.total_edge_count;
    summary.spike_count = (s64)engine.recorded_spikes.size();
    summary.mean_rate = engine.mean_firing_rate_hertz();
    summary.participation = engine.fraction_of_neurons_that_spiked();
    summary.build_seconds = build_seconds;
    summary.run_seconds = run_seconds;
    return summary;
}

} // namespace

int main(int argument_count, char **arguments) {
    const String output_directory = argument_count > 1 ? arguments[1] : "build";
    const s64 maximum_side = argument_count > 2 ? std::stoll(arguments[2]) : 64;

    initialize_gpu_context();

    std::vector<RunSummary> summaries;

    // ── the three drive patterns, at one size, on GLIF3 ──────────────────────────
    // GLIF3 because its after-spike currents make the network's own structure visible:
    // a uniformly driven GLIF3 sheet bands, and a patch-driven one spreads.
    for (TorusDrivePattern pattern : {TorusDrivePattern::UniformBackground,
                                      TorusDrivePattern::CentrePatch,
                                      TorusDrivePattern::WanderingPatch}) {
        GlifTorusParameters parameters;
        parameters.glif_index = 3;
        parameters.side_length = 48;
        parameters.drive_pattern = pattern;

        const String label = String("GLIF3 48x48 ") + pattern_name(pattern);
        const String video = output_directory + "/glif_torus_" +
                String(pattern == TorusDrivePattern::UniformBackground ? "uniform"
                       : pattern == TorusDrivePattern::CentrePatch ? "centre"
                                                                   : "wandering") +
                "_membrane.spire";
        summaries.push_back(run_one(output_directory, parameters, label, video));
        std::printf("%-28s done\n", label.c_str());
    }

    // ── one size sweep, to show what the topology constructor buys ───────────────
    for (s64 side = 16; side <= maximum_side; side *= 2) {
        GlifTorusParameters parameters;
        parameters.glif_index = 1;
        parameters.side_length = side;
        parameters.simulation_seconds = 0.25;

        const String label = "GLIF1 " + std::to_string(side) + "x" + std::to_string(side);
        summaries.push_back(run_one(output_directory, parameters, label, ""));
        std::printf("%-28s done\n", label.c_str());
    }

    std::printf("\n=== GLIF on a square torus ===\n");
    std::printf("%-28s %9s %9s %9s %8s %8s %8s %8s\n",
                "run", "neurons", "edges", "spikes", "rate Hz", "alive %",
                "build s", "run s");
    for (const RunSummary &summary : summaries) {
        std::printf("%-28s %9lld %9lld %9lld %8.2f %7.1f%% %8.2f %8.2f\n",
                    summary.label.c_str(), (long long)summary.neuron_count,
                    (long long)summary.edge_count, (long long)summary.spike_count,
                    summary.mean_rate, 100.0 * summary.participation,
                    summary.build_seconds, summary.run_seconds);
    }

    release_gpu_resources();
    return 0;
}
