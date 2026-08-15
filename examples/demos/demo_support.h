#pragma once

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "spikecorec/core/engine.h"
#include "spikecorec/core/backend.h"

namespace spikecorec::demos {

inline String demo_directory() { return "build/demos"; }

inline String demo_path(const String &demo_name, const String &suffix,
                        const String &directory = demo_directory()) {
    return directory + "/" + demo_name + suffix;
}

struct DemoResult {
    s64 neuron_count = 0;
    s64 edge_count = 0;
    s64 spike_count = 0;
    f64 mean_rate = 0.0;
    f64 participation = 0.0;
    f64 peak_synchrony = 0.0;
    f64 active_tick_fraction = 0.0;
    f64 build_seconds = 0.0;
    f64 run_seconds = 0.0;
};

// Runs one model and records it. `adjacency` empty means the document already carries its
// own connections; otherwise it supplies them and `synapse_component_id` names the synapse
// every edge uses.
inline DemoResult run_demo(const String &demo_name,
                           const String &lems_path,
                           const std::vector<std::vector<s32>> &adjacency,
                           const String &synapse_component_id,
                           f64 connection_weight,
                           f64 connection_delay_seconds,
                           s64 video_frame_stride) {
    using Clock = std::chrono::steady_clock;

    const auto build_started = Clock::now();
    SpikeEngine engine(lems_path, adjacency, synapse_component_id, connection_weight,
                       connection_delay_seconds);
    const f64 build_seconds = std::chrono::duration<f64>(Clock::now() - build_started).count();

    engine.record_membrane_video(demo_path(demo_name, "_membrane.spire"),
                                 video_frame_stride);

    const auto run_started = Clock::now();
    engine.run();
    const f64 run_seconds = std::chrono::duration<f64>(Clock::now() - run_started).count();

    engine.write_recordings();
    engine.write_spike_file(demo_path(demo_name, "_spikes.dat"));

    DemoResult result;
    result.neuron_count = engine.total_neuron_count;
    result.edge_count = engine.layout.total_edge_count;
    result.spike_count = (s64)engine.recorded_spikes.size();
    result.mean_rate = engine.mean_firing_rate_hertz();
    result.participation = engine.fraction_of_neurons_that_spiked();
    result.build_seconds = build_seconds;
    result.run_seconds = run_seconds;

    // Synchrony both ways. The ceiling catches a population firing as one; the floor
    // catches the same thing from the other side, because a network in lockstep puts every
    // spike on a handful of ticks. A rate on its own cannot tell a network from a
    // metronome -- which is exactly what an earlier million-cell run looked like.
    std::vector<s64> spikes_per_tick((size_t)engine.lifetime, 0);
    for (const RecordedSpike &spike : engine.recorded_spikes) {
        const s64 tick = (s64)(spike.time_seconds / (f64)engine.step_dt + 0.5);
        if (tick >= 0 && tick < engine.lifetime) spikes_per_tick[(size_t)tick] += 1;
    }
    s64 busiest = 0;
    s64 active_ticks = 0;
    for (s64 count : spikes_per_tick) {
        busiest = std::max(busiest, count);
        active_ticks += count > 0 ? 1 : 0;
    }
    result.peak_synchrony = (f64)busiest / (f64)engine.total_neuron_count;
    result.active_tick_fraction = (f64)active_ticks / (f64)engine.lifetime;

    return result;
}

// What the run did, and how to watch it.
inline void report_demo(const String &demo_name, const String &description,
                        const DemoResult &result, f64 simulated_seconds,
                        s64 render_max_frames, s64 render_frames_per_second) {
    std::printf("\n=== %s ===\n%s\n\n", demo_name.c_str(), description.c_str());
    std::printf("  neurons              %lld\n", (long long)result.neuron_count);
    std::printf("  edges                %lld\n", (long long)result.edge_count);
    std::printf("  spikes               %lld\n", (long long)result.spike_count);
    std::printf("  mean firing rate     %.2f Hz\n", result.mean_rate);
    std::printf("  neurons that fired   %.1f %%\n", 100.0 * result.participation);
    std::printf("  busiest tick         %.2f %% of the population\n",
                100.0 * result.peak_synchrony);
    std::printf("  ticks with activity  %.1f %%\n", 100.0 * result.active_tick_fraction);
    std::printf("  build / run          %.2f s / %.2f s\n",
                result.build_seconds, result.run_seconds);

    std::printf("\nrender it with:\n"
                "  python3 examples/render_membrane_video.py %s \\\n"
                "      --duration %g --max-frames %lld --frames-per-second %lld\n",
                demo_path(demo_name, "_membrane.spire").c_str(), simulated_seconds,
                (long long)render_max_frames, (long long)render_frames_per_second);
}

// Called at the top of every demo's main().
inline void begin_demo() {
    std::filesystem::create_directories(demo_directory());
    initialize_gpu_context();
}

} // namespace spikecorec::demos
