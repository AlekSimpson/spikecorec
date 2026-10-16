#pragma once

// Helpers shared by the test files: one GPU backend, a temporary directory per test, fixture
// paths, a LEMS document around a model, and spike statistics.

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <gtest/gtest.h>

#include "backend.h"
#include "engine.h"
#include "log.h"
#include "types.h"
#include "parser.h"

#ifndef SPIKECOREC_TEST_FIXTURE_DIR
#error "SPIKECOREC_TEST_FIXTURE_DIR must name tests/fixtures; the Makefile defines it"
#endif

namespace spikecorec::test_support {

// One backend for the whole runner, so tests do not each create a device. Each k^2-tree and
// weight matrix owns its own slab of it and frees the slab on destruction.
inline Engine_Backend &shared_backend() {
    static Engine_Backend backend;
    return backend;
}

// An absolute path into tests/fixtures, valid from any working directory.
inline String fixture_path(const String &relative_path) {
    return String(SPIKECOREC_TEST_FIXTURE_DIR) + "/" + relative_path;
}

// True when the NeuroML standard library the parser includes from is on disk.
inline bool standard_library_available() {
    const NML_Context context;
    return !context.STANDARD_LIBRARY_PATH.empty() && std::filesystem::exists(context.STANDARD_LIBRARY_PATH);
}

// A directory for the current test, removed on destruction. Its root is named after the process
// ID, so concurrent test runners never share a file.
class Temporary_Directory {
public:
    Temporary_Directory() {
        const ::testing::TestInfo *test = ::testing::UnitTest::GetInstance()->current_test_info();
        const String test_name = test ? String(test->test_suite_name()) + "." + test->name() : "outside_a_test";
        root = std::filesystem::temp_directory_path() / ("spikecorec_tests_" + std::to_string(getpid())) / test_name;
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root);
    }

    ~Temporary_Directory() {
        Error_Code ignored;
        std::filesystem::remove_all(root, ignored);
    }

    Temporary_Directory(const Temporary_Directory &) = delete;
    Temporary_Directory &operator=(const Temporary_Directory &) = delete;

    // Writes a file, creating any directories on the way, and returns its absolute path.
    String write(const String &relative_path, const String &contents) const {
        const Path destination = root / relative_path;
        std::filesystem::create_directories(destination.parent_path());
        Output_File_Stream file(destination);
        file << contents;
        return destination.string();
    }

    [[nodiscard]] String path_of(const String &relative_path) const { return (root / relative_path).string(); }

    [[nodiscard]] String path() const { return root.string(); }

private:
    Path root;
};

// Captures the engine's log output in memory while alive, so a test can check for a warning.
class Captured_Log {
public:
    Captured_Log() : sink(std::make_shared<log::Ostream_Sink>(text)) {
        sink->set_pattern("[%l] %v");
        log::logger().sinks().push_back(sink);
    }

    ~Captured_Log() {
        Vector<log::Sink_Pointer> &sinks = log::logger().sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink), sinks.end());
    }

    Captured_Log(const Captured_Log &) = delete;
    Captured_Log &operator=(const Captured_Log &) = delete;

    [[nodiscard]] bool contains(const String &fragment) {
        log::logger().flush();
        return text.str().find(fragment) != String::npos;
    }

private:
    Output_String_Stream text;
    Shared_Pointer<log::Ostream_Sink> sink;
};

// A LEMS document that includes the NeuroML core types and model_file and simulates network_id.
// simulation_children goes inside <Simulation>, e.g. output files.
inline String lems_document(const String &model_file, const String &network_id, const String &length,
                            const String &step, const String &simulation_children = "") {
    Output_String_Stream document;
    document << "<Lems>\n"
             << "  <Include file=\"Cells.xml\"/>\n"
             << "  <Include file=\"Synapses.xml\"/>\n"
             << "  <Include file=\"Inputs.xml\"/>\n"
             << "  <Include file=\"Networks.xml\"/>\n"
             << "  <Include file=\"Simulation.xml\"/>\n"
             << "  <Include file=\"" << model_file << "\"/>\n"
             << "  <Simulation id=\"sim1\" length=\"" << length << "\" step=\"" << step
             << "\" target=\"" << network_id << "\">\n"
             << simulation_children
             << "  </Simulation>\n"
             << "  <Target component=\"sim1\"/>\n"
             << "</Lems>\n";
    return document.str();
}

// What fit_tolerance allows a read of `value` to be off by: the tolerance times the larger of the
// value and the RMS of its plane's declared values.
inline f64 fit_allowance(f32 fit_tolerance, f64 value, const Vector<f32> &declared_plane) {
    f64 sum_of_squares = 0.0;
    for (f32 entry : declared_plane) sum_of_squares += static_cast<f64>(entry) * static_cast<f64>(entry);
    const f64 root_mean_square =
        declared_plane.empty() ? 0.0 : std::sqrt(sum_of_squares / static_cast<f64>(declared_plane.size()));
    return static_cast<f64>(fit_tolerance) * std::max(std::fabs(value), root_mean_square);
}

// One neuron's spike times, in order.
inline Vector<f64> spike_times_of(const Spike_Engine &engine, s64 neuron_index) {
    Vector<f64> times;
    for (const Recorded_Spike &spike : engine.recorded_spikes) {
        if (spike.neuron_index == neuron_index) times.push_back(spike.time_seconds);
    }
    return times;
}

inline Vector<f64> interspike_intervals(const Vector<f64> &spike_times) {
    Vector<f64> intervals;
    for (usize index = 1; index < spike_times.size(); index += 1) {
        intervals.push_back(spike_times[index] - spike_times[index - 1]);
    }
    return intervals;
}

inline f64 mean_of(const Vector<f64> &values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<f64>(values.size());
}

// Standard deviation over mean: 0 for a perfectly regular spike train, about 1 for a Poisson one.
inline f64 coefficient_of_variation(const Vector<f64> &values) {
    const f64 mean = mean_of(values);
    if (values.size() < 2 || mean == 0.0) return 0.0;
    f64 sum_of_squares = 0.0;
    for (f64 value : values) sum_of_squares += (value - mean) * (value - mean);
    return std::sqrt(sum_of_squares / static_cast<f64>(values.size() - 1)) / mean;
}

// The two-sample Kolmogorov-Smirnov statistic: the largest gap between the two empirical CDFs.
inline f64 kolmogorov_smirnov_statistic(Vector<f64> first, Vector<f64> second) {
    std::sort(first.begin(), first.end());
    std::sort(second.begin(), second.end());
    usize first_index = 0;
    usize second_index = 0;
    f64 largest_gap = 0.0;
    while (first_index < first.size() && second_index < second.size()) {
        const f64 value = std::min(first[first_index], second[second_index]);
        while (first_index < first.size() && first[first_index] <= value) first_index += 1;
        while (second_index < second.size() && second[second_index] <= value) second_index += 1;
        largest_gap = std::max(largest_gap, std::fabs(static_cast<f64>(first_index) / static_cast<f64>(first.size()) -
                                                      static_cast<f64>(second_index) / static_cast<f64>(second.size())));
    }
    return largest_gap;
}

// The statistic above which two samples of these sizes differ at significance level `alpha`:
// c(alpha) * sqrt((n + m) / (n m)), with c(alpha) = sqrt(-ln(alpha / 2) / 2).
inline f64 kolmogorov_smirnov_critical_value(usize first_size, usize second_size, f64 alpha) {
    const f64 coefficient = std::sqrt(-std::log(alpha / 2.0) / 2.0);
    return coefficient * std::sqrt(static_cast<f64>(first_size + second_size) / (static_cast<f64>(first_size) * static_cast<f64>(second_size)));
}

} // namespace spikecorec::test_support
