#pragma once

// What more than one test file needs: one GPU backend, a temporary directory per test, the
// fixture directory, a LEMS document around a model, and spike statistics.

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <gtest/gtest.h>
#include <spdlog/sinks/ostream_sink.h>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/engine.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/types.h"
#include "spikecorec/nml/parser.h"

#ifndef SPIKECOREC_TEST_FIXTURE_DIR
#error "SPIKECOREC_TEST_FIXTURE_DIR must name tests/fixtures; the Makefile defines it"
#endif

namespace spikecorec::test_support {

// One backend for the whole runner. Every k^2-tree and weight matrix carves its own slab out of
// it and releases that slab when it dies, so sharing the backend is not sharing storage; it only
// avoids standing up a device per test.
inline EngineBackend &shared_backend() {
    static EngineBackend backend;
    return backend;
}

// An absolute path into tests/fixtures, so a test finds its fixture from any working directory.
inline String fixture_path(const String &relative_path) {
    return String(SPIKECOREC_TEST_FIXTURE_DIR) + "/" + relative_path;
}

// True when the NeuroML standard library the parser includes from is on disk.
inline bool standard_library_available() {
    const NML_Context context;
    return !context.STANDARD_LIBRARY_PATH.empty() && std::filesystem::exists(context.STANDARD_LIBRARY_PATH);
}

// A directory of the current test's own, removed with the object. It sits under a root named
// after this process, so test runners that overlap on one machine never share a file.
class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const ::testing::TestInfo *test = ::testing::UnitTest::GetInstance()->current_test_info();
        const String test_name = test ? String(test->test_suite_name()) + "." + test->name() : "outside_a_test";
        root = std::filesystem::temp_directory_path() / ("spikecorec_tests_" + std::to_string(getpid())) / test_name;
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root);
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
    }

    TemporaryDirectory(const TemporaryDirectory &) = delete;
    TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

    // Writes a file, creating any directories on the way, and returns its absolute path.
    String write(const String &relative_path, const String &contents) const {
        const std::filesystem::path destination = root / relative_path;
        std::filesystem::create_directories(destination.parent_path());
        std::ofstream file(destination);
        file << contents;
        return destination.string();
    }

    [[nodiscard]] String path_of(const String &relative_path) const { return (root / relative_path).string(); }

    [[nodiscard]] String path() const { return root.string(); }

private:
    std::filesystem::path root;
};

// Everything the engine logs while the object lives, kept in memory so a test can check that a
// warning was given without reading the log file.
class CapturedLog {
public:
    CapturedLog() : sink(std::make_shared<spdlog::sinks::ostream_sink_mt>(text)) {
        sink->set_pattern("[%l] %v");
        log::logger().sinks().push_back(sink);
    }

    ~CapturedLog() {
        std::vector<log::SinkPointer> &sinks = log::logger().sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink), sinks.end());
    }

    CapturedLog(const CapturedLog &) = delete;
    CapturedLog &operator=(const CapturedLog &) = delete;

    [[nodiscard]] bool contains(const String &fragment) {
        log::logger().flush();
        return text.str().find(fragment) != String::npos;
    }

private:
    std::ostringstream text;
    std::shared_ptr<spdlog::sinks::ostream_sink_mt> sink;
};

// A LEMS document that includes the NeuroML core types and model_file and simulates network_id.
// simulation_children goes inside <Simulation>: output files, for instance.
inline String lems_document(const String &model_file, const String &network_id, const String &length,
                            const String &step, const String &simulation_children = "") {
    std::ostringstream document;
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

// One neuron's spike times, in order.
inline Vector<f64> spike_times_of(const SpikeEngine &engine, s64 neuron_index) {
    Vector<f64> times;
    for (const RecordedSpike &spike : engine.recorded_spikes) {
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
    return std::accumulate(values.begin(), values.end(), 0.0) / (f64)values.size();
}

// Standard deviation over mean: 0 for a perfectly regular spike train, about 1 for a Poisson one.
inline f64 coefficient_of_variation(const Vector<f64> &values) {
    const f64 mean = mean_of(values);
    if (values.size() < 2 || mean == 0.0) return 0.0;
    f64 sum_of_squares = 0.0;
    for (f64 value : values) sum_of_squares += (value - mean) * (value - mean);
    return std::sqrt(sum_of_squares / (f64)(values.size() - 1)) / mean;
}

} // namespace spikecorec::test_support
