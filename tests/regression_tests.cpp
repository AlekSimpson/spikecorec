// Spike statistics against the engine's own recorded runs in tests/fixtures/golden: per-neuron
// firing rates and pooled interspike intervals by two-sample Kolmogorov-Smirnov tests, plus the
// mean rate and the intervals' coefficient of variation. Exact spike times are never compared: a
// change within fit_tolerance can move a spike by a tick without changing the activity.
// Correctness is checked by the closed-form, statistical and jNeuroML tests. After an intended
// change to the numbers, re-record the files and review their diff:
//
//   SPIKECOREC_UPDATE_GOLDEN=1 make test TEST_ARGUMENTS=--gtest_filter='*Regression*'

#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>

#include "engine.h"
#include "support/test_support.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::test_support;

namespace {

struct Regression_Model {
    String name;           // tests/fixtures/golden/<name>.spikes
    String lems_fixture;   // under tests/fixtures
};

struct Recorded_Run {
    Vector<Pair<s64, s64>> spikes;  // (tick, neuron), in the order the engine recorded them
    s64 neuron_count = 0;
    s64 tick_count = 0;
};

Recorded_Run run_and_collect(const String &lems_path) {
    Spike_Engine engine(lems_path);
    engine.run();
    Recorded_Run run;
    run.neuron_count = engine.total_neuron_count;
    run.tick_count = engine.lifetime;
    for (const Recorded_Spike &spike : engine.recorded_spikes) {
        run.spikes.push_back({llround(spike.time_seconds / engine.step_dt), spike.neuron_index});
    }
    return run;
}

// Each neuron's spike count per tick, and every neuron's interspike intervals in ticks, pooled.
struct Spike_Statistics {
    Vector<f64> rates;
    Vector<f64> intervals;
};

Spike_Statistics statistics_of(const Vector<Pair<s64, s64>> &spikes, s64 neuron_count, s64 tick_count) {
    Vector<Vector<f64>> ticks_per_neuron(static_cast<usize>(neuron_count));
    for (const auto &[tick, neuron] : spikes) {
        if (neuron >= 0 && neuron < neuron_count) ticks_per_neuron[static_cast<usize>(neuron)].push_back(static_cast<f64>(tick));
    }
    Spike_Statistics statistics;
    for (Vector<f64> &ticks : ticks_per_neuron) {
        sort(ticks.begin(), ticks.end());
        statistics.rates.push_back(static_cast<f64>(ticks.size()) / static_cast<f64>(max<s64>(tick_count, 1)));
        const Vector<f64> intervals = interspike_intervals(ticks);
        statistics.intervals.insert(statistics.intervals.end(), intervals.begin(), intervals.end());
    }
    return statistics;
}

// Below this many pooled intervals the interval tests say nothing useful.
constexpr usize MINIMUM_INTERVAL_COUNT = 20;

} // namespace

class Regression : public ::testing::TestWithParam<Regression_Model> {};

TEST_P(Regression, spike_statistics_match_the_recorded_run) {
    const Regression_Model &model = GetParam();
    const Recorded_Run run = run_and_collect(fixture_path(model.lems_fixture));
    ASSERT_FALSE(run.spikes.empty()) << model.name << " produced no spikes";
    const String golden_path = fixture_path("golden/" + model.name + ".spikes");

    if (std::getenv("SPIKECOREC_UPDATE_GOLDEN") != nullptr) {
        Output_File_Stream golden(golden_path);
        ASSERT_TRUE(golden.good()) << "cannot write " << golden_path;
        golden << "# tick neuron, recorded from the engine; see tests/regression_tests.cpp\n";
        for (const auto &[tick, neuron] : run.spikes) golden << tick << " " << neuron << "\n";
        GTEST_SKIP() << "recorded " << run.spikes.size() << " spikes to " << golden_path;
    }

    Input_File_Stream golden(golden_path);
    ASSERT_TRUE(golden.good()) << golden_path << " is missing; record it with SPIKECOREC_UPDATE_GOLDEN=1";
    Vector<Pair<s64, s64>> recorded;
    String line;
    while (getline(golden, line)) {
        if (line.empty() || line[0] == '#') continue;
        Input_String_Stream fields(line);
        s64 tick = 0;
        s64 neuron = 0;
        fields >> tick >> neuron;
        recorded.push_back({tick, neuron});
    }
    ASSERT_FALSE(recorded.empty()) << golden_path << " holds no spikes";

    const Spike_Statistics engine_statistics = statistics_of(run.spikes, run.neuron_count, run.tick_count);
    const Spike_Statistics recorded_statistics = statistics_of(recorded, run.neuron_count, run.tick_count);

    EXPECT_NEAR(mean_of(engine_statistics.rates), mean_of(recorded_statistics.rates),
                0.10 * mean_of(recorded_statistics.rates)) << model.name << ": mean rate";
    EXPECT_LT(kolmogorov_smirnov_statistic(engine_statistics.rates, recorded_statistics.rates),
              kolmogorov_smirnov_critical_value(engine_statistics.rates.size(), recorded_statistics.rates.size(), 0.001))
        << model.name << ": per-neuron rates";

    if (engine_statistics.intervals.size() < MINIMUM_INTERVAL_COUNT ||
        recorded_statistics.intervals.size() < MINIMUM_INTERVAL_COUNT) {
        return;
    }
    EXPECT_LT(kolmogorov_smirnov_statistic(engine_statistics.intervals, recorded_statistics.intervals),
              kolmogorov_smirnov_critical_value(engine_statistics.intervals.size(), recorded_statistics.intervals.size(), 0.001))
        << model.name << ": interspike intervals";
    EXPECT_NEAR(coefficient_of_variation(engine_statistics.intervals), coefficient_of_variation(recorded_statistics.intervals),
                0.15 * coefficient_of_variation(recorded_statistics.intervals))
        << model.name << ": interval coefficient of variation";
}

INSTANTIATE_TEST_SUITE_P(RecordedRuns, Regression,
                         ::testing::Values(Regression_Model{"pynn_synapses", "nml/LEMS_pynn_synapses.xml"},
                                           Regression_Model{"glif_family", "nml/LEMS_glif_family.xml"},
                                           Regression_Model{"pynn_poisson_network", "nml/LEMS_pynn_poisson_network.xml"}),
                         [](const ::testing::TestParamInfo<Regression_Model> &info) { return info.param.name; });
