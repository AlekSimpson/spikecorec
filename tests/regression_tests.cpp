// Exact spike times, held against the engine's own recorded runs in tests/fixtures/golden. These
// catch any change in what the engine computes; whether it computes the right thing is what the
// closed-form, statistical and jNeuroML tests check. After a change meant to alter the numbers,
// re-record the files and review their diff:
//
//   SPIKECOREC_UPDATE_GOLDEN=1 make test TEST_ARGUMENTS=--gtest_filter='*Regression*'
//
// The files belong to the machine they were recorded on: another GPU may round differently.

#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>

#include "spikecorec/core/engine.h"
#include "support/test_support.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::test_support;

namespace {

struct RegressionModel {
    String name;           // tests/fixtures/golden/<name>.spikes
    String lems_fixture;   // under tests/fixtures
};

// Every spike of a run as (tick, neuron), in the order the engine recorded them.
Vector<Pair<s64, s64>> run_and_collect(const String &lems_path) {
    SpikeEngine engine(lems_path);
    engine.run();
    Vector<Pair<s64, s64>> spikes;
    for (const RecordedSpike &spike : engine.recorded_spikes) {
        spikes.push_back({llround(spike.time_seconds / engine.step_dt), spike.neuron_index});
    }
    return spikes;
}

} // namespace

class Regression : public ::testing::TestWithParam<RegressionModel> {};

TEST_P(Regression, spike_times_match_the_recorded_run) {
    const RegressionModel &model = GetParam();
    const Vector<Pair<s64, s64>> spikes = run_and_collect(fixture_path(model.lems_fixture));
    ASSERT_FALSE(spikes.empty()) << model.name << " produced no spikes";
    const String golden_path = fixture_path("golden/" + model.name + ".spikes");

    if (std::getenv("SPIKECOREC_UPDATE_GOLDEN") != nullptr) {
        std::ofstream golden(golden_path);
        ASSERT_TRUE(golden.good()) << "cannot write " << golden_path;
        golden << "# tick neuron, recorded from the engine; see tests/regression_tests.cpp\n";
        for (const auto &[tick, neuron] : spikes) golden << tick << " " << neuron << "\n";
        GTEST_SKIP() << "recorded " << spikes.size() << " spikes to " << golden_path;
    }

    std::ifstream golden(golden_path);
    ASSERT_TRUE(golden.good()) << golden_path << " is missing; record it with SPIKECOREC_UPDATE_GOLDEN=1";
    Vector<Pair<s64, s64>> recorded;
    String line;
    while (getline(golden, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        s64 tick = 0;
        s64 neuron = 0;
        fields >> tick >> neuron;
        recorded.push_back({tick, neuron});
    }

    usize first_difference = 0;
    while (first_difference < min(spikes.size(), recorded.size()) && spikes[first_difference] == recorded[first_difference]) {
        first_difference += 1;
    }
    EXPECT_EQ(spikes.size(), recorded.size()) << model.name;
    if (first_difference < min(spikes.size(), recorded.size())) {
        ADD_FAILURE() << model.name << ": spike " << first_difference << " is (tick " << spikes[first_difference].first
                      << ", neuron " << spikes[first_difference].second << "), recorded (tick "
                      << recorded[first_difference].first << ", neuron " << recorded[first_difference].second << ")";
    }
}

INSTANTIATE_TEST_SUITE_P(RecordedRuns, Regression,
                         ::testing::Values(RegressionModel{"pynn_synapses", "nml/LEMS_pynn_synapses.xml"},
                                           RegressionModel{"glif_family", "nml/LEMS_glif_family.xml"},
                                           RegressionModel{"pynn_poisson_network", "nml/LEMS_pynn_poisson_network.xml"}),
                         [](const ::testing::TestParamInfo<RegressionModel> &info) { return info.param.name; });
