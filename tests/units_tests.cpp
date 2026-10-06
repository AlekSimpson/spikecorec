#include <cmath>
#include <limits>
#include <stdexcept>
#include <gtest/gtest.h>

#include "spikecorec/core/types.h"
#include "spikecorec/core/units.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::units;

namespace {

constexpr f64 NOT_A_NUMBER = numeric_limits<f64>::quiet_NaN();
constexpr f64 INFINITE = numeric_limits<f64>::infinity();

} // namespace

// ── durations to ticks ──────────────────────────────────────────────────────────

TEST(TickConversion, rounds_to_the_nearest_tick) {
    // 2 / 0.01 is 199.99999999999997 in doubles; rounding, not truncation, makes it 200.
    EXPECT_EQ(seconds_to_ticks(2e-3, 1e-5), 200);
    EXPECT_EQ(seconds_to_ticks(1.0, 1e-4), 10000);
    EXPECT_EQ(seconds_to_ticks(1.0, 2.5e-5), 40000);
    EXPECT_EQ(ms_to_ticks(10.0, 3.0), 3);   // 3.33 rounds down
    EXPECT_EQ(ms_to_ticks(10.0, 4.0), 3);   // 2.5 rounds away from zero
    EXPECT_EQ(ms_to_ticks(100.0, 100.0), 1);
}

TEST(TickConversion, seconds_and_milliseconds_agree) {
    const Vector<Pair<f64, f64>> durations_and_steps = {{1.0, 2.5e-5}, {0.5, 5e-4}, {3e-3, 1e-4}};
    for (const auto &[seconds, step_seconds] : durations_and_steps) {
        EXPECT_EQ(seconds_to_ticks(seconds, step_seconds), ms_to_ticks(seconds * 1000.0, step_seconds * 1000.0));
    }
}

TEST(TickConversion, a_zero_duration_is_zero_ticks) {
    EXPECT_EQ(seconds_to_ticks(0.0, 1e-4), 0);
    EXPECT_EQ(ms_to_ticks(0.0, 0.1), 0);
}

TEST(TickConversion, a_step_that_is_not_positive_and_finite_throws) {
    for (f64 step : {0.0, -0.1, NOT_A_NUMBER, INFINITE}) {
        EXPECT_THROW(ms_to_ticks(10.0, step), invalid_argument) << "step " << step;
        EXPECT_THROW(seconds_to_ticks(1.0, step), invalid_argument) << "step " << step;
    }
}

TEST(TickConversion, a_duration_that_is_negative_or_not_finite_throws) {
    for (f64 duration : {-1.0, NOT_A_NUMBER, INFINITE}) {
        EXPECT_THROW(ms_to_ticks(duration, 0.1), invalid_argument) << "duration " << duration;
        EXPECT_THROW(seconds_to_ticks(duration, 1e-4), invalid_argument) << "duration " << duration;
    }
}

TEST(TickConversion, a_tick_count_past_s64_throws) {
    EXPECT_THROW(ms_to_ticks(1e300, 1e-300), invalid_argument);
    EXPECT_THROW(seconds_to_ticks(1e300, 1e-30), invalid_argument);
}

// ── milliseconds and seconds ────────────────────────────────────────────────────

TEST(TimeConversion, milliseconds_and_seconds_are_inverse) {
    EXPECT_DOUBLE_EQ(ms_to_seconds(25.0), 0.025);
    EXPECT_DOUBLE_EQ(seconds_to_ms(0.025), 25.0);
    for (f64 value : {0.0, 0.025, 1.0, 500.0}) {
        EXPECT_DOUBLE_EQ(ms_to_seconds(seconds_to_ms(value)), value);
        EXPECT_DOUBLE_EQ(seconds_to_ms(ms_to_seconds(value)), value);
    }
    EXPECT_THROW(ms_to_seconds(-1.0), invalid_argument);
    EXPECT_THROW(seconds_to_ms(-1e-3), invalid_argument);
}

// ── ticks to times ──────────────────────────────────────────────────────────────

TEST(TickToTime, spaces_ticks_evenly_over_the_duration) {
    EXPECT_DOUBLE_EQ(tick_to_ms(0, 1000.0, 40000), 0.0);
    EXPECT_DOUBLE_EQ(tick_to_ms(20000, 1000.0, 40000), 500.0);
    EXPECT_DOUBLE_EQ(tick_to_ms(40000, 1000.0, 40000), 1000.0);
    for (s64 tick : {0, 1, 123, 999}) {
        EXPECT_NEAR(tick_to_ms(tick, 1000.0, 1000), (f64)tick, 1e-9);
        EXPECT_NEAR(tick_to_seconds(tick, 1.0, 1000), (f64)tick * 1e-3, 1e-12);
    }
}

TEST(TickToTime, invalid_arguments_throw) {
    EXPECT_THROW(tick_to_ms(0, 1000.0, 0), invalid_argument);
    EXPECT_THROW(tick_to_ms(-1, 1000.0, 1000), invalid_argument);
    EXPECT_THROW(tick_to_ms(0, 0.0, 1000), invalid_argument);
    EXPECT_THROW(tick_to_seconds(0, 1.0, 0), invalid_argument);
    EXPECT_THROW(tick_to_seconds(-1, 1.0, 1000), invalid_argument);
    EXPECT_THROW(tick_to_seconds(0, -1.0, 1000), invalid_argument);
}

// ── quantities ──────────────────────────────────────────────────────────────────

TEST(Quantity, splits_the_magnitude_from_the_unit_suffix) {
    EXPECT_EQ(split_quantity("-60mV"), (Pair<f64, String>{-60.0, "mV"}));
    EXPECT_EQ(split_quantity("1.5e-3 mV"), (Pair<f64, String>{1.5e-3, "mV"}));
    EXPECT_EQ(split_quantity("  20 ms "), (Pair<f64, String>{20.0, "ms"}));
    EXPECT_EQ(split_quantity("+5"), (Pair<f64, String>{5.0, ""}));
    EXPECT_EQ(split_quantity("10"), (Pair<f64, String>{10.0, ""}));
}

TEST(Quantity, scales_to_si_with_the_built_in_table) {
    EXPECT_DOUBLE_EQ(parse_quantity("-60mV"), -0.06);
    EXPECT_DOUBLE_EQ(parse_quantity("90 pA"), 90e-12);
    EXPECT_DOUBLE_EQ(parse_quantity("0.2nF"), 0.2e-9);
    EXPECT_DOUBLE_EQ(parse_quantity("5ms"), 5e-3);
    EXPECT_DOUBLE_EQ(parse_quantity("0.03per_ms"), 30.0);
    EXPECT_DOUBLE_EQ(parse_quantity("10"), 10.0);

    EXPECT_DOUBLE_EQ(unit_suffix_scale("nS"), 1e-9);
    EXPECT_DOUBLE_EQ(unit_suffix_scale("um"), 1e-6);
    EXPECT_DOUBLE_EQ(unit_suffix_scale("Mohm"), 1e6);
    // An unknown suffix scales by 1; the document-aware NML_Context::resolve_quantity knows more.
    EXPECT_DOUBLE_EQ(unit_suffix_scale("quux"), 1.0);
}

TEST(Quantity, malformed_input_yields_zero_rather_than_throwing) {
    EXPECT_DOUBLE_EQ(parse_quantity(""), 0.0);
    EXPECT_DOUBLE_EQ(parse_quantity("mV"), 0.0);
    EXPECT_DOUBLE_EQ(parse_quantity("   "), 0.0);
    EXPECT_EQ(split_quantity("abc"), (Pair<f64, String>{0.0, ""}));
}
