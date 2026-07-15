#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

#include "spikecorec/core/types.h"
#include "spikecorec/core/time_utilities.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::time;

namespace {

bool approx(f64 first, f64 second, f64 epsilon = 1e-9) {
    return std::fabs(first - second) <= epsilon * (1.0 + std::fabs(second));
}

} // namespace

// ── tick_count_from_ms ────────────────────────────────────────────────────────

TEST(TickCountFromMs, typical_values) {
    EXPECT_EQ(tick_count_from_ms(1000.0, 0.025), 40000);
    EXPECT_EQ(tick_count_from_ms(1000.0, 0.1),   10000);
    EXPECT_EQ(tick_count_from_ms(500.0,  0.5),   1000);
    EXPECT_EQ(tick_count_from_ms(1.0,    1.0),   1);
}

TEST(TickCountFromMs, rounds_to_nearest_tick) {
    // 10.0 / 3.0 = 3.333... → rounds to 3
    EXPECT_EQ(tick_count_from_ms(10.0, 3.0), 3);
    // 10.0 / 4.0 = 2.5 → rounds to 3 (half-away-from-zero)
    EXPECT_EQ(tick_count_from_ms(10.0, 4.0), 3);
}

TEST(TickCountFromMs, step_equal_to_duration_gives_one_tick) {
    EXPECT_EQ(tick_count_from_ms(100.0, 100.0), 1);
}

TEST(TickCountFromMs, invalid_zero_step) {
    EXPECT_THROW(tick_count_from_ms(1000.0, 0.0),  invalid_argument);
}

TEST(TickCountFromMs, invalid_negative_step) {
    EXPECT_THROW(tick_count_from_ms(1000.0, -1.0), invalid_argument);
}

TEST(TickCountFromMs, invalid_zero_duration) {
    EXPECT_THROW(tick_count_from_ms(0.0, 0.025),   invalid_argument);
}

TEST(TickCountFromMs, invalid_negative_duration) {
    EXPECT_THROW(tick_count_from_ms(-500.0, 0.025), invalid_argument);
}

// ── tick_count_from_seconds ───────────────────────────────────────────────────

TEST(TickCountFromSeconds, typical_values) {
    EXPECT_EQ(tick_count_from_seconds(1.0,    0.000025), 40000);
    EXPECT_EQ(tick_count_from_seconds(1.0,    0.0001),   10000);
    EXPECT_EQ(tick_count_from_seconds(0.5,    0.0005),   1000);
}

TEST(TickCountFromSeconds, matches_ms_variant) {
    // tick_count_from_seconds(1.0, 0.000025) == tick_count_from_ms(1000.0, 0.025)
    EXPECT_EQ(tick_count_from_seconds(1.0, 0.000025),
              tick_count_from_ms(1000.0, 0.025));
}

TEST(TickCountFromSeconds, invalid_zero_step) {
    EXPECT_THROW(tick_count_from_seconds(1.0, 0.0),   invalid_argument);
}

TEST(TickCountFromSeconds, invalid_negative_step) {
    EXPECT_THROW(tick_count_from_seconds(1.0, -0.001), invalid_argument);
}

TEST(TickCountFromSeconds, invalid_zero_duration) {
    EXPECT_THROW(tick_count_from_seconds(0.0, 0.001),  invalid_argument);
}

TEST(TickCountFromSeconds, invalid_negative_duration) {
    EXPECT_THROW(tick_count_from_seconds(-1.0, 0.001), invalid_argument);
}

// ── ms_to_seconds / seconds_to_ms ────────────────────────────────────────────

TEST(UnitConversion, ms_to_seconds_typical) {
    EXPECT_TRUE(approx(ms_to_seconds(1000.0), 1.0));
    EXPECT_TRUE(approx(ms_to_seconds(500.0),  0.5));
    EXPECT_TRUE(approx(ms_to_seconds(25.0),   0.025));
    EXPECT_TRUE(approx(ms_to_seconds(0.0),    0.0));
}

TEST(UnitConversion, seconds_to_ms_typical) {
    EXPECT_TRUE(approx(seconds_to_ms(1.0),   1000.0));
    EXPECT_TRUE(approx(seconds_to_ms(0.5),   500.0));
    EXPECT_TRUE(approx(seconds_to_ms(0.025), 25.0));
    EXPECT_TRUE(approx(seconds_to_ms(0.0),   0.0));
}

TEST(UnitConversion, conversions_are_inverse) {
    for (f64 value : {0.0, 1.0, 0.025, 500.0, 1000.0}) {
        EXPECT_TRUE(approx(ms_to_seconds(seconds_to_ms(value)), value));
        EXPECT_TRUE(approx(seconds_to_ms(ms_to_seconds(value)), value));
    }
}

TEST(UnitConversion, ms_to_seconds_invalid_negative) {
    EXPECT_THROW(ms_to_seconds(-1.0),    invalid_argument);
}

TEST(UnitConversion, seconds_to_ms_invalid_negative) {
    EXPECT_THROW(seconds_to_ms(-0.001),  invalid_argument);
}

// ── tick_to_ms ────────────────────────────────────────────────────────────────

TEST(TickToMs, first_tick_is_zero) {
    EXPECT_TRUE(approx(tick_to_ms(0, 1000.0, 40000), 0.0));
}

TEST(TickToMs, last_tick_is_total_duration) {
    EXPECT_TRUE(approx(tick_to_ms(40000, 1000.0, 40000), 1000.0));
}

TEST(TickToMs, midpoint_tick) {
    // tick 20000 of 40000 → halfway through 1000ms → 500ms
    EXPECT_TRUE(approx(tick_to_ms(20000, 1000.0, 40000), 500.0));
}

TEST(TickToMs, single_tick_simulation) {
    EXPECT_TRUE(approx(tick_to_ms(0, 100.0, 1), 0.0));
    EXPECT_TRUE(approx(tick_to_ms(1, 100.0, 1), 100.0));
}

TEST(TickToMs, linear_spacing) {
    // Successive ticks should be evenly spaced by total_ms / total_ticks
    const f64 total_ms = 1000.0;
    const s64 total_ticks = 1000;
    const f64 expected_step = total_ms / static_cast<f64>(total_ticks);
    for (s64 tick = 0; tick < total_ticks; ++tick) {
        f64 expected = tick * expected_step;
        EXPECT_TRUE(approx(tick_to_ms(tick, total_ms, total_ticks), expected));
    }
}

TEST(TickToMs, invalid_zero_total_ticks) {
    EXPECT_THROW(tick_to_ms(0, 1000.0, 0),    invalid_argument);
}

TEST(TickToMs, invalid_negative_total_ticks) {
    EXPECT_THROW(tick_to_ms(0, 1000.0, -1),   invalid_argument);
}

TEST(TickToMs, invalid_negative_tick) {
    EXPECT_THROW(tick_to_ms(-1, 1000.0, 1000), invalid_argument);
}

TEST(TickToMs, invalid_zero_total_ms) {
    EXPECT_THROW(tick_to_ms(0, 0.0, 1000),    invalid_argument);
}

TEST(TickToMs, invalid_negative_total_ms) {
    EXPECT_THROW(tick_to_ms(0, -1.0, 1000),   invalid_argument);
}

// ── tick_to_seconds ───────────────────────────────────────────────────────────

TEST(TickToSeconds, first_tick_is_zero) {
    EXPECT_TRUE(approx(tick_to_seconds(0, 1.0, 40000), 0.0));
}

TEST(TickToSeconds, last_tick_is_total_duration) {
    EXPECT_TRUE(approx(tick_to_seconds(40000, 1.0, 40000), 1.0));
}

TEST(TickToSeconds, midpoint_tick) {
    EXPECT_TRUE(approx(tick_to_seconds(20000, 1.0, 40000), 0.5));
}

TEST(TickToSeconds, matches_ms_variant_converted) {
    // tick_to_seconds(n, s, t) == ms_to_seconds(tick_to_ms(n, s*1000, t))
    for (s64 tick : {0, 100, 20000, 40000}) {
        f64 via_seconds = tick_to_seconds(tick, 1.0, 40000);
        f64 via_ms      = ms_to_seconds(tick_to_ms(tick, 1000.0, 40000));
        EXPECT_TRUE(approx(via_seconds, via_ms));
    }
}

TEST(TickToSeconds, invalid_zero_total_ticks) {
    EXPECT_THROW(tick_to_seconds(0, 1.0, 0),    invalid_argument);
}

TEST(TickToSeconds, invalid_negative_tick) {
    EXPECT_THROW(tick_to_seconds(-1, 1.0, 1000), invalid_argument);
}

TEST(TickToSeconds, invalid_negative_total_seconds) {
    EXPECT_THROW(tick_to_seconds(0, -1.0, 1000), invalid_argument);
}

// ── unit_value_to_si ──────────────────────────────────────────────────────────

TEST(UnitValueToSi, converts_millivolts) {
    EXPECT_TRUE(approx(unit_value_to_si("-70mV"), -0.07));
}

TEST(UnitValueToSi, converts_nanoamps) {
    EXPECT_TRUE(approx(unit_value_to_si("0.5nA"), 5e-10));
}

TEST(UnitValueToSi, converts_nanosiemens) {
    EXPECT_TRUE(approx(unit_value_to_si("5nS"), 5e-9));
}

TEST(UnitValueToSi, converts_nanofarads) {
    EXPECT_TRUE(approx(unit_value_to_si("1.0nF"), 1e-9));
}

TEST(UnitValueToSi, converts_microsiemens) {
    EXPECT_TRUE(approx(unit_value_to_si("0.05uS"), 5e-8));
}

TEST(UnitValueToSi, converts_milliseconds) {
    EXPECT_TRUE(approx(unit_value_to_si("3ms"), 0.003));
}

TEST(UnitValueToSi, passes_through_dimensionless_bare_number) {
    EXPECT_TRUE(approx(unit_value_to_si("0.02"), 0.02));
    EXPECT_TRUE(approx(unit_value_to_si("1"), 1.0));
}

TEST(UnitValueToSi, converts_non_power_of_ten_minute_scale) {
    EXPECT_TRUE(approx(unit_value_to_si("2min"), 120.0));
}

TEST(UnitValueToSi, throws_on_unknown_unit_symbol) {
    EXPECT_THROW(unit_value_to_si("5zz"), invalid_argument);
}

TEST(UnitValueToSi, throws_on_malformed_numeric_prefix) {
    EXPECT_THROW(unit_value_to_si("abc"), invalid_argument);
}
