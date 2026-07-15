#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include "spikecorec/core/time_utilities.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::time {

namespace {

// Hand-picked subset of third_party/neuroml2/std_lib/NeuroMLCoreDimensions.xml's
// <Unit symbol dimension power scale offset> declarations, covering the units
// Phase-1 GLIF/synapse/generator parameters actually use. Scale is 10^power
// for every entry here except "min"/"hour", which the source file itself
// gives an explicit non-power-of-ten scale= override for. Kept as a
// hand-maintained mirror rather than parsed from the file at runtime
// (deliberate choice) — if NeuroMLCoreDimensions.xml gains/changes a unit
// this table needs, update this table to match by hand.
const UnorderedMap<String, f64> UNIT_SYMBOL_TO_SCALE = {
    {"s", 1.0}, {"ms", 1e-3}, {"min", 60.0}, {"hour", 3600.0},
    {"V", 1.0}, {"mV", 1e-3},
    {"ohm", 1.0}, {"kohm", 1e3}, {"Mohm", 1e6},
    {"S", 1.0}, {"mS", 1e-3}, {"uS", 1e-6}, {"nS", 1e-9}, {"pS", 1e-12},
    {"F", 1.0}, {"uF", 1e-6}, {"nF", 1e-9}, {"pF", 1e-12},
    {"A", 1.0}, {"uA", 1e-6}, {"nA", 1e-9}, {"pA", 1e-12},
};

} // namespace

s64 tick_count_from_ms(f64 total_ms, f64 ms_step) {
    if (ms_step <= 0.0) throw invalid_argument("ms_step must be > 0");
    if (total_ms <= 0.0) throw invalid_argument("total_ms must be > 0");

    return static_cast<s64>(std::round(total_ms / ms_step));
}

s64 tick_count_from_seconds(f64 total_seconds, f64 seconds_step) {
    if (seconds_step <= 0.0) throw invalid_argument("seconds_step must be > 0");
    if (total_seconds <= 0.0) throw invalid_argument("total_seconds must be > 0");

    return tick_count_from_ms(seconds_to_ms(total_seconds), seconds_to_ms(seconds_step));
}

f64 ms_to_seconds(f64 ms) {
    if (ms < 0.0) throw invalid_argument("ms must be >= 0");
    return ms / 1000.0;
}

f64 seconds_to_ms(f64 seconds) {
    if (seconds < 0.0) throw invalid_argument("seconds must be >= 0");
    return seconds * 1000.0;
}

f64 tick_to_ms(s64 tick, f64 total_ms, s64 total_ticks) {
    if (total_ticks <= 0) throw invalid_argument("total_ticks must be > 0");
    if (tick < 0)         throw invalid_argument("tick must be >= 0");
    if (total_ms <= 0.0)  throw invalid_argument("total_ms must be > 0");

    return (static_cast<f64>(tick) / static_cast<f64>(total_ticks)) * total_ms;
}

f64 tick_to_seconds(s64 tick, f64 total_seconds, s64 total_ticks) {
    if (total_ticks <= 0)    throw invalid_argument("total_ticks must be > 0");
    if (tick < 0)            throw invalid_argument("tick must be >= 0");
    if (total_seconds <= 0.0) throw invalid_argument("total_seconds must be > 0");

    return ms_to_seconds(
        tick_to_ms(tick, seconds_to_ms(total_seconds), total_ticks)
    );
}

f64 unit_value_to_si(const String &value_text) {
    const char *text_start = value_text.c_str();
    char *number_end = nullptr;
    f64 numeric_value = std::strtod(text_start, &number_end);

    if (number_end == text_start) {
        throw invalid_argument("unit_value_to_si: no parsable leading number in '" + value_text + "'");
    }

    String unit_symbol(number_end);
    if (unit_symbol.empty()) return numeric_value;

    auto scale_entry = UNIT_SYMBOL_TO_SCALE.find(unit_symbol);
    if (scale_entry == UNIT_SYMBOL_TO_SCALE.end()) {
        throw invalid_argument("unit_value_to_si: unknown unit symbol '" + unit_symbol + "' in '" + value_text + "'");
    }

    return numeric_value * scale_entry->second;
}

} // namespace spikecorec::time
