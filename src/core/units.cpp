#include <cmath>
#include <stdexcept>

#include "spikecorec/core/units.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::units {

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

// Covers the suffixes the NeuroML standard library actually uses. Everything the engine
// sees is SI, so conversion happens at parse time and never downstream.
f64 unit_suffix_scale(const String &suffix) {
    static const UnorderedMap<String, f64> scales = {
        {"", 1.0}, {"none", 1.0},
        {"V", 1.0}, {"mV", 1e-3},
        {"A", 1.0}, {"mA", 1e-3}, {"uA", 1e-6}, {"nA", 1e-9}, {"pA", 1e-12},
        {"S", 1.0}, {"mS", 1e-3}, {"uS", 1e-6}, {"nS", 1e-9}, {"pS", 1e-12},
        {"F", 1.0}, {"mF", 1e-3}, {"uF", 1e-6}, {"nF", 1e-9}, {"pF", 1e-12},
        {"s", 1.0}, {"ms", 1e-3}, {"us", 1e-6},
        {"Hz", 1.0}, {"per_s", 1.0}, {"per_ms", 1e3},
        {"m", 1.0}, {"cm", 1e-2}, {"um", 1e-6},
        {"M", 1.0}, {"mM", 1e-3},
        {"ohm", 1.0}, {"kohm", 1e3}, {"Mohm", 1e6},
        {"degC", 1.0}, {"K", 1.0},
    };

    auto entry = scales.find(suffix);
    if (entry == scales.end()) return 1.0;

    return entry->second;
}

} // namespace spikecorec::units
