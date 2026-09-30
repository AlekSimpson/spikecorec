#include <cmath>
#include <stdexcept>

#include "spikecorec/core/units.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::units {

s64 ms_to_ticks(f64 total_ms, f64 ms_step) {
    return static_cast<s64>(std::round(total_ms / ms_step));
}

s64 seconds_to_ticks(f64 total_seconds, f64 seconds_step) {
    return ms_to_ticks(seconds_to_ms(total_seconds), seconds_to_ms(seconds_step));
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

// Splits "-60mV" into its numeric magnitude and its unit suffix. Shared by parse_quantity
// and NML_Context::resolve_quantity so both scan numbers identically.
Pair<f64, String> split_quantity(const String &value) {
    if (value.empty()) return {0.0, ""};

    usize cursor = 0;
    while (cursor < value.size() && isspace(static_cast<unsigned char>(value[cursor]))) cursor += 1;

    usize number_start = cursor;
    if (cursor < value.size() && (value[cursor] == '-' || value[cursor] == '+')) cursor += 1;
    while (cursor < value.size() &&
           (isdigit(static_cast<unsigned char>(value[cursor])) || value[cursor] == '.')) {
        cursor += 1;
    }

    // Exponent, e.g. "1.5e-3mV".
    if (cursor < value.size() && (value[cursor] == 'e' || value[cursor] == 'E')) {
        usize exponent_cursor = cursor + 1;
        if (exponent_cursor < value.size() &&
            (value[exponent_cursor] == '-' || value[exponent_cursor] == '+')) {
            exponent_cursor += 1;
        }
        if (exponent_cursor < value.size() &&
            isdigit(static_cast<unsigned char>(value[exponent_cursor]))) {
            cursor = exponent_cursor;
            while (cursor < value.size() &&
                   isdigit(static_cast<unsigned char>(value[cursor]))) cursor += 1;
        }
    }

    if (cursor == number_start) return {0.0, ""};

    f64 magnitude = 0.0;
    try {
        magnitude = std::stod(value.substr(number_start, cursor - number_start));
    } catch (const std::exception &) {
        return {0.0, ""};
    }

    while (cursor < value.size() && isspace(static_cast<unsigned char>(value[cursor]))) cursor += 1;

    String suffix = value.substr(cursor);
    while (!suffix.empty() && isspace(static_cast<unsigned char>(suffix.back()))) suffix.pop_back();

    return {magnitude, suffix};
}

// "-60mV" -> -0.06 against the built-in table only.
f64 parse_quantity(const String &value) {
    auto [magnitude, suffix] = split_quantity(value);
    return magnitude * units::unit_suffix_scale(suffix);
}

} // namespace spikecorec::units
