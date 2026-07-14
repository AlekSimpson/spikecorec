#include <cmath>
#include <stdexcept>

#include "spikecorec/core/time_utilities.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::time {

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

} // namespace spikecorec::time
