#pragma once

#include "spikecorec/core/types.h"

namespace spikecorec::time {

s64 tick_count_from_ms(f64 total_ms, f64 ms_step);
s64 tick_count_from_seconds(f64 total_seconds, f64 seconds_step);
f64 ms_to_seconds(f64 ms);
f64 seconds_to_ms(f64 seconds);
f64 tick_to_ms(s64 tick, f64 total_ms, s64 total_ticks);
f64 tick_to_seconds(s64 tick, f64 total_seconds, s64 total_ticks);

// Converts a NeuroML/LEMS quantity string (e.g. "-70mV", "0.5nA", "3") to its
// canonical SI value (e.g. -0.07, 5e-10, 3.0). A bare number with no unit
// suffix is treated as already-dimensionless and passed through unchanged.
// Throws std::invalid_argument if the string has no parsable leading number,
// or if the unit suffix isn't a recognized NeuroML/LEMS unit symbol.
f64 unit_value_to_si(const String &value_text);

} // end namespace











