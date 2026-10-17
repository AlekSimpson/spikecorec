#pragma once

#include "types.h"

namespace spikecorec::units {

// A <Unit>: si = raw * scale + offset, with LEMS's 10^power folded into scale.
struct Unit_Definition {
    f64 scale = 1.0;
    f64 offset = 0.0;
    // The <Unit>'s dimension, e.g. "voltage"; "" when it states none.
    String dimension;
};

f64 ms_to_seconds(f64 ms);
s64 ms_to_ticks(f64 total_ms, f64 ms_step);
f64 seconds_to_ms(f64 seconds);
f64 tick_to_ms(s64 tick, f64 total_ms, s64 total_ticks);

f64 tick_to_seconds(s64 tick, f64 total_seconds, s64 total_ticks);
s64 seconds_to_ticks(f64 seconds, f64 step_dt);

// SI scale for a NeuroML unit suffix, e.g. "mV" -> 1e-3. Unknown suffixes scale by 1.
f64 unit_suffix_scale(const String &suffix);

// Whether the built-in table that unit_suffix_scale reads has the suffix.
bool unit_suffix_is_known(const String &suffix);

// "-60mV" -> -0.06 by the built-in table only. NML_Context::resolve_quantity also uses <Unit>s.
f64 parse_quantity(const String &value);

// Splits "-60mV" into {-60, "mV"}; {0, ""} when the value does not start with a number.
Pair<f64, String> split_quantity(const String &value);

} // end namespace











