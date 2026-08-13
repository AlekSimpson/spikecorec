#pragma once

#include "spikecorec/core/types.h"

namespace spikecorec::units {

s64 tick_count_from_ms(f64 total_ms, f64 ms_step);
s64 tick_count_from_seconds(f64 total_seconds, f64 seconds_step);
f64 ms_to_seconds(f64 ms);
f64 seconds_to_ms(f64 seconds);
f64 tick_to_ms(s64 tick, f64 total_ms, s64 total_ticks);
f64 tick_to_seconds(s64 tick, f64 total_seconds, s64 total_ticks);

// SI scale for a NeuroML unit suffix, e.g. "mV" -> 1e-3. Covers every unit symbol the
// vendored standard library declares that converts by a scale, plus "" and "none" for
// dimensionless values.
//
// THROWS std::invalid_argument on a suffix it does not know, naming it. Scaling an
// unrecognised suffix by 1.0 instead would read "2min" as 2 seconds and a misspelling as a
// plausible number, with no diagnostic anywhere.
//
// THROWS on "degC" too, which the standard library declares as offset="273.15": an offset
// cannot be expressed as a scale, so temperatures must be resolved through a
// document-declared <Unit> (NML_Parser::resolve_quantity), which does apply offsets.
f64 unit_suffix_scale(const String &suffix);

} // end namespace











