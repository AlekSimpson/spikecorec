#pragma once

#include "spikecorec/core/types.h"

namespace spikecorec::time {

s64 tick_count_from_ms(f64 total_ms, f64 ms_step);
s64 tick_count_from_seconds(f64 total_seconds, f64 seconds_step);
f64 ms_to_seconds(f64 ms);
f64 seconds_to_ms(f64 seconds);
f64 tick_to_ms(s64 tick, f64 total_ms, s64 total_ticks);
f64 tick_to_seconds(s64 tick, f64 total_seconds, s64 total_ticks);

} // end namespace











