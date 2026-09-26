#pragma once

#include "spikecorec/core/types.h"

namespace spikecorec::units {

// A <Unit> declaration: the affine map from a written magnitude onto its dimension's SI
// unit. LEMS defines it as `si = raw * scale * 10^power + offset`; `power` is folded into
// `scale` at ingest, and `offset` is non-zero only for degC.
struct UnitDefinition {
    f64 scale = 1.0;
    f64 offset = 0.0;
};

s64 tick_count_from_ms(f64 total_ms, f64 ms_step);
s64 tick_count_from_seconds(f64 total_seconds, f64 seconds_step);
f64 ms_to_seconds(f64 ms);
f64 seconds_to_ms(f64 seconds);
f64 tick_to_ms(s64 tick, f64 total_ms, s64 total_ticks);
f64 tick_to_seconds(s64 tick, f64 total_seconds, s64 total_ticks);
s64 seconds_to_ticks(f64 seconds, f64 step_dt);

// SI scale for a NeuroML unit suffix, e.g. "mV" -> 1e-3. Unknown suffixes scale by 1.
f64 unit_suffix_scale(const String &suffix);

// "-60mV" -> -0.06 using only the built-in unit table. NML_Parser::resolve_quantity is
// the document-aware form, and is what the parser itself uses.
f64 parse_quantity(const String &value);

// Splits "-60mV" into its numeric magnitude and unit suffix, so parse_quantity and
// resolve_quantity share one number scanner.
Pair<f64, String> split_quantity(const String &value);

} // end namespace











