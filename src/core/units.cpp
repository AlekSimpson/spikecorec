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

// Every unit symbol the vendored NeuroML standard library declares, plus the dimensionless
// spellings and a few SI prefixes the library does not name but documents use (mA, mF, us).
// Everything the engine sees is SI, so conversion happens at parse time and never
// downstream; each scale below is the standard library's own 10^power * scale.
//
// The table is not a fallback for symbols it does not know. A suffix it cannot place is a
// misspelling or a unit nobody has taught this engine, and scaling by 1.0 silently turns
// "2min" into 2 seconds and a typo into a plausible number -- see the throw below.
f64 unit_suffix_scale(const String &suffix) {
    static const UnorderedMap<String, f64> scales = {
        {"", 1.0}, {"none", 1.0},
        {"V", 1.0}, {"mV", 1e-3}, {"per_V", 1.0}, {"per_mV", 1e3},
        {"A", 1.0}, {"mA", 1e-3}, {"uA", 1e-6}, {"nA", 1e-9}, {"pA", 1e-12},
        {"S", 1.0}, {"mS", 1e-3}, {"uS", 1e-6}, {"nS", 1e-9}, {"pS", 1e-12},
        {"F", 1.0}, {"mF", 1e-3}, {"uF", 1e-6}, {"nF", 1e-9}, {"pF", 1e-12},
        {"s", 1.0}, {"ms", 1e-3}, {"us", 1e-6}, {"min", 60.0}, {"hour", 3600.0},
        {"Hz", 1.0}, {"per_s", 1.0}, {"per_ms", 1e3},
        {"per_min", 0.01666666667}, {"per_hour", 0.00027777777778},
        {"m", 1.0}, {"cm", 1e-2}, {"um", 1e-6},
        {"m2", 1.0}, {"cm2", 1e-4}, {"um2", 1e-12},
        {"m3", 1.0}, {"cm3", 1e-6}, {"um3", 1e-18}, {"litre", 1e-3},
        {"M", 1.0}, {"mM", 1e-3}, {"mol_per_m3", 1.0}, {"mol_per_cm3", 1e6},
        {"ohm", 1.0}, {"kohm", 1e3}, {"Mohm", 1e6},
        {"ohm_m", 1.0}, {"ohm_cm", 1e-2}, {"kohm_cm", 10.0},
        {"degC", 1.0}, {"K", 1.0},
        {"C", 1.0}, {"e", 1.602176634e-19}, {"mol", 1.0},
        {"C_per_mol", 1.0}, {"pC_per_umol", 1e-6}, {"nA_ms_per_amol", 1e6},
        {"J_per_K_per_mol", 1.0}, {"fJ_per_K_per_umol", 1e-9},
        {"A_per_m2", 1.0}, {"mA_per_cm2", 10.0}, {"uA_per_cm2", 1e-2},
        {"S_per_m2", 1.0}, {"S_per_cm2", 1e4}, {"mS_per_cm2", 10.0}, {"uS_per_cm2", 1e-2},
        {"F_per_m2", 1.0}, {"uF_per_cm2", 1e-2},
        {"S_per_V", 1.0}, {"nS_per_mV", 1e-6},
        {"m_per_s", 1.0}, {"cm_per_s", 1e-2}, {"cm_per_ms", 10.0}, {"um_per_ms", 1e-3},
        {"mol_per_m_per_A_per_s", 1.0}, {"mol_per_cm_per_uA_per_ms", 1e11},
        {"umol_per_cm_per_nA_per_ms", 1e8},
    };

    auto entry = scales.find(suffix);
    if (entry == scales.end()) {
        throw invalid_argument("unit_suffix_scale: unknown unit suffix '" + suffix +
                               "'. Scaling an unrecognised suffix by 1.0 would silently "
                               "reinterpret the value rather than report it");
    }

    return entry->second;
}

} // namespace spikecorec::units
