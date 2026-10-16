#pragma once

// The engine's public headers, for a program that wants them in one include.

#include "types.h"
#include "units.h"
#include "backend.h"
#include "k2tree.h"
#include "weight_matrix.h"
#include "recording.h"
#include "topologies.h"
#include "parser.h"
#include "engine.h"

namespace spikecorec {

// The values joined in order.
inline String combine(const Vector<String> &values) {
    String result;
    for (const String &value : values) result += value;
    return result;
}

// Each value followed by filler.
inline String combine(const String &filler, const Vector<String> &values) {
    String result;
    for (const String &value : values) result += value + filler;
    return result;
}

} // namespace spikecorec
