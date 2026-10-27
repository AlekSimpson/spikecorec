#pragma once

// The engine's public headers, for a program that wants them in one include.

#include <cstdlib>

#include "core/types.h"
#include "core/log.h"
#include "core/allocators/linear_allocator.h"

using namespace spikecorec::log;

namespace spikecorec {

namespace Program_Context {

constexpr s64 NEVER_SPIKED_TICK = -1;
static constexpr u64 PARTITION_ALIGNMENT = 256;

inline constexpr u64 KB = 1000;
inline constexpr u64 MB = 1000000;
inline constexpr u64 GB = 1000000000;

inline constexpr u64 TRANSIENT_BUFFER_SIZE = 10 * MB;
inline constexpr u64 PERSISTENT_BUFFER_SIZE = 50 * MB;

inline Linear_Allocator transient = Linear_Allocator(TRANSIENT_BUFFER_SIZE);
inline Linear_Allocator persistent = Linear_Allocator(PERSISTENT_BUFFER_SIZE);

inline void panic(const String &message) {
    transient.clear();
    persistent.clear();

    logger().critical(message);

    exit(EXIT_FAILURE);
}


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

}

} // namespace spikecorec
