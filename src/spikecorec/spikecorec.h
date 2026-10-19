#pragma once

// The engine's public headers, for a program that wants them in one include.

#include <cstdlib>

#include "core/types.h"
#include "core/log.h"
#include "core/allocators/linear_allocator.h"
#include "core/allocators/pool_allocator.h"

using namespace spikecorec::log;

namespace spikecorec {

namespace Program_Context {

inline constexpr u64 KB = 1000;
inline constexpr u64 MB = 1000000;
inline constexpr u64 GB = 1000000000;

inline constexpr u64 SCRATCH_SIZE = 10 * MB;
inline constexpr u64 POOL_COUNT = 10;
inline constexpr u64 POOL_DEPTH_BYTESIZE = 10 * MB;

inline Linear_Allocator scratch = Linear_Allocator(SCRATCH_SIZE);
inline Pool_Allocator pool = Pool_Allocator(POOL_COUNT, POOL_DEPTH_BYTESIZE);

inline void panic(const String &message) {
    scratch.clear();
    pool.clear();

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
