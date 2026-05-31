#pragma once

#include <cstdint>
#include <cstddef>

namespace spikecorec {

    using f32 = float;
    using f64 = double;
    using s32 = int32_t;
    using s64 = int64_t;
    using u16 = uint16_t;
    using u32 = uint32_t;
    using u64 = uint64_t;
    using usize = size_t;

    struct alignas(16) float4 {
        f32 x, y, z, w;
    };


} // namespace spikecorec
