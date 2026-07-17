#pragma once

#include <any>
#include <cstdint>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace spikecorec {

    using String = std::string;
    using Any = std::any;

    using f32 = float;
    using f64 = double;
    using s32 = int32_t;
    using s64 = int64_t;
    using u8 = uint8_t;
    using u16 = uint16_t;
    using u32 = uint32_t;
    using u64 = uint64_t;
    using usize = size_t;

    template <typename K, typename V>
    using UnorderedMap = std::unordered_map<K, V>;

    template <typename T>
    using Vector = std::vector<T>;

    struct alignas(16) float4 {
        f32 x, y, z, w;
    };


} // namespace spikecorec
