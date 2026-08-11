#pragma once

#include <any>
#include <cstdint>
#include <cstddef>
#include <optional>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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

    template <typename KeyType, typename ValueType>
    using UnorderedMap = std::unordered_map<KeyType, ValueType>;

    template <typename SomeType>
    using Vector = std::vector<SomeType>;

    template <typename SomeType>
    using Set = std::unordered_set<SomeType>;

    template <typename SomeType>
    using Optional = std::optional<SomeType>;

    template <typename SomeType>
    using Stack = std::stack<SomeType>;

    template <typename FirstType, typename SecondType>
    using Pair = std::pair<FirstType, SecondType>;

    struct alignas(16) float4 {
        f32 x, y, z, w;
    };


} // namespace spikecorec
