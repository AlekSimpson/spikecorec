//
// Created by Alek Simpson on 6/7/26.
//

#include <random>
#include <algorithm>
#include <unordered_set>

#include "spikecorec/core/topologies.h"
#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::log;

namespace {
    mt19937_64 make_rng(s64 seed) {
        if (seed >= 0) return mt19937_64(static_cast<u64>(seed));
        random_device rd;
        return mt19937_64((static_cast<u64>(rd()) << 32) | static_cast<u64>(rd()));
    }
}

namespace spikecorec {

    vector<vector<s32>> square_torus(s64 side_length) {
        logger().debug("square_torus: side_length={}", side_length);
        if (side_length < 1) {
            throw_invalid_argument(logger(), fmt::format("square_torus: side_length must be >= 1 (got {})", side_length));
        }

        s64 cell_count = side_length * side_length;
        vector<vector<s32>> result(static_cast<usize>(cell_count));
        for (s64 i = 0; i < cell_count; ++i) {
            s64 row = i / side_length;
            s64 column = i % side_length;
            s64 right = row * side_length + (column + 1) % side_length;
            s64 left  = row * side_length + ((column - 1) + side_length) % side_length;
            s64 down  = ((row + 1) % side_length) * side_length + column;
            s64 up    = (((row - 1) + side_length) % side_length) * side_length + column;

            vector<s32> neighbors;
            for (s64 candidate : {right, left, down, up}) {
                if (candidate != i) {
                    neighbors.push_back(static_cast<s32>(candidate));
                }
            }
            result[static_cast<usize>(i)] = std::move(neighbors);
        }
        logger().debug("square_torus: cell_count={}", cell_count);
        return result;
    }

    vector<vector<s32>> small_world_torus(s64 side_length, s32 random_fanout, s64 seed) {
        logger().debug("small_world_torus: side_length={} random_fanout={} seed={}", side_length, random_fanout, seed);
        if (side_length < 1) {
            throw_invalid_argument(logger(), fmt::format("small_world_torus: side_length must be >= 1 (got {})", side_length));
        }
        random_fanout = std::max(0, random_fanout);

        vector<vector<s32>> base = square_torus(side_length);
        s64 cell_count = side_length * side_length;

        mt19937_64 rng = make_rng(seed);
        uniform_int_distribution<s64> dist(0, cell_count - 1);

        vector<vector<s32>> out(static_cast<usize>(cell_count));
        for (s64 i = 0; i < cell_count; ++i) {
            vector<s32> children = base[static_cast<usize>(i)];
            unordered_set<s32> used(children.begin(), children.end());
            used.insert(static_cast<s32>(i));

            s64 target_count = std::min<s64>(
                4 + random_fanout,
                static_cast<s64>(children.size()) + std::max<s64>(0, cell_count - static_cast<s64>(used.size()))
            );
            while (static_cast<s64>(children.size()) < target_count) {
                s32 candidate = static_cast<s32>(dist(rng));
                if (used.count(candidate)) continue;
                used.insert(candidate);
                children.push_back(candidate);
            }
            out[static_cast<usize>(i)] = std::move(children);
        }
        logger().debug("small_world_torus: cell_count={}", cell_count);
        return out;
    }

    vector<vector<s32>> random_fixed_outdegree(s64 side_length, s32 fanout, s64 seed) {
        logger().debug("random_fixed_outdegree: side_length={} fanout={} seed={}", side_length, fanout, seed);
        if (side_length < 1) {
            throw_invalid_argument(logger(), fmt::format("random_fixed_outdegree: side_length must be >= 1 (got {})", side_length));
        }
        s64 cell_count = side_length * side_length;
        fanout = std::max(0, std::min<s32>(fanout, static_cast<s32>(cell_count - 1)));

        mt19937_64 rng = make_rng(seed);
        uniform_int_distribution<s64> dist(0, cell_count - 1);

        vector<vector<s32>> out(static_cast<usize>(cell_count));
        for (s64 i = 0; i < cell_count; ++i) {
            if (fanout == 0) {
                out[static_cast<usize>(i)] = {};
                continue;
            }

            unordered_set<s32> chosen;
            vector<s32> picked;
            picked.reserve(static_cast<usize>(fanout));
            while (static_cast<s32>(picked.size()) < fanout) {
                s32 candidate = static_cast<s32>(dist(rng));
                if (candidate == static_cast<s32>(i) || chosen.count(candidate)) continue;
                chosen.insert(candidate);
                picked.push_back(candidate);
            }
            out[static_cast<usize>(i)] = std::move(picked);
        }
        logger().debug("random_fixed_outdegree: cell_count={} effective_fanout={}", cell_count, fanout);
        return out;
    }

} // namespace spikecorec
