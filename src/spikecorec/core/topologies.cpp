//
// Created by Alek Simpson on 6/7/26.
//

#include <random>
#include <algorithm>
#include <unordered_set>

#include "topologies.h"
#include "log.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::log;

namespace {
    Mersenne_Twister_64 make_rng(s64 seed) {
        if (seed >= 0) return Mersenne_Twister_64(static_cast<u64>(seed));
        Random_Device rd;
        return Mersenne_Twister_64((static_cast<u64>(rd()) << 32) | static_cast<u64>(rd()));
    }
}

namespace spikecorec {

    Vector<Vector<s32>> square_torus(s64 side_length) {
        logger().debug("square_torus: side_length={}", side_length);
        if (side_length < 1) {
            throw_invalid_argument(logger(), log::format("square_torus: side_length must be >= 1 (got {})", side_length));
        }

        s64 cell_count = side_length * side_length;
        Vector<Vector<s32>> result(static_cast<usize>(cell_count));
        for (s64 i = 0; i < cell_count; ++i) {
            s64 row = i / side_length;
            s64 column = i % side_length;
            s64 right = row * side_length + (column + 1) % side_length;
            s64 left  = row * side_length + ((column - 1) + side_length) % side_length;
            s64 down  = ((row + 1) % side_length) * side_length + column;
            s64 up    = (((row - 1) + side_length) % side_length) * side_length + column;

            // Below a side of 3, opposite directions wrap to the same node, so each is kept once.
            Vector<s32> neighbors;
            for (s64 candidate : {right, left, down, up}) {
                const s32 neighbor = static_cast<s32>(candidate);
                if (candidate == i || std::find(neighbors.begin(), neighbors.end(), neighbor) != neighbors.end()) continue;
                neighbors.push_back(neighbor);
            }
            result[static_cast<usize>(i)] = std::move(neighbors);
        }
        logger().debug("square_torus: cell_count={}", cell_count);
        return result;
    }

    Vector<Vector<s32>> small_world_torus(s64 side_length, s32 random_fanout, s64 seed) {
        logger().debug("small_world_torus: side_length={} random_fanout={} seed={}", side_length, random_fanout, seed);
        if (side_length < 1) {
            throw_invalid_argument(logger(), log::format("small_world_torus: side_length must be >= 1 (got {})", side_length));
        }
        random_fanout = std::max(0, random_fanout);

        Vector<Vector<s32>> base = square_torus(side_length);
        s64 cell_count = side_length * side_length;

        Mersenne_Twister_64 rng = make_rng(seed);
        Uniform_Integer_Distribution<s64> dist(0, cell_count - 1);

        Vector<Vector<s32>> out(static_cast<usize>(cell_count));
        for (s64 i = 0; i < cell_count; ++i) {
            Vector<s32> children = base[static_cast<usize>(i)];
            Set<s32> used(children.begin(), children.end());
            used.insert(static_cast<s32>(i));

            s64 target_count = std::min<s64>(
                static_cast<s64>(children.size()) + random_fanout,
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

    Vector<Vector<s32>> random_fixed_outdegree(s64 side_length, s32 fanout, s64 seed) {
        logger().debug("random_fixed_outdegree: side_length={} fanout={} seed={}", side_length, fanout, seed);
        if (side_length < 1) {
            throw_invalid_argument(logger(), log::format("random_fixed_outdegree: side_length must be >= 1 (got {})", side_length));
        }
        s64 cell_count = side_length * side_length;
        fanout = std::max(0, std::min<s32>(fanout, static_cast<s32>(cell_count - 1)));

        Mersenne_Twister_64 rng = make_rng(seed);
        Uniform_Integer_Distribution<s64> dist(0, cell_count - 1);

        Vector<Vector<s32>> out(static_cast<usize>(cell_count));
        for (s64 i = 0; i < cell_count; ++i) {
            if (fanout == 0) {
                out[static_cast<usize>(i)] = {};
                continue;
            }

            Set<s32> chosen;
            Vector<s32> picked;
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
