//
// Created by Alek Simpson on 6/7/26.
//

#include <random>
#include <stdexcept>
#include <algorithm>
#include <unordered_set>

#include "spikecorec/core/topologies.h"

using namespace std;
using namespace spikecorec;

namespace {
    mt19937_64 make_rng(s64 seed) {
        if (seed >= 0) return mt19937_64(static_cast<u64>(seed));
        random_device rd;
        return mt19937_64((static_cast<u64>(rd()) << 32) | static_cast<u64>(rd()));
    }
}

namespace spikecorec {

    vector<vector<s32>> square_torus(s64 k) {
        if (k < 1) throw std::invalid_argument("k must be >= 1.");

        s64 n = k * k;
        vector<vector<s32>> result(static_cast<usize>(n));
        for (s64 i = 0; i < n; ++i) {
            s64 row = i / k;
            s64 col = i % k;
            s64 right = row * k + (col + 1) % k;
            s64 left  = row * k + ((col - 1) + k) % k;
            s64 down  = ((row + 1) % k) * k + col;
            s64 up    = (((row - 1) + k) % k) * k + col;
            result[static_cast<usize>(i)] = {
                static_cast<s32>(right), static_cast<s32>(left),
                static_cast<s32>(down), static_cast<s32>(up)
            };
        }
        return result;
    }

    vector<vector<s32>> small_world_torus(s64 k, s32 random_fanout, s64 seed) {
        if (k < 1) throw std::invalid_argument("k must be >= 1.");
        random_fanout = std::max(0, random_fanout);

        vector<vector<s32>> base = square_torus(k);
        s64 n = k * k;

        mt19937_64 rng = make_rng(seed);
        uniform_int_distribution<s64> dist(0, n - 1);

        vector<vector<s32>> out(static_cast<usize>(n));
        for (s64 i = 0; i < n; ++i) {
            vector<s32> children = base[static_cast<usize>(i)];
            unordered_set<s32> used(children.begin(), children.end());
            used.insert(static_cast<s32>(i));

            s64 target_count = std::min<s64>(
                4 + random_fanout,
                static_cast<s64>(children.size()) + std::max<s64>(0, n - static_cast<s64>(used.size()))
            );
            while (static_cast<s64>(children.size()) < target_count) {
                s32 candidate = static_cast<s32>(dist(rng));
                if (used.count(candidate)) continue;
                used.insert(candidate);
                children.push_back(candidate);
            }
            out[static_cast<usize>(i)] = std::move(children);
        }
        return out;
    }

    vector<vector<s32>> random_fixed_outdegree(s64 k, s32 fanout, s64 seed) {
        if (k < 1) throw std::invalid_argument("k must be >= 1.");
        s64 n = k * k;
        fanout = std::max(0, std::min<s32>(fanout, static_cast<s32>(n - 1)));

        mt19937_64 rng = make_rng(seed);
        uniform_int_distribution<s64> dist(0, n - 1);

        vector<vector<s32>> out(static_cast<usize>(n));
        for (s64 i = 0; i < n; ++i) {
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
        return out;
    }

} // namespace spikecorec
