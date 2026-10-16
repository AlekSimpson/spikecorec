//
// Created by Alek Simpson on 6/7/26.
//
#pragma once

#include <vector>

#include "types.h"

using namespace std;

namespace spikecorec {

    // 4-neighbor wraparound grid (right, left, down, up), without self-loops or repeats: 4
    // distinct neighbors from a side of 3, 2 at a side of 2, none at a side of 1.
    Vector<Vector<s32>> square_torus(s64 side_length);

    // square_torus plus random_fanout distinct random long-range shortcuts per node, capped by
    // node count. seed < 0 seeds from std::random_device; seed >= 0 is deterministic.
    Vector<Vector<s32>> small_world_torus(s64 side_length, s32 random_fanout = 4, s64 seed = -1);

    // Random directed graph: each node gets fanout distinct targets, at most node count - 1,
    // with no self-loops. seed < 0 seeds from std::random_device.
    Vector<Vector<s32>> random_fixed_outdegree(s64 side_length, s32 fanout = 8, s64 seed = -1);

} // namespace spikecorec
