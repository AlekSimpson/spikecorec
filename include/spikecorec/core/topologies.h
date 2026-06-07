//
// Created by Alek Simpson on 6/7/26.
//
#pragma once

#include <vector>

#include "spikecorec/core/types.h"

using namespace std;

namespace spikecorec {

    // 4-neighbor wraparound grid (right, left, down, up). Every node has exactly
    // 4 children. Mirrors spikecore/topologies.py's square_torus.
    vector<vector<s32>> square_torus(s64 k);

    // square_torus plus deterministic random long-range shortcuts so activity can
    // mix across the reservoir instead of only drifting locally. Every node ends
    // up with the same out-degree (4 + random_fanout, capped by reservoir size).
    // seed < 0 seeds from std::random_device (non-deterministic); seed >= 0 is
    // used directly. Mirrors spikecore/topologies.py's small_world_torus — output
    // is structurally equivalent but not bit-identical to the NumPy PCG64 version.
    vector<vector<s32>> small_world_torus(s64 k, s32 random_fanout = 4, s64 seed = -1);

    // Directed random graph with the same out-degree (no self-loops, no repeats)
    // for every node — removes the torus/local-grid edges entirely. Mirrors
    // spikecore/topologies.py's random_fixed_outdegree — output is structurally
    // equivalent but not bit-identical to the NumPy PCG64 version.
    vector<vector<s32>> random_fixed_outdegree(s64 k, s32 fanout = 8, s64 seed = -1);

} // namespace spikecorec
