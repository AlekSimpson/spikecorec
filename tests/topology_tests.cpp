#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <unordered_set>
#include <vector>
#include <gtest/gtest.h>

#include "spikecorec/core/types.h"
#include "spikecorec/core/topologies.h"

using namespace std;
using namespace spikecorec;


void validate_topology(vector<vector<s32>> &network, s64 expected_network_width) {
    EXPECT_EQ((s64)network.size(), 16);

    for (s64 index = 0; index < (s64)network.size(); ++index) {
        EXPECT_EQ((s64)network[(usize)index].size(), expected_network_width);

        unordered_set<s32> seen;
        for (s32 child : network[(usize)index]) {
            EXPECT_TRUE(child >= 0 && child < 16);
            EXPECT_NE(child, (s32)index);
            EXPECT_TRUE(seen.insert(child).second);
        }
    }
}

TEST(Topologies, topology_initialization) {
    auto square = square_torus(4);
    auto small_world = small_world_torus(4, /*random_fanout=*/4, /*seed=*/42);
    auto random = random_fixed_outdegree(4, /*fanout=*/6, /*seed=*/42);

    validate_topology(square, 4);
    validate_topology(small_world, 8);
    validate_topology(random, 6);
}

TEST(Topologies, square_torus_edge_cases) {
    // k == 1: all four torus-wraparound directions point back to the single cell
    // itself; since self-loops are not supported, square_torus omits them rather
    // than reporting node 0 as its own neighbor, so node 0 has no neighbors at all.
    auto one = square_torus(1);
    EXPECT_EQ((s64)one.size(), 1);
    EXPECT_EQ((s64)one[0].size(), 0);

    // k == 3: exact 4-neighbor torus wrapping for node 0 → {right=1,left=2,down=3,up=6}.
    auto three = square_torus(3);
    EXPECT_EQ((s64)three.size(), 9);
    EXPECT_EQ((unordered_set<s32>(three[0].begin(), three[0].end())),
              (unordered_set<s32>{1, 2, 3, 6}));

    // k < 1 is rejected.
    EXPECT_THROW(square_torus(0), std::invalid_argument);
    EXPECT_THROW(square_torus(-2), std::invalid_argument);
}

TEST(Topologies, small_world_determinism) {
    // Same seed → byte-identical structure.
    auto first = small_world_torus(4, /*random_fanout=*/3, /*seed=*/123);
    auto second = small_world_torus(4, /*random_fanout=*/3, /*seed=*/123);
    EXPECT_EQ(first, second);

    // Out-degree = 4 (torus) + 3 (shortcuts) = 7, all distinct, no self-loops.
    for (s64 index = 0; index < (s64)first.size(); ++index) {
        EXPECT_EQ((s64)first[(usize)index].size(), 7);
        unordered_set<s32> seen(first[(usize)index].begin(), first[(usize)index].end());
        EXPECT_EQ((s64)seen.size(), 7);
        EXPECT_EQ(seen.count((s32)index), 0u);
    }

    // random_fanout 0 → just the torus; negative clamps to 0.
    for (auto &row : small_world_torus(4, /*random_fanout=*/0, /*seed=*/1))
        EXPECT_EQ((s64)row.size(), 4);
    for (auto &row : small_world_torus(4, /*random_fanout=*/-5, /*seed=*/1))
        EXPECT_EQ((s64)row.size(), 4);
}

TEST(Topologies, random_fixed_outdegree_edge_cases) {
    // fanout 0 → every row empty.
    auto zero_fanout = random_fixed_outdegree(3, /*fanout=*/0, /*seed=*/7);
    EXPECT_EQ((s64)zero_fanout.size(), 9);
    for (auto &row : zero_fanout) EXPECT_TRUE(row.empty());

    // Determinism.
    EXPECT_EQ(random_fixed_outdegree(4, 5, 99), random_fixed_outdegree(4, 5, 99));

    // fanout clamped to node_count - 1; requesting more → exactly node_count - 1 distinct, no self-loops.
    auto full = random_fixed_outdegree(3, /*fanout=*/100, /*seed=*/3); // node_count=9 → clamp to 8
    for (s64 index = 0; index < (s64)full.size(); ++index) {
        EXPECT_EQ((s64)full[(usize)index].size(), 8);
        unordered_set<s32> seen(full[(usize)index].begin(), full[(usize)index].end());
        EXPECT_EQ((s64)seen.size(), 8);
        EXPECT_EQ(seen.count((s32)index), 0u);
    }

    // k < 1 throws.
    EXPECT_THROW(random_fixed_outdegree(0, 4, 1), std::invalid_argument);
}
