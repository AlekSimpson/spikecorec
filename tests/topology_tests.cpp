#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <unordered_set>
#include <vector>
#include <gtest/gtest.h>

#include "types.h"
#include "topologies.h"

using namespace std;
using namespace spikecorec;


void validate_topology(Vector<Vector<s32>> &network, s64 expected_network_width) {
    EXPECT_EQ(static_cast<s64>(network.size()), 16);

    for (s64 index = 0; index < static_cast<s64>(network.size()); ++index) {
        EXPECT_EQ(static_cast<s64>(network[static_cast<usize>(index)].size()), expected_network_width);

        Set<s32> seen;
        for (s32 child : network[static_cast<usize>(index)]) {
            EXPECT_TRUE(child >= 0 && child < 16);
            EXPECT_NE(child, static_cast<s32>(index));
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
    EXPECT_EQ(static_cast<s64>(one.size()), 1);
    EXPECT_EQ(static_cast<s64>(one[0].size()), 0);

    // k == 3: exact 4-neighbor torus wrapping for node 0 → {right=1,left=2,down=3,up=6}.
    auto three = square_torus(3);
    EXPECT_EQ(static_cast<s64>(three.size()), 9);
    EXPECT_EQ((Set<s32>(three[0].begin(), three[0].end())),
              (Set<s32>{1, 2, 3, 6}));

    // k < 1 is rejected.
    EXPECT_THROW(square_torus(0), Invalid_Argument);
    EXPECT_THROW(square_torus(-2), Invalid_Argument);
}

TEST(Topologies, small_world_determinism) {
    // Same seed → byte-identical structure.
    auto first = small_world_torus(4, /*random_fanout=*/3, /*seed=*/123);
    auto second = small_world_torus(4, /*random_fanout=*/3, /*seed=*/123);
    EXPECT_EQ(first, second);

    // Out-degree = 4 (torus) + 3 (shortcuts) = 7, all distinct, no self-loops.
    for (s64 index = 0; index < static_cast<s64>(first.size()); ++index) {
        EXPECT_EQ(static_cast<s64>(first[static_cast<usize>(index)].size()), 7);
        Set<s32> seen(first[static_cast<usize>(index)].begin(), first[static_cast<usize>(index)].end());
        EXPECT_EQ(static_cast<s64>(seen.size()), 7);
        EXPECT_EQ(seen.count(static_cast<s32>(index)), 0u);
    }

    // random_fanout 0 → just the torus; negative clamps to 0.
    for (auto &row : small_world_torus(4, /*random_fanout=*/0, /*seed=*/1))
        EXPECT_EQ(static_cast<s64>(row.size()), 4);
    for (auto &row : small_world_torus(4, /*random_fanout=*/-5, /*seed=*/1))
        EXPECT_EQ(static_cast<s64>(row.size()), 4);
}

TEST(Topologies, random_fixed_outdegree_edge_cases) {
    // fanout 0 → every row empty.
    auto zero_fanout = random_fixed_outdegree(3, /*fanout=*/0, /*seed=*/7);
    EXPECT_EQ(static_cast<s64>(zero_fanout.size()), 9);
    for (auto &row : zero_fanout) EXPECT_TRUE(row.empty());

    // Determinism.
    EXPECT_EQ(random_fixed_outdegree(4, 5, 99), random_fixed_outdegree(4, 5, 99));

    // fanout clamped to node_count - 1; requesting more → exactly node_count - 1 distinct, no self-loops.
    auto full = random_fixed_outdegree(3, /*fanout=*/100, /*seed=*/3); // node_count=9 → clamp to 8
    for (s64 index = 0; index < static_cast<s64>(full.size()); ++index) {
        EXPECT_EQ(static_cast<s64>(full[static_cast<usize>(index)].size()), 8);
        Set<s32> seen(full[static_cast<usize>(index)].begin(), full[static_cast<usize>(index)].end());
        EXPECT_EQ(static_cast<s64>(seen.size()), 8);
        EXPECT_EQ(seen.count(static_cast<s32>(index)), 0u);
    }

    // k < 1 throws.
    EXPECT_THROW(random_fixed_outdegree(0, 4, 1), Invalid_Argument);
}
