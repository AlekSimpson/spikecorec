#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <gtest/gtest.h>

#include "types.h"
#include "backend.h"
#include "k2tree.h"
#include "support/test_support.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::test_support;

namespace {

// A fixed 8-node directed graph, without self-loops.
Vector<Vector<s32>> k2_reference_adjacency() {
    return {
        {1, 4, 7}, // 0
        {2},       // 1
        {0, 3, 5}, // 2
        {},        // 3 — no out-edges
        {},        // 4
        {6, 7},    // 5
        {0},       // 6
        {3, 6},    // 7
    };
}

} // namespace

TEST(K2_Tree, from_adjacency_list_invalid_branching_factor) {
    auto adjacency_list = k2_reference_adjacency();
    const s32 node_count = 0;
    auto result = K2_Tree::from_adjacency_list(shared_backend(), adjacency_list, node_count, -1);
    EXPECT_FALSE(result.has_value());
}

TEST(K2_Tree, adjacent_and_neighbors) {
    auto adjacency = k2_reference_adjacency();
    const s32 node_count = 8;
    K2_Tree tree = *K2_Tree::from_adjacency_list(shared_backend(), adjacency, node_count);

    for (s32 source = 0; source < node_count; ++source) {
        Set<s32> row(adjacency[static_cast<usize>(source)].begin(), adjacency[static_cast<usize>(source)].end());
        for (s32 target = 0; target < node_count; ++target)
            EXPECT_EQ(tree.adjacent(source, target), (row.count(target) ? 1 : 0));
    }

    Vector<u32> buffer(node_count);
    for (s32 source = 0; source < node_count; ++source) {
        s64 count = tree.get_neighbors(source, buffer.data(), node_count);
        Vector<u32> neighbors(buffer.begin(), buffer.begin() + count);
        Vector<u32> expected(adjacency[static_cast<usize>(source)].begin(), adjacency[static_cast<usize>(source)].end());
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(neighbors, expected);
    }
}

TEST(K2_Tree, adjacent_and_predecessors) {
    auto adjacency = k2_reference_adjacency();
    const s32 node_count = 8;
    K2_Tree tree = *K2_Tree::from_adjacency_list(shared_backend(), adjacency, node_count);

    // Expected predecessors are the transpose of the fixture, not derived from tree.adjacent(), so
    // a wrong adjacent() cannot pass by agreeing with itself.
    Vector<Vector<s32>> predecessors(static_cast<usize>(node_count));
    for (s32 source = 0; source < node_count; ++source)
        for (s32 target : adjacency[static_cast<usize>(source)])
            predecessors[static_cast<usize>(target)].push_back(source);

    Vector<u32> buffer(node_count);
    for (s32 target = 0; target < node_count; ++target) {
        s64 count = tree.get_predecessors(target, buffer.data(), node_count);
        Vector<u32> found(buffer.begin(), buffer.begin() + count);
        std::sort(found.begin(), found.end());
        Vector<u32> expected(predecessors[static_cast<usize>(target)].begin(), predecessors[static_cast<usize>(target)].end());
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(found, expected) << "target=" << target;
    }
}

TEST(K2_Tree, predecessors_bounds_and_degenerate) {
    // Out-of-range or degenerate queries write nothing and return 0, like get_neighbors.
    K2_Tree tree = *K2_Tree::from_adjacency_list(shared_backend(), k2_reference_adjacency(), 8);
    Vector<u32> buffer(8);
    EXPECT_EQ(tree.get_predecessors(-1, buffer.data(), 8), 0);
    EXPECT_EQ(tree.get_predecessors(8, buffer.data(), 8), 0);
    EXPECT_EQ(tree.get_predecessors(0, buffer.data(), 0), 0);

    Vector<Vector<s32>> single_isolated = {{}};
    K2_Tree isolated = *K2_Tree::from_adjacency_list(shared_backend(), single_isolated, 1);
    EXPECT_EQ(isolated.get_predecessors(0, buffer.data(), 8), 0);

    // Node 3 has predecessors 2 and 7; a limit of 1 must write exactly one real predecessor.
    Set<u32> node3_predecessors = {2, 7};
    s64 truncated = tree.get_predecessors(3, buffer.data(), 1);
    EXPECT_EQ(truncated, 1);
    EXPECT_TRUE(node3_predecessors.count(buffer[0])) << "wrote a non-predecessor: " << buffer[0];
}

TEST(K2_Tree, single_node_and_bounds) {
    // A single node's only possible edge is the self-loop (0, 0); the tree takes one level for it.
    Vector<u32> buffer(4);
    Vector<Vector<s32>> single_with_self_loop = {{0}};
    K2_Tree looped = *K2_Tree::from_adjacency_list(shared_backend(), single_with_self_loop, 1);
    EXPECT_EQ(looped.tree_height, 1);
    EXPECT_EQ(looped.adjacent(0, 0), 1);
    ASSERT_EQ(looped.get_neighbors(0, buffer.data(), 4), 1);
    EXPECT_EQ(buffer[0], 0u);
    ASSERT_EQ(looped.get_predecessors(0, buffer.data(), 4), 1);
    EXPECT_EQ(buffer[0], 0u);

    // An isolated single node reports no edge.
    Vector<Vector<s32>> single_isolated = {{}};
    K2_Tree isolated = *K2_Tree::from_adjacency_list(shared_backend(), single_isolated, 1);
    EXPECT_EQ(isolated.adjacent(0, 0), 0);
    EXPECT_EQ(isolated.get_neighbors(0, buffer.data(), 4), 0);

    K2_Tree tree = *K2_Tree::from_adjacency_list(shared_backend(), k2_reference_adjacency(), 8);
    EXPECT_EQ(tree.adjacent(-1, 0), 0);
    EXPECT_EQ(tree.adjacent(0, 8), 0);
    EXPECT_EQ(tree.adjacent(8, 0), 0);
    EXPECT_EQ(tree.get_neighbors(-1, buffer.data(), 4), 0);
    EXPECT_EQ(tree.get_neighbors(8, buffer.data(), 4), 0);
    EXPECT_EQ(tree.get_neighbors(0, buffer.data(), 0), 0);
}

TEST(K2_Tree, self_loops_are_edges_like_any_other) {
    // Through both the adjacency-list and edge-array entry points, a self-loop is stored, walked in
    // its row and its column, and listed in order with the other edges.
    Vector<Vector<s32>> adjacency_with_self_loop = {
        {1, 2},
        {1, 2}, // node 1: self-loop + normal edge
        {0}
    };
    Vector<u32> source_nodes = {0, 0, 1, 1, 2};
    Vector<u32> target_nodes = {1, 2, 1, 2, 0};
    Vector<K2_Tree> trees;
    trees.push_back(*K2_Tree::from_adjacency_list(shared_backend(), adjacency_with_self_loop, 3));
    trees.push_back(*K2_Tree::from_edges(shared_backend(), source_nodes.data(), target_nodes.data(),
                                         static_cast<s64>(source_nodes.size()), 3));
    for (const K2_Tree &tree : trees) {
        EXPECT_EQ(tree.adjacent(1, 1), 1);
        EXPECT_EQ(tree.adjacent(0, 0), 0);
        EXPECT_EQ(tree.adjacent(2, 2), 0);
        Vector<u32> buffer(4);
        ASSERT_EQ(tree.get_neighbors(1, buffer.data(), 4), 2);
        EXPECT_EQ(buffer[0], 1u);
        EXPECT_EQ(buffer[1], 2u);
        ASSERT_EQ(tree.get_predecessors(1, buffer.data(), 4), 2);
        EXPECT_EQ(buffer[0], 0u);
        EXPECT_EQ(buffer[1], 1u);
    }
}

TEST(K2_Tree, save_load) {
    auto adjacency = k2_reference_adjacency();
    const s32 node_count = 8;
    K2_Tree tree = *K2_Tree::from_adjacency_list(shared_backend(), adjacency, node_count);

    const Temporary_Directory directory;
    const String path = directory.path_of("save_load.bin");
    tree.save(path.c_str());
    K2_Tree loaded = K2_Tree::load(shared_backend(), path.c_str());

    EXPECT_EQ(loaded.node_count, tree.node_count);
    EXPECT_EQ(loaded.tree_height, tree.tree_height);
    for (s32 source = 0; source < node_count; ++source)
        for (s32 target = 0; target < node_count; ++target)
            EXPECT_EQ(loaded.adjacent(source, target), tree.adjacent(source, target));
}

TEST(K2_Tree, from_edges) {
    auto adjacency = k2_reference_adjacency();
    const s32 node_count = 8;
    Vector<u32> source_nodes, target_nodes;
    for (s32 source = 0; source < node_count; ++source)
        for (s32 target : adjacency[static_cast<usize>(source)]) {
            source_nodes.push_back(source);
            target_nodes.push_back(target);
        }

    K2_Tree from_edge_list = *K2_Tree::from_edges(shared_backend(), source_nodes.data(), target_nodes.data(),
                                                static_cast<s64>(source_nodes.size()), node_count);
    K2_Tree from_adjacency = *K2_Tree::from_adjacency_list(shared_backend(), adjacency, node_count);
    for (s32 source = 0; source < node_count; ++source)
        for (s32 target = 0; target < node_count; ++target)
            EXPECT_EQ(from_edge_list.adjacent(source, target), from_adjacency.adjacent(source, target));
}

TEST(K2_Tree, from_edges_invalid_branching_factor) {
    auto adjacency_list = k2_reference_adjacency();
    const s32 node_count = 8;
    Vector<u32> source_nodes, target_nodes;
    for (usize source = 0; source < static_cast<usize>(node_count); ++source)
        for (s32 target : adjacency_list[source]) {
            source_nodes.push_back(static_cast<u32>(source));
            target_nodes.push_back(target);
        }

    auto result = K2_Tree::from_edges(shared_backend(), source_nodes.data(), target_nodes.data(),
                                     static_cast<s64>(source_nodes.size()), node_count, 10);
    EXPECT_FALSE(result.has_value());

    result = K2_Tree::from_edges(shared_backend(), source_nodes.data(), target_nodes.data(),
                                static_cast<s64>(source_nodes.size()), node_count, -1);
    EXPECT_FALSE(result.has_value());
}


// A branching factor below two never subdivides, so the build would never finish.
TEST(K2_Tree, a_branching_factor_below_two_is_rejected) {
    const auto adjacency = k2_reference_adjacency();
    Vector<u32> source_nodes, target_nodes;
    for (s32 source = 0; source < 8; source += 1) {
        for (s32 target : adjacency[static_cast<usize>(source)]) {
            source_nodes.push_back(source);
            target_nodes.push_back(target);
        }
    }
    for (s32 branching_factor : {0, 1}) {
        EXPECT_FALSE(K2_Tree::from_adjacency_list(shared_backend(), adjacency, 8, branching_factor).has_value())
            << "branching factor " << branching_factor;
        EXPECT_FALSE(K2_Tree::from_edges(shared_backend(), source_nodes.data(), target_nodes.data(),
                                        static_cast<s64>(source_nodes.size()), 8, branching_factor).has_value())
            << "branching factor " << branching_factor;
    }
}

// The subblock table holds a u16 popcount, which overflows past 2048 words per superblock.
TEST(K2_Tree, a_superblock_size_outside_the_subblock_range_is_rejected) {
    const auto adjacency = k2_reference_adjacency();
    Vector<u32> source_nodes, target_nodes;
    for (s32 source = 0; source < 8; source += 1) {
        for (s32 target : adjacency[static_cast<usize>(source)]) {
            source_nodes.push_back(source);
            target_nodes.push_back(target);
        }
    }
    for (s32 superblock_size : {0, -1, K2_Tree::MAX_SUPERBLOCK_SIZE_WORDS + 1}) {
        EXPECT_FALSE(K2_Tree::from_adjacency_list(shared_backend(), adjacency, 8, 4, superblock_size).has_value())
            << "superblock size " << superblock_size;
        EXPECT_FALSE(K2_Tree::from_edges(shared_backend(), source_nodes.data(), target_nodes.data(),
                                        static_cast<s64>(source_nodes.size()), 8, 4, superblock_size).has_value())
            << "superblock size " << superblock_size;
    }
    EXPECT_TRUE(K2_Tree::from_adjacency_list(shared_backend(), adjacency, 8, 4, K2_Tree::MAX_SUPERBLOCK_SIZE_WORDS)
                    .has_value());
}

// trace() is the debug twin of adjacent() and must give the same answer for every pair, self-loops included.
TEST(K2_Tree, trace_agrees_with_adjacent) {
    const s32 node_count = 129;
    Mersenne_Twister random_engine(4242);
    Bernoulli_Distribution has_edge(0.1);
    Vector<Vector<s32>> adjacency(static_cast<usize>(node_count));
    for (s32 source = 0; source < node_count; source += 1) {
        for (s32 target = 0; target < node_count; target += 1) {
            if (has_edge(random_engine)) adjacency[static_cast<usize>(source)].push_back(target);
        }
    }
    K2_Tree tree = *K2_Tree::from_adjacency_list(shared_backend(), adjacency, node_count, 2);

    for (s32 source = 0; source < node_count; source += 1) {
        for (s32 target = 0; target < node_count; target += 1) {
            ASSERT_EQ(tree.trace(source, target), tree.adjacent(source, target)) << source << " -> " << target;
        }
    }
}

TEST(K2_Tree, random_graphs_answer_every_query_for_every_branching_factor) {
    // Node counts that are powers of no branching factor, so every tree is padded; self-loops are drawn too.
    for (s32 branching_factor : {2, 3, 4, 5}) {
        for (s32 node_count : {7, 50, 129}) {
            Mersenne_Twister random_engine(static_cast<unsigned>(1000 * branching_factor + node_count));
            Bernoulli_Distribution has_edge(0.1);
            Vector<Vector<s32>> adjacency(static_cast<usize>(node_count));
            Vector<Vector<s32>> predecessors(static_cast<usize>(node_count));
            for (s32 source = 0; source < node_count; source += 1) {
                for (s32 target = 0; target < node_count; target += 1) {
                    if (!has_edge(random_engine)) continue;
                    adjacency[static_cast<usize>(source)].push_back(target);
                    predecessors[static_cast<usize>(target)].push_back(source);
                }
            }
            K2_Tree tree = *K2_Tree::from_adjacency_list(shared_backend(), adjacency, node_count, branching_factor);
            const String label = "branching factor " + to_string(branching_factor) + ", " + to_string(node_count) + " nodes";

            for (s32 source = 0; source < node_count; source += 1) {
                const Set<s32> row(adjacency[static_cast<usize>(source)].begin(), adjacency[static_cast<usize>(source)].end());
                for (s32 target = 0; target < node_count; target += 1) {
                    ASSERT_EQ(tree.adjacent(source, target), row.count(target) ? 1 : 0)
                        << label << ", edge " << source << " -> " << target;
                }
            }

            Vector<u32> buffer(static_cast<usize>(node_count));
            for (s32 node = 0; node < node_count; node += 1) {
                const s64 neighbor_count = tree.get_neighbors(node, buffer.data(), node_count);
                Vector<u32> neighbors(buffer.begin(), buffer.begin() + neighbor_count);
                sort(neighbors.begin(), neighbors.end());
                const Vector<u32> expected_neighbors(adjacency[static_cast<usize>(node)].begin(),
                                                     adjacency[static_cast<usize>(node)].end());
                EXPECT_EQ(neighbors, expected_neighbors) << label << ", neighbors of " << node;

                const s64 predecessor_count = tree.get_predecessors(node, buffer.data(), node_count);
                Vector<u32> found(buffer.begin(), buffer.begin() + predecessor_count);
                sort(found.begin(), found.end());
                const Vector<u32> expected_predecessors(predecessors[static_cast<usize>(node)].begin(),
                                                        predecessors[static_cast<usize>(node)].end());
                EXPECT_EQ(found, expected_predecessors) << label << ", predecessors of " << node;
            }
        }
    }
}
