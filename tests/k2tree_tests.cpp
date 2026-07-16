#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <vector>
#include <gtest/gtest.h>

#include "spikecorec/core/types.h"
#include "spikecorec/core/k2tree.h"

using namespace std;
using namespace spikecorec;

namespace {

// Small fixed 8-node directed graph with two self-loops (nodes 1 and 4).
vector<vector<s32>> k2_reference_adjacency() {
    return {
        {1, 4, 7}, // 0
        {1, 2},    // 1 — self-loop
        {0, 3, 5}, // 2
        {},        // 3 — no out-edges
        {4},       // 4 — self-loop
        {6, 7},    // 5
        {0},       // 6
        {3, 6},    // 7
    };
}

} // namespace

TEST(K2Tree, from_adjacency_list_invalid_branching_factor) {
    auto adjacency_list = k2_reference_adjacency();
    const s32 node_count = 0;
    auto result = K2Tree::from_adjacency_list(adjacency_list, node_count, -1);
    EXPECT_FALSE(result.has_value());
}

TEST(K2Tree, adjacent_and_neighbors) {
    auto adjacency = k2_reference_adjacency();
    const s32 node_count = 8;
    K2Tree tree = *K2Tree::from_adjacency_list(adjacency, node_count);

    for (s32 source = 0; source < node_count; ++source) {
        unordered_set<s32> row(adjacency[(usize)source].begin(), adjacency[(usize)source].end());
        for (s32 target = 0; target < node_count; ++target)
            EXPECT_EQ(tree.adjacent(source, target), (row.count(target) ? 1 : 0));
    }
    EXPECT_EQ(tree.adjacent(1, 1), 1);
    EXPECT_EQ(tree.adjacent(4, 4), 1);

    vector<s32> buffer(node_count);
    for (s32 source = 0; source < node_count; ++source) {
        s64 count = tree.get_neighbors(source, buffer.data(), node_count);
        vector<s32> neighbors(buffer.begin(), buffer.begin() + count);
        vector<s32> expected = adjacency[(usize)source];
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(neighbors, expected);
    }
}

TEST(K2Tree, adjacent_batch) {
    auto adjacency = k2_reference_adjacency();
    const s32 node_count = 8;
    K2Tree tree = *K2Tree::from_adjacency_list(adjacency, node_count);

    vector<s32> source_nodes, target_nodes;
    for (s32 source = 0; source < node_count; ++source)
        for (s32 target = 0; target < node_count; ++target) {
            source_nodes.push_back(source);
            target_nodes.push_back(target);
        }
    vector<uint8_t> output(source_nodes.size());
    tree.adjacent_batch(source_nodes.data(), target_nodes.data(), output.data(), (s32)source_nodes.size());

    for (usize query = 0; query < source_nodes.size(); ++query)
        EXPECT_EQ(output[query], (uint8_t)tree.adjacent(source_nodes[query], target_nodes[query]));
}

TEST(K2Tree, single_node_and_bounds) {
    // A single-node graph must still be able to represent its one possible edge --
    // the self-loop (0,0) -- which needs a one-level tree (tree_height=1), not zero
    // levels; compute_tree_parameters previously special-cased node_count<=1 to
    // tree_height=0, producing an entirely empty bit array that could represent no
    // edge at all, including this self-loop. Fixed alongside ticket SC-52/D2.
    vector<vector<s32>> single = {{0}};
    K2Tree one = *K2Tree::from_adjacency_list(single, 1);
    EXPECT_EQ(one.tree_height, 1);
    vector<s32> buffer(4);
    EXPECT_EQ(one.adjacent(0, 0), 1);
    EXPECT_EQ(one.get_neighbors(0, buffer.data(), 4), 1);
    EXPECT_EQ(buffer[0], 0);

    // An isolated single node (no self-loop declared) correctly reports no edge.
    vector<vector<s32>> single_isolated = {{}};
    K2Tree isolated = *K2Tree::from_adjacency_list(single_isolated, 1);
    EXPECT_EQ(isolated.tree_height, 1);
    EXPECT_EQ(isolated.adjacent(0, 0), 0);
    EXPECT_EQ(isolated.get_neighbors(0, buffer.data(), 4), 0);

    K2Tree tree = *K2Tree::from_adjacency_list(k2_reference_adjacency(), 8);
    EXPECT_EQ(tree.adjacent(-1, 0), 0);
    EXPECT_EQ(tree.adjacent(0, 8), 0);
    EXPECT_EQ(tree.adjacent(8, 0), 0);
    EXPECT_EQ(tree.get_neighbors(-1, buffer.data(), 4), 0);
    EXPECT_EQ(tree.get_neighbors(8, buffer.data(), 4), 0);
    EXPECT_EQ(tree.get_neighbors(0, buffer.data(), 0), 0);
}

TEST(K2Tree, save_load) {
    auto adjacency = k2_reference_adjacency();
    const s32 node_count = 8;
    K2Tree tree = *K2Tree::from_adjacency_list(adjacency, node_count);

    const char *path = "/tmp/spikecorec_test_k2tree.bin";
    tree.save(path);
    K2Tree loaded = K2Tree::load(path);

    EXPECT_EQ(loaded.node_count, tree.node_count);
    EXPECT_EQ(loaded.tree_height, tree.tree_height);
    for (s32 source = 0; source < node_count; ++source)
        for (s32 target = 0; target < node_count; ++target)
            EXPECT_EQ(loaded.adjacent(source, target), tree.adjacent(source, target));
}

TEST(K2Tree, from_edges) {
    auto adjacency = k2_reference_adjacency();
    const s32 node_count = 8;
    vector<s32> source_nodes, target_nodes;
    for (s32 source = 0; source < node_count; ++source)
        for (s32 target : adjacency[(usize)source]) {
            source_nodes.push_back(source);
            target_nodes.push_back(target);
        }

    K2Tree from_edge_list = *K2Tree::from_edges(source_nodes.data(), target_nodes.data(),
                                                (s32)source_nodes.size(), node_count);
    K2Tree from_adj = *K2Tree::from_adjacency_list(adjacency, node_count);
    for (s32 source = 0; source < node_count; ++source)
        for (s32 target = 0; target < node_count; ++target)
            EXPECT_EQ(from_edge_list.adjacent(source, target), from_adj.adjacent(source, target));
}

TEST(K2Tree, from_edges_invalid_branching_factor) {
    auto adjacency_list = k2_reference_adjacency();
    const s32 node_count = 8;
    vector<s32> source_nodes, target_nodes;
    for (usize source = 0; source < (usize)node_count; ++source)
        for (s32 target : adjacency_list[source]) {
            source_nodes.push_back((s32)source);
            target_nodes.push_back(target);
        }

    auto result = K2Tree::from_edges(source_nodes.data(), target_nodes.data(),
                                     (s32)source_nodes.size(), node_count, 10);
    EXPECT_FALSE(result.has_value());

    result = K2Tree::from_edges(source_nodes.data(), target_nodes.data(),
                                (s32)source_nodes.size(), node_count, -1);
    EXPECT_FALSE(result.has_value());
}
