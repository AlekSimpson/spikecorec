#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <utility>
#include <gtest/gtest.h>

#include "spikecorec/core/types.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/core/topologies.h"

using namespace std;
using namespace spikecorec;

namespace {

bool approx(f32 first, f32 second, f32 epsilon = 1e-3f) {
    return std::fabs(first - second) <= epsilon * (1.0f + std::fabs(second));
}

// Bit-for-bit comparison — used where the two code paths being compared are
// expected to be exactly reproducible (not merely "close enough").
bool bits_equal(f32 first, f32 second) {
    return std::memcmp(&first, &second, sizeof(f32)) == 0;
}

vector<vector<s32>> make_irregular_network() {
    return {
        {1, 2, 3},    // node 0: out-degree 3
        {2},          // node 1: out-degree 1
        {0, 1, 3, 4}, // node 2: out-degree 4 (longest row)
        {},           // node 3: isolated, out-degree 0
        {0}           // node 4: out-degree 1
    };
}

} // namespace

TEST(WeightMatrix, construction) {
    auto network = square_torus(4);                // 16 nodes, every row length 4
    WeightMatrix weight_matrix(network, /*rank=*/8);
    EXPECT_EQ(weight_matrix.node_count, 16);
    EXPECT_EQ(weight_matrix.max_neighbor_count, 4);
    EXPECT_EQ(weight_matrix.rank, 8);
    EXPECT_EQ(weight_matrix.rank_float4_stride, 2);
    EXPECT_FALSE(weight_matrix.using_constant_weight);

    // rank = -1 → min(64, node_count)
    auto network2 = square_torus(3);               // 9 nodes
    WeightMatrix weight_matrix2(network2, /*rank=*/-1);
    EXPECT_EQ(weight_matrix2.rank, 9);

    // empty network rejected
    vector<vector<s32>> empty;
    EXPECT_THROW({ WeightMatrix bad(empty); }, std::invalid_argument);
}

TEST(WeightMatrix, constant_weight) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);

    weight_matrix.set_constant_weight(0.5f);
    EXPECT_TRUE(weight_matrix.using_constant_weight);
    EXPECT_EQ(weight_matrix.constant_weight, 0.5f);
    EXPECT_TRUE(approx(weight_matrix.get(0, 1), 0.5f));
    EXPECT_TRUE(approx(weight_matrix.get(5, 9), 0.5f));

    weight_matrix.set_constant_weight(-0.3f);
    EXPECT_TRUE(approx(weight_matrix.get(2, 7), -0.3f));

    weight_matrix.set_constant_weight(0.0f);
    EXPECT_TRUE(approx(weight_matrix.get(0, 1), 0.0f));
}

TEST(WeightMatrix, stats_and_scale) {
    auto network = square_torus(4);                // every node has exactly 4 neighbors → no padding
    WeightMatrix weight_matrix(network, /*rank=*/8);

    weight_matrix.set_constant_weight(0.5f);
    WeightStats stats = weight_matrix.neighbor_weight_stats();
    EXPECT_TRUE(approx(stats.mean, 0.5f));
    EXPECT_TRUE(approx(stats.root_mean_square, 0.5f));
    EXPECT_LT(stats.standard_deviation, 1e-2f);
    EXPECT_TRUE(approx(stats.min_value, 0.5f));
    EXPECT_TRUE(approx(stats.max_value, 0.5f));

    ScaleResult result = weight_matrix.scale_neighbor_weights_to_root_mean_square(2.0f);
    EXPECT_TRUE(approx(result.after.root_mean_square, 2.0f, 1e-2f));
    EXPECT_FALSE(weight_matrix.using_constant_weight);

    weight_matrix.scale_neighbor_weights_to_root_mean_square(0.0f);
    EXPECT_TRUE(approx(weight_matrix.neighbor_weight_stats().root_mean_square, 0.0f, 1e-5f));

    EXPECT_THROW(weight_matrix.scale_neighbor_weights_to_root_mean_square(-1.0f),
                 std::invalid_argument);
}

TEST(WeightMatrix, get_neighbors) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);

    vector<s32> buffer((usize)weight_matrix.max_neighbor_count);
    for (s64 node = 0; node < weight_matrix.node_count; ++node) {
        s64 count = weight_matrix.get_neighbors(node, buffer.data());
        unordered_set<s32> neighbors(buffer.begin(), buffer.begin() + count);
        unordered_set<s32> expected(network[(usize)node].begin(), network[(usize)node].end());
        EXPECT_EQ(neighbors, expected);
    }
}

TEST(WeightMatrix, get_neighbors_out_of_bounds) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);

    vector<s32> buffer((usize)weight_matrix.max_neighbor_count);
    EXPECT_EQ(weight_matrix.get_neighbors(-1, buffer.data()), 0);
    EXPECT_EQ(weight_matrix.get_neighbors(16, buffer.data()), 0);
    EXPECT_EQ(weight_matrix.get_neighbors(17, buffer.data()), 0);
    EXPECT_EQ(weight_matrix.get_neighbors(4590, buffer.data()), 0);
}

TEST(WeightMatrix, get_out_of_bounds) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, 8);

    EXPECT_EQ(weight_matrix.get(-1, 10), 0.0f);
    EXPECT_EQ(weight_matrix.get(3, 50), 0.0f);
}

TEST(WeightMatrix, update_out_of_bounds) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);

    f32 before = weight_matrix.get(0, 1);
    weight_matrix.update(-1, 1, 1.0f);
    weight_matrix.update(0, 4590, 1.0f);
    f32 after = weight_matrix.get(0, 1);
    EXPECT_TRUE(approx(before, after));
}

TEST(WeightMatrix, update) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);

    f32 before = weight_matrix.get(0, 1);
    weight_matrix.update(/*source=*/0, /*target=*/1, /*delta=*/50.0f,
                         /*learning_rate=*/0.2f, /*l2_regularization=*/1.0f, /*iterations=*/40);
    f32 after = weight_matrix.get(0, 1);

    EXPECT_TRUE(std::isfinite(after));
    EXPECT_GT(after, before);
}

TEST(WeightMatrix, save_load) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);
    weight_matrix.set_constant_weight(0.75f);

    const char *path = "/tmp/spikecorec_test_wm.bin";
    weight_matrix.save(path);

    WeightMatrix loaded(network, /*rank=*/8);
    loaded.load_from_disk(path);
    EXPECT_TRUE(approx(loaded.get(0, 1), 0.75f));
    EXPECT_TRUE(approx(loaded.get(7, 3), 0.75f));
}

TEST(WeightMatrix, neighbor_weights_values) {
    auto network = square_torus(4);                // 16 nodes, out-degree 4, no padding
    WeightMatrix weight_matrix(network, /*rank=*/8);

    vector<f32> weights((usize)(weight_matrix.node_count * weight_matrix.max_neighbor_count));
    weight_matrix.neighbor_weights(weights.data());

    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    for (s64 node = 0; node < weight_matrix.node_count; ++node) {
        s64 degree = weight_matrix.get_neighbors(node, neighbor_buffer.data());
        for (s64 slot = 0; slot < weight_matrix.max_neighbor_count; ++slot) {
            f32 got = weights[(usize)(node * weight_matrix.max_neighbor_count + slot)];
            if (slot < degree) {
                f32 expected = weight_matrix.get((s32)node, neighbor_buffer[(usize)slot]);
                EXPECT_TRUE(approx(got, expected, 1e-3f));
            } else {
                EXPECT_EQ(got, 0.0f);
            }
        }
    }
}

// ── free-function helper ──────────────────────────────────────────────────────

TEST(WeightMatrix, can_safely_cast_s64_to_s32_boundaries) {
    EXPECT_TRUE(can_safely_cast_s64_to_s32(0));
    EXPECT_TRUE(can_safely_cast_s64_to_s32(std::numeric_limits<s32>::min()));
    EXPECT_TRUE(can_safely_cast_s64_to_s32(std::numeric_limits<s32>::max()));
    EXPECT_FALSE(can_safely_cast_s64_to_s32((s64)std::numeric_limits<s32>::max() + 1));
    EXPECT_FALSE(can_safely_cast_s64_to_s32((s64)std::numeric_limits<s32>::min() - 1));
    EXPECT_FALSE(can_safely_cast_s64_to_s32(std::numeric_limits<s64>::max()));
    EXPECT_FALSE(can_safely_cast_s64_to_s32(std::numeric_limits<s64>::min()));
}

// ── check_index_inbounds, exercised directly rather than only via get()/get_neighbors() ──

TEST(WeightMatrix, check_index_inbounds_direct) {
    auto network = square_torus(4); // 16 nodes
    WeightMatrix weight_matrix(network, /*rank=*/8);

    EXPECT_TRUE(weight_matrix.check_index_inbounds(0));
    EXPECT_TRUE(weight_matrix.check_index_inbounds(15));
    EXPECT_FALSE(weight_matrix.check_index_inbounds(-1));
    EXPECT_FALSE(weight_matrix.check_index_inbounds(16));

    EXPECT_TRUE(weight_matrix.check_index_inbounds(0, 15));
    EXPECT_FALSE(weight_matrix.check_index_inbounds(-1, 0));
    EXPECT_FALSE(weight_matrix.check_index_inbounds(0, 16));
    EXPECT_FALSE(weight_matrix.check_index_inbounds(16, 16));
}

// ── rank edge cases ────────────────────────────────────────────────────────────

TEST(WeightMatrix, rank_equals_one) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/1);
    EXPECT_EQ(weight_matrix.rank, 1);
    EXPECT_EQ(weight_matrix.rank_float4_stride, 1);

    f32 value = weight_matrix.get(0, 1);
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_TRUE(bits_equal(weight_matrix.get(0, 1), value)); // repeatable
}

TEST(WeightMatrix, rank_equals_node_count) {
    auto network = square_torus(4);                  // 16 nodes
    WeightMatrix weight_matrix(network, /*rank=*/16); // rank == node_count: no compression
                                                       // benefit, but must still work
    EXPECT_EQ(weight_matrix.rank, 16);
    EXPECT_EQ(weight_matrix.rank_float4_stride, 4);

    vector<f32> weights((usize)(weight_matrix.node_count * weight_matrix.max_neighbor_count));
    weight_matrix.neighbor_weights(weights.data());
    for (f32 weight : weights) {
        EXPECT_TRUE(std::isfinite(weight));
    }
}

TEST(WeightMatrix, node_count_one_with_self_loop) {
    // side_length=1 makes every square_torus direction wrap back onto the same
    // single cell, so network[0] is four duplicate self-loop entries.
    auto network = square_torus(1);
    ASSERT_EQ(network.size(), 1u);
    ASSERT_EQ(network[0].size(), 4u);
    for (s32 neighbor : network[0]) EXPECT_EQ(neighbor, 0);

    WeightMatrix weight_matrix(network);
    EXPECT_EQ(weight_matrix.node_count, 1);
    EXPECT_EQ(weight_matrix.max_neighbor_count, 4); // raw row length, duplicates included
    EXPECT_EQ(weight_matrix.rank, 1);                // default: min(64, node_count)

    // get() is a bounds-checked matrix lookup, not edge-gated (see
    // get_ignores_edge_existence_bounds_checked_only below) -- it returns a
    // finite, deterministic value for (0,0) regardless of what the k^2-tree
    // reports for that pair.
    f32 value = weight_matrix.get(0, 0);
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_TRUE(bits_equal(weight_matrix.get(0, 0), value));

    // NOTE (discovered, not fixed here -- out of scope for this test-only round):
    // K2Tree::compute_tree_parameters special-cases node_count<=1 to tree_height=0,
    // which (per build_tree_arrays) builds an entirely empty internal/leaf bit
    // array. A single-node graph therefore cannot represent ANY edge in its
    // k^2-tree -- including the self-loop this adjacency list declares --
    // regardless of what was passed to from_adjacency_list: both
    // weight_matrix.k2tree.adjacent(0, 0) and weight_matrix.get_neighbors(0, ...)
    // report "no edge" here even though network[0] == {0, 0, 0, 0}. This looks
    // like a genuine edge-case bug in the node_count==1 path shared by K2Tree
    // and WeightMatrix -- flagged for the team rather than asserted as correct
    // behavior in this suite.
}

TEST(WeightMatrix, max_rank_boundary_allowed_and_rejected) {
    auto network = square_torus(4);

    // rank=256 -> rank_float4_stride=64 == MAX_RANK_FLOAT4_STRIDE, still allowed
    WeightMatrix weight_matrix(network, /*rank=*/256);
    EXPECT_EQ(weight_matrix.rank_float4_stride, 64);
    EXPECT_TRUE(std::isfinite(weight_matrix.get(0, 1)));

    // rank=257 -> rank_float4_stride=65 > MAX_RANK_FLOAT4_STRIDE, rejected
    EXPECT_THROW({ WeightMatrix too_large(network, /*rank=*/257); }, std::invalid_argument);
}

TEST(WeightMatrix, rank_zero_defaults_like_negative_one) {
    auto network = square_torus(3); // 9 nodes
    WeightMatrix rank_zero(network, /*rank=*/0);
    WeightMatrix rank_default(network, /*rank=*/-1);
    EXPECT_EQ(rank_zero.rank, rank_default.rank);
    EXPECT_EQ(rank_zero.rank, 9);
}

// ── determinism / bit-exact repeatability ──────────────────────────────────────
// These matter because the upcoming shared-basis generalization (#52) needs this
// suite as an exact regression baseline, not just an approximate one.

TEST(WeightMatrix, get_is_bitwise_deterministic_across_repeated_calls) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8, /*check_indexing=*/true,
                               /*max_neighbor_count=*/-1, /*weight_seed=*/42);

    f32 first_value = weight_matrix.get(3, 12);
    for (int iteration = 0; iteration < 100; ++iteration) {
        EXPECT_TRUE(bits_equal(weight_matrix.get(3, 12), first_value))
            << "get() must be bit-for-bit reproducible across repeated calls with unchanged state";
    }
}

TEST(WeightMatrix, neighbor_weights_is_bitwise_deterministic_across_repeated_dispatches) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8, /*check_indexing=*/true,
                               /*max_neighbor_count=*/-1, /*weight_seed=*/7);

    usize total = (usize)(weight_matrix.node_count * weight_matrix.max_neighbor_count);
    vector<f32> first_pass(total);
    vector<f32> second_pass(total);
    weight_matrix.neighbor_weights(first_pass.data());
    weight_matrix.neighbor_weights(second_pass.data());

    EXPECT_EQ(std::memcmp(first_pass.data(), second_pass.data(), total * sizeof(f32)), 0)
        << "neighbor_weights() GPU dispatch must be bit-for-bit reproducible across repeated calls";
}

TEST(WeightMatrix, get_and_neighbor_weights_agree_closely) {
    // get() is the host-side dot product; neighbor_weights() is the GPU-kernel path
    // (Metal's neighbor_weights_kernel). Both accumulate rank_float4_stride lanes in
    // the same x/y/z/w order, but empirically they are NOT bit-identical -- observed
    // differences are a handful of ULPs (relative error ~1e-7, well within float32
    // rounding), consistent with the GPU using a fused multiply-add for `dot()` where
    // the CPU does sequential multiply-then-add. Worth keeping in mind for the #52
    // shared-basis generalization's bit-compatibility requirement: these two
    // reconstruction paths are numerically consistent today, not bit-for-bit equal.
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8, /*check_indexing=*/true,
                               /*max_neighbor_count=*/-1, /*weight_seed=*/99);

    vector<f32> gpu_weights((usize)(weight_matrix.node_count * weight_matrix.max_neighbor_count));
    weight_matrix.neighbor_weights(gpu_weights.data());

    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    for (s64 node = 0; node < weight_matrix.node_count; ++node) {
        s64 degree = weight_matrix.get_neighbors(node, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            f32 gpu_value = gpu_weights[(usize)(node * weight_matrix.max_neighbor_count + slot)];
            f32 cpu_value = weight_matrix.get((s32)node, neighbor_buffer[(usize)slot]);
            EXPECT_TRUE(approx(gpu_value, cpu_value, 1e-5f))
                << "node=" << node << " slot=" << slot << " gpu=" << gpu_value << " cpu=" << cpu_value;
        }
    }
}

// ── weight_seed reproducibility (constructor parameter previously untested) ────

TEST(WeightMatrix, weight_seed_reproducibility) {
    auto network = square_torus(4);

    WeightMatrix first(network, /*rank=*/8, /*check_indexing=*/true,
                       /*max_neighbor_count=*/-1, /*weight_seed=*/123);
    WeightMatrix second(network, /*rank=*/8, /*check_indexing=*/true,
                        /*max_neighbor_count=*/-1, /*weight_seed=*/123);

    for (s32 source = 0; source < 16; ++source) {
        for (s32 target = 0; target < 16; ++target) {
            EXPECT_TRUE(bits_equal(first.get(source, target), second.get(source, target)));
        }
    }
}

TEST(WeightMatrix, different_weight_seeds_produce_different_weights) {
    auto network = square_torus(4);

    WeightMatrix first(network, /*rank=*/8, /*check_indexing=*/true,
                       /*max_neighbor_count=*/-1, /*weight_seed=*/123);
    WeightMatrix second(network, /*rank=*/8, /*check_indexing=*/true,
                        /*max_neighbor_count=*/-1, /*weight_seed=*/456);

    bool any_difference = false;
    for (s32 source = 0; source < 16 && !any_difference; ++source) {
        for (s32 target = 0; target < 16; ++target) {
            if (!bits_equal(first.get(source, target), second.get(source, target))) {
                any_difference = true;
                break;
            }
        }
    }
    EXPECT_TRUE(any_difference);
}

TEST(WeightMatrix, update_reproducibility_with_seeded_construction) {
    auto network = square_torus(4);

    WeightMatrix first(network, /*rank=*/8, /*check_indexing=*/true,
                       /*max_neighbor_count=*/-1, /*weight_seed=*/55);
    WeightMatrix second(network, /*rank=*/8, /*check_indexing=*/true,
                        /*max_neighbor_count=*/-1, /*weight_seed=*/55);

    first.update(/*source=*/0, /*target=*/1, /*delta=*/50.0f, /*learning_rate=*/0.2f,
                 /*l2_regularization=*/1.0f, /*iterations=*/40);
    second.update(/*source=*/0, /*target=*/1, /*delta=*/50.0f, /*learning_rate=*/0.2f,
                  /*l2_regularization=*/1.0f, /*iterations=*/40);

    EXPECT_TRUE(bits_equal(first.get(0, 1), second.get(0, 1)));
    // untouched pairs should also still agree exactly
    EXPECT_TRUE(bits_equal(first.get(3, 7), second.get(3, 7)));
}

// ── max_neighbor_count: default derivation, explicit padding, explicit truncation ──

TEST(WeightMatrix, max_neighbor_count_derived_from_longest_row) {
    auto network = make_irregular_network();
    WeightMatrix weight_matrix(network, /*rank=*/4);
    EXPECT_EQ(weight_matrix.max_neighbor_count, 4);

    vector<s32> buffer(4);
    EXPECT_EQ(weight_matrix.get_neighbors(3, buffer.data()), 0); // isolated node
    EXPECT_EQ(weight_matrix.get_neighbors(1, buffer.data()), 1);
    EXPECT_EQ(buffer[0], 2);

    vector<f32> weights((usize)(weight_matrix.node_count * weight_matrix.max_neighbor_count));
    weight_matrix.neighbor_weights(weights.data());
    // node 3 (isolated): every slot sentinel-padded
    for (s64 slot = 0; slot < 4; ++slot) {
        EXPECT_EQ(weights[(usize)(3 * 4 + slot)], 0.0f);
    }
    // node 1 (out-degree 1): slot 0 real, slots 1-3 sentinel. GPU (neighbor_weights)
    // and CPU (get()) paths are numerically consistent but not always bit-identical
    // (see get_and_neighbor_weights_agree_closely), so compare with a tight tolerance.
    EXPECT_TRUE(approx(weights[(usize)(1 * 4 + 0)], weight_matrix.get(1, 2), 1e-5f));
    for (s64 slot = 1; slot < 4; ++slot) {
        EXPECT_EQ(weights[(usize)(1 * 4 + slot)], 0.0f);
    }
}

TEST(WeightMatrix, max_neighbor_count_explicit_override_larger_than_natural) {
    auto network = make_irregular_network();
    WeightMatrix weight_matrix(network, /*rank=*/4, /*check_indexing=*/true,
                               /*max_neighbor_count=*/10);
    EXPECT_EQ(weight_matrix.max_neighbor_count, 10);

    vector<s32> buffer(10);
    s64 degree = weight_matrix.get_neighbors(2, buffer.data());
    EXPECT_EQ(degree, 4); // real degree, not padded up to 10

    unordered_set<s32> expected = {0, 1, 3, 4};
    unordered_set<s32> actual(buffer.begin(), buffer.begin() + degree);
    EXPECT_EQ(actual, expected);

    vector<f32> weights((usize)(weight_matrix.node_count * weight_matrix.max_neighbor_count));
    weight_matrix.neighbor_weights(weights.data());
    for (s64 slot = 4; slot < 10; ++slot) {
        EXPECT_EQ(weights[(usize)(2 * 10 + slot)], 0.0f); // padding beyond real degree
    }
}

TEST(WeightMatrix, max_neighbor_count_explicit_override_truncates) {
    auto network = make_irregular_network();
    // node 2 has real out-degree 4; capping max_neighbor_count at 2 must truncate cleanly
    WeightMatrix weight_matrix(network, /*rank=*/4, /*check_indexing=*/true,
                               /*max_neighbor_count=*/2);
    EXPECT_EQ(weight_matrix.max_neighbor_count, 2);

    vector<s32> buffer(2);
    s64 degree = weight_matrix.get_neighbors(2, buffer.data());
    EXPECT_EQ(degree, 2);
    for (s64 slot = 0; slot < degree; ++slot) {
        EXPECT_TRUE(weight_matrix.k2tree.adjacent(2, buffer[(usize)slot]));
    }
}

// ── disconnected / self-loop / non-edge structural cases ───────────────────────

TEST(WeightMatrix, no_edges_at_all_graph) {
    vector<vector<s32>> network(5); // 5 isolated nodes, no edges anywhere
    WeightMatrix weight_matrix(network, /*rank=*/4);
    EXPECT_EQ(weight_matrix.max_neighbor_count, 0);

    vector<s32> buffer(1);
    for (s64 node = 0; node < 5; ++node) {
        EXPECT_EQ(weight_matrix.get_neighbors(node, buffer.data()), 0);
    }

    WeightStats stats = weight_matrix.neighbor_weight_stats();
    EXPECT_EQ(stats.mean, 0.0f);
    EXPECT_EQ(stats.standard_deviation, 0.0f);
    EXPECT_EQ(stats.root_mean_square, 0.0f);
    EXPECT_EQ(stats.min_value, 0.0f);
    EXPECT_EQ(stats.max_value, 0.0f);

    // neighbor_weights() on a zero-sized output is documented to be a no-op, not a crash
    vector<f32> empty_output;
    weight_matrix.neighbor_weights(empty_output.data());
}

TEST(WeightMatrix, get_ignores_edge_existence_bounds_checked_only) {
    // get() is documented as a plain U*V^T matrix-element lookup: it is bounds-checked
    // but NOT edge-gated. A non-adjacent pair still returns the raw dot product rather
    // than 0 -- unlike neighbor_weights()/get_neighbors(), which are restricted to real
    // k^2-tree edges. This is the intended contract, not a bug.
    auto network = make_irregular_network();
    WeightMatrix weight_matrix(network, /*rank=*/4);

    ASSERT_FALSE(weight_matrix.k2tree.adjacent(3, 0)); // node 3 has no outgoing edges at all
    f32 value = weight_matrix.get(3, 0);
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_TRUE(bits_equal(weight_matrix.get(3, 0), value)); // deterministic regardless of edge existence
}

TEST(WeightMatrix, self_loop_mixed_with_normal_edges) {
    vector<vector<s32>> network = {
        {1, 2},
        {1, 2}, // node 1: self-loop + normal edge
        {0}
    };
    WeightMatrix weight_matrix(network, /*rank=*/4);

    EXPECT_TRUE(weight_matrix.k2tree.adjacent(1, 1));
    vector<s32> buffer(2);
    s64 degree = weight_matrix.get_neighbors(1, buffer.data());
    unordered_set<s32> neighbors(buffer.begin(), buffer.begin() + degree);
    EXPECT_EQ(neighbors, (unordered_set<s32>{1, 2}));
    EXPECT_TRUE(std::isfinite(weight_matrix.get(1, 1)));
}

// ── degenerate weight values ────────────────────────────────────────────────────

TEST(WeightMatrix, all_zero_weight_exact) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);
    weight_matrix.set_constant_weight(0.0f);

    EXPECT_EQ(weight_matrix.get(0, 1), 0.0f);
    EXPECT_EQ(weight_matrix.get(5, 9), 0.0f);
    EXPECT_EQ(weight_matrix.get(15, 4), 0.0f);

    WeightStats stats = weight_matrix.neighbor_weight_stats();
    EXPECT_EQ(stats.mean, 0.0f);
    EXPECT_EQ(stats.standard_deviation, 0.0f);
    EXPECT_EQ(stats.root_mean_square, 0.0f);
    EXPECT_EQ(stats.min_value, 0.0f);
    EXPECT_EQ(stats.max_value, 0.0f);
}

TEST(WeightMatrix, extreme_magnitude_constant_weight) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);

    weight_matrix.set_constant_weight(1e10f);
    EXPECT_TRUE(std::isfinite(weight_matrix.get(0, 1)));
    EXPECT_TRUE(approx(weight_matrix.get(0, 1), 1e10f, 1e-3f));

    weight_matrix.set_constant_weight(-1e10f);
    EXPECT_TRUE(std::isfinite(weight_matrix.get(0, 1)));
    EXPECT_TRUE(approx(weight_matrix.get(0, 1), -1e10f, 1e-3f));

    weight_matrix.set_constant_weight(1e-10f);
    EXPECT_TRUE(std::isfinite(weight_matrix.get(0, 1)));
    EXPECT_TRUE(approx(weight_matrix.get(0, 1), 1e-10f, 1e-2f));
}

// ── save/load across a dimension change (the reallocation branch of load_from_disk) ──
//
// NOTE (discovered, not fixed here -- out of scope for this test-only round):
// set_constant_weight()'s scale factor is derived from the logical `rank`
// (sqrtf(|value| / rank)), but the fill loop writes that scale into all
// rank_float4_stride * 4 lanes, and get()/neighbor_weights() sum over every one
// of those lanes. When rank is a multiple of 4, rank_float4_stride * 4 == rank
// and everything cancels out correctly. When it is NOT (e.g. rank=3, so
// rank_float4_stride=1 but 4 lanes actually get filled), the extra padding
// lane(s) still contribute full signal to the dot product, and get() reproduces
// value * (rank_float4_stride * 4) / rank instead of value -- e.g. rank=3
// inflates 0.42 to ~0.56. Ranks below use multiples of 4 to isolate what this
// test is actually about (the load_from_disk reallocation branch) from that
// separate discrepancy, which is flagged for the team instead.

TEST(WeightMatrix, save_load_reallocates_on_dimension_change) {
    auto small_network = square_torus(3); // 9 nodes
    WeightMatrix source(small_network, /*rank=*/4); // rank_float4_stride=1
    source.set_constant_weight(0.42f);

    const char *path = "/tmp/spikecorec_test_wm_realloc.bin";
    source.save(path);

    auto large_network = square_torus(4); // 16 nodes, different rank/node_count
    WeightMatrix destination(large_network, /*rank=*/12); // rank_float4_stride=3
    ASSERT_NE(destination.node_count, source.node_count);
    ASSERT_NE(destination.rank_float4_stride, source.rank_float4_stride);

    destination.load_from_disk(path);
    EXPECT_EQ(destination.node_count, source.node_count);
    EXPECT_EQ(destination.rank, source.rank);
    EXPECT_EQ(destination.rank_float4_stride, source.rank_float4_stride);
    EXPECT_TRUE(approx(destination.get(0, 1), 0.42f));
    EXPECT_TRUE(approx(destination.get(8, 2), 0.42f));
}

// ── move semantics ──────────────────────────────────────────────────────────────
// NOTE: WeightMatrix's move-ASSIGNMENT operator (`operator=(WeightMatrix&&) = default`)
// is deliberately not exercised here. Assigning into any live WeightMatrix instance --
// the only kind reachable through the public API, since the default constructor is
// deleted -- trips the assert in GpuPointer::operator=(GpuPointer&&) ("... would leak
// an existing ... allocation") and aborts the whole process; confirmed with a standalone
// repro outside this suite. Encoding that as a passing/expected-crash test here would
// lock in what looks like a latent design bug as if it were the intended contract, so
// it is flagged in the report instead of tested. Move-CONSTRUCTION has no such issue
// (GpuPointer's move constructor never asserts) and is exercised below.

TEST(WeightMatrix, move_construction_preserves_state) {
    auto network = square_torus(4);
    WeightMatrix original(network, /*rank=*/8);
    original.set_constant_weight(0.6f);
    f32 expected = original.get(2, 5);

    WeightMatrix moved(std::move(original));
    EXPECT_TRUE(approx(moved.get(2, 5), expected));
    EXPECT_EQ(moved.node_count, 16);
}
