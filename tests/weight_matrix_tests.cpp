#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <gtest/gtest.h>

#include "spikecorec/core/types.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/core/topologies.h"

using namespace std;
using namespace spikecorec;

namespace {

bool approx(f32 first, f32 second, f32 epsilon = 1e-3f) {
    return std::fabs(first - second) <= epsilon * (1.0f + std::fabs(second));
}

u32 float_bit_pattern(f32 value) {
    u32 bit_pattern = 0;
    std::memcpy(&bit_pattern, &value, sizeof(u32));
    return bit_pattern;
}

// Any node that is NOT a neighbor of source_node, for the edge-scoped "not a real
// edge" rejection tests. Returns -1 if the graph is complete out of source_node.
s32 find_non_neighbor(const WeightMatrix &weight_matrix, s32 source_node) {
    for (s64 candidate = 0; candidate < weight_matrix.node_count; ++candidate) {
        if (candidate == source_node) continue;
        if (!weight_matrix.k2tree.adjacent(source_node, (s32)candidate)) {
            return (s32)candidate;
        }
    }
    return -1;
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

// A null `coefficients` argument must reproduce the pre-shared-basis (pre-#52)
// gpu_neighbor_weights numerics EXACTLY, so that every caller that predates Ck keeps
// its current results. Checked bit-for-bit against the reserved DEFAULT_MATRIX_INDEX
// all-ones coefficient vector, which is itself documented to reduce to the original
// dot(U,V).
TEST(WeightMatrix, null_coefficients_match_all_ones_coefficients_bit_for_bit) {
    auto network = square_torus(4);
    // rank 6 -> rank_float4_stride 2 -> 8 effective lanes, so two padding lanes are
    // exercised alongside the six logical ones.
    WeightMatrix weight_matrix(network, /*rank=*/6, /*check_indexing=*/true,
                               /*max_neighbor_count=*/-1, /*weight_seed=*/42);

    s64 total_pair_count = weight_matrix.node_count * weight_matrix.max_neighbor_count;
    ASSERT_GT(total_pair_count, 0);

    vector<f32> all_ones_weights((usize)total_pair_count);
    weight_matrix.neighbor_weights(all_ones_weights.data());

    GpuPointer<f32> device_weights = allocate<f32>((usize)total_pair_count * sizeof(f32));
    gpu_neighbor_weights(
        weight_matrix.U_matrix.get_contents(),
        weight_matrix.V_matrix.get_contents(),
        weight_matrix.k2tree.internal_node_words.get_contents(),
        weight_matrix.k2tree.leaf_node_words.get_contents(),
        weight_matrix.k2tree.rank_superblock_table.get_contents(),
        weight_matrix.k2tree.rank_subblock_table.get_contents(),
        weight_matrix.k2tree.branching_factor,
        weight_matrix.k2tree.superblock_size_words,
        weight_matrix.k2tree.padded_node_count,
        weight_matrix.k2tree.tree_height,
        weight_matrix.k2tree.internal_bit_count,
        weight_matrix.node_count,
        weight_matrix.max_neighbor_count,
        weight_matrix.rank_float4_stride,
        /*coefficients=*/nullptr,
        device_weights.get_contents()
    );
    synchronize_gpu_work();
    vector<f32> null_coefficient_weights(
        device_weights.get_contents(), device_weights.get_contents() + total_pair_count);
    deallocate(std::move(device_weights));

    for (s64 pair_index = 0; pair_index < total_pair_count; ++pair_index) {
        EXPECT_EQ(float_bit_pattern(all_ones_weights[(usize)pair_index]),
                  float_bit_pattern(null_coefficient_weights[(usize)pair_index]))
            << "pair_index=" << pair_index;
    }
}

TEST(WeightMatrix, edge_delay_reads_back_exactly) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);

    // Delay costs nothing until a per-edge delay is actually set.
    EXPECT_EQ(weight_matrix.delay_matrix_index, -1);
    EXPECT_EQ(weight_matrix.matrix_count(), 1);

    const s32 source_node = 0;
    const s32 target_node = network[0][0];
    const s32 other_target_node = network[0][1];

    weight_matrix.set_edge_delay_ticks(source_node, target_node, 7);
    EXPECT_EQ(weight_matrix.get_edge_delay_ticks(source_node, target_node), 7);

    // Stored as one more matrix in the shared-basis family, registered lazily.
    EXPECT_EQ(weight_matrix.delay_matrix_index, 1);
    EXPECT_EQ(weight_matrix.matrix_count(), 2);

    // Every other real edge still reads the constant default.
    EXPECT_EQ(weight_matrix.get_edge_delay_ticks(source_node, other_target_node),
              weight_matrix.constant_delay_ticks);

    // Setting again overwrites rather than accumulating.
    weight_matrix.set_edge_delay_ticks(source_node, target_node, 3);
    EXPECT_EQ(weight_matrix.get_edge_delay_ticks(source_node, target_node), 3);
    EXPECT_EQ(weight_matrix.matrix_count(), 2);

    // The >= 1 floor is preserved.
    EXPECT_THROW(weight_matrix.set_edge_delay_ticks(source_node, target_node, 0),
                 std::invalid_argument);
    EXPECT_THROW(weight_matrix.set_edge_delay_ticks(source_node, target_node, -4),
                 std::invalid_argument);
    EXPECT_EQ(weight_matrix.get_edge_delay_ticks(source_node, target_node), 3);

    // set_edge_delay_ticks does not flip the constant-delay dispatch hint.
    EXPECT_TRUE(weight_matrix.using_constant_delay_ticks);
}

TEST(WeightMatrix, edge_delay_survives_refit) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);

    const s32 source_node = 2;
    const s32 target_node = network[2][0];
    weight_matrix.set_edge_delay_ticks(source_node, target_node, 5);

    // An ordinary per-edge matrix with a pending delta, so refit has real work to do
    // and genuinely moves the shared U/V basis under the delay matrix.
    s64 ordinary_matrix_index =
        weight_matrix.add_coefficient_vector(vector<f32>((usize)weight_matrix.rank, 0.5f));
    weight_matrix.accumulate_edge_delta(ordinary_matrix_index, source_node, target_node, 0.25f);

    f32 basis_value_before = weight_matrix.U_matrix.get_contents()[0].x;
    weight_matrix.refit(/*sweep_count=*/2);
    EXPECT_NE(float_bit_pattern(basis_value_before),
              float_bit_pattern(weight_matrix.U_matrix.get_contents()[0].x));

    EXPECT_EQ(weight_matrix.get_edge_delay_ticks(source_node, target_node), 5);

    // The delay matrix's Ck stays pinned at all-zero, so the full reconstruction
    // path agrees exactly with the accessor even after the basis moved underneath it.
    EXPECT_EQ(float_bit_pattern(weight_matrix.get_for_matrix(source_node, target_node,
                                                            weight_matrix.delay_matrix_index)),
              float_bit_pattern(5.0f));

    // The ordinary matrix's Sk was folded into the basis and cleared; the delay
    // matrix's was deliberately left alone, which is what keeps delays exact.
    EXPECT_FALSE(weight_matrix.sparse_delta_touched[(usize)ordinary_matrix_index]);
    EXPECT_TRUE(weight_matrix.sparse_delta_touched[(usize)weight_matrix.delay_matrix_index]);
}

TEST(WeightMatrix, constant_delay_applies_to_every_real_edge) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);

    weight_matrix.set_constant_delay_ticks(4);
    EXPECT_TRUE(weight_matrix.using_constant_delay_ticks);
    EXPECT_EQ(weight_matrix.constant_delay_ticks, 4);

    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    for (s64 node_index = 0; node_index < weight_matrix.node_count; ++node_index) {
        s64 degree = weight_matrix.get_neighbors(node_index, neighbor_buffer.data());
        ASSERT_GT(degree, 0);
        for (s64 slot = 0; slot < degree; ++slot) {
            EXPECT_EQ(weight_matrix.get_edge_delay_ticks((s32)node_index,
                                                         neighbor_buffer[(usize)slot]), 4);
        }
    }

    // A purely constant delay allocates no per-edge storage at all.
    EXPECT_EQ(weight_matrix.delay_matrix_index, -1);

    // A per-edge delay overrides the constant for that one edge only.
    weight_matrix.set_edge_delay_ticks(0, network[0][0], 9);
    EXPECT_EQ(weight_matrix.get_edge_delay_ticks(0, network[0][0]), 9);
    EXPECT_EQ(weight_matrix.get_edge_delay_ticks(0, network[0][1]), 4);

    EXPECT_THROW(weight_matrix.set_constant_delay_ticks(0), std::invalid_argument);
}

TEST(WeightMatrix, edge_delay_on_nonexistent_edge_throws) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);

    const s32 source_node = 0;
    s32 non_neighbor = find_non_neighbor(weight_matrix, source_node);
    ASSERT_GE(non_neighbor, 0);

    EXPECT_THROW(weight_matrix.set_edge_delay_ticks(source_node, non_neighbor, 2),
                 std::invalid_argument);
    EXPECT_THROW((void)weight_matrix.get_edge_delay_ticks(source_node, non_neighbor),
                 std::invalid_argument);

    // Out-of-bounds indices are rejected the same way.
    EXPECT_THROW(weight_matrix.set_edge_delay_ticks(source_node, 4590, 2),
                 std::invalid_argument);
    EXPECT_THROW(weight_matrix.set_edge_delay_ticks(-1, 0, 2), std::invalid_argument);
    EXPECT_THROW((void)weight_matrix.get_edge_delay_ticks(source_node, 4590),
                 std::invalid_argument);

    // A rejected call registers nothing.
    EXPECT_EQ(weight_matrix.delay_matrix_index, -1);
    EXPECT_EQ(weight_matrix.matrix_count(), 1);
}
