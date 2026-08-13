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
#include "spikecorec/core/backend.h"
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

// Reconstructs an f32 from an exact bit pattern — used for the golden-value
// regression tests below, so the expected value is pinned to an exact bit
// pattern rather than a decimal literal (which could round differently on parse).
f32 from_bits(uint32_t bits) {
    f32 value;
    std::memcpy(&value, &bits, sizeof(f32));
    return value;
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

vector<vector<s32>> make_irregular_network() {
    return {
        {1, 2, 3},    // node 0: out-degree 3
        {2},          // node 1: out-degree 1
        {0, 1, 3, 4}, // node 2: out-degree 4 (longest row)
        {},           // node 3: isolated, out-degree 0
        {0}           // node 4: out-degree 1
    };
}

// Deterministic, RNG-free "true" row for the known-low-rank refit fixture
// below -- a simple arithmetic pattern, not randomly generated, so the fixture
// is fully reproducible without needing to seed/carry a generator.
vector<f32> deterministic_row(s64 node_index, s64 lane_count, s64 row_seed) {
    vector<f32> row((usize)lane_count);
    for (s64 lane_index = 0; lane_index < lane_count; ++lane_index) {
        s64 pattern = (node_index * 7 + lane_index * 13 + row_seed * 31) % 11;
        row[(usize)lane_index] = (f32)pattern - 5.0f;
    }
    return row;
}

// Σ_lane u_row[lane]·coefficients[lane]·v_row[lane] -- the same reconstruction
// formula WeightMatrix::reconstruct_entry implements, computed test-side
// against the fixture's own "true" rows/coefficients.
f32 low_rank_dot(const vector<f32> &u_row, const vector<f32> &coefficients, const vector<f32> &v_row) {
    f32 sum = 0.0f;
    for (usize lane_index = 0; lane_index < u_row.size(); ++lane_index) {
        sum += u_row[lane_index] * coefficients[lane_index] * v_row[lane_index];
    }
    return sum;
}

// Whitebox check for whether a matrix's Sk currently holds no accumulated
// deltas at all. sparse_delta_buffers is a fixed-capacity GPU-resident array
// per matrix (ticket #53/D3 rework), not a hash map, so there is no
// container-level .empty() to call -- this instead reads the raw contents
// directly (the same way sparse_delta_buffer_uses_position_indexed_gpu_resident_layout
// above does) and confirms every slot is exactly 0.0f, matching the old map's
// "no entries" semantics. Returns true trivially when the matrix has no
// representable neighbor slots at all (its buffer is never allocated).
bool sparse_delta_buffer_contents_are_all_zero(const WeightMatrix &weight_matrix, s64 matrix_index) {
    s64 total_slot_count = weight_matrix.node_count * weight_matrix.max_neighbor_count;
    if (total_slot_count == 0) {
        return true;
    }
    const f32 *delta_data = weight_matrix.sparse_delta_buffers[(usize)matrix_index].get_contents();
    for (s64 slot_index = 0; slot_index < total_slot_count; ++slot_index) {
        if (delta_data[slot_index] != 0.0f) {
            return false;
        }
    }
    return true;
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

TEST(WeightMatrix, node_count_one_with_self_loop_throws) {
    // A single-node graph whose only possible edge is a self-loop must be
    // rejected, not silently accepted -- self-loops are not supported (see
    // K2Tree's self-loop validation in build_tree_arrays, which every
    // WeightMatrix construction funnels through). square_torus(1) no longer
    // produces this network itself (it now omits the degenerate wraparound
    // self-loop rather than reporting it -- see Topologies.square_torus_edge_cases),
    // so this network is declared explicitly here to exercise the rejection.
    vector<vector<s32>> network_with_self_loop = {{0}};
    EXPECT_THROW({ WeightMatrix weight_matrix(network_with_self_loop); }, std::invalid_argument);
}

TEST(WeightMatrix, node_count_one_with_no_edges) {
    // A single, edge-free node (what square_torus(1) now produces) constructs
    // normally: get() is a bounds-checked matrix lookup, not edge-gated (see
    // get_ignores_edge_existence_bounds_checked_only below), so it returns a
    // finite, deterministic value for (0,0) regardless of there being no edge.
    vector<vector<s32>> network(1); // one node, no outgoing edges

    WeightMatrix weight_matrix(network);
    EXPECT_EQ(weight_matrix.node_count, 1);
    EXPECT_EQ(weight_matrix.max_neighbor_count, 0);
    EXPECT_EQ(weight_matrix.rank, 1); // default: min(64, node_count)

    f32 value = weight_matrix.get(0, 0);
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_TRUE(bits_equal(weight_matrix.get(0, 0), value));
    EXPECT_EQ(weight_matrix.k2tree.adjacent(0, 0), 0);
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

// ── shared-basis (#52/D2): DEFAULT_MATRIX_INDEX must reproduce pre-#52 values
// bit-for-bit ───────────────────────────────────────────────────────────────
//
// The hex bit patterns below were captured by running this exact
// WeightMatrix(...)/get()/neighbor_weights() call sequence against the
// pre-#52 implementation (git-stashed back to the commit before the
// shared-basis change was made), for a representative spread of
// configurations: rank=1 (smallest possible, 3 unused-but-populated padding
// lanes), rank=8 (multiple of 4, no padding), rank=6 and rank=10 (not
// multiples of 4, so padding lanes are exercised), and node_count=1. This is
// the literal enforcement of "single-matrix Ck=1 reproduces current weights
// bit-for-bit" -- not just a self-consistency check, but a real comparison
// against the values the pre-shared-basis dot(U,V) actually produced.
TEST(WeightMatrix, get_reproduces_pre_shared_basis_values_bit_for_bit) {
    {
        // rank=1: rank_float4_stride=1, 3 unused-but-populated lanes.
        auto network = square_torus(4);
        WeightMatrix weight_matrix(network, /*rank=*/1, true, -1, /*weight_seed=*/42);
        EXPECT_TRUE(bits_equal(weight_matrix.get(0, 1), from_bits(0x3eb25f3au)));
        EXPECT_TRUE(bits_equal(weight_matrix.get(3, 12), from_bits(0x40242ee2u)));
    }
    {
        // rank=8: multiple of 4, no padding.
        auto network = square_torus(4);
        WeightMatrix weight_matrix(network, /*rank=*/8, true, -1, /*weight_seed=*/42);
        EXPECT_TRUE(bits_equal(weight_matrix.get(0, 1), from_bits(0x3e011f04u)));
        EXPECT_TRUE(bits_equal(weight_matrix.get(3, 12), from_bits(0xc0880ba9u)));
    }
    {
        // rank=6: not a multiple of 4, rank_float4_stride=2, 2 padding lanes.
        auto network = square_torus(4);
        WeightMatrix weight_matrix(network, /*rank=*/6, true, -1, /*weight_seed=*/42);
        EXPECT_TRUE(bits_equal(weight_matrix.get(0, 1), from_bits(0x3e011f04u)));
        EXPECT_TRUE(bits_equal(weight_matrix.get(3, 12), from_bits(0xc0880ba9u)));
    }
    {
        // rank=10: not a multiple of 4, rank_float4_stride=3, 2 padding lanes.
        auto network = square_torus(4);
        WeightMatrix weight_matrix(network, /*rank=*/10, true, -1, /*weight_seed=*/42);
        EXPECT_TRUE(bits_equal(weight_matrix.get(0, 1), from_bits(0xc02378a2u)));
        EXPECT_TRUE(bits_equal(weight_matrix.get(3, 12), from_bits(0xbfffe81bu)));
    }
    {
        // node_count=1.
        auto network = square_torus(1);
        WeightMatrix weight_matrix(network, /*rank=*/-1, true, -1, /*weight_seed=*/42);
        EXPECT_TRUE(bits_equal(weight_matrix.get(0, 0), from_bits(0x3dad54c2u)));
    }
}

TEST(WeightMatrix, neighbor_weights_reproduces_pre_shared_basis_values_bit_for_bit) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8, true, -1, /*weight_seed=*/42);
    vector<f32> weights((usize)(weight_matrix.node_count * weight_matrix.max_neighbor_count));
    weight_matrix.neighbor_weights(weights.data());
    EXPECT_TRUE(bits_equal(weights[0], from_bits(0x3e011f04u)));
    EXPECT_TRUE(bits_equal(weights[5], from_bits(0xc02627beu)));
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

// ── shared-basis (#52/D2): multiple matrices sharing one basis ────────────────

TEST(WeightMatrix, multiple_matrices_share_one_basis_and_reconstruct_distinctly) {
    // rank=3 (small, hand-checkable), a fixed weight_seed for reproducible U/V.
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/3, /*check_indexing=*/true,
                               /*max_neighbor_count=*/-1, /*weight_seed=*/7);
    ASSERT_EQ(weight_matrix.matrix_count(), 1); // just DEFAULT_MATRIX_INDEX so far

    vector<f32> coefficients_a = {2.0f, 0.5f, -1.0f};
    vector<f32> coefficients_b = {-3.0f, 1.0f, 4.0f};
    s64 matrix_a = weight_matrix.add_coefficient_vector(coefficients_a);
    s64 matrix_b = weight_matrix.add_coefficient_vector(coefficients_b);
    EXPECT_NE(matrix_a, matrix_b);
    EXPECT_EQ(weight_matrix.matrix_count(), 3);

    s32 source_node = 0, target_node = 1;
    f32 default_value = weight_matrix.get(source_node, target_node);
    f32 value_a = weight_matrix.get_for_matrix(source_node, target_node, matrix_a);
    f32 value_b = weight_matrix.get_for_matrix(source_node, target_node, matrix_b);

    // Hand-computed expected values: U[0] and V[1]'s actual float4 components for
    // this exact fixture (rank=3, weight_seed=7, square_torus(4)) were read out and
    // the Σ U[0,r]·Ck[r]·V[1,r] sums computed externally (rank=3 -> rank_float4_stride=1,
    // so the padding lane -- the 4th component, un-specified by coefficients_a/b above
    // and padded to 1.0f by add_coefficient_vector -- still contributes u.w*v.w in
    // full, same as it does for DEFAULT_MATRIX_INDEX):
    //   U[0] = (1.06929338, -0.69152844, -0.0486776829, 0.377959013)
    //   V[1] = (-0.014391955, 3.40045786, 0.38932386, 0.801111579)
    //   default (Ck=1):        Σ U[0,r]·V[1,r]         = -2.0830665831
    //   matrix_a (Ck={2,0.5,-1}): Σ U[0,r]·Ck[r]·V[1,r] = -0.8847963789
    //   matrix_b (Ck={-3,1,4}):   Σ U[0,r]·Ck[r]·V[1,r] = -2.0783638445
    EXPECT_TRUE(approx(default_value, -2.0830665831f, 1e-4f));
    EXPECT_TRUE(approx(value_a, -0.8847963789f, 1e-4f));
    EXPECT_TRUE(approx(value_b, -2.0783638445f, 1e-4f));

    // Distinct, non-trivial Ck vectors sharing one basis must reconstruct to
    // distinct values (proving the basis is genuinely shared and Ck genuinely
    // differentiates), and each must differ from the default (Ck=1) value too.
    EXPECT_FALSE(approx(value_a, value_b, 1e-6f));
    EXPECT_FALSE(approx(value_a, default_value, 1e-6f));
    EXPECT_FALSE(approx(value_b, default_value, 1e-6f));

    // neighbor_weights_for_matrix must agree with get_for_matrix at real edges.
    vector<f32> weights_a((usize)(weight_matrix.node_count * weight_matrix.max_neighbor_count));
    weight_matrix.neighbor_weights_for_matrix(weights_a.data(), matrix_a);
    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    for (s64 node = 0; node < weight_matrix.node_count; ++node) {
        s64 degree = weight_matrix.get_neighbors(node, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            f32 gpu_value = weights_a[(usize)(node * weight_matrix.max_neighbor_count + slot)];
            f32 cpu_value = weight_matrix.get_for_matrix((s32)node, neighbor_buffer[(usize)slot], matrix_a);
            EXPECT_TRUE(approx(gpu_value, cpu_value, 1e-5f));
        }
    }

    // set_coefficient_vector overwrites a registered matrix's Ck in place.
    vector<f32> coefficients_a_updated = {1.0f, 1.0f, 1.0f};
    weight_matrix.set_coefficient_vector(matrix_a, coefficients_a_updated);
    EXPECT_TRUE(approx(weight_matrix.get_for_matrix(source_node, target_node, matrix_a), default_value, 1e-5f));
}

TEST(WeightMatrix, add_coefficient_vector_rejects_wrong_length) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/4);
    EXPECT_THROW(weight_matrix.add_coefficient_vector({1.0f, 2.0f}), std::invalid_argument);
    EXPECT_THROW(weight_matrix.set_coefficient_vector(WeightMatrix::DEFAULT_MATRIX_INDEX, {1.0f, 2.0f}),
                 std::invalid_argument);
}

TEST(WeightMatrix, matrix_index_out_of_range_throws) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/4);
    EXPECT_THROW((void)weight_matrix.get_for_matrix(0, 1, /*matrix_index=*/1), std::invalid_argument);
    EXPECT_THROW(weight_matrix.set_coefficient_vector(/*matrix_index=*/5, {1.0f, 1.0f, 1.0f, 1.0f}),
                 std::invalid_argument);
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

TEST(WeightMatrix, construction_rejects_self_loop_even_mixed_with_normal_edges) {
    // Self-loops are never supported -- not silently accepted, not silently
    // dropped -- even when the self-loop is just one edge among otherwise normal
    // edges in a larger network (see K2Tree's self-loop validation in
    // build_tree_arrays, which every WeightMatrix construction funnels through).
    vector<vector<s32>> network_with_self_loop = {
        {1, 2},
        {1, 2}, // node 1: self-loop + normal edge
        {0}
    };
    EXPECT_THROW({ WeightMatrix weight_matrix(network_with_self_loop, /*rank=*/4); }, std::invalid_argument);

    // The same network with the self-loop removed constructs normally.
    vector<vector<s32>> network_without_self_loop = {
        {1, 2},
        {2}, // node 1: self-loop removed, normal edge kept
        {0}
    };
    WeightMatrix weight_matrix(network_without_self_loop, /*rank=*/4);
    EXPECT_FALSE(weight_matrix.k2tree.adjacent(1, 1));
    EXPECT_TRUE(weight_matrix.k2tree.adjacent(1, 2));
    vector<s32> buffer(2);
    s64 degree = weight_matrix.get_neighbors(1, buffer.data());
    unordered_set<s32> neighbors(buffer.begin(), buffer.begin() + degree);
    EXPECT_EQ(neighbors, (unordered_set<s32>{2}));
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

// ── set_constant_weight: exact reconstruction regardless of rank%4 ────────────
//
// Fixed alongside ticket SC-52/D2 (discovered by the test-hardening pass): the
// scale factor used to be derived from the logical `rank` (sqrtf(|value| / rank)),
// but the fill loop writes that scale into all rank_float4_stride * 4 lanes, and
// get()/neighbor_weights() sum over every one of those lanes. When rank was a
// multiple of 4, rank_float4_stride * 4 == rank and everything canceled out
// correctly; otherwise (e.g. rank=3) the extra padding lane(s) still contributed
// full signal to the dot product, and get() overshot value * (rank_float4_stride *
// 4) / rank instead of value (e.g. rank=3 inflated 0.42 to ~0.56). The scale
// factor is now derived from rank_float4_stride * 4 (the true lane count actually
// filled/summed), so this holds regardless of rank % 4.
TEST(WeightMatrix, constant_weight_exact_regardless_of_rank_padding) {
    auto network = square_torus(3); // 9 nodes

    WeightMatrix rank_multiple_of_four(network, /*rank=*/4); // rank_float4_stride=1, no padding
    rank_multiple_of_four.set_constant_weight(0.42f);
    EXPECT_TRUE(approx(rank_multiple_of_four.get(0, 1), 0.42f, 1e-5f));
    EXPECT_TRUE(approx(rank_multiple_of_four.get(5, 8), 0.42f, 1e-5f));

    WeightMatrix rank_not_multiple_of_four(network, /*rank=*/3); // rank_float4_stride=1, 1 padding lane
    rank_not_multiple_of_four.set_constant_weight(0.42f);
    EXPECT_TRUE(approx(rank_not_multiple_of_four.get(0, 1), 0.42f, 1e-5f));
    EXPECT_TRUE(approx(rank_not_multiple_of_four.get(5, 8), 0.42f, 1e-5f));
}

// ── save/load across a dimension change (the reallocation branch of load_from_disk) ──

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

TEST(WeightMatrix, move_construction_preserves_state) {
    auto network = square_torus(4);
    WeightMatrix original(network, /*rank=*/8);
    original.set_constant_weight(0.6f);
    f32 expected = original.get(2, 5);

    WeightMatrix moved(std::move(original));
    EXPECT_TRUE(approx(moved.get(2, 5), expected));
    EXPECT_EQ(moved.node_count, 16);
}

// Fixed alongside ticket SC-52/D2 (discovered by the test-hardening pass):
// WeightMatrix's move-ASSIGNMENT operator used to be `= default`, which move-
// assigned each GpuPointer member via GpuPointer::operator=(GpuPointer&&) --
// asserting the destination pointer is null, which is never true for a live
// WeightMatrix (its default constructor is deleted) and aborted the whole
// process. WeightMatrix now has an explicit move-assignment operator that
// deallocates its own existing GPU buffers first.
TEST(WeightMatrix, move_assignment_into_a_live_object_transfers_state_without_aborting) {
    auto network = square_torus(4);
    WeightMatrix source(network, /*rank=*/8);
    source.set_constant_weight(0.6f);
    f32 expected = source.get(2, 5);

    WeightMatrix destination(network, /*rank=*/8);
    destination.set_constant_weight(-0.25f); // distinct pre-assignment state

    destination = std::move(source); // must not abort
    EXPECT_TRUE(approx(destination.get(2, 5), expected));
    EXPECT_EQ(destination.node_count, 16);
}

// ── sparse delta buffer / Sk (ticket #53/D3) ───────────────────────────────────

// A freshly-constructed WeightMatrix's Sk is untouched for every matrix, and
// lookup_sparse_delta's untouched fast path performs no slot search at all -- so
// get()/neighbor_weights() must stay bit-for-bit identical to the exact golden
// hex patterns ticket #52/D2 pinned before Sk existed at all (see
// get_reproduces_pre_shared_basis_values_bit_for_bit /
// neighbor_weights_reproduces_pre_shared_basis_values_bit_for_bit above).
TEST(WeightMatrix, get_and_neighbor_weights_bit_identical_when_sparse_delta_buffer_untouched) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8, true, -1, /*weight_seed=*/42);
    EXPECT_TRUE(bits_equal(weight_matrix.get(0, 1), from_bits(0x3e011f04u)));
    EXPECT_TRUE(bits_equal(weight_matrix.get(3, 12), from_bits(0xc0880ba9u)));

    vector<f32> weights((usize)(weight_matrix.node_count * weight_matrix.max_neighbor_count));
    weight_matrix.neighbor_weights(weights.data());
    EXPECT_TRUE(bits_equal(weights[0], from_bits(0x3e011f04u)));
    EXPECT_TRUE(bits_equal(weights[5], from_bits(0xc02627beu)));
}

// Whitebox check of the redesigned Sk storage (ticket #53/D3 rework, replacing
// the original per-matrix hash map with a GPU-resident, fixed-capacity array):
// one GpuPointer<f32> per matrix, sized exactly node_count*max_neighbor_count,
// laid out row-major by source node with position = that neighbor's slot in
// k2tree.get_neighbors(source_node, ...) enumeration order -- the SAME shape
// neighbor_weights()'s own output array already uses. An untouched matrix's
// array is all-zero and sparse_delta_touched reports false; accumulate_edge_delta
// writes exactly one slot and flips sparse_delta_touched to true, leaving every
// other slot exactly 0.0f.
TEST(WeightMatrix, sparse_delta_buffer_uses_position_indexed_gpu_resident_layout) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8, /*check_indexing=*/true, -1, /*weight_seed=*/5);

    ASSERT_EQ(weight_matrix.sparse_delta_buffers.size(), (usize)weight_matrix.matrix_count());
    ASSERT_EQ(weight_matrix.sparse_delta_touched.size(), (usize)weight_matrix.matrix_count());
    EXPECT_FALSE(weight_matrix.sparse_delta_touched[(usize)WeightMatrix::DEFAULT_MATRIX_INDEX]);

    s64 total_slots = weight_matrix.node_count * weight_matrix.max_neighbor_count;
    const f32 *delta_data =
        weight_matrix.sparse_delta_buffers[(usize)WeightMatrix::DEFAULT_MATRIX_INDEX].get_contents();
    for (s64 index = 0; index < total_slots; ++index) {
        EXPECT_EQ(delta_data[index], 0.0f);
    }

    s32 source_node = 0, target_node = 1;
    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    s64 degree = weight_matrix.get_neighbors(source_node, neighbor_buffer.data());
    s64 expected_slot = -1;
    for (s64 slot = 0; slot < degree; ++slot) {
        if (neighbor_buffer[(usize)slot] == target_node) {
            expected_slot = slot;
            break;
        }
    }
    ASSERT_NE(expected_slot, -1);

    weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, source_node, target_node, 4.0f);
    EXPECT_TRUE(weight_matrix.sparse_delta_touched[(usize)WeightMatrix::DEFAULT_MATRIX_INDEX]);

    s64 expected_index = source_node * weight_matrix.max_neighbor_count + expected_slot;
    for (s64 index = 0; index < total_slots; ++index) {
        if (index == expected_index) {
            EXPECT_EQ(delta_data[index], 4.0f);
        } else {
            EXPECT_EQ(delta_data[index], 0.0f);
        }
    }
}

// Round-trip proof (ticket #53/D3 acceptance criterion): accumulate_edge_delta
// then a read of the same edge returns exactly (the pre-update value) + delta.
TEST(WeightMatrix, accumulate_edge_delta_round_trips_a_single_bump) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8, /*check_indexing=*/true, -1, /*weight_seed=*/7);
    s32 source_node = 0, target_node = 1;
    ASSERT_TRUE(weight_matrix.k2tree.adjacent(source_node, target_node));

    f32 before = weight_matrix.get(source_node, target_node);
    weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, source_node, target_node, 5.0f);
    f32 after = weight_matrix.get(source_node, target_node);

    EXPECT_TRUE(bits_equal(after, before + 5.0f));
}

// Three separate accedge-style bumps to the same edge must sum correctly, in
// the same left-to-right accumulation order Sk[i,j] += x uses internally.
TEST(WeightMatrix, accumulate_edge_delta_round_trips_multiple_bumps_summing) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8, /*check_indexing=*/true, -1, /*weight_seed=*/13);
    s32 source_node = 2, target_node = 3;
    ASSERT_TRUE(weight_matrix.k2tree.adjacent(source_node, target_node));

    f32 before = weight_matrix.get(source_node, target_node);
    weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, source_node, target_node, 2.0f);
    weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, source_node, target_node, -0.5f);
    weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, source_node, target_node, 3.25f);
    f32 after = weight_matrix.get(source_node, target_node);

    f32 expected_delta = 0.0f;
    expected_delta += 2.0f;
    expected_delta += -0.5f;
    expected_delta += 3.25f;
    EXPECT_TRUE(bits_equal(after, before + expected_delta));
}

// Bumping matrix A's Sk at an edge must not affect matrix B's (or
// DEFAULT_MATRIX_INDEX's) read at the same edge -- each matrix's Sk is its own
// array, never jumbled together (the ticket body's explicit requirement).
TEST(WeightMatrix, accumulate_edge_delta_does_not_mix_between_matrices) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/3, /*check_indexing=*/true, -1, /*weight_seed=*/7);
    vector<f32> coefficients_a = {2.0f, 0.5f, -1.0f};
    vector<f32> coefficients_b = {-3.0f, 1.0f, 4.0f};
    s64 matrix_a = weight_matrix.add_coefficient_vector(coefficients_a);
    s64 matrix_b = weight_matrix.add_coefficient_vector(coefficients_b);

    s32 source_node = 0, target_node = 1;
    ASSERT_TRUE(weight_matrix.k2tree.adjacent(source_node, target_node));

    f32 default_before = weight_matrix.get(source_node, target_node);
    f32 matrix_a_before = weight_matrix.get_for_matrix(source_node, target_node, matrix_a);
    f32 matrix_b_before = weight_matrix.get_for_matrix(source_node, target_node, matrix_b);

    weight_matrix.accumulate_edge_delta(matrix_a, source_node, target_node, 10.0f);

    EXPECT_TRUE(bits_equal(weight_matrix.get_for_matrix(source_node, target_node, matrix_a),
                            matrix_a_before + 10.0f));
    // matrix_b and DEFAULT_MATRIX_INDEX share the same basis/edge but not matrix
    // A's Sk -- their reads must be completely unaffected.
    EXPECT_TRUE(bits_equal(weight_matrix.get_for_matrix(source_node, target_node, matrix_b), matrix_b_before));
    EXPECT_TRUE(bits_equal(weight_matrix.get(source_node, target_node), default_before));
}

// loadedge/accedge are edge-scoped IR ops (IR spec §3.3) -- accumulate_edge_delta
// must reject a bump to a pair that is not a real k^2-tree edge, and must not
// mutate any state when it does (unlike get()/update(), which are deliberately
// edge-unrestricted, bounds-only).
TEST(WeightMatrix, accumulate_edge_delta_rejects_non_edge_and_does_not_mutate_state) {
    auto network = make_irregular_network();
    WeightMatrix weight_matrix(network, /*rank=*/4);
    ASSERT_FALSE(weight_matrix.k2tree.adjacent(3, 0)); // node 3 has no outgoing edges at all

    f32 before = weight_matrix.get(3, 0);
    EXPECT_THROW(weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, 3, 0, 100.0f),
                 std::invalid_argument);
    f32 after = weight_matrix.get(3, 0);
    EXPECT_TRUE(bits_equal(before, after));
}

TEST(WeightMatrix, accumulate_edge_delta_rejects_out_of_bounds_indices) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);
    EXPECT_THROW(weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, -1, 1, 1.0f),
                 std::invalid_argument);
    EXPECT_THROW(weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, 0, 4590, 1.0f),
                 std::invalid_argument);
}

TEST(WeightMatrix, accumulate_edge_delta_rejects_bad_matrix_index) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/4);
    EXPECT_THROW(weight_matrix.accumulate_edge_delta(/*matrix_index=*/5, 0, 1, 1.0f), std::invalid_argument);
}

// New failure mode introduced by the position-indexed array redesign (ticket
// #53/D3 rework): an explicit max_neighbor_count override smaller than a node's
// true degree (already-supported truncation -- see
// max_neighbor_count_explicit_override_truncates) means some of that node's real
// edges are not enumerable within the truncated neighbor list at all, even
// though k2tree.adjacent still reports them as genuine edges. Those edges have
// no representable array slot, so accumulate_edge_delta must reject them rather
// than write out of bounds.
TEST(WeightMatrix, accumulate_edge_delta_rejects_real_edge_not_representable_within_truncated_max_neighbor_count) {
    auto network = make_irregular_network();
    // node 2 has real out-degree 4 ({0,1,3,4}); capping max_neighbor_count at 2
    // makes only the first 2 (in k2tree traversal order) representable.
    WeightMatrix weight_matrix(network, /*rank=*/4, /*check_indexing=*/true,
                               /*max_neighbor_count=*/2);
    ASSERT_EQ(weight_matrix.max_neighbor_count, 2);

    vector<s32> buffer(2);
    s64 degree = weight_matrix.get_neighbors(2, buffer.data());
    ASSERT_EQ(degree, 2);
    unordered_set<s32> representable(buffer.begin(), buffer.begin() + degree);

    s32 unrepresentable_neighbor = -1;
    for (s32 candidate : {0, 1, 3, 4}) {
        if (representable.find(candidate) == representable.end()) {
            unrepresentable_neighbor = candidate;
            break;
        }
    }
    ASSERT_NE(unrepresentable_neighbor, -1);
    ASSERT_TRUE(weight_matrix.k2tree.adjacent(2, unrepresentable_neighbor)); // genuinely a real edge

    EXPECT_THROW(
        weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, 2, unrepresentable_neighbor, 1.0f),
        std::invalid_argument);
}

// neighbor_weights()'s batched output must reflect Sk exactly the way
// get()/get_for_matrix()'s single-entry reads do, for every real edge -- not
// just at the one edge under test above.
TEST(WeightMatrix, neighbor_weights_reflects_sparse_delta_consistently_with_get) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8, true, -1, /*weight_seed=*/11);

    weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, 0, 1, 3.0f);
    weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, 2, 3, -1.5f);
    weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, 5, 9, 0.25f);

    vector<f32> weights((usize)(weight_matrix.node_count * weight_matrix.max_neighbor_count));
    weight_matrix.neighbor_weights(weights.data());

    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    for (s64 node = 0; node < weight_matrix.node_count; ++node) {
        s64 degree = weight_matrix.get_neighbors(node, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            f32 batch_value = weights[(usize)(node * weight_matrix.max_neighbor_count + slot)];
            f32 single_value = weight_matrix.get((s32)node, neighbor_buffer[(usize)slot]);
            EXPECT_TRUE(approx(batch_value, single_value, 1e-4f))
                << "node=" << node << " neighbor=" << neighbor_buffer[(usize)slot];
        }
    }
}

// Same consistency check via neighbor_weights_for_matrix()/get_for_matrix() on
// a non-default matrix, so the overlay path is proven for both entry points.
TEST(WeightMatrix, neighbor_weights_for_matrix_reflects_sparse_delta_consistently_with_get_for_matrix) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/6, true, -1, /*weight_seed=*/23);
    s64 matrix_index = weight_matrix.add_coefficient_vector({1.5f, -0.5f, 2.0f, 0.0f, 1.0f, -1.0f});

    weight_matrix.accumulate_edge_delta(matrix_index, 1, 2, 4.0f);
    weight_matrix.accumulate_edge_delta(matrix_index, 7, 6, -2.25f);

    vector<f32> weights((usize)(weight_matrix.node_count * weight_matrix.max_neighbor_count));
    weight_matrix.neighbor_weights_for_matrix(weights.data(), matrix_index);

    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    for (s64 node = 0; node < weight_matrix.node_count; ++node) {
        s64 degree = weight_matrix.get_neighbors(node, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            f32 batch_value = weights[(usize)(node * weight_matrix.max_neighbor_count + slot)];
            f32 single_value = weight_matrix.get_for_matrix((s32)node, neighbor_buffer[(usize)slot], matrix_index);
            EXPECT_TRUE(approx(batch_value, single_value, 1e-4f))
                << "node=" << node << " neighbor=" << neighbor_buffer[(usize)slot];
        }
    }
}

TEST(WeightMatrix, move_construction_preserves_sparse_delta_buffer) {
    auto network = square_torus(4);
    WeightMatrix original(network, /*rank=*/8, true, -1, /*weight_seed=*/3);
    original.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, 0, 1, 2.5f);
    f32 expected = original.get(0, 1);

    WeightMatrix moved(std::move(original));
    EXPECT_TRUE(bits_equal(moved.get(0, 1), expected));
}

TEST(WeightMatrix, move_assignment_preserves_sparse_delta_buffer) {
    auto network = square_torus(4);
    WeightMatrix source(network, /*rank=*/8, true, -1, /*weight_seed=*/3);
    source.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, 0, 1, 2.5f);
    f32 expected = source.get(0, 1);

    WeightMatrix destination(network, /*rank=*/8);
    destination = std::move(source);
    EXPECT_TRUE(bits_equal(destination.get(0, 1), expected));
}

// save()/load_from_disk() deliberately do not persist Sk (see the design
// comment on save() in weight_matrix.cpp). The non-reallocating branch (same
// node_count/rank_float4_stride) must leave the loading instance's own Sk
// untouched; the reallocating branch must reset it back to a single empty
// DEFAULT_MATRIX_INDEX slot, in lockstep with coefficient_vectors.
TEST(WeightMatrix, load_from_disk_does_not_persist_sparse_delta_buffer) {
    auto network = square_torus(4);
    WeightMatrix source(network, /*rank=*/8);
    source.set_constant_weight(0.42f);
    source.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, 0, 1, 5.0f); // never persisted

    const char *path = "/tmp/spikecorec_test_wm_sparse_delta.bin";
    source.save(path);

    // Non-reallocating load: same node_count/rank_float4_stride as `source`.
    WeightMatrix destination(network, /*rank=*/8);
    destination.load_from_disk(path);
    // Only U/V (0.42 constant weight) came across -- source's +5.0 Sk bump did not.
    EXPECT_TRUE(approx(destination.get(0, 1), 0.42f));
}

TEST(WeightMatrix, load_from_disk_resets_sparse_delta_buffer_on_reallocation) {
    auto small_network = square_torus(3); // 9 nodes, rank_float4_stride=1
    WeightMatrix source(small_network, /*rank=*/4);
    source.set_constant_weight(0.42f);
    source.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, 0, 1, 5.0f); // never persisted

    const char *path = "/tmp/spikecorec_test_wm_sparse_delta_realloc.bin";
    source.save(path);

    auto large_network = square_torus(4); // 16 nodes, rank_float4_stride=3 -- forces reallocation
    WeightMatrix destination(large_network, /*rank=*/12);
    ASSERT_TRUE(destination.k2tree.adjacent(0, 1));
    s64 matrix_b = destination.add_coefficient_vector(vector<f32>(12, 1.0f));
    destination.accumulate_edge_delta(matrix_b, 0, 1, 3.0f); // pre-load Sk, must be wiped by reallocation

    destination.load_from_disk(path);
    EXPECT_EQ(destination.matrix_count(), 1); // reset back to just DEFAULT_MATRIX_INDEX
    EXPECT_TRUE(approx(destination.get(0, 1), 0.42f)); // pure reconstruction, no leftover Sk
}

// ── per-edge delay storage (ticket #64/F3's future consumer) ──────────────────
//
// This stage only adds storage + setters; nothing reads edge_delay_ticks/
// constant_delay_ticks yet (a future stage wires that in) -- see the header
// comment on using_constant_delay_ticks.

TEST(WeightMatrix, constant_delay_ticks_defaults_to_one_tick) {
    // A "no explicit delay configured" WeightMatrix must behave byte-for-byte
    // like today's undelayed engine -- the engine's existing implicit one-tick
    // network_inputs latency.
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);
    EXPECT_TRUE(weight_matrix.using_constant_delay_ticks);
    EXPECT_EQ(weight_matrix.constant_delay_ticks, 1);
}

TEST(WeightMatrix, set_constant_delay_ticks_updates_value_and_flag) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);

    weight_matrix.set_constant_delay_ticks(5);
    EXPECT_TRUE(weight_matrix.using_constant_delay_ticks);
    EXPECT_EQ(weight_matrix.constant_delay_ticks, 5);
}

TEST(WeightMatrix, set_constant_delay_ticks_rejects_non_positive_values) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);

    EXPECT_THROW(weight_matrix.set_constant_delay_ticks(0), std::invalid_argument);
    EXPECT_THROW(weight_matrix.set_constant_delay_ticks(-1), std::invalid_argument);
}

// Every real edge must default to a delay of 1 tick (the default single-tick
// delay), before any set_edge_delay_ticks call -- checked across a real
// multi-edge, irregular-degree adjacency, not just one edge.
//
// Rewritten from nightly's own whitebox version, which read a flat
// [node_count * max_neighbor_count] `edge_delay_ticks` array directly. Per-edge
// delay now lives as a registered, all-zero-Ck matrix in the shared-basis
// family (see delay_matrix_index in weight_matrix.h) rather than as its own
// dense array, so the same invariant is asserted through the
// get_edge_delay_ticks accessor instead.
TEST(WeightMatrix, edge_delay_ticks_defaults_to_one_for_every_edge) {
    auto network = make_irregular_network();
    WeightMatrix weight_matrix(network, /*rank=*/4);

    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    for (s64 node = 0; node < weight_matrix.node_count; ++node) {
        s64 degree = weight_matrix.get_neighbors(node, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            EXPECT_EQ(weight_matrix.get_edge_delay_ticks((s32)node, neighbor_buffer[(usize)slot]), 1)
                << "node=" << node << " slot=" << slot;
        }
    }
}

// Same rewrite as above: "only the one edge moved" is asserted by reading every
// real edge back through get_edge_delay_ticks rather than by scanning a flat
// per-edge array.
TEST(WeightMatrix, set_edge_delay_ticks_updates_the_correct_slot) {
    auto network = make_irregular_network();
    WeightMatrix weight_matrix(network, /*rank=*/4);

    const s32 source_node = 2, target_node = 3; // node 2: out-degree 4, {0,1,3,4}
    ASSERT_TRUE(weight_matrix.k2tree.adjacent(source_node, target_node));

    weight_matrix.set_edge_delay_ticks(source_node, target_node, 7);

    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    for (s64 node = 0; node < weight_matrix.node_count; ++node) {
        s64 degree = weight_matrix.get_neighbors(node, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            s32 neighbor = neighbor_buffer[(usize)slot];
            s32 expected_delay = (node == (s64)source_node && neighbor == target_node) ? 7 : 1;
            EXPECT_EQ(weight_matrix.get_edge_delay_ticks((s32)node, neighbor), expected_delay)
                << "node=" << node << " neighbor=" << neighbor;
        }
    }
}

TEST(WeightMatrix, set_edge_delay_ticks_rejects_non_edge) {
    auto network = make_irregular_network();
    WeightMatrix weight_matrix(network, /*rank=*/4);
    ASSERT_FALSE(weight_matrix.k2tree.adjacent(3, 0)); // node 3 has no outgoing edges at all

    EXPECT_THROW(weight_matrix.set_edge_delay_ticks(3, 0, 2), std::invalid_argument);
}

TEST(WeightMatrix, set_edge_delay_ticks_rejects_non_positive_delay) {
    auto network = make_irregular_network();
    WeightMatrix weight_matrix(network, /*rank=*/4);
    s32 source_node = 0, target_node = 1;
    ASSERT_TRUE(weight_matrix.k2tree.adjacent(source_node, target_node));

    EXPECT_THROW(weight_matrix.set_edge_delay_ticks(source_node, target_node, 0), std::invalid_argument);
}

// ── per-edge delay as a registered shared-basis matrix ────────────────────────
//
// Delay is stored as one more matrix in the Ck/Sk family, registered lazily and
// pinned to an all-zero Ck, rather than as a flat per-edge array — these cover
// that storage's own invariants (lazy registration, exact read-back, survival
// across refit).

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

// ── total_edge_count (ticket #54/D4) ──────────────────────────────────────────

TEST(WeightMatrix, total_edge_count_matches_the_network) {
    auto network = square_torus(4); // 16 nodes, out-degree 4 each
    WeightMatrix weight_matrix(network, /*rank=*/4);
    EXPECT_EQ(weight_matrix.total_edge_count, 16 * 4);
}

TEST(WeightMatrix, total_edge_count_respects_max_neighbor_count_truncation) {
    // node 2's real out-degree is 4; capping max_neighbor_count at 2 truncates
    // it to 2 for get_neighbors() -- total_edge_count must agree (it is
    // computed via the same get_neighbors() walk, not a raw sum of the
    // adjacency list's row lengths -- see the constructor's comment).
    auto network = make_irregular_network();
    WeightMatrix truncated(network, /*rank=*/4, /*check_indexing=*/true, /*max_neighbor_count=*/2);
    // node0: min(3,2)=2, node1: min(1,2)=1, node2: min(4,2)=2, node3: 0, node4: min(1,2)=1.
    EXPECT_EQ(truncated.total_edge_count, 2 + 1 + 2 + 0 + 1);
}

// ── periodic refit (ticket #54/D4) ────────────────────────────────────────────
//
// D4 math memo §6 is explicit that refit is an approximation-quality claim,
// not an exactness claim (unlike ticket #52's Ck=1 bit-exactness bar) -- these
// tests check relative fit quality / near-no-op behavior against tolerances,
// not bit-for-bit reproduction.

TEST(WeightMatrix, refit_recovers_a_known_low_rank_fixture_within_tolerance) {
    // rank=4 -> rank_float4_stride=1 -> effective_lane_count (the padded
    // dimensionality reconstruct_entry actually sums over) equals the logical
    // rank exactly, so this fixture's "true" rows/coefficients are exactly
    // what refit() fits against -- no float4 padding lanes to complicate it.
    //
    // A single matrix (DEFAULT_MATRIX_INDEX) is used here -- deliberately, not
    // an oversight -- to isolate "does the ALS implementation converge to a
    // known rank-4 factorization" from any additional joint-convergence
    // difficulty a cold-started MULTI-matrix coupled fit can introduce (see
    // refit_couples_matrices_through_the_shared_basis below for the
    // multi-matrix coupling behavior itself, which is a qualitative check, not
    // a quantitative recovery one). This also matches the ticket body's own
    // phrasing: "pick a 'true' U*, V*, Ck*" -- singular Ck.
    auto network = random_fixed_outdegree(/*side_length=*/4, /*fanout=*/8, /*seed=*/123); // 16 nodes, out-degree 8
    const s64 rank = 4;
    WeightMatrix weight_matrix(network, rank, /*check_indexing=*/true, -1, /*weight_seed=*/7);

    vector<f32> coefficients_default = deterministic_row(0, rank, /*row_seed=*/100);
    weight_matrix.set_coefficient_vector(WeightMatrix::DEFAULT_MATRIX_INDEX, coefficients_default);

    vector<vector<f32>> true_rows((usize)weight_matrix.node_count);
    vector<vector<f32>> true_columns((usize)weight_matrix.node_count);
    for (s64 node_index = 0; node_index < weight_matrix.node_count; ++node_index) {
        true_rows[(usize)node_index] = deterministic_row(node_index, rank, /*row_seed=*/node_index * 3 + 1);
        true_columns[(usize)node_index] = deterministic_row(node_index, rank, /*row_seed=*/node_index * 5 + 2);
    }

    // Inject the fixture's "true" values as an Sk delta (true - current
    // reconstruction), so every get_for_matrix() read returns exactly the true
    // value pre-refit -- this is a one-shot full-signal injection (unlike a
    // typical small warm-start perturbation between refits in production
    // usage), so a larger-than-default sweep_count is used below for THIS
    // test to let ALS actually converge from what is effectively a cold start
    // relative to the injected signal (D4 math memo §3/§5 explicitly support
    // tuning sweep_count up for atypically large accumulated drift).
    for (s64 source_node = 0; source_node < weight_matrix.node_count; ++source_node) {
        for (s32 target_node : network[(usize)source_node]) {
            f32 true_value = low_rank_dot(true_rows[(usize)source_node], coefficients_default, true_columns[(usize)target_node]);
            f32 current_value = weight_matrix.get_for_matrix((s32)source_node, target_node, WeightMatrix::DEFAULT_MATRIX_INDEX);
            weight_matrix.accumulate_edge_delta(
                WeightMatrix::DEFAULT_MATRIX_INDEX, (s32)source_node, target_node, true_value - current_value);
        }
    }

    weight_matrix.refit(/*sweep_count=*/80, /*ridge_regularization=*/1e-4f);

    f64 squared_error_sum = 0.0;
    f64 squared_true_sum = 0.0;
    s64 edge_count = 0;
    for (s64 source_node = 0; source_node < weight_matrix.node_count; ++source_node) {
        for (s32 target_node : network[(usize)source_node]) {
            f32 true_value = low_rank_dot(true_rows[(usize)source_node], coefficients_default, true_columns[(usize)target_node]);
            f32 reconstructed = weight_matrix.get_for_matrix((s32)source_node, target_node, WeightMatrix::DEFAULT_MATRIX_INDEX);
            f64 error = (f64)reconstructed - (f64)true_value;
            squared_error_sum += error * error;
            squared_true_sum += (f64)true_value * (f64)true_value;
            ++edge_count;
        }
    }
    f64 true_rms = std::sqrt(squared_true_sum / (f64)edge_count);
    f64 relative_rms_error = std::sqrt(squared_error_sum / (f64)edge_count) / (true_rms > 1e-6 ? true_rms : 1e-6);
    EXPECT_LT(relative_rms_error, 0.05);
}

TEST(WeightMatrix, refit_clears_sparse_delta_buffer_for_every_matrix) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/4, /*check_indexing=*/true, -1, /*weight_seed=*/5);
    s64 matrix_a = weight_matrix.add_coefficient_vector({1.0f, 2.0f, -1.0f, 0.5f});

    weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, 0, 1, 3.0f);
    weight_matrix.accumulate_edge_delta(matrix_a, 2, 3, -2.0f);
    ASSERT_FALSE(sparse_delta_buffer_contents_are_all_zero(weight_matrix, WeightMatrix::DEFAULT_MATRIX_INDEX));
    ASSERT_FALSE(sparse_delta_buffer_contents_are_all_zero(weight_matrix, matrix_a));

    weight_matrix.refit();

    for (s64 matrix_index = 0; matrix_index < weight_matrix.matrix_count(); ++matrix_index) {
        EXPECT_TRUE(sparse_delta_buffer_contents_are_all_zero(weight_matrix, matrix_index));
        EXPECT_FALSE(weight_matrix.sparse_delta_touched[(usize)matrix_index]);
    }
}

TEST(WeightMatrix, refit_is_a_near_no_op_on_an_untouched_sparse_delta_buffer) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/4, /*check_indexing=*/true, -1, /*weight_seed=*/9);
    s64 matrix_a = weight_matrix.add_coefficient_vector({1.0f, -0.5f, 2.0f, 0.25f});

    vector<pair<s32, s32>> probe_edges = {{0, 1}, {5, 9}, {12, 8}};
    for (const auto &probe_edge : probe_edges) {
        ASSERT_TRUE(weight_matrix.k2tree.adjacent(probe_edge.first, probe_edge.second));
    }

    vector<f32> default_before, matrix_a_before;
    for (const auto &probe_edge : probe_edges) {
        default_before.push_back(weight_matrix.get_for_matrix(probe_edge.first, probe_edge.second, WeightMatrix::DEFAULT_MATRIX_INDEX));
        matrix_a_before.push_back(weight_matrix.get_for_matrix(probe_edge.first, probe_edge.second, matrix_a));
    }

    // Sk is empty for every matrix here -- nothing has been accumulate_edge_delta'd
    // since construction. A single sweep from a point already at a stationary
    // point of the unregularized loss should leave U/V/Ck numerically close to
    // where they started (D4 math memo §6) -- the small ridge term perturbs
    // them slightly, but not "meaningfully."
    weight_matrix.refit();

    for (usize index = 0; index < probe_edges.size(); ++index) {
        f32 default_after = weight_matrix.get_for_matrix(probe_edges[index].first, probe_edges[index].second, WeightMatrix::DEFAULT_MATRIX_INDEX);
        f32 matrix_a_after = weight_matrix.get_for_matrix(probe_edges[index].first, probe_edges[index].second, matrix_a);
        EXPECT_TRUE(approx(default_after, default_before[index], 0.05f));
        EXPECT_TRUE(approx(matrix_a_after, matrix_a_before[index], 0.05f));
    }
}

TEST(WeightMatrix, refit_couples_matrices_through_the_shared_basis) {
    // Bumping only DEFAULT_MATRIX_INDEX's Sk and refitting legitimately
    // perturbs matrix_b's reconstruction too, by a small amount -- because U/V
    // are shared and refit against pooled data across the whole matrix family
    // (D4 math memo §6). This is EXPECTED behavior, not a bug -- a future
    // reader should not file this as a correctness defect.
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/4, /*check_indexing=*/true, -1, /*weight_seed=*/11);
    s64 matrix_b = weight_matrix.add_coefficient_vector({1.0f, -1.0f, 0.5f, 2.0f});

    s32 probe_source = 0, probe_target = 1;
    ASSERT_TRUE(weight_matrix.k2tree.adjacent(probe_source, probe_target));
    f32 matrix_b_before = weight_matrix.get_for_matrix(probe_source, probe_target, matrix_b);

    // Bump ONLY DEFAULT_MATRIX_INDEX's Sk, substantially, across every real edge.
    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    for (s64 node = 0; node < weight_matrix.node_count; ++node) {
        s64 degree = weight_matrix.get_neighbors(node, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX,
                (s32)node, neighbor_buffer[(usize)slot], 25.0f);
        }
    }
    ASSERT_TRUE(sparse_delta_buffer_contents_are_all_zero(weight_matrix, matrix_b)); // matrix_b itself untouched

    weight_matrix.refit();

    f32 matrix_b_after = weight_matrix.get_for_matrix(probe_source, probe_target, matrix_b);
    f32 change_magnitude = std::fabs(matrix_b_after - matrix_b_before);
    EXPECT_GT(change_magnitude, 1e-4f)
        << "matrix_b's reconstruction should move -- U/V are shared and refit pooled across "
        << "every matrix, even though matrix_b's own Sk was empty (expected, see comment above)";
    EXPECT_LT(change_magnitude, 12.5f)
        << "the coupling should be a small perturbation relative to the +25.0 bump directly "
        << "applied to the OTHER matrix, not a comparably large change";
}

TEST(WeightMatrix, refit_handles_degenerate_low_degree_nodes_without_producing_nan_or_inf) {
    // node 1: out-degree 1 (< rank); node 3: out-degree 0 (isolated for
    // outgoing edges, though still a target of node 0's edge) -- both exercise
    // the ridge term's necessity for a rank-deficient Gram matrix (D4 math
    // memo §2.2).
    auto network = make_irregular_network();
    WeightMatrix weight_matrix(network, /*rank=*/4, /*check_indexing=*/true, -1, /*weight_seed=*/13);
    s64 matrix_a = weight_matrix.add_coefficient_vector({1.5f, -2.0f, 0.5f, 3.0f});

    weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, 0, 1, 4.0f);
    weight_matrix.accumulate_edge_delta(matrix_a, 2, 3, -3.5f);

    weight_matrix.refit();

    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    for (s64 node = 0; node < weight_matrix.node_count; ++node) {
        s64 degree = weight_matrix.get_neighbors(node, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            f32 default_value = weight_matrix.get_for_matrix((s32)node, neighbor_buffer[(usize)slot], WeightMatrix::DEFAULT_MATRIX_INDEX);
            f32 matrix_a_value = weight_matrix.get_for_matrix((s32)node, neighbor_buffer[(usize)slot], matrix_a);
            EXPECT_TRUE(std::isfinite(default_value)) << "node=" << node;
            EXPECT_TRUE(std::isfinite(matrix_a_value)) << "node=" << node;
        }
    }

    for (s64 matrix_index = 0; matrix_index < weight_matrix.matrix_count(); ++matrix_index) {
        EXPECT_TRUE(sparse_delta_buffer_contents_are_all_zero(weight_matrix, matrix_index));
    }
}

TEST(WeightMatrix, refit_on_a_graph_with_no_edges_is_a_safe_no_op) {
    vector<vector<s32>> network(5); // 5 isolated nodes, no edges anywhere
    WeightMatrix weight_matrix(network, /*rank=*/4);
    EXPECT_EQ(weight_matrix.total_edge_count, 0);

    weight_matrix.advance_tick();
    weight_matrix.refit(); // must not crash (e.g. divide-by-zero building an empty point cloud)
    EXPECT_EQ(weight_matrix.ticks_since_last_refit, 0);
    for (s64 matrix_index = 0; matrix_index < weight_matrix.matrix_count(); ++matrix_index) {
        EXPECT_TRUE(sparse_delta_buffer_contents_are_all_zero(weight_matrix, matrix_index));
    }
}

TEST(WeightMatrix, refit_rejects_invalid_arguments) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/4);
    EXPECT_THROW(weight_matrix.refit(/*sweep_count=*/0), std::invalid_argument);
    EXPECT_THROW(weight_matrix.refit(/*sweep_count=*/-1), std::invalid_argument);
    EXPECT_THROW(weight_matrix.refit(/*sweep_count=*/2, /*ridge_regularization=*/0.0f), std::invalid_argument);
    EXPECT_THROW(weight_matrix.refit(/*sweep_count=*/2, /*ridge_regularization=*/-1e-4f), std::invalid_argument);
}

// ── refit-interval knob (ticket #54/D4) ───────────────────────────────────────

TEST(WeightMatrix, refit_interval_knob_default_and_tick_count_trigger) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/4);

    EXPECT_EQ(weight_matrix.refit_every_n_ticks, 200); // documented default (D4 math memo §5)
    EXPECT_FALSE(weight_matrix.is_refit_due());         // fresh construction: 0 ticks elapsed

    for (s64 tick = 0; tick < weight_matrix.refit_every_n_ticks - 1; ++tick) {
        weight_matrix.advance_tick();
    }
    EXPECT_FALSE(weight_matrix.is_refit_due());

    weight_matrix.advance_tick(); // reaches refit_every_n_ticks
    EXPECT_TRUE(weight_matrix.is_refit_due());

    weight_matrix.refit();
    EXPECT_FALSE(weight_matrix.is_refit_due()); // ticks_since_last_refit reset by refit()
}

TEST(WeightMatrix, refit_occupancy_threshold_is_disabled_by_default) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/4);
    EXPECT_LT(weight_matrix.refit_occupancy_threshold_fraction, 0.0f); // disabled sentinel

    // Fill Sk to full occupancy -- with the threshold disabled, is_refit_due()
    // must still only key off the tick-count trigger (D4 math memo §5: "default
    // this to disabled... unless the caller opts in").
    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    for (s64 node = 0; node < weight_matrix.node_count; ++node) {
        s64 degree = weight_matrix.get_neighbors(node, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX,
                (s32)node, neighbor_buffer[(usize)slot], 1.0f);
        }
    }
    EXPECT_TRUE(approx(weight_matrix.max_sparse_delta_occupancy_fraction(), 1.0f, 1e-3f));
    EXPECT_FALSE(weight_matrix.is_refit_due());
}

TEST(WeightMatrix, refit_occupancy_threshold_triggers_early_when_enabled) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/4);
    weight_matrix.refit_occupancy_threshold_fraction = 0.1f; // opt in

    EXPECT_FALSE(weight_matrix.is_refit_due()); // nothing accumulated yet, 0 ticks elapsed

    // Bump enough distinct edges to exceed 10% of total_edge_count.
    s64 target_bump_count = (s64)(weight_matrix.total_edge_count * 0.15) + 1;
    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    s64 bumped_count = 0;
    for (s64 node = 0; node < weight_matrix.node_count && bumped_count < target_bump_count; ++node) {
        s64 degree = weight_matrix.get_neighbors(node, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree && bumped_count < target_bump_count; ++slot) {
            weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX,
                (s32)node, neighbor_buffer[(usize)slot], 1.0f);
            ++bumped_count;
        }
    }

    EXPECT_TRUE(weight_matrix.is_refit_due()); // occupancy trigger fired, even with 0 ticks elapsed
}

// ── exact per-edge weights ────────────────────────────────────────────────────
//
// U and V are seeded from N(0,1), so an edge's low-rank reconstruction is of order 1
// whatever the model's weights are. Expressing a weight as a delta against that
// reconstruction is what these tests exist to rule out: for a realistic synaptic weight
// (NeuroML specifies conductances and currents at 1e-9 to 1e-12 in SI), `weight -
// reconstruction` rounds in f32 to `-reconstruction` exactly, and the weight reads back
// as 0. set_edge_weight pins the default matrix's low-rank term to zero and stores the
// value itself, so the round trip is exact at every magnitude.

namespace {

// Every real edge's current weight, in the row-major-by-source-node, get_neighbors slot
// order neighbor_weights()/sparse_delta_buffers share.
struct EdgeWeightSnapshot {
    vector<s32> source_nodes;
    vector<s32> target_nodes;
    vector<f32> values;
};

EdgeWeightSnapshot snapshot_every_edge_weight(const WeightMatrix &weight_matrix) {
    EdgeWeightSnapshot snapshot;
    vector<s32> neighbor_buffer((usize)max((s64)1, weight_matrix.max_neighbor_count));
    for (s64 node_index = 0; node_index < weight_matrix.node_count; ++node_index) {
        s64 degree = weight_matrix.get_neighbors(node_index, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            snapshot.source_nodes.push_back((s32)node_index);
            snapshot.target_nodes.push_back(neighbor_buffer[(usize)slot]);
            snapshot.values.push_back(
                weight_matrix.get((s32)node_index, neighbor_buffer[(usize)slot]));
        }
    }
    return snapshot;
}

} // namespace

TEST(WeightMatrix, set_edge_weight_round_trips_exactly_across_the_full_magnitude_range) {
    // 64 nodes, 4 out-edges each: big enough that the reconstruction at every edge below is
    // a real order-1 number rather than a degenerate near-zero.
    auto network = square_torus(8);
    WeightMatrix weight_matrix(network, /*rank=*/32, /*check_indexing=*/true,
                               /*max_neighbor_count=*/-1, /*weight_seed=*/42);

    const vector<f32> weights_to_store = {
        2.5e-8f,   // the weight this ticket started from: 2.5e-8 - 0.7 rounds to -0.7 in f32
        1e-12f, -4.5e-12f, 1e-9f, -3.75e-9f, 1.5e-6f, -2.5e-3f, 1.0f, -7.25f, 1e3f, 0.0f,
    };
    ASSERT_LE(weights_to_store.size(), (usize)weight_matrix.node_count);

    // The fixture only proves anything if the values being displaced are non-trivial.
    for (usize index = 0; index < weights_to_store.size(); ++index) {
        EXPECT_GT(std::fabs(weight_matrix.get((s32)index, network[index][0])), 1e-2f)
            << "index=" << index << ": the reservoir reconstruction at this edge is too "
               "small for the test to prove anything";
    }

    for (usize index = 0; index < weights_to_store.size(); ++index) {
        weight_matrix.set_edge_weight((s32)index, network[index][0], weights_to_store[index]);
    }

    // Bit-for-bit, not approximately: the stored value IS the weight, so there is no
    // rounding left to tolerate.
    for (usize index = 0; index < weights_to_store.size(); ++index) {
        EXPECT_EQ(float_bit_pattern(weight_matrix.get((s32)index, network[index][0])),
                  float_bit_pattern(weights_to_store[index]))
            << "index=" << index << " weight=" << weights_to_store[index];
    }

    // The GPU-side bulk read reconstructs from the same Ck/Sk pair the propagate kernel
    // uses, so it has to agree bit-for-bit too -- a host-only fix would leave the
    // simulation itself running on the annihilated values.
    vector<f32> bulk_weights(
        (usize)(weight_matrix.node_count * weight_matrix.max_neighbor_count));
    weight_matrix.neighbor_weights(bulk_weights.data());
    for (usize index = 0; index < weights_to_store.size(); ++index) {
        // network[index][0] is the first neighbor get_neighbors enumerates only if the
        // adjacency is already sorted, so locate the slot rather than assuming it.
        vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
        s64 degree = weight_matrix.get_neighbors((s64)index, neighbor_buffer.data());
        s64 written_slot = -1;
        for (s64 slot = 0; slot < degree; ++slot) {
            if (neighbor_buffer[(usize)slot] == network[index][0]) written_slot = slot;
        }
        ASSERT_GE(written_slot, 0) << "index=" << index;
        EXPECT_EQ(float_bit_pattern(bulk_weights[(usize)((s64)index * weight_matrix.max_neighbor_count
                                                          + written_slot)]),
                  float_bit_pattern(weights_to_store[index]))
            << "index=" << index;
    }

    EXPECT_TRUE(weight_matrix.using_exact_edge_weights);
    // No new matrix and no new buffer: the weights live in the default matrix's own Sk,
    // which is allocated at construction whether or not anything is ever written into it.
    EXPECT_EQ(weight_matrix.matrix_count(), 1);
}

TEST(WeightMatrix, set_edge_weight_leaves_every_other_edge_bit_identical) {
    auto network = square_torus(6);
    WeightMatrix weight_matrix(network, /*rank=*/16, /*check_indexing=*/true,
                               /*max_neighbor_count=*/-1, /*weight_seed=*/9);

    const EdgeWeightSnapshot before = snapshot_every_edge_weight(weight_matrix);
    ASSERT_GT(before.values.size(), 0u);

    const s32 written_source = 3;
    const s32 written_target = network[3][1];
    weight_matrix.set_edge_weight(written_source, written_target, 2.5e-8f);

    const EdgeWeightSnapshot after = snapshot_every_edge_weight(weight_matrix);
    ASSERT_EQ(after.values.size(), before.values.size());

    // Switching into exact mode rewrites how every edge is STORED (its value moves out of
    // the low-rank plane and into Sk), so the guarantee that matters is that it changes
    // what no edge but the written one READS BACK as.
    for (usize edge_index = 0; edge_index < before.values.size(); ++edge_index) {
        if (before.source_nodes[edge_index] == written_source &&
            before.target_nodes[edge_index] == written_target) {
            EXPECT_EQ(float_bit_pattern(after.values[edge_index]),
                      float_bit_pattern(2.5e-8f));
            continue;
        }
        EXPECT_EQ(float_bit_pattern(after.values[edge_index]),
                  float_bit_pattern(before.values[edge_index]))
            << "edge " << before.source_nodes[edge_index] << " -> "
            << before.target_nodes[edge_index] << " moved";
    }
}

TEST(WeightMatrix, exact_edge_weights_survive_refit) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8, /*check_indexing=*/true,
                               /*max_neighbor_count=*/-1, /*weight_seed=*/11);

    // Realistic SI magnitudes on every edge, so the fit has a full point cloud to work on.
    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    vector<f32> expected_weights;
    for (s64 node_index = 0; node_index < weight_matrix.node_count; ++node_index) {
        s64 degree = weight_matrix.get_neighbors(node_index, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            f32 weight = 2.5e-9f * (f32)(1 + node_index) - 1e-11f * (f32)slot;
            weight_matrix.set_edge_weight((s32)node_index, neighbor_buffer[(usize)slot], weight);
            expected_weights.push_back(weight);
        }
    }

    // An ordinary per-edge matrix with a pending delta, so refit has real work to do and
    // genuinely moves the shared U/V basis under the weights.
    s64 ordinary_matrix_index =
        weight_matrix.add_coefficient_vector(vector<f32>((usize)weight_matrix.rank, 0.5f));
    weight_matrix.accumulate_edge_delta(ordinary_matrix_index, 0, network[0][0], 0.25f);

    f32 basis_value_before = weight_matrix.U_matrix.get_contents()[0].x;
    weight_matrix.refit(/*sweep_count=*/2);
    EXPECT_NE(float_bit_pattern(basis_value_before),
              float_bit_pattern(weight_matrix.U_matrix.get_contents()[0].x));

    usize expected_index = 0;
    for (s64 node_index = 0; node_index < weight_matrix.node_count; ++node_index) {
        s64 degree = weight_matrix.get_neighbors(node_index, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            EXPECT_EQ(float_bit_pattern(
                          weight_matrix.get((s32)node_index, neighbor_buffer[(usize)slot])),
                      float_bit_pattern(expected_weights[expected_index]))
                << "edge " << node_index << " -> " << neighbor_buffer[(usize)slot];
            ++expected_index;
        }
    }

    // The ordinary matrix's Sk was folded into the basis and cleared; the weights were
    // deliberately left alone, which is what keeps them exact.
    EXPECT_FALSE(weight_matrix.sparse_delta_touched[(usize)ordinary_matrix_index]);
    EXPECT_TRUE(weight_matrix.sparse_delta_touched[(usize)WeightMatrix::DEFAULT_MATRIX_INDEX]);
}

TEST(WeightMatrix, exact_edge_weights_are_excluded_from_the_refit_occupancy_trigger) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/4);
    weight_matrix.refit_occupancy_threshold_fraction = 0.1f; // opt in

    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    for (s64 node_index = 0; node_index < weight_matrix.node_count; ++node_index) {
        s64 degree = weight_matrix.get_neighbors(node_index, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            weight_matrix.set_edge_weight((s32)node_index, neighbor_buffer[(usize)slot], 2.5e-9f);
        }
    }

    // Every edge now holds a weight in the default matrix's Sk. Counted as drift, that is
    // permanent 100% occupancy, and the trigger would demand a refit on every single tick.
    EXPECT_EQ(weight_matrix.max_sparse_delta_occupancy_fraction(), 0.0f);
    EXPECT_FALSE(weight_matrix.is_refit_due());

    // An ordinary matrix's Sk still counts, so the trigger is exempted, not disabled.
    s64 ordinary_matrix_index =
        weight_matrix.add_coefficient_vector(vector<f32>((usize)weight_matrix.rank, 1.0f));
    s64 target_bump_count = (s64)(weight_matrix.total_edge_count * 0.15) + 1;
    s64 bumped_count = 0;
    for (s64 node_index = 0;
         node_index < weight_matrix.node_count && bumped_count < target_bump_count; ++node_index) {
        s64 degree = weight_matrix.get_neighbors(node_index, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree && bumped_count < target_bump_count; ++slot) {
            weight_matrix.accumulate_edge_delta(ordinary_matrix_index, (s32)node_index,
                                                neighbor_buffer[(usize)slot], 1.0f);
            ++bumped_count;
        }
    }
    EXPECT_TRUE(weight_matrix.is_refit_due());
}

TEST(WeightMatrix, accumulate_edge_delta_stays_exact_on_top_of_an_exact_weight) {
    // The plasticity shape: a small per-edge update against a small weight. With the weight
    // stored as itself rather than as a delta against an order-1 reconstruction, the sum is
    // accurate to the weight's own scale instead of to the reconstruction's.
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8, /*check_indexing=*/true,
                               /*max_neighbor_count=*/-1, /*weight_seed=*/5);

    const s32 source_node = 1;
    const s32 target_node = network[1][0];
    weight_matrix.set_edge_weight(source_node, target_node, 2.5e-9f);
    weight_matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX,
                                        source_node, target_node, 1e-10f);

    EXPECT_EQ(float_bit_pattern(weight_matrix.get(source_node, target_node)),
              float_bit_pattern(2.5e-9f + 1e-10f));
}

TEST(WeightMatrix, set_edge_weight_rejects_non_edges_and_out_of_bounds_indices) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);

    const s32 source_node = 0;
    s32 non_neighbor = find_non_neighbor(weight_matrix, source_node);
    ASSERT_GE(non_neighbor, 0);

    EXPECT_THROW(weight_matrix.set_edge_weight(source_node, non_neighbor, 1e-9f),
                 std::invalid_argument);
    EXPECT_THROW(weight_matrix.set_edge_weight(source_node, 4590, 1e-9f), std::invalid_argument);
    EXPECT_THROW(weight_matrix.set_edge_weight(-1, 0, 1e-9f), std::invalid_argument);

    // A rejected call leaves the matrix exactly as it was -- including not having switched
    // into exact mode behind the caller's back.
    EXPECT_FALSE(weight_matrix.using_exact_edge_weights);
    EXPECT_TRUE(sparse_delta_buffer_contents_are_all_zero(weight_matrix,
                                                          WeightMatrix::DEFAULT_MATRIX_INDEX));
}

TEST(WeightMatrix, scale_neighbor_weights_refuses_a_matrix_holding_exact_weights) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);
    weight_matrix.set_edge_weight(0, network[0][0], 2.5e-9f);

    // Scaling moves U/V, which no longer contribute to a weight at all -- the call would
    // report a scale factor and change nothing.
    EXPECT_THROW((void)weight_matrix.scale_neighbor_weights_to_root_mean_square(1.0f),
                 std::invalid_argument);
    EXPECT_EQ(float_bit_pattern(weight_matrix.get(0, network[0][0])),
              float_bit_pattern(2.5e-9f));
}

TEST(WeightMatrix, set_coefficient_vector_refuses_a_matrix_holding_exact_weights) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);
    weight_matrix.set_edge_weight(0, network[0][0], 2.5e-9f);

    // The default matrix's all-zero Ck is the whole reason an exact weight reads back as
    // exactly its Sk entry. Writing a Ck onto it -- the shape a mistyped matrix_index takes,
    // since DEFAULT_MATRIX_INDEX is a valid index -- would put an order-1 low-rank term on
    // top of a 2.5e-9 weight, i.e. destroy every weight in the matrix at once.
    vector<f32> coefficients((usize)weight_matrix.rank, 1.0f);
    EXPECT_THROW(weight_matrix.set_coefficient_vector(WeightMatrix::DEFAULT_MATRIX_INDEX,
                                                      coefficients),
                 std::invalid_argument);
    EXPECT_EQ(float_bit_pattern(weight_matrix.get(0, network[0][0])),
              float_bit_pattern(2.5e-9f));

    // Registering another matrix in the family is still allowed, and its Ck is still
    // writable -- only the pinned matrices are refused.
    s64 ordinary_matrix_index = weight_matrix.add_coefficient_vector(coefficients);
    weight_matrix.set_coefficient_vector(ordinary_matrix_index,
                                         vector<f32>((usize)weight_matrix.rank, 0.5f));
    EXPECT_EQ(float_bit_pattern(weight_matrix.get(0, network[0][0])),
              float_bit_pattern(2.5e-9f));
}

TEST(WeightMatrix, set_coefficient_vector_refuses_the_delay_matrix) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);

    const s32 source_node = 0;
    const s32 target_node = network[0][0];
    weight_matrix.set_edge_delay_ticks(source_node, target_node, 7);
    ASSERT_GE(weight_matrix.delay_matrix_index, 0);

    // Same invariant as the exact weights above: the delay matrix's Ck is pinned to all-zero
    // so a delay reconstructs as exactly its Sk entry. A Ck here reconstructs delays as
    // garbage tick counts.
    EXPECT_THROW(weight_matrix.set_coefficient_vector(weight_matrix.delay_matrix_index,
                                                      vector<f32>((usize)weight_matrix.rank, 1.0f)),
                 std::invalid_argument);
    EXPECT_EQ(weight_matrix.get_edge_delay_ticks(source_node, target_node), 7);
}

TEST(WeightMatrix, move_assignment_preserves_exact_edge_weights) {
    auto network = square_torus(4);
    WeightMatrix source_matrix(network, /*rank=*/8);
    source_matrix.set_edge_weight(0, network[0][0], 2.5e-9f);

    WeightMatrix destination_matrix(network, /*rank=*/8);
    destination_matrix = std::move(source_matrix);

    EXPECT_TRUE(destination_matrix.using_exact_edge_weights);
    EXPECT_EQ(float_bit_pattern(destination_matrix.get(0, network[0][0])),
              float_bit_pattern(2.5e-9f));
}

TEST(WeightMatrix, load_from_disk_leaves_exact_mode_off_after_reallocating) {
    auto network = square_torus(4);
    WeightMatrix saved_matrix(network, /*rank=*/4); // rank_float4_stride 1
    const char *path = "/tmp/spikecorec_test_wm_exact_weight_realloc.bin";
    saved_matrix.save(path);

    // A different rank_float4_stride forces load_from_disk to reallocate, which resets the
    // matrix family back to a single fresh all-ones default over an empty Sk -- no longer
    // exact mode, and the flag must not claim otherwise.
    WeightMatrix loading_matrix(network, /*rank=*/8); // rank_float4_stride 2
    loading_matrix.set_edge_weight(0, network[0][0], 2.5e-9f);
    ASSERT_TRUE(loading_matrix.using_exact_edge_weights);

    loading_matrix.load_from_disk(path);
    EXPECT_FALSE(loading_matrix.using_exact_edge_weights);
}

// ── per-edge synapse state (the peredge family) ──────────────────────────────

TEST(WeightMatrix, configure_per_edge_variable_count_registers_one_matrix_per_variable) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);

    const s64 matrix_count_before = weight_matrix.matrix_count();
    weight_matrix.configure_per_edge_variable_count(3);

    EXPECT_EQ(weight_matrix.per_edge_variable_count, 3);
    EXPECT_EQ(weight_matrix.matrix_count(), matrix_count_before + 3);
    EXPECT_EQ(weight_matrix.per_edge_variable_matrix_base, matrix_count_before);
    ASSERT_NE(weight_matrix.per_edge_variable_values.pointer, nullptr);

    for (s64 variable_index = 0; variable_index < 3; ++variable_index) {
        const s64 matrix_index = weight_matrix.per_edge_variable_matrix_index(variable_index);
        EXPECT_EQ(matrix_index, matrix_count_before + variable_index);
        EXPECT_TRUE(weight_matrix.is_per_edge_variable_matrix(matrix_index));
    }
    EXPECT_FALSE(weight_matrix.is_per_edge_variable_matrix(WeightMatrix::DEFAULT_MATRIX_INDEX));

    // Idempotent for the same count; a DIFFERENT non-zero count would orphan the matrices
    // already registered and move every plane offset, so it is refused rather than silently
    // reallocated.
    weight_matrix.configure_per_edge_variable_count(3);
    EXPECT_EQ(weight_matrix.matrix_count(), matrix_count_before + 3);
    EXPECT_THROW(weight_matrix.configure_per_edge_variable_count(4), std::invalid_argument);
    EXPECT_THROW(weight_matrix.configure_per_edge_variable_count(-1), std::invalid_argument);
}

TEST(WeightMatrix, per_edge_variables_round_trip_exactly_at_realistic_si_magnitudes) {
    // The whole point of pinning these matrices' Ck to all-zero. Expressed as a delta on top
    // of an order-1 low-rank reconstruction, a 1e-12 synapse state is rounded away entirely
    // and reads back as the reconstruction itself -- a network that looks mis-modelled rather
    // than mis-rounded.
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8, /*check_indexing=*/true,
                               /*max_neighbor_count=*/-1, /*weight_seed=*/23);
    weight_matrix.configure_per_edge_variable_count(2);

    const vector<f32> magnitudes = {1e-12f, 2.5e-9f, -7.5e-11f, 0.0f, 1e3f, -1.0f};

    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    vector<f32> expected_values;
    usize magnitude_position = 0;
    for (s64 node_index = 0; node_index < weight_matrix.node_count; ++node_index) {
        const s64 degree = weight_matrix.get_neighbors(node_index, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            for (s64 variable_index = 0; variable_index < 2; ++variable_index) {
                const f32 value = magnitudes[magnitude_position % magnitudes.size()] *
                                  (f32)(1 + variable_index);
                magnitude_position += 1;
                weight_matrix.set_edge_variable(variable_index, (s32)node_index,
                                                neighbor_buffer[(usize)slot], value);
                expected_values.push_back(value);
            }
        }
    }

    usize expected_position = 0;
    for (s64 node_index = 0; node_index < weight_matrix.node_count; ++node_index) {
        const s64 degree = weight_matrix.get_neighbors(node_index, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            for (s64 variable_index = 0; variable_index < 2; ++variable_index) {
                const f32 stored = weight_matrix.get_edge_variable(
                        variable_index, (s32)node_index, neighbor_buffer[(usize)slot]);
                EXPECT_EQ(float_bit_pattern(stored),
                          float_bit_pattern(expected_values[expected_position]))
                        << "variable " << variable_index << " on edge " << node_index << " -> "
                        << neighbor_buffer[(usize)slot];
                ++expected_position;
            }
        }
    }
}

TEST(WeightMatrix, per_edge_variables_are_addressed_by_edge_not_by_endpoint) {
    // Two out-edges of one source, and the same target reached from two sources: a layout
    // keyed on either endpoint rather than on the edge would have these overwrite each other.
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);
    weight_matrix.configure_per_edge_variable_count(1);

    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    const s64 degree = weight_matrix.get_neighbors(0, neighbor_buffer.data());
    ASSERT_GE(degree, 2);

    weight_matrix.set_edge_variable(0, 0, neighbor_buffer[0], 3.0e-9f);
    weight_matrix.set_edge_variable(0, 0, neighbor_buffer[1], 7.0e-9f);
    EXPECT_FLOAT_EQ(weight_matrix.get_edge_variable(0, 0, neighbor_buffer[0]), 3.0e-9f);
    EXPECT_FLOAT_EQ(weight_matrix.get_edge_variable(0, 0, neighbor_buffer[1]), 7.0e-9f);

    // A second source reaching the first target keeps its own slot.
    vector<s32> predecessor_buffer((usize)weight_matrix.max_neighbor_count);
    const s64 predecessor_count =
            weight_matrix.get_predecessors(neighbor_buffer[0], predecessor_buffer.data());
    ASSERT_GE(predecessor_count, 2);
    s32 other_source = -1;
    for (s64 slot = 0; slot < predecessor_count; ++slot) {
        if (predecessor_buffer[(usize)slot] != 0) other_source = predecessor_buffer[(usize)slot];
    }
    ASSERT_GE(other_source, 0);

    weight_matrix.set_edge_variable(0, other_source, neighbor_buffer[0], 1.1e-9f);
    EXPECT_FLOAT_EQ(weight_matrix.get_edge_variable(0, 0, neighbor_buffer[0]), 3.0e-9f);
    EXPECT_FLOAT_EQ(weight_matrix.get_edge_variable(0, other_source, neighbor_buffer[0]), 1.1e-9f);
}

TEST(WeightMatrix, per_edge_variables_survive_refit) {
    // Folding a 1e-12 synapse state into the lossy low-rank plane would round it away, so
    // these matrices are exempt from the fit AND from the Sk clear -- the same trade per-edge
    // delay and exact weights already make. Missing either exemption is silent.
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8, /*check_indexing=*/true,
                               /*max_neighbor_count=*/-1, /*weight_seed=*/31);
    weight_matrix.configure_per_edge_variable_count(2);

    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    vector<f32> expected_values;
    for (s64 node_index = 0; node_index < weight_matrix.node_count; ++node_index) {
        const s64 degree = weight_matrix.get_neighbors(node_index, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            // Realistic weights too, so the fit has a full point cloud and genuinely moves
            // the shared basis under the per-edge variables.
            weight_matrix.set_edge_weight((s32)node_index, neighbor_buffer[(usize)slot],
                                          2.5e-9f * (f32)(1 + node_index));
            for (s64 variable_index = 0; variable_index < 2; ++variable_index) {
                const f32 value = 1e-12f * (f32)(1 + slot) + 3e-10f * (f32)variable_index;
                weight_matrix.set_edge_variable(variable_index, (s32)node_index,
                                                neighbor_buffer[(usize)slot], value);
                expected_values.push_back(value);
            }
        }
    }

    const f32 basis_value_before = weight_matrix.U_matrix.get_contents()[0].x;
    weight_matrix.refit(/*sweep_count=*/2);
    EXPECT_NE(float_bit_pattern(basis_value_before),
              float_bit_pattern(weight_matrix.U_matrix.get_contents()[0].x))
            << "refit did not move the shared basis, so surviving it proves nothing";

    usize expected_position = 0;
    for (s64 node_index = 0; node_index < weight_matrix.node_count; ++node_index) {
        const s64 degree = weight_matrix.get_neighbors(node_index, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            for (s64 variable_index = 0; variable_index < 2; ++variable_index) {
                EXPECT_EQ(float_bit_pattern(weight_matrix.get_edge_variable(
                                  variable_index, (s32)node_index,
                                  neighbor_buffer[(usize)slot])),
                          float_bit_pattern(expected_values[expected_position]))
                        << "variable " << variable_index << " on edge " << node_index << " -> "
                        << neighbor_buffer[(usize)slot] << " did not survive refit";
                ++expected_position;
            }
        }
    }

    // Their Ck stayed pinned to all-zero, which is what the survival above rests on: any
    // non-zero lane would put an order-1 low-rank term on top of a 1e-12 value.
    for (s64 variable_index = 0; variable_index < 2; ++variable_index) {
        const s64 matrix_index = weight_matrix.per_edge_variable_matrix_index(variable_index);
        const f32 *coefficients =
                weight_matrix.coefficient_vectors[(usize)matrix_index].get_contents();
        for (s64 lane_index = 0; lane_index < weight_matrix.rank_float4_stride * 4; ++lane_index) {
            EXPECT_EQ(float_bit_pattern(coefficients[lane_index]), float_bit_pattern(0.0f))
                    << "variable " << variable_index << " lane " << lane_index;
        }
    }
}

TEST(WeightMatrix, per_edge_variables_are_excluded_from_the_refit_occupancy_trigger) {
    // Their planes hold the model's live state permanently, not drift awaiting a refit.
    // Counting them would pin the occupancy fraction near 100% forever and make the trigger
    // fire on every tick, exactly as it would for the delay and exact-weight planes.
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);
    weight_matrix.configure_per_edge_variable_count(1);

    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    for (s64 node_index = 0; node_index < weight_matrix.node_count; ++node_index) {
        const s64 degree = weight_matrix.get_neighbors(node_index, neighbor_buffer.data());
        for (s64 slot = 0; slot < degree; ++slot) {
            weight_matrix.set_edge_variable(0, (s32)node_index, neighbor_buffer[(usize)slot],
                                            4.0e-10f);
        }
    }

    EXPECT_FLOAT_EQ(weight_matrix.max_sparse_delta_occupancy_fraction(), 0.0f);

    weight_matrix.refit_occupancy_threshold_fraction = 0.5f;
    weight_matrix.refit_every_n_ticks = 1000000;
    EXPECT_FALSE(weight_matrix.is_refit_due());
}

TEST(WeightMatrix, set_coefficient_vector_refuses_a_per_edge_variable_matrix) {
    // Writing any Ck onto one -- including the neutral 1.0f this method puts in the padding
    // lanes -- restores an order-1 low-rank term on top of values that are routinely 1e-12,
    // corrupting every per-edge synapse state at once with no diagnostic.
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);
    weight_matrix.configure_per_edge_variable_count(1);

    const s64 matrix_index = weight_matrix.per_edge_variable_matrix_index(0);
    EXPECT_THROW(weight_matrix.set_coefficient_vector(
                         matrix_index, vector<f32>((usize)weight_matrix.rank, 0.5f)),
                 std::invalid_argument);
}

TEST(WeightMatrix, per_edge_variable_accessors_reject_non_edges_and_bad_indices) {
    auto network = square_torus(4);
    WeightMatrix weight_matrix(network, /*rank=*/8);
    weight_matrix.configure_per_edge_variable_count(1);

    vector<s32> neighbor_buffer((usize)weight_matrix.max_neighbor_count);
    const s64 degree = weight_matrix.get_neighbors(0, neighbor_buffer.data());
    ASSERT_GT(degree, 0);

    EXPECT_THROW(weight_matrix.set_edge_variable(1, 0, neighbor_buffer[0], 1.0f),
                 std::invalid_argument);
    EXPECT_THROW((void)weight_matrix.get_edge_variable(-1, 0, neighbor_buffer[0]),
                 std::invalid_argument);
    EXPECT_THROW(weight_matrix.set_edge_variable(0, -1, neighbor_buffer[0], 1.0f),
                 std::invalid_argument);

    // A pair that is not a real edge: loadedge/accedge are edge-scoped operations.
    s32 non_neighbor = -1;
    for (s32 candidate = 0; candidate < (s32)weight_matrix.node_count; ++candidate) {
        if (weight_matrix.k2tree.adjacent(0, candidate)) continue;
        non_neighbor = candidate;
        break;
    }
    ASSERT_GE(non_neighbor, 0);
    EXPECT_THROW(weight_matrix.set_edge_variable(0, 0, non_neighbor, 1.0f),
                 std::invalid_argument);
    EXPECT_THROW((void)weight_matrix.get_edge_variable(0, 0, non_neighbor),
                 std::invalid_argument);
}

TEST(WeightMatrix, per_edge_variables_survive_move_assignment) {
    auto network = square_torus(4);
    WeightMatrix source_matrix(network, /*rank=*/8);
    source_matrix.configure_per_edge_variable_count(2);
    source_matrix.set_edge_variable(1, 0, network[0][0], 6.25e-12f);

    WeightMatrix destination_matrix(network, /*rank=*/8);
    destination_matrix = std::move(source_matrix);

    EXPECT_EQ(destination_matrix.per_edge_variable_count, 2);
    EXPECT_EQ(float_bit_pattern(destination_matrix.get_edge_variable(1, 0, network[0][0])),
              float_bit_pattern(6.25e-12f));
}

TEST(WeightMatrix, per_edge_variables_are_unregistered_when_load_from_disk_reallocates) {
    // Reallocating resets the whole matrix family, so a per_edge_variable_matrix_base left
    // pointing into it would have is_per_edge_variable_matrix claim the fresh default slot --
    // and refit would then stop clearing it.
    auto network = square_torus(4);
    WeightMatrix saved_matrix(network, /*rank=*/4); // rank_float4_stride 1
    const char *path = "/tmp/spikecorec_test_wm_per_edge_realloc.bin";
    saved_matrix.save(path);

    WeightMatrix loading_matrix(network, /*rank=*/8); // rank_float4_stride 2
    loading_matrix.configure_per_edge_variable_count(2);
    ASSERT_EQ(loading_matrix.per_edge_variable_count, 2);

    loading_matrix.load_from_disk(path);
    EXPECT_EQ(loading_matrix.per_edge_variable_count, 0);
    EXPECT_EQ(loading_matrix.per_edge_variable_matrix_base, -1);
    EXPECT_FALSE(
            loading_matrix.is_per_edge_variable_matrix(WeightMatrix::DEFAULT_MATRIX_INDEX));
}
