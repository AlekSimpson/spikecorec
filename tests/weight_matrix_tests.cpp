#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

#include <gtest/gtest.h>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/types.h"
#include "spikecorec/core/weight_matrix.h"

using namespace std;
using namespace spikecorec;

namespace {

// One backend for the file. Every matrix carves its own slab and releases it when it dies,
// so sharing the backend is not sharing storage -- it only avoids standing up a Metal
// device per test.
EngineBackend &test_backend() {
    static EngineBackend backend;
    return backend;
}

// Two source nodes into one target, plus an unconnected fourth. Small enough that every
// edge ordinal can be named in a comment, which is what the run tables are indexed by.
//
//   node 0 -> 1, 2      ordinals 0, 1
//   node 1 -> 2         ordinal  2
//   node 2 ->           (none)
//   node 3 ->           (none)
vector<vector<s32>> small_network() {
    return {{1, 2}, {2}, {}, {}};
}

// One run per edge, each with its own weight and delay, so the runs are distinguishable
// from each other and from a matrix that ignored them.
void declare_one_run_per_edge(WeightMatrix &matrix,
                              const Vector<f32> &weights,
                              const Vector<s32> &delays) {
    Vector<s64> first_edge_ordinal;
    Vector<s64> edge_count;
    Vector<s32> synapse_prototype;
    for (usize index = 0; index < weights.size(); index += 1) {
        first_edge_ordinal.push_back((s64)index);
        edge_count.push_back(1);
        synapse_prototype.push_back((s32)index);
    }
    matrix.declare_projections(first_edge_ordinal, edge_count, synapse_prototype, weights, delays);
}

} // namespace

// ── edge numbering ────────────────────────────────────────────────────────────────

// The canonical ordinal is what every run table and every staged delta is keyed by, so a
// disagreement about it silently gives edges the wrong synapse. It has to be a prefix sum
// over REAL out-degree -- not over a padded maximum, which is the storage this class
// exists to avoid.
TEST(WeightMatrix, edges_are_numbered_by_real_out_degree) {
    WeightMatrix matrix(test_backend(), small_network());

    EXPECT_EQ(matrix.total_edge_count, 3);

    EXPECT_EQ(matrix.edge_ordinal(0, 1).value(), 0);
    EXPECT_EQ(matrix.edge_ordinal(0, 2).value(), 1);
    EXPECT_EQ(matrix.edge_ordinal(1, 2).value(), 2);

    // Node 2 has an incoming edge but no outgoing one, and node 3 has neither.
    EXPECT_FALSE(matrix.edge_ordinal(2, 0).has_value());
    EXPECT_FALSE(matrix.edge_ordinal(3, 0).has_value());

    // A pair that is not an edge has no ordinal even when both endpoints exist.
    EXPECT_FALSE(matrix.edge_ordinal(1, 0).has_value());
}

// The ordering has to be stable, because the codegen bakes run boundaries against it while
// the engine hands the same boundaries to this class. Both derive it independently.
TEST(WeightMatrix, edge_numbering_is_contiguous_and_covers_every_edge) {
    WeightMatrix matrix(test_backend(), small_network());

    vector<bool> seen((usize)matrix.total_edge_count, false);
    vector<s32> neighbors((usize)max<s64>(matrix.max_neighbor_count, 1));
    for (s64 source = 0; source < matrix.node_count; source += 1) {
        const s64 degree = matrix.get_neighbors(source, neighbors.data());
        for (s64 slot = 0; slot < degree; slot += 1) {
            const optional<s64> ordinal = matrix.edge_ordinal((s32)source, neighbors[(usize)slot]);
            ASSERT_TRUE(ordinal.has_value());
            ASSERT_GE(*ordinal, 0);
            ASSERT_LT(*ordinal, matrix.total_edge_count);
            EXPECT_FALSE(seen[(usize)*ordinal]) << "ordinal " << *ordinal << " used twice";
            seen[(usize)*ordinal] = true;
        }
    }
    EXPECT_EQ(count(seen.begin(), seen.end(), true), matrix.total_edge_count);
}

// ── the shared basis ──────────────────────────────────────────────────────────────

// The whole design rests on this: a network wired out of projections is representable
// EXACTLY, with no fit iteration, because each run gets its own latent lane and no other
// lane is nonzero at both endpoints of that run's edges.
TEST(WeightMatrix, projection_weights_reconstruct_exactly) {
    WeightMatrix matrix(test_backend(), small_network());
    declare_one_run_per_edge(matrix, {0.25f, 1.75f, 0.001f}, {10, 20, 30});

    EXPECT_FLOAT_EQ(matrix.get(0, 1), 0.25f);
    EXPECT_FLOAT_EQ(matrix.get(0, 2), 1.75f);
    EXPECT_FLOAT_EQ(matrix.get(1, 2), 0.001f);

    // The engine's own measurement of the same thing, which is what decides whether a
    // model is allowed to load at all.
    EXPECT_FLOAT_EQ(matrix.measured_weight_fit_error, 0.0f);
}

// Realistic synaptic weights sit at 1e-9 and below. The representation this replaced
// seeded U and V from N(0,1) and stored the residual, which rounded a 5e-10 weight away
// entirely -- the bug that made someone pin the coefficients to zero and store the values
// raw. Scale is a property of the basis, so it has to survive at any magnitude.
TEST(WeightMatrix, tiny_weights_survive_the_basis) {
    WeightMatrix matrix(test_backend(), small_network());
    declare_one_run_per_edge(matrix, {5.0e-10f, 1.0e-12f, 2.5e-9f}, {1, 1, 1});

    EXPECT_FLOAT_EQ(matrix.get(0, 1), 5.0e-10f);
    EXPECT_FLOAT_EQ(matrix.get(0, 2), 1.0e-12f);
    EXPECT_FLOAT_EQ(matrix.get(1, 2), 2.5e-9f);
    EXPECT_FLOAT_EQ(matrix.measured_weight_fit_error, 0.0f);
}

// Delay shares U and V with weight and differs only by its coefficient row. It must
// round-trip as an exact integer, not to a tolerance: one tick out reads the wrong row of
// the spike-history ring, which is a different simulation rather than a rounder one.
TEST(WeightMatrix, delays_round_trip_as_exact_integers) {
    WeightMatrix matrix(test_backend(), small_network());
    declare_one_run_per_edge(matrix, {0.25f, 1.75f, 0.001f}, {10, 20, 30});

    EXPECT_EQ(matrix.get_edge_delay_ticks(0, 1), 10);
    EXPECT_EQ(matrix.get_edge_delay_ticks(0, 2), 20);
    EXPECT_EQ(matrix.get_edge_delay_ticks(1, 2), 30);
}

// One value for the whole network needs no basis at all, and the uniform-delay path is
// what keeps a topology-built network from reconstructing a delay per edge.
TEST(WeightMatrix, a_single_run_uses_the_constant_delay_path) {
    WeightMatrix matrix(test_backend(), small_network());
    matrix.declare_projections({0}, {3}, {0}, {0.5f}, {7});

    EXPECT_TRUE(matrix.using_constant_delay_ticks);
    EXPECT_EQ(matrix.get_edge_delay_ticks(0, 1), 7);
    EXPECT_EQ(matrix.get_edge_delay_ticks(1, 2), 7);
    EXPECT_FLOAT_EQ(matrix.get(0, 1), 0.5f);
    EXPECT_FLOAT_EQ(matrix.get(1, 2), 0.5f);
}

// ── prototype runs ────────────────────────────────────────────────────────────────

// Prototype index deliberately does NOT go through the basis: it selects a switch case in
// the kernel, so a reconstruction error would be a wrong synapse type rather than a small
// numeric one.
TEST(WeightMatrix, synapse_prototype_comes_from_the_run_table) {
    WeightMatrix matrix(test_backend(), small_network());

    // Two runs: ordinals 0-1 use prototype 0, ordinal 2 uses prototype 1.
    matrix.declare_projections({0, 2}, {2, 1}, {0, 1}, {0.25f, 0.75f}, {1, 1});

    EXPECT_EQ(matrix.get_edge_synapse_prototype(0, 1), 0);
    EXPECT_EQ(matrix.get_edge_synapse_prototype(0, 2), 0);
    EXPECT_EQ(matrix.get_edge_synapse_prototype(1, 2), 1);

    // Not an edge, so no run contains it.
    EXPECT_EQ(matrix.get_edge_synapse_prototype(3, 0), -1);

    // The two runs carry different weights, which is what tells a correct lookup apart
    // from one that always answers with the first run.
    EXPECT_FLOAT_EQ(matrix.get(0, 1), 0.25f);
    EXPECT_FLOAT_EQ(matrix.get(1, 2), 0.75f);
}

// ── storage ───────────────────────────────────────────────────────────────────────

// The invariant, asserted rather than assumed: nothing this class allocates may scale with
// node_count * max_neighbor_count. A regression here is the whole design coming undone,
// and it would be invisible in every numerical test above.
TEST(WeightMatrix, nothing_is_sized_by_the_padded_neighbour_count) {
    // A hub node gives max_neighbor_count a value far above the average degree, which is
    // exactly the shape padded storage wastes the most on.
    vector<vector<s32>> hub_network((usize)64);
    for (s32 target = 1; target < 64; target += 1) hub_network[0].push_back(target);
    for (s32 source = 1; source < 64; source += 1) hub_network[(usize)source].push_back(0);

    WeightMatrix matrix(test_backend(), hub_network);

    ASSERT_EQ(matrix.max_neighbor_count, 63);
    ASSERT_EQ(matrix.total_edge_count, 126);

    const u64 padded_plane_bytes =
            (u64)matrix.node_count * (u64)matrix.max_neighbor_count * sizeof(f32);

    // The basis scales with nodes and rank, and the scratch with edges. Neither may reach
    // the size of even one padded plane on a graph this lopsided.
    EXPECT_LT(matrix.U_matrix.total_bytes, padded_plane_bytes);
    EXPECT_LT(matrix.V_matrix.total_bytes, padded_plane_bytes);
    EXPECT_EQ(matrix.neighbor_weight_scratch.total_bytes,
              (u64)matrix.total_edge_count * sizeof(f32));
    EXPECT_EQ(matrix.edge_row_offset.total_bytes,
              (u64)(matrix.node_count + 1) * sizeof(s64));
}

// The correction layer is bounded by capacity, never by edge count. A matrix given no
// capacity simply carries no corrections -- the basis is then the whole answer, accuracy
// hit included.
TEST(WeightMatrix, the_correction_layer_is_bounded_by_capacity) {
    WeightMatrix without(test_backend(), small_network());
    EXPECT_EQ(without.sparse_delta_capacity, 0);
    EXPECT_TRUE(without.sparse_delta_edge_ordinal.is_empty());

    WeightMatrix with(test_backend(), small_network(), -1, true, -1, 7, /*capacity=*/128);
    EXPECT_EQ(with.sparse_delta_capacity, 128);
    EXPECT_FALSE(with.sparse_delta_row_start.is_empty());
    EXPECT_FALSE(with.sparse_delta_edge_ordinal.is_empty());

    // Never sized by the edge count, however many edges there are.
    EXPECT_EQ(with.sparse_delta_edge_ordinal.total_bytes,
              (u64)WeightMatrix::MATRIX_COUNT * 128 * sizeof(s64));
}

// An update queues rather than moving U/V, and becomes visible to reads once merged. This
// is the whole read-path contract: basis plus correction, never one or the other.
TEST(WeightMatrix, a_queued_correction_changes_what_a_read_returns) {
    WeightMatrix matrix(test_backend(), small_network(), -1, true, -1, 7, /*capacity=*/128);
    declare_one_run_per_edge(matrix, {0.25f, 1.75f, 0.001f}, {1, 1, 1});

    ASSERT_FLOAT_EQ(matrix.get(0, 1), 0.25f);

    matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, 0, 1, 0.5f);
    EXPECT_FLOAT_EQ(matrix.get(0, 1), 0.75f);

    // Only that edge moves. A correction is per-edge, and one that leaked into its
    // neighbours would be the CSR row bounds being wrong.
    EXPECT_FLOAT_EQ(matrix.get(0, 2), 1.75f);
    EXPECT_FLOAT_EQ(matrix.get(1, 2), 0.001f);

    // Corrections on the same edge accumulate rather than replacing.
    matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, 0, 1, -0.25f);
    EXPECT_FLOAT_EQ(matrix.get(0, 1), 0.5f);
}

// refit re-optimises the basis toward the values the corrections point at, then drops the
// corrections it has absorbed. The values a read returns must survive that unchanged --
// that is the entire point of the loop.
TEST(WeightMatrix, refit_absorbs_corrections_without_changing_what_reads_return) {
    WeightMatrix matrix(test_backend(), small_network(), -1, true, -1, 7, /*capacity=*/128);
    declare_one_run_per_edge(matrix, {0.25f, 1.75f, 0.001f}, {1, 1, 1});

    matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, 0, 1, 0.5f);
    matrix.accumulate_edge_delta(WeightMatrix::DEFAULT_MATRIX_INDEX, 1, 2, 0.1f);

    const f32 before_0_1 = matrix.get(0, 1);
    const f32 before_0_2 = matrix.get(0, 2);
    const f32 before_1_2 = matrix.get(1, 2);

    matrix.refit();

    EXPECT_NEAR(matrix.get(0, 1), before_0_1, 1e-3f);
    EXPECT_NEAR(matrix.get(0, 2), before_0_2, 1e-3f);
    EXPECT_NEAR(matrix.get(1, 2), before_1_2, 1e-3f);
}

// ── whole-network reads ───────────────────────────────────────────────────────────

// neighbor_weights writes one value per REAL edge, indexed by ordinal -- no padding rows
// and no sentinels to skip. It also has a GPU path and a host fallback, and the two must
// agree, which this checks by construction: the values come back either way.
TEST(WeightMatrix, neighbor_weights_is_indexed_by_edge_ordinal) {
    WeightMatrix matrix(test_backend(), small_network());
    declare_one_run_per_edge(matrix, {0.25f, 1.75f, 0.001f}, {1, 1, 1});

    vector<f32> edge_weights((usize)matrix.total_edge_count, -1.0f);
    matrix.neighbor_weights(edge_weights.data());

    ASSERT_EQ(edge_weights.size(), 3u);
    EXPECT_NEAR(edge_weights[0], 0.25f, 1e-6f);
    EXPECT_NEAR(edge_weights[1], 1.75f, 1e-6f);
    EXPECT_NEAR(edge_weights[2], 0.001f, 1e-6f);

    // Every entry was written; a padded implementation would leave sentinels behind.
    for (f32 weight : edge_weights) EXPECT_NE(weight, -1.0f);
}

TEST(WeightMatrix, statistics_are_taken_over_edges_not_padded_slots) {
    WeightMatrix matrix(test_backend(), small_network());
    declare_one_run_per_edge(matrix, {1.0f, 2.0f, 3.0f}, {1, 1, 1});

    const WeightStats stats = matrix.neighbor_weight_stats();

    // Mean of exactly the three real edges. Padding slots would drag it toward zero.
    EXPECT_NEAR(stats.mean, 2.0f, 1e-5f);
    EXPECT_NEAR(stats.min_value, 1.0f, 1e-5f);
    EXPECT_NEAR(stats.max_value, 3.0f, 1e-5f);
}

// ── lifetime ──────────────────────────────────────────────────────────────────────

// EnginePointer owns nothing; the slab does. Moving must hand that slab over exactly once,
// or the two objects both release it.
TEST(WeightMatrix, moving_transfers_the_slab_exactly_once) {
    WeightMatrix original(test_backend(), small_network());
    declare_one_run_per_edge(original, {0.25f, 1.75f, 0.001f}, {1, 1, 1});

    WeightMatrix moved = std::move(original);
    EXPECT_FLOAT_EQ(moved.get(0, 1), 0.25f);
    EXPECT_EQ(moved.total_edge_count, 3);

    // The source must not still believe it owns anything: both destructors run.
    EXPECT_EQ(original.owning_backend, nullptr);

    // Move-assignment over a live matrix releases the destination's own slab first.
    WeightMatrix destination(test_backend(), small_network());
    destination = std::move(moved);
    EXPECT_FLOAT_EQ(destination.get(0, 1), 0.25f);
    EXPECT_EQ(moved.owning_backend, nullptr);
}

// The empty network is an ordinary state, not an error: it is what an engine holds before
// it has parsed a model, and what a model with no connections keeps.
TEST(WeightMatrix, the_default_matrix_owns_nothing_and_destructs_cleanly) {
    WeightMatrix empty;
    EXPECT_EQ(empty.node_count, 0);
    EXPECT_EQ(empty.total_edge_count, 0);
    EXPECT_EQ(empty.owning_backend, nullptr);
    EXPECT_TRUE(empty.U_matrix.is_empty());
}

// ── serialization ─────────────────────────────────────────────────────────────────

// The coefficients are where the declared values actually live, so a save that dropped
// them would restore a basis that reconstructs nothing meaningful.
TEST(WeightMatrix, save_and_load_round_trip_preserves_reconstructed_values) {
    const string path = filesystem::temp_directory_path().string() +
                        "/spikecorec_weight_matrix_" + to_string(getpid()) + ".bin";

    WeightMatrix saved(test_backend(), small_network());
    declare_one_run_per_edge(saved, {0.25f, 1.75f, 0.001f}, {10, 20, 30});
    saved.save(path.c_str());

    WeightMatrix loaded(test_backend(), small_network());
    loaded.load_from_disk(path.c_str());

    EXPECT_FLOAT_EQ(loaded.get(0, 1), 0.25f);
    EXPECT_FLOAT_EQ(loaded.get(0, 2), 1.75f);
    EXPECT_FLOAT_EQ(loaded.get(1, 2), 0.001f);

    EXPECT_EQ(loaded.get_edge_delay_ticks(0, 1), 10);
    EXPECT_EQ(loaded.get_edge_delay_ticks(1, 2), 30);

    filesystem::remove(path);
}
