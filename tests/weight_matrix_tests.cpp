#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "backend.h"
#include "types.h"
#include "weight_matrix.h"
#include "support/test_support.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::test_support;

namespace {

constexpr s64 WEIGHT = Weight_Matrix::WEIGHT_PLANE;
constexpr s64 DELAY = Weight_Matrix::DELAY_PLANE;
constexpr s64 FIRST_STATE = Weight_Matrix::FIRST_STATE_VARIABLE_PLANE;

// Two sources into node 2, plus an unconnected node 3. Small enough to name every edge:
//
//   node 0 -> 1, 2      ordinals 0, 1
//   node 1 -> 2         ordinal  2
//   node 2 ->           (none)
//   node 3 ->           (none)
Vector<Vector<s32>> small_network() {
    return {{1, 2}, {2}, {}, {}};
}

// Every source connects to out_degree distinct other nodes, chosen at random.
Vector<Vector<s32>> random_network(s32 node_count, s32 out_degree, unsigned seed) {
    Mersenne_Twister random_engine(seed);
    Vector<Vector<s32>> network(static_cast<usize>(node_count));
    for (s32 source = 0; source < node_count; source += 1) {
        Vector<s32> candidates;
        for (s32 target = 0; target < node_count; target += 1) {
            if (target != source) candidates.push_back(target);
        }
        shuffle(candidates.begin(), candidates.end(), random_engine);
        candidates.resize(static_cast<usize>(out_degree));
        sort(candidates.begin(), candidates.end());
        network[static_cast<usize>(source)] = candidates;
    }
    return network;
}

// The (source, target) pair of every edge, indexed by ordinal.
Vector<Pair<s32, s32>> edges_by_ordinal(const Weight_Matrix &matrix) {
    Vector<Pair<s32, s32>> edges(static_cast<usize>(matrix.total_edge_count));
    Vector<u32> neighbors(static_cast<usize>(max<s64>(matrix.max_neighbor_count, 1)));
    for (s64 source = 0; source < matrix.node_count; source += 1) {
        const s64 degree = matrix.get_neighbors(source, neighbors.data());
        for (s64 slot = 0; slot < degree; slot += 1) {
            const s32 target = static_cast<s32>(neighbors[static_cast<usize>(slot)]);
            const s64 ordinal = *matrix.edge_ordinal(static_cast<s32>(source), target);
            edges[static_cast<usize>(ordinal)] = {static_cast<s32>(source), target};
        }
    }
    return edges;
}

// One run per edge, so values_per_plane[plane][ordinal] is that edge's starting value.
void declare_one_run_per_edge(Weight_Matrix &matrix, const Vector<Vector<f32>> &values_per_plane) {
    Vector<s64> first_edge_ordinal;
    Vector<s64> edge_count;
    Vector<s32> synapse_prototype;
    for (s64 ordinal = 0; ordinal < matrix.total_edge_count; ordinal += 1) {
        first_edge_ordinal.push_back(ordinal);
        edge_count.push_back(1);
        synapse_prototype.push_back(0);
    }
    matrix.declare_runs(first_edge_ordinal, edge_count, synapse_prototype, values_per_plane);
}

// Every edge's value in one plane, basis plus S, by ordinal.
Vector<f32> plane_values(const Weight_Matrix &matrix, s64 plane) {
    Vector<f32> values(static_cast<usize>(matrix.total_edge_count));
    matrix.neighbor_weights_for_matrix(values.data(), plane);
    return values;
}

// The worst error over the edges, relative to the larger of each expected value and the plane's
// RMS: what fit_tolerance bounds.
f64 worst_relative_error(const Vector<f32> &read, const Vector<f32> &expected) {
    f64 sum_of_squares = 0.0;
    for (f32 value : expected) sum_of_squares += static_cast<f64>(value) * static_cast<f64>(value);
    const f64 root_mean_square = expected.empty() ? 1.0 : sqrt(sum_of_squares / static_cast<f64>(expected.size()));
    const f64 scale_floor = root_mean_square > 0.0 ? root_mean_square : 1.0;
    f64 worst = 0.0;
    for (usize index = 0; index < expected.size(); index += 1) {
        const f64 scale = max(fabs(static_cast<f64>(expected[index])), scale_floor);
        worst = max(worst, fabs(static_cast<f64>(read[index]) - static_cast<f64>(expected[index])) / scale);
    }
    return worst;
}

// Queues an update the way the kernel does, for compact_pending_deltas to merge into S. A full
// queue is merged first, as the engine merges it every tick.
void queue_update(Weight_Matrix &matrix, s64 plane, s64 ordinal, f32 delta) {
    ASSERT_GT(matrix.pending_delta_capacity, 0);
    s32 &count = matrix.pending_delta_control.get_contents_as<s32>()[Weight_Matrix::PENDING_WRITE_COUNT_SLOT];
    if (count >= matrix.pending_delta_capacity) matrix.compact_pending_deltas();
    matrix.pending_delta_edge_ordinal.get_contents_as<u32>()[count] = static_cast<u32>(ordinal);
    matrix.pending_delta_value.get_contents_as<f32>()[count] = delta;
    matrix.pending_delta_matrix_index.get_contents_as<u8>()[count] = static_cast<u8>(plane);
    count += 1;
}

Vector<f32> uniform_values(s64 count, f32 low, f32 high, unsigned seed) {
    Mersenne_Twister random_engine(seed);
    Uniform_Real_Distribution<f32> distribution(low, high);
    Vector<f32> values(static_cast<usize>(count));
    for (f32 &value : values) value = distribution(random_engine);
    return values;
}

// Per-connection values with structure and no random process: a part from the source plus a part
// from the target, or their product, which a basis can hold below raw size. Values drawn at random
// per connection have nothing for a basis to find.
f32 structured_weight(s32 source, s32 target) {
    return 0.5f + 0.1f * static_cast<f32>(source % 7) + 0.05f * static_cast<f32>(target % 5);
}
f32 structured_delay(s32 source, s32 target) {
    return static_cast<f32>(1 + source % 6 + 2 * (target % 4));
}
f32 structured_state(s32 source, s32 target) {
    return 1e-9f * (static_cast<f32>(source % 5) - 2.0f) * (1.0f + static_cast<f32>(target % 3));
}

// One value per edge, by ordinal, from a function of the edge's ends.
Vector<f32> values_per_edge(const Weight_Matrix &matrix, f32 (*value_of)(s32, s32)) {
    Vector<f32> values;
    for (const auto &[source, target] : edges_by_ordinal(matrix)) values.push_back(value_of(source, target));
    return values;
}

// A four-plane matrix: weight, delay and the first state plane differ per connection, with
// structure; the last plane is all zeros.
struct Structured_Values_Matrix {
    Weight_Matrix matrix;
    Vector<Vector<f32>> declared;

    Structured_Values_Matrix(s32 node_count, s32 out_degree, s64 updated_plane_count)
        : matrix(shared_backend(), random_network(node_count, out_degree, 11), -1, true, -1, /*weight_seed=*/7,
                 /*matrix_count=*/4, updated_plane_count) {
        declared = {values_per_edge(matrix, structured_weight), values_per_edge(matrix, structured_delay),
                    values_per_edge(matrix, structured_state),
                    Vector<f32>(static_cast<usize>(matrix.total_edge_count), 0.0f)};
        declare_one_run_per_edge(matrix, declared);
    }
};

} // namespace

// ── edge numbering ──────────────────────────────────────────────────────────────

// Edge ordinals are a prefix sum over real out-degree, not over a padded maximum.
TEST(Weight_Matrix, edges_are_numbered_by_real_out_degree) {
    Weight_Matrix matrix(shared_backend(), small_network());

    EXPECT_EQ(matrix.total_edge_count, 3);
    EXPECT_EQ(matrix.edge_ordinal(0, 1).value(), 0);
    EXPECT_EQ(matrix.edge_ordinal(0, 2).value(), 1);
    EXPECT_EQ(matrix.edge_ordinal(1, 2).value(), 2);

    // Node 2 has an incoming edge but no outgoing one, node 3 has neither, and (1, 0) is no edge.
    EXPECT_FALSE(matrix.edge_ordinal(2, 0).has_value());
    EXPECT_FALSE(matrix.edge_ordinal(3, 0).has_value());
    EXPECT_FALSE(matrix.edge_ordinal(1, 0).has_value());
}

TEST(Weight_Matrix, edge_numbering_is_contiguous_and_covers_every_edge) {
    Weight_Matrix matrix(shared_backend(), random_network(40, 6, 5));

    Vector<bool> seen(static_cast<usize>(matrix.total_edge_count), false);
    for (const auto &[source, target] : edges_by_ordinal(matrix)) {
        const s64 ordinal = *matrix.edge_ordinal(source, target);
        EXPECT_FALSE(seen[static_cast<usize>(ordinal)]) << "ordinal " << ordinal << " used twice";
        seen[static_cast<usize>(ordinal)] = true;
    }
    EXPECT_EQ(count(seen.begin(), seen.end(), true), matrix.total_edge_count);
}

// A target outside the network, or a neighbour bound below a real row, would lose edges.
TEST(Weight_Matrix, a_network_it_cannot_hold_is_refused) {
    EXPECT_THROW(Weight_Matrix(shared_backend(), {{1, 9}, {0}, {}}), Invalid_Argument);
    EXPECT_THROW(Weight_Matrix(shared_backend(), {{1, -1}, {0}, {}}), Invalid_Argument);
    EXPECT_THROW(Weight_Matrix(shared_backend(), {{1, 2, 3}, {0}, {}, {}}, -1, true, /*max_neighbor_count=*/2),
                 Invalid_Argument);
}

// The basis is unconstrained off the edges, so reads of a pair that is not an edge give nothing.
TEST(Weight_Matrix, a_pair_that_is_not_an_edge_reads_as_no_value) {
    Weight_Matrix matrix(shared_backend(), small_network());
    const Vector<f32> weights = {0.5f, 0.5f, 0.5f};
    declare_one_run_per_edge(matrix, {weights, {3.0f, 3.0f, 3.0f}});

    EXPECT_FALSE(matrix.edge_ordinal(0, 3).has_value());
    EXPECT_EQ(matrix.get(0, 3), 0.0f);
    EXPECT_EQ(matrix.get_for_matrix(0, 3, DELAY), 0.0f);
    EXPECT_EQ(matrix.get_edge_delay_ticks(0, 3), 0);
    EXPECT_EQ(matrix.get_edge_synapse_prototype(0, 3), -1);
    EXPECT_NEAR(matrix.get(0, 1), 0.5, fit_allowance(matrix.fit_tolerance, 0.5, weights));
    EXPECT_EQ(matrix.get_edge_delay_ticks(0, 1), 3);
}

// A hub with a large in-degree: the get_predecessors buffer is bounded by the largest in-degree,
// not the largest out-degree.
TEST(Weight_Matrix, get_predecessors_lists_every_source_into_a_node) {
    const s32 node_count = 64;
    Vector<Vector<s32>> star(static_cast<usize>(node_count));
    for (s32 source = 1; source < node_count; source += 1) star[static_cast<usize>(source)].push_back(0);
    star[0].push_back(1);

    Weight_Matrix matrix(shared_backend(), star);
    EXPECT_EQ(matrix.max_neighbor_count, 1);
    EXPECT_EQ(matrix.max_predecessor_count, node_count - 1);

    Vector<u32> buffer(static_cast<usize>(matrix.max_predecessor_count));
    const s64 found_count = matrix.get_predecessors(0, buffer.data());
    Vector<u32> found(buffer.begin(), buffer.begin() + found_count);
    sort(found.begin(), found.end());
    Vector<u32> expected(static_cast<usize>(node_count - 1));
    iota(expected.begin(), expected.end(), 1);
    EXPECT_EQ(found, expected);

    EXPECT_EQ(matrix.get_predecessors(1, buffer.data()), 1);
    EXPECT_EQ(buffer[0], 0u);
}

// ── declared values ─────────────────────────────────────────────────────────────

// Every plane reads back within fit_tolerance of its declared values. Three edges and four planes
// leave one lane per plane, so each plane's values are a source factor times a target factor.
TEST(Weight_Matrix, run_values_reconstruct_within_the_tolerance_on_every_plane) {
    Weight_Matrix matrix(shared_backend(), small_network(), -1, true, -1, 7, /*matrix_count=*/4);
    const Vector<Vector<f32>> declared = {{0.25f, 1.75f, 0.001f}, {10.0f, 20.0f, 30.0f},
                                          {-1.0f, -0.5f, 3e-3f}, {0.0f, 0.0f, 0.0f}};
    declare_one_run_per_edge(matrix, declared);

    const Vector<Pair<s32, s32>> edges = edges_by_ordinal(matrix);
    for (s64 plane = 0; plane < 4; plane += 1) {
        for (usize ordinal = 0; ordinal < edges.size(); ordinal += 1) {
            const f32 value = declared[static_cast<usize>(plane)][ordinal];
            EXPECT_NEAR(matrix.get_for_matrix(edges[ordinal].first, edges[ordinal].second, plane), value,
                        fit_allowance(matrix.fit_tolerance, value, declared[static_cast<usize>(plane)]))
                << "plane " << plane << ", ordinal " << ordinal;
        }
        EXPECT_LE(matrix.measured_fit_error[static_cast<usize>(plane)], matrix.fit_tolerance) << "plane " << plane;
    }
    EXPECT_NEAR(matrix.get(0, 2), 1.75, fit_allowance(matrix.fit_tolerance, 1.75, declared[WEIGHT]));
}

// Synaptic weights sit at 1e-9 and below; the basis must reproduce them at any magnitude.
TEST(Weight_Matrix, tiny_values_survive_the_basis) {
    Weight_Matrix matrix(shared_backend(), small_network());
    const Vector<f32> weights = {5.0e-10f, 1.0e-12f, 2.5e-9f};
    declare_one_run_per_edge(matrix, {weights, {1.0f, 1.0f, 1.0f}});

    EXPECT_NEAR(matrix.get(0, 1), 5.0e-10, fit_allowance(matrix.fit_tolerance, 5.0e-10, weights));
    EXPECT_NEAR(matrix.get(0, 2), 1.0e-12, fit_allowance(matrix.fit_tolerance, 1.0e-12, weights));
    EXPECT_NEAR(matrix.get(1, 2), 2.5e-9, fit_allowance(matrix.fit_tolerance, 2.5e-9, weights));
}

// A delay is used as a whole number of ticks, at least one.
TEST(Weight_Matrix, delays_land_on_their_ticks) {
    Weight_Matrix matrix(shared_backend(), small_network());
    declare_one_run_per_edge(matrix, {{1.0f, 1.0f, 1.0f}, {10.0f, 0.4f, 30.0f}});

    EXPECT_EQ(matrix.get_edge_delay_ticks(0, 1), 10);
    EXPECT_EQ(matrix.get_edge_delay_ticks(0, 2), 1);
    EXPECT_EQ(matrix.get_edge_delay_ticks(1, 2), 30);
}

// The prototype selects a switch case in the kernel, so it comes from the run table, never the
// basis.
TEST(Weight_Matrix, synapse_prototype_comes_from_the_run_table) {
    Weight_Matrix matrix(shared_backend(), small_network());
    // Ordinals 0-1 use prototype 0 and ordinal 2 uses prototype 1, with different weights.
    matrix.declare_runs({0, 2}, {2, 1}, {0, 1}, {{0.25f, 0.75f}, {1.0f, 1.0f}});

    EXPECT_EQ(matrix.get_edge_synapse_prototype(0, 1), 0);
    EXPECT_EQ(matrix.get_edge_synapse_prototype(0, 2), 0);
    EXPECT_EQ(matrix.get_edge_synapse_prototype(1, 2), 1);
    EXPECT_EQ(matrix.get_edge_synapse_prototype(3, 0), -1);
    const Vector<f32> weights = {0.25f, 0.25f, 0.75f};
    EXPECT_NEAR(matrix.get(0, 1), 0.25, fit_allowance(matrix.fit_tolerance, 0.25, weights));
    EXPECT_NEAR(matrix.get(1, 2), 0.75, fit_allowance(matrix.fit_tolerance, 0.75, weights));
}

// Values that differ per connection, with structure, meet fit_tolerance on every plane within the
// lane budgets, and every delay lands on its tick.
TEST(Weight_Matrix, per_connection_values_meet_the_tolerance_on_every_plane) {
    Structured_Values_Matrix structured_values(/*node_count=*/200, /*out_degree=*/12, /*updated_plane_count=*/0);
    const Weight_Matrix &matrix = structured_values.matrix;

    for (s64 plane = 0; plane < 3; plane += 1) {
        const Vector<f32> read = plane_values(matrix, plane);
        EXPECT_LE(worst_relative_error(read, structured_values.declared[static_cast<usize>(plane)]), matrix.fit_tolerance)
            << "plane " << plane;
        EXPECT_LE(matrix.measured_fit_error[static_cast<usize>(plane)], matrix.fit_tolerance) << "plane " << plane;
    }

    s64 delays_off_their_tick = 0;
    for (const auto &[source, target] : edges_by_ordinal(matrix)) {
        const s64 ordinal = *matrix.edge_ordinal(source, target);
        const s32 declared_ticks = static_cast<s32>(structured_values.declared[static_cast<usize>(DELAY)][static_cast<usize>(ordinal)]);
        delays_off_their_tick += matrix.get_edge_delay_ticks(source, target) != declared_ticks ? 1 : 0;
    }
    EXPECT_EQ(delays_off_their_tick, 0);

    // A plane of zeros reads exactly zero.
    const Vector<f32> zeros = plane_values(matrix, FIRST_STATE + 1);
    EXPECT_EQ(zeros, Vector<f32>(zeros.size(), 0.0f));
    EXPECT_LE(matrix.rank, MAX_RANK_FLOAT4_STRIDE * Weight_Matrix::LANE_GROUP);
}

// ── updates ─────────────────────────────────────────────────────────────────────

// A read is the basis plus S: an update shows at once, on its own edge and plane only, and updates
// to one edge add up.
TEST(Weight_Matrix, an_update_changes_only_its_own_edge_and_plane) {
    Weight_Matrix matrix(shared_backend(), small_network(), -1, true, -1, 7, /*matrix_count=*/3);
    const Vector<f32> weights = {0.25f, 1.75f, 0.001f};
    const Vector<f32> delays = {1.0f, 2.0f, 3.0f};
    declare_one_run_per_edge(matrix, {weights, delays, {0.0f, 0.0f, 0.0f}});

    // An update adds to whatever the basis reads, exactly.
    const f32 basis_value = matrix.get(0, 1);
    EXPECT_NEAR(basis_value, 0.25, fit_allowance(matrix.fit_tolerance, 0.25, weights));
    const f32 other_values[2] = {matrix.get(0, 2), matrix.get(1, 2)};
    matrix.accumulate_edge_delta(WEIGHT, 0, 1, 0.5f);
    EXPECT_FLOAT_EQ(matrix.get(0, 1), basis_value + 0.5f);
    EXPECT_EQ(matrix.get(0, 2), other_values[0]);
    EXPECT_EQ(matrix.get(1, 2), other_values[1]);
    EXPECT_NEAR(matrix.get_for_matrix(0, 1, DELAY), 1.0, fit_allowance(matrix.fit_tolerance, 1.0, delays));

    matrix.accumulate_edge_delta(WEIGHT, 0, 1, -0.25f);
    EXPECT_FLOAT_EQ(matrix.get(0, 1), basis_value + 0.25f);

    matrix.accumulate_edge_delta(FIRST_STATE, 1, 2, 4.0f);
    EXPECT_FLOAT_EQ(matrix.get_for_matrix(1, 2, FIRST_STATE), 4.0f);
    EXPECT_FLOAT_EQ(matrix.get_for_matrix(0, 1, FIRST_STATE), 0.0f);
    EXPECT_EQ(matrix.sparse_delta_entry_count[static_cast<usize>(WEIGHT)], 1);
    EXPECT_EQ(matrix.sparse_delta_entry_count[static_cast<usize>(FIRST_STATE)], 1);
}

// compact_pending_deltas sums each edge's queued updates into one S entry, drops any that sum to
// exactly zero, and empties the queue.
TEST(Weight_Matrix, compact_pending_deltas_merges_the_queue_into_S) {
    Weight_Matrix matrix(shared_backend(), small_network(), -1, true, -1, 7, /*matrix_count=*/4,
                        /*updated_plane_count=*/2);
    declare_one_run_per_edge(matrix, {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
                                      {0.0f, 0.0f, 0.0f}});

    queue_update(matrix, FIRST_STATE, 0, 1.0f);
    queue_update(matrix, FIRST_STATE, 0, 0.5f);
    queue_update(matrix, FIRST_STATE + 1, 1, 2.0f);
    queue_update(matrix, FIRST_STATE + 1, 1, -2.0f);
    queue_update(matrix, FIRST_STATE + 1, 2, -3.0f);
    matrix.compact_pending_deltas();

    EXPECT_EQ(matrix.pending_delta_count(), 0);
    EXPECT_EQ(matrix.sparse_delta_entry_count[static_cast<usize>(FIRST_STATE)], 1);
    EXPECT_EQ(matrix.sparse_delta_entry_count[static_cast<usize>(FIRST_STATE + 1)], 1);
    EXPECT_FLOAT_EQ(matrix.get_for_matrix(0, 1, FIRST_STATE), 1.5f);
    EXPECT_FLOAT_EQ(matrix.get_for_matrix(0, 2, FIRST_STATE + 1), 0.0f);
    EXPECT_FLOAT_EQ(matrix.get_for_matrix(1, 2, FIRST_STATE + 1), -3.0f);

    // A later merge adds onto what S already holds.
    queue_update(matrix, FIRST_STATE, 0, 0.25f);
    matrix.compact_pending_deltas();
    EXPECT_FLOAT_EQ(matrix.get_for_matrix(0, 1, FIRST_STATE), 1.75f);
}

// S's pool is sized to the refit threshold, a fifth of the edges by default, for each plane the
// kernel updates, plus the plasticity reserve; the queue is sized like S.
TEST(Weight_Matrix, S_is_sized_to_the_refit_threshold_for_each_updated_plane) {
    EXPECT_FLOAT_EQ(Weight_Matrix::DEFAULT_REFIT_OCCUPANCY_THRESHOLD_FRACTION, 0.2f);

    Weight_Matrix matrix(shared_backend(), random_network(50, 7, 3), -1, true, -1, 7, /*matrix_count=*/4,
                        /*updated_plane_count=*/2);
    const s64 edge_count = matrix.total_edge_count;
    // The threshold is an f32 (0.2000000030), so a fifth of these 350 edges rounds up to 71.
    const s64 fifth = static_cast<s64>(ceil(static_cast<f64>(0.2f) * static_cast<f64>(edge_count)));
    EXPECT_EQ(matrix.pending_delta_capacity, max<s64>(2 * fifth, 2 * matrix.max_neighbor_count));

    matrix.plasticity_reserve_entries = 16;
    declare_one_run_per_edge(matrix, {uniform_values(edge_count, 0.5f, 1.5f, 4), Vector<f32>(static_cast<usize>(edge_count), 1.0f),
                                      Vector<f32>(static_cast<usize>(edge_count), 0.0f),
                                      Vector<f32>(static_cast<usize>(edge_count), 0.0f)});
    EXPECT_EQ(matrix.sparse_delta_capacity, 2 * fifth + 16);
    EXPECT_EQ(matrix.delta_capacity_for_threshold(0.5f), 2 * static_cast<s64>(ceil(0.5 * static_cast<f64>(edge_count))) + 16);
    // Without a threshold S must be able to hold every edge of every updated plane.
    EXPECT_EQ(matrix.delta_capacity_for_threshold(0.0f), 2 * edge_count + 16);
}

// A matrix whose kernel updates no plane keeps no room in S or the queue.
TEST(Weight_Matrix, no_updated_plane_keeps_no_update_room) {
    Weight_Matrix matrix(shared_backend(), random_network(50, 7, 3));
    declare_one_run_per_edge(matrix, {uniform_values(matrix.total_edge_count, 0.5f, 1.5f, 4),
                                      Vector<f32>(static_cast<usize>(matrix.total_edge_count), 1.0f)});
    EXPECT_EQ(matrix.sparse_delta_capacity, 0);
    EXPECT_EQ(matrix.pending_delta_capacity, 0);
    EXPECT_EQ(matrix.sparse_delta_edge_ordinal.total_bytes, 0u);
}

// A merge that would overflow S grows it instead of dropping an update.
TEST(Weight_Matrix, S_grows_rather_than_dropping_an_update) {
    Weight_Matrix matrix(shared_backend(), random_network(30, 5, 6), -1, true, -1, 7, /*matrix_count=*/3,
                        /*updated_plane_count=*/1);
    const s64 edge_count = matrix.total_edge_count;
    declare_one_run_per_edge(matrix, {Vector<f32>(static_cast<usize>(edge_count), 1.0f), Vector<f32>(static_cast<usize>(edge_count), 1.0f),
                                      Vector<f32>(static_cast<usize>(edge_count), 0.0f)});
    matrix.resize_delta_capacity(1);

    for (s64 ordinal = 0; ordinal < edge_count; ordinal += 1) {
        queue_update(matrix, FIRST_STATE, ordinal, static_cast<f32>(ordinal + 1));
    }
    matrix.compact_pending_deltas();

    EXPECT_GE(matrix.sparse_delta_capacity, edge_count);
    EXPECT_EQ(matrix.sparse_delta_entry_count[static_cast<usize>(FIRST_STATE)], edge_count);
    Vector<f32> expected(static_cast<usize>(edge_count));
    iota(expected.begin(), expected.end(), 1.0f);
    EXPECT_EQ(plane_values(matrix, FIRST_STATE), expected);
}

TEST(Weight_Matrix, a_refit_is_due_once_the_fullest_plane_reaches_the_threshold) {
    Weight_Matrix matrix(shared_backend(), random_network(20, 5, 8), -1, true, -1, 7, /*matrix_count=*/3,
                        /*updated_plane_count=*/1);
    const s64 edge_count = matrix.total_edge_count;  // 100
    declare_one_run_per_edge(matrix, {Vector<f32>(static_cast<usize>(edge_count), 1.0f), Vector<f32>(static_cast<usize>(edge_count), 1.0f),
                                      Vector<f32>(static_cast<usize>(edge_count), 0.0f)});

    for (s64 ordinal = 0; ordinal < 19; ordinal += 1) queue_update(matrix, FIRST_STATE, ordinal, 1.0f);
    matrix.compact_pending_deltas();
    EXPECT_FLOAT_EQ(matrix.sparse_delta_occupancy_fraction(), 0.19f);
    EXPECT_FALSE(matrix.is_refit_due(0.2f));

    queue_update(matrix, FIRST_STATE, 19, 1.0f);
    matrix.compact_pending_deltas();
    EXPECT_TRUE(matrix.is_refit_due(0.2f));
    EXPECT_FALSE(matrix.is_refit_due(0.0f)) << "a threshold of zero never refits";
}

// ── refits ──────────────────────────────────────────────────────────────────────

// With state updates on most edges, a refit folds S into the basis, empties S, and keeps every
// plane close to what it read before.
TEST(Weight_Matrix, a_refit_of_synapse_state_empties_S_and_keeps_every_plane_close_to_what_it_read) {
    Structured_Values_Matrix structured_values(/*node_count=*/200, /*out_degree=*/12, /*updated_plane_count=*/2);
    Weight_Matrix &matrix = structured_values.matrix;

    // Every edge's state changes by a source part times a target part.
    const Vector<Pair<s32, s32>> edges = edges_by_ordinal(matrix);
    for (usize ordinal = 0; ordinal < edges.size(); ordinal += 1) {
        const auto &[source, target] = edges[ordinal];
        queue_update(matrix, FIRST_STATE, static_cast<s64>(ordinal),
                     1e-9f * static_cast<f32>(source % 3 + 1) * (static_cast<f32>(target % 4) - 1.5f));
        queue_update(matrix, FIRST_STATE + 1, static_cast<s64>(ordinal),
                     1e-9f * (static_cast<f32>(source % 4) - 1.5f) * static_cast<f32>(target % 3 + 1));
    }
    matrix.compact_pending_deltas();

    Vector<Vector<f32>> before;
    for (s64 plane = 0; plane < 4; plane += 1) before.push_back(plane_values(matrix, plane));
    matrix.refit();

    // A refit is fairly accurate rather than exact: a miss is logged and the run goes on, so it is
    // held to twice fit_tolerance.
    const f64 refit_allowance = 2.0 * static_cast<f64>(matrix.fit_tolerance);
    for (s64 plane = 0; plane < 4; plane += 1) {
        EXPECT_EQ(matrix.sparse_delta_entry_count[static_cast<usize>(plane)], 0) << "plane " << plane;
        EXPECT_LE(worst_relative_error(plane_values(matrix, plane), before[static_cast<usize>(plane)]), refit_allowance)
            << "plane " << plane;
        EXPECT_LE(matrix.measured_fit_error[static_cast<usize>(plane)], refit_allowance) << "plane " << plane;
    }
}

// Plasticity tests are disabled for now: they update the weight plane.
// The same with weight updates on a twentieth of the edges as well.
TEST(Weight_Matrix, DISABLED_a_refit_empties_S_and_keeps_every_plane_within_the_tolerance) {
    Structured_Values_Matrix structured_values(/*node_count=*/200, /*out_degree=*/12, /*updated_plane_count=*/3);
    Weight_Matrix &matrix = structured_values.matrix;
    const s64 edge_count = matrix.total_edge_count;

    Mersenne_Twister random_engine(21);
    Normal_Distribution<f32> state_change(0.0f, 1e-9f);
    Bernoulli_Distribution changes_state(0.9);
    Bernoulli_Distribution changes_weight(0.05);
    for (s64 ordinal = 0; ordinal < edge_count; ordinal += 1) {
        if (changes_state(random_engine)) {
            queue_update(matrix, FIRST_STATE, ordinal, state_change(random_engine));
            queue_update(matrix, FIRST_STATE + 1, ordinal, state_change(random_engine));
        }
        if (changes_weight(random_engine)) queue_update(matrix, WEIGHT, ordinal, 0.01f);
    }
    matrix.compact_pending_deltas();

    Vector<Vector<f32>> before;
    for (s64 plane = 0; plane < 4; plane += 1) before.push_back(plane_values(matrix, plane));
    matrix.refit();

    for (s64 plane = 0; plane < 4; plane += 1) {
        EXPECT_EQ(matrix.sparse_delta_entry_count[static_cast<usize>(plane)], 0) << "plane " << plane;
        EXPECT_LE(worst_relative_error(plane_values(matrix, plane), before[static_cast<usize>(plane)]), matrix.fit_tolerance)
            << "plane " << plane;
        EXPECT_LE(matrix.measured_fit_error[static_cast<usize>(plane)], matrix.fit_tolerance) << "plane " << plane;
    }
}

// One NaN or infinite value cannot be held. It must not set the scale of the plane's other
// edges, and the fit must report it instead of a perfect fit.
TEST(Weight_Matrix, a_non_finite_value_on_one_edge_is_reported_and_leaves_the_others_alone) {
    for (const f32 bad_value : {Numeric_Limits<f32>::quiet_NaN(), Numeric_Limits<f32>::infinity()}) {
        Captured_Log captured_log;
        Structured_Values_Matrix structured_values(/*node_count=*/200, /*out_degree=*/12, /*updated_plane_count=*/1);
        Weight_Matrix &matrix = structured_values.matrix;
        const Vector<f32> before = plane_values(matrix, WEIGHT);

        // Through S and a refit, the way a kernel update arrives.
        queue_update(matrix, WEIGHT, 5, bad_value);
        matrix.compact_pending_deltas();
        matrix.refit();

        const Vector<f32> after = plane_values(matrix, WEIGHT);
        Vector<f32> other_before = before;
        Vector<f32> other_after = after;
        other_before.erase(other_before.begin() + 5);
        other_after.erase(other_after.begin() + 5);
        EXPECT_LE(worst_relative_error(other_after, other_before), matrix.fit_tolerance) << bad_value;
        for (f32 value : other_after) ASSERT_TRUE(std::isfinite(value)) << bad_value;
        EXPECT_TRUE(std::isinf(matrix.measured_fit_error[WEIGHT])) << bad_value;
        EXPECT_GT(matrix.worst_fit_error(), matrix.fit_tolerance) << bad_value;
        EXPECT_TRUE(captured_log.contains("cannot hold")) << bad_value;

        EXPECT_THROW(matrix.accumulate_edge_delta(WEIGHT, 0, random_network(200, 12, 11)[0][0], bad_value),
                     Invalid_Argument);

        // Declared, on the per-plane fit.
        Weight_Matrix declared(shared_backend(), random_network(200, 12, 11), -1, true, -1, 7);
        Vector<f32> weights = values_per_edge(declared, structured_weight);
        weights[3] = bad_value;
        declare_one_run_per_edge(declared, {weights, values_per_edge(declared, structured_delay)});
        const Vector<f32> read = plane_values(declared, WEIGHT);
        Vector<f32> other_read = read;
        Vector<f32> other_weights = weights;
        other_read.erase(other_read.begin() + 3);
        other_weights.erase(other_weights.begin() + 3);
        EXPECT_LE(worst_relative_error(other_read, other_weights), declared.fit_tolerance) << bad_value;
        EXPECT_TRUE(std::isinf(declared.measured_fit_error[WEIGHT])) << bad_value;
    }
}

// A refit changes only lanes no other plane reads, so planes without updates stay bit-identical.
TEST(Weight_Matrix, refitting_one_plane_leaves_every_other_plane_bit_identical) {
    Structured_Values_Matrix structured_values(/*node_count=*/150, /*out_degree=*/10, /*updated_plane_count=*/1);
    Weight_Matrix &matrix = structured_values.matrix;

    Vector<Vector<f32>> before;
    for (s64 plane = 0; plane < 4; plane += 1) before.push_back(plane_values(matrix, plane));

    for (s64 ordinal = 0; ordinal < matrix.total_edge_count; ordinal += 2) {
        queue_update(matrix, FIRST_STATE, ordinal, 1e-9f * static_cast<f32>(ordinal % 7));
    }
    matrix.compact_pending_deltas();
    matrix.refit();

    for (s64 plane : {WEIGHT, DELAY, FIRST_STATE + 1}) {
        EXPECT_EQ(plane_values(matrix, plane), before[static_cast<usize>(plane)]) << "plane " << plane;
    }
}

TEST(Weight_Matrix, a_refit_with_no_updates_changes_nothing) {
    Structured_Values_Matrix structured_values(/*node_count=*/100, /*out_degree=*/8, /*updated_plane_count=*/0);
    Weight_Matrix &matrix = structured_values.matrix;
    const s64 rank_before = matrix.rank;

    Vector<Vector<f32>> before;
    for (s64 plane = 0; plane < 4; plane += 1) before.push_back(plane_values(matrix, plane));
    matrix.refit();

    EXPECT_EQ(matrix.rank, rank_before);
    for (s64 plane = 0; plane < 4; plane += 1) {
        EXPECT_EQ(plane_values(matrix, plane), before[static_cast<usize>(plane)]) << "plane " << plane;
    }
}

// Plasticity's pattern: weight updates on a few nodes' edges, then a refit, repeated. Every refit
// must stay fast, and the weights must not drift from the running total of the updates.
// Plasticity tests are disabled for now: they update the weight plane.
TEST(Weight_Matrix, DISABLED_repeated_weight_updates_stay_fast_and_within_the_tolerance) {
    Structured_Values_Matrix structured_values(/*node_count=*/200, /*out_degree=*/12, /*updated_plane_count=*/1);
    Weight_Matrix &matrix = structured_values.matrix;
    const Vector<Pair<s32, s32>> edges = edges_by_ordinal(matrix);

    Vector<f32> running_total = plane_values(matrix, WEIGHT);
    Mersenne_Twister random_engine(31);
    Bernoulli_Distribution spiked(0.1);
    Normal_Distribution<f32> step(0.01f, 0.003f);
    for (s32 cycle = 0; cycle < 10; cycle += 1) {
        Vector<bool> node_spiked(static_cast<usize>(matrix.node_count));
        for (usize node = 0; node < node_spiked.size(); node += 1) node_spiked[node] = spiked(random_engine);
        for (usize ordinal = 0; ordinal < edges.size(); ordinal += 1) {
            if (!node_spiked[static_cast<usize>(edges[ordinal].first)] && !node_spiked[static_cast<usize>(edges[ordinal].second)]) continue;
            const f32 delta = node_spiked[static_cast<usize>(edges[ordinal].second)] ? step(random_engine) : -step(random_engine);
            queue_update(matrix, WEIGHT, static_cast<s64>(ordinal), delta);
            running_total[ordinal] += delta;
        }
        matrix.compact_pending_deltas();

        const auto started = Steady_Clock::now();
        matrix.refit();
        const f64 seconds = spikecorec::Duration<f64>(Steady_Clock::now() - started).count();
        // 2,400 edges refit in about 4 ms; the bound only catches a refit gone slow.
        EXPECT_LT(seconds, 2.0) << "cycle " << cycle;
        EXPECT_LE(worst_relative_error(plane_values(matrix, WEIGHT), running_total), matrix.fit_tolerance)
            << "cycle " << cycle;
    }
}

// ── whole-network reads ─────────────────────────────────────────────────────────

// One value per real edge, by ordinal: no padding rows and no sentinels.
TEST(Weight_Matrix, neighbor_weights_is_indexed_by_edge_ordinal) {
    Weight_Matrix matrix(shared_backend(), small_network());
    const Vector<f32> weights = {0.25f, 1.75f, 0.001f};
    declare_one_run_per_edge(matrix, {weights, {1.0f, 1.0f, 1.0f}});

    Vector<f32> edge_weights(static_cast<usize>(matrix.total_edge_count), -1.0f);
    matrix.neighbor_weights(edge_weights.data());
    for (usize ordinal = 0; ordinal < weights.size(); ordinal += 1) {
        EXPECT_NEAR(edge_weights[ordinal], weights[ordinal], fit_allowance(matrix.fit_tolerance, weights[ordinal], weights))
            << "ordinal " << ordinal;
    }
}

TEST(Weight_Matrix, statistics_are_taken_over_edges_not_padded_slots) {
    Weight_Matrix matrix(shared_backend(), small_network());
    const Vector<f32> weights = {1.0f, 2.0f, 3.0f};
    declare_one_run_per_edge(matrix, {weights, {1.0f, 1.0f, 1.0f}});

    const Weight_Stats statistics = matrix.neighbor_weight_stats();
    EXPECT_NEAR(statistics.mean, 2.0f, fit_allowance(matrix.fit_tolerance, 2.0, weights));
    EXPECT_NEAR(statistics.min_value, 1.0f, fit_allowance(matrix.fit_tolerance, 1.0, weights));
    EXPECT_NEAR(statistics.max_value, 3.0f, fit_allowance(matrix.fit_tolerance, 3.0, weights));
}

// A tight spread around a large mean: the standard deviation must not cancel away in f32.
TEST(Weight_Matrix, the_standard_deviation_of_tight_weights_is_accurate) {
    Weight_Matrix matrix(shared_backend(), random_network(/*node_count=*/512, /*out_degree=*/20, 17));
    const s64 edge_count = matrix.total_edge_count;
    const Vector<f32> weights = uniform_values(edge_count, 1.0f - 1e-3f, 1.0f + 1e-3f, 5);
    declare_one_run_per_edge(matrix, {weights, Vector<f32>(static_cast<usize>(edge_count), 1.0f)});

    const Vector<f32> read = plane_values(matrix, WEIGHT);
    f64 mean = 0.0;
    for (f32 weight : read) mean += static_cast<f64>(weight);
    mean /= static_cast<f64>(edge_count);
    f64 variance = 0.0;
    for (f32 weight : read) variance += (static_cast<f64>(weight) - mean) * (static_cast<f64>(weight) - mean);
    const f64 expected_standard_deviation = sqrt(variance / static_cast<f64>(edge_count));

    const Weight_Stats statistics = matrix.neighbor_weight_stats();
    EXPECT_NEAR(statistics.standard_deviation, expected_standard_deviation, 0.01 * expected_standard_deviation);
    EXPECT_NEAR(statistics.mean, mean, 1e-6);
}

// ── storage ─────────────────────────────────────────────────────────────────────

// Nothing may be sized by node_count * max_neighbor_count. A hub makes that padded size far larger
// than the real edge count.
TEST(Weight_Matrix, nothing_is_sized_by_the_padded_neighbour_count) {
    Vector<Vector<s32>> hub_network(static_cast<usize>(64));
    for (s32 target = 1; target < 64; target += 1) hub_network[0].push_back(target);
    for (s32 source = 1; source < 64; source += 1) hub_network[static_cast<usize>(source)].push_back(0);

    Weight_Matrix matrix(shared_backend(), hub_network);
    ASSERT_EQ(matrix.max_neighbor_count, 63);
    ASSERT_EQ(matrix.total_edge_count, 126);

    const u64 padded_plane_bytes = static_cast<u64>(matrix.node_count) * static_cast<u64>(matrix.max_neighbor_count) * sizeof(f32);
    EXPECT_LT(matrix.U_matrix.total_bytes, padded_plane_bytes);
    EXPECT_LT(matrix.V_matrix.total_bytes, padded_plane_bytes);
    EXPECT_EQ(matrix.edge_row_offset.total_bytes, static_cast<u64>(matrix.node_count + 1) * sizeof(s64));
}

// ── lifetime ────────────────────────────────────────────────────────────────────

// A move hands the slab over, so only one object releases it.
TEST(Weight_Matrix, moving_transfers_the_slab_exactly_once) {
    Weight_Matrix original(shared_backend(), small_network());
    declare_one_run_per_edge(original, {{0.25f, 1.75f, 0.001f}, {1.0f, 1.0f, 1.0f}});

    const f32 value = original.get(0, 1);
    Weight_Matrix moved = std::move(original);
    EXPECT_EQ(moved.get(0, 1), value);
    EXPECT_EQ(original.owning_backend, nullptr);

    Weight_Matrix destination(shared_backend(), small_network());
    destination = std::move(moved);
    EXPECT_EQ(destination.get(0, 1), value);
    EXPECT_EQ(moved.owning_backend, nullptr);
}

TEST(Weight_Matrix, the_default_matrix_owns_nothing_and_destructs_cleanly) {
    Weight_Matrix empty;
    EXPECT_EQ(empty.node_count, 0);
    EXPECT_EQ(empty.total_edge_count, 0);
    EXPECT_EQ(empty.owning_backend, nullptr);
    EXPECT_TRUE(empty.U_matrix.is_empty());
}

// ── saving ──────────────────────────────────────────────────────────────────────

// The basis and the updates in S both survive a save and a load.
TEST(Weight_Matrix, save_and_load_preserve_every_value) {
    const Temporary_Directory directory;
    const String path = directory.path_of("weights.bin");

    Weight_Matrix saved(shared_backend(), small_network());
    declare_one_run_per_edge(saved, {{0.25f, 1.75f, 0.001f}, {10.0f, 20.0f, 30.0f}});
    saved.accumulate_edge_delta(WEIGHT, 1, 2, 0.5f);
    saved.save(path.c_str());

    Weight_Matrix loaded(shared_backend(), small_network());
    loaded.load_from_disk(path.c_str());
    EXPECT_EQ(loaded.get(0, 1), saved.get(0, 1));
    EXPECT_EQ(loaded.get(0, 2), saved.get(0, 2));
    EXPECT_EQ(loaded.get(1, 2), saved.get(1, 2));
    EXPECT_NEAR(loaded.get(1, 2), 0.501, fit_allowance(loaded.fit_tolerance, 0.001, {0.25f, 1.75f, 0.001f}));
    EXPECT_EQ(loaded.get_edge_delay_ticks(0, 1), 10);
    EXPECT_EQ(loaded.get_edge_delay_ticks(1, 2), 30);
}

// A file from another topology, or a truncated one, is refused and leaves the matrix as it was.
TEST(Weight_Matrix, a_file_from_another_network_is_refused) {
    const Temporary_Directory directory;
    const String path = directory.path_of("weights.bin");

    Weight_Matrix saved(shared_backend(), {{1, 2}, {2}, {3}, {}});
    declare_one_run_per_edge(saved, {{0.25f, 1.75f, 0.5f, 2.0f}, {1.0f, 1.0f, 1.0f, 1.0f}});
    saved.accumulate_edge_delta(WEIGHT, 2, 3, 0.5f);
    saved.save(path.c_str());

    Weight_Matrix other(shared_backend(), {{}, {}, {}, {0}});
    declare_one_run_per_edge(other, {{0.75f}, {4.0f}});
    const f32 other_value = other.get(3, 0);
    EXPECT_THROW(other.load_from_disk(path.c_str()), Invalid_Argument);
    EXPECT_EQ(other.get(3, 0), other_value);
    EXPECT_EQ(other.sparse_delta_entry_count[WEIGHT], 0);

    const String truncated_path = directory.path_of("truncated.bin");
    {
        Input_File_Stream source(path, Stream_Base::binary);
        String bytes((Input_Stream_Buffer_Iterator<char>(source)), Input_Stream_Buffer_Iterator<char>());
        Output_File_Stream truncated(truncated_path, Stream_Base::binary);
        truncated.write(bytes.data(), static_cast<std::streamsize>(bytes.size() - 6));
    }
    Weight_Matrix same_network(shared_backend(), {{1, 2}, {2}, {3}, {}});
    declare_one_run_per_edge(same_network, {{0.25f, 1.75f, 0.5f, 2.0f}, {1.0f, 1.0f, 1.0f, 1.0f}});
    EXPECT_THROW(same_network.load_from_disk(truncated_path.c_str()), Invalid_Argument);
    same_network.load_from_disk(path.c_str());
    EXPECT_EQ(same_network.get(2, 3), saved.get(2, 3));
}
