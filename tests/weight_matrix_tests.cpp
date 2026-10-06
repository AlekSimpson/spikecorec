#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/types.h"
#include "spikecorec/core/weight_matrix.h"
#include "support/test_support.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::test_support;

namespace {

constexpr s64 WEIGHT = WeightMatrix::WEIGHT_PLANE;
constexpr s64 DELAY = WeightMatrix::DELAY_PLANE;
constexpr s64 FIRST_STATE = WeightMatrix::FIRST_STATE_VARIABLE_PLANE;

// Two source nodes into one target, plus an unconnected fourth. Small enough to name every edge:
//
//   node 0 -> 1, 2      ordinals 0, 1
//   node 1 -> 2         ordinal  2
//   node 2 ->           (none)
//   node 3 ->           (none)
vector<vector<s32>> small_network() {
    return {{1, 2}, {2}, {}, {}};
}

// Every source reaches out_degree distinct other nodes, chosen at random.
vector<vector<s32>> random_network(s32 node_count, s32 out_degree, unsigned seed) {
    mt19937 random_engine(seed);
    vector<vector<s32>> network((usize)node_count);
    for (s32 source = 0; source < node_count; source += 1) {
        vector<s32> candidates;
        for (s32 target = 0; target < node_count; target += 1) {
            if (target != source) candidates.push_back(target);
        }
        shuffle(candidates.begin(), candidates.end(), random_engine);
        candidates.resize((usize)out_degree);
        sort(candidates.begin(), candidates.end());
        network[(usize)source] = candidates;
    }
    return network;
}

// The (source, target) pair of every edge, indexed by ordinal.
vector<Pair<s32, s32>> edges_by_ordinal(const WeightMatrix &matrix) {
    vector<Pair<s32, s32>> edges((usize)matrix.total_edge_count);
    vector<s32> neighbors((usize)max<s64>(matrix.max_neighbor_count, 1));
    for (s64 source = 0; source < matrix.node_count; source += 1) {
        const s64 degree = matrix.get_neighbors(source, neighbors.data());
        for (s64 slot = 0; slot < degree; slot += 1) {
            const s64 ordinal = *matrix.edge_ordinal((s32)source, neighbors[(usize)slot]);
            edges[(usize)ordinal] = {(s32)source, neighbors[(usize)slot]};
        }
    }
    return edges;
}

// One run per edge, so values_per_plane[plane][ordinal] is that edge's starting value.
void declare_one_run_per_edge(WeightMatrix &matrix, const Vector<Vector<f32>> &values_per_plane) {
    Vector<s64> first_edge_ordinal;
    Vector<s64> edge_count;
    Vector<s32> synapse_prototype;
    for (s64 ordinal = 0; ordinal < matrix.total_edge_count; ordinal += 1) {
        first_edge_ordinal.push_back(ordinal);
        edge_count.push_back(1);
        synapse_prototype.push_back(0);
    }
    matrix.declare_projections(first_edge_ordinal, edge_count, synapse_prototype, values_per_plane);
}

// What every edge of one plane reads, basis plus S, by ordinal.
Vector<f32> plane_values(const WeightMatrix &matrix, s64 plane) {
    Vector<f32> values((usize)matrix.total_edge_count);
    matrix.neighbor_weights_for_matrix(values.data(), plane);
    return values;
}

// The worst error over the edges, against the larger of each expected value and the plane's RMS:
// what fit_tolerance bounds.
f64 worst_relative_error(const Vector<f32> &read, const Vector<f32> &expected) {
    f64 sum_of_squares = 0.0;
    for (f32 value : expected) sum_of_squares += (f64)value * (f64)value;
    const f64 root_mean_square = expected.empty() ? 1.0 : sqrt(sum_of_squares / (f64)expected.size());
    const f64 scale_floor = root_mean_square > 0.0 ? root_mean_square : 1.0;
    f64 worst = 0.0;
    for (usize index = 0; index < expected.size(); index += 1) {
        const f64 scale = max(fabs((f64)expected[index]), scale_floor);
        worst = max(worst, fabs((f64)read[index] - (f64)expected[index]) / scale);
    }
    return worst;
}

// Queues an update the way the kernel does, in the device-side queue that compact_pending_deltas
// merges into S.
void queue_update(WeightMatrix &matrix, s64 plane, s64 ordinal, f32 delta) {
    s32 &count = *matrix.pending_delta_count.get_contents_as<s32>();
    ASSERT_LT(count, matrix.pending_delta_capacity);
    matrix.pending_delta_edge_ordinal.get_contents_as<s64>()[count] = ordinal;
    matrix.pending_delta_value.get_contents_as<f32>()[count] = delta;
    matrix.pending_delta_matrix_index.get_contents_as<s32>()[count] = (s32)plane;
    count += 1;
}

Vector<f32> uniform_values(s64 count, f32 low, f32 high, unsigned seed) {
    mt19937 random_engine(seed);
    uniform_real_distribution<f32> distribution(low, high);
    Vector<f32> values((usize)count);
    for (f32 &value : values) value = distribution(random_engine);
    return values;
}

Vector<f32> whole_tick_delays(s64 count, s32 shortest, s32 longest, unsigned seed) {
    mt19937 random_engine(seed);
    uniform_int_distribution<s32> distribution(shortest, longest);
    Vector<f32> values((usize)count);
    for (f32 &value : values) value = (f32)distribution(random_engine);
    return values;
}

// A matrix whose four planes all start from values drawn per connection, so no lane count short
// of the exact solve reproduces them.
struct RandomValuesMatrix {
    WeightMatrix matrix;
    Vector<Vector<f32>> declared;

    RandomValuesMatrix(s32 node_count, s32 out_degree, s64 updated_plane_count)
        : matrix(shared_backend(), random_network(node_count, out_degree, 11), -1, true, -1, /*weight_seed=*/7,
                 /*matrix_count=*/4, updated_plane_count) {
        const s64 edge_count = matrix.total_edge_count;
        declared = {uniform_values(edge_count, 0.5f, 1.5f, 1), whole_tick_delays(edge_count, 1, 30, 2),
                    uniform_values(edge_count, -2e-9f, 2e-9f, 3), Vector<f32>((usize)edge_count, 0.0f)};
        declare_one_run_per_edge(matrix, declared);
    }
};

} // namespace

// ── edge numbering ──────────────────────────────────────────────────────────────

// The ordinal keys every run table and every update, and it is a prefix sum over real out-degree,
// never over a padded maximum.
TEST(WeightMatrix, edges_are_numbered_by_real_out_degree) {
    WeightMatrix matrix(shared_backend(), small_network());

    EXPECT_EQ(matrix.total_edge_count, 3);
    EXPECT_EQ(matrix.edge_ordinal(0, 1).value(), 0);
    EXPECT_EQ(matrix.edge_ordinal(0, 2).value(), 1);
    EXPECT_EQ(matrix.edge_ordinal(1, 2).value(), 2);

    // Node 2 has an incoming edge but no outgoing one, node 3 has neither, and (1, 0) is no edge.
    EXPECT_FALSE(matrix.edge_ordinal(2, 0).has_value());
    EXPECT_FALSE(matrix.edge_ordinal(3, 0).has_value());
    EXPECT_FALSE(matrix.edge_ordinal(1, 0).has_value());
}

TEST(WeightMatrix, edge_numbering_is_contiguous_and_covers_every_edge) {
    WeightMatrix matrix(shared_backend(), random_network(40, 6, 5));

    vector<bool> seen((usize)matrix.total_edge_count, false);
    for (const auto &[source, target] : edges_by_ordinal(matrix)) {
        const s64 ordinal = *matrix.edge_ordinal(source, target);
        EXPECT_FALSE(seen[(usize)ordinal]) << "ordinal " << ordinal << " used twice";
        seen[(usize)ordinal] = true;
    }
    EXPECT_EQ(count(seen.begin(), seen.end(), true), matrix.total_edge_count);
}

// A hub with an in-degree far above the average: the buffer get_predecessors writes into is
// bounded by the largest in-degree, not the largest out-degree.
TEST(WeightMatrix, get_predecessors_lists_every_source_into_a_node) {
    const s32 node_count = 64;
    vector<vector<s32>> star((usize)node_count);
    for (s32 source = 1; source < node_count; source += 1) star[(usize)source].push_back(0);
    star[0].push_back(1);

    WeightMatrix matrix(shared_backend(), star);
    EXPECT_EQ(matrix.max_neighbor_count, 1);
    EXPECT_EQ(matrix.max_predecessor_count, node_count - 1);

    vector<s32> buffer((usize)matrix.max_predecessor_count);
    const s64 found_count = matrix.get_predecessors(0, buffer.data());
    vector<s32> found(buffer.begin(), buffer.begin() + found_count);
    sort(found.begin(), found.end());
    vector<s32> expected((usize)(node_count - 1));
    iota(expected.begin(), expected.end(), 1);
    EXPECT_EQ(found, expected);

    EXPECT_EQ(matrix.get_predecessors(1, buffer.data()), 1);
    EXPECT_EQ(buffer[0], 0);
}

// ── declared values ─────────────────────────────────────────────────────────────

// A few distinct combinations of values get one lane each, so every plane reads exactly what was
// declared, at no fitting cost.
TEST(WeightMatrix, projection_values_reconstruct_exactly_on_every_plane) {
    WeightMatrix matrix(shared_backend(), small_network(), -1, true, -1, 7, /*matrix_count=*/4);
    const Vector<Vector<f32>> declared = {{0.25f, 1.75f, 0.001f}, {10.0f, 20.0f, 30.0f},
                                          {-1.0f, 0.0f, 3e-3f}, {0.0f, 0.0f, 0.0f}};
    declare_one_run_per_edge(matrix, declared);

    const vector<Pair<s32, s32>> edges = edges_by_ordinal(matrix);
    for (s64 plane = 0; plane < 4; plane += 1) {
        for (usize ordinal = 0; ordinal < edges.size(); ordinal += 1) {
            EXPECT_FLOAT_EQ(matrix.get_for_matrix(edges[ordinal].first, edges[ordinal].second, plane),
                            declared[(usize)plane][ordinal])
                << "plane " << plane << ", ordinal " << ordinal;
        }
        EXPECT_FLOAT_EQ(matrix.measured_fit_error[(usize)plane], 0.0f) << "plane " << plane;
    }
    EXPECT_FLOAT_EQ(matrix.get(0, 2), 1.75f);
}

// Synaptic weights sit at 1e-9 and below. Scale belongs to the basis, so it survives any magnitude.
TEST(WeightMatrix, tiny_values_survive_the_basis) {
    WeightMatrix matrix(shared_backend(), small_network());
    declare_one_run_per_edge(matrix, {{5.0e-10f, 1.0e-12f, 2.5e-9f}, {1.0f, 1.0f, 1.0f}});

    EXPECT_FLOAT_EQ(matrix.get(0, 1), 5.0e-10f);
    EXPECT_FLOAT_EQ(matrix.get(0, 2), 1.0e-12f);
    EXPECT_FLOAT_EQ(matrix.get(1, 2), 2.5e-9f);
}

// A delay is used as a whole number of ticks, at least one.
TEST(WeightMatrix, delays_land_on_their_ticks) {
    WeightMatrix matrix(shared_backend(), small_network());
    declare_one_run_per_edge(matrix, {{1.0f, 1.0f, 1.0f}, {10.0f, 0.4f, 30.0f}});

    EXPECT_EQ(matrix.get_edge_delay_ticks(0, 1), 10);
    EXPECT_EQ(matrix.get_edge_delay_ticks(0, 2), 1);
    EXPECT_EQ(matrix.get_edge_delay_ticks(1, 2), 30);
}

// The prototype selects a switch case in the kernel, so it comes from the run table and never from
// a reconstruction.
TEST(WeightMatrix, synapse_prototype_comes_from_the_run_table) {
    WeightMatrix matrix(shared_backend(), small_network());
    // Ordinals 0-1 use prototype 0 and ordinal 2 uses prototype 1, with different weights.
    matrix.declare_projections({0, 2}, {2, 1}, {0, 1}, {{0.25f, 0.75f}, {1.0f, 1.0f}});

    EXPECT_EQ(matrix.get_edge_synapse_prototype(0, 1), 0);
    EXPECT_EQ(matrix.get_edge_synapse_prototype(0, 2), 0);
    EXPECT_EQ(matrix.get_edge_synapse_prototype(1, 2), 1);
    EXPECT_EQ(matrix.get_edge_synapse_prototype(3, 0), -1);
    EXPECT_FLOAT_EQ(matrix.get(0, 1), 0.25f);
    EXPECT_FLOAT_EQ(matrix.get(1, 2), 0.75f);
}

// Values drawn per connection have no structure for a basis to find, so construction falls back on
// each plane's exact solve. Every plane must still meet fit_tolerance and every delay its tick.
TEST(WeightMatrix, per_connection_random_values_meet_the_tolerance_on_every_plane) {
    RandomValuesMatrix random_values(/*node_count=*/200, /*out_degree=*/12, /*updated_plane_count=*/0);
    const WeightMatrix &matrix = random_values.matrix;

    for (s64 plane = 0; plane < 3; plane += 1) {
        const Vector<f32> read = plane_values(matrix, plane);
        EXPECT_LE(worst_relative_error(read, random_values.declared[(usize)plane]), matrix.fit_tolerance)
            << "plane " << plane;
        EXPECT_LE(matrix.measured_fit_error[(usize)plane], matrix.fit_tolerance) << "plane " << plane;
    }

    s64 delays_off_their_tick = 0;
    for (const auto &[source, target] : edges_by_ordinal(matrix)) {
        const s64 ordinal = *matrix.edge_ordinal(source, target);
        const s32 declared_ticks = (s32)random_values.declared[(usize)DELAY][(usize)ordinal];
        delays_off_their_tick += matrix.get_edge_delay_ticks(source, target) != declared_ticks ? 1 : 0;
    }
    EXPECT_EQ(delays_off_their_tick, 0);

    // A plane of zeros needs no lanes and reads exactly zero.
    const Vector<f32> zeros = plane_values(matrix, FIRST_STATE + 1);
    EXPECT_EQ(zeros, Vector<f32>(zeros.size(), 0.0f));
    EXPECT_LE(matrix.rank, MAX_RANK_FLOAT4_STRIDE * WeightMatrix::LANE_GROUP);
}

// ── updates ─────────────────────────────────────────────────────────────────────

// A read is the basis plus S: an update shows at once, on its own edge and plane only, and updates
// to the same edge add up.
TEST(WeightMatrix, an_update_changes_only_its_own_edge_and_plane) {
    WeightMatrix matrix(shared_backend(), small_network(), -1, true, -1, 7, /*matrix_count=*/3);
    declare_one_run_per_edge(matrix, {{0.25f, 1.75f, 0.001f}, {1.0f, 2.0f, 3.0f}, {0.0f, 0.0f, 0.0f}});

    matrix.accumulate_edge_delta(WEIGHT, 0, 1, 0.5f);
    EXPECT_FLOAT_EQ(matrix.get(0, 1), 0.75f);
    EXPECT_FLOAT_EQ(matrix.get(0, 2), 1.75f);
    EXPECT_FLOAT_EQ(matrix.get(1, 2), 0.001f);
    EXPECT_FLOAT_EQ(matrix.get_for_matrix(0, 1, DELAY), 1.0f);

    matrix.accumulate_edge_delta(WEIGHT, 0, 1, -0.25f);
    EXPECT_FLOAT_EQ(matrix.get(0, 1), 0.5f);

    matrix.accumulate_edge_delta(FIRST_STATE, 1, 2, 4.0f);
    EXPECT_FLOAT_EQ(matrix.get_for_matrix(1, 2, FIRST_STATE), 4.0f);
    EXPECT_FLOAT_EQ(matrix.get_for_matrix(0, 1, FIRST_STATE), 0.0f);
    EXPECT_EQ(matrix.sparse_delta_entry_count[(usize)WEIGHT], 1);
    EXPECT_EQ(matrix.sparse_delta_entry_count[(usize)FIRST_STATE], 1);
}

// The kernel queues updates; compact_pending_deltas sums each edge's into one entry of S, drops any
// that came back to exactly zero, and empties the queue.
TEST(WeightMatrix, compact_pending_deltas_merges_the_queue_into_S) {
    WeightMatrix matrix(shared_backend(), small_network(), -1, true, -1, 7, /*matrix_count=*/4,
                        /*updated_plane_count=*/2);
    declare_one_run_per_edge(matrix, {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
                                      {0.0f, 0.0f, 0.0f}});

    queue_update(matrix, FIRST_STATE, 0, 1.0f);
    queue_update(matrix, FIRST_STATE, 0, 0.5f);
    queue_update(matrix, FIRST_STATE + 1, 1, 2.0f);
    queue_update(matrix, FIRST_STATE + 1, 1, -2.0f);
    queue_update(matrix, FIRST_STATE + 1, 2, -3.0f);
    matrix.compact_pending_deltas();

    EXPECT_EQ(*matrix.pending_delta_count.get_contents_as<s32>(), 0);
    EXPECT_EQ(matrix.sparse_delta_entry_count[(usize)FIRST_STATE], 1);
    EXPECT_EQ(matrix.sparse_delta_entry_count[(usize)(FIRST_STATE + 1)], 1);
    EXPECT_FLOAT_EQ(matrix.get_for_matrix(0, 1, FIRST_STATE), 1.5f);
    EXPECT_FLOAT_EQ(matrix.get_for_matrix(0, 2, FIRST_STATE + 1), 0.0f);
    EXPECT_FLOAT_EQ(matrix.get_for_matrix(1, 2, FIRST_STATE + 1), -3.0f);

    // A later merge adds onto what S already holds.
    queue_update(matrix, FIRST_STATE, 0, 0.25f);
    matrix.compact_pending_deltas();
    EXPECT_FLOAT_EQ(matrix.get_for_matrix(0, 1, FIRST_STATE), 1.75f);
}

// S is sized to the refit threshold, a fifth of the edges by default, plus the plasticity reserve.
TEST(WeightMatrix, S_is_sized_to_the_refit_threshold) {
    EXPECT_FLOAT_EQ(WeightMatrix::DEFAULT_REFIT_OCCUPANCY_THRESHOLD_FRACTION, 0.2f);

    WeightMatrix matrix(shared_backend(), random_network(50, 7, 3));
    matrix.plasticity_reserve_entries = 16;
    declare_one_run_per_edge(matrix, {uniform_values(matrix.total_edge_count, 0.5f, 1.5f, 4),
                                      Vector<f32>((usize)matrix.total_edge_count, 1.0f)});

    // The threshold is an f32, so 0.2 is 0.2000000030 and a fifth of these 350 edges rounds up to 71.
    const s64 edge_count = matrix.total_edge_count;
    EXPECT_EQ(matrix.sparse_delta_capacity, (s64)ceil((f64)0.2f * (f64)edge_count) + 16);
    EXPECT_EQ(matrix.delta_capacity_for_threshold(0.5f), (s64)ceil(0.5 * (f64)edge_count) + 16);
    // Without a threshold S must be able to hold every edge.
    EXPECT_EQ(matrix.delta_capacity_for_threshold(0.0f), edge_count + 16);
}

// When a merge would overflow S, S grows instead: no update is dropped.
TEST(WeightMatrix, S_grows_rather_than_dropping_an_update) {
    WeightMatrix matrix(shared_backend(), random_network(30, 5, 6), -1, true, -1, 7, /*matrix_count=*/3,
                        /*updated_plane_count=*/1);
    const s64 edge_count = matrix.total_edge_count;
    declare_one_run_per_edge(matrix, {Vector<f32>((usize)edge_count, 1.0f), Vector<f32>((usize)edge_count, 1.0f),
                                      Vector<f32>((usize)edge_count, 0.0f)});
    matrix.resize_delta_capacity(1);

    for (s64 ordinal = 0; ordinal < edge_count; ordinal += 1) {
        queue_update(matrix, FIRST_STATE, ordinal, (f32)(ordinal + 1));
    }
    matrix.compact_pending_deltas();

    EXPECT_GE(matrix.sparse_delta_capacity, edge_count);
    EXPECT_EQ(matrix.sparse_delta_entry_count[(usize)FIRST_STATE], edge_count);
    Vector<f32> expected((usize)edge_count);
    iota(expected.begin(), expected.end(), 1.0f);
    EXPECT_EQ(plane_values(matrix, FIRST_STATE), expected);
}

TEST(WeightMatrix, a_refit_is_due_once_the_fullest_plane_reaches_the_threshold) {
    WeightMatrix matrix(shared_backend(), random_network(20, 5, 8), -1, true, -1, 7, /*matrix_count=*/3,
                        /*updated_plane_count=*/1);
    const s64 edge_count = matrix.total_edge_count;  // 100
    declare_one_run_per_edge(matrix, {Vector<f32>((usize)edge_count, 1.0f), Vector<f32>((usize)edge_count, 1.0f),
                                      Vector<f32>((usize)edge_count, 0.0f)});

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

// Synapse state changes on most edges: a refit folds S into the basis, empties S, and leaves every
// plane within fit_tolerance of what it read before.
TEST(WeightMatrix, a_refit_empties_S_and_keeps_every_plane_within_the_tolerance) {
    RandomValuesMatrix random_values(/*node_count=*/200, /*out_degree=*/12, /*updated_plane_count=*/3);
    WeightMatrix &matrix = random_values.matrix;
    const s64 edge_count = matrix.total_edge_count;

    mt19937 random_engine(21);
    normal_distribution<f32> state_change(0.0f, 1e-9f);
    bernoulli_distribution changes_state(0.9);
    bernoulli_distribution changes_weight(0.05);
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
        EXPECT_EQ(matrix.sparse_delta_entry_count[(usize)plane], 0) << "plane " << plane;
        EXPECT_LE(worst_relative_error(plane_values(matrix, plane), before[(usize)plane]), matrix.fit_tolerance)
            << "plane " << plane;
        EXPECT_LE(matrix.measured_fit_error[(usize)plane], matrix.fit_tolerance) << "plane " << plane;
    }
}

// A refit changes only lanes no other plane reads, so the planes without updates come out of it
// bit for bit as they went in.
TEST(WeightMatrix, refitting_one_plane_leaves_every_other_plane_bit_identical) {
    RandomValuesMatrix random_values(/*node_count=*/150, /*out_degree=*/10, /*updated_plane_count=*/1);
    WeightMatrix &matrix = random_values.matrix;

    Vector<Vector<f32>> before;
    for (s64 plane = 0; plane < 4; plane += 1) before.push_back(plane_values(matrix, plane));

    for (s64 ordinal = 0; ordinal < matrix.total_edge_count; ordinal += 2) {
        queue_update(matrix, FIRST_STATE, ordinal, 1e-9f * (f32)(ordinal % 7));
    }
    matrix.compact_pending_deltas();
    matrix.refit();

    for (s64 plane : {WEIGHT, DELAY, FIRST_STATE + 1}) {
        EXPECT_EQ(plane_values(matrix, plane), before[(usize)plane]) << "plane " << plane;
    }
}

TEST(WeightMatrix, a_refit_with_no_updates_changes_nothing) {
    RandomValuesMatrix random_values(/*node_count=*/100, /*out_degree=*/8, /*updated_plane_count=*/0);
    WeightMatrix &matrix = random_values.matrix;
    const s64 rank_before = matrix.rank;

    Vector<Vector<f32>> before;
    for (s64 plane = 0; plane < 4; plane += 1) before.push_back(plane_values(matrix, plane));
    matrix.refit();

    EXPECT_EQ(matrix.rank, rank_before);
    for (s64 plane = 0; plane < 4; plane += 1) {
        EXPECT_EQ(plane_values(matrix, plane), before[(usize)plane]) << "plane " << plane;
    }
}

// Plasticity's pattern: weight updates on the edges of a few nodes, then a refit, over and over.
// Every refit must stay fast, and the weights must not drift from the running total of the updates.
TEST(WeightMatrix, repeated_weight_updates_stay_fast_and_within_the_tolerance) {
    RandomValuesMatrix random_values(/*node_count=*/200, /*out_degree=*/12, /*updated_plane_count=*/1);
    WeightMatrix &matrix = random_values.matrix;
    const vector<Pair<s32, s32>> edges = edges_by_ordinal(matrix);

    Vector<f32> running_total = plane_values(matrix, WEIGHT);
    mt19937 random_engine(31);
    bernoulli_distribution spiked(0.1);
    normal_distribution<f32> step(0.01f, 0.003f);
    for (s32 cycle = 0; cycle < 10; cycle += 1) {
        vector<bool> node_spiked((usize)matrix.node_count);
        for (usize node = 0; node < node_spiked.size(); node += 1) node_spiked[node] = spiked(random_engine);
        for (usize ordinal = 0; ordinal < edges.size(); ordinal += 1) {
            if (!node_spiked[(usize)edges[ordinal].first] && !node_spiked[(usize)edges[ordinal].second]) continue;
            const f32 delta = node_spiked[(usize)edges[ordinal].second] ? step(random_engine) : -step(random_engine);
            queue_update(matrix, WEIGHT, (s64)ordinal, delta);
            running_total[ordinal] += delta;
        }
        matrix.compact_pending_deltas();

        const auto started = chrono::steady_clock::now();
        matrix.refit();
        const f64 seconds = chrono::duration<f64>(chrono::steady_clock::now() - started).count();
        // 2,400 edges took about 4 ms when measured; the bound only catches a refit gone slow.
        EXPECT_LT(seconds, 2.0) << "cycle " << cycle;
        EXPECT_LE(worst_relative_error(plane_values(matrix, WEIGHT), running_total), matrix.fit_tolerance)
            << "cycle " << cycle;
    }
}

// The exact solve needs (largest degree + 4) lanes per plane, all within the 256-lane limit. Three
// planes of degree-80 random values take 252, so the fourth gets 4: it must say it missed the
// tolerance rather than pass quietly, and the other planes must still meet it.
TEST(WeightMatrix, a_plane_without_enough_lanes_warns_that_it_misses_the_tolerance) {
    CapturedLog captured_log;
    WeightMatrix matrix(shared_backend(), random_network(/*node_count=*/128, /*out_degree=*/80, 13), -1, true, -1,
                        7, /*matrix_count=*/4);
    const s64 edge_count = matrix.total_edge_count;
    const Vector<Vector<f32>> declared = {uniform_values(edge_count, 0.5f, 1.5f, 1), whole_tick_delays(edge_count, 1, 30, 2),
                                          uniform_values(edge_count, -1.0f, 1.0f, 3),
                                          uniform_values(edge_count, -1.0f, 1.0f, 4)};
    declare_one_run_per_edge(matrix, declared);

    EXPECT_LE(matrix.rank, MAX_RANK_FLOAT4_STRIDE * WeightMatrix::LANE_GROUP);
    for (s64 plane = 0; plane < 3; plane += 1) {
        EXPECT_LE(matrix.measured_fit_error[(usize)plane], matrix.fit_tolerance) << "plane " << plane;
    }
    EXPECT_GT(matrix.measured_fit_error[3], matrix.fit_tolerance);
    EXPECT_TRUE(captured_log.contains("short of the"));
}

// ── whole-network reads ─────────────────────────────────────────────────────────

// One value per real edge, by ordinal: no padding rows and no sentinels.
TEST(WeightMatrix, neighbor_weights_is_indexed_by_edge_ordinal) {
    WeightMatrix matrix(shared_backend(), small_network());
    declare_one_run_per_edge(matrix, {{0.25f, 1.75f, 0.001f}, {1.0f, 1.0f, 1.0f}});

    vector<f32> edge_weights((usize)matrix.total_edge_count, -1.0f);
    matrix.neighbor_weights(edge_weights.data());
    EXPECT_NEAR(edge_weights[0], 0.25f, 1e-6f);
    EXPECT_NEAR(edge_weights[1], 1.75f, 1e-6f);
    EXPECT_NEAR(edge_weights[2], 0.001f, 1e-6f);
}

TEST(WeightMatrix, statistics_are_taken_over_edges_not_padded_slots) {
    WeightMatrix matrix(shared_backend(), small_network());
    declare_one_run_per_edge(matrix, {{1.0f, 2.0f, 3.0f}, {1.0f, 1.0f, 1.0f}});

    const WeightStats statistics = matrix.neighbor_weight_stats();
    EXPECT_NEAR(statistics.mean, 2.0f, 1e-5f);
    EXPECT_NEAR(statistics.min_value, 1.0f, 1e-5f);
    EXPECT_NEAR(statistics.max_value, 3.0f, 1e-5f);
}

// ── storage ─────────────────────────────────────────────────────────────────────

// Nothing may be sized by node_count * max_neighbor_count. A hub makes the padded size far larger
// than the real edge count, which is where padded storage would show.
TEST(WeightMatrix, nothing_is_sized_by_the_padded_neighbour_count) {
    vector<vector<s32>> hub_network((usize)64);
    for (s32 target = 1; target < 64; target += 1) hub_network[0].push_back(target);
    for (s32 source = 1; source < 64; source += 1) hub_network[(usize)source].push_back(0);

    WeightMatrix matrix(shared_backend(), hub_network);
    ASSERT_EQ(matrix.max_neighbor_count, 63);
    ASSERT_EQ(matrix.total_edge_count, 126);

    const u64 padded_plane_bytes = (u64)matrix.node_count * (u64)matrix.max_neighbor_count * sizeof(f32);
    EXPECT_LT(matrix.U_matrix.total_bytes, padded_plane_bytes);
    EXPECT_LT(matrix.V_matrix.total_bytes, padded_plane_bytes);
    EXPECT_EQ(matrix.neighbor_weight_scratch.total_bytes, (u64)matrix.total_edge_count * sizeof(f32));
    EXPECT_EQ(matrix.edge_row_offset.total_bytes, (u64)(matrix.node_count + 1) * sizeof(s64));
}

// ── lifetime ────────────────────────────────────────────────────────────────────

// The slab moves exactly once, or both objects release it.
TEST(WeightMatrix, moving_transfers_the_slab_exactly_once) {
    WeightMatrix original(shared_backend(), small_network());
    declare_one_run_per_edge(original, {{0.25f, 1.75f, 0.001f}, {1.0f, 1.0f, 1.0f}});

    WeightMatrix moved = std::move(original);
    EXPECT_FLOAT_EQ(moved.get(0, 1), 0.25f);
    EXPECT_EQ(original.owning_backend, nullptr);

    WeightMatrix destination(shared_backend(), small_network());
    destination = std::move(moved);
    EXPECT_FLOAT_EQ(destination.get(0, 1), 0.25f);
    EXPECT_EQ(moved.owning_backend, nullptr);
}

TEST(WeightMatrix, the_default_matrix_owns_nothing_and_destructs_cleanly) {
    WeightMatrix empty;
    EXPECT_EQ(empty.node_count, 0);
    EXPECT_EQ(empty.total_edge_count, 0);
    EXPECT_EQ(empty.owning_backend, nullptr);
    EXPECT_TRUE(empty.U_matrix.is_empty());
}

// ── saving ──────────────────────────────────────────────────────────────────────

// The basis and the updates S still holds both survive a save and a load.
TEST(WeightMatrix, save_and_load_preserve_every_value) {
    const TemporaryDirectory directory;
    const String path = directory.path_of("weights.bin");

    WeightMatrix saved(shared_backend(), small_network());
    declare_one_run_per_edge(saved, {{0.25f, 1.75f, 0.001f}, {10.0f, 20.0f, 30.0f}});
    saved.accumulate_edge_delta(WEIGHT, 1, 2, 0.5f);
    saved.save(path.c_str());

    WeightMatrix loaded(shared_backend(), small_network());
    loaded.load_from_disk(path.c_str());
    EXPECT_FLOAT_EQ(loaded.get(0, 1), 0.25f);
    EXPECT_FLOAT_EQ(loaded.get(0, 2), 1.75f);
    EXPECT_FLOAT_EQ(loaded.get(1, 2), 0.501f);
    EXPECT_EQ(loaded.get_edge_delay_ticks(0, 1), 10);
    EXPECT_EQ(loaded.get_edge_delay_ticks(1, 2), 30);
}
