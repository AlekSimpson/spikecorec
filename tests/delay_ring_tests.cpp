#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <vector>

#include <gtest/gtest.h>

#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/nml/delay_ring.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── Spike-delay subsystem (the "delay ring") host-side sizing/allocation tests (ticket #64 [F3]) ──
//
// Exercises compute_max_delay_ticks()/allocate_delay_ring() in isolation (no AssembledModel/GPU
// dispatch involved) -- the pure "how big is the ring, and what per-edge delay-in-ticks value did
// each connection resolve to" logic. The full acceptance criterion (a delayed edge delivers exactly
// `delay` ticks later, through a real AssembledModel::step_tick call) is exercised end to end in
// master_kernel_tests.cpp's own ticket #64 section.

namespace {

ProjectionEntry make_projection_with_one_connection(s32 source, s32 target, f64 weight, f64 delay_seconds) {
    ProjectionEntry projection;
    projection.id = "TestProjection";
    ConnectionEntry connection;
    connection.source_neuron_index = source;
    connection.target_neuron_index = target;
    connection.weight = weight;
    connection.delay = delay_seconds;
    projection.connections.push_back(connection);
    return projection;
}

} // namespace

TEST(DelayRing, compute_max_delay_ticks_is_one_with_no_connections_at_all) {
    ModelSpecification model;
    model.total_neuron_count = 3;
    EXPECT_EQ(compute_max_delay_ticks(model, /*dt_seconds=*/0.001f), 1);
}

TEST(DelayRing, compute_max_delay_ticks_floors_a_zero_or_absent_delay_to_one_tick) {
    ModelSpecification model;
    model.total_neuron_count = 2;
    model.projections.push_back(make_projection_with_one_connection(0, 1, /*weight=*/1.0, /*delay_seconds=*/0.0));
    EXPECT_EQ(compute_max_delay_ticks(model, /*dt_seconds=*/0.001f), 1);
}

TEST(DelayRing, compute_max_delay_ticks_returns_the_longest_delay_across_every_connection_in_whole_ticks) {
    const f32 dt_seconds = 0.001f; // 1ms per tick
    ModelSpecification model;
    model.total_neuron_count = 4;
    model.projections.push_back(make_projection_with_one_connection(0, 1, 1.0, 0.0));   // -> 1 tick
    model.projections.push_back(make_projection_with_one_connection(0, 2, 1.0, 0.003)); // -> 3 ticks
    model.projections.push_back(make_projection_with_one_connection(0, 3, 1.0, 0.005)); // -> 5 ticks (longest)

    EXPECT_EQ(compute_max_delay_ticks(model, dt_seconds), 5);
}

TEST(DelayRing, allocate_delay_ring_sizes_ring_slot_count_to_max_delay_ticks_plus_one) {
    const f32 dt_seconds = 0.001f;
    vector<vector<s32>> adjacency = {{1, 2, 3}, {}, {}, {}};
    WeightMatrix weights(adjacency, /*rank=*/1);

    ModelSpecification model;
    model.total_neuron_count = 4;
    model.projections.push_back(make_projection_with_one_connection(0, 1, 1.0, 0.0));   // 1 tick
    model.projections.push_back(make_projection_with_one_connection(0, 2, 1.0, 0.003)); // 3 ticks
    model.projections.push_back(make_projection_with_one_connection(0, 3, 1.0, 0.005)); // 5 ticks

    DelayRingAllocation ring = allocate_delay_ring(model, weights, dt_seconds);
    EXPECT_EQ(ring.ring_slot_count, 6); // max_delay_ticks (5) + 1
    EXPECT_EQ(ring.neuron_count, 4);
    EXPECT_EQ(ring.node_count, weights.node_count);
    EXPECT_EQ(ring.max_neighbor_count, weights.max_neighbor_count);
}

// Sizing genuinely SCALES with the longest delay actually present -- not a fixed constant: a model
// whose longest delay is bigger gets a bigger ring, in exact lockstep with compute_max_delay_ticks.
TEST(DelayRing, ring_slot_count_scales_with_the_longest_delay_actually_present) {
    const f32 dt_seconds = 0.001f;
    vector<vector<s32>> adjacency = {{1}, {}};
    WeightMatrix small_delay_weights(adjacency, /*rank=*/1);
    WeightMatrix large_delay_weights(adjacency, /*rank=*/1);

    ModelSpecification small_delay_model;
    small_delay_model.total_neuron_count = 2;
    small_delay_model.projections.push_back(make_projection_with_one_connection(0, 1, 1.0, 0.002)); // 2 ticks

    ModelSpecification large_delay_model;
    large_delay_model.total_neuron_count = 2;
    large_delay_model.projections.push_back(make_projection_with_one_connection(0, 1, 1.0, 0.050)); // 50 ticks

    DelayRingAllocation small_ring = allocate_delay_ring(small_delay_model, small_delay_weights, dt_seconds);
    DelayRingAllocation large_ring = allocate_delay_ring(large_delay_model, large_delay_weights, dt_seconds);

    EXPECT_EQ(small_ring.ring_slot_count, 3);
    EXPECT_EQ(large_ring.ring_slot_count, 51);
    EXPECT_GT(large_ring.ring_slot_count, small_ring.ring_slot_count);
}

TEST(DelayRing, allocate_delay_ring_defaults_every_real_edge_to_one_tick_absent_an_explicit_delay) {
    const f32 dt_seconds = 0.001f;
    vector<vector<s32>> adjacency = {{1}, {}};
    WeightMatrix weights(adjacency, /*rank=*/1);

    ModelSpecification model;
    model.total_neuron_count = 2;
    model.projections.push_back(make_projection_with_one_connection(0, 1, 1.0, /*delay_seconds=*/0.0));

    DelayRingAllocation ring = allocate_delay_ring(model, weights, dt_seconds);

    vector<s32> neighbor_buffer((usize)weights.max_neighbor_count);
    s64 degree = weights.get_neighbors(0, neighbor_buffer.data());
    ASSERT_EQ(degree, 1);
    ASSERT_EQ(neighbor_buffer[0], 1);

    const s32 *edge_delay_contents = ring.edge_delay_ticks.get_contents();
    EXPECT_EQ(edge_delay_contents[0 * weights.max_neighbor_count + 0], 1);
}

TEST(DelayRing, allocate_delay_ring_stores_the_real_per_edge_delay_in_whole_ticks) {
    const f32 dt_seconds = 0.001f; // 1ms per tick
    vector<vector<s32>> adjacency = {{1}, {}};
    WeightMatrix weights(adjacency, /*rank=*/1);

    ModelSpecification model;
    model.total_neuron_count = 2;
    model.projections.push_back(make_projection_with_one_connection(0, 1, 1.0, /*delay_seconds=*/0.005)); // 5 ticks

    DelayRingAllocation ring = allocate_delay_ring(model, weights, dt_seconds);

    vector<s32> neighbor_buffer((usize)weights.max_neighbor_count);
    s64 degree = weights.get_neighbors(0, neighbor_buffer.data());
    ASSERT_EQ(degree, 1);
    ASSERT_EQ(neighbor_buffer[0], 1);

    const s32 *edge_delay_contents = ring.edge_delay_ticks.get_contents();
    EXPECT_EQ(edge_delay_contents[0 * weights.max_neighbor_count + 0], 5);
}

// A connection whose (source, target) pair isn't a representable k^2-tree neighbor of the SUPPLIED
// WeightMatrix (built here from an edge-free adjacency, deliberately mismatched against the model's
// own connection) is silently skipped -- no crash, and every edge_delay_ticks slot stays at its
// default (there are no real edges at all in this WeightMatrix for it to have overwritten).
TEST(DelayRing, allocate_delay_ring_skips_a_connection_that_is_not_a_representable_k2tree_neighbor) {
    const f32 dt_seconds = 0.001f;
    vector<vector<s32>> edge_free_adjacency = {{}, {}};
    WeightMatrix weights(edge_free_adjacency, /*rank=*/1, /*check_indexing=*/true, /*max_neighbor_count=*/4);

    ModelSpecification model;
    model.total_neuron_count = 2;
    model.projections.push_back(make_projection_with_one_connection(0, 1, 1.0, 0.009)); // 9 ticks, if it applied

    EXPECT_NO_THROW({
        DelayRingAllocation ring = allocate_delay_ring(model, weights, dt_seconds);
        const s32 *edge_delay_contents = ring.edge_delay_ticks.get_contents();
        for (s64 index = 0; index < weights.node_count * weights.max_neighbor_count; ++index) {
            EXPECT_EQ(edge_delay_contents[index], 1) << "index=" << index;
        }
    });
}

TEST(DelayRing, allocate_delay_ring_zero_initializes_the_input_ring_and_pending_active_bookkeeping) {
    const f32 dt_seconds = 0.001f;
    vector<vector<s32>> adjacency = {{1}, {}};
    WeightMatrix weights(adjacency, /*rank=*/1);

    ModelSpecification model;
    model.total_neuron_count = 2;
    model.projections.push_back(make_projection_with_one_connection(0, 1, 1.0, 0.003)); // 3 ticks

    DelayRingAllocation ring = allocate_delay_ring(model, weights, dt_seconds);
    s64 ring_element_count = ring.ring_slot_count * ring.neuron_count;

    const f32 *input_ring_contents = ring.input_ring.get_contents();
    for (s64 index = 0; index < ring_element_count; ++index) {
        EXPECT_EQ(input_ring_contents[index], 0.0f) << "index=" << index;
    }

    const s32 *pending_generation_contents = ring.pending_active_generation.get_contents();
    for (s64 index = 0; index < ring_element_count; ++index) {
        EXPECT_EQ(pending_generation_contents[index], -1) << "index=" << index;
    }

    const s32 *pending_count_contents = ring.pending_active_neuron_count.get_contents();
    for (s64 slot = 0; slot < ring.ring_slot_count; ++slot) {
        EXPECT_EQ(pending_count_contents[slot], 0) << "slot=" << slot;
    }
}
