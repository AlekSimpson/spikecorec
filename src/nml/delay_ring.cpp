#include "spikecorec/nml/delay_ring.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "spikecorec/core/log.h"

using namespace std;

namespace spikecorec::nml {

namespace {

void fail(const String &message) { log::throw_runtime_error(log::logger(), "delay_ring: " + message); }

// Converts one connection's own delay (SI seconds, already unit-resolved by resolve.cpp's own
// unit_value_to_si -- see model_specification.cpp) to whole ticks against `dt_seconds`, floored to
// the engine's existing minimum one-tick latency (arch §4.4 / IR spec §3.5). A plain `connection`
// with no `delay` attribute at all (ConnectionEntry::delay == 0.0, model_specification.cpp's own
// default) lands here exactly, reproducing today's implicit one-tick latency as this subsystem's own
// zero-delay case.
s64 delay_seconds_to_ticks(f64 delay_seconds, f32 dt_seconds) {
    if (dt_seconds <= 0.0f) fail("dt_seconds must be > 0");
    s64 ticks = (s64)std::llround(delay_seconds / (f64)dt_seconds);
    return ticks < 1 ? 1 : ticks;
}

} // namespace

s64 compute_max_delay_ticks(const ModelSpecification &model, f32 dt_seconds) {
    s64 max_ticks = 1; // the engine's existing implicit minimum, even with no connections at all
    for (const auto &projection : model.projections) {
        for (const auto &connection : projection.connections) {
            s64 ticks = delay_seconds_to_ticks(connection.delay, dt_seconds);
            if (ticks > max_ticks) max_ticks = ticks;
        }
    }
    return max_ticks;
}

DelayRingAllocation::~DelayRingAllocation() {
    deallocate(std::move(input_ring));
    deallocate(std::move(edge_delay_ticks));
    deallocate(std::move(pending_active_neuron_indices));
    deallocate(std::move(pending_active_neuron_count));
    deallocate(std::move(pending_active_generation));
}

DelayRingAllocation allocate_delay_ring(const ModelSpecification &model, const WeightMatrix &weights,
                                         f32 dt_seconds) {
    DelayRingAllocation result;
    result.neuron_count = model.total_neuron_count;
    result.node_count = weights.node_count;
    result.max_neighbor_count = weights.max_neighbor_count;
    result.ring_slot_count = compute_max_delay_ticks(model, dt_seconds) + 1;

    s64 ring_element_count = result.ring_slot_count * result.neuron_count;
    s64 edge_element_count = result.node_count * result.max_neighbor_count;

    result.input_ring = allocate<f32>((usize)ring_element_count * sizeof(f32));
    memset(result.input_ring.get_contents(), 0, (usize)ring_element_count * sizeof(f32));

    result.pending_active_neuron_indices = allocate<s32>((usize)ring_element_count * sizeof(s32));
    memset(result.pending_active_neuron_indices.get_contents(), 0, (usize)ring_element_count * sizeof(s32));

    result.pending_active_neuron_count = allocate<s32>((usize)result.ring_slot_count * sizeof(s32));
    memset(result.pending_active_neuron_count.get_contents(), 0, (usize)result.ring_slot_count * sizeof(s32));

    result.pending_active_generation = allocate<s32>((usize)ring_element_count * sizeof(s32));
    s32 *generation_contents = result.pending_active_generation.get_contents();
    for (s64 index = 0; index < ring_element_count; ++index) generation_contents[index] = -1;

    // edge_delay_ticks: skip the allocation entirely for an edge-free model (matches
    // WeightMatrix::allocate_sparse_delta_buffer's own "avoid a zero-byte GPU allocation" contract)
    // -- there is no neighbor slot any propagate kernel could ever look one up at.
    if (edge_element_count > 0) {
        result.edge_delay_ticks = allocate<s32>((usize)edge_element_count * sizeof(s32));
        s32 *edge_delay_contents = result.edge_delay_ticks.get_contents();
        for (s64 index = 0; index < edge_element_count; ++index) edge_delay_contents[index] = 1;

        // Populate real per-edge delays (arch §4.4): walk every connection, locate its k^2-tree
        // neighbor slot the same way WeightMatrix::find_neighbor_slot does (get_neighbors is that
        // method's own public equivalent), and overwrite that slot's default.
        vector<s32> neighbor_buffer((usize)result.max_neighbor_count);
        for (const auto &projection : model.projections) {
            for (const auto &connection : projection.connections) {
                s64 degree = weights.get_neighbors(connection.source_neuron_index, neighbor_buffer.data());
                for (s64 slot = 0; slot < degree; ++slot) {
                    if (neighbor_buffer[(usize)slot] == connection.target_neuron_index) {
                        s64 ticks = delay_seconds_to_ticks(connection.delay, dt_seconds);
                        edge_delay_contents[connection.source_neuron_index * result.max_neighbor_count + slot] =
                            (s32)ticks;
                        break;
                    }
                }
            }
        }
    }

    return result;
}

} // namespace spikecorec::nml
