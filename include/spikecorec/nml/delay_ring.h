#pragma once

#include "spikecorec/core/backend.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/nml/model_specification.h"

namespace spikecorec::nml {

// ── Spike-delay subsystem (the "delay ring") -- ticket #64 [F3]; arch §4.4, §6; IR spec §3.5 ─────
//
// Generalizes `network_inputs`'s existing implicit minimum-one-tick latency (IR spec §3.5) to an
// arbitrary per-edge delay (`ConnectionEntry::delay`, model_specification.h) via a ring of
// `max_delay_ticks + 1` slots (arch §4.4): a scattered spike's contribution lands in the ring slot
// for tick `now + delay_ticks`, not always `now + 1`. An edge with `ConnectionEntry::delay == 0.0`
// (a plain `connection`, no `delay` attribute given) floors to 1 tick, so the ring's zero-delay case
// reproduces Phase-1's existing implicit one-tick latency exactly (arch §4.4's "Early phases keep
// uniform/zero delay").
//
// Granularity: per-edge -- the finest of the three the ticket allows (uniform / per-projection /
// per-edge), and the one `ConnectionEntry::delay` already stores per connection. A uniform or
// per-projection model is simply the special case where every edge happens to carry the same delay
// value, so no separate representation is needed for those.
//
// This does NOT compress into the shared-basis `WeightMatrix` (U/V/Ck/Sk, ticket #52-54): delay is a
// routing/timing property, not a reconstructible weight-like value (see this ticket's own body, and
// weight_matrix.h's own Ck/Sk machinery) -- `edge_delay_ticks` below is its own dedicated per-edge
// array, addressed the SAME way `WeightMatrix::sparse_delta_buffers` already is (row-major by source
// node, same slot order `k2tree.get_neighbors`/`k2t_find_nth_neighbor` enumerate), so the propagate
// kernel's existing neighbor-walk loop can look a delay up with no extra search.
//
// Active-set x delay interaction (arch §4.4/§6): a delayed target must wake at the tick its
// delivery actually lands on, not next tick -- so the existing single next-tick-ahead active-set
// enqueue bookkeeping (`next_active_neuron_indices`/`count`/`active_generation`, master_kernel.cpp)
// is ring-generalized here too: one pending-active list PER ring slot
// (`pending_active_neuron_indices`/`pending_active_neuron_count`), deduplicated per slot via
// `pending_active_generation` (mirrors `active_generation`'s own compare-exchange dedup pattern,
// truncated to a 32-bit tick number for the same reason the existing `active_generation` already is
// -- see master_kernel.cpp's propagate kernel).

struct DelayRingAllocation {
    s64 neuron_count = 0;
    s64 node_count = 0;         // WeightMatrix::node_count this ring's edge_delay_ticks was built
                                 // against -- must match for edge_delay_ticks's addressing to be valid
    s64 max_neighbor_count = 0; // WeightMatrix::max_neighbor_count -- see edge_delay_ticks below
    s64 ring_slot_count = 0;    // max_delay_ticks + 1 (always >= 2 -- minimum delay is 1 tick)

    // Ring-buffered generalization of `network_inputs`: [ring_slot_count * neuron_count],
    // slot-major -- slot s's own [neuron_count] row is exactly what today's flat `network_inputs`
    // holds for the tick that currently maps to slot s (`tick % ring_slot_count`). Zero-initialized.
    GpuPointer<f32> input_ring;

    // Per-edge delay, in whole ticks (>= 1) -- [node_count * max_neighbor_count], addressed
    // [source_node * max_neighbor_count + slot] with `slot` the same k2tree neighbor-enumeration
    // index `k2t_find_nth_neighbor`'s loop already uses (see allocate_delay_ring). Defaults to 1
    // (padding slots beyond a node's real degree are never read -- k2t_find_nth_neighbor returns -1
    // for those before edge_delay_ticks is ever indexed).
    GpuPointer<s32> edge_delay_ticks;

    // Ring-generalized active-set enqueue bookkeeping (arch §4.4/§6) -- one
    // next_active_neuron_indices/count/active_generation triple PER ring slot, replacing the single
    // "next tick only" triple master_kernel.cpp's pre-#64 propagate stage used.
    GpuPointer<s32> pending_active_neuron_indices; // [ring_slot_count * neuron_count]
    GpuPointer<s32> pending_active_neuron_count;   // [ring_slot_count]
    GpuPointer<s32> pending_active_generation;     // [ring_slot_count * neuron_count] -- the (32-bit,
                                                    // truncated) absolute arrival tick last enqueued
                                                    // at this (slot, neuron); -1 = never. Comparing
                                                    // against the CURRENT arrival tick (not just "is
                                                    // this slot dirty") is what correctly tells one
                                                    // lap's entry apart from a stale one left over
                                                    // from `ring_slot_count` ticks ago at the same
                                                    // physical slot.

    DelayRingAllocation() = default;
    DelayRingAllocation(const DelayRingAllocation &) = delete;
    DelayRingAllocation &operator=(const DelayRingAllocation &) = delete;
    DelayRingAllocation(DelayRingAllocation &&) = default;
    DelayRingAllocation &operator=(DelayRingAllocation &&) = default;
    ~DelayRingAllocation();
};

// Longest per-connection delay across `model`, in whole ticks against `dt_seconds` (the simulation's
// fixed tick duration, in SI seconds -- the same value passed to AssembledModel::step_tick's own
// `dt` every tick), floored to 1 (the engine's existing implicit minimum latency) -- so this is
// always >= 1, even for a model with no connections at all. `ConnectionEntry::delay` is already
// unit-resolved to SI seconds by the resolve pass (resolve.cpp's own unit_value_to_si), so this is a
// plain `round(delay_seconds / dt_seconds)`, floored to 1.
s64 compute_max_delay_ticks(const ModelSpecification &model, f32 dt_seconds);

// Builds and sizes a DelayRingAllocation for `model` against `weights`' own k^2-tree slot layout
// (`weights.node_count`/`weights.max_neighbor_count` -- `weights` MUST be built from the SAME
// adjacency `model`'s connections describe, so slot indices line up; `model.adjacency`, if set, is
// exactly that instance). `ring_slot_count` is `compute_max_delay_ticks(model, dt_seconds) + 1`.
//
// Populates `edge_delay_ticks` by walking every ProjectionEntry's ConnectionEntry list and locating
// each (source, target) pair's k^2-tree neighbor slot (mirrors WeightMatrix::find_neighbor_slot's
// own approach, via the public WeightMatrix::get_neighbors). A connection that doesn't resolve to a
// representable neighbor slot (e.g. truncated by max_neighbor_count, or simply not present in the
// adjacency `weights` was built from) is skipped: it can never be scattered by the ring-aware
// propagate kernel either (that kernel only ever walks REAL k^2-tree neighbor slots), so its delay
// value is moot -- matching WeightMatrix's own existing truncation contract elsewhere (arch §0.3).
//
// A source node with more than one connection landing on the SAME target (a multigraph -- multiple
// synapses between the same anatomical pair) collapses to ONE k^2-tree edge slot, same as weight
// already does for that case (model_specification.h's own documented limitation) -- whichever
// connection is walked last for that (source, target) pair wins.
DelayRingAllocation allocate_delay_ring(const ModelSpecification &model, const WeightMatrix &weights,
                                         f32 dt_seconds);

} // namespace spikecorec::nml
