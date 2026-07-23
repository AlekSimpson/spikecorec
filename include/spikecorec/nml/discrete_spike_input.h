#pragma once

#include "spikecorec/core/types.h"

namespace spikecorec::nml {

// ── Discrete host-provided spike-array driving (ticket #100 [T1]) ───────────────────────────────
//
// A second, independent driving mechanism alongside ticket #58's pulseGenerator-derived
// StimulusSchedule (stimulus_schedule.h): a literal, host-provided sequence of 0s/1s -- one value
// per tick, one column per designated "input" neuron -- with no corresponding NeuroML generator
// ComponentType at all. `SpikeEngine::start_static_record`'s own `input_spikes` parameter
// (src/core/engine.h) is the precedent for the SHAPE of this input (a tick-major dense array), but
// it only drives the legacy hardcoded engine, not `AssembledModel` (nml/master_kernel.h) -- this
// file is the narrowly-scoped plumbing needed to drive the latter instead.
//
// What "wiring this into AssembledModel's live per-tick loop" actually required, on investigation:
// nothing beyond this struct. `AssembledModel::step_tick` already reads
// `ModelRuntimeBuffers::network_inputs` (or, in delay-ring mode, whatever ring slot a caller passes
// as that tick's own network_inputs -- see delay_ring.h) as an ordinary, generically-writable `f32*`
// array every tick; ticket #61's own established convention (exit_model_validation_tests.cpp) already
// injects continuous pulseGenerator current directly into that same array from the host, one line per
// tick, immediately before calling step_tick. `apply_to_network_inputs` below is exactly that same
// one-line injection, generalized to read its per-tick amount from a caller-supplied dense 0/1 array
// instead of a continuous window -- no engine-side change was needed, only this convenience wrapper.
// Deliberately NOT a general new stimulus subsystem (no NML/LEMS surface, no ComponentType, no
// schedule-window abstraction): a bare struct plus one method, scoped to exactly the shape ticket
// #100/#101 both need.

struct DiscreteSpikeInputSchedule {
    // Which neurons this schedule drives, in the same order `spike_bits`' inner index refers to.
    Vector<s32> target_neuron_indices;

    // spike_bits[tick][index] -- 0 or 1 -- whether target_neuron_indices[index] is driven at `tick`.
    // A caller may supply fewer rows than the simulation's own tick count; `apply_to_network_inputs`
    // is a no-op for any tick past `spike_bits.size() - 1`.
    Vector<Vector<u8>> spike_bits;

    // The real, SI-unit current (amperes) injected into `network_inputs` for one tick when a bit is
    // set -- forward-Euler-integrated by the SAME per-ComponentType `.tick` code a real synapse's
    // contribution would be (ticket #61's own established `network_inputs`-injection convention;
    // see this header's own doc comment above), not a dimensionless spike count.
    f32 current_amplitude_amperes = 1.0e-9f;

    // Adds `current_amplitude_amperes` to `network_inputs[target_neuron_indices[index]]` for every
    // `index` whose `spike_bits[tick][index]` is nonzero. Call once per tick, before
    // `AssembledModel::step_tick`, mirroring exactly how a continuous StimulusSchedule window is applied
    // (exit_model_validation_tests.cpp's own established per-tick injection pattern). `network_inputs`
    // must have at least as many elements as the model's own total_neuron_count (or, in delay-ring
    // mode, be the caller's own current ring-slot slice -- see delay_ring.h); a no-op if
    // `tick < 0` or `tick >= (s64)spike_bits.size()`.
    void apply_to_network_inputs(f32 *network_inputs, s64 tick) const;
};

} // namespace spikecorec::nml
