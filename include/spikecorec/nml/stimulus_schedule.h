#pragma once

#include "spikecorec/nml/model_specification.h"

namespace spikecorec::nml {

// ── Host-precomputed stimulus inputs (ticket #58 [E1]; arch §3.3 D4, §5
// Phase 1) ────────────────────────────────────────────────────────────────
//
// `pulseGenerator` (arch §3.3 D4) is the one Phase-1 Inputs ComponentType --
// "Host-precomputed into the stimulus buffer in Phase 1; on-device
// generators in Phase 2" (arch §3.3/§5). Its whole current trajectory is a
// deterministic function of three constants (`delay`/`duration`/`amplitude`,
// canonical SI after resolve) and is entirely knowable in advance, so there
// is nothing to evaluate per tick on the GPU: this module computes the
// finished per-neuron, per-tick current schedule ONCE on the host from a
// `ModelSpecification`'s `StimulusEntry`/`TypeLibraryEntry` data (ticket #7
// [A5]) and hands it to the engine in exactly the shape it already knows how
// to consume.
//
// Where this plugs in (arch §0.2): "External stimulus is added straight into
// `membrane_potentials`" -- unlike a synapse's contribution (which scatters
// into `network_inputs` with a >=1-tick latency, arch §0.1/IR spec §3.5),
// stimulus lands directly in the cell's own voltage-like state the same tick
// it's due. `SpikeEngine::step_simulation`'s `input_values` parameter (and
// `start_static_record`'s tick-major `input_spikes`) already implement
// exactly that addition (`gpu_add_network_input` atomically adds straight
// into `membrane_potentials`, backend.cpp) -- so `to_dense_input_spikes`
// below renders this schedule directly into that existing parameter shape;
// no new engine-side plumbing is needed to drive today's hardcoded LIF cell
// with a precomputed pulseGenerator schedule.
//
// Scope boundary (documented, not silently dropped): splicing a
// host-precomputed schedule into the GENERATED per-ComponentType master
// kernel's own `@integrate` code is a separate concern from this ticket.
// The locked IR spec (v1.0) reserves exactly `dt`/`tick`/`network_inputs` as
// engine-supplied `.tick` reads (IR spec §3.1) and has no "external
// stimulus" operand -- adding one is a spec change, and master-kernel
// assembly (ticket #6 [C3]) that would consume it isn't merged yet either.
// That wiring belongs to ticket #61 [H1] (Phase-1 validation & wiring, which
// depends on this ticket). This module only produces the schedule.

// pulseGenerator's own documented default (third_party/neuroml2/std_lib/
// Inputs.xml: `<Property name="weight" dimension="none" defaultValue="1"/>`)
// -- see build_stimulus_schedule's doc comment for why this can't be read
// off `baked_constants` itself.
constexpr f64 DEFAULT_PULSE_GENERATOR_WEIGHT = 1.0;

// The host-precomputed stimulus schedule for one model (ticket #58 [E1]).
// Flat, parallel arrays -- one slot per `StimulusEntry` -- rather than a
// dense `[neuron_count x tick_count]` array (Phase-1 stimulus counts are
// tiny, so a dense per-tick buffer would waste memory for no benefit;
// `to_dense_input_spikes` below renders exactly the dense slice an engine
// call actually needs, on demand). `window_count` is known up front from
// `model.stimuli.size()`, so the arrays are sized once at construction and
// never resized.
struct StimulusSchedule {
    // neuron_index -> every schedule slot targeting it. Almost always exactly
    // one entry, but current_at() sums however many are present -- e.g. two
    // separate `explicitInput`s (two overlapping pulseGenerators) bound to
    // the same neuron legitimately sum, the same way two physical current
    // sources into one cell would.
    UnorderedMap<s64, Vector<s32>> windows_by_neuron;

    s64 *start_ticks = nullptr;    // [window_count], inclusive
    s64 *end_ticks = nullptr;      // [window_count], exclusive
    f64 *current_values = nullptr; // [window_count], canonical SI amperes
    s32 *target_neurons = nullptr; // [window_count] -- which neuron slot `i` targets

    s32 window_count = 0;

    StimulusSchedule(StimulusSchedule &&other) noexcept;
    StimulusSchedule &operator=(StimulusSchedule &&other) noexcept;
    StimulusSchedule(const StimulusSchedule &) = delete;
    StimulusSchedule &operator=(const StimulusSchedule &) = delete;

    // Allocates window_count_ slots, all initially unassigned to any neuron.
    // A caller fills every slot via set_window() before reading current_at()/
    // to_dense_input_spikes() (build_stimulus_schedule below always does).
    explicit StimulusSchedule(s32 window_count_);
    ~StimulusSchedule();

    // Assigns schedule slot `index` (< window_count, passed to the
    // constructor) to `neuron_index`'s window list, with the given
    // half-open [start_tick, end_tick) range and constant current.
    void set_window(s32 index, s64 neuron_index, s64 start_tick, s64 end_tick, f64 current);

    // Total injected current for `neuron_index` at `tick` -- the sum of
    // every window that targets that neuron and contains `tick` (0.0 if
    // no window targets that neuron at all, or none of the ones that do
    // contain `tick`).
    [[nodiscard]] f64 current_at(s32 neuron_index, s64 tick) const;

    // Renders this schedule as a tick-major matrix: result[tick][index] is
    // `current_at(target_neuron_indices[index], tick)`, for
    // `tick` in `[0, tick_count)` -- exactly the shape
    // `SpikeEngine::start_static_record`'s own `input_spikes` parameter
    // already expects (positionally matched to a set of input neurons), so
    // this schedule can drive a real simulation with no new engine-side
    // plumbing (see this header's doc comment above).
    [[nodiscard]] Vector<Vector<f32>> to_dense_input_spikes(const Vector<s32> &target_neuron_indices, s64 tick_count) const;
};

// Builds the stimulus schedule from every `StimulusEntry` in `model` (ticket
// #7 [A5]). `seconds_step` is the model's `dt` in seconds, used to convert
// each pulseGenerator's resolved (canonical SI) `delay`/`duration` into tick
// counts via `time::tick_count_from_seconds`.
//
// Reads, per stimulus, its Inputs `TypeLibraryEntry::baked_constants`:
// `delay`/`duration`/`amplitude` (required -- a `pulseGenerator` always
// declares these as `Parameter`s with no default) and `weight` (a
// `Property` with NeuroML-documented `defaultValue="1"` -- Property default
// values are not yet applied at resolve time, arch §3.1, so a `weight`
// absent from `baked_constants` here defaults to 1.0 to match that
// documented default, not because of any choice made in this module).
//
// Throws std::runtime_error if a `StimulusEntry`'s input component is not a
// `pulseGenerator` (Phase 1's only Inputs surface, arch §5) or if its
// `TypeLibraryEntry::baked_constants` is missing `delay`/`duration`/
// `amplitude`.
StimulusSchedule build_stimulus_schedule(const ModelSpecification &model, f64 seconds_step);

} // namespace spikecorec::nml
