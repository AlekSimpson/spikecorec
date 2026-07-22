// ── Example: axonal delays via the delay ring ───────────────────────────────────────────────────
//
// By default a spike reaches its target with an implicit one-tick latency: the source scatters its
// weight into `network_inputs[target]`, and the target drains it on its next step. That is fine
// until a model declares real per-edge delays — `<connectionWD ... delay="5ms"/>` — which need a
// spike to arrive many ticks later, at a per-edge distance.
//
// The delay ring is how that works. Instead of one flat `network_inputs` array, the engine keeps a
// ring of `max_delay_ticks + 1` slots, each a full `[neuron_count]` row. A spike scatters into the
// slot for its own arrival tick rather than into "next tick"; the tick loop reads slot
// `tick % ring_slot_count`. Per-edge delays are precomputed into whole ticks at allocation time.
//
// What this example demonstrates:
//   1. allocate_delay_ring deriving per-edge delays in ticks from each connection's `delay`
//   2. AssembledModel(..., enable_delay_ring=true) selecting the ring-based deliver/propagate stages
//   3. The buffer contract change — in ring mode, `network_inputs`, the active-set enqueue arrays
//      and `active_generation` are all superseded by the ring's own equivalents and go unallocated
//   4. Measuring the actual delivery offset: the tick the source fires versus the tick a nonzero
//      contribution shows up in the target's ring slot
//
// ── Why this example still uses a constant scattered weight (ticket #131 does NOT apply here) ────
// Ticket #131's real per-edge synapse dispatch is deliberately NOT wired up when
// `enable_delay_ring=true`: the delay ring and #131's synapse dispatch have not been integrated with
// each other (master_kernel.h's own documented scope boundary — a model built with
// `enable_delay_ring=true` keeps exactly its pre-#131 behavior, the fixed scalar propagate-ring
// path, regardless of what its projections declare). So `weights.set_constant_weight(0.6f)` below is
// still the REAL mechanism this example's scattered value comes from, unlike the torus/GLIF-E/I
// examples — this is one of the two documented exceptions to "AssembledModel now dispatches real
// synapse dynamics automatically" (the other being a model with no projections at all).
//
// Run:  ./build/examples/delayed_coupling_example [--ticks 1200] [--dt 0.0001]

#include <cmath>
#include <cstring>
#include <iostream>

#include "spikecorec/nml/delay_ring.h"

#include "nml_pipeline_support.h"

using namespace spikecorec;
using namespace spikecorec::nml;
using namespace spikecorec::examples;

int main(int argument_count, char **argument_values) {
    ExampleOptions options = parse_example_options(argument_count, argument_values, ExampleOptions{1200, 1e-4f});
    GpuContextScope gpu_context_scope;

    ModelSpecification model = load_model_specification("delayed_coupling_network");
    Vector<IrProgram> programs = lower_type_library_to_ir(model);
    print_model_summary(model, programs);
    if (options.print_ir) print_ir_programs(model, programs);

    print_heading("Declared connection delays");
    for (const ProjectionEntry &projection : model.projections) {
        for (const ConnectionEntry &connection : projection.connections) {
            std::cout << "  " << projection.id << ": neuron " << connection.source_neuron_index
                      << " → neuron " << connection.target_neuron_index
                      << "  delay = " << format_seconds(connection.delay)
                      << "  (" << (s64)std::round(connection.delay / (f64)options.dt_seconds) << " ticks)\n";
        }
    }

    ModelAllocation allocation = allocate_model(model, programs);
    seed_membrane_potentials_from_resting_parameter(allocation, model); // GLIF1Cell: OnStart v = EL

    WeightMatrix weights = build_weight_matrix(model);
    // An arbitrary nonzero placeholder — the REAL mechanism here, not documentation (see this file's
    // own header comment on why ticket #131's real per-edge synapse dispatch does not apply in ring
    // mode). This example measures delivery TIMING, which the ring derives purely from each
    // connection's own delay attribute, independent of whatever value is scattered — so unlike the
    // GLIF E/I example, no magnitude claim is being made here at all.
    weights.set_constant_weight(0.6f);

    // ── 1. Ring allocation ──────────────────────────────────────────────────────────────────────
    DelayRingAllocation ring = allocate_delay_ring(model, weights, options.dt_seconds);

    print_heading("Delay ring");
    std::cout << "  ring slots         : " << ring.ring_slot_count
              << "   (max delay in ticks + 1)\n"
              << "  neurons per slot   : " << ring.neuron_count << "\n"
              << "  input_ring elements: " << ring.ring_slot_count * ring.neuron_count
              << "   (slot-major: slot s's row is what flat network_inputs would hold for that tick)\n";

    // ── 2. Ring-aware assembly ──────────────────────────────────────────────────────────────────
    AssembledModel assembled_model(model, programs, /*enable_delay_ring=*/true);

    // ── 3. The reduced buffer set ring mode needs ───────────────────────────────────────────────
    // No network_inputs, no active-set enqueue arrays, no active_generation — the ring supersedes
    // all of them. Passing them anyway would be ignored; step_tick throws if `delay_ring` is null
    // while the model was assembled for ring mode (and vice versa).
    GpuPointer<s64> last_spiked = allocate<s64>((usize)model.total_neuron_count * sizeof(s64));
    std::fill(last_spiked.get_contents(), last_spiked.get_contents() + model.total_neuron_count, (s64)-1);
    GpuPointer<bool> emit_spike_flags = allocate<bool>((usize)model.total_neuron_count * sizeof(bool));
    std::memset(emit_spike_flags.get_contents(), 0, (usize)model.total_neuron_count * sizeof(bool));

    ModelRuntimeBuffers buffers;
    buffers.allocation = &allocation;
    buffers.weights = &weights;
    buffers.last_spiked = last_spiked.get_contents();
    buffers.emit_port_flags["spike"] = emit_spike_flags.get_contents();
    buffers.delay_ring = &ring;

    // ── 4. Tick loop ────────────────────────────────────────────────────────────────────────────
    const f64 seconds_per_tick = (f64)options.dt_seconds;
    const s64 stimulus_delay_ticks = (s64)std::round(0.010 / seconds_per_tick);
    const s64 stimulus_duration_ticks = (s64)std::round(0.006 / seconds_per_tick);
    const f32 stimulus_amplitude_amperes = 0.5e-9f; // amplitude="0.5nA"
    const s32 source_neuron_index = 0;              // SourcePop, declared first
    const s32 target_neuron_index = 1;              // TargetPop, declared second
    const f32 delivery_epsilon = 1e-9f;

    print_heading("Simulating");
    std::cout << "  " << options.tick_count << " ticks × " << options.dt_seconds * 1000.0f << "ms = "
              << format_seconds((f64)options.tick_count * options.dt_seconds) << "\n";

    Vector<s64> source_spike_ticks;
    Vector<s64> target_delivery_ticks;

    for (s64 tick = 0; tick < options.tick_count; ++tick) {
        const s64 current_slot = tick % ring.ring_slot_count;
        const s64 current_slot_base = current_slot * model.total_neuron_count;

        // Read this tick's ring slot BEFORE stimulating or stepping, so it reflects only what a
        // PRIOR tick's propagate stage already scattered into it.
        f32 target_input_this_tick = ring.input_ring.get_contents()[current_slot_base + target_neuron_index];
        if (std::fabs(target_input_this_tick) > delivery_epsilon) target_delivery_ticks.push_back(tick);

        if (tick >= stimulus_delay_ticks && tick < stimulus_delay_ticks + stimulus_duration_ticks) {
            // In ring mode the stimulus goes into the CURRENT slot rather than a flat array — the
            // generated kernel reads whatever this tick's slot holds under the same reserved
            // `network_inputs` parameter name.
            ring.input_ring.get_contents()[current_slot_base + source_neuron_index] += stimulus_amplitude_amperes;
        }

        assembled_model.step_tick(buffers, options.dt_seconds, tick, tick + 1);
        if (buffers.last_spiked[source_neuron_index] == tick) source_spike_ticks.push_back(tick);
    }

    // ── 5. Results: measured delay vs. declared delay ───────────────────────────────────────────
    print_heading("Delivery timing");
    print_spike_times("SourcePop[0] fired", source_spike_ticks, options.dt_seconds);
    print_spike_times("TargetPop[0] received", target_delivery_ticks, options.dt_seconds);

    if (!source_spike_ticks.empty() && !target_delivery_ticks.empty()) {
        s64 measured_offset_ticks = target_delivery_ticks.front() - source_spike_ticks.front();
        std::cout << "\n  first spike at tick    " << source_spike_ticks.front()
                  << "  (" << format_seconds((f64)source_spike_ticks.front() * seconds_per_tick) << ")\n"
                  << "  first delivery at tick " << target_delivery_ticks.front()
                  << "  (" << format_seconds((f64)target_delivery_ticks.front() * seconds_per_tick) << ")\n"
                  << "  measured offset        " << measured_offset_ticks << " ticks ("
                  << format_seconds((f64)measured_offset_ticks * seconds_per_tick) << ")\n"
                  << "\n  Compare that against the connection's declared delay printed above. Without\n"
                  << "  the ring, every delivery would land exactly one tick after the spike.\n";
    } else {
        std::cout << "\n  No spike/delivery pair observed — try a longer run with --ticks.\n";
    }

    return 0;
}
