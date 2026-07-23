// ── Example: STDP plasticity on a real GLIF network (ticket #129 [T8]) ──────────────────────────
//
// examples/stdp_plasticity_example.cpp already walks the STDP wiring path end to end, but its own
// "AssembledModel" section (section 5) deliberately uses a minimal, 2-neuron GLIF1/LIF-equivalent
// fixture -- adaptation-free by construction, so it never exercises a real GLIF cell's own
// after-spike-current/threshold-adaptation state interacting with a weight-changing plasticity rule.
// This example is the GLIF counterpart: the SAME 8×8 GLIF3 torus every `glif*_torus_network_example`
// uses (glif_torus_network.h), but built with a real per-edge delay and driven with active STDP.
//
// ── Why delay-ring mode, not the torus examples' own real per-edge synapse dispatch ──────────────
// `AssembledModel::enable_plasticity` throws if real per-edge synapse dispatch (ticket #131) is
// active -- STDP's rank-1 nudge of the shared U/V basis is not yet compensated against a peredge
// synapse's own Ck reconstruction sharing that same basis (see master_kernel.h's own "ticket #132"
// doc comment). The one combination `enable_plasticity` already accepts is `enable_delay_ring=true`,
// which forces #131's dispatch off entirely -- so this network is wired with a real, non-trivial
// per-edge delay (`connection_delay`, ticket #64) rather than the torus examples' usual
// `expOneSynapse` conductance. `--gbase` (inherited from the shared torus option parser) therefore has
// NO effect here, the same documented exception `delayed_coupling_example`/`poisson_population_example`
// already are (see examples/README.md's own "Real synaptic propagation" section).
//
// ── Why the scattered weight is a real, fixed magnitude, decoupled from what STDP measures ───────
// `weights.set_constant_weight(...)` does two things: it fixes `constant_weight`/`using_constant_weight`
// (what the delay ring's own propagate stage actually scatters, every tick, regardless of anything
// STDP does later), AND it bakes that same value into U/V as their OWN initial encoding. Because
// `using_constant_weight` stays true for the rest of the run, the ring's own propagate stage keeps
// scattering that FIXED value forever -- STDP's own updates (which mutate U/V directly) never feed
// back into what actually propagates, so the torus's own spike-frequency/spread behavior stays exactly
// as reliable as every other torus example's, while `WeightMatrix::get()` (always a real U/V
// reconstruction, independent of `using_constant_weight` -- weight_matrix.cpp's own `get()`) still
// shows STDP's real rank-1 nudge moving the STORED weight underneath it. This example prints exactly
// that: the stimulated corner's own 4 real torus edges, before and after.
//
// ── The recording (ticket #138's own NetworkActivityRecorder, same as every torus example) ────────
// Two `.spire` streams over the whole 64-neuron population -- membrane potential and a spike-raster
// mask -- played back with `examples/render_spire_video.py --side 8`, no changes needed to that
// script (same `row * side_length + column` flat-index convention every torus example already uses).
//
// Run:  ./build/examples/glif_stdp_plasticity_example [--ticks 5000] [--dt 0.0001] [--side 8]

#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cstring>
#include <iostream>

#include "spikecorec/core/topologies.h"
#include "spikecorec/nml/delay_ring.h"
#include "spikecorec/nml/stimulus_schedule.h"

#include "glif_torus_network.h"

using namespace spikecorec;
using namespace spikecorec::nml;
using namespace spikecorec::examples;

int main(int argument_count, char **argument_values) {
    TorusExampleOptions options = parse_torus_example_options(
        argument_count, argument_values,
        TorusExampleOptions{{5000, 1e-4f, false, false}, 8, "10nS", "recordings", true});

    GpuContextScope gpu_context_scope;

    // ── 1. Generate the torus, wired with a real per-edge delay instead of the usual expOneSynapse
    //      dispatch (see this file's own header comment) ───────────────────────────────────────────
    TorusNetworkOptions network_options;
    network_options.variant = GlifVariant::Glif3;
    network_options.side_length = options.side_length;
    network_options.stimulated_neuron_indices = {0};
    network_options.stimulus_amplitude = "600pA";
    network_options.connection_delay = "2ms"; // non-empty -> delay-ring mode (ticket #64)

    ModelSpecification model =
        load_generated_model("glif_stdp_torus", generate_glif_torus_network_nml(network_options));

    // ── 2. Lowering ─────────────────────────────────────────────────────────────────────────────
    Vector<IrProgram> programs = lower_type_library_to_ir(model);
    print_model_summary(model, programs);
    if (options.base.print_ir) print_ir_programs(model, programs);

    print_heading("Torus topology");
    std::cout << "  grid            : " << options.side_length << " × " << options.side_length << "\n"
              << "  neurons         : " << model.total_neuron_count << "\n"
              << "  connections     : " << model.projections[0].connections.size()
              << "   (4 wraparound neighbours per neuron)\n"
              << "  stimulated      : neuron 0 (grid corner)\n"
              << "  connection delay: " << network_options.connection_delay
              << "   (real per-edge delay, ticket #64 -- required for STDP, see this file's own header comment)\n";

    // ── 3. Allocation + initial state ──────────────────────────────────────────────────────────
    ModelAllocation allocation = allocate_model(model, programs);
    seed_glif_initial_state(allocation, model, GlifVariant::Glif3);

    // ── 4. Weight matrix + delay ring ──────────────────────────────────────────────────────────
    WeightMatrix weights = build_weight_matrix(model);
    // A real, fixed scattered magnitude -- see this file's own header comment for why STDP's own
    // updates below never feed back into it (using_constant_weight stays true for the whole run).
    const f32 scattered_weight = 0.5f;
    weights.set_constant_weight(scattered_weight);

    DelayRingAllocation ring = allocate_delay_ring(model, weights, options.base.dt_seconds);
    print_heading("Delay ring");
    std::cout << "  ring slots      : " << ring.ring_slot_count << "   (max delay in ticks + 1)\n"
              << "  scattered weight: " << scattered_weight << "   (fixed for the whole run, see above)\n";

    // ── 5. Assembly + STDP (ticket #132) ───────────────────────────────────────────────────────
    AssembledModel assembled_model(model, programs, /*enable_delay_ring=*/true);
    const f32 stdp_learning_rate = 0.2f;
    assembled_model.enable_plasticity(stdp_learning_rate);

    print_heading("Plasticity (ticket #132)");
    std::cout << "  plasticity_enabled : " << std::boolalpha << assembled_model.plasticity_enabled() << "\n"
              << "  learning_rate      : " << stdp_learning_rate << "\n"
              << "  every neuron that fires walks its own real k^2-tree neighbours and nudges their\n"
              << "  stored weight down (WeightMatrix::update, stage 7) -- see the summary at the end.\n";

    // ── 6. Runtime buffers -- ring mode needs none of the flat network_inputs/active-set arrays
    //      (superseded by the ring's own equivalents, see delay_ring.h) ───────────────────────────
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

    // ── 7. Stimulus schedule -- injected into the ring's CURRENT slot every tick, ring mode's own
    //      equivalent of the flat network_inputs accumulation the other torus examples use ─────────
    StimulusSchedule schedule = build_stimulus_schedule(model, (f64)options.base.dt_seconds);

    // ── 8. STDP's own real k^2-tree edges out of the driven corner, sampled before the run ────────
    Vector<Vector<s32>> torus_adjacency = square_torus(options.side_length);
    const s32 sampled_source_neuron_index = 0;
    Vector<s32> sampled_targets = torus_adjacency[(usize)sampled_source_neuron_index];
    Vector<f32> sampled_weight_before(sampled_targets.size());
    for (usize sample_index = 0; sample_index < sampled_targets.size(); ++sample_index) {
        sampled_weight_before[sample_index] = weights.get(sampled_source_neuron_index, sampled_targets[sample_index]);
    }

    // ── 9. Recording (ticket #138) ─────────────────────────────────────────────────────────────
    std::unique_ptr<NetworkActivityRecorder> recorder;
    String membrane_recording_path = options.record_directory + "/glif_stdp_torus_membrane.spire";
    String spike_recording_path = options.record_directory + "/glif_stdp_torus_spikes.spire";
    if (options.record) {
        ensure_directory_exists(options.record_directory);
        recorder = std::make_unique<NetworkActivityRecorder>(
            membrane_recording_path, spike_recording_path, model.total_neuron_count);
    }

    // ── 10. Tick loop ───────────────────────────────────────────────────────────────────────────
    print_heading("Simulating");
    std::cout << "  " << options.base.tick_count << " ticks × " << options.base.dt_seconds * 1000.0f << "ms = "
              << format_seconds((f64)options.base.tick_count * options.base.dt_seconds) << "\n";

    Vector<s64> spike_count_by_neuron((usize)model.total_neuron_count, 0);
    Vector<s64> first_spike_tick_by_neuron((usize)model.total_neuron_count, -1);
    Vector<s64> stimulated_spike_ticks;

    for (s64 tick = 0; tick < options.base.tick_count; ++tick) {
        s64 current_slot = tick % ring.ring_slot_count;
        s64 current_slot_base = current_slot * model.total_neuron_count;
        for (s32 neuron_index = 0; neuron_index < model.total_neuron_count; ++neuron_index) {
            f64 current = schedule.current_at(neuron_index, tick);
            if (current != 0.0) ring.input_ring.get_contents()[current_slot_base + neuron_index] += (f32)current;
        }

        assembled_model.step_tick(buffers, options.base.dt_seconds, tick, tick + 1);

        if (recorder) recorder->record_tick(allocation, model, buffers.last_spiked, tick);

        for (s32 neuron_index = 0; neuron_index < model.total_neuron_count; ++neuron_index) {
            if (buffers.last_spiked[neuron_index] != tick) continue;
            ++spike_count_by_neuron[(usize)neuron_index];
            if (first_spike_tick_by_neuron[(usize)neuron_index] == -1) {
                first_spike_tick_by_neuron[(usize)neuron_index] = tick;
            }
            if (neuron_index == sampled_source_neuron_index) stimulated_spike_ticks.push_back(tick);
        }
    }

    if (recorder) {
        recorder->finish();
        print_heading("Recording");
        std::cout << "  membrane potential  : " << membrane_recording_path << "\n"
                  << "  spike raster        : " << spike_recording_path << "\n"
                  << "  render with         : ./examples/render_spire_video.py " << spike_recording_path
                  << " --side " << options.side_length << " --membrane " << membrane_recording_path << "\n";
    }

    // ── 11. Network activity ───────────────────────────────────────────────────────────────────
    s64 total_spike_count = 0;
    s32 neurons_that_fired = 0;
    for (s64 count : spike_count_by_neuron) {
        total_spike_count += count;
        if (count > 0) ++neurons_that_fired;
    }

    print_heading("Network activity");
    std::cout << "  total spikes       " << total_spike_count << "\n"
              << "  neurons that fired " << neurons_that_fired << " / " << model.total_neuron_count << "\n\n";
    print_torus_grid("Spikes per neuron", spike_count_by_neuron, options.side_length, "spike count");

    print_heading("Spread across the torus");
    print_torus_grid("First spike tick", first_spike_tick_by_neuron, options.side_length, "tick");

    print_spike_times("TorusPop[0]", stimulated_spike_ticks, options.base.dt_seconds);

    // ── 12. STDP's measured effect on the driven corner's own 4 real edges ────────────────────────
    print_heading("Plasticity effect (ticket #132)");
    for (usize sample_index = 0; sample_index < sampled_targets.size(); ++sample_index) {
        f32 weight_after = weights.get(sampled_source_neuron_index, sampled_targets[sample_index]);
        std::cout << "  neuron " << sampled_source_neuron_index << " → neuron " << sampled_targets[sample_index]
                  << "   before=" << sampled_weight_before[sample_index] << "  after=" << weight_after
                  << "  change=" << (weight_after - sampled_weight_before[sample_index]) << "\n";
    }

    std::cout << "\nEvery real edge out of a neuron that fired at least once (nearly every neuron above, see the\n"
              << "'Spikes per neuron' grid) was walked by apply_stdp_plasticity and nudged the same way -- the\n"
              << "sample above is just the driven corner's own 4 neighbours, printed for readability. This is a\n"
              << "REAL rank-1 nudge of the same shared U/V basis reconstructed above, not a placeholder: see\n"
              << "tests/glif_stdp_network_tests.cpp for the network-scale test this example mirrors.\n";

    return 0;
}
