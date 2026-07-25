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
// `SpikeEngine::enable_plasticity` throws if real per-edge synapse dispatch (ticket #131) is
// active -- STDP's rank-1 nudge of the shared U/V basis is not yet compensated against a peredge
// synapse's own Ck reconstruction sharing that same basis (see engine.h's own "ticket #132" doc
// comment). The one combination `enable_plasticity` already accepts is a real per-edge delay forcing
// the delay ring (ring_slot_count > 1, engine.cpp), which forces #131's dispatch off entirely -- so
// this network is wired with a real, non-trivial per-edge delay (`connection_delay`, ticket #64)
// rather than the torus examples' usual `expOneSynapse` conductance. `--gbase` (inherited from the
// shared torus option parser) therefore has NO effect here, the same documented exception
// `delayed_coupling_example`/`poisson_population_example` already are (see examples/README.md's own
// "Real synaptic propagation" section).
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

#include "spikecorec/core/engine.h"
#include "spikecorec/core/topologies.h"
#include "spikecorec/nml/delay_ring.h"
#include "spikecorec/nml/stimulus_schedule.h"

#include "glif_torus_network.h"

using namespace spikecorec;
using namespace spikecorec::nml;
using namespace spikecorec::examples;

// `Vector<...>` is spelled out fully as `spikecorec::Vector<...>` throughout this file, unlike every
// other example prior to the SpikeEngine migration. spikecorec/core/engine.h pulls in a file-scope
// `using namespace spikecorec::log;`, which declares its OWN `Vector` alias template -- ambiguous
// with `spikecorec::Vector` for bare unqualified `Vector<...>` lookup (two alias templates of the
// same name from two using-directives at the same scope, regardless of expanding to the identical
// type). Mirrors what tests/simple_lif_stdp_network_tests.cpp/tests/end_to_end_network_tests.cpp/
// examples/stdp_plasticity_example.cpp already do for the same reason.

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
    spikecorec::Vector<IrProgram> programs = lower_type_library_to_ir(model);
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

    // ── 3-5. SpikeEngine builds its own ModelAllocation + WeightMatrix internally, converting this
    // model's real per-connection delay to whole ticks and folding `network_inputs` into a ring
    // accordingly, then every `.tick` section → one master kernel, compiled once (see this file's
    // own header comment and delayed_coupling_example.cpp's identical fold). ─────────────────────
    SpikeEngine engine(model, programs, options.base.dt_seconds);
    seed_glif_initial_state(engine.nml_allocation_, model, GlifVariant::Glif3);

    // A real, fixed scattered magnitude -- see this file's own header comment for why STDP's own
    // updates below never feed back into it (using_constant_weight stays true for the whole run).
    const f32 scattered_weight = 0.5f;
    engine.weights.set_constant_weight(scattered_weight);

    // Exactly one real per-connection delay value in this torus, so the uniform-delay path applies
    // (SpikeEngine's constructor, engine.cpp): weights.set_constant_delay_ticks(...), not the
    // per-edge array -- so `weights.constant_delay_ticks` is this model's own real ring_slot_count.
    const s64 ring_slot_count = engine.weights.constant_delay_ticks;

    // compute_max_delay_ticks is the same free, standalone seconds->ticks scan SpikeEngine's own
    // constructor uses upstream of the ring fold (delay_ring.h) -- reported here purely as
    // information about the model's own declared delay, independent of the engine's internal ring
    // representation.
    print_heading("Delay ring");
    std::cout << "  ring slots      : " << compute_max_delay_ticks(model, options.base.dt_seconds) + 1
              << "   (max delay in ticks + 1)\n"
              << "  scattered weight: " << scattered_weight << "   (fixed for the whole run, see above)\n";

    // ── 6. STDP (ticket #132) ───────────────────────────────────────────────────────────────────
    const f32 stdp_learning_rate = 0.2f;
    engine.enable_plasticity(stdp_learning_rate);

    print_heading("Plasticity (ticket #132)");
    std::cout << "  plasticity_enabled : " << std::boolalpha << engine.plasticity_enabled() << "\n"
              << "  learning_rate      : " << stdp_learning_rate << "\n"
              << "  every neuron that fires walks its own real k^2-tree neighbours and nudges their\n"
              << "  stored weight down (WeightMatrix::update, stage 7) -- see the summary at the end.\n";

    // ── 7. Stimulus schedule -- injected into the ring's CURRENT slot every tick, ring mode's own
    //      equivalent of the flat network_inputs accumulation the other torus examples use ─────────
    StimulusSchedule schedule = build_stimulus_schedule(model, (f64)options.base.dt_seconds);

    // ── 8. STDP's own real k^2-tree edges out of the driven corner, sampled before the run ────────
    spikecorec::Vector<spikecorec::Vector<s32>> torus_adjacency = square_torus(options.side_length);
    const s32 sampled_source_neuron_index = 0;
    spikecorec::Vector<s32> sampled_targets = torus_adjacency[(usize)sampled_source_neuron_index];
    spikecorec::Vector<f32> sampled_weight_before(sampled_targets.size());
    for (usize sample_index = 0; sample_index < sampled_targets.size(); ++sample_index) {
        sampled_weight_before[sample_index] =
            engine.weights.get(sampled_source_neuron_index, sampled_targets[sample_index]);
    }

    // ── 9. Recording (ticket #138) ─────────────────────────────────────────────────────────────
    std::unique_ptr<NetworkActivityRecorder> recorder;
    String membrane_recording_path = options.record_directory + "/glif_stdp_torus_membrane" + options.record_extension;
    String spike_recording_path = options.record_directory + "/glif_stdp_torus_spikes" + options.record_extension;
    if (options.record) {
        ensure_directory_exists(options.record_directory);
        recorder = std::make_unique<NetworkActivityRecorder>(
            membrane_recording_path, spike_recording_path, model.total_neuron_count);
    }
    TicksPerSecondTelemetry ticks_per_second(
        options.record_directory, "glif_stdp_torus", options.record, options.record_stride);

    // ── 10. Tick loop ───────────────────────────────────────────────────────────────────────────
    print_heading("Simulating");
    std::cout << "  " << options.base.tick_count << " ticks × " << options.base.dt_seconds * 1000.0f << "ms = "
              << format_seconds((f64)options.base.tick_count * options.base.dt_seconds) << "\n";

    spikecorec::Vector<s64> spike_count_by_neuron((usize)model.total_neuron_count, 0);
    spikecorec::Vector<s64> first_spike_tick_by_neuron((usize)model.total_neuron_count, -1);
    spikecorec::Vector<s64> stimulated_spike_ticks;

    for (s64 tick = 0; tick < options.base.tick_count; ++tick) {
        s64 current_slot = tick % ring_slot_count;
        s64 current_slot_base = current_slot * model.total_neuron_count;
        for (s32 neuron_index = 0; neuron_index < model.total_neuron_count; ++neuron_index) {
            f64 current = schedule.current_at(neuron_index, tick);
            if (current != 0.0) {
                engine.network_inputs.get_contents()[current_slot_base + neuron_index] += (f32)current;
            }
        }

        engine.step_tick(options.base.dt_seconds, tick, tick + 1);

        if (recorder && tick % options.record_stride == 0) {
            recorder->record_tick(engine.nml_allocation_, model, engine.last_spiked.get_contents(), tick);
        }
        ticks_per_second.record_tick(tick, options.base.tick_count);

        for (s32 neuron_index = 0; neuron_index < model.total_neuron_count; ++neuron_index) {
            if (engine.last_spiked.get_contents()[neuron_index] != tick) continue;
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
                  << "  ticks/sec telemetry : " << ticks_per_second.path()
                  << "   (auto-detected by render_spire_video.py, no flag needed)\n"
                  << "  render with         : ./examples/render_spire_video.py " << spike_recording_path
                  << " --side " << options.side_length << " --membrane " << membrane_recording_path
                  << (options.record_stride > 1
                          ? " --dt " + std::to_string(options.base.dt_seconds * (f32)options.record_stride)
                          : String(""))
                  << "\n";
        if (options.record_stride > 1) {
            std::cout << "  record stride       : every " << options.record_stride << " ticks ("
                      << (options.base.tick_count + options.record_stride - 1) / options.record_stride
                      << " frames recorded instead of " << options.base.tick_count << ")\n";
        }
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
        f32 weight_after = engine.weights.get(sampled_source_neuron_index, sampled_targets[sample_index]);
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
