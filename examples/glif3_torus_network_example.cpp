// ── Example: a torus network of GLIF3 cells ─────────────────────────────────────────────────────
//
// Start here. This is the complete NeuroML → GPU codegen path over a real network: 64 GLIF3 cells
// wired into an 8×8 wraparound grid (256 connections), driven by a current step, with spikes
// propagating across the torus.
//
// What it demonstrates:
//   1. Generating NeuroML for a network too large to hand-type, then running it through the exact
//      same parse → resolve → lower path as any checked-in file (see glif_torus_network.h)
//   2. per-ComponentType IR generation                (ticket #50)
//   3. `.alloc` interpretation into engine buffers    (ticket #5)
//   4. `.tick` assembly into one compiled master kernel and per-tick dispatch (tickets #55/#6)
//   5. `explicitInput`/`pulseGenerator` → a host-precomputed stimulus schedule (ticket #58)
//   6. Reading per-neuron state back out of the structure-of-arrays `cell_state` layout
//   7. Real per-edge synapse dispatch (ticket #131) driving genuine, whole-network propagation
//   8. Recording every neuron's membrane potential and spike raster to `.spire` files (ticket #138),
//      played back by `examples/render_spire_video.py`
//
// ── What GLIF3 is ───────────────────────────────────────────────────────────────────────────────
// A leaky integrate-and-fire cell plus two after-spike currents, `asc1` and `asc2`. Both step down
// by a fixed amount on every spike and decay back toward zero with their own time constants (100 ms
// and 10 ms here). Because they are negative, they oppose the drive, so each spike makes the next
// one harder to reach: inter-spike intervals lengthen under a constant current. That is
// spike-frequency adaptation, and it emerges from the LEMS description — there is no hand-written
// GLIF3 kernel anywhere in the engine.
//
// ── Real synaptic propagation, not a placeholder (ticket #131 supersedes the old weight caveat) ──
// Earlier revisions of this example stipulated an explicit placeholder current per arriving spike,
// because AssembledModel's propagate stage did not yet invoke a projection's synapse ComponentType
// dynamics (the "spike-scatter batch construction subsystem" no ticket had built). That subsystem is
// now built (ticket #131): whenever a model declares real projections, AssembledModel dispatches
// that projection's actual synapse type automatically, no extra opt-in required. So the torus below
// is wired through a real, vendored `expOneSynapse` — `--gbase` sets its actual conductance
// amplitude (folded through `g' = -g/tauDecay`, `g += gbase` on delivery, `i = g*(erev-v)`), not a
// stand-in current. See glif_torus_network.h's own `parse_torus_example_options` doc comment for how
// the default was chosen.
//
// Run:  ./build/examples/glif3_torus_network_example [--side 8] [--ticks 5000] [--gbase 10nS]

// The backend header comes first, before anything that pulls a `String` alias into global scope —
// metal-cpp declares its own `NS::String`, and the two collide if Metal.hpp is included later. Every
// *_tests.cpp file in this tree opens with this same block for the same reason.
#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <iostream>

#include "spikecorec/core/engine.h"
#include "spikecorec/nml/stimulus_schedule.h"

#include "glif_torus_network.h"

using namespace spikecorec;
using namespace spikecorec::nml;
using namespace spikecorec::examples;

// `Vector<...>` is spelled out fully as `spikecorec::Vector<...>` throughout this file, unlike every
// other torus example prior to the SpikeEngine migration. spikecorec/core/engine.h pulls in a
// file-scope `using namespace spikecorec::log;`, which declares its OWN `Vector` alias template --
// ambiguous with `spikecorec::Vector` for bare unqualified `Vector<...>` lookup (two alias templates
// of the same name from two using-directives at the same scope, regardless of expanding to the
// identical type). Mirrors what tests/simple_lif_stdp_network_tests.cpp/tests/end_to_end_network_
// tests.cpp/examples/stdp_plasticity_example.cpp already do for the same reason.

int main(int argument_count, char **argument_values) {
    TorusExampleOptions options = parse_torus_example_options(
        argument_count, argument_values,
        TorusExampleOptions{{5000, 1e-4f, false, false}, 8, "10nS", "recordings", true});

    GpuContextScope gpu_context_scope;

    // ── 1. Generate the network, then run it through the front-end ──────────────────────────────
    TorusNetworkOptions network_options;
    network_options.variant = GlifVariant::Glif3;
    network_options.side_length = options.side_length;
    // Drive one corner of the torus so activity has somewhere to spread from.
    network_options.stimulated_neuron_indices = {0};
    network_options.stimulus_amplitude = "600pA";
    network_options.synapse_gbase = options.synapse_gbase;

    ModelSpecification model = load_generated_model("glif3_torus", generate_glif_torus_network_nml(network_options));

    // ── 2. Lowering: one small IR program per ComponentType ─────────────────────────────────────
    spikecorec::Vector<IrProgram> programs = lower_type_library_to_ir(model);
    print_model_summary(model, programs);
    if (options.base.print_ir) print_ir_programs(model, programs);

    print_heading("Torus topology");
    std::cout << "  grid            : " << options.side_length << " × " << options.side_length << "\n"
              << "  neurons         : " << model.total_neuron_count << "\n"
              << "  connections     : " << model.projections[0].connections.size()
              << "   (4 wraparound neighbours per neuron)\n"
              << "  stimulated      : neuron 0 (grid corner)\n"
              << "  synapse         : expOneSynapse, gbase=" << options.synapse_gbase
              << " (real per-edge dispatch, ticket #131)\n";

    // ── 3. Assembly: SpikeEngine builds its own ModelAllocation + WeightMatrix internally, then
    // every `.tick` section → one master kernel, compiled once ─────────────────────────────────
    SpikeEngine engine(model, programs, options.base.dt_seconds);
    print_heading("Allocation");
    std::cout << "  cell_state elements : " << engine.nml_allocation_.cell_state_element_count
              << "   (" << model.type_library[0].state_variable_count << " state variables × "
              << model.total_neuron_count << " neurons, structure-of-arrays per population)\n";

    seed_glif_initial_state(engine.nml_allocation_, model, GlifVariant::Glif3);

    // ── 4. Stimulus ─────────────────────────────────────────────────────────────────────────────
    StimulusSchedule schedule = build_stimulus_schedule(model, (f64)options.base.dt_seconds);
    print_heading("Stimulus schedule");
    for (s32 window_index = 0; window_index < schedule.window_count; ++window_index) {
        std::cout << "  neuron " << schedule.target_neurons[window_index]
                  << "  ticks [" << schedule.start_ticks[window_index] << ", " << schedule.end_ticks[window_index] << ")"
                  << "  = " << format_seconds((f64)schedule.start_ticks[window_index] * options.base.dt_seconds)
                  << " … " << format_seconds((f64)schedule.end_ticks[window_index] * options.base.dt_seconds)
                  << "  current " << schedule.current_values[window_index] * 1e12 << " pA\n";
    }

    // ── 5. Recording (ticket #138) ──────────────────────────────────────────────────────────────
    // Two `.spire` streams over the whole 64-neuron population: membrane potential and a 0/1 spike
    // raster mask, one frame per tick each — a real recorded artifact of the whole network's
    // activity, playable with `examples/render_spire_video.py --side 8`.
    std::unique_ptr<NetworkActivityRecorder> recorder;
    String membrane_recording_path = options.record_directory + "/glif3_torus_membrane" + options.record_extension;
    String spike_recording_path = options.record_directory + "/glif3_torus_spikes" + options.record_extension;
    if (options.record) {
        ensure_directory_exists(options.record_directory);
        recorder = std::make_unique<NetworkActivityRecorder>(
            membrane_recording_path, spike_recording_path, model.total_neuron_count);
    }
    TicksPerSecondTelemetry ticks_per_second(
        options.record_directory, "glif3_torus", options.record, options.record_stride);

    // ── 6. Tick loop ────────────────────────────────────────────────────────────────────────────
    print_heading("Simulating");
    std::cout << "  " << options.base.tick_count << " ticks × " << options.base.dt_seconds * 1000.0f << "ms = "
              << format_seconds((f64)options.base.tick_count * options.base.dt_seconds) << "\n";

    const s32 stimulated_neuron_index = 0;
    spikecorec::Vector<f32> stimulated_membrane_trace;
    stimulated_membrane_trace.reserve((usize)options.base.tick_count);
    spikecorec::Vector<s64> stimulated_spike_ticks;
    spikecorec::Vector<s64> spike_count_by_neuron((usize)model.total_neuron_count, 0);
    spikecorec::Vector<s64> first_spike_tick_by_neuron((usize)model.total_neuron_count, -1);

    for (s64 tick = 0; tick < options.base.tick_count; ++tick) {
        for (s32 neuron_index = 0; neuron_index < model.total_neuron_count; ++neuron_index) {
            engine.network_inputs.get_contents()[neuron_index] += (f32)schedule.current_at(neuron_index, tick);
        }

        engine.step_tick(options.base.dt_seconds, tick, tick + 1);

        if (recorder && tick % options.record_stride == 0) {
            recorder->record_tick(engine.nml_allocation_, model, engine.last_spiked.get_contents(), tick);
        }
        ticks_per_second.record_tick(tick, options.base.tick_count);

        stimulated_membrane_trace.push_back(
            read_membrane_potential(engine.nml_allocation_, model, /*population_index=*/0, stimulated_neuron_index));

        for (s32 neuron_index = 0; neuron_index < model.total_neuron_count; ++neuron_index) {
            if (engine.last_spiked.get_contents()[neuron_index] != tick) continue;

            ++spike_count_by_neuron[(usize)neuron_index];
            if (first_spike_tick_by_neuron[(usize)neuron_index] == -1) {
                first_spike_tick_by_neuron[(usize)neuron_index] = tick;
            }
            if (neuron_index == stimulated_neuron_index) stimulated_spike_ticks.push_back(tick);
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

    // ── 7. Results ──────────────────────────────────────────────────────────────────────────────
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

    print_heading("Stimulated neuron (grid corner)");
    print_trace_plot("TorusPop[0]/v", stimulated_membrane_trace, options.base.dt_seconds);
    print_spike_times("TorusPop[0]", stimulated_spike_ticks, options.base.dt_seconds);
    print_interspike_interval_summary(stimulated_spike_ticks, options.base.dt_seconds);

    std::cout << "\nThe lengthening intervals are GLIF3's two after-spike currents accumulating.\n"
              << "Compare with glif5_torus_network_example, where an adaptive THRESHOLD produces the\n"
              << "same qualitative effect through a completely different mechanism.\n";

    return 0;
}
