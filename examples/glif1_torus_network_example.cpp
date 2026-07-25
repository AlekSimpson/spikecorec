// ── Example: a torus network of GLIF1 cells (plain leaky integrate-and-fire) ────────────────────
//
// The same 8×8 wraparound grid as glif3_torus_network_example, running the SIMPLEST GLIF variant.
// Read glif3_torus_network_example first — this one focuses on what GLIF1 is (and is not).
//
// ── What GLIF1 is ───────────────────────────────────────────────────────────────────────────────
// Plain leaky integrate-and-fire: one state variable (`v`), a FIXED threshold `vth`, and a flat
// reset to `vreset` on every spike. No after-spike currents (GLIF3's `asc1`/`asc2`), no adaptive
// threshold (GLIF4/GLIF5's `theta`) — just the refractory-regime timer every GLIF variant shares.
// There is no spike-frequency adaptation here: absent a changing input, GLIF1's inter-spike
// intervals stay flat rather than lengthening the way GLIF3/GLIF5's do.
//
// GLIF1-5, for context: GLIF1 (this file) is plain LIF; GLIF2 adds a scaled reset; GLIF3 adds the
// after-spike currents; GLIF4 adds the adaptive threshold instead; GLIF5 combines the last two. All
// five are linear, so all five are tagged `closed_form_advanceable` — the active-set optimization
// may fast-forward a quiet one across many ticks in a single closed-form step (compare
// izhikevich_network_example, which is tagged `[nonlinear]` and must never be skipped).
//
// A separate, hand-typed GLIF1 network already exists (`glif_ei_network_example.cpp`, a 5-neuron
// excitatory/inhibitory fixture exercising several synapse types at once). This example is the
// GLIF1 counterpart to the GLIF3/GLIF5 TORUS walkthroughs instead — a single, homogeneous,
// generated 64-neuron network with real per-edge synaptic propagation.
//
// ── Real synaptic propagation (ticket #131) ─────────────────────────────────────────────────────
// As in the GLIF3 example, the torus is wired through a real, vendored `expOneSynapse` —
// AssembledModel dispatches its actual gbase/tauDecay/erev-derived per-edge conductance
// automatically (no opt-in needed once the model declares real projections, which the generated
// torus always does). `--gbase` sets that real conductance amplitude; see
// glif_torus_network.h's own `parse_torus_example_options` doc comment for how the GLIF3/GLIF5
// defaults were chosen — this file's default was measured the same way, against this exact
// parameter set/topology.
//
// Run:  ./build/examples/glif1_torus_network_example [--side 8] [--ticks 5000] [--gbase 10nS]

// See glif3_torus_network_example.cpp — the backend header must precede anything that pulls a
// `String` alias into global scope, or it collides with metal-cpp's own `NS::String`.
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
    network_options.variant = GlifVariant::Glif1;
    network_options.side_length = options.side_length;
    // Drive one corner of the torus so activity has somewhere to spread from.
    network_options.stimulated_neuron_indices = {0};
    network_options.stimulus_amplitude = "600pA";
    network_options.synapse_gbase = options.synapse_gbase;

    ModelSpecification model = load_generated_model("glif1_torus", generate_glif_torus_network_nml(network_options));

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

    seed_glif_initial_state(engine.nml_allocation_, model, GlifVariant::Glif1);

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
    String membrane_recording_path = options.record_directory + "/glif1_torus_membrane" + options.record_extension;
    String spike_recording_path = options.record_directory + "/glif1_torus_spikes" + options.record_extension;
    if (options.record) {
        ensure_directory_exists(options.record_directory);
        recorder = std::make_unique<NetworkActivityRecorder>(
            membrane_recording_path, spike_recording_path, model.total_neuron_count);
    }
    TicksPerSecondTelemetry ticks_per_second(
        options.record_directory, "glif1_torus", options.record, options.record_stride);

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

    std::cout << "\nGLIF1 has no adaptation mechanism at all: after an initial charging transient the\n"
              << "intervals above settle to a constant steady-state rate rather than continuously\n"
              << "lengthening, in contrast with glif3_torus_network_example's own after-spike-current\n"
              << "adaptation and glif5_torus_network_example's (after-spike currents + adaptive\n"
              << "threshold together).\n";

    return 0;
}
