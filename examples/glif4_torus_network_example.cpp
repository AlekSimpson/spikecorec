// ── Example: a torus network of GLIF4 cells (adaptive threshold, no after-spike currents) ───────
//
// The same 8×8 wraparound grid as glif3_torus_network_example, running GLIF4. Read glif5's own
// example too — GLIF5 is GLIF3 + GLIF4 combined, and this file isolates GLIF4's own mechanism on
// its own, with no after-spike currents opposing it at all.
//
// ── What GLIF4 is ───────────────────────────────────────────────────────────────────────────────
// Plain leaky integrate-and-fire, but with an ADAPTIVE THRESHOLD instead of after-spike currents.
// `theta` is a STATE VARIABLE, not a constant: it relaxes toward `thetaInf` with time constant
// `tauTheta` and jumps by `thetaSpikeAdd` on every spike, so the firing condition is `v > theta`
// rather than `v > vth` — the bar itself rises after every spike, then relaxes back down. No
// `asc1`/`asc2` here at all (that is GLIF3's mechanism, and GLIF5's when both are combined) — GLIF4
// adapts purely through the moving threshold.
//
// GLIF1-5, for context: GLIF1 is plain LIF; GLIF2 adds a scaled reset; GLIF3 adds after-spike
// currents; GLIF4 (this file) adds the adaptive threshold instead; GLIF5 combines the last two.
// All five are linear, so all five are tagged `closed_form_advanceable`.
//
// ── Real synaptic propagation (ticket #131) ─────────────────────────────────────────────────────
// As in the GLIF3 example, the torus is wired through a real, vendored `expOneSynapse` —
// AssembledModel dispatches its actual gbase/tauDecay/erev-derived per-edge conductance
// automatically. `--gbase` sets that real conductance amplitude; see glif_torus_network.h's own
// `parse_torus_example_options` doc comment for how the GLIF3/GLIF5 defaults were chosen — this
// file's default was measured the same way, against this exact parameter set/topology.
//
// Run:  ./build/examples/glif4_torus_network_example [--side 8] [--ticks 5000] [--gbase 10nS]

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

    // ── 1. Generate and parse ───────────────────────────────────────────────────────────────────
    TorusNetworkOptions network_options;
    network_options.variant = GlifVariant::Glif4;
    network_options.side_length = options.side_length;
    network_options.stimulated_neuron_indices = {0};
    network_options.stimulus_amplitude = "600pA";
    network_options.synapse_gbase = options.synapse_gbase;

    ModelSpecification model = load_generated_model("glif4_torus", generate_glif_torus_network_nml(network_options));

    spikecorec::Vector<IrProgram> programs = lower_type_library_to_ir(model);
    print_model_summary(model, programs);
    if (options.base.print_ir) print_ir_programs(model, programs);

    print_heading("Torus topology");
    std::cout << "  grid            : " << options.side_length << " × " << options.side_length << "\n"
              << "  neurons         : " << model.total_neuron_count << "\n"
              << "  connections     : " << model.projections[0].connections.size()
              << "   (4 wraparound neighbours per neuron)\n"
              << "  synapse         : expOneSynapse, gbase=" << options.synapse_gbase
              << " (real per-edge dispatch, ticket #131)\n";

    // ── 2. Assembly and the GLIF4-specific initial state ────────────────────────────────────────
    // SpikeEngine builds its own ModelAllocation + WeightMatrix internally, then every `.tick`
    // section → one master kernel, compiled once.
    SpikeEngine engine(model, programs, options.base.dt_seconds);

    print_heading("State variables");
    std::cout << "  GLIF4 declares " << model.type_library[0].state_variable_count
              << " state variables per neuron:\n"
              << "    slot 0  v                       membrane potential\n"
              << "    slot 1  theta                   ADAPTIVE THRESHOLD (a state variable, not a constant)\n"
              << "    slot 2  refractoryTimeElapsed   refractory-regime timer\n"
              << "\n  cell_state elements : " << engine.nml_allocation_.cell_state_element_count << "\n";

    // OnStart is `v = EL, theta = thetaInf`. allocate_model does not evaluate OnStart, so seeding
    // theta by hand is REQUIRED here — left at zero it would sit above the -50mV threshold and no
    // neuron would ever fire.
    seed_glif_initial_state(engine.nml_allocation_, model, GlifVariant::Glif4);

    std::cout << "  population tagged   : "
              << (population_is_closed_form_advanceable(model, 0) ? "closed-form advanceable" : "nonlinear")
              << "   (GLIF is linear in its own state variables)\n";

    // ── 3. Stimulus ─────────────────────────────────────────────────────────────────────────────
    StimulusSchedule schedule = build_stimulus_schedule(model, (f64)options.base.dt_seconds);
    print_heading("Stimulus schedule");
    for (s32 window_index = 0; window_index < schedule.window_count; ++window_index) {
        std::cout << "  neuron " << schedule.target_neurons[window_index]
                  << "  ticks [" << schedule.start_ticks[window_index] << ", " << schedule.end_ticks[window_index] << ")"
                  << "  current " << schedule.current_values[window_index] * 1e12 << " pA\n";
    }

    // ── 4. Recording (ticket #138) ──────────────────────────────────────────────────────────────
    std::unique_ptr<NetworkActivityRecorder> recorder;
    String membrane_recording_path = options.record_directory + "/glif4_torus_membrane" + options.record_extension;
    String spike_recording_path = options.record_directory + "/glif4_torus_spikes" + options.record_extension;
    if (options.record) {
        ensure_directory_exists(options.record_directory);
        recorder = std::make_unique<NetworkActivityRecorder>(
            membrane_recording_path, spike_recording_path, model.total_neuron_count);
    }
    TicksPerSecondTelemetry ticks_per_second(
        options.record_directory, "glif4_torus", options.record, options.record_stride);

    // ── 5. Tick loop ────────────────────────────────────────────────────────────────────────────
    print_heading("Simulating");
    std::cout << "  " << options.base.tick_count << " ticks × " << options.base.dt_seconds * 1000.0f << "ms = "
              << format_seconds((f64)options.base.tick_count * options.base.dt_seconds) << "\n";

    const s32 stimulated_neuron_index = 0;
    const PopulationEntry &population = model.populations[0];
    const s32 theta_slot = glif_state_variable_slot(GlifVariant::Glif4, "theta");

    spikecorec::Vector<f32> membrane_trace;
    spikecorec::Vector<f32> threshold_trace;
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

        membrane_trace.push_back(
            read_membrane_potential(engine.nml_allocation_, model, /*population_index=*/0, stimulated_neuron_index));
        threshold_trace.push_back(engine.nml_allocation_.cell_state.get_contents()[
            state_element_index(engine.nml_allocation_, population, 0, theta_slot, stimulated_neuron_index)]);

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

    // ── 6. Results ──────────────────────────────────────────────────────────────────────────────
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

    // ── 7. The adaptive threshold, in isolation ─────────────────────────────────────────────────
    print_heading("Membrane potential (v) — stimulated neuron");
    print_trace_plot("TorusPop[0]/v", membrane_trace, options.base.dt_seconds);

    print_heading("Adaptive threshold (theta) — the bar the cell has to clear");
    print_trace_plot("TorusPop[0]/theta", threshold_trace, options.base.dt_seconds);

    print_heading("Spike train");
    print_spike_times("TorusPop[0]", stimulated_spike_ticks, options.base.dt_seconds);
    print_interspike_interval_summary(stimulated_spike_ticks, options.base.dt_seconds);

    if (!threshold_trace.empty()) {
        f32 resting_threshold = threshold_trace.front();
        f32 peak_threshold = *std::max_element(threshold_trace.begin(), threshold_trace.end());
        std::cout << "\n  theta at rest  " << resting_threshold * 1000.0f << " mV   (= thetaInf)\n"
                  << "  theta at peak  " << peak_threshold * 1000.0f << " mV   (+"
                  << (peak_threshold - resting_threshold) * 1000.0f << " mV of accumulated adaptation)\n";
    }

    std::cout << "\nGLIF4 adapts through theta ALONE — no after-spike currents pulling v down (that is\n"
              << "GLIF3's mechanism, and both together is GLIF5's). Compare this theta trace with\n"
              << "glif5_torus_network_example's own: GLIF5's asc1/asc2 add a second, independent\n"
              << "adaptation mechanism on top of the same threshold dynamics shown here.\n";

    return 0;
}
