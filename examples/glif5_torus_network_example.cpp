// ── Example: a torus network of GLIF5 cells (adaptive threshold) ────────────────────────────────
//
// The same 8×8 wraparound grid as glif3_torus_network_example, running the most complete GLIF
// variant. Read that example first — this one focuses on what GLIF5 adds.
//
// ── What GLIF5 is ───────────────────────────────────────────────────────────────────────────────
// GLIF5 = GLIF3 + GLIF4. It carries GLIF3's two after-spike currents (`asc1`/`asc2`) AND GLIF4's
// adaptive threshold, so it has two independent adaptation mechanisms working at once:
//
//   after-spike currents   `asc1`, `asc2` step down on every spike and decay back toward zero.
//                          They oppose the drive, so the cell needs longer to climb.
//
//   adaptive threshold     `theta` is a STATE VARIABLE, not a constant. It relaxes toward `thetaInf`
//                          with time constant `tauTheta` and jumps by `thetaSpikeAdd` on every spike.
//                          The firing condition is `v > theta` — the bar itself rises.
//
// GLIF1-4, for context: GLIF1 is plain leaky integrate-and-fire; GLIF2 adds biologically realistic
// reset; GLIF3 adds the after-spike currents; GLIF4 adds the adaptive threshold instead. GLIF5
// combines the last two. All five are linear, so all five are tagged `closed_form_advanceable` — the
// active-set optimization may fast-forward a quiet one across many ticks in a single closed-form
// step (compare izhikevich_network_example, which is tagged `[nonlinear]` and must never be skipped).
//
// This example plots `v` and `theta` together, which is the whole point: watch the threshold climb
// away from the membrane potential after each spike, then relax back down.
//
// ── Real synaptic propagation (ticket #131) ─────────────────────────────────────────────────────
// As in the GLIF3 example, the torus is wired through a real, vendored `expOneSynapse` —
// AssembledModel dispatches its actual gbase/tauDecay/erev-derived per-edge conductance
// automatically (no opt-in needed once the model declares real projections, which the generated
// torus always does). `--gbase` sets that real conductance amplitude; see
// glif_torus_network.h's own `parse_torus_example_options` doc comment for how the default was
// chosen.
//
// Run:  ./build/examples/glif5_torus_network_example [--side 8] [--ticks 5000] [--gbase 15nS]

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
        TorusExampleOptions{{5000, 1e-4f, false, false}, 8, "15nS", "recordings", true});

    GpuContextScope gpu_context_scope;

    // ── 1. Generate and parse ───────────────────────────────────────────────────────────────────
    TorusNetworkOptions network_options;
    network_options.variant = GlifVariant::Glif5;
    network_options.side_length = options.side_length;
    network_options.stimulated_neuron_indices = {0};
    network_options.stimulus_amplitude = "600pA";
    network_options.synapse_gbase = options.synapse_gbase;

    ModelSpecification model = load_generated_model("glif5_torus", generate_glif_torus_network_nml(network_options));

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

    // ── 2. Assembly and the GLIF5-specific initial state ────────────────────────────────────────
    // SpikeEngine builds its own ModelAllocation + WeightMatrix internally, then every `.tick`
    // section → one master kernel, compiled once.
    SpikeEngine engine(model, programs, options.base.dt_seconds);

    print_heading("State variables");
    std::cout << "  GLIF5 declares " << model.type_library[0].state_variable_count
              << " state variables per neuron:\n"
              << "    slot 0  v                       membrane potential\n"
              << "    slot 1  theta                   ADAPTIVE THRESHOLD (a state variable, not a constant)\n"
              << "    slot 2  asc1                    after-spike current, tau = 100ms\n"
              << "    slot 3  asc2                    after-spike current, tau = 10ms\n"
              << "    slot 4  refractoryTimeElapsed   refractory-regime timer\n"
              << "\n  cell_state elements : " << engine.nml_allocation_.cell_state_element_count << "\n";

    // OnStart is `v = EL, theta = thetaInf, asc1 = asc2 = 0`. allocate_model does not evaluate
    // OnStart, so seeding theta by hand is REQUIRED here — left at zero it would sit above the
    // -50mV threshold and no neuron would ever fire.
    seed_glif_initial_state(engine.nml_allocation_, model, GlifVariant::Glif5);

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
    String membrane_recording_path = options.record_directory + "/glif5_torus_membrane.spire";
    String spike_recording_path = options.record_directory + "/glif5_torus_spikes.spire";
    if (options.record) {
        ensure_directory_exists(options.record_directory);
        recorder = std::make_unique<NetworkActivityRecorder>(
            membrane_recording_path, spike_recording_path, model.total_neuron_count);
    }

    // ── 5. Tick loop ────────────────────────────────────────────────────────────────────────────
    print_heading("Simulating");
    std::cout << "  " << options.base.tick_count << " ticks × " << options.base.dt_seconds * 1000.0f << "ms = "
              << format_seconds((f64)options.base.tick_count * options.base.dt_seconds) << "\n";

    const s32 stimulated_neuron_index = 0;
    const PopulationEntry &population = model.populations[0];
    const s32 theta_slot = glif_state_variable_slot(GlifVariant::Glif5, "theta");
    const s32 asc1_slot = glif_state_variable_slot(GlifVariant::Glif5, "asc1");

    spikecorec::Vector<f32> membrane_trace;
    spikecorec::Vector<f32> threshold_trace;
    spikecorec::Vector<f32> after_spike_current_trace;
    spikecorec::Vector<s64> stimulated_spike_ticks;
    spikecorec::Vector<s64> spike_count_by_neuron((usize)model.total_neuron_count, 0);
    spikecorec::Vector<s64> first_spike_tick_by_neuron((usize)model.total_neuron_count, -1);

    for (s64 tick = 0; tick < options.base.tick_count; ++tick) {
        for (s32 neuron_index = 0; neuron_index < model.total_neuron_count; ++neuron_index) {
            engine.network_inputs.get_contents()[neuron_index] += (f32)schedule.current_at(neuron_index, tick);
        }

        engine.step_tick(options.base.dt_seconds, tick, tick + 1);

        if (recorder) recorder->record_tick(engine.nml_allocation_, model, engine.last_spiked.get_contents(), tick);

        membrane_trace.push_back(
            read_membrane_potential(engine.nml_allocation_, model, /*population_index=*/0, stimulated_neuron_index));
        threshold_trace.push_back(engine.nml_allocation_.cell_state.get_contents()[
            state_element_index(engine.nml_allocation_, population, 0, theta_slot, stimulated_neuron_index)]);
        after_spike_current_trace.push_back(engine.nml_allocation_.cell_state.get_contents()[
            state_element_index(engine.nml_allocation_, population, 0, asc1_slot, stimulated_neuron_index)]);

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
                  << "  render with         : ./examples/render_spire_video.py " << spike_recording_path
                  << " --side " << options.side_length << " --membrane " << membrane_recording_path << "\n";
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

    // ── 7. The two adaptation mechanisms, side by side ──────────────────────────────────────────
    print_heading("Membrane potential (v) — stimulated neuron");
    print_trace_plot("TorusPop[0]/v", membrane_trace, options.base.dt_seconds);

    print_heading("Adaptive threshold (theta) — the bar the cell has to clear");
    print_trace_plot("TorusPop[0]/theta", threshold_trace, options.base.dt_seconds);

    print_heading("After-spike current (asc1) — GLIF3's mechanism, still present");
    print_trace_plot("TorusPop[0]/asc1", after_spike_current_trace, options.base.dt_seconds,
                     /*axis_scale=*/1e12f, "pA");

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

    std::cout << "\nTwo mechanisms are adapting at once here: theta rising makes the target harder to\n"
              << "reach, while asc1/asc2 pull the membrane potential down. GLIF3 has only the second.\n";

    return 0;
}
