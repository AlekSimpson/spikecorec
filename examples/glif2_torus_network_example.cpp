// ── Example: a torus network of GLIF2 cells (scaled reset) ──────────────────────────────────────
//
// The same 8×8 wraparound grid as glif3_torus_network_example, running GLIF2. Read that example
// first — this one focuses on what GLIF2 adds.
//
// ── What GLIF2 is ───────────────────────────────────────────────────────────────────────────────
// GLIF1 (plain leaky integrate-and-fire) plus a biologically realistic RESET RULE: instead of
// snapping straight back to a fixed `vreset` on every spike, GLIF2 resets to
//
//     v <- vreset + resetScale * (v - vth)
//
// — a reset that scales with how far PAST threshold `v` overshot on the triggering tick.
// `resetScale=0` degenerates to GLIF1's exact flat reset (`v <- vreset`, no dependence on the
// overshoot at all); this example uses `resetScale=0.4` (the more interesting, "scaled" case),
// which lands measurably ABOVE `vreset` after a spike, never bit-exactly at it. That is the same
// contrast `tests/end_to_end_network_tests.cpp`'s own
// `glif2_ring_network_discrete_spike_array_smallest_anchor` asserts between its "flat"
// (`resetScale=0`) and "scaled" (`resetScale=0.4`) populations — this example runs only the scaled
// case live, since a single homogeneous torus population is this generator's own established shape
// (see glif_torus_network.h).
//
// GLIF1-5, for context: GLIF1 is plain LIF; GLIF2 (this file) adds the scaled reset above; GLIF3
// adds after-spike currents; GLIF4 adds an adaptive threshold instead; GLIF5 combines the last two.
// All five are linear, so all five are tagged `closed_form_advanceable`.
//
// ── Real synaptic propagation (ticket #131) ─────────────────────────────────────────────────────
// As in the GLIF3 example, the torus is wired through a real, vendored `expOneSynapse` —
// AssembledModel dispatches its actual gbase/tauDecay/erev-derived per-edge conductance
// automatically. `--gbase` sets that real conductance amplitude; see glif_torus_network.h's own
// `parse_torus_example_options` doc comment for how the GLIF3/GLIF5 defaults were chosen — this
// file's default was measured the same way, against this exact parameter set/topology.
//
// Run:  ./build/examples/glif2_torus_network_example [--side 8] [--ticks 5000] [--gbase 10nS]

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
    network_options.variant = GlifVariant::Glif2;
    network_options.side_length = options.side_length;
    // Drive one corner of the torus so activity has somewhere to spread from.
    network_options.stimulated_neuron_indices = {0};
    network_options.stimulus_amplitude = "600pA";
    network_options.synapse_gbase = options.synapse_gbase;

    ModelSpecification model = load_generated_model("glif2_torus", generate_glif_torus_network_nml(network_options));

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

    seed_glif_initial_state(engine.nml_allocation_, model, GlifVariant::Glif2);

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
    String membrane_recording_path = options.record_directory + "/glif2_torus_membrane" + options.record_extension;
    String spike_recording_path = options.record_directory + "/glif2_torus_spikes" + options.record_extension;
    if (options.record) {
        ensure_directory_exists(options.record_directory);
        recorder = std::make_unique<NetworkActivityRecorder>(
            membrane_recording_path, spike_recording_path, model.total_neuron_count);
    }
    TicksPerSecondTelemetry ticks_per_second(
        options.record_directory, "glif2_torus", options.record, options.record_stride);

    // ── 6. The reset rule this example is about ─────────────────────────────────────────────────
    const PopulationEntry &population = model.populations[0];
    const TypeLibraryEntry &cell_entry = model.type_library[(usize)population.type_library_index];
    const f32 vreset_volts = (f32)cell_entry.baked_constants.at("vreset");
    const f32 reset_scale = (f32)cell_entry.baked_constants.at("resetScale");

    // ── 7. Tick loop ────────────────────────────────────────────────────────────────────────────
    print_heading("Simulating");
    std::cout << "  " << options.base.tick_count << " ticks × " << options.base.dt_seconds * 1000.0f << "ms = "
              << format_seconds((f64)options.base.tick_count * options.base.dt_seconds) << "\n";

    const s32 stimulated_neuron_index = 0;
    spikecorec::Vector<f32> stimulated_membrane_trace;
    stimulated_membrane_trace.reserve((usize)options.base.tick_count);
    spikecorec::Vector<s64> stimulated_spike_ticks;
    spikecorec::Vector<f32> post_reset_membrane_potential_by_spike;
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
            if (neuron_index == stimulated_neuron_index) {
                stimulated_spike_ticks.push_back(tick);
                // The reset already ran this tick (stage 5, before Advance/Record) — read `v` right
                // back out, exactly the same "capture right after the reset" pattern
                // tests/end_to_end_network_tests.cpp's own GLIF2 anchor test uses.
                post_reset_membrane_potential_by_spike.push_back(
                    read_membrane_potential(engine.nml_allocation_, model, /*population_index=*/0, stimulated_neuron_index));
            }
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

    // ── 8. Results ──────────────────────────────────────────────────────────────────────────────
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

    print_heading("The scaled reset (resetScale=0.4)");
    std::cout << "  vreset (the flat/GLIF1 floor)   " << vreset_volts * 1000.0f << " mV\n"
              << "  resetScale                      " << reset_scale << "\n";
    if (!post_reset_membrane_potential_by_spike.empty()) {
        f32 minimum_post_reset = post_reset_membrane_potential_by_spike.front();
        f32 maximum_post_reset = post_reset_membrane_potential_by_spike.front();
        for (f32 value : post_reset_membrane_potential_by_spike) {
            minimum_post_reset = std::min(minimum_post_reset, value);
            maximum_post_reset = std::max(maximum_post_reset, value);
        }
        std::cout << "  v right after each reset        " << minimum_post_reset * 1000.0f << " … "
                  << maximum_post_reset * 1000.0f << " mV   (over "
                  << post_reset_membrane_potential_by_spike.size() << " observed spikes)\n"
                  << "  above vreset by                 " << (minimum_post_reset - vreset_volts) * 1000.0f
                  << " … " << (maximum_post_reset - vreset_volts) * 1000.0f << " mV\n"
                  << "\nUnlike GLIF1's exact flat reset (v <- vreset, every time), GLIF2's reset lands\n"
                  << "measurably ABOVE vreset here, scaling with how far past vth the tick that\n"
                  << "triggered it overshot — the same contrast\n"
                  << "tests/end_to_end_network_tests.cpp's own\n"
                  << "glif2_ring_network_discrete_spike_array_smallest_anchor asserts between its\n"
                  << "flat (resetScale=0) and scaled (resetScale=0.4) populations.\n";
    } else {
        std::cout << "\n  the stimulated neuron never spiked -- no reset to inspect; try --gbase or "
                     "--ticks\n";
    }

    return 0;
}
