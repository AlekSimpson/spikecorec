// ── Example: a nonlinear point cell (izhikevich2007Cell) ────────────────────────────────────────
//
// Phase 2 of the codegen epic adds nonlinear point cells. Everything the GLIF examples show still
// applies unchanged — same parse/lower/allocate/assemble/step_tick path, same generic codegen — but
// two things change once the dynamics stop being linear.
//
// What this example is really about:
//
//   1. The active-set × nonlinear rule. The engine's active-set optimization can skip a quiet neuron
//      for many ticks and then catch it up with one closed-form `apply_decay`. That is only valid
//      for analytically-integrable (linear) dynamics — all of GLIF. A nonlinear cell must integrate
//      exactly one dt per active tick and can never be fast-forwarded. Lowering classifies each cell
//      type via cell_dynamics_are_closed_form_advanceable, and this example prints that tag below:
//      GLIF prints closed-form advanceable, izhikevich2007Cell prints nonlinear. That single bit is
//      what keeps a skip-dispatch fast path from silently corrupting nonlinear cells.
//
//   2. Two state variables instead of one. izhikevich2007Cell carries a recovery variable `u`
//      alongside `v`, and its OnStart sets `v = v0`. allocate_model zero-initializes cell_state and
//      does not apply OnStart, so both are seeded by hand here — which also makes the
//      structure-of-arrays layout of cell_state concrete (see seed_izhikevich_initial_state below).
//
// Regular-spiking parameters produce a rising-then-settling spike train under a current step. Note
// that this is genuinely nonlinear integration on the GPU, from a LEMS description, with no
// hand-written izhikevich kernel anywhere in the engine.
//
// ── Why TargetPop stays silent here (a real, orthogonal, pre-existing gap — ticket #131) ─────────
// This fixture's own DrivenPop → TargetPop connection is wired through `alphaCurrentSynapse`
// (current-based). Ticket #131 made AssembledModel dispatch a projection's real synapse
// ComponentType automatically whenever the model declares one — no opt-in needed — but
// `alphaCurrentSynapse`'s own coupled `I`/`J` TimeDerivative pair isn't a shape synapse_lowering.cpp
// recognizes yet (its own documented limitation: "general per-edge forward-Euler integration for an
// arbitrary right-hand side is out of Phase-1 scope"), so `I` never integrates and TargetPop never
// receives any current at all — the SAME gap glif_ei_network_example's own header comment documents
// for ExcPop[2]. Fixing that lowering gap is real engine work, out of this ticket's example/tooling
// scope, so this example stays what it always was: a single-neuron nonlinear-dynamics demonstration,
// not a propagation demonstration (see the torus examples for that).
//
// Run:  ./build/examples/izhikevich_network_example [--ticks 3000] [--dt 0.0001] [--print-ir]

#include <cmath>
#include <iostream>

#include "spikecorec/core/engine.h"
#include "nml_pipeline_support.h"

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

namespace {

// izhikevich2007Cell's OnStart is `v = v0, u = 0`, with both populations declaring v0 = -60mV.
//
// This is also the clearest place to see cell_state's layout: a population's chunk is
// structure-of-arrays, so state slot 0 (`v`) occupies the first `population.size` elements of the
// chunk and slot 1 (`u`) the next `population.size`.
void seed_izhikevich_initial_state(ModelAllocation &allocation, const ModelSpecification &model) {
    const f32 initial_membrane_potential_volts = -0.06f; // v0 = -60mV, both populations
    seed_state_variable_everywhere(allocation, model, /*state_slot_index=*/0, initial_membrane_potential_volts);
    seed_state_variable_everywhere(allocation, model, /*state_slot_index=*/1, 0.0f); // u
}

} // namespace

int main(int argument_count, char **argument_values) {
    ExampleOptions options = parse_example_options(argument_count, argument_values, ExampleOptions{3000, 1e-4f});
    GpuContextScope gpu_context_scope;

    ModelSpecification model = load_model_specification("izhikevich_network");
    spikecorec::Vector<IrProgram> programs = lower_type_library_to_ir(model);
    print_model_summary(model, programs);
    if (options.print_ir) print_ir_programs(model, programs);

    // ── The active-set × nonlinear tag, read back from the engine ──────────────────────────────
    // SpikeEngine builds its own ModelAllocation + WeightMatrix internally, then every `.tick`
    // section → one master kernel, compiled once.
    SpikeEngine engine(model, programs, options.dt_seconds);
    seed_izhikevich_initial_state(engine.nml_allocation_, model);

    // set_constant_weight(0.0f) is documentation, not the mechanism: this model has a real
    // projection, so SpikeEngine dispatches alphaCurrentSynapse's own dynamics automatically
    // (ticket #131) and forces this WeightMatrix's own scattered contribution to zero regardless.
    // See this file's own header comment for why TargetPop stays silent anyway.
    engine.weights.set_constant_weight(0.0f);

    print_heading("Active-set × nonlinear tags");
    std::cout << "  A closed-form-advanceable population may be skipped for many ticks and caught up\n"
              << "  in one step. A nonlinear population must integrate exactly one dt per tick.\n\n";
    for (usize population_index = 0; population_index < model.populations.size(); ++population_index) {
        std::cout << "    " << std::left << std::setw(16) << model.populations[population_index].id << std::right
                  << (population_is_closed_form_advanceable(model, population_index)
                          ? "closed-form advanceable  (skipping permitted)"
                          : "nonlinear                (must never be fast-forwarded)")
                  << "\n";
    }

    // ── Stimulus: reconstructed from the file's own pulseGenerator (inputList, not explicitInput) ─
    const f64 seconds_per_tick = (f64)options.dt_seconds;
    const s64 stimulus_delay_ticks = (s64)std::round(0.010 / seconds_per_tick);
    const s64 stimulus_duration_ticks = (s64)std::round(0.200 / seconds_per_tick);
    const f32 stimulus_amplitude_amperes = 150e-12f; // amplitude="150pA"
    const s32 driven_neuron_index = 0;               // DrivenPop's neuron, declared first

    print_heading("Simulating");
    std::cout << "  " << options.tick_count << " ticks × " << options.dt_seconds * 1000.0f << "ms = "
              << format_seconds((f64)options.tick_count * options.dt_seconds) << "\n"
              << "  driving neuron " << driven_neuron_index << " with " << stimulus_amplitude_amperes * 1e12f
              << " pA for " << format_seconds((f64)stimulus_duration_ticks * seconds_per_tick) << "\n";

    spikecorec::Vector<f32> driven_membrane_trace;
    driven_membrane_trace.reserve((usize)options.tick_count);
    spikecorec::Vector<f32> recovery_variable_trace;
    recovery_variable_trace.reserve((usize)options.tick_count);
    spikecorec::Vector<s64> driven_spike_ticks;

    const PopulationEntry &driven_population = model.populations[0];

    for (s64 tick = 0; tick < options.tick_count; ++tick) {
        if (tick >= stimulus_delay_ticks && tick < stimulus_delay_ticks + stimulus_duration_ticks) {
            engine.network_inputs.get_contents()[driven_neuron_index] += stimulus_amplitude_amperes;
        }

        engine.step_tick(options.dt_seconds, tick, tick + 1);

        driven_membrane_trace.push_back(read_membrane_potential(engine.nml_allocation_, model, /*population_index=*/0, 0));
        recovery_variable_trace.push_back(engine.nml_allocation_.cell_state.get_contents()[
            state_element_index(engine.nml_allocation_, driven_population, /*population_index=*/0, /*state_slot_index=*/1, 0)]);
        if (engine.last_spiked.get_contents()[driven_neuron_index] == tick) driven_spike_ticks.push_back(tick);
    }

    print_heading("Membrane potential (v)");
    print_trace_plot("DrivenPop[0]/v", driven_membrane_trace, options.dt_seconds);

    // `u` is a current, not a voltage — hence its own axis scale rather than the millivolt default.
    print_heading("Recovery variable (u)");
    print_trace_plot("DrivenPop[0]/u", recovery_variable_trace, options.dt_seconds, /*axis_scale=*/1e12f, "pA");

    print_heading("Spike train");
    print_spike_times("DrivenPop[0]", driven_spike_ticks, options.dt_seconds);
    print_interspike_interval_summary(driven_spike_ticks, options.dt_seconds);

    std::cout << "\nThe u trace is what limits the firing rate: it ratchets up with every spike and\n"
              << "decays between them, so each successive spike needs a little more drive than the last.\n"
              << "That is a nonlinear cell integrating one dt per tick — never fast-forwarded.\n";

    return 0;
}
