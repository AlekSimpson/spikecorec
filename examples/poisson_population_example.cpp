// ── Example: an on-device stimulus generator (SpikeSourcePoisson) ───────────────────────────────
//
// Phase 1 handles stimulus by precomputing it on the host: a `pulseGenerator`'s whole trajectory is
// three resolved constants, so there is nothing to evaluate per tick on the GPU (that is what
// glif3_torus_network_example's `build_stimulus_schedule` does).
//
// A Poisson spike source is different. It has to draw a random number every tick for every neuron,
// so precomputing it on the host would mean shipping a full [neurons × ticks] array to the device.
// Instead the generator is lowered to real device code like any other ComponentType, a population is
// bound directly to it, and it is dispatched per tick exactly like a population of cells.
//
// What this example demonstrates:
//   1. `lower_type_library_to_ir(model, /*lower_inputs_on_device=*/true)` — the second stimulus path.
//      An Inputs ComponentType becomes a real IR program instead of the empty placeholder the
//      host-precomputed path leaves behind.
//   2. Seeding `rng_state`, the per-neuron xorshift32 buffer any kernel using `rand`/`randn` reads.
//      It must be nonzero per neuron: xorshift32 is stuck at zero forever once it reaches zero.
//   3. Checking the emitted rate against the declared rate statistically, which is the only kind of
//      comparison that is meaningful for a PRNG-driven source.
//
// This model declares no projections at all, so ticket #131's real per-edge synapse dispatch never
// applies here (there is nothing for it to dispatch) — the trivial ring adjacency below exists
// purely to satisfy WeightMatrix's constructor, exactly as it did before that ticket.
//
// ── A known caveat, so the numbers below are not misread ────────────────────────────────────────
// The real SpikeSourcePoisson ComponentType seeds its next-spike time at OnStart as
// `start - log(random)/rate`. allocate_model does not apply a state directive's initial_value yet, so
// the lowered version zero-initializes instead. With `start = 0` that costs one benign extra spike
// per neuron. With `start > 0` it instead fires every neuron once per tick from window onset until
// the accumulator catches up — a burst of roughly `start × rate` spurious spikes. This model uses
// `start = 0`; on-device Poisson with a nonzero start should be treated as unverified.
//
// Run:  ./build/examples/poisson_population_example [--ticks 4000] [--dt 0.0001] [--print-ir]

#include <cmath>
#include <iostream>

#include "nml_pipeline_support.h"

using namespace spikecorec;
using namespace spikecorec::nml;
using namespace spikecorec::examples;

int main(int argument_count, char **argument_values) {
    ExampleOptions options = parse_example_options(argument_count, argument_values, ExampleOptions{4000, 1e-4f});
    GpuContextScope gpu_context_scope;

    ModelSpecification model = load_model_specification("poisson_population");

    // ── 1. The on-device stimulus path ──────────────────────────────────────────────────────────
    // Note the second argument. With it false (the default everywhere else in these examples), an
    // Inputs entry would get an empty placeholder and this population would have no kernel at all.
    Vector<IrProgram> programs = lower_type_library_to_ir(model, /*lower_inputs_on_device=*/true);
    print_model_summary(model, programs);
    if (options.print_ir) print_ir_programs(model, programs);

    const s32 population_size = model.total_neuron_count;

    ModelAllocation allocation = allocate_model(model, programs);

    // WeightMatrix rejects an edge-free adjacency, and this model declares no projections — so a
    // trivial ring (neuron n → n+1, since self-loops are rejected too) exists purely to satisfy the
    // constructor. Only spike timing matters here; nothing is ever propagated.
    Vector<Vector<s32>> adjacency((usize)population_size);
    for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
        adjacency[(usize)neuron_index] = {(neuron_index + 1) % population_size};
    }
    WeightMatrix weights(adjacency, /*rank=*/1);
    weights.set_constant_weight(0.0f);

    AssembledModel assembled_model(model, programs);

    // ── 2. rng_state ────────────────────────────────────────────────────────────────────────────
    // make_live_model_buffers seeds each neuron with (index + 1) * 2654435761u | 1u — distinct and
    // guaranteed nonzero. step_tick only ever advances this state; it never re-seeds it.
    LiveModelBuffers live =
        make_live_model_buffers(allocation, weights, model.total_neuron_count, /*seed_rng_state=*/true);

    print_heading("On-device generator");
    std::cout << "  population size : " << population_size << "\n"
              << "  rng_state       : seeded, " << population_size << " nonzero xorshift32 states\n";
    for (const TypeLibraryEntry &entry : model.type_library) {
        if (entry.category != TypeLibraryCategory::Inputs) continue;
        std::cout << "  generator       : " << entry.component_type_name << "\n";
        for (const auto &baked_constant : entry.baked_constants) {
            std::cout << "      " << std::left << std::setw(12) << baked_constant.first << std::right
                      << baked_constant.second << "\n";
        }
    }

    // ── 3. Tick loop ────────────────────────────────────────────────────────────────────────────
    print_heading("Simulating");
    std::cout << "  " << options.tick_count << " ticks × " << options.dt_seconds * 1000.0f << "ms = "
              << format_seconds((f64)options.tick_count * options.dt_seconds) << "\n";

    s64 total_spike_count = 0;
    Vector<Vector<s64>> spike_ticks_by_neuron((usize)population_size);

    for (s64 tick = 0; tick < options.tick_count; ++tick) {
        assembled_model.step_tick(live.buffers, options.dt_seconds, tick, tick + 1);
        for (s32 neuron_index = 0; neuron_index < population_size; ++neuron_index) {
            if (live.buffers.last_spiked[neuron_index] == tick) {
                ++total_spike_count;
                spike_ticks_by_neuron[(usize)neuron_index].push_back(tick);
            }
        }
    }

    // ── 4. Statistical check against the declared rate ──────────────────────────────────────────
    const f64 simulated_seconds = (f64)options.tick_count * (f64)options.dt_seconds;
    const f64 observed_rate_hertz = (f64)total_spike_count / ((f64)population_size * simulated_seconds);

    s32 neurons_that_fired = 0;
    for (const auto &spike_ticks : spike_ticks_by_neuron) {
        if (!spike_ticks.empty()) ++neurons_that_fired;
    }

    print_heading("Results");
    std::cout << "  simulated window   " << format_seconds(simulated_seconds) << "\n"
              << "  total spikes       " << total_spike_count << "\n"
              << "  neurons that fired " << neurons_that_fired << " / " << population_size << "\n"
              << "  observed rate      " << std::fixed << std::setprecision(2) << observed_rate_hertz
              << " Hz per neuron\n";
    std::cout.unsetf(std::ios::floatfield);

    for (const TypeLibraryEntry &entry : model.type_library) {
        if (entry.category != TypeLibraryCategory::Inputs) continue;
        auto declared_rate = entry.baked_constants.find("rate");
        if (declared_rate == entry.baked_constants.end()) continue;

        const f64 declared_rate_hertz = declared_rate->second;
        // The OnStart seeding gap (see this file's header comment) costs exactly one extra spike per
        // neuron at `start = 0` — every neuron fires once on tick 0, which is the solid leading
        // column in the raster below. Subtracting it recovers the rate the generator is really
        // running at, so the two numbers can be compared honestly rather than looking like a bug.
        const f64 rate_excluding_onset_spike =
            (f64)(total_spike_count - population_size) / ((f64)population_size * simulated_seconds);

        std::cout << "  declared rate      " << declared_rate_hertz << " Hz per neuron\n"
                  << "  minus onset spike  " << std::fixed << std::setprecision(2)
                  << rate_excluding_onset_spike << " Hz per neuron"
                  << "   (one spurious tick-0 spike per neuron — see header comment)\n";
        std::cout.unsetf(std::ios::floatfield);
    }

    print_heading("Spike raster (first 16 neurons)");
    Vector<Vector<s64>> raster_subset(
        spike_ticks_by_neuron.begin(),
        spike_ticks_by_neuron.begin() + (long)std::min<usize>(spike_ticks_by_neuron.size(), 16));
    print_spike_raster(raster_subset, options.tick_count);

    std::cout << "\nEach neuron draws independently from its own PRNG stream, so this will never match\n"
              << "jNeuroML spike for spike — only in aggregate. See this file's header comment for the\n"
              << "OnStart seeding caveat before trusting a run with a nonzero `start`.\n";

    return 0;
}
