// ── NOT BUILT: spike-timing-dependent plasticity on a two-neuron pair ───────────────────
//
// This program is excluded from `make examples` on purpose. It is here as the shape the
// example will take, and as a precise record of what has to land first.
//
// What it would demonstrate: STDP -- an edge whose weight moves as a function of the RELATIVE
// TIMING of the spikes at its two ends. A pre-before-post (causal) pairing potentiates it; a
// post-before-pre (anti-causal) pairing depresses it. The measurement is the whole example:
// build one pair, drive it causally, read WeightMatrix::get() before and after; then drive
// the same pair anti-causally and read it again. Two numbers moving in opposite directions is
// the demonstration.
//
// ── what is missing, exactly ─────────────────────────────────────────────────────────────
//
// There is no plasticity mechanism on this branch at all. Not a partial one -- none.
//
//  1. `SpikeEngine`'s constructor takes an `enable_hebbian_learning` flag, and that flag's
//     ONLY effect is to allocate the `last_tick_updated` buffer (src/core/engine.cpp). No
//     other line of the engine reads it.
//
//  2. `src/nml/kernel_codegen.cpp` emits nothing for plasticity. The generated master kernel
//     covers stages 2-5 (Integrate, Detect, Emit, Reset) plus the propagate/delay-ring
//     boilerplate; stage 7, Plasticity, has no emitter, and `last_tick_updated` is never read
//     by generated code. Grepping the whole of src/ for a weight-update call reaching the
//     tick loop finds nothing.
//
//  3. Nothing maps a NeuroML synapse ComponentType onto a plasticity rule. The vendored
//     `stdpSynapse` is marked "EXAMPLE NOT YET WORKING!!!!" in the standard library and
//     declares none of tauPlus/tauMinus/aPlus/aMinus, so a real rule would have to come from
//     a hand-authored ComponentType -- and there is no path from one to a weight update
//     either way, since synapse ComponentTypes are not lowered at all yet (see
//     examples/README.md).
//
// So this cannot be ported honestly. An example that ran a network with
// `enable_hebbian_learning=true` and printed weights before and after would print the SAME
// number twice, and calling that a demonstration of STDP would be a lie about the engine. The
// body below is the measurement it should make, written against the API as it would have to
// exist; it does not compile or run today.
//
// Depends on: NML plasticity mechanism wiring (issue #66) and synapse dynamics lowering.

#include "../example_support.h"

using namespace spikecorec;
using namespace spikecorec::examples;

int main() {
    std::cerr << "stdp_plasticity_example is not buildable on this branch: the engine has no\n"
                 "plasticity mechanism. See this file's own header comment.\n";
    return 1;

#if 0
    // The measurement, for whenever the mechanism exists. Two runs over one pre -> post edge:
    // one driven so the pre-synaptic cell leads, one so it lags.
    //
    //     GpuContextScope gpu_context;
    //     for (bool causal_pairing : {true, false}) {
    //         String model_path = write_spike_pair_model(model_directory, causal_pairing);
    //         SpikeEngine engine(model_path, /*enable_hebbian_learning=*/true);
    //
    //         const f32 weight_before = engine.weights.get(0, 1);
    //         run_simulation(engine, /*tick_count=*/-1);
    //         const f32 weight_after = engine.weights.get(0, 1);
    //
    //         std::cout << (causal_pairing ? "  pre before post: " : "  post before pre: ")
    //                   << weight_before << " -> " << weight_after << "\n";
    //         engine.shutdown();
    //     }
    //
    // Causal must end higher than it started and anti-causal lower; anything else is not STDP,
    // whatever the numbers move by.
#endif
}
