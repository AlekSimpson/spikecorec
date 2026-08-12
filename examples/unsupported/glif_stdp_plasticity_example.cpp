// ── NOT BUILT: STDP running on a real GLIF network ──────────────────────────────────────
//
// This program is excluded from `make examples` on purpose, for exactly the same reason
// unsupported/stdp_plasticity_example.cpp is: there is no plasticity mechanism on this
// branch. Read that file's header comment first -- it lists the three specific gaps.
//
// What this one would add on top of it: scale, and interaction. stdp_plasticity_example uses a
// deliberately minimal two-neuron pair, which is adaptation-free by construction, so it never
// shows a real cell's own after-spike-current state interacting with a weight-changing rule.
// This is the GLIF counterpart -- the SAME 8x8 GLIF3 torus glif3_torus_network_example runs,
// with STDP live for the whole run -- and it would report the driven corner's own four real
// edges before and after, so a weight visibly moves underneath a network that is still
// propagating normally.
//
// ── what is missing, exactly ─────────────────────────────────────────────────────────────
//
// Everything unsupported/stdp_plasticity_example.cpp lists, unchanged: `enable_hebbian_
// learning` only allocates `last_tick_updated`, `src/nml/kernel_codegen.cpp` emits nothing for
// stage 7, and no NeuroML synapse ComponentType maps onto a plasticity rule.
//
// Nothing ELSE here is blocked. The torus itself, its propagation, its recordings and the
// before/after `WeightMatrix::get()` reads all work today -- glif3_torus_network_example is
// this program minus the plasticity. That is precisely why it cannot ship as-is: it would run
// to completion, print two identical weights, and read as though STDP had simply had no
// effect.
//
// Depends on: NML plasticity mechanism wiring (issue #66) and synapse dynamics lowering.

#include "../glif_torus_network.h"

using namespace spikecorec;
using namespace spikecorec::examples;

int main() {
    std::cerr << "glif_stdp_plasticity_example is not buildable on this branch: the engine has\n"
                 "no plasticity mechanism. See this file's own header comment, and run\n"
                 "glif3_torus_network_example for the same network without it.\n";
    return 1;

#if 0
    // The measurement, for whenever the mechanism exists.
    //
    //     GpuContextScope gpu_context;
    //     TorusNetworkOptions torus;
    //     String model_path = write_torus_model(model_directory, GlifVariant::Glif3, torus);
    //     SpikeEngine engine(model_path, /*enable_hebbian_learning=*/true);
    //
    //     // The driven corner's four real edges, which carry the most traffic in the run.
    //     const s64 corner_neighbors[4] = {1, 7, 8, 56};   // side_length == 8
    //     spikecorec::Vector<f32> weight_before;
    //     for (s64 neighbor : corner_neighbors) weight_before.push_back(engine.weights.get(0, neighbor));
    //
    //     run_simulation(engine, /*tick_count=*/-1);
    //
    //     for (usize slot = 0; slot < 4; ++slot) {
    //         std::cout << "  0 -> " << corner_neighbors[slot] << "   " << weight_before[slot]
    //                   << " -> " << engine.weights.get(0, corner_neighbors[slot]) << "\n";
    //     }
    //
    // WeightMatrix::get() is always a real U/V reconstruction, so a moved weight there is a
    // moved weight in storage, not a bookkeeping figure kept alongside it.
#endif
}
