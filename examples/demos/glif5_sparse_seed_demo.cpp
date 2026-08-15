// GLIF5 on a 100x100 torus driven by ten cells and nothing else
//
// Ten randomly chosen neurons receive a scattered spike train. The other 9,990 receive no
// external input whatsoever -- every spike they fire is caused by one of their four
// neighbours. The sheet is an excitable medium: each seed launches a front that spreads
// outward, fronts annihilate where they meet because the cells between them are still
// refractory, and GLIF5's after-spike currents and adapting threshold decide how far a
// front gets before the tissue it is crossing stops responding.
//
// The synapse strength is load-bearing here in a way it is not in the other demos. A cell
// has four neighbours and no background current, so it sits at its leak reversal and needs
// the whole 20 mV to threshold from a single arrival.
//
// The tempting figure is e * ibase * tau / C, the total charge an alphaCurrentSynapse
// delivers divided by the capacitance. That is the answer for a pure integrator and it is
// wrong here by a factor of two and a half: the cell leaks with tau_m = 10 ms while the
// alpha current peaks at tau_s = 5 ms, so half the charge has drained away before the rest
// arrives. Solving dV/dt = -(V - EL)/tau_m + I(t)/C for an alpha input gives a peak of
// about 5.3e7 * ibase volts, so 20 mV needs roughly 380 pA -- not the 147 pA the
// charge-only figure suggests. 200 pA was tried first and the sheet stayed dark except for
// the ten seeds, which is exactly what the leak predicts.
//
// 600 pA gives about 32 mV of margin over the 20 mV threshold.
//
// Produces build/demos/glif5_sparse_seed_membrane.mp4.

#include "demo_support.h"
#include "torus_model.h"

using namespace spikecorec;
using namespace spikecorec::demos;

int main() {
    begin_demo();

    TorusDemoParameters parameters;
    parameters.glif_index = 5;
    parameters.side_length = 100;
    parameters.drive = TorusDrive::SparseSeeds;
    parameters.seed_neuron_count = 10;
    parameters.seed_input_rate_hertz = 20.0;
    parameters.simulation_seconds = 1.0;

    // A front moves one cell per millisecond here (the connection delay). At the 5 ms
    // refractory every other demo uses, cells behind a front recover long before it has
    // gone anywhere, activity re-enters, and all ten thousand cells simply fire at their
    // ceiling -- measured at 146 Hz against the 200 Hz the refractory period allows. A
    // 30 ms tail is thirty cells wide, which is what makes fronts visible as fronts.
    parameters.refractory_period = "30ms";

    // ~5.3e7 * 600 pA = 32 mV against the 20 mV a GLIF5 needs from EL to thetaInf, so one
    // arrival fires one neighbour with margin and a front can travel.
    parameters.synapse_ibase_amperes = 600e-12;

    const String name = "glif5_sparse_seed";
    const String lems_path = write_torus_model(name, parameters);

    const std::vector<std::vector<s32>> adjacency = square_torus(parameters.side_length);

    const DemoResult result = run_demo(name, lems_path, adjacency, "torusSynapse",
                                       parameters.connection_weight,
                                       parameters.connection_delay_seconds,
                                       /*video_frame_stride=*/10);

    report_demo(name,
                "GLIF5 on a 100x100 square torus. Ten cells hear from outside; the other\n"
                "9,990 fire only on what arrives from their four neighbours.",
                result, parameters.simulation_seconds,
                /*render_max_frames=*/300, /*render_frames_per_second=*/30);

    // Worth stating outright, because a rate on its own would not distinguish the two: if
    // this had failed to propagate, only the ten seeds would ever fire.
    std::printf("\n  seeds driven externally   %lld of %lld cells\n",
                (long long)parameters.seed_neuron_count, (long long)result.neuron_count);
    std::printf("  cells that fired anyway   %lld\n",
                (long long)(result.participation * (f64)result.neuron_count));

    release_gpu_resources();
    return 0;
}
