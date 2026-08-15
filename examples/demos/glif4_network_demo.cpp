// GLIF4 balanced excitatory/inhibitory network, randomly wired
//
// The cell is GLIF1 plus an adapting threshold.
//
// Produces build/demos/glif4_network_membrane.mp4 (with .nml, LEMS, .spire and spikes beside it).

#include "demo_support.h"
#include "network_model.h"

using namespace spikecorec;
using namespace spikecorec::demos;

int main() {
    begin_demo();

    NetworkDemoParameters parameters;
    parameters.glif_index = 4;

    const String name = "glif4_network";
    const String lems_path = write_network_model(name, parameters);

    // The document carries its own connections, so no topology is supplied here.
    const DemoResult result = run_demo(name, lems_path, {}, "", 1.0, 0.0,
                                       /*video_frame_stride=*/10);

    report_demo(name,
                "GLIF4 balanced network: 400 excitatory + 100 inhibitory, "
                "randomly wired.\nThe cell is GLIF1 plus an adapting threshold.",
                result, parameters.simulation_seconds,
                /*render_max_frames=*/300, /*render_frames_per_second=*/30);

    release_gpu_resources();
    return 0;
}
