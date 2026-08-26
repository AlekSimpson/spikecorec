// GLIF3 torus under a wandering patch step drive
//
// The driven disc moves every 200 ms, so the network is chased rather\n// than settling.
//
// Produces build/demos/torus_wandering_patch_membrane.mp4 (with .nml, LEMS, .spire and spikes beside it).

#include "demo_support.h"
#include "torus_model.h"

using namespace spikecorec;
using namespace spikecorec::demos;

int main() {
    begin_demo();

    TorusDemoParameters parameters;
    parameters.glif_index = 3;
    parameters.side_length = 48;
    parameters.drive = TorusDrive::WanderingPatch;
    parameters.simulation_seconds = 1.0;

    const String name = "torus_wandering_patch";
    const String lems_path = write_torus_model(name, parameters);

    // Connectivity from the engine's own topology rather than from the document.
    const std::vector<std::vector<s32>> adjacency = square_torus(parameters.side_length);

    const DemoResult result = run_demo(name, lems_path, adjacency, "torusSynapse",
                                       parameters.connection_weight,
                                       parameters.connection_delay_seconds,
                                       /*video_frame_stride=*/10);

    report_demo(name,
                "GLIF3 on a 48x48 square torus.\nThe driven disc moves every 200 ms, so the network is chased rather\n// than settling.",
                result, parameters.simulation_seconds,
                /*render_max_frames=*/300, /*render_frames_per_second=*/30);

    return 0;
}
