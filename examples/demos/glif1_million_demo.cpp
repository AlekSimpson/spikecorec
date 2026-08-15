// GLIF1 on a 1024x1024 torus: 1,048,576 cells, 4,194,304 edges
//
// 96 recorded frames is 8 seconds at 12 fps, and at 4 MB a frame that is\n// 384 MB of recording. Render time, not disk, is what caps this: each\n// 2560x1628 frame of a million points takes about a second to draw.
//
// Produces build/demos/glif1_million_membrane.mp4 (with .nml, LEMS, .spire and spikes beside it).

#include "demo_support.h"
#include "torus_model.h"

using namespace spikecorec;
using namespace spikecorec::demos;

int main() {
    begin_demo();

    TorusDemoParameters parameters;
    parameters.glif_index = 1;
    parameters.side_length = 1024;
    parameters.drive = TorusDrive::ScatteredSpikes;
    parameters.simulation_seconds = 0.096;

    const String name = "glif1_million";
    const String lems_path = write_torus_model(name, parameters);

    // Connectivity from the engine's own topology rather than from the document.
    const std::vector<std::vector<s32>> adjacency = square_torus(parameters.side_length);

    const DemoResult result = run_demo(name, lems_path, adjacency, "torusSynapse",
                                       parameters.connection_weight,
                                       parameters.connection_delay_seconds,
                                       /*video_frame_stride=*/10);

    report_demo(name,
                "GLIF1 on a 1024x1024 square torus.\n96 recorded frames is 8 seconds at 12 fps, and at 4 MB a frame that is\n// 384 MB of recording. Render time, not disk, is what caps this: each\n// 2560x1628 frame of a million points takes about a second to draw.",
                result, parameters.simulation_seconds,
                /*render_max_frames=*/96, /*render_frames_per_second=*/12);

    release_gpu_resources();
    return 0;
}
