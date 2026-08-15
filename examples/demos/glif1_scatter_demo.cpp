// GLIF1 on a torus under scattered spike-train input
//
// Every cell driven by its own random spike train. All five scatter demos\n// use the same trains and the same seed, so the only difference between their\n// videos is the cell type.
//
// Produces build/demos/glif1_scatter_membrane.mp4 (with .nml, LEMS, .spire and spikes beside it).

#include "demo_support.h"
#include "torus_model.h"

using namespace spikecorec;
using namespace spikecorec::demos;

int main() {
    begin_demo();

    TorusDemoParameters parameters;
    parameters.glif_index = 1;
    parameters.side_length = 64;
    parameters.drive = TorusDrive::ScatteredSpikes;
    parameters.simulation_seconds = 1.0;

    const String name = "glif1_scatter";
    const String lems_path = write_torus_model(name, parameters);

    // Connectivity from the engine's own topology rather than from the document.
    const std::vector<std::vector<s32>> adjacency = square_torus(parameters.side_length);

    const DemoResult result = run_demo(name, lems_path, adjacency, "torusSynapse",
                                       parameters.connection_weight,
                                       parameters.connection_delay_seconds,
                                       /*video_frame_stride=*/10);

    report_demo(name,
                "GLIF1 on a 64x64 square torus.\nEvery cell driven by its own random spike train. All five scatter demos\n// use the same trains and the same seed, so the only difference between their\n// videos is the cell type.",
                result, parameters.simulation_seconds,
                /*render_max_frames=*/300, /*render_frames_per_second=*/30);

    release_gpu_resources();
    return 0;
}
