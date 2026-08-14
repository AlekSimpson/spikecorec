// One iafCell under constant current, run from its LEMS document and checked against the
// closed-form solution of the same equation. If the generated kernel integrates anything
// other than dv/dt = (gL*(EL - v) + I)/C, the interspike interval printed here will not be
// the analytic one.

#include <cmath>
#include <cstdio>

#include "spikecorec/core/engine.h"
#include "spikecorec/core/backend.h"

using namespace spikecorec;

int main(int argument_count, char **arguments) {
    const std::string model_path = argument_count > 1
            ? arguments[1]
            : "tests/fixtures/nml/LEMS_iaf_single_cell.xml";

    initialize_gpu_context();

    SpikeEngine engine(model_path);
    engine.run();
    engine.write_recordings();

    // tau * ln((I/gL - (reset - EL)) / (I/gL - (thresh - EL)))
    const double membrane_time_constant = 100e-12 / 5e-9;
    const double drive_in_volts = 90e-12 / 5e-9;
    const double expected_interval = membrane_time_constant *
            std::log((drive_in_volts - (-0.070 - -0.065)) /
                     (drive_in_volts - (-0.050 - -0.065)));

    std::printf("\n=== iafCell under 90 pA ===\n");
    std::printf("spikes recorded          : %zu\n", engine.recorded_spikes.size());
    std::printf("analytic interspike gap  : %.6f s\n", expected_interval);

    if (engine.recorded_spikes.size() >= 3) {
        const double measured =
                (engine.recorded_spikes[2].time_seconds -
                 engine.recorded_spikes[1].time_seconds);
        std::printf("measured interspike gap  : %.6f s\n", measured);
        std::printf("relative error           : %.4f %%\n",
                    100.0 * std::fabs(measured - expected_interval) / expected_interval);
    }

    std::printf("mean firing rate         : %.3f Hz\n", engine.mean_firing_rate_hertz());
    std::printf("final membrane potential : %.6f V\n", engine.read_state_variable(0, "v"));

    release_gpu_resources();
    return 0;
}
