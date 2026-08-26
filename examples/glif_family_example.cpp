// One cell of each GLIF type under the same current step. Prints each cell's spike count,
// its first interspike interval and its last, which is where the family's differences show:
// GLIF1 and GLIF2 fire at a constant rate, while the after-spike currents (GLIF3),
// the adapting threshold (GLIF4) and both together (GLIF5) stretch the interval out.

#include <cstdio>
#include <vector>

#include "spikecorec/core/engine.h"
#include "spikecorec/core/backend.h"

using namespace spikecorec;

int main(int argument_count, char **arguments) {
    const std::string model_path = argument_count > 1
            ? arguments[1] : "tests/fixtures/nml/LEMS_glif_family.xml";


    SpikeEngine engine(model_path);
    engine.run();
    engine.write_recordings();

    const char *names[5] = {"GLIF1", "GLIF2", "GLIF3", "GLIF4", "GLIF5"};

    std::printf("\n=== GLIF family, 500 pA step (2.5x rheobase), 500 ms ===\n");
    std::printf("%-7s %7s %12s %12s %10s\n",
                "type", "spikes", "first ISI", "last ISI", "adaptation");

    for (s64 cell = 0; cell < 5; cell += 1) {
        std::vector<double> times;
        for (const RecordedSpike &spike : engine.recorded_spikes) {
            if (spike.neuron_index == cell) times.push_back(spike.time_seconds);
        }

        if (times.size() < 3) {
            std::printf("%-7s %7zu %12s %12s %10s\n", names[cell], times.size(), "-", "-", "-");
            continue;
        }

        const double first = times[1] - times[0];
        const double last = times[times.size() - 1] - times[times.size() - 2];
        std::printf("%-7s %7zu %10.2f ms %10.2f ms %9.2fx\n",
                    names[cell], times.size(), first * 1e3, last * 1e3, last / first);
    }

    std::printf("\nmean firing rate across all five: %.2f Hz\n",
                engine.mean_firing_rate_hertz());

    return 0;
}
