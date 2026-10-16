#include "src/spikecorec/core/engine.h"

using namespace spikecorec;

int main() {
    Spike_Engine engine = Spike_Engine("LEMS_glif1_network.xml");

    engine.run();

    engine.shutdown();

    return 0;
}
