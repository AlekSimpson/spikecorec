#pragma once

#include "spikecorec/core/types.h"

namespace spikecorec {

// CPU-side main loop. Owns program state and drives the backend.
class Engine {
public:
    Engine();
    ~Engine();

    // Initializes the GPU backend and any CPU-side state.
    void init();

    // Runs one iteration of the main loop.
    void step();

    // Tears down the backend cleanly.
    void shutdown();

private:
    bool running_ = false;
};

} // namespace spikecorec
