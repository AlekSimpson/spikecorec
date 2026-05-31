#include "spikecorec/core/engine.h"
#include "spikecorec/core/backend.h"

namespace spikecorec {

SpikeEngine::SpikeEngine() = default;
SpikeEngine::~SpikeEngine() { if (running) shutdown(); }

void SpikeEngine::init() {
}

void SpikeEngine::step() {
    // Drive the GPU backend each tick.
    // Replace with actual kernel dispatches, e.g.:
    //   backend::run_step(input_buf_, output_buf_, n_);
}

void SpikeEngine::shutdown() {
}

} // namespace spikecorec
