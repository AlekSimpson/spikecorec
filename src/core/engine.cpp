#include "spikecorec/core/engine.h"
#include "spikecorec/core/backend.h"

namespace spikecorec {

Engine::Engine() = default;
Engine::~Engine() { if (running_) shutdown(); }

void Engine::init() {
    this->running_ = true;
}

void Engine::step() {
    // Drive the GPU backend each tick.
    // Replace with actual kernel dispatches, e.g.:
    //   backend::run_step(input_buf_, output_buf_, n_);
}

void Engine::shutdown() {
    this->running_ = false;
}

} // namespace spikecorec
