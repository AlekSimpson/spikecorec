#pragma once

#include "spikecorec/core/types.h"

// CPU-side interface that core/ calls to dispatch work to the active GPU backend.
// Implemented in src/cuda/ or src/metal/ depending on the compile-time BACKEND flag.

namespace spikecorec::backend {

// Called once at program startup to initialize the GPU context.
void init();

// Called once at program shutdown to release GPU resources.
void shutdown();

// Example dispatch — replace with your actual kernel signatures.
// void run_step(const f32* input, f32* output, usize n);

} // namespace spikecorec::backend
