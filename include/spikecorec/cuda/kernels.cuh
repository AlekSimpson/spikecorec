#pragma once

#include "spikecorec/core/types.h"
#include <cuda_runtime.h>

namespace spikecorec::cuda {

// Launch parameters
struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem = 0;
    cudaStream_t stream = nullptr;
};

LaunchConfig default_config(usize n, usize threads_per_block = 256);

} // namespace spikecorec::cuda
