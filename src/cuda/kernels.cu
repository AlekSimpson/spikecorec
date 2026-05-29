#include "spikecorec/cuda/kernels.cuh"

namespace spikecorec::cuda {

LaunchConfig default_config(usize n, usize threads_per_block) {
    LaunchConfig cfg;
    cfg.block = dim3(static_cast<unsigned>(threads_per_block));
    cfg.grid  = dim3(static_cast<unsigned>((n + threads_per_block - 1) / threads_per_block));
    return cfg;
}

} // namespace spikecorec::cuda
