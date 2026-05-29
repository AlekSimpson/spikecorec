#include "spikecorec/cuda/kernels.cuh"
#include <cassert>
#include <cstdio>

int main() {
    auto cfg = spikecorec::cuda::default_config(1024);
    assert(cfg.block.x == 256);
    assert(cfg.grid.x  == 4);

    printf("cuda tests passed\n");
    return 0;
}
