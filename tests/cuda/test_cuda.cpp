#include "spikecorec/cuda/kernels.cuh"
#include <cassert>
#include <cstdio>

int main() {
    auto config = spikecorec::cuda::default_launch_config(1024);
    assert(config.block.x == 256);
    assert(config.grid.x  == 4);

    printf("cuda tests passed\n");
    return 0;
}
