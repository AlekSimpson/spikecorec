#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <gtest/gtest.h>
#include "spikecorec/core/backend.h"

using namespace spikecorec;

int main(int argc, char **argv) {
    initialize_gpu_context();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    release_gpu_resources();
    return result;
}
