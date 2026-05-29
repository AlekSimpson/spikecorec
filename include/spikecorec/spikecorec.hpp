#pragma once

#include "spikecorec/core/types.hpp"

#if defined(SPIKECOREC_CUDA)
#  include "spikecorec/cuda/kernels.cuh"
#elif defined(SPIKECOREC_METAL)
#  include "spikecorec/metal/kernels.hpp"
#endif
