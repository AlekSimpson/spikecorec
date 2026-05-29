#pragma once

#include "spikecorec/core/types.h"

#if defined(SPIKECOREC_CUDA)
#  include "spikecorec/cuda/kernels.cuh"
#elif defined(SPIKECOREC_METAL)
#  include "spikecorec/metal/kernels.h"
#endif
