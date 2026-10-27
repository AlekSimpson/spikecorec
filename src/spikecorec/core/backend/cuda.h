//
// Created by Alek Simpson on 10/24/26.
//
#pragma once

#ifdef SPIKECOREC_CUDA

#include <nvrtc.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../cuda/precompiled_kernels.h"
#include "../types.h"

namespace spikecorec {

struct Engine_Function {
    CUfunction cuda_function{};
    CUmodule cuda_module{};  // null for a precompiled kernel, which the CUDA runtime owns
    // Set for the precompiled kernels, which read every inline argument through a pointer, as their
    // Metal versions read a constant-address-space argument (see cuda/precompiled_kernels.h).
    bool inline_arguments_by_reference = false;
};

struct Engine_Dispatch {
    Engine_Function *function = nullptr;
    Vector<void *> parameters;
    s64 job_count = 0;
};

struct CUDA_Device {
    CUdevice cuda_gpu_device = 0;
    CUcontext cuda_context = nullptr;
    s64 threads_per_block = 256;

    CUDA_Device(){};
};

}

#endif