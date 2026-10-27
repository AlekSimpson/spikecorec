//
// Created by Alek Simpson on 10/24/26.
//
#pragma once

#ifdef SPIKECOREC_METAL

#include <Metal/Metal.hpp>

#include "../types.h"

namespace spikecorec {

struct Engine_Function {
    MTL::ComputePipelineState *pipeline_state = nullptr;
};

struct Engine_Dispatch {
    Engine_Function *function = nullptr;
    Vector<void *> parameters;
    s64 job_count = 0;
};

struct METAL_Device {
    MTL::Device *device = nullptr;
    MTL::CommandQueue *queue = nullptr;
    MTL::Library *default_library = nullptr;

    s64 threads_per_block = 256;

    METAL_Device(){};
};

}

#endif