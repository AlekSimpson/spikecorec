#pragma once

#include "spikecorec/core/types.h"
#include "spikecorec/core/backend.h"

namespace spikecorec {

struct EngineAllocator {
    u64 cpu_total_bytes;
    u64 gpu_total_bytes;
    s32 cpu_cursor = 0;
    s32 gpu_cursor = 0;

    void *pool_cpu_memory;
    GpuPointer<void> pool_gpu_memory;

    EngineAllocator(u64 cpu_total_bytes, u64 gpu_total_bytes);
    ~EngineAllocator();

    void *allocate_cpu(u64 element_bytes, s64 length);
    GpuPointer<void> allocate_gpu(u64 element_bytes, s64 length);
};

}





