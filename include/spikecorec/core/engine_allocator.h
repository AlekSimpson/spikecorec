#pragma once

#include "spikecorec/core/types.h"
#include "spikecorec/core/backend.h"

namespace spikecorec {

using PoolIndex = s64;

struct EngineAllocator {
    s64 max_pool_count;

    s64 *pool_sizes;
    s64 *pool_cursors;
    u64 **pool_cpu_memory;
    GpuPointer<u64> *pool_gpu_memory;

    EngineAllocator();
    EngineAllocator(s64 max_pools);
    ~EngineAllocator();

    u64 *allocate_cpu(u64 element_bytes, int length);
    GpuPointer<u64> allocate_gpu(u64 element_bytes, int length);
};

}





