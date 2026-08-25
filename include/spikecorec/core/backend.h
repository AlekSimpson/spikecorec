#pragma once

#include <cassert>

#ifdef SPIKECOREC_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include "spikecorec/core/types.h"

namespace spikecorec::backend {

enum class EngineDatatype {
    SIGNED32, 
    SIGNED64, 
    FLOAT32, 
    FLOAT64, 
    UNSIGNED32, 
    UNSIGNED64, 
    BOOLEAN, 
};

struct EnginePointer {
    void *base_pointer = nullptr;
    s64 offset;
    u64 total_bytes;
    u64 alignment;
};

struct EngineFunction {
#ifdef SPIKECOREC_CUDA
    CUfunction function;
#elif defined(SPIKECOREC_METAL)
    MTL::Function *function;
#endif
};

struct AbstractBackend {
    UnorderedMap<EngineDatatype, u64> type_alignments = {
        {SIGNED32, 4},
        {SIGNED64, 8},
        {FLOAT32, 4},
        {FLOAT64, 8},
        {UNSIGNED32, 4},
        {UNSIGNED64, 8},
        {BOOLEAN, 1}
    };

    s64 last_offset = 0;
    u64 max_bytes = 0;
    s64 threads_per_block;
    bool memory_has_been_allocated = false;
    EnginePointer base_pointer;
};

struct MetalBackend : AbstractBackend {
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    MTL::Buffer *base_memory_pointer;

    MetalBackend(s64 threads_per_block) : threads_per_block(threads_per_block) {
    }

    ~MetalBackend() {
        if (device != nullptr) {
            device->release();
        }
    };

    Optional<EngineFunction> create_function(const String &name, const String &source_code);
    bool run_function(
            MTL::Function *function, 
            Vector<EnginePointer> &parameters, 
            s64 job_count);

    MetalBackend &partition(
            u64 bytes, 
            EngineDatatype datatype, 
            Vector<EnginePointer> &partition_pointer);
    EnginePointer allocate();
    void deallocate_all();
};

struct CudaBackend : AbstractBackend {
    CUdevice cuda_gpu_device;
    CUcontext cuda_context;
    void *base_memory_pointer;

    CudaBackend(s64 threads_per_block) : threads_per_block(threads_per_block) {
        cuInit(0); // once per process, flags must be 0

        cuDeviceGet(&cuda_gpu_device, 0); // 0 = device ordinal

        cuDevicePrimaryCtxRetain(&cuda_context, cuda_gpu_device);
        cuCtxSetCurrent(cuda_context);
    };

    ~CudaBackend() {
        cuDevicePrimaryCtxRelease(cuda_gpu_device);
    }

    Optional<EngineFunction> create_function(const String &name, const String &source_code);
    bool run_function(
            CUfunction &function, 
            Vector<EnginePointer> &parameters, 
            s64 job_count);

    CudaBackend &partition(
            u64 bytes, 
            EngineDatatype datatype, 
            Vector<EnginePointer> &partition_pointer);
    EnginePointer allocate();
    void deallocate_all();
};

#ifdef SPIKECOREC_CUDA
using EngineBackend = CudaBackend;
#elif defined(SPIKECOREC_METAL)
using EngineBackend = MetalBackend;
#endif








