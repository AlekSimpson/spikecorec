//
// Created by Alek Simpson on 5/30/26.
//

#ifdef SPIKECOREC_CUDA
#include <cuda.h>
#include <nvrtc.h>
#include <stdexcept>
#include <string>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#include <dlfcn.h>
#include <stdexcept>
#endif

#include "spikecorec/core/backend.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/types.h"

using namespace std;
using namespace spikecorec::log;

namespace spikecorec::backend {

/// mac code

MetalBackend &MetalBackend::partition(
    u64 bytes, EngineDatatype datatype, Vector<EnginePointer> &partitions
) {
    if (memory_has_been_allocated) {
        logger().warn(
            "Partitioning data when memory has already been allocated.");
        return this;
    }

    EnginePointer new_partition;
    new_partition.base_pointer = base_memory_pointer->get_contents();
    new_partition.offset = (partitions.empty())
        ? 0
        : partitions.back().offset + partitions.back().total_bytes;
    new_partition.total_bytes = bytes;
    new_partition.alignment = type_alignments[datatype];
    partitions.push_back(new_partition);
    max_bytes += bytes;
    return this;
    
}

void *MetalBackend::allocate() {
    if (memory_has_been_allocated) return nullptr;

    memory_has_been_allocated = true;
    base_memory_pointer = device->newBuffer(
        max_bytes, 
        MTL::ResourceStorageModeShared
    );

    return EnginePointer{base_memory_pointer->get_contents(), 0, max_bytes, 8};
}

void MetalBackend::deallocate_all() {
    if (base_memory_pointer != nullptr) {
        base_memory_pointer->release();
    }
    memory_has_been_allocated = false;
}

EngineFunction MetalBackend::create_function(
    const String &name, const String &source_code
) {
    auto swift_string = [](const String &text) {
        return NS::String::string(text, NS::UTF8StringEncoding);
    };

    NS::Error *error = nullptr;

    MTL::Library *library = device->newLibrary(
            swift_string(source_code), nullptr, &error);

    MTL::Function *function = library->newFunction(swift_string(name));
    return EngineFunction{function};
}

bool MetalBackend::run_function(
    MTL::Function *function, Vector<EnginePointer> &parameters, s64 job_count
) {
    NS::Error* error = nullptr;

    MTL::CommandQueue *queue = device->newCommandQueue();
    MTL::CommandBuffer *command_buffer = queue->commandBuffer();
    MTL::ComputeCommandEncoder *encoder = command_buffer->computeCommandEncoder();

    MTL::ComputePipelineState *pipeline =
        device->newComputePipelineState(function, &error);
    
    encoder->setComputePipelineState(pipeline);

    s64 index = 0;
    for (EnginePointer parameter: parameters) {
        encoder->setBuffer(parameter.base_pointer, parameter.offset, index);
        index++;
    }

    encoder->dispatchThreads(MTL::Size(element_count, 1, 1),
                             MTL::Size(threads_per_group, 1, 1));
    encoder->endEncoding();
    
    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    if (error == nullptr) {
        // TODO: do error logging here and stuff
        return false;
    }

    return true;
}

/// cuda code

CudaBackend &CudaBackend::partition(
    u64 bytes, EngineDatatype datatype, Vector<EnginePointer> &partitions
) {
    if (memory_has_been_allocated) {
        logger().warn(
            "Partitioning data when memory has already been allocated.");
        return this;
    }

    EnginePointer new_partition;
    new_partition.offset = (partitions.empty())
        ? 0
        : partitions.back().offset + partitions.back().total_bytes;
    new_partitions.total_bytes = bytes;
    new_partitions.alignment = type_alignments[datatype];
    new_partitions.base_pointer = base_memory_pointer;

    return this;
}

EnginePointer CudaBackend::allocate() {
    if (memory_has_been_allocated) return nullptr;

    memory_has_been_allocated = true;
    cudaMallocManaged(&base_memory_pointer, max_bytes);
    return EnginePointer{base_memory_pointer, 0, max_bytes, 8};
}

void CudaBackend::deallocate_all() {
    if (base_memory_pointer != nullptr) {
        cudaFree(base_memory_pointer);
    }
    memory_has_been_allocated = false;
}

EngineFunction CudaBackend::create_function(
    const String &name, const String &source_code
) {
    nvrtcProgram program;
    nvrtcCreateProgram(&program, source_code, combine(name, ".cu"), 0, nullptr, nullptr);
    
    // TODO: const char *options[] = { "--gpu-architecture=compute_87" };   // Orin
    const char *options[] = {};
    nvrtcResult compile_result = nvrtcCompileProgram(program, 1, options);
    
    size_t log_size;
    nvrtcGetProgramLogSize(program, &log_size);
    if (log_size > 1) {
        Vector<char> log(log_size);
        nvrtcGetProgramLog(program, log.data());
        logger().error("{}", log.data());
    }
    if (compile_result != NVRTC_SUCCESS) { 
        return EngineFunction{};
    }
    
    size_t parallel_thread_executor_size;
    nvrtcGetPTXSize(program, &parallel_thread_executor_size);
    Vector<u8> parallel_thread_executor(parallel_thread_executor_size);
    nvrtcGetPTX(program, parallel_thread_executor.data());
    nvrtcDestroyProgram(&program);
    
    CUmodule module;
    cuModuleLoadData(&module, parallel_thread_executor.data());
    
    CUfunction kernel;
    cuModuleGetFunction(&kernel, module, name);

    return EngineFunction{kernel};
}

bool CudaBackend::run_function(
    CUfuntion &function, Vector<EnginePointer> &parameters, s32 job_count
) {
    s64 block_count = (job_count + threads_per_block - 1) / threads_per_block;

    void **parameter_pointers = new void*[parameters.size()];
    void **parameters_ = new void*[parameters.size() + 1];
    size_t index = 0;
    for (EnginePointer parameter: parameters) {
        u8 *base = static_cast<u8 *>(parameter.base_pointer);
        parameter_pointers[index] = base + parameter.offset;
        parameters[index] = &parameter_pointers[index];
        index++;
    }

    parameters_[parameters.size()] = &job_count;
    
    constexpr s32 USE_DYNAMIC_SHARED_MEMORY = 0;
    CUresult launch_result = cuLaunchKernel(
        function,
        block_count, 1, 1,         // grid
        threads_per_block, 1, 1,   // block
        USE_DYNAMIC_SHARED_MEMORY, // dynamic shared memory
        nullptr,
        parameters_, nullptr
    );
    
    cudaError_t execution_result = cudaDeviceSynchronize();

    delete[] parameter_pointers;
    delete[] parameters_;

    // TODO: add logging around the launch result for both error and success scenarios
    // return launch_result;

    return true;
}



} // namespace spikecorec
