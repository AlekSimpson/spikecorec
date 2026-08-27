//
// Created by Alek Simpson on 5/30/26.
//

#ifdef SPIKECOREC_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#include <dlfcn.h>
#endif

#include <algorithm>
#include <stdexcept>
#include <string>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/types.h"

using namespace std;

namespace spikecorec {

// The aligned start of the next range in a chunk. Every partition is pushed to at least
// PARTITION_ALIGNMENT so a `constant` kernel binding and a float4 read are both legal at
// its offset, whatever the datatype asks for.
u64 aligned_partition_offset(u64 cursor, u64 alignment) {
    return (cursor + alignment - 1) & ~(alignment - 1);
}


#ifdef SPIKECOREC_METAL

// default.metallib is built by the Makefile and placed alongside the build artifacts. A
// command line tool has no app bundle for newDefaultLibrary() to search, so locate the
// metallib next to the loaded binary (via dladdr) and load it explicitly.
MTL::Library *load_default_metal_library(MTL::Device *device) {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void *>(&load_default_metal_library), &info) && info.dli_fname) {
        String binary_path(info.dli_fname);
        const usize last_slash = binary_path.find_last_of('/');
        const String directory = (last_slash == String::npos) ? "." : binary_path.substr(0, last_slash);
        const String metallib_path = directory + "/default.metallib";

        NS::String *path_string = NS::String::string(metallib_path.c_str(), NS::UTF8StringEncoding);
        NS::URL *url = NS::URL::fileURLWithPath(path_string);
        NS::Error *error = nullptr;
        if (MTL::Library *library = device->newLibrary(url, &error)) return library;
    }
    return device->newDefaultLibrary();
}

String describe_metal_error(NS::Error *error) {
    return error
        ? String(error->localizedDescription()->utf8String()) 
        : String("unknown error");
}

MetalBackend::MetalBackend(s64 threads_per_block_) {
    // Assigned rather than initialized in the member-init list: threads_per_block belongs
    // to AbstractBackend, and a derived constructor cannot name a base's member there.
    threads_per_block = threads_per_block_;

    device = MTL::CreateSystemDefaultDevice();
    if (device == nullptr) {
        log::throw_runtime_error(log::logger(),
                "MetalBackend: no Metal device is available on this system");
    }
    queue = device->newCommandQueue();
    default_library = load_default_metal_library(device);

    log::logger().debug("MetalBackend: device ready, threads_per_block={}", threads_per_block);
}

MetalBackend::~MetalBackend() {
    deallocate_all();
    if (default_library != nullptr) default_library->release();
    if (queue != nullptr) queue->release();
    if (device != nullptr) device->release();
}

MetalBackend &MetalBackend::partition(
    u64 bytes, EngineDatatype datatype, Vector<EnginePointer> &partitions
) {
    if (memory_has_been_allocated) {
        memory_has_been_allocated = false;
        last_offset = 0;
        max_bytes = 0;
    }

    if (bytes == 0) {
        partitions.push_back(EnginePointer{});
        return *this;
    }

    const u64 alignment = std::max<u64>(type_alignments[datatype], PARTITION_ALIGNMENT);
    const u64 aligned_offset = aligned_partition_offset((u64)last_offset, alignment);

    // base_pointer stays null until allocate() runs
    partitions.push_back(EnginePointer{
        nullptr, 
        (s64)aligned_offset, 
        bytes, 
        alignment, 
        false
    });

    last_offset = (s64)(aligned_offset + bytes);
    max_bytes = (u64)last_offset;
    return *this;
}

EnginePointer MetalBackend::allocate(Vector<EnginePointer> &partitions) {
    if (memory_has_been_allocated) return base_pointer;

    memory_has_been_allocated = true;
    if (max_bytes == 0) {
        // nothing was allocated, return nullptr
        base_pointer = EnginePointer{};
        return base_pointer;
    }

    MTL::Buffer *slab = device->newBuffer(max_bytes, MTL::ResourceStorageModeShared);
    if (slab == nullptr) {
        // its possible that newBuffer returns null when something is wrong with the device
        log::throw_runtime_error(log::logger(),
                "MetalBackend::allocate: the GPU could not allocate " + to_string(max_bytes) +
                " bytes (" + to_string(max_bytes / (1024 * 1024)) + " MiB) — out of device memory");
    }

    slabs.push_back(slab);

    // Only handles from THIS chunk are still null, so a caller reusing one vector across
    // chunks does not get its earlier handles re-aimed at the new slab.
    for (EnginePointer &partition: partitions) {
        if (partition.base_pointer != nullptr || partition.total_bytes == 0) continue;
        partition.base_pointer = slab;
    }

    log::logger().debug("MetalBackend::allocate: slab of {} bytes, {} chunks live",
                        max_bytes, slabs.size());

    base_pointer = EnginePointer{slab, 0, max_bytes, PARTITION_ALIGNMENT, false};
    return base_pointer;
}

void MetalBackend::deallocate_slab(const EnginePointer &slab) {
    if (slab.base_pointer == nullptr || slab.inline_scalar) return;

    MTL::Buffer *buffer = static_cast<MTL::Buffer *>(slab.base_pointer);
    const auto entry = std::find(slabs.begin(), slabs.end(), buffer);
    if (entry == slabs.end()) return;

    slabs.erase(entry);
    buffer->release();
}

void MetalBackend::deallocate_all() {
    for (MTL::Buffer *slab : slabs) slab->release();
    slabs.clear();
    base_pointer = EnginePointer{};
    memory_has_been_allocated = false;
    last_offset = 0;
    max_bytes = 0;
}

Optional<EngineFunction> MetalBackend::create_function(
    const String &name, const String &source_code
) {
    log::logger().debug("create_function: name={} source_bytes={}", name, source_code.size());

    // The language version is stated rather than inherited. Passing null options lets the
    // runtime compiler pick a default that varies by process -- the same generated source
    // compiled from a C++ host and failed from a Python one with "unknown type name
    // 'atomic_float'", because atomic_float is Metal 3 and the default landed below it.
    // The generated kernel needs Metal 3 for its atomic float scatter, so it asks.
    MTL::CompileOptions *options = MTL::CompileOptions::alloc()->init();
    options->setLanguageVersion(MTL::LanguageVersion3_0);

    NS::Error *error = nullptr;
    MTL::Library *library = device->newLibrary(
        NS::String::string(source_code.c_str(), NS::UTF8StringEncoding),
        options,
        &error
    );
    options->release();
    if (library == nullptr) {
        log::logger().critical("create_function: newLibrary failed: {}", describe_metal_error(error));
        return std::nullopt;
    }

    MTL::Function *function = library->newFunction(
            NS::String::string(name.c_str(), NS::UTF8StringEncoding));
    library->release();
    if (function == nullptr) {
        log::logger().critical("create_function: '{}' not found in the compiled library", name);
        return std::nullopt;
    }

    error = nullptr;
    MTL::ComputePipelineState *pipeline = device->newComputePipelineState(function, &error);
    function->release();
    if (pipeline == nullptr) {
        log::logger().critical("create_function: newComputePipelineState failed: {}",
                               describe_metal_error(error));
        return std::nullopt;
    }

    return EngineFunction{pipeline};
}

Optional<EngineFunction> MetalBackend::load_precompiled_function(const String &name) {
    if (default_library == nullptr) {
        log::logger().critical("load_precompiled_function: no default.metallib was loaded");
        return std::nullopt;
    }

    MTL::Function *function = default_library->newFunction(
            NS::String::string(name.c_str(), NS::UTF8StringEncoding));
    if (function == nullptr) {
        log::logger().critical("load_precompiled_function: '{}' not found in default.metallib", name);
        return std::nullopt;
    }

    NS::Error *error = nullptr;
    MTL::ComputePipelineState *pipeline = device->newComputePipelineState(function, &error);
    function->release();
    if (pipeline == nullptr) {
        log::logger().critical("load_precompiled_function: newComputePipelineState failed: {}",
                               describe_metal_error(error));
        return std::nullopt;
    }

    return EngineFunction{pipeline};
}

void MetalBackend::release_function(EngineFunction &function) {
    if (function.pipeline_state == nullptr) return;
    function.pipeline_state->release();
    function.pipeline_state = nullptr;
}

bool MetalBackend::run_function(
    EngineFunction &function, Vector<EnginePointer> &parameters, s64 job_count
) {
    if (function.pipeline_state == nullptr) {
        log::logger().critical("run_function: the function has no pipeline state");
        return false;
    }

    MTL::CommandBuffer *command_buffer = queue->commandBuffer();
    MTL::ComputeCommandEncoder *encoder = command_buffer->computeCommandEncoder();
    encoder->setComputePipelineState(function.pipeline_state);

    for (usize index = 0; index < parameters.size(); index += 1) {
        const EnginePointer &parameter = parameters[index];

        if (parameter.inline_scalar) {
            // `constant T &x [[buffer(N)]]` accepts inline constant data as readily as a
            // buffer binding, which is what lets a scalar travel as an EnginePointer.
            encoder->setBytes(parameter.base_pointer, parameter.total_bytes, index);
        } else {
            encoder->setBuffer(static_cast<MTL::Buffer *>(parameter.base_pointer),
                               (NS::UInteger)parameter.offset, index);
        }
    }

    // dispatchThreads takes TOTAL threads, not threadgroups -- job_count is the element
    // count, and Metal handles the non-multiple tail itself.
    const s64 thread_count = std::max<s64>(job_count, 1);
    const s64 group_width = std::min<s64>(threads_per_block, thread_count);
    encoder->dispatchThreads(MTL::Size::Make((NS::UInteger)thread_count, 1, 1),
                             MTL::Size::Make((NS::UInteger)group_width, 1, 1));
    encoder->endEncoding();
    encoder->release();

    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    if (command_buffer->status() == MTL::CommandBufferStatusError) {
        const String description = describe_metal_error(command_buffer->error());
        log::logger().critical("run_function: command buffer failed: {} (job_count={}, parameters={})",
                               description, job_count, parameters.size());
        command_buffer->release();
        return false;
    }

    log::logger().trace("run_function: dispatched {} threads in groups of {}, {} parameters",
                        thread_count, group_width, parameters.size());
    command_buffer->release();
    return true;
}

#endif // SPIKECOREC_METAL

#ifdef SPIKECOREC_CUDA

String describe_cuda_driver_error(CUresult result) {
    const char *error_name = nullptr;
    const char *error_string = nullptr;
    cuGetErrorName(result, &error_name);
    error_name = error_name != nullptr ? error_name : "unknown";

    cuGetErrorString(result, &error_string);
    error_string = error_string != nullptr ? error_string : "unknown";

    return String(error_name) + " (" + String(error_string) + ")";
}

CudaBackend::CudaBackend(s64 threads_per_block_argument) {
    threads_per_block = threads_per_block_argument;

    cuInit(0); // once per process, flags must be 0
    cuDeviceGet(&cuda_gpu_device, 0); // 0 = device ordinal
    cuDevicePrimaryCtxRetain(&cuda_context, cuda_gpu_device);
    cuCtxSetCurrent(cuda_context);

    log::logger().debug("CudaBackend: context ready, threads_per_block={}", threads_per_block);
}

CudaBackend::~CudaBackend() {
    deallocate_all();
    cuDevicePrimaryCtxRelease(cuda_gpu_device);
}

CudaBackend &CudaBackend::partition(
    u64 bytes, EngineDatatype datatype, Vector<EnginePointer> &partitions
) {
    if (memory_has_been_allocated) {
        memory_has_been_allocated = false;
        last_offset = 0;
        max_bytes = 0;
    }

    if (bytes == 0) {
        partitions.push_back(EnginePointer{});
        return *this;
    }

    const u64 alignment = std::max<u64>(type_alignments[datatype], PARTITION_ALIGNMENT);
    const u64 aligned_offset = aligned_partition_offset((u64)last_offset, alignment);

    partitions.push_back(EnginePointer{nullptr, (s64)aligned_offset, bytes, alignment, false});

    last_offset = (s64)(aligned_offset + bytes);
    max_bytes = (u64)last_offset;
    return *this;
}

EnginePointer CudaBackend::allocate(Vector<EnginePointer> &partitions) {
    if (memory_has_been_allocated) return base_pointer;

    memory_has_been_allocated = true;
    if (max_bytes == 0) {
        base_pointer = EnginePointer{};
        return base_pointer;
    }

    void *slab = nullptr;
    const cudaError_t allocation_result = cudaMallocManaged(&slab, max_bytes);
    if (allocation_result != cudaSuccess || slab == nullptr) {
        log::throw_runtime_error(log::logger(),
                "CudaBackend::allocate: cudaMallocManaged failed for " + to_string(max_bytes) +
                " bytes: " + cudaGetErrorString(allocation_result));
    }
    slabs.push_back(slab);

    for (EnginePointer &partition_handle : partitions) {
        if (partition_handle.base_pointer != nullptr || partition_handle.total_bytes == 0) continue;
        partition_handle.base_pointer = slab;
    }

    log::logger().debug("CudaBackend::allocate: slab of {} bytes, {} chunks live",
                        max_bytes, slabs.size());

    base_pointer = EnginePointer{slab, 0, max_bytes, PARTITION_ALIGNMENT, false};
    return base_pointer;
}

void CudaBackend::deallocate_slab(const EnginePointer &slab) {
    if (slab.base_pointer == nullptr || slab.inline_scalar) return;

    const auto entry = std::find(slabs.begin(), slabs.end(), slab.base_pointer);
    if (entry == slabs.end()) return;

    slabs.erase(entry);
    cudaFree(slab.base_pointer);
}

void CudaBackend::deallocate_all() {
    for (void *slab : slabs) cudaFree(slab);
    slabs.clear();
    base_pointer = EnginePointer{};
    memory_has_been_allocated = false;
    last_offset = 0;
    max_bytes = 0;
}

Optional<EngineFunction> CudaBackend::create_function(
    const String &name, const String &source_code
) {
    log::logger().debug("create_function: name={} source_bytes={}", name, source_code.size());

    const String program_name = name + ".cu";

    nvrtcProgram program{};
    nvrtcCreateProgram(&program, source_code.c_str(), program_name.c_str(), 0, nullptr, nullptr);

    // TODO: const char *options[] = { "--gpu-architecture=compute_87" };   // Orin
    // Zero options, and an option COUNT of zero to match -- claiming one against an empty
    // array is undefined behaviour.
    const nvrtcResult compile_result = nvrtcCompileProgram(program, 0, nullptr);

    size_t compile_log_size = 0;
    nvrtcGetProgramLogSize(program, &compile_log_size);
    if (compile_log_size > 1) {
        Vector<char> compile_log(compile_log_size);
        nvrtcGetProgramLog(program, compile_log.data());
        log::logger().error("create_function: nvrtc log: {}", compile_log.data());
    }
    if (compile_result != NVRTC_SUCCESS) {
        log::logger().critical("create_function: nvrtcCompileProgram failed: {}",
                               nvrtcGetErrorString(compile_result));
        nvrtcDestroyProgram(&program);
        return std::nullopt;
    }

    size_t parallel_thread_executor_size = 0;
    nvrtcGetPTXSize(program, &parallel_thread_executor_size);
    Vector<char> parallel_thread_executor(parallel_thread_executor_size);
    nvrtcGetPTX(program, parallel_thread_executor.data());
    nvrtcDestroyProgram(&program);

    EngineFunction function{};

    const CUresult module_result =
            cuModuleLoadData(&function.cuda_module, parallel_thread_executor.data());
    if (module_result != CUDA_SUCCESS) {
        log::logger().critical("create_function: cuModuleLoadData failed: {}",
                               describe_cuda_driver_error(module_result));
        return std::nullopt;
    }

    const CUresult function_result =
            cuModuleGetFunction(&function.cuda_function, function.cuda_module, name.c_str());
    if (function_result != CUDA_SUCCESS) {
        log::logger().critical("create_function: '{}' not found in the compiled module: {}",
                               name, describe_cuda_driver_error(function_result));
        cuModuleUnload(function.cuda_module);
        return std::nullopt;
    }

    return function;
}

Optional<EngineFunction> CudaBackend::load_precompiled_function(const String &name) {
    // CUDA compiles its kernels at build time into the static library rather than into a
    // loadable module, so there is nothing to look up by name here. The CUDA counterparts
    // of the precompiled Metal shaders are called directly.
    log::logger().critical("load_precompiled_function: '{}' -- the CUDA backend has no "
                           "runtime-loadable precompiled module", name);
    return std::nullopt;
}

void CudaBackend::release_function(EngineFunction &function) {
    if (function.cuda_module == nullptr) return;
    cuModuleUnload(function.cuda_module);
    function.cuda_module = nullptr;
    function.cuda_function = nullptr;
}

void CudaBackend::advise_read_mostly(const EnginePointer &range, u64 byte_count) {
    if (range.is_empty() || byte_count == 0) return;
    cudaMemAdvise(range.get_contents(), byte_count, cudaMemAdviseSetReadMostly, 0);
}

void CudaBackend::prefetch_to_gpu(const EnginePointer &range, u64 byte_count) {
    if (range.is_empty() || byte_count == 0) return;
    cudaMemPrefetchAsync(range.get_contents(), byte_count, 0);
}

void CudaBackend::prefetch_to_cpu(const EnginePointer &range, u64 byte_count) {
    if (range.is_empty() || byte_count == 0) return;
    cudaMemPrefetchAsync(range.get_contents(), byte_count, cudaCpuDeviceId);
}

bool CudaBackend::run_function(
    EngineFunction &function, Vector<EnginePointer> &parameters, s64 job_count
) {
    const s64 block_count = (job_count + threads_per_block - 1) / threads_per_block;

    // cuLaunchKernel takes pointers TO each argument's value. For a slab range the value
    // is the device address, so it needs a stable cell to point at; for an inline scalar
    // the caller's own storage already is that cell.
    Vector<void *> argument_values(parameters.size(), nullptr);
    Vector<void *> argument_pointers(parameters.size(), nullptr);

    for (usize index = 0; index < parameters.size(); index += 1) {
        const EnginePointer &parameter = parameters[index];
        if (parameter.inline_scalar) {
            argument_pointers[index] = parameter.base_pointer;
        } else {
            argument_values[index] = parameter.get_contents();
            argument_pointers[index] = &argument_values[index];
        }
    }

    constexpr u32 USE_DYNAMIC_SHARED_MEMORY = 0;
    const CUresult launch_result = cuLaunchKernel(
        function.cuda_function,
        (u32)block_count, 1, 1,       // grid
        (u32)threads_per_block, 1, 1, // block
        USE_DYNAMIC_SHARED_MEMORY,
        nullptr,
        argument_pointers.data(), nullptr
    );

    if (launch_result != CUDA_SUCCESS) {
        log::logger().critical("run_function: cuLaunchKernel failed: {} "
                               "-- grid={} block={} job_count={} parameters={}",
                               describe_cuda_driver_error(launch_result),
                               block_count, threads_per_block, job_count, parameters.size());
        return false;
    }

    const cudaError_t execution_result = cudaDeviceSynchronize();
    if (execution_result != cudaSuccess) {
        log::logger().critical("run_function: cudaDeviceSynchronize failed after launch: {}",
                               cudaGetErrorString(execution_result));
        return false;
    }

    log::logger().trace("run_function: launched grid={} block={} job_count={} parameters={}",
                        block_count, threads_per_block, job_count, parameters.size());
    return true;
}

#endif // SPIKECOREC_CUDA

} // namespace spikecorec
