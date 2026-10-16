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
#include <string>

#include "backend.h"
#include "log.h"
#include "types.h"

using namespace std;

namespace spikecorec {

u64 aligned_partition_offset(u64 cursor, u64 alignment) {
    return (cursor + alignment - 1) & ~(alignment - 1);
}


#ifdef SPIKECOREC_METAL

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

// Metal returns command buffers, encoders, errors and strings autoreleased: the caller does not
// own them. This pool frees them when the scope ends, on whatever thread runs it, instead of in
// the thread's own pool (which on the main thread never drains).
struct Autorelease_Pool_Scope {
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
    Autorelease_Pool_Scope() = default;
    Autorelease_Pool_Scope(const Autorelease_Pool_Scope &) = delete;
    Autorelease_Pool_Scope &operator=(const Autorelease_Pool_Scope &) = delete;
    ~Autorelease_Pool_Scope() { pool->release(); }
};

Engine_Backend::Engine_Backend(s64 threads_per_block_) {
    threads_per_block = threads_per_block_;

    device = MTL::CreateSystemDefaultDevice();
    if (device == nullptr) {
        log::throw_runtime_error(log::logger(),
                "Metal_Backend: no Metal device is available on this system");
    }
    queue = device->newCommandQueue();
    default_library = load_default_metal_library(device);

    log::logger().debug("Metal_Backend: device ready, threads_per_block={}", threads_per_block);
}

Engine_Backend::~Engine_Backend() {
    deallocate_all();
    if (default_library != nullptr) default_library->release();
    if (queue != nullptr) queue->release();
    if (device != nullptr) device->release();
}

Engine_Backend &Engine_Backend::partition(
    u64 bytes, Engine_Datatype datatype, Vector<Engine_Pointer> &partitions
) {
    if (memory_has_been_allocated) {
        memory_has_been_allocated = false;
        last_offset = 0;
        max_bytes = 0;
    }

    if (bytes == 0) {
        partitions.push_back(Engine_Pointer{});
        return *this;
    }

    const u64 alignment = std::max<u64>(TYPE_ALIGNMENTS.at(datatype), PARTITION_ALIGNMENT);
    const u64 aligned_offset = aligned_partition_offset(static_cast<u64>(last_offset), alignment);

    // base_pointer stays null until allocate() runs
    partitions.push_back(Engine_Pointer{
        nullptr, 
        static_cast<s64>(aligned_offset), 
        bytes, 
        alignment, 
        false
    });

    last_offset = static_cast<s64>(aligned_offset + bytes);
    max_bytes = static_cast<u64>(last_offset);
    return *this;
}

Engine_Pointer Engine_Backend::allocate(Vector<Engine_Pointer> &partitions) {
    if (memory_has_been_allocated) return base_pointer;

    memory_has_been_allocated = true;
    if (max_bytes == 0) {
        base_pointer = Engine_Pointer{};
        return base_pointer;
    }

    MTL::Buffer *slab = device->newBuffer(max_bytes, MTL::ResourceStorageModeShared);
    if (slab == nullptr) {
        log::throw_runtime_error(log::logger(),
                "Metal_Backend::allocate: the GPU could not allocate " + to_string(max_bytes) +
                " bytes (" + to_string(max_bytes / (1024 * 1024)) + " MiB) — out of device memory");
    }

    slabs.push_back(slab);
    allocated_byte_count += max_bytes;
    peak_allocated_byte_count = std::max(peak_allocated_byte_count, allocated_byte_count);

    for (Engine_Pointer &partition: partitions) {
        if (partition.base_pointer != nullptr || partition.total_bytes == 0) continue;
        partition.base_pointer = slab;
    }

    log::logger().debug("Metal_Backend::allocate: slab of {} bytes, {} chunks live",
                        max_bytes, slabs.size());

    base_pointer = Engine_Pointer{slab, 0, max_bytes, PARTITION_ALIGNMENT, false};
    return base_pointer;
}

void Engine_Backend::deallocate_slab(const Engine_Pointer &slab) {
    if (slab.base_pointer == nullptr || slab.inline_scalar) return;

    auto *buffer = static_cast<MTL::Buffer *>(slab.base_pointer);
    const auto entry = std::find(slabs.begin(), slabs.end(), buffer);
    if (entry == slabs.end()) return;

    slabs.erase(entry);
    allocated_byte_count -= std::min<u64>(allocated_byte_count, buffer->length());
    buffer->release();
}

void Engine_Backend::deallocate_all() {
    for (MTL::Buffer *slab : slabs) slab->release();
    slabs.clear();
    allocated_byte_count = 0;
    base_pointer = Engine_Pointer{};
    memory_has_been_allocated = false;
    last_offset = 0;
    max_bytes = 0;
}

Optional<Engine_Function> Engine_Backend::create_function(
    const String &name, const String &source_code
) const {
    log::logger().debug("create_function: name={} source_bytes={}", name, source_code.size());
    const Autorelease_Pool_Scope autorelease_pool;

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

    return Engine_Function{pipeline};
}

Optional<Engine_Function> Engine_Backend::load_precompiled_function(const String &name) const {
    const Autorelease_Pool_Scope autorelease_pool;
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

    return Engine_Function{pipeline};
}

void Engine_Backend::release_function(Engine_Function &function) {
    if (function.pipeline_state == nullptr) return;
    function.pipeline_state->release();
    function.pipeline_state = nullptr;
}

bool Engine_Backend::run_function(
    Engine_Function &function, Vector<Engine_Pointer> &parameters, s64 job_count
) const {
    if (function.pipeline_state == nullptr) {
        log::logger().critical("run_function: the function has no pipeline state");
        return false;
    }

    // Both are autoreleased, so this pool, not an explicit release(), frees them.
    const Autorelease_Pool_Scope autorelease_pool;
    MTL::CommandBuffer *command_buffer = queue->commandBuffer();
    MTL::ComputeCommandEncoder *encoder = command_buffer->computeCommandEncoder();
    encoder->setComputePipelineState(function.pipeline_state);

    for (usize index = 0; index < parameters.size(); index += 1) {
        const Engine_Pointer &parameter = parameters[index];

        if (parameter.inline_scalar) {
            encoder->setBytes(parameter.base_pointer, parameter.total_bytes, index);
        } else {
            encoder->setBuffer(static_cast<MTL::Buffer *>(parameter.base_pointer), parameter.offset, index);
        }
    }

    const s64 thread_count = std::max<s64>(job_count, 1);
    const s64 group_width = std::min<s64>(threads_per_block, thread_count);
    encoder->dispatchThreads(MTL::Size::Make(thread_count, 1, 1),
                             MTL::Size::Make(group_width, 1, 1));
    encoder->endEncoding();

    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    if (command_buffer->status() == MTL::CommandBufferStatusError) {
        const String description = describe_metal_error(command_buffer->error());
        log::logger().critical("run_function: command buffer failed: {} (job_count={}, parameters={})",
                               description, job_count, parameters.size());
        return false;
    }

    log::logger().trace("run_function: dispatched {} threads in groups of {}, {} parameters",
                        thread_count, group_width, parameters.size());
    return true;
}

bool Engine_Backend::run_functions(Vector<Engine_Dispatch> &dispatches) const {
    if (dispatches.empty()) return true;
    for (const Engine_Dispatch &dispatch : dispatches) {
        if (dispatch.function == nullptr || dispatch.function->pipeline_state == nullptr) {
            log::logger().critical("run_functions: a launch has no pipeline state");
            return false;
        }
    }

    // One serial encoder, which runs the launches in order, with a buffer barrier between each so
    // a launch sees what the ones before it wrote. An encoder per launch costs about 30 us each.
    const Autorelease_Pool_Scope autorelease_pool;
    MTL::CommandBuffer *command_buffer = queue->commandBuffer();
    MTL::ComputeCommandEncoder *encoder = command_buffer->computeCommandEncoder(MTL::DispatchTypeSerial);
    for (usize dispatch_index = 0; dispatch_index < dispatches.size(); dispatch_index += 1) {
        Engine_Dispatch &dispatch = dispatches[dispatch_index];
        if (dispatch_index > 0) encoder->memoryBarrier(MTL::BarrierScopeBuffers);
        encoder->setComputePipelineState(dispatch.function->pipeline_state);
        for (usize index = 0; index < dispatch.parameters.size(); index += 1) {
            const Engine_Pointer &parameter = dispatch.parameters[index];
            if (parameter.inline_scalar) {
                encoder->setBytes(parameter.base_pointer, parameter.total_bytes, index);
            } else {
                encoder->setBuffer(static_cast<MTL::Buffer *>(parameter.base_pointer), parameter.offset, index);
            }
        }
        const s64 thread_count = std::max<s64>(dispatch.job_count, 1);
        const s64 group_width = std::min<s64>(threads_per_block, thread_count);
        encoder->dispatchThreads(MTL::Size::Make(thread_count, 1, 1), MTL::Size::Make(group_width, 1, 1));
    }
    encoder->endEncoding();

    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    if (command_buffer->status() == MTL::CommandBufferStatusError) {
        const String description = describe_metal_error(command_buffer->error());
        log::logger().critical("run_functions: command buffer of {} launches failed: {}", dispatches.size(), description);
        return false;
    }
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

Engine_Backend::Engine_Backend(s64 threads_per_block_argument) {
    threads_per_block = threads_per_block_argument;

    cuInit(0); // flags must be 0
    cuDeviceGet(&cuda_gpu_device, 0); // device ordinal 0
    cuDevicePrimaryCtxRetain(&cuda_context, cuda_gpu_device);
    cuCtxSetCurrent(cuda_context);

    log::logger().debug("Cuda_Backend: context ready, threads_per_block={}", threads_per_block);
}

Engine_Backend::~Engine_Backend() {
    deallocate_all();
    cuDevicePrimaryCtxRelease(cuda_gpu_device);
}

Engine_Backend &Engine_Backend::partition(
    u64 bytes, Engine_Datatype datatype, Vector<Engine_Pointer> &partitions
) {
    if (memory_has_been_allocated) {
        memory_has_been_allocated = false;
        last_offset = 0;
        max_bytes = 0;
    }

    if (bytes == 0) {
        partitions.push_back(Engine_Pointer{});
        return *this;
    }

    const u64 alignment = std::max<u64>(TYPE_ALIGNMENTS.at(datatype), PARTITION_ALIGNMENT);
    const u64 aligned_offset = aligned_partition_offset(static_cast<u64>(last_offset), alignment);

    partitions.push_back(Engine_Pointer{nullptr, static_cast<s64>(aligned_offset), bytes, alignment, false});

    last_offset = static_cast<s64>(aligned_offset + bytes);
    max_bytes = static_cast<u64>(last_offset);
    return *this;
}

Engine_Pointer Engine_Backend::allocate(Vector<Engine_Pointer> &partitions) {
    if (memory_has_been_allocated) return base_pointer;

    memory_has_been_allocated = true;
    if (max_bytes == 0) {
        base_pointer = Engine_Pointer{};
        return base_pointer;
    }

    void *slab = nullptr;
    const cudaError_t allocation_result = cudaMallocManaged(&slab, max_bytes);
    if (allocation_result != cudaSuccess || slab == nullptr) {
        log::throw_runtime_error(log::logger(),
                "Engine_Backend::allocate: cudaMallocManaged failed for " + to_string(max_bytes) +
                " bytes: " + cudaGetErrorString(allocation_result));
    }
    slabs.push_back(slab);
    slab_byte_counts.push_back(max_bytes);
    allocated_byte_count += max_bytes;
    peak_allocated_byte_count = std::max(peak_allocated_byte_count, allocated_byte_count);

    for (Engine_Pointer &partition_handle : partitions) {
        if (partition_handle.base_pointer != nullptr || partition_handle.total_bytes == 0) continue;
        partition_handle.base_pointer = slab;
    }

    log::logger().debug("Cuda_Backend::allocate: slab of {} bytes, {} chunks live",
                        max_bytes, slabs.size());

    base_pointer = Engine_Pointer{slab, 0, max_bytes, PARTITION_ALIGNMENT, false};
    return base_pointer;
}

void Engine_Backend::deallocate_slab(const Engine_Pointer &slab) {
    if (slab.base_pointer == nullptr || slab.inline_scalar) return;

    const auto entry = std::find(slabs.begin(), slabs.end(), slab.base_pointer);
    if (entry == slabs.end()) return;

    const usize slab_index = static_cast<usize>(entry - slabs.begin());
    allocated_byte_count -= std::min<u64>(allocated_byte_count, slab_byte_counts[slab_index]);
    slab_byte_counts.erase(slab_byte_counts.begin() + static_cast<s64>(slab_index));
    slabs.erase(entry);
    cudaFree(slab.base_pointer);
}

void Engine_Backend::deallocate_all() {
    for (void *slab : slabs) cudaFree(slab);
    slabs.clear();
    slab_byte_counts.clear();
    allocated_byte_count = 0;
    base_pointer = Engine_Pointer{};
    memory_has_been_allocated = false;
    last_offset = 0;
    max_bytes = 0;
}

Optional<Engine_Function> Engine_Backend::create_function(
    const String &name, const String &source_code
) {
    log::logger().debug("create_function: name={} source_bytes={}", name, source_code.size());

    const String program_name = name + ".cu";

    nvrtcProgram program{};
    nvrtcCreateProgram(&program, source_code.c_str(), program_name.c_str(), 0, nullptr, nullptr);

    // TODO: pass --gpu-architecture (compute_87 for Orin).
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

    Engine_Function function{};

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

Optional<Engine_Function> Engine_Backend::load_precompiled_function(const String &name) {
    // The CUDA backend has no precompiled kernels; callers fall back to their host paths, which is
    // the normal path here, not an error.
    log::logger().debug("load_precompiled_function: '{}' -- the CUDA backend has no "
                        "runtime-loadable precompiled module; the caller answers on the host", name);
    return std::nullopt;
}

void Engine_Backend::release_function(Engine_Function &function) {
    if (function.cuda_module == nullptr) return;
    cuModuleUnload(function.cuda_module);
    function.cuda_module = nullptr;
    function.cuda_function = nullptr;
}

void Engine_Backend::advise_read_mostly(const Engine_Pointer &range, u64 byte_count) {
    if (range.is_empty() || byte_count == 0) return;
    cudaMemAdvise(range.get_contents(), byte_count, cudaMemAdviseSetReadMostly, 0);
}

void Engine_Backend::prefetch_to_gpu(const Engine_Pointer &range, u64 byte_count) {
    if (range.is_empty() || byte_count == 0) return;
    cudaMemPrefetchAsync(range.get_contents(), byte_count, 0);
}

void Engine_Backend::prefetch_to_cpu(const Engine_Pointer &range, u64 byte_count) {
    if (range.is_empty() || byte_count == 0) return;
    cudaMemPrefetchAsync(range.get_contents(), byte_count, cudaCpuDeviceId);
}

bool Engine_Backend::run_function(
    Engine_Function &function, Vector<Engine_Pointer> &parameters, s64 job_count
) {
    const s64 block_count = (job_count + threads_per_block - 1) / threads_per_block;

    // cuLaunchKernel takes a pointer to each argument's value. A buffer's device address needs
    // a stable cell to point at; an inline scalar's own storage is that cell.
    Vector<void *> argument_values(parameters.size(), nullptr);
    Vector<void *> argument_pointers(parameters.size(), nullptr);

    for (usize index = 0; index < parameters.size(); index += 1) {
        const Engine_Pointer &parameter = parameters[index];
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
        static_cast<u32>(block_count), 1, 1,       // grid
        static_cast<u32>(threads_per_block), 1, 1, // block
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

bool Engine_Backend::run_functions(Vector<Engine_Dispatch> &dispatches) {
    for (Engine_Dispatch &dispatch : dispatches) {
        if (dispatch.function == nullptr || !run_function(*dispatch.function, dispatch.parameters, dispatch.job_count)) {
            return false;
        }
    }
    return true;
}

#endif // SPIKECOREC_CUDA

} // namespace spikecorec
