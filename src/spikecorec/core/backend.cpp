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

void throw_on_cuda_driver_error(CUresult result, const String &operation) {
    if (result == CUDA_SUCCESS) return;
    log::throw_runtime_error(log::logger(),
            "Cuda_Backend: " + operation + " failed: " + describe_cuda_driver_error(result));
}

// The NVRTC target for the device: its own real architecture when NVRTC can build for it, so the
// module loads with no JIT step; otherwise PTX for the newest virtual architecture NVRTC has below
// it, which the driver compiles at load. Empty when NVRTC has none, which leaves NVRTC's default.
struct Nvrtc_Target {
    String architecture_option;
    bool builds_device_code = false;
};

Nvrtc_Target nvrtc_target_for_device(CUdevice device) {
    int major_version = 0;
    int minor_version = 0;
    cuDeviceGetAttribute(&major_version, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor_version, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    const int device_architecture = major_version * 10 + minor_version;

    int architecture_count = 0;
    if (nvrtcGetNumSupportedArchs(&architecture_count) != NVRTC_SUCCESS || architecture_count <= 0) return {};
    Vector<int> supported_architectures(static_cast<usize>(architecture_count));
    if (nvrtcGetSupportedArchs(supported_architectures.data()) != NVRTC_SUCCESS) return {};

    const auto exact = std::find(supported_architectures.begin(), supported_architectures.end(), device_architecture);
    if (exact != supported_architectures.end()) {
        return {"--gpu-architecture=sm_" + to_string(device_architecture), true};
    }
    int newest_below = 0;
    for (int architecture : supported_architectures) {
        if (architecture <= device_architecture) newest_below = std::max(newest_below, architecture);
    }
    if (newest_below == 0) return {};
    return {"--gpu-architecture=compute_" + to_string(newest_below), false};
}

Engine_Backend::Engine_Backend(s64 threads_per_block_argument) {
    threads_per_block = threads_per_block_argument;

    throw_on_cuda_driver_error(cuInit(0), "cuInit");  // flags must be 0
    throw_on_cuda_driver_error(cuDeviceGet(&cuda_gpu_device, 0), "cuDeviceGet");  // device ordinal 0
    throw_on_cuda_driver_error(cuDevicePrimaryCtxRetain(&cuda_context, cuda_gpu_device), "cuDevicePrimaryCtxRetain");
    throw_on_cuda_driver_error(cuCtxSetCurrent(cuda_context), "cuCtxSetCurrent");

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
    // Driver calls act on the calling thread's current context, and a thread the engine did not
    // start on has none.
    cuCtxSetCurrent(cuda_context);

    const String program_name = name + ".cu";

    nvrtcProgram program{};
    const nvrtcResult create_result =
            nvrtcCreateProgram(&program, source_code.c_str(), program_name.c_str(), 0, nullptr, nullptr);
    if (create_result != NVRTC_SUCCESS) {
        log::logger().critical("create_function: nvrtcCreateProgram failed: {}", nvrtcGetErrorString(create_result));
        return std::nullopt;
    }

    const Nvrtc_Target target = nvrtc_target_for_device(cuda_gpu_device);
    Vector<const char *> compile_options;
    if (!target.architecture_option.empty()) compile_options.push_back(target.architecture_option.c_str());
    const nvrtcResult compile_result =
            nvrtcCompileProgram(program, static_cast<int>(compile_options.size()), compile_options.data());

    // After a successful compile the log holds only warnings, such as a declared but unused variable.
    size_t compile_log_size = 0;
    nvrtcGetProgramLogSize(program, &compile_log_size);
    if (compile_log_size > 1) {
        Vector<char> compile_log(compile_log_size);
        nvrtcGetProgramLog(program, compile_log.data());
        if (compile_result == NVRTC_SUCCESS) {
            log::logger().debug("create_function: nvrtc log: {}", compile_log.data());
        } else {
            log::logger().error("create_function: nvrtc log: {}", compile_log.data());
        }
    }
    if (compile_result != NVRTC_SUCCESS) {
        log::logger().critical("create_function: nvrtcCompileProgram failed: {}",
                               nvrtcGetErrorString(compile_result));
        nvrtcDestroyProgram(&program);
        return std::nullopt;
    }

    // A real architecture gives a CUBIN the device runs as is; a virtual one gives PTX.
    size_t module_image_size = 0;
    if (target.builds_device_code) {
        nvrtcGetCUBINSize(program, &module_image_size);
    } else {
        nvrtcGetPTXSize(program, &module_image_size);
    }
    Vector<char> module_image(module_image_size);
    if (target.builds_device_code) {
        nvrtcGetCUBIN(program, module_image.data());
    } else {
        nvrtcGetPTX(program, module_image.data());
    }
    nvrtcDestroyProgram(&program);

    Engine_Function function{};

    const CUresult module_result = cuModuleLoadData(&function.cuda_module, module_image.data());
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
    cuCtxSetCurrent(cuda_context);
    cuModuleUnload(function.cuda_module);
    function.cuda_module = nullptr;
    function.cuda_function = nullptr;
}

// CUDA 13 names the device of a hint by a cudaMemLocation; CUDA 12 by its ordinal.
#if CUDART_VERSION >= 13000
cudaMemLocation managed_memory_location(int device_ordinal) {
    cudaMemLocation location{};
    if (device_ordinal == cudaCpuDeviceId) {
        location.type = cudaMemLocationTypeHost;
        location.id = 0;
    } else {
        location.type = cudaMemLocationTypeDevice;
        location.id = device_ordinal;
    }
    return location;
}
#endif

void Engine_Backend::advise_read_mostly(const Engine_Pointer &range, u64 byte_count) {
    if (range.is_empty() || byte_count == 0) return;
#if CUDART_VERSION >= 13000
    // The location is ignored for read-mostly.
    cudaMemAdvise(range.get_contents(), byte_count, cudaMemAdviseSetReadMostly, managed_memory_location(0));
#else
    cudaMemAdvise(range.get_contents(), byte_count, cudaMemAdviseSetReadMostly, 0);
#endif
}

void Engine_Backend::prefetch_to_gpu(const Engine_Pointer &range, u64 byte_count) {
    if (range.is_empty() || byte_count == 0) return;
#if CUDART_VERSION >= 13000
    cudaMemPrefetchAsync(range.get_contents(), byte_count, managed_memory_location(0), 0);
#else
    cudaMemPrefetchAsync(range.get_contents(), byte_count, 0);
#endif
}

void Engine_Backend::prefetch_to_cpu(const Engine_Pointer &range, u64 byte_count) {
    if (range.is_empty() || byte_count == 0) return;
#if CUDART_VERSION >= 13000
    cudaMemPrefetchAsync(range.get_contents(), byte_count, managed_memory_location(cudaCpuDeviceId), 0);
#else
    cudaMemPrefetchAsync(range.get_contents(), byte_count, cudaCpuDeviceId);
#endif
}

bool Engine_Backend::run_function(
    Engine_Function &function, Vector<Engine_Pointer> &parameters, s64 job_count
) {
    if (function.cuda_function == nullptr) {
        log::logger().critical("run_function: the function has no CUDA kernel");
        return false;
    }
    // The launch needs this thread's current context to be the engine's.
    cuCtxSetCurrent(cuda_context);

    // At least one thread, as on Metal: a grid of zero blocks is an invalid launch, and the
    // kernels return early for threads past their job count.
    const s64 thread_count = std::max<s64>(job_count, 1);
    const s64 block_count = (thread_count + threads_per_block - 1) / threads_per_block;

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
