//
// Created by Alek Simpson on 5/30/26.
//

#ifdef SPIKECOREC_CUDA
#include <cuda.h>
#include <nvrtc.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include "spikecorec/core/backend.h"

using namespace std;

namespace spikecorec {

// ── global backend state ─────────────────────────────────────────────────────

#ifdef SPIKECOREC_CUDA
static CUdevice  g_device  = 0;
static CUcontext g_context = nullptr;

#elif defined(SPIKECOREC_METAL)
static MTL::Device*       g_device = nullptr;
static MTL::CommandQueue* g_queue  = nullptr;
static unordered_map<void*, MTL::Buffer*> g_buffer_map;
#endif

// ── KernelHandle ─────────────────────────────────────────────────────────────

struct KernelHandle {
#ifdef SPIKECOREC_CUDA
    CUfunction cuda_kernel_function{};
    CUmodule   cuda_module{};
#elif defined(SPIKECOREC_METAL)
    MTL::ComputePipelineState* pipeline_state = nullptr;
#endif
};

// ── lifecycle ─────────────────────────────────────────────────────────────────

void initialize_gpu_context() {
#ifdef SPIKECOREC_CUDA
    cuInit(0);
    cuDeviceGet(&g_device, 0);
    cuCtxCreate(&g_context, 0, g_device);

#elif defined(SPIKECOREC_METAL)
    g_device = MTL::CreateSystemDefaultDevice();
    g_queue  = g_device->newCommandQueue();
#endif
}

void release_gpu_resources() {
#ifdef SPIKECOREC_CUDA
    cuCtxDestroy(g_context);
    g_context = nullptr;

#elif defined(SPIKECOREC_METAL)
    for (auto& [ptr, buffer] : g_buffer_map)
        buffer->release();
    g_buffer_map.clear();

    if (g_queue)  { g_queue->release();  g_queue  = nullptr; }
    if (g_device) { g_device->release(); g_device = nullptr; }
#endif
}

// ── memory ────────────────────────────────────────────────────────────────────

void* allocate_bytes(usize byte_size) {
#ifdef SPIKECOREC_CUDA
    void* pointer = nullptr;
    cudaMallocManaged(&pointer, byte_size);
    return pointer;

#elif defined(SPIKECOREC_METAL)
    return g_device->newBuffer(byte_size, MTL::ResourceStorageModeShared);

#else
    return nullptr;
#endif
}

void deallocate_bytes(void* platform_handle) {
    if (!platform_handle) return;

#ifdef SPIKECOREC_CUDA
    cudaFree(platform_handle);

#elif defined(SPIKECOREC_METAL)
    static_cast<MTL::Buffer*>(platform_handle)->release();
#endif
}

// ── synchronization ───────────────────────────────────────────────────────────

void synchronize_gpu_work() {
#ifdef SPIKECOREC_CUDA
    cudaDeviceSynchronize();
#endif
    // no-op on Metal — dispatch() already blocks via waitUntilCompleted
}

// ── kernel lifecycle ──────────────────────────────────────────────────────────

KernelHandle compile_kernel(const char* source, const char* function_name) {
    KernelHandle handle;

#ifdef SPIKECOREC_CUDA
    nvrtcProgram program;
    nvrtcCreateProgram(&program, source, nullptr, 0, nullptr, nullptr);
    nvrtcCompileProgram(program, 0, nullptr);

    size_t ptx_size;
    nvrtcGetPTXSize(program, &ptx_size);
    char* ptx = new char[ptx_size];
    nvrtcGetPTX(program, ptx);
    nvrtcDestroyProgram(&program);

    cuModuleLoadData(&handle.cuda_module, ptx);
    delete[] ptx;
    cuModuleGetFunction(&handle.cuda_kernel_function, handle.cuda_module, function_name);

#elif defined(SPIKECOREC_METAL)
    NS::Error* error = nullptr;
    NS::String* source_string   = NS::String::string(source, NS::UTF8StringEncoding);
    MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
    MTL::Library* library = g_device->newLibrary(source_string, options, &error);
    options->release();

    NS::String* function_name_string = NS::String::string(function_name, NS::UTF8StringEncoding);
    MTL::Function* function = library->newFunction(function_name_string);
    library->release();

    handle.pipeline_state = g_device->newComputePipelineState(function, &error);
    function->release();
#endif

    return handle;
}

void release_kernel(KernelHandle handle) {
#ifdef SPIKECOREC_CUDA
    cuModuleUnload(handle.cuda_module);

#elif defined(SPIKECOREC_METAL)
    if (handle.pipeline_state)
        handle.pipeline_state->release();
#endif
}

// ── dispatch ──────────────────────────────────────────────────────────────────

void dispatch(
    KernelHandle handle,
    LaunchConfig config,
    const void *const *args,
    const usize *arg_sizes,
    u32 arg_count
) {
#ifdef SPIKECOREC_CUDA
    cuLaunchKernel(
        handle.cuda_kernel_function,
        config.grid_size,  1, 1,
        config.block_size, 1, 1,
        0,
        nullptr,
        const_cast<void **>(args),
        nullptr
    );

#elif defined(SPIKECOREC_METAL)
    MTL::CommandBuffer *command_buffer = g_queue->commandBuffer();
    MTL::ComputeCommandEncoder *encoder = command_buffer->computeCommandEncoder();

    encoder->setComputePipelineState(handle.pipeline_state);

    for (u32 i = 0; i < arg_count; ++i) {
        auto it = g_buffer_map.find(const_cast<void*>(args[i]));
        if (it != g_buffer_map.end()) {
            // pointer is a unified memory allocation — bind its MTLBuffer
            encoder->setBuffer(it->second, 0, i);
        } else {
            // scalar argument — pass by value
            encoder->setBytes(args[i], arg_sizes[i], i);
        }
    }

    MTL::Size threadgroups_per_grid  = MTL::Size::Make(config.grid_size,  1, 1);
    MTL::Size threads_per_threadgroup = MTL::Size::Make(config.block_size, 1, 1);
    encoder->dispatchThreadgroups(threadgroups_per_grid, threads_per_threadgroup);

    encoder->endEncoding();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    command_buffer->release();
    encoder->release();
#endif
}

} // namespace spikecorec
