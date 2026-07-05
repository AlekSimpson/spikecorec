//
// Created by Alek Simpson on 5/30/26.
//

#ifdef SPIKECOREC_CUDA
#include <cuda.h>
#include <nvrtc.h>
#include "spikecorec/cuda/kernels.cuh"
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#include <dlfcn.h>
#include <stdexcept>
#endif

#include "spikecorec/core/backend.h"
#include "spikecorec/core/log.h"

using namespace std;

namespace spikecorec {

// ── global backend state ─────────────────────────────────────────────────────

#ifdef SPIKECOREC_CUDA
static CUdevice  global_device  = 0;
static CUcontext global_context = nullptr;

#elif defined(SPIKECOREC_METAL)
static MTL::Device       *global_device          = nullptr;
static MTL::CommandQueue *global_queue           = nullptr;
static MTL::Library      *global_default_library = nullptr;
static unordered_map<void *, MTL::Buffer *> global_buffer_map;

// The shaders in src/metal/kernels.metal are compiled ahead-of-time by the
// Makefile into default.metallib, placed alongside the build artifacts. Command
// line tools have no app bundle for newDefaultLibrary() to search, so locate the
// metallib next to the loaded binary (via dladdr) and load it explicitly.
static MTL::Library *load_default_metal_library(MTL::Device *device) {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void *>(&load_default_metal_library), &info) && info.dli_fname) {
        string binary_path(info.dli_fname);
        size_t last_slash = binary_path.find_last_of('/');
        string directory = (last_slash == string::npos) ? "." : binary_path.substr(0, last_slash);
        string metallib_path = directory + "/default.metallib";

        NS::String *path_string = NS::String::string(metallib_path.c_str(), NS::UTF8StringEncoding);
        NS::URL *url = NS::URL::fileURLWithPath(path_string);
        NS::Error *error = nullptr;
        if (MTL::Library *library = device->newLibrary(url, &error))
            return library;
    }
    return device->newDefaultLibrary();
}
#endif

// ── KernelHandle ─────────────────────────────────────────────────────────────
// Defined for both backends (must be a complete type wherever compile_kernel /
// dispatch / release_kernel are compiled). The Metal-only helper below stays
// guarded since it touches MTL types.

struct KernelHandle {
#ifdef SPIKECOREC_CUDA
    CUfunction cuda_kernel_function{};
    CUmodule   cuda_module{};
#elif defined(SPIKECOREC_METAL)
    MTL::ComputePipelineState *pipeline_state = nullptr;
#endif
};

#ifdef SPIKECOREC_METAL
// Looks up a kernel function compiled ahead-of-time into default.metallib and
// builds a pipeline state for it
static KernelHandle load_precompiled_kernel(const char *function_name) {
    log::logger().debug("load_precompiled_kernel: function_name={}", function_name);
    KernelHandle handle;
    NS::String *function_name_string = NS::String::string(function_name, NS::UTF8StringEncoding);
    MTL::Function *function = global_default_library->newFunction(function_name_string);
    if (!function) {
        log::logger().critical("load_precompiled_kernel: function not found in library: {}", function_name);
        throw runtime_error("Metal: function not found in library: " + string(function_name));
    }

    NS::Error *error = nullptr;
    handle.pipeline_state = global_device->newComputePipelineState(function, &error);
    function->release();
    if (!handle.pipeline_state) {
        string description = error ? error->localizedDescription()->utf8String() : "unknown error";
        log::logger().critical("load_precompiled_kernel: newComputePipelineState failed: {}", description);
        throw runtime_error("Metal: newComputePipelineState failed: " + description);
    }
    return handle;
}
#endif

// ── lifecycle ─────────────────────────────────────────────────────────────────

void initialize_gpu_context() {
#ifdef SPIKECOREC_CUDA
    log::logger().debug("initialize_gpu_context: backend=CUDA");
    cuInit(0);
    cuDeviceGet(&global_device, 0);
    cuCtxCreate(&global_context, 0, global_device);

#elif defined(SPIKECOREC_METAL)
    log::logger().debug("initialize_gpu_context: backend=Metal");
    global_device          = MTL::CreateSystemDefaultDevice();
    global_queue           = global_device->newCommandQueue();
    global_default_library = load_default_metal_library(global_device);
#endif
    log::logger().debug("initialize_gpu_context: done");
}

void release_gpu_resources() {
#ifdef SPIKECOREC_CUDA
    log::logger().debug("release_gpu_resources: backend=CUDA");
    cuCtxDestroy(global_context);
    global_context = nullptr;

#elif defined(SPIKECOREC_METAL)
    log::logger().debug("release_gpu_resources: releasing buffer_count={}", global_buffer_map.size());
    for (auto& [pointer, buffer] : global_buffer_map) {
        buffer->release();
    }
    global_buffer_map.clear();

    if (global_default_library) {
        global_default_library->release();
        global_default_library = nullptr;
    }
    if (global_queue)  {
        global_queue->release();
        global_queue  = nullptr;
    }
    if (global_device) {
        global_device->release();
        global_device = nullptr;
    }
#endif
}

// ── memory ────────────────────────────────────────────────────────────────────

void* allocate_bytes(usize byte_size) {
    log::logger().trace("allocate_bytes: byte_size={}", byte_size);
#ifdef SPIKECOREC_CUDA
    void *pointer = nullptr;
    cudaMallocManaged(&pointer, byte_size);
    return pointer;

#elif defined(SPIKECOREC_METAL)
    MTL::Buffer *buffer = global_device->newBuffer(byte_size, MTL::ResourceStorageModeShared);
    // index by the unified-memory data pointer — that's what callers pass around
    // (GpuPointer::get_contents()), and what dispatch() must resolve back to a buffer
    global_buffer_map[buffer->contents()] = buffer;
    return buffer;

#else
    return nullptr;
#endif
}

void deallocate_bytes(void *platform_handle) {
    if (!platform_handle) return;
    log::logger().trace("deallocate_bytes");

#ifdef SPIKECOREC_CUDA
    cudaFree(platform_handle);

#elif defined(SPIKECOREC_METAL)
    auto *buffer = static_cast<MTL::Buffer *>(platform_handle);
    global_buffer_map.erase(buffer->contents());
    buffer->release();
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

KernelHandle compile_kernel(const char *source, const char *function_name) {
    log::logger().debug("compile_kernel: function_name={}", function_name);
    KernelHandle handle;

#ifdef SPIKECOREC_CUDA
    nvrtcProgram program;
    nvrtcCreateProgram(&program, source, nullptr, 0, nullptr, nullptr);
    nvrtcCompileProgram(program, 0, nullptr);

    size_t parallel_thread_execution_size;
    nvrtcGetPTXSize(program, &parallel_thread_execution_size);
    char *parallel_thread_execution= new char[parallel_thread_execution_size];
    nvrtcGetPTX(program, parallel_thread_execution);
    nvrtcDestroyProgram(&program);

    cuModuleLoadData(&handle.cuda_module, parallel_thread_execution);
    delete[] parallel_thread_execution;
    cuModuleGetFunction(&handle.cuda_kernel_function, handle.cuda_module, function_name);

#elif defined(SPIKECOREC_METAL)
    NS::Error *error = nullptr;
    NS::String *source_string   = NS::String::string(source, NS::UTF8StringEncoding);
    MTL::CompileOptions *options = MTL::CompileOptions::alloc()->init();
    MTL::Library *library = global_device->newLibrary(source_string, options, &error);
    options->release();
    if (!library) {
        string description = error ? error->localizedDescription()->utf8String() : "unknown error";
        log::logger().critical("compile_kernel: newLibrary failed: {}", description);
        throw runtime_error("Metal: newLibrary failed: " + description);
    }

    NS::String *function_name_string = NS::String::string(function_name, NS::UTF8StringEncoding);
    MTL::Function *function = library->newFunction(function_name_string);
    library->release();
    if (!function) {
        log::logger().critical("compile_kernel: function not found in library: {}", function_name);
        throw runtime_error("Metal: function not found in library: " + string(function_name));
    }

    error = nullptr;
    handle.pipeline_state = global_device->newComputePipelineState(function, &error);
    function->release();
    if (!handle.pipeline_state) {
        string description = error ? error->localizedDescription()->utf8String() : "unknown error";
        log::logger().critical("compile_kernel: newComputePipelineState failed: {}", description);
        throw runtime_error("Metal: newComputePipelineState failed: " + description);
    }
#endif

    return handle;
}

void release_kernel(KernelHandle handle) {
    log::logger().debug("release_kernel");
#ifdef SPIKECOREC_CUDA
    cuModuleUnload(handle.cuda_module);

#elif defined(SPIKECOREC_METAL)
    if (handle.pipeline_state)
        handle.pipeline_state->release();
#endif
}

// ── command batching ─────────────────────────────────────────────────────────

struct MetalCommandBatch {
#ifdef SPIKECOREC_METAL
    MTL::CommandBuffer *command_buffer = nullptr;
#endif
};

MetalCommandBatch *begin_command_batch() {
#ifdef SPIKECOREC_CUDA
    return nullptr;

#elif defined(SPIKECOREC_METAL)
    auto *batch = new MetalCommandBatch();
    batch->command_buffer = global_queue->commandBuffer();
    return batch;
#else
    return nullptr;
#endif
}

void commit_command_batch(MetalCommandBatch *batch) {
#ifdef SPIKECOREC_CUDA
    (void)batch;

#elif defined(SPIKECOREC_METAL)
    if (!batch) return;

    log::logger().trace("commit_command_batch");
    batch->command_buffer->commit();
    batch->command_buffer->waitUntilCompleted();
    if (batch->command_buffer->status() == MTL::CommandBufferStatusError) {
        NS::Error *error = batch->command_buffer->error();
        string description = error ? error->localizedDescription()->utf8String() : "unknown error";
        log::logger().critical("commit_command_batch: command buffer failed: {}", description);
        batch->command_buffer->release();
        delete batch;
        throw runtime_error("Metal: command buffer failed: " + description);
    }
    batch->command_buffer->release();
    delete batch;
#endif
}

// ── metal_dispatch ──────────────────────────────────────────────────────────────────

void metal_dispatch(
    KernelHandle handle,
    LaunchConfig config,
    const void *const *args,
    const usize *arg_sizes,
    u32 arg_count,
    MetalCommandBatch *batch
) {
#ifdef SPIKECOREC_METAL
    log::logger().trace("metal dispatch: grid_size={} block_size={} arg_count={} batched={}",
                        config.grid_size, config.block_size, arg_count, batch != nullptr);

    MTL::CommandBuffer *command_buffer = batch ? batch->command_buffer : global_queue->commandBuffer();
    MTL::ComputeCommandEncoder *encoder = command_buffer->computeCommandEncoder();

    encoder->setComputePipelineState(handle.pipeline_state);

    for (u32 i = 0; i < arg_count; ++i) {
        // convention: args[i] always points to the argument's storage (matches
        // CUDA's cuLaunchKernel kernelParams). For pointer-typed arguments that
        // storage holds a data pointer — try to resolve it back to its MTLBuffer.
        // Only attempt this when the slot is pointer-sized, so scalar args (which
        // may be smaller than sizeof(void*)) are never over-read.
        void *candidate = nullptr;
        if (arg_sizes[i] == sizeof(void *)) {
            candidate = *reinterpret_cast<void * const *>(args[i]);
        }
        auto it = candidate ? global_buffer_map.find(candidate) : global_buffer_map.end();
        if (it != global_buffer_map.end()) {
            // resolved to a unified memory allocation — bind its MTLBuffer
            encoder->setBuffer(it->second, 0, i);
        } else {
            // scalar argument — pass by value
            encoder->setBytes(args[i], arg_sizes[i], i);
        }
    }

    MTL::Size threadgroups_per_grid = MTL::Size::Make(config.grid_size,  1, 1);
    MTL::Size threads_per_threadgroup = MTL::Size::Make(config.block_size, 1, 1);
    encoder->dispatchThreadgroups(threadgroups_per_grid, threads_per_threadgroup);

    encoder->endEncoding();
    encoder->release();

    if (!batch) {
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
        if (command_buffer->status() == MTL::CommandBufferStatusError) {
            NS::Error *error = command_buffer->error();
            string description = error ? error->localizedDescription()->utf8String() : "unknown error";
            log::logger().critical("metal_dispatch: command buffer failed: {}", description);
            command_buffer->release();
            throw runtime_error("Metal: command buffer failed: " + description);
        }
        command_buffer->release();
    }
#endif
}

// ── kernel wrappers: WeightMatrix + K2Tree ───────────────────────────────────

void gpu_neighbor_weights(
    const float4 *U,
    const float4 *V,
    const u32    *internal_node_words,
    const u32    *leaf_node_words,
    const u32    *rank_superblock_table,
    const u16    *rank_subblock_table,
    s32 branching_factor,
    s32 superblock_size_words,
    s32 padded_node_count,
    s32 tree_height,
    s32 internal_bit_count,
    s64 node_count,
    s64 max_neighbor_count,
    s64 rank_float4_stride,
    f32 *output_weights
) {
    s64 total_pairs = node_count * max_neighbor_count;
    if (total_pairs <= 0) return;
    log::logger().trace("gpu_neighbor_weights: node_count={} max_neighbor_count={} total_pairs={}",
                        node_count, max_neighbor_count, total_pairs);

#ifdef SPIKECOREC_CUDA
    s64 total_pairs = node_count * max_neighbor_count;
    if (total_pairs <= 0) return;
    LaunchConfig config = cuda::default_launch_config(static_cast<usize>(total_pairs));
    cuda::neighbor_weights_kernel<<<config.grid, config.block, 0, stream>>>(
        U, V,
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, padded_node_count, tree_height, internal_bit_count,
        node_count, max_neighbor_count, rank_float4_stride, output_weights
    );
    synchronize_gpu_work(); // concurrentManagedAccess=0: finish before host touches managed memory

#elif defined(SPIKECOREC_METAL)
    static KernelHandle kernel_handle = load_precompiled_kernel("neighbor_weights_kernel");

    constexpr u32 threads_per_block = 256;
    LaunchConfig config{
        static_cast<u32>((total_pairs + threads_per_block - 1) / threads_per_block),
        threads_per_block
    };
    const void *args[] = {
        &U, &V,
        &internal_node_words, &leaf_node_words, &rank_superblock_table, &rank_subblock_table,
        &branching_factor, &superblock_size_words, &padded_node_count, &tree_height, &internal_bit_count,
        &node_count, &max_neighbor_count, &rank_float4_stride,
        &output_weights
    };
    const usize arg_sizes[] = {
        sizeof(void*), sizeof(void*),
        sizeof(void*), sizeof(void*), sizeof(void*), sizeof(void*),
        sizeof(s32), sizeof(s32), sizeof(s32), sizeof(s32), sizeof(s32),
        sizeof(s64), sizeof(s64), sizeof(s64),
        sizeof(void*)
    };
    metal_dispatch(kernel_handle, config, args, arg_sizes, 15);

#endif
}

void gpu_scale_uv(
    float4 *U,
    float4 *V,
    s64 total_float4_element_count,
    f32 scale_factor
) {
    if (total_float4_element_count <= 0) return;
    log::logger().trace("gpu_scale_uv: total_float4_element_count={} scale_factor={}",
                        total_float4_element_count, scale_factor);

#ifdef SPIKECOREC_CUDA
    if (total_float4_element_count <= 0) return;

    LaunchConfig config = cuda::default_launch_config(static_cast<usize>(total_float4_element_count));
    cuda::scale_uv_kernel<<<config.grid, config.block, 0, stream>>>(
        U, V, total_float4_element_count, scale_factor
    );

    synchronize_gpu_work(); // concurrentManagedAccess=0: finish before host touches managed memory

#elif defined(SPIKECOREC_METAL)
    static KernelHandle kernel_handle = load_precompiled_kernel("scale_uv_kernel");

    constexpr u32 threads_per_block = 256;
    LaunchConfig config{
        static_cast<u32>((total_float4_element_count + threads_per_block - 1) / threads_per_block),
        threads_per_block
    };
    const void *args[] = { &U, &V, &total_float4_element_count, &scale_factor };
    const usize arg_sizes[] = { sizeof(void*), sizeof(void*), sizeof(s64), sizeof(f32) };
    metal_dispatch(kernel_handle, config, args, arg_sizes, 4);
#endif
}

void gpu_add_network_input(f32 *membrane_potentials, s32 *input_neuron_indices, const f32 *input_values, s64 element_count, MetalCommandBatch *batch) {
    if (element_count <= 0) return;

    log::logger().trace("gpu_add_network_input: element_count={}", element_count);

#ifdef SPIKECOREC_CUDA
    (void)batch;
    if (element_count <= 0) return;
    
    LaunchConfig config = cuda::default_launch_config(static_cast<usize>(element_count));
    cuda::add_network_input_kernel<<<config.grid, config.block, 0, stream>>>(
        membrane_potentials, input_neuron_indices, input_values, element_count
    );

    synchronize_gpu_work(); // concurrentManagedAccess=0: finish before host touches managed memory

#elif defined(SPIKECOREC_METAL)
    static KernelHandle kernel_handle = load_precompiled_kernel("add_network_input_kernel");

    constexpr u32 threads_per_block = 256;
    LaunchConfig config{
        static_cast<u32>((element_count + threads_per_block - 1) / threads_per_block),
        threads_per_block
    };
    const void *args[] = { &membrane_potentials, &input_neuron_indices, &input_values, &element_count };
    const usize arg_sizes[] = { sizeof(void*), sizeof(void*), sizeof(void*), sizeof(s64) };
    metal_dispatch(kernel_handle, config, args, arg_sizes, 4, batch);

#endif
}

void gpu_decay_all_neurons(
    f32 *membrane_potentials,
    s64 *last_tick_updated,
    s64  neuron_count,
    s64  tick,
    f32  resting_mp,
    f32  decay_rate,
    MetalCommandBatch *batch
) {
    if (neuron_count <= 0) return;
    log::logger().trace("gpu_decay_all_neurons: neuron_count={} tick={}", neuron_count, tick);

#ifdef SPIKECOREC_CUDA
    (void)batch;
    if (neuron_count <= 0) return;

    LaunchConfig config = cuda::default_launch_config(static_cast<usize>(neuron_count));
    cuda::decay_all_neurons_kernel<<<config.grid, config.block, 0, stream>>>(
        membrane_potentials, last_tick_updated, neuron_count, tick, resting_mp, decay_rate
    );

    synchronize_gpu_work(); // concurrentManagedAccess=0: finish before host touches managed memory

#elif defined(SPIKECOREC_METAL)
    static KernelHandle kernel_handle = load_precompiled_kernel("decay_all_neurons_kernel");

    constexpr u32 threads_per_block = 256;
    LaunchConfig config{
        static_cast<u32>((neuron_count + threads_per_block - 1) / threads_per_block),
        threads_per_block
    };
    const void *args[] = { &membrane_potentials, &last_tick_updated, &neuron_count, &tick, &resting_mp, &decay_rate };
    const usize arg_sizes[] = { sizeof(void*), sizeof(void*), sizeof(s64), sizeof(s64), sizeof(f32), sizeof(f32) };
    metal_dispatch(kernel_handle, config, args, arg_sizes, 6, batch);
#endif
}

void gpu_merge_input_neurons(
    s32       *active_neuron_indices,
    s32       *active_neuron_count,
    const s64 *override_input_neurons,
    s64        override_count,
    MetalCommandBatch *batch
) {
    if (override_count <= 0) return;
    log::logger().trace("gpu_merge_input_neurons: override_count={}", override_count);

#ifdef SPIKECOREC_CUDA
    (void)batch;
    if (override_count <= 0) return;

    LaunchConfig config = cuda::default_launch_config(static_cast<usize>(override_count));
    cuda::merge_input_neurons_kernel<<<config.grid, config.block, 0, stream>>>(
        active_neuron_indices, active_neuron_count, override_input_neurons, override_count
    );

    synchronize_gpu_work(); // concurrentManagedAccess=0: finish before host touches managed memory

#elif defined(SPIKECOREC_METAL)
    static KernelHandle kernel_handle = load_precompiled_kernel("merge_input_neurons_kernel");

    constexpr u32 threads_per_block = 256;
    LaunchConfig config{
        static_cast<u32>((override_count + threads_per_block - 1) / threads_per_block),
        threads_per_block
    };
    const void *args[] = { &active_neuron_indices, &active_neuron_count, &override_input_neurons, &override_count };
    const usize arg_sizes[] = { sizeof(void*), sizeof(void*), sizeof(void*), sizeof(s64) };
    metal_dispatch(kernel_handle, config, args, arg_sizes, 4, batch);
#endif
}

void gpu_reservoir_features(
    s64       neuron_count,
    s64       tick,
    f32       spike_tau,
    f32       voltage_scale,
    f32      *membrane_potentials,
    const s64*last_spiked,
    s64      *last_tick_updated,
    f32       resting_mp,
    f32       decay_rate,
    f32      *output_buffer
) {
    if (neuron_count <= 0) return;
    log::logger().trace("gpu_reservoir_features: neuron_count={} tick={} spike_tau={} voltage_scale={}",
                        neuron_count, tick, spike_tau, voltage_scale);

#ifdef SPIKECOREC_CUDA
    if (neuron_count <= 0) return;

    LaunchConfig config = cuda::default_launch_config(static_cast<usize>(neuron_count));
    cuda::reservoir_features_kernel<<<config.grid, config.block, 0, stream>>>(
        neuron_count, tick, spike_tau, voltage_scale, membrane_potentials,
        last_spiked, last_tick_updated, resting_mp, decay_rate, output_buffer
    );

    synchronize_gpu_work(); // concurrentManagedAccess=0: finish before host touches managed memory

#elif defined(SPIKECOREC_METAL)
    static KernelHandle kernel_handle = load_precompiled_kernel("reservoir_features_kernel");

    constexpr u32 threads_per_block = 256;
    LaunchConfig config{
        static_cast<u32>((neuron_count + threads_per_block - 1) / threads_per_block),
        threads_per_block
    };
    const void *args[] = {
        &neuron_count, &tick, &spike_tau, &voltage_scale,
        &membrane_potentials, &last_spiked, &last_tick_updated,
        &resting_mp, &decay_rate, &output_buffer
    };
    const usize arg_sizes[] = {
        sizeof(s64), sizeof(s64), sizeof(f32), sizeof(f32),
        sizeof(void*), sizeof(void*), sizeof(void*),
        sizeof(f32), sizeof(f32), sizeof(void*)
    };
    metal_dispatch(kernel_handle, config, args, arg_sizes, 10);
#endif
}

void gpu_weight_update(
    float4 *U,
    float4 *V,
    s64 rank_float4_stride,
    s32 source_node,
    s32 target_node,
    f32 delta,
    f32 learning_rate,
    f32 l2_regularization,
    s32 iterations
) {
    if (rank_float4_stride <= 0 || iterations <= 0) return;
    log::logger().trace("gpu_weight_update: source_node={} target_node={} delta={} iterations={}",
                        source_node, target_node, delta, iterations);

#ifdef SPIKECOREC_CUDA
    if (rank_float4_stride <= 0 || iterations <= 0) return;

    unsigned threads = 32u;
    while (static_cast<s64>(threads) < rank_float4_stride) {
        threads <<= 1;
    }
    threads = threads > 1024u ? 1024u : threads;

    usize shared_bytes = static_cast<usize>(2 * rank_float4_stride) * sizeof(float4);
    cuda::weight_update_kernel<<<1, threads, shared_bytes, stream>>>(
        U, V, rank_float4_stride, source_node, target_node,
        delta, learning_rate, l2_regularization, iterations
    );

    synchronize_gpu_work(); // concurrentManagedAccess=0: finish before host touches managed memory

#elif defined(SPIKECOREC_METAL)
    static KernelHandle kernel_handle = load_precompiled_kernel("weight_update_kernel");

    // single threadgroup; rank_float4_stride is always small (e.g. 16 lanes for rank=64),
    // round up to a multiple of the SIMD width so the reduction covers every active lane
    u32 threads = 32;
    while (static_cast<s64>(threads) < rank_float4_stride) threads <<= 1;
    threads = threads > 1024 ? 1024 : threads;

    LaunchConfig config{1u, threads};
    const void *args[] = {
        &U, &V, &rank_float4_stride, &source_node, &target_node,
        &delta, &learning_rate, &l2_regularization, &iterations
    };
    const usize arg_sizes[] = {
        sizeof(void*), sizeof(void*), sizeof(s64), sizeof(s32), sizeof(s32),
        sizeof(f32), sizeof(f32), sizeof(f32), sizeof(s32)
    };
    metal_dispatch(kernel_handle, config, args, arg_sizes, 9);
#endif
}

void gpu_k2tree_adjacent_batch(
    const u32 *internal_node_words,
    const u32 *leaf_node_words,
    const u32 *rank_superblock_table,
    const u16 *rank_subblock_table,
    s32 branching_factor,
    s32 superblock_size_words,
    s32 node_count,
    s32 padded_node_count,
    s32 tree_height,
    s32 internal_bit_count,
    const s32 *source_indices,
    const s32 *target_indices,
    uint8_t *output_buffer,
    s32 query_count
) {
    if (query_count <= 0) return;
    log::logger().trace("gpu_k2tree_adjacent_batch: query_count={}", query_count);

#ifdef SPIKECOREC_CUDA
    if (query_count <= 0) return;

    LaunchConfig config = cuda::default_launch_config(static_cast<usize>(query_count));
    cuda::k2tree_adjacent_batch_kernel<<<config.grid, config.block, 0, stream>>>(
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, node_count, padded_node_count,
        tree_height, internal_bit_count, source_indices, target_indices,
        output_buffer, query_count
    );

    synchronize_gpu_work(); // concurrentManagedAccess=0: finish before host touches managed memory

#elif defined(SPIKECOREC_METAL)
    static KernelHandle kernel_handle = load_precompiled_kernel("k2tree_adjacent_batch_kernel");

    constexpr u32 threads_per_block = 256;
    LaunchConfig config{
        (static_cast<u32>(query_count) + threads_per_block - 1) / threads_per_block,
        threads_per_block
    };
    const void *args[] = {
        &internal_node_words, &leaf_node_words, &rank_superblock_table, &rank_subblock_table,
        &branching_factor, &superblock_size_words, &node_count, &padded_node_count,
        &tree_height, &internal_bit_count,
        &source_indices, &target_indices, &output_buffer, &query_count
    };
    const usize arg_sizes[] = {
        sizeof(void*), sizeof(void*), sizeof(void*), sizeof(void*),
        sizeof(s32), sizeof(s32), sizeof(s32), sizeof(s32),
        sizeof(s32), sizeof(s32),
        sizeof(void*), sizeof(void*), sizeof(void*), sizeof(s32)
    };
    metal_dispatch(kernel_handle, config, args, arg_sizes, 14);
#endif
}

void gpu_k2tree_get_neighbors_batch(
    const u32 *internal_node_words,
    const u32 *leaf_node_words,
    const u32 *rank_superblock_table,
    const u16 *rank_subblock_table,
    s32 branching_factor,
    s32 superblock_size_words,
    s32 node_count,
    s32 padded_node_count,
    s32 tree_height,
    s32 internal_bit_count,
    const s32 *source_node_indices,
    s32 query_count,
    s32 max_neighbor_count,
    s32 *output_buffer
) {
    s64 total_pairs = static_cast<s64>(query_count) * max_neighbor_count;
    if (total_pairs <= 0) return;
    log::logger().trace("gpu_k2tree_get_neighbors_batch: query_count={} max_neighbor_count={} total_pairs={}",
                        query_count, max_neighbor_count, total_pairs);

#ifdef SPIKECOREC_CUDA
    s64 total_pairs = static_cast<s64>(query_count) * max_neighbor_count;
    if (total_pairs <= 0) return;

    LaunchConfig config = cuda::default_launch_config(static_cast<usize>(total_pairs));
    cuda::k2tree_get_neighbors_batch_kernel<<<config.grid, config.block, 0, stream>>>(
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, node_count, padded_node_count,
        tree_height, internal_bit_count, source_node_indices, query_count,
        max_neighbor_count, output_buffer
    );

    synchronize_gpu_work(); // concurrentManagedAccess=0: finish before host touches managed memory

#elif defined(SPIKECOREC_METAL)
    static KernelHandle kernel_handle = load_precompiled_kernel("k2tree_get_neighbors_batch_kernel");

    constexpr u32 threads_per_block = 256;
    LaunchConfig config{
        (static_cast<u32>(total_pairs) + threads_per_block - 1) / threads_per_block,
        threads_per_block
    };
    const void *args[] = {
        &internal_node_words, &leaf_node_words, &rank_superblock_table, &rank_subblock_table,
        &branching_factor, &superblock_size_words, &node_count, &padded_node_count,
        &tree_height, &internal_bit_count,
        &source_node_indices, &query_count, &max_neighbor_count, &output_buffer
    };
    const usize arg_sizes[] = {
        sizeof(void*), sizeof(void*), sizeof(void*), sizeof(void*),
        sizeof(s32), sizeof(s32), sizeof(s32), sizeof(s32),
        sizeof(s32), sizeof(s32),
        sizeof(void*), sizeof(s32), sizeof(s32), sizeof(void*)
    };
    metal_dispatch(kernel_handle, config, args, arg_sizes, 14);
#endif
}

void gpu_step(
   s64           tick,
   s64           next_tick,
   s32           spike_period,
   f32           spike_threshold,
   f32           learning_rate,
   f32           decay_rate,
   f32           resting_mp,
   const float4 *U,
   const float4 *V,
   s64           rank_float4_stride,
   f32           constant_weight,
   const u32    *internal_node_words,
   const u32    *leaf_node_words,
   const u32    *rank_superblock_table,
   const u16    *rank_subblock_table,
   s32           branching_factor,
   s32           superblock_size_words,
   s32           padded_node_count,
   s32           tree_height,
   s32           internal_bit_count,
   s64           neuron_count,
   f32          *network_inputs,
   f32          *membrane_potentials,
   s64          *last_spiked,
   s64          *last_tick_updated,
   const s32    *active_neuron_indices,
   const s32    *active_neuron_count,
   s32          *next_active_neuron_indices,
   s32          *next_active_neuron_count,
   s32          *active_generation,
   s32           thread_count_per_block,
   s32           block_count,
   MetalCommandBatch *batch
) {
    if (tick < 0 || next_tick < 0) return;
    log::logger().trace("gpu_step: tick={} next_tick={} neuron_count={} thread_count_per_block={} block_count={}",
                        tick, next_tick, neuron_count, thread_count_per_block, block_count);

#ifdef SPIKECOREC_CUDA
    (void)batch;
    if (block_count <= 0 || thread_count_per_block <= 0) return;

    cuda::step_kernel<<<static_cast<unsigned>(block_count), static_cast<unsigned>(thread_count_per_block), 0, stream>>>(
        tick, next_tick, spike_period, spike_threshold, learning_rate, decay_rate, resting_mp,
        U, V, rank_float4_stride, constant_weight,
        internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,
        branching_factor, superblock_size_words, padded_node_count, tree_height, internal_bit_count,
        neuron_count, network_inputs, membrane_potentials, last_spiked, last_tick_updated,
        active_neuron_indices, active_neuron_count, next_active_neuron_indices, next_active_neuron_count,
        active_generation
    );

    synchronize_gpu_work(); // concurrentManagedAccess=0: finish before host touches managed memory

#elif defined(SPIKECOREC_METAL)
    static KernelHandle kernel_handle = load_precompiled_kernel("step");
    constexpr u32 threads_per_block = 256;
    LaunchConfig config{
        (static_cast<u32>(neuron_count) + threads_per_block - 1) / threads_per_block,
        threads_per_block
    };

    const void *args[] = {
       &tick, &next_tick, &spike_period, &spike_threshold,
       &learning_rate, &decay_rate, &resting_mp, &U, &V,
       &rank_float4_stride, &constant_weight, &internal_node_words, &leaf_node_words,
       &rank_superblock_table, &rank_subblock_table, &branching_factor,
       &superblock_size_words, &padded_node_count, &tree_height,
       &internal_bit_count, &neuron_count, &network_inputs, &membrane_potentials,
       &last_spiked, &last_tick_updated, &active_neuron_indices, &active_neuron_count,
       &next_active_neuron_indices, &next_active_neuron_count,
       &active_generation, &thread_count_per_block, &block_count
    };
    const usize arg_sizes[] = {
        sizeof(s64), sizeof(s64), sizeof(s32), sizeof(f32), sizeof(f32), sizeof(f32), sizeof(f32),
        sizeof(float4 *), sizeof(float4 *), sizeof(s64), sizeof(f32), sizeof(const u32 *), sizeof(const u32 *), sizeof(const u32 *),
        sizeof(const u16 *), sizeof(s32), sizeof(s32), sizeof(s32), sizeof(s32), sizeof(s32), sizeof(s64),
        sizeof(f32 *), sizeof(f32 *), sizeof(s64 *), sizeof(s64 *), sizeof(const s32 *), sizeof(const s32 *),
        sizeof(s32 *), sizeof(s32 *), sizeof(s32 *),
        sizeof(s32), sizeof(s32)
    };

    metal_dispatch(kernel_handle, config, args, arg_sizes, 31, batch);

#endif
}

} // namespace spikecorec
