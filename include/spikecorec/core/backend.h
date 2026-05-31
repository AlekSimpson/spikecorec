#pragma once

#include <cassert>
#include "spikecorec/core/types.h"

namespace spikecorec {
    // Called once at program startup to initialize the GPU context.
    void initialize_gpu_context();

    // Called once at program shutdown to release GPU resources.
    void release_gpu_resources();

    template<typename T>
    struct GpuPointer {
    #ifdef SPIKECOREC_CUDA
        T *pointer = nullptr;

        T*       get_contents()       { return pointer; }
        const T* get_contents() const { return pointer; }
    #elif defined(SPIKECOREC_METAL)
        MTL::Buffer *buffer = nullptr;

        T*       get_contents()       { return static_cast<T*>(buffer->contents()); }
        const T* get_contents() const { return static_cast<const T*>(buffer->contents()); }
    #endif

        GpuPointer() = default;

        GpuPointer(const GpuPointer &) = delete;

        GpuPointer &operator=(const GpuPointer &) = delete;

        GpuPointer(GpuPointer &&other) noexcept {
        #ifdef SPIKECOREC_CUDA
            pointer = other.pointer;
            other.pointer = nullptr;
        #elif defined(SPIKECOREC_METAL)
            buffer = other.buffer;
            other.buffer = nullptr;
        #endif
        }

        GpuPointer &operator=(GpuPointer &&other) noexcept {
            if (this != &other) {
            #ifdef SPIKECOREC_CUDA
                assert(pointer == nullptr &&
                       "GpuPointer move assignment would leak an existing CUDA allocation — "
                       "call backend::deallocate() before reassigning");
                pointer = other.pointer;
                other.pointer = nullptr;
            #elif defined(SPIKECOREC_METAL)
                assert(buffer == nullptr &&
                    "GpuPointer move assignment would leak an existing MTL::Buffer — "
                    "call backend::deallocate() before reassigning");
                buffer = other.buffer;
                other.buffer = nullptr;
            #endif
            }
            return *this;
        }
    };

    // --- buffer ---
    // non-template bridges — defined in backend.cpp, hide platform types from callers
    void* allocate_bytes(usize byte_size);
    void  deallocate_bytes(void* platform_handle);

    template<typename T>
    GpuPointer<T> allocate(usize byte_size) {
        GpuPointer<T> ptr;
    #ifdef SPIKECOREC_CUDA
        ptr.pointer = static_cast<T*>(allocate_bytes(byte_size));
    #elif defined(SPIKECOREC_METAL)
        ptr.buffer  = static_cast<MTL::Buffer*>(allocate_bytes(byte_size));
    #endif
        return ptr;
    }

    template<typename T>
    void deallocate(GpuPointer<T> ptr) {
    #ifdef SPIKECOREC_CUDA
        deallocate_bytes(ptr.pointer);
    #elif defined(SPIKECOREC_METAL)
        deallocate_bytes(ptr.buffer);
    #endif
    }

    // CUDA hint, no-op on Metal
    template<typename T>
    void prefetch_to_gpu(const GpuPointer<T>& ptr, usize size_in_bytes) {
    #ifdef SPIKECOREC_CUDA
        cudaMemPrefetchAsync(ptr.pointer, size_in_bytes, 0);
    #endif
    }

    // --- synchronization ---
    // wait for all in-flight GPU work to complete
    void synchronize_gpu_work();

    // --- kernel lifecycle ---
    // opaque — MTLComputePipelineState* on Metal,
    // CUfunction on CUDA
    struct KernelHandle;

    KernelHandle compile_kernel(
        const char *source, const char *function_name);

    void release_kernel(KernelHandle handle);


    // --- dispatch ---
    struct LaunchConfig {
        u32 grid_size;
        u32 block_size;
    };

    // generic dispatch:
    // - takes raw pointers, backend resolves to MTLBuffers
    //   internally on Metal
    void dispatch(
        KernelHandle handle,
        LaunchConfig config,
        const void *const *args,
        const usize *arg_sizes,
        u32 arg_count
    );

    // --- atomic ops (for backends that need explicit support) ---
    //void atomic_add_f32(f32 *address, f32 value);
} // namespace spikecorec::backend
