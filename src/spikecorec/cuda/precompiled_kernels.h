#pragma once

// The CUDA backend's precompiled kernels: nvcc builds kernels.cu and weight_fit_kernels.cu into
// the library, and Engine_Backend::load_precompiled_function finds a kernel by name here.
//
// They take the arguments their Metal versions take, in the same order. Every argument Metal binds
// to the constant address space (a `constant T &` or `constant T *`, which the host may fill from
// its own bytes or from a buffer) is a pointer here: run_function copies an inline argument's bytes
// into device memory and passes their address, and passes a buffer's address as it is.

namespace spikecorec::cuda {

// The host stub of the kernel `name` in kernels.cu, or nullptr when that file has none by the name.
const void *kernels_symbol(const char *name);

// The same for weight_fit_kernels.cu.
const void *weight_fit_kernels_symbol(const char *name);

} // namespace spikecorec::cuda
