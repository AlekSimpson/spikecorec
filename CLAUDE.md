# spikecorec — project conventions

## Naming

- All source and header filenames must be **lowercase snake_case**
  - `engine.cpp`, `kernels.cuh`, `metal_cpp_impl.cpp` ✓
  - `Engine.cpp`, `MyKernels.cu` ✗
- Headers use `.h` (not `.hpp`), except CUDA headers which use `.cuh`
- C++ classes and types use **PascalCase** (`Engine`, `Context`, `LaunchConfig`)
- Functions and variables use **snake_case** (`make_context`, `running_`)
- Namespaces use **lowercase** (`spikecorec`, `spikecorec::metal`, `spikecorec::cuda`)

## Project structure

```
include/spikecorec/   — public headers (.hpp, .cuh)
src/core/             — CPU-side logic: main loop, Engine, orchestration
src/cuda/             — CUDA kernels (.cu) + backend interface implementation
src/metal/            — Metal-cpp host code (.cpp) + GPU shader source (.metal)
src/bindings/         — pybind11 Python bindings
python/spikecorec/    — Python package
tests/                — C++ tests
examples/             — example programs
third_party/          — vendored dependencies (metal-cpp submodule)
```

## Architecture

- `src/core/core.cpp` is the **CPU-side home** — `Engine` main loop + Metal/CUDA backend dispatch implementation all live here
- `include/spikecorec/core/backend.h` defines the CPU↔GPU dispatch interface
- `src/cuda/` and `src/metal/` contain GPU kernel code and the metal-cpp one-time impl file
- Metal host code uses **metal-cpp** (pure C++, no `.mm` / Objective-C++)
- Backend is selected at compile time via `-DSPIKECOREC_METAL` or `-DSPIKECOREC_CUDA`

## Build

```bash
make              # auto-detect (Metal on macOS, CUDA elsewhere)
make metal
make cuda
make python       # builds pybind11 Python extension
make clean
```
