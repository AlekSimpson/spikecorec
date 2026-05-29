# spikecorec

A high-performance C++ library with CUDA and Metal GPU backends, importable as a Python package via pybind11.

## Requirements

| Tool | Purpose |
|---|---|
| C++17 compiler (`clang++` / `g++`) | Core and binding compilation |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) | CUDA backend (`nvcc`) |
| macOS 13+ with Xcode CLI tools | Metal backend |
| Python 3.9+ with `pybind11` | Python bindings |

Install Python dependencies:
```bash
pip install pybind11
```

Add pybind11 as a git submodule (optional, for offline builds):
```bash
git submodule add https://github.com/pybind/pybind11 third_party/pybind11
```

---

## Directory structure

```
spikecorec/
├── include/
│   └── spikecorec/
│       ├── spikecorec.hpp      # Top-level include
│       ├── core/               # Platform-agnostic types & interfaces
│       ├── cuda/               # CUDA headers (.cuh)
│       └── metal/              # Metal headers (.hpp)
├── src/
│   ├── core/                   # Platform-agnostic implementation (.cpp)
│   ├── cuda/                   # CUDA kernels (.cu)
│   ├── metal/
│   │   ├── *.mm                # Metal Obj-C++ implementation
│   │   └── shaders/            # Metal shader source (.metal)
│   └── bindings/               # pybind11 module definition
├── python/
│   └── spikecorec/             # Python package
│       └── __init__.py
├── tests/
│   ├── *.cpp                   # Core C++ tests
│   ├── cuda/                   # CUDA-specific tests
│   └── metal/                  # Metal-specific tests
├── examples/
├── third_party/                # Place pybind11 submodule here
├── Makefile
├── pyproject.toml
└── setup.py
```

---

## Building the C++ library

The Makefile auto-detects the platform (Metal on macOS, CUDA elsewhere). Override with `BACKEND=`.

```bash
make              # auto-detect
make cuda         # force CUDA
make metal        # force Metal
make info         # show detected toolchain
make clean        # remove build/
```

The output is a static library in `build/`:

| Backend | Artifact |
|---|---|
| CUDA | `build/libspikecorec_cuda.a` |
| Metal | `build/libspikecorec_metal.a` + `build/default.metallib` |

---

## Python bindings

The Python extension wraps the C++ library via pybind11.

**Editable install (recommended for development):**
```bash
pip install pybind11
make python                        # auto-detect backend
SPIKECOREC_BACKEND=metal make python
SPIKECOREC_BACKEND=cuda  make python
```

**Or directly with pip:**
```bash
SPIKECOREC_BACKEND=metal pip install -e .
```

**Usage:**
```python
import spikecorec

print(spikecorec.__version__)
```

---

## Tests

```bash
make test           # C++ tests, auto-detect backend
make test-cuda
make test-metal
```

---

## Adding new GPU kernels

1. Declare the interface in `include/spikecorec/{cuda,metal}/`.
2. Implement in `src/{cuda,metal}/`.
3. Expose to Python in `src/bindings/bindings.cpp`.
