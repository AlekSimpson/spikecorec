# Building spikecorec

All build entry points for spikecorec: the Makefile (C++ libs, tests, examples)
and the Python extension (setuptools + pybind11). Run from the repo root.

## Prerequisites

    pip install pybind11
    brew install bear        # optional, for compile_commands.json
    git submodule update --init   # metal-cpp, required for Metal builds

`zlib` / `liblzma` / `libbz2` are optional and auto-detected via `pkg-config`
at build time (enables compressed `.spire` recordings). Uncompressed `.spire`
always works without them.

---

## Makefile targets

The Makefile auto-detects backend by platform: `metal` on macOS (Darwin),
`cuda` everywhere else. Override with `BACKEND=cuda|metal make <target>`.

### Libraries

- `make`
  Default backend for your platform (`metal` on macOS, `cuda` elsewhere).

- `make metal`
  Metal backend: builds `build/libspikecorec_metal.a` + `build/default.metallib`.
  Requires macOS.

- `make cuda`
  CUDA backend: builds `build/libspikecorec_cuda.a`.
  Requires `nvcc` on `PATH` or `CUDA_PATH` set.

### Python extension

- `make python`
  Editable pip install of `spikecorec._spikecorec`, backend auto-detected.

- `SPIKECOREC_BACKEND=metal make python`
  Force Metal backend for the Python extension.

- `SPIKECOREC_BACKEND=cuda make python`
  Force CUDA backend for the Python extension.

- `pip install -e .`
  Same as `make python`, called directly (setup.py auto-detects backend).

- `SPIKECOREC_BACKEND=metal pip install -e .`
  Direct pip equivalent, Metal forced.

- `SPIKECOREC_BACKEND=cuda pip install -e .`
  Direct pip equivalent, CUDA forced.

- `SPIKECOREC_CUDA_ARCH=sm_87 SPIKECOREC_BACKEND=cuda pip install -e .`
  Direct pip build, pin CUDA arch (passed to `nvcc -arch=`).

### Tests

Each backend builds two separate binaries (`test_core.cpp` and the backend
smoke test each define their own `main()`, so they can't be linked together).
Both `test-metal` and `test-cuda` build their two binaries AND execute them as
part of the same `make` invocation — you don't run anything separately, and
the recipe fails (non-zero exit) if either binary aborts or asserts.

- `make test`
  Builds and runs the tests for the auto-detected backend (equivalent to
  `make test-metal` on macOS, `make test-cuda` elsewhere). No separate run
  step needed.

- `make test-metal`
  Builds `build/test_runner_metal` (from `tests/test_core.cpp`, the main
  unit-test suite) and `build/test_smoke_metal` (from `tests/metal/*.cpp`,
  a one-off check that a Metal device is available), then immediately runs
  both binaries in sequence. Output/pass-fail prints straight to the
  terminal — there's no separate test report file.

- `make test-cuda`
  Same idea for CUDA: builds `build/test_runner_cuda` and
  `build/test_smoke_cuda` (from `tests/cuda/*.cpp`), then runs both.

If you want to re-run the tests without recompiling, invoke the produced
binaries directly:

    build/test_runner_metal
    build/test_smoke_metal
    build/test_runner_cuda
    build/test_smoke_cuda

### Examples

- `make examples`
  Examples for the auto-detected backend.

- `make examples-metal`
  Builds `build/metal_example` from `examples/metal_example.cpp`.

- `make examples-cuda`
  Builds `build/cuda_example` from `examples/cuda_example.cpp`.

### Utilities

- `make info`
  Print detected platform, toolchain, and optional-library status.

- `make compdb`
  Regenerate `compile_commands.json` via `bear` (needs `bear` installed).

- `make clean`
  Remove `build/`.

### Overriding toolchain variables

Any of these can be set on the command line, e.g. `CUDA_PATH=/opt/cuda make cuda`:

- `CXX` — default `clang++` (Darwin) / `g++` (else) — host C++ compiler
- `NVCC` — default `nvcc` — CUDA compiler
- `CUDA_PATH` — default `/usr/local/cuda` — CUDA toolkit root
- `PYTHON` — default `python3` — Python interpreter used for `make python`
- `BACKEND` — default `metal` (Darwin) / `cuda` (else) — force backend for
  `make`, `make test`, `make examples`, `make python`

---

## Direct pip / setuptools builds

Bypassing the Makefile entirely (`setup.py` does its own backend detection
and `pkg-config` probing, mirroring the Makefile):

    pip install -e .                              # auto-detect backend
    pip install -e .[dev]                         # + pytest, numpy for python/tests
    python setup.py build_ext --inplace           # build extension in place, no install

---

## Known gaps (current tree)

- `make examples-metal` / `make examples-cuda` reference `examples/metal_example.cpp`
  and `examples/cuda_example.cpp`, which don't exist yet in `examples/` (only
  `demo_script.py` is present) — these targets currently fail at compile time.
