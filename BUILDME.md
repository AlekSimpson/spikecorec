# Building spikecorec

All build entry points for spikecorec: the Makefile (C++ libs, tests, examples)
and the Python extension (setuptools + pybind11). Run from the repo root.

## Prerequisites

    pip install pybind11
    brew install bear        # optional, for compile_commands.json
    git submodule update --init   # metal-cpp (Metal builds) + spdlog (logging) + googletest (tests)

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

### Python tests

- `make test-python`
  Builds the extension (`make python`) then runs the pytest suite under `python/tests/`
  (`pip install -e .[dev]` first, for `pytest`/`numpy`). Includes `test_glif_network.py`
  (ticket #133) — an end-to-end run of the NeuroML → GPU codegen path via
  `spikecorec.NmlNetworkRunner`, never touching `SpikeEngine`.

- `pytest python/tests` (or `python -m pytest python/tests`)
  Same thing directly, once the extension is already built.

### Tests

Tests use [GoogleTest](https://github.com/google/googletest), compiled from the
vendored submodule at `third_party/googletest/`. Each `make test-*` target
builds a single test runner binary and immediately executes it. The recipe fails
(non-zero exit) if any test fails.

- `make test`
  Builds and runs all tests for the auto-detected backend (`test-metal` on
  macOS, `test-cuda` elsewhere).

- `make test-metal`
  Builds `build/test_runner_metal` from all `tests/*.cpp` files and runs it.

- `make test-cuda`
  Builds `build/test_runner_cuda` from all `tests/*.cpp` files and runs it.

**Running a subset of tests** — invoke the binary directly after the first build
and use `--gtest_filter=SUITE.TEST` (shell-glob patterns, `:` separates multiple):

    # All tests in a suite
    ./build/test_runner_metal --gtest_filter='K2Tree.*'
    ./build/test_runner_metal --gtest_filter='WeightMatrix.*'
    ./build/test_runner_metal --gtest_filter='SpikeEngine.*'

    # One specific test
    ./build/test_runner_metal --gtest_filter='SpikeEngine.spike_fanout'

    # Multiple suites
    ./build/test_runner_metal --gtest_filter='K2Tree.*:WeightMatrix.*'

    # Wildcard on test name across all suites
    ./build/test_runner_metal --gtest_filter='*.save_load'

    # Negate — run everything except one suite
    ./build/test_runner_metal --gtest_filter='-AsyncSpireWriter.*'

    # List all test names without running them
    ./build/test_runner_metal --gtest_list_tests

### Sanitizers (AddressSanitizer + UndefinedBehaviorSanitizer)

- `make test-asan`
  Builds and runs the same GoogleTest suite as `make test`, but compiled with
  `-fsanitize=address,undefined` (host/CPU C++ code only — GPU-side Metal/CUDA
  device kernel code isn't sanitizable this way, so it's out of scope for this
  target). Objects land under `build-asan/` (a separate directory from `build/`,
  via `SANITIZE=1 BUILD_DIR=build-asan`), so a sanitized build never collides with
  a normal one — no `make clean` needed between the two.

  Sanitized runs are substantially slower than a normal `make test` (ASan/UBSan
  instrumentation overhead, plus `-O1` instead of `-O2`) — budget several minutes
  for the full suite, most of it in the two largest fixtures in
  `tests/end_to_end_network_tests.cpp` (`EndToEndGlifNetworks.
  glif5_large_network_current_injection_largest_anchor`, 2000 neurons / 10000
  ticks, and `EndToEndGlifNetworks.
  isolated_pair_subgraph_behaves_identically_regardless_of_surrounding_network_size`,
  a 2000-neuron population run twice). Both already report their own wall-clock
  time to stdout.

  Equivalent to running `make SANITIZE=1 BUILD_DIR=build-asan test-metal` (or
  `test-cuda`) directly, if you want the sanitizer flags without the separate
  build directory.

**Test suite index:**

| Suite | File | What it covers |
|---|---|---|
| `Backend` | `engine_tests.cpp` | Type sizes, `GpuPointer` alloc/move |
| `SpikeEngine` | `engine_tests.cpp`, `recording_tests.cpp` | Construction, step loop, reset, reservoir features, input guards, bifurcation scaling, decay path, spike propagation, plasticity, recording |
| `K2Tree` | `k2tree_tests.cpp` | Adjacency queries, batch ops, bounds, save/load, invalid branching factor |
| `SpireCodec` | `recording_tests.cpp` | Raw roundtrip, compression codecs, zero-frame file, header overflow, truncation error |
| `AsyncSpireWriter` | `recording_tests.cpp` | Backpressure, unbounded queue, no re-entry after error |
| `SimulationRecorder` | `recording_tests.cpp` | Frame-size validation |
| `Topologies` | `topology_tests.cpp` | `square_torus`, `small_world_torus`, `random_fixed_outdegree` — basic and edge cases |
| `WeightMatrix` | `weight_matrix_tests.cpp` | Construction, constant/random weights, stats, scaling, neighbor queries, bounds, update, save/load |

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
  Remove `build/`. (`make clean BUILD_DIR=build-asan` removes the separate
  sanitized build directory from `make test-asan`.)

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
