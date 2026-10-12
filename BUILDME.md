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
    ./build/test_runner_metal --gtest_filter='K2_Tree.*'
    ./build/test_runner_metal --gtest_filter='Weight_Matrix.*'
    ./build/test_runner_metal --gtest_filter='Spike_Engine.*'

    # One specific test
    ./build/test_runner_metal --gtest_filter='Spike_Engine.spike_fanout'

    # Multiple suites
    ./build/test_runner_metal --gtest_filter='K2_Tree.*:Weight_Matrix.*'

    # Wildcard on test name across all suites
    ./build/test_runner_metal --gtest_filter='*.save_load'

    # Negate — run everything except one suite
    ./build/test_runner_metal --gtest_filter='-Async_Spire_Writer.*'

    # List all test names without running them
    ./build/test_runner_metal --gtest_list_tests

**Test suite index:**

| Suite | File | What it covers |
|---|---|---|
| `Expression`, `Random_Generator`, `Randomness` | `engine_tests.cpp` | LEMS expression evaluation, random distributions, seeded reproducibility |
| `GeneratedKernel`, `CellTypes`, `CellInputs` | `engine_tests.cpp` | Generated kernel functions, per-type state and equations, refused models |
| `SingleCell`, `CellEvents`, `Connections`, `Inputs`, `Topology` | `engine_tests.cpp` | Single-cell trajectories, `OnEvent` arrivals, delays and per-edge synapse state, every input category, topology-driven connectivity |
| `GlifFamily`, `GlifNetworks`, `Glif_Network_Aliveness` | `engine_tests.cpp` | GLIF type behaviour and network activity |
| `ExpCurrSynapse`, `AlphaCurrSynapse`, `Reference` | `synapse_tests.cpp` | Current-based synapse time courses; statistical comparison with jNeuroML |
| `Regression` | `regression_tests.cpp` | Spike times against the recorded golden runs |
| `Quantity`, `Resolution`, `Instances`, `Include`, `Dynamics`, `Structure`, `Simulation`, `Input_Tree`, `Errors`, `Schema` | `nml_tests.cpp` | NeuroML/LEMS parsing, resolution, includes, inputs and error reporting |
| `Quantity`, `TickConversion`, `TickToTime`, `TimeConversion` | `units_tests.cpp` | Unit parsing and time/tick conversion |
| `K2_Tree` | `k2tree_tests.cpp` | Adjacency queries, batch ops, bounds, save/load, invalid branching factor |
| `Weight_Matrix` | `weight_matrix_tests.cpp` | Construction, constant/random weights, stats, scaling, neighbor queries, bounds, update, save/load |
| `Topologies` | `topology_tests.cpp` | `square_torus`, `small_world_torus`, `random_fixed_outdegree` — basic and edge cases |
| `SpireCodec` | `recording_tests.cpp` | Raw roundtrip, compression codecs, zero-frame file, header overflow, truncation error |
| `Async_Spire_Writer` | `recording_tests.cpp` | Backpressure, unbounded queue, no re-entry after error |
| `Simulation_Recorder` | `recording_tests.cpp` | Frame-size validation |
| `Engine_Logger`, `Log_Format`, `Log_Level`, `Rotating_File_Sink` | `log_tests.cpp` | Message formatting, level filtering, file rotation |

### Examples

- `make examples`
  Examples for the auto-detected backend.

- `make examples-metal`
  Builds every `examples/*.cpp` into `build/examples/`.

- `make examples-cuda`
  Not supported yet: the kernel codegen emits Metal only, so the target stops with a message.

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

- `examples/` currently has no `.cpp` programs (it holds `models/`, `notebooks/` and
  `video_utils.py`), so `make examples-metal` builds nothing; `make examples-cuda` is not
  supported yet.
