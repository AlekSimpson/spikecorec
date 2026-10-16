# spikecorec

A GPU spiking-neural-network simulation engine driven by NeuroML. It generates one GPU kernel
from a model's LEMS dynamics (Metal on macOS, CUDA elsewhere), stores connectivity and every
per-connection value compressed, and is importable as a Python package via pybind11.

## Requirements

| Tool | Purpose |
|---|---|
| C++17 compiler (`clang++` / `g++`) | Core and binding compilation |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) | CUDA backend (`nvcc`) |
| macOS 13+ with Xcode CLI tools | Metal backend |
| [uv](https://docs.astral.sh/uv/) | Python bindings: `make python` builds into a uv-managed `.venv` |
| `zlib` / `liblzma` / `libbz2` (optional, via `pkg-config`) | gzip/xz/bz2 support for `.spire` recordings — auto-detected at build time (`make info` shows what was found); uncompressed `.spire` always works |
| `bear` (optional) | Regenerate `compile_commands.json` for IDE tooling |

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv, for the Python bindings
brew install bear                                 # optional, for IDE setup
```

Clone with submodules to get metal-cpp:
```bash
git clone --recurse-submodules <repo-url>
# or after cloning:
git submodule update --init
```

---

## Directory structure

```
spikecorec/
├── src/spikecorec/         # headers (.h) live beside their sources; include root is src/
│   ├── spikecorec.h        # top-level include
│   ├── core/               # CPU-side logic: Engine main loop, weight matrix, k²-tree, Metal/CUDA dispatch
│   ├── nml/                # NeuroML parsing, resolution and kernel codegen (Metal and CUDA source)
│   ├── metal/              # metal-cpp implementation file + fixed Metal kernels (.metal, .metalinc)
│   ├── cuda/               # the same fixed kernels for CUDA (.cu, .cuh), built by nvcc
│   └── bindings/           # pybind11 Python bindings
├── python/spikecorec/      # Python package
├── tests/
├── docs/                   # design notes and reviews
├── examples/
├── third_party/metal-cpp/  # metal-cpp submodule
├── Makefile
├── pyproject.toml
└── setup.py
```

### Architecture

- **`src/spikecorec/core/engine.cpp`** is the CPU-side home — the `Spike_Engine` main loop
- **`src/spikecorec/core/backend.cpp`** implements the Metal/CUDA backend dispatch (`Engine_Backend`):
  `create_function` compiles kernel source at runtime, `run_function` launches it
- **`src/spikecorec/nml/`** generates the model's `master_step` kernel, as Metal or CUDA source
- **`src/spikecorec/metal/`** contains `metal_cpp_impl.cpp` (instantiates metal-cpp once), the
  k²-tree device helpers and the fixed kernels (the weight-matrix fits), built into `default.metallib`
- **`src/spikecorec/cuda/`** holds the CUDA versions of the fixed kernels and the k²-tree helpers,
  built by nvcc into the CUDA library
- Metal host code uses **metal-cpp** (pure C++, no Objective-C++)
- Backend selected at compile time via `-DSPIKECOREC_METAL` or `-DSPIKECOREC_CUDA`

---

## Build

The Makefile auto-detects the platform (Metal on macOS, CUDA elsewhere).
See **[`BUILDME.md`](BUILDME.md)** for the full list of build commands
(libraries, tests, examples, Python extension, direct pip/setuptools builds).

```bash
make              # auto-detect backend
make metal        # force Metal
make cuda         # force CUDA
make clean        # remove build/
make info         # show detected toolchain
make compdb       # regenerate compile_commands.json for IDE
```

Output static libraries:

| Backend | Artifact |
|---|---|
| CUDA | `build/libspikecorec_cuda.a` |
| Metal | `build/libspikecorec_metal.a` + `build/default.metallib` |

> **Note:** Metal shader compilation requires the Metal Toolchain component.
> If you see a `missing Metal Toolchain` error, run:
> ```bash
> xcodebuild -downloadComponent MetalToolchain
> ```

---

## Python bindings

```bash
make python                         # auto-detect backend
SPIKECOREC_BACKEND=metal make python
SPIKECOREC_BACKEND=cuda  make python
```

It creates `.venv` with uv on first use, installs `pybind11`, `numpy`, `setuptools` and
`matplotlib` into it, and builds the extension there as an editable install:

```bash
.venv/bin/python -c "import spikecorec; print(spikecorec.__version__)"
```

See **[`python/docs/README.md`](python/docs/README.md)** for the Python guide (`Spike_Engine`,
topology generators, `Weight_Matrix`, the parsed model, recording to `.spire` files) and
**[`python/docs/API_REFERENCE.md`](python/docs/API_REFERENCE.md)** for every name.

---

## Tests

Tests use [GoogleTest](https://github.com/google/googletest) (vendored as a submodule).

```bash
make test          # build + run all C++ tests (auto-detects backend), recurrent networks included
make test-metal    # Metal backend explicitly
make test-cuda     # CUDA backend explicitly
make test-python   # the Python binding tests in tests/python, with pytest in .venv
```

`make test TEST_ARGUMENTS=--gtest_filter='Weight_Matrix.*'` passes arguments through to the runner.
Fixtures live in `tests/fixtures`:

- `reference/` holds jNeuroML's outputs, regenerated with `tests/fixtures/reference/regenerate_references.sh`.
  The engine is compared with them statistically (rate and interval distributions) and by waveform,
  never spike for spike.
- `golden/` holds spikes recorded from the engine itself. The regression tests compare every run's
  spike statistics with them (rates and interspike intervals), never spike for spike, to catch a
  change in what the engine computes. `SPIKECOREC_UPDATE_GOLDEN=1` re-records them.

To run a subset after the initial build, invoke the runner directly with `--gtest_filter`:

```bash
# All tests in a suite
./build/test_runner_metal --gtest_filter='K2_Tree.*'

# A single test
./build/test_runner_metal --gtest_filter='SingleCell.a_pulse_delivers_exactly_its_charge'

# Wildcard across all suites
./build/test_runner_metal --gtest_filter='*.save_load'

# List every test name without running
./build/test_runner_metal --gtest_list_tests
```

See **[`BUILDME.md`](BUILDME.md)** for every build target and variable.

---

## How the GPU code is made

- **The simulation kernel is generated.** At construction, `src/spikecorec/nml/` turns the model's
  LEMS dynamics into one `master_step` kernel (Metal or CUDA source), which
  `Engine_Backend::create_function` compiles; `Engine_Backend::run_function` launches it every
  tick. `Spike_Engine::master_kernel_source` holds it as compiled.
- **The fixed kernels** (the weight-matrix fits, the read-back of a plane) are in
  `src/spikecorec/metal/*.metal`, built into `default.metallib`, and in `src/spikecorec/cuda/*.cu`,
  built by nvcc for `CUDA_ARCH`. `Engine_Backend::load_precompiled_function` finds them by name.
  Each has a host fallback, which runs when its kernel does not load.
- Expose anything new to Python in `src/spikecorec/bindings/bindings.cpp`, under its C++ name.
- Run `make compdb` to keep IDE tooling current.
