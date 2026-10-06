# spikecorec

A high-performance C++ library with CUDA and Metal GPU backends, importable as a Python package via pybind11.

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
├── include/spikecorec/
│   ├── spikecorec.h        # top-level include
│   ├── core/               # platform-agnostic types & interfaces (.h)
│   ├── cuda/               # CUDA headers (.cuh)
│   └── metal/              # Metal headers (.h)
├── src/
│   ├── core/               # CPU-side logic: Engine main loop + Metal/CUDA dispatch
│   ├── cuda/               # CUDA kernels (.cu)
│   ├── metal/              # Metal-cpp host code (.cpp) + GPU shaders (.metal)
│   └── bindings/           # pybind11 Python bindings
├── python/spikecorec/      # Python package
├── tests/
├── examples/
├── third_party/metal-cpp/  # metal-cpp submodule
├── Makefile
├── pyproject.toml
└── setup.py
```

### Architecture

- **`src/core/core.cpp`** is the CPU-side home — `Engine` main loop and Metal/CUDA backend dispatch all live here
- **`src/metal/`** contains `metal_cpp_impl.cpp` (instantiates metal-cpp once) and `.metal` GPU shader files
- **`src/cuda/`** contains CUDA kernel definitions
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

See **[`python/README.md`](python/README.md)** for full Python API documentation
and usage examples (`SpikeEngine`, topology generators, `WeightMatrix`, reservoir
features, weight scaling, simulation recording to `.spire` files, etc.).

---

## Tests

Tests use [GoogleTest](https://github.com/google/googletest) (vendored as a submodule).

```bash
make test          # build + run all C++ tests (auto-detects backend), recurrent networks included
make test-metal    # Metal backend explicitly
make test-cuda     # CUDA backend explicitly
make test-python   # the Python binding tests in tests/python, with pytest in .venv
```

`make test TEST_ARGUMENTS=--gtest_filter='WeightMatrix.*'` passes arguments through to the runner.
Fixtures live in `tests/fixtures`:

- `reference/` holds jNeuroML's outputs, regenerated with `tests/fixtures/reference/regenerate_references.sh`.
  The engine is compared with them statistically (rate and interval distributions) and by waveform,
  never spike for spike.
- `golden/` holds spikes recorded from the engine itself. The regression tests hold every run to
  them exactly, to catch any change in what the engine computes. After an intended change,
  re-record them and review the diff:
  `SPIKECOREC_UPDATE_GOLDEN=1 make test TEST_ARGUMENTS="--gtest_filter=*Regression*"`.

To run a subset after the initial build, invoke the runner directly with `--gtest_filter`:

```bash
# All tests in a suite
./build/test_runner_metal --gtest_filter='K2Tree.*'

# A single test
./build/test_runner_metal --gtest_filter='SingleCell.a_pulse_delivers_exactly_its_charge'

# Wildcard across all suites
./build/test_runner_metal --gtest_filter='*.save_load'

# List every test name without running
./build/test_runner_metal --gtest_list_tests
```

See **[`BUILDME.md`](BUILDME.md)** for the full filter reference and suite index.

---

## Adding GPU kernels

1. Write the GPU kernel in `src/metal/your_kernel.metal` (Metal) or `src/cuda/your_kernel.cu` (CUDA)
2. Declare the CPU-side dispatch interface in `include/spikecorec/core/backend.h`
3. Implement the dispatch in `src/core/core.cpp` (inside the appropriate `#ifdef` block)
4. Call it from `Engine::step()` in `src/core/engine.cpp`
5. Expose to Python in `src/bindings/bindings.cpp` if needed
6. Run `make compdb` to keep IDE tooling current
