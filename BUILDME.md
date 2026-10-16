# Building spikecorec

Run everything from the repo root. The backend is Metal on macOS and CUDA elsewhere.

## Prerequisites

- `git submodule update --init`: metal-cpp (Metal builds) and googletest (tests).
- **libxml2**, required; found through `pkg-config`.
- **uv**, for the Python extension.
- **Metal:** macOS with the Metal Toolchain component.
- **CUDA:** the CUDA toolkit (`nvcc`, NVRTC).
- **Optional:** zlib, liblzma and libbz2, for compressed `.spire` recordings. They are detected
  through `pkg-config`.

## Targets

| Command | What it does |
|---|---|
| `make` | Library for the platform's backend |
| `make metal` | `build/libspikecorec_metal.a` and `build/default.metallib` |
| `make cuda` | `build/libspikecorec_cuda.a` |
| `make cli` | Builds `main.cpp` into `build/cli` against the backend's library |
| `make metal-debug`, `make cuda-debug`, `make cli-debug` | The same with debug info, into `build/debug/` |
| `make python` | See [Python extension](#python-extension) |
| `make test` | Builds and runs the C++ suite (`build/test_runner_<backend>`) |
| `make test-python` | Runs `tests/python` with pytest in `.venv` |
| `make examples` | Builds `examples/*.cpp` into `build/examples/` (Metal only; there are none at present) |
| `make demos` | Runs `examples/demo_*.py` (there are none at present) |
| `make info` | Prints the platform, toolchain and optional libraries found |
| `make compdb` | Regenerates `compile_commands.json` with `bear` |
| `make clean` | Removes `build/` |

## Debug builds

`DEBUG=1`, or a `*-debug` target, builds with `-g -O0` into `build/debug/`, so debug and
optimised objects never mix. The Metal shaders get source line tables, and nvcc gets `-G`. Any
target works with it, for example `make test DEBUG=1`. Debug the program with
`lldb build/debug/cli`.

The kernel the engine generates at runtime is compiled without debug info either way.

## Python extension

`make python`:
1. creates `.venv` with uv (Python 3.13);
2. installs pybind11, numpy, setuptools and matplotlib into it;
3. installs the extension in editable mode;
4. copies `default.metallib` into the package.

Run Python with `.venv/bin/python`. Use `make python` rather than `pip install -e .`, which skips
the metallib copy.

## Selecting tests

```bash
make test TEST_ARGUMENTS='--gtest_filter=Weight_Matrix.*'
make test TEST_ARGUMENTS='--gtest_filter=K2_Tree.*:Weight_Matrix.*'
make test-python TEST_ARGUMENTS='-k refit'
./build/test_runner_metal --gtest_list_tests
```

## Variables

Set on the command line, for example `make test BACKEND=metal`:

| Variable | Default |
|---|---|
| `BACKEND` | `metal` on macOS, `cuda` elsewhere |
| `CUDA_PATH` | `/usr/local/cuda` |
| `DEBUG` | `0`; `1` for a debug build |
| `CLI_SOURCE` | `main.cpp` |
| `CXX` | `clang++` on macOS, `g++` elsewhere |
| `NVCC` | `nvcc` |
| `UV` | `uv` |
| `PYTHON_VERSION` | `3.13` |
| `TEST_ARGUMENTS` | empty |
