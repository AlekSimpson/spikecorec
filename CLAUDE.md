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

## CRITICAL: U/V factorization is a memory optimization, NOT learning

The `WeightMatrix` stores adjacency weights as a low-rank factorization `W ≈ U * Vᵀ`
(rank controlled by the `rank` constructor parameter). **This is purely a memory
compression scheme.** Large spiking networks would require terabytes of RAM to store
a full dense weight matrix; the U/V factorization combined with the k²-tree adjacency
structure makes massive networks tractable.

**Do not confuse this with task learning or training.** The U/V matrices encode the
connection strengths between neurons in the graph — nothing more. They do not implement
backpropagation, gradient descent, or any task-level learning algorithm. The `rank`
parameter controls compression fidelity, not model capacity in any machine-learning
sense. When you see `gpu_weight_update`, `learning_rate`, or Hebbian update logic in
the engine, that is local synaptic plasticity (spike-timing-dependent updates to
individual edge weights) — it is also not task learning; it is a biological simulation
feature that modifies how compressed weights evolve during a simulation run.

In short: **U/V factorization = memory savings. The engine does not train on tasks.**

## Build

```bash
make              # auto-detect (Metal on macOS, CUDA elsewhere)
make metal
make cuda
make python       # builds pybind11 Python extension
make clean
```

## GitHub

- Repo: `AlekSimpson/spikecorec` (default branch `main`)
- Project board: **spikecorec_project_board** — project number `5`, owner `AlekSimpson`
  (`gh project list --owner AlekSimpson` to re-derive the number if it ever changes)
- Branches follow `SC-<issue-number>_<short-description>`, e.g. `SC-25_add_branching_factor_bound_checks`
  is the working branch for issue #25 — the numeric prefix is the GitHub issue number, so
  `gh issue view <N>` finds the corresponding issue directly.
- Issue labels: severity (`severity: critical|high|medium|low`) + category
  (`correctness`, `performance`, `enhancement`, plus the GitHub defaults like `bug`/`documentation`).
- `gh pr create` does **not** automatically attach the PR to the project board, even with
  `Closes #N` in the body (that only closes the issue on merge). After creating a PR, attach it explicitly:
  ```bash
  gh project item-add 5 --owner AlekSimpson --url <pr-url>
  ```
