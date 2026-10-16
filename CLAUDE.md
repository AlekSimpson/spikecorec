# spikecorec — project conventions

spikecorec is a GPU spiking-neural-network **simulation engine** (Metal on macOS, CUDA elsewhere).
It simulates NeuroML/LEMS-defined point-neuron networks: codegen turns a model's cell, input and
synapse dynamics into one GPU kernel, and the connectivity and every per-connection value are
stored compressed (k²-tree + low-rank basis) — see "NeuroML → GPU codegen" below.

## Naming

- All source and header filenames must be **lowercase snake_case**
  - `engine.cpp`, `kernels.cuh`, `metal_cpp_impl.cpp` ✓
  - `Engine.cpp`, `MyKernels.cu` ✗
- Headers use `.h` (not `.hpp`), except CUDA headers which use `.cuh`
- C++ classes and types use **Camel_Case** — capitalized words joined by underscores (`Spike_Engine`, `NML_Context`, `Engine_Backend`, `Weight_Matrix`)
- Functions and variables use **snake_case** (`step_simulation`, `run_function`)
- Namespaces use **lowercase** (`spikecorec`, `spikecorec::metal`, `spikecorec::cuda`)

## Project structure

```
src/spikecorec/       — include root is `src/`; headers (.h, .cuh) live beside their sources
src/spikecorec/core/      — CPU-side logic: main loop, Engine, weight matrix, k²-tree, backend dispatch
src/spikecorec/nml/       — NeuroML parsing, resolution and kernel codegen (Metal and CUDA source)
src/spikecorec/metal/     — metal-cpp impl file, k²-tree device helpers (.metalinc), fixed kernels (.metal)
src/spikecorec/cuda/      — the same fixed kernels and k²-tree device helpers for CUDA (.cu, .cuh), built by nvcc
src/spikecorec/bindings/  — pybind11 Python bindings
python/spikecorec/    — Python package
tests/                — C++ tests
examples/             — example programs
third_party/          — vendored dependencies (metal-cpp submodule)
docs/                 — design notes and reviews
```

## Architecture

- `src/spikecorec/core/engine.cpp` is the **CPU-side home** — the `Spike_Engine` main loop
- `src/spikecorec/core/backend.cpp` implements the Metal/CUDA backend dispatch (`Engine_Backend`)
- `src/spikecorec/core/backend.h` defines the CPU↔GPU dispatch interface
- `src/spikecorec/nml/` generates the simulation kernel (`master_step`) as Metal or CUDA source;
  `src/spikecorec/metal/` holds the metal-cpp one-time impl file and the fixed Metal kernels (the
  weight-matrix fits), and `src/spikecorec/cuda/` their CUDA ports, precompiled by nvcc at build time.
  Each fixed kernel has a host fallback, which runs when its GPU kernel can't be loaded
- Metal host code uses **metal-cpp** (pure C++, no `.mm` / Objective-C++)
- Backend is selected at compile time via `-DSPIKECOREC_METAL` or `-DSPIKECOREC_CUDA`

## Engine execution model (load-bearing facts)

These are the fixed engine facts any codegen/simulation work must respect:

- **Clock-driven tick loop, 9 stages.** Every tick advances all active state by one `dt`:
  1 Deliver · 2 Integrate · 3 Detect · 4 Emit · 5 Reset · 6 Propagate · 7 Plasticity · 8 Record ·
  9 Advance. Stages 1/6/8/9 are engine-owned; 2–5 are the per-cell/synapse dynamics. LEMS events
  (`OnCondition`/`OnEvent`/`EventOut`) are **per-tick conditionals**, not an async scheduler.
- **Runtime kernel compilation already exists on both backends** via `Engine_Backend::create_function(name, source_code)`
  in `src/spikecorec/core/backend.cpp` (Metal `newLibrary`; CUDA NVRTC → PTX → `cuModuleGetFunction`). Codegen
  output is fundamentally "a device-code string" the engine compiles and runs.
- **Generic kernel launcher.** `Engine_Backend::run_function(function, parameters, job_count)` in
  `src/spikecorec/core/backend.cpp` launches any compiled function with positional arguments on both
  backends (Metal compute encoder; CUDA `cuLaunchKernel`). The CUDA path has not been run on
  hardware yet.
- **Every cell integrates exactly one `dt` per tick** (forward Euler). Synapses run only in their
  source's active window (`synapse_active_ticks` after its spike: the longest delay plus the
  slowest synapse's settle time) and are set exactly back to rest when it ends.
- **`network_inputs`** is the per-neuron synaptic input accumulator: a source scatters its weight
  into `network_inputs[target]`; the target drains it at its next step (an implicit ≥1-tick latency).
  Inputs (`explicitInput`/`inputList`) run in the kernel, in their target cell's thread, before the
  cell's own dynamics.

## CRITICAL: U/V factorization is a memory optimization, NOT learning

The `Weight_Matrix` stores adjacency weights as a low-rank factorization `W ≈ U * Vᵀ`
(its rank, the number of lanes, is chosen per plane by the fits within `fit_tolerance`). **This is
purely a memory compression scheme.** Large spiking networks would require terabytes of RAM to store
a full dense weight matrix; the U/V factorization combined with the k²-tree adjacency
structure makes massive networks tractable.

**Do not confuse this with task learning or training.** The U/V matrices encode the
connection strengths between neurons in the graph — nothing more. They do not implement
backpropagation, gradient descent, or any task-level learning algorithm. The `rank`
parameter controls compression fidelity, not model capacity in any machine-learning
sense. When you see `plasticity_learning_rate`, `learning_rate`, or Hebbian update logic in
the engine, that is local synaptic plasticity (spike-timing-dependent updates to
individual edge weights) — it is also not task learning; it is a biological simulation
feature that modifies how compressed weights evolve during a simulation run.

The codegen **generalizes** this same compression to per-edge synapse state (a shared U/V
basis + per-plane coefficient vectors `Ck` + a sparse update buffer S + periodic refit). That
generalization is **still memory compression, not learning** — the same invariant holds.

In short: **U/V factorization = memory savings. The engine does not train on tasks.**

## NeuroML → GPU codegen

The engine simulates LEMS-defined dynamics through a codegen path; nothing is hand-written per
model.

**Codegen pipeline.** `LEMS + included NML + NeuroML2CoreTypes → parse → XSD validate (the main
file) → resolve (units→SI, paths, extends) → master_step kernel → compile`. The kernel is built as
a **kernel AST**: the fixed kernel text (`kernel_boilerplate.cpp`, one version for Metal and one
for CUDA) is parsed into the AST, codegen translates each ComponentType's LEMS dynamics into AST
nodes and inserts them (cells per population, their inputs, synapses in the connection walk),
constants fixed for the run are baked in, and the AST is emitted as MSL or CUDA source. It is
compiled once at construction with `Engine_Backend::create_function`.

Cell and synapse state is never declared in the kernel: it lives in engine buffers the engine
allocates at init (cell state per neuron; per-edge synapse state as planes of the weight matrix),
which the kernel indexes directly.

**Three phases** (each additive):
- **Phase 1** — GLIF cells (GLIF1–GLIF5 in full) + the full synapse model (current-based,
  conductance-based, and per-edge), single cell through networks.
- **Phase 2** — nonlinear point cells (izhikevich/AdEx/…), spike delays, on-device generators,
  within-population heterogeneity.
- **Phase 3** — biophysical / multicompartment (HH channels, concentrations, kinetic schemes,
  morphology) — the full NML surface.

## Build

```bash
make              # auto-detect (Metal on macOS, CUDA elsewhere)
make metal
make cuda
make python       # builds the Python extension into .venv (uv); run with .venv/bin/python
make test         # the C++ suite (GoogleTest); TEST_ARGUMENTS=--gtest_filter=... selects tests
make test-python  # the Python binding tests (pytest in .venv)
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
- **Sub-issues** are set via the REST API (`gh` has no direct command). To make issue `C` a child of
  issue `P`: `gh api --method POST repos/AlekSimpson/spikecorec/issues/<P>/sub_issues -F sub_issue_id=<C-database-id>`
  where the database id comes from `gh api repos/AlekSimpson/spikecorec/issues/<C> --jq .id`. Read a
  parent's children via `.../issues/<P>/sub_issues` (the REST issue object has no `.parent` field).
- Heads up: the auto-mode Bash gate blocks **bulk/loop** `gh issue create`; create/edit issues with
  explicit per-issue commands, not shell loops.
