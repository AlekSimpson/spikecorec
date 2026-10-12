# spikecorec — project conventions

spikecorec is a GPU spiking-neural-network **simulation engine** (Metal on macOS, CUDA elsewhere).
Today it runs a single hardcoded leaky-integrate-and-fire (LIF) cell over a compressed adjacency
graph. The **current major effort is a NeuroML → GPU codegen path** so the engine can
simulate arbitrary LEMS-defined cell/synapse dynamics — see "NeuroML → GPU codegen" below.

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
src/spikecorec/core/      — CPU-side logic: main loop, Engine, orchestration
src/spikecorec/nml/       — NeuroML parsing, resolution and kernel codegen
src/spikecorec/cuda/      — CUDA kernels (.cu) + backend interface implementation
src/spikecorec/metal/     — Metal-cpp host code (.cpp) + GPU shader source (.metal)
src/spikecorec/bindings/  — pybind11 Python bindings
python/spikecorec/    — Python package
tests/                — C++ tests
examples/             — example programs
third_party/          — vendored dependencies (metal-cpp submodule)
docs/                 — design docs (the NML codegen architecture + IR spec live here)
```

## Architecture

- `src/spikecorec/core/engine.cpp` is the **CPU-side home** — the `Spike_Engine` main loop
- `src/spikecorec/core/backend.cpp` implements the Metal/CUDA backend dispatch (`Engine_Backend`)
- `src/spikecorec/core/backend.h` defines the CPU↔GPU dispatch interface
- `src/spikecorec/cuda/` and `src/spikecorec/metal/` contain GPU kernel code and the metal-cpp one-time impl file
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
- **Active-set lazy decay is only valid for LINEAR dynamics.** The active-set optimization advances
  a skipped neuron across many ticks in one closed-form `apply_decay`. That is valid only for
  analytically-integrable (linear) cells (all of GLIF). Nonlinear cells (izhikevich/AdEx/HH) must
  integrate exactly one `dt` per active tick and re-enqueue while non-resting.
- **`network_inputs`** is the per-neuron synaptic input accumulator: a source scatters its weight
  into `network_inputs[target]`; the target drains it at its next step (an implicit ≥1-tick latency).
  External stimulus is added straight into `membrane_potentials`.

## CRITICAL: U/V factorization is a memory optimization, NOT learning

The `Weight_Matrix` stores adjacency weights as a low-rank factorization `W ≈ U * Vᵀ`
(rank controlled by the `rank` constructor parameter). **This is purely a memory
compression scheme.** Large spiking networks would require terabytes of RAM to store
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
basis + per-matrix coefficient vectors `Ck` + sparse delta buffers `Sk` + periodic refit). That generalization is **still memory compression, not learning** — the same
invariant holds.

In short: **U/V factorization = memory savings. The engine does not train on tasks.**

## NeuroML → GPU codegen

Add a NeuroML (NML/LEMS) → GPU kernel codegen path so the engine can simulate arbitrary
LEMS-defined dynamics instead of the single hardcoded LIF cell.

**Codegen pipeline.** `NML + LEMS + NeuroML2CoreTypes → parse → XSD validate → resolve (units→SI,
IDref wiring, extends/Fixed flatten) → lower to a flat ModelSpecification → per-ComponentType IR →
runnable master kernel`. Codegen is **generic**: it emits one small IR program per ComponentType,
never a hand-written kernel.

**The IR has two sections with different fates:**
- **`.alloc`** — declares the model-specific engine buffers (`param`/`state`/`accum`/`peredge`/
  `regime`/`expose`/`require`). **Interpreted by the engine at init** to size/allocate buffers; NOT
  compiled. Base engine buffers (`network_inputs`, k²-tree, U/V basis, `last_spiked`, active set) are
  implicit/engine-owned and never appear in `.alloc`.
- **`.tick`** — stage-tagged single-op instructions over named operands (no registers, no expression
  grammar). **Template-substituted to MSL/CUDA** and assembled with every other type's `.tick` into
  **one master kernel** (dispatch per neuron by cell-type boundary), compiled once via
  `Engine_Backend::create_function` and cached. Governing principle: IR→GPU source is near-direct template
  substitution — each op maps to one fixed snippet.

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
