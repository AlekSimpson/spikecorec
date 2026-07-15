# spikecorec — project conventions

spikecorec is a GPU spiking-neural-network **simulation engine** (Metal on macOS, CUDA elsewhere).
Today it runs a single hardcoded leaky-integrate-and-fire (LIF) cell over a compressed adjacency
graph. The **current major effort is a NeuroML → GPU codegen path** (epic #1) so the engine can
simulate arbitrary LEMS-defined cell/synapse dynamics — see "Current epic" below.

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
docs/                 — design docs (the NML codegen architecture + IR spec live here)
```

## Architecture

- `src/core/core.cpp` is the **CPU-side home** — `Engine` main loop + Metal/CUDA backend dispatch implementation all live here
- `include/spikecorec/core/backend.h` defines the CPU↔GPU dispatch interface
- `src/cuda/` and `src/metal/` contain GPU kernel code and the metal-cpp one-time impl file
- Metal host code uses **metal-cpp** (pure C++, no `.mm` / Objective-C++)
- Backend is selected at compile time via `-DSPIKECOREC_METAL` or `-DSPIKECOREC_CUDA`

## Engine execution model (load-bearing facts)

These are the fixed engine facts any codegen/simulation work must respect:

- **Clock-driven tick loop, 9 stages.** Every tick advances all active state by one `dt`:
  1 Deliver · 2 Integrate · 3 Detect · 4 Emit · 5 Reset · 6 Propagate · 7 Plasticity · 8 Record ·
  9 Advance. Stages 1/6/8/9 are engine-owned; 2–5 are the per-cell/synapse dynamics. LEMS events
  (`OnCondition`/`OnEvent`/`EventOut`) are **per-tick conditionals**, not an async scheduler.
- **Runtime kernel compilation already exists on both backends** via `compile_kernel(source, name)`
  in `src/core/backend.cpp` (Metal `newLibrary`; CUDA NVRTC → PTX → `cuModuleGetFunction`). Codegen
  output is fundamentally "a device-code string" the engine compiles and runs.
- **CUDA has no generic kernel launcher.** `metal_dispatch` is already a generic positional-arg
  launcher; the CUDA side only launches its fixed precompiled kernels. Running a generated master
  kernel on CUDA needs a `cuda_dispatch(CUfunction, grid, block, void** params)` wrapper — this is
  ticket **#56 (C4)**.
- **Active-set lazy decay is only valid for LINEAR dynamics.** The active-set optimization advances
  a skipped neuron across many ticks in one closed-form `apply_decay`. That is valid only for
  analytically-integrable (linear) cells (all of GLIF). Nonlinear cells (izhikevich/AdEx/HH) must
  integrate exactly one `dt` per active tick and re-enqueue while non-resting — ticket **#62 (F1)**.
- **`network_inputs`** is the per-neuron synaptic input accumulator: a source scatters its weight
  into `network_inputs[target]`; the target drains it at its next step (an implicit ≥1-tick latency).
  External stimulus is added straight into `membrane_potentials`.

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

The codegen epic **generalizes** this same compression to per-edge synapse state (a shared U/V
basis + per-matrix coefficient vectors `Ck` + sparse delta buffers `Sk` + periodic refit; tickets
#52/#53/#54). That generalization is **still memory compression, not learning** — the same
invariant holds.

In short: **U/V factorization = memory savings. The engine does not train on tasks.**

## Current epic: NeuroML → GPU codegen (#1)

Add a NeuroML (NML/LEMS) → GPU kernel codegen path so the engine can simulate arbitrary
LEMS-defined dynamics instead of the single hardcoded LIF cell.

**Authoritative design docs — read these before doing any codegen work:**
- `docs/nml_codegen_architecture.md` — master architecture: engine grounding (§0), classification
  (§1), the 9-stage tick scaffold (§2), the **per-tag ComponentType reference (§3, the centerpiece)**,
  allocation data structures (§4), the three-phase roadmap (§5), ticket mapping (§6).
- `docs/nml_ir_spec.md` — the **IR spec, LOCKED v1.0** for Phases 1–2. Any change to the locked
  surface goes through an explicit spec revision.

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
  `compile_kernel` and cached. Governing principle: IR→GPU source is near-direct template
  substitution — each op maps to one fixed snippet.

**Three phases** (each additive):
- **Phase 1** — GLIF cells (GLIF1–GLIF5 in full) + the full synapse model (current-based,
  conductance-based, and per-edge), single cell through networks.
- **Phase 2** — nonlinear point cells (izhikevich/AdEx/…), spike delays, on-device generators,
  within-population heterogeneity.
- **Phase 3** — biophysical / multicompartment (HH channels, concentrations, kinetic schemes,
  morphology) — the full NML surface.

**Ticket hierarchy** (GitHub sub-issues; the epic body #1 holds the authoritative checklist +
build-order graph). Tickets carry `[CODE]` title prefixes and are labeled `codegen` + `phase-N`:

```
#1  [Epic] NeuroML → GPU codegen
├── #2  [A3] Parser (front-end)         → #3 [A1] std-lib · #8 [A2] XSD · #7 [A5] ModelSpec · #49 [A4] resolve/lower
├── #4  [B1] IR representation          → #50 [B2] cell lowering · #51 [B3] synapse lowering ·
│                                          #55 [C1] IR→GPU source · #5 [C2] .alloc allocator ·
│                                          #6 [C3] master-kernel assembly · #56 [C4] cuda_dispatch
├── #52 [D2] shared-basis WeightMatrix  → #53 [D3] sparse Sk + loadedge/accedge · #54 [D4] refit · #57 [D1] aggregated accumulators
├── #61 [H1] Phase-1 validation & wiring→ #58 [E1] stimulus · #59 [E2] recording · #60 [X1] diagnostics
├── #67 [H2] Phase 2                    → #62 [F1] active-set×nonlinear · #63 [F2] nonlinear cells ·
│                                          #64 [F3] delay ring · #65 [F4] on-device gens+heterogeneity · #66 [F5] plasticity wiring
└── #74 [H3] Phase 3                    → #68 [G1] channels/gates · #69 [G2] concentrations ·
                                           #70 [G3] multicompartment (+ deferred IR construct) ·
                                           #71 [G4] kinetic schemes · #72 [G5] stiff integrators · #73 [G6] Q10
```

Notes for working these tickets:
- **Build order is a strict prefix:** front-end (A1→A5) → IR gen (B) → IR back-end (C) → the rest.
  `#56 (C4)` has no deps and only blocks `#6 (C3)`. Sub-issue links are single-parent; where a
  ticket supports more than one parent, the extra links live in the body as "Depends on".
- Each ticket body cites the exact arch/IR-spec sections needed to implement it — no extra design
  work should be required.
- Phase 3 tickets (G*) are intentionally **coarse**; flesh them out when Phase 3 is scheduled. The
  one deferred IR item (child/neighbor-set access for multicompartment/kinetic schemes) is pinned
  down in `#70 (G3)`.

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
- Feature/Dev Branches follow `SC-<issue-number>_<short-description>`, e.g. `SC-25_add_branching_factor_bound_checks`
  is the working branch for issue #25 — the numeric prefix is the GitHub issue number, so
  `gh issue view <N>` finds the corresponding issue directly.
- Issue labels: severity (`severity: critical|high|medium|low`) + category
  (`correctness`, `performance`, `enhancement`, plus the GitHub defaults like `bug`/`documentation`).
  The NML codegen epic adds `codegen`, `phase-1`/`phase-2`/`phase-3`, and `epic`.
- `gh pr create` does **not** automatically attach the PR to the project board, even with
  `Closes #N` in the body (that only closes the issue on merge). After creating a PR, attach it explicitly:
  ```bash
  gh project item-add 5 --owner AlekSimpson --url <pr-url>
  ```
- When a coding agent is assigned a task to work on, it should always check out a new feature branch to do the work on.
  - Once a dev feature is complete and the review agent has approved it then all dev branches must make PRs into the "nightly" branch of the project. 
  - ALL LLM DEV BRANCHES THAT OPEN A PR TO MASTER WILL BE DECLINED
- **Sub-issues** are set via the REST API (`gh` has no direct command). To make issue `C` a child of
  issue `P`: `gh api --method POST repos/AlekSimpson/spikecorec/issues/<P>/sub_issues -F sub_issue_id=<C-database-id>`
  where the database id comes from `gh api repos/AlekSimpson/spikecorec/issues/<C> --jq .id`. Read a
  parent's children via `.../issues/<P>/sub_issues` (the REST issue object has no `.parent` field).
- Heads up: the auto-mode Bash gate blocks **bulk/loop** `gh issue create`; create/edit issues with
  explicit per-issue commands, not shell loops.
