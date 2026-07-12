# NML → GPU Codegen — Master Architecture

**Purpose.** The single reference for how NeuroML (NML) / LEMS models map onto the spikecorec engine
for GPU codegen. The center of the document is a per-tag reference (§3) for every element that can
appear in a LEMS `ComponentType`, answering for each: **what it specifies, which stage of the
per-tick step it affects, and what it forces to be allocated.** Everything else — engine grounding,
classification, the step scaffold, allocation, and the three-phase roadmap — frames that mapping.
This document consolidates and supersedes the earlier planning notes; where anything disagrees, this
document is authoritative.

**Altitude.** This document discusses implementation the way an architecture should — at the level of
the **patterns, data structures, and algorithms** that fit the feature goals (see §4 for allocation
layouts and data structures, and the tag reference in §3 for how each element maps to per-tick
behavior and storage). It does **not** include literal code samples, exact kernel signatures, or a
mandated AST/unparser design; those stay the implementer's choice. Two things are fixed as *decisions*
rather than left open: the codegen is generic — each ComponentType is lowered independently to a small
IR, which the engine assembles into **one master kernel** as the final compiled output (generic at the
IR level, one kernel at the source level — see `nml_ir_spec.md`) — and the tag → stage → allocation
semantics of §3. The invariant is:
fix *what each tag means*, *where in the tick it lands*, and *the kind of structure that serves it*;
leave the byte-level code to implementation.

**Conventions.** Values are canonical SI (units resolved away before codegen). Tables are avoided in
favor of nested bullets. Short code appears only to clarify *semantics*, never as an implementation
spec.

---

## 0. Engine grounding — what exists today

The fixed scaffold every NML model must map onto. File references are to the current tree.

### 0.1 `SpikeEngine` state (per-neuron flat arrays)

From `include/spikecorec/core/engine.h`. All `[neuron_count]`, in unified/shared memory:

- `network_inputs` `f32` — the recurrent synaptic accumulator. A spiking source atomically adds its
  edge weight into `network_inputs[target]`; the target drains it (exchange→0) at the top of its
  next step. One global, current-based, decaying channel.
- `membrane_potentials` `f32` — the single scalar cell state `v`.
- `last_spiked` `s64` — tick each neuron last fired (emit output; STDP timing; refractory clock).
- `last_tick_updated` `s64` — tick each neuron was last integrated (drives lazy decay, see 0.3).
- Active-set arrays (`active_neuron_indices`, `active_generation`, counts, …) — the active-set
  optimization.
- Scalar config: `resting_membrane_potential`, `decay_rate`, `learning_rate`, `spike_period`,
  `spike_threshold`.

### 0.2 The current step already implements the whole tick for one hardcoded cell

`src/metal/kernels.metal` `step` (and the CUDA twin), one thread per active neuron, does — in one
pass — deliver (drain `network_inputs` + lazy decay), integrate (leaky relaxation), detect
(threshold), emit (`last_spiked=tick`), reset (refractory via `spike_period`), propagate (k²-tree
row-walk scattering weights into `network_inputs[child]`), plasticity (inline rank-1 STDP on U/V),
and active-set enqueue. Two facts inherited by any NML mapping:
- **Implicit one-tick synaptic latency:** a source scatters into `network_inputs[child]`; the child
  drains it at *its* next step.
- **External stimulus is added straight into `membrane_potentials`** (not `network_inputs`).

### 0.3 `WeightMatrix` — exact support + low-rank state

- `K2Tree` — exact adjacency, bit-packed, GPU-walkable; pins the exact edge set.
- `U_matrix`/`V_matrix` `float4[node_count × rank_stride]`; weight of edge `i→j` = `dot(U[i], V[j])`,
  reconstructed only at real edges. `update(...)` is a rank-1 proximal Hebbian update (the
  plasticity path). This is the single-matrix, unit-coefficient special case of the shared-basis
  per-edge representation that §4.3 generalizes for richer synapses.

### 0.4 Backend can compile generated code at runtime (both targets)

`compile_kernel(source, name)` already compiles a *source string* at runtime — Metal via
`newLibrary`, CUDA via NVRTC. So codegen's output is fundamentally "a string of device code" that
the engine compiles and runs; the mechanism already exists on both backends. One backend gap to note:
Metal already exposes a generic runtime dispatch (arguments resolved positionally), whereas the CUDA
side currently launches only its fixed, precompiled kernels — a generic "launch an arbitrary compiled
function" wrapper is the missing piece for running the generated master kernel on CUDA. Codegen emits
per-ComponentType IR (see `nml_ir_spec.md`), lowers and assembles it into one master kernel, and
compiles that once at model-build time, caching it (recompiled only when the model changes).

### 0.5 The one hard constraint the scaffold imposes

The engine's active-set optimization *lazily* advances a skipped neuron across many ticks in one
closed-form decay step. That is valid only for **analytically-integrable (linear)** dynamics.
General NML `TimeDerivative` systems (izhikevich, AdEx, HH) are nonlinear and **cannot be skipped and
fast-forwarded** — they must integrate exactly one `dt` on every tick they are active, and re-enqueue
themselves while non-resting. Linear types (all of GLIF) keep the closed-form lazy advance and reuse
the active set unchanged. This is a semantic constraint on the mapping, independent of how codegen is
implemented; the codegen tags each cell type linear/nonlinear to apply it.

---

## 1. Classification and lifecycle

### 1.1 Three buckets

```
Is the node a ComponentType?
├─ No ───────────────────► STATICS     units/dimensions, attribute VALUES, <include>, IDrefs, ids
└─ Yes → declares <Dynamics>?
          ├─ Yes ─────────► DYNAMICS    → per-tick step behavior (the codegen target)
          └─ No ──────────► STRUCTURES  → engine init: allocation + topology + I/O
```

- Dynamics ComponentTypes are the only things that become per-tick step behavior.
- Structures configure layout/wiring/IO; consumed at build time.
- Statics never stream to the backend: units and `<include>` **evaporate at resolve**; parameter
  *values* and IDrefs **attach to** a Dynamics/Structure node.
- Declaration-vs-value seam: `<Parameter name="a"/>` (declaration, part of the type) is
  Dynamics/Structure; `a="0.02"` (the value) is a Static consumed later.

### 1.2 Statics lifecycle subtypes (S1–S7)

- **S1 Units/dimensions** — `<Dimension>`, `<Unit>`, dimensioned literals. **Evaporate at resolve**:
  a unit pass normalizes everything to canonical SI so downstream is unit-free; time quantities route
  through SC-9 to tick counts.
- **S2 Parameter values** — attribute values; destinations: baked constants (default), engine/model
  config, or initial state (S3).
- **S3 Initial values** — `initMembPotential`, `v0`, gate steady states, Ca pools → seed state
  arrays at INIT.
- **S4 Global sim constants** — `duration`→lifetime, `step`→dt (SC-9), `seed`, `temperature`
  (Q10, deferred).
- **S5 Directives** — `<include href>` merges the std lib (#3) and user files; gone after resolve.
- **S6 Identity/metadata** — `id` (symbol-table key), `notes`/`annotation`/etc. (logged/ignored).
- **S7 IDrefs** — `component`/`cell`, `presynapticPopulation`/`postsynapticPopulation`, `synapse`,
  `ionChannel`, `target`, … → become integer indices/ranges at resolve; unresolved = error (#8).

### 1.3 Pipeline

```
inputs: .nml + LEMS_*.xml + NeuroML2CoreTypes/ (#3)
   ▼ PARSE (#2)      XML → transient tree
   ▼ VALIDATE (#8)   XSD conformance gate
   ▼ RESOLVE         merge <include>; units→SI; IDref wiring; ComponentType extends/Fixed merge;
   │                 sort nodes Static/Dynamics/Structure
   ▼ LOWER → ModelSpecification (#7)   flat tables; only per-type dynamics keep expression trees
   ├─► ALLOCATE (#5)   populations → state vector + boundaries, synapse state, delay ring, regimes
   ├─► CODEGEN (#4)    per Dynamics ComponentType → per-tick step behavior (§3 tag → stage)
   └─► INIT            adjacency → WeightMatrix; S3 → seeds; S2/S4 → config
   ▼ RUN             per-tick loop; outputs (ST6) → recording
```

Resolution mechanics worth fixing as data-structure choices: a **symbol table keyed by `id`** backs
IDref wiring (S7); `extends`/`Fixed` are handled by a **merge pass** that flattens each ComponentType
so codegen only ever sees a fully-resolved type; the parse tree is transient and thrown away at LOWER.

### 1.4 The lowered ModelSpecification (data structure)

After resolve, the model is *lowered* into a flat, index-addressed representation — the
ModelSpecification (#7) — and the parse tree is discarded.

- **Why flat indexed tables, not a walked tree:** the counts that matter here are *types* and
  *populations* (dozens to hundreds), never neurons (millions). Tables over types/populations are
  cheap, cache-friendly, and avoid pointer-chasing; allocation and topology are built by scanning
  them. This is the single most important data-structure decision in the front half of the pipeline.
- **What it holds** (each a flat table, index-addressed):
  - Cell-type and synapse-type library — each entry carries its dynamics (the expression trees plus
    stage structure), baked constants, initial values, and classification flags (conductance-based vs
    current-based, aggregatable vs per-edge).
  - Populations — each: cell-type index, size, neuron-index range, and state-vector chunk base offset.
  - Adjacency + initial weights — the existing `vector<vector<s32>>` → `WeightMatrix`/k²-tree path.
  - Projection/synapse routing and any delay data; input (stimulus) specs; output (recording)
    selections.
- **The one thing that stays tree-shaped:** the *expression ASTs inside each type's dynamics* (an ODE
  right-hand side is irreducibly a tree). Everything else is flat. This is the seam between "resolved
  once, then indexed" and "carried into codegen as trees."

---

## 2. The 9-stage step scaffold (the fixed skeleton tags map onto)

**Execution model: clock-driven.** The engine integrates on a fixed time grid — every tick advances
all active state by one `dt`. LEMS event elements (`OnCondition`, `OnEvent`, `EventOut`) are **not**
an asynchronous event scheduler; they are per-tick conditionals evaluated inside the compute phase.
This is the only model that supports arbitrary nonlinear `TimeDerivative` systems (whose spike times
have no closed form), and it matches the reference LEMS simulators.

Every tick runs these stages. Stages marked *fixed* are engine-owned; stages marked *NML* are filled
from a ComponentType's dynamics. The tag reference (§3) names, for each tag, which of these it
affects.

- **1 Deliver** *(fixed)* — read spikes arriving this tick (delay ring / accumulator) + external
  stimulus → per-neuron delivered input. Also where synapse-arrival handlers (`OnEvent`) apply.
- **2 Integrate** *(NML — `TimeDerivative`)* — advance all neuron + synapse state one `dt` by
  numerical integration.
- **3 Detect** *(NML — `OnCondition` test)* — test spike/threshold conditions on post-integration,
  pre-reset state.
- **4 Emit** *(NML — `EventOut`)* — firing neurons raise a spike; write `last_spiked`.
- **5 Reset** *(NML — `OnCondition` `StateAssignment`)* — reset firers (v→reset, adaptation bump,
  refractory entry).
- **6 Propagate** *(fixed)* — scatter each spike to targets (k²-tree), into the delay slot due later.
- **7 Plasticity** *(NML-configured, engine-run)* — weight update from pre/post timing
  (`WeightMatrix::update`).
- **8 Record** *(NML-configured — `OutputFile`/`EventOutputFile`)* — sample state + spike events.
- **9 Advance** *(fixed)* — tick++, refractory/active-set bookkeeping.

Ordering law: detect → emit → reset. The compute phase (2–5) is what a cell/synapse type's dynamics
fills; stages 1/6 are the two ends of spike delivery; 7/8/9 are cross-cutting/fixed.

---

## 3. ComponentType tag reference (the centerpiece)

For every tag: **what it specifies**, the **stage(s)** it affects (from §2), its **effect in that
stage** (behavior it contributes — described semantically, not as generated code), and its
**allocation** impact. `V_t` = a cell type's intrinsic state-variable count; per-edge synapse state
uses the shared-basis representation of §4.3.

### 3.1 Declaration-level tags

- **`ComponentType`** (`name`, `extends`)
  - Specifies: a reusable type. With `<Dynamics>` → a Dynamics node; without → a Structure node.
  - Stage: build-time (resolve).
  - Effect: `extends` merges the parent's declarations and entire `<Dynamics>` into the child at
    resolve — codegen only ever sees the fully-flattened type. The `extends` chain also *classifies*:
    `baseConductanceBasedSynapse` in the chain → this synapse's current is `g·(erev − v)` (needs the
    postsynaptic `v`, so it belongs inside the neuron's integrate stage); a current-based chain →
    `i = g`. `baseCell` + morphology → biophysical (deferred).
  - Allocation: none directly; the merged `StateVariable` set defines `V_t`.

- **`Parameter`** (`name`, `dimension`)
  - Specifies: that instances of this type *have* a named, time-invariant quantity of the given
    dimension (e.g. izhikevich `a,b,c,d`). **The tag is a declaration only — it carries no value and
    no signal of whether the value is uniform.** Whether it becomes a baked constant or a per-neuron
    value is never readable from the tag; it is decided from the instance *values* at lower time.
  - Stage: whichever stages its dynamics reference it in.
  - Effect: supplies a value into expressions. The **bake-vs-parameterize rule**: gather the
    parameter's resolved value across every neuron in one population/chunk — *all
    equal* → **bake** it as a literal in the generated code (zero memory, best constant-folding); *not
    all equal* → **parameterize** (a per-neuron array read as `param[neuron]`). In standard NeuroML a
    `population` binds one `component`, so its neurons share one value → baked by default; genuine
    within-population heterogeneity only arises from Structure-level constructs (the modeler splitting
    cells into distinct components, per-cell values drawn from a distribution, or — biophysical,
    Phase 3 — `inhomogeneousParameter`/`inhomogeneousValue` varying a density along a morphology).
  - Allocation: baked → none (the literal lives in the generated code); heterogeneous → one
    per-neuron array of population size for that parameter. Applies only to time-invariant quantities
    (`Parameter`/`Constant`/`DerivedParameter`/`Property`/`Fixed`); a `StateVariable` evolves and is
    always per-neuron, never baked.

- **`Constant`** (`name`, `value`) — Specifies a fixed literal. Stage: referencing stages. Effect:
  a constant in expressions. Allocation: none.

- **`DerivedParameter`** (`name`, `value`/`select`) — Specifies a value computed from
  parameters/constants (e.g. `1/tau`). Stage: build-time (init). Effect: precomputed once and baked;
  not recomputed per tick. Allocation: none.

- **`Property`** (`name`, `defaultValue`) — Specifies a settable parameter with a default (often
  `weight`, `delay` on connections; spatial props). Like `Parameter`, the declaration carries no
  uniformity signal — it follows the same bake-vs-parameterize rule (above), and is in fact the more
  common carrier of genuine per-instance variation (per-connection `weight`/`delay`, spatially-varying
  values). Stage: referencing stages / build-time. Effect: connection `weight`/`delay` route into the
  edge weight / delay data. Allocation: baked → none; `delay` → the delay data (§4.4).

- **`Exposure`** (`name`, `dimension`)
  - Specifies: an observable quantity (`v`, `u`, `iSyn`, `caConc`) that recording may select.
  - Stage: interacts with 8 Record.
  - Effect: no behavior by itself. If a *stored* `StateVariable` is exposed, recording reads it
    directly. If a *derived* value is exposed and selected, it must be kept live long enough to
    sample (otherwise a derived value is transient).
  - Allocation: none unless recorded; a recorded derived exposure may need a scratch per-neuron slot.

- **`Requirement`** (`name`, `dimension`)
  - Specifies: a quantity the type needs from its enclosing scope (a channel requires the cell's `v`;
    a concentration model requires an ion current).
  - Stage: consumed wherever the dynamics use it (usually 2 Integrate).
  - Effect: a binding — when a child's dynamics are composed into a parent, the requirement name is
    bound to the parent's owning variable. Unbound = resolve error.
  - Allocation: none (it is a binding; storage lives with the owner of the quantity).

- **`EventPort`** (`name`, `direction=in|out`)
  - Specifies: a spike port. `out` = this component can fire; `in` = it can receive spikes.
  - Stage: `out` → 4 Emit / 6 Propagate; `in` → 1 Deliver (arrival handling).
  - Effect: `out` is where `EventOut` fires; `in` is where an `OnEvent` handler runs on arrival.
  - Allocation: none itself; propagation uses the delay data + k²-tree.

- **`Child`** (`name`, `type`) / **`Children`** (`name`, `type`, `min`, `max`)
  - Specifies: containment — a component holds sub-components (a cell holds gates/synapses; a channel
    holds gates).
  - Stage: composed into the owner's stages (usually 2 Integrate); a `select`/`reduce` over
    `Children` runs there too.
  - Effect: the child's dynamics become part of the owner's per-tick behavior. `Children` + a parent
    `DerivedVariable reduce="add"` = "sum over the children" (e.g. total synaptic current). For
    aggregatable synapses this sum collapses to reading one per-neuron accumulator instead of
    iterating edges.
  - Allocation: each child instance adds its state to the owner — gates/pools → extra intrinsic
    state (`V_t`, biophysical, deferred); aggregatable synapses → one shared per-neuron accumulator
    slot (§4.2); non-aggregatable synapses → per-edge state (§4.3).

- **`ComponentReference`** (`name`, `type`) / **`Link`** (`name`, `type`)
  - Specifies: a reference to another component (projection→its synapse type; channelDensity→its
    channel) or a sibling instance (`Link`, e.g. a gap-junction peer).
  - Stage: build-time wiring; a `Link`-read of a peer's value happens in 2 Integrate (graded/gap,
    deferred).
  - Effect: tells the engine *which* dynamics govern an edge/population.
  - Allocation: none directly.

- **`Attachments`** (`name`, `type`)
  - Specifies: an attachment point — instances of `type` (synapses) attach to this component (a
    cell); the seam that ties synapse dynamics to a target cell.
  - Stage: 1 Deliver (arriving synaptic input lands here).
  - Effect: routes a projection's synapse contributions onto the postsynaptic cell.
  - Allocation: the postsynaptic accumulator slot (§4.2) or per-edge state (§4.3), per the synapse's
    aggregatability.

- **`Text`** (`name`) / **`Path`** (`name`) — Specifies a string/structural-path parameter
  (metadata, output file names, `select` targets). Stage: build-time. Effect: none in the step.
  Allocation: none.

- **`Fixed`** (`parameter`, `value`) — Specifies a pinned value for an inherited parameter. Stage:
  build-time (resolve). Effect: forces that parameter to a baked constant. Allocation: none.

- **`Dynamics`** (container) — Specifies the compute behavior. Stage: 2–5 (+1/6 via event handlers).
  Effect: its body is the codegen target (§3.2). Allocation: its `StateVariable`s are the primary
  driver (§4.1).

- **`Structure`** (container: `ChildInstance`, `MultiInstantiate`, `ForEach`, `With`,
  `EventConnection`, `Tunnel`, `Assign`) — Specifies how instances and connections are *built*
  (distinct from the "Structures" bucket name). Stage: build-time. Effect: instantiates populations
  and spike edges → allocation + adjacency; no per-tick behavior. Allocation: drives `neuron_count`,
  populations, and the k²-tree adjacency.

- **`Simulation`** (container: `Run`, `Record`, `EventRecord`, `DataDisplay`, `DataWriter`,
  `EventWriter`) — Specifies the run + outputs. Stage: build-time + 8 Record. Effect: `Run`→lifetime/
  dt/target; `Record`/`EventRecord`/writers→recording selections. Allocation: recording buffers;
  `total`/`step` fix lifetime and dt.

### 3.2 Tags inside `<Dynamics>`

- **`StateVariable`** (`name`, `dimension`, `exposure`)
  - Specifies: a continuously-evolving quantity (`v`, `u`, gate `q`, synapse `g`).
  - Stage: allocation feeds all of 2–5 (read in Integrate, tested in Detect, written in Reset).
  - Effect: it is loaded, evolved, and stored each tick; if `exposure` is set it is directly
    recordable.
  - Allocation: **the primary driver** — one state slot per instance (a per-neuron value for cell
    state; for a per-edge synapse state, a per-matrix coefficient vector `Ck` over the shared basis
    plus its sparse delta buffer `Sk`, §4.3). A cell with `v,u` needs two; HH needs five.

- **`TimeDerivative`** (`variable`, `value`)
  - Specifies: the ODE `d(variable)/dt = value`.
  - Stage: **2 Integrate.**
  - Effect: advances `variable` one `dt` by numerical integration (method chosen per-derivative:
    forward Euler by default; exponential Euler for a state whose RHS is linear in it — gating, linear
    conductance decay, the leaky case, where it matches the engine's existing exponential decay
    exactly; higher-order/implicit where stiff, deferred). Method selection is a static property of
    the RHS the parser determines.
  - Allocation: none beyond the `StateVariable` it advances (multi-stage integrators use registers).

- **`DerivedVariable`** (`name`, `exposure`, and `value` or `select`+`reduce`+`required`)
  - Specifies: a quantity computed each tick — either an expression (`value`) or an aggregation over
    children (`select`/`reduce`).
  - Stage: the stage that consumes it (usually 2 Integrate; sometimes 3 Detect).
  - Effect: `value` form = an inline computed quantity (a current term, a rate). `select`/`reduce`
    form = a sum/product over selected child exposures (e.g. total synaptic current). With
    aggregatable synapses the reduction is precollapsed to one accumulator read.
  - Allocation: none, unless exposed-and-recorded, or needed across a pass boundary → a scratch slot.

- **`ConditionalDerivedVariable`** with **`Case`** (`condition`, `value`)
  - Specifies: a piecewise-defined derived value (e.g. a rate law guarded at a singularity).
  - Stage: the stage of the value it defines (usually 2).
  - Effect: a branch — the value takes the first matching case, else the default.
  - Allocation: none.

- **`OnStart`** (contains `StateAssignment`)
  - Specifies: initial values (`v=v0`, `u=v0*b`).
  - Stage: **build-time INIT — no per-tick stage.**
  - Effect: seeds the state arrays (Statics S3), not part of the per-tick step.
  - Allocation: writes into already-allocated state; non-uniform seeds need a fill (SC-17).

- **`OnCondition`** (`test`) containing **`StateAssignment`**, **`EventOut`**, **`Transition`**
  - Specifies: "when `test` holds, do these actions" — the threshold/reset/spike rule.
  - Stage: `test` → **3 Detect**; nested `EventOut` → **4 Emit**; nested `StateAssignment` → **5
    Reset**; nested `Transition` → 9 (regime change).
  - Effect: the canonical spike path — `test="v > vpeak"`, then emit a spike, then reset (`v=c`,
    `u=u+d`). Honors detect→emit→reset ordering (the test reads pre-reset state).
  - Allocation: none beyond referenced state; a nested `Transition` needs the regime index (§4.5).

- **`OnEvent`** (`port`) containing **`StateAssignment`**
  - Specifies: "when a spike arrives on `port`, do these actions" — the arrival handler (`g += weight`).
  - Stage: **1 Deliver** (applied as input lands).
  - Effect: bumps synaptic state on arrival. For aggregatable synapses this is an add into the
    per-neuron accumulator; for per-edge synapses, an accumulate into the edge's sparse delta buffer
    `Sk[i,j]` (§4.3).
  - Allocation: none new — writes the synapse state it references.

- **`StateAssignment`** (`variable`, `value`) — Specifies a discrete write (vs continuous
  `TimeDerivative`). Stage: 5 Reset (in `OnCondition`), 1 Deliver (in `OnEvent`), or INIT (in
  `OnStart`). Effect: `variable = value` at that point. Allocation: none beyond the target.

- **`EventOut`** (`port`) — Specifies "fire a spike on `port`." Stage: **4 Emit** (feeds 6 Propagate).
  Effect: writes `last_spiked` and marks the neuron for scatter to downstream targets. Allocation:
  none new.

- **`Regime`** (`name`, `initial`) containing **`OnEntry`**, **`TimeDerivative`**, **`OnCondition`**,
  **`Transition`**
  - Specifies: a state machine — the dynamics differ by mode (e.g. integrating vs refractory in
    `iafRefCell`).
  - Stage: a mode switch over 2–5; `Transition` acts at 9.
  - Effect: the active regime selects which integrate/detect/reset applies; `OnEntry` runs on entry
    (e.g. set a refractory timer); `Transition` changes the regime.
  - Allocation: **one regime-index slot per instance**; any per-regime timer is an extra
    `StateVariable`.

- **`OnEntry`** (contains `StateAssignment`) — Specifies actions on entering a regime. Stage: 9 →
  next tick's 2. Effect: assignments guarded by "just entered." Allocation: none beyond referenced
  state.

- **`Transition`** (`regime`) — Specifies a regime change. Stage: 9. Effect: sets the regime index.
  Allocation: writes the regime index.

- **`KineticScheme`** (`nodes`, `stateVariable`, `edges`, `forwardRate`, `reverseRate`) — Specifies a
  Markov ion-channel scheme (`ionChannelKS`). Stage: 2 Integrate. Effect: integrates coupled
  occupancy ODEs. Allocation: one state slot per scheme node per instance + rate scratch.
  **Deferred (biophysical).**

### 3.3 Standard NML ComponentTypes by bucket (the D1–D6 / ST1–ST6 families)

Concrete instances of the tags above — the full target surface.

- **Dynamics / point cells (D1)** — `izhikevich2007Cell` (`v,u`), `iafCell`/`iafTauCell`/`iafRefCell`
  (`v` + refractory regime), `adExIaFCell` (`v,w`), `fitzHughNagumoCell` (`V,W`),
  `hindmarshRose`/`pinskyRinzelCA3Cell` (3+). Fill stages 2–5; allocate `V_t` per neuron. **Primary
  early target — linear GLIF/LIF cells in Phase 1, nonlinear cells in Phase 2.**
- **Dynamics / channel gates (D2)** — `gateHH*`, `HH*Rate`, `q10*`. Composed into a biophysical cell
  (§3.1 `Child`), not standalone. **Deferred.**
- **Dynamics / synapses (D3)** — `alphaCurrentSynapse` (current-based, aggregatable), `expOneSynapse`
  (aggregatable; `g(erev−v)` if conductance-based), `expTwoSynapse`/`alphaSynapse` (aggregatable),
  `blockingPlasticSynapse` (NMDA, **per-edge**), `gradedSynapse`/`gapJunction` (continuous,
  **deferred**). Split: §4.2 vs §4.3.
- **Dynamics / inputs (D4)** — `pulseGenerator`, `sineGenerator`/`rampGenerator`, `voltageClamp`,
  spike sources (`spikeArray`/`SpikeSourcePoisson`/…), Poisson synaptic drive, `compoundInput`.
  Host-precomputed into the stimulus buffer in Phase 1; on-device generators in Phase 2.
- **Dynamics / concentration (D5)** — `decayingPoolConcentrationModel`/`fixedFactorConcentrationModel`.
  Extra per-neuron state coupled to channel currents. **Deferred.**
- **Dynamics / plasticity (D6)** — STDP mechanisms → the weight-update path (stage 7); Tsodyks–
  Markram short-term plasticity → a *synapse state variable* (stage 2). Presence ↔ `enable_plasticity`
  / absence ↔ `disable_plasticity` (SC-11). Params (`τ±,A±`) → baked constants. Per CLAUDE.md: local
  synaptic plasticity, not task learning, not the U/V compression.
- **Structures / network (ST1)** — `network`/`networkWithTemperature` → one `ModelSpecification` → one
  `SpikeEngine`; `temperature`→Q10 (deferred).
- **Structures / populations (ST2, #5)** — `population` (`size`,`component`) → a contiguous neuron
  range + state-vector chunk; `populationList`/`instance` → explicit placement (positions ignored for
  point cells). Emits the cell-type boundaries.
- **Structures / projections (ST3)** — `projection`, `electricalProjection` (gap, deferred),
  `continuousProjection` (graded, deferred), `connection`/`connectionWD` (`weight`→edge state seed,
  `delay`→delay data). Every connection → a k²-tree edge.
- **Structures / cell def (ST4)** — `cell` (point → *is* the Dynamics component; biophysical →
  composes channels, deferred), `morphology`/`segment` (multicompartment, deferred),
  `biophysicalProperties`/`membraneProperties`/`channelDensity`, `intracellularProperties`.
- **Structures / type library (ST5)** — `ionChannelHH`/`ionChannel`/`ionChannelKS` (deferred),
  synapse defs (are D3). Become the type tables; each distinct referenced type is codegen'd once.
- **Structures / simulation+output (ST6)** — `Simulation`(`length`,`step`,`target`)→run config;
  `Display`/`Line`→plot selection; `OutputFile`/`OutputColumn`→trace logging (`SimulationRecorder`);
  `EventOutputFile`/`EventSelection`→spike logging (from `last_spiked`).

---

## 4. Allocation — data structures and layout

What the tags force into memory, why, and the layout/algorithm that fits each — at architecture
altitude (sizing intuition and structural choices, not byte-exact addressing).

### 4.1 Per-cell-type state (widened state vector)

Today one scalar `v` per neuron. NML cells carry several state variables, so each population needs a
per-type-width slice: population `p` of cell type `t` needs `size_p × V_t` state values (plus any
aggregated synapse slots, §4.2). `StateVariable` declarations drive this. Neuron-level bookkeeping
(`last_spiked`, `last_tick_updated`, regime index, active-set) stays one-per-neuron regardless of `V_t`.

- **Layout that fits — struct-of-arrays within per-population chunks.** Partition one widened state
  buffer into a contiguous chunk per population; *within* a chunk store all neurons' first state
  variable, then all neurons' second, and so on (SoA), rather than interleaving a neuron's variables
  (AoS). The reason is coalescing: adjacent threads process adjacent neurons, and under SoA they touch
  adjacent addresses for the same variable — the dominant performance factor on both backends. A
  neuron's state is located from its population's chunk base offset, the variable's stride within the
  chunk, and the neuron's local index.
- **Supporting structure — an O(P) boundary array.** The only bookkeeping this needs is a small
  per-population table of chunk base offsets (the "cell-type boundaries," #5). The engine consumes it
  to place and locate state, and the assembled master kernel uses it to dispatch each neuron/thread to
  its cell type's code by cell-type boundary; it is engine bookkeeping, not part of the generated kernel.
- **Rejected alternative — one global array per canonical variable.** Wastes slots for every type
  that lacks a given variable and scatters a cell's state across buffers; the per-population SoA chunk
  packs tightly and keeps each cell type's slice contiguous.

### 4.2 Aggregated per-neuron synapse state

Linear conductance synapses of the same type converging on one neuron superpose, so only the *sum*
need be stored: one accumulator per postsynaptic neuron per aggregatable synapse type (this is the
generalization of today's `network_inputs`, but typed and decaying by the synapse's own `tau`).
Cheap: `O(neuron_count × types × state_vars)`; per-tick decay is a uniform scale. Default path.

### 4.3 Per-edge synapse state — shared basis + per-matrix coefficients + a sparse delta buffer (decided)

**The problem.** Non-aggregatable synapses (NMDA, per-synapse short-term plasticity) need a value stored
*per edge*, and a network has several such per-edge matrices over the same connections — the weight
plus each per-edge synapse state variable. Storing each one densely, or even as its own separate
low-rank factorization, wastes memory.

**The idea, in one picture.** Think of each per-edge matrix as a single point in a huge space. Across
the whole family of matrices, those points cluster near a low-dimensional "plane." Fit that plane once
(the shared basis `U`, `V`), then store each matrix cheaply as *its position on the plane* (a short
coefficient vector `Ck`) instead of the full matrix. Updates that don't lie on the plane are set aside
in a scratchpad (`Sk`) and folded back in periodically by re-fitting the plane. That is the whole
scheme — everything below is detail. (This strictly generalizes today's `WeightMatrix`: one matrix, a
flat plane, no scratchpad.)

**The three things you store:**
- **`U`, `V` — the shared basis** (`[node_count × rank]` each, *one* pair for the entire family). This
  is the "plane." The weight and every per-edge state variable share it; it is not copied per variable.
- **`Ck` — one short coefficient vector** (`[rank]`) per matrix `k`. It says where matrix `k` sits on
  the plane.
- **`Sk` — one sparse scratchpad** (CSR/CSC) per matrix `k`, holding raw updates not yet folded into
  the plane. One per matrix, so different variables' updates never get jumbled together.

**The three operations:**
- **Read** value of matrix `k` on edge `i→j`: reconstruct from the plane and add any pending scratch —
  `U[i]·diag(Ck)·V[j]ᵀ + Sk[i,j]`. The reconstruction is just today's `dot(U[i],V[j])` with `V` first
  reweighted by `Ck`, so the hot path barely changes.
- **Update** (e.g. a spike bumps a conductance by `+x`): add it to the scratchpad, `Sk[i,j] += x`.
  Cheap and local — no basis touched.
- **Refit** (every `t` ticks): re-fit `U`, `V`, and all `Ck` to the current values (plane + scratch),
  then clear the scratchpads. This is the one heavy step, amortized over `t` ticks.

**Why the scratchpad instead of editing the plane directly.** Two things go wrong if you skip `Sk`:
- You *can't* edit the shared `U`/`V` in place — they define the plane for *every* matrix, so moving
  them for one matrix corrupts all the others.
- You *can* edit a matrix's own `Ck` in place, but that only slides it *along the existing plane*. Any
  part of an update pointing *off* the plane gets flattened to ≈zero and lost, and over time the plane
  stops fitting. The scratchpad keeps those off-plane updates intact so the periodic refit can tilt the
  plane to follow them. (Per CLAUDE.md this is still memory compression of edge values, not learning.)

**Precise form.** value(`k`, `i→j`) = `Σ_r U[i,r]·Ck[r]·V[j,r] + Sk[i,j]`. The current engine is the
special case: one matrix, `Ck = 1`, empty `S`.

**Memory.** One shared basis `O(node_count × rank)` + tiny coefficient vectors `O(matrices × rank)` +
scratchpads `O(nnz in S)`. Sharing the basis (instead of one per variable) is the win.

**The one open knob.** Refit *every tick* → scratchpads stay empty, reads never touch `S`, but you pay
the refit constantly. Refit *every `t` ticks* → cheap amortized refit, but reads must consult a
scratchpad that grows denser (and slower) between refits. The interval is not yet chosen.

**When this path is used (unchanged rule; parser, build time):** a synapse type uses per-edge storage
iff its dynamics don't superpose across converging edges (nonlinear-in-state, edge-dependent, or
non-additive arrival); linear/superposable synapses use the cheaper per-neuron accumulator (§4.2).

### 4.4 Spike delays (new subsystem)

`delay` on connections is an edge property, currently unmodeled beyond the implicit one-tick latency.
Adding real delays needs a buffer that holds, per future tick, the input due then (a ring of
`max_delay_ticks + 1` slots) so a scattered spike lands at the right tick. Sizing scales with the
*longest* delay. Granularity (uniform / per-projection / per-edge) is orthogonal; per-edge delays
need a per-edge delay array (they do not compress into U/V). Early phases keep uniform/zero delay (the
existing implicit one-tick model); the full delay ring is Phase 2.

### 4.5 Regime index and scratch

Cell types with `Regime`s need one regime-index slot per instance (refractory, kinetic schemes);
refractory timers are extra state variables. Recorded *derived* exposures may need a scratch
per-neuron slot; intra-tick derived values are transient.

### 4.6 Recording

`OutputColumn`/`Line` select `(population, exposure, stride)` → sampled from the state slices into
the existing `SimulationRecorder`; `EventOutputFile`/`EventSelection` → spike records from
`last_spiked`. Membrane traces reuse today's `.spire` path; other exposures add recorder streams.

---

## 5. Phased development roadmap (three phases)

Three strategic phases, each grouping a related set of changes and unlocking a broader class of
runnable models. **Phase 1** delivers the prioritized target — the full GLIF family *and* the synapse
machinery those circuits use (current-based, conductance-based, and per-edge synapses), single cell
through networks. **Phase 2** adds nonlinear intrinsic cell dynamics and timing (delays). **Phase 3**
adds biophysics and morphology. Each phase is additive; nothing in a later phase is needed to ship an
earlier one.

**Note on the active set.** The engine's lazy multi-tick decay only holds for linear,
closed-form-advanceable dynamics (§0.5). Linear GLIF cells with current-based synapses reuse it
unchanged; conductance-based coupling (a `g·v` term) and nonlinear cells (Phase 2) require integrating
one `dt` per active tick — the active set still bounds *which* neurons run, it just gives no
multi-tick fast-forward for those.

**GLIF1–GLIF5 are all fully implementable in Phase 1** (LIF, after-spike currents, adaptive/
voltage-dependent threshold, biologically-defined reset rules), as single cells *and* wired into
networks.

### Phase 1 — GLIF cells with the full synapse model (single cell → networks)
- Unlocks: **GLIF1–GLIF5 in full**, plus any other linear point cell, both standalone and in networks
  — wired with the full synapse model: current-based, conductance-based, and per-edge synapses. The
  prioritized modeling target.
- Why it groups: the GLIF family is the priority, and real GLIF circuits need realistic synapses, so
  the synapse machinery ships with them. Linear cells reuse most of the engine directly (active set,
  lazy decay, `network_inputs`-style accumulation, k²-tree, WeightMatrix, STDP, recording); the new
  capabilities are the per-cell-type **multi-variable state vector** and the **synapse machinery**
  (aggregated per-neuron conductances, the conductance driving-force term, and per-edge state).
- NML surface:
  - Whole pipeline: `ComponentType`/`extends`/`Fixed`, `Parameter`/`Constant`/`DerivedParameter`/
    `Property`, Statics S1–S7, the resolve/lower pass.
  - Dynamics: multiple `StateVariable`s (`v`, threshold `θ`, after-spike currents `ASC_k`, synapse
    `g`); multiple linear `TimeDerivative`s; `DerivedVariable` (incl. `select`/`reduce` summing ASCs or
    synaptic input); `OnStart` init; `OnCondition` (threshold vs. a constant *or* a state variable) +
    multi-target `StateAssignment` reset rules + `EventOut`; `OnEvent` arrival bump; `Regime` for the
    refractory period; `ConditionalDerivedVariable`/`Case`.
  - Inputs: `pulseGenerator` (D4, host-precomputed).
  - Synapses: aggregatable current-based (`alphaCurrentSynapse` / current-based `expOneSynapse`);
    conductance-based (`baseConductanceBasedSynapse` chain — `expOneSynapse`/`expTwoSynapse` with
    `erev`); non-aggregatable per-edge (`blockingPlasticSynapse`/NMDA, short-term plasticity).
  - Structures: ST1, ST2 (multiple populations/types), ST3 (weights, no delay), ST6 (`OutputFile`
    traces + `EventOutputFile` spikes).
- Engine work: #2 parser, #3 std-lib bundle, #8 XSD, #7 `ModelSpecification`, resolve/lower; the
  **widened per-cell-type multi-variable state vector** (§4.1) + cell-type boundaries (#5); generic
  per-type codegen for linear ODE systems + multi-state resets (#4); the **synapse machinery** —
  aggregated per-neuron accumulators (§4.2), the conductance driving-force term `g(erev−v)` inside
  integrate, and **per-edge synapse state** (shared low-rank basis + per-matrix coefficient vectors +
  sparse delta buffers, §4.3); regime index for refractory
  (§4.5); exponential/forward Euler. Linear cells with current-based synapses stay
  closed-form-advanceable and reuse the active set unchanged; cells receiving conductance-based /
  per-edge input use the per-tick integration path (§0.5).
- Exit models: a GLIF3/GLIF5 single cell reproducing after-spike-current and spike-frequency
  adaptation under a current step; a current- and conductance-based GLIF E/I network (including an
  NMDA synapse) producing a spike raster.

### Phase 2 — Nonlinear cell dynamics and timing
- Unlocks: nonlinear point cells (`izhikevich2007Cell`, `adExIaFCell`, `fitzHughNagumoCell`,
  `hindmarshRose`); realistic spike delays; on-device stimulus generators and within-population
  heterogeneity.
- Why it groups: each item breaks a simplifying assumption Phase 1 leaned on — nonlinear intrinsic
  dynamics break closed-form advancement, delays break instantaneous propagation — and on-device
  generation / heterogeneity round out point-neuron networks at scale.
- NML surface: nonlinear `TimeDerivative` RHS (`v²`, exponential terms); `connection`/`connectionWD`
  `delay`; `sineGenerator`/`rampGenerator`/`voltageClamp`, Poisson spike sources, `compoundInput`;
  per-instance `Property`/`Parameter` heterogeneity.
- Engine work:
  - **Active-set × nonlinear rule** (§0.5): nonlinear types integrate exactly one `dt` per active
    tick (no lazy skip) and re-enqueue while non-resting; forward Euler for the nonlinear RHS.
  - The **delay ring** subsystem (§4.4) + the active-set × delay interaction (pending-active list per
    future slot).
  - On-device generator functions with a per-thread RNG; parameterized (non-baked) cell code with
    per-neuron parameter arrays (§3.1).
- Exit models: an izhikevich network; a delayed-coupling network; a Poisson-driven population
  generated on device.

### Phase 3 — Biophysical and multicompartment (the full NML surface)
- Unlocks: Hodgkin-Huxley channels, ion concentrations, kinetic schemes, and multicompartment
  morphology — the complete feature set.
- Why it groups: these all require the same new machinery — composing sub-cell dynamics (gates, pools,
  channels) into cells, expanding state per compartment, intra-cell coupling, and stiff integration —
  and none of it is needed for anything in Phases 1–2.
- NML surface: D2 gates (`gateHH*`, `HH*Rate`), `ionChannelHH`/`ionChannel`/`ionChannelKS`,
  `biophysicalProperties`/`channelDensity`, D5 concentration models, `Child`/`Children` composition,
  `morphology`/`segment`/`segmentGroup`, `KineticScheme`, Q10/`temperature`.
- Engine work: compose channel/gate dynamics into biophysical cells (extra intrinsic state per gate /
  pool); multicompartment state expansion + intra-cell axial coupling (`parent_index` links);
  higher-order/implicit integrators for stiffness (RK4 / backward Euler); kinetic-scheme occupancy
  ODEs.
- Exit models: an HH multicompartment cell; a biophysical microcircuit — the full culmination.

---

## 6. Ticket mapping

Tickets #2/#3/#7/#8 and the first slice of #4/#5 are all **Phase 1** work (pipeline + full GLIF); the
rest of #4/#5 grow across Phases 2–3 as the feature surface widens (see §5 for the sequencing).

- **#2 Parser** — XML → tree; process `<include>`; emit the pre-lowering model. *(Phase 1.)*
- **#3 Std lib** — vendor `NeuroML2CoreTypes/`; merge at resolve. *(Phase 1.)*
- **#8 XSD validator** — structural gate before resolve. *(Phase 1.)*
- **#7 `ModelSpecification`** — the flat lowered tables (cell/synapse types with dynamics trees,
  populations + boundaries, adjacency + weights, inputs, outputs). *(Phase 1 minimal; extended per phase.)*
- **#5 Allocation** — widened state vector + boundaries (§4.1, **Phase 1**), aggregated synapse
  accumulators (§4.2, **Phase 1**), per-edge synapse state — shared basis + `Ck` + sparse `Sk` (§4.3,
  **Phase 1**), regime indices (§4.5,
  **Phase 1**); delay data (§4.4, **Phase 2**).
- **#4 Codegen** — per-Dynamics-ComponentType generation of the compute stages, constants baked,
  runtime-compiled on both backends. *(Linear cells + full synapse model — current/conductance/
  per-edge — in Phase 1; nonlinear cells + delays in Phase 2; biophysical in Phase 3.)*
- **#6 (reframed)** — turn cell-type dynamics into the runnable step: lower each ComponentType's IR
  `.tick` to GPU source and **assemble them into one master kernel** (per-neuron dispatch by cell-type
  boundary), compiled once and cached. Codegen stays generic — only per-type IR is emitted; the master
  kernel is an artifact of the assembly, not hand-authored (see `nml_ir_spec.md`).
- **New (not in current tickets)** — a spike-**delay** subsystem (§4.4, **Phase 2**); the
  **active-set × nonlinear-dynamics** correctness rule (§0.5, **Phase 2**) as an explicit codegen
  requirement.
