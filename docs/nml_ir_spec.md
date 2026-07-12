# spikecorec IR — Specification (v1.0 — locked for Phases 1–2)

**Status: LOCKED for Phases 1–2** (2026-07-12). The syntax, op set, and control model here are stable;
build against them. The one deferred item is the Phase-3 child/neighbor-set access construct (§6),
which is pinned down when the Phase-3 tickets are written. Any change to the locked surface goes
through an explicit spec revision. Companion to `nml_codegen_architecture.md` (referenced as "arch §N").

Governing principle: **IR → GPU source is near-direct template substitution.** Each leaf instruction is
a single operation that maps to one fixed MSL/CUDA snippet with its named operands filled in; each
control construct maps to one control template. The back-end does essentially no analysis between
reading the IR and emitting code — hence single-op leaves, plain-name operands, no expression grammar,
no register allocation in the IR.

---

## 1. What the IR is, and why

A small, backend-agnostic instruction set between the resolved NML model and the GPU. Codegen emits
**IR, one program per ComponentType** (each cell type, each synapse type) — so the front-end stays
generic (lower one type at a time; never reason about other types or MSL vs CUDA).

Two sections, different fates:
- **`.alloc`** — declares the model-specific things the engine must set up. **Interpreted at init**,
  not compiled.
- **`.tick`** — the per-tick compute, single-op instructions + structured control over named values,
  stage-tagged. **Lowered to GPU source** and assembled with every other type's `.tick` into **one
  master kernel** (`compile_kernel`), run each tick.

```
resolved ComponentType
   → IR { .alloc, .tick }
        .alloc → engine interprets at init → sizes/initializes model-specific buffers
        .tick  → template-substituted to MSL/CUDA → assembled → ONE master kernel → runs per tick
```

---

## 2. `.alloc` — engine-interpreted; model-specific shape only

The engine's base data structures are **implicit and untouched by the IR**: `network_inputs` (the
universal per-neuron input / delay ring), the cell-state base, `last_spiked`, `last_tick_updated`, the
active-set arrays, the k²-tree adjacency, and the shared per-edge `U`/`V` basis when needed — all
always present, engine-owned. `.alloc` never declares them.

`.alloc` specifies only:
1. **Constants** — `param <name> = <literal>` (baked into `.tick`); `param <name> : dyn <dtype>` for a
   per-neuron array (arch §3.1).
2. **Cell-state-vector dimensions** — each `state <name> : <dtype>` adds a per-neuron slot to the
   widened cell-state buffer (arch §4.1); summed across cell types, these give its shape.
3. **WeightMatrix per-edge config** — each `peredge <name>` tells the engine this synapse type carries
   a per-edge state variable, so the shared-basis WeightMatrix is initialized to hold that many
   per-edge variables (`Ck` + sparse `Sk` per variable, arch §4.3).

Supporting directives: `accum <name> : <dtype>` (per-neuron synapse accumulator feeding
`network_inputs`, arch §4.2); `regime <name>` (per-neuron regime index, arch §4.5); `expose <name>`
(recordable, arch §4.6); `require <name> from <scope>` (bind a quantity from another component; a read,
no allocation).

At init the engine reads every program's `.alloc`, sizes the widened cell-state vector, sets the
WeightMatrix per-edge count, creates accumulators/regime indices, and bakes constants into codegen.

---

## 3. `.tick` — single-op instructions + structured control, over named values

### 3.1 Operands — names only
Every operand is a name: model quantities (`state`, `param`/constant, `accum`, `peredge`, `regime`,
exposed, required); **intermediates** the lowering introduces, named `t0, t1, …` (single-assignment);
reserved engine reads `dt`, `tick`, `network_inputs`. No registers — those belong to the generated
MSL/CUDA and are the back-end's job.

### 3.2 Stage tags
`@deliver @integrate @detect @emit @reset @propagate @plasticity @record` (arch §2). Stages 1/6/8/9 are
engine-owned; a program supplies only the per-type hooks there. Model code is mainly
`@integrate @detect @emit @reset`.

### 3.3 Leaf instructions (each maps to one target snippet)
- **Arithmetic:** `add sub mul div` `dst,a,b` · `neg dst,a` · `fma dst,a,b,c` (`a*b+c`) · `mod dst,a,b`.
- **Math (LEMS surface):** `exp log dst,a` · `pow dst,a,b` · `sqrt abs floor ceil dst,a` ·
  `sin cos tan sinh cosh tanh dst,a` · `min max dst,a,b` · `expdecay dst,a,tau` (`a*exp(-dt/tau)`).
- **Random (per-thread RNG):** `rand dst` (uniform [0,1)) · `randn dst` (standard normal). *(Phase-2
  Poisson/stochastic sources.)*
- **Compare (→ boolean name):** `gt lt ge le eq ne` `dst,a,b`.
- **Boolean:** `and or dst,a,b` · `not dst,a`.
- **Move:** `mov dst,src`.
- **Edge access (universal):** `loadedge dst,<var>@<edge>` (read; reconstructs basis+`Sk` for a
  `peredge` var) · `accedge <var>@<edge>,<value>` (accumulate: `Sk[edge]+=value` for `peredge`; a
  per-neuron accumulate for `accum`).
- **Event:** `emit <port>`.
- **Regime:** `set_regime <name>,<value>`.

### 3.4 Control constructs (uniform; each maps to one control template)
- **`if <cond> { … } [elif <cond> { … }] … [else { … }]`** — the single branching construct. `<cond>`
  is a boolean-valued name (from a compare/boolean op). Covers thresholds, `Case` chains, regime
  dispatch (`eq is_r, regime, 1` then `if is_r { … }`), and any nested NML control.
- **`forall <set> { … }`** — iteration. `<set>` is `neuron_in` / `neuron_out` (the current neuron's
  in/out edges); inside, `@edge` is the current edge. Maps to a for-loop template over k²-tree
  neighbors. *(Generalizes to child/neighbor sets in Phase 3 — §6.)*
- **`onevent <port> { … }`** — spike-arrival handler (lives in `@deliver`); its body runs when a spike
  is delivered on `<port>`.

A single edge op may also take `neuron_in`/`neuron_out` directly as its edge to repeat across the set:
`accedge g@neuron_out, weight` (scatter).

### 3.5 `network_inputs` — the universal cell input (min 1-tick latency)
Every synapse type — current or conductance, aggregated or per-edge — computes its **finished total
current** with `.tick` instructions and writes it into `network_inputs`; the cell reads
`network_inputs` in `@integrate`. A conductance synapse computes its own `g·(erev − v)` in `.tick`
(reading the postsynaptic `v` via `require v`) before writing — so there is no special cell-side case.

A write to `network_inputs` reaches the reading cell **at least one tick later** (minimum delay = 1):
contributions are scheduled into a future deliver slot (`now + delay`, `delay ≥ 1`) and delivered at the
top of that tick — the generalization of the engine's existing one-tick latency and the delay ring
(arch §4.4).

---

## 4. Examples (v4 syntax, provisional)

**GLIF1 — leaky integrate-and-fire:**
```
.alloc
  state v : f32
  param C, gL, EL, vth, vreset
.tick
  @integrate
    sub t0, EL, v                  ; t0 = EL - v
    mul t0, gL, t0                 ; t0 = gL*(EL - v)
    add t0, network_inputs, t0     ; t0 = I + gL*(EL - v)
    div t0, t0, C                  ; t0 = dv/dt
    mul t0, t0, dt
    add v, v, t0                   ; v += dt*dv/dt
  @detect
    gt spiked, v, vth
  @emit
    if spiked { emit spike }
  @reset
    if spiked { mov v, vreset }
```

**expOne — current-based, aggregatable synapse:**
```
.alloc
  accum g : f32
  param tau, weight
.tick
  @deliver
    onevent in { accedge g@edge, weight }        ; aggregated → g[target] += weight
  @integrate
    expdecay g, g, tau                            ; dg/dt = -g/tau
    add network_inputs, network_inputs, g         ; contribute current to the cell
```

**NMDA — conductance, per-edge synapse (Mg-block elided):**
```
.alloc
  peredge g
  require v from postsynaptic
  param tau, weight, erev
.tick
  @deliver
    onevent in { accedge g@edge, weight }         ; Sk[edge] += weight
  @integrate
    forall neuron_in {
      loadedge t0, g@edge                          ; per-edge g (basis + Sk)
      sub t1, erev, v                              ; erev - v   (v via require)
      mul t0, t0, t1                               ; g*(erev - v)   [× Mg-block, elided]
      add network_inputs, network_inputs, t0       ; finished current into the cell
    }
```

**Refractory regime (iafRefCell-style, sketch) — shows uniform control on a state machine:**
```
.alloc
  state v : f32
  regime r                                         ; 0 integrating, 1 refractory
  param ..., vth, vreset, t_ref
.tick
  @integrate
    eq is_ref, r, 1
    if is_ref { ... hold v, count down ... } else { ... normal v integration ... }
  @detect
    eq is_int, r, 0
    gt over, v, vth
    and fire, is_int, over
  @emit
    if fire { emit spike }
  @reset
    if fire { mov v, vreset  set_regime r, 1 }      ; enter refractory
```

---

## 5. Assembly into the master kernel

One master kernel for the whole step, assembled from the per-type `.tick` programs:
- one thread per neuron, dispatching by **cell-type boundary** (arch §4.1) to its type's
  `@integrate/@detect/@emit/@reset` code;
- **engine-fixed stages** (deliver-drain, k²-tree propagate/scatter, active-set enqueue, record) are the
  shared scaffold the generated code splices into;
- **composition is through `network_inputs`** with a ≥1-tick delay (§3.5): synapse programs write it,
  cells read it a later tick — so no cell embeds synapse internals and there is no same-tick ordering
  constraint between them.

Codegen stays generic (only per-type IR is emitted); the master kernel is an artifact of the
IR→source assembly, not authored by hand.

---

## 6. Op-set completeness and remaining questions

**Completeness target.** The op set is sized so that every dynamics expressible in NML/LEMS across all
three phases is describable:
- LEMS math-function surface — `exp/log/pow/sqrt/abs/floor/ceil/mod`, `sin/cos/tan/sinh/cosh/tanh`,
  `min/max` — covers HH rate laws, concentration models, arbitrary `TimeDerivative`/`DerivedVariable`
  right-hand sides (Phases 1–3).
- Stochastic sources — `rand`/`randn` — cover Poisson/`SpikeSourcePoisson`/`poissonFiringSynapse`
  (Phase 2).
- Conditions — full compare set + `and/or/not` + the uniform `if/elif/else` — express thresholds,
  `ConditionalDerivedVariable`/`Case`, and regime/`KineticScheme` conditionals.
- Structure — `emit`/`onevent`, `loadedge`/`accedge`, `forall neuron_in/neuron_out`, `set_regime` —
  cover event emission/arrival, per-edge and aggregated synapse state, edge iteration, and state
  machines.

**Known expansion point (Phase 3).** Multicompartment axial coupling needs a cell to read a *neighbor
compartment's* state, and channels/kinetic schemes need iteration over *child sets* (gates in a
channel, states in a scheme). Both generalize `neuron_in`/`neuron_out` + `forall` to child/neighbor
sets; the exact construct (and any accompanying access op) is deferred to the Phase-3 tickets rather
than guessed now. Nonlinear cells, delays, and on-device generators (Phase 2) need **no** new ops —
just more of the existing arithmetic/random ops.

**Still open (minor):**
- The concrete Phase-3 child/neighbor construct (above).
- Whether `expdecay` (and any other macro-ish ops) stay first-class or become a back-end-recognized
  pattern over `exp`/`mul`.
- Exact `onevent` body scoping and how `@edge` binds inside it.
