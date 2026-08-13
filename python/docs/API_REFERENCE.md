# `spikecorec` Python API reference

A parameter-by-parameter reference for everything `spikecorec` exposes to Python. For the
walkthrough and the explanations of *why* things are shaped this way, see
[`README.md`](README.md) — this document is the lookup table.

Every signature below reflects the actual pybind11 bindings
(`src/bindings/bindings.cpp`); argument names are keyword-usable
(`engine.run(tick_count=100)` works).

## Contents

- [The shape of the API](#the-shape-of-the-api)
- [Module-level functions](#module-level-functions)
- [`SpikeEngine`](#spikeengine)
- [`WeightMatrix`](#weightmatrix)
- [`SimulationRecorder`](#simulationrecorder)
- [`WeightStats`](#weightstats)
- [What is deliberately not bound](#what-is-deliberately-not-bound)

---

## The shape of the API

The engine is driven by a **model file**, not by Python. A NeuroML/LEMS document states the
cell dynamics, the network wiring, the stimulus, the timestep, the run length and the
recordings; `SpikeEngine` parses it, allocates and compiles everything it implies, and then
does exactly one thing per call — advance every neuron by one `dt`.

```python
engine = spikecorec.SpikeEngine("model.xml")
engine.run(engine.lifetime)
engine.shutdown()
frames = spikecorec.read_spire_recording(engine.recording_output_filenames()[0])
```

Every quantity that crosses this boundary is **SI**. A `C="100pF"` reads back as `1e-10`, a
membrane potential as `-0.07` (volts), a synaptic current as `6e-10` (amperes).

---

## Module-level functions

### `set_log_level(level) -> None`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `level` | `str` | required | `"trace"`, `"debug"`, `"info"`, `"warn"`, `"err"`, `"critical"` or `"off"` |

Sets the engine's own console logging level. The logger is a process-wide singleton, so this
affects every engine in the process, including ones already constructed.

Worth leaving at `"warn"` rather than `"off"`: several things the engine can only warn about
are things you want to see. A recording that selects a variable its cell type does not
declare, or an `enable_hebbian_learning=True` that applies no plasticity, are both warnings
and neither is otherwise visible.

---

### `read_spire_recording(filename) -> np.ndarray[float32]`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | `str` | required | Path to a `.spire` / `.spire.gz` / `.spire.xz` / `.spire.bz2` file |

**Returns:** `np.ndarray[float32]` of shape `(frame_count, column_count)` — one frame per
tick, in tick order.

Compression is inferred from the extension. `.gz`/`.xz`/`.bz2` decode only if the
corresponding library was available when the extension was built; otherwise this raises.

The columns are whatever the recording's `<OutputFile>` selected, in selection order. An
`<OutputFile>` or `<EventOutputFile>` with no child selections records **every neuron** —
its first state variable for a value file, its spike flag for an event file.

Read this **after** `SpikeEngine.shutdown()`. Frames are buffered and only flushed when the
recorder is finished, so reading a live recording comes up short by up to a chunk.

---

## `SpikeEngine`

### Constructor

```python
SpikeEngine(model_path, enable_hebbian_learning=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` | required | NeuroML or LEMS model file. A `<neuroml>` root is XSD-validated first; a `<Lems>` root is not (the NeuroML schema does not describe one) |
| `enable_hebbian_learning` | `bool` | `False` | Allocates the `last_tick_updated` buffer **and nothing else** — see below |

Parses and validates the model, sizes and allocates every model buffer, runs the `OnStart`
bodies, compiles the generated master kernel, and opens one recorder per `<OutputFile>` /
`<EventOutputFile>` the model declares.

Raises `ValueError`/`RuntimeError` with the reason on a model that does not parse, does not
validate, or asks for something the codegen refuses (a conductance-based synapse, for
instance).

> **`enable_hebbian_learning` currently does nothing to the simulation.** It allocates
> `last_tick_updated`; no kernel reads that buffer and no weight update runs, so the resulting
> run is bit-identical to one built with `False`. The engine logs a warning saying so at
> construction. Do not read a spike-timing-dependent result out of a run built with it.

> **The model file must state a `<Simulation>`.** `step_dt` and `lifetime` come from it. A
> model without one currently only *warns* ("No Simulation instance found"), leaves `step_dt`
> at `0.0`, and then integrates nothing while reporting successful ticks. Check
> `engine.step_dt` after construction if the model is not one you wrote.

---

### Methods

#### `step_simulation(tick) -> None`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tick` | `int` | required | The tick index being advanced |

Advances every neuron by exactly one `dt`: delivers tick `tick`'s external stimulus into the
delay ring, runs the generated master kernel plus the ring clear behind it, and writes one
frame to every recorder.

> **`tick` is the caller's, and the engine does not check it.** It indexes the precomputed
> stimulus streams *and* selects the delay-ring row (`tick % network_input_ring_depth`), so a
> repeated, skipped or out-of-order tick produces a wrong simulation rather than an error.
> Step from `0` upwards by one, or use `run()`.

A negative `tick` raises `ValueError`. That one *is* checked, because the ring row is resolved
in signed arithmetic and a negative tick indexes the buffer out of bounds on the device rather
than merely giving a wrong answer. The check lives in the binding, not in the engine.

---

#### `run(tick_count, first_tick=0) -> None`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tick_count` | `int` | required | How many ticks to advance |
| `first_tick` | `int` | `0` | Tick index of the first step; must be non-negative |

`step_simulation` over `first_tick .. first_tick + tick_count - 1`, keeping the loop in C++
and releasing the GIL for its duration.

Note that `spike_flags()` only ever describes the **most recent** tick, so a `run()` loop
cannot see the emissions of any tick but the last. Drive `step_simulation` yourself when you
need per-tick emissions, or record them with an `<EventOutputFile>`.

---

#### `shutdown() -> None`

Finishes every recorder — which is what flushes the last buffered `.spire` frames to disk —
and releases the compiled kernels. Idempotent, and the destructor calls it if you did not.
`alive` becomes `False`.

The engine's GPU buffers are *not* freed here: they are sub-ranges of an arena the engine
owns, released when the engine object itself is collected.

---

#### `cell_type_names() -> list[str]`

**Returns:** cell type names, in the order their indices refer to (so
`cell_type_names()[cell_type_index()[n]]` is neuron `n`'s type).

---

#### `state_variable_names(cell_type_index) -> list[str]`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cell_type_index` | `int` | required | Index into `cell_type_names()` |

**Returns:** that type's `<StateVariable>` names, in the order they occupy its state chunk.

Raises `IndexError` for an index naming no type.

---

#### `parameter_names(cell_type_index) -> list[str]`

As above, for the type's `<Parameter>` names.

---

#### `recording_output_filenames() -> list[str]`

**Returns:** every file the model's `<OutputFile>`/`<EventOutputFile>` declarations write, in
the order the recorders were opened. Readable with `read_spire_recording()` after
`shutdown()`.

---

### Read-back accessors

Every one of these **copies** into a fresh numpy array. No GPU handle crosses the language
boundary: the engine's buffers are arena sub-ranges that must never be freed by anything but
the arena, and there is no sound cross-runtime ownership story for one.

A buffer this model sized at zero elements reads back as an empty array rather than raising.

#### `state_variable_values(name) -> np.ndarray[float32]`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | A `<StateVariable>` name |

**Returns:** that variable for every neuron, by **global neuron index**, in SI —
`state_variable_values("v")` is the membrane potential in volts.

`cell_state` is sectioned by cell *type*, so this gathers rather than slicing: the variable's
slot is resolved per neuron from that neuron's own type. Two cell types may well put `"v"` in
different slots.

**Raises** `RuntimeError`, naming the neuron and its type, if any neuron's cell type does not
declare `name`. (The engine's own recording path falls back to that neuron's *first* state
variable in the same situation, with only a warning. This accessor deliberately does not.)

---

#### `parameter_values(name) -> np.ndarray[float32]`

The same, for a `<Parameter>`. In SI: a `C="100pF"` reads back as `1e-10`.

---

#### `spike_flags() -> np.ndarray[int32]`

`1` for every neuron that emitted on the tick just stepped, `0` otherwise. Length
`total_neuron_count`.

The master kernel lowers every flag at the top of each tick, so this describes **only** the
most recent `step_simulation` call.

---

#### `last_spiked() -> np.ndarray[int64]`

Tick each neuron last emitted on. Length `total_neuron_count`.

> **Initialised to `0`, not to `-1`.** A value of `0` means either "fired on tick 0" or "has
> never fired", and nothing in this array distinguishes them. Accumulate `spike_flags()`, or
> record an `<EventOutputFile>`, if you need to know whether a neuron ever fired at all.

---

#### `cell_state() -> np.ndarray[float32]`

The whole cell-state buffer, flat, length `cell_state_element_count`. Sectioned by cell
**type** — every neuron of type 0, then every neuron of type 1 — not by neuron index. Neuron
`n`'s chunk starts at `cell_state_base()[n]`. A type that declares any `<Regime>` carries one
extra slot after its state variables, holding its current regime index.

Prefer `state_variable_values()` unless you specifically want the raw layout.

---

#### `cell_parameters() -> np.ndarray[float32]`

The whole cell-parameter buffer, flat, length `cell_parameter_element_count`, sectioned the
same way. Neuron `n`'s chunk starts at `cell_parameter_base()[n]`.

---

#### `cell_state_base() -> np.ndarray[int32]`

Where each neuron's own chunk starts inside `cell_state()`.

#### `cell_parameter_base() -> np.ndarray[int32]`

Where each neuron's own chunk starts inside `cell_parameters()`.

#### `cell_type_index() -> np.ndarray[int32]`

Each neuron's cell type index, into `cell_type_names()`.

---

#### `synapse_state() -> np.ndarray[float32]`

Per-(target neuron, wired synapse prototype) synapse state, flat, length
`synapse_state_element_count`. Wired prototype `p`'s slice starts at (the preceding
prototypes' total state width) × `total_neuron_count`.

---

#### `network_inputs() -> np.ndarray[float32]`

The synaptic delay ring, flat. Reshape to
`(network_input_ring_depth, network_input_plane_count, total_neuron_count)`:

- row `tick % network_input_ring_depth` is what tick `tick` reads, and is cleared behind it;
- plane `0` is delivered current in **amperes** — where external stimulus lands, where each
  synapse's output is added, and where an edge naming no synapse scatters its raw weight;
- plane `1 + p` holds arrivals awaiting wired synapse prototype `p`.

---

### Properties

| Name | Type | Description |
|---|---|---|
| `total_neuron_count` | `int` | Neurons in the model, summed over every population |
| `input_neuron_count` | `int` | Number of wired `(input, target)` stimulus **streams** — two inputs onto one neuron count twice, so this is not a count of driven neurons |
| `lifetime` | `int` | Ticks the model's `<Simulation length/step>` works out to |
| `step_dt` | `float` | Seconds per tick, SI |
| `simulation_duration` | `float` | Seconds the model's `<Simulation length>` resolved to |
| `alive` | `bool` | `False` after `shutdown()` |
| `hebbian_learning_enabled` | `bool` | What was passed to the constructor. Applies no plasticity — see the constructor note |
| `simulation_seed` | `int` | The model's `<Simulation seed>`, or `0` |
| `network_input_ring_depth` | `int` | Rows in the delay ring; strictly greater than the model's largest per-edge delay |
| `network_input_plane_count` | `int` | Planes per ring row: 1 + one per wired synapse prototype |
| `cell_state_element_count` | `int` | Length of `cell_state()` |
| `cell_parameter_element_count` | `int` | Length of `cell_parameters()` |
| `synapse_state_element_count` | `int` | Length of `synapse_state()` |
| `weights` | `WeightMatrix` | The wired adjacency (read-only, borrowed from the engine) |

---

## `WeightMatrix`

Reachable only through `SpikeEngine.weights`. The wiring comes from the model, so there is
nothing here to construct or mutate from Python; it is bound so you can ask **what the engine
actually wired**. A model whose connection paths did not resolve produces a perfectly
well-formed, fully disconnected network that runs and looks merely quiet, and this is the only
thing that tells those apart.

Weights are stored **exactly** per edge (`using_exact_edge_weights`), not as a low-rank `U·V`
reconstruction. A NeuroML weight is routinely `1e-9` or smaller in SI, and expressing that as
a delta against an order-1 `U·V` reconstruction rounds to zero in float32.

### Properties

| Name | Type | Description |
|---|---|---|
| `node_count` | `int` | Nodes in the adjacency (= `total_neuron_count`) |
| `max_neighbor_count` | `int` | Upper bound on outgoing neighbours per node |
| `rank` | `int` | Latent factor dimensionality of the `U·V` basis — a **memory-compression** parameter, nothing to do with learning |
| `using_exact_edge_weights` | `bool` | `True` once any edge weight has been set exactly, which the engine constructor always does |

### Methods

#### `get(source_node, target_node) -> float`

Weight of the edge `source_node -> target_node`. Returns `0.0` for a pair that is not an edge.

#### `get_edge_delay_ticks(source_node, target_node) -> int`

That edge's conduction delay in whole ticks. Always at least `1`: the delay ring requires it,
so a `delay="0ms"` in the model reads back as `1` here.

#### `get_neighbors(node_index) -> list[int]`

That node's outgoing neighbours, in k²-tree traversal order — which is the order the generated
kernel walks them in.

#### `neighbor_weight_stats() -> WeightStats`

mean / standard deviation / RMS / min / max over every edge weight.

---

## `SimulationRecorder`

The buffering/compression layer the engine's own recorders are built on, exposed for callers
who want to write a `.spire` file from values they gathered themselves between
`step_simulation` calls, rather than from an `<OutputFile>` the model declares.

### Constructor

```python
SimulationRecorder(filename, value_count, compression="auto", compression_level=None,
                   compression_async=False, queue_max=8, chunk_bytes=4194304)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | `str` | required | Output path |
| `value_count` | `int` | required | Values per frame, fixed for the file's lifetime |
| `compression` | `str \| None` | `"auto"` | `"auto"`/`None` infers from the extension; `"none"`, `"gzip"`, `"xz"`, `"bz2"` are explicit |
| `compression_level` | `int \| None` | `None` | Native level for the chosen codec (1–9 gzip/bz2, 0–9 xz); `None` is 6 |
| `compression_async` | `bool` | `False` | Compress on a background thread. Named this, not `async`, because `async` is a Python keyword |
| `queue_max` | `int` | `8` | Bounded queue depth when `compression_async=True`; `record_frame` blocks when full |
| `chunk_bytes` | `int` | `4194304` | Buffer size before a flush |

### Properties

| Name | Type | Description |
|---|---|---|
| `value_count` | `int` | Values per frame |

### Methods

#### `record_frame(values) -> None`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `values` | array-like of `float32` | required | Exactly `value_count` values |

Appends one frame. A wrongly-sized array is rejected rather than read out of bounds.

#### `finish() -> None`

Flushes and closes. Safe to call more than once; the destructor calls it if you did not.

---

## `WeightStats`

Returned by `WeightMatrix.neighbor_weight_stats()`.

| Name | Type | Description |
|---|---|---|
| `mean` | `float` | Mean edge weight |
| `standard_deviation` | `float` | Standard deviation |
| `root_mean_square` | `float` | RMS |
| `min_value` | `float` | Minimum |
| `max_value` | `float` | Maximum |

---

## What is deliberately not bound

Named here because their absence is a decision, not an oversight.

**The pre-NeuroML engine surface.** `setup_lifetime`, `set_input_neurons`, `reset_state`, the
adjacency-list constructor, `is_alive()`, `estimate_bifurcation_weight`, the
`scale_*_weights_near_bifurcation` pair, `get_reservoir_features_vector`,
`start_static_record`, `get_membrane_potentials`, the active-set accessors and the
`resting_membrane_potential`/`decay_rate`/`learning_rate`/`spike_threshold`/`spike_period`
properties are all commented out in `include/spikecorec/core/engine.h` as legacy pending
rework. Binding names whose implementations no longer exist is what broke `make python`
through the engine rewrite.

**The topology generators** (`square_torus`, `small_world_torus`,
`random_fixed_outdegree`). They return an adjacency list, and the only thing that ever
consumed one from Python was the legacy `SpikeEngine` constructor. A model's wiring comes from
its `<projection>`s now, so there is nothing on this surface to hand one to.

**`SpikeEngine.use_constant_weight`.** It is live in C++ — when set, the engine forwards
`weights.constant_weight` to the kernel for every edge — but it is a *second* constant-weight
flag, independent of `WeightMatrix.using_constant_weight`, which is what the host-side
`WeightMatrix.get()` consults. Setting one from Python without the other makes the GPU run on
a weight the read-back does not report: a wrong-numbers bug with nothing to see. It also has
no NeuroML-path meaning, since a model states its own weights.

**`ScaleResult` / `ScaledReservoirResult`.** Return types of the bifurcation-scaling methods
above; nothing bound returns one.
