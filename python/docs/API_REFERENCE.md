# `spikecorec` Python API reference

A scannable, parameter-by-parameter reference for everything `spikecorec`
exposes to Python. For tutorials, a quick-start walkthrough, and explanations
of *why* things work the way they do, see [`../README.md`](../README.md) —
this document is the lookup table.

All signatures below reflect the actual pybind11 bindings
(`src/bindings/bindings.cpp`); argument names are keyword-usable
(`engine.step_simulation(input_values=..., tick=...)` works).

## Contents

- [Module-level functions](#module-level-functions)
- [`SpikeEngine`](#spikeengine)
- [`WeightMatrix`](#weightmatrix)
- [`SimulationRecorder`](#simulationrecorder)
- [`NmlNetworkRunner`](#nmlnetworkrunner)
- [Result types](#result-types)

---

## Module-level functions

### `square_torus(k) -> list[list[int]]`

4-neighbor wraparound `k * k` grid. Every node has exactly 4 outgoing
neighbors (right, left, down, up). Deterministic — no seed.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `k` | `int` | required | Grid side length; produces `k * k` nodes indexed `0 .. k*k - 1` |

**Returns:** `list[list[int]]` — adjacency list, `result[i]` is node `i`'s outgoing neighbor indices.

---

### `small_world_torus(k, random_fanout=4, seed=-1) -> list[list[int]]`

`square_torus` plus `random_fanout` deterministic long-range shortcuts per
node, so activity can mix across the reservoir rather than only drifting
locally.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `k` | `int` | required | Grid side length; produces `k * k` nodes |
| `random_fanout` | `int` | `4` | Extra long-range edges per node. Every node ends up with out-degree `4 + random_fanout` (capped by reservoir size) |
| `seed` | `int` | `-1` | RNG seed. `< 0` seeds from `std::random_device` (non-deterministic) |

**Returns:** `list[list[int]]` — adjacency list. Structurally equivalent to
`spikecore/topologies.py`'s NumPy/PCG64 reference, but **not bit-identical**
(different RNG stream).

---

### `random_fixed_outdegree(k, fanout=8, seed=-1) -> list[list[int]]`

Directed random graph — no torus/grid structure at all. Every node has the
same out-degree `fanout`, with no self-loops and no duplicate edges.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `k` | `int` | required | Produces `k * k` nodes |
| `fanout` | `int` | `8` | Out-degree of every node |
| `seed` | `int` | `-1` | RNG seed. `< 0` seeds from `std::random_device` (non-deterministic) |

**Returns:** `list[list[int]]` — adjacency list. Structurally equivalent to
`spikecore/topologies.py`'s NumPy/PCG64 reference, but **not bit-identical**.

---

### `read_spire_recording(filename) -> np.ndarray[float32]`

Decodes a `.spire` recording (written by `start_static_record` /
`SimulationRecorder`, or by the `spikecore` Python reference — the format is
byte-for-byte interoperable). Auto-detects `.gz`/`.xz`/`.bz2` compression from
the filename extension.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | `str` | required | Path to a `.spire`, `.spire.gz`, `.spire.xz`, or `.spire.bz2` file |

**Returns:** `np.ndarray[float32]` of shape `(frame_count, neuron_count)`.

**Raises:** `RuntimeError` if the final frame is truncated, or if the file
was compressed with a codec `spikecorec` wasn't built with support for.

---

## `SpikeEngine`

### Constructor

```python
SpikeEngine(network, shape, rank=1, resting_mp=0.1, decay_rate=0.01, learning_rate=0.00222)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `network` | `list[list[int]]` | required | Adjacency list (e.g. from a topology generator). Copied into a k²-tree-backed `WeightMatrix` at construction time |
| `shape` | `list[int]` | required | `[rows, cols]`; `rows * cols` (= `neuron_count`) must equal `len(network)` |
| `rank` | `int` | `1` | Latent-factor dimensionality for the low-rank `U·V` weight approximation. `-1` → `min(64, neuron_count)` |
| `resting_mp` | `float` | `0.1` | Resting membrane potential — initial value of `resting_membrane_potential` |
| `decay_rate` | `float` | `0.01` | Exponential decay rate toward `resting_mp` |
| `learning_rate` | `float` | `0.00222` | Hebbian weight-update step size |

> Other simulation constants not settable via the constructor —
> `spike_period` (default `1`) and `spike_threshold` (default `1.0`) — can be
> set afterward as [properties](#properties).

---

### Methods

#### `set_input_neurons(input_neuron_list) -> None`

Designates which neurons receive external input each tick.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_neuron_list` | `list[int]` | required | Neuron indices. `step_simulation`'s `input_values[i]` is added to the membrane potential of `input_neuron_list[i]` — the two lists are matched positionally and must be the same length |

Sets `engine.input_neuron_count = len(input_neuron_list)`.

---

#### `step_simulation(input_values, tick, override_input_neurons=[], decay_all_neurons=False) -> None`

Advances the simulation by one tick.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_values` | `list[float]` | required | Values added to the membrane potentials of the neurons configured via `set_input_neurons`, matched positionally. Must be the same length as the input-neuron list |
| `tick` | `int` | required | Current tick index (used for spike-timing bookkeeping) |
| `override_input_neurons` | `list[int]` | `[]` | Extra neuron indices merged into this tick's active set — forces those neurons to be processed even if they weren't already active (e.g. injecting activity at arbitrary locations) |
| `decay_all_neurons` | `bool` | `False` | If `True`, decays *every* neuron toward `resting_mp` based on elapsed time since its last update before the step runs |

Order of operations each call:
1. (optional) decay-all pass if `decay_all_neurons=True`
2. add `input_values` into the configured input neurons' membrane potentials
3. merge `override_input_neurons` into the active set
4. propagate spikes through the network for one tick, applying Hebbian weight updates along active edges

---

#### `reset_state(last_spiked_value=0, active_gen_value=-1) -> None`

Returns the engine to its initial state: membrane potentials reset to
`resting_mp`, spike/update/active-set bookkeeping cleared, network input
accumulator zeroed.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `last_spiked_value` | `int` | `0` | Value every entry of `last_spiked` is reset to |
| `active_gen_value` | `int` | `-1` | Value every entry of the active-generation tags is reset to |

---

#### `setup_lifetime(lifetime, allocate_logs, max_log_bytes=536870912) -> None`

Allocates host-side per-neuron membrane-potential log buffers
(`neuron_count * lifetime` floats) for recording over a fixed-length run.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lifetime` | `int` | required | Number of ticks to allocate log space for |
| `allocate_logs` | `bool` | required | Whether to actually allocate the log buffers |
| `max_log_bytes` | `int` | `512 * 1024 * 1024` (512 MiB) | Upper bound on the allocation size |

**Raises:** `RuntimeError` if `neuron_count * lifetime * sizeof(float32)` would exceed `max_log_bytes`.

---

#### `is_alive() -> bool`

Returns whether the engine's GPU resources are still allocated (i.e.
`shutdown()` has not yet run).

---

#### `shutdown() -> None`

Frees the engine's GPU buffers. Safe to call multiple times. Also called
automatically when the `SpikeEngine` is garbage collected. After this,
`running` is `False` and `is_alive()` returns `False`.

---

#### `estimate_bifurcation_weight(input_period=1) -> tuple[float, float]`

Estimates the edge weights that would put a neuron with the engine's current
`resting_membrane_potential`, `spike_threshold`, and `decay_rate` right at its
firing threshold under periodic stimulation.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_period` | `int` | `1` | Period (in ticks) of the periodic input stimulus |

**Returns:** `(w_accum, w_instant)` — both `float`:
- `w_accum` — weight estimate assuming input accumulates over the period
- `w_instant` — weight estimate assuming an instantaneous (single-tick) input

---

#### `scale_uniform_weights_near_bifurcation(input_period=1, scale=1.2, freeze_learning=False) -> tuple[float, float, float]`

Sets **every** edge to the same constant weight, computed near the
bifurcation point.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_period` | `int` | `1` | Period (in ticks) of the periodic input stimulus, passed to `estimate_bifurcation_weight` |
| `scale` | `float` | `1.2` | Multiplier applied to `w_accum` to get the target weight |
| `freeze_learning` | `bool` | `False` | If `True`, sets `learning_rate = 0` afterward — see [Disabling Hebbian learning](#disabling-hebbian-learning) |

**Returns:** `(target, w_accum, w_instant)` — all `float`. `target = w_accum * scale` is the constant weight applied to every edge; `weights.using_constant_weight` becomes `True` and `weights.constant_weight == target`.

---

#### `scale_randomized_weights_near_bifurcation(input_period=1, scale=1.2, freeze_learning=False) -> ScaledReservoirResult`

Instead of collapsing to a constant, rescales the existing (randomized) edge
weights so their RMS matches a target near the bifurcation point.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_period` | `int` | `1` | Period (in ticks) of the periodic input stimulus |
| `scale` | `float` | `1.2` | Multiplier applied to `w_accum` to compute the target RMS (`target = abs(w_accum * scale)`) |
| `freeze_learning` | `bool` | `False` | If `True`, sets `learning_rate = 0` afterward — see [Disabling Hebbian learning](#disabling-hebbian-learning) |

**Returns:** [`ScaledReservoirResult`](#scaledreservoirresult).

---

#### `get_reservoir_features_vector(tick, spike_tau, voltage_scale) -> np.ndarray[float32]`

Builds a feature vector suitable for a downstream readout layer.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tick` | `int` | required | Current tick index |
| `spike_tau` | `float` | required | Decay constant for the per-neuron spike trace. Must be `> 0`, else the call is a no-op |
| `voltage_scale` | `float` | required | Scale factor for the membrane-potential features. Must be `> 0`, else the call is a no-op |

**Returns:** `np.ndarray[float32]` of shape `(2 * neuron_count + 1,)`:
- `[0 : neuron_count]` — per-neuron spike trace, `exp(-(tick - last_spiked) / spike_tau)` (`0` if the neuron has never fired)
- `[neuron_count : 2*neuron_count]` — per-neuron scaled membrane potential, `(membrane_potential - resting_mp) / voltage_scale`
- `[-1]` — bias term, always `1.0`

---

#### `start_static_record(input_spikes, lifetime, filename, record_membrane=True, record_stride=1, compression="auto", compression_level=None, full_decay=True, compression_async=False, compression_queue_max=8, compression_chunk_bytes=4194304) -> None`

Drives its own tick loop for `lifetime` ticks and writes a `.spire` recording
to `filename`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_spikes` | `list[list[float]]` | required | `input_spikes[tick][i]` is added to the membrane potential of `input_neuron_indices[i]` (set via `set_input_neurons`) for that tick. Must provide at least `lifetime` rows; each row must have exactly `input_neuron_count` values, or `RuntimeError` is raised before any file is written |
| `lifetime` | `int` | required | Number of ticks to simulate and (optionally) record |
| `filename` | `str` | required | Output path for the `.spire` recording |
| `record_membrane` | `bool` | `True` | Whether to snapshot membrane potentials into the recording |
| `record_stride` | `int` | `1` | Record a frame every `record_stride` ticks |
| `compression` | `str \| None` | `"auto"` | `"auto"` (infer from `filename` extension), `"none"`, `"gzip"`, `"xz"`, or `"bz2"` |
| `compression_level` | `int \| None` | `None` | Codec-native level/preset (1-9 for gzip/bz2, 0-9 for xz). `None` → default `6` |
| `full_decay` | `bool` | `True` | On each *recorded* tick, runs a full decay pass (advancing `last_tick_updated` for every neuron) immediately before the membrane-potential snapshot. This is a real, stateful operation |
| `compression_async` | `bool` | `False` | If `True`, compression and file I/O run on a background thread fed by a bounded queue so the simulation loop isn't blocked on I/O. This call releases the GIL while it runs |
| `compression_queue_max` | `int` | `8` | Max queued chunks when `compression_async=True`. `0` → unbounded (no backpressure) |
| `compression_chunk_bytes` | `int` | `4 * 1024 * 1024` (4 MiB) | Bytes per queued chunk when `compression_async=True` |

Each tick, the configured input neurons are forced into the active set
(mirrors the reference's per-tick `_add_active`) before the step runs.

See [`.spire` format details](../README.md#format) in the README.

---

### Read-back accessors

All copy GPU buffer contents into fresh numpy arrays — no GPU memory is
exposed directly to Python.

| Method | Returns | Description |
|---|---|---|
| `get_membrane_potentials()` | `np.ndarray[float32]`, shape `(neuron_count,)` | Current membrane potential per neuron |
| `get_network_inputs()` | `np.ndarray[float32]`, shape `(neuron_count,)` | Pending external-input accumulator per neuron |
| `get_last_spiked()` | `np.ndarray[int64]`, shape `(neuron_count,)` | Tick each neuron last fired |
| `get_last_tick_updated()` | `np.ndarray[int64]`, shape `(neuron_count,)` | Tick each neuron was last processed |
| `get_active_neuron_indices()` | `np.ndarray[int32]`, shape `(active_neuron_count,)` | Indices of neurons in the current active set |
| `get_active_neuron_count()` | `int` | Size of the current active set |

---

### Properties

| Name | Type | Access | Default | Description |
|---|---|---|---|---|
| `neuron_count` | `int` | read-only | — | Total neurons (`shape[0] * shape[1]`) |
| `input_neuron_count` | `int` | read-only | `0` | Length of the list passed to `set_input_neurons` |
| `resting_membrane_potential` | `float` | read/write | `0.1` (or constructor's `resting_mp`) | Membrane potential neurons decay toward |
| `decay_rate` | `float` | read/write | `0.01` (or constructor's `decay_rate`) | Exponential decay rate toward rest |
| `learning_rate` | `float` | read/write | `0.00222` (or constructor's `learning_rate`) | Hebbian update step size. Set to `0` to disable Hebbian learning entirely — see [Disabling Hebbian learning](#disabling-hebbian-learning) |
| `spike_period` | `int` | read/write | `1` | Refractory ticks after a spike before a neuron can fire again |
| `spike_threshold` | `float` | read/write | `1.0` | Membrane potential at which a neuron spikes |
| `use_constant_weight` | `bool` | read/write | `False` | When `True`, every edge uses `weights.constant_weight` instead of the learned `U·V` factorization |
| `running` | `bool` | read-only | `True` until `shutdown()` | `False` once `shutdown()` has run |
| `weights` | [`WeightMatrix`](#weightmatrix) | read-only | — | The engine's weight matrix; lifetime tied to the engine |

### Disabling Hebbian learning

```python
engine.learning_rate = 0.0
```

Setting `learning_rate` to `0` turns off the per-tick Hebbian `U`/`V` weight
updates entirely — the GPU kernels scale every update by `learning_rate`, so
`0` means no change is applied (and the synaptic-decay term is also gated on
`learning_rate != 0`). This can be toggled at any time and re-enabled later
by setting `learning_rate` back to a nonzero value.

If you're also scaling weights toward the bifurcation point,
`scale_uniform_weights_near_bifurcation(..., freeze_learning=True)` and
`scale_randomized_weights_near_bifurcation(..., freeze_learning=True)` zero
`learning_rate` for you as part of that call.

---

## `WeightMatrix`

Accessible read-only via `engine.weights`. Wraps a k²-tree-compressed sparse
adjacency structure plus a low-rank `U·V` weight factorization. There is no
standalone constructor exposed to Python — instances only come from
`SpikeEngine.weights`.

### Properties

| Name | Type | Description |
|---|---|---|
| `node_count` | `int` | Number of nodes |
| `max_neighbor_count` | `int` | Upper bound on neighbors per node (row-padding width) |
| `rank` | `int` | Latent factor dimensionality |
| `rank_float4_stride` | `int` | `ceil(rank / 4)` — internal SIMD row stride |
| `constant_weight` | `float` | Value used for every edge when `using_constant_weight` is `True` |
| `using_constant_weight` | `bool` | Whether all edges currently use `constant_weight` instead of `U·V` |

### Methods

#### `get(source_node, target_node) -> float`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source_node` | `int` | required | Source node index |
| `target_node` | `int` | required | Target node index |

**Returns:** `float` — weight of the edge `source_node -> target_node`.

---

#### `get_neighbors(node_index) -> list[int]`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `node_index` | `int` | required | Node to query |

**Returns:** `list[int]` — outgoing neighbor indices of `node_index`.

---

#### `neighbor_weight_stats() -> WeightStats`

**Returns:** [`WeightStats`](#weightstats) — mean/stddev/RMS/min/max over all edge weights.

---

#### `set_constant_weight(value) -> None`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `value` | `float` | required | New value for `constant_weight` |

Does **not** itself set `using_constant_weight` — pair with
`engine.use_constant_weight = True` to actually switch the engine over to
constant weights.

---

## `SimulationRecorder`

Standalone buffering/compression/recording layer underlying
`start_static_record`, exposed for callers who want to drive their own tick
loop (e.g. mixing custom stimulus logic between `step_simulation` calls)
while still producing a `.spire` file.

### Constructor

```python
SimulationRecorder(filename, neuron_count, compression="auto", compression_level=None,
                    compression_async=False, queue_max=8, chunk_bytes=4194304)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | `str` | required | Output path for the `.spire` recording |
| `neuron_count` | `int` | required | Frame width — every call to `record_frame` must pass an array of exactly this length |
| `compression` | `str \| None` | `"auto"` | `"auto"`, `"none"`, `"gzip"`, `"xz"`, or `"bz2"` |
| `compression_level` | `int \| None` | `None` | Codec-native level/preset; `None` → default `6` |
| `compression_async` | `bool` | `False` | Run compression/I-O on a background thread fed by a bounded queue (named `compression_async`, not `async` — `async` is a reserved keyword since Python 3.7) |
| `queue_max` | `int` | `8` | Max queued chunks when `compression_async=True`. `0` → unbounded |
| `chunk_bytes` | `int` | `4 * 1024 * 1024` (4 MiB) | Bytes per queued chunk |

### Properties

| Name | Type | Access | Description |
|---|---|---|---|
| `neuron_count` | `int` | read-only | Frame width this recorder expects |

### Methods

#### `record_frame(membrane_potentials) -> None`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `membrane_potentials` | `np.ndarray[float32]`, shape `(neuron_count,)` | required | One frame's membrane potentials |

Appends the frame's bytes to an internal buffer, flushing to the (possibly
compressed, possibly async) sink once `chunk_bytes` is reached.

**Raises:** `RuntimeError` if the array's length doesn't equal `recorder.neuron_count`.

---

#### `finish() -> None`

Flushes any remaining buffered bytes and closes the underlying file. Call
exactly once.

---

## `NmlNetworkRunner`

The NeuroML/LEMS → GPU codegen path (epic #1), exposed to Python (ticket #133) — separate from,
and never dependent on, `SpikeEngine`. Wraps the whole parse → resolve → lower → allocate →
assemble → step_tick pipeline behind one class; this is the minimum surface for "construct, run,
and read back a GLIF network," not a binding of every pipeline stage (`ModelSpecification`,
`IrProgram`, `ModelAllocation`, and `AssembledModel` all stay C++-internal).

### Constructor

```python
NmlNetworkRunner(nml_file_path, dt_seconds=1e-4)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nml_file_path` | `str` | required | Path to a top-level `<neuroml>` document. `<include href="...">` targets resolve relative to this file's own directory |
| `dt_seconds` | `float` | `1e-4` | Simulation timestep, seconds |

Seeds every population's membrane potential (`v`) to its cell type's own `EL` when one is baked
(every GLIF variant's `OnStart` is `v = EL`); a cell type with no `EL` is left at zero. Only the
Phase-1 stimulus shape `explicitInput` + `pulseGenerator` is driven automatically each tick — a
model using `<inputList>` for its stimulus instead sees no automatic current injection here.

### Methods

#### `step() -> None`

Advances the simulation by exactly one tick: injects this tick's precomputed pulseGenerator
stimulus contribution (if any) into every neuron, then runs one tick of cell/synapse dynamics.

---

#### `run(tick_count) -> None`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tick_count` | `int` | required | Number of ticks to advance |

Convenience loop over `step()`. Releases the GIL for the duration of the call.

---

#### `membrane_potentials() -> np.ndarray[float32]`

**Returns:** shape `(neuron_count,)` — every neuron's `v` (cell-state slot 0), by global neuron
index across every population.

---

#### `last_spiked() -> np.ndarray[int64]`

**Returns:** shape `(neuron_count,)` — tick each neuron last fired, or `-1` if it never has.

### Properties

| Name | Type | Access | Description |
|---|---|---|---|
| `neuron_count` | `int` | read-only | Total neurons across every population in the model |
| `current_tick` | `int` | read-only | Number of ticks advanced so far |

---

## Result types

These are plain read-only structs (`def_readonly` properties only — no
methods, no constructors exposed to Python).

### `WeightStats`

| Field | Type | Description |
|---|---|---|
| `mean` | `float` | Mean edge weight |
| `standard_deviation` | `float` | Standard deviation of edge weights |
| `root_mean_square` | `float` | RMS of edge weights |
| `min_value` | `float` | Minimum edge weight |
| `max_value` | `float` | Maximum edge weight |

### `ScaleResult`

| Field | Type | Description |
|---|---|---|
| `target_root_mean_square` | `float` | RMS value weights were scaled toward |
| `scale_factor` | `float` | Multiplier applied to every edge weight |
| `before` | `WeightStats` | Stats before scaling |
| `after` | `WeightStats` | Stats after scaling |

### `ScaledReservoirResult`

Returned by [`scale_randomized_weights_near_bifurcation`](#scale_randomized_weights_near_bifurcationinput_period1-scale12-freeze_learningfalse---scaledreservoirresult).

| Field | Type | Description |
|---|---|---|
| `weight_scale_result` | `ScaleResult` | Before/after `WeightStats`, the applied `scale_factor`, and `target_root_mean_square` (the RMS value weights were scaled toward) |
| `w_accum` | `float` | Underlying accumulated-input bifurcation-weight estimate |
| `w_instant` | `float` | Underlying instantaneous-input bifurcation-weight estimate |
