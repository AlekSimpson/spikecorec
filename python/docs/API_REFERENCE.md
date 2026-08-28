# `spikecorec` Python API reference

Every public name in the extension. See [README.md](README.md) for the guide.

```python
import spikecorec as spc
```

---

## Module functions

### `square_torus(side_length) -> list[list[int]]`

`side_length ** 2` cells, each wired to its 4 neighbours with the edges wrapped.
Deterministic.

```python
spc.square_torus(3)[0]      # [1, 2, 3, 6]
```

### `small_world_torus(side_length, random_fanout=4, seed=-1) -> list[list[int]]`

`square_torus` plus `random_fanout` long-range shortcuts per node. `seed < 0` is
non-deterministic.

### `random_fixed_outdegree(side_length, fanout=8, seed=-1) -> list[list[int]]`

Directed random graph, no grid structure. Every node has out-degree `fanout`, no
self-loops, no duplicates.

```python
len(spc.random_fixed_outdegree(22, fanout=20, seed=1))    # 484
```

### `read_spire_recording(filename) -> np.ndarray[float32]`

A `.spire` file as `(frames, neurons)` in volts.

```python
frames = spc.read_spire_recording("out.spire")
frames.shape        # (1000, 25)
```

### `set_log_level(level) -> None`

One of `trace debug info warn error critical off`.

---

## `SpikeEngine`

### Constructors

```python
SpikeEngine(lems_input_file, enable_hebbian_plasticity=False)
```
Connectivity comes from the document's `<projection>` elements.

```python
SpikeEngine(lems_input_file, adjacency, synapse_component_id,
            connection_weight=1.0, connection_delay_seconds=0.0,
            enable_hebbian_plasticity=False)
```
Connectivity from `adjacency` (one row per neuron); every edge uses the named synapse.

```python
SpikeEngine(lems_input_file, adjacency, synapse_component_ids,
            synapse_proportions=[], connection_weight=1.0,
            connection_delay_seconds=0.0, enable_hebbian_plasticity=False)
```
Each neuron draws one of `synapse_component_ids`; all its outgoing edges use it.
`synapse_proportions` is normalised, empty means an equal share each. Seeded from the
document.

Raises `RuntimeError` if the adjacency row count differs from the population size, a
named synapse is not declared, or the model uses dynamics the engine will not simulate.

> `enable_hebbian_plasticity=True` does not currently work on the Metal backend: the
> generated kernel exceeds Metal's 31-buffer limit and its plasticity block references
> parameters under the wrong names. Leave it `False`.

```python
engine = spc.SpikeEngine("LEMS_torus.xml", spc.square_torus(5),
                         ["excitatorySynapse", "inhibitorySynapse"],
                         synapse_proportions=[0.8, 0.2],
                         connection_delay_seconds=1e-3)
```

### Methods

#### `run() -> None`
Every tick of the document's run, back to back.

#### `step_simulation(tick) -> None`
One tick. Pass a monotonically increasing `tick`.

```python
for tick in range(engine.lifetime):
    engine.step_simulation(tick)
```

#### `read_state_variable(neuron_index, variable_name) -> float`
One neuron's variable, read from the GPU. Raises if the cell type does not declare it.

```python
engine.read_state_variable(0, "v")      # -0.057
```

#### `state_variable_array(variable_name) -> np.ndarray[float32]`
The same variable for every neuron. Every cell type must declare it.

#### `mean_firing_rate_hertz() -> float`
#### `fraction_of_neurons_that_spiked() -> float`

```python
engine.mean_firing_rate_hertz()             # 22.8
engine.fraction_of_neurons_that_spiked()    # 0.84
```

#### `record_membrane_video(path, frame_stride=1) -> None`
Records every neuron's `v` every `frame_stride` ticks to `path`. **Call before `run()`.**
Raises if a cell type has no `v`.

#### `write_recordings() -> None`
Writes the model's `OutputFile` / `EventOutputFile` elements and closes the recording.

#### `write_spike_file(path) -> None`
Every spike as `time<TAB>neuron`, all neurons, no document involvement.

#### `shutdown() -> None`

### Properties

| name | type | meaning |
|---|---|---|
| `total_neuron_count` | int | neurons across all populations |
| `lifetime` | int | ticks in the run |
| `step_dt` | float | seconds per tick |
| `alive` | bool | constructed and not shut down |
| `spike_counts` | int64[n] | spikes per neuron |
| `spike_times` | (float64[], int64[]) | `(times_seconds, neuron_indices)` |
| `synapse_choice_per_neuron` | int32[n] | index into `synapse_component_ids`; empty otherwise |
| `hebbian_plasticity_enabled` | bool | |
| `weights` | `WeightMatrix` | by reference |

```python
times, neurons = engine.spike_times
first_of_cell_3 = times[neurons == 3][0]
```

---

## `WeightMatrix`

Reached as `engine.weights`. Weights are reconstructed from a shared low-rank basis plus
sparse corrections — never one float per edge — so reads are computations.

### Properties

| name | type | meaning |
|---|---|---|
| `node_count` | int | |
| `total_edge_count` | int | |
| `rank` | int | lanes in the basis |
| `max_neighbor_count` | int | widest adjacency row |
| `measured_weight_fit_error` | float | worst relative error; `0.0` when exact |
| `sparse_delta_capacity` | int | correction slots allocated |
| `sparse_delta_occupancy_fraction` | float | how full they are |
| `refit_occupancy_threshold_fraction` | float | occupancy that makes a refit due |

### Methods

#### `get(source_node, target_node) -> float`
Reconstruction plus correction.

```python
engine.weights.get(0, 1)        # 1.0
```

#### `get_for_matrix(source_node, target_node, matrix_index) -> float`
`0` is weight, `1` is delay.

#### `get_edge_delay_ticks(source, target) -> int`
#### `get_edge_synapse_prototype(source, target) -> int`
#### `edge_ordinal(source, target) -> int | None`
The edge's number in the canonical ordering, or `None` if not an edge.

#### `neighbors(node_index) -> np.ndarray[int32]`
#### `predecessors(node_index) -> np.ndarray[int32]`

```python
engine.weights.neighbors(0)         # array([1, 4, 5, 20], dtype=int32)
```

#### `edge_weights() -> np.ndarray[float32]`
Every edge's weight in canonical order, length `total_edge_count`.

#### `weight_stats() -> WeightStats`

```python
s = engine.weights.weight_stats()
s.mean, s.root_mean_square, s.min_value, s.max_value, s.standard_deviation
```

#### `accumulate_edge_delta(matrix_index, source, target, delta) -> None`
Queues a correction, visible to reads immediately and folded into the basis at the next
refit.

**Needs capacity.** `sparse_delta_capacity` is sized at construction from what the basis
actually missed, so a model the basis reproduces exactly gets **zero** slots and this
call does nothing:

```python
w = engine.weights
w.sparse_delta_capacity          # 0 when measured_weight_fit_error == 0.0
w.accumulate_edge_delta(0, 0, 1, 0.25)
w.get(0, 1)                      # still 1.0 -- nowhere to put the correction
```

#### `compact_pending_deltas() -> None`
#### `is_refit_due() -> bool`
`True` once occupancy passes `refit_occupancy_threshold_fraction`.

#### `refit(sweep_count=4, ridge_regularization=1e-3) -> None`
Re-optimises the basis toward the corrections, then drops the ones it absorbed.

```python
if w.is_refit_due():
    w.compact_pending_deltas()
    w.refit()
```

#### `save(path) -> None`
#### `load_from_disk(path) -> None`

---

## `WeightStats`

Read-only: `mean`, `standard_deviation`, `min_value`, `max_value`, `root_mean_square`.

---

## `SimulationRecorder`

Writes `.spire` directly, for custom recording loops. `record_membrane_video` is the
usual way in.

```python
SimulationRecorder(filename, neuron_count, compression="auto",
                   compression_level=None, compression_async=False,
                   queue_max=8, chunk_bytes=4194304)
```

#### `record_frame(membrane_potentials) -> None`
One frame; length must equal `neuron_count`.

#### `finish() -> None`
Flushes and closes. Raises if `record_frame` is called afterwards.

#### `neuron_count` — int

```python
recorder = spc.SimulationRecorder("out.spire", engine.total_neuron_count)
for tick in range(engine.lifetime):
    engine.step_simulation(tick)
    if tick % 5 == 0:
        recorder.record_frame(engine.state_variable_array("v"))
recorder.finish()
```

---

## `RecordedSpike`

Read-only: `time_seconds`, `neuron_index`. Prefer `engine.spike_times`, which is arrays.
