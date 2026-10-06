# `spikecorec` Python API reference

Every public name in the extension. See [README.md](README.md) for the guide.

The module mirrors the C++: `spikecorec` is the `spikecorec` namespace, `spikecorec.nml` is
`spikecorec::nml` and `spikecorec.units` is `spikecorec::units`. Every class, member and
function keeps its C++ name. Arrays come back as numpy copies, never as views of engine
memory, so they stay valid after the engine is gone.

```python
import spikecorec as spc
from spikecorec import nml, units
```

---

## Module `spikecorec`

### Constants

| name | value | meaning |
|---|---|---|
| `NEVER_SPIKED_TICK` | `-1` | `last_spiked` of a neuron that has not fired |
| `MAX_RANK_FLOAT4_STRIDE` | `64` | the basis holds at most `64 * 4 = 256` lanes |
| `DEFAULT_BRANCHING_FACTOR` | `4` | the k²-tree's branching factor |
| `LOG_PATH` | `"logs/spikecorec.log"` | where the engine's log file goes |

### `set_log_level(level) -> None`

One of `trace debug info warn err critical off`.

### Topology generators

Adjacency lists (`list[list[int]]`, one row per neuron) for the connectivity-in-code
constructors.

| function | result |
|---|---|
| `square_torus(side_length)` | `side_length ** 2` cells, each wired to its 4 neighbours, edges wrapped |
| `small_world_torus(side_length, random_fanout=4, seed=-1)` | `square_torus` plus `random_fanout` long-range shortcuts per cell |
| `random_fixed_outdegree(side_length, fanout=8, seed=-1)` | directed random graph, out-degree `fanout`, no self-loops or repeats |

`seed < 0` is non-deterministic.

```python
spc.square_torus(3)[0]      # [1, 2, 3, 6]
```

---

## `SpikeEngine`

### Constructors

```python
SpikeEngine(lems_input_file, enable_hebbian_plasticity=False, fit_tolerance=FIT_TOLERANCE_STANDARD)
```
Everything comes from the document, connectivity included.

```python
SpikeEngine(lems_input_file, adjacency, synapse_component_id,
            connection_weight=1.0, connection_delay_seconds=0.0,
            enable_hebbian_plasticity=False, fit_tolerance=FIT_TOLERANCE_STANDARD)
```
Connectivity from `adjacency` instead, where `adjacency[source]` lists `source`'s
targets. Every edge carries the named synapse, `connection_weight` and
`connection_delay_seconds`.

```python
SpikeEngine(lems_input_file, adjacency, synapse_component_ids, synapse_proportions=[],
            connection_weight=1.0, connection_delay_seconds=0.0,
            enable_hebbian_plasticity=False, fit_tolerance=FIT_TOLERANCE_STANDARD)
```
Several synapses. The edges are split, in adjacency order, into one contiguous share per
synapse, sized by `synapse_proportions` (normalised; empty means equal shares). Nothing is
random. Because edges are ordered by source, a share is a run of source neurons, but a
neuron at a boundary can carry both. `weights.get_edge_synapse_prototype` gives any
edge's synapse, as an index into `synapse_component_ids`.

Construction parses and validates the document, allocates the model's buffers, fits the
weight matrix, and generates and compiles the kernel. It raises `RuntimeError` when the
adjacency names a neuron outside the network, a named synapse is not declared, or the model
uses dynamics the engine does not simulate (it names the ComponentType and why).

`enable_hebbian_plasticity` is accepted, but the generated kernel stages no plasticity
updates yet, so it changes nothing; the engine logs a warning saying so.

```python
engine = spc.SpikeEngine("LEMS_torus.xml", spc.square_torus(5),
                         ["excitatorySynapse", "inhibitorySynapse"],
                         synapse_proportions=[0.8, 0.2],
                         connection_delay_seconds=1e-3)
```

### What it was built from

| name | type | meaning |
|---|---|---|
| `context` | `nml.NML_Context` | the parsed model, by reference. Don't call `parse()` or `reset()` on it |
| `weights` | `WeightMatrix` | every per-edge value, by reference |
| `total_neuron_count` | int | neurons across all populations |
| `lifetime` | int | ticks in the run |
| `step_dt` | float | seconds per tick |
| `simulation_seed` | int | the document's seed, `0` when it names none |
| `random_generator` | `nml.RandomGenerator` | draws every `random()` the run calls, seeded with `simulation_seed`, by reference |
| `random_values_count` | int | slots in `random_values`: one per `random()` call per neuron that runs it |
| `spike_history_row_count` | int | rows in the spike-history ring |
| `alive` | bool | constructed and not shut down |
| `hebbian_plasticity_enabled` | bool | |

### The generated kernel

| name | type | meaning |
|---|---|---|
| `master_kernel_source` | str | the generated kernel exactly as compiled, every constant baked in |
| `kernel_parameter_names` | list[str] | the kernel's bound parameters, in binding order |
| `synapse_active_ticks` | int | ticks after a neuron spikes during which its outgoing synapses run: the longest delay plus the slowest synapse's settle time. `-1` means every synapse runs every tick |
| `projection_run_count` | int | runs of identical edges the kernel looks synapses up in |

### Weight-matrix fit and refit settings

The weight matrix is fitted once at construction and refit during a run. A refit folds the
updates in S (see `WeightMatrix`) into the basis and empties S. Refits happen after a tick's
updates are merged, when either trigger fires and the last refit was at least
`minimum_ticks_between_refits` ago.

| name | default | meaning |
|---|---|---|
| `fit_tolerance` | `1e-3` | worst relative error a fit or refit may leave on any per-edge value, against the larger of the value and its plane's RMS. Read/write; also a constructor argument, for the construction fit |
| `FIT_TOLERANCE_PRECISE` | `1e-5` | preset for comparisons against reference simulators; delays exact up to 50 000 ticks |
| `FIT_TOLERANCE_ACCURATE` | `1e-4` | preset for validation runs and reported results; up to 5 000 ticks |
| `FIT_TOLERANCE_STANDARD` | `1e-3` | the default; up to 500 ticks |
| `FIT_TOLERANCE_COMPACT` | `1e-2` | preset for the largest networks, where memory matters most; up to 50 ticks |
| `refit_occupancy_threshold_fraction` | `0.2` | refit once the fullest plane of S holds updates on this fraction of the edges; `0` disables. Also the size S is kept at per plane, so setting it resizes S |
| `set_refit_occupancy_threshold_fraction(fraction)` | | the same setter as a method |
| `refit_every_n_ticks` | `0` | also refit every this many ticks; `0` disables |
| `minimum_ticks_between_refits` | `1000` | refits are at least this far apart; S grows in between if it must. `0` allows any spacing |
| `last_refit_tick` | | read-only; the construction fit counts as tick 0 |

A tighter tolerance costs rank (memory) and fitting time. A delay stays on its exact tick
while `tolerance * delay < 0.5` ticks.

```python
engine.fit_tolerance = spc.SpikeEngine.FIT_TOLERANCE_ACCURATE   # for every later refit
engine.refit_occupancy_threshold_fraction = 0.5                 # S resized to half the edges per plane
engine.minimum_ticks_between_refits = 2000
```

### Plasticity settings

| name | default | meaning |
|---|---|---|
| `DEFAULT_PLASTICITY_DELTA_CAPACITY` | `65536` | slots per plane of S reserved for plasticity when it is enabled |
| `plasticity_fold_every_n_ticks` | `64` | with plasticity on, merge the queued updates every this many ticks |
| `plasticity_target_root_mean_square` | | with plasticity on, the weights are scaled back to this RMS after each refit; negative disables |

### Running

#### `run() -> None`
Every tick of the document's run, back to back.

#### `step_simulation(tick) -> None`
One tick: stimulus, the kernel, merging the tick's updates into S, a refit when one is due,
recording. Pass a monotonically increasing `tick`.

```python
for tick in range(engine.lifetime):
    engine.step_simulation(tick)
```

### Reading state

#### `read_state_variable(neuron_index, variable_name) -> float`
One neuron's variable, read from the GPU. Raises if its cell type does not declare it.

#### `state_variable_array(variable_name) -> np.ndarray[float32]`
The same variable for every neuron. Every cell type must declare it.

#### `synapse_state_variable_plane(synapse_component_id, variable_name) -> int`
The `weights` plane that holds one of a synapse's LEMS state variables on its edges.

```python
plane = engine.synapse_state_variable_plane("excitatorySynapse", "I")
engine.weights.neighbor_weights_for_matrix(plane)    # I on every edge
```

#### Device buffers

Copies of the engine's GPU buffers. Each raises `RuntimeError` after `shutdown()`.

| name | shape | meaning |
|---|---|---|
| `cell_state` | float32[`context.get_cell_state_size()`] | every cell's state variables; `context.resolve_path("pop[0]/v")` gives an index |
| `network_inputs` | float32[2][neurons] | synaptic input accumulators, alternating by tick parity |
| `event_arrival_count` | uint32[2][neurons] | spikes that reached each cell, alternating by tick parity; read by cells with an `OnEvent`. Shape [0][neurons] when no cell type has one |
| `spike_history` | uint8[`spike_history_row_count`][neurons] | row `tick % rows` holds that tick's spikes |
| `last_spiked` | int64[neurons] | each neuron's last spike tick, or `NEVER_SPIKED_TICK` |
| `random_values` | float32[`random_values_count`] | the last tick's uniform draws, refilled before every dispatch |

### Stimulus

What the engine collected from the document's inputs.

| name | type |
|---|---|
| `continuous_injection_targets` | int64[] |
| `continuous_injection_amplitudes` | float32[] |
| `continuous_injection_start_ticks`, `continuous_injection_end_ticks` | int64[] |
| `scheduled_spike_trains` | list[`SpikeEngine.ScheduledSpikeTrain`]: `neuron_index`, `magnitude`, `event_ticks`, `cursor` |

### Results

| name | type | meaning |
|---|---|---|
| `spike_counts_per_neuron` | int64[neurons] | spikes per neuron so far |
| `recorded_spikes` | (float64[], int64[]) | `(time_seconds, neuron_index)`, parallel arrays over every spike so far |
| `traced_selections` | list[`RecordingSelection`] | the document's `OutputColumn`s, one per column of `recorded_traces` |
| `recorded_trace_times` | float64[] | seconds |
| `recorded_traces` | float32[ticks][selections] | |
| `mean_firing_rate_hertz()` | float | spikes per neuron per second |
| `fraction_of_neurons_that_spiked()` | float | |

```python
times, neurons = engine.recorded_spikes
first_of_cell_3 = times[neurons == 3][0]
```

### Writing results

#### `record_membrane_video(path, frame_stride=1) -> None`
Records every neuron's `v` every `frame_stride` ticks to a `.spire` file. **Call before
`run()`.** Raises if a cell type has no `v`. `membrane_video_frame_stride` reads it back.

#### `write_recordings() -> None`
Closes the membrane video and writes the document's `OutputFile` / `EventOutputFile`
elements.

#### `write_spike_file(path) -> None`
Every spike as `time<TAB>neuron`.

#### `shutdown() -> None`
Releases the GPU buffers. The destructor calls it too.

---

## `WeightMatrix`

Reached as `engine.weights`. Every per-edge value of the network's synapses (its weight,
its delay in ticks, and each of its LEMS state variables) is one **plane**. Planes are
stored as `M_k = U diag(Ck) Vᵀ` over the edges the k²-tree holds: a shared low-rank basis
`U`, `V` with one coefficient row `Ck` per plane, never one float per edge. Reads are
computations. The basis trades a measured accuracy for storage, within `fit_tolerance`.

Changes during a run go into the **sparse delta matrix S**, one sparse row set per plane.
Every read adds S. A refit fits each plane that has updates to basis + S, changing only that
plane's own lanes (lanes no other plane reads), and empties S. A plane without updates is left
exactly as it is.

### Planes and constants

| name | value | meaning |
|---|---|---|
| `WEIGHT_PLANE` | `0` | |
| `DELAY_PLANE` | `1` | in ticks |
| `FIRST_STATE_VARIABLE_PLANE` | `2` | the synapse type's StateVariables follow, in order; see `SpikeEngine.synapse_state_variable_plane` |
| `LANE_GROUP` | `4` | the rank is always a multiple of this |
| `DEFAULT_FIT_TOLERANCE` | `1e-3` | |
| `DEFAULT_FIT_RIDGE` | `1e-4` | ridge regularization of a refit |
| `FIT_STALL_IMPROVEMENT`, `PLANE_FIT_STALL_WINDOW_SWEEP_COUNT` | `0.05`, `10` | a plane's fit at one lane count has stalled when its worst error improved by less than 5% over 10 sweeps; it then adds lanes |
| `PLANE_FIT_MAXIMUM_SWEEP_COUNT` | `100` | sweeps at one lane count at most. Once a plane has its largest degree + 4 lanes, one exact solve replaces the sweeps |
| `DEFAULT_REFIT_OCCUPANCY_THRESHOLD_FRACTION` | `0.2` | |

### Shape and fit

| name | type | meaning |
|---|---|---|
| `node_count`, `total_edge_count`, `max_neighbor_count`, `max_predecessor_count` | int | |
| `matrix_count` | int | planes: weight, delay, then the largest synapse type's state variables |
| `updated_plane_count` | int | planes the kernel updates each tick |
| `rank`, `rank_float4_stride` | int | lanes in the basis, and lanes / 4 |
| `fit_tolerance` | float | the tolerance refits run to (read/write; the engine sets it from `SpikeEngine.fit_tolerance` before each refit it runs) |
| `measured_fit_error` | list[float] | per plane, the worst relative error the last fit left |
| `worst_fit_error()` | float | the worst of those |
| `check_indexing` | bool | read/write |
| `k2tree` | `K2Tree` | by reference |

### The basis, copied out

| name | shape |
|---|---|
| `U_matrix`, `V_matrix` | float32[node_count][rank] |
| `coefficients` | float32[matrix_count][rank] |
| `edge_row_offset` | int64[node_count + 1]: node `n`'s edges are ordinals `edge_row_offset[n]` to `edge_row_offset[n + 1]` |
| `projection_first_edge_ordinal`, `projection_edge_count` | int64[runs] |
| `projection_synapse_prototype` | int32[runs] |

### Reads

| method | result |
|---|---|
| `get(source_node, target_node)` | the weight plane at one edge, S included |
| `get_for_matrix(source_node, target_node, matrix_index)` | any plane at one edge, S included |
| `get_edge_delay_ticks(source_node, target_node)` | the delay plane rounded to whole ticks, at least one, as the kernel uses it |
| `get_edge_synapse_prototype(source_node, target_node)` | the edge's synapse, as an index into `context.simulation.synapse_instances` |
| `edge_ordinal(source_node, target_node)` | the edge's number in canonical order, or `None` |
| `get_neighbors(node_index)` | int32[]: targets, in ordinal order |
| `get_predecessors(node_index)` | int32[]: every source of an edge into node_index |
| `neighbor_weights()` | float32[total_edge_count]: the weight plane, by ordinal |
| `neighbor_weights_for_matrix(matrix_index)` | float32[total_edge_count]: any plane, by ordinal |
| `neighbor_weight_stats()` | `WeightStats` of the weight plane |
| `check_index_inbounds(source, target)`, `check_index_inbounds(node_index)` | bool |

```python
w = engine.weights
w.neighbor_weights_for_matrix(w.DELAY_PLANE)      # every delay, in ticks
```

### S, the sparse delta matrix

| name | meaning |
|---|---|
| `sparse_delta_capacity` | update slots per plane |
| `sparse_delta_entry_count` | list: updates each plane holds |
| `sparse_delta_occupancy_fraction()` | fraction of the edges holding an update, in the fullest plane |
| `sparse_deltas(matrix_index)` | `(edge_ordinals, deltas)`: one plane's updates |
| `sparse_delta_row_start` | int32[matrix_count][node_count + 1]: the CSR row starts |
| `sparse_delta_edge_ordinal`, `sparse_delta_value` | [matrix_count][capacity]: the CSR entries; the first `sparse_delta_entry_count[m]` of row `m` are live |
| `delta_capacity_for_threshold(fraction)` | the capacity a refit threshold asks for (every edge at `0`), plus `plasticity_reserve_entries` |
| `resize_delta_capacity(new_capacity)` | resizes S, keeping its updates; it never shrinks below them |
| `plasticity_reserve_entries` | read/write: slots per plane kept for plasticity, applied at the next resize |
| `pending_delta_capacity` | room in the device's per-tick update queue, fixed when the kernel is built |
| `pending_delta_count` | updates the device queued and `compact_pending_deltas` has not merged yet |
| `pending_deltas()` | `(edge_ordinals, deltas, matrix_indices)` of those |

### Updates and refits

#### `accumulate_edge_delta(matrix_index, source_node, target_node, delta) -> None`
Adds `delta` to one edge in one plane, through S. Reads see it at once; the next refit folds
it into the basis. S grows if the plane is full.

#### `compact_pending_deltas() -> None`
Merges the updates the kernel queued into S, growing S if a plane would overflow.
`step_simulation` does this every tick.

#### `is_refit_due(occupancy_threshold_fraction) -> bool`
`True` once the fullest plane of S holds updates on that fraction of the edges.

#### `refit(ridge_regularization=1e-4) -> None`
Fits the basis to basis + S, starting from the current basis, until every plane meets
`fit_tolerance`, adding lanes when the current rank cannot. Then empties S.

#### `scale_neighbor_weights_to_root_mean_square(target_root_mean_square, epsilon=1e-12) -> ScaleResult`
Scales the weight plane (its coefficient row and its updates) to a target RMS.

```python
w = engine.weights
w.accumulate_edge_delta(w.WEIGHT_PLANE, 0, 1, 0.25)
w.get(0, 1)                      # 1.25
w.refit()
w.sparse_delta_entry_count       # [0, 0, 0, 0]
```

#### `save(path) -> None`, `load_from_disk(path) -> None`
The basis and S; the k²-tree is not saved, so a load goes into a matrix over the same
network, and must match its node and plane counts.

### `WeightStats` and `ScaleResult`

`WeightStats`: `mean`, `standard_deviation`, `root_mean_square`, `min_value`,
`max_value`. `ScaleResult`: `target_root_mean_square`, `scale_factor`, `before`, `after`.

---

## `K2Tree`

Reached as `engine.weights.k2tree`: which (source, target) pairs are edges.

| name | meaning |
|---|---|
| `branching_factor`, `superblock_size_words`, `node_count`, `padded_node_count`, `tree_height`, `internal_bit_count` | the tree's shape |
| `internal_node_words_length`, `leaf_node_words_length`, `rank_superblock_length`, `rank_subblock_length` | its arrays' lengths |
| `ADJACENT_BATCH_QUERY_CAP` | queries `adjacent_batch` stages in one pass |
| `adjacent(source_node, target_node)` | `1` for an edge, else `0` |
| `get_neighbors(node_index, max_neighbor_count)`, `get_predecessors(node_index, max_neighbor_count)` | int32[] |
| `adjacent_batch(source_indices, target_indices)` | uint8[]: `adjacent` for each pair |
| `trace(source_node, target_node)` | logs the walk for one pair, returns `adjacent` |
| `save(path)` | |

---

## Module `spikecorec.nml`

The parsed model, as the C++ front-end holds it. Everything is read-only, and handed out by
reference: each object keeps its context (and so its engine) alive.

### `NML_Context`

```python
context = nml.NML_Context()
context.parse("LEMS_model.xml")      # no GPU involved
```

| name | meaning |
|---|---|
| `parse(main_filepath)` | parses the document, everything it includes and the standard library it names |
| `reset()` | empties the context |
| `validate_lems_schema(lems_filepath)` | XSD-validates a NeuroML document (a LEMS root passes unchecked); errors in `last_schema_validation_errors` |
| `simulation` | `NML_SimulationContext` |
| `component_types` | dict[str, `NML_ComponentType`], the standard library's included |
| `component_instances` | dict[str, `NML_ComponentInstance`], keyed by full path such as `"network/cells"` |
| `model_units` | dict[str, `units.UnitDefinition`] |
| `model_constants` | dict[str, float], SI |
| `document_roots`, `main_document_root` | the parsed XML trees (`NML_Node`) |
| `document_filepaths`, `target_component_id` | |
| `STANDARD_LIBRARY_PATH`, `NML_SCHEMA_PATH` | baked in at build time |
| `find_instance(instance_id)` | `NML_ComponentInstance` or `None` |
| `is_instance_of(instance, type_name)` | whether its type is `type_name` or extends it |
| `resolve_quantity(value)` | `"-60mV"` → `-0.06`, with the document's own units |
| `resolve_path(path, current=None)` | a LEMS path relative to the target network, such as `"cells[0]/v"`, → its `cell_state` index, or `-1` |
| `neuron_index_of(cell_memory_index)` | the neuron whose cell holds that index, or `-1` |
| `get_cell_state_size()`, `get_population_size(population)`, `get_cell_variable_count(cell_instance_index)` | |

### `NML_SimulationContext`

| name | meaning |
|---|---|
| `simulation_component_id`, `target_network_id` | |
| `step_dt`, `simulation_duration` | seconds |
| `total_tick_count`, `total_neuron_count`, `total_edge_count` | |
| `maximum_edge_delay` | ticks |
| `random_seed` | int or `None` |
| `population_base_indices` | dict: population path → where its cells start in `cell_state` |
| `global_constants` | dict[str, float] |
| `cell_instances` | list[`NML_ComponentInstance`] |
| `synapse_instances` | list[`NML_ComponentInstance`], in prototype order: an edge's prototype is its index here |
| `network_data` | `AdjacencyList` |
| `input_profiles` | list[`SimulationInputConfig`] |
| `recording_profiles` | list[`RecordingConfig`] |

### `NML_ComponentType`

`name`, `extends` (`NML_ComponentType` or `None`), `source_file`, `source_node`
(`NML_Node`), `dynamics` (list of `NML_DynamicsExpression`), `state_variable_names`
(inherited first; for a synapse these are its per-edge planes, in order), and
`find_declaration(namespace_key)`, which searches up the `extends` chain and returns an
`NML_Node` or `None`.

### `NML_ComponentInstance`

`id`, `instance_data` (attributes, verbatim), `data_order` (child ids, in source order),
`parent_instance`, `component_type`, `has_value(key)`, `value_or(key, fallback="")`.

### `NML_DynamicsExpression`

One `<Dynamics>` entry: `source_tag` (`NML_DeclarationType`), `target`, `expression`,
`regime_name`, `condition`, `select`, `reduce`.

### `NML_Node` and `NML_Tag`

A parsed XML element: `NML_Node.body` is its `NML_Tag` and `NML_Node.children` its element
children. `NML_Tag` has `tag_name`, `tag_type` (`NML_DeclarationType`), `attributes`,
`has_attribute(name)`, `get_attribute(name)`, `get_attribute_or(name, fallback)`,
`namespace_key()` and `is_declaration_type()`.

### Connectivity and stimulus

| class | members |
|---|---|
| `AdjacencyList` | `list` (one list of `NML_NetworkEdge` per source), `in_network(node_index)`, `is_parent(parent, prospective_child)`, `len()`, `edge_arrays()` → dict of parallel arrays `parent`, `child`, `weight`, `delay_ticks`, `component_id` |
| `NML_NetworkEdge` | `component_id`, `weight`, `delay_ticks`, `parent`, `child` |
| `SimulationInputConfig` | `input_component_id`, `targets`, `amplitude`, `rate`, `start_tick`, `end_tick` (`0` runs to the end), `continuous_current_injection` |
| `InputTarget` | `neuron_index`, `weight`, `event_ticks` |

### Functions

| function | result |
|---|---|
| `network_populations(context)` | the target network's populations, in document order |
| `population_cell(context, population)` | the cell instance a population is made of |
| `component_parameter_values(context, cell)` | dict: every Parameter, Constant and DerivedParameter, SI |
| `starting_values(context, cell, random_generator=None)` | the same plus each state variable's starting value (OnStart, else 0); an OnStart that calls `random()` draws from `random_generator` |
| `spike_history_length(context)` | rows in the engine's spike-history ring |
| `evaluate_lems(expression, values, owner_name="python", random_generator=None)` | a LEMS expression's value; every name it reads must be in `values`, and `random()` draws from `random_generator` |
| `tokenize_lems(expression, owner_name="python")` | list of `LemsExpressionToken` (`lexeme`, `kind`: `LemsExpressionToken.Kind`) |
| `output_format_for_filename(filename)` | `OutputFileFormat` |

### `RandomGenerator`

Random values from built-in distributions, from one `std::mt19937_64` engine.
`RandomGenerator(seed=0)`.

| method | result |
|---|---|
| `uniform()` | uniform on (0, 1), never 0 or 1 even as float32; LEMS `random(1)` |
| `normal(mean, standard_deviation)` | normal |
| `exponential(rate)` | exponential with that rate |
| `poisson(mean)` | Poisson count |

`NML_DeclarationType` enumerates the LEMS declaration tags (`Parameter`, `StateVariable`,
`TimeDerivative`, `OnEvent`, ..., `NOT_A_TYPE`).

```python
for population in nml.network_populations(engine.context):
    cell = nml.population_cell(engine.context, population)
    print(population.id, cell.component_type.name, cell.component_type.state_variable_names)
```

---

## Module `spikecorec.units`

| name | meaning |
|---|---|
| `UnitDefinition` | `scale`, `offset`: SI value = written value * scale + offset |
| `parse_quantity(value)` | `"-60mV"` → `-0.06`, built-in units only |
| `split_quantity(value)` | `"-60mV"` → `(-60.0, "mV")` |
| `unit_suffix_scale(suffix)` | `"mV"` → `1e-3`; unknown suffixes scale by 1 |
| `seconds_to_ticks(seconds, step_dt)`, `ms_to_ticks(total_ms, ms_step)` | rounded to the nearest tick. Raises `ValueError` when the step is not positive and finite, the duration is negative or not finite, or the tick count does not fit in a 64-bit integer |
| `ms_to_seconds(ms)`, `seconds_to_ms(seconds)` | |
| `tick_to_seconds(tick, total_seconds, total_ticks)`, `tick_to_ms(tick, total_ms, total_ticks)` | |

---

## Recording

### `.spire` files

A 4-byte big-endian neuron count, then native float32 frames, optionally compressed.

| name | meaning |
|---|---|
| `read_spire_recording(filename)` | float32[frames][neurons]; compression from the extension |
| `SpireWriter(filename, neuron_count)` | uncompressed writer: `write_frame(array)` (length must be `neuron_count`), `neuron_count` |
| `SpireReader(filename)` | uncompressed reader: `read_frame()` → array or `None` at the end, `neuron_count` |
| `SimulationRecorder(filename, neuron_count, compression="auto", compression_level=None, compression_async=False, queue_max=8, chunk_bytes=4194304)` | buffered, optionally compressed writer: `record_frame(array)`, `finish()`, `neuron_count` |
| `SpireCompression` | `None_`, `Gzip`, `Xz`, `Bz2` |
| `resolve_spire_compression(filename, requested="auto")` | `(SpireCompression, filename)`; `"auto"` reads the extension (`.gz`, `.xz`, `.bz2`) |

```python
recorder = spc.SimulationRecorder("out.spire.xz", engine.total_neuron_count)
for tick in range(engine.lifetime):
    engine.step_simulation(tick)
    if tick % 5 == 0:
        recorder.record_frame(engine.state_variable_array("v"))
recorder.finish()
```

### Recording configuration

What the parser read from the document's `OutputFile` / `EventOutputFile` elements.

| class | members |
|---|---|
| `RecordingConfig` | `output_filenames`, `file_output_format`, `selections`, `recordings_count`, `DEFAULT_MAX_LOG_BYTES` |
| `RecordingSelection` | `quantity_path` (e.g. `"pop1[0]/v"`), `variable_name`, `event_port`, `neuron_index` |
| `OutputFileFormat` | `SPIRE`, `SPIREGZIP`, `SPIREBZ2`, `SPIREXZ`, `SPIKE_EVENTS`, `NML_STANDARD` |

---

## Not exposed

- **The GPU backend, raw device pointers and the kernel code generator** (`EngineBackend`,
  `EnginePointer`, `Codegen`, the kernel AST and its parser). They need the engine's GPU
  context. The generated kernel is readable as `SpikeEngine.master_kernel_source`.
- **Constructing a `WeightMatrix` or `K2Tree` from Python, or
  `WeightMatrix.declare_projections`.** Both need the GPU backend, and re-declaring an
  engine's matrix would leave the kernel's copy of the projection runs stale.
- **The parser's internal stages** (`parse_component_types`, `create_component_instance`
  and the rest), which `parse()` runs in order. On their own they leave a half-built context.
- **The `.spire` byte-stream plumbing** (`SpireSink`, `SpireSource`, `AsyncSpireWriter`), which
  `SimulationRecorder` wraps.
- **`plasticity_learning_rate`, `plasticity_l2_regularization` and
  `plasticity_iterations`.** Nothing in the engine reads them yet.
