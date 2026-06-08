# `spikecorec` — Python bindings

`spikecorec` is a pybind11 wrapper around the C++/CUDA/Metal spiking-network
engine in this repo. It exposes the same simulation primitives as the
`spikecore` Python reference (`SpikeEngine`, topology generators, weight
inspection) while running the actual step loop on the GPU.

## Installation

Build and install the extension as an editable package (auto-detects the
backend — Metal on macOS, CUDA elsewhere):

```bash
pip install pybind11
make python
```

Or force a backend explicitly:

```bash
SPIKECOREC_BACKEND=metal make python
SPIKECOREC_BACKEND=cuda  make python
```

This runs `pip install -e .` under the hood, so `import spikecorec` works from
anywhere once it completes. On Metal it also copies `default.metallib` next to
the compiled `.so` — required at import time for the GPU kernels to load.

> **Note:** the C++ GPU context is initialized once, at module import
> (`initialize_gpu_context()`), and intentionally never torn down — there is no
> safe ordering between releasing GPU resources and Python's garbage collector
> finalizing `SpikeEngine`/`WeightMatrix` objects. The OS reclaims GPU memory at
> process exit, the same way most CUDA-backed Python extensions behave.

```python
import spikecorec
print(spikecorec.__version__)
```

## Quick start

```python
import numpy as np
import spikecorec

# Build an 8x8 reservoir (64 neurons) on a 4-neighbor wraparound torus.
network = spikecorec.square_torus(8)
engine = spikecorec.SpikeEngine(network, shape=[8, 8], rank=4)

# Designate which neurons receive external input each tick.
engine.set_input_neurons([0, 1, 2])

# Step the simulation. input_values[i] is added to the membrane potential of
# input_neuron_indices[i] (set above) before the step's spike propagation runs.
for tick in range(50):
    engine.step_simulation([2.0, 2.0, 2.0], tick=tick)

# Read GPU state back as numpy arrays.
mp = engine.get_membrane_potentials()
print("membrane potentials:", mp[:5])
print("active neurons this tick:", engine.get_active_neuron_count())

# Pull a feature vector for use in a downstream readout/classifier.
features = engine.get_reservoir_features_vector(tick=50, spike_tau=10.0, voltage_scale=1.0)
print("feature vector shape:", features.shape)  # (2 * neuron_count + 1,)

engine.shutdown()
```

## Topology generators

Build adjacency lists (`list[list[int]]`, node `i`'s entry holds the indices of
its outgoing neighbors) for use as a `SpikeEngine`'s `network` argument. These
mirror `spikecore/topologies.py`; randomized generators are **structurally**
equivalent to the NumPy/PCG64 reference but not bit-identical (different RNG
stream).

```python
spikecorec.square_torus(k)
# 4-neighbor wraparound k*k grid. Every node has exactly 4 neighbors
# (right, left, down, up). Deterministic — no seed.

spikecorec.small_world_torus(k, random_fanout=4, seed=-1)
# square_torus plus `random_fanout` deterministic long-range shortcuts per
# node, so activity can mix across the reservoir rather than only drifting
# locally. Every node ends up with out-degree 4 + random_fanout (capped by
# reservoir size). seed < 0 seeds from std::random_device (non-deterministic).

spikecorec.random_fixed_outdegree(k, fanout=8, seed=-1)
# Directed random graph — no torus/grid structure at all. Every node has the
# same out-degree `fanout`, with no self-loops and no duplicate edges.
```

All three return `k * k` nodes indexed `0 .. k*k - 1`.

## `SpikeEngine`

```python
SpikeEngine(network, shape, rank=1, resting_mp=0.1, decay_rate=0.01, learning_rate=0.00222)
```

- `network` — `list[list[int]]` adjacency list (e.g. from a topology generator
  above). Copied into a k²-tree-backed `WeightMatrix` at construction time.
- `shape` — `[rows, cols]`; `neuron_count = rows * cols` must equal
  `len(network)`.
- `rank` — latent-factor dimensionality for the low-rank weight
  approximation (`-1` → `min(64, neuron_count)`).
- `resting_mp`, `decay_rate`, `learning_rate` — simulation constants (resting
  membrane potential, exponential decay rate toward rest, Hebbian learning
  rate).

### Running the simulation

```python
engine.set_input_neurons(input_neuron_list)
```
Designates which neurons receive external input. `step_simulation`'s
`input_values[i]` is added to the membrane potential of
`input_neuron_list[i]` — the two lists are matched positionally and must be
the same length.

```python
engine.step_simulation(input_values, tick, override_input_neurons=[], decay_all_neurons=False)
```
Advances the simulation by one tick:
1. (optionally) decays every neuron toward `resting_mp` based on elapsed time
   since its last update (`decay_all_neurons=True`),
2. adds `input_values` into the membrane potentials of the configured input
   neurons,
3. merges `override_input_neurons` into this tick's active set (forces those
   neurons to be processed even if they weren't already active — useful for
   injecting activity at arbitrary locations, e.g. a "live keyboard input"
   feature),
4. propagates spikes through the network for one tick, applying Hebbian
   weight updates along active edges.

```python
engine.reset_state(last_spiked_value=0, active_gen_value=-1)
```
Returns the engine to its initial state: membrane potentials back to
`resting_mp`, spike/update/active-set bookkeeping cleared, network input
accumulator zeroed.

```python
engine.is_alive() -> bool
engine.shutdown()
```
`shutdown()` frees the engine's GPU buffers; `is_alive()` reflects whether
that has happened. The engine also calls `shutdown()` automatically when
garbage collected.

### Reading state back

All of these copy GPU buffer contents into fresh numpy arrays (no GPU memory
is exposed directly to Python — there's no sound cross-runtime ownership story
for that):

```python
engine.get_membrane_potentials()    # float32[neuron_count]
engine.get_network_inputs()         # float32[neuron_count] — pending input accumulator
engine.get_last_spiked()            # int64[neuron_count]   — tick each neuron last fired
engine.get_last_tick_updated()      # int64[neuron_count]   — tick each neuron last processed
engine.get_active_neuron_indices()  # int32[active_neuron_count] — current active set
engine.get_active_neuron_count()    # int — size of the current active set
```

### Reservoir features

```python
engine.get_reservoir_features_vector(tick, spike_tau, voltage_scale) -> np.ndarray[float32]
```
Builds a `2 * neuron_count + 1`-element feature vector suitable for a
downstream readout layer:
- `features[0:neuron_count]` — per-neuron spike trace,
  `exp(-(tick - last_spiked) / spike_tau)` (`0` if the neuron has never fired)
- `features[neuron_count:2*neuron_count]` — per-neuron scaled membrane
  potential, `(membrane_potential - resting_mp) / voltage_scale`
- `features[-1]` — bias term, always `1.0`

Both `spike_tau` and `voltage_scale` must be positive (the call is a no-op,
returning an unmodified buffer, if either is `<= 0`).

### Weight scaling near the bifurcation point

These tune the network's weights so that, on average, a neuron receiving
periodic input of period `input_period` sits near its spiking threshold —
the "edge of chaos" regime that gives reservoirs rich dynamics.

```python
engine.estimate_bifurcation_weight(input_period=1) -> (w_accum, w_instant)
```
Returns the accumulated-input and instantaneous-input weight estimates that
would put a neuron with the engine's current `resting_membrane_potential`,
`spike_threshold`, and `decay_rate` right at its firing threshold under
periodic stimulation.

```python
engine.scale_uniform_weights_near_bifurcation(input_period=1, scale=1.2, freeze_learning=False)
    -> (target, w_accum, w_instant)
```
Sets every edge to the same constant weight `target = w_accum * scale`
(`weights.using_constant_weight` becomes `True`). If `freeze_learning` is set,
`learning_rate` is zeroed.

```python
engine.scale_randomized_weights_near_bifurcation(input_period=1, scale=1.2, freeze_learning=False)
    -> ScaledReservoirResult
```
Instead of collapsing to a constant, rescales the existing (randomized) edge
weights so their RMS matches `target = abs(w_accum * scale)`. Returns a
`ScaledReservoirResult`:

| field | meaning |
|---|---|
| `weight_scale_result` | `ScaleResult` — before/after `WeightStats` and the applied `scale_factor` |
| `target_root_mean_square` | the RMS value weights were scaled toward |
| `w_accum`, `w_instant` | the underlying bifurcation-weight estimates |

### Properties

| name | type | meaning |
|---|---|---|
| `neuron_count` | `int`, read-only | total neurons (`shape[0] * shape[1]`) |
| `input_neuron_count` | `int`, read-only | length of the list passed to `set_input_neurons` |
| `resting_membrane_potential` | `float`, read/write | rest potential neurons decay toward |
| `decay_rate` | `float`, read/write | exponential decay rate toward rest |
| `learning_rate` | `float`, read/write | Hebbian update step size |
| `spike_period` | `int`, read/write | refractory ticks after a spike before a neuron can fire again |
| `spike_threshold` | `float`, read/write | membrane potential at which a neuron spikes |
| `use_constant_weight` | `bool`, read/write | when `True`, every edge uses `weights.constant_weight` instead of the learned `U·V` factorization |
| `running` | `bool`, read-only | `False` once `shutdown()` has run |
| `weights` | `WeightMatrix`, read-only | the engine's weight matrix (see below) — its lifetime is tied to the engine |

### `setup_lifetime` (membrane-potential logging)

```python
engine.setup_lifetime(lifetime, allocate_logs, max_log_bytes=512 * 1024 * 1024)
```
Allocates host-side per-neuron membrane-potential log buffers
(`neuron_count * lifetime` floats) for recording over a fixed-length run.
Raises `RuntimeError` if the requested allocation would exceed
`max_log_bytes`.

## Simulation recording (`.spire` format)

`spikecorec` can record a run's membrane potentials to disk in the same
`.spire` binary format used by the `spikecore` Python reference — files are
**byte-for-byte interoperable** between the two implementations (and with
`gzip`/`xz`/`bzip2`-compressed variants of either).

### Format

```
offset 0:  u32, big-endian   — neuron_count
offset 4:  f32[neuron_count] — frame 0 membrane potentials (native byte order)
offset 4 + neuron_count*4:   — frame 1 ...
...repeated until EOF
```

No magic bytes or version marker. The whole byte sequence (header included)
may optionally be wrapped end-to-end in gzip (`.gz`/`.gzip`), xz/lzma
(`.xz`/`.lzma`), or bzip2 (`.bz2`) — `spikecorec` auto-detects this from the
filename extension when `compression="auto"` (the default).

> **Note:** gzip/xz/bz2 support depends on `zlib`/`liblzma`/`libbz2` being
> available when `spikecorec` was built (the Makefile/`setup.py` probe for
> them via `pkg-config` and degrade gracefully). Run `make info` to check
> which were detected (`zlib found`/`lzma found`/`bzip2 found`); uncompressed
> `.spire` always works regardless. Reading/writing a format whose library
> wasn't available raises a `RuntimeError` with a clear message.

### Recording a run

```python
engine.start_static_record(
    input_spikes, lifetime, filename,
    record_membrane=True, record_stride=1,
    compression="auto", compression_level=None,
    full_decay=True,
    compression_async=False, compression_queue_max=8,
    compression_chunk_bytes=4 * 1024 * 1024,
)
```

Drives its own tick loop for `lifetime` ticks and writes a `.spire` recording
to `filename` — a direct port of the `spikecore` reference's
`start_static_record`:

- `input_spikes` — `list[list[float]]`; `input_spikes[tick][i]` is added to
  the membrane potential of `input_neuron_indices[i]` (as set via
  `set_input_neurons`) for that tick. Must provide at least `lifetime` ticks,
  and each tick's row must contain exactly `input_neuron_count` values — a
  wrong-width or empty row raises `RuntimeError` before any file is written.
- Each tick, the configured input neurons are forced into the active set
  (mirrors the reference's per-tick `_add_active` call) before the step runs.
- `record_membrane`/`record_stride` — whether/how often (every `record_stride`
  ticks) to snapshot membrane potentials into the recording.
- `compression`/`compression_level` — `"auto"` (infer from `filename`'s
  extension), `"none"`, `"gzip"`, `"xz"`, or `"bz2"`; `compression_level` maps
  to each codec's native level/preset (1-9 for gzip/bz2, 0-9 for xz; default
  `6`, matching the reference).
- `full_decay` — on each *recorded* tick, runs a full decay pass (advancing
  `last_tick_updated` for every neuron, not just active ones) immediately
  before the membrane-potential snapshot — exactly mirrors the reference's
  `_decay_all` timing. This is a real, stateful operation, not a preview.
- `compression_async`/`compression_queue_max`/`compression_chunk_bytes` —
  when `compression_async=True`, compression and file I/O run on a background
  thread fed by a bounded queue (`compression_queue_max` chunks of up to
  `compression_chunk_bytes` bytes each) so the simulation loop isn't blocked
  on slow I/O — this call releases the GIL while it runs, so other Python
  threads keep making progress too. `compression_queue_max=0` makes the queue
  unbounded (no backpressure), matching Python's `queue.Queue(maxsize=0)`.

```python
engine.set_input_neurons([0, 1, 2])
input_spikes = [[2.0, 2.0, 2.0] for _ in range(500)]
engine.start_static_record(
    input_spikes, lifetime=500, filename="run.spire.gz",
    record_stride=2, compression_async=True,
)
```

### Reading a recording back

```python
spikecorec.read_spire_recording(filename) -> np.ndarray[float32]  # (frame_count, neuron_count)
```

Auto-detects compression from the extension, parses the header, and decodes
every frame into one contiguous `(frame_count, neuron_count)` array. Raises
`RuntimeError` if the final frame is truncated.

```python
recording = spikecorec.read_spire_recording("run.spire.gz")
print(recording.shape)  # (250, neuron_count)
```

### `SimulationRecorder` (custom per-tick recording loops)

`start_static_record` is built on top of a standalone `SimulationRecorder` —
exposed directly for callers who want to drive their own tick loop (e.g.
mixing in custom stimulus logic between `step_simulation` calls) while still
producing a `.spire` file:

```python
recorder = spikecorec.SimulationRecorder(
    filename, neuron_count,
    compression="auto", compression_level=None,
    compression_async=False, queue_max=8, chunk_bytes=4 * 1024 * 1024,
)
recorder.record_frame(membrane_potentials)  # float32[neuron_count]
recorder.finish()                           # flush + close — call exactly once

recorder.neuron_count                       # int, read-only — frame width it expects
```

`record_frame` appends one frame's worth of bytes to an internal buffer,
flushing to the (possibly compressed, possibly async) sink once `chunk_bytes`
is reached; `finish()` flushes any remainder and closes the underlying file.
The array passed to `record_frame` must have exactly `recorder.neuron_count`
elements — a wrong-size array raises `RuntimeError` rather than reading out of
bounds. As with `start_static_record`, `queue_max=0` makes the async queue
unbounded.

## `WeightMatrix`

Accessible read-only via `engine.weights`. Wraps a k²-tree-compressed sparse
adjacency structure plus a low-rank `U·V` weight factorization.

```python
wm = engine.weights

wm.node_count            # int — number of nodes
wm.max_neighbor_count    # int — upper bound on neighbors per node (row padding width)
wm.rank                  # int — latent factor dimensionality
wm.rank_float4_stride    # int — ceil(rank / 4); internal SIMD row stride
wm.constant_weight       # float — value used for every edge when using_constant_weight is True
wm.using_constant_weight # bool

wm.get(source_node, target_node) -> float          # weight of a specific edge
wm.get_neighbors(node_index) -> list[int]          # outgoing neighbor indices of a node
wm.neighbor_weight_stats() -> WeightStats          # mean/stddev/RMS/min/max over all edge weights
wm.set_constant_weight(value)                      # set the constant-weight value
```

`WeightStats` and `ScaleResult` are plain read-only structs:

| `WeightStats` | | `ScaleResult` | |
|---|---|---|---|
| `mean` | `float` | `target_root_mean_square` | `float` |
| `standard_deviation` | `float` | `scale_factor` | `float` |
| `root_mean_square` | `float` | `before` | `WeightStats` |
| `min_value` | `float` | `after` | `WeightStats` |
| `max_value` | `float` | | |

## Putting it together: a minimal reservoir-readout loop

```python
import numpy as np
import spikecorec

network = spikecorec.small_world_torus(k=16, random_fanout=4, seed=42)
engine = spikecorec.SpikeEngine(network, shape=[16, 16], rank=8)
engine.set_input_neurons(list(range(8)))

# Push the reservoir's weights toward the edge-of-chaos regime for period-1 input.
engine.scale_randomized_weights_near_bifurcation(input_period=1, scale=1.2)

feature_log = []
for tick in range(200):
    drive = np.random.uniform(0.5, 2.5, size=8).tolist()
    engine.step_simulation(drive, tick=tick, decay_all_neurons=True)
    if tick % 10 == 0:
        feature_log.append(engine.get_reservoir_features_vector(tick, spike_tau=10.0, voltage_scale=1.0))

features = np.stack(feature_log)
print(features.shape)  # (20, 2 * 256 + 1)

engine.shutdown()
```

## Running the tests

The C++ core/codec test suite (topology, k²-tree, weight-matrix, engine, and
`.spire` recording tests — `tests/test_core.cpp`) builds and runs with:

```bash
make test          # auto-detects backend (Metal on macOS, CUDA elsewhere)
make test-metal    # force the Metal build
```

To build the Python extension itself, point `make python` at your interpreter.
The project's intended Python environment is the `spike_engine` conda env
(`numpy`, `pybind11`, etc. already present):

```bash
make python PYTHON=/path/to/envs/spike_engine/bin/python
```
