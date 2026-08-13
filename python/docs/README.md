# `spikecorec` — Python bindings

`spikecorec` is a pybind11 wrapper around the C++/Metal/CUDA spiking-network engine in this
repo. The engine simulates whatever a NeuroML/LEMS document describes: it parses the model,
generates a GPU kernel from the LEMS dynamics, compiles it at runtime, and runs the tick loop
on the GPU.

> **Looking for a parameter/method lookup while developing?** See
> [`API_REFERENCE.md`](API_REFERENCE.md) — a table-based reference of every function, class,
> method, property and default. This document is the walkthrough.

## Installation

Build and install the extension as an editable package (auto-detects the backend — Metal on
macOS, CUDA elsewhere):

```bash
make python PYTHON=/path/to/python
```

The interpreter needs `pybind11` to build and `numpy` to be useful; `pytest` on top of that to
run the test suite. `PYTHON` defaults to `python3`, which is only right if that interpreter is
the one with those packages.

Or force a backend explicitly:

```bash
SPIKECOREC_BACKEND=metal make python PYTHON=...
SPIKECOREC_BACKEND=cuda  make python PYTHON=...
```

This runs `pip install -e .` under the hood, so `import spikecorec` works from anywhere once
it completes. On Metal it also copies `default.metallib` next to the compiled `.so`.

The build also needs **libxml2** (via `pkg-config`) — the NeuroML/LEMS front end is not
optional. `setup.py` fails with that named as the reason if it is missing.

> **Note:** the GPU context is initialized once, at module import
> (`initialize_gpu_context()`), and intentionally never torn down — there is no safe ordering
> between releasing GPU resources and Python finalizing a `SpikeEngine`. The OS reclaims GPU
> memory at process exit, the same way most GPU-backed Python extensions behave.

```python
import spikecorec
print(spikecorec.__version__)
```

## The idea

**The model drives the engine, not Python.** A NeuroML/LEMS document states everything: the
cell dynamics (as LEMS `<ComponentType>` declarations), the populations, the wiring, the
stimulus, the timestep, the run length and the recordings. `SpikeEngine` reads the file and
does one thing per call — advance every neuron by one `dt`.

There is no Python API for building a network out of an adjacency list, designating input
neurons, or pushing a vector of drive in each tick. That was the pre-NeuroML engine; those
methods are commented out in `engine.h` and are not bound. If you want a different network,
write a different model.

**Everything crossing the boundary is SI.** A `C="100pF"` reads back as `1e-10` farads, a
membrane potential as `-0.07` volts, a synaptic current as `6e-10` amperes. Nothing is scaled
for readability on the way out.

## Quick start

```python
import numpy as np
import spikecorec

spikecorec.set_log_level("warn")

engine = spikecorec.SpikeEngine("python/tests/fixtures/glif3_ring_network_top.nml")

print(engine.total_neuron_count)          # 8
print(engine.step_dt, engine.lifetime)    # 0.0001 s per tick, 3000 ticks
print(engine.cell_type_names())           # ['GLIF3Cell']
print(engine.state_variable_names(0))     # ['v', 'asc1', 'asc2', 'refractoryTimeElapsed']

engine.run(engine.lifetime)

print(engine.state_variable_values("v"))  # membrane potential per neuron, in volts
print(engine.state_variable_values("asc1"))   # after-spike current, in amperes

engine.shutdown()
```

`examples/demo_script.py` is this, end to end, with recordings.

## Writing the model

A model needs two things the engine will not invent for you: a `<Simulation>` giving `step`
and `length`, and a root element it can read.

The NeuroML XSD describes `<neuroml>` documents. A raw LEMS `<ComponentType>` does not
validate against it, and neither does a `<Simulation>`. `SpikeEngine` XSD-validates only a
top-level file whose root **is** `<neuroml>`, so the usual shape is a `<Lems>` wrapper that
includes the network and states the simulation:

```xml
<Lems>
  <Target component="ringSimulation"/>
  <Include file="glif3_ring_network.nml"/>
  <Simulation id="ringSimulation" length="300ms" step="0.1ms" target="Glif3RingNet">
    <OutputFile id="membrane" fileName="/absolute/path/membrane.spire">
      <OutputColumn id="v0" quantity="RingPop[0]/v"/>
    </OutputFile>
    <EventOutputFile id="spikes" fileName="/absolute/path/spikes.spire" format="TIME_ID"/>
  </Simulation>
</Lems>
```

`<Include file="..."/>` resolves against the **including file's own** directory, not the
process working directory, so a generated wrapper in a scratch directory should name the
network by absolute path. `<OutputFile fileName>` is *not* resolved that way — it is used as
written, so a relative one lands wherever the process happens to be running.

Two things to watch, because neither is an error:

- **A model with no `<Simulation>` only warns.** `step_dt` stays `0.0` and `lifetime` stays
  `0`; every tick then integrates nothing while reporting success. Check `engine.step_dt`
  after constructing a model you did not write.
- **A `<Simulation>` with no `<OutputFile>` records nothing**, which is what you want for a
  test but not for a run you meant to keep. `engine.recording_output_filenames()` says what
  was actually opened.

Conductance-based synapses (`expOneSynapse` and friends, anything declaring `erev`/`gbase`)
are **refused by name** at construction rather than approximated. Use a current-based synapse
(`alphaCurrentSynapse`, `expCurrSynapse`) until the driving-force lowering lands.

## Running the simulation

```python
for tick in range(engine.lifetime):
    engine.step_simulation(tick)
```

or, keeping the loop in C++ and dropping the GIL for its duration:

```python
engine.run(engine.lifetime)
```

The tick index is **yours**. The engine keeps no counter and does not check the one you pass,
and it is not decorative: it indexes the precomputed stimulus streams *and* selects which row
of the synaptic delay ring is read. A repeated, skipped or out-of-order tick produces a wrong
simulation quietly. Step from `0` upwards by one.

The one tick the binding does refuse is a negative one, which is an out-of-bounds device access
rather than a wrong answer: the ring row is `tick % ring_depth` in signed arithmetic.

### Counting spikes

`spike_flags()` is this tick's emissions and nothing else — the master kernel lowers every flag
at the top of each tick — so a `run()` loop cannot see the emissions of any tick but the last.
To count them, drive the loop yourself:

```python
spike_counts = np.zeros(engine.total_neuron_count, dtype=np.int64)
for tick in range(engine.lifetime):
    engine.step_simulation(tick)
    spike_counts += engine.spike_flags()
```

`last_spiked()` is the cheaper alternative, with one sharp edge: it is **zero-filled at
construction**, not filled with `-1`. A `0` in it means either "fired on tick 0" or "never
fired", and there is nothing in the array to tell those apart. Use it for *when* a neuron last
fired, not for *whether* it ever did.

## Reading state back

Every accessor copies into a fresh numpy array; no GPU handle crosses the boundary.

```python
membrane_potentials = engine.state_variable_values("v")       # volts, per global neuron index
after_spike_current = engine.state_variable_values("asc1")    # amperes
capacitance         = engine.parameter_values("C")            # farads
```

`state_variable_values` / `parameter_values` gather rather than slice. `cell_state` is
sectioned by cell **type** — every neuron of type 0, then every neuron of type 1 — so the slot
a variable occupies is resolved per neuron, from that neuron's own type. Two cell types may
put `"v"` in different slots, and in a heterogeneous network one of them may not declare it at
all; asking for a variable a neuron's type does not declare **raises**, naming the neuron, its
type and what that type does declare.

That last part is the one place the binding deliberately differs from the engine. The engine's
own recording path, handed a quantity a cell type does not declare, warns and records that
neuron's *first* state variable instead. A read-back accessor doing the same would answer a
question about `asc1` with a membrane potential and look entirely plausible doing it.

The raw layout is available too, if you want it: `cell_state()`, `cell_parameters()`,
`cell_state_base()`, `cell_parameter_base()`, `cell_type_index()`, `synapse_state()` and
`network_inputs()`.

## Inspecting the wiring

```python
for source_neuron in range(engine.total_neuron_count):
    for target_neuron in engine.weights.get_neighbors(source_neuron):
        print(source_neuron, "->", target_neuron,
              engine.weights.get(source_neuron, target_neuron),
              engine.weights.get_edge_delay_ticks(source_neuron, target_neuron))
```

Worth doing on any model you did not write. A model whose connection paths did not resolve
produces a perfectly well-formed, fully **disconnected** network: the cells exist, the kernel
runs, every neuron integrates alone, and the result looks like a quiet network rather than a
broken one. The adjacency read-back is the only thing that tells those apart.

Two notes on what you will see:

- Delays come back in **ticks**, floored at `1`. The delay ring requires every delay to be at
  least one tick, so a `delay="0ms"` reads back as `1`.
- Two projections between the same ordered pair collapse onto the one adjacency slot that pair
  has. Their weights **sum**; conflicting delays or conflicting synapses **throw** at
  construction rather than picking one.

The `rank` property, and the `U·V` factorization behind it, are a **memory-compression**
scheme for very large adjacencies. They are not learning of any kind, and edge weights are
stored exactly regardless (`using_exact_edge_weights`).

## Recordings (`.spire`)

A `.spire` file is a 4-byte big-endian value count followed by raw `float32` frames, one per
tick. Compression is by extension: `.spire.gz`, `.spire.xz`, `.spire.bz2`.

The engine opens one recorder per `<OutputFile>` / `<EventOutputFile>` in the model, and writes
one frame to each per `step_simulation`. Columns are the file's own selections, in selection
order; a file with no child selections records every neuron (its first state variable for a
value file, its spike flag for an event file).

```python
engine.run(engine.lifetime)
engine.shutdown()                       # THIS is what flushes the last frames

frames = spikecorec.read_spire_recording(engine.recording_output_filenames()[0])
print(frames.shape)                     # (tick_count, column_count)
```

Read the file **after** `shutdown()`. Frames are buffered and only flushed when the recorder is
finished; a live read comes up short by up to a chunk.

`examples/render_spire_video.py` renders a `.spire` pair (spikes plus membrane) to a video.

### Writing your own

`SimulationRecorder` is the same buffering/compression layer, exposed for values you gather
yourself between ticks:

```python
recorder = spikecorec.SimulationRecorder("custom.spire.gz", engine.total_neuron_count)
for tick in range(engine.lifetime):
    engine.step_simulation(tick)
    recorder.record_frame(engine.state_variable_values("asc1"))
recorder.finish()
```

## Running the tests

```bash
make python-test PYTHON=/path/to/venv/bin/python
```

That builds the extension and runs `python/tests` against it: a binding-layer smoke test, an
end-to-end run of the checked-in 8-neuron GLIF3 ring fixture, a recording round-trip, and a
stimulus test that recovers the injected current from the membrane trajectory. A few seconds
in total.

One test there is an `xfail`: `test_pulse_generator_duration_is_the_duration_it_states`. A
`<pulseGenerator>` currently injects on one tick more than its `duration` states, because the
delivery window is computed with an exclusive end index and then filled inclusively. Its
`reason` string carries the two files and the two candidate one-line fixes.

It is **not** part of `make check`, which has to stay runnable with no arguments — whether
`python3` happens to have pybind11, numpy and pytest is a property of the machine, not of the
repository, and a `make check` that failed on a missing pytest would be reporting the
environment instead of the code. The C++ suite and the examples are what `make check` covers:

```bash
make check         # make test + make run-examples
make test-metal    # force the Metal build of the C++ suite
```
