# `spikecorec` — Python bindings

GPU spiking-neural-network simulation driven by NeuroML. You supply a LEMS document; the
engine generates a GPU kernel for those dynamics, compiles it, and runs the network.

The Python API mirrors the C++ one name for name: `spikecorec` holds the engine, its weight
matrix and recording; `spikecorec.nml` the parsed model; `spikecorec.units` unit
conversion. [API_REFERENCE.md](API_REFERENCE.md) lists everything.

## Install

```bash
make python                     # builds the extension into .venv (Metal on macOS)
.venv/bin/python -c "import spikecorec; print('ok')"
```

Needs [uv](https://docs.astral.sh/uv/) and libxml2 (found through `pkg-config`). The first
`make python` creates `.venv` with Python 3.13 (`PYTHON_VERSION=3.12 make python` picks
another) and installs `pybind11`, `numpy`, `setuptools` and `matplotlib` into it. Run
scripts with `.venv/bin/python`, or `source .venv/bin/activate` first.

## Quick start

```python
import spikecorec as spc

engine = spc.SpikeEngine("examples/models/LEMS_single_cell.xml")
engine.run()

print(engine.total_neuron_count)             # 1
print(engine.mean_firing_rate_hertz())       # 100.0
print(engine.read_state_variable(0, "v"))    # -0.07  (volts)
engine.shutdown()
```

Everything (cells, synapses, stimulus, timestep, run length) comes from the document.
Membrane potentials are **volts**; multiply by 1000 for mV.

## Connectivity from Python

Writing one `<connection>` element per edge does not scale. Pass an adjacency list
instead, one row per neuron, and the document declares only cells, synapses and stimulus.

```python
# A -> B -> C
engine = spc.SpikeEngine("examples/models/LEMS_three_cell_chain.xml",
                         [[1], [2], []],
                         "chainSynapse",
                         connection_weight=1.0,
                         connection_delay_seconds=1e-3)
engine.run()
print(engine.spike_counts_per_neuron)    # [10 10 10]
```

### Topology generators

Each takes a `side_length` and returns `side_length ** 2` rows.

```python
spc.square_torus(10)                                 # 100 cells, 4 neighbours, wrapped
spc.small_world_torus(10, random_fanout=4, seed=1)   # torus + long-range shortcuts
spc.random_fixed_outdegree(22, fanout=20, seed=1)    # 484 cells, no grid structure
```

### Several synapses

Pass a **list** of synapses and a share for each. The edges are split, in adjacency order,
into one contiguous share per synapse. Nothing is random: the topology is exactly the list
you gave. Edges are ordered by source cell, so with `[0.8, 0.2]` the first 80% of cells send
only the first synapse and the rest only the second, except a cell at the boundary, which can
send both.

```python
engine = spc.SpikeEngine("examples/models/LEMS_glif1_torus.xml",
                         spc.square_torus(5),
                         ["excitatorySynapse", "inhibitorySynapse"],
                         synapse_proportions=[0.8, 0.2],
                         connection_weight=1.0,
                         connection_delay_seconds=1e-3)
engine.run()

w = engine.weights
w.get_edge_synapse_prototype(0, 1)    # 0: excitatorySynapse, the first in the list
```

Omit `synapse_proportions` for equal shares.

## Reading results

```python
engine.spike_counts_per_neuron           # int64[neurons], spikes per neuron
times, neurons = engine.recorded_spikes  # parallel arrays over every spike
engine.read_state_variable(7, "v")       # one neuron, one variable
engine.state_variable_array("v")         # float32[neurons]
engine.mean_firing_rate_hertz()
engine.fraction_of_neurons_that_spiked()
engine.recorded_traces                   # the document's OutputColumns, [tick][column]
```

The raw device state is there too, as copies: `cell_state`, `last_spiked`,
`network_inputs` and `spike_history`.

## Recording

`record_membrane_video` must be called **before** `run()`, since it is what makes the run
record anything.

```python
engine.record_membrane_video("out.spire", 5)   # every 5th tick, all neurons
engine.run()
engine.write_recordings()                      # closes it; writes the model's OutputFiles
engine.write_spike_file("out_spikes.dat")      # "time<TAB>neuron" per spike

frames = spc.read_spire_recording("out.spire")
frames.shape                                   # (frames, neurons), float32 volts
```

`.spire` is a 4-byte big-endian neuron count followed by native float32 frames, compressed
when the filename ends in `.gz`, `.xz` or `.bz2`.

### Rendering a video

```python
import sys; sys.path.insert(0, "examples")
from video_utils import render_membrane_video

render_membrane_video("out.spire",
                      spikes_path="out_spikes.dat",
                      duration=0.5,
                      trace_neurons=[(3, "excitatory"), (11, "inhibitory")])
```

Writes `out.mp4` (needs ffmpeg; a `.gif` output path uses Pillow).

## Driving the tick loop yourself

```python
for tick in range(engine.lifetime):
    engine.step_simulation(tick)
    if tick % 10 == 0:
        trace.append(engine.read_state_variable(0, "v"))
```

Pass a monotonically increasing tick; the engine uses it for the spike-history ring and
the refractory gate. Reading state copies from the GPU, so sample, don't read every tick.

## The parsed model

`engine.context` is the model the engine was built from. `nml.NML_Context` parses one on
its own, with no GPU.

```python
from spikecorec import nml

context = nml.NML_Context()
context.parse("examples/models/LEMS_glif1_torus.xml")
for population in nml.network_populations(context):
    cell = nml.population_cell(context, population)
    print(population.id, context.get_population_size(population),
          cell.component_type.name, cell.component_type.state_variable_names)

context.resolve_path("sheet[0]/v")                  # index into engine.cell_state
nml.starting_values(context, cell)                  # parameters and OnStart values, SI
context.component_types["GLIF1Cell"].dynamics       # the type's <Dynamics>, in order
```

## The weight matrix

Every per-edge value of a synapse (its weight, its delay in ticks, and each of its LEMS
state variables) is one **plane** of the weight matrix. Planes are never stored one float
per edge: each is a shared low-rank basis `U diag(Ck) Vᵀ` with its own coefficient row,
plus a sparse matrix S of the updates since the last refit. Reads are computations.

```python
w = engine.weights
w.matrix_count, w.rank, w.total_edge_count
w.get(0, 1)                                                # the weight of edge 0 -> 1
w.neighbor_weights_for_matrix(w.DELAY_PLANE)               # every delay, by edge ordinal
plane = engine.synapse_state_variable_plane("excitatorySynapse", "I")
w.neighbor_weights_for_matrix(plane)                       # the synapse's I on every edge
w.measured_fit_error                                       # per plane, what the fit cost
```

The basis trades a measured accuracy for storage. Fits and refits run until every plane is
within `engine.fit_tolerance`; `SpikeEngine.FIT_TOLERANCE_PRECISE`, `_ACCURATE`,
`_STANDARD` (the default, `1e-3`) and `_COMPACT` are presets.

During a run, synapse state changes every tick and the changes go into S. Once the fullest
plane of S holds updates on `refit_occupancy_threshold_fraction` of the edges, the engine
refits and empties S. That fraction is also the size S is kept at.

```python
engine.refit_occupancy_threshold_fraction = 0.5    # default 0.75; resizes S now
engine.minimum_ticks_between_refits = 2000         # default 1000; S grows in between
engine.refit_every_n_ticks = 5000                  # default 0, off
engine.fit_tolerance = spc.SpikeEngine.FIT_TOLERANCE_ACCURATE
```

## What the engine supports and refuses

Any NeuroML model that fits the engine's constraints runs, whether its ComponentTypes are
standard or your own:

- **Cells:** point neurons whose `<Dynamics>` is built from state variables with `OnStart`
  values, derived variables (conditional ones included), time derivatives (forward Euler),
  `OnCondition` blocks that assign state and emit spikes, and optionally an integrating
  regime paired with a refractory one. `random()` draws a fresh value for each neuron on
  every call, seeded by the `<Simulation>`'s `seed`, so spike sources such as
  `spikeGeneratorPoisson` run as populations.
- **Synapses:** current-based. The equations can use the synapse's own state, its
  parameters and the connection's weight, and can react to arriving spikes with `OnEvent`.
- **Stimulus:** `pulseGenerator` and `pulseGeneratorDL`, a constant input over a time
  window, and `spikeArray`, whose spikes are delivered as kicks into the target's input
  (its `amplitude`, or else one sized from the target's capacitance and threshold).
- **Connectivity:** any, from the document or from Python.

Anything else fails at construction rather than simulating incorrectly: synapses that read
the target cell, `random()` in a synapse, `OnEvent` in a cell, other regime structures, ion
channels, concentrations, multicompartment morphologies and kinetic schemes. So does a
synapse or input that does not provide what its target cell reads, such as a current
(`pulseGenerator`) into a cell that reads a dimensionless input (`izhikevichCell`). A
conductance-based synapse, for example, reads the target cell's `v`, which a synapse's
per-edge code cannot see:

```python
spc.SpikeEngine("LEMS_refused.xml", [[1], []], "conductanceSynapse")
# RuntimeError: Synapse ComponentType 'expTwoSynapse' ('conductanceSynapse'): 'v' is not
#               a state variable, a set parameter, a constant, a derived variable or weight
```

The exception for now is the other input types. They are accepted, but do not yet behave as
NeuroML defines them:

- `sineGenerator` and `sineGeneratorDL` run as a constant input at their amplitude.
- `timedSynapticInput` delivers its spikes as kicks, like `spikeArray`, instead of through
  its synapse.
- `rampGenerator`, `compoundInput` (and their DL variants), `poissonFiringSynapse`,
  `transientPoissonFiringSynapse`, `voltageClamp` and `voltageClampTriple` deliver nothing.

## Tutorial

`examples/notebooks/01_tutorial_getting_started.py` is the tutorial in jupytext format:
`jupytext --to ipynb 01_tutorial_getting_started.py`. `examples/notebooks/demo.ipynb`
renders videos of four models.

## Logging

```python
spc.set_log_level("warn")    # trace debug info warn err critical off
```

## Tests

```bash
make test        # the C++ suite
```
