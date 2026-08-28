# `spikecorec` — Python bindings

GPU spiking-neural-network simulation driven by NeuroML. You supply a LEMS document; the
engine generates a GPU kernel for those dynamics, compiles it, and runs the network.

## Install

```bash
make python          # builds the extension in place (Metal on macOS)
python3 -c "import spikecorec; print('ok')"
```

Needs `pybind11`, `numpy`, and `matplotlib` for the renderer.

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

Everything — cells, synapses, stimulus, timestep, run length — comes from the document.
Membrane potentials are **volts**; multiply by 1000 for mV.

## Connectivity from Python

Writing one `<connection>` element per edge does not scale. Pass an adjacency list
instead, one row per neuron, and the document declares only cells/synapses/stimulus.

```python
# A -> B -> C
engine = spc.SpikeEngine("examples/models/LEMS_three_cell_chain.xml",
                         [[1], [2], []],
                         "chainSynapse",
                         connection_weight=1.0,
                         connection_delay_seconds=1e-3)
engine.run()
print(engine.spike_counts)          # [10 10 10]
```

### Topology generators

Each takes a `side_length` and returns `side_length ** 2` rows.

```python
spc.square_torus(10)                            # 100 cells, 4 neighbours, wrapped
spc.small_world_torus(10, random_fanout=4, seed=1)   # torus + long-range shortcuts
spc.random_fixed_outdegree(22, fanout=20, seed=1)    # 484 cells, no grid structure
```

### Excitatory and inhibitory cells

Pass a **list** of synapses. Each neuron draws one, and every edge leaving it carries
that one — so the draw is what makes a cell excitatory or inhibitory (Dale's law).

```python
engine = spc.SpikeEngine("examples/models/LEMS_glif1_torus.xml",
                         spc.square_torus(5),
                         ["excitatorySynapse", "inhibitorySynapse"],
                         synapse_proportions=[0.8, 0.2],   # cortical ratio
                         connection_weight=1.0,
                         connection_delay_seconds=1e-3)
engine.run()

choice = engine.synapse_choice_per_neuron    # index into the list, per neuron
excitatory = [i for i, c in enumerate(choice) if c == 0]
```

Omit `synapse_proportions` for an equal share each. The draw is seeded from the
document, so it is the same every run.

## Reading results

```python
engine.spike_counts                    # int64[neurons], spikes per neuron
times, neurons = engine.spike_times    # parallel arrays over every spike
engine.read_state_variable(7, "v")     # one neuron, one variable
engine.state_variable_array("v")       # float32[neurons]
engine.mean_firing_rate_hertz()
engine.fraction_of_neurons_that_spiked()
```

## Recording

`record_membrane_video` must be called **before** `run()` — it is what makes the run
record anything.

```python
engine.record_membrane_video("out.spire", 5)   # every 5th tick, all neurons
engine.run()
engine.write_recordings()                      # closes it; writes the model's OutputFiles
engine.write_spike_file("out_spikes.dat")      # "time<TAB>neuron" per spike

frames = spc.read_spire_recording("out.spire")
frames.shape                                   # (frames, neurons), float32 volts
```

`.spire` is a 4-byte big-endian neuron count followed by native float32 frames,
xz-compressed.

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

Pass a monotonically increasing tick — the engine uses it for the spike-history ring and
the refractory gate. Reading state copies from the GPU, so sample, don't read every tick.

## Weights

Weights are never stored one float per edge. They are reconstructed from a shared
low-rank basis plus sparse corrections, so `get` is a computation, not a lookup.

```python
w = engine.weights
w.node_count, w.total_edge_count, w.rank
w.get(0, 1)                        # reconstructed weight
w.neighbors(0)                     # int32 targets
w.predecessors(1)                  # int32 sources
w.weight_stats().root_mean_square
w.measured_weight_fit_error        # 0.0 when the basis is exact
w.edge_weights()                   # float32[total_edge_count], canonical order
```

Correction slots are allocated from what the basis actually missed, so an exactly-fitted
model has `sparse_delta_capacity == 0` and `accumulate_edge_delta` has nowhere to write.

See [API_REFERENCE.md](API_REFERENCE.md) for the rest.

## What the engine refuses

Supported: GLIF1–GLIF5, `iafCell`, current-based synapses (`alphaCurrentSynapse` and
friends), `pulseGenerator` and `spikeArray` stimulus, arbitrary connectivity.

Anything else fails at construction rather than simulating incorrectly:

```python
spc.SpikeEngine("model_with_expTwoSynapse.xml", [[1], []], "conductanceSynapse")
# RuntimeError: dynamics_codegen: 'expTwoSynapse' is conductance-based;
#               Phase 1 simulates current-based synapses only
```

## Demos and tutorial

```bash
make demos                                  # runs all four, renders videos
python3 examples/demo_glif5_torus.py        # or just one
```

`examples/notebooks/01_tutorial_getting_started.py` is the tutorial in jupytext format:
`jupytext --to ipynb 01_tutorial_getting_started.py`.

## Logging

```python
spc.set_log_level("warn")    # trace debug info warn error critical off
```

## Tests

```bash
make test        # the C++ suite
```
