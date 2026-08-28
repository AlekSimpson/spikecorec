# Examples

Every example here builds a model as NeuroML/LEMS, hands it to `SpikeEngine`, and runs it.
Nothing about the dynamics is compiled into the engine: the tick kernel is generated from
the model's own ComponentTypes at construction and compiled by the backend.

```bash
make examples          # builds every .cpp in this directory into build/examples/
```

The `.h` files beside them are shared model generators, not programs — the tests include
the same headers, so what the tests assert is what the demos show.

---

## `iaf_single_cell_example`

One `iafCell` under a constant 90 pA current, checked against the closed-form solution of
the same equation.

```bash
./build/examples/iaf_single_cell_example
```

```
analytic interspike gap  : 0.040738 s
measured interspike gap  : 0.040700 s
relative error           : 0.0924 %
```

The 0.09% is explicit-Euler discretisation at dt = 0.05 ms plus one tick of
threshold crossing — not a modelling error.

---

## `glif_family_example`

One cell of each GLIF type under the same 500 pA step (2.5× rheobase), so their
adaptation can be compared directly. This is also the model that puts five different cell
types through the generated kernel's per-type dispatch at once.

```bash
./build/examples/glif_family_example
```

```
type     spikes    first ISI     last ISI   adaptation
GLIF1        50      10.05 ms      10.05 ms      1.00x
GLIF2        50      10.05 ms      10.05 ms      1.00x
GLIF3        17      14.05 ms      34.15 ms      2.43x
GLIF4        32      11.50 ms      15.95 ms      1.39x
GLIF5        15      16.40 ms      38.50 ms      2.35x
```

GLIF1's interval is checkable by hand: with C = 100 pF and gL = 10 nS the membrane time
constant is 10 ms and the drive settles toward −20 mV, so charging from the −70 mV reset
to the −50 mV threshold takes `10 ms · ln(50/30)` = 5.11 ms, and the 5 ms refractory
period brings it to 10.11 ms against 10.05 ms measured.

GLIF3 (after-spike currents), GLIF4 (adapting threshold) and GLIF5 (both) slow down over
the step. GLIF1 and GLIF2 do not, because nothing in them accumulates across spikes.

---

## `glif_network_example` — the one with videos

A recurrent balanced network for each GLIF type: 400 excitatory + 100 inhibitory cells,
10,000 connections with 1–4 ms delays, `alphaCurrentSynapse` coupling, two seconds.

```bash
./build/examples/glif_network_example [output_directory]     # defaults to build/
```

```
type     spikes  rate Hz  alive %  peak sync     mid Hz     end Hz
GLIF1     23020    23.02     94.4      16.0%      23.13      22.78
GLIF2     22934    22.93     94.0      16.0%      22.84      22.74
GLIF3      5507     5.51     99.8      16.0%       5.42       5.49
GLIF4     14845    14.85     99.6      16.0%      14.91      15.10
GLIF5      5169     5.17     99.8      16.0%       4.82       3.90
```

The five differ because their cell dynamics differ, and it shows at the network level:
GLIF1 and GLIF2 fire asynchronously at 23 Hz, while the adapting types settle far lower
and GLIF3/GLIF5 develop a population rhythm near 10 Hz — their after-spike currents
recover on a shared time constant and pull the network into bands.

Each run writes three things per network into the output directory:

| file | contents |
|---|---|
| `glif<N>_network_spikes.dat` | every spike, as `time  neuron` |
| `glif<N>_network_traces.dat` | six membrane traces, as a time-first column matrix |
| `glif<N>_network_membrane.spire` | every neuron's membrane potential every 10 ticks |

### Rendering a video

```bash
python3 examples/demo_glif1_torus.py    # runs, records and renders in one go
```

Writes `build/glif3_network_membrane.mp4`: the population as a grid coloured by membrane
potential, six individual traces, and the spike raster, with a marker sweeping all three
in step. It finds the matching `_spikes.dat` automatically.

Needs `numpy` and `matplotlib`, plus `ffmpeg` on PATH for mp4 (`--output x.gif` uses
Pillow instead). On this machine the interpreter with both is `/opt/homebrew/bin/python3.10`.

Useful flags: `--max-frames` subsamples (a 2 s run recorded every 10 ticks is 2,000 frames
and 600 at 30 fps is already 20 seconds of watching), `--frames-per-second`, `--title`,
and `--spikes` to point at a raster explicitly.

---

## `balanced_network_example`

The same balanced-network structure built from the standard library's `iafCell` rather
than a GLIF type — 1,000 cells, 20,000 edges, one second, 18.12 Hz.

```bash
./build/examples/balanced_network_example
```

---

## What the models may use

Phase 1 simulates GLIF and integrate-and-fire cells with current-based synapses. Anything
outside that is refused at load time, naming the ComponentType and the phase it belongs
to, rather than loading and quietly simulating something else:

- conductance-based synapses (`expOneSynapse` and the rest of that family)
- plasticity and block mechanisms composed into a synapse
- `ConditionalDerivedVariable` / `Case`
- `random()`
- spike-train stimulus (`spikeArray`, `timedSynapticInput`) — a train becomes current only
  through a synapse, and that path is not wired up
- any regime shape other than the active/refractory pair every GLIF cell declares, which
  the engine lowers to a comparison against when the cell last fired
