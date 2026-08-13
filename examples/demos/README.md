# spikecorec demos

The nine programs in `examples/` are each built around one claim, on a network sized to make
that claim cheaply — an 8×8 torus, 64 cells, a number printed at the end. The demos here have a
different job: they exist to be **watched**. Each one runs a GLIF network big enough and long
enough that a spreading wavefront is motion on a screen, records both streams the renderer
needs, and is turned into a playable video by a single `make` target.

Nothing else is different. The demos use the same GLIF `<ComponentType>` declarations, the same
membrane parameters, the same current-based `alphaCurrentSynapse` and the same torus wiring as
the examples, out of `examples/glif_torus_network.h`. A demo is the examples' network at demo
scale, not a second model.

## One command

```bash
make demo-videos DEMO_PYTHON=/path/to/python
```

That builds the engine, builds the demos, runs all five GLIF variants and renders all five
videos, from scratch. It is the command to re-run whenever the model underneath changes.

`DEMO_PYTHON` has to point at an interpreter with **numpy** and **matplotlib** — the renderer's
dependencies, which the C++/Metal/CUDA build itself does not have. Any virtualenv will do:

```bash
python3 -m venv /tmp/spikecorec_render
/tmp/spikecorec_render/bin/pip install numpy matplotlib
make demo-videos DEMO_PYTHON=/tmp/spikecorec_render/bin/python
```

Plain `make demo-videos` works wherever `python3` already has both.

Two smaller targets sit underneath it:

| Target | Does |
| --- | --- |
| `make demos` | builds `examples/demos/*.cpp` → `build/demos/` |
| `make run-demos` | builds **and runs** every variant, recording but not rendering (seconds) |
| `make demo-videos` | the above, plus a rendered video per variant (a few minutes) |

## Where the output goes

```
build/demo_videos/
├── glif1_demo.gif                        ← the videos
├── glif2_demo.gif
├── …
├── glif1/
│   ├── glif1_demo_membrane.spire         ← <OutputFile>, membrane potential per neuron per tick
│   ├── glif1_demo_spikes.spire           ← <EventOutputFile>, the spike raster
│   └── run.log                           ← what the demo printed
└── …
```

Everything is under `build/`, which `.gitignore` already covers — **no video or recording is ever
committed**. The `.spire` pair is kept next to each video so a run can be re-rendered (different
frame rate, colormap, marker size) without re-simulating:

```bash
./examples/render_spire_video.py build/demo_videos/glif3/glif3_demo_spikes.spire \
    --membrane build/demo_videos/glif3/glif3_demo_membrane.spire \
    --side 48 --output glif3_slow.gif --fps 10
```

The demos are **not** part of `make check`. `check` builds and runs every `examples/*.cpp` on
every invocation, and a demo renders several hundred video frames through matplotlib — far too
slow to sit in a gate. `examples/demos/*.cpp` is deliberately outside `EX_SRCS`' wildcard for
that reason.

## The programs

### `glif_family_demo` — the GLIF family at demo scale

One program, `--variant glif1 | glif2 | glif3 | glif4 | glif5`. This is one binary rather than
the five near-identical files `examples/` uses for the same five variants: at demo scale the only
thing that differs between them is the enum and the closing paragraph, so five copies of the same
`main()` would be five things to keep in step through the synapse rework that is coming.

**The network.** 48 × 48 = 2304 cells on a wraparound grid, each wired to its four neighbours
(9216 connections), one `alphaCurrentSynapse` per projection, one cell in the **middle** driven
by a current step, 250 ms at a 0.1 ms step.

- **48 on a side** is chosen against the video, not against the engine. At 48 the wavefront needs
  on the order of a hundred simulated milliseconds to reach the far side, which reads as motion,
  and one neuron is still several pixels wide in the renderer's fixed 600-pixel frame. A 128-wide
  grid runs just as happily and renders as noise.
- **Driven from the middle**, where the examples drive a corner. Their first-spike ASCII grid
  reads most clearly with the wavefront starting in one; a video reads better the other way
  round, as a front expanding symmetrically in all four directions at once, with the torus
  wraparound showing up as four fronts converging on the corners.
- **The drive switches on at 5 ms**, where the examples wait 20 ms. At demo scale that wait is
  200 ticks of a completely dead recording at the head of the video.
- **250 ms** is long enough for the front to cross the population, wrap and collide with itself,
  with a stretch of settled network-wide firing after it.

**What you see.** The driven cell fires periodically, so each of its spikes launches its own
front: the video is a set of concentric rings expanding out of the middle, filling the grid,
wrapping at the edges and meeting. The membrane-potential heatmap behind the spike markers is
what makes the front continuous rather than a scatter of dots — a cell being carried towards
threshold is already bright before it fires.

**What it prints.** Every run reports the three things that separate a demo that works from one
whose video shows nothing:

- the total spike count and **how many distinct neurons fired**, against the population size;
- the **wavefront spread** — the tick of the first spike anywhere, and the tick at which the last
  neuron to join fired for the first time, which is how long the front took to reach everything;
- an **activity timeline**, spikes per equal slice of the run, so activity confined to one instant
  is visible as one bar rather than being averaged away.

Then the variant's own signature, which is the one thing that differs between the five:

| Variant | What the run reports |
| --- | --- |
| `glif1` | the driven cell's inter-spike intervals, which settle to a constant — it has no adaptation mechanism at all, and is the baseline the other four are read against |
| `glif2` | how far above `vreset` the membrane lands after each reset. GLIF1 would print zero every time |
| `glif3` | both after-spike currents at the end of the run — hyperpolarizing, so each spike makes the next harder to reach |
| `glif4` | the adaptive threshold `theta` at the end of the run, against its resting `thetaInf` |
| `glif5` | GLIF3's after-spike currents *and* GLIF4's threshold, both live at once |

The totals fall as the adaptation gets stronger, which is the family's whole point visible in one
number per run — read them off the five `run.log`s rather than from this file, because they move
with the synapse model.

## A note on the numbers

Current-based synapse dynamics are being reworked: a spike will deliver its whole weighted output
into a single target tick rather than a time course. Spike counts, firing rates and the video
frames themselves all move when that lands, which is exactly why `make demo-videos` regenerates
everything from scratch and why no specific figure is written down here. Re-run the target and
read the fresh `run.log`s.
