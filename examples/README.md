# spikecorec examples

Runnable walkthroughs of the NeuroML → GPU codegen path (epic #1). Each program takes a real
NeuroML/LEMS model, runs it through the full pipeline, and prints what came out at every stage.

These mirror the validation suite in `tests/` — same models, same pipeline, minus the assertions.
Where a test asserts a number, an example prints it and explains what it means. If you want the
numeric comparisons against jNeuroML/pyNeuroML reference data, read
`tests/exit_model_validation_tests.cpp`.

## Build and run

```bash
make            # build the engine library first (Metal on macOS, CUDA elsewhere)
make examples   # → build/examples/

./build/examples/glif3_torus_network_example
```

Every example accepts these flags:

| Flag | Meaning |
| --- | --- |
| `--ticks <count>` | number of ticks to simulate |
| `--dt <seconds>` | timestep, in seconds (default `0.0001`, i.e. 0.1 ms) |
| `--print-ir` | dump each ComponentType's lowered `.alloc`/`.tick` IR |
| `--verbose` | show engine log output (quieted to errors by default) |

The torus/discrete-spike-input examples add:

| Flag | Meaning |
| --- | --- |
| `--side <length>` | torus edge length; the network holds `length²` neurons (default 8 → 64) |
| `--gbase <siemens>` | the torus's real `expOneSynapse` conductance amplitude, e.g. `10nS` (default `10nS`/`15nS` — see below) |
| `--record-dir <path>` | where the GLIF torus examples write their `.spire` pairs (default `recordings/`) |
| `--no-record` | skip writing `.spire` files entirely |
| `--couple` | `discrete_spike_input_example` only — rebuild the torus WITH its real 4-neighbor wiring (off by default; see that example's own entry) |

### Where models come from

Five of the examples (one per GLIF variant, plus `discrete_spike_input_example`) **generate**
their NeuroML (`glif_torus_network.h`) because a 64-neuron torus has 256 connections and
hand-typing that is not reasonable. The rest load checked-in files from `tests/fixtures/nml/` — the
same ones that drive both spikecorec and the reference simulator. Point the fixture-loading
examples at your own models with:

```bash
SPIKECOREC_NML_MODEL_DIR=/path/to/models ./build/examples/izhikevich_network_example
```

Either way the document goes through the identical parse → resolve → lower path; nothing is
special-cased for generated input. Each model is a `<name>.nml` content file plus a thin
`<include>`-only `<name>_top.nml` wrapper, because `NML_Parser` XSD-validates only the top-level file
it is handed, and raw LEMS ComponentType declarations do not validate against the NeuroML schema.

### Real synaptic propagation, not a placeholder (ticket #131)

Earlier revisions of this suite scattered spikes through the k²-tree/`WeightMatrix` path with an
explicit **placeholder** current (`--weight`), because `AssembledModel`'s propagate stage did not yet
invoke a projection's synapse ComponentType's own dynamics. That subsystem now exists: whenever a
model declares a real projection, `AssembledModel` dispatches that projection's actual synapse type
automatically — no opt-in needed, and a `WeightMatrix::set_constant_weight` call has no effect on
such a model (its contribution is forced to zero, since the real per-edge dispatch already supplies
it). Concretely:

- the torus examples (`glif1_torus_network_example`, `glif2_torus_network_example`,
  `glif3_torus_network_example`, `glif4_torus_network_example`, `glif5_torus_network_example`,
  `discrete_spike_input_example --couple`) are wired through a real, vendored `expOneSynapse` —
  `--gbase` sets its actual conductance amplitude, not a stand-in current;
- `glif_ei_network_example` shows real `ExcPop[0] → InhPop[0]` propagation through a real
  `expOneSynapse`, though one of its three synapse types (`alphaCurrentSynapse`, also used by
  `izhikevich_network_example`) still doesn't propagate — a real, orthogonal, documented lowering
  gap (its coupled `I`/`J` TimeDerivative isn't a shape `synapse_lowering.cpp` recognizes yet), not a
  placeholder;
- `delayed_coupling_example` (ring mode) and `poisson_population_example` (no projections at all) are
  the two remaining cases where a constant scattered weight is still the real mechanism — see each
  file's own header comment for why.

Every example's own header comment states which of these applies to it. Topology, routing, and
timing were always real; what changed is that synaptic *magnitude* now is too, wherever a real
projection exists.

### Recording network activity to `.spire`, and rendering it

Every GLIF torus example (all five variants) drives a `NetworkActivityRecorder`
(`nml_pipeline_support.h`) that writes two parallel `.spire` files per run, one frame per tick each:
whole-population membrane potential, and a 0.0/1.0 spike-raster mask (matching
`include/spikecorec/nml/output_recording.h`'s own `RecordingSourceKind::SpikeRaster` convention).
This is the caller `master_kernel.h`'s own doc comment says recording needs ("not wired to
SimulationRecorder ... a caller must drive it") — these examples are that caller.

Render either recording with `render_spire_video.py`:

```bash
pip install -e ".[examples]"     # numpy + matplotlib, if not already installed
make python                      # builds the spikecorec extension render_spire_video.py imports

./build/examples/glif3_torus_network_example --ticks 5000
./examples/render_spire_video.py recordings/glif3_torus_spikes.spire \
    --side 8 --membrane recordings/glif3_torus_membrane.spire
```

This produces `recordings/glif3_torus_spikes.mp4` (pass `--output foo.gif` for a gif instead, no
`ffmpeg` required for that path): every neuron rendered at its real `(row, column)` torus position —
the SAME `row * side_length + column` flat-index convention `glif_torus_network.h` generates the
network with — membrane potential as a per-tick heatmap, spikes overlaid as bright markers on the
tick they occur. The GLIF3 run's own first-spike-tick gradient (see that example's own entry below)
is what you are watching propagate frame by frame: a wavefront spreading in Manhattan distance from
the driven corner and meeting itself at the antipode — genuine whole-network activity, not just the
directly-stimulated cell. See `render_spire_video.py`'s own module docstring for every flag.

## The examples

Read them in this order — each assumes the previous one.

### `glif3_torus_network_example.cpp` — start here

64 GLIF3 cells on an 8×8 wraparound grid, 256 connections, driven at one corner. Covers parse →
resolve → lower → allocate → assemble → compile → tick, `explicitInput`/`pulseGenerator` →
host-precomputed stimulus schedule, real per-edge `expOneSynapse` dispatch (ticket #131), and
recording the whole run to `.spire` (ticket #138) for `render_spire_video.py`.

**GLIF3** is leaky integrate-and-fire plus two after-spike currents, `asc1` and `asc2`. Both step
down on every spike and decay back toward zero (time constants 100 ms and 10 ms). Being negative,
they oppose the drive, so each spike makes the next harder to reach — inter-spike intervals lengthen
over the run. That is spike-frequency adaptation, emerging from the LEMS description with no
hand-written GLIF3 kernel anywhere in the engine.

The output worth looking at is the first-spike-tick grid, which shows the wavefront spreading from
the driven corner in Manhattan distance and meeting itself at the antipode — the torus wraparound
made visible (default `--gbase 10nS`, empirically the "knee" where the whole 64-neuron torus
recruits with a clean spatial gradient rather than staying local or saturating instantly):

```
  First spike tick   (tick per neuron, range 140 … 557)
    . - + # % # + -
    - = + # % # + =
    + + * % % % * +
    # # % % % % % #
    % % % % @ % % %
    # # % % % % % #
    + + * % % % * +
    - = + # % # + =
```

### `glif5_torus_network_example.cpp` — the full GLIF variant

The same torus running GLIF5, which is **GLIF3 + GLIF4**: it keeps the two after-spike currents and
adds an *adaptive threshold*. `theta` is a state variable, not a constant — it relaxes toward
`thetaInf` and jumps by `thetaSpikeAdd` on every spike, so the firing condition is `v > theta` and
the bar itself rises. Because reaching threshold is harder than GLIF3's fixed `vth`, this example's
default `--gbase` (15nS) is higher than GLIF3's.

For orientation across the family: **GLIF1** is plain leaky integrate-and-fire, **GLIF2** adds a
scaled reset, **GLIF3** adds the after-spike currents, **GLIF4** adds the adaptive threshold instead,
and **GLIF5** combines the last two. All five are linear in their own state variables, so all five
are tagged `closed_form_advanceable`. Also records to `.spire`, same as the GLIF3 example.

### `glif1_torus_network_example.cpp` — plain leaky integrate-and-fire

The same torus running **GLIF1**, the simplest variant: one state variable (`v`), a fixed threshold
`vth`, and a flat reset to `vreset` — no after-spike currents, no adaptive threshold, just the
refractory-regime timer every GLIF variant shares. There is no spike-frequency adaptation here:
under a constant current the inter-spike intervals settle to a constant steady-state rate rather
than continuously lengthening, the baseline the other four variants' adaptation mechanisms are
contrasted against. A separate, hand-typed GLIF1 network already exists
(`glif_ei_network_example.cpp`, below); this is the GLIF1 counterpart to the GLIF3/GLIF5 torus
walkthroughs instead — a single, homogeneous, generated 64-neuron network with real per-edge
synaptic propagation and its own `.spire` recording.

### `glif2_torus_network_example.cpp` — scaled reset

The same torus running **GLIF2**: GLIF1 plus a reset rule that scales with how far past `vth` the
membrane potential overshot on the triggering tick — `v <- vreset + resetScale * (v - vth)` rather
than GLIF1's flat `v <- vreset`. This example uses `resetScale=0.4` (the more interesting,
non-degenerate case — `resetScale=0` would be bit-exactly GLIF1's own reset) and prints `v` right
after every spike, showing it land measurably above `vreset` rather than snapping back to it — the
same flat-vs-scaled contrast `tests/end_to_end_network_tests.cpp`'s own
`glif2_ring_network_discrete_spike_array_smallest_anchor` asserts.

### `glif4_torus_network_example.cpp` — adaptive threshold, no after-spike currents

The same torus running **GLIF4**: leaky integrate-and-fire with an adaptive threshold instead of
after-spike currents. `theta` is a state variable that relaxes toward `thetaInf` and jumps by
`thetaSpikeAdd` on every spike, so the firing condition is `v > theta` rather than `v > vth` — read
`glif5_torus_network_example` afterward to see the SAME threshold mechanism combined with GLIF3's
after-spike currents. This example isolates the threshold mechanism on its own, plotting `v` and
`theta` together the same way the GLIF5 example does, and prints the matching adaptation signature
(`tests/end_to_end_network_tests.cpp`'s own `glif4_ring_network_discrete_spike_array_medium_anchor`
asserts `theta` strictly increasing across the first few spikes) — lengthening inter-spike intervals
and a `theta` that climbs well above its resting `thetaInf` under sustained drive.

### `glif_ei_network_example.cpp` — many populations, many synapse types

**E/I means excitatory/inhibitory** — the standard cortical motif of two neuron populations, one
whose spikes push targets toward firing and one that pushes them away. This model has an `ExcPop` and
an `InhPop` of GLIF1 cells wired by three projections carrying three *different* synapse
ComponentTypes: `expOneSynapse` (conductance-based, vendored), `alphaCurrentSynapse` (current-based,
vendored), and an NMDA-style conductance-based synapse. That mix is the point — it exercises both
synapse flavors the Phase-1 synapse model has to cover.

Shows the flat neuron index space populations share, per-population dispatch, and projections
becoming the k²-tree adjacency the propagate stage scatters through. `ExcPop[0] → InhPop[0]` fires
via a real `expOneSynapse` per-edge conductance — genuine propagation beyond the directly-stimulated
cell. `ExcPop[2]` stays silent, a real, orthogonal, and separate gap this example's own header
comment documents (`alphaCurrentSynapse`'s coupled `I`/`J` TimeDerivative isn't a shape
`synapse_lowering.cpp` recognizes yet). It is also the smallest network here at 5 neurons, because it
is a checked-in fixture shared with the reference simulator rather than a generated network — the
torus examples (and their recorded/rendered output) are the primary "whole-network propagation"
story this suite tells.

### `izhikevich_network_example.cpp` — nonlinear dynamics (Phase 2)

Same pipeline, nonlinear cell. The point is the **active-set × nonlinear rule**: the engine's
active-set optimization can skip a quiet neuron for many ticks and catch it up with one closed-form
step, which is valid *only* for analytically-integrable dynamics. Lowering tags each cell type
`closed_form_advanceable` or not, and this example prints the tag — GLIF prints
`[closed-form advanceable]`, `izhikevich2007Cell` prints `[nonlinear]`. That single bit is what stops
a skip-dispatch fast path from silently corrupting nonlinear cells.

Also shows seeding a two-state-variable cell (`v` and the recovery variable `u`), which makes
`cell_state`'s structure-of-arrays layout concrete. `TargetPop` stays silent here — the same
`alphaCurrentSynapse` gap `glif_ei_network_example` documents, not a bug specific to this file.

### `delayed_coupling_example.cpp` — axonal delays

A spike normally arrives with an implicit one-tick latency. Real per-edge delays
(`<connectionWD delay="10ms"/>`) need the **delay ring**: instead of one flat `network_inputs`
array, a ring of `max_delay_ticks + 1` slots, each a full `[neuron_count]` row, with spikes scattered
into the slot for their own arrival tick.

The run measures the delivery offset directly — source fires, target receives exactly the declared
delay later (10ms in the checked-in fixture). Also shows the buffer contract change: in ring mode
`network_inputs`, the active-set enqueue arrays and `active_generation` are all superseded and go
unallocated. Ring mode disables ticket #131's real per-edge synapse dispatch entirely (the two have
not been integrated with each other), so this is one of the two examples where a constant scattered
weight is still the real propagation mechanism, not documentation of a no-op.

### `discrete_spike_input_example.cpp` — driving from a literal 0/1 array

The **third** way to drive a model, after NeuroML-declared stimulus and on-device generators: a
host-provided bit sequence, one value per tick per input neuron, with no NeuroML generator
ComponentType behind it at all.

```
[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, ...]
```

Use it when the input comes from outside the model entirely — recorded data, an upstream encoder, a
dataset, another simulation. `DiscreteSpikeInputSchedule` (`nml/discrete_spike_input.h`) is
deliberately thin, a struct plus one method, because `step_tick` already reads `network_inputs` as an
ordinary writable array every tick.

Pass your own array with `--pattern 0001001010` (brackets and commas are ignored, so
`[0,0,0,1,0,0,1,0,1,0]` works too) and set its timescale with `--bit-ticks`. The run feeds the same
pattern to two neurons at two rates and prints input bits against output spikes, aligned:

```
  slow — neuron 0, one pattern element every 100 ticks
    input bits    │      |     |    |         |      |   |          |     |    |
    output spikes │      |     |    |         |      |   |          |     |    |
                   12 bits in → 12 spikes out
                   ✓ every bit produced exactly one spike
```

The fast stream loses about half its bits, and the run separates the two causes rather than lumping
them: bits that hit the 5 ms refractory period, versus bits that landed on a free cell but fell short
of threshold because GLIF3's after-spike currents were still holding `v` below rest. They call for
opposite fixes — slow the input down, or raise `--amplitude`.

**Lateral coupling is off by default** — this generates the SAME torus as the GLIF3 example but with
`include_lateral_connections=false` (`glif_torus_network.h`), i.e. no `<projection>` at all, so the
bit-to-spike mapping above is exact and unambiguous (ticket #131 made real per-edge dispatch
unconditional whenever a projection exists, so the old "`--weight 0` disables coupling" trick no
longer works). Pass `--couple` to rebuild the SAME network WITH its real 4-neighbor
`expOneSynapse` wiring instead, and watch activity spread from both input sites.

### `poisson_population_example.cpp` — on-device generators

The **second stimulus path**. A `pulseGenerator` is three constants, so it is precomputed on the
host. A Poisson source has to draw a random number per neuron per tick, so it is lowered to real
device code and dispatched like any other population — `lower_type_library_to_ir(model,
/*lower_inputs_on_device=*/true)`. Also covers seeding `rng_state`, the per-neuron xorshift32 buffer
(must be nonzero: xorshift32 is stuck at zero forever once it reaches zero).

Comparison is necessarily statistical — two PRNG streams never match spike for spike. The run prints
the observed rate against the declared 10 Hz, and separates out the one spurious tick-0 spike per
neuron caused by a documented `OnStart` seeding gap, landing at ~9.8 Hz. This model declares no
projections at all, so ticket #131 never applies here — a trivial placeholder ring adjacency exists
purely to satisfy `WeightMatrix`'s constructor.

### `stdp_plasticity_example.cpp` — your own ComponentType, and plasticity wiring

The only example that writes its own NeuroML rather than loading a fixture, because the vendored
`stdpSynapse` is marked *"EXAMPLE NOT YET WORKING!!!!"* and declares none of the STDP parameters. So
it doubles as the example for **bringing hand-authored LEMS dynamics into the engine**.

STDP detection is structural — any Synapse ComponentType baking all four of
`tauPlus`/`tauMinus`/`aPlus`/`aMinus` counts — which keeps codegen generic. The file documents two
real limitations of the target kernel that the wiring maps onto rather than hides: the kernel has
exactly one dial (`learning_rate`, fed from `aMinus`), and its update is always a depression, not the
two-sided window textbook STDP describes.

Two wiring targets are exercised: the original `SpikeEngine` overload (Phase-1's own SC-11 API), and
`AssembledModel`'s own overload (ticket #132). The latter demonstrates `AssembledModel::
enable_plasticity`'s documented incompatibility guard directly — it throws when called on a model
with real per-edge synapse dispatch active (ticket #131), and succeeds once the SAME structural
model is instead built with a real per-edge delay and `enable_delay_ring=true` (which disables that
dispatch). See `tests/assembled_model_plasticity_tests.cpp` for a full run that actually steps this
combination and measures a weight depress.

## Shared code

`nml_pipeline_support.h` holds only what is identical across examples: model loading, runtime buffer
setup, `cell_state` addressing, console output, and `NetworkActivityRecorder` (ticket #138's
per-tick `.spire` recording driver). Model-specific logic — stimulus, initial state, what gets
measured — stays inline in each example, because that is the part worth reading.

`glif_torus_network.h` builds on it with the network generator: all five GLIF ComponentType
declarations (reused verbatim from the lowering tests' fixtures), `square_torus` → `<connection>`/
real `expOneSynapse` rendering (`include_lateral_connections` toggles whether a projection is
emitted at all), per-variant state-variable slot lookup, initial-state seeding, and the torus grid
plot.

Four things worth knowing before writing your own driver:

- **`GpuContextScope` must be the first local in `main`.** `release_gpu_resources()` frees every
  buffer the backend handed out, so it has to run *after* the last object owning one
  (`ModelAllocation`, `WeightMatrix`, `AssembledModel`, …). Locals destruct in reverse declaration
  order, so declaring the guard first makes it destruct last. Calling `release_gpu_resources()` by
  hand at the end of `main` instead frees those buffers while their owners are still alive, and the
  destructors then double-free — a segfault at exit, after a run that otherwise looked fine.
- **`allocate_model` does not apply `OnStart`.** It zero-initializes `cell_state`; seeding initial
  state is the caller's job. `seed_membrane_potentials_from_resting_parameter` does this for GLIF's
  `v = EL`; other cell types need their own. This bites hardest on GLIF5: leave `theta` at its
  zero-initialized value and it sits *above* the −50 mV threshold, so no neuron ever fires and the
  network looks silently dead rather than erroring.
- **Include the backend header first.** Every example opens with the same guarded
  `<Metal/Metal.hpp>` / `<cuda_runtime.h>` block the test files use. metal-cpp declares its own
  `NS::String`, which collides with the engine's `String` alias if `Metal.hpp` is pulled in later
  through a transitive include.
- **A real projection means real synapse dispatch, unconditionally.** `WeightMatrix::
  set_constant_weight` no longer has any effect once `AssembledModel` is built from a model with
  `model.projections` non-empty (ticket #131) — see "Real synaptic propagation" above before
  copying a `set_constant_weight` call into a new example and expecting it to do anything.

## Python

`demo_script.py` predates this work and drives the original hardcoded LIF engine through the
pybind11 bindings (`make python`), not the codegen path. There is no other Python surface for the
NML pipeline besides `render_spire_video.py` (ticket #138), which only reads a `.spire` recording
back via `spikecorec.read_spire_recording()` — see that script's own module docstring, and "Recording
network activity to `.spire`, and rendering it" above, for how to run it.
