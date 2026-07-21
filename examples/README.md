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

The two torus examples add:

| Flag | Meaning |
| --- | --- |
| `--side <length>` | torus edge length; the network holds `length²` neurons (default 8 → 64) |
| `--weight <amperes>` | placeholder current delivered per arriving spike (default `2.5e-8`) |

### Where models come from

Two of the examples **generate** their NeuroML (`glif_torus_network.h`) because a 64-neuron torus has
256 connections and hand-typing that is not reasonable. The rest load checked-in files from
`tests/fixtures/nml/` — the same ones that drive both spikecorec and the reference simulator. Point
the fixture-loading examples at your own models with:

```bash
SPIKECOREC_NML_MODEL_DIR=/path/to/models ./build/examples/izhikevich_network_example
```

Either way the document goes through the identical parse → resolve → lower path; nothing is
special-cased for generated input. Each model is a `<name>.nml` content file plus a thin
`<include>`-only `<name>_top.nml` wrapper, because `NML_Parser` XSD-validates only the top-level file
it is handed, and raw LEMS ComponentType declarations do not validate against the NeuroML schema.

### One caveat that applies to every network example

The propagate stage scatters spikes through the k²-tree/WeightMatrix path, but it does **not** yet
invoke a projection's synapse ComponentType per-edge dynamics — that needs a spike-scatter batch
construction subsystem no ticket has built yet. So the value a spike delivers downstream is always an
explicit **placeholder**, never something derived from the model's synapse parameters.

Topology, routing, and timing are real. Synaptic magnitude is stipulated. Each example says which
placeholder it uses and why:

- the torus examples set a calibrated nonzero current (`--weight`, default 25 nA) so the network
  actually propagates — see `glif_torus_network.h` for how that number was arrived at;
- the GLIF E/I example forces it to exactly **zero**, because a plausible-looking nonzero value there
  would invite exactly the wrong conclusion about synapse dynamics that are not running.

The synapse ComponentTypes still parse, resolve, and lower to their own IR programs correctly in
every case — that part is real.

## The examples

Read them in this order — each assumes the previous one.

### `glif3_torus_network_example.cpp` — start here

64 GLIF3 cells on an 8×8 wraparound grid, 256 connections, driven at one corner. Covers parse →
resolve → lower → allocate → assemble → compile → tick, plus `explicitInput`/`pulseGenerator` →
host-precomputed stimulus schedule.

**GLIF3** is leaky integrate-and-fire plus two after-spike currents, `asc1` and `asc2`. Both step
down on every spike and decay back toward zero (time constants 100 ms and 10 ms). Being negative,
they oppose the drive, so each spike makes the next harder to reach — inter-spike intervals lengthen
by ~2.4× over the run. That is spike-frequency adaptation, emerging from the LEMS description with no
hand-written GLIF3 kernel anywhere in the engine.

The output worth looking at is the first-spike-tick grid, which shows the wavefront spreading from
the driven corner in Manhattan distance and meeting itself at the antipode — the torus wraparound
made visible:

```
  First spike tick   (tick per neuron, range 140 … 148)
    . : - = + = - :
    : - = + * + = -
    - = + * # * + =
    = + * # % # * +
    + * # % @ % # *
    = + * # % # * +
    - = + * # * + =
    : - = + * + = -
```

### `glif5_torus_network_example.cpp` — the full GLIF variant

The same torus running GLIF5, which is **GLIF3 + GLIF4**: it keeps the two after-spike currents and
adds an *adaptive threshold*. `theta` is a state variable, not a constant — it relaxes toward
`thetaInf` and jumps by `thetaSpikeAdd` on every spike, so the firing condition is `v > theta` and
the bar itself rises. Over a run it climbs ~12 mV above rest while `asc1` falls to about −330 pA:
two independent adaptation mechanisms working at once, plotted side by side.

For orientation across the family: **GLIF1** is plain leaky integrate-and-fire, **GLIF2** adds a
scaled reset, **GLIF3** adds the after-spike currents, **GLIF4** adds the adaptive threshold instead,
and **GLIF5** combines the last two. All five are linear in their own state variables, so all five
are tagged `closed_form_advanceable`.

### `glif_ei_network_example.cpp` — many populations, many synapse types

**E/I means excitatory/inhibitory** — the standard cortical motif of two neuron populations, one
whose spikes push targets toward firing and one that pushes them away. This model has an `ExcPop` and
an `InhPop` of GLIF1 cells wired by three projections carrying three *different* synapse
ComponentTypes: `expOneSynapse` (conductance-based, vendored), `alphaCurrentSynapse` (current-based,
vendored), and an NMDA-style conductance-based synapse. That mix is the point — it exercises both
synapse flavors the Phase-1 synapse model has to cover.

Shows the flat neuron index space populations share, per-population dispatch, and projections
becoming the k²-tree adjacency the propagate stage scatters through.

This is the one example that forces every weight to exactly zero (see the caveat above), so only the
directly stimulated neuron fires. jNeuroML running this same file — with the real synapse dynamics —
shows downstream propagation the engine cannot yet reproduce. It is also the smallest example here at
5 neurons, because it is a checked-in fixture shared with the reference simulator rather than a
generated network.

### `izhikevich_network_example.cpp` — nonlinear dynamics (Phase 2)

Same pipeline, nonlinear cell. The point is the **active-set × nonlinear rule**: the engine's
active-set optimization can skip a quiet neuron for many ticks and catch it up with one closed-form
step, which is valid *only* for analytically-integrable dynamics. Lowering tags each cell type
`closed_form_advanceable` or not, and this example prints the tag — GLIF prints
`[closed-form advanceable]`, `izhikevich2007Cell` prints `[nonlinear]`. That single bit is what stops
a skip-dispatch fast path from silently corrupting nonlinear cells.

Also shows seeding a two-state-variable cell (`v` and the recovery variable `u`), which makes
`cell_state`'s structure-of-arrays layout concrete.

### `delayed_coupling_example.cpp` — axonal delays

A spike normally arrives with an implicit one-tick latency. Real per-edge delays
(`<connectionWD delay="10ms"/>`) need the **delay ring**: instead of one flat `network_inputs`
array, a ring of `max_delay_ticks + 1` slots, each a full `[neuron_count]` row, with spikes scattered
into the slot for their own arrival tick.

The run measures the delivery offset directly — source fires at tick 150, target receives at tick
250, a measured 100-tick / 10 ms offset matching the declared delay exactly. Also shows the buffer
contract change: in ring mode `network_inputs`, the active-set enqueue arrays and `active_generation`
are all superseded and go unallocated.

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

Lateral coupling defaults to **off** here (`--weight 0`) so the bit-to-spike mapping is unambiguous;
pass `--weight 2.5e-8` to couple the torus and watch activity spread from both input sites.

### `poisson_population_example.cpp` — on-device generators

The **second stimulus path**. A `pulseGenerator` is three constants, so it is precomputed on the
host. A Poisson source has to draw a random number per neuron per tick, so it is lowered to real
device code and dispatched like any other population — `lower_type_library_to_ir(model,
/*lower_inputs_on_device=*/true)`. Also covers seeding `rng_state`, the per-neuron xorshift32 buffer
(must be nonzero: xorshift32 is stuck at zero forever once it reaches zero).

Comparison is necessarily statistical — two PRNG streams never match spike for spike. The run prints
the observed rate against the declared 10 Hz, and separates out the one spurious tick-0 spike per
neuron caused by a documented `OnStart` seeding gap, landing at ~9.8 Hz.

### `stdp_plasticity_example.cpp` — your own ComponentType, and plasticity wiring

The only example that writes its own NeuroML rather than loading a fixture, because the vendored
`stdpSynapse` is marked *"EXAMPLE NOT YET WORKING!!!!"* and declares none of the STDP parameters. So
it doubles as the example for **bringing hand-authored LEMS dynamics into the engine**.

STDP detection is structural — any Synapse ComponentType baking all four of
`tauPlus`/`tauMinus`/`aPlus`/`aMinus` counts — which keeps codegen generic. The file documents two
real limitations of the target kernel that the wiring maps onto rather than hides: the kernel has
exactly one dial (`learning_rate`, fed from `aMinus`), and its update is always a depression, not the
two-sided window textbook STDP describes.

## Shared code

`nml_pipeline_support.h` holds only what is identical across examples: model loading, runtime buffer
setup, `cell_state` addressing, and console output. Model-specific logic — stimulus, initial state,
what gets measured — stays inline in each example, because that is the part worth reading.

`glif_torus_network.h` builds on it with the network generator: the GLIF3/GLIF5 ComponentType
declarations (reused verbatim from the lowering tests' fixtures), `square_torus` → `<connection>`
rendering, per-variant state-variable slot lookup, initial-state seeding, and the torus grid plot.

Three things worth knowing before writing your own driver:

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

## Python

`demo_script.py` predates this work and drives the original hardcoded LIF engine through the
pybind11 bindings (`make python`), not the codegen path. There is no Python surface for the NML
pipeline yet.
