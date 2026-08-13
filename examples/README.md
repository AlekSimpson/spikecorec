# spikecorec examples

Runnable programs that drive the NeuroML → GPU path end to end. Each one hands `SpikeEngine` a
NeuroML/LEMS file, steps it, and reports what came out.

The engine surface is small enough to state in full:

```cpp
SpikeEngine engine(model_file_path, /*enable_hebbian_learning=*/false);
for (s64 tick = 0; tick < engine.lifetime; ++tick) engine.step_simulation(tick);
engine.shutdown();
```

Everything else — the cell dynamics, the wiring, the stimulus, the recordings — lives in the
model. What each example adds is only the reading: which state variable it watches, what it
prints, and what that number means.

## Build and run

```bash
make            # build the engine library first (Metal on macOS, CUDA elsewhere)
make examples   # → build/examples/

./build/examples/glif1_torus_network_example
```

Flags, all optional:

| Flag | Meaning |
| --- | --- |
| `--ticks <count>` | run this many ticks instead of the model's own `<Simulation length=…>` |
| `--synapse-current <nA>` | peak current one arriving spike injects (see "Synaptic strength" below) |
| `--side <length>` | torus edge length; the network holds `length²` neurons (default 8 → 64) |
| `--record-dir <path>` | where `.spire` recordings are written (default `recordings/`) |
| `--pattern <bits>` | `discrete_spike_input_example` only: the 0/1 sequence to drive with |
| `--verbose` | show engine log output (quieted to warnings by default) |

## Checking that they all still work

```bash
make check          # the test suite AND every example, built and run
make run-examples   # just the examples
```

`make test` compiles and runs `tests/` and **nothing under `examples/`** — not one example is so
much as compiled by it. That is not hypothetical: when the engine started refusing
conductance-based synapses, seven of the nine examples here stopped constructing an engine at
all while `make test` went on reporting a full pass.

`make check` closes that hole. It runs the suite, then builds and runs every example in
`examples/*.cpp`, and fails if any of them exits non-zero — printing the tail of the offending
run. Each example is run with `--record-dir build/example_runs/<name>/`, so a check writes
nothing into the working tree, and its console output is kept next to it as
`build/example_runs/<name>.log`. The whole thing takes about twenty seconds from a warm build.

## The examples

Read them in this order.

### `glif1_torus_network_example` — start here

64 GLIF1 cells on an 8×8 wraparound grid, 256 connections, one corner driven by a current
step. Covers the whole path in one program: parse → wire → generate → compile → tick →
record.

GLIF1 is the simplest of the family — one state variable, a fixed threshold, a flat reset, and
a refractory period. It has no adaptation mechanism at all, so its inter-spike intervals settle
to a constant (5.6 ms, against the 5 ms refractory floor). That flat series is the baseline the
other four variants are read against.

The output worth looking at is the first-spike grid, which shows the wavefront spreading from
the driven corner in Manhattan distance and meeting itself at the antipode — the torus
wraparound made visible:

```
  First spike tick   (range 240 ... 449; '.' = never fired)
    @ % # * * * # %
    % % # * + * # %
    # # * + + + * #
    * * + + = + + *
    * + + = - = + +
    * * + + = + + *
    # # * + + + * #
    % % # * + * # %
```

All 64 cells fire, and the run reports 1908 spikes over 2000 ticks. The wavefront takes 209
ticks (20.9 ms) to cross the torus, about 2.6 ms per hop: 1 ms of declared conduction delay
plus the time the alpha current needs to carry a resting neighbour the 20 mV to threshold.

### `glif2_torus_network_example` — the scaled reset

The same torus, with GLIF1's flat reset replaced by `v <- vreset + resetScale * (v - vth)`.
Prints how far above `vreset` the driven corner lands after each spike — a few hundred
microvolts to about 2 mV, varying with how much synaptic current happened to be flowing on the
crossing tick. GLIF1 would print zero every time.

### `glif3_torus_network_example` — after-spike currents, and adaptation

The same torus plus two after-spike currents, `asc1` and `asc2`. Both step down on every spike
(they are hyperpolarizing) and decay back towards zero with time constants of 100 ms and 10 ms,
so each spike makes the next harder to reach. The driven corner's inter-spike intervals lengthen
— 5.7 ms between the first pair, 5.9 ms between the last, climbing through 5.6 → 5.7 → 5.8 →
5.9 — where GLIF1's sit flat at 5.6 ms, and `asc1` has reached −1425 pA by the end of the run
against a 600 pA drive. That is spike-frequency adaptation, emerging from the LEMS description
with no hand-written GLIF3 kernel anywhere in the engine.

The shift is small because the torus is recurrently driven: every cell has four neighbours
firing into it, so the intervals sit close to the 5 ms refractory floor and the after-spike
currents can only push them so far off it. `asc1`'s magnitude is the mechanism seen directly,
without the floor in the way.

### `glif4_torus_network_example` — the adaptive threshold

The same torus with an adaptive threshold instead of after-spike currents. `theta` is a state
variable that relaxes towards `thetaInf` and jumps on every spike, so the firing condition is
`v > theta` and the bar itself rises. Prints `theta` climbing from −47.0 mV to −29.6 mV over the
corner's first twelve spikes, against a resting `thetaInf` of −50 mV, with the intervals
lengthening 5.7 → 6.1 ms as it goes.

### `glif5_torus_network_example` — the full variant

GLIF3 + GLIF4: after-spike currents *and* an adaptive threshold, both live at once. Prints both
halves at the end of the run — `asc1` at −1350 pA and `theta` at −26.1 mV — and its intervals
lengthen furthest of the five, 5.8 → 6.5 ms, because both mechanisms push the same way.

### `glif_ei_network_example` — many populations, one flat index space

Eight excitatory GLIF1 cells and two inhibitory GLIF3 cells. Shows the flat neuron index space
populations share (`ExcPop` is 0..7, `InhPop` is 8..9, nothing is renumbered), `cell_state`
sectioned by *type* rather than by population, and inhibition measurably inhibiting: it builds
the same network twice, once with the inhibitory synapse's `ibase` zeroed and once with it live,
and reports the excitatory population's spike count both times.

```
                            ExcPop spikes   InhPop spikes
    inhibitory ibase      0 nA           176              51
    inhibitory ibase  -0.05 nA            83              31
```

Sign is the only thing that makes a connection inhibitory here, and it lives in the synapse: both
projections run an `alphaCurrentSynapse`, and the inhibitory one's `ibase` is negative, so the
current it injects drives `v` away from threshold. That is a *current-based* inhibitory synapse
and not the conductance-based inhibition of a real cortical model, where `i = g * (erev - v)`
with `erev` below rest — see "Synaptic strength" below for why the engine refuses those rather
than running them as if they were current-based.

−0.05 nA is a twentieth of the excitatory synapse's 1 nA and still roughly halves the excitatory
output, because the inhibitory synapse is the slower of the two (8 ms against 2 ms, so four times
the charge per spike) and because each of the two inhibitory cells projects to all eight
excitatory ones.

### `izhikevich_network_example` — a nonlinear cell, from the standard library

The one example that loads a checked-in fixture unmodified:
`tests/fixtures/nml/izhikevich_network_top.nml`. `izhikevich2007Cell` carries a `v*v` term, so
it has no closed-form trajectory; it also has two state variables (`v` and the recovery variable
`u`); and it needs no inline `<ComponentType>` at all, because it is real vendored NeuroML that
the parser merges from `third_party/neuroml2/std_lib/`.

The fixture's `<OutputFile fileName="…">` paths are relative, so its recordings land wherever
the process is running, under the names the fixture chose — the model's decision, not the
engine's. The example changes into `--record-dir` before constructing the engine rather than
littering wherever it was launched from, which is the only lever a caller has over a model that
names its outputs relatively.

### `delayed_coupling_example` — axonal delays

A connection that declares no delay still costs one tick. A real `<connectionWD delay="10ms"/>`
has to cost exactly ten milliseconds, which is what the delay ring is for: `network_inputs` is a
ring of `max_delay_ticks + 1` rows, each a full `[neuron_count]` wide, and an arrival is
scattered into the row belonging to its own arrival tick.

This measures the offset rather than describing it. One source, three targets, three declared
delays (1 ms, 5 ms, 10 ms), each target's membrane potential watched tick by tick:

```
    target   declared    measured offset    as declared?
         1        1ms             10 ticks    yes
         2        5ms             50 ticks    yes
         3       10ms            100 ticks    yes
```

### `discrete_spike_input_example` — driving from a literal 0/1 array

The third way to drive a model, after a NeuroML-declared generator and a NeuroML-declared
`<spikeArray>`: a host-provided bit sequence, one value per tick per neuron, with no generator
ComponentType behind it. `input_event_streams` is an ordinary public vector and
`step_simulation` reads it every tick, so a caller appending its own `NeuronInputStream` is
indistinguishable, to the engine, from one the parser built. The model here declares no
`<explicitInput>` at all.

The same pattern is fed to two cells at two rates. The fast one loses bits, and the run says
exactly why — the count of bits that arrive inside the 5 ms refractory period matches the
shortfall exactly:

```
  slow -- neuron 0, one pattern element every 100 ticks (10ms)
    input bits    |     |   |       | |   |
    output spikes |     |   |       | |   |
                   6 bits in -> 6 spikes out

  fast -- neuron 1, one pattern element every 20 ticks (2ms)
    input bits    |     |   |       | |   |
    output spikes |     |           |     |
                   6 bits in -> 4 spikes out
```

## Not built: `examples/unsupported/`

Three programs live under `examples/unsupported/` and are deliberately outside the build
(`make examples` globs `examples/*.cpp`, not the subdirectory). Each needs an engine capability
that does not exist yet, and each names it precisely in its own header comment.

| Program | What is missing |
| --- | --- |
| `poisson_population_example.cpp` | (1) `SpikeSourcePoisson` extends `baseSpikeSource`, whose `RuntimeCategory` is Input, so a `<population>` of them throws during `parse_lems`: *"references component 'poissonInstance' of type 'SpikeSourcePoisson', which is not classified as a cell"*. (2) Its dynamics call `random(1)`, which `kernel_codegen.cpp` rejects outright — and there is no per-neuron RNG state buffer in `SpikeEngine` for a stream to live in. |
| `stdp_plasticity_example.cpp` | There is no plasticity mechanism at all. `enable_hebbian_learning` only allocates `last_tick_updated`; nothing reads it. `kernel_codegen.cpp` emits nothing for stage 7 (Plasticity). Nothing maps a NeuroML synapse ComponentType onto a weight-update rule. |
| `glif_stdp_plasticity_example.cpp` | The same three gaps. Everything else it needs — the torus, its propagation, the before/after `WeightMatrix::get()` reads — works today, which is exactly why it cannot ship: it would run to completion and print two identical weights. |

## Things worth knowing before writing your own

### Synaptic strength lives in the synapse's `ibase`, and the synapse is run

A `<projection>` must name a synapse instance, and every example here declares one. **Its own
dynamics are executed.** On each arrival the engine runs the synapse's `OnEvent` handler with
`weight` bound to the summed weight of every edge arriving on that tick, integrates its state
one `dt`, and adds the current it exposes as `i` into the target's synaptic accumulator. For the
`alphaCurrentSynapse` these examples use, that is:

```xml
<TimeDerivative variable="I" value="(2.7182818284590451*J - I)/tau"/>
<TimeDerivative variable="J" value="-J/tau"/>
<OnEvent port="in">
    <StateAssignment variable="J" value="J + weight * ibase"/>
</OnEvent>
```

so one arrival injects an alpha-shaped current that rises to a peak of `weight * ibase` exactly
`tau` after it lands and decays over the same again — not a single-tick impulse. The magnitude
therefore lives in **`ibase`**, which declares `dimension="current"` and is written with its unit
(`ibase="1nA"`); `<connectionWD weight="…">` is the dimensionless `xs:float` multiplier LEMS says
it is and stays `1`. `--synapse-current` is named after `ibase`.

The accumulator holds **amperes**: the cells reduce it with
`<DerivedVariable name="iSyn" dimension="current" select="synapses[*]/i" reduce="add"/>` and use
the result as a current directly, and an `<explicitInput>` current injector adds into the very
same place. See "Weights are stored exactly" below.

**Conductance-based synapses are refused, not approximated.** `expOneSynapse`, `alphaSynapse`,
`expTwoSynapse` and friends compute `i = g * (erev - v)` — a driving force that depends on the
postsynaptic voltage and *reverses sign* as `v` crosses `erev`. Running one as though it were
current-based is a different model, not an approximation, so `kernel_codegen` rejects it by name
at engine construction:

```
kernel_codegen: conductance-based synapses are not supported yet. This type declares a
reversal potential / baseline conductance … (ComponentType 'expOneSynapse')
```

Every example here uses a current-based synapse for that reason. Picking `tau` and `ibase` is a
modelling decision, not a translation: an alpha current delivers `e * ibase * tau` of charge
spread over roughly `4 * tau`, where a conductance synapse delivered a voltage-dependent current
instead, so the numbers do not carry over from a conductance-based model and have to be
re-derived against the membrane they are driving.

### Weights are stored exactly, at any magnitude, so write them in SI

`WeightMatrix` holds adjacency as `W ≈ U·Vᵀ` plus a sparse per-edge delta (a memory-compression
scheme, not learning). `U` and `V` are initialised from a normal(0, 1) draw, so the
reconstruction for any edge is order 1 — but a per-edge weight is **not** stored as a delta
against it. The first `set_edge_weight` call pins the default matrix's coefficient vector to
all-zero, which makes the low-rank term identically `0.0f`, so the stored value *is* the weight.
It round-trips bit-for-bit at any magnitude, verified in `tests/weight_matrix_tests.cpp` down to
`1e-12`, and the GPU propagate kernel reconstructs the same bits because it reads the same
coefficient vector and delta buffer.

So write your currents in **SI, with real units**, exactly as NeuroML expects: a
`<pulseGenerator amplitude="0.6nA"/>` is `6e-10 A`, and an `<alphaCurrentSynapse ibase="1nA"/>`
delivering into the same accumulator is `1e-9 A`. A dimensionless `amplitude="0.6"` is invalid
LEMS — `pulseGenerator/amplitude` carries `dimension="current"` in `Inputs.xml`, and jNeuroML or
NEURON would reject it — so do not reach for a scale parameter to keep the numbers large. Nothing
here needs them large.

`connectionWD/weight` is the one attribute with no unit to write: it is `xs:float` in the schema
and genuinely dimensionless, a multiplier on the synapse's own `gbase`/`ibase`. It stays `1` in
every example here, and the amperes live in the synapse's `ibase` where the dimension is
declared. The examples spell those magnitudes `1.0 * NANOAMPERE` in C++ and emit them through
`nanoampere_attribute()` so a bare `1e-09` never reaches an attribute that wants a unit.

### The refractory period is a Heaviside gate, not a `<Regime>`

The canonical GLIF declarations in `tests/fixtures/nml/glif*_single_cell.nml` put the refractory
period in a `<Regime>`/`<Transition>` pair. **The engine's codegen rejects `<Regime>`**, so
every model in this directory expresses the same behaviour without one:

```xml
<StateVariable name="refractoryTimeRemaining" dimension="time"/>
<DerivedVariable name="integrating" value="H(0 - refractoryTimeRemaining)"/>
<TimeDerivative variable="refractoryTimeRemaining" value="-1"/>
<TimeDerivative variable="v" value="integrating * (...) / C"/>
```

A spike sets `refractoryTimeRemaining` to `tRef`; it counts down; and `integrating` — LEMS's
Heaviside step — gates the membrane derivative to zero for exactly as long as it is positive.
`v` holds at `vreset` through the refractory period and resumes afterwards. This is a real
refractory period, not an omission.

`<Transition>`, `<OnEntry>`, `ConditionalDerivedVariable` and `random()` are likewise rejected by
codegen, each by name — as is `OnEvent` **on a cell**. On a synapse `OnEvent port="in"` is the
one handler that *is* lowered, and a synapse without one is rejected in turn, because a synapse
that ignored its arrivals would silently swallow every spike routed through it. A `select=` path is only lowerable when it reduces over
`synapses` — anything reaching into another child structure is refused rather than silently bound
to something else.

### Declare the GPU context guard before anything that holds a GPU buffer

`GpuContextScope` (`example_support.h`) calls `release_gpu_resources()` in its destructor, which
frees every buffer the backend handed out. It has to run *after* the last object owning one, and
locals destruct in reverse declaration order — so declaring the guard first makes it destruct
last. Freeing by hand at the end of `main` instead frees those buffers while `SpikeEngine` is
still alive, and its destructor then double-frees: a crash at exit, after a run that otherwise
looked fine.

The one thing that goes *ahead* of it is `configure_logging`, because initialising the GPU
context builds the engine's logger singleton and the singleton takes its level from whoever
creates it first.

### Seed your state variables

`<OnStart>` is honoured, and it is the only thing that seeds state. A GLIF5-shaped cell that
leaves `theta` at its zero-initialised value has it sitting far *above* the −50 mV threshold, so
no neuron ever fires and the network looks silently dead rather than erroring.

## Recording, and rendering it

Recording needs no driver: `SpikeEngine` builds one `SimulationRecorder` per `<OutputFile>` /
`<EventOutputFile>` the model declares and writes one frame per tick automatically. An
`<OutputFile>` records a state variable (the membrane-potential analogue); an `<EventOutputFile>`
records a 0.0/1.0 spike mask.

Render either with `render_spire_video.py`:

```bash
pip install numpy matplotlib

./build/examples/glif3_torus_network_example
./examples/render_spire_video.py recordings/glif3_torus_spikes.spire \
    --side 8 --membrane recordings/glif3_torus_membrane.spire
```

That writes `recordings/glif3_torus_spikes.mp4` (pass `--output foo.gif` for a gif instead, no
`ffmpeg` required for that path): every neuron at its real `(row, column)` torus position — the
same `row * side_length + column` flat index the generator wires the network with — membrane
potential as a per-tick heatmap, spikes overlaid as bright markers. The script reads the
`.spire` byte format itself and does **not** need the Python extension. See its module docstring
for every flag.

One known limitation on the engine side: an `<OutputColumn quantity="Pop/0/cell/v"/>` path whose
variable name the engine cannot resolve falls back to the neuron's *first* state variable and
logs a warning. `izhikevich_network_example` trips this, so its two recorded columns are both
`v` rather than `v` and `u`.

## Python

`demo_script.py` predates the NeuroML path entirely: it drives the original hardcoded-LIF engine
through the pybind11 bindings (`SpikeEngine(network, shape=…, rank=…)`, `set_input_neurons`,
`step_simulation([values], tick=…)`). None of those constructors or methods exist on this branch
— they are commented out in `include/spikecorec/core/engine.h` pending rework — and
`src/bindings/bindings.cpp` still references them, so `make python` does not build here. The
script is kept as-is rather than rewritten against an API that has not been decided.

`render_spire_video.py` is the only Python that works today, and it needs nothing from
spikecorec but the `.spire` files themselves.
