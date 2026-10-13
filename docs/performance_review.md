# Performance review

State as of 2026-10-07, branch `agent_integration_2`, commit `843fb03`. This is a read-only
review: nothing was changed. It is based on:
- the generated master kernels of nine models;
- the shared device helpers;
- the host code that runs each tick;
- the benchmarks below.

The impact ratings are estimates from reading the code; nothing has been profiled yet.

## Measurements

### Test suite

`make test` takes 3 min 45 s for 177 tests. Ten network-scale tests account for about 194 s of
that:

| Test | Time |
|---|---|
| GLIF networks: 484 neurons, 9,680 connections, 20,000 ticks (6 tests) | 19–30 s each |
| `poissonFiringSynapse` statistics, 100,000 ticks | 16 s |
| The two 200-neuron PyNN network runs (golden run and jNeuroML comparison) | 15 s each |
| Seed reproducibility, 400 neurons | 10 s |

`make test-python` takes 3.4 s.

The C++ runner used 18% of one CPU core over the whole run, so it spends most of its time waiting
on the GPU.

### Simulation speed

**Setup:**
- each neuron is an `iafCell` with a 90 pA `pulseGenerator` attached through an `inputList`;
- connectivity is `random_fixed_outdegree(side, 20, 1)`;
- each connection has an `alphaCurrentSynapse` with `ibase` = 2 pA, weight 1 and a 1 ms delay;
- dt = 0.1 ms, giving firing rates of about 20–30 Hz;
- each run is 2,000 ticks, except 500 ticks at 100k neurons.

Real time at dt = 0.1 ms needs 10,000 ticks per second.

| Neurons | Connections | Construction | ms per tick | Ticks per second | Simulated s per wall s |
|---|---|---|---|---|---|
| 100 | 2,000 | 0.12 s | 0.45 | ~2,240 | 0.22 |
| 1,024 | 20,480 | 0.11 s | 1.25 | ~800 | 0.080 |
| 10,000 | 200,000 | 0.35 s | 4.1 | ~247 | 0.025 |
| 99,856 | 2.0 M | 7.2 s | 84 | ~12 | 0.001 |
| 10,000, no connections | 0 | 0.13 s | 0.18 | ~5,500 | 0.55 |

What the table shows:
- **Fixed cost per tick.** There is a floor of about 0.15–0.2 ms even for tiny models.
- **Cells and inputs are cheap.** 10,000 neurons with their inputs but no connections take 0.18 ms per tick.
- **Connections dominate.** Adding 200,000 connections takes the same 10,000 neurons from 0.18 ms to 4.1 ms per tick.
- **Cost grows faster than size.** From 10,000 to 100,000 neurons the time per tick rises about 20×, not 10×. Neuron-updates per second fall from 2.5 million to 1.2 million.
- **Construction is only fast here because every connection has the same weight**, which makes the fit trivial. Varied weights take longer to construct.

### Models reviewed

| Model | What it exercises |
|---|---|
| GLIF1 network: 64 cells, random out-degree 8, alpha synapses, pulse inputs | cell, refractory gate, connection loop |
| GLIF5 network, same shape | adapting threshold, after-spike currents |
| `glif_family` fixture | all five GLIF types in one kernel |
| `pynn_synapses` fixture | `IF_curr_exp`, `IF_curr_alpha`, two synapse types |
| `pynn_poisson_network` fixture | Poisson sources, 200 projection runs, two synapse types |
| `izhikevich2007Cell`, `adExIaFCell`, `iafRefCell` with two synapse types | nonlinear cells, refractory-regime dynamics |
| Every input category on one population | compound, ramp, clamp, `timedSynapticInput`, `poissonFiringSynapse`, `spikeArray` |
| A cell with an `OnEvent` | `event_arrival_count` |
| `spikeGeneratorPoisson` sources driving `iafCell`s | `random()` in cells |

## A. Connection loop, run per connection per tick

### 1. Repeated edge reads (high)

Each connection calls `spikecorec_load_edge` 5 times per tick, and up to 10 on arrival or settle
ticks, though a synapse type touches at most 4 distinct planes. In `pynn_synapses` the alpha
synapse reads plane 3 four times in one iteration. Every call:
- rebuilds the value from the factorization;
- searches the sparse update buffer.

### 2. The same basis rows reloaded for every plane (high)

`spikecorec_reconstruct_edge` computes `U[source] · (C_plane × V[target])` and loads both rows on
every call. The rows are the same for every plane of an edge, and `U[source]` is the same for every
connection of the neuron. Loading `U[source]` once per neuron and `V[target]` once per connection,
then computing every plane from them, removes most of that memory traffic.

The coefficients are also loaded as four scalar reads per lane instead of one vector read.

### 3. A binary search for every read and every write (high)

`spikecorec_edge_delta_index` binary-searches the source's row of the plane's sparse update buffer.
It runs for every `load_edge` and every `accumulate_edge`. The connection loop already visits a
neuron's connections in the same order as that row is sorted, so a moving position per plane would
replace every search.

### 4. One write per state variable, and an indirect `OnEvent` (medium)

Each synapse state variable gets its own `spikecorec_accumulate_edge` per tick, each with its own
search, and an arrival adds another write to the same plane.

The `OnEvent` change is emitted as the new value minus the old one: `(I + weight) - I`. That is two
extra reads that cancel out, and in 32-bit floats the subtraction loses precision when `I` is
large. For an assignment of the form `x = x + something`, the change is just `something`.

### 5. One global atomic counter for queued updates (medium)

A connection whose update has no entry in the sparse buffer yet appends to the pending queue. Every
append, from every thread, goes through the single `pending_delta_count` atomic, so all threads
contend on one address.

### 6. Synapse-type lookup per connection per tick (medium)

With more than one synapse type, `spikecorec_synapse_prototype` binary-searches the projection runs
for every connection on every tick. That is about 8 steps for the 200 runs in
`pynn_poisson_network`. The type never changes, and a branch per type follows the lookup.

### 7. Constant connection data rebuilt every tick (medium)

The delay (plane 1) is rebuilt through the factorization and rounded from a float on every tick.
The weight (plane 0) is rebuilt on every arrival. Without plasticity both are constant.

### 8. Costly k²-tree walk (medium)

Each target costs:
- one call to `k2t_next_neighbor`, a `while` loop whose length depends on the data;
- integer divisions by divisors known only at run time;
- rank lookups with popcounts.

The data-dependent loop also blocks unrolling. Each thread allocates five walk stacks of
`MAX_K2TREE_HEIGHT` = 32 ints (640 bytes), although the tree height is 2–4 and known at compile
time. Arrays that size likely spill from registers to slower thread memory.

### 9. The loop covers the whole active window (medium)

A neuron walks its connections for `synapse_active_ticks` after each spike. That window is the
longest delay plus the slowest synapse's settle time, taken across every synapse type: 887 ticks in
the GLIF network, 3,610 in `pynn_synapses`. A neuron firing more often than that walks every
connection every tick. That includes synapses still at rest because their spike hasn't arrived yet.

### 10. Atomic scatter into targets (medium)

Synapse currents reach their targets through `atomic_float` adds into `network_inputs`. Targets with
many incoming connections serialize those adds.

### 11. Arrival counting for every target (low)

Once any cell type has an `OnEvent`, every arrival atomically increments `event_arrival_count`. This
happens even for targets whose type never reads the count.

### 12. Divergence (medium)

Neurons in the same GPU group differ in:
- number of connections;
- active window;
- synapse types.

The group runs for as long as its slowest member.

## B. Cell and input code

### 13. Strided memory layout (high)

`cell_state` and `input_values` are laid out per neuron: `neuron_index * count + offset`. Adjacent
threads read addresses `count` floats apart instead of consecutive ones, which wastes memory
bandwidth. Storing each variable as its own array per population would make those reads
consecutive.

### 14. State read and written in device memory repeatedly (medium)

GLIF5's `theta` is read three times per tick, and `v` is read and written several times. The
compiler may not be allowed to keep these in registers, because writes through `cell_state` could
alias the reads. Loading each state variable into a local once and storing it once at the end
avoids that.

### 15. `t` recomputed at every use (low–medium)

The time is recomputed with `precise::divide(tick, steps_per_second)` everywhere it appears: up to
8 times in one Poisson source's handler, and 4 times per pulse input. It could be computed once at
the top of the kernel.

### 16. Float division in the refractory test (low)

Each neuron does one division per tick. Comparing `tick − last_spiked` against the duration
converted to ticks at compile time would avoid it.

### 17. Dead state and dead writes (medium)

- **Leftover refractory variables.** The refractory stamp (`lastSpikeTime` in `iafRefCell`,
  `adExIaFCell` and the PyNN cells) is written on every spike. GLIF's `refractoryTimeElapsed` still
  has a slot per neuron. Neither is ever read, because the refractory regime is lowered to the
  `last_spiked` test.
- **Unread timers.** `tsince` is integrated every tick in Poisson sources, `spikeArray` and the
  Poisson inputs. Only its exposure uses it, and nothing reads the exposure.
- **Zero derivatives.** A derivative of 0, such as `tnextUsed` and `tnextIdeal`, still generates a
  read-modify-write every tick.

### 18. Work done before the regime test (low–medium)

In `adExIaFCell`, `iMemb` (including its `exp`) and `v`'s derivative are computed every tick, even
while the cell is refractory and `v` is held. Both `w` derivatives, one for each regime, are also
computed every tick although they are identical and only one is used.

### 19. Pure functions of `t` stored and read back (low)

Pulse, sine and ramp inputs write their `i` into `input_values` and read it straight back, every
tick.

### 20. If-else chains for dispatch (medium as populations grow)

Choosing a population is an `if (neuron_index < …) else if …` chain, so with many populations each
thread makes many comparisons. GPU groups that straddle a population boundary run both branches.
Choosing each input entry's type has the same shape.

### 21. Input code duplicated per population (low)

Each input type is compiled into every population it reaches, so the kernel grows with populations
× input types.

### 22. Unsimplified unit arithmetic (low)

The PyNN cells generate expressions like `(0.001f * X) / 0.001f` and `0.0f + …`. `fold_constants`
folds arithmetic between two constants, but not identities like these.

## C. Host work per tick

### 23. One command buffer and a full wait per tick (high)

The Metal `Engine_Backend::run_function` (`src/spikecorec/core/backend.cpp`) commits one command
buffer per tick and calls `waitUntilCompleted`.
That is a full CPU–GPU round trip every tick, and it is the 0.15–0.2 ms floor seen even in tiny
models. The GPU sits idle while the CPU does its per-tick work and the reverse, and ticks are never
batched into one submission.

### 24. Merging updates every tick (high)

`Weight_Matrix::compact_pending_deltas` runs every tick whenever synapse state changes. On each run
it:
- copies every entry of the sparse update buffer for every plane (`current_deltas`);
- fully sorts each changed plane;
- rebuilds that plane's index.

That is O(S log S) work on the CPU per tick, with S up to about 90% of connections while synapses
are active. It is the most likely cause of the faster-than-linear growth: 84 ms per tick at
2 million connections.

### 25. Random numbers drawn on the CPU for every slot (medium)

`Spike_Engine::step_simulation` refills `random_values` with one draw per slot on every tick. A
Poisson source only uses its draw on the tick it fires, so with 100,000 Poisson cells that is
100,000 draws per tick, almost all of them unused.

### 26. Kernel arguments rebuilt each tick (low)

The argument list is rebuilt through map lookups and function-object calls on every tick.

### 27. Spike recording scans all neurons (low)

`record_tick` scans the whole spike row on the CPU every tick.

## D. Memory

### 28. Pending queue size (medium)

The pending queue is sized to 2 × connections × updated planes. Each entry is 16 bytes: an `s64`
ordinal, an `f32` value and an `s32` plane index. At 2 million connections with two state planes
that is 128 MB.

### 29. Planes sized for the largest synapse type (low–medium)

Every connection carries the planes of the largest synapse type in the model. In
`pynn_poisson_network`, the `expCurrSynapse` connections carry a fourth plane they never use.

## E. Outside the tick loop

Items 30–32 were added on 2026-10-12 by the resilience review (`docs/resilience_review.md`), at
commit `04d1b3d`. Unlike the items above, they were measured.

### 30. Per-neuron lookups rescan the network (medium)

Several host helpers look up a neuron's population or memory index by rescanning the network's
children. `get_population_size` also recounts a population's `<instance>` children on every call.
Each lookup is O(N), and they are called once per neuron or once per input target, so the helpers
are O(N²):
- `state_variable_array`;
- the setup for `record_membrane_video`;
- engine construction.

The documented `Simulation_Recorder` example calls `state_variable_array` every 5 ticks. At 4000
cells one call costs about 60 ticks of GPU time.

- **Location:**
  - `src/spikecorec/core/engine.cpp:346-376` (`population_of_neuron`, `cell_memory_index_of`);
  - `src/spikecorec/nml/dynamics.cpp:366-376` (`network_populations`);
  - `src/spikecorec/nml/parser.cpp:1065-1091` (`neuron_index_of`, `get_population_size`), called
    per input target at `dynamics.cpp:676`.
- **Measured:** times in ms; ticks stay at about 2.7 ms throughout.

  | Model | Measured | 500 | 1000 | 2000 | 4000 cells |
  |---|---|---|---|---|---|
  | One `explicitInput` per cell | `state_variable_array` | 2.4 | 10 | 43 | 181 |
  | | Construction | 39 | 68 | 185 | 666 |
  | `populationList` population | `state_variable_array` | 1.7 | 7.7 | 35 | 158 |
  | Plain population | `state_variable_array` | 0.1–0.3 | | | |

  `parse()` alone is also quadratic in the number of `explicitInput`s: 0.28 s at 4000 inputs and
  2.6 s at 12,000. That cause was not traced.

### 31. The projection scan is quadratic in the number of connections (medium)

`fit_basis_from_projections` runs a linear `find` over every distinct value combination seen so
far, once per projection run. That is O(runs × distinct). The 256-lane check runs only after the
whole scan. With per-connection values (one run per connection, as `build_weight_matrix`
produces), it therefore makes about E²/2 vector comparisons before giving up.

- **Location:** `src/spikecorec/core/weight_matrix.cpp:858-870`.
- **Measured:**

  | Connections | Scan time | Share of `declare_projections` |
  |---|---|---|
  | 10,240 | 0.025 s | 14% |
  | 40,960 | 0.31 s | 49% |
  | 163,840 | 5.6 s | 86% |

  For the last row, the fit itself took 0.86 s. Extrapolated to the 204,800-connection benchmark,
  the scan takes about 9 s. That is the construction time `docs/weight_matrix_status.md`
  (limitation 6) attributes to the fewer-lanes sweeps.

### 32. The k²-tree build briefly uses about 60× the finished tree's memory (medium)

`K2_Tree::from_adjacency_list` keeps an `Unordered_Map<u64, s32>` entry per non-empty block per
level. It also stores every tree bit as an `s32` in a growing vector before packing. Peak memory
during the build is about 400–530 bytes per connection, against 6–8 bytes per connection in the
finished tree. This caps the size of network that can be built at all.

- **Location:** `src/spikecorec/core/k2tree.cpp:187-225`.
- **Measured** (`random_fixed_outdegree`, out-degree 10):

  | Connections | Build time | Finished tree | Peak memory growth |
  |---|---|---|---|
  | 100k | 0.025 s | 0.5 MB | 34 MB |
  | 1M | 0.7 s | 7.5 MB | 395 MB |
  | 2.5M | 2.9 s | 21 MB | 1328 MB |

## Where to start

- **Kernel:** items 1, 2, 3, 4 and 7, done together. That is: fetch each connection's data once,
  then loop over it.
  - Load `U[source]` once per neuron.
  - Walk the sparse update buffers with moving positions.
  - Compute every plane from one load of `V[target]`.
  - Write each plane once.
- **Host:** item 23 for small models and item 24 for large ones.
- **Layout:** item 13, which affects every kernel.
