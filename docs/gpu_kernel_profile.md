# GPU kernel profile

Measured on 2026-10-20 at commit `d5ddda7`, on an Apple M5 Max (40 GPU cores, Metal). Unlike
`docs/performance_review.md`, which was written from reading the code, everything here was timed. It
covers:
- the whole tick, CPU and GPU;
- `master_step` broken down by code region, with source lines;
- equivalent rewrites of the kernel and a different split of work across GPU threads, each checked to
  leave the same state;
- construction and refits, including the fixed fit kernels.

The CUDA half is built and ready on the DGX Spark but has not run: its GPU was busy with another
user's job all day (see "CUDA").

## Summary

- **The k²-tree walk is the tick.** Walking a neuron's row of the k²-tree takes 95–99% of
  `master_step`'s GPU time in networks whose synapses keep per-connection state, and 77% in M1.
  Everything else together is under 5%, because it is hidden behind the walk's latency.
  This covers the synapse dynamics, the S lookups, the weight-matrix reconstruction, the scatter and
  the cells.
- **The walk is latency bound.** A row walk checks 3–27 tree nodes per connection: a node is set when
  any row in its block has a connection, so the walk visits many nodes that are empty for its own row.
  Each check is a chain of dependent reads, run by one thread. 50 rows take as long as 4,000.
- **A better walker, same layout, one thread per neuron, makes the walk 4.6–5.6× faster.** It reads a
  node's four bits on the row at once, makes one rank query per node, uses no divisions, and keeps
  stacks only as deep as the tree. Measured on all four workloads: kernel time down to 22–38%, state
  identical.
- **Splitting each row's walk across a 32-thread SIMD group is 13–62× faster.** The group walks level
  by level, with prefix sums placing each level's nodes in order.
  - The kernel takes 1.6% of its current time on CUBA, 5.4% on the 10,000-neuron network and 7.6% on
    M1, with state identical.
  - It needs threadgroup memory for the frontier: 5 KB on CUBA, 31 KB on M1. M3's rows would
    exceed the 32 KB limit, which needs a chunked version.
- **Construction of stateful-synapse networks is calibration.** The calibration ticks are ordinary
  `master_step` dispatches, so the walk fixes speed construction by the same factor.
  - CUBA: 97.6% of the 8.3 s build.
  - The 10,000-neuron alpha network: 95% of 4.5 s.
  - The 100,000-neuron alpha network: about 80 of its 87 s, derived from the per-tick cost.
- **On the host, the per-tick costs are the 0.2 ms round trip and M1's random numbers.**
  - The round trip is a `waitUntilCompleted` per tick, and is 4.5–15% of a tick.
  - M1 refills 500,000 random values on the CPU every tick, 1.6 ms or 15.5% of its tick. Half of the
    slots are never read.
  - The queue merge into S, the refit check and spike recording are each under 2%.
- **After the walk is fixed, the host round trip becomes the limit.** At CUBA scale the kernel would
  take 0.07 ms and the round trip 0.2 ms, so batching ticks per submission becomes the next step. That
  needs the merge of the pending queue into S on the GPU first.

## Setup

### Workloads

| Name | Network | Neurons | Connections | Planes (updated) | Active window (ticks) |
|---|---|---|---|---|---|
| `cuba` | M2, Vogels-Abbott CUBA, seed 1, `expCurrSynapse` per connection | 4,000 | 319,775 | 3 (1) | 1,377 |
| `alpha_1000` | `iafCell` + `alphaCurrentSynapse`, out-degree 20, delay 1 ms, 90 pA pulse | 1,000 | 20,000 | 4 (2) | 887 |
| `alpha_10000` | the same | 10,000 | 200,000 | 4 (2) | 887 |
| `alpha_100000` | the same | 100,000 | 2,000,000 | 4 (2) | 887 |
| `brunel_c` | M1 point C, seed 1: delta synapses (weighted arrivals), 20 × 1 kHz Poisson inputs per neuron | 12,500 | 14,870,796 | 2 (0) | 14 |

The alpha networks are the scaling network of `docs/performance_review.md`. Every run used the
default `fit_tolerance` and refit settings.

### Methods

All tools are in the session scratchpad
(`/private/tmp/claude-501/-Users-aleksimpson-Desktop-projects-spikecorec/f0094297-9ef9-4d93-bb32-ca082ea1d8c9/scratchpad/profiling/`).
They were built from a copy of the tree, so the repository is unchanged.

- **The harness** (`profile_harness.cpp`):
  - It links the library as it is.
  - It times a tick by repeating `Spike_Engine::step_simulation` and `dispatch_tick` statement for
    statement, with a clock around each part and the command buffer's own GPU start and end times.
  - It reproduces the engine exactly: the same 11,204 spikes in 5,000 CUBA ticks as `step_simulation`
    itself.
- **Kernel variants** (`make_variants.py`):
  - Rewrites of the generated `master_step` source, either removing one region at a time (the ablation
    ladder) or replacing a region with an equivalent one.
  - At sampled ticks the harness snapshots every buffer the kernel writes (the model slab, S's slab and
    the queue's slab). It runs each variant from the snapshot 5 times, restores the snapshot, then steps
    the tick for real.
  - A rewrite counts as equivalent when the cell state, `network_inputs`, S and the queued edges it
    leaves match the original kernel's. They matched to 3 × 10⁻⁷ relative in every case, which is float
    summation order in the scatter.
- **A standalone walk benchmark** (`walk_benchmark.cpp`, `make_walkers.py`):
  - It runs only the row walk, on k²-trees dumped from the engine, three ways.
  - It checks every row against the tree's own row offsets and against the other walkers.
- **Instruments:**
  - `xctrace` Time Profiler for construction's CPU time.
  - Metal System Trace for its GPU intervals and the compiler's register-spill events.
- **An instrumented copy of the library** for the construction phase breakdown and the per-kernel GPU
  time of the fits. It marks construction's phases, and the backend records each dispatch's GPU time
  under the kernel's name.

### Caveats

- Another process used the GPU during part of the variant runs. Samples where the unchanged kernel ran
  more than 1.3× its median were dropped (2 of 20 for `alpha_10000`, 7 of 30 for `brunel_c`); the paired
  ratios agree with an earlier clean run.
- One tick of the 100,000-neuron run stalled for 667 s of wall time with 0.15 ms of GPU work (the
  process was suspended), and is excluded.
- Metal has no command-line shader profiler, so lines are attributed by ablation, not by sampling.
  - On the spark, `ncu` cannot sample lines either: GPU counters are admin-only there
    (`RmProfilingAdminOnly=1`), and `perf_event_paranoid=4` blocks CPU sampling.
  - With sudo, `ncu --set full --import-source yes` on an NVRTC build with `--generate-line-info` would
    give per-line stall counts.

## The whole tick

Milliseconds per tick on Metal. The mean is over the measured ticks, with ticks that refit counted
separately.

| Part (source) | `alpha_1000` | `cuba` | `alpha_10000` | `brunel_c` | `alpha_100000` |
|---|---|---|---|---|---|
| Random values on the CPU (`engine.cpp:800-803`) | 0 | 0 | 0 | **1.594 (15.5%)** | 0 |
| Kernel arguments (`engine.cpp:809-817`) | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| Encode and commit (`backend.cpp:256-277`) | 0.013 | 0.010 | 0.013 | 0.005 | 0.012 |
| **`master_step` on the GPU** | **1.141 (84%)** | **4.198 (93%)** | **3.999 (94%)** | **8.257 (80%)** | **85.98 (99%)** |
| Round trip beyond GPU time (`waitUntilCompleted`, `backend.cpp:278`) | 0.204 (15%) | 0.201 (4.5%) | 0.211 (4.9%) | 0.407 (4.0%) | 0.217 |
| Queue merge into S (`compact_pending_deltas`, `weight_matrix.cpp:782-945`) | 0.000 | 0.073 (1.6%) | 0.002 | 0 | 0.041 |
| Deferred walk passes (`engine.cpp:828-836`) | 0.006 | 0.015 | 0.032 | 0 | 0.626 |
| Refit check (`engine.cpp:780-782`) | 0 | 0 | 0 | 0 | 0 |
| Spike recording (`record_tick`, `engine.cpp:947-979`) | 0.001 | 0.003 | 0.010 | 0.009 | 0.074 |
| **Whole tick** | **1.357** | **4.496** | **4.267** | **10.272** | **86.98** |
| Refit, per refit (`weight_matrix.cpp:1214-1253`) | 26.6 | 20.0 | 95.5 | none | 1,560 |
| Refit per tick, spread over 1,000 ticks | 0.027 | 0.020 | 0.096 | 0 | 1.56 |

- **Per-tick speeds.** The CUBA tick is 4.5 ms. The alpha networks take 1.36, 4.27 and 87 ms per tick
  at 1,000, 10,000 and 100,000 neurons, matching the 2026-10-07 baseline (1.25, 4.1 and 84 ms).
- **Neurons walking.** At these rates almost every neuron is inside its active window: 1,867 of 4,000
  on CUBA, and all of them on the alpha networks.
- **Deferred passes.** These are rare in steady state: 6 in 3,000 CUBA ticks, around refits.
- **CPU random values** (`brunel_c`): the kernel reserves one slot per input entry for each population
  that runs the input type (`dynamics.cpp:2743-2751`).
  - With two populations and 250,000 entries that is 500,000 draws per tick. Entries are numbered across
    both populations, so the inhibitory population reads slots 450,000 to 499,999, and slots 200,000 to
    449,999 are drawn but never read.
  - Of the slots that are read, a draw is used only when its input fires, about 10% of ticks at 1 kHz.
- **Refit at 100,000 neurons:** 1.56 s, mostly fits that walk the tree (see "Refits").

## `master_step` by code region

Each step of the ablation ladder adds one region. In the table:
- "Kernel ms" is the mean over samples of each variant's median of 5 runs;
- "Step" is a variant minus the one above it;
- "Paired" is the median over samples of the variant's time over the full kernel's time at the same
  tick.

Line numbers are in `src/spikecorec/nml/kernel_boilerplate.cpp` (Metal half) unless named.

### CUBA (per-connection exponential synapse), 20 samples

| Region added | Source | Kernel ms | Step | Paired |
|---|---|---|---|---|
| Stages 1–5: drain inputs, cells, spike row | `kernel_boilerplate.cpp:188-204`; cells from `Codegen::translate_component_instance` (`dynamics.cpp:2472`) | 0.005 | 0.005 | 0.001 |
| Active-window test and queue reservation | `:209-234` | 0.005 | 0.000 | 0.001 |
| **k²-tree walk** (stacks `:236-247`, loop `:251-264`, `k2t_next_neighbor` in `metal/k2tree_device.metalinc:126-188`) | | **4.014** | **4.009** | **0.955** |
| Delay load and arrival test | `:266-279` (`spikecorec_load_edge` `:65-80`) | 4.014 | 0.000 | 0.956 |
| Synapse prototype lookup | `dynamics.cpp:1643-1647` → `spikecorec_synapse_prototype` `:109-124` | 4.016 | 0.001 | 0.956 |
| Synapse dynamics: plane loads and arithmetic | `Codegen::translate_synapse_type` (`dynamics.cpp:3025`), loads emitted by `load_edge` (`dynamics.cpp:142-149`) | 4.165 | 0.149 | 0.992 |
| Full kernel (+ S and queue writes, scatter) | `spikecorec_accumulate_edge` `:85-106`; scatter `dynamics.cpp:3594` | 4.199 | 0.034 | 1.000 |

Removing the scatter alone saves nothing measurable, and the S and queue writes alone 0.036 ms. On
`alpha_1000` and `alpha_10000` the walk is 94% and 98% of the kernel, with the same pattern.

### M1 point C (weighted arrivals, 1.39 million runs), 23 samples

| Region added | Source | Kernel ms | Step | Paired |
|---|---|---|---|---|
| Stages 1–5, with 20 Poisson inputs per neuron | `dynamics.cpp:2555-2606` (input entries) | 0.033 | 0.033 | 0.003 |
| Window, reservation, arrival-due test | `:209-234`; `dynamics.cpp:1683-1701` | 0.033 | 0.000 | 0.003 |
| **k²-tree walk** | as above | **6.129** | **6.096** | **0.787** |
| Delay load and arrival test | `:266-279` | 6.834 | 0.705 | 0.871 |
| **Synapse prototype lookup**: a binary search over 1,389,675 runs for every connection | `:109-124`, called from `dynamics.cpp:1643-1647` | 8.154 | **1.319** | 1.029 |
| Full kernel (arrival test, weight load, scatter) | `Codegen::translate_weighted_arrival` (`dynamics.cpp:3234-3323`) | 7.916 | (−0.24, noise) | 1.000 |

Only the neurons whose spike reaches its targets on the next tick walk: about 47 per tick, each walking
about 1,190 connections in one thread.

### Why the walk costs so much

| Workload | Tree height | Connections per row | Largest frontier per level | Nodes checked per row | Per connection |
|---|---|---|---|---|---|
| `alpha_1000` | 5 | 20 | 4, 16, 63, 76, 20 | 152 | 7.6 |
| `cuba` | 6 | 80 | 4, 16, 63, 250, 318, 107 | 608 | 7.6 |
| `alpha_10000` | 7 | 20 | 3, 10, 40, 157, 271, 80, 20 | 540 | 27.0 |
| `brunel_c` | 7 | 1,190 | 4, 13, 49, 196, 782, 2,567, 1,286 | 3,540 | 3.0 |

- **Visits.** A row's walk visits every node, on its row band, whose block holds a connection in any
  of the block's rows. On random connectivity most of those are empty for the walking row. One level
  above the leaves, a node (a 4 × 4 block) is set for 28% of blocks on CUBA, but holds a connection
  on the walking row for only 8%.
- **Per node.** The current walk handles each visited node in four iterations, one per child. Each
  iteration:
  - tests one bit;
  - for every set internal child, makes a rank query (superblock table, subblock table and bit word,
    `k2tree_device.metalinc:14-29`);
  - divides twice by run-time block sizes (`:160-161`).
- **Stacks.** The five 32-entry stacks are indexed at run time. Metal's compiler reports a 400-byte
  spill per thread for `master_step`: a register-spill event on each of the 1,475 calibration ticks
  traced.
- **One thread per row.** In the standalone benchmark, walking 50 rows takes 4.0 ms against 4.9 ms for
  all 4,000 (CUBA), and 22.6 ms against 39.8 ms for all 12,500 (M1). The time is set by the longest
  chain of dependent reads in one thread, not by throughput.
- **Occupancy.** CUBA runs 4,000 threads, about 3 SIMD groups per GPU core.

## Faster walks and other rewrites

### Standalone walk benchmark (exact on every row)

Every row of each tree, then 2,000 rows, then 50 rows. Best of 5. Milliseconds of GPU time.

| Tree | Rows | Current serial walk | Nibble walk (one thread per row) | Level walk (one SIMD group per row) |
|---|---|---|---|---|
| `cuba` | 4,000 / 50 | 4.87 / 4.01 | 0.93 / 0.74 (5.2× / 5.4×) | 0.056 / 0.017 (86× / 240×) |
| `alpha_1000` | 1,000 / 50 | 1.06 / 0.87 | 0.20 / 0.16 (5.4× / 5.6×) | 0.008 / 0.008 (125× / 113×) |
| `alpha_10000` | 10,000 / 50 | 3.95 / 3.23 | 0.82 / 0.63 (4.8× / 5.1×) | 0.132 / 0.018 (30× / 179×) |
| `brunel_c` | 12,500 / 50 | 39.8 / 22.6 | 8.6 / 4.4 (4.6× / 5.1×) | 5.39 / 0.070 (7.4× / 321×) |

- **Nibble walk** (`nibble_walker.py`):
  - The stack holds, per level, the node's column base, its four bits on the walking row, the bits
    not yet taken, and the rank of the node's first bit.
  - A node costs one word load and one rank query.
  - Its set children come out by count-trailing-zeros, and child ranks are the node's rank plus a
    popcount.
  - Row offsets are `(row >> 2(height − level − 1)) & 3`.
- **Level walk:** each level's frontier sits in threadgroup memory, one node per lane. Each lane reads
  its node's four bits on the row, and a SIMD prefix sum of the child counts places the children in
  column order. After the leaf level, the frontier is the row's targets in edge order, and the lanes
  split the connections 32 at a time.

### In the engine's kernel

Paired ratio to the full kernel (lower is faster). Every rewrite leaves the same state.

| Variant | What changes | `alpha_1000` | `cuba` | `alpha_10000` | `brunel_c` |
|---|---|---|---|---|---|
| p01 | each plane loaded once per connection | 0.988 | 0.991 | 0.998 | 1.005 |
| p02 | `rank_float4_stride` baked | 0.989 | 0.992 | 0.995 | 0.997 |
| p03 | p01 + p02, U[source] once per neuron, V[target] once per connection | 0.980 | 0.814 | 0.993 | 0.983 |
| p04 | S searched with a moving position per plane instead of a binary search | 0.972 | 0.980 | 0.989 | 0.995 |
| p05a | walk stacks sized to the tree height | 0.628 | 0.645 | 0.653 | 0.714 |
| p05b | divisions by block sizes as shifts | 0.888 | 0.873 | 0.873 | 0.847 |
| p05 | p05a + p05b | 0.543 | 0.552 | 0.566 | 0.621 |
| p06 | p03 + p04 + p05 | 0.507 | 0.529 | 0.556 | 0.622 |
| **p07** | **nibble walk** | **0.251** | **0.235** | **0.223** | **0.383** |
| p08 | p07 + p03 + p04 | 0.203 | 0.209 | 0.215 | 0.378 |
| p09 | prototype from a run position that moves forward (one binary search per neuron) | | 1.000 | | 0.858 |
| **p10** | **p08 + p09** | | **0.203** | | **0.227** |
| **w01** | **one SIMD group per neuron, level walk; lane 0 runs stages 1–5 and the reservation** | **0.018** | **0.016** | **0.054** | **0.076** |

Kernel milliseconds:

| Workload | Before | After p10 (or p08) | After w01 |
|---|---|---|---|
| `alpha_1000` | 1.13 | 0.23 | 0.020 |
| `cuba` | 4.20 | 0.86 | 0.068 |
| `alpha_10000` | 4.0 | 0.86 | 0.22 |
| `brunel_c` | 7.9 | 1.80 | 0.60 |

`alpha_10000` and `brunel_c` are taken from the clean first run times the paired ratios.

Points the table shows:
- The rewrites of the per-connection code (p01–p04, p09) are worth little while the walk dominates.
  Once the walk is cheap they show: on top of the nibble walk they save another 14% on CUBA (p08
  against p07), and the prototype position another 40% on M1 (p10 against p08).
- **About w01's design:**
  - **Threadgroup memory.** The frontier is sized to the largest frontier: 320 nodes on CUBA (5 KB),
    and 2,600 on M1 with 16-bit columns (31 KB of the 32 KB limit).
  - **M3.** At 77,000 neurons and about 3,900 connections per row (16,384+ padded columns, so 32-bit
    columns), the frontier would not fit. It needs a chunked frontier (walk the top levels level by
    level, then each subtree that fits), or a split of the row by column band with a prefix sum of
    per-band counts.
  - **The cell stage.** w01 runs stages 1–5 in lane 0 of each group, so it dispatches 32 threads per
    neuron. A production version would use two dispatches in one command buffer: the cells one thread
    per neuron, appending walking neurons to a list; then the walk one SIMD group per listed neuron
    (`dispatchThreadgroups` with an indirect buffer on Metal), with no CPU readback.
  - **S writes need no extra synchronization.** Lanes of one group update distinct connections' S
    entries, and only that neuron's group touches its row.

## Construction

Instrumented copy of the library. Seconds since the previous mark.

| Phase (source) | `cuba` | `alpha_10000` | `brunel_c` |
|---|---|---|---|
| XSD validation and LEMS/NeuroML parse (`engine.cpp:107-113`) | 0.006 | 0.027 | 0.513 |
| Connection arrays sorted, topology checked (`engine.cpp:115-129`, `684-739`) | 0.007 | 0.004 | 0.318 |
| Codegen checks and collection (`engine.cpp:165-177`) | 0.000 | 0.000 | 0.224 |
| Model memory and starting state (`engine.cpp:179-197`) | 0.039 | 0.000 | 0.951 |
| Runs gathered from the connections (`engine.cpp:303-405`) | 0.005 | 0.002 | 0.277 |
| k²-tree build (Weight_Matrix initializer) | 0.017 | 0.015 | 0.709 |
| Row offsets, basis, seeding (`weight_matrix.cpp:132-187`) | 0.032 | 0.059 | 0.582 |
| Construction fits (`declare_runs`, `weight_matrix.cpp:1069-1137`) | 0.054 | 0.073 | 0.472 |
| Kernel translation, emission, compile | 0.003 | 0.002 | 0.027 |
| **Calibration ticks** (`calibrate_zero_start_planes`, `engine.cpp:886-898`) | **8.068** | **4.308** | none |
| Calibration lane-budget fits (`engine.cpp:907-910`) | 0.021 | 0.038 | none |
| **Total** | **8.27** | **4.54** | **4.09** |

Inside the calibration ticks:

| | `cuba` | `alpha_10000` |
|---|---|---|
| Ticks (`master_step` dispatches) | 1,475 | 950 |
| `master_step` GPU time | 7.42 s | 3.74 s |
| `dispatch_tick` wall time | 7.77 s | 4.08 s |
| `plane_update_sum_of_squares` on the CPU, every tick over all of S (`engine.cpp:890`, `weight_matrix.cpp:983-994`) | 0.30 s | 0.22 s |
| Copying the kept tick's values | 0.001 s | 0.002 s |

- **Calibration is construction** whenever a plane starts at zero. At 100,000 neurons the same 950
  ticks at 84 ms each make about 80 of the 87 s measured. Time Profiler attributes the rest of that
  build's CPU time to the fits' value lists: `plane_values_from` over `K2_Tree::get_neighbors` on
  worker threads.
- **The fixed fit kernels are a small part of construction.** On CUBA, all fit dispatches together
  take 0.03 s of GPU time; on M1, 0.22 s of its 0.47 s of fits.
- **Construction CPU samples.** Most construction time is GPU waiting, so the CPU samples are few:
  0.66 s on CUBA and 0.29 s on M1, of which libxml2 parsing is 41% and freeing the document 18%.

## Refits

On `alpha_10000` a refit takes 82 ms (`refit: total`):
- 27 ms of GPU time in 6 `fit_walk_residual_kernel` dispatches;
- 22 ms in 4 `fit_walk_row_solve_kernel` dispatches;
- the remainder on the host, keying S by target and preparing the problems.

At 100,000 neurons a refit takes 1.56 s.

The fit kernels walk the tree with the same structure as `master_step`:
- `fit_walk_next` in `metal/weight_fit_kernels.metal:184-252`;
- per-node divisions at `:221` and `:233`;
- 32-entry stacks in `Fit_Walk` (`:126-141`).

Metal's compiler reports 688- and 784-byte spills for them. The nibble walk applies to them unchanged,
and to `k2t_next_predecessor` (`k2tree_device.metalinc:194-256`) for column walks.

## Recommendations, in order

All of these keep the k²-tree, the factorized planes, S and the queue as they are. They change how the
kernel walks and computes, not what is stored.

1. **Replace the row walk with the nibble walk.**
   - Where: `k2t_next_neighbor` in `metal/k2tree_device.metalinc:126-188` and its CUDA port
     (`kernel_boilerplate.cpp:321-381`, `cuda/k2tree_device.cuh`); also `fit_walk_next`.
   - Bake the tree height into the generated kernel, as the other tree constants already are.
   - Measured gain: the kernel takes 22–38% of its current time, and calibration (most of
     construction for stateful synapses) drops in proportion.
   - The smallest change with the largest measured effect.
2. **Walk each active neuron's row with a SIMD group** (w01), as two dispatches per tick: cells one
   thread per neuron, then the walk one group per active neuron.
   - Measured gain: 13–62× on the kernel.
   - Needs a chunked frontier or a column-band split for M3-sized rows.
3. **Prototype from a moving run position** (p09).
   - Where: the generated call at `dynamics.cpp:1643-1647`.
   - Measured gain: 14% of M1's kernel now, and 40% once the walk is fast.
   - Also: hoist U[source] and V[target] and use moving S positions (p03, p04); 14% on CUBA after the
     walk fix.
4. **Random values only where read.** Number the random slots per population's own entries, not all
   entries (`dynamics.cpp:2743-2751`). This halves M1's 1.6 ms per tick of CPU draws.
   - Drawing on the GPU would remove the cost, but it changes the random stream, so it needs checking
     against the random() specification.
5. **Calibration's per-tick sum of squares.**
   - Where: `engine.cpp:890`, `weight_matrix.cpp:983-994`.
   - It scans all of S on the CPU every tick: 0.2–0.3 s per build, and more once the ticks are fast.
     Evaluate it every N ticks, or on the GPU.
6. **Then the host round trip.** With items 1–2 the CUBA tick would be about 0.07 ms of kernel against
   0.2 ms of `waitUntilCompleted` plus 0.07 ms of merge. Encoding several ticks per command buffer,
   synchronizing with a shared event, needs three things:
   - the queue merge on the GPU (`compact_pending_deltas`);
   - deferred passes as indirect dispatches;
   - spikes recorded into a device buffer.

## CUDA

Built on the spark with the same harness and variant generator.
- Location: `~/profiling_scratch/`, with `profile_harness_cuda`, the CUDA library and the workloads.
- Every CUDA variant compiles with `nvcc -arch=sm_121`. That includes the warp version of w01, which
  uses shuffles for the prefix sums and shared memory for the frontier.
- `run_cuda_profile.sh` runs the whole set: tick parts, variants and an `nsys` timeline per workload.
  It refuses to start while another process is using the GPU.

It was not run because the GPU was in use by another user's job all day.

## Reproducing

Profiling directory, as in "Methods":
- **The whole tick:** `profile_harness_metal <workload> step <warmup> <ticks> <csv>`, then
  `summarize_steps.py`.
- **Kernel regions and rewrites:** `make_variants.py <kernel> <directory> metal <height> <stride>
  <planes> <frontier capacity>`, then `profile_harness_metal <workload> variants ...`, then
  `summarize_variants.py`.
- **Walks alone:** `profile_harness_metal <workload> dump_tree <prefix>`, then `make_walkers.py`, then
  `walk_benchmark`.
- **Construction phases and fit kernels:** the instrumented build (`build_harness_instrumented.sh`),
  with `construct` or `step`.
- **Background:** the literature survey these build on is `literature_survey.md` in the same directory.
