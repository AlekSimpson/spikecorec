# WeightMatrix: state, problems and limitations

State as of 2026-10-05, branch `agent_integration_2`. The refit and construction changes described
here are in the working tree and not yet committed.

## What it stores

Every per-edge value of the network's synapses lives in the `WeightMatrix`, one **plane** per
variable:
- plane 0: the weight;
- plane 1: the delay, in ticks;
- plane 2 and up: the synapse type's LEMS StateVariables, in order.

Every plane is treated the same way by the fit, the refit and the tolerance.

**Topology.** The k²-tree says which `(source, target)` pairs are edges. An edge's ordinal is
`edge_row_offset[source] + slot`, where slot is its position in the k²-tree walk.

**Values.** A shared basis says what each edge's values are:

```
plane k at edge i -> j  =  sum over lanes l of  U[i, l] * C[k, l] * V[j, l]
```

| Buffer | Shape | Meaning |
|---|---|---|
| `U` | `node_count × rank` floats | row `i` belongs to node `i` as a source |
| `V` | `node_count × rank` floats | row `j` belongs to node `j` as a target |
| `coefficients` (C) | `plane_count × rank` floats | row `k` is plane `k` |

- A **lane** is one column index `l`, shared by U, V and every coefficient row. `rank` is the
  number of lanes.
- Lanes come in groups of 4 (`LANE_GROUP`, float4 storage).
- The kernel's limit is 256 lanes (`MAX_RANK_FLOAT4_STRIDE` 64 × 4).
- The basis costs `4 × (2 × node_count × rank + plane_count × rank)` bytes.

**S, the sparse delta matrix.** S holds the updates made since the last fit, as a per-plane CSR
over source rows.
- Each entry is an s64 edge ordinal plus an f32 delta: 12 bytes.
- Every read adds S, so a value reflects its updates at once.
- A refit folds S into the basis and empties it. S never holds fit residuals.

**The per-tick queue.** The device queues the updates of edges that have no entry in S yet.
`compact_pending_deltas` merges them into S after every tick.
- Each entry is an s64 ordinal, an f32 value and an s32 plane index: 16 bytes.
- Capacity is `2 × edge_count × updated_plane_count`, plus the plasticity reserve.

**Tolerance.** `fit_tolerance` (default `1e-3`) is the worst relative error a fit may leave on
any value, measured against the larger of the value and its plane's RMS. At `1e-3`, every delay
up to 500 ticks stays on its exact tick.

## How fitting works

**Construction** (`declare_projections`):
- **Few distinct combinations.** When the projection runs carry at most 256 distinct
  combinations of initial values, the fit is exact: one lane per combination, shared by the
  planes that use it (`fit_basis_from_projections`).
- **Otherwise.** All coefficients are cleared, so every lane is free, and each plane is fitted on
  lanes of its own (`fit_plane`).

**Refit** (`refit`): only planes with entries in S are refit, each by `fit_plane`. A plane without
updates is left exactly as it is. S is then emptied.

**`fit_plane`**, for one plane:
1. **Its lanes.** A plane may change only its **own lanes**, the ones it reads and no other plane
   does.
   - A lane shared with another plane stays fixed, and its contribution is subtracted from the
     target.
   - A lane no plane reads is free to take.
   - So refitting one plane can never change another plane's values.
2. **Fewer lanes first, for memory.**
   - It warm-starts at the plane's current lanes, then doubles.
   - The sweeps are alternating least squares on only the rows of nodes at either end of a
     changed edge, weighted by `1 / max(|value|, RMS)²` so the least squares count what the
     tolerance counts.
   - It keeps the best sweep and stops as soon as every touched edge meets the tolerance.
   - A lane count fails after a 10-sweep stall window (`PLANE_FIT_STALL_WINDOW_SWEEP_COUNT`) or
     at most 100 sweeps (`PLANE_FIT_MAXIMUM_SWEEP_COUNT`).
3. **The exact solve.** Once the plane would reach `round_up(min(max out-degree, max in-degree)) +
   4` own lanes:
   - one side's rows (U or V) are filled with fresh N(0, 1) values;
   - each node on the other side gets the minimum-norm solution of its own few equations;
   - this reproduces any values to about 1e-6.
4. **In parallel.** Per-node solves run across all CPU cores.

**Defaults:**

| Setting | Default |
|---|---|
| `refit_occupancy_threshold_fraction` | `0.2` (it was `0.75`) |
| `minimum_ticks_between_refits` | `1000` |
| `refit_every_n_ticks` | `0` |

## Measured

Benchmarks used GLIF1 networks with alpha current synapses (4 planes). Every connection had a
random weight in U(0.5, 1.5) and a random delay of 5–30 ticks, the worst case for compression.
Plasticity was simulated with synthetic weight-plane updates on the edges of a random subset of
nodes, because the generated kernel does not stage plasticity updates yet.

| | 484 cells, 9,680 edges | 4,096 cells, 204,800 edges |
|---|---|---|
| Construction fit | 0.3 s for the whole engine build; weight and delay about 1e-6 off; every delay on its tick | 9.9 s (plus 19 s of NeuroML parsing) |
| Synapse refits | first 165 ms, then 6–8 ms | first 1.25 s, then 197 ms |
| Worst refit error (tolerance 1e-3) | 5e-6 | 4e-6 |
| Plasticity refits | 4 ms, about 2,000 edges updated | 128 ms, about 4,000 edges updated |
| Plasticity drift after 20 refits | 7e-6 against the exact running total | (5 cycles run) |
| Basis | 2.4× raw (24 lanes per plane, rank 96) | 2.2× raw (56 lanes per plane, rank 224) |

- **Spike statistics.** On the 484-cell network they match a run with no refits at all (S holding
  every update, so every value exact): 49.38 Hz against 49.44 Hz, with no neuron more than 2
  spikes apart.
- **The old refit,** on the uniform-weight version of the 484-cell network, took 163 s and left
  the weight and delay planes 1.4% off.

## Outstanding problems and limitations

### 1. Memory: values that differ independently per edge cannot be compressed

- **No factorization can store them in fewer numbers than raw.** A plane whose values vary
  independently from edge to edge carries `edge_count` independent numbers, and a basis of `rank`
  lanes holds about `2 × node_count × rank`. The floor is about 1× raw, which is about half the
  average degree in lanes per plane.
- **The exact solve stops at about twice that floor.** It uses one lane per edge per node plus
  4, which is 2.2–2.4× raw on the benchmarks.
- **Getting nearer the floor needs a better optimizer.** A fit that solves both U and V together
  could get close to 1.1–1.2× raw. The fewer-lanes sweeps don't reach it: at 16 lanes on the
  degree-20 network there were enough numbers in principle, but 100 sweeps didn't meet the
  tolerance.

Per edge, with 4 planes, on the 204,800-edge network:

| | Values | Topology | Update buffer | Total per edge |
|---|---|---|---|---|
| Raw values with a CSR index | 16 B | 4 B | none (values update in place) | about 20 B |
| WeightMatrix at the floor | about 16 B | about 3 B (k²-tree, random wiring) | — | about 19 B |
| WeightMatrix, measured | 36 B | about 3 B | about 100 B (S at 75% plus the queue) | about 140 B |

### 2. What counts as structure

The basis compresses a plane only when its values have structure, and **the structure has to be
the right kind. It has to be close to a sum of a few products of a source pattern and a target
pattern. Per-projection values, cell-type blocks, scaling by the source or by the target, and
position-based formulas all qualify. A rule that pairs each edge with its own value, like a lookup
table or a hash, doesn't, even though it isn't random.**

Examples of how many lanes a plane needs:

| Values | Lanes |
|---|---|
| Every edge the same | 1 |
| One value per projection or cell-type pair | about one per distinct combination |
| `a[source] × b[target]` | 1, even though every edge differs |
| Squared distance between 3D positions (`|x_i|² + |x_j|² − 2 x_i·x_j`) | 5 exactly; plain distance needs more |
| Independent per-connection draws | about one per edge per node; no compression |

- **The tolerance decides what counts as random.** Any per-edge variation larger than
  `fit_tolerance` has to be reproduced, so the basis needs separate numbers for each edge. A
  structured weight plus 5% per-connection jitter is incompressible at `1e-3`, but at a tolerance
  of `1e-1` the jitter could be ignored and the structured part would fit in a few lanes.
- **Synapse state inherits the structure of the weights and delays.** The state on an edge is
  roughly weight × a time course that starts when its spike arrives.
  - With per-projection weights and delays, it depends only on the source and compresses.
  - With per-connection delays, each edge's time course is shifted by its own delay, so it
    doesn't.
- **Many published models draw per-connection weights and delays independently.** One example is
  the Potjans–Diesmann cortical microcircuit, which draws both from normal distributions.
- **Structured networks are not measured yet** with the new refit. The estimate is a basis about
  10× under raw: the uniform test network fit in 4 lanes, 15.5 KB against 155 KB. If the
  fewer-lanes sweeps stall on such a network, the plane jumps to the exact solve and over 2× raw.

### 3. S cannot stay at a fixed size while synapse state changes on most edges

- **S grows rather than lose an update.** When a tick's updates don't fit, `compact_pending_deltas`
  enlarges S, and the next refit shrinks it back to the threshold's size. The 20% default
  therefore sets S's size after a refit, not its peak.
- **One tick's updates set the floor.** S must hold at least one tick of updates. On the 484-cell
  network, one tick after S is emptied, 90% of edges already have an update in each
  synapse-state plane (8,720 of 9,680): every active synapse's state changes every tick, and at
  49 Hz nearly every synapse stays active. So S's peak is about 90% of edges whatever the refit
  spacing.
- **Refitting every tick doesn't help.** With `minimum_ticks_between_refits = 0`, a refit ran on
  almost every tick (19,781 of 20,000) and the run took 162 s instead of 25 s. Peak memory
  stayed the same, because S still grows within the tick.
- **The queue has the same floor.** It is sized for 2 entries per edge per updated plane so no
  tick can overflow it, and it is the largest single buffer: 9.8 MB on the 204,800-edge network at
  12-byte entries.
- **A fixed small S works** only when few edges change per tick: plasticity-only weight changes,
  or networks where most synapses sit at rest.
- **Shrinking this floor** needs a change to how per-tick synapse state reaches the weight
  matrix.

Estimated totals for the 204,800-edge network, with S at 20% and the queue at 12-byte entries.
The structured basis is an estimate.

| | Structured values | Per-connection random values |
|---|---|---|
| Basis | ~0.4 MB | 7.3 MB |
| k²-tree | ~0.55 MB | ~0.55 MB |
| S at 20% | 2.0 MB | 2.0 MB |
| Queue | 9.8 MB | 9.8 MB |
| Total | ~12.7 MB (3.9× raw) | ~19.7 MB (6.0× raw) |
| Total with the queue cut to 20% | ~4.9 MB (1.5× raw) | ~11.8 MB (3.6× raw) |

Raw values are 3.3 MB. Because of the floor above, S at 20% and a 20% queue are not reachable
while most synapses are active.

S is also allocated for every plane, including planes that never update (weight and delay
without plasticity).

### 4. The 256-lane limit

- **The tolerance is only guaranteed while there are enough lanes.** Every changing plane needs
  `largest degree + 4` lanes for the exact solve, all within 256 in total. With 4 changing planes
  that ends at a degree of about 60.
- **Denser networks** fall back to the fewer-lanes sweeps up to the lane limit. They can miss the
  tolerance, and log a warning when they do. No value is stored anywhere else.

### 5. Kernel read cost

Each read of a plane sums over every lane in the basis (96–224 on the benchmarks), even though only
that plane's own lanes (24–56) are non-zero. So the GPU does about 4× more work per edge read than
it needs to. The effect on tick time is not measured.

### 6. Wasted fewer-lanes attempts

On per-connection random values the fewer-lanes sweeps never met the tolerance. They took about 9 s
of the 204,800-edge network's construction and most of its first refit. They could be capped or
dropped.

### 7. Plasticity refits re-solve every node

Once a plane has reached the exact solve, every later refit of it re-solves every node, although
only the nodes with updates need it. Re-solving only those should make plasticity refits
proportional to the number of updates. Not measured.

### 8. Drift

- **The exact solve** reproduces values to about 1e-6, so drift over many refits stays small
  (7e-6 after 20 plasticity refits).
- **The fewer-lanes sweeps** may leave up to `fit_tolerance` of error on each refit, and that can
  accumulate over many refits. Not measured.

### 9. Other

- **Plasticity is not generated yet.** The generated kernel does not stage plasticity updates, so
  weights do not change during a run even with plasticity enabled.
- **`accumulate_edge_delta` is slow per call.** It rebuilds its plane's CSR on every call, so a
  call costs time proportional to S. The device path through the queue does not.
- **Parsing large networks is slow.** A 25 MB NeuroML file with 204,800 `connectionWD` elements
  took 19 s on one core.
- **Tests.** `tests/weight_matrix_tests.cpp` does not compile and uses removed APIs
  (`fit_rank_budget`, the old rank search). No test covers `fit_plane`, the per-plane lanes, or
  the exact solve.
