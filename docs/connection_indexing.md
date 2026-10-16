# How connections are stored and indexed

Every connection gets a number, its edge ordinal, and everything else about it is looked up by
that number. This page walks through the pieces in the order they are used. The code is mostly in
`src/spikecorec/core/k2tree.cpp` and `src/spikecorec/core/weight_matrix.cpp`.

## Terms

- **Edge ordinal:** a connection's position in the fixed order of all connections.
- **Plane:** one quantity stored for every connection: the weight, the delay, or one synapse
  state variable.
- **Lane:** one of the shared building blocks that the compression combines to rebuild every
  plane. A lane has no meaning of its own; more lanes give more detail.
- **Run:** a consecutive block of connections that start with the same declared values.

## 1. Who connects to whom: the k²-tree

The k²-tree stores which neurons connect to which: "i connects to j", with no values. Given a
neuron, it lists that neuron's targets in ascending order.

It can also list a neuron's sources (`get_predecessors`), but only on the CPU. Nothing in the
engine uses that yet, and the GPU helpers only walk targets.

## 2. Numbering the connections: edge ordinals

Go through the neurons in order (0, 1, 2, …). For each one, take its targets in ascending order
and number the connections 0, 1, 2, … as you go.

Example: neuron 0 connects to {2, 3} and neuron 1 to {2}.

| Connection | Ordinal |
|---|---|
| 0→2 | 0 |
| 0→3 | 1 |
| 1→2 | 2 |

`edge_row_offset[i]` is the ordinal of neuron i's first connection: 0 for neuron 0 and 2 for
neuron 1. A connection's ordinal is therefore

```
edge_row_offset[source] + its position among that source's targets
```

## 3. The values each connection carries: planes

Each connection has a weight, a delay and one value per synapse state variable. Each of these is a
plane. Keeping one array per plane, with a value for every connection, is too big for large
networks, so all the planes are compressed into three tables:
- **U:** one row per neuron, as a source;
- **V:** one row per neuron, as a target;
- **C:** one row per plane.

The value of connection i→j on plane k is

```
value(i→j, k) = Σ_lane  U[i, lane] · C[k, lane] · V[j, lane]
```

The columns shared by U, V and C are the lanes. The number of lanes is the rank.

## 4. Values that change during a run: S

Synapse state changes every tick, and the compressed tables cannot be edited for a single
connection. Changes therefore go into S: for each plane, a list of "connection ordinal → amount
changed".
- **Reading** a value gives the compressed value plus that connection's entry in S, if it has one.
- **A refit** from time to time folds S back into U, V and C and empties S.

On the GPU, the kernel writes its changes into a pending queue. After each tick the host merges
that queue into S (`compact_pending_deltas`).

## 5. Where each connection came from: runs

Consecutive connections with the same synapse type, weight and delay form one block of ordinals: a
run. For each run the code stores:

| What is stored | Field |
|---|---|
| Its first ordinal | `run_first_edge_ordinal` |
| How many connections it has | `run_edge_count` |
| Its synapse type | `run_synapse_prototype` |
| Its starting values | the initial values passed to `declare_runs` |

A binary search of `run_first_edge_ordinal` with a connection's ordinal finds its run, and
from that its synapse type and starting values.

## 6. Building it at the start: fitting

The runs give every connection's starting values. U, V and C are then chosen to reproduce them,
one plane at a time, each on lanes of its own. For each plane, construction fits from scratch at
1, 2, 4, ... lanes until one is within `fit_tolerance`, then bisects for the fewest lanes that are.

Each fit runs one of two candidates, chosen by `Weight_Matrix::fit_method`: masked alternating
least squares (`fit_plane_by_masked_alternating_least_squares`, the default) or limited-memory
BFGS (`fit_plane_by_limited_memory_broyden_fletcher_goldfarb_shanno`). Both minimize the same
weighted, ridge-regularized squared error over the connections, in f32, on the GPU.

## 7. One tick on the GPU

Each neuron's thread:
1. walks its k²-tree row to get its targets in order, counting ordinals as it goes;
2. for each connection, finds the synapse type from the runs;
3. reads the weight, delay and synapse state from U, V and C plus S;
4. updates the synapse state through the pending queue;
5. when a spike arrives, adds the synapse's current to the target's `network_inputs`.
