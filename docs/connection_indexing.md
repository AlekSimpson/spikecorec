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

## 4. Values that change during a run: S and the queue

Synapse state changes during a run, and the compressed tables cannot be edited for a single
connection. Changes therefore go into S, a list of "connection ordinal → amount changed":
- **Layout:** one pool shared by every plane, plane after plane. Within a plane, entries are
  grouped by source neuron and sorted by ordinal, so a plane without changes takes no room.
- **Reading** a value gives the compressed value plus that connection's entry in S, if it has one.
- **A refit** from time to time folds S back into U, V and C and empties S.

On the GPU, a thread changes an entry already in S in place. A change to a connection with no
entry yet goes into the update queue, and after each tick the host merges the queue into S
(`compact_pending_deltas`). Before walking its connections, a neuron reserves queue room for every
change it can make; when the queue is full, its walk runs in another pass of the same tick, after
a merge, so no change is lost.

## 5. Where each connection came from: runs

Consecutive connections with the same synapse type, weight and delay form one block of ordinals: a
run. `declare_runs` takes, for each run:

| What is given | Field |
|---|---|
| Its first ordinal | `run_first_edge_ordinal` |
| How many connections it has | `run_edge_count` |
| Its synapse type | `run_synapse_prototype` |
| Its starting values | the initial values passed to `declare_runs` |

Once the starting values are fitted, only the synapse type is needed, so neighbouring runs with
the same type merge into one. A binary search of `run_first_edge_ordinal` with a connection's
ordinal finds its run, and from that its synapse type.

## 6. Building it at the start: fitting

The runs give every connection's starting values. U, V and C are then chosen to reproduce them,
one plane at a time, each on lanes of its own:
- **Lane budgets:** each plane gets a fixed number of lanes: the fewest that hold its starting
  values within `fit_tolerance` (found by fitting at 1, 2, 4, ... lanes, then bisecting), times
  one plus `expected_active_edge_fraction`. Together they stay below one float per connection per
  plane.
- **Planes that start at zero** (synapse state) take no lanes at first. The engine spikes every
  neuron once, runs until the synapses settle, and sizes their budget from the values they take,
  then restores the starting state (calibration).
- **Methods:** the lane searches use limited-memory BFGS, which needs fewer lanes; every refit
  during the run uses masked alternating least squares, which needs far less scratch memory. Both
  minimize the same weighted, ridge-regularized squared error over the connections, in f32, on
  the GPU.

## 7. One tick on the GPU

Each neuron's thread:
1. walks its k²-tree row to get its targets in order, counting ordinals as it goes;
2. for each connection, finds the synapse type from the runs;
3. reads the weight, delay and synapse state from U, V and C plus S;
4. sums each state variable's change for the tick, and writes it once: in place in S, or through
   the queue;
5. adds the synapse's current to the target's `network_inputs`.
