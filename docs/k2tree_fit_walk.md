# Proposal: fits that walk the k²-tree instead of building a fit graph

Not implemented. This records the design discussed on 2026-10-15 for removing the last
per-connection arrays a weight-matrix fit allocates.

## Where things stand

Every construction fit and every refit builds a **fit graph** from the k²-tree, uses it for the
fit and frees it at the end (`Weight_Matrix::build_fit_graph`):

| Array | Size | What reads it |
|---|---|---|
| `edge_target` | 4 bytes per connection | a source row's connections: U-side solves and gradients, residuals |
| `incoming_edge` | 4 bytes per connection | a target's incoming connections, as ordinals: V-side solves and gradients |
| `outgoing_offset`, `incoming_offset` | 4 bytes per neuron each | where each neuron's lists start |

An edge's source is not stored: the fits find it from `outgoing_offset` by binary search.

That is 8 bytes per connection for the length of a fit, a quarter of twice CSR (CSR and CSC with
float values) at 3 planes. A fit also holds, per connection, the plane's targets and the weighted
residuals (`residual_target`, `weighted_residual`) and, during a refit, the list of connections
with a moved end.

The fit graph exists because a fit needs two traversals:
- **Outgoing:** for each source row, its connections in ordinal order, to solve or take the
  gradient of its U row.
- **Incoming:** for each target row, its incoming connections, with each one's source and
  ordinal, to solve or take the gradient of its V row. The ordinal indexes the target values, the
  residuals and S.

Flat arrays make both cheap. The k²-tree can answer both too, but not as cheaply.

## The proposal

### 1. Outgoing side: walk the row

The fit kernels walk each source's k²-tree row with `k2t_next_neighbor`, exactly as `master_step`
does each tick. The walk yields targets in ordinal order, so a connection's ordinal is
`edge_row_offset[source] + slot`. This replaces `edge_target`.

### 2. Incoming side: walk the column

A device column walk, a port of `collect_column_predecessors` (`k2tree.cpp`, host only today)
next to `k2t_next_neighbor` in `k2tree_device.metalinc`, yields a target's sources in ascending
order. This replaces `incoming_edge` and both offset arrays.

The column walk gives the source but not the connection's ordinal, and the ordinal is what the
target values, the residuals and S are indexed by. The three ways around that:
- **Key S by target node instead of ordinal.** Within a source's row, ordinal order is target
  order, so a row's S entries sorted by target support the same binary search, looked up by
  `(source, target)`. The tick already knows both. This is the change that makes the rest work
  without ordinals.
- **Compute targets and residuals on the fly.** During a refit, a connection's target value is
  the plane's old basis value plus its S entry. With a copy of the plane's own U and V rows
  (per neuron, not per connection) and S keyed by target, a kernel can compute it from
  `(source, target)` alone, and the residual from it. That removes `residual_target`,
  `weighted_residual` and the moved-connection list.
- **A row-rank query** (the number of the source's targets below a given target) would give the
  ordinal directly. It is another k²-tree walk per connection, so it costs the most time.

### 3. Construction

At construction the target values come from the runs, which are indexed by ordinal. The outgoing
side has the ordinal from its row walk. The incoming side needs either the row-rank query, or a
temporary array of each connection's starting value (4 bytes per connection, construction only,
which is the lowest priority).

## What it would save and cost

- **Memory:** a fit's per-connection scratch goes from about 16–20 bytes (fit graph, targets,
  residuals, the moved list) to nothing during refits. What remains is per neuron: the plane's
  working U and V, the old rows, and the row-solve scratch.
- **Time:** every traversal becomes a k²-tree walk instead of an array read, with data-dependent
  loops, runtime divisions and popcounts (performance review, item 8), and the column walk visits
  more of the tree than a row walk. A masked least squares sweep traverses every connection twice,
  and an L-BFGS iteration several times, so fits would be several times slower until the walks
  themselves are made faster.
- **Risk:** S keyed by target changes the tick's S lookup and the host merge, and every fit kernel
  changes. The fit results should not change, so the existing tests cover it.

## Order of work

1. Key S by target node (tick lookup, queue entries, the host merge, save and load).
2. Device column walk in `k2tree_device.metalinc`, checked against `get_predecessors`.
3. Fit kernels: row walks for the outgoing side, column walks for the incoming side.
4. Targets and residuals computed on the fly during refits; drop `residual_target`,
   `weighted_residual` and the moved-connection list.
5. Construction: decide between the row-rank query and a temporary starting-value array.
