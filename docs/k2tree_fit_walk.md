# Fits that walk the k²-tree

Implemented 2026-10-16. A refit keeps no array per connection: it reads the connections by walking
the k²-tree and computes each connection's target where it reads it. Construction and calibration
read a temporary list of the connections instead (12 bytes per connection, freed when the plane's
fit ends; a lane-search probe adds a 4-byte residual per connection while it fits its new lanes),
because walking made construction fits 13 to 15 times slower and construction memory matters
least.

## What a fit reads

A fit works on one plane's own lanes, copied into working arrays (one row per node). For each
connection it needs the connection's ends and the value the own lanes must hold there.

- **Connections:** a fit walks a source's row of the k²-tree for its outgoing connections (the U
  side) and a target's column for its incoming ones (the V side). The kernels walk both with
  `fit_walk_next` (`weight_fit_kernels.metal`, `weight_fit_kernels.cu`): the descent of
  `k2t_next_neighbor` along a row, or of `k2t_next_predecessor` along a column, split among
  walkers. The host fallback uses `get_neighbors` and `get_predecessors`.
- **Target:** what the own lanes must hold at `(source, target)`, in units of the plane's scale:
  the basis over `target_lanes` plus the connection's entry in the fit's values (`fit_target` in
  `weight_fit_kernels.metal` and `weight_fit_kernels.cu`, `Weight_Matrix::fit_target` on the
  host).
- **The fit's values** (`Weight_Matrix::Fit_Values`) are CSR over source rows, each row keyed by
  target node and sorted, so a row walk reads them with a cursor and a column walk by binary
  search.
  - **Refit:** the plane's S entries. Before the plane's fit, `key_plane_updates_by_target`
    rewrites its keys from edge ordinals to target nodes in place (within a row both orders agree).
    The refit empties S at the end. Ticks, the queue, the merge, save and load still use ordinals.
    The targets are the plane's own lanes (its values before the fit) plus the updates, so a
    second fit from fresh rows is tried before either is stored.
  - **Construction and calibration:** every connection's value in a slab of their own
    (`plane_values_from` with `list_every_connection`), one plane at a time: target and value per
    connection in edge-ordinal order, plus each node's incoming connections (`incoming_offset`,
    `incoming_entry`), 12 bytes per connection. The kernels then read rows and columns from these
    arrays instead of walking (`connections_listed`). The lanes the plane shares with other planes
    count against the values (there are almost never any).
- **Error:** a refit measures the plane's worst error over every edge before it stores the new
  lanes, while the lanes still hold the values the updates add to. Construction keeps the error
  of the search's chosen fit, whose probes touch every edge.

## How the kernels walk

A walk is a chain of dependent reads (bits, rank tables and the walk's stack), so one walk per
row leaves the GPU waiting. Each row is split among `walkers_per_row` walkers, each walking its
share of the other ends and skipping the subtrees outside it. There is about one walker per 16
connections of an average row, as a power of two from 1 to 32: 32 for M1 and M3, 4 for M2, and
1 at 20 connections per row.

- **Residual** (`fit_walk_residual_kernel`): one thread per walker, each on its own.
- **Row solve and gradient** (`fit_walk_row_solve_kernel`, `fit_walk_gradient_kernel`): a SIMD
  group takes 32 / walkers_per_row rows. In each round every walker finds its next connection,
  then the group handles the round's connections one at a time, each thread taking its share of
  the lanes of that connection's row. The Givens rotations fold into each row's own R; the
  gradient sums build up in the gradient buffer, each value written by one thread only.
- The touched connections of a refit are the moved sources' row walks plus the moved targets'
  column walks, which skip connections whose source moved.
- With the connections listed, each walker reads its share of the node's list instead of the
  tree; a column's entries give their source by binary search of the row starts.

## The lane search

`search_fewest_lanes` doubles the lanes until a fit meets the tolerance, then bisects. It also
stops doubling once two doublings in a row each cut the worst error by less than 5%, and keeps
the closest fit. Values that more lanes barely help, random per-connection spread for example,
would otherwise run to the lane cap with ever costlier probes. The first probe starts from fresh
rows (each row's and column's mean target). Every later probe, doubling or bisecting, starts from
the fit at the last lane count that missed the tolerance: its lanes as they are, and the new lanes
first fitted alone to what that fit leaves (commits `642dc7d` and `a15a542`).

## Measurements (M5 Max)

Refit memory is what one refit after a run added to the process footprint and to the device
buffers; time is that refit's wall time.

| Network | Refit before | Refit after |
|---|---|---|
| 10k neurons, 200k connections | +5 MB host, +5 MB device, 0.08 s | +2 MB, +1 MB, 0.05 s |
| M2 stand-in, 318k connections | +11 MB, +8 MB, 0.05 s | +10 MB, +2 MB, 0.02 s |
| M1 stand-in, 15.7M connections | +109 MB, +117 MB, 0.72 s | +0 MB, +1 MB, 0.13 s |

Refits got faster because building the fit graph was most of a refit's time.

Walking costs about 1.4 ns per connection on M1's rows (1,250 connections each) and 4 to 6 ns on
M2's (80 each), against about 0.3 ns to read an array. Profiling a construction (Instruments'
Metal System Trace) showed the GPU busy 97% of the time, so construction moved to the connection
list:

| Network | Construction, fit graph | Walking | Connection list |
|---|---|---|---|
| M2 stand-in | 9.5 s | 26.9 s | 10.4 s |
| M1 stand-in | 7.5 s, device peak 0.30 GB | 7.4 s, 0.19 GB | 5.8 s, 0.25 GB |

Measured before the warm start (`642dc7d`) and the residual pre-fit (`a15a542`): on a weight
matrix of 10M connections in eight Potjans-Diesmann populations, a limited-memory
BFGS iteration costs about 9 ms with the list against 61 ms walking. With 10% weight spread and
50% delay spread at a 10% tolerance, construction takes 7.8 s: the lane search stops at 8 lanes
on the weight plane (76% worst error) and 1 on the delay plane (86%), where it used to double
toward its 1,496-lane cap. No fit reaches 10% there: the exact population means already leave
86%, since a few connections sit that far from any smooth fit.

The fit walk is the same descent as the tick's `k2t_next_neighbor` (performance review, item 8),
but separate code, so a faster walk must go into both. Replacing its runtime divisions with shifts
(branching factor 4) saved 20 to 25% in a test.
