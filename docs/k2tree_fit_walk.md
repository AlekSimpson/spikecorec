# Fits that walk the k²-tree

Implemented 2026-10-16. A weight-matrix fit (construction, calibration and every refit) keeps no
array per connection. It reads the connections by walking the k²-tree and computes each
connection's target where it reads it.

## What a fit reads

A fit works on one plane's own lanes, copied into working arrays (one row per node). For each
connection it needs the connection's ends and the value the own lanes must hold there.

- **Connections:** a fit walks a source's row of the k²-tree for its outgoing connections (the U
  side) and a target's column for its incoming ones (the V side). The column walk is
  `k2t_next_predecessor` in `k2tree_device.metalinc`, the mirror of `k2t_next_neighbor`.
- **Target:** what the own lanes must hold at `(source, target)`, in units of the plane's scale:
  the basis over `target_lanes` plus the connection's entry in the fit's values (`fit_target` in
  `weight_fit_kernels.metal`, `Weight_Matrix::fit_target` on the host).
- **The fit's values** (`Weight_Matrix::Fit_Values`) are CSR over source rows, each row keyed by
  target node and sorted, so a row walk reads them with a cursor and a column walk by binary
  search.
  - **Refit:** the plane's S entries. Before the plane's fit, `key_plane_updates_by_target`
    rewrites its keys from edge ordinals to target nodes in place (within a row both orders agree).
    The refit empties S at the end. Ticks, the queue, the merge, save and load still use ordinals.
    The targets are the plane's own lanes (its values before the fit) plus the updates, so a
    second fit from fresh rows is tried before either is stored.
  - **Construction and calibration:** the plane's nonzero values in a slab of their own
    (`plane_values_from`), 8 bytes per nonzero value, one plane at a time. The lanes the plane
    shares with other planes count against the values (there are almost never any).
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

## Measurements (M5 Max)

Refit memory is what one refit after a run added to the process footprint and to the device
buffers; time is that refit's wall time.

| Network | Refit before | Refit after |
|---|---|---|
| 10k neurons, 200k connections | +5 MB host, +5 MB device, 0.08 s | +2 MB, +1 MB, 0.05 s |
| M2 stand-in, 318k connections | +11 MB, +8 MB, 0.05 s | +10 MB, +2 MB, 0.02 s |
| M1 stand-in, 15.7M connections | +109 MB, +117 MB, 0.72 s | +0 MB, +1 MB, 0.13 s |

Refits got faster because building the fit graph was most of a refit's time.

Construction fits got slower. A walk costs about 1.4 ns per connection on M1's rows (1,250
connections each) and 4 to 6 ns on M2's (80 each), against about 0.3 ns to read an array, so a
limited-memory BFGS iteration costs more:

| Network | Construction before | Construction after |
|---|---|---|
| 10k neurons | 4.7 s | 5.0 s |
| M2 stand-in | 9.5 s | 26.9 s (about 900 iterations per lane-count probe) |
| M1 stand-in | 7.5 s, device peak 0.30 GB | 7.4 s, device peak 0.19 GB |

The walk is the same one the tick's connection loop uses (performance review, item 8), so making
it faster speeds up the fits and the tick alike. Replacing its runtime divisions with shifts
(branching factor 4) saved 20 to 25% in a test.
