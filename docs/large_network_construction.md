# Building the weight matrix for large networks (M3 investigation)

Notes from 2026-10-16, written to pick the work up the next day. The problem: building the weight
matrix for the full Potjans-Diesmann microcircuit (M3) takes hours, almost all of it in the
construction lane search, and almost all of that in the delay plane.

## The network

M3 at full scale:
- 77,169 neurons in 8 populations, and 298,880,968 synapses (fixed total number per projection).
- 284,818,746 distinct ordered pairs: 13.4M pairs repeat, at most 7 times, and 6,651 are self-connections.
- Multiplicity goes in the connection's synapse (`delta_current_synapse_xk`, current step k pA), so the weight matrix holds one connection per pair.

The three conditions (from the reference agent's `m3_parameters.json`):
- **t3b_structured:** low-rank weights and delays.
  - weight(s->t) = mean_weight[P(t)][P(s)] * (1 + 0.1 * (a_s + b_t) / sqrt(2));
  - delay(s->t) = mean_delay[type of P(s)] * (1 + 0.5 * (c_s + d_t) / sqrt(2)), clipped at max(delay, 0.1 ms), then rounded to whole steps: floor(delay_ms * 10 + 0.5).
  - a, b, c and d are standard normals per neuron. The mean delay is 1.5 ms for excitatory sources and 0.75 ms for inhibitory ones.
- **t3b and t3a:** each synapse's weight and delay drawn independently. Weights are normal(mean, 0.1 |mean|) with the sign kept; delays are normal(mean, 0.5 mean), redrawn until at least 0.05 ms, then rounded.

SpikeCore's model and drivers: `~/Documents/neurocomputation/spikecore_runs/M3/` (on spark: `~/neurocomputation_benchmarks/spikecore_M3/`).
- `m3_microcircuit_model.py`, `run_spikecore_m3.py`, `timing_check.py`;
- `fit_diagnostic.py`: the reduced-network timing used below.

## Which fit runs where

| Phase | Algorithm | Setting | Default |
|---|---|---|---|
| Construction: each plane's lane count (`declare_runs`, `calibrate_lane_budget`) | L-BFGS, strong-Wolfe line search | `Weight_Matrix::lane_search_fit_method` | L-BFGS |
| Refits during a run (when S passes its occupancy threshold) | masked ALS (rows by streaming Givens QR on the GPU) | `fit_method` (Spike_Engine argument) | ALS |

**Construction lane search** (`search_fewest_lanes`), per plane:
- It doubles the lane count (1, 2, 4, ...) until a fit meets `fit_tolerance`, then bisects between the last count that missed and the first that met.
- Each probe stops at the tolerance, at a stall (less than 5% better over 100 iterations), or at 2,000 iterations.
- Doubling also stops after two doublings that each cut the worst error by less than 5%.
- The plane's budget is then the lanes found × (1 + `expected_active_edge_fraction`, default 0.2), within the cap. The cap: total lanes never exceed one float per connection per plane.
- L-BFGS was chosen here because it holds structured planes on fewer lanes than ALS, and the runtime footprint comes first.

**Refits** use ALS within the fixed budgets:
- warm, moving only the rows whose connections changed;
- a warm miss worse than twice the plane's last error is retried fresh.

**Cost model.** Every L-BFGS iteration is a full pass over every connection (construction fits read a temporary connection list, 12 bytes a connection), and costs roughly in proportion to connections × lanes. Probes near the minimum lane count take 1,300-1,900 iterations.

**How the tolerance is measured.** A connection's error is |fitted - value| / max(|value|, plane RMS). Below the plane's RMS it is an absolute error of 0.1% of the RMS; above it, a relative one. For the delay plane, whose RMS is about 15 ticks, values below the RMS (every clipped delay among them) must be held to about 0.015 ticks.

## Measurements

**Full size** (t3b_structured, 285M connections): stopped after 1 h 40 min on the Mac (Metal), and after 1 h 15 min on spark (CUDA). Both were still in the lane search.
- Log lines printed nothing during it: the engine's console log goes through stdio's `stdout`, which is block-buffered when it goes to a file. Run diagnostics under `script` to see them line by line.

**3% of the network** (8,955,733 connections, `fit_diagnostic.py rounded 0.03`, Mac, sharing the GPU with M1 runs):

| | Fresh probes (before `642dc7d`) | Warm probes (`642dc7d`) |
|---|---|---|
| Weight plane | 8 lanes, 82 s | 4 lanes, 21 s |
| Delay plane | 129 lanes (522 s); over the cap, cut to 106 and refit at 0.19% | 66 lanes, 538 s |
| Budgets (searched -> budget) | [8 -> 10, 129 -> 106], cap 116 | [4 -> 5, 66 -> 80], cap 116 |
| Worst error | 0.19%, misses the 0.1% tolerance | 0.10% |
| Basis | 71.6 MB, the same as one float per connection per plane | 54.3 MB |
| Construction | 727 s | 564 s |

**The warm delay plane's probes** (continuous delays; the rounded run's are nearly the same):

| Lanes | Iterations | Worst error | Stopped at | Seconds |
|---|---|---|---|---|
| 1 | 32 | 70.7% | failed line search | 1.1 |
| 2 | 114 | 49.1% | failed line search | 2.3 |
| 4 | 278 | 31.8% | stall | 2.6 |
| 8 | 297 | 17.0% | stall | 3.0 |
| 16 | 468 | 5.5% | stall | 5.7 |
| 32 | 647 | 3.9% | stall | 10.4 |
| 64 | 2,000 | 0.28% | iteration cap | 58.1 |
| 128 | 378 | 0.099% | tolerance | 23.3 |
| 96 | 654 | 0.100% | tolerance | 30.8 |
| 80 | 821 | 0.100% | tolerance | 36.9 |
| 72 | 1,314 | 0.100% | tolerance | 61.1 |
| 68 | 1,478 | 0.100% | tolerance | 71.0 |
| 66 | 1,870 | 0.100% | tolerance | 91.7 |
| 65 | 1,813 | 0.128% | stall | 88.4 |

The bisection (96 down to 65 lanes) is 380 of the plane's 487 s. Probes get dearer as they near the minimum. A passing probe stops right at the tolerance, so its error says nothing about how far below the minimum the count could go.

**10% on spark** (fresh probes, 28.5M connections): the weight plane keeps 5 lanes, found in 517 s. The delay plane was stopped before it finished.

**Continuous delays** (the formula clipped at 0.1 ms but not rounded): the delay plane needs the same 66 lanes as the rounded delays at 3%. Grid rounding is not what costs lanes. That idea is dropped.

## The delay plane: the clip at 0.1 ms is what costs lanes

Without the clip, the structured delay is exactly rank 2:

    delay(s->t) = m_s (1 + 0.5 c_s / sqrt 2) * 1  +  (0.5 m_s / sqrt 2) * d_t

where m_s is the source's mean delay. In lanes: u = [m_s (1 + 0.5 c_s / sqrt 2), 0.5 m_s / sqrt 2] and v = [1, d_t]. The weights have the same shape, and their clip (sign) never binds, since that needs a 10-sigma deviation. That is why the weight plane fits in 4-5 lanes.

The clip max(delay, 0.1 ms) binds whenever (c_s + d_t) / sqrt 2 < -(1 - 0.1 / m_s) / 0.5:
- excitatory sources (m_s = 1.5 ms): below -1.87 sigma, 3.1% of their connections;
- inhibitory sources (m_s = 0.75 ms): below -1.73 sigma, 4.2%.

So about 3-4% of connections, scattered over the matrix, sit at 1 tick where the rank-2 formula gives anything from 1 tick down to well below zero. At -3 sigma an excitatory delay's formula is -0.53 ms, 6 ticks below the clipped value.
- The fit has to hold every one of those points to about 0.015 ticks (see how the tolerance is measured).
- Scattered exceptions have no low-rank structure, so they take many lanes: 66 at 3% of the network, and more as each row holds more of them.

### The open question: store the unclipped delay and let the kernel clip it

**The kernel already clips.** In `master_step` (`kernel_boilerplate.cpp`, both the Metal and CUDA text), a connection's delay is read from the delay plane and turned into ticks:

    delay_ticks = round(fitted delay plane value)
    if (delay_ticks < 1) delay_ticks = 1
    if (delay_ticks > spike_history_length - 1) delay_ticks = spike_history_length - 1

**Unclipped storage would deliver exactly the reference's delays.** If the delay plane held the unclipped formula, the kernel would deliver the reference's delays:
- For d >= 0.1 ms: the reference's floor(10 d + 0.5) equals round(10 d), since round() rounds halves away from zero and d > 0.
- For 0.05 <= d < 0.1 ms: the reference clips to 0.1 ms (1 tick), and round(10 d) is 1.
- For d < 0.05 ms, zero and negative included: the reference gives 1 tick, and round() gives 0 or less, which the kernel clamps to 1.

The plane would then hold the rank-2 formula. It should fit in about as few lanes as the weight plane (4-5 at 3%), and its search would take seconds to minutes instead of hours.

**What it changes in the engine.** Today a given delay must be finite and not negative. These places would change:
- `apply_topology` refuses a negative delay ("each must be finite and not negative").
- `units::seconds_to_ticks` / `ms_to_ticks` throw on negative input.
- Every negative delay would have to mean "delivered after one tick", the same as a delay that rounds to zero today. That would hold:
  - at construction: the longest delay (`maximum_edge_delay`), and the shortest delay that the weighted-arrival walk check reads;
  - in the sampled connections codegen reads;
  - in the kernel, which already clamps.
- The Python constructors and `API_REFERENCE.md` would document it.
- Tests: a negative delay delivers after one tick, exactly like 0.1 ms. The structured M3 delay plane fits in a few lanes.

**Considerations:**
- A negative delay has no physical meaning. It would be accepted only as "the formula before the one-tick floor". An alternative with the same effect is an explicit option that says the given delays are to be floored at one tick.
- Rounding a fitted delay is already approximate today. A connection whose exact delay lies within the fit error (about 0.015 ticks) of a half-tick boundary can land one tick off. That holds whether the plane stores the clipped or the unclipped values, so the change adds no new kind of error.
- Clipping at 0 instead of 0.1 ms needs no engine change, but still leaves about 2.3% of connections clipped, below -2 sigma. That remains the same kind of scattered exception, so it is not a fix.
- It only helps the structured condition. In t3b and t3a every delay is an independent draw, which no low rank holds (next section).

## The random-spread conditions (t3b, t3a)

Every connection's weight and delay is an independent draw, so neither plane has low-rank structure beyond the per-projection means. The lane search will climb while each doubling still helps by 5%, possibly toward the cap. At full size the cap is about 3,700 lanes (one float per connection per plane), and fits there over 285M connections would take days.
- On an M3-like 10M-connection benchmark with random spread, the CUDA agent saw 358 lanes after 18 fits (1,130 s).
- The Mac's run of the same benchmark stopped at 8 lanes. Its network differed: libc++ and libstdc++ distributions draw different networks.
- Not yet measured on M3 itself. The next step is `fit_diagnostic.py` at 3% with the random spread.

## Ways to cut construction time

| Idea | Status | Effect at 3% |
|---|---|---|
| Warm-start each probe from the last fit that missed | committed `642dc7d` | weight plane 4x faster with half the lanes; delay plane half the lanes, about as slow; build 22% faster |
| 1. Residual pre-fit: fit only the new lanes against the values less the start fit (a construction-only buffer, 4 B a connection), then the joint fit | committed `a15a542`; test suites not yet run on it | spark, 3%: build 1,017 s -> 688 s (-32%); weights 6 -> 4 lanes, delays 66 -> 69 lanes, same basis size |
| 2. Estimated probes: the next bisection probe where the last two failing errors, on a log scale, reach the tolerance | tried and dropped (code removed) | alone 927 s (-9%) but 2 more delay lanes; together with idea 1, 755 s, slower than idea 1 alone |
| Truncate the passing fit to the probe's lane count (best rank-n approximation of its product, from QR of U and V and a small SVD) and start there | not tried | might cut the iterations of each bisection probe |
| Search with ALS (fast), then let L-BFGS refine near the count it finds | not tried | |
| Search on a random sample of the connections, then fit once on all and check the tolerance | not tried | a sample changes how many lanes noise-like values seem to need, so the count may be off |
| Store unclipped structured delays and let the kernel clamp (above) | needs a decision | delay plane expected at a few lanes |

## Lane-search variants on spark (CUDA, 3%, rounded delays, 2026-10-16 evening)

| Variant | Weight lanes | Delay lanes | Basis | Build |
|---|---|---|---|---|
| Warm (`642dc7d`) | 6 | 66 | 54.3 MB | 1,017 s |
| Warm + residual pre-fit | 4 | 69 | 54.3 MB | 688 s |
| Warm + estimated probes | 6 | 68 | 56.8 MB | 927 s |
| Both | 4 | 69 | 54.3 MB | 755 s |

- Logs: spark `~/neurocomputation_benchmarks/spikecore_M3/logs/lane_search_experiment/`.
- Spark's fits run about 2-2.5x slower than the Mac's at equal lanes. Its fits also start from a different basis (the libstdc++ normal distribution), so its lane counts differ from the Mac's (weights 6 there against 4 on the Mac with warm probes).

## Where to pick up (2026-10-17)

- **Residual pre-fit** (`a15a542`, branch `development`): run `make test` and `make test-python` on it, then measure the full-size build again.
- **M1 runs** moved to spark: `~/neurocomputation_benchmarks/spikecore_M1/`.
  - 15 of 40 spike files are done (point A, point B seeds 1-5; Mac, Metal, `bc2da36`).
  - The other 25 run on spark (CUDA) from the worktree `~/spikecore_run_tree` (`a15a542`); the task is in `AGENT_TASK.md`, the progress in `RESULTS.md`.
  - Point B seeds 2-4 ran at about 19 min each on the Mac against 3.5 min for seed 1, with no job of ours on the GPU. 77% of the time was waiting on the kernel. Cause not found.
- **Decisions waiting for the user:**
  - the unclipped structured delays (above);
  - what to do about the random-spread conditions' delay planes (measure at 3% first).

## Where the other work stands (2026-10-17)

- **spikecorec** `development` at `a15a542` holds:
  - self-connections;
  - the numpy array constructor;
  - exact delta-synapse (weighted) arrivals, with tracked totals;
  - the CUDA fit kernels;
  - the warm-start lane search, with the residual pre-fit.

  Tests pass on Metal and CUDA up to `642dc7d`; `a15a542` is not yet tested.
- **Benchmark runs:** `~/Documents/neurocomputation/spikecore_runs/` (README, RESULTS_LOG, PRESENTATION_PLAN).
  - M2 is done and scored, with figures.
  - M1: point A is scored; the remaining runs are on spark (above).
  - M3 has its model and its cell checked against NEST, but no full runs: they are blocked on this construction time.
