"""Validates the GPU spike engine's run-to-run nondeterminism.

With learned (``U.V``) weights and near-threshold dynamics, ``step`` propagates
spikes via float ``atomic_fetch_add`` whose scheduler-dependent ordering is not
associative — so the *exact* spike pattern differs between runs even with the
seed, inputs, and parameters all held fixed (see
``notebooks/nondeterminism_demo.py``). These tests pin that behaviour down:

  * identical configs produce different spike patterns,
  * the divergence starts negligible and grows over ticks,
  * a *non-spiking* run (no atomic propagation) is perfectly reproducible —
    isolating the nondeterminism to the spiking/atomic path rather than general
    flakiness,
  * the effect is present across topologies.

All recordings use pytest's ``tmp_path`` — no ``.spire`` files persist.
"""
import numpy as np
import pytest

import spikecorec as spc
from helpers import (build_network, make_engine, run_static, normal_stimulus,
                     spread_per_tick)

# Enough repeats that "all identical" would be astronomically unlikely if the
# engine were truly nondeterministic, while keeping the suite fast.
N_RUNS = 16


def _run_validation_config(path, seed=1042):
    """The spike_validation.py setup: small-world torus, learned rank-1 weights,
    Gaussian drive — a regime that sits near threshold and reliably diverges."""
    net = spc.small_world_torus(4, seed=seed)
    engine = make_engine(net, 4, rank=1, resting_mp=0.1, decay_rate=0.22, learning_rate=0.00222)
    stimulus = normal_stimulus(10, 3, seed, scale=2.0)
    return run_static(engine, [1, 4, 12], stimulus, path)


def test_spike_pattern_varies_across_identical_runs(tmp_path):
    patterns, counts = set(), []
    for i in range(N_RUNS):
        last_spiked, _ = _run_validation_config(tmp_path / f"r{i}.spire")
        patterns.add(tuple(last_spiked.tolist()))
        counts.append(int((last_spiked != 0).sum()))

    assert min(counts) >= 1, "engine should always spike under this drive"
    assert len(patterns) >= 2, (
        f"expected nondeterministic spike patterns across identical runs, "
        f"but all {N_RUNS} runs were identical (counts={counts})")


def test_divergence_grows_over_ticks(tmp_path):
    recordings = [_run_validation_config(tmp_path / f"r{i}.spire")[1] for i in range(N_RUNS)]
    spread = spread_per_tick(recordings)

    # First recorded frame is identical up to float rounding; by the last frame
    # the runs have visibly diverged.
    assert spread[0] < 1e-3
    assert spread[-1] > 0.5
    assert spread[-1] > spread[0]


def test_nonspiking_run_is_reproducible(tmp_path):
    # No spikes => no atomic propagation => bit-identical across runs. This
    # controls for general flakiness and pins the nondeterminism to the atomics.
    recordings, last_spikes = [], []
    for i in range(6):
        net = spc.square_torus(4)
        engine = make_engine(net, 4, rank=1, learning_rate=0.0)   # threshold 1.0 > rest
        last_spiked, recording = run_static(
            engine, [0, 5, 10], np.zeros((8, 3)), tmp_path / f"r{i}.spire")
        recordings.append(recording)
        last_spikes.append(last_spiked)

    assert float(np.stack(recordings).std(axis=0).max()) == 0.0
    assert all(int((ls != 0).sum()) == 0 for ls in last_spikes)


@pytest.mark.parametrize("kind,k,inputs,seed", [
    ("small_world_torus", 5, [0, 12, 24], 7),
    ("random_fixed_outdegree", 4, [1, 7, 14], 3),
])
def test_nondeterminism_present_across_topologies(kind, k, inputs, seed, tmp_path):
    patterns = set()
    for i in range(N_RUNS):
        net = build_network(kind, k, seed=seed, fanout=6)
        engine = make_engine(net, k, rank=1, learning_rate=0.00222)   # learned weights
        stimulus = normal_stimulus(10, len(inputs), seed, scale=3.0)
        last_spiked, _ = run_static(engine, inputs, stimulus, tmp_path / f"r{i}.spire")
        patterns.add(tuple(last_spiked.tolist()))

    assert len(patterns) >= 2, f"expected nondeterminism for {kind}, got identical runs"
