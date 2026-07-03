"""Thorough validation that the Metal spike engine produces normal, expected
spiking activity — across topologies, network sizes, input sets, and engine
parameters.

This is the unit-test counterpart of ``notebooks/spike_validation.py``: rather
than a single check, it sweeps many configurations to ensure spiking is a robust
property of the engine and not a fluke of one setup. Because GPU propagation is
nondeterministic (see ``test_nondeterminism.py``), assertions check *invariant
properties* (spikes happen / don't happen, shapes, finiteness, bounds, decay)
rather than exact spike patterns.

All recordings go to pytest's ``tmp_path`` and are auto-cleaned — no ``.spire``
files are left in the repo.
"""
import numpy as np
import pytest

import spikecorec as spc
from helpers import (RESTING, TOPOLOGIES, build_network, make_engine, run_static,
                     constant_stimulus, normal_stimulus)


# ── strongly-driven input neurons must spike, on every topology and size ──────
@pytest.mark.parametrize("kind", TOPOLOGIES)
@pytest.mark.parametrize("k", [3, 4, 6])
def test_strong_input_produces_spikes(kind, k, tmp_path):
    n = k * k
    net = build_network(kind, k, seed=k)
    engine = make_engine(net, k, rank=1, decay_rate=0.25, learning_rate=0.0)
    inputs = sorted({0, n // 3, n // 2, n - 1})
    last_spiked, recording = run_static(
        engine, inputs, constant_stimulus(8, len(inputs), 6.0), tmp_path / "r.spire")

    assert recording.shape == (8, n)
    assert np.isfinite(recording).all()
    assert int((last_spiked != 0).sum()) >= 1            # something spiked
    assert any(last_spiked[i] != 0 for i in inputs)      # the driven neurons spiked


# ── a spike must propagate past the input neurons through the topology ─────────
@pytest.mark.parametrize("kind", TOPOLOGIES)
def test_spike_propagation_beyond_inputs(kind, tmp_path):
    k = 5
    n = k * k
    net = build_network(kind, k, seed=7, fanout=6)
    # Strong constant synaptic weight guarantees a spiking neuron pushes its
    # neighbours over threshold, so activity must spread beyond the single input.
    engine = make_engine(net, k, rank=1, learning_rate=0.0, constant_weight=3.0)
    inputs = [n // 2]
    last_spiked, _ = run_static(
        engine, inputs, constant_stimulus(10, 1, 6.0), tmp_path / "r.spire")

    spiked = set(np.flatnonzero(last_spiked != 0).tolist())
    assert len(spiked) > len(inputs)
    assert spiked - set(inputs)                          # non-input neurons spiked


# ── the spike_validation.py scenario, repeated across several seeds ────────────
@pytest.mark.parametrize("seed", [0, 1, 1042, 2024])
def test_random_normal_stimulus_like_validation(seed, tmp_path):
    k = 4
    n = k * k
    net = spc.small_world_torus(k, seed=seed)
    engine = make_engine(net, k, rank=1, resting_mp=0.1, decay_rate=0.22, learning_rate=0.00222)
    stimulus = normal_stimulus(10, 3, seed, scale=3.0)
    last_spiked, recording = run_static(engine, [1, 4, 12], stimulus, tmp_path / "r.spire")

    assert recording.shape == (10, n)
    assert np.isfinite(recording).all()
    assert int((last_spiked != 0).sum()) >= 1


# ── spiking is robust to the rank of the learned weight factorisation ──────────
@pytest.mark.parametrize("rank", [1, 4, 16, 64])
def test_rank_variation_spikes(rank, tmp_path):
    k = 4
    n = k * k
    net = spc.small_world_torus(k, seed=3)
    engine = make_engine(net, k, rank=rank, learning_rate=0.0, constant_weight=2.0)
    last_spiked, recording = run_static(
        engine, [0, 8], constant_stimulus(8, 2, 6.0), tmp_path / "r.spire")

    assert recording.shape == (8, n)
    assert int((last_spiked != 0).sum()) >= 1


# ── negative control: no stimulus → no spikes, potentials stay at rest ─────────
@pytest.mark.parametrize("kind", TOPOLOGIES)
def test_no_input_no_spikes(kind, tmp_path):
    k = 4
    net = build_network(kind, k, seed=1)
    engine = make_engine(net, k, rank=1, learning_rate=0.0)   # threshold 1.0 > rest 0.1
    last_spiked, recording = run_static(
        engine, [0, 5, 10], np.zeros((8, 3)), tmp_path / "r.spire")

    assert int((last_spiked != 0).sum()) == 0
    assert np.allclose(recording, RESTING, atol=1e-5)


# ── the threshold gates spiking: low fires, very high suppresses ──────────────
@pytest.mark.parametrize("threshold,expect_spikes", [(0.5, True), (50.0, False)])
def test_threshold_controls_spiking(threshold, expect_spikes, tmp_path):
    k = 4
    net = spc.square_torus(k)
    engine = make_engine(net, k, rank=1, learning_rate=0.0, spike_threshold=threshold)
    last_spiked, _ = run_static(
        engine, [5], constant_stimulus(8, 1, 2.0), tmp_path / "r.spire")

    assert (int((last_spiked != 0).sum()) > 0) is expect_spikes


# ── last_spiked entries are valid tick indices within the simulated window ─────
def test_spike_times_within_bounds(tmp_path):
    k, lifetime = 5, 12
    net = spc.small_world_torus(k, seed=9)
    engine = make_engine(net, k, rank=1, learning_rate=0.0, constant_weight=2.0)
    last_spiked, _ = run_static(
        engine, [0, 12, 24], constant_stimulus(lifetime, 3, 6.0), tmp_path / "r.spire")

    assert int(last_spiked.min()) >= 0
    assert int(last_spiked.max()) < lifetime
    assert int((last_spiked != 0).sum()) >= 1


# ── the recorded membrane trace must actually move away from rest ──────────────
@pytest.mark.parametrize("kind", TOPOLOGIES)
def test_recording_deviates_from_rest(kind, tmp_path):
    k = 4
    net = build_network(kind, k, seed=2)
    engine = make_engine(net, k, rank=1, learning_rate=0.0, constant_weight=2.0)
    _, recording = run_static(
        engine, [3, 9], constant_stimulus(8, 2, 6.0), tmp_path / "r.spire")

    assert float(np.abs(recording - RESTING).max()) > 0.1


# ── record_stride subsamples frames; lifetime sets the count ──────────────────
def test_record_stride_controls_frame_count(tmp_path):
    k = 4
    n = k * k
    net = spc.square_torus(k)
    engine = make_engine(net, k, rank=1, learning_rate=0.0, constant_weight=2.0)
    engine.set_input_neurons([0, 5, 10])
    path = str(tmp_path / "r.spire")
    engine.start_static_record(constant_stimulus(12, 3, 6.0).tolist(), 12, path,
                               record_stride=3, compression=None)
    recording = spc.read_spire_recording(path)

    assert recording.shape == (4, n)        # ticks 0,3,6,9 recorded


# ── pure sub-threshold drive then silence: membrane decays back toward rest ────
def test_pure_decay_returns_toward_rest():
    k = 4
    net = spc.square_torus(k)
    engine = make_engine(net, k, rank=1, learning_rate=0.0, spike_threshold=100.0)
    engine.set_input_neurons([5])

    engine.step_simulation([3.0], 0, [5], False)
    after_drive = float(engine.get_membrane_potentials()[5])
    for tick in range(1, 8):
        engine.step_simulation([0.0], tick, [5], False)
    after_quiet = float(engine.get_membrane_potentials()[5])

    assert after_drive > RESTING + 0.5          # stimulus raised it
    assert after_quiet < after_drive            # then it decayed
    assert abs(after_quiet - RESTING) < 0.2     # back near rest


# ── bifurcation scaling conditions the reservoir, which then spikes ───────────
def test_bifurcation_scaling_produces_activity(tmp_path):
    k = 4
    net = spc.small_world_torus(k, seed=4)
    engine = make_engine(net, k, rank=4, decay_rate=0.22, learning_rate=0.00222)
    target, w_accum, w_instant = engine.scale_uniform_weights_near_bifurcation(1, 5.0, True)

    assert engine.use_constant_weight is True
    assert w_accum > 0 and abs(target) > 0
    last_spiked, _ = run_static(
        engine, [5], constant_stimulus(8, 1, 3.0), tmp_path / "r.spire")
    assert int((last_spiked != 0).sum()) >= 1


# ── the learned U·V weight path (use_constant_weight stays False) also spikes ──
def test_learned_weight_path_spikes(tmp_path):
    k = 5
    n = k * k
    net = spc.small_world_torus(k, seed=11)
    engine = make_engine(net, k, rank=8, learning_rate=0.001)

    assert engine.use_constant_weight is False
    last_spiked, recording = run_static(
        engine, [0, 12, 24], constant_stimulus(10, 3, 8.0), tmp_path / "r.spire")
    assert int((last_spiked != 0).sum()) >= 1
    assert np.isfinite(recording).all()


# ── varying resting potential / decay / spike_period still yields activity ─────
@pytest.mark.parametrize("resting_mp,decay_rate,spike_period", [
    (0.0, 0.1, 1),
    (0.2, 0.5, 2),
    (-0.1, 0.05, 3),
])
def test_membrane_parameter_variations_spike(resting_mp, decay_rate, spike_period, tmp_path):
    k = 4
    net = spc.small_world_torus(k, seed=6)
    engine = make_engine(net, k, rank=1, resting_mp=resting_mp, decay_rate=decay_rate,
                         learning_rate=0.0, spike_period=spike_period, constant_weight=2.0)
    last_spiked, recording = run_static(
        engine, [2, 9], constant_stimulus(10, 2, 7.0), tmp_path / "r.spire")

    assert np.isfinite(recording).all()
    assert int((last_spiked != 0).sum()) >= 1
