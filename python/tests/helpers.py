"""Shared helpers for the spikecorec Python test-suite.

All simulation output is written to caller-supplied paths (the tests pass
pytest's ``tmp_path``), so no ``.spire`` files are ever left behind in the repo.
"""
import numpy as np

import spikecorec as spc
from spikecorec import SpikeEngine

# SpikeEngine's default resting membrane potential used throughout the tests.
RESTING = 0.1

# The three network generators exposed by the bindings. Each produces a
# directed adjacency list over ``k * k`` neurons.
TOPOLOGIES = ["square_torus", "small_world_torus", "random_fixed_outdegree"]


def build_network(kind, k, seed=0, fanout=None):
    """Construct one of the bound topologies by name (see ``TOPOLOGIES``)."""
    if kind == "square_torus":
        return spc.square_torus(k)
    if kind == "small_world_torus":
        return spc.small_world_torus(k, random_fanout=4 if fanout is None else fanout, seed=seed)
    if kind == "random_fixed_outdegree":
        return spc.random_fixed_outdegree(k, fanout=8 if fanout is None else fanout, seed=seed)
    raise ValueError(f"unknown topology: {kind!r}")


def make_engine(network, k, *, rank=1, resting_mp=RESTING, decay_rate=0.22,
                learning_rate=0.00222, spike_threshold=None, spike_period=None,
                constant_weight=None):
    """Build a SpikeEngine over a ``k x k`` network with optional overrides.

    Passing ``constant_weight`` switches the engine onto the constant-weight
    propagation path (``use_constant_weight = True``); otherwise spikes propagate
    through the learned rank-``rank`` ``U.V`` weights.
    """
    engine = SpikeEngine(network, [k, k], rank=rank, resting_mp=resting_mp,
                         decay_rate=decay_rate, learning_rate=learning_rate)
    if spike_threshold is not None:
        engine.spike_threshold = spike_threshold
    if spike_period is not None:
        engine.spike_period = spike_period
    if constant_weight is not None:
        engine.use_constant_weight = True
        engine.weights.set_constant_weight(constant_weight)
    return engine


def run_static(engine, inputs, stimulus, path):
    """Drive ``engine`` with ``stimulus`` via ``start_static_record`` and read back.

    ``stimulus`` is a ``(lifetime, len(inputs))`` array. Returns
    ``(last_spiked, recording)`` where ``recording`` is ``(lifetime, neuron_count)``.
    Writes a raw (uncompressed) ``.spire`` file to ``path`` — pass a temp path.
    """
    stimulus = np.asarray(stimulus, dtype=np.float64)
    engine.set_input_neurons(list(inputs))
    engine.start_static_record(stimulus.tolist(), int(stimulus.shape[0]),
                               str(path), compression=None)
    return engine.get_last_spiked(), spc.read_spire_recording(str(path))


def constant_stimulus(lifetime, n_inputs, value):
    """A ``(lifetime, n_inputs)`` array holding ``value`` everywhere."""
    return np.full((lifetime, n_inputs), float(value))


def normal_stimulus(lifetime, n_inputs, seed, scale=1.0):
    """Reproducible Gaussian stimulus, scaled by ``scale``."""
    return np.random.default_rng(seed).normal(size=(lifetime, n_inputs)) * scale


def spread_per_tick(recordings):
    """Given stacked recordings ``(runs, ticks, neurons)`` return, per tick, the
    max over neurons of the across-run std-dev — i.e. how much the runs disagree."""
    return np.stack(recordings).std(axis=0).max(axis=1)
