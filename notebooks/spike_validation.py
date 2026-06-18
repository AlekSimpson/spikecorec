"""
spike_validation.py — validates that the Metal spike engine actually spikes.

Builds a small small-world-torus reservoir, drives a few input neurons with
random stimulus for a handful of ticks via start_static_record, then reads back
per-neuron spike times (last_spiked) and the recorded membrane-potential trace
to confirm the network produced spikes (rather than sitting at its resting
potential the whole time).
"""
from spikecorec import SpikeEngine
import spikecorec as spc
import numpy as np
import os
import tempfile

seed = 1042
side_length = 4
lifetime = 10

network = spc.small_world_torus(side_length, seed=seed)
engine = SpikeEngine(
    network,
    [side_length, side_length],
    rank=1,
    resting_mp=0.1,
    decay_rate=0.22,
    learning_rate=0.00222,
)

inputs = [1, 4, 12]

rng = np.random.default_rng(seed)
random_inputs = rng.normal(size=(lifetime, 3))

engine.set_input_neurons(inputs)

# start_static_record drives the engine for `lifetime` ticks: it forces the
# input neurons into the active set each tick, applies that tick's stimulus row,
# steps the network, and writes one membrane-potential frame per tick.
record_path = os.path.join(tempfile.gettempdir(), "spike_validation.spire")
engine.start_static_record(
    random_inputs.tolist(),
    lifetime,
    record_path,
    compression=None,
)

last_spiked = engine.get_last_spiked()
recording = spc.read_spire_recording(record_path)
os.remove(record_path)

spiked_neurons = np.flatnonzero(last_spiked != 0)
num_spiked = int(spiked_neurons.size)

print(f"neurons:             {engine.neuron_count}")
print(f"input neurons:       {inputs}")
print(f"ticks simulated:     {lifetime}")
print(f"neurons that spiked: {num_spiked} / {engine.neuron_count}")
print(f"last_spiked tick:    {last_spiked.tolist()}")
print(f"recording shape:     {recording.shape}")
print(f"peak |membrane|:     {np.abs(recording).max():.3f}")

assert num_spiked > 0, "VALIDATION FAILED: no neuron spiked — engine is not spiking"
print("\nVALIDATION PASSED: the Metal spike engine spikes.")
