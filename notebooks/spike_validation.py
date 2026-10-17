"""
Checks that a small small-world-torus network spikes under its drive.

The 25 GLIF1 cells and their drive come from examples/models/LEMS_glif1_torus.xml; the
connectivity is a small-world torus with an excitatory synapse on every edge. Every cell's
membrane potential is recorded to a .spire file and read back.
"""
import os
import tempfile

import numpy as np

import spikecorec as spc

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "models")

seed = 1042
side_length = 5  # the example sheet has 25 cells

network = spc.small_world_torus(side_length, random_fanout=4, seed=seed)
engine = spc.Spike_Engine(
    os.path.join(MODELS, "LEMS_glif1_torus.xml"),
    network,
    "excitatorySynapse",
    connection_weight=1.0,
    connection_delay_seconds=1e-3,
)

record_path = os.path.join(tempfile.gettempdir(), "spike_validation.spire")
engine.record_membrane_video(record_path)
engine.run()
engine.write_recordings()

recording = spc.read_spire_recording(record_path)
os.remove(record_path)

spike_counts = engine.spike_counts_per_neuron
num_spiked = int(np.count_nonzero(spike_counts))

print(f"neurons:             {engine.total_neuron_count}")
print(f"edges:               {engine.weights.total_edge_count}")
print(f"ticks simulated:     {engine.ticks_stepped}")
print(f"neurons that spiked: {num_spiked} / {engine.total_neuron_count}")
print(f"mean rate:           {engine.mean_firing_rate_hertz():.1f} Hz")
print(f"recording shape:     {recording.shape}")
print(f"peak |membrane|:     {np.abs(recording).max():.3f}")

engine.shutdown()

assert num_spiked > 0, "VALIDATION FAILED: no neuron spiked; the engine is not spiking"
print("\nVALIDATION PASSED: the spike engine spikes.")
