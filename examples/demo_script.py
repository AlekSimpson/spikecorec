#!/usr/bin/env python3
"""The whole Python surface, in one run.

Ported from the pre-NeuroML version of this file, which built a torus out of an adjacency
list, designated input neurons by index and pushed a vector of drive into them each tick.
None of that exists any more: a model states its own cells, wiring, stimulus, dt, run length
and recordings, and the engine is handed the file.

Run it with any interpreter that has the extension installed:

    make python PYTHON=/path/to/python
    /path/to/python examples/demo_script.py
"""
import os
import tempfile

import spikecorec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RING_NETWORK_PATH = os.path.join(
    REPO_ROOT, "python", "tests", "fixtures", "glif3_ring_network.nml")


def write_model(directory, membrane_recording_path):
    """A LEMS model around the checked-in ring network, recording neuron 0's membrane.

    The network file is included by absolute path; includes resolve against the including
    file's own directory, and this wrapper is written into a scratch directory.
    """
    model_path = os.path.join(directory, "ring_model.xml")
    with open(model_path, "w") as model_file:
        model_file.write(
            "<Lems>\n"
            '  <Target component="demoSimulation"/>\n'
            '  <Include file="%s"/>\n'
            '  <Simulation id="demoSimulation" length="300ms" step="0.1ms"'
            ' target="Glif3RingNet">\n'
            '    <OutputFile id="membrane" fileName="%s">\n'
            '      <OutputColumn id="v0" quantity="RingPop[0]/v"/>\n'
            "    </OutputFile>\n"
            "  </Simulation>\n"
            "</Lems>\n" % (RING_NETWORK_PATH, membrane_recording_path)
        )
    return model_path


def main():
    spikecorec.set_log_level("warn")

    scratch_directory = tempfile.mkdtemp(prefix="spikecorec_demo_")
    membrane_recording_path = os.path.join(scratch_directory, "membrane.spire")
    model_path = write_model(scratch_directory, membrane_recording_path)

    engine = spikecorec.SpikeEngine(model_path)

    print("model      : %s" % model_path)
    print("neurons    : %d of cell type %s"
          % (engine.total_neuron_count, ", ".join(engine.cell_type_names())))
    print("dt         : %g s per tick, %d ticks (%g s)"
          % (engine.step_dt, engine.lifetime, engine.simulation_duration))
    print("stimulus   : %d wired input stream(s)" % engine.input_neuron_count)
    print("state vars : %s" % ", ".join(engine.state_variable_names(0)))

    print("\nwiring (source -> target, weight, delay in ticks):")
    for source_neuron in range(engine.total_neuron_count):
        for target_neuron in engine.weights.get_neighbors(source_neuron):
            print("  %d -> %d  weight=%g  delay=%d"
                  % (source_neuron, target_neuron,
                     engine.weights.get(source_neuron, target_neuron),
                     engine.weights.get_edge_delay_ticks(source_neuron, target_neuron)))

    # One tick at a time, so this tick's emissions can be counted. spike_flags() only ever
    # describes the most recent step, so a run() loop cannot see them.
    spike_counts = [0] * engine.total_neuron_count
    for tick in range(engine.lifetime):
        engine.step_simulation(tick)
        for neuron_index, flag in enumerate(engine.spike_flags()):
            spike_counts[neuron_index] += int(flag)

    print("\nspikes per neuron : %s" % spike_counts)
    print("membrane (mV)     : %s"
          % ["%.2f" % (value * 1e3) for value in engine.state_variable_values("v")])
    print("asc1 (pA)         : %s"
          % ["%.2f" % (value * 1e12) for value in engine.state_variable_values("asc1")])

    # Shutting the engine down is what finishes the recorders and flushes the last buffered
    # frames; reading the file before this would come up short.
    engine.shutdown()

    frames = spikecorec.read_spire_recording(membrane_recording_path)
    print("\nrecording         : %s" % membrane_recording_path)
    print("frames x columns  : %d x %d" % frames.shape)
    print("neuron 0 membrane : min %.2f mV, max %.2f mV"
          % (frames[:, 0].min() * 1e3, frames[:, 0].max() * 1e3))


if __name__ == "__main__":
    main()
