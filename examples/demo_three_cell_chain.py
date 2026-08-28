"""Three excitatory cells in a line, A to B to C, with a pulse train driving A."""

import spikecorec as spc

from video_utils import *


def three_cell_chain_demo():
    simulation_seconds = 0.5
    video_frame_stride = 5

    model = Path(__file__).resolve().parent / "models" / "LEMS_three_cell_chain.xml"

    # A reaches B, B reaches C, C reaches nothing. Written out rather than generated:
    # at three cells the adjacency list is the clearest statement of the network there
    # is.
    topology = [[1], [2], []]

    engine = spc.SpikeEngine(str(model), topology, "chainSynapse",
                             connection_weight=1.0,
                             connection_delay_seconds=1e-3)

    engine.record_membrane_video(demo_path("three_cell_chain_membrane.spire"),
                                 video_frame_stride)
    engine.run()
    engine.write_recordings()
    engine.write_spike_file(demo_path("three_cell_chain_spikes.dat"))

    spike_counts = engine.spike_counts
    spike_times, spike_neurons = engine.spike_times

    print("neurons            : %d" % engine.total_neuron_count)
    print("edges              : %d" % engine.weights.total_edge_count)
    print("spikes per cell    : A=%d  B=%d  C=%d"
          % (spike_counts[0], spike_counts[1], spike_counts[2]))
    print("mean firing rate   : %.2f Hz" % engine.mean_firing_rate_hertz())

    # The chain conducts if every pulse into A produces a spike in each cell in turn.
    # The lag from A to C is what the 1 ms conduction delay per edge buys, and printing
    # it is how the delay stops being a number in a document and starts being something
    # the run demonstrates.
    first_of = {}
    for time, neuron in zip(spike_times, spike_neurons):
        if neuron not in first_of:
            first_of[neuron] = time
    if len(first_of) == 3:
        print("first spike        : A at %.1f ms, B at %.1f ms, C at %.1f ms"
              % (1000.0 * first_of[0], 1000.0 * first_of[1], 1000.0 * first_of[2]))
        print("A to C conduction  : %.1f ms" % (1000.0 * (first_of[2] - first_of[0])))
    else:
        print("the chain did not conduct: only cells %s ever fired"
              % sorted(first_of.keys()))

    engine.shutdown()

    render_membrane_video(demo_path("three_cell_chain_membrane.spire"),
                          spikes_path=demo_path("three_cell_chain_spikes.dat"),
                          duration=simulation_seconds,
                          # B and C rather than A: A is driven by a spikeArray, which
                          # injects for a single tick, so its whole excursion happens
                          # between two recorded frames and its trace is a flat line at
                          # rest. Its spikes are still in the raster and in the marker row
                          # above the traces. B and C are driven synaptically and show the
                          # alpha current charging them, which is the thing worth seeing.
                          trace_neurons=[(1, "cell B, driven by A"),
                                         (2, "cell C, end of chain")],
                          title="Three-cell chain, A to B to C")


if __name__ == "__main__":
    three_cell_chain_demo()
