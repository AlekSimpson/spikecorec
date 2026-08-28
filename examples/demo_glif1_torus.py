"""A 5x5 sheet of GLIF1 cells on a torus, half excitatory and half inhibitory."""

import spikecorec as spc

from video_utils import *


def glif1_torus_demo():
    side_length = 5
    simulation_seconds = 0.5
    video_frame_stride = 5

    model = Path(__file__).resolve().parent / "models" / "LEMS_glif1_torus.xml"

    # Every cell reaches its four neighbours, with the sheet's edges wrapped so no cell
    # sits on a boundary. The document declares no connections at all; this is the whole
    # of the network's wiring.
    topology = spc.square_torus(side_length)

    # Two synapses to draw from. The engine gives each cell one of them and every edge
    # leaving that cell carries it, so the draw is what decides which cells are
    # excitatory and which inhibitory. 4:1 is the usual cortical ratio.
    engine = spc.SpikeEngine(str(model), topology,
                             ["excitatorySynapse", "inhibitorySynapse"],
                             synapse_proportions=[0.8, 0.2],
                             connection_weight=1.0,
                             connection_delay_seconds=1e-3)

    engine.record_membrane_video(demo_path("glif1_torus_membrane.spire"),
                                 video_frame_stride)
    engine.run()
    engine.write_recordings()
    engine.write_spike_file(demo_path("glif1_torus_spikes.dat"))

    choice = engine.synapse_choice_per_neuron
    spike_counts = engine.spike_counts
    excitatory_count = sum(1 for value in choice if value == 0)
    inhibitory_count = sum(1 for value in choice if value == 1)

    print("neurons            : %d (%d excitatory, %d inhibitory)"
          % (engine.total_neuron_count, excitatory_count, inhibitory_count))
    print("edges              : %d" % engine.weights.total_edge_count)
    print("spikes             : %d" % int(sum(spike_counts)))
    print("mean firing rate   : %.2f Hz" % engine.mean_firing_rate_hertz())
    print("neurons that fired : %.0f %%"
          % (100.0 * engine.fraction_of_neurons_that_spiked()))

    # One cell of each kind for the trace panel, picked from the cells that actually
    # fired so neither trace is a flat line.
    def first_active_cell(wanted_choice):
        return next(index for index in range(engine.total_neuron_count)
                    if choice[index] == wanted_choice and spike_counts[index] > 0)

    excitatory_cell = first_active_cell(0)
    inhibitory_cell = first_active_cell(1)
    print("tracing            : excitatory cell %d, inhibitory cell %d"
          % (excitatory_cell, inhibitory_cell))

    engine.shutdown()

    render_membrane_video(demo_path("glif1_torus_membrane.spire"),
                          spikes_path=demo_path("glif1_torus_spikes.dat"),
                          duration=simulation_seconds,
                          trace_neurons=[(excitatory_cell, "excitatory cell %d" % excitatory_cell),
                                         (inhibitory_cell, "inhibitory cell %d" % inhibitory_cell)],
                          title="GLIF1, 5x5 torus, 4:1 excitatory/inhibitory")


if __name__ == "__main__":
    glif1_torus_demo()
