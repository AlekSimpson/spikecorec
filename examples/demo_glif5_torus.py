"""A 10x10 sheet of adapting GLIF5 cells on a torus, excitatory and inhibitory."""

import spikecorec as spc

from video_utils import *


def glif5_torus_demo():
    side_length = 10
    simulation_seconds = 0.5
    video_frame_stride = 5

    model = Path(__file__).resolve().parent / "models" / "LEMS_glif5_torus.xml"

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

    engine.record_membrane_video(demo_path("glif5_torus_membrane.spire"),
                                 video_frame_stride)
    engine.run()
    engine.write_recordings()
    engine.write_spike_file(demo_path("glif5_torus_spikes.dat"))

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

    # GLIF5 adapts: firing raises a cell's own threshold and injects a hyperpolarising
    # current, so the rate falls over the first few spikes. Comparing the first and last
    # interval of the busiest cell is the cheapest way to see that in a number.
    spike_times, spike_neurons = engine.spike_times
    busiest = max(range(engine.total_neuron_count), key=lambda index: spike_counts[index])
    own_times = [time for time, neuron in zip(spike_times, spike_neurons)
                 if neuron == busiest]
    if len(own_times) > 2:
        first_interval = own_times[1] - own_times[0]
        last_interval = own_times[-1] - own_times[-2]
        print("adaptation         : cell %d went from %.1f ms to %.1f ms between spikes "
              "(%.2fx)" % (busiest, 1000.0 * first_interval, 1000.0 * last_interval,
                           last_interval / first_interval))

    def first_active_cell(wanted_choice):
        return next(index for index in range(engine.total_neuron_count)
                    if choice[index] == wanted_choice and spike_counts[index] > 0)

    # One cell of each kind for the trace panel, picked from the cells that actually
    # fired so neither trace is a flat line.
    excitatory_cell = first_active_cell(0)
    inhibitory_cell = first_active_cell(1)
    print("tracing            : excitatory cell %d, inhibitory cell %d"
          % (excitatory_cell, inhibitory_cell))

    engine.shutdown()

    render_membrane_video(demo_path("glif5_torus_membrane.spire"),
                          spikes_path=demo_path("glif5_torus_spikes.dat"),
                          duration=simulation_seconds,
                          trace_neurons=[(excitatory_cell, "excitatory cell %d" % excitatory_cell),
                                         (inhibitory_cell, "inhibitory cell %d" % inhibitory_cell)],
                          title="GLIF5, 10x10 torus, adapting cells")


if __name__ == "__main__":
    glif5_torus_demo()
