"""
The Python binding end to end: build a network, simulate it, inspect the compressed
weights.

Everything the engine needs comes from a LEMS document -- the cell dynamics, the
stimulus, the timestep and the run length -- while the connectivity is supplied from
Python by one of the topology generators. That split is the point of the two-argument
constructor: a 64-cell torus has 256 edges, and a document that spelled each one out as
a <connection> element would be longer than the model it describes.

Build the extension with `make python`, then:

    python3 examples/demo_script.py
"""

import os
import tempfile

import spikecorec

SIDE_LENGTH = 8
NEURON_COUNT = SIDE_LENGTH * SIDE_LENGTH
DRIVEN_NEURONS = [0, 9, 18, 27]


def write_model(directory):
    """Writes the NeuroML document and its LEMS wrapper, returning the LEMS path.

    GLIF1 with C = 100 pF and gL = 10 nS has a membrane time constant of 10 ms and a
    rheobase of gL * (vth - EL) = 200 pA, so the 260 pA drive on the seeded cells fires
    them while the 180 pA background leaves every other cell just under threshold, able
    to fire only when a spike arrives along an edge. Its 5 ms refractory period caps any
    cell at 200 Hz, which is what keeps the sheet a network rather than a saturated one.

    GLIF1-5 are not in the NeuroML standard library, so the document includes their
    ComponentTypes by absolute path the same way the C++ demos do.
    """
    cell_types_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir,
                     "tests", "fixtures", "nml", "glif_cell_types.nml"))
    model_path = os.path.join(directory, "torus_demo.nml")
    lems_path = os.path.join(directory, "LEMS_torus_demo.xml")

    # Seeded cells are driven over rheobase and fire on their own. Every other cell is
    # held just under it, so it can only fire when a spike arrives along an edge -- which
    # is what makes the spike count below a measurement of the network rather than of the
    # stimulus.
    explicit_inputs = "\n".join(
        '        <explicitInput target="torusPopulation[%d]" input="%s"/>'
        % (neuron, "seedDrive" if neuron in DRIVEN_NEURONS else "backgroundDrive")
        for neuron in range(NEURON_COUNT)
    )

    with open(model_path, "w") as model:
        model.write(
            '<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="TorusDemo">\n'
            '\n'
            '    <include href="%s"/>\n'
            '\n'
            '    <GLIF1Cell id="torusCell" C="100pF" gL="10nS" EL="-70mV"\n'
            '               vreset="-70mV" t_ref="5ms" vth="-50mV"/>\n'
            '\n'
            '    <alphaCurrentSynapse id="torusSynapse" tau="5 ms" ibase="40 pA"/>\n'
            '\n'
            '    <pulseGenerator id="seedDrive" delay="0 ms" duration="500 ms"\n'
            '                    amplitude="260 pA"/>\n'
            '    <pulseGenerator id="backgroundDrive" delay="0 ms" duration="500 ms"\n'
            '                    amplitude="180 pA"/>\n'
            '\n'
            '    <network id="torusNetwork">\n'
            '        <population id="torusPopulation" component="torusCell" size="%d"/>\n'
            '%s\n'
            '    </network>\n'
            '\n'
            '</neuroml>\n' % (cell_types_path, NEURON_COUNT, explicit_inputs)
        )

    with open(lems_path, "w") as lems:
        lems.write(
            '<Lems>\n'
            '    <Include file="Cells.xml"/>\n'
            '    <Include file="Synapses.xml"/>\n'
            '    <Include file="Inputs.xml"/>\n'
            '    <Include file="Networks.xml"/>\n'
            '    <Include file="Simulation.xml"/>\n'
            '    <Include file="torus_demo.nml"/>\n'
            '\n'
            '    <Simulation id="sim1" length="500ms" step="0.1ms" target="torusNetwork">\n'
            '    </Simulation>\n'
            '\n'
            '    <Target component="sim1"/>\n'
            '</Lems>\n'
        )

    return lems_path


def main():
    with tempfile.TemporaryDirectory() as directory:
        lems_path = write_model(directory)

        # Each cell reaches its four torus neighbours. The document declares no
        # connections at all -- these are the only ones the engine will have.
        adjacency = spikecorec.square_torus(SIDE_LENGTH)

        engine = spikecorec.SpikeEngine(lems_path, adjacency, "torusSynapse",
                                        connection_weight=1.0,
                                        connection_delay_seconds=1e-3)
        engine.run()

        spike_counts = engine.spike_counts
        print("neurons            : %d" % engine.total_neuron_count)
        print("edges              : %d" % engine.weights.total_edge_count)
        print("spikes             : %d" % sum(spike_counts))
        print("mean firing rate   : %.2f Hz" % engine.mean_firing_rate_hertz())
        print("neurons that fired : %.1f %%"
              % (100.0 * engine.fraction_of_neurons_that_spiked()))

        # The seeded cells fire from their own drive; anything else fired because a
        # spike travelled along an edge, which is what makes this a network run rather
        # than 64 independent cells.
        driven_spikes = sum(spike_counts[neuron] for neuron in DRIVEN_NEURONS)
        print("spikes from drive  : %d" % driven_spikes)
        print("spikes from network: %d" % (sum(spike_counts) - driven_spikes))

        # The weights are never stored one float per edge -- they are reconstructed from
        # the shared low-rank basis, so reading one back is a computation rather than a
        # lookup.
        statistics = engine.weights.weight_stats()
        print("\nweight matrix")
        print("  rank             : %d" % engine.weights.rank)
        print("  weight[0 -> %d]   : %.6f"
              % (adjacency[0][0], engine.weights.get(0, adjacency[0][0])))
        print("  mean / rms       : %.6f / %.6f"
              % (statistics.mean, statistics.root_mean_square))
        print("  worst fit error  : %.3e" % engine.weights.measured_weight_fit_error)
        print("  corrections used : %.1f %% of capacity"
              % (100.0 * engine.weights.sparse_delta_occupancy_fraction))

        engine.shutdown()


if __name__ == "__main__":
    main()
