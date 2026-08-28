# ---
# Code cells for 01_tutorial_getting_started.ipynb, in jupytext "percent" format.
# `jupytext --to ipynb 01_tutorial_getting_started.py` produces the notebook.
#
# Audience: researchers and adopters. Every cell runs against the engine as built by
# `make python`; nothing here is pseudocode.
# ---

# %% [markdown]
# # Getting started with spikecorec
#
# spikecorec simulates spiking neural networks on the GPU, driven by NeuroML. You hand it
# a LEMS document describing cells, synapses, stimulus and a run; it generates a GPU
# kernel for those dynamics, compiles it, and runs the network.
#
# This notebook goes from the shortest complete run to driving the tick loop yourself.
# It assumes `make python` has been run.

# %%
import os
import textwrap

import numpy
import matplotlib.pyplot as pyplot

import spikecorec as spc

# The engine logs at info by default, which is noisy inside a notebook.
spc.set_log_level("warn")

MODELS = os.path.join("..", "models")

# %% [markdown]
# ## 1. The shortest complete run
#
# One cell, one current step, 500 ms. The document says everything: the cell's
# parameters, the stimulus, the timestep and the run length. `run()` executes every tick.

# %%
engine = spc.SpikeEngine(os.path.join(MODELS, "LEMS_single_cell.xml"))
engine.run()

spike_times, _ = engine.spike_times
print(f"{engine.total_neuron_count} neuron, {spike_times.size} spikes, "
      f"{engine.mean_firing_rate_hertz():.1f} Hz")
print(f"final membrane potential: {1000 * engine.read_state_variable(0, 'v'):.2f} mV")
engine.shutdown()

# %% [markdown]
# That is a GLIF1 cell at 2.5x rheobase. Its interspike interval is derivable by hand:
# the 5 ms refractory period plus `tau * ln(50/30)`, which is 10.1 ms, so almost exactly
# 100 Hz. The engine reports 50 spikes in 500 ms.
#
# ## 2. The model that just ran
#
# Nothing is hidden in Python. Here is the document, and it is the whole specification.

# %%
with open(os.path.join(MODELS, "single_cell.nml")) as model_file:
    print(model_file.read())

# %% [markdown]
# ### Changing a parameter and re-running
#
# Drop the drive from 500 pA to 250 pA, which is 1.25x rheobase rather than 2.5x, and the
# cell should fire much more slowly. Membrane potentials are volts in the engine, so
# multiply by 1000 to read millivolts.

# %%
def run_at_amplitude(amplitude_picoamperes, directory="/tmp/spikecorec_tutorial"):
    """Rewrites the drive amplitude, runs, and returns (spike count, rate)."""
    os.makedirs(directory, exist_ok=True)

    with open(os.path.join(MODELS, "single_cell.nml")) as model_file:
        model = model_file.read()
    model = model.replace('amplitude="500pA"', f'amplitude="{amplitude_picoamperes}pA"')
    model = model.replace('href="glif_cell_types.nml"',
                          f'href="{os.path.abspath(os.path.join(MODELS, "glif_cell_types.nml"))}"')

    with open(os.path.join(directory, "single_cell.nml"), "w") as model_file:
        model_file.write(model)
    with open(os.path.join(MODELS, "LEMS_single_cell.xml")) as lems_file:
        lems = lems_file.read()
    with open(os.path.join(directory, "LEMS_single_cell.xml"), "w") as lems_file:
        lems_file.write(lems)

    engine = spc.SpikeEngine(os.path.join(directory, "LEMS_single_cell.xml"))
    engine.run()
    times, _ = engine.spike_times
    rate = engine.mean_firing_rate_hertz()
    engine.shutdown()
    return times.size, rate


for amplitude in (250, 500, 1000):
    count, rate = run_at_amplitude(amplitude)
    print(f"{amplitude:5d} pA -> {count:3d} spikes, {rate:6.1f} Hz")

# %% [markdown]
# That is an f-I curve in three points: firing rate rises with drive, and below the
# 200 pA rheobase the cell would not fire at all.
#
# ## 3. Building a network in code
#
# Writing a `<connection>` element per edge does not scale: a million-neuron network
# would be a million lines of XML. So connectivity can come from Python instead, as an
# adjacency list with one row per neuron. The document then declares only the cells, the
# synapse and the stimulus.
#
# Three cells in a line, A to B to C:

# %%
engine = spc.SpikeEngine(os.path.join(MODELS, "LEMS_three_cell_chain.xml"),
                         [[1], [2], []],           # A reaches B, B reaches C
                         "chainSynapse",
                         connection_weight=1.0,
                         connection_delay_seconds=1e-3)
engine.run()

counts = engine.spike_counts
print(f"spikes per cell: A={counts[0]} B={counts[1]} C={counts[2]}")

times, neurons = engine.spike_times
first_of = {}
for time, neuron in zip(times, neurons):
    first_of.setdefault(neuron, time)
print("first spike: " + ", ".join(f"{name} at {1000 * first_of[index]:.1f} ms"
                                  for index, name in enumerate("ABC")))
engine.shutdown()

# %% [markdown]
# A to C takes about 13 ms, of which only 2 ms is the 1 ms wire delay on each edge. The
# rest is the synapse: an alpha current peaks a full `tau` after the spike that caused it,
# so the target needs several milliseconds to charge. The delay parameter sets when a
# spike **arrives**, not when it has an **effect**.
#
# ## 4. The same network written as NeuroML
#
# The adjacency list above is equivalent to this, which is what you would write if the
# connectivity belonged in the document. For three edges it is perfectly reasonable; for
# ten thousand it is not.

# %%
print(textwrap.dedent("""
    <network id="chainNetwork">
        <population id="chain" component="chainCell" size="3"/>

        <projection id="chainProjection" presynapticPopulation="chain"
                    postsynapticPopulation="chain" synapse="chainSynapse">
            <connectionWD id="0" preCellId="../chain[0]" postCellId="../chain[1]"
                          weight="1" delay="1 ms"/>
            <connectionWD id="1" preCellId="../chain[1]" postCellId="../chain[2]"
                          weight="1" delay="1 ms"/>
        </projection>
    </network>
"""))

# %% [markdown]
# ## 5. A balanced excitatory/inhibitory network
#
# Real cortex is roughly four excitatory cells to every inhibitory one. Hand the engine a
# list of synapses and it draws one per cell, so the draw is what makes a cell excitatory
# or inhibitory. Every edge leaving a cell carries that cell's synapse, which is Dale's
# law: a neuron is one thing or the other, not a mixture.
#
# The topology comes from a generator; `square_torus` wires each cell to its four
# neighbours with the edges wrapped.

# %%
engine = spc.SpikeEngine(os.path.join(MODELS, "LEMS_glif1_torus.xml"),
                         spc.square_torus(5),
                         ["excitatorySynapse", "inhibitorySynapse"],
                         synapse_proportions=[0.8, 0.2],
                         connection_weight=1.0,
                         connection_delay_seconds=1e-3)
engine.run()

choice = engine.synapse_choice_per_neuron
excitatory = [index for index, value in enumerate(choice) if value == 0]
inhibitory = [index for index, value in enumerate(choice) if value == 1]
print(f"{len(excitatory)} excitatory, {len(inhibitory)} inhibitory cells")
print(f"{engine.weights.total_edge_count} edges, "
      f"{engine.mean_firing_rate_hertz():.1f} Hz, "
      f"{100 * engine.fraction_of_neurons_that_spiked():.0f}% of cells fired")

times, neurons = engine.spike_times

figure, axes = pyplot.subplots(figsize=(9, 3.2))
axes.scatter([t for t, n in zip(times, neurons) if n in set(excitatory)],
             [n for n in neurons if n in set(excitatory)],
             s=6, label="excitatory")
axes.scatter([t for t, n in zip(times, neurons) if n in set(inhibitory)],
             [n for n in neurons if n in set(inhibitory)],
             s=6, label="inhibitory")
axes.set_xlabel("time (s)")
axes.set_ylabel("neuron")
axes.set_title("spike raster, 5x5 GLIF1 torus")
axes.legend(loc="upper right", fontsize=8)
pyplot.tight_layout()
pyplot.show()

engine.shutdown()

# %% [markdown]
# ## 6. Recording to .spire and reading it back
#
# `record_membrane_video` records every neuron's membrane potential every Nth tick to a
# `.spire` file. It has to be called **before** `run()`, because it is what makes the run
# record anything.
#
# The format is a 4-byte big-endian neuron count followed by native float32 frames, and
# `read_spire_recording` hands it back as a plain numpy array.

# %%
recording_path = "/tmp/spikecorec_tutorial/torus_membrane.spire"
os.makedirs(os.path.dirname(recording_path), exist_ok=True)

engine = spc.SpikeEngine(os.path.join(MODELS, "LEMS_glif1_torus.xml"),
                         spc.square_torus(5),
                         ["excitatorySynapse", "inhibitorySynapse"],
                         synapse_proportions=[0.8, 0.2],
                         connection_weight=1.0,
                         connection_delay_seconds=1e-3)
engine.record_membrane_video(recording_path, 5)   # every 5th tick
engine.run()
engine.write_recordings()                          # closes the recording
engine.shutdown()

frames = spc.read_spire_recording(recording_path)
print(f"frames x neurons: {frames.shape}, dtype {frames.dtype}")
print(f"membrane range  : {1000 * frames.min():.1f} to {1000 * frames.max():.1f} mV")

figure, axes = pyplot.subplots(figsize=(9, 3.2))
axes.plot(numpy.linspace(0, 0.5, frames.shape[0]), 1000 * frames[:, 0])
axes.set_xlabel("time (s)")
axes.set_ylabel("membrane potential (mV)")
axes.set_title("cell 0")
pyplot.tight_layout()
pyplot.show()

# %% [markdown]
# ## 7. Reading state and spikes back
#
# Three ways in, depending on what you want:
#
# - `read_state_variable(neuron, name)` for one cell's one variable
# - `state_variable_array(name)` for that variable across the whole population
# - `spike_counts` and `spike_times` for what fired and when

# %%
engine = spc.SpikeEngine(os.path.join(MODELS, "LEMS_glif1_torus.xml"),
                         spc.square_torus(5),
                         ["excitatorySynapse", "inhibitorySynapse"],
                         synapse_proportions=[0.8, 0.2],
                         connection_weight=1.0,
                         connection_delay_seconds=1e-3)
engine.run()

print(f"cell 0 v            : {1000 * engine.read_state_variable(0, 'v'):.2f} mV")

membrane = engine.state_variable_array("v")
print(f"population v        : {membrane.shape}, "
      f"mean {1000 * membrane.mean():.2f} mV")

counts = engine.spike_counts
busiest = int(numpy.argmax(counts))
print(f"busiest cell        : {busiest}, {counts[busiest]} spikes")

times, neurons = engine.spike_times
print(f"total spikes        : {times.size}")
engine.shutdown()

# %% [markdown]
# ## 8. Driving your own tick loop
#
# `run()` is a loop over `step_simulation`. Calling it yourself lets you read state or
# inject something between ticks. The tick index matters: the engine uses it for the
# spike-history ring and the refractory gate, so pass a monotonically increasing count.

# %%
engine = spc.SpikeEngine(os.path.join(MODELS, "LEMS_single_cell.xml"))

trace = []
for tick in range(engine.lifetime):
    engine.step_simulation(tick)
    if tick % 10 == 0:
        trace.append(engine.read_state_variable(0, "v"))

print(f"{engine.lifetime} ticks driven by hand, {len(trace)} samples kept")

figure, axes = pyplot.subplots(figsize=(9, 3.2))
axes.plot(numpy.linspace(0, 0.5, len(trace)), 1000 * numpy.array(trace))
axes.set_xlabel("time (s)")
axes.set_ylabel("membrane potential (mV)")
axes.set_title("driven one tick at a time")
pyplot.tight_layout()
pyplot.show()

engine.shutdown()

# %% [markdown]
# Reading state on every tick is slow: it copies from the GPU each time. Sampling every
# tenth tick, as above, costs almost nothing.
#
# ## 9. What is supported, and what gets refused
#
# The engine covers GLIF1 through GLIF5, `iafCell`, current-based synapses
# (`alphaCurrentSynapse` and friends), `pulseGenerator` and `spikeArray` stimulus, and
# arbitrary connectivity.
#
# What it does not cover, it refuses at construction rather than simulating incorrectly.
# Conductance-based synapses are the clearest example: their current depends on the
# target's own membrane potential, so the incoming edges of a cell cannot be collapsed
# into one accumulator the way current-based synapses can.

# %%
directory = "/tmp/spikecorec_tutorial"
os.makedirs(directory, exist_ok=True)

with open(os.path.join(directory, "refused.nml"), "w") as model_file:
    model_file.write(f"""<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="Refused">
    <include href="{os.path.abspath(os.path.join(MODELS, 'glif_cell_types.nml'))}"/>
    <GLIF1Cell id="cell" C="100pF" gL="10nS" EL="-70mV" vreset="-70mV"
               t_ref="5ms" vth="-50mV"/>
    <expTwoSynapse id="conductanceSynapse" gbase="1nS" erev="0mV"
                   tauRise="1ms" tauDecay="5ms"/>
    <pulseGenerator id="drive" delay="0ms" duration="100ms" amplitude="300pA"/>
    <network id="net">
        <population id="pop" component="cell" size="2"/>
        <explicitInput target="pop[0]" input="drive"/>
    </network>
</neuroml>
""")

with open(os.path.join(directory, "LEMS_refused.xml"), "w") as lems_file:
    lems_file.write("""<Lems>
    <Include file="Cells.xml"/>
    <Include file="Synapses.xml"/>
    <Include file="Inputs.xml"/>
    <Include file="Networks.xml"/>
    <Include file="Simulation.xml"/>
    <Include file="refused.nml"/>
    <Simulation id="sim1" length="100ms" step="0.1ms" target="net"/>
    <Target component="sim1"/>
</Lems>
""")

try:
    engine = spc.SpikeEngine(os.path.join(directory, "LEMS_refused.xml"),
                             [[1], []], "conductanceSynapse")
    print("constructed - this model is supported after all")
except RuntimeError as refusal:
    print("refused, as it should be:")
    print(f"  {refusal}")

# %% [markdown]
# The message names the offending ComponentType and says why. That is the contract: a
# model the engine cannot simulate correctly fails loudly at construction rather than
# producing a plausible-looking recording nobody can check.
