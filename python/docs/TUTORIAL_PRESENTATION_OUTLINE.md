# spikecorec: an introductory tutorial

Slide outline for a lab presentation to computational neuroscience researchers who have not
used spikecorec. It moves from the big picture to concepts, then what the engine
supports, then the Python API from first run to advanced use, and ends with internals.
Each slide only uses ideas introduced on earlier ones.

---

## Part 1: The big picture

### Slide 1: Title
- spikecorec: GPU spiking-network simulation from NeuroML
- Tutorial for the lab: what it is, why it exists, and how to use it from Python
- Presenter, date, link to the repository

### Slide 2: Agenda
- What spikecorec is, and why we built it
- The few concepts you need: NeuroML/LEMS, ticks, how a run works
- What the engine can simulate today, and what it refuses
- The Python API, from a one-line run to inspecting the model and connectivity
- Under the hood: how connectivity is compressed, and how accurate it is
- Limitations, roadmap, resources

### Slide 3: What is spikecorec?
- A simulation engine for spiking neural networks that runs on the GPU
- Input: a standard NeuroML/LEMS model description
- Output: spikes, membrane traces and recordings, in Python as numpy arrays
- It generates GPU code for your model's dynamics; nothing is hand-written per model
- Used from Python (`import spikecorec`), built in C++ underneath

### Slide 4: Why build it?
- **Scale:** connectivity is the memory bottleneck. A dense weight matrix for 1 million
  neurons is 10¹² entries, about 4 TB of float32
- **Standard models:** describe a model once in NeuroML and simulate it; no simulator-specific
  model code
- **Any dynamics, no kernel writing:** cell and synapse equations come from the model's LEMS
  definitions, and the GPU code is generated from them
- **Speed:** the whole network advances in parallel on the GPU, every timestep
- **Accessible:** a small Python API for researchers who don't write GPU code
- Speaker prompt: where this fits next to the tools the lab already uses

### Slide 5: The core ideas in one slide
- **NeuroML in:** the model file defines cells, synapses, network, stimulus and run length
- **Code generation:** the engine turns the model's equations into one GPU program, compiled
  when the engine is built
- **Clock-driven:** the network advances in fixed timesteps ("ticks")
- **Compressed connectivity:** which neurons connect, and every per-connection value, are
  stored compressed rather than as a dense matrix
- **Python on top:** build, run and read results from a notebook

---

## Part 2: Concepts you need

### Slide 6: NeuroML and LEMS in two minutes
- **NeuroML (.nml):** the model itself: cells, synapses, populations, projections and inputs
- **LEMS (.xml):** the simulation settings: timestep, duration, which network, what to record
- **ComponentTypes:** each cell or synapse type defines its own parameters and dynamics
  (state variables, time derivatives, threshold conditions, spike events)
- **Standard library:** the NeuroML core ComponentTypes are included, so a model only
  declares its instances; your own custom ComponentTypes work the same way
- Show: `examples/models/single_cell.nml` and `LEMS_single_cell.xml`

### Slide 7: How a run works: ticks
- Time advances in steps of `dt`, which the LEMS file sets (e.g. 0.05 ms)
- Each tick, for every neuron in parallel:
  - apply stimulus
  - integrate the cell's equations one step (forward Euler)
  - detect threshold crossings, emit spikes, reset
  - deliver spikes along outgoing connections, after each connection's delay
  - update synapse state and record
- Events (thresholds, spike arrivals) are checks made every tick, not a separate event queue
- A spike reaches its target at least one tick after it fires

### Slide 8: What happens when you create an engine
- Parse the LEMS file, the NeuroML files it includes, and the standard library types they use
- Allocate GPU memory for every cell's state
- Build the compressed connectivity and fit every per-connection value
- Generate the GPU program for this model and compile it
- Construction is the expensive step; ticks after it are fast
- Unsupported models fail here, with a message naming the component and the reason

---

## Part 3: What it simulates today

### Slide 9: Kinds of SNN models supported today
- **Networks:** networks of point neurons (single-compartment), clock-driven
  - several cell types in one network, one per population
  - excitatory and inhibitory populations, with several synapse types in one network
  - any sparse connectivity, recurrent or feedforward: grids, small-world, random or
    hand-specified, with a weight and delay on every connection
  - driven by constant-current pulses (`pulseGenerator`), spike lists (`spikeArray`) and
    populations of random spike sources (for example `spikeGeneratorPoisson`)
- **Neurons:**
  - leaky integrate-and-fire with threshold and reset
  - with timed refractory periods
  - with adaptation: after-spike currents and threshold adaptation
  - nonlinear integrate-and-fire with a recovery variable, reset when the voltage peaks
  - spike sources with random intervals, such as Poisson generators
  - custom point-neuron models written in LEMS from the same building blocks
- **Synapses:**
  - current-based, with their own LEMS kinetics (for example exponential or alpha-shaped
    current), and separate state on every connection
- The next slides give the exact constraints a model has to fit

### Slide 10: Cell models
- Any point-neuron ComponentType, standard or custom, whose LEMS `<Dynamics>` is built from:
  - state variables with starting values (`OnStart`)
  - derived variables, including conditional ones
  - time derivatives, integrated with forward Euler every tick
  - threshold conditions (`OnCondition`) that assign state and emit spikes
  - spike handlers (`OnEvent`) that run once for each spike reaching the cell
  - optionally a refractory period: an integrating regime and a refractory regime that
    transitions back to it
  - `random()`, a fresh draw for each neuron on every call, seeded by the simulation's seed
- Every name in the equations has to be a state variable, parameter, constant or derived
  variable
- Each population can use a different cell type; one GPU program covers all of them

### Slide 11: Synapses and connections
- Any current-based synapse defined in LEMS: its equations can use its own state, its
  parameters and the connection's weight, but not the target cell's variables
- A synapse can have its own state and time derivatives, and react to each arriving spike
  (`OnEvent`)
- Every connection has its own synapse state: two connections of the same type evolve
  independently
- Per-connection weight and delay; delays in whole ticks, at least one tick
- Several synapse types in one network (e.g. excitatory and inhibitory)
- Connectivity from the NeuroML document, or from Python as an adjacency list

### Slide 12: Stimulus and recording
- Stimulus that behaves as NeuroML defines it, wired with `explicitInput` or `inputList`:
  - `pulseGenerator` / `pulseGeneratorDL`: a constant input over a time window
  - `spikeArray`: spikes at listed times, delivered as kicks into the target's input
- Accepted but not correct yet (being fixed):
  - `sineGenerator` runs as a constant pulse at its amplitude
  - `timedSynapticInput` delivers kicks instead of driving its synapse
  - ramp, compound, Poisson-synapse and voltage-clamp inputs deliver nothing
- Recording: the document's `OutputFile` (variable traces) and `EventOutputFile` (spikes)
- Plus engine-side recordings: every spike, and every neuron's membrane potential as a
  `.spire` file (optionally compressed) for video rendering

### Slide 13: What it refuses, and why that is good
- Synapses whose current depends on the target cell, such as conductance-based synapses,
  which read its voltage
- `random()` inside a synapse
- Regime structures other than the integrating/refractory pair
- Ion channels, concentrations, multicompartment morphologies and kinetic schemes
- A synapse or input that does not provide what its target cell reads, for example a current
  into a cell whose input is dimensionless
- These are refused when the engine is created, with the component and the reason, so
  they never run silently wrong. The exception today is the input types on slide 12
- Plasticity: the flag exists, but nothing changes weights during a run yet
- Platform: macOS with Metal runs today. CUDA code is generated, but the launcher that
  runs it is not built yet

---

## Part 4: The Python API, step by step

### Slide 14: Installing
- `make python` builds the extension into a uv-managed `.venv` (macOS, Metal)
- Needs uv and libxml2; `make python` installs pybind11, numpy and matplotlib into `.venv`;
  ffmpeg for videos
- `import spikecorec as spc`
- Docs: `python/docs/README.md` (guide) and `python/docs/API_REFERENCE.md` (every name)

### Slide 15: Your first simulation
- `engine = spc.SpikeEngine("examples/models/LEMS_single_cell.xml")`
- `engine.run()` runs every tick the document declares
- `engine.mean_firing_rate_hertz()`, `engine.read_state_variable(0, "v")` (volts)
- `engine.shutdown()` frees the GPU memory
- Result: a GLIF1 cell at 2.5× rheobase fires 50 spikes in 500 ms (100 Hz), matching the
  analytic interspike interval

### Slide 16: Reading results
- `engine.spike_counts_per_neuron`: spikes per neuron
- `engine.recorded_spikes`: `(times_seconds, neuron_indices)` as two numpy arrays
- `engine.state_variable_array("v")`: one variable for every neuron
- `engine.fraction_of_neurons_that_spiked()`
- `engine.recorded_traces` and `recorded_trace_times`: the document's recorded columns
- Everything comes back as numpy copies, safe to keep after the engine is gone

### Slide 17: Experimenting with a model
- Change a parameter in the document and rebuild the engine
- Example from the tutorial: an f–I curve. 250, 500 and 1000 pA give 48, 100 and 140 Hz
- The document is the single source of truth: no hidden Python-side model settings

### Slide 18: Networks from Python
- Writing one XML element per connection does not scale; pass an adjacency list instead
- `SpikeEngine(lems_file, adjacency, "chainSynapse", connection_weight=1.0, connection_delay_seconds=1e-3)`
- `adjacency[source]` lists that neuron's targets; the document only declares cells, synapses
  and stimulus
- Three-cell chain A→B→C: first spikes at 20.0, 26.6 and 33.2 ms
- Discussion point: the synapse's rise time, not the 1 ms wire delay, dominates the lag

### Slide 19: Topology generators
- `spc.square_torus(side)`: a grid with 4 neighbours each, edges wrapped
- `spc.small_world_torus(side, random_fanout, seed)`: the torus plus long-range shortcuts
- `spc.random_fixed_outdegree(side, fanout, seed)`: a random graph with a fixed out-degree
- Each returns a plain list of lists, so you can build your own too

### Slide 20: Excitatory and inhibitory populations
- Pass several synapses with a share for each:
  `["excitatorySynapse", "inhibitorySynapse"], synapse_proportions=[0.8, 0.2]`
- Connections are split in order into one contiguous share per synapse; nothing is random
- Connections are ordered by source neuron, so roughly the first 80% of neurons are
  excitatory; a neuron at the boundary can carry both
- `engine.weights.get_edge_synapse_prototype(source, target)` says which synapse any
  connection carries
- Example: a 5×5 GLIF1 torus, 20 excitatory and 5 inhibitory cells, 55.9 Hz

### Slide 21: Driving the clock yourself
- `run()` is a loop over `engine.step_simulation(tick)`
- Step manually to sample state during a run, or stop early
- Pass increasing tick numbers: the engine uses them for delays and refractory periods
- Reading state copies from the GPU, so sample every Nth tick rather than every tick

### Slide 22: Recording and visualization
- `engine.record_membrane_video("out.spire", frame_stride)` must come **before** `run()`
- `engine.write_recordings()` closes it and writes the document's output files
- `engine.write_spike_file("spikes.dat")` writes `time<TAB>neuron` lines
- `spc.read_spire_recording(path)` returns a `(frames, neurons)` array
- `examples/video_utils.py`: `render_membrane_video(...)` makes an mp4 with traces
- For custom loops: `spc.SimulationRecorder`, `SpireWriter` and `SpireReader`

### Slide 23: Inspecting the model: `spikecorec.nml`
- `engine.context` is the parsed model the engine was built from
- `spc.nml.NML_Context().parse(path)` inspects a model with no GPU at all
- Populations and their cells: `nml.network_populations`, `nml.population_cell`
- Types: `component_types[...]` with `state_variable_names` and `dynamics` (the LEMS
  equations, as written)
- Values in SI units: `nml.starting_values`, `nml.component_parameter_values`, `resolve_quantity`
- Connectivity and inputs: `simulation.network_data.edge_arrays()`, `simulation.input_profiles`
- `spikecorec.units`: `parse_quantity("-60mV")`, `seconds_to_ticks(...)`

### Slide 24: Looking at raw engine state
- Copies of the GPU buffers: `engine.cell_state`, `last_spiked`, `network_inputs`,
  `spike_history`
- `context.resolve_path("pop[0]/v")` gives a variable's position in `cell_state`
- `engine.master_kernel_source`: the exact GPU program generated for your model
- Useful for debugging a model, and for seeing what "code generation" means

---

## Part 5: Under the hood

### Slide 25: Connectivity storage, part 1: which neurons connect
- Most pairs of neurons are not connected, so a dense matrix wastes almost all its memory
- spikecorec stores which pairs connect in a **k²-tree**: a compressed bitmap that skips
  empty regions, and that the GPU walks directly
- Reachable from Python: `engine.weights.k2tree` (`adjacent`, `get_neighbors`, ...)

### Slide 26: Connectivity storage, part 2: every value on every connection
- Each connection carries values: weight, delay and each synapse state variable. Each kind
  of value is one **plane**
- No plane is stored as one number per connection. Each is a low-rank factorization:
  plane k = `U · diag(Ck) · Vᵀ`, with `U` and `V` shared by all planes and one small
  coefficient row `Ck` per plane
- A read is a short computation, not a lookup

### Slide 27: Reading connection values from Python
- `w = engine.weights`
- `w.get(source, target)`: one connection's weight
- `w.neighbor_weights_for_matrix(plane)`: one plane for every connection
- Planes: `w.WEIGHT_PLANE`, `w.DELAY_PLANE`, then the synapse's state variables
- `engine.synapse_state_variable_plane("excitatorySynapse", "I")` finds a variable's plane
- Basis inspection: `w.rank`, `w.U_matrix`, `w.V_matrix`, `w.coefficients`

### Slide 28: How accurate is the compression?
- The fit is made to a stated tolerance: the worst relative error allowed on any value
- Presets on `SpikeEngine`: `FIT_TOLERANCE_PRECISE` (1e-5), `ACCURATE` (1e-4), `STANDARD`
  (1e-3, the default), `COMPACT` (1e-2)
- Tighter means more memory and longer fitting
- Delays stay on their exact tick while tolerance × delay is under half a tick: up to 500
  ticks at the default
- Each plane gets the fewest lanes that meet the tolerance, up to its largest degree + 4, where
  one exact solve always meets it; `w.measured_fit_error` reports what each plane actually got
- Networks built from a few uniform projections are stored exactly

### Slide 29: Values that change during a run
- Synapse state changes every tick; the changes go into a sparse update buffer **S**, which
  every read includes
- When S is full enough, the engine **refits**: it folds S into the factorization and empties S
- Settings:
  - `refit_occupancy_threshold_fraction`: 0.2 by default; also sets how large S is kept
  - `minimum_ticks_between_refits`: 1000 by default
  - `refit_every_n_ticks`: off by default
  - `fit_tolerance`
- Observable: `w.sparse_delta_entry_count`, `w.sparse_delta_occupancy_fraction()`,
  `engine.last_refit_tick`

### Slide 30: Built-in efficiency
- Synapses only run while they can be active: from a presynaptic spike until the synapse has
  settled back to rest (`engine.synapse_active_ticks`); quiet parts of the network cost little
- Everything fixed for a run (neuron count, the connectivity tree's shape, the spike-history
  length, the synapse activity window) is compiled into the GPU program as constants
- Delays use a ring of recent spikes instead of a queue of events in flight
- Practical tips:
  - sample state rather than reading it every tick
  - choose the loosest tolerance your analysis allows
  - leave the refit defaults unless refits show up in timing

---

## Part 6: Wrapping up

### Slide 31: Current limitations
- macOS/Metal only until the CUDA launcher lands
- Model constraints: point neurons, current-based synapses, at most an
  integrating/refractory regime pair
- No synaptic plasticity during a run yet
- Stimulus: only pulse generators and spike arrays behave as NeuroML defines; sine, ramp,
  compound, timed-synaptic, Poisson-synapse and voltage-clamp inputs are being fixed
- Fitting large networks at construction can take minutes; tuning is ongoing

### Slide 32: Roadmap
- Next: richer point-neuron dynamics (more regime shapes, events into cells), synapses that
  read the target cell (conductance-based), on-device stimulus generators, heterogeneous
  populations, plasticity
- Later: biophysical models: ion channels, concentrations, multicompartment morphologies,
  kinetic schemes, stiff integrators
- CUDA backend for Linux and NVIDIA GPUs
- Faster fitting and refitting for large, busy networks

### Slide 33: Resources
- `python/docs/README.md`: the guide
- `python/docs/API_REFERENCE.md`: every class, method and setting
- `examples/notebooks/01_tutorial_getting_started.py`: the hands-on tutorial (jupytext)
- `examples/notebooks/demo.ipynb`: four models rendered as videos
- `examples/models/`: small NeuroML models to start from

### Slide 34: Hands-on exercises
- Run the single cell, then change its drive and plot an f–I curve
- Build a 10×10 torus with an 80/20 excitatory/inhibitory split and plot a raster
- Find the excitatory synapse's `I` plane and plot it across connections after a run
- Record a membrane video and render it
- Open `engine.master_kernel_source` and find your cell's equations in it

### Slide 35: Questions
- Questions and discussion
- What models from the lab's work should we try next?
