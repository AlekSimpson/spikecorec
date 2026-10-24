# spikecorec: lab presentation outline (draft)

Draft of 2026-10-19. The talk is for the lab: modelers and biologists, most of whom have not
used spikecorec. Assume no GPU or compression background and explain each step. The PI's
request (2026-10-16) was to "go back to the beginning and just explain the main ideas"; it
does not need to be finished.

**Format:** six parts, about 26 slides plus live demos, roughly 40 minutes. Each slide lists:
- what it says;
- the visual;
- speaker notes, where the point is not obvious.

`[TODO]` marks what only the presenter can supply. The numbers are measured; their sources are
listed at the end.

**The demos and their tutorials:**
- `examples/notebooks/01_tutorial_getting_started.ipynb`: one cell to a small E/I network.
- `examples/notebooks/02_tutorial_large_networks.ipynb`: the CUBA benchmark network from numpy
  arrays, what the weight matrix holds and how much memory it takes, structured against random
  weights, a membrane video, and the generated code.

**Time budget:**

| Part | Slides | Minutes |
|---|---|---|
| 1. Introduction | 1-3 | 3 |
| 2. The olfactory bulb model | 4-7 | 6 |
| 3. spikecorec | 8-15 | 10 |
| 4. The weight matrix | 16-21 | 9 |
| 5. Pros and cons | 22-23 | 4 |
| 6. Usage and demos | 24-28 | 8 |

---

## Part 1: Introduction

### Slide 1: Title
- spikecorec: large spiking networks on one GPU, from NeuroML
- Presenter, lab, date, and a link to the repository
- **Visual:** the CUBA network's membrane video looping (notebook 02, section 7)

### Slide 2: Where this talk goes
- The olfactory bulb model we work with, and what made it hard to work with in NEURON
- spikecorec: what it is and what it can simulate
- The weight matrix: how it stores a network's connections, and what that costs
- Using it from Python: demos

### Slide 3: The goal
- Simulate large networks of spiking neurons quickly, on hardware the lab has: a Mac, or one
  machine with an NVIDIA GPU
- The model is a standard document (NeuroML), not code written for one simulator
- At scale the limit is memory for the connections, not the neurons. Take 1 million neurons
  with 1,000 connections each, and a weight, a delay and one synapse variable per connection:
  - 1 billion connections;
  - about 32 GB as plain sparse lists in both directions;
  - 4 TB per value as a dense matrix.
- spikecorec stores connectivity compressed, so that networks like this fit on one GPU
- **Say:** the engine is the means. The goal is to iterate on network models quickly.

---

## Part 2: The olfactory bulb model

### Slide 4: The OB model
- Birgiolas (2019), ASU dissertation: *Towards Brains in the Cloud: A Biophysically Realistic
  Computational Model of Olfactory Bulb*
- **Cells:** mitral, tufted and granule cells, five variants of each, with reconstructed
  morphologies (NeuroMorpho.org). They were fitted to and validated against experimental data
  with NeuronUnit.
- **Network:**
  - cells are placed in the bulb's layers, as reconstructed in Blender;
  - reciprocal dendrodendritic synapses (AMPA/NMDA onto granule cells, GABA back) form wherever
    dendrites pass within 5 µm;
  - gap junctions join the tufts within a glomerulus.
- **Input and output:** odor input as glomerular activation patterns; output includes a
  simulated LFP.
- **Result:** reproduces the two-cluster gamma fingerprint (30-120 Hz) in the LFP
- **Our use:** the ketamine / HFO question, and gamma-beta alternation
- **Visual:**
  - first `OlfactoryBulb/media/cell-gallery-start.jpg` and `reconstructed-bulb-layers.png`;
  - then `gamma-fingerprint-simulation-output.png`.
- **Say:** keep this short, since the lab knows the model. It sets up what follows.

### Slide 5: NEURON and NeuroML
- **NEURON** is the standard simulator for biophysically detailed neurons:
  - each cell is a tree of cable sections;
  - ion channels and synapses are mechanisms written in NMODL (`.mod` files);
  - models are built in hoc or Python;
  - large networks run in parallel over MPI.
- **The OB model is written for NEURON:**
  - 13 `.mod` mechanisms;
  - cells from NEURON's Cell Builder, wrapped in Python;
  - the network built from slice files;
  - parameter sets as Python classes.
- **NeuroML** is a standard, simulator-independent description of a model in XML: cells,
  channels, synapses, populations, projections and inputs.
- **LEMS**, underneath NeuroML, defines each type's dynamics: state variables, equations and
  events. One NeuroML model runs in NEURON, NEST, Brian and others through jNeuroML/pyNeuroML.
- **The OB model has no NeuroML version.** What exists so far:
  - the live NEURON model dumped to one flat hoc file (`export.py`);
  - the slice's connectivity as an adjacency list (`parse_topology.py`).
- **Visual:** a small integrate-and-fire cell as NeuroML (`examples/models/single_cell.nml`)

### Slide 6: Challenges with NEURON
- **BlenderNEURON:**
  - The network's structure is built interactively: NEURON sends the cells to Blender, where
    they are placed and bent into the bulb's layers.
  - Rebuilding or changing the structure needs Blender, the add-on and the `.blend` files.
  - It was one student's tool, it is limited, and much of it is visualization the simulation
    doesn't need.
- **Model control:**
  - A model's settings live in many places: `.mod` files, hoc, Python parameter classes, an
    SQLite database and the slice files.
  - Some values are hard to justify; tufted cells have a specific capacitance of
    7.9 µF/cm², for example.
  - It is hard to isolate which mechanism produces an effect, such as what drives the HFO in
    the LFP.
  - It was built for one study, and is hard to iterate on.
- **Performance:**
  - The documented run is a 1.8 s simulation of a 193-cell slice (10 mitral, 24 tufted,
    159 granule cells), on 16 MPI processes.
  - The documentation plans for 96-core cloud machines.
  - `[TODO: your own measured run time for one simulation]`
  - Parameter sweeps and larger slices are out of reach on a workstation.
- **Say:** none of this is a fault of NEURON for what it is built for, which is detailed
  single-cell biophysics. It is what got in the way of network-level questions.

### Slide 7: What we wanted instead
- **The model as a document:** change a value in one place and rerun; the same file runs in
  other simulators
- **Network-level questions with point neurons:** many runs, each quick
- **Scale:** from hundreds of cells to millions, on one machine
- **Python first:** notebooks, with numpy in and out
- **Visual:** the four points, leading into spikecorec

---

## Part 3: spikecorec

### Slide 8: What spikecorec is
- A simulator for networks of spiking point neurons
- **In:** a NeuroML/LEMS model. **Out:** spikes and state traces, as numpy arrays
- It generates GPU code from the model's own equations and compiles it when the engine is
  built; nothing is written by hand for each model
- **Clock-driven:** the whole network advances one timestep (a "tick") at a time, with every
  neuron in parallel
- A C++ engine with a Python package on top
- **Visual:** the pipeline as a diagram:
  - LEMS + NeuroML + the standard types;
  - parse, validate, resolve units and paths;
  - one GPU kernel, compiled;
  - spikes and traces.

### Slide 9: How a run works
- Each tick, for every neuron in parallel:
  - run its inputs;
  - integrate its equations one `dt` (forward Euler);
  - detect the threshold, spike and reset;
  - deliver spikes along its outgoing connections, after each connection's delay;
  - record.
- Thresholds and arrivals are checks made every tick, not a queue of events
- A spike reaches its target exactly its delay later, and at least one tick later
- **Visual:** the nine stages as a ring: Deliver, Integrate, Detect, Emit, Reset, Propagate,
  Plasticity, Record, Advance. Stages 2-5 are the model's own dynamics, which the generated
  code fills in.

### Slide 10: Current-based synapses
- A synapse adds a current to its target. The current depends on the synapse's own state and
  its connection's weight, never on the target's voltage.
- **For contrast:** a conductance-based synapse scales by (E_rev - V) of its target.
  spikecorec refuses those when the engine is built, and says why.
- **Why this design:**
  - a synapse's update runs while its source's connections are walked, and needs nothing from
    the target;
  - every connection still keeps its own synapse state, for example its own exponential or
    alpha current.
- **Delta synapses** (each spike adds a fixed jump to the target) keep no state at all. A neuron
  walks its connections only on the tick an arrival is due.
- The standard large-network benchmarks are current-based: Brunel 2000, Vogels-Abbott CUBA and
  Potjans-Diesmann 2014
- **Say:** this is the main modelling restriction, and it is what makes per-connection state
  cheap.

### Slide 11: Arbitrary topologies
- Any directed graph: recurrent or feedforward, any degree distribution, self-connections
  included
- Every connection has its own weight, delay and synapse type, with several synapse types in
  one network
- Repeated connections between one pair (multapses) become one connection whose synapse
  carries the multiplicity
- **Three ways in:**
  - projections written in the NeuroML document;
  - from Python: an adjacency list, or a function that gives each connection's values;
  - five numpy arrays (source, target, synapse, weight, delay), for millions of connections.
- Built-in generators: a torus grid, a small world, and a random graph with fixed out-degree
- **Measured:** Brunel's model A (12,500 neurons, 14.9 million connections) builds in 4.4 s
  from arrays
- **Visual:** the same three-cell chain as XML connections, as an adjacency list, and as
  arrays

### Slide 12: NeuroML support
- Reads standard LEMS/NeuroML with the NeuroML core types included; custom ComponentTypes work
  the same way
- **Cells:** any point-neuron type built from LEMS dynamics. That covers:
  - state variables, derived variables and time derivatives;
  - threshold conditions and spike handlers;
  - an integrating/refractory regime pair;
  - `random()`.
  This includes GLIF1-5, the integrate-and-fire variants, Izhikevich, AdEx, the PyNN cells, and
  your own types.
- **Synapses:** current-based, with their own kinetics
- **Inputs:** every NeuroML input (pulse, sine, ramp, Poisson, spike arrays, voltage clamps),
  run on the GPU
- **Recording:** the document's output files, plus every spike and membrane videos
- Anything outside this is refused when the engine is built, naming the component and the
  reason. It never runs silently wrong.
- **Visual:** the CUBA demo's custom cell, side by side with its generated line:
  - the cell is PyNN's `IF_curr_exp` plus a random starting voltage, about 25 lines of LEMS;
  - its `TimeDerivative` becomes one line of generated code (notebook 02, section 8).

### Slide 13: GPU accelerated by default
- **Backends:**
  - Metal on Apple silicon Macs;
  - CUDA on NVIDIA GPUs, tested on an NVIDIA GB10 (DGX Spark) with CUDA 13.
- The same model runs on either; the backend is chosen when spikecorec is built
- One GPU thread per neuron. The model's kernel is compiled at run time for the device.
- The weight-matrix fits also run on the GPU, with a CPU fallback
- Both test suites pass on both backends: 238 C++ tests and 24 Python tests
- **Speed** (M5 Max, Metal, a baseline from before any speed work; 20 connections per neuron,
  alpha synapses):

  | Neurons | Connections | ms per tick | Ticks per second |
  |---|---|---|---|
  | 1,024 | 20,480 | 1.25 | ~800 |
  | 10,000 | 200,000 | 4.1 | ~247 |
  | 99,856 | 2.0 M | 84 | ~12 |
  | 10,000, no connections | 0 | 0.18 | ~5,500 |

- **Say:** connections, not neurons, are what cost time.

### Slide 14: The Python package
- `import spikecorec as spc`, built with `make python`
- **A complete run in three lines:**
  - `engine = spc.Spike_Engine("LEMS_model.xml")`
  - `engine.run()`
  - `times, neurons = engine.recorded_spikes`
- numpy in (connection arrays) and numpy out (spikes, state and recordings)
- The API mirrors the C++ engine one to one. From Python you can inspect:
  - the parsed model (`spc.nml`);
  - the weight matrix;
  - the generated GPU code.
- **Docs:**
  - `python/docs/README.md`, the guide;
  - `python/docs/API_REFERENCE.md`, every name;
  - the two tutorial notebooks.

### Slide 15: Does it give the right answers?
- **Method:** validated statistically against NEST (the reference) and Brian2 on standard
  networks, 10 seeds each:
  - statistics: firing rates, ISI CV, pairwise correlations and spectra;
  - score: the KL divergence of each statistic's distribution, divided by NEST's own
    seed-to-seed spread (pass ≤ 1, marginal 1-2, fail > 2);
  - Brian2 is scored the same way, as a check on the scoring.
- **Vogels-Abbott CUBA** (4,000 neurons): 5 of 6 pass, and the excitatory CV is marginal
  (1.07). Rates are 5.69 / 5.62 Hz against NEST's 5.49 / 5.55.
- **Brunel 2000 model A**, points A-D (12,500 neurons):
  - A, B and C pass; A's correlation is marginal at 1.29 (Brian2: 1.23);
  - D's rate is 3.4% low and fails, which we trace to forward Euler on the leak.
- **Brunel phase diagram** (59 grid points): the same dynamical regime as NEST at 56 points
  (Brian2: 54)
- **Potjans-Diesmann microcircuit** (77,000 neurons, 299 million synapses): in progress
- **Visual:** the scorecard, figure 6 `[TODO: make it]`. Until then use
  `spikecore_runs/figures/output/m2_figure_3_verdicts.png` and `m2_figure_1_raster.png`.
- **Say:** we don't compare spike times. In a chaotic network a 10⁻¹³ change in one voltage
  diverges, so what must match is the statistics.

---

## Part 4: The weight matrix

### Slide 16: The storage problem
- A network's connections dominate its memory: N neurons and E connections, each carrying P
  values (weight, delay and each synapse state variable)
- **Dense matrix:** N² × P floats, so 4 TB per value at 1 million neurons
- **Sparse lists (CSR):** one index plus P values per connection. Walking outgoing connections
  (spikes) and incoming ones (fits, inspection) needs two copies, CSR + CSC.
- Real networks are built by rules (populations, probabilities, cell properties), so the
  values have structure that plain lists don't use
- spikecorec stores the two halves separately:
  - which pairs connect (the topology);
  - the values on them.
- **Visual:** an adjacency matrix, mostly empty, with the connections coloured by weight

### Slide 17: Topology: the k²-tree
- **Structure:**
  - Split the N × N adjacency matrix into 4 × 4 blocks (k = 4), recursively.
  - Each block gets one bit: 1 if it holds any connection.
  - Only blocks with a 1 are split further, and the last level holds the connection bits.
- Empty regions cost one bit, at the highest level they appear. Clustered connectivity
  compresses best.
- It is stored as flat bit arrays plus rank tables. The GPU walks it directly:
  - a row gives a neuron's targets;
  - a column gives its sources;
  - there are no pointers and no second copy.
- **Measured:** on the CUBA network (4,000 neurons, 2% connectivity, 321,000 connections) it
  takes 18.8 bits per connection, against 32 for one sparse index and 64 for CSR + CSC
- **Visual:**
  - an 8 × 8 matrix with k = 2 for legibility;
  - its blocks, the tree with a bit on each node, and the two bit strings.

### Slide 18: Values: planes and a shared low-rank basis
- **Planes:** each kind of per-connection value is a plane. Plane 0 is the weight, plane 1
  the delay, then the synapse's state variables.
- **Factorization:** no plane is stored as one number per connection. The value on plane k
  from source s to target t is

      value_k(s, t) = Σ over lanes l of  U[s, l] · C[k, l] · V[t, l]

  that is, plane k = U · diag(C_k) · Vᵀ, read only where the tree has a connection.
- **U, V and C:**
  - U and V have one row per neuron and one column per lane;
  - C has one row per plane, which selects and scales that plane's own lanes.
- **Memory:** 2 × N × lanes floats for every plane together, instead of E floats per plane.
  A plane with population structure needs a handful of lanes.
- The fit uses only existing connections; everywhere else is unconstrained
- **This is compression, not learning:** the lanes only store the network
- **Visual:**
  - U (N × L) times diag(C_k) times Vᵀ (L × N), masked by the tree's pattern;
  - colour each plane's lanes.

### Slide 19: Choosing the lanes
- **Tolerance:** the worst error allowed on any value, relative to the larger of the value
  and its plane's RMS. The default is 0.1%, with presets from 0.001% to 1%.
- **Lane search, per plane:**
  - fit at 1, 2, 4, ... lanes until the tolerance is met, then bisect to the fewest (L-BFGS, on
    the GPU);
  - doubling stops once it stops helping.
- **Budgets:**
  - each plane keeps that count plus 20% as a fixed budget for the run;
  - the total never exceeds one float per connection per plane.
- **Delays** are rounded to whole ticks when read, and stay exact up to 500 ticks at the
  default tolerance
- **Synapse-state planes** start at zero, so they are sized by a short calibration run: every
  neuron spikes once, then every state is restored.

### Slide 20: Values that change: S and refits
- Synapse state changes every tick. The changes go into a sparse update buffer, **S**, which
  every read adds in.
- **Refits:**
  - when S holds changes on 20% of the connections, and at least 1,000 ticks have passed, a
    refit folds S into U, V and C and empties S;
  - the refit is warm-started alternating least squares on the GPU;
  - only the rows at either end of a changed connection move.
- S grows rather than drop a change, and shrinks back after each refit
- Delta synapses keep no state, so they need no plane, no S and no refits
- **Visual:** a timeline: ticks pass, S fills, a refit runs, S is empty again

### Slide 21: Measured on the CUBA network
4,000 neurons, 321,000 connections and three planes (weight, delay, synapse current), with
weights given three ways:
- the benchmark's own two weights;
- a structured ±10% spread (one factor per source cell and one per target cell);
- a random ±10% spread on every connection.

| | Benchmark weights | Structured spread | Random spread |
|---|---|---|---|
| Lanes: weight / delay / synapse current | 2 / 2 / 2 | 3 / 2 / 3 | 50 / 2 / 53 |
| Worst fit error, weight plane | 0.08% | 0.10% | 0.10% |
| Basis (U, V, C) | 0.26 MB | 0.26 MB | 3.46 MB |
| Basis / raw values (3.85 MB) | 7% | 7% | 90% |
| Whole weight matrix, as built | 2.1 MB | 2.1 MB | 5.4 MB |
| As built, share of CSR + CSC (10.3 MB) | 21% | 21% | 52% |
| Construction | 8.2 s | 8.8 s | 25 s |
| ms per tick | 4.3 | 4.3 | 5.3 |

- **During a run S grows.** Each synapse runs for 138 ms after its source spikes, so at
  5.7 Hz most connections hold a change. Between refits S grows to room for 80% of the
  connections. The whole matrix then comes to 3.7 MB, 36% of CSR + CSC, with the benchmark
  weights.
- **Visual:** the stacked bar chart from notebook 02, section 6: tree, basis, S and queue,
  against the CSR + CSC and raw-value lines.

---

## Part 5: Pros and cons

### Slide 22: Pros
- **Memory:**
  - the whole matrix is a fifth to a third of CSR + CSC on the CUBA network;
  - a synapse variable kept on every connection costs a few lanes per neuron, not a float per
    connection.
- **Scales with structure, not with connection count:**
  - the basis is lanes × neurons, not connections;
  - denser networks gain more. CUBA has 80 connections per neuron and Brunel's model A 1,190;
    Brunel holds its weights in 4 lanes at 0.08% worst error.
- **One structure answers both directions** (targets and sources), where lists need two
  copies
- **GPU-native:** the kernel walks the tree and computes each value where it needs it
- **Accuracy is a setting:** the tolerance trades memory against error

### Slide 23: Cons
- **Approximate:** every value is within the tolerance (0.1% by default), not bit-exact
- **Unstructured values don't compress:**
  - a random spread on every connection needed 50 lanes, 90% of the raw size;
  - the cap keeps the basis below one float per connection per plane, so this worst case is
    about raw size plus the tree;
  - repeated pairs stored as doubled weights left 48% errors in Brunel's network, so the
    multiplicity goes in the synapse instead.
- **Construction cost:**
  - the lane search takes 8-25 s at 321,000 connections;
  - 3% of the Potjans-Diesmann network takes about 10 minutes;
  - at full size (285 million connections) it had not finished after more than an hour, before
    the latest speedups `[TODO: re-measure]`.
- **Reads are computations:**
  - each value is a dot product over lanes, plus a walk of the tree;
  - walking connections dominates tick time: 10,000 neurons take 0.18 ms per tick with no
    connections and 4.1 ms with 200,000.
- **Changing values cost memory and time:**
  - S and its queue are the largest part during a CUBA run;
  - refits interrupt the run.
- **Complexity:** harder to debug and to explain than arrays
- **Say:** the design bets that real networks are built by rules. Where they are, it saves a
  lot; where values are pure noise, it falls back to about raw size.

---

## Part 6: Usage and demos

### Slide 24: Demo 1: one cell from NeuroML (notebook 01, sections 1-2)
- `Spike_Engine` on `LEMS_single_cell.xml`: 50 spikes in 500 ms. That is a GLIF1 cell at 2.5×
  rheobase, and it matches the analytic interspike interval.
- Show `single_cell.nml`, change the drive, and plot the f-I curve: 250, 500 and 1000 pA
  give 48, 100 and 140 Hz
- **Say:** the document is the whole model; there are no hidden settings in Python.

### Slide 25: Demo 2: a benchmark network from numpy arrays (notebook 02, sections 1-3)
- **The model:** the Vogels-Abbott CUBA network, 3,200 excitatory and 800 inhibitory cells at
  2% connectivity
- **Building it:**
  - the document declares the cells, synapses and populations, including a custom
    ComponentType of about 25 lines;
  - the connections are five numpy arrays;
  - it builds in about 8 s.
- **Result:** a raster and the population rates, about 5.7 Hz against NEST's 5.5 Hz

### Slide 26: Demo 3: inside the weight matrix (notebook 02, sections 4-6)
- Lanes per plane, worst fit errors, and weights read back against those given
- The memory breakdown against CSR + CSC
- Rebuild with structured and with random weights, and watch the lanes and the memory change

### Slide 27: Demo 4: a membrane video (notebook 02, section 7)
- **From a run to a video:**
  - `record_membrane_video` before the run writes a `.spire` file;
  - `render_membrane_video` draws it as a grid of all 4,000 cells, one excitatory and one
    inhibitory trace in millivolts, and a spike raster.
- Half a second of activity, with the spikes visible on the traces
- **Then the generated code** (section 8): `master_kernel_source`, with each `TimeDerivative`
  as one line of GPU code and its parameters baked in

### Slide 28: Getting started, and questions
- `make python`, then the two tutorial notebooks
- `python/docs/README.md` and `API_REFERENCE.md`; small models in `examples/models/`
- **Scope:** point neurons and current-based synapses
- **What's next:**
  - `[TODO: what you want to say is next, e.g. the Potjans-Diesmann validation, faster
    construction for very large networks, and an OB-like network]`
- Questions: what models from the lab's work should we try first?

---

## Backup slides

### B1: What a cell model can contain
- State variables with starting values (`OnStart`), derived variables (conditional ones
  included), and time derivatives integrated with forward Euler every tick
- Threshold conditions (`OnCondition`) that assign state and emit spikes, and spike handlers
  (`OnEvent`) that run once for each spike that reaches the cell
- An integrating regime and a refractory regime that returns to it; either can have its own
  derivatives, conditions and `OnEntry`
- `random()`: a fresh draw for each neuron on every call, seeded by the simulation's seed

### B2: What gets refused, and why
- Synapses whose current depends on the target cell, and `random()` inside a synapse
- Regime structures other than the integrating/refractory pair
- Ion channels, concentrations, multicompartment morphologies and kinetic schemes
- A synapse or input that does not provide what its target cell reads
- Each is refused when the engine is built, with the component and the reason

### B3: Fit and refit settings
- `fit_tolerance`, with presets `FIT_TOLERANCE_PRECISE` (1e-5), `ACCURATE` (1e-4), `STANDARD`
  (1e-3, the default) and `COMPACT` (1e-2)
- `expected_active_edge_fraction` (0.2): the share added to each plane's lanes
- `refit_occupancy_threshold_fraction` (0.2), `minimum_ticks_between_refits` (1000) and
  `refit_every_n_ticks` (off)
- What to watch:
  - `weights.measured_fit_error`;
  - `weights.sparse_delta_occupancy_fraction()`;
  - `engine.last_refit_tick`.

### B4: Python API in one slide
- **Build:**
  - `Spike_Engine(lems)`;
  - `Spike_Engine(lems, adjacency, synapse_id, connection_weight=..., connection_delay_seconds=...)`;
  - `Spike_Engine(lems, synapse_ids, source_neurons=..., target_neurons=..., synapse_indices=..., weights=..., delays_seconds=...)`.
- **Run:** `run()`, `run(after_each_tick)`, or `step_simulation(tick)` one tick at a time
- **Read:** `recorded_spikes`, `spike_counts_per_neuron`, `state_variable_array("v")`,
  `recorded_traces`
- **Record:** `record_membrane_video(path, stride)`, `write_recordings()`, `write_spike_file(path)`
- **Inspect:** `engine.context` (the parsed model), `engine.weights` (planes, lanes, k²-tree),
  `master_kernel_source`

---

## Where the numbers come from

- **CUBA memory, lanes, construction and tick times:** measured 2026-10-19 on the M5 Max
  (Metal, `a15a542`) with notebook 02. S's growth during a run was traced tick by tick in the
  same session.
- **Validation:** `~/Documents/neurocomputation/spikecore_runs/RESULTS_LOG.md` (M2, M1 point A).
  M1 points B-D and the phase sweep are in spark's `~/neurocomputation_benchmarks/spikecore_M1/RESULTS.md`.
- **Brunel construction (4.4 s, 4 lanes, 0.08%) and the 48% error of doubled weights:** RESULTS_LOG,
  2026-10-16
- **Speed table:** `docs/performance_review.md`, simulation speed (2026-10-07 baseline)
- **Potjans-Diesmann construction times:** `docs/large_network_construction.md`
- **The OB model:** the `OlfactoryBulb` repository (readme, `docs-source/`, `olfactorybulb/`) and
  the lab meeting transcripts of 2026-06-19 to 2026-10-16
