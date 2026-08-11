# NeuroML / LEMS — general notes

General language semantics, independent of this repo's implementation.


## ComponentTypes and `extends`

- `extends` is optional and single-parent. A ComponentType with no parent is legal and simulable.

- Custom ComponentTypes usually extend a std-lib base, because the base is what makes them usable:
  a population's `component` must reach `baseCell`, a projection's `synapse` must reach `baseSynapse`,
  an `explicitInput` target must reach `basePointCurrent`. That contract is what makes the extends
  chain a viable classifier.

- Three ways that assumption breaks:
  1. A type may be a root with no `extends`. About a third of the std lib is roots — 93 of 273.
  2. A user type may extend another user type, multi-hop, never naming a std-lib type directly.
  3. The parent may be declared later in the same file, or in a different file.

- Most real NeuroML documents define zero custom ComponentTypes — they instantiate std-lib ones
  (`<izhikevich2007Cell id="cell0" .../>`). Custom types are the LEMS escape hatch, used when no
  std-lib cell matches the model.

Consequence: declaration order is not significant. `extends` is a name reference resolved by lookup,
so resolution must be two-phase — catalog every type across every included file first, then walk the
chains. Single-pass resolution breaks on forward and cross-file references, both of which occur in
the std lib itself.


## Duplicate ComponentType names

- ComponentType names are a single flat global namespace. No scoping, versioning, or per-file
  namespacing.

- Redefinition is not a defined override. The reference implementation (jLEMS) treats it as an error.
  Exact error behavior is worth verifying against jLEMS if you need to match it.

- The common source of apparent duplicates is diamond includes, not redefinition: `Cells.xml` and
  `Synapses.xml` both include `NeuroMLCoreDimensions.xml`. LEMS makes includes idempotent, so the
  type is defined once.

- Therefore: dedupe includes by resolved canonical path, not by the raw `href` string. Then treat any
  surviving duplicate as a real conflict. Silent first-wins or last-wins makes the resulting model
  depend on filesystem iteration order.


## What can share a name

Names must be unique within a namespace. Collisions are legal only across namespaces.

The namespaces inside one ComponentType:

    variables       Parameter, Constant, DerivedParameter, Property,
                    Requirement, StateVariable, DerivedVariable
    exposures       Exposure
    event ports     EventPort
    path segments   Child, Children, Attachments, ComponentReference, Link
    regimes         Regime

- The variables namespace is the unusual one: seven tags feed a single namespace, so `Parameter "tau"`
  vs `StateVariable "tau"` is a conflict despite being different tags. Every other namespace is made
  of one or two tags.

- `Exposure "v"` + `StateVariable "v"` is the universal convention, not a collision — a variable binds
  to an exposure of the same name via `exposure="v"`.

- `Children "synapses"` + `Attachments "synapses"` IS a conflict; both are path segments.

- Across different ComponentTypes there is no constraint at all. `v`, `tau`, `C`, `spike` recur
  everywhere. Names are only meaningful within their declaring type.

- Inheritance override: a subtype redeclaring an ancestor's name is legal and is the override
  mechanism — most-derived wins. Not a duplicate.

- Instance ids repeat across populations. `Pop1[3]` and `Pop2[3]` are different neurons; indices are
  scoped to their population.


## Path segment tags

All five declare a name addressable in `select=` paths. They differ in who owns the target and when
it appears.

    Child                exactly one sub-component, nested literally in the XML
    Children             a list (0..N) of sub-components, nested literally in the XML
    Attachments          like Children, but populated at network build time, not written in the XML
    ComponentReference   a reference by id to a component owned elsewhere
    Link                 also a reference to another component instance; peer links within a structure

- `Child` and `Children` own their contents. `ComponentReference` and `Link` are pointers — the
  instance supplies an id.

- `Attachments` is how network connectivity reaches cell dynamics:

      <Attachments name="synapses" type="basePointCurrent"/>
      <DerivedVariable name="iSyn" dimension="current" select="synapses[*]/i" reduce="add"/>

  The cell declares that synapses may attach; nothing is nested in the XML. A `<projection>` attaches
  a synapse instance at build time, and the cell sums current over whatever landed there.

- `Link` vs `ComponentReference` is contextual — `Link` in kinetic schemes, `ComponentReference` for
  "this component uses that one". Treating them as one namespace is correct.


## Regime

Turns a ComponentType into a state machine. Dynamics declared inside a regime apply only while the
component is in that regime.

    <Regime name="integrating" initial="true">
      <TimeDerivative variable="v" value="(gL * (EL - v) + iSyn) / C"/>
      <OnCondition test="v .gt. vth">
        <EventOut port="spike"/>
        <StateAssignment variable="v" value="vreset"/>
        <Transition regime="refractory"/>
      </OnCondition>
    </Regime>

    <Regime name="refractory">
      <OnEntry>
        <StateAssignment variable="refractoryTimeElapsed" value="0"/>
      </OnEntry>
      <TimeDerivative variable="refractoryTimeElapsed" value="1"/>
      <OnCondition test="refractoryTimeElapsed .geq. t_ref">
        <Transition regime="integrating"/>
      </OnCondition>
    </Regime>

- `initial="true"` marks the starting regime.
- `<OnEntry>` runs once on entry, not per tick.
- `<Transition regime="..."/>` inside an `OnCondition` performs the switch.
- In `refractory`, `v` has no `TimeDerivative` at all — it is frozen. Without regimes this needs a
  conditional multiplier on every derivative.
- A ComponentType may have dynamics both inside regimes and directly under `<Dynamics>`; the latter
  always apply.

Consequence for parsers: regimes carry the `TimeDerivative`s. Any declaration extractor that stops
descending at a `<Regime>` loses the cell's entire dynamics.


## Structural ComponentTypes

Many ComponentTypes map to no runtime entity at all. They are the structural layer: read once at
build time to decide what instances exist and how they are wired, then done. Nothing about them is
integrated.

Every network-assembly tag below has no `<Dynamics>` block. That test is not universal across all of
NeuroML — some biophysical container types carry a trivial `<Dynamics>` purely to re-expose a
parameter — but it holds exactly for the network-construction layer.

Network container:

    network                  top-level container: holds populations, projections, and inputs
    networkWithTemperature   same, plus an explicit temperature for temperature-dependent
                             elements (Q10 scaling on channel rates)

Population contents and placement:

    instance                 one instance of a component in a populationList
    location                 the (x,y,z) coordinate of a single instance
    region                   a named 3D region to place cells in (marked "work in progress")
    rectangularExtent        a 3D box, used to define a region

Two population styles: `population` with `size="N"` only says how many; `populationList` holds
explicit `instance`/`location` children, which is what gives real 3D coordinates.

Connectivity — the edge declarations inside a projection:

    connection                      event edge; processed via a NEW synapse instance
    connectionWD                    same, plus per-edge weight and delay
    explicitConnection              explicit event edge between named components
    synapticConnection              explicit edge naming its own synapse
    synapticConnectionWD            same, plus weight and delay
    electricalConnection            gap-junction edge
    electricalConnectionInstance    gap junction, for populationList-style populations
    electricalConnectionInstanceW   same, with weight
    continuousConnection            graded/analog edge (no discrete events)
    continuousConnectionInstance    graded edge, populationList style
    continuousConnectionInstanceW   same, with weight

The `...Instance` suffix means it addresses `../population/i/cellType` paths (populationList style)
rather than plain indices. The `W` suffix means it carries a weight.

Their containers — `projection`, `electricalProjection`, `continuousProjection` — are structural too:
a projection declares the pre/post populations and the synapse prototype, and its child connections
declare the individual edges.

Input wiring:

    explicitInput   attaches one input component to one target cell
    inputList       a list of inputs applied to a population, naming the input component once
    input           a single entry in an inputList; can name segmentId and fractionAlong
    inputW          same as input, plus a weight to scale it

`explicitInput` is the one-off form; `inputList`/`input` is the bulk form.

Simulation control — a different layer, the LEMS Simulation file rather than the NeuroML model.
These decide what gets simulated and recorded, nothing about the network's content:

    Simulation        the run itself: length, timestep (dt), and target network
    Display           a plot window
    Line              one trace on a Display
    OutputFile        a file to save recorded values into
    OutputColumn      one recorded quantity within an OutputFile
    EventOutputFile   a file to save spike events into
    EventSelection    which event source to record into that file
    Meta              metadata attached to the simulation

Cell-internal structure — the same idea one level down, describing a cell's anatomy rather than a
network's:

    morphology, segment, parent, proximal, distal             the 3D branching structure
    segmentGroup, member, include, path, from, to, subTree    named groups of segments
    biophysicalProperties, membraneProperties,
      intracellularProperties                                 containers
    channelDensity (+ GHK/Nernst/NonUniform/VShift variants)  where channels sit and how dense
    specificCapacitance, resistivity, initMembPotential,
      spikeThresh                                             per-region scalars
    species, concentrationModel variants                      ion pools that change with time
    inhomogeneousParameter, inhomogeneousValue,
      variableParameter, proximalDetails, distalDetails       parameters varying along a morphology

Metadata — inert, no effect on simulation: `notes`, `annotation`, `property`, `baseStandalone`, plus
roughly 50 RDF / Dublin Core / FOAF wrapper types (`dc_creator`, `bqbiol_is`, `foaf_name`, ...) that
exist only so provenance can be embedded in the file.

The load-bearing subset is small. `network`, `population`/`populationList` with `instance`/`location`,
`projection` with the `connection*` family, and `explicitInput`/`inputList`/`input` are what turn a
pile of ComponentType definitions into an actual graph of instances.


## Point cells vs biophysical cells

- Abstract point cells (`iafCell`, `izhikevich2007Cell`, `adExIaFCell`, `fitzHughNagumoCell`, the GLIF
  variants) write their dynamics directly as `TimeDerivative`s on a few state variables. No channels,
  no gates, no morphology, no compartments.

- Biophysical cells are a `<cell>` with `<morphology>` and `<biophysicalProperties>`, where
  `<channelDensity>` references `<ionChannel>` components carrying `<gate>`s (`gateHHrates`, ...) or
  kinetic schemes (`KSState`/`KSTransition`). Membrane current is decomposed into per-channel gated
  conductances, per compartment.

Two distinctions worth keeping separate:

- Channels are not the same thing as nonlinearity. Izhikevich and AdEx are nonlinear point cells with
  no channels at all. Nonlinearity is the shape of the ODE; channels are a decomposition of current.

- "Conductance-based" is overloaded. `expOneSynapse` computes `g * (erev - v)` and is a
  conductance-based SYNAPSE — per-edge state, no ion-channel machinery involved. A network of point
  cells can be full of conductances without any channel components.


## Misc syntax

- Comparisons in `test=` use `.gt. .geq. .lt. .leq. .eq. .neq.` — `<` and `>` cannot appear raw in
  XML attributes.

- Raw `<ComponentType>` declarations do not validate against the NeuroML2 XSD. The usual workaround is
  to put them in their own file and hand the validator a top-level file that only `<include>`s it.
