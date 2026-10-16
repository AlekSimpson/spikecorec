# Resilience review

State as of 2026-10-12, branch `agent_integration_2`, commit `04d1b3d`. This review identifies
and confirms defects; it changes no code. Performance findings from the same review are in
`docs/performance_review.md` (items 30–32). Findings 44–51 were added on 2026-10-13, found while
checking the code comments.

Every finding below was confirmed in one of two ways:
- **reproduced**: a scratch program or NeuroML document showed the failure;
- **traced**: the code path was followed line by line.

Severity:
- **high**: silently wrong results in ordinary use, or a crash in ordinary use;
- **medium**: wrong results or crashes that need a less common input or call sequence;
- **low**: diagnostics, edge cases, or API misuse.

## Contents

- A. Regressions from the restructure (fixed)
- B. Wrong simulation results
- C. Input documents accepted when they should be refused
- D. Recording and output files
- E. Crashes, memory safety and threading
- F. Smaller API and diagnostic defects
- G. Found during the comment check

## A. Regressions from the restructure (fixed)

The restructure in `16d02bb` introduced the following. Each was found, fixed and pushed during
this review; they are listed here for the record.

| Defect | Effect | Fixed in |
|---|---|---|
| `TYPE_ALIGNMENTS[datatype]` on a `const` map (`backend.cpp`) | build failed | `b36b0a4` |
| `log::Shared_Pointer` (`engine.h`) | build failed | `b36b0a4` |
| `static_cast` in `k2tree_device.metalinc`, which the kernel parser could not read | every engine construction threw | `21279e3` |
| `Duration` ambiguous with the macOS `MacTypes.h` typedef (`weight_matrix_tests.cpp`) | test build failed | `21279e3` |
| `grow_basis` coefficient copy bound `lane < lane_count` instead of `old_lane_count` | corrupted coefficients and run-to-run variation; 15–17 failing tests | `04d1b3d` |

After `04d1b3d`, `make test` passes 188/188 and `make test-python` 17/17. Apart from these, the
restructure changed names, casts and formatting only. `log.cpp` was rewritten from spdlog to an
in-house formatter, and all 130 format calls were checked against it.

## B. Wrong simulation results

### 1. The per-combination basis mixes lanes, so weights and delays read back wrong (high)

When a network's per-connection values come from a small set of combinations,
`Weight_Matrix::fit_basis_from_connections` gives each combination its own lane. It then sets
`U[source, lane] = 1` for every source with any edge in that lane and `V[target, lane] = 1` for
every target with any edge in it.

Edge i→j then reads the sum over every lane in which i has some outgoing edge and j has some
incoming edge, not just its own lane. The result is exact only if no source/target pair shares a
second lane through other edges. When the measured error misses the tolerance, nothing falls back
to `fit_plane`: the engine logs a warning and runs with the wrong values.

`docs/weight_matrix_status.md` describes this path as exact.

- **Location:** `src/spikecorec/core/weight_matrix.cpp:893-912` (indicator rows),
  `:811-821` (no fallback when the error is too large).
- **Trigger:** two weights in one projection, or two projections between the same populations
  with different weights or delays. Concretely, `pynn_poisson_network` with the `expTargets`
  connection weights alternating 0.3 / 0.6, delays 2 ms.
- **Confirmed (reproduced):**
  - That model: 1250 of 2500 weights wrong (largest error 0.6) and 1250 delays wrong (every
    2 ms delay reads 4 ms); `measured_fit_error` is 1.74 and 1.0. The unmodified fixture reads
    back exactly.
  - Network `{{2,3},{2},{},{}}` with weights 1, 2, 2: `get(0,2)` is 3, declared 1.
  - 200 nodes × 12 edges, constant weight 0.5, delays 1–30 ticks: 2316 of 2400 delays and weights
    wrong, largest delay read 170 ticks.
- **Fixed (2026-10-14):** `fit_basis_from_connections` is removed at the user's request, with
  the rest of the old construction and refit fits. Construction now fits each plane on its own
  lanes with one of two candidate fits (`Weight_Matrix::fit_method`). The network
  `{{2,3},{2},{},{}}` with weights 1, 2, 2 now reads 1, 2, 2 (measured error 1.5e-4).

### 2. A fitted delay that rounds past the spike-history ring wraps to a 0–1 tick delay (high)

The kernel takes each delay as `round(load_edge(delay plane))`, which is the fitted value and may
exceed the declared maximum delay D by up to the tolerance. It clamps only from below
(`delay < 1`). The ring has exactly D + 1 rows.

A delay of D + 1 therefore reads `(tick - (D+1)) % (D+1) == tick % (D+1)`, the row the source wrote
earlier in the same tick: the arrival fires on the spike tick itself. D + 2 gives one tick, and so
on. The real arrival is never delivered, because its row has been overwritten by then.

Related case: when the lanes run out, allocation is greedy in plane order. With random weights at
fan-in 260, plane 0 took all 256 lanes[^lane-limit] and the delay plane got none. Every delay then read 0 and
was clamped to 1 tick; the only signal was the refit warning.

- **Location:** `src/spikecorec/nml/kernel_boilerplate.cpp:243-251` (Metal), `:568-576` (CUDA);
  ring size at `src/spikecorec/nml/dynamics.cpp:380-382`; lane allocation at
  `weight_matrix.cpp:1401-1411`.
- **Trigger:** any fit on the approximate path with delays above about `0.5 / fit_tolerance`
  ticks (over 500 ticks at the default 1e-3, over 50 at 1e-2). Concretely: 300 × 300 cells,
  fan-in 260, uniform weight, random delays of 1–800 ticks at dt 0.025 ms, tolerance 1e-2.
- **Confirmed (reproduced):** 26 of 78,000 edges have a kernel delay ≥ the ring length. Edge
  8→379, declared 799 ticks, fitted 800.687: synapse current starts on the source's spike tick.
  Control edge 8→301 (declared 789) arrives 789 ticks after the spike.
- **Fixed:** the kernel holds a delay read past the ring to its last row (`spike_history_length - 1`) on Metal and CUDA. When a plane runs out of lanes with fewer than an even share, at construction or at a refit, the lanes are divided evenly (`lane_share_per_plane`, 256 / plane count) and every plane is fitted again, with a warning.

### 3. Every `select` in a cell reads the whole network input (medium)

`selected_value` translates any `reduce="add"` select into `network_input`, whatever attachment
(`synapses`, `extras`), exposure (`i`, `g`) or input `destination` it names. A cell with two
selects receives the whole input once per select. `check_cell_inputs` only confirms that the
exposure exists, so the model is accepted.

- **Location:** `src/spikecorec/nml/dynamics.cpp:1648-1652`; acceptance at `:1384-1402`.
- **Trigger:** a cell with `iSyn select="synapses[*]/i"` and `iExtra select="extras[*]/i"`,
  `dv/dt = (iSyn + 3*iExtra)/C`, and a 1 nA, 1 ms pulse with `destination="synapses"`.
- **Confirmed (reproduced):** v reaches 40 mV; by the LEMS definition only `iSyn` sees the pulse
  and v reaches about 10 mV (jNeuroML: 9 mV).
- **Fixed:** `check_cell_inputs` refuses a cell type that reads its inputs through more than one select, and a synapse or input whose `destination` is not the attachment the select reads; edges and inputs now carry `destination`.

### 4. Unary minus binds tighter than `^`, and `^` is right-associative (medium)

In the LEMS expression parser a leading minus takes a single primary as its operand. `-x^2`
therefore becomes `pow(-x, 2)`, positive for even powers; `exp(-x^2)` grows instead of decaying.
`2^3^2` gives 512 where jLEMS gives 64. Both the kernel and the host evaluator (OnStart,
DerivedParameter) are affected. No standard-library expression uses these forms.

The same tokenizer also accepts malformed number literals: `1.2.3` reads as 1.2 and `1..5` as 1.0.

- **Location:** `src/spikecorec/nml/lems_parser.cpp:75-81` (unary operand), `:125`, `:142`
  (associativity); literals at `lems_tokenizer.cpp:93-96` and `dynamics.cpp:2584`.
- **Confirmed (reproduced):** with x = 3, `-x^2` gives 9 (jLEMS -9). The generated kernel line
  is `pow(-3.0f, 2.0f)`. With x = 100, the GPU and `evaluate_lems('-x ^ 2')` both give 10000;
  jNeuroML gives -10000.
- **Fixed:** a leading sign now applies to the whole power (`-x^2` is `-(x^2)`) on the host and the GPU, and a literal with two decimal points is refused. `^` is left-associative, as in jLEMS: `2^3^2` is 64.

### 5. A population of a cell type with no state variables shifts the next population (medium)

A population whose cell type declares no StateVariable does not advance the base index, so it
shares base index 0 with the next population. Paths into it resolve into the next population's
neurons, and connections are wired to the wrong cells.

- **Location:** `src/spikecorec/nml/parser.cpp:773` (base index not advanced at size 0), `:1073`
  (`neuron_index_of` skips such populations), `:663-665` (`resolve_path`).
- **Trigger:** populations `targets` (a normal cell) and `clocks` (a stateless type), with a
  connection `../clocks[1]` → `../targets[1]`.
- **Confirmed (reproduced):** both populations have base index 0. The edge becomes (2, 3)
  instead of (1, 3); the clocks fire 149 times and the target receives nothing.
- **Fixed:** `NML_Context::cell_slot_count` gives a cell type with no state variables one slot per cell, used by every layout calculation, so its population keeps its own base index and cells.

### 6. Calling `run()` twice runs the second pass on stale state (medium)

`run()` restarts `tick` at 0, but keeps everything else from the first run:
- `last_spiked`, `spike_history` and `network_inputs`;
- cell state and the sparse update buffer S;
- the random stream;
- the recorded data.

There is no reset. With `tick - last_spiked` negative, every neuron stays refractory until
`tick > last_spiked + t_ref`. The recorded times then overlap the first run, and the rate is
divided by one lifetime.

- **Location:** `src/spikecorec/core/engine.cpp:517-526`, `step_simulation` at `:464-508`.
- **Confirmed (reproduced):** on `LEMS_glif_family.xml` the first run gives 164 spikes. The second
  run gives one spike, at 0.4897 s (last spike plus the refractory period). The trace has 20,000
  rows with times 0–0.49995 twice.
- **Fixed:** `run()` steps only the ticks not yet stepped (so it resumes a stopped run) and throws once every tick has run.

### 7. A NaN or inf on one edge corrupts the whole plane, and the fit error reports 0 (medium)

One NaN target makes the plane's RMS scale NaN, so every residual and coefficient becomes NaN.
One inf target makes every finite residual 0, so the plane is fitted to 0.
`measure_fit_error` and `worst_fit_error` use `std::max`, which drops NaN, and an inf scale makes
every error 0. The plane therefore reports a perfect fit, and the construction warning never
fires.

- **Location:** `weight_matrix.cpp:1138` (scale), `:1152-1153` (residual and error scale),
  `:1446`, `:1454` (error maximum).
- **Trigger:** a declared value of NaN or inf, or a NaN delta in S followed by `refit()`.
- **Confirmed (reproduced):**
  - One NaN weight among 800 edges: all 800 read non-finite, measured error 0.
  - The same after a NaN delta and refit.
  - One inf weight: all 800 read exactly 0, measured error 0.
- **Fixed:** the fit scale is taken over finite values only, a non-finite target gets no weight, `measure_fit_error` reports such a plane as infinitely off and `worst_fit_error` counts a NaN as infinite; `accumulate_edge_delta` refuses a non-finite delta and the engine refuses a non-finite weight or synapse start.

### 8. The tick passed to `step_simulation` is not validated (medium)

The kernel uses `tick % 2` and `tick % length` for its two-row buffers and rings, and indexes
each spike train at `offset + tick`.
- **Past `lifetime`:** a spike train reads the next train's region, and the last train reads past
  the end of its buffer. Whatever it reads becomes kicks.
- **A negative tick:** the modulo goes negative. The kernel reads and zeroes memory before
  `network_inputs`, which is the tail of `cell_state` in the same slab.
- **Ticks that are not consecutive:** they break the row parity, so synaptic input is never
  drained.

The API reference asks only for increasing ticks.

- **Location:** `src/spikecorec/nml/dynamics.cpp:2285-2290` (unbounded train index; trains sized
  at `:598`), `kernel_boilerplate.cpp:188-193`, `:201`; no check in `engine.cpp:464`;
  `python/docs/API_REFERENCE.md:164`.
- **Confirmed (reproduced):**
  - Two `spikeArray` inputs stepped past a 500-tick lifetime: neuron 0 fires at 510, 520 and 530
    (train B's spikes), and neuron 1 fires from tick 524 onward on bytes past the buffer.
  - `step_simulation(-1)` on a 500-cell model leaves 242 cells at v = -70000.07.
  - pynn_synapses stepped on even ticks only: targets fire `[0, 0, 0, 0]` instead of
    `[0, 3, 3, 5]`.
- **Fixed:** `step_simulation` requires `tick == ticks_stepped` and `tick < lifetime`, and throws otherwise.

### 9. Zero-width synapse shares put edges on the wrong synapse (low)

When `synapse_proportions` splits a topology's edges across synapses, a share that rounds to zero
edges advances the synapse index by only one, so an edge lands on a synapse whose share is empty.
- **All-zero proportions:** NaN boundaries, and the index runs past the end of both arrays (an
  out-of-bounds read), followed by a misleading "names no synapse" error.
- **Negative proportions:** accepted.

- **Location:** `engine.cpp:420-441`; the `if` at `:440` needs to be a loop, and the proportions
  are not validated.
- **Confirmed (reproduced):** proportions `[0, 0, 1]` and `[0.05, 0.05, 0.9]` assign edge 0→1 to
  the first synapse (share 0); `[0, 0]` raises "the connection 0 -> 2 names no synapse";
  `[1, -1]` is accepted.
- **Fixed:** a share that rounds to no edges takes none (the `if` is a `while`), and proportions must be finite, not negative and not all zero.

### 10. `mean_firing_rate_hertz` divides by the document's lifetime, not the ticks stepped (low)

- **Location:** `engine.cpp:584`.
- **Confirmed (reproduced):** 1000 of 10,000 ticks stepped by hand, 20 spikes: true rate 80 Hz,
  reported 8 Hz.
- **Fixed:** the rate divides by the ticks stepped (`ticks_stepped`), not the lifetime.

### 11. Recorded traces are labelled one step early, and the initial state is never recorded (low)

`record_tick` stores the state after the kernel has advanced it to (tick + 1) · dt, but labels
the row `tick · dt`. The engine's own closed-form test uses `(tick + 1) · dt` for the same state.

- **Location:** `engine.cpp:552`.
- **Confirmed (reproduced):** x(0) = 0, x' = 1000/s, dt = 1 ms: the file's first rows are
  `0 1`, `0.001 2`; the state at 0 (x = 0) is absent.
- **Won't fix:** left as it is (user decision, 2026-10-13).

### 12. Float32 time and accumulated `tnext` make regular generators drift (low)

`t = f32(tick) / steps`, and a spikeGenerator advances `tnext = tnext + period` in f32 state.
Each addition rounds to the ulp of t. Because the period is not a multiple of that ulp, the error
is systematic: in [16, 32) s, a 1 ms period becomes 0.99945 ms. Past 2^24 ticks `f32(tick)` is no
longer exact, and past 256 s at dt 0.025 ms neighbouring ticks share one t.

- **Location:** `dynamics.cpp:105-112` (`time_of_ticks`); the generated `tnext` update.
- **Confirmed (reproduced):** 1 ms period, dt 0.1 ms, 30 s: 30,004 spikes instead of 29,999; the
  last lands at tick 299,992 instead of 300,040. A float32 model of the update predicts -162,
  -1225 and +10,966 ticks of drift after 60, 200 and 600 s at dt 0.025 ms and a 10 ms period.
- **Partly fixed:** `t` is the tick's time computed on the host in f64 and passed to the kernel as `simulation_time`: f32 on Metal, f64 on CUDA. The `tnext = tnext + period` drift remains, because state is f32.

### 13. Float atomic adds make runs differ in the last bit (low)

Several edges arriving at one target in the same tick are summed with `atomic_fetch_add` /
`atomicAdd` on `float`, in thread-completion order. The next-row inputs differ by one ulp from
run to run, and the difference carries into cell state. Spike trains matched in every run tried.
Exact comparison against golden runs relies on bitwise reproducibility, which this does not
guarantee.

- **Location:** `dynamics.cpp:2468`, `:2779-2787`.
- **Confirmed (reproduced):** `pynn_poisson_network`, two runs: 100,824 `network_inputs` slots
  differ over the run (first at tick 56: 2.8967465e-10 vs 2.8967462e-10), 2 final state floats
  differ, spike trains identical.
- **Won't fix:** user decision, 2026-10-13.

## C. Input documents accepted when they should be refused

### 14. Included NeuroML files are never validated against the schema (high)

`validate_lems_schema` checks only the file passed to the engine, and skips it when its root is
not `<neuroml>`. The schema does not allow `<Simulation>` inside `<neuroml>`, so every runnable
model has a LEMS root, and its included `.nml` files are never validated. Elements whose tag is
not a known ComponentType are then dropped with only a warning.

- **Location:** `engine.cpp:55`; `parser.cpp:967-985`, `:586`.
- **Confirmed (reproduced):**
  - `leakReversal="-50mv"` fails the schema when the `.nml` is validated directly, but passes
    through the LEMS wrapper, and the engine builds.
  - `<connectionWd>` (wrong case) gives a warning and a network with 0 edges.
- **Not fixed:** validating included `.nml` files against the XSD would refuse documents that declare or use their own ComponentTypes, which most test models and the shipped examples do; that needs your decision. A tag that matches a ComponentType but for case (`<connectionWd>`) is now refused, and unit typos are refused by finding 15's fix.

### 15. Unknown units and malformed quantities resolve to wrong values (high)

`split_quantity` and `resolve_quantity` handle bad input as follows:
- **Unknown suffix** (`mv`, `kHz`, `ns`, `msec`, `GOhm`): scales by 1.
- **Trailing text:** kept as part of the suffix, so it also scales by 1 (`"-60mV;"` gives -60,
  `"3 mV extra"` gives 3).
- **Text that is not a number:** becomes 0 (`"abc"`, `"nan"`); overflow (`"1e400"`) also becomes 0.
- **`"1.2.3mV"`:** becomes 1.2 mV.
- **No dimension check:** `tau="30"` is 30 s.

Together with finding 14, a unit typo silently changes values by a factor of 1000 or more,
including dt.

- **Location:** `src/spikecorec/core/units.cpp:70`, `:78-118`; `parser.cpp:1043-1055`.
- **Confirmed (reproduced):**
  - `leakReversal="-50mv"` resolves to -50 (volts); a cell that should fire never does.
  - `<Simulation length="100 msec" step="0.1 msec">` gives dt 0.1 s and 1000 ticks of 100 s.
- **Fixed:** `NML_Context::resolve_quantity` refuses text that is not one finite number with an optional unit, and units neither the document nor the standard units declare; `resolve_quantity_of_dimension` also checks the unit against the parameter's dimension, used for every parameter and for step, length, weight and delay.

### 16. `synapticConnection`, `electricalProjection` and `continuousProjection` are silently dropped (high)

The network loop reads only children that are instances of `projection`. These three schema-valid
forms become instances but contribute no edges, and nothing is refused. That includes the gap
junction and graded synapse types the engine does not support.

- **Location:** `parser.cpp:786-835` (filter at `:788`).
- **Confirmed (reproduced):** a schema-valid document with all three: `total_edge_count` 0, no
  synapse instances, the source fires 9 times and the targets never. With `synapticConnection`
  and an `alphaCurrentSynapse`, jNeuroML's target fires 7 times; the engine's target never fires.
- **Fixed:** a network child that is an `explicitConnection` (`synapticConnection`), `electricalProjection` or `continuousProjection` is refused, naming it.

### 17. An included file's `<Target>` overrides the main file's (medium)

The last `<Target>` processed wins, and included files are processed after the main file.

- **Location:** `parser.cpp:309-311`; queue order at `:204`, `:239-241`.
- **Confirmed (reproduced):** the main file targets `mainSimulation` (1000 steps), and an include
  declares its own `<Target>`. The engine runs `simulationFromInclude` (dt 0.5 ms, 10 ticks);
  jNeuroML runs `mainSimulation`.
- **Fixed:** the main file's `<Target>` wins; without one, the first an included file declares.

### 18. The generic `<Component type="...">` form is dropped (medium)

jLEMS accepts `<Component id="..." type="spike" time="..."/>`. The parser warns "Ignoring
<Component>" and drops it: inside a spikeArray every spike disappears without an error, and at
top level the later error says the component is not declared.

- **Location:** `parser.cpp:584-588`.
- **Confirmed (reproduced):** the native form schedules 2 spikes and the target fires twice; the
  generic form schedules 0. jNeuroML emits both spikes for both forms.
- **Fixed:** `<Component type="X">` is instantiated as type X; one naming an undeclared type is refused.

### 19. A cell `OnEvent` that reads a DerivedVariable generates a kernel that does not compile (low)

The OnEvent loop is placed before the derived variables, so it uses `derived_<name>` before the
name is declared. The input path refuses this case explicitly; the cell path does not, and the
run fails with a compiler error instead of a refusal.

- **Location:** `dynamics.cpp:1734-1736`, `:1811-1814`; the input path's refusal at
  `:2117-2123`.
- **Confirmed (reproduced):** `<DerivedVariable name="kick" value="jump * 2"/>` with
  `<OnEvent port="in"><StateAssignment variable="v" value="v + kick"/></OnEvent>` gives
  `use of undeclared identifier 'derived_kick'`.
- **Fixed:** a cell OnEvent, refractory exit or integrating-regime OnEntry that reads a derived variable is refused with an error naming it.

### 20. A cell id equal to one of its state variable names breaks variable paths (low)

The path resolver strips the part of the path that matches the cell id, so the variable is lost.

- **Location:** `parser.cpp:671-675`.
- **Confirmed (reproduced):** an izhikevich2007Cell with id `u`: `pop0[0]/u` resolves to v's slot
  (0), while a population whose cell has another id resolves `u` correctly (3).
- **Fixed:** the cell id is stripped only where the path repeats it (after an instance, or followed by `/variable`), so `pop0[0]/u` with cell id `u` reads `u`.

### 21. Missing or malformed required attributes are silently defaulted (low)

- A `connectionWD` with no `weight` or `delay` gets weight 1 and delay 0.
- A Simulation with no `length` runs 0 ticks.
- `size="2.6"` gives 3 neurons and `size="two"` gives 0.
- `target[1.9]` resolves to index 1.

- **Location:** `parser.cpp:640`, `:714-716`, `:757`, `:823-829`.
- **Confirmed (reproduced).**
- **Fixed:** a connectionWD without weight or delay, a Simulation without length, a size or index that is not a whole number (`2.6`, `two`, `[1.9]`) are refused.

### 22. Some errors give no element or path (low)

`target="pop[x]"` raises `ValueError: stoll: no conversion`, and `delay="-1ms"` raises
`ValueError: seconds must be >= 0`; neither names the element.

- **Location:** `parser.cpp:640`, `:649` (`stoll`), `:827` (`seconds_to_ticks`).
- **Confirmed (reproduced).**
- **Fixed:** a malformed index resolves to no neuron, so the caller's error names the path, and a negative delay is refused naming the connection.

## D. Recording and output files

### 23. Recording selections that do not resolve corrupt the output files (medium)

An OutputColumn that does not resolve, for example a DerivedVariable such as `iMemb` or a
synapse path, is kept at index -1 with a warning and then left out of the file. Every later
column shifts left, so a reader that indexes columns by declaration order reads the wrong
quantity. An EventOutputFile whose selections all fail to resolve treats the empty set as no
filter and writes every neuron's spikes.

- **Location:** `parser.cpp:927-934`; `engine.cpp:163-165`, `:650-658`, `:669-675`.
- **Confirmed (reproduced):**
  - Three declared columns produce two value columns.
  - `<EventSelection select="pop0[7]">` on a 3-cell population writes 12 rows covering all 3
    cells.
- **Fixed:** `Spike_Engine` refuses an OutputColumn or EventSelection that resolves to nothing it records, naming it; the parser still records index -1, as before.

### 24. `EventOutputFile` ignores `format`, the selection `id` and the path (low)

Rows are always `time  global-neuron-index`. The standard library defines the id column as the
EventSelection id, in `TIME_ID` or `ID_TIME` order. `fileName` is opened relative to the process
working directory, not the LEMS file, and the `path` attribute is ignored. The existing fixtures
hide this because their ids equal the global index.

- **Location:** `parser.cpp:900-923`; `engine.cpp:661`.
- **Confirmed (reproduced):** `format="ID_TIME"` with `<EventSelection id="7" select="pop0[1]">`.
  jNeuroML writes `7 1.0E-4`; the engine writes `0 1`, `0.0138 1`.
- **Fixed:** an EventOutputFile writes each EventSelection's id in TIME_ID or ID_TIME order (another format is refused), and relative file names are written beside the LEMS file, inside the element's `path`.

### 25. An `OutputFile` named `*.spire*` is written as plain text (medium)

The parser assigns the SPIRE formats from the file name, but `write_recordings` special-cases
only `SPIKE_EVENTS` and writes everything else as tab-separated text. The file cannot be read
back.

- **Location:** `parser.cpp:909-911`, `src/spikecorec/nml/utilities.cpp:29-32`;
  `engine.cpp:648-686`.
- **Confirmed (reproduced):** `traces.spire.gz` parses as `SPIREGZIP`, but the file starts with
  `0\t-0.0697500035\t...`, and `read_spire_recording` raises "gzread failed".
- **Fixed:** an OutputFile named `*.spire*` is written through `Simulation_Recorder`, one frame per recorded tick, one value per column.

### 26. The raw and xz sinks swallow write errors (medium)

`Raw_Sink` and `Xz_Sink` never check the stream's failure state, so a full disk silently
truncates the recording and `finish()` returns normally. The gzip and bz2 sinks do throw.

- **Location:** `src/spikecorec/core/recording.cpp:160-166` (raw), `:274`, `:297` (xz).
- **Confirmed (reproduced):** with the file size limited to 64 KiB, 400 KB of frames: raw (sync
  and async) and xz return normally with 65,536-byte files that read back as "Truncated
  recording"; gzip raises "gzwrite failed".
- **Fixed:** `Raw_Sink` and `Xz_Sink` check the stream after every write and on close, and throw.

### 27. A gzip recorder destroyed after a write error aborts the process (medium)

After a failed write, the recorder's destructor swallows the `finish()` error. The sink's
destructor then calls `close()`, which throws from a `noexcept` destructor, so the process ends
in `std::terminate` instead of raising an exception.
- **Reproduced:** gzip.
- **Same code pattern:** `~Bz2_Sink` and `~Xz_Sink`.

- **Location:** `recording.cpp:191`, `:207-213`; also `:260`, `:364`; the async writer leaves
  cleanup to the sink destructor at `:441-447`.
- **Confirmed (reproduced):** `Simulation_Recorder("x.spire.gz")`, frames until "gzwrite failed",
  then `del recorder` with the disk still full: exit status 134, "terminating due to uncaught
  exception of type std::runtime_error: gzclose failed". Sync and async both abort.
- **Fixed:** the gzip, xz and bz2 sink destructors log a failing `close()` instead of throwing.

### 28. The gzip compression level is not validated (low)

The level is formatted into the mode string as `"wb%d"`, and zlib reads it one digit at a time,
keeping the last. The xz and bz2 sinks clamp their levels.

- **Location:** `recording.cpp:184-185`.
- **Confirmed (reproduced):**
  - Level 10 behaves as 0 (stored): 800,142 bytes for an 800,004-byte payload.
  - Level 15 behaves as 5, and -1 as 1.
- **Fixed:** the gzip level is clamped to 0-9 before it is written into the mode string.

### 29. Python's `Spire_Writer` cannot be closed or flushed (low)

Only `write_frame` and `neuron_count` are bound, so nothing reaches the disk until the object is
garbage-collected.

- **Location:** `src/spikecorec/bindings/bindings.cpp:208-219`.
- **Confirmed (reproduced):** with the writer alive the file has 0 bytes and does not read; after
  `del` it has 52 bytes and reads back with shape (3, 4).
- **Fixed:** `Spire_Writer` has `flush()` and `close()`, bound in Python with context-manager support.

### 30. `write_spike_file` and `write_recordings` report open failures only in the log (low)

- **Location:** `engine.cpp:618-622`, `:642-646`.
- **Confirmed (reproduced):** `write_spike_file("/nonexistent_directory/spikes.dat")` returns
  without raising; only an `[error]` line is logged.
- **Fixed:** `write_spike_file` and `write_recordings` throw when a file cannot be opened or written.

## E. Crashes, memory safety and threading

### 31. Any thread that steps the engine crashes the process when it exits (high)

`run_function` calls `release()` on the command buffer and the compute encoder. Both come from
`commandBuffer()` and `computeCommandEncoder()`, which return autoreleased objects the caller does
not own. When a thread's autorelease pool drains at thread exit, it releases objects that are
already freed.

The main thread's pool never drains, so the test suite never sees this. Running the engine from
`threading.Thread`, a thread pool, a server or a C++ `std::thread` kills the process.

- **Location:** `src/spikecorec/core/backend.cpp:237-238` (objects obtained), `:257`, `:266`,
  `:272` (`release()` calls).
- **Confirmed (reproduced):**
  - Ten ticks of `LEMS_glif_family.xml` on a Python thread: "worker finished 10 ticks", then a
    segmentation fault, exit 139.
  - A C++ program with the same Metal calls on a `std::thread` exits cleanly without the two
    `release()` calls, and hits the Objective-C over-release trap with them.
- **Fixed:** `run_function`, `create_function` and `load_precompiled_function` drain their autoreleased objects in a local `NS::AutoreleasePool` instead of calling `release()`.

### 32. `run` and `step_simulation` after `shutdown()` crash instead of raising (medium)

`shutdown()` releases the Metal buffers but keeps their handles. The bindings guard the buffer
properties with `require_alive`, but not `run` or `step_simulation`, which immediately read the
freed `random_values` buffer.

The `Weight_Matrix` that Python can still hold after shutdown has a null backend, so
`resize_delta_capacity` dereferences null.

- **Location:** `bindings.cpp:972-977`; `engine.cpp:466`, `:700`; `weight_matrix.cpp:583`.
- **Confirmed (reproduced):**
  - `engine.shutdown(); engine.step_simulation(1)`: exit 139.
  - `w = engine.weights; engine.shutdown(); w.resize_delta_capacity(16)`: exit 139.
- **Fixed:** `run` and `step_simulation` throw after `shutdown()`, and a `Weight_Matrix` without storage throws from `resize_delta_capacity`, `resize_basis` and `grow_basis`.

### 33. Reading results from another thread during `run()` returns torn data (medium)

`run()` releases the GIL, but the engine has no lock. The result getters copy from
`recorded_traces`, `recorded_spikes` and similar vectors while `record_tick` appends to them and
reallocates. The getters also size the copy from `recorded_trace_times`, which grows before
`recorded_traces`.

- **Location:** `bindings.cpp:972-977` (GIL released), `:1021-1027` (unlocked copies);
  `engine.cpp:535-536`, `:552-554` (appends).
- **Confirmed (reproduced):** a second thread polling `engine.recorded_traces` during `run()`
  got torn snapshots in 3 of 3 attempts (10–12 per run). One snapshot read
  `[-1.58e+05, 2.16e+12, 8.9e-41]` where the recorded value was -0.06975.
- **Fixed:** the recorded results are appended and copied under `Spike_Engine::results_mutex`.

### 34. `Weight_Matrix::load_from_disk` accepts a file from another topology and writes out of bounds (medium)

The loader checks the node and plane counts, but neither `total_edge_count` nor the range of the
S ordinals. S entries saved from a larger network then index past `total_edge_count` on every
read, refit or neighbour-weights call. It is reachable from Python through
`engine.weights.load_from_disk`.

- **Location:** `weight_matrix.cpp:1636` (checks); out-of-bounds write at `:1525`, same pattern at
  `:958`.
- **Confirmed (reproduced):** save from `{{1,2},{2},{3},{}}` with an S entry on ordinal 3, then
  load into `{{},{},{},{0}}`. Under AddressSanitizer: `heap-buffer-overflow` in
  `neighbor_weights_for_matrix` (`:1525`). Without it, the load is accepted silently.
- **Fixed:** the save format records the edge count (new tag); a load checks node, edge and plane counts, the rank, every update's ordinal and truncation before it changes the matrix.

### 35. Using the parse context after a failed parse segfaults (low)

`parse()` leaves a half-built context behind when it throws, and later queries dereference it.

- **Location:** `parser.cpp:1113` (`get_cell_state_size`), `:663` (`resolve_path`).
- **Confirmed (reproduced):** a population naming an undeclared cell makes `parse` raise; a
  following `get_cell_state_size()` exits with status 139.
- **Fixed:** `parse()` resets the context when it throws.

### 36. `NML_Context.get_population_size(None)` segfaults (low)

pybind11 accepts `None` for the pointer argument, and it is dereferenced without a check.

- **Location:** `bindings.cpp:522` → `parser.cpp:1084`.
- **Confirmed (traced):** a crash report shows the fault inside
  `NML_Context::get_population_size` (address 0x28), called straight from the pybind11
  dispatcher.
- **Fixed:** `get_population_size(nullptr)` throws `Invalid_Argument` (`ValueError` in Python).

### 37. A connection target outside the network writes out of bounds in the constructor (low)

The constructor increments `in_degree[target]` without a bounds check. The k²-tree then drops
the edge silently. Only reachable through the C++ API; the engine's adjacency path validates
targets first.

- **Location:** `weight_matrix.cpp:172`; the drop at `k2tree.cpp:191-193`.
- **Confirmed (reproduced):** `Weight_Matrix(backend, {{1, 9}, {0}, {}})` gives a
  `heap-buffer-overflow` under AddressSanitizer at `:172`; without it, it constructs with 2 edges.
- **Fixed:** the constructor refuses a target outside the network.

### 38. A k²-tree superblock above 2048 words overflows the u16 subblock table (low)

The within-superblock popcount is stored as `u16`. Above 2048 words it wraps, and neighbour
queries return wrong rows. A superblock size of 0 divides by zero. Neither is checked.

- **Location:** `src/spikecorec/core/k2tree.cpp:262`; no check in `from_adjacency_list` or
  `from_edges`.
- **Confirmed (reproduced):** 10k nodes, 300k edges:

  | Superblock size (words) | Wrong rows of 10,000 |
  |---|---|
  | 1024, 2048 | 0 |
  | 4096 | 5120 |
  | 16384 | 10,000 |
- **Fixed:** both factories reject a superblock size outside 1-2048 (`K2_Tree::MAX_SUPERBLOCK_SIZE_WORDS`).

### 39. Ctrl-C is ignored for the whole of `run()` (low)

The loop never calls `PyErr_CheckSignals`, so `KeyboardInterrupt` arrives only after the last
tick.

- **Location:** `engine.cpp:520`; `bindings.cpp:972-974`.
- **Confirmed (reproduced):** SIGINT 0.5 s into a 20,000-tick run raised `KeyboardInterrupt` only
  after 13.7 s.
- **Fixed:** `run()` takes an `after_each_tick` callback; the Python binding uses it to call back and to raise a pending Ctrl-C about every 0.1 s.

## F. Smaller API and diagnostic defects

### 40. `K2_Tree::trace` reads an uninitialized `level_bit_offset` (low)

`s32 level_bit_offset, rank_inclusive = 0;` initializes only the second variable. `trace` is a
debug query, but it is exposed to Python and documented as giving `adjacent()`'s answer.

- **Location:** `k2tree.cpp:818`, used at `:826`.
- **Confirmed (reproduced):** 12,004 of 312,624 pairs disagree with `adjacent()`; with the
  variable initialized to 0, none do.
- **Fixed:** `level_bit_offset` starts at 0.

### 41. Reads of a pair that is not an edge return the basis value (low)

- `get`, `get_for_matrix` and `get_edge_delay_ticks` on a non-edge return whatever the basis
  reconstructs, while `edge_ordinal` returns nullopt and `get_edge_synapse_prototype` returns -1
  for the same pair.
- An explicit `max_neighbor_count` below the real out-degree silently truncates rows and drops
  edges.

- **Location:** `weight_matrix.cpp:527`; `:163-168` with `:202`.
- **Confirmed (reproduced):**
  - Non-edge 0→3: `get` is 0.5, delay 3 ticks, `edge_ordinal` empty, prototype -1.
  - `max_neighbor_count=2` on a row of 3 gives 2 edges.
- **Fixed:** `get`, `get_for_matrix` and `get_edge_delay_ticks` read 0 for a pair that is not an edge, and an explicit `max_neighbor_count` below the longest row throws.

### 42. `neighbor_weight_stats` reports a wrong standard deviation (low)

The sums are `f32` and the variance is computed as E[x²] − mean², which cancels for tight
distributions. The RMS it also returns is accurate enough.

- **Location:** `weight_matrix.cpp:1535-1553`.
- **Confirmed (reproduced):** weights 1.0 ± U(1e-3):
  - 10,240 edges: 2.39e-3 (double precision gives 5.74e-4).
  - 100,000 edges: 1.66e-3 (5.78e-4).
- **Fixed:** the sums are f64 and the variance is taken about the mean in a second pass.

### 43. `square_torus(2)` and `small_world_torus(2)` emit duplicate edges; `square_torus(1)` emits none (low)

With a side of 2, left and right (and up and down) are the same node. `topologies.h` documents
exactly 4 children.

- **Location:** `src/spikecorec/core/topologies.cpp:43`; contract in `topologies.h:14-15`.
- **Confirmed (reproduced):**
  - `square_torus(2)` rows look like `[1, 1, 2, 2]`.
  - `small_world_torus(2, 1)` gives 3 distinct neighbours instead of 5.
  - `square_torus(1)` gives `[[]]`.
- **Fixed:** each torus neighbour is kept once (2 at a side of 2) and shortcuts are counted from the torus degree; a side of 1 still gives no edges, as an existing test asserts.

## G. Found during the comment check

### 44. A `populationList` whose instance ids are not 0..n-1 is refused (medium)

`resolve_path` uses an `<instance>`'s id as its position in the population, then refuses any id
at or past the population size. NeuroML allows any instance ids, so a valid document with ids
1 and 2 fails to load.

- **Location:** `src/spikecorec/nml/parser.cpp:635`, size check at `:645`.
- **Confirmed (reproduced):** a two-cell `populationList` with instance ids 1 and 2 and one
  connection `../pop/1/cell → ../pop/2/cell` raises "has an endpoint that resolves to no neuron".
- **Fixed:** an instance is addressed by its position among the list's instances (`instance_positions`), whatever its id.

### 45. An input `OnEvent` that calls `random()` is refused, and `random_slot_` names are captured (low)

- `on_event_body` checks the names of the already-translated body. A `random()` call there has
  become `random_values[random_slot_N]`, and `random_values` is a reserved input name, so the
  part is refused with "uses the name 'random_values', which the generated kernel reserves".
- `check_part_names` does not refuse LEMS names that start with `random_slot_`, so
  `resolve_part_names` silently turns such a name into a random-value index. The cell path
  refuses them.

- **Location:** `src/spikecorec/nml/dynamics.cpp:2053-2060`, `:2092-2101`; the translation of
  `random()` at `:2575-2578`; the cell path's check at `:159` and `:1786`.
- **Confirmed (traced).**
- **Fixed:** the input OnEvent's names are checked before `random()` is translated, and `check_part_names` refuses `random_slot_` names.

### 46. `Recording_Selection::variable_name` is never set (low)

The parser fills `quantity_path`, `event_port` and `neuron_index` only, so `variable_name` is
always empty, although the header and the Python property describe it as the trailing variable.

- **Location:** `parser.cpp:897-916`; `src/spikecorec/core/recording.h:43`;
  `src/spikecorec/bindings/bindings.cpp:168`.
- **Confirmed (traced).**
- **Fixed:** the parser sets a column's `variable_name` (and each selection's `selection_id`).

### 47. On CUDA, every neighbour-weight and adjacency query retries a kernel load and logs at critical level (low)

`Weight_Matrix::ensure_function` and `K2_Tree::adjacent_batch` cache the function only when
`load_precompiled_function` succeeds. On CUDA it always fails and logs at critical level, so every
call logs again before the host path runs.

- **Location:** `src/spikecorec/core/weight_matrix.cpp:456-466`, `src/spikecorec/core/k2tree.cpp:733-742`,
  `src/spikecorec/core/backend.cpp:433-437`.
- **Confirmed (traced).** CUDA cannot be built on this machine.
- **Fixed:** `Weight_Matrix::ensure_function` and `K2_Tree::adjacent_batch` do not retry a failed load, and the CUDA miss logs at debug. Built only on Metal.
- **Later (2026-10-15):** `K2_Tree::adjacent_batch` and `k2tree_adjacent_batch_kernel` were removed. Nothing in the engine called them, and their query staging kept 576 KB resident in every k²-tree.

### 48. `spikecorec.h` cannot compile (low)

It includes `types.h`, `kernels.h` and `kernels.cuh` by paths that do not exist, and defines
non-inline functions in a header. Nothing includes it.

- **Location:** `src/spikecorec/spikecorec.h`.
- **Confirmed (traced).**
- **Fixed:** `spikecorec.h` includes the real public headers and its helpers are `inline`; a test includes it.

### 49. Nine of the eleven kernels in `kernels.metal` are never loaded (low)

Host code loads only `neighbor_weights_kernel` and `k2tree_adjacent_batch_kernel`. `scale_uv`,
`add_network_input`, `weight_update`, `k2tree_get_neighbors_batch`, `step`,
`step_no_active_optimization`, `decay_all_neurons`, `merge_input_neurons` and
`reservoir_features` are compiled into `default.metallib` but never run.

- **Location:** `src/spikecorec/metal/kernels.metal`; the only loads are at
  `weight_matrix.cpp:1462` and `k2tree.cpp:738`.
- **Confirmed (traced).**
- **Fixed:** the nine kernels are commented out in `kernels.metal`, with what only they used:
  `MAX_RANK_FLOAT4_STRIDE`, `MAX_SIMD_GROUPS`, `apply_decay` and `step_apply_hebbian_update`.
  Other unused code commented out with them: `Recorded_Value`, `Recording_Config::recored_data`
  and `add_output` in `recording.h`, and the uncalled copy of `collect_schema_validation_error`
  in `utilities.cpp`.

### 50. A bare `pytest` collects no tests (low)

`pyproject.toml` sets `testpaths = ["python/tests"]`, which holds only a stale `__pycache__`. The
tests are in `tests/python`; `make test-python` works because it passes the path itself.

- **Location:** `pyproject.toml:24`.
- **Confirmed (reproduced):** `.venv/bin/python -m pytest --collect-only` reports "no tests
  collected".
- **Fixed:** `testpaths` is `tests/python`.

### 51. The demo notebook and `notebooks/spike_validation.py` call APIs that no longer exist (low)

- `examples/notebooks/demo.ipynb` constructs `spc.SpikeEngine`; the class is `Spike_Engine`. Its
  other engine calls exist.
- `notebooks/spike_validation.py` uses the removed reservoir API: `Spike_Engine(network, shape,
  ...)`, `set_input_neurons`, `start_static_record` and `get_last_spiked`.

- **Confirmed (reproduced):** none of these names exist on the built module.
- **Fixed:** `notebooks/spike_validation.py` uses the current API. `examples/notebooks/demo.ipynb` is gitignored and not in this tree: regenerate it from `01_tutorial_getting_started.py`, which uses `Spike_Engine`.

## Where to start

- **Findings 1 and 2.** Ordinary networks run with wrong weights and delays and only a warning.
  For 1, either restrict the indicator basis to combinations whose sources and targets do not
  overlap, or fall back to `fit_plane` when the measured error misses the tolerance. For 2, clamp
  the kernel delay to `[1, spike_history_length - 1]`, or size the ring with a margin for the fit
  error.
- **Findings 14, 15 and 16.** Typos and unsupported connection forms run silently instead of
  failing. Validating included files, rejecting unknown units, and refusing unread network
  children would turn them into errors.
- **Finding 31.** Any threaded use of the engine crashes. The fix is to drop the two `release()`
  calls, or to retain the objects explicitly.
- **Findings 6 and 8.** `run()` twice and unchecked ticks; a reset, or a refusal, closes both.

[^lane-limit]: The 256-lane limit is `MAX_RANK_FLOAT4_STRIDE` (64 float4s) in `weight_matrix.h`.
    It came from `kernels.metal`, where it sized fixed-size arrays holding a whole U row per
    thread; Metal needs those sizes at compile time. Only the unused kernels had such arrays
    (finding 49, now commented out). The generated `master_step`, `neighbor_weights_kernel` and
    `k2tree_adjacent_batch_kernel` loop over `rank_float4_stride` with no fixed array, so nothing
    that runs needs the limit. Raising it costs memory (U and V each grow by node count × lanes ×
    4 bytes), tick time and fit time.
