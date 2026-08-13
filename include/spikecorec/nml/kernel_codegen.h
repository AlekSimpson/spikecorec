#pragma once

#include "spikecorec/core/types.h"
#include "spikecorec/nml/nml.h"

namespace spikecorec::nml {

// Turns a parsed model's stage-tagged dynamics into GPU kernel source. There is no
// intermediate representation: a CellTypeSpecification's Vector<DynamicsInstruction> is
// read directly and device source comes out the other side.
//
// Backend is chosen at generation time from SPIKECOREC_METAL / SPIKECOREC_CUDA, the same
// switch the rest of the engine compiles under.
//
// ── Memory layout this generator assumes ─────────────────────────────────────
// One flat `cell_state` array sectioned by cell TYPE: every cell of type 0 first, then
// every cell of type 1, and so on. Within a section a cell occupies
// cell_state_slot_count(type) floats -- its StateVariables, plus ONE appended slot holding
// the current regime index when the type declares any Regime (see CellRegimeLayout below).
// `cell_parameters` follows the identical scheme with parameter_names.size() floats per
// cell.
//
// Global neuron indices are never renumbered. The engine precomputes each neuron's base
// offsets on the CPU, so a kernel thread resolves its slot with one load rather than by
// searching type boundaries:
//     cell_state_base[neuron_index], cell_parameter_base[neuron_index],
//     cell_type_index[neuron_index]
//
// ── Shape of the generated source ────────────────────────────────────────────
// One device function per cell type plus one master kernel that switches on the neuron's
// type index and calls the matching function. Per-type functions keep the switch flat and
// let the shader compiler inline each body independently.

// Maps an NML identifier onto the C expression that reads it.
//
// The FIRST definition of a name wins; later ones are ignored. That is what makes the
// insertion order the resolution precedence, so callers insert in precedence order:
//     state variables -> parameters -> derived locals -> global constants -> built-ins
// LEMS feeds several tags into one `variables` namespace (a Parameter and a StateVariable
// named "tau" collide), so a collision is a real possibility rather than a hypothetical,
// and silently letting the last writer win would change which storage the kernel reads.
struct SymbolTable {
    UnorderedMap<String, String> identifier_expressions;

    // Named in the "unknown identifier" diagnostic, so the error points at the
    // ComponentType the expression came from rather than just the expression.
    String component_type_name;

    // Records `identifier` as reading `read_expression`, unless it is already defined.
    //
    // `read_expression` must be a primary expression -- an atom, an indexing, a call, or
    // something already parenthesised -- because it is substituted without being wrapped.
    // Everything this module generates ("cell_state[state_base + 3]", a local's name, a
    // float literal, "(dt * (float)tick)") satisfies that.
    void define(const String &identifier, const String &read_expression);

    bool contains(const String &identifier) const;

    // Throws naming the identifier and the ComponentType when the name is unknown. An
    // unresolved identifier is never passed through verbatim: the shader compiler would
    // either reject it far from its source or, worse, bind it to something real.
    const String &read_expression_for(const String &identifier) const;
};

// ── Regimes ──────────────────────────────────────────────────────────────────
//
// A LEMS <Regime> is a mode the cell is in: while it is active, only the TimeDerivatives and
// OnConditions declared inside it apply, and a <Transition> moves the cell to another one.
// GLIF2-5 all use the same two-regime shape, where `v` has a TimeDerivative in `integrating`
// and NONE in `refractory` -- that absence IS the refractory period, since a variable with no
// derivative in the active regime does not move.
//
// There is no regime machinery in the generated kernel. Regimes are a COMPILE-TIME
// construct: what reaches the GPU is an ordinary if/else chain over one integer, which is
// exactly what a regime means.
//
//  1. One integer per neuron holds the active regime index. It lives in the cell's own state
//     chunk, in a slot appended after every StateVariable -- see cell_state_slot_count. The
//     master kernel's argument table is full (31 of Metal's 31), so a buffer of its own is
//     not available; and it is per-neuron state, which is what that chunk is for. Stored as
//     a float like everything else in cell_state: a regime index is a small integer, which
//     f32 represents exactly, so comparing it against a literal is exact rather than
//     approximate.
//  2. Per state variable, an if / else-if / else chain on that integer. A regime that
//     declares no TimeDerivative for the variable emits NOTHING in its branch, so the
//     variable keeps the value it entered the tick with. Freezing is the absence of code,
//     not a special case.
//  3. A <Transition regime="X"/> compiles to a store of X's index into that slot, and X's
//     <OnEntry> StateAssignments are inlined at the transition site. Nothing anywhere asks
//     "did I enter a regime this tick" -- entry is the transition, so its body is emitted
//     where the transition is.
//
// An OnCondition declared inside a regime is guarded by that regime being active as well as
// by its own test. Everything outside any regime -- GLIF3's asc1/asc2 TimeDerivatives, its
// OnStart -- stays unconditional and keeps running in every regime.
//
// The regime index is read ONCE, into a local, before any of the above. Every guard reads
// that local, so a transition taken this tick is observed from the NEXT tick on, and two
// regimes' OnConditions cannot both fire because the first one moved the cell into the
// second's regime.
struct CellRegimeLayout {
    // Regime names in declaration order; the position is the regime index the generated
    // source compares against. Empty when the ComponentType declares no Regime at all, which
    // is what every other query here keys off.
    Vector<String> regime_names;

    // The regime marked initial="true", which is where the initialize kernel starts every
    // neuron of the type. -1 only when there are no regimes -- a type WITH regimes and none
    // marked initial is refused rather than defaulted, because "starts refractory" and
    // "starts integrating" are different models and nothing in the document says which.
    s64 initial_regime_index = -1;

    // Slot of the regime index inside this type's cell_state chunk: one past the last
    // StateVariable, so every real state variable keeps the slot it already had.
    usize regime_state_slot = 0;

    bool has_regimes() const { return !regime_names.empty(); }

    // -1 when no regime carries the name.
    s64 index_of(const String &regime_name) const;
};

// Resolves one cell type's Regime declarations. Throws naming the ComponentType on a
// duplicate regime name, a nested Regime, an unrecognised initial= value, more than one
// initial regime, or regimes with none marked initial.
//
// Public because the ENGINE needs the same answer the generator does: it sizes cell_state
// and computes every neuron's cell_state_base, and the appended regime slot has to be part
// of both or the two disagree about where a neuron's state begins.
CellRegimeLayout resolve_cell_regimes(const CellTypeSpecification &cell_type);

// How many floats one neuron of `cell_type` occupies in `cell_state`: its StateVariables,
// plus one appended slot when the type declares any Regime. Derived here, once, rather than
// recomputed on each side.
usize cell_state_slot_count(const CellTypeSpecification &cell_type);

// Generated source plus what the engine needs to bind and launch it.
//
// `argument_names` makes the argument order data rather than a comment both sides have to
// remember. The engine binds positionally against it, so the two cannot silently drift.
// Both generators currently emit this order, and both emit the same list:
//
//   0  cell_state               device float *
//   1  cell_state_residual      device float *        -- compensated accumulation, see below
//   2  cell_parameters          device const float *
//   3  network_inputs           device float *        -- the delay ring, see below
//   4  last_spiked              device long *         -- WRITE ONLY, see below
//   5  spike_flags              device int *
//   6  cell_state_base          device const int *
//   7  cell_parameter_base      device const int *
//   8  cell_type_index          device const int *
//   9  neuron_count             constant int &
//   10 dt                       constant float &
//   11 tick                     constant long &
//   12 internal_node_words      device const uint *
//   13 leaf_node_words          device const uint *
//   14 rank_superblock_table    device const uint *
//   15 rank_subblock_table      device const ushort *
//   16 U_matrix                 device const float *
//   17 V_matrix                 device const float *
//   18 edge_weight_coefficients device const float *  -- the weight matrix's Ck
//   19 edge_weight_deltas       device const float *  -- the weight matrix's Sk
//   20 edge_attributes          device int *          -- three per-edge int planes, see below
//   21 edge_synapse_state       device float *        -- per-edge synapse state, see below
//   22 k2tree_shape             device const int *    -- five k^2-tree shape scalars, see below
//   23 rank_float4_stride       constant long &
//   24 constant_weight          constant float &
//   25 constant_weight_enabled  constant int &
//   26 max_neighbor_count       constant int &
//   27 ring_depth               constant int &
//
// Arguments 12-22 are the adjacency and per-edge state stage 6 (Propagate) walks; 23-27
// describe their shape. The initialize kernel takes the identical list so the engine binds
// one argument set for both entry points.
//
// That is 28 of Metal's 31 per-stage buffer argument slots. Two consolidations keep it
// there, both of the shape the ring already uses -- one buffer, baked plane offsets:
//
//   edge_attributes packs three [neuron_count * max_neighbor_count] int planes, in order:
//     plane 0  each edge's delay in whole ticks
//     plane 1  which lowered synapse program the edge runs, or -1 for none
//     plane 2  the tick each edge's synapse state was last advanced through, or -1 before
//              it has ever been advanced (lazy synapse updates only -- see below)
//
//   k2tree_shape packs the five k^2-tree shape scalars, in order: branching_factor,
//   superblock_size_words, padded_node_count, tree_height, internal_bit_count.
//
// last_spiked is WRITTEN and never read: the generated source stores the current tick into
// it on every emission (emit_spike) and no emitted expression loads it. That is what lets
// the engine seed it with SpikeEngine::NEURON_NEVER_SPIKED_TICK (-1) as a "has never fired"
// sentinel -- a neuron that has not spiked still holds the seed, because nothing on device
// overwrites it with anything but a real, non-negative spike tick. Anything that later
// emits a READ of last_spiked owes that sentinel a `< 0` guard, or it will compute an
// interval against tick -1.
//
// constant_weight_enabled is a flag rather than `constant_weight != 0`: a model may set a
// constant weight of exactly zero, and a magnitude cannot double as a mode.
//
// U_matrix / V_matrix are the host's float4 buffers declared as plain floats -- identical
// bytes, and scalar lanes make the on-device reconstruction the same arithmetic
// WeightMatrix performs on the CPU. `rank_float4_stride` still counts float4 elements, so a
// row is four times that many lanes.
struct GeneratedKernel {
    String source;
    String function_name;
    Vector<String> argument_names;
};

// Per-tick dynamics: Deliver, then Integrate, then Detect, then the Reset and Emit bodies
// each OnCondition gates, then any ungated Emit and Reset, then Propagate. That is the
// engine's documented stage order (3 Detect, 4 Emit, 5 Reset), and it is the emission order
// too: an ungated reset emitted ahead of the Detect blocks would be clobbered by the same
// tick's integrate result instead of overriding it.
//
// Every TimeDerivative is evaluated against the state as it stood at entry and written
// back only once all of them have been computed, so two variables that reference each
// other integrate consistently instead of one seeing the other's updated value.
//
// ── Compensated accumulation ─────────────────────────────────────────────────
// Forward Euler is a running sum, and an f32 running sum loses the low bits of every
// increment that is small beside the total. Fifty additions of a 0.1 ms dt reach
// 4.9999989569e-03 where f32(5 ms) is 4.9999998882e-03, so a `refractoryTimeElapsed .geq.
// t_ref` fires one tick late and the refractory period is a tick too long -- for essentially
// every t_ref at that dt, and a 20% error at a 0.5 ms one.
//
// Metal has no double precision, so each CELL state variable carries an f32 residual instead:
// the part of the last increment that did not fit, folded into the next one. That is Kahan
// compensated summation, and it applies to every state variable of every ComponentType rather
// than to any particular comparison. The residuals live in `cell_state_residual`, laid out
// exactly parallel to `cell_state`, and are discarded wherever a variable is ASSIGNED
// (OnStart, an OnCondition's reset, a regime's OnEntry) because they describe a sum that no
// longer exists.
//
// Two things worth knowing about it:
//  - The emitted step routes the sum through two `volatile` locals. The residual is an
//    exact-arithmetic identity, and Metal compiles with fast math on by default, so without
//    them the optimiser folds it to a literal zero -- and with only one of them it reassociates
//    into a form that evaluates to zero at run time. Both were checked in the emitted AIR.
//  - It removes the error that GROWS with the number of steps. What it cannot remove is that
//    f32(dt) and f32(t_ref) are separately rounded: at t_ref = 5 ms and dt = 0.1 ms they agree
//    and the crossing lands on the reference's tick, but at t_ref = 1 ms the exact product
//    10 * f32(0.1 ms) is still 0.63 ulp below f32(1 ms) and the crossing is a tick late.
//    Closing that would need the accumulator to be a true two-float pair and dt to be carried
//    as two floats.
//
// Per-edge synapse state carries no residual: it is one compressed plane per state variable
// across every edge, so residuals would double the largest allocation the engine makes, and
// what they buy is a threshold crossed on the right tick -- a property of a cell's
// OnConditions, which a synapse declares none of.
//
// ── What Detect, Emit and Reset observe: the POST-INTEGRATE state ─────────────
// Stages 3 to 5 run after Integrate, so everything they read is this tick's integrated
// state -- the cell_state reads inside an OnCondition's test, the cell_state reads on a
// StateAssignment's right-hand side, AND the DerivedVariable locals either of them names.
// DerivedVariables are therefore evaluated twice: once before the Euler step, feeding the
// TimeDerivative right-hand sides, and again after the write-back, so a threshold or reset
// value that is itself a DerivedVariable of a state variable is compared against the same
// state the other half of the comparison came from.
//
// LEMS evaluates its derived variables once per tick, before the conditionals, so a
// consistently PRE-integrate reading would also be defensible -- but it would mean
// reverting the conditionals' own cell_state reads to pre-integrate temporaries too, which
// would notice a threshold crossing one tick late and reset a v that has already moved past
// the threshold. What is not defensible is the mixture: half a comparison pre-integrate and
// half post. Post-integrate is the half both are made to agree on, because it is what the
// engine's stage order and the existing reset semantics already imply.
//
// A handler's StateAssignments are SIMULTANEOUS, as in LEMS: every right-hand side is
// evaluated before anything is written back, so "v = u" and "u = v" in one OnCondition swap
// rather than collapsing onto u's value. This is the same temporaries treatment the
// TimeDerivative path gets, and it applies to OnStart bodies as well.
//
// A DerivedVariable written as a `select=` path over the attached synapses
// ("synapses[*]/i" on iafCell, "synapses[*]/I" on izhikevichCell) lowers to a PLAIN load of
// this tick's `network_inputs` ring row, and binds under its own name like any other
// DerivedVariable. Every other path -- "ionChannel/g", "populations[*]/i",
// "concentrationModels[species='ca']/concentration" -- reaches into a child structure with
// no engine buffer behind it and throws naming the path. Paths and arithmetic arrive in the
// same field, so they are separated by shape: see select_path_head_name in the
// implementation for why "iMemb/C" stays a division.
//
// ── Stage 6, Propagate ───────────────────────────────────────────────────────
// Every cell device function ends with the same generated epilogue: if this neuron raised
// its spike flag this tick, walk its row of the k^2-tree and add each edge's weight into
// the target's slot of `network_inputs`. It is emitted rather than dispatched because the
// supporting engine infrastructure is identical for every cell type, and the engine must
// therefore NOT dispatch a propagation kernel of its own.
//
// `network_inputs` is a delay ring: one flat allocation of ring_depth * neuron_count floats
// indexed as [row][neuron]. An arrival lands in row (tick + edge delay) % ring_depth and a
// cell reads row tick % ring_depth. Arrivals are atomic -- many sources converge on one
// target -- but the READ is a plain load, and it does not clear anything: the row is zeroed
// as a whole row after the dispatch, by generate_ring_row_clear_kernel below.
//
// `ring_depth` exceeds the model's largest per-edge delay -- the engine computes it -- so an
// arrival can never land in the row being read this tick.
//
// ── Synapses: per-edge state, impulsive delivery ─────────────────────────────
// A synapse's StateVariables are PER EDGE, and they live in the WeightMatrix's compressed
// per-edge variable family (one registered matrix per state variable, its Ck pinned to
// all-zero so the stored plane is the whole value -- see
// WeightMatrix::per_edge_variable_values). `edge_synapse_state` is that block, bound
// straight through: variable v's plane starts at v * neuron_count * max_neighbor_count, and
// inside a plane an edge sits at source_neuron * max_neighbor_count + its neighbour slot,
// the same slot convention every other per-edge array uses.
//
// Lowered synapse program p owns the state_variable_count planes starting at its
// state_variable_offset; both offsets are BAKED into the generated source, as are the
// prototype's own parameter values, which are constants of the prototype rather than of any
// edge. Nothing tracks a prototype at run time: `edge_attributes` plane 1 names a lowered
// program, and that is the whole of it.
//
// What is instantaneous is the DELIVERY, not the dynamics. On a presynaptic spike, and
// inside the source neuron's own propagation walk, each outgoing edge:
//
//   1. advances its own state to this tick (see the two update policies below),
//   2. snapshots the state variables the arrival handler is about to move,
//   3. runs the OnEvent handler with `weight` bound to that EDGE's weight, and
//   4. writes the whole of the charge that event's response will deliver into the ONE ring
//      slot at (tick + that edge's delay), as the current that delivers it in one tick.
//
// An edge whose projection names no synapse delivers its raw weight instead, which is what
// network_inputs meant before there were synapse dynamics at all.
//
// ── What "the whole of the charge" means, and why not a sample of `i` ────────
// Write Q(x) for the total charge a synapse left at state x and never touched again would
// still deliver: the sum of i(x_n) * dt over the ticks that follow. An event moves the state
// from x to x' = handler(x, weight), so the charge THAT event adds is Q(x') - Q(x). The tail
// Q(x) was already delivered in full by whichever earlier event produced it, and delivering
// any of it again would inject more total charge than the synapse ever carries.
//
// Sampling `i` at the instant of the spike is the thing this exists not to do.
// alphaCurrentSynapse's OnEvent bumps `J` while its `i` exposes `I`, and `I` only grows out of
// `J` over LATER ticks -- so an instantaneous sample returns whatever `I` had decayed to,
// which for an isolated first spike is exactly zero at any weight. A network built out of
// those propagates nothing, silently, while every unit test on the state itself passes.
//
// For dynamics linear in the state -- which every current-based synapse lowered here is, and
// which is CHECKED rather than assumed -- Q is a linear functional, so Q(x') - Q(x) is a fixed
// combination of the state CHANGE the handler made:
//     delivered charge = sum over v of charge_coefficient[v] * (x'[v] - x[v])
// Only the coefficients are precomputed, by integrating the synapse's own lowered dynamics on
// the host from each unit basis state until the output decays -- no closed form is written
// down for any synapse type, so this works for whatever current-based dynamics a document
// declares. The state change is read off the edge's LIVE state at the spike, so a handler
// whose assignment depends on the state it finds (a depressing or facilitating synapse) still
// delivers a different amount on each successive spike.
//
// The ring slot holds a CURRENT that the target integrates over exactly one tick, so charge Q
// is written into it as Q / dt.
//
// ── Lazy vs eager synapse updates ────────────────────────────────────────────
// `use_lazy_synapse_updates` selects which of two equivalent schemes advances that per-edge
// state. It is baked into the source rather than passed, so the unused half is not compiled.
//
//   lazy (default)  Nothing touches an edge until its source spikes. The delivery above
//                   first catches the edge up from `edge_attributes` plane 2 -- the tick it
//                   was last advanced through -- to this tick, then records this tick there.
//                   Costs nothing at all for an edge whose source is silent.
//   eager           Every edge advances one dt every tick, in the source neuron's thread,
//                   ahead of the cell dynamics; the delivery then does no catch-up.
//
// The catch-up applies the SAME per-tick step the eager pass does, once per elapsed tick,
// so the two agree exactly rather than approximately -- which is what makes eager usable as
// the reference the lazy path is checked against. It is valid on the same grounds the
// engine's own active-set lazy decay is (CLAUDE.md): the state observed at a spike is the
// state that would have been there, because nothing else observes it in between.
//
// Throws, naming the construct and the ComponentType, on anything below.
GeneratedKernel generate_tick_kernel(const NML_ParseResult &parse_result,
                                     bool use_lazy_synapse_updates = true);

// One lowered synapse program: the wired prototype it came from and which per-edge variable
// planes hold its state.
//
// The engine and this generator must agree on this list exactly -- a program's POSITION in
// it is what `edge_attributes` plane 1 stores per edge, and its state_variable_offset is
// where the generated source reads and writes. Derived here, once, rather than computed
// independently on each side.
struct SynapseProgramLayout {
    s64 prototype_index = -1;
    s64 state_variable_offset = 0;
    s64 state_variable_count = 0;
};

// The synapse prototypes at least one edge actually delivers through, in prototype order.
// A declared-but-unwired synapse contributes nothing to a simulation, so it is neither
// lowered nor allocated for.
Vector<SynapseProgramLayout> resolve_synapse_programs(const NML_ParseResult &parse_result);

// How many per-edge variable planes the model needs: the total state width of every lowered
// synapse program. This is the count the engine hands to
// WeightMatrix::configure_per_edge_variable_count.
s64 per_edge_synapse_variable_count(const NML_ParseResult &parse_result);

// OnStart bodies only -- every cell type's, and every wired synapse prototype's -- run once
// at initialisation. Skips every other stage.
GeneratedKernel generate_initialize_kernel(const NML_ParseResult &parse_result);

// Zeroes row `tick % ring_depth` of `network_inputs`, one thread per neuron.
// The engine dispatches it immediately behind the tick kernel, in the same command batch, so within
// tick T the order is: host adds T's stimulus into row T % ring_depth -> the tick kernel
// reads that row and propagates into rows (T + delay) % ring_depth -> this kernel zeroes row
// T % ring_depth.
//
// The window is the load-bearing part, and there is exactly one correct one:
//
//  - It cannot run before or during the tick kernel: that row holds the arrivals the kernel
//    is about to read.
//  - It cannot be deferred to just before the row is next used. Delays run from 1 to
//    ring_depth - 1, so the earliest write back into row T % ring_depth comes from tick T+1
//    (an edge of delay ring_depth - 1 firing at T+1 arrives at T + ring_depth). Clearing any
//    later would wipe exactly the delayed arrivals this scheme exists to preserve.
//
// Clearing the whole row rather than each cell clearing its own slot as it reads is what
// makes a cell type that never reduces over its synapses safe: its neurons' slots are still
// emptied, where under a per-slot drain they accumulated every arrival forever.
//
// What makes the plain read and this clear race-free is that every delay is >= 1, so no
// thread writes row T % ring_depth during tick T's dispatch. The engine asserts that where
// it flattens the per-edge delays.
//
// `argument_names` is a subset of the master kernel's, bound by the same machinery:
// network_inputs, neuron_count, tick, ring_depth.
GeneratedKernel generate_ring_row_clear_kernel();

// NML/LEMS expression syntax -> C-family syntax, via a tokenizer and a recursive-descent
// parser. Textual substitution cannot do this correctly: `^` is exponentiation and
// right-associative, `.eq.` is a token that also appears inside ordinary identifiers, and
// unary minus needs real precedence. Every one of those failures is silent and numeric.
//
// Exposed because it is the substantive half of this module and is worth testing on its
// own, independently of any surrounding kernel.
String translate_expression(const String &nml_expression, const SymbolTable &symbols);

// ── Deliberately unsupported, each a throw naming the construct + ComponentType ──
//
// Phase 1 is GLIF, which needs none of these. They throw rather than being skipped: a
// silently omitted transition or arrival handler is a model that runs and is wrong.
//
//  - OnEvent (DynamicsStage::Arrival) on a CELL. Synapses lower theirs -- it is how a spike
//    arrival reaches a synapse, so there is no synapse support without it -- but a cell's
//    would need an incoming-event source the engine does not model.
//  - A CONDUCTANCE-BASED synapse, i.e. one declaring erev / gbase / gbase1 / gbase2. It
//    computes i = g * (erev - v), a driving force that depends on the postsynaptic voltage
//    and reverses sign as v crosses erev, where a current-based synapse injects a fixed
//    current profile. Those are different models, not approximations of each other, so the
//    generator names the synapse type and stops rather than running it as current-based.
//  - A synapse whose SynapseTypeSpecification::requires_per_edge_state is set. Its state is
//    now stored per edge like every other synapse's, so THAT is no longer the obstacle; what
//    the flag actually records (see nml.cpp's requires_per_edge_state) is a <Children
//    type="basePlasticityMechanism"/> or "baseBlockMechanism" declaration, and this generator
//    lowers no child-component structure. Running the parent's dynamics alone would silently
//    drop the mechanism that makes the synapse depress, facilitate or block.
//  - Regime / Transition / OnEntry on a SYNAPSE. Cells support them (see CellRegimeLayout);
//    a synapse's would need the same appended slot in its per-edge state, and no synapse in
//    the standard library declares one.
//  - ConditionalDerivedVariable / Case. Not a choice: DynamicsInstruction carries a Case's
//    `value` but not its `condition` attribute (nml.cpp's collect_dynamics_instructions
//    reads only value/test/select), so the per-case tests never reach codegen. Emitting
//    the cases without their guards would be silently wrong, so it throws instead.
//  - A `select=` path over anything but the attached synapses -- see generate_tick_kernel.
//  - random(x). A deterministic per-neuron stream needs a seed argument, and adding one
//    would change the argument order above for a function Phase 1 never calls.
//
// Four more come from resolving what one arriving spike delivers (see generate_tick_kernel).
// Each of them would otherwise be a plausible wrong number on every spike, for the whole run:
//  - A synapse NONLINEAR in its state variables. The delivered charge is precomputed as a
//    fixed combination of the state change the handler makes, and that decomposition only
//    exists for linear dynamics. Measured by probing, not assumed: the response to twice a
//    state variable has to be twice the response, and the response to all of them at once has
//    to be the sum.
//  - A synapse whose exposed `i` never decays. Its response to one event carries unbounded
//    charge, and no scalar expresses that.
//  - A synapse whose dynamics read the absolute time `t`. What a spike delivers would then
//    depend on when it arrived, which one set of coefficients resolved at generation time
//    cannot express.
//  - A synapse whose OnEvent handler moves only state the exposed `i` never depends on. Every
//    spike through it would deliver exactly zero, which is the same silent swallowing a
//    synapse with no handler at all would do.
} // namespace spikecorec::nml
