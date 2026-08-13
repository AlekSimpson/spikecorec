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
//   1  cell_parameters          device const float *
//   2  network_inputs           device float *        -- the delay ring, see below
//   3  last_spiked              device long *
//   4  spike_flags              device int *
//   5  cell_state_base          device const int *
//   6  cell_parameter_base      device const int *
//   7  cell_type_index          device const int *
//   8  neuron_count             constant int &
//   9  dt                       constant float &
//   10 tick                     constant long &
//   11 internal_node_words      device const uint *
//   12 leaf_node_words          device const uint *
//   13 rank_superblock_table    device const uint *
//   14 rank_subblock_table      device const ushort *
//   15 U_matrix                 device const float *
//   16 V_matrix                 device const float *
//   17 edge_weight_coefficients device const float *  -- the weight matrix's Ck
//   18 edge_weight_deltas       device const float *  -- the weight matrix's Sk
//   19 edge_delay_ticks         device const int *    -- per-edge delay, whole ticks
//   20 branching_factor         constant int &
//   21 superblock_size_words    constant int &
//   22 padded_node_count        constant int &
//   23 tree_height              constant int &
//   24 internal_bit_count       constant int &
//   25 rank_float4_stride       constant long &
//   26 constant_weight          constant float &
//   27 max_neighbor_count       constant int &
//   28 ring_depth               constant int &
//   29 synapse_state            device float *        -- aggregated synapse state, see below
//   30 edge_synapse_plane       device const int *    -- per-edge, which ring plane it feeds
//
// Arguments 11-19 are the adjacency and per-edge state stage 6 (Propagate) walks; 20-28
// describe their shape. The initialize kernel takes the identical list so the engine binds
// one argument set for both entry points, and simply never reads the propagation half.
//
// That is 31 arguments, which is exactly Metal's per-stage buffer argument table limit.
// Anything added from here on has to consolidate two logical arrays into one buffer with
// baked offsets, the way the network_inputs ring already packs its planes.
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
// `network_inputs` is a delay ring: one flat allocation of
// ring_depth * plane_count * neuron_count floats indexed as [row][plane][neuron]. An arrival
// lands in row (tick + edge delay) % ring_depth and a cell reads row tick % ring_depth.
// Arrivals are atomic -- many sources converge on one target -- but the READ is a plain
// load, and it does not clear anything: the row is zeroed as a whole row after the dispatch,
// by generate_ring_row_clear_kernel below.
//
// `ring_depth` exceeds the model's largest per-edge delay -- the engine computes it -- so an
// arrival can never land in the row being read this tick.
//
// ── The ring's planes, and stage 2 for synapses ──────────────────────────────
// A ring row is `plane_count` = 1 + wired_synapse_prototype_indices().size() planes wide:
//
//   plane 0        the DELIVERED CURRENT every cell reads through its `synapses[*]/i` path.
//                  External stimulus is added here by the host, synapse outputs by the
//                  synapse stage, and an edge that names no synapse scatters its raw weight
//                  straight into it -- which is the pre-synapse behaviour, kept because an
//                  edge with no synapse component has no dynamics to run.
//   plane 1 + p    arrivals awaiting wired synapse prototype p. An edge whose projection
//                  names that prototype scatters its weight here instead, and the synapse
//                  stage drains it.
//
// Synapse state is aggregated PER (target neuron, synapse prototype), not per edge. A sum
// of same-prototype synapses is a single synapse of the summed arrival weight -- their state
// equations are linear and share one parameter set -- so many converging edges of one
// prototype share one state. Keyed by PROTOTYPE rather than by type because the parameters
// are a property of the prototype: two alphaCurrentSynapse instances with different `tau`
// decay at different rates and must not share a state or a plane. That is also what stops
// two current-based synapses on one target from being silently pooled: they land in
// different planes and integrate separately.
//
// `synapse_state` is that storage: wired prototype p's slice starts at
// (sum of the preceding prototypes' state variable counts) * neuron_count, and within it a
// neuron occupies its type's state_variable_names.size() consecutive floats. The offsets are
// baked into the generated source, so no per-prototype descriptor buffer is needed; so are
// the prototypes' parameter values, which are constants of the prototype.
//
// Per tick, per neuron, ahead of the cell dynamics and in the same thread (so the write and
// the cell's read need no synchronisation -- one thread owns one target's slots):
//
//   1. Deliver: read plane 1+p of this tick's row. Non-zero means arrivals are due, and the
//      OnEvent handler runs once with `weight` bound to the SUMMED arrival weight. For a
//      handler affine in `weight` -- every current-based synapse in the standard library --
//      that is exactly the sum of the individual arrivals' effects.
//   2. Integrate one dt, forward Euler, same simultaneous write-back as a cell.
//   3. Add the synapse's `i` exposure into plane 0 of this tick's row.
//
// Throws, naming the construct and the ComponentType, on anything below.
GeneratedKernel generate_tick_kernel(const NML_ParseResult &parse_result);

// The synapse prototypes at least one edge actually delivers through, in prototype order.
// A declared-but-unwired synapse contributes nothing to a simulation, so it is neither
// lowered nor allocated for.
//
// The engine and this generator must agree on this list exactly: a prototype's POSITION in
// it selects its slice of `synapse_state`, and that position + 1 is its plane in the
// network_inputs ring. Derived here, once, rather than computed independently on each side.
Vector<s64> wired_synapse_prototype_indices(const NML_ParseResult &parse_result);

// OnStart bodies only -- every cell type's, and every wired synapse prototype's -- run once
// at initialisation. Skips every other stage.
GeneratedKernel generate_initialize_kernel(const NML_ParseResult &parse_result);

// Zeroes every plane of row `tick % ring_depth` of `network_inputs`, one thread per neuron.
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
// network_inputs, neuron_count, tick, ring_depth. It takes the model because the row's
// plane count comes out of it and is baked into the source rather than passed -- the
// argument table has no slot left to spend on a scalar.
GeneratedKernel generate_ring_row_clear_kernel(const NML_ParseResult &parse_result);

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
//  - A synapse whose SynapseTypeSpecification::requires_per_edge_state is set: its state
//    does not superpose across converging edges, so the per-target aggregation above is
//    invalid for it. Per-edge synapse state is a separate ticket.
//  - Regime / Transition / OnEntry on a SYNAPSE. Cells support them (see CellRegimeLayout);
//    a synapse's would need the same appended state slot in `synapse_state`, and no synapse
//    in the standard library declares one.
//  - ConditionalDerivedVariable / Case. Not a choice: DynamicsInstruction carries a Case's
//    `value` but not its `condition` attribute (nml.cpp's collect_dynamics_instructions
//    reads only value/test/select), so the per-case tests never reach codegen. Emitting
//    the cases without their guards would be silently wrong, so it throws instead.
//  - A `select=` path over anything but the attached synapses -- see generate_tick_kernel.
//  - random(x). A deterministic per-neuron stream needs a seed argument, and adding one
//    would change the argument order above for a function Phase 1 never calls.
} // namespace spikecorec::nml
