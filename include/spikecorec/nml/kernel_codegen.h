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
// state_variable_names.size() floats. `cell_parameters` follows the identical scheme with
// parameter_names.size() floats per cell.
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
//
// Arguments 11-19 are the adjacency and per-edge state stage 6 (Propagate) walks; 20-28
// describe their shape. The initialize kernel takes the identical list so the engine binds
// one argument set for both entry points, and simply never reads the propagation half.
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
// each OnCondition gates, then Propagate.
//
// Every TimeDerivative is evaluated against the state as it stood at entry and written
// back only once all of them have been computed, so two variables that reference each
// other integrate consistently instead of one seeing the other's updated value.
//
// A DerivedVariable written as a `select=` path over the attached synapses
// ("synapses[*]/i" on iafCell, "synapses[*]/I" on izhikevichCell) lowers to a read of this
// tick's `network_inputs` ring row, and binds under its own name like any other
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
// indexed as [row][neuron]. An arrival lands in row (tick + edge delay) % ring_depth; a cell
// reads row tick % ring_depth and clears it, so the row is empty when the ring wraps back
// onto it. `ring_depth` exceeds the model's largest per-edge delay -- the engine computes it
// -- so an arrival can never land in the row being drained this tick.
//
// Throws, naming the construct and the ComponentType, on anything below.
GeneratedKernel generate_tick_kernel(const NML_ParseResult &parse_result);

// OnStart bodies only, run once at initialisation. Skips every other stage.
GeneratedKernel generate_initialize_kernel(const NML_ParseResult &parse_result);

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
//  - OnEvent (DynamicsStage::Arrival) -- incoming-spike handlers.
//  - Regime / Transition / OnEntry (DynamicsStage::RegimeEntry), and any instruction
//    carrying a non-empty regime_name.
//  - ConditionalDerivedVariable / Case. Not a choice: DynamicsInstruction carries a Case's
//    `value` but not its `condition` attribute (nml.cpp's collect_dynamics_instructions
//    reads only value/test/select), so the per-case tests never reach codegen. Emitting
//    the cases without their guards would be silently wrong, so it throws instead.
//  - A `select=` path over anything but the attached synapses -- see generate_tick_kernel.
//  - random(x). A deterministic per-neuron stream needs a seed argument, and adding one
//    would change the argument order above for a function Phase 1 never calls.
} // namespace spikecorec::nml
