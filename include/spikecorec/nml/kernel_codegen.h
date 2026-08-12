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
//   0  cell_state           device float *
//   1  cell_parameters      device const float *
//   2  network_inputs       device float *
//   3  last_spiked          device long *
//   4  spike_flags          device int *
//   5  cell_state_base      device const int *
//   6  cell_parameter_base  device const int *
//   7  cell_type_index      device const int *
//   8  neuron_count         constant int &
//   9  dt                   constant float &
//   10 tick                 constant long &
//
// Arguments 0-7 are buffers; 8-10 are scalars bound by value. The initialize kernel takes
// the identical list so the engine binds one argument set for both entry points.
struct GeneratedKernel {
    String source;
    String function_name;
    Vector<String> argument_names;
};

// Per-tick dynamics: Integrate, then Detect, then the Reset and Emit bodies each
// OnCondition gates.
//
// Every TimeDerivative is evaluated against the state as it stood at entry and written
// back only once all of them have been computed, so two variables that reference each
// other integrate consistently instead of one seeing the other's updated value.
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
//  - DerivedVariable written as a `select=` path (e.g. "synapses[*]/i"). The path lands in
//    `expression` and fails to parse as an expression, which reports as a malformed
//    expression naming the ComponentType.
//  - random(x). A deterministic per-neuron stream needs a seed argument, and adding one
//    would change the argument order above for a function Phase 1 never calls.
} // namespace spikecorec::nml
