#pragma once

#include "spikecorec/core/types.h"
#include "spikecorec/nml/nml.h"

namespace spikecorec::nml {

// What each name inside an NML expression is bound to in the generated kernel: the
// declared NML name maps to the C expression that reads it. "leakConductance" ->
// "cell_parameters[parameter_base + 3]", "v" -> "state_0", "iSyn" -> "network_input".
using SymbolTable = UnorderedMap<String, String>;

// "leakConductance * (leakReversal - v) + iSyn" -> "((param_3 * (param_4 - state_0)) + input)".
//
// Throws naming the offending token, the expression and `owner_name` on an unknown
// identifier, an unknown function, or a syntax error. It never emits an unresolved name
// and hopes the target compiler finds a definition -- a silently mistranslated
// expression is how a network ends up quietly simulating the wrong dynamics.
String translate_expression(const String &expression,
                            const SymbolTable &symbols,
                            const String &owner_name);

// The starting value of one OnStart StateAssignment, folded on the host because OnStart
// runs once at init rather than per tick.
//
// Phase 1 accepts a numeric literal, a parameter name, or either of those negated --
// which covers every GLIF cell and current-based synapse in the standard library
// (`v = leakReversal`, `I = 0`). Anything else throws rather than silently defaulting the
// variable to zero, because a cell that starts at 0 V when it should start at its leak
// reversal is above threshold on tick 0 and fires the whole network once before going
// permanently quiet.
f64 evaluate_initial_value(const String &expression,
                           const Vector<String> &parameter_names,
                           const Vector<Real> &parameter_values,
                           const String &owner_name);

// Where everything the model needs sits in the engine's buffers. Derived once from the
// parse result and consumed by two callers that must agree exactly: the engine, which
// sizes and fills the buffers, and the codegen, which bakes the same offsets into the
// kernel it emits as `constant` tables. Computing it in one place is what keeps them from
// disagreeing by a slot.
struct ModelLayout {
    s64 total_neuron_count = 0;

    // Cell state is laid out one contiguous chunk per population, and inside a chunk one
    // contiguous run per state variable -- so neighbouring threads read neighbouring
    // addresses, and no neuron reserves slots for a cell type it is not. Variable k of the
    // local_index'th neuron of population p sits at
    // population_state_base[p] + k * populations[p].neuron_count + local_index.
    Vector<s64> population_state_base;
    s64 cell_state_length = 0;

    // One row per prototype, in prototype order; a population's neurons all read the row
    // of the prototype the population instantiates.
    Vector<s64> cell_prototype_parameter_base;
    s64 cell_parameter_length = 0;

    Vector<s64> synapse_prototype_parameter_base;
    s64 synapse_parameter_length = 0;

    // Planes in WeightMatrix::per_edge_variable_values. Plane 0 holds each edge's synapse
    // prototype index; plane 1 + k holds state variable k of whatever synapse type that
    // edge's prototype is. Types with fewer state variables simply use fewer planes, so
    // the count is one plus the widest synapse type in the model.
    s64 per_edge_variable_count = 0;
    s64 widest_synapse_state_count = 0;

    // Rows in the spike-history ring: one more than the longest connection delay, so the
    // row a thread reads for a delayed arrival is never the row being written this tick.
    // Never below 2, because the minimum delay is one tick.
    s64 spike_history_length = 2;
    s64 maximum_edge_delay = 0;
    s64 total_edge_count = 0;
};

ModelLayout compute_model_layout(const NML_ParseResult &parse_result);

// The comparison a cell spikes on, taken from the OnCondition that carries its EventOut:
// `v .gt. thresh` yields ("v", "thresh"). Both come back as names — the caller decides
// whether each is a parameter, a state variable or a literal, because that differs by
// model: GLIF1 tests against the parameter `vth` while GLIF4 tests against the state
// variable `theta`.
//
// Returns false when the type emits no spike, or when its condition is not a comparison
// this can read.
bool find_spike_threshold_condition(const CellTypeSpecification &cell_type,
                                    String &return_state_variable,
                                    String &return_threshold_symbol);

// The complete master kernel source for this model: the fixed 9-stage scaffold with one
// generated body spliced in per cell type and per synapse type. Compile with
// compile_kernel(source.c_str(), "master_step").
//
// Throws on any construct Phase 1 does not simulate -- a conductance-based synapse, a
// synapse type whose state does not superpose, a regime-scoped cell -- naming the
// ComponentType. A model that cannot be simulated correctly must fail to load rather than
// produce a plausible-looking recording of the wrong thing.
String generate_master_kernel(const NML_ParseResult &parse_result, const ModelLayout &layout);

} // namespace spikecorec::nml
