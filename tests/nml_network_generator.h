#pragma once

#include <string>

#include "spikecorec/core/types.h"

namespace spikecorec::nml::network_generation {

// ── Programmatic NeuroML network generator (ticket #100 [T1]) ───────────────────────────────────
//
// Emits valid NeuroML `<population>`/`<projection>`/`<explicitInput>` XML for an arbitrary neuron
// count and connectivity pattern -- so a network of, say, 2000 neurons doesn't need 2000 hand-typed
// `<population>` elements. Every GLIF ComponentType text block below is reused VERBATIM (same
// equations, same Parameter/dimension declarations) from tests/cell_lowering_tests.cpp's own
// GLIF1-5 fixtures (ticket #50) -- see that file's own header comment for the modeling rationale
// of each variant; not re-derived here, per this ticket's own instruction.
//
// A generated document intentionally omits the `xsi`/`schemaLocation` attributes and
// `<Simulation>`/`<OutputFile>`/`<EventOutputFile>` blocks every OTHER checked-in fixture in this
// tree carries (e.g. glif3_single_cell.nml) -- those exist purely so the SAME file also drives real
// jLEMS/pyneuroml; ticket #100's own validation is internal-consistency only (no jLEMS reference
// needed, per explicit user direction), so this generator only emits what spikecorec's own
// front-end needs. Every caller writes the returned string as a plain content file and parses it
// through a thin "<include>-only top.nml" wrapper (this tree's own established convention --
// NML_Parser only XSD-validates the TOP-LEVEL file it's given), never as a top-level document
// itself.

enum class GlifVariant { Glif1 = 1, Glif2 = 2, Glif3 = 3, Glif4 = 4, Glif5 = 5 };

// The ComponentType name for `variant` ("GLIF1Cell".."GLIF5Cell").
String glif_component_type_name(GlifVariant variant);

// The full `<ComponentType name="GLIF<N>Cell" ...>` declaration for `variant`.
String glif_component_type_xml(GlifVariant variant);

// The declared-order slot index (0-based) of state variable `state_variable_name` within
// `variant`'s own `<Dynamics>` -- e.g. "v" -> 0 for every variant, "asc1" -> 2 for GLIF3/GLIF5,
// "theta" -> 1 for GLIF4/GLIF5. A population's cell_state chunk is structure-of-arrays across the
// whole population (allocator.h's own doc comment): state variable slot `k`'s own [population.size]
// row sits at `cell_type_boundaries[population_index] + k * population.size`, so this slot index is
// exactly what a caller needs to read that state variable back out of ModelAllocation::cell_state
// for any neuron in a population of this variant. Throws std::runtime_error if `variant` declares
// no state variable of that name.
s32 glif_state_variable_slot(GlifVariant variant, const String &state_variable_name);

// One homogeneous population of `neuron_count` neurons of GLIF variant `variant`, all bound to the
// SAME cell-instance component (Phase-1's uniform-population baked-constants convention -- every
// neuron in one GeneratedPopulation shares identical resolved Parameter values).
struct GeneratedPopulation {
    String population_id;
    GlifVariant variant = GlifVariant::Glif1;
    String bound_instance_id;
    String cell_instance_attributes; // e.g. "C=\"100pF\" gL=\"10nS\" EL=\"-70mV\" vth=\"-50mV\" ..."
    s32 neuron_count = 0;
};

// One projection's connectivity: deterministic fixed-out-degree wiring from every neuron of
// `presynaptic_population_id` to `out_degree` downstream targets in `postsynaptic_population_id`
// (`target_local_index = (source_local_index + offset) % postsynaptic neuron_count`, for `offset`
// in `[1, out_degree]`) -- deliberately simple and deterministic (not a random graph) so a caller
// can predict exactly who is connected to whom for internal-consistency/timing checks. A connection
// that would land on its own source neuron (only possible for a size-1 self-projection) is silently
// skipped -- K2Tree rejects self-loops (see inputs_lowering's own established workaround for the
// same constraint). `delay_attribute`, if non-empty (e.g. "5ms"), emits `<connectionWD ... delay=
// "5ms" weight="1"/>` (a real, non-trivial per-edge delay -- ticket #64's delay ring); empty emits a
// plain `<connection>` (no delay attribute -- floors to the engine's implicit one-tick latency).
struct GeneratedProjection {
    String projection_id;
    String presynaptic_population_id;
    String postsynaptic_population_id;
    String synapse_instance_id; // must be declared in the caller-supplied synapse_declarations_xml
    s32 out_degree = 1;
    String delay_attribute;
};

// One `explicitInput`-driven pulseGenerator continuous-current stimulus (ticket #58) targeting one
// specific neuron -- `delay`/`duration`/`amplitude` are verbatim NML unit-suffixed literals (e.g.
// "10ms", "0.5nA"), matching every other checked-in fixture's own pulseGenerator attribute style.
struct GeneratedStimulus {
    String population_id;
    s32 local_index = 0;
    String pulse_generator_id;
    String delay;
    String duration;
    String amplitude;
};

// Renders one complete, standalone NeuroML document string (content-file shape, not a top-level
// document -- see this header's own doc comment): every distinct GLIF ComponentType actually used
// by `populations` (deduplicated, first-seen order), one bound cell instance per population,
// `synapse_declarations_xml` inserted verbatim (caller-supplied, e.g. a real vendored
// `<expOneSynapse id="..." .../>` -- this generator has no opinion on synapse ComponentType choice,
// since AssembledModel's propagate stage doesn't yet invoke real per-edge synapse dynamics either
// way, see tests/end_to_end_network_tests.cpp's own header comment), then a `<network>` with every
// population/explicitInput/projection+connection. Throws std::runtime_error if a
// GeneratedProjection names a population id not present in `populations`.
String generate_network_nml(
    const String &document_id,
    const Vector<GeneratedPopulation> &populations,
    const String &synapse_declarations_xml,
    const Vector<GeneratedProjection> &projections,
    const Vector<GeneratedStimulus> &stimuli);

} // namespace spikecorec::nml::network_generation
