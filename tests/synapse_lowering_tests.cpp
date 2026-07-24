#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <gtest/gtest.h>

#include "spikecorec/core/engine.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/nml/cell_lowering.h"
#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/synapse_lowering.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// core/engine.h (needed below for SpikeEngine, the alphaCurrentSynapse acceptance test's own
// migration off AssembledModel) has its own file-scope `using namespace spikecorec::log;`, which
// makes the ALIAS TEMPLATE `spikecorec::log::Vector` ambiguous with `spikecorec::Vector` for a plain
// unqualified `Vector<...>` (two distinct `using namespace` directives at the same scope, each making
// a same-named alias TEMPLATE visible) -- exactly the clash tests/simple_lif_stdp_network_tests.cpp's
// own header comment already documents. So this file spells out `spikecorec::Vector<...>` at its
// (few) own bare-`Vector<...>` call sites below, rather than the bare `Vector<...>` this file used
// before engine.h was pulled in.

// ── Synapse dynamics -> IR lowering tests (ticket #51 [B3]) ──────────────
//
// Every synapse now lowers uniformly through the sole per-edge shape (arch
// §4.3 design revision -- the old aggregatable-per-neuron-accumulator vs
// per-edge storage split is gone, see synapse_lowering.h's header comment).
// These tests cover the one classification distinction that's still
// meaningful -- current-based vs conductance-based, which determines what
// VALUE the `forall` body computes (`i = g` vs `i = g*(erev-v)`), not how
// state is stored. Where the real vendored std-lib ComponentType is
// directly usable (`expOneSynapse` -- needs nothing this front-end can't
// already parse), the fixture below references it directly by tag, with no
// inline `<ComponentType>` declaration, exactly like
// model_specification_tests.cpp's own synapse fixtures. Two targets need a
// custom, self-contained fixture instead (same precedent cell_lowering_tests.cpp
// itself sets for GLIF1-5, none of which reuse a real Allen ComponentType
// verbatim either):
//   - `expTwoSynapse` (real): its OnEvent references `waveformFactor`, a
//     `<DerivedParameter>` -- a LEMS tag ticket #2/#49's front-end does not
//     currently extract into any typed field (grep confirms no
//     `DerivedParameter` handling anywhere in nml.cpp/resolve.cpp), so
//     lowering the real type's Dynamics verbatim would throw "undeclared
//     identifier 'waveformFactor'" for a reason unrelated to this ticket's
//     own scope. `TestExpTwoSynapse` below reproduces the identical
//     structure (two decaying state variables A/B, a derived `g` from their
//     difference, a derived `i = g*(erev-v)`, an OnEvent bumping both A and
//     B) with `waveformFactor` declared as a plain `Parameter` instead of a
//     `DerivedParameter` -- sidestepping the gap without masking it. It also
//     exercises the multi-state-variable per-edge case (two `peredge` slots
//     loaded in the same `forall` body) that the single-state-variable
//     fixtures below don't.
//   - `blockingPlasticSynapse` (real, the NMDA target): its real Dynamics
//     compose child `plasticityMechanisms`/`blockMechanisms` components via
//     `select`/`reduce="multiply"` DerivedVariables -- full sub-component
//     composition, explicitly out of this ticket's scope (arch §4.3's own
//     NMDA example elides the Mg-block the same way, and blockingPlasticSynapse
//     itself extends expTwoSynapse so also inherits the waveformFactor gap
//     above). `TestNmdaSynapse` below reproduces the IR spec §4 NMDA
//     example's own simplified shape almost exactly (single decaying
//     conductance `g`, `i = g*(erev-v)`, a `Children` declaration for
//     fidelity to the real type's shape), just without composing the
//     elided mechanism.
//
// Every fixture is routed through a trivial one-neuron self-loop network (a
// `DummyCell` used purely as projection plumbing, irrelevant to what's under
// test) and the whole parse -> resolve_and_lower -> build_model_specification
// pipeline, exactly like cell_lowering_tests.cpp's own
// build_cell_type_library_entry.

namespace {

String write_temp_file(const String &filename, const String &contents) {
    String path = (std::filesystem::temp_directory_path() / filename).string();
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

const TypeLibraryEntry &type_library_entry_for(const ModelSpecification &specification, const String &bound_instance_id) {
    for (const auto &entry : specification.type_library) {
        if (entry.bound_instance_id == bound_instance_id) return entry;
    }
    throw std::runtime_error("no type library entry for '" + bound_instance_id + "'");
}

const String DUMMY_CELL_COMPONENT_TYPE =
    "  <ComponentType name=\"DummyCell\" extends=\"baseCell\">"
    "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
    "      <TimeDerivative variable=\"v\" value=\"network_inputs / C\"/>"
    "    </Dynamics>"
    "  </ComponentType>";

// Wraps one synapse ComponentType (either a real std-lib tag with an empty
// `synapse_component_type_xml`, or a custom inline declaration) plus one
// bound instance into a minimal two-neuron network (node 0 -> node 1; K2Tree
// rejects self-loops, so pre and post must be distinct nodes), runs it
// through parse -> resolve_and_lower -> build_model_specification, and
// returns the resulting synapse TypeLibraryEntry.
TypeLibraryEntry build_synapse_type_library_entry(
    const String &fixture_id, const String &synapse_component_type_xml,
    const String &synapse_tag, const String &synapse_instance_attributes
) {
    write_temp_file("spikecorec_synapse_lowering_" + fixture_id + "_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"SynapseLowering" + fixture_id + "Content\">"
        + DUMMY_CELL_COMPONENT_TYPE +
        "  <DummyCell id=\"dummyCellInstance\" C=\"1.0e-10\"/>"
        + synapse_component_type_xml +
        "  <" + synapse_tag + " id=\"synapseInstance\" " + synapse_instance_attributes + "/>"
        "  <network id=\"Net\">"
        "    <population id=\"Pop\" component=\"dummyCellInstance\" size=\"2\"/>"
        "    <projection id=\"Proj\" presynapticPopulation=\"Pop\" postsynapticPopulation=\"Pop\" synapse=\"synapseInstance\">"
        "      <connection id=\"0\" preCellId=\"Pop/0/dummyCellInstance\" postCellId=\"Pop/1/dummyCellInstance\"/>"
        "    </projection>"
        "  </network>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_synapse_lowering_" + fixture_id + "_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"SynapseLowering" + fixture_id + "Top\">"
        "  <include href=\"spikecorec_synapse_lowering_" + fixture_id + "_content.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    ModelSpecification specification = build_model_specification(resolved);
    return type_library_entry_for(specification, "synapseInstance");
}

const String EXP_ONE_CURRENT_SYNAPSE_COMPONENT_TYPE =
    "  <ComponentType name=\"ExpOneCurrentSynapse\" extends=\"baseCurrentBasedSynapse\">"
    "    <Property name=\"weight\" dimension=\"none\" defaultValue=\"1\"/>"
    "    <Parameter name=\"tau\" dimension=\"time\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"g\" dimension=\"current\" exposure=\"g\"/>"
    "      <DerivedVariable name=\"i\" exposure=\"i\" dimension=\"current\" value=\"g\"/>"
    "      <TimeDerivative variable=\"g\" value=\"-g / tau\"/>"
    "      <OnStart>"
    "        <StateAssignment variable=\"g\" value=\"0\"/>"
    "      </OnStart>"
    "      <OnEvent port=\"in\">"
    "        <StateAssignment variable=\"g\" value=\"g + weight\"/>"
    "      </OnEvent>"
    "    </Dynamics>"
    "  </ComponentType>";

const String TEST_EXP_TWO_SYNAPSE_COMPONENT_TYPE =
    "  <ComponentType name=\"TestExpTwoSynapse\" extends=\"baseConductanceBasedSynapse\">"
    "    <Property name=\"weight\" dimension=\"none\" defaultValue=\"1\"/>"
    "    <Parameter name=\"tauRise\" dimension=\"time\"/>"
    "    <Parameter name=\"tauDecay\" dimension=\"time\"/>"
    "    <Parameter name=\"waveformFactor\" dimension=\"none\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"A\" dimension=\"none\"/>"
    "      <StateVariable name=\"B\" dimension=\"none\"/>"
    "      <DerivedVariable name=\"g\" dimension=\"conductance\" exposure=\"g\" value=\"gbase * (B - A)\"/>"
    "      <DerivedVariable name=\"i\" exposure=\"i\" dimension=\"current\" value=\"g * (erev - v)\"/>"
    "      <TimeDerivative variable=\"A\" value=\"-A / tauRise\"/>"
    "      <TimeDerivative variable=\"B\" value=\"-B / tauDecay\"/>"
    "      <OnStart>"
    "        <StateAssignment variable=\"A\" value=\"0\"/>"
    "        <StateAssignment variable=\"B\" value=\"0\"/>"
    "      </OnStart>"
    "      <OnEvent port=\"in\">"
    "        <StateAssignment variable=\"A\" value=\"A + weight * waveformFactor\"/>"
    "        <StateAssignment variable=\"B\" value=\"B + weight * waveformFactor\"/>"
    "      </OnEvent>"
    "    </Dynamics>"
    "  </ComponentType>";

// A per-edge synapse whose `TimeDerivative` is declared but does NOT match
// the recognized linear-decay shape (`1 / tau`, a constant charging rate --
// no `-state` numerator, so `detect_linear_decay_shape` returns nullopt) --
// exercises the general forward-Euler fallback (this synapse's own RHS
// references no OTHER per-edge state variable, unlike alphaCurrentSynapse's
// own `I`/`J`, so this is the fallback's simplest possible shape).
const String TEST_NON_DECAYING_SYNAPSE_COMPONENT_TYPE =
    "  <ComponentType name=\"TestNonDecayingSynapse\" extends=\"baseConductanceBasedSynapse\">"
    "    <Property name=\"weight\" dimension=\"none\" defaultValue=\"1\"/>"
    "    <Parameter name=\"tau\" dimension=\"time\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"g\" dimension=\"conductance\" exposure=\"g\"/>"
    "      <DerivedVariable name=\"i\" exposure=\"i\" dimension=\"current\" value=\"g * (erev - v)\"/>"
    "      <TimeDerivative variable=\"g\" value=\"1 / tau\"/>"
    "      <OnStart>"
    "        <StateAssignment variable=\"g\" value=\"0\"/>"
    "      </OnStart>"
    "      <OnEvent port=\"in\">"
    "        <StateAssignment variable=\"g\" value=\"g + weight\"/>"
    "      </OnEvent>"
    "    </Dynamics>"
    "  </ComponentType>";

const String TEST_NMDA_SYNAPSE_COMPONENT_TYPE =
    "  <ComponentType name=\"TestNmdaSynapse\" extends=\"baseConductanceBasedSynapse\">"
    "    <Property name=\"weight\" dimension=\"none\" defaultValue=\"1\"/>"
    "    <Parameter name=\"tau\" dimension=\"time\"/>"
    "    <Children name=\"blockMechanisms\" type=\"baseBlockMechanism\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"g\" dimension=\"conductance\" exposure=\"g\"/>"
    "      <DerivedVariable name=\"i\" exposure=\"i\" dimension=\"current\" value=\"g * (erev - v)\"/>"
    "      <TimeDerivative variable=\"g\" value=\"-g / tau\"/>"
    "      <OnStart>"
    "        <StateAssignment variable=\"g\" value=\"0\"/>"
    "      </OnStart>"
    "      <OnEvent port=\"in\">"
    "        <StateAssignment variable=\"g\" value=\"g + weight\"/>"
    "      </OnEvent>"
    "    </Dynamics>"
    "  </ComponentType>";

// SpikeEngine migration note (real, newly-surfaced gap -- see this test's own header comment further
// below for the full detail): plain `DummyCell` (used everywhere else in this file) declares no
// `EventOut` at all, so `collect_emit_port_names`/SpikeEngine's own `nml_emit_port_flags_` register
// ZERO emit ports for a model built from it -- there is then no real per-neuron flag buffer for
// `SpikeEngine::force_emit("spike", ...)` to target (it correctly throws "not a known emit port" in
// that case, exactly per its own scoped-setter contract). The OLD `AssembledModel`/`ModelRuntimeBuffers`
// path never needed a real declared port here because `AssembledModel::dispatch_synapse_delivery_events`
// (ticket #131) unions over WHATEVER keys are present in the caller-supplied `emit_port_flags` map
// directly (master_kernel.cpp), independent of the model's own declared ports -- a looser, ad hoc
// mechanism `ModelRuntimeBuffers` allowed but `SpikeEngine::force_emit` deliberately does not
// (force_emit is scoped to real, model-declared emit ports on purpose). This dedicated
// `DummyCellWithSpikePort` ComponentType (used ONLY by this one acceptance test, not by any other
// fixture in this file) is the minimal, honest fix: it gives neuron 0 a REAL, structurally-declared
// `spike` EventOut so SpikeEngine allocates a real flag buffer for `force_emit` to target, gated by an
// `OnCondition` that is provably never satisfied on its own (neuron 0 has no incoming connection in
// this 2-neuron 0->1 network, so its own `v` never leaves its initial 0 -- `v .gt. spikeThreshold`
// with any positive `spikeThreshold` can therefore never fire from the cell's own dynamics; the ONLY
// way "spike" ever becomes true is the explicit `force_emit` call below). This changes nothing about
// the test's own OBSERVABLE behavior (neuron 0's own accumulate/threshold cycle was never exercised
// or asserted on either before or after this change) -- it only makes the model structurally support
// the same "designated neuron can be forced to spike externally" capability the pre-migration test
// already relied on, through the correctly-scoped mechanism `force_emit` actually provides.
const String DUMMY_CELL_WITH_SPIKE_PORT_COMPONENT_TYPE =
    "  <ComponentType name=\"DummyCellWithSpikePort\" extends=\"baseCell\">"
    "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
    "    <Parameter name=\"spikeThreshold\" dimension=\"voltage\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
    "      <TimeDerivative variable=\"v\" value=\"network_inputs / C\"/>"
    "      <OnCondition test=\"v .gt. spikeThreshold\">"
    "        <EventOut port=\"spike\"/>"
    "      </OnCondition>"
    "    </Dynamics>"
    "  </ComponentType>";

// Builds a small, real 2-neuron network (node 0 -> node 1, the same
// self-loop-avoiding wiring `build_synapse_type_library_entry`
// itself uses) through a real, vendored `alphaCurrentSynapse` instance, and
// returns the FULL ModelSpecification (not just the synapse's own
// TypeLibraryEntry) so a caller can build IR programs and run a SpikeEngine
// end to end -- the numeric acceptance test below needs both
// the cell AND synapse type-library entries, unlike every other fixture in
// this file. Uses `DummyCellWithSpikePort` (see its own doc comment just above), not plain
// `DummyCell`, so `force_emit("spike", 0)` has a real emit port to target.
ModelSpecification build_alpha_current_synapse_network_specification(
    const String &tau_attribute, const String &ibase_attribute, const String &weight_attribute
) {
    write_temp_file("spikecorec_synapse_lowering_alpha_acceptance_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"SynapseLoweringAlphaAcceptanceContent\">"
        + DUMMY_CELL_WITH_SPIKE_PORT_COMPONENT_TYPE +
        "  <DummyCellWithSpikePort id=\"dummyCellInstance\" C=\"1.0e-10\" spikeThreshold=\"1V\"/>"
        "  <alphaCurrentSynapse id=\"alphaSynapseInstance\" tau=\"" + tau_attribute + "\" ibase=\"" +
        ibase_attribute + "\" weight=\"" + weight_attribute + "\"/>"
        "  <network id=\"Net\">"
        "    <population id=\"Pop\" component=\"dummyCellInstance\" size=\"2\"/>"
        "    <projection id=\"Proj\" presynapticPopulation=\"Pop\" postsynapticPopulation=\"Pop\" synapse=\"alphaSynapseInstance\">"
        "      <connection id=\"0\" preCellId=\"Pop/0/dummyCellInstance\" postCellId=\"Pop/1/dummyCellInstance\"/>"
        "    </projection>"
        "  </network>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_synapse_lowering_alpha_acceptance_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"SynapseLoweringAlphaAcceptanceTop\">"
        "  <include href=\"spikecorec_synapse_lowering_alpha_acceptance_content.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

// Hand-forward-Euler-integrates the SAME two coupled ODEs alphaCurrentSynapse's own real Dynamics
// declare (`dJ/dt = -J/tau`, `dI/dt = (e*J - I)/tau`, `OnStart`: I=J=0, `OnEvent`: `J += weight *
// ibase`), matching -- statement for statement -- the exact discretization synapse_lowering.cpp's
// own lowering now produces (verified by reading gpu_source.cpp's emit_loadedge/emit_accedge_body
// and master_kernel.cpp's own step_tick dispatch order, see this test file's own header comment on
// the acceptance test below for the full derivation):
//   - a presynaptic spike's `OnEvent` bump to `J` is applied BEFORE this SAME tick's own I/J
//     integration reads it (dispatch_synapse_delivery_events always runs immediately before
//     dispatch_synapse_integrate_edges, same tick, master_kernel.cpp's own step_tick) --
//     accumulate the bump into `state_j` first, every tick, unconditionally (a no-op except on
//     `spike_tick`).
//   - `I`'s own general forward-Euler fallback reads `J`'s CURRENT (just-bumped-if-applicable,
//     not-yet-decayed-this-tick) value and `I`'s own CURRENT (not-yet-updated) value, computes
//     `dt * rhs`, and the resulting NEW `I` is what synapse_lowering.cpp's own DerivedVariable
//     `i = I` exposes -- i.e. THIS tick's `network_inputs` contribution is the just-integrated
//     (not the stale, pre-tick) `I`.
//   - `J`'s own recognized closed-form decay shape then decays the SAME (just-bumped) `J` value
//     via the exact exponential `J *= exp(-dt/tau)` (`expdecay`, not a linear forward-Euler
//     approximation) -- this becomes the tick-start `J` the NEXT tick's own integration reads.
struct AlphaCurrentSynapseForwardEulerResult {
    spikecorec::Vector<f64> exposed_current; // one value per tick -- the network_inputs contribution that tick
};

AlphaCurrentSynapseForwardEulerResult run_alpha_current_synapse_forward_euler(
    f64 tau_seconds, f64 ibase_amperes, f64 weight, f64 dt_seconds, s64 tick_count, s64 spike_tick
) {
    AlphaCurrentSynapseForwardEulerResult result;
    result.exposed_current.reserve((usize)tick_count);

    const f64 euler_constant = 2.7182818284590451; // the real ComponentType's own literal (Synapses.xml)
    f64 state_i = 0.0; // OnStart: I = 0
    f64 state_j = 0.0; // OnStart: J = 0

    for (s64 tick = 0; tick < tick_count; ++tick) {
        if (tick == spike_tick) state_j += weight * ibase_amperes; // OnEvent, delivered before this tick's own integrate
        f64 delta_i = dt_seconds * (euler_constant * state_j - state_i) / tau_seconds;
        f64 new_state_i = state_i + delta_i;
        result.exposed_current.push_back(new_state_i); // DerivedVariable i = I, read AFTER I's own update
        state_j *= std::exp(-dt_seconds / tau_seconds);  // J's own closed-form decay over this dt
        state_i = new_state_i;
    }
    return result;
}

} // namespace

// ── expOneSynapse (real, conductance-based) ──────────────────────────────

TEST(SynapseLoweringPerEdge, lowers_exp_one_synapse_conductance_based) {
    TypeLibraryEntry entry = build_synapse_type_library_entry(
        "exp_one_conductance", "", "expOneSynapse",
        "gbase=\"1nS\" erev=\"0mV\" tauDecay=\"3ms\" weight=\"2\"");

    ASSERT_EQ(entry.category, TypeLibraryCategory::Synapse);
    ASSERT_TRUE(entry.is_conductance_based);

    IrProgram program = lower_synapse_to_ir(entry);
    EXPECT_EQ(program.component_type_name, "expOneSynapse");

    String expected =
        ".alloc\n"
        "  require v from postsynaptic\n"
        "  peredge g\n"
        "  param tauDecay = 0.0030000000000000001\n"
        "  param gbase = 1.0000000000000001e-09\n"
        "  param erev = 0\n"
        "  param weight = 2\n"
        "  expose g\n"
        "  expose i\n"
        ".tick\n"
        "  @deliver\n"
        "    onevent in {\n"
        "      mul t0, weight, gbase\n"
        "      accedge g@edge, t0\n"
        "    }\n"
        "  @integrate\n"
        "    forall neuron_in {\n"
        "      loadedge edge_g_old, g@edge\n"
        "      expdecay edge_g, edge_g_old, tauDecay\n"
        "      sub edge_g_delta, edge_g, edge_g_old\n"
        "      accedge g@edge, edge_g_delta\n"
        "      sub i, erev, v\n"
        "      mul i, edge_g, i\n"
        "      add network_inputs, network_inputs, i\n"
        "    }\n";

    // Conductance-based: `i = g*(erev-v)` (arch §3.5's own invariant -- "no
    // special cell-side case" -- is exactly this: the synapse itself
    // computes `g*(erev-v)` via `require v` before writing network_inputs),
    // realized through the sole per-edge shape every synapse now uses
    // (arch §4.3 design revision). `g`'s TimeDerivative (`-g/tauDecay`)
    // matches the recognized linear-decay shape (target 0), so it IS now
    // lowered via the accumulate-only read-decay-writeback-delta pattern
    // (`loadedge` the old value, `expdecay` it, `accedge` the delta back --
    // see synapse_lowering.h) before the finished-current computation reads
    // the now-decayed `edge_g`.
    EXPECT_EQ(print_ir_program(program), expected);
}

// ── ExpOneCurrentSynapse (custom, current-based) ─────────────────────────

TEST(SynapseLoweringPerEdge, lowers_exp_one_synapse_current_based) {
    TypeLibraryEntry entry = build_synapse_type_library_entry(
        "exp_one_current", EXP_ONE_CURRENT_SYNAPSE_COMPONENT_TYPE, "ExpOneCurrentSynapse",
        "tau=\"5ms\" weight=\"3\"");

    ASSERT_EQ(entry.category, TypeLibraryCategory::Synapse);
    ASSERT_FALSE(entry.is_conductance_based);

    IrProgram program = lower_synapse_to_ir(entry);
    EXPECT_EQ(program.component_type_name, "ExpOneCurrentSynapse");

    String expected =
        ".alloc\n"
        "  peredge g\n"
        "  param tau = 0.0050000000000000001\n"
        "  param weight = 3\n"
        "  expose g\n"
        "  expose i\n"
        ".tick\n"
        "  @deliver\n"
        "    onevent in {\n"
        "      accedge g@edge, weight\n"
        "    }\n"
        "  @integrate\n"
        "    forall neuron_in {\n"
        "      loadedge edge_g_old, g@edge\n"
        "      expdecay edge_g, edge_g_old, tau\n"
        "      sub edge_g_delta, edge_g, edge_g_old\n"
        "      accedge g@edge, edge_g_delta\n"
        "      mov i, edge_g\n"
        "      add network_inputs, network_inputs, i\n"
        "    }\n";

    // Current-based: the trivial identity `i = g` DerivedVariable is lowered
    // through the exact same generic `forall`/`loadedge` machinery a
    // conductance-based synapse's `g*(erev-v)` uses -- no special case for
    // either (see synapse_lowering.cpp's own comment). The `@deliver`
    // onevent is BYTE-identical to the locked IR spec's own illustrative
    // `accedge g@edge, weight` (the increment here is a bare Property leaf,
    // so emit_expression resolves it with zero extra instructions). `g`'s
    // TimeDerivative (`-g/tau`) matches the recognized linear-decay shape, so
    // it decays via read-decay-writeback-delta before the trivial `i = g`
    // identity is computed (the one extra `mov i, edge_g` is the same
    // class of cosmetic-only deviation cell_lowering_tests.cpp's own
    // PlainLifCell test already documents for `.alloc`'s literal-vs-bare-param
    // difference -- a genuinely trivial identity DerivedVariable still
    // costs one harmless `mov`).
    EXPECT_EQ(print_ir_program(program), expected);
}

// ── TestExpTwoSynapse (custom, conductance-based, two state variables) ───

TEST(SynapseLoweringPerEdge, lowers_exp_two_synapse) {
    TypeLibraryEntry entry = build_synapse_type_library_entry(
        "exp_two", TEST_EXP_TWO_SYNAPSE_COMPONENT_TYPE, "TestExpTwoSynapse",
        "gbase=\"2nS\" erev=\"0mV\" tauRise=\"1ms\" tauDecay=\"10ms\" waveformFactor=\"1.5\" weight=\"2\"");

    ASSERT_EQ(entry.category, TypeLibraryCategory::Synapse);
    ASSERT_TRUE(entry.is_conductance_based);

    IrProgram program = lower_synapse_to_ir(entry);
    EXPECT_EQ(program.component_type_name, "TestExpTwoSynapse");

    String expected =
        ".alloc\n"
        "  require v from postsynaptic\n"
        "  peredge A\n"
        "  peredge B\n"
        "  param tauRise = 0.001\n"
        "  param tauDecay = 0.01\n"
        "  param waveformFactor = 1.5\n"
        "  param gbase = 2.0000000000000001e-09\n"
        "  param erev = 0\n"
        "  param weight = 2\n"
        "  expose g\n"
        "  expose i\n"
        ".tick\n"
        "  @deliver\n"
        "    onevent in {\n"
        "      mul t0, weight, waveformFactor\n"
        "      accedge A@edge, t0\n"
        "      mul t0, weight, waveformFactor\n"
        "      accedge B@edge, t0\n"
        "    }\n"
        "  @integrate\n"
        "    forall neuron_in {\n"
        "      loadedge edge_A_old, A@edge\n"
        "      expdecay edge_A, edge_A_old, tauRise\n"
        "      sub edge_A_delta, edge_A, edge_A_old\n"
        "      accedge A@edge, edge_A_delta\n"
        "      loadedge edge_B_old, B@edge\n"
        "      expdecay edge_B, edge_B_old, tauDecay\n"
        "      sub edge_B_delta, edge_B, edge_B_old\n"
        "      accedge B@edge, edge_B_delta\n"
        "      sub g, edge_B, edge_A\n"
        "      mul g, gbase, g\n"
        "      sub i, erev, v\n"
        "      mul i, g, i\n"
        "      add network_inputs, network_inputs, i\n"
        "    }\n";

    // Two state variables both become their own `peredge` slot, each
    // independently decayed via its own read-decay-writeback-delta sequence
    // (A's `-A/tauRise`, B's `-B/tauDecay`, both matching the recognized
    // linear-decay shape) before the plain-value DerivedVariable `g` is
    // computed from the now-decayed `edge_A`/`edge_B` registers -- a case
    // none of the single-state-variable fixtures above exercise; the
    // finished-current DerivedVariable `i` reads `g` afterward -- declaration
    // order alone gets this right, no dependency analysis needed (same
    // convention cell_lowering.cpp's own DerivedVariable iteration relies
    // on).
    EXPECT_EQ(print_ir_program(program), expected);
}

// ── TestNmdaSynapse (custom, conductance-based, per-edge) ────────────────

TEST(SynapseLoweringPerEdge, lowers_nmda_style_synapse) {
    TypeLibraryEntry entry = build_synapse_type_library_entry(
        "nmda", TEST_NMDA_SYNAPSE_COMPONENT_TYPE, "TestNmdaSynapse",
        "gbase=\"1nS\" erev=\"0mV\" tau=\"50ms\" weight=\"1\"");

    ASSERT_EQ(entry.category, TypeLibraryCategory::Synapse);
    ASSERT_TRUE(entry.is_conductance_based);

    IrProgram program = lower_synapse_to_ir(entry);
    EXPECT_EQ(program.component_type_name, "TestNmdaSynapse");

    String expected =
        ".alloc\n"
        "  require v from postsynaptic\n"
        "  peredge g\n"
        "  param tau = 0.050000000000000003\n"
        "  param gbase = 1.0000000000000001e-09\n"
        "  param erev = 0\n"
        "  param weight = 1\n"
        "  expose g\n"
        "  expose i\n"
        ".tick\n"
        "  @deliver\n"
        "    onevent in {\n"
        "      accedge g@edge, weight\n"
        "    }\n"
        "  @integrate\n"
        "    forall neuron_in {\n"
        "      loadedge edge_g_old, g@edge\n"
        "      expdecay edge_g, edge_g_old, tau\n"
        "      sub edge_g_delta, edge_g, edge_g_old\n"
        "      accedge g@edge, edge_g_delta\n"
        "      sub i, erev, v\n"
        "      mul i, edge_g, i\n"
        "      add network_inputs, network_inputs, i\n"
        "    }\n";

    // Matches the locked IR spec's own NMDA example (§4) closely: `peredge g`
    // + `onevent in { accedge g@edge, weight }` + `forall neuron_in { loadedge
    // ...; ...; add network_inputs,network_inputs,... }`, plus the read-decay-
    // writeback-delta sequence for `g`'s TimeDerivative (`-g/tau`, which
    // matches the recognized linear-decay shape) that the accumulate-only
    // per-edge storage requires (arch §4.3; see synapse_lowering.h). Differs
    // from the spec's own illustrative (decay-eliding) example only
    // cosmetically in register naming (`edge_g`/`i` here vs. the spec's own
    // `t0`/`t1` reuse).
    EXPECT_EQ(print_ir_program(program), expected);
}

// ── TestNonDecayingSynapse (custom, non-decay-shaped TimeDerivative) ─────

// A per-edge synapse whose `TimeDerivative` (`1 / tau`, a constant charging
// rate) does NOT match the recognized linear-decay shape -- exercises the
// general forward-Euler fallback in its simplest possible shape (the RHS
// references no OTHER per-edge state variable, unlike alphaCurrentSynapse's
// own `I`/`J` -- see the dedicated alphaCurrentSynapse tests below for that
// cross-reference case). `g`'s TimeDerivative used to be silently dropped
// here (only a warning, no `.tick` lowering at all); it is now actually
// integrated: `loadedge` the current value, evaluate `1/tau` into a fresh
// temporary, scale by `dt`, `accedge` the delta, and update the `edge_g`
// register in place so the DerivedVariable `i` below reads the
// freshly-integrated value.
TEST(SynapseLoweringPerEdge, forward_euler_lowers_a_per_edge_time_derivative_that_is_not_decay_shaped) {
    TypeLibraryEntry entry = build_synapse_type_library_entry(
        "non_decaying", TEST_NON_DECAYING_SYNAPSE_COMPONENT_TYPE, "TestNonDecayingSynapse",
        "gbase=\"1nS\" erev=\"0mV\" tau=\"50ms\" weight=\"1\"");
    ASSERT_TRUE(std::holds_alternative<SynapseType>(entry.dynamics.flattened));
    const SynapseType &synapse = std::get<SynapseType>(entry.dynamics.flattened);
    ASSERT_EQ(synapse.time_derivatives.size(), 1u);
    ASSERT_EQ(synapse.time_derivatives[0].variable, "g");

    IrProgram program = lower_synapse_to_ir(entry);
    String rendered_program = print_ir_program(program);

    String expected =
        ".alloc\n"
        "  require v from postsynaptic\n"
        "  peredge g\n"
        "  param tau = 0.050000000000000003\n"
        "  param gbase = 1.0000000000000001e-09\n"
        "  param erev = 0\n"
        "  param weight = 1\n"
        "  expose g\n"
        "  expose i\n"
        ".tick\n"
        "  @deliver\n"
        "    onevent in {\n"
        "      accedge g@edge, weight\n"
        "    }\n"
        "  @integrate\n"
        "    forall neuron_in {\n"
        "      loadedge edge_g, g@edge\n"
        "      div t0, 1, tau\n"
        "      mul t0, t0, dt\n"
        "      accedge g@edge, t0\n"
        "      add edge_g, edge_g, t0\n"
        "      sub i, erev, v\n"
        "      mul i, edge_g, i\n"
        "      add network_inputs, network_inputs, i\n"
        "    }\n";
    EXPECT_EQ(rendered_program, expected);
}

// ── Error handling ────────────────────────────────────────────────────────

TEST(SynapseLoweringErrors, throws_when_lowering_a_non_synapse_type_library_entry) {
    write_temp_file("spikecorec_synapse_lowering_cell_only_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"SynapseLoweringCellOnlyContent\">"
        + DUMMY_CELL_COMPONENT_TYPE +
        "  <DummyCell id=\"dummyCellInstance\" C=\"1.0e-10\"/>"
        "  <network id=\"Net\">"
        "    <population id=\"Pop\" component=\"dummyCellInstance\" size=\"1\"/>"
        "  </network>"
        "</neuroml>");
    String top_path = write_temp_file("spikecorec_synapse_lowering_cell_only_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"SynapseLoweringCellOnlyTop\">"
        "  <include href=\"spikecorec_synapse_lowering_cell_only_content.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    ModelSpecification specification = build_model_specification(resolved);
    const TypeLibraryEntry &cell_entry = type_library_entry_for(specification, "dummyCellInstance");
    ASSERT_EQ(cell_entry.category, TypeLibraryCategory::Cell);

    EXPECT_THROW(lower_synapse_to_ir(cell_entry), std::runtime_error);
}

TEST(SynapseLoweringErrors, lower_all_synapse_types_skips_non_synapse_type_library_entries) {
    write_temp_file("spikecorec_synapse_lowering_mixed_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"SynapseLoweringMixedContent\">"
        + DUMMY_CELL_COMPONENT_TYPE +
        "  <DummyCell id=\"dummyCellInstance\" C=\"1.0e-10\"/>"
        "  <expOneSynapse id=\"synapseInstance\" gbase=\"1nS\" erev=\"0mV\" tauDecay=\"3ms\"/>"
        "  <network id=\"Net\">"
        "    <population id=\"Pop\" component=\"dummyCellInstance\" size=\"2\"/>"
        "    <projection id=\"Proj\" presynapticPopulation=\"Pop\" postsynapticPopulation=\"Pop\" synapse=\"synapseInstance\">"
        "      <connection id=\"0\" preCellId=\"Pop/0/dummyCellInstance\" postCellId=\"Pop/1/dummyCellInstance\"/>"
        "    </projection>"
        "  </network>"
        "</neuroml>");
    String top_path = write_temp_file("spikecorec_synapse_lowering_mixed_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"SynapseLoweringMixedTop\">"
        "  <include href=\"spikecorec_synapse_lowering_mixed_content.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    ModelSpecification specification = build_model_specification(resolved);
    ASSERT_EQ(specification.type_library.size(), 2u); // one cell, one synapse

    spikecorec::Vector<IrProgram> programs = lower_all_synapse_types_to_ir(specification);
    ASSERT_EQ(programs.size(), 1u);
    EXPECT_EQ(programs[0].component_type_name, "expOneSynapse");
}

// ── alphaCurrentSynapse (real, current-based, coupled I/J TimeDerivative) ─
//
// The exact gap a full skeptical ticket audit flagged: `I`'s own TimeDerivative
// (`(2.7182818284590451*J - I)/tau`) does not match the recognized closed-form
// linear-decay shape (its "target" half, `2.7182818284590451*J`, is a compound
// Binary node, not a bare leaf -- detect_linear_decay_shape's own documented
// scope), so it used to be silently dropped, leaving `I` pinned at 0 forever.
// synapse_lowering.cpp's own general forward-Euler fallback now lowers it.

TEST(SynapseLoweringPerEdge, lowers_real_alpha_current_synapse_couples_I_and_J) {
    TypeLibraryEntry entry = build_synapse_type_library_entry(
        "alpha_current", "", "alphaCurrentSynapse", "tau=\"10ms\" ibase=\"1nA\" weight=\"2\"");

    ASSERT_EQ(entry.category, TypeLibraryCategory::Synapse);
    ASSERT_FALSE(entry.is_conductance_based);
    ASSERT_TRUE(std::holds_alternative<SynapseType>(entry.dynamics.flattened));
    const SynapseType &synapse = std::get<SynapseType>(entry.dynamics.flattened);
    ASSERT_EQ(synapse.state_variables.size(), 2u);
    EXPECT_EQ(synapse.state_variables[0].name, "I");
    EXPECT_EQ(synapse.state_variables[1].name, "J");
    ASSERT_EQ(synapse.time_derivatives.size(), 2u);

    IrProgram program = lower_synapse_to_ir(entry);
    String rendered_program = print_ir_program(program);

    String expected =
        ".alloc\n"
        "  peredge I\n"
        "  peredge J\n"
        "  param tau = 0.01\n"
        "  param ibase = 1.0000000000000001e-09\n"
        "  param weight = 2\n"
        "  expose i\n"
        ".tick\n"
        "  @deliver\n"
        "    onevent in {\n"
        "      mul t0, weight, ibase\n"
        "      accedge J@edge, t0\n"
        "    }\n"
        "  @integrate\n"
        "    forall neuron_in {\n"
        "      loadedge edge_I, I@edge\n"
        "      loadedge edge_J, J@edge\n"
        "      mul t0, 2.7182818284590451, edge_J\n"
        "      sub t0, t0, edge_I\n"
        "      div t0, t0, tau\n"
        "      mul t0, t0, dt\n"
        "      accedge I@edge, t0\n"
        "      add edge_I, edge_I, t0\n"
        "      loadedge edge_J_old, J@edge\n"
        "      expdecay edge_J, edge_J_old, tau\n"
        "      sub edge_J_delta, edge_J, edge_J_old\n"
        "      accedge J@edge, edge_J_delta\n"
        "      mov i, edge_I\n"
        "      add network_inputs, network_inputs, i\n"
        "    }\n";

    // Neither state variable was dropped: `J`'s own `-J/tau` matches the recognized closed-form
    // decay shape (the same read-decay-writeback-delta sequence every other decaying fixture in
    // this file produces); `I`'s own RHS does NOT match that shape (its "target" half is a compound
    // `2.7182818284590451*J`, not a bare leaf) but is now lowered via the general forward-Euler
    // fallback instead of being silently dropped -- `J` (referenced by `I`'s own RHS, not yet
    // touched this tick since `J` is declared after `I`) is loaded fresh into `edge_J`, the whole
    // RHS is evaluated into a temporary, scaled by `dt`, `accedge`'d as `I`'s own delta, and
    // `edge_I` updated in place so the finished-current identity `i = I` below reads the
    // just-integrated value, not a permanently-stale one.
    EXPECT_EQ(rendered_program, expected);
}

// ── alphaCurrentSynapse acceptance: real, numeric, end-to-end alpha-shaped current ───────────────
//
// Drives a real 2-neuron network through SpikeEngine's own NML-model constructor (ticket #131's real
// per-edge synapse dispatch machinery -- the SAME machinery master_kernel_tests.cpp's own
// MasterKernelSynapseDispatch tests and exit_model_validation_tests.cpp's own GLIF E/I network use,
// now folded into SpikeEngine itself)
// for enough ticks that a single presynaptic spike produces a measurable, non-zero, alpha-shaped
// postsynaptic current: starts at/near 0, rises, peaks around t=tau after the spike (the classical
// alpha-function kernel's own analytic peak time, not just "eventually nonzero"), then decays back
// down. Compared, tick for tick, against run_alpha_current_synapse_forward_euler's own
// hand-integrated reference (same discretization synapse_lowering.cpp's own fix produces, see that
// function's header comment for the exact derivation).
//
// The postsynaptic neuron's own current is read directly via `network_inputs[target]` right after
// each `step_tick` call, rather than through the postsynaptic cell's own (irrelevant) `v` --
// SpikeEngine's own step_tick sequence (engine.cpp, ported as-is from master_kernel.cpp) drains
// network_inputs to exactly 0 before this tick's fresh synaptic dispatch writes into it (see
// engine.cpp's own "fixed deliver-drain" comment), and alphaCurrentSynapse's `i = I` identity is the
// ONLY edge feeding this neuron, so `network_inputs[target]` right after `step_tick` returns is
// EXACTLY this tick's `I` value -- no separate per-edge read-back machinery (e.g.
// WeightMatrix::get_for_matrix) is needed. The presynaptic spike itself is forced directly via
// `SpikeEngine::force_emit("spike", 0)`, exactly on `spike_tick` -- the same real, minimal capability
// `ModelRuntimeBuffers::emit_port_flags` used to provide before SpikeEngine folded AssembledModel in.
// This is why the network below is built on `DummyCellWithSpikePort`, not plain `DummyCell` --
// see that ComponentType's own doc comment (just above build_alpha_current_synapse_network_
// specification) for a real, newly-surfaced gap this migration discovered: `force_emit` is correctly
// scoped to a model's own REAL, declared emit ports, unlike the old `ModelRuntimeBuffers::
// emit_port_flags` map, which `AssembledModel`'s synapse-delivery dispatch read as an arbitrary,
// caller-supplied bag of flags independent of any cell's own declared ports.
// The scalar k^2-tree weight-matrix path is left at DEFAULT_MATRIX_INDEX's own all-ones coefficients
// but `set_constant_weight(0.0f)`'d anyway as an explicit, harmless no-op (ticket #131 already forces
// this dispatch's own weight contribution to 0 whenever `model.projections` is non-empty -- see
// engine.h's own "ticket #131" doc comment -- matching exit_model_validation_tests.cpp's own
// established convention of stating the placeholder explicitly rather than deleting the call).
TEST(SynapseLoweringAlphaCurrentAcceptance, real_alpha_current_synapse_produces_alpha_shaped_current) {
    const String tau_attribute = "10ms";
    const String ibase_attribute = "1nA";
    const String weight_attribute = "2";
    const f64 tau_seconds = 0.01;
    const f64 ibase_amperes = 1e-9;
    const f64 weight = 2.0;
    const f32 dt_seconds = 1e-4f;
    const s64 tick_count = 400;
    const s64 spike_tick = 2;

    ModelSpecification model =
        build_alpha_current_synapse_network_specification(tau_attribute, ibase_attribute, weight_attribute);
    ASSERT_EQ(model.total_neuron_count, 2);
    ASSERT_EQ(model.type_library.size(), 2u); // DummyCellWithSpikePort + alphaCurrentSynapse
    ASSERT_EQ(model.projections.size(), 1u);

    spikecorec::Vector<IrProgram> programs;
    programs.reserve(model.type_library.size());
    for (const auto &entry : model.type_library) {
        if (entry.category == TypeLibraryCategory::Cell) {
            programs.push_back(lower_cell_to_ir(entry));
        } else {
            ASSERT_EQ(entry.category, TypeLibraryCategory::Synapse);
            programs.push_back(lower_synapse_to_ir(entry));
        }
    }

    // SpikeEngine's own ModelSpecification constructor builds `engine.weights` itself (via
    // `nml::build_weight_matrix_from_projections`, rank=1, straight from `model.projections`' single
    // 0->1 connection -- the identical adjacency this test used to build by hand), replacing this
    // test's own previously-separate `allocate_model`/manual-adjacency `WeightMatrix`/`AssembledModel`
    // construction.
    SpikeEngine engine(model, programs, dt_seconds);
    engine.weights.set_constant_weight(0.0f); // explicit no-op -- see this test's own header comment

    spikecorec::Vector<f32> observed_current;
    observed_current.reserve((usize)tick_count);
    for (s64 tick = 0; tick < tick_count; ++tick) {
        if (tick == spike_tick) engine.force_emit("spike", 0); // one presynaptic spike, exactly this tick
        engine.step_tick(dt_seconds, tick, tick + 1);
        observed_current.push_back(engine.network_inputs.get_contents()[1]);
    }

    AlphaCurrentSynapseForwardEulerResult reference = run_alpha_current_synapse_forward_euler(
        tau_seconds, ibase_amperes, weight, (f64)dt_seconds, tick_count, spike_tick);
    ASSERT_EQ(observed_current.size(), reference.exposed_current.size());

    // Tight, tick-for-tick numeric match against the hand-computed reference -- both use the exact
    // same discretization; the only expected divergence is float32 (engine) vs. float64 (reference)
    // rounding, not a magnitude- or shape-class difference.
    for (s64 tick = 0; tick < tick_count; ++tick) {
        f64 reference_value = reference.exposed_current[(usize)tick];
        f64 tolerance = std::fabs(reference_value) * 1e-3 + 1e-13;
        EXPECT_NEAR((f64)observed_current[(usize)tick], reference_value, tolerance) << "tick=" << tick;
    }

    // Qualitative alpha shape: near zero before the spike, rises, peaks, decays back down -- not
    // just "is nonzero at some point".
    for (s64 tick = 0; tick < spike_tick; ++tick) {
        EXPECT_NEAR(observed_current[(usize)tick], 0.0f, 1e-15f) << "tick=" << tick;
    }

    usize peak_tick_index = 0;
    for (usize tick_index = 0; tick_index < observed_current.size(); ++tick_index) {
        if (observed_current[tick_index] > observed_current[peak_tick_index]) peak_tick_index = tick_index;
    }
    EXPECT_GT(peak_tick_index, (usize)spike_tick)
        << "current should rise after the spike, not peak immediately";
    EXPECT_LT(peak_tick_index, (usize)tick_count - 1)
        << "current should decay back down before the horizon ends, not still be rising";
    EXPECT_GT(observed_current[peak_tick_index], 1e-10f)
        << "peak current should be a real, measurable magnitude, not near-zero noise";
    EXPECT_LT(observed_current[(usize)tick_count - 1], observed_current[peak_tick_index] * 0.5f)
        << "current should have decayed back down substantially by the end of the horizon";
}
