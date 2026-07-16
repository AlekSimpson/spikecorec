#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <gtest/gtest.h>
#include <spdlog/sinks/ostream_sink.h>

#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/synapse_lowering.h"
#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── Synapse dynamics -> IR lowering tests (ticket #51 [B3]) ──────────────
//
// Covers the three shapes ticket #51 scopes: aggregatable current-based,
// aggregatable conductance-based, and per-edge (all conductance-based).
// Where the real vendored std-lib ComponentType is directly usable
// (`expOneSynapse`, `alphaCurrentSynapse` -- neither needs anything this
// front-end can't already parse), the fixture below references it directly
// by tag, with no inline `<ComponentType>` declaration, exactly like
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
//     `DerivedParameter` -- sidestepping the gap without masking it.
//   - `blockingPlasticSynapse` (real, the NMDA target): its real Dynamics
//     compose child `plasticityMechanisms`/`blockMechanisms` components via
//     `select`/`reduce="multiply"` DerivedVariables -- full sub-component
//     composition, explicitly out of this ticket's scope (arch §4.3's own
//     NMDA example elides the Mg-block the same way, and blockingPlasticSynapse
//     itself extends expTwoSynapse so also inherits the waveformFactor gap
//     above). `TestNmdaSynapse` below reproduces the IR spec §4 NMDA
//     example's own simplified shape almost exactly (single decaying
//     conductance `g`, `i = g*(erev-v)`, one `Children` declaration so
//     ticket #7's own `synapse_is_aggregatable` classifier correctly marks
//     it per-edge -- matching the real blockingPlasticSynapse's actual
//     per-edge classification, just without composing the elided mechanism).
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

// Temporarily attaches an in-memory sink to the shared `log::logger()` so a
// test can assert on a specific warning firing (review follow-up on ticket
// #51: the per-edge-synapse-declares-a-TimeDerivative diagnostic). Removes
// itself in the destructor -- the sink holds a reference to `captured_text_`,
// a member of THIS object, so it must not outlive it.
class ScopedLogCapture {
public:
    ScopedLogCapture() : sink_(std::make_shared<spdlog::sinks::ostream_sink_mt>(captured_text_)) {
        log::logger().sinks().push_back(sink_);
    }

    ~ScopedLogCapture() {
        auto &sinks = log::logger().sinks();
        sinks.erase(std::remove(sinks.begin(), sinks.end(), sink_), sinks.end());
    }

    String text() const { return captured_text_.str(); }

private:
    std::ostringstream captured_text_;
    std::shared_ptr<spdlog::sinks::ostream_sink_mt> sink_;
};

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
// bound instance into a minimal one-neuron self-loop network, runs it
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
        "    <population id=\"Pop\" component=\"dummyCellInstance\" size=\"1\"/>"
        "    <projection id=\"SelfLoop\" presynapticPopulation=\"Pop\" postsynapticPopulation=\"Pop\" synapse=\"synapseInstance\">"
        "      <connection id=\"0\" preCellId=\"Pop/0/dummyCellInstance\" postCellId=\"Pop/0/dummyCellInstance\"/>"
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

} // namespace

// ── expOneSynapse (real, conductance-based, aggregatable) ────────────────

TEST(SynapseLoweringAggregatable, lowers_exp_one_synapse_conductance_based) {
    TypeLibraryEntry entry = build_synapse_type_library_entry(
        "exp_one_conductance", "", "expOneSynapse",
        "gbase=\"1nS\" erev=\"0mV\" tauDecay=\"3ms\" weight=\"2\"");

    ASSERT_EQ(entry.category, TypeLibraryCategory::Synapse);
    ASSERT_TRUE(entry.is_conductance_based);
    ASSERT_TRUE(entry.is_aggregatable);

    IrProgram program = lower_synapse_to_ir(entry);
    EXPECT_EQ(program.component_type_name, "expOneSynapse");

    String expected =
        ".alloc\n"
        "  require v from postsynaptic\n"
        "  accum g : f32\n"
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
        "    expdecay g, g, tauDecay\n"
        "    sub i, erev, v\n"
        "    mul i, g, i\n"
        "    add network_inputs, network_inputs, i\n";

    // Conductance-based: `i = g*(erev-v)` (arch §3.5's own invariant -- "no
    // special cell-side case" -- is exactly this: the synapse itself
    // computes `g*(erev-v)` via `require v` before writing network_inputs).
    EXPECT_EQ(print_ir_program(program), expected);
}

// ── ExpOneCurrentSynapse (custom, current-based, aggregatable) ───────────

TEST(SynapseLoweringAggregatable, lowers_exp_one_synapse_current_based) {
    TypeLibraryEntry entry = build_synapse_type_library_entry(
        "exp_one_current", EXP_ONE_CURRENT_SYNAPSE_COMPONENT_TYPE, "ExpOneCurrentSynapse",
        "tau=\"5ms\" weight=\"3\"");

    ASSERT_EQ(entry.category, TypeLibraryCategory::Synapse);
    ASSERT_FALSE(entry.is_conductance_based);
    ASSERT_TRUE(entry.is_aggregatable);

    IrProgram program = lower_synapse_to_ir(entry);
    EXPECT_EQ(program.component_type_name, "ExpOneCurrentSynapse");

    String expected =
        ".alloc\n"
        "  accum g : f32\n"
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
        "    expdecay g, g, tau\n"
        "    mov i, g\n"
        "    add network_inputs, network_inputs, i\n";

    // Matches the locked IR spec's own "expOne -- current-based, aggregatable
    // synapse" example (§4) almost exactly: the `@deliver` onevent is
    // BYTE-identical to the spec's `accedge g@edge, weight` (the increment
    // here is a bare Property leaf, so emit_expression resolves it with zero
    // extra instructions). `@integrate` has one extra `mov i, g` the spec's
    // own illustrative text elides (it treats the trivial identity
    // `i = g` as needing no instruction at all) -- this lowering's
    // DerivedVariable handling is uniform/generic (every DerivedVariable,
    // "i" included, is computed into its own name, same as cell_lowering.cpp's
    // convention), so a genuinely trivial identity DerivedVariable still
    // costs one harmless `mov`, exactly the same class of cosmetic-only
    // deviation cell_lowering_tests.cpp's own PlainLifCell test already
    // documents for `.alloc`'s literal-vs-bare-param difference.
    EXPECT_EQ(print_ir_program(program), expected);
}

// ── TestExpTwoSynapse (custom, conductance-based, aggregatable, two state
// variables) ──────────────────────────────────────────────────────────────

TEST(SynapseLoweringAggregatable, lowers_exp_two_synapse) {
    TypeLibraryEntry entry = build_synapse_type_library_entry(
        "exp_two", TEST_EXP_TWO_SYNAPSE_COMPONENT_TYPE, "TestExpTwoSynapse",
        "gbase=\"2nS\" erev=\"0mV\" tauRise=\"1ms\" tauDecay=\"10ms\" waveformFactor=\"1.5\" weight=\"2\"");

    ASSERT_EQ(entry.category, TypeLibraryCategory::Synapse);
    ASSERT_TRUE(entry.is_conductance_based);
    ASSERT_TRUE(entry.is_aggregatable);

    IrProgram program = lower_synapse_to_ir(entry);
    EXPECT_EQ(program.component_type_name, "TestExpTwoSynapse");

    String expected =
        ".alloc\n"
        "  require v from postsynaptic\n"
        "  accum A : f32\n"
        "  accum B : f32\n"
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
        "    expdecay A, A, tauRise\n"
        "    expdecay B, B, tauDecay\n"
        "    sub g, B, A\n"
        "    mul g, gbase, g\n"
        "    sub i, erev, v\n"
        "    mul i, g, i\n"
        "    add network_inputs, network_inputs, i\n";

    // Two state variables both become their own `accum` slot (arch §4.2);
    // the plain-value DerivedVariable `g` is computed (from the just-decayed
    // A/B) before the finished-current DerivedVariable `i` that reads it --
    // declaration order alone gets this right, no dependency analysis
    // needed (same convention cell_lowering.cpp's own DerivedVariable
    // iteration relies on).
    EXPECT_EQ(print_ir_program(program), expected);
}

// ── alphaCurrentSynapse (real, current-based, aggregatable) ──────────────

TEST(SynapseLoweringAggregatable, lowers_alpha_current_synapse) {
    TypeLibraryEntry entry = build_synapse_type_library_entry(
        "alpha_current", "", "alphaCurrentSynapse",
        "tau=\"2ms\" ibase=\"0.1nA\" weight=\"2\"");

    ASSERT_EQ(entry.category, TypeLibraryCategory::Synapse);
    ASSERT_FALSE(entry.is_conductance_based);
    ASSERT_TRUE(entry.is_aggregatable);

    IrProgram program = lower_synapse_to_ir(entry);
    EXPECT_EQ(program.component_type_name, "alphaCurrentSynapse");

    String expected =
        ".alloc\n"
        "  accum I : f32\n"
        "  accum J : f32\n"
        "  param tau = 0.002\n"
        "  param ibase = 1.0000000000000002e-10\n"
        "  param weight = 2\n"
        "  expose i\n"
        ".tick\n"
        "  @deliver\n"
        "    onevent in {\n"
        "      mul t0, weight, ibase\n"
        "      accedge J@edge, t0\n"
        "    }\n"
        "  @integrate\n"
        "    mul t0, 2.7182818284590451, J\n"
        "    sub t0, t0, I\n"
        "    div t0, t0, tau\n"
        "    mul t0, t0, dt\n"
        "    add I, I, t0\n"
        "    expdecay J, J, tau\n"
        "    mov i, I\n"
        "    add network_inputs, network_inputs, i\n";

    // `I`'s TimeDerivative `(2.7182818284590451*J - I)/tau` doesn't match
    // the linear-decay shape (its target `2.7182818284590451*J` isn't a bare
    // leaf), so it takes the forward-Euler fallback -- exactly like a GLIF
    // cell's own non-decay-shaped TimeDerivative would.
    EXPECT_EQ(print_ir_program(program), expected);
}

// ── TestNmdaSynapse (custom, conductance-based, per-edge) ────────────────

TEST(SynapseLoweringPerEdge, lowers_nmda_style_synapse) {
    TypeLibraryEntry entry = build_synapse_type_library_entry(
        "nmda", TEST_NMDA_SYNAPSE_COMPONENT_TYPE, "TestNmdaSynapse",
        "gbase=\"1nS\" erev=\"0mV\" tau=\"50ms\" weight=\"1\"");

    ASSERT_EQ(entry.category, TypeLibraryCategory::Synapse);
    ASSERT_TRUE(entry.is_conductance_based);
    ASSERT_FALSE(entry.is_aggregatable);

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
        "      loadedge edge_g, g@edge\n"
        "      sub i, erev, v\n"
        "      mul i, edge_g, i\n"
        "      add network_inputs, network_inputs, i\n"
        "    }\n";

    // Matches the locked IR spec's own NMDA example (§4) almost exactly:
    // `peredge g` + `onevent in { accedge g@edge, weight }` +
    // `forall neuron_in { loadedge ...; ...; add network_inputs,network_inputs,... }`.
    // Differs only cosmetically in register naming (`edge_g`/`i` here vs. the
    // spec's own `t0`/`t1` reuse) and in omitting an explicit per-edge decay
    // instruction -- see synapse_lowering.h's header comment for why: `g`'s
    // TimeDerivative (`-g/tau`) is intentionally NOT lowered for a `peredge`
    // variable -- arch §4.3's shared-basis scheme is pure memory compression
    // (Read/Update/Refit, none of which integrates an ODE), so per-edge
    // time-evolution is simply deferred/unspecified in Phase 1, matching the
    // provisional IR spec's own NMDA example (it declares `param tau` but
    // never references it in `.tick`, and never decays its per-edge `g`
    // either). `TestNmdaSynapse` DOES declare this TimeDerivative (see the
    // fixture above), so this test also exercises the accompanying
    // diagnostic (warns_when_per_edge_synapse_declares_a_time_derivative,
    // below, asserts the warning text itself -- this test only pins the
    // resulting `.tick` shape).
    EXPECT_EQ(print_ir_program(program), expected);
}

// The `TestNmdaSynapse` fixture above declares a `TimeDerivative` for its
// per-edge `g` (`-g / tau`) that the lowering intentionally does not emit
// into `.tick` (see this file's and synapse_lowering.h's comments on why).
// Every OTHER unsupported shape in synapse_lowering.cpp throws; this one
// silent drop instead logs a warning (review follow-up on ticket #51) so a
// future real per-edge synapse with genuine decay dynamics is at least
// observable, not silently lost. Asserts both halves: the warning fires,
// AND the resulting `.tick` is unchanged (still the same shape as
// `lowers_nmda_style_synapse` above -- the diagnostic is additive, not a
// behavior change).
TEST(SynapseLoweringPerEdge, warns_when_per_edge_synapse_declares_a_time_derivative) {
    TypeLibraryEntry entry = build_synapse_type_library_entry(
        "nmda_warning", TEST_NMDA_SYNAPSE_COMPONENT_TYPE, "TestNmdaSynapse",
        "gbase=\"1nS\" erev=\"0mV\" tau=\"50ms\" weight=\"1\"");
    ASSERT_FALSE(entry.is_aggregatable);
    ASSERT_TRUE(std::holds_alternative<SynapseType>(entry.dynamics.flattened));
    const SynapseType &synapse = std::get<SynapseType>(entry.dynamics.flattened);
    ASSERT_EQ(synapse.time_derivatives.size(), 1u);
    ASSERT_EQ(synapse.time_derivatives[0].variable, "g");

    ScopedLogCapture log_capture;
    IrProgram program = lower_synapse_to_ir(entry);
    String captured_log_text = log_capture.text();

    EXPECT_NE(captured_log_text.find("TestNmdaSynapse"), String::npos);
    EXPECT_NE(captured_log_text.find("'g'"), String::npos);
    EXPECT_NE(captured_log_text.find("TimeDerivative"), String::npos);

    // Behavior is unchanged: the TimeDerivative is still correctly omitted
    // (no `expdecay` anywhere, and `tau` is declared in `.alloc` but never
    // read in `.tick`), exactly matching lowers_nmda_style_synapse's own
    // pinned expected text above.
    String rendered_program = print_ir_program(program);
    EXPECT_NE(rendered_program.find("peredge g"), String::npos);
    EXPECT_NE(rendered_program.find("param tau"), String::npos);
    EXPECT_EQ(rendered_program.find("expdecay"), String::npos);

    usize tick_section_start = rendered_program.find(".tick");
    ASSERT_NE(tick_section_start, String::npos);
    String tick_section = rendered_program.substr(tick_section_start);
    EXPECT_EQ(tick_section.find("tau"), String::npos);
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
        "    <population id=\"Pop\" component=\"dummyCellInstance\" size=\"1\"/>"
        "    <projection id=\"SelfLoop\" presynapticPopulation=\"Pop\" postsynapticPopulation=\"Pop\" synapse=\"synapseInstance\">"
        "      <connection id=\"0\" preCellId=\"Pop/0/dummyCellInstance\" postCellId=\"Pop/0/dummyCellInstance\"/>"
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

    Vector<IrProgram> programs = lower_all_synapse_types_to_ir(specification);
    ASSERT_EQ(programs.size(), 1u);
    EXPECT_EQ(programs[0].component_type_name, "expOneSynapse");
}
