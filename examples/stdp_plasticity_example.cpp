// ── Example: wiring an NML STDP synapse to the engine's plasticity path ─────────────────────────
//
// The other examples in this directory all load a checked-in `.nml` fixture. This one writes its own
// NeuroML on the fly, because it needs a ComponentType no vendored file provides: the NeuroML
// standard library's `stdpSynapse` is marked "EXAMPLE NOT YET WORKING!!!!" in its own description
// and declares none of the STDP parameters, so there is nothing to load.
//
// That makes this example useful for a second reason: it shows the whole path from a
// **hand-authored ComponentType** through parse → resolve → ModelSpecification, which is what you
// would do to bring your own LEMS dynamics into the engine.
//
// ── What "STDP wiring" actually means here (read before drawing conclusions) ────────────────────
//
// Spike-timing-dependent plasticity is a LOCAL SYNAPTIC mechanism — spike-timing-driven updates to
// individual edge weights. It is a biological simulation feature. It is NOT task learning, gradient
// descent, or anything to do with the U/V rank parameter, which is purely a memory-compression
// scheme.
//
// Detection is structural rather than by ComponentType name: any Synapse ComponentType baking all
// four of `tauPlus`/`tauMinus`/`aPlus`/`aMinus` counts as having an STDP spec. That matches the
// community naming convention and keeps the codegen generic — no hand-written per-type special case.
//
// Two real limitations of the TARGET KERNEL, which this wiring layer maps onto rather than hides:
//
//   1. The stage-7 kernel has exactly one externally-supplied dial, `learning_rate`. Its recency
//      shape (`pow(tick_delta, -3)`) is baked into the kernel source. So only ONE of the four spec
//      values can reach it: `aMinus` is used, because —
//   2. The kernel's update is always a DEPRESSION. `decay_delta` is `-learning_rate * pow(...)`,
//      which is non-positive on every path; there is no branch anywhere that makes it positive. So
//      this is not the two-sided potentiate/depress window textbook STDP describes.
//      `tauPlus`/`tauMinus`/`aPlus` are required for presence detection but cannot influence the
//      math.
//
// ── Two wiring targets: SpikeEngine's hardcoded-LIF path, and its NML mode (ticket #132) ─────────
// `apply_stdp_wiring` is overloaded on two simulation objects (SpikeEngine, and the older
// AssembledModel this ticket's own migration is retiring call sites of). Sections 1-4 below exercise
// the `SpikeEngine&` overload against the original hardcoded-LIF path (Phase-1's own SC-11 API).
// Section 5 exercises the SAME `SpikeEngine&` overload again, but against a SpikeEngine built via
// the NML `ModelSpecification` constructor instead — enable_plasticity/disable_plasticity are
// unified across both construction paths, so no separate overload is needed. See that section's own
// comment for why it needs a SEPARATE two-neuron model (a per-edge-delay ring, not the plain
// one-tick-latency connection sections 1-4 use) to actually call `enable_plasticity` without
// throwing: ticket #131's real per-edge synapse dispatch and ticket #132's STDP both write the
// shared U/V basis, and `SpikeEngine::enable_plasticity` refuses to run alongside the former
// (engine.cpp's own documented guard) unless the model's real per-edge delay forces the delay ring
// (ring_slot_count > 1), which switches off #131's dispatch entirely.
//
// Also note `enable_plasticity` is a no-op on an engine/model that already has plasticity enabled —
// it will not overwrite an existing learning rate. Hence the explicit `disable_plasticity()` calls
// below.
//
// Run:  ./build/examples/stdp_plasticity_example

#include <filesystem>
#include <fstream>
#include <iostream>

#include "spikecorec/core/engine.h"
#include "spikecorec/nml/plasticity_wiring.h"

#include "nml_pipeline_support.h"

using namespace spikecorec;
using namespace spikecorec::nml;
using namespace spikecorec::examples;

// `Vector<...>` is spelled out fully as `spikecorec::Vector<...>` throughout this file, unlike every
// other example here. plasticity_wiring.h pulls in core/engine.h, which has a file-scope
// `using namespace spikecorec::log;` — and that namespace declares its own `Vector` alias template.
// Two alias templates of the same name from two using-directives at the same scope are ambiguous for
// unqualified lookup regardless of expanding to the identical type, so the qualification is forced.
// This mirrors what tests/simple_lif_stdp_network_tests.cpp does for the same reason.

namespace {

// A minimal integrate-and-fire cell, just enough to hang a projection off.
const String DEMO_CELL_COMPONENT_TYPE =
    "  <ComponentType name=\"DemoCell\" extends=\"baseCell\">"
    "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
    "    <Dynamics>"
    "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
    "      <TimeDerivative variable=\"v\" value=\"network_inputs / C\"/>"
    "    </Dynamics>"
    "  </ComponentType>";

// A self-contained, STDP-shaped conductance-based synapse. The four parameter names below are what
// make find_stdp_spec recognize it; the rest is an ordinary exponential-decay synapse.
const String STDP_SYNAPSE_COMPONENT_TYPE =
    "  <ComponentType name=\"DemoStdpSynapse\" extends=\"baseConductanceBasedSynapse\">"
    "    <Property name=\"weight\" dimension=\"none\" defaultValue=\"1\"/>"
    "    <Parameter name=\"tau\" dimension=\"time\"/>"
    "    <Parameter name=\"tauPlus\" dimension=\"time\"/>"
    "    <Parameter name=\"tauMinus\" dimension=\"time\"/>"
    "    <Parameter name=\"aPlus\" dimension=\"none\"/>"
    "    <Parameter name=\"aMinus\" dimension=\"none\"/>"
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

String write_temporary_file(const String &file_name, const String &contents) {
    String path = (std::filesystem::temp_directory_path() / file_name).string();
    std::ofstream output_file(path);
    output_file << contents;
    return path;
}

// Builds a two-neuron network around `synapse_component_type_xml` and runs it through the front-end.
// `connection_element_xml` is the projection's own single `<connection .../>` or
// `<connectionWD .../>` child element, so section 5 below can ask for a real per-edge delay without
// a second, near-duplicate document-assembly function.
//
// The document is split into a content file plus a thin `<include>`-only top file, the same
// convention the checked-in fixtures use: NML_Parser XSD-validates only the top-level file it is
// handed, and raw LEMS ComponentType declarations do not validate against the NeuroML schema.
ModelSpecification build_two_neuron_model(
    const String &fixture_id, const String &synapse_component_type_xml, const String &synapse_tag,
    const String &synapse_instance_attributes,
    const String &connection_element_xml = "<connection id=\"0\" preCellId=\"Pop/0/demoCellInstance\" "
                                            "postCellId=\"Pop/1/demoCellInstance\"/>"
) {
    const String content_file_name = "spikecorec_example_stdp_" + fixture_id + "_content.nml";
    write_temporary_file(content_file_name,
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"StdpExample" + fixture_id + "Content\">"
        + DEMO_CELL_COMPONENT_TYPE +
        "  <DemoCell id=\"demoCellInstance\" C=\"1.0e-10\"/>"
        + synapse_component_type_xml +
        "  <" + synapse_tag + " id=\"synapseInstance\" " + synapse_instance_attributes + "/>"
        "  <network id=\"Net\">"
        "    <population id=\"Pop\" component=\"demoCellInstance\" size=\"2\"/>"
        "    <projection id=\"Proj\" presynapticPopulation=\"Pop\" postsynapticPopulation=\"Pop\" "
        "synapse=\"synapseInstance\">"
        + connection_element_xml +
        "    </projection>"
        "  </network>"
        "</neuroml>");

    String top_path = write_temporary_file("spikecorec_example_stdp_" + fixture_id + "_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"StdpExample" + fixture_id + "Top\">"
        "  <include href=\"" + content_file_name + "\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

} // namespace

int main(int argument_count, char **argument_values) {
    ExampleOptions options = parse_example_options(argument_count, argument_values, ExampleOptions{200, 1e-4f});
    GpuContextScope gpu_context_scope;

    // ── 1. A model whose synapse declares STDP parameters ───────────────────────────────────────
    ModelSpecification stdp_model = build_two_neuron_model(
        "with", STDP_SYNAPSE_COMPONENT_TYPE, "DemoStdpSynapse",
        "gbase=\"1nS\" erev=\"0mV\" tau=\"5ms\" tauPlus=\"20ms\" tauMinus=\"20ms\" aPlus=\"0.01\" aMinus=\"0.012\"");

    spikecorec::Vector<IrProgram> stdp_programs = lower_type_library_to_ir(stdp_model);
    print_model_summary(stdp_model, stdp_programs);
    // The hand-authored ComponentTypes above lower to IR like any vendored type — `--print-ir` shows
    // the `.alloc`/`.tick` sections they produce.
    if (options.print_ir) print_ir_programs(stdp_model, stdp_programs);

    // ── 2. Structural detection ─────────────────────────────────────────────────────────────────
    print_heading("STDP spec detection");
    for (const TypeLibraryEntry &entry : stdp_model.type_library) {
        if (entry.category != TypeLibraryCategory::Synapse) continue;

        std::optional<StdpSpec> spec = find_stdp_spec(entry);
        std::cout << "  " << entry.component_type_name << " → "
                  << (spec.has_value() ? "STDP spec found" : "no STDP spec") << "\n";
        if (!spec.has_value()) continue;

        std::cout << "      tauPlus  = " << format_seconds(spec->tau_plus) << "\n"
                  << "      tauMinus = " << format_seconds(spec->tau_minus) << "\n"
                  << "      aPlus    = " << spec->a_plus << "\n"
                  << "      aMinus   = " << spec->a_minus << "\n"
                  << "\n      maps to learning_rate = " << map_stdp_spec_to_learning_rate(*spec)
                  << "   (aMinus — the kernel's only dial; see this file's header comment)\n";
    }

    // ── 3. Applying the wiring to a live legacy engine (SpikeEngine, Phase 1's original SC-11) ───
    // A SpikeEngine needs its own network; this uses the same two-neuron shape as the model above.
    spikecorec::Vector<spikecorec::Vector<s32>> engine_network = {{1}, {0}};
    SpikeEngine engine(&engine_network, /*shape=*/{2}, /*rank=*/1);

    // enable_plasticity is a no-op when plasticity is already on, so start from a known-off state.
    engine.disable_plasticity();

    print_heading("Legacy SpikeEngine wiring");
    std::cout << "  before apply_stdp_wiring : plasticity_enabled=" << std::boolalpha << engine.plasticity_enabled()
              << "  learning_rate=" << engine.learning_rate << "\n";

    apply_stdp_wiring(stdp_model, engine);

    std::cout << "  after  apply_stdp_wiring : plasticity_enabled=" << engine.plasticity_enabled()
              << "  learning_rate=" << engine.learning_rate << "\n";

    // ── 4. The absence case ─────────────────────────────────────────────────────────────────────
    // Presence of a spec enables plasticity; absence disables it. A plain vendored synapse declares
    // no STDP parameters, so wiring the same engine against it turns plasticity back off.
    ModelSpecification plain_model = build_two_neuron_model(
        "without", /*synapse_component_type_xml=*/"", "expOneSynapse", "gbase=\"1nS\" erev=\"0mV\" tauDecay=\"5ms\"");

    apply_stdp_wiring(plain_model, engine);

    std::cout << "\n  a model whose synapse has no STDP parameters (expOneSynapse):\n"
              << "  after  apply_stdp_wiring : plasticity_enabled=" << engine.plasticity_enabled()
              << "  learning_rate=" << engine.learning_rate << "\n";

    engine.shutdown();

    std::cout << "\nOne global learning_rate exists per engine, so a model declaring STDP on more than\n"
              << "one synapse type cannot represent all of them at once — the first is used and a\n"
              << "warning is logged (run with --verbose to see it). That is a documented Phase-1 limit.\n";

    // ── 5. Applying the wiring to SpikeEngine's NML mode (ticket #132) ─────────────────────────
    print_heading("SpikeEngine (NML mode) wiring (ticket #132)");

    // `stdp_model` above has a real per-edge projection (ticket #131 dispatches its DemoStdpSynapse
    // automatically, the same way every torus/GLIF-E/I example's own expOneSynapse gets dispatched)
    // and has no real per-edge delay configured (ring_slot_count == 1) — exactly the combination
    // SpikeEngine::enable_plasticity documents as unsafe (both would write the shared U/V basis
    // uncoordinated) and refuses. Demonstrated directly rather than silently worked around:
    SpikeEngine stdp_engine_flat(stdp_model, stdp_programs, options.dt_seconds);
    std::cout << "  flat-mode SpikeEngine (real per-edge dispatch active, ticket #131):\n";
    try {
        apply_stdp_wiring(stdp_model, stdp_engine_flat);
        std::cout << "    apply_stdp_wiring did NOT throw — unexpected, see this file's header comment\n";
    } catch (const std::exception &enable_plasticity_error) {
        std::cout << "    apply_stdp_wiring threw, as documented: " << enable_plasticity_error.what() << "\n";
    }

    // The SAME structural model, but wired through a REAL per-edge delay (`<connectionWD delay=...>`)
    // — a real per-edge delay forces the delay ring (ring_slot_count > 1, engine.cpp), which disables
    // #131's synapse dispatch entirely, so `enable_plasticity` is safe here.
    ModelSpecification stdp_ring_model = build_two_neuron_model(
        "ring", STDP_SYNAPSE_COMPONENT_TYPE, "DemoStdpSynapse",
        "gbase=\"1nS\" erev=\"0mV\" tau=\"5ms\" tauPlus=\"20ms\" tauMinus=\"20ms\" aPlus=\"0.01\" aMinus=\"0.012\"",
        "<connectionWD id=\"0\" preCellId=\"Pop/0/demoCellInstance\" postCellId=\"Pop/1/demoCellInstance\" "
        "weight=\"1\" delay=\"5ms\"/>");
    spikecorec::Vector<IrProgram> stdp_ring_programs = lower_type_library_to_ir(stdp_ring_model);

    SpikeEngine stdp_engine_ring(stdp_ring_model, stdp_ring_programs, options.dt_seconds);
    std::cout << "\n  ring-mode SpikeEngine (real per-edge delay, #131's dispatch disabled):\n";
    apply_stdp_wiring(stdp_ring_model, stdp_engine_ring);
    std::cout << "    plasticity_enabled=" << stdp_engine_ring.plasticity_enabled()
              << "  (mapped from this model's own aMinus=0.012 via apply_stdp_wiring)\n";

    std::cout << "\nThe integration mechanism itself — real cell dynamics, a non-trivial per-edge\n"
              << "delay, and active STDP together in one SpikeEngine tick loop — is ticket #132's\n"
              << "own deliverable; see tests/assembled_model_plasticity_tests.cpp for a full run that\n"
              << "steps this exact combination and measures the weight actually depress.\n";

    return 0;
}
