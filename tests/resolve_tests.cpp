#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <variant>
#include <gtest/gtest.h>

#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

namespace {

String write_temp_file(const String &filename, const String &contents) {
    String path = (std::filesystem::temp_directory_path() / filename).string();
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

const ResolvedInstance *find_instance_by_tag(const Vector<ResolvedInstance> &instances, const String &tag) {
    for (const auto &instance : instances) {
        if (instance.tag_name == tag) return &instance;
    }
    return nullptr;
}

// A GLIF-shaped cell that extends a 3-hop ancestor chain with REAL content
// at every hop: GLIF3TestCell -> GLIFBaseCell (user-defined) ->
// baseCellMembPot -> baseSpikingCell -> baseCell (std-lib), so that
// asserting the flattened type carries every ancestor's declarations (not
// just GLIF3TestCell's own) actually exercises the merge across multiple
// hops, through both user-defined and vendored std-lib ancestors. Also
// exercises `Fixed` pinning an INHERITED parameter (`asc_amp1`, declared on
// GLIFBaseCell, pinned from GLIF3TestCell).
//
// The ComponentType defs + instance live in an <include>-d file (like ticket
// #2's own GLIF fixtures): only the top-level file passed to parser.parse()
// is XSD-gated, and the NeuroML2 XSD's Nml2Quantity_* patterns whitelist a
// fixed unit-symbol set per dimension -- fine for real units ("-70mV"), but
// would reject a deliberately-bad unit before RESOLVE ever ran it. Routing
// custom types through <include> keeps this test (and the bad-unit test
// below) exercising the resolve pass itself, not the XSD gate.
String write_glif_chain_fixture() {
    write_temp_file("spikecorec_resolve_glif_chain_types.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"ResolveGlifChainTypes\">"
        "  <ComponentType name=\"GLIFBaseCell\" extends=\"baseCellMembPot\">"
        "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
        "    <Parameter name=\"asc_amp1\" dimension=\"current\"/>"
        "    <Exposure name=\"baseExposure\" dimension=\"none\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <ComponentType name=\"GLIF3TestCell\" extends=\"GLIFBaseCell\">"
        "    <Parameter name=\"gL\" dimension=\"conductance\"/>"
        "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
        "    <Parameter name=\"vth\" dimension=\"voltage\"/>"
        "    <Parameter name=\"vreset\" dimension=\"voltage\"/>"
        "    <Fixed parameter=\"asc_amp1\" value=\"10pA\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"ASC1\" dimension=\"current\"/>"
        "      <TimeDerivative variable=\"v\" value=\"(gL * (EL - v) + iSyn) / C\"/>"
        "      <OnCondition test=\"v .gt. vth\">"
        "        <EventOut port=\"spike\"/>"
        "        <StateAssignment variable=\"v\" value=\"vreset\"/>"
        "      </OnCondition>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <GLIF3TestCell id=\"glifInstance\" C=\"1.0e-10\" gL=\"1.0e-8\" EL=\"-70mV\" vth=\"-50mV\" vreset=\"-65mV\" asc_amp1=\"10pA\"/>"
        "</neuroml>");

    return write_temp_file("spikecorec_resolve_glif_chain_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"ResolveGlifChainTop\">"
        "  <include href=\"spikecorec_resolve_glif_chain_types.nml\"/>"
        "</neuroml>");
}

} // namespace

// ── SymbolTable ──────────────────────────────────────────────────────────

TEST(SymbolTable, register_symbol_assigns_stable_increasing_indices) {
    SymbolTable symbols;
    s32 first_index = symbols.register_symbol("alpha");
    s32 second_index = symbols.register_symbol("beta");
    s32 first_index_again = symbols.register_symbol("alpha");

    EXPECT_EQ(first_index, 0);
    EXPECT_EQ(second_index, 1);
    EXPECT_EQ(first_index_again, first_index);
}

TEST(SymbolTable, contains_and_index_of_reflect_registered_symbols) {
    SymbolTable symbols;
    EXPECT_FALSE(symbols.contains("gamma"));

    s32 index = symbols.register_symbol("gamma");
    EXPECT_TRUE(symbols.contains("gamma"));
    EXPECT_EQ(symbols.index_of("gamma"), index);
}

TEST(SymbolTable, index_of_throws_on_unregistered_symbol) {
    SymbolTable symbols;
    EXPECT_THROW(symbols.index_of("neverRegistered"), std::runtime_error);
}

// ── extends/Fixed flattening (acceptance criterion: fully flattened GLIF chain) ──

TEST(ResolveAndLower, flattens_multi_hop_extends_chain_merging_every_ancestors_declarations) {
    String path = write_glif_chain_fixture();

    NML_Parser parser;
    parser.parse(path);

    ResolvedModel model = resolve_and_lower(parser);

    ASSERT_EQ(model.types.count("GLIF3TestCell"), 1u);
    const ResolvedComponentType &resolved = model.types.at("GLIF3TestCell");
    EXPECT_EQ(resolved.bucket, ComponentTypeBucket::Dynamics);

    ASSERT_TRUE(std::holds_alternative<CellType>(resolved.flattened));
    const CellType &flattened_cell = std::get<CellType>(resolved.flattened);

    // Own declarations (4 parameters, 1 state variable, 1 time derivative, 1
    // on-condition) plus the ancestor chain's: `C`/`asc_amp1` (GLIFBaseCell),
    // `v` state variable (GLIFBaseCell's Dynamics), `baseExposure`
    // (GLIFBaseCell) + `v` exposure (baseCellMembPot, 2 hops up), and
    // `spike` EventPort (baseSpikingCell, 3 hops up) -- none of which
    // GLIF3TestCell itself declares.
    EXPECT_EQ(flattened_cell.parameters.size(), 6u);
    EXPECT_EQ(flattened_cell.state_variables.size(), 2u);
    EXPECT_EQ(flattened_cell.time_derivatives.size(), 1u);
    EXPECT_EQ(flattened_cell.on_conditions.size(), 1u);
    EXPECT_EQ(flattened_cell.exposures.size(), 2u);
    ASSERT_EQ(flattened_cell.event_ports.size(), 1u);
    EXPECT_EQ(flattened_cell.event_ports[0].name, "spike");
    EXPECT_EQ(flattened_cell.event_ports[0].direction, "out");

    auto has_parameter_named = [&](const String &name) {
        for (const auto &parameter : flattened_cell.parameters) {
            if (parameter.name == name) return true;
        }
        return false;
    };
    EXPECT_TRUE(has_parameter_named("C"));         // GLIFBaseCell
    EXPECT_TRUE(has_parameter_named("asc_amp1"));  // GLIFBaseCell
    EXPECT_TRUE(has_parameter_named("gL"));        // GLIF3TestCell's own

    // `Fixed parameter="asc_amp1" value="10pA"` (pinning an INHERITED
    // parameter) resolved to its SI value: 10pA = 10 * 1e-12 = 1e-11.
    ASSERT_EQ(resolved.fixed_parameter_values.count("asc_amp1"), 1u);
    EXPECT_NEAR(resolved.fixed_parameter_values.at("asc_amp1"), 1e-11, 1e-18);
}

TEST(ResolveAndLower, bucket_sorts_structure_only_types_separately_from_dynamics) {
    String path = write_glif_chain_fixture();

    NML_Parser parser;
    parser.parse(path);

    ResolvedModel model = resolve_and_lower(parser);

    ASSERT_EQ(model.types.count("population"), 1u);
    EXPECT_EQ(model.types.at("population").bucket, ComponentTypeBucket::Structure);

    ASSERT_EQ(model.types.count("projection"), 1u);
    EXPECT_EQ(model.types.at("projection").bucket, ComponentTypeBucket::Structure);

    ASSERT_EQ(model.types.count("GLIF3TestCell"), 1u);
    EXPECT_EQ(model.types.at("GLIF3TestCell").bucket, ComponentTypeBucket::Dynamics);
}

// ── Units → SI (acceptance criterion: real dimensioned parameter converted end-to-end) ──

TEST(ResolveAndLower, converts_dimensioned_instance_attributes_to_si_end_to_end) {
    String path = write_glif_chain_fixture();

    NML_Parser parser;
    parser.parse(path);

    ResolvedModel model = resolve_and_lower(parser);

    const ResolvedInstance *cell_instance = find_instance_by_tag(model.instances, "GLIF3TestCell");
    ASSERT_NE(cell_instance, nullptr);
    EXPECT_EQ(cell_instance->id, "glifInstance");

    ASSERT_EQ(cell_instance->numeric_attributes.count("EL"), 1u);
    EXPECT_NEAR(cell_instance->numeric_attributes.at("EL"), -0.07, 1e-12);

    ASSERT_EQ(cell_instance->numeric_attributes.count("vth"), 1u);
    EXPECT_NEAR(cell_instance->numeric_attributes.at("vth"), -0.05, 1e-12);

    ASSERT_EQ(cell_instance->numeric_attributes.count("asc_amp1"), 1u);
    EXPECT_NEAR(cell_instance->numeric_attributes.at("asc_amp1"), 1e-11, 1e-18);
}

// ── IDref wiring (acceptance criterion: unresolved IDref = clear error; resolved = index) ──

TEST(ResolveAndLower, resolves_a_valid_idref_to_the_referenced_instances_symbol_index) {
    String path = write_temp_file("spikecorec_resolve_valid_idref_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"ResolveValidIdrefTestDoc\">"
        "  <izhikevichCell id=\"izInstance\" v0=\"-70mV\" thresh=\"30mV\" a=\"0.02\" b=\"0.2\" c=\"-65\" d=\"6\"/>"
        "  <network id=\"TestNet\">"
        "    <population id=\"Pop1\" component=\"izInstance\" size=\"10\"/>"
        "  </network>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(path);

    ResolvedModel model = resolve_and_lower(parser);

    const ResolvedInstance *cell_instance = find_instance_by_tag(model.instances, "izhikevichCell");
    ASSERT_NE(cell_instance, nullptr);
    s32 expected_index = model.symbols.index_of("izInstance");
    EXPECT_EQ(cell_instance->symbol_index, expected_index);

    const ResolvedInstance *network = find_instance_by_tag(model.instances, "network");
    ASSERT_NE(network, nullptr);
    const ResolvedInstance *population = find_instance_by_tag(network->children, "population");
    ASSERT_NE(population, nullptr);

    ASSERT_EQ(population->idref_attributes.count("component"), 1u);
    EXPECT_EQ(population->idref_attributes.at("component"), expected_index);

    ASSERT_EQ(population->numeric_attributes.count("size"), 1u);
    EXPECT_NEAR(population->numeric_attributes.at("size"), 10.0, 1e-12);
}

TEST(ResolveAndLower, throws_a_clear_error_on_an_unresolved_idref) {
    String path = write_temp_file("spikecorec_resolve_unresolved_idref_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"ResolveUnresolvedIdrefTestDoc\">"
        "  <network id=\"TestNet\">"
        "    <population id=\"Pop1\" component=\"thisInstanceDoesNotExist\" size=\"10\"/>"
        "  </network>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(path);

    EXPECT_THROW(resolve_and_lower(parser), std::runtime_error);
}

// ── Units → SI errors (acceptance criterion: unrecognized unit = clear error) ──

TEST(ResolveAndLower, throws_a_clear_error_on_an_unrecognized_unit) {
    // A custom ComponentType/instance routed through <include>, same
    // rationale as write_glif_chain_fixture(): the top-level file's XSD gate
    // would reject an official element's dimensioned attribute with a bogus
    // unit before RESOLVE ever saw it, so this exercises the resolve pass's
    // own unit conversion instead of the XSD gate.
    write_temp_file("spikecorec_resolve_bad_unit_types.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"ResolveBadUnitTypes\">"
        "  <ComponentType name=\"BadUnitTestCell\" extends=\"baseCell\">"
        "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
        "  </ComponentType>"
        "  <BadUnitTestCell id=\"badUnitInstance\" EL=\"-70zorkmids\"/>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_resolve_bad_unit_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"ResolveBadUnitTop\">"
        "  <include href=\"spikecorec_resolve_bad_unit_types.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);

    EXPECT_THROW(resolve_and_lower(parser), std::invalid_argument);
}
