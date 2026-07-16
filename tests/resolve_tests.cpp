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

// Small helpers for inspecting the still-tree-shaped OnStart/OnCondition/...
// bodies (arch §1.4: expression-tree content stays opaque even after
// flattening) -- mirrors nml_tests.cpp's own private helpers of the same name.
String get_string_attribute(const NML_Node &node, const String &key) {
    auto entry = node.attributes.find(key);
    if (entry == node.attributes.end()) return "";
    return std::any_cast<String>(entry->second);
}

const NML_Node *find_child_by_tag(const NML_Node &node, const String &tag) {
    for (const auto &child : node.body) {
        if (child.tag_name == tag) return &child;
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

// ticket #60 [X1]: the exception's own message (what a caller actually
// inspects, not just the side-channel log line) must name the offending
// IDref value, the attribute it was on, and the element it was on.
TEST(ResolveAndLower, unresolved_idref_exception_names_the_value_attribute_and_element) {
    String path = write_temp_file("spikecorec_resolve_unresolved_idref_message_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"ResolveUnresolvedIdrefMessageTestDoc\">"
        "  <network id=\"TestNet\">"
        "    <population id=\"Pop1\" component=\"thisInstanceDoesNotExist\" size=\"10\"/>"
        "  </network>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(path);

    try {
        resolve_and_lower(parser);
        FAIL() << "expected resolve_and_lower to throw on an unresolved IDref";
    } catch (const std::runtime_error &error) {
        String message = error.what();
        EXPECT_NE(message.find("thisInstanceDoesNotExist"), String::npos);
        EXPECT_NE(message.find("component"), String::npos);
        EXPECT_NE(message.find("population"), String::npos);
    }
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

// ticket #60 [X1]: same requirement as the unresolved-IDref case above, but
// for a unit-conversion failure -- the exception's own message must name the
// attribute, the element, and the unresolvable value, not just the bare
// "unknown unit symbol" text unit_value_to_si itself produces (which knows
// nothing about which attribute/element it was called for).
TEST(ResolveAndLower, unrecognized_unit_exception_names_the_attribute_element_and_value) {
    write_temp_file("spikecorec_resolve_bad_unit_types_message.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"ResolveBadUnitTypesMessage\">"
        "  <ComponentType name=\"BadUnitTestCellMessage\" extends=\"baseCell\">"
        "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
        "  </ComponentType>"
        "  <BadUnitTestCellMessage id=\"badUnitInstanceMessage\" EL=\"-70zorkmids\"/>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_resolve_bad_unit_top_message.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"ResolveBadUnitTopMessage\">"
        "  <include href=\"spikecorec_resolve_bad_unit_types_message.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);

    try {
        resolve_and_lower(parser);
        FAIL() << "expected resolve_and_lower to throw on an unrecognized unit";
    } catch (const std::invalid_argument &error) {
        String message = error.what();
        EXPECT_NE(message.find("EL"), String::npos);
        EXPECT_NE(message.find("BadUnitTestCellMessage"), String::npos);
        EXPECT_NE(message.find("zorkmids"), String::npos);
    }
}

// ticket #60 [X1]: same class of gap, but for a `Fixed` pin's value
// (apply_fixed_pins) rather than an instance attribute's -- the exception's
// own message must name the Fixed parameter and its unconvertible value.
// Flattening every cataloged ComponentType (not just ones an instance
// actually uses) happens unconditionally in resolve_and_lower's own pass 2,
// so this fires with no instance of BadFixedTestCell required at all.
TEST(ResolveAndLower, unconvertible_fixed_parameter_value_exception_names_the_parameter_and_value) {
    write_temp_file("spikecorec_resolve_bad_fixed_types.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"ResolveBadFixedTypes\">"
        "  <ComponentType name=\"BadFixedBaseCell\" extends=\"baseCell\">"
        "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
        "  </ComponentType>"
        "  <ComponentType name=\"BadFixedTestCell\" extends=\"BadFixedBaseCell\">"
        "    <Fixed parameter=\"EL\" value=\"-70zorkmids\"/>"
        "  </ComponentType>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_resolve_bad_fixed_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"ResolveBadFixedTop\">"
        "  <include href=\"spikecorec_resolve_bad_fixed_types.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);

    try {
        resolve_and_lower(parser);
        FAIL() << "expected resolve_and_lower to throw on an unconvertible Fixed value";
    } catch (const std::invalid_argument &error) {
        String message = error.what();
        EXPECT_NE(message.find("EL"), String::npos);
        EXPECT_NE(message.find("zorkmids"), String::npos);
    }
}

// ── Regression: ancestor-only `OnStart` survives flattening (review finding #1) ──
//
// `iafTauCell` (Cells.xml) declares its own `OnStart` (`v = leakReversal`)
// but no other Dynamics-bearing descendant in this fixture redeclares one --
// exactly the real-world shape the finding flagged: a user cell extending a
// std-lib cell without repeating its `OnStart` must still carry that
// ancestor's initial-value assignment through flattening, not lose it.
TEST(ResolveAndLower, merges_ancestor_only_on_start_across_extends_chain) {
    String path = write_temp_file("spikecorec_resolve_on_start_inheritance_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"OnStartInheritanceTestDoc\">"
        "  <ComponentType name=\"ChildOfIafTauCell\" extends=\"iafTauCell\">"
        "    <Parameter name=\"extraParam\" dimension=\"none\"/>"
        "  </ComponentType>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(path);

    ResolvedModel model = resolve_and_lower(parser);

    ASSERT_EQ(model.types.count("ChildOfIafTauCell"), 1u);
    ASSERT_TRUE(std::holds_alternative<CellType>(model.types.at("ChildOfIafTauCell").flattened));
    const CellType &flattened_cell = std::get<CellType>(model.types.at("ChildOfIafTauCell").flattened);

    // ChildOfIafTauCell declares no <Dynamics>/<OnStart> of its own -- this
    // OnStart can only have come from iafTauCell across the extends chain.
    ASSERT_EQ(flattened_cell.on_starts.size(), 1u);
    const NML_Node *state_assignment = find_child_by_tag(flattened_cell.on_starts[0].body, "StateAssignment");
    ASSERT_NE(state_assignment, nullptr);
    EXPECT_EQ(get_string_attribute(*state_assignment, "variable"), "v");
    EXPECT_EQ(get_string_attribute(*state_assignment, "value"), "leakReversal");
}

// ── Regression: descendant declarations override, not duplicate, an ancestor's (review finding #4/minor) ──

TEST(ResolveAndLower, deduplicates_overridden_declarations_keeping_the_most_derived) {
    String path = write_temp_file("spikecorec_resolve_dedup_override_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"DedupOverrideTestDoc\">"
        "  <ComponentType name=\"DedupBaseCell\" extends=\"baseCell\">"
        "    <Parameter name=\"sharedParam\" dimension=\"voltage\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"v\" dimension=\"voltage\"/>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <ComponentType name=\"DedupChildCell\" extends=\"DedupBaseCell\">"
        "    <Parameter name=\"sharedParam\" dimension=\"voltage\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
        "      <TimeDerivative variable=\"v\" value=\"0\"/>"
        "    </Dynamics>"
        "  </ComponentType>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(path);

    ResolvedModel model = resolve_and_lower(parser);

    ASSERT_EQ(model.types.count("DedupChildCell"), 1u);
    const CellType &flattened_cell = std::get<CellType>(model.types.at("DedupChildCell").flattened);

    // `sharedParam` and `v` are each declared once on the ancestor and once
    // (redeclared) on the child -- the flattened set must carry exactly one
    // entry per name (not two), and it must be the child's (most-derived)
    // version: `v`'s `exposure="v"` only appears on the child's redeclaration.
    ASSERT_EQ(flattened_cell.parameters.size(), 1u);
    EXPECT_EQ(flattened_cell.parameters[0].name, "sharedParam");

    ASSERT_EQ(flattened_cell.state_variables.size(), 1u);
    EXPECT_EQ(flattened_cell.state_variables[0].name, "v");
    EXPECT_EQ(flattened_cell.state_variables[0].exposure, "v");
}

// ── Regression: `target` is path-shaped, not an IDref; `input` is (review finding #3) ──

TEST(ResolveAndLower, treats_path_shaped_target_as_opaque_and_resolves_genuine_input_idref) {
    String path = write_temp_file("spikecorec_resolve_target_vs_input_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"TargetVsInputTestDoc\">"
        "  <pulseGenerator id=\"pulseGen1\" delay=\"10ms\" duration=\"100ms\" amplitude=\"1nA\"/>"
        "  <network id=\"net1\">"
        "    <population id=\"Pop0\" component=\"pulseGen1\" size=\"1\"/>"
        "    <explicitInput target=\"Pop0[0]\" input=\"pulseGen1\"/>"
        "  </network>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(path);

    ResolvedModel model;
    EXPECT_NO_THROW(model = resolve_and_lower(parser));

    const ResolvedInstance *network = find_instance_by_tag(model.instances, "network");
    ASSERT_NE(network, nullptr);
    const ResolvedInstance *explicit_input = find_instance_by_tag(network->children, "explicitInput");
    ASSERT_NE(explicit_input, nullptr);

    // `target="Pop0[0]"` is a population-index path, not a bare id -- passed
    // through as an opaque string, never IDref-resolved.
    ASSERT_EQ(explicit_input->string_attributes.count("target"), 1u);
    EXPECT_EQ(explicit_input->string_attributes.at("target"), "Pop0[0]");
    EXPECT_EQ(explicit_input->idref_attributes.count("target"), 0u);

    // `input="pulseGen1"` IS a genuine bare-id IDref, resolved to the
    // generator instance's own symbol index.
    ASSERT_EQ(explicit_input->idref_attributes.count("input"), 1u);
    EXPECT_EQ(explicit_input->idref_attributes.at("input"), model.symbols.index_of("pulseGen1"));
}

// ── Regression: an unlisted string attribute is opaque, not a "bad unit" (review finding #2) ──

TEST(ResolveAndLower, treats_an_unlisted_non_numeric_attribute_as_opaque_instead_of_throwing) {
    // A custom instance attribute name outside every hardcoded IDref/opaque
    // list, routed through <include> since the custom tag itself isn't a
    // schema-known element (see write_glif_chain_fixture's rationale).
    write_temp_file("spikecorec_resolve_unlisted_string_types.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"UnlistedStringTypesDoc\">"
        "  <ComponentType name=\"TaggedTestCell\" extends=\"baseCell\">"
        "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
        "  </ComponentType>"
        "  <TaggedTestCell id=\"taggedInstance\" EL=\"-70mV\" label=\"excitatory\"/>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_resolve_unlisted_string_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"UnlistedStringTop\">"
        "  <include href=\"spikecorec_resolve_unlisted_string_types.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);

    ResolvedModel model;
    EXPECT_NO_THROW(model = resolve_and_lower(parser));

    const ResolvedInstance *cell_instance = find_instance_by_tag(model.instances, "TaggedTestCell");
    ASSERT_NE(cell_instance, nullptr);

    // "excitatory" doesn't even look like a dimensioned literal (no leading
    // number) -- it must land as an opaque string, not be force-fed to
    // unit_value_to_si and throw.
    ASSERT_EQ(cell_instance->string_attributes.count("label"), 1u);
    EXPECT_EQ(cell_instance->string_attributes.at("label"), "excitatory");

    // The genuinely-dimensioned attribute alongside it still converts normally.
    ASSERT_EQ(cell_instance->numeric_attributes.count("EL"), 1u);
    EXPECT_NEAR(cell_instance->numeric_attributes.at("EL"), -0.07, 1e-12);
}

// ── ancestor_chain (ticket #7 [A5]'s additive gap fix, option (a)) ─────────
//
// merge_chain's walk now also records the same-category ancestor names it
// visits, so a synapse's classification (arch §3.1 `ComponentType`: does the
// `extends` chain pass through `baseConductanceBasedSynapse`?) can be
// determined from a ResolvedComponentType alone, without re-walking
// NML_Parser::library.

TEST(ResolveAndLower, captures_ancestor_chain_across_a_synapse_extends_walk) {
    String path = write_temp_file("spikecorec_resolve_ancestor_chain_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"AncestorChainTestDoc\">"
        "  <expOneSynapse id=\"expSynInstance\" gbase=\"1nS\" erev=\"0mV\" tauDecay=\"3ms\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(path);

    ResolvedModel model = resolve_and_lower(parser);

    ASSERT_EQ(model.types.count("expOneSynapse"), 1u);
    const Vector<String> &chain = model.types.at("expOneSynapse").ancestor_chain;

    // expOneSynapse -> baseConductanceBasedSynapse -> baseVoltageDepSynapse
    // -> baseSynapse -- the walk stops there since baseSynapse's own parent,
    // basePointCurrent, is cataloged as a different category (InputsType),
    // matching merge_chain's "stops... the moment it reaches an ancestor
    // cataloged under a different category" rule.
    ASSERT_EQ(chain.size(), 4u);
    EXPECT_EQ(chain[0], "expOneSynapse");
    EXPECT_EQ(chain[1], "baseConductanceBasedSynapse");
    EXPECT_EQ(chain[2], "baseVoltageDepSynapse");
    EXPECT_EQ(chain[3], "baseSynapse");
}
