#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <any>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <variant>
#include <gtest/gtest.h>
#include <libxml/parser.h>
#include <libxml/tree.h>

#include "spikecorec/nml/nml.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

namespace {

String get_string_attribute(const NML_Node &component, const String &key) {
    auto entry = component.attributes.find(key);
    if (entry == component.attributes.end()) return "";
    return std::any_cast<String>(entry->second);
}

String write_temp_file(const String &filename, const String &contents) {
    String path = (std::filesystem::temp_directory_path() / filename).string();
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

// Finds the first direct child of `node` with the given tag name, or nullptr.
// Used to make assertions about merged-tree shape readable without depending
// on exact child ordering across includes.
const NML_Node *find_child_by_tag(const NML_Node &node, const String &tag) {
    for (const auto &child : node.body) {
        if (child.tag_name == tag) return &child;
    }
    return nullptr;
}

} // namespace

// ── NML_Node ─────────────────────────────────────────────────

TEST(NmlNode, construction) {
    NML_Node component("ComponentType");
    EXPECT_EQ(component.tag_name, "ComponentType");
    EXPECT_TRUE(component.attributes.empty());
    EXPECT_TRUE(component.body.empty());
}

TEST(NmlNode, add_attribute_stores_any_value) {
    NML_Node component("ComponentType");
    component.add_attribute("name", String("iafCell"));

    auto entry = component.attributes.find("name");
    ASSERT_NE(entry, component.attributes.end());
    EXPECT_EQ(std::any_cast<String>(entry->second), "iafCell");
}

TEST(NmlNode, nest_appends_child) {
    NML_Node parent("ComponentType");
    NML_Node child("Parameter");
    child.add_attribute("name", String("a"));

    parent.nest(child);

    ASSERT_EQ(parent.body.size(), 1u);
    EXPECT_EQ(parent.body[0].tag_name, "Parameter");
    EXPECT_EQ(get_string_attribute(parent.body[0], "name"), "a");
}

// Directly exercises deep-copy correctness: mutating the original after the
// copy is taken must not affect the copy, for both `attributes` (std::any,
// copies itself) and `body` (Vector<NML_Node>, must recurse).
TEST(NmlNode, copy_constructor_deep_copies_body_and_attributes) {
    NML_Node parent("ComponentType");
    parent.add_attribute("name", String("testCell"));

    NML_Node child("Parameter");
    child.add_attribute("name", String("a"));
    parent.nest(child);

    NML_Node copy(parent);

    parent.add_attribute("name", String("mutated"));
    parent.body.clear();

    EXPECT_EQ(copy.tag_name, "ComponentType");
    EXPECT_EQ(get_string_attribute(copy, "name"), "testCell");
    ASSERT_EQ(copy.body.size(), 1u);
    EXPECT_EQ(copy.body[0].tag_name, "Parameter");
    EXPECT_EQ(get_string_attribute(copy.body[0], "name"), "a");
}

// ── NML_Parser::xml_node_to_nml_node ────────────────────────

TEST(XmlNodeToNmlNode, parses_tag_attributes_and_nested_children) {
    const char *xml =
        "<Lems>"
        "  <ComponentType name=\"testCell\" extends=\"baseCell\">"
        "    <Parameter name=\"a\" dimension=\"none\"/>"
        "    <Parameter name=\"b\" dimension=\"voltage\"/>"
        "  </ComponentType>"
        "</Lems>";

    xmlDocPtr document = xmlReadMemory(xml, static_cast<int>(strlen(xml)), "test.xml", nullptr, XML_PARSE_NOBLANKS);
    ASSERT_NE(document, nullptr);

    xmlNodePtr root = xmlDocGetRootElement(document);
    ASSERT_NE(root, nullptr);

    xmlNodePtr component_type_node = root->children;
    ASSERT_NE(component_type_node, nullptr);
    ASSERT_TRUE(xmlStrEqual(component_type_node->name, BAD_CAST "ComponentType"));

    NML_Parser parser;
    NML_Node component = parser.xml_node_to_nml_node(component_type_node);

    EXPECT_EQ(component.tag_name, "ComponentType");
    EXPECT_EQ(get_string_attribute(component, "name"), "testCell");
    EXPECT_EQ(get_string_attribute(component, "extends"), "baseCell");

    ASSERT_EQ(component.body.size(), 2u);
    EXPECT_EQ(component.body[0].tag_name, "Parameter");
    EXPECT_EQ(get_string_attribute(component.body[0], "name"), "a");
    EXPECT_EQ(get_string_attribute(component.body[0], "dimension"), "none");
    EXPECT_EQ(component.body[1].tag_name, "Parameter");
    EXPECT_EQ(get_string_attribute(component.body[1], "name"), "b");

    xmlFreeDoc(document);
}

// ── NML_Parser: standard library loading ────────────────────

TEST(NmlParser, standard_library_path_is_baked_and_exists) {
    NML_Parser parser;
    EXPECT_FALSE(parser.STANDARD_LIBRARY_PATH.empty());
    EXPECT_TRUE(std::filesystem::exists(parser.STANDARD_LIBRARY_PATH));
}

// Acceptance criterion: "Core types resolve fully offline" — load_standard_library
// only ever reads STANDARD_LIBRARY_PATH off the local filesystem, no network.
TEST(NmlParser, load_standard_library_loads_vendored_bundle_offline) {
    NML_Parser parser;
    bool failed = parser.load_standard_library();

    EXPECT_FALSE(failed);
    EXPECT_GT(parser.library.size(), 100u);
}

// iafCell's own `extends` is "baseIafCapCell", several hops short of
// `baseCell` — this also exercises classify_component_type's extends-chain
// walk, not just a direct-parent check.
TEST(NmlParser, get_type_by_name_finds_a_real_core_type) {
    NML_Parser parser;
    ASSERT_FALSE(parser.load_standard_library());

    ComponentTypeEntry &entry = parser.get_type_by_name("iafCell");
    ASSERT_TRUE(std::holds_alternative<CellType>(entry));

    CellType &iaf_cell = std::get<CellType>(entry);
    EXPECT_EQ(iaf_cell.name, "iafCell");
    EXPECT_EQ(iaf_cell.raw.tag_name, "ComponentType");
    EXPECT_GT(iaf_cell.raw.body.size(), 0u);
    EXPECT_GT(iaf_cell.state_variables.size(), 0u);
    EXPECT_GT(iaf_cell.time_derivatives.size(), 0u);
    EXPECT_GT(iaf_cell.on_conditions.size(), 0u);
}

// This is also the merge-point/lookup API the resolve pass (A4) will call;
// a missing type must fail loudly rather than return something usable.
TEST(NmlParser, get_type_by_name_throws_on_missing_type) {
    NML_Parser parser;
    ASSERT_FALSE(parser.load_standard_library());

    EXPECT_THROW(parser.get_type_by_name("definitelyNotARealType_xyz"), std::runtime_error);
}

// ── ComponentType classification (typed `library`, ticket #2 [A3]) ──────
//
// Every cataloged <ComponentType> is classified into one of CellType,
// SynapseType, InputsType/GeneratorType, PopulationType, or
// ProjectType/ConnectionType (arch §3.3's D1/D3/D4/ST2/ST3 buckets), or left
// as the bare ComponentTypeBase identity when it fits none of them. These
// exercise the classification against real vendored std-lib types, across
// every category.

TEST(NmlComponentTypeClassification, classifies_pulse_generator_as_inputs_type) {
    NML_Parser parser;
    ASSERT_FALSE(parser.load_standard_library());

    ComponentTypeEntry &entry = parser.get_type_by_name("pulseGenerator");
    ASSERT_TRUE(std::holds_alternative<InputsType>(entry));

    InputsType &pulse_generator = std::get<InputsType>(entry);
    EXPECT_GT(pulse_generator.parameters.size(), 0u);
    EXPECT_GT(pulse_generator.properties.size(), 0u);
    EXPECT_GT(pulse_generator.state_variables.size(), 0u);
    EXPECT_GT(pulse_generator.on_conditions.size(), 0u);
}

// `poissonFiringSynapse` is an InputsType that also carries a
// ComponentReference to the synapse instance it drives.
TEST(NmlComponentTypeClassification, inputs_type_captures_component_reference_to_driven_synapse) {
    NML_Parser parser;
    ASSERT_FALSE(parser.load_standard_library());

    ComponentTypeEntry &entry = parser.get_type_by_name("poissonFiringSynapse");
    ASSERT_TRUE(std::holds_alternative<InputsType>(entry));

    InputsType &poisson_firing_synapse = std::get<InputsType>(entry);
    ASSERT_GT(poisson_firing_synapse.component_references.size(), 0u);
    EXPECT_EQ(poisson_firing_synapse.component_references[0].name, "synapse");
}

// `population` itself only directly declares `size` — its `component`
// ComponentReference is declared on its parent, `basePopulation` (PARSE
// doesn't flatten `extends`; that merge is RESOLVE's job, ticket #49), which
// classifies as PopulationType in its own right (its own name is one of the
// category's anchors).
TEST(NmlComponentTypeClassification, classifies_population_as_population_type) {
    NML_Parser parser;
    ASSERT_FALSE(parser.load_standard_library());

    ComponentTypeEntry &population_entry = parser.get_type_by_name("population");
    ASSERT_TRUE(std::holds_alternative<PopulationType>(population_entry));
    PopulationType &population = std::get<PopulationType>(population_entry);
    ASSERT_GT(population.parameters.size(), 0u);
    EXPECT_EQ(population.parameters[0].name, "size");

    ComponentTypeEntry &base_population_entry = parser.get_type_by_name("basePopulation");
    ASSERT_TRUE(std::holds_alternative<PopulationType>(base_population_entry));
    PopulationType &base_population = std::get<PopulationType>(base_population_entry);
    ASSERT_GT(base_population.component_references.size(), 0u);
    EXPECT_EQ(base_population.component_references[0].name, "component");
}

// `projection` and `connection` are two different shapes (pre/post
// population routing vs. pre/post cell-path routing) but both fall into the
// single ProjectType/ConnectionType category.
TEST(NmlComponentTypeClassification, classifies_projection_and_connection_as_project_type) {
    NML_Parser parser;
    ASSERT_FALSE(parser.load_standard_library());

    ComponentTypeEntry &projection_entry = parser.get_type_by_name("projection");
    ASSERT_TRUE(std::holds_alternative<ProjectType>(projection_entry));
    ProjectType &projection = std::get<ProjectType>(projection_entry);
    ASSERT_GT(projection.component_references.size(), 0u);
    EXPECT_EQ(projection.component_references[0].name, "synapse");
    EXPECT_GT(projection.path_fields.size(), 0u);
    EXPECT_GT(projection.children.size(), 0u);

    ComponentTypeEntry &connection_entry = parser.get_type_by_name("connection");
    ASSERT_TRUE(std::holds_alternative<ConnectionType>(connection_entry));
    EXPECT_GT(std::get<ConnectionType>(connection_entry).path_fields.size(), 0u);
}

// `morphology` declares no `extends` and doesn't fit any of the five
// categories (it's Structures/multicompartment, deferred to Phase 3) — it
// must stay the bare, unclassified ComponentTypeBase, with its raw node
// still intact rather than being dropped or misclassified.
TEST(NmlComponentTypeClassification, leaves_out_of_scope_types_unclassified) {
    NML_Parser parser;
    ASSERT_FALSE(parser.load_standard_library());

    ComponentTypeEntry &entry = parser.get_type_by_name("morphology");
    EXPECT_TRUE(std::holds_alternative<ComponentTypeBase>(entry));
    EXPECT_FALSE(std::get<ComponentTypeBase>(entry).raw.body.empty());
}

// ── NML_Parser::validate_against_schema (ticket #8 [A2]) ────

TEST(ValidateAgainstSchema, schema_path_is_baked_and_exists) {
    NML_Parser parser;
    EXPECT_FALSE(parser.NML_SCHEMA_PATH.empty());
    EXPECT_TRUE(std::filesystem::exists(parser.NML_SCHEMA_PATH));
}

TEST(ValidateAgainstSchema, accepts_a_conformant_neuroml_document) {
    String path = write_temp_file("spikecorec_valid_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"TestDoc\">"
        "  <izhikevichCell id=\"izTest\" v0=\"-70mV\" thresh=\"30mV\" a=\"0.02\" b=\"0.2\" c=\"-65\" d=\"6\"/>"
        "</neuroml>");

    NML_Parser parser;
    EXPECT_TRUE(parser.validate_against_schema(path));
}

TEST(ValidateAgainstSchema, rejects_a_document_with_an_unknown_element) {
    String path = write_temp_file("spikecorec_invalid_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"TestDoc\">"
        "  <thisTagDoesNotExistInSchema id=\"oops\"/>"
        "</neuroml>");

    NML_Parser parser;
    EXPECT_FALSE(parser.validate_against_schema(path));
}

// A LEMS file (root <Lems>) is a real, well-formed XML document, but it is
// not a NeuroML2 document — validating it against the NeuroML2 XSD must
// still fail rather than silently pass.
TEST(ValidateAgainstSchema, rejects_a_wellformed_but_wrong_schema_document) {
    NML_Parser parser;
    String lems_file = parser.STANDARD_LIBRARY_PATH + "/Cells.xml";
    EXPECT_FALSE(parser.validate_against_schema(lems_file));
}

TEST(ValidateAgainstSchema, rejects_a_missing_file) {
    NML_Parser parser;
    EXPECT_FALSE(parser.validate_against_schema("/tmp/definitely_not_a_real_file_xyz.nml"));
}

// ── NML_Parser::parse / ingest_file ─────────────────────────

TEST(NmlParserParse, parses_a_single_file_into_root_node) {
    String path = write_temp_file("spikecorec_parse_single_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"TestDoc\">"
        "  <izhikevichCell id=\"izTest\" v0=\"-70mV\" thresh=\"30mV\" a=\"0.02\" b=\"0.2\" c=\"-65\" d=\"6\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(path);

    ASSERT_NE(parser.root_node, nullptr);
    EXPECT_EQ(parser.root_node->tag_name, "neuroml");
    ASSERT_EQ(parser.root_node->body.size(), 1u);
    EXPECT_EQ(parser.root_node->body[0].tag_name, "izhikevichCell");
}

// Inline user-defined ComponentTypes must be cataloged into `library` by
// name, and must NOT also be duplicated as a body child of root_node — only
// non-ComponentType tags belong in the walked tree.
TEST(NmlParserParse, catalogs_inline_component_type_without_duplicating_it_in_the_tree) {
    String path = write_temp_file("spikecorec_parse_inline_comptype_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"TestDoc\">"
        "  <izhikevichCell id=\"izTest\" v0=\"-70mV\" thresh=\"30mV\" a=\"0.02\" b=\"0.2\" c=\"-65\" d=\"6\"/>"
        "  <ComponentType name=\"parseTestInlineCell\" extends=\"baseCell\">"
        "    <Parameter name=\"a\" dimension=\"none\"/>"
        "  </ComponentType>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(path);

    ASSERT_EQ(parser.library.count("parseTestInlineCell"), 1u);
    ASSERT_TRUE(std::holds_alternative<CellType>(parser.library.at("parseTestInlineCell")));
    EXPECT_EQ(std::get<CellType>(parser.library.at("parseTestInlineCell")).parameters.size(), 1u);

    ASSERT_EQ(parser.root_node->body.size(), 1u);
    EXPECT_EQ(parser.root_node->body[0].tag_name, "izhikevichCell");
}

// <include href="..."> must be followed recursively and spliced away: its
// own contents merge into the same root_node/library, the <include> tag
// itself never appears as a tree node.
TEST(NmlParserParse, follows_include_and_splices_it_into_the_shared_tree) {
    write_temp_file("spikecorec_parse_included_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"IncludedDoc\">"
        "  <ComponentType name=\"parseTestIncludedCell\" extends=\"baseCell\">"
        "    <Parameter name=\"a\" dimension=\"none\"/>"
        "  </ComponentType>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_parse_top_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"TopDoc\">"
        "  <include href=\"spikecorec_parse_included_test.nml\"/>"
        "  <izhikevichCell id=\"izTest\" v0=\"-70mV\" thresh=\"30mV\" a=\"0.02\" b=\"0.2\" c=\"-65\" d=\"6\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);

    ASSERT_EQ(parser.library.count("parseTestIncludedCell"), 1u);
    ASSERT_TRUE(std::holds_alternative<CellType>(parser.library.at("parseTestIncludedCell")));
    EXPECT_EQ(std::get<CellType>(parser.library.at("parseTestIncludedCell")).parameters.size(), 1u);

    ASSERT_EQ(parser.root_node->body.size(), 1u);
    EXPECT_EQ(parser.root_node->body[0].tag_name, "izhikevichCell");
}

// Included files are never schema-gated: the vendored Cells.xml is a raw
// LEMS document (root <Lems>) that fails NeuroML2 XSD validation outright
// (see ValidateAgainstSchema.rejects_a_wellformed_but_wrong_schema_document),
// yet following it via <include> must succeed and populate `library` with
// its real ComponentTypes.
TEST(NmlParserParse, does_not_schema_gate_included_files) {
    NML_Parser path_helper;
    String cells_file = path_helper.STANDARD_LIBRARY_PATH + "/Cells.xml";

    String top_path = write_temp_file("spikecorec_parse_stdlib_include_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"TopDoc\">"
        "  <include href=\"" + cells_file + "\"/>"
        "</neuroml>");

    NML_Parser parser;
    EXPECT_NO_THROW(parser.parse(top_path));

    EXPECT_GT(parser.library.count("iafCell"), 0u);
}

// The top-level file passed to parse() IS schema-gated: a well-formed but
// non-conformant document must fail loudly rather than silently build a
// bogus tree.
TEST(NmlParserParse, throws_on_schema_invalid_top_level_file) {
    String path = write_temp_file("spikecorec_parse_invalid_schema_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"TestDoc\">"
        "  <thisTagDoesNotExistInSchema id=\"oops\"/>"
        "</neuroml>");

    NML_Parser parser;
    EXPECT_THROW(parser.parse(path), std::runtime_error);
}

TEST(NmlParserParse, ingest_file_on_a_missing_path_does_not_crash_or_set_root) {
    NML_Parser parser;
    EXPECT_NO_THROW(parser.ingest_file("/definitely/does/not/exist_xyz.nml", false));
    EXPECT_EQ(parser.root_node, nullptr);
}

// ── Phase-1 example models (ticket #2 [A3] acceptance criteria) ─────────
//
// CLAUDE.md's Phase 1 scope is "GLIF cells with the full synapse model,
// single cell through networks". GLIF ComponentTypes are user-authored LEMS
// (they are not part of the vendored NeuroML2CoreTypes bundle), so these
// fixtures model that real-world shape: a user-defined GLIF1 cell (mirroring
// the GLIF1 example in docs/nml_ir_spec.md §4) pulled in via `<include href>`,
// alongside instances of vendored std-lib synapse types.

// Example A: a single GLIF1 cell, defined and instantiated in one included
// file, wired into a top-level document via one level of <include>.
TEST(NmlParserParsePhase1Example, parses_single_glif_cell_model_with_merged_include) {
    write_temp_file("spikecorec_phase1_glif1_cell_type.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"Phase1GLIF1CellType\">"
        "  <ComponentType name=\"GLIF1Cell\" extends=\"baseCell\">"
        "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
        "    <Parameter name=\"gL\" dimension=\"conductance\"/>"
        "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
        "    <Parameter name=\"vth\" dimension=\"voltage\"/>"
        "    <Parameter name=\"vreset\" dimension=\"voltage\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
        "      <TimeDerivative variable=\"v\" value=\"(gL * (EL - v) + iSyn) / C\"/>"
        "      <OnStart>"
        "        <StateAssignment variable=\"v\" value=\"EL\"/>"
        "      </OnStart>"
        "      <OnCondition test=\"v .gt. vth\">"
        "        <EventOut port=\"spike\"/>"
        "        <StateAssignment variable=\"v\" value=\"vreset\"/>"
        "      </OnCondition>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <GLIF1Cell id=\"glif1CellInstance\" C=\"1.0e-10\" gL=\"1.0e-8\" EL=\"-0.07\" vth=\"-0.05\" vreset=\"-0.07\"/>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_phase1_glif1_single_cell.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"Phase1GLIF1SingleCell\">"
        "  <include href=\"spikecorec_phase1_glif1_cell_type.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    EXPECT_NO_THROW(parser.parse(top_path));

    // The user-defined GLIF1Cell ComponentType, from the included file,
    // merged into the parser's library alongside the (already loaded) A1
    // std-lib bundle, and classified as a CellType (extends "baseCell").
    ASSERT_EQ(parser.library.count("GLIF1Cell"), 1u);
    ASSERT_TRUE(std::holds_alternative<CellType>(parser.library.at("GLIF1Cell")));
    CellType &glif1_cell_type = std::get<CellType>(parser.library.at("GLIF1Cell"));
    EXPECT_EQ(glif1_cell_type.state_variables.size(), 1u);
    EXPECT_EQ(glif1_cell_type.state_variables[0].name, "v");
    EXPECT_EQ(glif1_cell_type.time_derivatives.size(), 1u);
    EXPECT_EQ(glif1_cell_type.on_conditions.size(), 1u);
    EXPECT_GT(parser.library.count("iafCell"), 0u);

    // The GLIF1Cell instance (a non-ComponentType tag) is spliced into the
    // shared tree, not cataloged into `library`.
    const NML_Node *cell_instance = find_child_by_tag(*parser.root_node, "GLIF1Cell");
    ASSERT_NE(cell_instance, nullptr);
    EXPECT_EQ(get_string_attribute(*cell_instance, "id"), "glif1CellInstance");
}

// Example B: a GLIF excitatory/inhibitory network — the cell type file
// itself includes a synapse-instances file, so following the include graph
// two levels deep is required for a full merge. Exercises current-based
// (alphaCurrentSynapse), conductance-based (expOneSynapse), and per-edge
// (blockingPlasticSynapse / NMDA) synapses from the A1 std-lib bundle
// alongside a projection wiring two populations of the user-defined cell.
TEST(NmlParserParsePhase1Example, parses_glif_network_model_with_nested_includes_merged) {
    write_temp_file("spikecorec_phase1_synapse_instances.xml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"Phase1SynapseInstances\">"
        "  <expOneSynapse id=\"glifExcSynapse\" gbase=\"1nS\" erev=\"0mV\" tauDecay=\"3ms\"/>"
        "  <alphaCurrentSynapse id=\"glifCurrSynapse\" tau=\"2ms\" ibase=\"0.1nA\"/>"
        "  <blockingPlasticSynapse id=\"glifNmdaSynapse\" gbase=\"1nS\" erev=\"0mV\" tauRise=\"1ms\" tauDecay=\"50ms\"/>"
        "</neuroml>");

    write_temp_file("spikecorec_phase1_glif1_network_cell_type.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"Phase1GLIF1NetworkCellType\">"
        "  <include href=\"spikecorec_phase1_synapse_instances.xml\"/>"
        "  <ComponentType name=\"GLIF1Cell\" extends=\"baseCell\">"
        "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
        "    <Parameter name=\"gL\" dimension=\"conductance\"/>"
        "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
        "    <Parameter name=\"vth\" dimension=\"voltage\"/>"
        "    <Parameter name=\"vreset\" dimension=\"voltage\"/>"
        "    <Attachments name=\"synapses\" type=\"basePointCurrent\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
        "      <DerivedVariable name=\"iSyn\" dimension=\"current\" exposure=\"iSyn\" select=\"synapses[*]/i\" reduce=\"add\"/>"
        "      <TimeDerivative variable=\"v\" value=\"(gL * (EL - v) + iSyn) / C\"/>"
        "      <OnStart>"
        "        <StateAssignment variable=\"v\" value=\"EL\"/>"
        "      </OnStart>"
        "      <OnCondition test=\"v .gt. vth\">"
        "        <EventOut port=\"spike\"/>"
        "        <StateAssignment variable=\"v\" value=\"vreset\"/>"
        "      </OnCondition>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <GLIF1Cell id=\"glif1ExcCellInstance\" C=\"1.0e-10\" gL=\"1.0e-8\" EL=\"-0.07\" vth=\"-0.05\" vreset=\"-0.07\"/>"
        "  <GLIF1Cell id=\"glif1InhCellInstance\" C=\"1.0e-10\" gL=\"1.2e-8\" EL=\"-0.065\" vth=\"-0.05\" vreset=\"-0.065\"/>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_phase1_glif_network.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"Phase1GLIFNetwork\">"
        "  <include href=\"spikecorec_phase1_glif1_network_cell_type.nml\"/>"
        "  <network id=\"GLIFExcInhNet\">"
        "    <population id=\"ExcPop\" component=\"glif1ExcCellInstance\" size=\"80\"/>"
        "    <population id=\"InhPop\" component=\"glif1InhCellInstance\" size=\"20\"/>"
        "    <projection id=\"ExcToInh\" presynapticPopulation=\"ExcPop\" postsynapticPopulation=\"InhPop\" synapse=\"glifExcSynapse\">"
        "      <connection id=\"0\" preCellId=\"../ExcPop/0/glif1ExcCellInstance\" postCellId=\"../InhPop/0/glif1InhCellInstance\"/>"
        "    </projection>"
        "  </network>"
        "</neuroml>");

    NML_Parser parser;
    EXPECT_NO_THROW(parser.parse(top_path));

    // The user-defined cell type, merged from two include levels deep, and
    // classified as a CellType.
    ASSERT_EQ(parser.library.count("GLIF1Cell"), 1u);
    ASSERT_TRUE(std::holds_alternative<CellType>(parser.library.at("GLIF1Cell")));
    EXPECT_GT(std::get<CellType>(parser.library.at("GLIF1Cell")).attachments.size(), 0u);

    // The A1 std-lib synapse types (current-based, conductance-based, and
    // per-edge/NMDA) are all still available alongside the user's types,
    // each classified as a SynapseType (extends "baseSynapse").
    ASSERT_GT(parser.library.count("alphaCurrentSynapse"), 0u);
    ASSERT_GT(parser.library.count("expOneSynapse"), 0u);
    ASSERT_GT(parser.library.count("blockingPlasticSynapse"), 0u);
    EXPECT_TRUE(std::holds_alternative<SynapseType>(parser.library.at("alphaCurrentSynapse")));
    EXPECT_TRUE(std::holds_alternative<SynapseType>(parser.library.at("expOneSynapse")));
    EXPECT_TRUE(std::holds_alternative<SynapseType>(parser.library.at("blockingPlasticSynapse")));

    // expOneSynapse's own Dynamics directly declares its conductance state
    // and decay, plus the arrival handler that bumps it (§3.2 OnEvent). Its
    // postsynaptic-`v` Requirement is declared on its parent,
    // baseVoltageDepSynapse (PARSE doesn't flatten `extends`; that merge is
    // RESOLVE's job, ticket #49) — also a SynapseType in its own right.
    SynapseType &exp_one_synapse = std::get<SynapseType>(parser.library.at("expOneSynapse"));
    EXPECT_GT(exp_one_synapse.state_variables.size(), 0u);
    EXPECT_GT(exp_one_synapse.time_derivatives.size(), 0u);
    EXPECT_GT(exp_one_synapse.on_events.size(), 0u);

    ASSERT_TRUE(std::holds_alternative<SynapseType>(parser.library.at("baseVoltageDepSynapse")));
    EXPECT_GT(std::get<SynapseType>(parser.library.at("baseVoltageDepSynapse")).requirements.size(), 0u);

    // The whole include graph (top-level network, the cell-type file, and
    // its own nested synapse-instances include) is spliced into one shared
    // tree hanging off root_node.
    const NML_Node *network = find_child_by_tag(*parser.root_node, "network");
    ASSERT_NE(network, nullptr);
    const NML_Node *exc_population = find_child_by_tag(*network, "population");
    ASSERT_NE(exc_population, nullptr);
    EXPECT_EQ(get_string_attribute(*exc_population, "component"), "glif1ExcCellInstance");
    EXPECT_NE(find_child_by_tag(*network, "projection"), nullptr);

    EXPECT_NE(find_child_by_tag(*parser.root_node, "GLIF1Cell"), nullptr);
    EXPECT_NE(find_child_by_tag(*parser.root_node, "expOneSynapse"), nullptr);
    EXPECT_NE(find_child_by_tag(*parser.root_node, "blockingPlasticSynapse"), nullptr);
}
