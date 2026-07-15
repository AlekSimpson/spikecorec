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

TEST(NmlParser, get_type_by_name_finds_a_real_core_type) {
    NML_Parser parser;
    ASSERT_FALSE(parser.load_standard_library());

    NML_Node &iaf_cell = parser.get_type_by_name("iafCell");
    EXPECT_EQ(iaf_cell.tag_name, "ComponentType");
    EXPECT_EQ(get_string_attribute(iaf_cell, "name"), "iafCell");
    EXPECT_GT(iaf_cell.body.size(), 0u);
}

// This is also the merge-point/lookup API the resolve pass (A4) will call;
// a missing type must fail loudly rather than return something usable.
TEST(NmlParser, get_type_by_name_throws_on_missing_type) {
    NML_Parser parser;
    ASSERT_FALSE(parser.load_standard_library());

    EXPECT_THROW(parser.get_type_by_name("definitelyNotARealType_xyz"), std::runtime_error);
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
    EXPECT_EQ(parser.library.at("parseTestInlineCell").body.size(), 1u);

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
    EXPECT_EQ(parser.library.at("parseTestIncludedCell").body.size(), 1u);

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
