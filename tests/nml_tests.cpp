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

String get_string_attribute(const ComponentType &component, const String &key) {
    auto entry = component.attributes.find(key);
    if (entry == component.attributes.end()) return "";
    return std::any_cast<String>(entry->second);
}

} // namespace

// ── ComponentType ────────────────────────────────────────────

TEST(ComponentType, construction) {
    ComponentType component("ComponentType");
    EXPECT_EQ(component.tag, "ComponentType");
    EXPECT_TRUE(component.attributes.empty());
    EXPECT_TRUE(component.body.empty());
}

TEST(ComponentType, add_attribute_stores_any_value) {
    ComponentType component("ComponentType");
    component.add_attribute("name", String("iafCell"));

    auto entry = component.attributes.find("name");
    ASSERT_NE(entry, component.attributes.end());
    EXPECT_EQ(std::any_cast<String>(entry->second), "iafCell");
}

TEST(ComponentType, nest_appends_child) {
    ComponentType parent("ComponentType");
    ComponentType child("Parameter");
    child.add_attribute("name", String("a"));

    parent.nest(child);

    ASSERT_EQ(parent.body.size(), 1u);
    EXPECT_EQ(parent.body[0].tag, "Parameter");
    EXPECT_EQ(get_string_attribute(parent.body[0], "name"), "a");
}

// Directly exercises deep-copy correctness: mutating the original after the
// copy is taken must not affect the copy, for both `attributes` (std::any,
// copies itself) and `body` (Vector<ComponentType>, must recurse).
TEST(ComponentType, copy_constructor_deep_copies_body_and_attributes) {
    ComponentType parent("ComponentType");
    parent.add_attribute("name", String("testCell"));

    ComponentType child("Parameter");
    child.add_attribute("name", String("a"));
    parent.nest(child);

    ComponentType copy(parent);

    parent.add_attribute("name", String("mutated"));
    parent.body.clear();

    EXPECT_EQ(copy.tag, "ComponentType");
    EXPECT_EQ(get_string_attribute(copy, "name"), "testCell");
    ASSERT_EQ(copy.body.size(), 1u);
    EXPECT_EQ(copy.body[0].tag, "Parameter");
    EXPECT_EQ(get_string_attribute(copy.body[0], "name"), "a");
}

// ── xml_node_to_component_type ──────────────────────────────

TEST(XmlNodeToComponentType, parses_tag_attributes_and_nested_children) {
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

    ComponentType component = xml_node_to_component_type(component_type_node);

    EXPECT_EQ(component.tag, "ComponentType");
    EXPECT_EQ(get_string_attribute(component, "name"), "testCell");
    EXPECT_EQ(get_string_attribute(component, "extends"), "baseCell");

    ASSERT_EQ(component.body.size(), 2u);
    EXPECT_EQ(component.body[0].tag, "Parameter");
    EXPECT_EQ(get_string_attribute(component.body[0], "name"), "a");
    EXPECT_EQ(get_string_attribute(component.body[0], "dimension"), "none");
    EXPECT_EQ(component.body[1].tag, "Parameter");
    EXPECT_EQ(get_string_attribute(component.body[1], "name"), "b");

    xmlFreeDoc(document);
}

// ── NML_StandardLibrary ─────────────────────────────────────

TEST(NmlStandardLibrary, standard_library_path_is_baked_and_exists) {
    NML_StandardLibrary library;
    EXPECT_FALSE(library.STANDARD_LIBRARY_PATH.empty());
    EXPECT_TRUE(std::filesystem::exists(library.STANDARD_LIBRARY_PATH));
}

// Acceptance criterion: "Core types resolve fully offline" — load_library
// only ever reads STANDARD_LIBRARY_PATH off the local filesystem, no network.
TEST(NmlStandardLibrary, load_library_loads_vendored_bundle_offline) {
    NML_StandardLibrary library;
    bool failed = library.load_library();

    EXPECT_FALSE(failed);
    EXPECT_GT(library.standard_library.size(), 100u);
}

TEST(NmlStandardLibrary, get_type_by_name_finds_a_real_core_type) {
    NML_StandardLibrary library;
    ASSERT_FALSE(library.load_library());

    ComponentType &iaf_cell = library.get_type_by_name("iafCell");
    EXPECT_EQ(iaf_cell.tag, "ComponentType");
    EXPECT_EQ(get_string_attribute(iaf_cell, "name"), "iafCell");
    EXPECT_GT(iaf_cell.body.size(), 0u);
}

// This is also the merge-point/lookup API the resolve pass (A4) will call;
// a missing type must fail loudly rather than return something usable.
TEST(NmlStandardLibrary, get_type_by_name_throws_on_missing_type) {
    NML_StandardLibrary library;
    ASSERT_FALSE(library.load_library());

    EXPECT_THROW(library.get_type_by_name("definitelyNotARealType_xyz"), std::runtime_error);
}

// ── validate_against_schema (ticket #8 [A2]) ────────────────

namespace {

String write_temp_file(const String &filename, const String &contents) {
    String path = (std::filesystem::temp_directory_path() / filename).string();
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

} // namespace

TEST(ValidateAgainstSchema, schema_path_is_baked_and_exists) {
    EXPECT_FALSE(NML_SCHEMA_PATH.empty());
    EXPECT_TRUE(std::filesystem::exists(NML_SCHEMA_PATH));
}

TEST(ValidateAgainstSchema, accepts_a_conformant_neuroml_document) {
    String path = write_temp_file("spikecorec_valid_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"TestDoc\">"
        "  <izhikevichCell id=\"izTest\" v0=\"-70mV\" thresh=\"30mV\" a=\"0.02\" b=\"0.2\" c=\"-65\" d=\"6\"/>"
        "</neuroml>");

    EXPECT_TRUE(validate_against_schema(path));
}

TEST(ValidateAgainstSchema, rejects_a_document_with_an_unknown_element) {
    String path = write_temp_file("spikecorec_invalid_test.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"TestDoc\">"
        "  <thisTagDoesNotExistInSchema id=\"oops\"/>"
        "</neuroml>");

    EXPECT_FALSE(validate_against_schema(path));
}

// A LEMS file (root <Lems>) is a real, well-formed XML document, but it is
// not a NeuroML2 document — validating it against the NeuroML2 XSD must
// still fail rather than silently pass.
TEST(ValidateAgainstSchema, rejects_a_wellformed_but_wrong_schema_document) {
    NML_StandardLibrary library;
    String lems_file = library.STANDARD_LIBRARY_PATH + "/Cells.xml";
    EXPECT_FALSE(validate_against_schema(lems_file));
}

TEST(ValidateAgainstSchema, rejects_a_missing_file) {
    EXPECT_FALSE(validate_against_schema("/tmp/definitely_not_a_real_file_xyz.nml"));
}
