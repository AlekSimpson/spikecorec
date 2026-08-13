#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include "spikecorec/nml/nml.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

namespace {

// Fixtures are written to disk because the parser's unit of work is a file: include
// resolution, canonical-path deduping and cycle detection are all path behaviour, and a
// string-fed parser would test none of it. Each fixture gets its own directory so tests
// cannot see each other's files.
class FixtureDirectory {
public:
    explicit FixtureDirectory(const String &test_name) {
        root_ = filesystem::temp_directory_path() / "spikecorec_nml_tests" / test_name;
        filesystem::remove_all(root_);
        filesystem::create_directories(root_);
    }

    ~FixtureDirectory() {
        std::error_code ignored;
        filesystem::remove_all(root_, ignored);
    }

    FixtureDirectory(const FixtureDirectory &) = delete;
    FixtureDirectory &operator=(const FixtureDirectory &) = delete;

    // Returns the absolute path of the written file.
    String write(const String &relative_name, const String &contents) const {
        filesystem::path destination = root_ / relative_name;
        filesystem::create_directories(destination.parent_path());

        ofstream file(destination);
        file << contents;
        file.close();

        return destination.string();
    }

    String path_of(const String &relative_name) const {
        return (root_ / relative_name).string();
    }

private:
    filesystem::path root_;
};

bool standard_library_available() {
    NML_Parser parser;
    return !parser.STANDARD_LIBRARY_PATH.empty() &&
           filesystem::exists(parser.STANDARD_LIBRARY_PATH);
}

// A complete, simulable two-population model. Reused by the export tests so each one
// asserts on the same known-good document rather than inventing its own.
String two_population_network_nml() {
    return R"(<neuroml id="testnet">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <izhikevich2007Cell id="cell1" C="100pF" v0="-60mV" k="0.7nS_per_mV"
                        vr="-60mV" vt="-40mV" vpeak="35mV" a="0.03per_ms"
                        b="-2nS" c="-50mV" d="100pA"/>
    <expOneSynapse id="syn0" gbase="0.5nS" erev="0mV" tauDecay="3ms"/>
    <pulseGenerator id="pg0" delay="10ms" duration="40ms" amplitude="0.5nA"/>

    <network id="net1">
        <population id="pop1" component="cell0" size="3"/>
        <population id="pop2" component="cell1" size="2"/>

        <projection id="proj1" presynapticPopulation="pop1"
                    postsynapticPopulation="pop2" synapse="syn0">
            <connectionWD id="0" preCellId="../pop1[0]" postCellId="../pop2[1]"
                          weight="2.5" delay="2ms"/>
            <connection id="1" preCellId="../pop1[2]" postCellId="../pop2[0]"/>
        </projection>

        <explicitInput target="pop1[0]" input="pg0"/>
        <inputList id="il1" component="pg0" population="pop2">
            <input id="0" target="../pop2[0]"/>
            <inputW id="1" target="../pop2[1]" weight="3.0"/>
        </inputList>
    </network>
</neuroml>
)";
}

String lems_simulation_xml(const String &included_file) {
    return R"(<Lems>
    <Target component="sim1"/>
    <Include file=")" + included_file + R"("/>
    <Simulation id="sim1" length="100ms" step="0.01ms" target="net1" seed="1234">
        <OutputFile id="of1" fileName="out.dat">
            <OutputColumn id="c0" quantity="pop1[0]/v"/>
            <OutputColumn id="c1" quantity="pop2[1]/v"/>
        </OutputFile>
        <EventOutputFile id="ef1" fileName="spikes.dat" format="TIME_ID">
            <EventSelection id="0" select="pop2[1]" eventPort="spike"/>
        </EventOutputFile>
    </Simulation>
</Lems>
)";
}

// Names of a type's declarations of one tag, in the order the resolved type holds them.
Vector<String> declaration_names(const ComponentType &component_type,
                                 NML_DeclarationType tag_type) {
    Vector<String> names;
    for (const NML_Declaration *declaration : component_type.declarations.find_all(tag_type)) {
        names.push_back(declaration->value_or("name"));
    }
    return names;
}

// The result stores entities, not name->index maps, so tests scan for what they want the
// same way a consumer would.
const CellTypeSpecification *cell_type_named(const NML_ParseResult &result, const String &name) {
    for (const CellTypeSpecification &cell_type : result.cell_types) {
        if (cell_type.name == name) return &cell_type;
    }
    return nullptr;
}

const SynapseTypeSpecification *synapse_type_named(const NML_ParseResult &result,
                                                   const String &name) {
    for (const SynapseTypeSpecification &synapse_type : result.synapse_types) {
        if (synapse_type.name == name) return &synapse_type;
    }
    return nullptr;
}

const ComponentPrototype *prototype_named(const Vector<ComponentPrototype> &prototypes,
                                          const String &instance_id) {
    for (const ComponentPrototype &prototype : prototypes) {
        if (prototype.instance_id == instance_id) return &prototype;
    }
    return nullptr;
}

// A prototype's value for one named parameter, looked up through its type's column order.
f64 parameter_value(const Vector<String> &column_names, const Vector<Real> &values,
                    const String &parameter_name) {
    auto column = std::find(column_names.begin(), column_names.end(), parameter_name);
    EXPECT_NE(column, column_names.end()) << parameter_name;
    if (column == column_names.end()) return 0.0;
    return values[static_cast<usize>(column - column_names.begin())].float64;
}

} // namespace


// ── quantities and units ─────────────────────────────────────────────────────────

TEST(Quantity, splits_magnitude_from_unit_suffix) {
    EXPECT_EQ(split_quantity("-60mV").first, -60.0);
    EXPECT_EQ(split_quantity("-60mV").second, "mV");

    EXPECT_EQ(split_quantity("1.5e-3 mV").first, 1.5e-3);
    EXPECT_EQ(split_quantity("1.5e-3 mV").second, "mV");

    // A bare number has no suffix, and must not be mistaken for one.
    EXPECT_EQ(split_quantity("10").second, "");
    EXPECT_EQ(split_quantity("10").first, 10.0);
}

TEST(Quantity, scales_to_si_with_the_builtin_table) {
    EXPECT_DOUBLE_EQ(parse_quantity("-60mV"), -0.06);
    EXPECT_DOUBLE_EQ(parse_quantity("5ms"), 0.005);
    EXPECT_DOUBLE_EQ(parse_quantity("0.2nF"), 0.2e-9);
    EXPECT_DOUBLE_EQ(parse_quantity("10"), 10.0);
}

TEST(Quantity, malformed_input_yields_zero_rather_than_throwing) {
    EXPECT_DOUBLE_EQ(parse_quantity(""), 0.0);
    EXPECT_DOUBLE_EQ(parse_quantity("mV"), 0.0);
    EXPECT_DOUBLE_EQ(parse_quantity("   "), 0.0);
}

TEST(Quantity, declared_units_take_precedence_over_the_builtin_table) {
    FixtureDirectory fixture("declared_units");
    // A deliberately non-SI redefinition: if the document's <Unit> is being consulted,
    // "mV" scales by 1000 here rather than the built-in 1e-3.
    fixture.write("units.xml", R"(<Lems>
    <Unit symbol="mV" dimension="voltage" power="3"/>
    <Unit symbol="quux" dimension="none" power="0" scale="7"/>
</Lems>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("units.xml"));

    EXPECT_DOUBLE_EQ(parser.resolve_quantity("2mV"), 2000.0);
    EXPECT_DOUBLE_EQ(parser.resolve_quantity("3quux"), 21.0);

    // The built-in table still answers for symbols no document declared.
    EXPECT_DOUBLE_EQ(parser.resolve_quantity("5ms"), 0.005);

    // The free function is unaffected by what a parser ingested.
    EXPECT_DOUBLE_EQ(parse_quantity("2mV"), 0.002);
}

TEST(Quantity, unit_offset_is_applied) {
    FixtureDirectory fixture("unit_offset");
    fixture.write("units.xml", R"(<Lems>
    <Unit symbol="degC" dimension="temperature" offset="273.15"/>
</Lems>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("units.xml"));

    // Offsets are why the built-in scale-only table cannot express temperature.
    EXPECT_DOUBLE_EQ(parser.resolve_quantity("20degC"), 293.15);
    EXPECT_DOUBLE_EQ(parser.resolve_quantity("0degC"), 273.15);
}

TEST(Quantity, standard_library_units_are_ingested) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    NML_Parser parser;
    parser.parse_lems("");

    // NeuroMLCoreDimensions.xml declares these; scale, power and offset must all survive.
    EXPECT_DOUBLE_EQ(parser.resolve_quantity("20degC"), 293.15);
    EXPECT_DOUBLE_EQ(parser.resolve_quantity("-60mV"), -0.06);
    EXPECT_DOUBLE_EQ(parser.resolve_quantity("2min"), 120.0);
}


// ── ComponentType resolution ─────────────────────────────────────────────────────

TEST(Resolution, flattens_a_multi_hop_extends_chain) {
    FixtureDirectory fixture("multi_hop");
    fixture.write("types.xml", R"(<Lems>
    <ComponentType name="base_a">
        <Parameter name="alpha" dimension="none"/>
        <Parameter name="beta" dimension="none"/>
    </ComponentType>
    <ComponentType name="mid_b" extends="base_a">
        <Parameter name="gamma" dimension="none"/>
    </ComponentType>
    <ComponentType name="leaf_c" extends="mid_b">
        <Parameter name="delta" dimension="none"/>
    </ComponentType>
</Lems>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("types.xml"));
    parser.resolve_all_component_types();

    const ComponentType &leaf = parser.declared_component_types.at("leaf_c");
    EXPECT_EQ(leaf.extends, "mid_b");

    // Every ancestor's parameters, ancestors first, then the type's own.
    EXPECT_EQ(declaration_names(leaf, NML_DeclarationType::Parameter),
              (Vector<String>{"alpha", "beta", "gamma", "delta"}));
}

TEST(Resolution, resolves_a_forward_reference_within_one_file) {
    FixtureDirectory fixture("forward_reference");
    // The child is declared before the parent. Declaration order is not significant:
    // extends is a name reference resolved by lookup.
    fixture.write("types.xml", R"(<Lems>
    <ComponentType name="child_type" extends="parent_type">
        <Parameter name="own" dimension="none"/>
    </ComponentType>
    <ComponentType name="parent_type">
        <Parameter name="inherited" dimension="none"/>
    </ComponentType>
</Lems>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("types.xml"));
    parser.resolve_all_component_types();

    EXPECT_EQ(declaration_names(parser.declared_component_types.at("child_type"),
                                NML_DeclarationType::Parameter),
              (Vector<String>{"inherited", "own"}));
}

TEST(Resolution, resolves_extends_across_files) {
    FixtureDirectory fixture("cross_file");
    fixture.write("parent.xml", R"(<Lems>
    <ComponentType name="parent_type">
        <Parameter name="inherited" dimension="none"/>
    </ComponentType>
</Lems>
)");
    // No include: the two files are ingested independently, so resolution has to span
    // them from the global catalogue rather than from one document's scope.
    fixture.write("child.xml", R"(<Lems>
    <ComponentType name="child_type" extends="parent_type">
        <Parameter name="own" dimension="none"/>
    </ComponentType>
</Lems>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("child.xml"));
    parser.ingest_document(fixture.path_of("parent.xml"));
    parser.resolve_all_component_types();

    EXPECT_EQ(declaration_names(parser.declared_component_types.at("child_type"),
                                NML_DeclarationType::Parameter),
              (Vector<String>{"inherited", "own"}));
}

TEST(Resolution, a_redeclared_parameter_overrides_rather_than_duplicates) {
    FixtureDirectory fixture("override");
    fixture.write("types.xml", R"(<Lems>
    <ComponentType name="parent_type">
        <Parameter name="alpha" dimension="none"/>
        <Parameter name="beta" dimension="voltage"/>
        <Parameter name="gamma" dimension="none"/>
    </ComponentType>
    <ComponentType name="child_type" extends="parent_type">
        <Parameter name="beta" dimension="current"/>
    </ComponentType>
</Lems>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("types.xml"));
    parser.resolve_all_component_types();

    const ComponentType &child = parser.declared_component_types.at("child_type");

    // One beta, not two, and it keeps the ancestor's position in declaration order.
    EXPECT_EQ(declaration_names(child, NML_DeclarationType::Parameter),
              (Vector<String>{"alpha", "beta", "gamma"}));

    // The most-derived declaration wins.
    const NML_Declaration *beta =
            child.declarations.find(NML_DeclarationType::Parameter, "beta");
    ASSERT_NE(beta, nullptr);
    EXPECT_EQ(beta->value_or("dimension"), "current");
}

TEST(Resolution, a_state_variable_overrides_an_inherited_parameter_of_the_same_name) {
    FixtureDirectory fixture("cross_tag_override");
    // Seven tags share the `variables` namespace, so this is a collision, not a pair.
    fixture.write("types.xml", R"(<Lems>
    <ComponentType name="parent_type">
        <Parameter name="tau" dimension="time"/>
    </ComponentType>
    <ComponentType name="child_type" extends="parent_type">
        <Dynamics>
            <StateVariable name="tau" dimension="time"/>
        </Dynamics>
    </ComponentType>
</Lems>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("types.xml"));
    parser.resolve_all_component_types();

    const ComponentType &child = parser.declared_component_types.at("child_type");
    EXPECT_TRUE(declaration_names(child, NML_DeclarationType::Parameter).empty());
    EXPECT_EQ(declaration_names(child, NML_DeclarationType::StateVariable),
              (Vector<String>{"tau"}));
}

TEST(Resolution, fixed_pins_an_inherited_parameter) {
    FixtureDirectory fixture("fixed");
    fixture.write("types.xml", R"(<Lems>
    <ComponentType name="parent_type">
        <Parameter name="tau" dimension="time"/>
        <Parameter name="loose" dimension="time"/>
    </ComponentType>
    <ComponentType name="child_type" extends="parent_type">
        <Fixed parameter="tau" value="10ms"/>
    </ComponentType>
</Lems>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("types.xml"));
    parser.resolve_all_component_types();

    const ComponentType &child = parser.declared_component_types.at("child_type");

    const NML_Declaration *tau =
            child.declarations.find(NML_DeclarationType::Parameter, "tau");
    ASSERT_NE(tau, nullptr);
    EXPECT_EQ(tau->value_or("value"), "10ms");

    // An unpinned parameter is untouched.
    const NML_Declaration *loose =
            child.declarations.find(NML_DeclarationType::Parameter, "loose");
    ASSERT_NE(loose, nullptr);
    EXPECT_EQ(loose->value_or("value"), "");

    // The parent itself must not acquire the pin.
    const NML_Declaration *parent_tau =
            parser.declared_component_types.at("parent_type")
                    .declarations.find(NML_DeclarationType::Parameter, "tau");
    ASSERT_NE(parent_tau, nullptr);
    EXPECT_EQ(parent_tau->value_or("value"), "");
}

TEST(Resolution, an_extends_cycle_throws) {
    FixtureDirectory fixture("cycle");
    fixture.write("types.xml", R"(<Lems>
    <ComponentType name="type_a" extends="type_b"/>
    <ComponentType name="type_b" extends="type_a"/>
</Lems>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("types.xml"));

    EXPECT_THROW(parser.resolve_all_component_types(), std::runtime_error);
}

TEST(Resolution, an_unresolved_extends_throws_naming_the_missing_type) {
    FixtureDirectory fixture("missing_parent");
    fixture.write("types.xml", R"(<Lems>
    <ComponentType name="orphan_type" extends="no_such_type"/>
</Lems>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("types.xml"));

    try {
        parser.resolve_all_component_types();
        FAIL() << "expected resolve_all_component_types to throw";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(String(error.what()).find("no_such_type"), String::npos);
    }
}

TEST(Resolution, declaration_order_within_a_tag_group_follows_source_order) {
    FixtureDirectory fixture("declaration_order");
    fixture.write("types.xml", R"(<Lems>
    <ComponentType name="ordered_type">
        <Parameter name="first" dimension="none"/>
        <Parameter name="second" dimension="none"/>
        <Parameter name="third" dimension="none"/>
        <Parameter name="fourth" dimension="none"/>
    </ComponentType>
</Lems>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("types.xml"));
    parser.resolve_all_component_types();

    const ComponentType &ordered = parser.declared_component_types.at("ordered_type");

    // Parameter order is the column order of the exported starting-parameter rows, so a
    // reversal here silently mislabels every parameter the engine reads.
    EXPECT_EQ(ordered.ordered_parameter_names(),
              (Vector<String>{"first", "second", "third", "fourth"}));
}


// ── includes ─────────────────────────────────────────────────────────────────────

TEST(Include, resolves_relative_to_the_including_file_not_the_process_directory) {
    FixtureDirectory fixture("relative_include");
    fixture.write("nested/leaf.xml", R"(<Lems>
    <ComponentType name="leaf_type">
        <Parameter name="leaf_parameter" dimension="none"/>
    </ComponentType>
</Lems>
)");
    fixture.write("nested/branch.xml", R"(<Lems>
    <Include file="leaf.xml"/>
</Lems>
)");
    fixture.write("root.xml", R"(<Lems>
    <Include file="nested/branch.xml"/>
</Lems>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("root.xml"));
    parser.resolve_all_component_types();

    // branch.xml names "leaf.xml" with no directory, so it only resolves if the include
    // is taken relative to branch.xml's own directory.
    EXPECT_EQ(parser.declared_component_types.count("leaf_type"), 1u);
}

TEST(Include, a_cycle_terminates_instead_of_recursing_forever) {
    FixtureDirectory fixture("include_cycle");
    fixture.write("first.xml", R"(<Lems>
    <Include file="second.xml"/>
    <ComponentType name="first_type"/>
</Lems>
)");
    fixture.write("second.xml", R"(<Lems>
    <Include file="first.xml"/>
    <ComponentType name="second_type"/>
</Lems>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("first.xml"));

    EXPECT_EQ(parser.component_type_catalogue.count("first_type"), 1u);
    EXPECT_EQ(parser.component_type_catalogue.count("second_type"), 1u);
}

TEST(Include, a_diamond_include_is_idempotent_and_not_a_redefinition) {
    FixtureDirectory fixture("diamond");
    fixture.write("shared.xml", R"(<Lems>
    <ComponentType name="shared_type"/>
</Lems>
)");
    fixture.write("left.xml", R"(<Lems>
    <Include file="shared.xml"/>
</Lems>
)");
    fixture.write("right.xml", R"(<Lems>
    <Include file="./shared.xml"/>
</Lems>
)");
    fixture.write("root.xml", R"(<Lems>
    <Include file="left.xml"/>
    <Include file="right.xml"/>
</Lems>
)");

    // Deduping on the raw href would let "shared.xml" and "./shared.xml" through as two
    // different files and turn shared_type into a spurious duplicate.
    NML_Parser parser;
    EXPECT_NO_THROW(parser.ingest_document(fixture.path_of("root.xml")));
    EXPECT_EQ(parser.component_type_catalogue.count("shared_type"), 1u);
}

TEST(Include, a_genuine_redefinition_across_two_files_throws) {
    FixtureDirectory fixture("redefinition");
    fixture.write("first.xml", R"(<Lems>
    <ComponentType name="clashing_type">
        <Parameter name="from_first" dimension="none"/>
    </ComponentType>
</Lems>
)");
    fixture.write("second.xml", R"(<Lems>
    <ComponentType name="clashing_type">
        <Parameter name="from_second" dimension="none"/>
    </ComponentType>
</Lems>
)");
    fixture.write("root.xml", R"(<Lems>
    <Include file="first.xml"/>
    <Include file="second.xml"/>
</Lems>
)");

    NML_Parser parser;
    try {
        parser.ingest_document(fixture.path_of("root.xml"));
        FAIL() << "expected a duplicate ComponentType to throw";
    } catch (const std::runtime_error &error) {
        // Silent first-wins would make the model depend on ingest order.
        EXPECT_NE(String(error.what()).find("clashing_type"), String::npos);
    }
}


// ── dynamics extraction ──────────────────────────────────────────────────────────

TEST(Dynamics, regime_scoped_instructions_carry_their_regime_and_gating_condition) {
    FixtureDirectory fixture("regimes");
    fixture.write("types.xml", R"(<Lems>
    <ComponentType name="regime_cell">
        <Dynamics>
            <StateVariable name="v" dimension="voltage"/>
            <Regime name="integrating" initial="true">
                <TimeDerivative variable="v" value="(vrest - v) / tau"/>
                <OnCondition test="v .gt. vthresh">
                    <StateAssignment variable="v" value="vreset"/>
                    <EventOut port="spike"/>
                    <Transition regime="refractory"/>
                </OnCondition>
            </Regime>
            <Regime name="refractory">
                <OnEntry>
                    <StateAssignment variable="elapsed" value="0"/>
                </OnEntry>
                <TimeDerivative variable="elapsed" value="1"/>
            </Regime>
        </Dynamics>
    </ComponentType>
</Lems>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("types.xml"));
    parser.resolve_all_component_types();

    const ComponentType &cell = parser.declared_component_types.at("regime_cell");

    // A Regime carries the cell's derivatives; an extractor that stops descending at
    // <Regime> loses the entire dynamics.
    const NML_Declaration *integrating =
            cell.declarations.find(NML_DeclarationType::Regime, "integrating");
    ASSERT_NE(integrating, nullptr);
    EXPECT_FALSE(integrating->children.empty());

    // Walk the resolved declarations the way the export does and check the tagging.
    bool saw_regime_scoped_derivative = false;
    bool saw_gated_assignment = false;
    bool saw_gated_event = false;

    for (const NML_Declaration &declaration : cell.declarations.for_all()) {
        if (declaration.tag_type != NML_DeclarationType::Regime) continue;
        if (declaration.value_or("name") != "integrating") continue;

        for (const NML_Declaration &child : declaration.children) {
            if (child.tag_type == NML_DeclarationType::TimeDerivative &&
                child.value_or("variable") == "v") {
                saw_regime_scoped_derivative = true;
            }
            if (child.tag_type != NML_DeclarationType::OnCondition) continue;

            EXPECT_EQ(child.value_or("test"), "v .gt. vthresh");
            for (const NML_Declaration &fired : child.children) {
                if (fired.tag_type == NML_DeclarationType::StateAssignment) {
                    saw_gated_assignment = true;
                }
                if (fired.tag_type == NML_DeclarationType::EventOut) saw_gated_event = true;
            }
        }
    }

    EXPECT_TRUE(saw_regime_scoped_derivative);
    EXPECT_TRUE(saw_gated_assignment);
    EXPECT_TRUE(saw_gated_event);
}

TEST(Dynamics, exported_program_tags_stages_regimes_and_conditions) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("exported_dynamics");
    fixture.write("net.nml", two_population_network_nml());
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    const CellTypeSpecification *iaf = cell_type_named(result, "iafCell");
    ASSERT_NE(iaf, nullptr);
    ASSERT_FALSE(iaf->dynamics.empty());

    // iafCell writes v from two different handlers:
    //   <OnStart>                          <StateAssignment variable="v" value="leakReversal"/>
    //   <OnCondition test="v .gt. thresh"> <StateAssignment variable="v" value="reset"/>
    // Both are StateAssignments on v, so the tag alone cannot tell them apart. The stage
    // has to come from the enclosing handler -- tagging the OnStart one as Reset would
    // make the assembled tick re-apply it every tick and pin v to leakReversal.
    bool saw_voltage_derivative = false;
    bool saw_initialisation = false;
    bool saw_conditional_reset = false;

    for (const DynamicsInstruction &instruction : iaf->dynamics) {
        if (instruction.source_tag == NML_DeclarationType::TimeDerivative &&
            instruction.target == "v") {
            EXPECT_EQ(instruction.stage, DynamicsStage::Integrate);
            EXPECT_EQ(instruction.expression, "iMemb / C");
            saw_voltage_derivative = true;
        }

        if (instruction.source_tag != NML_DeclarationType::StateAssignment) continue;
        if (instruction.target != "v") continue;

        if (instruction.expression == "leakReversal") {
            EXPECT_EQ(instruction.stage, DynamicsStage::Initialize);
            // Nothing gates an OnStart body; it is located by stage.
            EXPECT_EQ(instruction.condition, "");
            saw_initialisation = true;
        } else if (instruction.expression == "reset") {
            EXPECT_EQ(instruction.stage, DynamicsStage::Reset);
            // A reset is only meaningful with the condition that fires it.
            EXPECT_EQ(instruction.condition, "v .gt. thresh");
            saw_conditional_reset = true;
        }
    }

    EXPECT_TRUE(saw_voltage_derivative);
    EXPECT_TRUE(saw_initialisation);
    EXPECT_TRUE(saw_conditional_reset);

    // The spike EventOut is gated by the same threshold condition.
    bool saw_gated_spike = false;
    for (const DynamicsInstruction &instruction : iaf->dynamics) {
        if (instruction.source_tag != NML_DeclarationType::EventOut) continue;
        EXPECT_EQ(instruction.stage, DynamicsStage::Emit);
        EXPECT_EQ(instruction.target, "spike");
        EXPECT_EQ(instruction.condition, "v .gt. thresh");
        saw_gated_spike = true;
    }
    EXPECT_TRUE(saw_gated_spike);
}

TEST(Dynamics, an_on_event_body_is_gated_by_its_port) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("on_event");
    fixture.write("net.nml", two_population_network_nml());
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    const SynapseTypeSpecification *synapse = synapse_type_named(result, "expOneSynapse");
    ASSERT_NE(synapse, nullptr);

    // expOneSynapse steps its conductance when a spike arrives:
    //   <OnEvent port="in"><StateAssignment variable="g" value="g + gbase"/></OnEvent>
    // Which port delivered the event is the gate, and it has to survive flattening.
    bool saw_arrival_assignment = false;
    for (const DynamicsInstruction &instruction : synapse->dynamics) {
        if (instruction.source_tag != NML_DeclarationType::StateAssignment) continue;
        if (instruction.condition.empty()) continue;

        EXPECT_EQ(instruction.stage, DynamicsStage::Arrival);
        EXPECT_EQ(instruction.condition, "in");
        saw_arrival_assignment = true;
    }
    EXPECT_TRUE(saw_arrival_assignment);
}


// ── runtime categories ───────────────────────────────────────────────────────────

TEST(Category, standard_library_types_classify_by_their_extends_chain) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    NML_Parser parser;
    parser.parse_lems("");

    auto category_of = [&](const String &type_name) {
        auto entry = parser.declared_component_types.find(type_name);
        EXPECT_NE(entry, parser.declared_component_types.end()) << type_name;
        if (entry == parser.declared_component_types.end()) return RuntimeCategory::General;
        return entry->second.runtime_category;
    };

    EXPECT_EQ(category_of("iafCell"), RuntimeCategory::Cell);
    EXPECT_EQ(category_of("izhikevich2007Cell"), RuntimeCategory::Cell);
    EXPECT_EQ(category_of("expOneSynapse"), RuntimeCategory::Synapse);
    EXPECT_EQ(category_of("expTwoSynapse"), RuntimeCategory::Synapse);
    EXPECT_EQ(category_of("pulseGenerator"), RuntimeCategory::InputScheme);
    EXPECT_EQ(category_of("spikeArray"), RuntimeCategory::InputScheme);

    EXPECT_EQ(category_of("network"), RuntimeCategory::Graph);
    EXPECT_EQ(category_of("population"), RuntimeCategory::Graph);
    EXPECT_EQ(category_of("populationList"), RuntimeCategory::Graph);
    EXPECT_EQ(category_of("projection"), RuntimeCategory::Graph);
    EXPECT_EQ(category_of("inputList"), RuntimeCategory::Graph);
    EXPECT_EQ(category_of("explicitInput"), RuntimeCategory::Graph);

    // connection and input have no `extends` at all, so they only classify if they are
    // category roots in their own right.
    EXPECT_EQ(category_of("connection"), RuntimeCategory::Graph);
    EXPECT_EQ(category_of("connectionWD"), RuntimeCategory::Graph);
    EXPECT_EQ(category_of("synapticConnection"), RuntimeCategory::Graph);
    EXPECT_EQ(category_of("input"), RuntimeCategory::Graph);
    EXPECT_EQ(category_of("inputW"), RuntimeCategory::Graph);
}

TEST(Category, the_whole_standard_library_resolves) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    NML_Parser parser;
    ASSERT_NO_THROW(parser.parse_lems(""));

    // Every catalogued type must resolve; a stale per-file resolver silently leaves
    // default-constructed entries behind.
    EXPECT_EQ(parser.declared_component_types.size(), parser.component_type_catalogue.size());
    EXPECT_GT(parser.declared_component_types.size(), 250u);

    for (const auto &[type_name, component_type] : parser.declared_component_types) {
        EXPECT_EQ(component_type.name, type_name);

        if (component_type.extends.empty()) continue;
        // A resolved child must be at least as rich as its parent.
        auto parent = parser.declared_component_types.find(component_type.extends);
        ASSERT_NE(parent, parser.declared_component_types.end()) << type_name;
        EXPECT_GE(component_type.declarations.for_all().size(),
                  parent->second.declarations.for_all().size()) << type_name;
    }
}


// ── export ───────────────────────────────────────────────────────────────────────

TEST(Export, reads_simulation_settings_from_the_lems_target) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("simulation_settings");
    fixture.write("net.nml", two_population_network_nml());
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    EXPECT_EQ(result.simulation_component_id, "sim1");
    EXPECT_EQ(result.target_network_id, "net1");
    EXPECT_DOUBLE_EQ(result.step_dt, 1e-5);            // 0.01ms
    EXPECT_DOUBLE_EQ(result.simulation_duration, 0.1); // 100ms
    EXPECT_EQ(result.total_tick_count, 10000);
    ASSERT_TRUE(result.random_seed.has_value());
    EXPECT_EQ(*result.random_seed, 1234u);
}

TEST(Export, numbers_populations_contiguously_in_document_order) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("population_layout");
    fixture.write("net.nml", two_population_network_nml());
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    EXPECT_EQ(result.neurons.size(), 5u);
    ASSERT_EQ(result.populations.size(), 2u);

    EXPECT_EQ(result.populations[0].population_name, "pop1");
    EXPECT_EQ(result.populations[0].first_neuron_index, 0);
    EXPECT_EQ(result.populations[0].neuron_count, 3);

    EXPECT_EQ(result.populations[1].population_name, "pop2");
    EXPECT_EQ(result.populations[1].first_neuron_index, 3);
    EXPECT_EQ(result.populations[1].neuron_count, 2);

    // Every neuron carries the type, prototype and population it belongs to.
    s64 pop1_type = result.populations[0].cell_type_index;
    s64 pop2_type = result.populations[1].cell_type_index;
    EXPECT_NE(pop1_type, pop2_type);

    for (usize neuron = 0; neuron < result.neurons.size(); neuron += 1) {
        s64 expected_population = neuron < 3 ? 0 : 1;
        const PopulationLayout &population =
                result.populations[static_cast<usize>(expected_population)];

        EXPECT_EQ(result.neurons[neuron].population_index, expected_population) << neuron;
        EXPECT_EQ(result.neurons[neuron].cell_type_index, population.cell_type_index) << neuron;
        EXPECT_EQ(result.neurons[neuron].prototype_index, population.prototype_index) << neuron;
    }
}

TEST(Export, cell_type_entries_are_self_describing) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("cell_types");
    fixture.write("net.nml", two_population_network_nml());
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    ASSERT_EQ(result.cell_types.size(), 2u);
    EXPECT_NE(cell_type_named(result, "iafCell"), nullptr);
    EXPECT_NE(cell_type_named(result, "izhikevich2007Cell"), nullptr);

    for (const CellTypeSpecification &cell_type : result.cell_types) {
        EXPECT_FALSE(cell_type.name.empty());
        EXPECT_FALSE(cell_type.state_variable_names.empty()) << cell_type.name;
        EXPECT_FALSE(cell_type.parameter_names.empty()) << cell_type.name;
        EXPECT_FALSE(cell_type.dynamics.empty()) << cell_type.name;
    }

    // Every prototype names a real type, and its row width matches that type's columns.
    ASSERT_FALSE(result.cell_prototypes.empty());
    for (const ComponentPrototype &prototype : result.cell_prototypes) {
        ASSERT_GE(prototype.type_index, 0);
        ASSERT_LT(prototype.type_index, static_cast<s64>(result.cell_types.size()));

        const CellTypeSpecification &cell_type =
                result.cell_types[static_cast<usize>(prototype.type_index)];
        EXPECT_EQ(prototype.starting_parameters.size(), cell_type.parameter_names.size())
                << prototype.instance_id;
    }

    // Every population points at a real prototype, and they agree on the type.
    for (const PopulationLayout &population : result.populations) {
        ASSERT_GE(population.prototype_index, 0);
        ASSERT_LT(population.prototype_index,
                  static_cast<s64>(result.cell_prototypes.size()));
        EXPECT_EQ(result.cell_prototypes[static_cast<usize>(population.prototype_index)]
                          .type_index,
                  population.cell_type_index);
    }
}

TEST(Export, parameters_are_converted_to_si_in_declared_column_order) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("parameter_values");
    fixture.write("net.nml", two_population_network_nml());
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    const CellTypeSpecification *iaf = cell_type_named(result, "iafCell");
    ASSERT_NE(iaf, nullptr);

    const ComponentPrototype *cell0 = prototype_named(result.cell_prototypes, "cell0");
    ASSERT_NE(cell0, nullptr);
    ASSERT_EQ(cell0->starting_parameters.size(), iaf->parameter_names.size());

    auto value_of = [&](const String &parameter_name) {
        return parameter_value(iaf->parameter_names, cell0->starting_parameters,
                               parameter_name);
    };

    // cell0's attributes, in SI.
    EXPECT_DOUBLE_EQ(value_of("leakReversal"), -0.05);       // -50mV
    EXPECT_DOUBLE_EQ(value_of("thresh"), -0.055);            // -55mV
    EXPECT_DOUBLE_EQ(value_of("reset"), -0.07);              // -70mV
    EXPECT_DOUBLE_EQ(value_of("C"), 0.2e-9);                 // 0.2nF
    EXPECT_DOUBLE_EQ(value_of("leakConductance"), 0.01e-6);  // 0.01uS
}

TEST(Export, cell_state_offsets_are_the_running_sum_of_state_sizes) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("state_offsets");
    fixture.write("net.nml", two_population_network_nml());
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    ASSERT_EQ(result.cell_types.size(), 2u);

    s32 expected_offset = 0;
    for (const CellTypeSpecification &cell_type : result.cell_types) {
        EXPECT_EQ(cell_type.state_offset, expected_offset) << cell_type.name;
        expected_offset += static_cast<s32>(cell_type.state_variable_names.size());
    }

    // The first type starts at zero and the offsets strictly advance.
    EXPECT_EQ(result.cell_types[0].state_offset, 0);
    EXPECT_GT(expected_offset, 0);

    // iafCell has one state variable (v); izhikevich2007Cell has two (v, u).
    const CellTypeSpecification *iaf = cell_type_named(result, "iafCell");
    ASSERT_NE(iaf, nullptr);
    EXPECT_EQ(iaf->state_variable_names, (Vector<String>{"v"}));
}

TEST(Export, synapse_types_are_classified_for_storage) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("synapse_types");
    fixture.write("net.nml", two_population_network_nml());
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    ASSERT_EQ(result.synapse_types.size(), 1u);
    const SynapseTypeSpecification *synapse = synapse_type_named(result, "expOneSynapse");
    ASSERT_NE(synapse, nullptr);

    // expOneSynapse computes g * (erev - v), so it needs the postsynaptic voltage. The
    // is_conductance_based classification this used to assert is commented out for now
    // (see SynapseTypeSpecification): conductance-based support is deferred, and the
    // generator refuses such a synapse by name instead of carrying a flag nothing acts on.
    // KernelCodegenSynapse.conductance_based_synapse_is_refused_by_name covers the refusal.

    // It carries no plasticity or block child, so its state superposes and it does not
    // need per-edge storage.
    EXPECT_FALSE(synapse->requires_per_edge_state);

    EXPECT_FALSE(synapse->state_variable_names.empty());
    EXPECT_FALSE(synapse->dynamics.empty());
}

TEST(Export, connections_resolve_to_global_neuron_indices_with_weight_and_delay) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("connections");
    fixture.write("net.nml", two_population_network_nml());
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    ASSERT_EQ(result.neurons.size(), 5u);

    // pop1[0] is neuron 0; pop2[1] is 3 + 1 = 4.
    ASSERT_EQ(result.neurons[0].outgoing_edges.size(), 1u);
    const NetworkEdge &weighted = result.neurons[0].outgoing_edges[0];
    EXPECT_EQ(weighted.target_neuron_index, 4);
    EXPECT_DOUBLE_EQ(weighted.weight, 2.5);
    // 2ms at dt=0.01ms. Truncation would give 199 here.
    EXPECT_EQ(weighted.delay_tick_count, 200);
    ASSERT_GE(weighted.synapse_type_index, 0);
    EXPECT_EQ(result.synapse_types[static_cast<usize>(weighted.synapse_type_index)].name,
              "expOneSynapse");
    ASSERT_GE(weighted.synapse_prototype_index, 0);
    EXPECT_EQ(result.synapse_prototypes[static_cast<usize>(weighted.synapse_prototype_index)]
                      .instance_id,
              "syn0");

    // pop1[2] is neuron 2; pop2[0] is neuron 3. A plain <connection> carries neither
    // weight nor delay, so it takes the defaults.
    ASSERT_EQ(result.neurons[2].outgoing_edges.size(), 1u);
    const NetworkEdge &plain = result.neurons[2].outgoing_edges[0];
    EXPECT_EQ(plain.target_neuron_index, 3);
    EXPECT_DOUBLE_EQ(plain.weight, 1.0);
    EXPECT_EQ(plain.delay_tick_count, 0);

    // Neurons with no outgoing edges stay empty rather than being dropped.
    EXPECT_TRUE(result.neurons[1].outgoing_edges.empty());
    EXPECT_TRUE(result.neurons[3].outgoing_edges.empty());
}

TEST(Export, adjacency_projection_matches_the_stored_edges) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("adjacency");
    fixture.write("net.nml", two_population_network_nml());
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    // The plain source -> targets form WeightMatrix consumes is derived, not stored, so
    // there is only one edge list that can be wrong.
    Vector<Vector<s64>> adjacency = build_adjacency_list(result);

    ASSERT_EQ(adjacency.size(), result.neurons.size());
    for (usize source = 0; source < result.neurons.size(); source += 1) {
        const Vector<NetworkEdge> &edges = result.neurons[source].outgoing_edges;
        ASSERT_EQ(adjacency[source].size(), edges.size()) << source;

        for (usize edge = 0; edge < edges.size(); edge += 1) {
            EXPECT_EQ(adjacency[source][edge], edges[edge].target_neuron_index);
        }
    }

    EXPECT_EQ(adjacency[0], (Vector<s64>{4}));
    EXPECT_EQ(adjacency[2], (Vector<s64>{3}));
    EXPECT_TRUE(adjacency[1].empty());
}

TEST(Export, inputs_resolve_their_targets_and_times) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("inputs");
    fixture.write("net.nml", two_population_network_nml());
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    // One profile from <explicitInput>, one from <inputList>.
    ASSERT_EQ(result.input_profiles.size(), 2u);

    const SimulationInputConfig &explicit_input = result.input_profiles[0];
    EXPECT_EQ(explicit_input.input_component_id, "pg0");
    EXPECT_EQ(explicit_input.input_component_type_name, "pulseGenerator");
    EXPECT_TRUE(explicit_input.continuous_current_injection);
    EXPECT_DOUBLE_EQ(explicit_input.amplitude, 0.5e-9); // 0.5nA
    ASSERT_EQ(explicit_input.targets.size(), 1u);
    EXPECT_EQ(explicit_input.targets[0].neuron_index, 0);
    // A continuous injector reports its span as exactly two event_ticks entries,
    // {delay, delay + duration}.
    // 10ms and 10ms + 40ms at dt = 0.01ms. Truncation would give 999 and 4998.
    ASSERT_EQ(explicit_input.targets[0].event_ticks.size(), 2u);
    EXPECT_EQ(explicit_input.targets[0].event_ticks.front(), 1000);
    EXPECT_EQ(explicit_input.targets[0].event_ticks.back(), 5000);

    const SimulationInputConfig &input_list = result.input_profiles[1];
    ASSERT_EQ(input_list.targets.size(), 2u);
    EXPECT_EQ(input_list.targets[0].neuron_index, 3);
    EXPECT_EQ(input_list.targets[1].neuron_index, 4);
    // <input> carries no weight and defaults to 1; <inputW> carries 3.
    EXPECT_DOUBLE_EQ(input_list.targets[0].weight, 1.0);
    EXPECT_DOUBLE_EQ(input_list.targets[1].weight, 3.0);
}

TEST(Export, spike_array_inputs_become_tick_indexed_event_trains) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("spike_array");
    fixture.write("net.nml", R"(<neuroml id="spikenet">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <spikeArray id="train0">
        <spike id="0" time="30ms"/>
        <spike id="1" time="10ms"/>
        <spike id="2" time="20ms"/>
    </spikeArray>
    <network id="net1">
        <population id="pop1" component="cell0" size="2"/>
        <explicitInput target="pop1[1]" input="train0"/>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    ASSERT_EQ(result.input_profiles.size(), 1u);
    const SimulationInputConfig &train = result.input_profiles[0];

    EXPECT_EQ(train.input_component_type_name, "spikeArray");
    EXPECT_FALSE(train.continuous_current_injection);

    // The train rides on the target it feeds, sorted into tick order regardless of
    // document order.
    ASSERT_EQ(train.targets.size(), 1u);
    EXPECT_EQ(train.targets[0].neuron_index, 1);
    EXPECT_EQ(train.targets[0].event_ticks, (Vector<s32>{1000, 2000, 3000}));
    EXPECT_EQ(train.max_delay_time, 3000);
}

TEST(Export, recording_selections_resolve_to_neurons_and_variables) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("recordings");
    fixture.write("net.nml", two_population_network_nml());
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    ASSERT_EQ(result.recording_profiles.size(), 2u);

    const RecordingConfig &columns = result.recording_profiles[0];
    ASSERT_EQ(columns.output_filenames.size(), 1u);
    EXPECT_EQ(columns.output_filenames[0], "out.dat");
    EXPECT_EQ(columns.file_output_format[0], OutputFileFormat::NML_STANDARD);
    EXPECT_EQ(columns.recordings_count, 2);
    ASSERT_EQ(columns.selections.size(), 2u);

    EXPECT_EQ(columns.selections[0].quantity_path, "pop1[0]/v");
    EXPECT_EQ(columns.selections[0].variable_name, "v");
    EXPECT_EQ(columns.selections[0].neuron_index, 0);

    EXPECT_EQ(columns.selections[1].quantity_path, "pop2[1]/v");
    EXPECT_EQ(columns.selections[1].neuron_index, 4);

    const RecordingConfig &events = result.recording_profiles[1];
    EXPECT_EQ(events.output_filenames[0], "spikes.dat");
    EXPECT_EQ(events.file_output_format[0], OutputFileFormat::SPIKE_EVENTS);
    ASSERT_EQ(events.selections.size(), 1u);
    EXPECT_EQ(events.selections[0].neuron_index, 4);
    EXPECT_EQ(events.selections[0].event_port, "spike");
}

TEST(Export, populations_sharing_a_component_type_keep_their_own_parameter_values) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("shared_type_distinct_values");
    // Both populations are iafCell, but they name different prototypes and differ in
    // leakReversal -- an excitatory/inhibitory pair built from one cell type. Keying
    // starting parameters by type would collapse these onto one row and silently
    // simulate two populations with identical parameters.
    fixture.write("net.nml", R"(<neuroml id="sharednet">
    <iafCell id="cellA" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <iafCell id="cellB" leakReversal="-90mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <network id="net1">
        <population id="pop1" component="cellA" size="2"/>
        <population id="pop2" component="cellB" size="2"/>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    // One ComponentType, because that is genuinely what both populations instantiate.
    ASSERT_EQ(result.cell_types.size(), 1u);
    ASSERT_EQ(result.populations.size(), 2u);
    EXPECT_EQ(result.populations[0].cell_type_index, result.populations[1].cell_type_index);

    // Two prototypes, because the two populations are parameterised differently.
    ASSERT_EQ(result.cell_prototypes.size(), 2u);
    EXPECT_EQ(result.populations[0].component_instance_id, "cellA");
    EXPECT_EQ(result.populations[1].component_instance_id, "cellB");
    EXPECT_NE(result.populations[0].prototype_index, result.populations[1].prototype_index);

    const Vector<String> &columns = result.cell_types[0].parameter_names;
    auto leak_reversal_of = [&](usize population_index) {
        const ComponentPrototype &prototype = result.cell_prototypes[static_cast<usize>(
                result.populations[population_index].prototype_index)];
        return parameter_value(columns, prototype.starting_parameters, "leakReversal");
    };

    EXPECT_DOUBLE_EQ(leak_reversal_of(0), -0.05);
    EXPECT_DOUBLE_EQ(leak_reversal_of(1), -0.09);

    // The neurons of each population inherit their population's prototype.
    EXPECT_EQ(result.neurons[0].prototype_index, result.populations[0].prototype_index);
    EXPECT_EQ(result.neurons[3].prototype_index, result.populations[1].prototype_index);
}

TEST(Export, projections_sharing_a_synapse_type_keep_their_own_parameter_values) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("shared_synapse_type");
    fixture.write("net.nml", R"(<neuroml id="synnet">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <expOneSynapse id="synFast" gbase="0.5nS" erev="0mV" tauDecay="1ms"/>
    <expOneSynapse id="synSlow" gbase="0.5nS" erev="0mV" tauDecay="50ms"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="3"/>
        <projection id="projFast" presynapticPopulation="pop1"
                    postsynapticPopulation="pop1" synapse="synFast">
            <connection id="0" preCellId="../pop1[0]" postCellId="../pop1[1]"/>
        </projection>
        <projection id="projSlow" presynapticPopulation="pop1"
                    postsynapticPopulation="pop1" synapse="synSlow">
            <connection id="0" preCellId="../pop1[0]" postCellId="../pop1[2]"/>
        </projection>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    ASSERT_EQ(result.synapse_types.size(), 1u);
    ASSERT_EQ(result.synapse_prototypes.size(), 2u);

    // Neuron 0 has one edge from each projection; they share a type but not a prototype.
    ASSERT_EQ(result.neurons[0].outgoing_edges.size(), 2u);
    const NetworkEdge &fast = result.neurons[0].outgoing_edges[0];
    const NetworkEdge &slow = result.neurons[0].outgoing_edges[1];
    EXPECT_EQ(fast.synapse_type_index, slow.synapse_type_index);
    EXPECT_NE(fast.synapse_prototype_index, slow.synapse_prototype_index);

    const Vector<String> &columns = result.synapse_types[0].parameter_names;
    auto tau_decay_of = [&](s64 prototype_index) {
        return parameter_value(
                columns,
                result.synapse_prototypes[static_cast<usize>(prototype_index)]
                        .starting_parameters,
                "tauDecay");
    };

    EXPECT_DOUBLE_EQ(tau_decay_of(fast.synapse_prototype_index), 0.001);
    EXPECT_DOUBLE_EQ(tau_decay_of(slow.synapse_prototype_index), 0.05);
}

TEST(Export, is_deterministic_across_parses) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("determinism");
    fixture.write("net.nml", two_population_network_nml());
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    // Two independent parsers. Anything derived from hash-map iteration order would
    // differ between them.
    NML_Parser first_parser;
    NML_ParseResult first = first_parser.parse_lems(fixture.path_of("LEMS.xml"));

    NML_Parser second_parser;
    NML_ParseResult second = second_parser.parse_lems(fixture.path_of("LEMS.xml"));

    // Type order fixes every type index, so it has to be stable.
    ASSERT_EQ(first.cell_types.size(), second.cell_types.size());
    for (usize index = 0; index < first.cell_types.size(); index += 1) {
        EXPECT_EQ(first.cell_types[index].name, second.cell_types[index].name);
        EXPECT_EQ(first.cell_types[index].state_offset, second.cell_types[index].state_offset);
        EXPECT_EQ(first.cell_types[index].parameter_names,
                  second.cell_types[index].parameter_names);
    }

    ASSERT_EQ(first.synapse_types.size(), second.synapse_types.size());
    for (usize index = 0; index < first.synapse_types.size(); index += 1) {
        EXPECT_EQ(first.synapse_types[index].name, second.synapse_types[index].name);
    }

    // Prototype order fixes every prototype index.
    ASSERT_EQ(first.cell_prototypes.size(), second.cell_prototypes.size());
    for (usize index = 0; index < first.cell_prototypes.size(); index += 1) {
        EXPECT_EQ(first.cell_prototypes[index].instance_id,
                  second.cell_prototypes[index].instance_id);
        EXPECT_EQ(first.cell_prototypes[index].type_index,
                  second.cell_prototypes[index].type_index);

        ASSERT_EQ(first.cell_prototypes[index].starting_parameters.size(),
                  second.cell_prototypes[index].starting_parameters.size());
        for (usize column = 0;
             column < first.cell_prototypes[index].starting_parameters.size(); column += 1) {
            EXPECT_DOUBLE_EQ(
                    first.cell_prototypes[index].starting_parameters[column].float64,
                    second.cell_prototypes[index].starting_parameters[column].float64);
        }
    }

    // Neuron numbering, and therefore the whole graph.
    ASSERT_EQ(first.neurons.size(), second.neurons.size());
    for (usize neuron = 0; neuron < first.neurons.size(); neuron += 1) {
        EXPECT_EQ(first.neurons[neuron].cell_type_index,
                  second.neurons[neuron].cell_type_index);
        EXPECT_EQ(first.neurons[neuron].prototype_index,
                  second.neurons[neuron].prototype_index);
        EXPECT_EQ(first.neurons[neuron].population_index,
                  second.neurons[neuron].population_index);
    }
    EXPECT_EQ(build_adjacency_list(first), build_adjacency_list(second));

    ASSERT_EQ(first.populations.size(), second.populations.size());
    for (usize index = 0; index < first.populations.size(); index += 1) {
        EXPECT_EQ(first.populations[index].population_name,
                  second.populations[index].population_name);
        EXPECT_EQ(first.populations[index].first_neuron_index,
                  second.populations[index].first_neuron_index);
        EXPECT_EQ(first.populations[index].prototype_index,
                  second.populations[index].prototype_index);
    }
}

TEST(Export, reparsing_with_the_same_parser_does_not_duplicate_the_network) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("reparse");
    fixture.write("net.nml", two_population_network_nml());
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    // instantiate() appends each child onto its parent's structured_instance_data, so a
    // second pass over the same nodes must rebuild the table rather than add to it.
    // Otherwise every population and connection appears twice.
    NML_Parser parser;
    NML_ParseResult first = parser.parse_lems(fixture.path_of("LEMS.xml"));
    NML_ParseResult second = parser.parse_lems(fixture.path_of("LEMS.xml"));

    EXPECT_EQ(second.neurons.size(), first.neurons.size());
    EXPECT_EQ(second.populations.size(), first.populations.size());
    EXPECT_EQ(build_adjacency_list(second), build_adjacency_list(first));
    EXPECT_EQ(second.input_profiles.size(), first.input_profiles.size());
    EXPECT_EQ(second.recording_profiles.size(), first.recording_profiles.size());
    EXPECT_EQ(second.cell_prototypes.size(), first.cell_prototypes.size());
}

TEST(Export, resolves_populationlist_paths) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("population_list");
    // populationList sizes itself from its <instance> children, and its connections
    // address cells as "../pop/0/cellType" rather than "../pop[0]".
    fixture.write("net.nml", R"(<neuroml id="listnet">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <expOneSynapse id="syn0" gbase="0.5nS" erev="0mV" tauDecay="3ms"/>
    <network id="net1">
        <populationList id="pop1" component="cell0">
            <instance id="0"><location x="0" y="0" z="0"/></instance>
            <instance id="1"><location x="10" y="0" z="0"/></instance>
            <instance id="2"><location x="20" y="0" z="0"/></instance>
        </populationList>
        <projection id="proj1" presynapticPopulation="pop1"
                    postsynapticPopulation="pop1" synapse="syn0">
            <connection id="0" preCellId="../pop1/0/cell0" postCellId="../pop1/2/cell0"/>
        </projection>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    // Size comes from the three <instance> children, not from a size attribute.
    EXPECT_EQ(result.neurons.size(), 3u);
    ASSERT_EQ(result.populations.size(), 1u);
    EXPECT_EQ(result.populations[0].neuron_count, 3);

    ASSERT_EQ(result.neurons[0].outgoing_edges.size(), 1u);
    EXPECT_EQ(result.neurons[0].outgoing_edges[0].target_neuron_index, 2);
}


// ── error handling ───────────────────────────────────────────────────────────────

TEST(Errors, a_connection_naming_an_unknown_population_throws) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("bad_endpoint");
    fixture.write("net.nml", R"(<neuroml id="badnet">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <expOneSynapse id="syn0" gbase="0.5nS" erev="0mV" tauDecay="3ms"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="2"/>
        <projection id="proj1" presynapticPopulation="pop1"
                    postsynapticPopulation="pop1" synapse="syn0">
            <connection id="0" preCellId="../pop1[0]" postCellId="../nosuchpop[0]"/>
        </projection>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    try {
        parser.parse_lems(fixture.path_of("LEMS.xml"));
        FAIL() << "expected an unresolvable connection endpoint to throw";
    } catch (const std::runtime_error &error) {
        // Silently dropping the edge would produce a quietly wrong network.
        EXPECT_NE(String(error.what()).find("nosuchpop"), String::npos);
    }
}

TEST(Errors, an_out_of_range_cell_index_throws) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("index_out_of_range");
    fixture.write("net.nml", R"(<neuroml id="badnet">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <expOneSynapse id="syn0" gbase="0.5nS" erev="0mV" tauDecay="3ms"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="2"/>
        <projection id="proj1" presynapticPopulation="pop1"
                    postsynapticPopulation="pop1" synapse="syn0">
            <connection id="0" preCellId="../pop1[0]" postCellId="../pop1[7]"/>
        </projection>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    EXPECT_THROW(parser.parse_lems(fixture.path_of("LEMS.xml")), std::runtime_error);
}

TEST(Errors, a_population_naming_an_unknown_component_throws) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("bad_component");
    fixture.write("net.nml", R"(<neuroml id="badnet">
    <network id="net1">
        <population id="pop1" component="no_such_cell" size="2"/>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    try {
        parser.parse_lems(fixture.path_of("LEMS.xml"));
        FAIL() << "expected an unresolvable population component to throw";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(String(error.what()).find("no_such_cell"), String::npos);
    }
}

TEST(Errors, unmodelled_metadata_tags_are_skipped_rather_than_fatal) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("metadata");
    // Real documents carry RDF/Dublin Core provenance that maps to no runtime entity.
    // None of it affects the simulation, so it must not fail the parse.
    fixture.write("net.nml", R"(<neuroml id="metanet">
    <someUnmodelledAnnotation foo="bar"/>
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <network id="net1">
        <someOtherAnnotation/>
        <population id="pop1" component="cell0" size="2"/>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result;
    ASSERT_NO_THROW(result = parser.parse_lems(fixture.path_of("LEMS.xml")));

    EXPECT_EQ(result.neurons.size(), 2u);
    EXPECT_EQ(result.cell_types.size(), 1u);
}


// ── NML_Node ─────────────────────────────────────────────────────────────────────
//
// The node type is the parser's own currency: the ComponentType catalogue holds every
// declaration node by value from ingest until resolution, long after the libxml document
// that produced it has been freed, so construction and copying have to be sound in their
// own right.

TEST(Node, construction_starts_empty) {
    NML_Node component("ComponentType");

    EXPECT_EQ(component.tag_name, "ComponentType");
    EXPECT_TRUE(component.attributes.empty());
    EXPECT_TRUE(component.body.empty());
}

TEST(Node, attributes_round_trip_by_name) {
    NML_Node component("ComponentType");
    component.add_attribute("name", "iafCell");

    EXPECT_TRUE(component.has_attribute("name"));
    EXPECT_EQ(component.get_attribute("name"), "iafCell");

    // Absent is not an error, and it is not a stored empty string either.
    EXPECT_FALSE(component.has_attribute("extends"));
    EXPECT_EQ(component.get_attribute("extends"), "");
}

TEST(Node, nest_appends_a_child_in_order) {
    NML_Node parent("ComponentType");

    NML_Node first_child("Parameter");
    first_child.add_attribute("name", "alpha");
    NML_Node second_child("Parameter");
    second_child.add_attribute("name", "beta");

    parent.nest(first_child);
    parent.nest(second_child);

    ASSERT_EQ(parent.body.size(), 2u);
    EXPECT_EQ(parent.body[0].get_attribute("name"), "alpha");
    EXPECT_EQ(parent.body[1].get_attribute("name"), "beta");

    // find_child answers with the first child of a tag, and with nullptr for a tag the
    // node does not carry.
    const NML_Node *found = parent.find_child("Parameter");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->get_attribute("name"), "alpha");
    EXPECT_EQ(parent.find_child("Dynamics"), nullptr);
}

TEST(Node, copying_deep_copies_the_body_and_attributes) {
    NML_Node parent("ComponentType");
    parent.add_attribute("name", "testCell");

    NML_Node child("Parameter");
    child.add_attribute("name", "alpha");
    parent.nest(child);

    NML_Node copy(parent);

    // A shallow copy would let this reach into the copy and empty it -- which is exactly
    // what happens to a catalogued type when the document it came from goes out of scope.
    parent.add_attribute("name", "mutated");
    parent.body.clear();

    EXPECT_EQ(copy.tag_name, "ComponentType");
    EXPECT_EQ(copy.get_attribute("name"), "testCell");
    ASSERT_EQ(copy.body.size(), 1u);
    EXPECT_EQ(copy.body[0].tag_name, "Parameter");
    EXPECT_EQ(copy.body[0].get_attribute("name"), "alpha");
}

TEST(Node, xml_conversion_reads_tag_attributes_and_nested_children) {
    const char *document_text =
        "<Lems>"
        "  <ComponentType name=\"testCell\" extends=\"baseCell\">"
        "    <Parameter name=\"alpha\" dimension=\"none\"/>"
        "    <Parameter name=\"beta\" dimension=\"voltage\"/>"
        "  </ComponentType>"
        "</Lems>";

    xmlDocPtr document = xmlReadMemory(document_text,
                                       static_cast<int>(strlen(document_text)),
                                       "test.xml", nullptr, XML_PARSE_NOBLANKS);
    ASSERT_NE(document, nullptr);

    xmlNodePtr root = xmlDocGetRootElement(document);
    ASSERT_NE(root, nullptr);
    xmlNodePtr component_type_node = root->children;
    ASSERT_NE(component_type_node, nullptr);

    NML_Parser parser;
    NML_Node component = parser.xml_node_to_nml_node(component_type_node);

    EXPECT_EQ(component.tag_name, "ComponentType");
    EXPECT_EQ(component.get_attribute("name"), "testCell");
    EXPECT_EQ(component.get_attribute("extends"), "baseCell");

    // Children keep source order; the parameter order they carry is the column order of
    // every exported starting-parameter row.
    ASSERT_EQ(component.body.size(), 2u);
    EXPECT_EQ(component.body[0].get_attribute("name"), "alpha");
    EXPECT_EQ(component.body[0].get_attribute("dimension"), "none");
    EXPECT_EQ(component.body[1].get_attribute("name"), "beta");
    EXPECT_EQ(component.body[1].get_attribute("dimension"), "voltage");

    xmlFreeDoc(document);
}


// ── schema validation ────────────────────────────────────────────────────────────

TEST(Schema, path_is_baked_in_and_exists) {
    NML_Parser parser;
    EXPECT_FALSE(parser.NML_SCHEMA_PATH.empty());
    EXPECT_TRUE(filesystem::exists(parser.NML_SCHEMA_PATH));
}

TEST(Schema, accepts_a_conformant_neuroml_document) {
    FixtureDirectory fixture("schema_valid");
    String path = fixture.write("valid.nml", R"(<neuroml
    xmlns="http://www.neuroml.org/schema/neuroml2" id="TestDoc">
    <izhikevichCell id="izTest" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65" d="6"/>
</neuroml>
)");

    NML_Parser parser;
    EXPECT_TRUE(parser.validate_against_schema(path));
}

TEST(Schema, rejects_a_document_with_an_unknown_element) {
    FixtureDirectory fixture("schema_unknown_element");
    String path = fixture.write("invalid.nml", R"(<neuroml
    xmlns="http://www.neuroml.org/schema/neuroml2" id="TestDoc">
    <thisTagDoesNotExistInSchema id="oops"/>
</neuroml>
)");

    NML_Parser parser;
    EXPECT_FALSE(parser.validate_against_schema(path));
}

TEST(Schema, records_the_offending_element_and_line) {
    FixtureDirectory fixture("schema_error_detail");
    String path = fixture.write("invalid.nml", R"(<neuroml
    xmlns="http://www.neuroml.org/schema/neuroml2" id="TestDoc">
    <thisTagDoesNotExistInSchema id="oops"/>
</neuroml>
)");

    // libxml2's own handler only writes to stderr, where neither a caller nor this
    // codebase's logger can see it. "Does not validate" with no location is unactionable.
    NML_Parser parser;
    ASSERT_FALSE(parser.validate_against_schema(path));

    EXPECT_NE(parser.last_schema_validation_errors.find("thisTagDoesNotExistInSchema"),
              String::npos);
    EXPECT_NE(parser.last_schema_validation_errors.find("line"), String::npos);
}

TEST(Schema, rejects_a_wellformed_document_of_another_schema) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    // Cells.xml is well-formed XML and a valid LEMS document, but its root is <Lems>, not
    // <neuroml>. Passing it must fail rather than validate vacuously.
    NML_Parser parser;
    EXPECT_FALSE(parser.validate_against_schema(parser.STANDARD_LIBRARY_PATH + "/Cells.xml"));
}

TEST(Schema, rejects_a_missing_file) {
    NML_Parser parser;
    EXPECT_FALSE(parser.validate_against_schema("/definitely/not/a/real/file_xyz.nml"));
}


// ── ingest ───────────────────────────────────────────────────────────────────────

TEST(Ingest, a_component_type_is_catalogued_and_not_left_as_an_instance_node) {
    FixtureDirectory fixture("ingest_catalogue");
    fixture.write("doc.nml", R"(<neuroml id="TestDoc">
    <izhikevichCell id="izTest" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65" d="6"/>
    <ComponentType name="inlineTestCell" extends="baseCell">
        <Parameter name="alpha" dimension="none"/>
    </ComponentType>
</neuroml>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("doc.nml"));

    EXPECT_EQ(parser.component_type_catalogue.count("inlineTestCell"), 1u);

    // A ComponentType is a declaration, not a thing to instantiate. Leaving it among the
    // instance nodes would have the export try to build a neuron out of it.
    ASSERT_EQ(parser.document_instance_nodes.size(), 1u);
    EXPECT_EQ(parser.document_instance_nodes[0].tag_name, "izhikevichCell");
    EXPECT_EQ(parser.document_instance_nodes[0].get_attribute("id"), "izTest");
}

TEST(Ingest, an_included_files_instances_join_the_including_documents) {
    FixtureDirectory fixture("ingest_include_merge");
    fixture.write("included.nml", R"(<neuroml id="IncludedDoc">
    <ComponentType name="includedTestCell" extends="baseCell">
        <Parameter name="alpha" dimension="none"/>
    </ComponentType>
    <includedTestCell id="fromIncludedFile" alpha="1"/>
</neuroml>
)");
    fixture.write("top.nml", R"(<neuroml id="TopDoc">
    <include href="included.nml"/>
    <izhikevichCell id="fromTopFile" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65" d="6"/>
</neuroml>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("top.nml"));

    // One shared catalogue and one shared instance list across the whole include graph;
    // the <include> tag itself is spliced away rather than surviving as a node.
    EXPECT_EQ(parser.component_type_catalogue.count("includedTestCell"), 1u);

    Vector<String> instance_ids;
    for (const NML_Node &node : parser.document_instance_nodes) {
        EXPECT_NE(node.tag_name, "include");
        instance_ids.push_back(node.get_attribute("id"));
    }
    EXPECT_EQ(instance_ids, (Vector<String>{"fromIncludedFile", "fromTopFile"}));
}

TEST(Ingest, a_standard_library_lems_file_can_be_included_by_a_neuroml_document) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    NML_Parser path_helper;
    FixtureDirectory fixture("ingest_stdlib_include");
    fixture.write("top.nml",
                  "<neuroml id=\"TopDoc\">\n"
                  "    <include href=\"" + path_helper.STANDARD_LIBRARY_PATH +
                  "/Cells.xml\"/>\n"
                  "</neuroml>\n");

    // Cells.xml is a LEMS document that the NeuroML2 XSD rejects outright (see
    // Schema.rejects_a_wellformed_document_of_another_schema). Following it must still
    // work: ingest is not schema-gated, and mixed LEMS/NeuroML include graphs are normal.
    NML_Parser parser;
    ASSERT_NO_THROW(parser.ingest_document(fixture.path_of("top.nml")));

    EXPECT_EQ(parser.component_type_catalogue.count("iafCell"), 1u);
}

TEST(Ingest, a_missing_path_is_not_fatal_and_leaves_the_parser_empty) {
    NML_Parser parser;
    EXPECT_NO_THROW(parser.ingest_document("/definitely/does/not/exist_xyz.nml"));

    EXPECT_TRUE(parser.component_type_catalogue.empty());
    EXPECT_TRUE(parser.document_instance_nodes.empty());
}

TEST(Ingest, a_missing_include_does_not_abandon_the_rest_of_the_file) {
    FixtureDirectory fixture("ingest_missing_include");
    fixture.write("root.xml", R"(<Lems>
    <Include file="no_such_file.xml"/>
    <ComponentType name="declared_after_the_bad_include"/>
</Lems>
)");

    NML_Parser parser;
    EXPECT_NO_THROW(parser.ingest_document(fixture.path_of("root.xml")));

    // The include is reported and skipped; the declarations after it still land.
    EXPECT_EQ(parser.component_type_catalogue.count("declared_after_the_bad_include"), 1u);
}


// ── inheritance across a real extends chain ──────────────────────────────────────

TEST(Resolution, a_four_hop_glif_chain_merges_every_ancestors_declarations) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("glif_chain");
    // Four hops with real content at every one, through both user-defined and vendored
    // std-lib ancestors: GlifChainCell -> GlifChainBase (user) -> baseCellMembPot ->
    // baseSpikingCell -> baseCell (std lib). A merge that only reaches the direct parent
    // passes a one-hop test and loses everything above it.
    fixture.write("types.xml", R"(<Lems>
    <ComponentType name="GlifChainBase" extends="baseCellMembPot">
        <Parameter name="C" dimension="capacitance"/>
        <Parameter name="asc_amp1" dimension="current"/>
        <Exposure name="baseExposure" dimension="none"/>
        <Dynamics>
            <StateVariable name="v" dimension="voltage" exposure="v"/>
        </Dynamics>
    </ComponentType>
    <ComponentType name="GlifChainCell" extends="GlifChainBase">
        <Parameter name="gL" dimension="conductance"/>
        <Parameter name="EL" dimension="voltage"/>
        <Parameter name="vth" dimension="voltage"/>
        <Parameter name="vreset" dimension="voltage"/>
        <Fixed parameter="asc_amp1" value="10pA"/>
        <Dynamics>
            <StateVariable name="ASC1" dimension="current"/>
            <TimeDerivative variable="v" value="(gL * (EL - v) + iSyn) / C"/>
            <OnCondition test="v .gt. vth">
                <EventOut port="spike"/>
                <StateAssignment variable="v" value="vreset"/>
            </OnCondition>
        </Dynamics>
    </ComponentType>
</Lems>
)");

    NML_Parser parser;
    parser.parse_lems("");
    parser.ingest_document(fixture.path_of("types.xml"));
    parser.resolve_all_component_types();

    const ComponentType &cell = parser.declared_component_types.at("GlifChainCell");

    // Ancestors first, then its own: C/asc_amp1 come from GlifChainBase, the other four
    // from GlifChainCell.
    EXPECT_EQ(cell.ordered_parameter_names(),
              (Vector<String>{"C", "asc_amp1", "gL", "EL", "vth", "vreset"}));

    // `v` is GlifChainBase's, `ASC1` is its own; the redeclaration of `v` does not
    // duplicate it.
    EXPECT_EQ(cell.ordered_state_variable_names(), (Vector<String>{"v", "ASC1"}));

    EXPECT_EQ(declaration_names(cell, NML_DeclarationType::TimeDerivative).size(), 1u);
    EXPECT_EQ(cell.declarations.find_all(NML_DeclarationType::OnCondition).size(), 1u);

    // baseExposure is GlifChainBase's own; `v` is baseCellMembPot's, two hops up.
    Vector<String> exposure_names = declaration_names(cell, NML_DeclarationType::Exposure);
    EXPECT_NE(std::find(exposure_names.begin(), exposure_names.end(), "baseExposure"),
              exposure_names.end());
    EXPECT_NE(std::find(exposure_names.begin(), exposure_names.end(), "v"),
              exposure_names.end());

    // The `spike` EventPort is declared on baseSpikingCell, three hops up, and GlifChainCell
    // emits through it without ever redeclaring it.
    const NML_Declaration *spike_port =
            cell.declarations.find(NML_DeclarationType::EventPort, "spike");
    ASSERT_NE(spike_port, nullptr);
    EXPECT_EQ(spike_port->value_or("direction"), "out");

    // The Fixed pin reaches a Parameter this type never declared, two hops up the chain.
    const NML_Declaration *pinned =
            cell.declarations.find(NML_DeclarationType::Parameter, "asc_amp1");
    ASSERT_NE(pinned, nullptr);
    EXPECT_EQ(pinned->value_or("value"), "10pA");
}

TEST(Resolution, a_redeclared_state_variable_keeps_the_most_derived_attributes) {
    FixtureDirectory fixture("state_variable_override");
    // The child restates `v` only to add `exposure="v"`. Carrying the ancestor's copy
    // instead would silently drop the exposure the rest of the model reads through.
    fixture.write("types.xml", R"(<Lems>
    <ComponentType name="parent_type">
        <Dynamics>
            <StateVariable name="v" dimension="voltage"/>
        </Dynamics>
    </ComponentType>
    <ComponentType name="child_type" extends="parent_type">
        <Dynamics>
            <StateVariable name="v" dimension="voltage" exposure="v"/>
            <TimeDerivative variable="v" value="0"/>
        </Dynamics>
    </ComponentType>
</Lems>
)");

    NML_Parser parser;
    parser.ingest_document(fixture.path_of("types.xml"));
    parser.resolve_all_component_types();

    const ComponentType &child = parser.declared_component_types.at("child_type");

    EXPECT_EQ(child.ordered_state_variable_names(), (Vector<String>{"v"}));
    const NML_Declaration *state_variable =
            child.declarations.find(NML_DeclarationType::StateVariable, "v");
    ASSERT_NE(state_variable, nullptr);
    EXPECT_EQ(state_variable->value_or("exposure"), "v");
}

TEST(Resolution, an_ancestors_on_start_survives_a_subtype_that_declares_none) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("inherited_on_start");
    // iafTauCell declares <OnStart><StateAssignment variable="v" value="leakReversal"/>.
    // A subtype that adds a parameter and nothing else must still start at leakReversal;
    // losing the ancestor's OnStart starts every such cell from an uninitialised voltage.
    fixture.write("types.xml", R"(<Lems>
    <ComponentType name="child_of_iaf_tau_cell" extends="iafTauCell">
        <Parameter name="extra_parameter" dimension="none"/>
    </ComponentType>
</Lems>
)");

    NML_Parser parser;
    parser.parse_lems("");
    parser.ingest_document(fixture.path_of("types.xml"));
    parser.resolve_all_component_types();

    const ComponentType &child =
            parser.declared_component_types.at("child_of_iaf_tau_cell");

    Vector<const NML_Declaration *> on_starts =
            child.declarations.find_all(NML_DeclarationType::OnStart);
    ASSERT_EQ(on_starts.size(), 1u);
    ASSERT_EQ(on_starts[0]->children.size(), 1u);
    EXPECT_EQ(on_starts[0]->children[0].tag_type, NML_DeclarationType::StateAssignment);
    EXPECT_EQ(on_starts[0]->children[0].value_or("variable"), "v");
    EXPECT_EQ(on_starts[0]->children[0].value_or("value"), "leakReversal");
}

TEST(Resolution, a_subtypes_on_start_is_applied_after_its_ancestors) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("on_start_ordering");
    // An OnStart accumulates rather than overriding, so a subtype declaring one leaves the
    // model with two initialisers for the same variable. Whichever runs last decides the
    // starting voltage, so the subtype's has to be the later of the two -- the reverse
    // order silently pins every subtype to its ancestor's initial value.
    fixture.write("types.xml", R"(<Lems>
    <ComponentType name="overriding_cell" extends="iafTauCell">
        <Dynamics>
            <OnStart>
                <StateAssignment variable="v" value="reset"/>
            </OnStart>
        </Dynamics>
    </ComponentType>
</Lems>
)");
    fixture.write("net.nml", R"(<neuroml id="onstartnet">
    <overriding_cell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV" tau="10ms"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="1"/>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", R"(<Lems>
    <Target component="sim1"/>
    <Include file="types.xml"/>
    <Include file="net.nml"/>
    <Simulation id="sim1" length="100ms" step="0.01ms" target="net1"/>
</Lems>
)");

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    const CellTypeSpecification *cell = cell_type_named(result, "overriding_cell");
    ASSERT_NE(cell, nullptr);

    Vector<String> initial_assignments;
    for (const DynamicsInstruction &instruction : cell->dynamics) {
        if (instruction.stage != DynamicsStage::Initialize) continue;
        if (instruction.source_tag != NML_DeclarationType::StateAssignment) continue;
        if (instruction.target != "v") continue;
        initial_assignments.push_back(instruction.expression);
    }

    EXPECT_EQ(initial_assignments, (Vector<String>{"leakReversal", "reset"}));
}

TEST(Resolution, structural_declarations_are_inherited_across_the_extends_chain) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    NML_Parser parser;
    parser.parse_lems("");

    auto type_named = [&](const String &type_name) -> const ComponentType & {
        auto entry = parser.declared_component_types.find(type_name);
        EXPECT_NE(entry, parser.declared_component_types.end()) << type_name;
        return entry->second;
    };

    // `population` declares only `size` itself; the `component` ComponentReference that
    // names what it instantiates is basePopulation's, one hop up.
    const ComponentType &population = type_named("population");
    EXPECT_NE(population.declarations.find(NML_DeclarationType::Parameter, "size"), nullptr);
    EXPECT_NE(population.declarations.find(NML_DeclarationType::ComponentReference,
                                           "component"), nullptr);

    // A projection routes through a named synapse and addresses its endpoint populations
    // through Path declarations.
    const ComponentType &projection = type_named("projection");
    EXPECT_NE(projection.declarations.find(NML_DeclarationType::ComponentReference,
                                           "synapse"), nullptr);
    EXPECT_NE(projection.declarations.find(NML_DeclarationType::Path,
                                           "presynapticPopulation"), nullptr);

    // A connection addresses cells rather than populations, through its own Paths.
    const ComponentType &connection = type_named("connection");
    EXPECT_NE(connection.declarations.find(NML_DeclarationType::Path, "preCellId"), nullptr);
    EXPECT_NE(connection.declarations.find(NML_DeclarationType::Path, "postCellId"), nullptr);

    // An input scheme may itself drive a synapse instance, named the same way.
    const ComponentType &poisson_firing_synapse = type_named("poissonFiringSynapse");
    EXPECT_NE(poisson_firing_synapse.declarations.find(
                      NML_DeclarationType::ComponentReference, "synapse"), nullptr);

    // expOneSynapse's Requirement for the postsynaptic voltage is declared two hops up, on
    // baseVoltageDepSynapse, and is what makes it conductance-based.
    const ComponentType &exp_one_synapse = type_named("expOneSynapse");
    EXPECT_NE(exp_one_synapse.declarations.find(NML_DeclarationType::Requirement, "v"),
              nullptr);
    EXPECT_NE(exp_one_synapse.declarations.find(NML_DeclarationType::Parameter, "erev"),
              nullptr);
}


// ── runtime categories, continued ────────────────────────────────────────────────

TEST(Category, a_synapse_classifies_by_its_nearest_categorised_ancestor) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    NML_Parser parser;
    parser.parse_lems("");

    // baseSynapse extends basePointCurrent, and basePointCurrent is the root of the
    // InputScheme category. The walk has to stop at the NEAREST categorised ancestor, so
    // every synapse in the library classifies as a Synapse rather than as a stimulus --
    // which would send every projection's synapse down the input-wiring path instead.
    auto category_of = [&](const String &type_name) {
        auto entry = parser.declared_component_types.find(type_name);
        EXPECT_NE(entry, parser.declared_component_types.end()) << type_name;
        if (entry == parser.declared_component_types.end()) return RuntimeCategory::General;
        return entry->second.runtime_category;
    };

    EXPECT_EQ(category_of("basePointCurrent"), RuntimeCategory::InputScheme);
    EXPECT_EQ(category_of("baseSynapse"), RuntimeCategory::Synapse);
    EXPECT_EQ(category_of("baseVoltageDepSynapse"), RuntimeCategory::Synapse);
    EXPECT_EQ(category_of("baseConductanceBasedSynapse"), RuntimeCategory::Synapse);
    EXPECT_EQ(category_of("alphaCurrentSynapse"), RuntimeCategory::Synapse);
    EXPECT_EQ(category_of("blockingPlasticSynapse"), RuntimeCategory::Synapse);

    // pulseGenerator extends basePointCurrent directly, so it keeps that category.
    EXPECT_EQ(category_of("pulseGenerator"), RuntimeCategory::InputScheme);
}

TEST(Category, types_chaining_through_another_files_intermediate_base_still_classify) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    NML_Parser parser;
    parser.parse_lems("");

    // PyNN.xml's types reach their category anchor only through intermediate bases in
    // Cells.xml and Synapses.xml: IF_curr_alpha -> basePyNNIaFCell -> basePyNNCell ->
    // baseCellMembPot -> baseSpikingCell -> baseCell. Classifying a file's types as that
    // file is read would make this depend on directory iteration order.
    auto category_of = [&](const String &type_name) {
        auto entry = parser.declared_component_types.find(type_name);
        EXPECT_NE(entry, parser.declared_component_types.end()) << type_name;
        if (entry == parser.declared_component_types.end()) return RuntimeCategory::General;
        return entry->second.runtime_category;
    };

    EXPECT_EQ(category_of("IF_curr_alpha"), RuntimeCategory::Cell);
    EXPECT_EQ(category_of("expCondSynapse"), RuntimeCategory::Synapse);

    // And the merge crossed those files too, not just the classification.
    const ComponentType &if_curr_alpha =
            parser.declared_component_types.at("IF_curr_alpha");
    EXPECT_FALSE(if_curr_alpha.ordered_state_variable_names().empty());
    EXPECT_FALSE(if_curr_alpha.ordered_parameter_names().empty());
}

TEST(Category, a_type_matching_no_category_root_stays_general) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    NML_Parser parser;
    parser.parse_lems("");

    // morphology declares no `extends` and is none of the five runtime categories -- it is
    // Phase-3 multicompartment structure. It must survive resolution intact rather than be
    // dropped or forced into a category it does not belong to.
    const ComponentType &morphology = parser.declared_component_types.at("morphology");
    EXPECT_EQ(morphology.runtime_category, RuntimeCategory::General);
    EXPECT_FALSE(morphology.declarations.for_all().empty());
}

TEST(Category, a_subtype_declared_before_its_base_still_classifies) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("forward_reference_category");
    // The derived type is declared textually before the intermediate base that carries the
    // link to a category anchor, so the category can only be found once the whole file has
    // been ingested.
    fixture.write("types.xml", R"(<Lems>
    <ComponentType name="forward_derived_cell" extends="forward_intermediate_base">
        <Parameter name="alpha" dimension="none"/>
    </ComponentType>
    <ComponentType name="forward_intermediate_base" extends="baseCell">
        <Parameter name="beta" dimension="none"/>
    </ComponentType>
</Lems>
)");

    NML_Parser parser;
    parser.parse_lems("");
    parser.ingest_document(fixture.path_of("types.xml"));
    parser.resolve_all_component_types();

    EXPECT_EQ(parser.declared_component_types.at("forward_derived_cell").runtime_category,
              RuntimeCategory::Cell);
    EXPECT_EQ(parser.declared_component_types.at("forward_intermediate_base")
                      .runtime_category,
              RuntimeCategory::Cell);
}

TEST(Category, the_standard_library_carries_a_usable_iaf_cell) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    NML_Parser parser;
    parser.parse_lems("");

    // iafCell's own `extends` is baseIafCapCell, several hops short of baseCell, and its
    // dynamics are spread over that chain. Everything a kernel needs has to be present on
    // the flattened type.
    const ComponentType &iaf_cell = parser.declared_component_types.at("iafCell");
    EXPECT_EQ(iaf_cell.runtime_category, RuntimeCategory::Cell);
    EXPECT_FALSE(iaf_cell.ordered_state_variable_names().empty());
    EXPECT_FALSE(iaf_cell.ordered_parameter_names().empty());
    EXPECT_FALSE(iaf_cell.declarations.find_all(NML_DeclarationType::TimeDerivative).empty());
    EXPECT_FALSE(iaf_cell.declarations.find_all(NML_DeclarationType::OnCondition).empty());
}

TEST(Category, resolving_a_type_no_document_declares_throws_naming_it) {
    NML_Parser parser;

    // The same lookup the resolve pass uses for `extends`. A miss must fail loudly rather
    // than hand back a default-constructed type that silently declares nothing.
    Vector<String> resolution_stack;
    try {
        parser.resolve_component_type("definitelyNotARealType_xyz", resolution_stack);
        FAIL() << "expected resolve_component_type to throw on an undeclared type";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(String(error.what()).find("definitelyNotARealType_xyz"), String::npos);
    }
}


// ── quantities, continued ────────────────────────────────────────────────────────

TEST(Quantity, an_unknown_unit_suffix_throws_rather_than_scaling_by_one) {
    NML_Parser parser;

    // Scaling an unrecognised suffix by 1.0 turns a misspelling into a plausible number:
    // "-70zorkmids" would become -70 V instead of -70 mV, and nothing downstream could
    // tell. The message has to name the suffix it could not place.
    try {
        parser.resolve_quantity("-70zorkmids");
        FAIL() << "expected an unknown unit suffix to throw";
    } catch (const std::invalid_argument &error) {
        EXPECT_NE(String(error.what()).find("zorkmids"), String::npos);
    }

    EXPECT_THROW(parse_quantity("-70zorkmids"), std::invalid_argument);
}

TEST(Quantity, a_fixed_pin_carrying_an_unknown_unit_throws_when_it_is_exported) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("bad_fixed_unit");
    fixture.write("net.nml", R"(<neuroml id="badfixednet">
    <ComponentType name="bad_fixed_base" extends="baseCell">
        <Parameter name="EL" dimension="voltage"/>
        <Dynamics>
            <StateVariable name="v" dimension="voltage" exposure="v"/>
        </Dynamics>
    </ComponentType>
    <ComponentType name="bad_fixed_cell" extends="bad_fixed_base">
        <Fixed parameter="EL" value="-70zorkmids"/>
    </ComponentType>
    <bad_fixed_cell id="cell0"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="1"/>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    // The pin supplies the value the instance omits, so an unconvertible pin is a silently
    // wrong starting parameter unless it is reported.
    NML_Parser parser;
    EXPECT_THROW(parser.parse_lems(fixture.path_of("LEMS.xml")), std::invalid_argument);
}


// ── export, continued ────────────────────────────────────────────────────────────

TEST(Export, a_fixed_pin_supplies_the_value_an_instance_omits) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("fixed_pin_value");
    fixture.write("net.nml", R"(<neuroml id="pinnednet">
    <ComponentType name="pinned_base" extends="baseCell">
        <Parameter name="asc_amp1" dimension="current"/>
        <Parameter name="EL" dimension="voltage"/>
        <Dynamics>
            <StateVariable name="v" dimension="voltage" exposure="v"/>
        </Dynamics>
    </ComponentType>
    <ComponentType name="pinned_cell" extends="pinned_base">
        <Fixed parameter="asc_amp1" value="10pA"/>
    </ComponentType>
    <pinned_cell id="cell0" EL="-70mV"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="1"/>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    const CellTypeSpecification *cell = cell_type_named(result, "pinned_cell");
    ASSERT_NE(cell, nullptr);
    const ComponentPrototype *prototype = prototype_named(result.cell_prototypes, "cell0");
    ASSERT_NE(prototype, nullptr);

    // The instance names EL and not asc_amp1, so asc_amp1 can only have come from the pin
    // two hops up the chain -- and it has to arrive in SI, not as the written magnitude.
    EXPECT_DOUBLE_EQ(
            parameter_value(cell->parameter_names, prototype->starting_parameters, "EL"),
            -0.07);
    EXPECT_DOUBLE_EQ(
            parameter_value(cell->parameter_names, prototype->starting_parameters,
                            "asc_amp1"),
            1e-11);
}

TEST(Export, a_property_default_value_fills_a_column_the_instance_omits) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("property_default");
    // expOneSynapse declares <Property name="weight" dimension="none" defaultValue="1"/>
    // and syn0 does not name it. A Property is a starting-parameter column like any other,
    // so leaving it at zero would silence every synapse that omits it.
    fixture.write("net.nml", R"(<neuroml id="propertynet">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <expOneSynapse id="syn0" gbase="0.5nS" erev="0mV" tauDecay="3ms"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="2"/>
        <projection id="proj1" presynapticPopulation="pop1"
                    postsynapticPopulation="pop1" synapse="syn0">
            <connection id="0" preCellId="../pop1[0]" postCellId="../pop1[1]"/>
        </projection>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    const SynapseTypeSpecification *synapse = synapse_type_named(result, "expOneSynapse");
    ASSERT_NE(synapse, nullptr);
    const ComponentPrototype *prototype =
            prototype_named(result.synapse_prototypes, "syn0");
    ASSERT_NE(prototype, nullptr);

    // Parameters come first and the Property after, so a consumer reading by column name
    // rather than by position is the only safe way to read the row -- which is what
    // parameter_value does.
    EXPECT_NE(std::find(synapse->parameter_names.begin(), synapse->parameter_names.end(),
                        "weight"),
              synapse->parameter_names.end());
    EXPECT_DOUBLE_EQ(parameter_value(synapse->parameter_names,
                                     prototype->starting_parameters, "weight"),
                     1.0);
    EXPECT_DOUBLE_EQ(parameter_value(synapse->parameter_names,
                                     prototype->starting_parameters, "gbase"),
                     0.5e-9);
}

TEST(Export, a_synapse_with_a_block_or_plasticity_child_needs_per_edge_state) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("per_edge_synapse");
    // blockingPlasticSynapse declares <Children type="basePlasticityMechanism"/> and
    // <Children type="baseBlockMechanism"/>: its arrival response depends on per-edge
    // state, so converging edges cannot be summed into one accumulator the way
    // alphaCurrentSynapse's can. Getting this backwards silently merges edges that must
    // stay distinct.
    fixture.write("net.nml", R"(<neuroml id="peredgenet">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <alphaCurrentSynapse id="synCurrent" tau="2ms" ibase="0.1nA"/>
    <blockingPlasticSynapse id="synNmda" gbase="1nS" erev="0mV" tauRise="1ms" tauDecay="50ms"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="3"/>
        <projection id="projCurrent" presynapticPopulation="pop1"
                    postsynapticPopulation="pop1" synapse="synCurrent">
            <connection id="0" preCellId="../pop1[0]" postCellId="../pop1[1]"/>
        </projection>
        <projection id="projNmda" presynapticPopulation="pop1"
                    postsynapticPopulation="pop1" synapse="synNmda">
            <connection id="0" preCellId="../pop1[0]" postCellId="../pop1[2]"/>
        </projection>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    const SynapseTypeSpecification *current_based =
            synapse_type_named(result, "alphaCurrentSynapse");
    ASSERT_NE(current_based, nullptr);
    EXPECT_FALSE(current_based->requires_per_edge_state);

    const SynapseTypeSpecification *blocking =
            synapse_type_named(result, "blockingPlasticSynapse");
    ASSERT_NE(blocking, nullptr);
    EXPECT_TRUE(blocking->requires_per_edge_state);
}

TEST(Export, a_network_scope_synaptic_connection_becomes_an_edge) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("network_scope_connections");
    // NeuroML lets a network wire itself with <synapticConnection>/<synapticConnectionWD>
    // hanging straight off <network>, with no <projection> at all -- the network
    // ComponentType declares them as its own `synapticConnections` children. Reaching
    // connections only through projections dropped every one of these silently: the model
    // parsed and simulated fully disconnected.
    fixture.write("net.nml", R"(<neuroml id="directnet">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <alphaCurrentSynapse id="syn0" tau="2ms" ibase="0.1nA"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="3"/>
        <synapticConnection from="pop1[0]" to="pop1[1]" synapse="syn0"/>
        <synapticConnectionWD from="pop1[0]" to="pop1[2]" synapse="syn0"
                              weight="2.5" delay="2ms"/>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    ASSERT_EQ(result.neurons.size(), 3u);
    ASSERT_EQ(result.neurons[0].outgoing_edges.size(), 2u);

    const NetworkEdge &plain = result.neurons[0].outgoing_edges[0];
    EXPECT_EQ(plain.target_neuron_index, 1);
    EXPECT_DOUBLE_EQ(plain.weight, 1.0);
    EXPECT_EQ(plain.delay_tick_count, 0);

    // The WD form carries its own weight and delay, and 2ms at dt = 0.01ms is 200 ticks.
    const NetworkEdge &weighted = result.neurons[0].outgoing_edges[1];
    EXPECT_EQ(weighted.target_neuron_index, 2);
    EXPECT_DOUBLE_EQ(weighted.weight, 2.5);
    EXPECT_EQ(weighted.delay_tick_count, 200);

    // Each connection names its own synapse, since it has no projection to inherit one
    // from; both resolve to the one alphaCurrentSynapse prototype the document declares.
    ASSERT_EQ(result.synapse_prototypes.size(), 1u);
    EXPECT_EQ(plain.synapse_prototype_index, weighted.synapse_prototype_index);
    ASSERT_GE(plain.synapse_type_index, 0);
    EXPECT_EQ(result.synapse_types[static_cast<usize>(plain.synapse_type_index)].name,
              "alphaCurrentSynapse");

    EXPECT_EQ(build_adjacency_list(result)[0], (Vector<s64>{1, 2}));
}

TEST(Export, a_recording_path_naming_the_component_instance_resolves_its_variable) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("recording_component_path");
    // The populationList spelling names the population's own component INSTANCE between
    // the index and the variable: "pop1/0/cell0/v". Failing to recognise that segment
    // leaves it in the variable name ("cell0/v"), which no cell type declares -- and the
    // engine then falls back to recording the neuron's first state variable, silently
    // logging the wrong quantity for any cell with more than one.
    fixture.write("net.nml", R"(<neuroml id="pathnet">
    <izhikevich2007Cell id="cell0" C="100pF" v0="-60mV" k="0.7nS_per_mV"
                        vr="-60mV" vt="-40mV" vpeak="35mV" a="0.03per_ms"
                        b="-2nS" c="-50mV" d="100pA"/>
    <network id="net1">
        <populationList id="pop1" component="cell0">
            <instance id="0"><location x="0" y="0" z="0"/></instance>
            <instance id="1"><location x="10" y="0" z="0"/></instance>
        </populationList>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", R"(<Lems>
    <Target component="sim1"/>
    <Include file="net.nml"/>
    <Simulation id="sim1" length="100ms" step="0.01ms" target="net1">
        <OutputFile id="of1" fileName="out.dat">
            <OutputColumn id="c0" quantity="pop1/0/cell0/v"/>
            <OutputColumn id="c1" quantity="pop1/1/cell0/u"/>
            <OutputColumn id="c2" quantity="pop1[1]/v"/>
        </OutputFile>
    </Simulation>
</Lems>
)");

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    ASSERT_EQ(result.recording_profiles.size(), 1u);
    const RecordingConfig &columns = result.recording_profiles[0];
    ASSERT_EQ(columns.selections.size(), 3u);

    EXPECT_EQ(columns.selections[0].neuron_index, 0);
    EXPECT_EQ(columns.selections[0].variable_name, "v");

    // izhikevich2007Cell declares v and u, so a variable name the type does not carry is
    // not a harmless mislabel -- it selects a different state slot.
    EXPECT_EQ(columns.selections[1].neuron_index, 1);
    EXPECT_EQ(columns.selections[1].variable_name, "u");

    // The bracket spelling of the same cell keeps working.
    EXPECT_EQ(columns.selections[2].neuron_index, 1);
    EXPECT_EQ(columns.selections[2].variable_name, "v");

    const CellTypeSpecification *cell = cell_type_named(result, "izhikevich2007Cell");
    ASSERT_NE(cell, nullptr);
    for (const RecordingSelection &selection : columns.selections) {
        EXPECT_NE(std::find(cell->state_variable_names.begin(),
                            cell->state_variable_names.end(), selection.variable_name),
                  cell->state_variable_names.end())
                << selection.quantity_path;
    }
}

TEST(Export, constants_are_namespaced_by_the_type_that_declares_them) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("constant_scoping");
    fixture.write("net.nml", R"(<neuroml id="constantnet">
    <ComponentType name="scaling_cell" extends="baseCell">
        <Constant name="SCALE" dimension="none" value="7"/>
        <Dynamics>
            <StateVariable name="v" dimension="voltage" exposure="v"/>
        </Dynamics>
    </ComponentType>
    <scaling_cell id="cell0"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="1"/>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", R"(<Lems>
    <Target component="sim1"/>
    <Include file="net.nml"/>
    <Constant name="SCALE" dimension="none" value="3"/>
    <Simulation id="sim1" length="100ms" step="0.01ms" target="net1"/>
</Lems>
)");

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    // Two constants both called SCALE: one at document scope, one owned by a
    // ComponentType. Sharing one key would have whichever is read last silently replace
    // the other -- and the standard library really does declare same-named constants on
    // different types.
    ASSERT_EQ(result.global_constants.count("SCALE"), 1u);
    EXPECT_DOUBLE_EQ(result.global_constants.at("SCALE").float64, 3.0);

    ASSERT_EQ(result.global_constants.count("scaling_cell.SCALE"), 1u);
    EXPECT_DOUBLE_EQ(result.global_constants.at("scaling_cell.SCALE").float64, 7.0);

    // A standard-library constant carrying a unit is converted like any other quantity:
    // channelDensityGHK2's VOLT_SCALE is 1mV.
    ASSERT_EQ(result.global_constants.count("channelDensityGHK2.VOLT_SCALE"), 1u);
    EXPECT_DOUBLE_EQ(result.global_constants.at("channelDensityGHK2.VOLT_SCALE").float64,
                     1e-3);
}


// ── error handling, continued ────────────────────────────────────────────────────

TEST(Errors, a_connection_index_that_is_not_a_number_throws) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("malformed_connection_index");
    // Reading "abc" as 0 would wire a real edge onto a neuron the document never named.
    fixture.write("net.nml", R"(<neuroml id="badindexnet">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <alphaCurrentSynapse id="syn0" tau="2ms" ibase="0.1nA"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="3"/>
        <projection id="proj1" presynapticPopulation="pop1"
                    postsynapticPopulation="pop1" synapse="syn0">
            <connection id="0" preCellId="../pop1[0]" postCellId="../pop1[abc]"/>
        </projection>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    try {
        parser.parse_lems(fixture.path_of("LEMS.xml"));
        FAIL() << "expected a malformed connection index to throw";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(String(error.what()).find("pop1[abc]"), String::npos);
    }
}

TEST(Errors, a_slash_form_connection_index_that_is_not_a_number_throws) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("malformed_slash_index");
    fixture.write("net.nml", R"(<neuroml id="badslashnet">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <alphaCurrentSynapse id="syn0" tau="2ms" ibase="0.1nA"/>
    <network id="net1">
        <populationList id="pop1" component="cell0">
            <instance id="0"><location x="0" y="0" z="0"/></instance>
            <instance id="1"><location x="10" y="0" z="0"/></instance>
        </populationList>
        <projection id="proj1" presynapticPopulation="pop1"
                    postsynapticPopulation="pop1" synapse="syn0">
            <connection id="0" preCellId="../pop1/0/cell0" postCellId="../pop1/abc/cell0"/>
        </projection>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    EXPECT_THROW(parser.parse_lems(fixture.path_of("LEMS.xml")), std::runtime_error);
}

TEST(Errors, a_projection_naming_something_that_is_not_a_synapse_throws) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("projection_bad_synapse");
    // The reference resolves to a real instance, just the wrong kind of one. Accepting it
    // would build edges whose synapse dynamics are a cell's.
    fixture.write("net.nml", R"(<neuroml id="badsynnet">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="2"/>
        <projection id="proj1" presynapticPopulation="pop1"
                    postsynapticPopulation="pop1" synapse="cell0">
            <connection id="0" preCellId="../pop1[0]" postCellId="../pop1[1]"/>
        </projection>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    try {
        parser.parse_lems(fixture.path_of("LEMS.xml"));
        FAIL() << "expected a projection naming a cell as its synapse to throw";
    } catch (const std::runtime_error &error) {
        String message = error.what();
        EXPECT_NE(message.find("cell0"), String::npos);
        EXPECT_NE(message.find("iafCell"), String::npos);
    }
}

TEST(Errors, a_population_naming_a_component_that_is_not_a_cell_throws) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("population_bad_component");
    // A declared ComponentType that reaches no category root at all -- ion channels,
    // morphology and biophysical properties are all this shape. Naming one as a
    // population's component is a Phase-3 boundary, not a missing declaration, and the two
    // need different fixes.
    fixture.write("net.nml", R"(<neuroml id="noncellnet">
    <ComponentType name="mysterious_type"/>
    <mysterious_type id="mystery"/>
    <network id="net1">
        <population id="pop1" component="mystery" size="1"/>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    try {
        parser.parse_lems(fixture.path_of("LEMS.xml"));
        FAIL() << "expected a population naming a non-cell component to throw";
    } catch (const std::runtime_error &error) {
        String message = error.what();
        EXPECT_NE(message.find("mystery"), String::npos);
        EXPECT_NE(message.find("mysterious_type"), String::npos);
        EXPECT_NE(message.find("not classified as a cell"), String::npos);
    }
}

TEST(Errors, an_unresolvable_population_component_names_the_population_and_the_value) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("population_component_message");
    fixture.write("net.nml", R"(<neuroml id="unresolvednet">
    <network id="net1">
        <population id="pop1" component="thisInstanceDoesNotExist" size="10"/>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    try {
        parser.parse_lems(fixture.path_of("LEMS.xml"));
        FAIL() << "expected an unresolvable population component to throw";
    } catch (const std::runtime_error &error) {
        // A caller inspecting .what() has to be able to locate the failure without also
        // reading the log: the offending value, the attribute it sat on, and the element.
        String message = error.what();
        EXPECT_NE(message.find("thisInstanceDoesNotExist"), String::npos);
        EXPECT_NE(message.find("component"), String::npos);
        EXPECT_NE(message.find("pop1"), String::npos);
    }
}

TEST(Errors, an_input_list_target_naming_an_unknown_population_throws) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("input_list_bad_target");
    fixture.write("net.nml", R"(<neuroml id="badinputnet">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <pulseGenerator id="pg0" delay="10ms" duration="40ms" amplitude="0.5nA"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="3"/>
        <inputList id="il1" component="pg0" population="pop1">
            <input id="0" target="pop1[0]"/>
            <input id="1" target="nosuchpop[0]"/>
        </inputList>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    try {
        parser.parse_lems(fixture.path_of("LEMS.xml"));
        FAIL() << "expected an inputList target naming an unknown population to throw";
    } catch (const std::runtime_error &error) {
        // Dropping the unresolvable target would leave a stimulus quietly driving fewer
        // neurons than the document asks for.
        EXPECT_NE(String(error.what()).find("nosuchpop"), String::npos);
    }
}

TEST(Errors, an_explicit_input_target_out_of_range_throws) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("explicit_input_out_of_range");
    fixture.write("net.nml", R"(<neuroml id="badstimnet">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <pulseGenerator id="pg0" delay="10ms" duration="40ms" amplitude="0.5nA"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="2"/>
        <explicitInput target="pop1[9]" input="pg0"/>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    try {
        parser.parse_lems(fixture.path_of("LEMS.xml"));
        FAIL() << "expected an out-of-range explicitInput target to throw";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(String(error.what()).find("pop1[9]"), String::npos);
    }
}

TEST(Errors, a_negative_population_size_throws) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("negative_population_size");
    // A negative size makes no member neurons, so the model looks fine -- except that the
    // layout it writes carries neuron_count = -2, which every consumer indexing that range
    // reads as an enormous extent.
    fixture.write("net.nml", R"(<neuroml id="negativenet">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="-2"/>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", lems_simulation_xml("net.nml"));

    NML_Parser parser;
    try {
        parser.parse_lems(fixture.path_of("LEMS.xml"));
        FAIL() << "expected a negative population size to throw";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(String(error.what()).find("pop1"), String::npos);
    }
}

TEST(Errors, an_unresolvable_recording_path_is_marked_rather_than_read_as_neuron_zero) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not bundled";

    FixtureDirectory fixture("unresolvable_recording");
    // A recording is read-only, so an unresolvable one is deliberately not fatal the way a
    // connection or a stimulus target is. What it must never do is fall back to a real
    // neuron: index 0 would log a genuine trace under a column the document asked for
    // somewhere else entirely.
    fixture.write("net.nml", R"(<neuroml id="recordingnet">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV"
             C="0.2nF" leakConductance="0.01uS"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="3"/>
    </network>
</neuroml>
)");
    fixture.write("LEMS.xml", R"(<Lems>
    <Target component="sim1"/>
    <Include file="net.nml"/>
    <Simulation id="sim1" length="100ms" step="0.01ms" target="net1">
        <OutputFile id="of1" fileName="out.dat">
            <OutputColumn id="c0" quantity="nosuchpop[0]/v"/>
            <OutputColumn id="c1" quantity="pop1[99]/v"/>
            <OutputColumn id="c2" quantity="pop1/v"/>
        </OutputFile>
    </Simulation>
</Lems>
)");

    NML_Parser parser;
    NML_ParseResult result = parser.parse_lems(fixture.path_of("LEMS.xml"));

    ASSERT_EQ(result.recording_profiles.size(), 1u);
    const RecordingConfig &columns = result.recording_profiles[0];
    ASSERT_EQ(columns.selections.size(), 3u);

    // An unknown population, an out-of-range index, and a slash-form path with no index at
    // all: none of the three names a cell, and all three are kept as the -1 sentinel with
    // their raw path intact for the diagnostic.
    for (const RecordingSelection &selection : columns.selections) {
        EXPECT_EQ(selection.neuron_index, -1) << selection.quantity_path;
        EXPECT_FALSE(selection.quantity_path.empty());
    }
}
