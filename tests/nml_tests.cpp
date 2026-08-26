#include <algorithm>
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

    // expOneSynapse computes g * (erev - v): it declares erev and requires v, so it needs
    // the postsynaptic voltage.
    EXPECT_TRUE(synapse->is_conductance_based);

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
    Vector<Vector<s32>> adjacency = build_adjacency_list(result);

    ASSERT_EQ(adjacency.size(), result.neurons.size());
    for (usize source = 0; source < result.neurons.size(); source += 1) {
        const Vector<NetworkEdge> &edges = result.neurons[source].outgoing_edges;
        ASSERT_EQ(adjacency[source].size(), edges.size()) << source;

        for (usize edge = 0; edge < edges.size(); edge += 1) {
            EXPECT_EQ(adjacency[source][edge], edges[edge].target_neuron_index);
        }
    }

    EXPECT_EQ(adjacency[0], (Vector<s32>{4}));
    EXPECT_EQ(adjacency[2], (Vector<s32>{3}));
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
    // 10ms and 10ms + 40ms at dt = 0.01ms. Truncation would give 999 and 4998.
    EXPECT_EQ(explicit_input.start_tick, 1000);
    EXPECT_EQ(explicit_input.end_tick, 5000);
    ASSERT_EQ(explicit_input.targets.size(), 1u);
    EXPECT_EQ(explicit_input.targets[0].neuron_index, 0);

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

    // A spikeArray declares no delay or duration of its own, so the window stays at its
    // default. The train's own last event is not a "max delay" -- the engine sizes its
    // delay ring from connection delays, not from how late a stimulus fires.
    EXPECT_EQ(train.start_tick, 0);
    EXPECT_EQ(train.end_tick, 0);
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
