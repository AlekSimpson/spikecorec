#include <algorithm>
#include <stdexcept>
#include <gtest/gtest.h>

#include "spikecorec/core/recording.h"
#include "spikecorec/core/types.h"
#include "spikecorec/core/units.h"
#include "spikecorec/nml/dynamics.h"
#include "spikecorec/nml/parser.h"
#include "support/test_support.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;
using namespace spikecorec::test_support;

namespace {

// Parses a model the way the engine does: the whole standard library first, then the file.
void parse_model(NML_Context &context, const String &main_filepath) {
    ASSERT_TRUE(standard_library_available()) << "the NeuroML standard library is not at "
                                              << context.STANDARD_LIBRARY_PATH;
    context.parse(main_filepath);
}

// The message of what parsing main_filepath throws, or "" when it does not throw.
String parse_error(const String &main_filepath) {
    NML_Context context;
    try {
        context.parse(main_filepath);
    } catch (const runtime_error &error) {
        return error.what();
    }
    return "";
}

const NML_ComponentType &type_named(const NML_Context &context, const String &type_name) {
    return context.component_types.at(type_name);
}

// The dynamics entries of one type that came from one kind of tag.
Vector<NML_DynamicsExpression> entries_of(const NML_ComponentType &component_type, NML_DeclarationType tag) {
    Vector<NML_DynamicsExpression> entries;
    for (const NML_DynamicsExpression &entry : component_type.dynamics) {
        if (entry.source_tag == tag) entries.push_back(entry);
    }
    return entries;
}

// Two populations of different cell types, a projection with a connectionWD and one with a plain
// connection, an explicitInput and an inputList with one plain and one weighted input. Valid NeuroML,
// whose schema wants synapses before cells.
String two_population_network() {
    return R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="testnet">
    <expOneSynapse id="syn0" gbase="0.5nS" erev="0mV" tauDecay="3ms"/>
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV" C="0.2nF" leakConductance="0.01uS"/>
    <izhikevich2007Cell id="cell1" C="100pF" v0="-60mV" k="0.7nS_per_mV" vr="-60mV" vt="-40mV" vpeak="35mV"
                        a="0.03per_ms" b="-2nS" c="-50mV" d="100pA"/>
    <pulseGenerator id="pg0" delay="10ms" duration="40ms" amplitude="0.5nA"/>
    <network id="net1">
        <population id="pop1" component="cell0" size="3"/>
        <population id="pop2" component="cell1" size="2"/>
        <projection id="proj1" presynapticPopulation="pop1" postsynapticPopulation="pop2" synapse="syn0">
            <connectionWD id="0" preCellId="../pop1[0]" postCellId="../pop2[1]" weight="2.5" delay="2ms"/>
        </projection>
        <projection id="proj2" presynapticPopulation="pop1" postsynapticPopulation="pop2" synapse="syn0">
            <connection id="0" preCellId="../pop1[2]" postCellId="../pop2[0]"/>
        </projection>
        <explicitInput target="pop1[0]" input="pg0"/>
        <inputList id="il1" component="pg0" population="pop2">
            <input id="0" target="../pop2[0]" destination="synapses"/>
            <inputW id="1" target="../pop2[1]" destination="synapses" weight="3.0"/>
        </inputList>
    </network>
</neuroml>
)";
}

String recorded_simulation(const String &included_file, const String &step = "0.01ms") {
    return R"(<Lems>
    <Target component="sim1"/>
    <Include file=")" + included_file + R"("/>
    <Simulation id="sim1" length="100ms" step=")" + step + R"(" target="net1" seed="1234">
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

// A one-population network document with `network_body` inside <network id="net1">.
String one_population_network(const String &network_body, const String &declarations = "") {
    return R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="net">
    <iafCell id="cell0" leakReversal="-50mV" thresh="-55mV" reset="-70mV" C="0.2nF" leakConductance="0.01uS"/>
    <expOneSynapse id="syn0" gbase="0.5nS" erev="0mV" tauDecay="3ms"/>
)" + declarations + R"(    <network id="net1">
)" + network_body + R"(    </network>
</neuroml>
)";
}

// The two-population model written out and parsed.
struct TwoPopulationModel {
    TemporaryDirectory directory;
    NML_Context context;

    TwoPopulationModel() {
        directory.write("net.nml", two_population_network());
        parse_model(context, directory.write("LEMS.xml", recorded_simulation("net.nml")));
    }

    // Where a variable of pop2's cells (izhikevich2007Cell, two state variables) sits in cell memory.
    s64 izhikevich_memory_index(s64 cell, const String &variable) const {
        const Vector<String> &names = type_named(context, "izhikevich2007Cell").state_variable_names;
        const s64 offset = find(names.begin(), names.end(), variable) - names.begin();
        return context.simulation.population_base_indices.at("net1/pop2") + cell * (s64)names.size() + offset;
    }
};

} // namespace

// ── quantities ──────────────────────────────────────────────────────────────────

TEST(Quantity, a_declared_unit_takes_precedence_over_the_built_in_table) {
    const TemporaryDirectory directory;
    // A deliberately non-SI redefinition: if the document's <Unit> is consulted, mV scales by 1000.
    const String main_file = directory.write("units.xml", R"(<Lems>
    <Unit symbol="mV" dimension="voltage" power="3"/>
    <Unit symbol="quux" dimension="none" power="0" scale="7"/>
</Lems>
)");
    NML_Context context;
    parse_model(context, main_file);

    EXPECT_DOUBLE_EQ(context.resolve_quantity("2mV"), 2000.0);
    EXPECT_DOUBLE_EQ(context.resolve_quantity("3quux"), 21.0);
    EXPECT_DOUBLE_EQ(context.resolve_quantity("5ms"), 0.005);
    // The free function knows only the built-in table.
    EXPECT_DOUBLE_EQ(units::parse_quantity("2mV"), 0.002);
}

TEST(Quantity, standard_library_units_carry_their_scale_power_and_offset) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("units.xml", R"(<Lems>
    <Unit symbol="offsetUnit" dimension="temperature" offset="10"/>
</Lems>
)");
    NML_Context context;
    parse_model(context, main_file);

    EXPECT_DOUBLE_EQ(context.resolve_quantity("20degC"), 293.15);
    EXPECT_DOUBLE_EQ(context.resolve_quantity("-60mV"), -0.06);
    EXPECT_DOUBLE_EQ(context.resolve_quantity("2min"), 120.0);
    EXPECT_DOUBLE_EQ(context.resolve_quantity("0.7nS_per_mV"), 0.7e-6);
    EXPECT_DOUBLE_EQ(context.resolve_quantity("5offsetUnit"), 15.0);
    EXPECT_DOUBLE_EQ(context.resolve_quantity("12"), 12.0);
}

// A <Constant> resolves after every document is read, so it may use a unit a later file declares.
TEST(Quantity, constants_resolve_with_units_declared_in_any_file) {
    const TemporaryDirectory directory;
    directory.write("later_units.xml", R"(<Lems>
    <Unit symbol="doubleUnit" dimension="none" scale="2"/>
</Lems>
)");
    const String main_file = directory.write("constants.xml", R"(<Lems>
    <Include file="later_units.xml"/>
    <Constant name="threshold_constant" dimension="voltage" value="-50mV"/>
    <Constant name="doubled_constant" dimension="none" value="3doubleUnit"/>
</Lems>
)");
    NML_Context context;
    parse_model(context, main_file);

    EXPECT_DOUBLE_EQ(context.model_constants.at("threshold_constant").float64, -0.05);
    EXPECT_DOUBLE_EQ(context.model_constants.at("doubled_constant").float64, 6.0);
    EXPECT_DOUBLE_EQ(context.simulation.global_constants.at("doubled_constant").float64, 6.0);
}

// ── ComponentType resolution ────────────────────────────────────────────────────

TEST(Resolution, a_multi_hop_extends_chain_links_every_ancestor) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("types.xml", R"(<Lems>
    <ComponentType name="base_a">
        <Parameter name="alpha" dimension="none"/>
        <Dynamics><StateVariable name="first" dimension="none"/></Dynamics>
    </ComponentType>
    <ComponentType name="mid_b" extends="base_a">
        <Parameter name="gamma" dimension="none"/>
        <Dynamics><StateVariable name="second" dimension="none"/></Dynamics>
    </ComponentType>
    <ComponentType name="leaf_c" extends="mid_b">
        <Dynamics><StateVariable name="third" dimension="none"/></Dynamics>
    </ComponentType>
</Lems>
)");
    NML_Context context;
    parse_model(context, main_file);

    const NML_ComponentType &leaf = type_named(context, "leaf_c");
    ASSERT_NE(leaf.extends, nullptr);
    EXPECT_EQ(leaf.extends->name, "mid_b");
    ASSERT_NE(leaf.extends->extends, nullptr);
    EXPECT_EQ(leaf.extends->extends->name, "base_a");
    EXPECT_EQ(leaf.extends->extends->extends, nullptr);

    // Declarations are found up the chain, and state variables come ancestors first.
    EXPECT_NE(leaf.find_declaration("var:alpha"), nullptr);
    EXPECT_NE(leaf.find_declaration("var:gamma"), nullptr);
    EXPECT_EQ(leaf.find_declaration("var:missing"), nullptr);
    EXPECT_EQ(leaf.state_variable_names, (Vector<String>{"first", "second", "third"}));
}

TEST(Resolution, extends_resolves_forward_references_and_across_files) {
    const TemporaryDirectory directory;
    directory.write("parent.xml", R"(<Lems>
    <ComponentType name="parent_type"><Parameter name="inherited" dimension="none"/></ComponentType>
</Lems>
)");
    // The child comes before the parent's file is read: extends is a name, resolved by lookup.
    const String main_file = directory.write("child.xml", R"(<Lems>
    <ComponentType name="child_type" extends="parent_type"><Parameter name="own" dimension="none"/></ComponentType>
    <ComponentType name="early_child" extends="late_parent"/>
    <ComponentType name="late_parent"/>
    <Include file="parent.xml"/>
</Lems>
)");
    NML_Context context;
    parse_model(context, main_file);

    ASSERT_NE(type_named(context, "child_type").extends, nullptr);
    EXPECT_EQ(type_named(context, "child_type").extends->name, "parent_type");
    EXPECT_NE(type_named(context, "child_type").find_declaration("var:inherited"), nullptr);
    ASSERT_NE(type_named(context, "early_child").extends, nullptr);
    EXPECT_EQ(type_named(context, "early_child").extends->name, "late_parent");
}

// A subtype's declaration shadows its ancestor's of the same name, and several tags share the
// variable namespace, so a StateVariable can shadow a Parameter.
TEST(Resolution, a_redeclared_name_shadows_the_ancestors_declaration) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("types.xml", R"(<Lems>
    <ComponentType name="parent_type">
        <Parameter name="beta" dimension="voltage"/>
        <Parameter name="tau" dimension="time"/>
        <Dynamics><StateVariable name="x" dimension="none"/></Dynamics>
    </ComponentType>
    <ComponentType name="child_type" extends="parent_type">
        <Parameter name="beta" dimension="current"/>
        <Parameter name="x" dimension="none"/>
        <Dynamics><StateVariable name="tau" dimension="time"/></Dynamics>
    </ComponentType>
</Lems>
)");
    NML_Context context;
    parse_model(context, main_file);
    const NML_ComponentType &child = type_named(context, "child_type");

    const NML_Node *beta = child.find_declaration("var:beta");
    ASSERT_NE(beta, nullptr);
    EXPECT_EQ(beta->body.get_attribute("dimension"), "current");

    const NML_Node *tau = child.find_declaration("var:tau");
    ASSERT_NE(tau, nullptr);
    EXPECT_EQ(tau->body.tag_type, NML_DeclarationType::StateVariable);

    // x is a state variable of the parent but a parameter of the child, so the child has no slot for it.
    EXPECT_EQ(child.state_variable_names, (Vector<String>{"tau"}));
    EXPECT_EQ(type_named(context, "parent_type").state_variable_names, (Vector<String>{"x"}));
}

TEST(Resolution, an_extends_cycle_throws) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("types.xml", R"(<Lems>
    <ComponentType name="type_a" extends="type_b"/>
    <ComponentType name="type_b" extends="type_a"/>
</Lems>
)");
    EXPECT_NE(parse_error(main_file).find("Cyclic ComponentType extends chain"), String::npos);
}

TEST(Resolution, an_unresolved_extends_throws_naming_the_missing_type) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("types.xml", R"(<Lems>
    <ComponentType name="orphan_type" extends="no_such_type"/>
</Lems>
)");
    EXPECT_NE(parse_error(main_file).find("no_such_type"), String::npos);
}

// A type's own <Dynamics> replaces its ancestors'; without one it inherits the nearest.
TEST(Resolution, dynamics_come_from_the_nearest_type_that_declares_them) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("types.xml", R"(<Lems>
    <ComponentType name="parent_type">
        <Dynamics><StateVariable name="x" dimension="none"/><TimeDerivative variable="x" value="1"/></Dynamics>
    </ComponentType>
    <ComponentType name="replacing_type" extends="parent_type">
        <Dynamics><StateVariable name="y" dimension="none"/><TimeDerivative variable="y" value="2"/></Dynamics>
    </ComponentType>
    <ComponentType name="inheriting_type" extends="parent_type"/>
</Lems>
)");
    NML_Context context;
    parse_model(context, main_file);

    const Vector<NML_DynamicsExpression> replaced =
            entries_of(type_named(context, "replacing_type"), NML_DeclarationType::TimeDerivative);
    ASSERT_EQ(replaced.size(), 1u);
    EXPECT_EQ(replaced[0].target, "y");

    const Vector<NML_DynamicsExpression> inherited =
            entries_of(type_named(context, "inheriting_type"), NML_DeclarationType::TimeDerivative);
    ASSERT_EQ(inherited.size(), 1u);
    EXPECT_EQ(inherited[0].target, "x");
    EXPECT_EQ(inherited[0].expression, "1");
}

// ── component instances ─────────────────────────────────────────────────────────

TEST(Instances, only_attributes_the_type_declares_are_bound) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("instances.xml", R"(<Lems>
    <ComponentType name="bound_type">
        <Parameter name="declared" dimension="none"/>
        <Text name="label"/>
        <Dynamics><StateVariable name="state" dimension="none"/></Dynamics>
    </ComponentType>
    <bound_type id="example" declared="3" label="hello" undeclared="4" state="5"/>
</Lems>
)");
    NML_Context context;
    parse_model(context, main_file);

    const NML_ComponentInstance *example = context.find_instance("example");
    ASSERT_NE(example, nullptr);
    EXPECT_TRUE(context.is_instance_of(example, "bound_type"));
    EXPECT_EQ(example->value_or("declared"), "3");
    EXPECT_EQ(example->value_or("label"), "hello");
    EXPECT_FALSE(example->has_value("undeclared"));
    // A state variable is not instance-set.
    EXPECT_FALSE(example->has_value("state"));
}

// <Fixed> pins a parameter for every instance of the subtype, over whatever the instance says.
TEST(Instances, fixed_pins_an_inherited_parameter) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("instances.xml", R"(<Lems>
    <ComponentType name="parent_type">
        <Parameter name="tau" dimension="time"/>
        <Parameter name="loose" dimension="time"/>
    </ComponentType>
    <ComponentType name="child_type" extends="parent_type">
        <Fixed parameter="tau" value="10ms"/>
    </ComponentType>
    <child_type id="pinned" tau="5ms" loose="3ms"/>
    <parent_type id="unpinned" tau="5ms"/>
</Lems>
)");
    NML_Context context;
    parse_model(context, main_file);

    EXPECT_EQ(context.find_instance("pinned")->value_or("tau"), "10ms");
    EXPECT_EQ(context.find_instance("pinned")->value_or("loose"), "3ms");
    EXPECT_EQ(context.find_instance("unpinned")->value_or("tau"), "5ms");
}

TEST(Instances, children_are_keyed_by_their_path_and_unnamed_ones_by_type_and_ordinal) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("instances.xml", R"(<Lems>
    <ComponentType name="holder_type"><Children name="members" type="member_type"/></ComponentType>
    <ComponentType name="member_type"/>
    <holder_type id="holder">
        <member_type id="named"/>
        <member_type/>
        <member_type/>
    </holder_type>
</Lems>
)");
    NML_Context context;
    parse_model(context, main_file);

    const NML_ComponentInstance *holder = context.find_instance("holder");
    ASSERT_NE(holder, nullptr);
    EXPECT_EQ(holder->data_order,
              (Vector<String>{"holder/named", "holder/member_type0", "holder/member_type1"}));
    ASSERT_NE(context.find_instance("holder/member_type1"), nullptr);
    EXPECT_EQ(context.find_instance("holder/member_type1")->parent_instance, holder);
}

TEST(Instances, a_duplicate_instance_id_throws) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("instances.xml", R"(<Lems>
    <ComponentType name="plain_type"/>
    <plain_type id="same"/>
    <plain_type id="same"/>
</Lems>
)");
    EXPECT_NE(parse_error(main_file).find("Duplicate component instance 'same'"), String::npos);
}

// Real documents carry annotations that map to no ComponentType; they are skipped, not fatal.
TEST(Instances, tags_with_no_component_type_are_skipped) {
    const TemporaryDirectory directory;
    directory.write("net.nml", one_population_network(R"(        <someOtherAnnotation/>
        <population id="pop1" component="cell0" size="2"/>
)", "    <someUnmodelledAnnotation foo=\"bar\"/>\n"));
    NML_Context context;
    parse_model(context, directory.write("LEMS.xml", recorded_simulation("net.nml")));

    EXPECT_EQ(context.simulation.total_neuron_count, 2);
    EXPECT_EQ(context.find_instance("someUnmodelledAnnotation0"), nullptr);
}

// ── includes ────────────────────────────────────────────────────────────────────

TEST(Include, resolves_relative_to_the_including_file) {
    const TemporaryDirectory directory;
    directory.write("nested/leaf.xml", R"(<Lems><ComponentType name="leaf_type"/></Lems>)");
    directory.write("nested/branch.xml", R"(<Lems><Include file="leaf.xml"/></Lems>)");
    const String main_file = directory.write("root.xml", R"(<Lems><Include file="nested/branch.xml"/></Lems>)");
    NML_Context context;
    parse_model(context, main_file);

    // branch.xml names leaf.xml with no directory, so it resolves only against branch.xml's own.
    EXPECT_EQ(context.component_types.count("leaf_type"), 1u);
}

TEST(Include, a_cycle_terminates) {
    const TemporaryDirectory directory;
    directory.write("second.xml", R"(<Lems><Include file="first.xml"/><ComponentType name="second_type"/></Lems>)");
    const String main_file =
            directory.write("first.xml", R"(<Lems><Include file="second.xml"/><ComponentType name="first_type"/></Lems>)");
    NML_Context context;
    parse_model(context, main_file);

    EXPECT_EQ(context.component_types.count("first_type"), 1u);
    EXPECT_EQ(context.component_types.count("second_type"), 1u);
}

// "shared.xml" and "./shared.xml" are one file: deduplicating on the raw text would read it twice
// and report its type as a redefinition.
TEST(Include, a_diamond_include_reads_the_shared_file_once) {
    const TemporaryDirectory directory;
    directory.write("shared.xml", R"(<Lems><ComponentType name="shared_type"/></Lems>)");
    directory.write("left.xml", R"(<Lems><Include file="shared.xml"/></Lems>)");
    directory.write("right.xml", R"(<Lems><Include file="./shared.xml"/></Lems>)");
    const String main_file =
            directory.write("root.xml", R"(<Lems><Include file="left.xml"/><Include file="right.xml"/></Lems>)");
    NML_Context context;
    EXPECT_NO_THROW(parse_model(context, main_file));
    EXPECT_EQ(context.component_types.count("shared_type"), 1u);
}

TEST(Include, a_type_defined_in_two_files_throws) {
    const TemporaryDirectory directory;
    directory.write("first.xml", R"(<Lems><ComponentType name="clashing_type"/></Lems>)");
    directory.write("second.xml", R"(<Lems><ComponentType name="clashing_type"/></Lems>)");
    const String main_file =
            directory.write("root.xml", R"(<Lems><Include file="first.xml"/><Include file="second.xml"/></Lems>)");
    EXPECT_NE(parse_error(main_file).find("Duplicate ComponentType 'clashing_type'"), String::npos);
}

// ── dynamics ────────────────────────────────────────────────────────────────────

// Entries carry their regime, and an event handler's actions carry the handler's condition.
TEST(Dynamics, regime_entries_carry_their_regime_and_condition) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("types.xml", R"(<Lems>
    <ComponentType name="regime_cell">
        <Dynamics>
            <StateVariable name="v" dimension="voltage"/>
            <StateVariable name="elapsed" dimension="time"/>
            <Regime name="integrating" initial="true">
                <TimeDerivative variable="v" value="(vrest - v) / tau"/>
                <OnCondition test="v .gt. vthresh">
                    <StateAssignment variable="v" value="vreset"/>
                    <EventOut port="spike"/>
                    <Transition regime="refractory"/>
                </OnCondition>
            </Regime>
            <Regime name="refractory">
                <OnEntry><StateAssignment variable="elapsed" value="0"/></OnEntry>
                <TimeDerivative variable="elapsed" value="1"/>
            </Regime>
        </Dynamics>
    </ComponentType>
</Lems>
)");
    NML_Context context;
    parse_model(context, main_file);
    const NML_ComponentType &cell = type_named(context, "regime_cell");

    const Vector<NML_DynamicsExpression> regimes = entries_of(cell, NML_DeclarationType::Regime);
    ASSERT_EQ(regimes.size(), 2u);
    EXPECT_EQ(regimes[0].target, "integrating");
    EXPECT_EQ(regimes[0].expression, "true");
    EXPECT_EQ(regimes[1].target, "refractory");

    const Vector<NML_DynamicsExpression> derivatives = entries_of(cell, NML_DeclarationType::TimeDerivative);
    ASSERT_EQ(derivatives.size(), 2u);
    EXPECT_EQ(derivatives[0].target, "v");
    EXPECT_EQ(derivatives[0].regime_name, "integrating");
    EXPECT_EQ(derivatives[1].target, "elapsed");
    EXPECT_EQ(derivatives[1].regime_name, "refractory");

    const Vector<NML_DynamicsExpression> assignments = entries_of(cell, NML_DeclarationType::StateAssignment);
    ASSERT_EQ(assignments.size(), 2u);
    EXPECT_EQ(assignments[0].target, "v");
    EXPECT_EQ(assignments[0].expression, "vreset");
    EXPECT_EQ(assignments[0].condition, "v .gt. vthresh");
    EXPECT_EQ(assignments[1].target, "elapsed");
    EXPECT_EQ(assignments[1].regime_name, "refractory");
    EXPECT_EQ(assignments[1].condition, "");

    const Vector<NML_DynamicsExpression> events = entries_of(cell, NML_DeclarationType::EventOut);
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0].target, "spike");
    EXPECT_EQ(events[0].condition, "v .gt. vthresh");

    const Vector<NML_DynamicsExpression> transitions = entries_of(cell, NML_DeclarationType::Transition);
    ASSERT_EQ(transitions.size(), 1u);
    EXPECT_EQ(transitions[0].target, "refractory");
}

// iafCell writes v in two handlers. The OnStart assignment is an entry of its own; the reset
// carries the threshold condition. Tagging them alike would reapply OnStart every tick.
TEST(Dynamics, on_start_is_kept_apart_from_a_conditional_reset) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("root.xml", "<Lems/>");
    NML_Context context;
    parse_model(context, main_file);
    const NML_ComponentType &iaf = type_named(context, "iafCell");

    const Vector<NML_DynamicsExpression> on_start = entries_of(iaf, NML_DeclarationType::OnStart);
    ASSERT_EQ(on_start.size(), 1u);
    EXPECT_EQ(on_start[0].target, "v");
    EXPECT_EQ(on_start[0].expression, "leakReversal");

    const Vector<NML_DynamicsExpression> resets = entries_of(iaf, NML_DeclarationType::StateAssignment);
    ASSERT_EQ(resets.size(), 1u);
    EXPECT_EQ(resets[0].target, "v");
    EXPECT_EQ(resets[0].expression, "reset");
    EXPECT_EQ(resets[0].condition, "v .gt. thresh");

    const Vector<NML_DynamicsExpression> derivatives = entries_of(iaf, NML_DeclarationType::TimeDerivative);
    ASSERT_EQ(derivatives.size(), 1u);
    EXPECT_EQ(derivatives[0].expression, "iMemb / C");

    const Vector<NML_DynamicsExpression> derived = entries_of(iaf, NML_DeclarationType::DerivedVariable);
    ASSERT_EQ(derived.size(), 2u);
    EXPECT_EQ(derived[0].target, "iSyn");
    EXPECT_EQ(derived[0].select, "synapses[*]/i");
    EXPECT_EQ(derived[0].reduce, "add");
}

// An arrival's assignment follows the OnEvent entry that names its port.
TEST(Dynamics, an_on_event_entry_names_its_port) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("root.xml", "<Lems/>");
    NML_Context context;
    parse_model(context, main_file);
    const NML_ComponentType &synapse = type_named(context, "expOneSynapse");

    const auto on_event = find_if(synapse.dynamics.begin(), synapse.dynamics.end(),
                                  [](const NML_DynamicsExpression &entry) {
                                      return entry.source_tag == NML_DeclarationType::OnEvent;
                                  });
    ASSERT_NE(on_event, synapse.dynamics.end());
    EXPECT_EQ(on_event->target, "in");
    ASSERT_NE(on_event + 1, synapse.dynamics.end());
    EXPECT_EQ((on_event + 1)->source_tag, NML_DeclarationType::StateAssignment);
    EXPECT_EQ((on_event + 1)->target, "g");
    EXPECT_EQ((on_event + 1)->expression, "g + (weight * gbase)");
}

// <Structure> is read like <Dynamics>: a spike sends its event to its parent, and a
// timedSynapticInput instantiates its synapse and sends its own events into it.
TEST(Structure, child_instances_aliases_and_event_connections_are_recorded) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("root.xml", "<Lems/>");
    NML_Context context;
    parse_model(context, main_file);

    const Vector<NML_StructureEntry> &spike = type_named(context, "spike").structure;
    ASSERT_EQ(spike.size(), 3u);
    EXPECT_EQ(spike[0].source_tag, NML_DeclarationType::With);
    EXPECT_EQ(spike[0].instance, "this");
    EXPECT_EQ(spike[0].alias, "a");
    EXPECT_EQ(spike[1].source_tag, NML_DeclarationType::With);
    EXPECT_EQ(spike[1].instance, "parent");
    EXPECT_EQ(spike[1].alias, "b");
    EXPECT_EQ(spike[2].source_tag, NML_DeclarationType::EventConnection);
    EXPECT_EQ(spike[2].source, "a");
    EXPECT_EQ(spike[2].target, "b");
    EXPECT_EQ(spike[2].receiver, "");

    const Vector<NML_StructureEntry> &timed = type_named(context, "timedSynapticInput").structure;
    ASSERT_EQ(timed.size(), 4u);
    EXPECT_EQ(timed[0].source_tag, NML_DeclarationType::ChildInstance);
    EXPECT_EQ(timed[0].component, "synapse");
    EXPECT_EQ(timed[2].instance, "spikeTarget");
    EXPECT_EQ(timed[3].source_tag, NML_DeclarationType::EventConnection);

    // A type without its own <Structure> inherits its nearest ancestor's.
    EXPECT_EQ(type_named(context, "spikeGeneratorRefPoisson").structure.size(),
              type_named(context, "spikeGeneratorPoisson").structure.size());
    EXPECT_TRUE(type_named(context, "pulseGenerator").structure.empty());
}

TEST(Dynamics, conditional_derived_variables_become_one_entry_per_case) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("types.xml", R"(<Lems>
    <ComponentType name="case_type">
        <Dynamics>
            <StateVariable name="x" dimension="none"/>
            <ConditionalDerivedVariable name="rectified" dimension="none">
                <Case condition="x .gt. 0" value="x"/>
                <Case value="0"/>
            </ConditionalDerivedVariable>
        </Dynamics>
    </ComponentType>
</Lems>
)");
    NML_Context context;
    parse_model(context, main_file);

    const Vector<NML_DynamicsExpression> cases = entries_of(type_named(context, "case_type"), NML_DeclarationType::Case);
    ASSERT_EQ(cases.size(), 2u);
    EXPECT_EQ(cases[0].target, "rectified");
    EXPECT_EQ(cases[0].condition, "x .gt. 0");
    EXPECT_EQ(cases[0].expression, "x");
    EXPECT_EQ(cases[1].condition, "");
    EXPECT_EQ(cases[1].expression, "0");
}

// Every type in the standard library links its extends chain, and the cells' state variables are
// what NeuroML declares.
TEST(Resolution, the_whole_standard_library_resolves) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("root.xml", "<Lems/>");
    NML_Context context;
    ASSERT_NO_THROW(parse_model(context, main_file));

    EXPECT_GT(context.component_types.size(), 250u);
    for (const auto &[type_name, component_type] : context.component_types) {
        EXPECT_EQ(component_type.name, type_name);
        const String parent_name = component_type.source_node->body.get_attribute("extends");
        if (parent_name.empty()) {
            EXPECT_EQ(component_type.extends, nullptr) << type_name;
        } else {
            ASSERT_NE(component_type.extends, nullptr) << type_name;
            EXPECT_EQ(component_type.extends->name, parent_name) << type_name;
        }
    }
    EXPECT_EQ(type_named(context, "iafCell").state_variable_names, (Vector<String>{"v"}));
    Vector<String> izhikevich_names = type_named(context, "izhikevich2007Cell").state_variable_names;
    sort(izhikevich_names.begin(), izhikevich_names.end());
    EXPECT_EQ(izhikevich_names, (Vector<String>{"u", "v"}));
}

// ── the simulation ──────────────────────────────────────────────────────────────

TEST(Simulation, settings_come_from_the_lems_target) {
    const TwoPopulationModel model;
    const NML_Context::NML_SimulationContext &simulation = model.context.simulation;

    EXPECT_EQ(simulation.simulation_component_id, "sim1");
    EXPECT_EQ(simulation.target_network_id, "net1");
    EXPECT_DOUBLE_EQ(simulation.step_dt, 1e-5);
    EXPECT_DOUBLE_EQ(simulation.simulation_duration, 0.1);
    EXPECT_EQ(simulation.total_tick_count, 10000);
    ASSERT_TRUE(simulation.random_seed.has_value());
    EXPECT_EQ(*simulation.random_seed, 1234u);
}

// Populations sit in cell memory in document order, each cell taking one slot per state variable.
TEST(Simulation, populations_are_laid_out_in_document_order) {
    const TwoPopulationModel model;
    const NML_Context &context = model.context;
    const NML_Context::NML_SimulationContext &simulation = context.simulation;

    EXPECT_EQ(simulation.total_neuron_count, 5);
    ASSERT_EQ(simulation.cell_instances.size(), 2u);
    EXPECT_EQ(simulation.cell_instances[0].id, "cell0");
    EXPECT_EQ(simulation.cell_instances[1].id, "cell1");

    // pop1 is three iafCells of one slot each; pop2 follows with two cells of two slots each.
    EXPECT_EQ(simulation.population_base_indices.at("net1/pop1"), 0);
    EXPECT_EQ(simulation.population_base_indices.at("net1/pop2"), 3);
    EXPECT_EQ(context.get_cell_state_size(), 7);

    EXPECT_EQ(context.resolve_path("pop1[2]"), 2);
    EXPECT_EQ(context.resolve_path("pop2[1]"), 5);
    EXPECT_EQ(context.resolve_path("pop2[1]/v"), model.izhikevich_memory_index(1, "v"));
    EXPECT_EQ(context.resolve_path("pop2[1]/u"), model.izhikevich_memory_index(1, "u"));
    EXPECT_EQ(context.resolve_path("pop2[1]/no_such_variable"), -1);
    EXPECT_EQ(context.resolve_path("no_such_population[0]"), -1);

    for (s64 neuron = 0; neuron < 3; neuron += 1) EXPECT_EQ(context.neuron_index_of(neuron), neuron);
    EXPECT_EQ(context.neuron_index_of(model.izhikevich_memory_index(0, "v")), 3);
    EXPECT_EQ(context.neuron_index_of(model.izhikevich_memory_index(1, "u")), 4);
    EXPECT_EQ(context.neuron_index_of(7), -1);
}

// Instance values stay as written and resolve to SI when used.
TEST(Simulation, cell_parameters_resolve_to_si) {
    const TwoPopulationModel model;
    const NML_ComponentInstance *cell = model.context.find_instance("cell0");
    ASSERT_NE(cell, nullptr);
    EXPECT_EQ(cell->value_or("leakReversal"), "-50mV");

    const auto si_value = [&](const String &parameter) {
        return model.context.resolve_quantity(cell->value_or(parameter));
    };
    EXPECT_DOUBLE_EQ(si_value("leakReversal"), -0.05);
    EXPECT_DOUBLE_EQ(si_value("thresh"), -0.055);
    EXPECT_DOUBLE_EQ(si_value("reset"), -0.07);
    EXPECT_DOUBLE_EQ(si_value("C"), 0.2e-9);
    EXPECT_DOUBLE_EQ(si_value("leakConductance"), 0.01e-6);
}

TEST(Simulation, connections_resolve_to_neurons_with_their_weight_and_delay) {
    const TwoPopulationModel model;
    const NML_Context::NML_SimulationContext &simulation = model.context.simulation;

    EXPECT_EQ(simulation.total_edge_count, 2);
    ASSERT_EQ(simulation.synapse_instances.size(), 1u);
    EXPECT_EQ(simulation.synapse_instances[0].id, "syn0");

    // pop1[0] is neuron 0 and pop2[1] is neuron 4; 2 ms at 0.01 ms is 200 ticks, not 199.
    ASSERT_EQ(simulation.network_data.list[0].size(), 1u);
    const NML_NetworkEdge &weighted = simulation.network_data.list[0][0];
    EXPECT_EQ(weighted.child, 4);
    EXPECT_FLOAT_EQ(weighted.weight, 2.5f);
    EXPECT_EQ(weighted.delay_ticks, 200);
    EXPECT_EQ(weighted.component_id, "syn0");
    EXPECT_EQ(simulation.maximum_edge_delay, 200);

    // A plain <connection> states neither weight nor delay.
    ASSERT_EQ(simulation.network_data.list[2].size(), 1u);
    const NML_NetworkEdge &plain = simulation.network_data.list[2][0];
    EXPECT_EQ(plain.child, 3);
    EXPECT_FLOAT_EQ(plain.weight, 1.0f);
    EXPECT_EQ(plain.delay_ticks, 0);

    EXPECT_TRUE(simulation.network_data.list[1].empty());
    EXPECT_TRUE(simulation.network_data.list[3].empty());
}

// An input profile keeps the input component and its targets; the input's own dynamics are compiled.
TEST(Simulation, inputs_resolve_their_targets) {
    const TwoPopulationModel model;
    const Vector<SimulationInputConfig> &profiles = model.context.simulation.input_profiles;
    ASSERT_EQ(profiles.size(), 2u);

    const SimulationInputConfig &explicit_input = profiles[0];
    EXPECT_EQ(explicit_input.input_component_id, "pg0");
    ASSERT_EQ(explicit_input.targets.size(), 1u);
    EXPECT_EQ(explicit_input.targets[0].neuron_index, 0);

    // Targets are the cell-memory index of the cell's first slot: pop2 starts at 3, two slots a cell.
    // <input> weighs 1 and <inputW> carries its own weight.
    const SimulationInputConfig &input_list = profiles[1];
    ASSERT_EQ(input_list.targets.size(), 2u);
    EXPECT_EQ(input_list.targets[0].neuron_index, 3);
    EXPECT_EQ(input_list.targets[1].neuron_index, 5);
    EXPECT_DOUBLE_EQ(input_list.targets[0].weight, 1.0);
    EXPECT_DOUBLE_EQ(input_list.targets[1].weight, 3.0);
}

// ── input component trees ───────────────────────────────────────────────────────

namespace {

// Parses a one-population network with `declarations` and an explicitInput of input_id onto
// pop1[0], and returns the input's tree.
InputTree input_tree_of(NML_Context &context, const TemporaryDirectory &directory, const String &declarations,
                        const String &input_id) {
    directory.write("net.nml", one_population_network(R"(        <population id="pop1" component="cell0" size="2"/>
        <explicitInput target="pop1[0]" input=")" + input_id + R"("/>
)", declarations));
    parse_model(context, directory.write("LEMS.xml", recorded_simulation("net.nml")));
    const Codegen compiler(context, nullptr);
    return compiler.flatten_input_tree(*context.find_instance(input_id));
}

Vector<s64> ticks_with_spikes(const InputSpikeTrain &train) {
    Vector<s64> ticks;
    for (usize tick = 0; tick < train.spike_counts.size(); tick += 1) {
        if (train.spike_counts[tick] != 0) ticks.push_back((s64)tick);
    }
    return ticks;
}

} // namespace

// A spikeArray's spikes become one count per tick of the run, delivered to the spikeArray itself.
TEST(InputTree, a_spike_array_becomes_a_count_per_tick) {
    const TemporaryDirectory directory;
    NML_Context context;
    const InputTree tree = input_tree_of(context, directory, R"(    <spikeArray id="train0">
        <spike id="0" time="30ms"/>
        <spike id="1" time="10ms"/>
        <spike id="2" time="20ms"/>
    </spikeArray>
)", "train0");

    ASSERT_EQ(tree.parts.size(), 1u);
    EXPECT_EQ(tree.parts[0].instance->id, "train0");
    EXPECT_EQ(tree.state_size, (s64)type_named(context, "spikeArray").state_variable_names.size());
    EXPECT_TRUE(tree.routes.empty());
    ASSERT_EQ(tree.spike_trains.size(), 1u);
    EXPECT_EQ(tree.spike_trains[0].receiver, 0);
    // 100 ms at 0.01 ms.
    ASSERT_EQ((s64)tree.spike_trains[0].spike_counts.size(), context.simulation.total_tick_count);
    EXPECT_EQ(ticks_with_spikes(tree.spike_trains[0]), (Vector<s64>{1000, 2000, 3000}));
}

// A timedSynapticInput instantiates its own synapse, its spikes go to it, and it sends its own
// events into the synapse. Two spike times that round to the same tick count twice.
TEST(InputTree, a_timed_synaptic_input_routes_its_spikes_into_its_own_synapse) {
    const TemporaryDirectory directory;
    NML_Context context;
    const InputTree tree = input_tree_of(context, directory, R"(    <timedSynapticInput id="timed0" synapse="syn0" spikeTarget="./syn0">
        <spike id="0" time="5ms"/>
        <spike id="1" time="5.001ms"/>
        <spike id="2" time="7ms"/>
    </timedSynapticInput>
)", "timed0");

    ASSERT_EQ(tree.parts.size(), 2u);
    EXPECT_EQ(tree.parts[1].instance->id, "syn0");
    EXPECT_EQ(tree.parts[1].parent, 0);
    EXPECT_EQ(tree.parts[1].role, "synapse");
    const s64 own_state = (s64)type_named(context, "timedSynapticInput").state_variable_names.size();
    EXPECT_EQ(tree.parts[1].first_state_slot, own_state);
    EXPECT_EQ(tree.state_size, own_state + (s64)type_named(context, "expOneSynapse").state_variable_names.size());

    ASSERT_EQ(tree.spike_trains.size(), 1u);
    EXPECT_EQ(tree.spike_trains[0].receiver, 0);
    EXPECT_EQ(ticks_with_spikes(tree.spike_trains[0]), (Vector<s64>{500, 700}));
    EXPECT_EQ(tree.spike_trains[0].spike_counts[500], 2);

    ASSERT_EQ(tree.routes.size(), 1u);
    EXPECT_EQ(tree.routes[0].source, 0);
    EXPECT_EQ(tree.routes[0].target, 1);
}

TEST(InputTree, a_poisson_firing_synapse_routes_its_spikes_into_its_own_synapse) {
    const TemporaryDirectory directory;
    NML_Context context;
    const InputTree tree = input_tree_of(context, directory, R"(    <poissonFiringSynapse id="poisson0" averageRate="50Hz" synapse="syn0" spikeTarget="./syn0"/>
)", "poisson0");

    ASSERT_EQ(tree.parts.size(), 2u);
    EXPECT_EQ(tree.parts[1].role, "synapse");
    EXPECT_TRUE(tree.spike_trains.empty());
    ASSERT_EQ(tree.routes.size(), 1u);
    EXPECT_EQ(tree.routes[0].source, 0);
    EXPECT_EQ(tree.routes[0].target, 1);
}

// Each generator in a compoundInput is a part of its own, which the compound sums through its select.
TEST(InputTree, a_compound_input_holds_each_generator_as_a_part) {
    const TemporaryDirectory directory;
    NML_Context context;
    const InputTree tree = input_tree_of(context, directory, R"(    <compoundInput id="compound0">
        <pulseGenerator id="pulse" delay="1ms" duration="2ms" amplitude="1nA"/>
        <sineGenerator id="sine" delay="0ms" phase="0" duration="10ms" amplitude="1nA" period="5ms"/>
    </compoundInput>
)", "compound0");

    ASSERT_EQ(tree.parts.size(), 3u);
    for (usize part = 1; part < 3; part += 1) {
        EXPECT_EQ(tree.parts[part].parent, 0);
        EXPECT_EQ(tree.parts[part].role, "currents");
    }
    EXPECT_EQ(tree.parts[1].instance->component_type->name, "pulseGenerator");
    EXPECT_EQ(tree.parts[2].instance->component_type->name, "sineGenerator");
    // A child fills the Children slot because its type extends the slot's.
    EXPECT_TRUE(context.is_instance_of(tree.parts[1].instance, "basePointCurrent"));
    EXPECT_FALSE(context.is_instance_of(tree.parts[1].instance, "baseSpikeSource"));
    EXPECT_TRUE(tree.routes.empty());
    EXPECT_TRUE(tree.spike_trains.empty());
}

// Selections hold the cell-memory index of what they record.
TEST(Simulation, recordings_resolve_their_selections) {
    const TwoPopulationModel model;
    const Vector<RecordingConfig> &profiles = model.context.simulation.recording_profiles;
    ASSERT_EQ(profiles.size(), 2u);

    const RecordingConfig &columns = profiles[0];
    EXPECT_EQ(columns.output_filenames, (Vector<String>{"out.dat"}));
    EXPECT_EQ(columns.file_output_format[0], OutputFileFormat::NML_STANDARD);
    EXPECT_EQ(columns.recordings_count, 2);
    ASSERT_EQ(columns.selections.size(), 2u);
    EXPECT_EQ(columns.selections[0].quantity_path, "pop1[0]/v");
    EXPECT_EQ(columns.selections[0].neuron_index, 0);
    EXPECT_EQ(columns.selections[1].quantity_path, "pop2[1]/v");
    EXPECT_EQ(columns.selections[1].neuron_index, model.izhikevich_memory_index(1, "v"));

    const RecordingConfig &events = profiles[1];
    EXPECT_EQ(events.output_filenames, (Vector<String>{"spikes.dat"}));
    EXPECT_EQ(events.file_output_format[0], OutputFileFormat::SPIKE_EVENTS);
    ASSERT_EQ(events.selections.size(), 1u);
    EXPECT_EQ(events.selections[0].neuron_index, 5);
    EXPECT_EQ(events.selections[0].event_port, "spike");
}

// An excitatory/inhibitory pair built from one cell type, and two projections with one synapse
// type: each keeps its own instance and values.
TEST(Simulation, instances_of_one_type_keep_their_own_values) {
    const TemporaryDirectory directory;
    directory.write("net.nml", R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="sharednet">
    <iafCell id="cellA" leakReversal="-50mV" thresh="-55mV" reset="-70mV" C="0.2nF" leakConductance="0.01uS"/>
    <iafCell id="cellB" leakReversal="-90mV" thresh="-55mV" reset="-70mV" C="0.2nF" leakConductance="0.01uS"/>
    <expOneSynapse id="synFast" gbase="0.5nS" erev="0mV" tauDecay="1ms"/>
    <expOneSynapse id="synSlow" gbase="0.5nS" erev="0mV" tauDecay="50ms"/>
    <network id="net1">
        <population id="pop1" component="cellA" size="2"/>
        <population id="pop2" component="cellB" size="2"/>
        <projection id="projFast" presynapticPopulation="pop1" postsynapticPopulation="pop2" synapse="synFast">
            <connection id="0" preCellId="../pop1[0]" postCellId="../pop2[0]"/>
        </projection>
        <projection id="projSlow" presynapticPopulation="pop1" postsynapticPopulation="pop2" synapse="synSlow">
            <connection id="0" preCellId="../pop1[0]" postCellId="../pop2[1]"/>
        </projection>
    </network>
</neuroml>
)");
    NML_Context context;
    parse_model(context, directory.write("LEMS.xml", recorded_simulation("net.nml")));
    const NML_Context::NML_SimulationContext &simulation = context.simulation;

    ASSERT_EQ(simulation.cell_instances.size(), 2u);
    EXPECT_DOUBLE_EQ(context.resolve_quantity(simulation.cell_instances[0].value_or("leakReversal")), -0.05);
    EXPECT_DOUBLE_EQ(context.resolve_quantity(simulation.cell_instances[1].value_or("leakReversal")), -0.09);

    ASSERT_EQ(simulation.synapse_instances.size(), 2u);
    EXPECT_EQ(simulation.synapse_instances[0].id, "synFast");
    EXPECT_EQ(simulation.synapse_instances[1].id, "synSlow");
    ASSERT_EQ(simulation.network_data.list[0].size(), 2u);
    EXPECT_EQ(simulation.network_data.list[0][0].component_id, "synFast");
    EXPECT_EQ(simulation.network_data.list[0][1].component_id, "synSlow");
}

// A population of type populationList takes its size from its <instance> children, and its cells
// are addressed as "../pop1/0/cell0".
TEST(Simulation, a_population_list_is_sized_and_addressed_by_its_instances) {
    const TemporaryDirectory directory;
    directory.write("net.nml", one_population_network(R"(        <population id="pop1" component="cell0" type="populationList" size="3">
            <instance id="0"><location x="0" y="0" z="0"/></instance>
            <instance id="1"><location x="10" y="0" z="0"/></instance>
            <instance id="2"><location x="20" y="0" z="0"/></instance>
        </population>
        <projection id="proj1" presynapticPopulation="pop1" postsynapticPopulation="pop1" synapse="syn0">
            <connection id="0" preCellId="../pop1/0/cell0" postCellId="../pop1/2/cell0"/>
        </projection>
)"));
    NML_Context context;
    parse_model(context, directory.write("LEMS.xml", recorded_simulation("net.nml")));

    EXPECT_EQ(context.simulation.total_neuron_count, 3);
    ASSERT_EQ(context.simulation.network_data.list[0].size(), 1u);
    EXPECT_EQ(context.simulation.network_data.list[0][0].child, 2);
}

// Two contexts parsing one model agree on everything indices depend on, and a context parsing it
// twice does not double it.
TEST(Simulation, parsing_is_deterministic_and_repeatable) {
    const TemporaryDirectory directory;
    directory.write("net.nml", two_population_network());
    const String main_file = directory.write("LEMS.xml", recorded_simulation("net.nml"));

    NML_Context first;
    parse_model(first, main_file);
    NML_Context second;
    parse_model(second, main_file);
    parse_model(second, main_file);

    for (const NML_Context *context : {&first, &second}) {
        EXPECT_EQ(context->simulation.total_neuron_count, 5);
        EXPECT_EQ(context->simulation.total_edge_count, 2);
        EXPECT_EQ(context->simulation.input_profiles.size(), 2u);
        EXPECT_EQ(context->simulation.recording_profiles.size(), 2u);
        EXPECT_EQ(context->simulation.cell_instances.size(), 2u);
    }
    EXPECT_EQ(first.simulation.population_base_indices, second.simulation.population_base_indices);
    for (s64 neuron = 0; neuron < 5; neuron += 1) {
        const Vector<NML_NetworkEdge> &first_edges = first.simulation.network_data.list[(usize)neuron];
        const Vector<NML_NetworkEdge> &second_edges = second.simulation.network_data.list[(usize)neuron];
        ASSERT_EQ(first_edges.size(), second_edges.size()) << "neuron " << neuron;
        for (usize index = 0; index < first_edges.size(); index += 1) {
            EXPECT_EQ(first_edges[index].child, second_edges[index].child);
            EXPECT_EQ(first_edges[index].delay_ticks, second_edges[index].delay_ticks);
        }
    }
}

// ── errors ──────────────────────────────────────────────────────────────────────

TEST(Errors, a_connection_to_an_unknown_population_throws_naming_it) {
    const TemporaryDirectory directory;
    directory.write("net.nml", one_population_network(R"(        <population id="pop1" component="cell0" size="2"/>
        <projection id="proj1" presynapticPopulation="pop1" postsynapticPopulation="pop1" synapse="syn0">
            <connection id="0" preCellId="../pop1[0]" postCellId="../nosuchpop[0]"/>
        </projection>
)"));
    EXPECT_NE(parse_error(directory.write("LEMS.xml", recorded_simulation("net.nml"))).find("nosuchpop"),
              String::npos);
}

TEST(Errors, a_cell_index_past_the_population_throws) {
    const TemporaryDirectory directory;
    directory.write("net.nml", one_population_network(R"(        <population id="pop1" component="cell0" size="2"/>
        <projection id="proj1" presynapticPopulation="pop1" postsynapticPopulation="pop1" synapse="syn0">
            <connection id="0" preCellId="../pop1[0]" postCellId="../pop1[7]"/>
        </projection>
)"));
    EXPECT_NE(parse_error(directory.write("LEMS.xml", recorded_simulation("net.nml"))).find("pop1[7]"),
              String::npos);
}

TEST(Errors, a_population_of_an_unknown_component_throws_naming_it) {
    const TemporaryDirectory directory;
    directory.write("net.nml", one_population_network(R"(        <population id="pop1" component="no_such_cell" size="2"/>
)"));
    EXPECT_NE(parse_error(directory.write("LEMS.xml", recorded_simulation("net.nml"))).find("no_such_cell"),
              String::npos);
}

TEST(Errors, a_projection_through_an_unknown_synapse_throws_naming_it) {
    const TemporaryDirectory directory;
    directory.write("net.nml", one_population_network(R"(        <population id="pop1" component="cell0" size="2"/>
        <projection id="proj1" presynapticPopulation="pop1" postsynapticPopulation="pop1" synapse="no_such_synapse">
            <connection id="0" preCellId="../pop1[0]" postCellId="../pop1[1]"/>
        </projection>
)"));
    EXPECT_NE(parse_error(directory.write("LEMS.xml", recorded_simulation("net.nml"))).find("no_such_synapse"),
              String::npos);
}

TEST(Errors, an_input_to_no_neuron_throws) {
    const TemporaryDirectory directory;
    directory.write("net.nml", one_population_network(R"(        <population id="pop1" component="cell0" size="2"/>
        <explicitInput target="pop1[5]" input="pg0"/>
)", "    <pulseGenerator id=\"pg0\" delay=\"0ms\" duration=\"10ms\" amplitude=\"1nA\"/>\n"));
    EXPECT_NE(parse_error(directory.write("LEMS.xml", recorded_simulation("net.nml"))).find("pop1[5]"),
              String::npos);
}

// Every time in the model is divided by the step, so a step of zero is refused at parse time.
TEST(Errors, a_simulation_without_a_usable_step_throws) {
    const TemporaryDirectory directory;
    directory.write("net.nml", one_population_network(R"(        <population id="pop1" component="cell0" size="2"/>
)"));
    EXPECT_NE(parse_error(directory.write("LEMS.xml", recorded_simulation("net.nml", "0ms"))).find("no usable step"),
              String::npos);
}

TEST(Errors, a_target_naming_no_instance_throws) {
    const TemporaryDirectory directory;
    const String main_file = directory.write("LEMS.xml", R"(<Lems><Target component="no_such_simulation"/></Lems>)");
    EXPECT_NE(parse_error(main_file).find("no_such_simulation"), String::npos);
}

// ── schema validation ───────────────────────────────────────────────────────────

TEST(Schema, a_valid_neuroml_document_passes_and_an_unknown_element_fails_naming_it) {
    const TemporaryDirectory directory;
    const String valid = directory.write("valid.nml", two_population_network());
    const String invalid = directory.write("invalid.nml", R"(<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="bad">
    <thisTagDoesNotExistInSchema id="x"/>
</neuroml>
)");
    const String lems = directory.write("LEMS.xml", recorded_simulation("valid.nml"));

    NML_Context context;
    EXPECT_TRUE(context.validate_lems_schema(valid)) << context.last_schema_validation_errors;
    EXPECT_FALSE(context.validate_lems_schema(invalid));
    EXPECT_NE(context.last_schema_validation_errors.find("thisTagDoesNotExistInSchema"), String::npos);
    // A LEMS document is not a NeuroML document; there is no schema to check it against.
    EXPECT_TRUE(context.validate_lems_schema(lems));
}
