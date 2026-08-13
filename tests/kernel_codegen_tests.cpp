#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "spikecorec/nml/kernel_codegen.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

namespace {

// ── Expression fixtures ──────────────────────────────────────────────────────
// Every symbol reads back as "read_<name>", so an expected string shows the shape of the
// parse without any surrounding storage detail.
SymbolTable make_expression_symbol_table() {
    SymbolTable symbols;
    symbols.component_type_name = "testCell";
    symbols.define("alpha", "read_alpha");
    symbols.define("beta", "read_beta");
    symbols.define("gamma", "read_gamma");
    // Contains the letters of a dotted operator without the dots. A textual substitution
    // pass would corrupt this name; a tokenizer cannot.
    symbols.define("rate_eq_constant", "read_rate_eq_constant");
    return symbols;
}

String translate(const String &nml_expression) {
    return translate_expression(nml_expression, make_expression_symbol_table());
}

// ── Model fixtures ───────────────────────────────────────────────────────────

DynamicsInstruction make_instruction(DynamicsStage stage, NML_DeclarationType source_tag,
                                     const String &target, const String &expression,
                                     const String &condition = "") {
    DynamicsInstruction instruction(stage, source_tag);
    instruction.target = target;
    instruction.expression = expression;
    instruction.condition = condition;
    return instruction;
}

// A leaky integrate-and-fire cell: one state variable, forward-Euler decay, a threshold
// condition that resets and emits, and an OnStart that seeds the membrane potential.
CellTypeSpecification make_integrate_and_fire_cell_type() {
    CellTypeSpecification cell_type;
    cell_type.name = "iafCell";
    cell_type.state_variable_names = {"v"};
    cell_type.parameter_names = {"leakConductance", "leakReversal", "capacitance", "thresh",
                                 "reset"};

    const String threshold_test = "v .gt. thresh";

    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "v",
            "(leakConductance * (leakReversal - v)) / capacitance"));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Detect, NML_DeclarationType::OnCondition, "", threshold_test));
    cell_type.dynamics.push_back(make_instruction(DynamicsStage::Reset,
                                                  NML_DeclarationType::StateAssignment, "v",
                                                  "reset", threshold_test));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Emit, NML_DeclarationType::EventOut, "spike", "", threshold_test));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Initialize, NML_DeclarationType::StateAssignment, "v", "leakReversal"));

    return cell_type;
}

// Two state variables that reference each other, so the generated code has to compute both
// derivatives before writing either back. Also carries a DerivedVariable that a later
// TimeDerivative consumes.
CellTypeSpecification make_izhikevich_cell_type() {
    CellTypeSpecification cell_type;
    cell_type.name = "izhikevichCell";
    cell_type.state_variable_names = {"v", "u"};
    cell_type.parameter_names = {"a", "b", "c", "d", "thresh"};

    const String threshold_test = "v .geq. thresh";

    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::DerivedVariable, "vsquared", "v * v"));
    cell_type.dynamics.push_back(make_instruction(DynamicsStage::Integrate,
                                                  NML_DeclarationType::TimeDerivative, "v",
                                                  "0.04 * vsquared + 5 * v + 140 - u"));
    cell_type.dynamics.push_back(make_instruction(DynamicsStage::Integrate,
                                                  NML_DeclarationType::TimeDerivative, "u",
                                                  "a * (b * v - u)"));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Detect, NML_DeclarationType::OnCondition, "", threshold_test));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Reset, NML_DeclarationType::StateAssignment, "v", "c", threshold_test));
    cell_type.dynamics.push_back(make_instruction(DynamicsStage::Reset,
                                                  NML_DeclarationType::StateAssignment, "u",
                                                  "u + d", threshold_test));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Emit, NML_DeclarationType::EventOut, "spike", "", threshold_test));

    return cell_type;
}

// A cell that drains the synaptic accumulator, shaped like the standard library's iafCell:
// a `select=` DerivedVariable whose bound name a later TimeDerivative consumes.
CellTypeSpecification make_synaptic_input_cell_type(const String &select_path) {
    CellTypeSpecification cell_type;
    cell_type.name = "iafCell";
    cell_type.state_variable_names = {"v"};
    cell_type.parameter_names = {"leakConductance", "leakReversal", "capacitance", "thresh",
                                 "reset"};

    const String threshold_test = "v .gt. thresh";

    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::DerivedVariable, "iSyn", select_path));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "v",
            "(leakConductance * (leakReversal - v) + iSyn) / capacitance"));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Detect, NML_DeclarationType::OnCondition, "", threshold_test));
    cell_type.dynamics.push_back(make_instruction(DynamicsStage::Reset,
                                                  NML_DeclarationType::StateAssignment, "v",
                                                  "reset", threshold_test));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Emit, NML_DeclarationType::EventOut, "spike", "", threshold_test));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Initialize, NML_DeclarationType::StateAssignment, "v", "leakReversal"));

    return cell_type;
}

// One cell type whose only DerivedVariable carries `select_path`, for the paths that are
// expected to be refused.
NML_ParseResult make_select_path_model(const String &select_path) {
    CellTypeSpecification cell_type;
    cell_type.name = "selectCell";
    cell_type.state_variable_names = {"v"};
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::DerivedVariable, "quantity",
            select_path));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "v", "quantity"));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);
    return parse_result;
}

NML_ParseResult make_two_cell_type_model() {
    NML_ParseResult parse_result;
    parse_result.step_dt = 1e-5;
    parse_result.cell_types.push_back(make_integrate_and_fire_cell_type());
    parse_result.cell_types.push_back(make_izhikevich_cell_type());
    return parse_result;
}

// A cell whose threshold is a DerivedVariable OF the state variable being integrated, and
// whose one handler swaps its two state variables. Both halves of the threshold comparison
// therefore move within a tick, and the handler's two assignments read what the other writes
// -- which is what separates a per-statement lowering from LEMS's own semantics.
CellTypeSpecification make_derived_threshold_cell_type() {
    CellTypeSpecification cell_type;
    cell_type.name = "derivedThresholdCell";
    cell_type.state_variable_names = {"v", "u"};
    cell_type.parameter_names = {"tau", "thresholdScale"};

    const String threshold_test = "v .gt. scaledThreshold";

    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::DerivedVariable, "scaledThreshold",
            "thresholdScale * v"));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "v", "0 - v / tau"));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Detect, NML_DeclarationType::OnCondition, "", threshold_test));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Reset, NML_DeclarationType::StateAssignment, "v", "u", threshold_test));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Reset, NML_DeclarationType::StateAssignment, "u", "v", threshold_test));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Emit, NML_DeclarationType::EventOut, "spike", "", threshold_test));

    return cell_type;
}

NML_ParseResult make_derived_threshold_model() {
    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(make_derived_threshold_cell_type());
    return parse_result;
}

// ── Synapse fixtures ─────────────────────────────────────────────────────────
//
// alphaCurrentSynapse, verbatim from third_party/neuroml2/std_lib/Synapses.xml: two coupled
// state variables integrated every tick, an OnEvent on port "in" bumping one of them by
// weight * ibase, and a DerivedVariable `i` exposing the other as the delivered current.
SynapseTypeSpecification make_alpha_current_synapse_type() {
    SynapseTypeSpecification synapse_type;
    synapse_type.name = "alphaCurrentSynapse";
    synapse_type.state_variable_names = {"I", "J"};
    synapse_type.parameter_names = {"weight", "tau", "ibase"};

    synapse_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::DerivedVariable, "i", "I"));
    synapse_type.dynamics.push_back(
            make_instruction(DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "I",
                             "(2.7182818284590451*J - I)/tau"));
    synapse_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "J", "-J/tau"));
    synapse_type.dynamics.push_back(make_instruction(
            DynamicsStage::Initialize, NML_DeclarationType::StateAssignment, "I", "0"));
    synapse_type.dynamics.push_back(make_instruction(
            DynamicsStage::Initialize, NML_DeclarationType::StateAssignment, "J", "0"));
    // An OnEvent handler's `condition` carries the port it hangs off, which is how the
    // arrival body is found again after the dynamics are flattened.
    synapse_type.dynamics.push_back(
            make_instruction(DynamicsStage::Arrival, NML_DeclarationType::StateAssignment, "J",
                             "J + weight * ibase", "in"));

    return synapse_type;
}

// expOneSynapse: conductance-based, so it declares erev and gbase and computes
// i = g * (erev - v) against a postsynaptic voltage it does not own.
SynapseTypeSpecification make_conductance_synapse_type() {
    SynapseTypeSpecification synapse_type;
    synapse_type.name = "expOneSynapse";
    synapse_type.state_variable_names = {"g"};
    synapse_type.parameter_names = {"weight", "gbase", "erev", "tauDecay"};

    synapse_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::DerivedVariable, "i", "g * (erev - v)"));
    synapse_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "g", "-g / tauDecay"));
    synapse_type.dynamics.push_back(
            make_instruction(DynamicsStage::Arrival, NML_DeclarationType::StateAssignment, "g",
                             "g + (weight * gbase)", "in"));

    return synapse_type;
}

ComponentPrototype make_synapse_prototype(const String &instance_id, s64 type_index,
                                          const Vector<f64> &parameter_values) {
    ComponentPrototype prototype;
    prototype.instance_id = instance_id;
    prototype.type_index = type_index;
    for (const f64 parameter_value : parameter_values) {
        Real resolved;
        resolved.float64 = parameter_value;
        prototype.starting_parameters.push_back(resolved);
    }
    return prototype;
}

// Two neurons, one edge, delivering through `synapse_prototype_index` -- or through nothing
// when that is -1, which is what makes a prototype unwired.
void wire_one_edge(NML_ParseResult &parse_result, s64 synapse_prototype_index) {
    parse_result.neurons.clear();

    Neuron source_neuron;
    NetworkEdge edge;
    edge.target_neuron_index = 1;
    edge.synapse_prototype_index = synapse_prototype_index;
    source_neuron.outgoing_edges.push_back(edge);

    parse_result.neurons.push_back(source_neuron);
    parse_result.neurons.push_back(Neuron{});
}

// One synaptic-input cell, one alphaCurrentSynapse prototype, one edge through it.
NML_ParseResult make_alpha_synapse_model() {
    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(make_synaptic_input_cell_type("synapses[*]/i"));
    parse_result.synapse_types.push_back(make_alpha_current_synapse_type());
    parse_result.synapse_prototypes.push_back(
            make_synapse_prototype("alphaSyn", 0, {1.0, 2.0e-3, 1.0e-9}));
    wire_one_edge(parse_result, 0);
    return parse_result;
}

// ── Metal compilation ────────────────────────────────────────────────────────
// Guarded on the Metal build rather than on the host OS: shelling out to xcrun from a CUDA
// build is exactly the defect tracked as issue #117.
#ifdef SPIKECOREC_METAL

bool metal_compiler_is_available() {
    return system("xcrun -sdk macosx metal --version > /dev/null 2>&1") == 0;
}

// Runs the real shader compiler over `source`. Codegen whose output the shader compiler
// rejects is worthless, and only an end-to-end compile catches it.
bool compile_as_metal(const String &source, const String &case_name, String &compiler_output) {
    const filesystem::path directory =
            filesystem::temp_directory_path() / "spikecorec_kernel_codegen_tests" / case_name;
    filesystem::remove_all(directory);
    filesystem::create_directories(directory);

    const filesystem::path source_path = directory / "generated.metal";
    const filesystem::path object_path = directory / "generated.air";
    const filesystem::path log_path = directory / "compiler.log";

    {
        ofstream source_stream(source_path);
        source_stream << source;
    }

    const String command = "xcrun -sdk macosx metal -c '" + source_path.string() + "' -o '" +
                           object_path.string() + "' > '" + log_path.string() + "' 2>&1";
    const int exit_code = system(command.c_str());

    ifstream log_stream(log_path);
    compiler_output.assign(istreambuf_iterator<char>(log_stream), istreambuf_iterator<char>());

    return exit_code == 0;
}

#endif

} // namespace

// ── Operator precedence and associativity ────────────────────────────────────

TEST(KernelCodegenExpression, MultiplicationBindsTighterThanAddition) {
    EXPECT_EQ(translate("alpha + beta * gamma"), "(read_alpha + (read_beta * read_gamma))");
    EXPECT_EQ(translate("(alpha + beta) * gamma"), "((read_alpha + read_beta) * read_gamma)");
}

TEST(KernelCodegenExpression, SubtractionIsLeftAssociative) {
    EXPECT_EQ(translate("alpha - beta - gamma"), "((read_alpha - read_beta) - read_gamma)");
}

TEST(KernelCodegenExpression, CaretBecomesPowAndIsRightAssociative) {
    EXPECT_EQ(translate("alpha ^ beta"), "pow(read_alpha, read_beta)");
    EXPECT_EQ(translate("alpha ^ beta ^ gamma"),
              "pow(read_alpha, pow(read_beta, read_gamma))");
}

TEST(KernelCodegenExpression, CaretBindsTighterThanUnaryMinusAndMultiplication) {
    EXPECT_EQ(translate("-alpha ^ 2"), "(-pow(read_alpha, 2.0f))");
    EXPECT_EQ(translate("beta * alpha ^ 2"), "(read_beta * pow(read_alpha, 2.0f))");
    EXPECT_EQ(translate("2 ^ -1"), "pow(2.0f, (-1.0f))");
}

TEST(KernelCodegenExpression, UnaryMinusNestsAndUnaryPlusIsDropped) {
    EXPECT_EQ(translate("-alpha"), "(-read_alpha)");
    EXPECT_EQ(translate("alpha * -beta"), "(read_alpha * (-read_beta))");
    EXPECT_EQ(translate("- -alpha"), "(-(-read_alpha))");
    EXPECT_EQ(translate("+alpha"), "read_alpha");
}

TEST(KernelCodegenExpression, LogicalOperatorsBindLooserThanComparisons) {
    EXPECT_EQ(translate("alpha .gt. beta .and. beta .gt. gamma"),
              "((read_alpha > read_beta) && (read_beta > read_gamma))");
    EXPECT_EQ(translate("alpha .gt. beta .or. beta .gt. gamma .and. alpha .lt. gamma"),
              "((read_alpha > read_beta) || ((read_beta > read_gamma) && "
              "(read_alpha < read_gamma)))");
}

// ── Dotted operators ─────────────────────────────────────────────────────────

TEST(KernelCodegenExpression, EveryDottedOperatorTranslates) {
    EXPECT_EQ(translate("alpha .gt. beta"), "(read_alpha > read_beta)");
    EXPECT_EQ(translate("alpha .lt. beta"), "(read_alpha < read_beta)");
    EXPECT_EQ(translate("alpha .geq. beta"), "(read_alpha >= read_beta)");
    EXPECT_EQ(translate("alpha .leq. beta"), "(read_alpha <= read_beta)");
    EXPECT_EQ(translate("alpha .eq. beta"), "(read_alpha == read_beta)");
    EXPECT_EQ(translate("alpha .neq. beta"), "(read_alpha != read_beta)");
    EXPECT_EQ(translate("alpha .and. beta"), "(read_alpha && read_beta)");
    EXPECT_EQ(translate("alpha .or. beta"), "(read_alpha || read_beta)");
}

TEST(KernelCodegenExpression, BareComparisonOperatorsPassThrough) {
    EXPECT_EQ(translate("alpha > beta"), "(read_alpha > read_beta)");
    EXPECT_EQ(translate("alpha < beta"), "(read_alpha < read_beta)");
    EXPECT_EQ(translate("alpha >= beta"), "(read_alpha >= read_beta)");
    EXPECT_EQ(translate("alpha <= beta"), "(read_alpha <= read_beta)");
    EXPECT_EQ(translate("alpha == beta"), "(read_alpha == read_beta)");
    EXPECT_EQ(translate("alpha != beta"), "(read_alpha != read_beta)");
}

TEST(KernelCodegenExpression, DottedOperatorsNeedNoSurroundingWhitespace) {
    EXPECT_EQ(translate("alpha.eq.beta"), "(read_alpha == read_beta)");
    EXPECT_EQ(translate("alpha.geq.beta"), "(read_alpha >= read_beta)");
}

TEST(KernelCodegenExpression, OperatorLettersInsideAnIdentifierAreNotAnOperator) {
    EXPECT_EQ(translate("rate_eq_constant"), "read_rate_eq_constant");
    EXPECT_EQ(translate("rate_eq_constant .gt. alpha"),
              "(read_rate_eq_constant > read_alpha)");
}

TEST(KernelCodegenExpression, NumberFollowedByDottedOperatorSplitsCorrectly) {
    // The '.' opening ".gt." must not be eaten as the number's decimal point.
    EXPECT_EQ(translate("0.gt.1"), "(0.0f > 1.0f)");
    EXPECT_EQ(translate("alpha .gt. 0"), "(read_alpha > 0.0f)");
}

TEST(KernelCodegenExpression, UnknownDottedOperatorThrows) {
    EXPECT_THROW(translate("alpha .xor. beta"), runtime_error);
}

// ── Functions ────────────────────────────────────────────────────────────────

TEST(KernelCodegenExpression, StandardLibraryFunctionsMapToCNames) {
    EXPECT_EQ(translate("exp(alpha)"), "exp(read_alpha)");
    EXPECT_EQ(translate("sqrt(alpha)"), "sqrt(read_alpha)");
    EXPECT_EQ(translate("tanh(alpha)"), "tanh(read_alpha)");
    EXPECT_EQ(translate("abs(alpha)"), "fabs(read_alpha)");
    EXPECT_EQ(translate("ceil(alpha)"), "ceil(read_alpha)");
    EXPECT_EQ(translate("floor(alpha)"), "floor(read_alpha)");
}

TEST(KernelCodegenExpression, NaturalAndBaseTenLogarithmsAreNotSwapped) {
    // LEMS `ln` is C `log`; LEMS `log` is base 10, which is C `log10`.
    EXPECT_EQ(translate("ln(alpha)"), "log(read_alpha)");
    EXPECT_EQ(translate("log(alpha)"), "log10(read_alpha)");
}

TEST(KernelCodegenExpression, HeavisideBecomesAConditional) {
    EXPECT_EQ(translate("H(alpha)"), "((read_alpha) >= 0.0f ? 1.0f : 0.0f)");
    EXPECT_EQ(translate("H(alpha - beta)"),
              "(((read_alpha - read_beta)) >= 0.0f ? 1.0f : 0.0f)");
}

TEST(KernelCodegenExpression, CallsNest) {
    EXPECT_EQ(translate("exp(sin(alpha) + 1)"), "exp((sin(read_alpha) + 1.0f))");
    EXPECT_EQ(translate("exp(-alpha / beta)"), "exp(((-read_alpha) / read_beta))");
    EXPECT_EQ(translate("sqrt(exp(ln(alpha)))"), "sqrt(exp(log(read_alpha)))");
}

TEST(KernelCodegenExpression, UnknownFunctionThrows) {
    EXPECT_THROW(translate("bogus_function(alpha)"), runtime_error);
}

TEST(KernelCodegenExpression, RandomIsReportedAsUnsupported) {
    try {
        translate("random(alpha)");
        FAIL() << "expected random() to be rejected";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("random"), String::npos);
        EXPECT_NE(message.find("testCell"), String::npos);
    }
}

TEST(KernelCodegenExpression, WrongArgumentCountThrows) {
    EXPECT_THROW(translate("exp(alpha, beta)"), runtime_error);
    EXPECT_THROW(translate("exp()"), runtime_error);
}

// ── Literals ─────────────────────────────────────────────────────────────────

TEST(KernelCodegenExpression, NumbersBecomeFloatLiterals) {
    EXPECT_EQ(translate("1"), "1.0f");
    EXPECT_EQ(translate("0.5"), "0.5f");
    EXPECT_EQ(translate(".5"), ".5f");
    EXPECT_EQ(translate("1e-3"), "1e-3f");
    EXPECT_EQ(translate("2.5E6"), "2.5E6f");
    EXPECT_EQ(translate("140"), "140.0f");
}

// ── Malformed input ──────────────────────────────────────────────────────────

TEST(KernelCodegenExpression, MalformedExpressionsThrow) {
    EXPECT_THROW(translate(""), runtime_error);
    EXPECT_THROW(translate("alpha +"), runtime_error);
    EXPECT_THROW(translate("alpha beta"), runtime_error);
    EXPECT_THROW(translate("(alpha + beta"), runtime_error);
    EXPECT_THROW(translate("alpha $ beta"), runtime_error);
    // A select= path lands in `expression`; it must be reported, not passed through.
    EXPECT_THROW(translate("synapses[*]/i"), runtime_error);
}

TEST(KernelCodegenExpression, UnknownIdentifierThrowsNamingItAndItsComponentType) {
    try {
        translate("alpha + mystery_variable");
        FAIL() << "expected an unknown identifier to be rejected";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("mystery_variable"), String::npos);
        EXPECT_NE(message.find("testCell"), String::npos);
    }
}

// ── Symbol resolution precedence ─────────────────────────────────────────────

TEST(KernelCodegenSymbolTable, FirstDefinitionWins) {
    SymbolTable symbols;
    symbols.component_type_name = "testCell";
    symbols.define("tau", "from_state");
    symbols.define("tau", "from_parameter");

    EXPECT_TRUE(symbols.contains("tau"));
    EXPECT_EQ(symbols.read_expression_for("tau"), "from_state");
    EXPECT_FALSE(symbols.contains("absent"));
    EXPECT_THROW(symbols.read_expression_for("absent"), runtime_error);
}

TEST(KernelCodegenSymbolResolution, EachCategoryResolvesInPrecedenceOrder) {
    CellTypeSpecification cell_type;
    cell_type.name = "precedenceCell";
    // "collide_state_parameter" is both a state variable and a parameter; the state
    // variable outranks it. "collide_parameter_constant" is both a parameter and a global
    // constant; the parameter outranks it.
    cell_type.state_variable_names = {"collide_state_parameter", "v"};
    cell_type.parameter_names = {"collide_state_parameter", "collide_parameter_constant"};

    cell_type.dynamics.push_back(
            make_instruction(DynamicsStage::Integrate, NML_DeclarationType::DerivedVariable,
                             "collide_derived_constant", "1"));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "v",
            "collide_state_parameter + collide_parameter_constant + collide_derived_constant"
            " + free_constant + t + dt"));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    Real seven;
    seven.float64 = 7.0;
    Real eight;
    eight.float64 = 8.0;
    Real nine;
    nine.float64 = 9.0;
    parse_result.global_constants["collide_parameter_constant"] = seven;
    parse_result.global_constants["collide_derived_constant"] = eight;
    parse_result.global_constants["free_constant"] = nine;

    const String source = generate_tick_kernel(parse_result).source;

    // State variable beats parameter: slot 0 of the state chunk, not of the parameters.
    EXPECT_NE(source.find("cell_state[state_base + 0]"), String::npos);
    EXPECT_EQ(source.find("cell_parameters[parameter_base + 0]"), String::npos);
    // Parameter beats global constant.
    EXPECT_NE(source.find("cell_parameters[parameter_base + 1]"), String::npos);
    EXPECT_EQ(source.find("7.000000000e+00f"), String::npos);
    // Derived local beats global constant.
    EXPECT_NE(source.find("derived_collide_derived_constant"), String::npos);
    EXPECT_EQ(source.find("8.000000000e+00f"), String::npos);
    // Global constant resolves when nothing above it claims the name.
    EXPECT_NE(source.find("9.000000000e+00f"), String::npos);
    // Built-ins sit at the bottom of the chain.
    EXPECT_NE(source.find("(dt * (float)tick)"), String::npos);
}

TEST(KernelCodegenSymbolResolution, ComponentTypeConstantsResolveByBareName) {
    CellTypeSpecification cell_type;
    cell_type.name = "constantCell";
    cell_type.state_variable_names = {"v"};
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "v", "MSEC"));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    // nml.cpp namespaces a ComponentType's own <Constant>s by their owning type, while the
    // expression writes the bare name.
    Real millisecond;
    millisecond.float64 = 0.001;
    parse_result.global_constants["constantCell.MSEC"] = millisecond;

    EXPECT_NE(generate_tick_kernel(parse_result).source.find("1.000000000e-03f"), String::npos);
}

TEST(KernelCodegenSymbolResolution, UnresolvableIdentifierInAModelThrows) {
    CellTypeSpecification cell_type;
    cell_type.name = "brokenCell";
    cell_type.state_variable_names = {"v"};
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "v", "v * nowhere"));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected an unresolvable identifier to be rejected";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("nowhere"), String::npos);
        EXPECT_NE(message.find("brokenCell"), String::npos);
    }
}

// ── Kernel assembly ──────────────────────────────────────────────────────────

TEST(KernelCodegenKernel, ArgumentNamesAreTheDocumentedBindingOrder) {
    const NML_ParseResult parse_result = make_two_cell_type_model();
    const Vector<String> expected_argument_names = {
        "cell_state",           "cell_parameters",       "network_inputs",
        "last_spiked",          "spike_flags",           "cell_state_base",
        "cell_parameter_base",  "cell_type_index",       "neuron_count",
        "dt",                   "tick",                  "internal_node_words",
        "leaf_node_words",      "rank_superblock_table", "rank_subblock_table",
        "U_matrix",             "V_matrix",              "edge_weight_coefficients",
        "edge_weight_deltas",   "edge_delay_ticks",      "branching_factor",
        "superblock_size_words","padded_node_count",     "tree_height",
        "internal_bit_count",   "rank_float4_stride",    "constant_weight",
        "max_neighbor_count",   "ring_depth",            "synapse_state",
        "edge_synapse_plane",
    };

    const GeneratedKernel tick_kernel = generate_tick_kernel(parse_result);
    EXPECT_EQ(tick_kernel.function_name, "simulate_tick");
    EXPECT_EQ(tick_kernel.argument_names, expected_argument_names);

    // The engine binds one argument set for both entry points.
    const GeneratedKernel initialize_kernel = generate_initialize_kernel(parse_result);
    EXPECT_EQ(initialize_kernel.function_name, "initialize_cell_state");
    EXPECT_EQ(initialize_kernel.argument_names, expected_argument_names);
}

TEST(KernelCodegenKernel, TwoCellTypesProduceTwoDeviceFunctionsAndTwoSwitchCases) {
    const String source = generate_tick_kernel(make_two_cell_type_model()).source;

    EXPECT_NE(source.find("cell_type_step_iafCell("), String::npos);
    EXPECT_NE(source.find("cell_type_step_izhikevichCell("), String::npos);

    const usize switch_position = source.find("switch (cell_type_index[neuron_index])");
    ASSERT_NE(switch_position, String::npos);

    const String switch_body = source.substr(switch_position);
    EXPECT_NE(switch_body.find("case 0:"), String::npos);
    EXPECT_NE(switch_body.find("case 1:"), String::npos);
    EXPECT_NE(switch_body.find("cell_type_step_iafCell("), String::npos);
    EXPECT_NE(switch_body.find("cell_type_step_izhikevichCell("), String::npos);

    // The switch dispatches; it does not inline the bodies. Each device function is
    // defined once, before the kernel.
    EXPECT_LT(source.find("cell_type_step_izhikevichCell("), switch_position);
}

TEST(KernelCodegenKernel, TimeDerivativesReadCurrentStateBeforeAnyWriteBack) {
    const String source = generate_tick_kernel(make_two_cell_type_model()).source;

    // Scoped to the izhikevich device function: iafCell also integrates a "v" in slot 0,
    // and an unscoped search would match its statements instead.
    const usize function_position = source.find("inline void cell_type_step_izhikevichCell(");
    ASSERT_NE(function_position, String::npos);
    const String function_body = source.substr(function_position);

    const usize next_v_position = function_body.find("float next_v =");
    const usize next_u_position = function_body.find("float next_u =");
    const usize write_v_position = function_body.find("cell_state[state_base + 0] = next_v;");
    const usize write_u_position = function_body.find("cell_state[state_base + 1] = next_u;");

    ASSERT_NE(next_v_position, String::npos);
    ASSERT_NE(next_u_position, String::npos);
    ASSERT_NE(write_v_position, String::npos);
    ASSERT_NE(write_u_position, String::npos);

    // u's derivative reads v, and v's derivative reads u. Both temporaries must be
    // computed before either write-back, or the second one integrates against a value
    // that has already moved.
    EXPECT_LT(next_u_position, write_v_position);
    EXPECT_LT(next_v_position, write_u_position);
}

TEST(KernelCodegenKernel, DerivedVariableBecomesALocalUsedByLaterInstructions) {
    const String source = generate_tick_kernel(make_two_cell_type_model()).source;

    const usize declaration_position = source.find("float derived_vsquared =");
    ASSERT_NE(declaration_position, String::npos);
    EXPECT_NE(source.find("(0.04f * derived_vsquared)"), String::npos);
    EXPECT_LT(declaration_position, source.find("(0.04f * derived_vsquared)"));
}

TEST(KernelCodegenKernel, ConditionGatesItsResetsAndEmit) {
    const String source = generate_tick_kernel(make_two_cell_type_model()).source;

    const usize condition_position =
            source.find("if ((cell_state[state_base + 0] > cell_parameters[parameter_base + 3]))");
    ASSERT_NE(condition_position, String::npos);

    // The reset and the spike both belong to that condition's block, which closes after
    // them.
    const String block = source.substr(condition_position);
    const usize reset_position = block.find("cell_state[state_base + 0] = cell_parameters");
    const usize spike_flag_position = block.find("spike_flags[neuron_index] = 1;");
    const usize last_spiked_position = block.find("last_spiked[neuron_index] = tick;");
    const usize block_end_position = block.find("\n    }\n");

    ASSERT_NE(block_end_position, String::npos);
    EXPECT_LT(reset_position, block_end_position);
    EXPECT_LT(spike_flag_position, block_end_position);
    EXPECT_LT(last_spiked_position, block_end_position);
}

TEST(KernelCodegenKernel, DerivedVariablesAreReevaluatedBeforeDetectReadsThem) {
    // Detect and Reset read the state this tick's Integrate just produced. A DerivedVariable
    // they name has to be re-evaluated against that same state, or one half of the comparison
    // is post-integrate (the cell_state read inside the test) and the other is a local
    // computed a whole dt earlier -- which is exactly wrong the moment a threshold is itself
    // derived from the variable being integrated, as it is here.
    const String source = generate_tick_kernel(make_derived_threshold_model()).source;

    const String declaration = "float derived_scaledThreshold = ";
    const String write_back = "cell_state[state_base + 0] = next_v;";
    const String reevaluation = "\n    derived_scaledThreshold = ";
    const String threshold_test = "if ((cell_state[state_base + 0] > derived_scaledThreshold))";

    const usize declaration_position = source.find(declaration);
    const usize write_back_position = source.find(write_back);
    const usize reevaluation_position = source.find(reevaluation);
    const usize test_position = source.find(threshold_test);

    ASSERT_NE(declaration_position, String::npos);
    ASSERT_NE(write_back_position, String::npos);
    ASSERT_NE(reevaluation_position, String::npos) << "the derived threshold is never "
                                                      "recomputed after the integrate step";
    ASSERT_NE(test_position, String::npos);

    // Declared before the Euler step (the TimeDerivative reads it), re-evaluated after the
    // write-back, and only then compared against.
    EXPECT_LT(declaration_position, write_back_position);
    EXPECT_LT(write_back_position, reevaluation_position);
    EXPECT_LT(reevaluation_position, test_position);

    // The re-evaluation is an assignment to the existing local, not a second declaration:
    // every expression written against the derived variable keeps reading the one name.
    EXPECT_EQ(source.find(declaration, declaration_position + 1), String::npos);
}

TEST(KernelCodegenKernel, DerivedVariablesAreNotReevaluatedWhenNothingIntegrates) {
    // A type with no TimeDerivative writes no state between the declaration and the
    // conditionals, so there is nothing for a re-evaluation to see. Emitting one anyway would
    // be dead code in every generated kernel for a type shaped like this.
    CellTypeSpecification cell_type = make_derived_threshold_cell_type();
    cell_type.dynamics.erase(cell_type.dynamics.begin() + 1); // the one TimeDerivative

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    const String source = generate_tick_kernel(parse_result).source;
    EXPECT_NE(source.find("float derived_scaledThreshold = "), String::npos);
    EXPECT_EQ(source.find("\n    derived_scaledThreshold = "), String::npos);
}

TEST(KernelCodegenKernel, StateAssignmentsInOneHandlerAreSimultaneous) {
    // LEMS evaluates a handler's assignments against the state as it stood when the handler
    // fired. Lowered one statement at a time, "v = u; u = v" assigns u to itself instead of
    // swapping -- and it is the document's ordering, not the model's, that decides.
    const String source = generate_tick_kernel(make_derived_threshold_model()).source;

    const usize first_temporary = source.find("float assigned_v = cell_state[state_base + 1];");
    const usize second_temporary = source.find("float assigned_u = cell_state[state_base + 0];");
    const usize first_write = source.find("cell_state[state_base + 0] = assigned_v;");
    const usize second_write = source.find("cell_state[state_base + 1] = assigned_u;");

    ASSERT_NE(first_temporary, String::npos);
    ASSERT_NE(second_temporary, String::npos);
    ASSERT_NE(first_write, String::npos);
    ASSERT_NE(second_write, String::npos);

    // Both right-hand sides are read before either is written back.
    EXPECT_LT(second_temporary, first_write);
    EXPECT_LT(first_temporary, second_write);
}

TEST(KernelCodegenKernel, ALoneStateAssignmentSkipsTheTemporary) {
    // A single assignment cannot observe its own write, so it goes straight to storage --
    // which is every handler in GLIF, and the initialize kernel's OnStart bodies. Scoped to
    // iafCell, whose OnCondition assigns once; izhikevichCell assigns twice in the same model
    // and does get the temporaries.
    const String source = generate_tick_kernel(make_two_cell_type_model()).source;

    const usize iaf_position = source.find("inline void cell_type_step_iafCell(");
    const usize izhikevich_position = source.find("inline void cell_type_step_izhikevichCell(");
    ASSERT_NE(iaf_position, String::npos);
    ASSERT_NE(izhikevich_position, String::npos);
    const String iaf_body = source.substr(iaf_position, izhikevich_position - iaf_position);

    EXPECT_NE(iaf_body.find("cell_state[state_base + 0] = cell_parameters[parameter_base + 4];"),
              String::npos);
    EXPECT_EQ(iaf_body.find("float assigned_"), String::npos);

    // The initialize kernel's one-assignment OnStart bodies are direct too.
    const String initialize_source =
            generate_initialize_kernel(make_two_cell_type_model()).source;
    EXPECT_NE(initialize_source.find(
                      "cell_state[state_base + 0] = cell_parameters[parameter_base + 1];"),
              String::npos);
    EXPECT_EQ(initialize_source.find("float assigned_"), String::npos);
}

TEST(KernelCodegenKernel, UngatedResetAndEmitFollowTheDetectBlocks) {
    // Stage order is Detect, then Emit, then Reset. An ungated reset emitted ahead of the
    // Detect blocks would be overwritten by whatever this tick's conditions wrote, instead of
    // overriding it. Well-formed LEMS never produces one, but if it ever does it must run in
    // the documented order rather than in whichever order is convenient to emit.
    CellTypeSpecification cell_type = make_integrate_and_fire_cell_type();
    cell_type.dynamics.push_back(make_instruction(DynamicsStage::Reset,
                                                  NML_DeclarationType::StateAssignment, "v",
                                                  "leakReversal"));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    const String source = generate_tick_kernel(parse_result).source;

    const usize detect_position =
            source.find("if ((cell_state[state_base + 0] > cell_parameters[parameter_base + 3]))");
    const usize ungated_reset_position =
            source.find("\n    cell_state[state_base + 0] = cell_parameters[parameter_base + 1];");

    ASSERT_NE(detect_position, String::npos);
    ASSERT_NE(ungated_reset_position, String::npos);
    EXPECT_LT(detect_position, ungated_reset_position);
}

TEST(KernelCodegenKernel, TheTickKernelLowersEachNeuronsSpikeFlagItself) {
    // The flag is raised by the dynamics and nothing else lowers it, so the kernel clears it
    // for the neuron the thread owns. Clearing it from the host between two launches instead
    // needs an explicit synchronisation on CUDA managed memory with concurrentManagedAccess
    // == 0, and clearing it after the dispatch would destroy the tick's own output.
    const NML_ParseResult parse_result = make_two_cell_type_model();
    const String source = generate_tick_kernel(parse_result).source;

    const usize clear_position = source.find("spike_flags[neuron_index] = 0;");
    const usize switch_position = source.find("switch (cell_type_index[neuron_index])");
    ASSERT_NE(clear_position, String::npos);
    ASSERT_NE(switch_position, String::npos);
    EXPECT_LT(clear_position, switch_position) << "the flag is lowered after the dynamics that "
                                                  "raise it";

    // The initialize kernel raises no flags, so it lowers none either.
    EXPECT_EQ(generate_initialize_kernel(parse_result).source.find("spike_flags[neuron_index] = 0;"),
              String::npos);
}

TEST(KernelCodegenKernel, InitializeKernelCarriesOnStartOnly) {
    const NML_ParseResult parse_result = make_two_cell_type_model();
    const String source = generate_initialize_kernel(parse_result).source;

    EXPECT_NE(source.find("kernel void initialize_cell_state("), String::npos);
    EXPECT_NE(source.find("cell_type_initialize_iafCell("), String::npos);
    // v = leakReversal, the OnStart body.
    EXPECT_NE(source.find("cell_state[state_base + 0] = cell_parameters[parameter_base + 1];"),
              String::npos);

    // Nothing from the tick stages leaks in. The only `if` is the kernel's own bounds
    // check, so the threshold test is what to look for rather than the keyword.
    EXPECT_EQ(source.find("float next_v"), String::npos);
    EXPECT_EQ(source.find("spike_flags[neuron_index] = 1;"), String::npos);
    EXPECT_EQ(source.find("cell_parameters[parameter_base + 3])"), String::npos);

    // Conversely, the tick kernel does not re-run the OnStart body.
    const String tick_source = generate_tick_kernel(parse_result).source;
    EXPECT_EQ(tick_source.find("cell_state[state_base + 0] = cell_parameters[parameter_base + 1];"),
              String::npos);
}

TEST(KernelCodegenKernel, ModelWithNoCellTypesStillProducesADispatchableKernel) {
    const NML_ParseResult parse_result;
    const String source = generate_tick_kernel(parse_result).source;

    EXPECT_NE(source.find("kernel void simulate_tick("), String::npos);
    EXPECT_NE(source.find("default:"), String::npos);
}

// ── select= paths ────────────────────────────────────────────────────────────

TEST(KernelCodegenSelectPath, SelectionOverSynapsesReadsTheInputAccumulator) {
    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(make_synaptic_input_cell_type("synapses[*]/i"));

    const String source = generate_tick_kernel(parse_result).source;

    // The path binds to the one local the ring row is loaded into, and that load is PLAIN:
    // nothing writes this tick's row while this tick's kernel runs, and the row is emptied
    // afterwards by the engine's clear kernel rather than by the reader.
    EXPECT_NE(source.find("float derived_iSyn = synaptic_input_accumulator;"), String::npos);
    // Plane 0: the delivered-current plane, which is where stimulus lands, where every
    // synapse adds its output, and where a synapse-free edge scatters its raw weight.
    EXPECT_NE(source.find("int synaptic_input_index = network_input_ring_index(\n"
                          "            tick, 0, ring_depth, neuron_count, neuron_index);"),
              String::npos);
    EXPECT_NE(source.find("float synaptic_input_accumulator = "
                          "network_inputs[synaptic_input_index];"),
              String::npos);

    // No read-and-clear anywhere: a reader that emptied its own slot is exactly what left the
    // slots of every non-reading cell type accumulating for the whole run.
    EXPECT_EQ(source.find("atomic_exchange_explicit"), String::npos);
    EXPECT_EQ(source.find("atomicExch"), String::npos);
}

TEST(KernelCodegenSelectPath, SelectedMemberNameDoesNotChangeTheBinding) {
    // iafCell selects "i" and izhikevichCell selects "I" from what is, on this engine, the
    // same one accumulator per neuron. Predicating the binding on the member name would
    // bind one of them and reject the other.
    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(make_synaptic_input_cell_type("synapses[*]/I"));

    const String source = generate_tick_kernel(parse_result).source;
    EXPECT_NE(source.find("float derived_iSyn = synaptic_input_accumulator;"), String::npos);
}

TEST(KernelCodegenSelectPath, TwoSelectionsShareOneLoadOfTheRingRow) {
    // Two reductions over the synapses read one consistent value, from a single load, rather
    // than loading the slot once per selection.
    CellTypeSpecification cell_type = make_synaptic_input_cell_type("synapses[*]/i");
    cell_type.dynamics.insert(
            cell_type.dynamics.begin() + 1,
            make_instruction(DynamicsStage::Integrate, NML_DeclarationType::DerivedVariable,
                             "iSynAgain", "synapses[*]/i"));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    const String source = generate_tick_kernel(parse_result).source;
    EXPECT_NE(source.find("float derived_iSyn = synaptic_input_accumulator;"), String::npos);
    EXPECT_NE(source.find("float derived_iSynAgain = synaptic_input_accumulator;"), String::npos);

    const String load = "float synaptic_input_accumulator = ";
    const usize first_load = source.find(load);
    ASSERT_NE(first_load, String::npos);
    EXPECT_EQ(source.find(load, first_load + 1), String::npos);
}

TEST(KernelCodegenSelectPath, ACellWithNoSynapticSelectionReadsNoRingRow) {
    // A type that never reduces over its synapses reads nothing from the ring. Its neurons'
    // slots are still emptied every tick -- the engine clears the whole row behind the tick
    // kernel -- so nothing accumulates there either.
    const String source = generate_tick_kernel(make_two_cell_type_model()).source;
    EXPECT_EQ(source.find("synaptic_input_accumulator"), String::npos);
}

TEST(KernelCodegenSelectPath, BoundNameResolvesInALaterTimeDerivative) {
    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(make_synaptic_input_cell_type("synapses[*]/i"));

    const String source = generate_tick_kernel(parse_result).source;

    const usize binding_position = source.find("float derived_iSyn =");
    const usize use_position = source.find("+ derived_iSyn)");
    ASSERT_NE(binding_position, String::npos);
    ASSERT_NE(use_position, String::npos);
    EXPECT_LT(binding_position, use_position);
}

TEST(KernelCodegenSelectPath, PathsWithNoEngineBufferBehindThemAreRejectedByPath) {
    // Every one of these appears in the vendored standard library, on a biophysical type.
    // Binding any of them to the synaptic accumulator would run and be silently wrong.
    const Vector<String> unsupported_paths = {
        "ionChannel/g",
        "populations[*]/i",
        "concentrationModels[species='ca']/concentration",
    };

    for (const String &path : unsupported_paths) {
        try {
            generate_tick_kernel(make_select_path_model(path));
            FAIL() << "expected select=\"" << path << "\" to be rejected";
        } catch (const runtime_error &error) {
            const String message = error.what();
            EXPECT_NE(message.find(path), String::npos) << "message did not name the path: "
                                                        << message;
            EXPECT_NE(message.find("selectCell"), String::npos) << message;
        }
    }
}

TEST(KernelCodegenSelectPath, DivisionOfTwoBareNamesStaysADivision) {
    // hindmarshRose1984Cell carries select="synapses[*]/i" and value="iMemb/C" in one
    // Dynamics block, so "contains a slash" cannot be what marks a path. A head that
    // resolves to a readable value means the slash was the division operator.
    CellTypeSpecification cell_type;
    cell_type.name = "divisionCell";
    cell_type.state_variable_names = {"v"};
    cell_type.parameter_names = {"iMemb", "C"};
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::DerivedVariable, "rate", "iMemb/C"));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "v", "rate"));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    const String source = generate_tick_kernel(parse_result).source;
    EXPECT_NE(source.find("float derived_rate = (cell_parameters[parameter_base + 0] / "
                          "cell_parameters[parameter_base + 1]);"),
              String::npos);
}

// ── Stage 6, Propagate ───────────────────────────────────────────────────────

TEST(KernelCodegenPropagation, EveryCellDeviceFunctionEndsWithTheSpikeGuardedEpilogue) {
    // Boilerplate, identical for every cell type -- including one that declares no EventOut
    // at all, which simply never raises the flag the guard tests.
    const String source = generate_tick_kernel(make_two_cell_type_model()).source;

    usize search_position = 0;
    usize epilogue_count = 0;
    while ((search_position = source.find("if (spike_flags[neuron_index] != 0) {",
                                          search_position)) != String::npos) {
        epilogue_count += 1;
        search_position += 1;
    }
    EXPECT_EQ(epilogue_count, 2u) << "one epilogue per cell device function";

    // It is an epilogue: the guard sits after the dynamics of the type it belongs to.
    const usize iaf_position = source.find("inline void cell_type_step_iafCell(");
    const usize izhikevich_position = source.find("inline void cell_type_step_izhikevichCell(");
    ASSERT_NE(iaf_position, String::npos);
    ASSERT_NE(izhikevich_position, String::npos);
    const String iaf_body = source.substr(iaf_position, izhikevich_position - iaf_position);
    EXPECT_LT(iaf_body.find("float next_v ="), iaf_body.find("propagate_spike("));
}

TEST(KernelCodegenPropagation, ArrivalsAreRingIndexedByDelayAndAddedAtomically) {
    const String source = generate_tick_kernel(make_two_cell_type_model()).source;

    // The arrival row is this tick plus the edge's own delay, the plane is whichever the edge
    // delivers through, and many sources converge on one target in a tick, so the add has to
    // be atomic.
    EXPECT_NE(source.find("tick + (long)edge_delay_ticks[edge_slot],\n"
                          "                edge_synapse_plane[edge_slot], ring_depth,"),
              String::npos);
    EXPECT_NE(source.find("atomic_fetch_add_explicit(arrival_slot, edge_weight, "
                          "memory_order_relaxed);"),
              String::npos);

    // The per-edge slot is the source's row of the flat per-edge arrays plus this
    // neighbour's position in the walk -- the convention WeightMatrix indexes by.
    EXPECT_NE(source.find("int edge_slot = neuron_index * max_neighbor_count + neighbor_slot;"),
              String::npos);
    // Walking past max_neighbor_count would read into the next source's row.
    EXPECT_NE(source.find("while (neighbor_slot < max_neighbor_count &&"), String::npos);
}

TEST(KernelCodegenPropagation, EdgeWeightIsTheLowRankReconstructionPlusItsSparseDelta) {
    const String source = generate_tick_kernel(make_two_cell_type_model()).source;

    // Sigma U[i,r] * Ck[r] * V[j,r], then the sparse delta that carries the reconstruction
    // onto the model's exact edge weight -- together, what WeightMatrix::get() reports.
    EXPECT_NE(source.find("reconstructed_weight += U_matrix[source_lane_base + lane] *"),
              String::npos);
    EXPECT_NE(source.find("(edge_weight_coefficients[lane] * V_matrix[target_lane_base + lane])"),
              String::npos);
    EXPECT_NE(source.find("edge_weight = reconstructed_weight + edge_weight_deltas[edge_slot];"),
              String::npos);

    // rank_float4_stride counts float4 elements; a row is four times that many lanes.
    EXPECT_NE(source.find("long row_lane_count = rank_float4_stride * 4;"), String::npos);
}

TEST(KernelCodegenPropagation, InitializeKernelNeitherPropagatesNorDrains) {
    // OnStart runs before any tick, so scattering a spike or emptying a ring row out of it
    // would deliver current the simulation never generated.
    const String source = generate_initialize_kernel(make_two_cell_type_model()).source;

    EXPECT_EQ(source.find("propagate_spike"), String::npos);
    EXPECT_EQ(source.find("k2tree_next_neighbor"), String::npos);
    EXPECT_EQ(source.find("network_input_ring_index"), String::npos);

    // It still declares the identical argument list, so the engine binds one set for both.
    EXPECT_NE(source.find("constant int       &ring_depth [[ buffer(28) ]]"), String::npos);
    EXPECT_NE(source.find("device const int   *edge_synapse_plane [[ buffer(30) ]]"),
              String::npos);
}

// ── the end-of-tick ring row clear ───────────────────────────────────────────

TEST(KernelCodegenRingClear, ClearsExactlyThisTicksRowAndNothingElse) {
    const GeneratedKernel clear_kernel =
            generate_ring_row_clear_kernel(make_two_cell_type_model());

    EXPECT_EQ(clear_kernel.function_name, "clear_network_input_ring_row");
    EXPECT_EQ(clear_kernel.argument_names,
              (Vector<String>{"network_inputs", "neuron_count", "tick", "ring_depth"}));

    // One row -- this tick's -- and within it only the columns belonging to the thread's
    // neuron. Clearing more would discard arrivals already scheduled into later rows, which
    // is the one thing the ring exists to hold.
    EXPECT_NE(clear_kernel.source.find("int ring_row = (int)(tick % (long)ring_depth);"),
              String::npos);
    EXPECT_NE(clear_kernel.source.find(
                      "int plane_base = ring_row * SPIKECOREC_NETWORK_INPUT_PLANE_COUNT;"),
              String::npos);
    EXPECT_NE(clear_kernel.source.find("network_inputs[(plane_base + plane_index) * "
                                       "neuron_count + neuron_index] = 0.0f;"),
              String::npos);

    // A model with no wired synapse is one plane wide: the delivered-current plane.
    EXPECT_NE(clear_kernel.source.find("#define SPIKECOREC_NETWORK_INPUT_PLANE_COUNT 1"),
              String::npos);

    // Bounds-checked like every other entry point, and it touches nothing but the ring.
    EXPECT_NE(clear_kernel.source.find("if (neuron_index >= neuron_count) return;"), String::npos);
    EXPECT_EQ(clear_kernel.source.find("cell_state"), String::npos);
    EXPECT_EQ(clear_kernel.source.find("spike_flags"), String::npos);
}

#ifdef SPIKECOREC_METAL
TEST(KernelCodegenRingClear, GeneratedRingClearKernelCompilesAsMetal) {
    if (!metal_compiler_is_available()) GTEST_SKIP() << "the Metal shader compiler is unavailable";

    String compiler_output;
    ASSERT_TRUE(compile_as_metal(
                        generate_ring_row_clear_kernel(make_two_cell_type_model()).source,
                        "ring_row_clear", compiler_output))
            << compiler_output;
}
#endif

// ── Synapse dynamics ─────────────────────────────────────────────────────────

TEST(KernelCodegenSynapse, ArrivalIsDeliveredThenIntegratedThenAddedToTheInputPlane) {
    const String source = generate_tick_kernel(make_alpha_synapse_model()).source;

    // Stage 1, Deliver: this tick's row of the prototype's own arrival plane, which is plane
    // 1 -- plane 0 being the delivered current every cell reads.
    EXPECT_NE(source.find("int synapse_arrival_index = network_input_ring_index(\n"
                          "            tick, 1, ring_depth, neuron_count, neuron_index);"),
              String::npos);
    EXPECT_NE(source.find("float synapse_arrival_weight = "
                          "network_inputs[synapse_arrival_index];"),
              String::npos);

    // The OnEvent body, gated on anything having arrived, with `weight` reading the SUMMED
    // arrival rather than the prototype's own value: that substitution is what makes one
    // evaluation stand in for every converging edge.
    EXPECT_NE(source.find("if (synapse_arrival_weight != 0.0f) {"), String::npos);
    EXPECT_NE(source.find("synapse_state[synapse_state_base + 1] = "
                          "(synapse_state[synapse_state_base + 1] + "
                          "(synapse_arrival_weight * 1.000000000e-09f));"),
              String::npos);

    // Stage 2, Integrate: both state variables, every tick, arrival or not, and written back
    // only once both derivatives have been computed.
    EXPECT_NE(source.find("float next_I = synapse_state[synapse_state_base + 0] + dt *"),
              String::npos);
    EXPECT_NE(source.find("float next_J = synapse_state[synapse_state_base + 1] + dt *"),
              String::npos);
    EXPECT_LT(source.find("float next_J ="),
              source.find("synapse_state[synapse_state_base + 0] = next_I;"));

    // Delivery converges on the input buffer: `i` added into plane 0 of this tick's row,
    // which is exactly what the cell's own "synapses[*]/i" path then reads.
    EXPECT_NE(source.find("int synapse_output_index = network_input_ring_index(\n"
                          "            tick, 0, ring_depth, neuron_count, neuron_index);"),
              String::npos);
    EXPECT_NE(source.find("network_inputs[synapse_output_index] += derived_i;"), String::npos);
}

TEST(KernelCodegenSynapse, SynapseParametersAreBakedAndStateIsPerTargetNeuron) {
    const String source = generate_tick_kernel(make_alpha_synapse_model()).source;

    // tau is a constant of the PROTOTYPE, so it is a literal rather than a buffer read --
    // which is what keeps the argument table from needing a synapse-parameter buffer.
    EXPECT_NE(source.find("2.000000000e-03f"), String::npos);
    EXPECT_EQ(source.find("synapse_parameters"), String::npos);

    // One slice per (prototype, neuron): the first prototype starts at offset zero and a
    // neuron occupies its type's two state variables.
    EXPECT_NE(source.find("int synapse_state_base = 0 * neuron_count + neuron_index * 2;"),
              String::npos);

    // The synapse runs ahead of the cell that reads what it delivered, in the same thread.
    EXPECT_LT(source.find("synapse_step_alphaSyn(synapse_state"),
              source.find("switch (cell_type_index[neuron_index])"));
}

TEST(KernelCodegenSynapse, TwoPrototypesGetTheirOwnPlaneAndTheirOwnStateSlice) {
    // Two alphaCurrentSynapse prototypes differing only in tau. Pooling them would have to
    // decay one at the other's rate, so each gets its own arrival plane and its own state.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    parse_result.synapse_prototypes.push_back(
            make_synapse_prototype("alphaSlow", 0, {1.0, 8.0e-3, 1.0e-9}));

    NetworkEdge second_edge;
    second_edge.target_neuron_index = 1;
    second_edge.synapse_prototype_index = 1;
    parse_result.neurons[0].outgoing_edges.push_back(second_edge);

    const String source = generate_tick_kernel(parse_result).source;

    EXPECT_NE(source.find("#define SPIKECOREC_NETWORK_INPUT_PLANE_COUNT 3"), String::npos);
    EXPECT_NE(source.find("tick, 1, ring_depth, neuron_count, neuron_index);"), String::npos);
    EXPECT_NE(source.find("tick, 2, ring_depth, neuron_count, neuron_index);"), String::npos);

    // The second prototype's slice starts past the first's two state variables.
    EXPECT_NE(source.find("int synapse_state_base = 0 * neuron_count + neuron_index * 2;"),
              String::npos);
    EXPECT_NE(source.find("int synapse_state_base = 2 * neuron_count + neuron_index * 2;"),
              String::npos);

    // Each decays at its own tau, which is the whole reason they are kept apart.
    EXPECT_NE(source.find("2.000000000e-03f"), String::npos);
    EXPECT_NE(source.find("8.000000000e-03f"), String::npos);
}

TEST(KernelCodegenSynapse, AnUnwiredPrototypeIsNeitherLoweredNorGivenAPlane) {
    // A synapse the model declares but no edge delivers through changes nothing about a
    // simulation, so it costs no plane, no state and no code -- and, being unreachable, is
    // not rejected for being conductance-based either.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    parse_result.synapse_types.push_back(make_conductance_synapse_type());
    parse_result.synapse_prototypes.push_back(
            make_synapse_prototype("unwiredConductance", 1, {1.0, 5.0e-10, 0.0, 3.0e-3}));

    EXPECT_EQ(wired_synapse_prototype_indices(parse_result), (Vector<s64>{0}));

    const String source = generate_tick_kernel(parse_result).source;
    EXPECT_NE(source.find("#define SPIKECOREC_NETWORK_INPUT_PLANE_COUNT 2"), String::npos);
    EXPECT_EQ(source.find("unwiredConductance"), String::npos);
}

TEST(KernelCodegenSynapse, EdgeThroughNoSynapseKeepsThePlainDeliveredWeight) {
    // A projection naming no synapse has no dynamics to run, so its raw weight is the
    // delivered current: plane 0, the one every cell reads.
    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(make_synaptic_input_cell_type("synapses[*]/i"));
    wire_one_edge(parse_result, /*synapse_prototype_index=*/-1);

    EXPECT_TRUE(wired_synapse_prototype_indices(parse_result).empty());

    const String source = generate_tick_kernel(parse_result).source;
    EXPECT_NE(source.find("#define SPIKECOREC_NETWORK_INPUT_PLANE_COUNT 1"), String::npos);
    EXPECT_EQ(source.find("synapse_step_"), String::npos);

    // The edge's plane is still read from the per-edge array rather than assumed, so a model
    // mixing synapse-carrying and synapse-free edges routes each of them correctly.
    EXPECT_NE(source.find("edge_synapse_plane[edge_slot], ring_depth,"), String::npos);
}

TEST(KernelCodegenSynapse, OnStartRunsInTheInitializeKernelAndNothingElseDoes) {
    const String source = generate_initialize_kernel(make_alpha_synapse_model()).source;

    EXPECT_NE(source.find("void synapse_initialize_alphaSyn("), String::npos);
    EXPECT_NE(source.find("synapse_state[synapse_state_base + 0] = assigned_I;"), String::npos);
    EXPECT_NE(source.find("synapse_state[synapse_state_base + 1] = assigned_J;"), String::npos);

    // Initialisation neither integrates nor delivers: doing either before the first tick
    // would inject current the simulation never generated.
    EXPECT_EQ(source.find("float next_I"), String::npos);
    EXPECT_EQ(source.find("network_input_ring_index"), String::npos);
}

TEST(KernelCodegenSynapse, ConductanceBasedSynapseIsRefusedByName) {
    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(make_synaptic_input_cell_type("synapses[*]/i"));
    parse_result.synapse_types.push_back(make_conductance_synapse_type());
    parse_result.synapse_prototypes.push_back(
            make_synapse_prototype("condSyn", 0, {1.0, 5.0e-10, 0.0, 3.0e-3}));
    wire_one_edge(parse_result, 0);

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a conductance-based synapse to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("expOneSynapse"), String::npos) << message;
        EXPECT_NE(message.find("conductance-based"), String::npos) << message;
        // Named for what it is, not reported as an unresolvable 'v'.
        EXPECT_NE(message.find("erev"), String::npos) << message;
    }
}

TEST(KernelCodegenSynapse, PerEdgeStateSynapseIsRefusedByName) {
    // A synapse carrying a plasticity or block mechanism does not superpose across
    // converging edges, so the aggregated per-target state this generator emits would be the
    // wrong model for it.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    parse_result.synapse_types[0].requires_per_edge_state = true;

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a synapse needing per-edge state to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("alphaCurrentSynapse"), String::npos) << message;
        EXPECT_NE(message.find("superpose"), String::npos) << message;
    }
}

TEST(KernelCodegenSynapse, ArrivalHandlerThatIgnoresTheWeightIsRefused) {
    // Arrivals converging on one target are summed into one weight and the handler runs
    // once. A handler that never reads that weight would apply many spikes as one, silently,
    // so it is refused rather than aggregated.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    for (DynamicsInstruction &instruction : parse_result.synapse_types[0].dynamics) {
        if (instruction.stage != DynamicsStage::Arrival) continue;
        instruction.expression = "J + ibase";
    }

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected an arrival handler ignoring 'weight' to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("alphaCurrentSynapse"), String::npos) << message;
        EXPECT_NE(message.find("weight"), String::npos) << message;
    }
}

TEST(KernelCodegenSynapse, SynapseWithNoArrivalHandlerIsRefused) {
    // Nothing routes a spike anywhere but the "in" port, so a synapse with no handler there
    // would swallow every spike delivered through it.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    Vector<DynamicsInstruction> without_arrival;
    for (const DynamicsInstruction &instruction : parse_result.synapse_types[0].dynamics) {
        if (instruction.stage == DynamicsStage::Arrival) continue;
        without_arrival.push_back(instruction);
    }
    parse_result.synapse_types[0].dynamics = without_arrival;

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a synapse with no OnEvent handler to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("alphaCurrentSynapse"), String::npos) << message;
        EXPECT_NE(message.find("OnEvent"), String::npos) << message;
    }
}

TEST(KernelCodegenSynapse, SynapseExposingNoCurrentIsRefused) {
    // `i` is what a postsynaptic cell's "synapses[*]/i" path selects, so a synapse without
    // one has nothing to deliver.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    for (DynamicsInstruction &instruction : parse_result.synapse_types[0].dynamics) {
        if (instruction.source_tag != NML_DeclarationType::DerivedVariable) continue;
        instruction.target = "notTheCurrent";
    }

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a synapse exposing no current to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("alphaCurrentSynapse"), String::npos) << message;
        EXPECT_NE(message.find("exposes no 'i'"), String::npos) << message;
    }
}

// ── Unsupported constructs ───────────────────────────────────────────────────

TEST(KernelCodegenUnsupported, EventArrivalIsRejectedByName) {
    CellTypeSpecification cell_type = make_integrate_and_fire_cell_type();
    cell_type.dynamics.push_back(make_instruction(DynamicsStage::Arrival,
                                                  NML_DeclarationType::OnEvent, "in", ""));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected OnEvent to be rejected";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("OnEvent"), String::npos);
        EXPECT_NE(message.find("iafCell"), String::npos);
    }
}

TEST(KernelCodegenUnsupported, RegimesAreRejectedByName) {
    CellTypeSpecification cell_type = make_integrate_and_fire_cell_type();
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::RegimeEntry, NML_DeclarationType::Regime, "refractory", ""));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a Regime to be rejected";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("Regime"), String::npos);
        EXPECT_NE(message.find("iafCell"), String::npos);
    }
}

TEST(KernelCodegenUnsupported, InstructionInsideARegimeIsRejected) {
    CellTypeSpecification cell_type;
    cell_type.name = "regimeCell";
    cell_type.state_variable_names = {"v"};

    DynamicsInstruction instruction = make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "v", "1");
    instruction.regime_name = "integrating";
    cell_type.dynamics.push_back(instruction);

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    EXPECT_THROW(generate_tick_kernel(parse_result), runtime_error);
}

TEST(KernelCodegenUnsupported, ConditionalDerivedVariableIsRejectedByName) {
    CellTypeSpecification cell_type = make_integrate_and_fire_cell_type();
    cell_type.dynamics.push_back(
            make_instruction(DynamicsStage::Integrate,
                             NML_DeclarationType::ConditionalDerivedVariable, "iSyn", ""));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a ConditionalDerivedVariable to be rejected";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("ConditionalDerivedVariable"), String::npos);
        EXPECT_NE(message.find("iafCell"), String::npos);
    }
}

TEST(KernelCodegenUnsupported, StateVariableDeclarationsAreSkippedNotRejected) {
    // A StateVariable lands on the RegimeEntry stage purely as the parser's default. It is
    // a declaration, so it must not trip the regime rejection.
    CellTypeSpecification cell_type = make_integrate_and_fire_cell_type();
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::RegimeEntry, NML_DeclarationType::StateVariable, "v", ""));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    EXPECT_NO_THROW(generate_tick_kernel(parse_result));
}

TEST(KernelCodegenUnsupported, StateAssignmentToANonStateVariableIsRejected) {
    CellTypeSpecification cell_type;
    cell_type.name = "mistypedCell";
    cell_type.state_variable_names = {"v"};
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "not_a_state", "1"));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    EXPECT_THROW(generate_tick_kernel(parse_result), runtime_error);
}

TEST(KernelCodegenUnsupported, ResetWithNoMatchingConditionIsRejected) {
    CellTypeSpecification cell_type;
    cell_type.name = "orphanCell";
    cell_type.state_variable_names = {"v"};
    cell_type.dynamics.push_back(make_instruction(DynamicsStage::Reset,
                                                  NML_DeclarationType::StateAssignment, "v", "0",
                                                  "v .gt. 1"));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    EXPECT_THROW(generate_tick_kernel(parse_result), runtime_error);
}

// ── End-to-end shader compilation ────────────────────────────────────────────

#ifdef SPIKECOREC_METAL

TEST(KernelCodegenMetal, GeneratedTickKernelCompilesAsMetal) {
    ASSERT_TRUE(metal_compiler_is_available())
            << "xcrun metal is unavailable, so the generated source was never compiled";

    String compiler_output;
    const String source = generate_tick_kernel(make_two_cell_type_model()).source;
    EXPECT_TRUE(compile_as_metal(source, "tick", compiler_output))
            << "generated Metal source failed to compile:\n"
            << compiler_output << "\n--- source ---\n"
            << source;
}

TEST(KernelCodegenMetal, GeneratedInitializeKernelCompilesAsMetal) {
    ASSERT_TRUE(metal_compiler_is_available())
            << "xcrun metal is unavailable, so the generated source was never compiled";

    String compiler_output;
    const String source = generate_initialize_kernel(make_two_cell_type_model()).source;
    EXPECT_TRUE(compile_as_metal(source, "initialize", compiler_output))
            << "generated Metal source failed to compile:\n"
            << compiler_output << "\n--- source ---\n"
            << source;
}

TEST(KernelCodegenMetal, CellConsumingSynapticInputCompilesAsMetal) {
    // The whole point of binding the select path: a cell that drains the accumulator has to
    // produce source the shader compiler accepts, not just source containing the right
    // substring. network_inputs is a kernel argument, so a mis-emitted read is caught here.
    ASSERT_TRUE(metal_compiler_is_available())
            << "xcrun metal is unavailable, so the generated source was never compiled";

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(make_synaptic_input_cell_type("synapses[*]/i"));

    String compiler_output;
    const String source = generate_tick_kernel(parse_result).source;
    EXPECT_TRUE(compile_as_metal(source, "synaptic_input", compiler_output))
            << "generated Metal source failed to compile:\n"
            << compiler_output << "\n--- source ---\n"
            << source;
}

TEST(KernelCodegenMetal, PropagatingCellCompilesAsMetal) {
    // The propagation epilogue brings the whole k^2-tree walk, the low-rank reconstruction
    // and two flavours of device atomic into the generated source. None of that is checked
    // by a substring search; only the real shader compiler settles whether the walk's
    // thread-address-space stack, the atomic casts and the ring arithmetic are well formed.
    ASSERT_TRUE(metal_compiler_is_available())
            << "xcrun metal is unavailable, so the generated source was never compiled";

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(make_synaptic_input_cell_type("synapses[*]/i"));
    parse_result.cell_types.push_back(make_izhikevich_cell_type());

    String compiler_output;
    const String source = generate_tick_kernel(parse_result).source;
    EXPECT_TRUE(compile_as_metal(source, "propagation", compiler_output))
            << "generated Metal source failed to compile:\n"
            << compiler_output << "\n--- source ---\n"
            << source;
}

TEST(KernelCodegenMetal, SynapseCarryingKernelCompilesAsMetal) {
    // The synapse stage adds a second storage layout, a plane-indexed ring and a device
    // function called from the master kernel ahead of the cell switch. Only the real shader
    // compiler settles whether all of that is well formed together.
    ASSERT_TRUE(metal_compiler_is_available())
            << "xcrun metal is unavailable, so the generated source was never compiled";

    NML_ParseResult parse_result = make_alpha_synapse_model();
    parse_result.synapse_prototypes.push_back(
            make_synapse_prototype("alphaSlow", 0, {1.0, 8.0e-3, 1.0e-9}));

    NetworkEdge second_edge;
    second_edge.target_neuron_index = 1;
    second_edge.synapse_prototype_index = 1;
    parse_result.neurons[0].outgoing_edges.push_back(second_edge);

    String compiler_output;
    const String tick_source = generate_tick_kernel(parse_result).source;
    EXPECT_TRUE(compile_as_metal(tick_source, "synapse_tick", compiler_output))
            << "generated Metal source failed to compile:\n"
            << compiler_output << "\n--- source ---\n"
            << tick_source;

    const String initialize_source = generate_initialize_kernel(parse_result).source;
    EXPECT_TRUE(compile_as_metal(initialize_source, "synapse_initialize", compiler_output))
            << "generated Metal source failed to compile:\n"
            << compiler_output << "\n--- source ---\n"
            << initialize_source;

    const String clear_source = generate_ring_row_clear_kernel(parse_result).source;
    EXPECT_TRUE(compile_as_metal(clear_source, "synapse_ring_clear", compiler_output))
            << "generated Metal source failed to compile:\n"
            << compiler_output << "\n--- source ---\n"
            << clear_source;
}

TEST(KernelCodegenMetal, GeneratedMathHeavyKernelCompilesAsMetal) {
    // Drives the translated forms the shader compiler is most likely to reject: pow from
    // '^', the Heaviside conditional, the remapped logarithms and a float-literal-only
    // expression tree.
    CellTypeSpecification cell_type;
    cell_type.name = "mathCell";
    cell_type.state_variable_names = {"v"};
    cell_type.parameter_names = {"tau", "scale", "thresh"};

    const String threshold_test = "v .geq. thresh .and. tau .gt. 0";

    cell_type.dynamics.push_back(make_instruction(DynamicsStage::Integrate,
                                                  NML_DeclarationType::DerivedVariable, "decay",
                                                  "exp(-dt / tau) ^ 2"));
    cell_type.dynamics.push_back(
            make_instruction(DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "v",
                             "decay * scale * H(v) + ln(2) - log(10) + sqrt(abs(v)) + t"));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Detect, NML_DeclarationType::OnCondition, "", threshold_test));
    cell_type.dynamics.push_back(make_instruction(DynamicsStage::Reset,
                                                  NML_DeclarationType::StateAssignment, "v",
                                                  "-70e-3", threshold_test));
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Emit, NML_DeclarationType::EventOut, "spike", "", threshold_test));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    String compiler_output;
    const String source = generate_tick_kernel(parse_result).source;
    EXPECT_TRUE(compile_as_metal(source, "math", compiler_output))
            << "generated Metal source failed to compile:\n"
            << compiler_output << "\n--- source ---\n"
            << source;
}

#endif
