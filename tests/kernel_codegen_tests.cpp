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
        "cell_state",      "cell_parameters", "network_inputs",      "last_spiked",
        "spike_flags",     "cell_state_base", "cell_parameter_base", "cell_type_index",
        "neuron_count",    "dt",              "tick",
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
    EXPECT_NE(source.find("float derived_iSyn = network_inputs[neuron_index];"), String::npos);
}

TEST(KernelCodegenSelectPath, SelectedMemberNameDoesNotChangeTheBinding) {
    // iafCell selects "i" and izhikevichCell selects "I" from what is, on this engine, the
    // same one accumulator per neuron. Predicating the binding on the member name would
    // bind one of them and reject the other.
    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(make_synaptic_input_cell_type("synapses[*]/I"));

    const String source = generate_tick_kernel(parse_result).source;
    EXPECT_NE(source.find("float derived_iSyn = network_inputs[neuron_index];"), String::npos);
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
