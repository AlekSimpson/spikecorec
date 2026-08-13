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

// ── Regime fixtures ──────────────────────────────────────────────────────────

DynamicsInstruction make_regime_instruction(DynamicsStage stage, NML_DeclarationType source_tag,
                                            const String &target, const String &expression,
                                            const String &regime_name,
                                            const String &condition = "") {
    DynamicsInstruction instruction = make_instruction(stage, source_tag, target, expression,
                                                       condition);
    instruction.regime_name = regime_name;
    return instruction;
}

// A <Regime> declaration. Its `expression` carries the initial= attribute, which is the only
// record of where a cell starts -- see DynamicsInstruction.
DynamicsInstruction make_regime_declaration(const String &regime_name,
                                            const String &initial_attribute) {
    return make_instruction(DynamicsStage::RegimeEntry, NML_DeclarationType::Regime, regime_name,
                            initial_attribute);
}

// GLIF3's shape, which is the shape every regime-bearing fixture in tests/fixtures/nml uses:
//
//   asc1               regime-free TimeDerivative -- decays in BOTH regimes
//   v                  TimeDerivative in `integrating` only -- frozen while refractory
//   refractoryTime     TimeDerivative in `refractory` only -- frozen while integrating
//   integrating        OnCondition v > vth: spike, reset v, bump asc1, transition
//   refractory         OnEntry refractoryTime = 0; OnCondition refractoryTime >= t_ref:
//                      transition back
//
// The absences are the point: `v` having no derivative in `refractory` IS the refractory
// period, and asc1's derivative being outside both regimes is what keeps it decaying through
// one.
CellTypeSpecification make_two_regime_cell_type() {
    CellTypeSpecification cell_type;
    cell_type.name = "glif3Cell";
    cell_type.state_variable_names = {"v", "asc1", "refractoryTimeElapsed"};
    cell_type.parameter_names = {"tau", "vth", "vreset", "ascAdd1", "tauAsc1", "t_ref"};

    const String threshold_test = "v .gt. vth";
    const String refractory_test = "refractoryTimeElapsed .geq. t_ref";

    cell_type.dynamics.push_back(make_regime_declaration("integrating", "true"));
    cell_type.dynamics.push_back(make_regime_declaration("refractory", ""));

    cell_type.dynamics.push_back(make_instruction(DynamicsStage::Integrate,
                                                  NML_DeclarationType::TimeDerivative, "asc1",
                                                  "0 - asc1 / tauAsc1"));

    cell_type.dynamics.push_back(make_regime_instruction(DynamicsStage::Integrate,
                                                         NML_DeclarationType::TimeDerivative, "v",
                                                         "(0 - v) / tau", "integrating"));
    cell_type.dynamics.push_back(make_regime_instruction(DynamicsStage::Detect,
                                                         NML_DeclarationType::OnCondition, "",
                                                         threshold_test, "integrating"));
    cell_type.dynamics.push_back(make_regime_instruction(
            DynamicsStage::Emit, NML_DeclarationType::EventOut, "spike", "", "integrating",
            threshold_test));
    cell_type.dynamics.push_back(make_regime_instruction(
            DynamicsStage::Reset, NML_DeclarationType::StateAssignment, "v", "vreset",
            "integrating", threshold_test));
    cell_type.dynamics.push_back(make_regime_instruction(
            DynamicsStage::Reset, NML_DeclarationType::StateAssignment, "asc1", "asc1 + ascAdd1",
            "integrating", threshold_test));
    cell_type.dynamics.push_back(make_regime_instruction(
            DynamicsStage::RegimeEntry, NML_DeclarationType::Transition, "refractory", "",
            "integrating", threshold_test));

    cell_type.dynamics.push_back(make_regime_instruction(
            DynamicsStage::RegimeEntry, NML_DeclarationType::OnEntry, "", "", "refractory"));
    cell_type.dynamics.push_back(make_regime_instruction(
            DynamicsStage::RegimeEntry, NML_DeclarationType::StateAssignment,
            "refractoryTimeElapsed", "0", "refractory"));
    cell_type.dynamics.push_back(make_regime_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative,
            "refractoryTimeElapsed", "1", "refractory"));
    cell_type.dynamics.push_back(make_regime_instruction(DynamicsStage::Detect,
                                                         NML_DeclarationType::OnCondition, "",
                                                         refractory_test, "refractory"));
    cell_type.dynamics.push_back(make_regime_instruction(
            DynamicsStage::RegimeEntry, NML_DeclarationType::Transition, "integrating", "",
            "refractory", refractory_test));

    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Initialize, NML_DeclarationType::StateAssignment, "v", "0"));

    return cell_type;
}

NML_ParseResult make_two_regime_model() {
    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(make_two_regime_cell_type());
    return parse_result;
}

// One regime, which is therefore always active: no dispatch and no guards, so nothing reads
// the regime index at all.
CellTypeSpecification make_one_regime_cell_type() {
    CellTypeSpecification cell_type;
    cell_type.name = "oneRegimeCell";
    cell_type.state_variable_names = {"v"};
    cell_type.parameter_names = {"tau"};

    cell_type.dynamics.push_back(make_regime_declaration("running", "true"));
    cell_type.dynamics.push_back(make_regime_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "v", "0 - v / tau",
            "running"));

    return cell_type;
}

NML_ParseResult make_one_regime_model() {
    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(make_one_regime_cell_type());
    return parse_result;
}

// The body of the one cell device function in `source`, so an assertion about generated code
// cannot be satisfied by an unrelated part of the kernel.
String cell_device_function_body(const String &source, const String &function_name) {
    const usize signature_start = source.find(function_name + "(");
    if (signature_start == String::npos) return "";
    const usize body_start = source.find("{", signature_start);
    if (body_start == String::npos) return "";
    const usize body_end = source.find("\n}\n", body_start);
    if (body_end == String::npos) return "";
    return source.substr(body_start, body_end - body_start);
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

// A single-state-variable current synapse: `i` exposes the state directly, the state decays
// exponentially, and one arrival adds weight * ibase to it.
//
// One state variable is the shape the linearity check reduces furthest on -- with nothing to
// combine, additivity has to be probed against a state and its negation rather than against
// two states -- so it is what the sign probes are written against.
SynapseTypeSpecification make_exponential_current_synapse_type() {
    SynapseTypeSpecification synapse_type;
    synapse_type.name = "expCurrentSynapse";
    synapse_type.state_variable_names = {"I"};
    synapse_type.parameter_names = {"weight", "tau", "ibase"};

    synapse_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::DerivedVariable, "i", "I"));
    synapse_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "I", "-I / tau"));
    synapse_type.dynamics.push_back(make_instruction(
            DynamicsStage::Initialize, NML_DeclarationType::StateAssignment, "I", "0"));
    synapse_type.dynamics.push_back(
            make_instruction(DynamicsStage::Arrival, NML_DeclarationType::StateAssignment, "I",
                             "I + weight * ibase", "in"));

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
//
// `step_dt` is set because a synapse's delivery is resolved against it: what an arriving
// spike delivers is the total charge the response to that event carries, which is integrated
// out of the synapse's own dynamics at the model's own tick.
NML_ParseResult make_alpha_synapse_model() {
    NML_ParseResult parse_result;
    parse_result.step_dt = 1e-4;
    parse_result.cell_types.push_back(make_synaptic_input_cell_type("synapses[*]/i"));
    parse_result.synapse_types.push_back(make_alpha_current_synapse_type());
    parse_result.synapse_prototypes.push_back(
            make_synapse_prototype("alphaSyn", 0, {1.0, 2.0e-3, 1.0e-9}));
    wire_one_edge(parse_result, 0);
    return parse_result;
}

// One synaptic-input cell, one expCurrentSynapse prototype, one edge through it.
//
// `tau` and `step_dt` are arguments because both the linearity probes and the probe BUDGET are
// resolved against them: the whole point of the budget being model time is that the same tau
// builds at any dt.
NML_ParseResult make_exponential_synapse_model(f64 step_dt = 1.0e-4, f64 tau = 5.0e-3) {
    NML_ParseResult parse_result;
    parse_result.step_dt = step_dt;
    parse_result.cell_types.push_back(make_synaptic_input_cell_type("synapses[*]/i"));
    parse_result.synapse_types.push_back(make_exponential_current_synapse_type());
    parse_result.synapse_prototypes.push_back(
            make_synapse_prototype("expSyn", 0, {1.0, tau, 1.0e-9}));
    wire_one_edge(parse_result, 0);
    return parse_result;
}

// Replaces the expression of the one instruction of `source_tag` writing `target`.
void rewrite_synapse_expression(NML_ParseResult &parse_result, NML_DeclarationType source_tag,
                                const String &target, const String &expression) {
    for (DynamicsInstruction &instruction : parse_result.synapse_types[0].dynamics) {
        if (instruction.source_tag != source_tag) continue;
        if (instruction.target != target) continue;
        instruction.expression = expression;
    }
}

// The charge one unit of the alpha synapse's `J` goes on to deliver, integrated the same way
// the generator does it: forward Euler at `step_dt` until `I` has run its course. The
// continuous answer is e * tau; this is that same quantity as the discrete scheme actually
// produces it, which is what the generated kernel is checked against.
//
// 20000 steps is 2 seconds of model time at a 0.1 ms tick, a thousand time constants for the
// 2 ms synapse below -- the tail past that is far beneath the tolerances it is compared at.
f64 alpha_synapse_unit_charge(f64 step_dt, f64 tau) {
    f64 current_i = 0.0;
    f64 current_j = 1.0;
    f64 charge = 0.0;
    for (usize step = 0; step < 20000; ++step) {
        charge += current_i * step_dt;
        const f64 next_i =
                current_i + step_dt * ((2.7182818284590451 * current_j - current_i) / tau);
        const f64 next_j = current_j + step_dt * (-current_j / tau);
        current_i = next_i;
        current_j = next_j;
    }
    return charge;
}

// The charge coefficient the generator baked into `source`, read back out of the delivery it
// emitted. Compared numerically rather than as text: the generator stops integrating once the
// output has decayed and the reference above stops at a fixed step, so the two agree to far
// more digits than they do characters.
f64 baked_charge_coefficient(const String &source) {
    const usize charge_position = source.find("float delivered_charge =");
    if (charge_position == String::npos) return NAN;
    const usize literal_start = source.find_first_not_of(" \n", source.find("\n", charge_position));
    if (literal_start == String::npos) return NAN;
    return stod(source.substr(literal_start));
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
        "cell_state",           "cell_state_residual",   "cell_parameters",
        "network_inputs",       "last_spiked",           "spike_flags",
        "cell_state_base",
        "cell_parameter_base",  "cell_type_index",       "neuron_count",
        "dt",                   "tick",                  "internal_node_words",
        "leaf_node_words",      "rank_superblock_table", "rank_subblock_table",
        "U_matrix",             "V_matrix",              "edge_weight_coefficients",
        "edge_weight_deltas",   "edge_attributes",       "edge_synapse_state",
        "k2tree_shape",         "rank_float4_stride",    "constant_weight",
        "constant_weight_enabled",                       "max_neighbor_count",
        "ring_depth",
    };

    // Metal allows 31 buffer arguments per stage and the table used to sit at exactly 31,
    // which is why the regime index had to be squeezed into a cell_state slot. Asserted so a
    // future addition that spends the remaining headroom is a deliberate decision.
    EXPECT_EQ(expected_argument_names.size(), 28u);

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
    // One value per (row, neuron): a spike delivers the whole scalar its synapse computed,
    // and stimulus and synapse-free edges land in the same slot.
    EXPECT_NE(source.find("SpikecorecBufferIndex synaptic_input_index = network_input_ring_index(\n"
                          "            tick, ring_depth, neuron_count, neuron_index);"),
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

    // The arrival row is this tick plus the edge's own delay, and many sources converge on
    // one target in a tick, so the add has to be atomic.
    EXPECT_NE(source.find("tick + (long)edge_attributes[SPIKECOREC_EDGE_ATTRIBUTE_DELAY_PLANE"
                          " * edge_slot_count + edge_slot],\n"
                          "                ring_depth, neuron_count, target_neuron_index);"),
              String::npos);
    EXPECT_NE(source.find("atomic_fetch_add_explicit(arrival_slot, delivered_current, "
                          "memory_order_relaxed);"),
              String::npos);

    // Walking past max_neighbor_count would read into the next source's row.
    EXPECT_NE(source.find("while (neighbor_slot < max_neighbor_count &&"), String::npos);
}

TEST(KernelCodegenPropagation, PerEdgeSlotIsComputedInSixtyFourBits) {
    // neuron_index * max_neighbor_count overflows a signed int at ~33M neurons of degree 64,
    // and the wrapped negative index reads another edge's delay and another edge's synapse
    // state rather than faulting. The host computes the same product in s64.
    const String source = generate_tick_kernel(make_two_cell_type_model()).source;

    EXPECT_NE(source.find("SpikecorecBufferIndex edge_slot =\n"
                          "                (SpikecorecBufferIndex)neuron_index * "
                          "(SpikecorecBufferIndex)max_neighbor_count +\n"
                          "                (SpikecorecBufferIndex)neighbor_slot;"),
              String::npos)
            << source;
    EXPECT_EQ(source.find("int edge_slot = neuron_index"), String::npos) << source;

    // The plane stride the slot is offset by is 64-bit for the same reason.
    EXPECT_NE(source.find("SpikecorecBufferIndex edge_slot_count =\n"
                          "            (SpikecorecBufferIndex)neuron_count * "
                          "(SpikecorecBufferIndex)max_neighbor_count;"),
              String::npos)
            << source;
}

TEST(KernelCodegenPropagation, ConstantWeightIsSelectedByAnExplicitFlagNotByItsMagnitude) {
    // A model may legitimately configure a constant weight of exactly zero. Reading the
    // magnitude as the mode would silently hand it reconstructed U/V weights instead.
    const String source = generate_tick_kernel(make_two_cell_type_model()).source;

    EXPECT_NE(source.find("if (constant_weight_enabled == 0) {"), String::npos) << source;
    EXPECT_EQ(source.find("if (constant_weight == 0.0f)"), String::npos) << source;
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
    EXPECT_NE(source.find("SpikecorecBufferIndex row_lane_count = rank_float4_stride * 4;"),
              String::npos);
}

TEST(KernelCodegenPropagation, InitializeKernelNeitherPropagatesNorDrains) {
    // OnStart runs before any tick, so scattering a spike or draining a ring row out of it
    // would deliver current the simulation never generated.
    const String source = generate_initialize_kernel(make_two_cell_type_model()).source;

    EXPECT_EQ(source.find("propagate_spike"), String::npos);
    EXPECT_EQ(source.find("network_inputs["), String::npos);

    // It still declares the identical argument list, so the engine binds one set for both.
    EXPECT_NE(source.find("constant int       &ring_depth [[ buffer(27) ]]"), String::npos)
            << source;
    EXPECT_NE(source.find("device float       *edge_synapse_state [[ buffer(21) ]]"), String::npos)
            << source;
}

// ── the end-of-tick ring row clear ───────────────────────────────────────────

TEST(KernelCodegenRingClear, ClearsExactlyThisTicksRowAndNothingElse) {
    const GeneratedKernel clear_kernel = generate_ring_row_clear_kernel();

    EXPECT_EQ(clear_kernel.function_name, "clear_network_input_ring_row");
    EXPECT_EQ(clear_kernel.argument_names,
              (Vector<String>{"network_inputs", "neuron_count", "tick", "ring_depth"}));

    // One row -- this tick's -- and within it only the column belonging to the thread's
    // neuron. Clearing more would discard arrivals already scheduled into later rows, which
    // is the one thing the ring exists to hold.
    EXPECT_NE(clear_kernel.source.find("int ring_row = (int)(tick % (long)ring_depth);"),
              String::npos);
    EXPECT_NE(clear_kernel.source.find(
                      "network_inputs[(SpikecorecBufferIndex)ring_row * "
                      "(SpikecorecBufferIndex)neuron_count + "
                      "(SpikecorecBufferIndex)neuron_index] = 0.0f;"),
              String::npos)
            << clear_kernel.source;

    // Bounds-checked like every other entry point, and it touches nothing but the ring.
    EXPECT_NE(clear_kernel.source.find("if (neuron_index >= neuron_count) return;"), String::npos);
    EXPECT_EQ(clear_kernel.source.find("cell_state"), String::npos);
    EXPECT_EQ(clear_kernel.source.find("spike_flags"), String::npos);
}

#ifdef SPIKECOREC_METAL
TEST(KernelCodegenRingClear, GeneratedRingClearKernelCompilesAsMetal) {
    if (!metal_compiler_is_available()) GTEST_SKIP() << "the Metal shader compiler is unavailable";

    String compiler_output;
    ASSERT_TRUE(compile_as_metal(generate_ring_row_clear_kernel().source, "ring_row_clear",
                                 compiler_output))
            << compiler_output;
}
#endif

// ── Synapse dynamics ─────────────────────────────────────────────────────────

TEST(KernelCodegenSynapse, DeliveryIsOneScalarIntoOneRingSlotAtTheEdgesDelay) {
    const String source = generate_tick_kernel(make_alpha_synapse_model()).source;

    // The delivery runs inside the SOURCE's propagation walk, per out-edge, and its result
    // is one scalar the walk scatters into one slot -- not a per-target accumulation the
    // postsynaptic thread has to drain.
    EXPECT_NE(source.find("delivered_current = synapse_deliver_alphaSyn("), String::npos)
            << source;
    EXPECT_NE(source.find("tick + (long)edge_attributes[SPIKECOREC_EDGE_ATTRIBUTE_DELAY_PLANE"
                          " * edge_slot_count + edge_slot],\n"
                          "                ring_depth, neuron_count, target_neuron_index);"),
              String::npos)
            << source;
    EXPECT_NE(source.find("atomic_fetch_add_explicit(arrival_slot, delivered_current, "
                          "memory_order_relaxed);"),
              String::npos)
            << source;

    // Nothing per-neuron and nothing per-prototype survives into the tick: no aggregated
    // state buffer, no per-neuron synapse stage ahead of the cell dynamics, no ring plane
    // dimension, no per-edge plane index.
    EXPECT_EQ(source.find("*synapse_state,"), String::npos) << source;
    EXPECT_EQ(source.find("synapse_step_"), String::npos) << source;
    EXPECT_EQ(source.find("SPIKECOREC_NETWORK_INPUT_PLANE_COUNT"), String::npos) << source;
    EXPECT_EQ(source.find("edge_synapse_plane"), String::npos) << source;

    // The state base is a function of the EDGE, never of the target neuron -- which is what
    // "per edge" means here.
    EXPECT_NE(source.find("edge_synapse_state_base = 0 * edge_slot_count + edge_slot;"),
              String::npos)
            << source;
    EXPECT_EQ(source.find("edge_synapse_state_base = 0 * (SpikecorecBufferIndex)neuron_count"),
              String::npos)
            << source;
}

TEST(KernelCodegenSynapse, DeliveredChargeIsMeasuredAcrossTheArrivalHandler) {
    const String source = generate_tick_kernel(make_alpha_synapse_model()).source;

    // The state the handler is about to move is snapshotted first, and the delivered amount
    // is read off the CHANGE the handler made. Sampling the `i` exposure instead delivers
    // whatever `I` had decayed to -- exactly zero for an isolated first spike, at any weight,
    // which reads as a plausible "the network is just weakly coupled".
    EXPECT_EQ(source.find("float derived_i ="), String::npos) << source;

    const usize snapshot_position = source.find(
            "float synapse_state_before_J = "
            "edge_synapse_state[edge_synapse_state_base + 1 * edge_slot_count];");
    ASSERT_NE(snapshot_position, String::npos) << source;

    // `weight` inside the handler is THIS edge's weight, which is what makes the delivered
    // scalar a function of the edge rather than of a pooled arrival.
    const usize handler_position =
            source.find("float synapse_arrival_weight = edge_weight;", snapshot_position);
    const usize assignment_position = source.find("] = (edge_synapse_state[", handler_position);
    const usize charge_position = source.find("float delivered_charge =", assignment_position);
    const usize return_position = source.find("return delivered_charge / dt;", charge_position);
    ASSERT_NE(handler_position, String::npos) << source;
    ASSERT_NE(assignment_position, String::npos) << source;
    ASSERT_NE(charge_position, String::npos) << source;
    ASSERT_NE(return_position, String::npos) << source;
    EXPECT_LT(snapshot_position, handler_position);
    EXPECT_LT(handler_position, assignment_position);
    EXPECT_LT(assignment_position, charge_position);
    EXPECT_LT(charge_position, return_position);

    // The coefficient is the charge one unit of `J` goes on to deliver, and the scalar is a
    // CURRENT: the ring slot it lands in is integrated over exactly one tick, so charge Q is
    // delivered as Q / dt.
    EXPECT_NEAR(baked_charge_coefficient(source), alpha_synapse_unit_charge(1e-4, 2e-3),
                alpha_synapse_unit_charge(1e-4, 2e-3) * 1e-6)
            << source;
}

TEST(KernelCodegenSynapse, StateIsPerEdgeAndParametersAreBaked) {
    const String source = generate_tick_kernel(make_alpha_synapse_model()).source;

    // tau is a constant of the PROTOTYPE, so it is a literal rather than a buffer read --
    // which is what keeps the argument table from needing a synapse-parameter buffer.
    EXPECT_NE(source.find("2.000000000e-03f"), String::npos);
    EXPECT_EQ(source.find("synapse_parameters"), String::npos);

    // One plane per state variable, indexed by the edge rather than by either endpoint. The
    // first program's planes start at zero and its two variables are one plane apart, which
    // is exactly WeightMatrix's per-edge variable layout.
    EXPECT_NE(source.find("SpikecorecBufferIndex edge_synapse_state_base = 0 * "
                          "edge_slot_count + edge_slot;"),
              String::npos)
            << source;
    EXPECT_NE(source.find("edge_synapse_state[edge_synapse_state_base + 0 * edge_slot_count]"),
              String::npos)
            << source;
    EXPECT_NE(source.find("edge_synapse_state[edge_synapse_state_base + 1 * edge_slot_count]"),
              String::npos)
            << source;
}

TEST(KernelCodegenSynapse, TwoProgramsGetTheirOwnIndexAndTheirOwnStatePlanes) {
    // Two alphaCurrentSynapse prototypes differing only in tau. Pooling them would have to
    // decay one at the other's rate, so each is lowered separately and owns its own planes.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    parse_result.synapse_prototypes.push_back(
            make_synapse_prototype("alphaSlow", 0, {1.0, 8.0e-3, 1.0e-9}));

    NetworkEdge second_edge;
    second_edge.target_neuron_index = 1;
    second_edge.synapse_prototype_index = 1;
    parse_result.neurons[0].outgoing_edges.push_back(second_edge);

    const Vector<SynapseProgramLayout> programs = resolve_synapse_programs(parse_result);
    ASSERT_EQ(programs.size(), 2u);
    EXPECT_EQ(programs[0].state_variable_offset, 0);
    EXPECT_EQ(programs[0].state_variable_count, 2);
    EXPECT_EQ(programs[1].state_variable_offset, 2);
    EXPECT_EQ(programs[1].state_variable_count, 2);
    EXPECT_EQ(per_edge_synapse_variable_count(parse_result), 4);

    const String source = generate_tick_kernel(parse_result).source;

    // Each edge names one of them, and the switch is what routes it.
    EXPECT_NE(source.find("case 0:"), String::npos);
    EXPECT_NE(source.find("case 1:"), String::npos);
    EXPECT_NE(source.find("delivered_current = synapse_deliver_alphaSyn("), String::npos);
    EXPECT_NE(source.find("delivered_current = synapse_deliver_alphaSlow("), String::npos);

    // The second program's planes start past the first's two.
    EXPECT_NE(source.find("SpikecorecBufferIndex edge_synapse_state_base = 0 * "
                          "edge_slot_count + edge_slot;"),
              String::npos);
    EXPECT_NE(source.find("SpikecorecBufferIndex edge_synapse_state_base = 2 * "
                          "edge_slot_count + edge_slot;"),
              String::npos);

    // Each decays at its own tau, which is the whole reason they are kept apart.
    EXPECT_NE(source.find("2.000000000e-03f"), String::npos);
    EXPECT_NE(source.find("8.000000000e-03f"), String::npos);
}

TEST(KernelCodegenSynapse, AnUnwiredPrototypeIsNeitherLoweredNorAllocatedFor) {
    // A synapse the model declares but no edge delivers through changes nothing about a
    // simulation, so it costs no planes and no code -- and, being unreachable, is not
    // rejected for being conductance-based either.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    parse_result.synapse_types.push_back(make_conductance_synapse_type());
    parse_result.synapse_prototypes.push_back(
            make_synapse_prototype("unwiredConductance", 1, {1.0, 5.0e-10, 0.0, 3.0e-3}));

    const Vector<SynapseProgramLayout> programs = resolve_synapse_programs(parse_result);
    ASSERT_EQ(programs.size(), 1u);
    EXPECT_EQ(programs[0].prototype_index, 0);
    EXPECT_EQ(per_edge_synapse_variable_count(parse_result), 2);

    const String source = generate_tick_kernel(parse_result).source;
    EXPECT_EQ(source.find("unwiredConductance"), String::npos);
}

TEST(KernelCodegenSynapse, EdgeThroughNoSynapseKeepsThePlainDeliveredWeight) {
    // A projection naming no synapse has no dynamics to run, so its raw weight is the
    // delivered current -- which is what network_inputs meant before there were synapses.
    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(make_synaptic_input_cell_type("synapses[*]/i"));
    wire_one_edge(parse_result, /*synapse_prototype_index=*/-1);

    EXPECT_TRUE(resolve_synapse_programs(parse_result).empty());
    EXPECT_EQ(per_edge_synapse_variable_count(parse_result), 0);

    const String source = generate_tick_kernel(parse_result).source;
    EXPECT_EQ(source.find("synapse_deliver_"), String::npos);

    // Falls straight through to the raw weight, with no dispatch emitted at all.
    EXPECT_NE(source.find("float delivered_current = edge_weight;"), String::npos) << source;
    EXPECT_EQ(source.find("SPIKECOREC_EDGE_ATTRIBUTE_PROGRAM_PLANE * edge_slot_count"),
              String::npos)
            << source;
}

TEST(KernelCodegenSynapse, MixedModelStillReadsEachEdgesProgramFromItsOwnSlot) {
    // A model where one edge names a synapse and another does not: the program index is read
    // per edge rather than assumed, so the two route differently.
    //
    // Onto DIFFERENT targets, because the two deliver incompatible quantities into one
    // network_inputs slot and a target receiving both is refused -- see
    // MixedDeliveryOntoOneTargetIsRefused.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    parse_result.neurons.push_back(Neuron{});
    NetworkEdge synapse_free_edge;
    synapse_free_edge.target_neuron_index = 2;
    synapse_free_edge.synapse_prototype_index = -1;
    parse_result.neurons[0].outgoing_edges.push_back(synapse_free_edge);

    const String source = generate_tick_kernel(parse_result).source;
    EXPECT_NE(source.find("switch (edge_attributes[SPIKECOREC_EDGE_ATTRIBUTE_PROGRAM_PLANE "
                          "* edge_slot_count + edge_slot]) {"),
              String::npos)
            << source;
    // An edge whose slot holds no program keeps the weight the line above put there.
    EXPECT_NE(source.find("float delivered_current = edge_weight;"), String::npos) << source;
    EXPECT_NE(source.find("default:"), String::npos);
}

TEST(KernelCodegenSynapse, OnStartRunsInTheInitializeKernelPerEdge) {
    const String source = generate_initialize_kernel(make_alpha_synapse_model()).source;

    EXPECT_NE(source.find("void synapse_initialize_alphaSyn("), String::npos) << source;
    EXPECT_NE(source.find("edge_synapse_state[edge_synapse_state_base + 0 * edge_slot_count]"
                          " = assigned_I;"),
              String::npos)
            << source;
    EXPECT_NE(source.find("edge_synapse_state[edge_synapse_state_base + 1 * edge_slot_count]"
                          " = assigned_J;"),
              String::npos)
            << source;

    // Per EDGE, so it walks the adjacency the same way the tick kernel does.
    EXPECT_NE(source.find("void initialize_out_edge_synapses("), String::npos) << source;
    EXPECT_NE(source.find("k2tree_next_neighbor("), String::npos);

    // Initialisation neither integrates nor delivers: doing either before the first tick
    // would inject current the simulation never generated. The ring helper is emitted (both
    // entry points share one preamble) but nothing in the initialize kernel calls it.
    EXPECT_EQ(source.find("float next_I"), String::npos);
    EXPECT_EQ(source.find("network_inputs["), String::npos) << source;
    EXPECT_EQ(source.find("= network_input_ring_index("), String::npos) << source;
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

TEST(KernelCodegenSynapse, SynapseWithAChildMechanismIsRefusedByName) {
    // requires_per_edge_state records a <Children type="basePlasticityMechanism"/> or
    // "baseBlockMechanism" declaration. State being per-edge is no longer the obstacle; the
    // child structure, which this generator lowers none of, is.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    parse_result.synapse_types[0].requires_per_edge_state = true;

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a synapse with a child mechanism to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("alphaCurrentSynapse"), String::npos) << message;
        EXPECT_NE(message.find("child component"), String::npos) << message;
    }
}

TEST(KernelCodegenSynapse, ArrivalHandlerThatIgnoresTheWeightIsLoweredRatherThanRefused) {
    // Under per-edge delivery the handler runs once per ARRIVING SPIKE on one edge, so a
    // handler that does not read `weight` applies one spike as one spike. The refusal that
    // used to guard the old per-target aggregation would now reject a legitimate model.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    for (DynamicsInstruction &instruction : parse_result.synapse_types[0].dynamics) {
        if (instruction.stage != DynamicsStage::Arrival) continue;
        instruction.expression = "J + ibase";
    }

    const String source = generate_tick_kernel(parse_result).source;
    EXPECT_NE(source.find("synapse_deliver_alphaSyn("), String::npos) << source;
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

TEST(KernelCodegenSynapse, SynapseWhoseHandlerCannotReachTheExposedCurrentIsRefused) {
    // A handler that moves only state the exposed current never depends on sets nothing in
    // motion, so every spike through it would deliver exactly zero -- silently, on every edge,
    // for the whole run. Refused for the same reason a synapse with no handler at all is.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    for (DynamicsInstruction &instruction : parse_result.synapse_types[0].dynamics) {
        // `i` exposes I, and I is fed by J. Redirect the arrival onto a third variable that
        // nothing else reads.
        if (instruction.stage != DynamicsStage::Arrival) continue;
        instruction.target = "unread";
        instruction.expression = "unread + weight * ibase";
    }
    parse_result.synapse_types[0].state_variable_names.push_back("unread");
    parse_result.synapse_types[0].dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "unread",
            "-unread / tau"));

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a handler that cannot reach the exposed current to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("alphaCurrentSynapse"), String::npos) << message;
        EXPECT_NE(message.find("zero charge"), String::npos) << message;
    }
}

TEST(KernelCodegenSynapse, SynapseWhoseExposedCurrentNeverDecaysIsRefused) {
    // Delivering the whole of an event's response at the spike needs that response to be
    // finite. A synapse whose exposed current does not decay would deliver without end, and
    // there is no scalar that expresses that -- so it is named rather than truncated at
    // whatever the probe happened to reach.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    Vector<DynamicsInstruction> without_decay;
    for (const DynamicsInstruction &instruction : parse_result.synapse_types[0].dynamics) {
        // Drop I's decay, so `i` holds whatever J drove it to, for ever.
        if (instruction.source_tag == NML_DeclarationType::TimeDerivative &&
            instruction.target == "I") {
            continue;
        }
        without_decay.push_back(instruction);
    }
    parse_result.synapse_types[0].dynamics = without_decay;

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a non-decaying synapse to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("alphaCurrentSynapse"), String::npos) << message;
        EXPECT_NE(message.find("does not decay"), String::npos) << message;
    }
}

TEST(KernelCodegenSynapse, SynapseNonlinearInItsStateIsRefused) {
    // What one event delivers is precomputed as a fixed combination of the state change its
    // handler makes, and that decomposition only exists for dynamics linear in the state.
    // Running a nonlinear one through it anyway would deliver a plausible wrong number on
    // every spike, so the nonlinearity is measured and named instead of assumed away.
    //
    // The extra term is a decaying CUBIC rather than the quadratic this test used to carry.
    // The quadratic diverged from the probe's own starting states, so the refusal it produced
    // was the one for a response that never settles rather than the one for a nonlinear one --
    // a correct refusal for the wrong reason, and not the refusal this test is about. Being
    // odd, it also passes the negation probe, so what catches it is homogeneity at two.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    rewrite_synapse_expression(parse_result, NML_DeclarationType::TimeDerivative, "I",
                               "(2.7182818284590451*J - I - I*I*I)/tau");

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a synapse nonlinear in its state to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("alphaCurrentSynapse"), String::npos) << message;
        EXPECT_NE(message.find("not linear"), String::npos) << message;
    }
}

TEST(KernelCodegenSynapse, LinearSingleStateVariableSynapseIsAccepted) {
    // The control for the three refusals below: the same one-variable shape, linear, accepted,
    // and delivering the charge the discrete scheme actually carries. A decaying exponential's
    // Euler sum telescopes to exactly tau, which is what makes this an exact expectation rather
    // than a calibrated one.
    const String source = generate_tick_kernel(make_exponential_synapse_model()).source;

    EXPECT_NE(source.find("synapse_deliver_expSyn("), String::npos) << source;
    EXPECT_NEAR(baked_charge_coefficient(source), 5.0e-3, 5.0e-3 * 1e-5);
}

TEST(KernelCodegenSynapse, RectifyingSynapseIsRefusedRatherThanDeliveringTheWrongSign) {
    // `i = I * H(I)` is positively homogeneous of degree one AND additive across the whole
    // non-negative orthant, so a linearity check whose every probe point has all coordinates
    // >= 0 passes it. It is not linear: an inhibitory edge drives I negative, the true charge
    // is negative, and a coefficient measured only at positive I delivers the EXCITATORY
    // magnitude -- the right number with the wrong sign, on every inhibitory spike.
    //
    // Caught by the probe at -I, which the accepted synapse above passes.
    NML_ParseResult parse_result = make_exponential_synapse_model();
    rewrite_synapse_expression(parse_result, NML_DeclarationType::DerivedVariable, "i",
                               "I * H(I)");

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a rectifying synapse to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("expCurrentSynapse"), String::npos) << message;
        EXPECT_NE(message.find("not linear"), String::npos) << message;
        EXPECT_NE(message.find("minus one 'I'"), String::npos) << message;
    }
}

TEST(KernelCodegenSynapse, SynapseDeliveringTheMagnitudeOfItsStateIsRefused) {
    // The other half-plane failure, and the one that is worse: `i = abs(I)` passes every
    // non-negative probe exactly, and an inhibitory edge then delivers +Q where the model says
    // -Q. On a 100pF target a 1nA-scale event is the difference between +50mV and -50mV in one
    // tick.
    NML_ParseResult parse_result = make_exponential_synapse_model();
    rewrite_synapse_expression(parse_result, NML_DeclarationType::DerivedVariable, "i", "abs(I)");

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a synapse delivering the magnitude of its state to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("expCurrentSynapse"), String::npos) << message;
        EXPECT_NE(message.find("not linear"), String::npos) << message;
        EXPECT_NE(message.find("minus one 'I'"), String::npos) << message;
    }
}

TEST(KernelCodegenSynapse, SynapseLinearOnEveryAxisButNotAtMixedSignIsRefused) {
    // What the MIXED-SIGN pair probe adds over the single-axis ones. The cross term below is
    // gated by H(0 - 1 - I*J), which is zero at every point the other probes visit -- each
    // axis at one, at two and at minus one, and every variable at one -- and one at
    // (I, J) = (1, -1), where its whole first step lands in the charge.
    //
    // Constructed rather than taken from a real synapse, and deliberately so: the natural
    // half-plane nonlinearities (a rectifier, a magnitude) are already caught one probe
    // earlier by the negation probe, so isolating the PAIR probe takes a model built for it.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    rewrite_synapse_expression(parse_result, NML_DeclarationType::TimeDerivative, "I",
                               "(2.7182818284590451*J - I + H(0 - 1 - I*J))/tau");

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a synapse nonlinear only at mixed sign to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("alphaCurrentSynapse"), String::npos) << message;
        EXPECT_NE(message.find("not linear"), String::npos) << message;
        EXPECT_NE(message.find("one 'I' less one 'J'"), String::npos) << message;
    }
}

TEST(KernelCodegenSynapse, ChargeProbeBudgetIsModelTimeRatherThanAStepCount) {
    // A 200ms current synapse is entirely ordinary, and whether it BUILDS must be a property
    // of the model rather than of the tick it is simulated at. Under a fixed 200000-step
    // budget it built at dt = 0.1ms and threw at dt = 0.01ms, so halving the timestep to check
    // that a result had converged turned a working model into a construction-time exception.
    for (const f64 step_dt : {1.0e-4, 1.0e-5}) {
        const String source =
                generate_tick_kernel(make_exponential_synapse_model(step_dt, 0.2)).source;
        EXPECT_NEAR(baked_charge_coefficient(source), 0.2, 0.2 * 1e-5) << "dt = " << step_dt;
    }
}

TEST(KernelCodegenSynapse, ProbeThatDoesNotConvergeNamesTheStateVariableItProbed) {
    // The refusal has to name the probe that ran out. A synapse whose exposed current decays
    // perfectly well from one state variable can still have another whose probe never
    // converges, and reporting "this synapse's 'i' does not decay" of the type as a whole
    // sends the reader to a declaration that is fine.
    //
    // `i = I + W`: I decays, W grows. The probe from one 'I' converges; the one from one 'W'
    // is the one that does not.
    NML_ParseResult parse_result = make_exponential_synapse_model();
    parse_result.synapse_types[0].state_variable_names.push_back("W");
    rewrite_synapse_expression(parse_result, NML_DeclarationType::DerivedVariable, "i", "I + W");
    parse_result.synapse_types[0].dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "W", "W / tau"));

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a synapse whose charge does not converge to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("expCurrentSynapse"), String::npos) << message;
        EXPECT_NE(message.find("does not decay"), String::npos) << message;
        EXPECT_NE(message.find("one 'W'"), String::npos) << message;
        // And not blamed on the variable whose probe converged.
        EXPECT_EQ(message.find("one 'I'"), String::npos) << message;
    }
}

TEST(KernelCodegenSynapse, DepressingSynapseWithALongRecoveryConstantIsAccepted) {
    // A state variable the exposed current never reads offers no output decay to measure
    // against, so its probe ran on until the variable happened to ROUND onto its own limit --
    // about 37 time constants, against the 16 the decay criterion needs. A Tsodyks-Markram
    // recovery constant of 800ms was therefore refused at the project's default 0.1ms tick, on
    // a synapse whose exposed current decays in half a millisecond, and the refusal blamed
    // that current.
    //
    // The output being exactly zero for one step more than the synapse has state variables is
    // what now ends those probes: for dynamics affine in the state -- which is what the checks
    // beside it establish -- that many consecutive zeros force every later one.
    NML_ParseResult parse_result;
    parse_result.step_dt = 1.0e-4;
    parse_result.cell_types.push_back(make_synaptic_input_cell_type("synapses[*]/i"));

    SynapseTypeSpecification synapse_type;
    synapse_type.name = "depressingCurrentSynapse";
    synapse_type.state_variable_names = {"I", "available"};
    synapse_type.parameter_names = {"weight", "tau", "ibase", "tauRecovery", "releaseFraction"};
    synapse_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::DerivedVariable, "i", "I"));
    synapse_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "I", "-I / tau"));
    synapse_type.dynamics.push_back(
            make_instruction(DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative,
                             "available", "(1 - available) / tauRecovery"));
    synapse_type.dynamics.push_back(make_instruction(
            DynamicsStage::Initialize, NML_DeclarationType::StateAssignment, "I", "0"));
    synapse_type.dynamics.push_back(make_instruction(
            DynamicsStage::Initialize, NML_DeclarationType::StateAssignment, "available", "1"));
    synapse_type.dynamics.push_back(
            make_instruction(DynamicsStage::Arrival, NML_DeclarationType::StateAssignment, "I",
                             "I + weight * ibase * available", "in"));
    synapse_type.dynamics.push_back(
            make_instruction(DynamicsStage::Arrival, NML_DeclarationType::StateAssignment,
                             "available", "available * (1 - releaseFraction)", "in"));

    parse_result.synapse_types.push_back(synapse_type);
    parse_result.synapse_prototypes.push_back(make_synapse_prototype(
            "depressingSyn", 0, {1.0, 5.0e-4, 1.0e-9, /*tauRecovery=*/0.8, 0.5}));
    wire_one_edge(parse_result, 0);

    const String source = generate_tick_kernel(parse_result).source;

    EXPECT_NE(source.find("synapse_deliver_depressingSyn("), String::npos) << source;
    // `available` carries no charge of its own -- `i` never reads it -- so the only term the
    // delivery emits is I's, and its coefficient is the exposed current's own tau.
    EXPECT_NEAR(baked_charge_coefficient(source), 5.0e-4, 5.0e-4 * 1e-5);
}

TEST(KernelCodegenSynapse, MixedDeliveryOntoOneTargetIsRefused) {
    // An edge through a synapse delivers a current in amps; an edge through no synapse
    // delivers its raw dimensionless weight. Onto ONE target they sum in one network_inputs
    // slot and are drained by one `synapses[*]/i` read as a current, so a weight of 2.5
    // arrives as 2.5 A -- 2.5e6 V in one tick on a 100pF cell, about eight orders of magnitude
    // past what the synapse edge beside it delivers. Refused, because the result is a
    // plausible number rather than a crash.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    NetworkEdge synapse_free_edge;
    synapse_free_edge.target_neuron_index = 1; // the target the synapse edge already reaches
    synapse_free_edge.synapse_prototype_index = -1;
    synapse_free_edge.weight = 2.5;
    parse_result.neurons[0].outgoing_edges.push_back(synapse_free_edge);

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a target receiving both kinds of edge to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("neuron 1"), String::npos) << message;
        EXPECT_NE(message.find("network_inputs"), String::npos) << message;
    }
}

TEST(KernelCodegen, ThresholdCrossingRuleReproducesTheMeasuredPairs) {
    // The rule the construction-time warning evaluates, against the pairs measured on this
    // engine at its default tick. 5ms lands on the reference's tick; 0.5ms, 1ms and 2ms are
    // each 0.63 ulp short and hold a tick longer.
    EXPECT_TRUE(threshold_crossing_lands_on_reference_tick(1.0e-4, 5.0e-3));
    EXPECT_TRUE(threshold_crossing_lands_on_reference_tick(1.0e-4, 3.0e-3));
    EXPECT_TRUE(threshold_crossing_lands_on_reference_tick(1.0e-4, 2.5e-3));
    EXPECT_FALSE(threshold_crossing_lands_on_reference_tick(1.0e-4, 2.0e-3));
    EXPECT_FALSE(threshold_crossing_lands_on_reference_tick(1.0e-4, 1.0e-3));
    EXPECT_FALSE(threshold_crossing_lands_on_reference_tick(1.0e-4, 5.0e-4));

    // It is a property of the PAIR, not of the threshold: the same 2ms lands exactly at a
    // 0.5ms tick, where four increments scale by a power of two and lose nothing.
    EXPECT_TRUE(threshold_crossing_lands_on_reference_tick(5.0e-4, 2.0e-3));

    // A threshold no tick reaches, and a step of zero, are both "nothing to report" rather
    // than a division.
    EXPECT_TRUE(threshold_crossing_lands_on_reference_tick(1.0e-4, 0.0));
    EXPECT_TRUE(threshold_crossing_lands_on_reference_tick(0.0, 5.0e-3));
}

TEST(KernelCodegenSynapse, SynapseReadingTheAbsoluteClockIsRefused) {
    // What an arriving spike delivers is resolved once, at generation time. A synapse whose
    // dynamics read `t` would deliver a different amount depending on when the spike arrived,
    // which one set of precomputed coefficients cannot express.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    for (DynamicsInstruction &instruction : parse_result.synapse_types[0].dynamics) {
        if (instruction.source_tag != NML_DeclarationType::TimeDerivative) continue;
        if (instruction.target != "J") continue;
        instruction.expression = "-J/tau * (1 + t/tau)";
    }

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a synapse reading the absolute clock to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("alphaCurrentSynapse"), String::npos) << message;
        EXPECT_NE(message.find("absolute time"), String::npos) << message;
    }
}

TEST(KernelCodegenSynapse, ModelWithNoSimulationStepCannotResolveWhatASpikeDelivers) {
    // The charge one event delivers is integrated out of the synapse's own dynamics at the
    // model's own tick, so there is no answer without one. Named rather than divided by zero.
    NML_ParseResult parse_result = make_alpha_synapse_model();
    parse_result.step_dt = 0.0;

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a model with no simulation step to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("alphaCurrentSynapse"), String::npos) << message;
        EXPECT_NE(message.find("simulation step"), String::npos) << message;
    }
}

// ── Lazy vs eager synapse updates ────────────────────────────────────────────

TEST(KernelCodegenSynapse, LazyUpdatesCatchUpFromTheEdgesOwnLastAdvancedTick) {
    const String source =
            generate_tick_kernel(make_alpha_synapse_model(), /*use_lazy_synapse_updates=*/true)
                    .source;

    // The catch-up runs from the tick the edge was last advanced THROUGH up to this tick, so
    // an edge whose plane still holds -1 takes the one step tick 0's eager pass would have.
    EXPECT_NE(source.find("int synapse_last_update_tick =\n"
                          "            edge_attributes[SPIKECOREC_EDGE_ATTRIBUTE_UPDATE_TICK_PLANE"
                          " * edge_slot_count + edge_slot];"),
              String::npos)
            << source;
    EXPECT_NE(source.find("for (SpikecorecBufferIndex catch_up_tick = "
                          "(SpikecorecBufferIndex)synapse_last_update_tick;\n"
                          "                catch_up_tick < tick; ++catch_up_tick) {"),
              String::npos)
            << source;
    EXPECT_NE(source.find("edge_attributes[SPIKECOREC_EDGE_ATTRIBUTE_UPDATE_TICK_PLANE * "
                          "edge_slot_count + edge_slot] = (int)tick;"),
              String::npos)
            << source;

    // Nothing walks every edge every tick under this policy -- that is the whole point.
    EXPECT_EQ(source.find("advance_out_edge_synapses"), String::npos) << source;
}

TEST(KernelCodegenSynapse, EagerUpdatesAdvanceEveryOutEdgeEveryTickAndNeverCatchUp) {
    const String source =
            generate_tick_kernel(make_alpha_synapse_model(), /*use_lazy_synapse_updates=*/false)
                    .source;

    EXPECT_NE(source.find("void advance_out_edge_synapses("), String::npos) << source;
    EXPECT_NE(source.find("synapse_advance_alphaSyn(edge_synapse_state, edge_slot, "
                          "edge_slot_count, dt);"),
              String::npos)
            << source;

    // The advance precedes the cell dynamics, and so precedes the propagation that reads it.
    EXPECT_LT(source.find("advance_out_edge_synapses(internal_node_words"),
              source.find("switch (cell_type_index[neuron_index])"));

    // No catch-up at all: the state is already current, so consulting the last-advanced plane
    // would double-count. The plane's name is still #defined -- the layout is the engine's,
    // not the policy's -- but nothing indexes it.
    EXPECT_EQ(source.find("catch_up_tick"), String::npos) << source;
    EXPECT_EQ(source.find("SPIKECOREC_EDGE_ATTRIBUTE_UPDATE_TICK_PLANE * edge_slot_count"),
              String::npos)
            << source;
}

TEST(KernelCodegenSynapse, LazyCatchUpAppliesLiterallyTheEagerPerTickStep) {
    // Bit-for-bit agreement between the two policies rests on them emitting the same
    // arithmetic in the same order: the catch-up is the eager step, in a loop. Comparing the
    // emitted step text is what stops the two drifting into "close enough".
    const String lazy_source =
            generate_tick_kernel(make_alpha_synapse_model(), /*use_lazy_synapse_updates=*/true)
                    .source;
    const String eager_source =
            generate_tick_kernel(make_alpha_synapse_model(), /*use_lazy_synapse_updates=*/false)
                    .source;

    // Compared from the first Euler temporary to the end of the block, with leading
    // whitespace removed: the catch-up sits one level deeper inside its loop, and the
    // indentation is the only thing about it that is allowed to differ.
    auto integration_step_text = [](const String &source) {
        const usize step_start = source.find("float next_I = ");
        if (step_start == String::npos) return String();
        const usize step_end = source.find("next_J;", step_start);
        if (step_end == String::npos) return String();

        String stripped;
        bool at_line_start = true;
        for (const char character : source.substr(step_start, step_end - step_start)) {
            if (character == '\n') {
                at_line_start = true;
                stripped.push_back(character);
                continue;
            }
            if (at_line_start && character == ' ') continue;
            at_line_start = false;
            stripped.push_back(character);
        }
        return stripped;
    };

    const String lazy_step_text = integration_step_text(lazy_source);
    const String eager_step_text = integration_step_text(eager_source);
    ASSERT_FALSE(lazy_step_text.empty()) << lazy_source;
    ASSERT_FALSE(eager_step_text.empty()) << eager_source;
    EXPECT_EQ(lazy_step_text, eager_step_text);
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

TEST(KernelCodegenUnsupported, ARegimeNestedInsideAnotherIsRejected) {
    CellTypeSpecification cell_type = make_two_regime_cell_type();
    cell_type.dynamics.push_back(make_regime_instruction(
            DynamicsStage::RegimeEntry, NML_DeclarationType::Regime, "inner", "", "integrating"));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a nested Regime to be rejected";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("nested"), String::npos);
        EXPECT_NE(message.find("glif3Cell"), String::npos);
    }
}

TEST(KernelCodegenUnsupported, ATransitionOutsideAnOnConditionIsRejected) {
    // Nothing would gate it, so it would fire on every tick and the cell would never settle
    // in any regime.
    CellTypeSpecification cell_type = make_two_regime_cell_type();
    cell_type.dynamics.push_back(make_regime_instruction(DynamicsStage::RegimeEntry,
                                                         NML_DeclarationType::Transition,
                                                         "refractory", "", "integrating"));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    EXPECT_THROW(generate_tick_kernel(parse_result), runtime_error);
}

TEST(KernelCodegenUnsupported, ATransitionToAnUndeclaredRegimeIsRejectedByName) {
    CellTypeSpecification cell_type = make_two_regime_cell_type();
    for (DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.source_tag != NML_DeclarationType::Transition) continue;
        if (instruction.target != "refractory") continue;
        instruction.target = "recovering";
    }

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a Transition to an undeclared regime to be rejected";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("recovering"), String::npos);
        EXPECT_NE(message.find("glif3Cell"), String::npos);
    }
}

TEST(KernelCodegenUnsupported, ADerivedVariableInsideARegimeIsRejected) {
    // It would be emitted as an ordinary local, evaluated and visible in every regime, which
    // is not what declaring it inside one means.
    CellTypeSpecification cell_type = make_two_regime_cell_type();
    cell_type.dynamics.push_back(make_regime_instruction(DynamicsStage::Integrate,
                                                         NML_DeclarationType::DerivedVariable,
                                                         "scaled", "v * 2", "integrating"));

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

// ── Regimes ──────────────────────────────────────────────────────────────────

TEST(KernelCodegenRegimes, RegimesAreIndexedInDeclarationOrderWithTheInitialOneRecorded) {
    const CellRegimeLayout layout = resolve_cell_regimes(make_two_regime_cell_type());

    ASSERT_EQ(layout.regime_names.size(), 2u);
    EXPECT_EQ(layout.regime_names[0], "integrating");
    EXPECT_EQ(layout.regime_names[1], "refractory");
    EXPECT_EQ(layout.index_of("integrating"), 0);
    EXPECT_EQ(layout.index_of("refractory"), 1);
    EXPECT_EQ(layout.index_of("nonexistent"), -1);
    EXPECT_EQ(layout.initial_regime_index, 0);
    EXPECT_TRUE(layout.has_regimes());
}

TEST(KernelCodegenRegimes, TheRegimeIndexIsOneSlotAppendedAfterTheStateVariables) {
    // Appended rather than inserted, so every real state variable keeps the slot it had --
    // which is what lets recording and the engine's own layout stay unchanged.
    const CellTypeSpecification regime_cell = make_two_regime_cell_type();
    ASSERT_EQ(regime_cell.state_variable_names.size(), 3u);
    EXPECT_EQ(cell_state_slot_count(regime_cell), 4u);
    EXPECT_EQ(resolve_cell_regimes(regime_cell).regime_state_slot, 3u);

    // A type with no Regime pays nothing for the mechanism.
    const CellTypeSpecification plain_cell = make_integrate_and_fire_cell_type();
    EXPECT_EQ(cell_state_slot_count(plain_cell), plain_cell.state_variable_names.size());
    EXPECT_FALSE(resolve_cell_regimes(plain_cell).has_regimes());
}

TEST(KernelCodegenRegimes, AModelWithNoInitialRegimeThrowsNamingTheComponentType) {
    CellTypeSpecification cell_type = make_two_regime_cell_type();
    for (DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.source_tag != NML_DeclarationType::Regime) continue;
        instruction.expression = "";
    }

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a model with no initial regime to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("glif3Cell"), String::npos) << message;
        EXPECT_NE(message.find("initial"), String::npos) << message;
    }

    // The initialize kernel refuses it too: whichever entry point is generated first is where
    // a caller finds out, and both go through the same resolution.
    EXPECT_THROW(generate_initialize_kernel(parse_result), runtime_error);
}

TEST(KernelCodegenRegimes, TwoInitialRegimesAreRefused) {
    CellTypeSpecification cell_type = make_two_regime_cell_type();
    for (DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.source_tag != NML_DeclarationType::Regime) continue;
        instruction.expression = "true";
    }

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    EXPECT_THROW(generate_tick_kernel(parse_result), runtime_error);
}

TEST(KernelCodegenRegimes, ARegimeDeclaredTwiceIsRefused) {
    CellTypeSpecification cell_type = make_two_regime_cell_type();
    cell_type.dynamics.push_back(make_regime_declaration("refractory", ""));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    EXPECT_THROW(generate_tick_kernel(parse_result), runtime_error);
}

TEST(KernelCodegenRegimes, AnUnrecognisedInitialAttributeIsRefused) {
    CellTypeSpecification cell_type = make_two_regime_cell_type();
    for (DynamicsInstruction &instruction : cell_type.dynamics) {
        if (instruction.source_tag != NML_DeclarationType::Regime) continue;
        if (instruction.target != "integrating") continue;
        instruction.expression = "yes";
    }

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    EXPECT_THROW(generate_tick_kernel(parse_result), runtime_error);
}

TEST(KernelCodegenRegimes, TheRegimeIndexIsReadOnceIntoALocalAndDispatchedOnAsAnInteger) {
    const String body = cell_device_function_body(
            generate_tick_kernel(make_two_regime_model()).source, "cell_type_step_glif3Cell");
    ASSERT_FALSE(body.empty());

    // Read exactly once. Reading it again after a Transition has stored this tick's new value
    // would let the regime a cell just entered run its own OnCondition in the same tick.
    EXPECT_NE(body.find("int active_regime_index = (int)cell_state[state_base + 3];"),
              String::npos)
            << body;
    EXPECT_EQ(body.find("active_regime_index = (int)",
                        body.find("active_regime_index = (int)") + 1),
              String::npos)
            << "the regime index is read more than once:\n"
            << body;
}

TEST(KernelCodegenRegimes, AVariableWithNoDerivativeInARegimeEmitsNothingAndSoIsFrozen) {
    const String body = cell_device_function_body(
            generate_tick_kernel(make_two_regime_model()).source, "cell_type_step_glif3Cell");
    ASSERT_FALSE(body.empty());

    // `v` is seeded with its current value and only assigned inside the `integrating` branch.
    // The refractory branch carries no assignment at all -- that ABSENCE is the refractory
    // period. A zero derivative or a hold instruction would also freeze it, and would also
    // hide a regime that genuinely forgot to declare one.
    EXPECT_NE(body.find("float next_v = cell_state[state_base + 0];"), String::npos) << body;
    EXPECT_NE(body.find("if (active_regime_index == 0) {\n        float increment_v = dt *"),
              String::npos)
            << body;
    EXPECT_NE(body.find("        volatile float rounded_v = cell_state[state_base + 0] + "
                        "increment_v;\n        next_v = rounded_v;"),
              String::npos)
            << body;
    EXPECT_NE(body.find("} else {\n        // Regime 'refractory' declares no TimeDerivative for "
                        "'v', so it holds its value.\n    }"),
              String::npos)
            << body;

    // And the mirror image: refractoryTimeElapsed only advances while refractory.
    EXPECT_NE(body.find("if (active_regime_index == 0) {\n        // Regime 'integrating' declares "
                        "no TimeDerivative for 'refractoryTimeElapsed'"),
              String::npos)
            << body;
}

TEST(KernelCodegenRegimes, ARegimeFreeDerivativeIntegratesInEveryRegime) {
    const String body = cell_device_function_body(
            generate_tick_kernel(make_two_regime_model()).source, "cell_type_step_glif3Cell");
    ASSERT_FALSE(body.empty());

    // asc1's decay carries no regime, so it is emitted unguarded -- before the dispatch and
    // outside every branch. A GLIF3 whose after-spike currents stopped decaying during the
    // refractory period would still produce a spike train, just the wrong one.
    const usize decay_position = body.find("float next_asc1 = rounded_asc1;");
    ASSERT_NE(decay_position, String::npos) << body;
    EXPECT_LT(decay_position, body.find("if (active_regime_index ==")) << body;
}

TEST(KernelCodegenRegimes, EveryDerivativeIsWrittenBackAfterAllOfThemAreComputed) {
    const String body = cell_device_function_body(
            generate_tick_kernel(make_two_regime_model()).source, "cell_type_step_glif3Cell");
    ASSERT_FALSE(body.empty());

    // The regime dispatch computes into a temporary and the write-back follows it, exactly as
    // the regime-free path does, so a regime-scoped derivative reading a regime-free variable
    // still sees the state as it stood at tick entry.
    const usize dispatch_position = body.find("if (active_regime_index == 0) {");
    const usize write_back_position = body.find("cell_state[state_base + 0] = next_v;");
    ASSERT_NE(dispatch_position, String::npos) << body;
    ASSERT_NE(write_back_position, String::npos) << body;
    EXPECT_LT(dispatch_position, write_back_position) << body;
}

TEST(KernelCodegenRegimes, ARegimeScopedOnConditionIsGuardedByItsRegimeBeingActive) {
    const String body = cell_device_function_body(
            generate_tick_kernel(make_two_regime_model()).source, "cell_type_step_glif3Cell");
    ASSERT_FALSE(body.empty());

    // Both tests are ANDed with their own regime index. Without the guard the refractory
    // countdown's condition would be live while integrating -- and since
    // refractoryTimeElapsed is frozen at whatever the last refractory period left, it would
    // be permanently true, transitioning the cell back to integrating every tick.
    EXPECT_NE(body.find("if (active_regime_index == 0 && ((cell_state[state_base + 0] > "),
              String::npos)
            << body;
    EXPECT_NE(body.find("if (active_regime_index == 1 && ((cell_state[state_base + 2] >= "),
              String::npos)
            << body;
}

TEST(KernelCodegenRegimes, ATransitionStoresTheTargetIndexAndInlinesItsOnEntry) {
    const String body = cell_device_function_body(
            generate_tick_kernel(make_two_regime_model()).source, "cell_type_step_glif3Cell");
    ASSERT_FALSE(body.empty());

    // The whole of the transition: one store of the target regime's index, then that regime's
    // OnEntry body, inlined right here. There is no "did I just enter a regime" check
    // anywhere in the kernel -- entering IS the transition.
    EXPECT_NE(body.find("cell_state[state_base + 3] = 1.0f;\n        cell_state[state_base + 2] = "
                        "0.0f;"),
              String::npos)
            << body;

    // Coming back the other way needs no OnEntry, because `integrating` declares none.
    EXPECT_NE(body.find("if (active_regime_index == 1 && "), String::npos) << body;
    EXPECT_NE(body.find("cell_state[state_base + 3] = 0.0f;"), String::npos) << body;
}

TEST(KernelCodegenRegimes, OnEntryIsEmittedOnlyAtTransitionSitesSoItCannotRunTwice) {
    const String body = cell_device_function_body(
            generate_tick_kernel(make_two_regime_model()).source, "cell_type_step_glif3Cell");
    ASSERT_FALSE(body.empty());

    // refractoryTimeElapsed = 0 appears exactly once in the tick body: at the one transition
    // into `refractory`. If it were emitted anywhere the refractory regime runs, the
    // countdown would be reset on every refractory tick and the cell would never leave.
    usize occurrence_count = 0;
    for (usize position = body.find("cell_state[state_base + 2] = 0.0f;");
         position != String::npos;
         position = body.find("cell_state[state_base + 2] = 0.0f;", position + 1)) {
        occurrence_count += 1;
    }
    EXPECT_EQ(occurrence_count, 1u) << body;
}

TEST(KernelCodegenRegimes, InitializeSeedsTheInitialRegimeAndRunsItsOnEntry) {
    const String body = cell_device_function_body(
            generate_initialize_kernel(make_two_regime_model()).source,
            "cell_type_initialize_glif3Cell");
    ASSERT_FALSE(body.empty());

    // `integrating` is index 0 and declares no OnEntry, so seeding is the whole of it.
    EXPECT_NE(body.find("cell_state[state_base + 3] = 0.0f;"), String::npos) << body;

    // Marking the OTHER regime initial has to move the seed, not merely reorder the source.
    CellTypeSpecification refractory_first = make_two_regime_cell_type();
    for (DynamicsInstruction &instruction : refractory_first.dynamics) {
        if (instruction.source_tag != NML_DeclarationType::Regime) continue;
        instruction.expression = instruction.target == "refractory" ? "true" : "";
    }
    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(refractory_first);

    const String refractory_body =
            cell_device_function_body(generate_initialize_kernel(parse_result).source,
                                      "cell_type_initialize_glif3Cell");
    EXPECT_NE(refractory_body.find("cell_state[state_base + 3] = 1.0f;"), String::npos)
            << refractory_body;
    // ... and its OnEntry runs, because entering the initial regime is an entry like any
    // other.
    EXPECT_NE(refractory_body.find("cell_state[state_base + 2] = 0.0f;"), String::npos)
            << refractory_body;
}

TEST(KernelCodegenRegimes, TwoRegimesWithTheSameConditionTestKeepTheirOwnBodies) {
    // The join from a Reset/Emit/Transition back to the OnCondition that fires it is keyed on
    // the regime AND the test. On the test alone, two regimes that happen to test the same
    // thing -- which costs nothing to write and reads as obviously equivalent -- would each
    // execute the other's resets and the other's transition.
    CellTypeSpecification cell_type;
    cell_type.name = "sharedTestCell";
    cell_type.state_variable_names = {"v", "marker"};
    cell_type.parameter_names = {"thresh"};

    const String shared_test = "v .gt. thresh";

    cell_type.dynamics.push_back(make_regime_declaration("first", "true"));
    cell_type.dynamics.push_back(make_regime_declaration("second", ""));
    cell_type.dynamics.push_back(make_regime_instruction(DynamicsStage::Integrate,
                                                         NML_DeclarationType::TimeDerivative, "v",
                                                         "1", "first"));
    cell_type.dynamics.push_back(make_regime_instruction(
            DynamicsStage::Detect, NML_DeclarationType::OnCondition, "", shared_test, "first"));
    cell_type.dynamics.push_back(make_regime_instruction(
            DynamicsStage::Reset, NML_DeclarationType::StateAssignment, "marker", "11", "first",
            shared_test));
    cell_type.dynamics.push_back(make_regime_instruction(
            DynamicsStage::Detect, NML_DeclarationType::OnCondition, "", shared_test, "second"));
    cell_type.dynamics.push_back(make_regime_instruction(
            DynamicsStage::Reset, NML_DeclarationType::StateAssignment, "marker", "22", "second",
            shared_test));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    const String body = cell_device_function_body(
            generate_tick_kernel(parse_result).source, "cell_type_step_sharedTestCell");
    ASSERT_FALSE(body.empty());

    // Two separate guarded blocks, each carrying exactly its own assignment. The residual
    // reset that follows each is the assignment's own: an assigned variable's outstanding
    // accumulation belongs to a sum that no longer exists.
    EXPECT_NE(body.find("if (active_regime_index == 0 && ((cell_state[state_base + 0] > "
                        "cell_parameters[parameter_base + 0]))) {\n        cell_state[state_base + "
                        "1] = 11.0f;\n        cell_state_residual[state_base + 1] = 0.0f;\n    }"),
              String::npos)
            << body;
    EXPECT_NE(body.find("if (active_regime_index == 1 && ((cell_state[state_base + 0] > "
                        "cell_parameters[parameter_base + 0]))) {\n        cell_state[state_base + "
                        "1] = 22.0f;\n        cell_state_residual[state_base + 1] = 0.0f;\n    }"),
              String::npos)
            << body;
}

TEST(KernelCodegenRegimes, AVariableDerivedBothInsideAndOutsideARegimeIsRefused) {
    // Which one applies is not decidable from the document, and silently preferring either
    // changes how the variable moves in every regime.
    CellTypeSpecification cell_type = make_two_regime_cell_type();
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "v", "0"));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected a doubly-declared TimeDerivative to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("'v'"), String::npos) << message;
        EXPECT_NE(message.find("glif3Cell"), String::npos) << message;
    }
}

TEST(KernelCodegenRegimes, TwoTimeDerivativesForOneVariableOutsideAnyRegimeAreRefused) {
    // Left alone this emits two `float next_v` declarations and fails inside the shader
    // compiler, far from the model that caused it.
    CellTypeSpecification cell_type = make_integrate_and_fire_cell_type();
    cell_type.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "v", "0"));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    EXPECT_THROW(generate_tick_kernel(parse_result), runtime_error);
}

TEST(KernelCodegenRegimes, TwoTimeDerivativesForOneVariableInsideOneRegimeAreRefused) {
    // The regime-scoped path used to take the FIRST match and drop the rest, silently. A model
    // splitting its membrane derivative into a leak term and a synaptic term -- which is a
    // perfectly ordinary way to write it -- would then lose the whole of its synaptic input
    // with no diagnostic anywhere, while the regime-free path already refused the same shape.
    CellTypeSpecification cell_type = make_two_regime_cell_type();
    cell_type.dynamics.push_back(make_regime_instruction(DynamicsStage::Integrate,
                                                         NML_DeclarationType::TimeDerivative, "v",
                                                         "iSyn / C", "integrating"));

    NML_ParseResult parse_result;
    parse_result.cell_types.push_back(cell_type);

    try {
        generate_tick_kernel(parse_result);
        FAIL() << "expected two TimeDerivatives for one variable in one regime to be refused";
    } catch (const runtime_error &error) {
        const String message = error.what();
        EXPECT_NE(message.find("'v'"), String::npos) << message;
        EXPECT_NE(message.find("integrating"), String::npos) << message;
        EXPECT_NE(message.find("glif3Cell"), String::npos) << message;
        // Named for what it is, rather than surfacing as the unresolvable 'iSyn' the dropped
        // expression happens to contain.
        EXPECT_NE(message.find("more than one TimeDerivative"), String::npos) << message;
    }
}

TEST(KernelCodegenRegimes, ASingleRegimeNeedsNoDispatchAtAll) {
    const String body = cell_device_function_body(
            generate_tick_kernel(make_one_regime_model()).source, "cell_type_step_oneRegimeCell");
    ASSERT_FALSE(body.empty());
    EXPECT_EQ(body.find("if (active_regime_index =="), String::npos)
            << "a regime that is always active needs no guard:\n"
            << body;
    // And with no guard and no dispatch, nothing reads the regime index -- so it is not
    // declared either. An unused local is a shader-compiler warning on every kernel a model
    // of this shape produces.
    EXPECT_EQ(body.find("int active_regime_index"), String::npos) << body;
    EXPECT_NE(body.find("volatile float rounded_v = cell_state[state_base + 0] + increment_v;"),
              String::npos)
            << body;
    EXPECT_NE(body.find("\n    next_v = rounded_v;"), String::npos) << body;
}

// ── Flat buffer offsets are 64-bit ───────────────────────────────────────────

TEST(KernelCodegenBufferIndex, RingAndSynapseOffsetsAreComputedInSixtyFourBits) {
    // The host sizes and indexes network_inputs and the per-edge planes in s64. In `int` the
    // two disagree silently past INT_MAX -- 33M neurons of degree 64 already crosses it for
    // the per-edge slot -- and an offset wraps to a negative value, which lands in another
    // edge's or another neuron's slot rather than faulting.
    const String source = generate_tick_kernel(make_alpha_synapse_model()).source;

    EXPECT_NE(source.find("typedef long SpikecorecBufferIndex;"), String::npos) << source;
    EXPECT_NE(source.find("inline long network_input_ring_index("), String::npos) << source;
    EXPECT_EQ(source.find("inline int network_input_ring_index("), String::npos) << source;
    EXPECT_NE(source.find("SpikecorecBufferIndex edge_synapse_state_base ="), String::npos)
            << source;
    EXPECT_EQ(source.find("int edge_synapse_state_base ="), String::npos) << source;
    EXPECT_EQ(source.find("int edge_slot ="), String::npos) << source;
    EXPECT_EQ(source.find("int edge_slot_count ="), String::npos) << source;
    EXPECT_EQ(source.find("int arrival_index ="), String::npos) << source;
    EXPECT_EQ(source.find("int synaptic_input_index ="), String::npos) << source;
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

TEST(KernelCodegenMetal, TwoRegimeCellCompilesAsMetalWithoutWarnings) {
    // The regime dispatch emits an if / else-if / else chain with one branch deliberately
    // EMPTY, and a Transition writes an integer literal into a float slot. Both are the kind
    // of thing a substring assertion is happy with and the shader compiler is not.
    //
    // Warnings are failures here, not noise: the two shapes this generator can produce that
    // the compiler merely warns about -- an unused regime local, an empty branch -- are
    // exactly the shapes the regime lowering emits, and a warning on every generated kernel
    // is what stops anyone reading the ones that matter.
    ASSERT_TRUE(metal_compiler_is_available())
            << "xcrun metal is unavailable, so the generated source was never compiled";

    struct GeneratedCase {
        String case_name;
        String source;
    };
    const Vector<GeneratedCase> generated_cases = {
        {"regime_tick", generate_tick_kernel(make_two_regime_model()).source},
        {"regime_initialize", generate_initialize_kernel(make_two_regime_model()).source},
        {"one_regime_tick", generate_tick_kernel(make_one_regime_model()).source},
        {"one_regime_initialize", generate_initialize_kernel(make_one_regime_model()).source},
    };

    for (const GeneratedCase &generated_case : generated_cases) {
        String compiler_output;
        EXPECT_TRUE(compile_as_metal(generated_case.source, generated_case.case_name,
                                     compiler_output))
                << generated_case.case_name << ": generated Metal source failed to compile:\n"
                << compiler_output << "\n--- source ---\n"
                << generated_case.source;
        EXPECT_TRUE(compiler_output.empty())
                << generated_case.case_name << ": the shader compiler warned:\n"
                << compiler_output << "\n--- source ---\n"
                << generated_case.source;
    }
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

TEST(KernelCodegenMetal, SynapseCarryingKernelCompilesAsMetalUnderBothUpdatePolicies) {
    // The synapse path adds a per-edge storage layout, a per-edge program switch inside the
    // propagation walk and a second adjacency walk under the eager policy. Only the real
    // shader compiler settles whether all of that is well formed together, and warnings are
    // failures: a warning on every generated kernel is what stops anyone reading the ones
    // that matter.
    ASSERT_TRUE(metal_compiler_is_available())
            << "xcrun metal is unavailable, so the generated source was never compiled";

    NML_ParseResult parse_result = make_alpha_synapse_model();
    parse_result.synapse_prototypes.push_back(
            make_synapse_prototype("alphaSlow", 0, {1.0, 8.0e-3, 1.0e-9}));

    NetworkEdge second_edge;
    second_edge.target_neuron_index = 1;
    second_edge.synapse_prototype_index = 1;
    parse_result.neurons[0].outgoing_edges.push_back(second_edge);

    // A handler that never reads `weight` is legal now that delivery is per edge, and it is
    // the shape that leaves the arrival local declared and unread.
    NML_ParseResult weight_ignoring_model = make_alpha_synapse_model();
    for (DynamicsInstruction &instruction : weight_ignoring_model.synapse_types[0].dynamics) {
        if (instruction.stage != DynamicsStage::Arrival) continue;
        instruction.expression = "J + ibase";
    }

    // A synapse whose `i` is a StateVariable rather than a DerivedVariable, so the delivery
    // returns a storage read with no local in front of it -- and one that declares no
    // DerivedVariable at all, so the "drop the locals nothing reads" pass has nothing to
    // keep.
    SynapseTypeSpecification exponential_current_synapse;
    exponential_current_synapse.name = "expCurrentSynapse";
    exponential_current_synapse.state_variable_names = {"i"};
    exponential_current_synapse.parameter_names = {"weight", "tau", "ibase"};
    exponential_current_synapse.dynamics.push_back(make_instruction(
            DynamicsStage::Integrate, NML_DeclarationType::TimeDerivative, "i", "-i / tau"));
    exponential_current_synapse.dynamics.push_back(
            make_instruction(DynamicsStage::Arrival, NML_DeclarationType::StateAssignment, "i",
                             "i + weight * ibase", "in"));

    NML_ParseResult state_exposed_model;
    state_exposed_model.step_dt = 1e-4;
    state_exposed_model.cell_types.push_back(make_synaptic_input_cell_type("synapses[*]/i"));
    state_exposed_model.synapse_types.push_back(exponential_current_synapse);
    state_exposed_model.synapse_prototypes.push_back(
            make_synapse_prototype("expSyn", 0, {1.0, 5.0e-4, 1.0e-9}));
    wire_one_edge(state_exposed_model, 0);

    struct GeneratedCase {
        String case_name;
        String source;
    };
    const Vector<GeneratedCase> generated_cases = {
        {"synapse_tick_lazy", generate_tick_kernel(parse_result, true).source},
        {"synapse_tick_eager", generate_tick_kernel(parse_result, false).source},
        {"synapse_initialize", generate_initialize_kernel(parse_result).source},
        {"synapse_ring_clear", generate_ring_row_clear_kernel().source},
        {"synapse_weight_ignoring", generate_tick_kernel(weight_ignoring_model, true).source},
        {"synapse_state_exposed", generate_tick_kernel(state_exposed_model, true).source},
    };

    for (const GeneratedCase &generated_case : generated_cases) {
        String compiler_output;
        EXPECT_TRUE(compile_as_metal(generated_case.source, generated_case.case_name,
                                     compiler_output))
                << generated_case.case_name << ": generated Metal source failed to compile:\n"
                << compiler_output << "\n--- source ---\n"
                << generated_case.source;
        EXPECT_TRUE(compiler_output.empty())
                << generated_case.case_name << ": the shader compiler warned:\n"
                << compiler_output << "\n--- source ---\n"
                << generated_case.source;
    }
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
