#include "spikecorec/nml/ir.h"

#include <sstream>

using namespace std;
using namespace spikecorec;

namespace spikecorec::nml {

namespace {

// ── Small local helpers ──────────────────────────────────────────────────

// std::visit helper: combines a set of lambdas into one callable overload
// set, one lambda per variant alternative.
template <class... Alternatives>
struct Overloaded : Alternatives... {
    using Alternatives::operator()...;
};
template <class... Alternatives>
Overloaded(Alternatives...) -> Overloaded<Alternatives...>;

String indent(int level) {
    return String(static_cast<usize>(level) * 2, ' ');
}

String mnemonic(BinaryOpcode opcode) {
    switch (opcode) {
        case BinaryOpcode::Add: return "add";
        case BinaryOpcode::Sub: return "sub";
        case BinaryOpcode::Mul: return "mul";
        case BinaryOpcode::Div: return "div";
        case BinaryOpcode::Mod: return "mod";
        case BinaryOpcode::Pow: return "pow";
        case BinaryOpcode::Min: return "min";
        case BinaryOpcode::Max: return "max";
        case BinaryOpcode::ExpDecay: return "expdecay";
        case BinaryOpcode::Gt: return "gt";
        case BinaryOpcode::Lt: return "lt";
        case BinaryOpcode::Ge: return "ge";
        case BinaryOpcode::Le: return "le";
        case BinaryOpcode::Eq: return "eq";
        case BinaryOpcode::Ne: return "ne";
        case BinaryOpcode::And: return "and";
        case BinaryOpcode::Or: return "or";
    }
    return "";
}

String mnemonic(UnaryOpcode opcode) {
    switch (opcode) {
        case UnaryOpcode::Neg: return "neg";
        case UnaryOpcode::Exp: return "exp";
        case UnaryOpcode::Log: return "log";
        case UnaryOpcode::Sqrt: return "sqrt";
        case UnaryOpcode::Abs: return "abs";
        case UnaryOpcode::Floor: return "floor";
        case UnaryOpcode::Ceil: return "ceil";
        case UnaryOpcode::Sin: return "sin";
        case UnaryOpcode::Cos: return "cos";
        case UnaryOpcode::Tan: return "tan";
        case UnaryOpcode::Sinh: return "sinh";
        case UnaryOpcode::Cosh: return "cosh";
        case UnaryOpcode::Tanh: return "tanh";
        case UnaryOpcode::Not: return "not";
    }
    return "";
}

String mnemonic(RandomOpcode opcode) {
    switch (opcode) {
        case RandomOpcode::Rand: return "rand";
        case RandomOpcode::Randn: return "randn";
    }
    return "";
}

String edge_reference_text(EdgeSetReference reference) {
    switch (reference) {
        case EdgeSetReference::CurrentEdge: return "edge";
        case EdgeSetReference::NeuronIn: return "neuron_in";
        case EdgeSetReference::NeuronOut: return "neuron_out";
    }
    return "";
}

// ── .tick printing (mutually recursive: instruction lists <-> control
//    constructs whose bodies are themselves instruction lists) ───────────

void print_instruction(const TickInstruction &instruction, int indent_level, ostringstream &output);

void print_instruction_list(const Vector<TickInstruction> &instructions, int indent_level, ostringstream &output) {
    for (const auto &instruction : instructions) {
        print_instruction(instruction, indent_level, output);
    }
}

void print_if_instruction(const IfInstruction &control, int indent_level, ostringstream &output) {
    output << indent(indent_level) << "if " << control.condition << " {\n";
    print_instruction_list(control.then_body, indent_level + 1, output);
    for (const auto &branch : control.else_if_branches) {
        output << indent(indent_level) << "} elif " << branch.condition << " {\n";
        print_instruction_list(branch.body, indent_level + 1, output);
    }
    if (control.else_body.has_value()) {
        output << indent(indent_level) << "} else {\n";
        print_instruction_list(*control.else_body, indent_level + 1, output);
    }
    output << indent(indent_level) << "}\n";
}

void print_forall_instruction(const ForAllInstruction &control, int indent_level, ostringstream &output) {
    output << indent(indent_level) << "forall " << edge_reference_text(control.edge_set) << " {\n";
    print_instruction_list(control.body, indent_level + 1, output);
    output << indent(indent_level) << "}\n";
}

void print_onevent_instruction(const OnEventInstruction &control, int indent_level, ostringstream &output) {
    output << indent(indent_level) << "onevent " << control.port_name << " {\n";
    print_instruction_list(control.body, indent_level + 1, output);
    output << indent(indent_level) << "}\n";
}

void print_instruction(const TickInstruction &instruction, int indent_level, ostringstream &output) {
    std::visit(Overloaded{
        [&](const BinaryInstruction &leaf) {
            output << indent(indent_level) << mnemonic(leaf.opcode) << " " << leaf.destination << ", "
                   << leaf.operand_a << ", " << leaf.operand_b << "\n";
        },
        [&](const UnaryInstruction &leaf) {
            output << indent(indent_level) << mnemonic(leaf.opcode) << " " << leaf.destination << ", "
                   << leaf.operand_a << "\n";
        },
        [&](const FusedMultiplyAddInstruction &leaf) {
            output << indent(indent_level) << "fma " << leaf.destination << ", " << leaf.operand_a << ", "
                   << leaf.operand_b << ", " << leaf.operand_c << "\n";
        },
        [&](const RandomInstruction &leaf) {
            output << indent(indent_level) << mnemonic(leaf.opcode) << " " << leaf.destination << "\n";
        },
        [&](const MoveInstruction &leaf) {
            output << indent(indent_level) << "mov " << leaf.destination << ", " << leaf.source << "\n";
        },
        [&](const LoadEdgeInstruction &leaf) {
            output << indent(indent_level) << "loadedge " << leaf.destination << ", " << leaf.variable_name << "@"
                   << edge_reference_text(leaf.edge_reference) << "\n";
        },
        [&](const AccumulateEdgeInstruction &leaf) {
            output << indent(indent_level) << "accedge " << leaf.variable_name << "@"
                   << edge_reference_text(leaf.edge_reference) << ", " << leaf.value << "\n";
        },
        [&](const EmitInstruction &leaf) {
            output << indent(indent_level) << "emit " << leaf.port_name << "\n";
        },
        [&](const SetRegimeInstruction &leaf) {
            output << indent(indent_level) << "set_regime " << leaf.regime_name << ", " << leaf.value << "\n";
        },
        [&](const IfInstruction &control) { print_if_instruction(control, indent_level, output); },
        [&](const ForAllInstruction &control) { print_forall_instruction(control, indent_level, output); },
        [&](const OnEventInstruction &control) { print_onevent_instruction(control, indent_level, output); },
    }, instruction.operation);
}

// ── .alloc printing (flat, one directive per line) ───────────────────────

String print_alloc_directive(const AllocDirective &directive) {
    return std::visit(Overloaded{
        [](const ParamConstantDirective &param) -> String {
            if (param.literal_value.has_value()) return "param " + param.name + " = " + *param.literal_value;
            return "param " + param.name;
        },
        [](const ParamDynamicDirective &param) -> String {
            return "param " + param.name + " : dyn " + param.dtype;
        },
        [](const StateDirective &state) -> String {
            String rendered = "state " + state.name + " : " + state.dtype;
            if (state.initial_value.has_value()) rendered += " = " + *state.initial_value;
            return rendered;
        },
        [](const AccumDirective &accum) -> String {
            return "accum " + accum.name + " : " + accum.dtype;
        },
        [](const PeredgeDirective &peredge) -> String {
            return "peredge " + peredge.name;
        },
        [](const RegimeDirective &regime) -> String {
            return "regime " + regime.name;
        },
        [](const ExposeDirective &expose) -> String {
            return "expose " + expose.name;
        },
        [](const RequireDirective &require) -> String {
            return "require " + require.name + " from " + require.scope;
        },
    }, directive);
}

void print_stage(ostringstream &output, const String &stage_name, const Vector<TickInstruction> &instructions) {
    if (instructions.empty()) return;
    output << indent(1) << "@" << stage_name << "\n";
    print_instruction_list(instructions, 2, output);
}

} // namespace

String print_ir_program(const IrProgram &program) {
    ostringstream output;

    output << ".alloc\n";
    for (const auto &directive : program.alloc) {
        output << indent(1) << print_alloc_directive(directive) << "\n";
    }

    output << ".tick\n";
    print_stage(output, "deliver", program.tick.deliver);
    print_stage(output, "integrate", program.tick.integrate);
    print_stage(output, "detect", program.tick.detect);
    print_stage(output, "emit", program.tick.emit);
    print_stage(output, "reset", program.tick.reset);
    print_stage(output, "propagate", program.tick.propagate);
    print_stage(output, "plasticity", program.tick.plasticity);
    print_stage(output, "record", program.tick.record);

    return output.str();
}

} // namespace spikecorec::nml
