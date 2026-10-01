#include "spikecorec/nml/dynamics.h"

#include <sstream>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/log.h"
#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"
#include "spikecorec/nml/node.h"
#include "spikecorec/nml/parser.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

void Codegen::allocate_cell_model_memory() {
    device->partition(sizeof(f32) * context.get_cell_state_size(), EngineDatatype::FLOAT32, data_partitions)            // cell_state
          .partition(sizeof(s64) * 2 * context.simulation.total_neuron_count, EngineDatatype::SIGNED64, data_partitions)   // network_inputs
          .partition(sizeof(u8) * (context.simulation.maximum_edge_delay + 1) * context.simulation.total_neuron_count,
                     EngineDatatype::UNSIGNED8, data_partitions)                                                           // spike_history
          .partition(sizeof(s64) * context.simulation.total_neuron_count, EngineDatatype::SIGNED64, data_partitions)       // last_spiked
          .partition(sizeof(f32), EngineDatatype::FLOAT32, data_partitions);                                               // empty_edge_plane

    EnginePointer slab = device->allocate(data_partitions);
    data_partitions.push_back(slab);

    log::logger().debug("Codegen: {} bytes: cell_state {}, network_inputs {}, spike_history {}, last_spiked {}",
                        slab.total_bytes, context.get_cell_state_size(), 2 * context.simulation.total_neuron_count,
                        (context.simulation.maximum_edge_delay + 1) * context.simulation.total_neuron_count,
                        context.simulation.total_neuron_count);
}

// AST construction

KernelNode *new_declaration(const String &type, const String &name, KernelNode *value) {
    KernelBinaryNode *declaration = new_node<BinaryNode>(KernelNodeType::DECLARATION, type);
    declaration->left = new_node(KernelNodeType::IDENTIFIER, name);
    declaration->right = value;
    return declaration;
}

KernelNode *new_assignment(KernelNode *target, const String &assignment_operator, KernelNode *value) {
    KernelBinaryNode *assignment = new_node<BinaryNode>(KernelNodeType::ASSIGNMENT, assignment_operator);
    assignment->left = target;
    assignment->right = value;
    return assignment;
}

KernelNode *new_expression(const String &expression_operator, KernelNode *left, KernelNode *right) {
    KernelBinaryNode *expression = new_node<BinaryNode>(KernelNodeType::EXPRESSION, expression_operator);
    expression->left = left;
    expression->right = right;
    return expression;
}

KernelNode *new_cast(const String &type, KernelNode *value) {
    KernelUnaryNode *cast = new_node<UnaryNode>(KernelNodeType::CAST, type);
    cast->child = value;
    return cast;
}

KernelListNode *new_conditional() {
    return new_node<ListNode>(KernelNodeType::CONDITIONAL, "");
}

void add_branch(KernelListNode *conditional, KernelNode *test, KernelNode *body) {
    if (body->body.syntax_type != KernelNodeType::BLOCK) throw runtime_error("Conditional branch body must be a BLOCK");
    if (!test && conditional->children.empty()) throw runtime_error("Conditional cannot start with else");
    if (!conditional->children.empty() && !static_cast<KernelBinaryNode *>(conditional->children.back())->left)
        throw runtime_error("Conditional already ends with else");

    KernelBinaryNode *branch = new_node<BinaryNode>(KernelNodeType::CONDITIONAL_BRANCH, "");
    branch->left = test;
    branch->right = body;
    conditional->children.push_back(branch);
}

// AST generation

KernelNode *Codegen::create_kernel_root() {
}

KernelNode *Codegen::translate_kernel_code() {
}

KernelNode *Codegen::translate_component_type(NML_ComponentType &component_type) {
}

KernelNode *Codegen::translate_component_instance(NML_ComponentInstance &component_instance) {
}

KernelNode *Codegen::translate_expression(NML_DynamicsExpression &expression) {
}

f64 Codegen::get_starting_parameters(NML_DynamicsExpression &expression) {
}

// code generation

const UnorderedMap<String, String> &kernel_type_spellings(KernelBackend backend) {
    static const UnorderedMap<String, String> metal_spellings = {
        {"void", "void"}, {"bool", "bool"}, {"u8", "uchar"}, {"u16", "ushort"}, {"s32", "int"},
        {"u32", "uint"}, {"s64", "long"}, {"f32", "float"}};
    static const UnorderedMap<String, String> cuda_spellings = {
        {"void", "void"}, {"bool", "bool"}, {"u8", "unsigned char"}, {"u16", "unsigned short"}, {"s32", "int"},
        {"u32", "unsigned int"}, {"s64", "long long"}, {"f32", "float"}, {"f64", "double"}};
    return backend == KernelBackend::METAL ? metal_spellings : cuda_spellings;
}

String Codegen::compile() {
    return compile(root);
}

String Codegen::compile(KernelNode *node) {
    switch (node->body.syntax_type) {
        case KernelNodeType::LITERAL:              return node->body.token;
        case KernelNodeType::IDENTIFIER:           return node->body.token;
        case KernelNodeType::EXPRESSION:           return expression_to_string(node);
        case KernelNodeType::BLOCK:                return block_to_string(node);
        case KernelNodeType::DEVICE_FUNCTION_IMPL: return device_function_impl_to_string(node);
        case KernelNodeType::FUNCTION_CALL:        return function_call_to_string(node);
        case KernelNodeType::KERNEL_FUNCTION_IMPL: return kernel_function_impl_to_string(node);
        case KernelNodeType::DECLARATION:          return declaration_to_string(node);
        case KernelNodeType::ASSIGNMENT:           return assignment_to_string(node);
        case KernelNodeType::CONDITIONAL:          return conditional_to_string(node);
        case KernelNodeType::CAST:                 return cast_to_string(node);
        case KernelNodeType::FOR:                  return for_to_string(node);
        case KernelNodeType::WHILE:                return while_to_string(node);
        case KernelNodeType::JUMP:                 return jump_to_string(node);
        case KernelNodeType::ATTRIBUTE:            return attribute_to_string(node);
        case KernelNodeType::PROGRAM:              return program_to_string(node);
        case KernelNodeType::CONDITIONAL_BRANCH:
        case KernelNodeType::PARAMETER:
        case KernelNodeType::PARAMETER_LIST:       break;
    }
    throw runtime_error("Kernel node cannot be emitted on its own");
}

String Codegen::compile_or_empty(KernelNode *node) {
    return node ? compile(node) : String();
}

// Words that are neutral type names are mapped; qualifiers and platform-only types pass through.
String Codegen::type_to_string(const String &type) const {
    const UnorderedMap<String, String> &spellings = kernel_type_spellings(backend);
    std::istringstream words(type);
    String word;
    String source;
    while (words >> word) {
        auto spelling = spellings.find(word);
        if (!source.empty()) source += " ";
        source += spelling == spellings.end() ? word : spelling->second;
    }
    return source;
}

String Codegen::program_to_string(KernelNode *node) {
    String source;
    for (KernelNode *item : static_cast<KernelListNode *>(node)->children) source += compile(item) + "\n\n";
    return source;
}

String Codegen::attribute_to_string(KernelNode *node) {
    if (backend != KernelBackend::METAL) throw runtime_error("Attribute '" + node->body.token + "' has no CUDA equivalent");
    return "[[ " + node->body.token + " ]]";
}

// The tree is the precedence, so operators are fully parenthesized.
String Codegen::expression_to_string(KernelNode *node) {
    const String &expression_operator = node->body.token;
    if (KernelBinaryNode *binary = dynamic_cast<KernelBinaryNode *>(node)) {
        if (expression_operator == "[]") return compile(binary->left) + "[" + compile(binary->right) + "]";
        if (expression_operator == "." || expression_operator == "->")
            return compile(binary->left) + expression_operator + compile(binary->right);
        return "(" + compile(binary->left) + " " + expression_operator + " " + compile(binary->right) + ")";
    }
    if (KernelUnaryNode *unary = dynamic_cast<KernelUnaryNode *>(node))
        return "(" + expression_operator + compile(unary->child) + ")";
    if (KernelTrinaryNode *ternary = dynamic_cast<KernelTrinaryNode *>(node))
        return "(" + compile(ternary->left) + " ? " + compile(ternary->middle) + " : " + compile(ternary->right) + ")";
    throw runtime_error("EXPRESSION node has no operands");
}

String Codegen::cast_to_string(KernelNode *node) {
    KernelUnaryNode *cast = static_cast<KernelUnaryNode *>(node);
    return "((" + type_to_string(cast->body.token) + ")" + compile(cast->child) + ")";
}

String Codegen::function_call_to_string(KernelNode *node) {
    String arguments;
    for (KernelNode *argument : static_cast<KernelListNode *>(node)->children) {
        if (!arguments.empty()) arguments += ", ";
        arguments += compile(argument);
    }
    return node->body.token + "(" + arguments + ")";
}

// The block adds the semicolons, so declarations and assignments can be reused in loop headers.
String Codegen::block_to_string(KernelNode *node) {
    static const Set<KernelNodeType> compound_statements = {
        KernelNodeType::BLOCK, KernelNodeType::CONDITIONAL, KernelNodeType::FOR, KernelNodeType::WHILE,
        KernelNodeType::DEVICE_FUNCTION_IMPL, KernelNodeType::KERNEL_FUNCTION_IMPL};

    String source = "{\n";
    indentation_depth += 1;
    for (KernelNode *statement : static_cast<KernelListNode *>(node)->children) {
        source += String(4 * indentation_depth, ' ') + compile(statement);
        if (compound_statements.count(statement->body.syntax_type) == 0) source += ";";
        source += "\n";
    }
    indentation_depth -= 1;
    return source + String(4 * indentation_depth, ' ') + "}";
}

String Codegen::declaration_to_string(KernelNode *node) {
    KernelBinaryNode *declaration = static_cast<KernelBinaryNode *>(node);
    String source = type_to_string(declaration->body.token) + " " + compile(declaration->left);
    if (declaration->right) source += " = " + compile(declaration->right);
    return source;
}

String Codegen::assignment_to_string(KernelNode *node) {
    KernelBinaryNode *assignment = static_cast<KernelBinaryNode *>(node);
    return compile(assignment->left) + " " + assignment->body.token + " " + compile(assignment->right);
}

String Codegen::conditional_to_string(KernelNode *node) {
    String source;
    for (KernelNode *branch_node : static_cast<KernelListNode *>(node)->children) {
        KernelBinaryNode *branch = static_cast<KernelBinaryNode *>(branch_node);
        if (!source.empty()) source += " else ";
        if (branch->left) source += "if (" + compile(branch->left) + ") ";
        source += compile(branch->right);
    }
    return source;
}

String Codegen::for_to_string(KernelNode *node) {
    const Vector<KernelNode *> &parts = static_cast<KernelListNode *>(node)->children;
    return "for (" + compile_or_empty(parts[0]) + "; " + compile_or_empty(parts[1]) + "; " +
           compile_or_empty(parts[2]) + ") " + compile(parts[3]);
}

String Codegen::while_to_string(KernelNode *node) {
    KernelBinaryNode *loop = static_cast<KernelBinaryNode *>(node);
    return "while (" + compile(loop->left) + ") " + compile(loop->right);
}

String Codegen::jump_to_string(KernelNode *node) {
    KernelUnaryNode *jump = static_cast<KernelUnaryNode *>(node);
    if (!jump->child) return jump->body.token;
    return jump->body.token + " " + compile(jump->child);
}

// Metal buffer indices come from position, so inserting a parameter never renumbers anything by hand.
String Codegen::parameters_to_string(KernelNode *node, bool number_buffers) {
    String source;
    s64 buffer_index = 0;
    for (KernelNode *parameter_node : static_cast<KernelListNode *>(node)->children) {
        KernelBinaryNode *parameter = static_cast<KernelBinaryNode *>(parameter_node);
        if (!source.empty()) source += ",";
        source += "\n    " + type_to_string(parameter->body.token) + " " + compile(parameter->left);
        if (parameter->right) {
            source += " " + compile(parameter->right);
        } else if (number_buffers && backend == KernelBackend::METAL) {
            source += " [[ buffer(" + std::to_string(buffer_index) + ") ]]";
            buffer_index += 1;
        }
    }
    return source + "\n";
}

String Codegen::device_function_impl_to_string(KernelNode *node) {
    KernelTrinaryNode *function = static_cast<KernelTrinaryNode *>(node);
    const String qualifier = backend == KernelBackend::METAL ? "inline " : "__device__ inline ";
    return qualifier + type_to_string(function->body.token) + " " + compile(function->left) +
           "(" + parameters_to_string(function->middle, false) + ") " + compile(function->right);
}

String Codegen::kernel_function_impl_to_string(KernelNode *node) {
    KernelTrinaryNode *function = static_cast<KernelTrinaryNode *>(node);
    const String qualifier = backend == KernelBackend::METAL ? "kernel " : "extern \"C\" __global__ ";
    return qualifier + type_to_string(function->body.token) + " " + compile(function->left) +
           "(" + parameters_to_string(function->middle, true) + ") " + compile(function->right);
}

}

