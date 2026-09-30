#include "spikecorec/nml/dynamics.h"

#include "spikecorec/core/backend.h"
#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"
#include "spikecorec/nml/node.h"
#include "spikecorec/nml/parser.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

void Codegen::allocate_cell_model_memory() {
    const NML_Context::NML_SimulationContext &simulation = context.simulation;

    device->partition(sizeof(f32) * simulation.cell_state_length, EngineDatatype::FLOAT32, data_partitions)
        .partition(sizeof(s64) * 2 * simulation.total_neuron_count, EngineDatatype::SIGNED64, data_partitions)
       .partition(sizeof(u8) * (simulation.maximum_edge_delay + 1) * simulation.total_neuron_count,
                  EngineDatatype::UNSIGNED8, data_partitions)
       .partition(sizeof(s64) * simulation.total_neuron_count, EngineDatatype::SIGNED64, data_partitions)
       .partition(sizeof(f32), EngineDatatype::FLOAT32, data_partitions);

    EnginePointer root = device->allocate(data_partitions);

    data_partitions.push_back(root);

    logger->debug("SpikeEngine: {} bytes — cell_state {}, network_inputs {}, spike_history {}, "
                  root.total_bytes, simulation.cell_state_length, 2 * simulation.total_neuron_count,
                  (simulation.maximum_edge_delay+1) * simulation.total_neuron_count,
                  simulation.total_neuron_count);
}

// AST generation

KernelNode *Codegen::kernel_root() {
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

String Codegen::compile() {
    switch (root->body.syntax_type) {
        case LITERAL:
            return literal_to_string(root);
        case IDENTIFIER:
            return identifier_to_string(root);
        case EXPRESSION:
            return expression_to_string(root);
        case BLOCK:
            return block_to_string(root);
        case DEVICE_FUNCTION_IMPL:
            return device_function_impl_to_string(root);
        case FUNCTION_CALL:
            return function_call_to_string(root);
        case KERNEL_FUNCTION_IMPL:
            return kernel_function_impl_to_string(root);
        default:
            break;
    }

    return "";
}

String Codegen::compile(KernelNode *node) {
    switch (node->body.syntax_type) {
        case LITERAL:
            return literal_to_string(node);
        case IDENTIFIER:
            return identifier_to_string(node);
        case EXPRESSION:
            return expression_to_string(node);
        case BLOCK:
            return block_to_string(node);
        case DEVICE_FUNCTION_IMPL:
            return device_function_impl_to_string(node);
        case FUNCTION_CALL:
            return function_call_to_string(node);
        case KERNEL_FUNCTION_IMPL:
            return kernel_function_impl_to_string(node);
        default:
            break;
    }

    return "";
}

String literal_to_string(KernelNode *node) {}
String identifier_to_string(KernelNode *node) {}
String expression_to_string(KernelNode *node) {}
String block_to_string(KernelNode *node) {}
String device_function_impl_to_string(KernelNode *node) {}
String function_call_to_string(KernelNode *node) {}
String kernel_function_impl(KernelNode *node) {}

}

