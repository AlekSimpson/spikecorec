#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_set>

#include "spikecorec/nml/master_kernel.h"

#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::nml {

const char *const MASTER_KERNEL_DRAIN_NAME = "spikecorec_master_drain_network_inputs";
const char *const MASTER_KERNEL_PROPAGATE_NAME = "spikecorec_master_propagate";

namespace {

void fail(const String &message) {
    log::throw_runtime_error(log::logger(), "master_kernel: " + message);
}

// std::visit helper matching the SAME idiom already used at every other std::visit site in this
// codebase (ir.cpp, gpu_source.cpp) -- keeps this file's visitor style consistent with the rest of
// the NML pipeline rather than introducing its own `if constexpr` alternative.
template <class... Alternatives>
struct Overloaded : Alternatives... {
    using Alternatives::operator()...;
};
template <class... Alternatives>
Overloaded(Alternatives...) -> Overloaded<Alternatives...>;

// ── emit-port discovery (a small, independent scan -- see master_kernel.h's own header comment
// for why this is intentionally NOT sourced from gpu_source.cpp's internal bookkeeping the way the
// per-neuron-function parameter list is: a mismatch here only means "propagate looks at the wrong
// port name," easily caught by a test, not a silent argument-order/memory-corruption risk) ────────

void collect_emit_ports_recursive(const Vector<TickInstruction> &instructions, Vector<String> &ports_in_order,
                                   unordered_set<String> &seen) {
    for (const auto &instruction : instructions) {
        std::visit(
            Overloaded{
                [&](const EmitInstruction &emit_instruction) {
                    if (seen.insert(emit_instruction.port_name).second) {
                        ports_in_order.push_back(emit_instruction.port_name);
                    }
                },
                [&](const IfInstruction &if_instruction) {
                    collect_emit_ports_recursive(if_instruction.then_body, ports_in_order, seen);
                    for (const auto &branch : if_instruction.else_if_branches) {
                        collect_emit_ports_recursive(branch.body, ports_in_order, seen);
                    }
                    if (if_instruction.else_body.has_value()) {
                        collect_emit_ports_recursive(*if_instruction.else_body, ports_in_order, seen);
                    }
                },
                [&](const ForAllInstruction &for_all_instruction) {
                    collect_emit_ports_recursive(for_all_instruction.body, ports_in_order, seen);
                },
                [&](const OnEventInstruction &on_event_instruction) {
                    collect_emit_ports_recursive(on_event_instruction.body, ports_in_order, seen);
                },
                [&](const auto &) {},
            },
            instruction.operation);
    }
}

Vector<String> collect_emit_ports_from_program(const IrProgram &program) {
    Vector<String> ports_in_order;
    unordered_set<String> seen;
    Vector<const Vector<TickInstruction> *> stages = {
        &program.tick.integrate, &program.tick.detect,     &program.tick.emit,       &program.tick.reset,
        &program.tick.propagate, &program.tick.plasticity, &program.tick.record,
    };
    for (const auto *stage : stages) collect_emit_ports_recursive(*stage, ports_in_order, seen);
    return ports_in_order;
}

} // namespace

// ── assembly ─────────────────────────────────────────────────────────────────────────────────

namespace {

GpuSource build_drain_kernel_gpu_source() {
    GpuSource source;
    source.msl_source =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "\n"
        "kernel void spikecorec_master_drain_network_inputs(\n"
        "    device float *network_inputs [[ buffer(0) ]],\n"
        "    constant long &neuron_count [[ buffer(1) ]],\n"
        "    uint thread_id [[ thread_position_in_grid ]]\n"
        ") {\n"
        "    long neuron_index = (long)thread_id;\n"
        "    if (neuron_index >= neuron_count) return;\n"
        "    network_inputs[neuron_index] = 0.0f;\n"
        "}\n";
    source.cuda_source =
        "__global__ void spikecorec_master_drain_network_inputs(\n"
        "    float *network_inputs,\n"
        "    long long neuron_count\n"
        ") {\n"
        "    long long neuron_index = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (neuron_index >= neuron_count) return;\n"
        "    network_inputs[neuron_index] = 0.0f;\n"
        "}\n";
    source.functions = {GpuFunctionSignature{MASTER_KERNEL_DRAIN_NAME, {"network_inputs", "neuron_count"}}};
    return source;
}

// The fixed engine-owned k^2-tree propagate/scatter + active-set-enqueue stage (arch §2's stage 6
// + stage 9's enqueue bookkeeping) -- reproduces exactly what the CURRENT hardcoded `step`/
// `step_kernel` already does for spike propagation (src/metal/kernels.metal, src/cuda/kernels.cu),
// minus the STDP Hebbian update (stage 7, not this ticket's scope) and minus reading/writing
// membrane potential (now the per-type `_tick` kernel's own job). Consumes `emit_<port>` flags a
// population's `_tick` kernel set this tick (ticket #55's `emit` convention) instead of a
// hardcoded threshold check. Uses gpu_source.cpp's own `k2t_find_nth_neighbor` slot-query loop
// (reused verbatim via k2tree_walk_preamble_msl/cuda) rather than `step`'s more elaborate
// single-DFS-walk `k2t_next_neighbor` helper -- an O(D·H) vs O(D²·H) difference for a spiking
// neuron's own out-degree D, matching the SAME simplification ticket #55's own `forall` lowering
// already accepted (gpu_source.h), not a new one this ticket introduces.
GpuSource build_propagate_kernel_gpu_source() {
    GpuSource source;

    String msl_body =
        "kernel void spikecorec_master_propagate(\n"
        "    constant long        &tick                       [[ buffer(0) ]],\n"
        "    constant long        &next_tick                  [[ buffer(1) ]],\n"
        "    const device float4  *U                          [[ buffer(2) ]],\n"
        "    const device float4  *V                          [[ buffer(3) ]],\n"
        "    constant long        &rank_float4_stride         [[ buffer(4) ]],\n"
        "    constant float       &constant_weight            [[ buffer(5) ]],\n"
        "    constant int         &using_constant_weight      [[ buffer(6) ]],\n"
        "    const device uint    *internal_node_words        [[ buffer(7) ]],\n"
        "    const device uint    *leaf_node_words             [[ buffer(8) ]],\n"
        "    const device uint    *rank_superblock_table      [[ buffer(9) ]],\n"
        "    const device ushort  *rank_subblock_table        [[ buffer(10) ]],\n"
        "    constant int         &branching_factor           [[ buffer(11) ]],\n"
        "    constant int         &superblock_size_words      [[ buffer(12) ]],\n"
        "    constant int         &padded_node_count          [[ buffer(13) ]],\n"
        "    constant int         &tree_height                [[ buffer(14) ]],\n"
        "    constant int         &internal_bit_count         [[ buffer(15) ]],\n"
        "    constant long        &neuron_count               [[ buffer(16) ]],\n"
        "    constant long        &max_neighbor_count         [[ buffer(17) ]],\n"
        "    device float         *network_inputs             [[ buffer(18) ]],\n"
        "    device long          *last_spiked                [[ buffer(19) ]],\n"
        "    device int           *next_active_neuron_indices [[ buffer(20) ]],\n"
        "    device int           *next_active_neuron_count   [[ buffer(21) ]],\n"
        "    device int           *active_generation          [[ buffer(22) ]],\n"
        "    device bool          *emit_spike                 [[ buffer(23) ]],\n"
        "    uint thread_id [[ thread_position_in_grid ]]\n"
        ") {\n"
        "    long neuron_index = (long)thread_id;\n"
        "    if (neuron_index >= neuron_count) return;\n"
        "    if (!emit_spike[neuron_index]) return;\n"
        "    emit_spike[neuron_index] = false;\n"
        "\n"
        "    last_spiked[neuron_index] = tick;\n"
        "    int next_tick_i = (int)next_tick;\n"
        "\n"
        "    const device float4 *u_row = U + (long)neuron_index * rank_float4_stride;\n"
        "\n"
        "    for (long slot = 0; slot < max_neighbor_count; ++slot) {\n"
        "        int child = k2t_find_nth_neighbor(\n"
        "            internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,\n"
        "            branching_factor, superblock_size_words, (int)neuron_count, padded_node_count,\n"
        "            tree_height, internal_bit_count, (int)neuron_index, (int)slot\n"
        "        );\n"
        "        if (child < 0) continue;\n"
        "\n"
        "        float weight = constant_weight;\n"
        "        if (using_constant_weight == 0) {\n"
        "            const device float4 *v_row = V + (long)child * rank_float4_stride;\n"
        "            float dot_product = 0.0f;\n"
        "            for (long lane = 0; lane < rank_float4_stride; ++lane) {\n"
        "                dot_product += dot(u_row[lane], v_row[lane]);\n"
        "            }\n"
        "            weight = dot_product;\n"
        "        }\n"
        "\n"
        "        device atomic_float *input_slot = (device atomic_float *)(network_inputs + child);\n"
        "        atomic_fetch_add_explicit(input_slot, weight, memory_order_relaxed);\n"
        "\n"
        "        device atomic_int *child_generation_slot = (device atomic_int *)(active_generation + child);\n"
        "        int previous_child_generation =\n"
        "            atomic_exchange_explicit(child_generation_slot, next_tick_i, memory_order_relaxed);\n"
        "        if (previous_child_generation != next_tick_i) {\n"
        "            device atomic_int *next_count_slot = (device atomic_int *)next_active_neuron_count;\n"
        "            int position = atomic_fetch_add_explicit(next_count_slot, 1, memory_order_relaxed);\n"
        "            next_active_neuron_indices[position] = child;\n"
        "        }\n"
        "    }\n"
        "\n"
        "    device atomic_int *self_generation_slot = (device atomic_int *)(active_generation + neuron_index);\n"
        "    int previous_self_generation =\n"
        "        atomic_exchange_explicit(self_generation_slot, next_tick_i, memory_order_relaxed);\n"
        "    if (previous_self_generation != next_tick_i) {\n"
        "        device atomic_int *next_count_slot = (device atomic_int *)next_active_neuron_count;\n"
        "        int position = atomic_fetch_add_explicit(next_count_slot, 1, memory_order_relaxed);\n"
        "        next_active_neuron_indices[position] = (int)neuron_index;\n"
        "    }\n"
        "}\n";

    source.msl_source = "#include <metal_stdlib>\nusing namespace metal;\n" + k2tree_walk_preamble_msl() + "\n" + msl_body;

    String cuda_body =
        "__global__ void spikecorec_master_propagate(\n"
        "    long long             tick,\n"
        "    long long             next_tick,\n"
        "    const float4          *U,\n"
        "    const float4          *V,\n"
        "    long long             rank_float4_stride,\n"
        "    float                 constant_weight,\n"
        "    int                   using_constant_weight,\n"
        "    const unsigned int    *internal_node_words,\n"
        "    const unsigned int    *leaf_node_words,\n"
        "    const unsigned int    *rank_superblock_table,\n"
        "    const unsigned short  *rank_subblock_table,\n"
        "    int                   branching_factor,\n"
        "    int                   superblock_size_words,\n"
        "    int                   padded_node_count,\n"
        "    int                   tree_height,\n"
        "    int                   internal_bit_count,\n"
        "    long long             neuron_count,\n"
        "    long long             max_neighbor_count,\n"
        "    float                 *network_inputs,\n"
        "    long long             *last_spiked,\n"
        "    int                   *next_active_neuron_indices,\n"
        "    int                   *next_active_neuron_count,\n"
        "    int                   *active_generation,\n"
        "    bool                  *emit_spike\n"
        ") {\n"
        "    long long neuron_index = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (neuron_index >= neuron_count) return;\n"
        "    if (!emit_spike[neuron_index]) return;\n"
        "    emit_spike[neuron_index] = false;\n"
        "\n"
        "    last_spiked[neuron_index] = tick;\n"
        "    int next_tick_i = (int)next_tick;\n"
        "\n"
        "    const float4 *u_row = U + (long long)neuron_index * rank_float4_stride;\n"
        "\n"
        "    for (long long slot = 0; slot < max_neighbor_count; ++slot) {\n"
        "        int child = k2t_find_nth_neighbor(\n"
        "            internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,\n"
        "            branching_factor, superblock_size_words, (int)neuron_count, padded_node_count,\n"
        "            tree_height, internal_bit_count, (int)neuron_index, (int)slot\n"
        "        );\n"
        "        if (child < 0) continue;\n"
        "\n"
        "        float weight = constant_weight;\n"
        "        if (using_constant_weight == 0) {\n"
        "            const float4 *v_row = V + (long long)child * rank_float4_stride;\n"
        "            float dot_product = 0.0f;\n"
        "            for (long long lane = 0; lane < rank_float4_stride; ++lane) {\n"
        "                float4 u4 = u_row[lane];\n"
        "                float4 v4 = v_row[lane];\n"
        "                dot_product += u4.x * v4.x + u4.y * v4.y + u4.z * v4.z + u4.w * v4.w;\n"
        "            }\n"
        "            weight = dot_product;\n"
        "        }\n"
        "\n"
        "        atomicAdd(&network_inputs[child], weight);\n"
        "\n"
        "        int previous_child_generation = atomicExch(&active_generation[child], next_tick_i);\n"
        "        if (previous_child_generation != next_tick_i) {\n"
        "            int position = atomicAdd(next_active_neuron_count, 1);\n"
        "            next_active_neuron_indices[position] = child;\n"
        "        }\n"
        "    }\n"
        "\n"
        "    int previous_self_generation = atomicExch(&active_generation[neuron_index], next_tick_i);\n"
        "    if (previous_self_generation != next_tick_i) {\n"
        "        int position = atomicAdd(next_active_neuron_count, 1);\n"
        "        next_active_neuron_indices[position] = (int)neuron_index;\n"
        "    }\n"
        "}\n";

    source.cuda_source = "#include <vector_types.h>\n" + k2tree_walk_preamble_cuda() + "\n" + cuda_body;

    source.functions = {GpuFunctionSignature{
        MASTER_KERNEL_PROPAGATE_NAME,
        {"tick", "next_tick", "U", "V", "rank_float4_stride", "constant_weight", "using_constant_weight",
         "internal_node_words", "leaf_node_words", "rank_superblock_table", "rank_subblock_table",
         "branching_factor", "superblock_size_words", "padded_node_count", "tree_height", "internal_bit_count",
         "neuron_count", "max_neighbor_count", "network_inputs", "last_spiked", "next_active_neuron_indices",
         "next_active_neuron_count", "active_generation", "emit_spike"}}};
    return source;
}

} // namespace

// ── ticket #64 [F3]: ring-based deliver-drain/propagate (the "delay ring") ──────────────────────
//
// Only assembled/compiled when the model needs a delay ring (SpikeEngine's own NML-mode
// constructor, src/core/engine.cpp). Same fixed-stage role as the two kernels above (deliver-drain / k^2-tree
// propagate+scatter+active-set-enqueue), generalized so a scattered spike lands in the ring slot
// due at `tick + delay_ticks` instead of always the very next tick -- see delay_ring.h for the ring
// design (slot math, why ring_slot_count = max_delay_ticks + 1, and the per-slot pending-active
// dedup). Reuses the SAME k2t_find_nth_neighbor walk / U-V-or-constant weight reconstruction the
// non-ring propagate kernel above already does -- this ticket's own scope is real per-edge DELAY,
// not real per-edge weight (still #52-54/#57's job, unchanged here).
//
// Exported (not file-local like the two kernel builders above) so master_kernel_tests.cpp can
// genuinely compile their exact MSL text via the real `xcrun -sdk macosx metal -c` toolchain
// directly, mirroring gpu_source_tests.cpp's/this file's own `compiles_as_msl` helper -- the same
// scrutiny AssembledMasterKernelSource's public drain_network_inputs_source/propagate_source
// fields already get, without adding these two ticket-#64-only kernels to that struct's own
// established (and already-tested) contract.

namespace {
const char *const MASTER_KERNEL_DRAIN_RING_NAME = "spikecorec_master_drain_network_inputs_ring";
const char *const MASTER_KERNEL_PROPAGATE_RING_NAME = "spikecorec_master_propagate_ring";
} // namespace

GpuSource build_drain_ring_kernel_gpu_source() {
    GpuSource source;
    source.msl_source =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "\n"
        "kernel void spikecorec_master_drain_network_inputs_ring(\n"
        "    device float *network_inputs_ring [[ buffer(0) ]],\n"
        "    constant long &neuron_count       [[ buffer(1) ]],\n"
        "    constant long &current_slot       [[ buffer(2) ]],\n"
        "    uint thread_id [[ thread_position_in_grid ]]\n"
        ") {\n"
        "    long neuron_index = (long)thread_id;\n"
        "    if (neuron_index >= neuron_count) return;\n"
        "    network_inputs_ring[current_slot * neuron_count + neuron_index] = 0.0f;\n"
        "}\n";
    source.cuda_source =
        "__global__ void spikecorec_master_drain_network_inputs_ring(\n"
        "    float *network_inputs_ring,\n"
        "    long long neuron_count,\n"
        "    long long current_slot\n"
        ") {\n"
        "    long long neuron_index = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (neuron_index >= neuron_count) return;\n"
        "    network_inputs_ring[current_slot * neuron_count + neuron_index] = 0.0f;\n"
        "}\n";
    source.functions = {
        GpuFunctionSignature{MASTER_KERNEL_DRAIN_RING_NAME, {"network_inputs_ring", "neuron_count", "current_slot"}}};
    return source;
}

GpuSource build_propagate_ring_kernel_gpu_source() {
    GpuSource source;

    String msl_body =
        "kernel void spikecorec_master_propagate_ring(\n"
        "    constant long        &tick                       [[ buffer(0) ]],\n"
        "    constant long        &ring_slot_count             [[ buffer(1) ]],\n"
        "    const device float4  *U                          [[ buffer(2) ]],\n"
        "    const device float4  *V                          [[ buffer(3) ]],\n"
        "    constant long        &rank_float4_stride         [[ buffer(4) ]],\n"
        "    constant float       &constant_weight            [[ buffer(5) ]],\n"
        "    constant int         &using_constant_weight      [[ buffer(6) ]],\n"
        "    const device uint    *internal_node_words        [[ buffer(7) ]],\n"
        "    const device uint    *leaf_node_words             [[ buffer(8) ]],\n"
        "    const device uint    *rank_superblock_table      [[ buffer(9) ]],\n"
        "    const device ushort  *rank_subblock_table        [[ buffer(10) ]],\n"
        "    constant int         &branching_factor           [[ buffer(11) ]],\n"
        "    constant int         &superblock_size_words      [[ buffer(12) ]],\n"
        "    constant int         &padded_node_count          [[ buffer(13) ]],\n"
        "    constant int         &tree_height                [[ buffer(14) ]],\n"
        "    constant int         &internal_bit_count         [[ buffer(15) ]],\n"
        "    constant long        &neuron_count               [[ buffer(16) ]],\n"
        "    constant long        &max_neighbor_count         [[ buffer(17) ]],\n"
        "    const device int     *edge_delay_ticks           [[ buffer(18) ]],\n"
        "    device float         *network_inputs_ring        [[ buffer(19) ]],\n"
        "    device long          *last_spiked                [[ buffer(20) ]],\n"
        "    device int           *pending_active_neuron_indices [[ buffer(21) ]],\n"
        "    device int           *pending_active_neuron_count   [[ buffer(22) ]],\n"
        "    device int           *pending_active_generation     [[ buffer(23) ]],\n"
        "    device bool          *emit_spike                 [[ buffer(24) ]],\n"
        "    uint thread_id [[ thread_position_in_grid ]]\n"
        ") {\n"
        "    long neuron_index = (long)thread_id;\n"
        "    if (neuron_index >= neuron_count) return;\n"
        "    if (!emit_spike[neuron_index]) return;\n"
        "    emit_spike[neuron_index] = false;\n"
        "\n"
        "    last_spiked[neuron_index] = tick;\n"
        "\n"
        "    const device float4 *u_row = U + (long)neuron_index * rank_float4_stride;\n"
        "\n"
        "    for (long slot = 0; slot < max_neighbor_count; ++slot) {\n"
        "        int child = k2t_find_nth_neighbor(\n"
        "            internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,\n"
        "            branching_factor, superblock_size_words, (int)neuron_count, padded_node_count,\n"
        "            tree_height, internal_bit_count, (int)neuron_index, (int)slot\n"
        "        );\n"
        "        if (child < 0) continue;\n"
        "\n"
        "        float weight = constant_weight;\n"
        "        if (using_constant_weight == 0) {\n"
        "            const device float4 *v_row = V + (long)child * rank_float4_stride;\n"
        "            float dot_product = 0.0f;\n"
        "            for (long lane = 0; lane < rank_float4_stride; ++lane) {\n"
        "                dot_product += dot(u_row[lane], v_row[lane]);\n"
        "            }\n"
        "            weight = dot_product;\n"
        "        }\n"
        "\n"
        "        int delay_ticks = edge_delay_ticks[neuron_index * max_neighbor_count + slot];\n"
        "        long arrival_tick = tick + (long)delay_ticks;\n"
        "        long target_slot = arrival_tick % ring_slot_count;\n"
        "        int arrival_tick_i = (int)arrival_tick;\n"
        "\n"
        "        device atomic_float *input_slot =\n"
        "            (device atomic_float *)(network_inputs_ring + target_slot * neuron_count + child);\n"
        "        atomic_fetch_add_explicit(input_slot, weight, memory_order_relaxed);\n"
        "\n"
        "        device atomic_int *child_generation_slot =\n"
        "            (device atomic_int *)(pending_active_generation + target_slot * neuron_count + child);\n"
        "        int previous_child_generation =\n"
        "            atomic_exchange_explicit(child_generation_slot, arrival_tick_i, memory_order_relaxed);\n"
        "        if (previous_child_generation != arrival_tick_i) {\n"
        "            device atomic_int *count_slot = (device atomic_int *)(pending_active_neuron_count + target_slot);\n"
        "            int position = atomic_fetch_add_explicit(count_slot, 1, memory_order_relaxed);\n"
        "            pending_active_neuron_indices[target_slot * neuron_count + position] = child;\n"
        "        }\n"
        "    }\n"
        "\n"
        "    long self_slot = (tick + 1) % ring_slot_count;\n"
        "    int self_arrival_i = (int)(tick + 1);\n"
        "    device atomic_int *self_generation_slot =\n"
        "        (device atomic_int *)(pending_active_generation + self_slot * neuron_count + neuron_index);\n"
        "    int previous_self_generation =\n"
        "        atomic_exchange_explicit(self_generation_slot, self_arrival_i, memory_order_relaxed);\n"
        "    if (previous_self_generation != self_arrival_i) {\n"
        "        device atomic_int *count_slot = (device atomic_int *)(pending_active_neuron_count + self_slot);\n"
        "        int position = atomic_fetch_add_explicit(count_slot, 1, memory_order_relaxed);\n"
        "        pending_active_neuron_indices[self_slot * neuron_count + position] = (int)neuron_index;\n"
        "    }\n"
        "}\n";

    source.msl_source = "#include <metal_stdlib>\nusing namespace metal;\n" + k2tree_walk_preamble_msl() + "\n" + msl_body;

    String cuda_body =
        "__global__ void spikecorec_master_propagate_ring(\n"
        "    long long             tick,\n"
        "    long long             ring_slot_count,\n"
        "    const float4          *U,\n"
        "    const float4          *V,\n"
        "    long long             rank_float4_stride,\n"
        "    float                 constant_weight,\n"
        "    int                   using_constant_weight,\n"
        "    const unsigned int    *internal_node_words,\n"
        "    const unsigned int    *leaf_node_words,\n"
        "    const unsigned int    *rank_superblock_table,\n"
        "    const unsigned short  *rank_subblock_table,\n"
        "    int                   branching_factor,\n"
        "    int                   superblock_size_words,\n"
        "    int                   padded_node_count,\n"
        "    int                   tree_height,\n"
        "    int                   internal_bit_count,\n"
        "    long long             neuron_count,\n"
        "    long long             max_neighbor_count,\n"
        "    const int             *edge_delay_ticks,\n"
        "    float                 *network_inputs_ring,\n"
        "    long long             *last_spiked,\n"
        "    int                   *pending_active_neuron_indices,\n"
        "    int                   *pending_active_neuron_count,\n"
        "    int                   *pending_active_generation,\n"
        "    bool                  *emit_spike\n"
        ") {\n"
        "    long long neuron_index = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (neuron_index >= neuron_count) return;\n"
        "    if (!emit_spike[neuron_index]) return;\n"
        "    emit_spike[neuron_index] = false;\n"
        "\n"
        "    last_spiked[neuron_index] = tick;\n"
        "\n"
        "    const float4 *u_row = U + (long long)neuron_index * rank_float4_stride;\n"
        "\n"
        "    for (long long slot = 0; slot < max_neighbor_count; ++slot) {\n"
        "        int child = k2t_find_nth_neighbor(\n"
        "            internal_node_words, leaf_node_words, rank_superblock_table, rank_subblock_table,\n"
        "            branching_factor, superblock_size_words, (int)neuron_count, padded_node_count,\n"
        "            tree_height, internal_bit_count, (int)neuron_index, (int)slot\n"
        "        );\n"
        "        if (child < 0) continue;\n"
        "\n"
        "        float weight = constant_weight;\n"
        "        if (using_constant_weight == 0) {\n"
        "            const float4 *v_row = V + (long long)child * rank_float4_stride;\n"
        "            float dot_product = 0.0f;\n"
        "            for (long long lane = 0; lane < rank_float4_stride; ++lane) {\n"
        "                float4 u4 = u_row[lane];\n"
        "                float4 v4 = v_row[lane];\n"
        "                dot_product += u4.x * v4.x + u4.y * v4.y + u4.z * v4.z + u4.w * v4.w;\n"
        "            }\n"
        "            weight = dot_product;\n"
        "        }\n"
        "\n"
        "        int delay_ticks = edge_delay_ticks[neuron_index * max_neighbor_count + slot];\n"
        "        long long arrival_tick = tick + (long long)delay_ticks;\n"
        "        long long target_slot = arrival_tick % ring_slot_count;\n"
        "        int arrival_tick_i = (int)arrival_tick;\n"
        "\n"
        "        atomicAdd(&network_inputs_ring[target_slot * neuron_count + child], weight);\n"
        "\n"
        "        int previous_child_generation =\n"
        "            atomicExch(&pending_active_generation[target_slot * neuron_count + child], arrival_tick_i);\n"
        "        if (previous_child_generation != arrival_tick_i) {\n"
        "            int position = atomicAdd(&pending_active_neuron_count[target_slot], 1);\n"
        "            pending_active_neuron_indices[target_slot * neuron_count + position] = child;\n"
        "        }\n"
        "    }\n"
        "\n"
        "    long long self_slot = (tick + 1) % ring_slot_count;\n"
        "    int self_arrival_i = (int)(tick + 1);\n"
        "    int previous_self_generation =\n"
        "        atomicExch(&pending_active_generation[self_slot * neuron_count + neuron_index], self_arrival_i);\n"
        "    if (previous_self_generation != self_arrival_i) {\n"
        "        int position = atomicAdd(&pending_active_neuron_count[self_slot], 1);\n"
        "        pending_active_neuron_indices[self_slot * neuron_count + position] = (int)neuron_index;\n"
        "    }\n"
        "}\n";

    source.cuda_source = "#include <vector_types.h>\n" + k2tree_walk_preamble_cuda() + "\n" + cuda_body;

    source.functions = {GpuFunctionSignature{
        MASTER_KERNEL_PROPAGATE_RING_NAME,
        {"tick", "ring_slot_count", "U", "V", "rank_float4_stride", "constant_weight", "using_constant_weight",
         "internal_node_words", "leaf_node_words", "rank_superblock_table", "rank_subblock_table",
         "branching_factor", "superblock_size_words", "padded_node_count", "tree_height", "internal_bit_count",
         "neuron_count", "max_neighbor_count", "edge_delay_ticks", "network_inputs_ring", "last_spiked",
         "pending_active_neuron_indices", "pending_active_neuron_count", "pending_active_generation",
         "emit_spike"}}};
    return source;
}

Vector<String> collect_emit_port_names(const ModelSpecification &model,
                                        const Vector<IrProgram> &type_library_ir_programs) {
    Vector<String> ports_in_order;
    unordered_set<String> seen;
    for (const auto &population : model.populations) {
        if (population.type_library_index < 0 ||
            (usize)population.type_library_index >= type_library_ir_programs.size()) {
            continue;
        }
        Vector<String> program_ports =
            collect_emit_ports_from_program(type_library_ir_programs[(usize)population.type_library_index]);
        for (const auto &port : program_ports) {
            if (seen.insert(port).second) ports_in_order.push_back(port);
        }
    }
    return ports_in_order;
}

AssembledMasterKernelSource assemble_master_kernel_source(const ModelSpecification &model,
                                                           const Vector<IrProgram> &type_library_ir_programs) {
    if (type_library_ir_programs.size() != model.type_library.size()) {
        fail("type_library_ir_programs.size() (" + std::to_string(type_library_ir_programs.size()) +
             ") must match model.type_library.size() (" + std::to_string(model.type_library.size()) + ")");
    }

    AssembledMasterKernelSource assembled;
    assembled.population_gpu_sources.reserve(model.populations.size());
    for (const auto &population : model.populations) {
        if (population.type_library_index < 0 ||
            (usize)population.type_library_index >= type_library_ir_programs.size()) {
            fail("population '" + population.id + "' has an out-of-range type_library_index (" +
                 std::to_string(population.type_library_index) + ")");
        }
        const IrProgram &program = type_library_ir_programs[(usize)population.type_library_index];
        assembled.population_gpu_sources.push_back(lower_ir_program_to_gpu_source(program));
    }

    assembled.drain_network_inputs_source = build_drain_kernel_gpu_source();
    assembled.propagate_source = build_propagate_kernel_gpu_source();
    return assembled;
}

// ── compile + cache + dispatch ──────────────────────────────────────────────────────────────────
//
// (see master_kernel.h's own doc comment for compile_kernel_or_throw_with_source)

KernelHandle compile_kernel_or_throw_with_source(const String &source_text, const String &function_name,
                                                  const String &kernel_label, const String &ir_dump) {
    try {
        return compile_kernel(source_text.c_str(), function_name.c_str());
    } catch (const std::exception &compile_error) {
        String message = "master_kernel: compile_kernel failed for " + kernel_label + " (kernel function '" +
                          function_name + "'): " + compile_error.what() +
                          "\n--- generated GPU source that failed to compile ---\n" + source_text;
        if (!ir_dump.empty()) {
            message += "\n--- IR program ('.alloc'/'.tick') that produced it ---\n" + ir_dump;
        }
        log::throw_runtime_error(log::logger(), message);
    }
}

// The AssembledModel class that used to live here (compile+cache+dispatch runtime for an assembled
// model, plus ticket #131's spike-scatter batch-construction subsystem and ticket #132's real STDP
// support) has been folded into SpikeEngine's own NML-mode constructor/step_simulation -- see
// include/spikecorec/core/engine.h's "Stage 1"/"Stage 2 of folding nml::AssembledModel into
// SpikeEngine" doc comments and src/core/engine.cpp for the code that runs today.

} // namespace spikecorec::nml
