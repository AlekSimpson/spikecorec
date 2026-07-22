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

// Position (0-based) of `state_name` among `program.alloc`'s StateDirective entries, in
// declaration order. MUST match allocate_model's own enumeration order (allocator.cpp iterates
// `program.alloc` the same way to compute state_variable_count) -- this fixes each state
// variable's SoA sub-offset within a population's cell_state chunk (arch §4.1). Both this function
// and allocate_model iterate the exact same Vector<AllocDirective> in the exact same order, so
// there is no independent classification to drift out of sync (unlike gpu_source.cpp's parameter
// list, which additionally bakes/dedups/detects RNG-and-walk-need -- see master_kernel.h).
s32 state_variable_offset(const IrProgram &program, const String &state_name) {
    s32 offset = 0;
    for (const auto &directive : program.alloc) {
        if (const auto *state = std::get_if<StateDirective>(&directive)) {
            if (state->name == state_name) return offset;
            ++offset;
        }
    }
    fail("'" + state_name + "' is not a StateDirective of program '" + program.component_type_name + "'");
    return -1;
}

// ── dispatch argument construction ──────────────────────────────────────────────────────────────
//
// Builds the args[]/arg_sizes[] pair metal_dispatch/cuda_dispatch expect, owning stable storage
// for every argument's bytes for the lifetime of this builder (matching the pattern every existing
// kernel wrapper in backend.cpp already uses: a handful of locals + `const void *args[] = {&a, &b,
// ...}`, generalized here to an arbitrary, runtime-determined argument count).
class DispatchArgumentBuilder {
public:
    void add_pointer(const void *pointer) { add_bytes(&pointer, sizeof(void *)); }
    void add_f32(f32 value) { add_bytes(&value, sizeof(f32)); }
    void add_s64(s64 value) { add_bytes(&value, sizeof(s64)); }
    void add_s32(s32 value) { add_bytes(&value, sizeof(s32)); }

    void dispatch(KernelHandle handle, LaunchConfig config) const {
        Vector<const void *> args(slots_.size());
        Vector<usize> sizes(slots_.size());
        for (usize index = 0; index < slots_.size(); ++index) {
            args[index] = slots_[index].data();
            sizes[index] = slots_[index].size();
        }
#ifdef SPIKECOREC_METAL
        metal_dispatch(handle, config, args.data(), sizes.data(), (u32)slots_.size());
#elif defined(SPIKECOREC_CUDA)
        cuda_dispatch(handle, config, args.data(), sizes.data(), (u32)slots_.size());
#endif
    }

private:
    void add_bytes(const void *value, usize size) {
        Vector<u8> storage(size);
        std::memcpy(storage.data(), value, size);
        slots_.push_back(std::move(storage));
    }

    Vector<Vector<u8>> slots_;
};

LaunchConfig launch_config_for(s64 element_count) {
    constexpr u32 threads_per_block = 256;
    if (element_count <= 0) return LaunchConfig{0, threads_per_block};
    u32 grid = (u32)((element_count + threads_per_block - 1) / threads_per_block);
    return LaunchConfig{grid, threads_per_block};
}

const String &source_text_for_this_backend(const GpuSource &source) {
#ifdef SPIKECOREC_CUDA
    return source.cuda_source;
#else
    return source.msl_source;
#endif
}

// Resolves one population `_tick` kernel parameter name to a dispatch argument, appended onto
// `builder` -- see master_kernel.h's own header comment for exactly which parameter kinds this
// supports (the ones Phase-1's GLIF-family cell lowering, cell_lowering.cpp, actually emits, plus
// ticket #65 [F4]'s `rng_state`) and which it deliberately still throws on
// (require/tree-walk/shared-basis/un-baked-bare-param).
void append_cell_tick_argument(DispatchArgumentBuilder &builder, const String &parameter_name,
                                const IrProgram &program, s32 type_library_index, s32 neuron_index_begin,
                                s32 population_size, s64 cell_state_chunk_base_offset, ModelAllocation &allocation,
                                f32 dt, s64 tick, f32 *network_inputs, u32 *rng_state,
                                const UnorderedMap<String, bool *> &emit_port_flags) {
    if (parameter_name == "dt") {
        builder.add_f32(dt);
        return;
    }
    if (parameter_name == "tick") {
        builder.add_s64(tick);
        return;
    }
    if (parameter_name == "network_inputs") {
        builder.add_pointer(network_inputs + neuron_index_begin);
        return;
    }
    if (parameter_name == "rng_state") {
        if (rng_state == nullptr) {
            fail("cell-type kernel parameter 'rng_state' (rand/randn) needs ModelRuntimeBuffers::rng_state "
                 "to be set (non-null) -- allocate + seed one u32 per neuron before calling step_tick "
                 "(see master_kernel.h)");
        }
        builder.add_pointer(rng_state + neuron_index_begin);
        return;
    }
    if (parameter_name == "neuron_count") {
        builder.add_s64((s64)population_size);
        return;
    }
    if (parameter_name.rfind("emit_", 0) == 0) {
        String port_name = parameter_name.substr(5);
        auto found = emit_port_flags.find(port_name);
        if (found == emit_port_flags.end()) {
            fail("no emit-port flag buffer supplied in ModelRuntimeBuffers::emit_port_flags for port '" +
                 port_name + "' (parameter '" + parameter_name + "')");
        }
        builder.add_pointer(found->second + neuron_index_begin);
        return;
    }

    for (const auto &directive : program.alloc) {
        if (const auto *state = std::get_if<StateDirective>(&directive)) {
            if (state->name != parameter_name) continue;
            s32 offset_within_type = state_variable_offset(program, parameter_name);
            f32 *base = allocation.cell_state.get_contents() + cell_state_chunk_base_offset +
                        (s64)offset_within_type * population_size;
            builder.add_pointer(base);
            return;
        }
        if (const auto *accum = std::get_if<AccumDirective>(&directive)) {
            if (accum->name != parameter_name) continue;
            auto found = allocation.accumulators.find(type_scoped_key(type_library_index, parameter_name));
            if (found == allocation.accumulators.end()) {
                fail("accum '" + parameter_name + "' has no allocated buffer in ModelAllocation::accumulators");
            }
            builder.add_pointer(found->second.get_contents() + neuron_index_begin);
            return;
        }
        if (const auto *param_dynamic = std::get_if<ParamDynamicDirective>(&directive)) {
            if (param_dynamic->name != parameter_name) continue;
            auto found = allocation.dynamic_parameter_arrays.find(type_scoped_key(type_library_index, parameter_name));
            if (found == allocation.dynamic_parameter_arrays.end()) {
                fail("param:dyn '" + parameter_name + "' has no allocated buffer in ModelAllocation::dynamic_parameter_arrays");
            }
            builder.add_pointer(found->second.get_contents() + neuron_index_begin);
            return;
        }
        if (const auto *regime = std::get_if<RegimeDirective>(&directive)) {
            if (regime->name != parameter_name) continue;
            if (!allocation.has_regime_index) fail("regime '" + parameter_name + "' has no allocated regime_indices buffer");
            s32 *base = allocation.regime_indices.get_contents() + neuron_index_begin;
            builder.add_pointer(base);
            return;
        }
        if (const auto *expose = std::get_if<ExposeDirective>(&directive)) {
            if (expose->name != parameter_name) continue;
            auto found = allocation.derived_exposure_scratch_buffers.find(type_scoped_key(type_library_index, parameter_name));
            if (found == allocation.derived_exposure_scratch_buffers.end()) {
                fail("expose '" + parameter_name + "' has no derived-exposure-scratch buffer (and no matching "
                     "state slot resolved it first) -- ModelAllocation::derived_exposure_scratch_buffers");
            }
            builder.add_pointer(found->second.get_contents() + neuron_index_begin);
            return;
        }
        if (const auto *require = std::get_if<RequireDirective>(&directive)) {
            if (require->name != parameter_name) continue;
            fail("cell-type kernel parameter '" + parameter_name + "' resolves to a `require` binding -- "
                 "not supported by this ticket's per-neuron dispatch (out of Phase-1 GLIF-cell scope; "
                 "see master_kernel.h)");
        }
        if (const auto *param_constant = std::get_if<ParamConstantDirective>(&directive)) {
            if (param_constant->name != parameter_name) continue;
            fail("cell-type kernel parameter '" + parameter_name + "' is an un-baked (bare) `param` -- "
                 "Phase-1's allocate_model has no established value source for this case (it only logs a "
                 "warning, see allocator.cpp), so this ticket's dispatch cannot resolve it either");
        }
    }

    fail("cell-type kernel parameter '" + parameter_name + "' is not a recognized reserved name, .alloc name, "
         "or emit port for this ticket's dispatch -- likely the k^2-tree-walk/shared-basis block "
         "(forall/loadedge/accedge), out of scope for a Phase-1 GLIF-family cell (see master_kernel.h)");
}

// ── ticket #131: spike-scatter batch-construction subsystem -- dispatch-argument resolution ────────
//
// Resolves a `require <name> from postsynaptic` binding (gpu_source.cpp's `name[target_node]`) to a
// real pointer into the postsynaptic cell's own packed `cell_state` chunk, offset so that indexing
// it with a GLOBAL `target_node` lands on the right LOCAL slot (see master_kernel.h's own doc
// comment for why this is safe only because dispatch is per-projection, one postsynaptic population
// per call). Only a name matching a StateDirective on the postsynaptic cell type is supported (every
// real Phase-1 GLIF cell's `v`) -- a derived-only (Expose-without-State) postsynaptic quantity throws.
f32 *resolve_require_binding_pointer(const String &name, const IrProgram &postsynaptic_cell_program,
                                      ModelAllocation &allocation, s64 postsynaptic_chunk_base_offset,
                                      s32 postsynaptic_population_size, s32 postsynaptic_neuron_index_begin) {
    for (const auto &directive : postsynaptic_cell_program.alloc) {
        const auto *state = std::get_if<StateDirective>(&directive);
        if (state == nullptr || state->name != name) continue;
        s32 offset_within_type = state_variable_offset(postsynaptic_cell_program, name);
        return allocation.cell_state.get_contents() + postsynaptic_chunk_base_offset +
               (s64)offset_within_type * postsynaptic_population_size - postsynaptic_neuron_index_begin;
    }
    fail("synapse dispatch: require '" + name + "' does not match any StateDirective on the postsynaptic cell "
         "type '" + postsynaptic_cell_program.component_type_name + "' -- a derived-only (Expose-without-State) "
         "postsynaptic quantity is not supported by this ticket's dispatch (see master_kernel.h)");
    return nullptr;
}

// Resolves one synapse edge-parallel function (`_integrate_edges` or `_deliver_<port>`) parameter
// name to a dispatch argument, appended onto `builder` -- the edge-parallel analogue of
// append_cell_tick_argument above (see master_kernel.h's own doc comment for the full design).
// `matrix_index_by_peredge_name` is this synapse type's own registered WeightMatrix matrix indices
// (plain data, not AssembledModel::SynapseTypeRuntimeInfo itself -- that type is private to
// AssembledModel, matching how append_cell_tick_argument above already takes plain fields rather
// than a whole PopulationRuntimeInfo).
void append_synapse_edge_argument(DispatchArgumentBuilder &builder, const String &parameter_name,
                                   const IrProgram &synapse_program,
                                   const UnorderedMap<String, s64> &matrix_index_by_peredge_name,
                                   const IrProgram &postsynaptic_cell_program, s32 postsynaptic_population_size,
                                   s32 postsynaptic_neuron_index_begin, ModelAllocation &allocation,
                                   s64 postsynaptic_chunk_base_offset, WeightMatrix &weights, f32 dt, s64 tick,
                                   f32 *network_inputs, const s32 *source_node_indices, const s32 *target_node_indices,
                                   const s32 *edge_slot_indices, s64 event_count) {
    if (parameter_name == "dt") { builder.add_f32(dt); return; }
    if (parameter_name == "tick") { builder.add_s64(tick); return; }
    if (parameter_name == "network_inputs") { builder.add_pointer(network_inputs); return; }
    if (parameter_name == "source_node_indices") { builder.add_pointer(source_node_indices); return; }
    if (parameter_name == "target_node_indices") { builder.add_pointer(target_node_indices); return; }
    if (parameter_name == "edge_slot_indices") { builder.add_pointer(edge_slot_indices); return; }
    if (parameter_name == "event_count") { builder.add_s64(event_count); return; }
    if (parameter_name == "U") { builder.add_pointer(weights.U_matrix.get_contents()); return; }
    if (parameter_name == "V") { builder.add_pointer(weights.V_matrix.get_contents()); return; }
    if (parameter_name == "rank_float4_stride") { builder.add_s64(weights.rank_float4_stride); return; }
    if (parameter_name == "max_neighbor_count") { builder.add_s64(weights.max_neighbor_count); return; }
    if (parameter_name == "rng_state") {
        fail("synapse dispatch: parameter 'rng_state' (rand/randn) is not supported -- no Phase-1 synapse "
             "ComponentType references it (see master_kernel.h)");
    }
    if (parameter_name.rfind("coefficients_", 0) == 0) {
        String peredge_name = parameter_name.substr(String("coefficients_").size());
        auto found = matrix_index_by_peredge_name.find(peredge_name);
        if (found == matrix_index_by_peredge_name.end()) {
            fail("synapse dispatch: peredge '" + peredge_name + "' has no registered WeightMatrix matrix_index");
        }
        builder.add_pointer(weights.coefficient_vectors[(usize)found->second].get_contents());
        return;
    }
    if (parameter_name.rfind("sparse_delta_", 0) == 0) {
        String peredge_name = parameter_name.substr(String("sparse_delta_").size());
        auto found = matrix_index_by_peredge_name.find(peredge_name);
        if (found == matrix_index_by_peredge_name.end()) {
            fail("synapse dispatch: peredge '" + peredge_name + "' has no registered WeightMatrix matrix_index");
        }
        builder.add_pointer(weights.sparse_delta_buffers[(usize)found->second].get_contents());
        return;
    }

    for (const auto &directive : synapse_program.alloc) {
        if (const auto *param_constant = std::get_if<ParamConstantDirective>(&directive)) {
            if (param_constant->name != parameter_name) continue;
            fail("synapse dispatch: parameter '" + parameter_name + "' is an un-baked (bare) `param` -- "
                 "Phase-1's allocate_model has no established value source for this case (matches "
                 "append_cell_tick_argument's own established scope boundary)");
        }
        if (const auto *require = std::get_if<RequireDirective>(&directive)) {
            if (require->name != parameter_name) continue;
            builder.add_pointer(resolve_require_binding_pointer(parameter_name, postsynaptic_cell_program, allocation,
                                                                 postsynaptic_chunk_base_offset,
                                                                 postsynaptic_population_size,
                                                                 postsynaptic_neuron_index_begin));
            return;
        }
    }

    fail("synapse dispatch: parameter '" + parameter_name + "' is not a recognized reserved name, .alloc name, "
         "or edge-dispatch parameter for ticket #131's synapse dispatch (see master_kernel.h)");
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
// Only assembled/compiled when AssembledModel is constructed with enable_delay_ring=true (see
// master_kernel.h). Same fixed-stage role as the two kernels above (deliver-drain / k^2-tree
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

AssembledModel::AssembledModel(const ModelSpecification &model, const Vector<IrProgram> &type_library_ir_programs,
                                bool enable_delay_ring) {
    AssembledMasterKernelSource assembled = assemble_master_kernel_source(model, type_library_ir_programs);

    type_library_ir_programs_ = type_library_ir_programs;
    total_neuron_count_ = model.total_neuron_count;
    emit_port_names_ = collect_emit_port_names(model, type_library_ir_programs);

    populations_.resize(model.populations.size());
    for (usize index = 0; index < model.populations.size(); ++index) {
        const PopulationEntry &population = model.populations[index];
        const GpuSource &source = assembled.population_gpu_sources[index];

        PopulationRuntimeInfo &info = populations_[index];
        info.type_library_index = population.type_library_index;
        info.neuron_index_begin = population.neuron_index_begin;
        info.population_size = population.size;

        if (source.functions.empty()) continue; // no per-neuron content -- nothing to compile/dispatch

        info.has_kernel = true;
        info.parameter_names_in_order = source.functions[0].parameter_names_in_order;
        const IrProgram &program = type_library_ir_programs[(usize)population.type_library_index];
        info.handle = compile_kernel_or_throw_with_source(
            source_text_for_this_backend(source), source.functions[0].function_name,
            "population '" + population.id + "' (ComponentType '" + program.component_type_name + "')",
            print_ir_program(program));
    }

    drain_parameter_names_ = assembled.drain_network_inputs_source.functions.at(0).parameter_names_in_order;
    drain_kernel_handle_ = compile_kernel_or_throw_with_source(
        source_text_for_this_backend(assembled.drain_network_inputs_source),
        assembled.drain_network_inputs_source.functions.at(0).function_name,
        "the engine-fixed deliver-drain kernel", "");

    propagate_parameter_names_ = assembled.propagate_source.functions.at(0).parameter_names_in_order;
    propagate_kernel_handle_ = compile_kernel_or_throw_with_source(
        source_text_for_this_backend(assembled.propagate_source),
        assembled.propagate_source.functions.at(0).function_name,
        "the engine-fixed propagate kernel", "");

    // ── ticket #64 [F3]: ring-based deliver-drain/propagate, only when opted into ─────────────────
    delay_ring_enabled_ = enable_delay_ring;
    if (delay_ring_enabled_) {
        GpuSource drain_ring_source = build_drain_ring_kernel_gpu_source();
        drain_ring_parameter_names_ = drain_ring_source.functions.at(0).parameter_names_in_order;
        drain_ring_kernel_handle_ = compile_kernel_or_throw_with_source(
            source_text_for_this_backend(drain_ring_source), drain_ring_source.functions.at(0).function_name,
            "the engine-fixed ring deliver-drain kernel (ticket #64)", "");

        GpuSource propagate_ring_source = build_propagate_ring_kernel_gpu_source();
        propagate_ring_parameter_names_ = propagate_ring_source.functions.at(0).parameter_names_in_order;
        propagate_ring_kernel_handle_ = compile_kernel_or_throw_with_source(
            source_text_for_this_backend(propagate_ring_source), propagate_ring_source.functions.at(0).function_name,
            "the engine-fixed ring propagate kernel (ticket #64)", "");
    }

    // ── ticket #131: spike-scatter batch-construction subsystem -- compile every USED synapse
    // type's edge-parallel functions once (topology/matrix-index registration is deferred to the
    // first step_tick call, see ensure_synapse_dispatch_topology_built). Deliberately NOT activated
    // when `enable_delay_ring` is true (ticket #64's delay ring and this ticket's synapse dispatch
    // have not been integrated with each other, see master_kernel.h) -- rather than throwing on a
    // model that happens to combine the two (every existing delay-ring test/example ALREADY builds
    // its model with real projections, since those previously had zero effect either way -- nothing
    // ever dispatched a synapse type's own functions before this ticket), `projections_` is simply
    // left empty in that case, so step_tick's `!projections_.empty()` gate leaves ring-mode
    // step_tick's behavior exactly as it was before this ticket. ─────────────────────────────────
    projections_ = enable_delay_ring ? Vector<ProjectionEntry>{} : model.projections;
    projection_edge_topology_.resize(projections_.size());

    for (const ProjectionEntry &projection : projections_) {
        if (projection.postsynaptic_population_index < 0 ||
            (usize)projection.postsynaptic_population_index >= model.populations.size()) {
            fail("AssembledModel: projection '" + projection.id + "' has an out-of-range postsynaptic_population_index (" +
                 std::to_string(projection.postsynaptic_population_index) + ")");
        }
        s32 synapse_type_index = projection.synapse_type_library_index;
        if (synapse_type_index < 0 || (usize)synapse_type_index >= model.type_library.size()) {
            fail("AssembledModel: projection '" + projection.id + "' has an out-of-range synapse_type_library_index (" +
                 std::to_string(synapse_type_index) + ")");
        }
        if (model.type_library[(usize)synapse_type_index].category != TypeLibraryCategory::Synapse) {
            fail("AssembledModel: projection '" + projection.id + "' references type_library[" +
                 std::to_string(synapse_type_index) + "], which is not a Synapse-category entry");
        }
        if (synapse_types_by_type_library_index_.count(synapse_type_index) > 0) continue; // already compiled

        const IrProgram &synapse_program = type_library_ir_programs[(usize)synapse_type_index];
        GpuSource synapse_source = lower_synapse_ir_program_to_gpu_source(synapse_program);
        if (synapse_source.functions.empty()) {
            fail("AssembledModel: synapse type '" + synapse_program.component_type_name +
                 "' lowered to zero GPU functions (lower_synapse_ir_program_to_gpu_source should always emit "
                 "at least the `_integrate_edges` function)");
        }

        SynapseTypeRuntimeInfo synapse_info;
        for (usize function_index = 0; function_index < synapse_source.functions.size(); ++function_index) {
            const GpuFunctionSignature &signature = synapse_source.functions[function_index];
            bool is_integrate_edges_function = (function_index + 1 == synapse_source.functions.size());
            String label = "synapse type '" + synapse_program.component_type_name + "' function '" +
                            signature.function_name + "'";
            KernelHandle handle = compile_kernel_or_throw_with_source(
                source_text_for_this_backend(synapse_source), signature.function_name, label,
                print_ir_program(synapse_program));
            if (is_integrate_edges_function) {
                synapse_info.integrate_edges_handle = handle;
                synapse_info.integrate_edges_parameter_names_in_order = signature.parameter_names_in_order;
            } else {
                synapse_info.deliver_functions.push_back(
                    DeliverFunctionRuntimeInfo{handle, signature.parameter_names_in_order});
            }
        }
        synapse_types_by_type_library_index_.emplace(synapse_type_index, std::move(synapse_info));
    }
}

AssembledModel::~AssembledModel() {
    for (auto &info : populations_) {
        if (info.has_kernel) release_kernel(info.handle);
    }
    release_kernel(drain_kernel_handle_);
    release_kernel(propagate_kernel_handle_);
    if (delay_ring_enabled_) {
        release_kernel(drain_ring_kernel_handle_);
        release_kernel(propagate_ring_kernel_handle_);
    }

    // ── ticket #131: spike-scatter batch-construction subsystem ─────────────────────────────────
    for (auto &type_entry : synapse_types_by_type_library_index_) {
        release_kernel(type_entry.second.integrate_edges_handle);
        for (auto &deliver_function : type_entry.second.deliver_functions) release_kernel(deliver_function.handle);
    }
    for (auto &topology : projection_edge_topology_) {
        deallocate(std::move(topology.source_nodes));
        deallocate(std::move(topology.target_nodes));
        deallocate(std::move(topology.forward_slots));
        deallocate(std::move(topology.delivery_scratch_source_nodes));
        deallocate(std::move(topology.delivery_scratch_target_nodes));
        deallocate(std::move(topology.delivery_scratch_edge_slots));
    }
}

bool AssembledModel::population_is_closed_form_advanceable(usize population_index) const {
    const PopulationRuntimeInfo &info = populations_.at(population_index);
    if (!info.has_kernel) return false;
    return type_library_ir_programs_.at((usize)info.type_library_index).closed_form_advanceable;
}

void AssembledModel::step_tick(const ModelRuntimeBuffers &buffers, f32 dt, s64 tick, s64 next_tick) {
    if (buffers.allocation == nullptr || buffers.weights == nullptr) {
        fail("step_tick: ModelRuntimeBuffers::allocation/weights must be non-null");
    }
    if (delay_ring_enabled_ != (buffers.delay_ring != nullptr)) {
        fail("step_tick: ModelRuntimeBuffers::delay_ring must be non-null if and only if this "
             "AssembledModel was constructed with enable_delay_ring=true (ticket #64)");
    }

    // ── ticket #131: spike-scatter batch-construction subsystem -- topology is built lazily here
    // (needs buffers.weights, only available at step_tick time); the two new dispatches themselves
    // run later, at the SAME point in the tick the fixed scalar propagate stage already writes
    // network_inputs (see below) -- not before this tick's cell dispatches -- so that
    // network_inputs already reflects THIS tick's fresh synaptic current by the time step_tick
    // RETURNS, exactly matching the existing scalar-weight path's own observable timing (arch §0.2/
    // ir_spec.md §3.5: a write becomes visible right after the tick it's computed in, read at the
    // NEXT tick's cell dispatch) -- see master_kernel.h's own doc comment for the full derivation.
    if (!projections_.empty()) {
        ensure_synapse_dispatch_topology_built(buffers);
    }

    const s64 *chunk_base_offsets = buffers.allocation->cell_type_boundaries.get_contents();

    // ── ticket #64 [F3]: which network_inputs buffer this tick's population dispatch reads from --
    // either the flat, pre-#64 buffer (delay_ring == nullptr, byte-for-byte the same as before this
    // ticket) or this tick's own ring slot (delay_ring != nullptr, see delay_ring.h). Either way, the
    // per-population `_tick` kernels below still take a parameter literally named `network_inputs`
    // (append_cell_tick_argument, unchanged by this ticket) -- only WHERE it points changes.
    f32 *network_inputs_for_this_tick = buffers.network_inputs;
    s64 current_ring_slot = 0;
    if (buffers.delay_ring != nullptr) {
        current_ring_slot = tick % buffers.delay_ring->ring_slot_count;
        network_inputs_for_this_tick =
            buffers.delay_ring->input_ring.get_contents() + current_ring_slot * total_neuron_count_;
    }

    // stages 2-5: one dispatch per population, over its own full neuron range (arch §4.1's
    // cell-type-boundary dispatch -- one dispatch per boundary, not a single mega-dispatch with a
    // runtime branch; see master_kernel.h's own header comment for why this is faithful to "one
    // thread per neuron, dispatching by cell-type boundary" without needing to merge every type's
    // generated code into one literal kernel function). Every population dispatches its FULL range
    // here regardless of population_is_closed_form_advanceable (arch §0.5, ticket #62 [F1]'s own
    // still-deliberate scope boundary -- see master_kernel.h's step_tick doc comment): this is safe
    // for a nonlinear-tagged population (never fast-forwarded/skipped, exactly the correctness rule
    // requires) and does not yet exploit the tag for a closed_form_advanceable population's own
    // optimization (a separate, future skip-dispatch ticket's job).
    for (usize index = 0; index < populations_.size(); ++index) {
        const PopulationRuntimeInfo &info = populations_[index];
        if (!info.has_kernel) continue;

        const IrProgram &program = type_library_ir_programs_[(usize)info.type_library_index];
        s64 chunk_base_offset = chunk_base_offsets[index];

        DispatchArgumentBuilder builder;
        for (const String &parameter_name : info.parameter_names_in_order) {
            append_cell_tick_argument(builder, parameter_name, program, info.type_library_index,
                                       info.neuron_index_begin, info.population_size, chunk_base_offset,
                                       *buffers.allocation, dt, tick, network_inputs_for_this_tick, buffers.rng_state,
                                       buffers.emit_port_flags);
        }
        builder.dispatch(info.handle, launch_config_for(info.population_size));
    }

    if (buffers.delay_ring == nullptr) {
        // fixed deliver-drain: network_inputs has now been read by every population's own `_tick`
        // kernel this tick -- zero it so the propagate stage below writes THIS tick's fresh
        // contributions, not an accumulation on top of what was already consumed (ir_spec.md §3.5's
        // >=1-tick latency: a write below is only ever read at the NEXT tick's per-population pass
        // above, never this one, since that pass already ran before this drain).
        {
            DispatchArgumentBuilder builder;
            builder.add_pointer(buffers.network_inputs);
            builder.add_s64(total_neuron_count_);
            builder.dispatch(drain_kernel_handle_, launch_config_for(total_neuron_count_));
        }

        // Reset the active-set enqueue counter to 0 before this tick's propagate dispatches run --
        // matches the real engine's own per-tick reset (src/core/engine.cpp resets
        // next_active_neuron_count to 0 once per tick, before the dispatch that performs the enqueue).
        // The propagate kernel's `active_generation` dedup only prevents re-enqueuing the SAME neuron
        // WITHIN this tick (it compares against next_tick, which is constant for the whole tick); it
        // does nothing to bound the counter ACROSS ticks. Without this reset, `position` grows without
        // bound tick over tick and eventually writes past next_active_neuron_indices's own
        // [total_neuron_count] allocation.
        *buffers.next_active_neuron_count = 0;

        // ── ticket #131: spike-scatter batch-construction subsystem -- delivery-event construction
        // + `_deliver_<port>` dispatch (bumps Sk for this tick's fresh spikes), immediately followed
        // by `_integrate_edges` (reads that same, now-fresh Sk, decays, and writes the finished
        // current into network_inputs, already drained above) -- so by the time step_tick returns,
        // network_inputs already reflects this tick's synaptic contribution, exactly matching the
        // fixed scalar propagate stage's own observable timing (see master_kernel.h). Delivery must
        // run BEFORE the fixed scalar propagate dispatch below, which reads+clears
        // buffers.emit_port_flags for its own last_spiked/active-set bookkeeping (this dispatch
        // reads, but does not clear, the SAME flags). ──────────────────────────────────────────────
        if (!projections_.empty()) {
            dispatch_synapse_delivery_events(buffers, dt, tick);
            dispatch_synapse_integrate_edges(buffers, dt, tick);
        }

        // fixed k^2-tree propagate/scatter + active-set-enqueue (stage 6/9): one dispatch per distinct
        // emit-port name, each over the WHOLE model's neuron range (a spiking neuron's downstream
        // targets come from the model-wide k^2-tree/WeightMatrix, not a population-scoped one). A
        // model with real per-edge synapse projections (ticket #131) has this dispatch's OWN weight
        // contribution forced to zero -- `buffers.weights` itself is left untouched -- since
        // dispatch_synapse_delivery_events/dispatch_synapse_integrate_edges above/below already
        // supply the real per-edge current; last_spiked + active-set enqueue are unaffected.
        f32 constant_weight_for_this_dispatch = projections_.empty() ? buffers.weights->constant_weight : 0.0f;
        s32 using_constant_weight_for_this_dispatch =
            projections_.empty() ? (buffers.weights->using_constant_weight ? 1 : 0) : 1;
        for (const String &port_name : emit_port_names_) {
            auto found = buffers.emit_port_flags.find(port_name);
            if (found == buffers.emit_port_flags.end()) {
                fail("step_tick: no emit-port flag buffer supplied in ModelRuntimeBuffers::emit_port_flags for "
                     "port '" +
                     port_name + "'");
            }

            DispatchArgumentBuilder builder;
            builder.add_s64(tick);
            builder.add_s64(next_tick);
            builder.add_pointer(buffers.weights->U_matrix.get_contents());
            builder.add_pointer(buffers.weights->V_matrix.get_contents());
            builder.add_s64(buffers.weights->rank_float4_stride);
            builder.add_f32(constant_weight_for_this_dispatch);
            builder.add_s32(using_constant_weight_for_this_dispatch);
            builder.add_pointer(buffers.weights->k2tree.internal_node_words.get_contents());
            builder.add_pointer(buffers.weights->k2tree.leaf_node_words.get_contents());
            builder.add_pointer(buffers.weights->k2tree.rank_superblock_table.get_contents());
            builder.add_pointer(buffers.weights->k2tree.rank_subblock_table.get_contents());
            builder.add_s32(buffers.weights->k2tree.branching_factor);
            builder.add_s32(buffers.weights->k2tree.superblock_size_words);
            builder.add_s32(buffers.weights->k2tree.padded_node_count);
            builder.add_s32(buffers.weights->k2tree.tree_height);
            builder.add_s32(buffers.weights->k2tree.internal_bit_count);
            builder.add_s64(total_neuron_count_);
            builder.add_s64(buffers.weights->max_neighbor_count);
            builder.add_pointer(buffers.network_inputs);
            builder.add_pointer(buffers.last_spiked);
            builder.add_pointer(buffers.next_active_neuron_indices);
            builder.add_pointer(buffers.next_active_neuron_count);
            builder.add_pointer(buffers.active_generation);
            builder.add_pointer(found->second);
            builder.dispatch(propagate_kernel_handle_, launch_config_for(total_neuron_count_));
        }

        // ── ticket #132: stage 7 STDP plasticity -- runs immediately after propagate above, once
        // every emit port's dispatch has finished writing this tick's own `last_spiked` entries (see
        // master_kernel.h's own "ticket #132" doc comment). No-op when plasticity is disabled.
        apply_stdp_plasticity(buffers, tick);
        return;
    }

    // ── ticket #64 [F3]: ring-based deliver-drain/propagate ─────────────────────────────────────
    //
    // fixed deliver-drain: this tick's own ring slot has now been read by every population's
    // `_tick` kernel above (via network_inputs_for_this_tick) -- zero just that slot, ready to be
    // reused ring_slot_count ticks from now (see delay_ring.h for why this never collides with
    // anything the propagate-ring dispatch below writes THIS tick -- delay is always >= 1, so it
    // never targets the slot that just got drained).
    {
        DispatchArgumentBuilder builder;
        builder.add_pointer(buffers.delay_ring->input_ring.get_contents());
        builder.add_s64(total_neuron_count_);
        builder.add_s64(current_ring_slot);
        builder.dispatch(drain_ring_kernel_handle_, launch_config_for(total_neuron_count_));
    }

    // Reset this tick's own pending-active slot's count to 0 -- mirrors the flat path's
    // next_active_neuron_count reset above, ring-generalized: slot current_ring_slot's list was
    // populated over the PAST several ticks by whichever earlier ticks' propagate-ring dispatch
    // scattered a delayed arrival landing exactly on `tick` (delay_ring.h explains why resetting it
    // here, before this tick's OWN propagate-ring dispatch runs, is always safe -- this tick can
    // only ever populate SOME OTHER slot, never this one).
    buffers.delay_ring->pending_active_neuron_count.get_contents()[current_ring_slot] = 0;

    for (const String &port_name : emit_port_names_) {
        auto found = buffers.emit_port_flags.find(port_name);
        if (found == buffers.emit_port_flags.end()) {
            fail("step_tick: no emit-port flag buffer supplied in ModelRuntimeBuffers::emit_port_flags for "
                 "port '" +
                 port_name + "'");
        }

        DispatchArgumentBuilder builder;
        builder.add_s64(tick);
        builder.add_s64(buffers.delay_ring->ring_slot_count);
        builder.add_pointer(buffers.weights->U_matrix.get_contents());
        builder.add_pointer(buffers.weights->V_matrix.get_contents());
        builder.add_s64(buffers.weights->rank_float4_stride);
        builder.add_f32(buffers.weights->constant_weight);
        builder.add_s32(buffers.weights->using_constant_weight ? 1 : 0);
        builder.add_pointer(buffers.weights->k2tree.internal_node_words.get_contents());
        builder.add_pointer(buffers.weights->k2tree.leaf_node_words.get_contents());
        builder.add_pointer(buffers.weights->k2tree.rank_superblock_table.get_contents());
        builder.add_pointer(buffers.weights->k2tree.rank_subblock_table.get_contents());
        builder.add_s32(buffers.weights->k2tree.branching_factor);
        builder.add_s32(buffers.weights->k2tree.superblock_size_words);
        builder.add_s32(buffers.weights->k2tree.padded_node_count);
        builder.add_s32(buffers.weights->k2tree.tree_height);
        builder.add_s32(buffers.weights->k2tree.internal_bit_count);
        builder.add_s64(total_neuron_count_);
        builder.add_s64(buffers.weights->max_neighbor_count);
        builder.add_pointer(buffers.delay_ring->edge_delay_ticks.get_contents());
        builder.add_pointer(buffers.delay_ring->input_ring.get_contents());
        builder.add_pointer(buffers.last_spiked);
        builder.add_pointer(buffers.delay_ring->pending_active_neuron_indices.get_contents());
        builder.add_pointer(buffers.delay_ring->pending_active_neuron_count.get_contents());
        builder.add_pointer(buffers.delay_ring->pending_active_generation.get_contents());
        builder.add_pointer(found->second);
        builder.dispatch(propagate_ring_kernel_handle_, launch_config_for(total_neuron_count_));
    }

    // ── ticket #132: stage 7 STDP plasticity -- same call as the flat branch above (the delay ring
    // only changes WHEN network_inputs is delivered, not spike-timing/last_spiked, so the exact same
    // host loop applies unchanged in ring mode -- see master_kernel.h's own "ticket #132" doc comment).
    apply_stdp_plasticity(buffers, tick);
}

// ── ticket #132: real STDP support on AssembledModel ─────────────────────────────────────────────
//
// (see master_kernel.h's own "ticket #132" doc comment for the full design this implements.)

bool AssembledModel::plasticity_enabled() const { return stdp_learning_rate_ > 0.0f; }

void AssembledModel::enable_plasticity(f32 learning_rate) {
    if (plasticity_enabled()) return;

    if (!projections_.empty()) {
        fail("enable_plasticity: this AssembledModel has real per-edge synapse dispatch active "
             "(ticket #131 projections) -- STDP's rank-1 nudge of the shared U/V basis is not yet "
             "compensated against a peredge synapse's own Ck reconstruction sharing that basis, so "
             "combining the two is not supported (see master_kernel.h's own 'ticket #132' doc "
             "comment)");
    }

    stdp_learning_rate_ = learning_rate;
}

void AssembledModel::disable_plasticity() {
    if (!plasticity_enabled()) return;

    stdp_learning_rate_ = 0.0f;
}

void AssembledModel::apply_stdp_plasticity(const ModelRuntimeBuffers &buffers, s64 tick) {
    if (stdp_learning_rate_ == 0.0f) return;

    s64 max_neighbor_count = buffers.weights->max_neighbor_count;
    if (max_neighbor_count <= 0) return;
    Vector<s32> neighbor_scratch((usize)max_neighbor_count);

    for (s64 source_node = 0; source_node < total_neuron_count_; ++source_node) {
        if (buffers.last_spiked[source_node] != tick) continue; // did not fire this tick

        s64 neighbor_count = buffers.weights->get_neighbors(source_node, neighbor_scratch.data());
        for (s64 slot = 0; slot < neighbor_count; ++slot) {
            s32 child = neighbor_scratch[(usize)slot];
            s64 child_last_spiked = buffers.last_spiked[child];
            if (child_last_spiked == 0 || child_last_spiked == tick) continue;

            s64 signed_tick_delta = tick - child_last_spiked;
            f32 tick_delta = (f32)(signed_tick_delta < 0 ? -signed_tick_delta : signed_tick_delta);
            f32 decay_delta = -stdp_learning_rate_ * std::pow(tick_delta, -3.0f);
            buffers.weights->update((s32)source_node, child, decay_delta, /*learning_rate=*/0.5f,
                                     /*l2_regularization=*/1.0f, /*iterations=*/1);
        }
    }
}

// ── ticket #131: spike-scatter batch-construction subsystem ──────────────────────────────────────
//
// (see master_kernel.h's own "spike-scatter batch-construction subsystem" doc comment for the full
// design this implements.)

void AssembledModel::ensure_synapse_dispatch_topology_built(const ModelRuntimeBuffers &buffers) {
    if (synapse_dispatch_topology_built_) return;

    s64 max_neighbor_count = buffers.weights->max_neighbor_count;
    Vector<s32> neighbor_scratch((usize)(max_neighbor_count > 0 ? max_neighbor_count : 0));

    for (usize projection_index = 0; projection_index < projections_.size(); ++projection_index) {
        const ProjectionEntry &projection = projections_[projection_index];
        ProjectionEdgeTopology &topology = projection_edge_topology_[projection_index];

        s32 synapse_type_index = projection.synapse_type_library_index;
        SynapseTypeRuntimeInfo &synapse_info = synapse_types_by_type_library_index_.at(synapse_type_index);
        if (!synapse_info.matrix_indices_registered) {
            // Every peredge variable this synapse type declares gets a real WeightMatrix
            // matrix_index, its coefficient vector all-zeros (arch §4.3: a peredge state variable's
            // Phase-1 default initial value is 0 everywhere -- a zero Ck makes the shared-basis
            // reconstruction contribute exactly 0 regardless of U/V's own random values, so a
            // never-yet-delivered edge's per-edge state reads back as exactly 0, matching that
            // default; only accedge's own Sk accumulation ever moves it away from 0 thereafter).
            // add_coefficient_vector's own logical-rank coefficients ARE all-zero, but it (like
            // set_coefficient_vector) unconditionally fills every PADDING lane (beyond the logical
            // `rank`, up to `rank_float4_stride*4` -- real whenever rank isn't a multiple of 4) with
            // 1.0f, a neutral multiplier suited to DEFAULT_MATRIX_INDEX's own "reduces to dot(U,V)"
            // contract, not to a genuinely-all-zero matrix -- U/V's OWN padding lanes are real,
            // random values (needed for that same default-matrix contract), so a 1.0f padding
            // coefficient would reconstruct a real, nonzero, spurious contribution from them. Zero
            // the WHOLE registered coefficient buffer (a public WeightMatrix field) directly
            // afterward, overriding that padding fill, so this matrix's reconstruction is exactly 0
            // until a real accedge Sk bump moves it.
            const IrProgram &synapse_program = type_library_ir_programs_[(usize)synapse_type_index];
            Vector<f32> zero_coefficients((usize)buffers.weights->rank, 0.0f);
            for (const auto &directive : synapse_program.alloc) {
                if (const auto *peredge = std::get_if<PeredgeDirective>(&directive)) {
                    s64 matrix_index = buffers.weights->add_coefficient_vector(zero_coefficients);
                    s64 effective_lane_count = buffers.weights->rank_float4_stride * 4;
                    memset(buffers.weights->coefficient_vectors[(usize)matrix_index].get_contents(), 0,
                           (usize)effective_lane_count * sizeof(f32));
                    synapse_info.matrix_index_by_peredge_name[peredge->name] = matrix_index;
                }
            }
            synapse_info.matrix_indices_registered = true;
        }

        Vector<s32> source_nodes_host;
        Vector<s32> target_nodes_host;
        Vector<s32> forward_slots_host;
        source_nodes_host.reserve(projection.connections.size());
        target_nodes_host.reserve(projection.connections.size());
        forward_slots_host.reserve(projection.connections.size());

        for (const ConnectionEntry &connection : projection.connections) {
            s64 neighbor_count = buffers.weights->k2tree.get_neighbors(connection.source_neuron_index,
                                                                        neighbor_scratch.data(), max_neighbor_count);
            s32 forward_slot = -1;
            for (s64 slot = 0; slot < neighbor_count; ++slot) {
                if (neighbor_scratch[(usize)slot] == connection.target_neuron_index) {
                    forward_slot = (s32)slot;
                    break;
                }
            }
            if (forward_slot < 0) {
                fail("ensure_synapse_dispatch_topology_built: connection " +
                     std::to_string(connection.source_neuron_index) + " -> " +
                     std::to_string(connection.target_neuron_index) + " (projection '" + projection.id +
                     "') is not a real edge in ModelRuntimeBuffers::weights's k^2-tree -- weights must reflect "
                     "the SAME adjacency model.projections describes");
            }
            source_nodes_host.push_back(connection.source_neuron_index);
            target_nodes_host.push_back(connection.target_neuron_index);
            forward_slots_host.push_back(forward_slot);
        }

        topology.edge_count = (s64)source_nodes_host.size();
        if (topology.edge_count > 0) {
            usize byte_count = (usize)topology.edge_count * sizeof(s32);
            topology.source_nodes = allocate<s32>(byte_count);
            topology.target_nodes = allocate<s32>(byte_count);
            topology.forward_slots = allocate<s32>(byte_count);
            memcpy(topology.source_nodes.get_contents(), source_nodes_host.data(), byte_count);
            memcpy(topology.target_nodes.get_contents(), target_nodes_host.data(), byte_count);
            memcpy(topology.forward_slots.get_contents(), forward_slots_host.data(), byte_count);

            // Reused every tick, refilled with THIS tick's fired-source subset -- see
            // dispatch_synapse_delivery_events.
            topology.delivery_scratch_source_nodes = allocate<s32>(byte_count);
            topology.delivery_scratch_target_nodes = allocate<s32>(byte_count);
            topology.delivery_scratch_edge_slots = allocate<s32>(byte_count);
        }
    }

    synapse_dispatch_topology_built_ = true;
}

void AssembledModel::dispatch_synapse_integrate_edges(const ModelRuntimeBuffers &buffers, f32 dt, s64 tick) {
    const s64 *chunk_base_offsets = buffers.allocation->cell_type_boundaries.get_contents();

    for (usize projection_index = 0; projection_index < projections_.size(); ++projection_index) {
        const ProjectionEntry &projection = projections_[projection_index];
        const ProjectionEdgeTopology &topology = projection_edge_topology_[projection_index];
        if (topology.edge_count == 0) continue;

        s32 synapse_type_index = projection.synapse_type_library_index;
        const SynapseTypeRuntimeInfo &synapse_info = synapse_types_by_type_library_index_.at(synapse_type_index);
        const IrProgram &synapse_program = type_library_ir_programs_[(usize)synapse_type_index];

        const PopulationRuntimeInfo &postsynaptic_population =
            populations_.at((usize)projection.postsynaptic_population_index);
        const IrProgram &postsynaptic_cell_program =
            type_library_ir_programs_[(usize)postsynaptic_population.type_library_index];
        s64 postsynaptic_chunk_base_offset = chunk_base_offsets[(usize)projection.postsynaptic_population_index];

        DispatchArgumentBuilder builder;
        for (const String &parameter_name : synapse_info.integrate_edges_parameter_names_in_order) {
            append_synapse_edge_argument(
                builder, parameter_name, synapse_program, synapse_info.matrix_index_by_peredge_name,
                postsynaptic_cell_program, postsynaptic_population.population_size,
                postsynaptic_population.neuron_index_begin, *buffers.allocation, postsynaptic_chunk_base_offset,
                *buffers.weights, dt, tick, buffers.network_inputs, topology.source_nodes.get_contents(),
                topology.target_nodes.get_contents(), topology.forward_slots.get_contents(), topology.edge_count);
        }
        builder.dispatch(synapse_info.integrate_edges_handle, launch_config_for(topology.edge_count));
    }
}

void AssembledModel::dispatch_synapse_delivery_events(const ModelRuntimeBuffers &buffers, f32 dt, s64 tick) {
    const s64 *chunk_base_offsets = buffers.allocation->cell_type_boundaries.get_contents();

    // "did this neuron fire THIS tick, on any tracked port" -- a union across every distinct
    // EventPort name (mirrors the fixed scalar propagate stage's own per-port dispatch loop, which
    // likewise treats firing on ANY tracked port as "scatter to every downstream target" -- see
    // master_kernel.h). Read here, NOT cleared -- the fixed scalar propagate dispatch immediately
    // after this call still does that, once, for its own last_spiked/active-set bookkeeping.
    Vector<bool> fired_this_tick((usize)total_neuron_count_, false);
    for (const auto &port_entry : buffers.emit_port_flags) {
        const bool *port_flags = port_entry.second;
        for (s64 neuron_index = 0; neuron_index < total_neuron_count_; ++neuron_index) {
            if (port_flags[neuron_index]) fired_this_tick[(usize)neuron_index] = true;
        }
    }

    for (usize projection_index = 0; projection_index < projections_.size(); ++projection_index) {
        const ProjectionEntry &projection = projections_[projection_index];
        ProjectionEdgeTopology &topology = projection_edge_topology_[projection_index];
        if (topology.edge_count == 0) continue;

        const s32 *all_source_nodes = topology.source_nodes.get_contents();
        const s32 *all_target_nodes = topology.target_nodes.get_contents();
        const s32 *all_forward_slots = topology.forward_slots.get_contents();
        s32 *scratch_source_nodes = topology.delivery_scratch_source_nodes.get_contents();
        s32 *scratch_target_nodes = topology.delivery_scratch_target_nodes.get_contents();
        s32 *scratch_edge_slots = topology.delivery_scratch_edge_slots.get_contents();

        s64 delivered_event_count = 0;
        for (s64 edge_index = 0; edge_index < topology.edge_count; ++edge_index) {
            if (!fired_this_tick[(usize)all_source_nodes[edge_index]]) continue;
            scratch_source_nodes[delivered_event_count] = all_source_nodes[edge_index];
            scratch_target_nodes[delivered_event_count] = all_target_nodes[edge_index];
            scratch_edge_slots[delivered_event_count] = all_forward_slots[edge_index];
            ++delivered_event_count;
        }
        if (delivered_event_count == 0) continue;

        s32 synapse_type_index = projection.synapse_type_library_index;
        const SynapseTypeRuntimeInfo &synapse_info = synapse_types_by_type_library_index_.at(synapse_type_index);
        const IrProgram &synapse_program = type_library_ir_programs_[(usize)synapse_type_index];

        const PopulationRuntimeInfo &postsynaptic_population =
            populations_.at((usize)projection.postsynaptic_population_index);
        const IrProgram &postsynaptic_cell_program =
            type_library_ir_programs_[(usize)postsynaptic_population.type_library_index];
        s64 postsynaptic_chunk_base_offset = chunk_base_offsets[(usize)projection.postsynaptic_population_index];

        for (const DeliverFunctionRuntimeInfo &deliver_function : synapse_info.deliver_functions) {
            DispatchArgumentBuilder builder;
            for (const String &parameter_name : deliver_function.parameter_names_in_order) {
                append_synapse_edge_argument(builder, parameter_name, synapse_program,
                                              synapse_info.matrix_index_by_peredge_name, postsynaptic_cell_program,
                                              postsynaptic_population.population_size,
                                              postsynaptic_population.neuron_index_begin, *buffers.allocation,
                                              postsynaptic_chunk_base_offset, *buffers.weights, dt, tick,
                                              buffers.network_inputs, scratch_source_nodes, scratch_target_nodes,
                                              scratch_edge_slots, delivered_event_count);
            }
            builder.dispatch(deliver_function.handle, launch_config_for(delivered_event_count));
        }
    }
}

} // namespace spikecorec::nml
