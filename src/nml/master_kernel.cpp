#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

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
// supports (the ones Phase-1's GLIF-family cell lowering, cell_lowering.cpp, actually emits) and
// which it deliberately throws on (require/rng_state/tree-walk/shared-basis/un-baked-bare-param).
void append_cell_tick_argument(DispatchArgumentBuilder &builder, const String &parameter_name,
                                const IrProgram &program, s32 type_library_index, s32 neuron_index_begin,
                                s32 population_size, s64 cell_state_chunk_base_offset, ModelAllocation &allocation,
                                f32 dt, s64 tick, f32 *network_inputs,
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

    if (parameter_name == "rng_state") {
        fail("cell-type kernel parameter 'rng_state' (rand/randn) is not wired by this ticket's dispatch "
             "(Phase-2 scope)");
    }
    fail("cell-type kernel parameter '" + parameter_name + "' is not a recognized reserved name, .alloc name, "
         "or emit port for this ticket's dispatch -- likely the k^2-tree-walk/shared-basis block "
         "(forall/loadedge/accedge), out of scope for a Phase-1 GLIF-family cell (see master_kernel.h)");
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
                                       *buffers.allocation, dt, tick, network_inputs_for_this_tick,
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

        // fixed k^2-tree propagate/scatter + active-set-enqueue (stage 6/9): one dispatch per distinct
        // emit-port name, each over the WHOLE model's neuron range (a spiking neuron's downstream
        // targets come from the model-wide k^2-tree/WeightMatrix, not a population-scoped one).
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
            builder.add_pointer(buffers.network_inputs);
            builder.add_pointer(buffers.last_spiked);
            builder.add_pointer(buffers.next_active_neuron_indices);
            builder.add_pointer(buffers.next_active_neuron_count);
            builder.add_pointer(buffers.active_generation);
            builder.add_pointer(found->second);
            builder.dispatch(propagate_kernel_handle_, launch_config_for(total_neuron_count_));
        }
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
}

} // namespace spikecorec::nml
