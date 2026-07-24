//
// Created by Alek Simpson on 5/30/26.
//

#include <cstring>
#include <cmath>
#include <new>

#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include "spikecorec/core/engine.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/recording.h"
#include "spikecorec/core/topologies.h"
#include "spikecorec/nml/master_kernel.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::log;

// ── constructor / destructor ──────────────────────────────────────────────────

SpikeEngine::SpikeEngine()
    : logger(make_logger())
    , weights(random_fixed_outdegree(15), 1, true)
    , neuron_count(15 * 15)
    , input_neuron_count(0)
    , thread_count_per_block(256)
    , block_count(0)
    , resting_membrane_potential(0.1f)
    , decay_rate(0.01f)
    , learning_rate(0.0f) // plasticity off by default
    , spike_period(1)
    , spike_threshold(1.0f)
    , alive(true)
    , active_set_optimization_enabled(true)
{

    block_count = (s32) ((neuron_count + thread_count_per_block - 1) / thread_count_per_block);

    usize neuron_f32_byte_size = (usize) neuron_count * sizeof(f32);
    usize neuron_s32_byte_size = (usize) neuron_count * sizeof(s32);
    usize neuron_s64_byte_size = (usize) neuron_count * sizeof(s64);

    network_inputs = allocate<f32>(neuron_f32_byte_size);
    memset(network_inputs.get_contents(), 0, neuron_f32_byte_size);

    membrane_potentials = allocate<f32>(neuron_f32_byte_size);
    std::fill(membrane_potentials.get_contents(),
            membrane_potentials.get_contents() + neuron_count,
            resting_membrane_potential);

    last_spiked = allocate<s64>(neuron_s64_byte_size);
    memset(last_spiked.get_contents(), 0, neuron_s64_byte_size);

    last_tick_updated = allocate<s64>(neuron_s64_byte_size);
    memset(last_tick_updated.get_contents(), 0, neuron_s64_byte_size);

    active_neuron_indices = allocate<s32>(neuron_s32_byte_size);
    next_active_neuron_indices = allocate<s32>(neuron_s32_byte_size);

    active_neuron_count = allocate<s32>(sizeof(s32));
    next_active_neuron_count = allocate<s32>(sizeof(s32));
    active_neuron_count.get_contents()[0] = 0;
    next_active_neuron_count.get_contents()[0] = 0;

    active_generation = allocate<s32>(neuron_s32_byte_size);
    s32 *active_generation_data = active_generation.get_contents();
    for (s64 neuron_index = 0; neuron_index < neuron_count; ++neuron_index)
        active_generation_data[neuron_index] = -1;

    input_staging = allocate<f32>(neuron_f32_byte_size);
    override_staging = allocate<s64>(neuron_s64_byte_size);

    prefetch_to_gpu(network_inputs, neuron_f32_byte_size);
    prefetch_to_gpu(membrane_potentials, neuron_f32_byte_size);
    prefetch_to_gpu(last_spiked, neuron_s64_byte_size);
    prefetch_to_gpu(last_tick_updated, neuron_s64_byte_size);
    prefetch_to_gpu(active_neuron_indices, neuron_s32_byte_size);
    prefetch_to_gpu(next_active_neuron_indices, neuron_s32_byte_size);
    prefetch_to_gpu(active_neuron_count, sizeof(s32));
    prefetch_to_gpu(next_active_neuron_count, sizeof(s32));
    prefetch_to_gpu(active_generation, neuron_s32_byte_size);
    prefetch_to_gpu(input_staging, neuron_f32_byte_size);
    prefetch_to_gpu(override_staging, neuron_s64_byte_size);

    logger->debug("SpikeEngine buffers allocated: neuron_f32_byte_size={} neuron_s32_byte_size={} "
                  "neuron_s64_byte_size={} thread_count_per_block={} block_count={}",
                  neuron_f32_byte_size, neuron_s32_byte_size, neuron_s64_byte_size,
                  thread_count_per_block, block_count);
    logger->info("SpikeEngine constructed: neuron_count={} resting_mp={} decay_rate={} learning_rate={}",
                  neuron_count, resting_membrane_potential, decay_rate, learning_rate);
}

SpikeEngine::SpikeEngine(
    vector<vector<s32>> *network,
    const vector<s64> &shape,
    s64 rank,
    f32 resting_mp,
    f32 decay_rate,
    f32 learning_rate,
    bool plasticity_enabled, 
    bool active_set_optimization_enabled
)
    : logger(make_logger())
    , weights(*network, rank, true)
    , neuron_count(shape[0] * shape[1])
    , input_neuron_count(0)
    , thread_count_per_block(256)
    , block_count(0)
    , resting_membrane_potential(resting_mp)
    , decay_rate(decay_rate)
    , learning_rate(learning_rate)
    , spike_period(1)
    , spike_threshold(1.0f)
    , alive(true)
    , active_set_optimization_enabled(active_set_optimization_enabled)
{
    if (!plasticity_enabled && learning_rate > 0.0f) {
        throw std::runtime_error("Spike engine cannot be initialized with learning rate > 0.0f while plasticity is disabled.");
    }

    if (!plasticity_enabled) learning_rate = 0.0f;


    block_count = (s32) ((neuron_count + thread_count_per_block - 1) / thread_count_per_block);

    usize neuron_f32_byte_size = (usize) neuron_count * sizeof(f32);
    usize neuron_s32_byte_size = (usize) neuron_count * sizeof(s32);
    usize neuron_s64_byte_size = (usize) neuron_count * sizeof(s64);

    network_inputs = allocate<f32>(neuron_f32_byte_size);
    memset(network_inputs.get_contents(), 0, neuron_f32_byte_size);

    membrane_potentials = allocate<f32>(neuron_f32_byte_size);
    std::fill(membrane_potentials.get_contents(),
            membrane_potentials.get_contents() + neuron_count,
            resting_membrane_potential);

    last_spiked = allocate<s64>(neuron_s64_byte_size);
    memset(last_spiked.get_contents(), 0, neuron_s64_byte_size);

    last_tick_updated = allocate<s64>(neuron_s64_byte_size);
    memset(last_tick_updated.get_contents(), 0, neuron_s64_byte_size);

    active_neuron_indices = allocate<s32>(neuron_s32_byte_size);
    next_active_neuron_indices = allocate<s32>(neuron_s32_byte_size);

    active_neuron_count = allocate<s32>(sizeof(s32));
    next_active_neuron_count = allocate<s32>(sizeof(s32));
    active_neuron_count.get_contents()[0] = 0;
    next_active_neuron_count.get_contents()[0] = 0;

    active_generation = allocate<s32>(neuron_s32_byte_size);
    s32 *active_generation_data = active_generation.get_contents();
    for (s64 neuron_index = 0; neuron_index < neuron_count; ++neuron_index)
        active_generation_data[neuron_index] = -1;

    input_staging = allocate<f32>(neuron_f32_byte_size);
    override_staging = allocate<s64>(neuron_s64_byte_size);

    prefetch_to_gpu(network_inputs, neuron_f32_byte_size);
    prefetch_to_gpu(membrane_potentials, neuron_f32_byte_size);
    prefetch_to_gpu(last_spiked, neuron_s64_byte_size);
    prefetch_to_gpu(last_tick_updated, neuron_s64_byte_size);
    prefetch_to_gpu(active_neuron_indices, neuron_s32_byte_size);
    prefetch_to_gpu(next_active_neuron_indices, neuron_s32_byte_size);
    prefetch_to_gpu(active_neuron_count, sizeof(s32));
    prefetch_to_gpu(next_active_neuron_count, sizeof(s32));
    prefetch_to_gpu(active_generation, neuron_s32_byte_size);
    prefetch_to_gpu(input_staging, neuron_f32_byte_size);
    prefetch_to_gpu(override_staging, neuron_s64_byte_size);

    logger->debug("SpikeEngine buffers allocated: neuron_f32_byte_size={} neuron_s32_byte_size={} "
                  "neuron_s64_byte_size={} thread_count_per_block={} block_count={}",
                  neuron_f32_byte_size, neuron_s32_byte_size, neuron_s64_byte_size,
                  thread_count_per_block, block_count);
    logger->info("SpikeEngine constructed: neuron_count={} resting_mp={} decay_rate={} learning_rate={}",
                  neuron_count, resting_mp, decay_rate, learning_rate);
}

// ── NML-mode helpers (Stage 1 of folding nml::AssembledModel into SpikeEngine -- see
// include/spikecorec/nml/master_kernel.h's own "REFACTOR" comments above class AssembledModel, and
// SpikeEngine's own ModelSpecification constructor/step_tick below). Small, file-local mirrors of
// master_kernel.cpp's own (anonymous-namespace, so not directly reusable) helpers of the same
// shape -- independent copies rather than exporting those, to avoid touching master_kernel.h/.cpp
// at all for this additive stage. ────────────────────────────────────────────────────────────────
namespace {

const String &nml_source_text_for_this_backend(const nml::GpuSource &source) {
#ifdef SPIKECOREC_CUDA
    return source.cuda_source;
#else
    return source.msl_source;
#endif
}

LaunchConfig nml_launch_config_for(s64 element_count) {
    constexpr u32 threads_per_block = 256;
    if (element_count <= 0) return LaunchConfig{0, threads_per_block};
    u32 grid = (u32)((element_count + threads_per_block - 1) / threads_per_block);
    return LaunchConfig{grid, threads_per_block};
}

// Position (0-based) of `state_name` among `program.alloc`'s StateDirective entries, in
// declaration order -- mirrors master_kernel.cpp's own state_variable_offset (MUST match
// nml::allocate_model's own enumeration order, which this does since both iterate the exact same
// Vector<AllocDirective> in the exact same order).
s32 nml_state_variable_offset(const nml::IrProgram &program, const String &state_name) {
    s32 offset = 0;
    for (const auto &directive : program.alloc) {
        if (const auto *state = std::get_if<nml::StateDirective>(&directive)) {
            if (state->name == state_name) return offset;
            ++offset;
        }
    }
    log::throw_runtime_error(log::logger(),
        "SpikeEngine (nml): '" + state_name + "' is not a StateDirective of program '" +
        program.component_type_name + "'");
}

// Builds the args[]/arg_sizes[] pair metal_dispatch/cuda_dispatch expect -- mirrors
// master_kernel.cpp's own DispatchArgumentBuilder (same shape, independent copy for the same
// reason as the two helpers above).
class NmlDispatchArgumentBuilder {
public:
    void add_pointer(const void *pointer) { add_bytes(&pointer, sizeof(void *)); }
    void add_f32(f32 value) { add_bytes(&value, sizeof(f32)); }
    void add_s64(s64 value) { add_bytes(&value, sizeof(s64)); }
    void add_s32(s32 value) { add_bytes(&value, sizeof(s32)); }

    void dispatch(KernelHandle handle, LaunchConfig config) const {
        vector<const void *> args(slots_.size());
        vector<usize> sizes(slots_.size());
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
        vector<u8> storage(size);
        std::memcpy(storage.data(), value, size);
        slots_.push_back(std::move(storage));
    }

    vector<vector<u8>> slots_;
};

// ── delay-ring fold (SpikeEngine-only) -- ring_slot_count derivation + the one unified,
// ring-capable drain/propagate kernel pair. See the three REFACTOR comments in delay_ring.h/
// master_kernel.h/master_kernel.cpp this generalizes -- those files/AssembledModel are left
// completely untouched by this fold; this is new, additive capability on SpikeEngine alone. ──────

// Converts one connection's own delay (SI seconds, already unit-resolved by resolve.cpp's own
// unit_value_to_si -- see model_specification.cpp) to whole ticks against `dt_seconds`, floored to
// the engine's existing implicit one-tick latency -- the SAME math delay_ring.cpp's own (anonymous-
// namespace, so not directly reusable from here without touching that untouchable file) private
// delay_seconds_to_ticks performs, mirrored here rather than exposed from there, matching this
// file's own established "small, file-local mirror" convention (see
// nml_compute_ring_slot_count_from_weight_matrix's own doc comment just below). A plain `connection`
// with no `delay` attribute at all (ConnectionEntry::delay == 0.0, model_specification.cpp's own
// default) lands here exactly, reproducing today's implicit one-tick latency as this conversion's
// own zero-delay case.
s64 nml_delay_seconds_to_ticks(f64 delay_seconds, f32 dt_seconds) {
    if (dt_seconds <= 0.0f) {
        log::throw_runtime_error(log::logger(), "SpikeEngine (nml): dt_seconds must be > 0");
    }
    s64 ticks = (s64)std::llround(delay_seconds / (f64)dt_seconds);
    return ticks < 1 ? 1 : ticks;
}

// `nml_compute_ring_slot_count_from_weight_matrix`-equivalent of delay_ring.h's own
// compute_max_delay_ticks(model, dt_seconds): same "scan for the longest delay, floored to the
// engine's existing implicit 1-tick minimum" shape, but reading the delay straight off `weights`
// (constant_delay_ticks/using_constant_delay_ticks/edge_delay_ticks -- already whole-tick-
// denominated, weight_matrix.h) instead of walking a ModelSpecification's connections through a
// separate DelayRingAllocation (REFACTOR comment #1, delay_ring.h) -- no dt_seconds conversion is
// needed in THIS function specifically, since by the time it runs, `weights`' own delay fields are
// already whole-tick-denominated (the real SI-seconds->ticks conversion, against the SAME math
// delay_ring.cpp's own (untouched) delay_seconds_to_ticks performs, now happens upstream of this
// call, in the ModelSpecification constructor's own delay-seeding step below -- see
// nml_delay_seconds_to_ticks). Padding slots beyond a node's real degree are never written past
// their default (1), so scanning the WHOLE edge_delay_ticks array (real edges and padding together)
// gives exactly the same answer as restricting to real edges only.
s64 nml_compute_ring_slot_count_from_weight_matrix(const WeightMatrix &weights) {
    if (weights.using_constant_delay_ticks) {
        return weights.constant_delay_ticks < 1 ? 1 : weights.constant_delay_ticks;
    }
    s64 edge_element_count = weights.node_count * weights.max_neighbor_count;
    if (edge_element_count <= 0 || weights.edge_delay_ticks.pointer == nullptr) return 1;
    const s32 *delay_data = weights.edge_delay_ticks.get_contents();
    s64 max_ticks = 1;
    for (s64 index = 0; index < edge_element_count; ++index) {
        if (delay_data[index] > max_ticks) max_ticks = delay_data[index];
    }
    return max_ticks;
}

const char *const NML_UNIFIED_PROPAGATE_KERNEL_NAME = "spikecorec_engine_unified_propagate";

// The delay-ring fold's ONE compiled propagate kernel (REFACTOR comment #3, master_kernel.cpp) --
// always compiled, whether or not this model has any real per-edge delay. Adapted directly from
// master_kernel.cpp's own build_propagate_ring_kernel_gpu_source (that file/function are left
// untouched -- this is an independent copy, matching this file's own established "small, file-local
// mirror" convention above): same ring-slot scatter math (a fired neuron's own weight lands in the
// ring slot due at `tick + delay_ticks`, self-reenqueue lands at `next_tick`'s own slot), but this
// version reads delay straight off WeightMatrix's own constant_delay_ticks/using_constant_delay_ticks/
// edge_delay_ticks fields (REFACTOR comment #2, delay_ring.h) -- mirroring how it already reads
// constant_weight/using_constant_weight/U/V for the weight side -- instead of a separate
// DelayRingAllocation::edge_delay_ticks array. `ring_slot_count == 1` collapses every `% ring_slot_count`
// to slot 0 unconditionally, reproducing today's flat single-buffer propagate exactly.
nml::GpuSource nml_build_unified_propagate_kernel_gpu_source() {
    nml::GpuSource source;

    String msl_body =
        "kernel void spikecorec_engine_unified_propagate(\n"
        "    constant long        &tick                       [[ buffer(0) ]],\n"
        "    constant long        &next_tick                  [[ buffer(1) ]],\n"
        "    constant long        &ring_slot_count            [[ buffer(2) ]],\n"
        "    const device float4  *U                          [[ buffer(3) ]],\n"
        "    const device float4  *V                          [[ buffer(4) ]],\n"
        "    constant long        &rank_float4_stride         [[ buffer(5) ]],\n"
        "    constant float       &constant_weight            [[ buffer(6) ]],\n"
        "    constant int         &using_constant_weight      [[ buffer(7) ]],\n"
        "    const device uint    *internal_node_words        [[ buffer(8) ]],\n"
        "    const device uint    *leaf_node_words             [[ buffer(9) ]],\n"
        "    const device uint    *rank_superblock_table      [[ buffer(10) ]],\n"
        "    const device ushort  *rank_subblock_table        [[ buffer(11) ]],\n"
        "    constant int         &branching_factor           [[ buffer(12) ]],\n"
        "    constant int         &superblock_size_words      [[ buffer(13) ]],\n"
        "    constant int         &padded_node_count          [[ buffer(14) ]],\n"
        "    constant int         &tree_height                [[ buffer(15) ]],\n"
        "    constant int         &internal_bit_count         [[ buffer(16) ]],\n"
        "    constant long        &neuron_count               [[ buffer(17) ]],\n"
        "    constant long        &max_neighbor_count         [[ buffer(18) ]],\n"
        "    constant int         &constant_delay_ticks       [[ buffer(19) ]],\n"
        "    constant int         &using_constant_delay_ticks [[ buffer(20) ]],\n"
        "    const device int     *edge_delay_ticks           [[ buffer(21) ]],\n"
        "    device float         *network_inputs_ring        [[ buffer(22) ]],\n"
        "    device long          *last_spiked                [[ buffer(23) ]],\n"
        "    device int           *next_active_neuron_indices [[ buffer(24) ]],\n"
        "    device int           *next_active_neuron_count   [[ buffer(25) ]],\n"
        "    device int           *active_generation          [[ buffer(26) ]],\n"
        "    device bool          *emit_spike                 [[ buffer(27) ]],\n"
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
        "        int delay_ticks = using_constant_delay_ticks != 0\n"
        "            ? constant_delay_ticks\n"
        "            : edge_delay_ticks[neuron_index * max_neighbor_count + slot];\n"
        "        long arrival_tick = tick + (long)delay_ticks;\n"
        "        long target_slot = arrival_tick % ring_slot_count;\n"
        "        int arrival_tick_i = (int)arrival_tick;\n"
        "\n"
        "        device atomic_float *input_slot =\n"
        "            (device atomic_float *)(network_inputs_ring + target_slot * neuron_count + child);\n"
        "        atomic_fetch_add_explicit(input_slot, weight, memory_order_relaxed);\n"
        "\n"
        "        device atomic_int *child_generation_slot =\n"
        "            (device atomic_int *)(active_generation + target_slot * neuron_count + child);\n"
        "        int previous_child_generation =\n"
        "            atomic_exchange_explicit(child_generation_slot, arrival_tick_i, memory_order_relaxed);\n"
        "        if (previous_child_generation != arrival_tick_i) {\n"
        "            device atomic_int *count_slot = (device atomic_int *)(next_active_neuron_count + target_slot);\n"
        "            int position = atomic_fetch_add_explicit(count_slot, 1, memory_order_relaxed);\n"
        "            next_active_neuron_indices[target_slot * neuron_count + position] = child;\n"
        "        }\n"
        "    }\n"
        "\n"
        "    long self_slot = next_tick % ring_slot_count;\n"
        "    int self_tag_i = (int)next_tick;\n"
        "    device atomic_int *self_generation_slot =\n"
        "        (device atomic_int *)(active_generation + self_slot * neuron_count + neuron_index);\n"
        "    int previous_self_generation =\n"
        "        atomic_exchange_explicit(self_generation_slot, self_tag_i, memory_order_relaxed);\n"
        "    if (previous_self_generation != self_tag_i) {\n"
        "        device atomic_int *count_slot = (device atomic_int *)(next_active_neuron_count + self_slot);\n"
        "        int position = atomic_fetch_add_explicit(count_slot, 1, memory_order_relaxed);\n"
        "        next_active_neuron_indices[self_slot * neuron_count + position] = (int)neuron_index;\n"
        "    }\n"
        "}\n";

    source.msl_source =
        "#include <metal_stdlib>\nusing namespace metal;\n" + nml::k2tree_walk_preamble_msl() + "\n" + msl_body;

    String cuda_body =
        "__global__ void spikecorec_engine_unified_propagate(\n"
        "    long long             tick,\n"
        "    long long             next_tick,\n"
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
        "    int                   constant_delay_ticks,\n"
        "    int                   using_constant_delay_ticks,\n"
        "    const int             *edge_delay_ticks,\n"
        "    float                 *network_inputs_ring,\n"
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
        "        int delay_ticks = using_constant_delay_ticks != 0\n"
        "            ? constant_delay_ticks\n"
        "            : edge_delay_ticks[neuron_index * max_neighbor_count + slot];\n"
        "        long long arrival_tick = tick + (long long)delay_ticks;\n"
        "        long long target_slot = arrival_tick % ring_slot_count;\n"
        "        int arrival_tick_i = (int)arrival_tick;\n"
        "\n"
        "        atomicAdd(&network_inputs_ring[target_slot * neuron_count + child], weight);\n"
        "\n"
        "        int previous_child_generation =\n"
        "            atomicExch(&active_generation[target_slot * neuron_count + child], arrival_tick_i);\n"
        "        if (previous_child_generation != arrival_tick_i) {\n"
        "            int position = atomicAdd(&next_active_neuron_count[target_slot], 1);\n"
        "            next_active_neuron_indices[target_slot * neuron_count + position] = child;\n"
        "        }\n"
        "    }\n"
        "\n"
        "    long long self_slot = next_tick % ring_slot_count;\n"
        "    int self_tag_i = (int)next_tick;\n"
        "    int previous_self_generation =\n"
        "        atomicExch(&active_generation[self_slot * neuron_count + neuron_index], self_tag_i);\n"
        "    if (previous_self_generation != self_tag_i) {\n"
        "        int position = atomicAdd(&next_active_neuron_count[self_slot], 1);\n"
        "        next_active_neuron_indices[self_slot * neuron_count + position] = (int)neuron_index;\n"
        "    }\n"
        "}\n";

    source.cuda_source = "#include <vector_types.h>\n" + nml::k2tree_walk_preamble_cuda() + "\n" + cuda_body;

    source.functions = {nml::GpuFunctionSignature{
        NML_UNIFIED_PROPAGATE_KERNEL_NAME,
        {"tick", "next_tick", "ring_slot_count", "U", "V", "rank_float4_stride", "constant_weight",
         "using_constant_weight", "internal_node_words", "leaf_node_words", "rank_superblock_table",
         "rank_subblock_table", "branching_factor", "superblock_size_words", "padded_node_count",
         "tree_height", "internal_bit_count", "neuron_count", "max_neighbor_count", "constant_delay_ticks",
         "using_constant_delay_ticks", "edge_delay_ticks", "network_inputs_ring", "last_spiked",
         "next_active_neuron_indices", "next_active_neuron_count", "active_generation", "emit_spike"}}};
    return source;
}

} // namespace

SpikeEngine::NmlResolvedArgument SpikeEngine::resolve_nml_cell_tick_argument(
    const String &parameter_name, const nml::IrProgram &program, s32 type_library_index,
    s32 neuron_index_begin, s64 cell_state_chunk_base_offset, s32 population_size,
    u32 *rng_state_base, const UnorderedMap<String, GpuPointer<bool>> &emit_port_flags
) const {
    NmlResolvedArgument argument;

    if (parameter_name == "dt") {
        argument.kind = NmlResolvedArgument::Kind::Dt;
        return argument;
    }
    if (parameter_name == "tick") {
        argument.kind = NmlResolvedArgument::Kind::Tick;
        return argument;
    }
    if (parameter_name == "network_inputs") {
        // delay-ring fold: network_inputs is now ring-shaped -- which slot this population reads
        // from rotates every tick (tick % nml_ring_slot_count_), so only neuron_index_begin is
        // cached here; step_tick recomputes the real pointer against the CURRENT ring slot.
        argument.kind = NmlResolvedArgument::Kind::NetworkInputsRingOffset;
        argument.neuron_index_begin = neuron_index_begin;
        return argument;
    }
    if (parameter_name == "rng_state") {
        if (rng_state_base == nullptr) {
            log::throw_runtime_error(*logger,
                "SpikeEngine (nml): cell-type kernel parameter 'rng_state' (rand/randn) needs "
                "rng_state to have been allocated -- this should not happen since the constructor "
                "scans for rng_state usage before compiling any kernel");
        }
        argument.kind = NmlResolvedArgument::Kind::FixedPointer;
        argument.fixed_pointer = rng_state_base + neuron_index_begin;
        return argument;
    }
    if (parameter_name == "neuron_count") {
        argument.kind = NmlResolvedArgument::Kind::PopulationSize;
        return argument;
    }
    if (parameter_name.rfind("emit_", 0) == 0) {
        String port_name = parameter_name.substr(5);
        auto found = emit_port_flags.find(port_name);
        if (found == emit_port_flags.end()) {
            log::throw_runtime_error(*logger,
                "SpikeEngine (nml): no emit-port flag buffer allocated for port '" + port_name +
                "' (parameter '" + parameter_name + "')");
        }
        argument.kind = NmlResolvedArgument::Kind::FixedPointer;
        argument.fixed_pointer = found->second.get_contents() + neuron_index_begin;
        return argument;
    }

    for (const auto &directive : program.alloc) {
        if (const auto *state = std::get_if<nml::StateDirective>(&directive)) {
            if (state->name != parameter_name) continue;
            s32 offset_within_type = nml_state_variable_offset(program, parameter_name);
            const f32 *base = nml_allocation_.cell_state.get_contents() + cell_state_chunk_base_offset +
                        (s64)offset_within_type * population_size;
            argument.kind = NmlResolvedArgument::Kind::FixedPointer;
            argument.fixed_pointer = base;
            return argument;
        }
        if (const auto *accum = std::get_if<nml::AccumDirective>(&directive)) {
            if (accum->name != parameter_name) continue;
            auto found = nml_allocation_.accumulators.find(nml::type_scoped_key(type_library_index, parameter_name));
            if (found == nml_allocation_.accumulators.end()) {
                log::throw_runtime_error(*logger, "SpikeEngine (nml): accum '" + parameter_name +
                    "' has no allocated buffer in ModelAllocation::accumulators");
            }
            argument.kind = NmlResolvedArgument::Kind::FixedPointer;
            argument.fixed_pointer = found->second.get_contents() + neuron_index_begin;
            return argument;
        }
        if (const auto *param_dynamic = std::get_if<nml::ParamDynamicDirective>(&directive)) {
            if (param_dynamic->name != parameter_name) continue;
            auto found = nml_allocation_.dynamic_parameter_arrays.find(
                nml::type_scoped_key(type_library_index, parameter_name));
            if (found == nml_allocation_.dynamic_parameter_arrays.end()) {
                log::throw_runtime_error(*logger, "SpikeEngine (nml): param:dyn '" + parameter_name +
                    "' has no allocated buffer in ModelAllocation::dynamic_parameter_arrays");
            }
            argument.kind = NmlResolvedArgument::Kind::FixedPointer;
            argument.fixed_pointer = found->second.get_contents() + neuron_index_begin;
            return argument;
        }
        if (const auto *regime = std::get_if<nml::RegimeDirective>(&directive)) {
            if (regime->name != parameter_name) continue;
            if (!nml_allocation_.has_regime_index) {
                log::throw_runtime_error(*logger,
                    "SpikeEngine (nml): regime '" + parameter_name + "' has no allocated regime_indices buffer");
            }
            argument.kind = NmlResolvedArgument::Kind::FixedPointer;
            argument.fixed_pointer = nml_allocation_.regime_indices.get_contents() + neuron_index_begin;
            return argument;
        }
        if (const auto *expose = std::get_if<nml::ExposeDirective>(&directive)) {
            if (expose->name != parameter_name) continue;
            auto found = nml_allocation_.derived_exposure_scratch_buffers.find(
                nml::type_scoped_key(type_library_index, parameter_name));
            if (found == nml_allocation_.derived_exposure_scratch_buffers.end()) {
                log::throw_runtime_error(*logger, "SpikeEngine (nml): expose '" + parameter_name +
                    "' has no derived-exposure-scratch buffer (and no matching state slot resolved it "
                    "first) -- ModelAllocation::derived_exposure_scratch_buffers");
            }
            argument.kind = NmlResolvedArgument::Kind::FixedPointer;
            argument.fixed_pointer = found->second.get_contents() + neuron_index_begin;
            return argument;
        }
        if (const auto *require = std::get_if<nml::RequireDirective>(&directive)) {
            if (require->name != parameter_name) continue;
            log::throw_runtime_error(*logger, "SpikeEngine (nml): cell-type kernel parameter '" + parameter_name +
                "' resolves to a `require` binding -- not supported by this dispatch (out of Phase-1 "
                "GLIF-cell scope; see master_kernel.h)");
        }
        if (const auto *param_constant = std::get_if<nml::ParamConstantDirective>(&directive)) {
            if (param_constant->name != parameter_name) continue;
            log::throw_runtime_error(*logger, "SpikeEngine (nml): cell-type kernel parameter '" + parameter_name +
                "' is an un-baked (bare) `param` -- allocate_model has no established value source for "
                "this case (see master_kernel.h)");
        }
    }

    log::throw_runtime_error(*logger, "SpikeEngine (nml): cell-type kernel parameter '" + parameter_name +
        "' is not a recognized reserved name, .alloc name, or emit port for this dispatch -- likely the "
        "k^2-tree-walk/shared-basis block (forall/loadedge/accedge), out of scope for a Phase-1 "
        "GLIF-family cell (see master_kernel.h)");
}

SpikeEngine::SpikeEngine(nml::ModelSpecification &model, const Vector<nml::IrProgram> &type_library_ir_programs,
                          f32 dt_seconds)
    : logger(make_logger())
    , weights(nml::build_weight_matrix_from_projections(model))
    , nml_allocation_(nml::allocate_model(model, type_library_ir_programs))
    , neuron_count(model.total_neuron_count)
    , input_neuron_count(0)
    , thread_count_per_block(256)
    , block_count(0)
    , resting_membrane_potential(0.0f)
    , decay_rate(0.0f)
    , learning_rate(0.0f)
    , spike_period(1)
    , spike_threshold(1.0f)
    , alive(true)
    , active_set_optimization_enabled(false)
    , nml_mode_enabled_(true)
{
    // Owned copy, kept only for Stage 2's own lazy first-step_tick-call synapse-dispatch-topology
    // build (ensure_nml_synapse_dispatch_topology_built) -- see engine.h's own doc comment on
    // nml_type_library_ir_programs_ for why this can't be resolved away at construction time the way
    // Stage 1's cell-tick dispatch plans are.
    nml_type_library_ir_programs_ = type_library_ir_programs;

    block_count = (s32) ((neuron_count + thread_count_per_block - 1) / thread_count_per_block);

    // ── delay-ring fold (SpikeEngine-only) -- seed `weights`' own per-edge delay from
    // model.projections' ConnectionEntry::delay (real SI seconds, converted to whole ticks against
    // this model's own `dt_seconds` via nml_delay_seconds_to_ticks -- the SAME math delay_ring.cpp's
    // own, separate, untouched compute_max_delay_ticks/allocate_delay_ring use), then determine
    // nml_ring_slot_count_ from `weights` (REFACTOR comment #1, delay_ring.h: delay lives in
    // WeightMatrix now, not a separate DelayRingAllocation). A connection whose delay converts to
    // <= 1 tick (including the "no delay attribute given" default of 0.0 seconds) leaves its edge at
    // weights' own already-established 1-tick baseline, so an ordinary model (no caller-configured
    // delay) sees nml_ring_slot_count_ == 1 -- exactly today's flat, single-buffer behavior.
    //
    // Mirrors WeightMatrix's own using_constant_weight/constant_weight vs U*V split (weight_matrix.h)
    // for the uniform case: if every connection converts to the SAME whole-tick count, that single
    // value is set via the simpler weights.set_constant_delay_ticks(...) path (also covers "no
    // connections at all", which trivially leaves weights at its own default constant 1-tick delay);
    // only a model whose connections actually disagree on delay falls through to the true per-edge
    // weights.set_edge_delay_ticks(...) path (skipping any edge whose own delay already matches the
    // 1-tick default every slot starts at).
    {
        bool any_connection_seen = false;
        bool all_connections_share_one_delay = true;
        s64 shared_delay_ticks = 1;
        for (const nml::ProjectionEntry &projection : model.projections) {
            for (const nml::ConnectionEntry &connection : projection.connections) {
                s64 delay_ticks = nml_delay_seconds_to_ticks(connection.delay, dt_seconds);
                if (!any_connection_seen) {
                    shared_delay_ticks = delay_ticks;
                    any_connection_seen = true;
                } else if (delay_ticks != shared_delay_ticks) {
                    all_connections_share_one_delay = false;
                }
            }
        }

        if (any_connection_seen && all_connections_share_one_delay) {
            weights.set_constant_delay_ticks((s32)shared_delay_ticks);
        } else if (!all_connections_share_one_delay) {
            weights.using_constant_delay_ticks = false;
            for (const nml::ProjectionEntry &projection : model.projections) {
                for (const nml::ConnectionEntry &connection : projection.connections) {
                    s64 delay_ticks = nml_delay_seconds_to_ticks(connection.delay, dt_seconds);
                    if (delay_ticks <= 1) continue; // already the array's own 1-tick default
                    weights.set_edge_delay_ticks(connection.source_neuron_index, connection.target_neuron_index,
                                                  (s32)delay_ticks);
                }
            }
        }
    }
    nml_ring_slot_count_ = nml_compute_ring_slot_count_from_weight_matrix(weights);

    usize neuron_f32_byte_size = (usize) neuron_count * sizeof(f32);
    usize neuron_s32_byte_size = (usize) neuron_count * sizeof(s32);
    usize neuron_s64_byte_size = (usize) neuron_count * sizeof(s64);
    usize ring_f32_byte_size = (usize) nml_ring_slot_count_ * neuron_f32_byte_size;
    usize ring_s32_byte_size = (usize) nml_ring_slot_count_ * neuron_s32_byte_size;
    usize ring_slot_count_s32_byte_size = (usize) nml_ring_slot_count_ * sizeof(s32);

    // ── the 5 engine-owned buffers an NML-derived model actually needs (matches
    // examples/nml_pipeline_support.h's own make_live_model_buffers seeding convention, NOT the
    // hardcoded-LIF constructors' convention above -- last_spiked is seeded to -1 ("never fired"),
    // not 0, since an NML `_tick` kernel checks last_spiked differently than the hardcoded LIF
    // kernel does). membrane_potentials/last_tick_updated/active_neuron_indices/active_neuron_count/
    // input_staging/override_staging are hardcoded-LIF-only buffers, left unallocated -- the NML
    // step_tick path below never reads them.
    //
    // network_inputs/next_active_neuron_indices/active_generation are ring-shaped
    // [nml_ring_slot_count_ * neuron_count] (next_active_neuron_count is [nml_ring_slot_count_])
    // -- the delay-ring fold (see nml_ring_slot_count_'s own doc comment, engine.h). Every slot
    // starts zeroed/reset exactly like the old single flat buffer did, so ring_slot_count == 1
    // reproduces that flat buffer's own initial state byte for byte. ──
    network_inputs = allocate<f32>(ring_f32_byte_size);
    memset(network_inputs.get_contents(), 0, ring_f32_byte_size);

    last_spiked = allocate<s64>(neuron_s64_byte_size);
    std::fill(last_spiked.get_contents(), last_spiked.get_contents() + neuron_count, (s64)-1);

    next_active_neuron_indices = allocate<s32>(ring_s32_byte_size);

    next_active_neuron_count = allocate<s32>(ring_slot_count_s32_byte_size);
    memset(next_active_neuron_count.get_contents(), 0, ring_slot_count_s32_byte_size);

    active_generation = allocate<s32>(ring_s32_byte_size);
    std::fill(active_generation.get_contents(),
              active_generation.get_contents() + nml_ring_slot_count_ * neuron_count, (s32)-1);

    prefetch_to_gpu(network_inputs, ring_f32_byte_size);
    prefetch_to_gpu(last_spiked, neuron_s64_byte_size);
    prefetch_to_gpu(next_active_neuron_indices, ring_s32_byte_size);
    prefetch_to_gpu(next_active_neuron_count, ring_slot_count_s32_byte_size);
    prefetch_to_gpu(active_generation, ring_s32_byte_size);

    // ── assemble + compile the master kernel (ticket #6's own assembly, ported from
    // nml::AssembledModel's constructor -- see master_kernel.cpp). Only population_gpu_sources is
    // used below -- the flat drain_network_inputs_source/propagate_source this also builds are
    // unused (this fold always compiles its OWN unified, ring-capable drain/propagate kernels
    // instead, see below and this file's own "delay-ring fold" helpers above), matching REFACTOR
    // comment #3 (master_kernel.cpp): one drain kernel and one propagate kernel, always, whether
    // or not this model has any real per-edge delay. ──
    nml::AssembledMasterKernelSource assembled = nml::assemble_master_kernel_source(model, type_library_ir_programs);

    // build_drain_ring_kernel_gpu_source (master_kernel.h, unmodified) is already exactly the
    // unified drain kernel this fold needs: it zeroes network_inputs_ring[current_slot], which for
    // ring_slot_count == 1 is always slot 0 -- precisely today's flat drain.
    nml::GpuSource unified_drain_source = nml::build_drain_ring_kernel_gpu_source();
    nml_drain_kernel_ = nml::compile_kernel_or_throw_with_source(
        nml_source_text_for_this_backend(unified_drain_source),
        unified_drain_source.functions.at(0).function_name,
        "the engine-fixed unified (delay-ring-capable) deliver-drain kernel", "");

    nml::GpuSource unified_propagate_source = nml_build_unified_propagate_kernel_gpu_source();
    nml_propagate_kernel_ = nml::compile_kernel_or_throw_with_source(
        nml_source_text_for_this_backend(unified_propagate_source),
        unified_propagate_source.functions.at(0).function_name,
        "the engine-fixed unified (delay-ring-capable) propagate kernel", "");

    nml_emit_port_names_ = nml::collect_emit_port_names(model, type_library_ir_programs);
    for (const String &port_name : nml_emit_port_names_) {
        GpuPointer<bool> flags = allocate<bool>((usize)neuron_count * sizeof(bool));
        memset(flags.get_contents(), 0, (usize)neuron_count * sizeof(bool));
        prefetch_to_gpu(flags, (usize)neuron_count * sizeof(bool));
        nml_emit_port_flags_.emplace(port_name, std::move(flags));
    }

    // rng_state (ticket #65 [F4]) is only needed if some population's kernel actually references
    // rand/randn -- scanned up front so the fixed-pointer resolution below can bind it once,
    // matching examples/nml_pipeline_support.h's own make_live_model_buffers seeding scheme.
    bool any_population_needs_rng_state = false;
    for (const auto &source : assembled.population_gpu_sources) {
        if (source.functions.empty()) continue;
        for (const String &parameter_name : source.functions[0].parameter_names_in_order) {
            if (parameter_name == "rng_state") { any_population_needs_rng_state = true; break; }
        }
        if (any_population_needs_rng_state) break;
    }

    u32 *rng_state_base = nullptr;
    if (any_population_needs_rng_state) {
        nml_rng_state_ = allocate<u32>((usize)neuron_count * sizeof(u32));
        for (s64 neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
            // xorshift32 is stuck at zero forever once it reaches zero -- every seed must be
            // nonzero and per-neuron distinct (same scheme make_live_model_buffers uses).
            nml_rng_state_.get_contents()[neuron_index] = (u32)((neuron_index + 1) * 2654435761u) | 1u;
        }
        prefetch_to_gpu(nml_rng_state_, (usize)neuron_count * sizeof(u32));
        rng_state_base = nml_rng_state_.get_contents();
    }

    // ── precompute each population's dispatch plan (compiled kernel + resolved argument list) so
    // step_tick() below never needs ModelAllocation/IrProgram/ModelSpecification again (satisfies
    // master_kernel.h's own "the compiled IR should just be a list of instructions" REFACTOR note
    // above class AssembledModel). ──
    const s64 *chunk_base_offsets = nml_allocation_.cell_type_boundaries.get_contents();

    for (usize population_index = 0; population_index < model.populations.size(); ++population_index) {
        const nml::PopulationEntry &population = model.populations[population_index];
        const nml::GpuSource &source = assembled.population_gpu_sources[population_index];

        // Recorded unconditionally (not just for populations with a per-neuron kernel) -- Stage 2's
        // synapse dispatch needs this for a projection's POSTSYNAPTIC population, which is not
        // necessarily one this loop below builds a NmlPopulationDispatchPlan for.
        nml_populations_.push_back(
            NmlPopulationInfo{population.type_library_index, population.neuron_index_begin, population.size});

        if (source.functions.empty()) continue; // no per-neuron content -- nothing to dispatch

        const nml::IrProgram &program = type_library_ir_programs[(usize)population.type_library_index];

        NmlPopulationDispatchPlan plan;
        plan.kernel_handle = nml::compile_kernel_or_throw_with_source(
            nml_source_text_for_this_backend(source), source.functions[0].function_name,
            "population '" + population.id + "' (ComponentType '" + program.component_type_name + "')",
            nml::print_ir_program(program));
        plan.population_size = population.size;
        plan.arguments.reserve(source.functions[0].parameter_names_in_order.size());
        for (const String &parameter_name : source.functions[0].parameter_names_in_order) {
            plan.arguments.push_back(resolve_nml_cell_tick_argument(
                parameter_name, program, population.type_library_index, population.neuron_index_begin,
                chunk_base_offsets[population_index], population.size, rng_state_base,
                nml_emit_port_flags_));
        }
        nml_population_dispatch_plans_.push_back(std::move(plan));
    }

    // ── Stage 2 (ticket #131): spike-scatter batch-construction subsystem -- compile every USED
    // synapse type's edge-parallel functions once here; topology/matrix-index registration + the
    // per-projection dispatch-argument precomputation is deferred to the first step_tick call (see
    // ensure_nml_synapse_dispatch_topology_built), since it needs `weights`'s own real k^2-tree,
    // ported from nml::AssembledModel's own constructor (master_kernel.cpp). Deliberately NOT
    // activated when nml_ring_slot_count_ > 1 (real per-edge delay configured) -- mirrors
    // nml::AssembledModel's own established enable_delay_ring precedent (master_kernel.cpp's
    // constructor: "ticket #64's delay ring and this ticket's synapse dispatch have not been
    // integrated with each other") exactly, since this Stage 2 dispatch's own resolved
    // "network_inputs" argument (resolve_nml_synapse_edge_argument) is a single FIXED pointer
    // baked once at topology-build time, not ring-slot-aware -- untested/unsafe for a model with
    // real ring rotation. `projections_` is simply left empty in that case, so step_tick's own
    // `!nml_projections_.empty()` gate leaves this fold's own ring-mode behavior exactly what
    // nml::AssembledModel's ring mode already established. ──────────────────────────────────────
    nml_projections_ = (nml_ring_slot_count_ > 1) ? Vector<nml::ProjectionEntry>{} : model.projections;
    nml_projection_edge_topology_.resize(nml_projections_.size());

    for (const nml::ProjectionEntry &projection : nml_projections_) {
        if (projection.postsynaptic_population_index < 0 ||
            (usize)projection.postsynaptic_population_index >= model.populations.size()) {
            log::throw_runtime_error(*logger,
                "SpikeEngine (nml): projection '" + projection.id + "' has an out-of-range "
                "postsynaptic_population_index (" + std::to_string(projection.postsynaptic_population_index) + ")");
        }
        s32 synapse_type_index = projection.synapse_type_library_index;
        if (synapse_type_index < 0 || (usize)synapse_type_index >= model.type_library.size()) {
            log::throw_runtime_error(*logger,
                "SpikeEngine (nml): projection '" + projection.id + "' has an out-of-range "
                "synapse_type_library_index (" + std::to_string(synapse_type_index) + ")");
        }
        if (model.type_library[(usize)synapse_type_index].category != nml::TypeLibraryCategory::Synapse) {
            log::throw_runtime_error(*logger,
                "SpikeEngine (nml): projection '" + projection.id + "' references type_library[" +
                std::to_string(synapse_type_index) + "], which is not a Synapse-category entry");
        }
        if (nml_synapse_types_by_type_library_index_.count(synapse_type_index) > 0) continue; // already compiled

        const nml::IrProgram &synapse_program = type_library_ir_programs[(usize)synapse_type_index];
        nml::GpuSource synapse_source = nml::lower_synapse_ir_program_to_gpu_source(synapse_program);
        if (synapse_source.functions.empty()) {
            log::throw_runtime_error(*logger,
                "SpikeEngine (nml): synapse type '" + synapse_program.component_type_name +
                "' lowered to zero GPU functions (lower_synapse_ir_program_to_gpu_source should always emit "
                "at least the `_integrate_edges` function)");
        }

        SynapseTypeRuntimeInfo synapse_info;
        for (usize function_index = 0; function_index < synapse_source.functions.size(); ++function_index) {
            const nml::GpuFunctionSignature &signature = synapse_source.functions[function_index];
            bool is_integrate_edges_function = (function_index + 1 == synapse_source.functions.size());
            String label = "synapse type '" + synapse_program.component_type_name + "' function '" +
                            signature.function_name + "'";
            KernelHandle handle = nml::compile_kernel_or_throw_with_source(
                nml_source_text_for_this_backend(synapse_source), signature.function_name, label,
                nml::print_ir_program(synapse_program));
            if (is_integrate_edges_function) {
                synapse_info.integrate_edges_handle = handle;
                synapse_info.integrate_edges_parameter_names_in_order = signature.parameter_names_in_order;
            } else {
                synapse_info.deliver_functions.push_back(
                    DeliverFunctionRuntimeInfo{handle, signature.parameter_names_in_order});
            }
        }
        nml_synapse_types_by_type_library_index_.emplace(synapse_type_index, std::move(synapse_info));
    }

    logger->info("SpikeEngine (nml) constructed: neuron_count={} populations={} emit_ports={} projections={}",
                  neuron_count, model.populations.size(), nml_emit_port_names_.size(), nml_projections_.size());
}

void SpikeEngine::step_tick(f32 dt, s64 tick, s64 next_tick) {
    if (!nml_mode_enabled_) {
        log::throw_runtime_error(*logger,
            "SpikeEngine::step_tick: this engine was not constructed via the NML ModelSpecification "
            "constructor");
    }

    // ── Stage 2 (ticket #131): topology is built lazily here (needs `weights`'s real k^2-tree, only
    // available once the engine is already constructed) -- see master_kernel.h's own doc comment for
    // the full derivation of why this ordering (topology build before the population dispatch loop,
    // the two synapse dispatches themselves after it, alongside the fixed propagate stage) matters. ──
    if (!nml_projections_.empty()) {
        ensure_nml_synapse_dispatch_topology_built();
    }

    // ── delay-ring fold: which ring slot this tick's population dispatch reads its own
    // network_inputs from -- current_ring_slot is always 0 when nml_ring_slot_count_ == 1 (the
    // ordinary "no delay configured" case), collapsing to exactly today's flat single-buffer
    // behavior (see engine.h's own doc comment on nml_ring_slot_count_).
    s64 current_ring_slot = tick % nml_ring_slot_count_;
    f32 *network_inputs_ring_base = network_inputs.get_contents();

    // stages 2-5: one dispatch per population with a per-neuron kernel, over its own full neuron
    // range (arch §4.1's cell-type-boundary dispatch) -- mirrors
    // nml::AssembledModel::step_tick's own population dispatch loop (master_kernel.cpp), reading
    // straight off the precomputed plan instead of re-deriving each argument's binding every tick.
    for (const NmlPopulationDispatchPlan &plan : nml_population_dispatch_plans_) {
        NmlDispatchArgumentBuilder builder;
        for (const NmlResolvedArgument &argument : plan.arguments) {
            switch (argument.kind) {
                case NmlResolvedArgument::Kind::Dt:
                    builder.add_f32(dt);
                    break;
                case NmlResolvedArgument::Kind::Tick:
                    builder.add_s64(tick);
                    break;
                case NmlResolvedArgument::Kind::PopulationSize:
                    builder.add_s64((s64)plan.population_size);
                    break;
                case NmlResolvedArgument::Kind::FixedPointer:
                    builder.add_pointer(argument.fixed_pointer);
                    break;
                case NmlResolvedArgument::Kind::NetworkInputsRingOffset:
                    builder.add_pointer(network_inputs_ring_base + current_ring_slot * neuron_count +
                                         argument.neuron_index_begin);
                    break;
            }
        }
        builder.dispatch(plan.kernel_handle, nml_launch_config_for(plan.population_size));
    }

    // fixed deliver-drain: this tick's own ring slot has now been read by every population's own
    // `_tick` kernel this tick -- zero just that slot so the propagate stage below writes THIS
    // tick's fresh contributions, not an accumulation on top of what was already consumed
    // (ir_spec.md §3.5's >=1-tick latency). ring_slot_count == 1 always targets slot 0, exactly
    // today's flat drain.
    {
        NmlDispatchArgumentBuilder builder;
        builder.add_pointer(network_inputs.get_contents());
        builder.add_s64(neuron_count);
        builder.add_s64(current_ring_slot);
        builder.dispatch(nml_drain_kernel_, nml_launch_config_for(neuron_count));
    }

    // Reset this tick's own ring slot's active-set enqueue counter to 0 before this tick's
    // propagate dispatches run -- matches nml::AssembledModel::step_tick's own per-tick reset,
    // ring-generalized (delay is always >= 1, so this tick's own propagate dispatch below can
    // never target this SAME slot -- see nml::DelayRingAllocation's own ring-mode step_tick for the
    // identical reasoning, master_kernel.cpp).
    next_active_neuron_count.get_contents()[current_ring_slot] = 0;

    // ── Stage 2 (ticket #131): real per-edge synapse dispatch -- delivery-event construction +
    // `_deliver_<port>` dispatch (bumps Sk for this tick's fresh spikes), immediately followed by
    // `_integrate_edges` (reads that same, now-fresh Sk, decays, and writes the finished current into
    // network_inputs, already drained above) -- so by the time step_tick returns, network_inputs
    // already reflects this tick's synaptic contribution, matching the fixed scalar propagate stage's
    // own observable timing. Must run BEFORE the fixed scalar propagate dispatch below, which
    // reads+clears nml_emit_port_flags_ for its own last_spiked/active-set bookkeeping (this dispatch
    // reads, but does not clear, the SAME flags). Never runs when nml_ring_slot_count_ > 1 --
    // nml_projections_ is forced empty at construction time for that case (see this file's own
    // constructor doc comment on this exact restriction). ──
    if (!nml_projections_.empty()) {
        dispatch_nml_synapse_delivery_events(dt, tick);
        dispatch_nml_synapse_integrate_edges(dt, tick);
    }

    // fixed k^2-tree propagate/scatter + active-set-enqueue (stage 6/9): one dispatch per distinct
    // emit-port name, each over the whole model's neuron range. A model with real per-edge synapse
    // projections (ticket #131) has this dispatch's OWN weight contribution forced to zero -- `weights`
    // itself is left untouched -- since the two synapse dispatches above already supply the real
    // per-edge current; last_spiked + active-set enqueue are unaffected.
    f32 constant_weight_for_this_dispatch = nml_projections_.empty() ? weights.constant_weight : 0.0f;
    s32 using_constant_weight_for_this_dispatch =
        nml_projections_.empty() ? (weights.using_constant_weight ? 1 : 0) : 1;
    // delay-ring fold: read straight off `weights`' own delay configuration (REFACTOR comment #2,
    // delay_ring.h) -- edge_delay_ticks may be a null GpuPointer for an edge-free WeightMatrix
    // (node_count * max_neighbor_count == 0), so guard the get_contents() call (never dereferenced
    // by the kernel either way when max_neighbor_count <= 0).
    const s32 *edge_delay_ticks_pointer =
        weights.edge_delay_ticks.pointer != nullptr ? weights.edge_delay_ticks.get_contents() : nullptr;
    for (const String &port_name : nml_emit_port_names_) {
        auto found = nml_emit_port_flags_.find(port_name);
        if (found == nml_emit_port_flags_.end()) {
            log::throw_runtime_error(*logger,
                "SpikeEngine::step_tick: no emit-port flag buffer allocated for port '" + port_name + "'");
        }

        NmlDispatchArgumentBuilder builder;
        builder.add_s64(tick);
        builder.add_s64(next_tick);
        builder.add_s64(nml_ring_slot_count_);
        builder.add_pointer(weights.U_matrix.get_contents());
        builder.add_pointer(weights.V_matrix.get_contents());
        builder.add_s64(weights.rank_float4_stride);
        builder.add_f32(constant_weight_for_this_dispatch);
        builder.add_s32(using_constant_weight_for_this_dispatch);
        builder.add_pointer(weights.k2tree.internal_node_words.get_contents());
        builder.add_pointer(weights.k2tree.leaf_node_words.get_contents());
        builder.add_pointer(weights.k2tree.rank_superblock_table.get_contents());
        builder.add_pointer(weights.k2tree.rank_subblock_table.get_contents());
        builder.add_s32(weights.k2tree.branching_factor);
        builder.add_s32(weights.k2tree.superblock_size_words);
        builder.add_s32(weights.k2tree.padded_node_count);
        builder.add_s32(weights.k2tree.tree_height);
        builder.add_s32(weights.k2tree.internal_bit_count);
        builder.add_s64(neuron_count);
        builder.add_s64(weights.max_neighbor_count);
        builder.add_s32(weights.constant_delay_ticks);
        builder.add_s32(weights.using_constant_delay_ticks ? 1 : 0);
        builder.add_pointer(edge_delay_ticks_pointer);
        builder.add_pointer(network_inputs.get_contents());
        builder.add_pointer(last_spiked.get_contents());
        builder.add_pointer(next_active_neuron_indices.get_contents());
        builder.add_pointer(next_active_neuron_count.get_contents());
        builder.add_pointer(active_generation.get_contents());
        builder.add_pointer(found->second.get_contents());
        builder.dispatch(nml_propagate_kernel_, nml_launch_config_for(neuron_count));
    }

    // ── ticket #132: stage 7 STDP plasticity -- runs immediately after propagate above, once every
    // emit port's dispatch has finished writing this tick's own last_spiked entries. No-op when
    // plasticity is disabled (learning_rate == 0.0f).
    apply_nml_stdp_plasticity(tick);
}

void SpikeEngine::force_emit(const String &port_name, s64 neuron_index) {
    if (!nml_mode_enabled_) {
        log::throw_runtime_error(*logger,
            "SpikeEngine::force_emit: this engine was not constructed via the NML ModelSpecification "
            "constructor");
    }

    auto found = nml_emit_port_flags_.find(port_name);
    if (found == nml_emit_port_flags_.end()) {
        log::throw_runtime_error(*logger,
            "SpikeEngine::force_emit: '" + port_name + "' is not a known emit port for this model");
    }

    if (neuron_index < 0 || neuron_index >= neuron_count) {
        log::throw_runtime_error(*logger,
            "SpikeEngine::force_emit: neuron_index " + std::to_string(neuron_index) +
            " is out of range for neuron_count=" + std::to_string(neuron_count));
    }

    found->second.get_contents()[neuron_index] = true;
}

// ── Stage 2 of folding nml::AssembledModel into SpikeEngine: real per-edge synapse dispatch (ticket
// #131) -- ported from nml::AssembledModel's own (anonymous-namespace) append_synapse_edge_argument/
// ensure_synapse_dispatch_topology_built/dispatch_synapse_integrate_edges/
// dispatch_synapse_delivery_events (src/nml/master_kernel.cpp). ──────────────────────────────────────

SpikeEngine::NmlResolvedSynapseArgument SpikeEngine::resolve_nml_synapse_edge_argument(
    const String &parameter_name, const nml::IrProgram &synapse_program,
    const UnorderedMap<String, s64> &matrix_index_by_peredge_name,
    const nml::IrProgram &postsynaptic_cell_program, s32 postsynaptic_population_size,
    s32 postsynaptic_neuron_index_begin, s64 postsynaptic_chunk_base_offset,
    const s32 *source_node_indices_pointer, const s32 *target_node_indices_pointer,
    const s32 *edge_slot_indices_pointer
) const {
    NmlResolvedSynapseArgument argument;

    if (parameter_name == "dt") {
        argument.kind = NmlResolvedSynapseArgument::Kind::Dt;
        return argument;
    }
    if (parameter_name == "tick") {
        argument.kind = NmlResolvedSynapseArgument::Kind::Tick;
        return argument;
    }
    if (parameter_name == "event_count") {
        argument.kind = NmlResolvedSynapseArgument::Kind::EventCount;
        return argument;
    }
    if (parameter_name == "network_inputs") {
        argument.kind = NmlResolvedSynapseArgument::Kind::FixedPointer;
        argument.fixed_pointer = network_inputs.get_contents();
        return argument;
    }
    if (parameter_name == "source_node_indices") {
        argument.kind = NmlResolvedSynapseArgument::Kind::FixedPointer;
        argument.fixed_pointer = source_node_indices_pointer;
        return argument;
    }
    if (parameter_name == "target_node_indices") {
        argument.kind = NmlResolvedSynapseArgument::Kind::FixedPointer;
        argument.fixed_pointer = target_node_indices_pointer;
        return argument;
    }
    if (parameter_name == "edge_slot_indices") {
        argument.kind = NmlResolvedSynapseArgument::Kind::FixedPointer;
        argument.fixed_pointer = edge_slot_indices_pointer;
        return argument;
    }
    if (parameter_name == "U") {
        argument.kind = NmlResolvedSynapseArgument::Kind::FixedPointer;
        argument.fixed_pointer = weights.U_matrix.get_contents();
        return argument;
    }
    if (parameter_name == "V") {
        argument.kind = NmlResolvedSynapseArgument::Kind::FixedPointer;
        argument.fixed_pointer = weights.V_matrix.get_contents();
        return argument;
    }
    if (parameter_name == "rank_float4_stride") {
        argument.kind = NmlResolvedSynapseArgument::Kind::RankFloat4Stride;
        return argument;
    }
    if (parameter_name == "max_neighbor_count") {
        argument.kind = NmlResolvedSynapseArgument::Kind::MaxNeighborCount;
        return argument;
    }
    if (parameter_name == "rng_state") {
        log::throw_runtime_error(*logger,
            "SpikeEngine (nml): synapse dispatch: parameter 'rng_state' (rand/randn) is not supported -- "
            "no Phase-1 synapse ComponentType references it (see master_kernel.h)");
    }
    if (parameter_name.rfind("coefficients_", 0) == 0) {
        String peredge_name = parameter_name.substr(String("coefficients_").size());
        auto found = matrix_index_by_peredge_name.find(peredge_name);
        if (found == matrix_index_by_peredge_name.end()) {
            log::throw_runtime_error(*logger, "SpikeEngine (nml): synapse dispatch: peredge '" + peredge_name +
                "' has no registered WeightMatrix matrix_index");
        }
        argument.kind = NmlResolvedSynapseArgument::Kind::FixedPointer;
        argument.fixed_pointer = weights.coefficient_vectors[(usize)found->second].get_contents();
        return argument;
    }
    if (parameter_name.rfind("sparse_delta_", 0) == 0) {
        String peredge_name = parameter_name.substr(String("sparse_delta_").size());
        auto found = matrix_index_by_peredge_name.find(peredge_name);
        if (found == matrix_index_by_peredge_name.end()) {
            log::throw_runtime_error(*logger, "SpikeEngine (nml): synapse dispatch: peredge '" + peredge_name +
                "' has no registered WeightMatrix matrix_index");
        }
        argument.kind = NmlResolvedSynapseArgument::Kind::FixedPointer;
        argument.fixed_pointer = weights.sparse_delta_buffers[(usize)found->second].get_contents();
        return argument;
    }

    for (const auto &directive : synapse_program.alloc) {
        if (const auto *param_constant = std::get_if<nml::ParamConstantDirective>(&directive)) {
            if (param_constant->name != parameter_name) continue;
            log::throw_runtime_error(*logger, "SpikeEngine (nml): synapse dispatch: parameter '" + parameter_name +
                "' is an un-baked (bare) `param` -- allocate_model has no established value source for this "
                "case (matches resolve_nml_cell_tick_argument's own established scope boundary)");
        }
        if (const auto *require = std::get_if<nml::RequireDirective>(&directive)) {
            if (require->name != parameter_name) continue;
            for (const auto &postsynaptic_directive : postsynaptic_cell_program.alloc) {
                const auto *state = std::get_if<nml::StateDirective>(&postsynaptic_directive);
                if (state == nullptr || state->name != parameter_name) continue;
                s32 offset_within_type = nml_state_variable_offset(postsynaptic_cell_program, parameter_name);
                argument.kind = NmlResolvedSynapseArgument::Kind::FixedPointer;
                argument.fixed_pointer = nml_allocation_.cell_state.get_contents() + postsynaptic_chunk_base_offset +
                    (s64)offset_within_type * postsynaptic_population_size - postsynaptic_neuron_index_begin;
                return argument;
            }
            log::throw_runtime_error(*logger, "SpikeEngine (nml): synapse dispatch: require '" + parameter_name +
                "' does not match any StateDirective on the postsynaptic cell type '" +
                postsynaptic_cell_program.component_type_name + "' -- a derived-only (Expose-without-State) "
                "postsynaptic quantity is not supported by this dispatch (see master_kernel.h)");
        }
    }

    log::throw_runtime_error(*logger, "SpikeEngine (nml): synapse dispatch: parameter '" + parameter_name +
        "' is not a recognized reserved name, .alloc name, or edge-dispatch parameter for this dispatch "
        "(see master_kernel.h)");
}

void SpikeEngine::ensure_nml_synapse_dispatch_topology_built() {
    if (nml_synapse_dispatch_topology_built_) return;

    s64 max_neighbor_count = weights.max_neighbor_count;
    Vector<s32> neighbor_scratch((usize)(max_neighbor_count > 0 ? max_neighbor_count : 0));

    const s64 *chunk_base_offsets = nml_allocation_.cell_type_boundaries.get_contents();

    for (usize projection_index = 0; projection_index < nml_projections_.size(); ++projection_index) {
        const nml::ProjectionEntry &projection = nml_projections_[projection_index];
        ProjectionEdgeTopology &topology = nml_projection_edge_topology_[projection_index];

        s32 synapse_type_index = projection.synapse_type_library_index;
        SynapseTypeRuntimeInfo &synapse_info = nml_synapse_types_by_type_library_index_.at(synapse_type_index);
        const nml::IrProgram &synapse_program = nml_type_library_ir_programs_[(usize)synapse_type_index];

        if (!synapse_info.matrix_indices_registered) {
            // Every peredge variable this synapse type declares gets a real WeightMatrix matrix_index,
            // its coefficient vector all-zeros -- but add_coefficient_vector (like
            // set_coefficient_vector) unconditionally fills every PADDING lane (beyond the logical
            // `rank`, up to rank_float4_stride*4) with 1.0f, a neutral multiplier suited to
            // DEFAULT_MATRIX_INDEX's own "reduces to dot(U,V)" contract, not to a genuinely-all-zero
            // matrix -- U/V's own padding lanes are real, random values, so a 1.0f padding coefficient
            // would reconstruct a real, nonzero, spurious contribution from them. Zero the WHOLE
            // registered coefficient buffer directly afterward, overriding that padding fill, so this
            // matrix's reconstruction is exactly 0 until a real accedge Sk bump moves it (mirrors
            // nml::AssembledModel::ensure_synapse_dispatch_topology_built's own identical dance).
            Vector<f32> zero_coefficients((usize)weights.rank, 0.0f);
            for (const auto &directive : synapse_program.alloc) {
                if (const auto *peredge = std::get_if<nml::PeredgeDirective>(&directive)) {
                    s64 matrix_index = weights.add_coefficient_vector(zero_coefficients);
                    s64 effective_lane_count = weights.rank_float4_stride * 4;
                    memset(weights.coefficient_vectors[(usize)matrix_index].get_contents(), 0,
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

        for (const nml::ConnectionEntry &connection : projection.connections) {
            s64 neighbor_count = weights.k2tree.get_neighbors(connection.source_neuron_index,
                                                                neighbor_scratch.data(), max_neighbor_count);
            s32 forward_slot = -1;
            for (s64 slot = 0; slot < neighbor_count; ++slot) {
                if (neighbor_scratch[(usize)slot] == connection.target_neuron_index) {
                    forward_slot = (s32)slot;
                    break;
                }
            }
            if (forward_slot < 0) {
                log::throw_runtime_error(*logger,
                    "SpikeEngine (nml): ensure_nml_synapse_dispatch_topology_built: connection " +
                    std::to_string(connection.source_neuron_index) + " -> " +
                    std::to_string(connection.target_neuron_index) + " (projection '" + projection.id +
                    "') is not a real edge in this engine's WeightMatrix k^2-tree -- weights must reflect "
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
            // dispatch_nml_synapse_delivery_events.
            topology.delivery_scratch_source_nodes = allocate<s32>(byte_count);
            topology.delivery_scratch_target_nodes = allocate<s32>(byte_count);
            topology.delivery_scratch_edge_slots = allocate<s32>(byte_count);

            // Precompute this projection's own resolved dispatch-argument lists -- the earliest point
            // this CAN happen (needs the edge arrays just allocated above, plus the postsynaptic
            // population's own info/IR, both available now).
            const NmlPopulationInfo &postsynaptic_population =
                nml_populations_.at((usize)projection.postsynaptic_population_index);
            const nml::IrProgram &postsynaptic_cell_program =
                nml_type_library_ir_programs_[(usize)postsynaptic_population.type_library_index];
            s64 postsynaptic_chunk_base_offset = chunk_base_offsets[(usize)projection.postsynaptic_population_index];

            topology.integrate_edges_arguments.reserve(synapse_info.integrate_edges_parameter_names_in_order.size());
            for (const String &parameter_name : synapse_info.integrate_edges_parameter_names_in_order) {
                topology.integrate_edges_arguments.push_back(resolve_nml_synapse_edge_argument(
                    parameter_name, synapse_program, synapse_info.matrix_index_by_peredge_name,
                    postsynaptic_cell_program, postsynaptic_population.population_size,
                    postsynaptic_population.neuron_index_begin, postsynaptic_chunk_base_offset,
                    topology.source_nodes.get_contents(), topology.target_nodes.get_contents(),
                    topology.forward_slots.get_contents()));
            }

            topology.deliver_arguments.reserve(synapse_info.deliver_functions.size());
            for (const DeliverFunctionRuntimeInfo &deliver_function : synapse_info.deliver_functions) {
                Vector<NmlResolvedSynapseArgument> resolved_arguments;
                resolved_arguments.reserve(deliver_function.parameter_names_in_order.size());
                for (const String &parameter_name : deliver_function.parameter_names_in_order) {
                    resolved_arguments.push_back(resolve_nml_synapse_edge_argument(
                        parameter_name, synapse_program, synapse_info.matrix_index_by_peredge_name,
                        postsynaptic_cell_program, postsynaptic_population.population_size,
                        postsynaptic_population.neuron_index_begin, postsynaptic_chunk_base_offset,
                        topology.delivery_scratch_source_nodes.get_contents(),
                        topology.delivery_scratch_target_nodes.get_contents(),
                        topology.delivery_scratch_edge_slots.get_contents()));
                }
                topology.deliver_arguments.push_back(std::move(resolved_arguments));
            }
        }
    }

    nml_synapse_dispatch_topology_built_ = true;
}

void SpikeEngine::dispatch_nml_synapse_integrate_edges(f32 dt, s64 tick) {
    for (usize projection_index = 0; projection_index < nml_projections_.size(); ++projection_index) {
        const nml::ProjectionEntry &projection = nml_projections_[projection_index];
        const ProjectionEdgeTopology &topology = nml_projection_edge_topology_[projection_index];
        if (topology.edge_count == 0) continue;

        const SynapseTypeRuntimeInfo &synapse_info =
            nml_synapse_types_by_type_library_index_.at(projection.synapse_type_library_index);

        NmlDispatchArgumentBuilder builder;
        for (const NmlResolvedSynapseArgument &argument : topology.integrate_edges_arguments) {
            switch (argument.kind) {
                case NmlResolvedSynapseArgument::Kind::Dt:
                    builder.add_f32(dt);
                    break;
                case NmlResolvedSynapseArgument::Kind::Tick:
                    builder.add_s64(tick);
                    break;
                case NmlResolvedSynapseArgument::Kind::EventCount:
                    builder.add_s64(topology.edge_count);
                    break;
                case NmlResolvedSynapseArgument::Kind::RankFloat4Stride:
                    builder.add_s64(weights.rank_float4_stride);
                    break;
                case NmlResolvedSynapseArgument::Kind::MaxNeighborCount:
                    builder.add_s64(weights.max_neighbor_count);
                    break;
                case NmlResolvedSynapseArgument::Kind::FixedPointer:
                    builder.add_pointer(argument.fixed_pointer);
                    break;
            }
        }
        builder.dispatch(synapse_info.integrate_edges_handle, nml_launch_config_for(topology.edge_count));
    }
}

void SpikeEngine::dispatch_nml_synapse_delivery_events(f32 dt, s64 tick) {
    // "did this neuron fire THIS tick, on any tracked port" -- a union across every distinct
    // EventPort name (mirrors the fixed scalar propagate stage's own per-port dispatch loop). Read
    // here, NOT cleared -- the fixed scalar propagate dispatch still does that, once, for its own
    // last_spiked/active-set bookkeeping.
    Vector<bool> fired_this_tick((usize)neuron_count, false);
    for (const auto &port_entry : nml_emit_port_flags_) {
        const bool *port_flags = port_entry.second.get_contents();
        for (s64 neuron_index = 0; neuron_index < neuron_count; ++neuron_index) {
            if (port_flags[neuron_index]) fired_this_tick[(usize)neuron_index] = true;
        }
    }

    for (usize projection_index = 0; projection_index < nml_projections_.size(); ++projection_index) {
        const nml::ProjectionEntry &projection = nml_projections_[projection_index];
        ProjectionEdgeTopology &topology = nml_projection_edge_topology_[projection_index];
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

        const SynapseTypeRuntimeInfo &synapse_info =
            nml_synapse_types_by_type_library_index_.at(projection.synapse_type_library_index);

        for (usize deliver_index = 0; deliver_index < synapse_info.deliver_functions.size(); ++deliver_index) {
            const DeliverFunctionRuntimeInfo &deliver_function = synapse_info.deliver_functions[deliver_index];
            const Vector<NmlResolvedSynapseArgument> &resolved_arguments = topology.deliver_arguments[deliver_index];

            NmlDispatchArgumentBuilder builder;
            for (const NmlResolvedSynapseArgument &argument : resolved_arguments) {
                switch (argument.kind) {
                    case NmlResolvedSynapseArgument::Kind::Dt:
                        builder.add_f32(dt);
                        break;
                    case NmlResolvedSynapseArgument::Kind::Tick:
                        builder.add_s64(tick);
                        break;
                    case NmlResolvedSynapseArgument::Kind::EventCount:
                        builder.add_s64(delivered_event_count);
                        break;
                    case NmlResolvedSynapseArgument::Kind::RankFloat4Stride:
                        builder.add_s64(weights.rank_float4_stride);
                        break;
                    case NmlResolvedSynapseArgument::Kind::MaxNeighborCount:
                        builder.add_s64(weights.max_neighbor_count);
                        break;
                    case NmlResolvedSynapseArgument::Kind::FixedPointer:
                        builder.add_pointer(argument.fixed_pointer);
                        break;
                }
            }
            builder.dispatch(deliver_function.handle, nml_launch_config_for(delivered_event_count));
        }
    }
}

// ── ticket #132: real STDP support, ported from nml::AssembledModel::apply_stdp_plasticity
// (master_kernel.cpp) -- reuses SpikeEngine's own existing `learning_rate` field instead of a second
// nml_-prefixed one (see engine.h's own doc comment on apply_nml_stdp_plasticity). ──────────────────

void SpikeEngine::apply_nml_stdp_plasticity(s64 tick) {
    if (learning_rate == 0.0f) return;

    s64 max_neighbor_count = weights.max_neighbor_count;
    if (max_neighbor_count <= 0) return;
    Vector<s32> neighbor_scratch((usize)max_neighbor_count);

    for (s64 source_node = 0; source_node < neuron_count; ++source_node) {
        if (last_spiked.get_contents()[source_node] != tick) continue; // did not fire this tick

        s64 neighbor_count = weights.get_neighbors(source_node, neighbor_scratch.data());
        for (s64 slot = 0; slot < neighbor_count; ++slot) {
            s32 child = neighbor_scratch[(usize)slot];
            s64 child_last_spiked = last_spiked.get_contents()[child];
            // "never fired" is < 0 here (this engine's own -1 seed convention for NML mode) -- see
            // engine.h's own doc comment on this method for why this differs from the ported
            // original's `== 0` check.
            if (child_last_spiked < 0 || child_last_spiked == tick) continue;

            s64 signed_tick_delta = tick - child_last_spiked;
            f32 tick_delta = (f32)(signed_tick_delta < 0 ? -signed_tick_delta : signed_tick_delta);
            f32 decay_delta = -learning_rate * std::pow(tick_delta, -3.0f);
            weights.update((s32)source_node, child, decay_delta, /*learning_rate=*/0.5f,
                           /*l2_regularization=*/1.0f, /*iterations=*/1);
        }
    }
}

// ── move constructor / move-assignment ────────────────────────────────────────
//
// Hand-written (not `= default`) for the same reason WeightMatrix::operator=(WeightMatrix&&)
// is hand-written (weight_matrix.cpp): every GpuPointer/Vector/UnorderedMap member here already
// has move semantics safe enough for a defaulted SpikeEngine move, EXCEPT nml_drain_kernel_/
// nml_propagate_kernel_ (KernelHandle -- backend.h -- has no move semantics of its own, so a
// defaulted move would bitwise-copy them into the destination without clearing the source) and
// cell_state_logs (a raw owned f32** with the same no-move-semantics problem). Left alone, the
// moved-from source's alive/nml_mode_enabled_ flags would also stay true (whatever they were on
// the source), so its destructor would call shutdown() again and re-release/re-delete resources
// the destination now owns -- a double-release/double-free.
SpikeEngine::SpikeEngine(SpikeEngine &&other) noexcept
    : logger(std::move(other.logger))
    , weights(std::move(other.weights))
    , nml_allocation_(std::move(other.nml_allocation_))
    , network_inputs(std::move(other.network_inputs))
    , membrane_potentials(std::move(other.membrane_potentials))
    , last_spiked(std::move(other.last_spiked))
    , last_tick_updated(std::move(other.last_tick_updated))
    , active_neuron_indices(std::move(other.active_neuron_indices))
    , next_active_neuron_indices(std::move(other.next_active_neuron_indices))
    , active_neuron_count(std::move(other.active_neuron_count))
    , next_active_neuron_count(std::move(other.next_active_neuron_count))
    , active_generation(std::move(other.active_generation))
    , input_neuron_indices(std::move(other.input_neuron_indices))
    , input_staging(std::move(other.input_staging))
    , override_staging(std::move(other.override_staging))
    , cell_state_logs(other.cell_state_logs)
    , lifetime(other.lifetime)
    , neuron_count(other.neuron_count)
    , input_neuron_count(other.input_neuron_count)
    , thread_count_per_block(other.thread_count_per_block)
    , block_count(other.block_count)
    , resting_membrane_potential(other.resting_membrane_potential)
    , decay_rate(other.decay_rate)
    , learning_rate(other.learning_rate)
    , spike_period(other.spike_period)
    , spike_threshold(other.spike_threshold)
    , use_constant_weight(other.use_constant_weight)
    , alive(other.alive)
    , active_set_optimization_enabled(other.active_set_optimization_enabled)
    , nml_mode_enabled_(other.nml_mode_enabled_)
    , nml_ring_slot_count_(other.nml_ring_slot_count_)
    , nml_population_dispatch_plans_(std::move(other.nml_population_dispatch_plans_))
    , nml_emit_port_flags_(std::move(other.nml_emit_port_flags_))
    , nml_emit_port_names_(std::move(other.nml_emit_port_names_))
    , nml_drain_kernel_(other.nml_drain_kernel_)
    , nml_propagate_kernel_(other.nml_propagate_kernel_)
    , nml_rng_state_(std::move(other.nml_rng_state_))
    , nml_projections_(std::move(other.nml_projections_))
    , nml_populations_(std::move(other.nml_populations_))
    , nml_type_library_ir_programs_(std::move(other.nml_type_library_ir_programs_))
    , nml_synapse_types_by_type_library_index_(std::move(other.nml_synapse_types_by_type_library_index_))
    , nml_projection_edge_topology_(std::move(other.nml_projection_edge_topology_))
    , nml_synapse_dispatch_topology_built_(other.nml_synapse_dispatch_topology_built_)
{
    // The initializer list above only copied cell_state_logs/nml_drain_kernel_/
    // nml_propagate_kernel_'s VALUE into this object (none of the three have real move semantics) --
    // reset the source to an inert value so its destructor's shutdown() call, if it ran, could never
    // double-free/double-release the same resource. Belt and suspenders: alive/nml_mode_enabled_
    // are cleared on the source immediately below too, so shutdown() never actually runs on it at all.
    other.cell_state_logs = nullptr;
    other.nml_drain_kernel_ = KernelHandle{};
    other.nml_propagate_kernel_ = KernelHandle{};

    // Prevents the moved-from source's destructor from calling shutdown() at all (matches
    // shutdown()'s own `if (!alive) return;` guard at the top) -- it no longer owns anything to
    // release.
    other.alive = false;
    other.nml_mode_enabled_ = false;
}

SpikeEngine &SpikeEngine::operator=(SpikeEngine &&other) noexcept {
    if (this == &other) {
        return *this;
    }

    // Release this (destination) engine's own already-live resources first, mirroring
    // WeightMatrix::operator=(WeightMatrix&&)'s "deallocate the destination's own buffers before
    // absorbing the incoming state" pattern -- shutdown() is exactly the code path that already does
    // so for everything it owns directly (network_inputs/... buffers, NML kernels/dispatch plans).
    if (alive) {
        shutdown();
    }

    logger = std::move(other.logger);
    weights = std::move(other.weights); // WeightMatrix's own hand-written operator=(&&) already
                                         // deallocates weights' own existing buffers first.

    // ModelAllocation's own move-assignment operator is defaulted, which would move-assign its
    // direct GpuPointer members (cell_state/cell_type_boundaries/regime_indices) via
    // GpuPointer::operator=(&&) -- which asserts the destination pointer is null, always false for
    // this engine's own already-live nml_allocation_. This is the exact same hazard
    // WeightMatrix::operator=(&&) already works around for its own k2tree sub-object (see
    // weight_matrix.cpp): ModelAllocation's move CONSTRUCTOR has no such issue (also defaulted, but
    // GpuPointer's move constructor never asserts), so destroy this engine's own nml_allocation_
    // (safely deallocates whatever it currently owns, via ~ModelAllocation()) and move-construct the
    // incoming one in its place instead of going through operator=.
    nml_allocation_.~ModelAllocation();
    new (&nml_allocation_) nml::ModelAllocation(std::move(other.nml_allocation_));

    network_inputs = std::move(other.network_inputs);
    membrane_potentials = std::move(other.membrane_potentials);
    last_spiked = std::move(other.last_spiked);
    last_tick_updated = std::move(other.last_tick_updated);
    active_neuron_indices = std::move(other.active_neuron_indices);
    next_active_neuron_indices = std::move(other.next_active_neuron_indices);
    active_neuron_count = std::move(other.active_neuron_count);
    next_active_neuron_count = std::move(other.next_active_neuron_count);
    active_generation = std::move(other.active_generation);
    input_neuron_indices = std::move(other.input_neuron_indices);
    input_staging = std::move(other.input_staging);
    override_staging = std::move(other.override_staging);

    cell_state_logs = other.cell_state_logs;
    other.cell_state_logs = nullptr;

    lifetime = other.lifetime;
    neuron_count = other.neuron_count;
    input_neuron_count = other.input_neuron_count;
    thread_count_per_block = other.thread_count_per_block;
    block_count = other.block_count;
    resting_membrane_potential = other.resting_membrane_potential;
    decay_rate = other.decay_rate;
    learning_rate = other.learning_rate;
    spike_period = other.spike_period;
    spike_threshold = other.spike_threshold;
    use_constant_weight = other.use_constant_weight;
    alive = other.alive;
    active_set_optimization_enabled = other.active_set_optimization_enabled;

    nml_mode_enabled_ = other.nml_mode_enabled_;
    nml_ring_slot_count_ = other.nml_ring_slot_count_;
    nml_population_dispatch_plans_ = std::move(other.nml_population_dispatch_plans_);
    nml_emit_port_flags_ = std::move(other.nml_emit_port_flags_);
    nml_emit_port_names_ = std::move(other.nml_emit_port_names_);

    // Same "no real move semantics of their own" hazard as in the move constructor above.
    nml_drain_kernel_ = other.nml_drain_kernel_;
    nml_propagate_kernel_ = other.nml_propagate_kernel_;
    other.nml_drain_kernel_ = KernelHandle{};
    other.nml_propagate_kernel_ = KernelHandle{};

    nml_rng_state_ = std::move(other.nml_rng_state_);
    nml_projections_ = std::move(other.nml_projections_);
    nml_populations_ = std::move(other.nml_populations_);
    nml_type_library_ir_programs_ = std::move(other.nml_type_library_ir_programs_);
    nml_synapse_types_by_type_library_index_ = std::move(other.nml_synapse_types_by_type_library_index_);
    nml_projection_edge_topology_ = std::move(other.nml_projection_edge_topology_);
    nml_synapse_dispatch_topology_built_ = other.nml_synapse_dispatch_topology_built_;

    // Prevents the moved-from source's destructor from calling shutdown() at all -- see the move
    // constructor's own comment above for the full reasoning.
    other.alive = false;
    other.nml_mode_enabled_ = false;

    return *this;
}

SpikeEngine::~SpikeEngine() {
    // Guarded on alive (matching shutdown()'s own `if (!alive) return;` idiom), not just to skip
    // shutdown() itself: a moved-from engine's logger is null too (shared_ptr correctly empties its
    // source on move, same as every GpuPointer member), so logging unconditionally here would
    // dereference a null logger on a moved-from object's destruction.
    if (alive) {
        logger->info("Spike Engine shutting down.");
        shutdown();
    }
}

bool SpikeEngine::plasticity_enabled() {
    return learning_rate > 0.0f;
}

void SpikeEngine::enable_plasticity(f32 _learning_rate) {
    if (plasticity_enabled()) return;

    // ── ticket #132: coordinating with ticket #131's real per-edge synapse dispatch -- STDP's rank-1
    // nudge of the shared U/V basis is not yet compensated against a peredge synapse's own Ck
    // reconstruction sharing that basis, so combining the two is not supported (mirrors
    // nml::AssembledModel::enable_plasticity's own identical guard, master_kernel.cpp). A no-op for
    // both hardcoded-LIF constructors above, which never set nml_mode_enabled_/nml_projections_.
    if (nml_mode_enabled_ && !nml_projections_.empty()) {
        log::throw_runtime_error(*logger,
            "enable_plasticity: this SpikeEngine has real per-edge synapse dispatch active "
            "(ticket #131 projections) -- STDP's rank-1 nudge of the shared U/V basis is not yet "
            "compensated against a peredge synapse's own Ck reconstruction sharing that basis, so "
            "combining the two is not supported (see master_kernel.h's own 'ticket #132' doc "
            "comment)");
    }

    learning_rate = _learning_rate;
}

void SpikeEngine::disable_plasticity() {
    if (!plasticity_enabled()) return;

    learning_rate = 0.0f;
}

void SpikeEngine::setup_lifetime(int lifetime_, bool allocate_logs, s64 max_log_bytes) {
    logger->debug("setup_lifetime: lifetime={} allocate_logs={} max_log_bytes={}",
                  lifetime_, allocate_logs, max_log_bytes);
    lifetime = lifetime_;
    if (lifetime < 0 || !allocate_logs) {
        logger->info("Not allocating logs for run data.");
        return;
    }

    s32 size_of_f32 = 4;
    s64 required_bytes = neuron_count * lifetime * size_of_f32;
    if (max_log_bytes < required_bytes) {
        throw_runtime_error(*logger,
            fmt::format("setup_lifetime: refusing to allocate membrane potential log "
                        "({} neurons x {} ticks = {} bytes exceeds {}-byte budget; "
                        "pass a larger max_log_bytes to enable recording)",
                        neuron_count, lifetime, required_bytes, max_log_bytes));
    }

    cell_state_logs = new f32*[neuron_count];
    for (s64 i = 0; i < neuron_count; ++i)
        cell_state_logs[i] = new f32[lifetime];

    logger->debug("setup_lifetime: allocated cell_state_logs for {} neurons x {} ticks", neuron_count, lifetime);
}

void SpikeEngine::set_input_neurons(const vector<s32> &input_neuron_list) {
    logger->debug("set_input_neurons: input_neuron_count={}", input_neuron_list.size());
    if (input_neuron_list.empty()) return;

    s32 s32_byte_size = 4;
    if (input_neuron_indices.pointer != nullptr) deallocate(std::move(input_neuron_indices));
    input_neuron_indices = allocate<s32>(neuron_count * s32_byte_size);
    memcpy(
        input_neuron_indices.get_contents(),
        input_neuron_list.data(),
        input_neuron_list.size() * s32_byte_size
    );
    input_neuron_count = (s64) input_neuron_list.size();

    prefetch_to_gpu(input_neuron_indices, (usize)neuron_count * s32_byte_size);
}

void SpikeEngine::reset_state(s64 last_spiked_value, s32 active_gen_value) {
    logger->debug("reset_state: last_spiked_value={} active_gen_value={}", last_spiked_value, active_gen_value);
    s32 f32_byte_size = 4;
    usize neuron_s64_byte_size = (usize) neuron_count * sizeof(s64);

    memset(network_inputs.get_contents(), 0, (usize) neuron_count * f32_byte_size);
    std::fill(membrane_potentials.get_contents(),
            membrane_potentials.get_contents() + neuron_count,
            resting_membrane_potential);
    std::fill(last_spiked.get_contents(), last_spiked.get_contents() + neuron_count, last_spiked_value);
    memset(last_tick_updated.get_contents(), 0, neuron_s64_byte_size);
    active_neuron_count.get_contents()[0] = 0;
    next_active_neuron_count.get_contents()[0] = 0;
    std::fill(active_generation.get_contents(), active_generation.get_contents() + neuron_count, active_gen_value);
}

void SpikeEngine::step_simulation(
    const vector<f32> &input_values,
    s64 tick,
    const vector<s64> &override_input_neurons,
    bool decay_all_neurons
) {
    if (input_values.empty()) {
        log::throw_runtime_error(*logger, fmt::format("step_simulation: input_values is empty (tick={})", tick));
    }

    logger->trace("step_simulation: tick={} input_values.size={} override_input_neurons.size={} decay_all_neurons={}",
                  tick, input_values.size(), override_input_neurons.size(), decay_all_neurons);

    MetalCommandBatch *batch = begin_command_batch();

    if (decay_all_neurons) {
        gpu_decay_all_neurons(
            membrane_potentials.get_contents(),
            last_tick_updated.get_contents(),
            neuron_count,
            tick,
            resting_membrane_potential,
            decay_rate,
            batch);
    }

    memcpy(input_staging.get_contents(), input_values.data(), input_values.size() * sizeof(f32));

    // QUESTION: anyway to combine these two steps? what exactly is the merge_input_neurons for?
    gpu_add_network_input(
        membrane_potentials.get_contents(),
        input_neuron_indices.get_contents(),
        input_staging.get_contents(),
        (s64)input_values.size(),
        batch);

    next_active_neuron_count.get_contents()[0] = 0;

    if (!override_input_neurons.empty()) {
        memcpy(override_staging.get_contents(), override_input_neurons.data(), override_input_neurons.size() * sizeof(s64));

        gpu_merge_input_neurons(
            active_neuron_indices.get_contents(),
            active_neuron_count.get_contents(),
            override_staging.get_contents(),
            (s64)override_input_neurons.size(),
            batch);
    }

    gpu_step(
        tick,
        tick + 1,
        spike_period,
        spike_threshold,
        learning_rate,
        decay_rate,
        resting_membrane_potential,
        weights.U_matrix.get_contents(),
        weights.V_matrix.get_contents(),
        weights.rank_float4_stride,
        use_constant_weight ? weights.constant_weight : 0.0,
        weights.k2tree.internal_node_words.get_contents(),
        weights.k2tree.leaf_node_words.get_contents(),
        weights.k2tree.rank_superblock_table.get_contents(),
        weights.k2tree.rank_subblock_table.get_contents(),
        weights.k2tree.branching_factor,
        weights.k2tree.superblock_size_words,
        weights.k2tree.padded_node_count,
        weights.k2tree.tree_height,
        weights.k2tree.internal_bit_count,
        neuron_count,
        network_inputs.get_contents(),
        membrane_potentials.get_contents(),
        last_spiked.get_contents(),
        last_tick_updated.get_contents(),
        active_neuron_indices.get_contents(),
        active_neuron_count.get_contents(),
        next_active_neuron_indices.get_contents(),
        next_active_neuron_count.get_contents(),
        active_generation.get_contents(),
        active_set_optimization_enabled,
        thread_count_per_block,
        block_count,
        batch);

    commit_command_batch(batch);

    std::swap(active_neuron_indices, next_active_neuron_indices);
    std::swap(active_neuron_count, next_active_neuron_count);

    logger->trace("step_simulation: tick={} completed, active_neuron_count={}",
                  tick, active_neuron_count.get_contents()[0]);
}

void SpikeEngine::start_static_record(
    const vector<vector<f32>> &input_spikes,
    s64 lifetime,
    const string &filename,
    bool record_membrane,
    s64 record_stride,
    optional<string> compression,
    optional<int> compression_level,
    bool full_decay,
    bool compression_async,
    usize compression_queue_max,
    usize compression_chunk_bytes
) {
    if (lifetime < 0) {
        log::throw_runtime_error(*logger, fmt::format("start_static_record: lifetime must be >= 0 (got {})", lifetime));
    }
    if (record_stride < 1) {
        log::throw_runtime_error(*logger,
            fmt::format("start_static_record: record_stride must be >= 1 (got {})", record_stride));
    }
    if (input_neuron_count <= 0) {
        log::throw_runtime_error(*logger,
            "start_static_record: no input neurons configured (call set_input_neurons first)");
    }
    if ((s64)input_spikes.size() < lifetime) {
        log::throw_runtime_error(*logger,
            fmt::format("start_static_record: input_spikes has {} ticks but lifetime requires {}",
                        input_spikes.size(), lifetime));
    }

    // Each tick's input row is positionally matched to input_neuron_indices, so
    // it must contain exactly input_neuron_count values.
    for (s64 tick = 0; tick < lifetime; ++tick) {
        if ((s64)input_spikes[(usize)tick].size() != input_neuron_count) {
            log::throw_runtime_error(*logger,
                fmt::format("start_static_record: input_spikes[{}] has {} values but there are {} input neurons",
                            tick, input_spikes[(usize)tick].size(), input_neuron_count));
        }
    }

    logger->info("start_static_record: lifetime={} filename={}", lifetime, filename);

    SimulationRecorder recorder(
        filename, neuron_count, compression, compression_level,
        compression_async, compression_queue_max, compression_chunk_bytes);

    // Forces input neurons into the active set every tick regardless of
    // whether they're already active
    vector<s64> override_input_neurons((usize)input_neuron_count);
    const s32 *input_indices = input_neuron_indices.get_contents();
    for (s64 i = 0; i < input_neuron_count; ++i)
        override_input_neurons[(usize)i] = (s64)input_indices[i];

    for (s64 tick = 0; tick < lifetime; ++tick) {
        step_simulation(input_spikes[(usize)tick], tick, override_input_neurons, /*decay_all_neurons=*/false);

        if (record_membrane && tick % record_stride == 0) {
            // _decay_all(tick) runs after step() and only on recorded ticks,
            // immediately before the membrane-potential snapshot
            if (full_decay) {
                gpu_decay_all_neurons(
                    membrane_potentials.get_contents(),
                    last_tick_updated.get_contents(),
                    neuron_count,
                    tick,
                    resting_membrane_potential,
                    decay_rate);
            }

            synchronize_gpu_work();
            prefetch_to_cpu(membrane_potentials, (usize)neuron_count * sizeof(f32));
            recorder.record_frame(membrane_potentials.get_contents(), neuron_count);
        }
    }

    recorder.finish();
    logger->info("start_static_record: finished");
}

pair<f32, f32> SpikeEngine::estimate_bifurcation_weight(s32 input_period) const {
    f32 decay_factor = std::pow(1.0f - decay_rate, (f32)input_period);
    f32 w_accum = (spike_threshold - resting_membrane_potential) * (1.0f - decay_factor);
    f32 w_instant = spike_threshold - resting_membrane_potential;
    return make_pair(w_accum, w_instant);
}

void SpikeEngine::scale_uniform_weights_near_bifurcation(
    f32 *target, f32 *w_accum, f32 *w_instant,
    s32 input_period, f32 scale, bool freeze_learning, const bool *use_constant_weight_
) {
    auto [w_accum_, w_instant_] = estimate_bifurcation_weight(input_period);
    *w_accum = w_accum_;
    *w_instant = w_instant_;
    *target = *w_accum * scale;
    weights.set_constant_weight(*target);

    if (freeze_learning) {
        learning_rate = 0.0f;
    }

    use_constant_weight = use_constant_weight_ != nullptr
        ? *use_constant_weight_
        : freeze_learning;

    logger->debug("scale_uniform_weights_near_bifurcation: input_period={} scale={} target={} "
                  "w_accum={} w_instant={} use_constant_weight={}",
                  input_period, scale, *target, *w_accum, *w_instant, use_constant_weight);
}

ScaledReservoirResult SpikeEngine::scale_randomized_weights_near_bifurcation(s32 input_period, f32 scale, bool freeze_learning) {
    auto [w_accum, w_instant] = estimate_bifurcation_weight(input_period);
    f32 target = abs(w_accum * scale);
    ScaleResult result = weights.scale_neighbor_weights_to_root_mean_square(target);

    use_constant_weight = false;

    if (freeze_learning) {
        learning_rate = 0.0f;
    }

    logger->debug("scale_randomized_weights_near_bifurcation: input_period={} scale={} target={} "
                  "w_accum={} w_instant={}",
                  input_period, scale, target, w_accum, w_instant);

    return ScaledReservoirResult{result,w_accum,w_instant};
}

void SpikeEngine::get_reservoir_features_vector(s64 tick, f32 spike_tau, f32 voltage_scale, GpuPointer<f32> output_buffer) {
    logger->trace("get_reservoir_features_vector: tick={} spike_tau={} voltage_scale={}", tick, spike_tau, voltage_scale);
    if (spike_tau <= 0.0f) {
        logger->warn("get_reservoir_features_vector: spike_tau was <= 0.0. Aborting.");
        return;
    }
    if (voltage_scale <= 0.0f) {
        logger->warn("get_reservoir_features_vector: voltage_scale was <= 0.0. Aborting.");
        return;
    }
    gpu_reservoir_features(
        neuron_count,
        tick,
        spike_tau,
        voltage_scale,
        membrane_potentials.get_contents(),
        last_spiked.get_contents(),
        last_tick_updated.get_contents(),
        resting_membrane_potential,
        decay_rate,
        output_buffer.get_contents());
}

void SpikeEngine::shutdown() {
    if (!alive) return;

    logger->info("shutdown: releasing GPU buffers");

    if (cell_state_logs != nullptr) {
        for (s64 i = 0; i < neuron_count; ++i)
            delete[] cell_state_logs[i];
        delete[] cell_state_logs;
        cell_state_logs = nullptr;
    }

    deallocate(std::move(network_inputs));
    deallocate(std::move(membrane_potentials));
    deallocate(std::move(last_spiked));
    deallocate(std::move(last_tick_updated));
    deallocate(std::move(active_neuron_indices));
    deallocate(std::move(next_active_neuron_indices));
    deallocate(std::move(active_neuron_count));
    deallocate(std::move(next_active_neuron_count));
    deallocate(std::move(active_generation));
    deallocate(std::move(input_neuron_indices));
    deallocate(std::move(input_staging));
    deallocate(std::move(override_staging));

    // ── NML-mode resources (Stage 1 of folding nml::AssembledModel into SpikeEngine) -- guarded on
    // nml_mode_enabled_ since the two hardcoded-LIF constructors above never compile any of these
    // (calling release_kernel on a never-compiled, default-constructed KernelHandle is unsafe on
    // CUDA -- cuModuleUnload on a null CUmodule -- so this must not run unconditionally). ──
    if (nml_mode_enabled_) {
        for (auto &plan : nml_population_dispatch_plans_) release_kernel(plan.kernel_handle);
        release_kernel(nml_drain_kernel_);
        release_kernel(nml_propagate_kernel_);
        for (auto &entry : nml_emit_port_flags_) deallocate(std::move(entry.second));
        deallocate(std::move(nml_rng_state_));

        // ── Stage 2 (ticket #131) -- release every synapse-type kernel handle + deallocate every
        // ProjectionEdgeTopology's GpuPointers, mirroring nml::AssembledModel::~AssembledModel's own
        // cleanup exactly (master_kernel.cpp). ──
        for (auto &type_entry : nml_synapse_types_by_type_library_index_) {
            release_kernel(type_entry.second.integrate_edges_handle);
            for (auto &deliver_function : type_entry.second.deliver_functions) release_kernel(deliver_function.handle);
        }
        for (auto &topology : nml_projection_edge_topology_) {
            deallocate(std::move(topology.source_nodes));
            deallocate(std::move(topology.target_nodes));
            deallocate(std::move(topology.forward_slots));
            deallocate(std::move(topology.delivery_scratch_source_nodes));
            deallocate(std::move(topology.delivery_scratch_target_nodes));
            deallocate(std::move(topology.delivery_scratch_edge_slots));
        }
    }

    alive = false;
}
























