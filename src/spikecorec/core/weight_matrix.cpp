//
// Created by Alek Simpson on 5/30/26.
//

#include <algorithm>
#include <atomic>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "weight_matrix.h"
#include "backend.h"
#include "log.h"


namespace spikecorec {

using namespace std;
using namespace spikecorec;

// Tags the save format (edge count, basis plus S); a file with another tag is rejected.
constexpr u32 WEIGHT_MATRIX_SAVE_MAGIC = 0x574D5432;

// Runs body(index) for every index in [0, count) on all cores; each body writes only its own index.
template <typename Body>
void for_each_index_in_parallel(s64 count, const Body &body) {
    constexpr s64 INDICES_PER_CLAIM = 8;
    const s64 worker_count = min<s64>(max<s64>(static_cast<s64>(Thread::hardware_concurrency()), 1),
                                      (count + INDICES_PER_CLAIM - 1) / INDICES_PER_CLAIM);
    if (worker_count <= 1) {
        for (s64 index = 0; index < count; index += 1) body(index);
        return;
    }

    Atomic<s64> next_index{0};
    auto work = [&]() {
        for (s64 first = next_index.fetch_add(INDICES_PER_CLAIM); first < count;
             first = next_index.fetch_add(INDICES_PER_CLAIM)) {
            const s64 last = min(first + INDICES_PER_CLAIM, count);
            for (s64 index = first; index < last; index += 1) body(index);
        }
    };
    Vector<Thread> workers;
    for (s64 worker = 1; worker < worker_count; worker += 1) workers.emplace_back(work);
    work();
    for (Thread &worker : workers) worker.join();
}

s64 round_up_to_lane_group(usize value) {
    const s64 group = Weight_Matrix::LANE_GROUP;
    return ((max<s64>((s64)value, 1) + group - 1) / group) * group;
}

// Over the finite values only, so one NaN or inf cannot set the scale of every other value.
// 0 for an empty, all-zero or all non-finite sample.
template <typename Value>
f64 root_mean_square(const Vector<Value> &values) {
    f64 sum_of_squares = 0.0;
    s64 finite_count = 0;
    for (const Value value : values) {
        if (!isfinite(static_cast<f64>(value))) continue;
        sum_of_squares += static_cast<f64>(value) * static_cast<f64>(value);
        finite_count += 1;
    }
    return finite_count == 0 ? 0.0 : sqrt(sum_of_squares / static_cast<f64>(finite_count));
}

// As edge_weight_of and held_target in weight_fit_kernels.metal: an edge's weight, 1 / max(|target|,
// 1)^2 with the target in units of the plane's RMS, and the target the lanes are fitted to. A
// non-finite target gets no weight and counts as zero.
f32 edge_weight_of(f32 target) {
    if (!isfinite(target)) return 0.0f;
    const f32 error_scale = max(fabsf(target), 1.0f);
    return 1.0f / (error_scale * error_scale);
}

f32 held_target(f32 target) {
    return isfinite(target) ? target : 0.0f;
}

bool can_safely_cast_s64_to_s32(s64 value) {
    return value >= static_cast<s64>(Numeric_Limits<s32>::min()) && value <= static_cast<s64>(Numeric_Limits<s32>::max());
}

const Vector<Vector<s32>> &Weight_Matrix::validate_network(const Vector<Vector<s32>> &network) {
    if (network.empty()) {
        log::throw_invalid_argument(log::logger(),
            "Weight_Matrix: network must have at least one neuron (got 0)");
    }
    const s64 node_count = static_cast<s64>(network.size());
    for (s64 source = 0; source < node_count; source += 1) {
        for (s32 target : network[static_cast<usize>(source)]) {
            if (target >= 0 && target < node_count) continue;
            log::throw_invalid_argument(log::logger(),
                "Weight_Matrix: node " + to_string(source) + " connects to " + to_string(target) +
                ", outside the network's " + to_string(node_count) + " nodes");
        }
    }
    return network;
}

// The network's k^2-tree; throws when it cannot be built (the factory's warning says why).
static K2_Tree build_network_tree(Engine_Backend &backend, const Vector<Vector<s32>> &network) {
    Optional<K2_Tree> tree = K2_Tree::from_adjacency_list(backend, network, static_cast<s64>(network.size()));
    if (!tree.has_value()) {
        log::throw_runtime_error(log::logger(),
            "Weight_Matrix: the network's k^2-tree cannot be built (see the warning above); a network of " +
            to_string(network.size()) + " nodes");
    }
    return std::move(*tree);
}

Weight_Matrix::Weight_Matrix(
    Engine_Backend &backend,
    const Vector<Vector<s32>> &network,
    s64 rank,
    bool check_indexing,
    s64 max_neighbor_count,
    s64 weight_seed,
    s64 matrix_count,
    s64 updated_plane_count
)
    : k2tree(build_network_tree(backend, validate_network(network)))
    , matrix_count(matrix_count)
    , updated_plane_count(updated_plane_count)
    , owning_backend(&backend)
    , node_count(static_cast<s64>(network.size()))
    , check_indexing(check_indexing)
{
    if (matrix_count < FIRST_STATE_VARIABLE_PLANE) {
        log::throw_invalid_argument(log::logger(),
            "Weight_Matrix: matrix_count " + to_string(matrix_count) + " leaves no room for the weight "
            "and delay planes");
    }
    if (updated_plane_count < 0 || updated_plane_count > matrix_count) {
        log::throw_invalid_argument(log::logger(),
            "Weight_Matrix: updated_plane_count " + to_string(updated_plane_count) + " is not between 0 and the " +
            to_string(matrix_count) + " planes");
    }

    // A bound below the longest row would cut that row's edges off every walk.
    s64 longest_row = 0;
    for (const Vector<s32> &row : network) longest_row = max(longest_row, static_cast<s64>(row.size()));
    if (max_neighbor_count >= 0 && max_neighbor_count < longest_row) {
        log::throw_invalid_argument(log::logger(),
            "Weight_Matrix: max_neighbor_count " + to_string(max_neighbor_count) + " is below the longest row's " +
            to_string(longest_row) + " targets");
    }
    this->max_neighbor_count = max_neighbor_count < 0 ? longest_row : max_neighbor_count;

    Vector<s64> in_degree(node_count, 0);
    for (const Vector<s32> &row : network) {
        for (s32 target : row) in_degree[static_cast<usize>(target)] += 1;
    }
    max_predecessor_count = in_degree.empty() ? 0 : *max_element(in_degree.begin(), in_degree.end());

    this->rank = (rank > 0) ? round_up_to_lane_group(rank) : LANE_GROUP;
    rank_float4_stride = this->rank / LANE_GROUP;

    if (rank_float4_stride > MAX_RANK_FLOAT4_STRIDE) {
        log::throw_invalid_argument(log::logger(),
            "Weight_Matrix: rank " + to_string(this->rank) + " exceeds the kernel limit of " +
            to_string(static_cast<s64>(MAX_RANK_FLOAT4_STRIDE) * LANE_GROUP) + " lanes");
    }

    build_edge_row_offset();
    // S and the queue hold edge ordinals as u32 and planes as u8.
    if (total_edge_count > static_cast<s64>(Numeric_Limits<u32>::max())) {
        log::throw_invalid_argument(log::logger(),
            "Weight_Matrix: " + to_string(total_edge_count) + " edges is more than the 2^32 S can index");
    }
    if (matrix_count > static_cast<s64>(Numeric_Limits<u8>::max())) {
        log::throw_invalid_argument(log::logger(),
            "Weight_Matrix: " + to_string(matrix_count) + " planes is more than the 255 the update queue can name");
    }
    allocate_basis_storage();
    allocate_delta_storage();
    pending_delta_capacity = pending_capacity_for_threshold(DEFAULT_REFIT_OCCUPANCY_THRESHOLD_FRACTION);
    allocate_pending_storage();

    basis_seed = (weight_seed >= 0) ? static_cast<unsigned>(weight_seed) : Random_Device{}();
    seed_basis(basis_seed);

    log::logger().debug("Weight_Matrix constructed: node_count={} edges={} rank={} max_neighbor_count={}",
                        node_count, total_edge_count, this->rank, this->max_neighbor_count);
}

void Weight_Matrix::build_edge_row_offset() {
    edge_row_offset_host.assign(static_cast<usize>(node_count) + 1, 0);

    Vector<u32> neighbor_buffer((max<s64>(max_neighbor_count, 1)));
    s64 running_ordinal = 0;
    for (s64 node_index = 0; node_index < node_count; node_index += 1) {
        edge_row_offset_host[static_cast<usize>(node_index)] = running_ordinal;
        running_ordinal += k2tree.get_neighbors(static_cast<u32>(node_index), neighbor_buffer.data(),
                                                max_neighbor_count);
    }
    edge_row_offset_host[static_cast<usize>(node_count)] = running_ordinal;
    total_edge_count = running_ordinal;
}

void Weight_Matrix::allocate_basis_storage() {
    const u64 matrix_byte_size = static_cast<u64>(node_count) * static_cast<u64>(rank_float4_stride) * sizeof(float4);
    const u64 lane_count = static_cast<u64>(rank_float4_stride) * static_cast<u64>(LANE_GROUP);

    Vector<Engine_Pointer> partitions;
    owning_backend
        ->partition(matrix_byte_size, Engine_Datatype::FLOAT32X4, partitions)                 // U
        .partition(matrix_byte_size, Engine_Datatype::FLOAT32X4, partitions)                  // V
        .partition(static_cast<u64>(matrix_count) * lane_count * sizeof(f32), Engine_Datatype::FLOAT32, partitions)
        .partition(static_cast<u64>(node_count + 1) * sizeof(s64), Engine_Datatype::SIGNED64, partitions);
    owning_slab = owning_backend->allocate(partitions);

    U_matrix = partitions[0];
    V_matrix = partitions[1];
    coefficients = partitions[2];
    edge_row_offset = partitions[3];

    memcpy(edge_row_offset.get_contents(), edge_row_offset_host.data(),
           (static_cast<usize>(node_count) + 1) * sizeof(s64));

    owning_backend->advise_read_mostly(edge_row_offset, static_cast<u64>(node_count + 1) * sizeof(s64));
    owning_backend->prefetch_to_gpu(U_matrix, matrix_byte_size);
    owning_backend->prefetch_to_gpu(V_matrix, matrix_byte_size);
}

void Weight_Matrix::allocate_delta_storage() {
    // Pool positions are s32 in the CSR row starts.
    if (sparse_delta_capacity > static_cast<s64>(Numeric_Limits<s32>::max())) {
        log::throw_runtime_error(log::logger(),
            "Weight_Matrix: S would need " + to_string(sparse_delta_capacity) + " entries, more than the 2^31 it "
            "can index; refit more often");
    }
    Vector<Engine_Pointer> partitions;
    owning_backend
        ->partition(static_cast<u64>(matrix_count) * static_cast<u64>(node_count + 1) * sizeof(s32),
                    Engine_Datatype::SIGNED32, partitions)
        .partition(static_cast<u64>(sparse_delta_capacity) * sizeof(u32), Engine_Datatype::UNSIGNED32, partitions)
        .partition(static_cast<u64>(sparse_delta_capacity) * sizeof(f32), Engine_Datatype::FLOAT32, partitions);
    delta_slab = owning_backend->allocate(partitions);

    sparse_delta_row_start = partitions[0];
    sparse_delta_edge_ordinal = partitions[1];
    sparse_delta_value = partitions[2];

    if (!sparse_delta_row_start.is_empty()) {
        memset(sparse_delta_row_start.get_contents(), 0,
               static_cast<usize>(matrix_count) * (static_cast<usize>(node_count) + 1) * sizeof(s32));
    }
    sparse_delta_entry_count.assign(matrix_count, 0);
}

void Weight_Matrix::allocate_pending_storage() {
    Vector<Engine_Pointer> partitions;
    owning_backend
        ->partition(static_cast<u64>(pending_delta_capacity) * sizeof(u32), Engine_Datatype::UNSIGNED32, partitions)
        .partition(static_cast<u64>(pending_delta_capacity) * sizeof(f32), Engine_Datatype::FLOAT32, partitions)
        .partition(static_cast<u64>(pending_delta_capacity) * sizeof(u8), Engine_Datatype::UNSIGNED8, partitions)
        .partition(static_cast<u64>(DEFERRED_LIST_SLOT + 2 * node_count) * sizeof(s32), Engine_Datatype::SIGNED32,
                   partitions);
    pending_slab = owning_backend->allocate(partitions);

    pending_delta_edge_ordinal = partitions[0];
    pending_delta_value = partitions[1];
    pending_delta_matrix_index = partitions[2];
    pending_delta_control = partitions[3];
    memset(pending_delta_control.get_contents(), 0, static_cast<usize>(DEFERRED_LIST_SLOT) * sizeof(s32));
}

s64 Weight_Matrix::pending_delta_count() const {
    if (pending_delta_control.is_empty()) return 0;
    return pending_delta_control.get_contents_as<s32>()[PENDING_WRITE_COUNT_SLOT];
}

void Weight_Matrix::begin_first_walk_pass() {
    s32 *control = pending_delta_control.get_contents_as<s32>();
    control[PENDING_RESERVED_SLOT] = 0;
    control[WALK_PASS_SLOT] = 0;
    control[WALKED_NEURON_COUNT_SLOT] = 0;
    control[DEFERRED_NEURON_COUNT_SLOT] = 0;
}

s64 Weight_Matrix::begin_next_walk_pass() {
    s32 *control = pending_delta_control.get_contents_as<s32>();
    const s32 deferred = control[DEFERRED_NEURON_COUNT_SLOT];
    control[PENDING_RESERVED_SLOT] = 0;
    control[DEFERRED_NEURON_COUNT_SLOT] = 0;
    if (deferred == 0) {
        control[WALK_PASS_SLOT] = 0;
        control[WALKED_NEURON_COUNT_SLOT] = 0;
        return 0;
    }
    control[WALK_PASS_SLOT] += 1;
    control[WALKED_NEURON_COUNT_SLOT] = deferred;
    return deferred;
}

void Weight_Matrix::resize_basis(s64 new_rank) {
    const s64 rounded_rank = round_up_to_lane_group(new_rank);
    if (rounded_rank == rank) return;
    require_storage("resize_basis");

    if (rounded_rank / LANE_GROUP > MAX_RANK_FLOAT4_STRIDE) {
        log::throw_invalid_argument(log::logger(),
            "Weight_Matrix: rank " + to_string(rounded_rank) + " exceeds the kernel limit of " +
            to_string(static_cast<s64>(MAX_RANK_FLOAT4_STRIDE) * LANE_GROUP) + " lanes");
    }

    owning_backend->deallocate_slab(owning_slab);
    rank = rounded_rank;
    rank_float4_stride = rank / LANE_GROUP;
    allocate_basis_storage();

    seed_basis(basis_seed);
}

void Weight_Matrix::seed_basis(unsigned seed) const {
    Mersenne_Twister random_engine(seed);
    Normal_Distribution<f32> normal_distribution_unit(0.0f, 1.0f);

    auto *u_data = U_matrix.get_contents_as<float4>();
    auto *v_data = V_matrix.get_contents_as<float4>();
    const s64 total_float4_element_count = node_count * rank_float4_stride;
    for (s64 element_index = 0; element_index < total_float4_element_count; ++element_index) {
        u_data[element_index] = {normal_distribution_unit(random_engine),
                                 normal_distribution_unit(random_engine),
                                 normal_distribution_unit(random_engine),
                                 normal_distribution_unit(random_engine)};
        v_data[element_index] = {normal_distribution_unit(random_engine),
                                 normal_distribution_unit(random_engine),
                                 normal_distribution_unit(random_engine),
                                 normal_distribution_unit(random_engine)};
    }

    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    f32 *coefficient_data = coefficients.get_contents_as<f32>();
    for (s64 lane_index = 0; lane_index < matrix_count * lane_count; ++lane_index) {
        coefficient_data[lane_index] = 1.0f;
    }
}

Weight_Matrix::Weight_Matrix(Weight_Matrix &&other) noexcept
    : k2tree(std::move(other.k2tree))
    , matrix_count(other.matrix_count)
    , U_matrix(other.U_matrix)
    , V_matrix(other.V_matrix)
    , coefficients(other.coefficients)
    , edge_row_offset(other.edge_row_offset)
    , sparse_delta_row_start(other.sparse_delta_row_start)
    , sparse_delta_edge_ordinal(other.sparse_delta_edge_ordinal)
    , sparse_delta_value(other.sparse_delta_value)
    , pending_delta_edge_ordinal(other.pending_delta_edge_ordinal)
    , pending_delta_value(other.pending_delta_value)
    , pending_delta_matrix_index(other.pending_delta_matrix_index)
    , pending_delta_control(other.pending_delta_control)
    , updated_plane_count(other.updated_plane_count)
    , pending_delta_capacity(other.pending_delta_capacity)
    , sparse_delta_capacity(other.sparse_delta_capacity)
    , sparse_delta_entry_count(std::move(other.sparse_delta_entry_count))
    , plasticity_reserve_entries(other.plasticity_reserve_entries)
    , fit_tolerance(other.fit_tolerance)
    , fit_method(other.fit_method)
    , expected_active_edge_fraction(other.expected_active_edge_fraction)
    , run_first_edge_ordinal(std::move(other.run_first_edge_ordinal))
    , run_edge_count(std::move(other.run_edge_count))
    , run_synapse_prototype(std::move(other.run_synapse_prototype))
    , owning_backend(other.owning_backend)
    , owning_slab(other.owning_slab)
    , delta_slab(other.delta_slab)
    , pending_slab(other.pending_slab)
    , node_count(other.node_count)
    , total_edge_count(other.total_edge_count)
    , max_neighbor_count(other.max_neighbor_count)
    , max_predecessor_count(other.max_predecessor_count)
    , rank(other.rank)
    , rank_float4_stride(other.rank_float4_stride)
    , check_indexing(other.check_indexing)
    , measured_fit_error(std::move(other.measured_fit_error))
    , edge_row_offset_host(std::move(other.edge_row_offset_host))
    , basis_seed(other.basis_seed)
{
    other.owning_backend = nullptr;
    other.owning_slab = Engine_Pointer{};
    other.delta_slab = Engine_Pointer{};
    other.pending_slab = Engine_Pointer{};
    other.node_count = 0;
    other.total_edge_count = 0;
}

Weight_Matrix &Weight_Matrix::operator=(Weight_Matrix &&other) noexcept {
    if (this == &other) return *this;

    if (owning_backend != nullptr) {
        owning_backend->deallocate_slab(owning_slab);
        owning_backend->deallocate_slab(delta_slab);
        owning_backend->deallocate_slab(pending_slab);
    }

    k2tree = std::move(other.k2tree);
    matrix_count = other.matrix_count;
    U_matrix = other.U_matrix;
    V_matrix = other.V_matrix;
    coefficients = other.coefficients;
    edge_row_offset = other.edge_row_offset;
    sparse_delta_row_start = other.sparse_delta_row_start;
    sparse_delta_edge_ordinal = other.sparse_delta_edge_ordinal;
    sparse_delta_value = other.sparse_delta_value;
    pending_delta_edge_ordinal = other.pending_delta_edge_ordinal;
    pending_delta_value = other.pending_delta_value;
    pending_delta_matrix_index = other.pending_delta_matrix_index;
    pending_delta_control = other.pending_delta_control;
    updated_plane_count = other.updated_plane_count;
    pending_delta_capacity = other.pending_delta_capacity;
    sparse_delta_entry_count = std::move(other.sparse_delta_entry_count);
    sparse_delta_capacity = other.sparse_delta_capacity;
    plasticity_reserve_entries = other.plasticity_reserve_entries;
    fit_tolerance = other.fit_tolerance;
    fit_method = other.fit_method;
    expected_active_edge_fraction = other.expected_active_edge_fraction;
    run_first_edge_ordinal = std::move(other.run_first_edge_ordinal);
    run_edge_count = std::move(other.run_edge_count);
    run_synapse_prototype = std::move(other.run_synapse_prototype);
    owning_backend = other.owning_backend;
    owning_slab = other.owning_slab;
    delta_slab = other.delta_slab;
    pending_slab = other.pending_slab;
    node_count = other.node_count;
    total_edge_count = other.total_edge_count;
    max_neighbor_count = other.max_neighbor_count;
    max_predecessor_count = other.max_predecessor_count;
    rank = other.rank;
    rank_float4_stride = other.rank_float4_stride;
    check_indexing = other.check_indexing;
    measured_fit_error = std::move(other.measured_fit_error);
    edge_row_offset_host = std::move(other.edge_row_offset_host);
    basis_seed = other.basis_seed;

    other.owning_backend = nullptr;
    other.owning_slab = Engine_Pointer{};
    other.delta_slab = Engine_Pointer{};
    other.pending_slab = Engine_Pointer{};
    other.node_count = 0;
    other.total_edge_count = 0;
    return *this;
}

Weight_Matrix::~Weight_Matrix() {
    if (owning_backend == nullptr) return;
    owning_backend->deallocate_slab(owning_slab);
    owning_backend->deallocate_slab(delta_slab);
    owning_backend->deallocate_slab(pending_slab);
}

bool Weight_Matrix::check_index_inbounds(s32 node_index) const {
    return check_indexing && node_index >= 0 && node_index < node_count;
}

bool Weight_Matrix::check_index_inbounds(s32 source, s32 target) const {
    return check_index_inbounds(source) && check_index_inbounds(target);
}

void Weight_Matrix::validate_matrix_index(s64 matrix_index) const {
    if (matrix_index < 0 || matrix_index >= matrix_count) {
        log::throw_invalid_argument(log::logger(),
            "Weight_Matrix: matrix_index " + to_string(matrix_index) + " out of range [0, " +
            to_string(matrix_count) + ")");
    }
}

s64 Weight_Matrix::get_neighbors(s64 node_index, u32 *output_buffer) const {
    if (!can_safely_cast_s64_to_s32(node_index)) return 0;
    if (node_index < 0 || node_index >= node_count) return 0;
    return k2tree.get_neighbors(static_cast<u32>(node_index), output_buffer, max_neighbor_count);
}

s64 Weight_Matrix::get_predecessors(s64 node_index, u32 *output_buffer) const {
    if (!can_safely_cast_s64_to_s32(node_index)) return 0;
    if (node_index < 0 || node_index >= node_count) return 0;
    return k2tree.get_predecessors(static_cast<u32>(node_index), output_buffer, max_predecessor_count);
}

Optional<s64> Weight_Matrix::edge_ordinal(s32 source_node, s32 target_node) const {
    if (source_node < 0 || source_node >= node_count) return nullopt;

    Vector<u32> neighbor_buffer((max<s64>(max_neighbor_count, 1)));
    const s64 degree = k2tree.get_neighbors(static_cast<u32>(source_node), neighbor_buffer.data(), max_neighbor_count);
    for (s64 slot = 0; slot < degree; slot += 1) {
        if (neighbor_buffer[static_cast<usize>(slot)] != static_cast<u32>(target_node)) continue;

        return edge_row_offset_host[static_cast<usize>(source_node)] + slot;
    }
    return nullopt;
}

f32 *Weight_Matrix::coefficient_row(s64 matrix_index) const {
    return coefficients.get_contents_as<f32>() + matrix_index * rank_float4_stride * LANE_GROUP;
}

Engine_Pointer Weight_Matrix::coefficient_range(s64 matrix_index) const {
    const u64 row_bytes = static_cast<u64>(rank_float4_stride) * static_cast<u64>(LANE_GROUP) * sizeof(f32);
    Engine_Pointer range = coefficients;
    range.offset += static_cast<s64>(static_cast<u64>(matrix_index) * row_bytes);
    range.total_bytes = row_bytes;
    return range;
}

bool Weight_Matrix::ensure_function(Engine_Function &function, const String &name) const {
#ifdef SPIKECOREC_METAL
    if (function.pipeline_state != nullptr) return true;
#else
    if (function.cuda_function != nullptr) return true;
#endif
    if (owning_backend == nullptr || precompiled_functions_unavailable) return false;

    Optional<Engine_Function> loaded = owning_backend->load_precompiled_function(name);
    if (!loaded.has_value()) {
        precompiled_functions_unavailable = true;
        return false;
    }

    function = *loaded;
    return true;
}

f32 Weight_Matrix::reconstruct_entry(
    s32 source_node, s32 target_node, const f32 *coefficient_values
) const {
    const float4 *u_row = U_matrix.get_contents_as<float4>() + source_node * rank_float4_stride;
    const float4 *v_row = V_matrix.get_contents_as<float4>() + target_node * rank_float4_stride;

    f32 dot_product = 0.0f;
    for (s64 float4_index = 0; float4_index < rank_float4_stride; ++float4_index) {
        const f32 *lane_coefficients = coefficient_values + float4_index * LANE_GROUP;
        dot_product += u_row[float4_index].x * (lane_coefficients[0] * v_row[float4_index].x)
                     + u_row[float4_index].y * (lane_coefficients[1] * v_row[float4_index].y)
                     + u_row[float4_index].z * (lane_coefficients[2] * v_row[float4_index].z)
                     + u_row[float4_index].w * (lane_coefficients[3] * v_row[float4_index].w);
    }
    return dot_product;
}

f32 Weight_Matrix::sparse_delta_for(s64 matrix_index, s32 source_node, s64 edge_ordinal) const {
    if (sparse_delta_entry_count[static_cast<usize>(matrix_index)] == 0) return 0.0f;
    if (sparse_delta_row_start.is_empty()) return 0.0f;

    const s32 *row_start = sparse_delta_row_start.get_contents_as<s32>() +
                           matrix_index * (node_count + 1);
    const u32 *entry_ordinal = sparse_delta_edge_ordinal.get_contents_as<u32>();
    const f32 *entry_value = sparse_delta_value.get_contents_as<f32>();

    s32 low = row_start[source_node];
    s32 high = row_start[source_node + 1];
    while (low < high) {
        const s32 middle = low + (high - low) / 2;

        if (static_cast<s64>(entry_ordinal[middle]) < edge_ordinal) {
            low = middle + 1;
            continue;
        }

        high = middle;
    }
    if (low < row_start[source_node + 1] && static_cast<s64>(entry_ordinal[low]) == edge_ordinal) {
        return entry_value[low];
    }
    return 0.0f;
}

f32 Weight_Matrix::get_for_matrix(s32 source_node, s32 target_node, s64 matrix_index) const {
    validate_matrix_index(matrix_index);
    if (!check_index_inbounds(source_node, target_node)) return 0.0f;

    // The basis is unconstrained off the edges, so a pair that is not an edge holds no value.
    const Optional<s64> ordinal = edge_ordinal(source_node, target_node);
    if (!ordinal.has_value()) return 0.0f;

    return reconstruct_entry(source_node, target_node, coefficient_row(matrix_index)) +
           sparse_delta_for(matrix_index, source_node, *ordinal);
}

f32 Weight_Matrix::get(s32 source_node, s32 target_node) const {
    return get_for_matrix(source_node, target_node, WEIGHT_PLANE);
}

s32 Weight_Matrix::get_edge_delay_ticks(s32 source_node, s32 target_node) const {
    if (!edge_ordinal(source_node, target_node).has_value()) return 0;
    return max<s32>(static_cast<s32>(lroundf(get_for_matrix(source_node, target_node, DELAY_PLANE))), 1);
}

s32 Weight_Matrix::get_edge_synapse_prototype(s32 source_node, s32 target_node) const {
    const Optional<s64> ordinal = edge_ordinal(source_node, target_node);
    if (!ordinal.has_value()) return -1;

    const auto entry = upper_bound(run_first_edge_ordinal.begin(),
                                   run_first_edge_ordinal.end(), *ordinal);
    if (entry == run_first_edge_ordinal.begin()) return -1;

    const usize run_index = static_cast<usize>(entry - run_first_edge_ordinal.begin() - 1);
    if (*ordinal >= run_first_edge_ordinal[run_index] + run_edge_count[run_index]) {
        return -1;
    }
    return run_synapse_prototype[run_index];
}

s64 Weight_Matrix::delta_capacity_for_threshold(f32 occupancy_threshold_fraction) const {
    // A refit empties S once a plane reaches the threshold; without one, S holds every edge. Only
    // the planes the kernel updates gain entries during a run.
    const s64 update_entries = occupancy_threshold_fraction > 0.0f
            ? static_cast<s64>(ceil(static_cast<f64>(occupancy_threshold_fraction) * static_cast<f64>(total_edge_count)))
            : total_edge_count;
    return updated_plane_count * min(update_entries, total_edge_count) + plasticity_reserve_entries;
}

void Weight_Matrix::require_storage(const char *operation) const {
    if (owning_backend != nullptr) return;
    log::throw_runtime_error(log::logger(),
        String("Weight_Matrix::") + operation + ": this matrix has no storage (it is empty, moved from, or its "
        "engine has been shut down)");
}

void Weight_Matrix::resize_delta_capacity(s64 new_capacity) {
    if (new_capacity == sparse_delta_capacity) return;
    require_storage("resize_delta_capacity");

    s64 held = 0;
    for (s64 entry_count : sparse_delta_entry_count) held += entry_count;
    if (held > new_capacity) {
        log::throw_runtime_error(log::logger(),
            "Weight_Matrix::resize_delta_capacity: S holds " + to_string(held) + " updates, more than the " +
            to_string(new_capacity) + " asked for");
    }
    move_delta_storage(new_capacity);
}

void Weight_Matrix::move_delta_storage(s64 new_capacity) {
    const Engine_Pointer old_slab = delta_slab;
    const Engine_Pointer old_row_start = sparse_delta_row_start;
    const Engine_Pointer old_edge_ordinal = sparse_delta_edge_ordinal;
    const Engine_Pointer old_value = sparse_delta_value;
    const Vector<s64> entry_count = sparse_delta_entry_count;
    s64 held = 0;
    for (s64 count : entry_count) held += count;

    sparse_delta_capacity = new_capacity;
    allocate_delta_storage();
    if (!old_row_start.is_empty() && !sparse_delta_row_start.is_empty()) {
        memcpy(sparse_delta_row_start.get_contents(), old_row_start.get_contents(),
               static_cast<usize>(matrix_count) * (static_cast<usize>(node_count) + 1) * sizeof(s32));
    }
    if (held > 0) {
        memcpy(sparse_delta_edge_ordinal.get_contents(), old_edge_ordinal.get_contents(), static_cast<usize>(held) * sizeof(u32));
        memcpy(sparse_delta_value.get_contents(), old_value.get_contents(), static_cast<usize>(held) * sizeof(f32));
    }
    sparse_delta_entry_count = entry_count;
    owning_backend->deallocate_slab(old_slab);
}

s64 Weight_Matrix::pending_capacity_for_threshold(f32 occupancy_threshold_fraction) const {
    return max(delta_capacity_for_threshold(occupancy_threshold_fraction), updated_plane_count * max_neighbor_count);
}

void Weight_Matrix::resize_pending_delta_capacity(s64 new_capacity) {
    require_storage("resize_pending_delta_capacity");
    owning_backend->deallocate_slab(pending_slab);
    pending_delta_capacity = max<s64>(new_capacity, 0);
    allocate_pending_storage();
}

s64 Weight_Matrix::grown_delta_capacity(s64 required_entries) const {
    // Doubling keeps a growing S from reallocating every tick. Each plane holds at most one update
    // per edge, and during a run only the planes the kernel updates gain any, so doubling stops at
    // their edges (a host update to another plane still gets the room it needs).
    const s64 updated_planes_limit = max<s64>(updated_plane_count, 1) * total_edge_count + plasticity_reserve_entries;
    return min(matrix_count * total_edge_count + plasticity_reserve_entries,
               max(required_entries, min(2 * max<s64>(sparse_delta_capacity, 1), updated_planes_limit)));
}

void Weight_Matrix::rebuild_sparse_deltas(Vector<Vector<Pair<s64, f32>>> &deltas) {
    s64 held = 0;
    for (const Vector<Pair<s64, f32>> &plane : deltas) held += static_cast<s64>(plane.size());
    if (held > sparse_delta_capacity) {
        // Grows S's own slab; the lists being rebuilt are the whole of S.
        owning_backend->deallocate_slab(delta_slab);
        sparse_delta_capacity = grown_delta_capacity(held);
        allocate_delta_storage();
        log::logger().debug("rebuild_sparse_deltas: S grew to {} updates", sparse_delta_capacity);
    }

    s32 *row_start_base = sparse_delta_row_start.get_contents_as<s32>();
    u32 *entry_ordinal = sparse_delta_edge_ordinal.get_contents_as<u32>();
    f32 *entry_value = sparse_delta_value.get_contents_as<f32>();
    s64 position = 0;
    for (s64 matrix_index = 0; matrix_index < matrix_count; ++matrix_index) {
        Vector<Pair<s64, f32>> &plane = deltas[static_cast<usize>(matrix_index)];
        sort(plane.begin(), plane.end(),
             [](const Pair<s64, f32> &left, const Pair<s64, f32> &right) { return left.first < right.first; });

        const s64 plane_start = position;
        for (const Pair<s64, f32> &delta : plane) {
            entry_ordinal[position] = static_cast<u32>(delta.first);
            entry_value[position] = delta.second;
            position += 1;
        }

        if (row_start_base == nullptr) continue;
        s32 *row_start = row_start_base + matrix_index * (node_count + 1);
        s64 delta_index = 0;
        for (s64 node_index = 0; node_index <= node_count; ++node_index) {
            const s64 row_first_ordinal = edge_row_offset_host[static_cast<usize>(node_index)];
            while (delta_index < static_cast<s64>(plane.size()) &&
                   plane[static_cast<usize>(delta_index)].first < row_first_ordinal) {
                delta_index += 1;
            }
            row_start[node_index] = static_cast<s32>(plane_start + delta_index);
        }
        sparse_delta_entry_count[static_cast<usize>(matrix_index)] = static_cast<s64>(plane.size());
    }
}

Vector<Vector<Pair<s64, f32>>> Weight_Matrix::current_deltas() const {
    Vector<Vector<Pair<s64, f32>>> deltas(matrix_count);
    if (sparse_delta_row_start.is_empty()) return deltas;

    const s32 *row_start_base = sparse_delta_row_start.get_contents_as<s32>();
    const u32 *entry_ordinal = sparse_delta_edge_ordinal.get_contents_as<u32>();
    const f32 *entry_value = sparse_delta_value.get_contents_as<f32>();
    for (s64 matrix_index = 0; matrix_index < matrix_count; ++matrix_index) {
        const s64 entry_count = sparse_delta_entry_count[static_cast<usize>(matrix_index)];
        if (entry_count == 0) continue;

        const s64 plane_start = row_start_base[matrix_index * (node_count + 1)];
        Vector<Pair<s64, f32>> &plane = deltas[static_cast<usize>(matrix_index)];
        plane.reserve(static_cast<usize>(entry_count));
        for (s64 index = plane_start; index < plane_start + entry_count; ++index) {
            plane.emplace_back(static_cast<s64>(entry_ordinal[index]), entry_value[index]);
        }
    }
    return deltas;
}

void Weight_Matrix::clear_sparse_deltas() {
    if (!sparse_delta_row_start.is_empty()) {
        memset(sparse_delta_row_start.get_contents(), 0,
               static_cast<usize>(matrix_count) * (static_cast<usize>(node_count) + 1) * sizeof(s32));
    }
    sparse_delta_entry_count.assign(matrix_count, 0);
}

void Weight_Matrix::accumulate_edge_delta(
    s64 matrix_index, s32 source_node, s32 target_node, f32 delta
) {
    validate_matrix_index(matrix_index);
    if (!isfinite(delta)) {
        log::throw_invalid_argument(log::logger(),
            "Weight_Matrix::accumulate_edge_delta: the update to (" + to_string(source_node) + ", " +
            to_string(target_node) + ") is " + to_string(delta) + ", which no plane can hold");
    }

    const Optional<s64> ordinal = edge_ordinal(source_node, target_node);
    if (!ordinal.has_value()) {
        log::throw_invalid_argument(log::logger(),
            "Weight_Matrix::accumulate_edge_delta: (" + to_string(source_node) + ", " +
            to_string(target_node) + ") is not an edge, so it has no value to update");
    }

    Vector<Vector<Pair<s64, f32>>> deltas = current_deltas();
    Vector<Pair<s64, f32>> &plane = deltas[static_cast<usize>(matrix_index)];
    auto existing = find_if(plane.begin(), plane.end(),
                            [&ordinal](const Pair<s64, f32> &entry) { return entry.first == *ordinal; });
    if (existing != plane.end()) {
        existing->second += delta;
        if (existing->second == 0.0f) plane.erase(existing);
    } else if (delta != 0.0f) {
        plane.emplace_back(*ordinal, delta);
    }
    rebuild_sparse_deltas(deltas);
}

void Weight_Matrix::compact_pending_deltas() {
    if (pending_delta_control.is_empty()) return;

    s32 *pending_count = pending_delta_control.get_contents_as<s32>() + PENDING_WRITE_COUNT_SLOT;
    const s64 staged = min<s64>(*pending_count, pending_delta_capacity);
    if (staged <= 0) {
        *pending_count = 0;
        return;
    }

    // Walks reserve their room, so only an update queued without a reservation can overflow.
    if (static_cast<s64>(*pending_count) > pending_delta_capacity) {
        log::logger().warn("compact_pending_deltas: {} updates were dropped: the queue holds {}",
                           static_cast<s64>(*pending_count) - pending_delta_capacity, pending_delta_capacity);
    }

    const u32 *staged_ordinal = pending_delta_edge_ordinal.get_contents_as<u32>();
    const f32 *staged_value = pending_delta_value.get_contents_as<f32>();
    const u8 *staged_matrix_index = pending_delta_matrix_index.get_contents_as<u8>();

    // The queued updates by plane, sorted by edge, with each edge's updates summed. A sum of zero
    // changes nothing, so it is dropped. Only the queue is copied; S is merged in place.
    Vector<Vector<Pair<u32, f32>>> queued(static_cast<usize>(matrix_count));
    {
        Vector<s64> queued_count(static_cast<usize>(matrix_count), 0);
        for (s64 index = 0; index < staged; ++index) {
            if (staged_matrix_index[index] < matrix_count) queued_count[staged_matrix_index[index]] += 1;
        }
        for (s64 matrix_index = 0; matrix_index < matrix_count; ++matrix_index) {
            queued[static_cast<usize>(matrix_index)].reserve(static_cast<usize>(queued_count[static_cast<usize>(matrix_index)]));
        }
    }
    for (s64 index = 0; index < staged; ++index) {
        const s64 matrix_index = staged_matrix_index[index];
        if (matrix_index >= matrix_count) continue;
        queued[static_cast<usize>(matrix_index)].emplace_back(staged_ordinal[index], staged_value[index]);
    }
    *pending_count = 0;
    for (Vector<Pair<u32, f32>> &plane : queued) {
        sort(plane.begin(), plane.end(),
             [](const Pair<u32, f32> &left, const Pair<u32, f32> &right) { return left.first < right.first; });
        usize kept = 0;
        for (usize index = 0; index < plane.size(); index += 1) {
            if (kept > 0 && plane[kept - 1].first == plane[index].first) {
                plane[kept - 1].second += plane[index].second;
            } else {
                plane[kept] = plane[index];
                kept += 1;
            }
        }
        plane.resize(kept);
        plane.erase(remove_if(plane.begin(), plane.end(), [](const Pair<u32, f32> &update) { return update.second == 0.0f; }),
                    plane.end());
    }

    // An update to an edge already in S is added to its entry, leaving only edges S lacks queued.
    u32 *entry_ordinal = sparse_delta_edge_ordinal.get_contents_as<u32>();
    f32 *entry_value = sparse_delta_value.get_contents_as<f32>();
    {
        s64 start = 0;
        for (s64 matrix_index = 0; matrix_index < matrix_count; ++matrix_index) {
            const s64 entry_count = sparse_delta_entry_count[static_cast<usize>(matrix_index)];
            Vector<Pair<u32, f32>> &updates = queued[static_cast<usize>(matrix_index)];
            usize kept = 0;
            for (const Pair<u32, f32> &update : updates) {
                const u32 *found = lower_bound(entry_ordinal + start, entry_ordinal + start + entry_count, update.first);
                if (found != entry_ordinal + start + entry_count && *found == update.first) {
                    entry_value[found - entry_ordinal] += update.second;
                } else {
                    updates[kept] = update;
                    kept += 1;
                }
            }
            updates.resize(kept);
            start += entry_count;
        }
    }

    // Drop the entries that are now zero, with those the device zeroed in place (it changes an S
    // entry in place even to zero, so that no edge in S ever queues). The planes stay packed from
    // position 0.
    Vector<s64> plane_start(static_cast<usize>(matrix_count), 0);
    {
        s64 read = 0;
        s64 write = 0;
        for (s64 matrix_index = 0; matrix_index < matrix_count; ++matrix_index) {
            s64 &entry_count = sparse_delta_entry_count[static_cast<usize>(matrix_index)];
            plane_start[static_cast<usize>(matrix_index)] = write;
            const s64 plane_end = read + entry_count;
            for (; read < plane_end; read += 1) {
                if (entry_value[read] == 0.0f) continue;
                entry_ordinal[write] = entry_ordinal[read];
                entry_value[write] = entry_value[read];
                write += 1;
            }
            entry_count = write - plane_start[static_cast<usize>(matrix_index)];
        }
    }

    // Each plane gains one entry per queued edge.
    Vector<s64> merged_count(static_cast<usize>(matrix_count), 0);
    s64 merged_total = 0;
    for (s64 matrix_index = 0; matrix_index < matrix_count; ++matrix_index) {
        merged_count[static_cast<usize>(matrix_index)] = sparse_delta_entry_count[static_cast<usize>(matrix_index)] +
                                                         static_cast<s64>(queued[static_cast<usize>(matrix_index)].size());
        merged_total += merged_count[static_cast<usize>(matrix_index)];
    }

    // S grows rather than lose an update; the engine shrinks it after the next refit.
    if (merged_total > sparse_delta_capacity) {
        move_delta_storage(grown_delta_capacity(merged_total));
        entry_ordinal = sparse_delta_edge_ordinal.get_contents_as<u32>();
        entry_value = sparse_delta_value.get_contents_as<f32>();
        log::logger().debug("compact_pending_deltas: S grew to {} updates", sparse_delta_capacity);
    }

    // Planes merge last to first, each from its end. Every step writes an entry, so a plane's write
    // position never falls below its read position; its merged entries start no earlier than its
    // entries do and end where the next plane's merged entries start, so no entry is overwritten
    // before it is read.
    s64 merged_end = merged_total;
    for (s64 matrix_index = matrix_count - 1; matrix_index >= 0; --matrix_index) {
        const Vector<Pair<u32, f32>> &updates = queued[static_cast<usize>(matrix_index)];
        const s64 start = plane_start[static_cast<usize>(matrix_index)];
        const s64 merged_start = merged_end - merged_count[static_cast<usize>(matrix_index)];
        s64 read = start + sparse_delta_entry_count[static_cast<usize>(matrix_index)];
        s64 write = merged_end;
        for (usize update_index = updates.size(); update_index > 0; --update_index) {
            const Pair<u32, f32> &update = updates[update_index - 1];
            for (; read > start && entry_ordinal[read - 1] > update.first; --read) {
                write -= 1;
                entry_ordinal[write] = entry_ordinal[read - 1];
                entry_value[write] = entry_value[read - 1];
            }
            write -= 1;
            entry_ordinal[write] = update.first;
            entry_value[write] = update.second;
        }
        // The entries below the plane's first queued edge move as one block, if at all.
        const s64 remaining = read - start;
        if (merged_start != start) {
            memmove(entry_ordinal + merged_start, entry_ordinal + start, static_cast<usize>(remaining) * sizeof(u32));
            memmove(entry_value + merged_start, entry_value + start, static_cast<usize>(remaining) * sizeof(f32));
        }
        sparse_delta_entry_count[static_cast<usize>(matrix_index)] = merged_count[static_cast<usize>(matrix_index)];
        plane_start[static_cast<usize>(matrix_index)] = merged_start;
        merged_end = merged_start;
    }

    s32 *row_start_base = sparse_delta_row_start.get_contents_as<s32>();
    if (row_start_base != nullptr) {
        for (s64 matrix_index = 0; matrix_index < matrix_count; ++matrix_index) {
            s32 *row_start = row_start_base + matrix_index * (node_count + 1);
            const s64 plane_end = plane_start[static_cast<usize>(matrix_index)] + sparse_delta_entry_count[static_cast<usize>(matrix_index)];
            s64 position = plane_start[static_cast<usize>(matrix_index)];
            for (s64 node_index = 0; node_index <= node_count; ++node_index) {
                const s64 row_first_ordinal = edge_row_offset_host[static_cast<usize>(node_index)];
                while (position < plane_end && static_cast<s64>(entry_ordinal[position]) < row_first_ordinal) position += 1;
                row_start[node_index] = static_cast<s32>(position);
            }
        }
    }
    log::logger().debug("compact_pending_deltas: merged {} updates", staged);
}

Vector<s64> Weight_Matrix::planes_awaiting_lane_budget() const {
    Vector<s64> planes;
    if (total_edge_count == 0) return planes;
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        if (lanes_owned_by(matrix_index).empty()) planes.push_back(matrix_index);
    }
    return planes;
}

Vector<Pair<s64, f32>> Weight_Matrix::plane_updates(s64 matrix_index) const {
    validate_matrix_index(matrix_index);
    Vector<Pair<s64, f32>> updates;
    const s64 entry_count = sparse_delta_entry_count[static_cast<usize>(matrix_index)];
    if (entry_count == 0 || sparse_delta_row_start.is_empty()) return updates;
    const s64 plane_start = sparse_delta_row_start.get_contents_as<s32>()[matrix_index * (node_count + 1)];
    const u32 *entry_ordinal = sparse_delta_edge_ordinal.get_contents_as<u32>();
    const f32 *entry_value = sparse_delta_value.get_contents_as<f32>();
    updates.reserve(static_cast<usize>(entry_count));
    for (s64 index = plane_start; index < plane_start + entry_count; index += 1) {
        updates.emplace_back(static_cast<s64>(entry_ordinal[index]), entry_value[index]);
    }
    return updates;
}

Vector<f32> Weight_Matrix::plane_update_values(s64 matrix_index) const {
    validate_matrix_index(matrix_index);
    Vector<f32> values(static_cast<usize>(max<s64>(total_edge_count, 0)), 0.0f);
    const s64 entry_count = sparse_delta_entry_count[static_cast<usize>(matrix_index)];
    if (entry_count == 0 || sparse_delta_row_start.is_empty()) return values;
    const s64 plane_start = sparse_delta_row_start.get_contents_as<s32>()[matrix_index * (node_count + 1)];
    const u32 *entry_ordinal = sparse_delta_edge_ordinal.get_contents_as<u32>();
    const f32 *entry_value = sparse_delta_value.get_contents_as<f32>();
    for (s64 index = plane_start; index < plane_start + entry_count; index += 1) values[entry_ordinal[index]] = entry_value[index];
    return values;
}

f64 Weight_Matrix::plane_update_sum_of_squares(s64 matrix_index) const {
    validate_matrix_index(matrix_index);
    const s64 entry_count = sparse_delta_entry_count[static_cast<usize>(matrix_index)];
    if (entry_count == 0 || sparse_delta_row_start.is_empty()) return 0.0;
    const s64 plane_start = sparse_delta_row_start.get_contents_as<s32>()[matrix_index * (node_count + 1)];
    const f32 *entry_value = sparse_delta_value.get_contents_as<f32>();
    f64 sum_of_squares = 0.0;
    for (s64 index = plane_start; index < plane_start + entry_count; index += 1) {
        sum_of_squares += static_cast<f64>(entry_value[index]) * static_cast<f64>(entry_value[index]);
    }
    return sum_of_squares;
}

void Weight_Matrix::calibrate_lane_budget(s64 matrix_index, const Vector<f32> &calibration_values) {
    validate_matrix_index(matrix_index);
    if (static_cast<s64>(calibration_values.size()) != total_edge_count) {
        log::throw_invalid_argument(log::logger(),
            "Weight_Matrix::calibrate_lane_budget: " + to_string(calibration_values.size()) + " values for " +
            to_string(total_edge_count) + " edges");
    }
    if (!lanes_owned_by(matrix_index).empty()) return;

    const Fit_Values values = plane_values_from([&](s64 ordinal) { return calibration_values[static_cast<usize>(ordinal)]; });
    const f64 values_root_mean_square = plane_value_statistics(matrix_index, values).root_mean_square;
    if (values_root_mean_square == 0.0) return;

    s64 other_planes_lane_count = 0;
    for (s64 plane = 0; plane < matrix_count; plane += 1) {
        if (plane != matrix_index) other_planes_lane_count += static_cast<s64>(lanes_owned_by(plane).size());
    }
    const s64 room = max<s64>(1, lane_cap() - other_planes_lane_count);

    const Fit_Method refit_method = fit_method;
    fit_method = lane_search_fit_method;
    Plane_Fit_Result searched_fit;
    try {
        searched_fit = search_fewest_lanes(matrix_index, values, DEFAULT_FIT_RIDGE, fit_tolerance, 0, room);
    } catch (...) {
        fit_method = refit_method;
        throw;
    }
    fit_method = refit_method;

    const s64 searched = static_cast<s64>(lanes_owned_by(matrix_index).size());
    const f64 overhead = 1.0 + static_cast<f64>(clamp(expected_active_edge_fraction, 0.0f, 1.0f));
    const s64 budget = min(room, max<s64>(1, static_cast<s64>(ceil(static_cast<f64>(searched) * overhead))));
    if (budget > searched) take_lanes(matrix_index, budget - searched, values_root_mean_square);

    // The plane's values are still all zero: U goes to zero and V keeps the calibrated rows.
    f32 *u_data = U_matrix.get_contents_as<f32>();
    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    for (s64 lane : lanes_owned_by(matrix_index)) {
        for (s64 node_index = 0; node_index < node_count; node_index += 1) u_data[node_index * lane_count + lane] = 0.0f;
    }
    record_fit_error(matrix_index, 0.0);

    log::logger().debug("calibrate_lane_budget: plane {} needs {} lanes for its calibration values{}; budget {} "
                        "(room {} under the cap)", matrix_index, searched,
                        searched_fit.meets_tolerance ? "" : ", missing fit_tolerance", budget, room);
}

void Weight_Matrix::discard_updates(f32 occupancy_threshold_fraction) {
    clear_sparse_deltas();
    if (!pending_delta_control.is_empty()) {
        pending_delta_control.get_contents_as<s32>()[PENDING_WRITE_COUNT_SLOT] = 0;
        begin_first_walk_pass();
    }
    resize_delta_capacity(delta_capacity_for_threshold(occupancy_threshold_fraction));
}

f32 Weight_Matrix::sparse_delta_occupancy_fraction() const {
    if (total_edge_count <= 0) return 0.0f;

    s64 worst = 0;
    for (s64 matrix_index = 0; matrix_index < matrix_count; ++matrix_index) {
        worst = max(worst, sparse_delta_entry_count[static_cast<usize>(matrix_index)]);
    }
    return static_cast<f32>(worst) / static_cast<f32>(total_edge_count);
}

bool Weight_Matrix::is_refit_due(f32 occupancy_threshold_fraction) const {
    if (occupancy_threshold_fraction <= 0.0f) return false;
    return sparse_delta_occupancy_fraction() >= occupancy_threshold_fraction;
}

void Weight_Matrix::declare_runs(
    const Vector<s64> &first_edge_ordinal,
    const Vector<s64> &edge_count,
    const Vector<s32> &synapse_prototype,
    const Vector<Vector<f32>> &initial_values
) {
    const usize run_count = first_edge_ordinal.size();
    bool shaped = edge_count.size() == run_count && synapse_prototype.size() == run_count &&
                  static_cast<s64>(initial_values.size()) == matrix_count;
    for (const Vector<f32> &value_per_run : initial_values) shaped = shaped && value_per_run.size() == run_count;
    if (!shaped) {
        log::throw_invalid_argument(log::logger(),
            "Weight_Matrix::declare_runs: the runs need one edge count and prototype each, and "
            "initial_values one row per plane with one value per run");
    }

    run_first_edge_ordinal = first_edge_ordinal;
    run_edge_count = edge_count;
    run_synapse_prototype = synapse_prototype;

    if (run_count == 0 || total_edge_count == 0) {
        merge_runs_by_prototype();
        log::logger().debug("declare_runs: nothing to declare ({} runs, {} edges)",
                            run_count, total_edge_count);
        return;
    }

    // One plane's values at a time, as a fit reads them: its nonzero values, 8 bytes each.
    const auto plane_values = [&](s64 matrix_index) {
        const Vector<f32> &value_per_run = initial_values[static_cast<usize>(matrix_index)];
        return plane_values_from([&](s64 ordinal) {
            const auto entry = upper_bound(run_first_edge_ordinal.begin(), run_first_edge_ordinal.end(), ordinal);
            return value_per_run[static_cast<usize>(max<s64>(entry - run_first_edge_ordinal.begin() - 1, 0))];
        });
    };

    {
        const Fit_Method refit_method = fit_method;
        fit_method = lane_search_fit_method;
        try {
            fit_every_plane_within_lane_budgets(plane_values, DEFAULT_FIT_RIDGE);
        } catch (...) {
            fit_method = refit_method;
            throw;
        }
        fit_method = refit_method;
    }
    compact_basis();

    // The basis holds the declared values, so S starts empty and only ever holds what changes.
    clear_sparse_deltas();
    resize_delta_capacity(delta_capacity_for_threshold(DEFAULT_REFIT_OCCUPANCY_THRESHOLD_FRACTION));
    merge_runs_by_prototype();

    if (worst_fit_error() > fit_tolerance) {
        log::logger().warn("Weight_Matrix: at rank {} the basis reproduces the declared values to within {:.2f}%, "
                           "short of the {:.2f}% tolerance", rank, 100.0 * worst_fit_error(), 100.0 * fit_tolerance);
    }

    const s64 basis_bytes = (2 * node_count * rank + matrix_count * rank) * static_cast<s64>(sizeof(f32));
    const s64 delta_bytes =
            matrix_count * (node_count + 1) * static_cast<s64>(sizeof(s32)) +
            sparse_delta_capacity * static_cast<s64>(sizeof(u32) + sizeof(f32)) +
            pending_delta_capacity * static_cast<s64>(sizeof(u32) + sizeof(f32) + sizeof(u8));
    log::logger().info("Weight_Matrix: {} runs over {} edges and {} planes at rank {}, worst error {:.2f}%; {} bytes "
                       "of basis + {} bytes of update buffers, against {} bytes for one float per edge per plane",
                       run_count, total_edge_count, matrix_count, rank, 100.0 * worst_fit_error(), basis_bytes,
                       delta_bytes, total_edge_count * matrix_count * static_cast<s64>(sizeof(f32)));
}

void Weight_Matrix::merge_runs_by_prototype() {
    Vector<s64> merged_first_edge_ordinal;
    Vector<s64> merged_edge_count;
    Vector<s32> merged_synapse_prototype;
    for (usize run = 0; run < run_first_edge_ordinal.size(); run += 1) {
        const bool extends_previous = !merged_first_edge_ordinal.empty() &&
                                      merged_synapse_prototype.back() == run_synapse_prototype[run] &&
                                      merged_first_edge_ordinal.back() + merged_edge_count.back() == run_first_edge_ordinal[run];
        if (extends_previous) {
            merged_edge_count.back() += run_edge_count[run];
            continue;
        }
        merged_first_edge_ordinal.push_back(run_first_edge_ordinal[run]);
        merged_edge_count.push_back(run_edge_count[run]);
        merged_synapse_prototype.push_back(run_synapse_prototype[run]);
    }
    run_first_edge_ordinal = std::move(merged_first_edge_ordinal);
    run_edge_count = std::move(merged_edge_count);
    run_synapse_prototype = std::move(merged_synapse_prototype);
}

void Weight_Matrix::grow_basis(s64 new_rank) {
    const s64 grown_rank = round_up_to_lane_group(new_rank);
    if (grown_rank <= rank) return;
    require_storage("grow_basis");
    if (grown_rank / LANE_GROUP > MAX_RANK_FLOAT4_STRIDE) {
        log::throw_invalid_argument(log::logger(),
            "Weight_Matrix: rank " + to_string(grown_rank) + " exceeds the kernel limit of " +
            to_string(static_cast<s64>(MAX_RANK_FLOAT4_STRIDE) * LANE_GROUP) + " lanes");
    }

    const s64 old_lane_count = rank_float4_stride * LANE_GROUP;
    Vector<f32> saved_u(U_matrix.get_contents_as<f32>(),
                        U_matrix.get_contents_as<f32>() + node_count * old_lane_count);
    Vector<f32> saved_v(V_matrix.get_contents_as<f32>(),
                        V_matrix.get_contents_as<f32>() + node_count * old_lane_count);
    Vector<f32> saved_coefficients(coefficients.get_contents_as<f32>(),
                                   coefficients.get_contents_as<f32>() + matrix_count * old_lane_count);

    owning_backend->deallocate_slab(owning_slab);
    rank = grown_rank;
    rank_float4_stride = rank / LANE_GROUP;
    allocate_basis_storage();

    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    Mersenne_Twister random_engine(basis_seed + static_cast<unsigned>(rank));
    Normal_Distribution<f32> normal_distribution_unit(0.0f, 1.0f);
    f32 *u_data = U_matrix.get_contents_as<f32>();
    f32 *v_data = V_matrix.get_contents_as<f32>();
    for (s64 node_index = 0; node_index < node_count; ++node_index) {
        s64 lane, index;

        for (lane = 0; lane < old_lane_count; ++lane) {
            index = node_index * lane_count + lane;
            u_data[index] = saved_u[static_cast<usize>(node_index * old_lane_count + lane)];
            v_data[index] = saved_v[static_cast<usize>(node_index * old_lane_count + lane)];
        }
        for (; lane < lane_count; ++lane) {
            index = node_index * lane_count + lane;
            u_data[index] = normal_distribution_unit(random_engine);
            v_data[index] = normal_distribution_unit(random_engine);
        }
    }
    for (s64 matrix_index = 0; matrix_index < matrix_count; ++matrix_index) {
        f32 *coefficient_values = coefficient_row(matrix_index);
        s64 lane;
        for (lane = 0; lane < old_lane_count; ++lane) {
            coefficient_values[lane] = saved_coefficients[static_cast<usize>(matrix_index * old_lane_count + lane)];
        }
        for (; lane < lane_count; ++lane) {
            coefficient_values[lane] = 0.0f;
        }
    }
}

void Weight_Matrix::refit(f32 ridge_regularization) {
    if (total_edge_count == 0) {
        clear_sparse_deltas();
        return;
    }

    // A plane's fit reads its S entries keyed by target node, and its values from the basis plus
    // them. Refitting a plane changes only lanes no other plane reads, so the planes after it still
    // read the values they held before. A plane with no updates is left as it is.
    const auto started = Steady_Clock::now();
    bool meets_tolerance = true;
    s64 refit_plane_count = 0;
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        if (sparse_delta_entry_count[static_cast<usize>(matrix_index)] == 0) continue;
        key_plane_updates_by_target(matrix_index);
        try {
            const Fit_Values updates = updates_of_plane(matrix_index);
            meets_tolerance = refit_plane(matrix_index, updates, ridge_regularization).meets_tolerance && meets_tolerance;
        } catch (...) {
            key_plane_updates_by_ordinal(matrix_index);
            throw;
        }
        refit_plane_count += 1;
    }
    clear_sparse_deltas();
    if (refit_plane_count == 0) return;

    // A plane that was not refit keeps the error of the fit that made it.
    String error_per_plane;
    for (s64 matrix_index = 0; matrix_index < matrix_count; ++matrix_index) {
        if (matrix_index > 0) error_per_plane += ", ";
        error_per_plane += log::format_floating(".3f", 100.0 * measured_fit_error[static_cast<usize>(matrix_index)]) + "%";
    }
    log::logger().debug("refit: rank {}, worst relative error per plane [{}]; {} planes refit in {:.4f} s", rank,
                        error_per_plane, refit_plane_count, Duration<f64>(Steady_Clock::now() - started).count());
    if (!meets_tolerance) {
        log::logger().warn("refit: at rank {} the basis is {:.2f}% off, short of the {:.2f}% tolerance", rank,
                           100.0 * worst_fit_error(), 100.0 * fit_tolerance);
    }
}

// ── fitting ─────────────────────────────────────────────────────────────────

namespace {

// The most threads a reducing fit kernel runs. Each strides over its share of the indices, so a
// reduction leaves at most this many threads' worth of threadgroup partials for the host to add.
constexpr s64 FIT_REDUCTION_THREAD_LIMIT = 1 << 16;

// Device scratch the alternating least squares row solves use at most, in floats: a quarter of the
// plane values the matrix stands for (one float per edge per plane), and never less than
// FIT_SOLVE_SCRATCH_MINIMUM_FLOAT_COUNT nor more than FIT_SOLVE_SCRATCH_FLOAT_LIMIT. Rows are
// solved in batches whose triangles fit.
constexpr s64 FIT_SOLVE_SCRATCH_FLOAT_LIMIT = 1 << 24;
constexpr s64 FIT_SOLVE_SCRATCH_MINIMUM_FLOAT_COUNT = 1 << 16;

// Threads a walking fit kernel gives each row: one SIMD group (FIT_WALK_THREADS_PER_ROW in
// weight_fit_kernels.metal).
constexpr s64 FIT_WALK_THREADS_PER_ROW = 32;

// The most basis vectors one dot-product pass takes; FIT_MAXIMUM_BASIS_VECTOR_COUNT in
// weight_fit_kernels.metal.
constexpr s64 FIT_MAXIMUM_BASIS_VECTOR_COUNT = 64;
static_assert(2 * Weight_Matrix::LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO_HISTORY_LENGTH + 1 <=
              FIT_MAXIMUM_BASIS_VECTOR_COUNT, "the quasi-Newton history outgrows the dot-product kernel");

// Indices per chunk in the host versions of the reductions.
constexpr s64 HOST_REDUCTION_CHUNK = 4096;

s64 reduction_thread_count(s64 count) {
    return max<s64>(1, min(count, FIT_REDUCTION_THREAD_LIMIT));
}

// Connections an average walker finds: about this many per walker keeps the descent each walk
// starts with small next to the connections it finds.
constexpr s64 FIT_CONNECTIONS_PER_WALKER = 16;

// The walkers a walking kernel splits each row among: about one per FIT_CONNECTIONS_PER_WALKER
// connections of an average row, a power of two from 1 to a SIMD group.
u32 walkers_per_row_for(s64 edge_count, s64 node_count) {
    const s64 wanted = edge_count / max<s64>(node_count * FIT_CONNECTIONS_PER_WALKER, 1);
    u32 walkers = 1;
    while (walkers < FIT_WALK_THREADS_PER_ROW && 2 * static_cast<s64>(walkers) <= wanted) walkers *= 2;
    return walkers;
}

s64 round_up_to_simd_group(s64 thread_count) {
    return (thread_count + FIT_WALK_THREADS_PER_ROW - 1) / FIT_WALK_THREADS_PER_ROW * FIT_WALK_THREADS_PER_ROW;
}

// The threads fit_walk_residual_kernel runs over row_count rows: one per walker, in whole SIMD
// groups, and at most FIT_REDUCTION_THREAD_LIMIT, each thread striding over the walks.
s64 walk_thread_count(s64 row_count, u32 walkers_per_row) {
    return min(round_up_to_simd_group(max<s64>(row_count, 1) * walkers_per_row), FIT_REDUCTION_THREAD_LIMIT);
}

// The threads the row solve and the gradient run over row_count rows: a SIMD group per
// FIT_WALK_THREADS_PER_ROW / walkers_per_row rows. The gradient's are at most
// FIT_REDUCTION_THREAD_LIMIT, each group striding over the rows; the solve's are not limited.
s64 walk_group_thread_count(s64 row_count, u32 walkers_per_row) {
    const s64 rows_per_group = FIT_WALK_THREADS_PER_ROW / walkers_per_row;
    return (max<s64>(row_count, 1) + rows_per_group - 1) / rows_per_group * FIT_WALK_THREADS_PER_ROW;
}

// Threadgroups a reduction over count indices runs, as run_function groups the threads.
s64 reduction_group_count(s64 count, s64 threads_per_block) {
    const s64 thread_count = reduction_thread_count(count);
    const s64 group_width = max<s64>(1, min(threads_per_block, thread_count));
    return (thread_count + group_width - 1) / group_width;
}

s64 chunk_count_for(s64 count) {
    return (count + HOST_REDUCTION_CHUNK - 1) / HOST_REDUCTION_CHUNK;
}

f64 sum_of(const f32 *values, s64 count) {
    f64 sum = 0.0;
    for (s64 index = 0; index < count; index += 1) sum += static_cast<f64>(values[index]);
    return sum;
}

// Values are never negative; a NaN counts as infinite.
f64 largest_of(const f32 *values, s64 count) {
    f64 largest = 0.0;
    for (s64 index = 0; index < count; index += 1) {
        const f64 value = static_cast<f64>(values[index]);
        largest = isnan(value) ? Numeric_Limits<f64>::infinity() : max(largest, value);
    }
    return largest;
}

// Bytes bound inline (setBytes on Metal), for a small array.
template <typename Element_Type>
Engine_Pointer inline_array_argument(const Vector<Element_Type> &values) {
    return Engine_Pointer{
        const_cast<Element_Type *>(values.data()),
        0,
        values.size() * sizeof(Element_Type),
        alignof(Element_Type),
        true,
    };
}

} // namespace

Weight_Matrix::Fit_Scratch::Fit_Scratch(Engine_Backend &backend, const Vector<u64> &byte_counts)
    : backend(&backend) {
    for (u64 bytes : byte_counts) {
        backend.partition(max<u64>(bytes, sizeof(u32)), Engine_Datatype::UNSIGNED64, buffers);
    }
    slab = backend.allocate(buffers);
}

Weight_Matrix::Fit_Scratch::~Fit_Scratch() {
    backend->deallocate_slab(slab);
}

// ── a fit's values ──────────────────────────────────────────────────────────

template <typename Visit>
void Weight_Matrix::for_each_connection_of(u32 node, bool along_column, Vector<u32> &buffer, const Visit &visit) const {
    const s64 limit = along_column ? max_predecessor_count : max_neighbor_count;
    if (static_cast<s64>(buffer.size()) < max<s64>(limit, 1)) buffer.resize(static_cast<usize>(max<s64>(limit, 1)));
    const s64 count = along_column ? k2tree.get_predecessors(node, buffer.data(), limit)
                                   : k2tree.get_neighbors(node, buffer.data(), limit);
    for (s64 index = 0; index < count; index += 1) visit(buffer[static_cast<usize>(index)]);
}

Weight_Matrix::Fit_Values Weight_Matrix::plane_values_from(const Function<f32(s64)> &value_of_ordinal) const {
    require_storage("plane_values_from");
    Vector<s64> row_entry_count(static_cast<usize>(node_count), 0);
    for_each_index_in_parallel(node_count, [&](s64 source_node) {
        s64 count = 0;
        for (s64 ordinal = edge_row_offset_host[static_cast<usize>(source_node)];
             ordinal < edge_row_offset_host[static_cast<usize>(source_node) + 1]; ordinal += 1) {
            if (value_of_ordinal(ordinal) != 0.0f) count += 1;
        }
        row_entry_count[static_cast<usize>(source_node)] = count;
    });
    s64 entry_count = 0;
    for (s64 count : row_entry_count) entry_count += count;
    if (entry_count > static_cast<s64>(Numeric_Limits<s32>::max())) {
        log::throw_runtime_error(log::logger(),
            "Weight_Matrix: a plane's " + to_string(entry_count) + " nonzero values are more than the 2^31 a fit "
            "can index");
    }

    Fit_Values values;
    values.entry_count = entry_count;
    values.storage = make_unique<Fit_Scratch>(*owning_backend, Vector<u64>{
        static_cast<u64>(node_count + 1) * sizeof(s32), static_cast<u64>(entry_count) * sizeof(u32),
        static_cast<u64>(entry_count) * sizeof(f32)});
    values.row_start = values.storage->buffers[0];
    values.target = values.storage->buffers[1];
    values.value = values.storage->buffers[2];

    s32 *row_start = values.row_start.get_contents_as<s32>();
    s64 position = 0;
    for (s64 node_index = 0; node_index < node_count; node_index += 1) {
        row_start[node_index] = static_cast<s32>(position);
        position += row_entry_count[static_cast<usize>(node_index)];
    }
    row_start[node_count] = static_cast<s32>(position);

    // A row's targets come from its walk, in ordinal order.
    u32 *entry_target = values.target.get_contents_as<u32>();
    f32 *entry_value = values.value.get_contents_as<f32>();
    for_each_index_in_parallel(node_count, [&](s64 source_node) {
        if (row_entry_count[static_cast<usize>(source_node)] == 0) return;
        const s64 first_ordinal = edge_row_offset_host[static_cast<usize>(source_node)];
        s64 entry = row_start[source_node];
        s64 ordinal = first_ordinal;
        Vector<u32> buffer;
        for_each_connection_of(static_cast<u32>(source_node), false, buffer, [&](u32 target_node) {
            const f32 edge_value = value_of_ordinal(ordinal);
            ordinal += 1;
            if (edge_value == 0.0f) return;
            entry_target[entry] = target_node;
            entry_value[entry] = edge_value;
            entry += 1;
        });
    });
    return values;
}

Weight_Matrix::Fit_Values Weight_Matrix::updates_of_plane(s64 matrix_index) const {
    Fit_Values updates;
    updates.row_start = sparse_delta_row_start;
    updates.row_start.offset += matrix_index * (node_count + 1) * static_cast<s64>(sizeof(s32));
    updates.row_start.total_bytes = static_cast<u64>(node_count + 1) * sizeof(s32);
    updates.target = sparse_delta_edge_ordinal;
    updates.value = sparse_delta_value;
    updates.entry_count = sparse_delta_entry_count[static_cast<usize>(matrix_index)];
    updates.adds_to_basis = true;
    return updates;
}

namespace {

// Rewrites the keys of one plane's S rows between edge ordinals and target nodes, each row from
// its walk. Within a row the two orders agree, so the entries stay sorted.
void rekey_rows(const K2_Tree &tree, const Vector<s64> &edge_row_offset, s64 node_count, s64 max_neighbor_count,
                const s32 *row_start, u32 *keys, bool to_target_nodes) {
    for_each_index_in_parallel(node_count, [&](s64 source_node) {
        const s32 first = row_start[source_node];
        const s32 last = row_start[source_node + 1];
        if (first == last) return;
        Vector<u32> targets(static_cast<usize>(max<s64>(max_neighbor_count, 1)));
        const s64 degree = tree.get_neighbors(static_cast<u32>(source_node), targets.data(), max_neighbor_count);
        const s64 first_ordinal = edge_row_offset[static_cast<usize>(source_node)];
        for (s32 position = first; position < last; position += 1) {
            if (to_target_nodes) {
                keys[position] = targets[static_cast<usize>(static_cast<s64>(keys[position]) - first_ordinal)];
            } else {
                const u32 *found = lower_bound(targets.data(), targets.data() + degree, keys[position]);
                keys[position] = static_cast<u32>(first_ordinal + (found - targets.data()));
            }
        }
    });
}

} // namespace

void Weight_Matrix::key_plane_updates_by_target(s64 matrix_index) {
    if (sparse_delta_entry_count[static_cast<usize>(matrix_index)] == 0 || sparse_delta_row_start.is_empty()) return;
    rekey_rows(k2tree, edge_row_offset_host, node_count, max_neighbor_count,
               sparse_delta_row_start.get_contents_as<s32>() + matrix_index * (node_count + 1),
               sparse_delta_edge_ordinal.get_contents_as<u32>(), true);
}

void Weight_Matrix::key_plane_updates_by_ordinal(s64 matrix_index) {
    if (sparse_delta_entry_count[static_cast<usize>(matrix_index)] == 0 || sparse_delta_row_start.is_empty()) return;
    rekey_rows(k2tree, edge_row_offset_host, node_count, max_neighbor_count,
               sparse_delta_row_start.get_contents_as<s32>() + matrix_index * (node_count + 1),
               sparse_delta_edge_ordinal.get_contents_as<u32>(), false);
}

Weight_Matrix::Plane_Value_Statistics Weight_Matrix::plane_value_statistics(s64 matrix_index,
                                                                            const Fit_Values &values) const {
    Plane_Value_Statistics statistics;
    if (total_edge_count == 0) return statistics;
    const s32 *row_start = values.row_start.get_contents_as<s32>();
    const u32 *entry_target = values.target.get_contents_as<u32>();
    const f32 *entry_value = values.value.get_contents_as<f32>();

    // Per row, the sum of the finite values' squares and the count of the others.
    Vector<f64> row_sum_of_squares(static_cast<usize>(node_count), 0.0);
    Vector<s64> row_non_finite_count(static_cast<usize>(node_count), 0);
    if (!values.adds_to_basis) {
        // The values are the entries; every other edge holds zero.
        for_each_index_in_parallel(node_count, [&](s64 source_node) {
            for (s32 entry = row_start[source_node]; entry < row_start[source_node + 1]; entry += 1) {
                const f64 value = static_cast<f64>(entry_value[entry]);
                if (!isfinite(value)) {
                    row_non_finite_count[static_cast<usize>(source_node)] += 1;
                    continue;
                }
                row_sum_of_squares[static_cast<usize>(source_node)] += value * value;
            }
        });
    } else {
        // Each edge's value is the basis plus its entry, which takes every row's walk.
        const f32 *coefficient_values = coefficient_row(matrix_index);
        for_each_index_in_parallel(node_count, [&](s64 source_node) {
            s32 entry = row_start[source_node];
            const s32 end = row_start[source_node + 1];
            Vector<u32> buffer;
            for_each_connection_of(static_cast<u32>(source_node), false, buffer, [&](u32 target_node) {
                while (entry < end && entry_target[entry] < target_node) entry += 1;
                const f32 update = (entry < end && entry_target[entry] == target_node) ? entry_value[entry] : 0.0f;
                const f64 value = static_cast<f64>(reconstruct_entry(static_cast<s32>(source_node), static_cast<s32>(target_node),
                                                                     coefficient_values)) + static_cast<f64>(update);
                if (!isfinite(value)) {
                    row_non_finite_count[static_cast<usize>(source_node)] += 1;
                    return;
                }
                row_sum_of_squares[static_cast<usize>(source_node)] += value * value;
            });
        });
    }

    f64 sum_of_squares = 0.0;
    for (s64 node_index = 0; node_index < node_count; node_index += 1) {
        sum_of_squares += row_sum_of_squares[static_cast<usize>(node_index)];
        statistics.non_finite_count += row_non_finite_count[static_cast<usize>(node_index)];
    }
    const s64 finite_count = total_edge_count - statistics.non_finite_count;
    statistics.root_mean_square = finite_count > 0 ? sqrt(sum_of_squares / static_cast<f64>(finite_count)) : 0.0;
    return statistics;
}

f32 Weight_Matrix::fit_value_at(const Fit_Values &values, u32 source_node, u32 target_node) {
    const s32 *row_start = values.row_start.get_contents_as<s32>();
    const u32 *entry_target = values.target.get_contents_as<u32>();
    const s32 first = row_start[source_node];
    const s32 last = row_start[source_node + 1];
    if (first == last) return 0.0f;
    const u32 *found = lower_bound(entry_target + first, entry_target + last, target_node);
    if (found == entry_target + last || *found != target_node) return 0.0f;
    return values.value.get_contents_as<f32>()[found - entry_target];
}

f32 Weight_Matrix::fit_target(const Plane_Fit_Problem &problem, u32 source_node, u32 target_node, f32 value) const {
    f32 basis_part = 0.0f;
    if (!problem.target_lanes.empty()) {
        const s64 basis_lane_count = rank_float4_stride * LANE_GROUP;
        const f32 *u_row = U_matrix.get_contents_as<f32>() + static_cast<s64>(source_node) * basis_lane_count;
        const f32 *v_row = V_matrix.get_contents_as<f32>() + static_cast<s64>(target_node) * basis_lane_count;
        const f32 *coefficient_values = coefficient_row(problem.matrix_index);
        for (u32 lane : problem.target_lanes) basis_part += u_row[lane] * coefficient_values[lane] * v_row[lane];
    }
    const f32 inverse_scale = problem.scale > 0.0 ? static_cast<f32>(1.0 / problem.scale) : 0.0f;
    return (problem.target_lane_sign * basis_part + value) * inverse_scale;
}

Weight_Matrix::Fit_Walk_Constants Weight_Matrix::fit_walk_constants(const Plane_Fit_Problem &problem,
                                                                    f32 ridge_regularization) const {
    Fit_Walk_Constants constants{};
    constants.node_count = static_cast<u32>(node_count);
    constants.branching_factor = static_cast<u32>(k2tree.branching_factor);
    constants.superblock_size_words = static_cast<u32>(k2tree.superblock_size_words);
    constants.padded_node_count = static_cast<u32>(k2tree.padded_node_count);
    constants.tree_height = static_cast<u32>(k2tree.tree_height);
    constants.internal_bit_count = static_cast<u32>(k2tree.internal_bit_count);
    constants.basis_lane_count = static_cast<u32>(rank_float4_stride * LANE_GROUP);
    constants.target_lane_count = static_cast<u32>(problem.target_lanes.size());
    constants.target_lane_sign = problem.target_lane_sign;
    constants.inverse_scale = problem.scale > 0.0 ? static_cast<f32>(1.0 / problem.scale) : 0.0f;
    constants.lane_count = static_cast<u32>(problem.own_lanes.size());
    constants.ridge = ridge_regularization;
    constants.walkers_per_row = walkers_per_row_for(total_edge_count, node_count);
    return constants;
}

Vector<Engine_Pointer> Weight_Matrix::fit_walk_arguments(const Plane_Fit_Problem &problem,
                                                         const Fit_Walk_Constants &constants) const {
    // Every argument must be a real buffer. An empty one (a tree too small for internal bits, or
    // values without entries) is never read, so a buffer of the problem's stands in for it.
    const auto bound = [&](const Engine_Pointer &pointer) {
        return pointer.is_empty() ? problem.moved_row_buffer : pointer;
    };
    return {
        bound(k2tree.internal_node_words),        // 0
        bound(k2tree.leaf_node_words),            // 1
        bound(k2tree.rank_superblock_table),      // 2
        bound(k2tree.rank_subblock_table),        // 3
        U_matrix,                                 // 4
        V_matrix,                                 // 5
        coefficient_range(problem.matrix_index),  // 6
        problem.target_lane_buffer,               // 7
        problem.values->row_start,                // 8
        bound(problem.values->target),            // 9
        bound(problem.values->value),             // 10
        inline_scalar_argument(constants),        // 11
    };
}

// ── lanes ───────────────────────────────────────────────────────────────────

Vector<s64> Weight_Matrix::lanes_owned_by(s64 matrix_index) const {
    Vector<s64> lanes;
    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    for (s64 lane = 0; lane < lane_count; lane += 1) {
        if (coefficient_row(matrix_index)[lane] == 0.0f) continue;
        bool read_by_another_plane = false;
        for (s64 plane = 0; plane < matrix_count; plane += 1) {
            read_by_another_plane = read_by_another_plane || (plane != matrix_index && coefficient_row(plane)[lane] != 0.0f);
        }
        if (!read_by_another_plane) lanes.push_back(lane);
    }
    return lanes;
}

Vector<s64> Weight_Matrix::lanes_shared_by(s64 matrix_index) const {
    Vector<s64> lanes;
    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    for (s64 lane = 0; lane < lane_count; lane += 1) {
        if (coefficient_row(matrix_index)[lane] == 0.0f) continue;
        bool read_by_another_plane = false;
        for (s64 plane = 0; plane < matrix_count; plane += 1) {
            read_by_another_plane = read_by_another_plane || (plane != matrix_index && coefficient_row(plane)[lane] != 0.0f);
        }
        if (read_by_another_plane) lanes.push_back(lane);
    }
    return lanes;
}

s64 Weight_Matrix::free_lane_count() const {
    s64 free_count = 0;
    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    for (s64 lane = 0; lane < lane_count; lane += 1) {
        bool read = false;
        for (s64 plane = 0; plane < matrix_count; plane += 1) read = read || coefficient_row(plane)[lane] != 0.0f;
        free_count += read ? 0 : 1;
    }
    return free_count;
}

s64 Weight_Matrix::available_lane_count(s64 matrix_index) const {
    const s64 lane_limit = MAX_RANK_FLOAT4_STRIDE * LANE_GROUP;
    return static_cast<s64>(lanes_owned_by(matrix_index).size()) + free_lane_count() +
           (lane_limit - rank_float4_stride * LANE_GROUP);
}

void Weight_Matrix::take_lanes(s64 matrix_index, s64 count, f64 scale) {
    if (count <= 0) return;

    Vector<s64> taken;
    const s64 old_lane_count = rank_float4_stride * LANE_GROUP;
    for (s64 lane = 0; lane < old_lane_count && static_cast<s64>(taken.size()) < count; lane += 1) {
        bool read = false;
        for (s64 plane = 0; plane < matrix_count; plane += 1) read = read || coefficient_row(plane)[lane] != 0.0f;
        if (!read) taken.push_back(lane);
    }
    const s64 missing = count - static_cast<s64>(taken.size());
    if (missing > 0) {
        grow_basis(old_lane_count + missing);
        for (s64 lane = old_lane_count; lane < old_lane_count + missing; lane += 1) taken.push_back(lane);
    }

    const s64 lane_limit = MAX_RANK_FLOAT4_STRIDE * LANE_GROUP;
    Mersenne_Twister random_engine(basis_seed + static_cast<unsigned>(matrix_index * lane_limit + taken.front()));
    Normal_Distribution<f32> normal_distribution_unit(0.0f, 1.0f);
    f32 *u_data = U_matrix.get_contents_as<f32>();
    f32 *v_data = V_matrix.get_contents_as<f32>();
    f32 *coefficient_values = coefficient_row(matrix_index);
    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    for (s64 lane : taken) {
        for (s64 node_index = 0; node_index < node_count; node_index += 1) {
            u_data[node_index * lane_count + lane] = 0.0f;
            v_data[node_index * lane_count + lane] = normal_distribution_unit(random_engine);
        }
        coefficient_values[lane] = static_cast<f32>(scale);
    }
}

void Weight_Matrix::release_lanes(s64 matrix_index, s64 kept_count) {
    const Vector<s64> own_lanes = lanes_owned_by(matrix_index);
    f32 *u_data = U_matrix.get_contents_as<f32>();
    f32 *v_data = V_matrix.get_contents_as<f32>();
    f32 *coefficient_values = coefficient_row(matrix_index);
    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    for (usize index = static_cast<usize>(max<s64>(kept_count, 0)); index < own_lanes.size(); index += 1) {
        const s64 lane = own_lanes[index];
        for (s64 node_index = 0; node_index < node_count; node_index += 1) {
            u_data[node_index * lane_count + lane] = 0.0f;
            v_data[node_index * lane_count + lane] = 0.0f;
        }
        coefficient_values[lane] = 0.0f;
    }
}

void Weight_Matrix::compact_basis() {
    const s64 old_lane_count = rank_float4_stride * LANE_GROUP;
    Vector<s64> read_lanes;
    for (s64 lane = 0; lane < old_lane_count; lane += 1) {
        bool read = false;
        for (s64 plane = 0; plane < matrix_count; plane += 1) read = read || coefficient_row(plane)[lane] != 0.0f;
        if (read) read_lanes.push_back(lane);
    }
    const s64 compact_rank = round_up_to_lane_group(read_lanes.size());
    if (compact_rank >= rank) return;
    require_storage("compact_basis");

    const usize kept_count = read_lanes.size();
    Vector<f32> saved_u(static_cast<usize>(node_count) * kept_count);
    Vector<f32> saved_v(static_cast<usize>(node_count) * kept_count);
    Vector<f32> saved_coefficients(static_cast<usize>(matrix_count) * kept_count);
    const f32 *u_data = U_matrix.get_contents_as<f32>();
    const f32 *v_data = V_matrix.get_contents_as<f32>();
    for (s64 node_index = 0; node_index < node_count; node_index += 1) {
        for (usize index = 0; index < kept_count; index += 1) {
            saved_u[static_cast<usize>(node_index) * kept_count + index] = u_data[node_index * old_lane_count + read_lanes[index]];
            saved_v[static_cast<usize>(node_index) * kept_count + index] = v_data[node_index * old_lane_count + read_lanes[index]];
        }
    }
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        for (usize index = 0; index < kept_count; index += 1) {
            saved_coefficients[static_cast<usize>(matrix_index) * kept_count + index] = coefficient_row(matrix_index)[read_lanes[index]];
        }
    }
    owning_backend->deallocate_slab(owning_slab);
    rank = compact_rank;
    rank_float4_stride = rank / LANE_GROUP;
    allocate_basis_storage();

    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    f32 *compact_u = U_matrix.get_contents_as<f32>();
    f32 *compact_v = V_matrix.get_contents_as<f32>();
    memset(compact_u, 0, static_cast<usize>(node_count * lane_count) * sizeof(f32));
    memset(compact_v, 0, static_cast<usize>(node_count * lane_count) * sizeof(f32));
    memset(coefficients.get_contents(), 0, static_cast<usize>(matrix_count * lane_count) * sizeof(f32));
    for (s64 node_index = 0; node_index < node_count; node_index += 1) {
        for (usize index = 0; index < kept_count; index += 1) {
            compact_u[node_index * lane_count + static_cast<s64>(index)] = saved_u[static_cast<usize>(node_index) * kept_count + index];
            compact_v[node_index * lane_count + static_cast<s64>(index)] = saved_v[static_cast<usize>(node_index) * kept_count + index];
        }
    }
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        for (usize index = 0; index < kept_count; index += 1) {
            coefficient_row(matrix_index)[index] = saved_coefficients[static_cast<usize>(matrix_index) * kept_count + index];
        }
    }
    log::logger().debug("compact_basis: {} lanes in use, rank {} after compaction", kept_count, rank);
}

// ── one plane's fit ─────────────────────────────────────────────────────────

Weight_Matrix::Plane_Fit_Problem Weight_Matrix::prepare_plane_fit(s64 matrix_index, const Fit_Values &values,
                                                                  bool every_row_moves) {
    Plane_Fit_Problem problem;
    problem.matrix_index = matrix_index;
    problem.target_tolerance = static_cast<f64>(fit_tolerance);
    problem.values = &values;

    // Updates add to the plane's values before the fit, which its own lanes hold. Whole values are
    // held by the own lanes together with the lanes the plane shares with other planes, which stay
    // as they are, so the own lanes hold the rest. Planes own their lanes, so there are almost never
    // any shared ones.
    const Vector<s64> shared_lanes = lanes_shared_by(matrix_index);
    for (s64 lane : values.adds_to_basis ? lanes_owned_by(matrix_index) : shared_lanes) {
        problem.target_lanes.push_back(static_cast<u32>(lane));
    }
    problem.target_lane_sign = values.adds_to_basis ? 1.0f : -1.0f;

    const Plane_Value_Statistics statistics = plane_value_statistics(matrix_index, values);
    problem.non_finite_value_count = statistics.non_finite_count;
    problem.value_root_mean_square = statistics.root_mean_square;
    problem.scale = statistics.root_mean_square;

    // An all-zero plane still has to cancel what its shared lanes give, against their RMS.
    if (problem.scale == 0.0 && !shared_lanes.empty()) {
        const f32 *u_data = U_matrix.get_contents_as<f32>();
        const f32 *v_data = V_matrix.get_contents_as<f32>();
        const f32 *coefficient_values = coefficient_row(matrix_index);
        const s64 basis_lane_count = rank_float4_stride * LANE_GROUP;
        Vector<f64> row_sum_of_squares(static_cast<usize>(node_count), 0.0);
        for_each_index_in_parallel(node_count, [&](s64 source_node) {
            Vector<u32> buffer;
            for_each_connection_of(static_cast<u32>(source_node), false, buffer, [&](u32 target_node) {
                f64 value = 0.0;
                for (s64 lane : shared_lanes) {
                    value += static_cast<f64>(u_data[source_node * basis_lane_count + lane]) *
                             static_cast<f64>(coefficient_values[lane]) *
                             static_cast<f64>(v_data[static_cast<s64>(target_node) * basis_lane_count + lane]);
                }
                row_sum_of_squares[static_cast<usize>(source_node)] += value * value;
            });
        });
        f64 sum_of_squares = 0.0;
        for (f64 row_sum : row_sum_of_squares) sum_of_squares += row_sum;
        problem.scale = sqrt(sum_of_squares / static_cast<f64>(total_edge_count));
    }

    // The rows that move: every row with a connection, or both ends of each entry.
    problem.source_moves.assign(static_cast<usize>(node_count), 0);
    Vector<u8> target_moves(static_cast<usize>(node_count), 0);
    problem.every_edge_is_touched = every_row_moves;
    if (every_row_moves) {
        for_each_index_in_parallel(node_count, [&](s64 node_index) {
            const bool has_targets = edge_row_offset_host[static_cast<usize>(node_index) + 1] >
                                     edge_row_offset_host[static_cast<usize>(node_index)];
            problem.source_moves[static_cast<usize>(node_index)] = has_targets ? 1 : 0;
            Vector<u32> buffer(static_cast<usize>(max<s64>(max_predecessor_count, 1)));
            const s64 source_count = k2tree.get_predecessors(static_cast<u32>(node_index), buffer.data(), max_predecessor_count);
            target_moves[static_cast<usize>(node_index)] = source_count > 0 ? 1 : 0;
        });
    } else {
        const s32 *row_start = values.row_start.get_contents_as<s32>();
        const u32 *entry_target = values.target.get_contents_as<u32>();
        for (s64 source_node = 0; source_node < node_count; source_node += 1) {
            for (s32 entry = row_start[source_node]; entry < row_start[source_node + 1]; entry += 1) {
                problem.source_moves[static_cast<usize>(source_node)] = 1;
                target_moves[entry_target[entry]] = 1;
            }
        }
    }
    for (s64 node_index = 0; node_index < node_count; node_index += 1) {
        if (problem.source_moves[static_cast<usize>(node_index)]) problem.moved_rows.push_back(static_cast<u32>(node_index));
    }
    problem.moved_source_count = static_cast<s64>(problem.moved_rows.size());
    for (s64 node_index = 0; node_index < node_count; node_index += 1) {
        if (target_moves[static_cast<usize>(node_index)]) problem.moved_rows.push_back(static_cast<u32>(node_index));
    }

    const u64 partial_bytes = static_cast<u64>(reduction_group_count(FIT_REDUCTION_THREAD_LIMIT, owning_backend->threads_per_block) *
                                               FIT_MAXIMUM_BASIS_VECTOR_COUNT) * sizeof(f32);
    const u64 column_partial_bytes = static_cast<u64>(2 * FIT_REDUCTION_THREAD_LIMIT) * sizeof(f32);
    problem.storage = make_unique<Fit_Scratch>(*owning_backend, Vector<u64>{
        static_cast<u64>(node_count) * sizeof(u8), problem.moved_rows.size() * sizeof(u32),
        problem.target_lanes.size() * sizeof(u32), partial_bytes, partial_bytes, column_partial_bytes, partial_bytes,
        partial_bytes, partial_bytes * MAXIMUM_DOT_PRODUCT_LEFT_SLOT_COUNT});
    problem.source_move_buffer = problem.storage->buffers[0];
    problem.moved_row_buffer = problem.storage->buffers[1];
    problem.target_lane_buffer = problem.storage->buffers[2];
    problem.first_partials = problem.storage->buffers[3];
    problem.second_partials = problem.storage->buffers[4];
    problem.column_partials = problem.storage->buffers[5];
    problem.gradient_first_partials = problem.storage->buffers[6];
    problem.gradient_second_partials = problem.storage->buffers[7];
    problem.dot_partials = problem.storage->buffers[8];

    if (node_count > 0) memcpy(problem.source_move_buffer.get_contents(), problem.source_moves.data(), static_cast<usize>(node_count));
    if (!problem.moved_rows.empty()) {
        memcpy(problem.moved_row_buffer.get_contents(), problem.moved_rows.data(), problem.moved_rows.size() * sizeof(u32));
    }
    if (!problem.target_lanes.empty()) {
        memcpy(problem.target_lane_buffer.get_contents(), problem.target_lanes.data(), problem.target_lanes.size() * sizeof(u32));
    }
    return problem;
}

void Weight_Matrix::fill_fresh_rows(const Plane_Fit_Problem &problem, unsigned seed) const {
    const s64 lane_count = static_cast<s64>(problem.own_lanes.size());
    f32 *working_u = problem.working_u.get_contents_as<f32>();
    f32 *working_v = problem.working_v.get_contents_as<f32>();
    Mersenne_Twister random_engine(seed);
    Normal_Distribution<f32> distribution(0.0f, static_cast<f32>(pow(static_cast<f64>(max<s64>(lane_count, 1)), -0.25)));
    for (s64 index = 0; index < node_count * lane_count; index += 1) {
        working_u[index] = distribution(random_engine);
        working_v[index] = distribution(random_engine);
    }
    if (lane_count == 0) return;

    // The first two lanes hold a source part plus a target part: each source row's mean, and each
    // target column's mean less the mean of every value. Values of that shape start next to their
    // exact fit, which a random start can miss for good. The means are over the finite targets.
    const Fit_Values &values = *problem.values;
    Vector<f64> row_sum(static_cast<usize>(node_count), 0.0);
    Vector<s64> row_count(static_cast<usize>(node_count), 0);
    Vector<f64> column_sum(static_cast<usize>(node_count), 0.0);
    Vector<s64> column_count(static_cast<usize>(node_count), 0);
    for_each_index_in_parallel(node_count, [&](s64 node_index) {
        const u32 node = static_cast<u32>(node_index);
        Vector<u32> buffer;
        for_each_connection_of(node, false, buffer, [&](u32 target_node) {
            const f32 target = fit_target(problem, node, target_node, fit_value_at(values, node, target_node));
            if (!isfinite(target)) return;
            row_sum[static_cast<usize>(node_index)] += static_cast<f64>(target);
            row_count[static_cast<usize>(node_index)] += 1;
        });
        if (lane_count < 2) return;
        for_each_connection_of(node, true, buffer, [&](u32 source_node) {
            const f32 target = fit_target(problem, source_node, node, fit_value_at(values, source_node, node));
            if (!isfinite(target)) return;
            column_sum[static_cast<usize>(node_index)] += static_cast<f64>(target);
            column_count[static_cast<usize>(node_index)] += 1;
        });
    });
    f64 overall_sum = 0.0;
    s64 overall_count = 0;
    for (s64 node_index = 0; node_index < node_count; node_index += 1) {
        overall_sum += row_sum[static_cast<usize>(node_index)];
        overall_count += row_count[static_cast<usize>(node_index)];
    }
    const f64 overall_mean = overall_count > 0 ? overall_sum / static_cast<f64>(overall_count) : 0.0;
    for_each_index_in_parallel(node_count, [&](s64 node_index) {
        const usize node = static_cast<usize>(node_index);
        // The random part already in place, shrunk, stays on: a mean of zero would otherwise start
        // the fit on a saddle where every gradient is zero.
        working_u[node_index * lane_count] =
            (row_count[node] > 0 ? static_cast<f32>(row_sum[node] / static_cast<f64>(row_count[node])) : 0.0f) +
            0.1f * working_u[node_index * lane_count];
        working_v[node_index * lane_count] = 1.0f + 0.1f * working_v[node_index * lane_count];
        if (lane_count < 2) return;

        working_u[node_index * lane_count + 1] = 1.0f + 0.1f * working_u[node_index * lane_count + 1];
        working_v[node_index * lane_count + 1] =
            (column_count[node] > 0 ? static_cast<f32>(column_sum[node] / static_cast<f64>(column_count[node]) - overall_mean)
                                    : 0.0f) +
            0.1f * working_v[node_index * lane_count + 1];
    });
}

void Weight_Matrix::load_working_columns(Plane_Fit_Problem &problem) {
    problem.own_lanes = lanes_owned_by(problem.matrix_index);
    const usize own_count = problem.own_lanes.size();
    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    f32 *u_data = U_matrix.get_contents_as<f32>();
    const f32 *v_data = V_matrix.get_contents_as<f32>();
    f32 *coefficient_values = coefficient_row(problem.matrix_index);

    // Own lanes carry the scale as their coefficient, so the working arrays hold values in the
    // units the fits work in. A lane fitted at another scale is rescaled without changing its values.
    for (s64 lane : problem.own_lanes) {
        const f32 factor = static_cast<f32>(static_cast<f64>(coefficient_values[lane]) / problem.scale);
        for (s64 node_index = 0; node_index < node_count; node_index += 1) {
            u_data[node_index * lane_count + lane] *= factor;
        }
        coefficient_values[lane] = static_cast<f32>(problem.scale);
    }

    const u64 working_bytes = static_cast<u64>(node_count) * max<u64>(own_count, 1) * sizeof(f32);
    problem.working_storage = make_unique<Fit_Scratch>(*owning_backend, Vector<u64>{working_bytes, working_bytes});
    problem.working_u = problem.working_storage->buffers[0];
    problem.working_v = problem.working_storage->buffers[1];
    f32 *working_u = problem.working_u.get_contents_as<f32>();
    f32 *working_v = problem.working_v.get_contents_as<f32>();
    for_each_index_in_parallel(node_count, [&](s64 node_index) {
        for (usize column = 0; column < own_count; column += 1) {
            working_u[static_cast<usize>(node_index) * own_count + column] = u_data[node_index * lane_count + problem.own_lanes[column]];
            working_v[static_cast<usize>(node_index) * own_count + column] = v_data[node_index * lane_count + problem.own_lanes[column]];
        }
    });
}

void Weight_Matrix::store_working_columns(const Plane_Fit_Problem &problem) const {
    const usize own_count = problem.own_lanes.size();
    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    f32 *u_data = U_matrix.get_contents_as<f32>();
    f32 *v_data = V_matrix.get_contents_as<f32>();
    const f32 *working_u = problem.working_u.get_contents_as<f32>();
    const f32 *working_v = problem.working_v.get_contents_as<f32>();
    for_each_index_in_parallel(node_count, [&](s64 node_index) {
        for (usize column = 0; column < own_count; column += 1) {
            u_data[node_index * lane_count + problem.own_lanes[column]] = working_u[static_cast<usize>(node_index) * own_count + column];
            v_data[node_index * lane_count + problem.own_lanes[column]] = working_v[static_cast<usize>(node_index) * own_count + column];
        }
    });
}

f64 Weight_Matrix::fit_plane_with_selected_method(Plane_Fit_Problem &problem, f32 ridge_regularization) {
    if (problem.own_lanes.empty()) return evaluate_edges(problem).worst_error;

    switch (fit_method) {
        case Fit_Method::LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO:
            return fit_plane_by_limited_memory_broyden_fletcher_goldfarb_shanno(problem, ridge_regularization);
        case Fit_Method::MASKED_ALTERNATING_LEAST_SQUARES:
            return fit_plane_by_masked_alternating_least_squares(problem, ridge_regularization);
    }
    return fit_plane_by_masked_alternating_least_squares(problem, ridge_regularization);
}

f64 Weight_Matrix::fit_plane_by_masked_alternating_least_squares(Plane_Fit_Problem &problem,
                                                                 f32 ridge_regularization) {
    const auto started = Steady_Clock::now();
    const s64 lane_count = static_cast<s64>(problem.own_lanes.size());
    const s64 moved_row_count = static_cast<s64>(problem.moved_rows.size());
    const s64 largest_side = max(problem.moved_source_count, moved_row_count - problem.moved_source_count);
    const s64 scratch_float_count = clamp<s64>(total_edge_count * matrix_count / 4, FIT_SOLVE_SCRATCH_MINIMUM_FLOAT_COUNT,
                                               FIT_SOLVE_SCRATCH_FLOAT_LIMIT);
    const s64 batch_row_count = max<s64>(1, min(scratch_float_count / (lane_count * lane_count + 2 * lane_count),
                                                max<s64>(largest_side, 1)));

    Fit_Scratch scratch(*owning_backend, Vector<u64>{
        static_cast<u64>(batch_row_count * lane_count * lane_count) * sizeof(f32),
        static_cast<u64>(batch_row_count * 2 * lane_count) * sizeof(f32)});
    const Engine_Pointer triangle_scratch = scratch.buffers[0];
    const Engine_Pointer vector_scratch = scratch.buffers[1];

    // The best sweep's working arrays, whole: balancing rescales rows that do not move too.
    const usize working_value_count = static_cast<usize>(node_count * lane_count);
    Vector<f32> best_u(problem.working_u.get_contents_as<f32>(), problem.working_u.get_contents_as<f32>() + working_value_count);
    Vector<f32> best_v(problem.working_v.get_contents_as<f32>(), problem.working_v.get_contents_as<f32>() + working_value_count);

    // Sweeps until every touched edge meets the tolerance, or the best worst-case error stalls,
    // keeping the rows of the best sweep.
    f64 best_error = evaluate_edges(problem).worst_error;
    Vector<f64> best_error_after_sweep = {best_error};
    s32 sweep = 1;
    for (; best_error > problem.target_tolerance && sweep <= MASKED_ALTERNATING_LEAST_SQUARES_MAXIMUM_SWEEP_COUNT;
         sweep += 1) {
        balance_lanes(problem);
        solve_rows(problem, true, ridge_regularization, triangle_scratch, vector_scratch, batch_row_count);
        balance_lanes(problem);
        solve_rows(problem, false, ridge_regularization, triangle_scratch, vector_scratch, batch_row_count);

        const f64 error = evaluate_edges(problem).worst_error;
        log::logger().trace("masked alternating least squares: plane {} sweep {}, worst error {:.3e}",
                            problem.matrix_index, sweep, error);
        if (error < best_error) {
            best_error = error;
            best_u.assign(problem.working_u.get_contents_as<f32>(), problem.working_u.get_contents_as<f32>() + working_value_count);
            best_v.assign(problem.working_v.get_contents_as<f32>(), problem.working_v.get_contents_as<f32>() + working_value_count);
        }
        best_error_after_sweep.push_back(best_error);
        if (sweep >= MASKED_ALTERNATING_LEAST_SQUARES_STALL_WINDOW_SWEEP_COUNT &&
            best_error > (1.0 - FIT_STALL_IMPROVEMENT) *
                         best_error_after_sweep[static_cast<usize>(sweep - MASKED_ALTERNATING_LEAST_SQUARES_STALL_WINDOW_SWEEP_COUNT)]) {
            break;
        }
    }
    memcpy(problem.working_u.get_contents(), best_u.data(), working_value_count * sizeof(f32));
    memcpy(problem.working_v.get_contents(), best_v.data(), working_value_count * sizeof(f32));

    log::logger().debug("masked alternating least squares: plane {} at {} lanes, {} moved rows, {} sweeps, worst "
                        "error {:.3e}, {:.3f} s", problem.matrix_index, lane_count, moved_row_count,
                        best_error_after_sweep.size() - 1, best_error,
                        Duration<f64>(Steady_Clock::now() - started).count());
    return best_error;
}

f64 Weight_Matrix::fit_plane_by_limited_memory_broyden_fletcher_goldfarb_shanno(Plane_Fit_Problem &problem,
                                                                                f32 ridge_regularization) {
    const auto started = Steady_Clock::now();
    constexpr u32 history_length = LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO_HISTORY_LENGTH;
    constexpr u32 vector_count = 2 * history_length + 1;
    constexpr u32 gradient_slot = 2 * history_length;
    const u64 length = static_cast<u64>(problem.moved_rows.size() * problem.own_lanes.size());
    const u64 vector_bytes = max<u64>(length, 1) * sizeof(f32);

    // basis holds the history: slot s < history_length is a step, slot history_length + s its
    // gradient change, and the last slot the current gradient. products holds the slots' dot
    // products; history the pairs in use (count, then their step slots, oldest first);
    // coefficients the direction as a combination of the slots; status what the direction step
    // reports (see fit_quasi_newton_direction_kernel); step_length the first trial's step.
    Fit_Scratch scratch(*owning_backend, Vector<u64>{
        vector_bytes, vector_bytes, vector_bytes, vector_bytes, vector_bytes * vector_count,
        static_cast<u64>(vector_count * vector_count) * sizeof(f32), static_cast<u64>(1 + history_length) * sizeof(u32),
        static_cast<u64>(vector_count) * sizeof(f32), 4 * sizeof(f32), sizeof(f32)});
    const Engine_Pointer point = scratch.buffers[0];
    const Engine_Pointer direction = scratch.buffers[1];
    const Engine_Pointer trial_gradient = scratch.buffers[2];
    const Engine_Pointer best_point = scratch.buffers[3];
    const Engine_Pointer basis = scratch.buffers[4];
    const Engine_Pointer products = scratch.buffers[5];
    const Engine_Pointer history = scratch.buffers[6];
    const Engine_Pointer coefficients = scratch.buffers[7];
    const Engine_Pointer status = scratch.buffers[8];
    const Engine_Pointer step_length = scratch.buffers[9];
    // Slots never written are multiplied by a zero coefficient, which must not meet a NaN.
    memset(basis.get_contents(), 0, vector_bytes * vector_count);
    memset(products.get_contents(), 0, static_cast<usize>(vector_count * vector_count) * sizeof(f32));
    memset(history.get_contents(), 0, static_cast<usize>(1 + history_length) * sizeof(u32));
    memset(direction.get_contents(), 0, vector_bytes);

    struct Evaluation {
        f64 objective = 0.0;
        f64 worst_error = 0.0;
        f64 slope = 0.0;  // the objective's derivative along direction
    };
    auto evaluation_from = [](const Edge_Evaluation &edges, const Gradient_Evaluation &gradient) {
        Evaluation evaluation{edges.loss + gradient.ridge_term, edges.worst_error, gradient.slope};
        if (!isfinite(evaluation.objective) || !isfinite(evaluation.slope)) {
            evaluation.objective = Numeric_Limits<f64>::infinity();
            evaluation.slope = 0.0;
        }
        return evaluation;
    };

    // Moves the rows to point + step * direction and evaluates there, leaving the gradient in
    // trial_gradient: one command buffer.
    f64 evaluated_step = 0.0;
    auto evaluate_at = [&](f64 step) {
        Edge_Evaluation edges;
        Gradient_Evaluation gradient;
        const bool batched = run_fit_steps([&] {
            scatter_point(problem, point, direction, static_cast<f32>(step));
            edges = evaluate_edges(problem);
            gradient = evaluate_gradient(problem, ridge_regularization, direction, trial_gradient);
        });
        if (batched) {
            edges = read_edge_evaluation(problem);
            gradient = read_gradient_evaluation(problem);
        }
        evaluated_step = step;
        return evaluation_from(edges, gradient);
    };

    // An iteration's first command buffer, which carries most iterations whole: make the point in
    // the working arrays current (recording its step and gradient change in `recorded_slot` when
    // there is a pair), refresh the changed slots' dot products, keep the point as the best when
    // asked, take the quasi-Newton direction, and evaluate the direction's first trial step. After a
    // failed line search the point stays as it was (nothing to accept) and the history restarts.
    enum class Iteration_Start_Kind { FIRST_GRADIENT, RECORD_PAIR, AFTER_FAILED_SEARCH };
    struct Iteration_Start {
        Evaluation first_trial;
        f64 slope = 0.0;
        f64 first_step = 0.0;
        bool steepest_descent = true;
        bool zero_gradient = false;
    };
    auto start_iteration = [&](Iteration_Start_Kind kind, u32 recorded_slot, bool keep_best) {
        Edge_Evaluation edges;
        Gradient_Evaluation gradient;
        const u32 change_slot = history_length + recorded_slot;
        const bool batched = run_fit_steps([&] {
            if (keep_best) gather_point(problem, best_point);
            if (kind == Iteration_Start_Kind::RECORD_PAIR) {
                accept_step(problem, point, trial_gradient, basis, recorded_slot, change_slot, gradient_slot, true);
                update_dot_products(problem, basis, length, vector_count, {recorded_slot, change_slot, gradient_slot}, products);
            } else if (kind == Iteration_Start_Kind::FIRST_GRADIENT) {
                accept_step(problem, point, trial_gradient, basis, 0, 0, gradient_slot, false);
                update_dot_products(problem, basis, length, vector_count, {gradient_slot}, products);
            }
            quasi_newton_direction(products, history, history_length,
                                   kind == Iteration_Start_Kind::RECORD_PAIR ? recorded_slot : Numeric_Limits<u32>::max(),
                                   kind == Iteration_Start_Kind::AFTER_FAILED_SEARCH, coefficients, status, step_length);
            linear_combination(basis, length, vector_count, coefficients, direction);
            scatter_point(problem, point, direction, step_length);
            edges = evaluate_edges(problem);
            gradient = evaluate_gradient(problem, ridge_regularization, direction, trial_gradient);
        });
        if (batched) {
            edges = read_edge_evaluation(problem);
            gradient = read_gradient_evaluation(problem);
        }
        const f32 *reported = status.get_contents_as<f32>();
        Iteration_Start start;
        start.first_trial = evaluation_from(edges, gradient);
        start.slope = static_cast<f64>(reported[0]);
        start.first_step = static_cast<f64>(reported[1]);
        start.steepest_descent = reported[2] != 0.0f;
        start.zero_gradient = reported[3] != 0.0f;
        evaluated_step = start.first_step;
        return start;
    };

    // Strong Wolfe line search along direction from the current point (Nocedal and Wright,
    // algorithms 3.5 and 3.6), from a first trial already evaluated. Leaves the rows and
    // trial_gradient at the step it returns, unless it evaluated another step last.
    struct Trial {
        f64 step = 0.0;
        Evaluation evaluation;
    };
    Evaluation current;
    auto search_line = [&](const Trial &first_trial, f64 start_slope) -> Optional<Trial> {
        s32 evaluation_count = 1;
        auto evaluate = [&](f64 step) {
            evaluation_count += 1;
            return Trial{step, evaluate_at(step)};
        };
        auto decreases_enough = [&](const Trial &trial) {
            return trial.evaluation.objective <= current.objective + WOLFE_SUFFICIENT_DECREASE * trial.step * start_slope;
        };
        auto flattens_enough = [&](const Trial &trial) {
            return fabs(trial.evaluation.slope) <= -WOLFE_CURVATURE * start_slope;
        };
        // A cubic through both ends' values and slopes, kept in the middle 80% of the bracket;
        // the midpoint when the cubic has no usable minimum.
        auto interpolate = [](const Trial &low, const Trial &high) {
            const f64 first = low.step;
            const f64 second = high.step;
            const f64 first_value = low.evaluation.objective;
            const f64 second_value = high.evaluation.objective;
            const f64 first_slope = low.evaluation.slope;
            const f64 second_slope = high.evaluation.slope;
            f64 step = 0.5 * (first + second);
            if (isfinite(first_value) && isfinite(second_value) && first != second) {
                const f64 combined = first_slope + second_slope - 3.0 * (first_value - second_value) / (first - second);
                const f64 discriminant = combined * combined - first_slope * second_slope;
                if (discriminant >= 0.0) {
                    const f64 root = copysign(sqrt(discriminant), second - first);
                    const f64 denominator = second_slope - first_slope + 2.0 * root;
                    if (denominator != 0.0) {
                        const f64 candidate = second - (second - first) * (second_slope + root - combined) / denominator;
                        const f64 margin = 0.1 * fabs(second - first);
                        if (candidate >= min(first, second) + margin && candidate <= max(first, second) - margin) {
                            step = candidate;
                        }
                    }
                }
            }
            return step;
        };
        // low decreases the objective enough and is the lowest point seen; high brackets it.
        auto zoom = [&](Trial low, Trial high) -> Optional<Trial> {
            while (evaluation_count < LINE_SEARCH_MAXIMUM_EVALUATION_COUNT) {
                const Trial trial = evaluate(interpolate(low, high));
                if (!decreases_enough(trial) || trial.evaluation.objective >= low.evaluation.objective) {
                    high = trial;
                } else {
                    if (flattens_enough(trial)) return trial;
                    if (trial.evaluation.slope * (high.step - low.step) >= 0.0) high = low;
                    low = trial;
                }
                if (fabs(high.step - low.step) <= 1e-12 * max(1.0, fabs(low.step))) break;
            }
            // Out of evaluations: the low end still decreased the objective enough.
            if (low.step > 0.0) return low;
            return nullopt;
        };

        Trial previous{0.0, current};
        Trial trial = first_trial;
        for (s32 attempt = 0;; attempt += 1) {
            if (attempt > 0) {
                if (evaluation_count >= LINE_SEARCH_MAXIMUM_EVALUATION_COUNT) break;
                trial = evaluate(previous.step * 2.0);
            }
            if (!decreases_enough(trial) || (attempt > 0 && trial.evaluation.objective >= previous.evaluation.objective)) {
                return zoom(previous, trial);
            }
            if (flattens_enough(trial)) return trial;
            if (trial.evaluation.slope >= 0.0) return zoom(trial, previous);
            previous = trial;
        }
        if (previous.step > 0.0) return previous;
        return nullopt;
    };

    // The starting point and its gradient.
    balance_lanes(problem);
    gather_point(problem, point);
    current = evaluate_at(0.0);
    Iteration_Start start = start_iteration(Iteration_Start_Kind::FIRST_GRADIENT, 0, false);

    const char *stop_reason = "the iteration cap";
    f64 best_error = current.worst_error;
    gather_point(problem, best_point);
    Vector<f64> best_error_after_iteration = {best_error};
    Vector<f64> objective_after_iteration = {current.objective};
    s32 iteration = 1;
    while (best_error > problem.target_tolerance) {
        if (start.zero_gradient) {
            stop_reason = "a zero gradient";
            break;
        }
        const Optional<Trial> accepted = search_line(Trial{start.first_step, start.first_trial}, start.slope);
        if (!accepted.has_value()) {
            log::logger().trace("limited-memory BFGS: plane {} iteration {}, the line search found no step along "
                                "slope {:.3e} from objective {:.9e}{}", problem.matrix_index, iteration, start.slope,
                                current.objective, start.steepest_descent ? " (steepest descent)" : "");
            // Retry from steepest descent; a failed steepest descent step means no progress.
            if (start.steepest_descent) {
                stop_reason = "a failed steepest-descent line search";
                break;
            }
            start = start_iteration(Iteration_Start_Kind::AFTER_FAILED_SEARCH, 0, false);
            continue;
        }
        if (evaluated_step != accepted->step) evaluate_at(accepted->step);

        current = accepted->evaluation;
        const bool improves = current.worst_error < best_error;
        if (improves) best_error = current.worst_error;
        best_error_after_iteration.push_back(best_error);
        objective_after_iteration.push_back(current.objective);
        log::logger().trace("limited-memory BFGS: plane {} iteration {}, step {:.3e}, objective {:.9e}, worst error "
                            "{:.3e}, {} pairs", problem.matrix_index, iteration, accepted->step, current.objective,
                            current.worst_error, history.get_contents_as<u32>()[0]);

        // The worst edge lags the objective, which is what the iterations lower, so a fit has
        // stalled only when neither improved enough over the window.
        const s64 window = LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO_STALL_WINDOW_ITERATION_COUNT;
        const s64 recorded = static_cast<s64>(best_error_after_iteration.size()) - 1;
        const bool stalled =
            recorded >= window &&
            best_error > (1.0 - FIT_STALL_IMPROVEMENT) * best_error_after_iteration[static_cast<usize>(recorded - window)] &&
            current.objective > (1.0 - FIT_STALL_IMPROVEMENT) * objective_after_iteration[static_cast<usize>(recorded - window)];
        const bool capped = iteration >= LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO_MAXIMUM_ITERATION_COUNT;
        if (best_error <= problem.target_tolerance || stalled || capped) {
            if (improves) gather_point(problem, best_point);
            if (stalled) stop_reason = "a stall";
            break;
        }

        // The pair goes into a free slot, or replaces the oldest; the direction step keeps it only
        // when its curvature is positive, which keeps the estimate positive definite.
        const u32 *history_values = history.get_contents_as<u32>();
        const u32 pair_count = history_values[0];
        u32 slot = 0;
        if (pair_count == history_length) {
            slot = history_values[1];
        } else {
            while (find(history_values + 1, history_values + 1 + pair_count, slot) != history_values + 1 + pair_count) slot += 1;
        }
        start = start_iteration(Iteration_Start_Kind::RECORD_PAIR, slot, improves);
        iteration += 1;
    }
    if (best_error <= problem.target_tolerance) stop_reason = "the tolerance";
    scatter_point(problem, best_point, best_point, 0.0f);

    log::logger().debug("limited-memory BFGS: plane {} at {} lanes, {} moved rows, {} iterations, worst error "
                        "{:.3e}, stopped at {}, {:.3f} s", problem.matrix_index, problem.own_lanes.size(),
                        problem.moved_rows.size(), iteration, best_error, stop_reason,
                        Duration<f64>(Steady_Clock::now() - started).count());
    return best_error;
}

Weight_Matrix::Plane_Fit_Result Weight_Matrix::search_fewest_lanes(s64 matrix_index, const Fit_Values &values,
                                                                  f32 ridge_regularization, f32 search_tolerance,
                                                                  s64 required_lane_count, s64 lane_limit) {
    const auto started = Steady_Clock::now();
    if (values.adds_to_basis && !lanes_owned_by(matrix_index).empty()) {
        log::throw_runtime_error(log::logger(),
            "Weight_Matrix::search_fewest_lanes: plane " + to_string(matrix_index) + " holds lanes, and the search "
            "would free them from under the updates it fits");
    }
    Plane_Fit_Problem problem = prepare_plane_fit(matrix_index, values, true);
    problem.target_tolerance = static_cast<f64>(search_tolerance);
    const bool holds_finite_values = problem.non_finite_value_count == 0;
    const f64 unholdable_error = holds_finite_values ? 0.0 : Numeric_Limits<f64>::infinity();
    if (!holds_finite_values) {
        log::logger().warn("Weight_Matrix: plane {} holds {} NaN or infinite values, which the basis cannot hold",
                           matrix_index, problem.non_finite_value_count);
    }

    // An all-zero plane needs no lanes of its own and reads exactly zero.
    if (problem.value_root_mean_square == 0.0) {
        release_lanes(matrix_index, 0);
        if (problem.scale == 0.0) return Plane_Fit_Result{holds_finite_values, unholdable_error};
    }

    // Each probe fits from fresh N(0, lane_count^-1/4) rows, so a row product starts near the scale,
    // in its own working arrays; the basis changes once, when the search has chosen.
    release_lanes(matrix_index, 0);
    auto use_working_arrays = [&](s64 lane_count) {
        problem.own_lanes.assign(static_cast<usize>(lane_count), -1);
        const u64 working_bytes = static_cast<u64>(node_count * lane_count) * sizeof(f32);
        problem.working_storage = make_unique<Fit_Scratch>(*owning_backend, Vector<u64>{working_bytes, working_bytes});
        problem.working_u = problem.working_storage->buffers[0];
        problem.working_v = problem.working_storage->buffers[1];
    };
    const s64 available = lane_limit > 0 ? min(available_lane_count(matrix_index), lane_limit)
                                         : available_lane_count(matrix_index);
    if (available == 0) {
        // No lanes: the plane reads what its shared lanes give.
        use_working_arrays(0);
        return Plane_Fit_Result{false, max(evaluate_edges(problem).worst_error, unholdable_error)};
    }

    auto fit_fresh = [&](s64 lane_count) {
        use_working_arrays(lane_count);
        fill_fresh_rows(problem, basis_seed + 7919u * static_cast<unsigned>(matrix_index + 1) + static_cast<unsigned>(lane_count));
        return fit_plane_with_selected_method(problem, ridge_regularization);
    };

    struct Kept_Fit {
        s64 lane_count = 0;
        f64 error = Numeric_Limits<f64>::infinity();
        Vector<f32> working_u;
        Vector<f32> working_v;
    };
    auto keep = [&](Kept_Fit &kept, s64 lane_count, f64 error) {
        kept.lane_count = lane_count;
        kept.error = error;
        const usize value_count = static_cast<usize>(node_count * lane_count);
        kept.working_u.assign(problem.working_u.get_contents_as<f32>(), problem.working_u.get_contents_as<f32>() + value_count);
        kept.working_v.assign(problem.working_v.get_contents_as<f32>(), problem.working_v.get_contents_as<f32>() + value_count);
    };

    // Double until a lane count meets the tolerance, then bisect between it and the last that did
    // not; or, with a required count, fit at that count alone.
    Kept_Fit passing;
    Kept_Fit closest;
    s64 failing_lane_count = 0;
    Vector<Pair<s64, f64>> attempts;
    if (required_lane_count > 0) {
        const s64 lane_count = min(required_lane_count, available);
        const f64 error = fit_fresh(lane_count);
        attempts.push_back({lane_count, error});
        keep(error <= static_cast<f64>(search_tolerance) ? passing : closest, lane_count, error);
    }
    for (s64 lane_count = 1; required_lane_count <= 0; lane_count *= 2) {
        lane_count = min(lane_count, available);
        const f64 error = fit_fresh(lane_count);
        attempts.push_back({lane_count, error});
        if (error <= static_cast<f64>(search_tolerance)) {
            keep(passing, lane_count, error);
            break;
        }
        failing_lane_count = lane_count;
        if (closest.lane_count == 0 || error < closest.error) keep(closest, lane_count, error);
        if (lane_count >= available) break;
    }
    while (required_lane_count <= 0 && passing.lane_count > 0 && passing.lane_count - failing_lane_count > 1) {
        const s64 middle = (passing.lane_count + failing_lane_count) / 2;
        const f64 error = fit_fresh(middle);
        attempts.push_back({middle, error});
        if (error <= static_cast<f64>(search_tolerance)) keep(passing, middle, error);
        else failing_lane_count = middle;
    }

    // Puts the chosen fit back into the basis. Every edge was touched, so its error is the plane's.
    const Kept_Fit &chosen = passing.lane_count > 0 ? passing : closest;
    const s64 own_count = static_cast<s64>(lanes_owned_by(matrix_index).size());
    if (own_count < chosen.lane_count) take_lanes(matrix_index, chosen.lane_count - own_count, problem.scale);
    else release_lanes(matrix_index, chosen.lane_count);
    load_working_columns(problem);
    memcpy(problem.working_u.get_contents(), chosen.working_u.data(), chosen.working_u.size() * sizeof(f32));
    memcpy(problem.working_v.get_contents(), chosen.working_v.data(), chosen.working_v.size() * sizeof(f32));
    store_working_columns(problem);

    String attempt_list;
    for (const auto &[lane_count, error] : attempts) {
        if (!attempt_list.empty()) attempt_list += ", ";
        attempt_list += to_string(lane_count) + ": " + log::format_floating(".3f", 100.0 * error) + "%";
    }
    log::logger().debug("search_fewest_lanes: plane {} keeps {} lanes at worst error {:.3f}% after [{}], {:.3f} s",
                        matrix_index, chosen.lane_count, 100.0 * chosen.error, attempt_list,
                        Duration<f64>(Steady_Clock::now() - started).count());

    return Plane_Fit_Result{passing.lane_count > 0 && holds_finite_values, max(chosen.error, unholdable_error)};
}

Weight_Matrix::Plane_Fit_Result Weight_Matrix::refit_plane(s64 matrix_index, const Fit_Values &updates,
                                                          f32 ridge_regularization) {
    f32 *u_data = U_matrix.get_contents_as<f32>();
    const f32 *v_data = V_matrix.get_contents_as<f32>();
    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    const Vector<s64> own_lanes = lanes_owned_by(matrix_index);

    // Lanes that hold nothing (all of U zero) move every row. When V holds rows (calibrated, or
    // kept while the plane's values were all zero) the fit starts from them, solving U against
    // them first; otherwise it starts from fresh rows.
    bool holds_nothing = true;
    bool holds_target_rows = false;
    for (s64 lane : own_lanes) {
        for (s64 node_index = 0; node_index < node_count && holds_nothing; node_index += 1) {
            holds_nothing = u_data[node_index * lane_count + lane] == 0.0f;
        }
        for (s64 node_index = 0; node_index < node_count && !holds_target_rows; node_index += 1) {
            holds_target_rows = v_data[node_index * lane_count + lane] != 0.0f;
        }
    }
    Plane_Fit_Problem problem = prepare_plane_fit(matrix_index, updates, holds_nothing);
    const bool holds_finite_values = problem.non_finite_value_count == 0;
    const f64 unholdable_error = holds_finite_values ? 0.0 : Numeric_Limits<f64>::infinity();
    if (!holds_finite_values) {
        log::logger().warn("Weight_Matrix: plane {} holds {} NaN or infinite values, which the basis cannot hold",
                           matrix_index, problem.non_finite_value_count);
    }

    // A plane whose values are all zero reads exactly zero with its U columns at zero, and keeps its
    // lanes for when it holds values again.
    if (problem.value_root_mean_square == 0.0 && lanes_shared_by(matrix_index).empty()) {
        for (s64 lane : own_lanes) {
            for (s64 node_index = 0; node_index < node_count; node_index += 1) u_data[node_index * lane_count + lane] = 0.0f;
        }
        record_fit_error(matrix_index, unholdable_error);
        return Plane_Fit_Result{holds_finite_values, unholdable_error};
    }

    // A plane with no lanes yet (it started at zero, or was loaded without lanes) gets its budget
    // now, once, from the values it first holds: the fewest lanes that meet fit_tolerance, times one
    // plus expected_active_edge_fraction, within what the cap leaves after the other planes.
    if (own_lanes.empty()) {
        s64 other_planes_lane_count = 0;
        for (s64 plane = 0; plane < matrix_count; plane += 1) {
            if (plane != matrix_index) other_planes_lane_count += static_cast<s64>(lanes_owned_by(plane).size());
        }
        const s64 room = max<s64>(1, lane_cap() - other_planes_lane_count);
        const Plane_Fit_Result searched_fit = search_fewest_lanes(matrix_index, updates, ridge_regularization,
                                                                  fit_tolerance, 0, room);
        const s64 searched = static_cast<s64>(lanes_owned_by(matrix_index).size());
        const f64 overhead = 1.0 + static_cast<f64>(clamp(expected_active_edge_fraction, 0.0f, 1.0f));
        const s64 budget = min(room, max<s64>(1, static_cast<s64>(ceil(static_cast<f64>(searched) * overhead))));
        if (budget > searched) take_lanes(matrix_index, budget - searched, problem.value_root_mean_square);
        log::logger().debug("refit_plane: plane {} takes its lane budget at its first values: {} searched -> {} "
                            "(room {} under the cap)", matrix_index, searched, budget, room);
        record_fit_error(matrix_index, searched_fit.worst_error);
        return searched_fit;
    }

    load_working_columns(problem);
    const s64 own_count = static_cast<s64>(problem.own_lanes.size());
    const unsigned fresh_seed = basis_seed + 7919u * static_cast<unsigned>(matrix_index + 1) + static_cast<unsigned>(own_count);
    if (holds_nothing && !holds_target_rows) fill_fresh_rows(problem, fresh_seed);
    f64 error = fit_plane_with_selected_method(problem, ridge_regularization);

    // A warm start can sit in a poor region: when it misses the tolerance and does more than twice
    // as badly as the plane's last fit, a fresh start at the same lanes, moving every row, is tried
    // and the better fit kept. A plane whose lanes only hold its values that well is not refit
    // twice. Both fits read the plane's values from its lanes as they were, so neither is stored
    // until one is chosen.
    Plane_Fit_Problem fresh_problem;
    const Plane_Fit_Problem *chosen = &problem;
    const f64 last_error = static_cast<usize>(matrix_index) < measured_fit_error.size()
                               ? static_cast<f64>(measured_fit_error[static_cast<usize>(matrix_index)])
                               : 0.0;
    if (!holds_nothing && error > static_cast<f64>(fit_tolerance) && error > 2.0 * last_error) {
        fresh_problem = prepare_plane_fit(matrix_index, updates, true);
        load_working_columns(fresh_problem);
        fill_fresh_rows(fresh_problem, fresh_seed);
        const f64 fresh_error = fit_plane_with_selected_method(fresh_problem, ridge_regularization);
        if (fresh_error < error) {
            chosen = &fresh_problem;
            error = fresh_error;
        }
    }

    // The worst error over every edge, measured while the plane's lanes still hold the values the
    // updates add to.
    const f64 worst_error = max(evaluate_edges(*chosen, true).worst_error, unholdable_error);
    store_working_columns(*chosen);
    record_fit_error(matrix_index, worst_error);

    log::logger().debug("refit_plane: plane {} on {} own lanes, {} updates{}{}, worst error {:.3f}%", matrix_index,
                        own_count, updates.entry_count,
                        holds_nothing ? (holds_target_rows ? ", from its V rows" : ", from fresh rows") : "",
                        chosen == &fresh_problem ? ", fitted again from fresh rows" : "", 100.0 * worst_error);
    return Plane_Fit_Result{error <= static_cast<f64>(fit_tolerance) && holds_finite_values, worst_error};
}

s64 Weight_Matrix::lane_cap() const {
    const s64 kernel_limit = MAX_RANK_FLOAT4_STRIDE * LANE_GROUP;
    // The basis holds (2 * node_count + matrix_count) floats per lane, with its lanes rounded up to
    // the lane group; it must stay below total_edge_count * matrix_count floats.
    const s64 floats_per_lane = 2 * node_count + matrix_count;
    const s64 raw_floats = total_edge_count * matrix_count;
    const s64 below_raw = floats_per_lane > 0 ? max<s64>(raw_floats - 1, 0) / floats_per_lane : 0;
    // The basis is allocated in lane groups, so the lanes of the smallest basis that gives every
    // plane a lane are there whatever the budgets; using them costs nothing.
    const s64 smallest_basis = round_up_to_lane_group(static_cast<usize>(matrix_count));
    return min(kernel_limit, max((below_raw / LANE_GROUP) * LANE_GROUP, smallest_basis));
}

bool Weight_Matrix::fit_every_plane_within_lane_budgets(const Function<Fit_Values(s64)> &plane_values,
                                                        f32 ridge_regularization) {
    memset(coefficients.get_contents(), 0,
           static_cast<usize>(matrix_count) * static_cast<usize>(rank_float4_stride * LANE_GROUP) * sizeof(f32));

    // The fewest lanes that hold each plane's starting values within fit_tolerance.
    Vector<f64> scale(static_cast<usize>(matrix_count), 0.0);
    Vector<s64> searched(static_cast<usize>(matrix_count), 0);
    Vector<char> meets(static_cast<usize>(matrix_count), 1);
    s64 largest_searched = 0;
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        const usize plane = static_cast<usize>(matrix_index);
        const Fit_Values values = plane_values(matrix_index);
        const Plane_Value_Statistics statistics = plane_value_statistics(matrix_index, values);
        scale[plane] = statistics.root_mean_square;
        meets[plane] = statistics.non_finite_count == 0 ? 1 : 0;
        if (scale[plane] == 0.0) {
            record_fit_error(matrix_index, meets[plane] ? 0.0 : Numeric_Limits<f64>::infinity());
            continue;
        }
        const Plane_Fit_Result result = search_fewest_lanes(matrix_index, values, ridge_regularization, fit_tolerance);
        meets[plane] = result.meets_tolerance;
        record_fit_error(matrix_index, result.worst_error);
        searched[plane] = static_cast<s64>(lanes_owned_by(matrix_index).size());
        largest_searched = max(largest_searched, searched[plane]);
    }

    // Each budget: the searched count times one plus the expected active fraction. A plane that
    // starts at zero is taken to become active: it asks for the largest searched count of any plane,
    // times the same.
    const f64 overhead = 1.0 + static_cast<f64>(clamp(expected_active_edge_fraction, 0.0f, 1.0f));
    const s64 cap = lane_cap();
    Vector<s64> budget(static_cast<usize>(matrix_count), 0);
    s64 total = 0;
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        const usize plane = static_cast<usize>(matrix_index);
        if (scale[plane] == 0.0) continue;
        budget[plane] = max<s64>(1, static_cast<s64>(ceil(static_cast<f64>(searched[plane]) * overhead)));
        total += budget[plane];
    }
    const s64 zero_start_budget = max<s64>(1, static_cast<s64>(ceil(static_cast<f64>(largest_searched) * overhead)));
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        if (scale[static_cast<usize>(matrix_index)] != 0.0) continue;
        budget[static_cast<usize>(matrix_index)] = zero_start_budget;
        total += zero_start_budget;
    }

    // Together they never pass the cap. Lanes come off the largest budget first, so a plane that
    // needs few lanes keeps them and the cut falls on the planes that need the most.
    const s64 asked = total;
    while (total > cap) {
        auto largest = max_element(budget.begin(), budget.end());
        if (*largest <= 1) break;
        *largest -= 1;
        total -= 1;
    }
    if (total > cap) {
        log::logger().warn("Weight_Matrix: even one lane per plane ({} lanes) makes the basis larger than one float "
                           "per edge per plane, which {} lanes would keep it under", total, cap);
    } else if (asked > cap) {
        log::logger().warn("Weight_Matrix: the planes' lane budgets asked for {} lanes; {} keep the basis below one "
                           "float per edge per plane, so the largest budgets were cut", asked, cap);
    }

    // Each plane with values takes its budget. Lanes added to a fitted plane start with U zero, so
    // its values stay as fitted; a plane fitted on more lanes than its budget is fitted again on its
    // budget. A plane that starts at zero takes none and reads exactly zero; its room under the cap
    // stays free for its first refit.
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        const usize plane = static_cast<usize>(matrix_index);
        if (scale[plane] == 0.0) continue;
        if (budget[plane] > searched[plane]) {
            take_lanes(matrix_index, budget[plane] - searched[plane], scale[plane]);
        } else if (budget[plane] < searched[plane]) {
            const Fit_Values values = plane_values(matrix_index);
            const Plane_Fit_Result result = search_fewest_lanes(matrix_index, values, ridge_regularization, fit_tolerance,
                                                                budget[plane]);
            meets[plane] = result.meets_tolerance;
            record_fit_error(matrix_index, result.worst_error);
        }
    }

    String budget_list;
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        if (!budget_list.empty()) budget_list += ", ";
        budget_list += to_string(searched[static_cast<usize>(matrix_index)]) + " -> " + to_string(budget[static_cast<usize>(matrix_index)]);
    }
    log::logger().debug("Weight_Matrix: lane budgets per plane (searched -> budget) [{}], cap {}", budget_list, cap);
    return all_of(meets.begin(), meets.end(), [](char plane_meets) { return plane_meets != 0; });
}

// ── fit steps ───────────────────────────────────────────────────────────────

bool Weight_Matrix::run_fit_kernel(const String &name, Vector<Engine_Pointer> &parameters, s64 job_count) const {
    Engine_Function &function = fit_functions[name];
    if (!ensure_function(function, name)) return false;
    if (collecting_batch == nullptr) return owning_backend->run_function(function, parameters, job_count);

    for (Engine_Pointer &parameter : parameters) {
        if (!parameter.inline_scalar) continue;
        const u8 *bytes = static_cast<const u8 *>(parameter.base_pointer);
        collecting_batch->scalar_storage.emplace_back(bytes, bytes + parameter.total_bytes);
        parameter.base_pointer = collecting_batch->scalar_storage.back().data();
    }
    collecting_batch->dispatches.push_back(Engine_Dispatch{&function, parameters, job_count});
    return true;
}

bool Weight_Matrix::fit_kernels_available() const {
    static const char *const FIT_KERNEL_NAMES[] = {
        "fit_walk_residual_kernel", "fit_walk_row_solve_kernel", "fit_walk_gradient_kernel", "fit_scatter_point_kernel",
        "fit_gather_point_kernel", "fit_accept_step_kernel", "fit_column_square_sums_kernel",
        "fit_scale_columns_kernel", "fit_dot_products_kernel", "fit_linear_combination_kernel",
        "fit_reduce_dot_products_kernel", "fit_quasi_newton_direction_kernel"};
    for (const char *name : FIT_KERNEL_NAMES) {
        if (!ensure_function(fit_functions[name], name)) return false;
    }
    return true;
}

bool Weight_Matrix::run_fit_steps(const Function<void()> &encode) const {
    if (!fit_kernels_available()) {
        encode();
        return false;
    }
    Fit_Kernel_Batch batch;
    collecting_batch = &batch;
    encode();
    collecting_batch = nullptr;
    if (!owning_backend->run_functions(batch.dispatches)) {
        log::throw_runtime_error(log::logger(), "Weight_Matrix: a batch of " + to_string(batch.dispatches.size()) +
                                                " fit kernels failed on the device");
    }
    return true;
}

Weight_Matrix::Edge_Evaluation Weight_Matrix::evaluate_edges(const Plane_Fit_Problem &problem, bool every_edge) const {
    // Every edge: every node's row walk. The touched edges: the moved sources' row walks, then the
    // moved targets' column walks, which skip the connections of moved sources.
    const bool rows_are_every_node = every_edge || problem.every_edge_is_touched;
    const u32 row_walk_count = static_cast<u32>(rows_are_every_node ? node_count : problem.moved_source_count);
    const u32 column_walk_count = static_cast<u32>(
        rows_are_every_node ? 0 : static_cast<s64>(problem.moved_rows.size()) - problem.moved_source_count);
    const u32 walkers_per_row = walkers_per_row_for(total_edge_count, node_count);
    const s64 row_thread_count = walk_thread_count(row_walk_count, walkers_per_row);
    const s64 column_thread_count = walk_thread_count(column_walk_count, walkers_per_row);
    problem.row_walk_partial_count = reduction_group_count(row_thread_count, owning_backend->threads_per_block);
    problem.column_walk_partial_count =
        column_walk_count > 0 ? reduction_group_count(column_thread_count, owning_backend->threads_per_block) : 0;

    const Fit_Walk_Constants constants = fit_walk_constants(problem, 0.0f);
    const u32 first_row = 0;
    const u32 first_column = static_cast<u32>(problem.moved_source_count);
    const u32 walk_rows = 0;
    const u32 walk_columns = 1;
    const u32 every_node = rows_are_every_node ? 1u : 0u;
    const u32 listed_nodes = 0;
    const auto partials_after = [&](const Engine_Pointer &partials) {
        // The column walks' partials follow the most the row walks can write.
        const s64 skipped_bytes = reduction_group_count(FIT_REDUCTION_THREAD_LIMIT, owning_backend->threads_per_block) *
                                  static_cast<s64>(sizeof(f32));
        Engine_Pointer later = partials;
        later.offset += skipped_bytes;
        later.total_bytes -= static_cast<u64>(skipped_bytes);
        return later;
    };
    Vector<Engine_Pointer> row_parameters = fit_walk_arguments(problem, constants);
    row_parameters.insert(row_parameters.end(), {
        problem.working_u,                          // 12
        problem.working_v,                          // 13
        problem.moved_row_buffer,                   // 14
        inline_scalar_argument(first_row),          // 15
        inline_scalar_argument(row_walk_count),     // 16
        inline_scalar_argument(walk_rows),          // 17
        inline_scalar_argument(every_node),         // 18
        problem.source_move_buffer,                 // 19
        problem.first_partials,                     // 20
        problem.second_partials,                    // 21
    });
    if (run_fit_kernel("fit_walk_residual_kernel", row_parameters, row_thread_count)) {
        if (column_walk_count > 0) {
            Vector<Engine_Pointer> column_parameters = fit_walk_arguments(problem, constants);
            column_parameters.insert(column_parameters.end(), {
                problem.working_u,
                problem.working_v,
                problem.moved_row_buffer,
                inline_scalar_argument(first_column),
                inline_scalar_argument(column_walk_count),
                inline_scalar_argument(walk_columns),
                inline_scalar_argument(listed_nodes),
                problem.source_move_buffer,
                partials_after(problem.first_partials),
                partials_after(problem.second_partials),
            });
            if (!run_fit_kernel("fit_walk_residual_kernel", column_parameters, column_thread_count)) {
                log::throw_runtime_error(log::logger(),
                    "Weight_Matrix: fit_walk_residual_kernel ran for the row walks but not for the column walks");
            }
        }
        return collecting_batch != nullptr ? Edge_Evaluation{} : read_edge_evaluation(problem);
    }

    const Fit_Values &values = *problem.values;
    const s64 lane_count = static_cast<s64>(problem.own_lanes.size());
    const f32 *working_u = problem.working_u.get_contents_as<f32>();
    const f32 *working_v = problem.working_v.get_contents_as<f32>();
    const s64 walk_count = static_cast<s64>(row_walk_count) + column_walk_count;
    Vector<f64> walk_loss(static_cast<usize>(walk_count), 0.0);
    Vector<f64> walk_worst_error(static_cast<usize>(walk_count), 0.0);
    for_each_index_in_parallel(walk_count, [&](s64 walk) {
        const bool along_column = walk >= row_walk_count;
        const u32 node = along_column ? problem.moved_rows[static_cast<usize>(problem.moved_source_count + walk - row_walk_count)]
                         : rows_are_every_node ? static_cast<u32>(walk)
                                               : problem.moved_rows[static_cast<usize>(walk)];
        Vector<u32> buffer;
        for_each_connection_of(node, along_column, buffer, [&](u32 other_end) {
            const u32 source_node = along_column ? other_end : node;
            const u32 target_node = along_column ? node : other_end;
            if (along_column && problem.source_moves[source_node] != 0) return;
            const f32 target = fit_target(problem, source_node, target_node, fit_value_at(values, source_node, target_node));
            const f32 *source_row = working_u + static_cast<s64>(source_node) * lane_count;
            const f32 *target_row = working_v + static_cast<s64>(target_node) * lane_count;
            f32 value = 0.0f;
            for (s64 lane = 0; lane < lane_count; lane += 1) value += source_row[lane] * target_row[lane];
            const f32 residual = value - held_target(target);
            const f32 weight = edge_weight_of(target);
            f64 edge_loss = static_cast<f64>(0.5f * weight * residual * residual);
            f64 edge_error = static_cast<f64>(fabs(residual) * sqrt(weight));
            if (!isfinite(edge_loss) || !isfinite(edge_error)) {
                edge_loss = Numeric_Limits<f64>::infinity();
                edge_error = Numeric_Limits<f64>::infinity();
            }
            walk_loss[static_cast<usize>(walk)] += edge_loss;
            walk_worst_error[static_cast<usize>(walk)] = max(walk_worst_error[static_cast<usize>(walk)], edge_error);
        });
    });
    Edge_Evaluation evaluation;
    for (s64 walk = 0; walk < walk_count; walk += 1) {
        evaluation.loss += walk_loss[static_cast<usize>(walk)];
        evaluation.worst_error = max(evaluation.worst_error, walk_worst_error[static_cast<usize>(walk)]);
    }
    return evaluation;
}

Weight_Matrix::Edge_Evaluation Weight_Matrix::read_edge_evaluation(const Plane_Fit_Problem &problem) const {
    const s64 column_offset = reduction_group_count(FIT_REDUCTION_THREAD_LIMIT, owning_backend->threads_per_block);
    const f32 *loss = problem.first_partials.get_contents_as<f32>();
    const f32 *worst_error = problem.second_partials.get_contents_as<f32>();
    return Edge_Evaluation{
        sum_of(loss, problem.row_walk_partial_count) + sum_of(loss + column_offset, problem.column_walk_partial_count),
        max(largest_of(worst_error, problem.row_walk_partial_count),
            largest_of(worst_error + column_offset, problem.column_walk_partial_count))};
}

Weight_Matrix::Gradient_Evaluation Weight_Matrix::read_gradient_evaluation(const Plane_Fit_Problem &problem) const {
    const s64 thread_count = min(walk_group_thread_count(static_cast<s64>(problem.moved_rows.size()),
                                                         walkers_per_row_for(total_edge_count, node_count)),
                                 FIT_REDUCTION_THREAD_LIMIT);
    const s64 group_count = reduction_group_count(thread_count, owning_backend->threads_per_block);
    return Gradient_Evaluation{sum_of(problem.gradient_first_partials.get_contents_as<f32>(), group_count),
                               sum_of(problem.gradient_second_partials.get_contents_as<f32>(), group_count)};
}

void Weight_Matrix::solve_rows(const Plane_Fit_Problem &problem, bool solving_sources, f32 ridge_regularization,
                               const Engine_Pointer &triangle_scratch, const Engine_Pointer &vector_scratch,
                               s64 batch_row_count) const {
    const s64 moved_row_count = static_cast<s64>(problem.moved_rows.size());
    const s64 first_slot = solving_sources ? 0 : problem.moved_source_count;
    const s64 row_count = solving_sources ? problem.moved_source_count : moved_row_count - problem.moved_source_count;
    const u32 lane_count = static_cast<u32>(problem.own_lanes.size());
    const Engine_Pointer &solved_side = solving_sources ? problem.working_u : problem.working_v;
    const Engine_Pointer &fixed_side = solving_sources ? problem.working_v : problem.working_u;
    const u32 solving_targets = solving_sources ? 0u : 1u;
    const Fit_Walk_Constants constants = fit_walk_constants(problem, ridge_regularization);

    for (s64 batch_start = 0; batch_start < row_count; batch_start += batch_row_count) {
        const u32 first_row = static_cast<u32>(first_slot + batch_start);
        const u32 batch_count = static_cast<u32>(min(batch_row_count, row_count - batch_start));
        Vector<Engine_Pointer> parameters = fit_walk_arguments(problem, constants);
        parameters.insert(parameters.end(), {
            solved_side,                              // 12
            fixed_side,                               // 13
            problem.moved_row_buffer,                 // 14
            inline_scalar_argument(first_row),        // 15
            inline_scalar_argument(batch_count),      // 16
            inline_scalar_argument(solving_targets),  // 17
            triangle_scratch,                         // 18
            vector_scratch,                           // 19
        });
        if (run_fit_kernel("fit_walk_row_solve_kernel", parameters,
                           walk_group_thread_count(batch_count, constants.walkers_per_row))) continue;

        f32 *solved = solved_side.get_contents_as<f32>();
        const f32 *fixed = fixed_side.get_contents_as<f32>();
        const Fit_Values &values = *problem.values;
        // As fit_walk_row_solve_kernel: Givens rotations fold each connection's row into R, from
        // ridge^(1/2) I.
        for_each_index_in_parallel(batch_count, [&](s64 index) {
            const u32 row = problem.moved_rows[first_row + static_cast<usize>(index)];
            Vector<f32> triangle(static_cast<usize>(lane_count) * lane_count, 0.0f);
            Vector<f32> projected(lane_count, 0.0f);
            Vector<f32> edge_row(lane_count, 0.0f);
            for (u32 lane = 0; lane < lane_count; lane += 1) triangle[lane * lane_count + lane] = sqrt(ridge_regularization);

            Vector<u32> buffer;
            for_each_connection_of(row, !solving_sources, buffer, [&](u32 other_end) {
                const u32 source_node = solving_sources ? row : other_end;
                const u32 target_node = solving_sources ? other_end : row;
                const f32 target = fit_target(problem, source_node, target_node, fit_value_at(values, source_node, target_node));
                const f32 *other_row = fixed + static_cast<u64>(other_end) * lane_count;
                const f32 root_weight = sqrt(edge_weight_of(target));
                for (u32 lane = 0; lane < lane_count; lane += 1) edge_row[lane] = root_weight * other_row[lane];
                f32 edge_target = root_weight * held_target(target);
                for (u32 pivot = 0; pivot < lane_count; pivot += 1) {
                    const f32 incoming = edge_row[pivot];
                    if (incoming == 0.0f) continue;
                    f32 *pivot_row = triangle.data() + static_cast<usize>(pivot) * lane_count;
                    const f32 radius = sqrt(pivot_row[pivot] * pivot_row[pivot] + incoming * incoming);
                    const f32 cosine = pivot_row[pivot] / radius;
                    const f32 sine = incoming / radius;
                    pivot_row[pivot] = radius;
                    for (u32 column = pivot + 1; column < lane_count; column += 1) {
                        const f32 kept = pivot_row[column];
                        const f32 folded = edge_row[column];
                        pivot_row[column] = cosine * kept + sine * folded;
                        edge_row[column] = cosine * folded - sine * kept;
                    }
                    const f32 kept_target = projected[pivot];
                    projected[pivot] = cosine * kept_target + sine * edge_target;
                    edge_target = cosine * edge_target - sine * kept_target;
                }
            });

            for (u32 countdown = lane_count; countdown > 0; countdown -= 1) {
                const u32 first = countdown - 1;
                const f32 diagonal = triangle[first * lane_count + first];
                if (!(diagonal != 0.0f) || !isfinite(diagonal)) return;
                f32 sum = projected[first];
                for (u32 inner = first + 1; inner < lane_count; inner += 1) sum -= triangle[first * lane_count + inner] * projected[inner];
                projected[first] = sum / diagonal;
            }
            f32 *solved_row = solved + static_cast<u64>(row) * lane_count;
            for (u32 lane = 0; lane < lane_count; lane += 1) solved_row[lane] = projected[lane];
        });
    }
}

Weight_Matrix::Gradient_Evaluation Weight_Matrix::evaluate_gradient(
    const Plane_Fit_Problem &problem, f32 ridge_regularization, const Engine_Pointer &direction,
    const Engine_Pointer &gradient
) const {
    const u32 moved_source_count = static_cast<u32>(problem.moved_source_count);
    const u32 moved_row_count = static_cast<u32>(problem.moved_rows.size());
    const u32 lane_count = static_cast<u32>(problem.own_lanes.size());
    const Fit_Walk_Constants constants = fit_walk_constants(problem, ridge_regularization);
    Vector<Engine_Pointer> parameters = fit_walk_arguments(problem, constants);
    parameters.insert(parameters.end(), {
        problem.working_u,                            // 12
        problem.working_v,                            // 13
        problem.moved_row_buffer,                     // 14
        inline_scalar_argument(moved_source_count),   // 15
        inline_scalar_argument(moved_row_count),      // 16
        direction,                                    // 17
        gradient,                                     // 18
        problem.gradient_first_partials,              // 19
        problem.gradient_second_partials,             // 20
    });
    const s64 thread_count = min(walk_group_thread_count(moved_row_count, constants.walkers_per_row),
                                 FIT_REDUCTION_THREAD_LIMIT);
    if (run_fit_kernel("fit_walk_gradient_kernel", parameters, thread_count)) {
        return collecting_batch != nullptr ? Gradient_Evaluation{} : read_gradient_evaluation(problem);
    }

    const Fit_Values &values = *problem.values;
    const f32 *working_u = problem.working_u.get_contents_as<f32>();
    const f32 *working_v = problem.working_v.get_contents_as<f32>();
    const f32 *direction_values = direction.get_contents_as<f32>();
    f32 *gradient_values = gradient.get_contents_as<f32>();
    Vector<f64> slot_ridge_term(moved_row_count, 0.0);
    Vector<f64> slot_slope(moved_row_count, 0.0);
    for_each_index_in_parallel(moved_row_count, [&](s64 slot) {
        const u32 row = problem.moved_rows[static_cast<usize>(slot)];
        const bool along_column = slot >= moved_source_count;
        const f32 *own_side = along_column ? working_v : working_u;
        const f32 *other_side = along_column ? working_u : working_v;
        Vector<f32> sums(lane_count, 0.0f);
        Vector<u32> buffer;
        for_each_connection_of(row, along_column, buffer, [&](u32 other_end) {
            const u32 source_node = along_column ? other_end : row;
            const u32 target_node = along_column ? row : other_end;
            const f32 target = fit_target(problem, source_node, target_node, fit_value_at(values, source_node, target_node));
            const f32 *source_row = working_u + static_cast<u64>(source_node) * lane_count;
            const f32 *target_row = working_v + static_cast<u64>(target_node) * lane_count;
            f32 value = 0.0f;
            for (u32 lane = 0; lane < lane_count; lane += 1) value += source_row[lane] * target_row[lane];
            const f32 weighted_residual = edge_weight_of(target) * (value - held_target(target));
            const f32 *other_row = other_side + static_cast<u64>(other_end) * lane_count;
            for (u32 lane = 0; lane < lane_count; lane += 1) sums[lane] += weighted_residual * other_row[lane];
        });
        for (u32 lane = 0; lane < lane_count; lane += 1) {
            const u64 index = static_cast<u64>(slot) * lane_count + lane;
            const f32 row_value = own_side[static_cast<u64>(row) * lane_count + lane];
            const f32 sum = sums[lane] + ridge_regularization * row_value;
            gradient_values[index] = sum;
            slot_ridge_term[static_cast<usize>(slot)] += static_cast<f64>(0.5f * ridge_regularization * row_value * row_value);
            slot_slope[static_cast<usize>(slot)] += static_cast<f64>(sum * direction_values[index]);
        }
    });
    Gradient_Evaluation evaluation;
    for (u32 slot = 0; slot < moved_row_count; slot += 1) {
        evaluation.ridge_term += slot_ridge_term[slot];
        evaluation.slope += slot_slope[slot];
    }
    return evaluation;
}

void Weight_Matrix::balance_lanes(const Plane_Fit_Problem &problem) const {
    const u32 lane_count = static_cast<u32>(problem.own_lanes.size());
    if (lane_count == 0 || node_count == 0) return;
    const u32 node_count_argument = static_cast<u32>(node_count);
    const u32 chunk_count = static_cast<u32>(max<s64>(1, min<s64>(node_count, FIT_REDUCTION_THREAD_LIMIT / lane_count)));

    Vector<f64> u_square_sum(lane_count, 0.0);
    Vector<f64> v_square_sum(lane_count, 0.0);
    Vector<Engine_Pointer> sum_parameters = {
        problem.working_u, problem.working_v, inline_scalar_argument(node_count_argument),
        inline_scalar_argument(lane_count), inline_scalar_argument(chunk_count), problem.column_partials,
    };
    const f32 *working_u = problem.working_u.get_contents_as<f32>();
    const f32 *working_v = problem.working_v.get_contents_as<f32>();
    if (run_fit_kernel("fit_column_square_sums_kernel", sum_parameters, static_cast<s64>(chunk_count) * lane_count)) {
        const f32 *partials = problem.column_partials.get_contents_as<f32>();
        for (u32 chunk = 0; chunk < chunk_count; chunk += 1) {
            for (u32 lane = 0; lane < lane_count; lane += 1) {
                u_square_sum[lane] += static_cast<f64>(partials[(static_cast<u64>(chunk) * 2) * lane_count + lane]);
                v_square_sum[lane] += static_cast<f64>(partials[(static_cast<u64>(chunk) * 2 + 1) * lane_count + lane]);
            }
        }
    } else {
        for_each_index_in_parallel(lane_count, [&](s64 lane) {
            f32 u_sum = 0.0f;
            f32 v_sum = 0.0f;
            for (s64 node_index = 0; node_index < node_count; node_index += 1) {
                const f32 u_value = working_u[node_index * lane_count + lane];
                const f32 v_value = working_v[node_index * lane_count + lane];
                u_sum += u_value * u_value;
                v_sum += v_value * v_value;
            }
            u_square_sum[static_cast<usize>(lane)] = static_cast<f64>(u_sum);
            v_square_sum[static_cast<usize>(lane)] = static_cast<f64>(v_sum);
        });
    }

    // c = (|V| / |U|)^(1/2) makes |U| c = |V| / c. A lane with an empty side has nothing to balance.
    Vector<f32> factors(lane_count, 1.0f);
    bool any_lane_scales = false;
    for (u32 lane = 0; lane < lane_count; lane += 1) {
        if (!(u_square_sum[lane] > 0.0) || !(v_square_sum[lane] > 0.0)) continue;
        const f64 factor = pow(v_square_sum[lane] / u_square_sum[lane], 0.25);
        if (!isfinite(factor) || !(factor > 0.0)) continue;
        factors[lane] = static_cast<f32>(factor);
        any_lane_scales = any_lane_scales || factors[lane] != 1.0f;
    }
    if (!any_lane_scales) return;

    Vector<Engine_Pointer> scale_parameters = {
        problem.working_u, problem.working_v, inline_scalar_argument(node_count_argument),
        inline_scalar_argument(lane_count), inline_array_argument(factors),
    };
    const s64 length = node_count * static_cast<s64>(lane_count);
    if (run_fit_kernel("fit_scale_columns_kernel", scale_parameters, reduction_thread_count(length))) return;

    f32 *scaled_u = problem.working_u.get_contents_as<f32>();
    f32 *scaled_v = problem.working_v.get_contents_as<f32>();
    for_each_index_in_parallel(length, [&](s64 index) {
        const f32 factor = factors[static_cast<usize>(index % lane_count)];
        scaled_u[index] *= factor;
        scaled_v[index] /= factor;
    });
}

void Weight_Matrix::scatter_point(const Plane_Fit_Problem &problem, const Engine_Pointer &point,
                                  const Engine_Pointer &direction, f32 step_length) const {
    const u32 moved_source_count = static_cast<u32>(problem.moved_source_count);
    const u32 moved_row_count = static_cast<u32>(problem.moved_rows.size());
    const u32 lane_count = static_cast<u32>(problem.own_lanes.size());
    const s64 length = static_cast<s64>(moved_row_count) * lane_count;
    Vector<Engine_Pointer> parameters = {
        problem.working_u, problem.working_v, problem.moved_row_buffer, inline_scalar_argument(moved_source_count),
        inline_scalar_argument(moved_row_count), inline_scalar_argument(lane_count), point, direction,
        inline_scalar_argument(step_length),
    };
    if (run_fit_kernel("fit_scatter_point_kernel", parameters, reduction_thread_count(length))) return;

    f32 *working_u = problem.working_u.get_contents_as<f32>();
    f32 *working_v = problem.working_v.get_contents_as<f32>();
    const f32 *point_values = point.get_contents_as<f32>();
    const f32 *direction_values = direction.get_contents_as<f32>();
    for_each_index_in_parallel(length, [&](s64 index) {
        const u32 slot = static_cast<u32>(index / lane_count);
        const u32 lane = static_cast<u32>(index % lane_count);
        f32 *side = slot < moved_source_count ? working_u : working_v;
        const f32 offset = step_length != 0.0f ? step_length * direction_values[index] : 0.0f;
        side[static_cast<u64>(problem.moved_rows[slot]) * lane_count + lane] = point_values[index] + offset;
    });
}

void Weight_Matrix::scatter_point(const Plane_Fit_Problem &problem, const Engine_Pointer &point,
                                  const Engine_Pointer &direction, const Engine_Pointer &step_length) const {
    const u32 moved_source_count = static_cast<u32>(problem.moved_source_count);
    const u32 moved_row_count = static_cast<u32>(problem.moved_rows.size());
    const u32 lane_count = static_cast<u32>(problem.own_lanes.size());
    const s64 length = static_cast<s64>(moved_row_count) * lane_count;
    Vector<Engine_Pointer> parameters = {
        problem.working_u, problem.working_v, problem.moved_row_buffer, inline_scalar_argument(moved_source_count),
        inline_scalar_argument(moved_row_count), inline_scalar_argument(lane_count), point, direction, step_length,
    };
    if (run_fit_kernel("fit_scatter_point_kernel", parameters, reduction_thread_count(length))) return;
    scatter_point(problem, point, direction, *step_length.get_contents_as<f32>());
}

void Weight_Matrix::gather_point(const Plane_Fit_Problem &problem, const Engine_Pointer &point) const {
    const u32 moved_source_count = static_cast<u32>(problem.moved_source_count);
    const u32 moved_row_count = static_cast<u32>(problem.moved_rows.size());
    const u32 lane_count = static_cast<u32>(problem.own_lanes.size());
    const s64 length = static_cast<s64>(moved_row_count) * lane_count;
    Vector<Engine_Pointer> parameters = {
        problem.working_u, problem.working_v, problem.moved_row_buffer, inline_scalar_argument(moved_source_count),
        inline_scalar_argument(moved_row_count), inline_scalar_argument(lane_count), point,
    };
    if (run_fit_kernel("fit_gather_point_kernel", parameters, reduction_thread_count(length))) return;

    const f32 *working_u = problem.working_u.get_contents_as<f32>();
    const f32 *working_v = problem.working_v.get_contents_as<f32>();
    f32 *point_values = point.get_contents_as<f32>();
    for_each_index_in_parallel(length, [&](s64 index) {
        const u32 slot = static_cast<u32>(index / lane_count);
        const u32 lane = static_cast<u32>(index % lane_count);
        const f32 *side = slot < moved_source_count ? working_u : working_v;
        point_values[index] = side[static_cast<u64>(problem.moved_rows[slot]) * lane_count + lane];
    });
}

void Weight_Matrix::accept_step(const Plane_Fit_Problem &problem, const Engine_Pointer &point,
                                const Engine_Pointer &trial_gradient, const Engine_Pointer &basis, u32 step_slot,
                                u32 gradient_change_slot, u32 gradient_slot, bool record_history) const {
    const u32 moved_source_count = static_cast<u32>(problem.moved_source_count);
    const u32 moved_row_count = static_cast<u32>(problem.moved_rows.size());
    const u32 lane_count = static_cast<u32>(problem.own_lanes.size());
    const u32 record = record_history ? 1u : 0u;
    const s64 length = static_cast<s64>(moved_row_count) * lane_count;
    Vector<Engine_Pointer> parameters = {
        problem.working_u,                              // 0
        problem.working_v,                              // 1
        problem.moved_row_buffer,                       // 2
        inline_scalar_argument(moved_source_count),     // 3
        inline_scalar_argument(moved_row_count),        // 4
        inline_scalar_argument(lane_count),             // 5
        point,                                          // 6
        trial_gradient,                                 // 7
        basis,                                          // 8
        inline_scalar_argument(step_slot),              // 9
        inline_scalar_argument(gradient_change_slot),   // 10
        inline_scalar_argument(gradient_slot),          // 11
        inline_scalar_argument(record),                 // 12
    };
    if (run_fit_kernel("fit_accept_step_kernel", parameters, reduction_thread_count(length))) return;

    const f32 *working_u = problem.working_u.get_contents_as<f32>();
    const f32 *working_v = problem.working_v.get_contents_as<f32>();
    f32 *point_values = point.get_contents_as<f32>();
    const f32 *trial_gradient_values = trial_gradient.get_contents_as<f32>();
    f32 *basis_values = basis.get_contents_as<f32>();
    for_each_index_in_parallel(length, [&](s64 index) {
        const u32 slot = static_cast<u32>(index / lane_count);
        const u32 lane = static_cast<u32>(index % lane_count);
        const f32 *side = slot < moved_source_count ? working_u : working_v;
        const f32 accepted = side[static_cast<u64>(problem.moved_rows[slot]) * lane_count + lane];
        if (record_history) {
            basis_values[static_cast<s64>(step_slot) * length + index] = accepted - point_values[index];
            basis_values[static_cast<s64>(gradient_change_slot) * length + index] =
                trial_gradient_values[index] - basis_values[static_cast<s64>(gradient_slot) * length + index];
        }
        point_values[index] = accepted;
        basis_values[static_cast<s64>(gradient_slot) * length + index] = trial_gradient_values[index];
    });
}

void Weight_Matrix::update_dot_products(const Plane_Fit_Problem &problem, const Engine_Pointer &basis, u64 length,
                                        u32 vector_count, const Vector<u32> &left_slots,
                                        const Engine_Pointer &products) const {
    const u32 left_count = static_cast<u32>(left_slots.size());
    const u32 group_count = static_cast<u32>(reduction_group_count(static_cast<s64>(length), owning_backend->threads_per_block));
    Vector<Engine_Pointer> dot_parameters = {
        basis, inline_scalar_argument(length), inline_scalar_argument(vector_count), inline_array_argument(left_slots),
        inline_scalar_argument(left_count), problem.dot_partials,
    };
    if (run_fit_kernel("fit_dot_products_kernel", dot_parameters, reduction_thread_count(static_cast<s64>(length)))) {
        Vector<Engine_Pointer> reduce_parameters = {
            problem.dot_partials, inline_scalar_argument(group_count), inline_scalar_argument(vector_count),
            inline_array_argument(left_slots), inline_scalar_argument(left_count), products,
        };
        if (run_fit_kernel("fit_reduce_dot_products_kernel", reduce_parameters, static_cast<s64>(left_count) * vector_count)) {
            return;
        }
    }

    // The host: as the two kernels, with the sums in f64.
    const f32 *basis_values = basis.get_contents_as<f32>();
    const s64 chunk_count = chunk_count_for(static_cast<s64>(length));
    Vector<Vector<f64>> chunk_sums(static_cast<usize>(chunk_count), Vector<f64>(static_cast<usize>(left_count) * vector_count, 0.0));
    for_each_index_in_parallel(chunk_count, [&](s64 chunk) {
        const u64 last = min<u64>(static_cast<u64>(chunk + 1) * HOST_REDUCTION_CHUNK, length);
        for (u64 index = static_cast<u64>(chunk) * HOST_REDUCTION_CHUNK; index < last; index += 1) {
            for (u32 left = 0; left < left_count; left += 1) {
                const f32 left_value = basis_values[static_cast<u64>(left_slots[left]) * length + index];
                for (u32 slot = 0; slot < vector_count; slot += 1) {
                    chunk_sums[static_cast<usize>(chunk)][left * vector_count + slot] +=
                        static_cast<f64>(left_value * basis_values[static_cast<u64>(slot) * length + index]);
                }
            }
        }
    });
    f32 *product_values = products.get_contents_as<f32>();
    for (u32 left = 0; left < left_count; left += 1) {
        for (u32 slot = 0; slot < vector_count; slot += 1) {
            f64 sum = 0.0;
            for (const Vector<f64> &sums : chunk_sums) sum += sums[left * vector_count + slot];
            product_values[left_slots[left] * vector_count + slot] = static_cast<f32>(sum);
            if (find(left_slots.begin(), left_slots.end(), slot) == left_slots.end()) {
                product_values[slot * vector_count + left_slots[left]] = static_cast<f32>(sum);
            }
        }
    }
}

void Weight_Matrix::quasi_newton_direction(const Engine_Pointer &products, const Engine_Pointer &history,
                                           u32 history_length, u32 new_pair_slot, bool reset_history,
                                           const Engine_Pointer &coefficients, const Engine_Pointer &status,
                                           const Engine_Pointer &step_length) const {
    const u32 reset = reset_history ? 1u : 0u;
    Vector<Engine_Pointer> parameters = {
        products, history, inline_scalar_argument(history_length), inline_scalar_argument(new_pair_slot),
        inline_scalar_argument(reset), coefficients, status, step_length,
    };
    if (run_fit_kernel("fit_quasi_newton_direction_kernel", parameters, 1)) return;

    // The host: as the kernel, line for line.
    const f32 *product_values = products.get_contents_as<f32>();
    u32 *history_values = history.get_contents_as<u32>();
    f32 *coefficient_values = coefficients.get_contents_as<f32>();
    f32 *status_values = status.get_contents_as<f32>();
    const u32 vector_count = 2 * history_length + 1;
    const u32 gradient_slot = 2 * history_length;
    auto product = [&](u32 first, u32 second) { return product_values[first * vector_count + second]; };

    u32 count = reset_history ? 0u : history_values[0];
    if (new_pair_slot < history_length) {
        if (count == history_length) {
            for (u32 index = 1; index < count; index += 1) history_values[index] = history_values[index + 1];
            count -= 1;
        }
        const f32 curvature = product(new_pair_slot, history_length + new_pair_slot);
        const f32 change_norm = product(history_length + new_pair_slot, history_length + new_pair_slot);
        if (isfinite(curvature) && curvature > 1e-10f * change_norm) {
            history_values[1 + count] = new_pair_slot;
            count += 1;
        }
    }
    history_values[0] = count;

    fill(coefficient_values, coefficient_values + vector_count, 0.0f);
    coefficient_values[gradient_slot] = 1.0f;
    Vector<f32> alpha(history_length, 0.0f);
    for (u32 countdown = count; countdown > 0; countdown -= 1) {
        const u32 step_slot = history_values[countdown];
        const u32 change_slot = history_length + step_slot;
        f32 dot_product = 0.0f;
        for (u32 slot = 0; slot < vector_count; slot += 1) dot_product += coefficient_values[slot] * product(step_slot, slot);
        alpha[countdown - 1] = dot_product / product(step_slot, change_slot);
        coefficient_values[change_slot] -= alpha[countdown - 1];
    }
    if (count > 0) {
        const u32 newest = history_values[count];
        const f32 gamma = product(newest, history_length + newest) / product(history_length + newest, history_length + newest);
        for (u32 slot = 0; slot < vector_count; slot += 1) coefficient_values[slot] *= gamma;
    }
    for (u32 index = 0; index < count; index += 1) {
        const u32 step_slot = history_values[index + 1];
        const u32 change_slot = history_length + step_slot;
        f32 dot_product = 0.0f;
        for (u32 slot = 0; slot < vector_count; slot += 1) dot_product += coefficient_values[slot] * product(change_slot, slot);
        coefficient_values[step_slot] += alpha[index] - dot_product / product(step_slot, change_slot);
    }

    f32 slope = 0.0f;
    for (u32 slot = 0; slot < vector_count; slot += 1) slope -= coefficient_values[slot] * product(gradient_slot, slot);
    bool steepest = count == 0;
    if (!(slope < 0.0f) || !isfinite(slope)) {
        history_values[0] = 0;
        fill(coefficient_values, coefficient_values + vector_count, 0.0f);
        coefficient_values[gradient_slot] = 1.0f;
        slope = -product(gradient_slot, gradient_slot);
        steepest = true;
    }
    for (u32 slot = 0; slot < vector_count; slot += 1) coefficient_values[slot] = -coefficient_values[slot];
    const f32 first_step = steepest ? min(1.0f, 1.0f / sqrt(max(product(gradient_slot, gradient_slot), 1e-30f))) : 1.0f;
    status_values[0] = slope;
    status_values[1] = first_step;
    status_values[2] = steepest ? 1.0f : 0.0f;
    status_values[3] = slope < 0.0f ? 0.0f : 1.0f;
    *step_length.get_contents_as<f32>() = first_step;
}

void Weight_Matrix::linear_combination(const Engine_Pointer &basis, u64 length, u32 vector_count,
                                       const Engine_Pointer &coefficients, const Engine_Pointer &output) const {
    Vector<Engine_Pointer> parameters = {
        basis, inline_scalar_argument(length), inline_scalar_argument(vector_count), coefficients, output,
    };
    if (run_fit_kernel("fit_linear_combination_kernel", parameters, reduction_thread_count(static_cast<s64>(length)))) {
        return;
    }

    const f32 *basis_values = basis.get_contents_as<f32>();
    const f32 *coefficient_values = coefficients.get_contents_as<f32>();
    f32 *output_values = output.get_contents_as<f32>();
    for_each_index_in_parallel(static_cast<s64>(length), [&](s64 index) {
        f32 sum = 0.0f;
        for (u32 slot = 0; slot < vector_count; slot += 1) {
            sum += coefficient_values[slot] * basis_values[static_cast<u64>(slot) * length + static_cast<u64>(index)];
        }
        output_values[index] = sum;
    });
}

void Weight_Matrix::record_fit_error(s64 matrix_index, f64 worst_error) {
    if (static_cast<s64>(measured_fit_error.size()) != matrix_count) measured_fit_error.assign(matrix_count, 0.0f);
    measured_fit_error[static_cast<usize>(matrix_index)] = static_cast<f32>(worst_error);
}

f32 Weight_Matrix::worst_fit_error() const {
    // std::max drops a NaN, so a NaN error counts as infinite.
    f32 worst = 0.0f;
    for (f32 error : measured_fit_error) worst = isnan(error) ? Numeric_Limits<f32>::infinity() : max(worst, error);
    return worst;
}

void Weight_Matrix::neighbor_weights(f32 *output_weights) const {
    neighbor_weights_for_matrix(output_weights, WEIGHT_PLANE);
}

void Weight_Matrix::neighbor_weights_for_matrix(f32 *output_weights, s64 matrix_index) const {
    validate_matrix_index(matrix_index);
    if (total_edge_count == 0) return;

    bool reconstructed_on_gpu = ensure_function(neighbor_weights_function, "neighbor_weights_kernel");
    if (reconstructed_on_gpu) {
        // The kernel's output, one value per edge, held only for this read.
        const Fit_Scratch output(*owning_backend, {static_cast<u64>(total_edge_count) * sizeof(f32)});
        const Engine_Pointer &neighbor_weight_output = output.buffers[0];
        const s32 branching_factor = k2tree.branching_factor;
        const s32 superblock_size_words = k2tree.superblock_size_words;
        const s32 padded_node_count = k2tree.padded_node_count;
        const s32 tree_height = k2tree.tree_height;
        const s32 internal_bit_count = k2tree.internal_bit_count;
        const s64 node_count_argument = node_count;
        const s64 total_edge_count_argument = total_edge_count;
        const s64 rank_float4_stride_argument = rank_float4_stride;

        Vector<Engine_Pointer> parameters = {
            U_matrix,                                       // 0
            V_matrix,                                       // 1
            k2tree.internal_node_words,                     // 2
            k2tree.leaf_node_words,                         // 3
            k2tree.rank_superblock_table,                   // 4
            k2tree.rank_subblock_table,                     // 5
            inline_scalar_argument(branching_factor),       // 6
            inline_scalar_argument(superblock_size_words),  // 7
            inline_scalar_argument(padded_node_count),      // 8
            inline_scalar_argument(tree_height),            // 9
            inline_scalar_argument(internal_bit_count),     // 10
            inline_scalar_argument(node_count_argument),    // 11
            inline_scalar_argument(total_edge_count_argument), // 12
            inline_scalar_argument(rank_float4_stride_argument), // 13
            coefficient_range(matrix_index),                // 14
            edge_row_offset,                                // 15
            neighbor_weight_output,                         // 16
        };

        if (owning_backend->run_function(neighbor_weights_function, parameters, total_edge_count)) {
            owning_backend->prefetch_to_cpu(neighbor_weight_output,
                                            static_cast<u64>(total_edge_count) * sizeof(f32));
            memcpy(output_weights, neighbor_weight_output.get_contents(),
                   static_cast<usize>(total_edge_count) * sizeof(f32));
            reconstructed_on_gpu = true;
        } else {
            log::logger().warn("neighbor_weights: the GPU dispatch failed; answering on the host");
        }
    }

    if (!reconstructed_on_gpu) {
        const f32 *coefficient_values = coefficient_row(matrix_index);
        Vector<u32> neighbor_buffer((max<s64>(max_neighbor_count, 1)));
        for (s64 source_node = 0; source_node < node_count; ++source_node) {
            const s64 degree = k2tree.get_neighbors(static_cast<u32>(source_node), neighbor_buffer.data(),
                                                    max_neighbor_count);
            for (s64 slot = 0; slot < degree; ++slot) {
                const s64 ordinal = edge_row_offset_host[static_cast<usize>(source_node)] + slot;
                output_weights[static_cast<usize>(ordinal)] = reconstruct_entry(
                        static_cast<s32>(source_node), static_cast<s32>(neighbor_buffer[static_cast<usize>(slot)]),
                        coefficient_values);
            }
        }
    }

    // Like every read, the recent updates in S are added on top of the basis.
    const Vector<Vector<Pair<s64, f32>>> deltas = current_deltas();
    for (const Pair<s64, f32> &delta : deltas[static_cast<usize>(matrix_index)]) {
        output_weights[static_cast<usize>(delta.first)] += delta.second;
    }
}

Weight_Stats Weight_Matrix::neighbor_weight_stats() const {
    if (total_edge_count == 0) return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    Vector<f32> weight_buffer(total_edge_count);
    neighbor_weights(weight_buffer.data());

    // In f64, with the variance taken about the mean: E[x^2] - mean^2 cancels for tight values.
    f64 weight_sum = 0.0;
    f64 sum_of_squares = 0.0;
    f32 min_weight = weight_buffer[0];
    f32 max_weight = weight_buffer[0];
    for (s64 edge_index = 0; edge_index < total_edge_count; ++edge_index) {
        const f64 weight = static_cast<f64>(weight_buffer[static_cast<usize>(edge_index)]);
        weight_sum += weight;
        sum_of_squares += weight * weight;
        min_weight = min(min_weight, weight_buffer[static_cast<usize>(edge_index)]);
        max_weight = max(max_weight, weight_buffer[static_cast<usize>(edge_index)]);
    }

    const f64 edge_count = static_cast<f64>(total_edge_count);
    const f64 mean = weight_sum / edge_count;
    f64 sum_of_squared_deviations = 0.0;
    for (f32 weight : weight_buffer) {
        const f64 deviation = static_cast<f64>(weight) - mean;
        sum_of_squared_deviations += deviation * deviation;
    }
    return {static_cast<f32>(mean),
            static_cast<f32>(sqrt(sum_of_squared_deviations / edge_count)),
            static_cast<f32>(sqrt(sum_of_squares / edge_count)),
            min_weight,
            max_weight};
}

Scale_Result Weight_Matrix::scale_neighbor_weights_to_root_mean_square(
    f32 target_root_mean_square, f32 epsilon
) const {
    if (target_root_mean_square < 0.0f) {
        log::throw_invalid_argument(log::logger(),
            "scale_neighbor_weights_to_root_mean_square: target must be non-negative");
    }

    const Weight_Stats stats_before = neighbor_weight_stats();
    const f32 current_root_mean_square = max(stats_before.root_mean_square, epsilon);
    const f32 scale_factor = target_root_mean_square / current_root_mean_square;

    // Only the weight plane: its coefficient row and its updates. U and V are shared with every
    // other plane, so scaling them would rescale delays and synapse state too.
    f32 *weight_coefficients = coefficient_row(WEIGHT_PLANE);
    for (s64 lane = 0; lane < rank_float4_stride * LANE_GROUP; lane += 1) {
        weight_coefficients[lane] *= scale_factor;
    }
    const s64 weight_entry_count = sparse_delta_entry_count[static_cast<usize>(WEIGHT_PLANE)];
    if (weight_entry_count > 0) {
        f32 *weight_deltas = sparse_delta_value.get_contents_as<f32>() +
                             sparse_delta_row_start.get_contents_as<s32>()[WEIGHT_PLANE * (node_count + 1)];
        for (s64 index = 0; index < weight_entry_count; ++index) weight_deltas[index] *= scale_factor;
    }

    const Weight_Stats stats_after = neighbor_weight_stats();
    log::logger().debug("scale_neighbor_weights_to_root_mean_square: target={} factor={} "
                        "rms_before={} rms_after={}",
                        target_root_mean_square, scale_factor,
                        stats_before.root_mean_square, stats_after.root_mean_square);

    return {target_root_mean_square, scale_factor, stats_before, stats_after};
}

void Weight_Matrix::save(const char *filepath) const {
    Output_File_Stream file(filepath, Stream_Base::binary);
    file.write(reinterpret_cast<const char *>(&WEIGHT_MATRIX_SAVE_MAGIC), sizeof(u32));
    file.write(reinterpret_cast<const char *>(&node_count), sizeof(s64));
    file.write(reinterpret_cast<const char *>(&total_edge_count), sizeof(s64));
    file.write(reinterpret_cast<const char *>(&matrix_count), sizeof(s64));
    file.write(reinterpret_cast<const char *>(&rank), sizeof(s64));
    file.write(reinterpret_cast<const char *>(&rank_float4_stride), sizeof(s64));

    const s64 basis_bytes = node_count * rank_float4_stride * static_cast<s64>(sizeof(float4));
    file.write(reinterpret_cast<const char *>(U_matrix.get_contents()), basis_bytes);
    file.write(reinterpret_cast<const char *>(V_matrix.get_contents()), basis_bytes);

    const s64 coefficient_bytes = matrix_count * rank_float4_stride * LANE_GROUP * static_cast<s64>(sizeof(f32));
    file.write(reinterpret_cast<const char *>(coefficients.get_contents()), coefficient_bytes);

    // The updates not yet folded into the basis, one plane at a time.
    file.write(reinterpret_cast<const char *>(&sparse_delta_capacity), sizeof(s64));
    const Vector<Vector<Pair<s64, f32>>> deltas = current_deltas();
    for (const Vector<Pair<s64, f32>> &plane : deltas) {
        const s64 entry_count = static_cast<s64>(plane.size());
        file.write(reinterpret_cast<const char *>(&entry_count), sizeof(s64));
        for (const Pair<s64, f32> &delta : plane) {
            file.write(reinterpret_cast<const char *>(&delta.first), sizeof(s64));
            file.write(reinterpret_cast<const char *>(&delta.second), sizeof(f32));
        }
    }

    log::logger().debug("Weight_Matrix::save: {} node_count={} rank={}", filepath, node_count, rank);
}

void Weight_Matrix::load_from_disk(const char *filepath) {
    Input_File_Stream file(filepath, Stream_Base::binary);
    u32 magic = 0;
    file.read(reinterpret_cast<char *>(&magic), sizeof(u32));
    if (magic != WEIGHT_MATRIX_SAVE_MAGIC) {
        log::throw_invalid_argument(log::logger(),
            String("Weight_Matrix::load_from_disk: ") + filepath + " is not a weight matrix file this version can read");
    }

    s64 saved_node_count = 0;
    s64 saved_edge_count = 0;
    s64 saved_matrix_count = 0;
    s64 saved_rank = 0;
    s64 saved_rank_float4_stride = 0;
    file.read(reinterpret_cast<char *>(&saved_node_count), sizeof(s64));
    file.read(reinterpret_cast<char *>(&saved_edge_count), sizeof(s64));
    file.read(reinterpret_cast<char *>(&saved_matrix_count), sizeof(s64));
    file.read(reinterpret_cast<char *>(&saved_rank), sizeof(s64));
    file.read(reinterpret_cast<char *>(&saved_rank_float4_stride), sizeof(s64));

    const String described_file = String("Weight_Matrix::load_from_disk: ") + filepath;
    if (!file) log::throw_invalid_argument(log::logger(), described_file + " ends inside its header");
    if (saved_node_count != node_count || saved_edge_count != total_edge_count || saved_matrix_count != matrix_count) {
        log::throw_invalid_argument(log::logger(),
            described_file + " holds " + to_string(saved_node_count) + " nodes, " + to_string(saved_edge_count) +
            " edges and " + to_string(saved_matrix_count) + " planes but this matrix has " + to_string(node_count) +
            ", " + to_string(total_edge_count) + " and " + to_string(matrix_count) +
            " -- the adjacency and planes have to match");
    }
    if (saved_rank_float4_stride < 1 || saved_rank_float4_stride > MAX_RANK_FLOAT4_STRIDE ||
        saved_rank != saved_rank_float4_stride * LANE_GROUP) {
        log::throw_invalid_argument(log::logger(), described_file + " gives an invalid rank " + to_string(saved_rank));
    }

    // Everything is read and checked before the matrix changes, so a bad file leaves it as it was.
    const usize basis_values = static_cast<usize>(node_count * saved_rank);
    Vector<f32> saved_u(basis_values);
    Vector<f32> saved_v(basis_values);
    Vector<f32> saved_coefficients(static_cast<usize>(matrix_count * saved_rank));
    file.read(reinterpret_cast<char *>(saved_u.data()), static_cast<streamsize>(basis_values * sizeof(f32)));
    file.read(reinterpret_cast<char *>(saved_v.data()), static_cast<streamsize>(basis_values * sizeof(f32)));
    file.read(reinterpret_cast<char *>(saved_coefficients.data()),
              static_cast<streamsize>(saved_coefficients.size() * sizeof(f32)));

    s64 saved_delta_capacity = 0;
    file.read(reinterpret_cast<char *>(&saved_delta_capacity), sizeof(s64));
    Vector<Vector<Pair<s64, f32>>> saved_deltas(static_cast<usize>(matrix_count));
    s64 held_entry_count = 0;
    for (Vector<Pair<s64, f32>> &plane : saved_deltas) {
        s64 entry_count = 0;
        file.read(reinterpret_cast<char *>(&entry_count), sizeof(s64));
        if (!file || entry_count < 0 || entry_count > total_edge_count) {
            log::throw_invalid_argument(log::logger(), described_file + " holds a plane of S with " +
                                        to_string(entry_count) + " updates over " + to_string(total_edge_count) + " edges");
        }
        plane.resize(static_cast<usize>(entry_count));
        for (Pair<s64, f32> &delta : plane) {
            file.read(reinterpret_cast<char *>(&delta.first), sizeof(s64));
            file.read(reinterpret_cast<char *>(&delta.second), sizeof(f32));
            if (file && delta.first >= 0 && delta.first < total_edge_count) continue;
            log::throw_invalid_argument(log::logger(), described_file + " holds an update on edge " +
                                        to_string(delta.first) + ", outside this matrix's " +
                                        to_string(total_edge_count) + " edges");
        }
        held_entry_count += entry_count;
    }
    if (!file) log::throw_invalid_argument(log::logger(), described_file + " is truncated");

    resize_basis(saved_rank);
    memcpy(U_matrix.get_contents(), saved_u.data(), basis_values * sizeof(f32));
    memcpy(V_matrix.get_contents(), saved_v.data(), basis_values * sizeof(f32));
    memcpy(coefficients.get_contents(), saved_coefficients.data(), saved_coefficients.size() * sizeof(f32));

    clear_sparse_deltas();
    const s64 capacity_limit = matrix_count * total_edge_count + plasticity_reserve_entries;
    resize_delta_capacity(max(held_entry_count, min(max<s64>(saved_delta_capacity, 0), capacity_limit)));
    rebuild_sparse_deltas(saved_deltas);

    log::logger().debug("Weight_Matrix::load_from_disk: {} node_count={} rank={}", filepath, node_count, rank);
}

} // namespace spikecorec
