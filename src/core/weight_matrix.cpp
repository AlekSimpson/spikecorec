//
// Created by Alek Simpson on 5/30/26.
//

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;

namespace {

// Bumped when the deltas joined the file: an older file holds no S and cannot be read as one.
constexpr u32 WEIGHT_MATRIX_SAVE_MAGIC = 0x574D5459;

// Runs body(index) for every index in [0, count) across the machine's cores. Each body must write
// only what belongs to its own index.
template <typename Body>
void for_each_index_in_parallel(s64 count, const Body &body) {
    constexpr s64 INDICES_PER_CLAIM = 8;
    const s64 worker_count = min<s64>(max<s64>((s64)thread::hardware_concurrency(), 1),
                                      (count + INDICES_PER_CLAIM - 1) / INDICES_PER_CLAIM);
    if (worker_count <= 1) {
        for (s64 index = 0; index < count; index += 1) body(index);
        return;
    }

    atomic<s64> next_index{0};
    auto work = [&]() {
        for (s64 first = next_index.fetch_add(INDICES_PER_CLAIM); first < count;
             first = next_index.fetch_add(INDICES_PER_CLAIM)) {
            const s64 last = min(first + INDICES_PER_CLAIM, count);
            for (s64 index = first; index < last; index += 1) body(index);
        }
    };
    vector<thread> workers;
    for (s64 worker = 1; worker < worker_count; worker += 1) workers.emplace_back(work);
    work();
    for (thread &worker : workers) worker.join();
}

s64 round_up_to_lane_group(s64 value) {
    const s64 group = WeightMatrix::LANE_GROUP;
    return ((max<s64>(value, 1) + group - 1) / group) * group;
}

// 0 for an empty or all-zero sample.
template <typename Value>
f64 root_mean_square(const vector<Value> &values) {
    if (values.empty()) return 0.0;
    f64 sum_of_squares = 0.0;
    for (Value value : values) sum_of_squares += (f64)value * (f64)value;
    return sqrt(sum_of_squares / (f64)values.size());
}

f32 field_scale_of(const Vector<f32> &targets) {
    if (targets.empty()) return 1.0f;
    f64 sum_of_squares = 0.0;
    for (f32 value : targets) sum_of_squares += (f64)value * (f64)value;
    const f64 root_mean_square = sqrt(sum_of_squares / (f64)targets.size());
    return (root_mean_square > 0.0) ? (f32)root_mean_square : 1.0f;
}

bool solve_symmetric_in_place(vector<f64> &gram, vector<f64> &right_hand_side, s64 dimension,
                              f64 ridge_regularization) {
    for (s64 index = 0; index < dimension; index += 1) {
        gram[(usize)(index * dimension + index)] += ridge_regularization;
    }

    // Cholesky: gram = L * L^T, computed in the lower triangle.
    for (s64 row = 0; row < dimension; row += 1) {
        for (s64 column = 0; column <= row; column += 1) {
            f64 sum = gram[(usize)(row * dimension + column)];
            for (s64 index = 0; index < column; index += 1) {
                sum -= gram[(usize)(row * dimension + index)] *
                       gram[(usize)(column * dimension + index)];
            }
            if (row == column) {
                if (sum <= 0.0) return false;
                gram[(usize)(row * dimension + column)] = sqrt(sum);
            } else {
                gram[(usize)(row * dimension + column)] =
                        sum / gram[(usize)(column * dimension + column)];
            }
        }
    }

    for (s64 row = 0; row < dimension; row += 1) {
        f64 sum = right_hand_side[(usize)row];
        for (s64 index = 0; index < row; index += 1) {
            sum -= gram[(usize)(row * dimension + index)] * right_hand_side[(usize)index];
        }
        right_hand_side[(usize)row] = sum / gram[(usize)(row * dimension + row)];
    }
    for (s64 row = dimension - 1; row >= 0; row -= 1) {
        f64 sum = right_hand_side[(usize)row];
        for (s64 index = row + 1; index < dimension; index += 1) {
            sum -= gram[(usize)(index * dimension + row)] * right_hand_side[(usize)index];
        }
        right_hand_side[(usize)row] = sum / gram[(usize)(row * dimension + row)];
    }
    return true;
}

} // namespace

namespace spikecorec {

bool can_safely_cast_s64_to_s32(s64 value) {
    return value >= (s64)numeric_limits<s32>::min() && value <= (s64)numeric_limits<s32>::max();
}

const vector<vector<s32>> &WeightMatrix::validate_network(const vector<vector<s32>> &network) {
    if (network.empty()) {
        log::throw_invalid_argument(log::logger(),
            "WeightMatrix: network must have at least one neuron (got 0)");
    }
    return network;
}

WeightMatrix::WeightMatrix(
    EngineBackend &backend,
    const vector<vector<s32>> &network,
    s64 rank,
    bool check_indexing,
    s64 max_neighbor_count,
    s64 weight_seed,
    s64 matrix_count,
    s64 updated_plane_count
)
    : k2tree(*K2Tree::from_adjacency_list(backend, validate_network(network), (s32)network.size()))
    , matrix_count(matrix_count)
    , updated_plane_count(updated_plane_count)
    , sparse_delta_capacity(0)
    , owning_backend(&backend)
    , node_count((s64)network.size())
    , check_indexing(check_indexing)
{
    if (matrix_count < FIRST_STATE_VARIABLE_PLANE) {
        log::throw_invalid_argument(log::logger(),
            "WeightMatrix: matrix_count " + to_string(matrix_count) + " leaves no room for the weight "
            "and delay planes");
    }
    if (updated_plane_count < 0 || updated_plane_count > matrix_count) {
        log::throw_invalid_argument(log::logger(),
            "WeightMatrix: updated_plane_count " + to_string(updated_plane_count) + " is not between 0 and the " +
            to_string(matrix_count) + " planes");
    }

    this->max_neighbor_count = max_neighbor_count;
    if (max_neighbor_count < 0) {
        s64 longest_row = 0;
        for (const vector<s32> &row : network) longest_row = max(longest_row, (s64)row.size());
        this->max_neighbor_count = longest_row;
    }

    vector<s64> in_degree((usize)node_count, 0);
    for (const vector<s32> &row : network) {
        for (s32 target : row) in_degree[(usize)target] += 1;
    }
    max_predecessor_count = in_degree.empty() ? 0 : *max_element(in_degree.begin(), in_degree.end());

    this->rank = (rank > 0) ? round_up_to_lane_group(rank) : LANE_GROUP;
    rank_float4_stride = this->rank / LANE_GROUP;

    if (rank_float4_stride > MAX_RANK_FLOAT4_STRIDE) {
        log::throw_invalid_argument(log::logger(),
            "WeightMatrix: rank " + to_string(this->rank) + " exceeds the kernel limit of " +
            to_string((s64)MAX_RANK_FLOAT4_STRIDE * LANE_GROUP) + " lanes");
    }

    build_edge_row_offset();
    allocate_storage();

    basis_seed = (weight_seed >= 0) ? (unsigned)weight_seed : random_device{}();
    seed_basis(basis_seed);

    log::logger().debug("WeightMatrix constructed: node_count={} edges={} rank={} max_neighbor_count={}",
                        node_count, total_edge_count, this->rank, this->max_neighbor_count);
}

void WeightMatrix::build_edge_row_offset() {
    edge_row_offset_host.assign((usize)node_count + 1, 0);

    vector<s32> neighbor_buffer((usize)max<s64>(max_neighbor_count, 1));
    s64 running_ordinal = 0;
    for (s64 node_index = 0; node_index < node_count; node_index += 1) {
        edge_row_offset_host[(usize)node_index] = running_ordinal;
        running_ordinal += k2tree.get_neighbors((s32)node_index, neighbor_buffer.data(),
                                                max_neighbor_count);
    }
    edge_row_offset_host[(usize)node_count] = running_ordinal;
    total_edge_count = running_ordinal;
}

void WeightMatrix::allocate_storage() {
    const u64 matrix_byte_size = (u64)node_count * (u64)rank_float4_stride * sizeof(float4);
    const u64 lane_count = (u64)rank_float4_stride * (u64)LANE_GROUP;
    // In one tick an edge with no entry in S can queue both its Euler step and its OnEvent
    // change in each plane the simulation updates.
    pending_delta_capacity = plasticity_reserve_entries + 2 * total_edge_count * updated_plane_count;

    Vector<EnginePointer> partitions;
    owning_backend
        ->partition(matrix_byte_size, EngineDatatype::FLOAT32X4, partitions)                 // U
        .partition(matrix_byte_size, EngineDatatype::FLOAT32X4, partitions)                  // V
        .partition((u64)matrix_count * lane_count * sizeof(f32), EngineDatatype::FLOAT32, partitions)
        .partition((u64)(node_count + 1) * sizeof(s64), EngineDatatype::SIGNED64, partitions)
        .partition((u64)total_edge_count * sizeof(f32), EngineDatatype::FLOAT32, partitions)
        .partition((u64)matrix_count * (u64)(node_count + 1) * sizeof(s32),
                   EngineDatatype::SIGNED32, partitions)
        .partition((u64)matrix_count * (u64)sparse_delta_capacity * sizeof(s64),
                   EngineDatatype::SIGNED64, partitions)
        .partition((u64)matrix_count * (u64)sparse_delta_capacity * sizeof(f32),
                   EngineDatatype::FLOAT32, partitions)
        .partition((u64)pending_delta_capacity * sizeof(s64), EngineDatatype::SIGNED64, partitions)
        .partition((u64)pending_delta_capacity * sizeof(f32), EngineDatatype::FLOAT32, partitions)
        .partition((u64)pending_delta_capacity * sizeof(s32), EngineDatatype::SIGNED32, partitions)
        .partition(pending_delta_capacity > 0 ? sizeof(s32) : 0,
                   EngineDatatype::SIGNED32, partitions);

    owning_slab = owning_backend->allocate(partitions);

    U_matrix = partitions[0];
    V_matrix = partitions[1];
    coefficients = partitions[2];
    edge_row_offset = partitions[3];
    neighbor_weight_scratch = partitions[4];
    sparse_delta_row_start = partitions[5];
    sparse_delta_edge_ordinal = partitions[6];
    sparse_delta_value = partitions[7];
    pending_delta_edge_ordinal = partitions[8];
    pending_delta_value = partitions[9];
    pending_delta_matrix_index = partitions[10];
    pending_delta_count = partitions[11];

    memcpy(edge_row_offset.get_contents(), edge_row_offset_host.data(),
           ((usize)node_count + 1) * sizeof(s64));

    if (!sparse_delta_row_start.is_empty()) {
        memset(sparse_delta_row_start.get_contents(), 0,
               (usize)matrix_count * ((usize)node_count + 1) * sizeof(s32));
    }
    sparse_delta_entry_count.assign((usize)matrix_count, 0);
    if (pending_delta_capacity > 0) {
        *pending_delta_count.get_contents_as<s32>() = 0;
    }

    owning_backend->advise_read_mostly(edge_row_offset, (u64)(node_count + 1) * sizeof(s64));
    owning_backend->prefetch_to_gpu(U_matrix, matrix_byte_size);
    owning_backend->prefetch_to_gpu(V_matrix, matrix_byte_size);
}

void WeightMatrix::resize_basis(s64 new_rank) {
    const s64 rounded_rank = round_up_to_lane_group(new_rank);
    if (rounded_rank == rank) return;

    if (rounded_rank / LANE_GROUP > MAX_RANK_FLOAT4_STRIDE) {
        log::throw_invalid_argument(log::logger(),
            "WeightMatrix: rank " + to_string(rounded_rank) + " exceeds the kernel limit of " +
            to_string((s64)MAX_RANK_FLOAT4_STRIDE * LANE_GROUP) + " lanes");
    }

    owning_backend->deallocate_slab(owning_slab);
    rank = rounded_rank;
    rank_float4_stride = rank / LANE_GROUP;
    allocate_storage();

    seed_basis(basis_seed);
}

void WeightMatrix::seed_basis(unsigned seed) {
    mt19937 random_engine(seed);
    normal_distribution<f32> normal_distribution_unit(0.0f, 1.0f);

    float4 *u_data = U_matrix.get_contents_as<float4>();
    float4 *v_data = V_matrix.get_contents_as<float4>();
    const s64 total_float4_element_count = node_count * rank_float4_stride;
    for (s64 element_index = 0; element_index < total_float4_element_count; element_index += 1) {
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
    for (s64 lane_index = 0; lane_index < matrix_count * lane_count; lane_index += 1) {
        coefficient_data[lane_index] = 1.0f;
    }
}

WeightMatrix::WeightMatrix(WeightMatrix &&other) noexcept
    : k2tree(std::move(other.k2tree))
    , matrix_count(other.matrix_count)
    , U_matrix(other.U_matrix)
    , V_matrix(other.V_matrix)
    , coefficients(other.coefficients)
    , edge_row_offset(other.edge_row_offset)
    , neighbor_weight_scratch(other.neighbor_weight_scratch)
    , sparse_delta_row_start(other.sparse_delta_row_start)
    , sparse_delta_edge_ordinal(other.sparse_delta_edge_ordinal)
    , sparse_delta_value(other.sparse_delta_value)
    , pending_delta_edge_ordinal(other.pending_delta_edge_ordinal)
    , pending_delta_value(other.pending_delta_value)
    , pending_delta_matrix_index(other.pending_delta_matrix_index)
    , pending_delta_count(other.pending_delta_count)
    , updated_plane_count(other.updated_plane_count)
    , pending_delta_capacity(other.pending_delta_capacity)
    , sparse_delta_capacity(other.sparse_delta_capacity)
    , sparse_delta_entry_count(std::move(other.sparse_delta_entry_count))
    , plasticity_reserve_entries(other.plasticity_reserve_entries)
    , fit_tolerance(other.fit_tolerance)
    , projection_first_edge_ordinal(std::move(other.projection_first_edge_ordinal))
    , projection_edge_count(std::move(other.projection_edge_count))
    , projection_synapse_prototype(std::move(other.projection_synapse_prototype))
    , owning_backend(other.owning_backend)
    , owning_slab(other.owning_slab)
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
    other.owning_slab = EnginePointer{};
    other.node_count = 0;
    other.total_edge_count = 0;
}

WeightMatrix &WeightMatrix::operator=(WeightMatrix &&other) noexcept {
    if (this == &other) return *this;

    if (owning_backend != nullptr) owning_backend->deallocate_slab(owning_slab);

    k2tree = std::move(other.k2tree);
    matrix_count = other.matrix_count;
    U_matrix = other.U_matrix;
    V_matrix = other.V_matrix;
    coefficients = other.coefficients;
    edge_row_offset = other.edge_row_offset;
    neighbor_weight_scratch = other.neighbor_weight_scratch;
    sparse_delta_row_start = other.sparse_delta_row_start;
    sparse_delta_edge_ordinal = other.sparse_delta_edge_ordinal;
    sparse_delta_value = other.sparse_delta_value;
    pending_delta_edge_ordinal = other.pending_delta_edge_ordinal;
    pending_delta_value = other.pending_delta_value;
    pending_delta_matrix_index = other.pending_delta_matrix_index;
    pending_delta_count = other.pending_delta_count;
    updated_plane_count = other.updated_plane_count;
    pending_delta_capacity = other.pending_delta_capacity;
    sparse_delta_entry_count = std::move(other.sparse_delta_entry_count);
    sparse_delta_capacity = other.sparse_delta_capacity;
    plasticity_reserve_entries = other.plasticity_reserve_entries;
    fit_tolerance = other.fit_tolerance;
    projection_first_edge_ordinal = std::move(other.projection_first_edge_ordinal);
    projection_edge_count = std::move(other.projection_edge_count);
    projection_synapse_prototype = std::move(other.projection_synapse_prototype);
    owning_backend = other.owning_backend;
    owning_slab = other.owning_slab;
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
    other.owning_slab = EnginePointer{};
    other.node_count = 0;
    other.total_edge_count = 0;
    return *this;
}

WeightMatrix::~WeightMatrix() {
    if (owning_backend != nullptr) owning_backend->deallocate_slab(owning_slab);
}

bool WeightMatrix::check_index_inbounds(s32 node_index) const {
    return check_indexing && node_index >= 0 && node_index < node_count;
}

bool WeightMatrix::check_index_inbounds(s32 source, s32 target) const {
    return check_index_inbounds(source) && check_index_inbounds(target);
}

void WeightMatrix::validate_matrix_index(s64 matrix_index) const {
    if (matrix_index < 0 || matrix_index >= matrix_count) {
        log::throw_invalid_argument(log::logger(),
            "WeightMatrix: matrix_index " + to_string(matrix_index) + " out of range [0, " +
            to_string(matrix_count) + ")");
    }
}

s64 WeightMatrix::get_neighbors(s64 node_index, s32 *output_buffer) const {
    if (!can_safely_cast_s64_to_s32(node_index)) return 0;
    if (node_index < 0 || node_index >= node_count) return 0;
    return k2tree.get_neighbors((s32)node_index, output_buffer, max_neighbor_count);
}

s64 WeightMatrix::get_predecessors(s64 node_index, s32 *output_buffer) const {
    if (!can_safely_cast_s64_to_s32(node_index)) return 0;
    if (node_index < 0 || node_index >= node_count) return 0;
    return k2tree.get_predecessors((s32)node_index, output_buffer, max_predecessor_count);
}

optional<s64> WeightMatrix::edge_ordinal(s32 source_node, s32 target_node) const {
    if (source_node < 0 || source_node >= node_count) return nullopt;

    vector<s32> neighbor_buffer((usize)max<s64>(max_neighbor_count, 1));
    const s64 degree = k2tree.get_neighbors(source_node, neighbor_buffer.data(), max_neighbor_count);
    for (s64 slot = 0; slot < degree; slot += 1) {
        if (neighbor_buffer[(usize)slot] == target_node) {
            return edge_row_offset_host[(usize)source_node] + slot;
        }
    }
    return nullopt;
}

f32 *WeightMatrix::coefficient_row(s64 matrix_index) const {
    return coefficients.get_contents_as<f32>() + matrix_index * rank_float4_stride * LANE_GROUP;
}

EnginePointer WeightMatrix::coefficient_range(s64 matrix_index) const {
    const u64 row_bytes = (u64)rank_float4_stride * (u64)LANE_GROUP * sizeof(f32);
    EnginePointer range = coefficients;
    range.offset += (s64)((u64)matrix_index * row_bytes);
    range.total_bytes = row_bytes;
    return range;
}

bool WeightMatrix::ensure_function(EngineFunction &function, const String &name) const {
#ifdef SPIKECOREC_METAL
    if (function.pipeline_state != nullptr) return true;
#else
    if (function.cuda_function != nullptr) return true;
#endif
    if (owning_backend == nullptr) return false;

    Optional<EngineFunction> loaded = owning_backend->load_precompiled_function(name);
    if (!loaded.has_value()) return false;

    function = *loaded;
    return true;
}

f32 WeightMatrix::reconstruct_entry(
    s32 source_node, s32 target_node, const f32 *coefficient_values
) const {
    const float4 *u_row = U_matrix.get_contents_as<float4>() + source_node * rank_float4_stride;
    const float4 *v_row = V_matrix.get_contents_as<float4>() + target_node * rank_float4_stride;

    f32 dot_product = 0.0f;
    for (s64 float4_index = 0; float4_index < rank_float4_stride; float4_index += 1) {
        const f32 *lane_coefficients = coefficient_values + float4_index * LANE_GROUP;
        dot_product += u_row[float4_index].x * (lane_coefficients[0] * v_row[float4_index].x)
                     + u_row[float4_index].y * (lane_coefficients[1] * v_row[float4_index].y)
                     + u_row[float4_index].z * (lane_coefficients[2] * v_row[float4_index].z)
                     + u_row[float4_index].w * (lane_coefficients[3] * v_row[float4_index].w);
    }
    return dot_product;
}

f32 WeightMatrix::sparse_delta_for(s64 matrix_index, s32 source_node, s64 edge_ordinal) const {
    if (sparse_delta_entry_count[(usize)matrix_index] == 0) return 0.0f;
    if (sparse_delta_row_start.is_empty()) return 0.0f;

    const s32 *row_start = sparse_delta_row_start.get_contents_as<s32>() +
                           matrix_index * (node_count + 1);
    const s64 *entry_ordinal = sparse_delta_edge_ordinal.get_contents_as<s64>() +
                              matrix_index * sparse_delta_capacity;
    const f32 *entry_value = sparse_delta_value.get_contents_as<f32>() +
                             matrix_index * sparse_delta_capacity;

    s32 low = row_start[source_node];
    s32 high = row_start[source_node + 1];
    while (low < high) {
        const s32 middle = low + (high - low) / 2;
        if (entry_ordinal[middle] < edge_ordinal) low = middle + 1;
        else high = middle;
    }
    if (low < row_start[source_node + 1] && entry_ordinal[low] == edge_ordinal) {
        return entry_value[low];
    }
    return 0.0f;
}

f32 WeightMatrix::get_for_matrix(s32 source_node, s32 target_node, s64 matrix_index) const {
    validate_matrix_index(matrix_index);
    if (!check_index_inbounds(source_node, target_node)) return 0.0f;

    const f32 reconstructed = reconstruct_entry(source_node, target_node,
                                                coefficient_row(matrix_index));

    const optional<s64> ordinal = edge_ordinal(source_node, target_node);
    if (!ordinal.has_value()) return reconstructed;

    return reconstructed + sparse_delta_for(matrix_index, source_node, *ordinal);
}

f32 WeightMatrix::get(s32 source_node, s32 target_node) const {
    return get_for_matrix(source_node, target_node, WEIGHT_PLANE);
}

s32 WeightMatrix::get_edge_delay_ticks(s32 source_node, s32 target_node) const {
    return max<s32>((s32)lroundf(get_for_matrix(source_node, target_node, DELAY_PLANE)), 1);
}

s32 WeightMatrix::get_edge_synapse_prototype(s32 source_node, s32 target_node) const {
    const optional<s64> ordinal = edge_ordinal(source_node, target_node);
    if (!ordinal.has_value()) return -1;

    const auto entry = upper_bound(projection_first_edge_ordinal.begin(),
                                   projection_first_edge_ordinal.end(), *ordinal);
    if (entry == projection_first_edge_ordinal.begin()) return -1;

    const usize run_index = (usize)(entry - projection_first_edge_ordinal.begin() - 1);
    if (*ordinal >= projection_first_edge_ordinal[run_index] + projection_edge_count[run_index]) {
        return -1;
    }
    return projection_synapse_prototype[run_index];
}

s64 WeightMatrix::delta_capacity_for_threshold(f32 occupancy_threshold_fraction) const {
    // A refit empties S once a plane reaches the threshold; without one, S holds every edge.
    const s64 update_entries = occupancy_threshold_fraction > 0.0f
            ? (s64)ceil((f64)occupancy_threshold_fraction * (f64)total_edge_count)
            : total_edge_count;
    return min(update_entries, total_edge_count) + plasticity_reserve_entries;
}

void WeightMatrix::resize_delta_capacity(s64 new_capacity) {
    if (new_capacity == sparse_delta_capacity) return;

    Vector<Vector<Pair<s64, f32>>> deltas = current_deltas();
    for (const Vector<Pair<s64, f32>> &plane : deltas) {
        if ((s64)plane.size() > new_capacity) {
            log::throw_runtime_error(log::logger(),
                "WeightMatrix::resize_delta_capacity: S holds " + to_string(plane.size()) + " updates in a plane, "
                "more than the " + to_string(new_capacity) + " asked for");
        }
    }

    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    vector<f32> saved_u(U_matrix.get_contents_as<f32>(),
                        U_matrix.get_contents_as<f32>() + node_count * lane_count);
    vector<f32> saved_v(V_matrix.get_contents_as<f32>(),
                        V_matrix.get_contents_as<f32>() + node_count * lane_count);
    vector<f32> saved_coefficients(coefficients.get_contents_as<f32>(),
                                   coefficients.get_contents_as<f32>() + matrix_count * lane_count);

    owning_backend->deallocate_slab(owning_slab);
    sparse_delta_capacity = new_capacity;
    allocate_storage();

    memcpy(U_matrix.get_contents(), saved_u.data(), saved_u.size() * sizeof(f32));
    memcpy(V_matrix.get_contents(), saved_v.data(), saved_v.size() * sizeof(f32));
    memcpy(coefficients.get_contents(), saved_coefficients.data(),
           saved_coefficients.size() * sizeof(f32));
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        if (!deltas[(usize)matrix_index].empty()) rebuild_sparse_delta(matrix_index, deltas[(usize)matrix_index]);
    }
}

s64 WeightMatrix::grown_delta_capacity(s64 required_entries) const {
    // Doubling keeps a growing S from reallocating every tick; a plane never holds more than one
    // update per edge.
    return min(total_edge_count + plasticity_reserve_entries,
               max(required_entries, 2 * max<s64>(sparse_delta_capacity, 1)));
}

void WeightMatrix::rebuild_sparse_delta(s64 matrix_index, Vector<Pair<s64, f32>> &deltas) {
    validate_matrix_index(matrix_index);
    if ((s64)deltas.size() > sparse_delta_capacity) {
        log::throw_runtime_error(log::logger(),
            "WeightMatrix::rebuild_sparse_delta: " + to_string(deltas.size()) + " updates do not fit plane " +
            to_string(matrix_index) + "'s " + to_string(sparse_delta_capacity) + " slots; it should have been refit");
    }

    sort(deltas.begin(), deltas.end(),
         [](const Pair<s64, f32> &left, const Pair<s64, f32> &right) {
             return left.first < right.first;
         });

    s32 *row_start = sparse_delta_row_start.get_contents_as<s32>() +
                     matrix_index * (node_count + 1);
    if (!deltas.empty()) {
        s64 *entry_ordinal = sparse_delta_edge_ordinal.get_contents_as<s64>() +
                            matrix_index * sparse_delta_capacity;
        f32 *entry_value = sparse_delta_value.get_contents_as<f32>() +
                           matrix_index * sparse_delta_capacity;
        for (usize index = 0; index < deltas.size(); index += 1) {
            entry_ordinal[index] = deltas[index].first;
            entry_value[index] = deltas[index].second;
        }
    }

    s64 delta_index = 0;
    for (s64 node_index = 0; node_index <= node_count; node_index += 1) {
        const s64 row_first_ordinal = edge_row_offset_host[(usize)node_index];
        while (delta_index < (s64)deltas.size() &&
               deltas[(usize)delta_index].first < row_first_ordinal) {
            delta_index += 1;
        }
        row_start[node_index] = (s32)delta_index;
    }

    sparse_delta_entry_count[(usize)matrix_index] = (s64)deltas.size();
}

Vector<Vector<Pair<s64, f32>>> WeightMatrix::current_deltas() const {
    Vector<Vector<Pair<s64, f32>>> deltas((usize)matrix_count);
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        const s64 entry_count = sparse_delta_entry_count[(usize)matrix_index];
        if (entry_count == 0) continue;
        const s64 *entry_ordinal = sparse_delta_edge_ordinal.get_contents_as<s64>() +
                                  matrix_index * sparse_delta_capacity;
        const f32 *entry_value = sparse_delta_value.get_contents_as<f32>() +
                                 matrix_index * sparse_delta_capacity;
        for (s64 index = 0; index < entry_count; index += 1) {
            deltas[(usize)matrix_index].push_back({entry_ordinal[index], entry_value[index]});
        }
    }
    return deltas;
}

void WeightMatrix::clear_sparse_deltas() {
    if (!sparse_delta_row_start.is_empty()) {
        memset(sparse_delta_row_start.get_contents(), 0,
               (usize)matrix_count * ((usize)node_count + 1) * sizeof(s32));
    }
    sparse_delta_entry_count.assign((usize)matrix_count, 0);
}

void WeightMatrix::accumulate_edge_delta(
    s64 matrix_index, s32 source_node, s32 target_node, f32 delta
) {
    validate_matrix_index(matrix_index);

    const optional<s64> ordinal = edge_ordinal(source_node, target_node);
    if (!ordinal.has_value()) {
        log::throw_invalid_argument(log::logger(),
            "WeightMatrix::accumulate_edge_delta: (" + to_string(source_node) + ", " +
            to_string(target_node) + ") is not an edge, so it has no value to update");
    }

    Vector<Vector<Pair<s64, f32>>> deltas = current_deltas();
    Vector<Pair<s64, f32>> &plane = deltas[(usize)matrix_index];
    auto existing = find_if(plane.begin(), plane.end(),
                            [&ordinal](const Pair<s64, f32> &entry) { return entry.first == *ordinal; });
    if (existing != plane.end()) {
        existing->second += delta;
        if (existing->second == 0.0f) plane.erase(existing);
    } else if (delta != 0.0f) {
        plane.push_back({*ordinal, delta});
    }

    if ((s64)plane.size() > sparse_delta_capacity) resize_delta_capacity(grown_delta_capacity((s64)plane.size()));
    rebuild_sparse_delta(matrix_index, plane);
}

void WeightMatrix::compact_pending_deltas() {
    if (pending_delta_count.is_empty()) return;

    s32 *pending_count = pending_delta_count.get_contents_as<s32>();
    const s64 staged = min<s64>((s64)*pending_count, pending_delta_capacity);
    if (staged <= 0) {
        *pending_count = 0;
        return;
    }

    if ((s64)*pending_count > pending_delta_capacity) {
        log::logger().warn("compact_pending_deltas: {} updates were dropped this interval "
                           "(capacity {}); compact more often",
                           (s64)*pending_count - pending_delta_capacity, pending_delta_capacity);
    }

    const s64 *staged_ordinal = pending_delta_edge_ordinal.get_contents_as<s64>();
    const f32 *staged_value = pending_delta_value.get_contents_as<f32>();
    const s32 *staged_matrix_index = pending_delta_matrix_index.get_contents_as<s32>();

    Vector<Vector<Pair<s64, f32>>> deltas = current_deltas();
    Vector<bool> plane_changed((usize)matrix_count, false);
    for (s64 index = 0; index < staged; index += 1) {
        const s64 matrix_index = staged_matrix_index[index];
        if (matrix_index < 0 || matrix_index >= matrix_count) continue;
        deltas[(usize)matrix_index].push_back({staged_ordinal[index], staged_value[index]});
        plane_changed[(usize)matrix_index] = true;
    }
    *pending_count = 0;

    // Each edge's updates summed into one entry. One that came back to exactly zero is dropped;
    // the device queues an empty update for it so it is merged here.
    s64 largest_plane = 0;
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        if (!plane_changed[(usize)matrix_index]) continue;
        Vector<Pair<s64, f32>> &plane = deltas[(usize)matrix_index];
        sort(plane.begin(), plane.end(),
             [](const Pair<s64, f32> &left, const Pair<s64, f32> &right) {
                 return left.first < right.first;
             });
        Vector<Pair<s64, f32>> merged;
        for (const Pair<s64, f32> &delta : plane) {
            if (!merged.empty() && merged.back().first == delta.first) {
                merged.back().second += delta.second;
            } else {
                merged.push_back(delta);
            }
        }
        merged.erase(remove_if(merged.begin(), merged.end(),
                               [](const Pair<s64, f32> &delta) { return delta.second == 0.0f; }),
                     merged.end());
        plane = std::move(merged);
        largest_plane = max(largest_plane, (s64)plane.size());
    }

    // S is full: it grows rather than lose an update, and shrinks back after the next refit.
    if (largest_plane > sparse_delta_capacity) {
        resize_delta_capacity(grown_delta_capacity(largest_plane));
        log::logger().debug("compact_pending_deltas: S grew to {} updates per plane", sparse_delta_capacity);
    }

    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        if (plane_changed[(usize)matrix_index]) rebuild_sparse_delta(matrix_index, deltas[(usize)matrix_index]);
    }
    log::logger().debug("compact_pending_deltas: merged {} updates", staged);
}

f32 WeightMatrix::sparse_delta_occupancy_fraction() const {
    if (total_edge_count <= 0) return 0.0f;

    s64 worst = 0;
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        worst = max(worst, sparse_delta_entry_count[(usize)matrix_index]);
    }
    return (f32)worst / (f32)total_edge_count;
}

bool WeightMatrix::is_refit_due(f32 occupancy_threshold_fraction) const {
    if (occupancy_threshold_fraction <= 0.0f) return false;
    return sparse_delta_occupancy_fraction() >= occupancy_threshold_fraction;
}

void WeightMatrix::declare_projections(
    const Vector<s64> &first_edge_ordinal,
    const Vector<s64> &edge_count,
    const Vector<s32> &synapse_prototype,
    const Vector<Vector<f32>> &initial_values
) {
    const usize run_count = first_edge_ordinal.size();
    bool shaped = edge_count.size() == run_count && synapse_prototype.size() == run_count &&
                  (s64)initial_values.size() == matrix_count;
    for (const Vector<f32> &value_per_run : initial_values) shaped = shaped && value_per_run.size() == run_count;
    if (!shaped) {
        log::throw_invalid_argument(log::logger(),
            "WeightMatrix::declare_projections: the runs need one edge count and prototype each, and "
            "initial_values one row per plane with one value per run");
    }

    projection_first_edge_ordinal = first_edge_ordinal;
    projection_edge_count = edge_count;
    projection_synapse_prototype = synapse_prototype;

    if (run_count == 0 || total_edge_count == 0) {
        log::logger().debug("declare_projections: nothing to declare ({} runs, {} edges)",
                            run_count, total_edge_count);
        return;
    }

    const Vector<Vector<f32>> targets = targets_from_projections(initial_values);

    if (!fit_basis_from_projections(initial_values)) {
        // Every lane free, then every plane fitted on lanes of its own, every edge new.
        memset(coefficients.get_contents(), 0,
               (usize)matrix_count * (usize)(rank_float4_stride * LANE_GROUP) * sizeof(f32));
        Vector<s64> every_edge((usize)total_edge_count);
        iota(every_edge.begin(), every_edge.end(), (s64)0);
        for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
            fit_plane(matrix_index, targets[(usize)matrix_index], every_edge, DEFAULT_FIT_RIDGE);
        }
    }
    measure_fit_error(targets);

    // The basis holds the declared values, so S starts empty and only ever holds what changes.
    clear_sparse_deltas();
    resize_delta_capacity(delta_capacity_for_threshold(DEFAULT_REFIT_OCCUPANCY_THRESHOLD_FRACTION));

    if (worst_fit_error() > fit_tolerance) {
        log::logger().warn("WeightMatrix: at rank {} the basis reproduces the declared values to within {:.3e} "
                           "relative, short of the {:.0e} tolerance", rank, worst_fit_error(), fit_tolerance);
    }

    const s64 basis_bytes = (2 * node_count * rank + matrix_count * rank) * (s64)sizeof(f32);
    const s64 delta_bytes =
            sparse_delta_capacity * matrix_count * (s64)(sizeof(s64) + sizeof(f32)) +
            pending_delta_capacity * (s64)(sizeof(s64) + sizeof(f32) + sizeof(s32));
    log::logger().info("WeightMatrix: {} projection runs over {} edges and {} planes at rank {}, worst relative "
                       "error {:.3e}; {} bytes of basis + {} bytes of update buffers, against {} bytes for one "
                       "float per edge per plane",
                       run_count, total_edge_count, matrix_count, rank, worst_fit_error(), basis_bytes,
                       delta_bytes, total_edge_count * matrix_count * (s64)sizeof(f32));
}

bool WeightMatrix::fit_basis_from_projections(const Vector<Vector<f32>> &initial_values) {
    const s64 run_count = (s64)projection_first_edge_ordinal.size();

    // Every plane's value on one run, in plane order.
    auto plane_values_of = [&](s64 run_index) {
        Vector<f32> values;
        for (const Vector<f32> &value_per_run : initial_values) values.push_back(value_per_run[(usize)run_index]);
        return values;
    };

    Vector<s64> lane_of_run((usize)max<s64>(run_count, 0), 0);
    Vector<Vector<f32>> lane_values;

    for (s64 run_index = 0; run_index < run_count; run_index += 1) {
        const Vector<f32> values = plane_values_of(run_index);
        const auto found = find(lane_values.begin(), lane_values.end(), values);
        lane_of_run[(usize)run_index] = found - lane_values.begin();
        if (found == lane_values.end()) lane_values.push_back(values);
    }

    const s64 distinct_count = (s64)lane_values.size();

    if (round_up_to_lane_group(distinct_count) > MAX_RANK_FLOAT4_STRIDE * LANE_GROUP) {
        return false;
    }

    resize_basis(distinct_count);

    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    f32 *u_data = U_matrix.get_contents_as<f32>();
    f32 *v_data = V_matrix.get_contents_as<f32>();

    memset(u_data, 0, (usize)node_count * (usize)lane_count * sizeof(f32));
    memset(v_data, 0, (usize)node_count * (usize)lane_count * sizeof(f32));

    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        f32 *coefficient_values = coefficient_row(matrix_index);
        for (s64 lane_index = 0; lane_index < lane_count; lane_index += 1) {
            coefficient_values[lane_index] =
                    lane_index < distinct_count ? lane_values[(usize)lane_index][(usize)matrix_index] : 0.0f;
        }
    }

    vector<s32> neighbor_buffer((usize)max<s64>(max_neighbor_count, 1));
    s64 run_index = 0;
    for (s64 source_node = 0; source_node < node_count; source_node += 1) {
        const s64 degree = k2tree.get_neighbors((s32)source_node, neighbor_buffer.data(),
                                                max_neighbor_count);
        for (s64 slot = 0; slot < degree; slot += 1) {
            const s64 ordinal = edge_row_offset_host[(usize)source_node] + slot;

            while (run_index + 1 < run_count &&
                   ordinal >= projection_first_edge_ordinal[(usize)run_index] +
                              projection_edge_count[(usize)run_index]) {
                run_index += 1;
            }
            if (run_index >= run_count) break;

            const s64 lane = lane_of_run[(usize)run_index];
            u_data[source_node * lane_count + lane] = 1.0f;
            v_data[neighbor_buffer[(usize)slot] * lane_count + lane] = 1.0f;
        }
    }

    log::logger().debug("fit_basis_from_projections: {} runs carry {} distinct combinations of "
                        "initial values, so the basis is rank {}", run_count, distinct_count, rank);
    return true;
}

Vector<Vector<f32>> WeightMatrix::targets_from_projections(const Vector<Vector<f32>> &initial_values) const {
    Vector<Vector<f32>> targets((usize)matrix_count,
                                Vector<f32>((usize)max<s64>(total_edge_count, 0), 0.0f));
    const s64 run_count = (s64)projection_first_edge_ordinal.size();
    if (run_count == 0) return targets;

    for (s64 ordinal = 0; ordinal < total_edge_count; ordinal += 1) {
        const auto entry = upper_bound(projection_first_edge_ordinal.begin(),
                                       projection_first_edge_ordinal.end(), ordinal);
        const s64 run_index = max<s64>(entry - projection_first_edge_ordinal.begin() - 1, 0);
        for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
            targets[(usize)matrix_index][(usize)ordinal] = initial_values[(usize)matrix_index][(usize)run_index];
        }
    }
    return targets;
}

Vector<Vector<f32>> WeightMatrix::targets_from_basis_and_deltas(
    const Vector<Vector<Pair<s64, f32>>> &deltas_per_matrix
) const {
    Vector<Vector<f32>> targets((usize)matrix_count,
                                Vector<f32>((usize)max<s64>(total_edge_count, 0), 0.0f));
    if (total_edge_count == 0) return targets;

    vector<s32> neighbor_buffer((usize)max<s64>(max_neighbor_count, 1));
    for (s64 source_node = 0; source_node < node_count; source_node += 1) {
        const s64 degree = k2tree.get_neighbors((s32)source_node, neighbor_buffer.data(),
                                                max_neighbor_count);
        for (s64 slot = 0; slot < degree; slot += 1) {
            const s64 ordinal = edge_row_offset_host[(usize)source_node] + slot;
            const s32 target_node = neighbor_buffer[(usize)slot];
            for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
                targets[(usize)matrix_index][(usize)ordinal] =
                        reconstruct_entry((s32)source_node, target_node, coefficient_row(matrix_index));
            }
        }
    }
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        for (const Pair<s64, f32> &delta : deltas_per_matrix[(usize)matrix_index]) {
            targets[(usize)matrix_index][(usize)delta.first] += delta.second;
        }
    }
    return targets;
}

void WeightMatrix::grow_basis(s64 new_rank) {
    const s64 grown_rank = round_up_to_lane_group(new_rank);
    if (grown_rank <= rank) return;
    if (grown_rank / LANE_GROUP > MAX_RANK_FLOAT4_STRIDE) {
        log::throw_invalid_argument(log::logger(),
            "WeightMatrix: rank " + to_string(grown_rank) + " exceeds the kernel limit of " +
            to_string((s64)MAX_RANK_FLOAT4_STRIDE * LANE_GROUP) + " lanes");
    }

    const s64 old_lane_count = rank_float4_stride * LANE_GROUP;
    vector<f32> saved_u(U_matrix.get_contents_as<f32>(),
                        U_matrix.get_contents_as<f32>() + node_count * old_lane_count);
    vector<f32> saved_v(V_matrix.get_contents_as<f32>(),
                        V_matrix.get_contents_as<f32>() + node_count * old_lane_count);
    vector<f32> saved_coefficients(coefficients.get_contents_as<f32>(),
                                   coefficients.get_contents_as<f32>() + matrix_count * old_lane_count);
    Vector<Vector<Pair<s64, f32>>> deltas = current_deltas();

    owning_backend->deallocate_slab(owning_slab);
    rank = grown_rank;
    rank_float4_stride = rank / LANE_GROUP;
    allocate_storage();

    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    mt19937 random_engine(basis_seed + (unsigned)rank);
    normal_distribution<f32> normal_distribution_unit(0.0f, 1.0f);
    f32 *u_data = U_matrix.get_contents_as<f32>();
    f32 *v_data = V_matrix.get_contents_as<f32>();
    for (s64 node_index = 0; node_index < node_count; node_index += 1) {
        for (s64 lane = 0; lane < lane_count; lane += 1) {
            const bool kept = lane < old_lane_count;
            u_data[node_index * lane_count + lane] =
                    kept ? saved_u[(usize)(node_index * old_lane_count + lane)] : normal_distribution_unit(random_engine);
            v_data[node_index * lane_count + lane] =
                    kept ? saved_v[(usize)(node_index * old_lane_count + lane)] : normal_distribution_unit(random_engine);
        }
    }
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        f32 *coefficient_values = coefficient_row(matrix_index);
        for (s64 lane = 0; lane < lane_count; lane += 1) {
            coefficient_values[lane] =
                    lane < old_lane_count ? saved_coefficients[(usize)(matrix_index * old_lane_count + lane)] : 0.0f;
        }
    }
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        if (!deltas[(usize)matrix_index].empty()) rebuild_sparse_delta(matrix_index, deltas[(usize)matrix_index]);
    }
}

void WeightMatrix::refit(f32 ridge_regularization) {
    if (total_edge_count == 0) {
        clear_sparse_deltas();
        return;
    }

    // Every plane's values now, the basis plus its updates, taken before any plane is refit.
    // Refitting a plane changes only lanes no other plane reads, so these stay true for the rest.
    const Vector<Vector<Pair<s64, f32>>> deltas_per_matrix = current_deltas();
    const Vector<Vector<f32>> targets = targets_from_basis_and_deltas(deltas_per_matrix);

    // A plane with no updates already reads what it should, so it is left exactly as it is.
    bool meets_tolerance = true;
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        const Vector<Pair<s64, f32>> &deltas = deltas_per_matrix[(usize)matrix_index];
        if (deltas.empty()) continue;

        Vector<s64> changed_edge_ordinals;
        changed_edge_ordinals.reserve(deltas.size());
        for (const Pair<s64, f32> &delta : deltas) changed_edge_ordinals.push_back(delta.first);
        meets_tolerance = fit_plane(matrix_index, targets[(usize)matrix_index], changed_edge_ordinals,
                                    ridge_regularization) && meets_tolerance;
    }
    clear_sparse_deltas();

    // A plane that was not refit keeps the error of the fit that made it.
    const Vector<f32> previous_fit_error = measured_fit_error;
    measure_fit_error(targets);
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        if (deltas_per_matrix[(usize)matrix_index].empty() && (s64)previous_fit_error.size() == matrix_count) {
            measured_fit_error[(usize)matrix_index] = previous_fit_error[(usize)matrix_index];
        }
    }

    String error_per_plane;
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        if (matrix_index > 0) error_per_plane += ", ";
        error_per_plane += to_string(measured_fit_error[(usize)matrix_index]);
    }
    log::logger().debug("refit: rank {}, worst relative error per plane [{}]", rank, error_per_plane);
    if (!meets_tolerance) {
        log::logger().warn("refit: at rank {} the basis is {:.3e} off, short of the {:.0e} tolerance", rank,
                           worst_fit_error(), fit_tolerance);
    }
}

bool WeightMatrix::fit_plane(
    s64 matrix_index, const Vector<f32> &targets, const Vector<s64> &changed_edge_ordinals,
    f32 ridge_regularization
) {
    const s64 lane_limit = MAX_RANK_FLOAT4_STRIDE * LANE_GROUP;
    auto lane_count = [this]() { return rank_float4_stride * LANE_GROUP; };

    // Each edge's ends, and each node's edges out and in.
    vector<s32> edge_source((usize)total_edge_count);
    vector<s32> edge_target((usize)total_edge_count);
    vector<vector<s64>> outgoing_edges((usize)node_count);
    vector<vector<s64>> incoming_edges((usize)node_count);
    {
        vector<s32> neighbor_buffer((usize)max<s64>(max_neighbor_count, 1));
        for (s64 source_node = 0; source_node < node_count; source_node += 1) {
            const s64 degree = k2tree.get_neighbors((s32)source_node, neighbor_buffer.data(),
                                                    max_neighbor_count);
            for (s64 slot = 0; slot < degree; slot += 1) {
                const s64 ordinal = edge_row_offset_host[(usize)source_node] + slot;
                const s32 target_node = neighbor_buffer[(usize)slot];
                edge_source[(usize)ordinal] = (s32)source_node;
                edge_target[(usize)ordinal] = target_node;
                outgoing_edges[(usize)source_node].push_back(ordinal);
                incoming_edges[(usize)target_node].push_back(ordinal);
            }
        }
    }

    // This plane may change only its own lanes: the ones it reads and no other plane does. A lane
    // another plane also reads stays as it is, and a lane no plane reads is free to take.
    auto lane_is_read_by = [this](s64 plane, s64 lane) { return coefficient_row(plane)[lane] != 0.0f; };
    auto lane_is_free = [&](s64 lane) {
        for (s64 plane = 0; plane < matrix_count; plane += 1) {
            if (lane_is_read_by(plane, lane)) return false;
        }
        return true;
    };
    vector<s64> own_lanes;
    vector<s64> shared_lanes;
    for (s64 lane = 0; lane < lane_count(); lane += 1) {
        if (!lane_is_read_by(matrix_index, lane)) continue;
        bool read_by_another_plane = false;
        for (s64 plane = 0; plane < matrix_count; plane += 1) {
            read_by_another_plane = read_by_another_plane || (plane != matrix_index && lane_is_read_by(plane, lane));
        }
        (read_by_another_plane ? shared_lanes : own_lanes).push_back(lane);
    }

    // What the shared lanes give each edge. They stay fixed, so this is known before any lane moves.
    vector<f64> shared_value((usize)total_edge_count, 0.0);
    {
        const f32 *u_data = U_matrix.get_contents_as<f32>();
        const f32 *v_data = V_matrix.get_contents_as<f32>();
        const f32 *coefficient_values = coefficient_row(matrix_index);
        const s64 lanes = lane_count();
        for_each_index_in_parallel(total_edge_count, [&](s64 ordinal) {
            const s64 source_node = edge_source[(usize)ordinal];
            const s64 target_node = edge_target[(usize)ordinal];
            for (s64 lane : shared_lanes) {
                shared_value[(usize)ordinal] += (f64)u_data[source_node * lanes + lane] *
                                                (f64)coefficient_values[lane] * (f64)v_data[target_node * lanes + lane];
            }
        });
    }

    // The scale errors are measured against: the plane's RMS. A plane whose every value is zero has
    // none, and must read exactly zero, so its own lanes are freed; only what shared lanes give is
    // left to cancel, measured against that.
    f64 scale = root_mean_square(targets);
    if (scale == 0.0) {
        for (s64 lane : own_lanes) coefficient_row(matrix_index)[lane] = 0.0f;
        own_lanes.clear();
        scale = root_mean_square(shared_value);
        if (scale == 0.0) return true;
    }

    // What the own lanes must hold, in units of the scale: the target less what the shared lanes
    // give. Each edge's error counts against max(|value|, RMS), as the tolerance does.
    vector<f64> residual((usize)total_edge_count);
    vector<f64> error_scale((usize)total_edge_count);
    for (s64 ordinal = 0; ordinal < total_edge_count; ordinal += 1) {
        const f64 target = (f64)targets[(usize)ordinal];
        residual[(usize)ordinal] = (target - shared_value[(usize)ordinal]) / scale;
        error_scale[(usize)ordinal] = max(fabs(target) / scale, 1.0);
    }

    // Own lanes carry the plane's RMS as their coefficient, so U and V hold values in the units the
    // solves work in. A lane fitted at an earlier RMS is rescaled; the values it gives stay the same.
    {
        f32 *u_data = U_matrix.get_contents_as<f32>();
        f32 *coefficient_values = coefficient_row(matrix_index);
        const s64 lanes = lane_count();
        for (s64 lane : own_lanes) {
            const f32 factor = (f32)((f64)coefficient_values[lane] / scale);
            for (s64 node_index = 0; node_index < node_count; node_index += 1) {
                u_data[node_index * lanes + lane] *= factor;
            }
            coefficient_values[lane] = (f32)scale;
        }
    }

    // The rows the sweeps may move: both ends of every changed edge. An edge touching neither keeps
    // its value exactly, so the sweeps measure only the edges that touch one.
    vector<char> source_is_dirty((usize)node_count, 0);
    vector<char> target_is_dirty((usize)node_count, 0);
    for (s64 ordinal : changed_edge_ordinals) {
        source_is_dirty[(usize)edge_source[(usize)ordinal]] = 1;
        target_is_dirty[(usize)edge_target[(usize)ordinal]] = 1;
    }
    vector<s64> dirty_sources;
    vector<s64> dirty_targets;
    for (s64 node_index = 0; node_index < node_count; node_index += 1) {
        if (source_is_dirty[(usize)node_index]) dirty_sources.push_back(node_index);
        if (target_is_dirty[(usize)node_index]) dirty_targets.push_back(node_index);
    }
    vector<s64> touched_edges;
    for (s64 ordinal = 0; ordinal < total_edge_count; ordinal += 1) {
        if (source_is_dirty[(usize)edge_source[(usize)ordinal]] || target_is_dirty[(usize)edge_target[(usize)ordinal]]) {
            touched_edges.push_back(ordinal);
        }
    }
    vector<s64> every_edge((usize)total_edge_count);
    iota(every_edge.begin(), every_edge.end(), (s64)0);

    // The worst relative error over some edges, from the own lanes as stored.
    auto worst_relative_error = [&](const vector<s64> &edges) {
        const f32 *u_data = U_matrix.get_contents_as<f32>();
        const f32 *v_data = V_matrix.get_contents_as<f32>();
        const s64 lanes = lane_count();
        vector<f64> error_of_edge(edges.size(), 0.0);
        for_each_index_in_parallel((s64)edges.size(), [&](s64 index) {
            const s64 ordinal = edges[(usize)index];
            const f32 *u_row = u_data + edge_source[(usize)ordinal] * lanes;
            const f32 *v_row = v_data + edge_target[(usize)ordinal] * lanes;
            f64 own_value = 0.0;
            for (s64 lane : own_lanes) own_value += (f64)u_row[lane] * (f64)v_row[lane];
            error_of_edge[(usize)index] = fabs(own_value - residual[(usize)ordinal]) / error_scale[(usize)ordinal];
        });
        return error_of_edge.empty() ? 0.0 : *max_element(error_of_edge.begin(), error_of_edge.end());
    };

    // Lanes this plane can come to own: its own, the free ones, and the room left under the limit.
    auto available_lane_count = [&]() {
        s64 free_count = 0;
        for (s64 lane = 0; lane < lane_count(); lane += 1) free_count += lane_is_free(lane) ? 1 : 0;
        return (s64)own_lanes.size() + free_count + (lane_limit - lane_count());
    };

    // Gives this plane more lanes, free ones first, then new ones. A lane it takes starts with U
    // zero, so no value changes until a solve uses it, and V N(0,1).
    auto take_lanes = [&](s64 count) {
        vector<s64> taken;
        for (s64 lane = 0; lane < lane_count() && (s64)taken.size() < count; lane += 1) {
            if (lane_is_free(lane)) taken.push_back(lane);
        }
        const s64 old_lane_count = lane_count();
        const s64 missing = count - (s64)taken.size();
        if (missing > 0) {
            grow_basis(old_lane_count + missing);
            for (s64 lane = old_lane_count; lane < old_lane_count + missing; lane += 1) taken.push_back(lane);
        }
        if (taken.empty()) return;

        mt19937 random_engine(basis_seed + (unsigned)(matrix_index * lane_limit + taken.front()));
        normal_distribution<f32> normal_distribution_unit(0.0f, 1.0f);
        f32 *u_data = U_matrix.get_contents_as<f32>();
        f32 *v_data = V_matrix.get_contents_as<f32>();
        const s64 lanes = lane_count();
        for (s64 lane : taken) {
            for (s64 node_index = 0; node_index < node_count; node_index += 1) {
                u_data[node_index * lanes + lane] = 0.0f;
                v_data[node_index * lanes + lane] = normal_distribution_unit(random_engine);
            }
            coefficient_row(matrix_index)[lane] = (f32)scale;
            own_lanes.push_back(lane);
        }
    };

    // Weighted least squares for one node's own-lane row, from the edges that use it: each edge
    // gives the other end's own-lane row, weighted by 1/error_scale^2 so that the squares count
    // what the tolerance counts.
    auto solve_row = [&](f32 *solved_row, const vector<s64> &edges, const f32 *other_side,
                         const vector<s32> &other_end) {
        const s64 own_count = (s64)own_lanes.size();
        const s64 lanes = lane_count();
        vector<f64> gram((usize)(own_count * own_count), 0.0);
        vector<f64> right_hand_side((usize)own_count, 0.0);
        vector<f64> other_row((usize)own_count);
        for (s64 ordinal : edges) {
            const f32 *other = other_side + other_end[(usize)ordinal] * lanes;
            for (s64 index = 0; index < own_count; index += 1) other_row[(usize)index] = (f64)other[own_lanes[(usize)index]];
            const f64 weight = 1.0 / (error_scale[(usize)ordinal] * error_scale[(usize)ordinal]);
            for (s64 row = 0; row < own_count; row += 1) {
                right_hand_side[(usize)row] += weight * other_row[(usize)row] * residual[(usize)ordinal];
                for (s64 column = 0; column <= row; column += 1) {
                    gram[(usize)(row * own_count + column)] += weight * other_row[(usize)row] * other_row[(usize)column];
                }
            }
        }
        if (!solve_symmetric_in_place(gram, right_hand_side, own_count, (f64)ridge_regularization)) return;
        for (s64 index = 0; index < own_count; index += 1) {
            solved_row[own_lanes[(usize)index]] = (f32)right_hand_side[(usize)index];
        }
    };

    // Sweeps at the current own lanes until every touched edge meets the tolerance, or the best
    // worst-case error stalls. Leaves the rows as they were at the best sweep and returns its error.
    auto sweep_until_stalled = [&]() {
        vector<f32> best_rows;
        auto copy_rows = [&](bool save) {
            const s64 lanes = lane_count();
            if (save) best_rows.clear();
            usize position = 0;
            auto copy_side = [&](const vector<s64> &nodes, f32 *data) {
                for (s64 node_index : nodes) {
                    for (s64 lane : own_lanes) {
                        if (save) {
                            best_rows.push_back(data[node_index * lanes + lane]);
                        } else {
                            data[node_index * lanes + lane] = best_rows[position];
                            position += 1;
                        }
                    }
                }
            };
            copy_side(dirty_sources, U_matrix.get_contents_as<f32>());
            copy_side(dirty_targets, V_matrix.get_contents_as<f32>());
        };

        f64 best_error = worst_relative_error(touched_edges);
        copy_rows(true);
        vector<f64> best_error_after_sweep = {best_error};
        for (s32 sweep = 1; best_error > (f64)fit_tolerance && sweep <= PLANE_FIT_MAXIMUM_SWEEP_COUNT; sweep += 1) {
            f32 *u_data = U_matrix.get_contents_as<f32>();
            f32 *v_data = V_matrix.get_contents_as<f32>();
            const s64 lanes = lane_count();
            for_each_index_in_parallel((s64)dirty_sources.size(), [&](s64 index) {
                const s64 node_index = dirty_sources[(usize)index];
                solve_row(u_data + node_index * lanes, outgoing_edges[(usize)node_index], v_data, edge_target);
            });
            for_each_index_in_parallel((s64)dirty_targets.size(), [&](s64 index) {
                const s64 node_index = dirty_targets[(usize)index];
                solve_row(v_data + node_index * lanes, incoming_edges[(usize)node_index], u_data, edge_source);
            });

            const f64 error = worst_relative_error(touched_edges);
            if (error < best_error) {
                best_error = error;
                copy_rows(true);
            }
            best_error_after_sweep.push_back(best_error);
            if (sweep >= PLANE_FIT_STALL_WINDOW_SWEEP_COUNT &&
                best_error > (1.0 - FIT_STALL_IMPROVEMENT) *
                             best_error_after_sweep[(usize)(sweep - PLANE_FIT_STALL_WINDOW_SWEEP_COUNT)]) {
                break;
            }
        }
        copy_rows(false);
        return best_error;
    };

    // The guarantee. With at least as many own lanes as any node has edges on one side, plus a lane
    // group of slack, the other side's rows can be anything in general position -- fresh N(0,1) --
    // and this side's rows then reproduce every edge exactly: per node, the minimum-norm solution of
    // its few equations. The side solved is the one whose largest degree is smaller.
    const bool solve_source_rows = max_neighbor_count <= max_predecessor_count;
    const s64 exact_lane_count = round_up_to_lane_group(min(max_neighbor_count, max_predecessor_count)) + LANE_GROUP;
    auto solve_exactly = [&]() {
        constexpr f64 EXACT_SOLVE_RIDGE_PER_LANE = 1.0e-10;
        const s64 lanes = lane_count();
        const s64 own_count = (s64)own_lanes.size();
        f32 *solved_side = solve_source_rows ? U_matrix.get_contents_as<f32>() : V_matrix.get_contents_as<f32>();
        f32 *fixed_side = solve_source_rows ? V_matrix.get_contents_as<f32>() : U_matrix.get_contents_as<f32>();
        const vector<vector<s64>> &edges_of_node = solve_source_rows ? outgoing_edges : incoming_edges;
        const vector<s32> &other_end = solve_source_rows ? edge_target : edge_source;

        mt19937 random_engine(basis_seed + 7919u * (unsigned)(matrix_index + 1));
        normal_distribution<f32> normal_distribution_unit(0.0f, 1.0f);
        for (s64 node_index = 0; node_index < node_count; node_index += 1) {
            for (s64 lane : own_lanes) fixed_side[node_index * lanes + lane] = normal_distribution_unit(random_engine);
        }

        for_each_index_in_parallel(node_count, [&](s64 node_index) {
            const vector<s64> &edges = edges_of_node[(usize)node_index];
            const s64 edge_count = (s64)edges.size();
            f32 *solved_row = solved_side + node_index * lanes;
            for (s64 lane : own_lanes) solved_row[lane] = 0.0f;
            if (edge_count == 0) return;

            // The other ends' own-lane rows A, one per edge; then y from (A A^T) y = residual, and
            // the row is A^T y.
            vector<f64> other_rows((usize)(edge_count * own_count));
            for (s64 edge_index = 0; edge_index < edge_count; edge_index += 1) {
                const f32 *other = fixed_side + other_end[(usize)edges[(usize)edge_index]] * lanes;
                for (s64 index = 0; index < own_count; index += 1) {
                    other_rows[(usize)(edge_index * own_count + index)] = (f64)other[own_lanes[(usize)index]];
                }
            }
            vector<f64> gram((usize)(edge_count * edge_count));
            vector<f64> right_hand_side((usize)edge_count);
            for (s64 row = 0; row < edge_count; row += 1) {
                right_hand_side[(usize)row] = residual[(usize)edges[(usize)row]];
                for (s64 column = 0; column <= row; column += 1) {
                    f64 dot_product = 0.0;
                    for (s64 index = 0; index < own_count; index += 1) {
                        dot_product += other_rows[(usize)(row * own_count + index)] *
                                       other_rows[(usize)(column * own_count + index)];
                    }
                    gram[(usize)(row * edge_count + column)] = dot_product;
                }
            }
            if (!solve_symmetric_in_place(gram, right_hand_side, edge_count,
                                          EXACT_SOLVE_RIDGE_PER_LANE * (f64)own_count)) {
                return;
            }
            for (s64 index = 0; index < own_count; index += 1) {
                f64 value = 0.0;
                for (s64 row = 0; row < edge_count; row += 1) {
                    value += right_hand_side[(usize)row] * other_rows[(usize)(row * own_count + index)];
                }
                solved_row[own_lanes[(usize)index]] = (f32)value;
            }
        });
    };

    // Already exact with the lanes it has -- a plane of zeros, say: nothing to fit.
    if (worst_relative_error(touched_edges) == 0.0) return true;

    // For memory, the fewest own lanes that meet the tolerance: the warm start at the lanes the plane
    // has, then doubling, stopping short of the exact count while the exact solve is available.
    const bool exact_is_available = exact_lane_count <= available_lane_count();
    bool meets_tolerance = false;
    s64 lane_target = max<s64>((s64)own_lanes.size(), LANE_GROUP);
    while (!meets_tolerance) {
        if (exact_is_available && lane_target >= exact_lane_count) break;
        lane_target = min(lane_target, available_lane_count());
        if ((s64)own_lanes.size() < lane_target) take_lanes(lane_target - (s64)own_lanes.size());
        meets_tolerance = sweep_until_stalled() <= (f64)fit_tolerance;
        if (!exact_is_available && (s64)own_lanes.size() >= available_lane_count()) break;
        lane_target *= 2;
    }

    if (!meets_tolerance && exact_is_available) {
        if ((s64)own_lanes.size() < exact_lane_count) take_lanes(exact_lane_count - (s64)own_lanes.size());
        solve_exactly();
        meets_tolerance = worst_relative_error(every_edge) <= (f64)fit_tolerance;
    }

    log::logger().debug("fit_plane: plane {} on {} own and {} shared lanes, {} changed edges, {} the tolerance",
                        matrix_index, own_lanes.size(), shared_lanes.size(), changed_edge_ordinals.size(),
                        meets_tolerance ? "meets" : "misses");
    return meets_tolerance;
}

void WeightMatrix::measure_fit_error(const Vector<Vector<f32>> &targets_per_matrix) {
    measured_fit_error.assign((usize)matrix_count, 0.0f);
    if (total_edge_count == 0) return;

    Vector<f32> field_scale((usize)matrix_count);
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        field_scale[(usize)matrix_index] = field_scale_of(targets_per_matrix[(usize)matrix_index]);
    }

    vector<s32> neighbor_buffer((usize)max<s64>(max_neighbor_count, 1));
    for (s64 source_node = 0; source_node < node_count; source_node += 1) {
        const s64 degree = k2tree.get_neighbors((s32)source_node, neighbor_buffer.data(),
                                                max_neighbor_count);
        for (s64 slot = 0; slot < degree; slot += 1) {
            const s64 ordinal = edge_row_offset_host[(usize)source_node] + slot;
            for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
                const f32 target = targets_per_matrix[(usize)matrix_index][(usize)ordinal];
                const f32 reconstructed = reconstruct_entry((s32)source_node, neighbor_buffer[(usize)slot],
                                                            coefficient_row(matrix_index));
                const f32 scale = max(fabsf(target), field_scale[(usize)matrix_index]);
                f32 &worst = measured_fit_error[(usize)matrix_index];
                worst = max(worst, fabsf(reconstructed - target) / max(scale, numeric_limits<f32>::min()));
            }
        }
    }
}

f32 WeightMatrix::worst_fit_error() const {
    f32 worst = 0.0f;
    for (f32 error : measured_fit_error) worst = max(worst, error);
    return worst;
}

void WeightMatrix::neighbor_weights(f32 *output_weights) const {
    neighbor_weights_for_matrix(output_weights, WEIGHT_PLANE);
}

void WeightMatrix::neighbor_weights_for_matrix(f32 *output_weights, s64 matrix_index) const {
    validate_matrix_index(matrix_index);
    if (total_edge_count == 0) return;

    bool reconstructed_on_device = false;
    if (ensure_function(neighbor_weights_function, "neighbor_weights_kernel")) {
        const s32 branching_factor = k2tree.branching_factor;
        const s32 superblock_size_words = k2tree.superblock_size_words;
        const s32 padded_node_count = k2tree.padded_node_count;
        const s32 tree_height = k2tree.tree_height;
        const s32 internal_bit_count = k2tree.internal_bit_count;
        const s64 node_count_argument = node_count;
        const s64 total_edge_count_argument = total_edge_count;
        const s64 rank_float4_stride_argument = rank_float4_stride;

        Vector<EnginePointer> parameters = {
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
            neighbor_weight_scratch,                        // 16
        };

        if (owning_backend->run_function(neighbor_weights_function, parameters, total_edge_count)) {
            owning_backend->prefetch_to_cpu(neighbor_weight_scratch,
                                            (u64)total_edge_count * sizeof(f32));
            memcpy(output_weights, neighbor_weight_scratch.get_contents(),
                   (usize)total_edge_count * sizeof(f32));
            reconstructed_on_device = true;
        } else {
            log::logger().warn("neighbor_weights: the GPU dispatch failed; answering on the host");
        }
    }

    if (!reconstructed_on_device) {
        const f32 *coefficient_values = coefficient_row(matrix_index);
        vector<s32> neighbor_buffer((usize)max<s64>(max_neighbor_count, 1));
        for (s64 source_node = 0; source_node < node_count; source_node += 1) {
            const s64 degree = k2tree.get_neighbors((s32)source_node, neighbor_buffer.data(),
                                                    max_neighbor_count);
            for (s64 slot = 0; slot < degree; slot += 1) {
                const s64 ordinal = edge_row_offset_host[(usize)source_node] + slot;
                output_weights[(usize)ordinal] = reconstruct_entry(
                        (s32)source_node, neighbor_buffer[(usize)slot], coefficient_values);
            }
        }
    }

    // Like every read, the recent updates in S are added on top of the basis.
    const Vector<Vector<Pair<s64, f32>>> deltas = current_deltas();
    for (const Pair<s64, f32> &delta : deltas[(usize)matrix_index]) {
        output_weights[(usize)delta.first] += delta.second;
    }
}

WeightStats WeightMatrix::neighbor_weight_stats() const {
    if (total_edge_count == 0) return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    vector<f32> weight_buffer((usize)total_edge_count);
    neighbor_weights(weight_buffer.data());

    f32 weight_sum = 0.0f;
    f32 sum_of_squares = 0.0f;
    f32 min_weight = weight_buffer[0];
    f32 max_weight = weight_buffer[0];
    for (s64 edge_index = 0; edge_index < total_edge_count; edge_index += 1) {
        const f32 weight = weight_buffer[(usize)edge_index];
        weight_sum += weight;
        sum_of_squares += weight * weight;
        min_weight = min(min_weight, weight);
        max_weight = max(max_weight, weight);
    }

    const f32 mean = weight_sum / (f32)total_edge_count;
    const f32 variance = (sum_of_squares / (f32)total_edge_count) - mean * mean;
    return {mean,
            sqrtf(variance > 0.0f ? variance : 0.0f),
            sqrtf(sum_of_squares / (f32)total_edge_count),
            min_weight,
            max_weight};
}

ScaleResult WeightMatrix::scale_neighbor_weights_to_root_mean_square(
    f32 target_root_mean_square, f32 epsilon
) {
    if (target_root_mean_square < 0.0f) {
        log::throw_invalid_argument(log::logger(),
            "scale_neighbor_weights_to_root_mean_square: target must be non-negative");
    }

    const WeightStats stats_before = neighbor_weight_stats();
    const f32 current_root_mean_square = max(stats_before.root_mean_square, epsilon);
    const f32 scale_factor = target_root_mean_square / current_root_mean_square;

    // Only the weight plane: its coefficient row and its updates. U and V are shared with every
    // other plane, so scaling them would rescale delays and synapse state too.
    f32 *weight_coefficients = coefficient_row(WEIGHT_PLANE);
    for (s64 lane = 0; lane < rank_float4_stride * LANE_GROUP; lane += 1) {
        weight_coefficients[lane] *= scale_factor;
    }
    f32 *weight_deltas = sparse_delta_value.get_contents_as<f32>() + WEIGHT_PLANE * sparse_delta_capacity;
    for (s64 index = 0; index < sparse_delta_entry_count[(usize)WEIGHT_PLANE]; index += 1) {
        weight_deltas[index] *= scale_factor;
    }

    const WeightStats stats_after = neighbor_weight_stats();
    log::logger().debug("scale_neighbor_weights_to_root_mean_square: target={} factor={} "
                        "rms_before={} rms_after={}",
                        target_root_mean_square, scale_factor,
                        stats_before.root_mean_square, stats_after.root_mean_square);

    return {target_root_mean_square, scale_factor, stats_before, stats_after};
}

void WeightMatrix::save(const char *filepath) const {
    ofstream file(filepath, ios::binary);
    file.write(reinterpret_cast<const char *>(&WEIGHT_MATRIX_SAVE_MAGIC), sizeof(u32));
    file.write(reinterpret_cast<const char *>(&node_count), sizeof(s64));
    file.write(reinterpret_cast<const char *>(&matrix_count), sizeof(s64));
    file.write(reinterpret_cast<const char *>(&rank), sizeof(s64));
    file.write(reinterpret_cast<const char *>(&rank_float4_stride), sizeof(s64));

    const s64 basis_bytes = node_count * rank_float4_stride * (s64)sizeof(float4);
    file.write(reinterpret_cast<const char *>(U_matrix.get_contents()), (streamsize)basis_bytes);
    file.write(reinterpret_cast<const char *>(V_matrix.get_contents()), (streamsize)basis_bytes);

    const s64 coefficient_bytes = matrix_count * rank_float4_stride * LANE_GROUP * (s64)sizeof(f32);
    file.write(reinterpret_cast<const char *>(coefficients.get_contents()), (streamsize)coefficient_bytes);

    // The updates not yet folded into the basis, one plane at a time.
    file.write(reinterpret_cast<const char *>(&sparse_delta_capacity), sizeof(s64));
    const Vector<Vector<Pair<s64, f32>>> deltas = current_deltas();
    for (const Vector<Pair<s64, f32>> &plane : deltas) {
        const s64 entry_count = (s64)plane.size();
        file.write(reinterpret_cast<const char *>(&entry_count), sizeof(s64));
        for (const Pair<s64, f32> &delta : plane) {
            file.write(reinterpret_cast<const char *>(&delta.first), sizeof(s64));
            file.write(reinterpret_cast<const char *>(&delta.second), sizeof(f32));
        }
    }

    log::logger().debug("WeightMatrix::save: {} node_count={} rank={}", filepath, node_count, rank);
}

void WeightMatrix::load_from_disk(const char *filepath) {
    ifstream file(filepath, ios::binary);
    u32 magic = 0;
    file.read(reinterpret_cast<char *>(&magic), sizeof(u32));
    if (magic != WEIGHT_MATRIX_SAVE_MAGIC) {
        log::throw_invalid_argument(log::logger(),
            String("WeightMatrix::load_from_disk: ") + filepath + " is not a weight matrix file this version can read");
    }

    s64 saved_node_count = 0;
    s64 saved_matrix_count = 0;
    s64 saved_rank = 0;
    s64 saved_rank_float4_stride = 0;
    file.read(reinterpret_cast<char *>(&saved_node_count), sizeof(s64));
    file.read(reinterpret_cast<char *>(&saved_matrix_count), sizeof(s64));
    file.read(reinterpret_cast<char *>(&saved_rank), sizeof(s64));
    file.read(reinterpret_cast<char *>(&saved_rank_float4_stride), sizeof(s64));

    if (saved_node_count != node_count || saved_matrix_count != matrix_count) {
        log::throw_invalid_argument(log::logger(),
            "WeightMatrix::load_from_disk: the file holds " + to_string(saved_node_count) + " nodes and " +
            to_string(saved_matrix_count) + " planes but this matrix has " + to_string(node_count) + " and " +
            to_string(matrix_count) + " -- the adjacency and planes have to match");
    }

    resize_basis(saved_rank);

    const s64 basis_bytes = node_count * rank_float4_stride * (s64)sizeof(float4);
    file.read(reinterpret_cast<char *>(U_matrix.get_contents()), (streamsize)basis_bytes);
    file.read(reinterpret_cast<char *>(V_matrix.get_contents()), (streamsize)basis_bytes);

    const s64 coefficient_bytes = matrix_count * rank_float4_stride * LANE_GROUP * (s64)sizeof(f32);
    file.read(reinterpret_cast<char *>(coefficients.get_contents()), (streamsize)coefficient_bytes);

    s64 saved_delta_capacity = 0;
    file.read(reinterpret_cast<char *>(&saved_delta_capacity), sizeof(s64));
    clear_sparse_deltas();
    resize_delta_capacity(saved_delta_capacity);
    for (s64 matrix_index = 0; matrix_index < matrix_count; matrix_index += 1) {
        s64 entry_count = 0;
        file.read(reinterpret_cast<char *>(&entry_count), sizeof(s64));
        Vector<Pair<s64, f32>> plane((usize)max<s64>(entry_count, 0));
        for (Pair<s64, f32> &delta : plane) {
            file.read(reinterpret_cast<char *>(&delta.first), sizeof(s64));
            file.read(reinterpret_cast<char *>(&delta.second), sizeof(f32));
        }
        rebuild_sparse_delta(matrix_index, plane);
    }

    log::logger().debug("WeightMatrix::load_from_disk: {} node_count={} rank={}", filepath, node_count, rank);
}

} // namespace spikecorec
