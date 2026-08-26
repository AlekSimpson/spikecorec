//
// Created by Alek Simpson on 5/30/26.
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <vector>

#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;

namespace {

constexpr u32 WEIGHT_MATRIX_SAVE_MAGIC = 0x574D5458;

s64 round_up_to_lane_group(s64 value) {
    const s64 group = WeightMatrix::LANE_GROUP;
    return ((max<s64>(value, 1) + group - 1) / group) * group;
}

// Solves (gram + ridge*I) x = right_hand_side in place, by Cholesky. The system is small --
// one per node, of dimension rank -- and symmetric positive definite once the ridge term is
// added, which is the whole reason the ridge is there rather than for regularisation alone:
// a node whose edges do not span the rank leaves the Gram matrix singular.
//
// Returns false when even the ridged system is not decomposable, in which case the caller
// leaves that row alone rather than writing a NaN into the basis.
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

    // Forward substitution, then back substitution.
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

// ── construction ──────────────────────────────────────────────────────────────────

WeightMatrix::WeightMatrix(
    EngineBackend &backend,
    const vector<vector<s32>> &network,
    s64 rank,
    bool check_indexing,
    s64 max_neighbor_count,
    s64 weight_seed,
    s64 sparse_delta_capacity
)
    : k2tree(*K2Tree::from_adjacency_list(backend, validate_network(network), (s32)network.size()))
    , sparse_delta_capacity(max<s64>(sparse_delta_capacity, 0))
    , owning_backend(&backend)
    , node_count((s64)network.size())
    , check_indexing(check_indexing)
{
    this->max_neighbor_count = max_neighbor_count;
    if (max_neighbor_count < 0) {
        s64 longest_row = 0;
        for (const vector<s32> &row : network) longest_row = max(longest_row, (s64)row.size());
        this->max_neighbor_count = longest_row;
    }

    // Provisional until declare_projections derives the real one. A caller that names a
    // rank outright gets it; -1 means "decide from the declarations", and four lanes is
    // the smallest basis that can hold anything at all.
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

    log::logger().debug("WeightMatrix constructed: node_count={} edges={} rank={} "
                        "max_neighbor_count={} plasticity_capacity={}",
                        node_count, total_edge_count, this->rank,
                        this->max_neighbor_count, this->sparse_delta_capacity);
}

// Walks every row once to number the edges. This is also where total_edge_count comes
// from, so the two can never disagree: an edge has an ordinal exactly when the walk found
// it, which keeps both consistent with an explicitly-truncating max_neighbor_count.
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

    Vector<EnginePointer> partitions;
    owning_backend
        ->partition(matrix_byte_size, EngineDatatype::FLOAT32X4, partitions)                 // U
        .partition(matrix_byte_size, EngineDatatype::FLOAT32X4, partitions)                  // V
        .partition((u64)MATRIX_COUNT * lane_count * sizeof(f32), EngineDatatype::FLOAT32, partitions)
        .partition((u64)(node_count + 1) * sizeof(s64), EngineDatatype::SIGNED64, partitions)
        // Sized by edges, not node_count * max_neighbor_count: a padded scratch here would
        // reintroduce exactly the allocation this class exists to avoid, and it would be
        // resident for the object's whole life.
        .partition((u64)total_edge_count * sizeof(f32), EngineDatatype::FLOAT32, partitions)
        // The sparse correction layer. Row starts always exist -- an empty CSR is all
        // zeros, which is a valid "no corrections anywhere" rather than an absent buffer
        // the read path has to test for.
        .partition((u64)MATRIX_COUNT * (u64)(node_count + 1) * sizeof(s32),
                   EngineDatatype::SIGNED32, partitions)
        .partition((u64)MATRIX_COUNT * (u64)sparse_delta_capacity * sizeof(s64),
                   EngineDatatype::SIGNED64, partitions)
        .partition((u64)MATRIX_COUNT * (u64)sparse_delta_capacity * sizeof(f32),
                   EngineDatatype::FLOAT32, partitions)
        .partition((u64)sparse_delta_capacity * sizeof(s64), EngineDatatype::SIGNED64, partitions)
        .partition((u64)sparse_delta_capacity * sizeof(f32), EngineDatatype::FLOAT32, partitions)
        .partition(sparse_delta_capacity > 0 ? sizeof(s32) : 0, EngineDatatype::SIGNED32, partitions);

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
    pending_delta_count = partitions[10];

    memcpy(edge_row_offset.get_contents(), edge_row_offset_host.data(),
           ((usize)node_count + 1) * sizeof(s64));

    if (!sparse_delta_row_start.is_empty()) {
        memset(sparse_delta_row_start.get_contents(), 0,
               (usize)MATRIX_COUNT * ((usize)node_count + 1) * sizeof(s32));
    }
    sparse_delta_entry_count.assign((usize)MATRIX_COUNT, 0);
    if (sparse_delta_capacity > 0) {
        *pending_delta_count.get_contents_as<s32>() = 0;
    }

    owning_backend->advise_read_mostly(edge_row_offset, (u64)(node_count + 1) * sizeof(s64));
    owning_backend->prefetch_to_gpu(U_matrix, matrix_byte_size);
    owning_backend->prefetch_to_gpu(V_matrix, matrix_byte_size);
}

// Releases the current chunk and runs a fresh partition round at the new rank. Used when
// declare_projections derives a rank the constructor could not know, and when a loaded
// file disagrees with the in-memory shape.
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

    // A fresh slab is uninitialised memory, and alternating least squares started from
    // whatever was there converges to nothing -- it was reading the previous allocation's
    // bytes as a starting basis. Seed it the same way the constructor does.
    seed_basis(basis_seed);
}

// Independent N(0,1) in every lane. Not a fit: it is the starting point one becomes, and
// what a matrix used as a random reservoir keeps. The seed is held so a resize reproduces
// the same starting basis rather than drifting on re-allocation.
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

    // Both coefficient rows start neutral, so an unfitted basis reconstructs the plain
    // dot(U, V) it always did.
    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    f32 *coefficient_data = coefficients.get_contents_as<f32>();
    for (s64 lane_index = 0; lane_index < MATRIX_COUNT * lane_count; lane_index += 1) {
        coefficient_data[lane_index] = 1.0f;
    }
}

WeightMatrix::WeightMatrix(WeightMatrix &&other) noexcept
    : k2tree(std::move(other.k2tree))
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
    , pending_delta_count(other.pending_delta_count)
    , sparse_delta_capacity(other.sparse_delta_capacity)
    , projection_first_edge_ordinal(std::move(other.projection_first_edge_ordinal))
    , projection_edge_count(std::move(other.projection_edge_count))
    , projection_synapse_prototype(std::move(other.projection_synapse_prototype))
    , owning_backend(other.owning_backend)
    , owning_slab(other.owning_slab)
    , node_count(other.node_count)
    , total_edge_count(other.total_edge_count)
    , max_neighbor_count(other.max_neighbor_count)
    , rank(other.rank)
    , rank_float4_stride(other.rank_float4_stride)
    , constant_weight(other.constant_weight)
    , using_constant_weight(other.using_constant_weight)
    , constant_delay_ticks(other.constant_delay_ticks)
    , using_constant_delay_ticks(other.using_constant_delay_ticks)
    , check_indexing(other.check_indexing)
    , measured_weight_fit_error(other.measured_weight_fit_error)
    , edge_row_offset_host(std::move(other.edge_row_offset_host))
{
    // The only reason this is not defaulted: the source must forget the slab, or both
    // objects release it.
    other.owning_backend = nullptr;
    other.owning_slab = EnginePointer{};
    other.node_count = 0;
    other.total_edge_count = 0;
}

WeightMatrix &WeightMatrix::operator=(WeightMatrix &&other) noexcept {
    if (this == &other) return *this;

    if (owning_backend != nullptr) owning_backend->deallocate_slab(owning_slab);

    k2tree = std::move(other.k2tree);
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
    pending_delta_count = other.pending_delta_count;
    sparse_delta_entry_count = other.sparse_delta_entry_count;
    sparse_delta_capacity = other.sparse_delta_capacity;
    projection_first_edge_ordinal = std::move(other.projection_first_edge_ordinal);
    projection_edge_count = std::move(other.projection_edge_count);
    projection_synapse_prototype = std::move(other.projection_synapse_prototype);
    owning_backend = other.owning_backend;
    owning_slab = other.owning_slab;
    node_count = other.node_count;
    total_edge_count = other.total_edge_count;
    max_neighbor_count = other.max_neighbor_count;
    rank = other.rank;
    rank_float4_stride = other.rank_float4_stride;
    constant_weight = other.constant_weight;
    using_constant_weight = other.using_constant_weight;
    constant_delay_ticks = other.constant_delay_ticks;
    using_constant_delay_ticks = other.using_constant_delay_ticks;
    check_indexing = other.check_indexing;
    measured_weight_fit_error = other.measured_weight_fit_error;
    edge_row_offset_host = std::move(other.edge_row_offset_host);

    other.owning_backend = nullptr;
    other.owning_slab = EnginePointer{};
    other.node_count = 0;
    other.total_edge_count = 0;
    return *this;
}

WeightMatrix::~WeightMatrix() {
    if (owning_backend != nullptr) owning_backend->deallocate_slab(owning_slab);
}

// ── indexing ──────────────────────────────────────────────────────────────────────

bool WeightMatrix::check_index_inbounds(s32 node_index) const {
    return check_indexing && node_index >= 0 && node_index < node_count;
}

bool WeightMatrix::check_index_inbounds(s32 source, s32 target) const {
    return check_index_inbounds(source) && check_index_inbounds(target);
}

void WeightMatrix::validate_matrix_index(s64 matrix_index) const {
    if (matrix_index < 0 || matrix_index >= MATRIX_COUNT) {
        log::throw_invalid_argument(log::logger(),
            "WeightMatrix: matrix_index " + to_string(matrix_index) + " out of range [0, " +
            to_string((s64)MATRIX_COUNT) + ")");
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
    return k2tree.get_predecessors((s32)node_index, output_buffer, max_neighbor_count);
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

// ── reconstruction ────────────────────────────────────────────────────────────────

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

// The correction Sk holds for one edge, or zero. Binary search inside the source row's
// slice, which is what the CSR layout buys: rows are contiguous and short.
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

// Basis plus correction. The basis is a lossy projection and Sk is what the rank did not
// capture, so neither half is the value on its own -- every read path goes through here.
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
    return get_for_matrix(source_node, target_node, DEFAULT_MATRIX_INDEX);
}

s32 WeightMatrix::get_edge_delay_ticks(s32 source_node, s32 target_node) const {
    if (using_constant_delay_ticks) return constant_delay_ticks;
    if (!check_index_inbounds(source_node, target_node)) return constant_delay_ticks;

    // A delay is a whole number of ticks, so rounding absorbs fit error the way it cannot
    // for a weight: the basis only has to land within half a tick. That slack is why the
    // delay matrix usually needs no corrections at all even when the weights do.
    const f32 corrected = get_for_matrix(source_node, target_node, DELAY_MATRIX_INDEX);
    return max<s32>((s32)lroundf(corrected), 1);
}

s32 WeightMatrix::get_edge_synapse_prototype(s32 source_node, s32 target_node) const {
    const optional<s64> ordinal = edge_ordinal(source_node, target_node);
    if (!ordinal.has_value()) return -1;

    // Runs are sorted and contiguous over the edge ordering, so the run containing an
    // ordinal is the last one starting at or below it.
    const auto entry = upper_bound(projection_first_edge_ordinal.begin(),
                                   projection_first_edge_ordinal.end(), *ordinal);
    if (entry == projection_first_edge_ordinal.begin()) return -1;

    const usize run_index = (usize)(entry - projection_first_edge_ordinal.begin() - 1);
    if (*ordinal >= projection_first_edge_ordinal[run_index] + projection_edge_count[run_index]) {
        return -1;
    }
    return projection_synapse_prototype[run_index];
}

// ── the sparse delta matrix ───────────────────────────────────────────────────────

// Rebuilds one matrix's CSR from a full correction list. When there are more corrections
// than capacity the largest by magnitude are kept: truncating by size makes the residual
// error bounded by the smallest kept correction, where truncating arbitrarily would leave
// the worst edges wrong.
void WeightMatrix::rebuild_sparse_delta(s64 matrix_index, Vector<Pair<s64, f32>> &corrections) {
    validate_matrix_index(matrix_index);
    if (sparse_delta_capacity <= 0 || sparse_delta_row_start.is_empty()) return;

    if ((s64)corrections.size() > sparse_delta_capacity) {
        nth_element(corrections.begin(), corrections.begin() + sparse_delta_capacity,
                    corrections.end(),
                    [](const Pair<s64, f32> &left, const Pair<s64, f32> &right) {
                        return fabsf(left.second) > fabsf(right.second);
                    });
        corrections.resize((usize)sparse_delta_capacity);
    }

    // The CSR wants entries ascending by ordinal, and ordinals are already grouped by
    // source row -- so one sort puts both the rows and their contents in order.
    sort(corrections.begin(), corrections.end(),
         [](const Pair<s64, f32> &left, const Pair<s64, f32> &right) {
             return left.first < right.first;
         });

    s32 *row_start = sparse_delta_row_start.get_contents_as<s32>() +
                     matrix_index * (node_count + 1);
    s64 *entry_ordinal = sparse_delta_edge_ordinal.get_contents_as<s64>() +
                        matrix_index * sparse_delta_capacity;
    f32 *entry_value = sparse_delta_value.get_contents_as<f32>() +
                       matrix_index * sparse_delta_capacity;

    for (usize index = 0; index < corrections.size(); index += 1) {
        entry_ordinal[index] = corrections[index].first;
        entry_value[index] = corrections[index].second;
    }

    // Row starts from the prefix sum of how many corrections fall in each row. An edge's
    // row is the one whose edge_row_offset range contains its ordinal.
    s64 correction_index = 0;
    for (s64 node_index = 0; node_index <= node_count; node_index += 1) {
        const s64 row_first_ordinal = edge_row_offset_host[(usize)node_index];
        while (correction_index < (s64)corrections.size() &&
               corrections[(usize)correction_index].first < row_first_ordinal) {
            correction_index += 1;
        }
        row_start[node_index] = (s32)correction_index;
    }

    sparse_delta_entry_count[(usize)matrix_index] = (s64)corrections.size();
}

void WeightMatrix::accumulate_edge_delta(
    s64 matrix_index, s32 source_node, s32 target_node, f32 delta
) {
    validate_matrix_index(matrix_index);
    if (sparse_delta_capacity <= 0) return;

    const optional<s64> ordinal = edge_ordinal(source_node, target_node);
    if (!ordinal.has_value()) {
        log::throw_invalid_argument(log::logger(),
            "WeightMatrix::accumulate_edge_delta: (" + to_string(source_node) + ", " +
            to_string(target_node) + ") is not an edge, so it has nothing to correct");
    }

    // Read, add, rebuild. Correct and simple; the device path queues into the pending
    // buffer instead precisely because doing this per update is what batching avoids.
    Vector<Pair<s64, f32>> corrections;
    const s64 existing_count = sparse_delta_entry_count[(usize)matrix_index];
    const s64 *entry_ordinal = sparse_delta_edge_ordinal.get_contents_as<s64>() +
                              matrix_index * sparse_delta_capacity;
    const f32 *entry_value = sparse_delta_value.get_contents_as<f32>() +
                             matrix_index * sparse_delta_capacity;

    bool merged = false;
    for (s64 index = 0; index < existing_count; index += 1) {
        const f32 value = (entry_ordinal[index] == *ordinal) ? entry_value[index] + delta
                                                             : entry_value[index];
        if (entry_ordinal[index] == *ordinal) merged = true;
        corrections.push_back({entry_ordinal[index], value});
    }
    if (!merged) corrections.push_back({*ordinal, delta});

    rebuild_sparse_delta(matrix_index, corrections);
}

// Merges what the device staged this interval into the CSR. Until this runs, those updates
// are queued but not yet visible to a read -- which is the batching latency the design
// accepts in exchange for not sorting on every update.
void WeightMatrix::compact_pending_deltas() {
    if (sparse_delta_capacity <= 0 || pending_delta_count.is_empty()) return;

    s32 *pending_count = pending_delta_count.get_contents_as<s32>();
    const s64 staged = min<s64>((s64)*pending_count, sparse_delta_capacity);
    if (staged <= 0) {
        *pending_count = 0;
        return;
    }

    if ((s64)*pending_count > sparse_delta_capacity) {
        log::logger().warn("compact_pending_deltas: {} updates were dropped this interval "
                           "(capacity {}); compact more often",
                           (s64)*pending_count - sparse_delta_capacity, sparse_delta_capacity);
    }

    const s64 *staged_ordinal = pending_delta_edge_ordinal.get_contents_as<s64>();
    const f32 *staged_value = pending_delta_value.get_contents_as<f32>();

    // Corrections go to the weight matrix: an update changes what an edge is worth.
    Vector<Pair<s64, f32>> corrections;
    const s64 existing_count = sparse_delta_entry_count[(usize)DEFAULT_MATRIX_INDEX];
    const s64 *entry_ordinal = sparse_delta_edge_ordinal.get_contents_as<s64>() +
                              DEFAULT_MATRIX_INDEX * sparse_delta_capacity;
    const f32 *entry_value = sparse_delta_value.get_contents_as<f32>() +
                             DEFAULT_MATRIX_INDEX * sparse_delta_capacity;
    for (s64 index = 0; index < existing_count; index += 1) {
        corrections.push_back({entry_ordinal[index], entry_value[index]});
    }
    for (s64 index = 0; index < staged; index += 1) {
        corrections.push_back({staged_ordinal[index], staged_value[index]});
    }

    // Sum duplicates rather than letting the later one win: two arrivals on one edge in an
    // interval are two updates, not a correction and a replacement.
    sort(corrections.begin(), corrections.end(),
         [](const Pair<s64, f32> &left, const Pair<s64, f32> &right) {
             return left.first < right.first;
         });
    Vector<Pair<s64, f32>> merged;
    for (const Pair<s64, f32> &correction : corrections) {
        if (!merged.empty() && merged.back().first == correction.first) {
            merged.back().second += correction.second;
        } else {
            merged.push_back(correction);
        }
    }

    rebuild_sparse_delta(DEFAULT_MATRIX_INDEX, merged);
    *pending_count = 0;

    log::logger().debug("compact_pending_deltas: merged {} updates, Sk now holds {} entries",
                        staged, sparse_delta_entry_count[(usize)DEFAULT_MATRIX_INDEX]);
}

f32 WeightMatrix::sparse_delta_occupancy_fraction() const {
    if (total_edge_count <= 0) return 0.0f;

    s64 worst = 0;
    for (s64 count : sparse_delta_entry_count) worst = max(worst, count);
    return (f32)worst / (f32)total_edge_count;
}

bool WeightMatrix::is_refit_due() const {
    if (refit_occupancy_threshold_fraction <= 0.0f) return false;
    return sparse_delta_occupancy_fraction() >= refit_occupancy_threshold_fraction;
}

// ── declaring the network's values ────────────────────────────────────────────────

void WeightMatrix::declare_projections(
    const Vector<s64> &first_edge_ordinal,
    const Vector<s64> &edge_count,
    const Vector<s32> &synapse_prototype,
    const Vector<f32> &weight,
    const Vector<s32> &delay_ticks
) {
    const usize run_count = first_edge_ordinal.size();
    if (edge_count.size() != run_count || synapse_prototype.size() != run_count ||
        weight.size() != run_count || delay_ticks.size() != run_count) {
        log::throw_invalid_argument(log::logger(),
            "WeightMatrix::declare_projections: the five arrays must be parallel");
    }

    projection_first_edge_ordinal = first_edge_ordinal;
    projection_edge_count = edge_count;
    projection_synapse_prototype = synapse_prototype;

    if (run_count == 0 || total_edge_count == 0) {
        log::logger().debug("declare_projections: nothing to declare ({} runs, {} edges)",
                            run_count, total_edge_count);
        return;
    }

    // One run means one value for the whole network, which needs no basis at all.
    bool every_delay_matches = true;
    for (usize run_index = 1; run_index < run_count; run_index += 1) {
        if (delay_ticks[run_index] != delay_ticks[0]) every_delay_matches = false;
    }
    // Decided here rather than inside either fit, because it is a property of what was
    // declared. Leaving it set when the delays vary makes get_edge_delay_ticks answer the
    // constant for every edge and ignore the basis entirely -- silently, and only for
    // models whose delays are not uniform.
    if (every_delay_matches) {
        set_constant_delay_ticks(max<s32>(delay_ticks[0], 1));
    } else {
        using_constant_delay_ticks = false;
    }

    const Vector<Vector<f32>> targets = targets_from_projections(weight, delay_ticks);

    // Two ways to get a basis. When the runs fit in the lane budget the indicator
    // construction is exact and free, so take it. Otherwise fit -- a projection onto a
    // linear subspace, approximate by design, with the residual left for Sk.
    if (!fit_basis_from_projections(weight, delay_ticks)) {
        // Rank from the dimension count: MATRIX_COUNT * total_edge_count constraints against
        // 2 * node_count * rank parameters. That is the rank at which an ARBITRARY field
        // becomes representable, so it is a ceiling -- capped again by what the kernel can
        // index, and by a default budget, because spending the ceiling on every model would
        // give up the compression this exists for.
        const s64 dimension_count_rank =
                (MATRIX_COUNT * total_edge_count) / max<s64>(2 * node_count, 1) + 1;
        const s64 budget_rank = min<s64>(dimension_count_rank, DEFAULT_FIT_RANK_BUDGET);
        resize_basis(min<s64>(budget_rank, MAX_RANK_FLOAT4_STRIDE * LANE_GROUP));

        log::logger().debug("declare_projections: {} runs exceed the lane budget; fitting at "
                            "rank {} instead of one lane per run", run_count, rank);
        fit_basis_to_targets(targets, DEFAULT_FIT_SWEEP_COUNT, DEFAULT_FIT_RIDGE);
    }

    // Whatever the basis did not capture goes into Sk, which is what turns an approximate
    // projection into an accurate read. An exact fit leaves it empty; an approximate one
    // leaves corrections behind, and refit() folds them back in once there are enough.
    const s64 weight_corrections =
            store_residual_corrections(DEFAULT_MATRIX_INDEX, targets[(usize)DEFAULT_MATRIX_INDEX]);
    const s64 delay_corrections =
            every_delay_matches
                    ? 0
                    : store_residual_corrections(DELAY_MATRIX_INDEX,
                                                 targets[(usize)DELAY_MATRIX_INDEX]);

    measured_weight_fit_error = measure_worst_relative_weight_error(weight);

    if (measured_weight_fit_error > WEIGHT_FIT_WARNING_TOLERANCE) {
        // Reported, not refused. The basis is a lossy projection by design and the residual
        // is the accepted price of the storage -- what matters is that the number is visible
        // rather than that it is zero.
        log::logger().warn("WeightMatrix: after corrections the worst declared weight is still "
                           "off by {:.3e} relative, past the {:.0e} line. Raise the rank or "
                           "the correction capacity if that matters for this model.",
                           measured_weight_fit_error, WEIGHT_FIT_WARNING_TOLERANCE);
    }

    log::logger().info("WeightMatrix: {} projections over {} edges at rank {} -- basis {} bytes, "
                       "{} weight and {} delay corrections held ({:.1f}% of edges), worst "
                       "weight error {:.3e}",
                       run_count, total_edge_count, rank,
                       2 * node_count * rank * (s64)sizeof(f32),
                       weight_corrections, delay_corrections,
                       100.0 * sparse_delta_occupancy_fraction(), measured_weight_fit_error);
}

// Walks the edge set, compares the basis against the targets, and hands whatever differs
// to Sk. An edge the basis already reproduces needs no entry, which is what keeps Sk empty
// for a projection-structured network rather than merely small.
//
// Delay gets half a tick of slack, because rounding absorbs anything smaller: a basis that
// lands within 0.4 of the right integer already reads back correctly, and an entry for it
// would be storage bought for nothing.
s64 WeightMatrix::store_residual_corrections(
    s64 matrix_index, const Vector<f32> &targets_by_edge_ordinal
) {
    if (sparse_delta_capacity <= 0 || total_edge_count == 0) return 0;
    if ((s64)targets_by_edge_ordinal.size() < total_edge_count) return 0;

    const f32 *coefficient_values = coefficient_row(matrix_index);
    const f32 negligible_residual =
            (matrix_index == DELAY_MATRIX_INDEX) ? 0.4f : 0.0f;

    vector<s32> neighbor_buffer((usize)max<s64>(max_neighbor_count, 1));
    Vector<Pair<s64, f32>> corrections;

    for (s64 source_node = 0; source_node < node_count; source_node += 1) {
        const s64 degree = k2tree.get_neighbors((s32)source_node, neighbor_buffer.data(),
                                                max_neighbor_count);
        for (s64 slot = 0; slot < degree; slot += 1) {
            const s64 ordinal = edge_row_offset_host[(usize)source_node] + slot;
            const f32 reconstructed = reconstruct_entry((s32)source_node,
                                                        neighbor_buffer[(usize)slot],
                                                        coefficient_values);
            const f32 residual = targets_by_edge_ordinal[(usize)ordinal] - reconstructed;
            if (fabsf(residual) > negligible_residual) corrections.push_back({ordinal, residual});
        }
    }

    rebuild_sparse_delta(matrix_index, corrections);
    return sparse_delta_entry_count[(usize)matrix_index];
}

// Builds U, V and both coefficient rows straight from the projection structure rather than
// fitting numerically, and is EXACT for a network wired out of population-to-population
// projections -- which is what NeuroML's <projection presynapticPopulation=
// postsynapticPopulation=> gives.
//
// The construction: give run k its own latent lane. U[i][k] is 1 when node i is a source of
// run k and 0 otherwise; V[j][k] is 1 when node j is a target of run k. Then Ck[k] holds
// the run's value, and Σ_m U[i][m]·Ck[m]·V[j][m] collapses to exactly Ck[k] for any edge
// (i, j) in run k.
//
// Why the other lanes vanish: lane m is nonzero at U[i] only if i is a source of run m, and
// nonzero at V[j] only if j is a target of run m. Populations are disjoint, so a node
// belongs to one source population and one target population -- meaning both can only hold
// for m == k. Where that disjointness does not hold, the measurement in
// declare_projections is what catches it.
bool WeightMatrix::fit_basis_from_projections(
    const Vector<f32> &weight, const Vector<s32> &delay_ticks
) {
    const s64 run_count = (s64)projection_first_edge_ordinal.size();

    // Only available while every run can have its own lane. Past that the construction has
    // no exactness left to offer and the general fit is the honest path.
    if (round_up_to_lane_group(run_count) > MAX_RANK_FLOAT4_STRIDE * LANE_GROUP) return false;

    resize_basis(run_count);

    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    f32 *u_data = U_matrix.get_contents_as<f32>();
    f32 *v_data = V_matrix.get_contents_as<f32>();

    memset(u_data, 0, (usize)node_count * (usize)lane_count * sizeof(f32));
    memset(v_data, 0, (usize)node_count * (usize)lane_count * sizeof(f32));

    f32 *weight_coefficients = coefficient_row(DEFAULT_MATRIX_INDEX);
    f32 *delay_coefficients = coefficient_row(DELAY_MATRIX_INDEX);
    for (s64 lane_index = 0; lane_index < lane_count; lane_index += 1) {
        const bool lane_is_a_run = lane_index < run_count;
        weight_coefficients[lane_index] = lane_is_a_run ? weight[(usize)lane_index] : 0.0f;
        delay_coefficients[lane_index] =
                lane_is_a_run ? (f32)max<s32>(delay_ticks[(usize)lane_index], 1) : 0.0f;
    }

    // One pass over every edge, marking the endpoints of the run it belongs to.
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

            u_data[source_node * lane_count + run_index] = 1.0f;
            v_data[neighbor_buffer[(usize)slot] * lane_count + run_index] = 1.0f;
        }
    }

    using_constant_weight = false;
    using_constant_delay_ticks = using_constant_delay_ticks && run_count <= 1;
    return true;
}

// ── fitting the basis ─────────────────────────────────────────────────────────────

Vector<Vector<f32>> WeightMatrix::targets_from_projections(
    const Vector<f32> &weight, const Vector<s32> &delay_ticks
) const {
    Vector<Vector<f32>> targets((usize)MATRIX_COUNT,
                                Vector<f32>((usize)max<s64>(total_edge_count, 0), 0.0f));
    const s64 run_count = (s64)projection_first_edge_ordinal.size();
    if (run_count == 0) return targets;

    for (s64 ordinal = 0; ordinal < total_edge_count; ordinal += 1) {
        // Runs are sorted and contiguous, so the run containing an ordinal is the last one
        // starting at or below it.
        const auto entry = upper_bound(projection_first_edge_ordinal.begin(),
                                       projection_first_edge_ordinal.end(), ordinal);
        const s64 run_index = max<s64>(entry - projection_first_edge_ordinal.begin() - 1, 0);

        targets[(usize)DEFAULT_MATRIX_INDEX][(usize)ordinal] = weight[(usize)run_index];
        targets[(usize)DELAY_MATRIX_INDEX][(usize)ordinal] =
                (f32)max<s32>(delay_ticks[(usize)run_index], 1);
    }
    return targets;
}

Vector<Vector<f32>> WeightMatrix::targets_from_current_values() const {
    Vector<Vector<f32>> targets((usize)MATRIX_COUNT,
                                Vector<f32>((usize)max<s64>(total_edge_count, 0), 0.0f));
    if (total_edge_count == 0) return targets;

    vector<s32> neighbor_buffer((usize)max<s64>(max_neighbor_count, 1));
    for (s64 source_node = 0; source_node < node_count; source_node += 1) {
        const s64 degree = k2tree.get_neighbors((s32)source_node, neighbor_buffer.data(),
                                                max_neighbor_count);
        for (s64 slot = 0; slot < degree; slot += 1) {
            const s64 ordinal = edge_row_offset_host[(usize)source_node] + slot;
            const s32 target_node = neighbor_buffer[(usize)slot];
            for (s64 matrix_index = 0; matrix_index < MATRIX_COUNT; matrix_index += 1) {
                // Reconstruction PLUS correction: the corrected value is what this matrix
                // currently reads, and re-optimising the basis toward it is exactly how a
                // correction gets absorbed into the basis and stops needing an entry.
                targets[(usize)matrix_index][(usize)ordinal] =
                        reconstruct_entry((s32)source_node, target_node,
                                          coefficient_row(matrix_index)) +
                        sparse_delta_for(matrix_index, (s32)source_node, ordinal);
            }
        }
    }
    return targets;
}

// Alternating least squares over the edge support. Three phases per sweep -- solve every
// U row, then every V row, then every coefficient row -- each of which is a small dense
// least-squares problem in `rank` unknowns.
//
// The support is what keeps this affordable: a row of U is fitted against that node's own
// out-edges, not against a whole matrix row, so a sweep costs O(total_edge_count * rank^2)
// rather than O(node_count^2 * rank^2). At a million nodes that is the difference between
// a construction step and an impossibility.
void WeightMatrix::fit_basis_to_targets(
    const Vector<Vector<f32>> &targets_per_matrix, s32 sweep_count, f32 ridge_regularization
) {
    if (total_edge_count == 0 || node_count == 0) return;

    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    f32 *u_data = U_matrix.get_contents_as<f32>();
    f32 *v_data = V_matrix.get_contents_as<f32>();

    // Every matrix shares one basis, and least squares minimises ABSOLUTE error -- so a
    // matrix whose values are numerically larger dominates the objective and the others are
    // fitted to whatever is left. Delay in ticks runs 10-40 while a weight is order 1, which
    // is enough for the weight field to be neglected entirely: a constant weight of 1.0,
    // representable at rank 1, came back with 100% error.
    //
    // Normalising each matrix by its own scale makes them contribute comparably. The scale
    // is folded back into that matrix's coefficient row afterwards, so the reconstruction
    // still lands on the original magnitude.
    Vector<f64> matrix_scale((usize)MATRIX_COUNT, 1.0);
    for (s64 matrix_index = 0; matrix_index < MATRIX_COUNT; matrix_index += 1) {
        f64 sum_of_squares = 0.0;
        for (s64 ordinal = 0; ordinal < total_edge_count; ordinal += 1) {
            const f64 value = (f64)targets_per_matrix[(usize)matrix_index][(usize)ordinal];
            sum_of_squares += value * value;
        }
        const f64 root_mean_square = sqrt(sum_of_squares / (f64)total_edge_count);
        matrix_scale[(usize)matrix_index] = (root_mean_square > 0.0) ? root_mean_square : 1.0;
    }

    Vector<Vector<f32>> normalised_targets = targets_per_matrix;
    for (s64 matrix_index = 0; matrix_index < MATRIX_COUNT; matrix_index += 1) {
        const f64 scale = matrix_scale[(usize)matrix_index];
        for (s64 ordinal = 0; ordinal < total_edge_count; ordinal += 1) {
            normalised_targets[(usize)matrix_index][(usize)ordinal] =
                    (f32)((f64)targets_per_matrix[(usize)matrix_index][(usize)ordinal] / scale);
        }
    }

    // Flat edge list, built once: every phase walks it, and re-walking the k^2-tree per
    // sweep would dominate the arithmetic it is there to serve.
    vector<s32> edge_source((usize)total_edge_count);
    vector<s32> edge_target((usize)total_edge_count);
    {
        vector<s32> neighbor_buffer((usize)max<s64>(max_neighbor_count, 1));
        for (s64 source_node = 0; source_node < node_count; source_node += 1) {
            const s64 degree = k2tree.get_neighbors((s32)source_node, neighbor_buffer.data(),
                                                    max_neighbor_count);
            for (s64 slot = 0; slot < degree; slot += 1) {
                const s64 ordinal = edge_row_offset_host[(usize)source_node] + slot;
                edge_source[(usize)ordinal] = (s32)source_node;
                edge_target[(usize)ordinal] = neighbor_buffer[(usize)slot];
            }
        }
    }

    // Incoming edges per node, so the V phase can gather its own equations without a
    // second tree walk per sweep.
    vector<vector<s64>> incoming_edges((usize)node_count);
    for (s64 ordinal = 0; ordinal < total_edge_count; ordinal += 1) {
        incoming_edges[(usize)edge_target[(usize)ordinal]].push_back(ordinal);
    }

    vector<f64> gram((usize)(lane_count * lane_count));
    vector<f64> right_hand_side((usize)lane_count);
    vector<f64> basis_row((usize)lane_count);

    for (s32 sweep = 0; sweep < sweep_count; sweep += 1) {
        // ── U rows ───────────────────────────────────────────────────────────────
        for (s64 source_node = 0; source_node < node_count; source_node += 1) {
            const s64 first = edge_row_offset_host[(usize)source_node];
            const s64 last = edge_row_offset_host[(usize)source_node + 1];
            if (first == last) continue;

            fill(gram.begin(), gram.end(), 0.0);
            fill(right_hand_side.begin(), right_hand_side.end(), 0.0);

            for (s64 ordinal = first; ordinal < last; ordinal += 1) {
                const s64 target_node = edge_target[(usize)ordinal];
                for (s64 matrix_index = 0; matrix_index < MATRIX_COUNT; matrix_index += 1) {
                    const f32 *coefficient_values = coefficient_row(matrix_index);
                    for (s64 lane = 0; lane < lane_count; lane += 1) {
                        basis_row[(usize)lane] = (f64)coefficient_values[lane] *
                                                 (f64)v_data[target_node * lane_count + lane];
                    }
                    const f64 target =
                            (f64)normalised_targets[(usize)matrix_index][(usize)ordinal];
                    for (s64 row = 0; row < lane_count; row += 1) {
                        right_hand_side[(usize)row] += basis_row[(usize)row] * target;
                        for (s64 column = 0; column <= row; column += 1) {
                            gram[(usize)(row * lane_count + column)] +=
                                    basis_row[(usize)row] * basis_row[(usize)column];
                        }
                    }
                }
            }
            for (s64 row = 0; row < lane_count; row += 1) {
                for (s64 column = row + 1; column < lane_count; column += 1) {
                    gram[(usize)(row * lane_count + column)] =
                            gram[(usize)(column * lane_count + row)];
                }
            }

            if (solve_symmetric_in_place(gram, right_hand_side, lane_count,
                                         (f64)ridge_regularization)) {
                for (s64 lane = 0; lane < lane_count; lane += 1) {
                    u_data[source_node * lane_count + lane] = (f32)right_hand_side[(usize)lane];
                }
            }
        }

        // ── V rows ───────────────────────────────────────────────────────────────
        for (s64 target_node = 0; target_node < node_count; target_node += 1) {
            const vector<s64> &ordinals = incoming_edges[(usize)target_node];
            if (ordinals.empty()) continue;

            fill(gram.begin(), gram.end(), 0.0);
            fill(right_hand_side.begin(), right_hand_side.end(), 0.0);

            for (s64 ordinal : ordinals) {
                const s64 source_node = edge_source[(usize)ordinal];
                for (s64 matrix_index = 0; matrix_index < MATRIX_COUNT; matrix_index += 1) {
                    const f32 *coefficient_values = coefficient_row(matrix_index);
                    for (s64 lane = 0; lane < lane_count; lane += 1) {
                        basis_row[(usize)lane] = (f64)coefficient_values[lane] *
                                                 (f64)u_data[source_node * lane_count + lane];
                    }
                    const f64 target =
                            (f64)normalised_targets[(usize)matrix_index][(usize)ordinal];
                    for (s64 row = 0; row < lane_count; row += 1) {
                        right_hand_side[(usize)row] += basis_row[(usize)row] * target;
                        for (s64 column = 0; column <= row; column += 1) {
                            gram[(usize)(row * lane_count + column)] +=
                                    basis_row[(usize)row] * basis_row[(usize)column];
                        }
                    }
                }
            }
            for (s64 row = 0; row < lane_count; row += 1) {
                for (s64 column = row + 1; column < lane_count; column += 1) {
                    gram[(usize)(row * lane_count + column)] =
                            gram[(usize)(column * lane_count + row)];
                }
            }

            if (solve_symmetric_in_place(gram, right_hand_side, lane_count,
                                         (f64)ridge_regularization)) {
                for (s64 lane = 0; lane < lane_count; lane += 1) {
                    v_data[target_node * lane_count + lane] = (f32)right_hand_side[(usize)lane];
                }
            }
        }

        // ── coefficient rows ─────────────────────────────────────────────────────
        // One per matrix, over every edge. This is what lets the matrices differ while
        // sharing a basis: U and V hold the structure, Ck holds each quantity's scaling
        // of it.
        for (s64 matrix_index = 0; matrix_index < MATRIX_COUNT; matrix_index += 1) {
            fill(gram.begin(), gram.end(), 0.0);
            fill(right_hand_side.begin(), right_hand_side.end(), 0.0);

            for (s64 ordinal = 0; ordinal < total_edge_count; ordinal += 1) {
                const s64 source_node = edge_source[(usize)ordinal];
                const s64 target_node = edge_target[(usize)ordinal];
                for (s64 lane = 0; lane < lane_count; lane += 1) {
                    basis_row[(usize)lane] = (f64)u_data[source_node * lane_count + lane] *
                                             (f64)v_data[target_node * lane_count + lane];
                }
                const f64 target = (f64)normalised_targets[(usize)matrix_index][(usize)ordinal];
                for (s64 row = 0; row < lane_count; row += 1) {
                    right_hand_side[(usize)row] += basis_row[(usize)row] * target;
                    for (s64 column = 0; column <= row; column += 1) {
                        gram[(usize)(row * lane_count + column)] +=
                                basis_row[(usize)row] * basis_row[(usize)column];
                    }
                }
            }
            for (s64 row = 0; row < lane_count; row += 1) {
                for (s64 column = row + 1; column < lane_count; column += 1) {
                    gram[(usize)(row * lane_count + column)] =
                            gram[(usize)(column * lane_count + row)];
                }
            }

            if (solve_symmetric_in_place(gram, right_hand_side, lane_count,
                                         (f64)ridge_regularization)) {
                f32 *coefficient_values = coefficient_row(matrix_index);
                for (s64 lane = 0; lane < lane_count; lane += 1) {
                    coefficient_values[lane] = (f32)right_hand_side[(usize)lane];
                }
            }
        }
    }

    // Undo the normalisation. Ck scales the reconstruction linearly, so putting each
    // matrix's own scale back here lands it on the magnitude its targets actually had --
    // while the fit above got to treat every matrix as equally important.
    for (s64 matrix_index = 0; matrix_index < MATRIX_COUNT; matrix_index += 1) {
        f32 *coefficient_values = coefficient_row(matrix_index);
        for (s64 lane = 0; lane < lane_count; lane += 1) {
            coefficient_values[lane] =
                    (f32)((f64)coefficient_values[lane] * matrix_scale[(usize)matrix_index]);
        }
    }

    using_constant_weight = false;
}

void WeightMatrix::refit(s32 sweep_count, f32 ridge_regularization) {
    if (total_edge_count == 0) return;

    // Fit to what the matrices currently READ -- reconstruction plus correction. Absorbing
    // the corrections into the basis is the point, so they are the target, not the noise.
    const Vector<Vector<f32>> targets = targets_from_current_values();
    fit_basis_to_targets(targets, sweep_count, ridge_regularization);

    // Whatever the re-optimised basis still misses stays corrected; the rest is now in
    // U/V and needs no entry.
    for (s64 matrix_index = 0; matrix_index < MATRIX_COUNT; matrix_index += 1) {
        store_residual_corrections(matrix_index, targets[(usize)matrix_index]);
    }

    log::logger().debug("refit: {} sweeps, Sk now holds {} weight and {} delay corrections",
                        sweep_count, sparse_delta_entry_count[(usize)DEFAULT_MATRIX_INDEX],
                        sparse_delta_entry_count[(usize)DELAY_MATRIX_INDEX]);
}

f32 WeightMatrix::measure_worst_relative_weight_error(const Vector<f32> &weight) const {
    const s64 run_count = (s64)projection_first_edge_ordinal.size();
    if (run_count == 0 || total_edge_count == 0) return 0.0f;

    const f32 *weight_coefficients = coefficient_row(DEFAULT_MATRIX_INDEX);
    vector<s32> neighbor_buffer((usize)max<s64>(max_neighbor_count, 1));

    f32 worst_relative_error = 0.0f;
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

            const f32 declared = weight[(usize)run_index];
            // Basis PLUS correction: what the engine actually reads. Measuring the basis
            // alone would report an error the corrections have already fixed, and hide the
            // one they have not.
            const f32 reconstructed =
                    reconstruct_entry((s32)source_node, neighbor_buffer[(usize)slot],
                                      weight_coefficients) +
                    sparse_delta_for(DEFAULT_MATRIX_INDEX, (s32)source_node, ordinal);
            // Relative to the declared magnitude, because synaptic weights span many
            // orders of magnitude and an absolute error means nothing across them. A
            // declared zero is compared absolutely, since nothing is relative to zero.
            const f32 scale = fabsf(declared);
            const f32 error = (scale > 0.0f) ? fabsf(reconstructed - declared) / scale
                                             : fabsf(reconstructed - declared);
            worst_relative_error = max(worst_relative_error, error);
        }
    }
    return worst_relative_error;
}

s64 WeightMatrix::count_delay_mismatches(const Vector<s32> &delay_ticks) const {
    const s64 run_count = (s64)projection_first_edge_ordinal.size();
    if (run_count == 0 || total_edge_count == 0) return 0;

    const f32 *delay_coefficients = coefficient_row(DELAY_MATRIX_INDEX);
    vector<s32> neighbor_buffer((usize)max<s64>(max_neighbor_count, 1));

    s64 mismatch_count = 0;
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

            const s32 declared = max<s32>(delay_ticks[(usize)run_index], 1);
            const f32 reconstructed = reconstruct_entry((s32)source_node,
                                                        neighbor_buffer[(usize)slot],
                                                        delay_coefficients);
            if (max<s32>((s32)lroundf(reconstructed), 1) != declared) mismatch_count += 1;
        }
    }
    return mismatch_count;
}

// ── whole-network reads ───────────────────────────────────────────────────────────

void WeightMatrix::neighbor_weights(f32 *output_weights) const {
    neighbor_weights_for_matrix(output_weights, DEFAULT_MATRIX_INDEX);
}

// Writes total_edge_count values indexed by edge ordinal -- no padding, no sentinel rows.
// One GPU thread per real edge; falls back to the host walk when default.metallib is not
// beside the binary, so a build without the AOT library still answers correctly.
void WeightMatrix::neighbor_weights_for_matrix(f32 *output_weights, s64 matrix_index) const {
    validate_matrix_index(matrix_index);
    if (total_edge_count == 0) return;

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
            return;
        }
        log::logger().warn("neighbor_weights: the GPU dispatch failed; answering on the host");
    }

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

// ── constant-value shortcuts ──────────────────────────────────────────────────────

void WeightMatrix::set_constant_weight(f32 value) {
    // Every lane of the group participates in the reconstruction, so the per-lane scale
    // has to be derived from rank_float4_stride * LANE_GROUP -- deriving it from the
    // logical rank overshoots whenever the two differ.
    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    const f32 scale = (value != 0.0f) ? sqrtf(fabsf(value) / (f32)lane_count) : 0.0f;
    const f32 v_fill_value = (value >= 0.0f) ? scale : -scale;

    float4 *u_data = U_matrix.get_contents_as<float4>();
    float4 *v_data = V_matrix.get_contents_as<float4>();
    const s64 total_float4_element_count = node_count * rank_float4_stride;
    for (s64 index = 0; index < total_float4_element_count; index += 1) {
        u_data[index] = {scale, scale, scale, scale};
        v_data[index] = {v_fill_value, v_fill_value, v_fill_value, v_fill_value};
    }

    f32 *weight_coefficients = coefficient_row(DEFAULT_MATRIX_INDEX);
    for (s64 lane_index = 0; lane_index < lane_count; lane_index += 1) {
        weight_coefficients[lane_index] = 1.0f;
    }

    constant_weight = value;
    using_constant_weight = true;
    log::logger().debug("set_constant_weight: value={}", value);
}

void WeightMatrix::set_constant_delay_ticks(s32 ticks) {
    if (ticks < 1) {
        log::throw_invalid_argument(log::logger(),
            "WeightMatrix::set_constant_delay_ticks: ticks must be >= 1 (got " +
            to_string(ticks) + ")");
    }
    constant_delay_ticks = ticks;
    using_constant_delay_ticks = true;
}

// ── plasticity ────────────────────────────────────────────────────────────────────

// One alternating-least-squares nudge of U[source] and V[target] toward a reconstruction
// `delta` higher than the current one. Nothing is stored: the delta is absorbed into the
// basis, which is what keeps per-edge storage at zero.
//
// The cost of that, stated plainly: moving U[source] moves every edge out of `source`, and
// moving V[target] moves every edge into `target`. That is the trade compression makes, and
// `rank` is the knob that decides how much the neighbours move.
void WeightMatrix::update(
    s32 source_node, s32 target_node, f32 delta,
    f32 learning_rate, f32 l2_regularization, s32 iterations
) {
    if (!check_index_inbounds(source_node, target_node)) return;

    if (ensure_function(weight_update_function, "weight_update_kernel")) {
        const s64 rank_float4_stride_argument = rank_float4_stride;
        const s32 source_node_argument = source_node;
        const s32 target_node_argument = target_node;
        const f32 delta_argument = delta;
        const f32 learning_rate_argument = learning_rate;
        const f32 l2_regularization_argument = l2_regularization;
        const s32 iterations_argument = iterations;

        Vector<EnginePointer> parameters = {
            U_matrix,
            V_matrix,
            inline_scalar_argument(rank_float4_stride_argument),
            inline_scalar_argument(source_node_argument),
            inline_scalar_argument(target_node_argument),
            inline_scalar_argument(delta_argument),
            inline_scalar_argument(learning_rate_argument),
            inline_scalar_argument(l2_regularization_argument),
            inline_scalar_argument(iterations_argument),
        };

        // One thread per float4 lane, and a job count equal to the lane count so the whole
        // update lands in a single threadgroup -- the kernel's anchors and its simd_sum
        // reductions are threadgroup-scoped and only mean anything that way.
        if (owning_backend->run_function(weight_update_function, parameters, rank_float4_stride)) {
            using_constant_weight = false;
            return;
        }
        log::logger().warn("update: the GPU dispatch failed; applying on the host");
    }

    const s64 lane_count = rank_float4_stride * LANE_GROUP;
    f32 *u_row = U_matrix.get_contents_as<f32>() + source_node * lane_count;
    f32 *v_row = V_matrix.get_contents_as<f32>() + target_node * lane_count;
    const f32 *weight_coefficients = coefficient_row(DEFAULT_MATRIX_INDEX);

    const f32 target_value = reconstruct_entry(source_node, target_node, weight_coefficients) + delta;

    for (s32 iteration = 0; iteration < iterations; iteration += 1) {
        f32 current = 0.0f;
        f32 u_gradient_norm = 0.0f;
        f32 v_gradient_norm = 0.0f;
        for (s64 lane_index = 0; lane_index < lane_count; lane_index += 1) {
            const f32 scaled_v = weight_coefficients[lane_index] * v_row[lane_index];
            const f32 scaled_u = weight_coefficients[lane_index] * u_row[lane_index];
            current += u_row[lane_index] * scaled_v;
            u_gradient_norm += scaled_v * scaled_v;
            v_gradient_norm += scaled_u * scaled_u;
        }

        const f32 residual = target_value - current;
        for (s64 lane_index = 0; lane_index < lane_count; lane_index += 1) {
            const f32 scaled_v = weight_coefficients[lane_index] * v_row[lane_index];
            const f32 scaled_u = weight_coefficients[lane_index] * u_row[lane_index];
            u_row[lane_index] += learning_rate * residual * scaled_v /
                                 (u_gradient_norm + l2_regularization);
            v_row[lane_index] += learning_rate * residual * scaled_u /
                                 (v_gradient_norm + l2_regularization);
        }
    }

    using_constant_weight = false;
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
    const f32 scale_factor = (target_root_mean_square > 0.0f)
            ? sqrtf(target_root_mean_square / current_root_mean_square)
            : 0.0f;

    // Split evenly between the two factors, so the product scales by scale_factor^2 = the
    // requested ratio and neither factor grows faster than the other.
    const s64 total_float4_element_count = node_count * rank_float4_stride;
    bool scaled_on_device = false;

    if (ensure_function(scale_uv_function, "scale_uv_kernel")) {
        const s64 element_count_argument = total_float4_element_count;
        const f32 scale_factor_argument = scale_factor;
        Vector<EnginePointer> parameters = {
            U_matrix,
            V_matrix,
            inline_scalar_argument(element_count_argument),
            inline_scalar_argument(scale_factor_argument),
        };
        scaled_on_device = owning_backend->run_function(scale_uv_function, parameters,
                                                        total_float4_element_count);
    }

    if (!scaled_on_device) {
        float4 *u_data = U_matrix.get_contents_as<float4>();
        float4 *v_data = V_matrix.get_contents_as<float4>();
        for (s64 index = 0; index < total_float4_element_count; index += 1) {
            u_data[index] = {u_data[index].x * scale_factor, u_data[index].y * scale_factor,
                             u_data[index].z * scale_factor, u_data[index].w * scale_factor};
            v_data[index] = {v_data[index].x * scale_factor, v_data[index].y * scale_factor,
                             v_data[index].z * scale_factor, v_data[index].w * scale_factor};
        }
    }

    constant_weight = 0.0f;
    using_constant_weight = false;

    const WeightStats stats_after = neighbor_weight_stats();
    log::logger().debug("scale_neighbor_weights_to_root_mean_square: target={} factor={} "
                        "rms_before={} rms_after={}",
                        target_root_mean_square, scale_factor,
                        stats_before.root_mean_square, stats_after.root_mean_square);

    return {target_root_mean_square, scale_factor, stats_before, stats_after};
}

// ── serialization ─────────────────────────────────────────────────────────────────

void WeightMatrix::save(const char *filepath) const {
    ofstream file(filepath, ios::binary);
    file.write(reinterpret_cast<const char *>(&WEIGHT_MATRIX_SAVE_MAGIC), sizeof(u32));
    file.write(reinterpret_cast<const char *>(&node_count), sizeof(s64));
    file.write(reinterpret_cast<const char *>(&rank), sizeof(s64));
    file.write(reinterpret_cast<const char *>(&rank_float4_stride), sizeof(s64));

    const s64 basis_bytes = node_count * rank_float4_stride * (s64)sizeof(float4);
    file.write(reinterpret_cast<const char *>(U_matrix.get_contents()), (streamsize)basis_bytes);
    file.write(reinterpret_cast<const char *>(V_matrix.get_contents()), (streamsize)basis_bytes);

    // Both coefficient rows: without them the basis reconstructs nothing meaningful, since
    // Ck is where the declared values actually live.
    const s64 coefficient_bytes = MATRIX_COUNT * rank_float4_stride * LANE_GROUP * (s64)sizeof(f32);
    file.write(reinterpret_cast<const char *>(coefficients.get_contents()), (streamsize)coefficient_bytes);

    // The delay fast path is state, not derived: a matrix whose delays all agree answers
    // from constant_delay_ticks and never reconstructs. Restoring the basis without these
    // gives back a matrix that reports the default one tick for every edge.
    const u8 constant_delay_flag = using_constant_delay_ticks ? 1 : 0;
    file.write(reinterpret_cast<const char *>(&constant_delay_flag), sizeof(u8));
    file.write(reinterpret_cast<const char *>(&constant_delay_ticks), sizeof(s32));

    log::logger().debug("WeightMatrix::save: {} node_count={} rank={}", filepath, node_count, rank);
}

void WeightMatrix::load_from_disk(const char *filepath) {
    ifstream file(filepath, ios::binary);
    u32 magic = 0;
    file.read(reinterpret_cast<char *>(&magic), sizeof(u32));

    s64 saved_node_count = 0;
    s64 saved_rank = 0;
    s64 saved_rank_float4_stride = 0;
    file.read(reinterpret_cast<char *>(&saved_node_count), sizeof(s64));
    file.read(reinterpret_cast<char *>(&saved_rank), sizeof(s64));
    file.read(reinterpret_cast<char *>(&saved_rank_float4_stride), sizeof(s64));

    if (saved_node_count != node_count) {
        log::throw_invalid_argument(log::logger(),
            "WeightMatrix::load_from_disk: the file holds " + to_string(saved_node_count) +
            " nodes but this matrix has " + to_string(node_count) +
            " -- the adjacency has to match, since the basis is indexed by node");
    }

    resize_basis(saved_rank);

    const s64 basis_bytes = node_count * rank_float4_stride * (s64)sizeof(float4);
    file.read(reinterpret_cast<char *>(U_matrix.get_contents()), (streamsize)basis_bytes);
    file.read(reinterpret_cast<char *>(V_matrix.get_contents()), (streamsize)basis_bytes);

    const s64 coefficient_bytes = MATRIX_COUNT * rank_float4_stride * LANE_GROUP * (s64)sizeof(f32);
    file.read(reinterpret_cast<char *>(coefficients.get_contents()), (streamsize)coefficient_bytes);

    u8 constant_delay_flag = 0;
    file.read(reinterpret_cast<char *>(&constant_delay_flag), sizeof(u8));
    file.read(reinterpret_cast<char *>(&constant_delay_ticks), sizeof(s32));
    using_constant_delay_ticks = constant_delay_flag != 0;

    using_constant_weight = false;
    log::logger().debug("WeightMatrix::load_from_disk: {} node_count={} rank={} magic={:#x}",
                        filepath, node_count, rank, magic);
}

} // namespace spikecorec
