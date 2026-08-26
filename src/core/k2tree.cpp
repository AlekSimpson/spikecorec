//
// Created by Alek Simpson on 5/30/26.
//

#include <unordered_map>
#include <vector>
#include <cstring>
#include <fstream>
#include <cstdio>
#include <optional>

#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include "spikecorec/core/k2tree.h"
#include "spikecorec/core/types.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/log.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::log;

static pair<s32, s32> compute_tree_parameters(s32 node_count, s32 branching_factor) {
    if (node_count <= 1) return {0, 1};

    s32 tree_height = 0, padded_size = 1;
    while (padded_size < node_count) {
        padded_size *= branching_factor;
        tree_height++;
    }
    return {tree_height, padded_size};
}

// Packs (level, block_row, block_column) into a single 64-bit key for child_masks.
// 8 bits for level / 28 bits each for block_row and block_column comfortably covers
// node counts into the hundreds of millions, far beyond the stated 1M-neuron scale.
static u64 pack_child_mask_key(s32 level, s32 block_row, s32 block_column) {
    return (u64(u32(level)) << 56) | (u64(u32(block_row)) << 28) | u64(u32(block_column));
}

static u32 get_bit(const u32 *words, s32 bit_index) {
    return (words[bit_index >> 5] >> (bit_index & 31)) & 1u;
}

static s32 rank1_exclusive(const u32 *internal_node_words, const u32 *superblock_table, const u16 *subblock_table,
                           s32 position, s32 superblock_size) {
    s32 word_index = position >> 5;
    s32 bit_offset = position & 31;
    s32 superblock_index = word_index / superblock_size;
    u32 superblock_base = superblock_table[superblock_index];
    u32 subblock_base = subblock_table[word_index];
    u32 partial_word_mask = (bit_offset == 0) ? 0u : ((1u << bit_offset) - 1u);
    u32 partial_word_popcount = (u32) __builtin_popcount(internal_node_words[word_index] & partial_word_mask);
    return (s32) (superblock_base + subblock_base + partial_word_popcount);
}

// Recursive row-walk: collects up to max_neighbor_count neighbor indices of the row
// belonging to `source_node` into output_buffer, descending only into subtrees that
// intersect that row and have at least one bit set. Mirrors K2Tree::adjacent's bit-position
// bookkeeping, but explores every column branch at each level instead of following a fixed
// target column.
static void collect_row_neighbors(
    const u32 *internal_words, const u32 *leaf_words,
    const u32 *superblock_data, const u16 *subblock_data,
    s32 branching_factor, s32 superblock_size_words,
    s32 node_count, s32 tree_height, s32 internal_bit_count,
    s32 level, s32 row_base, s32 column_base, s32 block_size, s32 level_bit_offset,
    s32 source_node, s32 *output_buffer, s64 max_neighbor_count, s64 &neighbors_found
) {
    if (neighbors_found >= max_neighbor_count) return;

    s32 branching_factor_squared = branching_factor * branching_factor;
    s32 child_block_size = block_size / branching_factor;
    s32 row_offset = (source_node - row_base) / child_block_size;

    for (s32 col_offset = 0; col_offset < branching_factor; col_offset++) {
        if (neighbors_found >= max_neighbor_count) return;

        s32 child_flat_index = row_offset * branching_factor + col_offset;
        s32 bit_position = level_bit_offset + child_flat_index;

        if (level == tree_height - 1) {
            if (get_bit(leaf_words, bit_position)) {
                s32 target_node = column_base + col_offset;
                if (target_node < node_count)
                    output_buffer[neighbors_found++] = target_node;
            }
        } else if (get_bit(internal_words, bit_position)) {
            s32 rank_inclusive = rank1_exclusive(internal_words, superblock_data, subblock_data,
                                                  bit_position, superblock_size_words) + 1;
            s32 raw_offset = branching_factor_squared * rank_inclusive;
            s32 child_level_bit_offset = (level + 1 == tree_height - 1)
                ? (raw_offset - internal_bit_count)
                : raw_offset;
            collect_row_neighbors(
                internal_words, leaf_words, superblock_data, subblock_data,
                branching_factor, superblock_size_words, node_count, tree_height, internal_bit_count,
                level + 1,
                row_base + row_offset * child_block_size,
                column_base + col_offset * child_block_size,
                child_block_size, child_level_bit_offset,
                source_node, output_buffer, max_neighbor_count, neighbors_found
            );
        }
    }
}

// Recursive column-walk: the exact mirror of collect_row_neighbors, collecting up to
// max_neighbor_count PREDECESSOR indices of `target_node`'s column into output_buffer (every
// source node with an edge into it). Descends only into subtrees that intersect that column
// and have at least one bit set. Same bit-position bookkeeping as collect_row_neighbors, but
// fixes the column offset (derived from the query node) at each level and explores every ROW
// branch, instead of fixing the row and exploring every column branch.
static void collect_column_predecessors(
    const u32 *internal_words, const u32 *leaf_words,
    const u32 *superblock_data, const u16 *subblock_data,
    s32 branching_factor, s32 superblock_size_words,
    s32 node_count, s32 tree_height, s32 internal_bit_count,
    s32 level, s32 row_base, s32 column_base, s32 block_size, s32 level_bit_offset,
    s32 target_node, s32 *output_buffer, s64 max_neighbor_count, s64 &predecessors_found
) {
    if (predecessors_found >= max_neighbor_count) return;

    s32 branching_factor_squared = branching_factor * branching_factor;
    s32 child_block_size = block_size / branching_factor;
    s32 column_offset = (target_node - column_base) / child_block_size;

    for (s32 row_offset = 0; row_offset < branching_factor; row_offset++) {
        if (predecessors_found >= max_neighbor_count) return;

        s32 child_flat_index = row_offset * branching_factor + column_offset;
        s32 bit_position = level_bit_offset + child_flat_index;

        if (level == tree_height - 1) {
            if (get_bit(leaf_words, bit_position)) {
                s32 source_node = row_base + row_offset;
                if (source_node < node_count)
                    output_buffer[predecessors_found++] = source_node;
            }
        } else if (get_bit(internal_words, bit_position)) {
            s32 rank_inclusive = rank1_exclusive(internal_words, superblock_data, subblock_data,
                                                  bit_position, superblock_size_words) + 1;
            s32 raw_offset = branching_factor_squared * rank_inclusive;
            s32 child_level_bit_offset = (level + 1 == tree_height - 1)
                ? (raw_offset - internal_bit_count)
                : raw_offset;
            collect_column_predecessors(
                internal_words, leaf_words, superblock_data, subblock_data,
                branching_factor, superblock_size_words, node_count, tree_height, internal_bit_count,
                level + 1,
                row_base + row_offset * child_block_size,
                column_base + column_offset * child_block_size,
                child_block_size, child_level_bit_offset,
                target_node, output_buffer, max_neighbor_count, predecessors_found
            );
        }
    }
}

static vector<u32> pack_bits_to_words(const vector<s32> &bits) {
    usize word_count = (bits.size() + 31) / 32;
    vector<u32> words(word_count, 0);
    for (usize index = 0; index < bits.size(); index++) {
        if (bits[index])
            words[index >> 5] |= (1u << (index & 31));
    }
    return words;
}

struct TreeArrays {
    vector<u32> internal_node_words;
    vector<u32> leaf_node_words;
    vector<u32> rank_superblock_table;
    vector<u16> rank_subblock_table;
    s32 tree_height;
    s32 padded_node_count;
    s32 internal_bit_count;
};

static TreeArrays build_tree_arrays(const vector<pair<s32, s32> > &edges, s32 node_count, s32 branching_factor,
                                    s32 superblock_size) {
    // Self-loops (i==j) are not supported and must never be silently accepted or
    // silently dropped -- checked here, unconditionally over every edge, before any
    // other branch (including the tree_height==0 early-return just below, which
    // would otherwise skip this entirely for node_count<=1). This is the single
    // point both K2Tree::from_adjacency_list and K2Tree::from_edges funnel through,
    // so it's the one place that can catch a self-loop from either entry point, and
    // it is what makes WeightMatrix reject one too.
    for (auto [source_node, target_node]: edges) {
        if (source_node == target_node) {
            throw_invalid_argument(logger(),
                fmt::format("K2Tree: self-loop edges are not supported (node {} -> {})",
                            source_node, target_node));
        }
    }

    auto [tree_height, padded_node_count] = compute_tree_parameters(node_count, branching_factor);
    s32 branching_factor_squared = branching_factor * branching_factor;

    if (tree_height == 0)
        return {{}, {}, {u32(0)}, {}, tree_height, padded_node_count, 0};

    // record which children are set at each (level, block_row, block_col)
    unordered_map<u64, s32> child_masks;
    for (auto [source_node, target_node]: edges) {
        s32 current_row = source_node;
        s32 current_column = target_node;
        if (current_row < 0 || current_row >= node_count || 
            current_column < 0 || current_column >= node_count) {
            continue;
        }

        s32 block_row = 0;
        s32 block_column = 0;
        s32 current_size = padded_node_count;
        for (s32 level = 0; level < tree_height; level++) {
            s32 block_size = current_size / branching_factor;
            s32 child_row_index = current_row / block_size;
            s32 child_column_index = current_column / block_size;
            s32 child_flat_index = child_row_index * branching_factor + child_column_index;
            child_masks[pack_child_mask_key(level, block_row, block_column)] |= (1 << child_flat_index);
            current_row = current_row % block_size;
            current_column = current_column % block_size;
            block_row = block_row * branching_factor + child_row_index;
            block_column = block_column * branching_factor + child_column_index;
            current_size = block_size;
        }
    }

    // BFS serialization into T (internal) and L (leaf) bit lists
    vector<s32> internal_bits, leaf_bits;
    vector<pair<s32, s32> > current_level_nodes = {{0, 0}};

    for (s32 level = 0; level < tree_height; level++) {
        vector<pair<s32, s32> > next_level_nodes;
        for (auto [block_row, block_column]: current_level_nodes) {
            auto it = child_masks.find(pack_child_mask_key(level, block_row, block_column));
            s32 child_presence_mask = (it != child_masks.end()) ? it->second : 0;

            auto &target_bit_array = (level < tree_height - 1) ? internal_bits : leaf_bits;
            for (s32 bit_index = 0; bit_index < branching_factor_squared; bit_index++)
                target_bit_array.push_back((child_presence_mask >> bit_index) & 1);

            if (level < tree_height - 1 && child_presence_mask != 0) {
                s32 remaining_children = child_presence_mask;
                while (remaining_children) {
                    s32 least_significant_bit = remaining_children & -remaining_children;
                    s32 child_flat_index = __builtin_ctz((u32) least_significant_bit);
                    s32 child_row_offset = child_flat_index / branching_factor;
                    s32 child_column_offset = child_flat_index - child_row_offset * branching_factor;
                    next_level_nodes.push_back({
                        block_row * branching_factor + child_row_offset,
                        block_column * branching_factor + child_column_offset
                    });
                    remaining_children ^= least_significant_bit;
                }
            }
        }
        current_level_nodes = std::move(next_level_nodes);
    }

    s32 internal_bit_count = (s32) internal_bits.size();
    vector<u32> internal_bit_words = pack_bits_to_words(internal_bits);
    vector<u32> leaf_bit_words = pack_bits_to_words(leaf_bits);

    // build two-level rank tables over T
    usize total_word_count = internal_bit_words.size();
    usize superblock_count = (total_word_count + (usize) superblock_size - 1) / (usize) superblock_size;
    vector<u32> superblock_table(superblock_count + 1, 0);
    vector<u16> subblock_table(total_word_count, 0);

    u32 running_popcount = 0;
    for (usize superblock_index = 0; superblock_index < superblock_count; superblock_index++) {
        superblock_table[superblock_index] = running_popcount;
        usize superblock_word_start = superblock_index * (usize) superblock_size;
        usize superblock_word_end = min(total_word_count, superblock_word_start + (usize) superblock_size);
        u32 within_superblock_popcount = 0;
        for (usize word_index = superblock_word_start; word_index < superblock_word_end; word_index++) {
            subblock_table[word_index] = (u16) within_superblock_popcount;
            within_superblock_popcount += (u32) __builtin_popcount(internal_bit_words[word_index]);
        }
        running_popcount += within_superblock_popcount;
    }
    superblock_table[superblock_count] = running_popcount;

    return {
        std::move(internal_bit_words), std::move(leaf_bit_words),
        std::move(superblock_table), std::move(subblock_table),
        tree_height, padded_node_count, internal_bit_count
    };
}

// Carves one slab holding the four bit arrays plus adjacent_batch's staging, copies the
// host-built arrays into it, and hands the ranges to the constructor. Every k^2-tree entry
// point funnels through here, so this is the single place a tree's storage is decided.
static K2Tree make_k2tree_from_arrays(
    EngineBackend &backend, TreeArrays &arrays,
    s32 node_count, s32 branching_factor, s32 superblock_size
) {
    const usize internal_node_words_length = arrays.internal_node_words.size();
    const usize leaf_node_words_length = arrays.leaf_node_words.size();
    const usize rank_superblock_length = arrays.rank_superblock_table.size();
    const usize rank_subblock_length = arrays.rank_subblock_table.size();

    spikecorec::Vector<EnginePointer> partitions;
    backend
        .partition(internal_node_words_length * sizeof(u32), EngineDatatype::UNSIGNED32, partitions)
        .partition(leaf_node_words_length * sizeof(u32), EngineDatatype::UNSIGNED32, partitions)
        .partition(rank_superblock_length * sizeof(u32), EngineDatatype::UNSIGNED32, partitions)
        .partition(rank_subblock_length * sizeof(u16), EngineDatatype::UNSIGNED16, partitions)
        // adjacent_batch's staging, sized once here rather than per call: the backend
        // hands out one chunk per partition -> allocate round, so there is no transient
        // allocation to reach for. Larger batches are chunked against this cap.
        .partition(K2Tree::ADJACENT_BATCH_QUERY_CAP * sizeof(s32), EngineDatatype::SIGNED32, partitions)
        .partition(K2Tree::ADJACENT_BATCH_QUERY_CAP * sizeof(s32), EngineDatatype::SIGNED32, partitions)
        .partition(K2Tree::ADJACENT_BATCH_QUERY_CAP * sizeof(u8), EngineDatatype::UNSIGNED8, partitions);

    const EnginePointer owning_slab = backend.allocate(partitions);

    // A zero-length array is legal -- the empty adjacency has no bits at all -- and
    // partition() hands back an empty handle for it, so guard each copy on its length
    // rather than assuming get_contents() is non-null.
    if (internal_node_words_length > 0) {
        memcpy(partitions[0].get_contents(), arrays.internal_node_words.data(),
               internal_node_words_length * sizeof(u32));
    }
    if (leaf_node_words_length > 0) {
        memcpy(partitions[1].get_contents(), arrays.leaf_node_words.data(),
               leaf_node_words_length * sizeof(u32));
    }
    if (rank_superblock_length > 0) {
        memcpy(partitions[2].get_contents(), arrays.rank_superblock_table.data(),
               rank_superblock_length * sizeof(u32));
    }
    if (rank_subblock_length > 0) {
        memcpy(partitions[3].get_contents(), arrays.rank_subblock_table.data(),
               rank_subblock_length * sizeof(u16));
    }

    return K2Tree{
        backend,
        owning_slab,
        partitions[0],
        partitions[1],
        partitions[2],
        partitions[3],
        partitions[4],
        partitions[5],
        partitions[6],
        internal_node_words_length,
        leaf_node_words_length,
        rank_superblock_length,
        rank_subblock_length,
        branching_factor,
        superblock_size,
        node_count,
        arrays.padded_node_count,
        arrays.tree_height,
        arrays.internal_bit_count
    };
}

// ── constructor / destructor ──────────────────────────────────────────────────

K2Tree::K2Tree(
    EngineBackend &backend,
    EnginePointer owning_slab,

    EnginePointer internal_node_words,
    EnginePointer leaf_node_words,
    EnginePointer rank_superblock_table,
    EnginePointer rank_subblock_table,

    EnginePointer query_source_staging,
    EnginePointer query_target_staging,
    EnginePointer query_output_staging,

    usize internal_node_words_length,
    usize leaf_node_words_length,
    usize rank_superblock_length,
    usize rank_subblock_length,

    s32 branching_factor,
    s32 superblock_size_words,
    s32 node_count,
    s32 padded_node_count,
    s32 tree_height,
    s32 internal_bit_count
)
    : branching_factor(branching_factor)
      , superblock_size_words(superblock_size_words)
      , node_count(node_count)
      , padded_node_count(padded_node_count)
      , tree_height(tree_height)
      , internal_bit_count(internal_bit_count)
      , internal_node_words(internal_node_words)
      , leaf_node_words(leaf_node_words)
      , rank_superblock_table(rank_superblock_table)
      , rank_subblock_table(rank_subblock_table)
      , query_source_staging(query_source_staging)
      , query_target_staging(query_target_staging)
      , query_output_staging(query_output_staging)
      , owning_backend(&backend)
      , owning_slab(owning_slab)
      , internal_node_words_length(internal_node_words_length)
      , leaf_node_words_length(leaf_node_words_length)
      , rank_superblock_length(rank_superblock_length)
      , rank_subblock_length(rank_subblock_length) {

    // Written once at construction and read on every walk thereafter, which is exactly
    // the case for per-device read-only replicas rather than migration.
    backend.advise_read_mostly(this->internal_node_words, this->internal_node_words_length * sizeof(u32));
    backend.advise_read_mostly(this->leaf_node_words, this->leaf_node_words_length * sizeof(u32));
    backend.advise_read_mostly(this->rank_superblock_table, this->rank_superblock_length * sizeof(u32));
    backend.advise_read_mostly(this->rank_subblock_table, this->rank_subblock_length * sizeof(u16));

    backend.prefetch_to_gpu(this->internal_node_words, this->internal_node_words_length * sizeof(u32));
    backend.prefetch_to_gpu(this->leaf_node_words, this->leaf_node_words_length * sizeof(u32));
    backend.prefetch_to_gpu(this->rank_superblock_table, this->rank_superblock_length * sizeof(u32));
    backend.prefetch_to_gpu(this->rank_subblock_table, this->rank_subblock_length * sizeof(u16));
}

K2Tree::K2Tree(K2Tree &&other) noexcept
    : branching_factor(other.branching_factor)
      , superblock_size_words(other.superblock_size_words)
      , node_count(other.node_count)
      , padded_node_count(other.padded_node_count)
      , tree_height(other.tree_height)
      , internal_bit_count(other.internal_bit_count)
      , internal_node_words(other.internal_node_words)
      , leaf_node_words(other.leaf_node_words)
      , rank_superblock_table(other.rank_superblock_table)
      , rank_subblock_table(other.rank_subblock_table)
      , query_source_staging(other.query_source_staging)
      , query_target_staging(other.query_target_staging)
      , query_output_staging(other.query_output_staging)
      , owning_backend(other.owning_backend)
      , owning_slab(other.owning_slab)
      , internal_node_words_length(other.internal_node_words_length)
      , leaf_node_words_length(other.leaf_node_words_length)
      , rank_superblock_length(other.rank_superblock_length)
      , rank_subblock_length(other.rank_subblock_length) {

    // The whole reason this is not defaulted: the source must forget the slab, or both
    // objects release it. Everything else above is a plain value copy.
    other.owning_backend = nullptr;
    other.owning_slab = EnginePointer{};
    other.tree_height = 0;
    other.node_count = 0;
}

K2Tree &K2Tree::operator=(K2Tree &&other) noexcept {
    if (this == &other) return *this;

    // Release what this object already holds before taking over the incoming slab.
    if (owning_backend != nullptr) owning_backend->deallocate_slab(owning_slab);

    branching_factor = other.branching_factor;
    superblock_size_words = other.superblock_size_words;
    node_count = other.node_count;
    padded_node_count = other.padded_node_count;
    tree_height = other.tree_height;
    internal_bit_count = other.internal_bit_count;
    internal_node_words = other.internal_node_words;
    leaf_node_words = other.leaf_node_words;
    rank_superblock_table = other.rank_superblock_table;
    rank_subblock_table = other.rank_subblock_table;
    query_source_staging = other.query_source_staging;
    query_target_staging = other.query_target_staging;
    query_output_staging = other.query_output_staging;
    owning_backend = other.owning_backend;
    owning_slab = other.owning_slab;
    internal_node_words_length = other.internal_node_words_length;
    leaf_node_words_length = other.leaf_node_words_length;
    rank_superblock_length = other.rank_superblock_length;
    rank_subblock_length = other.rank_subblock_length;

    other.owning_backend = nullptr;
    other.owning_slab = EnginePointer{};
    other.tree_height = 0;
    other.node_count = 0;
    return *this;
}

K2Tree::~K2Tree() {
    // The four bit arrays and the query staging are all sub-ranges of one slab, and no
    // sub-range owns anything -- releasing the slab is what frees them.
    if (owning_backend != nullptr) owning_backend->deallocate_slab(owning_slab);
}

// ── factory methods ───────────────────────────────────────────────────────────

optional<K2Tree> K2Tree::from_adjacency_list(
    EngineBackend &backend,
    const vector<vector<s32> > &adjacency_list,
    s32 node_count,
    s32 branching_factor,
    s32 superblock_size
) {
    logger().debug("K2Tree::from_adjacency_list: adjacency_list.size={} node_count={} branching_factor={} "
                   "superblock_size={}", adjacency_list.size(), node_count, branching_factor, superblock_size);
    if (branching_factor > 5 || branching_factor < 0) {
        logger().warn("K2Tree::from_adjacency_list: rejecting out-of-range branching_factor={} (must be 0-5)",
                      branching_factor);
        return nullopt;
    }

    s32 effective_node_count = (node_count >= 0) ? node_count : (s32) adjacency_list.size();

    vector<pair<s32, s32> > edges;
    for (s32 source_node = 0; source_node < (s32) adjacency_list.size(); source_node++)
        for (s32 target_node: adjacency_list[source_node])
            edges.emplace_back(source_node, target_node);

    auto arrays = build_tree_arrays(edges, effective_node_count, branching_factor, superblock_size);
    logger().debug("K2Tree::from_adjacency_list: edge_count={} tree_height={} padded_node_count={} "
                   "internal_bit_count={}", edges.size(), arrays.tree_height, arrays.padded_node_count,
                   arrays.internal_bit_count);
    return make_k2tree_from_arrays(backend, arrays, effective_node_count, branching_factor, superblock_size);
}

optional<K2Tree> K2Tree::from_edges(
    EngineBackend &backend,
    const s32 *source_indices,
    const s32 *target_indices,
    s32 edge_count,
    s32 node_count,
    s32 branching_factor,
    s32 superblock_size
) {
    logger().debug("K2Tree::from_edges: edge_count={} node_count={} branching_factor={} superblock_size={}",
                   edge_count, node_count, branching_factor, superblock_size);
    if (branching_factor > 5 || branching_factor < 0) {
        logger().warn("K2Tree::from_edges: rejecting out-of-range branching_factor={} (must be 0-5)",
                      branching_factor);
        return nullopt;
    }

    vector<pair<s32, s32> > edges;
    edges.reserve((usize) edge_count);
    for (s32 edge_index = 0; edge_index < edge_count; edge_index++)
        edges.emplace_back(source_indices[edge_index], target_indices[edge_index]);

    auto arrays = build_tree_arrays(edges, node_count, branching_factor, superblock_size);
    logger().debug("K2Tree::from_edges: tree_height={} padded_node_count={} internal_bit_count={}",
                   arrays.tree_height, arrays.padded_node_count, arrays.internal_bit_count);
    return make_k2tree_from_arrays(backend, arrays, node_count, branching_factor, superblock_size);
}

// ── serialization ─────────────────────────────────────────────────────────────

static constexpr u32 SAVE_MAGIC = 0x4B325452; // "K2TR"

K2Tree K2Tree::load(EngineBackend &backend, const char *path) {
    logger().debug("K2Tree::load: path={}", path);
    ifstream file(path, ios::binary);

    u32 magic;
    file.read(reinterpret_cast<char *>(&magic), sizeof(magic));

    s32 branching_factor, superblock_size_words, node_count, padded_node_count, tree_height, internal_bit_count;
    file.read(reinterpret_cast<char *>(&branching_factor), sizeof(s32));
    file.read(reinterpret_cast<char *>(&superblock_size_words), sizeof(s32));
    file.read(reinterpret_cast<char *>(&node_count), sizeof(s32));
    file.read(reinterpret_cast<char *>(&padded_node_count), sizeof(s32));
    file.read(reinterpret_cast<char *>(&tree_height), sizeof(s32));
    file.read(reinterpret_cast<char *>(&internal_bit_count), sizeof(s32));

    usize internal_node_words_length, leaf_node_words_length, rank_superblock_length, rank_subblock_length;
    file.read(reinterpret_cast<char *>(&internal_node_words_length), sizeof(usize));
    file.read(reinterpret_cast<char *>(&leaf_node_words_length), sizeof(usize));
    file.read(reinterpret_cast<char *>(&rank_superblock_length), sizeof(usize));
    file.read(reinterpret_cast<char *>(&rank_subblock_length), sizeof(usize));

    // Same chunk layout as make_k2tree_from_arrays: the four arrays plus adjacent_batch's
    // staging, so a loaded tree is indistinguishable from a built one.
    spikecorec::Vector<EnginePointer> partitions;
    backend
        .partition(internal_node_words_length * sizeof(u32), EngineDatatype::UNSIGNED32, partitions)
        .partition(leaf_node_words_length * sizeof(u32), EngineDatatype::UNSIGNED32, partitions)
        .partition(rank_superblock_length * sizeof(u32), EngineDatatype::UNSIGNED32, partitions)
        .partition(rank_subblock_length * sizeof(u16), EngineDatatype::UNSIGNED16, partitions)
        .partition(K2Tree::ADJACENT_BATCH_QUERY_CAP * sizeof(s32), EngineDatatype::SIGNED32, partitions)
        .partition(K2Tree::ADJACENT_BATCH_QUERY_CAP * sizeof(s32), EngineDatatype::SIGNED32, partitions)
        .partition(K2Tree::ADJACENT_BATCH_QUERY_CAP * sizeof(u8), EngineDatatype::UNSIGNED8, partitions);

    const EnginePointer owning_slab = backend.allocate(partitions);

    if (internal_node_words_length > 0) {
        file.read(reinterpret_cast<char *>(partitions[0].get_contents()),
                  (streamsize) (internal_node_words_length * sizeof(u32)));
    }
    if (leaf_node_words_length > 0) {
        file.read(reinterpret_cast<char *>(partitions[1].get_contents()),
                  (streamsize) (leaf_node_words_length * sizeof(u32)));
    }
    if (rank_superblock_length > 0) {
        file.read(reinterpret_cast<char *>(partitions[2].get_contents()),
                  (streamsize) (rank_superblock_length * sizeof(u32)));
    }
    if (rank_subblock_length > 0) {
        file.read(reinterpret_cast<char *>(partitions[3].get_contents()),
                  (streamsize) (rank_subblock_length * sizeof(u16)));
    }

    logger().debug("K2Tree::load: path={} magic={:#x} node_count={} branching_factor={} tree_height={} "
                   "padded_node_count={} internal_bit_count={}",
                   path, magic, node_count, branching_factor, tree_height, padded_node_count, internal_bit_count);

    return K2Tree{
        backend,
        owning_slab,
        partitions[0],
        partitions[1],
        partitions[2],
        partitions[3],
        partitions[4],
        partitions[5],
        partitions[6],
        internal_node_words_length,
        leaf_node_words_length,
        rank_superblock_length,
        rank_subblock_length,
        branching_factor,
        superblock_size_words,
        node_count,
        padded_node_count,
        tree_height,
        internal_bit_count
    };
}

void K2Tree::save(const char *path) const {
    logger().debug("K2Tree::save: path={} node_count={} branching_factor={} tree_height={} "
                   "padded_node_count={} internal_bit_count={}",
                   path, node_count, branching_factor, tree_height, padded_node_count, internal_bit_count);
    ofstream file(path, ios::binary);

    file.write(reinterpret_cast<const char *>(&SAVE_MAGIC), sizeof(u32));
    file.write(reinterpret_cast<const char *>(&branching_factor), sizeof(s32));
    file.write(reinterpret_cast<const char *>(&superblock_size_words), sizeof(s32));
    file.write(reinterpret_cast<const char *>(&node_count), sizeof(s32));
    file.write(reinterpret_cast<const char *>(&padded_node_count), sizeof(s32));
    file.write(reinterpret_cast<const char *>(&tree_height), sizeof(s32));
    file.write(reinterpret_cast<const char *>(&internal_bit_count), sizeof(s32));
    file.write(reinterpret_cast<const char *>(&internal_node_words_length), sizeof(usize));
    file.write(reinterpret_cast<const char *>(&leaf_node_words_length), sizeof(usize));
    file.write(reinterpret_cast<const char *>(&rank_superblock_length), sizeof(usize));
    file.write(reinterpret_cast<const char *>(&rank_subblock_length), sizeof(usize));

    const u32 *internal_node_data = internal_node_words.get_contents_as<u32>();
    const u32 *leaf_node_data = leaf_node_words.get_contents_as<u32>();
    const u32 *rank_superblock_data = rank_superblock_table.get_contents_as<u32>();
    const u16 *rank_subblock_data = rank_subblock_table.get_contents_as<u16>();

    file.write(reinterpret_cast<const char *>(internal_node_data),
               (streamsize) (internal_node_words_length * sizeof(u32)));
    file.write(reinterpret_cast<const char *>(leaf_node_data), (streamsize) (leaf_node_words_length * sizeof(u32)));
    file.write(reinterpret_cast<const char *>(rank_superblock_data),
               (streamsize) (rank_superblock_length * sizeof(u32)));
    file.write(reinterpret_cast<const char *>(rank_subblock_data), (streamsize) (rank_subblock_length * sizeof(u16)));
}

// ── queries ───────────────────────────────────────────────────────────────────

s32 K2Tree::adjacent(s32 source_node, s32 target_node) const {
    if (source_node < 0 || source_node >= node_count ||
        target_node < 0 || target_node >= node_count) return 0;
    if (tree_height == 0) return 0;

    const u32 *internal_words = internal_node_words.get_contents_as<u32>();
    const u32 *leaf_words = leaf_node_words.get_contents_as<u32>();
    const u32 *superblock_data = rank_superblock_table.get_contents_as<u32>();
    const u16 *subblock_data = rank_subblock_table.get_contents_as<u16>();

    s32 branching_factor_squared = branching_factor * branching_factor;

    if (tree_height == 1) {
        s32 block_size = padded_node_count / branching_factor;
        s32 child_flat_index =
            (source_node / block_size) * branching_factor + (target_node / block_size);
        return (s32) get_bit(leaf_words, child_flat_index);
    }

    s32 level_bit_offset = 0, current_block_size = padded_node_count, rank_inclusive = 0;

    for (s32 level = 0; level < tree_height - 1; level++) {
        s32 block_size = current_block_size / branching_factor;
        s32 row_offset = source_node / block_size;
        s32 column_offset = target_node / block_size;
        s32 child_bit_position = level_bit_offset + row_offset * branching_factor + column_offset;

        if (!get_bit(internal_words, child_bit_position)) return 0;

        rank_inclusive = rank1_exclusive(internal_words, superblock_data, subblock_data, child_bit_position,
                                         superblock_size_words) + 1;

        if (level == tree_height - 2) {
            source_node = source_node % block_size;
            target_node = target_node % block_size;
            break;
        }

        level_bit_offset = branching_factor_squared * rank_inclusive;
        source_node = source_node % block_size;
        target_node = target_node % block_size;
        current_block_size = block_size;
    }

    s32 leaf_bit_offset = branching_factor_squared * rank_inclusive - internal_bit_count;
    return (s32) get_bit(leaf_words,
                         leaf_bit_offset + source_node * branching_factor + target_node);
}

s64 K2Tree::get_neighbors(s32 node_index, s32 *output_buffer, s64 max_neighbor_count) const {
    if (max_neighbor_count <= 0) return 0;
    if (node_index < 0 || node_index >= node_count || tree_height == 0) return 0;

    const u32 *internal_words = internal_node_words.get_contents_as<u32>();
    const u32 *leaf_words = leaf_node_words.get_contents_as<u32>();
    const u32 *superblock_data = rank_superblock_table.get_contents_as<u32>();
    const u16 *subblock_data = rank_subblock_table.get_contents_as<u16>();

    s64 neighbors_found = 0;
    collect_row_neighbors(
        internal_words, leaf_words, superblock_data, subblock_data,
        branching_factor, superblock_size_words, node_count, tree_height, internal_bit_count,
        0, 0, 0, padded_node_count, 0,
        node_index, output_buffer, max_neighbor_count, neighbors_found
    );
    return neighbors_found;
}

s64 K2Tree::get_predecessors(s32 node_index, s32 *output_buffer, s64 max_neighbor_count) const {
    if (max_neighbor_count <= 0) return 0;
    if (node_index < 0 || node_index >= node_count || tree_height == 0) return 0;

    const u32 *internal_words = internal_node_words.get_contents_as<u32>();
    const u32 *leaf_words = leaf_node_words.get_contents_as<u32>();
    const u32 *superblock_data = rank_superblock_table.get_contents_as<u32>();
    const u16 *subblock_data = rank_subblock_table.get_contents_as<u16>();

    s64 predecessors_found = 0;
    collect_column_predecessors(
        internal_words, leaf_words, superblock_data, subblock_data,
        branching_factor, superblock_size_words, node_count, tree_height, internal_bit_count,
        0, 0, 0, padded_node_count, 0,
        node_index, output_buffer, max_neighbor_count, predecessors_found
    );
    return predecessors_found;
}

void K2Tree::adjacent_batch(
    const s32 *source_indices,
    const s32 *target_indices,
    uint8_t *output_buffer,
    s32 query_count
) const {
    if (query_count <= 0) return;
    if (owning_backend == nullptr) {
        // The empty adjacency has no slab and therefore no staging. It also has no edges,
        // so every query answers 0.
        memset(output_buffer, 0, (usize)query_count * sizeof(uint8_t));
        return;
    }

    // source_indices/target_indices/output_buffer are caller-owned host memory (mirroring
    // the plain s32 node indices of the single-query `adjacent`), not GPU-visible, so
    // queries are staged into this tree's own staging ranges and the results copied back.
    //
    // The staging is sized once at construction (ADJACENT_BATCH_QUERY_CAP), because the
    // backend hands out one chunk per partition -> allocate round and there is no
    // transient allocation available. A larger batch is answered in chunks rather than
    // refused -- the cap bounds memory, not the query.
    s32 *staged_sources = query_source_staging.get_contents_as<s32>();
    s32 *staged_targets = query_target_staging.get_contents_as<s32>();
    uint8_t *staged_output = query_output_staging.get_contents_as<uint8_t>();

    // Built once, on the first batch. A build without default.metallib beside the binary
    // gets nullopt here and answers on the host instead of failing.
    bool dispatch_is_available = false;
#ifdef SPIKECOREC_METAL
    dispatch_is_available = adjacent_batch_function.pipeline_state != nullptr;
#endif
    if (!dispatch_is_available) {
        Optional<EngineFunction> loaded =
                owning_backend->load_precompiled_function("k2tree_adjacent_batch_kernel");
        if (loaded.has_value()) {
            adjacent_batch_function = *loaded;
            dispatch_is_available = true;
        }
    }

    for (s32 first_query = 0; first_query < query_count; first_query += (s32)ADJACENT_BATCH_QUERY_CAP) {
        const s32 chunk_size =
                std::min<s32>((s32)ADJACENT_BATCH_QUERY_CAP, query_count - first_query);

        memcpy(staged_sources, source_indices + first_query, (usize)chunk_size * sizeof(s32));
        memcpy(staged_targets, target_indices + first_query, (usize)chunk_size * sizeof(s32));

        bool answered_on_device = false;
        if (dispatch_is_available) {
            const s32 chunk_size_argument = chunk_size;
            spikecorec::Vector<EnginePointer> parameters = {
                internal_node_words,                            // 0
                leaf_node_words,                                // 1
                rank_superblock_table,                          // 2
                rank_subblock_table,                            // 3
                inline_scalar_argument(branching_factor),       // 4
                inline_scalar_argument(superblock_size_words),  // 5
                inline_scalar_argument(node_count),             // 6
                inline_scalar_argument(padded_node_count),      // 7
                inline_scalar_argument(tree_height),            // 8
                inline_scalar_argument(internal_bit_count),     // 9
                query_source_staging,                           // 10
                query_target_staging,                           // 11
                query_output_staging,                           // 12
                inline_scalar_argument(chunk_size_argument),    // 13
            };
            answered_on_device = owning_backend->run_function(adjacent_batch_function,
                                                              parameters, chunk_size);
        }

        if (!answered_on_device) {
            for (s32 query_index = 0; query_index < chunk_size; query_index += 1) {
                staged_output[query_index] = (uint8_t)adjacent(staged_sources[query_index],
                                                               staged_targets[query_index]);
            }
        }

        memcpy(output_buffer + first_query, staged_output, (usize)chunk_size * sizeof(uint8_t));
    }
}

s32 K2Tree::trace(s32 source_node, s32 target_node) const {
    logger().trace("source_node={} target_node={}  N={} H={} Npad={} k={}",
                    source_node, target_node, node_count, tree_height, padded_node_count,
                    branching_factor);

    if (source_node < 0 || source_node >= node_count ||
        target_node < 0 || target_node >= node_count) {
        logger().trace("  out of bounds => 0");
        return 0;
    }
    if (tree_height == 0) {
        logger().trace("  tree_height==0 => 0");
        return 0;
    }

    const u32 *internal_words = internal_node_words.get_contents_as<u32>();
    const u32 *leaf_words = leaf_node_words.get_contents_as<u32>();
    const u32 *superblock_data = rank_superblock_table.get_contents_as<u32>();
    const u16 *subblock_data = rank_subblock_table.get_contents_as<u16>();

    s32 branching_factor_squared = branching_factor * branching_factor;

    if (tree_height == 1) {
        s32 block_size = padded_node_count / branching_factor;
        s32 child_flat_index =
            (source_node / block_size) * branching_factor + (target_node / block_size);
        s32 leaf_bit = (s32) get_bit(leaf_words, child_flat_index);
        logger().trace("  tree_height==1: block_size={} child_flat_index={}  L[{}]={}",
                       block_size, child_flat_index, child_flat_index, leaf_bit);
        return leaf_bit;
    }

    s32 level_bit_offset, rank_inclusive = 0; 
    s32 current_block_size = padded_node_count;
    s32 row_remainder = 0, column_remainder = 0;

    for (s32 level = 0; level < tree_height - 1; level++) {
        s32 block_size = current_block_size / branching_factor;
        s32 row_offset = source_node / block_size;
        s32 column_offset = target_node / block_size;
        s32 child_bit_position = level_bit_offset + row_offset * branching_factor + column_offset;
        s32 bit_value = (s32) get_bit(internal_words, child_bit_position);
        s32 rank_exclusive = rank1_exclusive(internal_words, superblock_data, subblock_data, child_bit_position,
                                             superblock_size_words);
        rank_inclusive = rank_exclusive + 1;

        logger().trace(
            "  lvl={:2d} block_size={:6d} source_node={:6d} target_node={:6d} row_offset={} column_offset={} child_bit_pos={:8d} T[pos]={} rank_ex={} rank_incl={}",
            level, block_size, source_node, target_node, row_offset, column_offset,
            child_bit_position, bit_value, rank_exclusive, rank_inclusive);

        if (!bit_value) {
            logger().trace("  stop: T[pos]==0 => 0");
            return 0;
        }

        if (level == tree_height - 2) {
            row_remainder = source_node % block_size;
            column_remainder = target_node % block_size;
            break;
        }

        level_bit_offset = branching_factor_squared * rank_inclusive;
        source_node = source_node % block_size;
        target_node = target_node % block_size;
        current_block_size = block_size;
    }

    s32 leaf_bit_offset = branching_factor_squared * rank_inclusive - internal_bit_count;
    s32 leaf_child_index = row_remainder * branching_factor + column_remainder;
    s32 leaf_bit_index = leaf_bit_offset + leaf_child_index;
    s32 leaf_bit = (s32) get_bit(leaf_words, leaf_bit_index);

    logger().trace(
        "  leaf: rank_incl={} leaf_bit_offset={} row_remainder={} column_remainder={} leaf_child_index={} -> L[{}]={}",
        rank_inclusive, leaf_bit_offset, row_remainder, column_remainder, leaf_child_index, leaf_bit_index, leaf_bit);

    return leaf_bit;
}
