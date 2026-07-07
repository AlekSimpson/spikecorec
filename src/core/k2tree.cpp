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

// Recursive row-walk: collects up to max_neighbor_count neighbor indices of row `u`
// into output_buffer, descending only into subtrees that intersect u's row and have
// at least one bit set. Mirrors K2Tree::adjacent's bit-position bookkeeping, but
// explores every column branch at each level instead of following a fixed `v`.
// TODO: leave documentation about what 'u' is 
static void collect_row_neighbors(
    const u32 *internal_words, const u32 *leaf_words,
    const u32 *superblock_data, const u16 *subblock_data,
    s32 branching_factor, s32 superblock_size_words,
    s32 node_count, s32 tree_height, s32 internal_bit_count,
    s32 level, s32 row_base, s32 column_base, s32 block_size, s32 level_bit_offset,
    s32 u, s32 *output_buffer, s64 max_neighbor_count, s64 &neighbors_found
) {
    if (neighbors_found >= max_neighbor_count) return;

    s32 branching_factor_squared = branching_factor * branching_factor;
    s32 child_block_size = block_size / branching_factor;
    s32 row_offset = (u - row_base) / child_block_size;

    for (s32 col_offset = 0; col_offset < branching_factor; col_offset++) {
        if (neighbors_found >= max_neighbor_count) return;

        s32 child_flat_index = row_offset * branching_factor + col_offset;
        s32 bit_position = level_bit_offset + child_flat_index;

        if (level == tree_height - 1) {
            if (get_bit(leaf_words, bit_position)) {
                s32 v = column_base + col_offset;
                if (v < node_count)
                    output_buffer[neighbors_found++] = v;
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
                u, output_buffer, max_neighbor_count, neighbors_found
            );
        }
    }
}

static vector<u32> pack_bits_to_words(const vector<s32> &bits) {
    usize word_count = (bits.size() + 31) / 32;
    vector<u32> words(word_count, 0);
    for (usize i = 0; i < bits.size(); i++) {
        if (bits[i])
            words[i >> 5] |= (1u << (i & 31));
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

static K2Tree make_k2tree_from_arrays(TreeArrays &arrays, s32 node_count, s32 branching_factor, s32 superblock_size) {
    usize internal_node_words_length = arrays.internal_node_words.size();
    usize leaf_node_words_length = arrays.leaf_node_words.size();
    usize rank_superblock_length = arrays.rank_superblock_table.size();
    usize rank_subblock_length = arrays.rank_subblock_table.size();

    GpuPointer<u32> internal_node_gpu_pointer = allocate<u32>(internal_node_words_length * sizeof(u32));
    GpuPointer<u32> leaf_node_gpu_pointer = allocate<u32>(leaf_node_words_length * sizeof(u32));
    GpuPointer<u32> rank_superblock_gpu_pointer = allocate<u32>(rank_superblock_length * sizeof(u32));
    GpuPointer<u16> rank_subblock_gpu_pointer = allocate<u16>(rank_subblock_length * sizeof(u16));

    memcpy(internal_node_gpu_pointer.get_contents(), arrays.internal_node_words.data(),
           internal_node_words_length * sizeof(u32));
    memcpy(leaf_node_gpu_pointer.get_contents(), arrays.leaf_node_words.data(), leaf_node_words_length * sizeof(u32));
    memcpy(rank_superblock_gpu_pointer.get_contents(), arrays.rank_superblock_table.data(),
           rank_superblock_length * sizeof(u32));
    memcpy(rank_subblock_gpu_pointer.get_contents(), arrays.rank_subblock_table.data(), rank_subblock_length * sizeof(u16));

    return {
        std::move(internal_node_gpu_pointer),
        std::move(leaf_node_gpu_pointer),
        std::move(rank_superblock_gpu_pointer),
        std::move(rank_subblock_gpu_pointer),
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
    GpuPointer<u32> internal_node_words,
    GpuPointer<u32> leaf_node_words,
    GpuPointer<u32> rank_superblock_table,
    GpuPointer<u16> rank_subblock_table,

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
      , internal_node_words(std::move(internal_node_words))
      , leaf_node_words(std::move(leaf_node_words))
      , rank_superblock_table(std::move(rank_superblock_table))
      , rank_subblock_table(std::move(rank_subblock_table))
      , internal_node_words_length(internal_node_words_length)
      , leaf_node_words_length(leaf_node_words_length)
      , rank_superblock_length(rank_superblock_length)
      , rank_subblock_length(rank_subblock_length) {

    advise_read_mostly(this->internal_node_words, this->internal_node_words_length * sizeof(u32));
    advise_read_mostly(this->leaf_node_words, this->leaf_node_words_length * sizeof(u32));
    advise_read_mostly(this->rank_superblock_table, this->rank_superblock_length * sizeof(u32));
    advise_read_mostly(this->rank_subblock_table, this->rank_subblock_length * sizeof(u16));

    prefetch_to_gpu(this->internal_node_words, this->internal_node_words_length * sizeof(u32));
    prefetch_to_gpu(this->leaf_node_words, this->leaf_node_words_length * sizeof(u32));
    prefetch_to_gpu(this->rank_superblock_table, this->rank_superblock_length * sizeof(u32));
    prefetch_to_gpu(this->rank_subblock_table, this->rank_subblock_length * sizeof(u16));
}

K2Tree::~K2Tree() {
    deallocate(std::move(internal_node_words));
    deallocate(std::move(leaf_node_words));
    deallocate(std::move(rank_superblock_table));
    deallocate(std::move(rank_subblock_table));
}

// ── factory methods ───────────────────────────────────────────────────────────

optional<K2Tree> K2Tree::from_adjacency_list(
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
    for (s32 u = 0; u < (s32) adjacency_list.size(); u++)
        for (s32 v: adjacency_list[u])
            edges.emplace_back(u, v);

    auto arrays = build_tree_arrays(edges, effective_node_count, branching_factor, superblock_size);
    logger().debug("K2Tree::from_adjacency_list: edge_count={} tree_height={} padded_node_count={} "
                   "internal_bit_count={}", edges.size(), arrays.tree_height, arrays.padded_node_count,
                   arrays.internal_bit_count);
    return make_k2tree_from_arrays(arrays, effective_node_count, branching_factor, superblock_size);
}

optional<K2Tree> K2Tree::from_edges(
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
    for (s32 i = 0; i < edge_count; i++)
        edges.emplace_back(source_indices[i], target_indices[i]);

    auto arrays = build_tree_arrays(edges, node_count, branching_factor, superblock_size);
    logger().debug("K2Tree::from_edges: tree_height={} padded_node_count={} internal_bit_count={}",
                   arrays.tree_height, arrays.padded_node_count, arrays.internal_bit_count);
    return make_k2tree_from_arrays(arrays, node_count, branching_factor, superblock_size);
}

// ── serialization ─────────────────────────────────────────────────────────────

static constexpr u32 SAVE_MAGIC = 0x4B325452; // "K2TR"

K2Tree K2Tree::load(const char *path) {
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

    GpuPointer<u32> internal_node_gpu_pointer = allocate<u32>(internal_node_words_length * sizeof(u32));
    GpuPointer<u32> leaf_node_gpu_pointer = allocate<u32>(leaf_node_words_length * sizeof(u32));
    GpuPointer<u32> rank_superblock_gpu_pointer = allocate<u32>(rank_superblock_length * sizeof(u32));
    GpuPointer<u16> rank_subblock_gpu_pointer = allocate<u16>(rank_subblock_length * sizeof(u16));

    file.read(reinterpret_cast<char *>(internal_node_gpu_pointer.get_contents()),
              (streamsize) (internal_node_words_length * sizeof(u32)));
    file.read(reinterpret_cast<char *>(leaf_node_gpu_pointer.get_contents()),
              (streamsize) (leaf_node_words_length * sizeof(u32)));
    file.read(reinterpret_cast<char *>(rank_superblock_gpu_pointer.get_contents()),
              (streamsize) (rank_superblock_length * sizeof(u32)));
    file.read(reinterpret_cast<char *>(rank_subblock_gpu_pointer.get_contents()),
              (streamsize) (rank_subblock_length * sizeof(u16)));

    logger().debug("K2Tree::load: path={} magic={:#x} node_count={} branching_factor={} tree_height={} "
                   "padded_node_count={} internal_bit_count={}",
                   path, magic, node_count, branching_factor, tree_height, padded_node_count, internal_bit_count);

    return {
        std::move(internal_node_gpu_pointer),
        std::move(leaf_node_gpu_pointer),
        std::move(rank_superblock_gpu_pointer),
        std::move(rank_subblock_gpu_pointer),
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

    const u32 *internal_node_data = internal_node_words.get_contents();
    const u32 *leaf_node_data = leaf_node_words.get_contents();
    const u32 *rank_superblock_data = rank_superblock_table.get_contents();
    const u16 *rank_subblock_data = rank_subblock_table.get_contents();

    file.write(reinterpret_cast<const char *>(internal_node_data),
               (streamsize) (internal_node_words_length * sizeof(u32)));
    file.write(reinterpret_cast<const char *>(leaf_node_data), (streamsize) (leaf_node_words_length * sizeof(u32)));
    file.write(reinterpret_cast<const char *>(rank_superblock_data),
               (streamsize) (rank_superblock_length * sizeof(u32)));
    file.write(reinterpret_cast<const char *>(rank_subblock_data), (streamsize) (rank_subblock_length * sizeof(u16)));
}

// ── queries ───────────────────────────────────────────────────────────────────

s32 K2Tree::adjacent(s32 u, s32 v) const {
    // u and v are some nodes in the graph

    if (u < 0 || u >= node_count || v < 0 || v >= node_count) return 0;
    if (tree_height == 0) return 0;

    const u32 *internal_words = internal_node_words.get_contents();
    const u32 *leaf_words = leaf_node_words.get_contents();
    const u32 *superblock_data = rank_superblock_table.get_contents();
    const u16 *subblock_data = rank_subblock_table.get_contents();

    s32 branching_factor_squared = branching_factor * branching_factor;

    if (tree_height == 1) {
        s32 block_size = padded_node_count / branching_factor;
        s32 child_flat_index = (u / block_size) * branching_factor + (v / block_size);
        return (s32) get_bit(leaf_words, child_flat_index);
    }

    s32 level_bit_offset = 0, current_block_size = padded_node_count, rank_inclusive = 0;

    for (s32 level = 0; level < tree_height - 1; level++) {
        s32 block_size = current_block_size / branching_factor;
        s32 row_offset = u / block_size;
        s32 column_offset = v / block_size;
        s32 child_bit_position = level_bit_offset + row_offset * branching_factor + column_offset;

        if (!get_bit(internal_words, child_bit_position)) return 0;

        rank_inclusive = rank1_exclusive(internal_words, superblock_data, subblock_data, child_bit_position,
                                         superblock_size_words) + 1;

        if (level == tree_height - 2) {
            u = u % block_size;
            v = v % block_size;
            break;
        }

        level_bit_offset = branching_factor_squared * rank_inclusive;
        u = u % block_size;
        v = v % block_size;
        current_block_size = block_size;
    }

    s32 leaf_bit_offset = branching_factor_squared * rank_inclusive - internal_bit_count;
    return (s32) get_bit(leaf_words, leaf_bit_offset + u * branching_factor + v);
}

s64 K2Tree::get_neighbors(s32 node_index, s32 *output_buffer, s64 max_neighbor_count) const {
    if (max_neighbor_count <= 0) return 0;
    if (node_index < 0 || node_index >= node_count || tree_height == 0) return 0;

    const u32 *internal_words = internal_node_words.get_contents();
    const u32 *leaf_words = leaf_node_words.get_contents();
    const u32 *superblock_data = rank_superblock_table.get_contents();
    const u16 *subblock_data = rank_subblock_table.get_contents();

    s64 neighbors_found = 0;
    collect_row_neighbors(
        internal_words, leaf_words, superblock_data, subblock_data,
        branching_factor, superblock_size_words, node_count, tree_height, internal_bit_count,
        0, 0, 0, padded_node_count, 0,
        node_index, output_buffer, max_neighbor_count, neighbors_found
    );
    return neighbors_found;
}

void K2Tree::adjacent_batch(
    const s32 *source_indices,
    const s32 *target_indices,
    uint8_t *output_buffer,
    s32 query_count
) const {
    if (query_count <= 0) return;

    // note: source_indices/target_indices/output_buffer are caller-owned host memory
    // (mirrors the plain s32 u/v of the single-query `adjacent`) — not GPU-visible —
    // so queries are staged into unified-memory scratch buffers, the kernel writes
    // results into a scratch output buffer, and we copy the results back once done.
    GpuPointer<s32> device_source = allocate<s32>((usize)query_count * sizeof(s32));
    GpuPointer<s32> device_target = allocate<s32>((usize)query_count * sizeof(s32));
    GpuPointer<uint8_t> device_output = allocate<uint8_t>((usize)query_count * sizeof(uint8_t));

    memcpy(device_source.get_contents(), source_indices, (usize)query_count * sizeof(s32));
    memcpy(device_target.get_contents(), target_indices, (usize)query_count * sizeof(s32));

    gpu_k2tree_adjacent_batch(
        internal_node_words.get_contents(),
        leaf_node_words.get_contents(),
        rank_superblock_table.get_contents(),
        rank_subblock_table.get_contents(),
        branching_factor,
        superblock_size_words,
        node_count,
        padded_node_count,
        tree_height,
        internal_bit_count,
        device_source.get_contents(),
        device_target.get_contents(),
        device_output.get_contents(),
        query_count
    );
    synchronize_gpu_work();

    memcpy(output_buffer, device_output.get_contents(), (usize)query_count * sizeof(uint8_t));

    deallocate(std::move(device_source));
    deallocate(std::move(device_target));
    deallocate(std::move(device_output));
}

s32 K2Tree::trace(s32 u, s32 v) const {
    logger().trace("u={} v={}  N={} H={} Npad={} k={}",
                    u, v, node_count, tree_height, padded_node_count, branching_factor);

    if (u < 0 || u >= node_count || v < 0 || v >= node_count) {
        logger().trace("  out of bounds => 0");
        return 0;
    }
    if (tree_height == 0) {
        logger().trace("  tree_height==0 => 0");
        return 0;
    }

    const u32 *internal_words = internal_node_words.get_contents();
    const u32 *leaf_words = leaf_node_words.get_contents();
    const u32 *superblock_data = rank_superblock_table.get_contents();
    const u16 *subblock_data = rank_subblock_table.get_contents();

    s32 branching_factor_squared = branching_factor * branching_factor;

    if (tree_height == 1) {
        s32 block_size = padded_node_count / branching_factor;
        s32 child_flat_index = (u / block_size) * branching_factor + (v / block_size);
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
        s32 row_offset = u / block_size;
        s32 column_offset = v / block_size;
        s32 child_bit_position = level_bit_offset + row_offset * branching_factor + column_offset;
        s32 bit_value = (s32) get_bit(internal_words, child_bit_position);
        s32 rank_exclusive = rank1_exclusive(internal_words, superblock_data, subblock_data, child_bit_position,
                                             superblock_size_words);
        rank_inclusive = rank_exclusive + 1;

        logger().trace(
            "  lvl={:2d} block_size={:6d} u={:6d} v={:6d} row_offset={} column_offset={} child_bit_pos={:8d} T[pos]={} rank_ex={} rank_incl={}",
            level, block_size, u, v, row_offset, column_offset, child_bit_position, bit_value, rank_exclusive,
            rank_inclusive);

        if (!bit_value) {
            logger().trace("  stop: T[pos]==0 => 0");
            return 0;
        }

        if (level == tree_height - 2) {
            row_remainder = u % block_size;
            column_remainder = v % block_size;
            break;
        }

        level_bit_offset = branching_factor_squared * rank_inclusive;
        u = u % block_size;
        v = v % block_size;
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
