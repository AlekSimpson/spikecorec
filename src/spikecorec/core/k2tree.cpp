//
// Created by Alek Simpson on 5/30/26.
//

#include <algorithm>
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

#include "k2tree.h"
#include "types.h"
#include "backend.h"
#include "log.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::log;

static Pair<s64, s64> compute_tree_parameters(s64 node_count, s64 branching_factor) {
    if (node_count <= 1) return {0, 1};

    s64 tree_height = 0, padded_size = 1;
    while (padded_size < node_count) {
        padded_size *= branching_factor;
        tree_height++;
    }
    return {tree_height, padded_size};
}

static u32 get_bit(const u32 *words, s64 bit_index) {
    return (words[bit_index >> 5] >> (bit_index & 31)) & 1u;
}

static s64 rank1_exclusive(const u32 *internal_node_words, const u32 *superblock_table, const u16 *subblock_table,
                           s64 position, s64 superblock_size) {
    s64 word_index = position >> 5;
    s64 bit_offset = position & 31;
    s64 superblock_index = word_index / superblock_size;
    u32 superblock_base = superblock_table[superblock_index];
    u32 subblock_base = subblock_table[word_index];
    u32 partial_word_mask = (bit_offset == 0) ? 0u : ((1u << bit_offset) - 1u);
    u32 partial_word_popcount = static_cast<u32>(__builtin_popcount(internal_node_words[word_index] & partial_word_mask));
    return static_cast<s64>(superblock_base) + subblock_base + partial_word_popcount;
}

static void collect_row_neighbors(
    const u32 *internal_words, const u32 *leaf_words,
    const u32 *superblock_data, const u16 *subblock_data,
    s64 branching_factor, s64 superblock_size_words,
    s64 node_count, s64 tree_height, s64 internal_bit_count,
    s64 level, u32 row_base, u32 column_base, s64 block_size, s64 level_bit_offset,
    u32 source_node, u32 *output_buffer, s64 max_neighbor_count, s64 &neighbors_found
) {
    if (neighbors_found >= max_neighbor_count) return;

    s64 branching_factor_squared = branching_factor * branching_factor;
    s64 child_block_size = block_size / branching_factor;
    s64 row_offset = (source_node - row_base) / child_block_size;

    for (s64 col_offset = 0; col_offset < branching_factor; col_offset++) {
        if (neighbors_found >= max_neighbor_count) return;

        s64 child_flat_index = row_offset * branching_factor + col_offset;
        s64 bit_position = level_bit_offset + child_flat_index;

        if (level == tree_height - 1) {
            if (get_bit(leaf_words, bit_position)) {
                u32 target_node = static_cast<u32>(column_base + col_offset);
                if (target_node < node_count)
                    output_buffer[neighbors_found++] = target_node;
            }
        } else if (get_bit(internal_words, bit_position)) {
            s64 rank_inclusive = rank1_exclusive(internal_words, superblock_data, subblock_data,
                                                  bit_position, superblock_size_words) + 1;
            s64 raw_offset = branching_factor_squared * rank_inclusive;
            s64 child_level_bit_offset = (level + 1 == tree_height - 1)
                ? (raw_offset - internal_bit_count)
                : raw_offset;
            collect_row_neighbors(
                internal_words, leaf_words, superblock_data, subblock_data,
                branching_factor, superblock_size_words, node_count, tree_height, internal_bit_count,
                level + 1,
                static_cast<u32>(row_base + row_offset * child_block_size),
                static_cast<u32>(column_base + col_offset * child_block_size),
                child_block_size, child_level_bit_offset,
                source_node, output_buffer, max_neighbor_count, neighbors_found
            );
        }
    }
}

static void collect_column_predecessors(
    const u32 *internal_words, const u32 *leaf_words,
    const u32 *superblock_data, const u16 *subblock_data,
    s64 branching_factor, s64 superblock_size_words,
    s64 node_count, s64 tree_height, s64 internal_bit_count,
    s64 level, u32 row_base, u32 column_base, s64 block_size, s64 level_bit_offset,
    u32 target_node, u32 *output_buffer, s64 max_neighbor_count, s64 &predecessors_found
) {
    if (predecessors_found >= max_neighbor_count) return;

    s64 branching_factor_squared = branching_factor * branching_factor;
    s64 child_block_size = block_size / branching_factor;
    s64 column_offset = (target_node - column_base) / child_block_size;

    for (s64 row_offset = 0; row_offset < branching_factor; row_offset++) {
        if (predecessors_found >= max_neighbor_count) return;

        s64 child_flat_index = row_offset * branching_factor + column_offset;
        s64 bit_position = level_bit_offset + child_flat_index;

        if (level == tree_height - 1) {
            if (get_bit(leaf_words, bit_position)) {
                u32 source_node = static_cast<u32>(row_base + row_offset);
                if (source_node < node_count)
                    output_buffer[predecessors_found++] = source_node;
            }
        } else if (get_bit(internal_words, bit_position)) {
            s64 rank_inclusive = rank1_exclusive(internal_words, superblock_data, subblock_data,
                                                  bit_position, superblock_size_words) + 1;
            s64 raw_offset = branching_factor_squared * rank_inclusive;
            s64 child_level_bit_offset = (level + 1 == tree_height - 1)
                ? (raw_offset - internal_bit_count)
                : raw_offset;
            collect_column_predecessors(
                internal_words, leaf_words, superblock_data, subblock_data,
                branching_factor, superblock_size_words, node_count, tree_height, internal_bit_count,
                level + 1,
                static_cast<u32>(row_base + row_offset * child_block_size),
                static_cast<u32>(column_base + column_offset * child_block_size),
                child_block_size, child_level_bit_offset,
                target_node, output_buffer, max_neighbor_count, predecessors_found
            );
        }
    }
}

// Bits packed into u32 words, lowest bit first, appended in order.
struct Bit_Writer {
    Vector<u32> words;
    s64 bit_count = 0;

    void append(u32 bits, s64 width) {
        words.resize(static_cast<usize>((bit_count + width + 31) / 32), 0u);
        for (s64 bit = 0; bit < width; bit += 1) {
            if (((bits >> bit) & 1u) == 0u) continue;
            const s64 position = bit_count + bit;
            words[static_cast<usize>(position >> 5)] |= 1u << (position & 31);
        }
        bit_count += width;
    }

    void append_all(const Bit_Writer &other) {
        for (s64 first_bit = 0; first_bit < other.bit_count; first_bit += 32) {
            append(other.words[static_cast<usize>(first_bit >> 5)], min<s64>(32, other.bit_count - first_bit));
        }
    }
};

struct Tree_Arrays {
    Vector<u32> internal_node_words;
    Vector<u32> leaf_node_words;
    Vector<u32> rank_superblock_table;
    Vector<u16> rank_subblock_table;
    s64 tree_height;
    s64 padded_node_count;
    s64 internal_bit_count;
};

// The tree's bits are laid out level by level (breadth first), each node as its k^2 child bits. An
// edge's code lists, most significant first, the child it falls in at every level, so sorting the
// codes puts the edges in that order and a level's nodes are the codes' distinct prefixes, each
// with a bit for every next digit among its codes.
//
// row_band_edges(first_row, end_row, visit) calls visit(source, target) for every edge whose source
// is in [first_row, end_row). The edges are taken one top-level row band at a time: below the root,
// a band's nodes are contiguous at every level, so the bands' bits are appended band after band and
// only one band's codes (8 bytes an edge) are held at once. Edges with an end outside the network
// are skipped; callers refuse self-loops first.
template <typename Row_Band_Edges>
static Tree_Arrays build_tree_arrays(const Row_Band_Edges &row_band_edges, s64 node_count, s64 branching_factor,
                                     s64 superblock_size) {
    auto [tree_height, padded_node_count] = compute_tree_parameters(node_count, branching_factor);
    const s64 branching_factor_squared = branching_factor * branching_factor;

    if (tree_height == 0)
        return {{}, {}, {u32(0)}, {}, tree_height, padded_node_count, 0};

    // A code over tree_height digits of base k^2 is below padded_node_count^2, which fits a u64.
    // code / prefix_divisor[level] is the code's first `level` digits.
    Vector<u64> prefix_divisor(static_cast<usize>(tree_height + 1), 1);
    for (s64 level = tree_height - 1; level >= 0; level -= 1) {
        prefix_divisor[static_cast<usize>(level)] =
            prefix_divisor[static_cast<usize>(level + 1)] * static_cast<u64>(branching_factor_squared);
    }

    Vector<Bit_Writer> level_bits(static_cast<usize>(tree_height));
    u32 root_mask = 0;
    const s64 band_size = padded_node_count / branching_factor;
    Vector<u64> codes;
    for (s64 band = 0; band < branching_factor; band += 1) {
        codes.clear();
        row_band_edges(band * band_size, (band + 1) * band_size, [&](u32 source_node, u32 target_node) {
            if (source_node >= node_count || target_node >= node_count) return;

            u64 code = 0;
            u32 current_row = source_node;
            u32 current_column = target_node;
            s64 current_size = padded_node_count;
            for (s64 level = 0; level < tree_height; level += 1) {
                const s64 block_size = current_size / branching_factor;
                const s64 child_flat_index = (current_row / block_size) * branching_factor + current_column / block_size;
                code = code * static_cast<u64>(branching_factor_squared) + static_cast<u64>(child_flat_index);
                current_row = static_cast<u32>(current_row % block_size);
                current_column = static_cast<u32>(current_column % block_size);
                current_size = block_size;
            }
            codes.push_back(code);
        });
        if (codes.empty()) continue;
        std::sort(codes.begin(), codes.end());

        const auto digit_at = [&](u64 code, s64 level) {
            return static_cast<u32>((code / prefix_divisor[static_cast<usize>(level + 1)]) %
                                    static_cast<u64>(branching_factor_squared));
        };
        for (u64 code : codes) root_mask |= 1u << digit_at(code, 0);

        // Below the root: one node per distinct prefix, in order, as the mask of its next digits.
        for (s64 level = 1; level < tree_height; level += 1) {
            Bit_Writer &bits = level_bits[static_cast<usize>(level)];
            const u64 divisor = prefix_divisor[static_cast<usize>(level)];
            u64 node = codes.front() / divisor;
            u32 child_presence_mask = 0;
            for (u64 code : codes) {
                const u64 prefix = code / divisor;
                if (prefix != node) {
                    bits.append(child_presence_mask, branching_factor_squared);
                    child_presence_mask = 0;
                    node = prefix;
                }
                child_presence_mask |= 1u << digit_at(code, level);
            }
            bits.append(child_presence_mask, branching_factor_squared);
        }
        Vector<u64>().swap(codes);
    }
    level_bits[0].append(root_mask, branching_factor_squared);

    // Levels 0 .. tree_height - 2 hold the internal bits and the last level the leaf bits.
    Bit_Writer internal_bits;
    for (s64 level = 0; level < tree_height - 1; level += 1) internal_bits.append_all(level_bits[static_cast<usize>(level)]);
    const s64 internal_bit_count = internal_bits.bit_count;
    Vector<u32> internal_bit_words = std::move(internal_bits.words);
    Vector<u32> leaf_bit_words = std::move(level_bits[static_cast<usize>(tree_height - 1)].words);
    level_bits.clear();

    // build two-level rank tables over T
    usize total_word_count = internal_bit_words.size();
    usize superblock_count = (total_word_count + static_cast<usize>(superblock_size) - 1) / static_cast<usize>(superblock_size);
    Vector<u32> superblock_table(superblock_count + 1, 0);
    Vector<u16> subblock_table(total_word_count, 0);

    u32 running_popcount = 0;
    for (usize superblock_index = 0; superblock_index < superblock_count; superblock_index++) {
        superblock_table[superblock_index] = running_popcount;
        usize superblock_word_start = superblock_index * static_cast<usize>(superblock_size);
        usize superblock_word_end = min(total_word_count, superblock_word_start + static_cast<usize>(superblock_size));
        u32 within_superblock_popcount = 0;
        for (usize word_index = superblock_word_start; word_index < superblock_word_end; word_index++) {
            subblock_table[word_index] = static_cast<u16>(within_superblock_popcount);
            within_superblock_popcount += static_cast<u32>(__builtin_popcount(internal_bit_words[word_index]));
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

// The kernels index nodes and bits with 32-bit ints, so the tree's sizes are stored as s32. A
// leaf's bit offset is computed as internal_bit_count plus its place among the leaf bits, so
// the two counts together must fit.
static bool tree_fits_32_bit_indices(const Tree_Arrays &arrays, const char *factory_name) {
    const s64 leaf_bit_count = static_cast<s64>(arrays.leaf_node_words.size()) * 32;
    const s64 largest_index = max(arrays.padded_node_count, arrays.internal_bit_count + leaf_bit_count);
    if (largest_index <= Numeric_Limits<s32>::max()) return true;

    logger().warn("{}: rejecting a tree of {} padded nodes, {} internal bits and {} leaf bits; the GPU "
                  "indexes them with 32-bit ints", factory_name, arrays.padded_node_count,
                  arrays.internal_bit_count, leaf_bit_count);
    return false;
}

static K2_Tree make_k2tree_from_arrays(
    Engine_Backend &backend, Tree_Arrays &arrays,
    s64 node_count, s64 branching_factor, s64 superblock_size
) {
    const usize internal_node_words_length = arrays.internal_node_words.size();
    const usize leaf_node_words_length = arrays.leaf_node_words.size();
    const usize rank_superblock_length = arrays.rank_superblock_table.size();
    const usize rank_subblock_length = arrays.rank_subblock_table.size();

    spikecorec::Vector<Engine_Pointer> partitions;
    backend
        .partition(internal_node_words_length * sizeof(u32), Engine_Datatype::UNSIGNED32, partitions)
        .partition(leaf_node_words_length * sizeof(u32), Engine_Datatype::UNSIGNED32, partitions)
        .partition(rank_superblock_length * sizeof(u32), Engine_Datatype::UNSIGNED32, partitions)
        .partition(rank_subblock_length * sizeof(u16), Engine_Datatype::UNSIGNED16, partitions);

    const Engine_Pointer owning_slab = backend.allocate(partitions);

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

    return K2_Tree{
        backend,
        owning_slab,
        partitions[0],
        partitions[1],
        partitions[2],
        partitions[3],
        internal_node_words_length,
        leaf_node_words_length,
        rank_superblock_length,
        rank_subblock_length,
        static_cast<s32>(branching_factor),
        static_cast<s32>(superblock_size),
        static_cast<s32>(node_count),
        static_cast<s32>(arrays.padded_node_count),
        static_cast<s32>(arrays.tree_height),
        static_cast<s32>(arrays.internal_bit_count)
    };
}

K2_Tree::K2_Tree(
    Engine_Backend &backend,
    Engine_Pointer owning_slab,

    Engine_Pointer internal_node_words,
    Engine_Pointer leaf_node_words,
    Engine_Pointer rank_superblock_table,
    Engine_Pointer rank_subblock_table,

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
      , owning_backend(&backend)
      , owning_slab(owning_slab)
      , internal_node_words_length(internal_node_words_length)
      , leaf_node_words_length(leaf_node_words_length)
      , rank_superblock_length(rank_superblock_length)
      , rank_subblock_length(rank_subblock_length) {

    backend.advise_read_mostly(this->internal_node_words, this->internal_node_words_length * sizeof(u32));
    backend.advise_read_mostly(this->leaf_node_words, this->leaf_node_words_length * sizeof(u32));
    backend.advise_read_mostly(this->rank_superblock_table, this->rank_superblock_length * sizeof(u32));
    backend.advise_read_mostly(this->rank_subblock_table, this->rank_subblock_length * sizeof(u16));

    backend.prefetch_to_gpu(this->internal_node_words, this->internal_node_words_length * sizeof(u32));
    backend.prefetch_to_gpu(this->leaf_node_words, this->leaf_node_words_length * sizeof(u32));
    backend.prefetch_to_gpu(this->rank_superblock_table, this->rank_superblock_length * sizeof(u32));
    backend.prefetch_to_gpu(this->rank_subblock_table, this->rank_subblock_length * sizeof(u16));
}

K2_Tree::K2_Tree(K2_Tree &&other) noexcept
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
      , owning_backend(other.owning_backend)
      , owning_slab(other.owning_slab)
      , internal_node_words_length(other.internal_node_words_length)
      , leaf_node_words_length(other.leaf_node_words_length)
      , rank_superblock_length(other.rank_superblock_length)
      , rank_subblock_length(other.rank_subblock_length) {

    other.owning_backend = nullptr;
    other.owning_slab = Engine_Pointer{};
    other.tree_height = 0;
    other.node_count = 0;
}

K2_Tree &K2_Tree::operator=(K2_Tree &&other) noexcept {
    if (this == &other) return *this;

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
    owning_backend = other.owning_backend;
    owning_slab = other.owning_slab;
    internal_node_words_length = other.internal_node_words_length;
    leaf_node_words_length = other.leaf_node_words_length;
    rank_superblock_length = other.rank_superblock_length;
    rank_subblock_length = other.rank_subblock_length;

    other.owning_backend = nullptr;
    other.owning_slab = Engine_Pointer{};
    other.tree_height = 0;
    other.node_count = 0;
    return *this;
}

K2_Tree::~K2_Tree() {
    if (owning_backend != nullptr) owning_backend->deallocate_slab(owning_slab);
}

// ── factory methods ───────────────────────────────────────────────────────────

Optional<K2_Tree> K2_Tree::from_adjacency_list(
    Engine_Backend &backend,
    const Vector<Vector<s32> > &adjacency_list,
    s64 node_count,
    s64 branching_factor,
    s64 superblock_size
) {
    logger().debug("K2_Tree::from_adjacency_list: adjacency_list.size={} node_count={} branching_factor={} "
                   "superblock_size={}", adjacency_list.size(), node_count, branching_factor, superblock_size);
    // Below 2 the tree never subdivides; above 5 the k^2 child bits overflow the s32 child mask.
    if (branching_factor < 2 || branching_factor > 5) {
        logger().warn("K2_Tree::from_adjacency_list: rejecting out-of-range branching_factor={} (must be 2-5)",
                      branching_factor);
        return nullopt;
    }
    if (superblock_size < 1 || superblock_size > MAX_SUPERBLOCK_SIZE_WORDS) {
        logger().warn("K2_Tree::from_adjacency_list: rejecting out-of-range superblock_size={} (must be 1-{})",
                      superblock_size, MAX_SUPERBLOCK_SIZE_WORDS);
        return nullopt;
    }

    s64 effective_node_count = (node_count >= 0) ? node_count : static_cast<s64>(adjacency_list.size());

    // Every self-loop is refused, even one outside the network.
    for (usize source_node = 0; source_node < adjacency_list.size(); source_node++) {
        for (s32 target_node : adjacency_list[source_node]) {
            if (static_cast<s64>(target_node) != static_cast<s64>(source_node)) continue;
            throw_invalid_argument(logger(), log::format("K2_Tree: self-loop edges are not supported (node {} -> {})",
                                                         source_node, target_node));
        }
    }

    // A negative target becomes a large u32, which the build skips as out of range.
    s64 edge_count = 0;
    const auto row_band_edges = [&](s64 first_row, s64 end_row, const auto &visit) {
        const s64 last_row = min<s64>(end_row, static_cast<s64>(adjacency_list.size()));
        for (s64 source_node = first_row; source_node < last_row; source_node += 1) {
            for (s32 target_node : adjacency_list[static_cast<usize>(source_node)]) {
                visit(static_cast<u32>(source_node), static_cast<u32>(target_node));
            }
            edge_count += static_cast<s64>(adjacency_list[static_cast<usize>(source_node)].size());
        }
    };
    auto arrays = build_tree_arrays(row_band_edges, effective_node_count, branching_factor, superblock_size);
    logger().debug("K2_Tree::from_adjacency_list: edge_count={} tree_height={} padded_node_count={} "
                   "internal_bit_count={}", edge_count, arrays.tree_height, arrays.padded_node_count,
                   arrays.internal_bit_count);
    if (!tree_fits_32_bit_indices(arrays, "K2_Tree::from_adjacency_list")) return nullopt;
    return make_k2tree_from_arrays(backend, arrays, effective_node_count, branching_factor, superblock_size);
}

Optional<K2_Tree> K2_Tree::from_edges(
    Engine_Backend &backend,
    const u32 *source_indices,
    const u32 *target_indices,
    s64 edge_count,
    s64 node_count,
    s64 branching_factor,
    s64 superblock_size
) {
    logger().debug("K2_Tree::from_edges: edge_count={} node_count={} branching_factor={} superblock_size={}",
                   edge_count, node_count, branching_factor, superblock_size);
    // Below 2 the tree never subdivides; above 5 the k^2 child bits overflow the s32 child mask.
    if (branching_factor < 2 || branching_factor > 5) {
        logger().warn("K2_Tree::from_edges: rejecting out-of-range branching_factor={} (must be 2-5)",
                      branching_factor);
        return nullopt;
    }
    if (superblock_size < 1 || superblock_size > MAX_SUPERBLOCK_SIZE_WORDS) {
        logger().warn("K2_Tree::from_edges: rejecting out-of-range superblock_size={} (must be 1-{})",
                      superblock_size, MAX_SUPERBLOCK_SIZE_WORDS);
        return nullopt;
    }

    // Every self-loop is refused, even one outside the network.
    for (s64 edge_index = 0; edge_index < edge_count; edge_index++) {
        if (source_indices[edge_index] != target_indices[edge_index]) continue;
        throw_invalid_argument(logger(), log::format("K2_Tree: self-loop edges are not supported (node {} -> {})",
                                                     source_indices[edge_index], target_indices[edge_index]));
    }
    const auto row_band_edges = [&](s64 first_row, s64 end_row, const auto &visit) {
        for (s64 edge_index = 0; edge_index < edge_count; edge_index++) {
            const s64 source_node = source_indices[edge_index];
            if (source_node >= first_row && source_node < end_row) visit(source_indices[edge_index], target_indices[edge_index]);
        }
    };
    auto arrays = build_tree_arrays(row_band_edges, node_count, branching_factor, superblock_size);
    logger().debug("K2_Tree::from_edges: tree_height={} padded_node_count={} internal_bit_count={}",
                   arrays.tree_height, arrays.padded_node_count, arrays.internal_bit_count);
    if (!tree_fits_32_bit_indices(arrays, "K2_Tree::from_edges")) return nullopt;
    return make_k2tree_from_arrays(backend, arrays, node_count, branching_factor, superblock_size);
}

static constexpr u32 SAVE_MAGIC = 0x4B325452; // "K2TR"

K2_Tree K2_Tree::load(Engine_Backend &backend, const char *path) {
    logger().debug("K2_Tree::load: path={}", path);
    Input_File_Stream file(path, Stream_Base::binary);

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

    spikecorec::Vector<Engine_Pointer> partitions;
    backend
        .partition(internal_node_words_length * sizeof(u32), Engine_Datatype::UNSIGNED32, partitions)
        .partition(leaf_node_words_length * sizeof(u32), Engine_Datatype::UNSIGNED32, partitions)
        .partition(rank_superblock_length * sizeof(u32), Engine_Datatype::UNSIGNED32, partitions)
        .partition(rank_subblock_length * sizeof(u16), Engine_Datatype::UNSIGNED16, partitions);

    const Engine_Pointer owning_slab = backend.allocate(partitions);

    if (internal_node_words_length > 0) {
        file.read(reinterpret_cast<char *>(partitions[0].get_contents()),
                  static_cast<streamsize>(internal_node_words_length * sizeof(u32)));
    }
    if (leaf_node_words_length > 0) {
        file.read(reinterpret_cast<char *>(partitions[1].get_contents()),
                  static_cast<streamsize>(leaf_node_words_length * sizeof(u32)));
    }
    if (rank_superblock_length > 0) {
        file.read(reinterpret_cast<char *>(partitions[2].get_contents()),
                  static_cast<streamsize>(rank_superblock_length * sizeof(u32)));
    }
    if (rank_subblock_length > 0) {
        file.read(reinterpret_cast<char *>(partitions[3].get_contents()),
                  static_cast<streamsize>(rank_subblock_length * sizeof(u16)));
    }

    logger().debug("K2_Tree::load: path={} magic={:#x} node_count={} branching_factor={} tree_height={} "
                   "padded_node_count={} internal_bit_count={}",
                   path, magic, node_count, branching_factor, tree_height, padded_node_count, internal_bit_count);

    return K2_Tree{
        backend,
        owning_slab,
        partitions[0],
        partitions[1],
        partitions[2],
        partitions[3],
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

void K2_Tree::save(const char *path) const {
    logger().debug("K2_Tree::save: path={} node_count={} branching_factor={} tree_height={} "
                   "padded_node_count={} internal_bit_count={}",
                   path, node_count, branching_factor, tree_height, padded_node_count, internal_bit_count);
    Output_File_Stream file(path, Stream_Base::binary);

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
               static_cast<streamsize>(internal_node_words_length * sizeof(u32)));
    file.write(reinterpret_cast<const char *>(leaf_node_data), static_cast<streamsize>(leaf_node_words_length * sizeof(u32)));
    file.write(reinterpret_cast<const char *>(rank_superblock_data),
               static_cast<streamsize>(rank_superblock_length * sizeof(u32)));
    file.write(reinterpret_cast<const char *>(rank_subblock_data), static_cast<streamsize>(rank_subblock_length * sizeof(u16)));
}

// ── queries ───────────────────────────────────────────────────────────────────

s64 K2_Tree::adjacent(u32 source_node, u32 target_node) const {
    if (source_node >= static_cast<u32>(node_count) || target_node >= static_cast<u32>(node_count)) return 0;
    if (tree_height == 0) return 0;

    const u32 *internal_words = internal_node_words.get_contents_as<u32>();
    const u32 *leaf_words = leaf_node_words.get_contents_as<u32>();
    const u32 *superblock_data = rank_superblock_table.get_contents_as<u32>();
    const u16 *subblock_data = rank_subblock_table.get_contents_as<u16>();

    s64 branching_factor_squared = branching_factor * branching_factor;

    if (tree_height == 1) {
        s64 block_size = padded_node_count / branching_factor;
        s64 child_flat_index =
            (source_node / block_size) * branching_factor + (target_node / block_size);
        return static_cast<s64>(get_bit(leaf_words, child_flat_index));
    }

    s64 level_bit_offset = 0, current_block_size = padded_node_count, rank_inclusive = 0;

    for (s64 level = 0; level < tree_height - 1; level++) {
        s64 block_size = current_block_size / branching_factor;
        s64 row_offset = source_node / block_size;
        s64 column_offset = target_node / block_size;
        s64 child_bit_position = level_bit_offset + row_offset * branching_factor + column_offset;

        if (!get_bit(internal_words, child_bit_position)) return 0;

        rank_inclusive = rank1_exclusive(internal_words, superblock_data, subblock_data, child_bit_position,
                                         superblock_size_words) + 1;

        if (level == tree_height - 2) {
            source_node = static_cast<u32>(source_node % block_size);
            target_node = static_cast<u32>(target_node % block_size);
            break;
        }

        level_bit_offset = branching_factor_squared * rank_inclusive;
        source_node = static_cast<u32>(source_node % block_size);
        target_node = static_cast<u32>(target_node % block_size);
        current_block_size = block_size;
    }

    s64 leaf_bit_offset = branching_factor_squared * rank_inclusive - internal_bit_count;
    return static_cast<s64>(get_bit(leaf_words,
                         leaf_bit_offset + source_node * branching_factor + target_node));
}

s64 K2_Tree::get_neighbors(u32 node_index, u32 *output_buffer, s64 max_neighbor_count) const {
    if (max_neighbor_count <= 0) return 0;
    if (node_index >= static_cast<u32>(node_count) || tree_height == 0) return 0;

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

s64 K2_Tree::get_predecessors(u32 node_index, u32 *output_buffer, s64 max_neighbor_count) const {
    if (max_neighbor_count <= 0) return 0;
    if (node_index >= static_cast<u32>(node_count) || tree_height == 0) return 0;

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

s64 K2_Tree::trace(u32 source_node, u32 target_node) const {
    logger().trace("source_node={} target_node={}  N={} H={} Npad={} k={}",
                    source_node, target_node, node_count, tree_height, padded_node_count,
                    branching_factor);

    if (source_node >= static_cast<u32>(node_count) || target_node >= static_cast<u32>(node_count)) {
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

    s64 branching_factor_squared = branching_factor * branching_factor;

    if (tree_height == 1) {
        s64 block_size = padded_node_count / branching_factor;
        s64 child_flat_index =
            (source_node / block_size) * branching_factor + (target_node / block_size);
        s64 leaf_bit = static_cast<s64>(get_bit(leaf_words, child_flat_index));
        logger().trace("  tree_height==1: block_size={} child_flat_index={}  L[{}]={}",
                       block_size, child_flat_index, child_flat_index, leaf_bit);
        return leaf_bit;
    }

    s64 level_bit_offset = 0;
    s64 rank_inclusive = 0;
    s64 current_block_size = padded_node_count;
    u32 row_remainder = 0, column_remainder = 0;

    for (s64 level = 0; level < tree_height - 1; level++) {
        s64 block_size = current_block_size / branching_factor;
        s64 row_offset = source_node / block_size;
        s64 column_offset = target_node / block_size;
        s64 child_bit_position = level_bit_offset + row_offset * branching_factor + column_offset;
        s64 bit_value = static_cast<s64>(get_bit(internal_words, child_bit_position));
        s64 rank_exclusive = rank1_exclusive(internal_words, superblock_data, subblock_data, child_bit_position,
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
            row_remainder = static_cast<u32>(source_node % block_size);
            column_remainder = static_cast<u32>(target_node % block_size);
            break;
        }

        level_bit_offset = branching_factor_squared * rank_inclusive;
        source_node = static_cast<u32>(source_node % block_size);
        target_node = static_cast<u32>(target_node % block_size);
        current_block_size = block_size;
    }

    s64 leaf_bit_offset = branching_factor_squared * rank_inclusive - internal_bit_count;
    s64 leaf_child_index = row_remainder * branching_factor + column_remainder;
    s64 leaf_bit_index = leaf_bit_offset + leaf_child_index;
    s64 leaf_bit = static_cast<s64>(get_bit(leaf_words, leaf_bit_index));

    logger().trace(
        "  leaf: rank_incl={} leaf_bit_offset={} row_remainder={} column_remainder={} leaf_child_index={} -> L[{}]={}",
        rank_inclusive, leaf_bit_offset, row_remainder, column_remainder, leaf_child_index, leaf_bit_index, leaf_bit);

    return leaf_bit;
}
