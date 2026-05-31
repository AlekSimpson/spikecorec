//
// Created by Alek Simpson on 5/30/26.
//
#pragma once

#include <cstdint>
#include <vector>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/types.h"

using namespace std;

namespace spikecorec {
    #define DEFAULT_BRANCHING_FACTOR 4

    class K2Tree {
    public:
        // metadata
        s32 branching_factor; // k value, default 2
        s32 superblock_size_words; // words per rank superblock, default 1024
        s32 node_count; // number of nodes N
        s32 padded_node_count; // node_count padded up to branching_factor^tree_height
        s32 tree_height; // height H of the k^2-tree
        s32 internal_bit_count; // number of bits in internal_node_words

        // bit arrays packed as uint32 words — allocated in unified memory, readable from CPU and GPU
        GpuPointer<u32> internal_node_words; // internal node bits, levels 0..tree_height-2
        GpuPointer<u32> leaf_node_words; // leaf bits, level tree_height-1
        GpuPointer<u32> rank_superblock_table; // rank superblock table over internal_node_words
        GpuPointer<u16> rank_subblock_table; // rank subblock table over internal_node_words

        // lengths of the arrays above (in elements, not bytes)
        usize internal_node_words_length;
        usize leaf_node_words_length;
        usize rank_superblock_length;
        usize rank_subblock_length;

        K2Tree() = delete;
        K2Tree(const K2Tree&)            = delete;
        K2Tree& operator=(const K2Tree&) = delete;
        K2Tree(K2Tree&&)                 = default;
        K2Tree& operator=(K2Tree&&)      = default;

        K2Tree(
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
        );

        ~K2Tree();

        // factory: build from an adjacency list (node i -> list of neighbor indices)
        static K2Tree from_adjacency_list(
            const vector<vector<s32> > &adjacency_list,
            s32 node_count = -1,
            s32 branching_factor = DEFAULT_BRANCHING_FACTOR,
            s32 superblock_size = 1024
        );

        // factory: build from a flat edge list (parallel source[] and target[] arrays)
        static K2Tree from_edges(
            const s32 *source_indices,
            const s32 *target_indices,
            s32 edge_count,
            s32 node_count,
            s32 branching_factor = DEFAULT_BRANCHING_FACTOR,
            s32 superblock_size = 1024
        );

        // factory: deserialize from file
        static K2Tree load(const char *path);

        // single edge query: returns 1 if edge (u,v) exists, 0 otherwise
        [[nodiscard]] s32 adjacent(s32 u, s32 v) const;

        // batched edge query: writes 0 or 1 into output_buffer[i] for each (source_indices[i], target_indices[i])
        void adjacent_batch(
            const s32 *source_indices,
            const s32 *target_indices,
            uint8_t *output_buffer,
            s32 query_count
        ) const;

        // debug: prints the full tree traversal path for (u,v) and returns the result
        [[nodiscard]] s32 trace(s32 u, s32 v) const;

        // serialize to file
        void save(const char *path) const;
    };
}
