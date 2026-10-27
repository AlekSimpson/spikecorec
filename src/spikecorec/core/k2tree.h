//
// Created by Alek Simpson on 5/30/26.
//
#pragma once

#include <cstdint>
#include <vector>
#include <optional>

#include "backend/metal_backend.h"
#include "types.h"

using namespace std;

namespace spikecorec {
    #define DEFAULT_BRANCHING_FACTOR 4

    class K2_Tree {
    public:
        // The subblock table stores a u16 popcount within each superblock, which holds at most
        // 32 bits for each of 2047 words.
        static constexpr s32 MAX_SUPERBLOCK_SIZE_WORDS = 2048;

        // Defaults describe the empty adjacency; every walk returns early on tree_height 0. These are
        // s32 because the kernels read them as int. Node indices are u32, as on the GPU.
        s32 branching_factor = DEFAULT_BRANCHING_FACTOR;
        s32 superblock_size_words = 1024; // words per rank superblock
        s32 node_count = 0; // number of nodes N
        s32 padded_node_count = 0; // node_count padded up to branching_factor^tree_height
        s32 tree_height = 0; // height H of the k^2-tree
        s32 internal_bit_count = 0; // number of bits in internal_node_words

        // Bit arrays packed as u32 words; sub-ranges of the slab this tree owns.
        Engine_Pointer internal_node_words; // internal node bits, levels 0..tree_height-2
        Engine_Pointer leaf_node_words; // leaf bits, level tree_height-1
        Engine_Pointer rank_superblock_table; // rank superblock table over internal_node_words
        Engine_Pointer rank_subblock_table; // rank subblock table over internal_node_words

        // The slab holding every range above, and its backend; the destructor releases it.
        // Null when empty or moved from, so the slab is released once.
        Engine_Backend *owning_backend = nullptr;
        Engine_Pointer owning_slab;

        // lengths of the arrays above (in elements, not bytes)
        usize internal_node_words_length = 0;
        usize leaf_node_words_length = 0;
        usize rank_superblock_length = 0;
        usize rank_subblock_length = 0;

        // The empty adjacency: no nodes, no bits, no slab, no backend.
        K2_Tree() = default;
        K2_Tree(const K2_Tree&)            = delete;
        K2_Tree& operator=(const K2_Tree&) = delete;

        // A move clears the source's backend so the slab is released once.
        K2_Tree(K2_Tree&& other) noexcept;
        K2_Tree& operator=(K2_Tree&& other) noexcept;

        // Takes the ranges a factory below carved and filled; this object owns `owning_slab`.
        K2_Tree(
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
        );

        ~K2_Tree();

        // factory: build from an adjacency list (node i -> list of neighbor indices). Both factories
        // return nullopt for a branching factor outside 2-5, a superblock size outside
        // 1-MAX_SUPERBLOCK_SIZE_WORDS, or a tree too large for the kernels' 32-bit indices.
        static Optional<K2_Tree> from_adjacency_list(
            Engine_Backend &backend,
            const Vector<Vector<s32> > &adjacency_list,
            s64 node_count = -1,
            s64 branching_factor = DEFAULT_BRANCHING_FACTOR,
            s64 superblock_size = 1024
        );

        // factory: build from a flat edge list (parallel source[] and target[] arrays)
        static Optional<K2_Tree> from_edges(
            Engine_Backend &backend,
            const u32 *source_indices,
            const u32 *target_indices,
            s64 edge_count,
            s64 node_count,
            s64 branching_factor = DEFAULT_BRANCHING_FACTOR,
            s64 superblock_size = 1024
        );

        // factory: deserialize from file
        static K2_Tree load(Engine_Backend &backend, const char *path);

        // single edge query: returns 1 if the edge source_node -> target_node exists, 0 otherwise
        [[nodiscard]] s64 adjacent(u32 source_node, u32 target_node) const;

        // Writes up to max_neighbor_count targets of node_index into output_buffer, ascending,
        // and returns the count. Walks only the row's populated subtrees.
        [[nodiscard]] s64 get_neighbors(u32 node_index, u32 *output_buffer, s64 max_neighbor_count) const;

        // The column mirror of get_neighbors: writes up to max_neighbor_count sources u with an
        // edge u -> node_index, ascending, and returns the count. Walks only populated subtrees.
        [[nodiscard]] s64 get_predecessors(u32 node_index, u32 *output_buffer, s64 max_neighbor_count) const;

        // debug: logs the tree walk for source_node -> target_node; returns 1 if the edge exists
        [[nodiscard]] s64 trace(u32 source_node, u32 target_node) const;

        // serialize to file
        void save(const char *path) const;
    };
}
