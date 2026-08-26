//
// Created by Alek Simpson on 5/30/26.
//
#pragma once

#include <cstdint>
#include <vector>
#include <optional>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/types.h"

using namespace std;

namespace spikecorec {
    #define DEFAULT_BRANCHING_FACTOR 4

    class K2Tree {
    public:
        // Upper bound on the queries adjacent_batch stages in one pass. The staging
        // buffers are carved from this tree's own slab at construction, because the
        // backend hands out one chunk per partition -> allocate round and there is no
        // per-call allocation to fall back on. A larger batch is chunked, not refused.
        static constexpr s64 ADJACENT_BATCH_QUERY_CAP = 65536;

        // metadata. Initialized so a default-constructed K2Tree is the empty adjacency
        // rather than uninitialized memory: tree_height 0 is what every row walk, on host
        // and on device, already checks before descending.
        s32 branching_factor = DEFAULT_BRANCHING_FACTOR;
        s32 superblock_size_words = 1024; // words per rank superblock
        s32 node_count = 0; // number of nodes N
        s32 padded_node_count = 0; // node_count padded up to branching_factor^tree_height
        s32 tree_height = 0; // height H of the k^2-tree
        s32 internal_bit_count = 0; // number of bits in internal_node_words

        // Bit arrays packed as uint32 words. All four are sub-ranges of one slab that this
        // tree owns; none of them owns anything itself, which is what makes them plain
        // copyable values rather than the move-only handles they used to be.
        EnginePointer internal_node_words; // internal node bits, levels 0..tree_height-2
        EnginePointer leaf_node_words; // leaf bits, level tree_height-1
        EnginePointer rank_superblock_table; // rank superblock table over internal_node_words
        EnginePointer rank_subblock_table; // rank subblock table over internal_node_words

        // Staging for adjacent_batch, carved from the same slab (see ADJACENT_BATCH_QUERY_CAP).
        EnginePointer query_source_staging;
        EnginePointer query_target_staging;
        EnginePointer query_output_staging;

        // The backend this tree's slab came from, and the whole-chunk handle allocate()
        // returned for it. The destructor releases that one chunk, which frees every
        // sub-range above at once. Null on a default-constructed (empty) tree, and nulled
        // in the moved-from object so the chunk is released exactly once.
        EngineBackend *owning_backend = nullptr;
        EnginePointer owning_slab;

        // The ahead-of-time batch-query kernel, built into a pipeline on first use. Mutable
        // because adjacent_batch is const and building a pipeline is caching, not a change
        // to the adjacency this tree represents.
        mutable EngineFunction adjacent_batch_function;

        // lengths of the arrays above (in elements, not bytes)
        usize internal_node_words_length = 0;
        usize leaf_node_words_length = 0;
        usize rank_superblock_length = 0;
        usize rank_subblock_length = 0;

        // The empty adjacency: no nodes, no bits, no slab, no backend. Destructing one is
        // safe because the destructor no-ops on a null owning_backend, and every walk
        // bails on tree_height == 0 before touching an array.
        K2Tree() = default;
        K2Tree(const K2Tree&)            = delete;
        K2Tree& operator=(const K2Tree&) = delete;

        // Hand-written rather than defaulted, for one reason only: the moved-from object
        // must forget its backend, or both objects release the same slab. Every other
        // member is a plain value that a defaulted move would handle correctly.
        K2Tree(K2Tree&& other) noexcept;
        K2Tree& operator=(K2Tree&& other) noexcept;

        // Takes the ranges already carved and filled by a factory below. `owning_slab` is
        // the whole-chunk handle its partition -> allocate round returned; this object
        // owns it and releases it in the destructor.
        K2Tree(
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
        );

        ~K2Tree();

        // factory: build from an adjacency list (node i -> list of neighbor indices)
        static optional<K2Tree> from_adjacency_list(
            EngineBackend &backend,
            const vector<vector<s32> > &adjacency_list,
            s32 node_count = -1,
            s32 branching_factor = DEFAULT_BRANCHING_FACTOR,
            s32 superblock_size = 1024
        );

        // factory: build from a flat edge list (parallel source[] and target[] arrays)
        static optional<K2Tree> from_edges(
            EngineBackend &backend,
            const s32 *source_indices,
            const s32 *target_indices,
            s32 edge_count,
            s32 node_count,
            s32 branching_factor = DEFAULT_BRANCHING_FACTOR,
            s32 superblock_size = 1024
        );

        // factory: deserialize from file
        static K2Tree load(EngineBackend &backend, const char *path);

        // single edge query: returns 1 if the edge source_node -> target_node exists, 0 otherwise
        [[nodiscard]] s32 adjacent(s32 source_node, s32 target_node) const;

        // row enumeration: writes up to max_neighbor_count neighbor indices of `node_index`
        // into output_buffer (caller-allocated, at least max_neighbor_count elements), in
        // tree-traversal order. Returns the number of neighbors written (<= max_neighbor_count).
        // Walks only the populated subtrees of the row — never touches unrelated regions.
        [[nodiscard]] s64 get_neighbors(s32 node_index, s32 *output_buffer, s64 max_neighbor_count) const;

        // column enumeration (the exact mirror of get_neighbors): writes up to max_neighbor_count
        // PREDECESSOR indices of `node_index` into output_buffer — every node `u` for which the edge
        // u -> node_index exists (i.e. `node_index` acting as the target/column). Returns the number
        // written (<= max_neighbor_count). Reads the SAME stored bits get_neighbors does; it simply
        // fixes the column (from node_index) and walks the rows at each tree level instead of fixing
        // the row and walking the columns. Walks only populated subtrees of the column.
        [[nodiscard]] s64 get_predecessors(s32 node_index, s32 *output_buffer, s64 max_neighbor_count) const;

        // batched edge query: writes 0 or 1 into output_buffer[i] for each (source_indices[i], target_indices[i])
        void adjacent_batch(
            const s32 *source_indices,
            const s32 *target_indices,
            uint8_t *output_buffer,
            s32 query_count
        ) const;

        // debug: prints the full tree traversal path for the edge source_node -> target_node
        // and returns the result
        [[nodiscard]] s32 trace(s32 source_node, s32 target_node) const;

        // serialize to file
        void save(const char *path) const;
    };
}
