//
// Created by Alek Simpson on 5/30/26.
//
#pragma once

#include <vector>
#include <unordered_map>
#include <spikecorec/core/types.h>
#include <spikecorec/core/backend.h>
#include <spikecorec/core/k2tree.h>

using namespace std;

#define MAX_RANK_FLOAT4_STRIDE 64

namespace spikecorec {
    struct WeightStats {
        f32 mean;
        f32 standard_deviation;
        f32 root_mean_square;
        f32 min_value;
        f32 max_value;
    };

    struct ScaleResult {
        f32 target_root_mean_square;
        f32 scale_factor;
        WeightStats before;
        WeightStats after;
    };

    bool can_safely_cast_s64_to_s32(s64);


    class WeightMatrix {
    public:
        // Matrix index reserved for the default/single-matrix Ck used implicitly by
        // get()/neighbor_weights()/update()/etc. — always all-ones, so those methods
        // stay bit-compatible with the pre-shared-basis (single-matrix) behavior.
        static constexpr s64 DEFAULT_MATRIX_INDEX = 0;

        K2Tree k2tree;
        GpuPointer<float4> U_matrix;        // row-major [node_count][rank_float4_stride]
        GpuPointer<float4> V_matrix;

        // Shared-basis generalization (ticket #52/D2): U_matrix/V_matrix above are the
        // one shared low-rank basis for a whole family of logical matrices (the
        // connection weight, plus future per-edge synapse state variables). Each
        // matrix `k` in the family has its own coefficient vector Ck (rank_float4_stride*4
        // scalar f32 elements — the same effective lane count get()/neighbor_weights_kernel
        // already sum over, including the padding lanes beyond the logical `rank`); entry
        // (i,j) of matrix k reconstructs as Σ U[i,r]·Ck[r]·V[j,r]. Index DEFAULT_MATRIX_INDEX
        // is the reserved single-matrix slot: all-ones, so that reduces to today's dot(U,V).
        // This is memory compression (see CLAUDE.md's U/V factorization note) — Ck is
        // supplied by the caller, not learned/fit here.
        Vector<GpuPointer<f32>> coefficient_vectors;

        // Per-matrix sparse delta buffer (ticket #53/D3): Sk in arch §4.3. One sparse
        // map per matrix index (parallel to coefficient_vectors, same indexing),
        // holding raw per-edge updates that haven't yet been folded back into the
        // shared U/V plane by the periodic refit (ticket #54/D4, not yet implemented).
        // Keyed by a packed (source_node, target_node) edge pair (see pack_edge_key);
        // only ever contains entries for pairs that ARE real k^2-tree edges (see
        // accumulate_edge_delta) — loadedge/accedge are explicitly edge-scoped IR ops
        // (IR spec §3.3), unlike the raw, edge-unrestricted U*V lookups get()/update()
        // already support. A hash map, not a literal CSR/CSC array: K2Tree exposes no
        // stable per-edge integer index to align a dense values-array against, and the
        // hot-path accumulate ("Sk[i,j] += x" on every spike arrival) wants O(1)
        // average-case point updates rather than a fixed, presized sparsity structure.
        // True CSR/CSC bulk-compaction belongs to the periodic refit (#54), which needs
        // to iterate ALL of Sk efficiently — not to this per-edge accumulate/read path.
        Vector<UnorderedMap<s64, f32>> sparse_delta_buffers;

        s64 node_count;
        s64 max_neighbor_count;             // upper bound on neighbors per node — bounds the padded
                                            // [node_count * max_neighbor_count] neighbor_weights output;
                                            // rows for nodes with fewer neighbors are sentinel-padded (-1)
        s64 rank;                           // latent factor dimensionality
        s64 rank_float4_stride;             // ceil(rank / 4) — float4 elements per row
        f32 constant_weight;
        bool check_indexing;
        bool using_constant_weight;

        WeightMatrix() = delete;

        WeightMatrix(const WeightMatrix &) = delete;

        WeightMatrix &operator=(const WeightMatrix &) = delete;

        WeightMatrix(WeightMatrix &&) = default;

        // Explicit (not defaulted): the implicitly-defaulted move-assignment operator
        // would move-assign each GpuPointer member via GpuPointer::operator=(GpuPointer&&),
        // which asserts the destination pointer is null — always false for a live
        // WeightMatrix, since its default constructor is deleted. This deallocates the
        // destination's own GPU buffers first, then move-constructs the incoming state in.
        WeightMatrix &operator=(WeightMatrix &&other) noexcept;

        // network:           adjacency list — network[i] is the list of neighbors of node i
        // rank:              latent factor dimensionality; -1 → min(64, node_count)
        // max_neighbor_count: upper bound on neighbors any single node may have; -1 → derived
        //                     from the longest row in `network`
        // weight_seed:       seeds U/V initialization for reproducible weights; -1 → seed
        //                    from std::random_device (non-deterministic)
        WeightMatrix(
            const vector<vector<s32>> &network,
            s64 rank = -1,
            bool check_indexing = true,
            s64 max_neighbor_count = -1,
            s64 weight_seed = -1
        );

        ~WeightMatrix();


        bool check_index_inbounds(s32, s32) const;
        bool check_index_inbounds(s32) const;

    private:
        // Fatally exits if `network` is empty. Called from the constructor's
        // initializer list, before k2tree(K2Tree::from_adjacency_list(...)) —
        // k2tree is the first member and does real GPU work, so this must run
        // ahead of it rather than as a body-level check after the fact.
        static const vector<vector<s32>> &validate_network(const vector<vector<s32>> &network);

        // Fatally exits if matrix_index is not a currently-registered coefficient
        // vector index (i.e. not in [0, coefficient_vectors.size())).
        void validate_matrix_index(s64 matrix_index) const;

        // Allocates one coefficient vector of length rank_float4_stride*4: the first
        // logical_coefficients.size() lanes copy logical_coefficients verbatim, every
        // remaining lane (the padding lanes beyond the logical `rank` that get()/
        // neighbor_weights_kernel still sum over — see the coefficient_vectors comment
        // in the header) is set to the literal 1.0f (a neutral multiplier, matching how
        // those padding lanes behave for the default/single-matrix case). Called with an
        // empty vector for the reserved default slot, so every lane is exactly 1.0f.
        [[nodiscard]] GpuPointer<f32> allocate_coefficient_vector(const vector<f32> &logical_coefficients) const;

        // Shared math behind get()/get_for_matrix(): the U*Ck*V reconstruction, with
        // Ck folded inline into the same accumulation loop/position as the plain U*V
        // dot product this replaces — see weight_matrix.cpp for why that placement is
        // what makes the DEFAULT_MATRIX_INDEX (all-ones Ck) case bit-identical to the
        // pre-D2 dot(U,V).
        [[nodiscard]] f32 reconstruct_entry(s32 source_node, s32 target_node, const f32 *coefficient_values) const;

        // Shared GPU dispatch behind neighbor_weights()/neighbor_weights_for_matrix().
        void dispatch_neighbor_weights(f32 *output_weights, const f32 *coefficient_values) const;

        // Packs an edge (source_node, target_node) into sparse_delta_buffers' key
        // space. Only called once (source_node, target_node) are already known to be
        // non-negative (validated by accumulate_edge_delta / bounds-checked callers),
        // so the u32 cast of target_node is safe.
        static s64 pack_edge_key(s32 source_node, s32 target_node);

        // Returns Sk[matrix_index][source_node, target_node], or 0.0f if no entry
        // exists (untouched edge, or a Sk-free/freshly-constructed matrix). The
        // empty-map fast path performs no lookup at all, so a never-touched Sk adds
        // literally nothing to get()/get_for_matrix() — the bit-compatibility
        // guarantee ticket #52 established for DEFAULT_MATRIX_INDEX.
        [[nodiscard]] f32 lookup_sparse_delta(s64 matrix_index, s32 source_node, s32 target_node) const;

        // Shared host-side overlay behind neighbor_weights()/neighbor_weights_for_matrix():
        // adds each real edge's Sk contribution on top of the GPU's pure low-rank
        // reconstruction already written into output_weights. Skips all work (no
        // neighbor walk at all) when matrix_index's Sk is empty.
        void apply_sparse_delta_overlay(f32 *output_weights, s64 matrix_index) const;

    public:

        // writes up to max_neighbor_count neighbor indices of node_index into output_buffer
        // (caller-allocated, at least max_neighbor_count elements); returns the number of
        // neighbors written. Resolved via a k^2-tree row-walk — see K2Tree::get_neighbors.
        [[nodiscard]] s64 get_neighbors(s64 node_index, s32 *output_buffer) const;

        void set_constant_weight(f32 value);

        // writes node_count * max_neighbor_count dot products into output_weights, row-major
        // by source node; slots beyond a node's actual neighbor count are sentinel-padded
        void neighbor_weights(f32 *output_weights) const;

        // ── shared-basis family (ticket #52/D2) ──────────────────────────────────
        // Registers a new matrix sharing this instance's U/V basis, with coefficient
        // vector `coefficients` (must have exactly `rank` elements — the logical rank
        // passed to the constructor). Returns the new matrix's index, to pass to
        // get_for_matrix()/neighbor_weights_for_matrix(). This is memory-compression
        // bookkeeping only (see CLAUDE.md's U/V factorization note): Ck is supplied by
        // the caller, not fit/learned here (a future refit, ticket #54, may recompute it).
        s64 add_coefficient_vector(const vector<f32> &coefficients);

        // Overwrites the coefficient vector for an already-registered matrix_index
        // (must have exactly `rank` elements). Note: matrix_index DEFAULT_MATRIX_INDEX
        // is the reserved all-ones slot get()/neighbor_weights() rely on for
        // bit-compatibility — overwriting it breaks that guarantee for this instance.
        void set_coefficient_vector(s64 matrix_index, const vector<f32> &coefficients);

        // Number of matrices currently sharing this instance's U/V basis (always >= 1;
        // index DEFAULT_MATRIX_INDEX is the built-in default/single-matrix slot).
        [[nodiscard]] s64 matrix_count() const;

        // Same as get(), but reconstructs (source_node, target_node) using matrix_index's
        // coefficient vector instead of the default single-matrix Ck.
        [[nodiscard]] f32 get_for_matrix(s32 source_node, s32 target_node, s64 matrix_index) const;

        // Same as neighbor_weights(), but reconstructs using matrix_index's coefficient
        // vector instead of the default single-matrix Ck.
        void neighbor_weights_for_matrix(f32 *output_weights, s64 matrix_index) const;

        // ── sparse delta buffer (ticket #53/D3) ──────────────────────────────────
        // accedge: Sk[matrix_index][source_node, target_node] += delta (arch §4.3
        // "Update" — cheap and local, never touches U/V or any coefficient vector).
        // Deliberately edge-scoped, unlike get()/update()'s edge-unrestricted,
        // bounds-only contract: loadedge/accedge in the IR spec only ever address
        // real edges, so (source_node, target_node) must be an actual k^2-tree edge
        // (checked via k2tree.adjacent, independently of the check_indexing flag —
        // see weight_matrix.cpp for why this doesn't reuse check_index_inbounds) or
        // this throws std::invalid_argument, the same way validate_matrix_index does
        // for a bad matrix_index.
        void accumulate_edge_delta(s64 matrix_index, s32 source_node, s32 target_node, f32 delta);

        [[nodiscard]] WeightStats neighbor_weight_stats() const;

        ScaleResult scale_neighbor_weights_to_root_mean_square(
            f32 target_root_mean_square,
            f32 epsilon = 1e-12f
        );

        [[nodiscard]] f32 get(s32 source_node, s32 target_node) const;

        void update(
            s32 source_node,
            s32 target_node,
            f32 delta,
            f32 learning_rate = 0.5f,
            f32 l2_regularization = 1.0f,
            s32 iterations = 1
        );

        void save(const char *filepath) const;

        void load_from_disk(const char *filepath);
    };
} // namespace spikecorec
