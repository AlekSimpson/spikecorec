//
// Created by Alek Simpson on 5/30/26.
//
#pragma once

#include <vector>
#include <optional>
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

        // Per-matrix sparse delta buffer (ticket #53/D3): Sk in arch §4.3. One
        // GPU-resident, fixed-capacity array per matrix index (parallel to
        // coefficient_vectors, same indexing), holding raw per-edge updates that
        // haven't yet been folded back into the shared U/V plane by the periodic
        // refit (ticket #54/D4, not yet implemented).
        //
        // Position-indexed, not a hash map: each array is sized exactly
        // node_count * max_neighbor_count elements — the SAME shape and indexing
        // convention neighbor_weights()'s own output array already uses (row-major
        // by source node; within a source node's row, position = that neighbor's
        // slot in the same order k2tree.get_neighbors(source_node, ...) enumerates
        // it). This is deliberate: a std::unordered_map cannot be read or written
        // from inside a Metal/CUDA compute kernel at all, and ticket #55 (IR→GPU
        // source compilation) needs to generate real loadedge/accedge kernel code
        // against Sk the same way neighbor_weights_kernel already reads
        // U_matrix/V_matrix/coefficient_vectors directly on-device. The IR-level
        // consumers of loadedge/accedge always walk adjacency inside a
        // neighbor-enumeration loop and already know their current slot index (the
        // loop counter), so the GPU hot path never searches — it indexes
        // sparse_delta_buffers[matrix_index][source_node * max_neighbor_count +
        // current_slot] directly, O(1). Only the standalone host-side point-query
        // API (get()/get_for_matrix()/accumulate_edge_delta(), called with an
        // arbitrary (source_node, target_node) pair outside a loop context) needs to
        // locate a slot by searching — see find_neighbor_slot.
        //
        // Only ever written at slots that correspond to real k^2-tree edges (see
        // accumulate_edge_delta) — loadedge/accedge are explicitly edge-scoped IR
        // ops (IR spec §3.3), unlike the raw, edge-unrestricted U*V lookups
        // get()/update() already support. Every other slot (padding beyond a node's
        // real degree, and any real edge never accumulated into) stays exactly
        // 0.0f, matching the old map's "absent key -> 0" semantics.
        Vector<GpuPointer<f32>> sparse_delta_buffers;

        // Parallel to sparse_delta_buffers: whether accumulate_edge_delta has ever
        // been called for this matrix index. A zeroed dense array can't cheaply
        // answer "has anything ever been written here" the way an empty map could,
        // so this tracks it explicitly — preserving the original "no lookup/scan at
        // all for an untouched matrix" fast path (both for get()/get_for_matrix()'s
        // bit-compatibility guarantee and to avoid apply_sparse_delta_overlay's scan
        // over the whole array in the common untouched case).
        Vector<bool> sparse_delta_touched;

        s64 node_count;
        s64 max_neighbor_count;             // upper bound on neighbors per node — bounds the padded
                                            // [node_count * max_neighbor_count] neighbor_weights output;
                                            // rows for nodes with fewer neighbors are sentinel-padded (-1)
        s64 rank;                           // latent factor dimensionality
        s64 rank_float4_stride;             // ceil(rank / 4) — float4 elements per row
        f32 constant_weight;
        bool check_indexing;
        bool using_constant_weight;

        // Number of per-edge synapse-state variables (arch §4.3's `Ck`/`Sk` family)
        // this WeightMatrix's shared U/V basis is configured to support, on top of
        // the weight itself — set by configure_per_edge_variable_count() (ticket #5
        // [C2], from a model's `.alloc` `peredge` directive count, IR spec §2). This
        // is only the COUNT the engine has committed to; it does not yet allocate or
        // seed any real per-matrix `Ck` coefficient vectors or sparse `Sk` delta
        // buffers (that's tickets #52-54/#57).
        s64 per_edge_variable_count = 0;

        // Total real edges in the k^2-tree adjacency, computed once at
        // construction by walking get_neighbors() for every node (so it stays
        // consistent with an explicitly-truncating max_neighbor_count — see
        // weight_matrix.cpp). Used by the periodic refit (ticket #54/D4) to
        // size its point cloud and by max_sparse_delta_occupancy_fraction()
        // to normalize Sk size into a fraction of the graph.
        s64 total_edge_count;

        // Refit-interval knob (ticket #54/D4, arch §4.3's "one open knob"):
        // primary tick-count trigger. A plain, directly-settable field (like
        // max_neighbor_count/rank above) rather than a constructor parameter,
        // since it is a runtime-tunable knob, not fixed structure. Default is
        // a starting-point heuristic (the D4 math memo §5: "low hundreds of
        // ticks" for typical spiking rates/dt), not a universal constant.
        s64 refit_every_n_ticks;

        // Optional secondary Sk-occupancy-threshold trigger (memo §5).
        // Negative = disabled (the default) -- refit is due only via the
        // tick-count knob above unless the caller opts in by setting this to
        // a fraction in (0, 1].
        f32 refit_occupancy_threshold_fraction;

        // Ticks elapsed since the last refit() call (or since construction, if
        // refit() has never been called). Advanced by advance_tick(), reset to
        // 0 by refit().
        s64 ticks_since_last_refit;

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

        // Sets per_edge_variable_count (arch §4.3, ticket #5 [C2]). Throws if
        // count is negative.
        void configure_per_edge_variable_count(s64 count);

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

        // Allocates one sparse delta buffer of node_count * max_neighbor_count
        // elements, zero-filled (allocate<f32> does not zero-initialize itself —
        // see weight_matrix.cpp). Returns a default-constructed (null) GpuPointer
        // when node_count * max_neighbor_count is 0 (e.g. an edge-free network),
        // to avoid a zero-byte GPU allocation — deallocate()/get_contents() are
        // never invoked on that buffer, since a matrix with no representable
        // neighbor slots can never have anything accumulated into it.
        [[nodiscard]] GpuPointer<f32> allocate_sparse_delta_buffer() const;

        // Host-side-only slot search: returns target_node's position within
        // source_node's neighbor list (the same order k2tree.get_neighbors
        // enumerates it, and the same slot convention sparse_delta_buffers/
        // neighbor_weights() use), or nullopt if target_node is not one of
        // source_node's (representable, within max_neighbor_count) neighbors. A
        // bounded linear scan over at most max_neighbor_count entries — this is the
        // ONLY place a slot is searched for; the GPU-kernel hot path (loadedge/
        // accedge inside a neighbor-enumeration loop) already knows its current
        // slot as the loop counter and never calls this. Used only by the
        // standalone host-side point-query API (get()/get_for_matrix()/
        // accumulate_edge_delta()), never on a per-tick bulk path.
        [[nodiscard]] optional<s64> find_neighbor_slot(s32 source_node, s32 target_node) const;

        // Returns Sk[matrix_index][source_node, target_node], or 0.0f if
        // target_node is not a (representable) neighbor of source_node, or if
        // matrix_index's Sk has never been touched. The untouched fast path
        // performs no slot search at all, so a never-touched Sk adds literally
        // nothing to get()/get_for_matrix() — the bit-compatibility guarantee
        // ticket #52 established for DEFAULT_MATRIX_INDEX.
        [[nodiscard]] f32 lookup_sparse_delta(s64 matrix_index, s32 source_node, s32 target_node) const;

        // Shared host-side overlay behind neighbor_weights()/neighbor_weights_for_matrix():
        // adds each real edge's Sk contribution on top of the GPU's pure low-rank
        // reconstruction already written into output_weights. sparse_delta_buffers
        // and output_weights share the exact same node_count * max_neighbor_count,
        // row-major-by-source-node, same-slot-order shape, so this is a plain
        // element-wise add (no neighbor walk needed) — padding/never-accumulated
        // slots are exactly 0.0f, contributing nothing. Skips all work (no read at
        // all) when matrix_index's Sk is untouched.
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
        // for a bad matrix_index. Locates the target's array slot via
        // find_neighbor_slot and adds delta directly at
        // sparse_delta_buffers[matrix_index][source_node * max_neighbor_count +
        // slot] — no map insertion, the slot is guaranteed to exist for a real edge
        // representable within max_neighbor_count (see find_neighbor_slot).
        void accumulate_edge_delta(s64 matrix_index, s32 source_node, s32 target_node, f32 delta);

        // ── periodic refit (ticket #54/D4) ───────────────────────────────────────
        // Advances the tick counter is_refit_due()'s tick-count trigger reads.
        // Callers (the future per-tick engine/master-kernel wiring, ticket #61)
        // invoke this once per tick.
        void advance_tick();

        // True once either the tick-count interval (refit_every_n_ticks) or, if
        // enabled, the Sk-occupancy threshold (refit_occupancy_threshold_fraction)
        // has been reached -- whichever fires first (arch §4.3 / D4 math memo §5).
        [[nodiscard]] bool is_refit_due() const;

        // The largest fraction of total_edge_count any single matrix's Sk
        // currently holds an entry for (0 if every Sk is empty, or if the graph
        // has no edges at all).
        [[nodiscard]] f32 max_sparse_delta_occupancy_fraction() const;

        // Refit (arch §4.3's "Refit" operation / ticket #54/D4): re-fits U, V,
        // and every registered matrix's Ck to the current point cloud (the
        // pre-refit reconstruction + Sk at every real edge, read once before any
        // mutation), via `sweep_count` warm-started alternating-least-squares
        // sweeps -- closed-form ridge-regularized normal-equation solves, not an
        // iterative optimizer (D4 math memo §2.1-2.4) -- then clears every
        // matrix's Sk and resets the tick-count knob. This is a periodic
        // memory-compaction/basis-freshening step, not error correction (every
        // loadedge read was already exact beforehand); it is also not a
        // learning/training step (see CLAUDE.md's U/V factorization note).
        //
        // This legitimately moves EVERY registered matrix's Ck, including
        // DEFAULT_MATRIX_INDEX's -- the same documented tradeoff
        // set_coefficient_vector() already establishes (overwriting
        // DEFAULT_MATRIX_INDEX's Ck trades away the "reduces to plain dot(U,V)"
        // bit-compatibility guarantee for this instance from that point on).
        //
        // U and V are shared across the whole matrix family, so a refit
        // triggered by drift in ONE matrix's Sk legitimately perturbs every
        // OTHER matrix's reconstruction too, by a small amount, even matrices
        // whose own Sk was empty -- expected, not a bug (D4 math memo §6).
        void refit(s32 sweep_count = 2, f32 ridge_regularization = 1e-4f);

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
