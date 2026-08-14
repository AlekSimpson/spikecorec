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

        // Which matrix index in the shared-basis family above holds per-edge spike
        // delay (ticket #64/F3's future consumer), or -1 when no per-edge delay has
        // ever been set on this instance. Delay is per-edge state, so it is stored
        // exactly the way every other per-edge state variable is — as another
        // registered matrix in the Ck/Sk family — rather than as its own flat
        // [node_count * max_neighbor_count] array, which is precisely the dense
        // per-edge storage the k^2-tree + low-rank factorization exists to avoid.
        //
        // Registered lazily by the first successful set_edge_delay_ticks() call, so
        // a WeightMatrix that never sets a per-edge delay (the default — see
        // constant_delay_ticks below) allocates nothing for delay at all.
        //
        // This matrix's Ck is pinned to all-zero in EVERY lane, padding lanes
        // included, so its low-rank term is identically 0.0f and a delay reads back
        // as exactly its Sk entry, independent of whatever U and V currently hold.
        // That is what makes delays survive refit()'s U/V re-fit unchanged — see
        // set_edge_delay_ticks/refit in weight_matrix.cpp.
        s64 delay_matrix_index = -1;

        s64 node_count = 0;
        s64 max_neighbor_count = 0;             // upper bound on neighbors per node — bounds the padded
                                            // [node_count * max_neighbor_count] neighbor_weights output;
                                            // rows for nodes with fewer neighbors are sentinel-padded (-1)
        s64 rank = 0;                       // latent factor dimensionality
        s64 rank_float4_stride = 0;         // ceil(rank / 4) — float4 elements per row
        f32 constant_weight = 0.0f;
        bool check_indexing = true;
        bool using_constant_weight = false;

        // True once set_edge_weight() has switched the DEFAULT_MATRIX_INDEX weight matrix
        // into EXACT mode: its Ck is pinned to all-zero in every lane (padding included), so
        // its low-rank term is identically 0.0f and an edge's weight reads back as exactly
        // its Sk entry — the same device get_edge_delay_ticks/delay_matrix_index already
        // relies on, and for the same reason.
        //
        // Why a mode rather than just writing a delta: U/V are seeded from N(0,1), so the
        // reconstruction at any edge is of order 1. Expressing an exact weight `w` as the
        // delta `w - reconstruction` cannot survive f32 when |w| << |reconstruction| — the
        // sum reconstruction + Sk carries only ulp(reconstruction) ≈ 6e-8·|reconstruction|
        // of absolute resolution, so a realistic synaptic weight (NeuroML routinely
        // specifies conductances and currents at 1e-9 to 1e-12 in SI) is rounded away
        // entirely and reads back as 0. Pinning the low-rank term to zero is what removes
        // that error term: the stored value IS the weight, exact at every magnitude, and
        // the GPU propagate kernel reproduces it bit-for-bit because it reconstructs from
        // this same Ck and Sk pair.
        //
        // This is a REPRESENTATION change, not new storage (see CLAUDE.md's U/V
        // factorization note): Sk for the default matrix is already allocated at
        // construction, so exact weights cost no memory beyond what an ordinary
        // WeightMatrix already pays, and the k^2-tree still compresses the adjacency
        // itself. What it gives up is the low-rank plane's ability to summarize the weights
        // — which is exactly the trade the delay matrix already makes, and which is only
        // ever made by a caller that has real per-edge values to store.
        //
        // Consequences worth knowing, all documented on the methods concerned:
        //   - get() returns 0.0f for a pair that is not a real edge (there is no Sk slot
        //     for one), instead of the pre-exact-mode random reconstruction.
        //   - update()/scale_neighbor_weights_to_root_mean_square(), which move U/V, no
        //     longer move the weights. Per-edge updates (plasticity) go through
        //     accumulate_edge_delta, which stays exact in this mode because it adds
        //     directly onto the stored weight.
        //   - refit() leaves the default matrix's Sk alone, the same way it leaves the
        //     delay matrix's alone.
        // Never set on a WeightMatrix that is only ever used as a random reservoir: nothing
        // turns this on but set_edge_weight().
        bool using_exact_edge_weights = false;

        // Delay (in whole ticks) every edge uses when using_constant_delay_ticks is
        // true. Defaults to 1 — the engine's existing implicit one-tick
        // network_inputs latency (CLAUDE.md's engine execution model) — so an
        // ordinary WeightMatrix, constructed the same way it always has been, needs
        // zero new caller-side work to keep behaving exactly like today's undelayed
        // engine.
        s32 constant_delay_ticks = 1;

        // True by default (see constant_delay_ticks): whether a "no explicit delay
        // configured" WeightMatrix should read as constant_delay_ticks everywhere,
        // matching the using_constant_weight/constant_weight pattern above exactly.
        // set_constant_delay_ticks() sets this true as a side effect;
        // set_edge_delay_ticks() does NOT flip it — whatever code reads delay in a
        // future stage is responsible for choosing per-edge vs constant, the same
        // way propagate-kernel dispatch already chooses constant_weight vs U*V via
        // using_constant_weight.
        bool using_constant_delay_ticks = true;

        // Number of per-edge synapse-state variables (arch §4.3's `Ck`/`Sk` family)
        // this WeightMatrix's shared U/V basis carries, on top of the weight itself —
        // set by configure_per_edge_variable_count() (ticket #5 [C2], from a model's
        // `.alloc` `peredge` directive count, IR spec §2).
        s64 per_edge_variable_count = 0;

        // Family index of per-edge variable 0. The remaining per_edge_variable_count - 1
        // variables follow it consecutively, so variable v is matrix
        // per_edge_variable_matrix_base + v. -1 until configure_per_edge_variable_count()
        // has registered them (and whenever the count is zero).
        s64 per_edge_variable_matrix_base = -1;

        // Storage behind those matrices: one contiguous
        // [per_edge_variable_count][node_count * max_neighbor_count] f32 allocation,
        // VARIABLE-MAJOR — variable v's plane starts at v * node_count *
        // max_neighbor_count, and inside a plane the slot convention is the one every
        // other Sk uses (row-major by source node; within a row, the neighbour's
        // position in k^2-tree traversal order). Each per-edge variable matrix's entry
        // in sparse_delta_buffers is therefore a NULL handle owning nothing: its Sk
        // plane lives here instead, and sparse_delta_data_for() is what resolves either
        // spelling to a base pointer.
        //
        // Contiguous rather than one allocation per matrix because the generated kernel
        // reads and writes these planes ON DEVICE, once per spike per out-edge. A
        // Metal/CUDA kernel takes one pointer per argument and the master kernel's
        // argument table is nearly full, so one buffer per state variable is not
        // available; one buffer with baked plane offsets is exactly the consolidation
        // the argument-table comment in nml/kernel_codegen.h calls for.
        //
        // Like per-edge delay and exact per-edge weights, every one of these matrices
        // has its Ck pinned to all-zero in EVERY lane, padding lanes included, so its
        // low-rank term is identically 0.0f and the stored plane IS the value. That is
        // what keeps a synapse state variable exact at realistic SI magnitudes (1e-12
        // and smaller) and invariant under refit()'s re-fit of the shared basis.
        //
        // This is still memory compression bookkeeping, not learning (see CLAUDE.md's
        // U/V factorization note): the k^2-tree keeps compressing the adjacency, and
        // nothing here is fit to data.
        GpuPointer<f32> per_edge_variable_values;

        // Total real edges in the k^2-tree adjacency, computed once at
        // construction by walking get_neighbors() for every node (so it stays
        // consistent with an explicitly-truncating max_neighbor_count — see
        // weight_matrix.cpp). Used by the periodic refit (ticket #54/D4) to
        // size its point cloud and by max_sparse_delta_occupancy_fraction()
        // to normalize Sk size into a fraction of the graph.
        s64 total_edge_count = 0;

        // Refit-interval knob (ticket #54/D4, arch §4.3's "one open knob"):
        // primary tick-count trigger. A plain, directly-settable field (like
        // max_neighbor_count/rank above) rather than a constructor parameter,
        // since it is a runtime-tunable knob, not fixed structure. Default is
        // a starting-point heuristic (the D4 math memo §5: "low hundreds of
        // ticks" for typical spiking rates/dt), not a universal constant.
        static constexpr s64 DEFAULT_REFIT_EVERY_N_TICKS = 200;
        s64 refit_every_n_ticks = DEFAULT_REFIT_EVERY_N_TICKS;

        // Optional secondary Sk-occupancy-threshold trigger (memo §5).
        // Negative = disabled (the default) -- refit is due only via the
        // tick-count knob above unless the caller opts in by setting this to
        // a fraction in (0, 1].
        f32 refit_occupancy_threshold_fraction = -1.0f;

        // Ticks elapsed since the last refit() call (or since construction, if
        // refit() has never been called). Advanced by advance_tick(), reset to
        // 0 by refit().
        s64 ticks_since_last_refit = 0;

        // A network with no connections: an empty k^2-tree, no U/V basis, no per-edge
        // storage. This is what an engine holds before it has parsed a model, and what it
        // keeps for a model whose populations are never wired together -- both are ordinary
        // states rather than errors, so they are the default rather than an absent value.
        WeightMatrix() = default;

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

        // Registers `count` per-edge variable matrices in the shared-basis family and
        // allocates the contiguous zero-filled plane block behind them (arch §4.3,
        // ticket #5 [C2]). Each new matrix's Ck is pinned to all-zero in every lane, so
        // its stored plane is the whole value — the same representation per-edge delay
        // and exact per-edge weights already use, and for the same reason (see
        // per_edge_variable_values and using_exact_edge_weights).
        //
        // Idempotent for a repeated identical count. Throws std::invalid_argument if
        // count is negative, or if a DIFFERENT non-zero count is configured after the
        // first call: the already-registered matrices would be orphaned in the family
        // and their planes would be read at the wrong offsets, which is silent.
        void configure_per_edge_variable_count(s64 count);

        // The family index of per-edge variable `variable_index`. Throws
        // std::invalid_argument when the index is outside [0, per_edge_variable_count).
        [[nodiscard]] s64 per_edge_variable_matrix_index(s64 variable_index) const;

        // Whether `matrix_index` is one of the per-edge variable matrices. Those are
        // exempt from refit()'s Ck re-fit and from its Sk clear, and from the Sk
        // occupancy count, on exactly the grounds per-edge delay already is: their plane
        // is permanent storage the model reads back, not drift awaiting a refit.
        [[nodiscard]] bool is_per_edge_variable_matrix(s64 matrix_index) const;

        // Absolute per-edge point setter/getter for synapse state variable
        // `variable_index` on the real k^2-tree edge (source_node, target_node) — the
        // mirror of set_edge_weight()/set_edge_delay_ticks(), with the same edge-scoped
        // contract and the same absolute-write (never delta-against-a-reconstruction)
        // discipline that keeps 1e-12 values from being rounded away.
        //
        // Throws std::invalid_argument on a bad variable index, an out-of-bounds pair, a
        // pair that is not a real edge, or a real edge with no slot within
        // max_neighbor_count.
        void set_edge_variable(s64 variable_index, s32 source_node, s32 target_node, f32 value);

        [[nodiscard]] f32 get_edge_variable(s64 variable_index, s32 source_node,
                                            s32 target_node) const;

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

        // Forces every lane of `matrix_index`'s Ck to exactly 0.0f, padding lanes
        // included (add_coefficient_vector fills those with the neutral 1.0f — see
        // allocate_coefficient_vector — which would leave a nonzero low-rank term
        // whenever rank is not a multiple of 4). That pinning is what makes a matrix's
        // stored Sk entry its WHOLE value, invariant under any later refit of U/V.
        // Shared by the three representations that depend on it: per-edge delay, exact
        // per-edge weights, and the per-edge variable family.
        void pin_coefficient_vector_to_zero(s64 matrix_index);

        // The base of `matrix_index`'s Sk storage: its own allocation for an ordinary
        // matrix, or its plane inside the contiguous per_edge_variable_values block for
        // a per-edge variable matrix. Null when there are no representable neighbour
        // slots at all (an edge-free network), which is the one case no Sk is allocated
        // for. Every Sk read and write goes through this rather than indexing
        // sparse_delta_buffers directly, so the two storage spellings cannot diverge.
        [[nodiscard]] f32 *sparse_delta_data_for(s64 matrix_index);
        [[nodiscard]] const f32 *sparse_delta_data_for(s64 matrix_index) const;

        // Registers delay_matrix_index if it has not been registered yet, via the
        // ordinary add_coefficient_vector() family path, then pins that matrix's Ck to
        // all-zero. Idempotent.
        void ensure_delay_matrix_registered();

        // Switches the default matrix into exact mode (see using_exact_edge_weights), in
        // two steps that together leave every real edge reading back EXACTLY what it read
        // back before the call: first each real edge's current value (its low-rank
        // reconstruction plus whatever its Sk slot already held) is written into that Sk
        // slot, then every lane of the default matrix's Ck — padding lanes included — is
        // forced to 0.0f. The migration is what makes turning the mode on a lossless
        // change of representation rather than a silent reset of every edge the caller
        // has not written yet: the value materialized into Sk is the same f32 get()
        // already returned, so it round-trips bit-for-bit. Costs one reconstruction per
        // real edge, once, and is idempotent.
        void enable_exact_edge_weights();

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

        // writes up to max_neighbor_count PREDECESSOR indices of node_index into output_buffer
        // (caller-allocated, at least max_neighbor_count elements) — every node `u` with an edge
        // u -> node_index; returns the number written. The exact mirror of get_neighbors, resolved
        // via a k^2-tree column-walk — see K2Tree::get_predecessors.
        [[nodiscard]] s64 get_predecessors(s64 node_index, s32 *output_buffer) const;

        void set_constant_weight(f32 value);

        // Per-edge point-setter for the connection weight (the mirror of
        // set_edge_delay_ticks below): stores `weight` for the real k^2-tree edge
        // (source_node, target_node) so that get(source_node, target_node) returns it back
        // to within f32 representation error, at ANY magnitude — 1e-12 and 1e3 alike.
        // Throws std::invalid_argument if (source_node, target_node) is not a real edge, or
        // is a real edge with no slot within max_neighbor_count, on the same edge-scoped
        // contract accumulate_edge_delta documents.
        //
        // This is an absolute SET, not an accumulate: it writes the value into the edge's
        // Sk slot directly rather than adding `weight - get(...)` on top of a reconstruction
        // that may be many orders of magnitude larger, which is precisely the subtraction
        // that silently annihilates realistic synaptic weights (see using_exact_edge_weights
        // for the arithmetic). Callers with a real per-edge weight must use this rather than
        // building a delta themselves.
        //
        // The first call switches this WeightMatrix into exact mode (see
        // using_exact_edge_weights and enable_exact_edge_weights) — value-preservingly, so
        // every OTHER edge still reads back exactly what it did before. Subsequent calls
        // just overwrite one slot.
        void set_edge_weight(s32 source_node, s32 target_node, f32 weight);

        // Sets constant_delay_ticks to `ticks` and using_constant_delay_ticks to
        // true (mirroring set_constant_weight()'s using_constant_weight side
        // effect). Throws std::invalid_argument if ticks < 1 — delay is always at
        // least 1 tick.
        void set_constant_delay_ticks(s32 ticks);

        // Per-edge point-setter for spike delay (mirrors accumulate_edge_delta()'s
        // shape): sets the delay for the real k^2-tree edge (source_node,
        // target_node) to delay_ticks. Throws std::invalid_argument if (source_node,
        // target_node) is not a real edge (checked via k2tree.adjacent, same as
        // accumulate_edge_delta — see its own comment for why this doesn't reuse
        // check_index_inbounds), or if delay_ticks < 1. Does NOT flip
        // using_constant_delay_ticks — see that field's own header comment.
        //
        // Delay is stored as the delay_matrix_index member of the shared-basis
        // family, not in a flat per-edge array; callers do not need to know that.
        void set_edge_delay_ticks(s32 source_node, s32 target_node, s32 delay_ticks);

        // Reads back the delay, in whole ticks, of the real k^2-tree edge
        // (source_node, target_node). Throws std::invalid_argument on the same
        // conditions set_edge_delay_ticks does (out of bounds, not a real edge, not
        // representable within max_neighbor_count) — this is an edge-scoped
        // operation with the same contract.
        //
        // Returns constant_delay_ticks for any real edge that has never been given a
        // per-edge delay of its own. That fallback is unambiguous because a delay is
        // always >= 1, so this matrix's stored value of exactly 0.0f can only mean
        // "never set" (the same "absent entry -> 0.0f" convention every other Sk
        // uses). Like get()/using_constant_weight, this accessor always performs the
        // real per-edge lookup and never branches on using_constant_delay_ticks,
        // which stays purely the dispatch hint its own comment describes.
        //
        // The stored value is an f32 reconstruction, so it is ROUNDED to the nearest
        // whole tick rather than truncated, and clamped to >= 1.
        [[nodiscard]] s32 get_edge_delay_ticks(s32 source_node, s32 target_node) const;

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
        //
        // THROWS std::invalid_argument for every matrix whose Ck is pinned to
        // all-zero, because that pinning is what makes their Sk entry the whole stored
        // value: DEFAULT_MATRIX_INDEX once set_edge_weight() has put this instance in
        // exact mode (using_exact_edge_weights), delay_matrix_index once
        // set_edge_delay_ticks() has registered it, and every per-edge variable matrix
        // configure_per_edge_variable_count() registered. Any Ck written onto one restores
        // an order-1 low-rank term on top of values that are routinely 1e-9 or smaller,
        // corrupting every weight (or every delay) at once with no diagnostic. Use
        // set_edge_weight()/set_edge_delay_ticks() to change those values.
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
        // and every registered matrix's Ck EXCEPT DEFAULT_MATRIX_INDEX's, to the
        // current point cloud (the pre-refit reconstruction + Sk at every real
        // edge, read once before any mutation), via `sweep_count` warm-started
        // alternating-least-squares sweeps -- closed-form ridge-regularized
        // normal-equation solves, not an iterative optimizer (D4 math memo
        // §2.1-2.4) -- then clears every matrix's Sk and resets the tick-count
        // knob. This is a periodic memory-compaction/basis-freshening step, not
        // error correction (every loadedge read was already exact beforehand);
        // it is also not a learning/training step (see CLAUDE.md's U/V
        // factorization note).
        //
        // DEFAULT_MATRIX_INDEX's Ck is deliberately NEVER re-fit here (ticket
        // #103): it stays pinned for the whole lifetime of a WeightMatrix --
        // all-ones ordinarily, all-zero once set_edge_weight() has switched this
        // instance into exact mode -- matching every other method's
        // bit-compatibility assumption. The live GPU propagate kernel depends on
        // that pinning too: the engine binds
        // coefficient_vectors[DEFAULT_MATRIX_INDEX] as the generated kernel's
        // `edge_weight_coefficients` argument and the kernel reads it lane by lane
        // to reconstruct Σ U·Ck·V + Sk, so a moved Ck would corrupt the weights on
        // device exactly as it does on the host. U and V are still fit using the
        // default matrix's real edge data (they are shared across the whole matrix
        // family) -- only its own Ck update is skipped. (A caller can still move
        // DEFAULT_MATRIX_INDEX's Ck away from all-ones by calling
        // set_coefficient_vector() directly on it, which is refused outright in
        // exact mode -- see that method's own header comment -- but refit() itself
        // never does.)
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
