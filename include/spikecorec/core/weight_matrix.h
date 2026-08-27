//
// Created by Alek Simpson on 5/30/26.
//
#pragma once

#include <optional>
#include <vector>

#include "spikecorec/core/backend.h"
#include "spikecorec/core/k2tree.h"
#include "spikecorec/core/types.h"

using namespace std;

namespace spikecorec {

    // Must match MAX_RANK_FLOAT4_STRIDE in src/metal/kernels.metal and in the generated
    // master kernel: all three index fixed-size per-thread arrays with it.
    #define MAX_RANK_FLOAT4_STRIDE 64

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

    // The network's edges and everything stored about them.
    //
    // THE INVARIANT: no per-edge value is ever held in memory as a per-edge value. The
    // k^2-tree says which (i, j) pairs are edges; a shared low-rank basis U/V with one
    // coefficient vector Ck per quantity says what each edge's values are. Nothing here
    // is sized by node_count * max_neighbor_count, and nothing is sized by node_count^2.
    //
    // What that buys, concretely: a million-neuron network with a hundred outgoing edges
    // each stores its weights and delays in 2 * node_count * rank floats -- a few tens of
    // megabytes -- rather than the gigabytes a padded per-edge plane would take.
    //
    // This is memory compression, not learning. U and V encode connection strengths and
    // nothing else; `rank` controls how faithfully they can, and derive_rank() measures
    // that faithfulness against the model's own declarations rather than assuming it.
    class WeightMatrix {
    public:
        // The two quantities the basis carries. Both share U/V and differ only by their
        // coefficient row -- see fit_basis_from_projections for why that is exact for a
        // network wired out of population-to-population projections.
        static constexpr s64 DEFAULT_MATRIX_INDEX = 0;  // synaptic weight
        static constexpr s64 DELAY_MATRIX_INDEX = 1;    // delay in whole ticks, rounded
        static constexpr s64 MATRIX_COUNT = 2;

        // U and V are float4-typed, so a logical rank is always rounded up to a multiple
        // of four and every lane in the group participates in the reconstruction. The old
        // code left the padding lanes seeded with N(0,1) and summed them too, which made a
        // declared rank of 1 behave as a rank of 4 -- `rank` here is the honest number.
        static constexpr s64 LANE_GROUP = 4;

        // Above this relative error the fit is reported but still accepted; above the
        // maximum it is refused. Real synaptic weights are specified to two or three
        // significant figures, so a percent is generous and a hundredth of a percent is
        // already far tighter than the biology it stands for -- but a fit that has drifted
        // past the warning line is worth knowing about before it becomes a wrong answer.
        static constexpr f32 WEIGHT_FIT_WARNING_TOLERANCE = 1.0e-4f;

        // What the general fit spends when the exact construction is unavailable. Not the
        // rank an arbitrary field would need for an exact fit -- spending that on every
        // model gives up the compression this exists for. The residual goes to Sk, and if
        // a model wants more fidelity the rank is a constructor argument.
        static constexpr s64 DEFAULT_FIT_RANK_BUDGET = 32;
        static constexpr s32 DEFAULT_FIT_SWEEP_COUNT = 6;
        static constexpr f32 DEFAULT_FIT_RIDGE = 1.0e-4f;

        K2Tree k2tree;

        // The shared basis, row-major [node_count][rank_float4_stride].
        EnginePointer U_matrix;
        EnginePointer V_matrix;

        // One coefficient row per matrix, [MATRIX_COUNT][rank_float4_stride * LANE_GROUP].
        // Sized once at construction because the family is fixed: weight and delay.
        EnginePointer coefficients;

        // Prefix sum over real out-degree: edge_row_offset[n] is the ordinal of node n's
        // first outgoing edge and edge_row_offset[node_count] == total_edge_count. This is
        // what replaces max_neighbor_count padding -- an edge's ordinal is
        // edge_row_offset[source] + slot, with slot its position in k^2-tree traversal
        // order, and anything indexed by edge is sized total_edge_count.
        EnginePointer edge_row_offset;

        // Output for neighbor_weights(), sized by edges rather than padded. Preallocated
        // because the backend hands out one chunk per partition -> allocate round.
        EnginePointer neighbor_weight_scratch;

        // ── the sparse delta matrix (Sk) ─────────────────────────────────────────
        // The basis is a LOSSY projection: rank buys fidelity, and what the rank does not
        // capture lands here. Every read adds this correction back, so a read is accurate
        // even while the basis is only approximate -- and when it accumulates enough
        // entries, refit() re-optimises U/V to represent the corrected values better and
        // empties it. That loop is what lets rank be a storage dial rather than a promise.
        //
        // Genuinely sparse: CSR over source rows, holding only the edges that need a
        // correction. Edge ordinals are already grouped by source row, so the row slice of
        // an edge is contiguous and a lookup is a binary search inside it -- cheap enough
        // for the propagate walk to do per edge.
        //
        //   row_start[m * (node_count + 1) + n] .. [.. + n + 1]  is node n's slice for matrix m
        //   entry_edge_ordinal[slice]                            sorted ascending within the slice
        //   entry_delta[slice]                                   the correction to add
        EnginePointer sparse_delta_row_start;      // s32[MATRIX_COUNT * (node_count + 1)]
        EnginePointer sparse_delta_edge_ordinal;   // s64[MATRIX_COUNT * sparse_delta_capacity]
        EnginePointer sparse_delta_value;          // f32[MATRIX_COUNT * sparse_delta_capacity]

        // Updates arrive out of order and possibly from many device threads at once, so
        // they queue here and are merged into the CSR above on an interval. That batching
        // is deliberate: the merge is what the CSR's sortedness costs, and paying it per
        // update would defeat the point.
        EnginePointer pending_delta_edge_ordinal;  // s64[sparse_delta_capacity]
        EnginePointer pending_delta_value;         // f32[sparse_delta_capacity]
        EnginePointer pending_delta_count;         // s32[1], bumped atomically on device

        // How many corrections each matrix can hold, and how many it does. Capacity is not
        // guessed: declare_projections fits the basis, measures how many edges the fit
        // actually misses, and sizes this to that. A model whose structure the basis
        // captures allocates nothing here -- which is the common case, and the one a fixed
        // fraction used to charge for anyway.
        s64 sparse_delta_capacity = 0;
        Vector<s64> sparse_delta_entry_count = Vector<s64>((usize)MATRIX_COUNT, 0);

        // The most of the edge set corrections may occupy. This is the accuracy-for-storage
        // dial, and the only reason it is not simply "as many as needed": a field with no
        // structure to exploit needs one per edge, and corrections cost more per edge than
        // the values would. At 1.0 every model is reproduced exactly and an incompressible
        // one pays for it visibly; lower it to cap what that model may spend, and the
        // largest residuals are the ones kept.
        f32 correction_ceiling_fraction = 1.0f;

        // Room reserved on top of the fit's needs, for updates to queue into. Zero unless
        // something is going to write updates -- an exactly-fitted model with no plasticity
        // has nothing to queue and allocates nothing.
        s64 plasticity_reserve_entries = 0;

        // Refit when the corrections outgrow this fraction of the edge set -- the basis has
        // drifted far enough from the values that re-optimising it is worth the cost. The
        // primary trigger; a caller that wants a schedule instead can drive refit() itself.
        f32 refit_occupancy_threshold_fraction = 0.25f;

        // ── projection runs (structure of arrays, parallel) ───────────────────────
        // Which synapse prototype each edge uses, as runs over the canonical edge
        // ordering. A NeuroML projection names one synapse for every connection it
        // declares, so this is O(projections) -- one entry for the common single-
        // projection network. Prototype index deliberately does NOT go through the basis:
        // it selects a switch case in the kernel, and control flow should not ride on a
        // reconstruction. Lookup is a binary search on projection_first_edge_ordinal.
        Vector<s64> projection_first_edge_ordinal;
        Vector<s64> projection_edge_count;
        Vector<s32> projection_synapse_prototype;

        // The backend this matrix's slab came from and the whole-chunk handle it returned.
        // The destructor releases that one chunk, freeing every range above at once.
        EngineBackend *owning_backend = nullptr;
        EnginePointer owning_slab;

        s64 node_count = 0;
        s64 total_edge_count = 0;

        // Upper bound on any node's out-degree. Bounds the caller-supplied buffer
        // get_neighbors() writes into, and nothing else -- in particular it sizes no
        // allocation anywhere.
        s64 max_neighbor_count = 0;

        s64 rank = 0;
        s64 rank_float4_stride = 0; // ceil(rank / LANE_GROUP)

        f32 constant_weight = 0.0f;
        bool using_constant_weight = false;

        // Delay every edge uses when the model declares one value for all of them, which
        // the topology constructor guarantees by construction. Avoids a reconstruction per
        // edge for the case that needs none.
        s32 constant_delay_ticks = 1;
        bool using_constant_delay_ticks = true;

        bool check_indexing = true;

        // Worst relative error between a declared weight and its reconstruction, measured
        // over the edge set at the end of the fit. The honest report of how much the
        // compression cost, and what derive_rank() climbed until it satisfied.
        f32 measured_weight_fit_error = 0.0f;

        // The empty network: no edges, no basis, no slab, no backend.
        WeightMatrix() = default;

        WeightMatrix(const WeightMatrix &) = delete;
        WeightMatrix &operator=(const WeightMatrix &) = delete;

        // Hand-written for one reason: the moved-from object must forget its backend, or
        // both release the same slab. Every other member is a plain value.
        WeightMatrix(WeightMatrix &&other) noexcept;
        WeightMatrix &operator=(WeightMatrix &&other) noexcept;

        // network:            adjacency list -- network[i] is the neighbours of node i
        // rank:               -1 derives it from the declarations (see derive_rank)
        // max_neighbor_count: -1 derives it from the longest row
        // weight_seed:        seeds the basis before any fit; -1 uses hardware entropy
        // correction_ceiling_fraction: the most of the edge set corrections may occupy;
        //                     1.0 reproduces every model exactly.
        WeightMatrix(
            EngineBackend &backend,
            const vector<vector<s32>> &network,
            s64 rank = -1,
            bool check_indexing = true,
            s64 max_neighbor_count = -1,
            s64 weight_seed = -1,
            f32 correction_ceiling_fraction = 1.0f
        );

        ~WeightMatrix();

        // ── declaring the network's values ───────────────────────────────────────
        // One entry per projection, in canonical edge order. This is how a model states
        // its weights and delays: per projection, which is the form NeuroML gives and the
        // form the basis represents exactly. There is deliberately no per-edge setter --
        // a per-edge interface would invite per-edge storage.
        //
        // Derives the rank when the constructor was given -1, builds the basis, then
        // measures the result against these declarations and throws if it cannot
        // reproduce them within WEIGHT_FIT_MAXIMUM_TOLERANCE.
        void declare_projections(
            const Vector<s64> &first_edge_ordinal,
            const Vector<s64> &edge_count,
            const Vector<s32> &synapse_prototype,
            const Vector<f32> &weight,
            const Vector<s32> &delay_ticks
        );

        // ── reading values back ──────────────────────────────────────────────────
        [[nodiscard]] f32 get(s32 source_node, s32 target_node) const;
        [[nodiscard]] f32 get_for_matrix(s32 source_node, s32 target_node, s64 matrix_index) const;

        // Whole ticks, rounded from the reconstruction. Rounding is what makes a small fit
        // error harmless here where the same error in a weight would not be -- and why the
        // delay check at construction is an exact-integer one, not a tolerance.
        [[nodiscard]] s32 get_edge_delay_ticks(s32 source_node, s32 target_node) const;

        [[nodiscard]] s32 get_edge_synapse_prototype(s32 source_node, s32 target_node) const;

        // The ordinal of edge (source -> target), or nullopt when that pair is not an edge.
        [[nodiscard]] optional<s64> edge_ordinal(s32 source_node, s32 target_node) const;

        [[nodiscard]] s64 get_neighbors(s64 node_index, s32 *output_buffer) const;
        [[nodiscard]] s64 get_predecessors(s64 node_index, s32 *output_buffer) const;

        // Every edge's weight, indexed by edge ordinal -- total_edge_count values, with no
        // padding and no sentinels.
        void neighbor_weights(f32 *output_weights) const;
        void neighbor_weights_for_matrix(f32 *output_weights, s64 matrix_index) const;

        [[nodiscard]] WeightStats neighbor_weight_stats() const;

        // ── updates ──────────────────────────────────────────────────────────────
        // Queues a correction for one edge. It does not touch U/V: the delta lands in the
        // pending buffer, gets merged into Sk, and is folded into the basis by the next
        // refit. Reads see it from the moment it is merged.
        void accumulate_edge_delta(s64 matrix_index, s32 source_node, s32 target_node, f32 delta);

        // Merges whatever the device staged this interval into the CSR. Cheap, and the
        // point at which recent updates become visible to reads.
        void compact_pending_deltas();

        // Re-optimises U/V and every Ck against the values Sk currently corrects to, then
        // empties Sk. Expensive, which is why it is triggered rather than continuous --
        // the corrections are what it fits to, so it wants a batch of them.
        void refit(s32 sweep_count = 4, f32 ridge_regularization = 1e-3f);

        // True once the corrections have outgrown refit_occupancy_threshold_fraction of
        // the edge set.
        [[nodiscard]] bool is_refit_due() const;

        // Fraction of the edge set currently carrying a correction, across every matrix.
        [[nodiscard]] f32 sparse_delta_occupancy_fraction() const;

        // Applies one edge's delta straight into U/V as a rank-1 nudge, bypassing Sk.
        // A direct-manipulation entry point for tooling, not the simulation path.
        void update(s32 source_node, s32 target_node, f32 delta,
                    f32 learning_rate, f32 l2_regularization, s32 iterations);

        // Rescales U/V so the reconstructed weights reach a target RMS. Worth running
        // after a fold: many small rank-1 nudges can drift the basis in scale.
        ScaleResult scale_neighbor_weights_to_root_mean_square(f32 target_root_mean_square,
                                                              f32 epsilon = 1e-12f);

        void set_constant_weight(f32 value);
        void set_constant_delay_ticks(s32 ticks);

        [[nodiscard]] bool check_index_inbounds(s32 source, s32 target) const;
        [[nodiscard]] bool check_index_inbounds(s32 node_index) const;

        // One matrix's coefficient row, as a range the kernel can bind. Public because the
        // engine binds the weight and delay rows as separate kernel arguments.
        [[nodiscard]] EnginePointer coefficient_range(s64 matrix_index) const;

        void save(const char *filepath) const;
        void load_from_disk(const char *filepath);

    private:
        // The same prefix sum as edge_row_offset, kept host-side so the walks below can
        // index it without going through a device handle on every edge. Written once at
        // construction; the device copy is made from it.
        vector<s64> edge_row_offset_host;

        static const vector<vector<s32>> &validate_network(const vector<vector<s32>> &network);

        // The ahead-of-time kernels in src/metal/kernels.metal, each built into a pipeline
        // on first use and held for the object's life. Mutable because the reads that need
        // them (neighbor_weights, the stats it feeds) are const, and building a pipeline is
        // caching rather than a change in what this matrix represents.
        mutable EngineFunction neighbor_weights_function;
        mutable EngineFunction scale_uv_function;
        mutable EngineFunction weight_update_function;

        // Loads `name` from default.metallib into `function` if it is not already built.
        // Returns false when the AOT library has no such kernel, which is the signal to
        // fall back to the host path rather than to fail.
        [[nodiscard]] bool ensure_function(EngineFunction &function, const String &name) const;

        // Kept so a resize can reproduce the same starting basis instead of drifting on
        // re-allocation, and so a run stays reproducible across one.
        unsigned basis_seed = 0;

        // Fills U/V with independent N(0,1) and both coefficient rows with 1.0.
        void seed_basis(unsigned seed);

        void build_edge_row_offset();

        // Runs one partition -> allocate round for every buffer this matrix owns, at the
        // current rank and correction capacity. Called at construction and on every resize.
        void allocate_storage();

        // How many edges the fitted basis fails to reproduce -- the model's structural
        // complexity, measured rather than assumed. A network of uniform projections
        // answers zero; one with a value per edge answers with the edge count.
        [[nodiscard]] s64 count_edges_needing_correction(
                const Vector<Vector<f32>> &targets_per_matrix) const;

        // Re-partitions at a new correction capacity, carrying the fitted basis across.
        // Distinct from resize_basis, which re-seeds -- doing that here would discard the
        // fit whose residuals decided the capacity in the first place.
        void resize_correction_capacity(s64 new_capacity);

        // Σ_k U[i][k] * Ck[k] * V[j][k] -- the basis's own answer, BEFORE the sparse
        // correction. Only the read paths that then add Sk should call this.
        [[nodiscard]] f32 reconstruct_entry(s32 source_node, s32 target_node,
                                            const f32 *coefficient_values) const;

        // The correction Sk holds for one edge of one matrix, or zero when it holds none.
        // Binary search inside the source row's slice.
        [[nodiscard]] f32 sparse_delta_for(s64 matrix_index, s32 source_node, s64 edge_ordinal) const;

        // Rebuilds one matrix's CSR from a full list of (edge_ordinal, delta) pairs, keeping
        // the largest by magnitude when there are more than capacity. Keeping the largest is
        // what makes truncation a bounded accuracy loss rather than an arbitrary one.
        void rebuild_sparse_delta(s64 matrix_index, Vector<Pair<s64, f32>> &corrections);

        [[nodiscard]] f32 *coefficient_row(s64 matrix_index) const;

        // Builds U/V and both coefficient rows directly from the projection structure.
        // Exact for population-to-population projections, and free -- but only available
        // when the runs fit in the lane budget. Returns true when it was used.
        bool fit_basis_from_projections(const Vector<f32> &weight, const Vector<s32> &delay_ticks);

        // Alternating least squares over the EDGE SUPPORT only, for a basis shared by every
        // matrix with one coefficient row each. This is the general path: a projection onto
        // a linear subspace that approximates the targets, with whatever it misses left for
        // Sk to correct. targets_per_matrix[m][edge_ordinal] is what matrix m should read.
        void fit_basis_to_targets(const Vector<Vector<f32>> &targets_per_matrix,
                                  s32 sweep_count, f32 ridge_regularization);

        // Per-edge target values implied by the projection runs, one row per matrix.
        [[nodiscard]] Vector<Vector<f32>> targets_from_projections(
                const Vector<f32> &weight, const Vector<s32> &delay_ticks) const;

        // What every matrix currently reads, corrections included. This is what refit fits
        // to: the basis is re-optimised to represent the values Sk is currently correcting
        // it toward, which is how the corrections get absorbed.
        [[nodiscard]] Vector<Vector<f32>> targets_from_current_values() const;

        // Compares the basis against the per-edge targets and hands the difference to Sk.
        // Returns how many corrections were kept. Indexed by edge ordinal so the initial
        // fit and refit can both use it.
        s64 store_residual_corrections(s64 matrix_index,
                                       const Vector<f32> &targets_by_edge_ordinal);

        // Worst relative error over every edge, against the declared per-projection values.
        [[nodiscard]] f32 measure_worst_relative_weight_error(const Vector<f32> &weight) const;

        // Number of edges whose reconstructed delay does not round to the declared one.
        [[nodiscard]] s64 count_delay_mismatches(const Vector<s32> &delay_ticks) const;

        void resize_basis(s64 new_rank);

        void validate_matrix_index(s64 matrix_index) const;
    };
}
