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
    // coefficient vector Ck per plane says what each edge's values are:
    //
    //   plane k at edge (i, j) = sum over lanes l of U[i][l] * Ck[l] * V[j][l]
    //
    // i.e. M_k = U diag(Ck) V^T, fitted over the edges only (every other entry is free).
    // Nothing here is sized by node_count * max_neighbor_count, or by node_count^2.
    //
    // The basis holds every value. Fitting it to the network's structure trades a measured
    // accuracy for storage, and `rank` is that dial. What changes during a run (synapse
    // state, plasticity) goes into the sparse delta matrix S, which every read adds; once S
    // is dense enough, a refit folds it into U, V and every Ck and empties it.
    //
    // This is memory compression, not learning.
    class WeightMatrix {
    public:
        // Where the kernel finds each of a synapse's per-edge variables. Every plane is the same
        // kind of thing -- one per-edge value with its own coefficient row, fitted, refit and
        // measured the same way -- and these numbers only say which is which.
        static constexpr s64 WEIGHT_PLANE = 0;
        static constexpr s64 DELAY_PLANE = 1;                 // in ticks
        static constexpr s64 FIRST_STATE_VARIABLE_PLANE = 2;  // the synapse type's LEMS StateVariables, in order

        // U and V are float4-typed, so a logical rank is always rounded up to a multiple
        // of four and every lane in the group participates in the reconstruction. The old
        // code left the padding lanes seeded with N(0,1) and summed them too, which made a
        // declared rank of 1 behave as a rank of 4 -- `rank` here is the honest number.
        static constexpr s64 LANE_GROUP = 4;

        // The worst relative error a fit may leave on any per-edge value, against the larger of
        // the value and its plane's RMS. A delay stays on its tick while this times the delay is
        // under half a tick: 1e-3 keeps every delay up to 500 ticks exact.
        static constexpr f32 DEFAULT_FIT_TOLERANCE = 1.0e-3f;

        static constexpr f32 DEFAULT_FIT_RIDGE = 1.0e-4f;

        // A fit runs sweeps until every plane meets fit_tolerance, or until it has stalled at this
        // rank: its best worst-case error improved by less than FIT_STALL_IMPROVEMENT over the last
        // FIT_STALL_WINDOW_SWEEP_COUNT sweeps. Alternating least squares has long slow stretches
        // that a per-sweep test mistakes for a stall, so the test looks across a window, and at
        // the error the tolerance is about rather than the total squared error.
        static constexpr f64 FIT_STALL_IMPROVEMENT = 0.05;
        static constexpr s32 FIT_STALL_WINDOW_SWEEP_COUNT = 100;

        // A guard, not a budget: a fit that reaches it has failed to stall, and says so.
        static constexpr s32 MAXIMUM_FIT_SWEEP_COUNT = 2000;

        // The refit threshold S is sized for until whoever drives the refits says otherwise.
        static constexpr f32 DEFAULT_REFIT_OCCUPANCY_THRESHOLD_FRACTION = 0.75f;

        K2Tree k2tree;

        // How many per-edge variables a synapse carries, one plane each: its weight, its delay,
        // then its StateVariables.
        s64 matrix_count = FIRST_STATE_VARIABLE_PLANE;

        // The shared basis, row-major [node_count][rank_float4_stride].
        EnginePointer U_matrix;
        EnginePointer V_matrix;

        // One coefficient row per matrix, [matrix_count][rank_float4_stride * LANE_GROUP].
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

        // ── the sparse delta matrix (S) ──────────────────────────────────────────
        // The updates since the last fit, one plane at a time. Every read adds an edge's
        // delta to what the basis gives, so a value reflects its updates straight away. Once
        // a plane is dense enough, refit() fits U, V and every Ck to the updated values and
        // empties S. It never holds anything else: in particular not what a fit missed.
        //
        // CSR over source rows, holding only the edges that changed. Edge ordinals are
        // grouped by source row, so the row slice of an edge is contiguous and a lookup is a
        // binary search inside it -- cheap enough for the propagate walk to do per edge.
        //
        //   row_start[m * (node_count + 1) + n] .. [.. + n + 1]  is node n's slice for matrix m
        //   entry_edge_ordinal[slice]                            sorted ascending within the slice
        //   entry_delta[slice]                                   the update to add
        EnginePointer sparse_delta_row_start;      // s32[matrix_count * (node_count + 1)]
        EnginePointer sparse_delta_edge_ordinal;   // s64[matrix_count * sparse_delta_capacity]
        EnginePointer sparse_delta_value;          // f32[matrix_count * sparse_delta_capacity]

        // Updates arrive out of order and possibly from many device threads at once, so
        // they queue here and are merged into the CSR above on an interval. That batching
        // is deliberate: the merge is what the CSR's sortedness costs, and paying it per
        // update would defeat the point.
        EnginePointer pending_delta_edge_ordinal;  // s64[pending_delta_capacity]
        EnginePointer pending_delta_value;         // f32[pending_delta_capacity]
        EnginePointer pending_delta_matrix_index;  // s32[pending_delta_capacity]
        EnginePointer pending_delta_count;         // s32[1], bumped atomically on device

        // How many planes the simulation updates each tick, as the generated kernel reports it.
        s64 updated_plane_count = 0;

        // Room in the queue above: the plasticity reserve, plus two entries per updated plane per
        // edge, since in one tick an edge with no entry in S can queue its Euler step and its
        // OnEvent change.
        s64 pending_delta_capacity = 0;

        // How many updates each plane of S can hold, and how many it does. Sized to the refit
        // threshold (delta_capacity_for_threshold). A merge that would overflow it grows it
        // instead -- nothing is dropped -- and it goes back to the threshold's size after a refit.
        s64 sparse_delta_capacity = 0;
        Vector<s64> sparse_delta_entry_count = Vector<s64>((usize)FIRST_STATE_VARIABLE_PLANE, 0);

        // Room reserved in S for plasticity's weight updates, on top of the threshold's share.
        s64 plasticity_reserve_entries = 0;

        // A fixed rank for the general fit, or -1 to search for the smallest rank that meets
        // fit_tolerance.
        s64 fit_rank_budget = -1;

        // The worst relative error a fit may leave on any per-edge value (DEFAULT_FIT_TOLERANCE).
        // The rank search and a refit add lanes until every plane meets it.
        f32 fit_tolerance = DEFAULT_FIT_TOLERANCE;

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
        // Upper bound on any node's in-degree. Bounds the caller-supplied buffer
        // get_predecessors() writes into, and nothing else.
        s64 max_predecessor_count = 0;

        s64 rank = 0;
        s64 rank_float4_stride = 0; // ceil(rank / LANE_GROUP)

        bool check_indexing = true;

        // What the last fit cost: per plane, the worst relative error over the edge set against
        // the values it was fitted to.
        Vector<f32> measured_fit_error;

        // The empty network: no edges, no basis, no slab, no backend.
        WeightMatrix() = default;

        WeightMatrix(const WeightMatrix &) = delete;
        WeightMatrix &operator=(const WeightMatrix &) = delete;

        // Hand-written for one reason: the moved-from object must forget its backend, or
        // both release the same slab. Every other member is a plain value.
        WeightMatrix(WeightMatrix &&other) noexcept;
        WeightMatrix &operator=(WeightMatrix &&other) noexcept;

        // network:            adjacency list -- network[i] is the neighbours of node i
        // rank:               the starting rank; declare_projections chooses the one it fits at
        // max_neighbor_count: -1 derives it from the longest row
        // weight_seed:        seeds the basis before any fit; -1 uses hardware entropy
        // fit_rank_budget:    a fixed rank for the general fit, or -1 to search
        // matrix_count:       how many per-edge variables a synapse carries
        // updated_plane_count: how many of them the simulation updates each tick
        WeightMatrix(
            EngineBackend &backend,
            const vector<vector<s32>> &network,
            s64 rank = -1,
            bool check_indexing = true,
            s64 max_neighbor_count = -1,
            s64 weight_seed = -1,
            s64 fit_rank_budget = -1,
            s64 matrix_count = FIRST_STATE_VARIABLE_PLANE,
            s64 updated_plane_count = 0
        );

        ~WeightMatrix();

        // ── declaring the network's values ───────────────────────────────────────
        // One entry per projection run, in canonical edge order: initial_values[plane][run] is
        // every plane's starting value on that run. Per run is the form NeuroML gives and the
        // form the basis represents exactly. There is deliberately no per-edge setter -- a
        // per-edge interface would invite per-edge storage.
        //
        // Fits the basis to these values -- exactly, one lane per distinct combination, when
        // they fit in the lanes, otherwise at the smallest rank that meets fit_tolerance -- and
        // reports what the fit cost in measured_fit_error. S starts empty.
        void declare_projections(
            const Vector<s64> &first_edge_ordinal,
            const Vector<s64> &edge_count,
            const Vector<s32> &synapse_prototype,
            const Vector<Vector<f32>> &initial_values
        );

        // ── reading values back ──────────────────────────────────────────────────
        [[nodiscard]] f32 get(s32 source_node, s32 target_node) const;
        [[nodiscard]] f32 get_for_matrix(s32 source_node, s32 target_node, s64 matrix_index) const;

        // The delay plane's value rounded to whole ticks, at least one: how the kernel uses it.
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
        // Adds delta to one edge's value in one plane, through S. Reads see it at once. When
        // the plane's S is full, S grows; the update waits for the next refit like any other.
        void accumulate_edge_delta(s64 matrix_index, s32 source_node, s32 target_node, f32 delta);

        // Merges what the device queued this tick into S, summing each edge's updates into one
        // entry and dropping any that came back to exactly zero. S grows when a plane would
        // overflow, so nothing is dropped.
        void compact_pending_deltas();

        // Fits U, V and every Ck to the current values -- the basis plus S -- starting from the
        // current basis, until every plane meets fit_tolerance, adding lanes when the current
        // rank stalls short of it. Then empties S. Expensive, which is why it is batched behind
        // a threshold.
        void refit(f32 ridge_regularization = DEFAULT_FIT_RIDGE);

        // True once the fullest plane of S holds updates on occupancy_threshold_fraction of the
        // edge set. Zero or less never refits.
        [[nodiscard]] bool is_refit_due(f32 occupancy_threshold_fraction) const;

        // Fraction of the edge set holding an update, in the fullest plane.
        [[nodiscard]] f32 sparse_delta_occupancy_fraction() const;

        // S's capacity per plane for a refit threshold (every edge when it is zero), and a
        // resize to a capacity, keeping the updates S holds. It never shrinks below them.
        [[nodiscard]] s64 delta_capacity_for_threshold(f32 occupancy_threshold_fraction) const;
        void resize_delta_capacity(s64 new_capacity);

        // Scales the weight plane -- its coefficient row and its updates -- so the weights
        // reach a target RMS. U and V are shared with every plane, so they are left alone.
        ScaleResult scale_neighbor_weights_to_root_mean_square(f32 target_root_mean_square,
                                                              f32 epsilon = 1e-12f);


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

        // The ahead-of-time neighbor_weights_kernel in src/metal/kernels.metal, built into a
        // pipeline on first use and held for the object's life. Mutable because the reads
        // that need it are const, and building a pipeline is caching.
        mutable EngineFunction neighbor_weights_function;

        // Loads `name` from default.metallib into `function` if it is not already built.
        // Returns false when the AOT library has no such kernel, which is the signal to
        // fall back to the host path rather than to fail.
        [[nodiscard]] bool ensure_function(EngineFunction &function, const String &name) const;

        // Kept so a resize can reproduce the same starting basis instead of drifting on
        // re-allocation, and so a run stays reproducible across one.
        unsigned basis_seed = 0;

        // Fills U/V with independent N(0,1) and every coefficient row with 1.0.
        void seed_basis(unsigned seed);

        void build_edge_row_offset();

        // Runs one partition -> allocate round for every buffer this matrix owns, at the
        // current rank and correction capacity. Called at construction and on every resize.
        void allocate_storage();

        // Fits at ranks 4, 8, 16, ... until one meets fit_tolerance, then bisects down to the
        // smallest that does. Leaves the basis fitted at it, or at the lane limit when none does.
        s64 search_for_rank_meeting_tolerance(const Vector<Vector<f32>> &targets_per_matrix);

        // Adds lanes, keeping U, V and every Ck as they are. The new lanes get random U and V and
        // zero coefficients, so every value reads the same until the next fit uses them.
        void grow_basis(s64 new_rank);

        // The capacity S grows to when a plane needs required_entries.
        [[nodiscard]] s64 grown_delta_capacity(s64 required_entries) const;

        // Σ_k U[i][k] * Ck[k] * V[j][k] -- the basis's own answer, BEFORE the sparse
        // correction. Only the read paths that then add Sk should call this.
        [[nodiscard]] f32 reconstruct_entry(s32 source_node, s32 target_node,
                                            const f32 *coefficient_values) const;

        // The update S holds for one edge of one matrix, or zero when it holds none.
        // Binary search inside the source row's slice.
        [[nodiscard]] f32 sparse_delta_for(s64 matrix_index, s32 source_node, s64 edge_ordinal) const;

        // Rebuilds one matrix's CSR from a full list of (edge_ordinal, delta) pairs. The list
        // must fit the capacity: callers refit rather than drop an update.
        void rebuild_sparse_delta(s64 matrix_index, Vector<Pair<s64, f32>> &deltas);

        // Every plane's updates, as (edge_ordinal, delta) lists.
        [[nodiscard]] Vector<Vector<Pair<s64, f32>>> current_deltas() const;

        void clear_sparse_deltas();

        // Fits the basis to its own values plus these updates, then empties S.
        void fold_into_basis(const Vector<Vector<Pair<s64, f32>>> &deltas_per_matrix,
                             f32 ridge_regularization);

        [[nodiscard]] f32 *coefficient_row(s64 matrix_index) const;

        // Builds U/V and every coefficient row directly from the projection runs, one lane per
        // distinct combination of initial values. Exact for population-to-population
        // projections, and free -- but only available when the combinations fit in the lane
        // limit. Returns true when it was used.
        bool fit_basis_from_projections(const Vector<Vector<f32>> &initial_values);

        // Alternating least squares over the edges only, for a basis shared by every plane with
        // one coefficient row each: M_k = U diag(Ck) V^T. Runs sweeps from the current basis
        // until every plane meets fit_tolerance or the fit stalls at this rank, and returns
        // whether it met the tolerance. targets_per_matrix[m][edge_ordinal] is what plane m
        // should read.
        bool fit_basis_to_targets(const Vector<Vector<f32>> &targets_per_matrix, f32 ridge_regularization);

        // Per-edge values implied by the projection runs, one row per plane.
        [[nodiscard]] Vector<Vector<f32>> targets_from_projections(const Vector<Vector<f32>> &initial_values) const;

        // Each plane's value at every edge: the basis plus these updates. What a refit fits to.
        [[nodiscard]] Vector<Vector<f32>> targets_from_basis_and_deltas(
                const Vector<Vector<Pair<s64, f32>>> &deltas_per_matrix) const;

        // Fills measured_fit_error from the basis alone, against the values it was fitted to.
        void measure_fit_error(const Vector<Vector<f32>> &targets_per_matrix);

    public:
        // The worst of measured_fit_error over every plane.
        [[nodiscard]] f32 worst_fit_error() const;

    private:

        void resize_basis(s64 new_rank);

        void validate_matrix_index(s64 matrix_index) const;
    };
}
