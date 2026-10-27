//
// Created by Alek Simpson on 5/30/26.
//
#pragma once

#include <optional>
#include <vector>

#include "backend/metal_backend.h"
#include "k2tree.h"
#include "types.h"

using namespace std;

namespace spikecorec {

    // The lane limit, in float4s (8,192 lanes). No kernel sizes an array by it; the limit that
    // binds is lane_cap's, which keeps the basis smaller than one float per edge per plane. Each
    // lane costs U and V node_count * 4 bytes each, and fit time.
    #define MAX_RANK_FLOAT4_STRIDE 2048

    struct Weight_Stats {
        f32 mean;
        f32 standard_deviation;
        f32 root_mean_square;
        f32 min_value;
        f32 max_value;
    };

    struct Scale_Result {
        f32 target_root_mean_square;
        f32 scale_factor;
        Weight_Stats before;
        Weight_Stats after;
    };

    bool can_safely_cast_s64_to_s32(s64);

    // The network's edges and every value stored per edge. The k^2-tree says which (i, j) pairs
    // are edges; a shared low-rank basis with one coefficient row Ck per plane gives each value:
    //
    //   plane k at edge (i, j) = sum over lanes l of U[i][l] * Ck[l] * V[j][l]
    //
    // The basis is fitted over the edges only (other entries are unconstrained). Each plane owns
    // its lanes (no other plane reads them), and construction searches for the fewest lanes that
    // hold every value within fit_tolerance. Changes during a run go into the sparse update
    // buffer S, which every read adds; a refit folds S into U, V and C and empties it. Nothing
    // is sized by node_count^2 or node_count * max_neighbor_count.
    //
    // The U/V factorization is memory compression, not learning.
    class Weight_Matrix {
    public:
        // Plane indices. Every plane is fitted, refit and measured the same way.
        static constexpr s64 WEIGHT_PLANE = 0;
        static constexpr s64 DELAY_PLANE = 1;                 // in ticks
        static constexpr s64 FIRST_STATE_VARIABLE_PLANE = 2;  // the synapse's StateVariables, in order

        // U and V hold float4s, so rank rounds up to a multiple of four; a read sums every lane.
        static constexpr s64 LANE_GROUP = 4;

        // Worst relative error a fit may leave on any value, against max(|value|, the plane's RMS).
        // 1e-3 keeps every delay up to 500 ticks on its tick (error under half a tick).
        static constexpr f32 DEFAULT_FIT_TOLERANCE = 1.0e-3f;

        static constexpr f32 DEFAULT_FIT_RIDGE = 1.0e-7f;

        // The two candidate fits. Both fit one plane's own lanes in f32, on the GPU when its
        // kernels load and on the host otherwise, and both minimize the same objective: over the
        // edges, each value's error divided by max(|value|, the plane's RMS), squared, plus
        // DEFAULT_FIT_RIDGE times the squared rows that move.
        enum class Fit_Method {
            LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO,
            MASKED_ALTERNATING_LEAST_SQUARES,
        };

        // A fit at one lane count stops when every edge it moves meets fit_tolerance, when its best
        // worst-case error improved by less than FIT_STALL_IMPROVEMENT (a fraction) over the
        // method's stall window, or at the method's cap.
        static constexpr f64 FIT_STALL_IMPROVEMENT = 0.05;
        static constexpr s32 MASKED_ALTERNATING_LEAST_SQUARES_STALL_WINDOW_SWEEP_COUNT = 10;
        static constexpr s32 MASKED_ALTERNATING_LEAST_SQUARES_MAXIMUM_SWEEP_COUNT = 100;
        static constexpr s32 LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO_STALL_WINDOW_ITERATION_COUNT = 100;
        static constexpr s32 LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO_MAXIMUM_ITERATION_COUNT = 2000;

        // Step and gradient-change pairs the quasi-Newton fit keeps.
        static constexpr s32 LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO_HISTORY_LENGTH = 5;

        // Its line search: the strong Wolfe conditions' sufficient-decrease and curvature constants,
        // and the most objective evaluations one search may use.
        static constexpr f64 WOLFE_SUFFICIENT_DECREASE = 1.0e-4;
        static constexpr f64 WOLFE_CURVATURE = 0.9;
        static constexpr s32 LINE_SEARCH_MAXIMUM_EVALUATION_COUNT = 20;

        // Refit threshold the constructor sizes S and the queue for; the engine then sizes them for
        // its own threshold.
        static constexpr f32 DEFAULT_REFIT_OCCUPANCY_THRESHOLD_FRACTION = 0.20f;

        K2_Tree k2tree;

        // Plane count: 2 + the largest state-variable count of any synapse type.
        s64 matrix_count = FIRST_STATE_VARIABLE_PLANE;

        // The shared basis, row-major [node_count][rank_float4_stride] of float4.
        Engine_Pointer U_matrix;
        Engine_Pointer V_matrix;

        // One coefficient row per plane, [matrix_count][rank_float4_stride * LANE_GROUP].
        Engine_Pointer coefficients;

        // Prefix sum of out-degree: edge_row_offset[n] is node n's first edge ordinal and
        // edge_row_offset[node_count] == total_edge_count. Within a row, targets are ascending.
        Engine_Pointer edge_row_offset;

        // ── the sparse delta matrix (S) ──────────────────────────────────────────
        // Updates since the last fit; every read adds an edge's update to the basis value. A refit
        // folds S into U, V and C and empties it. S never holds fit error. One pool holds every
        // plane's updates, plane after plane, so a plane without updates takes no room. Within a
        // plane, entries are CSR over source rows: plane m, node n's entries are pool positions
        // row_start[m * (node_count + 1) + n] up to the next, sorted by edge ordinal for a binary search.
        Engine_Pointer sparse_delta_row_start;      // s32[matrix_count * (node_count + 1)]
        Engine_Pointer sparse_delta_edge_ordinal;   // u32[sparse_delta_capacity]
        Engine_Pointer sparse_delta_value;          // f32[sparse_delta_capacity]

        // The queue: device threads add an update here, unordered, when its edge has no S entry,
        // since they cannot insert into the sorted CSR; compact_pending_deltas merges the queue into
        // S between ticks, so the sort is paid once per batch rather than per update.
        Engine_Pointer pending_delta_edge_ordinal;  // u32[pending_delta_capacity]
        Engine_Pointer pending_delta_value;         // f32[pending_delta_capacity]
        Engine_Pointer pending_delta_matrix_index;  // u8[pending_delta_capacity]
        Engine_Pointer pending_delta_control;       // s32[DEFERRED_LIST_SLOT + 2 * node_count], see below

        // The queue's control block, bumped atomically on device. Before walking its edges a neuron
        // reserves room for every update it can queue (one per updated plane per edge with no S entry);
        // when the queue cannot take them, it defers its walk to another pass of the same tick, which
        // runs after compact_pending_deltas has merged the queue into S. Two lists alternate: a pass
        // walks the neurons in one and defers into the other.
        static constexpr s64 PENDING_WRITE_COUNT_SLOT = 0;       // updates queued
        static constexpr s64 PENDING_RESERVED_SLOT = 1;          // room reserved this pass
        static constexpr s64 WALK_PASS_SLOT = 2;                 // 0 runs every stage; later passes walk only
        static constexpr s64 WALKED_NEURON_COUNT_SLOT = 3;       // neurons this pass walks
        static constexpr s64 DEFERRED_NEURON_COUNT_SLOT = 4;     // neurons this pass deferred
        static constexpr s64 DEFERRED_LIST_SLOT = 5;             // then two lists of node_count

        // Planes the kernel updates each tick, as codegen reports.
        s64 updated_plane_count = 0;

        // Queue size, set by resize_pending_delta_capacity before the kernel is built (which bakes
        // it in); see pending_capacity_for_threshold. A walk the queue cannot take waits for another
        // pass, so a small queue costs passes, never updates.
        s64 pending_delta_capacity = 0;

        // Entries S's pool holds over every plane, and each plane's entries. Sized for the refit
        // threshold; a merge that would overflow grows it instead of dropping updates, and the
        // engine shrinks it on refit.
        s64 sparse_delta_capacity = 0;
        Vector<s64> sparse_delta_entry_count = Vector<s64>(static_cast<usize>(FIRST_STATE_VARIABLE_PLANE), 0);

        // Room reserved in S and the device queue for plasticity's weight updates.
        s64 plasticity_reserve_entries = 0;

        // Worst relative error a fit may leave; fits add lanes to a plane until they meet it.
        f32 fit_tolerance = DEFAULT_FIT_TOLERANCE;

        // The candidate refits use: masked alternating least squares by default, since it keeps no
        // quasi-Newton history, so a refit during a run needs far less scratch.
        Fit_Method fit_method = Fit_Method::MASKED_ALTERNATING_LEAST_SQUARES;

        // The candidate the lane searches use (declare_runs and calibrate_lane_budget), which set
        // each plane's lanes and so the runtime footprint: limited-memory BFGS by default, which
        // holds a plane's values on fewer lanes. They run while the matrix is built, not during a run.
        Fit_Method lane_search_fit_method = Fit_Method::LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO;

        // The expected fraction of edges whose values change between refits, from 0 to 1: the
        // estimate of the network's activity that sizes each plane's lane budget. declare_runs gives
        // a plane the fewest lanes that hold its starting values within fit_tolerance, times
        // (1 + this fraction); a plane that starts at zero gets the same from its calibration values
        // (calibrate_lane_budget) or, without them, from its first values at its first refit. The
        // budgets never change after that.
        // By default the refit threshold: a refit is due once that fraction of the edges hold updates,
        // so about that many change between refits.
        static constexpr f32 DEFAULT_EXPECTED_ACTIVE_EDGE_FRACTION = DEFAULT_REFIT_OCCUPANCY_THRESHOLD_FRACTION;
        f32 expected_active_edge_fraction = DEFAULT_EXPECTED_ACTIVE_EDGE_FRACTION;

        // ── runs (structure of arrays, parallel) ──────────────────────────────────
        // Each edge's synapse prototype, as runs over edge ordinals; lookup is a binary search on
        // run_first_edge_ordinal. The prototype bypasses the basis because it picks a kernel switch
        // case, and control flow must not depend on a reconstructed value. declare_runs keeps one run
        // per stretch of edges with the same prototype, whatever their values.
        Vector<s64> run_first_edge_ordinal;
        Vector<s64> run_edge_count;
        Vector<s32> run_synapse_prototype;

        // The slabs holding the buffers above, and their backend; the destructor releases them. The
        // basis (U, V, the coefficients and edge_row_offset), S and the queue each have their own, so
        // resizing one never copies the others.
        Engine_Backend *owning_backend = nullptr;
        Engine_Pointer owning_slab;
        Engine_Pointer delta_slab;
        Engine_Pointer pending_slab;

        s64 node_count = 0;
        s64 total_edge_count = 0;

        // Upper bound on any node's out-degree. Sizes get_neighbors() buffers. No device buffer uses it.
        s64 max_neighbor_count = 0;
        // Upper bound on any node's in-degree. Sizes get_predecessors() buffers.
        s64 max_predecessor_count = 0;

        s64 rank = 0;
        s64 rank_float4_stride = 0; // ceil(rank / LANE_GROUP)

        bool check_indexing = true;

        // Per plane, the worst relative error of the last fit against the values it fitted; infinite
        // when a value is NaN or infinite, which no basis holds.
        Vector<f32> measured_fit_error;

        // The empty network: no edges, no basis, no slab, no backend.
        Weight_Matrix() = default;

        Weight_Matrix(const Weight_Matrix &) = delete;
        Weight_Matrix &operator=(const Weight_Matrix &) = delete;

        // A move clears the source's backend so the slab is released once.
        Weight_Matrix(Weight_Matrix &&other) noexcept;
        Weight_Matrix &operator=(Weight_Matrix &&other) noexcept;

        // network:             network[i] lists node i's targets, each in [0, node count)
        // rank:                the starting rank; declare_runs chooses the one it fits at
        // max_neighbor_count:  -1 derives it from the longest row; a smaller bound throws
        // weight_seed:         seeds the basis before any fit; -1 uses hardware entropy
        // matrix_count:        the plane count
        // updated_plane_count: planes the kernel updates each tick
        Weight_Matrix(
            Engine_Backend &backend,
            const Vector<Vector<s32>> &network,
            s64 rank = -1,
            bool check_indexing = true,
            s64 max_neighbor_count = -1,
            s64 weight_seed = -1,
            s64 matrix_count = FIRST_STATE_VARIABLE_PLANE,
            s64 updated_plane_count = 0
        );

        ~Weight_Matrix();

        // ── declaring the network's values ───────────────────────────────────────
        // One entry per run, in edge-ordinal order; initial_values[plane][run] is that plane's
        // starting value on the run. No per-edge setter exists, so no per-edge storage is needed.
        // Fits the basis with fit_method (each plane within its lane budget), records
        // measured_fit_error, empties S, and keeps the runs only as prototype runs.
        void declare_runs(
            const Vector<s64> &first_edge_ordinal,
            const Vector<s64> &edge_count,
            const Vector<s32> &synapse_prototype,
            const Vector<Vector<f32>> &initial_values
        );

        // ── reading values back ──────────────────────────────────────────────────
        // One edge's value, the basis plus S; 0 for a pair that is not an edge.
        [[nodiscard]] f32 get(s32 source_node, s32 target_node) const;
        [[nodiscard]] f32 get_for_matrix(s32 source_node, s32 target_node, s64 matrix_index) const;

        // The delay plane's value rounded to whole ticks, at least one: how the kernel uses it.
        // 0 for a pair that is not an edge.
        [[nodiscard]] s32 get_edge_delay_ticks(s32 source_node, s32 target_node) const;

        [[nodiscard]] s32 get_edge_synapse_prototype(s32 source_node, s32 target_node) const;

        // The ordinal of edge (source -> target), or nullopt when that pair is not an edge.
        [[nodiscard]] Optional<s64> edge_ordinal(s32 source_node, s32 target_node) const;

        [[nodiscard]] s64 get_neighbors(s64 node_index, u32 *output_buffer) const;
        [[nodiscard]] s64 get_predecessors(s64 node_index, u32 *output_buffer) const;

        // Every edge's weight (basis plus S), indexed by edge ordinal; total_edge_count values.
        void neighbor_weights(f32 *output_weights) const;
        void neighbor_weights_for_matrix(f32 *output_weights, s64 matrix_index) const;

        [[nodiscard]] Weight_Stats neighbor_weight_stats() const;

        // ── updates ──────────────────────────────────────────────────────────────
        // Adds delta to one edge's value in one plane through S; reads see it at once. S grows
        // when full. A NaN or infinite delta throws.
        void accumulate_edge_delta(s64 matrix_index, s32 source_node, s32 target_node, f32 delta);

        // Merges the device's queued updates into S, summing per edge and dropping entries of exactly
        // zero, including those the device zeroed in place. S grows rather than overflow.
        void compact_pending_deltas();

        // ── calibration ─────────────────────────────────────────────────────────
        // Planes that hold no lanes yet: those that started at zero, waiting for their lane budget.
        [[nodiscard]] Vector<s64> planes_awaiting_lane_budget() const;

        // One plane's updates in S: as (edge ordinal, delta) by ordinal; as one value per edge (zero
        // where there is none); and their sum of squares.
        [[nodiscard]] Vector<Pair<s64, f32>> plane_updates(s64 matrix_index) const;
        [[nodiscard]] Vector<f32> plane_update_values(s64 matrix_index) const;
        [[nodiscard]] f64 plane_update_sum_of_squares(s64 matrix_index) const;

        // Sets the lane budget of a plane that holds no lanes, from values it is expected to hold
        // (calibration_values, one per edge): the fewest lanes, searched with lane_search_fit_method,
        // that hold them within fit_tolerance, times one plus expected_active_edge_fraction, within
        // what lane_cap leaves after the other planes. The plane keeps the fitted V rows and zero U,
        // so it reads exactly zero until its first refit, which starts from those rows. S is left as
        // it is: the fit holds a list of the connections and their values (12 bytes each) while it runs.
        void calibrate_lane_budget(s64 matrix_index, const Vector<f32> &calibration_values);

        // Empties S and the queue, dropping every update, and shrinks S to occupancy_threshold_fraction.
        void discard_updates(f32 occupancy_threshold_fraction);

        // Refits each plane that has updates to the basis plus S with fit_method, then empties S. A
        // plane's fit reads its S entries keyed by target node, and its values from the basis plus
        // them, so it keeps nothing per edge.
        // A plane changes only its own lanes, starting from them and moving only the rows at either
        // end of an updated edge. Its lane count never changes: construction chose it, or, for a
        // plane with no lanes yet (a state plane starts at zero), the plane's first refit chooses
        // it by the construction search. A refit that misses fit_tolerance warns. Planes without
        // updates read as before. Expensive, so the engine runs it behind a threshold.
        void refit(f32 ridge_regularization = DEFAULT_FIT_RIDGE);

        // True once the fullest plane of S holds updates on occupancy_threshold_fraction of the
        // edges. A threshold of zero or less never refits.
        [[nodiscard]] bool is_refit_due(f32 occupancy_threshold_fraction) const;

        // Fraction of the edges holding an update, in the fullest plane.
        [[nodiscard]] f32 sparse_delta_occupancy_fraction() const;

        // S's pool capacity for a refit threshold: that fraction of the edges (every edge when it is
        // zero or less) for each plane the kernel updates, plus the plasticity reserve.
        // resize_delta_capacity keeps S's updates; throws if they do not fit.
        [[nodiscard]] s64 delta_capacity_for_threshold(f32 occupancy_threshold_fraction) const;
        void resize_delta_capacity(s64 new_capacity);

        // The queue size for a refit threshold: like S (delta_capacity_for_threshold), and at least
        // one neuron's walk, so every pass walks at least one deferred neuron.
        [[nodiscard]] s64 pending_capacity_for_threshold(f32 occupancy_threshold_fraction) const;

        // Resizes the queue, emptying it. The kernel bakes the queue's size in, so the engine calls
        // this only before building its kernel.
        void resize_pending_delta_capacity(s64 new_capacity);

        // Updates the device queued and compact_pending_deltas has not merged.
        [[nodiscard]] s64 pending_delta_count() const;

        // A tick's walk passes: begin_first_walk_pass before its first dispatch; after each merge,
        // begin_next_walk_pass gives the neuron count the next dispatch walks, or 0 when none waits.
        void begin_first_walk_pass();
        [[nodiscard]] s64 begin_next_walk_pass();

        // Scales the weight plane (its coefficient row and its S entries) to a target RMS. U and V
        // are shared with every plane, so they stay as they are.
        Scale_Result scale_neighbor_weights_to_root_mean_square(f32 target_root_mean_square,
                                                              f32 epsilon = 1e-12f) const;


        [[nodiscard]] bool check_index_inbounds(s32 source, s32 target) const;
        [[nodiscard]] bool check_index_inbounds(s32 node_index) const;

        // One plane's coefficient row, as a range a kernel can bind.
        [[nodiscard]] Engine_Pointer coefficient_range(s64 matrix_index) const;

        void save(const char *filepath) const;
        void load_from_disk(const char *filepath);

    private:
        // Host copy of edge_row_offset, written at construction; the device copy is made from it.
        Vector<s64> edge_row_offset_host;

        static const Vector<Vector<s32>> &validate_network(const Vector<Vector<s32>> &network);

        // neighbor_weights_kernel from src/spikecorec/metal/kernels.metal, loaded on first use.
        // Mutable because the const reads that need it only cache it.
        mutable Engine_Function neighbor_weights_function;

        // Loads the precompiled kernel `name` into `function` once. False means the caller uses
        // the host path; a failed load is not retried.
        [[nodiscard]] bool ensure_function(Engine_Function &function, const String &name) const;
        mutable bool precompiled_functions_unavailable = false;

        // Seeds every random fill of the basis, so runs are reproducible across resizes.
        unsigned basis_seed = 0;

        // Fills U/V with independent N(0,1) and every coefficient row with 1.0.
        void seed_basis(unsigned seed) const;

        void build_edge_row_offset();

        // Allocate the basis slab at the current rank, S's slab at sparse_delta_capacity (empty) and
        // the queue's slab at pending_delta_capacity (empty). Each releases nothing; callers free the
        // slab they replace.
        void allocate_basis_storage();
        void allocate_delta_storage();
        void allocate_pending_storage();

        // Adds lanes, keeping U, V and every Ck. New lanes get random U and V and zero
        // coefficients, so every value reads the same until a fit uses them.
        void grow_basis(s64 new_rank);

        // The capacity S grows to when a plane needs required_entries.
        [[nodiscard]] s64 grown_delta_capacity(s64 required_entries) const;

        // Moves S into a new slab of new_capacity entries (at least the entries it holds). The
        // planes stay packed from position 0, so the row starts and entries are copied as they are.
        void move_delta_storage(s64 new_capacity);

        // Σ_k U[i][k] * Ck[k] * V[j][k]: the basis value for one coefficient row, without S.
        [[nodiscard]] f32 reconstruct_entry(s32 source_node, s32 target_node,
                                            const f32 *coefficient_values) const;

        // The update S holds for one edge in one plane, or zero. Binary search in the source row.
        [[nodiscard]] f32 sparse_delta_for(s64 matrix_index, s32 source_node, s64 edge_ordinal) const;

        // Rebuilds S from every plane's full list of (edge_ordinal, delta) pairs, sorting each list.
        // S grows first when they do not fit.
        void rebuild_sparse_deltas(Vector<Vector<Pair<s64, f32>>> &deltas);

        // Every plane's updates, as (edge_ordinal, delta) lists.
        [[nodiscard]] Vector<Vector<Pair<s64, f32>>> current_deltas() const;

        void clear_sparse_deltas();

        [[nodiscard]] f32 *coefficient_row(s64 matrix_index) const;

        // ── fitting ─────────────────────────────────────────────────────────────
        // A device slab of fit scratch, released when it goes out of scope. Every buffer holds at
        // least one element, so each is a real kernel argument.
        class Fit_Scratch {
        public:
            Fit_Scratch(Engine_Backend &backend, const Vector<u64> &byte_counts);
            ~Fit_Scratch();
            Fit_Scratch(const Fit_Scratch &) = delete;
            Fit_Scratch &operator=(const Fit_Scratch &) = delete;

            Vector<Engine_Pointer> buffers;

        private:
            Engine_Backend *backend;
            Engine_Pointer slab;
        };

        // A plane's values as a fit reads them: CSR over source rows, each row's entries keyed by
        // target node and sorted, with s32 positions as in S. A fit walks the k^2-tree for the
        // connections and looks each one up here, so it keeps nothing per edge. With adds_to_basis
        // set the entries are updates to what the plane's basis holds (a refit reads its S entries
        // this way); otherwise they are the plane's values. storage is set when the entries have a
        // slab of their own rather than being a view of S.
        //
        // With lists_every_connection set, the entries are every connection in edge-ordinal order,
        // zeros included, and incoming_offset and incoming_entry list each node's incoming entries,
        // sources ascending. The fit kernels then read the connections from these arrays instead of
        // walking the tree: 12 bytes per connection, for construction only, where fits are an order
        // of magnitude faster that way. Refits walk.
        struct Fit_Values {
            Unique_Pointer<Fit_Scratch> storage;
            Engine_Pointer row_start;  // s32[node_count + 1]: node n's entries are positions [n] up to [n + 1]
            Engine_Pointer target;     // u32 per entry
            Engine_Pointer value;      // f32 per entry
            s64 entry_count = 0;
            bool adds_to_basis = false;
            bool lists_every_connection = false;
            Engine_Pointer incoming_offset;  // s32[node_count + 1], when listed
            Engine_Pointer incoming_entry;   // u32 per connection, when listed
        };

        // A plane's values from one value per edge ordinal: those that are not zero, or every
        // connection with list_every_connection set.
        [[nodiscard]] Fit_Values plane_values_from(const Function<f32(s64)> &value_of_ordinal,
                                                   bool list_every_connection) const;

        // Plane matrix_index's entries in S, as updates to its basis, once key_plane_updates_by_target
        // has keyed them by target node.
        [[nodiscard]] Fit_Values updates_of_plane(s64 matrix_index) const;

        // Rekeys one plane's S entries between edge ordinals, which ticks read, and target nodes,
        // which fits read. Within a row both orders are the same, so the entries stay sorted.
        void key_plane_updates_by_target(s64 matrix_index);
        void key_plane_updates_by_ordinal(s64 matrix_index);

        // A plane's values over every edge, with values' entries in place: their root mean square over
        // the finite ones, and how many are not finite. For updates, each value is the basis plus its
        // update, which takes a walk over every edge; otherwise only the entries are read.
        struct Plane_Value_Statistics {
            f64 root_mean_square = 0.0;
            s64 non_finite_count = 0;
        };
        [[nodiscard]] Plane_Value_Statistics plane_value_statistics(s64 matrix_index, const Fit_Values &values) const;

        // The scalars every walking fit kernel reads, laid out as Fit_Walk_Constants in
        // weight_fit_kernels.metal.
        struct Fit_Walk_Constants {
            u32 node_count;
            u32 branching_factor;
            u32 superblock_size_words;
            u32 padded_node_count;
            u32 tree_height;
            u32 internal_bit_count;
            u32 basis_lane_count;
            u32 target_lane_count;
            f32 target_lane_sign;
            f32 inverse_scale;
            u32 lane_count;
            f32 ridge;
            u32 walkers_per_row;
            u32 connections_listed;
        };

        // One plane's fit. The plane's own lanes are copied into dense working arrays, one row per
        // node and one column per own lane, scaled so that the lanes' coefficient is the plane's RMS.
        struct Plane_Fit_Problem {
            s64 matrix_index = 0;
            Vector<s64> own_lanes;          // basis lanes, in working-column order
            f64 scale = 0.0;                // the plane's RMS: its own lanes' coefficient
            f64 value_root_mean_square = 0.0;  // the plane's values' RMS (the scale, unless they are all zero)
            s64 non_finite_value_count = 0;  // values no basis can hold; they get no weight
            f64 target_tolerance = 0.0;      // the worst relative error the candidates stop at

            // The rows the fit may move, sources then targets. When every row with a connection moves
            // (a fit from scratch), every connection is touched.
            s64 moved_source_count = 0;
            Vector<u32> moved_rows;
            bool every_edge_is_touched = false;

            // What the own lanes must hold at a connection (its target, fit_target in
            // weight_fit_kernels.metal), in units of the scale: the basis over target_lanes times
            // target_lane_sign, plus the connection's entry in values. For updates, target_lanes are
            // the plane's own lanes, whose values before the fit the updates add to; for whole values,
            // the lanes the plane shares with other planes, whose part the own lanes cancel. A target
            // is NaN for a value no basis can hold; its weight, 1 / max(|target|, 1)^2, is then zero.
            const Fit_Values *values = nullptr;
            Vector<u32> target_lanes;
            f32 target_lane_sign = 1.0f;

            // Per node, 1 when its row moves as a source. Evaluating the touched connections, the
            // row walks of the moved sources count theirs, and the column walks of the moved targets
            // skip those. Device copies follow; the partials take one reduction value per threadgroup.
            Vector<u8> source_moves;
            Unique_Pointer<Fit_Scratch> storage;
            Engine_Pointer source_move_buffer;
            Engine_Pointer moved_row_buffer;
            Engine_Pointer target_lane_buffer;
            Engine_Pointer first_partials;   // evaluate_edges: loss, the row walks' then the column walks'
            Engine_Pointer second_partials;  // evaluate_edges: worst error, likewise
            // How many partials evaluate_edges' last row and column walks wrote, for read_edge_evaluation.
            mutable s64 row_walk_partial_count = 0;
            mutable s64 column_walk_partial_count = 0;

            // fill_fresh_rows' starting rows: each source row's mean target, and each target column's
            // less the mean of every target. The targets do not change between the probes of a
            // search, so they are taken once.
            mutable Vector<f32> fresh_row_mean;
            mutable Vector<f32> fresh_column_mean;
            Engine_Pointer gradient_first_partials;   // evaluate_gradient: ridge term
            Engine_Pointer gradient_second_partials;  // evaluate_gradient: slope
            Engine_Pointer dot_partials;     // dot_products, MAXIMUM_DOT_PRODUCT_LEFT_SLOT_COUNT rows
            Engine_Pointer column_partials;  // balance_lanes' per-lane sums

            // [node_count][own_lanes.size()] each; reallocated when the lane count changes.
            Unique_Pointer<Fit_Scratch> working_storage;
            Engine_Pointer working_u;
            Engine_Pointer working_v;
        };

        // Sets up one plane's fit to values. With every_row_moves set, every row with a connection
        // moves; otherwise (updates) only the rows at either end of an entry.
        [[nodiscard]] Plane_Fit_Problem prepare_plane_fit(s64 matrix_index, const Fit_Values &values,
                                                          bool every_row_moves);

        // The arguments every walking fit kernel starts with, in the order weight_fit_kernels.metal
        // takes them. constants must outlive the launch.
        [[nodiscard]] Fit_Walk_Constants fit_walk_constants(const Plane_Fit_Problem &problem, f32 ridge_regularization) const;
        [[nodiscard]] Vector<Engine_Pointer> fit_walk_arguments(const Plane_Fit_Problem &problem,
                                                                const Fit_Walk_Constants &constants) const;

        // Host versions of the kernels' helpers: a connection's entry in the fit's values (zero when
        // there is none), and its target.
        [[nodiscard]] static f32 fit_value_at(const Fit_Values &values, u32 source_node, u32 target_node);
        [[nodiscard]] f32 fit_target(const Plane_Fit_Problem &problem, u32 source_node, u32 target_node, f32 value) const;

        // Host walks: calls visit(other_end) for each of node's targets (along its row) or sources
        // (along its column), ascending, using buffer for the walk's output.
        template <typename Visit>
        void for_each_connection_of(u32 node, bool along_column, Vector<u32> &buffer, const Visit &visit) const;

        // The rows a fit from scratch starts from: the first lane holds each source row's mean value
        // against target rows of one, the second rows of one against each target column's mean less
        // the overall mean, so values that are a source part plus a target part start next to their
        // exact fit; the other lanes hold N(0, lane_count^-1/4) values.
        void fill_fresh_rows(const Plane_Fit_Problem &problem, unsigned seed) const;

        // Points the problem at the plane's current own lanes, sets their coefficient to the scale
        // (rescaling U so no value changes) and copies them into fresh working arrays.
        void load_working_columns(Plane_Fit_Problem &problem);

        // Copies the working arrays back into the plane's own lanes of U and V.
        void store_working_columns(const Plane_Fit_Problem &problem) const;

        // The two candidates. Each fits the problem's own lanes at their current count from the
        // working arrays as they are, moving only the problem's rows, until the stopping rule
        // (see FIT_STALL_IMPROVEMENT) ends it. Leaves the best rows it found in the working
        // arrays and returns their worst relative error over the touched edges.
        f64 fit_plane_by_limited_memory_broyden_fletcher_goldfarb_shanno(Plane_Fit_Problem &problem,
                                                                         f32 ridge_regularization);
        f64 fit_plane_by_masked_alternating_least_squares(Plane_Fit_Problem &problem, f32 ridge_regularization);

        // Runs the candidate fit_method names.
        f64 fit_plane_with_selected_method(Plane_Fit_Problem &problem, f32 ridge_regularization);

        // A fit's outcome: whether it met its tolerance, and its worst relative error over every edge
        // (infinite when the plane holds a value no basis can hold).
        struct Plane_Fit_Result {
            bool meets_tolerance = true;
            f64 worst_error = 0.0;
        };

        // Construction: fits the plane from scratch at 1, 2, 4, ... own lanes until one meets
        // search_tolerance, then bisects for the fewest that do, and keeps that fit. When none does,
        // keeps the fit with the smallest error. With a required_lane_count above zero it fits at
        // exactly that many lanes instead. The probes work outside the basis, which takes only the
        // chosen lanes.
        // A lane_limit above zero caps the lanes the search may use. The doubling also stops once two
        // doublings in a row each cut the worst error by less than FIT_STALL_IMPROVEMENT: values that
        // more lanes barely help (random ones) keep the closest fit instead of running to the cap.
        // The search frees the plane's lanes first, so values that add to the basis must belong to a
        // plane that holds none.
        Plane_Fit_Result search_fewest_lanes(s64 matrix_index, const Fit_Values &values, f32 ridge_regularization,
                                             f32 search_tolerance, s64 required_lane_count = 0, s64 lane_limit = 0);

        // Refit: fits the plane within its lane budget. From its current lanes, moving only the
        // rows at either end of a changed edge; from fresh rows, moving every row, when its lanes
        // hold nothing. A plane with no lanes yet (one that started at zero) gets its budget here,
        // once, from the first values it holds: the fewest lanes that meet fit_tolerance, times one
        // plus expected_active_edge_fraction, within what lane_cap leaves after the other planes.
        // updates are the plane's S entries, keyed by target. Records the plane's measured_fit_error.
        Plane_Fit_Result refit_plane(s64 matrix_index, const Fit_Values &updates, f32 ridge_regularization);

        // Construction: frees every lane, searches every plane with starting values for its fewest
        // lanes, sets each plane's lane budget (see expected_active_edge_fraction), cuts the largest
        // budgets first while together they would pass lane_cap, and gives every plane with values
        // its budget, fitting again any plane cut below its fewest lanes. A plane that starts at zero
        // takes no lanes: it counts for the largest budget in the cut, which keeps room under the
        // cap, and gets its own budget at its first refit. Returns whether every plane meets
        // fit_tolerance, and records each plane's measured_fit_error.
        // plane_values(m) gives plane m's values; it is called once per use, so only one plane's
        // values are held at a time.
        bool fit_every_plane_within_lane_budgets(const Function<Fit_Values(s64)> &plane_values,
                                                 f32 ridge_regularization);

        // The most lanes the planes may hold together: the kernel limit, and the most that keep the
        // basis (U, V and the coefficients) smaller than one float per edge per plane. A network too
        // small for any basis to be that small gets the lanes of its smallest basis, one lane group
        // per four planes, which are allocated whatever the budgets.
        [[nodiscard]] s64 lane_cap() const;

        // ── lanes ───────────────────────────────────────────────────────────────
        // A plane owns the lanes only it reads (a nonzero coefficient); a lane no plane reads is free.
        [[nodiscard]] Vector<s64> lanes_owned_by(s64 matrix_index) const;
        [[nodiscard]] Vector<s64> lanes_shared_by(s64 matrix_index) const;  // read by it and another plane
        [[nodiscard]] s64 free_lane_count() const;

        // Lanes the plane could hold: its own, the free ones and the room left under the lane limit.
        [[nodiscard]] s64 available_lane_count(s64 matrix_index) const;

        // Gives the plane `count` more lanes, free ones first, then new ones, with coefficient
        // `scale`, U zero and V N(0,1), so no value changes until a fit uses them.
        void take_lanes(s64 matrix_index, s64 count, f64 scale);

        // Frees the plane's own lanes after the first kept_count, zeroing them.
        void release_lanes(s64 matrix_index, s64 kept_count);

        // Drops the lanes no plane reads, keeping the rest in order, and shrinks the basis to fit.
        void compact_basis();

        // ── fit steps ───────────────────────────────────────────────────────────
        // Each runs its kernel from weight_fit_kernels.metal, or the same arithmetic on the host
        // when the kernel cannot run (CUDA, or a kernel that did not load).
        mutable Unordered_Map<String, Engine_Function> fit_functions;
        [[nodiscard]] bool run_fit_kernel(const String &name, Vector<Engine_Pointer> &parameters, s64 job_count) const;

        // Fit kernel launches collected for one command buffer (Engine_Backend::run_functions).
        // Inline scalars are copied in, so the caller's values need not outlive the batch.
        struct Fit_Kernel_Batch {
            Deque<Vector<u8>> scalar_storage;
            Vector<Engine_Dispatch> dispatches;
        };
        // While set, run_fit_kernel adds each launch to it instead of running it, and the steps
        // that reduce return nothing: their read_* functions give the results once the batch ran.
        mutable Fit_Kernel_Batch *collecting_batch = nullptr;

        // True when every fit kernel loads, so a batch never mixes device and host steps.
        [[nodiscard]] bool fit_kernels_available() const;

        // Runs encode's fit steps as one command buffer when every fit kernel loads, else one by one.
        // Returns whether they were batched (then the reductions are read with the read_* functions).
        bool run_fit_steps(const Function<void()> &encode) const;

        struct Edge_Evaluation {
            f64 loss = 0.0;         // the sum of 0.5 * weight * residual^2 over the touched edges
            f64 worst_error = 0.0;  // the worst relative error over them
        };
        // Residuals of the working arrays at the touched edges, or at every edge with every_edge set.
        [[nodiscard]] Edge_Evaluation evaluate_edges(const Plane_Fit_Problem &problem, bool every_edge = false) const;
        [[nodiscard]] Edge_Evaluation read_edge_evaluation(const Plane_Fit_Problem &problem) const;

        // Masked alternating least squares: solves every moved row on one side, given the other, by
        // QR, in batches of batch_row_count rows. The scratch holds each batch row's triangle
        // (lane_count^2) and two vectors (2 * lane_count).
        void solve_rows(const Plane_Fit_Problem &problem, bool solving_sources, f32 ridge_regularization,
                        const Engine_Pointer &triangle_scratch, const Engine_Pointer &vector_scratch,
                        s64 batch_row_count) const;

        struct Gradient_Evaluation {
            f64 ridge_term = 0.0;  // 0.5 * ridge * |moved rows|^2
            f64 slope = 0.0;       // gradient . direction
        };
        // The packed gradient at the working arrays, recomputing each connection's residual.
        [[nodiscard]] Gradient_Evaluation evaluate_gradient(const Plane_Fit_Problem &problem, f32 ridge_regularization,
                                                            const Engine_Pointer &direction,
                                                            const Engine_Pointer &gradient) const;
        [[nodiscard]] Gradient_Evaluation read_gradient_evaluation(const Plane_Fit_Problem &problem) const;

        // Rescales each own lane so its U and V columns have the same norm: U by c and V by 1 / c,
        // which leaves every value the same and lowers the ridge term. Unbalanced lanes (U small
        // and V large, as new lanes start) let the ridge dominate the smaller side's solve.
        void balance_lanes(const Plane_Fit_Problem &problem) const;

        // Writes point + step_length * direction into the moved rows; gather_point packs them. The
        // step is a value, or a one-float buffer an earlier step of the same batch wrote.
        void scatter_point(const Plane_Fit_Problem &problem, const Engine_Pointer &point,
                           const Engine_Pointer &direction, f32 step_length) const;
        void scatter_point(const Plane_Fit_Problem &problem, const Engine_Pointer &point,
                           const Engine_Pointer &direction, const Engine_Pointer &step_length) const;
        void gather_point(const Plane_Fit_Problem &problem, const Engine_Pointer &point) const;

        // Makes the point in the working arrays and trial_gradient current, recording the step
        // and the gradient change in their history slots of basis when record_history is set.
        void accept_step(const Plane_Fit_Problem &problem, const Engine_Pointer &point,
                         const Engine_Pointer &trial_gradient, const Engine_Pointer &basis, u32 step_slot,
                         u32 gradient_change_slot, u32 gradient_slot, bool record_history) const;

        // Each of left_slots (at most MAXIMUM_DOT_PRODUCT_LEFT_SLOT_COUNT) dotted with each of the first
        // vector_count basis slots, in one pass, written into products ([vector_count][vector_count]
        // f32) as fit_reduce_dot_products_kernel writes it.
        static constexpr u32 MAXIMUM_DOT_PRODUCT_LEFT_SLOT_COUNT = 3;
        void update_dot_products(const Plane_Fit_Problem &problem, const Engine_Pointer &basis, u64 length,
                                 u32 vector_count, const Vector<u32> &left_slots, const Engine_Pointer &products) const;

        // The quasi-Newton direction: fit_quasi_newton_direction_kernel, which documents the buffers.
        // new_pair_slot is the slot of a pair just recorded, or u32's maximum for none.
        void quasi_newton_direction(const Engine_Pointer &products, const Engine_Pointer &history, u32 history_length,
                                    u32 new_pair_slot, bool reset_history, const Engine_Pointer &coefficients,
                                    const Engine_Pointer &status, const Engine_Pointer &step_length) const;

        // output = the sum over the first vector_count basis slots of coefficients[slot] * slot, with
        // the coefficients in a buffer an earlier step of the same batch may have written.
        void linear_combination(const Engine_Pointer &basis, u64 length, u32 vector_count,
                                const Engine_Pointer &coefficients, const Engine_Pointer &output) const;

        // Merges neighbouring runs with the same prototype, once the values are in the basis.
        void merge_runs_by_prototype();

        // Sets one plane's measured_fit_error.
        void record_fit_error(s64 matrix_index, f64 worst_error);

    public:
        // The worst of measured_fit_error over every plane.
        [[nodiscard]] f32 worst_fit_error() const;

    private:

        void resize_basis(s64 new_rank);

        void validate_matrix_index(s64 matrix_index) const;

        // Throws when the matrix has no slab to reallocate, as the empty matrix does.
        void require_storage(const char *operation) const;
    };
}
