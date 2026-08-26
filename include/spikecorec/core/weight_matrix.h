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
        static constexpr f32 WEIGHT_FIT_MAXIMUM_TOLERANCE = 1.0e-2f;

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

        // ── plasticity (opt-in; all three are empty when disabled) ────────────────
        // Per-edge weight deltas awaiting the next fold into U/V. Parallel arrays rather
        // than an array of pairs: the fold reads ordinals and values as separate typed
        // buffers, and a record layout would make each a strided load.
        //
        // Capacity is a fixed budget, never one slot per edge. A producer that finds the
        // store full triggers a fold instead of growing it, so this stays independent of
        // edge count -- which is the whole point.
        EnginePointer plasticity_edge_ordinals;   // s64[plasticity_delta_capacity]
        EnginePointer plasticity_delta_values;    // f32[plasticity_delta_capacity]
        EnginePointer plasticity_delta_count;     // s32[1], bumped atomically on device
        s64 plasticity_delta_capacity = 0;

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
        // plasticity_delta_capacity: 0 disables plasticity entirely, allocating nothing
        WeightMatrix(
            EngineBackend &backend,
            const vector<vector<s32>> &network,
            s64 rank = -1,
            bool check_indexing = true,
            s64 max_neighbor_count = -1,
            s64 weight_seed = -1,
            s64 plasticity_delta_capacity = 0
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

        // ── plasticity ───────────────────────────────────────────────────────────
        // Applies one edge's delta straight into U/V as a rank-1 nudge. The host-side
        // entry point; the simulation path stages deltas and folds them in batches.
        void update(s32 source_node, s32 target_node, f32 delta,
                    f32 learning_rate, f32 l2_regularization, s32 iterations);

        // Applies every delta staged in the plasticity buffers and clears the count. No-op
        // when plasticity is disabled or nothing has been staged.
        void fold_edge_deltas(f32 learning_rate, f32 l2_regularization, s32 iterations);

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

        void build_edge_row_offset();

        // Runs one partition -> allocate round for every buffer this matrix owns, at the
        // current rank. Called at construction and again by resize_basis.
        void allocate_storage();

        // Σ_k U[i][k] * Ck[k] * V[j][k], the one place a stored value is ever produced.
        [[nodiscard]] f32 reconstruct_entry(s32 source_node, s32 target_node,
                                            const f32 *coefficient_values) const;

        [[nodiscard]] f32 *coefficient_row(s64 matrix_index) const;

        // Builds U/V and both coefficient rows directly from the projection structure.
        // Exact for population-to-population projections -- see the implementation for the
        // argument. Returns the rank it used.
        s64 fit_basis_from_projections(const Vector<f32> &weight, const Vector<s32> &delay_ticks);

        // Worst relative error over every edge, against the declared per-projection values.
        [[nodiscard]] f32 measure_worst_relative_weight_error(const Vector<f32> &weight) const;

        // Number of edges whose reconstructed delay does not round to the declared one.
        [[nodiscard]] s64 count_delay_mismatches(const Vector<s32> &delay_ticks) const;

        void resize_basis(s64 new_rank);

        void validate_matrix_index(s64 matrix_index) const;
    };
}
