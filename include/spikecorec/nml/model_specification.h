#pragma once

#include <optional>

#include "spikecorec/nml/resolve.h"
#include "spikecorec/core/weight_matrix.h"

namespace spikecorec::nml {

// ── ModelSpecification (ticket #7 [A5]; arch §1.4) ──────────────────────
//
// The lowered, flat, index-addressed representation the whole back half of
// the codegen pipeline reads (allocation #5, cell/synapse lowering #50/#51,
// IR->GPU source #55, master-kernel assembly #6). Built once from a
// ResolvedModel (ticket #49 [A4]'s output) by build_model_specification().
//
// Why flat tables, not a walked tree (arch §1.4): the counts that matter
// here are *types* and *populations* (dozens-hundreds), never neurons
// (millions). The only thing that stays tree-shaped is the per-type
// expression ASTs carried inside `TypeLibraryEntry::dynamics` (already
// opaque NML_Node/string content from #49 -- TimeDerivativeDecl.value,
// OnConditionDecl/OnEventDecl/OnStartDecl bodies) -- an ODE right-hand side
// is irreducibly a tree.

// Which Dynamics category (arch §1.1/§3.3 D1/D3/D4) a TypeLibraryEntry's
// bound ComponentType belongs to. Mirrors which alternative of
// ComponentTypeEntry `TypeLibraryEntry::dynamics.flattened` holds -- kept as
// an explicit enum since downstream (allocation, codegen) branches on it far
// more often than it needs to re-inspect the variant.
enum class TypeLibraryCategory { Cell, Synapse, Inputs };

// One cell/synapse/inputs type as actually materialized by a population,
// projection, or stimulus in THIS model (arch §1.4's "cell-type and
// synapse-type library").
//
// Keyed to one specific BOUND COMPONENT INSTANCE, not the bare ComponentType
// declaration: in standard NeuroML a population/projection/explicitInput
// binds exactly one component, and that component's own attribute values are
// exactly what bake-vs-parameterize (arch §3.1 `Parameter`) bakes. So two
// populations sharing a ComponentType but bound to different components
// (e.g. an excitatory vs. inhibitory GLIF population -- same cell type,
// different gL/EL) get two distinct entries here, each with its own baked
// constants; multiple populations/projections/inputs that happen to bind the
// exact same component instance share one entry (deduplicated by
// `bound_instance_id`). A population's `type_library_index` (below) is an
// index into this per-(ComponentType, bound instance) table, matching arch
// §4.1's "cell-type boundary" -- each distinct bound instance is a
// potentially distinct generated-code flavor once baked constants differ.
struct TypeLibraryEntry {
    String component_type_name; // the underlying ComponentType, e.g. "GLIF1Cell"
    String bound_instance_id;   // the specific bound component's own id, e.g. "glif1ExcCellInstance"
    TypeLibraryCategory category = TypeLibraryCategory::Cell;

    // The flattened ComponentType (#49): its declarations plus the whole
    // `extends` chain merged in, one entry's `bucket` always
    // ComponentTypeBucket::Dynamics here. Also where `OnStart` -- the
    // "initial values" deliverable -- already lives (CellType::on_starts /
    // SynapseType::on_starts / InputsType::on_starts inside `.flattened`);
    // no separate copy is kept, since arch §1.4 says only the expression
    // ASTs stay tree-shaped and OnStart's StateAssignment bodies already are
    // exactly that.
    ResolvedComponentType dynamics;

    // Bake-vs-parameterize (arch §3.1 `Parameter`/`Property`), applied per
    // Phase-1's default rule: a population/projection/input binds one
    // component, so every neuron/edge/instance sharing this entry has the
    // SAME resolved value for each of the type's Parameters -- baked as a
    // literal (zero memory). Merges the type's own `Fixed`-pinned values
    // (`dynamics.fixed_parameter_values`) with the bound instance's own
    // numeric attribute values; the instance's value wins on a name
    // collision, since a bound instance's own value is strictly more
    // specific than an inherited `Fixed` pin. Genuine per-neuron
    // heterogeneity (arch §3.1: distinct components, per-cell distribution
    // draws, or Phase-3 `inhomogeneousParameter`) is out of Phase-1 scope --
    // it would hook in here as a per-entry flag plus a per-neuron array,
    // replacing a baked entry when a future Structure-level construct makes
    // the values non-uniform.
    UnorderedMap<String, f64> baked_constants;

    // Meaningful only when category == Synapse (arch §3.1 `ComponentType`,
    // §3.3 D3): does the `extends` chain pass through
    // `baseConductanceBasedSynapse`/`baseConductanceBasedSynapseTwo` (current
    // = g*(erev-v), needs postsynaptic v -> conductance-based) or not
    // (current = g -> current-based)? Always false for Cell/Inputs entries.
    bool is_conductance_based = false;

    // Per-instance state slot count (`V_t`, arch §4.1) -- state_variables
    // size after extends-flattening. Used to size population chunk offsets.
    s32 state_variable_count = 0;
};

// Populations — arch §1.4/§4.1: cell-type index, size, neuron-index range,
// and state-vector chunk base offset. Built by walking resolved
// `population`/`populationList` instances.
struct PopulationEntry {
    String id;
    s32 type_library_index = -1;
    s32 size = 0;
    s32 neuron_index_begin = 0;
    s32 neuron_index_end = 0; // exclusive
    s64 state_vector_chunk_base_offset = 0;
};

// One `connection`/`connectionWD` inside a projection: the per-edge routing
// + initial weight/delay data (arch §1.4, §3.1 `Property`, §4.4). `delay`
// is 0.0 for a plain `connection` (no `delay` attribute given) -- consumed
// by the delay-ring subsystem (ticket #64 [F3], nml/delay_ring.h), which
// floors a zero/absent delay to the engine's existing implicit one-tick
// latency (arch §4.4).
struct ConnectionEntry {
    s32 source_neuron_index = -1;
    s32 target_neuron_index = -1;
    f64 weight = 1.0;
    f64 delay = 0.0;
};

// Projection/synapse routing (arch §1.4, §3.3 ST3): one presynaptic
// population wired to one postsynaptic population through one synapse type,
// plus its connections.
struct ProjectionEntry {
    String id;
    s32 presynaptic_population_index = -1;
    s32 postsynaptic_population_index = -1;
    s32 synapse_type_library_index = -1;
    Vector<ConnectionEntry> connections;
};

// Input (stimulus) spec (arch §1.4, §3.3 D4) — one `explicitInput`: which
// input component instance drives which target neuron. Phase 1
// host-precomputes generator output (arch §5); this table is what a later
// INIT stage reads to do that.
struct StimulusEntry {
    s32 target_neuron_index = -1;
    s32 input_type_library_index = -1;
};

// Output (recording) selection (arch §1.4, §4.6): a state trace
// (`OutputColumn`/`Line`, `is_event_recording == false`, `exposure_name` the
// recorded quantity) or a spike record (`EventSelection`,
// `is_event_recording == true`, spikes read from `last_spiked`).
// `target_neuron_index` is -1 if `quantity_path`/`select` didn't resolve to a
// known population, OR named an out-of-range index within one that was
// resolved (kept for diagnostics rather than dropped or thrown -- unlike a
// connection/stimulus target, a recording target isn't trusted downstream as
// a raw array index, so an unresolved recording is a diagnostic, not fatal).
struct RecordingEntry {
    String id;
    String quantity_path;
    s32 target_neuron_index = -1;
    String exposure_name;
    bool is_event_recording = false;
};

// The lowered ModelSpecification (arch §1.4, ticket #7 [A5]). Populated from
// a ResolvedModel; consumed by allocation (#5) and codegen (#4/#50/#51/#55/
// #56) once those exist. Move-only (WeightMatrix is move-only).
struct ModelSpecification {
    Vector<TypeLibraryEntry> type_library;
    Vector<PopulationEntry> populations;
    Vector<ProjectionEntry> projections;
    Vector<StimulusEntry> stimuli;
    Vector<RecordingEntry> recordings;

    s32 total_neuron_count = 0;

    // Adjacency (arch §1.4): the existing `vector<vector<s32>>` ->
    // WeightMatrix/k²-tree path (§0.3), built from the union of every
    // projection's connections -- the exact edge SET (topology) only.
    // WeightMatrix's own U/V is a random-seeded low-rank factorization, not
    // a container for exact per-edge values (§0.3/§4.3) -- it does NOT hold
    // the real initial connection weights, so `adjacency->get(i, j)` is a
    // reconstructed/approximate value, not the modeled weight. The actual
    // initial weight (and delay) for each edge lives on its
    // `ConnectionEntry` (above), read from the model's `weight`/`delay`
    // Property values; seeding those exact values into U/V (or deciding not
    // to) is future allocator work (#52-#54/#57), not this ticket's job.
    // Unset if the model has no connections (WeightMatrix rejects an empty
    // network).
    std::optional<WeightMatrix> adjacency;
};

// Builds the ModelSpecification from a resolved model (ticket #49 [A4]'s
// output). `weight_matrix_rank` is forwarded to the WeightMatrix constructor
// (-1 -> its own default). Throws std::runtime_error if a
// population/projection/explicitInput references something resolve should
// have already guaranteed exists (a malformed ResolvedModel).
ModelSpecification build_model_specification(const ResolvedModel &resolved, s64 weight_matrix_rank = -1);

} // namespace spikecorec::nml
