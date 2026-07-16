#pragma once

#include "spikecorec/nml/ir.h"

namespace spikecorec::nml {

// ── IR `.tick` → GPU source (ticket #55 [C1]; docs/nml_ir_spec.md §3.3-§3.4, §5; arch §4.3) ──
//
// Lowers one IrProgram's `.tick` section to a standalone, compilable MSL source string and a
// standalone CUDA source string, by near-direct template substitution (the IR's own governing
// principle -- ir_spec.md line 8): every leaf instruction maps to one fixed snippet with its
// operands filled in, every control construct maps to one control template, no analysis of the
// dynamics themselves happens between reading the IR and emitting code.
//
// This header does NOT solve #6's master-kernel assembly (splicing many types' generated code into
// one kernel dispatched by cell-type boundary) or #5's `.alloc` allocator (sizing/placing the real
// engine buffers). Its job stops at "one IrProgram compiles to real MSL/CUDA source" -- the
// generated functions below are #6's raw material, wrapped in a calling convention of this ticket's
// own design (documented in full here, since #6 has nothing else to build on yet).
//
// ── What gets generated ──────────────────────────────────────────────────────────────────────
//
// Two kinds of generated top-level function per program, matching the two different dispatch
// granularities the locked spec's stages actually need (arch §5's "one thread per neuron" is true
// for @integrate/@detect/@emit/@reset/@propagate/@plasticity/@record, but NOT for @deliver's
// `onevent`, which is inherently per spike-delivery-edge):
//
// 1. **One combined per-neuron function**, named `<component_type_name>_tick`, holding
//    `@integrate`, `@detect`, `@emit`, `@reset`, `@propagate`, `@plasticity`, `@record` -- in that
//    canonical order, all in ONE function body sharing one local-variable scope. This is
//    deliberate, not a simplification: a boolean like GLIF1's `spiked` is computed in `@detect`
//    and read back in both `@emit` and `@reset` in the SAME tick for the SAME neuron -- splitting
//    stages into separate functions would lose that value between calls. Merging them mirrors
//    exactly what #6 will eventually do inside the master kernel's per-cell-type dispatch region
//    anyway (arch §5: "assemble every type's @integrate/@detect/@emit/@reset source into one
//    kernel"), so this ticket just does it one type early, standalone.
//    Dispatch: one thread per neuron of this type; `neuron_index` is the thread's index,
//    bounds-checked against a `neuron_count` parameter (mirrors every existing per-neuron kernel in
//    src/metal/kernels.metal, e.g. `decay_all_neurons_kernel`).
//
// 2. **One function per `onevent <port> { ... }` block found (top-level only) in `.tick.deliver`**,
//    named `<component_type_name>_deliver_<port>`. `.tick.deliver` is expected to hold nothing but
//    `onevent` blocks at its top level (true of every IR spec §4 example) -- anything else there is
//    unsupported (fatal) rather than guessed at. Dispatch: one thread per *delivered spike edge*
//    this tick, batched -- the function takes parallel `source_node_indices`/`target_node_indices`/
//    `edge_slot_indices` arrays + an `event_count` bound (mirrors the existing batch-of-queries
//    kernels, e.g. `k2tree_get_neighbors_batch_kernel`). `source_node`/`target_node`/`edge_slot`
//    inside the body are `int` locals read from those arrays at the thread's event index --
//    `edge_slot` is the slot within `source_node`'s own k^2-tree row (WeightMatrix's own Sk
//    indexing convention, weight_matrix.h), which the engine's not-yet-built spike-scatter
//    machinery would supply per delivered edge; this ticket does not build that machinery, only
//    documents the parameter shape it must eventually feed.
//
// If a program's `.tick` has nothing in the 7 per-neuron stages, function (1) is omitted; if
// `.tick.deliver` is empty, no function (2) is generated. A program can have zero, one, or several
// deliver functions (one per onevent block).
//
// ── Operand resolution (ir_spec.md §3.1) ────────────────────────────────────────────────────
//
// Every `.tick` operand is a plain name; this lowering classifies each occurrence into exactly one
// of four buckets and resolves it accordingly, uniformly across every leaf op that reads it:
//   - **Reserved**: `dt` -> the scalar parameter `dt`; `tick` -> the scalar parameter `tick`;
//     `network_inputs` -> `network_inputs[neuron_index]` in the per-neuron function, or
//     `network_inputs[target_node]` in a deliver function (network_inputs is the postsynaptic
//     neuron's accumulator either way -- ir_spec.md §3.5).
//   - **A `.alloc`-declared name**: a `param <name> = <literal>` bakes to a local
//     `const float <name> = <literal>;` (no parameter); a bare `param <name>` becomes a scalar
//     parameter (one value shared by every neuron of this type -- the natural reading of
//     "constant" for a name with no literal yet, since ir_spec.md's own §4 examples only ever use
//     the bare form and #50/#51's future lowering is what would eventually fill in an actual
//     literal or turn it `: dyn` for real per-neuron heterogeneity); `param <name> : dyn <dtype>`,
//     `state`, `accum`, `regime`, `expose`, `require` all become a per-neuron array parameter
//     resolved as `<name>[neuron_index]` in the per-neuron function. Referencing one of these
//     per-neuron array quantities as a plain operand *inside a deliver function* is unsupported
//     (fatal) -- which endpoint (source/target) it should index is genuinely ambiguous and no IR
//     spec §4 example needs it (their onevent bodies only ever reference a plain, non-array bare
//     `param`, e.g. `weight`, which resolves identically in either function kind). A `peredge`
//     name is NEVER resolved this way -- it has no plain-operand reading, only `loadedge`/`accedge`
//     (below); using one as a bare operand elsewhere is fatal.
//     `regime` defaults to a per-neuron `int` buffer; `expose`/`require`/bare-`param`'s dtype isn't
//     tracked by ir.h at all (no `dtype` field on those directives), so this lowering defaults
//     their backend type to `float`, matching every current spec example (Phase 1's LEMS surface
//     is float-valued) -- flagged here since a future ticket needing a non-float `expose`/`require`/
//     bare `param` would need ir.h itself extended first, not this file.
//   - **A numeric literal** (matches `-?[0-9]+(\.[0-9]+)?`, e.g. the `1`/`0` in
//     `set_regime r, 1`/`eq is_ref, r, 1`): substituted verbatim, unsuffixed (every current spec
//     example's bare literal operand is an integer regime tag compared against an `int` buffer, so
//     no float suffix is needed; a model needing a bare float literal operand would need one added
//     to ir.h's printed form first).
//   - **Anything else**: a local variable (an intermediate `t0,t1,...` or a compare/boolean
//     destination like `spiked`/`is_ref`). Declared once, at the top of its enclosing function, the
//     first time it appears as an instruction's destination (hoisted out of any enclosing
//     `if`/`forall` block so it stays visible for the rest of the function, C89-style) -- type is
//     `bool` for a compare/boolean op's destination (`gt/lt/ge/le/eq/ne/and/or/not`), `float`
//     otherwise. `t0,t1,...` names are NOT registers in the sense the locked spec means (ir_spec.md
//     §3.1's "no registers... that's the back end's job") -- this IS the back end, and here they
//     become ordinary named locals, single-assignment in every current example.
//
// ── `loadedge`/`accedge` (ir_spec.md §3.3, §4.3's shared-basis U/V + Ck + Sk) ────────────────
//
// `@edge` resolves to a (row_node, other_node, slot) triple depending on lexical context:
//   - inside a `forall neuron_in`/`forall neuron_out` body: row_node = `neuron_index` (the current
//     neuron), other_node/slot = that forall's own loop variables (see below).
//   - inside an `onevent` body: row_node = `source_node`, other_node = `target_node`,
//     slot = `edge_slot` (the onevent function's own parameters).
// `loadedge dst, <var>@edge` (var must be `peredge`) reconstructs
// `dot(U[row_node], Ck_var * V[other_node]) + Sk_var[row_node * max_neighbor_count + slot]` --
// exactly WeightMatrix's own `get_for_matrix` + sparse-delta-overlay math (weight_matrix.h/.cpp),
// and exactly `neighbor_weights_kernel`'s per-lane loop (src/metal/kernels.metal,
// src/cuda/kernels.cu) for the reconstruction half, with the MSL/CUDA divergence mirrored
// line-for-line: MSL builds a `float4` for the lane's Ck slice and calls `dot()`; CUDA manually
// expands the four lane products (matching neighbor_weights_kernel's own CUDA implementation,
// which avoids `float4` literal construction). `accedge <var>@edge, <value>` accumulates
// `Sk_var[row_node * max_neighbor_count + slot] += value` for a `peredge` var, or
// `<var>[other_node] += value` for an `accum` var (matches ir_spec.md §4's expOne comment:
// "aggregated -> g[target] += weight").
// `accedge <var>@neuron_in`/`@neuron_out` with NO enclosing `forall` (ir_spec.md §3.4's whole-set
// scatter/gather form, e.g. `accedge g@neuron_out, weight`) synthesizes the same for-loop a
// `forall` would, wrapping just that one instruction.
// `loadedge` only supports `@edge`; `loadedge <var>@neuron_in/@neuron_out` has no defined
// reduction into one destination and is unsupported (fatal) -- no IR spec §4 example needs it.
//
// **Known, documented limitation**: `k2tree.get_neighbors`/`k2t_find_nth_neighbor` only expose one
// direction -- a node's own row (its own outgoing adjacency; matches `step`'s existing scatter
// model, which walks a spiking neuron's OWN row to reach its downstream targets). There is no
// transposed/incoming-adjacency structure anywhere in this codebase yet. So `forall neuron_in` and
// `forall neuron_out` lower IDENTICALLY here -- both walk the current neuron's own row via
// `k2t_find_nth_neighbor`. This is a real gap, not a simplification this ticket can close (it would
// need either a second, transposed K2Tree per per-edge synapse type, or a new K2Tree capability --
// out of scope for a template-substitution ticket). Whoever wires up the real per-synapse-type
// k2tree construction (#50/#51/#5) needs to either (a) confirm the synapse's own k2tree is already
// built in the direction `forall neuron_in`'s callers actually need (i.e. row = target, not source,
// for a per-edge synapse's `@integrate` gather -- the NMDA example's own use, which needs a
// postsynaptic neuron's in-edges), or (b) extend K2Tree with a real transpose. Flagged, not solved.
//
// `forall`'s own loop: `for (long forall_slot_<N> = 0; forall_slot_<N> < max_neighbor_count;
// ++forall_slot_<N>) { int forall_neighbor_node_<N> = k2t_find_nth_neighbor(...); if
// (forall_neighbor_node_<N> < 0) continue; ...body... }` -- `<N>` is a per-forall-instance counter
// (unique within one generated function), so nested/sibling foralls never collide. Skipping a
// negative sentinel via `continue` (rather than `neighbor_weights_kernel`'s early `return`, since
// this loop iterates one neuron's several neighbors rather than being one thread per pair) is the
// one deviation the ticket's own body calls for.
//
// ── `rand`/`randn` (ir_spec.md §3.3) ─────────────────────────────────────────────────────────
//
// No RNG precedent exists anywhere in src/metal/kernels.metal or src/cuda/kernels.cu today,
// so this ticket picks one: a per-neuron persistent xorshift32 state (a new reserved buffer,
// `rng_state`, one `uint`/`unsigned int` per neuron -- NOT part of the locked spec's own reserved
// name list (`dt`/`tick`/`network_inputs`), since the spec doesn't name an RNG-state buffer; added
// here as a documented, minimal extension only when a program actually uses `rand`/`randn`).
// `rand dst` -> `dst = spikecorec_rand_uniform(rng_state[neuron_index]);` (uniform [0,1), top 24
// bits of one xorshift32 step). `randn dst` -> `dst = spikecorec_randn(rng_state[neuron_index]);`
// (Box-Muller over two uniform draws). Both helpers are emitted once, in the shared preamble,
// exactly like `k2t_find_nth_neighbor` already is for the existing kernels -- not a new pattern.
//
// ── `emit <port>` ────────────────────────────────────────────────────────────────────────────
//
// One per-neuron `bool` flag buffer per distinct port name emitted anywhere in the program, named
// `emit_<port>` (e.g. GLIF1's `emit spike` -> parameter `emit_spike`, set true at
// `emit_spike[neuron_index]`). This is a per-type placeholder for "this neuron just emitted a spike
// on this port" that the engine's not-yet-built propagate/deliver wiring would consume; it is not
// itself that wiring.
//
// ── Baked/scalar parameter and buffer ordering (the calling convention #6 gets) ─────────────
//
// Every generated function's parameter list is derived structurally from `program.alloc`, in
// `.alloc`'s own declaration order, filtered to only the names that function's own instructions
// actually reference (so the per-neuron `..._tick` function and each `..._deliver_<port>` function
// typically have different, independently-minimal signatures) -- followed by `rng_state` (if used),
// then one `emit_<port>` per port (if used), then up to three independently-gated peredge-related
// groups (a function can need any combination of these, not just "all or nothing"):
//   1. the k^2-tree WALK buffers (`internal_node_words`, `leaf_node_words`, `rank_superblock_table`,
//      `rank_subblock_table`, `branching_factor`, `superblock_size_words`, `padded_node_count`,
//      `tree_height`, `internal_bit_count`, `node_count`) -- present only if this function
//      discovers a neighbor via `forall`/a whole-set edge op (never true for a deliver function --
//      an onevent body's edge is already given directly as parameters, see below);
//   2. `max_neighbor_count` -- present whenever (1) is needed, or any `peredge` name is touched by
//      `loadedge`/`accedge` at all (it's the Sk index stride, needed even for an O(1) `accedge`
//      that never touches the basis);
//   3. the shared basis (`rank_float4_stride`, `U`, `V`, then one `coefficients_<name>` per
//      `loadedge`/`accedge`-touched `peredge` name) -- present whenever this function does a
//      `loadedge` on a peredge var ANYWHERE, whether or not (1) is also needed: a `forall` body's
//      `loadedge` (NMDA's `@integrate`) needs both the walk (to find the neighbor) and the basis
//      (to reconstruct its value); an `onevent` body's `loadedge` (e.g. a synapse reading its own
//      current value on spike arrival before updating it) needs only the basis -- its edge is
//      already known, so no walk. Each touched `peredge` name's own `sparse_delta_<name>` (Sk) is
//      always emitted (accumulate target for `accedge`, overlay term for `loadedge`), independent
//      of whether the basis is present.
// ...then the dispatch-bound parameter (`neuron_count` for the per-neuron function;
// `source_node_indices`/`target_node_indices`/`edge_slot_indices`/`event_count` for a deliver
// function). MSL parameters carry sequential `[[ buffer(N) ]]` indices in this same order
// (mirroring every existing kernel in src/metal/kernels.metal); CUDA parameters are positional in
// the same order. When #6 assembles many types' functions together, the shared preamble (k^2-tree
// walk helpers, RNG helpers) should be emitted once for the whole assembled file, not once per type
// -- this ticket emits it per program since each example here is compiled as its own standalone
// translation unit.
// One generated top-level function's name plus its ordered parameter-name list -- exposes this
// ticket's own "calling convention" (documented above) as DATA, not just text, so a caller (ticket
// #6's master-kernel assembly) can build a metal_dispatch/cuda_dispatch args[] array generically by
// resolving each name to a runtime pointer/value itself, instead of re-deriving which names a
// generated function references (duplicating this lowering's own internal scan/bake/dedup rules
// independently would risk silently drifting out of sync with the text actually emitted below).
// Names are exactly the argument identifiers used in the generated signature, in the same order,
// including the trailing dispatch-bound parameter(s) (`neuron_count` for the combined per-neuron
// function; `source_node_indices`/`target_node_indices`/`edge_slot_indices`/`event_count` for a
// deliver function).
struct GpuFunctionSignature {
    String function_name;
    Vector<String> parameter_names_in_order;
};

struct GpuSource {
    String msl_source;
    String cuda_source;

    // One entry per top-level function generated into msl_source/cuda_source, in the same order
    // they appear there: the combined per-neuron function first (if any), then one per onevent/
    // deliver block found in `.tick.deliver`, in that section's own order.
    Vector<GpuFunctionSignature> functions;
};

// Lowers program's `.tick` to standalone, compilable MSL and CUDA source (see the calling
// convention documented above). Does not touch `program.alloc` beyond reading it to derive
// parameter names/types/order -- `.alloc` itself is engine-interpreted at init (ticket #5), not
// compiled.
GpuSource lower_ir_program_to_gpu_source(const IrProgram &program);

// The shared k^2-tree bit-walk preamble (`k2t_find_nth_neighbor` + its rank/bit helpers) this
// lowering emits into a generated source whenever a program's `forall`/whole-set edge op needs it
// (see this header's own doc comment above). Exposed standalone so ticket #6's master-kernel
// assembly can reuse the SAME text for its own fixed k^2-tree propagate/scatter stage instead of
// maintaining a second copy that could drift out of sync with the walk this file's own `forall`
// lowering relies on.
String k2tree_walk_preamble_msl();
String k2tree_walk_preamble_cuda();

} // namespace spikecorec::nml
