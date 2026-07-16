#pragma once

#include "spikecorec/nml/nml.h"

namespace spikecorec::nml {

// ── RESOLVE + LOWER (ticket #49 [A4]; arch §1.1-§1.3, §3.1) ─────────────
//
// Turns the parsed tree (NML_Parser::library + ::root_node) into a
// resolved, unit-free, flattened model:
//   - a symbol table keyed by `id` wires every IDref (arch §1.2 S7);
//   - every dimensioned literal is normalized to canonical SI (S1/S2);
//   - every ComponentType's `extends` chain is merged and every `Fixed`
//     pins its parameter to a baked SI constant (§3.1 `ComponentType`,
//     `Fixed`);
//   - every ComponentType is bucketed Dynamics/Structure (§1.1).
//
// This is deliberately NOT the full ModelSpecification (ticket #7 [A5]) --
// no population/adjacency/projection-routing tables, no allocation layout,
// no recording selections. Just enough of a self-contained structure to
// hold a correct resolved+flattened model: nothing in it references back
// into the transient NML_Node tree or the un-flattened `library`, so the
// parser that produced it may be discarded once resolve_and_lower returns
// (arch §1.3 "the parse tree is transient... discarded at LOWER").

// `Fixed` (§3.1): pins an inherited Parameter to a literal, resolved to its
// SI value. Kept separate from ParameterDecl (which never carries a value,
// by ticket #2's design) since pinning is strictly a resolve-time concern.
struct FixedDecl {
    String parameter;
    String value;
};

// Which of arch §1.1's lifecycle buckets a resolved ComponentType falls
// into: CellType/SynapseType/InputsType declare <Dynamics> (D1/D3/D4) and
// drive per-tick step behavior; PopulationType/ProjectType and any
// out-of-scope ComponentTypeBase (ion channels, morphology, ... — ST4/ST5)
// are build-time-only Structures.
enum class ComponentTypeBucket { Dynamics, Structure };

// One ComponentType, fully flattened: its own declarations plus every
// ancestor's (walking the whole `extends` chain, arch §3.1), with every
// `Fixed` pinned to an SI constant. Codegen and everything downstream only
// ever sees this — nothing needs to walk `extends` again.
struct ResolvedComponentType {
    ComponentTypeEntry flattened;
    ComponentTypeBucket bucket = ComponentTypeBucket::Structure;
    UnorderedMap<String, f64> fixed_parameter_values;
};

// id/name → stable integer index, backing S7 IDref wiring. Shared by every
// declaration-level ComponentType name and every instance-level `id`
// anywhere in the parse tree (arch §1.3).
class SymbolTable {
public:
    // Assigns `key` the next free index the first time it is seen; returns
    // the existing index on every subsequent call for the same key.
    s32 register_symbol(const String &key);

    bool contains(const String &key) const;

    // Throws std::runtime_error if `key` was never registered (S7 unresolved
    // IDref).
    s32 index_of(const String &key) const;

private:
    UnorderedMap<String, s32> indices_;
};

// One instance-level node from the parse tree (arch §1.1 Statics),
// resolved: units → SI (S1/S2) and IDref attributes → symbol-table indices
// (S7). Attributes are split by kind rather than kept as raw strings so
// nothing downstream needs to re-guess which attribute is which.
struct ResolvedInstance {
    String tag_name;
    String id;               // this instance's own `id`, if it has one
    s32 symbol_index = -1;   // this instance's own symbol-table index, or -1 if it has no `id`

    UnorderedMap<String, f64> numeric_attributes;   // S1/S2: dimensioned literals, unit-resolved
    UnorderedMap<String, s32> idref_attributes;     // S7: resolved IDref -> symbol-table index
    UnorderedMap<String, String> string_attributes; // S6/path/metadata, passed through unresolved

    Vector<ResolvedInstance> children;
};

// The resolved, unit-free, flattened model (arch §1.3 RESOLVE stage).
struct ResolvedModel {
    UnorderedMap<String, ResolvedComponentType> types;
    Vector<ResolvedInstance> instances; // the former root_node->body, resolved
    SymbolTable symbols;
};

// Runs the whole RESOLVE + LOWER pass over an already-parsed NML_Parser.
// Throws std::runtime_error on an unresolved IDref (S7) and
// std::invalid_argument on an unrecognized/unconvertible dimensioned literal
// (S1) -- see arch §1.2. The returned ResolvedModel holds no reference back
// into `parser`; `parser` may be discarded once this returns.
ResolvedModel resolve_and_lower(const NML_Parser &parser);

} // namespace spikecorec::nml
