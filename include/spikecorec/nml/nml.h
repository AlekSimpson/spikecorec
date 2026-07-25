#pragma once

#include <initializer_list>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <variant>

#include "spikecorec/core/types.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {
// REFACTOR: this is for the most part good actually but still things can be consolidated and managed better

// deliverables:
//  - A lookup API the resolve pass calls to fetch a core ComponentType / include target by name/href.
//  - Document how the bundle is refreshed and where the version is pinned.

struct NML_Node {
    String tag_name;

    NML_Node(String tag_name) : tag_name(tag_name) {};

    NML_Node(const NML_Node &other) : tag_name(other.tag_name), attributes(other.attributes), body(other.body) {};

    UnorderedMap<String, Any> attributes;
    Vector<NML_Node> body;

    void add_attribute(String name, Any value);
    void nest(NML_Node component);

    // Returns attribute `name`'s value, or "" if this node has no such
    // attribute (every NML/LEMS attribute is parsed as a raw string --
    // see xml_node_to_nml_node -- so there is no other type it could be).
    String get_attr(const String &name) const;
};

// Generic declaration-tag extractor shared by nml.cpp's extract_parameters/extract_properties/...
// (ticket #2 [A3]) and resolve.cpp's extract_fixed_decls (ticket #49 [A4]) -- every one of those was
// hand-written as "for each of one or more tag names, find the matching direct children and push_back
// a DeclType built from each match's attributes," differing only in which tags and which attributes.
// `build` receives the actual matched child (not just its attributes), so it may still branch on
// `child.tag_name` for the rare case where the same DeclType is built differently depending on which
// tag matched (resolve.cpp's Fixed vs Constant) -- every other caller's `build` just ignores it.
// Iterates `tags` in the order given (not `node`'s own child order), so a caller with more than one
// tag -- e.g. extract_parameters's Parameter-then-Constant -- keeps the same tag-major ordering it
// always had, rather than document order.
template <typename DeclType, typename Builder>
Vector<DeclType> extract_decls(const NML_Node &node, std::initializer_list<const char *> tags, Builder build) {
    Vector<DeclType> result;
    for (const char *tag : tags) {
        for (const auto &child : node.body) {
            if (child.tag_name == tag) result.push_back(build(child));
        }
    }
    return result;
}

// ── ComponentType categories (ticket #2 [A3]; arch doc §3) ──────────────
//
// A declaration-level `<ComponentType>` is classified into exactly one of
// five categories below, each carrying the fields specific to what legally
// nests inside that category (arch §3's per-tag reference). The building
// blocks (ParameterDecl, StateVariableDecl, ...) each mirror one recurring
// LEMS declaration tag and are shared across categories — a synapse's
// Parameter and a cell's Parameter are the same shape; only which lists a
// category carries differs. The generic (non-declaration-level) parse tree
// stays plain NML_Node — this classification only applies to what
// NML_Parser catalogs into `library`.

// Several of the declaration-level tags below are structurally identical --
// just a name plus one more field -- differing only in which LEMS tag they
// mirror and what that tag means once resolved (§3.1). Consolidated into two
// shared shapes (`NamedDimensionDecl`, `NamedTypeRefDecl`), with every
// original name kept as a `using` alias -- so each declaration below is
// still its own distinct, individually-documented identifier; only the
// duplicated struct bodies are gone.

// Shared shape for `Parameter`/`Exposure`/`Requirement` (§3.1): a name plus a
// dimension, nothing more.
struct NamedDimensionDecl {
    String name;
    String dimension;
};

// `Parameter` (§3.1): a named, time-invariant, dimensioned quantity an
// instance carries. Whether it ends up baked or per-neuron is decided later,
// at resolve — not readable from the declaration alone.
using ParameterDecl = NamedDimensionDecl;

// `Property` (§3.1): like Parameter, but settable with a default — the more
// common carrier of genuine per-instance variation (e.g. `weight`).
struct PropertyDecl {
    String name;
    String dimension;
    String default_value;
};

// `Exposure` (§3.1): an observable quantity recording (or a parent
// ComponentType's `select`/`reduce`) may read.
using ExposureDecl = NamedDimensionDecl;

// `EventPort` (§3.1): a spike port; `direction` is "in" or "out".
struct EventPortDecl {
    String name;
    String direction;
};

// Shared shape for `Attachments`/`ComponentReference`/`Link`/`Children`/
// `Child` (§3.1): a name plus a reference to another type's name, nothing
// more.
struct NamedTypeRefDecl {
    String name;
    String type;
};

// `Attachments` (§3.1): a synapse-attachment point on a cell.
using AttachmentDecl = NamedTypeRefDecl;

// `Requirement` (§3.1): a quantity this type needs bound from its enclosing
// scope but does not own itself (near-universally postsynaptic `v`, for a
// synapse's `g * (erev - v)`).
using RequirementDecl = NamedDimensionDecl;

// `ComponentReference`/`Link` (§3.1) — and the closely related
// `InstanceRequirement`/`ComponentRequirement` forms used by
// electrical/graded synapse pairs and by projections — all "a reference to
// another component's type," unified into one shape.
using ComponentReferenceDecl = NamedTypeRefDecl;

// `Children`/`Child` (§3.1): named containment of sub-components (a
// population's `instance` list, a projection's `connection` list, a
// blockingPlasticSynapse's plasticity/block mechanisms, a spikeArray's
// explicit spike-time list, ...).
using ChildrenDecl = NamedTypeRefDecl;

// `Path`/`Text` (§3.1): a structural-path or string parameter — routing
// names like `presynapticPopulation`, `preCellId`, `destination`.
struct PathDecl {
    String name;
};

// `StateVariable` (§3.2): a continuously-evolving per-instance quantity.
struct StateVariableDecl {
    String name;
    String dimension;
    String exposure;
};

// `DerivedVariable` (§3.2): a quantity computed each tick — either an
// expression (`value`) or a `select`/`reduce` aggregation. The expression
// text stays opaque here (arch §1.4 — expression trees are the one thing
// that isn't flattened at this stage).
struct DerivedVariableDecl {
    String name;
    String dimension;
    String exposure;
    String value;
    String select;
    String reduce;
};

// `TimeDerivative` (§3.2): the ODE driving one state variable.
struct TimeDerivativeDecl {
    String variable;
    String value;
};

// `OnCondition` (§3.2): a conditional test (threshold/reset/spike rule). The
// nested `StateAssignment`/`EventOut`/`Transition` actions stay tree-shaped,
// kept as the handler's raw body (same rationale as DerivedVariableDecl).
struct OnConditionDecl {
    String test;
    NML_Node body;
};

// `OnEvent` (§3.2): a spike-arrival handler.
struct OnEventDecl {
    String port;
    NML_Node body;
};

// `Regime` (§3.2): a named dynamics mode (e.g. a refractory variant). The
// mode's own `OnEntry`/`TimeDerivative`/`OnCondition`/`Transition` body
// stays tree-shaped, same rationale as OnConditionDecl/OnEventDecl.
struct RegimeDecl {
    String name;
    String initial;
    NML_Node body;
};

// `OnStart` (§3.2): initial-value assignments (seeds state at INIT, arch
// §1.2 S3). Modeled explicitly (not left in `raw`, ticket #49 [A4]) so it
// merges across an `extends` chain the same way as every other Dynamics-nested
// tag — otherwise an ancestor-only `OnStart` (common in the std lib, e.g.
// `iafTauCell`) is silently lost once a descendant is flattened. The nested
// `StateAssignment`s stay tree-shaped, same rationale as OnConditionDecl.
struct OnStartDecl {
    NML_Node body;
};

// Identity common to every cataloged ComponentType (§1.2 S6, §3.1
// `ComponentType`): its name, what it extends, and the untouched raw parsed
// node — so nothing the categorized fields below don't model explicitly is
// lost (e.g. `Structure`/`MultiInstantiate`/`ChildInstance`).
//
// Provides an explicit (name, extends, raw) constructor so every category
// below can be built by construction rather than default-construct-then-
// assign — NML_Node has a user-provided copy constructor but relies on the
// (deprecated-by-the-standard) implicit copy-assignment operator, and this
// sidesteps ever invoking it.
struct ComponentTypeBase {
    String name;
    String extends;
    NML_Node raw;

    ComponentTypeBase() : raw(NML_Node("ComponentType")) {}
    ComponentTypeBase(String name, String extends, NML_Node raw)
        : name(std::move(name)), extends(std::move(extends)), raw(std::move(raw)) {}
};

// The five categories below still hold real per-category differences (arch
// §3.3), though a good chunk of their fields do overlap (all three Dynamics
// categories carry parameters/state_variables/derived_variables/
// time_derivatives/exposures/event_ports/on_starts; both Structure
// categories carry component_references/parameters/children). They are kept
// as five distinct C++ types in a std::variant here, rather than unified
// into one matrix/index-addressed type, because every downstream consumer
// narrows `ComponentTypeEntry` via std::get<T>/std::holds_alternative<T>
// keyed on exactly these five type identities: nml.cpp's own
// classify_component_type, resolve.cpp's merge_chain/flatten_component_type
// (a template parameterized on one of these five types), cell_lowering.cpp/
// synapse_lowering.cpp/inputs_lowering.cpp/output_recording.cpp/
// model_specification.cpp downstream, and ~20 call sites across
// nml_tests.cpp/resolve_tests.cpp/cell_lowering_tests.cpp. Re-deriving all of
// that against one SoA/index-addressed type is a real architecture change,
// not a mechanical rename -- left as a follow-up rather than attempted
// alongside the same-behavior struct cleanup done in this pass (see also
// ComponentTypeEntry below).

// CellType — Dynamics/point-cell ComponentTypes (arch §3.3 D1): `iafCell`,
// `iafRefCell`, `izhikevich2007Cell`, `adExIaFCell`, GLIF variants, etc.
// Classified by an `extends` chain reaching `baseCell`.
struct CellType : ComponentTypeBase {
    using ComponentTypeBase::ComponentTypeBase;
    Vector<ParameterDecl> parameters;
    Vector<StateVariableDecl> state_variables;
    Vector<DerivedVariableDecl> derived_variables;
    Vector<TimeDerivativeDecl> time_derivatives;
    Vector<OnConditionDecl> on_conditions;
    Vector<OnStartDecl> on_starts;
    Vector<ExposureDecl> exposures;
    Vector<EventPortDecl> event_ports;
    Vector<RegimeDecl> regimes;
    Vector<AttachmentDecl> attachments;
};

// SynapseType — Dynamics/synapse ComponentTypes (arch §3.3 D3):
// `expOneSynapse`, `alphaCurrentSynapse`, `blockingPlasticSynapse` (NMDA),
// `doubleSynapse`, etc. Classified by an `extends` chain reaching
// `baseSynapse`.
struct SynapseType : ComponentTypeBase {
    using ComponentTypeBase::ComponentTypeBase;
    Vector<ParameterDecl> parameters;
    Vector<PropertyDecl> properties;
    Vector<StateVariableDecl> state_variables;
    Vector<DerivedVariableDecl> derived_variables;
    Vector<TimeDerivativeDecl> time_derivatives;
    Vector<RequirementDecl> requirements;
    Vector<ExposureDecl> exposures;
    Vector<EventPortDecl> event_ports;
    Vector<OnEventDecl> on_events;
    Vector<OnStartDecl> on_starts;
    Vector<ComponentReferenceDecl> component_references;
    Vector<ChildrenDecl> children;
};

// InputsType (a.k.a. GeneratorType) — Dynamics/inputs ComponentTypes (arch
// §3.3 D4): `pulseGenerator`, `sineGenerator`, `spikeGeneratorPoisson`,
// `poissonFiringSynapse`, `spikeArray`, etc. Classified by an `extends`
// chain reaching `basePointCurrent` or `baseSpikeSource`.
struct InputsType : ComponentTypeBase {
    using ComponentTypeBase::ComponentTypeBase;
    Vector<ParameterDecl> parameters;
    Vector<PropertyDecl> properties;
    Vector<StateVariableDecl> state_variables;
    Vector<DerivedVariableDecl> derived_variables;
    Vector<TimeDerivativeDecl> time_derivatives;
    Vector<OnConditionDecl> on_conditions;
    Vector<OnEventDecl> on_events;
    Vector<OnStartDecl> on_starts;
    Vector<ExposureDecl> exposures;
    Vector<EventPortDecl> event_ports;
    Vector<ChildrenDecl> children;
    Vector<ComponentReferenceDecl> component_references;
};
using GeneratorType = InputsType;

// PopulationType — Structures/population ComponentTypes (arch §3.3 ST2):
// `population`, `populationList`, etc. No `<Dynamics>` (Structures, §1.1) —
// instead the instantiated cell/synapse-type reference, the size (or, for
// `populationList`-shaped types, an explicit instance list), and metadata.
struct PopulationType : ComponentTypeBase {
    using ComponentTypeBase::ComponentTypeBase;
    Vector<ComponentReferenceDecl> component_references; // the instantiated type (`component`)
    Vector<ParameterDecl> parameters;                     // `size`, ...
    Vector<ChildrenDecl> children;                         // `instance` list (populationList), notes/annotation/property
};

// ProjectType (a.k.a. ConnectionType) — Structures/projection ComponentTypes
// (arch §3.3 ST3): `projection`, `connection`/`connectionWD`,
// `electricalProjection`, `continuousProjection`, etc. No `<Dynamics>` —
// just the routing declarations wiring one population to another through a
// synapse. Covers both `projection`-shaped (pre/post population + a list of
// connections) and `connection`-shaped (pre/post cell path + weight/delay/
// destination) declarations; Phase 1 doesn't need them split further.
struct ProjectType : ComponentTypeBase {
    using ComponentTypeBase::ComponentTypeBase;
    Vector<ComponentReferenceDecl> component_references; // the routed synapse type, gap-junction peer, ...
    Vector<ParameterDecl> parameters;                     // `weight`, `delay`, ...
    Vector<PathDecl> path_fields;                         // `presynapticPopulation`, `preCellId`, `destination`, ...
    Vector<ChildrenDecl> children;                         // `connections`/`connectionsWD` lists
};
using ConnectionType = ProjectType;

// A cataloged ComponentType is exactly one of the five categories above, or
// (for anything the arch doc's §3.3 buckets don't cover — ion channels,
// morphology, biophysical properties, Phase 3) left as the bare identity
// with its raw node still intact. Stays a variant of five distinct types
// rather than one unified, index-addressed type -- see the note above
// CellType for why.
using ComponentTypeEntry = std::variant<ComponentTypeBase, CellType, SynapseType, InputsType, PopulationType, ProjectType>;

#ifndef SPIKECOREC_NML_STD_LIB_DIR
#define SPIKECOREC_NML_STD_LIB_DIR ""
#endif

#ifndef SPIKECOREC_NML_SCHEMA_PATH
#define SPIKECOREC_NML_SCHEMA_PATH ""
#endif

struct NML_Parser {
    UnorderedMap<String, ComponentTypeEntry> library;
    NML_Node *root_node = nullptr;

    const String STANDARD_LIBRARY_PATH = SPIKECOREC_NML_STD_LIB_DIR;
    const String NML_SCHEMA_PATH = SPIKECOREC_NML_SCHEMA_PATH;

    // Every element/line/message libxml2's XSD validator reported for the most recent
    // validate_against_schema() call, joined into one string (empty if that call found no errors,
    // or hasn't been made yet). libxml2's default error handler only prints validation errors to
    // stderr (invisible to both a caller catching ingest_file's exception and this codebase's own
    // logger) -- validate_against_schema captures them here instead, so ingest_file's thrown
    // exception can name exactly which element/line failed and why (ticket #60 [X1]).
    String last_schema_validation_errors;

    NML_Parser() {};

    bool validate_against_schema(const String &nml_file_path);

    void parse(const String &nml_input_file);

    void ingest_file(const String &nml_file_path, bool run_schema_validation);

    NML_Node xml_node_to_nml_node(xmlNodePtr node);

    bool load_standard_library();

    ComponentTypeEntry &get_type_by_name(const String &name);

private:
    // Raw `<ComponentType>` nodes cataloged so far, keyed by name. Classifying
    // is deliberately deferred (see classify_all_cataloged_types): a single
    // eager classification pass per node, run as each node is cataloged,
    // would make classification order-dependent on the unspecified order
    // std::filesystem::directory_iterator visits std-lib files in (and on
    // physical declaration order for any intra-file forward reference) —
    // an `extends` chain that hops through a not-yet-cataloged intermediate
    // parent would silently misclassify as the unclassified fallback. Instead
    // every raw node is ingested first, then classify_all_cataloged_types()
    // classifies the whole batch at once, so every chain walk always sees
    // every parent cataloged so far, regardless of ingestion order. Cleared
    // once parse() finishes (see there); not needed once classification for
    // that whole run is done, so it doesn't just double the memory `library`
    // already holds via each entry's own `.raw` copy.
    UnorderedMap<String, NML_Node> raw_component_types;

    // Records one parsed `<ComponentType>` node's raw form into
    // `raw_component_types`, unless a type of that name is already cataloged
    // (first-cataloged-wins, matching the original flat-library behavior).
    // Does not classify — see classify_all_cataloged_types. `source_path` is
    // only used for the warning/error log messages. Returns false iff the
    // node's `name` attribute is missing/non-string (a malformed
    // ComponentType) — a duplicate name is not a failure.
    bool catalog_raw_component_type(const NML_Node &component_type_node, const String &source_path);

    // Classifies every node in `raw_component_types` not yet in `library`
    // (additive — never clears/rebuilds `library`, so an already-classified
    // entry's address stays stable across repeated calls). Safe to call
    // repeatedly as more raw types are cataloged (load_standard_library()
    // calls it once its whole directory scan is done; parse() calls it again
    // once ingest_file()'s whole include graph is done) — each call sees
    // every raw type cataloged so far, so a chain walk never gives up on a
    // parent merely because it hasn't been reached yet.
    void classify_all_cataloged_types();

    // Determines which of the five categories `node` belongs to by walking
    // its `extends` chain (through `raw_component_types`) up to one of the
    // well-known anchor base types the arch doc's §3.3 classification
    // buckets are grounded on (`baseCell`, `baseSynapse`,
    // `basePointCurrent`/`baseSpikeSource`, `basePopulation`/`population`/
    // `populationList`, `projection`/`connection`/...). A type whose chain
    // reaches none of them is left unclassified (bare ComponentTypeBase).
    ComponentTypeEntry classify_component_type(const NML_Node &node);
};

}
