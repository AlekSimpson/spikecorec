#pragma once

#include <libxml/parser.h>
#include <libxml/tree.h>
#include <variant>

#include "spikecorec/core/types.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {
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
};

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

// `Parameter` (§3.1): a named, time-invariant, dimensioned quantity an
// instance carries. Whether it ends up baked or per-neuron is decided later,
// at resolve — not readable from the declaration alone.
struct ParameterDecl {
    String name;
    String dimension;
};

// `Property` (§3.1): like Parameter, but settable with a default — the more
// common carrier of genuine per-instance variation (e.g. `weight`).
struct PropertyDecl {
    String name;
    String dimension;
    String default_value;
};

// `Exposure` (§3.1): an observable quantity recording (or a parent
// ComponentType's `select`/`reduce`) may read.
struct ExposureDecl {
    String name;
    String dimension;
};

// `EventPort` (§3.1): a spike port; `direction` is "in" or "out".
struct EventPortDecl {
    String name;
    String direction;
};

// `Attachments` (§3.1): a synapse-attachment point on a cell.
struct AttachmentDecl {
    String name;
    String type;
};

// `Requirement` (§3.1): a quantity this type needs bound from its enclosing
// scope but does not own itself (near-universally postsynaptic `v`, for a
// synapse's `g * (erev - v)`).
struct RequirementDecl {
    String name;
    String dimension;
};

// `ComponentReference`/`Link` (§3.1) — and the closely related
// `InstanceRequirement`/`ComponentRequirement` forms used by
// electrical/graded synapse pairs and by projections — all "a reference to
// another component's type," unified into one shape.
struct ComponentReferenceDecl {
    String name;
    String type;
};

// `Children`/`Child` (§3.1): named containment of sub-components (a
// population's `instance` list, a projection's `connection` list, a
// blockingPlasticSynapse's plasticity/block mechanisms, a spikeArray's
// explicit spike-time list, ...).
struct ChildrenDecl {
    String name;
    String type;
};

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

// Identity common to every cataloged ComponentType (§1.2 S6, §3.1
// `ComponentType`): its name, what it extends, and the untouched raw parsed
// node — so nothing the categorized fields below don't model explicitly is
// lost (e.g. `Structure`/`MultiInstantiate`/`ChildInstance`, `OnStart`).
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
// with its raw node still intact.
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
