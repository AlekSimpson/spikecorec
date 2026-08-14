#pragma once

#include <libxml/parser.h>
#include <libxml/tree.h>

#include "spikecorec/core/types.h"
#include "spikecorec/core/recording.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {


// the five core primitives that spikecore engine translates and categorizes all neuroml components into
enum class RuntimeCategory {
    Cell, // instances relating to cell dynamics and state
    Synapse, // instances relating to synapse dynamics and state
    InputScheme, // instances relating to network input schemes
    Graph, // network instances specifies graph data
    General
};


struct NML_Declaration;

struct NML_Node {
    String tag_name;

    UnorderedMap<String, String> attributes;
    Vector<NML_Node> body;

    // Default-constructible and movable because the parser holds nodes by value: the
    // ComponentType catalogue keeps every declaration node alive from ingest until
    // resolution, long after the document that produced it goes out of scope.
    NML_Node() = default;
    NML_Node(String tag_name) : tag_name(std::move(tag_name)) {};

    NML_Node(const NML_Node &other) = default;
    NML_Node &operator=(const NML_Node &other) = default;

    NML_Node(NML_Node &&other) noexcept = default;
    NML_Node &operator=(NML_Node &&other) noexcept = default;

    bool has_attribute(const String &name) const;
    void add_attribute(const String &name, String value);
    String get_attribute(const String &name) const;
    void nest(NML_Node component);
    NML_Declaration to_declaration() const;
    bool is_declaration_type() const;

    // First direct child carrying `child_tag_name`, or nullptr.
    const NML_Node *find_child(const String &child_tag_name) const;
};


enum class NML_DeclarationType {
    NamedDimension,
    Parameter,
    DerivedParameter,
    Constant,
    Requirement,
    Exposure,
    Property,
    Fixed,
    Text,
    EventPort,
    NamedTypeReference,
    Attachment,
    ComponentReference,
    Link,
    Children,
    Child,
    Path,
    StateVariable,
    DerivedVariable,
    ConditionalDerivedVariable,
    Case,
    TimeDerivative,
    OnCondition,
    OnEvent,
    OnStart,
    OnEntry,
    StateAssignment,
    EventOut,
    Transition,
    Regime,

    NOT_A_TYPE
};

// Keyed by the literal LEMS element name, so a tag_name read straight off the parsed
// document resolves without translation.
inline NML_DeclarationType string_to_declaration_type(const String &value) {
    static const UnorderedMap<String, NML_DeclarationType> mapping = {
        {"Parameter", NML_DeclarationType::Parameter},
        {"DerivedParameter", NML_DeclarationType::DerivedParameter},
        {"Constant", NML_DeclarationType::Constant},
        {"Requirement", NML_DeclarationType::Requirement},
        {"Exposure", NML_DeclarationType::Exposure},
        {"Property", NML_DeclarationType::Property},
        {"Fixed", NML_DeclarationType::Fixed},
        {"Text", NML_DeclarationType::Text},
        {"EventPort", NML_DeclarationType::EventPort},
        {"Attachments", NML_DeclarationType::Attachment},
        {"ComponentReference", NML_DeclarationType::ComponentReference},
        {"Link", NML_DeclarationType::Link},
        {"Children", NML_DeclarationType::Children},
        {"Child", NML_DeclarationType::Child},
        {"Path", NML_DeclarationType::Path},
        {"StateVariable", NML_DeclarationType::StateVariable},
        {"DerivedVariable", NML_DeclarationType::DerivedVariable},
        {"ConditionalDerivedVariable", NML_DeclarationType::ConditionalDerivedVariable},
        {"Case", NML_DeclarationType::Case},
        {"TimeDerivative", NML_DeclarationType::TimeDerivative},
        {"OnCondition", NML_DeclarationType::OnCondition},
        {"OnEvent", NML_DeclarationType::OnEvent},
        {"OnStart", NML_DeclarationType::OnStart},
        {"OnEntry", NML_DeclarationType::OnEntry},
        {"StateAssignment", NML_DeclarationType::StateAssignment},
        {"EventOut", NML_DeclarationType::EventOut},
        {"Transition", NML_DeclarationType::Transition},
        {"Regime", NML_DeclarationType::Regime},
    };

    auto entry = mapping.find(value);
    if (entry == mapping.end()) return NML_DeclarationType::NOT_A_TYPE;

    return entry->second;
};

inline const Set<NML_DeclarationType> DYNAMICS_TYPES = {
    NML_DeclarationType::StateVariable,
    NML_DeclarationType::DerivedVariable,
    NML_DeclarationType::ConditionalDerivedVariable,
    NML_DeclarationType::Case,
    NML_DeclarationType::TimeDerivative,
    NML_DeclarationType::OnCondition,
    NML_DeclarationType::OnEvent,
    NML_DeclarationType::OnStart,
    NML_DeclarationType::OnEntry,
    NML_DeclarationType::StateAssignment,
    NML_DeclarationType::EventOut,
    NML_DeclarationType::Transition,
    NML_DeclarationType::Regime
};

bool is_declaration_type(s32 raw_type_value);
bool is_dynamic_type(NML_DeclarationType type);

struct NML_Declaration {
    NML_DeclarationType tag_type;

    UnorderedMap<String, String> datavalues;

    // Nested declarations, in source order. Dynamics are a tree: a Regime owns its
    // TimeDerivatives and OnConditions, an OnCondition owns the StateAssignments and
    // EventOuts it fires. Flattening loses which assignment belongs to which condition,
    // so children stay attached here rather than going into the sorted DeclarationList.
    Vector<NML_Declaration> children;

    NML_Declaration(NML_DeclarationType type) : tag_type(type) {};
    NML_Declaration(NML_DeclarationType type, UnorderedMap<String, String> values)
        : tag_type(type), datavalues(std::move(values)) {};

    void add_values(const UnorderedMap<String, String> &data) { datavalues = data; }

    String get_value(const String &key) const;
    bool has_value(const String &key) const;

    // get_value throws on a missing key. This is the "absent is legal" form, used
    // wherever an attribute is genuinely optional (a Parameter with no Fixed value, a
    // connection with no explicit weight).
    String value_or(const String &key, const String &fallback = "") const;

    // Identity of this declaration within its ComponentType, as "<namespace>:<name>".
    // Returns "" for declarations that accumulate rather than override (OnCondition,
    // OnStart, ...). See docs/nml_general_notes.md "What can share a name": seven tags
    // feed one `variables` namespace, so Parameter "tau" and StateVariable "tau" collide,
    // and a subtype redeclaring either overrides the ancestor's.
    String namespace_key() const;
};

// Which tick stage an instruction belongs to (arch doc section 2's 9-stage scaffold).
enum class DynamicsStage {
    Initialize,  // OnStart -- runs once at init, not per tick
    Integrate,   // TimeDerivative / DerivedVariable / ConditionalDerivedVariable
    Detect,      // OnCondition test
    Reset,       // StateAssignment fired by a condition or an event
    Emit,        // EventOut
    Arrival,     // OnEvent -- incoming spike handling
    RegimeEntry  // Regime / Transition / OnEntry bookkeeping
};

// One lowerable operation pulled out of a ComponentType's Dynamics. The expression is
// kept as verbatim NML syntax -- tokenising it belongs to the codegen back-end, not here.
struct DynamicsInstruction {
    DynamicsStage stage;
    NML_DeclarationType source_tag;

    String target;       // variable written, or exposure / port name
    String expression;   // value= / test= source expression, verbatim
    String regime_name;  // owning Regime, empty when the instruction is regime-free

    // Only meaningful on the Regime declaration itself: whether it carried
    // initial="true". A state machine has to start somewhere, and which regime that is
    // cannot be recovered from the instruction stream otherwise -- declaration order is
    // not it, since LEMS lets any regime be the initial one.
    bool is_initial_regime = false;
    // What gates this instruction: the OnCondition test for a Reset/Emit, or the port
    // name for anything an OnEvent handles. Empty when nothing gates it -- which is the
    // case for OnStart and OnEntry bodies, located by `stage` instead.
    String condition;

    DynamicsInstruction(DynamicsStage stage, NML_DeclarationType source_tag)
        : stage(stage), source_tag(source_tag) {};
};

struct DeclarationList {
    Vector<NML_Declaration> declarations;

    // How many declarations of each type the list holds; maintained by insert(), which
    // keeps `declarations` grouped by tag_type.
    UnorderedMap<NML_DeclarationType, usize> ordered_declaration_type_counts;

    DeclarationList() = default;
    ~DeclarationList() = default;

    DeclarationList(const DeclarationList &other) = default;
    DeclarationList &operator=(const DeclarationList &other) = default;

    DeclarationList(DeclarationList &&other) noexcept = default;
    DeclarationList &operator=(DeclarationList &&other) noexcept = default;

    const Vector<NML_Declaration> &for_all() const { return declarations; }

    // Appends, keeping `declarations` grouped by tag_type and preserving source order
    // within each group.
    void insert(const NML_Declaration &declaration);

    // Inheritance overlay: replaces the declaration sharing `declaration`'s
    // namespace_key() if one is present, otherwise inserts. This is what makes a subtype
    // redeclaring an ancestor's Parameter an override rather than a duplicate.
    void overlay(const NML_Declaration &declaration);

    // First declaration of `type` named `name`, or nullptr.
    const NML_Declaration *find(NML_DeclarationType type, const String &name) const;

    // Every declaration of `type`, in source order.
    Vector<const NML_Declaration *> find_all(NML_DeclarationType type) const;
};

struct ComponentType {
    DeclarationList declarations;
    String name;
    String extends; // parent type name, empty for a root type
    RuntimeCategory runtime_category = RuntimeCategory::General;

    ComponentType() = default;
    ~ComponentType() = default;

    ComponentType(const ComponentType &other) = default;
    ComponentType &operator=(const ComponentType &other) = default;

    ComponentType(ComponentType &&other) noexcept = default;
    ComponentType &operator=(ComponentType &&other) noexcept = default;

    ComponentType(String name, RuntimeCategory runtime_category = RuntimeCategory::General)
        : name(std::move(name)),
          runtime_category(runtime_category) {};
    ComponentType(String name, DeclarationList list,
                  RuntimeCategory runtime_category = RuntimeCategory::General)
        : declarations(std::move(list)),
          name(std::move(name)),
          runtime_category(runtime_category) {};

    // Parameter and Property names in declaration order. This is the column order of
    // NML_ParseResult::cell_starting_parameters / synapse_starting_parameters. Text is
    // excluded (it carries metadata strings, not scalars) and DerivedParameter too (it is
    // computed from other parameters rather than supplied by an instance).
    Vector<String> ordered_parameter_names() const;

    // StateVariable names in declaration order; the index is the variable's slot within
    // this type's state chunk.
    Vector<String> ordered_state_variable_names() const;
};

struct ComponentInstance {
    UnorderedMap<String, String> instance_data;
    Vector<String> structured_instance_data; // list of child instance ids, in source order
    String id;
    String component_type_name;
    String parent_instance_id; // empty for a document-scope instance
    const ComponentType *component_type = nullptr;

    ComponentInstance() = default;
    ~ComponentInstance() = default;

    ComponentInstance(const ComponentInstance &other) = default;
    ComponentInstance &operator=(const ComponentInstance &other) = default;

    ComponentInstance(ComponentInstance &&other) noexcept = default;
    ComponentInstance &operator=(ComponentInstance &&other) noexcept = default;

    String &get_value(String &key);
    RuntimeCategory get_runtime_category();

    bool has_value(const String &key) const;
    String value_or(const String &key, const String &fallback = "") const;
};

#ifndef SPIKECOREC_NML_STD_LIB_DIR
#define SPIKECOREC_NML_STD_LIB_DIR ""
#endif

#ifndef SPIKECOREC_NML_SCHEMA_PATH
#define SPIKECOREC_NML_SCHEMA_PATH ""
#endif

union Real {
    s64 int64;
    f64 float64;
};

// A <Unit> declaration: the affine map from a written magnitude onto its dimension's SI
// unit. LEMS defines it as `si = raw * scale * 10^power + offset`; `power` is folded into
// `scale` at ingest, and `offset` is non-zero only for degC.
struct UnitDefinition {
    f64 scale = 1.0;
    f64 offset = 0.0;
};

// A ComponentType classified as a cell: everything that is a property of the TYPE rather
// than of a particular instance of it. Its index in NML_ParseResult::cell_types is the
// cell type index used everywhere else.
struct CellTypeSpecification {
    String name;

    // Where this type's state variables begin inside the single engine-allocated
    // cell-state buffer (the successor to v1's `membrane_potentials`): the running sum of
    // the preceding types' state variable counts.
    s32 state_offset = 0;

    // In declaration order; the index is the variable's slot within this type's state
    // chunk, and the count is the type's state size.
    Vector<String> state_variable_names;

    // Column order of a prototype's starting_parameters.
    Vector<String> parameter_names;

    // Stage-tagged instruction program, ready for lowering into kernel code.
    Vector<DynamicsInstruction> dynamics;
};

struct SynapseTypeSpecification {
    String name;
    Vector<String> state_variable_names;
    Vector<String> parameter_names;
    Vector<DynamicsInstruction> dynamics;

    // Carries a driving force g*(erev - v), so it needs the postsynaptic voltage.
    bool is_conductance_based = false;

    // State does not superpose across converging edges, so it cannot be aggregated into
    // a single per-target accumulator. The default is aggregate.
    bool requires_per_edge_state = false;
};

// One parameterisation of a ComponentType. A population names an instance
// ("component=cellA"), not a type, and two instances of one type may differ, so the
// values live here rather than on the type.
struct ComponentPrototype {
    String instance_id;
    s64 type_index = -1; // into cell_types or synapse_types
    Vector<Real> starting_parameters; // ordered by the type's parameter_names
};

// One outgoing edge. Grouped rather than spread across parallel arrays so a source's
// edge list carries its full per-edge record in one place.
struct NetworkEdge {
    s64 target_neuron_index = -1;
    s64 synapse_type_index = -1;      // -1 when the projection names no synapse
    s64 synapse_prototype_index = -1; // into synapse_prototypes
    f64 weight = 1.0;                 // 1.0 when the connection carries no weight
    s64 delay_tick_count = 0;         // 0 when the connection carries no delay
};

// One neuron. Its index is the global neuron index every path resolves to.
struct Neuron {
    s64 cell_type_index = -1;  // the boundary the master kernel dispatches on
    s64 prototype_index = -1;  // into cell_prototypes
    s64 population_index = -1; // into populations
    Vector<NetworkEdge> outgoing_edges;
};

// Where a population's neurons sit in the global neuron numbering. Populations occupy
// contiguous ranges assigned in document order.
struct PopulationLayout {
    String population_name;
    String instance_id;
    String component_instance_id; // the prototype this population instantiates
    s64 first_neuron_index = 0;
    s64 neuron_count = 0;
    s64 cell_type_index = -1;
    s64 prototype_index = -1;
};

// One target of an input, with everything that varies per target.
struct InputTarget {
    s64 neuron_index = -1;
    f64 weight = 1.0;        // from <inputW weight="..."/>, 1.0 when unweighted
    Vector<s32> event_ticks; // spike train, empty for a continuous injector
};

struct SimulationInputConfig {
    // Which input component produced this profile, so a caller can tell a pulseGenerator
    // from a spikeArray without re-reading the document.
    String input_component_id;
    String input_component_type_name;

    Vector<InputTarget> targets;

    f64 amplitude = 0.0;
    f64 rate = 0.0;

    // <delay> and <delay> + <duration>, in ticks. A continuous injector is live for
    // start_tick <= tick < end_tick; end_tick == 0 means the component carried no
    // duration, which LEMS reads as "runs to the end of the simulation".
    s64 start_tick = 0;
    s64 end_tick = 0;

    bool continuous_current_injection = false;
};

struct NML_ParseResult {
    // Everything the engine constructor needs to allocate cell and synapse state, build
    // the weight matrix, wire inputs, and compile dynamics into kernel code.
    //
    // Entities are stored as arrays of structures: one entry per cell type, per
    // prototype, per population, per neuron. Anything an index needs to reach lives on
    // the entry itself, so there are no parallel arrays to keep aligned and no name->index
    // maps to consult. Where a name lookup is genuinely needed, scan -- these vectors hold
    // a handful of entries each.

    String simulation_component_id;
    String target_network_id;
    f64 step_dt = 0.0;             // (SI value) must be floating point: 0.01ms is 1e-5 s
    f64 simulation_duration = 0.0; // (SI value)
    s64 total_tick_count = 0;
    Optional<u64> random_seed;

    // Document-scope <Constant>s, plus each ComponentType's own namespaced by its type.
    UnorderedMap<String, Real> global_constants;

    // An entry's position is its type index; a prototype names the type it parameterises.
    Vector<CellTypeSpecification> cell_types;
    Vector<SynapseTypeSpecification> synapse_types;
    Vector<ComponentPrototype> cell_prototypes;
    Vector<ComponentPrototype> synapse_prototypes;

    // Populations occupy contiguous neuron ranges in document order; neurons[k] carries
    // its own type, prototype, population and outgoing edges.
    Vector<PopulationLayout> populations;
    Vector<Neuron> neurons;

    Vector<SimulationInputConfig> input_profiles;
    Vector<RecordingConfig> recording_profiles;
};

// Projection of neurons' outgoing edges into the plain source -> targets form WeightMatrix
// consumes. Derived on demand rather than stored, so there is one edge list to keep
// correct rather than two that can disagree.
Vector<Vector<s64>> build_adjacency_list(const NML_ParseResult &parse_result);

struct NML_Parser {
    UnorderedMap<String, ComponentType> declared_component_types;

    // Pure id -> instance lookup. Every ordering the export needs comes from the document
    // tree instead: document order from document_instance_nodes, child order from each
    // instance's structured_instance_data.
    UnorderedMap<String, ComponentInstance> instance_table;

    Set<String> parsed_neuroml_files;

    // Catalog phase. `extends` resolves by name, and forward + cross-file references are
    // both normal (docs/nml_general_notes.md), so every ComponentType across every
    // included file is collected here first and resolved only once ingest completes.
    UnorderedMap<String, NML_Node> component_type_catalogue;
    UnorderedMap<String, String> component_type_source_files;

    // Instance nodes in document order, stashed during ingest. They cannot be
    // instantiated inline: an instance may name a ComponentType declared in a file that
    // has not been read yet.
    Vector<NML_Node> document_instance_nodes;

    // <Unit> symbols declared by the parsed documents, consulted before falling back to
    // units::unit_suffix_scale so a model defining its own units resolves correctly.
    UnorderedMap<String, UnitDefinition> declared_units;

    // Document-scope <Constant>s and the <Target component="..."/> selection.
    UnorderedMap<String, Real> document_constants;
    String target_component_id;

    NML_Node main_document_root;

    const String STANDARD_LIBRARY_PATH = SPIKECOREC_NML_STD_LIB_DIR;
    const String NML_SCHEMA_PATH = SPIKECOREC_NML_SCHEMA_PATH;

    String last_schema_validation_errors;

    NML_Parser() {};

    bool validate_against_schema(const String &nml_file_path);

    NML_ParseResult parse_lems(const String &lems_main_file);

    void parse_neuroml(const String &nml_file_path);

    // Reads one document and folds its ComponentTypes, Units, Constants and instance
    // nodes into the parser. Recurses through <Include file="..."/> (LEMS) and
    // <include href="..."/> (NeuroML), resolving both against the including file's own
    // directory and deduping by canonical path.
    void ingest_document(const String &file_path);

    // Two-phase resolution, run once every document is ingested: walks each catalogued
    // type's `extends` chain, overlays ancestors' declarations under the type's own
    // (most-derived wins), applies <Fixed>, and assigns the runtime category.
    void resolve_all_component_types();
    const ComponentType &resolve_component_type(const String &type_name,
                                                Vector<String> &resolution_stack);

    // Runtime category from the `extends` chain, walking upward from `type_name` to the
    // first ancestor that is a known category root.
    RuntimeCategory category_for_component_type(const String &type_name) const;

    // "-60mV" -> -0.06, resolved against the <Unit>s this parser ingested and falling
    // back to units::unit_suffix_scale for symbols no document declared.
    f64 resolve_quantity(const String &value) const;

    void load_component_types_in_file(NML_Node *file_root);

    void extract_all_declarations_nested_in_node(NML_Node *node, DeclarationList &return_value);

    NML_Node xml_node_to_nml_node(xmlNodePtr node);

    xmlNodePtr get_xml_root(const String &filepath);

    void instantiate(NML_Node *instance_node, String &parent_instance_id);
    void instantiate(NML_Node *instance_node);

    void bind_instance_data(NML_Node *instance_node,
                            const ComponentType &component_type,
                            ComponentInstance &instance);

    NML_ParseResult export_model_details_to_engine(NML_Node *lems_root);
};

// "-60mV" -> -0.06 using only the built-in unit table. NML_Parser::resolve_quantity is
// the document-aware form, and is what the parser itself uses.
f64 parse_quantity(const String &value);

// Splits "-60mV" into its numeric magnitude and unit suffix, so parse_quantity and
// resolve_quantity share one number scanner.
Pair<f64, String> split_quantity(const String &value);

}
