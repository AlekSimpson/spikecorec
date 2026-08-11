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

    NML_Node(String tag_name) : tag_name(tag_name) {};

    NML_Node(const NML_Node &other) :
        tag_name(other.tag_name), attributes(other.attributes), body(other.body) {};

    NML_Node &operator=(const NML_Node &other) = default;

    bool has_attribute(const String &name) const;
    void add_attribute(const String &name, String value);
    String get_attribute(const String &name) const;
    void nest(NML_Node component);
    NML_Declaration to_declaration() const;
    bool is_declaration_type() const;
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
    String condition;    // for Reset/Emit: the OnCondition test that gates it

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

    void insert(const NML_Declaration &declaration);
};

struct ComponentType {
    DeclarationList declarations;
    String name;
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
};

struct ComponentInstance {
    UnorderedMap<String, String> instance_data;
    Vector<String> structured_instance_data; // list of child instance ids
    String id;
    const ComponentType *component_type;

    // todo: the five constructors (most new structs have this as todo)

    String &get_value(String &key);
    RuntimeCategory get_runtime_category();
};

struct ComponentInstanceVisitorContext {
    Stack<String> id_scope;

    String get_base_scope_id();
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

struct SimulationInputConfig {
    Vector<Vector<s32>> simulation_input_events;
    Vector<s64> input_neuron_indices;
    f64 amplitude;
    f64 rate;
    s64 start_tick;
    s64 end_tick;
    s64 max_delay_time;
    bool continuous_current_injection;
};

struct NML_ParseResult {
    // this is a struct that holds all the info required for the engine constructor
    // to allocate cell memory, configure and initialize the weight matrix,
    // allocate input buffer memory, and compile cell dynamics into kernel code

    UnorderedMap<String, Real> global_constants;

    // Cell state info
    UnorderedMap<String, s32> cell_state_offsets;
    Vector<Vector<Real>> cell_starting_parameters;
    Vector<s64> cell_state_size;

    // Synapse state info
    Vector<Vector<Real>> synapse_starting_parameters;
    Vector<s64> synapse_state_size;

    Set<s64> conductance_based_synapse_types;
    Set<s64> per_edge_synapse_projection_types; // default option is aggregate

    // Per-ComponentType instruction programs, ready for lowering into kernel code.
    // Keyed by component type name.
    UnorderedMap<String, Vector<DynamicsInstruction>> cell_dynamics;
    UnorderedMap<String, Vector<DynamicsInstruction>> synapse_dynamics;

    // Ordered state variable names per cell type; the index in this vector is the
    // variable's slot within that type's SoA state chunk.
    UnorderedMap<String, Vector<String>> cell_state_variable_names;

    // General engine construction data
    s64 total_cell_count = 0;
    s64 cell_type_count = 0;
    s64 synapse_type_count = 0;
    f64 step_dt = 0.0; // (SI value) must be floating point: 0.01ms is 1e-5 s
    f64 simulation_duration = 0.0; // (SI value)
    s64 total_tick_count = 0;
    Optional<u64> random_seed;

    Vector<Vector<s64>> network_adjacency_list;

    // Simulation I/O
    Vector<SimulationInputConfig> input_profiles;
    Vector<RecordingConfig> recording_profiles;
};

struct NML_Parser {
    UnorderedMap<String, ComponentType> declared_component_types;

    UnorderedMap<String, ComponentInstance> instance_table;

    Set<String> parsed_neuroml_files;

    const String STANDARD_LIBRARY_PATH = SPIKECOREC_NML_STD_LIB_DIR;
    const String NML_SCHEMA_PATH = SPIKECOREC_NML_SCHEMA_PATH;

    String last_schema_validation_errors;

    NML_Parser() {};

    bool validate_against_schema(const String &nml_file_path);

    NML_ParseResult parse_lems(const String &lems_main_file);

    void parse_neuroml(const String &nml_file_path);

    void load_component_types_in_file(NML_Node *file_root);

    void extract_all_declarations_nested_in_node(NML_Node *node, DeclarationList &return_value);

    NML_Node xml_node_to_nml_node(xmlNodePtr node);

    xmlNodePtr get_xml_root(const String &filepath);

    void instantiate(NML_Node *instance_node, String &parent_instance_id);
    void instantiate(NML_Node *instance_node);

    void bind_instance_data(NML_Node *instance_node,
                            const ComponentType &component_type,
                            ComponentInstance &instance);

    // Takes the LEMS document root so the export can read the things that only exist at
    // document scope: <Constant>s, and the Simulation instance carrying dt/length/seed
    // and the recording selections.
    NML_ParseResult export_model_details_to_engine(NML_Node *lems_root);
};

}
