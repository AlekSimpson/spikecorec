// FORMERLY: nml.h

#pragma once

#include <libxml/parser.h>
#include <libxml/tree.h>

#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"
#include "spikecorec/core/recording.h"

#include "spikecorec/nml/declarations.h"
#include "spikecorec/nml/node.h"
#include "spikecorec/nml/components.h"
#include "spikecorec/nml/dynamics.h"
#include "spikecorec/nml/utilities.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {
#ifndef SPIKECOREC_NML_STD_LIB_DIR
#define SPIKECOREC_NML_STD_LIB_DIR ""
#endif

#ifndef SPIKECOREC_NML_SCHEMA_PATH
#define SPIKECOREC_NML_SCHEMA_PATH ""
#endif

struct NML_Context {
    String simulation_component_id;
    String target_network_id;
    f64 step_dt = 0.0;
    f64 simulation_duration = 0.0;
    s64 total_tick_count = 0;
    s64 total_neuron_count = 0;
    Optional<u64> random_seed;
    s64 total_edge_count = 0;
    s64 maximum_edge_delay = 0;
    s64 cell_state_length;

    // keys are indices in cell_instances which map to indices in the engine cell memory where they start
    UnorderedMap<s64, s64> population_state_base;

    UnorderedMap<String, Real> global_constants;

    Vector<NML_ComponentInstance> cell_instances;
    Vector<NML_ComponentInstance> synapse_instances;

    AdjacencyList<NML_NetworkEdge> network_data;

    Vector<SimulationInputConfig> input_profiles;
    Vector<RecordingConfig> recording_profiles;

    s64 get_cell_variable_count(s64 cell_instance_index);

    NML_Context(): simulation_component_id(""), target_network_id(""), cell_state_length(0) {}
};

struct NML_Parser {
    UnorderedMap<String, ComponentType> declared_component_types;
    UnorderedMap<String, ComponentInstance> instance_table;

    Set<String> parsed_neuroml_files;

    // <Unit> symbols declared by the parsed documents, consulted before falling back to
    // units::unit_suffix_scale so a model defining its own units resolves correctly.
    UnorderedMap<String, UnitDefinition> custom_declared_units;

    // Document-scope <Constant>s and the <Target component="..."/> selection.
    UnorderedMap<String, Real> model_constants;
    String target_component_id;

    NML_Node main_document_root;

    const String STANDARD_LIBRARY_PATH = SPIKECOREC_NML_STD_LIB_DIR;
    const String NML_SCHEMA_PATH = SPIKECOREC_NML_SCHEMA_PATH;

    String last_schema_validation_errors;

    NML_Parser() {};

    NML_Context extract_neuroml_context(NML_Node *lems_root);

    bool validate_lems_schema(const String &lems_filepath);
    void parse_lems(const String &lems_main_file);
    void ingest_all_documents(const String &root_filepath);

    void resolve_all_component_types();
    const ComponentType &resolve_component_type(const String &type_name,
                                                Vector<String> &resolution_stack);

    f64 resolve_quantity(const String &value) const;

    void extract_all_declarations_nested_in_node(NML_Node *node, DeclarationList &return_value);
    xmlNodePtr get_xml_root(const String &filepath);

    void instantiate_component_type(NML_Node *instance_node, String &parent_instance_id);
    void instantiate_component_type(NML_Node *instance_node);
    void bind_instance_data(NML_Node *instance_node,
                            const ComponentType &component_type,
                            ComponentInstance &instance);
};

}





