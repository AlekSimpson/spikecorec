// FORMERLY: nml.h

#pragma once

#include <libxml/parser.h>
#include <libxml/tree.h>

#include "dynamics.h"

#include "units.h"
#include "types.h"
#include "recording.h"

#include "node.h"
#include "components.h"
#include "utilities.h"

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
    struct NML_Simulation_Context {
        String simulation_component_id;
        String target_network_id;
        f64 step_dt = 0.0;
        f64 simulation_duration = 0.0;
        s64 total_tick_count = 0;
        s64 total_neuron_count = 0;
        Optional<u64> random_seed;
        s64 total_edge_count = 0;
        s64 maximum_edge_delay = 0;
    
        // keys are indices in cell_instances which map to indices in the engine cell memory where they start
        Unordered_Map<String, s64> population_base_indices;
    
        Unordered_Map<String, Real> global_constants;
    
        Vector<NML_Component_Instance> cell_instances;
        Vector<NML_Component_Instance> synapse_instances;

        Adjacency_List network_data;
    
        Vector<Simulation_Input_Config> input_profiles;
        Vector<Recording_Config> recording_profiles;
    
        NML_Simulation_Context(): simulation_component_id(""), target_network_id(""), network_data(0, 0) {}
    };

    NML_Simulation_Context simulation;

    Unordered_Map<String, units::Unit_Definition> model_units;
    Unordered_Map<String, Real> model_constants;
    Unordered_Map<String, NML_Component_Type> component_types;
    Unordered_Map<String, NML_Component_Instance> component_instances;

    Vector<NML_Node *> document_roots;
    Vector<String> document_filepaths;

    String target_component_id;

    Node<NML_Tag> *main_document_root = nullptr;

    const String STANDARD_LIBRARY_PATH = SPIKECOREC_NML_STD_LIB_DIR;
    const String NML_SCHEMA_PATH = SPIKECOREC_NML_SCHEMA_PATH;

    String last_schema_validation_errors;

    NML_Context() {};
    ~NML_Context();

    // Copies and moves would share the owned document trees.
    NML_Context(const NML_Context &other) = delete;
    NML_Context &operator=(const NML_Context &other) = delete;
    NML_Context(NML_Context &&other) = delete;
    NML_Context &operator=(NML_Context &&other) = delete;

    void reset();
    void parse(const String &main_filepath);
    NML_Node *parse_neuroml_file(const String &filepath, Vector<String> &files_to_parse);
    NML_Node *parse_neuroml(const String &filepath);
    void parse_component_types();
    NML_Component_Type *create_component_type(const NML_Node *type_node, const String &source_filepath);
    void parse_component_instances();
    NML_Component_Instance *create_component_instance(
            const NML_Component_Type *component_type, const NML_Node *instance_node,
            NML_Component_Instance *parent_instance);
    void parse_simulation_details(NML_Node *lems_root); // previously extract_neuroml_context

    const NML_Component_Instance *find_instance(const String &instance_id) const;
    // Whether the instance's ComponentType is type_name or extends it.
    bool is_instance_of(const NML_Component_Instance *instance, const String &type_name) const;

    bool validate_lems_schema(const String &lems_filepath);
    f64 resolve_quantity(const String &value) const;
    xmlNodePtr get_xml_root(const String &filepath);

    s64 get_cell_variable_count(s64 cell_instance_index);
    s64 get_cell_state_size() const;
    s64 get_population_size(const NML_Component_Instance *population) const;

    // The neuron whose cells occupy cell_memory_index (a resolve_path result), or -1.
    s64 neuron_index_of(s64 cell_memory_index) const;
    s64 resolve_path(const String &path, const NML_Component_Instance *current = nullptr) const;
};

}





