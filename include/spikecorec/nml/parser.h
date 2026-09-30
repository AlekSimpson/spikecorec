// FORMERLY: nml.h

#pragma once

#include <libxml/parser.h>
#include <libxml/tree.h>

#include "spikecorec/nml/dynamics.h"

#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"
#include "spikecorec/core/recording.h"

#include "spikecorec/nml/node.h"
#include "spikecorec/nml/components.h"
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
    struct NML_SimulationContext {
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
        UnorderedMap<String, s64> population_base_indices;
    
        UnorderedMap<String, Real> global_constants;
    
        Vector<NML_ComponentInstance> cell_instances;
        Vector<NML_ComponentInstance> synapse_instances;

        AdjacencyList network_data;
    
        Vector<SimulationInputConfig> input_profiles;
        Vector<RecordingConfig> recording_profiles;
    
        NML_SimulationContext(): simulation_component_id(""), target_network_id(""), network_data(0, 0) {}
    };

    NML_SimulationContext simulation;

    UnorderedMap<String, units::UnitDefinition> model_units;
    UnorderedMap<String, Real> model_constants;
    UnorderedMap<String, NML_ComponentType> component_types;
    UnorderedMap<String, NML_ComponentInstance> component_instances;

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
    NML_ComponentType *create_component_type(const NML_Node *type_node, const String &source_filepath);
    void parse_component_instances();
    NML_ComponentInstance *create_component_instance(
            const NML_ComponentType *component_type, const NML_Node *instance_node,
            NML_ComponentInstance *parent_instance);
    void parse_simulation_details(NML_Node *lems_root); // previously extract_neuroml_context

    const NML_ComponentInstance *find_instance(const String &instance_id) const;
    bool is_instance_of(const NML_ComponentInstance *instance, const String &type_name) const;

    bool validate_lems_schema(const String &lems_filepath);
    f64 resolve_quantity(const String &value) const;
    xmlNodePtr get_xml_root(const String &filepath);

    s64 get_cell_variable_count(s64 cell_instance_index);
    s64 resolve_path(const String &path, const NML_ComponentInstance *current = nullptr) const;
};

}





