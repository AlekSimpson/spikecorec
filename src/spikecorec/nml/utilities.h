#pragma once

#include "units.h"
#include "types.h"
#include "recording.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

Output_File_Format output_format_for_filename(const String &filename);

union Real {
    s64 int64;
    f64 float64;
};

struct Input_Target {
    s64 neuron_index = -1;   // the target's cell-memory index, from resolve_path
    f64 weight = 1.0;        // from <inputW weight="..."/>, 1.0 when unweighted
    String destination;      // the cell's attachment it goes to; "" means "synapses"
};

// One input component and its targets. Each target runs its own copy of the input's dynamics
// in the kernel (see Codegen::flatten_input_tree).
struct Simulation_Input_Config {
    String input_component_id;
    Vector<Input_Target> targets;
};

struct NML_Network_Edge {
    String component_id;
    f32 weight;
    s64 delay_ticks;
    s64 parent;
    s64 child;
    String destination;  // the target's attachment the synapse goes to; "" means "synapses"
};

struct Adjacency_List {
    Vector<Vector<NML_Network_Edge>> list;

    Adjacency_List(int node_count, int max_edge_count) {
        list.resize(node_count);
        for (Vector<NML_Network_Edge> &inner_list : list) {
            inner_list.reserve(max_edge_count);
        }
    }

    void clear() {
        list.clear();
    }

    bool in_network(s64 node_index) {
        return (node_index >= 0 && node_index < static_cast<s64>(list.size()));
    }

    void add(NML_Network_Edge edge) {
        if (!in_network(edge.parent) || !in_network(edge.child)) {
            return;
        }

        list[edge.parent].push_back(edge);
    }

    bool is_parent(s64 parent, s64 prospective_child) {
        for (auto &edge: list[parent]) {
            if (edge.child != prospective_child) continue;
            return true;
        }
        return false;
    }
};

}
