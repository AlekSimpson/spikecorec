#pragma once

#include "spikecorec/core/units.h"
#include "spikecorec/core/types.h"
#include "spikecorec/core/recording.h"

using namespace spikecorec;
using namespace std;

namespace spikecorec::nml {

OutputFileFormat output_format_for_filename(const String &filename);

union Real {
    s64 int64;
    f64 float64;
};

struct InputTarget {
    s64 neuron_index = -1;
    f64 weight = 1.0;        // from <inputW weight="..."/>, 1.0 when unweighted
    Vector<s32> event_ticks; // spike train, empty for a continuous injector
};

struct SimulationInputConfig {
    String input_component_id;

    Vector<InputTarget> targets;

    f64 amplitude = 0.0;
    f64 rate = 0.0;

    // input is given while: start_tick <= tick < end_tick; 
    // end_tick == 0 means runs to the end of the simulation
    s64 start_tick = 0;
    s64 end_tick = 0;

    bool continuous_current_injection = false;
};

struct NML_NetworkEdge {
    String component_id;
    f32 weight;
    s64 delay_ticks;
    s64 parent;
    s64 child;
};

struct AdjacencyList {
    Vector<Vector<NML_NetworkEdge>> list;

    AdjacencyList(int node_count, int max_edge_count) {
        list.resize(node_count);
        for (Vector<NML_NetworkEdge> &inner_list : list) {
            inner_list.reserve(max_edge_count);
        }
    }

    void clear() {
        list.clear();
    }

    bool in_network(s64 node_index) {
        return (node_index >= 0 && node_index < static_cast<s64>(list.size()));
    }

    void add(NML_NetworkEdge edge) {
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
