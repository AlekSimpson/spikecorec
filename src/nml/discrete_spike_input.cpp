#include "spikecorec/nml/discrete_spike_input.h"

namespace spikecorec::nml {

void DiscreteSpikeInputSchedule::apply_to_network_inputs(f32 *network_inputs, s64 tick) const {
    if (tick < 0 || tick >= (s64)spike_bits.size()) return;

    const Vector<u8> &row = spike_bits[(usize)tick];
    for (usize index = 0; index < row.size() && index < target_neuron_indices.size(); ++index) {
        if (row[index] == 0) continue;
        network_inputs[target_neuron_indices[index]] += current_amplitude_amperes;
    }
}

} // namespace spikecorec::nml
