//
// Created by Alek Simpson on 10/18/26.
//

#include "types.h"
#include "../spikecorec.h"

#include "connectome.h"

using namespace spikecorec;
using namespace spikecorec::Program_Context;


Connectome::Connectome(s64 connection_count)
: connection_count(connection_count) {

  source = (s64 *) pool.allocate(sizeof(s64), connection_count);
  target = (s64 *) pool.allocate(sizeof(s64), connection_count);

  weight = (f32 *) pool.allocate(sizeof(s64), connection_count);
  delay_in_ticks = (s32 *) pool.allocate(sizeof(s32), connection_count);
}

Connectome::Connectome(s64 connection_count, Generation_Function generation_function)
: connection_count(connection_count) {
  // todo:
}

void Connectome::set_topology(
    Matrix_2D<s32> &topology,
    Matrix_2D<f64> &weights, Matrix_2D<s32> &delays
) {
    s64 neuron_count = topology.count();

    s64 index;
    for (s64 row = 0; row < neuron_count; ++row) {
        for (s64 column = 0; column < neuron_count; ++column) {
            if (topology.get(row, column) == 0) continue;

            index = row * neuron_count + column;
            if (index >= connection_count) {
                panic("Connectome::set_topology: index out of bounds");
            }

            connection_count++;

            source[index] = row;
            target[index] = column;

            weight[index] = weights.get(row, column);
            delay_in_ticks[index] = delays.get(row, column);
        }
    }
}






