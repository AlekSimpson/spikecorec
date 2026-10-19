//
// Created by Alek Simpson on 10/18/26.
//
#pragma once

#include "types.h"
#include "data_structures/Matrix_2D.h"

using namespace spikecorec;

struct Connectome {
    s64 *source = nullptr;
    s64 *target = nullptr;

    f32 *weight = nullptr;
    s32 *delay_in_ticks = nullptr;

    s64 connection_count = 0;

    Connectome() {};
    Connectome(s64 connection_count);

    void set_topology(
        Matrix_2D<s32> &topology,
        Matrix_2D<f64> &weights, Matrix_2D<s32> &delays);

    using Generation_Function = void (*)(s64);
    Connectome(s64 connection_count, Generation_Function generation_function);
};



