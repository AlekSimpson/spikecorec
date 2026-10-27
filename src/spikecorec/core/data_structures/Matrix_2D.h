//
// Created by Alek Simpson on 10/18/26.
//
#pragma once

#include "../types.h"

using namespace spikecorec;

template <typename Type>
struct Matrix_2D {
    s64 rows;
    s64 columns;
    s64 count;
    Type *data;

    Matrix_2D();
    Matrix_2D(Allocator &allocator, s64 rows, s64 columns);

    Type &get(s64 i, s64 j);
};










