//
// Created by Alek Simpson on 10/18/26.
//

#include "../../spikecorec.h"

#include "Matrix_2D.h"

using namespace spikecorec;
using namespace spikecorec::Program_Context;


template<typename Type>
Matrix_2D<Type>::Matrix_2D() {
    data = nullptr;
    rows = 0;
    columns = 0;
}

template<typename Type>
Matrix_2D<Type>::Matrix_2D(s64 rows, s64 columns) : rows(rows), columns(columns) {
    s64 count = rows * columns;
    data = (Type *) pool.allocate(sizeof(Type), count);
}

template<typename Type>
s64 Matrix_2D<Type>::count() const {
    return rows * columns;
}

template<typename Type>
Type &Matrix_2D<Type>::get(s64 i, s64 j) {
    return data[rows * i + j];
}







