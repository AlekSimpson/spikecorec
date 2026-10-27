//
// Created by Alek Simpson on 10/24/26.
//
#pragma once

#include "../types.h"

using namespace spikecorec;

struct Allocator {
    virtual ~Allocator() = default;
    virtual u8 *allocate(u64 bytes) = 0;
};



