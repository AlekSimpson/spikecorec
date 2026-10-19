//
// Created by Alek Simpson on 10/18/26.
//
#pragma once

#include "../types.h"

using namespace spikecorec;

struct Pool_Allocator {
    struct Pool {
        u64 *memory;
        u64 size;
        u64 offset;

        u64 *increment(s64 count = 1);
    };

    u64 pool_depth; // bytes
    u64 max_pool_count;

    Unordered_Map<u64, Pool> pools;

    Pool_Allocator();
    Pool_Allocator(u64 max_pool_count, u64 pool_depth);
    ~Pool_Allocator();

    u64 *allocate(u64 bytesize);
    u64 *allocate(u64 bytesize, s64 count);
    void release(u64*, u64 bytesize);
    void clear();
};
