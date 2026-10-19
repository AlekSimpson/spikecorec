//
// Created by Alek Simpson on 10/18/26.
//
#pragma once

#include "../types.h"

using namespace spikecorec;

struct Linear_Allocator {
    usize max_byte_size;

    u64 bookmark_offset = 0;
    u64 offset = 0;

    u64 *memory;

    Linear_Allocator();
    Linear_Allocator(usize max_size);
    ~Linear_Allocator();

    void set_bookmark(u64 offset);
    void set_offset(u64);

    u64 *allocate(u64 bytesize);
    void bookmark();
    void free(); // reset to bookmark
    void clear(); // resets to 0
};


