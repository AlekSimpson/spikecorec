//
// Created by Alek Simpson on 10/18/26.
//

#include "../types.h"
#include "../log.h"

#include "linear_allocator.h"

using namespace spikecorec;
using namespace spikecorec::log;

Linear_Allocator::Linear_Allocator() {
    max_byte_size = 0;
    memory = nullptr;
}


Linear_Allocator::Linear_Allocator(usize max_size) : max_byte_size(max_size) {
    memory = new u64[max_size];
}

Linear_Allocator::~Linear_Allocator() {
    if (memory != nullptr) {
        delete[] memory;
    }
}

u64 *Linear_Allocator::allocate(u64 bytesize) {
    if (offset + bytesize >= max_byte_size) {
        logger().critical("Linear Allocator: ran out of room, cannot allocate more than {} bytes.", max_byte_size);
        return nullptr;
    }

    u64 *return_value = &memory[offset];
    offset += bytesize;
    return return_value;
}

void Linear_Allocator::bookmark() {
    bookmark_offset = offset;
}

void Linear_Allocator::free() {
    offset = bookmark_offset;
}

void Linear_Allocator::clear() {
    offset = 0;
    bookmark_offset = 0;
}

void Linear_Allocator::set_bookmark(u64 _offset) {
    bookmark_offset = _offset;
}

void Linear_Allocator::set_offset(u64 value) {
    offset = value;
}

