//
// Created by Alek Simpson on 10/18/26.
//

#include "../types.h"
#include "../log.h"

#include "pool_allocator.h"

using namespace spikecorec;
using namespace spikecorec::log;

u64 *Pool_Allocator::Pool::increment(s64 count) {
    u64 *return_value = &memory[offset];
    offset += size * count;
    return return_value;
}

Pool_Allocator::Pool create_pool(u64 size, u64 offset) {
    return Pool_Allocator::Pool{nullptr, size, offset};
}

Pool_Allocator::Pool_Allocator() {
    this->max_pool_count = 0;
    this->pool_depth = 0;
}

Pool_Allocator::Pool_Allocator(u64 max_pool_count, u64 pool_depth) {
    this->max_pool_count = max_pool_count;
    this->pool_depth = pool_depth;

    pools.reserve(max_pool_count);
}

Pool_Allocator::~Pool_Allocator() {
    this->clear();
}

void Pool_Allocator::clear() {
    if (pools.empty()) return;

    for (const auto &[key, value] : pools) {
        if (value.memory != nullptr) delete[] value.memory;
    }
    pools.clear();
}

u64 *Pool_Allocator::allocate(u64 bytesize) {
    bool all_pools_in_use = (pools.size() == max_pool_count);
    bool have_seen_size_before = pools.find(bytesize) != pools.end();
    if (
        !have_seen_size_before
     && all_pools_in_use
    ) {
        logger().critical("Pool Allocator: Cannot add pool for size: {}. Ran out of room.", bytesize);
        return nullptr;
    }

    if (!have_seen_size_before) {
        auto *memory = new u64[pool_depth];
        pools[bytesize] = create_pool(bytesize, 0);
        pools[bytesize].memory = memory;
    }

    Pool &pool = pools[bytesize];
    return pool.increment();
}

u64 *Pool_Allocator::allocate(u64 bytesize, s64 count) {
    bool all_pools_in_use = (pools.size() == max_pool_count);
    bool have_seen_size_before = pools.find(bytesize) != pools.end();
    if (
        !have_seen_size_before
     && all_pools_in_use
    ) {
        logger().critical("Pool Allocator: Cannot add pool for size: {}. Ran out of room.", bytesize);
        return nullptr;
    }

    if (!have_seen_size_before) {
        auto *memory = new u64[pool_depth];
        pools[bytesize] = create_pool(bytesize, 0);
        pools[bytesize].memory = memory;
    }

    Pool &pool = pools[bytesize];
    return pool.increment(count);
}

void Pool_Allocator::release(u64 *pointer, u64 bytesize) {
    // todo:
}
