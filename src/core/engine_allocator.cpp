#include "spikecorec/core/engine_allocator.h"
#include "spikecorec/core/types.h"

u64 EngineAllocator::

EnginePointer EngineAllocator::allocate_memory(u64 total_byte_size, s64 length) {
    if (cursor == MAX_CAPACITY) return EnginePointer::NULLPOINTER();

    auto pointer = EnginePointer(&memory[cursor], length);
    cursor += 1;
    return pointer;
}

void EngineAllocator::deallocate_all_memory() {
    if (memory != nullptr) {
        delete[] memory;
    }
    cursor = 0;
}
