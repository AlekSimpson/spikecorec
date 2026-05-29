#include "spikecorec/core/types.hpp"
#include <cassert>
#include <cstdio>

int main() {
    // Placeholder — add tests here.
    static_assert(sizeof(spikecorec::f32) == 4);
    static_assert(sizeof(spikecorec::f64) == 8);

    printf("core tests passed\n");
    return 0;
}
