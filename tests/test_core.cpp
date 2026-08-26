#include <gtest/gtest.h>

#include "spikecorec/core/backend.h"

using namespace spikecorec;

// The GPU backend is an object each engine, weight matrix and k^2-tree owns and releases
// with itself, so there is no process-wide context for a test main to bring up or tear
// down any more. Tests that need a backend construct one.
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
