#include <gtest/gtest.h>

#include "backend.h"

using namespace spikecorec;

// No process-wide GPU setup; each test that needs a backend constructs its own.
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
