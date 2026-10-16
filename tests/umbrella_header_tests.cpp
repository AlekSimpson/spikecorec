// spikecorec.h, the one-include header, compiles on its own and its helpers join strings.

#include <gtest/gtest.h>

#include "spikecorec/spikecorec.h"

using namespace spikecorec;

TEST(UmbrellaHeader, combine_joins_values_in_order) {
    EXPECT_EQ(combine(Vector<String>{"a", "b", "c"}), "abc");
    EXPECT_EQ(combine(String(", "), Vector<String>{"a", "b"}), "a, b, ");
    EXPECT_EQ(combine(Vector<String>{}), "");
}
