#pragma once

#include <random>

#include "spikecorec/core/types.h"

using namespace spikecorec;

namespace spikecorec::nml {

// Random values from built-in distributions, all drawn from one engine seeded with the
// simulation seed. Before every tick the engine fills the kernel's random_values buffer from
// it, and OnStart values that call random() draw from it on the host.
struct RandomGenerator {
    std::mt19937_64 random_engine;

    explicit RandomGenerator(u64 seed = 0);

    // Uniform on (0, 1), never 0 or 1 even as a float, so log(random(1)) and
    // log(1 - random(1)) stay finite. This is what LEMS random(1) draws.
    f64 uniform();
    f64 normal(f64 mean, f64 standard_deviation);
    f64 exponential(f64 rate);
    s64 poisson(f64 mean);
};

}
