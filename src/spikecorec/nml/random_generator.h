#pragma once

#include <random>

#include "types.h"

using namespace spikecorec;

namespace spikecorec::nml {

// One engine, seeded with the simulation seed, for every draw: the random_values buffer,
// refilled before each tick, and the OnStart random() calls evaluated on the host.
struct Random_Generator {
    Mersenne_Twister_64 random_engine;

    explicit Random_Generator(u64 seed = 0);

    // Uniform on (0, 1), even as a float, so log(random(1)) and log(1 - random(1)) are finite.
    f64 uniform();
    f64 normal(f64 mean, f64 standard_deviation);
    f64 exponential(f64 rate);
    s64 poisson(f64 mean);
};

}
