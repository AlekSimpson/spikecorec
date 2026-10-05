#include "spikecorec/nml/random_generator.h"

namespace spikecorec::nml {

RandomGenerator::RandomGenerator(u64 seed): random_engine(seed) {}

f64 RandomGenerator::uniform() {
    // The middle of one of 2^23 equal bins, which a float holds exactly.
    return (static_cast<f64>(random_engine() >> 41) + 0.5) / 8388608.0;
}

f64 RandomGenerator::normal(f64 mean, f64 standard_deviation) {
    return std::normal_distribution<f64>(mean, standard_deviation)(random_engine);
}

f64 RandomGenerator::exponential(f64 rate) {
    return std::exponential_distribution<f64>(rate)(random_engine);
}

s64 RandomGenerator::poisson(f64 mean) {
    return std::poisson_distribution<s64>(mean)(random_engine);
}

}
