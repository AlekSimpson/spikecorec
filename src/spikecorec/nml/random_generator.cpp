#include "random_generator.h"

namespace spikecorec::nml {

Random_Generator::Random_Generator(u64 seed): random_engine(seed) {}

f64 Random_Generator::uniform() {
    // The middle of one of 2^23 equal bins, which a float holds exactly.
    return (static_cast<f64>(random_engine() >> 41) + 0.5) / 8388608.0;
}

f64 Random_Generator::normal(f64 mean, f64 standard_deviation) {
    return Normal_Distribution<f64>(mean, standard_deviation)(random_engine);
}

f64 Random_Generator::exponential(f64 rate) {
    return Exponential_Distribution<f64>(rate)(random_engine);
}

s64 Random_Generator::poisson(f64 mean) {
    return Poisson_Distribution<s64>(mean)(random_engine);
}

}
