#include <cmath>

#include "spikecorec/nml/plasticity_wiring.h"
#include "spikecorec/core/log.h"

using namespace std;

namespace spikecorec::nml {

namespace {
const char *const TAU_PLUS_NAME = "tauPlus";
const char *const TAU_MINUS_NAME = "tauMinus";
const char *const A_PLUS_NAME = "aPlus";
const char *const A_MINUS_NAME = "aMinus";
} // namespace

std::optional<StdpSpec> find_stdp_spec(const TypeLibraryEntry &synapse_entry) {
    if (synapse_entry.category != TypeLibraryCategory::Synapse) return std::nullopt;

    const auto &baked = synapse_entry.baked_constants;
    auto tau_plus = baked.find(TAU_PLUS_NAME);
    auto tau_minus = baked.find(TAU_MINUS_NAME);
    auto a_plus = baked.find(A_PLUS_NAME);
    auto a_minus = baked.find(A_MINUS_NAME);
    if (tau_plus == baked.end() || tau_minus == baked.end() || a_plus == baked.end() || a_minus == baked.end())
        return std::nullopt;

    return StdpSpec{
        (f32)tau_plus->second,
        (f32)tau_minus->second,
        (f32)a_plus->second,
        (f32)a_minus->second,
    };
}

f32 map_stdp_spec_to_learning_rate(const StdpSpec &spec) { return spec.a_minus; }

void apply_stdp_wiring(const ModelSpecification &model, SpikeEngine &engine) {
    std::optional<f32> mapped_learning_rate;

    for (const auto &entry : model.type_library) {
        std::optional<StdpSpec> spec = find_stdp_spec(entry);
        if (!spec) continue;

        f32 rate = map_stdp_spec_to_learning_rate(*spec);
        if (!mapped_learning_rate) {
            mapped_learning_rate = rate;
            continue;
        }
        if (std::fabs(*mapped_learning_rate - rate) > 1e-9f) {
            log::logger().warn(
                "apply_stdp_wiring: model has multiple STDP-shaped synapses mapping to different "
                "learning rates ({} vs {}) -- Phase 1's SpikeEngine has one global learning_rate "
                "scalar, so the first one found ('{}') wins",
                *mapped_learning_rate, rate, entry.bound_instance_id);
        }
    }

    if (mapped_learning_rate) {
        engine.enable_plasticity(*mapped_learning_rate);
    } else {
        engine.disable_plasticity();
    }
}

} // namespace spikecorec::nml
