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

f32 map_stdp_spec_to_learning_rate(const StdpSpec &spec) {
    // `aMinus` is a depression AMPLITUDE (NeuroML convention, always >= 0) feeding
    // step_apply_hebbian_update's own always-negative decay_delta = -learning_rate * pow(tick_delta,
    // -3) (kernels.metal). SpikeEngine::plasticity_enabled() reads `learning_rate > 0.0f`, so a
    // negative aMinus would report plasticity DISABLED while the kernel's own `learning_rate !=
    // 0.0f` dispatch guard still fires and flips decay_delta's sign to potentiation -- an
    // inconsistent state no real NeuroML model should produce, but worth rejecting explicitly
    // rather than silently mapping through.
    if (spec.a_minus < 0.0f) {
        log::throw_invalid_argument(log::logger(),
            "map_stdp_spec_to_learning_rate: aMinus must be >= 0 (a depression amplitude), got " +
            std::to_string(spec.a_minus));
    }
    return spec.a_minus;
}

f32 map_stdp_spec_to_learning_rate_plus(const StdpSpec &spec) {
    // `aPlus` is a potentiation AMPLITUDE (NeuroML convention, always >= 0) feeding the causal
    // (pre-before-post) predecessor-edge potentiation both stage-7 call sites now apply: a positive
    // potentiate_delta = +learning_rate_plus * pow(tick_delta, -3), the LTP mirror of the depression
    // side's own always-negative decay_delta. Rejected explicitly if negative, matching
    // map_stdp_spec_to_learning_rate's own aMinus guard rather than silently mapping a nonsensical
    // negative amplitude through.
    if (spec.a_plus < 0.0f) {
        log::throw_invalid_argument(log::logger(),
            "map_stdp_spec_to_learning_rate_plus: aPlus must be >= 0 (a potentiation amplitude), got " +
            std::to_string(spec.a_plus));
    }
    return spec.a_plus;
}

void apply_stdp_wiring(const ModelSpecification &model, SpikeEngine &engine) {
    std::optional<f32> mapped_learning_rate;
    f32 mapped_learning_rate_plus = 0.0f; // plus-side rate of whichever spec wins below
    String winning_synapse_id;

    for (const auto &entry : model.type_library) {
        std::optional<StdpSpec> spec = find_stdp_spec(entry);
        if (!spec) continue;

        f32 rate = map_stdp_spec_to_learning_rate(*spec);
        f32 rate_plus = map_stdp_spec_to_learning_rate_plus(*spec);
        if (!mapped_learning_rate) {
            mapped_learning_rate = rate;
            mapped_learning_rate_plus = rate_plus;
            winning_synapse_id = entry.bound_instance_id;
            continue;
        }
        if (std::fabs(*mapped_learning_rate - rate) > 1e-9f) {
            log::logger().warn(
                "apply_stdp_wiring: model has multiple STDP-shaped synapses mapping to different "
                "learning rates ({} vs {}) -- Phase 1's SpikeEngine has one global learning_rate "
                "scalar, so the first one found ('{}') wins, ignoring '{}'",
                *mapped_learning_rate, rate, winning_synapse_id, entry.bound_instance_id);
        }
    }

    if (mapped_learning_rate) {
        engine.enable_plasticity(*mapped_learning_rate, mapped_learning_rate_plus);
    } else {
        engine.disable_plasticity();
    }
}

} // namespace spikecorec::nml
