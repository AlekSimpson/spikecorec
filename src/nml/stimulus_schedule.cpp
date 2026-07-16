#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include "spikecorec/nml/stimulus_schedule.h"

#include "spikecorec/core/log.h"
#include "spikecorec/core/time_utilities.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::nml {

namespace {

// pulseGenerator's own documented default (third_party/neuroml2/std_lib/
// Inputs.xml: `<Property name="weight" dimension="none" defaultValue="1"/>`)
// -- see stimulus_schedule.h's doc comment for why this can't be read off
// `baked_constants` itself.
constexpr f64 DEFAULT_PULSE_GENERATOR_WEIGHT = 1.0;

f64 required_baked_constant(const TypeLibraryEntry &entry, const String &name) {
    auto value = entry.baked_constants.find(name);
    if (value == entry.baked_constants.end()) {
        log::throw_runtime_error(log::logger(),
            "stimulus_schedule: pulseGenerator '" + entry.bound_instance_id + "' has no resolved '" +
            name + "' value");
    }
    return value->second;
}

// `delay`/`duration` are seconds (canonical SI); `time::tick_count_from_seconds`
// rejects a zero-or-negative total (it's shared with call sites where that's
// meaningless), but a zero `delay` is an ordinary, common pulseGenerator
// configuration ("start immediately"), so it's special-cased here rather
// than routed through that guard.
s64 seconds_to_tick_count_allowing_zero(f64 total_seconds, f64 seconds_step, const String &context) {
    if (total_seconds < 0.0) {
        log::throw_runtime_error(log::logger(), "stimulus_schedule: " + context + " must be >= 0");
    }
    if (total_seconds == 0.0) return 0;
    return time::tick_count_from_seconds(total_seconds, seconds_step);
}

} // namespace

f64 StimulusSchedule::current_at(s32 neuron_index, s64 tick) const {
    f64 total_current = 0.0;
    for (const auto &window : windows) {
        if (window.target_neuron_index != neuron_index) continue;
        if (tick < window.start_tick || tick >= window.end_tick) continue;
        total_current += window.current_value;
    }
    return total_current;
}

Vector<Vector<f32>> StimulusSchedule::to_dense_input_spikes(const Vector<s32> &target_neuron_indices, s64 tick_count) const {
    Vector<Vector<f32>> dense((usize)tick_count, Vector<f32>(target_neuron_indices.size(), 0.0f));
    for (s64 tick = 0; tick < tick_count; ++tick) {
        for (usize index = 0; index < target_neuron_indices.size(); ++index) {
            dense[(usize)tick][index] = (f32)current_at(target_neuron_indices[index], tick);
        }
    }
    return dense;
}

StimulusSchedule build_stimulus_schedule(const ModelSpecification &model, f64 seconds_step) {
    StimulusSchedule schedule;

    for (const auto &stimulus : model.stimuli) {
        const TypeLibraryEntry &input_entry = model.type_library[(usize)stimulus.input_type_library_index];

        if (input_entry.component_type_name != "pulseGenerator") {
            log::throw_runtime_error(log::logger(),
                "stimulus_schedule: input '" + input_entry.bound_instance_id + "' is a '" +
                input_entry.component_type_name + "', not a pulseGenerator -- Phase 1 only "
                "host-precomputes pulseGenerator (arch §5; other D4 generators are on-device, Phase 2)");
        }

        f64 delay_seconds = required_baked_constant(input_entry, "delay");
        f64 duration_seconds = required_baked_constant(input_entry, "duration");
        f64 amplitude = required_baked_constant(input_entry, "amplitude");

        if (duration_seconds <= 0.0) {
            log::throw_runtime_error(log::logger(),
                "stimulus_schedule: pulseGenerator '" + input_entry.bound_instance_id +
                "' has duration <= 0 (never applies any current)");
        }

        auto weight_value = input_entry.baked_constants.find("weight");
        f64 weight = weight_value == input_entry.baked_constants.end()
            ? DEFAULT_PULSE_GENERATOR_WEIGHT
            : weight_value->second;

        s64 start_tick = seconds_to_tick_count_allowing_zero(delay_seconds, seconds_step,
            "pulseGenerator '" + input_entry.bound_instance_id + "' delay");
        s64 duration_ticks = seconds_to_tick_count_allowing_zero(duration_seconds, seconds_step,
            "pulseGenerator '" + input_entry.bound_instance_id + "' duration");

        StimulusWindow window;
        window.target_neuron_index = stimulus.target_neuron_index;
        window.start_tick = start_tick;
        window.end_tick = start_tick + duration_ticks;
        window.current_value = amplitude * weight;
        schedule.windows.push_back(window);
    }

    return schedule;
}

} // namespace spikecorec::nml
