#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <utility>

#include "spikecorec/nml/stimulus_schedule.h"

#include "spikecorec/core/log.h"
#include "spikecorec/core/time_utilities.h"

using namespace std;
using namespace spikecorec;

namespace spikecorec::nml {

namespace {

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

StimulusSchedule::StimulusSchedule(s32 window_count_)
    : window_count(window_count_) {
    windows_by_neuron.reserve((usize)window_count);
    start_ticks = new s64[(usize)window_count];
    end_ticks = new s64[(usize)window_count];
    current_values = new f64[(usize)window_count];
    target_neurons = new s32[(usize)window_count];
}

StimulusSchedule::StimulusSchedule(StimulusSchedule &&other) noexcept
    : windows_by_neuron(std::move(other.windows_by_neuron)),
      start_ticks(other.start_ticks),
      end_ticks(other.end_ticks),
      current_values(other.current_values),
      target_neurons(other.target_neurons),
      window_count(other.window_count) {
    other.start_ticks = nullptr;
    other.end_ticks = nullptr;
    other.current_values = nullptr;
    other.target_neurons = nullptr;
    other.window_count = 0;
}

StimulusSchedule &StimulusSchedule::operator=(StimulusSchedule &&other) noexcept {
    if (this == &other) return *this;

    delete[] start_ticks;
    delete[] end_ticks;
    delete[] current_values;
    delete[] target_neurons;

    windows_by_neuron = std::move(other.windows_by_neuron);
    start_ticks = other.start_ticks;
    end_ticks = other.end_ticks;
    current_values = other.current_values;
    target_neurons = other.target_neurons;
    window_count = other.window_count;

    other.start_ticks = nullptr;
    other.end_ticks = nullptr;
    other.current_values = nullptr;
    other.target_neurons = nullptr;
    other.window_count = 0;

    return *this;
}

StimulusSchedule::~StimulusSchedule() {
    delete[] start_ticks;
    delete[] end_ticks;
    delete[] current_values;
    delete[] target_neurons;
}

void StimulusSchedule::set_window(s32 index, s64 neuron_index, s64 start_tick, s64 end_tick, f64 current) {
    windows_by_neuron[neuron_index].push_back(index);
    start_ticks[index] = start_tick;
    end_ticks[index] = end_tick;
    current_values[index] = current;
    target_neurons[index] = (s32)neuron_index;
}

f64 StimulusSchedule::current_at(s32 neuron_index, s64 tick) const {
    auto found = windows_by_neuron.find(neuron_index);
    if (found == windows_by_neuron.end()) return 0.0;

    f64 total_current = 0.0;
    for (s32 window_index : found->second) {
        if (tick < start_ticks[window_index] || tick >= end_ticks[window_index]) continue;
        total_current += current_values[window_index];
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
    StimulusSchedule schedule((s32)model.stimuli.size());

    s32 index = 0;
    for (const auto &stimulus_entry : model.stimuli) {
        const TypeLibraryEntry &stimulus = model.type_library[(usize)stimulus_entry.input_type_library_index];

        // Phase 1 only host-precomputes pulseGenerator (arch §5); every other
        // Inputs ComponentType is lowered to on-device code instead (Phase 2)
        // and never produces a StimulusEntry this function is handed, so
        // reaching here with a non-pulseGenerator input is a genuine
        // front-end/lowering mismatch, not a case this function handles.
        if (stimulus.component_type_name != "pulseGenerator") {
            log::throw_runtime_error(log::logger(),
                "stimulus_schedule: input '" + stimulus.bound_instance_id + "' is a '" +
                stimulus.component_type_name + "', not a pulseGenerator -- Phase 1 only "
                "host-precomputes pulseGenerator (arch §5; other D4 generators are on-device, Phase 2)");
        }

        f64 delay_seconds = required_baked_constant(stimulus, "delay");
        f64 duration_seconds = required_baked_constant(stimulus, "duration");
        f64 amplitude = required_baked_constant(stimulus, "amplitude");

        if (duration_seconds <= 0.0) {
            log::throw_runtime_error(log::logger(),
                "stimulus_schedule: pulseGenerator '" + stimulus.bound_instance_id +
                "' has duration <= 0 (never applies any current)");
        }

        auto weight_value = stimulus.baked_constants.find("weight");
        f64 weight = weight_value == stimulus.baked_constants.end()
            ? DEFAULT_PULSE_GENERATOR_WEIGHT
            : weight_value->second;

        s64 start_tick = seconds_to_tick_count_allowing_zero(delay_seconds, seconds_step,
            "pulseGenerator '" + stimulus.bound_instance_id + "' delay");
        s64 duration_ticks = seconds_to_tick_count_allowing_zero(duration_seconds, seconds_step,
            "pulseGenerator '" + stimulus.bound_instance_id + "' duration");

        schedule.set_window(index, stimulus_entry.target_neuron_index, start_tick, start_tick + duration_ticks,
                             amplitude * weight);
        ++index;
    }

    return schedule;
}

} // namespace spikecorec::nml
