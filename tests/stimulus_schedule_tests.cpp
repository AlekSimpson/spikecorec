#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <gtest/gtest.h>

#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/stimulus_schedule.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── Host-precomputed stimulus schedule tests (ticket #58 [E1]) ───────────

namespace {

String write_temp_file(const String &filename, const String &contents) {
    String path = (std::filesystem::temp_directory_path() / filename).string();
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

const TypeLibraryEntry &type_library_entry_for(const ModelSpecification &specification, const String &bound_instance_id) {
    for (const auto &entry : specification.type_library) {
        if (entry.bound_instance_id == bound_instance_id) return entry;
    }
    throw std::runtime_error("no type library entry for '" + bound_instance_id + "'");
}

// A minimal cell type (no OnCondition/OnStart -- model_specification_tests.cpp's
// own bad-connection fixture uses exactly this shape) + one pulseGenerator
// driven by an explicitInput. Parameterized by the pulseGenerator's own
// `delay`/`duration`/`amplitude`/`weight` attributes so every unit test below
// varies only those.
ModelSpecification build_single_pulse_fixture(
    const String &fixture_id, const String &delay, const String &duration,
    const String &amplitude, const String &weight_attribute = ""
) {
    write_temp_file("spikecorec_stimulus_schedule_" + fixture_id + "_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"StimulusScheduleContent" + fixture_id + "\">"
        "  <ComponentType name=\"TestCell\" extends=\"baseCell\">"
        "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
        "    <Parameter name=\"gL\" dimension=\"conductance\"/>"
        "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
        "      <TimeDerivative variable=\"v\" value=\"(gL * (EL - v)) / C\"/>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <TestCell id=\"cellInstance\" C=\"1.0e-10\" gL=\"1.0e-8\" EL=\"-70mV\"/>"
        "  <pulseGenerator id=\"pulseGen1\" delay=\"" + delay + "\" duration=\"" + duration +
        "\" amplitude=\"" + amplitude + "\"" +
        (weight_attribute.empty() ? "" : " weight=\"" + weight_attribute + "\"") + "/>"
        "  <network id=\"Net\">"
        "    <population id=\"Pop\" component=\"cellInstance\" size=\"1\"/>"
        "    <explicitInput target=\"Pop[0]\" input=\"pulseGen1\"/>"
        "  </network>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_stimulus_schedule_" + fixture_id + "_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"StimulusScheduleTop" + fixture_id + "\">"
        "  <include href=\"spikecorec_stimulus_schedule_" + fixture_id + "_content.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

// Two pulseGenerators, both explicitInput-driven onto the SAME (only) neuron
// in a size-1 population -- exercises current_at()'s summing rule.
ModelSpecification build_two_overlapping_pulses_fixture() {
    write_temp_file("spikecorec_stimulus_schedule_overlap_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"StimulusScheduleOverlapContent\">"
        "  <ComponentType name=\"TestCell\" extends=\"baseCell\">"
        "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <TestCell id=\"cellInstance\" C=\"1.0e-10\"/>"
        "  <pulseGenerator id=\"pulseGenA\" delay=\"0ms\" duration=\"100ms\" amplitude=\"1nA\"/>"
        "  <pulseGenerator id=\"pulseGenB\" delay=\"20ms\" duration=\"50ms\" amplitude=\"2nA\"/>"
        "  <network id=\"Net\">"
        "    <population id=\"Pop\" component=\"cellInstance\" size=\"1\"/>"
        "    <explicitInput target=\"Pop[0]\" input=\"pulseGenA\"/>"
        "    <explicitInput target=\"Pop[0]\" input=\"pulseGenB\"/>"
        "  </network>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_stimulus_schedule_overlap_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"StimulusScheduleOverlapTop\">"
        "  <include href=\"spikecorec_stimulus_schedule_overlap_content.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

// A size-2 population with two separate pulseGenerators, each targeting a
// DIFFERENT neuron -- exercises wiring multiple `explicitInput` Structure-level
// specs into distinct, non-interfering schedule entries (the shape a real
// Phase-1 GLIF network drives its cells with).
ModelSpecification build_two_neuron_targets_fixture() {
    write_temp_file("spikecorec_stimulus_schedule_two_targets_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"StimulusScheduleTwoTargetsContent\">"
        "  <ComponentType name=\"TestCell\" extends=\"baseCell\">"
        "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <TestCell id=\"cellInstance\" C=\"1.0e-10\"/>"
        "  <pulseGenerator id=\"pulseGenA\" delay=\"5ms\" duration=\"20ms\" amplitude=\"1nA\"/>"
        "  <pulseGenerator id=\"pulseGenB\" delay=\"15ms\" duration=\"20ms\" amplitude=\"3nA\"/>"
        "  <network id=\"Net\">"
        "    <population id=\"Pop\" component=\"cellInstance\" size=\"2\"/>"
        "    <explicitInput target=\"Pop[0]\" input=\"pulseGenA\"/>"
        "    <explicitInput target=\"Pop[1]\" input=\"pulseGenB\"/>"
        "  </network>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_stimulus_schedule_two_targets_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"StimulusScheduleTwoTargetsTop\">"
        "  <include href=\"spikecorec_stimulus_schedule_two_targets_content.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

// A `sineGenerator` (a real vendored InputsType, `extends="basePointCurrent"`
// directly -- correctly classified as a Dynamics/Inputs ComponentType, unlike
// `pulseGeneratorDL`, whose own `basePointCurrentDL` root doesn't chain to
// the `basePointCurrent`/`baseSpikeSource` anchors nml.cpp's classifier
// checks for) driving the input instead of `pulseGenerator` -- Phase 1's
// documented unsupported-input-type boundary case.
ModelSpecification build_unsupported_input_fixture() {
    write_temp_file("spikecorec_stimulus_schedule_unsupported_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"StimulusScheduleUnsupportedContent\">"
        "  <ComponentType name=\"TestCell\" extends=\"baseCell\">"
        "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <TestCell id=\"cellInstance\" C=\"1.0e-10\"/>"
        "  <sineGenerator id=\"sineGen1\" phase=\"0\" delay=\"10ms\" duration=\"50ms\" amplitude=\"1nA\" period=\"20ms\"/>"
        "  <network id=\"Net\">"
        "    <population id=\"Pop\" component=\"cellInstance\" size=\"1\"/>"
        "    <explicitInput target=\"Pop[0]\" input=\"sineGen1\"/>"
        "  </network>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_stimulus_schedule_unsupported_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"StimulusScheduleUnsupportedTop\">"
        "  <include href=\"spikecorec_stimulus_schedule_unsupported_content.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

// A GLIF1Cell (the regime-less/refractory-less form -- docs/nml_ir_spec.md
// §4's own canonical GLIF1 example omits refractory too, per
// cell_lowering_tests.cpp's "PlainLifCell") + one pulseGenerator explicitInput,
// parameterized by the pulse's amplitude so the acceptance-criterion tests
// below can exercise both a sub-threshold and a spiking current step against
// the SAME cell.
ModelSpecification build_glif1_current_step_fixture(const String &fixture_id, const String &amplitude) {
    write_temp_file("spikecorec_stimulus_schedule_glif1_" + fixture_id + "_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"StimulusScheduleGlif1Content" + fixture_id + "\">"
        "  <ComponentType name=\"GLIF1Cell\" extends=\"baseCell\">"
        "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
        "    <Parameter name=\"gL\" dimension=\"conductance\"/>"
        "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
        "    <Parameter name=\"vth\" dimension=\"voltage\"/>"
        "    <Parameter name=\"vreset\" dimension=\"voltage\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
        "      <TimeDerivative variable=\"v\" value=\"(gL * (EL - v)) / C\"/>"
        "      <OnStart>"
        "        <StateAssignment variable=\"v\" value=\"EL\"/>"
        "      </OnStart>"
        "      <OnCondition test=\"v .gt. vth\">"
        "        <EventOut port=\"spike\"/>"
        "        <StateAssignment variable=\"v\" value=\"vreset\"/>"
        "      </OnCondition>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <GLIF1Cell id=\"glif1CellInstance\" C=\"1.0e-10\" gL=\"1.0e-8\" EL=\"-70mV\" vth=\"-50mV\" vreset=\"-70mV\"/>"
        "  <pulseGenerator id=\"pulseGen1\" delay=\"10ms\" duration=\"50ms\" amplitude=\"" + amplitude + "\"/>"
        "  <network id=\"Net\">"
        "    <population id=\"Pop\" component=\"glif1CellInstance\" size=\"1\"/>"
        "    <explicitInput target=\"Pop[0]\" input=\"pulseGen1\"/>"
        "  </network>"
        "</neuroml>");

    String top_path = write_temp_file("spikecorec_stimulus_schedule_glif1_" + fixture_id + "_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"StimulusScheduleGlif1Top" + fixture_id + "\">"
        "  <include href=\"spikecorec_stimulus_schedule_glif1_" + fixture_id + "_content.nml\"/>"
        "</neuroml>");

    NML_Parser parser;
    parser.parse(top_path);
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

// A forward-Euler reference integrator for the GLIF1 fixture above --
// mirroring exactly the discretization tests/cell_lowering_tests.cpp already
// pins down for a GLIF1Cell's own `TimeDerivative` (`v += dt*(gL*(EL-v) +
// I)/C`, forward Euler, no `expdecay`). This stands in for the not-yet-wired
// live master-kernel path (ticket #6/#61, see stimulus_schedule.h's doc
// comment) -- it is driven by `stimulus_schedule.h`'s own precomputed
// per-tick current (`I` below), exercising the actual deliverable this
// ticket produces end-to-end against a real GLIF1 cell's dynamics. Per tick:
// detect (test the PREVIOUS tick's post-integration `v` against `vth`) is
// checked before this tick's integrate step, matching arch §2's
// detect-before-next-integrate ordering.
struct ForwardEulerResult {
    Vector<f32> voltage_trace;
    optional<s64> spike_tick;
};

ForwardEulerResult run_glif1_forward_euler(
    f64 capacitance, f64 leak_conductance, f64 resting_potential,
    f64 spike_threshold, f64 reset_potential,
    f64 seconds_step, s64 tick_count, const Vector<f32> &stimulus_current
) {
    ForwardEulerResult result;
    result.voltage_trace.reserve((usize)tick_count);

    f64 membrane_potential = resting_potential; // OnStart: v = EL
    for (s64 tick = 0; tick < tick_count; ++tick) {
        if (!result.spike_tick.has_value() && membrane_potential > spike_threshold) {
            result.spike_tick = tick;
            membrane_potential = reset_potential;
        }
        f64 derivative = (leak_conductance * (resting_potential - membrane_potential) +
                          (f64)stimulus_current[(usize)tick]) / capacitance;
        membrane_potential += seconds_step * derivative;
        result.voltage_trace.push_back((f32)membrane_potential);
    }
    return result;
}

} // namespace

// ── Window construction from a single pulseGenerator ─────────────────────

TEST(StimulusSchedule, builds_window_with_default_weight_when_property_omitted) {
    ModelSpecification specification = build_single_pulse_fixture("default_weight", "10ms", "100ms", "1nA");
    StimulusSchedule schedule = build_stimulus_schedule(specification, /*seconds_step=*/0.001); // 1ms ticks

    ASSERT_EQ(schedule.window_count, 1);
    EXPECT_EQ(schedule.target_neurons[0], 0);
    EXPECT_EQ(schedule.start_ticks[0], 10);
    EXPECT_EQ(schedule.end_ticks[0], 110);
    EXPECT_NEAR(schedule.current_values[0], 1e-9, 1e-15); // amplitude * default weight (1.0)
}

TEST(StimulusSchedule, honors_explicit_weight_property) {
    ModelSpecification specification = build_single_pulse_fixture("explicit_weight", "10ms", "100ms", "1nA", "2");
    StimulusSchedule schedule = build_stimulus_schedule(specification, 0.001);

    ASSERT_EQ(schedule.window_count, 1);
    EXPECT_NEAR(schedule.current_values[0], 2e-9, 1e-15);
}

TEST(StimulusSchedule, treats_zero_delay_as_starting_at_tick_zero) {
    ModelSpecification specification = build_single_pulse_fixture("zero_delay", "0ms", "50ms", "0.5nA");
    StimulusSchedule schedule = build_stimulus_schedule(specification, 0.001);

    ASSERT_EQ(schedule.window_count, 1);
    EXPECT_EQ(schedule.start_ticks[0], 0);
    EXPECT_EQ(schedule.end_ticks[0], 50);
}

TEST(StimulusSchedule, throws_on_non_positive_duration) {
    ModelSpecification specification = build_single_pulse_fixture("zero_duration", "10ms", "0ms", "1nA");
    EXPECT_THROW(build_stimulus_schedule(specification, 0.001), std::runtime_error);
}

TEST(StimulusSchedule, throws_on_unsupported_input_component_type) {
    ModelSpecification specification = build_unsupported_input_fixture();

    // Asserting on the message (not just std::runtime_error) confirms this
    // throws from build_stimulus_schedule()'s own component_type_name check,
    // not from some earlier/unrelated failure (e.g. sineGenerator failing to
    // classify as a Dynamics ComponentType at all).
    try {
        build_stimulus_schedule(specification, 0.001);
        FAIL() << "expected build_stimulus_schedule to throw for a non-pulseGenerator input";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(String(error.what()).find("not a pulseGenerator"), String::npos);
    }
}

// ── current_at() / to_dense_input_spikes() ────────────────────────────────

TEST(StimulusSchedule, current_at_is_zero_outside_every_window) {
    ModelSpecification specification = build_single_pulse_fixture("outside_window", "10ms", "100ms", "1nA");
    StimulusSchedule schedule = build_stimulus_schedule(specification, 0.001);

    EXPECT_NEAR(schedule.current_at(0, 0), 0.0, 1e-15);
    EXPECT_NEAR(schedule.current_at(0, 9), 0.0, 1e-15);
    EXPECT_NEAR(schedule.current_at(0, 10), 1e-9, 1e-15);
    EXPECT_NEAR(schedule.current_at(0, 109), 1e-9, 1e-15);
    EXPECT_NEAR(schedule.current_at(0, 110), 0.0, 1e-15);
    EXPECT_NEAR(schedule.current_at(1, 50), 0.0, 1e-15); // different neuron entirely
}

TEST(StimulusSchedule, sums_current_from_multiple_windows_on_the_same_neuron) {
    ModelSpecification specification = build_two_overlapping_pulses_fixture();
    // seconds_step=1ms: pulseGenA is [0,100) ticks @1nA, pulseGenB is [20,70) ticks @2nA.
    StimulusSchedule schedule = build_stimulus_schedule(specification, 0.001);
    ASSERT_EQ(schedule.window_count, 2);

    EXPECT_NEAR(schedule.current_at(0, 10), 1e-9, 1e-15);       // only pulseGenA active
    EXPECT_NEAR(schedule.current_at(0, 30), 3e-9, 1e-15);       // both active: 1nA + 2nA
    EXPECT_NEAR(schedule.current_at(0, 80), 1e-9, 1e-15);       // pulseGenB has ended, only pulseGenA
    EXPECT_NEAR(schedule.current_at(0, 150), 0.0, 1e-15);       // both have ended
}

TEST(StimulusSchedule, wires_separate_explicit_inputs_to_distinct_non_interfering_target_neurons) {
    ModelSpecification specification = build_two_neuron_targets_fixture();
    StimulusSchedule schedule = build_stimulus_schedule(specification, 0.001); // 1ms ticks
    ASSERT_EQ(schedule.window_count, 2);

    // pulseGenA -> Pop[0] (neuron 0): [5, 25) ticks @ 1nA.
    EXPECT_NEAR(schedule.current_at(0, 4), 0.0, 1e-15);
    EXPECT_NEAR(schedule.current_at(0, 10), 1e-9, 1e-15);
    EXPECT_NEAR(schedule.current_at(0, 25), 0.0, 1e-15);

    // pulseGenB -> Pop[1] (neuron 1): [15, 35) ticks @ 3nA.
    EXPECT_NEAR(schedule.current_at(1, 10), 0.0, 1e-15);
    EXPECT_NEAR(schedule.current_at(1, 20), 3e-9, 1e-15);
    EXPECT_NEAR(schedule.current_at(1, 35), 0.0, 1e-15);

    // Neither neuron sees the other's current, even during their overlap
    // window [15, 25).
    EXPECT_NEAR(schedule.current_at(0, 20), 1e-9, 1e-15);
    EXPECT_NEAR(schedule.current_at(1, 20), 3e-9, 1e-15);
}

TEST(StimulusSchedule, renders_dense_input_spikes_matching_current_at) {
    ModelSpecification specification = build_two_overlapping_pulses_fixture();
    StimulusSchedule schedule = build_stimulus_schedule(specification, 0.001);

    Vector<s32> target_neuron_indices = {0};
    Vector<Vector<f32>> dense = schedule.to_dense_input_spikes(target_neuron_indices, 150);

    ASSERT_EQ(dense.size(), 150u);
    for (s64 tick = 0; tick < 150; ++tick) {
        ASSERT_EQ(dense[(usize)tick].size(), 1u);
        EXPECT_NEAR(dense[(usize)tick][0], (f32)schedule.current_at(0, tick), 1e-15);
    }
}

// ── Acceptance criterion: a current-step protocol drives a GLIF cell ──────
// correctly (ticket #58's own acceptance criterion; ticket #61's future
// validation suite exercises this same shape against the live engine once
// the master kernel is wired -- see stimulus_schedule.h's doc comment for
// why this test drives a hand-written, IR-faithful reference integrator
// instead).
//
// Ground truth used: the CONTINUOUS analytic solution of the leaky
// integrator ODE under a constant current step, not the existing hardcoded
// engine path (`SpikeEngine::step_simulation`/`gpu_step`). Reason: that
// engine path integrates its leak term via an exact exponential-decay
// closed form (`apply_decay`, backend.cpp) every tick, whereas
// tests/cell_lowering_tests.cpp already pins down that the ACTUAL GLIF1
// cell-lowering (#50) that will eventually run in the generated master
// kernel uses plain forward Euler for `v`'s own `TimeDerivative` -- a
// different numerical method. Comparing against the legacy engine would
// therefore validate a different discretization than the one that will
// really execute this cell; the continuous analytic solution is neutral
// ground truth either discretization should converge to, and is also the
// only ground truth available at all right now, since master-kernel
// assembly (#6) isn't merged/wired yet, so no live "GLIF cell run" exists
// to compare against.

TEST(StimulusSchedule, current_step_protocol_drives_subthreshold_glif1_trajectory_matching_analytic_solution) {
    // Sub-threshold: amplitude chosen so the driven steady state
    // (EL + I/gL = -70mV + 0.1nA/1e-8S = -60mV) stays below vth (-50mV).
    ModelSpecification specification = build_glif1_current_step_fixture("subthreshold", "0.1nA");
    const TypeLibraryEntry &cell = type_library_entry_for(specification, "glif1CellInstance");
    f64 capacitance = cell.baked_constants.at("C");
    f64 leak_conductance = cell.baked_constants.at("gL");
    f64 resting_potential = cell.baked_constants.at("EL");
    f64 spike_threshold = cell.baked_constants.at("vth");
    f64 reset_potential = cell.baked_constants.at("vreset");

    f64 seconds_step = 1e-5; // 0.01ms -- small relative to tau=C/gL=10ms
    StimulusSchedule schedule = build_stimulus_schedule(specification, seconds_step);

    s64 tick_count = 6000; // 60ms: covers the whole 10ms delay + 50ms duration window
    Vector<Vector<f32>> dense = schedule.to_dense_input_spikes({0}, tick_count);
    Vector<f32> stimulus_current((usize)tick_count);
    for (s64 tick = 0; tick < tick_count; ++tick) stimulus_current[(usize)tick] = dense[(usize)tick][0];

    ForwardEulerResult result = run_glif1_forward_euler(
        capacitance, leak_conductance, resting_potential, spike_threshold, reset_potential,
        seconds_step, tick_count, stimulus_current);

    EXPECT_FALSE(result.spike_tick.has_value());

    // Analytic v at the end of the pulse window (t = delay + duration = 60ms,
    // i.e. 50ms after the pulse turned on at v(delay) = EL):
    // v(t) = v_inf + (EL - v_inf) * exp(-(t - delay) / tau).
    f64 tau = capacitance / leak_conductance;
    f64 v_inf = resting_potential + 0.1e-9 / leak_conductance;
    f64 analytic_v_at_window_end = v_inf + (resting_potential - v_inf) * std::exp(-0.050 / tau);

    f64 simulated_v_at_window_end = result.voltage_trace.back();
    EXPECT_NEAR(simulated_v_at_window_end, analytic_v_at_window_end, 1e-4); // within 0.1mV
}

TEST(StimulusSchedule, current_step_protocol_drives_glif1_cell_to_spike_at_analytically_predicted_time) {
    // Supra-threshold: driven steady state (EL + I/gL = -70mV + 1nA/1e-8S =
    // +30mV) is well past vth (-50mV), so the cell spikes partway through
    // the pulse.
    ModelSpecification specification = build_glif1_current_step_fixture("spiking", "1nA");
    const TypeLibraryEntry &cell = type_library_entry_for(specification, "glif1CellInstance");
    f64 capacitance = cell.baked_constants.at("C");
    f64 leak_conductance = cell.baked_constants.at("gL");
    f64 resting_potential = cell.baked_constants.at("EL");
    f64 spike_threshold = cell.baked_constants.at("vth");
    f64 reset_potential = cell.baked_constants.at("vreset");

    f64 seconds_step = 1e-5; // 0.01ms
    StimulusSchedule schedule = build_stimulus_schedule(specification, seconds_step);

    s64 tick_count = 6000;
    Vector<Vector<f32>> dense = schedule.to_dense_input_spikes({0}, tick_count);
    Vector<f32> stimulus_current((usize)tick_count);
    for (s64 tick = 0; tick < tick_count; ++tick) stimulus_current[(usize)tick] = dense[(usize)tick][0];

    ForwardEulerResult result = run_glif1_forward_euler(
        capacitance, leak_conductance, resting_potential, spike_threshold, reset_potential,
        seconds_step, tick_count, stimulus_current);

    ASSERT_TRUE(result.spike_tick.has_value());

    // Analytic threshold-crossing time, measured from when the pulse turns
    // on (v(delay) = EL): v(t) = v_inf + (EL - v_inf)*exp(-(t-delay)/tau) = vth
    // => t - delay = -tau * ln((vth - v_inf) / (EL - v_inf)).
    f64 tau = capacitance / leak_conductance;
    f64 v_inf = resting_potential + 1e-9 / leak_conductance;
    f64 seconds_after_pulse_start = -tau * std::log((spike_threshold - v_inf) / (resting_potential - v_inf));
    s64 delay_ticks = 1000; // 10ms / 0.01ms
    f64 analytic_spike_tick = (f64)delay_ticks + seconds_after_pulse_start / seconds_step;

    // Forward Euler's own discretization error is small here (seconds_step/tau
    // = 1e-3), so the actual (integer) spike tick should land within a
    // couple of ticks of the continuous analytic estimate.
    EXPECT_NEAR((f64)result.spike_tick.value(), analytic_spike_tick, 2.0);
}
