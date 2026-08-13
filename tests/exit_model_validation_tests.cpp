#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include "spikecorec/core/engine.h"
#include "spikecorec/core/recording.h"
#include "spikecorec/nml/nml.h"

using namespace std;
using namespace spikecorec;

// ── GLIF single-cell exit models against jNeuroML/pyNeuroML ground truth (#61 [H1]) ──────────
//
// Every other test in this tree is self-referential: it asserts the engine does what we believe
// the engine should do. This file is the one independent check that exists. Each of the four
// fixtures below (tests/fixtures/nml/glif{2,3,4,5}_single_cell.nml) was run through
// jNeuroML/pyNeuroML, and its captured membrane trace and spike raster are checked in under
// tests/fixtures/reference_data/. The SAME .nml file drives both simulators, so a disagreement
// here is a disagreement about the dynamics, not about two different models.
//
// Scope is deliberately the four SINGLE-CELL GLIF models and nothing else. All four are driven by
// a `pulseGenerator` and declare no <projection> at all, so nothing they exercise depends on the
// synapse dynamics currently being reworked. The network fixtures in the same directory
// (glif_ei_network, izhikevich_network, delayed_coupling_network, poisson_population) are left
// alone for that reason.
//
// Fixtures are loaded through the `*_top.nml` wrappers, which <include> the real .nml. The
// LEMS_*.xml runners in the same directory are the jNeuroML entry points and are NOT usable here:
// their <Include file="Cells.xml"/> resolves against the fixture directory rather than the
// vendored standard library, so all six fail to parse.
//
// ── What these comparisons do and do not catch ───────────────────────────────────────────────
// The tolerances are cross-simulator tolerances, not bit-exactness: two independently discretized
// forward-Euler integrators drift by a few ticks around each threshold crossing over a 3500-tick
// run, and demanding identical floats would only ever measure that drift. A spike count within
// +-20%, first/last spike within 5ms, and a membrane sample within 1mV of SOME reference sample
// within 10 ticks is loose enough to absorb that drift and still orders of magnitude tighter than
// a wrong model, a missing mechanism or a units error, each of which misses by a factor rather
// than by a tick. The honest limit: a purely systematic few-tick timing shift is absorbed by
// design and is not caught here.
//
// That the band still discriminates was measured rather than assumed, with two throwaway negative
// controls run against this same harness: comparing glif3's own membrane trace against GLIF5's
// reference (a different but related model) puts 2718 of 3500 samples outside the band, and
// comparing it against glif3's own asc1 column (a current where a voltage belongs -- the shape a
// units error takes) puts all 3500 outside it. Against its own reference, glif3 puts zero outside.

namespace {

String fixture_path(const String &relative_path) {
    return String(SPIKECOREC_TEST_FIXTURES_DIR) + "/" + relative_path;
}

bool standard_library_available() {
    nml::NML_Parser parser;
    return !parser.STANDARD_LIBRARY_PATH.empty() &&
           filesystem::exists(parser.STANDARD_LIBRARY_PATH);
}

// The fixtures name their outputs relatively (`fileName="glif3_membrane_trace.dat"`), so the
// engine's recorders write them into the process's working directory. Each run is therefore
// performed inside its own temporary directory: a test run never writes into the working tree,
// and -- the reason this exists rather than just being tidy -- can never land on top of
// tests/fixtures/reference_data/'s own identically-named ground truth.
class ScopedWorkingDirectory {
public:
    explicit ScopedWorkingDirectory(const String &run_name) {
        previous_directory_ = filesystem::current_path();
        run_directory_ = filesystem::temp_directory_path() / "spikecorec_exit_model_runs" / run_name;
        filesystem::remove_all(run_directory_);
        filesystem::create_directories(run_directory_);
        filesystem::current_path(run_directory_);
    }

    ~ScopedWorkingDirectory() {
        std::error_code ignored;
        filesystem::current_path(previous_directory_, ignored);
        filesystem::remove_all(run_directory_, ignored);
    }

    ScopedWorkingDirectory(const ScopedWorkingDirectory &) = delete;
    ScopedWorkingDirectory &operator=(const ScopedWorkingDirectory &) = delete;

private:
    filesystem::path previous_directory_;
    filesystem::path run_directory_;
};

// ── running a fixture ────────────────────────────────────────────────────────────────────────

struct SingleCellRun {
    // The model's own <OutputFile> columns, in the order the fixture declares them, one sample per
    // tick: [column][tick]. Column 0 is `v` in all four fixtures.
    spikecorec::Vector<spikecorec::Vector<f32>> recorded_columns;

    // Ticks at which the model's own <EventOutputFile> recorded a spike.
    spikecorec::Vector<s64> spike_ticks;

    s64 tick_count = 0;
    f64 step_dt = 0.0;
};

// Drives one fixture end to end through the real engine: parse -> resolve -> codegen -> compile ->
// run -> record, then reads the recordings back out of the files the engine itself wrote. Nothing
// is seeded or reconstructed by hand -- the OnStart bodies, the pulseGenerator schedule and the
// recording selections all come from the .nml.
SingleCellRun run_single_cell_fixture(const String &fixture_base_name) {
    ScopedWorkingDirectory run_directory(fixture_base_name);

    String model_path = fixture_path("nml/" + fixture_base_name + "_top.nml");

    SingleCellRun run;
    String membrane_trace_filename;
    String spike_events_filename;

    {
        SpikeEngine engine(model_path, /*enable_hebbian_learning=*/false);

        run.tick_count = engine.lifetime;
        run.step_dt = engine.network_details.step_dt;

        for (const RecordingConfig &recording_profile : engine.recording_profiles) {
            for (usize output_index = 0; output_index < recording_profile.output_filenames.size();
                 output_index += 1) {
                const bool is_spike_event_file =
                        output_index < recording_profile.file_output_format.size() &&
                        recording_profile.file_output_format[output_index] ==
                                OutputFileFormat::SPIKE_EVENTS;
                if (is_spike_event_file) {
                    spike_events_filename = recording_profile.output_filenames[output_index];
                } else {
                    membrane_trace_filename = recording_profile.output_filenames[output_index];
                }
            }
        }

        for (s64 tick = 0; tick < engine.lifetime; tick += 1) engine.step_simulation(tick);

        // Closes the recorders, flushing every buffered frame -- the files are only complete
        // after this, and the engine is destroyed at the end of this scope anyway.
        engine.shutdown();
    }

    if (membrane_trace_filename.empty() || spike_events_filename.empty()) {
        throw std::runtime_error("exit_model_validation_tests: fixture '" + fixture_base_name +
                                 "' declared no membrane-trace and/or spike-event output file");
    }

    const SpireRecording membrane_recording = read_spire_recording(membrane_trace_filename);
    run.recorded_columns.assign((usize)membrane_recording.neuron_count, spikecorec::Vector<f32>());
    for (spikecorec::Vector<f32> &column : run.recorded_columns) column.reserve((usize)membrane_recording.frame_count);
    for (s64 frame_index = 0; frame_index < membrane_recording.frame_count; frame_index += 1) {
        for (s64 column_index = 0; column_index < membrane_recording.neuron_count; column_index += 1) {
            run.recorded_columns[(usize)column_index].push_back(
                    membrane_recording.frames[(usize)(frame_index * membrane_recording.neuron_count +
                                                      column_index)]);
        }
    }

    // A spike-event stream records each selected neuron's spike flag, one frame per tick, so a
    // nonzero sample is exactly "this neuron emitted on this tick".
    const SpireRecording spike_recording = read_spire_recording(spike_events_filename);
    for (s64 frame_index = 0; frame_index < spike_recording.frame_count; frame_index += 1) {
        for (s64 column_index = 0; column_index < spike_recording.neuron_count; column_index += 1) {
            if (spike_recording.frames[(usize)(frame_index * spike_recording.neuron_count +
                                               column_index)] != 0.0f) {
                run.spike_ticks.push_back(frame_index);
            }
        }
    }

    return run;
}

// ── reference data (jLEMS OutputFile / EventOutputFile TIME_ID plain text) ───────────────────
//
// An OutputFile row is "<time_seconds> <column0> <column1> ... <columnN>"; an EventOutputFile row
// (format="TIME_ID") is "<time_seconds> <selection_id>". Both are whitespace separated.

struct ReferenceTrace {
    spikecorec::Vector<f64> time_seconds;
    spikecorec::Vector<spikecorec::Vector<f64>> column_values; // column_values[row_index][column_index]
};

ReferenceTrace load_reference_trace(const String &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("exit_model_validation_tests: could not open reference trace file '" +
                                 path + "'");
    }

    ReferenceTrace trace;
    String line;
    while (std::getline(file, line)) {
        if (line.find_first_not_of(" \t\r\n") == String::npos) continue;
        std::istringstream line_stream(line);
        f64 time_value = 0.0;
        line_stream >> time_value;
        trace.time_seconds.push_back(time_value);

        spikecorec::Vector<f64> row_values;
        f64 value = 0.0;
        while (line_stream >> value) row_values.push_back(value);
        trace.column_values.push_back(std::move(row_values));
    }
    return trace;
}

struct ReferenceSpikeRecord {
    f64 time_seconds = 0.0;
    String selection_id;
};

spikecorec::Vector<ReferenceSpikeRecord> load_reference_spikes(const String &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("exit_model_validation_tests: could not open reference spikes file '" +
                                 path + "'");
    }

    spikecorec::Vector<ReferenceSpikeRecord> spikes;
    String line;
    while (std::getline(file, line)) {
        if (line.find_first_not_of(" \t\r\n") == String::npos) continue;
        std::istringstream line_stream(line);
        ReferenceSpikeRecord record;
        line_stream >> record.time_seconds >> record.selection_id;
        spikes.push_back(std::move(record));
    }
    return spikes;
}

spikecorec::Vector<f64> extract_spike_time_seconds(const spikecorec::Vector<ReferenceSpikeRecord> &spikes) {
    spikecorec::Vector<f64> times;
    times.reserve(spikes.size());
    for (const ReferenceSpikeRecord &spike : spikes) times.push_back(spike.time_seconds);
    return times;
}

// ── cross-simulator comparison ───────────────────────────────────────────────────────────────

const f64 SPIKE_COUNT_RELATIVE_TOLERANCE = 0.2;     // own spike count within +-20% of the reference's
const f64 SPIKE_EDGE_TIME_TOLERANCE_SECONDS = 5e-3; // first/last spike within 5ms of the reference's

// Reference row `tick + 1` is this tick's post-integration sample: row 0 is the t=0 initial
// condition jLEMS writes before stepping, whereas the engine records after each step_simulation.
s64 reference_row_for_tick(s64 tick) { return tick + 1; }

// The same convention in seconds. The engine's spike flag for a tick is recorded in the same frame
// as that tick's membrane sample, so it denotes the same instant that frame does: the END of the
// tick, t = (tick + 1) * dt. Converting with `tick * dt` instead would report every spike a tick
// early against a reference whose own first row is t=0.
f64 spike_time_seconds(s64 tick, f64 step_dt) { return (f64)(tick + 1) * step_dt; }

void expect_spike_train_roughly_matches_reference(const spikecorec::Vector<s64> &own_spike_ticks,
                                                  const spikecorec::Vector<f64> &reference_spike_times_seconds,
                                                  f64 step_dt, const String &label) {
    ASSERT_FALSE(reference_spike_times_seconds.empty())
            << label << ": reference has no spikes to compare against";

    const f64 reference_count = (f64)reference_spike_times_seconds.size();
    const f64 own_count = (f64)own_spike_ticks.size();
    EXPECT_LE(std::fabs(own_count - reference_count),
              std::max(1.0, reference_count * SPIKE_COUNT_RELATIVE_TOLERANCE))
            << label << ": own spike count=" << own_spike_ticks.size()
            << " reference spike count=" << reference_spike_times_seconds.size();
    if (own_spike_ticks.empty()) return; // the count check above already reports this

    const f64 own_first_spike_time_seconds = spike_time_seconds(own_spike_ticks.front(), step_dt);
    const f64 own_last_spike_time_seconds = spike_time_seconds(own_spike_ticks.back(), step_dt);
    EXPECT_NEAR(own_first_spike_time_seconds, reference_spike_times_seconds.front(),
                SPIKE_EDGE_TIME_TOLERANCE_SECONDS)
            << label << ": first spike time";
    EXPECT_NEAR(own_last_spike_time_seconds, reference_spike_times_seconds.back(),
                SPIKE_EDGE_TIME_TOLERANCE_SECONDS)
            << label << ": last spike time";
}

// Smallest distance between `own_value` and the reference column over a window of `tick_shift_radius`
// ticks either side of this tick's own reference row. The window absorbs a bounded timing shift --
// two independently discretized simulators' reset/refractory-exit instants land a few ticks apart,
// which would otherwise make every sample around every spike fail even when the dynamics agree. It
// does not absorb a magnitude-class error: a value produced by the wrong equation, the wrong sign or
// the wrong scale lands near no nearby reference sample either.
f64 nearest_reference_deviation(f32 own_value, const ReferenceTrace &reference_trace, s64 tick,
                                usize column_index, s64 tick_shift_radius) {
    f64 smallest_deviation = std::numeric_limits<f64>::infinity();
    for (s64 shift = -tick_shift_radius; shift <= tick_shift_radius; shift += 1) {
        const s64 reference_row = reference_row_for_tick(tick) + shift;
        if (reference_row < 0 || (usize)reference_row >= reference_trace.column_values.size()) continue;
        if (column_index >= reference_trace.column_values[(usize)reference_row].size()) continue;

        const f64 reference_value = reference_trace.column_values[(usize)reference_row][column_index];
        smallest_deviation = std::min(smallest_deviation, std::fabs((f64)own_value - reference_value));
    }
    return smallest_deviation;
}

struct ColumnComparison {
    f64 peak_aligned_deviation = 0.0; // largest |own - reference| at the SAME tick
    f64 rms_aligned_deviation = 0.0;  // root-mean-square of the same
    f64 peak_shifted_deviation = 0.0; // largest deviation after the tick-shift search

    // Reported, never asserted on: how many samples would fail without the tick-shift search at
    // all. It is what says how much of the agreement below is the dynamics agreeing and how much
    // is the search absorbing a reset landing one tick apart.
    s64 sample_count_outside_tolerance_aligned = 0;

    s64 sample_count_outside_tolerance = 0;
    s64 first_tick_outside_tolerance = -1;
    f32 own_value_at_first_failure = 0.0f;
};

ColumnComparison compare_column_against_reference(const spikecorec::Vector<f32> &own_column,
                                                  const ReferenceTrace &reference_trace,
                                                  usize column_index, f64 tolerance,
                                                  s64 tick_shift_radius) {
    ColumnComparison comparison;
    f64 squared_deviation_total = 0.0;
    s64 compared_sample_count = 0;

    for (usize tick = 0; tick < own_column.size(); tick += 1) {
        const s64 reference_row = reference_row_for_tick((s64)tick);
        if (reference_row >= 0 && (usize)reference_row < reference_trace.column_values.size() &&
            column_index < reference_trace.column_values[(usize)reference_row].size()) {
            const f64 aligned_deviation =
                    std::fabs((f64)own_column[tick] -
                              reference_trace.column_values[(usize)reference_row][column_index]);
            comparison.peak_aligned_deviation =
                    std::max(comparison.peak_aligned_deviation, aligned_deviation);
            squared_deviation_total += aligned_deviation * aligned_deviation;
            compared_sample_count += 1;
            if (aligned_deviation > tolerance) comparison.sample_count_outside_tolerance_aligned += 1;
        }

        const f64 shifted_deviation = nearest_reference_deviation(
                own_column[tick], reference_trace, (s64)tick, column_index, tick_shift_radius);
        comparison.peak_shifted_deviation =
                std::max(comparison.peak_shifted_deviation, shifted_deviation);
        if (shifted_deviation > tolerance) {
            comparison.sample_count_outside_tolerance += 1;
            if (comparison.first_tick_outside_tolerance < 0) {
                comparison.first_tick_outside_tolerance = (s64)tick;
                comparison.own_value_at_first_failure = own_column[tick];
            }
        }
    }

    if (compared_sample_count > 0) {
        comparison.rms_aligned_deviation = std::sqrt(squared_deviation_total / (f64)compared_sample_count);
    }
    return comparison;
}

// Measured numbers go to stdout for every model, passing or failing: this suite exists to produce a
// measurement, and a bare "OK" reports nothing about how close the engine actually is.
void report_column_comparison(const String &label, const String &column_name,
                              const ColumnComparison &comparison, f64 tolerance) {
    std::cout << "[ MEASURED ] " << label << " " << column_name
              << ": peak_aligned_deviation=" << comparison.peak_aligned_deviation
              << " rms_aligned_deviation=" << comparison.rms_aligned_deviation
              << " peak_deviation_after_tick_shift=" << comparison.peak_shifted_deviation
              << " samples_outside_tolerance_same_tick=" << comparison.sample_count_outside_tolerance_aligned
              << " samples_outside_tolerance=" << comparison.sample_count_outside_tolerance
              << " (tolerance=" << tolerance << ")" << std::endl;
}

void report_spike_train(const String &label, const spikecorec::Vector<s64> &own_spike_ticks,
                        const spikecorec::Vector<f64> &reference_spike_times_seconds, f64 step_dt) {
    std::cout << "[ MEASURED ] " << label << " spikes: own_count=" << own_spike_ticks.size()
              << " reference_count=" << reference_spike_times_seconds.size();
    if (!own_spike_ticks.empty() && !reference_spike_times_seconds.empty()) {
        std::cout << " own_first=" << spike_time_seconds(own_spike_ticks.front(), step_dt)
                  << "s reference_first=" << reference_spike_times_seconds.front()
                  << "s own_last=" << spike_time_seconds(own_spike_ticks.back(), step_dt)
                  << "s reference_last=" << reference_spike_times_seconds.back() << "s";
    }
    std::cout << std::endl;
}

// Everything a per-model test needs, gathered once.
struct ModelComparisonInputs {
    SingleCellRun run;
    ReferenceTrace reference_trace;
    spikecorec::Vector<ReferenceSpikeRecord> reference_spikes;
};

ModelComparisonInputs load_model_comparison_inputs(const String &fixture_base_name,
                                                   const String &reference_prefix) {
    ModelComparisonInputs inputs;
    inputs.run = run_single_cell_fixture(fixture_base_name);
    inputs.reference_trace = load_reference_trace(fixture_path(
            "reference_data/" + fixture_base_name + "/" + reference_prefix + "_membrane_trace.dat"));
    inputs.reference_spikes = load_reference_spikes(
            fixture_path("reference_data/" + fixture_base_name + "/" + reference_prefix + "_spikes.dat"));
    return inputs;
}

// 1mV, the resolution at which two forward-Euler integrators of the same GLIF cell can be said to
// agree; the deflections these models make are tens of mV.
const f64 VOLTAGE_TOLERANCE_VOLTS = 1e-3;

// 10 ticks = 1ms at these fixtures' own 0.1ms step. Covers the reset-instant drift between the two
// simulators without being wide enough to match an unrelated part of the trajectory.
const s64 TICK_SHIFT_RADIUS = 10;

void expect_membrane_trace_matches_reference(const ModelComparisonInputs &inputs, const String &label) {
    ASSERT_FALSE(inputs.run.recorded_columns.empty()) << label << ": nothing was recorded";
    // One recorded frame per tick, and 350ms at 0.1ms -> 3500 engine ticks against 3501 reference
    // rows (the extra row is jLEMS's t=0 initial condition, written before it steps).
    ASSERT_EQ(inputs.run.recorded_columns[0].size(), (usize)inputs.run.tick_count);
    ASSERT_EQ(inputs.run.recorded_columns[0].size() + 1, inputs.reference_trace.time_seconds.size());

    const ColumnComparison comparison =
            compare_column_against_reference(inputs.run.recorded_columns[0], inputs.reference_trace,
                                             /*column_index=*/0, VOLTAGE_TOLERANCE_VOLTS,
                                             TICK_SHIFT_RADIUS);
    report_column_comparison(label, "v", comparison, VOLTAGE_TOLERANCE_VOLTS);

    EXPECT_EQ(comparison.sample_count_outside_tolerance, 0)
            << label << ": membrane trace leaves the tolerance band at " << comparison.sample_count_outside_tolerance
            << " of " << inputs.run.recorded_columns[0].size() << " ticks; first at tick "
            << comparison.first_tick_outside_tolerance << " (own_v=" << comparison.own_value_at_first_failure
            << ", peak deviation after tick shift=" << comparison.peak_shifted_deviation << ")";
}

void expect_spike_train_matches_reference(const ModelComparisonInputs &inputs, const String &label) {
    const spikecorec::Vector<f64> reference_spike_times = extract_spike_time_seconds(inputs.reference_spikes);
    report_spike_train(label, inputs.run.spike_ticks, reference_spike_times, inputs.run.step_dt);
    expect_spike_train_roughly_matches_reference(inputs.run.spike_ticks, reference_spike_times,
                                                 inputs.run.step_dt, label);
}

} // namespace

// ── GLIF2: scaled reset ──────────────────────────────────────────────────────────────────────

TEST(ExitModelGlif2SingleCell, matches_pyneuroml_reference) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    const ModelComparisonInputs inputs = load_model_comparison_inputs("glif2_single_cell", "glif2");
    expect_spike_train_matches_reference(inputs, "glif2_single_cell");
    expect_membrane_trace_matches_reference(inputs, "glif2_single_cell");
}

// ── GLIF3: after-spike currents ──────────────────────────────────────────────────────────────

TEST(ExitModelGlif3SingleCell, matches_pyneuroml_reference) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    const ModelComparisonInputs inputs = load_model_comparison_inputs("glif3_single_cell", "glif3");
    expect_spike_train_matches_reference(inputs, "glif3_single_cell");
    expect_membrane_trace_matches_reference(inputs, "glif3_single_cell");
}

// The mechanism GLIF3 exists to demonstrate, checked against the reference rather than only against
// itself: asc1/asc2 are recorded columns 1 and 2 of the fixture's own <OutputFile>. Their tolerance
// is 10% of each one's own spike-time bump (ascAdd1=-100pA, ascAdd2=-200pA), the scale at which two
// simulators can be said to agree about a current that decays by orders of magnitude between spikes.
TEST(ExitModelGlif3SingleCell, after_spike_currents_match_reference) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    const ModelComparisonInputs inputs = load_model_comparison_inputs("glif3_single_cell", "glif3");
    ASSERT_EQ(inputs.run.recorded_columns.size(), 3u); // v, asc1, asc2

    const f64 first_current_tolerance = 1e-11;  // 10pA
    const f64 second_current_tolerance = 2e-11; // 20pA

    const ColumnComparison first_current_comparison = compare_column_against_reference(
            inputs.run.recorded_columns[1], inputs.reference_trace, /*column_index=*/1,
            first_current_tolerance, TICK_SHIFT_RADIUS);
    report_column_comparison("glif3_single_cell", "asc1", first_current_comparison,
                             first_current_tolerance);
    EXPECT_EQ(first_current_comparison.sample_count_outside_tolerance, 0)
            << "glif3 asc1 leaves the tolerance band; first at tick "
            << first_current_comparison.first_tick_outside_tolerance
            << " (own asc1=" << first_current_comparison.own_value_at_first_failure << ")";

    const ColumnComparison second_current_comparison = compare_column_against_reference(
            inputs.run.recorded_columns[2], inputs.reference_trace, /*column_index=*/2,
            second_current_tolerance, TICK_SHIFT_RADIUS);
    report_column_comparison("glif3_single_cell", "asc2", second_current_comparison,
                             second_current_tolerance);
    EXPECT_EQ(second_current_comparison.sample_count_outside_tolerance, 0)
            << "glif3 asc2 leaves the tolerance band; first at tick "
            << second_current_comparison.first_tick_outside_tolerance
            << " (own asc2=" << second_current_comparison.own_value_at_first_failure << ")";

    // Non-vacuity: the currents genuinely take on the hyperpolarizing values the spikes add, rather
    // than agreeing with the reference by both staying at zero.
    f32 most_negative_first_current = 0.0f;
    for (f32 value : inputs.run.recorded_columns[1]) {
        most_negative_first_current = std::min(most_negative_first_current, value);
    }
    f32 most_negative_second_current = 0.0f;
    for (f32 value : inputs.run.recorded_columns[2]) {
        most_negative_second_current = std::min(most_negative_second_current, value);
    }
    EXPECT_LT(most_negative_first_current, -5e-11f)  // ascAdd1 = -100pA, accumulated across spikes
            << "glif3's first after-spike current never became hyperpolarizing";
    EXPECT_LT(most_negative_second_current, -1e-10f) // ascAdd2 = -200pA
            << "glif3's second after-spike current never became hyperpolarizing";
}

// ── GLIF4: adapting threshold ────────────────────────────────────────────────────────────────

TEST(ExitModelGlif4SingleCell, matches_pyneuroml_reference) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    const ModelComparisonInputs inputs = load_model_comparison_inputs("glif4_single_cell", "glif4");
    expect_spike_train_matches_reference(inputs, "glif4_single_cell");
    expect_membrane_trace_matches_reference(inputs, "glif4_single_cell");
}

TEST(ExitModelGlif4SingleCell, adaptive_threshold_matches_reference) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    const ModelComparisonInputs inputs = load_model_comparison_inputs("glif4_single_cell", "glif4");
    ASSERT_EQ(inputs.run.recorded_columns.size(), 2u); // v, theta

    const ColumnComparison threshold_comparison = compare_column_against_reference(
            inputs.run.recorded_columns[1], inputs.reference_trace, /*column_index=*/1,
            VOLTAGE_TOLERANCE_VOLTS, TICK_SHIFT_RADIUS);
    report_column_comparison("glif4_single_cell", "theta", threshold_comparison,
                             VOLTAGE_TOLERANCE_VOLTS);
    EXPECT_EQ(threshold_comparison.sample_count_outside_tolerance, 0)
            << "glif4 theta leaves the tolerance band; first at tick "
            << threshold_comparison.first_tick_outside_tolerance
            << " (own theta=" << threshold_comparison.own_value_at_first_failure << ")";

    // Non-vacuity: the threshold actually adapts. thetaInf is -50mV and each spike adds 5mV, so a
    // threshold that never rises above thetaInf means the mechanism did not run at all.
    f32 highest_threshold = -std::numeric_limits<f32>::infinity();
    for (f32 value : inputs.run.recorded_columns[1]) highest_threshold = std::max(highest_threshold, value);
    EXPECT_GT(highest_threshold, -0.0475f) << "glif4's threshold never rose above thetaInf + 2.5mV";
}

// ── GLIF5: after-spike currents AND adapting threshold ───────────────────────────────────────

TEST(ExitModelGlif5SingleCell, matches_pyneuroml_reference) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    const ModelComparisonInputs inputs = load_model_comparison_inputs("glif5_single_cell", "glif5");
    expect_spike_train_matches_reference(inputs, "glif5_single_cell");
    expect_membrane_trace_matches_reference(inputs, "glif5_single_cell");
}

TEST(ExitModelGlif5SingleCell, adaptive_threshold_and_after_spike_currents_match_reference) {
    if (!standard_library_available()) GTEST_SKIP() << "NML standard library not vendored";

    const ModelComparisonInputs inputs = load_model_comparison_inputs("glif5_single_cell", "glif5");
    ASSERT_EQ(inputs.run.recorded_columns.size(), 4u); // v, theta, asc1, asc2

    const f64 first_current_tolerance = 1e-11;  // 10pA, 10% of ascAdd1
    const f64 second_current_tolerance = 2e-11; // 20pA, 10% of ascAdd2

    const ColumnComparison threshold_comparison = compare_column_against_reference(
            inputs.run.recorded_columns[1], inputs.reference_trace, /*column_index=*/1,
            VOLTAGE_TOLERANCE_VOLTS, TICK_SHIFT_RADIUS);
    report_column_comparison("glif5_single_cell", "theta", threshold_comparison,
                             VOLTAGE_TOLERANCE_VOLTS);
    EXPECT_EQ(threshold_comparison.sample_count_outside_tolerance, 0)
            << "glif5 theta leaves the tolerance band; first at tick "
            << threshold_comparison.first_tick_outside_tolerance
            << " (own theta=" << threshold_comparison.own_value_at_first_failure << ")";

    const ColumnComparison first_current_comparison = compare_column_against_reference(
            inputs.run.recorded_columns[2], inputs.reference_trace, /*column_index=*/2,
            first_current_tolerance, TICK_SHIFT_RADIUS);
    report_column_comparison("glif5_single_cell", "asc1", first_current_comparison,
                             first_current_tolerance);
    EXPECT_EQ(first_current_comparison.sample_count_outside_tolerance, 0)
            << "glif5 asc1 leaves the tolerance band; first at tick "
            << first_current_comparison.first_tick_outside_tolerance
            << " (own asc1=" << first_current_comparison.own_value_at_first_failure << ")";

    const ColumnComparison second_current_comparison = compare_column_against_reference(
            inputs.run.recorded_columns[3], inputs.reference_trace, /*column_index=*/3,
            second_current_tolerance, TICK_SHIFT_RADIUS);
    report_column_comparison("glif5_single_cell", "asc2", second_current_comparison,
                             second_current_tolerance);
    EXPECT_EQ(second_current_comparison.sample_count_outside_tolerance, 0)
            << "glif5 asc2 leaves the tolerance band; first at tick "
            << second_current_comparison.first_tick_outside_tolerance
            << " (own asc2=" << second_current_comparison.own_value_at_first_failure << ")";

    // Non-vacuity: both mechanisms genuinely act, which is the whole point of GLIF5.
    f32 highest_threshold = -std::numeric_limits<f32>::infinity();
    for (f32 value : inputs.run.recorded_columns[1]) highest_threshold = std::max(highest_threshold, value);
    EXPECT_GT(highest_threshold, -0.0475f) << "glif5's threshold never rose above thetaInf + 2.5mV";

    f32 most_negative_first_current = 0.0f;
    for (f32 value : inputs.run.recorded_columns[2]) {
        most_negative_first_current = std::min(most_negative_first_current, value);
    }
    f32 most_negative_second_current = 0.0f;
    for (f32 value : inputs.run.recorded_columns[3]) {
        most_negative_second_current = std::min(most_negative_second_current, value);
    }
    EXPECT_LT(most_negative_first_current, -5e-11f)
            << "glif5's first after-spike current never became hyperpolarizing";
    EXPECT_LT(most_negative_second_current, -1e-10f)
            << "glif5's second after-spike current never became hyperpolarizing";
}
