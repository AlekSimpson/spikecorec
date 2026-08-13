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
// ── WHAT IS ACTUALLY ASSERTED, AND HOW TIGHT IT REALLY IS ────────────────────────────────────
// Both simulators run the SAME model at the SAME 0.1ms step with the SAME forward-Euler scheme.
// They are therefore not merely expected to agree "closely" -- they are expected to agree
// TICK FOR TICK. That is what is asserted, with no tick-shift search anywhere in the verdict:
//
//   * the spike count is EXACTLY the reference's,
//   * EVERY spike lands on the reference's exact tick -- first, last, and every one between,
//   * EVERY recorded sample is within its column's band of the reference sample for the SAME
//     instant: 1mV on `v` and on the adapting threshold `theta`, 10pA on asc1 and 20pA on asc2
//     (10% of each after-spike current's own spike-time bump),
//   * every refractory hold is exactly as many samples long as the reference's.
//
// The measured result, once the engine defect described below is fixed, is ZERO samples outside
// those bands out of the 3499 compared per column, on all four models, and every spike on the
// reference's exact tick. The bands above are ceilings, not the achieved agreement: the largest
// same-tick deviation anywhere in the four runs is 11nV on `v` (glif3), 24uV on `theta` (glif5),
// 0.42pA on asc1 and 2.6pA on asc2 -- four to five orders of magnitude inside the asserted band
// on `v`.
//
// `v` reads 11nV rather than the 600uV this comment carried before because of a SECOND engine
// defect, since fixed: <pulseGenerator> wrote its injection window with an exclusive end tick
// into an event stream filled inclusively, so the pulse ran one tick long. That extra tick is
// I*dt/C of membrane voltage delivered after jNeuroML's pulse had already stopped -- 600uV on
// glif3 and glif4's 0.6nA/100pF, 220uV on glif2's 0.22nA, 250uV on glif4's 0.25nA -- and it was
// the entire `v` disagreement on three of the four models. glif5 read the numeric floor even
// before the fix only by coincidence: its last reference spike is at t=0.3171s and t_ref is 5ms,
// so the extra tick at 3200 landed inside a refractory hold with `v` pinned at vreset, where the
// current had nowhere to go.
//
// An earlier version of this file asserted something much weaker and advertised it as 1mV: it
// took, per sample, the SMALLEST deviation against any reference row within +-10 ticks. That
// accepts any value in the reference's own range over those 21 rows, widened by 1mV on each side.
// Measured on this reference data, that band is 3.2mV wide at the median and 22.0mV at the widest
// on glif2, 4.1mV and 29.1mV on glif5, and it exceeds 15mV on 5.7% and 6.9% of samples
// respectively -- so the advertised 1mV was really 15-29mV wherever the trace moved fast, which is
// precisely around every spike. A model drifting by up to 10 ticks scored a perfect zero.
//
// The tick-shift search survives ONLY as a printed diagnostic (`best_uniform_shift`, below):
// applied uniformly to the whole trace rather than per sample, it says whether a failure is "the
// right trajectory, N ticks off" or "the wrong trajectory", and the same figure over the first and
// last third of the run says whether that offset is DRIFTING. None of those three numbers gates a
// test.
//
// ── SAMPLE ALIGNMENT: own[tick] == reference row (tick + 2) ──────────────────────────────────
// Pinned by the stimulus onset, which is the one instant in the run whose timing depends on no
// cell parameter at all. The pulse is declared `delay="20ms"`, so at 0.1ms the engine's first
// deflected sample is own[200] and the reference's first deflected row is 202, and they agree to
// 7 significant figures on all four models (own[200]=-0.06978 == glif2 row 202=-0.06978,
// -0.069399998 == glif3 row 202=-0.0694, -0.069750004 == glif4 row 202=-0.06975). Two rows, not
// one: jLEMS writes a t=0 initial condition before it steps, AND applies a pulseGenerator's
// current on the step AFTER its OnCondition fires, where the engine applies it on the same step.
// The previous `tick + 1` was a systematic one-tick misalignment -- invisible only because the
// +-10 tick-shift search absorbed it. Under it, the four models' same-tick disagreement reads
// 238/10/36/1 samples; under the correct alignment it reads 325/19/81/11, and after the fix
// below it reads 0/0/0/0.
//
// The spike raster keeps its own convention: jLEMS timestamps the event one row BEFORE the row
// carrying the post-reset voltage, so a spike recorded in engine frame `f` is the reference's
// spike at tick `f + 1` -- the same row `f + 2` the membrane comparison uses.
//
// ── KNOWN ENGINE DEFECT THESE ASSERTIONS CURRENTLY FAIL ON ───────────────────────────────────
// The regime counter accumulates dt in f32, so after 50 steps `refractoryTimeElapsed` reaches
// 4.9999989e-3 against a 5e-3 threshold and the refractory->integrating transition fires one tick
// late. Every GLIF refractory hold is therefore 52 samples where jNeuroML's is 51, every ISI is
// one tick long, and the error accumulates over the run: glif2's spikes fall 0,1,2...9 ticks
// behind the reference's, and 325 of 3499 membrane samples land outside 1mV. The assertions in
// this file are written against what is CORRECT, not against that drift, so seven of them fail
// until it is fixed.
//
// That the fix is sufficient -- that nothing ELSE disagrees underneath it -- was measured, by
// running each fixture with t_ref one tick shorter, which produces exactly the tick counts a fixed
// engine produces at the declared 5ms. All four models then match the reference on every spike
// tick, every refractory hold and every sample of every recorded column, at zero shift.
//
// ── NEGATIVE CONTROLS ────────────────────────────────────────────────────────────────────────
// That the bands above discriminate at all is not assumed, it is a standing test. Two controls
// feed this same harness a deliberately wrong pairing (ExitModelNegativeControls) and three feed
// it a copy of glif2 with exactly one attribute perturbed by 2% (ExitModelPerturbationControls),
// and each asserts the harness REJECTS it. Without them a harness can go vacuous -- a tolerance
// widened, a comparison silently comparing a trace against itself -- and still report green.

namespace {

String fixture_path(const String &relative_path) {
    return String(SPIKECOREC_TEST_FIXTURES_DIR) + "/" + relative_path;
}

// The standard library is CHECKED IN under third_party/neuroml2/std_lib, not fetched, so its
// absence is a broken checkout rather than a configuration this suite may skip over. Skipping
// would silently retract every external check in this file and still report green -- the exact
// failure mode an exit-model suite exists to prevent -- so it is a hard failure instead.
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
//
// ── why every construction gets a directory NOBODY has used before ───────────────────────────
// Several tests run the same fixture, so `run_name` repeats: run_single_cell_fixture passes the
// fixture's base name, and "glif3_single_cell" alone is used by four of them. Building the path
// out of the name alone therefore meant destroying and recreating ONE path over and over inside
// a single process, with a chdir into it each time. On APFS a chdir into a directory that was
// just unlinked and recreated can leave the process attached to the doomed inode: the recorder's
// writes then land somewhere the subsequent open-for-read cannot see, and the run fails with
// "Failed to open file for reading: glif3_membrane_trace.dat" -- intermittently, at roughly one
// run in ten, and never as a numeric disagreement.
//
// A monotonic suffix removes the remove-then-recreate step entirely: create_directories always
// makes a directory that did not exist, so there is no unlinked inode to be attached to. The
// chdir stays -- it is load-bearing until the engine takes an explicit output root -- and so
// does the cleanup, which now runs against a path that will never be handed out again.
class ScopedWorkingDirectory {
public:
    explicit ScopedWorkingDirectory(const String &run_name) {
        previous_directory_ = filesystem::current_path();
        run_directory_ = runs_root() / (run_name + "_" + std::to_string(next_sequence_number()));
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
    // Cleared ONCE per process, before the first run and while the working directory is still
    // outside it, so a previous process that died mid-run leaves nothing behind. Removing it
    // here rather than per run is the whole point: nothing ever unlinks a directory this
    // process has chdir'd into.
    static const filesystem::path &runs_root() {
        static const filesystem::path root = [] {
            const filesystem::path path =
                    filesystem::temp_directory_path() / "spikecorec_exit_model_runs";
            std::error_code ignored;
            filesystem::remove_all(path, ignored);
            filesystem::create_directories(path);
            return path;
        }();
        return root;
    }

    static s64 next_sequence_number() {
        static s64 sequence_number = 0;
        return ++sequence_number;
    }

    filesystem::path previous_directory_;
    filesystem::path run_directory_;
};

// Where a generated fixture lives for the length of one test. Separate from ScopedWorkingDirectory
// because a perturbed fixture has to SURVIVE the run that reads it, and ScopedWorkingDirectory
// wipes its directory on construction.
class ScopedTemporaryDirectory {
public:
    explicit ScopedTemporaryDirectory(const String &directory_name) {
        directory_ = filesystem::temp_directory_path() / "spikecorec_exit_model_perturbations" /
                     directory_name;
        filesystem::remove_all(directory_);
        filesystem::create_directories(directory_);
    }

    ~ScopedTemporaryDirectory() {
        std::error_code ignored;
        filesystem::remove_all(directory_, ignored);
    }

    ScopedTemporaryDirectory(const ScopedTemporaryDirectory &) = delete;
    ScopedTemporaryDirectory &operator=(const ScopedTemporaryDirectory &) = delete;

    const filesystem::path &path() const { return directory_; }

private:
    filesystem::path directory_;
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

// Drives one model end to end through the real engine: parse -> resolve -> codegen -> compile ->
// run -> record, then reads the recordings back out of the files the engine itself wrote. Nothing
// is seeded or reconstructed by hand -- the OnStart bodies, the pulseGenerator schedule and the
// recording selections all come from the .nml.
SingleCellRun run_model(const String &model_path, const String &run_name) {
    ScopedWorkingDirectory run_directory(run_name);

    SingleCellRun run;
    String membrane_trace_filename;
    String spike_events_filename;
    String engine_input_path = model_path; // SpikeEngine takes a non-const String&

    {
        SpikeEngine engine(engine_input_path, /*enable_hebbian_learning=*/false);

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
        throw std::runtime_error("exit_model_validation_tests: model '" + model_path +
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

SingleCellRun run_single_cell_fixture(const String &fixture_base_name) {
    return run_model(fixture_path("nml/" + fixture_base_name + "_top.nml"), fixture_base_name);
}

// ── deliberately perturbed fixtures ──────────────────────────────────────────────────────────
//
// Written out of the checked-in glif2_single_cell.nml with ONE attribute substituted, rather than
// checked in as a second copy: the perturbation is then guaranteed to be the only difference, and
// a later edit to the real fixture cannot leave a stale copy behind. The include-only wrapper is
// the same trick glif2_single_cell_top.nml uses -- NML_Parser only XSD-validates the top-level
// file it is handed, and a bare ComponentType declaration is raw LEMS rather than schema-valid
// NeuroML.
String write_perturbed_glif2_fixture(const filesystem::path &destination_directory,
                                     const String &original_attribute,
                                     const String &replacement_attribute) {
    std::ifstream source_file(fixture_path("nml/glif2_single_cell.nml"));
    if (!source_file.is_open()) {
        throw std::runtime_error("exit_model_validation_tests: could not open glif2_single_cell.nml");
    }
    std::stringstream source_buffer;
    source_buffer << source_file.rdbuf();
    String fixture_text = source_buffer.str();

    const usize attribute_position = fixture_text.find(original_attribute);
    if (attribute_position == String::npos) {
        throw std::runtime_error("exit_model_validation_tests: glif2_single_cell.nml no longer "
                                 "declares '" + original_attribute + "'");
    }
    fixture_text.replace(attribute_position, original_attribute.size(), replacement_attribute);

    std::ofstream perturbed_fixture(destination_directory / "glif2_single_cell.nml");
    perturbed_fixture << fixture_text;
    perturbed_fixture.close();

    const filesystem::path wrapper_path = destination_directory / "glif2_perturbed_top.nml";
    std::ofstream wrapper(wrapper_path);
    wrapper << "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"Glif2PerturbedTop\">\n"
            << "  <include href=\"glif2_single_cell.nml\"/>\n"
            << "</neuroml>\n";
    wrapper.close();

    return wrapper_path.string();
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

// The reference row carrying the same instant this tick's sample does. See the header's
// "SAMPLE ALIGNMENT" section: pinned by the stimulus onset, agreeing to 7 significant figures on
// all four models. Two rows rather than one because jLEMS writes a t=0 initial condition before it
// steps AND applies a pulseGenerator's current one step after its OnCondition fires.
s64 reference_row_for_tick(s64 tick) { return tick + 2; }

// A reference spike timestamp as a tick index. The engine's spike flag for a tick is recorded in
// the same frame as that tick's membrane sample, and jLEMS timestamps a spike one row before the
// row carrying the post-reset voltage, so the event that lands in engine frame `f` is the reference
// spike at tick `f + 1` -- the same row `f + 2` reference_row_for_tick maps that frame to.
s64 reference_spike_tick(f64 time_seconds, f64 step_dt) {
    return (s64)std::llround(time_seconds / step_dt);
}

// ── spike raster ─────────────────────────────────────────────────────────────────────────────

struct SpikeTrainComparison {
    s64 own_spike_count = 0;
    s64 reference_spike_count = 0;

    // own_spike_ticks[k] + 1 - reference_spike_tick[k], over the spikes the two rasters have in
    // common. Zero everywhere is the assertion; the sequence itself is what says whether a failure
    // is a constant offset or an accumulating drift.
    spikecorec::Vector<s64> spike_tick_deviations;

    s64 mismatched_spike_count = 0;
    s64 largest_absolute_tick_deviation = 0;

    bool matches_reference() const {
        return own_spike_count == reference_spike_count && mismatched_spike_count == 0;
    }
};

SpikeTrainComparison compare_spike_train_against_reference(
        const spikecorec::Vector<s64> &own_spike_ticks,
        const spikecorec::Vector<f64> &reference_spike_times_seconds, f64 step_dt) {
    SpikeTrainComparison comparison;
    comparison.own_spike_count = (s64)own_spike_ticks.size();
    comparison.reference_spike_count = (s64)reference_spike_times_seconds.size();

    const usize paired_count = std::min(own_spike_ticks.size(), reference_spike_times_seconds.size());
    for (usize spike_index = 0; spike_index < paired_count; spike_index += 1) {
        const s64 deviation =
                own_spike_ticks[spike_index] + 1 -
                reference_spike_tick(reference_spike_times_seconds[spike_index], step_dt);
        comparison.spike_tick_deviations.push_back(deviation);
        if (deviation != 0) comparison.mismatched_spike_count += 1;
        comparison.largest_absolute_tick_deviation =
                std::max(comparison.largest_absolute_tick_deviation, (s64)std::llabs(deviation));
    }
    return comparison;
}

// ── refractory holds ─────────────────────────────────────────────────────────────────────────
//
// A GLIF cell's refractory regime declares no TimeDerivative for `v`, so `v` is HELD rather than
// integrated -- which shows up in a recorded trace as a bit-identical value repeated for the whole
// refractory period. Its length in samples is therefore the one directly observable consequence of
// t_ref, and comparing it hold for hold against the reference's says whether the refractory period
// itself is right, independently of anything the membrane trajectory does around it.

s64 constant_run_length_from(const spikecorec::Vector<f32> &samples, s64 start_index) {
    if (start_index < 0 || (usize)start_index >= samples.size()) return 0;
    s64 length = 1;
    while ((usize)(start_index + length) < samples.size() &&
           samples[(usize)(start_index + length)] == samples[(usize)start_index]) {
        length += 1;
    }
    return length;
}

s64 constant_run_length_from(const ReferenceTrace &trace, usize column_index, s64 start_row) {
    if (start_row < 0 || (usize)start_row >= trace.column_values.size()) return 0;
    if (column_index >= trace.column_values[(usize)start_row].size()) return 0;
    const f64 held_value = trace.column_values[(usize)start_row][column_index];
    s64 length = 1;
    while ((usize)(start_row + length) < trace.column_values.size() &&
           column_index < trace.column_values[(usize)(start_row + length)].size() &&
           trace.column_values[(usize)(start_row + length)][column_index] == held_value) {
        length += 1;
    }
    return length;
}

spikecorec::Vector<s64> engine_refractory_hold_lengths(const SingleCellRun &run) {
    spikecorec::Vector<s64> lengths;
    if (run.recorded_columns.empty()) return lengths;
    for (s64 spike_tick : run.spike_ticks) {
        lengths.push_back(constant_run_length_from(run.recorded_columns[0], spike_tick));
    }
    return lengths;
}

spikecorec::Vector<s64> reference_refractory_hold_lengths(
        const ReferenceTrace &reference_trace,
        const spikecorec::Vector<f64> &reference_spike_times_seconds, f64 step_dt) {
    spikecorec::Vector<s64> lengths;
    for (f64 spike_time : reference_spike_times_seconds) {
        // The post-reset voltage lands on the row after the one jLEMS timestamps the event on.
        lengths.push_back(constant_run_length_from(
                reference_trace, /*column_index=*/0, reference_spike_tick(spike_time, step_dt) + 1));
    }
    return lengths;
}

// ── recorded columns ─────────────────────────────────────────────────────────────────────────

// Only ever a diagnostic. Applied UNIFORMLY to the whole span rather than per sample, so it cannot
// smuggle in the per-sample local-range permissiveness the header describes: it answers "is this
// the right trajectory at the wrong time, and by how much", and comparing it across the run says
// whether that offset is constant or drifting.
const s64 UNIFORM_SHIFT_SEARCH_RADIUS = 12;

struct UniformShiftDiagnostic {
    s64 shift_ticks = 0;
    s64 sample_count_outside_tolerance = 0;
};

s64 count_samples_outside_at_uniform_shift(const spikecorec::Vector<f32> &own_column,
                                           const ReferenceTrace &reference_trace, usize column_index,
                                           f64 tolerance, s64 shift_ticks, s64 first_tick,
                                           s64 last_tick) {
    s64 count = 0;
    for (s64 tick = first_tick; tick < last_tick && (usize)tick < own_column.size(); tick += 1) {
        const s64 reference_row = reference_row_for_tick(tick) + shift_ticks;
        if (reference_row < 0 || (usize)reference_row >= reference_trace.column_values.size()) continue;
        if (column_index >= reference_trace.column_values[(usize)reference_row].size()) continue;
        const f64 deviation =
                std::fabs((f64)own_column[(usize)tick] -
                          reference_trace.column_values[(usize)reference_row][column_index]);
        if (deviation > tolerance) count += 1;
    }
    return count;
}

UniformShiftDiagnostic best_uniform_shift(const spikecorec::Vector<f32> &own_column,
                                          const ReferenceTrace &reference_trace, usize column_index,
                                          f64 tolerance, s64 first_tick, s64 last_tick) {
    UniformShiftDiagnostic best;
    best.shift_ticks = 0;
    best.sample_count_outside_tolerance = count_samples_outside_at_uniform_shift(
            own_column, reference_trace, column_index, tolerance, 0, first_tick, last_tick);
    for (s64 shift = -UNIFORM_SHIFT_SEARCH_RADIUS; shift <= UNIFORM_SHIFT_SEARCH_RADIUS; shift += 1) {
        if (shift == 0) continue;
        const s64 count = count_samples_outside_at_uniform_shift(
                own_column, reference_trace, column_index, tolerance, shift, first_tick, last_tick);
        if (count < best.sample_count_outside_tolerance) {
            best.sample_count_outside_tolerance = count;
            best.shift_ticks = shift;
        }
    }
    return best;
}

struct ColumnComparison {
    s64 compared_sample_count = 0;
    f64 peak_deviation = 0.0;             // largest |own - reference| at the SAME tick
    f64 root_mean_square_deviation = 0.0; // root-mean-square of the same

    // The asserted quantity. No tick-shift search: same tick, same instant.
    s64 sample_count_outside_tolerance = 0;
    s64 first_tick_outside_tolerance = -1;
    f32 own_value_at_first_failure = 0.0f;
    f64 reference_value_at_first_failure = 0.0;

    // Diagnostics only. `whole_run` says how far off a failing trace is as a whole; the first and
    // last third say whether that offset is drifting.
    UniformShiftDiagnostic whole_run_best_shift;
    UniformShiftDiagnostic first_third_best_shift;
    UniformShiftDiagnostic last_third_best_shift;
};

ColumnComparison compare_column_against_reference(const spikecorec::Vector<f32> &own_column,
                                                  const ReferenceTrace &reference_trace,
                                                  usize column_index, f64 tolerance) {
    ColumnComparison comparison;
    f64 squared_deviation_total = 0.0;

    for (usize tick = 0; tick < own_column.size(); tick += 1) {
        const s64 reference_row = reference_row_for_tick((s64)tick);
        if (reference_row < 0 || (usize)reference_row >= reference_trace.column_values.size()) continue;
        if (column_index >= reference_trace.column_values[(usize)reference_row].size()) continue;

        const f64 reference_value = reference_trace.column_values[(usize)reference_row][column_index];
        const f64 deviation = std::fabs((f64)own_column[tick] - reference_value);

        comparison.compared_sample_count += 1;
        comparison.peak_deviation = std::max(comparison.peak_deviation, deviation);
        squared_deviation_total += deviation * deviation;

        if (deviation > tolerance) {
            comparison.sample_count_outside_tolerance += 1;
            if (comparison.first_tick_outside_tolerance < 0) {
                comparison.first_tick_outside_tolerance = (s64)tick;
                comparison.own_value_at_first_failure = own_column[tick];
                comparison.reference_value_at_first_failure = reference_value;
            }
        }
    }

    if (comparison.compared_sample_count > 0) {
        comparison.root_mean_square_deviation =
                std::sqrt(squared_deviation_total / (f64)comparison.compared_sample_count);
    }

    const s64 tick_count = (s64)own_column.size();
    const s64 third = tick_count / 3;
    comparison.whole_run_best_shift =
            best_uniform_shift(own_column, reference_trace, column_index, tolerance, 0, tick_count);
    comparison.first_third_best_shift =
            best_uniform_shift(own_column, reference_trace, column_index, tolerance, 0, third);
    comparison.last_third_best_shift = best_uniform_shift(own_column, reference_trace, column_index,
                                                          tolerance, 2 * third, tick_count);
    return comparison;
}

// ── whole-model verdict ──────────────────────────────────────────────────────────────────────
//
// One structure for both directions: the per-model tests assert its parts individually so a
// failure names the part, and the negative controls assert the whole verdict is REJECT. Sharing
// it is the point -- a negative control that reimplemented the comparison would only prove its own
// reimplementation discriminates.

struct ModelComparison {
    SpikeTrainComparison spike_train;
    spikecorec::Vector<ColumnComparison> columns;
    spikecorec::Vector<s64> own_refractory_hold_lengths;
    spikecorec::Vector<s64> reference_refractory_hold_lengths;

    bool refractory_holds_match_reference() const {
        return own_refractory_hold_lengths == reference_refractory_hold_lengths;
    }

    bool agrees_with_reference() const {
        if (!spike_train.matches_reference()) return false;
        if (!refractory_holds_match_reference()) return false;
        for (const ColumnComparison &column : columns) {
            if (column.compared_sample_count == 0) return false;
            if (column.sample_count_outside_tolerance != 0) return false;
        }
        return true;
    }
};

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
// agree about a membrane voltage; the deflections these models make are tens of mV. Applied at the
// same tick, with no shift search -- see the header.
const f64 VOLTAGE_TOLERANCE_VOLTS = 1e-3;

// 10% of each after-spike current's own spike-time bump (ascAdd1=-100pA, ascAdd2=-200pA), the scale
// at which two simulators can be said to agree about a current that decays by orders of magnitude
// between spikes.
const f64 FIRST_AFTER_SPIKE_CURRENT_TOLERANCE_AMPS = 1e-11;
const f64 SECOND_AFTER_SPIKE_CURRENT_TOLERANCE_AMPS = 2e-11;

ModelComparison compare_run_against_reference(const ModelComparisonInputs &inputs,
                                              const spikecorec::Vector<f64> &column_tolerances) {
    const spikecorec::Vector<f64> reference_spike_times =
            extract_spike_time_seconds(inputs.reference_spikes);

    ModelComparison comparison;
    comparison.spike_train = compare_spike_train_against_reference(
            inputs.run.spike_ticks, reference_spike_times, inputs.run.step_dt);
    for (usize column_index = 0; column_index < column_tolerances.size(); column_index += 1) {
        if (column_index >= inputs.run.recorded_columns.size()) break;
        comparison.columns.push_back(compare_column_against_reference(
                inputs.run.recorded_columns[column_index], inputs.reference_trace, column_index,
                column_tolerances[column_index]));
    }
    comparison.own_refractory_hold_lengths = engine_refractory_hold_lengths(inputs.run);
    comparison.reference_refractory_hold_lengths = reference_refractory_hold_lengths(
            inputs.reference_trace, reference_spike_times, inputs.run.step_dt);
    return comparison;
}

// ── reporting ────────────────────────────────────────────────────────────────────────────────
//
// Measured numbers go to stdout for every model, passing or failing: this suite exists to produce a
// measurement, and a bare "OK" reports nothing about how close the engine actually is.

String format_tick_sequence(const spikecorec::Vector<s64> &values) {
    std::ostringstream text;
    text << "[";
    for (usize index = 0; index < values.size(); index += 1) {
        if (index > 0) text << " ";
        text << values[index];
    }
    text << "]";
    return text.str();
}

void report_column_comparison(const String &label, const String &column_name,
                              const ColumnComparison &comparison, f64 tolerance) {
    std::cout << "[ MEASURED ] " << label << " " << column_name
              << ": compared_samples=" << comparison.compared_sample_count
              << " peak_deviation=" << comparison.peak_deviation
              << " rms_deviation=" << comparison.root_mean_square_deviation
              << " samples_outside_tolerance=" << comparison.sample_count_outside_tolerance
              << " (tolerance=" << tolerance
              << ", same tick, no shift search) | diagnostic best_uniform_shift="
              << comparison.whole_run_best_shift.shift_ticks << " ticks (outside="
              << comparison.whole_run_best_shift.sample_count_outside_tolerance
              << "), first_third=" << comparison.first_third_best_shift.shift_ticks
              << " last_third=" << comparison.last_third_best_shift.shift_ticks << std::endl;
}

void report_spike_train(const String &label, const SpikeTrainComparison &comparison) {
    std::cout << "[ MEASURED ] " << label << " spikes: own_count=" << comparison.own_spike_count
              << " reference_count=" << comparison.reference_spike_count
              << " mismatched_spikes=" << comparison.mismatched_spike_count
              << " largest_tick_deviation=" << comparison.largest_absolute_tick_deviation
              << " per_spike_tick_deviation="
              << format_tick_sequence(comparison.spike_tick_deviations) << std::endl;
}

void report_refractory_holds(const String &label, const ModelComparison &comparison) {
    std::cout << "[ MEASURED ] " << label << " refractory holds (samples): own="
              << format_tick_sequence(comparison.own_refractory_hold_lengths)
              << " reference=" << format_tick_sequence(comparison.reference_refractory_hold_lengths)
              << std::endl;
}

void report_model_comparison(const String &label, const ModelComparison &comparison,
                             const spikecorec::Vector<f64> &column_tolerances) {
    report_spike_train(label, comparison.spike_train);
    for (usize column_index = 0; column_index < comparison.columns.size(); column_index += 1) {
        report_column_comparison(label, "column" + std::to_string(column_index),
                                 comparison.columns[column_index], column_tolerances[column_index]);
    }
    report_refractory_holds(label, comparison);
}

// ── per-model expectations ───────────────────────────────────────────────────────────────────

void expect_spike_train_matches_reference(const ModelComparison &comparison, const String &label) {
    EXPECT_EQ(comparison.spike_train.own_spike_count, comparison.spike_train.reference_spike_count)
            << label << ": spike count";

    // Every spike, not just the first and the last: a model that fires at the right rate at the
    // ends and at wrong times in between is exactly what a first/last-only check waves through.
    EXPECT_EQ(comparison.spike_train.mismatched_spike_count, 0)
            << label << ": " << comparison.spike_train.mismatched_spike_count << " of "
            << comparison.spike_train.spike_tick_deviations.size()
            << " spikes did not land on the reference's tick; per-spike deviation (ticks) "
            << format_tick_sequence(comparison.spike_train.spike_tick_deviations)
            << ". A deviation that GROWS along that sequence is an accumulating rate error, not a "
               "one-off timing tie";
}

void expect_refractory_holds_match_reference(const ModelComparison &comparison, const String &label) {
    EXPECT_EQ(comparison.own_refractory_hold_lengths, comparison.reference_refractory_hold_lengths)
            << label << ": the refractory hold is not the reference's length. own="
            << format_tick_sequence(comparison.own_refractory_hold_lengths) << " reference="
            << format_tick_sequence(comparison.reference_refractory_hold_lengths)
            << ". A uniform +1 here is the known f32 regime-counter defect described in this "
               "file's header";
}

void expect_column_matches_reference(const ColumnComparison &comparison, const String &label,
                                     const String &column_name, f64 tolerance) {
    ASSERT_GT(comparison.compared_sample_count, 0)
            << label << " " << column_name << ": nothing was compared";
    EXPECT_EQ(comparison.sample_count_outside_tolerance, 0)
            << label << " " << column_name << ": " << comparison.sample_count_outside_tolerance
            << " of " << comparison.compared_sample_count
            << " samples are outside +-" << tolerance << " of the reference sample for the SAME "
            << "tick; first at tick " << comparison.first_tick_outside_tolerance << " (own="
            << comparison.own_value_at_first_failure
            << ", reference=" << comparison.reference_value_at_first_failure
            << ", peak deviation=" << comparison.peak_deviation
            << "). Diagnostic: shifting the whole trace by "
            << comparison.whole_run_best_shift.shift_ticks << " ticks leaves "
            << comparison.whole_run_best_shift.sample_count_outside_tolerance
            << " outside, and the best shift moves from "
            << comparison.first_third_best_shift.shift_ticks << " over the first third to "
            << comparison.last_third_best_shift.shift_ticks
            << " over the last -- a moving shift is drift, a large residual at the best shift is "
               "the wrong trajectory";
}

void expect_membrane_trace_matches_reference(const ModelComparisonInputs &inputs,
                                             const ModelComparison &comparison, const String &label) {
    ASSERT_FALSE(inputs.run.recorded_columns.empty()) << label << ": nothing was recorded";
    // One recorded frame per tick, and 350ms at 0.1ms -> 3500 engine ticks against 3501 reference
    // rows (the extra row is jLEMS's t=0 initial condition, written before it steps).
    ASSERT_EQ(inputs.run.recorded_columns[0].size(), (usize)inputs.run.tick_count);
    ASSERT_EQ(inputs.run.recorded_columns[0].size() + 1, inputs.reference_trace.time_seconds.size());
    ASSERT_FALSE(comparison.columns.empty()) << label << ": no column was compared";

    // Under the tick+2 alignment the last engine tick has no reference row, so every other one must
    // be compared. Asserted so a truncated reference file cannot quietly shrink the coverage.
    EXPECT_EQ(comparison.columns[0].compared_sample_count, inputs.run.tick_count - 1)
            << label << ": the membrane comparison did not cover the whole run";

    expect_column_matches_reference(comparison.columns[0], label, "v", VOLTAGE_TOLERANCE_VOLTS);
}

} // namespace

// ── GLIF2: scaled reset ──────────────────────────────────────────────────────────────────────

TEST(ExitModelGlif2SingleCell, matches_pyneuroml_reference) {
    ASSERT_TRUE(standard_library_available())
            << "the vendored NML standard library is missing from third_party/neuroml2/std_lib";

    const ModelComparisonInputs inputs = load_model_comparison_inputs("glif2_single_cell", "glif2");
    const spikecorec::Vector<f64> column_tolerances = {VOLTAGE_TOLERANCE_VOLTS};
    const ModelComparison comparison = compare_run_against_reference(inputs, column_tolerances);
    report_model_comparison("glif2_single_cell", comparison, column_tolerances);

    expect_spike_train_matches_reference(comparison, "glif2_single_cell");
    expect_refractory_holds_match_reference(comparison, "glif2_single_cell");
    expect_membrane_trace_matches_reference(inputs, comparison, "glif2_single_cell");
}

// ── GLIF3: after-spike currents ──────────────────────────────────────────────────────────────

TEST(ExitModelGlif3SingleCell, matches_pyneuroml_reference) {
    ASSERT_TRUE(standard_library_available())
            << "the vendored NML standard library is missing from third_party/neuroml2/std_lib";

    const ModelComparisonInputs inputs = load_model_comparison_inputs("glif3_single_cell", "glif3");
    const spikecorec::Vector<f64> column_tolerances = {VOLTAGE_TOLERANCE_VOLTS};
    const ModelComparison comparison = compare_run_against_reference(inputs, column_tolerances);
    report_model_comparison("glif3_single_cell", comparison, column_tolerances);

    expect_spike_train_matches_reference(comparison, "glif3_single_cell");
    expect_refractory_holds_match_reference(comparison, "glif3_single_cell");
    expect_membrane_trace_matches_reference(inputs, comparison, "glif3_single_cell");
}

// The mechanism GLIF3 exists to demonstrate, checked against the reference rather than only against
// itself: asc1/asc2 are recorded columns 1 and 2 of the fixture's own <OutputFile>.
TEST(ExitModelGlif3SingleCell, after_spike_currents_match_reference) {
    ASSERT_TRUE(standard_library_available())
            << "the vendored NML standard library is missing from third_party/neuroml2/std_lib";

    const ModelComparisonInputs inputs = load_model_comparison_inputs("glif3_single_cell", "glif3");
    ASSERT_EQ(inputs.run.recorded_columns.size(), 3u); // v, asc1, asc2

    const ColumnComparison first_current_comparison = compare_column_against_reference(
            inputs.run.recorded_columns[1], inputs.reference_trace, /*column_index=*/1,
            FIRST_AFTER_SPIKE_CURRENT_TOLERANCE_AMPS);
    report_column_comparison("glif3_single_cell", "asc1", first_current_comparison,
                             FIRST_AFTER_SPIKE_CURRENT_TOLERANCE_AMPS);
    expect_column_matches_reference(first_current_comparison, "glif3_single_cell", "asc1",
                                    FIRST_AFTER_SPIKE_CURRENT_TOLERANCE_AMPS);

    const ColumnComparison second_current_comparison = compare_column_against_reference(
            inputs.run.recorded_columns[2], inputs.reference_trace, /*column_index=*/2,
            SECOND_AFTER_SPIKE_CURRENT_TOLERANCE_AMPS);
    report_column_comparison("glif3_single_cell", "asc2", second_current_comparison,
                             SECOND_AFTER_SPIKE_CURRENT_TOLERANCE_AMPS);
    expect_column_matches_reference(second_current_comparison, "glif3_single_cell", "asc2",
                                    SECOND_AFTER_SPIKE_CURRENT_TOLERANCE_AMPS);

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
    ASSERT_TRUE(standard_library_available())
            << "the vendored NML standard library is missing from third_party/neuroml2/std_lib";

    const ModelComparisonInputs inputs = load_model_comparison_inputs("glif4_single_cell", "glif4");
    const spikecorec::Vector<f64> column_tolerances = {VOLTAGE_TOLERANCE_VOLTS};
    const ModelComparison comparison = compare_run_against_reference(inputs, column_tolerances);
    report_model_comparison("glif4_single_cell", comparison, column_tolerances);

    expect_spike_train_matches_reference(comparison, "glif4_single_cell");
    expect_refractory_holds_match_reference(comparison, "glif4_single_cell");
    expect_membrane_trace_matches_reference(inputs, comparison, "glif4_single_cell");
}

TEST(ExitModelGlif4SingleCell, adaptive_threshold_matches_reference) {
    ASSERT_TRUE(standard_library_available())
            << "the vendored NML standard library is missing from third_party/neuroml2/std_lib";

    const ModelComparisonInputs inputs = load_model_comparison_inputs("glif4_single_cell", "glif4");
    ASSERT_EQ(inputs.run.recorded_columns.size(), 2u); // v, theta

    const ColumnComparison threshold_comparison = compare_column_against_reference(
            inputs.run.recorded_columns[1], inputs.reference_trace, /*column_index=*/1,
            VOLTAGE_TOLERANCE_VOLTS);
    report_column_comparison("glif4_single_cell", "theta", threshold_comparison,
                             VOLTAGE_TOLERANCE_VOLTS);
    expect_column_matches_reference(threshold_comparison, "glif4_single_cell", "theta",
                                    VOLTAGE_TOLERANCE_VOLTS);

    // Non-vacuity: the threshold actually adapts. thetaInf is -50mV and each spike adds 5mV, so a
    // threshold that never rises above thetaInf means the mechanism did not run at all.
    f32 highest_threshold = -std::numeric_limits<f32>::infinity();
    for (f32 value : inputs.run.recorded_columns[1]) highest_threshold = std::max(highest_threshold, value);
    EXPECT_GT(highest_threshold, -0.0475f) << "glif4's threshold never rose above thetaInf + 2.5mV";
}

// ── GLIF5: after-spike currents AND adapting threshold ───────────────────────────────────────

TEST(ExitModelGlif5SingleCell, matches_pyneuroml_reference) {
    ASSERT_TRUE(standard_library_available())
            << "the vendored NML standard library is missing from third_party/neuroml2/std_lib";

    const ModelComparisonInputs inputs = load_model_comparison_inputs("glif5_single_cell", "glif5");
    const spikecorec::Vector<f64> column_tolerances = {VOLTAGE_TOLERANCE_VOLTS};
    const ModelComparison comparison = compare_run_against_reference(inputs, column_tolerances);
    report_model_comparison("glif5_single_cell", comparison, column_tolerances);

    expect_spike_train_matches_reference(comparison, "glif5_single_cell");
    expect_refractory_holds_match_reference(comparison, "glif5_single_cell");
    expect_membrane_trace_matches_reference(inputs, comparison, "glif5_single_cell");
}

TEST(ExitModelGlif5SingleCell, adaptive_threshold_and_after_spike_currents_match_reference) {
    ASSERT_TRUE(standard_library_available())
            << "the vendored NML standard library is missing from third_party/neuroml2/std_lib";

    const ModelComparisonInputs inputs = load_model_comparison_inputs("glif5_single_cell", "glif5");
    ASSERT_EQ(inputs.run.recorded_columns.size(), 4u); // v, theta, asc1, asc2

    const ColumnComparison threshold_comparison = compare_column_against_reference(
            inputs.run.recorded_columns[1], inputs.reference_trace, /*column_index=*/1,
            VOLTAGE_TOLERANCE_VOLTS);
    report_column_comparison("glif5_single_cell", "theta", threshold_comparison,
                             VOLTAGE_TOLERANCE_VOLTS);
    expect_column_matches_reference(threshold_comparison, "glif5_single_cell", "theta",
                                    VOLTAGE_TOLERANCE_VOLTS);

    const ColumnComparison first_current_comparison = compare_column_against_reference(
            inputs.run.recorded_columns[2], inputs.reference_trace, /*column_index=*/2,
            FIRST_AFTER_SPIKE_CURRENT_TOLERANCE_AMPS);
    report_column_comparison("glif5_single_cell", "asc1", first_current_comparison,
                             FIRST_AFTER_SPIKE_CURRENT_TOLERANCE_AMPS);
    expect_column_matches_reference(first_current_comparison, "glif5_single_cell", "asc1",
                                    FIRST_AFTER_SPIKE_CURRENT_TOLERANCE_AMPS);

    const ColumnComparison second_current_comparison = compare_column_against_reference(
            inputs.run.recorded_columns[3], inputs.reference_trace, /*column_index=*/3,
            SECOND_AFTER_SPIKE_CURRENT_TOLERANCE_AMPS);
    report_column_comparison("glif5_single_cell", "asc2", second_current_comparison,
                             SECOND_AFTER_SPIKE_CURRENT_TOLERANCE_AMPS);
    expect_column_matches_reference(second_current_comparison, "glif5_single_cell", "asc2",
                                    SECOND_AFTER_SPIKE_CURRENT_TOLERANCE_AMPS);

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

// ── negative controls: a WRONG PAIRING must be rejected ──────────────────────────────────────
//
// Both feed this same harness a comparison that is known to be wrong and assert it says so. They
// exist because a validation suite's real failure mode is going vacuous -- comparing a trace with
// itself, or against a band wide enough to admit anything -- while still reporting green.

TEST(ExitModelNegativeControls, glif3_membrane_trace_is_rejected_against_glif5s_reference) {
    ASSERT_TRUE(standard_library_available())
            << "the vendored NML standard library is missing from third_party/neuroml2/std_lib";

    // A DIFFERENT but closely related model: GLIF5 is GLIF3 plus an adapting threshold, driven by
    // the same 0.6nA pulse over the same window. Nothing about the two traces is absurd next to
    // each other, which is what makes it the right control -- it is the near miss, not a wild one.
    const SingleCellRun glif3_run = run_single_cell_fixture("glif3_single_cell");
    const ReferenceTrace glif5_reference = load_reference_trace(
            fixture_path("reference_data/glif5_single_cell/glif5_membrane_trace.dat"));

    const ColumnComparison comparison = compare_column_against_reference(
            glif3_run.recorded_columns[0], glif5_reference, /*column_index=*/0,
            VOLTAGE_TOLERANCE_VOLTS);
    report_column_comparison("negative_control_glif3_v_vs_glif5_reference", "v", comparison,
                             VOLTAGE_TOLERANCE_VOLTS);

    // Measured 2909 of 3499. Asserted as a majority rather than as that exact count so the control
    // states what it means -- most of the run disagrees -- instead of pinning an incidental number.
    EXPECT_GT(comparison.sample_count_outside_tolerance, comparison.compared_sample_count / 2)
            << "the harness accepted glif3's membrane trace against GLIF5's reference; the "
               "membrane band is no longer discriminating";
}

TEST(ExitModelNegativeControls, glif3_membrane_trace_is_rejected_against_an_after_spike_current) {
    ASSERT_TRUE(standard_library_available())
            << "the vendored NML standard library is missing from third_party/neuroml2/std_lib";

    // A voltage compared against a current -- the shape a units or column-index error takes. The
    // two differ by nine orders of magnitude, so every single sample must land outside.
    const SingleCellRun glif3_run = run_single_cell_fixture("glif3_single_cell");
    const ReferenceTrace glif3_reference = load_reference_trace(
            fixture_path("reference_data/glif3_single_cell/glif3_membrane_trace.dat"));

    const ColumnComparison comparison = compare_column_against_reference(
            glif3_run.recorded_columns[0], glif3_reference, /*column_index=*/1,
            VOLTAGE_TOLERANCE_VOLTS);
    report_column_comparison("negative_control_glif3_v_vs_glif3_asc1", "v_against_asc1", comparison,
                             VOLTAGE_TOLERANCE_VOLTS);

    EXPECT_EQ(comparison.sample_count_outside_tolerance, comparison.compared_sample_count)
            << "the harness accepted glif3's membrane voltage against a CURRENT column";
}

// ── perturbation controls: a 2% PARAMETER ERROR must be rejected ─────────────────────────────
//
// Each takes the checked-in glif2 fixture, substitutes exactly one attribute, and asserts the
// harness rejects the result against the UNMODIFIED reference. The verdict is the same
// ModelComparison::agrees_with_reference() the per-model tests above assert the parts of, so these
// measure this harness rather than a parallel reimplementation of it.
//
// Not every 2% perturbation is detectable, and pretending otherwise would be the same dishonesty
// this file exists to remove: resetScale 0.3 -> 0.306 was measured and produces a trace IDENTICAL
// to the unmodified model's, because `v` crosses vth by microvolts and so `resetScale * (v - vth)`
// is microvolts times 2% either way. The three below are perturbations this data can actually see.

namespace {

void expect_perturbed_glif2_is_rejected(const String &control_name, const String &original_attribute,
                                        const String &replacement_attribute) {
    ScopedTemporaryDirectory perturbation_directory(control_name);
    const String model_path = write_perturbed_glif2_fixture(perturbation_directory.path(),
                                                            original_attribute, replacement_attribute);

    ModelComparisonInputs inputs;
    inputs.run = run_model(model_path, "perturbation_" + control_name);
    inputs.reference_trace = load_reference_trace(
            fixture_path("reference_data/glif2_single_cell/glif2_membrane_trace.dat"));
    inputs.reference_spikes =
            load_reference_spikes(fixture_path("reference_data/glif2_single_cell/glif2_spikes.dat"));

    const spikecorec::Vector<f64> column_tolerances = {VOLTAGE_TOLERANCE_VOLTS};
    const ModelComparison comparison = compare_run_against_reference(inputs, column_tolerances);
    report_model_comparison("perturbation_control_" + control_name, comparison, column_tolerances);

    EXPECT_FALSE(comparison.agrees_with_reference())
            << control_name << ": the harness ACCEPTED glif2 with " << original_attribute
            << " replaced by " << replacement_attribute
            << " against the unmodified reference. Spike-tick deviations "
            << format_tick_sequence(comparison.spike_train.spike_tick_deviations)
            << ", refractory holds " << format_tick_sequence(comparison.own_refractory_hold_lengths)
            << " against " << format_tick_sequence(comparison.reference_refractory_hold_lengths)
            << ", membrane samples outside tolerance "
            << (comparison.columns.empty() ? -1 : comparison.columns[0].sample_count_outside_tolerance);
}

} // namespace

// The reviewer's exact experiment. A 2% refractory error shortens every hold by one tick at this
// fixture's 0.1ms step, so a correct engine ends the run 10 ticks ahead of the reference.
//
// THIS TEST FAILS TODAY, and will pass unchanged once the f32 regime-counter defect described in
// this file's header is fixed. The two errors are the same size and cancel exactly: today's engine
// holds one tick too LONG, so at t_ref=4.9ms it holds for the reference's 51 samples and reproduces
// the reference tick for tick -- measured, zero samples outside 1mV, every spike on the reference's
// tick. On today's engine 4.9ms is genuinely the value that matches jNeuroML, so no honest harness
// can reject it; what is broken is the engine, not the discrimination. Deliberately NOT weakened to
// go green, because the moment the refractory hold is 51 samples at the declared 5ms, this model
// drifts one tick per spike and is rejected on all three grounds at once.
TEST(ExitModelPerturbationControls, rejects_a_two_percent_shorter_refractory_period) {
    ASSERT_TRUE(standard_library_available())
            << "the vendored NML standard library is missing from third_party/neuroml2/std_lib";

    expect_perturbed_glif2_is_rejected("t_ref_4900us", "t_ref=\"5ms\"", "t_ref=\"4.9ms\"");
}

// The same 2% error in the other direction, which today's engine defect does not mask. Measured:
// refractory holds 53 samples against the reference's 51, every ISI two ticks long, the last spike
// 18 ticks late, and 876 of 3499 membrane samples outside 1mV.
TEST(ExitModelPerturbationControls, rejects_a_two_percent_longer_refractory_period) {
    ASSERT_TRUE(standard_library_available())
            << "the vendored NML standard library is missing from third_party/neuroml2/std_lib";

    expect_perturbed_glif2_is_rejected("t_ref_5100us", "t_ref=\"5ms\"", "t_ref=\"5.1ms\"");
}

// A second parameter, and a different mechanism: membrane capacitance sets the time constant
// C/gL, so a 2% error is a 2% error in how fast `v` approaches threshold, with no effect at all on
// the refractory period. Measured: every ISI 295 ticks against the reference's 289, the last spike
// 59 ticks late, and 2242 of 3499 membrane samples outside 1mV. The same perturbation at 0.5%
// (100.5pF) was also measured and is also rejected -- 914 samples outside, ISI 291 -- so 2% is not
// near this harness's floor.
TEST(ExitModelPerturbationControls, rejects_a_two_percent_membrane_capacitance_error) {
    ASSERT_TRUE(standard_library_available())
            << "the vendored NML standard library is missing from third_party/neuroml2/std_lib";

    expect_perturbed_glif2_is_rejected("capacitance_102pF", "C=\"100pF\"", "C=\"102pF\"");
}
