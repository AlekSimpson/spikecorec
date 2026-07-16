#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <gtest/gtest.h>

#include "spikecorec/nml/nml.h"
#include "spikecorec/nml/resolve.h"
#include "spikecorec/nml/model_specification.h"
#include "spikecorec/nml/cell_lowering.h"
#include "spikecorec/nml/synapse_lowering.h"
#include "spikecorec/nml/allocator.h"
#include "spikecorec/nml/output_recording.h"
#include "spikecorec/core/recording.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::nml;

// ── Recording / output selections -> recorder streams tests (ticket #59 [E2]) ──
//
// Builds a real, lowerable Phase-1 GLIF excitatory/inhibitory network
// (GLIF1Cell fixture reused verbatim from allocator_tests.cpp -- two state
// variables `v`/`refractoryTimeElapsed`, a refractory Regime pair, and an
// exposed-but-never-stored `iSyn` DerivedVariable, the derived-exposure-
// scratch acceptance case) wired through a real conductance-based synapse,
// then exercises the full pipeline this ticket adds: resolve_recording_streams
// groups several OutputColumns/EventSelections into deduplicated streams, and
// sample_recording_stream reads real values back out of ModelAllocation's
// buffers -- proven correct by manually driving several "ticks" (writing
// synthetic values into cell_state/derived-scratch/last_spiked the same way a
// live generated kernel eventually would, since master-kernel assembly (#6)
// isn't merged yet -- see output_recording.h's own scope-boundary note),
// sampling each tick into a real `.spire` file via the existing
// SimulationRecorder, and reading the files back to confirm the written bytes
// exactly match what was staged.

namespace {

String write_temp_file(const String &filename, const String &contents) {
    String path = (std::filesystem::temp_directory_path() / filename).string();
    std::ofstream out(path);
    out << contents;
    out.close();
    return path;
}

const PopulationEntry &population_named(const ModelSpecification &specification, const String &id) {
    for (const auto &population : specification.populations) {
        if (population.id == id) return population;
    }
    throw std::runtime_error("no population '" + id + "'");
}

Vector<IrProgram> build_type_library_ir_programs(const ModelSpecification &model) {
    Vector<IrProgram> programs;
    programs.reserve(model.type_library.size());
    for (const auto &entry : model.type_library) {
        if (entry.category == TypeLibraryCategory::Cell) {
            programs.push_back(lower_cell_to_ir(entry));
        } else if (entry.category == TypeLibraryCategory::Synapse) {
            programs.push_back(lower_synapse_to_ir(entry));
        } else {
            IrProgram placeholder;
            placeholder.component_type_name = entry.component_type_name;
            programs.push_back(std::move(placeholder));
        }
    }
    return programs;
}

// The cell_lowering_tests.cpp/allocator_tests.cpp GLIF1Cell fixture verbatim
// (2 state variables `v`/`refractoryTimeElapsed`, a refractory Regime pair,
// and an exposed-but-never-stored `iSyn` DerivedVariable) wired into a
// 2-population network (ExcPop size 3, InhPop size 2) through a real
// expOneSynapse projection, plus a Simulation/OutputFile/EventOutputFile
// exercising this ticket's grouping: 3 OutputColumns on ExcPop's `v` (dedup
// to 1 stream), 1 OutputColumn on ExcPop[0]'s `iSyn` (derived-scratch
// stream), and 2 EventSelections on InhPop (dedup to 1 spike-raster stream).
String write_output_recording_fixture() {
    write_temp_file("spikecorec_output_recording_content.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"OutputRecordingContent\">"
        "  <ComponentType name=\"GLIF1Cell\" extends=\"baseCell\">"
        "    <Parameter name=\"C\" dimension=\"capacitance\"/>"
        "    <Parameter name=\"gL\" dimension=\"conductance\"/>"
        "    <Parameter name=\"EL\" dimension=\"voltage\"/>"
        "    <Parameter name=\"vth\" dimension=\"voltage\"/>"
        "    <Parameter name=\"vreset\" dimension=\"voltage\"/>"
        "    <Parameter name=\"t_ref\" dimension=\"time\"/>"
        "    <Attachments name=\"synapses\" type=\"basePointCurrent\"/>"
        "    <Dynamics>"
        "      <StateVariable name=\"v\" dimension=\"voltage\" exposure=\"v\"/>"
        "      <StateVariable name=\"refractoryTimeElapsed\" dimension=\"time\"/>"
        "      <DerivedVariable name=\"iSyn\" dimension=\"current\" exposure=\"iSyn\" select=\"synapses[*]/i\" reduce=\"add\"/>"
        "      <OnStart>"
        "        <StateAssignment variable=\"v\" value=\"EL\"/>"
        "        <StateAssignment variable=\"refractoryTimeElapsed\" value=\"0\"/>"
        "      </OnStart>"
        "      <Regime name=\"integrating\" initial=\"true\">"
        "        <TimeDerivative variable=\"v\" value=\"(gL * (EL - v) + iSyn) / C\"/>"
        "        <OnCondition test=\"v .gt. vth\">"
        "          <EventOut port=\"spike\"/>"
        "          <StateAssignment variable=\"v\" value=\"vreset\"/>"
        "          <Transition regime=\"refractory\"/>"
        "        </OnCondition>"
        "      </Regime>"
        "      <Regime name=\"refractory\">"
        "        <OnEntry>"
        "          <StateAssignment variable=\"refractoryTimeElapsed\" value=\"0\"/>"
        "        </OnEntry>"
        "        <TimeDerivative variable=\"refractoryTimeElapsed\" value=\"1\"/>"
        "        <OnCondition test=\"refractoryTimeElapsed .geq. t_ref\">"
        "          <Transition regime=\"integrating\"/>"
        "        </OnCondition>"
        "      </Regime>"
        "    </Dynamics>"
        "  </ComponentType>"
        "  <GLIF1Cell id=\"glif1ExcCellInstance\" C=\"1.0e-10\" gL=\"1.0e-8\" EL=\"-70mV\" vth=\"-50mV\" vreset=\"-70mV\" t_ref=\"2ms\"/>"
        "  <GLIF1Cell id=\"glif1InhCellInstance\" C=\"1.0e-10\" gL=\"1.2e-8\" EL=\"-65mV\" vth=\"-50mV\" vreset=\"-65mV\" t_ref=\"2ms\"/>"
        "  <expOneSynapse id=\"glifExcSynapse\" gbase=\"1nS\" erev=\"0mV\" tauDecay=\"3ms\" weight=\"2\"/>"
        "  <network id=\"OutputRecordingNet\">"
        "    <population id=\"ExcPop\" component=\"glif1ExcCellInstance\" size=\"3\"/>"
        "    <population id=\"InhPop\" component=\"glif1InhCellInstance\" size=\"2\"/>"
        "    <projection id=\"ExcToInh\" presynapticPopulation=\"ExcPop\" postsynapticPopulation=\"InhPop\" synapse=\"glifExcSynapse\">"
        "      <connection id=\"0\" preCellId=\"../ExcPop/0/glif1ExcCellInstance\" postCellId=\"../InhPop/0/glif1InhCellInstance\"/>"
        "    </projection>"
        "  </network>"
        "  <Simulation id=\"sim1\" length=\"1000ms\" step=\"0.1ms\" target=\"OutputRecordingNet\">"
        "    <OutputFile id=\"of0\" fileName=\"results.dat\">"
        "      <OutputColumn id=\"v_exc0\" quantity=\"ExcPop[0]/v\"/>"
        "      <OutputColumn id=\"v_exc1\" quantity=\"ExcPop[1]/v\"/>"
        "      <OutputColumn id=\"v_exc2\" quantity=\"ExcPop[2]/v\"/>"
        "      <OutputColumn id=\"iSyn_exc0\" quantity=\"ExcPop[0]/iSyn\"/>"
        "    </OutputFile>"
        "    <EventOutputFile id=\"spikesFile\" fileName=\"spikes.dat\" format=\"TIME_ID\">"
        "      <EventSelection id=\"sel0\" select=\"InhPop[0]\" eventPort=\"spike\"/>"
        "      <EventSelection id=\"sel1\" select=\"InhPop[1]\" eventPort=\"spike\"/>"
        "    </EventOutputFile>"
        "  </Simulation>"
        "</neuroml>");

    return write_temp_file("spikecorec_output_recording_top.nml",
        "<neuroml xmlns=\"http://www.neuroml.org/schema/neuroml2\" id=\"OutputRecordingTop\">"
        "  <include href=\"spikecorec_output_recording_content.nml\"/>"
        "</neuroml>");
}

ModelSpecification build_output_recording_test_model() {
    String path = write_output_recording_fixture();
    NML_Parser parser;
    parser.parse(path);
    ResolvedModel resolved = resolve_and_lower(parser);
    return build_model_specification(resolved);
}

} // namespace

// ── resolve_recording_streams: grouping + source-kind resolution ──────────

TEST(OutputRecording, groups_recordings_sharing_population_exposure_and_kind_into_one_stream) {
    ModelSpecification model = build_output_recording_test_model();
    Vector<IrProgram> programs = build_type_library_ir_programs(model);

    Vector<RecordingStream> streams = resolve_recording_streams(model, programs);

    ASSERT_EQ(streams.size(), 3u);

    const RecordingStream *v_stream = nullptr;
    const RecordingStream *iSyn_stream = nullptr;
    const RecordingStream *spike_stream = nullptr;
    for (const auto &stream : streams) {
        if (stream.is_event_recording) spike_stream = &stream;
        else if (stream.exposure_name == "v") v_stream = &stream;
        else if (stream.exposure_name == "iSyn") iSyn_stream = &stream;
    }
    ASSERT_NE(v_stream, nullptr);
    ASSERT_NE(iSyn_stream, nullptr);
    ASSERT_NE(spike_stream, nullptr);

    const PopulationEntry &exc_pop = population_named(model, "ExcPop");
    s32 exc_pop_index = -1, inh_pop_index = -1;
    for (s32 index = 0; index < (s32)model.populations.size(); ++index) {
        if (model.populations[(usize)index].id == "ExcPop") exc_pop_index = index;
        if (model.populations[(usize)index].id == "InhPop") inh_pop_index = index;
    }

    EXPECT_EQ(v_stream->population_index, exc_pop_index);
    EXPECT_EQ(v_stream->source_kind, RecordingSourceKind::DirectState);
    EXPECT_EQ(v_stream->state_variable_index, 0); // `v` is the first StateVariable declared
    EXPECT_EQ(v_stream->recording_entry_indices.size(), 3u); // ExcPop[0..2]/v

    EXPECT_EQ(iSyn_stream->population_index, exc_pop_index);
    EXPECT_EQ(iSyn_stream->source_kind, RecordingSourceKind::DerivedScratch);
    EXPECT_EQ(iSyn_stream->scratch_key, type_scoped_key(exc_pop.type_library_index, "iSyn"));
    EXPECT_EQ(iSyn_stream->recording_entry_indices.size(), 1u);

    EXPECT_EQ(spike_stream->population_index, inh_pop_index);
    EXPECT_EQ(spike_stream->source_kind, RecordingSourceKind::SpikeRaster);
    EXPECT_EQ(spike_stream->recording_entry_indices.size(), 2u); // InhPop[0], InhPop[1]
}

TEST(OutputRecording, skips_recording_with_unresolved_target_and_produces_no_stream) {
    ModelSpecification model = build_output_recording_test_model();
    Vector<IrProgram> programs = build_type_library_ir_programs(model);

    RecordingEntry unresolved;
    unresolved.id = "badRecording";
    unresolved.quantity_path = "NoSuchPop[0]/v";
    unresolved.target_neuron_index = -1;
    unresolved.exposure_name = "v";
    model.recordings.push_back(unresolved);

    Vector<RecordingStream> streams = resolve_recording_streams(model, programs);

    // Still exactly the 3 real streams -- the unresolved entry produced none.
    EXPECT_EQ(streams.size(), 3u);
}

TEST(OutputRecording, skips_recording_whose_exposure_matches_no_state_or_derived_variable) {
    ModelSpecification model = build_output_recording_test_model();
    Vector<IrProgram> programs = build_type_library_ir_programs(model);

    RecordingEntry unknown_exposure;
    unknown_exposure.id = "unknownExposureRecording";
    unknown_exposure.quantity_path = "ExcPop[0]/notAnExposure";
    unknown_exposure.target_neuron_index = 0; // ExcPop[0]
    unknown_exposure.exposure_name = "notAnExposure";
    model.recordings.push_back(unknown_exposure);

    Vector<RecordingStream> streams = resolve_recording_streams(model, programs);

    EXPECT_EQ(streams.size(), 3u);
}

TEST(OutputRecording, throws_when_ir_program_count_does_not_match_type_library_size) {
    ModelSpecification model = build_output_recording_test_model();
    Vector<IrProgram> programs = build_type_library_ir_programs(model);
    ASSERT_FALSE(programs.empty());
    programs.pop_back();

    EXPECT_THROW(resolve_recording_streams(model, programs), std::runtime_error);
}

// ── sample_recording_stream + real `.spire` round trip (acceptance criterion:
// "Traces + spike rasters are written for a Phase-1 model") ────────────────
//
// Master-kernel assembly (#6) isn't merged, so nothing yet drives a live
// per-tick kernel to populate cell_state/derived-scratch/last_spiked -- this
// test manually stages synthetic values into exactly those buffers the same
// way a live generated kernel eventually would, one "tick" at a time, then
// proves the resolve -> sample -> record -> read-back pipeline reproduces
// those staged values exactly out of real `.spire` files on disk.

TEST(OutputRecording, writes_and_reads_back_a_real_spire_state_trace_across_several_ticks) {
    ModelSpecification model = build_output_recording_test_model();
    Vector<IrProgram> programs = build_type_library_ir_programs(model);
    ModelAllocation allocation = allocate_model(model, programs);

    Vector<RecordingStream> streams = resolve_recording_streams(model, programs);
    const RecordingStream *v_stream = nullptr;
    for (const auto &stream : streams) {
        if (!stream.is_event_recording && stream.exposure_name == "v") v_stream = &stream;
    }
    ASSERT_NE(v_stream, nullptr);

    const PopulationEntry &exc_pop = population_named(model, "ExcPop");
    const s64 *boundaries = allocation.cell_type_boundaries.get_contents();
    s64 chunk_base_offset = boundaries[v_stream->population_index];
    f32 *cell_state = allocation.cell_state.get_contents();

    String output_path = (std::filesystem::temp_directory_path() / "spikecorec_output_recording_v_trace.spire").string();
    SimulationRecorder recorder(output_path, exc_pop.size);

    constexpr s64 TICK_COUNT = 5;
    Vector<Vector<f32>> staged_frames;
    for (s64 tick = 0; tick < TICK_COUNT; ++tick) {
        Vector<f32> frame(exc_pop.size);
        for (s32 local_index = 0; local_index < exc_pop.size; ++local_index) {
            f32 synthetic_value = -0.070f + 0.001f * (f32)tick + 0.0001f * (f32)local_index;
            cell_state[chunk_base_offset + (s64)v_stream->state_variable_index * exc_pop.size + local_index] = synthetic_value;
            frame[(usize)local_index] = synthetic_value;
        }
        staged_frames.push_back(frame);

        Vector<f32> sampled;
        sample_recording_stream(*v_stream, model, allocation, nullptr, tick, sampled);
        ASSERT_EQ(sampled.size(), (usize)exc_pop.size);
        for (s32 local_index = 0; local_index < exc_pop.size; ++local_index) {
            EXPECT_FLOAT_EQ(sampled[(usize)local_index], frame[(usize)local_index]);
        }
        recorder.record_frame(sampled.data(), (s64)sampled.size());
    }
    recorder.finish();

    SpireRecording read_back = read_spire_recording(output_path);
    EXPECT_EQ(read_back.neuron_count, exc_pop.size);
    ASSERT_EQ(read_back.frame_count, TICK_COUNT);
    for (s64 tick = 0; tick < TICK_COUNT; ++tick) {
        for (s32 local_index = 0; local_index < exc_pop.size; ++local_index) {
            f32 written = read_back.frames[(usize)(tick * exc_pop.size + local_index)];
            EXPECT_FLOAT_EQ(written, staged_frames[(usize)tick][(usize)local_index]);
        }
    }

    std::filesystem::remove(output_path);
}

TEST(OutputRecording, writes_and_reads_back_a_real_spire_derived_scratch_trace) {
    ModelSpecification model = build_output_recording_test_model();
    Vector<IrProgram> programs = build_type_library_ir_programs(model);
    ModelAllocation allocation = allocate_model(model, programs);

    Vector<RecordingStream> streams = resolve_recording_streams(model, programs);
    const RecordingStream *iSyn_stream = nullptr;
    for (const auto &stream : streams) {
        if (!stream.is_event_recording && stream.exposure_name == "iSyn") iSyn_stream = &stream;
    }
    ASSERT_NE(iSyn_stream, nullptr);

    const PopulationEntry &exc_pop = population_named(model, "ExcPop");
    f32 *scratch = allocation.derived_exposure_scratch_buffers.at(iSyn_stream->scratch_key).get_contents();

    String output_path = (std::filesystem::temp_directory_path() / "spikecorec_output_recording_iSyn_trace.spire").string();
    SimulationRecorder recorder(output_path, exc_pop.size);

    constexpr s64 TICK_COUNT = 3;
    Vector<Vector<f32>> staged_frames;
    for (s64 tick = 0; tick < TICK_COUNT; ++tick) {
        Vector<f32> frame(exc_pop.size);
        for (s32 local_index = 0; local_index < exc_pop.size; ++local_index) {
            s64 global_index = exc_pop.neuron_index_begin + local_index;
            f32 synthetic_value = 1.0e-9f * (f32)(tick + 1) * (f32)(local_index + 1);
            scratch[global_index] = synthetic_value;
            frame[(usize)local_index] = synthetic_value;
        }
        staged_frames.push_back(frame);

        Vector<f32> sampled;
        sample_recording_stream(*iSyn_stream, model, allocation, nullptr, tick, sampled);
        recorder.record_frame(sampled.data(), (s64)sampled.size());
    }
    recorder.finish();

    SpireRecording read_back = read_spire_recording(output_path);
    ASSERT_EQ(read_back.frame_count, TICK_COUNT);
    for (s64 tick = 0; tick < TICK_COUNT; ++tick) {
        for (s32 local_index = 0; local_index < exc_pop.size; ++local_index) {
            f32 written = read_back.frames[(usize)(tick * exc_pop.size + local_index)];
            EXPECT_FLOAT_EQ(written, staged_frames[(usize)tick][(usize)local_index]);
        }
    }

    std::filesystem::remove(output_path);
}

TEST(OutputRecording, writes_and_reads_back_a_real_spire_spike_raster) {
    ModelSpecification model = build_output_recording_test_model();
    Vector<IrProgram> programs = build_type_library_ir_programs(model);
    ModelAllocation allocation = allocate_model(model, programs);

    Vector<RecordingStream> streams = resolve_recording_streams(model, programs);
    const RecordingStream *spike_stream = nullptr;
    for (const auto &stream : streams) {
        if (stream.is_event_recording) spike_stream = &stream;
    }
    ASSERT_NE(spike_stream, nullptr);

    const PopulationEntry &inh_pop = population_named(model, "InhPop");
    Vector<s64> last_spiked((usize)model.total_neuron_count, -1);
    // InhPop[0] spikes at tick 2; InhPop[1] spikes at tick 4.
    last_spiked[(usize)(inh_pop.neuron_index_begin + 0)] = 2;
    last_spiked[(usize)(inh_pop.neuron_index_begin + 1)] = 4;

    String output_path = (std::filesystem::temp_directory_path() / "spikecorec_output_recording_spike_raster.spire").string();
    SimulationRecorder recorder(output_path, inh_pop.size);

    constexpr s64 TICK_COUNT = 6;
    Vector<Vector<f32>> expected_frames;
    for (s64 tick = 0; tick < TICK_COUNT; ++tick) {
        Vector<f32> sampled;
        sample_recording_stream(*spike_stream, model, allocation, last_spiked.data(), tick, sampled);
        ASSERT_EQ(sampled.size(), (usize)inh_pop.size);
        expected_frames.push_back(sampled);
        recorder.record_frame(sampled.data(), (s64)sampled.size());
    }
    recorder.finish();

    // Exactly the spikes staged above show up as a 1.0 in the appropriate frame/neuron.
    EXPECT_FLOAT_EQ(expected_frames[2][0], 1.0f);
    EXPECT_FLOAT_EQ(expected_frames[4][1], 1.0f);
    for (s64 tick = 0; tick < TICK_COUNT; ++tick) {
        for (s32 local_index = 0; local_index < inh_pop.size; ++local_index) {
            bool expect_spike = (tick == 2 && local_index == 0) || (tick == 4 && local_index == 1);
            EXPECT_FLOAT_EQ(expected_frames[(usize)tick][(usize)local_index], expect_spike ? 1.0f : 0.0f);
        }
    }

    SpireRecording read_back = read_spire_recording(output_path);
    ASSERT_EQ(read_back.frame_count, TICK_COUNT);
    for (s64 tick = 0; tick < TICK_COUNT; ++tick) {
        for (s32 local_index = 0; local_index < inh_pop.size; ++local_index) {
            f32 written = read_back.frames[(usize)(tick * inh_pop.size + local_index)];
            EXPECT_FLOAT_EQ(written, expected_frames[(usize)tick][(usize)local_index]);
        }
    }

    std::filesystem::remove(output_path);
}

TEST(OutputRecording, throws_when_spike_raster_stream_sampled_with_null_last_spiked) {
    ModelSpecification model = build_output_recording_test_model();
    Vector<IrProgram> programs = build_type_library_ir_programs(model);
    ModelAllocation allocation = allocate_model(model, programs);

    Vector<RecordingStream> streams = resolve_recording_streams(model, programs);
    const RecordingStream *spike_stream = nullptr;
    for (const auto &stream : streams) {
        if (stream.is_event_recording) spike_stream = &stream;
    }
    ASSERT_NE(spike_stream, nullptr);

    Vector<f32> sampled;
    EXPECT_THROW(sample_recording_stream(*spike_stream, model, allocation, nullptr, 0, sampled), std::runtime_error);
}
