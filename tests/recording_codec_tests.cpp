// Independent verification of the recording stack: the four compression backends,
// the .spire header, truncation handling, AsyncSpireWriter's bounded queue, and
// SimulationRecorder's chunking/finish semantics.
//
// Everything here is written against the property that matters for a recording:
// what comes back is bit-for-bit what went in, in the order it went in, or an
// error is raised. A codec that drops, reorders or rounds a frame yields data
// that still looks like a plausible simulation, so equality is asserted on the
// raw 32-bit patterns rather than on numeric closeness.
//
// Companion to recording_tests.cpp, which covers the same subsystem's basic
// surface; this file is the adversarial half.

#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

#include "spikecorec/core/types.h"
#include "spikecorec/core/recording.h"

using namespace std;
using namespace spikecorec;

namespace {

// Temporary paths carry this process's id. Several test binaries run on this
// machine at once and a fixed path lets one process truncate or delete a file
// another is mid-way through reading — which surfaces as an unrelated codec
// "failure". Same reasoning as ScopedRunDirectory in exit_model_validation_tests.
const filesystem::path &codec_scratch_directory() {
    static const filesystem::path directory = [] {
        const filesystem::path path =
                filesystem::temp_directory_path() /
                ("spikecorec_recording_codec_" + to_string(getpid()));
        error_code ignored;
        filesystem::remove_all(path, ignored);
        filesystem::create_directories(path);
        return path;
    }();
    return directory;
}

string scratch_path(const string &leaf_name) {
    return (codec_scratch_directory() / leaf_name).string();
}

vector<char> read_whole_file(const string &path) {
    ifstream file(path, ios::binary);
    return vector<char>((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
}

void write_whole_file(const string &path, const char *data, usize byte_count) {
    ofstream file(path, ios::binary | ios::trunc);
    file.write(data, (streamsize)byte_count);
}

f32 float_from_bits(u32 bit_pattern) {
    f32 value;
    memcpy(&value, &bit_pattern, sizeof(f32));
    return value;
}

u32 bits_from_float(f32 value) {
    u32 bit_pattern;
    memcpy(&bit_pattern, &value, sizeof(u32));
    return bit_pattern;
}

// Content chosen to defeat a codec that is subtly broken rather than absent.
// Constant frames compress to a handful of bytes and survive almost any bug, so
// every frame here is a different shape of hostile: signalling and quiet NaNs,
// both infinities, the smallest denormal, both zeros, the extremes of the range,
// a run of alternating extremes that no predictor can smooth over, an all-zero
// frame sitting directly beside a frame of huge magnitudes, and a pseudo-random
// frame that does not compress at all.
vector<vector<f32>> hostile_frames(s64 neuron_count) {
    vector<vector<f32>> frames;

    const u32 special_patterns[] = {
        0x7FC00000u, // quiet NaN
        0x7FA00000u, // signalling NaN
        0x7F800000u, // +infinity
        0xFF800000u, // -infinity
        0x00000001u, // smallest positive denormal
        0x80000001u, // smallest negative denormal
        0x00800000u, // smallest positive normal
        0x007FFFFFu, // largest denormal
        0x00000000u, // +0
        0x80000000u, // -0
        0x7F7FFFFFu, // FLT_MAX
        0xFF7FFFFFu, // -FLT_MAX
    };

    vector<f32> special_frame((usize)neuron_count);
    for (s64 index = 0; index < neuron_count; ++index) {
        special_frame[(usize)index] =
                float_from_bits(special_patterns[(usize)(index % 12)]);
    }
    frames.push_back(special_frame);

    vector<f32> alternating_extremes((usize)neuron_count);
    for (s64 index = 0; index < neuron_count; ++index) {
        alternating_extremes[(usize)index] =
                (index % 2 == 0) ? float_from_bits(0x7F7FFFFFu) : float_from_bits(0xFF7FFFFFu);
    }
    frames.push_back(alternating_extremes);

    frames.push_back(vector<f32>((usize)neuron_count, 0.0f));

    vector<f32> large_magnitudes((usize)neuron_count);
    for (s64 index = 0; index < neuron_count; ++index) {
        large_magnitudes[(usize)index] = (f32)((index + 1) * 1e30);
    }
    frames.push_back(large_magnitudes);

    frames.push_back(vector<f32>((usize)neuron_count, 0.0f));

    // A linear congruential bit pattern: incompressible, and any dropped or
    // reordered word shows up immediately.
    u32 state = 0x12345678u;
    for (s64 frame_index = 0; frame_index < 6; ++frame_index) {
        vector<f32> noise_frame((usize)neuron_count);
        for (s64 index = 0; index < neuron_count; ++index) {
            state = state * 1664525u + 1013904223u;
            noise_frame[(usize)index] = float_from_bits(state);
        }
        frames.push_back(noise_frame);
    }

    return frames;
}

// Bit-exact comparison. EXPECT_EQ on f32 would report every NaN as unequal to
// itself and would call +0 equal to -0 — both wrong for a fidelity check.
void expect_frames_bit_identical(const SpireRecording &recording,
                                 const vector<vector<f32>> &expected,
                                 s64 neuron_count,
                                 const string &context) {
    ASSERT_EQ(recording.neuron_count, neuron_count) << context;
    ASSERT_EQ(recording.frame_count, (s64)expected.size()) << context;
    ASSERT_EQ(recording.frames.size(), expected.size() * (usize)neuron_count) << context;

    for (usize frame_index = 0; frame_index < expected.size(); ++frame_index) {
        for (s64 neuron = 0; neuron < neuron_count; ++neuron) {
            u32 written = bits_from_float(expected[frame_index][(usize)neuron]);
            u32 read_back = bits_from_float(
                    recording.frames[frame_index * (usize)neuron_count + (usize)neuron]);
            ASSERT_EQ(written, read_back)
                    << context << ": frame " << frame_index << " neuron " << neuron
                    << " wrote 0x" << hex << written << " read 0x" << read_back;
        }
    }
}

struct CompressionCase {
    const char *name;
    const char *extension;
    bool available;
};

// The four backends, each flagged with whether its library was actually linked in.
// Absent ones are skipped loudly rather than passed over, so a run cannot imply it
// exercised a codec it never touched.
vector<CompressionCase> compression_cases() {
    return {
        {"raw", ".spire", true},
#ifdef SPIKECOREC_HAVE_ZLIB
        {"gzip", ".spire.gz", true},
#else
        {"gzip", ".spire.gz", false},
#endif
#ifdef SPIKECOREC_HAVE_LZMA
        {"xz", ".spire.xz", true},
#else
        {"xz", ".spire.xz", false},
#endif
#ifdef SPIKECOREC_HAVE_BZ2
        {"bz2", ".spire.bz2", true},
#else
        {"bz2", ".spire.bz2", false},
#endif
    };
}

} // namespace

// ── round-trip fidelity, every backend ───────────────────────────────────────

TEST(RecordingCodec, every_backend_round_trips_hostile_content_bit_exactly) {
    const s64 neuron_count = 17; // deliberately not a power of two
    const vector<vector<f32>> frames = hostile_frames(neuron_count);

    usize exercised_count = 0;
    for (const CompressionCase &compression_case : compression_cases()) {
        if (!compression_case.available) {
            GTEST_LOG_(WARNING) << "SKIPPING " << compression_case.name
                                << ": SPIKECOREC_HAVE_* was not defined at build time, so this "
                                   "backend is NOT compiled in and was NOT exercised";
            continue;
        }
        ++exercised_count;

        const string path = scratch_path(string("roundtrip_") + compression_case.name +
                                         compression_case.extension);
        {
            SimulationRecorder recorder(path, neuron_count, string("auto"), nullopt,
                                        /*async=*/false);
            for (const vector<f32> &frame : frames) {
                recorder.record_frame(frame.data(), neuron_count);
            }
            recorder.finish();
        }

        SpireRecording recording = read_spire_recording(path);
        expect_frames_bit_identical(recording, frames, neuron_count, compression_case.name);
    }

    // Raw is unconditional, so a run that exercised nothing is a broken harness.
    EXPECT_GE(exercised_count, 1u);
    RecordProperty("backends_exercised", (int)exercised_count);
}

// The compressed backends must all agree with raw byte for byte after decoding —
// a codec that quietly rounds or reorders would differ from raw even while
// round-tripping "successfully" against itself.
TEST(RecordingCodec, compressed_backends_decode_to_the_same_bytes_as_raw) {
    const s64 neuron_count = 13;
    const vector<vector<f32>> frames = hostile_frames(neuron_count);

    const string raw_path = scratch_path("agreement_raw.spire");
    {
        SimulationRecorder recorder(raw_path, neuron_count, string("none"), nullopt, false);
        for (const vector<f32> &frame : frames) recorder.record_frame(frame.data(), neuron_count);
        recorder.finish();
    }
    const vector<char> raw_bytes = read_whole_file(raw_path);
    ASSERT_EQ(raw_bytes.size(), sizeof(u32) + frames.size() * (usize)neuron_count * sizeof(f32));

    for (const CompressionCase &compression_case : compression_cases()) {
        if (!compression_case.available || string(compression_case.name) == "raw") continue;

        const string path = scratch_path(string("agreement_") + compression_case.name +
                                         compression_case.extension);
        {
            SimulationRecorder recorder(path, neuron_count, string("auto"), nullopt, false);
            for (const vector<f32> &frame : frames) recorder.record_frame(frame.data(), neuron_count);
            recorder.finish();
        }

        // Decode the whole container through the source seam and compare the
        // decompressed byte stream against the raw file's bytes.
        SpireCompression compression = resolve_spire_compression(path, nullopt).first;
        unique_ptr<SpireSource> source = make_spire_source(path, compression);
        vector<char> decoded;
        u8 staging[4096];
        while (usize byte_count = source->read(staging, sizeof(staging))) {
            decoded.insert(decoded.end(), staging, staging + byte_count);
        }
        EXPECT_EQ(decoded, raw_bytes) << compression_case.name
                                      << " decodes to different bytes than the raw encoder wrote";
    }
}

// A compressed stream whose body is altered must be rejected, not silently
// decoded into different numbers. Raw has no checksum and cannot detect this —
// asserted explicitly so the absence is a documented property rather than a gap.
TEST(RecordingCodec, compressed_backends_reject_a_corrupted_body) {
    const s64 neuron_count = 16;
    const vector<vector<f32>> frames = hostile_frames(neuron_count);

    for (const CompressionCase &compression_case : compression_cases()) {
        if (!compression_case.available) continue;
        const bool is_raw = string(compression_case.name) == "raw";

        const string path = scratch_path(string("corrupt_") + compression_case.name +
                                         compression_case.extension);
        {
            SimulationRecorder recorder(path, neuron_count, string("auto"), nullopt, false);
            for (const vector<f32> &frame : frames) recorder.record_frame(frame.data(), neuron_count);
            recorder.finish();
        }

        vector<char> container = read_whole_file(path);
        ASSERT_GT(container.size(), 8u);
        const usize flip_offset = container.size() / 2;
        container[flip_offset] = (char)(container[flip_offset] ^ 0x01);

        const string corrupted_path = scratch_path(string("corrupted_") + compression_case.name +
                                                   compression_case.extension);
        write_whole_file(corrupted_path, container.data(), container.size());

        if (is_raw) {
            // No integrity check exists in the bare frame stream; a flipped bit
            // becomes a different sample. This is the format's property, and it is
            // exactly why the compressed paths must not lose their checksums.
            SpireRecording recording = read_spire_recording(corrupted_path);
            EXPECT_EQ(recording.frame_count, (s64)frames.size());
            bool any_sample_differs = false;
            for (usize index = 0; index < recording.frames.size(); ++index) {
                u32 expected = bits_from_float(frames[index / (usize)neuron_count]
                                                     [index % (usize)neuron_count]);
                if (bits_from_float(recording.frames[index]) != expected) any_sample_differs = true;
            }
            EXPECT_TRUE(any_sample_differs)
                    << "flipping a byte of a raw .spire must change a sample";
        } else {
            EXPECT_THROW(read_spire_recording(corrupted_path), std::exception)
                    << compression_case.name
                    << " accepted a corrupted stream — its integrity check is not being enforced";
        }
    }
}

// ── the header ───────────────────────────────────────────────────────────────

// The 4-byte count is big-endian. Asserted on the literal bytes, not by
// round-tripping through the reader: writer and reader share one byte-swap
// helper, so a swap that was simply dropped on both sides would round-trip
// perfectly and still produce a file no other .spire tool can read.
TEST(RecordingCodec, header_is_four_bytes_big_endian) {
    const s64 neuron_count = 0x01020304; // every byte distinct
    const string path = scratch_path("header_endianness.spire");
    { SpireWriter writer(path, neuron_count); }

    const vector<char> bytes = read_whole_file(path);
    ASSERT_EQ(bytes.size(), 4u) << "a header-only .spire must be exactly 4 bytes";
    EXPECT_EQ((u8)bytes[0], 0x01u) << "most significant byte must come first";
    EXPECT_EQ((u8)bytes[1], 0x02u);
    EXPECT_EQ((u8)bytes[2], 0x03u);
    EXPECT_EQ((u8)bytes[3], 0x04u) << "least significant byte must come last";

    // And the same through SimulationRecorder, which writes its own header.
    const string recorder_path = scratch_path("header_endianness_recorder.spire");
    {
        SimulationRecorder recorder(recorder_path, neuron_count, string("none"), nullopt, false);
        recorder.finish();
    }
    const vector<char> recorder_bytes = read_whole_file(recorder_path);
    ASSERT_EQ(recorder_bytes.size(), 4u);
    EXPECT_EQ((u8)recorder_bytes[0], 0x01u);
    EXPECT_EQ((u8)recorder_bytes[1], 0x02u);
    EXPECT_EQ((u8)recorder_bytes[2], 0x03u);
    EXPECT_EQ((u8)recorder_bytes[3], 0x04u);
}

// A neuron_count of 0 makes every frame zero bytes wide, so payload after the
// header cannot be accounted for as frames. Returning "0 frames" would hand back
// an empty recording indistinguishable from a legitimately empty run.
TEST(RecordingCodec, zero_neuron_count_header_with_payload_is_rejected) {
    const string path = scratch_path("header_zero_with_payload.spire");
    const char bytes[] = {0, 0, 0, 0, 1, 2, 3, 4};
    write_whole_file(path, bytes, sizeof(bytes));

    EXPECT_THROW(read_spire_recording(path), std::exception);

    // A header-only file with a zero count remains a legitimate empty recording.
    const string empty_path = scratch_path("header_zero_empty.spire");
    write_whole_file(empty_path, bytes, 4);
    SpireRecording recording = read_spire_recording(empty_path);
    EXPECT_EQ(recording.neuron_count, 0);
    EXPECT_EQ(recording.frame_count, 0);
}

// An absurd count must cost the size of what is really in the file, not the size
// the header claims. Before this was fixed, an 8-byte file whose header read
// 0x20000000 drove a 2.15 GB zero-filled allocation (measured) purely because the
// frame buffer was sized from the header — the same shape as a garbage length
// driving an allocation elsewhere in this repo.
TEST(RecordingCodec, absurd_neuron_count_header_does_not_drive_a_huge_allocation) {
    const string path = scratch_path("header_absurd.spire");
    // 0x7FFFFFFF neurons => 8.6 GB a frame, from a 12-byte file.
    const unsigned char bytes[] = {0x7F, 0xFF, 0xFF, 0xFF, 1, 2, 3, 4, 5, 6, 7, 8};
    write_whole_file(path, (const char *)bytes, sizeof(bytes));

    const auto start = chrono::steady_clock::now();
    EXPECT_THROW(read_spire_recording(path), std::exception);
    const auto elapsed_milliseconds =
            chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count();

    // Zero-filling 8.6 GB cannot happen in this budget: the measured cost of the
    // 2.15 GB case alone was ~165 ms, so 8.6 GB is well over a second. Staying
    // under it demonstrates the buffer tracked real bytes rather than the header.
    EXPECT_LT(elapsed_milliseconds, 400)
            << "reading a 12-byte file took " << elapsed_milliseconds
            << " ms — the frame buffer is being sized from the header again";

    // The maximum representable count, from a file with no payload at all, is a
    // legitimate zero-frame recording and must stay cheap too.
    const string header_only_path = scratch_path("header_absurd_empty.spire");
    const unsigned char maximum_header[] = {0xFF, 0xFF, 0xFF, 0xFF};
    write_whole_file(header_only_path, (const char *)maximum_header, sizeof(maximum_header));
    SpireRecording recording = read_spire_recording(header_only_path);
    EXPECT_EQ(recording.neuron_count, (s64)UINT32_MAX);
    EXPECT_EQ(recording.frame_count, 0);
    EXPECT_TRUE(recording.frames.empty());
}

// SpireReader hands neuron_count() to the caller, who sizes its own frame buffer
// from it. A header claiming a frame wider than the whole remaining file is
// corrupt and must be refused before the caller allocates against it.
TEST(RecordingCodec, spire_reader_rejects_a_header_wider_than_the_file) {
    const string path = scratch_path("reader_absurd_header.spire");
    const unsigned char bytes[] = {0xFF, 0xFF, 0xFF, 0xFF, 1, 2, 3, 4, 5, 6, 7, 8};
    write_whole_file(path, (const char *)bytes, sizeof(bytes));

    EXPECT_THROW(SpireReader reader(path), std::exception);

    // A truncated-but-plausible header is NOT this error: it must still be
    // reported by read_frame as a truncated frame, so the two stay distinguishable.
    const string truncated_path = scratch_path("reader_truncated_frame.spire");
    const unsigned char truncated_bytes[] = {0, 0, 0, 2, // 2 neurons => 8 bytes a frame
                                             1, 2, 3, 4, 5, 6, 7, 8, // frame 0, complete
                                             9, 9, 9};               // frame 1, short
    write_whole_file(truncated_path, (const char *)truncated_bytes, sizeof(truncated_bytes));

    SpireReader reader(truncated_path);
    ASSERT_EQ(reader.neuron_count(), 2);
    vector<f32> buffer(2);
    EXPECT_TRUE(reader.read_frame(buffer.data()));
    EXPECT_THROW(reader.read_frame(buffer.data()), std::exception);
}

// ── truncation ───────────────────────────────────────────────────────────────

// Every truncation point of a raw .spire, from mid-header through mid-frame,
// must either yield a clean prefix or raise — never a silently short recording.
TEST(RecordingCodec, raw_truncation_at_every_offset_never_yields_a_silent_short_read) {
    const s64 neuron_count = 3;
    const s64 frame_count = 4;
    const string source_path = scratch_path("truncation_source.spire");
    {
        SpireWriter writer(source_path, neuron_count);
        for (s64 frame_index = 0; frame_index < frame_count; ++frame_index) {
            vector<f32> frame((usize)neuron_count);
            for (s64 neuron = 0; neuron < neuron_count; ++neuron)
                frame[(usize)neuron] = (f32)(frame_index * 10 + neuron) + 0.25f;
            writer.write_frame(frame.data());
        }
    }

    const vector<char> whole = read_whole_file(source_path);
    const usize frame_bytes = (usize)neuron_count * sizeof(f32);
    ASSERT_EQ(whole.size(), sizeof(u32) + (usize)frame_count * frame_bytes);

    const string truncated_path = scratch_path("truncation_cut.spire");
    for (usize kept_bytes = 0; kept_bytes < whole.size(); ++kept_bytes) {
        write_whole_file(truncated_path, whole.data(), kept_bytes);

        const bool header_is_complete = kept_bytes >= sizeof(u32);
        const usize payload_bytes = header_is_complete ? kept_bytes - sizeof(u32) : 0;
        const bool lands_on_a_frame_boundary = payload_bytes % frame_bytes == 0;

        if (header_is_complete && lands_on_a_frame_boundary) {
            // A whole number of frames survived: a valid, shorter recording.
            SpireRecording recording = read_spire_recording(truncated_path);
            EXPECT_EQ(recording.neuron_count, neuron_count) << "kept " << kept_bytes;
            EXPECT_EQ(recording.frame_count, (s64)(payload_bytes / frame_bytes))
                    << "kept " << kept_bytes;
        } else {
            // Mid-header or mid-frame: must raise rather than report a short recording.
            EXPECT_THROW(read_spire_recording(truncated_path), std::exception)
                    << "kept " << kept_bytes
                    << " bytes — a partial header or frame was accepted silently";
        }
    }
}

// The same for each compressed container. Truncation that removes payload must
// raise; truncation that removes only the trailing checksum/footer leaves every
// frame intact, and is asserted to return exactly those frames rather than
// something in between.
TEST(RecordingCodec, compressed_truncation_never_yields_wrong_frames) {
    const s64 neuron_count = 8;
    const vector<vector<f32>> frames = hostile_frames(neuron_count);

    for (const CompressionCase &compression_case : compression_cases()) {
        if (!compression_case.available || string(compression_case.name) == "raw") continue;

        const string path = scratch_path(string("compressed_truncation_") +
                                         compression_case.name + compression_case.extension);
        {
            SimulationRecorder recorder(path, neuron_count, string("auto"), nullopt, false);
            for (const vector<f32> &frame : frames) recorder.record_frame(frame.data(), neuron_count);
            recorder.finish();
        }

        const vector<char> whole = read_whole_file(path);
        ASSERT_GT(whole.size(), 8u) << compression_case.name;

        const string truncated_path = scratch_path(string("compressed_truncation_cut_") +
                                                   compression_case.name +
                                                   compression_case.extension);
        // Includes 0 and 2 (mid-header of the compressed container itself).
        const usize cut_points[] = {0, 2, whole.size() / 4, whole.size() / 2,
                                    whole.size() * 3 / 4, whole.size() - 1};
        for (usize kept_bytes : cut_points) {
            write_whole_file(truncated_path, whole.data(), kept_bytes);

            SpireRecording recording;
            bool threw = false;
            try {
                recording = read_spire_recording(truncated_path);
            } catch (const std::exception &) {
                threw = true;
            }

            if (threw) continue; // rejecting a truncated container is always acceptable

            // It returned frames, so every one of them must be correct and must be
            // a prefix of what was written — a truncated container may lose the
            // tail, never alter the head.
            ASSERT_EQ(recording.neuron_count, neuron_count)
                    << compression_case.name << " kept=" << kept_bytes;
            ASSERT_LE(recording.frame_count, (s64)frames.size())
                    << compression_case.name << " kept=" << kept_bytes
                    << " invented frames that were never written";
            for (s64 frame_index = 0; frame_index < recording.frame_count; ++frame_index) {
                for (s64 neuron = 0; neuron < neuron_count; ++neuron) {
                    u32 expected = bits_from_float(frames[(usize)frame_index][(usize)neuron]);
                    u32 actual = bits_from_float(
                            recording.frames[(usize)(frame_index * neuron_count + neuron)]);
                    ASSERT_EQ(expected, actual)
                            << compression_case.name << " kept=" << kept_bytes
                            << ": frame " << frame_index << " neuron " << neuron
                            << " came back altered";
                }
            }
        }
    }
}

// ── AsyncSpireWriter ─────────────────────────────────────────────────────────

// The queue is bounded and the producer blocks when it is full. Asserted by
// watching the depth the worker actually observes rather than by wall-clock
// timing: a timing assertion passes just as happily when the bound is removed
// and the machine is merely busy.
TEST(AsyncSpireWriterVerification, bounded_queue_blocks_the_producer) {
    struct DepthProbingSink : SpireSink {
        mutex guard;
        usize peak_backlog = 0;
        usize completed_writes = 0;
        vector<u8> received;

        void write(const u8 *data, usize byte_count) override {
            // Hold each chunk long enough that an unbounded producer would run
            // far ahead; the backlog it builds is what the bound is supposed to cap.
            this_thread::sleep_for(chrono::milliseconds(2));
            lock_guard<mutex> lock(guard);
            received.insert(received.end(), data, data + byte_count);
            ++completed_writes;
        }
        void close() override {}
    };

    constexpr usize max_queued_chunks = 3;
    constexpr usize chunk_count = 60;

    auto sink = make_unique<DepthProbingSink>();
    DepthProbingSink *sink_view = sink.get();
    AsyncSpireWriter writer(std::move(sink), max_queued_chunks);

    usize peak_outstanding = 0;
    for (usize chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
        writer.write(vector<u8>{(u8)chunk_index});
        // Chunks handed over minus chunks the sink has finished bounds the number
        // sitting in (or just leaving) the queue. With backpressure this stays at
        // the bound plus the one in flight; without it, it climbs to chunk_count.
        usize completed = 0;
        {
            lock_guard<mutex> lock(sink_view->guard);
            completed = sink_view->completed_writes;
        }
        usize outstanding = chunk_index + 1 - completed;
        peak_outstanding = std::max(peak_outstanding, outstanding);
    }
    writer.close();

    EXPECT_LE(peak_outstanding, max_queued_chunks + 2)
            << "the producer handed over " << peak_outstanding
            << " chunks ahead of the sink with a bound of " << max_queued_chunks
            << " — it is not blocking when the queue is full";
    EXPECT_EQ(sink_view->completed_writes, chunk_count);
}

// Saturating the queue must lose nothing and reorder nothing.
TEST(AsyncSpireWriterVerification, saturation_loses_and_reorders_nothing) {
    struct OrderRecordingSink : SpireSink {
        vector<u8> received;
        void write(const u8 *data, usize byte_count) override {
            received.insert(received.end(), data, data + byte_count);
        }
        void close() override {}
    };

    constexpr usize chunk_count = 2000;
    auto sink = make_unique<OrderRecordingSink>();
    OrderRecordingSink *sink_view = sink.get();
    AsyncSpireWriter writer(std::move(sink), /*max_queued_chunks=*/1);

    vector<u8> expected;
    for (usize chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
        // Two bytes a chunk so a swapped pair of chunks is visible as well as a
        // dropped one.
        vector<u8> chunk{(u8)(chunk_index & 0xFF), (u8)((chunk_index >> 8) & 0xFF)};
        expected.insert(expected.end(), chunk.begin(), chunk.end());
        writer.write(std::move(chunk));
    }
    writer.close();

    ASSERT_EQ(sink_view->received.size(), expected.size()) << "chunks were dropped under saturation";
    EXPECT_EQ(sink_view->received, expected) << "chunks were reordered under saturation";
}

// A worker-thread failure surfaces on the producer thread, from write() or from
// close(), and is never swallowed.
TEST(AsyncSpireWriterVerification, worker_failure_surfaces_on_the_producer_thread) {
    struct ThrowOnFirstWriteSink : SpireSink {
        void write(const u8 *, usize) override { throw std::runtime_error("worker sink exploded"); }
        void close() override {}
    };

    // Surfaced from a later write().
    {
        AsyncSpireWriter writer(make_unique<ThrowOnFirstWriteSink>(), /*max_queued_chunks=*/1);
        string message;
        for (int attempt = 0; attempt < 200 && message.empty(); ++attempt) {
            try {
                writer.write(vector<u8>{1});
                this_thread::sleep_for(chrono::milliseconds(1));
            } catch (const std::exception &error) {
                message = error.what();
            }
        }
        EXPECT_NE(message.find("worker sink exploded"), string::npos)
                << "the worker's exception never reached the producer via write()";
        EXPECT_THROW(writer.close(), std::exception);
    }

    // Surfaced from close() when the producer never writes again after the failure.
    {
        AsyncSpireWriter writer(make_unique<ThrowOnFirstWriteSink>(), /*max_queued_chunks=*/8);
        writer.write(vector<u8>{1});
        string message;
        try {
            writer.close();
            FAIL() << "close() must re-raise the worker's exception";
        } catch (const std::exception &error) {
            message = error.what();
        }
        EXPECT_NE(message.find("worker sink exploded"), string::npos);
    }
}

// A failing sink must not swallow the failure just because the producer is
// blocked on a full queue: the waiting write() has to wake and raise.
TEST(AsyncSpireWriterVerification, a_producer_blocked_on_a_full_queue_still_learns_of_the_failure) {
    struct BlockThenThrowSink : SpireSink {
        void write(const u8 *, usize) override {
            this_thread::sleep_for(chrono::milliseconds(5));
            throw std::runtime_error("sink failed while the producer waited");
        }
        void close() override {}
    };

    AsyncSpireWriter writer(make_unique<BlockThenThrowSink>(), /*max_queued_chunks=*/1);
    bool surfaced = false;
    // Enough writes that the producer is certain to have blocked on the bound.
    for (int attempt = 0; attempt < 500 && !surfaced; ++attempt) {
        try {
            writer.write(vector<u8>{(u8)attempt});
        } catch (const std::exception &error) {
            surfaced = string(error.what()).find("sink failed while the producer waited") != string::npos;
        }
    }
    EXPECT_TRUE(surfaced) << "the producer never learned the worker had failed";
    EXPECT_THROW(writer.close(), std::exception);
}

// ── SimulationRecorder chunking and lifetime ─────────────────────────────────

namespace {

// Records `frame_count` frames with the given chunk size and asserts a bit-exact
// round trip. Chunking is invisible in a correct implementation, which is the
// point: a boundary bug shows up as a duplicated or dropped frame, not an error.
void verify_chunking(usize chunk_bytes, s64 neuron_count, s64 frame_count,
                     bool asynchronous, const string &label) {
    const string path = scratch_path("chunking_" + label + ".spire");

    vector<vector<f32>> frames;
    u32 state = 0xABCDEF01u;
    for (s64 frame_index = 0; frame_index < frame_count; ++frame_index) {
        vector<f32> frame((usize)neuron_count);
        for (s64 neuron = 0; neuron < neuron_count; ++neuron) {
            state = state * 1664525u + 1013904223u;
            frame[(usize)neuron] = float_from_bits(state);
        }
        frames.push_back(frame);
    }

    {
        SimulationRecorder recorder(path, neuron_count, string("none"), nullopt,
                                    asynchronous, /*queue_max=*/2, chunk_bytes);
        for (const vector<f32> &frame : frames) recorder.record_frame(frame.data(), neuron_count);
        recorder.finish();
    }

    const vector<char> whole = read_whole_file(path);
    EXPECT_EQ(whole.size(), sizeof(u32) + (usize)frame_count * (usize)neuron_count * sizeof(f32))
            << label << ": on-disk size is wrong, so a chunk was dropped or written twice";

    SpireRecording recording = read_spire_recording(path);
    expect_frames_bit_identical(recording, frames, neuron_count, label);
}

} // namespace

TEST(SimulationRecorderVerification, chunk_boundaries_are_exact) {
    const s64 neuron_count = 16;                                    // 64 bytes a frame
    const usize frame_bytes = (usize)neuron_count * sizeof(f32);

    // The header is 4 bytes, so a buffer of header + N frames is never a multiple
    // of the frame size. Cover the boundary from both sides and dead on.

    // Flush triggers exactly at the end of a frame: chunk == 4 frames' bytes, and
    // the recording is a whole number of chunks.
    verify_chunking(frame_bytes * 4, neuron_count, 16, false, "exact_multiple");

    // Flush triggers in the middle of the frame stream: chunk size shares no
    // factor with the frame size.
    verify_chunking(frame_bytes * 4 + 1, neuron_count, 16, false, "straddling_high");
    verify_chunking(frame_bytes * 4 - 1, neuron_count, 16, false, "straddling_low");

    // Smaller than one chunk: nothing is written until finish().
    verify_chunking(4 * 1024 * 1024, neuron_count, 8, false, "smaller_than_chunk");

    // A chunk smaller than a single frame forces a flush on every frame.
    verify_chunking(1, neuron_count, 12, false, "chunk_below_frame_size");

    // The 4 MB default, crossed for real: 4 MB / 64 bytes = 65536 frames a chunk,
    // so this straddles the third boundary.
    verify_chunking(4 * 1024 * 1024, neuron_count, 65536 * 2 + 7, false, "default_chunk_crossed");

    // And the same boundaries through the async path, where a chunk is also a
    // queue entry.
    verify_chunking(frame_bytes * 4, neuron_count, 16, true, "async_exact_multiple");
    verify_chunking(frame_bytes * 4 + 1, neuron_count, 16, true, "async_straddling");
}

// Nothing reaches the sink before the chunk threshold is crossed — the property
// that makes the "smaller than a chunk" case meaningful.
TEST(SimulationRecorderVerification, a_sub_chunk_recording_is_written_only_at_finish) {
    const s64 neuron_count = 4;
    const string path = scratch_path("sub_chunk_deferred.spire");

    SimulationRecorder recorder(path, neuron_count, string("none"), nullopt, false,
                                /*queue_max=*/8, /*chunk_bytes=*/4 * 1024 * 1024);
    vector<f32> frame((usize)neuron_count, 1.5f);
    for (int frame_index = 0; frame_index < 10; ++frame_index) {
        recorder.record_frame(frame.data(), neuron_count);
    }

    EXPECT_EQ(read_whole_file(path).size(), 0u)
            << "bytes reached the file before the chunk threshold was crossed";

    recorder.finish();
    EXPECT_EQ(read_whole_file(path).size(),
              sizeof(u32) + 10u * (usize)neuron_count * sizeof(f32));
}

TEST(SimulationRecorderVerification, finish_is_idempotent) {
    const s64 neuron_count = 5;
    const s64 frame_count = 7;
    for (bool asynchronous : {false, true}) {
        const string path = scratch_path(asynchronous ? "idempotent_async.spire"
                                                      : "idempotent_sync.spire");
        vector<f32> frame((usize)neuron_count, 2.5f);
        {
            SimulationRecorder recorder(path, neuron_count, string("none"), nullopt, asynchronous,
                                        /*queue_max=*/2, /*chunk_bytes=*/16);
            for (s64 frame_index = 0; frame_index < frame_count; ++frame_index)
                recorder.record_frame(frame.data(), neuron_count);

            recorder.finish();
            // Repeated finishes must not double-flush, must not re-close the sink
            // (AsyncSpireWriter::close() raises if called twice), and must not throw.
            EXPECT_NO_THROW(recorder.finish());
            EXPECT_NO_THROW(recorder.finish());
            EXPECT_NO_THROW(recorder.finish());
            // The destructor must not close a second time either.
        }

        SpireRecording recording = read_spire_recording(path);
        EXPECT_EQ(recording.neuron_count, neuron_count);
        EXPECT_EQ(recording.frame_count, frame_count)
                << (asynchronous ? "async" : "sync")
                << ": a repeated finish() changed how many frames landed";
        for (f32 sample : recording.frames) EXPECT_EQ(bits_from_float(sample), bits_from_float(2.5f));
    }
}

TEST(SimulationRecorderVerification, a_recorder_destroyed_without_finish_still_produces_a_readable_file) {
    const s64 neuron_count = 6;
    const s64 frame_count = 9;

    for (bool asynchronous : {false, true}) {
        const string path = scratch_path(asynchronous ? "no_finish_async.spire"
                                                      : "no_finish_sync.spire");
        vector<vector<f32>> frames;
        {
            // Chunk size larger than the whole recording, so everything is still
            // sitting in the buffer when the destructor runs — if the destructor
            // did not flush, the file would hold only the header.
            SimulationRecorder recorder(path, neuron_count, string("none"), nullopt, asynchronous,
                                        /*queue_max=*/4, /*chunk_bytes=*/1024 * 1024);
            for (s64 frame_index = 0; frame_index < frame_count; ++frame_index) {
                vector<f32> frame((usize)neuron_count);
                for (s64 neuron = 0; neuron < neuron_count; ++neuron)
                    frame[(usize)neuron] = (f32)(frame_index * 100 + neuron) * 0.125f;
                frames.push_back(frame);
                recorder.record_frame(frame.data(), neuron_count);
            }
            // No finish() — the destructor is the only thing that can flush.
        }

        SpireRecording recording = read_spire_recording(path);
        expect_frames_bit_identical(recording, frames, neuron_count,
                                    asynchronous ? "destructor_async" : "destructor_sync");
    }
}

// ── declared output format ───────────────────────────────────────────────────

// The recorder writes SPIRE frames. Handing it a declared format it cannot
// produce must be an error, not a silent substitution: a caller that asked for
// the NML-standard column matrix and received binary frames has a file whose
// contents contradict its declaration, and nothing downstream would notice.
TEST(SimulationRecorderVerification, an_unimplemented_declared_format_is_refused) {
    const string path = scratch_path("declared_format.dat");

    EXPECT_THROW(
        (SimulationRecorder(path, 4, string("none"), nullopt, /*async=*/false,
                            /*queue_max=*/8, /*chunk_bytes=*/4096,
                            OutputFileFormat::NML_STANDARD)),
        std::exception);

    // Every format that IS a SPIRE frame stream is accepted, including under a
    // non-.spire filename — which is what the .dat fixtures in this repo rely on.
    for (OutputFileFormat format : {OutputFileFormat::SPIRE, OutputFileFormat::SPIREGZIP,
                                    OutputFileFormat::SPIREBZ2, OutputFileFormat::SPIREXZ,
                                    OutputFileFormat::SPIKE_EVENTS}) {
        SimulationRecorder recorder(path, 4, string("none"), nullopt, false, 8, 4096, format);
        vector<f32> frame(4, 1.0f);
        recorder.record_frame(frame.data(), 4);
        recorder.finish();
        EXPECT_EQ(read_spire_recording(path).frame_count, 1);
    }
}
