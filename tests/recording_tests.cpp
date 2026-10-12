#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <future>
#include <iterator>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

#include "types.h"
#include "recording.h"
#include "support/test_support.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::log;
using namespace spikecorec::test_support;

namespace {

// A sink whose writes wait at a gate until the test opens it, so a test can hold the writer's
// worker inside a write for as long as it needs instead of sleeping.
struct Gated_Sink : Spire_Sink {
    Mutex gate_mutex;
    Condition_Variable gate_opened;
    bool open = false;
    Atomic<int> write_count{0};

    void write(const u8 *, usize) override {
        Unique_Lock<Mutex> lock(gate_mutex);
        gate_opened.wait(lock, [this] { return open; });
        write_count += 1;
    }
    void close() override {}

    void open_gate() {
        {
            Lock_Guard<Mutex> lock(gate_mutex);
            open = true;
        }
        gate_opened.notify_all();
    }
};

// Throws on its failing_write-th write, and counts every call it gets.
struct Failing_Sink : Spire_Sink {
    int failing_write;
    int write_count = 0;
    int close_count = 0;
    explicit Failing_Sink(int failing_write) : failing_write(failing_write) {}

    void write(const u8 *, usize) override {
        write_count += 1;
        if (write_count == failing_write) throw Runtime_Error("sink failure");
    }
    void close() override { close_count += 1; }
};

struct Counting_Sink : Spire_Sink {
    int write_count = 0;
    void write(const u8 *, usize) override { write_count += 1; }
    void close() override {}
};

} // namespace

// ── .spire files ────────────────────────────────────────────────────────────────

TEST(SpireCodec, raw_frames_round_trip) {
    const Temporary_Directory directory;
    const String path = directory.path_of("raw.spire");
    const s64 neuron_count = 6;
    const s64 frame_count = 9;

    Vector<Vector<f32>> frames(static_cast<usize>(frame_count), Vector<f32>(static_cast<usize>(neuron_count)));
    for (s64 tick = 0; tick < frame_count; tick += 1) {
        for (s64 neuron = 0; neuron < neuron_count; neuron += 1) {
            frames[static_cast<usize>(tick)][static_cast<usize>(neuron)] = static_cast<f32>(tick * 100 + neuron) * 0.5f;
        }
    }
    {
        Spire_Writer writer(path, neuron_count);
        for (const Vector<f32> &frame : frames) writer.write_frame(frame.data());
    }

    {
        Spire_Reader reader(path);
        ASSERT_EQ(reader.neuron_count(), neuron_count);
        Vector<f32> buffer(static_cast<usize>(neuron_count));
        s64 tick = 0;
        while (reader.read_frame(buffer.data())) {
            EXPECT_EQ(buffer, frames[static_cast<usize>(tick)]) << "frame " << tick;
            tick += 1;
        }
        EXPECT_EQ(tick, frame_count);
    }

    const Spire_Recording recording = read_spire_recording(path);
    EXPECT_EQ(recording.neuron_count, neuron_count);
    EXPECT_EQ(recording.frame_count, frame_count);
    for (s64 tick = 0; tick < frame_count; tick += 1) {
        for (s64 neuron = 0; neuron < neuron_count; neuron += 1) {
            EXPECT_EQ(recording.frames[static_cast<usize>(tick * neuron_count + neuron)], frames[static_cast<usize>(tick)][static_cast<usize>(neuron)]);
        }
    }
}

TEST(SpireCodec, every_compression_round_trips) {
    const Temporary_Directory directory;
    const usize byte_count = 8000;
    Vector<u8> data(byte_count);
    for (usize index = 0; index < byte_count; index += 1) data[index] = static_cast<u8>((index * 37 + 11) & 0xFF);

    auto round_trip = [&](Spire_Compression compression, const String &file_name) {
        const String path = directory.path_of(file_name);
        {
            Unique_Pointer<Spire_Sink> sink = make_spire_sink(path, compression, 6);
            sink->write(data.data(), byte_count / 3);
            sink->write(data.data() + byte_count / 3, byte_count - byte_count / 3);
            sink->close();
        }
        Unique_Pointer<Spire_Source> source = make_spire_source(path, compression);
        Vector<u8> read_back;
        u8 buffer[4096];
        while (usize read_count = source->read(buffer, sizeof(buffer))) {
            read_back.insert(read_back.end(), buffer, buffer + read_count);
        }
        EXPECT_EQ(read_back, data) << file_name;
    };

#ifdef SPIKECOREC_HAVE_ZLIB
    round_trip(Spire_Compression::Gzip, "compressed.gz");
#endif
#ifdef SPIKECOREC_HAVE_LZMA
    round_trip(Spire_Compression::Xz, "compressed.xz");
#endif
#ifdef SPIKECOREC_HAVE_BZ2
    round_trip(Spire_Compression::Bz2, "compressed.bz2");
#endif
}

TEST(SpireCodec, a_recording_with_no_frames_reads_back_empty) {
    const Temporary_Directory directory;
    const String path = directory.path_of("empty.spire");
    const s64 neuron_count = 5;
    {
        Simulation_Recorder recorder(path, neuron_count, String("none"), nullopt, /*async=*/false);
        recorder.finish();
    }
    const Spire_Recording recording = read_spire_recording(path);
    EXPECT_EQ(recording.neuron_count, neuron_count);
    EXPECT_EQ(recording.frame_count, 0);
    EXPECT_TRUE(recording.frames.empty());
}

TEST(SpireCodec, a_neuron_count_past_the_header_field_throws) {
    const Temporary_Directory directory;
    const s64 too_many = static_cast<s64>(UINT32_MAX) + 1;
    EXPECT_THROW((Simulation_Recorder(directory.path_of("recorder.spire"), too_many, String("none"), nullopt,
                                     /*async=*/false)),
                 exception);
    EXPECT_THROW(Spire_Writer(directory.path_of("writer.spire"), too_many), Exception);
}

TEST(SpireCodec, a_truncated_final_frame_throws) {
    const Temporary_Directory directory;
    const String complete_path = directory.path_of("complete.spire");
    const String truncated_path = directory.path_of("truncated.spire");
    const s64 neuron_count = 4;
    {
        Spire_Writer writer(complete_path, neuron_count);
        for (s64 tick = 0; tick < 5; tick += 1) {
            const Vector<f32> frame(static_cast<usize>(neuron_count), static_cast<f32>(tick));
            writer.write_frame(frame.data());
        }
    }
    {
        Input_File_Stream complete(complete_path, Stream_Base::binary);
        const Vector<char> bytes((Input_Stream_Buffer_Iterator<char>(complete)), Input_Stream_Buffer_Iterator<char>());
        Output_File_Stream truncated(truncated_path, Stream_Base::binary);
        // The 4-byte header, four whole frames, and 6 bytes of the fifth.
        truncated.write(bytes.data(), 4 + neuron_count * 4 * 4 + 6);
    }

    try {
        read_spire_recording(truncated_path);
        FAIL() << "read_spire_recording must throw on a truncated final frame";
    } catch (const Exception &error) {
        EXPECT_NE(String(error.what()).find("Truncated recording"), String::npos) << error.what();
    }
}

TEST(Simulation_Recorder, a_frame_of_the_wrong_length_throws) {
    const Temporary_Directory directory;
    const s64 neuron_count = 8;
    Simulation_Recorder recorder(directory.path_of("frames.spire"), neuron_count, String("none"), nullopt,
                                /*async=*/false);

    const Vector<f32> whole_frame(static_cast<usize>(neuron_count), 1.0f);
    recorder.record_frame(whole_frame.data(), neuron_count);

    const Vector<f32> short_frame(3, 1.0f);
    try {
        recorder.record_frame(short_frame.data(), static_cast<s64>(short_frame.size()));
        FAIL() << "record_frame must reject a frame whose length is not neuron_count";
    } catch (const Exception &error) {
        EXPECT_NE(String(error.what()).find("does not match"), String::npos) << error.what();
    }
    recorder.finish();
}

// ── the asynchronous writer ─────────────────────────────────────────────────────

TEST(Async_Spire_Writer, a_full_queue_blocks_the_producer_until_the_sink_drains) {
    auto owned_sink = make_unique<Gated_Sink>();
    Gated_Sink *sink = owned_sink.get();
    Async_Spire_Writer writer(std::move(owned_sink), /*max_queued_chunks=*/2);

    // With the gate shut the worker holds chunk 0 inside the sink and the queue holds two more,
    // so the producer's fourth write has to wait.
    Atomic<int> accepted_count{0};
    auto producer = std::async(Launch::async, [&] {
        for (int index = 0; index < 8; index += 1) {
            writer.write(Vector<u8>{static_cast<u8>(index)});
            accepted_count += 1;
        }
    });
    const auto deadline = Steady_Clock::now() + Seconds(10);
    while (accepted_count < 3 && Steady_Clock::now() < deadline) this_thread::yield();
    // EXPECT rather than ASSERT throughout: returning early would leave the gate shut and the
    // writer's destructor waiting on its worker forever.
    EXPECT_EQ(accepted_count.load(), 3);
    EXPECT_EQ(producer.wait_for(chrono::milliseconds(100)), future_status::timeout)
        << "the producer finished while the sink was still blocked";
    EXPECT_EQ(accepted_count.load(), 3);

    sink->open_gate();
    producer.get();
    writer.close();
    EXPECT_EQ(sink->write_count.load(), 8);
}

TEST(Async_Spire_Writer, a_sink_failure_reaches_the_producer_and_close) {
    Async_Spire_Writer writer(make_unique<Failing_Sink>(/*failing_write=*/3), /*max_queued_chunks=*/1);

    // The queue holds one chunk, so the producer can get at most two chunks past the failing one
    // before a write sees the failure.
    bool write_threw = false;
    try {
        for (int index = 0; index < 20; index += 1) writer.write(Vector<u8>{static_cast<u8>(index)});
    } catch (const Exception &error) {
        write_threw = true;
        EXPECT_NE(String(error.what()).find("sink failure"), String::npos) << error.what();
    }
    EXPECT_TRUE(write_threw);
    EXPECT_THROW(writer.close(), Exception);
}

TEST(Async_Spire_Writer, a_failed_sink_is_never_called_again) {
    auto owned_sink = make_unique<Failing_Sink>(/*failing_write=*/2);
    Failing_Sink *sink = owned_sink.get();
    Async_Spire_Writer writer(std::move(owned_sink), /*max_queued_chunks=*/1);

    bool failure_surfaced = false;
    try {
        for (int index = 0; index < 10; index += 1) writer.write(Vector<u8>{static_cast<u8>(index)});
    } catch (const Exception &) {
        failure_surfaced = true;
    }
    try {
        writer.close();
    } catch (const Exception &) {
        failure_surfaced = true;
    }
    EXPECT_TRUE(failure_surfaced);
    EXPECT_EQ(sink->write_count, 2);
    EXPECT_EQ(sink->close_count, 0);
}

TEST(Async_Spire_Writer, an_unbounded_queue_takes_every_chunk_without_blocking) {
    auto owned_sink = make_unique<Counting_Sink>();
    Counting_Sink *sink = owned_sink.get();
    Async_Spire_Writer writer(std::move(owned_sink), /*max_queued_chunks=*/0);
    for (int index = 0; index < 16; index += 1) writer.write(Vector<u8>{static_cast<u8>(index)});
    writer.close();
    EXPECT_EQ(sink->write_count, 16);
}
