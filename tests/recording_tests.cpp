#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

#include "spikecorec/core/types.h"
#include "spikecorec/core/recording.h"
#include "spikecorec/core/engine.h"
#include "spikecorec/core/topologies.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::log;

namespace {

string read_file(const string &path) {
    ifstream file(path);
    return string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
}

} // namespace

TEST(SpireCodec, raw_roundtrip) {
    const string path = "/tmp/spikecorec_test_raw.spire";
    const s64 neuron_count = 6;
    const s64 frame_count = 9;

    vector<vector<f32>> frames((usize)frame_count, vector<f32>((usize)neuron_count));
    for (s64 tick = 0; tick < frame_count; ++tick)
        for (s64 neuron = 0; neuron < neuron_count; ++neuron)
            frames[(usize)tick][(usize)neuron] = (f32)(tick * 100 + neuron) * 0.5f;

    {
        SpireWriter writer(path, neuron_count);
        for (auto &frame : frames) writer.write_frame(frame.data());
    }

    {
        SpireReader reader(path);
        ASSERT_EQ(reader.neuron_count(), neuron_count);
        vector<f32> buffer((usize)neuron_count);
        s64 tick = 0;
        while (reader.read_frame(buffer.data())) {
            for (s64 neuron = 0; neuron < neuron_count; ++neuron)
                EXPECT_EQ(buffer[(usize)neuron], frames[(usize)tick][(usize)neuron]);
            ++tick;
        }
        EXPECT_EQ(tick, frame_count);
    }

    SpireRecording recording = read_spire_recording(path);
    EXPECT_EQ(recording.neuron_count, neuron_count);
    EXPECT_EQ(recording.frame_count, frame_count);
    for (s64 tick = 0; tick < frame_count; ++tick)
        for (s64 neuron = 0; neuron < neuron_count; ++neuron)
            EXPECT_EQ(recording.frames[(usize)(tick * neuron_count + neuron)],
                      frames[(usize)tick][(usize)neuron]);
}

TEST(SpireCodec, compression_roundtrip) {
    const usize byte_count = 8000;
    vector<u8> data(byte_count);
    for (usize index = 0; index < byte_count; ++index)
        data[index] = (u8)((index * 37 + 11) & 0xFF);

    auto roundtrip = [&](SpireCompression compression, const char *path) {
        {
            auto sink = make_spire_sink(path, compression, 6);
            sink->write(data.data(), byte_count / 3);
            sink->write(data.data() + byte_count / 3, byte_count - byte_count / 3);
            sink->close();
        }
        auto source = make_spire_source(path, compression);
        vector<u8> readback;
        u8 buf[4096];
        while (usize got = source->read(buf, sizeof(buf)))
            readback.insert(readback.end(), buf, buf + got);
        EXPECT_EQ(readback, data);
    };

#ifdef SPIKECOREC_HAVE_ZLIB
    roundtrip(SpireCompression::Gzip, "/tmp/spikecorec_test_compress.gz");
#endif
#ifdef SPIKECOREC_HAVE_LZMA
    roundtrip(SpireCompression::Xz, "/tmp/spikecorec_test_compress.xz");
#endif
#ifdef SPIKECOREC_HAVE_BZ2
    roundtrip(SpireCompression::Bz2, "/tmp/spikecorec_test_compress.bz2");
#endif
}

TEST(AsyncSpireWriter, backpressure_and_error_propagation) {
    struct SlowSink : SpireSink {
        int write_count = 0;
        void write(const u8 *, usize) override {
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
            ++write_count;
        }
        void close() override {}
    };
    {
        auto sink = make_unique<SlowSink>();
        SlowSink *raw = sink.get();
        AsyncSpireWriter writer(std::move(sink), /*max_queued_chunks=*/2);

        auto start = std::chrono::steady_clock::now();
        for (int index = 0; index < 8; ++index) writer.write(vector<u8>{(u8)index});
        writer.close();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();

        EXPECT_EQ(raw->write_count, 8);
        EXPECT_GE(elapsed, 8 * 15 - 20);
    }

    struct FlakySink : SpireSink {
        int call_count = 0;
        void write(const u8 *, usize) override {
            if (++call_count == 3) throw std::runtime_error("flaky sink failure");
        }
        void close() override {}
    };
    {
        AsyncSpireWriter writer(make_unique<FlakySink>(), /*max_queued_chunks=*/1);
        bool threw = false;
        try {
            for (int index = 0; index < 20; ++index) {
                writer.write(vector<u8>{(u8)index});
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        } catch (const std::exception &exception) {
            threw = true;
            EXPECT_NE(string(exception.what()).find("flaky sink failure"), string::npos);
        }
        EXPECT_TRUE(threw);

        bool close_threw = false;
        try { writer.close(); } catch (const std::exception &) { close_threw = true; }
        EXPECT_TRUE(close_threw);
    }
}

TEST(AsyncSpireWriter, unbounded_queue) {
    struct CountingSink : SpireSink {
        int write_count = 0;
        void write(const u8 *, usize) override { ++write_count; }
        void close() override {}
    };
    auto sink = make_unique<CountingSink>();
    CountingSink *raw = sink.get();
    AsyncSpireWriter writer(std::move(sink), /*max_queued_chunks=*/0);

    for (int index = 0; index < 16; ++index) writer.write(vector<u8>{(u8)index});
    writer.close();

    EXPECT_EQ(raw->write_count, 16);
}

TEST(AsyncSpireWriter, no_reentry_after_error) {
    struct CountingFailSink : SpireSink {
        int write_count = 0;
        int close_count = 0;
        int fail_on;
        explicit CountingFailSink(int fail_on_write) : fail_on(fail_on_write) {}
        void write(const u8 *, usize) override {
            if (++write_count == fail_on) throw std::runtime_error("sink boom");
        }
        void close() override { ++close_count; }
    };
    auto sink = make_unique<CountingFailSink>(/*fail_on=*/2);
    CountingFailSink *raw = sink.get();
    AsyncSpireWriter writer(std::move(sink), /*max_queued_chunks=*/1);

    bool surfaced = false;
    try {
        for (int index = 0; index < 10; ++index) {
            writer.write(vector<u8>{(u8)index});
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
    } catch (const std::exception &) { surfaced = true; }
    try { writer.close(); } catch (const std::exception &) { surfaced = true; }

    EXPECT_TRUE(surfaced);
    EXPECT_EQ(raw->write_count, 2);
    EXPECT_EQ(raw->close_count, 0);
}

TEST(SpireCodec, zero_frame_recording) {
    const string path = "/tmp/spikecorec_test_zero_frame.spire";
    const s64 neuron_count = 5;
    {
        SimulationRecorder recorder(path, neuron_count, string("none"), nullopt, /*async=*/false);
        recorder.finish();
    }
    SpireRecording recording = read_spire_recording(path);
    EXPECT_EQ(recording.neuron_count, neuron_count);
    EXPECT_EQ(recording.frame_count, 0);
    EXPECT_TRUE(recording.frames.empty());
}

TEST(SpireCodec, header_neuron_count_overflow) {
    const s64 too_big = (s64)UINT32_MAX + 1;

    EXPECT_THROW(
        (SimulationRecorder("/tmp/spikecorec_test_overflow.spire", too_big,
                            string("none"), nullopt, /*async=*/false)),
        std::exception
    );

    EXPECT_THROW(
        SpireWriter("/tmp/spikecorec_test_overflow2.spire", too_big),
        std::exception
    );
}

TEST(SpireCodec, truncated_decode_error) {
    const string good_path = "/tmp/spikecorec_test_trunc_source.spire";
    const string truncated_path = "/tmp/spikecorec_test_truncated.spire";
    const s64 neuron_count = 4;

    {
        SpireWriter writer(good_path, neuron_count);
        for (s64 tick = 0; tick < 5; ++tick) {
            vector<f32> frame((usize)neuron_count, (f32)tick);
            writer.write_frame(frame.data());
        }
    }

    {
        ifstream source_file(good_path, ios::binary);
        vector<char> all((istreambuf_iterator<char>(source_file)), istreambuf_iterator<char>());
        ofstream destination(truncated_path, ios::binary);
        destination.write(all.data(), 4 + neuron_count * 4 * 4 + 6);
    }

    string log_before = read_file("logs/spikecorec.log");
    try {
        read_spire_recording(truncated_path);
        FAIL() << "read_spire_recording must throw on a truncated final frame";
    } catch (const std::exception &exception) {
        EXPECT_NE(string(exception.what()).find("Truncated recording"), string::npos);
    }
    string appended = read_file("logs/spikecorec.log").substr(log_before.size());
    EXPECT_NE(appended.find("Truncated recording"), string::npos);
}

TEST(SimulationRecorder, record_frame_size_validation) {
    const string path = "/tmp/spikecorec_test_record_frame_size.spire";
    const s64 neuron_count = 8;
    SimulationRecorder recorder(path, neuron_count, string("none"), nullopt, /*async=*/false);

    vector<f32> correct_frame((usize)neuron_count, 1.0f);
    recorder.record_frame(correct_frame.data(), neuron_count);

    vector<f32> too_small(3, 1.0f);
    string log_before = read_file("logs/spikecorec.log");
    try {
        recorder.record_frame(too_small.data(), (s64)too_small.size());
        FAIL() << "record_frame must reject a frame whose length != neuron_count";
    } catch (const std::exception &exception) {
        EXPECT_NE(string(exception.what()).find("does not match"), string::npos);
    }
    string appended = read_file("logs/spikecorec.log").substr(log_before.size());
    EXPECT_NE(appended.find("does not match"), string::npos);

    recorder.finish();
}

TEST(SpikeEngine, start_static_record) {
    auto network = square_torus(4);
    SpikeEngine engine(&network, {4, 4}, /*rank=*/4);
    engine.set_input_neurons({0, 1, 2});

    const s64 lifetime = 30;
    const s64 stride = 2;
    vector<vector<f32>> input_spikes((usize)lifetime, vector<f32>{2.0f, 2.0f, 2.0f});

    const string path = "/tmp/spikecorec_test_record.spire";
    engine.start_static_record(input_spikes, lifetime, path, /*record_membrane=*/true, stride,
                               /*compression=*/string("none"), nullopt, /*full_decay=*/true,
                               /*compression_async=*/false);

    s64 expected_frame_count = (lifetime + stride - 1) / stride;
    s64 expected_size = 4 + expected_frame_count * engine.neuron_count * (s64)sizeof(f32);

    FILE *file = fopen(path.c_str(), "rb");
    ASSERT_NE(file, nullptr);
    fseek(file, 0, SEEK_END);
    long actual_size = ftell(file);
    fclose(file);
    EXPECT_EQ((s64)actual_size, expected_size);

    SpireRecording recording = read_spire_recording(path);
    EXPECT_EQ(recording.neuron_count, engine.neuron_count);
    EXPECT_EQ(recording.frame_count, expected_frame_count);
    for (f32 value : recording.frames) EXPECT_TRUE(std::isfinite(value));

    engine.shutdown();
}

TEST(SpikeEngine, start_static_record_bad_input_width) {
    auto network = square_torus(4);
    SpikeEngine engine(&network, {4, 4}, /*rank=*/4);
    engine.set_input_neurons({0, 1, 2});

    const s64 lifetime = 5;
    const string path = "/tmp/spikecorec_test_bad_width.spire";

    {
        vector<vector<f32>> wide((usize)lifetime, vector<f32>{1.0f, 1.0f, 1.0f});
        wide[2] = vector<f32>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        string log_before = read_file("logs/spikecorec.log");
        try {
            engine.start_static_record(wide, lifetime, path, true, 1, string("none"), nullopt, true, false);
            FAIL() << "start_static_record must reject an over-wide input row";
        } catch (const std::exception &exception) {
            EXPECT_NE(string(exception.what()).find("input neurons"), string::npos);
        }
        string appended = read_file("logs/spikecorec.log").substr(log_before.size());
        EXPECT_NE(appended.find("input neurons"), string::npos);
    }

    {
        vector<vector<f32>> empty_row((usize)lifetime, vector<f32>{1.0f, 1.0f, 1.0f});
        empty_row[1].clear();
        EXPECT_THROW(
            engine.start_static_record(empty_row, lifetime, path, true, 1, string("none"), nullopt, true, false),
            std::exception
        );
    }

    engine.shutdown();
}

TEST(SpikeEngine, start_static_record_variants) {
    auto network = square_torus(4);
    SpikeEngine engine(&network, {4, 4}, /*rank=*/8);
    engine.set_input_neurons({0, 1, 2});

    const s64 lifetime = 12;
    vector<vector<f32>> input((usize)lifetime, vector<f32>{1.0f, 1.0f, 1.0f});

    {
        const string path = "/tmp/spikecorec_test_norec.spire";
        engine.start_static_record(input, lifetime, path, /*record_membrane=*/false);
        EXPECT_EQ(read_spire_recording(path).frame_count, 0);
    }
    engine.reset_state();
    {
        const string path = "/tmp/spikecorec_test_stride.spire";
        engine.start_static_record(input, lifetime, path, /*record_membrane=*/true,
                                   /*record_stride=*/4);
        SpireRecording recording = read_spire_recording(path);
        EXPECT_EQ(recording.frame_count, 3);
        for (f32 value : recording.frames) EXPECT_TRUE(std::isfinite(value));
    }
    engine.reset_state();
    {
        const string path = "/tmp/spikecorec_test_nodecay.spire";
        engine.start_static_record(input, lifetime, path, true, 1, string("none"), nullopt,
                                   /*full_decay=*/false);
        SpireRecording recording = read_spire_recording(path);
        EXPECT_EQ(recording.frame_count, lifetime);
        EXPECT_EQ(recording.neuron_count, engine.neuron_count);
        for (f32 value : recording.frames) EXPECT_TRUE(std::isfinite(value));
    }

    engine.shutdown();
}
