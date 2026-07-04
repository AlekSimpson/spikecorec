#ifdef SPIKECOREC_CUDA
#include <cuda_runtime.h>
#elif defined(SPIKECOREC_METAL)
#include <Metal/Metal.hpp>
#endif

#include <cassert>
#include <cctype>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iterator>
#include <thread>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <unordered_set>

#include "spikecorec/core/types.h"
#include "spikecorec/core/backend.h"
#include "spikecorec/core/engine.h"
#include "spikecorec/core/weight_matrix.h"
#include "spikecorec/core/k2tree.h"
#include "spikecorec/core/topologies.h"
#include "spikecorec/core/recording.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::log;

namespace {

string read_file(const string &path) {
    ifstream file(path);
    return string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
}

// ── topology smoke tests ─────────────────────────────────────────────────────

void test_square_torus() {
    auto net = square_torus(4);
    assert(net.size() == 16);
    for (s64 i = 0; i < (s64)net.size(); ++i) {
        assert(net[i].size() == 4);
        unordered_set<s32> seen;
        for (s32 child : net[i]) {
            assert(child >= 0 && child < 16);
            assert(child != (s32)i && "square_torus must not contain self-loops");
            assert(seen.insert(child).second && "square_torus must not contain duplicate edges");
        }
    }
    printf("  square_torus: ok\n");
}

void test_small_world_torus() {
    auto net = small_world_torus(4, /*random_fanout=*/4, /*seed=*/42);
    assert(net.size() == 16);
    for (s64 i = 0; i < (s64)net.size(); ++i) {
        assert(net[i].size() == 8); // 4 torus neighbors + 4 random shortcuts
        unordered_set<s32> seen;
        for (s32 child : net[i]) {
            assert(child >= 0 && child < 16);
            assert(child != (s32)i && "small_world_torus must not contain self-loops");
            assert(seen.insert(child).second && "small_world_torus must not contain duplicate edges");
        }
    }
    printf("  small_world_torus: ok\n");
}

void test_random_fixed_outdegree() {
    auto net = random_fixed_outdegree(4, /*fanout=*/6, /*seed=*/42);
    assert(net.size() == 16);
    for (s64 i = 0; i < (s64)net.size(); ++i) {
        assert(net[i].size() == 6);
        unordered_set<s32> seen;
        for (s32 child : net[i]) {
            assert(child >= 0 && child < 16);
            assert(child != (s32)i && "random_fixed_outdegree must not contain self-loops");
            assert(seen.insert(child).second && "random_fixed_outdegree must not contain duplicate edges");
        }
    }
    printf("  random_fixed_outdegree: ok\n");
}

// ── engine smoke tests ───────────────────────────────────────────────────────

void test_construction() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4});
    assert(engine.neuron_count == 16);
    assert(engine.is_alive());
    engine.shutdown();
    assert(!engine.is_alive());
    printf("  construction: ok\n");
}

void test_step_loop() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/4);

    engine.set_input_neurons({0, 1, 2});
    assert(engine.input_neuron_count == 3);

    bool moved_from_resting = false;
    for (s64 tick = 0; tick < 10; ++tick) {
        engine.step_simulation({2.0f, 2.0f, 2.0f}, tick);

        const f32 *mp = engine.membrane_potentials.get_contents();
        for (s64 i = 0; i < engine.neuron_count; ++i) {
            assert(std::isfinite(mp[i]));
            if (std::fabs(mp[i] - engine.resting_membrane_potential) > 1e-6f)
                moved_from_resting = true;
        }

        s32 active_count = engine.active_neuron_count.get_contents()[0];
        assert(active_count >= 0 && active_count <= engine.neuron_count);
        const s32 *active_indices = engine.active_neuron_indices.get_contents();
        for (s32 i = 0; i < active_count; ++i) {
            assert(active_indices[i] >= 0 && active_indices[i] < engine.neuron_count);
        }
    }
    assert(moved_from_resting && "membrane potentials should evolve away from resting_mp under nonzero input");

    engine.shutdown();
    printf("  step_loop: ok\n");
}

void test_reset_state() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/4);

    engine.set_input_neurons({0, 1, 2});
    for (s64 tick = 0; tick < 10; ++tick)
        engine.step_simulation({2.0f, 2.0f, 2.0f}, tick);

    engine.reset_state();

    const f32 *mp = engine.membrane_potentials.get_contents();
    for (s64 i = 0; i < engine.neuron_count; ++i)
        assert(mp[i] == engine.resting_membrane_potential);
    assert(engine.active_neuron_count.get_contents()[0] == 0);

    engine.shutdown();
    printf("  reset_state: ok\n");
}

void test_reservoir_features() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/4);

    engine.set_input_neurons({0, 1, 2});
    for (s64 tick = 0; tick < 5; ++tick)
        engine.step_simulation({2.0f, 2.0f, 2.0f}, tick);

    s64 feature_count = 2 * engine.neuron_count + 1;
    GpuPointer<f32> output = allocate<f32>((usize)feature_count * sizeof(f32));
    f32 *raw = output.get_contents();

    // get_reservoir_features_vector takes its GpuPointer by value but only ever
    // calls .get_contents() on it (engine.cpp) — it neither stores nor frees the
    // handle. GpuPointer is move-only, so hand it a borrowed duplicate of the raw
    // handle and keep `output` as the sole owner responsible for deallocate().
    GpuPointer<f32> borrowed;
#ifdef SPIKECOREC_CUDA
    borrowed.pointer = output.pointer;
#elif defined(SPIKECOREC_METAL)
    borrowed.buffer = output.buffer;
#endif
    engine.get_reservoir_features_vector(5, /*spike_tau=*/10.0f, /*voltage_scale=*/1.0f, std::move(borrowed));

    for (s64 i = 0; i < feature_count; ++i)
        assert(std::isfinite(raw[i]));
    assert(raw[feature_count - 1] == 1.0f && "bias slot must be 1.0");

    deallocate(std::move(output));

    engine.shutdown();
    printf("  reservoir_features: ok\n");
}

void test_merge_input_neurons() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/4);

    engine.set_input_neurons({0});

    // gpu_merge_input_neurons appends override neurons into the *current* active
    // set, which gpu_step then consumes (and replaces with the next-tick active
    // set via swap) within the same step_simulation call — so the override
    // neurons themselves won't necessarily be in the post-swap active set.
    // What's directly observable is that gpu_step actually visited them this
    // tick: every processed active neuron gets last_tick_updated[n] = tick.
    s64 tick = 0;
    vector<s64> override_neurons = {5, 9};
    engine.step_simulation({2.0f}, tick, override_neurons);

    const s64 *last_tick_updated = engine.last_tick_updated.get_contents();
    assert(last_tick_updated[5] == tick && "override neuron 5 must be merged into the active set and processed by gpu_step");
    assert(last_tick_updated[9] == tick && "override neuron 9 must be merged into the active set and processed by gpu_step");

    engine.shutdown();
    printf("  merge_input_neurons: ok\n");
}

// ── recording / .spire codec tests ───────────────────────────────────────────

void test_spire_raw_roundtrip() {
    const string path = "/tmp/spikecorec_test_raw.spire";
    const s64 neuron_count = 6;
    const s64 frame_count = 9;

    vector<vector<f32>> frames(frame_count, vector<f32>(neuron_count));
    for (s64 t = 0; t < frame_count; ++t)
        for (s64 n = 0; n < neuron_count; ++n)
            frames[t][n] = (f32)(t * 100 + n) * 0.5f;

    {
        SpireWriter writer(path, neuron_count);
        for (auto &frame : frames) writer.write_frame(frame.data());
    }

    // SpireReader: frame-at-a-time
    {
        SpireReader reader(path);
        assert(reader.neuron_count() == neuron_count);
        vector<f32> buf(neuron_count);
        s64 t = 0;
        while (reader.read_frame(buf.data())) {
            for (s64 n = 0; n < neuron_count; ++n)
                assert(buf[n] == frames[t][n]);
            ++t;
        }
        assert(t == frame_count);
    }

    // read_spire_recording: whole-file decode, byte-for-byte
    SpireRecording recording = read_spire_recording(path);
    assert(recording.neuron_count == neuron_count);
    assert(recording.frame_count == frame_count);
    for (s64 t = 0; t < frame_count; ++t)
        for (s64 n = 0; n < neuron_count; ++n)
            assert(recording.frames[(usize)(t * neuron_count + n)] == frames[t][n]);

    printf("  spire_raw_roundtrip: ok\n");
}

void test_spire_compression_roundtrip() {
    const usize byte_count = 8000;
    vector<u8> data(byte_count);
    for (usize i = 0; i < byte_count; ++i) data[i] = (u8)((i * 37 + 11) & 0xFF);

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
        assert(readback == data);
    };

#ifdef SPIKECOREC_HAVE_ZLIB
    roundtrip(SpireCompression::Gzip, "/tmp/spikecorec_test_compress.gz");
    printf("  spire_compression_roundtrip (gzip): ok\n");
#endif
#ifdef SPIKECOREC_HAVE_LZMA
    roundtrip(SpireCompression::Xz, "/tmp/spikecorec_test_compress.xz");
    printf("  spire_compression_roundtrip (xz): ok\n");
#endif
#ifdef SPIKECOREC_HAVE_BZ2
    roundtrip(SpireCompression::Bz2, "/tmp/spikecorec_test_compress.bz2");
    printf("  spire_compression_roundtrip (bz2): ok\n");
#endif
}

void test_async_spire_writer() {
    // Backpressure: a sink that takes a fixed amount of time per write should
    // serialize the producer through the bounded queue, not let it race ahead.
    struct SlowSink : SpireSink {
        int writes = 0;
        void write(const u8 *, usize) override {
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
            ++writes;
        }
        void close() override {}
    };
    {
        auto sink = make_unique<SlowSink>();
        SlowSink *raw = sink.get();
        AsyncSpireWriter writer(std::move(sink), /*max_queued_chunks=*/2);

        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 8; ++i) writer.write(vector<u8>{(u8)i});
        writer.close();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();

        assert(raw->writes == 8);
        assert(elapsed >= 8 * 15 - 20 && "backpressure should serialize producer through bounded queue");
    }

    // Error propagation: a sink that throws on its 3rd write should surface
    // that exception from a subsequent write()/close() on the calling thread.
    struct FlakySink : SpireSink {
        int count = 0;
        void write(const u8 *, usize) override {
            if (++count == 3) throw std::runtime_error("flaky sink failure");
        }
        void close() override {}
    };
    {
        AsyncSpireWriter writer(make_unique<FlakySink>(), /*max_queued_chunks=*/1);
        bool threw = false;
        try {
            for (int i = 0; i < 20; ++i) {
                writer.write(vector<u8>{(u8)i});
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        } catch (const std::exception &e) {
            threw = true;
            assert(string(e.what()).find("flaky sink failure") != string::npos);
        }
        assert(threw && "write() must surface the worker thread's captured exception");

        bool close_threw = false;
        try { writer.close(); } catch (const std::exception &) { close_threw = true; }
        assert(close_threw && "close() must (re)throw the captured exception");
    }

    printf("  async_spire_writer (backpressure + error propagation): ok\n");
}

void test_start_static_record() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/4);
    engine.set_input_neurons({0, 1, 2});

    const s64 lifetime = 30;
    const s64 stride = 2;
    vector<vector<f32>> input_spikes(lifetime, vector<f32>{2.0f, 2.0f, 2.0f});

    const string path = "/tmp/spikecorec_test_record.spire";
    engine.start_static_record(input_spikes, lifetime, path, /*record_membrane=*/true, stride,
                               /*compression=*/string("none"), nullopt, /*full_decay=*/true,
                               /*compression_async=*/false);

    s64 expected_frames = (lifetime + stride - 1) / stride;
    s64 expected_size = 4 + expected_frames * engine.neuron_count * (s64)sizeof(f32);

    FILE *f = fopen(path.c_str(), "rb");
    assert(f != nullptr);
    fseek(f, 0, SEEK_END);
    long actual_size = ftell(f);
    fclose(f);
    assert(actual_size == expected_size && "recorded file size must match header + frame_count * neuron_count * sizeof(f32)");

    SpireRecording recording = read_spire_recording(path);
    assert(recording.neuron_count == engine.neuron_count);
    assert(recording.frame_count == expected_frames);
    for (f32 v : recording.frames) assert(std::isfinite(v));

    engine.shutdown();
    printf("  start_static_record: ok\n");
}

void test_spire_truncated_decode_error() {
    const string good_path = "/tmp/spikecorec_test_trunc_source.spire";
    const string trunc_path = "/tmp/spikecorec_test_truncated.spire";
    const s64 neuron_count = 4;

    {
        SpireWriter writer(good_path, neuron_count);
        for (s64 t = 0; t < 5; ++t) {
            vector<f32> frame(neuron_count, (f32)t);
            writer.write_frame(frame.data());
        }
    }

    // Copy header + 4 full frames + a few extra bytes of a 5th (partial) frame.
    {
        ifstream src(good_path, ios::binary);
        vector<char> all((istreambuf_iterator<char>(src)), istreambuf_iterator<char>());
        ofstream dst(trunc_path, ios::binary);
        dst.write(all.data(), 4 + neuron_count * 4 * 4 + 6);
    }

    string log_before = read_file("logs/spikecorec.log");
    bool threw = false;
    try {
        read_spire_recording(trunc_path);
    } catch (const std::exception &e) {
        threw = true;
        assert(string(e.what()).find("Truncated recording") != string::npos);
    }
    assert(threw && "read_spire_recording must throw on a truncated final frame");
    string appended = read_file("logs/spikecorec.log").substr(log_before.size());
    assert(appended.find("Truncated recording") != string::npos && "the critical log line must precede the throw");

    printf("  spire_truncated_decode_error: ok\n");
}

// record_frame must reject a frame whose length != neuron_count rather than
// reading out of bounds past the supplied pointer.
void test_record_frame_size_validation() {
    const string path = "/tmp/spikecorec_test_record_frame_size.spire";
    const s64 neuron_count = 8;
    SimulationRecorder recorder(path, neuron_count, string("none"), nullopt, /*async=*/false);

    vector<f32> correct(neuron_count, 1.0f);
    recorder.record_frame(correct.data(), neuron_count); // exact length — ok

    vector<f32> too_small(3, 1.0f);
    string log_before = read_file("logs/spikecorec.log");
    bool threw = false;
    try {
        recorder.record_frame(too_small.data(), (s64)too_small.size());
    } catch (const std::exception &e) {
        threw = true;
        assert(string(e.what()).find("does not match") != string::npos);
    }
    assert(threw && "record_frame must reject a frame whose length != neuron_count");
    string appended = read_file("logs/spikecorec.log").substr(log_before.size());
    assert(appended.find("does not match") != string::npos && "the critical log line must precede the throw");

    recorder.finish();
    printf("  record_frame_size_validation: ok\n");
}

// start_static_record must validate each input row's width up front: an over-wide
// row would drive an out-of-bounds GPU write, an empty row a mid-loop throw.
void test_start_static_record_bad_input_width() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/4);
    engine.set_input_neurons({0, 1, 2}); // input_neuron_count == 3

    const s64 lifetime = 5;
    const string path = "/tmp/spikecorec_test_bad_width.spire";

    // Row wider than input_neuron_count → rejected (would otherwise OOB the kernel).
    {
        vector<vector<f32>> wide(lifetime, vector<f32>{1.0f, 1.0f, 1.0f});
        wide[2] = vector<f32>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f}; // 5 > 3
        string log_before = read_file("logs/spikecorec.log");
        bool threw = false;
        try {
            engine.start_static_record(wide, lifetime, path, true, 1, string("none"), nullopt, true, false);
        } catch (const std::exception &e) {
            threw = true;
            assert(string(e.what()).find("input neurons") != string::npos);
        }
        assert(threw && "start_static_record must reject an over-wide input row");
        string appended = read_file("logs/spikecorec.log").substr(log_before.size());
        assert(appended.find("input neurons") != string::npos && "the critical log line must precede the throw");
    }

    // Empty row → rejected up front (no truncated file).
    {
        vector<vector<f32>> empty_row(lifetime, vector<f32>{1.0f, 1.0f, 1.0f});
        empty_row[1].clear(); // size 0 != 3
        bool threw = false;
        try {
            engine.start_static_record(empty_row, lifetime, path, true, 1, string("none"), nullopt, true, false);
        } catch (const std::exception &) { threw = true; }
        assert(threw && "start_static_record must reject an empty input row");
    }

    engine.shutdown();
    printf("  start_static_record_bad_input_width: ok\n");
}

// max_queued_chunks == 0 means "unbounded" (matches Python queue.Queue(maxsize=0));
// the producer must never block/deadlock and every chunk must be written.
void test_async_writer_unbounded_queue() {
    struct CountingSink : SpireSink {
        int writes = 0;
        void write(const u8 *, usize) override { ++writes; }
        void close() override {}
    };
    auto sink = make_unique<CountingSink>();
    CountingSink *raw = sink.get();
    AsyncSpireWriter writer(std::move(sink), /*max_queued_chunks=*/0);

    for (int i = 0; i < 16; ++i) writer.write(vector<u8>{(u8)i});
    writer.close();

    assert(raw->writes == 16 && "unbounded queue must accept and drain every chunk without deadlocking");
    printf("  async_writer_unbounded_queue: ok\n");
}

// After the sink throws once, the worker must stop touching it — no further
// write() and no close() on the failed sink (re-entry is UB for some backends).
void test_async_writer_no_reentry_after_error() {
    struct CountingFailSink : SpireSink {
        int writes = 0;
        int closes = 0;
        int fail_on;
        explicit CountingFailSink(int f) : fail_on(f) {}
        void write(const u8 *, usize) override {
            if (++writes == fail_on) throw std::runtime_error("sink boom");
        }
        void close() override { ++closes; }
    };
    auto sink = make_unique<CountingFailSink>(/*fail_on=*/2);
    CountingFailSink *raw = sink.get();
    AsyncSpireWriter writer(std::move(sink), /*max_queued_chunks=*/1);

    bool surfaced = false;
    try {
        for (int i = 0; i < 10; ++i) {
            writer.write(vector<u8>{(u8)i});
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
    } catch (const std::exception &) { surfaced = true; }
    try { writer.close(); } catch (const std::exception &) { surfaced = true; }

    assert(surfaced && "the sink's exception must surface on the producer thread");
    assert(raw->writes == 2 && "worker must not call write() again after the sink failed");
    assert(raw->closes == 0 && "worker must not close a sink that already failed");
    printf("  async_writer_no_reentry_after_error: ok\n");
}

// A recorder that never records a frame still writes a valid header-only file;
// decoding it must yield zero frames (not a crash or a spurious truncation error).
void test_spire_zero_frame_recording() {
    const string path = "/tmp/spikecorec_test_zero_frame.spire";
    const s64 neuron_count = 5;
    {
        SimulationRecorder recorder(path, neuron_count, string("none"), nullopt, /*async=*/false);
        recorder.finish(); // no record_frame calls
    }
    SpireRecording recording = read_spire_recording(path);
    assert(recording.neuron_count == neuron_count);
    assert(recording.frame_count == 0);
    assert(recording.frames.empty());
    printf("  spire_zero_frame_recording: ok\n");
}

// neuron_count must fit the .spire 32-bit header; an out-of-range count must be
// rejected at construction rather than silently truncated into the header.
void test_spire_header_neuron_count_overflow() {
    const s64 too_big = (s64)UINT32_MAX + 1;

    bool recorder_threw = false;
    try {
        SimulationRecorder recorder("/tmp/spikecorec_test_overflow.spire", too_big,
                                    string("none"), nullopt, /*async=*/false);
    } catch (const std::exception &) { recorder_threw = true; }
    assert(recorder_threw && "SimulationRecorder must reject neuron_count > UINT32_MAX");

    bool writer_threw = false;
    try {
        SpireWriter writer("/tmp/spikecorec_test_overflow2.spire", too_big);
    } catch (const std::exception &) { writer_threw = true; }
    assert(writer_threw && "SpireWriter must reject neuron_count > UINT32_MAX");

    printf("  spire_header_neuron_count_overflow: ok\n");
}

// ── shared helpers ───────────────────────────────────────────────────────────

// Mixed absolute/relative float comparison — tolerant of GPU/float rounding.
bool approx(f32 a, f32 b, f32 eps = 1e-3f) {
    return std::fabs(a - b) <= eps * (1.0f + std::fabs(b));
}

// Small fixed 8-node directed graph with two self-loops (nodes 1 and 4).
vector<vector<s32>> k2_reference_adjacency() {
    return {
        {1, 4, 7}, // 0
        {1, 2},    // 1 — self-loop
        {0, 3, 5}, // 2
        {},        // 3 — no out-edges
        {4},       // 4 — self-loop
        {6, 7},    // 5
        {0},       // 6
        {3, 6},    // 7
    };
}

// ── types / backend tests ────────────────────────────────────────────────────

void test_types_layout() {
    assert(sizeof(u8) == 1 && sizeof(u16) == 2 && sizeof(u32) == 4 && sizeof(u64) == 8);
    assert(sizeof(s32) == 4 && sizeof(s64) == 8);
    assert(sizeof(f32) == 4 && sizeof(f64) == 8);
    // Qualify explicitly: on the CUDA backend <cuda_runtime.h> also defines a
    // global ::float4, so an unqualified `float4` is ambiguous here.
    assert(sizeof(spikecorec::float4) == 16 && "float4 must be tightly packed");
    assert(alignof(spikecorec::float4) == 16 && "float4 must be 16-byte aligned for GPU layout");
    printf("  types_layout: ok\n");
}

void test_gpu_pointer_alloc() {
    const usize n = 32;
    GpuPointer<f32> buf = allocate<f32>(n * sizeof(f32));
    f32 *data = buf.get_contents();
    assert(data != nullptr);
    for (usize i = 0; i < n; ++i) data[i] = (f32)i * 1.5f;
    for (usize i = 0; i < n; ++i) assert(data[i] == (f32)i * 1.5f);

    // move-construction transfers ownership and nulls the source handle
    GpuPointer<f32> moved = std::move(buf);
    assert(moved.get_contents() == data);
#ifdef SPIKECOREC_METAL
    assert(buf.buffer == nullptr && "moved-from GpuPointer must be null");
#elif defined(SPIKECOREC_CUDA)
    assert(buf.pointer == nullptr && "moved-from GpuPointer must be null");
#endif
    deallocate(std::move(moved));
    printf("  gpu_pointer_alloc: ok\n");
}

// ── topology edge-case tests ─────────────────────────────────────────────────

void test_square_torus_edge_cases() {
    // k == 1: a single node whose four neighbors all wrap back to itself.
    auto one = square_torus(1);
    assert(one.size() == 1 && one[0].size() == 4);
    for (s32 c : one[0]) assert(c == 0);

    // k == 3: exact 4-neighbor torus wrapping for node 0 → {right=1,left=2,down=3,up=6}.
    auto three = square_torus(3);
    assert(three.size() == 9);
    assert((unordered_set<s32>(three[0].begin(), three[0].end()) == unordered_set<s32>{1, 2, 3, 6}));

    // k < 1 is rejected.
    bool threw = false;
    try { square_torus(0); } catch (const std::invalid_argument &) { threw = true; }
    assert(threw && "square_torus(0) must throw");
    threw = false;
    try { square_torus(-2); } catch (const std::invalid_argument &) { threw = true; }
    assert(threw && "square_torus(-2) must throw");

    printf("  square_torus_edge_cases: ok\n");
}

void test_small_world_determinism() {
    // Same seed → byte-identical structure.
    auto a = small_world_torus(4, /*random_fanout=*/3, /*seed=*/123);
    auto b = small_world_torus(4, /*random_fanout=*/3, /*seed=*/123);
    assert(a == b && "small_world_torus must be deterministic for a fixed seed");

    // Out-degree = 4 (torus) + 3 (shortcuts) = 7, all distinct, no self-loops.
    for (s64 i = 0; i < (s64)a.size(); ++i) {
        assert(a[(usize)i].size() == 7);
        unordered_set<s32> seen(a[(usize)i].begin(), a[(usize)i].end());
        assert((s64)seen.size() == 7 && "no duplicate edges");
        assert(seen.count((s32)i) == 0 && "no self-loop");
    }

    // random_fanout 0 → just the torus; negative clamps to 0.
    for (auto &row : small_world_torus(4, /*random_fanout=*/0, /*seed=*/1)) assert(row.size() == 4);
    for (auto &row : small_world_torus(4, /*random_fanout=*/-5, /*seed=*/1)) assert(row.size() == 4);

    printf("  small_world_determinism: ok\n");
}

void test_random_fixed_outdegree_edge_cases() {
    // fanout 0 → every row empty.
    auto z = random_fixed_outdegree(3, /*fanout=*/0, /*seed=*/7);
    assert(z.size() == 9);
    for (auto &row : z) assert(row.empty());

    // Determinism.
    assert(random_fixed_outdegree(4, 5, 99) == random_fixed_outdegree(4, 5, 99));

    // fanout clamped to n-1; requesting more → exactly n-1 distinct, no self-loops.
    auto full = random_fixed_outdegree(3, /*fanout=*/100, /*seed=*/3); // n=9 → clamp to 8
    for (s64 i = 0; i < (s64)full.size(); ++i) {
        assert(full[(usize)i].size() == 8);
        unordered_set<s32> seen(full[(usize)i].begin(), full[(usize)i].end());
        assert((s64)seen.size() == 8 && seen.count((s32)i) == 0);
    }

    // k < 1 throws.
    bool threw = false;
    try { random_fixed_outdegree(0, 4, 1); } catch (const std::invalid_argument &) { threw = true; }
    assert(threw);

    printf("  random_fixed_outdegree_edge_cases: ok\n");
}

// ── k2tree tests ─────────────────────────────────────────────────────────────

void test_k2tree_from_adjacency_list_invalid_branching_factor() {
    auto list = k2_reference_adjacency();
    const s32 node_count = 8;
    auto result = K2Tree::from_adjacency_list(list, node_count);
    assert(!result.has_value());
}

void test_k2tree_adjacent_and_neighbors() {
    auto adj = k2_reference_adjacency();
    const s32 n = 8;
    K2Tree tree = *K2Tree::from_adjacency_list(adj, n);

    // adjacent(u,v) must agree with the reference adjacency for ALL pairs.
    for (s32 u = 0; u < n; ++u) {
        unordered_set<s32> row(adj[(usize)u].begin(), adj[(usize)u].end());
        for (s32 v = 0; v < n; ++v)
            assert(tree.adjacent(u, v) == (row.count(v) ? 1 : 0));
    }
    assert(tree.adjacent(1, 1) == 1 && tree.adjacent(4, 4) == 1 && "self-loops must resolve");

    // get_neighbors must return exactly each node's neighbor set (ascending order).
    vector<s32> buf(n);
    for (s32 u = 0; u < n; ++u) {
        s64 count = tree.get_neighbors(u, buf.data(), n);
        vector<s32> got(buf.begin(), buf.begin() + count);
        vector<s32> expected = adj[(usize)u];
        std::sort(expected.begin(), expected.end());
        assert(got == expected && "get_neighbors must return the node's sorted neighbor set");
    }

    printf("  k2tree_adjacent_and_neighbors: ok\n");
}

void test_k2tree_adjacent_batch() {
    auto adj = k2_reference_adjacency();
    const s32 n = 8;
    K2Tree tree = *K2Tree::from_adjacency_list(adj, n);

    vector<s32> src, tgt;
    for (s32 u = 0; u < n; ++u)
        for (s32 v = 0; v < n; ++v) { src.push_back(u); tgt.push_back(v); }
    vector<uint8_t> out(src.size());
    tree.adjacent_batch(src.data(), tgt.data(), out.data(), (s32)src.size());

    for (usize q = 0; q < src.size(); ++q)
        assert(out[q] == (uint8_t)tree.adjacent(src[q], tgt[q])
               && "adjacent_batch (GPU) must match adjacent (CPU)");

    printf("  k2tree_adjacent_batch: ok\n");
}

void test_k2tree_single_node_and_bounds() {
    // node_count == 1 → tree_height 0; every query returns 0 (even a self-loop edge).
    vector<vector<s32>> single = {{0}};
    K2Tree one = *K2Tree::from_adjacency_list(single, 1);
    assert(one.tree_height == 0);
    vector<s32> buf(4);
    assert(one.adjacent(0, 0) == 0);
    assert(one.get_neighbors(0, buf.data(), 4) == 0);

    // Out-of-bounds / degenerate queries on a normal tree return 0.
    K2Tree tree = *K2Tree::from_adjacency_list(k2_reference_adjacency(), 8);
    assert(tree.adjacent(-1, 0) == 0 && tree.adjacent(0, 8) == 0 && tree.adjacent(8, 0) == 0);
    assert(tree.get_neighbors(-1, buf.data(), 4) == 0);
    assert(tree.get_neighbors(8, buf.data(), 4) == 0);
    assert(tree.get_neighbors(0, buf.data(), 0) == 0 && "max_neighbor_count 0 → 0 neighbors");

    printf("  k2tree_single_node_and_bounds: ok\n");
}

void test_k2tree_save_load() {
    auto adj = k2_reference_adjacency();
    const s32 n = 8;
    K2Tree tree = *K2Tree::from_adjacency_list(adj, n);

    const char *path = "/tmp/spikecorec_test_k2tree.bin";
    tree.save(path);
    K2Tree loaded = K2Tree::load(path);

    assert(loaded.node_count == tree.node_count && loaded.tree_height == tree.tree_height);
    for (s32 u = 0; u < n; ++u)
        for (s32 v = 0; v < n; ++v)
            assert(loaded.adjacent(u, v) == tree.adjacent(u, v) && "save/load must round-trip adjacency");

    printf("  k2tree_save_load: ok\n");
}

// ── weight_matrix tests ──────────────────────────────────────────────────────

void test_weight_matrix_construction() {
    auto net = square_torus(4);                  // 16 nodes, every row length 4
    WeightMatrix wm(net, /*rank=*/8);
    assert(wm.node_count == 16);
    assert(wm.max_neighbor_count == 4 && "max_neighbor_count = longest row");
    assert(wm.rank == 8 && wm.rank_float4_stride == 2);
    assert(!wm.using_constant_weight);

    // rank = -1 → min(64, node_count)
    auto net2 = square_torus(3);                 // 9 nodes
    WeightMatrix wm2(net2, /*rank=*/-1);
    assert(wm2.rank == 9);

    // empty network rejected
    vector<vector<s32>> empty;
    bool threw = false;
    try { WeightMatrix bad(empty); } catch (const std::invalid_argument &) { threw = true; }
    assert(threw && "WeightMatrix must reject an empty network");

    printf("  weight_matrix_construction: ok\n");
}



void test_weight_matrix_constant_weight() {
    auto net = square_torus(4);
    WeightMatrix wm(net, /*rank=*/8);            // rank % 4 == 0 → get() == constant value

    wm.set_constant_weight(0.5f);
    assert(wm.using_constant_weight && wm.constant_weight == 0.5f);
    assert(approx(wm.get(0, 1), 0.5f) && approx(wm.get(5, 9), 0.5f));

    wm.set_constant_weight(-0.3f);
    assert(approx(wm.get(2, 7), -0.3f));

    wm.set_constant_weight(0.0f);
    assert(approx(wm.get(0, 1), 0.0f));

    printf("  weight_matrix_constant_weight: ok\n");
}

void test_weight_matrix_stats_and_scale() {
    auto net = square_torus(4);                  // every node has exactly 4 neighbors → no padding
    WeightMatrix wm(net, /*rank=*/8);

    wm.set_constant_weight(0.5f);
    WeightStats stats = wm.neighbor_weight_stats();
    assert(approx(stats.mean, 0.5f) && approx(stats.root_mean_square, 0.5f));
    assert(stats.standard_deviation < 1e-2f);
    assert(approx(stats.min_value, 0.5f) && approx(stats.max_value, 0.5f));

    // scale RMS to a target
    ScaleResult result = wm.scale_neighbor_weights_to_root_mean_square(2.0f);
    assert(approx(result.after.root_mean_square, 2.0f, 1e-2f));
    assert(!wm.using_constant_weight && "scaling clears constant-weight mode");

    // target 0 → all weights driven to 0
    wm.scale_neighbor_weights_to_root_mean_square(0.0f);
    assert(approx(wm.neighbor_weight_stats().root_mean_square, 0.0f, 1e-5f));

    // negative target rejected
    bool threw = false;
    try { wm.scale_neighbor_weights_to_root_mean_square(-1.0f); }
    catch (const std::invalid_argument &) { threw = true; }
    assert(threw);

    printf("  weight_matrix_stats_and_scale: ok\n");
}

void test_weight_matrix_get_neighbors() {
    auto net = square_torus(4);
    WeightMatrix wm(net, /*rank=*/8);

    vector<s32> buf((usize)wm.max_neighbor_count);
    for (s64 node = 0; node < wm.node_count; ++node) {
        s64 count = wm.get_neighbors(node, buf.data());
        unordered_set<s32> got(buf.begin(), buf.begin() + count);
        unordered_set<s32> expected(net[(usize)node].begin(), net[(usize)node].end());
        assert(got == expected && "WeightMatrix::get_neighbors must match the source adjacency");
    }
    printf("  weight_matrix_get_neighbors: ok\n");
}

void test_weight_matrix_get_neighbors_out_of_bounds() {
    auto network = square_torus(4);
    WeightMatrix matrix(network, /*rank=*/8);

    vector<s32> buffer((usize)matrix.max_neighbor_count);
    s64 neighbor_count;

    neighbor_count = matrix.get_neighbors(-1, buffer.data());
    assert(neighbor_count == 0);

    neighbor_count = matrix.get_neighbors(16, buffer.data());
    assert(neighbor_count == 0);

    neighbor_count = matrix.get_neighbors(17, buffer.data());
    assert(neighbor_count == 0);

    neighbor_count = matrix.get_neighbors(4590, buffer.data());
    assert(neighbor_count == 0);

    printf("  weight_matrix_get_neighbors_out_of_bounds: ok\n");
}

void test_weight_matrix_get_out_of_bounds() {
    auto network = square_torus(4);
    WeightMatrix matrix(network, 8);

    f32 result_negative_index = matrix.get(-1, 10);
    assert(result_negative_index == 0.0);

    f32 result_high_index = matrix.get(3, 50);
    assert(result_high_index == 0.0);

    printf("  weight_matrix_get_out_of_bounds: ok\n");
}

void test_weight_matrix_update_out_of_bounds() {
    auto network = square_torus(4);
    WeightMatrix matrix(network, /*rank=*/8);

    f32 before = matrix.get(0, 1);
    matrix.update(-1, 1, 1.0f);
    matrix.update(0, 4590, 1.0f);
    f32 after = matrix.get(0, 1);
    assert(approx(before, after) && "out-of-bounds update() calls must be no-ops");

    printf("  weight_matrix_update_out_of_bounds: ok\n");
}

void test_weight_matrix_update() {
    auto net = square_torus(4);
    WeightMatrix wm(net, /*rank=*/8);

    f32 before = wm.get(0, 1);
    wm.update(/*source=*/0, /*target=*/1, /*delta=*/50.0f,
              /*learning_rate=*/0.2f, /*l2_regularization=*/1.0f, /*iterations=*/40);
    f32 after = wm.get(0, 1);

    assert(std::isfinite(after));
    assert(after > before && "a strong positive-delta proximal-Hebbian update must increase U·V");

    printf("  weight_matrix_update: ok\n");
}

void test_weight_matrix_save_load() {
    auto net = square_torus(4);
    WeightMatrix wm(net, /*rank=*/8);
    wm.set_constant_weight(0.75f);

    const char *path = "/tmp/spikecorec_test_wm.bin";
    wm.save(path);

    WeightMatrix wm2(net, /*rank=*/8);
    wm2.load_from_disk(path);
    assert(approx(wm2.get(0, 1), 0.75f) && approx(wm2.get(7, 3), 0.75f)
           && "save/load must round-trip the weights");

    printf("  weight_matrix_save_load: ok\n");
}

// ── engine: numeric / lifecycle tests ────────────────────────────────────────

void test_estimate_bifurcation_weight() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/8);
    for (s32 period : {1, 2, 5}) {
        auto [w_accum, w_instant] = engine.estimate_bifurcation_weight(period);
        f32 decay_factor = std::pow(1.0f - engine.decay_rate, (f32)period);
        f32 expected_accum = (engine.spike_threshold - engine.resting_membrane_potential) * (1.0f - decay_factor);
        f32 expected_instant = engine.spike_threshold - engine.resting_membrane_potential;
        assert(approx(w_accum, expected_accum, 1e-5f));
        assert(approx(w_instant, expected_instant, 1e-5f));
    }
    engine.shutdown();
    printf("  estimate_bifurcation_weight: ok\n");
}

void test_scale_uniform_near_bifurcation() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/8);

    f32 target = 0, w_accum = 0, w_instant = 0;
    engine.scale_uniform_weights_near_bifurcation(&target, &w_accum, &w_instant,
                                                  /*input_period=*/1, /*scale=*/1.2f, /*freeze_learning=*/true);
    auto [expect_accum, expect_instant] = engine.estimate_bifurcation_weight(1);
    assert(approx(w_accum, expect_accum, 1e-5f));
    assert(approx(target, expect_accum * 1.2f, 1e-5f));
    assert(engine.learning_rate == 0.0f && "freeze_learning must zero the learning rate");
    assert(engine.use_constant_weight && "freeze_learning defaults use_constant_weight to true");
    assert(approx(engine.weights.constant_weight, target, 1e-5f));
    assert(approx(engine.weights.get(0, 1), target, 1e-3f));

    engine.shutdown();
    printf("  scale_uniform_near_bifurcation: ok\n");
}

void test_scale_randomized_near_bifurcation() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/8);

    ScaledReservoirResult result =
        engine.scale_randomized_weights_near_bifurcation(/*input_period=*/1, /*scale=*/1.2f, /*freeze_learning=*/true);

    f32 expected_target = std::fabs(result.w_accum * 1.2f);
    assert(approx(result.weight_scale_result.target_root_mean_square, expected_target, 1e-5f));
    assert(approx(result.weight_scale_result.after.root_mean_square, expected_target, 1e-2f));
    assert(!engine.use_constant_weight);
    assert(engine.learning_rate == 0.0f);

    engine.shutdown();
    printf("  scale_randomized_near_bifurcation: ok\n");
}

void test_setup_lifetime() {
    auto net = square_torus(4);

    {   // allocates per-neuron logs when the budget allows
        SpikeEngine engine(&net, {4, 4}, /*rank=*/8);
        engine.setup_lifetime(/*lifetime=*/10, /*allocate_logs=*/true);
        assert(engine.lifetime == 10 && engine.cell_state_logs != nullptr);
        engine.shutdown();
    }
    {   // a too-small byte budget is rejected
        SpikeEngine engine(&net, {4, 4}, /*rank=*/8);
        bool threw = false;
        try { engine.setup_lifetime(/*lifetime=*/1000, /*allocate_logs=*/true, /*max_log_bytes=*/16); }
        catch (const std::runtime_error &) { threw = true; }
        assert(threw && "setup_lifetime must reject an over-budget log allocation");
        engine.shutdown();
    }
    {   // allocate_logs = false → no allocation
        SpikeEngine engine(&net, {4, 4}, /*rank=*/8);
        engine.setup_lifetime(/*lifetime=*/10, /*allocate_logs=*/false);
        assert(engine.cell_state_logs == nullptr);
        engine.shutdown();
    }
    printf("  setup_lifetime: ok\n");
}

void test_engine_input_and_step_guards() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/8);

    engine.set_input_neurons({});               // empty list → no-op
    assert(engine.input_neuron_count == 0);
    engine.set_input_neurons({0, 1, 2});
    assert(engine.input_neuron_count == 3);

    bool threw = false;
    try { engine.step_simulation({}, /*tick=*/0); }
    catch (const std::runtime_error &) { threw = true; }
    assert(threw && "step_simulation must reject empty input_values");

    engine.shutdown();
    printf("  engine_input_and_step_guards: ok\n");
}

void test_reset_state_generations() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/8);
    engine.set_input_neurons({0, 1, 2});
    for (s64 tick = 0; tick < 5; ++tick)
        engine.step_simulation({2.0f, 2.0f, 2.0f}, tick);

    engine.reset_state();
    const s32 *gen = engine.active_generation.get_contents();
    const s64 *last_spiked = engine.last_spiked.get_contents();
    for (s64 i = 0; i < engine.neuron_count; ++i) {
        assert(gen[i] == -1 && "reset_state must mark every neuron inactive (generation -1)");
        assert(last_spiked[i] == 0);
    }

    engine.shutdown();
    printf("  reset_state_generations: ok\n");
}

void test_reservoir_features_guard() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/8);
    engine.set_input_neurons({0, 1, 2});
    engine.step_simulation({2.0f, 2.0f, 2.0f}, 0);

    s64 feature_count = 2 * engine.neuron_count + 1;
    GpuPointer<f32> output = allocate<f32>((usize)feature_count * sizeof(f32));
    f32 *raw = output.get_contents();
    for (s64 i = 0; i < feature_count; ++i) raw[i] = -12345.0f;

    GpuPointer<f32> borrowed;
#ifdef SPIKECOREC_METAL
    borrowed.buffer = output.buffer;
#elif defined(SPIKECOREC_CUDA)
    borrowed.pointer = output.pointer;
#endif
    // spike_tau <= 0 → early return, output left untouched.
    engine.get_reservoir_features_vector(1, /*spike_tau=*/-1.0f, /*voltage_scale=*/1.0f, std::move(borrowed));
    for (s64 i = 0; i < feature_count; ++i)
        assert(raw[i] == -12345.0f && "non-positive spike_tau must leave the output buffer untouched");

    deallocate(std::move(output));
    engine.shutdown();
    printf("  reservoir_features_guard: ok\n");
}

void test_start_static_record_variants() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/8);
    engine.set_input_neurons({0, 1, 2});

    const s64 lifetime = 12;
    vector<vector<f32>> input(lifetime, vector<f32>{1.0f, 1.0f, 1.0f});

    {   // record_membrane = false → header-only file (0 frames)
        const string path = "/tmp/spikecorec_test_norec.spire";
        engine.start_static_record(input, lifetime, path, /*record_membrane=*/false);
        assert(read_spire_recording(path).frame_count == 0);
    }
    engine.reset_state();
    {   // record_stride = 4 over 12 ticks → ceil(12/4) = 3 frames
        const string path = "/tmp/spikecorec_test_stride.spire";
        engine.start_static_record(input, lifetime, path, /*record_membrane=*/true, /*record_stride=*/4);
        SpireRecording rec = read_spire_recording(path);
        assert(rec.frame_count == 3);
        for (f32 v : rec.frames) assert(std::isfinite(v));
    }
    engine.reset_state();
    {   // full_decay = false → still finite frames, one per tick
        const string path = "/tmp/spikecorec_test_nodecay.spire";
        engine.start_static_record(input, lifetime, path, true, 1, string("none"), nullopt, /*full_decay=*/false);
        SpireRecording rec = read_spire_recording(path);
        assert(rec.frame_count == lifetime && rec.neuron_count == engine.neuron_count);
        for (f32 v : rec.frames) assert(std::isfinite(v));
    }

    engine.shutdown();
    printf("  start_static_record_variants: ok\n");
}

void test_k2tree_from_edges() {
    // The same graph as a flat edge list must build an identical tree.
    auto adj = k2_reference_adjacency();
    const s32 n = 8;
    vector<s32> src, tgt;
    for (s32 u = 0; u < n; ++u)
        for (s32 v : adj[(usize)u]) { src.push_back(u); tgt.push_back(v); }

    K2Tree from_edge_list = *K2Tree::from_edges(src.data(), tgt.data(), (s32)src.size(), n);
    K2Tree from_adj = *K2Tree::from_adjacency_list(adj, n);
    for (s32 u = 0; u < n; ++u)
        for (s32 v = 0; v < n; ++v)
            assert(from_edge_list.adjacent(u, v) == from_adj.adjacent(u, v)
                   && "from_edges must build the same tree as from_adjacency_list");

    printf("  k2tree_from_edges: ok\n");
}

void test_k2tree_from_edges_invalid_branching_factor() {
    // The same graph as a flat edge list must build an identical tree.
    auto list = k2_reference_adjacency();
    const s32 node_count = 8;
    vector<s32> source, target;
    for (usize u = 0; u < node_count; ++u) {
        for (s32 v : list[u]) { 
            source.push_back(u); 
            target.push_back(v); 
        }
    }


    auto result = K2Tree::from_edges(source.data(), target.data(), (s32)source.size(), node_count, 10);
    assert(!result.has_value());

    result = K2Tree::from_edges(source.data(), target.data(), (s32)source.size(), node_count, -1);
    assert(!result.has_value());
}

void test_step_simulation_decay_path() {
    auto net = square_torus(4);
    SpikeEngine engine(&net, {4, 4}, /*rank=*/8);
    engine.set_input_neurons({0, 1, 2});

    // decay_all_neurons = true exercises gpu_decay_all_neurons inside step_simulation.
    const s64 last_tick = 5;
    for (s64 tick = 0; tick <= last_tick; ++tick)
        engine.step_simulation({2.0f, 2.0f, 2.0f}, tick, /*override_input_neurons=*/{}, /*decay_all_neurons=*/true);

    const f32 *mp = engine.membrane_potentials.get_contents();
    const s64 *last_updated = engine.last_tick_updated.get_contents();
    for (s64 i = 0; i < engine.neuron_count; ++i) {
        assert(std::isfinite(mp[i]));
        // a full-decay pass advances last_tick_updated for EVERY neuron, not just active ones
        assert(last_updated[i] == last_tick && "decay_all_neurons must touch every neuron each tick");
    }

    engine.shutdown();
    printf("  step_simulation_decay_path: ok\n");
}

// ── engine: deterministic spike-propagation / end-to-end tests ────────────────
//
// These pin down the *core dynamics* — threshold crossing, weight delivery,
// refractory gating, fan-out, and multi-hop wave propagation — on hand-built
// graphs where every spike time is known in advance. They run with plasticity
// frozen (learning_rate = 0) and a constant synaptic weight chosen so that a
// single upstream spike drives its downstream neighbor exactly one threshold
// crossing per tick, giving a fully deterministic spike schedule.
//
// last_spiked is initialised to 0 by the constructor, which doubles as the
// "never spiked" sentinel; with spike_period = 1 that makes (tick - last_spiked)
// == spike_period at tick 1, spuriously tripping the refractory branch for a
// never-fired neuron. The reference usage seeds last_spiked to a large negative
// value so a neuron's first spike is governed purely by its membrane potential;
// these tests do the same via seed_never_spiked() below.
void seed_never_spiked(SpikeEngine &engine, s64 sentinel = -1000) {
    s64 *last_spiked = engine.last_spiked.get_contents();
    for (s64 i = 0; i < engine.neuron_count; ++i) last_spiked[i] = sentinel;
}

// Configure an engine for a deterministic, plasticity-free propagation test:
// constant synaptic weight `w` (> spike_threshold - resting so one presynaptic
// spike drives the postsynaptic neuron over threshold in a single tick).
void configure_deterministic(SpikeEngine &engine, f32 w = 2.0f) {
    engine.spike_threshold = 1.0f;
    engine.spike_period = 1;
    engine.learning_rate = 0.0f;               // freeze plasticity
    engine.use_constant_weight = true;
    engine.weights.set_constant_weight(w);
    seed_never_spiked(engine);
}

// A single spike must (a) mark the firing neuron's spike time, (b) deliver the
// synaptic weight into each downstream neighbor's input accumulator, and
// (c) schedule those neighbors into the next-tick active set.
void test_spike_single_hop() {
    vector<vector<s32>> net = {{1}, {}};       // 0 -> 1
    SpikeEngine engine(&net, {2, 1}, /*rank=*/4);
    configure_deterministic(engine, /*w=*/2.0f);
    engine.set_input_neurons({0});

    // Drive neuron 0 over threshold at tick 0 (0.1 resting + 2.0 input = 2.1 > 1.0).
    engine.step_simulation({2.0f}, /*tick=*/0, /*override_input_neurons=*/{0});

    const s64 *last_spiked = engine.last_spiked.get_contents();
    assert(last_spiked[0] == 0 && "neuron 0 must record a spike at tick 0");

    // The weight must have landed in neuron 1's input accumulator (consumed next tick).
    const f32 *network_inputs = engine.network_inputs.get_contents();
    assert(approx(network_inputs[1], 2.0f) && "spike must deliver constant_weight to the downstream neighbor");

    // Neuron 1 must be scheduled for the next tick (post-swap active set).
    s32 active_count = engine.active_neuron_count.get_contents()[0];
    const s32 *active = engine.active_neuron_indices.get_contents();
    bool neighbor_scheduled = false;
    for (s32 i = 0; i < active_count; ++i) if (active[i] == 1) neighbor_scheduled = true;
    assert(neighbor_scheduled && "downstream neighbor must be enqueued into the next active set");

    engine.shutdown();
    printf("  spike_single_hop: ok\n");
}

// Sub-threshold input must NOT spike: no neuron records a spike, nothing propagates.
void test_spike_subthreshold_no_fire() {
    vector<vector<s32>> net = {{1}, {}};
    SpikeEngine engine(&net, {2, 1}, /*rank=*/4);
    configure_deterministic(engine, /*w=*/2.0f);
    engine.set_input_neurons({0});

    // 0.1 resting + 0.5 input = 0.6 < 1.0 threshold → no spike.
    engine.step_simulation({0.5f}, /*tick=*/0, /*override_input_neurons=*/{0});

    const s64 *last_spiked = engine.last_spiked.get_contents();
    for (s64 i = 0; i < engine.neuron_count; ++i)
        assert(last_spiked[i] == -1000 && "sub-threshold stimulation must not produce any spike");

    const f32 *network_inputs = engine.network_inputs.get_contents();
    for (s64 i = 0; i < engine.neuron_count; ++i)
        assert(network_inputs[i] == 0.0f && "no spike → no synaptic weight delivered");

    engine.shutdown();
    printf("  spike_subthreshold_no_fire: ok\n");
}

// A neuron that spikes at tick t must be refractory at t + spike_period: reset to
// resting, no re-spike, no further propagation on that tick.
void test_spike_refractory_period() {
    vector<vector<s32>> net = {{1}, {}};
    SpikeEngine engine(&net, {2, 1}, /*rank=*/4);
    configure_deterministic(engine, /*w=*/2.0f);   // spike_period == 1
    engine.set_input_neurons({0});

    // tick 0: neuron 0 spikes (last_spiked[0] = 0) and re-enqueues itself.
    engine.step_simulation({2.0f}, /*tick=*/0, /*override_input_neurons=*/{0});
    assert(engine.last_spiked.get_contents()[0] == 0);

    // tick 1: neuron 0 is one tick past its spike (1 - 0 == spike_period) → refractory:
    // it must reset to resting and NOT re-spike (last_spiked stays 0).
    engine.step_simulation({0.0f}, /*tick=*/1);
    const s64 *last_spiked = engine.last_spiked.get_contents();
    const f32 *mp = engine.membrane_potentials.get_contents();
    assert(last_spiked[0] == 0 && "refractory neuron must not re-spike at tick == last_spike + spike_period");
    assert(approx(mp[0], engine.resting_membrane_potential) && "refractory neuron must reset to resting potential");

    engine.shutdown();
    printf("  spike_refractory_period: ok\n");
}

// Fan-out: one spike into a node with several out-edges must drive ALL of them —
// every neighbor receives the weight and fires on the next tick.
void test_spike_fanout() {
    vector<vector<s32>> net = {{1, 2, 3}, {}, {}, {}};   // 0 -> {1,2,3}
    SpikeEngine engine(&net, {4, 1}, /*rank=*/4);
    configure_deterministic(engine, /*w=*/2.0f);
    engine.set_input_neurons({0});

    // tick 0: neuron 0 fires; all three neighbors must receive the weight.
    engine.step_simulation({2.0f}, /*tick=*/0, /*override_input_neurons=*/{0});
    const f32 *network_inputs = engine.network_inputs.get_contents();
    for (s32 n : {1, 2, 3})
        assert(approx(network_inputs[n], 2.0f) && "every fan-out neighbor must receive the synaptic weight");

    // tick 1: all three neighbors cross threshold and spike simultaneously.
    engine.step_simulation({0.0f}, /*tick=*/1);
    const s64 *last_spiked = engine.last_spiked.get_contents();
    for (s32 n : {1, 2, 3})
        assert(last_spiked[n] == 1 && "every fan-out neighbor must spike on the tick after the source");
    assert(last_spiked[0] == 0 && "source neuron keeps its original spike time");

    engine.shutdown();
    printf("  spike_fanout: ok\n");
}

// End-to-end: a directed chain 0->1->2->3->4 must carry a single stimulus as a
// travelling wave, exactly one hop per tick — neuron k fires precisely at tick k.
void test_spike_propagation_chain_e2e() {
    const s32 chain_length = 5;
    vector<vector<s32>> net(chain_length);
    for (s32 i = 0; i + 1 < chain_length; ++i) net[i] = {i + 1};   // i -> i+1
    net[chain_length - 1] = {};                                     // tail has no out-edge

    SpikeEngine engine(&net, {chain_length, 1}, /*rank=*/4);
    configure_deterministic(engine, /*w=*/2.0f);
    engine.set_input_neurons({0});

    // Stimulate the head once at tick 0; every later hop is driven purely by
    // upstream propagation.
    for (s64 tick = 0; tick < chain_length; ++tick) {
        vector<f32> input = (tick == 0) ? vector<f32>{2.0f} : vector<f32>{0.0f};
        vector<s64> override_neurons = (tick == 0) ? vector<s64>{0} : vector<s64>{};
        engine.step_simulation(input, tick, override_neurons);

        const f32 *mp = engine.membrane_potentials.get_contents();
        for (s64 i = 0; i < engine.neuron_count; ++i) assert(std::isfinite(mp[i]));
    }

    // The wavefront: neuron k first (and, within this window, only) spikes at tick k.
    const s64 *last_spiked = engine.last_spiked.get_contents();
    for (s32 k = 0; k < chain_length; ++k)
        assert(last_spiked[k] == k && "spike wave must advance exactly one neuron per tick along the chain");

    engine.shutdown();
    printf("  spike_propagation_chain_e2e: ok\n");
}

// End-to-end plasticity: with a nonzero learning_rate, a spike that reaches a
// recently-fired postsynaptic neuron must run the in-step STDP update and change
// U*V for that edge; with learning frozen the same scenario leaves it untouched.
void test_engine_plasticity_end_to_end() {
    auto run_edge_update = [](f32 learning_rate) -> pair<f32, f32> {
        vector<vector<s32>> net = {{1}, {}};       // 0 -> 1
        SpikeEngine engine(&net, {2, 1}, /*rank=*/8);
        engine.spike_threshold = 1.0f;
        engine.spike_period = 1;
        engine.learning_rate = learning_rate;
        engine.use_constant_weight = false;        // weight = U[0]·V[1] (mutable rows)
        engine.set_input_neurons({0});

        // Seed: neuron 0 free to spike; neuron 1 "spiked recently" (tick 1) so the
        // STDP branch (child_last_spiked not in {0, current tick}) fires on delivery.
        s64 *last_spiked = engine.last_spiked.get_contents();
        last_spiked[0] = -1000;
        last_spiked[1] = 1;

        f32 before = engine.weights.get(0, 1);
        // Strong stimulus at tick 3 → neuron 0 fires and propagates to neuron 1.
        engine.step_simulation({5.0f}, /*tick=*/3, /*override_input_neurons=*/{0});
        f32 after = engine.weights.get(0, 1);

        engine.shutdown();
        return {before, after};
    };

    auto [frozen_before, frozen_after] = run_edge_update(/*learning_rate=*/0.0f);
    assert(approx(frozen_before, frozen_after, 1e-6f) && "frozen learning must leave U*V unchanged");

    auto [live_before, live_after] = run_edge_update(/*learning_rate=*/0.5f);
    assert(std::fabs(live_after - live_before) > 1e-6f
           && "a nonzero learning_rate must apply the in-step STDP update to the active edge");

    printf("  engine_plasticity_end_to_end: ok\n");
}

// Cross-check the GPU neighbor_weights dot-product path against the CPU get():
// for random (non-constant) weights, each populated slot of neighbor_weights must
// equal U[source]·V[neighbor], and padding slots beyond a node's degree must be 0.
void test_weight_matrix_neighbor_weights_values() {
    auto net = square_torus(4);                    // 16 nodes, out-degree 4, no padding
    WeightMatrix wm(net, /*rank=*/8);              // random-initialised U/V

    vector<f32> weights((usize)(wm.node_count * wm.max_neighbor_count));
    wm.neighbor_weights(weights.data());

    vector<s32> neighbor_buf((usize)wm.max_neighbor_count);
    for (s64 node = 0; node < wm.node_count; ++node) {
        s64 degree = wm.get_neighbors(node, neighbor_buf.data());
        for (s64 slot = 0; slot < wm.max_neighbor_count; ++slot) {
            f32 got = weights[(usize)(node * wm.max_neighbor_count + slot)];
            if (slot < degree) {
                f32 expected = wm.get((s32)node, neighbor_buf[(usize)slot]);
                assert(approx(got, expected, 1e-3f)
                       && "GPU neighbor_weights must equal CPU U*V for each real neighbor");
            } else {
                assert(got == 0.0f && "padding slots beyond a node's degree must be sentinel-zero");
            }
        }
    }
    printf("  weight_matrix_neighbor_weights_values: ok\n");
}

} // namespace

int main() {
    static_assert(sizeof(spikecorec::f32) == 4);
    static_assert(sizeof(spikecorec::f64) == 8);

    initialize_gpu_context();

    printf("types / backend tests:\n");
    test_types_layout();
    test_gpu_pointer_alloc();

    printf("topology tests:\n");
    test_square_torus();
    test_small_world_torus();
    test_random_fixed_outdegree();
    test_square_torus_edge_cases();
    test_small_world_determinism();
    test_random_fixed_outdegree_edge_cases();

    printf("k2tree tests:\n");
    test_k2tree_adjacent_and_neighbors();
    test_k2tree_adjacent_batch();
    test_k2tree_from_edges();
    test_k2tree_single_node_and_bounds();
    test_k2tree_save_load();

    printf("weight_matrix tests:\n");
    test_weight_matrix_construction();
    test_weight_matrix_constant_weight();
    test_weight_matrix_stats_and_scale();
    test_weight_matrix_get_neighbors();
    test_weight_matrix_get_neighbors_out_of_bounds();
    test_weight_matrix_get_out_of_bounds();
    test_weight_matrix_update_out_of_bounds();
    test_weight_matrix_update();
    test_weight_matrix_save_load();

    printf("engine tests:\n");
    test_construction();
    test_step_loop();
    test_reset_state();
    test_reservoir_features();
    test_merge_input_neurons();
    test_estimate_bifurcation_weight();
    test_scale_uniform_near_bifurcation();
    test_scale_randomized_near_bifurcation();
    test_setup_lifetime();
    test_engine_input_and_step_guards();
    test_reset_state_generations();
    test_reservoir_features_guard();
    test_step_simulation_decay_path();
    test_start_static_record_variants();
    test_spike_single_hop();
    test_spike_subthreshold_no_fire();
    test_spike_refractory_period();
    test_spike_fanout();
    test_spike_propagation_chain_e2e();
    test_engine_plasticity_end_to_end();
    test_weight_matrix_neighbor_weights_values();

    printf("recording tests:\n");
    test_spire_raw_roundtrip();
    test_spire_compression_roundtrip();
    test_async_spire_writer();
    test_async_writer_unbounded_queue();
    test_async_writer_no_reentry_after_error();
    test_start_static_record();
    test_start_static_record_bad_input_width();
    test_record_frame_size_validation();
    test_spire_zero_frame_recording();
    test_spire_header_neuron_count_overflow();
    test_spire_truncated_decode_error();

    printf("core tests passed\n");
    return 0;
}
