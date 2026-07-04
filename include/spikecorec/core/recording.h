//
// Created by Alek Simpson on 6/7/26.
//
#pragma once

#include <condition_variable>
#include <deque>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "spikecorec/core/types.h"

using namespace std;

namespace spikecorec {

    enum class SpireCompression { None, Gzip, Xz, Bz2 };

    // Resolves the compression to use for `filename`. `requested` mirrors
    // _resolve_record_compression's `compression` argument:
    //   - "auto" (default) or nullopt → infer from filename extension
    //     (.gz/.gzip → Gzip, .xz/.lzma → Xz, .bz2 → Bz2, anything else → None)
    //   - "none"/"gzip"/"xz"/"bz2" → explicit request, validated
    // Returns the resolved compression and `filename` unchanged — the Python
    // reference appends an extension when one is missing for an explicit
    // request, but spikecorec callers always pass a complete path, so the
    // returned path is always exactly `filename`.
    pair<SpireCompression, string> resolve_spire_compression(
        const string &filename, optional<string> requested = string("auto"));

    // ── byte-stream sink/source seam ─────────────────────────────────────────
    //
    // Lets SpireWriter/SpireReader (Phase 4+) wrap any of {raw fstream, gzip,
    // xz, bz2} behind one interface — mirrors _open_record_file's /
    // _open_spire_file's "return a file-like object" design. `compression_level`
    // mirrors each library's native level/preset (1-9 for gzip/bz2, 0-9 "preset"
    // for xz); default 6 everywhere, matching _open_record_file's defaults.

    class SpireSink {
    public:
        virtual ~SpireSink() = default;
        virtual void write(const u8 *data, usize byte_count) = 0;
        virtual void close() = 0;
    };

    class SpireSource {
    public:
        virtual ~SpireSource() = default;
        // Reads up to `byte_count` bytes into `out`; returns the number of
        // bytes actually read (0 = clean EOF).
        virtual usize read(u8 *out, usize byte_count) = 0;
    };

    // Builds the sink/source for `compression` writing/reading `filename`.
    // Throws std::runtime_error if the corresponding SPIKECOREC_HAVE_* macro
    // wasn't defined at build time (library unavailable).
    unique_ptr<SpireSink> make_spire_sink(const string &filename, SpireCompression compression, int compression_level = 6);
    unique_ptr<SpireSource> make_spire_source(const string &filename, SpireCompression compression);

    class RawSink : public SpireSink {
    public:
        explicit RawSink(const string &filename);
        void write(const u8 *data, usize byte_count) override;
        void close() override;
    private:
        ofstream file_;
    };

    class RawSource : public SpireSource {
    public:
        explicit RawSource(const string &filename);
        usize read(u8 *out, usize byte_count) override;
    private:
        ifstream file_;
    };

#ifdef SPIKECOREC_HAVE_ZLIB
    class GzipSink : public SpireSink {
    public:
        GzipSink(const string &filename, int compression_level);
        ~GzipSink() override;
        void write(const u8 *data, usize byte_count) override;
        void close() override;
    private:
        void *gz_file_ = nullptr; // gzFile
        bool closed_ = false;
    };

    class GzipSource : public SpireSource {
    public:
        explicit GzipSource(const string &filename);
        ~GzipSource() override;
        usize read(u8 *out, usize byte_count) override;
    private:
        void *gz_file_ = nullptr; // gzFile
    };
#endif

#ifdef SPIKECOREC_HAVE_LZMA
    class XzSink : public SpireSink {
    public:
        XzSink(const string &filename, int compression_level);
        ~XzSink() override;
        void write(const u8 *data, usize byte_count) override;
        void close() override;
    private:
        void flush_stream(bool finish);
        ofstream file_;
        unique_ptr<struct lzma_stream_box> stream_; // pimpl — keeps <lzma.h> out of this header
        bool closed_ = false;
    };

    class XzSource : public SpireSource {
    public:
        explicit XzSource(const string &filename);
        ~XzSource() override;
        usize read(u8 *out, usize byte_count) override;
    private:
        ifstream file_;
        unique_ptr<struct lzma_stream_box> stream_;
        vector<u8> input_buffer_;
        bool input_exhausted_ = false;
        bool stream_ended_ = false;
    };
#endif

#ifdef SPIKECOREC_HAVE_BZ2
    class Bz2Sink : public SpireSink {
    public:
        Bz2Sink(const string &filename, int compression_level);
        ~Bz2Sink() override;
        void write(const u8 *data, usize byte_count) override;
        void close() override;
    private:
        FILE *file_ = nullptr;
        void *bz_file_ = nullptr; // BZFILE*
        bool closed_ = false;
    };

    class Bz2Source : public SpireSource {
    public:
        explicit Bz2Source(const string &filename);
        ~Bz2Source() override;
        usize read(u8 *out, usize byte_count) override;
    private:
        FILE *file_ = nullptr;
        void *bz_file_ = nullptr; // BZFILE*
        bool stream_ended_ = false;
    };
#endif

    // ── async background writer ──────────────────────────────────────────────
    //
    // Direct port of the Python reference's `_AsyncWriter` (spike_engine_cuda.py
    // :26-65): a bounded blocking queue feeds a background thread that drains
    // chunks into `sink`. `write()` blocks the calling (producer) thread when
    // the queue is full — backpressure, mirroring `queue.Queue.put(block=True)`.
    // Exceptions raised on the worker thread are captured and re-raised from
    // the next `write()`/`close()` call on the producer thread (mirrors the
    // reference's `self._error` propagation). A `nullopt` chunk is the
    // sentinel that tells the worker to flush, close the sink, and exit.
    class AsyncSpireWriter {
    public:
        AsyncSpireWriter() = delete;
        AsyncSpireWriter(const AsyncSpireWriter &) = delete;
        AsyncSpireWriter &operator=(const AsyncSpireWriter &) = delete;

        explicit AsyncSpireWriter(unique_ptr<SpireSink> sink, usize max_queued_chunks = 8);
        ~AsyncSpireWriter();

        // Blocks until there is room in the queue (backpressure), then enqueues
        // `chunk` for the worker thread to write. Re-raises any exception
        // already captured from the worker before/instead of enqueueing.
        void write(vector<u8> chunk);

        // Pushes the sentinel, joins the worker thread, and re-raises any
        // captured exception. Safe to call at most once (throws otherwise);
        // the destructor calls this automatically if not already closed.
        void close();

    private:
        void run();
        void rethrow_if_failed();

        unique_ptr<SpireSink> sink_;
        usize max_queued_chunks_;

        deque<optional<vector<u8>>> queue_; // nullopt = sentinel (EOF)
        mutex mutex_;
        condition_variable not_full_;
        condition_variable not_empty_;
        exception_ptr error_;
        bool closed_ = false;

        thread worker_;
    };

    // Sequential `.spire` encoder: writes the 4-byte big-endian neuron_count
    // header on construction, then raw float32 frames (native byte order) via
    // write_frame(). One instance per output file. Not compression-aware —
    // always writes a raw byte stream (see SpireSink/Phase 2 for compression).
    class SpireWriter {
    public:
        SpireWriter() = delete;
        SpireWriter(const SpireWriter &) = delete;
        SpireWriter &operator=(const SpireWriter &) = delete;

        SpireWriter(const string &filename, s64 neuron_count);

        void write_frame(const f32 *membrane_potentials);

        [[nodiscard]] s64 neuron_count() const { return neuron_count_; }

    private:
        ofstream file_;
        s64 neuron_count_;
    };

    // Sequential `.spire` decoder: reads the 4-byte big-endian neuron_count
    // header on construction, then yields frames via read_frame(). Returns
    // false at a clean end-of-file (no bytes remain before the next frame);
    // throws std::runtime_error if a frame is truncated mid-way (mirrors
    // play_spire_recording's "Truncated recording" check).
    class SpireReader {
    public:
        SpireReader() = delete;
        SpireReader(const SpireReader &) = delete;
        SpireReader &operator=(const SpireReader &) = delete;

        explicit SpireReader(const string &filename);

        [[nodiscard]] s64 neuron_count() const { return neuron_count_; }

        bool read_frame(f32 *out_buffer);

    private:
        ifstream file_;
        s64 neuron_count_;
    };

    // ── buffering / chunking layer ───────────────────────────────────────────
    //
    // Ties the raw codec (Phase 1), compression backends (Phase 2), and async
    // writer (Phase 3) together — equivalent to the inline buffering logic at
    // spike_engine_cuda.py:328-363. Resolves `compression`/`filename` (mirrors
    // _resolve_record_compression), writes the .spire header, accumulates
    // frame bytes into an internal buffer, and flushes to the (possibly async)
    // sink once `chunk_bytes` is reached. `record_frame` may be called any
    // number of times; `finish` flushes/closes exactly once (also invoked by
    // the destructor if the caller didn't, so partially-written files still
    // get closed on exception — mirrors the reference's try/finally cleanup).
    class SimulationRecorder {
    public:
        SimulationRecorder() = delete;
        SimulationRecorder(const SimulationRecorder &) = delete;
        SimulationRecorder &operator=(const SimulationRecorder &) = delete;

        SimulationRecorder(
            const string &filename, s64 neuron_count,
            optional<string> compression = string("auto"), optional<int> compression_level = nullopt,
            bool async = false, usize queue_max = 8, usize chunk_bytes = 4 * 1024 * 1024);

        ~SimulationRecorder();

        // Appends one frame's raw bytes (frame_length * sizeof(f32)) to the
        // internal buffer, flushing to the sink once chunk_bytes is reached.
        // `frame_length` must equal neuron_count() — it is validated so a caller
        // handing over a user-supplied array (e.g. the Python binding) cannot
        // trigger an out-of-bounds read of `membrane_potentials`.
        void record_frame(const f32 *membrane_potentials, s64 frame_length);

        [[nodiscard]] s64 neuron_count() const { return neuron_count_; }

        // Flushes any buffered bytes and closes the underlying sink/writer.
        // Safe to call multiple times — only the first call has an effect.
        void finish();

    private:
        void write_bytes(const u8 *data, usize byte_count);
        void flush_buffer();

        s64 neuron_count_;
        usize chunk_bytes_;
        vector<u8> buffer_;
        unique_ptr<SpireSink> sink_;                 // used directly when not async
        unique_ptr<AsyncSpireWriter> async_writer_;  // used when async
        bool finished_ = false;
    };

    // ── decoding / frame-loading ─────────────────────────────────────────────
    //
    // Ports the file-loading half of `play_spire_recording` (frame decoding
    // only — visualization stays out of scope). Auto-detects compression from
    // the filename extension (mirrors _open_spire_file), parses the header,
    // and reads every frame sequentially into one contiguous row-major buffer.
    // Throws std::runtime_error on a truncated final frame (mirrors the
    // reference's "Truncated recording" check). Backed by SpireSource/
    // SpireReader-equivalent decoding — callers wanting frame-at-a-time
    // streaming access can drive SpireReader directly instead.
    struct SpireRecording {
        s64 neuron_count;
        s64 frame_count;
        vector<f32> frames; // row-major [frame_count][neuron_count]
    };

    SpireRecording read_spire_recording(const string &filename);

} // namespace spikecorec
