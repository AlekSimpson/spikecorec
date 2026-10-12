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

#include "types.h"

using namespace std;

namespace spikecorec {
    enum class Output_File_Format {
        SPIRE, 
        SPIREGZIP,
        SPIREBZ2,
        SPIREXZ,
        SPIKE_EVENTS,
        NML_STANDARD // can be any file format
                     // but the data is always 
                     // written as a matrix of
                     // columns where the first
                     // column is time and the 
                     // rest are the values 
                     // specified to record
    };
    
    union Recorded_Value {
        // TODO: this is not final and is just filler code really until we 
        // use it and think of something better
        f64 float64;
        s64 int64;
        f32 float32;
        s32 int32;
    };
    
    // One recorded quantity, as named by a LEMS <OutputColumn quantity="pop1[0]/v"/> or
    // <EventSelection select="pop1[0]" eventPort="spike"/>. recordings_count counts these;
    // this is what each one actually selects.
    struct Recording_Selection {
        String quantity_path;   // verbatim LEMS path, e.g. "pop1[0]/v"
        String variable_name;   // trailing variable, e.g. "v"; "" for an event selection
        String event_port;      // EventSelection's port, e.g. "spike"; "" for a column
        s64 neuron_index = -1;
    };

    struct Recording_Config {
        Vector<String> output_filenames;
        Vector<Output_File_Format> file_output_format;
        Vector<Recorded_Value> recored_data;

        // What recordings_count counts; parallel to nothing, one entry per selected
        // quantity across this config's output files.
        Vector<Recording_Selection> selections;

        static constexpr s64 DEFAULT_MAX_LOG_BYTES = 512 * 1024 * 1014;
        s32 recordings_count;

        void add_output(String &filename);
    };



    enum class Spire_Compression { None, Gzip, Xz, Bz2 };

    // Resolves the compression to use for `filename`. `requested` mirrors
    // _resolve_record_compression's `compression` argument:
    //   - "auto" (default) or nullopt → infer from filename extension
    //     (.gz/.gzip → Gzip, .xz/.lzma → Xz, .bz2 → Bz2, anything else → None)
    //   - "none"/"gzip"/"xz"/"bz2" → explicit request, validated
    // Returns the resolved compression and `filename` unchanged — the Python
    // reference appends an extension when one is missing for an explicit
    // request, but spikecorec callers always pass a complete path, so the
    // returned path is always exactly `filename`.
    Pair<Spire_Compression, String> resolve_spire_compression(
        const String &filename, Optional<String> requested = String("auto"));

    // ── byte-stream sink/source seam ─────────────────────────────────────────
    //
    // Lets Spire_Writer/Spire_Reader (Phase 4+) wrap any of {raw fstream, gzip,
    // xz, bz2} behind one interface — mirrors _open_record_file's /
    // _open_spire_file's "return a file-like object" design. `compression_level`
    // mirrors each library's native level/preset (1-9 for gzip/bz2, 0-9 "preset"
    // for xz); default 6 everywhere, matching _open_record_file's defaults.

    class Spire_Sink {
    public:
        virtual ~Spire_Sink() = default;
        virtual void write(const u8 *data, usize byte_count) = 0;
        virtual void close() = 0;
    };

    class Spire_Source {
    public:
        virtual ~Spire_Source() = default;
        // Reads up to `byte_count` bytes into `out`; returns the number of
        // bytes actually read (0 = clean EOF).
        virtual usize read(u8 *out, usize byte_count) = 0;
    };

    // Builds the sink/source for `compression` writing/reading `filename`.
    // Throws std::runtime_error if the corresponding SPIKECOREC_HAVE_* macro
    // wasn't defined at build time (library unavailable).
    Unique_Pointer<Spire_Sink> make_spire_sink(const String &filename, Spire_Compression compression, int compression_level = 6);
    Unique_Pointer<Spire_Source> make_spire_source(const String &filename, Spire_Compression compression);

    class Raw_Sink : public Spire_Sink {
    public:
        explicit Raw_Sink(const String &filename);
        void write(const u8 *data, usize byte_count) override;
        void close() override;
    private:
        Output_File_Stream file_;
    };

    class Raw_Source : public Spire_Source {
    public:
        explicit Raw_Source(const String &filename);
        usize read(u8 *out, usize byte_count) override;
    private:
        Input_File_Stream file_;
    };

#ifdef SPIKECOREC_HAVE_ZLIB
    class Gzip_Sink : public Spire_Sink {
    public:
        Gzip_Sink(const String &filename, int compression_level);
        ~Gzip_Sink() override;
        void write(const u8 *data, usize byte_count) override;
        void close() override;
    private:
        void *gz_file_ = nullptr; // gzFile
        bool closed_ = false;
    };

    class Gzip_Source : public Spire_Source {
    public:
        explicit Gzip_Source(const String &filename);
        ~Gzip_Source() override;
        usize read(u8 *out, usize byte_count) override;
    private:
        void *gz_file_ = nullptr; // gzFile
    };
#endif

#ifdef SPIKECOREC_HAVE_LZMA
    class Xz_Sink : public Spire_Sink {
    public:
        Xz_Sink(const String &filename, int compression_level);
        ~Xz_Sink() override;
        void write(const u8 *data, usize byte_count) override;
        void close() override;
    private:
        void flush_stream(bool finish);
        Output_File_Stream file_;
        Unique_Pointer<struct Lzma_Stream_Box> stream_; // pimpl — keeps <lzma.h> out of this header
        bool closed_ = false;
    };

    class Xz_Source : public Spire_Source {
    public:
        explicit Xz_Source(const String &filename);
        ~Xz_Source() override;
        usize read(u8 *out, usize byte_count) override;
    private:
        Input_File_Stream file_;
        Unique_Pointer<struct Lzma_Stream_Box> stream_;
        Vector<u8> input_buffer_;
        bool input_exhausted_ = false;
        bool stream_ended_ = false;
    };
#endif

#ifdef SPIKECOREC_HAVE_BZ2
    class Bz2_Sink : public Spire_Sink {
    public:
        Bz2_Sink(const String &filename, int compression_level);
        ~Bz2_Sink() override;
        void write(const u8 *data, usize byte_count) override;
        void close() override;
    private:
        FILE *file_ = nullptr;
        void *bz_file_ = nullptr; // BZFILE*
        bool closed_ = false;
    };

    class Bz2_Source : public Spire_Source {
    public:
        explicit Bz2_Source(const String &filename);
        ~Bz2_Source() override;
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
    class Async_Spire_Writer {
    public:
        Async_Spire_Writer() = delete;
        Async_Spire_Writer(const Async_Spire_Writer &) = delete;
        Async_Spire_Writer &operator=(const Async_Spire_Writer &) = delete;

        explicit Async_Spire_Writer(Unique_Pointer<Spire_Sink> sink, usize max_queued_chunks = 8);
        ~Async_Spire_Writer();

        // Blocks until there is room in the queue (backpressure), then enqueues
        // `chunk` for the worker thread to write. Re-raises any exception
        // already captured from the worker before/instead of enqueueing.
        void write(Vector<u8> chunk);

        // Pushes the sentinel, joins the worker thread, and re-raises any
        // captured exception. Safe to call at most once (throws otherwise);
        // the destructor calls this automatically if not already closed.
        void close();

    private:
        void run();
        void rethrow_if_failed();

        Unique_Pointer<Spire_Sink> sink_;
        usize max_queued_chunks_;

        Deque<Optional<Vector<u8>>> queue_; // nullopt = sentinel (EOF)
        Mutex mutex_;
        Condition_Variable not_full_;
        Condition_Variable not_empty_;
        Exception_Pointer error_;
        bool closed_ = false;

        Thread worker_;
    };

    // Sequential `.spire` encoder: writes the 4-byte big-endian neuron_count
    // header on construction, then raw float32 frames (native byte order) via
    // write_frame(). One instance per output file. Not compression-aware —
    // always writes a raw byte stream (see Spire_Sink/Phase 2 for compression).
    class Spire_Writer {
    public:
        Spire_Writer() = delete;
        Spire_Writer(const Spire_Writer &) = delete;
        Spire_Writer &operator=(const Spire_Writer &) = delete;

        Spire_Writer(const String &filename, s64 neuron_count);

        void write_frame(const f32 *membrane_potentials);

        [[nodiscard]] s64 neuron_count() const { return neuron_count_; }

    private:
        Output_File_Stream file_;
        s64 neuron_count_;
    };

    // Sequential `.spire` decoder: reads the 4-byte big-endian neuron_count
    // header on construction, then yields frames via read_frame(). Returns
    // false at a clean end-of-file (no bytes remain before the next frame);
    // throws std::runtime_error if a frame is truncated mid-way (mirrors
    // play_spire_recording's "Truncated recording" check).
    class Spire_Reader {
    public:
        Spire_Reader() = delete;
        Spire_Reader(const Spire_Reader &) = delete;
        Spire_Reader &operator=(const Spire_Reader &) = delete;

        explicit Spire_Reader(const String &filename);

        [[nodiscard]] s64 neuron_count() const { return neuron_count_; }

        bool read_frame(f32 *out_buffer);

    private:
        Input_File_Stream file_;
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
    class Simulation_Recorder {
    public:
        Simulation_Recorder() = delete;
        Simulation_Recorder(const Simulation_Recorder &) = delete;
        Simulation_Recorder &operator=(const Simulation_Recorder &) = delete;

        Simulation_Recorder(
            const String &filename, s64 neuron_count,
            Optional<String> compression = String("auto"), Optional<int> compression_level = nullopt,
            bool async = false, usize queue_max = 8, usize chunk_bytes = 4 * 1024 * 1024);

        ~Simulation_Recorder();

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
        Vector<u8> buffer_;
        Unique_Pointer<Spire_Sink> sink_;                 // used directly when not async
        Unique_Pointer<Async_Spire_Writer> async_writer_;  // used when async
        bool finished_ = false;
    };

    // ── decoding / frame-loading ─────────────────────────────────────────────
    //
    // Ports the file-loading half of `play_spire_recording` (frame decoding
    // only — visualization stays out of scope). Auto-detects compression from
    // the filename extension (mirrors _open_spire_file), parses the header,
    // and reads every frame sequentially into one contiguous row-major buffer.
    // Throws std::runtime_error on a truncated final frame (mirrors the
    // reference's "Truncated recording" check). Backed by Spire_Source/
    // Spire_Reader-equivalent decoding — callers wanting frame-at-a-time
    // streaming access can drive Spire_Reader directly instead.
    struct Spire_Recording {
        s64 neuron_count;
        s64 frame_count;
        Vector<f32> frames; // row-major [frame_count][neuron_count]
    };

    Spire_Recording read_spire_recording(const String &filename);

} // namespace spikecorec
