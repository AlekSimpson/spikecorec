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
        NML_STANDARD // any other extension: rows of time, then each recorded value
    };

    // Unused: nothing stores recorded values this way yet.
    // union Recorded_Value {
    //     f64 float64;
    //     s64 int64;
    //     f32 float32;
    //     s32 int32;
    // };
    
    // One recorded quantity: a LEMS <OutputColumn quantity="pop1[0]/v"/> or
    // <EventSelection select="pop1[0]" eventPort="spike"/>.
    struct Recording_Selection {
        String quantity_path;   // verbatim LEMS path, e.g. "pop1[0]/v"
        String variable_name;   // the state variable it records, e.g. "v"; "" for an event selection
        String event_port;      // EventSelection's port, e.g. "spike"; "" for a column
        String selection_id;    // the element's own id; an event file writes it beside each time
        s64 neuron_index = -1;
    };

    struct Recording_Config {
        Vector<String> output_filenames;
        Vector<Output_File_Format> file_output_format;
        // Where a relative filename goes: the LEMS file's directory, or the element's path in it.
        String output_directory;
        // An EventOutputFile's column order, "TIME_ID" or "ID_TIME"; "" for an OutputFile.
        String event_format;
        // Unused: nothing reads or writes it.
        // Vector<Recorded_Value> recored_data;

        // One entry per OutputColumn or EventSelection; recordings_count is its size.
        Vector<Recording_Selection> selections;

        static constexpr s64 DEFAULT_MAX_LOG_BYTES = 512 * 1024 * 1014;
        s32 recordings_count;

        // Unused: declared but never defined or called.
        // void add_output(String &filename);
    };



    enum class Spire_Compression { None, Gzip, Xz, Bz2 };

    // "auto" or nullopt infers the compression from the extension (.gz/.gzip, .xz/.lzma, .bz2,
    // else none). Otherwise `requested` names it (none/raw, gzip/gz, xz/lzma, bz2/bzip2) and an
    // unknown name throws. The returned path is always `filename`.
    Pair<Spire_Compression, String> resolve_spire_compression(
        const String &filename, Optional<String> requested = String("auto"));

    // ── byte-stream sink/source seam ─────────────────────────────────────────
    //
    // One interface over raw, gzip, xz and bz2 streams, used by Simulation_Recorder and
    // read_spire_recording. `compression_level` is the library's own level (gzip and bz2 1-9,
    // xz preset 0-9); the default is 6.

    class Spire_Sink {
    public:
        virtual ~Spire_Sink() = default;
        virtual void write(const u8 *data, usize byte_count) = 0;
        virtual void close() = 0;
    };

    class Spire_Source {
    public:
        virtual ~Spire_Source() = default;
        // Returns the number of bytes read into `out`, at most `byte_count`; 0 at end of stream.
        virtual usize read(u8 *out, usize byte_count) = 0;
    };

    // Throws std::runtime_error when the codec's library was not available at build time.
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
    // A worker thread drains a bounded queue of chunks into `sink`. write() blocks while the
    // queue is full; max_queued_chunks = 0 means unbounded. A worker exception is rethrown from
    // the next write() or close().
    class Async_Spire_Writer {
    public:
        Async_Spire_Writer() = delete;
        Async_Spire_Writer(const Async_Spire_Writer &) = delete;
        Async_Spire_Writer &operator=(const Async_Spire_Writer &) = delete;

        explicit Async_Spire_Writer(Unique_Pointer<Spire_Sink> sink, usize max_queued_chunks = 8);
        ~Async_Spire_Writer();

        // Waits for queue space, then enqueues `chunk`; throws a captured worker error instead.
        void write(Vector<u8> chunk);

        // Drains the queue, closes the sink, joins the worker and rethrows any worker error.
        // A second call throws; the destructor calls it if needed.
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

    // Uncompressed `.spire` encoder. File layout: a 4-byte big-endian u32 neuron_count, then
    // frames of neuron_count float32 values in native byte order.
    class Spire_Writer {
    public:
        Spire_Writer() = delete;
        Spire_Writer(const Spire_Writer &) = delete;
        Spire_Writer &operator=(const Spire_Writer &) = delete;

        Spire_Writer(const String &filename, s64 neuron_count);

        // Each throws when the file cannot be written, and write_frame after close().
        void write_frame(const f32 *membrane_potentials);

        // Pushes the frames written so far to the file.
        void flush();

        // Flushes and closes the file; later calls do nothing. The destructor closes it too.
        void close();

        [[nodiscard]] s64 neuron_count() const { return neuron_count_; }

    private:
        Output_File_Stream file_;
        s64 neuron_count_;
        bool closed_ = false;
    };

    // Uncompressed `.spire` decoder. read_frame() returns false at a clean end of file and
    // throws std::runtime_error on a truncated frame.
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
    // Writes a `.spire` file through the resolved compression and an optional async writer.
    // The header and frames collect in a buffer that is flushed each time it reaches
    // `chunk_bytes`. The destructor calls finish() if the caller did not.
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

        // Throws unless `frame_length` equals neuron_count(), so a wrongly sized array from a
        // caller such as the Python binding is never read past its end.
        void record_frame(const f32 *membrane_potentials, s64 frame_length);

        [[nodiscard]] s64 neuron_count() const { return neuron_count_; }

        // Flushes the buffer and closes the sink; later calls do nothing.
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
    // read_spire_recording loads a whole `.spire` file, compression inferred from the extension,
    // and throws std::runtime_error on a truncated final frame. Spire_Reader reads an
    // uncompressed file one frame at a time.
    struct Spire_Recording {
        s64 neuron_count;
        s64 frame_count;
        Vector<f32> frames; // row-major [frame_count][neuron_count]
    };

    Spire_Recording read_spire_recording(const String &filename);

} // namespace spikecorec
