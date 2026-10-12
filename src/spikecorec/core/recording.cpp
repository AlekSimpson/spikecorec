//
// Created by Alek Simpson on 6/7/26.
//

#include <algorithm>
#include <cctype>
#include <cstdio>

#ifdef SPIKECOREC_HAVE_ZLIB
#include <zlib.h>
#endif
#ifdef SPIKECOREC_HAVE_LZMA
#include <lzma.h>
#endif
#ifdef SPIKECOREC_HAVE_BZ2
#include <bzlib.h>
#endif

#include "recording.h"
#include "log.h"

using namespace std;
using namespace spikecorec;

namespace {

    String lowercase(String value) {
        transform(value.begin(), value.end(), value.begin(),
                  [](unsigned char c) { return static_cast<char>(tolower(c)); });
        return value;
    }

    bool ends_with(const String &value, const String &suffix) {
        return value.size() >= suffix.size()
            && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    Spire_Compression compression_from_extension(const String &filename) {
        String lowered = lowercase(filename);
        if (ends_with(lowered, ".gz") || ends_with(lowered, ".gzip")) return Spire_Compression::Gzip;
        if (ends_with(lowered, ".xz") || ends_with(lowered, ".lzma")) return Spire_Compression::Xz;
        if (ends_with(lowered, ".bz2")) return Spire_Compression::Bz2;
        return Spire_Compression::None;
    }

    Spire_Compression compression_from_name(const String &name) {
        String lowered = lowercase(name);
        if (lowered == "none" || lowered == "raw") return Spire_Compression::None;
        if (lowered == "gzip" || lowered == "gz") return Spire_Compression::Gzip;
        if (lowered == "xz" || lowered == "lzma") return Spire_Compression::Xz;
        if (lowered == "bz2" || lowered == "bzip2") return Spire_Compression::Bz2;
        log::throw_runtime_error(log::logger(),
            log::format("Unknown .spire compression \"{}\" — expected one of: auto, none, gzip, xz, bz2", name));
    }

    // The .spire header is a big-endian u32; native float frames follow as-is.
    u32 to_big_endian_u32(u32 host_value) {
        return ((host_value & 0x000000FFu) << 24)
             | ((host_value & 0x0000FF00u) << 8)
             | ((host_value & 0x00FF0000u) >> 8)
             | ((host_value & 0xFF000000u) >> 24);
    }

    u32 from_big_endian_u32(u32 big_endian_value) {
        return to_big_endian_u32(big_endian_value); // byte-swap is its own inverse
    }

} // namespace

Pair<Spire_Compression, String> spikecorec::resolve_spire_compression(
    const String &filename, Optional<String> requested
) {
    Pair<Spire_Compression, String> result;
    if (!requested.has_value()) {
        result = {compression_from_extension(filename), filename};
    } else {
        String lowered = lowercase(*requested);
        result = (lowered == "auto")
            ? make_pair(compression_from_extension(filename), filename)
            : make_pair(compression_from_name(lowered), filename);
    }

    log::logger().debug("resolve_spire_compression: filename={} requested={} -> compression={}",
                        filename, requested.value_or("(none)"), static_cast<int>(result.first));
    return result;
}

// ── sink/source factory ──────────────────────────────────────────────────────

namespace {

    [[maybe_unused]] [[noreturn]] void throw_missing_library(const char *codec_name, const char *library_hint) {
        log::throw_runtime_error(log::logger(),
            log::format("spikecorec was built without {} support — rebuild with {} available, "
                        "or use an uncompressed .spire path", codec_name, library_hint));
    }

} // namespace

Unique_Pointer<Spire_Sink> spikecorec::make_spire_sink(const String &filename, Spire_Compression compression, [[maybe_unused]] int compression_level) {
    log::logger().debug("make_spire_sink: filename={} compression={} compression_level={}",
                        filename, static_cast<int>(compression), compression_level);
    switch (compression) {
        case Spire_Compression::None: return make_unique<Raw_Sink>(filename);
        case Spire_Compression::Gzip:
#ifdef SPIKECOREC_HAVE_ZLIB
            return make_unique<Gzip_Sink>(filename, compression_level);
#else
            throw_missing_library("gzip", "zlib (libz)");
#endif
        case Spire_Compression::Xz:
#ifdef SPIKECOREC_HAVE_LZMA
            return make_unique<Xz_Sink>(filename, compression_level);
#else
            throw_missing_library("xz", "liblzma");
#endif
        case Spire_Compression::Bz2:
#ifdef SPIKECOREC_HAVE_BZ2
            return make_unique<Bz2_Sink>(filename, compression_level);
#else
            throw_missing_library("bz2", "libbz2");
#endif
    }
    log::throw_runtime_error(log::logger(), "Unhandled Spire_Compression value");
}

Unique_Pointer<Spire_Source> spikecorec::make_spire_source(const String &filename, Spire_Compression compression) {
    log::logger().debug("make_spire_source: filename={} compression={}", filename, static_cast<int>(compression));
    switch (compression) {
        case Spire_Compression::None: return make_unique<Raw_Source>(filename);
        case Spire_Compression::Gzip:
#ifdef SPIKECOREC_HAVE_ZLIB
            return make_unique<Gzip_Source>(filename);
#else
            throw_missing_library("gzip", "zlib (libz)");
#endif
        case Spire_Compression::Xz:
#ifdef SPIKECOREC_HAVE_LZMA
            return make_unique<Xz_Source>(filename);
#else
            throw_missing_library("xz", "liblzma");
#endif
        case Spire_Compression::Bz2:
#ifdef SPIKECOREC_HAVE_BZ2
            return make_unique<Bz2_Source>(filename);
#else
            throw_missing_library("bz2", "libbz2");
#endif
    }
    log::throw_runtime_error(log::logger(), "Unhandled Spire_Compression value");
}

// ── Raw_Sink / Raw_Source ──────────────────────────────────────────────────────

Raw_Sink::Raw_Sink(const String &filename) : file_(filename, Stream_Base::binary) {
    log::logger().debug("Raw_Sink: filename={}", filename);
    if (!file_) log::throw_runtime_error(log::logger(), log::format("Failed to open file for writing: {}", filename));
}

void Raw_Sink::write(const u8 *data, usize byte_count) {
    file_.write(reinterpret_cast<const char *>(data), static_cast<streamsize>(byte_count));
}

void Raw_Sink::close() {
    file_.close();
}

Raw_Source::Raw_Source(const String &filename) : file_(filename, Stream_Base::binary) {
    log::logger().debug("Raw_Source: filename={}", filename);
    if (!file_) log::throw_runtime_error(log::logger(), log::format("Failed to open file for reading: {}", filename));
}

usize Raw_Source::read(u8 *out, usize byte_count) {
    file_.read(reinterpret_cast<char *>(out), static_cast<streamsize>(byte_count));
    return static_cast<usize>(file_.gcount());
}

// ── Gzip (zlib) ──────────────────────────────────────────────────────────────

#ifdef SPIKECOREC_HAVE_ZLIB

Gzip_Sink::Gzip_Sink(const String &filename, int compression_level) {
    log::logger().debug("Gzip_Sink: filename={} compression_level={}", filename, compression_level);
    char mode[8];
    snprintf(mode, sizeof(mode), "wb%d", compression_level);
    gz_file_ = gzopen(filename.c_str(), mode);
    if (gz_file_ == nullptr)
        log::throw_runtime_error(log::logger(), log::format("Failed to open gzip file for writing: {}", filename));
}

Gzip_Sink::~Gzip_Sink() { if (!closed_) close(); }

void Gzip_Sink::write(const u8 *data, usize byte_count) {
    // gzwrite takes an `unsigned` length; write in bounded slices so a chunk
    // larger than UINT_MAX can't silently truncate to its low 32 bits.
    constexpr usize MAX_IO = usize(1) << 30; // 1 GiB per call
    usize offset = 0;
    while (offset < byte_count) {
        unsigned slice = static_cast<unsigned>(std::min(byte_count - offset, MAX_IO));
        int written = gzwrite(static_cast<gzFile>(gz_file_), data + offset, slice);
        if (written <= 0 || static_cast<unsigned>(written) != slice)
            log::throw_runtime_error(log::logger(), "gzwrite failed (short write)");
        offset += slice;
    }
}

void Gzip_Sink::close() {
    if (closed_) return;
    closed_ = true;
    int result = gzclose(static_cast<gzFile>(gz_file_));
    gz_file_ = nullptr;
    if (result != Z_OK) log::throw_runtime_error(log::logger(), "gzclose failed");
}

Gzip_Source::Gzip_Source(const String &filename) {
    log::logger().debug("Gzip_Source: filename={}", filename);
    gz_file_ = gzopen(filename.c_str(), "rb");
    if (gz_file_ == nullptr)
        log::throw_runtime_error(log::logger(), log::format("Failed to open gzip file for reading: {}", filename));
}

Gzip_Source::~Gzip_Source() {
    if (gz_file_ != nullptr) gzclose(static_cast<gzFile>(gz_file_));
}

usize Gzip_Source::read(u8 *out, usize byte_count) {
    int result = gzread(static_cast<gzFile>(gz_file_), out, static_cast<unsigned>(byte_count));
    if (result < 0) log::throw_runtime_error(log::logger(), "gzread failed");
    return static_cast<usize>(result);
}

#endif // SPIKECOREC_HAVE_ZLIB

// ── Xz (liblzma) ─────────────────────────────────────────────────────────────

#ifdef SPIKECOREC_HAVE_LZMA

namespace spikecorec { struct Lzma_Stream_Box { lzma_stream stream = LZMA_STREAM_INIT; }; }

namespace {
    constexpr usize LZMA_CHUNK_SIZE = 64 * 1024;

    [[noreturn]] void throw_lzma_error(const char *what, lzma_ret code) {
        log::throw_runtime_error(log::logger(), log::format("{} failed with lzma_ret={}", what, static_cast<int>(code)));
    }
}

Xz_Sink::Xz_Sink(const String &filename, int compression_level)
    : file_(filename, Stream_Base::binary)
    , stream_(make_unique<Lzma_Stream_Box>())
{
    log::logger().debug("Xz_Sink: filename={} compression_level={}", filename, compression_level);
    if (!file_) log::throw_runtime_error(log::logger(), log::format("Failed to open xz file for writing: {}", filename));

    uint32_t preset = static_cast<uint32_t>(std::clamp(compression_level, 0, 9));
    lzma_ret rc = lzma_easy_encoder(&stream_->stream, preset, LZMA_CHECK_CRC64);
    if (rc != LZMA_OK) throw_lzma_error("lzma_easy_encoder", rc);
}

Xz_Sink::~Xz_Sink() { if (!closed_) close(); }

void Xz_Sink::flush_stream(bool finish) {
    Vector<u8> output_buffer(LZMA_CHUNK_SIZE);
    stream_->stream.next_out = output_buffer.data();
    stream_->stream.avail_out = output_buffer.size();

    lzma_action action = finish ? LZMA_FINISH : LZMA_RUN;
    for (;;) {
        lzma_ret rc = lzma_code(&stream_->stream, action);
        if (rc != LZMA_OK && rc != LZMA_STREAM_END) throw_lzma_error("lzma_code", rc);

        usize produced = output_buffer.size() - stream_->stream.avail_out;
        if (produced > 0) {
            file_.write(reinterpret_cast<const char *>(output_buffer.data()), static_cast<streamsize>(produced));
            stream_->stream.next_out = output_buffer.data();
            stream_->stream.avail_out = output_buffer.size();
        }

        if (rc == LZMA_STREAM_END) break;
        if (!finish && stream_->stream.avail_in == 0) break;
    }
}

void Xz_Sink::write(const u8 *data, usize byte_count) {
    stream_->stream.next_in = data;
    stream_->stream.avail_in = byte_count;
    flush_stream(/*finish=*/false);
}

void Xz_Sink::close() {
    if (closed_) return;
    closed_ = true;
    stream_->stream.next_in = nullptr;
    stream_->stream.avail_in = 0;
    flush_stream(/*finish=*/true);
    lzma_end(&stream_->stream);
    file_.close();
}

Xz_Source::Xz_Source(const String &filename)
    : file_(filename, Stream_Base::binary)
    , stream_(make_unique<Lzma_Stream_Box>())
    , input_buffer_(LZMA_CHUNK_SIZE)
{
    log::logger().debug("Xz_Source: filename={}", filename);
    if (!file_) log::throw_runtime_error(log::logger(), log::format("Failed to open xz file for reading: {}", filename));

    lzma_ret rc = lzma_stream_decoder(&stream_->stream, UINT64_MAX, LZMA_CONCATENATED);
    if (rc != LZMA_OK) throw_lzma_error("lzma_stream_decoder", rc);
}

Xz_Source::~Xz_Source() { lzma_end(&stream_->stream); }

usize Xz_Source::read(u8 *out, usize byte_count) {
    if (stream_ended_ || byte_count == 0) return 0;

    stream_->stream.next_out = out;
    stream_->stream.avail_out = byte_count;

    while (stream_->stream.avail_out == byte_count) {
        if (stream_->stream.avail_in == 0 && !input_exhausted_) {
            file_.read(reinterpret_cast<char *>(input_buffer_.data()), static_cast<streamsize>(input_buffer_.size()));
            usize bytes_read = static_cast<usize>(file_.gcount());
            if (bytes_read == 0) input_exhausted_ = true;
            stream_->stream.next_in = input_buffer_.data();
            stream_->stream.avail_in = bytes_read;
        }

        lzma_action action = input_exhausted_ ? LZMA_FINISH : LZMA_RUN;
        lzma_ret rc = lzma_code(&stream_->stream, action);

        if (rc == LZMA_STREAM_END) { stream_ended_ = true; break; }
        if (rc != LZMA_OK) throw_lzma_error("lzma_code", rc);
        if (input_exhausted_ && stream_->stream.avail_in == 0 && stream_->stream.avail_out == byte_count) break;
    }

    return byte_count - stream_->stream.avail_out;
}

#endif // SPIKECOREC_HAVE_LZMA

// ── Bz2 (libbz2) ─────────────────────────────────────────────────────────────

#ifdef SPIKECOREC_HAVE_BZ2

namespace {
    [[noreturn]] void throw_bz2_error(const char *what, int code) {
        log::throw_runtime_error(log::logger(), log::format("{} failed with BZ2 error={}", what, code));
    }
}

Bz2_Sink::Bz2_Sink(const String &filename, int compression_level) {
    log::logger().debug("Bz2_Sink: filename={} compression_level={}", filename, compression_level);
    file_ = fopen(filename.c_str(), "wb");
    if (file_ == nullptr)
        log::throw_runtime_error(log::logger(), log::format("Failed to open bz2 file for writing: {}", filename));

    int block_size_100k = std::clamp(compression_level, 1, 9);
    int bz_error = BZ_OK;
    bz_file_ = BZ2_bzWriteOpen(&bz_error, file_, block_size_100k, /*verbosity=*/0, /*work_factor=*/0);
    if (bz_error != BZ_OK) throw_bz2_error("BZ2_bzWriteOpen", bz_error);
}

Bz2_Sink::~Bz2_Sink() { if (!closed_) close(); }

void Bz2_Sink::write(const u8 *data, usize byte_count) {
    // BZ2_bzWrite takes an `int` length; write in bounded slices so a chunk
    // larger than INT_MAX can't overflow to a negative length.
    constexpr usize MAX_IO = usize(1) << 30; // 1 GiB per call
    usize offset = 0;
    while (offset < byte_count) {
        int slice = static_cast<int>(std::min(byte_count - offset, MAX_IO));
        int bz_error = BZ_OK;
        BZ2_bzWrite(&bz_error, static_cast<BZFILE *>(bz_file_), const_cast<u8 *>(data + offset), slice);
        if (bz_error != BZ_OK) throw_bz2_error("BZ2_bzWrite", bz_error);
        offset += static_cast<usize>(slice);
    }
}

void Bz2_Sink::close() {
    if (closed_) return;
    closed_ = true;
    int bz_error = BZ_OK;
    BZ2_bzWriteClose(&bz_error, static_cast<BZFILE *>(bz_file_), /*abandon=*/0, nullptr, nullptr);
    bz_file_ = nullptr;
    fclose(file_);
    file_ = nullptr;
    if (bz_error != BZ_OK) throw_bz2_error("BZ2_bzWriteClose", bz_error);
}

Bz2_Source::Bz2_Source(const String &filename) {
    log::logger().debug("Bz2_Source: filename={}", filename);
    file_ = fopen(filename.c_str(), "rb");
    if (file_ == nullptr)
        log::throw_runtime_error(log::logger(), log::format("Failed to open bz2 file for reading: {}", filename));

    int bz_error = BZ_OK;
    bz_file_ = BZ2_bzReadOpen(&bz_error, file_, /*verbosity=*/0, /*small=*/0, nullptr, 0);
    if (bz_error != BZ_OK) throw_bz2_error("BZ2_bzReadOpen", bz_error);
}

Bz2_Source::~Bz2_Source() {
    if (bz_file_ != nullptr) {
        int bz_error = BZ_OK;
        BZ2_bzReadClose(&bz_error, static_cast<BZFILE *>(bz_file_));
    }
    if (file_ != nullptr) fclose(file_);
}

usize Bz2_Source::read(u8 *out, usize byte_count) {
    if (stream_ended_ || byte_count == 0) return 0;

    int bz_error = BZ_OK;
    int bytes_read = BZ2_bzRead(&bz_error, static_cast<BZFILE *>(bz_file_), out, static_cast<int>(byte_count));

    if (bz_error == BZ_STREAM_END) stream_ended_ = true;
    else if (bz_error != BZ_OK) throw_bz2_error("BZ2_bzRead", bz_error);

    return static_cast<usize>(bytes_read);
}

#endif // SPIKECOREC_HAVE_BZ2

// ── Async_Spire_Writer ─────────────────────────────────────────────────────────

Async_Spire_Writer::Async_Spire_Writer(Unique_Pointer<Spire_Sink> sink, usize max_queued_chunks)
    : sink_(std::move(sink))
    , max_queued_chunks_(max_queued_chunks)
{
    log::logger().debug("Async_Spire_Writer: max_queued_chunks={}", max_queued_chunks);
    worker_ = Thread(&Async_Spire_Writer::run, this);
}

Async_Spire_Writer::~Async_Spire_Writer() {
    if (!closed_) {
        try { close(); } catch (...) { /* destructors must not throw */ }
    }
}

void Async_Spire_Writer::run() {
    // Once the sink has thrown, it may be in an unrecoverable state — for some
    // backends (e.g. libbz2) calling write()/close() again after an error is
    // undefined behavior. So after the first failure we stop touching the sink
    // entirely: we keep draining the queue (discarding chunks) so the producer's
    // blocking write()/close() calls observe the captured error rather than
    // deadlocking, but we never re-enter the failed sink. Resource cleanup is
    // left to the sink's own destructor.
    bool sink_failed = false;

    for (;;) {
        Optional<Vector<u8>> chunk;
        {
            Unique_Lock<Mutex> lock(mutex_);
            not_empty_.wait(lock, [this] { return !queue_.empty(); });
            chunk = std::move(queue_.front());
            queue_.pop_front();
        }
        not_full_.notify_one();

        if (!chunk.has_value()) { // sentinel — flush and exit
            if (!sink_failed) {
                try {
                    sink_->close();
                } catch (...) {
                    Lock_Guard<Mutex> lock(mutex_);
                    if (!error_) error_ = std::current_exception();
                }
            }
            return;
        }

        if (sink_failed) continue; // already errored — discard without re-entering the sink

        try {
            sink_->write(chunk->data(), chunk->size());
        } catch (...) {
            sink_failed = true;
            Lock_Guard<Mutex> lock(mutex_);
            if (!error_) error_ = std::current_exception();
        }
    }
}

void Async_Spire_Writer::rethrow_if_failed() {
    Exception_Pointer captured;
    {
        Lock_Guard<Mutex> lock(mutex_);
        captured = error_;
    }
    if (captured) std::rethrow_exception(captured);
}

void Async_Spire_Writer::write(Vector<u8> chunk) {
    rethrow_if_failed();

    {
        Unique_Lock<Mutex> lock(mutex_);
        // max_queued_chunks_ == 0 means "unbounded" (matching the Python
        // reference's queue.Queue(maxsize=0)) — never block the producer.
        not_full_.wait(lock, [this] {
            return max_queued_chunks_ == 0 || queue_.size() < max_queued_chunks_ || error_;
        });
        if (error_) {
            Exception_Pointer captured = error_;
            lock.unlock();
            std::rethrow_exception(captured);
        }
        queue_.emplace_back(std::move(chunk));
    }
    not_empty_.notify_one();
}

void Async_Spire_Writer::close() {
    log::logger().debug("Async_Spire_Writer::close");
    if (closed_) log::throw_runtime_error(log::logger(), "Async_Spire_Writer::close() called more than once");
    closed_ = true;

    {
        Lock_Guard<Mutex> lock(mutex_);
        queue_.emplace_back(nullopt); // sentinel
    }
    not_empty_.notify_one();

    if (worker_.joinable()) worker_.join();

    rethrow_if_failed();
}

// ── Simulation_Recorder ───────────────────────────────────────────────────────

Simulation_Recorder::Simulation_Recorder(
    const String &filename, s64 neuron_count,
    Optional<String> compression, Optional<int> compression_level,
    bool async, usize queue_max, usize chunk_bytes
)
    : neuron_count_(neuron_count)
    , chunk_bytes_(chunk_bytes)
{
    log::logger().debug("Simulation_Recorder: filename={} neuron_count={} async={} queue_max={} chunk_bytes={}",
                        filename, neuron_count, async, queue_max, chunk_bytes);
    if (neuron_count_ < 0 || neuron_count_ > static_cast<s64>(UINT32_MAX)) {
        log::throw_runtime_error(log::logger(),
            log::format("Simulation_Recorder: neuron_count {} does not fit in the .spire format's 32-bit header",
                        neuron_count_));
    }

    auto [resolved_compression, resolved_path] = resolve_spire_compression(filename, compression);
    int level = compression_level.value_or(6);

    Unique_Pointer<Spire_Sink> sink = make_spire_sink(resolved_path, resolved_compression, level);
    if (async) async_writer_ = make_unique<Async_Spire_Writer>(std::move(sink), queue_max);
    else       sink_ = std::move(sink);

    buffer_.reserve(chunk_bytes_);
    u32 header = to_big_endian_u32(static_cast<u32>(neuron_count_));
    const u8 *header_bytes = reinterpret_cast<const u8 *>(&header);
    buffer_.insert(buffer_.end(), header_bytes, header_bytes + sizeof(u32));
}

Simulation_Recorder::~Simulation_Recorder() {
    if (!finished_) {
        try { finish(); } catch (...) { /* destructors must not throw */ }
    }
}

void Simulation_Recorder::write_bytes(const u8 *data, usize byte_count) {
    if (async_writer_) async_writer_->write(Vector<u8>(data, data + byte_count));
    else               sink_->write(data, byte_count);
}

void Simulation_Recorder::flush_buffer() {
    if (buffer_.empty()) return;
    log::logger().trace("Simulation_Recorder::flush_buffer: buffer_size={}", buffer_.size());
    write_bytes(buffer_.data(), buffer_.size());
    buffer_.clear();
}

void Simulation_Recorder::record_frame(const f32 *membrane_potentials, s64 frame_length) {
    if (finished_) {
        log::throw_runtime_error(log::logger(), "Simulation_Recorder::record_frame called after finish()");
    }
    if (frame_length != neuron_count_) {
        log::throw_runtime_error(log::logger(),
            log::format("Simulation_Recorder::record_frame: frame length {} does not match "
                        "the recorder's neuron_count {}", frame_length, neuron_count_));
    }

    log::logger().trace("Simulation_Recorder::record_frame: frame_length={}", frame_length);

    const u8 *bytes = reinterpret_cast<const u8 *>(membrane_potentials);
    usize frame_bytes = static_cast<usize>(neuron_count_) * sizeof(f32);
    buffer_.insert(buffer_.end(), bytes, bytes + frame_bytes);

    if (buffer_.size() >= chunk_bytes_) flush_buffer();
}

void Simulation_Recorder::finish() {
    if (finished_) return;
    finished_ = true;

    log::logger().debug("Simulation_Recorder::finish");
    flush_buffer();
    if (async_writer_) async_writer_->close();
    else               sink_->close();
}

// ── Spire_Writer ──────────────────────────────────────────────────────────────

Spire_Writer::Spire_Writer(const String &filename, s64 neuron_count)
    : file_(filename, Stream_Base::binary)
    , neuron_count_(neuron_count)
{
    log::logger().debug("Spire_Writer: filename={} neuron_count={}", filename, neuron_count);
    if (!file_) log::throw_runtime_error(log::logger(), log::format("Failed to open .spire file for writing: {}", filename));
    if (neuron_count_ < 0 || neuron_count_ > static_cast<s64>(UINT32_MAX)) {
        log::throw_runtime_error(log::logger(),
            log::format("Spire_Writer: neuron_count {} does not fit in the .spire format's 32-bit header",
                        neuron_count_));
    }

    u32 header = to_big_endian_u32(static_cast<u32>(neuron_count_));
    file_.write(reinterpret_cast<const char *>(&header), sizeof(u32));
}

void Spire_Writer::write_frame(const f32 *membrane_potentials) {
    log::logger().trace("Spire_Writer::write_frame: neuron_count={}", neuron_count_);
    file_.write(reinterpret_cast<const char *>(membrane_potentials), static_cast<streamsize>(neuron_count_ * sizeof(f32)));
}

// ── Spire_Reader ──────────────────────────────────────────────────────────────

Spire_Reader::Spire_Reader(const String &filename)
    : file_(filename, Stream_Base::binary)
    , neuron_count_(0)
{
    log::logger().debug("Spire_Reader: filename={}", filename);
    if (!file_) log::throw_runtime_error(log::logger(), log::format("Failed to open .spire file for reading: {}", filename));

    u32 header;
    file_.read(reinterpret_cast<char *>(&header), sizeof(u32));
    if (file_.gcount() != static_cast<streamsize>(sizeof(u32))) {
        log::throw_runtime_error(log::logger(),
            log::format(".spire file is too short to contain a neuron_count header: {}", filename));
    }

    neuron_count_ = static_cast<s64>(from_big_endian_u32(header));
    log::logger().debug("Spire_Reader: filename={} neuron_count={}", filename, neuron_count_);
}

bool Spire_Reader::read_frame(f32 *out_buffer) {
    streamsize frame_bytes = static_cast<streamsize>(neuron_count_ * sizeof(f32));

    file_.read(reinterpret_cast<char *>(out_buffer), frame_bytes);
    streamsize bytes_read = file_.gcount();

    if (bytes_read == 0) return false; // clean EOF — no partial frame
    if (bytes_read != frame_bytes) {
        log::throw_runtime_error(log::logger(),
            log::format("Truncated recording: frame ended after {} of {} expected bytes",
                        bytes_read, frame_bytes));
    }

    log::logger().trace("Spire_Reader::read_frame: neuron_count={}", neuron_count_);
    return true;
}

// ── read_spire_recording ─────────────────────────────────────────────────────

Spire_Recording spikecorec::read_spire_recording(const String &filename) {
    log::logger().debug("read_spire_recording: filename={}", filename);
    Spire_Compression compression = resolve_spire_compression(filename, nullopt).first;
    Unique_Pointer<Spire_Source> source = make_spire_source(filename, compression);

    // Spire_Source::read may return short reads (e.g. compressed streams hand
    // back chunks as they decode), so loop until `byte_count` bytes are
    // collected or the source reaches a clean EOF.
    auto read_fully = [&](u8 *out, usize byte_count) -> usize {
        usize total = 0;
        while (total < byte_count) {
            usize got = source->read(out + total, byte_count - total);
            if (got == 0) break;
            total += got;
        }
        return total;
    };

    u32 header;
    if (read_fully(reinterpret_cast<u8 *>(&header), sizeof(u32)) != sizeof(u32)) {
        log::throw_runtime_error(log::logger(),
            log::format(".spire file is too short to contain a neuron_count header: {}", filename));
    }

    Spire_Recording recording;
    recording.neuron_count = static_cast<s64>(from_big_endian_u32(header));
    recording.frame_count = 0;

    usize frame_bytes = static_cast<usize>(recording.neuron_count) * sizeof(f32);
    Vector<f32> frame(static_cast<usize>(recording.neuron_count));

    for (;;) {
        usize bytes_read = read_fully(reinterpret_cast<u8 *>(frame.data()), frame_bytes);
        if (bytes_read == 0) break; // clean EOF — no partial frame

        if (bytes_read != frame_bytes) {
            log::throw_runtime_error(log::logger(),
                log::format("Truncated recording: frame ended after {} of {} expected bytes",
                            bytes_read, frame_bytes));
        }

        recording.frames.insert(recording.frames.end(), frame.begin(), frame.end());
        ++recording.frame_count;
    }

    log::logger().debug("read_spire_recording: filename={} neuron_count={} frame_count={}",
                        filename, recording.neuron_count, recording.frame_count);
    return recording;
}
