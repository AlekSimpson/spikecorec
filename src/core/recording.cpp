//
// Created by Alek Simpson on 6/7/26.
//

#include <stdexcept>
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

#include "spikecorec/core/recording.h"

using namespace std;
using namespace spikecorec;

namespace {

    string lowercase(string value) {
        transform(value.begin(), value.end(), value.begin(),
                  [](unsigned char c) { return (char)tolower(c); });
        return value;
    }

    bool ends_with(const string &value, const string &suffix) {
        return value.size() >= suffix.size()
            && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    SpireCompression compression_from_extension(const string &filename) {
        string lowered = lowercase(filename);
        if (ends_with(lowered, ".gz") || ends_with(lowered, ".gzip")) return SpireCompression::Gzip;
        if (ends_with(lowered, ".xz") || ends_with(lowered, ".lzma")) return SpireCompression::Xz;
        if (ends_with(lowered, ".bz2")) return SpireCompression::Bz2;
        return SpireCompression::None;
    }

    SpireCompression compression_from_name(const string &name) {
        string lowered = lowercase(name);
        if (lowered == "none" || lowered == "raw") return SpireCompression::None;
        if (lowered == "gzip" || lowered == "gz") return SpireCompression::Gzip;
        if (lowered == "xz" || lowered == "lzma") return SpireCompression::Xz;
        if (lowered == "bz2" || lowered == "bzip2") return SpireCompression::Bz2;
        throw std::runtime_error("Unknown .spire compression \"" + name + "\" — expected one of: auto, none, gzip, xz, bz2");
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

pair<SpireCompression, string> spikecorec::resolve_spire_compression(
    const string &filename, optional<string> requested
) {
    if (!requested.has_value()) return {compression_from_extension(filename), filename};

    string lowered = lowercase(*requested);
    if (lowered == "auto") return {compression_from_extension(filename), filename};

    return {compression_from_name(lowered), filename};
}

// ── sink/source factory ──────────────────────────────────────────────────────

namespace {

    [[maybe_unused]] [[noreturn]] void throw_missing_library(const char *codec_name, const char *library_hint) {
        throw std::runtime_error(string("spikecorec was built without ") + codec_name
            + " support — rebuild with " + library_hint + " available, or use an uncompressed .spire path");
    }

} // namespace

unique_ptr<SpireSink> spikecorec::make_spire_sink(const string &filename, SpireCompression compression, [[maybe_unused]] int compression_level) {
    switch (compression) {
        case SpireCompression::None: return make_unique<RawSink>(filename);
        case SpireCompression::Gzip:
#ifdef SPIKECOREC_HAVE_ZLIB
            return make_unique<GzipSink>(filename, compression_level);
#else
            throw_missing_library("gzip", "zlib (libz)");
#endif
        case SpireCompression::Xz:
#ifdef SPIKECOREC_HAVE_LZMA
            return make_unique<XzSink>(filename, compression_level);
#else
            throw_missing_library("xz", "liblzma");
#endif
        case SpireCompression::Bz2:
#ifdef SPIKECOREC_HAVE_BZ2
            return make_unique<Bz2Sink>(filename, compression_level);
#else
            throw_missing_library("bz2", "libbz2");
#endif
    }
    throw std::runtime_error("Unhandled SpireCompression value");
}

unique_ptr<SpireSource> spikecorec::make_spire_source(const string &filename, SpireCompression compression) {
    switch (compression) {
        case SpireCompression::None: return make_unique<RawSource>(filename);
        case SpireCompression::Gzip:
#ifdef SPIKECOREC_HAVE_ZLIB
            return make_unique<GzipSource>(filename);
#else
            throw_missing_library("gzip", "zlib (libz)");
#endif
        case SpireCompression::Xz:
#ifdef SPIKECOREC_HAVE_LZMA
            return make_unique<XzSource>(filename);
#else
            throw_missing_library("xz", "liblzma");
#endif
        case SpireCompression::Bz2:
#ifdef SPIKECOREC_HAVE_BZ2
            return make_unique<Bz2Source>(filename);
#else
            throw_missing_library("bz2", "libbz2");
#endif
    }
    throw std::runtime_error("Unhandled SpireCompression value");
}

// ── RawSink / RawSource ──────────────────────────────────────────────────────

RawSink::RawSink(const string &filename) : file_(filename, ios::binary) {
    if (!file_) throw std::runtime_error("Failed to open file for writing: " + filename);
}

void RawSink::write(const u8 *data, usize byte_count) {
    file_.write(reinterpret_cast<const char *>(data), (streamsize)byte_count);
}

void RawSink::close() {
    file_.close();
}

RawSource::RawSource(const string &filename) : file_(filename, ios::binary) {
    if (!file_) throw std::runtime_error("Failed to open file for reading: " + filename);
}

usize RawSource::read(u8 *out, usize byte_count) {
    file_.read(reinterpret_cast<char *>(out), (streamsize)byte_count);
    return (usize)file_.gcount();
}

// ── Gzip (zlib) ──────────────────────────────────────────────────────────────

#ifdef SPIKECOREC_HAVE_ZLIB

GzipSink::GzipSink(const string &filename, int compression_level) {
    char mode[8];
    snprintf(mode, sizeof(mode), "wb%d", compression_level);
    gz_file_ = gzopen(filename.c_str(), mode);
    if (gz_file_ == nullptr) throw std::runtime_error("Failed to open gzip file for writing: " + filename);
}

GzipSink::~GzipSink() { if (!closed_) close(); }

void GzipSink::write(const u8 *data, usize byte_count) {
    // gzwrite takes an `unsigned` length; write in bounded slices so a chunk
    // larger than UINT_MAX can't silently truncate to its low 32 bits.
    constexpr usize MAX_IO = usize(1) << 30; // 1 GiB per call
    usize offset = 0;
    while (offset < byte_count) {
        unsigned slice = (unsigned)std::min(byte_count - offset, MAX_IO);
        int written = gzwrite((gzFile)gz_file_, data + offset, slice);
        if (written <= 0 || (unsigned)written != slice)
            throw std::runtime_error("gzwrite failed (short write)");
        offset += slice;
    }
}

void GzipSink::close() {
    if (closed_) return;
    closed_ = true;
    int result = gzclose((gzFile)gz_file_);
    gz_file_ = nullptr;
    if (result != Z_OK) throw std::runtime_error("gzclose failed");
}

GzipSource::GzipSource(const string &filename) {
    gz_file_ = gzopen(filename.c_str(), "rb");
    if (gz_file_ == nullptr) throw std::runtime_error("Failed to open gzip file for reading: " + filename);
}

GzipSource::~GzipSource() {
    if (gz_file_ != nullptr) gzclose((gzFile)gz_file_);
}

usize GzipSource::read(u8 *out, usize byte_count) {
    int result = gzread((gzFile)gz_file_, out, (unsigned)byte_count);
    if (result < 0) throw std::runtime_error("gzread failed");
    return (usize)result;
}

#endif // SPIKECOREC_HAVE_ZLIB

// ── Xz (liblzma) ─────────────────────────────────────────────────────────────

#ifdef SPIKECOREC_HAVE_LZMA

namespace spikecorec { struct lzma_stream_box { lzma_stream stream = LZMA_STREAM_INIT; }; }

namespace {
    constexpr usize LZMA_CHUNK_SIZE = 64 * 1024;

    [[noreturn]] void throw_lzma_error(const char *what, lzma_ret code) {
        throw std::runtime_error(string(what) + " failed with lzma_ret=" + std::to_string((int)code));
    }
}

XzSink::XzSink(const string &filename, int compression_level)
    : file_(filename, ios::binary)
    , stream_(make_unique<lzma_stream_box>())
{
    if (!file_) throw std::runtime_error("Failed to open xz file for writing: " + filename);

    uint32_t preset = (uint32_t)std::clamp(compression_level, 0, 9);
    lzma_ret rc = lzma_easy_encoder(&stream_->stream, preset, LZMA_CHECK_CRC64);
    if (rc != LZMA_OK) throw_lzma_error("lzma_easy_encoder", rc);
}

XzSink::~XzSink() { if (!closed_) close(); }

void XzSink::flush_stream(bool finish) {
    vector<u8> output_buffer(LZMA_CHUNK_SIZE);
    stream_->stream.next_out = output_buffer.data();
    stream_->stream.avail_out = output_buffer.size();

    lzma_action action = finish ? LZMA_FINISH : LZMA_RUN;
    for (;;) {
        lzma_ret rc = lzma_code(&stream_->stream, action);
        if (rc != LZMA_OK && rc != LZMA_STREAM_END) throw_lzma_error("lzma_code", rc);

        usize produced = output_buffer.size() - stream_->stream.avail_out;
        if (produced > 0) {
            file_.write(reinterpret_cast<const char *>(output_buffer.data()), (streamsize)produced);
            stream_->stream.next_out = output_buffer.data();
            stream_->stream.avail_out = output_buffer.size();
        }

        if (rc == LZMA_STREAM_END) break;
        if (!finish && stream_->stream.avail_in == 0) break;
    }
}

void XzSink::write(const u8 *data, usize byte_count) {
    stream_->stream.next_in = data;
    stream_->stream.avail_in = byte_count;
    flush_stream(/*finish=*/false);
}

void XzSink::close() {
    if (closed_) return;
    closed_ = true;
    stream_->stream.next_in = nullptr;
    stream_->stream.avail_in = 0;
    flush_stream(/*finish=*/true);
    lzma_end(&stream_->stream);
    file_.close();
}

XzSource::XzSource(const string &filename)
    : file_(filename, ios::binary)
    , stream_(make_unique<lzma_stream_box>())
    , input_buffer_(LZMA_CHUNK_SIZE)
{
    if (!file_) throw std::runtime_error("Failed to open xz file for reading: " + filename);

    lzma_ret rc = lzma_stream_decoder(&stream_->stream, UINT64_MAX, LZMA_CONCATENATED);
    if (rc != LZMA_OK) throw_lzma_error("lzma_stream_decoder", rc);
}

XzSource::~XzSource() { lzma_end(&stream_->stream); }

usize XzSource::read(u8 *out, usize byte_count) {
    if (stream_ended_ || byte_count == 0) return 0;

    stream_->stream.next_out = out;
    stream_->stream.avail_out = byte_count;

    while (stream_->stream.avail_out == byte_count) {
        if (stream_->stream.avail_in == 0 && !input_exhausted_) {
            file_.read(reinterpret_cast<char *>(input_buffer_.data()), (streamsize)input_buffer_.size());
            usize bytes_read = (usize)file_.gcount();
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
        throw std::runtime_error(string(what) + " failed with BZ2 error=" + std::to_string(code));
    }
}

Bz2Sink::Bz2Sink(const string &filename, int compression_level) {
    file_ = fopen(filename.c_str(), "wb");
    if (file_ == nullptr) throw std::runtime_error("Failed to open bz2 file for writing: " + filename);

    int block_size_100k = std::clamp(compression_level, 1, 9);
    int bz_error = BZ_OK;
    bz_file_ = BZ2_bzWriteOpen(&bz_error, file_, block_size_100k, /*verbosity=*/0, /*work_factor=*/0);
    if (bz_error != BZ_OK) throw_bz2_error("BZ2_bzWriteOpen", bz_error);
}

Bz2Sink::~Bz2Sink() { if (!closed_) close(); }

void Bz2Sink::write(const u8 *data, usize byte_count) {
    // BZ2_bzWrite takes an `int` length; write in bounded slices so a chunk
    // larger than INT_MAX can't overflow to a negative length.
    constexpr usize MAX_IO = usize(1) << 30; // 1 GiB per call
    usize offset = 0;
    while (offset < byte_count) {
        int slice = (int)std::min(byte_count - offset, MAX_IO);
        int bz_error = BZ_OK;
        BZ2_bzWrite(&bz_error, (BZFILE *)bz_file_, const_cast<u8 *>(data + offset), slice);
        if (bz_error != BZ_OK) throw_bz2_error("BZ2_bzWrite", bz_error);
        offset += (usize)slice;
    }
}

void Bz2Sink::close() {
    if (closed_) return;
    closed_ = true;
    int bz_error = BZ_OK;
    BZ2_bzWriteClose(&bz_error, (BZFILE *)bz_file_, /*abandon=*/0, nullptr, nullptr);
    bz_file_ = nullptr;
    fclose(file_);
    file_ = nullptr;
    if (bz_error != BZ_OK) throw_bz2_error("BZ2_bzWriteClose", bz_error);
}

Bz2Source::Bz2Source(const string &filename) {
    file_ = fopen(filename.c_str(), "rb");
    if (file_ == nullptr) throw std::runtime_error("Failed to open bz2 file for reading: " + filename);

    int bz_error = BZ_OK;
    bz_file_ = BZ2_bzReadOpen(&bz_error, file_, /*verbosity=*/0, /*small=*/0, nullptr, 0);
    if (bz_error != BZ_OK) throw_bz2_error("BZ2_bzReadOpen", bz_error);
}

Bz2Source::~Bz2Source() {
    if (bz_file_ != nullptr) {
        int bz_error = BZ_OK;
        BZ2_bzReadClose(&bz_error, (BZFILE *)bz_file_);
    }
    if (file_ != nullptr) fclose(file_);
}

usize Bz2Source::read(u8 *out, usize byte_count) {
    if (stream_ended_ || byte_count == 0) return 0;

    int bz_error = BZ_OK;
    int bytes_read = BZ2_bzRead(&bz_error, (BZFILE *)bz_file_, out, (int)byte_count);

    if (bz_error == BZ_STREAM_END) stream_ended_ = true;
    else if (bz_error != BZ_OK) throw_bz2_error("BZ2_bzRead", bz_error);

    return (usize)bytes_read;
}

#endif // SPIKECOREC_HAVE_BZ2

// ── AsyncSpireWriter ─────────────────────────────────────────────────────────

AsyncSpireWriter::AsyncSpireWriter(unique_ptr<SpireSink> sink, usize max_queued_chunks)
    : sink_(std::move(sink))
    , max_queued_chunks_(max_queued_chunks)
{
    worker_ = thread(&AsyncSpireWriter::run, this);
}

AsyncSpireWriter::~AsyncSpireWriter() {
    if (!closed_) {
        try { close(); } catch (...) { /* destructors must not throw */ }
    }
}

void AsyncSpireWriter::run() {
    // Once the sink has thrown, it may be in an unrecoverable state — for some
    // backends (e.g. libbz2) calling write()/close() again after an error is
    // undefined behavior. So after the first failure we stop touching the sink
    // entirely: we keep draining the queue (discarding chunks) so the producer's
    // blocking write()/close() calls observe the captured error rather than
    // deadlocking, but we never re-enter the failed sink. Resource cleanup is
    // left to the sink's own destructor.
    bool sink_failed = false;

    for (;;) {
        optional<vector<u8>> chunk;
        {
            unique_lock<mutex> lock(mutex_);
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
                    lock_guard<mutex> lock(mutex_);
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
            lock_guard<mutex> lock(mutex_);
            if (!error_) error_ = std::current_exception();
        }
    }
}

void AsyncSpireWriter::rethrow_if_failed() {
    exception_ptr captured;
    {
        lock_guard<mutex> lock(mutex_);
        captured = error_;
    }
    if (captured) std::rethrow_exception(captured);
}

void AsyncSpireWriter::write(vector<u8> chunk) {
    rethrow_if_failed();

    {
        unique_lock<mutex> lock(mutex_);
        // max_queued_chunks_ == 0 means "unbounded" (matching the Python
        // reference's queue.Queue(maxsize=0)) — never block the producer.
        not_full_.wait(lock, [this] {
            return max_queued_chunks_ == 0 || queue_.size() < max_queued_chunks_ || error_;
        });
        if (error_) {
            exception_ptr captured = error_;
            lock.unlock();
            std::rethrow_exception(captured);
        }
        queue_.emplace_back(std::move(chunk));
    }
    not_empty_.notify_one();
}

void AsyncSpireWriter::close() {
    if (closed_) throw std::runtime_error("AsyncSpireWriter::close() called more than once");
    closed_ = true;

    {
        lock_guard<mutex> lock(mutex_);
        queue_.emplace_back(nullopt); // sentinel
    }
    not_empty_.notify_one();

    if (worker_.joinable()) worker_.join();

    rethrow_if_failed();
}

// ── SimulationRecorder ───────────────────────────────────────────────────────

SimulationRecorder::SimulationRecorder(
    const string &filename, s64 neuron_count,
    optional<string> compression, optional<int> compression_level,
    bool async, usize queue_max, usize chunk_bytes
)
    : neuron_count_(neuron_count)
    , chunk_bytes_(chunk_bytes)
{
    if (neuron_count_ < 0 || neuron_count_ > static_cast<s64>(UINT32_MAX))
        throw std::runtime_error("SimulationRecorder: neuron_count " + std::to_string(neuron_count_)
            + " does not fit in the .spire format's 32-bit header");

    auto [resolved_compression, resolved_path] = resolve_spire_compression(filename, compression);
    int level = compression_level.value_or(6);

    unique_ptr<SpireSink> sink = make_spire_sink(resolved_path, resolved_compression, level);
    if (async) async_writer_ = make_unique<AsyncSpireWriter>(std::move(sink), queue_max);
    else       sink_ = std::move(sink);

    buffer_.reserve(chunk_bytes_);
    u32 header = to_big_endian_u32((u32)neuron_count_);
    const u8 *header_bytes = reinterpret_cast<const u8 *>(&header);
    buffer_.insert(buffer_.end(), header_bytes, header_bytes + sizeof(u32));
}

SimulationRecorder::~SimulationRecorder() {
    if (!finished_) {
        try { finish(); } catch (...) { /* destructors must not throw */ }
    }
}

void SimulationRecorder::write_bytes(const u8 *data, usize byte_count) {
    if (async_writer_) async_writer_->write(vector<u8>(data, data + byte_count));
    else               sink_->write(data, byte_count);
}

void SimulationRecorder::flush_buffer() {
    if (buffer_.empty()) return;
    write_bytes(buffer_.data(), buffer_.size());
    buffer_.clear();
}

void SimulationRecorder::record_frame(const f32 *membrane_potentials, s64 frame_length) {
    if (finished_) throw std::runtime_error("SimulationRecorder::record_frame called after finish()");
    if (frame_length != neuron_count_)
        throw std::runtime_error("SimulationRecorder::record_frame: frame length " + std::to_string(frame_length)
            + " does not match the recorder's neuron_count " + std::to_string(neuron_count_));

    const u8 *bytes = reinterpret_cast<const u8 *>(membrane_potentials);
    usize frame_bytes = (usize)neuron_count_ * sizeof(f32);
    buffer_.insert(buffer_.end(), bytes, bytes + frame_bytes);

    if (buffer_.size() >= chunk_bytes_) flush_buffer();
}

void SimulationRecorder::finish() {
    if (finished_) return;
    finished_ = true;

    flush_buffer();
    if (async_writer_) async_writer_->close();
    else               sink_->close();
}

// ── SpireWriter ──────────────────────────────────────────────────────────────

SpireWriter::SpireWriter(const string &filename, s64 neuron_count)
    : file_(filename, ios::binary)
    , neuron_count_(neuron_count)
{
    if (!file_) throw std::runtime_error("Failed to open .spire file for writing: " + filename);
    if (neuron_count_ < 0 || neuron_count_ > static_cast<s64>(UINT32_MAX))
        throw std::runtime_error("SpireWriter: neuron_count " + std::to_string(neuron_count_)
            + " does not fit in the .spire format's 32-bit header");

    u32 header = to_big_endian_u32((u32)neuron_count_);
    file_.write(reinterpret_cast<const char *>(&header), sizeof(u32));
}

void SpireWriter::write_frame(const f32 *membrane_potentials) {
    file_.write(reinterpret_cast<const char *>(membrane_potentials), (streamsize)(neuron_count_ * sizeof(f32)));
}

// ── SpireReader ──────────────────────────────────────────────────────────────

SpireReader::SpireReader(const string &filename)
    : file_(filename, ios::binary)
    , neuron_count_(0)
{
    if (!file_) throw std::runtime_error("Failed to open .spire file for reading: " + filename);

    u32 header;
    file_.read(reinterpret_cast<char *>(&header), sizeof(u32));
    if (file_.gcount() != (streamsize)sizeof(u32))
        throw std::runtime_error(".spire file is too short to contain a neuron_count header: " + filename);

    neuron_count_ = (s64)from_big_endian_u32(header);
}

bool SpireReader::read_frame(f32 *out_buffer) {
    streamsize frame_bytes = (streamsize)(neuron_count_ * sizeof(f32));

    file_.read(reinterpret_cast<char *>(out_buffer), frame_bytes);
    streamsize bytes_read = file_.gcount();

    if (bytes_read == 0) return false; // clean EOF — no partial frame
    if (bytes_read != frame_bytes)
        throw std::runtime_error("Truncated recording: frame ended after " + std::to_string(bytes_read)
            + " of " + std::to_string(frame_bytes) + " expected bytes");

    return true;
}

// ── read_spire_recording ─────────────────────────────────────────────────────

SpireRecording spikecorec::read_spire_recording(const string &filename) {
    SpireCompression compression = resolve_spire_compression(filename, nullopt).first;
    unique_ptr<SpireSource> source = make_spire_source(filename, compression);

    // SpireSource::read may return short reads (e.g. compressed streams hand
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
    if (read_fully(reinterpret_cast<u8 *>(&header), sizeof(u32)) != sizeof(u32))
        throw std::runtime_error(".spire file is too short to contain a neuron_count header: " + filename);

    SpireRecording recording;
    recording.neuron_count = (s64)from_big_endian_u32(header);
    recording.frame_count = 0;

    usize frame_bytes = (usize)recording.neuron_count * sizeof(f32);
    vector<f32> frame((usize)recording.neuron_count);

    for (;;) {
        usize bytes_read = read_fully(reinterpret_cast<u8 *>(frame.data()), frame_bytes);
        if (bytes_read == 0) break; // clean EOF — no partial frame

        if (bytes_read != frame_bytes)
            throw std::runtime_error("Truncated recording: frame ended after " + std::to_string(bytes_read)
                + " of " + std::to_string(frame_bytes) + " expected bytes");

        recording.frames.insert(recording.frames.end(), frame.begin(), frame.end());
        ++recording.frame_count;
    }

    return recording;
}
