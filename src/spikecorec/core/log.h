//
// Created by Alek Simpson on 7/3/26.
//
#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "types.h"

using namespace std;

namespace spikecorec::log {

    enum class Log_Level {
        trace,
        debug,
        info,
        warn,
        error,
        critical,
        off
    };

    // The level's name as a pattern prints it: "trace", "debug", "info", "warning", "error",
    // "critical" or "off".
    [[nodiscard]] const char *level_name(Log_Level level);

    // Accepts "trace", "debug", "info", "warn", "warning", "err", "error", "critical" and "off".
    // Throws std::invalid_argument for anything else.
    [[nodiscard]] Log_Level level_from_string(const String &name);

    // One message after its pattern has been applied. When the pattern has a %^ ... %$ range,
    // [color_begin, color_end) is that range; otherwise both are zero.
    struct Formatted_Line {
        String text;
        size_t color_begin = 0;
        size_t color_end = 0;
    };

    // Where messages end up. A sink owns a pattern and turns each message into a line with it:
    //   %Y %m %d %H %M %S   year, month, day, hour, minute, second of the local time
    //   %e                  milliseconds, three digits
    //   %l                  the level's name
    //   %v                  the message
    //   %^ and %$           begin and end of the span a colored sink colors
    //   %%                  a literal percent sign
    // Every other character is copied as it is. log, flush and set_pattern may be called from any
    // thread; subclasses write inside the sink's lock and need no locking of their own.
    class Log_Sink {
    public:
        virtual ~Log_Sink() = default;

        void set_pattern(const String &pattern);
        void log(Log_Level level, String_View message);
        void flush();

    protected:
        virtual void write(Log_Level level, const Formatted_Line &line) = 0;
        virtual void flush_output() = 0;

    private:
        Mutex mutex_;
        String pattern_ = "%Y-%m-%d %H:%M:%S.%e [%^%l%$] %v";
    };

    using Sink_Pointer = Shared_Pointer<Log_Sink>;

    // Standard output, with the %^ ... %$ span in a color for the level when output is a terminal.
    class Console_Sink : public Log_Sink {
    protected:
        void write(Log_Level level, const Formatted_Line &line) override;
        void flush_output() override;
    };

    // A file that moves to <stem>.1<extension> when the next line would take it past
    // maximum_size_bytes, pushing <stem>.1 to <stem>.2 and so on. At most maximum_file_count
    // rotated files are kept. Parent directories are created as needed.
    class Rotating_File_Sink : public Log_Sink {
    public:
        Rotating_File_Sink(const String &path, size_t maximum_size_bytes, size_t maximum_file_count);

    protected:
        void write(Log_Level level, const Formatted_Line &line) override;
        void flush_output() override;

    private:
        [[nodiscard]] String rotated_path(size_t rotation_index) const;
        void open_file();
        void rotate();

        String path_;
        size_t maximum_size_bytes_;
        size_t maximum_file_count_;
        Output_File_Stream file_;
        size_t current_size_bytes_ = 0;
    };

    // Writes lines to a stream the caller keeps alive for as long as the sink is in use.
    class Ostream_Sink : public Log_Sink {
    public:
        explicit Ostream_Sink(Output_Stream &stream) : stream_(stream) {}

    protected:
        void write(Log_Level level, const Formatted_Line &line) override;
        void flush_output() override;

    private:
        Output_Stream &stream_;
    };

    // ── message formatting ────────────────────────────────────────────────────────
    // "{}" takes the next argument; "{:spec}" takes it with a specification of the form
    // [#][0][width][.precision][type], where type is d, x, X, o or b for integers, f, F, e, E, g or
    // G for floating point and s for text. "{{" and "}}" are literal braces. Throws
    // std::invalid_argument for a malformed specification or for more placeholders than arguments.

    using Argument_Formatter = Function<String(String_View specification)>;

    [[nodiscard]] String format_message(String_View format_string,
                                        const Vector<Argument_Formatter> &argument_formatters);

    [[nodiscard]] String format_signed(String_View specification, long long value);
    [[nodiscard]] String format_unsigned(String_View specification, unsigned long long value);
    [[nodiscard]] String format_floating(String_View specification, double value);
    [[nodiscard]] String format_boolean(String_View specification, bool value);
    [[nodiscard]] String format_text(String_View specification, String_View value);

    template <typename Argument>
    String format_argument(String_View specification, const Argument &argument) {
        if constexpr (is_same_v<Argument, bool>) {
            return format_boolean(specification, argument);
        } else if constexpr (is_integral_v<Argument> && !is_same_v<Argument, char>) {
            if constexpr (is_signed_v<Argument>) {
                return format_signed(specification, static_cast<long long>(argument));
            } else {
                return format_unsigned(specification, static_cast<unsigned long long>(argument));
            }
        } else if constexpr (is_floating_point_v<Argument>) {
            return format_floating(specification, static_cast<double>(argument));
        } else if constexpr (is_convertible_v<const Argument &, String_View>) {
            return format_text(specification, String_View(argument));
        } else {
            Output_String_Stream stream;
            stream << argument;
            return format_text(specification, stream.str());
        }
    }

    template <typename... Arguments>
    [[nodiscard]] String format(String_View format_string, const Arguments &...arguments) {
        Vector<Argument_Formatter> argument_formatters = {
            [&arguments](String_View specification) { return format_argument(specification, arguments); }...};
        return format_message(format_string, argument_formatters);
    }

    // Sends each message at or above its level to every sink. A message with no arguments is
    // written as it is, so braces in it are not read as placeholders.
    class Engine_Logger {
    public:
        Engine_Logger(Vector<Sink_Pointer> sinks, Log_Level level, Log_Level flush_level)
            : sinks_(std::move(sinks)), level_(level), flush_level_(flush_level) {}

        template <typename... Arguments>
        void trace(String_View format_string, const Arguments &...arguments) {
            log(Log_Level::trace, format_string, arguments...);
        }

        template <typename... Arguments>
        void debug(String_View format_string, const Arguments &...arguments) {
            log(Log_Level::debug, format_string, arguments...);
        }

        template <typename... Arguments>
        void info(String_View format_string, const Arguments &...arguments) {
            log(Log_Level::info, format_string, arguments...);
        }

        template <typename... Arguments>
        void warn(String_View format_string, const Arguments &...arguments) {
            log(Log_Level::warn, format_string, arguments...);
        }

        template <typename... Arguments>
        void error(String_View format_string, const Arguments &...arguments) {
            log(Log_Level::error, format_string, arguments...);
        }

        template <typename... Arguments>
        void critical(String_View format_string, const Arguments &...arguments) {
            log(Log_Level::critical, format_string, arguments...);
        }

        template <typename... Arguments>
        void log(Log_Level level, String_View format_string, const Arguments &...arguments) {
            if (!should_log(level)) return;
            if constexpr (sizeof...(arguments) == 0) {
                write(level, format_string);
            } else {
                write(level, format(format_string, arguments...));
            }
        }

        [[nodiscard]] bool should_log(Log_Level level) const { return level >= level_.load(); }

        void set_level(Log_Level level) { level_.store(level); }
        [[nodiscard]] Log_Level level() const { return level_.load(); }

        // Messages at or above this level are flushed to every sink as soon as they are written.
        void set_flush_level(Log_Level level) { flush_level_.store(level); }

        void set_pattern(const String &pattern);
        void flush();

        // The sinks messages go to. Change them only while no other thread is logging.
        [[nodiscard]] Vector<Sink_Pointer> &sinks() { return sinks_; }

    private:
        void write(Log_Level level, String_View message);

        Vector<Sink_Pointer> sinks_;
        Atomic<Log_Level> level_;
        Atomic<Log_Level> flush_level_;
    };

    const String LOG_PATH = "logs/spikecorec.log";

    inline Once_Flag logger_already_initialized;
    inline Vector<Sink_Pointer> logger_sinks;
    inline Shared_Pointer<Engine_Logger> global_logger;

    Engine_Logger &logger();

    Shared_Pointer<Engine_Logger> make_logger(Log_Level level = Log_Level::info);

    [[noreturn]] void throw_runtime_error(Engine_Logger &logger, const String &message);
    [[noreturn]] void throw_invalid_argument(Engine_Logger &logger, const String &message);

} // namespace spikecorec::log
