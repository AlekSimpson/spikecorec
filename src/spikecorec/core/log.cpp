//
// Created by Alek Simpson on 7/3/26.
//

#include "log.h"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

using namespace std;

namespace spikecorec::log {

// ── levels ────────────────────────────────────────────────────────────────────

const char *level_name(Log_Level level) {
    switch (level) {
        case Log_Level::trace:    return "trace";
        case Log_Level::debug:    return "debug";
        case Log_Level::info:     return "info";
        case Log_Level::warn:     return "warning";
        case Log_Level::error:    return "error";
        case Log_Level::critical: return "critical";
        case Log_Level::off:      return "off";
    }
    return "off";
}

Log_Level level_from_string(const String &name) {
    if (name == "trace") return Log_Level::trace;
    if (name == "debug") return Log_Level::debug;
    if (name == "info") return Log_Level::info;
    if (name == "warn" || name == "warning") return Log_Level::warn;
    if (name == "err" || name == "error") return Log_Level::error;
    if (name == "critical") return Log_Level::critical;
    if (name == "off") return Log_Level::off;
    throw Invalid_Argument("Unknown log level '" + name +
                                "'; expected trace, debug, info, warn, err, critical or off");
}

// ── pattern ───────────────────────────────────────────────────────────────────

namespace {

void append_padded_number(String &text, int value, size_t digit_count) {
    String digits = to_string(value);
    if (digits.size() < digit_count) text.append(digit_count - digits.size(), '0');
    text += digits;
}

Formatted_Line apply_pattern(const String &pattern, Log_Level level, String_View message) {
    const auto now = System_Clock::now();
    const time_t now_seconds = System_Clock::to_time_t(now);
    const int milliseconds = static_cast<int>(
        chrono::duration_cast<Milliseconds>(now.time_since_epoch()).count() % 1000);
    tm local_time{};
    localtime_r(&now_seconds, &local_time);

    Formatted_Line line;
    bool color_begin_seen = false;
    bool color_end_seen = false;
    for (size_t index = 0; index < pattern.size(); index += 1) {
        if (pattern[index] != '%' || index + 1 == pattern.size()) {
            line.text += pattern[index];
            continue;
        }
        index += 1;
        switch (pattern[index]) {
            case 'Y': append_padded_number(line.text, local_time.tm_year + 1900, 4); break;
            case 'm': append_padded_number(line.text, local_time.tm_mon + 1, 2); break;
            case 'd': append_padded_number(line.text, local_time.tm_mday, 2); break;
            case 'H': append_padded_number(line.text, local_time.tm_hour, 2); break;
            case 'M': append_padded_number(line.text, local_time.tm_min, 2); break;
            case 'S': append_padded_number(line.text, local_time.tm_sec, 2); break;
            case 'e': append_padded_number(line.text, milliseconds, 3); break;
            case 'l': line.text += level_name(level); break;
            case 'v': line.text.append(message.data(), message.size()); break;
            case '^': line.color_begin = line.text.size(); color_begin_seen = true; break;
            case '$': line.color_end = line.text.size(); color_end_seen = true; break;
            case '%': line.text += '%'; break;
            default:  line.text += '%'; line.text += pattern[index]; break;
        }
    }
    if (!color_begin_seen || !color_end_seen || line.color_end < line.color_begin) {
        line.color_begin = 0;
        line.color_end = 0;
    }
    return line;
}

} // namespace

// ── sinks ─────────────────────────────────────────────────────────────────────

void Log_Sink::set_pattern(const String &pattern) {
    Lock_Guard<Mutex> lock(mutex_);
    pattern_ = pattern;
}

void Log_Sink::log(Log_Level level, String_View message) {
    Lock_Guard<Mutex> lock(mutex_);
    write(level, apply_pattern(pattern_, level, message));
}

void Log_Sink::flush() {
    Lock_Guard<Mutex> lock(mutex_);
    flush_output();
}

namespace {

const char *terminal_color_code(Log_Level level) {
    switch (level) {
        case Log_Level::trace:    return "\033[37m";
        case Log_Level::debug:    return "\033[36m";
        case Log_Level::info:     return "\033[32m";
        case Log_Level::warn:     return "\033[33m\033[1m";
        case Log_Level::error:    return "\033[31m\033[1m";
        case Log_Level::critical: return "\033[1m\033[41m";
        case Log_Level::off:      return "";
    }
    return "";
}

constexpr const char *terminal_color_reset = "\033[m";

} // namespace

void Console_Sink::write(Log_Level level, const Formatted_Line &line) {
    static const bool output_is_terminal = isatty(fileno(stdout)) != 0;
    if (output_is_terminal && line.color_end > line.color_begin) {
        const char *color_code = terminal_color_code(level);
        fwrite(line.text.data(), 1, line.color_begin, stdout);
        fputs(color_code, stdout);
        fwrite(line.text.data() + line.color_begin, 1, line.color_end - line.color_begin, stdout);
        fputs(terminal_color_reset, stdout);
        fwrite(line.text.data() + line.color_end, 1, line.text.size() - line.color_end, stdout);
    } else {
        fwrite(line.text.data(), 1, line.text.size(), stdout);
    }
    fputc('\n', stdout);
}

void Console_Sink::flush_output() {
    fflush(stdout);
}

Rotating_File_Sink::Rotating_File_Sink(const String &path, size_t maximum_size_bytes,
                                       size_t maximum_file_count)
    : path_(path), maximum_size_bytes_(maximum_size_bytes), maximum_file_count_(maximum_file_count) {
    open_file();
}

String Rotating_File_Sink::rotated_path(size_t rotation_index) const {
    if (rotation_index == 0) return path_;
    const Path base(path_);
    Path rotated = base.parent_path() /
        (base.stem().string() + "." + to_string(rotation_index) + base.extension().string());
    return rotated.string();
}

void Rotating_File_Sink::open_file() {
    const Path parent = Path(path_).parent_path();
    if (!parent.empty()) filesystem::create_directories(parent);

    Error_Code size_error;
    const auto existing_size = filesystem::file_size(path_, size_error);
    current_size_bytes_ = size_error ? 0 : static_cast<size_t>(existing_size);

    file_.open(path_, Stream_Base::out | Stream_Base::app | Stream_Base::binary);
    if (!file_) throw Runtime_Error("Failed to open log file for writing: " + path_);
}

void Rotating_File_Sink::rotate() {
    file_.close();
    Error_Code ignored_error;
    for (size_t rotation_index = maximum_file_count_; rotation_index >= 1; rotation_index -= 1) {
        const String source = rotated_path(rotation_index - 1);
        const String target = rotated_path(rotation_index);
        if (!filesystem::exists(source, ignored_error)) continue;
        filesystem::remove(target, ignored_error);
        filesystem::rename(source, target, ignored_error);
    }
    if (maximum_file_count_ == 0) filesystem::remove(path_, ignored_error);

    file_.open(path_, Stream_Base::out | Stream_Base::trunc | Stream_Base::binary);
    if (!file_) throw Runtime_Error("Failed to reopen log file after rotation: " + path_);
    current_size_bytes_ = 0;
}

void Rotating_File_Sink::write(Log_Level, const Formatted_Line &line) {
    const size_t line_size_bytes = line.text.size() + 1;
    if (current_size_bytes_ > 0 && current_size_bytes_ + line_size_bytes > maximum_size_bytes_) rotate();
    file_.write(line.text.data(), static_cast<streamsize>(line.text.size()));
    file_.put('\n');
    current_size_bytes_ += line_size_bytes;
}

void Rotating_File_Sink::flush_output() {
    file_.flush();
}

void Ostream_Sink::write(Log_Level, const Formatted_Line &line) {
    stream_ << line.text << '\n';
}

void Ostream_Sink::flush_output() {
    stream_.flush();
}

// ── message formatting ────────────────────────────────────────────────────────

namespace {

struct Format_Specification {
    bool alternate = false;
    bool zero_padded = false;
    size_t width = 0;
    int precision = -1;
    char type = '\0';
};

[[noreturn]] void throw_bad_specification(String_View specification, const char *reason) {
    throw Invalid_Argument("Bad format specification '{:" + String(specification) + "}': " + reason);
}

Format_Specification parse_specification(String_View specification) {
    Format_Specification parsed;
    size_t index = 0;
    if (index < specification.size() && specification[index] == '#') {
        parsed.alternate = true;
        index += 1;
    }
    if (index < specification.size() && specification[index] == '0') {
        parsed.zero_padded = true;
        index += 1;
    }
    while (index < specification.size() && isdigit(static_cast<unsigned char>(specification[index]))) {
        parsed.width = parsed.width * 10 + static_cast<size_t>(specification[index] - '0');
        index += 1;
    }
    if (index < specification.size() && specification[index] == '.') {
        index += 1;
        if (index >= specification.size() || !isdigit(static_cast<unsigned char>(specification[index]))) {
            throw_bad_specification(specification, "a precision needs digits after the '.'");
        }
        parsed.precision = 0;
        while (index < specification.size() && isdigit(static_cast<unsigned char>(specification[index]))) {
            parsed.precision = parsed.precision * 10 + (specification[index] - '0');
            index += 1;
        }
    }
    if (index < specification.size()) {
        parsed.type = specification[index];
        index += 1;
    }
    if (index != specification.size()) throw_bad_specification(specification, "unsupported trailing characters");
    return parsed;
}

// Zero padding goes between the sign/prefix and the digits; space padding goes on the left.
String pad_number(const Format_Specification &parsed, const String &sign_and_prefix, const String &digits) {
    const size_t length = sign_and_prefix.size() + digits.size();
    if (length >= parsed.width) return sign_and_prefix + digits;
    const size_t padding = parsed.width - length;
    if (parsed.zero_padded) return sign_and_prefix + String(padding, '0') + digits;
    return String(padding, ' ') + sign_and_prefix + digits;
}

String format_integer(String_View specification, bool negative, unsigned long long magnitude) {
    const Format_Specification parsed = parse_specification(specification);
    String digits;
    String prefix;
    switch (parsed.type) {
        case '\0':
        case 'd': digits = to_string(magnitude); break;
        case 'x':
        case 'X': {
            char buffer[32];
            snprintf(buffer, sizeof(buffer), parsed.type == 'x' ? "%llx" : "%llX", magnitude);
            digits = buffer;
            if (parsed.alternate) prefix = parsed.type == 'x' ? "0x" : "0X";
            break;
        }
        case 'o': {
            char buffer[32];
            snprintf(buffer, sizeof(buffer), "%llo", magnitude);
            digits = buffer;
            if (parsed.alternate && magnitude != 0) prefix = "0";
            break;
        }
        case 'b': {
            if (magnitude == 0) digits = "0";
            for (unsigned long long remaining = magnitude; remaining != 0; remaining >>= 1) {
                digits.insert(digits.begin(), (remaining & 1ULL) ? '1' : '0');
            }
            if (parsed.alternate) prefix = "0b";
            break;
        }
        default: throw_bad_specification(specification, "not an integer type");
    }
    if (parsed.precision >= 0) throw_bad_specification(specification, "integers take no precision");
    return pad_number(parsed, (negative ? "-" : "") + prefix, digits);
}

} // namespace

String format_signed(String_View specification, long long value) {
    const bool negative = value < 0;
    const unsigned long long magnitude =
        negative ? 0ULL - static_cast<unsigned long long>(value) : static_cast<unsigned long long>(value);
    return format_integer(specification, negative, magnitude);
}

String format_unsigned(String_View specification, unsigned long long value) {
    return format_integer(specification, false, value);
}

String format_floating(String_View specification, double value) {
    const Format_Specification parsed = parse_specification(specification);
    String rendered;
    if (parsed.type == '\0' && parsed.precision < 0) {
        // The fewest significant digits that read back as the same double.
        char buffer[64];
        for (int significant_digits = 1; significant_digits <= 17; significant_digits += 1) {
            snprintf(buffer, sizeof(buffer), "%.*g", significant_digits, value);
            if (strtod(buffer, nullptr) == value || isnan(value)) break;
        }
        rendered = buffer;
    } else {
        char type = parsed.type == '\0' ? 'g' : parsed.type;
        if (String("fFeEgG").find(type) == String::npos) {
            throw_bad_specification(specification, "not a floating-point type");
        }
        const String printf_format = "%" + String(parsed.alternate ? "#" : "") + ".*" + type;
        char buffer[512];
        const int precision = parsed.precision >= 0 ? parsed.precision : 6;
        const int length = snprintf(buffer, sizeof(buffer), printf_format.c_str(), precision, value);
        rendered.assign(buffer, static_cast<size_t>(min<int>(length, sizeof(buffer) - 1)));
    }
    String sign;
    if (!rendered.empty() && rendered.front() == '-') {
        sign = "-";
        rendered.erase(rendered.begin());
    }
    return pad_number(parsed, sign, rendered);
}

String format_boolean(String_View specification, bool value) {
    const Format_Specification parsed = parse_specification(specification);
    if (parsed.type == '\0' || parsed.type == 's') return format_text(specification, value ? "true" : "false");
    return format_integer(specification, false, value ? 1 : 0);
}

String format_text(String_View specification, String_View value) {
    const Format_Specification parsed = parse_specification(specification);
    if (parsed.type != '\0' && parsed.type != 's') throw_bad_specification(specification, "not a text type");
    String text(value);
    if (parsed.precision >= 0 && text.size() > static_cast<size_t>(parsed.precision)) {
        text.resize(static_cast<size_t>(parsed.precision));
    }
    if (text.size() < parsed.width) text.append(parsed.width - text.size(), ' ');
    return text;
}

String format_message(String_View format_string, const Vector<Argument_Formatter> &argument_formatters) {
    String result;
    result.reserve(format_string.size() + 16 * argument_formatters.size());
    size_t next_argument = 0;
    for (size_t index = 0; index < format_string.size(); index += 1) {
        const char character = format_string[index];
        if (character == '}') {
            if (index + 1 < format_string.size() && format_string[index + 1] == '}') {
                result += '}';
                index += 1;
                continue;
            }
            throw Invalid_Argument("Unmatched '}' in format string: " + String(format_string));
        }
        if (character != '{') {
            result += character;
            continue;
        }
        if (index + 1 < format_string.size() && format_string[index + 1] == '{') {
            result += '{';
            index += 1;
            continue;
        }
        const size_t closing_brace = format_string.find('}', index);
        if (closing_brace == String_View::npos) {
            throw Invalid_Argument("Unmatched '{' in format string: " + String(format_string));
        }
        const String_View field = format_string.substr(index + 1, closing_brace - index - 1);
        String_View specification;
        if (!field.empty()) {
            if (field.front() != ':') {
                throw Invalid_Argument("Only automatic argument numbering is supported: " +
                                            String(format_string));
            }
            specification = field.substr(1);
        }
        if (next_argument >= argument_formatters.size()) {
            throw Invalid_Argument("More placeholders than arguments in format string: " +
                                        String(format_string));
        }
        result += argument_formatters[next_argument](specification);
        next_argument += 1;
        index = closing_brace;
    }
    return result;
}

// ── logger ────────────────────────────────────────────────────────────────────

void Engine_Logger::write(Log_Level level, String_View message) {
    for (const Sink_Pointer &sink : sinks_) sink->log(level, message);
    if (level >= flush_level_.load()) flush();
}

void Engine_Logger::set_pattern(const String &pattern) {
    for (const Sink_Pointer &sink : sinks_) sink->set_pattern(pattern);
}

void Engine_Logger::flush() {
    for (const Sink_Pointer &sink : sinks_) sink->flush();
}

void throw_invalid_argument(Engine_Logger &logger, const String &message) {
    logger.critical(message);
    logger.flush();
    throw Invalid_Argument(message);
}

void throw_runtime_error(Engine_Logger &logger, const String &message) {
    logger.critical(message);
    logger.flush();
    throw Runtime_Error(message);
}

Shared_Pointer<Engine_Logger> make_logger(Log_Level level) {
    call_once(logger_already_initialized, [level] {
        constexpr const char *log_pattern = "%Y-%m-%d %H:%M:%S.%e [%^%l%$] %v";

        auto console_sink = make_shared<Console_Sink>();
        auto file_sink = make_shared<Rotating_File_Sink>(
            LOG_PATH, /*maximum_size_bytes=*/10 * 1024 * 1024, /*maximum_file_count=*/5);
        logger_sinks = {console_sink, file_sink};

        global_logger = make_shared<Engine_Logger>(logger_sinks, level, level);
        global_logger->set_pattern(log_pattern);
    });

    return global_logger;
}

Engine_Logger &logger() {
    make_logger();
    return *global_logger;
}

} // namespace spikecorec::log
