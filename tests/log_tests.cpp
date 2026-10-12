#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <gtest/gtest.h>

#include "log.h"
#include "support/test_support.h"

using namespace std;
using namespace spikecorec;
using namespace spikecorec::log;
using namespace spikecorec::test_support;

// ── message formatting ──────────────────────────────────────────────────────────

TEST(Log_Format, replaces_each_placeholder_with_the_next_argument) {
    EXPECT_EQ(format("{} + {} = {}", 1, 2.5, "three"), "1 + 2.5 = three");
    EXPECT_EQ(format("{}", true), "true");
    EXPECT_EQ(format("{}", String("text")), "text");
}

TEST(Log_Format, applies_the_specifications_the_engine_uses) {
    EXPECT_EQ(format("{:#x}", 255), "0xff");
    EXPECT_EQ(format("{:6d}|", 42), "    42|");
    EXPECT_EQ(format("{:2d}", 123), "123");
    EXPECT_EQ(format("{:.2f}", 3.14159), "3.14");
    EXPECT_EQ(format("{:.1f}%", 99.95), "100.0%");
    EXPECT_EQ(format("{:.3e}", 0.000123456), "1.235e-04");
    EXPECT_EQ(format("{:.0e}", 1e-6), "1e-06");
    EXPECT_EQ(format("{:05d}", -42), "-0042");
}

TEST(Log_Format, prints_the_extreme_integers) {
    EXPECT_EQ(format("{}", Numeric_Limits<long long>::min()), "-9223372036854775808");
    EXPECT_EQ(format("{}", Numeric_Limits<unsigned long long>::max()), "18446744073709551615");
}

TEST(Log_Format, treats_doubled_braces_as_literal_braces) {
    EXPECT_EQ(format("{{{}}}", 7), "{7}");
}

TEST(Log_Format, rejects_more_placeholders_than_arguments_and_malformed_specifications) {
    EXPECT_THROW(format("{} {}", 1), Invalid_Argument);
    EXPECT_THROW(format("{:q}", 1), Invalid_Argument);
    EXPECT_THROW(format("{", 1), Invalid_Argument);
    EXPECT_THROW(format("}", 1), Invalid_Argument);
}

// ── the logger and its sinks ────────────────────────────────────────────────────

TEST(Engine_Logger, writes_a_message_through_the_pattern_of_each_sink) {
    Output_String_Stream text;
    auto sink = make_shared<Ostream_Sink>(text);
    sink->set_pattern("[%l] %v");
    Engine_Logger logger({sink}, Log_Level::info, Log_Level::info);

    logger.warn("rank {} of {}", 3, 8);

    EXPECT_EQ(text.str(), "[warning] rank 3 of 8\n");
}

TEST(Engine_Logger, drops_messages_below_its_level) {
    Output_String_Stream text;
    auto sink = make_shared<Ostream_Sink>(text);
    sink->set_pattern("%v");
    Engine_Logger logger({sink}, Log_Level::warn, Log_Level::warn);

    logger.info("hidden");
    logger.error("shown");
    logger.set_level(Log_Level::off);
    logger.critical("also hidden");

    EXPECT_EQ(text.str(), "shown\n");
}

TEST(Engine_Logger, writes_a_message_without_arguments_as_it_is) {
    Output_String_Stream text;
    auto sink = make_shared<Ostream_Sink>(text);
    sink->set_pattern("%v");
    Engine_Logger logger({sink}, Log_Level::info, Log_Level::info);

    logger.info("a {} in a message with no arguments");

    EXPECT_EQ(text.str(), "a {} in a message with no arguments\n");
}

TEST(Engine_Logger, stamps_the_default_pattern_with_the_time_and_level) {
    Output_String_Stream text;
    auto sink = make_shared<Ostream_Sink>(text);
    Engine_Logger logger({sink}, Log_Level::info, Log_Level::info);

    logger.info("hello");

    // 2026-10-10 15:37:04.084 [info] hello
    const String line = text.str();
    ASSERT_GT(line.size(), 30u);
    EXPECT_EQ(line[4], '-');
    EXPECT_EQ(line[10], ' ');
    EXPECT_EQ(line[13], ':');
    EXPECT_EQ(line[19], '.');
    EXPECT_EQ(line.substr(23), " [info] hello\n");
}

TEST(Log_Level, reads_the_names_the_python_binding_accepts) {
    EXPECT_EQ(level_from_string("warn"), Log_Level::warn);
    EXPECT_EQ(level_from_string("err"), Log_Level::error);
    EXPECT_EQ(level_from_string("off"), Log_Level::off);
    EXPECT_THROW(level_from_string("loud"), Invalid_Argument);
}

TEST(Rotating_File_Sink, moves_a_full_file_aside_and_keeps_only_the_newest_rotations) {
    Temporary_Directory directory;
    const Path path = Path(directory.path()) / "nested" / "run.log";
    {
        auto sink = make_shared<Rotating_File_Sink>(path.string(), /*maximum_size_bytes=*/20, /*maximum_file_count=*/2);
        sink->set_pattern("%v");
        for (const char *message : {"aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd"}) {
            sink->log(Log_Level::info, message);
        }
        sink->flush();
    }

    auto read_file = [](const Path &file) {
        Input_File_Stream stream(file);
        String_Stream contents;
        contents << stream.rdbuf();
        return contents.str();
    };
    EXPECT_EQ(read_file(path), "dddddddddd\n");
    EXPECT_EQ(read_file(path.parent_path() / "run.1.log"), "cccccccccc\n");
    EXPECT_EQ(read_file(path.parent_path() / "run.2.log"), "bbbbbbbbbb\n");
    EXPECT_FALSE(filesystem::exists(path.parent_path() / "run.3.log"));
}
