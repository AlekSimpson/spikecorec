//
// Created by Alek Simpson on 7/3/26.
//
#pragma once

#include <spdlog/spdlog.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace std;

namespace spikecorec::log {
    using LogLevel = spdlog::level::level_enum;
    using EngineLogger = spdlog::logger;
    using SinkPointer = spdlog::sink_ptr;
    using String = std::string;

    template <typename T>
    using SharedPointer = shared_ptr<T>;

    template <typename T>
    using Vector = vector<T>;

    using OnceFlag = once_flag;

    const String LOG_PATH = "logs/spikecorec.log";

    inline OnceFlag logger_already_initialized;
    inline Vector<SinkPointer> logger_sinks;
    inline SharedPointer<EngineLogger> global_logger;

    EngineLogger &logger();

    SharedPointer<EngineLogger> make_logger(LogLevel level = LogLevel::info);

    [[noreturn]] void throw_runtime_error(EngineLogger &logger, const String &message);
    [[noreturn]] void throw_invalid_argument(EngineLogger &logger, const String &message);

} // namespace spikecorec::log
