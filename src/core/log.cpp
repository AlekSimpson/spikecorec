//
// Created by Alek Simpson on 7/3/26.
//

#include "spikecorec/core/log.h"

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <random>
#include <stdexcept>
#include <vector>

using namespace std;

namespace spikecorec::log {

void throw_invalid_argument(EngineLogger &logger, const String &message) {
    logger.critical(message);
    logger.flush();
    throw std::invalid_argument(message);
}

void throw_runtime_error(EngineLogger &logger, const String &message) {
    logger.critical(message);
    logger.flush();
    throw std::runtime_error(message);
}

SharedPointer<EngineLogger> make_logger(LogLevel level) {
    call_once(logger_already_initialized, [level] {
        constexpr const char *log_pattern = "%Y-%m-%d %H:%M:%S.%e [%^%l%$] %v";

        auto console_sink = make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file_sink = make_shared<spdlog::sinks::rotating_file_sink_mt>(
            LOG_PATH, /*max_size=*/10 * 1024 * 1024, /*max_files=*/5);
        logger_sinks = {console_sink, file_sink};

        global_logger = make_shared<EngineLogger>("SpikeEngine", logger_sinks.begin(), logger_sinks.end());
        global_logger->set_pattern(log_pattern);
        global_logger->set_level(level);
        global_logger->flush_on(level);
    });


    return global_logger;
}

EngineLogger &logger() {
    make_logger();
    return *global_logger;
}





} // namespace spikecorec::log
