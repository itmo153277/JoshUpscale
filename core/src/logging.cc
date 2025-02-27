// Copyright 2022 Ivanov Viktor

#include "JoshUpscale/core/logging.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

#include "JoshUpscale/core.h"
#include "JoshUpscale/core/utils.h"

namespace JoshUpscale {

namespace core {

namespace logging {

namespace {

using LogTimestamp = std::chrono::system_clock::time_point;

std::ostream &operator<<(std::ostream &os, const LogTimestamp &ts) {
	auto timeStruct = LogTimestamp::clock::to_time_t(ts);
	auto flags = os.flags();
	os << std::put_time(std::localtime(&timeStruct), "%Y-%m-%d %H:%M:%S") << '.'
	   << std::setw(3) << std::setfill('0')
	   << (std::chrono::duration_cast<std::chrono::milliseconds>(
	           ts.time_since_epoch()) %
	          1000)
	          .count();
	os.flags(flags);
	return os;
}

const char *getLogLevelString(LogLevel level) {
	switch (level) {
	case LogLevel::INFO:
		return "INFO";
	case LogLevel::WARNING:
		return "WARNING";
	case LogLevel::ERROR:
		return "ERROR";
	default:
		unreachable();
	}
}

struct ConsoleLogSink : LogSink {
	void operator()(const char *tag, LogLevel logLevel,
	    const std::string &message) override {
		std::clog << LogTimestamp::clock::now() << ' '
		          << getLogLevelString(logLevel) << " [" << tag << "] "
		          << message << '\n';
	}
};

}  // namespace

ConsoleLogSink consoleLogSink;
LogSink *currentLogSink = &consoleLogSink;

}  // namespace logging

}  // namespace core

}  // namespace JoshUpscale
