// Copyright 2025 Viktor Ivanov

#include "JoshUpscale/obs/logging.h"

#include <JoshUpscale/core.h>
#include <obs-module.h>

#include <cstdarg>
#include <string>
#include <string_view>

namespace JoshUpscale {

namespace obs {

void log(core::LogLevel level, std::string_view format, ...) {
	int obsLogLevel;
	switch (level) {
	case core::LogLevel::ERROR:
		obsLogLevel = LOG_ERROR;
		break;
	case core::LogLevel::WARNING:
		obsLogLevel = LOG_WARNING;
		break;
	case core::LogLevel::INFO:
		[[fallthrough]];
	default:
		obsLogLevel = LOG_INFO;
		break;
	}
	std::va_list args;
	static constexpr std::string_view kTag = "[obs-joshupscale]: ";
	std::string newFormat;
	newFormat.reserve(kTag.size() + format.size());
	newFormat += kTag;
	newFormat += format;
	va_start(args, format);
	::blogva(obsLogLevel, newFormat.c_str(), args);
	va_end(args);
}

void logException() noexcept {
	try {
		log(core::LogLevel::ERROR, "Exception: %s",
		    core::getExceptionString().c_str());
	} catch (...) {  // NOLINT
	}
}

}  // namespace obs

}  // namespace JoshUpscale
