// Copyright 2022 Ivanov Viktor

#pragma once

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "JoshUpscale/core/exception.h"

#ifndef FUNCTION_NAME
#if defined(__GNUC__)
#define FUNCTION_NAME __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define FUNCTION_NAME __FUNCSIG__
#else
#define FUNCTION_NAME __func__
#endif
#endif

namespace JoshUpscale {

namespace core {

namespace logging {

using LogTimestamp = std::chrono::system_clock::time_point;

class LogInterface : public std::ostringstream {
public:
	LogInterface(const char *tag, const char *level)
	    : m_Timestamp{LogTimestamp::clock::now()}, m_Tag{tag}, m_Level{level} {
	}
	LogInterface(const LogInterface &) = delete;
	LogInterface(LogInterface &&s) noexcept
	    : std::ostringstream(std::move(s))
	    , m_Timestamp{s.m_Timestamp}
	    , m_Tag{s.m_Tag}
	    , m_Level{s.m_Level} {
	}
	LogInterface &operator=(const LogInterface &) = delete;
	LogInterface &operator=(LogInterface &&) = delete;
	~LogInterface() {
		if (tellp() > 0) {
			std::clog << formatMessage() << std::endl;
		}
	}

private:
	LogTimestamp m_Timestamp;
	const char *m_Tag;
	const char *m_Level;

	std::string formatMessage();
};

}  // namespace logging

inline logging::LogInterface log(const char *tag, const char *level) {
	return {tag, level};
}

inline logging::LogInterface logInfo(const char *tag) {
	return {tag, "INFO"};
}

inline logging::LogInterface logWarn(const char *tag) {
	return {tag, "WARNING"};
}

inline logging::LogInterface logError(const char *tag) {
	return {tag, "ERROR"};
}

inline void logException(const char *tag) noexcept {
	try {
		logError(tag) << getExceptionString();
	} catch (...) {
	}
}

#define LOG_INFO ::JoshUpscale::core::logInfo(FUNCTION_NAME)
#define LOG_WARN ::JoshUpscale::core::logWarn(FUNCTION_NAME)
#define LOG_ERROR ::JoshUpscale::core::logError(FUNCTION_NAME)
#define LOG_EXCEPTION ::JoshUpscale::core::logException(FUNCTION_NAME)

}  // namespace core

}  // namespace JoshUpscale
