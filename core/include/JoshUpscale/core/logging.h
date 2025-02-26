// Copyright 2022 Ivanov Viktor

#pragma once

#include <sstream>
#include <utility>

#include "JoshUpscale/core.h"
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

LogSink &getCurrentLogSink() noexcept;

namespace logging {

class LogInterface : public std::ostringstream {
public:
	LogInterface(const char *tag, LogLevel level) : m_Tag{tag}, m_Level{level} {
	}
	LogInterface(const LogInterface &) = delete;
	LogInterface(LogInterface &&s) noexcept
	    : std::ostringstream(std::move(s)), m_Tag{s.m_Tag}, m_Level{s.m_Level} {
	}
	LogInterface &operator=(const LogInterface &) = delete;
	LogInterface &operator=(LogInterface &&) = delete;
	~LogInterface() {
		if (tellp() > 0) {
			getCurrentLogSink()(m_Tag, m_Level, str());
		}
	}

private:
	const char *m_Tag;
	LogLevel m_Level;
};

}  // namespace logging

inline logging::LogInterface log(const char *tag, LogLevel level) {
	return {tag, level};
}

inline logging::LogInterface logInfo(const char *tag) {
	return {tag, LogLevel::INFO};
}

inline logging::LogInterface logWarn(const char *tag) {
	return {tag, LogLevel::WARNING};
}

inline logging::LogInterface logError(const char *tag) {
	return {tag, LogLevel::ERROR};
}

inline void logException(const char *tag) noexcept {
	auto logStream = logError(tag);
	printException(logStream);
}

#define LOG_INFO ::JoshUpscale::core::logInfo(FUNCTION_NAME)
#define LOG_WARN ::JoshUpscale::core::logWarn(FUNCTION_NAME)
#define LOG_ERROR ::JoshUpscale::core::logError(FUNCTION_NAME)
#define LOG_EXCEPTION ::JoshUpscale::core::logException(FUNCTION_NAME)

}  // namespace core

}  // namespace JoshUpscale
