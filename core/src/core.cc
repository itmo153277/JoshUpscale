// Copyright 2025 Viktor Ivanov

#include "JoshUpscale/core.h"

#include <sstream>
#include <string>

#include "JoshUpscale/core/exception.h"
#include "JoshUpscale/core/logging.h"

namespace JoshUpscale {

namespace core {

std::string getExceptionString() {
	std::ostringstream ss;
	printException(ss);
	return ss.str();
}

void setLogSink(LogSink *logSink) {
	logging::currentLogSink = logSink;
}

}  // namespace core

}  // namespace JoshUpscale
