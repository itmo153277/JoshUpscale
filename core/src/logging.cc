// Copyright 2022 Ivanov Viktor

#include "JoshUpscale/core/logging.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace JoshUpscale {

namespace core {

namespace logging {

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

std::string LogInterface::formatMessage() {
	std::ostringstream ss;
	ss << m_Timestamp << ' ' << m_Level << " [" << m_Tag << "] " << str();
	return ss.str();
}

}  // namespace logging

}  // namespace core

}  // namespace JoshUpscale
