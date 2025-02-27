// Copyright 2022 Ivanov Viktor

#pragma once

#include <NvInfer.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>

#include "JoshUpscale/core/logging.h"

#define TENSORRT_VERSION \
	(NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH)

namespace JoshUpscale {

namespace core {

namespace trt {

struct TrtException : std::runtime_error {
	TrtException() : TrtException("TensorRT general failure") {
	}
	explicit TrtException(const std::string &msg) : TrtException(msg.c_str()) {
	}
	explicit TrtException(const char *msg) : std::runtime_error(msg) {
	}
};

template <typename T>
T *throwIfNull(T *val) {
	if (val == nullptr) {
		throw TrtException();
	}
	return val;
}

template <typename T>
struct TrtPtr : std::unique_ptr<T> {
	using unique_ptr = std::unique_ptr<T>;
	using unique_ptr::get;

	TrtPtr(std::nullptr_t) : unique_ptr(nullptr) {  // NOLINT(runtime/explicit)
	}
	explicit TrtPtr(T *obj) : unique_ptr(throwIfNull(obj)) {
	}

	// Non-copyable, movable
	TrtPtr(const TrtPtr &) = delete;
	TrtPtr(TrtPtr &&) noexcept = default;
	TrtPtr &operator=(const TrtPtr &) = delete;
	TrtPtr &operator=(TrtPtr &&) noexcept = default;

	operator T *() const {
		return get();
	}
};

class ErrorRecorder : public nvinfer1::IErrorRecorder {
public:
	using ErrorCode = nvinfer1::ErrorCode;

	std::int32_t getNbErrors() const noexcept override {
		try {
			std::shared_lock lock{m_Mutex};
			if (m_HasValue) {
				return 1;
			}
		} catch (...) {
			printException();
		}
		return 0;
	}

	ErrorCode getErrorCode(std::int32_t errorIdx) const noexcept override {
		try {
			std::shared_lock lock{m_Mutex};
			if (m_HasValue && errorIdx == 0) {
				return m_ErrorCode;
			}
		} catch (...) {
			printException();
		}
		return ErrorCode::kSUCCESS;
	}

	ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept override {
		try {
			std::shared_lock lock{m_Mutex};
			if (m_HasValue && errorIdx == 0) {
				return m_ErrorDesc.c_str();
			}
		} catch (...) {
			printException();
		}
		return "";
	}

	bool hasOverflowed() const noexcept override {
		try {
			std::shared_lock lock{m_Mutex};
			return m_HasOverflowed;
		} catch (...) {
			printException();
		}
		return false;
	}

	void clear() noexcept override {
		try {
			std::unique_lock guard{m_Mutex};
			m_HasValue = false;
			m_HasOverflowed = false;
		} catch (...) {
			printException();
		}
	}

	bool reportError(ErrorCode val, ErrorDesc desc) noexcept override {
		try {
			std::unique_lock guard{m_Mutex};
			if (m_HasValue) {
				m_HasOverflowed = true;
			} else {
				m_HasValue = true;
				m_ErrorCode = val;
				m_ErrorDesc = desc;
			}
		} catch (...) {
			printException();
		}
		return true;
	}

	RefCount incRefCount() noexcept override {
		return ++m_RefCount;
	}
	RefCount decRefCount() noexcept override {
		return --m_RefCount;
	}

	[[noreturn]] void rethrowException() {
		try {
			throw;
		} catch (TrtException &e) {
			try {
				std::unique_lock guard{m_Mutex};
				if (m_HasValue) {
					throw TrtException(m_ErrorDesc);
				}
			} catch (TrtException &) {
				throw;
			} catch (...) {
				printException();
			}
			throw e;
		}
	}

private:
	std::atomic<RefCount> m_RefCount{0};
	mutable std::shared_mutex m_Mutex;
	bool m_HasValue = false;
	bool m_HasOverflowed = false;
	ErrorCode m_ErrorCode{ErrorCode::kSUCCESS};
	std::string m_ErrorDesc;

	static void printException() noexcept {
		logException("ErrorRecorder");
	}
};

class Logger : public nvinfer1::ILogger {
public:
	void log(Severity severity, const char *msg) noexcept override {
		static const char kTrtTag[] = "TensorRT";
		try {
			switch (severity) {
			case Severity::kERROR:
				[[fallthrough]];
			case Severity::kINTERNAL_ERROR:
				logError(kTrtTag) << msg;
				break;
			case Severity::kINFO:
				logInfo(kTrtTag) << msg;
				break;
			case Severity::kWARNING:
				logWarn(kTrtTag) << msg;
				break;
			default:
				break;
			}
		} catch (...) {  // NOLINT
		}
	}
};

}  // namespace trt

}  // namespace core

}  // namespace JoshUpscale
