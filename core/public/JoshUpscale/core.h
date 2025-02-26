// Copyright 2022 Ivanov Viktor

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "JoshUpscale/core/export.h"

namespace JoshUpscale {

namespace core {

enum class LogLevel : std::uint8_t { INFO, WARNING, ERROR };

struct LogSink {
	virtual void operator()(const char *tag, LogLevel logLevel,
	    const std::string &message) noexcept = 0;
};

JOSHUPSCALE_EXPORT void setLogSink(LogSink *sink);

enum class DataLocation : std::uint8_t { CPU, CUDA };

struct Image {
	void *ptr;
	DataLocation location;
	std::ptrdiff_t stride;
	std::size_t width;
	std::size_t height;
};

struct Runtime {
	virtual ~Runtime() {
	}

	virtual void processImage(
	    const Image &inputImage, const Image &outputImage) = 0;

	std::size_t getInputWidth() const {
		return m_InputWidth;
	}
	std::size_t getInputHeight() const {
		return m_InputHeight;
	}
	std::size_t getOuputWWidth() const {
		return m_OutputWidth;
	}
	std::size_t getOuputWHeight() const {
		return m_outputHeight;
	}

protected:
	std::size_t m_InputWidth;
	std::size_t m_InputHeight;
	std::size_t m_OutputWidth;
	std::size_t m_outputHeight;
};

JOSHUPSCALE_EXPORT Runtime *createRuntime(
    int deviceId, const std::filesystem::path &modelPath);

JOSHUPSCALE_EXPORT std::string getExceptionString();

}  // namespace core

}  // namespace JoshUpscale
