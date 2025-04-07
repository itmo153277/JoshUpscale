// Copyright 2025 Ivanov Viktor

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "JoshUpscale/core/export.h"

#ifdef _WIN32
struct ID3D11Texture2D;
struct ID3D11Device;
#endif

namespace JoshUpscale {

namespace core {

enum class LogLevel : std::uint8_t { INFO, WARNING, ERROR };

struct LogSink {
	virtual void operator()(
	    const char *tag, LogLevel logLevel, const std::string &message) = 0;
};

JOSHUPSCALE_EXPORT void setLogSink(LogSink *sink);

enum class DataLocation : std::uint8_t { CPU, CUDA, GRAPHICS_RESOURCE };

struct Image {
	void *ptr;
	DataLocation location;
	std::ptrdiff_t stride;
	std::size_t width;
	std::size_t height;
};

enum class GraphicsResourceImageType : std::uint8_t { INPUT, OUTPUT };

struct GraphicsResourceImage {
	virtual ~GraphicsResourceImage() {
	}

	Image getImage() const {
		return m_Image;
	}

protected:
	Image m_Image = {};
};

#ifdef _WIN32
JOSHUPSCALE_EXPORT int getD3D11DeviceIndex(ID3D11Device *d3d11Device);
JOSHUPSCALE_EXPORT GraphicsResourceImage *getD3D11Image(
    ID3D11Texture2D *d3d11Texture, GraphicsResourceImageType type);
#endif

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
	std::size_t getOutputWidth() const {
		return m_OutputWidth;
	}
	std::size_t getOutputHeight() const {
		return m_OutputHeight;
	}

protected:
	std::size_t m_InputWidth = 0;
	std::size_t m_InputHeight = 0;
	std::size_t m_OutputWidth = 0;
	std::size_t m_OutputHeight = 0;
};

JOSHUPSCALE_EXPORT Runtime *createRuntime(
    int deviceId, const std::filesystem::path &modelPath);

JOSHUPSCALE_EXPORT std::string getExceptionString();

}  // namespace core

}  // namespace JoshUpscale
