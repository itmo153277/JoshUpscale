// Copyright 2025 Viktor Ivanov

#include "JoshUpscale/core.h"

#ifdef _WIN32
#include <windows.h>
#ifdef ERROR
#undef ERROR
#endif
#include <cuda_d3d11_interop.h>
#include <d3d11.h>
#endif
#include <GL/gl.h>
#include <cuda_gl_interop.h>

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "JoshUpscale/core/cuda.h"
#include "JoshUpscale/core/exception.h"
#include "JoshUpscale/core/logging.h"
#include "JoshUpscale/core/tensor.h"
#include "JoshUpscale/core/tensorrt_backend.h"
#include "JoshUpscale/core/utils.h"

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

#ifdef _WIN32
namespace {

struct D3D11ResourceImage : GraphicsResourceImage {
	explicit D3D11ResourceImage(ID3D11Texture2D *d3d11Texture) {
		::cudaGraphicsResource_t resource;
		D3D11_TEXTURE2D_DESC desc;
		d3d11Texture->GetDesc(&desc);
		assert(desc.Format == DXGI_FORMAT_B8G8R8X8_UNORM ||
		       desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM);
		cuda::cudaCheck(::cudaGraphicsD3D11RegisterResource(
		    &resource, d3d11Texture, ::cudaGraphicsRegisterFlagsNone));
		m_Image.location = DataLocation::GRAPHICS_RESOURCE;
		m_Image.ptr = resource;
		m_Image.width = static_cast<std::size_t>(desc.Width);
		m_Image.height = static_cast<std::size_t>(desc.Height);
	}
	~D3D11ResourceImage() {
		::cudaGraphicsResource_t resource =
		    reinterpret_cast<::cudaGraphicsResource_t>(m_Image.ptr);
		::cudaGraphicsUnregisterResource(resource);
	}
};

}  // namespace

int getD3D11DeviceIndex(ID3D11Device *d3d11Device) {
	int device = -1;
	unsigned int deviceCount = 0;
	cuda::cudaCheck(::cudaD3D11GetDevices(
	    &deviceCount, &device, 1, d3d11Device, ::cudaD3D11DeviceListAll));
	if (deviceCount != 1) {
		throw std::runtime_error("Failed to determine CUDA device");
	}
	return device;
}

GraphicsResourceImage *getD3D11Image(ID3D11Texture2D *d3d11Texture,
    [[maybe_unused]] GraphicsResourceImageType type) {
	return new D3D11ResourceImage(d3d11Texture);
}
#endif

namespace {

struct GLResourceImage : GraphicsResourceImage {
	GLResourceImage(GLuint image, GraphicsResourceImageType type) {
		::cudaGraphicsResource_t resource;
		unsigned int flags = 0;
		switch (type) {
		case GraphicsResourceImageType::INPUT:
			flags = ::cudaGLMapFlagsReadOnly;
			break;
		case GraphicsResourceImageType::OUTPUT:
			flags = ::cudaGLMapFlagsWriteDiscard;
			break;
		default:
			unreachable();
		}
		::glBindTexture(GL_TEXTURE_2D, image);
		auto error = ::glGetError();
		if (error != GL_NO_ERROR) {
			throw std::runtime_error(
			    "Failed to bind texture: " + std::to_string(error));
		}
		GLint width = 0;
		GLint height = 0;
		::glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
		::glGetTexLevelParameteriv(
		    GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
		assert(width > 0 && height > 0);
		::glBindTexture(GL_TEXTURE_2D, 0);
		cuda::cudaCheck(::cudaGraphicsGLRegisterImage(
		    &resource, image, GL_TEXTURE_2D, flags));
		m_Image.location = DataLocation::GRAPHICS_RESOURCE;
		m_Image.ptr = resource;
		m_Image.width = static_cast<std::size_t>(width);
		m_Image.height = static_cast<std::size_t>(height);
	}
	~GLResourceImage() {
		::cudaGraphicsResource_t resource =
		    reinterpret_cast<::cudaGraphicsResource_t>(m_Image.ptr);
		::cudaGraphicsUnregisterResource(resource);
	}
};

}  // namespace

GraphicsResourceImage *getGLImage(
    std::uint32_t image, GraphicsResourceImageType type) {
	return new GLResourceImage(image, type);
}

int getGLDeviceIndex() {
	int device = -1;
	unsigned int deviceCount = 0;
	cuda::cudaCheck(
	    ::cudaGLGetDevices(&deviceCount, &device, 1, ::cudaGLDeviceListAll));
	if (deviceCount != 1) {
		throw std::runtime_error("Failed to determine CUDA device");
	}
	return device;
}

namespace {

struct TensorRTRuntime : Runtime {
	explicit TensorRTRuntime(
	    int deviceId, const std::filesystem::path &modelPath) {
		std::vector<std::byte> engine;
		{
			std::ifstream inputFile(modelPath,
			    std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
			inputFile.exceptions(
			    std::ifstream::badbit | std::ifstream::failbit);
			auto size = static_cast<std::size_t>(inputFile.tellg());
			engine.resize(size);
			inputFile.seekg(0);
			inputFile.read(reinterpret_cast<char *>(engine.data()),
			    static_cast<std::streamsize>(size));
		}
		cuda::DeviceContext cudaCtx(deviceId);
		m_Backend.emplace(engine);
		auto frameSize = m_Backend->getFrameSize();
		m_InputWidth = frameSize.inputWidth;
		m_InputHeight = frameSize.inputHeight;
		m_OutputWidth = frameSize.outputWidth;
		m_OutputHeight = frameSize.outputHeight;
	}

	void processImage(
	    const Image &inputImage, const Image &outputImage) override {
		assert(inputImage.width == m_InputWidth &&
		       inputImage.height == m_InputHeight &&
		       outputImage.width == m_OutputWidth &&
		       outputImage.height == m_OutputHeight);
		GenericTensor inputTensor{inputImage};
		GenericTensor outputTensor{outputImage};
		if (!m_Backend.has_value()) {
			unreachable();
		}
		m_Backend->process(inputTensor, outputTensor);
	}

private:
	std::optional<TensorRTBackend> m_Backend;
};

}  // namespace

Runtime *createRuntime(int deviceId, const std::filesystem::path &modelPath) {
	return new TensorRTRuntime(deviceId, modelPath);
}

}  // namespace core

}  // namespace JoshUpscale
