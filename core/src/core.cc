// Copyright 2025 Viktor Ivanov

#include "JoshUpscale/core.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "JoshUpscale/core/cuda.h"
#include "JoshUpscale/core/exception.h"
#include "JoshUpscale/core/logging.h"
#include "JoshUpscale/core/tensor.h"
#include "JoshUpscale/core/tensorrt_backend.h"

#ifdef _WIN32
#include <cuda_d3d11_interop.h>
#include <d3d11.h>
#endif

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

GraphicsResourceImage *getD3D11Image(ID3D11Texture2D *d3d11Texture,
    [[maybe_unused]] GraphicsResourceImageType type) {
	return new D3D11ResourceImage(d3d11Texture);
}
#endif

namespace {

struct TensorRTRuntime : Runtime {
	explicit TensorRTRuntime(
	    int deviceId, const std::filesystem::path &modelPath) {
		cuda::DeviceContext cudaCtx(deviceId);
		std::ifstream inputFile(modelPath,
		    std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
		inputFile.exceptions(std::ifstream::badbit | std::ifstream::failbit);
		auto size = static_cast<std::size_t>(inputFile.tellg());
		inputFile.seekg(0);
		std::vector<std::byte> engine{size};
		inputFile.read(reinterpret_cast<char *>(engine.data()),
		    static_cast<std::streamsize>(size));
		m_Backend = std::make_unique<TensorRTBackend>(engine);
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
		m_Backend->process(inputTensor, outputTensor);
	}

private:
	std::unique_ptr<TensorRTBackend> m_Backend = nullptr;
};

}  // namespace

Runtime *createRuntime(int deviceId, const std::filesystem::path &modelPath) {
	return new TensorRTRuntime(deviceId, modelPath);
}

}  // namespace core

}  // namespace JoshUpscale
