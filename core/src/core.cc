// Copyright 2025 Viktor Ivanov

#include "JoshUpscale/core.h"

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
		m_InputWidth = m_Backend->getInputWidth();
		m_InputHeight = m_Backend->getInputHeight();
		m_OutputWidth = m_Backend->getOutputWidth();
		m_outputHeight = m_Backend->getOutputHeight();
	}

	void processImage(
	    const Image &inputImage, const Image &outputImage) override {
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
