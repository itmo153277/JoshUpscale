// Copyright 2022 Ivanov Viktor

#include "JoshUpscale/core.h"

#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "JoshUpscale/core/cuda.h"
#include "JoshUpscale/core/tensor.h"
#include "JoshUpscale/core/tensorrt_backend.h"

namespace JoshUpscale {

namespace core {

struct TensorRTRuntime : Runtime {
	explicit TensorRTRuntime(int deviceId) {
		cuda::DeviceContext cudaCtx(deviceId);
	}

	void processImage(
	    const Image &inputImage, const Image &outputImage) override {
		Tensor inputTensor{inputImage};
		Tensor outputTensor{outputImage};
		m_Backend->processImage(inputTensor, outputTensor);
	}

private:
	std::unique_ptr<TensorRTBackend> m_Backend = nullptr;
};

Runtime *createRuntime(int deviceId, const std::filesystem::path &modelPath) {
	::YAML::Node modelConfig;
	{
		std::ifstream file(modelPath);
		file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		modelConfig = ::YAML::Load(file);
	}
	return new TensorRTRuntime(deviceId);
}

}  // namespace core

}  // namespace JoshUpscale
