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
#include "JoshUpscale/core/tensorrt_builder.h"

namespace JoshUpscale {

namespace core {

struct TensorRTRuntime : Runtime {
	explicit TensorRTRuntime(int deviceId,
	    const std::filesystem::path &modelPath,
	    const ::YAML::Node &modelConfig) {
		cuda::DeviceContext cudaCtx(deviceId);
		auto engine = buildTrtEngineCached(modelPath, modelConfig);
		std::vector<std::string> inputNames;
		inputNames.reserve(modelConfig["inputs"].size());
		for (const auto &input : modelConfig["inputs"]) {
			inputNames.push_back(input["name"].as<std::string>());
		}
		auto outputNames =
		    modelConfig["outputs"].as<std::vector<std::string>>();
		m_Backend =
		    std::make_unique<TensorRTBackend>(inputNames, outputNames, engine);
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
	return new TensorRTRuntime(deviceId, modelPath, modelConfig);
}

}  // namespace core

}  // namespace JoshUpscale
