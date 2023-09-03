// Copyright 2022 Ivanov Viktor

#include "JoshUpscale/core.h"

#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <fstream>
#include <memory>
#include <ranges>
#include <string>
#include <vector>

#include "JoshUpscale/core/cuda.h"
#include "JoshUpscale/core/tensor.h"
#include "JoshUpscale/core/tensorrt_backend.h"

namespace JoshUpscale {

namespace core {

struct TensorRTRuntime : Runtime {
	TensorRTRuntime(int deviceId, const std::vector<std::string> &inputNames,
	    const std::vector<std::string> &outputNames,
	    std::span<std::byte> engine) {
		cuda::DeviceContext cudaCtx(deviceId);
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

Runtime *createRuntime(
    int deviceId, const char *modelPath, const char *enginePath) {
	std::vector<std::byte> engine;
	{
		std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
		file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		auto size = static_cast<std::streamsize>(file.tellg());
		file.seekg(0, std::ios::beg);
		engine.resize(static_cast<std::size_t>(size));
		file.read(reinterpret_cast<char *>(engine.data()), size);
	}
	::YAML::Node modelConfig;
	{
		std::ifstream file(modelPath);
		file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		modelConfig = ::YAML::Load(file);
	}
	std::vector<std::string> inputNames;
	auto outputNames = modelConfig["outputs"].as<std::vector<std::string>>();
	for (auto input : modelConfig["inputs"]) {
		inputNames.push_back(input["name"].as<std::string>());
	}
	return new TensorRTRuntime(deviceId, inputNames, outputNames, engine);
}

}  // namespace core

}  // namespace JoshUpscale
