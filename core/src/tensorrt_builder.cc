// Copyright 2023 Ivanov Viktor

#include <NvInfer.h>
#include <NvInferRuntimeBase.h>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/yaml.h>

#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "JoshUpscale/core/exception.h"
#include "JoshUpscale/core/tensorrt.h"
#include "JoshUpscale/core/tensorrt_backend.h"

namespace YAML {

template <>
struct convert<::nvinfer1::Dims> {
	static bool decode(const Node &node,
	    ::nvinfer1::Dims &rhs) {  // NOLINT(runtime/references)
		if (!node.IsSequence() || node.size() >= 8) {
			return false;
		}
		rhs.nbDims = static_cast<std::int32_t>(node.size());
		for (std::size_t i = 0; i < node.size(); ++i) {
			rhs.d[i] = node[i].as<std::int32_t>();
		}
		return true;
	}
};

template <>
struct convert<::nvinfer1::DataType> {
	static bool decode(const Node &node,
	    ::nvinfer1::DataType &rhs) {  // NOLINT(runtime/references)
		auto name = node.as<std::string>();
#define ENUM_DEF(x) \
	{ #x, ::nvinfer1::DataType::k##x }
		static const std::unordered_map<std::string, ::nvinfer1::DataType>
		    enumMap = {
		        ENUM_DEF(FLOAT),
		        ENUM_DEF(HALF),
		        ENUM_DEF(INT8),
		        ENUM_DEF(INT32),
		        ENUM_DEF(BOOL),
		    };
#undef ENUM_DEF
		auto iter = enumMap.find(name);
		if (iter == enumMap.end()) {
			return false;
		}
		rhs = iter->second;
		return true;
	}
};

template <>
struct convert<::nvinfer1::LayerType> {
	static bool decode(const Node &node,
	    ::nvinfer1::LayerType &rhs) {  // NOLINT(runtime/references)
		auto name = node.as<std::string>();
#define ENUM_DEF(x) \
	{ #x, ::nvinfer1::LayerType::k##x }
		static const std::unordered_map<std::string, ::nvinfer1::LayerType>
		    enumMap = {
		        ENUM_DEF(ACTIVATION),
		        ENUM_DEF(CONCATENATION),
		        ENUM_DEF(CONSTANT),
		        ENUM_DEF(CONVOLUTION),
		        ENUM_DEF(DECONVOLUTION),
		        ENUM_DEF(ELEMENTWISE),
		        ENUM_DEF(GATHER),
		        ENUM_DEF(GRID_SAMPLE),
		        ENUM_DEF(IDENTITY),
		        ENUM_DEF(POOLING),
		        ENUM_DEF(REDUCE),
		        ENUM_DEF(RESIZE),
		        ENUM_DEF(SCALE),
		        ENUM_DEF(SHUFFLE),
		        ENUM_DEF(SLICE),
		        ENUM_DEF(UNARY),
		    };
#undef ENUM_DEF
		auto iter = enumMap.find(name);
		if (iter == enumMap.end()) {
			return false;
		}
		rhs = iter->second;
		return true;
	}
};

}  // namespace YAML

namespace JoshUpscale {

namespace core {

namespace {

std::filesystem::path getEnginePath(const ::YAML::Node &modelConfig) {
	return std::filesystem::temp_directory_path() / "JoshUpscale" /
	       (modelConfig["name"].as<std::string>() + ".trt");
}

std::vector<std::byte> readEngineFromFile(
    const std::filesystem::path &filePath) {
	try {
		std::ifstream inputFile(filePath,
		    std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
		inputFile.exceptions(std::ifstream::badbit | std::ifstream::failbit);
		auto size = static_cast<std::size_t>(inputFile.tellg());
		inputFile.seekg(0);
		std::vector<std::byte> engine{size};
		inputFile.read(reinterpret_cast<char *>(engine.data()),
		    static_cast<std::streamsize>(size));
		return engine;
	} catch (...) {
		throw_with_nested_id(
		    std::runtime_error("Failed to load TensorRT engine"));
	}
}

void saveEngine(
    const std::span<std::byte> &engine, const std::filesystem::path &filePath) {
	try {
		std::filesystem::create_directories(filePath.parent_path());
		std::ofstream outputFile{
		    filePath, std::ofstream::out | std::ofstream::binary};
		outputFile.exceptions(std::ofstream::badbit | std::ofstream::failbit);
		outputFile.write(reinterpret_cast<const char *>(engine.data()),
		    static_cast<std::streamsize>(engine.size()));
	} catch (...) {
		throw_with_nested_id(
		    std::runtime_error("Failed to save TensorRT engine"));
	}
}

struct GraphDeserializer {
	explicit GraphDeserializer(::nvinfer1::INetworkDefinition *network)
	    : m_Network(network) {
	}

	void deserialize(
	    const std::filesystem::path &modelPath, const ::YAML::Node &config) {
		loadWeights(
		    modelPath.parent_path() / config["weights"].as<std::string>());
		addInputs(config["inputs"]);
		addLayers(config["layers"]);
		markOutputs(config["outputs"]);
	}

private:
	struct WeightData {
		std::vector<std::byte> storage;
		::nvinfer1::Weights data;
	};

	::nvinfer1::INetworkDefinition *m_Network;
	std::unordered_map<std::string, ::nvinfer1::ITensor *> m_TensorMap;
	std::vector<WeightData> m_Weights;

	void loadWeights(const std::filesystem::path &path) {
		try {
			std::ifstream weightFile(
			    path, std::ifstream::in | std::ifstream::binary);
			weightFile.exceptions(
			    std::ifstream::badbit | std::ifstream::failbit);
			boost::iostreams::filtering_istream is;
			is.push(boost::iostreams::zlib_decompressor());
			is.push(weightFile);
			is.exceptions(std::ios::badbit | std::ios::failbit);
			while (!is.eof()) {
				WeightData weights;
				std::uint32_t dataType;
				std::uint32_t size;
				is.read(reinterpret_cast<char *>(&dataType), sizeof(dataType));
				is.read(reinterpret_cast<char *>(&size), sizeof(size));
				switch (dataType) {
				case 0:
					weights.data.type = ::nvinfer1::DataType::kINT32;
					break;
				case 1:
					weights.data.type = ::nvinfer1::DataType::kFLOAT;
					break;
				default:
					throw new std::runtime_error("Unknown weight type");
				}
				weights.storage.resize(static_cast<std::size_t>(size) * 4);
				is.read(reinterpret_cast<char *>(weights.storage.data()),
				    static_cast<std::streamsize>(weights.storage.size()));
				weights.data.count = static_cast<std::int64_t>(size);
				weights.data.values = weights.storage.data();
				m_Weights.emplace_back(std::move(weights));
			}
		} catch (...) {
			throw_with_nested_id(std::runtime_error("Failed to load weights"));
		}
	}

	void addInputs(const ::YAML::Node &inputs) {
		if (!inputs.IsSequence()) {
			throw std::runtime_error("Failed to load network");
		}
		for (const auto &input : inputs) {
			addInput(input);
		}
	}

	void addLayers(const ::YAML::Node &layers) {
		if (!layers.IsSequence()) {
			throw std::runtime_error("Failed to load network");
		}
		for (const auto &layer : layers) {
			addLayer(layer);
		}
	}

	void markOutputs(const ::YAML::Node &outputs) {
		if (!outputs.IsSequence()) {
			throw std::runtime_error("Failed to load network");
		}
		for (const auto &output : outputs) {
			auto *tensor = m_TensorMap[output.as<std::string>()];
			m_Network->markOutput(*tensor);
			tensor->setAllowedFormats(1 << static_cast<std::uint32_t>(
			                              ::nvinfer1::TensorFormat::kLINEAR));
		}
	}

	void setDynamicRange(const std::string &name, const ::YAML::Node &range) {
		auto *tensor = m_TensorMap[name];
		if (range.IsDefined() || range.IsNull()) {
			if (tensor->getType() == ::nvinfer1::DataType::kFLOAT) {
				throw std::runtime_error("Missing range values");
			}
		}
		if (!range.IsSequence() || range.size() != 2) {
			throw std::runtime_error("Invalid range");
		}
		auto minValue = range[0].as<float>();
		auto maxValue = range[1].as<float>();
		// TensorRT supports symmetric ranges only
		maxValue = std::fmax(std::fabs(minValue), std::fabs(maxValue));
		minValue = -maxValue;
		tensor->setDynamicRange(minValue, maxValue);
	}

	void addInput(const ::YAML::Node &config) {
		auto name = config["name"].as<std::string>();
		auto *tensor = m_Network->addInput(name.c_str(),
		    config["dtype"].as<::nvinfer1::DataType>(),
		    config["shape"].as<::nvinfer1::Dims>());
		m_TensorMap[name] = tensor;
		setDynamicRange(name, config["range"]);
	}

	void addLayer(const ::YAML::Node &config) {
		static const std::unordered_map<::nvinfer1::LayerType,
		    ::nvinfer1::ILayer *(GraphDeserializer::*) (const ::YAML::Node &)>
		    layerDeserializers = {};
		auto layerType = config["type"].as<::nvinfer1::LayerType>();
		auto deserializerIter = layerDeserializers.find(layerType);
		if (deserializerIter == layerDeserializers.end()) {
			throw std::runtime_error("Unsupported layer type");
		}
		auto *layer = (this->*deserializerIter->second)(config);
		auto layerName = config["name"].as<std::string>();
		layer->setName(layerName.c_str());
		if (config["precision"].IsDefined()) {
			auto precision = config["precision"].as<::nvinfer1::DataType>();
			layer->setPrecision(precision);
		}
		auto numOutputs = config["output_names"].size();
		if (layer->getNbOutputs() != static_cast<std::int32_t>(numOutputs)) {
			throw std::runtime_error("Layer output mismatch");
		}
		for (std::size_t i = 0; i < numOutputs; ++i) {
			auto *output = layer->getOutput(static_cast<std::int32_t>(i));
			if (output->getType() !=
			    config["output_dtypes"][i].as<::nvinfer1::DataType>()) {
				throw std::runtime_error("Layer output mismatch");
			}
			auto name = config["output_names"][i].as<std::string>();
			output->setName(name.c_str());
			m_TensorMap[name] = output;
			setDynamicRange(name, config["output_ranges"][i]);
		}
	}
};

void prepareBuilderConfig(
    ::nvinfer1::IBuilder *builder, ::nvinfer1::IBuilderConfig *builderConfig) {
	builderConfig->setMemoryPoolLimit(
	    ::nvinfer1::MemoryPoolType::kWORKSPACE, 2UL << 30);
	builderConfig->setFlag(::nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
	if (builder->platformHasFastFp16()) {
		builderConfig->setFlag(::nvinfer1::BuilderFlag::kFP16);
	}
	if (builder->platformHasFastInt8()) {
		builderConfig->setFlag(::nvinfer1::BuilderFlag::kINT8);
	}
	builderConfig->setFlag(
	    ::nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
	builderConfig->setFlag(::nvinfer1::BuilderFlag::kREJECT_EMPTY_ALGORITHMS);
	builderConfig->setFlag(::nvinfer1::BuilderFlag::kDIRECT_IO);
	if (builder->getNbDLACores() > 0) {
		builderConfig->setDefaultDeviceType(::nvinfer1::DeviceType::kDLA);
		builderConfig->setFlag(::nvinfer1::BuilderFlag::kGPU_FALLBACK);
	}
}

std::vector<std::byte> buildEngineInternal(
    const std::filesystem::path &modelPath, const ::YAML::Node &modelConfig) {
	try {
		trt::Logger logger;
		trt::ErrorRecorder errorRecorder;
		try {
			auto builder = trt::TrtPtr(::nvinfer1::createInferBuilder(logger));
			builder->setErrorRecorder(&errorRecorder);
			auto builderConfig = trt::TrtPtr(builder->createBuilderConfig());
			prepareBuilderConfig(builder, builderConfig);
			auto network = trt::TrtPtr(builder->createNetworkV2(
			    1 << static_cast<std::uint32_t>(::nvinfer1::
			            NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
			GraphDeserializer(network).deserialize(modelPath, modelConfig);
			auto engine = trt::TrtPtr(
			    builder->buildSerializedNetwork(*network, *builderConfig));
			const std::byte *enginePtr =
			    reinterpret_cast<const std::byte *>(engine->data());
			return {enginePtr, enginePtr + engine->size()};
		} catch (...) {
			errorRecorder.rethrowException();
		}
	} catch (...) {
		throw_with_nested_id(
		    std::runtime_error("Failed to build TensorRT engine"));
	}
}

}  // namespace

std::vector<std::byte> buildEngine(
    const std::filesystem::path &modelPath, const ::YAML::Node &modelConfig) {
	auto filePath = getEnginePath(modelConfig);
	if (std::filesystem::exists(filePath)) {
		return readEngineFromFile(filePath);
	}
	std::vector<std::byte> engine = buildEngineInternal(modelPath, modelConfig);
	saveEngine(engine, filePath);
	return engine;
}

}  // namespace core

}  // namespace JoshUpscale
