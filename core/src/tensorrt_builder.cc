// Copyright 2023 Ivanov Viktor

#include <NvInfer.h>
#include <yaml-cpp/yaml.h>

#include <boost/iostreams/filter/gzip.hpp>
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

namespace YAML {

template <>
struct convert<::nvinfer1::Dims> {
	static bool decode(const Node &node,
	    ::nvinfer1::Dims &rhs) {  // NOLINT(runtime/references)
		if (!node.IsSequence() ||
		    node.size() >=
		        static_cast<std::size_t>(::nvinfer1::Dims::MAX_DIMS)) {
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
struct convert<::nvinfer1::Permutation> {
	static bool decode(const Node &node,
	    ::nvinfer1::Permutation &rhs) {  // NOLINT(runtime/references)
		if (!node.IsSequence() ||
		    node.size() !=
		        static_cast<std::size_t>(::nvinfer1::Dims::MAX_DIMS)) {
			return false;
		}
		for (std::size_t i = 0; i < node.size(); ++i) {
			rhs.order[i] = node[i].as<std::int32_t>();
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
#if TENSORRT_VERSION >= 8501
			    ENUM_DEF(GRID_SAMPLE),
#endif
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

template <>
struct convert<::nvinfer1::ActivationType> {
	static bool decode(const Node &node,
	    ::nvinfer1::ActivationType &rhs) {  // NOLINT(runtime/references)
		auto name = node.as<std::string>();
#define ENUM_DEF(x) \
	{ #x, ::nvinfer1::ActivationType::k##x }
		static const std::unordered_map<std::string, ::nvinfer1::ActivationType>
		    enumMap = {
		        ENUM_DEF(RELU),
		        ENUM_DEF(TANH),
		        ENUM_DEF(LEAKY_RELU),
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
struct convert<::nvinfer1::PaddingMode> {
	static bool decode(const Node &node,
	    ::nvinfer1::PaddingMode &rhs) {  // NOLINT(runtime/references)
		auto name = node.as<std::string>();
#define ENUM_DEF(x) \
	{ #x, ::nvinfer1::PaddingMode::k##x }
		static const std::unordered_map<std::string, ::nvinfer1::PaddingMode>
		    enumMap = {
		        ENUM_DEF(EXPLICIT_ROUND_DOWN),
		        ENUM_DEF(EXPLICIT_ROUND_UP),
		        ENUM_DEF(SAME_UPPER),
		        ENUM_DEF(SAME_LOWER),
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
struct convert<::nvinfer1::GatherMode> {
	static bool decode(const Node &node,
	    ::nvinfer1::GatherMode &rhs) {  // NOLINT(runtime/references)
		auto name = node.as<std::string>();
#define ENUM_DEF(x) \
	{ #x, ::nvinfer1::GatherMode::k##x }
		static const std::unordered_map<std::string, ::nvinfer1::GatherMode>
		    enumMap = {
		        ENUM_DEF(DEFAULT),
		        ENUM_DEF(ELEMENT),
		        ENUM_DEF(ND),
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
struct convert<::nvinfer1::ElementWiseOperation> {
	static bool decode(const Node &node,
	    ::nvinfer1::ElementWiseOperation &rhs) {  // NOLINT(runtime/references)
		auto name = node.as<std::string>();
#define ENUM_DEF(x) \
	{ #x, ::nvinfer1::ElementWiseOperation::k##x }
		static const std::unordered_map<std::string,
		    ::nvinfer1::ElementWiseOperation>
		    enumMap = {
		        ENUM_DEF(SUM),
		        ENUM_DEF(PROD),
		        ENUM_DEF(MAX),
		        ENUM_DEF(MIN),
		        ENUM_DEF(SUB),
		        ENUM_DEF(DIV),
		        ENUM_DEF(POW),
		        ENUM_DEF(FLOOR_DIV),
		        ENUM_DEF(AND),
		        ENUM_DEF(OR),
		        ENUM_DEF(XOR),
		        ENUM_DEF(EQUAL),
		        ENUM_DEF(GREATER),
		        ENUM_DEF(LESS),
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
struct convert<::JoshUpscale::core::trt::InterpolationMode> {
	static bool decode(const Node &node,
	    ::JoshUpscale::core::trt::InterpolationMode
	        &rhs) {  // NOLINT(runtime/references)
		auto name = node.as<std::string>();
#define ENUM_DEF(x) \
	{ #x, ::JoshUpscale::core::trt::InterpolationMode::k##x }
		static const std::unordered_map<std::string,
		    ::JoshUpscale::core::trt::InterpolationMode>
		    enumMap = {
			    ENUM_DEF(NEAREST),
			    ENUM_DEF(LINEAR),
#if TENSORRT_VERSION >= 8501
			    ENUM_DEF(CUBIC),
#endif
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
struct convert<::JoshUpscale::core::trt::SampleMode> {
	static bool decode(const Node &node,
	    ::JoshUpscale::core::trt::SampleMode
	        &rhs) {  // NOLINT(runtime/references)
		auto name = node.as<std::string>();
#define ENUM_DEF(x) \
	{ #x, ::JoshUpscale::core::trt::SampleMode::k##x }
		static const std::unordered_map<std::string,
		    ::JoshUpscale::core::trt::SampleMode>
		    enumMap = {
#if TENSORRT_VERSION >= 8501
			    ENUM_DEF(STRICT_BOUNDS),
#else
			    {"STRICT_BOUNDS", ::nvinfer1::SliceMode::kDEFAULT},
#endif
			    ENUM_DEF(WRAP),
			    ENUM_DEF(CLAMP),
			    ENUM_DEF(FILL),
			    ENUM_DEF(REFLECT),
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
struct convert<::nvinfer1::ScaleMode> {
	static bool decode(const Node &node,
	    ::nvinfer1::ScaleMode &rhs) {  // NOLINT(runtime/references)
		auto name = node.as<std::string>();
#define ENUM_DEF(x) \
	{ #x, ::nvinfer1::ScaleMode::k##x }
		static const std::unordered_map<std::string, ::nvinfer1::ScaleMode>
		    enumMap = {
		        ENUM_DEF(UNIFORM),
		        ENUM_DEF(CHANNEL),
		        ENUM_DEF(ELEMENTWISE),
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
struct convert<::nvinfer1::PoolingType> {
	static bool decode(const Node &node,
	    ::nvinfer1::PoolingType &rhs) {  // NOLINT(runtime/references)
		auto name = node.as<std::string>();
#define ENUM_DEF(x) \
	{ #x, ::nvinfer1::PoolingType::k##x }
		static const std::unordered_map<std::string, ::nvinfer1::PoolingType>
		    enumMap = {
		        ENUM_DEF(MAX),
		        ENUM_DEF(AVERAGE),
		        ENUM_DEF(MAX_AVERAGE_BLEND),
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
struct convert<::nvinfer1::ReduceOperation> {
	static bool decode(const Node &node,
	    ::nvinfer1::ReduceOperation &rhs) {  // NOLINT(runtime/references)
		auto name = node.as<std::string>();
#define ENUM_DEF(x) \
	{ #x, ::nvinfer1::ReduceOperation::k##x }
		static const std::unordered_map<std::string,
		    ::nvinfer1::ReduceOperation>
		    enumMap = {
		        ENUM_DEF(SUM),
		        ENUM_DEF(PROD),
		        ENUM_DEF(MAX),
		        ENUM_DEF(MIN),
		        ENUM_DEF(AVG),
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
struct convert<::nvinfer1::UnaryOperation> {
	static bool decode(const Node &node,
	    ::nvinfer1::UnaryOperation &rhs) {  // NOLINT(runtime/references)
		auto name = node.as<std::string>();
#define ENUM_DEF(x) \
	{ #x, ::nvinfer1::UnaryOperation::k##x }
		static const std::unordered_map<std::string, ::nvinfer1::UnaryOperation>
		    enumMap = {
			    ENUM_DEF(EXP),
			    ENUM_DEF(LOG),
			    ENUM_DEF(SQRT),
			    ENUM_DEF(RECIP),
			    ENUM_DEF(ABS),
			    ENUM_DEF(NEG),
			    ENUM_DEF(SIN),
			    ENUM_DEF(COS),
			    ENUM_DEF(TAN),
			    ENUM_DEF(SINH),
			    ENUM_DEF(COSH),
			    ENUM_DEF(ASIN),
			    ENUM_DEF(ACOS),
			    ENUM_DEF(ATAN),
			    ENUM_DEF(ASINH),
			    ENUM_DEF(ACOSH),
			    ENUM_DEF(ATANH),
			    ENUM_DEF(CEIL),
			    ENUM_DEF(FLOOR),
			    ENUM_DEF(ERF),
			    ENUM_DEF(NOT),
			    ENUM_DEF(SIGN),
			    ENUM_DEF(ROUND),
#if TENSORRT_VESION >= 8601
			    ENUM_DEF(ISINF),
#endif
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
struct convert<::nvinfer1::ResizeCoordinateTransformation> {
	static bool decode(const Node &node,
	    ::nvinfer1::ResizeCoordinateTransformation
	        &rhs) {  // NOLINT(runtime/references)
		auto name = node.as<std::string>();
#define ENUM_DEF(x) \
	{ #x, ::nvinfer1::ResizeCoordinateTransformation::k##x }
		static const std::unordered_map<std::string,
		    ::nvinfer1::ResizeCoordinateTransformation>
		    enumMap = {
		        ENUM_DEF(ALIGN_CORNERS),
		        ENUM_DEF(ASYMMETRIC),
		        ENUM_DEF(HALF_PIXEL),
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
struct convert<::nvinfer1::ResizeSelector> {
	static bool decode(const Node &node,
	    ::nvinfer1::ResizeSelector &rhs) {  // NOLINT(runtime/references)
		auto name = node.as<std::string>();
#define ENUM_DEF(x) \
	{ #x, ::nvinfer1::ResizeSelector::k##x }
		static const std::unordered_map<std::string, ::nvinfer1::ResizeSelector>
		    enumMap = {
		        ENUM_DEF(FORMULA),
		        ENUM_DEF(UPPER),
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
struct convert<::nvinfer1::ResizeRoundMode> {
	static bool decode(const Node &node,
	    ::nvinfer1::ResizeRoundMode &rhs) {  // NOLINT(runtime/references)
		auto name = node.as<std::string>();
#define ENUM_DEF(x) \
	{ #x, ::nvinfer1::ResizeRoundMode::k##x }
		static const std::unordered_map<std::string,
		    ::nvinfer1::ResizeRoundMode>
		    enumMap = {
		        ENUM_DEF(HALF_UP),
		        ENUM_DEF(HALF_DOWN),
		        ENUM_DEF(FLOOR),
		        ENUM_DEF(CEIL),
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

std::filesystem::path getTimingCachePath() {
	return std::filesystem::temp_directory_path() / "JoshUpscale" /
	       "timingCache.bin";
}

std::vector<std::byte> readTimingCacheFile() {
	auto filePath = getTimingCachePath();
	if (!std::filesystem::exists(filePath)) {
		return {};
	}
	try {
		std::ifstream inputFile(filePath,
		    std::ifstream::in | std::ifstream::binary | std::ifstream::ate);
		inputFile.exceptions(std::ifstream::badbit | std::ifstream::failbit);
		auto size = static_cast<std::size_t>(inputFile.tellg());
		inputFile.seekg(0);
		std::vector<std::byte> timingCache{size};
		inputFile.read(reinterpret_cast<char *>(timingCache.data()),
		    static_cast<std::streamsize>(size));
		return timingCache;
	} catch (...) {
		throw_with_nested_id(
		    std::runtime_error("Failed to load TensorRT timing cache"));
	}
}

void saveTimingCache(::nvinfer1::IBuilderConfig *builderConfig) {
	auto filePath = getTimingCachePath();
	try {
		std::filesystem::create_directories(filePath.parent_path());
		trt::TrtPtr<::nvinfer1::ITimingCache> managedTimingCache{nullptr};
		const ::nvinfer1::ITimingCache *timingCache =
		    trt::throwIfNull(builderConfig->getTimingCache());
		if (std::filesystem::exists(filePath)) {
			auto oldTimingData = readTimingCacheFile();
			managedTimingCache.reset(
			    trt::throwIfNull(builderConfig->createTimingCache(
			        reinterpret_cast<const void *>(oldTimingData.data()),
			        oldTimingData.size())));
			managedTimingCache->combine(*timingCache, true);
			timingCache = managedTimingCache;
		}
		auto timingData = trt::TrtPtr(timingCache->serialize());
		std::ofstream outputFile{
		    filePath, std::ofstream::out | std::ofstream::binary};
		outputFile.exceptions(std::ofstream::badbit | std::ofstream::failbit);
		outputFile.write(reinterpret_cast<const char *>(timingData->data()),
		    static_cast<std::streamsize>(timingData->size()));
	} catch (...) {
		throw_with_nested_id(
		    std::runtime_error("Failed to save TensorRT timing cache"));
	}
}

trt::TrtPtr<::nvinfer1::ITimingCache> loadTimingCache(
    ::nvinfer1::IBuilderConfig *builderConfig) {
	auto timingData = readTimingCacheFile();
	auto timingCache = trt::TrtPtr(builderConfig->createTimingCache(
	    reinterpret_cast<const char *>(timingData.data()), timingData.size()));
	builderConfig->clearFlag(::nvinfer1::BuilderFlag::kDISABLE_TIMING_CACHE);
	if (!builderConfig->setTimingCache(*timingCache, true)) {
		throw trt::TrtException();
	}
	return timingCache;
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

std::size_t getDimensionsSize(::nvinfer1::Dims dims) {
	std::size_t size = 1;
	for (std::int32_t i = 0; i < dims.nbDims; ++i) {
		size *= static_cast<std::size_t>(dims.d[i]);
	}
	return size;
}

::nvinfer1::Dims convertDimensions(std::vector<std::size_t> dimensions) {
	::nvinfer1::Dims dims;
	dims.nbDims = static_cast<std::int32_t>(dimensions.size());
	for (std::size_t i = 0; i < dimensions.size(); ++i) {
		dims.d[i] = static_cast<std::int32_t>(dimensions[i]);
	}
	return dims;
}

bool dimensionsEqual(::nvinfer1::Dims lhs, ::nvinfer1::Dims rhs) {
	if (lhs.nbDims != rhs.nbDims) {
		return false;
	}
	return std::equal(lhs.d, lhs.d + lhs.nbDims, rhs.d);
}

struct GraphDeserializer {
	explicit GraphDeserializer(::nvinfer1::INetworkDefinition *network)
	    : m_Network(network) {
	}

	void deserialize(
	    const std::filesystem::path &modelPath, const ::YAML::Node &config) {
		loadWeights(
		    modelPath.parent_path() / config["weights"].as<std::string>());
		for (const auto &input : config["inputs"]) {
			addInput(input);
		}
		for (const auto &layer : config["layers"]) {
			addLayer(layer);
		}
		for (const auto &output : config["outputs"]) {
			auto *tensor = m_TensorMap[output.as<std::string>()];
			m_Network->markOutput(*tensor);
			tensor->setAllowedFormats(
			    1 << static_cast<::nvinfer1::TensorFormats>(
			        ::nvinfer1::TensorFormat::kLINEAR));
		}
		if (!validateNetwork()) {
			throw std::runtime_error("Model is invalid");
		}
	}

private:
	::nvinfer1::INetworkDefinition *m_Network;
	std::unordered_map<std::string, ::nvinfer1::ITensor *> m_TensorMap;
	std::vector<::nvinfer1::Weights> m_Weights;
	std::vector<std::vector<std::byte>> m_WeightData;

	bool validateNetwork() {
		if (m_Network->getNbInputs() == 1) {
			if (m_Network->getNbOutputs() != 1) {
				return false;
			}
		} else if (m_Network->getNbInputs() >= 3) {
			if (m_Network->getNbInputs() != m_Network->getNbOutputs()) {
				return false;
			}
			for (std::int32_t i = 1; i < m_Network->getNbInputs(); ++i) {
				if (!dimensionsEqual(m_Network->getInput(i)->getDimensions(),
				        m_Network->getOutput(i)->getDimensions())) {
					return false;
				}
			}
			if (!dimensionsEqual(m_Network->getInput(1)->getDimensions(),
			        convertDimensions({1, 3, 1080, 1920}))) {
				return false;
			}
			constexpr std::size_t maxSize = 3ULL * 272ULL * 480ULL;
			for (std::int32_t i = 2; i < m_Network->getNbInputs(); ++i) {
				if (getDimensionsSize(m_Network->getInput(i)->getDimensions()) >
				    maxSize) {
					return false;
				}
			}
		} else {
			return false;
		}
		if (!dimensionsEqual(m_Network->getInput(0)->getDimensions(),
		        convertDimensions({1, 270, 480, 3}))) {
			return false;
		}
		if (!dimensionsEqual(m_Network->getOutput(0)->getDimensions(),
		        convertDimensions({1, 1080, 1920, 3}))) {
			return false;
		}
		for (std::int32_t i = 0; i < m_Network->getNbInputs(); ++i) {
			if (m_Network->getInput(i)->getType() !=
			    ::nvinfer1::DataType::kFLOAT) {
				return false;
			}
		}
		for (std::int32_t i = 0; i < m_Network->getNbOutputs(); ++i) {
			if (m_Network->getOutput(i)->getType() !=
			    ::nvinfer1::DataType::kFLOAT) {
				return false;
			}
		}
		return true;
	}

	void loadWeights(const std::filesystem::path &path) {
		try {
			std::ifstream weightFile(
			    path, std::ifstream::in | std::ifstream::binary);
			weightFile.exceptions(
			    std::ifstream::badbit | std::ifstream::failbit);
			boost::iostreams::filtering_istream is;
			is.push(boost::iostreams::gzip_decompressor());
			is.push(weightFile);
			while (is) {
				::nvinfer1::Weights weights;
				std::uint32_t dataType;
				std::uint32_t size;
				is.read(reinterpret_cast<char *>(&dataType), sizeof(dataType));
				is.read(reinterpret_cast<char *>(&size), sizeof(size));
				switch (dataType) {
				case 0:
					weights.type = ::nvinfer1::DataType::kINT32;
					break;
				case 1:
					weights.type = ::nvinfer1::DataType::kFLOAT;
					break;
				default:
					throw new std::runtime_error("Unknown weight type");
				}
				auto &data = m_WeightData.emplace_back(
				    static_cast<std::size_t>(size) * 4);
				is.read(reinterpret_cast<char *>(data.data()),
				    static_cast<std::streamsize>(data.size()));
				weights.count = static_cast<std::int64_t>(size);
				weights.values = data.data();
				m_Weights.push_back(weights);
			}
		} catch (...) {
			throw_with_nested_id(std::runtime_error("Failed to load weights"));
		}
	}

	void setDynamicRange(const std::string &name, const ::YAML::Node &range) {
		auto *tensor = m_TensorMap[name];
		if (!range.IsDefined() || range.IsNull()) {
			if (tensor->getType() == ::nvinfer1::DataType::kFLOAT) {
				throw std::runtime_error("Missing range values");
			}
			return;
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
		// clang-format off
		static const std::unordered_map<::nvinfer1::LayerType,
		    ::nvinfer1::ILayer *(GraphDeserializer::*) (const ::YAML::Node &)>
		    layerDeserializers = {
		        {::nvinfer1::LayerType::kACTIVATION,
		            &GraphDeserializer::addActivation},
		        {::nvinfer1::LayerType::kCONCATENATION,
		            &GraphDeserializer::addConcat},
		        {::nvinfer1::LayerType::kCONSTANT,
		            &GraphDeserializer::addConstant},
		        {::nvinfer1::LayerType::kCONVOLUTION,
		            &GraphDeserializer::addConvolution},
		        {::nvinfer1::LayerType::kDECONVOLUTION,
		            &GraphDeserializer::addDeconvolution},
		        {::nvinfer1::LayerType::kELEMENTWISE,
		            &GraphDeserializer::addElementwise},
		        {::nvinfer1::LayerType::kGATHER,
		            &GraphDeserializer::addGather},
#if TENSORRT_VERSION >= 8501
		        {::nvinfer1::LayerType::kGRID_SAMPLE,
		            &GraphDeserializer::addGridSample},
#endif
		        {::nvinfer1::LayerType::kIDENTITY,
		            &GraphDeserializer::addIdentity},
		        {::nvinfer1::LayerType::kPOOLING,
		            &GraphDeserializer::addPooling},
		        {::nvinfer1::LayerType::kREDUCE,
		            &GraphDeserializer::addReduce},
		        {::nvinfer1::LayerType::kRESIZE,
		            &GraphDeserializer::addResize},
		        {::nvinfer1::LayerType::kSCALE,
		            &GraphDeserializer::addScale},
		        {::nvinfer1::LayerType::kSHUFFLE,
		            &GraphDeserializer::addShuffle},
		        {::nvinfer1::LayerType::kSLICE,
		            &GraphDeserializer::addSlice},
		        {::nvinfer1::LayerType::kUNARY,
		            &GraphDeserializer::addUnary},
		    };
		// clang-format on
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

	::nvinfer1::ILayer *addActivation(const ::YAML::Node &config) {
		if (config["inputs"].size() != 1) {
			throw std::runtime_error("Unsupported layer");
		}
		auto *inputTensor = m_TensorMap[config["inputs"][0].as<std::string>()];
		auto activationType =
		    config["activation_type"].as<::nvinfer1::ActivationType>();
		auto *layer = m_Network->addActivation(*inputTensor, activationType);
		layer->setAlpha(config["alpha"].as<float>());
		layer->setBeta(config["beta"].as<float>());
		return layer;
	}

	::nvinfer1::ILayer *addConcat(const ::YAML::Node &config) {
		std::vector<::nvinfer1::ITensor *> inputs;
		inputs.reserve(config["inputs"].size());
		for (const auto &input : config["inputs"]) {
			inputs.push_back(m_TensorMap[input.as<std::string>()]);
		}
		auto *layer = m_Network->addConcatenation(
		    inputs.data(), static_cast<std::int32_t>(inputs.size()));
		layer->setAxis(config["axis"].as<std::int32_t>());
		return layer;
	}

	::nvinfer1::ILayer *addConstant(const ::YAML::Node &config) {
		if (config["inputs"].size() != 0) {
			throw std::runtime_error("Unsupported layer");
		}
		auto shape = config["shape"].as<::nvinfer1::Dims>();
		auto &weights = m_Weights.at(config["weights"].as<std::size_t>());
		auto *layer = m_Network->addConstant(shape, weights);
		return layer;
	}

	::nvinfer1::ILayer *addConvolution(const ::YAML::Node &config) {
		if (config["inputs"].size() != 1) {
			throw std::runtime_error("Unsupported layer");
		}
		auto *inputTensor = m_TensorMap[config["inputs"][0].as<std::string>()];
		auto &kernel = m_Weights.at(config["kernel"].as<std::size_t>());
		auto &bias = m_Weights.at(config["bias"].as<std::size_t>());
		auto *layer = m_Network->addConvolutionNd(*inputTensor,
		    config["num_output_maps"].as<std::int32_t>(),
		    config["kernel_size_nd"].as<::nvinfer1::Dims>(), kernel, bias);
		layer->setPaddingMode(
		    config["padding_mode"].as<::nvinfer1::PaddingMode>());
		layer->setNbGroups(config["num_groups"].as<std::int32_t>());
		layer->setStrideNd(config["stride_nd"].as<::nvinfer1::Dims>());
		layer->setPaddingNd(config["padding_nd"].as<::nvinfer1::Dims>());
		layer->setDilationNd(config["dilation_nd"].as<::nvinfer1::Dims>());
		return layer;
	}

	::nvinfer1::ILayer *addDeconvolution(const ::YAML::Node &config) {
		if (config["inputs"].size() != 1) {
			throw std::runtime_error("Unsupported layer");
		}
		auto *inputTensor = m_TensorMap[config["inputs"][0].as<std::string>()];
		auto &kernel = m_Weights.at(config["kernel"].as<std::size_t>());
		auto &bias = m_Weights.at(config["bias"].as<std::size_t>());
		auto *layer = m_Network->addDeconvolutionNd(*inputTensor,
		    config["num_output_maps"].as<std::int32_t>(),
		    config["kernel_size_nd"].as<::nvinfer1::Dims>(), kernel, bias);
		layer->setPaddingMode(
		    config["padding_mode"].as<::nvinfer1::PaddingMode>());
		layer->setNbGroups(config["num_groups"].as<std::int32_t>());
		layer->setStrideNd(config["stride_nd"].as<::nvinfer1::Dims>());
		layer->setPaddingNd(config["padding_nd"].as<::nvinfer1::Dims>());
		layer->setDilationNd(config["dilation_nd"].as<::nvinfer1::Dims>());
		return layer;
	}

	::nvinfer1::ILayer *addElementwise(const ::YAML::Node &config) {
		if (config["inputs"].size() != 2) {
			throw std::runtime_error("Unsupported layer");
		}
		auto *inputTensor1 = m_TensorMap[config["inputs"][0].as<std::string>()];
		auto *inputTensor2 = m_TensorMap[config["inputs"][1].as<std::string>()];
		auto op = config["op"].as<::nvinfer1::ElementWiseOperation>();
		auto *layer =
		    m_Network->addElementWise(*inputTensor1, *inputTensor2, op);
		return layer;
	}

	::nvinfer1::ILayer *addGather(const ::YAML::Node &config) {
		if (config["inputs"].size() != 2) {
			throw std::runtime_error("Unsupported layer");
		}
		auto *inputTensor1 = m_TensorMap[config["inputs"][0].as<std::string>()];
		auto *indices = m_TensorMap[config["inputs"][1].as<std::string>()];
		auto mode = config["mode"].as<::nvinfer1::GatherMode>();
		auto *layer = m_Network->addGatherV2(*inputTensor1, *indices, mode);
		layer->setGatherAxis(config["axis"].as<std::int32_t>());
		layer->setNbElementWiseDims(
		    config["num_elementwise_dims"].as<std::int32_t>());
		return layer;
	}

#if TENSORRT_VERSION >= 8501
	::nvinfer1::ILayer *addGridSample(const ::YAML::Node &config) {
		if (config["inputs"].size() != 2) {
			throw std::runtime_error("Unsupported layer");
		}
		auto *inputTensor = m_TensorMap[config["inputs"][0].as<std::string>()];
		auto *grid = m_TensorMap[config["inputs"][1].as<std::string>()];
		auto *layer = m_Network->addGridSample(*inputTensor, *grid);
		layer->setAlignCorners(config["align_corners"].as<bool>());
		layer->setInterpolationMode(
		    config["interpolation_mode"].as<trt::InterpolationMode>());
		layer->setSampleMode(config["sample_mode"].as<trt::SampleMode>());
		return layer;
	}
#endif

	::nvinfer1::ILayer *addIdentity(const ::YAML::Node &config) {
		if (config["inputs"].size() != 1) {
			throw std::runtime_error("Unsupported layer");
		}
		auto *inputTensor = m_TensorMap[config["inputs"][0].as<std::string>()];
		auto *layer = m_Network->addIdentity(*inputTensor);
		layer->setOutputType(
		    0, config["output_dtypes"][0].as<::nvinfer1::DataType>());
		return layer;
	}

	::nvinfer1::ILayer *addPooling(const ::YAML::Node &config) {
		if (config["inputs"].size() != 1) {
			throw std::runtime_error("Unsupported layer");
		}
		auto *inputTensor = m_TensorMap[config["inputs"][0].as<std::string>()];
		auto *layer = m_Network->addPoolingNd(*inputTensor,
		    config["pooling_type"].as<::nvinfer1::PoolingType>(),
		    config["window_size_nd"].as<::nvinfer1::Dims>());
		layer->setPaddingMode(
		    config["padding_mode"].as<::nvinfer1::PaddingMode>());
		layer->setBlendFactor(config["blend_factor"].as<float>());
		layer->setAverageCountExcludesPadding(
		    config["average_count_excludes_padding"].as<bool>());
		layer->setStrideNd(config["stride_nd"].as<::nvinfer1::Dims>());
		layer->setPaddingNd(config["padding_nd"].as<::nvinfer1::Dims>());
		return layer;
	}

	::nvinfer1::ILayer *addReduce(const ::YAML::Node &config) {
		if (config["inputs"].size() != 1) {
			throw std::runtime_error("Unsupported layer");
		}
		auto *inputTensor = m_TensorMap[config["inputs"][0].as<std::string>()];
		auto *layer = m_Network->addReduce(*inputTensor,
		    config["op"].as<::nvinfer1::ReduceOperation>(),
		    config["axes"].as<std::uint32_t>(), config["keep_dims"].as<bool>());
		return layer;
	}

	::nvinfer1::ILayer *addResize(const ::YAML::Node &config) {
		// TODO(me): support dynamic resize
		if (config["inputs"].size() != 1) {
			throw std::runtime_error("Unsupported layer");
		}
		auto *inputTensor = m_TensorMap[config["inputs"][0].as<std::string>()];
		auto *layer = m_Network->addResize(*inputTensor);
		if (config["scales"]) {
			auto scales = config["scales"].as<std::vector<float>>();
			layer->setScales(
			    scales.data(), static_cast<std::int32_t>(scales.size()));
		} else if (config["shape"]) {
			layer->setOutputDimensions(config["shape"].as<::nvinfer1::Dims>());
		} else {
			throw std::runtime_error("Unsupported layer");
		}
		layer->setResizeMode(
		    config["resize_mode"].as<trt::InterpolationMode>());
		layer->setCoordinateTransformation(
		    config["coordinate_transformation"]
		        .as<::nvinfer1::ResizeCoordinateTransformation>());
		layer->setSelectorForSinglePixel(config["selector_for_single_pixel"]
		                                     .as<::nvinfer1::ResizeSelector>());
		layer->setNearestRounding(
		    config["nearest_rounding"].as<::nvinfer1::ResizeRoundMode>());
#if TENSORRT_VERSION >= 8501
		layer->setExcludeOutside(config["exclude_outside"].as<bool>());
		layer->setCubicCoeff(config["cubic_coeff"].as<float>());
#endif
		return layer;
	}

	::nvinfer1::ILayer *addScale(const ::YAML::Node &config) {
		if (config["inputs"].size() != 1) {
			throw std::runtime_error("Unsupported layer");
		}
		auto *inputTensor = m_TensorMap[config["inputs"][0].as<std::string>()];
		auto *layer = m_Network->addScale(*inputTensor,
		    config["mode"].as<::nvinfer1::ScaleMode>(),
		    m_Weights.at(config["shift"].as<std::size_t>()),
		    m_Weights.at(config["scale"].as<std::size_t>()),
		    m_Weights.at(config["power"].as<std::size_t>()));
		layer->setChannelAxis(config["channel_axis"].as<std::int32_t>());
		return layer;
	}

	::nvinfer1::ILayer *addShuffle(const ::YAML::Node &config) {
		if (config["inputs"].size() != 1) {
			throw std::runtime_error("Unsupported layer");
		}
		auto *inputTensor = m_TensorMap[config["inputs"][0].as<std::string>()];
		auto *layer = m_Network->addShuffle(*inputTensor);
		layer->setFirstTranspose(
		    config["first_transpose"].as<::nvinfer1::Permutation>());
		layer->setSecondTranspose(
		    config["second_transpose"].as<::nvinfer1::Permutation>());
		if (config["reshape_dims"] && !config["reshape_dims"].IsNull()) {
			layer->setReshapeDimensions(
			    config["reshape_dims"].as<::nvinfer1::Dims>());
		}
		layer->setZeroIsPlaceholder(config["zero_is_placeholder"].as<bool>());
		return layer;
	}

	::nvinfer1::ILayer *addSlice(const ::YAML::Node &config) {
		if (config["inputs"].size() != 1) {
			throw std::runtime_error("Unsupported layer");
		}
		auto *inputTensor = m_TensorMap[config["inputs"][0].as<std::string>()];
		auto *layer = m_Network->addSlice(*inputTensor,
		    config["start"].as<::nvinfer1::Dims>(),
		    config["shape"].as<::nvinfer1::Dims>(),
		    config["stride"].as<::nvinfer1::Dims>());
		layer->setMode(config["mode"].as<trt::SampleMode>());
		return layer;
	}

	::nvinfer1::ILayer *addUnary(const ::YAML::Node &config) {
		if (config["inputs"].size() != 1) {
			throw std::runtime_error("Unsupported layer");
		}
		auto *inputTensor = m_TensorMap[config["inputs"][0].as<std::string>()];
		auto *layer = m_Network->addUnary(
		    *inputTensor, config["op"].as<::nvinfer1::UnaryOperation>());
		return layer;
	}
};

void prepareBuilderConfig(
    ::nvinfer1::IBuilder *builder, ::nvinfer1::IBuilderConfig *builderConfig) {
	builderConfig->setMemoryPoolLimit(
	    ::nvinfer1::MemoryPoolType::kWORKSPACE, 2ULL << 30);
	builderConfig->setFlag(::nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
	if (builder->platformHasFastFp16()) {
		builderConfig->setFlag(::nvinfer1::BuilderFlag::kFP16);
	}
	if (builder->platformHasFastInt8()) {
		// INT8 is disabled because of image quality
		// builderConfig->setFlag(::nvinfer1::BuilderFlag::kINT8);
	}
	builderConfig->setFlag(
	    ::nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
	builderConfig->setFlag(::nvinfer1::BuilderFlag::kREJECT_EMPTY_ALGORITHMS);
	if (builder->getNbDLACores() > 0) {
		builderConfig->setDefaultDeviceType(::nvinfer1::DeviceType::kDLA);
		builderConfig->setFlag(::nvinfer1::BuilderFlag::kGPU_FALLBACK);
	}
#if TENSORRT_VERSION >= 8601
	builderConfig->setBuilderOptimizationLevel(5);
#endif
	builderConfig->setTacticSources(
	    (1 << static_cast<::nvinfer1::TacticSources>(
	         ::nvinfer1::TacticSource::kEDGE_MASK_CONVOLUTIONS)) |
	    (1 << static_cast<::nvinfer1::TacticSources>(
	         ::nvinfer1::TacticSource::kCUBLAS)) |
	    (1 << static_cast<::nvinfer1::TacticSources>(
	         ::nvinfer1::TacticSource::kCUBLAS_LT)));
}

}  // namespace

std::vector<std::byte> buildTrtEngine(
    const std::filesystem::path &modelPath, const ::YAML::Node &modelConfig) {
	try {
		trt::Logger logger;
		trt::ErrorRecorder errorRecorder;
		try {
			auto builder = trt::TrtPtr(::nvinfer1::createInferBuilder(logger));
			builder->setErrorRecorder(&errorRecorder);
			auto builderConfig = trt::TrtPtr(builder->createBuilderConfig());
			prepareBuilderConfig(builder, builderConfig);
			auto timingCache = loadTimingCache(builderConfig);
			auto network = trt::TrtPtr(builder->createNetworkV2(
			    1 << static_cast<::nvinfer1::NetworkDefinitionCreationFlags>(
			        ::nvinfer1::NetworkDefinitionCreationFlag::
			            kEXPLICIT_BATCH)));
			GraphDeserializer deserializer{network};
			deserializer.deserialize(modelPath, modelConfig);
			auto engine = trt::TrtPtr(
			    builder->buildSerializedNetwork(*network, *builderConfig));
			saveTimingCache(builderConfig);
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

std::vector<std::byte> buildTrtEngineCached(
    const std::filesystem::path &modelPath, const ::YAML::Node &modelConfig) {
	auto filePath = getEnginePath(modelConfig);
	if (std::filesystem::exists(filePath)) {
		return readEngineFromFile(filePath);
	}
	std::vector<std::byte> engine = buildTrtEngine(modelPath, modelConfig);
	saveEngine(engine, filePath);
	return engine;
}

}  // namespace core

}  // namespace JoshUpscale
