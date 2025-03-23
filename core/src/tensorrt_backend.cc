// Copyright 2025 Ivanov Viktor

#include "JoshUpscale/core/tensorrt_backend.h"

#include <NvInfer.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "JoshUpscale/core/cuda.h"
#include "JoshUpscale/core/cuda_convert.h"
#include "JoshUpscale/core/tensor.h"
#include "JoshUpscale/core/tensorrt.h"
#include "JoshUpscale/core/utils.h"

namespace JoshUpscale {

namespace core {

namespace {

constexpr std::int64_t kScale = 4;

cuda::DataType convertDataType(::nvinfer1::DataType type) {
	switch (type) {
	case ::nvinfer1::DataType::kUINT8:
		return cuda::DataType::UINT8;
	case ::nvinfer1::DataType::kHALF:
		return cuda::DataType::HALF;
	case ::nvinfer1::DataType::kFLOAT:
		return cuda::DataType::FLOAT;
	default:
		break;
	}
	throw std::invalid_argument("Unsupported tensor type");
}

bool compareShapes(const ::nvinfer1::Dims &lhs, const ::nvinfer1::Dims &rhs) {
	if (lhs.nbDims != rhs.nbDims) {
		return false;
	}
	for (std::size_t i = 0, size = static_cast<std::size_t>(lhs.nbDims);
	    i < size; ++i) {
		if (lhs.d[i] != rhs.d[i]) {
			return false;
		}
	}
	return true;
}

std::size_t calculateShapeSize(const ::nvinfer1::Dims &shape) {
	std::size_t result = 1;
	for (std::size_t i = 0, size = static_cast<std::size_t>(shape.nbDims);
	    i < size; ++i) {
		result *= static_cast<std::size_t>(shape.d[i]);
	}
	return result;
}

cuda::GenericCudaBuffer allocateTensor(
    ::nvinfer1::ICudaEngine *engine, std::int32_t index) {
	const char *tensorName = engine->getIOTensorName(index);
	if (engine->getTensorFormat(tensorName) !=
	    ::nvinfer1::TensorFormat::kLINEAR) {
		throw std::invalid_argument("Unsupported tensor format");
	}
	return cuda::GenericCudaBuffer(
	    calculateShapeSize(engine->getTensorShape(tensorName)),
	    convertDataType(engine->getTensorDataType(tensorName)));
}

::nvinfer1::Dims getInputShape(::nvinfer1::ICudaEngine *engine) {
	for (std::int32_t i = 0, size = engine->getNbIOTensors(); i < size; ++i) {
		const char *name = engine->getIOTensorName(i);
		if (engine->getTensorIOMode(name) == ::nvinfer1::TensorIOMode::kINPUT) {
			return engine->getTensorShape(name);
		}
	}
	unreachable();
}

void validateEngineIO(::nvinfer1::ICudaEngine *const engine,
    const std::vector<std::int32_t> &inputIndices,
    const std::vector<std::int32_t> &outputIndices) {
	if (inputIndices.size() != outputIndices.size()) {
		throw std::invalid_argument("Engine I/O mismatch");
	}
	for (std::size_t i = 0, size = inputIndices.size(); i < size; ++i) {
		const char *inputName = engine->getIOTensorName(inputIndices[i]);
		const char *outputName = engine->getIOTensorName(outputIndices[i]);
		if (engine->getTensorDataType(inputName) !=
		    engine->getTensorDataType(outputName)) {
			throw std::invalid_argument("Engine I/O mismatch");
		}
		auto shape = engine->getTensorShape(inputName);
		if (shape.nbDims != 4 || shape.d[0] != 1 ||
		    (i == 0 && shape.d[3] != 3) || (i != 0 && shape.d[1] != 3)) {
			throw std::invalid_argument("Unsupported input shape");
		}
		if (i == 0) {
			shape.d[1] *= kScale;
			shape.d[2] *= kScale;
		}
		if (!compareShapes(shape, engine->getTensorShape(outputName))) {
			throw std::invalid_argument("Engine I/O mismatch");
		}
	}
}

}  // namespace

TensorRTBackend::TensorRTBackend(std::span<std::byte> engine)
    : m_Device{cuda::getDevice()}
    , m_DeviceMemory{nullptr}
    , m_InputBuffer{nullptr}
    , m_InputBufferFp{nullptr}
    , m_OutputBuffer{nullptr}
    , m_OutputBufferFp{nullptr}
    , m_BindingsIdx{0}
    , m_Engine{nullptr}
    , m_Contexts{nullptr, nullptr}
    , m_CudaGraphs{nullptr, nullptr}
    , m_CudaGraphExec{nullptr, nullptr} {
	if (engine.size() < 20) {
		throw std::invalid_argument("Invalid engine");
	}
	std::uint32_t trtSize;
	std::memcpy(&trtSize, engine.data() + 16, sizeof(trtSize));
	std::size_t numElements = static_cast<std::size_t>(engine.back()) + 1;
	if (static_cast<std::size_t>(trtSize) + numElements != engine.size()) {
		trtSize += 24;
	}
	bool hasReindex = false;
	if (static_cast<std::size_t>(trtSize) + numElements == engine.size()) {
		hasReindex = true;
	} else {
		trtSize = static_cast<std::uint32_t>(engine.size());
	}
	try {
		auto runtime = trt::TrtPtr(::nvinfer1::createInferRuntime(m_Logger));
		runtime->setErrorRecorder(&m_ErrorRecorder);
		m_Engine = trt::TrtPtr(runtime->deserializeCudaEngine(
		    engine.data(), static_cast<std::size_t>(trtSize)));
		std::vector<std::int32_t> inputIndices;
		std::vector<std::int32_t> outputIndices;
		for (std::int32_t i = 0, size = m_Engine->getNbIOTensors(); i < size;
		    ++i) {
			const char *tensorName = m_Engine->getIOTensorName(i);
			switch (m_Engine->getTensorIOMode(tensorName)) {
			case ::nvinfer1::TensorIOMode::kINPUT:
				inputIndices.push_back(i);
				break;
			case ::nvinfer1::TensorIOMode::kOUTPUT:
				outputIndices.push_back(i);
				break;
			default:
				break;
			}
		}
		std::vector<std::int32_t> extraIndices;
		if (hasReindex) {
			std::vector<std::int32_t> newOutputIndices;
			newOutputIndices.reserve(inputIndices.size());
			for (auto iter = engine.end() -
			                 static_cast<std::ptrdiff_t>(numElements),
			          end = engine.end() - 1;
			    iter != end; ++iter) {
				newOutputIndices.push_back(
				    outputIndices.at(static_cast<std::size_t>(*iter)));
			}
			for (std::int32_t idx : outputIndices) {
				bool found = false;
				for (std::int32_t newIdx : newOutputIndices) {
					if (newIdx == idx) {
						found = true;
						break;
					}
				}
				if (!found) {
					extraIndices.push_back(idx);
				}
			}
			outputIndices = std::move(newOutputIndices);
		} else if (outputIndices.size() >= inputIndices.size()) {
			auto outputEnd = outputIndices.begin() +
			                 static_cast<std::ptrdiff_t>(inputIndices.size());
			extraIndices = {outputEnd, outputIndices.end()};
			outputIndices = {outputIndices.begin(), outputEnd};
		}
		validateEngineIO(m_Engine, inputIndices, outputIndices);
#if TENSORRT_VERSION >= 10000
		m_DeviceMemory = cuda::CudaBuffer<std::uint8_t>(
		    static_cast<std::size_t>(m_Engine->getDeviceMemorySizeV2()));
#else
		m_DeviceMemory =
		    cuda::CudaBuffer<std::uint8_t>(m_Engine->getDeviceMemorySize());
#endif
		m_InputBuffer = cuda::CudaBuffer<std::uint8_t>(
		    calculateShapeSize(m_Engine->getTensorShape(
		        m_Engine->getIOTensorName(inputIndices[0]))) /
		    3 * 4);
		m_InputBufferFp = allocateTensor(m_Engine, inputIndices[0]);
		m_OutputBuffer = cuda::CudaBuffer<std::uint8_t>(
		    calculateShapeSize(m_Engine->getTensorShape(
		        m_Engine->getIOTensorName(outputIndices[0]))) /
		    3 * 4);
		m_OutputBufferFp = allocateTensor(m_Engine, outputIndices[0]);
		for (std::size_t i = 0; i < 2; ++i) {
			for (std::size_t j = 1, size = inputIndices.size(); j < size; ++j) {
				m_InterBuffers.emplace_back(
				    allocateTensor(m_Engine, inputIndices[j]));
			}
		}
		for (std::int32_t idx : extraIndices) {
			m_ExtraBuffers.emplace_back(allocateTensor(m_Engine, idx));
		}
		for (std::size_t i = 0; i < 2; ++i) {
#if TENSORRT_VERSION >= 10000
			m_Contexts[i] = trt::TrtPtr(m_Engine->createExecutionContext(
			    ::nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
			m_Contexts[i]->setDeviceMemoryV2(m_DeviceMemory.get(),
			    static_cast<std::int64_t>(m_DeviceMemory.getByteSize()));
#else
			m_Contexts[i] = trt::TrtPtr(
			    m_Engine->createExecutionContextWithoutDeviceMemory());
			m_Contexts[i]->setDeviceMemory(m_DeviceMemory.get());
#endif
			m_Contexts[i]->setNvtxVerbosity(
			    ::nvinfer1::ProfilingVerbosity::kNONE);
			m_Contexts[i]->setEnqueueEmitsProfile(false);
			m_Contexts[i]->setTensorAddress(
			    m_Engine->getIOTensorName(inputIndices[0]),
			    m_InputBufferFp.get());
			m_Contexts[i]->setTensorAddress(
			    m_Engine->getIOTensorName(outputIndices[0]),
			    m_OutputBufferFp.get());
			for (std::size_t j = 0, size = extraIndices.size(); j < size; ++j) {
				m_Contexts[i]->setTensorAddress(
				    m_Engine->getIOTensorName(extraIndices[j]),
				    m_ExtraBuffers[j].get());
			}
			for (std::size_t j = 1, size = inputIndices.size(); j < size; ++j) {
				std::size_t inputOffset = i * (inputIndices.size() - 1);
				std::size_t outputOffset = (i ^ 1) * (inputIndices.size() - 1);
				m_Contexts[i]->setTensorAddress(
				    m_Engine->getIOTensorName(inputIndices[j]),
				    m_InterBuffers[inputOffset + j - 1].get());
				m_Contexts[i]->setTensorAddress(
				    m_Engine->getIOTensorName(outputIndices[j]),
				    m_InterBuffers[outputOffset + j - 1].get());
			}
			m_Stream.beginCapture();
			if (!m_Contexts[i]->enqueueV3(m_Stream)) {
				throw trt::TrtException();
			}
			m_CudaGraphs[i] = m_Stream.endCapture();
			m_CudaGraphExec[i] = m_CudaGraphs[i].instantiate();
			m_Stream.synchronize();
		}
	} catch (...) {
		m_ErrorRecorder.rethrowException();
	}
}

void TensorRTBackend::process(
    const GenericTensor &inputTensor, const GenericTensor &outputTensor) {
	cuda::DeviceContext deviceCtx(m_Device);
	cuda::cudaConvert(inputTensor, m_InputBufferFp, m_InputBuffer, m_Stream);
	m_CudaGraphExec[m_BindingsIdx].launch(m_Stream);
	cuda::cudaConvert(m_OutputBufferFp, outputTensor, m_OutputBuffer, m_Stream);
	m_Stream.synchronize();
	m_BindingsIdx ^= 1;
}

TensorRTBackend::FrameSize TensorRTBackend::getFrameSize() const {
	auto shape = getInputShape(m_Engine);
	return {
	    .inputWidth = static_cast<std::size_t>(shape.d[2]),
	    .inputHeight = static_cast<std::size_t>(shape.d[1]),
	    .outputWidth = static_cast<std::size_t>(shape.d[2] * kScale),
	    .outputHeight = static_cast<std::size_t>(shape.d[1] * kScale),
	};
}

}  // namespace core

}  // namespace JoshUpscale
