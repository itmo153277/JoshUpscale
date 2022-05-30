// Copyright 2022 Ivanov Viktor

#include "JoshUpscale/core/tensorrt_backend.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "JoshUpscale/core/cuda.h"
#include "JoshUpscale/core/tensorrt.h"

namespace JoshUpscale {

namespace core {

TensorRTBackend::TensorRTBackend(const std::vector<std::string> &inputNames,
    const std::vector<std::string> &outputNames, std::span<std::byte> engine)
    : m_Device{cuda::getDevice()}
    , m_InputBuffer{cuda::getPaddedSize(480 * 270 * 3)}
    , m_OutputBuffer{cuda::getPaddedSize(1920 * 1080 * 3)}
    , m_InputBufferFp{cuda::getPaddedSize(480 * 270 * 3)}
    , m_OutputBufferFp{cuda::getPaddedSize(1920 * 1080 * 3)}
    , m_Engine{nullptr}
    , m_Context{nullptr} {
	try {
		auto runtime = trt::TrtPtr{nvinfer1::createInferRuntime(m_Logger)};
		runtime->setErrorRecorder(&m_ErrorRecorder);
		m_Engine = trt::TrtPtr{
		    runtime->deserializeCudaEngine(engine.data(), engine.size())};
		m_Context = trt::TrtPtr{m_Engine->createExecutionContext()};
		std::unordered_map<std::string, void *> bindingMap;
		bindingMap[inputNames[0]] = m_InputBufferFp.get();
		bindingMap[outputNames[0]] = m_OutputBufferFp.get();
		if (outputNames.size() > 1) {
			m_InterBuffers.emplace_back(1920 * 1080 * 3);
			for (std::size_t i = 2; i < outputNames.size(); ++i) {
				m_InterBuffers.emplace_back(480 * 270 * 3);
			}
		}
		for (std::size_t i = 1; i < outputNames.size(); ++i) {
			bindingMap[inputNames[i]] = m_InterBuffers[i - 1].get();
			bindingMap[outputNames[i]] = m_InterBuffers[i - 1].get();
		}
		for (std::int32_t i = 0; i < m_Engine->getNbBindings(); ++i) {
			m_Bindings.push_back(bindingMap[m_Engine->getBindingName(i)]);
		}
	} catch (...) {
		m_ErrorRecorder.rethrowException();
	}
}

void TensorRTBackend::processImage(
    const Tensor &inputImage, const Tensor &outputImage) {
	cuda::DeviceContext cudaCtx(m_Device);

	try {
		cuda::cudaCopy(inputImage, m_InputBuffer, m_Stream);
		cuda::cudaCast(m_InputBuffer, m_InputBufferFp, m_Stream);
		if (!m_Context->enqueueV2(m_Bindings.data(), m_Stream, nullptr)) {
			throw trt::TrtException();
		}
		cuda::cudaCast(m_OutputBufferFp, m_OutputBuffer, m_Stream);
		cuda::cudaCopy(m_OutputBuffer, outputImage, m_Stream);
		m_Stream.synchronize();
	} catch (...) {
		m_ErrorRecorder.rethrowException();
	}
}

}  // namespace core

}  // namespace JoshUpscale
