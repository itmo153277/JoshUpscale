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
    , m_BindingsIdx{0}
    , m_Engine{nullptr}
    , m_Context{nullptr} {
	try {
		auto runtime = trt::TrtPtr{nvinfer1::createInferRuntime(m_Logger)};
		runtime->setErrorRecorder(&m_ErrorRecorder);
		m_Engine = trt::TrtPtr{
		    runtime->deserializeCudaEngine(engine.data(), engine.size())};
		m_Context = trt::TrtPtr{m_Engine->createExecutionContext()};
		auto interBufs = outputNames.size() - 1;
		if (interBufs > 0) {
			for (std::size_t i = 0; i < 2; ++i) {
				m_InterBuffers.emplace_back(1920 * 1080 * 3);
				for (std::size_t j = 0; j < interBufs - 1; ++j) {
					m_InterBuffers.emplace_back(480 * 272 * 3);
				}
			}
		}
		for (std::size_t i = 0; i < 2; ++i) {
			std::unordered_map<std::string, void *> bindingMap;
			bindingMap[inputNames[0]] = m_InputBufferFp.get();
			bindingMap[outputNames[0]] = m_OutputBufferFp.get();
			for (std::size_t j = 0; j < interBufs; ++j) {
				void *ptr1 = m_InterBuffers[j + i * interBufs].get();
				void *ptr2 = m_InterBuffers[j + (i ^ 1) * interBufs].get();
				bindingMap[inputNames[j + 1]] = ptr1;
				bindingMap[outputNames[j + 1]] = ptr2;
			}
			for (std::int32_t j = 0; j < m_Engine->getNbBindings(); ++j) {
				m_Bindings[i].push_back(
				    bindingMap[m_Engine->getBindingName(j)]);
			}
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
		if (!m_Context->enqueueV2(
		        m_Bindings[m_BindingsIdx].data(), m_Stream, nullptr)) {
			throw trt::TrtException();
		}
		cuda::cudaCast(m_OutputBufferFp, m_OutputBuffer, m_Stream);
		cuda::cudaCopy(m_OutputBuffer, outputImage, m_Stream);
		m_Stream.synchronize();
		m_BindingsIdx ^= 1;
	} catch (...) {
		m_ErrorRecorder.rethrowException();
	}
}

}  // namespace core

}  // namespace JoshUpscale
