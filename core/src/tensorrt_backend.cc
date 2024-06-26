// Copyright 2022 Ivanov Viktor

#include "JoshUpscale/core/tensorrt_backend.h"

#include <cstddef>
#include <span>
#include <string>

#include "JoshUpscale/core/cuda.h"
#include "JoshUpscale/core/tensorrt.h"

namespace JoshUpscale {

namespace core {

TensorRTBackend::TensorRTBackend(std::span<std::string> inputNames,
    std::span<std::string> outputNames, std::span<std::byte> engine)
    : m_Device{cuda::getDevice()}
    , m_InputBuffer{cuda::getPaddedSize(480ULL * 270ULL * 3)}
    , m_OutputBuffer{cuda::getPaddedSize(1920ULL * 1080ULL * 3)}
    , m_InputBufferFp{cuda::getPaddedSize(480ULL * 270ULL * 3)}
    , m_OutputBufferFp{cuda::getPaddedSize(1920ULL * 1080ULL * 3)}
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
				m_InterBuffers.emplace_back(1920ULL * 1080ULL * 3ULL);
				for (std::size_t j = 0; j < interBufs - 1; ++j) {
					m_InterBuffers.emplace_back(480ULL * 272ULL * 3ULL);
				}
			}
		}
#if TENSORRT_VERSION >= 8501
		m_Context->setTensorAddress(
		    inputNames[0].c_str(), m_InputBufferFp.get());
		m_Context->setTensorAddress(
		    outputNames[0].c_str(), m_OutputBufferFp.get());
		for (std::size_t i = 0; i < 2; ++i) {
			for (std::size_t j = 0; j < interBufs; ++j) {
				void *ptr1 = m_InterBuffers[j + i * interBufs].get();
				void *ptr2 = m_InterBuffers[j + (i ^ 1) * interBufs].get();
				m_BindingMaps[i][inputNames[j + 1]] = ptr1;
				m_BindingMaps[i][outputNames[j + 1]] = ptr2;
			}
		}
#else
		for (std::size_t i = 0; i < 2; ++i) {
			m_Bindings[i].resize(m_Engine->getNbBindings());
			m_Bindings[i][m_Engine->getBindingIndex(inputNames[0].c_str())] =
			    m_InputBufferFp.get();
			m_Bindings[i][m_Engine->getBindingIndex(outputNames[0].c_str())] =
			    m_OutputBufferFp.get();
			for (std::size_t j = 0; j < interBufs; ++j) {
				void *ptr1 = m_InterBuffers[j + i * interBufs].get();
				void *ptr2 = m_InterBuffers[j + (i ^ 1) * interBufs].get();
				m_Bindings[i][m_Engine->getBindingIndex(
				    inputNames[j + 1].c_str())] = ptr1;
				m_Bindings[i][m_Engine->getBindingIndex(
				    outputNames[j + 1].c_str())] = ptr2;
			}
		}
#endif
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
#if TENSORRT_VERSION >= 8501
		for (const auto &[tensorName, tensorAddr] :
		    m_BindingMaps[m_BindingsIdx]) {
			if (!m_Context->setTensorAddress(tensorName.c_str(), tensorAddr)) {
				throw trt::TrtException();
			}
		}
		if (!m_Context->enqueueV3(m_Stream)) {
			throw trt::TrtException();
		}
#else
		if (!m_Context->enqueueV2(
		        m_Bindings[m_BindingsIdx].data(), m_Stream, nullptr)) {
			throw trt::TrtException();
		}
#endif
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
