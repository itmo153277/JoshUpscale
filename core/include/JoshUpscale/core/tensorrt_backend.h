// Copyright 2025 Ivanov Viktor

#pragma once

#include <NvInfer.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "JoshUpscale/core/cuda.h"
#include "JoshUpscale/core/tensor.h"
#include "JoshUpscale/core/tensorrt.h"

namespace JoshUpscale {

namespace core {

class TensorRTBackend {
	explicit TensorRTBackend(std::span<std::byte> engine);

	void process(
	    const GenericTensor &inputTensor, const GenericTensor &outputTensor);

	std::size_t getInputWidth() const;
	std::size_t getInputHeight() const;
	std::size_t getOutputWidth() const;
	std::size_t getOutputHeight() const;

private:
	trt::ErrorRecorder m_ErrorRecorder;
	trt::Logger m_Logger;
	int m_Device;
	cuda::CudaBuffer<std::uint8_t> m_DeviceMemory;
	cuda::CudaBuffer<std::uint8_t> m_InputBuffer;
	cuda::GenericCudaBuffer m_InputBufferFp;
	cuda::CudaBuffer<std::uint8_t> m_OutputBuffer;
	cuda::GenericCudaBuffer m_OutputBufferFp;
	std::vector<cuda::GenericCudaBuffer> m_InterBuffers;
	std::vector<cuda::GenericCudaBuffer> m_ExtraBuffers;
	int m_BindingsIdx;
	trt::TrtPtr<::nvinfer1::ICudaEngine> m_Engine;
	trt::TrtPtr<::nvinfer1::IExecutionContext> m_Contexts[2];
	cuda::CudaGraph m_CudaGraphs[2];
	cuda::CudaGraphExec m_CudaGraphExec[2];
	cuda::CudaStream m_Stream;
};

}  // namespace core

}  // namespace JoshUpscale
