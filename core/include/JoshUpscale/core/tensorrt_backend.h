// Copyright 2022 Ivanov Viktor

#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "JoshUpscale/core/cuda.h"
#include "JoshUpscale/core/tensor.h"
#include "JoshUpscale/core/tensorrt.h"

namespace JoshUpscale {

namespace core {

class TensorRTBackend {
public:
	TensorRTBackend(std::span<std::string> inputNames,
	    std::span<std::string> outputNames, std::span<std::byte> engine);

	void processImage(const Tensor &inputImage, const Tensor &outputImage);

private:
	trt::ErrorRecorder m_ErrorRecorder;
	trt::Logger m_Logger;
	int m_Device;
	cuda::CudaBuffer<std::byte> m_InputBuffer;
	cuda::CudaBuffer<std::byte> m_OutputBuffer;
	cuda::CudaBuffer<float> m_InputBufferFp;
	cuda::CudaBuffer<float> m_OutputBufferFp;
	std::vector<cuda::CudaBuffer<float>> m_InterBuffers;
#if TENSORRT_VERSION >= 8501
	std::unordered_map<std::string, void *> m_BindingMaps[2];
#else
	std::vector<void *> m_Bindings[2];
#endif
	int m_BindingsIdx;
	trt::TrtPtr<nvinfer1::ICudaEngine> m_Engine;
	trt::TrtPtr<nvinfer1::IExecutionContext> m_Context;
	cuda::CudaStream m_Stream;
};

}  // namespace core

}  // namespace JoshUpscale
