// Copyright 2021 Ivanov Viktor

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "tf_wrappers.h"

namespace upscaler {

constexpr std::size_t INPUT_WIDTH = 480;
constexpr std::size_t INPUT_HEIGHT = 272;
constexpr std::size_t SCALE_FACTOR = 2;
constexpr std::size_t OUTPUT_WIDTH = INPUT_WIDTH * SCALE_FACTOR;
constexpr std::size_t OUTPUT_HEIGHT = INPUT_HEIGHT * SCALE_FACTOR;

struct SUpscaler {
	SUpscaler() = delete;
	explicit SUpscaler(const char *modelPath);
	SUpscaler(const SUpscaler &) = delete;
	SUpscaler(SUpscaler &&) = delete;

	void upscaleFrame(const std::uint8_t *buf, int stride);
	void writeOutput(std::uint8_t *buf, int stride);

private:
	smart::TF_Graph m_Graph;
	::TF_Output m_OutputOp;
	std::vector<::TF_Output> m_InputOp;
	smart::TF_Tensor<float> m_Input;
	smart::TF_Tensor<float> m_LastFrame;
	smart::TF_Tensor<float> m_PreGenTensor;
	smart::TF_Session m_Session;
};

}  // namespace upscaler
