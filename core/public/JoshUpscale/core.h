// Copyright 2022 Ivanov Viktor

#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

#include "JoshUpscale/core/export.h"

namespace JoshUpscale {

namespace core {

inline constexpr std::size_t INPUT_WIDTH = 480;
inline constexpr std::size_t INPUT_HEIGHT = 270;
inline constexpr std::size_t OUTPUT_WIDTH = 1920;
inline constexpr std::size_t OUTPUT_HEIGHT = 1080;

struct Image {
	void *ptr;
	std::ptrdiff_t stride;
	std::size_t width;
	std::size_t height;
};

struct Runtime {
	virtual ~Runtime() {
	}

	virtual void processImage(
	    const Image &inputImage, const Image &outputImage) = 0;
};

JOSHUPSCALE_EXPORT Runtime *createRuntime(
    int deviceId, const std::filesystem::path &modelPath);

}  // namespace core

}  // namespace JoshUpscale
