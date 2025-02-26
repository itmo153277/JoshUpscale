// Copyright 2022 Ivanov Viktor

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

#include "JoshUpscale/core/export.h"

namespace JoshUpscale {

namespace core {

enum class DataLocation : std::uint8_t { CPU, CUDA };

struct Image {
	void *ptr;
	DataLocation location;
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
