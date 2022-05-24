// Copyright 2022 Ivanov Viktor

#pragma once

#include <cstddef>

#include "JoshUpscale/core/export.h"

namespace JoshUpscale {

namespace core {

struct Image {
	void *ptr;
	std::ptrdiff_t stride;
	std::size_t width;
	std::size_t height;
};

}  // namespace core

}  // namespace JoshUpscale
