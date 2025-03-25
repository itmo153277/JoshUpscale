// Copyright 2025 Viktor Ivanov

#pragma once

#include <JoshUpscale/core.h>

#include <string_view>

namespace JoshUpscale {

namespace obs {

void log(core::LogLevel level, std::string_view format, ...);
void logException() noexcept;

}  // namespace obs

}  // namespace JoshUpscale
