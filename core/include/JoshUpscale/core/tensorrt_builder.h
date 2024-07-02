// Copyright 2023 Ivanov Viktor

#pragma once

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <vector>

#include "JoshUpscale/core.h"

namespace JoshUpscale {

namespace core {

std::vector<std::byte> buildTrtEngine(const std::filesystem::path &modelPath,
    const ::YAML::Node &modelConfig, Quantization quantization);

std::vector<std::byte> buildTrtEngineCached(
    const std::filesystem::path &modelPath, const ::YAML::Node &modelConfig,
    Quantization quantization);

}  // namespace core

}  // namespace JoshUpscale
