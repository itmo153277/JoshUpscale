// Copyright 2023 Ivanov Viktor

#pragma once

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <vector>

namespace JoshUpscale {

namespace core {

std::vector<std::byte> buildTrtEngine(
    const std::filesystem::path &modelPath, const ::YAML::Node &modelConfig);

std::vector<std::byte> buildTrtEngineCached(
    const std::filesystem::path &modelPath, const ::YAML::Node &modelConfig);

}  // namespace core

}  // namespace JoshUpscale
