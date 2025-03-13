// Copyright 2025 Ivanov Viktor

#pragma once

#include <cstdint>
#include <type_traits>

#include "JoshUpscale/core.h"
#include "JoshUpscale/core/cuda.h"
#include "JoshUpscale/core/tensor.h"
#include "JoshUpscale/core/utils.h"

namespace JoshUpscale {

namespace core {

namespace cuda {

template <typename From, typename To>
void cudaCast(const From &from, const To &to, const CudaStream &stream);

template <typename From, typename To>
void cudaCopy(const From &from, const To &to, const CudaStream &stream);

template <typename T>
void cudaConvert(const CpuTensor &from, const CudaBuffer<T> &to,
    const CudaBuffer<std::uint8_t> &internalBuffer, const CudaStream &stream) {
	bool compatible = std::is_same_v<T, std::uint8_t>;
	if constexpr (std::is_same_v<T, DynamicType>) {
		compatible = to.getDataType() == DataType::UINT8;
	}
	if (compatible) {
		cudaCopy(from, to, stream);
	} else {
		cudaCopy(from, internalBuffer, stream);
		cudaCast(internalBuffer, to, stream);
	}
}

template <typename T>
void cudaConvert(const CudaTensor &from, const CudaBuffer<T> &to,
    const CudaBuffer<std::uint8_t> &internalBuffer, const CudaStream &stream) {
	bool compatible = std::is_same_v<T, std::uint8_t>;
	if constexpr (std::is_same_v<T, DynamicType>) {
		compatible = to.getDataType() == DataType::UINT8;
	}
	if (compatible) {
		cudaCopy(from, to, stream);
	} else if (from.isPlain()) {
		cudaCast(from, to, stream);
	} else {
		cudaCopy(from, internalBuffer, stream);
		cudaCast(internalBuffer, to, stream);
	}
}

template <typename T>
void cudaConvert(const GraphicsResource &from, const CudaBuffer<T> &to,
    const CudaBuffer<std::uint8_t> &internalBuffer, const CudaStream &stream) {
	bool compatible = std::is_same_v<T, std::uint8_t>;
	if constexpr (std::is_same_v<T, DynamicType>) {
		compatible = to.getDataType() == DataType::UINT8;
	}
	if (compatible) {
		cudaCopy(from, to, stream);
	} else {
		cudaCopy(from, internalBuffer, stream);
		cudaCast(internalBuffer, to, stream);
	}
}

template <typename T>
void cudaConvert(const GraphicsResourceTensor &from, const CudaBuffer<T> &to,
    const CudaBuffer<std::uint8_t> &internalBuffer, const CudaStream &stream) {
	GraphicsResource resource = from.getResource(stream);
	cudaConvert(resource, to, internalBuffer, stream);
}

template <typename T>
void cudaConvert(const GenericTensor &from, const CudaBuffer<T> &to,
    const CudaBuffer<std::uint8_t> &internalBuffer, const CudaStream &stream) {
	switch (from.getLocation()) {
	case DataLocation::CPU:
		cudaConvert(
		    static_cast<const CpuTensor &>(from), to, internalBuffer, stream);
		break;
	case DataLocation::CUDA:
		cudaConvert(
		    static_cast<const CudaTensor &>(from), to, internalBuffer, stream);
		break;
	case DataLocation::GRAPHICS_RESOURCE:
		cudaConvert(static_cast<const GraphicsResourceTensor &>(from), to,
		    internalBuffer, stream);
		break;
	default:
		unreachable();
		break;
	}
}

template <typename T>
void cudaConvert(const CudaBuffer<T> &from, const CpuTensor &to,
    const CudaBuffer<std::uint8_t> &internalBuffer, const CudaStream &stream) {
	bool compatible = std::is_same_v<T, std::uint8_t>;
	if constexpr (std::is_same_v<T, DynamicType>) {
		compatible = from.getDataType() == DataType::UINT8;
	}
	if (compatible) {
		cudaCopy(from, to, stream);
	} else {
		cudaCast(from, internalBuffer, stream);
		cudaCopy(internalBuffer, to, stream);
	}
}

template <typename T>
void cudaConvert(const CudaBuffer<T> &from, const CudaTensor &to,
    const CudaBuffer<std::uint8_t> &internalBuffer, const CudaStream &stream) {
	bool compatible = std::is_same_v<T, std::uint8_t>;
	if constexpr (std::is_same_v<T, DynamicType>) {
		compatible = from.getDataType() == DataType::UINT8;
	}
	if (compatible) {
		cudaCopy(from, to, stream);
	} else if (to.isPlain()) {
		cudaCast(from, to, stream);
	} else {
		cudaCast(from, internalBuffer, stream);
		cudaCopy(internalBuffer, to, stream);
	}
}

template <typename T>
void cudaConvert(const CudaBuffer<T> &from, const GraphicsResource &to,
    const CudaBuffer<std::uint8_t> &internalBuffer, const CudaStream &stream) {
	bool compatible = std::is_same_v<T, std::uint8_t>;
	if constexpr (std::is_same_v<T, DynamicType>) {
		compatible = from.getDataType() == DataType::UINT8;
	}
	if (compatible) {
		cudaCopy(from, to, stream);
	} else {
		cudaCast(from, internalBuffer, stream);
		cudaCopy(internalBuffer, to, stream);
	}
}

template <typename T>
void cudaConvert(const CudaBuffer<T> &from, const GraphicsResourceTensor &to,
    const CudaBuffer<std::uint8_t> &internalBuffer, const CudaStream &stream) {
	GraphicsResource resource = to.getResource(stream);
	cudaConvert(from, resource, internalBuffer, stream);
}

template <typename T>
void cudaConvert(const CudaBuffer<T> &from, const GenericTensor &to,
    const CudaBuffer<std::uint8_t> &internalBuffer, const CudaStream &stream) {
	switch (to.getLocation()) {
	case DataLocation::CPU:
		cudaConvert(
		    from, static_cast<const CpuTensor &>(to), internalBuffer, stream);
		break;
	case DataLocation::CUDA:
		cudaConvert(
		    from, static_cast<const CudaTensor &>(to), internalBuffer, stream);
		break;
	case DataLocation::GRAPHICS_RESOURCE:
		cudaConvert(from, static_cast<const GraphicsResourceTensor &>(to),
		    internalBuffer, stream);
		break;
	default:
		unreachable();
		break;
	}
}

}  // namespace cuda

}  // namespace core

}  // namespace JoshUpscale
