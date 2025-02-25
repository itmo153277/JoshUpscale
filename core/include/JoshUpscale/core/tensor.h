// Copyright 2025 Ivanov Viktor

#pragma once

#include <cstddef>
#include <variant>

#include "JoshUpscale/core.h"

namespace JoshUpscale {

namespace core {

namespace detail {

constexpr std::ptrdiff_t getPlainStride(std::size_t width) {
	return static_cast<std::ptrdiff_t>(width) * 3;
}
}  // namespace detail

class Tensor {
public:
	explicit Tensor(const Image &img)
	    : m_Data(reinterpret_cast<char *>(img.ptr))
	    , m_Stride(img.stride)
	    , m_Width(img.width)
	    , m_Height(img.height) {
	}

	// Default-copyable, default-movable
	Tensor(const Tensor &) = default;
	Tensor(Tensor &&) noexcept = default;
	Tensor &operator=(const Tensor &) = default;
	Tensor &operator=(Tensor &&) noexcept = default;

	void copyTo(Tensor *dst) const;

	bool isPlain() const {
		return m_Stride == detail::getPlainStride(m_Width);
	}
	std::size_t getSize() const {
		return m_Width * m_Height * 3;
	}
	std::size_t getByteSize() const {
		return getSize() * sizeof(char);
	}
	std::size_t getWidth() const {
		return m_Width;
	}
	std::size_t getHeight() const {
		return m_Height;
	}
	std::ptrdiff_t getStride() const {
		return m_Stride;
	}
	char *data() const {
		return m_Data;
	}

private:
	char *m_Data;
	std::ptrdiff_t m_Stride;
	std::size_t m_Width;
	std::size_t m_Height;
};

struct CpuTensor : Tensor {
	explicit CpuTensor(const Image &img) : Tensor(img) {
	}

	// Default-copyable, default-movable
	CpuTensor(const CpuTensor &) = default;
	CpuTensor(CpuTensor &&) noexcept = default;
	CpuTensor &operator=(const CpuTensor &) = default;
	CpuTensor &operator=(CpuTensor &&) noexcept = default;
};

struct CudaTensor : Tensor {
	explicit CudaTensor(const Image &img) : Tensor(img) {
	}

	// Default-copyable, default-movable
	CudaTensor(const CudaTensor &) = default;
	CudaTensor(CudaTensor &&) noexcept = default;
	CudaTensor &operator=(const CudaTensor &) = default;
	CudaTensor &operator=(CudaTensor &&) noexcept = default;
};

class GenericTensor {
public:
	explicit GenericTensor(const Image &img)
	    : m_Location(img.location), m_Tensor{getTensor(img)} {
	}

	// Default-copyable, default-movable
	GenericTensor(const GenericTensor &) = default;
	GenericTensor(GenericTensor &&) noexcept = default;
	GenericTensor &operator=(const GenericTensor &) = default;
	GenericTensor &operator=(GenericTensor &&) noexcept = default;

	CpuTensor &getCpuTensor() {
		return std::get<CpuTensor>(m_Tensor);
	}
	const CpuTensor &getCpuTensor() const {
		return std::get<CpuTensor>(m_Tensor);
	}
	CudaTensor &getCudaTensor() {
		return std::get<CudaTensor>(m_Tensor);
	}
	const CudaTensor &getCudaTensor() const {
		return std::get<CudaTensor>(m_Tensor);
	}
	DataLocation getLocation() const {
		return m_Location;
	}

private:
	using TensorVarint = std::variant<CpuTensor, CudaTensor, std::nullptr_t>;

	DataLocation m_Location;
	TensorVarint m_Tensor;

	static TensorVarint getTensor(const Image &img) {
		switch (img.location) {
		case DataLocation::CPU:
			return CpuTensor(img);
		case DataLocation::CUDA:
			return CudaTensor(img);
		default:
			return nullptr;
		}
	}
};

}  // namespace core

}  // namespace JoshUpscale
