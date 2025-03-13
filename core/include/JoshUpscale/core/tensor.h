// Copyright 2025 Ivanov Viktor

#pragma once

#include <cassert>
#include <cstddef>

#include "JoshUpscale/core.h"
#include "JoshUpscale/core/cuda.h"
#include "JoshUpscale/core/utils.h"

namespace JoshUpscale {

namespace core {

namespace detail {

constexpr std::ptrdiff_t getPlainStride(std::size_t width) {
	return static_cast<std::ptrdiff_t>(width) * 4;
}
}  // namespace detail

class Tensor {
public:
	explicit Tensor(const Image &img)
	    : m_Data(reinterpret_cast<std::uint8_t *>(img.ptr))
	    , m_Stride(img.stride)
	    , m_Width(img.width)
	    , m_Height(img.height) {
	}

	// Default-copyable, default-movable
	Tensor(const Tensor &) = default;
	Tensor(Tensor &&) noexcept = default;
	Tensor &operator=(const Tensor &) = default;
	Tensor &operator=(Tensor &&) noexcept = default;

	bool isPlain() const {
		return m_Stride == detail::getPlainStride(m_Width);
	}
	std::size_t getSize() const {
		return m_Width * m_Height * 4;
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
	std::uint8_t *data() const {
		return m_Data;
	}

private:
	std::uint8_t *m_Data;
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

struct GraphicsResourceTensor {
	explicit GraphicsResourceTensor(const Image &img)
	    : m_Resource(reinterpret_cast<::cudaGraphicsResource_t>(img.ptr)) {
	}

	GraphicsResourceTensor(const GraphicsResourceTensor &) = default;
	GraphicsResourceTensor(GraphicsResourceTensor &&) noexcept = default;
	GraphicsResourceTensor &operator=(const GraphicsResourceTensor &) = default;
	GraphicsResourceTensor &operator=(
	    GraphicsResourceTensor &&) noexcept = default;

	cuda::GraphicsResource getResource(::cudaStream_t stream) const {
		return cuda::GraphicsResource(m_Resource, stream);
	}

private:
	::cudaGraphicsResource_t m_Resource;
};

class GenericTensor {
public:
	explicit GenericTensor(const Image &img) : m_Location(img.location) {
		switch (m_Location) {
		case DataLocation::CPU:
			new (m_Storage.getPtr()) CpuTensor(img);
			break;
		case DataLocation::CUDA:
			new (m_Storage.getPtr()) CudaTensor(img);
			break;
		case DataLocation::GRAPHICS_RESOURCE:
			new (m_Storage.getPtr()) GraphicsResourceTensor(img);
			break;
		default:
			unreachable();
		}
	}

	~GenericTensor() {
		switch (m_Location) {
		case DataLocation::CPU:
			reinterpret_cast<CpuTensor *>(m_Storage.getPtr())->~CpuTensor();
			break;
		case DataLocation::CUDA:
			reinterpret_cast<CudaTensor *>(m_Storage.getPtr())->~CudaTensor();
			break;
		case DataLocation::GRAPHICS_RESOURCE:
			reinterpret_cast<GraphicsResourceTensor *>(m_Storage.getPtr())
			    ->~GraphicsResourceTensor();
			break;
		default:
			unreachable();
		}
	}

	// Default-copyable, default-movable
	GenericTensor(const GenericTensor &) = default;
	GenericTensor(GenericTensor &&) noexcept = default;
	GenericTensor &operator=(const GenericTensor &) = default;
	GenericTensor &operator=(GenericTensor &&) noexcept = default;

	operator CpuTensor &() {
		assert(m_Location == DataLocation::CPU);
		return *reinterpret_cast<CpuTensor *>(m_Storage.getPtr());
	}
	operator const CpuTensor &() const {
		assert(m_Location == DataLocation::CPU);
		return *reinterpret_cast<CpuTensor *>(m_Storage.getPtr());
	}
	operator CudaTensor &() {
		assert(m_Location == DataLocation::CUDA);
		return *reinterpret_cast<CudaTensor *>(m_Storage.getPtr());
	}
	operator const CudaTensor &() const {
		assert(m_Location == DataLocation::CUDA);
		return *reinterpret_cast<CudaTensor *>(m_Storage.getPtr());
	}
	operator GraphicsResourceTensor &() {
		assert(m_Location == DataLocation::GRAPHICS_RESOURCE);
		return *reinterpret_cast<GraphicsResourceTensor *>(m_Storage.getPtr());
	}
	operator const GraphicsResourceTensor &() const {
		assert(m_Location == DataLocation::GRAPHICS_RESOURCE);
		return *reinterpret_cast<GraphicsResourceTensor *>(m_Storage.getPtr());
	}

	DataLocation getLocation() const {
		return m_Location;
	}

private:
	DataLocation m_Location;

	MultiClassStorage<CpuTensor, CudaTensor, GraphicsResourceTensor> m_Storage =
	    {};
};

}  // namespace core

}  // namespace JoshUpscale
