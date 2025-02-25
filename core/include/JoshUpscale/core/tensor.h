// Copyright 2022 Ivanov Viktor

#pragma once

#include <cstddef>
#include <memory>

#include "JoshUpscale/core.h"

namespace JoshUpscale {

namespace core {

namespace detail {

class TensorStorage {
protected:
	TensorStorage(std::size_t size, std::byte *ptr) : m_Size(size), m_Ptr(ptr) {
	}

public:
	// Non-copyable, non-movable
	TensorStorage(const TensorStorage &) = delete;
	TensorStorage(TensorStorage &&) = delete;
	TensorStorage &operator=(const TensorStorage &) = delete;
	TensorStorage &operator=(TensorStorage &&) = delete;

	virtual ~TensorStorage() {
	}

	std::byte *getPtr() const {
		return m_Ptr;
	}

	std::size_t getSize() const {
		return m_Size;
	}

protected:
	std::size_t m_Size;
	std::byte *m_Ptr;
};

using TensorStoragePtr = std::shared_ptr<TensorStorage>;

std::unique_ptr<TensorStorage> allocPlainTensor(
    std::size_t width, std::size_t height);

constexpr std::ptrdiff_t getPlainStride(std::size_t width) {
	return static_cast<std::ptrdiff_t>(width) * 3;
}

}  // namespace detail

class Tensor {
public:
	explicit Tensor(const Image &img)
	    : m_Data(reinterpret_cast<std::byte *>(img.ptr))
	    , m_Stride(img.stride)
	    , m_Width(img.width)
	    , m_Height(img.height) {
	}
	Tensor(std::size_t width, std::size_t height)
	    : m_Storage(detail::allocPlainTensor(width, height))
	    , m_Data(m_Storage->getPtr())
	    , m_Stride(detail::getPlainStride(width))
	    , m_Width(width)
	    , m_Height(height) {
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
		return m_Width * m_Height * 3 * sizeof(std::byte);
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
	std::byte *data() const {
		return m_Data;
	}

private:
	detail::TensorStoragePtr m_Storage = nullptr;
	std::byte *m_Data;
	std::ptrdiff_t m_Stride;
	std::size_t m_Width;
	std::size_t m_Height;
};

}  // namespace core

}  // namespace JoshUpscale
