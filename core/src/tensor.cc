// Copyright 2022 Ivanov Viktor

#include "JoshUpscale/core/tensor.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>

#ifdef _MSC_VER
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace JoshUpscale {

namespace core {

namespace detail {

namespace {

constexpr std::size_t defaultDataAlign = 16;

void *alignedAlloc(std::size_t alignment, std::size_t size) {
#if defined(_MSC_VER)
	return _aligned_malloc(size, alignment);
#else
	return std::aligned_alloc(alignment, size);
#endif
}

void alignedFree(void *ptr) {
#if defined(_MSC_VER)
	_aligned_free(ptr);
#else
	std::free(ptr);
#endif
}

void copy2dFast(std::size_t lineSize, std::size_t height, std::byte *src,
    std::ptrdiff_t srcStride, std::byte *dst, std::ptrdiff_t dstStride) {
	for (std::size_t y = 0, yLen = height; y < yLen;
	     ++y, src += srcStride, dst += dstStride) {
		std::memcpy(dst, src, lineSize);
	}
}

}  // namespace

template <std::size_t alignment = defaultDataAlign>
struct AlignedStorage : TensorStorage {
	explicit AlignedStorage(std::size_t size)
	    : TensorStorage(size,
	          reinterpret_cast<std::byte *>(alignedAlloc(alignment, size))) {
	}

	~AlignedStorage() {
		if (m_Ptr != nullptr) {
			alignedFree(m_Ptr);
			m_Ptr = nullptr;
		}
	}
};

std::unique_ptr<TensorStorage> allocPlainTensor(
    std::size_t width, std::size_t height) {
	return std::make_unique<AlignedStorage<>>(
	    width * height * 3 * sizeof(std::byte));
}

}  // namespace detail

void Tensor::copyTo(Tensor *dst) const {
	assert(m_Width == dst->m_Width && m_Height == dst->m_Height);
	if (isPlain() && dst->isPlain()) {
		std::memcpy(dst->m_Data, m_Data, getSize());
		return;
	}
	detail::copy2dFast(m_Width * 3 * sizeof(std::byte), m_Height, m_Data,
	    m_Stride, dst->m_Data, dst->m_Stride);
}

}  // namespace core

}  // namespace JoshUpscale
