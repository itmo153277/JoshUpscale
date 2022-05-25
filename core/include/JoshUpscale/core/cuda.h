// Copyright 2022 Ivanov Viktor

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstddef>
#include <exception>
#include <memory>
#include <utility>

#include "JoshUpscale/core/tensor.h"

namespace JoshUpscale {

namespace core {

namespace cuda {

struct CudaException : std::exception {
	CudaException() : CudaException("CUDA general failure") {
	}
	explicit CudaException(::cudaError_t error)
	    : CudaException(::cudaGetErrorString(error)) {
	}
	explicit CudaException(const char *msg) : std::exception(msg) {
	}
};

inline void cudaCheck(::cudaError_t error) {
	if (error != ::cudaSuccess) {
		throw CudaException(error);
	}
}

template <typename T>
struct CudaBuffer : std::unique_ptr<T, decltype(&::cudaFree)> {
	using unique_ptr = std::unique_ptr<T, decltype(&::cudaFree)>;

	explicit CudaBuffer(std::size_t size)
	    : unique_ptr(alloc(size), &::cudaFree), m_Size(size) {
	}

	// Non-copyable, default-movable
	CudaBuffer(const CudaBuffer &) = delete;
	CudaBuffer(CudaBuffer &&) noexcept = default;
	CudaBuffer &operator=(const CudaBuffer &) = delete;
	CudaBuffer &operator=(CudaBuffer &&) noexcept = default;

	T &operator*() = delete;
	T *operator->() = delete;

	std::size_t getSize() const {
		return m_Size;
	}

private:
	std::size_t m_Size;

	static T *alloc(std::size_t size) {
		void *result;
		cudaCheck(::cudaMalloc(&result, size * sizeof(T)));
		return reinterpret_cast<T *>(result);
	}
};

class CudaStream {
public:
	CudaStream() {
		cudaCheck(::cudaStreamCreate(&m_Stream));
	}
	explicit CudaStream(nullptr_t) {
	}
	~CudaStream() {
		if (m_Stream != nullptr) {
			::cudaStreamDestroy(m_Stream);
		}
	}

	// Non-copyable, movable
	CudaStream(const CudaStream &) = delete;
	CudaStream(CudaStream &&s) noexcept {
		m_Stream = s.m_Stream;
		s.m_Stream = nullptr;
	}
	CudaStream &operator=(const CudaStream &) = delete;
	CudaStream &operator=(CudaStream &&s) noexcept {
		if (this != &s) {
			this->~CudaStream();
			new (this) CudaStream(std::move(s));
		}
		return *this;
	}

	::cudaStream_t get() const {
		return m_Stream;
	}
	operator ::cudaStream_t() const {
		return m_Stream;
	}

private:
	::cudaStream_t m_Stream = nullptr;
};

template <typename From, typename To>
void cudaCast(const CudaBuffer<From> &from, const CudaBuffer<To> &to,
    const CudaStream &stream);

void cudaCopy(const Tensor &from, const CudaBuffer<std::uint8_t> &to,
    const CudaStream &stream);
void cudaCopy(const CudaBuffer<std::uint8_t> &from, const Tensor &to,
    const CudaStream &stream);

}  // namespace cuda

}  // namespace core

}  // namespace JoshUpscale
