// Copyright 2022 Ivanov Viktor

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

#include "JoshUpscale/core/tensor.h"

namespace JoshUpscale {

namespace core {

namespace cuda {

constexpr int WARP_SIZE = 32;
constexpr int ALIGN_SIZE = WARP_SIZE * 4;

struct CudaException : std::runtime_error {
	CudaException() : CudaException("CUDA general failure") {
	}
	explicit CudaException(::cudaError_t error)
	    : CudaException(::cudaGetErrorString(error)) {
	}
	explicit CudaException(const char *msg) : std::runtime_error(msg) {
	}
};

inline void cudaCheck(::cudaError_t error) {
	if (error != ::cudaSuccess) {
		throw CudaException(error);
	}
}

inline std::size_t getPaddedSize(
    std::size_t size, std::size_t align = ALIGN_SIZE) {
	return ((size + align - 1) / align) * align;
}

inline int getDevice() {
	int device;
	cudaCheck(::cudaGetDevice(&device));
	return device;
}

namespace detail {

template <typename T>
struct CudaDeleter {
	void operator()(T *ptr) {
		::cudaFree(ptr);
	}
};

}  // namespace detail

template <typename T>
struct CudaBuffer : std::unique_ptr<T, detail::CudaDeleter<T>> {
	using unique_ptr = std::unique_ptr<T, detail::CudaDeleter<T>>;

	explicit CudaBuffer(std::size_t size)
	    : unique_ptr(alloc(size)), m_Size(size) {
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
	std::size_t getByteSize() const {
		return m_Size * sizeof(T);
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
	explicit CudaStream(std::nullptr_t) {
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

	void synchronize() const {
		cudaCheck(::cudaStreamSynchronize(m_Stream));
	}

private:
	::cudaStream_t m_Stream = nullptr;
};

template <typename From, typename To>
void cudaCast(const CudaBuffer<From> &from, const CudaBuffer<To> &to,
    const CudaStream &stream);

template <typename From, typename To>
void cudaCopy(const From &from, const To &to, const CudaStream &stream);

template <typename From, typename To>
void cudaCopy(const CudaBuffer<From> &from, const CudaBuffer<To> &to,
    const CudaStream &stream, std::size_t size);

struct DeviceContext {
	explicit DeviceContext(int device) {
		cudaCheck(::cudaGetDevice(&m_Device));
		::cudaSetDevice(device);
	}
	~DeviceContext() {
		::cudaSetDevice(m_Device);
	}

private:
	int m_Device;
};

}  // namespace cuda

}  // namespace core

}  // namespace JoshUpscale
