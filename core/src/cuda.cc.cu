// Copyright 2022 Ivanov Viktor

#include <algorithm>
#include <cassert>
#include <cstddef>

#include "JoshUpscale/core.h"
#include "JoshUpscale/core/cuda.h"
#include "JoshUpscale/core/tensor.h"

namespace JoshUpscale {

namespace core {

namespace cuda {

namespace detail {

namespace {

constexpr unsigned int kBlockSize = 512;

template <typename From, typename To>
__global__ void castKernel(From *from, To *to, unsigned int numElements) {
	auto idx = (blockIdx.x * kBlockSize) + threadIdx.x;
	if (idx >= numElements) {
		return;
	}
	auto *srcPtr = from + idx;
	auto *dstPtr = to + idx;
	*dstPtr = static_cast<To>(*srcPtr);
}

#define DECLARE_SPEC(From, To)                     \
	template __global__ void castKernel<From, To>( \
	    From * from, To * to, unsigned int numElements)  // NOLINT

DECLARE_SPEC(char, float);
DECLARE_SPEC(float, char);
DECLARE_SPEC(__half, char);
DECLARE_SPEC(char, __half);
DECLARE_SPEC(__half, float);
DECLARE_SPEC(float, __half);

#undef DECLARE_SPEC

template <typename T>
struct UnmanagedCudaBuffer {
	std::size_t size;
	T *ptr;

	std::size_t getByteSize() {
		return size * sizeof(T);
	}
};

template <typename T>
UnmanagedCudaBuffer<T> toUnmanaged(const CudaBuffer<T> &s) {
	return {s.getSize(), s.get()};
}

UnmanagedCudaBuffer<char> toUnmanaged(const CudaTensor &s) {
	return {s.getSize(), s.data()};
}

template <typename From, typename To>
void cudaCast(const UnmanagedCudaBuffer<From> &from,
    const UnmanagedCudaBuffer<To> &to, const CudaStream &stream) {
	assert(from.size == to.size);
	assert(from.size % ALIGN_SIZE == 0);
	auto numElements = static_cast<unsigned int>(to.size);
	auto numBlocks =
	    static_cast<unsigned int>((numElements + kBlockSize - 1) / kBlockSize);
	castKernel<From, To>
	    <<<numBlocks, kBlockSize, 0, stream>>>(from.ptr, to.ptr, numElements);
	cudaCheck(::cudaGetLastError());
}

}  // namespace

}  // namespace detail

template <typename From, typename To>
void cudaCast(const From &from, const To &to, const CudaStream &stream) {
	return detail::cudaCast(
	    detail::toUnmanaged(from), detail::toUnmanaged(to), stream);
}

#define DECLARE_SPEC(From, To)        \
	template void cudaCast<From, To>( \
	    const From &from, const To &to, const CudaStream &stream);

DECLARE_SPEC(CudaBuffer<char>, CudaBuffer<float>);
DECLARE_SPEC(CudaBuffer<float>, CudaBuffer<char>);
DECLARE_SPEC(CudaBuffer<__half>, CudaBuffer<float>);
DECLARE_SPEC(CudaBuffer<float>, CudaBuffer<__half>);
DECLARE_SPEC(CudaBuffer<char>, CudaBuffer<__half>);
DECLARE_SPEC(CudaBuffer<__half>, CudaBuffer<char>);
DECLARE_SPEC(CudaTensor, CudaBuffer<float>);
DECLARE_SPEC(CudaTensor, CudaBuffer<__half>);
DECLARE_SPEC(CudaBuffer<__half>, CudaTensor);
DECLARE_SPEC(CudaBuffer<float>, CudaTensor);

#undef DECLARE_SPEC

namespace detail {

namespace {

template <typename T>
void cudaCopy(
    const CpuTensor &from, const CudaBuffer<T> &to, const CudaStream &stream) {
	std::size_t size = from.getByteSize();
	assert(from.getSize() >= size && to.getByteSize() >= size);
	if (from.isPlain()) {
		cudaCheck(::cudaMemcpyAsync(to.get(), from.data(), size,
		    ::cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
	} else {
		std::size_t lineLength = from.getWidth() * 3 * sizeof(std::byte);
		cudaCheck(::cudaMemcpy2DAsync(to.get(), lineLength, from.data(),
		    static_cast<std::size_t>(from.getStride()), lineLength,
		    from.getHeight(), ::cudaMemcpyKind::cudaMemcpyHostToDevice,
		    stream));
	}
}

template <typename T>
void cudaCopy(
    const CudaTensor &from, const CudaBuffer<T> &to, const CudaStream &stream) {
	std::size_t size = from.getByteSize();
	assert(from.getSize() >= size && to.getByteSize() >= size);
	if (from.isPlain()) {
		cudaCheck(::cudaMemcpyAsync(to.get(), from.data(), size,
		    ::cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));
	} else {
		std::size_t lineLength = from.getWidth() * 3 * sizeof(std::byte);
		cudaCheck(::cudaMemcpy2DAsync(to.get(), lineLength, from.data(),
		    static_cast<std::size_t>(from.getStride()), lineLength,
		    from.getHeight(), ::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
		    stream));
	}
}

template <typename T>
void cudaCopy(
    const CudaBuffer<T> &from, const CpuTensor &to, const CudaStream &stream) {
	std::size_t size = to.getByteSize();
	assert(from.getByteSize() >= size && to.getByteSize() >= size);
	if (to.isPlain()) {
		cudaCheck(::cudaMemcpyAsync(to.data(), from.get(), size,
		    ::cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
	} else {
		std::size_t lineLength = to.getWidth() * 3 * sizeof(std::byte);
		cudaCheck(::cudaMemcpy2DAsync(to.data(),
		    static_cast<std::size_t>(to.getStride()), from.get(), lineLength,
		    lineLength, to.getHeight(),
		    ::cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
	}
}

template <typename T>
void cudaCopy(
    const CudaBuffer<T> &from, const CudaTensor &to, const CudaStream &stream) {
	std::size_t size = to.getByteSize();
	assert(from.getByteSize() >= size && to.getByteSize() >= size);
	if (to.isPlain()) {
		cudaCheck(::cudaMemcpyAsync(to.data(), from.get(), size,
		    ::cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));
	} else {
		std::size_t lineLength = to.getWidth() * 3 * sizeof(std::byte);
		cudaCheck(::cudaMemcpy2DAsync(to.data(),
		    static_cast<std::size_t>(to.getStride()), from.get(), lineLength,
		    lineLength, to.getHeight(),
		    ::cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));
	}
}

template <typename T>
void cudaCopy(const CudaBuffer<T> &from, const CudaBuffer<T> &to,
    const CudaStream &stream) {
	cudaCopy(from, to, stream, std::min(from.getByteSize(), to.getByteSize()));
}

template <typename T>
void cudaCopy(const CudaBuffer<T> &from, const GenericTensor &to,
    const CudaStream &stream) {
	switch (to.getLocation()) {
	case DataLocation::CPU:
		cudaCopy(from, to.getCpuTensor(), stream);
		return;
	case DataLocation::CUDA:
		cudaCopy(from, to.getCudaTensor(), stream);
		return;
	}
}

template <typename T>
void cudaCopy(const GenericTensor &from, const CudaBuffer<T> &to,
    const CudaStream &stream) {
	switch (from.getLocation()) {
	case DataLocation::CPU:
		cudaCopy(from.getCpuTensor(), to, stream);
		return;
	case DataLocation::CUDA:
		cudaCopy(from.getCudaTensor(), to, stream);
		return;
	}
}

}  // namespace

}  // namespace detail

template <typename From, typename To>
void cudaCopy(const From &from, const To &to, const CudaStream &stream) {
	detail::cudaCopy(from, to, stream);
}

#define DECLARE_SPEC(From, To)        \
	template void cudaCopy<From, To>( \
	    const From &from, const To &to, const CudaStream &stream);

DECLARE_SPEC(CpuTensor, CudaBuffer<char>);
DECLARE_SPEC(CudaTensor, CudaBuffer<char>);
DECLARE_SPEC(GenericTensor, CudaBuffer<char>);
DECLARE_SPEC(CudaBuffer<char>, CpuTensor);
DECLARE_SPEC(CudaBuffer<char>, CudaTensor);
DECLARE_SPEC(CudaBuffer<char>, GenericTensor);
DECLARE_SPEC(CudaBuffer<char>, CudaBuffer<char>);
DECLARE_SPEC(CudaBuffer<float>, CudaBuffer<float>);
DECLARE_SPEC(CudaBuffer<__half>, CudaBuffer<__half>);

#undef DECLARE_SPEC

template <typename T>
void cudaCopy(const CudaBuffer<T> &from, const CudaBuffer<T> &to,
    const CudaStream &stream, std::size_t size) {
	assert(from.getByteSize() >= size && to.getByteSize() >= size);
	cudaCheck(::cudaMemcpyAsync(to.get(), from.get(), size,
	    ::cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
}

#define DECLARE_SPEC(T)                                  \
	template void cudaCopy<T>(const CudaBuffer<T> &from, \
	    const CudaBuffer<T> &to, const CudaStream &stream, std::size_t size);

DECLARE_SPEC(char);
DECLARE_SPEC(float);
DECLARE_SPEC(__half);

#undef DECLARE_SPEC

}  // namespace cuda

}  // namespace core

}  // namespace JoshUpscale
