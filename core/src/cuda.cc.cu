// Copyright 2022 Ivanov Viktor

#include <algorithm>
#include <cassert>
#include <cstddef>

#include "JoshUpscale/core/cuda.h"
#include "JoshUpscale/core/tensor.h"

namespace JoshUpscale {

namespace core {

namespace cuda {

constexpr unsigned int kBlockSize = 512;

namespace {

template <typename From, typename To>
__global__ void castKernel(From *from, To *to, unsigned int numElements) {
	auto idx = blockIdx.x * kBlockSize + threadIdx.x;
	if (idx >= numElements) {
		return;
	}
	auto *srcPtr = from + idx;
	auto *dstPtr = to + idx;
	*dstPtr = static_cast<To>(*srcPtr);
}

#define DECLARE_SPEC(From, To)                     \
	template __global__ void castKernel<From, To>( \
	    From * from, To * to, unsigned int numElements)

DECLARE_SPEC(std::byte, float);
DECLARE_SPEC(float, std::byte);

#undef DECLARE_SPEC

}  // namespace

template <typename From, typename To>
void cudaCast(const CudaBuffer<From> &from, const CudaBuffer<To> &to,
    const CudaStream &stream) {
	assert(from.getSize() == to.getSize());
	assert(from.getSize() % ALIGN_SIZE == 0);
	auto numElements = static_cast<unsigned int>(to.getSize());
	auto numBlocks =
	    static_cast<unsigned int>((numElements + kBlockSize - 1) / kBlockSize);
	castKernel<From, To><<<numBlocks, kBlockSize, 0, stream>>>(
	    from.get(), to.get(), numElements);
	cudaCheck(::cudaGetLastError());
}

#define DECLARE_SPEC(From, To)                                     \
	template void cudaCast<From, To>(const CudaBuffer<From> &from, \
	    const CudaBuffer<To> &to, const CudaStream &stream);

DECLARE_SPEC(std::byte, float);
DECLARE_SPEC(float, std::byte);

#undef DECLARE_SPEC

namespace detail {

namespace {

template <typename T>
void cudaCopy(
    const Tensor &from, const CudaBuffer<T> &to, const CudaStream &stream) {
	std::size_t size = from.getSize();
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
    const CudaBuffer<T> &from, const Tensor &to, const CudaStream &stream) {
	std::size_t size = to.getSize();
	assert(from.getByteSize() >= size && to.getSize() >= size);
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

template <typename From, typename To>
void cudaCopy(const CudaBuffer<From> &from, const CudaBuffer<To> &to,
    const CudaStream &stream) {
	cudaCopy(from, to, stream, std::min(from.getByteSize(), to.getByteSize()));
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

DECLARE_SPEC(Tensor, CudaBuffer<std::byte>);
DECLARE_SPEC(CudaBuffer<std::byte>, Tensor);
DECLARE_SPEC(CudaBuffer<std::byte>, CudaBuffer<std::byte>);
DECLARE_SPEC(CudaBuffer<float>, CudaBuffer<float>);

#undef DECLARE_SPEC

template <typename From, typename To>
void cudaCopy(const CudaBuffer<From> &from, const CudaBuffer<To> &to,
    const CudaStream &stream, std::size_t size) {
	assert(from.getByteSize() >= size && to.getByteSize() >= size);
	cudaCheck(::cudaMemcpyAsync(to.get(), from.get(), size,
	    ::cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
}

#define DECLARE_SPEC(From, To)                                     \
	template void cudaCopy<From, To>(const CudaBuffer<From> &from, \
	    const CudaBuffer<From> &to, const CudaStream &stream,      \
	    std::size_t size);

DECLARE_SPEC(std::byte, std::byte);
DECLARE_SPEC(float, float);

#undef DECLARE_SPEC

}  // namespace cuda

}  // namespace core

}  // namespace JoshUpscale
