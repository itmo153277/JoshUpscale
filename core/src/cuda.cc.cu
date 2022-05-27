// Copyright 2022 Ivanov Viktor

#include <cassert>
#include <cstdint>

#include "JoshUpscale/core/cuda.h"

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

template <typename T>
__global__ void castKernel<std::byte, T>(
    std::byte *from, T *to, unsigned int numElements) {
	auto srcIdx = blockIdx.x * kBlockSize + threadIdx.x;
	auto dstIdx = srcIdx * 4;
	if (dstIdx >= numElements) {
		return;
	}
	// Assume no UB
	auto *srcPtr = reinterpret_cast<uchar4 *>(from) + srcIdx;
	auto *dstPtr = to + dstIdx;
	dstPtr[0] = static_cast<T>(srcPtr->w);
	dstPtr[1] = static_cast<T>(srcPtr->x);
	dstPtr[2] = static_cast<T>(srcPtr->y);
	dstPtr[3] = static_cast<T>(srcPtr->z);
}

#define DECLARE_SPEC(From, To)                     \
	template __global__ void castKernel<From, To>( \
	    From * from, To * to, unsigned int numElements)

DECLARE_SPEC(std::byte, float);
DECLARE_SPEC(float, std::byte);

#undef DECLARE_SPEC

}  // namespace

template <typename T>
void cudaCast(const CudaBuffer<std::byte> &from, const CudaBuffer<T> &to,
    const CudaStream &stream) {
	constexpr unsigned int kDataPerBlock = kBlockSize * 4;
	assert(from.getSize() == to.getSize());
	assert(from.getSize() % ALIGN_SIZE == 0);
	auto numElements = static_cast<unsigned int>(from.getSize());
	auto numBlocks = static_cast<unsigned int>(
	    (numElements + kDataPerBlock - 1) / kDataPerBlock);
	castKernel<std::byte, T><<<numBlocks, kBlockSize, 0, stream>>>(
	    from.get(), to.get(), numElements);
	cudaCheck(::cudaGetLastError());
}

template <typename T>
void cudaCast(const CudaBuffer<T> &from, const CudaBuffer<std::byte> &to,
    const CudaStream &stream) {
	assert(from.getSize() == to.getSize());
	assert(from.getSize() % ALIGN_SIZE == 0);
	auto numElements = static_cast<unsigned int>(to.getSize());
	auto numBlocks =
	    static_cast<unsigned int>((numElements + kBlockSize - 1) / kBlockSize);
	castKernel<T, std::byte><<<numBlocks, kBlockSize, 0, stream>>>(
	    from.get(), to.get(), numElements);
	cudaCheck(::cudaGetLastError());
}

template <typename T>
void cudaCopy(
    const Tensor &from, const CudaBuffer<T> &to, const CudaStream &stream) {
	assert(from.getSize() <= to.getByteSize());
	if (from.isPlain()) {
		cudaCheck(::cudaMemcpyAsync(to.get(), from.data(), from.getSize(),
		    ::cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
		return;
	}
	std::size_t lineLength = from.getWidth() * 3 * sizeof(std::byte);
	cudaCheck(::cudaMemcpy2DAsync(to.get(), lineLength, from.data(),
	    static_cast<std::size_t>(from.getStride()) * sizeof(std::byte),
	    lineLength, from.getHeight(), ::cudaMemcpyKind::cudaMemcpyHostToDevice,
	    stream));
}

template <typename T>
void cudaCopy(
    const CudaBuffer<T> &from, const Tensor &to, const CudaStream &stream) {
	assert(from.getByteSize() >= to.getSize());
	if (to.isPlain()) {
		cudaCheck(::cudaMemcpyAsync(to.data(), from.get(), to.getSize(),
		    ::cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
		return;
	}
	std::size_t lineLength = to.getWidth() * 3 * sizeof(std::byte);
	cudaCheck(::cudaMemcpy2DAsync(to.data(),
	    static_cast<std::size_t>(to.getStride()) * sizeof(std::byte),
	    from.get(), lineLength, lineLength, to.getHeight(),
	    ::cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
}

template <typename From, typename To>
void cudaCopy(const CudaBuffer<From> &from, const CudaBuffer<To> &to,
    const CudaStream &stream) {
	assert(from.getByteSize() == to.getByteSize());
	cudaCheck(::cudaMemcpyAsync(to.data(), from.data(), to.getByteSize(),
	    ::cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
	return;
}

}  // namespace cuda

}  // namespace core

}  // namespace JoshUpscale
