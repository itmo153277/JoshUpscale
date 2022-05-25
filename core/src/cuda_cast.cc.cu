// Copyright 2022 Ivanov Viktor

#include <cassert>
#include <cstdint>

#include "JoshUpscale/core/cuda.h"

namespace JoshUpscale {

namespace core {

namespace cuda {

constexpr unsigned int kBlockSize = 512;

__global__ void castKernel(
    std::byte *from, float *to, unsigned int numElements) {
	auto srcIdx = blockIdx.x * kBlockSize + threadIdx.x;
	auto dstIdx = srcIdx * 4;
	if (dstIdx >= numElements) {
		return;
	}
	auto *srcPtr = reinterpret_cast<uchar4 *>(from) + srcIdx;
	auto *dstPtr = to + dstIdx;
	dstPtr[0] = static_cast<float>(srcPtr->w);
	dstPtr[1] = static_cast<float>(srcPtr->x);
	dstPtr[2] = static_cast<float>(srcPtr->y);
	dstPtr[3] = static_cast<float>(srcPtr->z);
}

__global__ void castKernel(
    float *from, std::byte *to, unsigned int numElements) {
	auto idx = blockIdx.x * kBlockSize + threadIdx.x;
	if (idx >= numElements) {
		return;
	}
	auto *srcPtr = from + idx;
	auto *dstPtr = reinterpret_cast<std::uint8_t *>(to) + idx;
	*dstPtr = static_cast<std::uint8_t>(*srcPtr);
}

template <>
void cudaCast(const CudaBuffer<std::byte> &from, const CudaBuffer<float> &to,
    const CudaStream &stream) {
	constexpr unsigned int kDataPerBlock = kBlockSize * 4;
	assert(from.getSize() * 4 == to.getSize());
    assert(to.getSize() % 4 == 0);
	auto numElements =
	    static_cast<unsigned int>(from.getSize() / sizeof(std::byte));
	auto numBlocks = static_cast<unsigned int>(
	    (numElements + kDataPerBlock - 1) / kDataPerBlock);
	castKernel<<<numBlocks, kBlockSize, 0, stream>>>(
	    from.get(), to.get(), numElements);
	cudaCheck(::cudaGetLastError());
}

template <>
void cudaCast(const CudaBuffer<float> &from, const CudaBuffer<std::byte> &to,
    const CudaStream &stream) {
	assert(from.getSize() == to.getSize() * 4);
	auto numElements =
	    static_cast<unsigned int>(to.getSize() / sizeof(std::byte));
	auto numBlocks =
	    static_cast<unsigned int>((numElements + kBlockSize - 1) / kBlockSize);
	castKernel<<<numBlocks, kBlockSize, 0, stream>>>(
	    from.get(), to.get(), numElements);
	cudaCheck(::cudaGetLastError());
}

}  // namespace cuda

}  // namespace core

}  // namespace JoshUpscale
