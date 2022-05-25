// Copyright 2022 Ivanov Viktor

#include "JoshUpscale/core/cuda.h"

#include <cassert>
#include <cstddef>

#include "JoshUpscale/core/tensor.h"

namespace JoshUpscale {

namespace core {

namespace cuda {

void cudaCopy(const Tensor &from, const CudaBuffer<std::uint8_t> &to,
    const CudaStream &stream) {
	assert(from.getSize() == to.getSize());
	if (from.isPlain()) {
		cudaCheck(::cudaMemcpyAsync(to.get(), from.data(), to.getSize(),
		    ::cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
		return;
	}
	std::size_t lineLength = from.getWidth() * 3 * sizeof(std::byte);
	cudaCheck(::cudaMemcpy2DAsync(to.get(), lineLength, from.data(),
	    static_cast<std::size_t>(from.getStride()) * sizeof(std::byte),
	    lineLength, from.getHeight(), ::cudaMemcpyKind::cudaMemcpyHostToDevice,
	    stream));
}

void cudaCopy(const CudaBuffer<std::uint8_t> &from, const Tensor &to,
    const CudaStream &stream) {
	assert(from.getSize() == to.getSize());
	if (to.isPlain()) {
		cudaCheck(::cudaMemcpyAsync(to.data(), from.get(), from.getSize(),
		    ::cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
		return;
	}
	std::size_t lineLength = to.getWidth() * 3 * sizeof(std::byte);
	cudaCheck(::cudaMemcpy2DAsync(to.data(),
	    static_cast<std::size_t>(to.getStride()) * sizeof(std::byte),
	    from.get(), lineLength, lineLength, to.getHeight(),
	    ::cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
}

}  // namespace cuda

}  // namespace core

}  // namespace JoshUpscale
