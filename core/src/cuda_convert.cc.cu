// Copyright 2022 Ivanov Viktor

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "JoshUpscale/core.h"
#include "JoshUpscale/core/cuda.h"
#include "JoshUpscale/core/cuda_convert.h"
#include "JoshUpscale/core/tensor.h"
#include "JoshUpscale/core/utils.h"

namespace JoshUpscale {

namespace core {

namespace cuda {

namespace detail {

namespace {

template <class T2, class T1>
__device__ T2 cuda_bit_cast(T1 t1) {
	T2 t2{};
	std::memcpy(&t2, &t1, sizeof(T1));
	return t2;
}

template <typename T, typename StorageType>
struct BaseDataTypeTraits {
	using storage_type = StorageType;

	__device__ static storage_type store(T val1, T val2, T val3) {
		storage_type val{};
		val.x = val1;
		val.y = val2;
		val.z = val3;
		return val;
	}
	__device__ static T get(T val) {
		return val;
	}
};

template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<std::uint8_t> : BaseDataTypeTraits<std::uint8_t, uchar4> {
};

template <>
struct DataTypeTraits<__half> {
	using storage_type = ushort3;

	__device__ static ushort3 store(__half val1, __half val2, __half val3) {
		return make_ushort3(cuda_bit_cast<std::uint16_t>(val1),
		    cuda_bit_cast<std::uint16_t>(val2),
		    cuda_bit_cast<std::uint16_t>(val3));
	}

	__device__ static __half get(std::uint16_t val) {
		return cuda_bit_cast<__half>(val);
	}
};

template <>
struct DataTypeTraits<float> : BaseDataTypeTraits<float, float3> {};

template <typename From, typename To>
struct CastTraits {
	__device__ static To convert(From value) {
		return static_cast<To>(value);
	}
};

template <typename T>
struct CastTraits<T, T> {};

template <>
struct CastTraits<std::uint8_t, __half> {
	__device__ static __half convert(std::uint8_t value) {
		return static_cast<__half>(static_cast<int>(value));
	}
};

constexpr unsigned int kBlockSize = 512;

template <typename From, typename To>
__global__ void castKernel(From *from, To *to, unsigned int numElements) {
	auto idx = (blockIdx.x * kBlockSize) + threadIdx.x;
	if (idx >= numElements) {
		return;
	}
	using src_storage = DataTypeTraits<From>::storage_type;
	using dst_storage = DataTypeTraits<To>::storage_type;
	src_storage src = reinterpret_cast<src_storage *>(from)[idx];
	reinterpret_cast<dst_storage *>(to)[idx] = DataTypeTraits<To>::store(
	    CastTraits<From, To>::convert(DataTypeTraits<From>::get(src.x)),
	    CastTraits<From, To>::convert(DataTypeTraits<From>::get(src.y)),
	    CastTraits<From, To>::convert(DataTypeTraits<From>::get(src.z)));
}

#define DECLARE_SPEC(From, To)                     \
	template __global__ void castKernel<From, To>( \
	    From * from, To * to, unsigned int numElements)  // NOLINT

DECLARE_SPEC(std::uint8_t, float);
DECLARE_SPEC(float, std::uint8_t);
DECLARE_SPEC(__half, std::uint8_t);
DECLARE_SPEC(std::uint8_t, __half);
DECLARE_SPEC(__half, float);
DECLARE_SPEC(float, __half);

#undef DECLARE_SPEC

template <typename T>
struct UnmanagedCudaBuffer {
	std::size_t size;
	T *ptr;

	std::size_t getByteSize() const {
		return size * sizeof(T);
	}

	std::size_t getNumElements() const {
		if constexpr (std::is_same_v<T, std::uint8_t>) {
			return size / 4;
		} else {
			return size / 3;
		}
	}

	UnmanagedCudaBuffer(std::size_t size, T *ptr) : size{size}, ptr{ptr} {
	}
	explicit UnmanagedCudaBuffer(const CudaBuffer<T> &s)
	    : UnmanagedCudaBuffer(s.getSize(), s.get()) {
	}
	explicit UnmanagedCudaBuffer(const CudaBuffer<DynamicType> &s)
	    : UnmanagedCudaBuffer(s.getSize(), reinterpret_cast<T *>(s.get())) {
	}
	template <typename Enable = std::enable_if<std::is_same_v<T, std::uint8_t>>>
	explicit UnmanagedCudaBuffer(const CudaTensor &s)
	    : UnmanagedCudaBuffer(s.getSize(), s.data()) {
		assert(s.isPlain());
	}
};

template <typename T>
struct ContainerTraits;

template <typename T>
struct ContainerTraits<CudaBuffer<T>> {
	using dataType = T;
};

template <>
struct ContainerTraits<CudaBuffer<DynamicType>> {};

template <>
struct ContainerTraits<CudaTensor> {
	using dataType = std::uint8_t;
};

template <typename T>
struct ContainerTraits<UnmanagedCudaBuffer<T>> {
	using dataType = T;
};

template <typename T, typename DataType = ContainerTraits<T>::dataType>
UnmanagedCudaBuffer<DataType> toUnmanaged(const T &s) {
	return UnmanagedCudaBuffer<DataType>(s);
}

template <typename From, typename To>
void cudaCastUnmanaged(const UnmanagedCudaBuffer<From> &from,
    const UnmanagedCudaBuffer<To> &to, const CudaStream &stream) {
	assert(from.getNumElements() == to.getNumElements());
	auto numElements = static_cast<unsigned int>(to.getNumElements());
	assert(numElements % WARP_SIZE == 0);
	auto numBlocks =
	    static_cast<unsigned int>((numElements + kBlockSize - 1) / kBlockSize);
	castKernel<From, To>
	    <<<numBlocks, kBlockSize, 0, stream>>>(from.ptr, to.ptr, numElements);
	cudaCheck(::cudaGetLastError());
}

template <typename From, typename To>
void cudaCast(const From &from, const To &to, const CudaStream &stream) {
	cudaCastUnmanaged(toUnmanaged(from), toUnmanaged(to), stream);
}

template <typename T>
void cudaCast(
    const CudaBuffer<T> &from, CudaBuffer<T> &to, const CudaStream &stream) {
	cudaCopy(from, to, stream);
}

template <typename T>
void cudaCast(const CudaBuffer<DynamicType> &from, const T &to,
    const CudaStream &stream) {
	using dataType = ContainerTraits<T>::dataType;
	switch (from.getDataType()) {
	case DataType::UINT8:
		if constexpr (std::is_same_v<dataType, std::uint8_t>) {
			cudaCopy(from, to, stream);
		} else {
			cudaCastUnmanaged(UnmanagedCudaBuffer<std::uint8_t>(from),
			    toUnmanaged(to), stream);
		}
		break;
	case DataType::HALF:
		if constexpr (std::is_same_v<dataType, __half>) {
			cudaCopy(from, to, stream);
		} else {
			cudaCastUnmanaged(
			    UnmanagedCudaBuffer<__half>(from), toUnmanaged(to), stream);
		}
		break;
	case DataType::FLOAT:
		if constexpr (std::is_same_v<dataType, float>) {
			cudaCopy(from, to, stream);
		} else {
			cudaCastUnmanaged(
			    UnmanagedCudaBuffer<float>(from), toUnmanaged(to), stream);
		}
		break;
	default:
		unreachable();
	}
}

template <typename T>
void cudaCast(const T &from, const CudaBuffer<DynamicType> &to,
    const CudaStream &stream) {
	using dataType = ContainerTraits<T>::dataType;
	switch (to.getDataType()) {
	case DataType::UINT8:
		if constexpr (std::is_same_v<dataType, std::uint8_t>) {
			cudaCopy(from, to, stream);
		} else {
			cudaCastUnmanaged(toUnmanaged(from),
			    UnmanagedCudaBuffer<std::uint8_t>(to), stream);
		}
		break;
	case DataType::HALF:
		if constexpr (std::is_same_v<dataType, __half>) {
			cudaCopy(from, to, stream);
		} else {
			cudaCastUnmanaged(
			    toUnmanaged(from), UnmanagedCudaBuffer<__half>(to), stream);
		}
		break;
	case DataType::FLOAT:
		if constexpr (std::is_same_v<dataType, float>) {
			cudaCopy(from, to, stream);
		} else {
			cudaCastUnmanaged(
			    toUnmanaged(from), UnmanagedCudaBuffer<float>(to), stream);
		}
		break;
	default:
		unreachable();
	}
}

void cudaCast(const CudaBuffer<DynamicType> &from,
    const CudaBuffer<DynamicType> &to, const CudaStream &stream) {
	if (from.getDataType() == to.getDataType()) {
		cudaCopy(from, to, stream);
		return;
	}
	switch (to.getDataType()) {
	case DataType::UINT8:
		cudaCast(from, UnmanagedCudaBuffer<std::uint8_t>(to), stream);
		break;
	case DataType::HALF:
		cudaCast(from, UnmanagedCudaBuffer<__half>(to), stream);
		break;
	case DataType::FLOAT:
		cudaCast(from, UnmanagedCudaBuffer<float>(to), stream);
		break;
	default:
		unreachable();
	}
}

}  // namespace

}  // namespace detail

template <typename From, typename To>
void cudaCast(const From &from, const To &to, const CudaStream &stream) {
	detail::cudaCast(from, to, stream);
}

#define DECLARE_SPEC(From, To)        \
	template void cudaCast<From, To>( \
	    const From &from, const To &to, const CudaStream &stream);

DECLARE_SPEC(CudaBuffer<std::uint8_t>, CudaBuffer<float>);
DECLARE_SPEC(CudaBuffer<float>, CudaBuffer<std::uint8_t>);
DECLARE_SPEC(CudaBuffer<__half>, CudaBuffer<float>);
DECLARE_SPEC(CudaBuffer<float>, CudaBuffer<__half>);
DECLARE_SPEC(CudaBuffer<std::uint8_t>, CudaBuffer<__half>);
DECLARE_SPEC(CudaBuffer<__half>, CudaBuffer<std::uint8_t>);
DECLARE_SPEC(CudaTensor, CudaBuffer<float>);
DECLARE_SPEC(CudaTensor, CudaBuffer<__half>);
DECLARE_SPEC(CudaBuffer<__half>, CudaTensor);
DECLARE_SPEC(CudaBuffer<float>, CudaTensor);
DECLARE_SPEC(CudaBuffer<DynamicType>, CudaBuffer<std::uint8_t>);
DECLARE_SPEC(CudaBuffer<DynamicType>, CudaBuffer<__half>);
DECLARE_SPEC(CudaBuffer<DynamicType>, CudaBuffer<float>);
DECLARE_SPEC(CudaBuffer<DynamicType>, CudaTensor);
DECLARE_SPEC(CudaBuffer<std::uint8_t>, CudaBuffer<DynamicType>);
DECLARE_SPEC(CudaBuffer<__half>, CudaBuffer<DynamicType>);
DECLARE_SPEC(CudaBuffer<float>, CudaBuffer<DynamicType>);
DECLARE_SPEC(CudaTensor, CudaBuffer<DynamicType>);
DECLARE_SPEC(CudaBuffer<DynamicType>, CudaBuffer<DynamicType>);

#undef DECLARE_SPEC

namespace detail {

namespace {

template <typename From, typename To>
struct CopyKind {};

template <typename T>
struct CopyKind<CpuTensor, CudaBuffer<T>> {
	static constexpr ::cudaMemcpyKind value =
	    ::cudaMemcpyKind::cudaMemcpyHostToDevice;
};

template <typename T>
struct CopyKind<CudaTensor, CudaBuffer<T>> {
	static constexpr ::cudaMemcpyKind value =
	    ::cudaMemcpyKind::cudaMemcpyDeviceToDevice;
};

template <typename T>
struct CopyKind<CudaBuffer<T>, CpuTensor> {
	static constexpr ::cudaMemcpyKind value =
	    ::cudaMemcpyKind::cudaMemcpyDeviceToHost;
};

template <typename T>
struct CopyKind<CudaBuffer<T>, CudaTensor> {
	static constexpr ::cudaMemcpyKind value =
	    ::cudaMemcpyKind::cudaMemcpyDeviceToDevice;
};

template <typename From, typename T>
void cudaCopy(
    const From &from, const CudaBuffer<T> &to, const CudaStream &stream) {
	auto copyKind = CopyKind<From, CudaBuffer<T>>::value;
	std::size_t size = from.getByteSize();
	assert(from.getSize() == size && to.getByteSize() == size);
	if constexpr (std::is_same_v<T, DynamicType>) {
		assert(to.getDataType() == DataType::UINT8);
	}
	if (from.isPlain()) {
		cudaCheck(
		    ::cudaMemcpyAsync(to.get(), from.data(), size, copyKind, stream));
	} else {
		std::size_t lineLength = from.getWidth() * 4 * sizeof(std::byte);
		cudaCheck(::cudaMemcpy2DAsync(to.get(), lineLength, from.data(),
		    static_cast<std::size_t>(from.getStride()), lineLength,
		    from.getHeight(), copyKind, stream));
	}
}

template <typename T>
void cudaCopy(const GraphicsResource &from, const CudaBuffer<T> &to,
    const CudaStream &stream) {
	auto array = from.get();
	::cudaExtent extent;
	cudaCheck(::cudaArrayGetInfo(nullptr, &extent, nullptr, array));
	assert(to.getByteSize() == extent.width * extent.height * extent.depth);
	std::size_t lineLength = extent.width * 4 * sizeof(std::byte);
	if constexpr (std::is_same_v<T, DynamicType>) {
		assert(to.getDataType() == DataType::UINT8);
	}
	cudaCheck(::cudaMemcpy2DFromArrayAsync(to.get(), lineLength, array, 0, 0,
	    lineLength, extent.height, ::cudaMemcpyDeviceToDevice, stream));
}

template <typename To, typename T>
void cudaCopy(
    const CudaBuffer<T> &from, const To &to, const CudaStream &stream) {
	auto copyKind = CopyKind<CudaBuffer<T>, To>::value;
	std::size_t size = to.getByteSize();
	assert(from.getByteSize() == size && to.getByteSize() == size);
	if constexpr (std::is_same_v<T, DynamicType>) {
		assert(from.getDataType() == DataType::UINT8);
	}
	if (to.isPlain()) {
		cudaCheck(
		    ::cudaMemcpyAsync(to.data(), from.get(), size, copyKind, stream));
	} else {
		std::size_t lineLength = to.getWidth() * 4 * sizeof(std::byte);
		cudaCheck(::cudaMemcpy2DAsync(to.data(),
		    static_cast<std::size_t>(to.getStride()), from.get(), lineLength,
		    lineLength, to.getHeight(), copyKind, stream));
	}
}

template <typename T>
void cudaCopy(const CudaBuffer<T> &from, const GraphicsResource &to,
    const CudaStream &stream) {
	auto array = to.get();
	::cudaExtent extent;
	cudaCheck(::cudaArrayGetInfo(nullptr, &extent, nullptr, array));
	std::size_t lineLength = extent.width * 4 * sizeof(std::byte);
	assert(from.getByteSize() == extent.width * extent.height * extent.depth);
	if constexpr (std::is_same_v<T, DynamicType>) {
		assert(from.getDataType() == DataType::UINT8);
	}
	cudaCheck(::cudaMemcpy2DToArrayAsync(array, 0, 0, from.get(), lineLength,
	    lineLength, extent.height, ::cudaMemcpyDeviceToDevice, stream));
}

template <typename T>
[[noreturn]] void cudaCopy([[maybe_unused]] const CudaBuffer<DynamicType> &from,
    [[maybe_unused]] const UnmanagedCudaBuffer<T> &to,
    [[maybe_unused]] const CudaStream &stream) {
	unreachable();
}

template <typename From, typename To,
    typename Enable = std::enable_if<std::is_same_v<From, DynamicType> ||
                                     std::is_same_v<To, DynamicType> ||
                                     std::is_same_v<From, To>>>
void cudaCopy(const CudaBuffer<From> &from, const CudaBuffer<To> &to,
    const CudaStream &stream) {
	std::size_t size = from.getByteSize();
	assert(from.getByteSize() == size && to.getByteSize() == size);
	if constexpr (std::is_same_v<From, DynamicType> &&
	              std::is_same_v<To, DynamicType>) {
		assert(from.getDataType() == to.getDataType());
	}
	cudaCheck(::cudaMemcpyAsync(to.get(), from.get(), size,
	    ::cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
}

template <typename T>
void cudaCopy(const CudaBuffer<T> &from, const GenericTensor &to,
    const CudaStream &stream) {
	switch (to.getLocation()) {
	case DataLocation::CPU:
		cudaCopy(from, static_cast<const CpuTensor &>(to), stream);
		return;
	case DataLocation::CUDA:
		cudaCopy(from, static_cast<const CudaTensor &>(to), stream);
		return;
	default:
		unreachable();
	}
}

template <typename T>
void cudaCopy(const GenericTensor &from, const CudaBuffer<T> &to,
    const CudaStream &stream) {
	switch (from.getLocation()) {
	case DataLocation::CPU:
		cudaCopy(static_cast<const CpuTensor &>(from), to, stream);
		return;
	case DataLocation::CUDA:
		cudaCopy(static_cast<const CudaTensor &>(from), to, stream);
		return;
	default:
		unreachable();
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

DECLARE_SPEC(CpuTensor, CudaBuffer<std::uint8_t>);
DECLARE_SPEC(CudaTensor, CudaBuffer<std::uint8_t>);
DECLARE_SPEC(GenericTensor, CudaBuffer<std::uint8_t>);
DECLARE_SPEC(GraphicsResource, CudaBuffer<std::uint8_t>);
DECLARE_SPEC(CpuTensor, CudaBuffer<DynamicType>);
DECLARE_SPEC(CudaTensor, CudaBuffer<DynamicType>);
DECLARE_SPEC(GenericTensor, CudaBuffer<DynamicType>);
DECLARE_SPEC(GraphicsResource, CudaBuffer<DynamicType>);
DECLARE_SPEC(CudaBuffer<std::uint8_t>, CpuTensor);
DECLARE_SPEC(CudaBuffer<std::uint8_t>, CudaTensor);
DECLARE_SPEC(CudaBuffer<std::uint8_t>, GenericTensor);
DECLARE_SPEC(CudaBuffer<std::uint8_t>, GraphicsResource);
DECLARE_SPEC(CudaBuffer<DynamicType>, CpuTensor);
DECLARE_SPEC(CudaBuffer<DynamicType>, CudaTensor);
DECLARE_SPEC(CudaBuffer<DynamicType>, GenericTensor);
DECLARE_SPEC(CudaBuffer<DynamicType>, GraphicsResource);
DECLARE_SPEC(CudaBuffer<std::uint8_t>, CudaBuffer<std::uint8_t>);
DECLARE_SPEC(CudaBuffer<float>, CudaBuffer<float>);
DECLARE_SPEC(CudaBuffer<__half>, CudaBuffer<__half>);
DECLARE_SPEC(CudaBuffer<std::uint8_t>, CudaBuffer<DynamicType>);
DECLARE_SPEC(CudaBuffer<float>, CudaBuffer<DynamicType>);
DECLARE_SPEC(CudaBuffer<__half>, CudaBuffer<DynamicType>);
DECLARE_SPEC(CudaBuffer<DynamicType>, CudaBuffer<std::uint8_t>);
DECLARE_SPEC(CudaBuffer<DynamicType>, CudaBuffer<float>);
DECLARE_SPEC(CudaBuffer<DynamicType>, CudaBuffer<__half>);
DECLARE_SPEC(CudaBuffer<DynamicType>, CudaBuffer<DynamicType>);

#undef DECLARE_SPEC

}  // namespace cuda

}  // namespace core

}  // namespace JoshUpscale
