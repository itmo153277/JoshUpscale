// Copyright 2025 Ivanov Viktor

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "JoshUpscale/core.h"
#include "JoshUpscale/core/tensor.h"
#include "JoshUpscale/core/utils.h"

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

struct DynamicType {};

enum class DataType : std::uint8_t { UINT8, FLOAT, HALF };

template <typename T>
struct CudaBuffer : std::unique_ptr<T, detail::CudaDeleter<T>> {
	using unique_ptr = std::unique_ptr<T, detail::CudaDeleter<T>>;
	using unique_ptr::get;

	explicit CudaBuffer(std::nullptr_t) : unique_ptr(nullptr) {
	}
	explicit CudaBuffer(std::size_t size)
	    : unique_ptr(alloc(size)), m_Size(size) {
		cudaCheck(::cudaMemset(get(), 0, getByteSize()));
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

template <>
struct CudaBuffer<DynamicType>
    : std::unique_ptr<void, detail::CudaDeleter<void>> {
	using unique_ptr = std::unique_ptr<void, detail::CudaDeleter<void>>;
	using unique_ptr::get;

	explicit CudaBuffer(std::nullptr_t) : unique_ptr(nullptr) {
	}
	explicit CudaBuffer(std::size_t size, DataType type)
	    : unique_ptr(alloc(size, type)), m_Size(size), m_DataType(type) {
		cudaCheck(::cudaMemset(get(), 0, getByteSize()));
	}

	// Non-copyable, default-movable
	CudaBuffer(const CudaBuffer &) = delete;
	CudaBuffer(CudaBuffer &&) noexcept = default;
	CudaBuffer &operator=(const CudaBuffer &) = delete;
	CudaBuffer &operator=(CudaBuffer &&) noexcept = default;

	std::size_t getSize() const {
		return m_Size;
	}
	std::size_t getByteSize() const {
		return m_Size * getDataTypeSize(m_DataType);
	}
	DataType getDataType() const {
		return m_DataType;
	}

private:
	std::size_t m_Size;
	DataType m_DataType;

	static void *alloc(std::size_t size, DataType type) {
		void *result;
		cudaCheck(::cudaMalloc(&result, size * getDataTypeSize(type)));
		return result;
	}

	static constexpr std::size_t getDataTypeSize(DataType type) {
		switch (type) {
		case DataType::UINT8:
			return sizeof(std::uint8_t);
		case DataType::HALF:
			return sizeof(__half);
		case DataType::FLOAT:
			return sizeof(float);
		default:
			unreachable();
		}
	}
};

using GenericCudaBuffer = CudaBuffer<DynamicType>;

class CudaGraphExec {
public:
	explicit CudaGraphExec(::cudaGraphExec_t instance) : m_GraphExec{instance} {
	}
	explicit CudaGraphExec(std::nullptr_t) {
	}
	~CudaGraphExec() {
		if (m_GraphExec != nullptr) {
			::cudaGraphExecDestroy(m_GraphExec);
		}
	}

	// Non-copyable, movable
	CudaGraphExec(const CudaGraphExec &) = delete;
	CudaGraphExec(CudaGraphExec &&s) noexcept {
		m_GraphExec = s.m_GraphExec;
		s.m_GraphExec = nullptr;
	}
	CudaGraphExec &operator=(const CudaGraphExec &) = delete;
	CudaGraphExec &operator=(CudaGraphExec &&s) noexcept {
		if (this != &s) {
			this->~CudaGraphExec();
			new (this) CudaGraphExec(std::move(s));
		}
		return *this;
	}

	::cudaGraphExec_t get() const {
		return m_GraphExec;
	}
	operator ::cudaGraphExec_t() const {
		return m_GraphExec;
	}

	void launch(::cudaStream_t stream) const {
		cudaCheck(::cudaGraphLaunch(m_GraphExec, stream));
	}

private:
	::cudaGraphExec_t m_GraphExec = nullptr;
};

class CudaGraph {
public:
	explicit CudaGraph(::cudaGraph_t graph) : m_Graph(graph) {
	}
	explicit CudaGraph(std::nullptr_t) {
	}
	~CudaGraph() {
		if (m_Graph != nullptr) {
			::cudaGraphDestroy(m_Graph);
		}
	}

	// Non-copyable, movable
	CudaGraph(const CudaGraph &) = delete;
	CudaGraph(CudaGraph &&s) noexcept {
		m_Graph = s.m_Graph;
		s.m_Graph = nullptr;
	}
	CudaGraph &operator=(const CudaGraph &) = delete;
	CudaGraph &operator=(CudaGraph &&s) noexcept {
		if (this != &s) {
			this->~CudaGraph();
			new (this) CudaGraph(std::move(s));
		}
		return *this;
	}

	::cudaGraph_t get() const {
		return m_Graph;
	}
	operator ::cudaGraph_t() const {
		return m_Graph;
	}

	CudaGraphExec instantieate() const {
		::cudaGraphExec_t instance;
		cudaCheck(::cudaGraphInstantiate(&instance, m_Graph));
		return CudaGraphExec(instance);
	}

private:
	::cudaGraph_t m_Graph = nullptr;
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

	void beginCapture() const {
		cudaCheck(
		    ::cudaStreamBeginCapture(m_Stream, ::cudaStreamCaptureModeGlobal));
	}

	CudaGraph endCapture() const {
		::cudaGraph_t graph;
		cudaCheck(::cudaStreamEndCapture(m_Stream, &graph));
		return CudaGraph(graph);
	}

private:
	::cudaStream_t m_Stream = nullptr;
};

template <typename From, typename To>
void cudaCast(const From &from, const To &to, const CudaStream &stream);

template <typename From, typename To>
void cudaCopy(const From &from, const To &to, const CudaStream &stream);

template <typename T>
void cudaCopy(const CpuTensor &from, const CudaBuffer<T> &to,
    const CudaBuffer<std::uint8_t> &internalBuffer, const CudaStream &stream) {
	if constexpr (std::is_same_v<T, std::uint8_t>) {
		cudaCopy(from, to, stream);
	} else {
		cudaCopy(from, internalBuffer, stream);
		cudaCast(internalBuffer, to, stream);
	}
}

template <typename T>
void cudaCopy(const CudaTensor &from, const CudaBuffer<T> &to,
    const CudaBuffer<std::uint8_t> &internalBuffer, const CudaStream &stream) {
	if constexpr (std::is_same_v<T, std::uint8_t>) {
		cudaCopy(from, to, stream);
	} else if (from.isPlain()) {
		cudaCast(from, to, stream);
	} else {
		cudaCopy(from, internalBuffer, stream);
		cudaCast(internalBuffer, to, stream);
	}
}

template <typename T>
void cudaCopy(const GenericTensor &from, const CudaBuffer<T> &to,
    const CudaBuffer<std::uint8_t> &internalBuffer, const CudaStream &stream) {
	switch (from.getLocation()) {
	case DataLocation::CPU:
		cudaCopy(
		    static_cast<const CpuTensor &>(from), to, internalBuffer, stream);
		break;
	case DataLocation::CUDA:
		cudaCopy(
		    static_cast<const CudaTensor &>(from), to, internalBuffer, stream);
		break;
	}
}

template <typename T>
void cudaCopy(const CudaBuffer<T> &from, const CpuTensor &to,
    const CudaBuffer<std::uint8_t> &internalBuffer, const CudaStream &stream) {
	if constexpr (std::is_same_v<T, std::uint8_t>) {
		cudaCopy(from, to, stream);
	} else {
		cudaCast(from, internalBuffer, stream);
		cudaCopy(internalBuffer, to, stream);
	}
}

template <typename T>
void cudaCopy(const CudaBuffer<T> &from, const CudaTensor &to,
    const CudaBuffer<std::uint8_t> &internalBuffer, const CudaStream &stream) {
	if constexpr (std::is_same_v<T, std::uint8_t>) {
		cudaCopy(from, to, stream);
	} else if (to.isPlain()) {
		cudaCast(from, to, stream);
	} else {
		cudaCast(from, internalBuffer, stream);
		cudaCopy(internalBuffer, to, stream);
	}
}

template <typename T>
void cudaCopy(const CudaBuffer<T> &from, const GenericTensor &to,
    const CudaBuffer<std::uint8_t> &internalBuffer, const CudaStream &stream) {
	switch (to.getLocation()) {
	case DataLocation::CPU:
		cudaCopy(
		    from, static_cast<const CpuTensor &>(to), internalBuffer, stream);
		break;
	case DataLocation::CUDA:
		cudaCopy(
		    from, static_cast<const CudaTensor &>(to), internalBuffer, stream);
		break;
	}
}

struct DeviceContext {
	explicit DeviceContext(int device) {
		cudaCheck(::cudaGetDevice(&m_Device));
		cudaCheck(::cudaSetDevice(device));
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
