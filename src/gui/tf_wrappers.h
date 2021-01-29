// Copyright 2021 Ivanov Viktor

#pragma once
#pragma warning(disable : 4190)

#include <tensorflow/c/c_api.h>
#include <tensorflow/c/c_api_experimental.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tf {

struct TFException : std::exception {
	TFException() = delete;
	explicit TFException(const TF_Status *status)
	    : std::exception(::TF_Message(status)) {
	}
};

}  // namespace tf

namespace smart {

#define DEFINE_SMART_TF_CLASS(TF_Class, TF_Constructor, TF_Destructor)  \
	struct TF_Class                                                     \
	    : std::unique_ptr<::TF_Class, decltype(&::TF_Destructor)> {     \
		using unique_ptr =                                              \
		    std::unique_ptr<::TF_Class, decltype(&::TF_Destructor)>;    \
                                                                        \
		TF_Class() : unique_ptr(::TF_Constructor(), &::TF_Destructor) { \
		}                                                               \
		TF_Class(const TF_Class &) = delete;                            \
		TF_Class(TF_Class &&) noexcept = default;                       \
                                                                        \
		TF_Class &operator=(const TF_Class &) = delete;                 \
		TF_Class &operator=(TF_Class &&) = default;                     \
	};

DEFINE_SMART_TF_CLASS(TF_Status, TF_NewStatus, TF_DeleteStatus);

DEFINE_SMART_TF_CLASS(TF_Graph, TF_NewGraph, TF_DeleteGraph);

DEFINE_SMART_TF_CLASS(TF_ImportGraphDefOptions, TF_NewImportGraphDefOptions,
    TF_DeleteImportGraphDefOptions);

DEFINE_SMART_TF_CLASS(
    TF_SessionOptions, TF_NewSessionOptions, TF_DeleteSessionOptions);

#undef DEFINE_SMART_TF_CLASS

struct TF_Buffer {
	TF_Buffer() = delete;
	explicit TF_Buffer(std::size_t size)
	    : m_Data(std::make_unique<char[]>(size)) {
		m_Buffer = ::TF_NewBuffer();
		m_Buffer->data = m_Data.get();
		m_Buffer->length = size;
		m_Buffer->data_deallocator = nullptr;
	}
	TF_Buffer(const TF_Buffer &) = delete;
	TF_Buffer(TF_Buffer &&s) noexcept : m_Data(std::move(s.m_Data)) {
		m_Buffer = s.m_Buffer;
		s.m_Buffer = nullptr;
	}
	~TF_Buffer() {
		if (m_Buffer) {
			TF_DeleteBuffer(m_Buffer);
		}
	}

	operator bool() {
		return m_Buffer;
	}
	char *getData() {
		return m_Data.get();
	}
	::TF_Buffer *get() {
		return m_Buffer;
	}

private:
	::TF_Buffer *m_Buffer = nullptr;
	std::unique_ptr<char[]> m_Data;
};

template <typename T>
struct TF_Tensor;

template <>
struct TF_Tensor<float> {
	TF_Tensor() = delete;
	TF_Tensor(const std::int64_t *dims, std::size_t numDims) {
		std::size_t len = sizeof(float);
		for (std::size_t i = 0; i < numDims; ++i) {
			len *= dims[i];
		}
		m_Tensor =
		    ::TF_AllocateTensor(TF_FLOAT, dims, static_cast<int>(numDims), len);
	}
	TF_Tensor(const TF_Tensor &) = delete;
	TF_Tensor(TF_Tensor &&s) noexcept {
		m_Tensor = s.m_Tensor;
		s.m_Tensor = nullptr;
	}
	TF_Tensor(::TF_Tensor *tensor)  // NOLINT(runtime/explicit)
	    : m_Tensor(tensor) {
	}
	~TF_Tensor() {
		if (m_Tensor) {
			::TF_DeleteTensor(m_Tensor);
		}
	}

	TF_Tensor &operator=(const TF_Tensor &) = delete;
	TF_Tensor &operator=(TF_Tensor &&s) noexcept {
		if (m_Tensor) {
			::TF_DeleteTensor(m_Tensor);
		}
		m_Tensor = s.m_Tensor;
		s.m_Tensor = nullptr;
		return *this;
	}

	::TF_Tensor *get() {
		return m_Tensor;
	}

private:
	::TF_Tensor *m_Tensor = nullptr;
};

struct TF_SessionOptionsProto {
	const void *proto;
	std::size_t size;
};

struct TF_Session {
	TF_Session() = delete;
	TF_Session(const TF_Graph &graph, const TF_SessionOptionsProto *options,
	    bool xla = false);
	TF_Session(const TF_Session &) = delete;
	TF_Session(TF_Session &&s) noexcept {
		m_Session = s.m_Session;
		s.m_Session = nullptr;
	}
	~TF_Session() {
		if (m_Session) {
			TF_Status status;
			::TF_CloseSession(m_Session, status.get());
			::TF_DeleteSession(m_Session, status.get());
		}
	}

	::TF_Session *get() {
		return m_Session;
	}

	TF_Tensor<float> run(const std::vector<::TF_Output> &inputOp,
	    const std::vector<::TF_Tensor *> &inputValue,
	    const TF_Output &outputOp);

private:
	::TF_Session *m_Session = nullptr;
};

}  // namespace smart

namespace tf {

smart::TF_Graph readGraph(const std::string &fileName);

}  // namespace tf
