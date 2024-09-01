// Copyright 2023 Viktor Ivanov

#pragma once

extern "C" {
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS  // NOLINT
#endif
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <obs-module.h>
#include <util/threading.h>
}

#include <JoshUpscale/core.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <utility>

namespace JoshUpscale {

namespace obs {

namespace detail {

struct AVDeleter {
	void operator()(void *data) {
		::av_free(data);
	}
};

}  // namespace detail

struct AVBuffer : std::unique_ptr<void, detail::AVDeleter> {
	using unique_ptr = std::unique_ptr<void, detail::AVDeleter>;

	explicit AVBuffer(std::nullptr_t) : unique_ptr(nullptr) {
	}
	explicit AVBuffer(std::size_t size) : unique_ptr(alloc(size)) {
	}

private:
	static void *alloc(std::size_t size) {
		void *ptr = ::av_malloc(size);
		if (ptr == nullptr) {
			throw std::bad_alloc();
		}
		std::memset(ptr, 0, size);
		return ptr;
	}
};

struct CroppedFrame {
	CroppedFrame()
	    : m_Width(0)
	    , m_Height(0)
	    , m_CropLeft(0)
	    , m_CropTop(0)
	    , m_CropRight(0)
	    , m_CropBottom(0)
	    , m_Buffer(nullptr) {
	}
	CroppedFrame(std::size_t width, std::size_t height, int left, int top,
	    int right, int bottom)
	    : m_Width(width)
	    , m_Height(height)
	    , m_CropLeft(left)
	    , m_CropTop(top)
	    , m_CropRight(right)
	    , m_CropBottom(bottom)
	    , m_Buffer(getBufferSize(width, height, left, top, right, bottom)) {
		m_Strides[0] = static_cast<int>(getSize(width, left, right) * 3);
		m_InputPlanes[0] = reinterpret_cast<std::uint8_t *>(m_Buffer.get());
		if (left < 0) {
			m_InputPlanes[0] =
			    m_InputPlanes[0] - static_cast<ptrdiff_t>(left * 3);
		}
		if (top < 0) {
			m_InputPlanes[0] =
			    m_InputPlanes[0] - static_cast<ptrdiff_t>(top * m_Strides[0]);
		}
		m_OutputPlanes[0] = reinterpret_cast<std::uint8_t *>(m_Buffer.get());
		if (left > 0) {
			m_OutputPlanes[0] =
			    m_OutputPlanes[0] + static_cast<ptrdiff_t>(left * 3);
		}
		if (top > 0) {
			m_OutputPlanes[0] =
			    m_OutputPlanes[0] + static_cast<ptrdiff_t>(top * m_Strides[0]);
		}
	}
	CroppedFrame(const CroppedFrame &) = delete;
	CroppedFrame(CroppedFrame &&) noexcept = default;
	CroppedFrame &operator=(const CroppedFrame &) = delete;
	CroppedFrame &operator=(CroppedFrame &&) noexcept = default;

	void realloc(std::size_t width, std::size_t height, int left, int top,
	    int right, int bottom) {
		if (width != m_Width || height != m_Height || left != m_CropLeft ||
		    top != m_CropTop || right != m_CropRight ||
		    bottom != m_CropBottom) {
			*this = CroppedFrame(width, height, left, top, right, bottom);
		}
	}

	const int *getStrides() const {
		return m_Strides;
	}

	std::uint8_t **getInputPlanes() {
		return m_InputPlanes;
	}
	const std::uint8_t *const *getOutputPlanes() const {
		return m_OutputPlanes;
	}

private:
	static std::size_t getSize(std::size_t base, int begin, int end) {
		std::size_t size = base;
		if (begin < 0) {
			size += static_cast<std::size_t>(-begin);
		}
		if (end < 0) {
			size += static_cast<std::size_t>(-end);
		}
		return size;
	}
	static std::size_t getBufferSize(std::size_t width, std::size_t height,
	    int left, int top, int right, int bottom) {
		return getSize(width, left, right) * getSize(height, top, bottom) * 3;
	}

	std::size_t m_Width;
	std::size_t m_Height;
	int m_CropLeft;
	int m_CropTop;
	int m_CropRight;
	int m_CropBottom;
	int m_Strides[4] = {};
	std::uint8_t *m_InputPlanes[4] = {};
	std::uint8_t *m_OutputPlanes[4] = {};
	AVBuffer m_Buffer;
};

namespace detail {

struct OBSFrameDeleter {
	void operator()(::obs_source_frame *frame) {
		::obs_source_frame_destroy(frame);
	}
};

}  // namespace detail

struct OBSFrame : std::unique_ptr<::obs_source_frame, detail::OBSFrameDeleter> {
	using unique_ptr =
	    std::unique_ptr<::obs_source_frame, detail::OBSFrameDeleter>;

	OBSFrame(std::size_t width, std::size_t height)
	    : unique_ptr(alloc(width, height)) {
		if (!::video_format_get_parameters(VIDEO_CS_SRGB, VIDEO_RANGE_FULL,
		        get()->color_matrix, get()->color_range_min,
		        get()->color_range_max)) {
			throw std::bad_alloc();
		}
	}

	operator ::obs_source_frame *() const {
		return get();
	}

private:
	static ::obs_source_frame *alloc(std::size_t width, std::size_t height) {
		auto *ptr = ::obs_source_frame_create(VIDEO_FORMAT_BGR3,
		    static_cast<std::uint32_t>(width),
		    static_cast<std::uint32_t>(height));
		if (ptr == nullptr) {
			throw std::bad_alloc();
		}
		ptr->refs = 1;
		return ptr;
	}
};

namespace detail {

struct OBSDeleter {
	void operator()(void *data) {
		::bfree(data);
	}
};

}  // namespace detail

template <class T>
struct OBSPtr : std::unique_ptr<T, detail::OBSDeleter> {
	using unique_ptr = std::unique_ptr<T, detail::OBSDeleter>;

	explicit OBSPtr(T *val) : unique_ptr(val) {
	}
};

template <typename T>
struct Defer {
	T m_DeferFn;
	explicit Defer(T &&fn) : m_DeferFn(std::move(fn)) {
	}
	Defer(const Defer &) = delete;
	Defer(Defer &&) noexcept = delete;
	~Defer() {
		m_DeferFn();
	}
};

struct JoshUpscaleFilter {
	static ::obs_source_info *getSourceInfo();

private:
	JoshUpscaleFilter(::obs_data_t *settings, ::obs_source_t *source);
	~JoshUpscaleFilter();

	static const char *getName(void *typeData) noexcept;

	static void *create(
	    ::obs_data_t *settings, ::obs_source_t *source) noexcept;

	static void destroy(void *data) noexcept;

	void update(::obs_data_t *settings) noexcept;

	::obs_source_frame *filterVideo(::obs_source_frame *frame) noexcept;

	void filterRemove(obs_source_t *source) noexcept;

	static ::obs_properties_t *getProperties(
	    void *data, void *typeData) noexcept;

	void addProperties(::obs_properties_t *props, void *typeData) noexcept;

	static void getDefaults(void *typeData, ::obs_data_t *settings) noexcept;

	template <auto Ptr>
	struct Callback {
		static consteval decltype(Ptr) getPtr() noexcept {
			return Ptr;
		}
	};

	template <typename R, typename... T,
	    R (JoshUpscaleFilter::*Ptr)(T...) noexcept>
	struct Callback<Ptr> {
		static consteval R (*getPtr() noexcept)(void *, T...) noexcept {
			return [](void *self, T... params) noexcept -> R {
				return (reinterpret_cast<JoshUpscaleFilter *>(self)->*Ptr)(
				    params...);
			};
		}
	};

	void copyFrame(::obs_source_frame *frame);

	void workerThread() noexcept;

	::obs_source_t *m_Source;
	std::unique_ptr<core::Runtime> m_Runtime = nullptr;
	AVBuffer m_InputBuffer;
	OBSFrame m_OutputFrame;
	::SwsContext *m_SwsCtxDecode = nullptr;
	::SwsContext *m_SwsCtxScale = nullptr;
	CroppedFrame m_CroppedFrame;
	int m_CurrentModel = -1;
	int m_LoadedModel = -1;
	int m_CropLeft = 0;
	int m_CropTop = 0;
	int m_CropRight = 0;
	int m_CropBottom = 0;
	int m_NextCropLeft = 0;
	int m_NextCropTop = 0;
	int m_NextCropRight = 0;
	int m_NextCropBottom = 0;
	bool m_LimitFps = false;
	bool m_NextLimitFps = false;
	std::uint64_t m_LastPts = 0;
	core::Quantization m_CurrentQuant = core::Quantization::NONE;
	core::Quantization m_LoadedQuant = core::Quantization::NONE;
	std::atomic<bool> m_Error = false;
	std::atomic<bool> m_Ready = false;
	std::atomic<bool> m_Busy = false;
	mutable std::mutex m_Mutex;
	std::thread m_WorkerThread;
	std::condition_variable m_Condition;
	bool m_Terminated = false;
};

}  // namespace obs

}  // namespace JoshUpscale
