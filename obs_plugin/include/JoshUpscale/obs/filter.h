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
#include <memory>
#include <mutex>
#include <thread>

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

	explicit AVBuffer(std::size_t size) : unique_ptr(alloc(size)) {
	}

private:
	static void *alloc(std::size_t size) {
		void *ptr = ::av_malloc(size);
		if (ptr == nullptr) {
			throw std::bad_alloc();
		}
		return ptr;
	}
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
	std::unique_ptr<core::Runtime> m_Runtime;
	AVBuffer m_InputBuffer;
	OBSFrame m_OutputFrame;
	::SwsContext *m_SwsCtx = nullptr;
	int m_CurrentModel = -1;
	int m_LoadedModel = -1;
	std::atomic<bool> m_Error = false;
	std::atomic<bool> m_Ready = false;
	mutable std::mutex m_Mutex;
	std::thread m_WorkerThread;
	std::condition_variable m_Condition;
	bool m_Terminated = false;
};

}  // namespace obs

}  // namespace JoshUpscale
