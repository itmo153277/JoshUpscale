// Copyright 2023 Viktor Ivanov

#pragma once

extern "C" {
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS  // NOLINT
#endif
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <obs-module.h>
}

#include <JoshUpscale/core.h>

#include <cstddef>
#include <memory>
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
		::obs_source_frame_free(frame);
	}
};

}  // namespace detail

struct OBSFrame : std::unique_ptr<::obs_source_frame, detail::OBSFrameDeleter> {
	using unique_ptr =
	    std::unique_ptr<::obs_source_frame, detail::OBSFrameDeleter>;

	OBSFrame(std::size_t width, std::size_t height)
	    : unique_ptr(alloc(width, height)) {
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
		return ptr;
	}
};

struct JoshUpscaleFilter {
	static ::obs_source_info *getSourceInfo();

private:
	explicit JoshUpscaleFilter(::obs_source_t *source);
	~JoshUpscaleFilter();

	const char *getName() noexcept;

	static void *create(::obs_data_t *params, ::obs_source_t *source) noexcept;

	static void destroy(void *data) noexcept;

	::obs_source_frame *filterVideo(::obs_source_frame *frame) noexcept;

	template <auto Ptr>
	struct Callback;

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

	template <typename R, typename... T, R (*Ptr)(T...) noexcept>
	struct Callback<Ptr> {
		static consteval R (*getPtr() noexcept)(T...) noexcept {
			return Ptr;
		}
	};

	void copyFrame(::obs_source_frame *frame);

	::obs_source_t *m_Source;
	std::unique_ptr<core::Runtime> m_Runtime;
	AVBuffer m_InputBuffer;
	OBSFrame m_OutputFrame;
	::SwsContext *m_SwsCtx = nullptr;
};

}  // namespace obs

}  // namespace JoshUpscale
