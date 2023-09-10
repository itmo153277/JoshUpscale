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
#include <new>
#include <stdexcept>
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

namespace detail {

inline ::AVPixelFormat convertFrameFormat(::video_format format) {
	switch (format) {
	case VIDEO_FORMAT_I444:
		return AV_PIX_FMT_YUV444P;
	case VIDEO_FORMAT_I420:
		return AV_PIX_FMT_YUV420P;
	case VIDEO_FORMAT_NV12:
		return AV_PIX_FMT_NV12;
	case VIDEO_FORMAT_YUY2:
		return AV_PIX_FMT_YUYV422;
	case VIDEO_FORMAT_UYVY:
		return AV_PIX_FMT_UYVY422;
	case VIDEO_FORMAT_YVYU:
		return AV_PIX_FMT_YVYU422;
	case VIDEO_FORMAT_RGBA:
		return AV_PIX_FMT_RGBA;
	case VIDEO_FORMAT_BGRA:
	case VIDEO_FORMAT_BGRX:
		return AV_PIX_FMT_BGRA;
	case VIDEO_FORMAT_Y800:
		return AV_PIX_FMT_GRAY8;
	case VIDEO_FORMAT_BGR3:
		return AV_PIX_FMT_BGR24;
	case VIDEO_FORMAT_I422:
		return AV_PIX_FMT_YUV422P;
	case VIDEO_FORMAT_I40A:
		return AV_PIX_FMT_YUVA420P;
	case VIDEO_FORMAT_I42A:
		return AV_PIX_FMT_YUVA422P;
	case VIDEO_FORMAT_YUVA:
		return AV_PIX_FMT_YUVA444P;
	case VIDEO_FORMAT_NONE:
	case VIDEO_FORMAT_AYUV:
	default:
		return AV_PIX_FMT_NONE;
	}
}

}  // namespace detail

struct SWScaleContext {
	~SWScaleContext() {
		if (m_Ctx != nullptr) {
			::sws_freeContext(m_Ctx);
		}
	}

	void scale(::obs_source_frame *inputFrame, void *outBuffer) {
		int srcW = static_cast<int>(inputFrame->width);
		int srcH = static_cast<int>(inputFrame->height);
		::AVPixelFormat srcFormat =
		    detail::convertFrameFormat(inputFrame->format);
		int dstW = static_cast<int>(core::INPUT_WIDTH);
		int dstH = static_cast<int>(core::INPUT_HEIGHT);
		::AVPixelFormat dstFormat = AV_PIX_FMT_BGR24;
		m_Ctx = ::sws_getCachedContext(m_Ctx, srcW, srcH, srcFormat, dstW, dstH,
		    dstFormat, SWS_POINT, nullptr, nullptr, nullptr);
		if (m_Ctx == nullptr) {
			throw std::bad_alloc();
		}
		int inStrides[4];
		for (std::size_t i = 0; i < 4; ++i) {
			inStrides[i] = static_cast<int>(inputFrame->linesize[i]);
		}
		std::uint8_t *outBuffers[4] = {
		    reinterpret_cast<std::uint8_t *>(outBuffer)};
		int outStrides[] = {core::INPUT_WIDTH * 3, 0};
		::sws_scale(m_Ctx, inputFrame->data, inStrides, 0, srcH, outBuffers,
		    outStrides);
	}

private:
	::SwsContext *m_Ctx = nullptr;
};

struct JoshUpscaleFilter {
	static ::obs_source_info *getSourceInfo();

private:
	explicit JoshUpscaleFilter(::obs_source_t *source);
	~JoshUpscaleFilter();

	const char *getName() {
		return "JoshUpscale";
	}

	static void *create(
	    [[maybe_unused]] ::obs_data_t *params, ::obs_source_t *source) {
		try {
			return new JoshUpscaleFilter(source);
		} catch (...) {
			return nullptr;
		}
	}

	static void destroy(void *data) {
		delete reinterpret_cast<JoshUpscaleFilter *>(data);
	}

	::obs_source_frame *filterVideo(::obs_source_frame *frame);

	template <auto Ptr>
	struct Callback;

	template <typename R, typename... T, R (JoshUpscaleFilter::*Ptr)(T...)>
	struct Callback<Ptr> {
		static R (*getPtr())(void *, T...) {
			return [](void *self, T... params) -> R {
				return (reinterpret_cast<JoshUpscaleFilter *>(self)->*Ptr)(
				    params...);
			};
		}
	};

	::obs_source_t *m_Source;
	std::unique_ptr<core::Runtime> m_Runtime;
	AVBuffer m_InputBuffer;
	OBSFrame m_OutputFrame;
	SWScaleContext m_ScaleContext;
};

}  // namespace obs

}  // namespace JoshUpscale
