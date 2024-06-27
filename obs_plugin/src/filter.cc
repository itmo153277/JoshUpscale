// Copyright 2023 Viktor Ivanov

#include "JoshUpscale/obs/filter.h"

#include <JoshUpscale/core.h>

#include <cstddef>
#include <stdexcept>
#include <thread>

namespace JoshUpscale {

namespace obs {

namespace {

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
	default:
		return AV_PIX_FMT_NONE;
	}
}

}  // namespace

::obs_source_info *JoshUpscaleFilter::getSourceInfo() {
	static struct Data {
		::obs_source_info info = {};
		Data() {
#define CALLBACK_DEF(name) (Callback<&JoshUpscaleFilter::name>::getPtr())
			info.id = "joshupscale";
			info.type = OBS_SOURCE_TYPE_FILTER;
			info.output_flags = OBS_SOURCE_ASYNC_VIDEO;
			info.get_name = CALLBACK_DEF(getName);
			info.create = CALLBACK_DEF(create);
			info.destroy = CALLBACK_DEF(destroy);
			info.filter_video = CALLBACK_DEF(filterVideo);
#undef CALLBACK_DEF
		}
	} data;
	return &data.info;
}

JoshUpscaleFilter::JoshUpscaleFilter(::obs_source_t *source)
    : m_Source(source)
    , m_InputBuffer(core::INPUT_WIDTH * core::INPUT_HEIGHT * 3)
    , m_OutputFrame(core::OUTPUT_WIDTH, core::OUTPUT_HEIGHT) {
	m_Runtime.reset(core::createRuntime(0, obs_module_file("model.yaml")));
}

JoshUpscaleFilter::~JoshUpscaleFilter() {
	if (m_SwsCtx != nullptr) {
		::sws_freeContext(m_SwsCtx);
	}
	// Ensure that injected frame is out
	::obs_source_t *parent = ::obs_filter_get_parent(m_Source);
	for (;;) {
		auto *frame = ::obs_source_get_frame(parent);
		::obs_source_release_frame(parent, frame);
		if (frame != m_OutputFrame) {
			break;
		}
		std::this_thread::yield();
	}
}

const char *JoshUpscaleFilter::getName() noexcept {
	return "JoshUpscale";
}

void *JoshUpscaleFilter::create(
    [[maybe_unused]] ::obs_data_t *params, ::obs_source_t *source) noexcept {
	try {
		return new JoshUpscaleFilter(source);
	} catch (...) {
		return nullptr;
	}
}

void JoshUpscaleFilter::destroy(void *data) noexcept {
	delete reinterpret_cast<JoshUpscaleFilter *>(data);
}

void JoshUpscaleFilter::copyFrame(::obs_source_frame *frame) {
	int srcW = static_cast<int>(frame->width);
	int srcH = static_cast<int>(frame->height);
	::AVPixelFormat srcFormat = convertFrameFormat(frame->format);
	int dstW = static_cast<int>(core::INPUT_WIDTH);
	int dstH = static_cast<int>(core::INPUT_HEIGHT);
	::AVPixelFormat dstFormat = AV_PIX_FMT_BGR24;
	m_SwsCtx = ::sws_getCachedContext(m_SwsCtx, srcW, srcH, srcFormat, dstW,
	    dstH, dstFormat, SWS_POINT, nullptr, nullptr, nullptr);
	if (m_SwsCtx == nullptr) {
		throw std::runtime_error("SwsCtx failure");
	}
	if (::format_is_yuv(frame->format)) {
		float rangeCoeff = frame->full_range ? (255.0F / 224.0F) : 1.0F;
		int coeff[4] = {
		    static_cast<int>(65536 * frame->color_matrix[2] * rangeCoeff),
		    static_cast<int>(65536 * frame->color_matrix[9] * rangeCoeff),
		    static_cast<int>(65536 * -frame->color_matrix[5] * rangeCoeff),
		    static_cast<int>(65536 * -frame->color_matrix[6] * rangeCoeff),
		};
		if (::sws_setColorspaceDetails(m_SwsCtx, coeff,
		        static_cast<int>(frame->full_range),
		        ::sws_getCoefficients(SWS_CS_DEFAULT), 1, 0, 1 << 16,
		        1 << 16) < 0) {
			throw std::runtime_error("SwsCtx failure");
		}
		m_OutputFrame->full_range = true;
	} else {
		m_OutputFrame->full_range = frame->full_range;
	}
	int inStrides[4] = {};
	for (std::size_t i = 0; i < 4; ++i) {
		inStrides[i] = static_cast<int>(frame->linesize[i]);
	}
	std::uint8_t *outBuffers[4] = {
	    reinterpret_cast<std::uint8_t *>(m_InputBuffer.get())};
	int outStrides[4] = {core::INPUT_WIDTH * 3};
	::sws_scale(
	    m_SwsCtx, frame->data, inStrides, 0, srcH, outBuffers, outStrides);
	m_OutputFrame->timestamp = frame->timestamp;
	m_OutputFrame->flip = frame->flip;
	::os_atomic_inc_long(&m_OutputFrame->refs);
}

::obs_source_frame *JoshUpscaleFilter::filterVideo(
    ::obs_source_frame *frame) noexcept {
	::obs_source_t *parent = ::obs_filter_get_parent(m_Source);
	try {
		copyFrame(frame);
		core::Image inputImage = {
		    .ptr = m_InputBuffer.get(),
		    .stride = static_cast<std::ptrdiff_t>(core::INPUT_WIDTH * 3),
		    .width = core::INPUT_WIDTH,
		    .height = core::INPUT_HEIGHT,
		};
		core::Image outputImage = {
		    .ptr = m_OutputFrame->data[0],
		    .stride = static_cast<std::ptrdiff_t>(m_OutputFrame->linesize[0]),
		    .width = core::OUTPUT_WIDTH,
		    .height = core::OUTPUT_HEIGHT,
		};
		m_Runtime->processImage(inputImage, outputImage);
	} catch (...) {
		return frame;
	}
	::obs_source_release_frame(parent, frame);
	return m_OutputFrame;
}

}  // namespace obs

}  // namespace JoshUpscale
