// Copyright 2021 Ivanov Viktor

#pragma once
#pragma warning(disable : 26812)

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/time.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>

namespace ffmpeg {

using AVError = int;
using pts_t = std::int64_t;

struct AVException : std::exception {
	AVError averror;

	AVException() = delete;
	explicit AVException(AVError averror) : averror(averror) {
		AVError ret = av_strerror(averror, m_Msg, AV_ERROR_MAX_STRING_SIZE);
		if (ret < 0) {
			m_Msg[0] = 0;
		}
	}
	const char *what() const noexcept override {
		if (m_Msg[0]) {
			return m_Msg;
		} else {
			return "Unknown error";
		}
	}

private:
	char m_Msg[AV_ERROR_MAX_STRING_SIZE];
};

template <typename F, typename... Params>
AVError callOrThrow(F f, Params &&...params) {
	AVError ret = f(std::forward<Params>(params)...);
	if (ret < 0) {
		throw AVException(ret);
	}
	return ret;
}

}  // namespace ffmpeg

namespace smart {

struct AVPointer : std::unique_ptr<void, decltype(&::av_free)> {
	using unique_ptr = std::unique_ptr<void, decltype(&::av_free)>;

	AVPointer() : unique_ptr(nullptr, &::av_free) {
	}
	AVPointer(const AVPointer &) = delete;
	AVPointer(AVPointer &&) noexcept = default;
	AVPointer(void *p) : unique_ptr(p, &::av_free) {
	}
};

struct AVDictionary {
	AVDictionary() = default;
	AVDictionary(const AVDictionary &) = delete;
	AVDictionary(AVDictionary &&s) noexcept {
		m_pDictionary = s.m_pDictionary;
		s.m_pDictionary = nullptr;
	}
	~AVDictionary() {
		::av_dict_free(&m_pDictionary);
	}

	::AVDictionary *&get() {
		return m_pDictionary;
	}

private:
	::AVDictionary *m_pDictionary = nullptr;
};

struct AVPacket {
	AVPacket() {
		m_pPacket = ::av_packet_alloc();
		if (m_pPacket == nullptr) {
			throw std::bad_alloc();
		}
	}
	AVPacket(const AVPacket &) = delete;
	AVPacket(AVPacket &&s) noexcept {
		m_pPacket = s.m_pPacket;
		s.m_pPacket = nullptr;
	}
	~AVPacket() {
		::av_packet_free(&m_pPacket);
	}

	::AVPacket *operator->() {
		return m_pPacket;
	}
	::AVPacket *get() {
		return m_pPacket;
	}

private:
	::AVPacket *m_pPacket;
};

struct AVImage {
	std::uint8_t *data[AV_NUM_DATA_POINTERS];
	int linesize[AV_NUM_DATA_POINTERS];

	AVImage(int width, int height, AVPixelFormat format) {
		ffmpeg::callOrThrow(
		    ::av_image_alloc, data, linesize, width, height, format, 16);
		m_Ptr.reset(data[0]);
	}
	AVImage(const AVImage &) = delete;
	AVImage(AVImage &&s) noexcept = default;

private:
	AVPointer m_Ptr;
};

struct AVFrame {
	AVFrame() {
		m_pFrame = ::av_frame_alloc();
		if (m_pFrame == nullptr) {
			throw std::bad_alloc();
		}
	}
	AVFrame(const AVFrame &) = delete;
	AVFrame(AVFrame &&s) noexcept {
		m_pFrame = s.m_pFrame;
		s.m_pFrame = nullptr;
	}
	~AVFrame() {
		::av_frame_free(&m_pFrame);
	}

	::AVFrame *operator->() {
		return m_pFrame;
	}
	::AVFrame *get() {
		return m_pFrame;
	}

private:
	::AVFrame *m_pFrame;
};

struct AVCodecContext {
	AVCodecContext() = default;
	explicit AVCodecContext(::AVCodecContext *p) : m_pCodecCtx(p) {
	}
	explicit AVCodecContext(const ::AVCodec *pCodec) {
		allocate(pCodec);
	}
	AVCodecContext(const AVCodecContext &) = delete;
	AVCodecContext(AVCodecContext &&s) noexcept {
		m_pCodecCtx = s.m_pCodecCtx;
		s.m_pCodecCtx = nullptr;
	}
	~AVCodecContext() {
		::avcodec_free_context(&m_pCodecCtx);
	}

	void allocate(const ::AVCodec *pCodec) {
		assert(m_pCodecCtx == nullptr);
		m_pCodecCtx = ::avcodec_alloc_context3(pCodec);
		if (m_pCodecCtx == nullptr) {
			throw std::bad_alloc();
		}
	}

	operator bool() const {
		return m_pCodecCtx;
	}
	::AVCodecContext *operator->() {
		return m_pCodecCtx;
	}
	const ::AVCodecContext *operator->() const {
		return m_pCodecCtx;
	}
	::AVCodecContext *get() {
		return m_pCodecCtx;
	}
	const ::AVCodecContext *get() const {
		return m_pCodecCtx;
	}

private:
	::AVCodecContext *m_pCodecCtx = nullptr;
};

struct AVFormatContext {
	AVFormatContext() = default;
	AVFormatContext(const AVFormatContext &) = delete;
	AVFormatContext(AVFormatContext &&s) noexcept {
		m_pFormatCtx = s.m_pFormatCtx;
		s.m_pFormatCtx = nullptr;
	}
	~AVFormatContext() {
		::avformat_close_input(&m_pFormatCtx);
	}

	::AVFormatContext *operator->() {
		return m_pFormatCtx;
	}
	::AVFormatContext *&get() {
		return m_pFormatCtx;
	}

private:
	::AVFormatContext *m_pFormatCtx = nullptr;
};

struct AVFilterGraph {
	AVFilterGraph() {
		m_pGraph = ::avfilter_graph_alloc();
		if (!m_pGraph) {
			throw std::bad_alloc();
		}
	}
	AVFilterGraph(const AVFilterGraph &) = delete;
	AVFilterGraph(AVFilterGraph &&s) noexcept {
		m_pGraph = s.m_pGraph;
		s.m_pGraph = nullptr;
	}
	~AVFilterGraph() {
		::avfilter_graph_free(&m_pGraph);
	}

	::AVFilterGraph *operator->() {
		return m_pGraph;
	}
	::AVFilterGraph *get() {
		return m_pGraph;
	}

private:
	::AVFilterGraph *m_pGraph = nullptr;
};

struct AVFilterInOut {
	AVFilterInOut() {
		m_InOut = ::avfilter_inout_alloc();
		if (!m_InOut) {
			throw std::bad_alloc();
		}
	}
	AVFilterInOut(const AVFilterInOut &) = delete;
	AVFilterInOut(AVFilterInOut &s) noexcept {
		m_InOut = s.m_InOut;
		s.m_InOut = nullptr;
	}
	~AVFilterInOut() {
		::avfilter_inout_free(&m_InOut);
	}

	::AVFilterInOut *operator->() {
		return m_InOut;
	}
	::AVFilterInOut *&get() {
		return m_InOut;
	}

private:
	::AVFilterInOut *m_InOut = nullptr;
};

struct SwsContext {
	SwsContext() = default;
	SwsContext(const SwsContext &) = delete;
	SwsContext(SwsContext &&s) noexcept {
		m_pCtx = s.m_pCtx;
		s.m_pCtx = nullptr;
	}
	~SwsContext() {
		::sws_freeContext(m_pCtx);
	}

	void reinit(int srcWidth, int srcHeight, ::AVPixelFormat srcFormat,
	    int dstWidth, int dstHeight, ::AVPixelFormat dstFormat, int flags) {
		m_pCtx = ::sws_getCachedContext(m_pCtx, srcWidth, srcHeight, srcFormat,
		    dstWidth, dstHeight, dstFormat, flags, nullptr, nullptr, nullptr);
		if (!m_pCtx) {
			throw std::bad_alloc();
		}
	}

	::SwsContext *get() {
		return m_pCtx;
	}

private:
	::SwsContext *m_pCtx = nullptr;
};

struct SwrContext {
	SwrContext() {
		m_pCtx = ::swr_alloc();
		if (!m_pCtx) {
			throw std::bad_alloc();
		}
	}
	SwrContext(const SwrContext &) = delete;
	SwrContext(SwrContext &&s) noexcept {
		m_pCtx = s.m_pCtx;
		s.m_pCtx = nullptr;
	}
	~SwrContext() {
		::swr_free(&m_pCtx);
	}

	::SwrContext *get() {
		return m_pCtx;
	}

private:
	::SwrContext *m_pCtx = nullptr;
};

}  // namespace smart

namespace ffmpeg {

void init();

enum class DXVA { AUTO, FORCED, OFF };

struct SPacketQueue {
	SPacketQueue() = delete;
	explicit SPacketQueue(std::size_t maxQueue) : m_MaxQueue(maxQueue) {
	}
	SPacketQueue(const SPacketQueue &) = delete;
	SPacketQueue(SPacketQueue &&) = delete;
	~SPacketQueue();

	smart::AVPacket consume();         // Waits for a package if dry
	smart::AVPacket consumeOrThrow();  // Throws if dry
	void produce(smart::AVPacket &&packet);
	void terminate();
	void flush();

private:
	std::size_t m_Size = 0;
	bool m_Terminated = false;
	bool m_Flush = false;
	std::size_t m_MaxQueue;
	std::queue<smart::AVPacket> m_PacketQueue;
	std::mutex m_Mutex;
	std::condition_variable m_cv;
};

struct SStreamInfo {
	int streamIndex = -1;
	smart::AVCodecContext codecCtx;
};

struct SVideoStreamInfo : SStreamInfo {
	::AVPixelFormat hwPixelFormat = AV_PIX_FMT_NONE;

private:
	struct SInternalData {
		::AVPixelFormat hwPixelFormat;

		explicit SInternalData(::AVPixelFormat hwPixelFormat)
		    : hwPixelFormat(hwPixelFormat) {
		}
	};

	std::unique_ptr<SInternalData> m_InternalData;

	friend SVideoStreamInfo openVideoStream(::AVFormatContext *, DXVA);
};

struct SGraphInfo {
	smart::AVFilterGraph graph;
	::AVFilterContext *srcCtx;
	::AVFilterContext *sinkCtx;
};

smart::AVFormatContext openSource(const char *source, const char *sourceType);
SVideoStreamInfo openVideoStream(::AVFormatContext *pFormatCtx, DXVA dxva);
SStreamInfo openAudioStream(::AVFormatContext *pFormatCtx);
SGraphInfo createVideoGraph(const SVideoStreamInfo *streamInfo,
    ::AVFormatContext *pFormatCtx, ::AVFrame *frame, ::AVPixelFormat outFormat,
    const char *filterStr);

}  // namespace ffmpeg
