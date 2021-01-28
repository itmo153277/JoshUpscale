#pragma once
#pragma warning(disable : 26812)

extern "C" {

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>
#include <libavutil/hwcontext.h>
#include <libavutil/time.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
}

#include <cassert>
#include <cstdint>
#include <exception>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>

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
AVError callOrThrow(F f, Params &&... params) {
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
	::AVPixelFormat hwPixelFormat;

private:
	struct SInternalData {
		::AVPixelFormat hwPixelFormat;

		SInternalData(::AVPixelFormat hwPixelFormat)
		    : hwPixelFormat(hwPixelFormat) {
		}
	};

	std::unique_ptr<SInternalData> m_InternalData;

	friend SVideoStreamInfo openVideoStream(::AVFormatContext *, DXVA);
};

smart::AVFormatContext openDshowSource(const char *source);
SVideoStreamInfo openVideoStream(::AVFormatContext *pFormatCtx, DXVA dxva);
SStreamInfo openAudioStream(::AVFormatContext *pFormatCtx);

}  // namespace ffmpeg
