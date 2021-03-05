// Copyright 2021 Ivanov Viktor

#pragma once

#include <atomic>
#include <functional>

#include "ffmpeg_wrappers.h"

namespace ffmpeg {

constexpr std::size_t MAX_VIDEO_QUEUE = 64 * 1024 * 1024;
constexpr std::size_t MAX_AUDIO_QUEUE = 64 * 1024 * 1024;

using DecoderCallback = std::function<pts_t(::AVFrame *frame)>;

struct SDecoder {
	SDecoder() = delete;
	SDecoder(const char *source, const char *sourceType, const char *options,
	    ffmpeg::DXVA dxva);
	SDecoder(const SDecoder &) = delete;
	SDecoder(SDecoder &&) = delete;
	~SDecoder();

	void terminate();
	void captureLoop();
	void videoDecoderLoop(DecoderCallback cb);
	void audioDecoderLoop(DecoderCallback cb);
	smart::AVFrame audioDecodeFrameOrThrow(pts_t minPts = AV_NOPTS_VALUE);

	const SVideoStreamInfo *getVideoStreamInfo() const {
		return &m_VideoStreamInfo;
	}
	const SStreamInfo *getAudioStreamInfo() const {
		return &m_AudioStreamInfo;
	}
	::AVFormatContext *getFormatContext() {
		return m_FormatCtx.get();
	}

private:
	smart::AVFormatContext m_FormatCtx;
	SVideoStreamInfo m_VideoStreamInfo;
	SStreamInfo m_AudioStreamInfo;
	SPacketQueue m_VideoQueue;
	SPacketQueue m_AudioQueue;
	std::atomic<bool> m_Terminated;
};

}  // namespace ffmpeg
