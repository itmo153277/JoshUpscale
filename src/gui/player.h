// Copyright 2021 Ivanov Viktor

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>

#include "ffmpeg_decoder.h"
#include "ffmpeg_wrappers.h"
#include "sdl_wrappers.h"

namespace player {

using ProcessCallback = std::function<void(void *buf, std::size_t stride)>;
using PresentCallback = std::function<void()>;

struct SPlayer {
	SPlayer() = delete;
	SPlayer(std::size_t inputWidth, std::size_t inputHeight,
	    std::size_t outputWidth, std::size_t outputHeight, const char *source,
	    const char *sourceType, const char *sourceOptions, ffmpeg::DXVA dxva,
	    const char *audioOut, bool showDebugInfo,
	    ProcessCallback processCallback, ProcessCallback writeCallback);
	SPlayer(const SPlayer &s) = delete;
	SPlayer(SPlayer &&) = delete;
	~SPlayer();

	void play(PresentCallback cb);
	void stop();

private:
	std::size_t m_InputWidth;
	std::size_t m_InputHeight;
	std::size_t m_OutputWidth;
	std::size_t m_OutputHeight;
	bool m_ShowDebugInfo;
	ProcessCallback m_ProcessCallback;
	ProcessCallback m_WriteCallback;
	ffmpeg::SDecoder m_Decoder;
	smart::SDL_Window m_Window;
	smart::SDL_Renderer m_Renderer;
	smart::SDL_Texture m_Texture;
	smart::SDL_mutex m_Mutex;
	::Uint32 m_PresentEvent;
	std::unique_ptr<smart::SDL_AudioDevice> m_AudioDevice = nullptr;
	smart::AVImage m_VideoBuffer;
	std::atomic<ffmpeg::pts_t> m_MasterClock{0};
	ffmpeg::pts_t m_StreamStart;
	Uint32 m_RenderTicks = 0;
	std::shared_mutex m_VideoSyncMutex;
	ffmpeg::pts_t m_VideoTimestamp = AV_NOPTS_VALUE;
	ffmpeg::pts_t m_VideoPts = AV_NOPTS_VALUE;
	ffmpeg::pts_t m_VideoStartPts = AV_NOPTS_VALUE;
	double m_VideoJitter = 0;
	::AVRational m_VideoPtsConv{};
	std::shared_mutex m_AudioSyncMutex;
	ffmpeg::pts_t m_AudioTimestamp = AV_NOPTS_VALUE;
	ffmpeg::pts_t m_AudioPts = AV_NOPTS_VALUE;
	ffmpeg::pts_t m_AudioPtsCorr = AV_NOPTS_VALUE;
	ffmpeg::pts_t m_AudioStartPts = AV_NOPTS_VALUE;
	double m_AudioJitter = 0;
	::AVRational m_AudioPtsConv{};
	::AVRational m_AudioQueueConv{};

	void videoThread();
	void audioThread();
	void printText(const std::string &s, int x, int y);
	void printLine(const std::string &s, int x, int y);
	ffmpeg::pts_t getVideoClock();
	void syncVideo(::AVFrame *frame);
	ffmpeg::pts_t getAudioClock();
	void syncAudio(::AVFrame *frame);
};

}  // namespace player
