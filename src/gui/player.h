// Copyright 2021 Ivanov Viktor

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

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
	    const char *sourceType, ffmpeg::DXVA dxva, const char *audioOut,
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
	ProcessCallback m_ProcessCallback;
	ProcessCallback m_WriteCallback;
	ffmpeg::SDecoder m_Decoder;
	smart::SDL_Window m_Window;
	smart::SDL_Renderer m_Renderer;
	smart::SDL_Texture m_Texture;
	smart::SDL_mutex m_Mutex;
	::Uint32 m_PresentEvent;
	std::unique_ptr<smart::SDL_AudioDevice> m_AudioDevice = nullptr;
	smart::AVPointer m_VideoBuffer = nullptr;
	std::size_t m_VideoBufferStride;
	std::atomic<ffmpeg::pts_t> m_MasterClock{0};
	std::atomic<ffmpeg::pts_t> m_ResetMasterClock{AV_NOPTS_VALUE};
	ffmpeg::pts_t m_StreamStart;

	void videoThread();
	void audioThread();
};

}  // namespace player
