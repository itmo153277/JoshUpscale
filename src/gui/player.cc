// Copyright 2021 Ivanov Viktor

#include "player.h"

#include <algorithm>
#include <deque>
#include <exception>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>
#include <tuple>
#include <vector>

constexpr ::AVSampleFormat sdlToFfmpegSampleFormat(::SDL_AudioFormat fmt) {
	switch (fmt) {
	case AUDIO_U8:
		return AV_SAMPLE_FMT_U8;
	case AUDIO_S16SYS:
		return AV_SAMPLE_FMT_S16;
	case AUDIO_S32SYS:
		return AV_SAMPLE_FMT_S32;
	case AUDIO_F32SYS:
		return AV_SAMPLE_FMT_FLT;
	default:
		return AV_SAMPLE_FMT_NONE;
	}
}

constexpr ::SDL_AudioFormat ffmpegToSdlSampleFormat(
    ::AVSampleFormat sampleFormat) {
	switch (sampleFormat) {
	case AV_SAMPLE_FMT_U8:
		return AUDIO_U8;
	case AV_SAMPLE_FMT_S16:
		return AUDIO_S16SYS;
	case AV_SAMPLE_FMT_S32:
		return AUDIO_S32SYS;
	case AV_SAMPLE_FMT_FLT:
		return AUDIO_F32SYS;
	default:
		return 0;
	}
}

player::SPlayer::SPlayer(std::size_t inputWidth, std::size_t inputHeight,
    std::size_t outputWidth, std::size_t outputHeight, const char *source,
    const char *sourceType, ffmpeg::DXVA dxva, const char *audioOut,
    bool showDebugInfo, ProcessCallback processCallback,
    ProcessCallback writeCallback)
    : m_InputWidth{inputWidth}
    , m_InputHeight{inputHeight}
    , m_OutputWidth{outputWidth}
    , m_OutputHeight{outputHeight}
    , m_ShowDebugInfo{showDebugInfo}
    , m_ProcessCallback{processCallback}
    , m_WriteCallback{writeCallback}
    , m_Decoder{source, sourceType, dxva}
    , m_Window{sdl::allocOrThrow(::SDL_CreateWindow("JoshUpscale",
          SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
          static_cast<int>(m_OutputWidth), static_cast<int>(m_OutputHeight),
          SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_HIDDEN))}
    , m_Renderer{sdl::allocOrThrow(
          ::SDL_CreateRenderer(m_Window.get(), -1, SDL_RENDERER_ACCELERATED))}
    , m_Texture{sdl::allocOrThrow(::SDL_CreateTexture(m_Renderer.get(),
          SDL_PIXELFORMAT_BGR24, SDL_TEXTUREACCESS_STREAMING,
          static_cast<int>(m_OutputWidth), static_cast<int>(m_OutputHeight)))}
    , m_Mutex{sdl::allocOrThrow(::SDL_CreateMutex())}
    , m_PresentEvent{::SDL_RegisterEvents(1)} {
	if (m_Decoder.getAudioStreamInfo()->codecCtx) {
		const ::AVCodecContext *pAudioCtx =
		    m_Decoder.getAudioStreamInfo()->codecCtx.get();
		::SDL_AudioSpec audioSpec{};
		audioSpec.freq = pAudioCtx->sample_rate;
		audioSpec.format = ffmpegToSdlSampleFormat(pAudioCtx->sample_fmt);
		audioSpec.channels = pAudioCtx->channels;
		m_AudioDevice =
		    std::make_unique<smart::SDL_AudioDevice>(audioOut, &audioSpec);
		if (sdlToFfmpegSampleFormat(m_AudioDevice->getAudioSpec()->format) ==
		    AV_SAMPLE_FMT_NONE) {
			throw std::invalid_argument("Unsupported audio format");
		}
	}
	m_VideoBufferStride = m_OutputWidth * 3 * sizeof(std::uint8_t);
	if ((m_VideoBufferStride % 16 != 0)) {
		m_VideoBufferStride += 16 - (m_VideoBufferStride % 16);
	}
	m_VideoBuffer.reset(::av_malloc(m_VideoBufferStride * m_OutputHeight));
	if (!m_VideoBuffer) {
		throw std::bad_alloc();
	}
	::av_dump_format(m_Decoder.getFormatContext(), 0, source, 0);
	m_StreamStart = ::av_gettime_relative();
}

player::SPlayer::~SPlayer() {
}

void player::SPlayer::play(PresentCallback cb) {
	::SDL_Event e;
	auto startTime = ::SDL_GetTicks();
	std::size_t lastPeriod = 0;
	std::size_t frameCount = 0;
	std::size_t framesToDelete = 0;
	if (m_AudioDevice) {
		::SDL_PauseAudioDevice(m_AudioDevice->getDeviceId(), false);
	}
	::SDL_ShowWindow(m_Window.get());
	sdl::callOrThrow(::SDL_RenderClear, m_Renderer.get());
	::SDL_RenderPresent(m_Renderer.get());

	std::exception_ptr exception = nullptr;
	std::vector<std::thread> threads;
	std::atomic<std::size_t> threadsRunning{0};
	std::mutex threadMutex;
	try {
		{
			std::lock_guard<std::mutex> lock(threadMutex);
			threads.emplace_back(
			    [this, &exception, &threadsRunning, &threadMutex] {
				    { std::lock_guard<std::mutex> lock(threadMutex); }
				    try {
					    m_Decoder.captureLoop();
					    if (--threadsRunning == 0) {
						    stop();
					    }
				    } catch (...) {
					    exception = std::current_exception();
					    stop();
				    }
			    });
			threads.emplace_back(
			    [this, &exception, &threadsRunning, &threadMutex] {
				    { std::lock_guard<std::mutex> lock(threadMutex); }
				    try {
					    videoThread();
					    if (--threadsRunning == 0) {
						    stop();
					    }
				    } catch (...) {
					    exception = std::current_exception();
					    stop();
				    }
			    });
			if (m_AudioDevice) {
				threads.emplace_back(
				    [this, &exception, &threadsRunning, &threadMutex] {
					    { std::lock_guard<std::mutex> lock(threadMutex); }
					    try {
						    audioThread();
						    if (--threadsRunning == 0) {
							    stop();
						    }
					    } catch (...) {
						    exception = std::current_exception();
						    stop();
					    }
				    });
			}
			threadsRunning = threads.size();
		}
		std::deque<std::tuple<Uint32, Uint32>> points;
		Uint32 frameStartTicks = ::SDL_GetTicks();
		std::size_t sampleSize = 0;
		if (m_AudioDevice) {
			sampleSize = ffmpeg::callOrThrow(::av_samples_get_buffer_size,
			    nullptr, m_AudioDevice->getAudioSpec()->channels, 1,
			    static_cast<::AVSampleFormat>(sdlToFfmpegSampleFormat(
			        m_AudioDevice->getAudioSpec()->format)),
			    1);
		}
		while (::SDL_WaitEvent(&e)) {
			if (e.type == SDL_QUIT) {
				m_Decoder.terminate();
				break;
			}
			if (e.type == m_PresentEvent) {
				Uint32 renderTicks;
				{
					sdl::SLockGuard lock(m_Mutex.get());
					sdl::callOrThrow(::SDL_UpdateTexture, m_Texture.get(),
					    nullptr, m_VideoBuffer.get(),
					    static_cast<int>(m_VideoBufferStride));
					renderTicks = m_RenderTicks;
				}
				sdl::callOrThrow(::SDL_RenderClear, m_Renderer.get());
				sdl::callOrThrow(::SDL_RenderCopy, m_Renderer.get(),
				    m_Texture.get(), nullptr, nullptr);
				auto curTime = ::SDL_GetTicks();
				if (m_ShowDebugInfo) {
					auto frameTicks = curTime - frameStartTicks;
					frameStartTicks = curTime;
					if (points.size() > m_OutputWidth / 5) {
						points.pop_front();
					}
					points.emplace_back(frameTicks, renderTicks);
					::SDL_SetRenderDrawColor(
					    m_Renderer.get(), 128, 128, 128, 128);
					std::vector<::SDL_Point> frameTickPoints{points.size()};
					std::vector<::SDL_Point> renderTickPoints{points.size()};
					Uint32 maxValue = 0;
					for (std::size_t i = 0; i < points.size(); ++i) {
						auto frameTicks = std::get<0>(points[i]);
						auto renderTicks = std::get<1>(points[i]);
						frameTickPoints[i].x = static_cast<int>(i) * 5;
						renderTickPoints[i].x = static_cast<int>(i) * 5;
						frameTickPoints[i].y =
						    static_cast<int>(m_OutputHeight) -
						    2 * static_cast<int>(frameTicks);
						renderTickPoints[i].y =
						    static_cast<int>(m_OutputHeight) -
						    2 * static_cast<int>(renderTicks);
						if (maxValue < frameTicks) {
							maxValue = frameTicks;
						}
						if (maxValue < renderTicks) {
							maxValue = renderTicks;
						}
					}
					for (int i = 1;
					     i < std::min(static_cast<int>(maxValue) / 10 + 1,
					             static_cast<int>(m_OutputHeight) / 20);
					     ++i) {
						::SDL_RenderDrawLine(m_Renderer.get(), 0,
						    static_cast<int>(m_OutputHeight) - i * 20,
						    static_cast<int>(m_OutputWidth),
						    static_cast<int>(m_OutputHeight) - i * 20);
						std::stringstream ss;
						ss << i * 10 << " ms";
						::stringRGBA(m_Renderer.get(), 0,
						    static_cast<int>(m_OutputHeight) - i * 20,
						    ss.str().c_str(), 128, 128, 128, 128);
					}
					::SDL_SetRenderDrawColor(
					    m_Renderer.get(), 255, 0, 0, SDL_ALPHA_OPAQUE);
					::SDL_RenderDrawLines(m_Renderer.get(),
					    frameTickPoints.data(),
					    static_cast<int>(frameTickPoints.size()));
					::SDL_SetRenderDrawColor(
					    m_Renderer.get(), 0, 0, 255, SDL_ALPHA_OPAQUE);
					::SDL_RenderDrawLines(m_Renderer.get(),
					    renderTickPoints.data(),
					    static_cast<int>(renderTickPoints.size()));
					std::stringstream ss;
					auto currentTimestamp =
					    ::av_gettime_relative() - m_StreamStart;
					auto videoLag = m_VideoTimestamp - currentTimestamp;
					ss << "Video lag: " << std::setprecision(3) << std::fixed
					   << videoLag / 1000.0 << " ms";
					::stringRGBA(m_Renderer.get(), 2, 2, ss.str().c_str(), 255,
					    255, 255, SDL_ALPHA_OPAQUE);
					if (m_AudioDevice) {
						ss.str("");
						auto audioLag = m_AudioTimestamp - currentTimestamp;
						auto audioQueueBytes = ::SDL_GetQueuedAudioSize(
						    m_AudioDevice->getDeviceId());
						auto audioQueueTime = ::av_rescale(audioQueueBytes,
						    1000000,
						    m_AudioDevice->getAudioSpec()->freq * sampleSize);
						ss << "Audio lag: " << std::setprecision(3)
						   << std::fixed << audioLag / 1000.0 << " ms";
						::stringRGBA(m_Renderer.get(), 2, 10, ss.str().c_str(),
						    255, 255, 255, SDL_ALPHA_OPAQUE);
						ss.str("");
						ss << "Audio queue: " << audioQueueBytes << " bytes / "
						   << std::setprecision(3) << std::fixed
						   << audioQueueTime / 1000.0 << " ms";
						::stringRGBA(m_Renderer.get(), 2, 18, ss.str().c_str(),
						    255, 255, 255, SDL_ALPHA_OPAQUE);
					}
				}
				::SDL_RenderPresent(m_Renderer.get());
				if (m_ShowDebugInfo) {
					frameCount++;
					std::size_t curPeriod =
					    static_cast<std::size_t>(curTime - startTime);
					if (curPeriod >= 1000) {
						double fps =
						    1000.0 * frameCount / (curPeriod + lastPeriod);
						lastPeriod = curPeriod;
						frameCount -= framesToDelete;
						framesToDelete = frameCount;
						startTime = curTime;
						std::stringstream ss;
						ss << "JoshUpscale (FPS: " << std::setprecision(3)
						   << fps << ')';
						::SDL_SetWindowTitle(m_Window.get(), ss.str().c_str());
					}
				}
				cb();
			}
		}
	} catch (...) {
		m_Decoder.terminate();
		throw;
	}
	for (auto &thread : threads) {
		thread.join();
	}
	if (exception != nullptr) {
		std::rethrow_exception(exception);
	}
	::SDL_HideWindow(m_Window.get());
	if (m_AudioDevice) {
		::SDL_PauseAudioDevice(m_AudioDevice->getDeviceId(), true);
	}
}

void player::SPlayer::stop() {
	m_Decoder.terminate();
	::SDL_Event quitEvent = {SDL_QUIT};
	::SDL_PushEvent(&quitEvent);
}

void player::SPlayer::videoThread() {
	std::size_t bufInStride = m_InputWidth * 3 * sizeof(std::uint8_t);
	if ((bufInStride % 16) != 0) {
		bufInStride += 16 - (bufInStride % 16);
	}
	smart::AVPointer bufIn = ::av_malloc(bufInStride * m_InputHeight);
	if (!bufIn) {
		throw std::bad_alloc();
	}
	std::uint8_t *const bufInSlices[4] = {
	    static_cast<std::uint8_t *>(bufIn.get())};
	int bufInStrides[4] = {static_cast<int>(bufInStride)};
	smart::SwsContext swsCtx;
	int inWidth = -1;
	int inHeight = -1;
	::AVPixelFormat inFormat = AV_PIX_FMT_NONE;
	::SDL_Event presentEvent = {m_PresentEvent};
	::AVStream *videoStream =
	    m_Decoder.getFormatContext()
	        ->streams[m_Decoder.getVideoStreamInfo()->streamIndex];
	Uint32 renderStartTicks = ::SDL_GetTicks();
	m_Decoder.videoDecoderLoop([&](::AVFrame *frame) {
		::AVPixelFormat frameFormat =
		    static_cast<::AVPixelFormat>(frame->format);
		if (frame->width != inWidth || frame->height != inHeight ||
		    frameFormat != inFormat) {
			inWidth = frame->width;
			inHeight = frame->height;
			inFormat = frameFormat;
			swsCtx.reinit(inWidth, inHeight, frameFormat,
			    static_cast<int>(m_InputWidth), static_cast<int>(m_InputHeight),
			    AV_PIX_FMT_BGR24, SWS_POINT);
		}
		::sws_scale(swsCtx.get(), frame->data, frame->linesize, 0,
		    frame->height, bufInSlices, bufInStrides);
		m_ProcessCallback(bufIn.get(), bufInStride);
		{
			sdl::SLockGuard lock(m_Mutex.get());
			m_WriteCallback(m_VideoBuffer.get(), m_VideoBufferStride);
			m_RenderTicks = ::SDL_GetTicks() - renderStartTicks;
		}
		ffmpeg::pts_t currentTimeStamp = frame->best_effort_timestamp;
		if (currentTimeStamp != AV_NOPTS_VALUE) {
			currentTimeStamp = ::av_rescale(
			    currentTimeStamp - videoStream->start_time,
			    static_cast<std::int64_t>(videoStream->time_base.num) * 1000000,
			    videoStream->time_base.den);
			m_VideoTimestamp = currentTimeStamp;
		}
		sdl::callOrThrow(::SDL_PushEvent, &presentEvent);
		renderStartTicks = ::SDL_GetTicks();
		return AV_NOPTS_VALUE;
	});
}

void player::SPlayer::audioThread() {
	assert(m_AudioDevice);
	smart::AVFrame convertedFrame;
	convertedFrame->channels = m_AudioDevice->getAudioSpec()->channels;
	convertedFrame->channel_layout = ::av_get_default_channel_layout(
	    m_AudioDevice->getAudioSpec()->channels);
	convertedFrame->sample_rate = m_AudioDevice->getAudioSpec()->freq;
	convertedFrame->format = static_cast<int>(
	    sdlToFfmpegSampleFormat(m_AudioDevice->getAudioSpec()->format));
	smart::SwrContext swrCtx;
	::AVStream *audioStream =
	    m_Decoder.getFormatContext()
	        ->streams[m_Decoder.getAudioStreamInfo()->streamIndex];
	std::uint64_t inLayout = 0;
	int inRate = 0;
	::AVSampleFormat inFormat = AV_SAMPLE_FMT_NONE;
	m_Decoder.audioDecoderLoop([&](::AVFrame *frame) {
		::AVSampleFormat frameFormat =
		    static_cast<::AVSampleFormat>(frame->format);
		if (!frame->channel_layout) {
			frame->channel_layout =
			    ::av_get_default_channel_layout(frame->channels);
		}
		if (frame->channel_layout != inLayout || frame->sample_rate != inRate ||
		    frameFormat != inFormat) {
			inLayout = frame->channel_layout;
			inRate = frame->sample_rate;
			inFormat = frameFormat;
			ffmpeg::callOrThrow(
			    ::swr_config_frame, swrCtx.get(), convertedFrame.get(), frame);
		}
		ffmpeg::callOrThrow(
		    ::swr_convert_frame, swrCtx.get(), convertedFrame.get(), frame);
		std::size_t len = ffmpeg::callOrThrow(::av_samples_get_buffer_size,
		    nullptr, convertedFrame->channels, convertedFrame->nb_samples,
		    static_cast<::AVSampleFormat>(convertedFrame->format), 1);
		ffmpeg::pts_t currentTimeStamp = frame->best_effort_timestamp;
		if (currentTimeStamp != AV_NOPTS_VALUE) {
			currentTimeStamp = ::av_rescale(
			    currentTimeStamp - audioStream->start_time,
			    static_cast<std::int64_t>(audioStream->time_base.num) * 1000000,
			    audioStream->time_base.den);
			m_AudioTimestamp = currentTimeStamp;
		}
		sdl::callOrThrow(::SDL_QueueAudio, m_AudioDevice->getDeviceId(),
		    convertedFrame->data[0], static_cast<Uint32>(len));
		return AV_NOPTS_VALUE;
	});
}
