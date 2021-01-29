// Copyright 2021 Ivanov Viktor

#include "player.h"

#include <exception>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <thread>
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
    ffmpeg::DXVA dxva, const char *audioOut, ProcessCallback processCallback,
    ProcessCallback writeCallback)
    : m_InputWidth{inputWidth}
    , m_InputHeight{inputHeight}
    , m_OutputWidth{outputWidth}
    , m_OutputHeight{outputHeight}
    , m_ProcessCallback{processCallback}
    , m_WriteCallback{writeCallback}
    , m_Decoder{source, dxva}
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
		while (::SDL_WaitEvent(&e)) {
			if (e.type == SDL_QUIT) {
				m_Decoder.terminate();
				break;
			}
			if (e.type == m_PresentEvent) {
				{
					sdl::SLockGuard lock(m_Mutex.get());
					sdl::callOrThrow(::SDL_UpdateTexture, m_Texture.get(),
					    nullptr, m_VideoBuffer.get(),
					    static_cast<int>(m_VideoBufferStride));
				}
				sdl::callOrThrow(::SDL_RenderClear, m_Renderer.get());
				sdl::callOrThrow(::SDL_RenderCopy, m_Renderer.get(),
				    m_Texture.get(), nullptr, nullptr);
				::SDL_RenderPresent(m_Renderer.get());
				frameCount++;
				auto curTime = ::SDL_GetTicks();
				std::size_t curPeriod =
				    static_cast<std::size_t>(curTime - startTime);
				if (curPeriod >= 1000) {
					double fps = 1000.0 * frameCount / (curPeriod + lastPeriod);
					lastPeriod = curPeriod;
					frameCount -= framesToDelete;
					framesToDelete = frameCount;
					startTime = curTime;
					std::stringstream ss;
					ss << "JoshUpscale (FPS: " << std::setprecision(3) << fps
					   << ')';
					::SDL_SetWindowTitle(m_Window.get(), ss.str().c_str());
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
	ffmpeg::pts_t frameStart = m_StreamStart;
	double jitter = 0;
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
		}
		ffmpeg::pts_t curClock = ::av_gettime_relative();
		ffmpeg::pts_t minClock =
		    curClock - m_StreamStart - 500000;  // 500ms behind real time
		ffmpeg::pts_t currentTimeStamp = frame->best_effort_timestamp;
		if (currentTimeStamp != AV_NOPTS_VALUE) {
			currentTimeStamp = ::av_rescale(
			    currentTimeStamp - videoStream->start_time,
			    static_cast<std::int64_t>(videoStream->time_base.num) * 1000000,
			    videoStream->time_base.den);
			if (m_MasterClock == AV_NOPTS_VALUE) {
				m_MasterClock = currentTimeStamp;
			}
			ffmpeg::pts_t masterClock = m_MasterClock;
			if (masterClock < minClock) {
				masterClock = minClock;
			}
			ffmpeg::pts_t frameTime =
			    currentTimeStamp - masterClock + static_cast<int>(jitter);
			ffmpeg::pts_t currentDuration = curClock - frameStart;
			if (currentDuration < frameTime) {
				::av_usleep(
				    static_cast<unsigned int>(frameTime - currentDuration));
				curClock = ::av_gettime_relative();
				currentDuration = curClock - frameStart;
				jitter += (frameTime - currentDuration - jitter) / 2;
			} else {
				jitter = 0;
			}
			m_MasterClock = currentTimeStamp;
		}
		frameStart = curClock;
		sdl::callOrThrow(::SDL_PushEvent, &presentEvent);
		return ::av_rescale(minClock, videoStream->time_base.den,
		           static_cast<std::int64_t>(videoStream->time_base.num) *
		               1000000) +
		       videoStream->start_time;
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
	std::size_t sampleSize = ffmpeg::callOrThrow(::av_samples_get_buffer_size,
	    nullptr, convertedFrame->channels, 1,
	    static_cast<::AVSampleFormat>(convertedFrame->format), 1);
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
		ffmpeg::pts_t curClock = ::av_gettime_relative();
		ffmpeg::pts_t minClock =
		    curClock - m_StreamStart - 600000;  // 600ms behind real time
		ffmpeg::pts_t currentTimeStamp = frame->best_effort_timestamp;
		if (currentTimeStamp != AV_NOPTS_VALUE) {
			currentTimeStamp =
			    ::av_rescale(currentTimeStamp - audioStream->start_time,
			        static_cast<std::int64_t>(audioStream->time_base.num) *
			            1000000,
			        audioStream->time_base.den) -
			    ::av_rescale(
			        ::SDL_GetQueuedAudioSize(m_AudioDevice->getDeviceId()),
			        1000000, convertedFrame->sample_rate * sampleSize);
			auto masterClock = m_MasterClock.load();
			if (currentTimeStamp - masterClock > 50000) {  // 50ms ahead video
				::av_usleep(
				    static_cast<unsigned int>(currentTimeStamp - masterClock));
			} else if (masterClock - currentTimeStamp > 50000 &&
			           currentTimeStamp > minClock) {  // 50ms behind video
				m_MasterClock = currentTimeStamp;
			}
		}
		sdl::callOrThrow(::SDL_QueueAudio, m_AudioDevice->getDeviceId(),
		    convertedFrame->data[0], static_cast<Uint32>(len));
		return ::av_rescale(minClock, audioStream->time_base.den,
		           static_cast<std::int64_t>(audioStream->time_base.num) *
		               1000000) +
		       audioStream->start_time;
	});
}
