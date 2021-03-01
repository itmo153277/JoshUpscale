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
    , m_PresentEvent{::SDL_RegisterEvents(1)}
    , m_VideoBuffer{static_cast<int>(m_OutputWidth),
          static_cast<int>(m_OutputHeight), AV_PIX_FMT_BGR24} {
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
	::AVStream *videoStream =
	    m_Decoder.getFormatContext()
	        ->streams[m_Decoder.getVideoStreamInfo()->streamIndex];
	m_StreamStart = ::av_gettime_relative();
	m_VideoPtsConv.num = videoStream->time_base.num * 1000000;
	m_VideoPtsConv.den = videoStream->time_base.den;
	m_VideoStartPts = videoStream->start_time;
	m_VideoPts = 0;
	m_VideoTimestamp = m_StreamStart;
	if (m_AudioDevice) {
		::AVStream *audioStream =
		    m_Decoder.getFormatContext()
		        ->streams[m_Decoder.getAudioStreamInfo()->streamIndex];
		m_AudioPtsConv.num = audioStream->time_base.num * 1000000;
		m_AudioPtsConv.den = audioStream->time_base.den;
		m_AudioStartPts = audioStream->start_time;
		m_AudioQueueConv.num = 1000000;
		m_AudioQueueConv.den =
		    m_AudioDevice->getAudioSpec()->freq *
		    ffmpeg::callOrThrow(::av_samples_get_buffer_size, nullptr,
		        m_AudioDevice->getAudioSpec()->channels, 1,
		        sdlToFfmpegSampleFormat(m_AudioDevice->getAudioSpec()->format),
		        1);
		m_AudioPts = 0;
		m_AudioTimestamp = m_StreamStart;
	}
	::av_dump_format(m_Decoder.getFormatContext(), 0, source, 0);
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
					    nullptr, m_VideoBuffer.data[0],
					    m_VideoBuffer.linesize[0]);
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
						    static_cast<int>(m_OutputHeight) - i * 20 + 1,
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
					auto videoLag = currentTimestamp - getVideoClock();
					ss << "Video lag: " << std::setprecision(3) << std::fixed
					   << videoLag / 1000.0 << " ms" << std::endl;
					ss << "Video jitter: " << std::setprecision(3) << std::fixed
					   << m_VideoJitter / 1000.0 << " ms" << std::endl;
					if (m_AudioDevice) {
						auto audioLag = currentTimestamp - getAudioClock();
						auto audioQueueBytes = ::SDL_GetQueuedAudioSize(
						    m_AudioDevice->getDeviceId());
						auto queueTime = ::av_rescale(audioQueueBytes,
						    m_AudioQueueConv.num, m_AudioQueueConv.den);
						ss << "Audio lag: " << std::setprecision(3)
						   << std::fixed << audioLag / 1000.0 << " ms"
						   << std::endl;
						ss << "Audio jitter: " << std::setprecision(3)
						   << std::fixed << m_AudioJitter / 1000.0 << " ms"
						   << std::endl;
						ss << "Audio queue: " << audioQueueBytes << " bytes / "
						   << std::setprecision(3) << std::fixed
						   << queueTime / 1000.0 << " ma" << std::endl;
						ss << "A-V lag: " << std::setprecision(3) << std::fixed
						   << (audioLag - videoLag) / 1000.0 << " ms"
						   << std::endl;
					}
					printText(ss.str(), 2, 2);
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
	smart::SwsContext swsCtx;
	int inWidth = -1;
	int inHeight = -1;
	::AVPixelFormat inFormat = AV_PIX_FMT_NONE;
	smart::AVImage bufIn{static_cast<int>(m_InputWidth),
	    static_cast<int>(m_InputHeight), AV_PIX_FMT_BGR24};
	::SDL_Event presentEvent = {m_PresentEvent};
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
		    frame->height, bufIn.data, bufIn.linesize);
		m_ProcessCallback(bufIn.data[0], bufIn.linesize[0]);
		{
			sdl::SLockGuard lock(m_Mutex.get());
			m_WriteCallback(m_VideoBuffer.data[0], m_VideoBuffer.linesize[0]);
			m_RenderTicks = ::SDL_GetTicks() - renderStartTicks;
		}
		syncVideo(frame);
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
		syncAudio(frame);
		sdl::callOrThrow(::SDL_QueueAudio, m_AudioDevice->getDeviceId(),
		    convertedFrame->data[0], static_cast<Uint32>(len));
		return AV_NOPTS_VALUE;
	});
}

void player::SPlayer::printText(const std::string &s, int x, int y) {
	std::stringstream ss;
	std::string line;
	ss.str(s);
	while (std::getline(ss, line)) {
		if (!s.empty()) {
			printLine(line, x, y);
		}
		y += 8;
	}
}

void player::SPlayer::printLine(const std::string &s, int x, int y) {
	int height = 8;
	int width = 8 * static_cast<int>(s.size());
	::boxRGBA(
	    m_Renderer.get(), x, y, x + width, y + height, 128, 128, 128, 128);
	::stringRGBA(
	    m_Renderer.get(), x, y, s.c_str(), 255, 255, 255, SDL_ALPHA_OPAQUE);
}

ffmpeg::pts_t player::SPlayer::getVideoClock() {
	std::shared_lock<std::shared_mutex> lock{m_VideoSyncMutex};
	if (m_VideoPts == AV_NOPTS_VALUE) {
		return AV_NOPTS_VALUE;
	}
	return m_VideoPts + ::av_gettime_relative() - m_VideoTimestamp;
}

void player::SPlayer::syncVideo(::AVFrame *frame) {
	ffmpeg::pts_t newPts = frame->best_effort_timestamp;
	if (newPts != AV_NOPTS_VALUE) {
		if (m_VideoStartPts == AV_NOPTS_VALUE) {
			m_VideoStartPts = newPts;
		}
		newPts = ::av_rescale(
		    newPts - m_VideoStartPts, m_VideoPtsConv.num, m_VideoPtsConv.den);
	}
	ffmpeg::pts_t currentTimestamp = ::av_gettime_relative();
	ffmpeg::pts_t currentPts = m_VideoPts;
	if (currentPts == AV_NOPTS_VALUE) {
		currentPts = newPts;
	} else {
		currentPts += currentTimestamp - m_VideoTimestamp;
	}
	if (newPts == AV_NOPTS_VALUE) {
		newPts = currentPts;
	}
	ffmpeg::pts_t delay = newPts - currentPts - static_cast<int>(m_VideoJitter);
	if (delay < 0 || delay > 500000) {  // Delay range: [0, 500] ms
		delay = 0;
	}
	ffmpeg::pts_t avDiff = currentPts - getAudioClock();
	if (avDiff >= 50000) {  // Audio lags more than 50 ms behind
		delay += avDiff;
	} else {
		avDiff = 0;
	}
	if (delay > 0) {
		::av_usleep(static_cast<unsigned int>(delay));
		currentTimestamp = ::av_gettime_relative();
		m_VideoJitter += ((currentTimestamp - m_VideoTimestamp) -
		                     (newPts - m_VideoPts) - avDiff) /
		                 2.0;
	} else {
		m_VideoJitter = 0;
	}
	{
		std::unique_lock<std::shared_mutex> lock(m_VideoSyncMutex);
		m_VideoPts = newPts;
		m_VideoTimestamp = currentTimestamp;
	}
}

ffmpeg::pts_t player::SPlayer::getAudioClock() {
	assert(m_AudioDevice);
	std::shared_lock<std::shared_mutex> lock{m_AudioSyncMutex};
	if (m_AudioPts == AV_NOPTS_VALUE) {
		return AV_NOPTS_VALUE;
	}
	return m_AudioPts + ::av_gettime_relative() - m_AudioTimestamp;
}

void player::SPlayer::syncAudio(::AVFrame *frame) {
	assert(m_AudioDevice);
	ffmpeg::pts_t pts = frame->best_effort_timestamp;
	if (pts == AV_NOPTS_VALUE) {
		return;
	}
	if (m_AudioStartPts == AV_NOPTS_VALUE) {
		m_AudioStartPts = pts;
	}
	pts = ::av_rescale(
	          pts - m_AudioStartPts, m_AudioPtsConv.num, m_AudioPtsConv.den) -
	      ::av_rescale(::SDL_GetQueuedAudioSize(m_AudioDevice->getDeviceId()),
	          m_AudioQueueConv.num, m_AudioQueueConv.den);
	ffmpeg::pts_t currentClock = getAudioClock();
	ffmpeg::pts_t queueTime =
	    ::av_rescale(::SDL_GetQueuedAudioSize(m_AudioDevice->getDeviceId()),
	        m_AudioQueueConv.num, m_AudioQueueConv.den);
	ffmpeg::pts_t delay = pts - currentClock;
	if (queueTime > 50000 && delay < queueTime) {  // Target queue is 50ms
		delay = (queueTime - 50000) / 2;
	}
	delay += static_cast<int>(m_AudioJitter);
	if (delay < 0) {
		delay = 0;
	}
	ffmpeg::pts_t avDiff = currentClock - getVideoClock();
	if (avDiff >= 50000) {  // Video lags more than 50ms behind
		delay += avDiff;
	}
	if (delay > 0) {
		::av_usleep(static_cast<unsigned int>(delay));
		ffmpeg::pts_t actualDelay = getAudioClock() - currentClock;
		m_AudioJitter += (delay - actualDelay - m_AudioJitter) / 2;
	} else {
		m_AudioJitter = 0;
	}
	{
		std::unique_lock<std::shared_mutex> lock{m_AudioSyncMutex};
		m_AudioPts = pts;
		m_AudioTimestamp = ::av_gettime_relative();
	}
}
