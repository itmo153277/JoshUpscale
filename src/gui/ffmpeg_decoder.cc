// Copyright 2021 Ivanov Viktor

#include "ffmpeg_decoder.h"

#include <thread>
#include <utility>

ffmpeg::SDecoder::SDecoder(const char *source, ffmpeg::DXVA dxva)
    : m_FormatCtx{ffmpeg::openDshowSource(source)}
    , m_VideoStreamInfo{ffmpeg::openVideoStream(m_FormatCtx.get(), dxva)}
    , m_AudioStreamInfo{ffmpeg::openAudioStream(m_FormatCtx.get())}
    , m_VideoQueue{ffmpeg::MAX_VIDEO_QUEUE}
    , m_AudioQueue{ffmpeg::MAX_AUDIO_QUEUE}
    , m_Terminated{false} {
}

ffmpeg::SDecoder::~SDecoder() {
	terminate();
}

void ffmpeg::SDecoder::terminate() {
	m_Terminated = true;
	m_VideoQueue.terminate();
	m_AudioQueue.terminate();
}

void ffmpeg::SDecoder::captureLoop() {
	try {
		smart::AVPacket packet;
		for (; !m_Terminated;) {
			std::this_thread::yield();
			ffmpeg::callOrThrow(
			    ::av_read_frame, m_FormatCtx.get(), packet.get());
			if (packet->stream_index == m_VideoStreamInfo.streamIndex) {
				m_VideoQueue.produce(std::move(packet));
			} else if (packet->stream_index == m_AudioStreamInfo.streamIndex &&
			           m_AudioStreamInfo.streamIndex >= 0) {
				m_AudioQueue.produce(std::move(packet));
			} else {
				::av_packet_unref(packet.get());
			}
		}
	} catch (const ffmpeg::AVException &e) {
		if (e.averror != AVERROR_EOF) {
			throw;
		}
	}
	m_VideoQueue.flush();
	m_AudioQueue.flush();
}

void ffmpeg::SDecoder::videoDecoderLoop(DecoderCallback cb) {
	try {
		ffmpeg::pts_t minPts = AV_NOPTS_VALUE;
		smart::AVFrame frame;
		smart::AVFrame swFrame;
		::AVCodecContext *pCodecCtx = m_VideoStreamInfo.codecCtx.get();
		::AVPixelFormat hwPixelFormat = m_VideoStreamInfo.hwPixelFormat;
		for (; !m_Terminated;) {
			std::this_thread::yield();
			smart::AVPacket packet = m_VideoQueue.consume();
			if (packet->pts != AV_NOPTS_VALUE && minPts != AV_NOPTS_VALUE &&
			    (packet->pts + packet->duration) < minPts) {
				pCodecCtx->skip_frame = AVDISCARD_NONREF;
			} else {
				pCodecCtx->skip_frame = AVDISCARD_NONE;
			}
			ffmpeg::callOrThrow(::avcodec_send_packet, pCodecCtx, packet.get());
			for (; !m_Terminated;) {
				std::this_thread::yield();
				int ret = ::avcodec_receive_frame(pCodecCtx, frame.get());
				if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
					break;
				}
				if (ret < 0) {
					throw ffmpeg::AVException(ret);
				}
				if (frame->best_effort_timestamp != AV_NOPTS_VALUE &&
				    minPts != AV_NOPTS_VALUE &&
				    frame->best_effort_timestamp < minPts) {
					continue;
				}
				if (frame->format == hwPixelFormat) {
					ffmpeg::callOrThrow(::av_hwframe_transfer_data,
					    swFrame.get(), frame.get(), 0);
					ffmpeg::callOrThrow(
					    ::av_frame_copy_props, swFrame.get(), frame.get());
					minPts = cb(swFrame.get());
				} else {
					minPts = cb(frame.get());
				}
			}
		}
	} catch (const ffmpeg::AVException &e) {
		if (e.averror != AVERROR_EOF) {
			throw;
		}
	}
}

void ffmpeg::SDecoder::audioDecoderLoop(DecoderCallback cb) {
	try {
		ffmpeg::pts_t minPts = AV_NOPTS_VALUE;
		smart::AVFrame frame;
		::AVCodecContext *pCodecCtx = m_AudioStreamInfo.codecCtx.get();
		for (; !m_Terminated;) {
			std::this_thread::yield();
			smart::AVPacket packet = m_AudioQueue.consume();
			if (packet->pts != AV_NOPTS_VALUE && minPts != AV_NOPTS_VALUE &&
			    (packet->pts + packet->duration) < minPts) {
				pCodecCtx->skip_frame = AVDISCARD_NONREF;
			} else {
				pCodecCtx->skip_frame = AVDISCARD_NONE;
			}
			ffmpeg::callOrThrow(::avcodec_send_packet, pCodecCtx, packet.get());
			for (; !m_Terminated;) {
				std::this_thread::yield();
				int ret = ::avcodec_receive_frame(pCodecCtx, frame.get());
				if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
					break;
				}
				if (ret < 0) {
					throw ffmpeg::AVException(ret);
				}
				if (frame->best_effort_timestamp != AV_NOPTS_VALUE &&
				    minPts != AV_NOPTS_VALUE &&
				    frame->best_effort_timestamp < minPts) {
					continue;
				}
				minPts = cb(frame.get());
			}
		}
	} catch (const ffmpeg::AVException &e) {
		if (e.averror != AVERROR_EOF) {
			throw;
		}
	}
}

smart::AVFrame ffmpeg::SDecoder::audioDecodeFrameOrThrow(pts_t minPts) {
	smart::AVFrame frame;
	::AVCodecContext *pCodecCtx = m_AudioStreamInfo.codecCtx.get();
	for (; !m_Terminated;) {
		for (; !m_Terminated;) {
			int ret = ::avcodec_receive_frame(pCodecCtx, frame.get());
			if (ret == AVERROR(EAGAIN)) {
				break;
			}
			if (ret < 0) {
				throw ffmpeg::AVException(ret);
			}
			if (frame->best_effort_timestamp != AV_NOPTS_VALUE &&
			    minPts != AV_NOPTS_VALUE &&
			    frame->best_effort_timestamp < minPts) {
				continue;
			}
			return frame;
		}
		smart::AVPacket packet = m_AudioQueue.consumeOrThrow();
		if (packet->pts != AV_NOPTS_VALUE && minPts != AV_NOPTS_VALUE &&
		    (packet->pts + packet->duration) < minPts) {
			pCodecCtx->skip_frame = AVDISCARD_NONREF;
		} else {
			pCodecCtx->skip_frame = AVDISCARD_NONE;
		}
		ffmpeg::callOrThrow(::avcodec_send_packet, pCodecCtx, packet.get());
	}
	throw AVException(AVERROR_EOF);
}
