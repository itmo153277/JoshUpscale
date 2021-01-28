#include "ffmpeg_wrappers.h"

void ffmpeg::init() {
	::avdevice_register_all();
}

smart::AVFormatContext ffmpeg::openDshowSource(const char *source) {
	smart::AVFormatContext formatCtx;
	::AVInputFormat *pInputFormat = ::av_find_input_format("dshow");

	smart::AVDictionary opts;
	ffmpeg::callOrThrow(
	    ::av_dict_set_int, &opts.get(), "audio_buffer_size", 10, 0);
	ffmpeg::callOrThrow(::avformat_open_input, &formatCtx.get(), source,
	    pInputFormat, &opts.get());
	ffmpeg::callOrThrow(::avformat_find_stream_info, formatCtx.get(), nullptr);
	return formatCtx;
}

ffmpeg::SVideoStreamInfo ffmpeg::openVideoStream(
    ::AVFormatContext *pFormatCtx, ffmpeg::DXVA dxva) {
	ffmpeg::SVideoStreamInfo streamInfo;
	::AVCodec *pCodec = nullptr;
	streamInfo.streamIndex = ffmpeg::callOrThrow(::av_find_best_stream,
	    pFormatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, &pCodec, 0);
	streamInfo.hwPixelFormat = AV_PIX_FMT_NONE;
	if (dxva != ffmpeg::DXVA::OFF) {
		for (int i = 0;; ++i) {
			const ::AVCodecHWConfig *pHwConfig =
			    ::avcodec_get_hw_config(pCodec, i);
			if (pHwConfig == nullptr) {
				break;
			}
			if (pHwConfig->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
			    pHwConfig->device_type == AV_HWDEVICE_TYPE_DXVA2) {
				streamInfo.hwPixelFormat = pHwConfig->pix_fmt;
				break;
			}
		}
	}
	if (streamInfo.hwPixelFormat == AV_PIX_FMT_NONE &&
	    dxva == ffmpeg::DXVA::FORCED) {
		throw ffmpeg::AVException(AVERROR_DECODER_NOT_FOUND);
	}
	streamInfo.codecCtx.allocate(pCodec);
	ffmpeg::callOrThrow(::avcodec_parameters_to_context,
	    streamInfo.codecCtx.get(),
	    pFormatCtx->streams[streamInfo.streamIndex]->codecpar);
	if (streamInfo.hwPixelFormat != AV_PIX_FMT_NONE) {
		if (dxva == ffmpeg::DXVA::FORCED) {
			streamInfo.m_InternalData =
			    std::make_unique<ffmpeg::SVideoStreamInfo::SInternalData>(
			        streamInfo.hwPixelFormat);
			streamInfo.codecCtx->opaque = streamInfo.m_InternalData.get();
			streamInfo.codecCtx->get_format = [](::AVCodecContext *ctx,
			                                      const ::AVPixelFormat *fmts) {
				const ::AVPixelFormat *fmt = fmts;
				auto pStreamInfo = static_cast<
				    const ffmpeg::SVideoStreamInfo::SInternalData *>(
				    ctx->opaque);
				for (; *fmt != AV_PIX_FMT_NONE &&
				       *fmt != pStreamInfo->hwPixelFormat;
				     ++fmt)
					;
				return *fmt;
			};
		}
		ffmpeg::callOrThrow(::av_hwdevice_ctx_create,
		    &streamInfo.codecCtx->hw_device_ctx, AV_HWDEVICE_TYPE_DXVA2,
		    nullptr, nullptr, 0);
	}
	ffmpeg::callOrThrow(
	    ::avcodec_open2, streamInfo.codecCtx.get(), pCodec, nullptr);
	return streamInfo;
}

ffmpeg::SStreamInfo ffmpeg::openAudioStream(::AVFormatContext *pFormatCtx) {
	ffmpeg::SStreamInfo streamInfo;
	::AVCodec *pCodec = nullptr;
	streamInfo.streamIndex = ::av_find_best_stream(
	    pFormatCtx, AVMEDIA_TYPE_AUDIO, -1, -1, &pCodec, 0);
	if (pCodec != nullptr) {
		streamInfo.codecCtx.allocate(pCodec);
		ffmpeg::callOrThrow(::avcodec_parameters_to_context,
		    streamInfo.codecCtx.get(),
		    pFormatCtx->streams[streamInfo.streamIndex]->codecpar);
		ffmpeg::callOrThrow(
		    ::avcodec_open2, streamInfo.codecCtx.get(), pCodec, nullptr);
	}
	return streamInfo;
}

ffmpeg::SPacketQueue::~SPacketQueue() {
	terminate();
}

smart::AVPacket ffmpeg::SPacketQueue::consume() {
	smart::AVPacket packet;
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		m_cv.wait(lock, [this] {
			return m_PacketQueue.size() > 0 || m_Terminated || m_Flush;
		});
		if (m_Terminated || (m_Flush && m_PacketQueue.size() == 0)) {
			throw ffmpeg::AVException(AVERROR_EOF);
		}
		::av_packet_move_ref(packet.get(), m_PacketQueue.front().get());
		m_PacketQueue.pop();
		m_Size -= packet.get()->size;
	}
	m_cv.notify_all();
	return packet;
}

smart::AVPacket ffmpeg::SPacketQueue::consumeOrThrow() {
	smart::AVPacket packet;
	{
		std::lock_guard<std::mutex> lock(m_Mutex);
		if (m_Terminated || (m_Flush && m_PacketQueue.size() == 0)) {
			throw ffmpeg::AVException(AVERROR_EOF);
		}
		if (m_PacketQueue.size() == 0) {
			throw ffmpeg::AVException(AVERROR(EAGAIN));
		}
		::av_packet_move_ref(packet.get(), m_PacketQueue.front().get());
		m_PacketQueue.pop();
		m_Size -= packet.get()->size;
	}
	m_cv.notify_all();
	return packet;
}

void ffmpeg::SPacketQueue::produce(smart::AVPacket &&packet) {
	ffmpeg::callOrThrow(::av_packet_make_refcounted, packet.get());
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		m_cv.wait(lock, [this] { return m_Size < m_MaxQueue || m_Terminated; });
		if (m_Terminated) {
			throw ffmpeg::AVException(AVERROR_EOF);
		}
		m_PacketQueue.emplace();
		m_Size += packet.get()->size;
		::av_packet_move_ref(m_PacketQueue.back().get(), packet.get());
	}
	m_cv.notify_all();
}

void ffmpeg::SPacketQueue::terminate() {
	{
		std::lock_guard<std::mutex> lock(m_Mutex);
		m_Terminated = true;
	}
	m_cv.notify_all();
}

void ffmpeg::SPacketQueue::flush() {
	{
		std::lock_guard<std::mutex> lock(m_Mutex);
		m_Flush = true;
	}
	m_cv.notify_all();
}
