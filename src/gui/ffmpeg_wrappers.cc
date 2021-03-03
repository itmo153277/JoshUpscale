// Copyright 2021 Ivanov Viktor

#include "ffmpeg_wrappers.h"

void ffmpeg::init() {
	::avdevice_register_all();
}

smart::AVFormatContext ffmpeg::openSource(
    const char *source, const char *sourceType) {
	smart::AVFormatContext formatCtx;
	::AVInputFormat *format = nullptr;
	if (sourceType != nullptr) {
		format = ::av_find_input_format(sourceType);
	}
	ffmpeg::callOrThrow(
	    ::avformat_open_input, &formatCtx.get(), source, format, nullptr);
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
				     ++fmt) {
				}
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

ffmpeg::SGraphInfo ffmpeg::createVideoGraph(const SVideoStreamInfo *streamInfo,
    ::AVFormatContext *pFormatCtx, ::AVFrame *frame, ::AVPixelFormat outFormat,
    const char *filterStr) {
	::AVStream *stream = pFormatCtx->streams[streamInfo->streamIndex];
	smart::AVFilterGraph graph;
	const AVFilter *buffersrc = ::avfilter_get_by_name("buffer");
	const AVFilter *buffersink = ::avfilter_get_by_name("buffersink");

	::AVFilterContext *srcCtx =
	    ::avfilter_graph_alloc_filter(graph.get(), buffersrc, "in");
	smart::AVPointer srcParamPtr;
	::AVBufferSrcParameters *srcParam = ::av_buffersrc_parameters_alloc();
	if (!srcParam) {
		throw std::bad_alloc();
	}
	srcParamPtr.reset(srcParam);
	srcParam->format = frame->format;
	srcParam->time_base = stream->time_base;
	srcParam->width = frame->width;
	srcParam->height = frame->height;
	srcParam->sample_aspect_ratio = streamInfo->codecCtx->sample_aspect_ratio;
	srcParam->frame_rate = ::av_guess_frame_rate(pFormatCtx, stream, frame);
	srcParam->hw_frames_ctx = streamInfo->codecCtx->hw_frames_ctx;
	ffmpeg::callOrThrow(::av_buffersrc_parameters_set, srcCtx, srcParam);
	ffmpeg::callOrThrow(::avfilter_init_str, srcCtx, nullptr);
	::AVFilterContext *sinkCtx = nullptr;
	ffmpeg::callOrThrow(::avfilter_graph_create_filter, &sinkCtx, buffersink,
	    "out", nullptr, nullptr, graph.get());
	::AVPixelFormat pixFmts[2] = {outFormat, AV_PIX_FMT_NONE};
	ffmpeg::AVError ret = av_opt_set_int_list(
	    sinkCtx, "pix_fmts", pixFmts, AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN);
	if (ret < 0) {
		throw ffmpeg::AVException(ret);
	}
	smart::AVFilterInOut inputs;
	smart::AVFilterInOut outputs;
	outputs->name = ::av_strdup("in");
	outputs->filter_ctx = srcCtx;
	outputs->pad_idx = 0;
	outputs->next = nullptr;
	inputs->name = ::av_strdup("out");
	inputs->filter_ctx = sinkCtx;
	inputs->pad_idx = 0;
	inputs->next = nullptr;
	ffmpeg::callOrThrow(::avfilter_graph_parse_ptr, graph.get(), filterStr,
	    &inputs.get(), &outputs.get(), nullptr);
	ffmpeg::callOrThrow(::avfilter_graph_config, graph.get(), nullptr);
	return {std::move(graph), srcCtx, sinkCtx};
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
