// Copyright 2023 Viktor Ivanov

#include "JoshUpscale/obs/filter.h"

#include <JoshUpscale/core.h>

#include <cstddef>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace JoshUpscale {

namespace obs {

namespace {

inline ::AVPixelFormat convertFrameFormat(::video_format format) {
	switch (format) {
	case VIDEO_FORMAT_I444:
		return AV_PIX_FMT_YUV444P;
	case VIDEO_FORMAT_I420:
		return AV_PIX_FMT_YUV420P;
	case VIDEO_FORMAT_NV12:
		return AV_PIX_FMT_NV12;
	case VIDEO_FORMAT_YUY2:
		return AV_PIX_FMT_YUYV422;
	case VIDEO_FORMAT_UYVY:
		return AV_PIX_FMT_UYVY422;
	case VIDEO_FORMAT_YVYU:
		return AV_PIX_FMT_YVYU422;
	case VIDEO_FORMAT_RGBA:
		return AV_PIX_FMT_RGBA;
	case VIDEO_FORMAT_BGRA:
	case VIDEO_FORMAT_BGRX:
		return AV_PIX_FMT_BGRA;
	case VIDEO_FORMAT_Y800:
		return AV_PIX_FMT_GRAY8;
	case VIDEO_FORMAT_BGR3:
		return AV_PIX_FMT_BGR24;
	case VIDEO_FORMAT_I422:
		return AV_PIX_FMT_YUV422P;
	case VIDEO_FORMAT_I40A:
		return AV_PIX_FMT_YUVA420P;
	case VIDEO_FORMAT_I42A:
		return AV_PIX_FMT_YUVA422P;
	case VIDEO_FORMAT_YUVA:
		return AV_PIX_FMT_YUVA444P;
	default:
		return AV_PIX_FMT_NONE;
	}
}

}  // namespace

::obs_source_info *JoshUpscaleFilter::getSourceInfo() {
	static struct Data {
		::obs_source_info info = {};
		Data() {
#define CALLBACK_DEF(name) (Callback<&JoshUpscaleFilter::name>::getPtr())
			info.id = "joshupscale";
			info.type = OBS_SOURCE_TYPE_FILTER;
			info.output_flags = OBS_SOURCE_ASYNC_VIDEO;
			info.get_name = CALLBACK_DEF(getName);
			info.create = CALLBACK_DEF(create);
			info.destroy = CALLBACK_DEF(destroy);
			info.update = CALLBACK_DEF(update);
			info.filter_video = CALLBACK_DEF(filterVideo);
			info.get_properties2 = CALLBACK_DEF(getProperties);
			info.get_defaults2 = CALLBACK_DEF(getDefaults);
#undef CALLBACK_DEF
		}
	} data;
	return &data.info;
}

JoshUpscaleFilter::JoshUpscaleFilter(
    [[maybe_unused]] ::obs_data_t *settings, ::obs_source_t *source)
    : m_Source(source)
    , m_InputBuffer(core::INPUT_WIDTH * core::INPUT_HEIGHT * 3)
    , m_OutputFrame(core::OUTPUT_WIDTH, core::OUTPUT_HEIGHT) {
	m_WorkerThread = std::thread(&JoshUpscaleFilter::workerThread, this);
	update(settings);
}

JoshUpscaleFilter::~JoshUpscaleFilter() {
	{
		std::unique_lock<std::mutex> lock(m_Mutex);
		m_Terminated = true;
	}
	m_Condition.notify_all();
	if (m_SwsCtx != nullptr) {
		::sws_freeContext(m_SwsCtx);
	}
	// Ensure that injected frame is out
	::obs_source_t *parent = ::obs_filter_get_parent(m_Source);
	for (;;) {
		auto *frame = ::obs_source_get_frame(parent);
		::obs_source_release_frame(parent, frame);
		if (frame != m_OutputFrame) {
			break;
		}
		std::this_thread::yield();
	}
	if (m_WorkerThread.joinable()) {
		m_WorkerThread.join();
	}
}

const char *JoshUpscaleFilter::getName(
    [[maybe_unused]] void *typeData) noexcept {
	return "JoshUpscale";
}

void *JoshUpscaleFilter::create(
    ::obs_data_t *settings, ::obs_source_t *source) noexcept {
	try {
		return new JoshUpscaleFilter(settings, source);
	} catch (...) {
		return nullptr;
	}
}

void JoshUpscaleFilter::destroy(void *data) noexcept {
	blog(LOG_INFO, "[obs-joshupscale] Start shutdown");
	delete reinterpret_cast<JoshUpscaleFilter *>(data);
	blog(LOG_INFO, "[obs-joshupscale] Shutdown finished");
}

void JoshUpscaleFilter::update(::obs_data_t *settings) noexcept {
	{
		std::unique_lock lock(m_Mutex);
		m_CurrentModel =
		    static_cast<int>(::obs_data_get_int(settings, "model"));
	}
	m_Condition.notify_all();
}

void JoshUpscaleFilter::copyFrame(::obs_source_frame *frame) {
	int srcW = static_cast<int>(frame->width);
	int srcH = static_cast<int>(frame->height);
	::AVPixelFormat srcFormat = convertFrameFormat(frame->format);
	int dstW = static_cast<int>(core::INPUT_WIDTH);
	int dstH = static_cast<int>(core::INPUT_HEIGHT);
	::AVPixelFormat dstFormat = AV_PIX_FMT_BGR24;
	m_SwsCtx = ::sws_getCachedContext(m_SwsCtx, srcW, srcH, srcFormat, dstW,
	    dstH, dstFormat, SWS_POINT, nullptr, nullptr, nullptr);
	if (m_SwsCtx == nullptr) {
		throw std::runtime_error("SwsCtx failure");
	}
	if (::format_is_yuv(frame->format)) {
		float rangeCoeff = frame->full_range ? (255.0F / 224.0F) : 1.0F;
		int coeff[4] = {
		    static_cast<int>(65536 * frame->color_matrix[2] * rangeCoeff),
		    static_cast<int>(65536 * frame->color_matrix[9] * rangeCoeff),
		    static_cast<int>(65536 * -frame->color_matrix[5] * rangeCoeff),
		    static_cast<int>(65536 * -frame->color_matrix[6] * rangeCoeff),
		};
		if (::sws_setColorspaceDetails(m_SwsCtx, coeff,
		        static_cast<int>(frame->full_range),
		        ::sws_getCoefficients(SWS_CS_DEFAULT), 1, 0, 1 << 16,
		        1 << 16) < 0) {
			throw std::runtime_error("SwsCtx failure");
		}
		m_OutputFrame->full_range = true;
	} else {
		m_OutputFrame->full_range = frame->full_range;
	}
	int inStrides[4] = {};
	for (std::size_t i = 0; i < 4; ++i) {
		inStrides[i] = static_cast<int>(frame->linesize[i]);
	}
	std::uint8_t *outBuffers[4] = {
	    reinterpret_cast<std::uint8_t *>(m_InputBuffer.get())};
	int outStrides[4] = {core::INPUT_WIDTH * 3};
	::sws_scale(
	    m_SwsCtx, frame->data, inStrides, 0, srcH, outBuffers, outStrides);
	m_OutputFrame->timestamp = frame->timestamp;
	m_OutputFrame->flip = frame->flip;
	::os_atomic_inc_long(&m_OutputFrame->refs);
}

void JoshUpscaleFilter::workerThread() noexcept {
	try {
		int newModel = -1;
		for (;;) {
			{
				std::unique_lock lock(m_Mutex);
				m_LoadedModel = newModel;
				m_Ready = true;
				::obs_source_update_properties(m_Source);
				m_Condition.wait(lock, [this] {
					return m_Terminated || m_CurrentModel != m_LoadedModel;
				});
				if (m_Terminated) {
					break;
				}
				newModel = m_CurrentModel;
				m_Ready = false;
				::obs_source_update_properties(m_Source);
			}
			static const char *models[4] = {
			    "model_fast.yaml",
			    "model.yaml",
			    "model_smooth.yaml",
			    "model_adapt.yaml",
			};
			blog(LOG_INFO, "[obs-joshupscale] Start building engine for %s",
			    models[newModel]);
			auto modelFile = OBSPtr(obs_module_file(models[newModel]));
			m_Runtime.reset(core::createRuntime(0, modelFile.get()));
			blog(LOG_INFO, "[obs-joshupscale] Engine build successful");
		}
	} catch (std::exception &e) {
		blog(LOG_ERROR, "[obs-joshupscale] Worker failed: %s", e.what());
		m_Error = true;
		::obs_source_update_properties(m_Source);
	}
}

::obs_source_frame *JoshUpscaleFilter::filterVideo(
    ::obs_source_frame *frame) noexcept {
	std::unique_lock lock(m_Mutex);
	if (!m_Ready) {
		return frame;
	}
	::obs_source_t *parent = ::obs_filter_get_parent(m_Source);
	try {
		copyFrame(frame);
		core::Image inputImage = {
		    .ptr = m_InputBuffer.get(),
		    .stride = static_cast<std::ptrdiff_t>(core::INPUT_WIDTH * 3),
		    .width = core::INPUT_WIDTH,
		    .height = core::INPUT_HEIGHT,
		};
		core::Image outputImage = {
		    .ptr = m_OutputFrame->data[0],
		    .stride = static_cast<std::ptrdiff_t>(m_OutputFrame->linesize[0]),
		    .width = core::OUTPUT_WIDTH,
		    .height = core::OUTPUT_HEIGHT,
		};
		m_Runtime->processImage(inputImage, outputImage);
	} catch (...) {
		return frame;
	}
	::obs_source_release_frame(parent, frame);
	return m_OutputFrame;
}

::obs_properties_t *JoshUpscaleFilter::getProperties(
    void *data, void *typeData) noexcept {
	::obs_properties_t *props = ::obs_properties_create();
	if (data != nullptr) {
		reinterpret_cast<JoshUpscaleFilter *>(data)->addProperties(
		    props, typeData);
	}
	return props;
}

void JoshUpscaleFilter::addProperties(
    ::obs_properties_t *props, [[maybe_unused]] void *typeData) noexcept {
	bool error = m_Error;
	bool ready = m_Ready;
	::obs_property_t *backendProp = ::obs_properties_add_list(props, "backend",
	    "Inference backend", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	::obs_property_list_add_int(backendProp, "TensorRT", 0);
	::obs_property_set_enabled(backendProp, false);
	::obs_property_t *modelProp = ::obs_properties_add_list(props, "model",
	    "Model variant", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	::obs_property_list_add_int(modelProp, "0: Fast", 0);
	::obs_property_list_add_int(modelProp, "1: Default", 1);
	::obs_property_list_add_int(modelProp, "2: Smooth", 2);
	::obs_property_list_add_int(modelProp, "3: Adaptive", 3);
	::obs_property_t *quantizationProp =
	    ::obs_properties_add_list(props, "quantization", "Quantization",
	        OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	::obs_property_list_add_int(quantizationProp, "0: None", 0);
	::obs_property_list_add_int(quantizationProp, "1: FP16", 1);
	::obs_property_list_add_int(quantizationProp, "2: INT8", 2);
	if (error || !ready) {
		::obs_property_set_enabled(modelProp, false);
		::obs_property_set_enabled(quantizationProp, false);
		::obs_property_t *statusProp =
		    ::obs_properties_add_text(props, "status", nullptr, OBS_TEXT_INFO);
		if (error) {
			::obs_property_set_description(statusProp,
			    "There was an error in the worker thread. Please check the "
			    "logs "
			    "for further details.");
			::obs_property_text_set_info_type(statusProp, OBS_TEXT_INFO_ERROR);
		} else if (!ready) {
			::obs_property_set_description(statusProp,
			    "Building the engine. It can take a few minutes. Please "
			    "wait...");
			::obs_property_text_set_info_type(
			    statusProp, OBS_TEXT_INFO_WARNING);
		}
	}
}

void JoshUpscaleFilter::getDefaults(
    [[maybe_unused]] void *typeData, ::obs_data_t *settings) noexcept {
	::obs_data_set_default_int(settings, "model", 3);
	::obs_data_set_default_int(settings, "quantization", 1);
}

}  // namespace obs

}  // namespace JoshUpscale
