// Copyright 2025 Viktor Ivanov

#include "JoshUpscale/obs/filter.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "JoshUpscale/core.h"
#include "JoshUpscale/obs/logging.h"
#include "JoshUpscale/obs/utils.h"

namespace JoshUpscale {

namespace obs {

::obs_source_info *JoshUpscaleFilter::getSourceInfo() {
	struct Eval {
		static consteval ::obs_source_info createSourceInfo() {
			::obs_source_info info = {};
#define CALLBACK_DEF(name) (Callback<&JoshUpscaleFilter::name>::getPtr())
			info.id = "joshupscale";
			info.type = OBS_SOURCE_TYPE_FILTER;
			info.output_flags = OBS_SOURCE_VIDEO;
			info.get_name = CALLBACK_DEF(getName);
			info.create = CALLBACK_DEF(create);
			info.destroy = CALLBACK_DEF(destroy);
			info.update = CALLBACK_DEF(update);
			info.get_defaults2 = CALLBACK_DEF(getDefaults);
			info.get_properties2 = CALLBACK_DEF(getProperties);
			info.video_render = CALLBACK_DEF(render);
			info.get_width = CALLBACK_DEF(getWidth);
			info.get_height = CALLBACK_DEF(getHeight);
			info.video_tick = CALLBACK_DEF(videoTick);
#undef CALLBACK_DEF
			return info;
		}
	};

	static ::obs_source_info info = Eval::createSourceInfo();
	return &info;
}

JoshUpscaleFilter::JoshUpscaleFilter(
    ::obs_data_t *settings, ::obs_source_t *source)
    : m_Source{source} {
	try {
		auto maskFile = OBSPtr(obs_module_file("mask.png"));
		::gs_image_file_init(&m_MaskImage, maskFile.get());
		auto blendEffectFile = OBSPtr(obs_module_file("effects/blend.effect"));
		{
			::obs_enter_graphics();
			defer {
				::obs_leave_graphics();
			};
			auto deviceType = ::gs_get_device_type();
			if (deviceType != GS_DEVICE_DIRECT3D_11 &&
			    deviceType != GS_DEVICE_OPENGL) {
				throw std::runtime_error("Unsupported renderer");
			}
#ifdef _WIN32
			if (deviceType == GS_DEVICE_DIRECT3D_11) {
				m_Device = core::getD3D11DeviceIndex(
				    reinterpret_cast<ID3D11Device *>(::gs_get_device_obj()));
			}
#endif
			if (deviceType == GS_DEVICE_OPENGL) {
				m_Device = core::getGLDeviceIndex();
			}
			if (m_Device < 0) {
				throw std::runtime_error("Unsupported render device");
			}
			m_RenderTarget = ::gs_texrender_create(GS_BGRX, GS_ZS_NONE);
			m_RenderInput = ::gs_texrender_create(GS_BGRX_UNORM, GS_ZS_NONE);
			m_ScaleEffect = ::obs_get_base_effect(OBS_EFFECT_BILINEAR_LOWRES);
			if (m_ScaleEffect != nullptr) {
				m_ScaleImgParam =
				    ::gs_effect_get_param_by_name(m_ScaleEffect, "image");
			}
			m_OutputEffect = ::obs_get_base_effect(OBS_EFFECT_DEFAULT);
			if (m_OutputEffect != nullptr) {
				m_OutputImgParam =
				    ::gs_effect_get_param_by_name(m_OutputEffect, "image");
			}
			m_BlendEffect =
			    ::gs_effect_create_from_file(blendEffectFile.get(), nullptr);
			if (m_BlendEffect != nullptr) {
				m_BlendImgParam =
				    ::gs_effect_get_param_by_name(m_BlendEffect, "image");
				m_BlendMaskParam =
				    ::gs_effect_get_param_by_name(m_BlendEffect, "mask");
			}
			::gs_image_file_init_texture(&m_MaskImage);
		}
		if (m_RenderTarget == nullptr || m_RenderInput == nullptr ||
		    m_ScaleEffect == nullptr || m_ScaleImgParam == nullptr ||
		    m_OutputEffect == nullptr || m_OutputImgParam == nullptr ||
		    m_BlendEffect == nullptr || m_BlendImgParam == nullptr ||
		    m_BlendMaskParam == nullptr || m_MaskImage.texture == nullptr) {
			throw std::runtime_error("Initialization failed");
		}
		update(settings);
	} catch (...) {
		cleanup();
		throw;
	}
}

JoshUpscaleFilter::~JoshUpscaleFilter() {
	cleanup();
}

const char *JoshUpscaleFilter::getName(
    [[maybe_unused]] void *typeData) noexcept {
	return ::obs_module_text("Name");
}

void *JoshUpscaleFilter::create(
    ::obs_data_t *settings, ::obs_source_t *source) noexcept {
	try {
		return new JoshUpscaleFilter(settings, source);
	} catch (...) {
		logException();
		return nullptr;
	}
}

void JoshUpscaleFilter::destroy() noexcept {
	delete this;
}

void JoshUpscaleFilter::update(::obs_data_t *settings) noexcept {
	std::int64_t preset = ::obs_data_get_int(settings, "preset");
	std::int64_t resolution = ::obs_data_get_int(settings, "resolution");
	m_RenderMaskedTarget = resolution == 1;
	m_LimitFps = ::obs_data_get_bool(settings, "limit_fps");
	static const char *models[4] = {
	    "model_psp.trt",
	    "model_psp_fast.trt",
	    "model_ps2.trt",
	    "model_ps2_fast.trt",
	};
	int model = static_cast<int>((resolution * 2) + preset);
	assert(model >= 0 && model < 4);
	if (model != m_Model) {
		m_FrameProcessed = false;
		m_Runtime.reset();
		initModel(models[model]);
		m_Model = model;
	}
}

void JoshUpscaleFilter::getDefaults(
    [[maybe_unused]] void *typeData, ::obs_data_t *settings) noexcept {
	::obs_data_set_default_int(settings, "preset", 0);
	::obs_data_set_default_int(settings, "resolution", 0);
	::obs_data_set_default_bool(settings, "limit_fps", true);
}

::obs_properties_t *JoshUpscaleFilter::getProperties(
    [[maybe_unused]] void *data, [[maybe_unused]] void *typeData) noexcept {
	::obs_properties_t *props = ::obs_properties_create();
	::obs_property_t *presetProp = ::obs_properties_add_list(props, "preset",
	    ::obs_module_text("Preset"), OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	::obs_property_list_add_int(
	    presetProp, ::obs_module_text("PresetQuality"), 0);
	::obs_property_list_add_int(
	    presetProp, ::obs_module_text("PresetPerformance"), 1);
	::obs_property_t *resolutionProp = ::obs_properties_add_list(props,
	    "resolution", ::obs_module_text("Resolution"), OBS_COMBO_TYPE_LIST,
	    OBS_COMBO_FORMAT_INT);
	::obs_property_list_add_int(
	    resolutionProp, ::obs_module_text("ResolutionPSP"), 0);
	::obs_property_list_add_int(
	    resolutionProp, ::obs_module_text("ResolutionPS2"), 1);
	::obs_properties_add_bool(
	    props, "limit_fps", ::obs_module_text("LimitFps"));
	return props;
}

void JoshUpscaleFilter::render(
    [[maybe_unused]] ::gs_effect_t *effect) noexcept {
	::obs_source_t *target = ::obs_filter_get_target(m_Source);
	::obs_source_t *parent = ::obs_filter_get_parent(m_Source);
	if (target == nullptr || parent == nullptr) {
		return;
	}
	std::uint32_t targetWidth = ::obs_source_get_base_width(target);
	std::uint32_t targetHeight = ::obs_source_get_base_height(target);
	if (m_Runtime == nullptr || targetWidth == 0 || targetHeight == 0) {
		::obs_source_skip_video_filter(m_Source);
		return;
	}
	if (targetWidth != m_LastWidth || targetHeight != m_LastHeight) {
		m_LastWidth = targetWidth;
		m_LastHeight = targetHeight;
		m_InputImage.reset();
		m_FrameProcessed = false;
	}
	if (!m_FrameProcessed) {
		if (!processFrame()) {
			::obs_source_skip_video_filter(m_Source);
			return;
		}
		m_FrameProcessed = true;
		m_FrameDuration = 0;
	}
	::gs_blend_state_push();
	::gs_reset_blend_state();
	::gs_effect_set_texture(m_OutputImgParam, m_OutputTexture);
	while (::gs_effect_loop(m_OutputEffect, "Draw")) {
		::gs_draw_sprite(m_OutputTexture, 0, 0, 0);
	}
	if (m_RenderMaskedTarget) {
		renderMaskedTarget();
	}
	::gs_blend_state_pop();
}

void JoshUpscaleFilter::videoTick(float seconds) noexcept {
	m_FrameDuration += seconds;
	if (m_FrameDuration > 0.03F || !m_LimitFps) {
		m_FrameProcessed = false;
	}
}

std::uint32_t JoshUpscaleFilter::getWidth() noexcept {
	if (m_Runtime == nullptr) {
		return ::obs_source_get_base_width(::obs_filter_get_target(m_Source));
	}
	return static_cast<std::uint32_t>(m_Runtime->getOutputWidth());
}

std::uint32_t JoshUpscaleFilter::getHeight() noexcept {
	if (m_Runtime == nullptr) {
		return ::obs_source_get_base_height(::obs_filter_get_target(m_Source));
	}
	return static_cast<std::uint32_t>(m_Runtime->getOutputHeight());
}

void JoshUpscaleFilter::createInputImage(::gs_texture_t *texture) {
	auto deviceType = ::gs_get_device_type();
	if (deviceType == GS_DEVICE_OPENGL) {
		auto glTexture =
		    *reinterpret_cast<std::uint32_t *>(::gs_texture_get_obj(texture));
		m_InputImage.reset(core::getGLImage(
		    glTexture, core::GraphicsResourceImageType::INPUT));
	}
#ifdef _WIN32
	if (deviceType == GS_DEVICE_DIRECT3D_11) {
		struct ID3D11Texture2D *d3d11Texture =
		    reinterpret_cast<ID3D11Texture2D *>(::gs_texture_get_obj(texture));
		m_InputImage.reset(core::getD3D11Image(
		    d3d11Texture, core::GraphicsResourceImageType::INPUT));
	}
#endif
	assert(m_InputImage);
}

void JoshUpscaleFilter::createOutputImage() {
	auto deviceType = ::gs_get_device_type();
	if (deviceType == GS_DEVICE_OPENGL) {
		auto glTexture = *reinterpret_cast<std::uint32_t *>(
		    ::gs_texture_get_obj(m_OutputTexture));
		m_OutputImage.reset(core::getGLImage(
		    glTexture, core::GraphicsResourceImageType::OUTPUT));
	}
#ifdef _WIN32
	if (deviceType == GS_DEVICE_DIRECT3D_11) {
		struct ID3D11Texture2D *d3d11Texture =
		    reinterpret_cast<ID3D11Texture2D *>(
		        ::gs_texture_get_obj(m_OutputTexture));
		m_OutputImage.reset(core::getD3D11Image(
		    d3d11Texture, core::GraphicsResourceImageType::OUTPUT));
	}
#endif
	assert(m_OutputImage);
}

void JoshUpscaleFilter::initModel(const char *model) noexcept {
	auto modelFile = OBSPtr(obs_module_file(model));
	try {
		if (modelFile == nullptr) {
			throw std::runtime_error(std::string("Model not found: ") + model);
		}
		::obs_enter_graphics();
		defer {
			::obs_leave_graphics();
		};
		m_Runtime.reset(core::createRuntime(m_Device, modelFile.get()));
		m_InputImage.reset();
		m_OutputImage.reset();
		if (m_TargetTexture != nullptr) {
			::gs_texture_destroy(m_TargetTexture);
		}
		m_TargetTexture = ::gs_texture_create(
		    static_cast<std::uint32_t>(m_Runtime->getInputWidth()),
		    static_cast<std::uint32_t>(m_Runtime->getInputHeight()),
		    GS_BGRX_UNORM, 1, nullptr, 0);
		if (m_OutputTexture != nullptr) {
			::gs_texture_destroy(m_OutputTexture);
		}
		m_OutputTexture = ::gs_texture_create(
		    static_cast<std::uint32_t>(m_Runtime->getOutputWidth()),
		    static_cast<std::uint32_t>(m_Runtime->getOutputHeight()),
		    GS_BGRX_UNORM, 1, nullptr, 0);
		if (m_TargetTexture == nullptr || m_OutputTexture == nullptr) {
			throw std::runtime_error("Initialization failed");
		}
		createOutputImage();
		log(core::LogLevel::INFO, "Successfully loaded model: %s", model);
	} catch (...) {
		logException();
		m_Runtime.reset();
	}
}

bool JoshUpscaleFilter::processFrame() noexcept {
	::obs_source_t *target = ::obs_filter_get_target(m_Source);
	::obs_source_t *parent = ::obs_filter_get_parent(m_Source);
	std::uint32_t targetWidth = ::obs_source_get_base_width(target);
	std::uint32_t targetHeight = ::obs_source_get_base_height(target);
	std::uint32_t targetFlags = ::obs_source_get_output_flags(target);
	bool custom_draw = (targetFlags & OBS_SOURCE_CUSTOM_DRAW) != 0;
	bool async = (targetFlags & OBS_SOURCE_ASYNC) != 0;
	{
		::gs_texrender_reset(m_RenderTarget);
		::gs_blend_state_push();
		defer {
			::gs_blend_state_pop();
		};
		::gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
		if (!::gs_texrender_begin(m_RenderTarget, targetWidth, targetHeight)) {
			return false;
		}
		defer {
			::gs_texrender_end(m_RenderTarget);
		};
		::vec4 clear_color;
		::vec4_zero(&clear_color);
		::gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0F, 0);
		::gs_ortho(0.0F, static_cast<float>(targetWidth), 0.0F,
		    static_cast<float>(targetHeight), -100.0F, 100.0F);
		if (target == parent && !custom_draw && !async) {
			::obs_source_default_render(target);
		} else {
			::obs_source_video_render(target);
		}
	}
	auto inputWidth = static_cast<std::uint32_t>(m_Runtime->getInputWidth());
	auto inputHeight = static_cast<std::uint32_t>(m_Runtime->getInputHeight());
	::gs_texture_t *inputTexture;
	if (targetWidth == inputWidth && targetHeight == inputHeight) {
		::gs_copy_texture(
		    m_TargetTexture, ::gs_texrender_get_texture(m_RenderTarget));
		inputTexture = m_TargetTexture;
	} else {
		::gs_texrender_reset(m_RenderInput);
		::gs_blend_state_push();
		defer {
			::gs_blend_state_pop();
		};
		::gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
		if (!::gs_texrender_begin(m_RenderInput, inputWidth, inputHeight)) {
			return false;
		}
		defer {
			::gs_texrender_end(m_RenderInput);
		};
		::gs_ortho(0.0F, static_cast<float>(inputWidth), 0.0F,
		    static_cast<float>(inputHeight), -100.0F, 100.0F);
		::gs_texture_t *tex = ::gs_texrender_get_texture(m_RenderTarget);
		::gs_effect_set_texture(m_ScaleImgParam, tex);
		while (::gs_effect_loop(m_ScaleEffect, "Draw")) {
			::gs_draw_sprite(tex, 0, inputWidth, inputHeight);
		}
		inputTexture = ::gs_texrender_get_texture(m_RenderInput);
	}
	try {
		if (m_InputImage == nullptr) {
			createInputImage(inputTexture);
		}
		m_Runtime->processImage(
		    m_InputImage->getImage(), m_OutputImage->getImage());
	} catch (...) {
		logException();
		return false;
	}
	return true;
}

void JoshUpscaleFilter::renderMaskedTarget() noexcept {
	::gs_texture_t *tex = ::gs_texrender_get_texture(m_RenderTarget);
	::gs_effect_set_texture(m_BlendImgParam, tex);
	::gs_effect_set_texture(m_BlendMaskParam, m_MaskImage.texture);
	while (::gs_effect_loop(m_BlendEffect, "Draw")) {
		::gs_draw_sprite(tex, 0,
		    static_cast<std::uint32_t>(m_Runtime->getOutputWidth()),
		    static_cast<std::uint32_t>(m_Runtime->getOutputHeight()));
	}
}

void JoshUpscaleFilter::cleanup() noexcept {
	::obs_enter_graphics();
	defer {
		::obs_leave_graphics();
	};
	m_InputImage.reset();
	m_OutputImage.reset();
	if (m_RenderInput != nullptr) {
		::gs_texrender_destroy(m_RenderInput);
	}
	if (m_RenderTarget != nullptr) {
		::gs_texrender_destroy(m_RenderTarget);
	}
	if (m_TargetTexture != nullptr) {
		::gs_texture_destroy(m_TargetTexture);
	}
	if (m_OutputTexture != nullptr) {
		::gs_texture_destroy(m_OutputTexture);
	}
	if (m_BlendEffect != nullptr) {
		::gs_effect_destroy(m_BlendEffect);
	}
	::gs_image_file_free(&m_MaskImage);
}

}  // namespace obs

}  // namespace JoshUpscale
