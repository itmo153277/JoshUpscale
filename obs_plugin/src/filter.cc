// Copyright 2025 Viktor Ivanov

#include "JoshUpscale/obs/filter.h"

#include <graphics/graphics.h>
#include <graphics/image-file.h>
#include <graphics/vec2.h>
#include <obs-module.h>

#include <cstdint>

#include "JoshUpscale/core.h"

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
    [[maybe_unused]] ::obs_data_t *settings, ::obs_source_t *source)
    : m_Source{source} {
	auto modelFile = OBSPtr(obs_module_file("model.trt"));
	m_Runtime.reset(core::createRuntime(0, modelFile.get()));
	auto maskFile = OBSPtr(obs_module_file("mask.png"));
	::gs_image_file_init(&m_MaskImage, maskFile.get());
	auto blendEffectFile = OBSPtr(obs_module_file("effects/blend.effect"));
	{
		::obs_enter_graphics();
		defer {
			::obs_leave_graphics();
		};
		m_RenderTarget = ::gs_texrender_create(GS_BGRX, GS_ZS_NONE);
		m_RenderInput = ::gs_texrender_create(GS_BGRX_UNORM, GS_ZS_NONE);
		m_ScaleEffect = ::obs_get_base_effect(OBS_EFFECT_BILINEAR_LOWRES);
		m_ScaleImgParam = ::gs_effect_get_param_by_name(m_ScaleEffect, "image");
		m_OutputEffect = ::obs_get_base_effect(OBS_EFFECT_DEFAULT);
		m_OutputImgParam =
		    ::gs_effect_get_param_by_name(m_OutputEffect, "image");
		m_TargetTexture = ::gs_texture_create(
		    static_cast<std::uint32_t>(m_Runtime->getInputWidth()),
		    static_cast<std::uint32_t>(m_Runtime->getInputHeight()),
		    GS_BGRX_UNORM, 1, nullptr, 0);
		m_OutputTexture = ::gs_texture_create(
		    static_cast<std::uint32_t>(m_Runtime->getOutputWidth()),
		    static_cast<std::uint32_t>(m_Runtime->getOutputHeight()),
		    GS_BGRX_UNORM, 1, nullptr, 0);
		createOutputImage();
		m_BlendEffect =
		    ::gs_effect_create_from_file(blendEffectFile.get(), nullptr);
		m_BlendImgParam = ::gs_effect_get_param_by_name(m_BlendEffect, "image");
		m_BlendMaskParam = ::gs_effect_get_param_by_name(m_BlendEffect, "mask");
		m_BlendScaleParam =
		    ::gs_effect_get_param_by_name(m_BlendEffect, "scale");
		::gs_image_file_init_texture(&m_MaskImage);
	}
}

JoshUpscaleFilter::~JoshUpscaleFilter() {
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

const char *JoshUpscaleFilter::getName(
    [[maybe_unused]] void *typeData) noexcept {
	return ::obs_module_text("Name");
}

void *JoshUpscaleFilter::create(
    ::obs_data_t *settings, ::obs_source_t *source) noexcept {
	try {
		return new JoshUpscaleFilter(settings, source);
	} catch (...) {
		::blog(LOG_ERROR, "Exception: %s", core::getExceptionString().c_str());
		return nullptr;
	}
}

void JoshUpscaleFilter::destroy() noexcept {
	delete this;
}

void JoshUpscaleFilter::render(
    [[maybe_unused]] ::gs_effect_t *effect) noexcept {
	::obs_source_t *const target = ::obs_filter_get_target(m_Source);
	::obs_source_t *const parent = ::obs_filter_get_parent(m_Source);
	if (target == nullptr || parent == nullptr) {
		::obs_source_skip_video_filter(m_Source);
		return;
	}
	const uint32_t targetWidth = ::obs_source_get_base_width(target);
	const uint32_t targetHeight = ::obs_source_get_base_height(target);
	if (targetWidth == 0 || targetHeight == 0) {
		::obs_source_skip_video_filter(m_Source);
		return;
	}
	if (targetWidth != m_LastWidth || targetHeight != m_LastHeight) {
		m_LastWidth = targetWidth;
		m_LastHeight = targetHeight;
		m_InputImage.reset();
		m_FrameReady = false;
	}
	if (!m_FrameReady) {
		const uint32_t targetFlags = ::obs_source_get_output_flags(target);
		bool custom_draw = (targetFlags & OBS_SOURCE_CUSTOM_DRAW) != 0;
		bool async = (targetFlags & OBS_SOURCE_ASYNC) != 0;
		::gs_texrender_reset(m_RenderTarget);
		::gs_blend_state_push();
		::gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
		if (::gs_texrender_begin(m_RenderTarget, targetWidth, targetHeight)) {
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
			::gs_texrender_end(m_RenderTarget);
		}
		::gs_blend_state_pop();
		auto inputWidth =
		    static_cast<std::uint32_t>(m_Runtime->getInputWidth());
		auto inputHeight =
		    static_cast<std::uint32_t>(m_Runtime->getInputHeight());
		::gs_texture_t *inputTexture;
		if (targetWidth == inputWidth && targetHeight == inputHeight) {
			::gs_copy_texture(
			    m_TargetTexture, ::gs_texrender_get_texture(m_RenderTarget));
			inputTexture = m_TargetTexture;
		} else {
			::gs_texrender_reset(m_RenderInput);
			::gs_blend_state_push();
			::gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
			if (::gs_texrender_begin(m_RenderInput, inputWidth, inputHeight)) {
				::gs_ortho(0.0F, static_cast<float>(inputWidth), 0.0F,
				    static_cast<float>(inputHeight), -100.0F, 100.0F);
				::gs_texture_t *tex =
				    ::gs_texrender_get_texture(m_RenderTarget);
				::gs_effect_set_texture(m_ScaleImgParam, tex);
				while (::gs_effect_loop(m_ScaleEffect, "Draw")) {
					::gs_draw_sprite(tex, 0, inputWidth, inputHeight);
				}
				::gs_texrender_end(m_RenderInput);
			}
			::gs_blend_state_pop();
			inputTexture = ::gs_texrender_get_texture(m_RenderInput);
		}
		try {
			if (m_InputImage == nullptr) {
				createInputImage(inputTexture);
			}
			m_Runtime->processImage(
			    m_InputImage->getImage(), m_OutputImage->getImage());
		} catch (...) {
			::blog(
			    LOG_ERROR, "Exception: %s", core::getExceptionString().c_str());
			::obs_source_skip_video_filter(m_Source);
			return;
		}
		m_FrameReady = true;
		m_FrameDuration = 0;
	}
	::gs_blend_state_push();
	::gs_reset_blend_state();
	::gs_effect_set_texture(m_OutputImgParam, m_OutputTexture);
	while (::gs_effect_loop(m_OutputEffect, "Draw")) {
		::gs_draw_sprite(m_OutputTexture, 0, 0, 0);
	}
	::gs_texture_t *tex = ::gs_texrender_get_texture(m_RenderTarget);
	::gs_effect_set_texture(m_BlendImgParam, tex);
	::gs_effect_set_texture(m_BlendMaskParam, m_MaskImage.texture);
	::vec2 scales;
	::vec2_set(&scales,
	    static_cast<float>(m_MaskImage.cx) / static_cast<float>(targetWidth),
	    static_cast<float>(m_MaskImage.cy) / static_cast<float>(targetHeight));
	::gs_effect_set_vec2(m_BlendScaleParam, &scales);
	while (::gs_effect_loop(m_BlendEffect, "Draw")) {
		::gs_draw_sprite(tex, 0, getWidth(), getHeight());
	}
	::gs_blend_state_pop();
}

void JoshUpscaleFilter::videoTick(float seconds) noexcept {
	m_FrameDuration += seconds;
	if (m_FrameDuration > 0.03F) {
		m_FrameReady = false;
	}
}

std::uint32_t JoshUpscaleFilter::getWidth() noexcept {
	return static_cast<std::uint32_t>(m_Runtime->getOutputWidth());
}

std::uint32_t JoshUpscaleFilter::getHeight() noexcept {
	return static_cast<std::uint32_t>(m_Runtime->getOutputHeight());
}

void JoshUpscaleFilter::createInputImage(::gs_texture_t *texture) {
#ifdef _WIN32
	struct ID3D11Texture2D *d3d11Texture =
	    reinterpret_cast<ID3D11Texture2D *>(::gs_texture_get_obj(texture));
	m_InputImage.reset(core::getD3D11Image(
	    d3d11Texture, core::GraphicsResourceImageType::INPUT));
#else
	(void) texture;
#endif
}

void JoshUpscaleFilter::createOutputImage() {
#ifdef _WIN32
	struct ID3D11Texture2D *d3d11Texture = reinterpret_cast<ID3D11Texture2D *>(
	    ::gs_texture_get_obj(m_OutputTexture));
	m_OutputImage.reset(core::getD3D11Image(
	    d3d11Texture, core::GraphicsResourceImageType::OUTPUT));
#endif
}

}  // namespace obs

}  // namespace JoshUpscale
