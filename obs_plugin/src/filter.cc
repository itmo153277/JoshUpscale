// Copyright 2025 Viktor Ivanov

#include "JoshUpscale/obs/filter.h"

#include <graphics/graphics.h>
#include <obs.h>
#include <util/base.h>

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
	float inputWidthFp = static_cast<float>(m_Runtime->getInputWidth());
	float inputHeightFp = static_cast<float>(m_Runtime->getInputHeight());
	::vec2_set(&m_Dimension, inputWidthFp, inputHeightFp);
	::vec2_set(&m_DimensionInv, 1.0F / inputWidthFp, 1.0F / inputHeightFp);
	{
		::obs_enter_graphics();
		defer {
			::obs_leave_graphics();
		};
		m_ScaleEffect = ::obs_get_base_effect(OBS_EFFECT_BICUBIC);
		m_ScaleImgParam = ::gs_effect_get_param_by_name(m_ScaleEffect, "image");
		m_ScaleDimensionParam =
		    ::gs_effect_get_param_by_name(m_ScaleEffect, "base_dimension");
		m_ScaleDimensionInvParam =
		    ::gs_effect_get_param_by_name(m_ScaleEffect, "base_dimension_i");
		m_OutputEffect = ::obs_get_base_effect(OBS_EFFECT_DEFAULT);
		m_OutputImgParam =
		    ::gs_effect_get_param_by_name(m_OutputEffect, "image");
		m_OutputTexture = ::gs_texture_create(
		    static_cast<std::uint32_t>(m_Runtime->getOutputWidth()),
		    static_cast<std::uint32_t>(m_Runtime->getOutputHeight()),
		    GS_BGRX_UNORM, 1, nullptr, 0);
		createOutputImage();
	}
}

JoshUpscaleFilter::~JoshUpscaleFilter() {
	::obs_enter_graphics();
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
	::obs_leave_graphics();
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
	if (!m_FrameReady) {
		::obs_source_t *const target = ::obs_filter_get_target(m_Source);
		::obs_source_t *const parent = ::obs_filter_get_parent(m_Source);
		const uint32_t target_flags = ::obs_source_get_output_flags(target);
		bool custom_draw = (target_flags & OBS_SOURCE_CUSTOM_DRAW) != 0;
		bool async = (target_flags & OBS_SOURCE_ASYNC) != 0;
		const uint32_t width = ::obs_source_get_base_width(target);
		const uint32_t height = ::obs_source_get_base_height(target);
		if (width != m_LastWidth || height != m_LastHeight) {
			m_LastWidth = width;
			m_LastHeight = height;
			if (m_TargetTexture != nullptr) {
				m_InputImage.reset();
				::gs_texture_destroy(m_TargetTexture);
				m_TargetTexture = nullptr;
			}
			if (m_RenderTarget != nullptr) {
				::gs_texrender_destroy(m_RenderTarget);
				m_RenderTarget = nullptr;
			}
		}
		if (m_RenderTarget == nullptr) {
			m_RenderTarget = ::gs_texrender_create(GS_BGRX, GS_ZS_NONE);
		}
		::gs_texrender_reset(m_RenderTarget);
		::gs_blend_state_push();
		::gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
		if (::gs_texrender_begin(m_RenderTarget, width, height)) {
			::vec4 clear_color;
			::vec4_zero(&clear_color);
			::gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0F, 0);
			::gs_ortho(0.0F, static_cast<float>(width), 0.0F,
			    static_cast<float>(height), -100.0F, 100.0F);
			if (target == parent && !custom_draw && !async) {
				::obs_source_default_render(target);
			} else {
				::obs_source_video_render(target);
			}
			::gs_texrender_end(m_RenderTarget);
		}
		::gs_blend_state_pop();
		auto inWidth = static_cast<std::uint32_t>(m_Runtime->getInputWidth());
		auto inHeight = static_cast<std::uint32_t>(m_Runtime->getInputHeight());
		::gs_texture_t *inputTexture;
		if (inWidth == width && inHeight == height) {
			if (m_RenderInput != nullptr) {
				m_InputImage.reset();
				::gs_texrender_destroy(m_RenderInput);
				m_RenderInput = nullptr;
			}
			if (m_TargetTexture == nullptr) {
				m_TargetTexture =
				    ::gs_texture_create(width, height, GS_BGRX_UNORM, 1, nullptr, 0);
			}
			::gs_copy_texture(
			    m_TargetTexture, ::gs_texrender_get_texture(m_RenderTarget));
			inputTexture = m_TargetTexture;
		} else {
			if (m_RenderInput == nullptr) {
				m_RenderInput =
				    ::gs_texrender_create(GS_BGRX_UNORM, GS_ZS_NONE);
			}
			::gs_texrender_reset(m_RenderInput);
			::gs_blend_state_push();
			::gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
			if (::gs_texrender_begin(m_RenderInput, inWidth, inHeight)) {
				::gs_ortho(0.0F, static_cast<float>(width), 0.0F,
				    static_cast<float>(height), -100.0F, 100.0F);
				::gs_texture_t *tex =
				    ::gs_texrender_get_texture(m_RenderTarget);
				::gs_effect_set_texture(m_ScaleImgParam, tex);
				::gs_effect_set_vec2(m_ScaleDimensionParam, &m_Dimension);
				::gs_effect_set_vec2(m_ScaleDimensionInvParam, &m_DimensionInv);
				while (::gs_effect_loop(m_ScaleEffect, "Draw")) {
					::gs_draw(GS_TRISTRIP, 0, 0);
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
	}
	::gs_blend_state_push();
	::gs_reset_blend_state();
	::gs_effect_set_texture(m_OutputImgParam, m_OutputTexture);
	while (::gs_effect_loop(m_OutputEffect, "Draw")) {
		::gs_draw_sprite(m_OutputTexture, 0, 0, 0);
	}
	::gs_blend_state_pop();
}

void JoshUpscaleFilter::videoTick([[maybe_unused]] float seconds) noexcept {
	m_FrameReady = false;
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
