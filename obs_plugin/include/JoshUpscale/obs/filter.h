// Copyright 2025 Viktor Ivanov

#pragma once

#include <JoshUpscale/core.h>
#include <graphics/image-file.h>
#include <obs-module.h>

#include <memory>

namespace JoshUpscale {

namespace obs {

namespace detail {

struct OBSDeleter {
	void operator()(void *data) noexcept {
		::bfree(data);
	}
};

}  // namespace detail

template <class T>
struct OBSPtr : std::unique_ptr<T, detail::OBSDeleter> {
	using unique_ptr = std::unique_ptr<T, detail::OBSDeleter>;

	explicit OBSPtr(T *val) noexcept : unique_ptr(val) {
	}
};

struct JoshUpscaleFilter {
	static ::obs_source_info *getSourceInfo();

private:
	JoshUpscaleFilter(::obs_data_t *settings, ::obs_source_t *source);
	~JoshUpscaleFilter();

	static const char *getName(void *typeData) noexcept;

	static void *create(
	    ::obs_data_t *settings, ::obs_source_t *source) noexcept;
	void destroy() noexcept;
	void update(::obs_data_t *settings) noexcept;
	static void getDefaults(void *typeData, ::obs_data_t *settings) noexcept;
	static ::obs_properties_t *getProperties(
	    void *data, void *typeData) noexcept;
	void render(::gs_effect_t *effect) noexcept;
	void videoTick(float seconds) noexcept;
	std::uint32_t getWidth() noexcept;
	std::uint32_t getHeight() noexcept;

	template <auto Ptr>
	struct Callback {
		static consteval decltype(Ptr) getPtr() {
			return Ptr;
		}
	};

	template <typename R, typename... T,
	    R (JoshUpscaleFilter::*Ptr)(T...) noexcept>
	struct Callback<Ptr> {
		static consteval R (*getPtr())(void *, T...) noexcept {
			return [](void *self, T... params) noexcept -> R {
				return (reinterpret_cast<JoshUpscaleFilter *>(self)->*Ptr)(
				    params...);
			};
		}
	};

	void createInputImage(::gs_texture_t *texture);
	void createOutputImage();
	void initModel(const char *model) noexcept;
	bool processFrame() noexcept;
	void renderMaskedTarget() noexcept;

	void cleanup() noexcept;

	::obs_source_t *m_Source;
	int m_Device = -1;
	std::unique_ptr<core::Runtime> m_Runtime = nullptr;
	::gs_texrender_t *m_RenderTarget = nullptr;
	::gs_texrender_t *m_RenderInput = nullptr;
	::gs_texture_t *m_TargetTexture = nullptr;
	::gs_texture_t *m_OutputTexture = nullptr;
	std::unique_ptr<core::GraphicsResourceImage> m_InputImage = nullptr;
	std::unique_ptr<core::GraphicsResourceImage> m_OutputImage = nullptr;
	std::uint32_t m_LastWidth = 0;
	std::uint32_t m_LastHeight = 0;
	::gs_effect_t *m_ScaleEffect = nullptr;
	::gs_eparam_t *m_ScaleImgParam = nullptr;
	::gs_effect_t *m_OutputEffect = nullptr;
	::gs_eparam_t *m_OutputImgParam = nullptr;
	::gs_effect_t *m_BlendEffect = nullptr;
	::gs_eparam_t *m_BlendImgParam = nullptr;
	::gs_eparam_t *m_BlendMaskParam = nullptr;
	::gs_image_file_t m_MaskImage = {};
	float m_FrameDuration = 0.0F;
	bool m_FrameProcessed = false;
	int m_Model = -1;
	bool m_LimitFps = true;
	bool m_RenderMaskedTarget = false;
};

}  // namespace obs

}  // namespace JoshUpscale
