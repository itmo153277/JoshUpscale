// Copyright 2025 Viktor Ivanov

#pragma once

#include <graphics/graphics.h>
extern "C" {
#include <obs-module.h>
}

#include <JoshUpscale/core.h>

#include <memory>
#include <utility>

namespace JoshUpscale {

namespace obs {

namespace detail {

struct OBSDeleter {
	void operator()(void *data) {
		::bfree(data);
	}
};

}  // namespace detail

template <class T>
struct OBSPtr : std::unique_ptr<T, detail::OBSDeleter> {
	using unique_ptr = std::unique_ptr<T, detail::OBSDeleter>;

	explicit OBSPtr(T *val) : unique_ptr(val) {
	}
};

template <typename T>
struct Defer {
	T m_DeferFn;
	explicit Defer(T &&fn) : m_DeferFn(std::move(fn)) {
	}
	Defer(const Defer &) = delete;
	Defer(Defer &&) noexcept = delete;
	~Defer() {
		m_DeferFn();
	}
};

struct DeferOp {};

template <typename T>
Defer<T> operator*([[maybe_unused]] DeferOp op, T &&fn) {
	return Defer{std::forward<T>(fn)};
}

#define DEFER_OP_NAME_(LINE) _defer##LINE
#define DEFER_OP_NAME(LINE) DEFER_OP_NAME_(LINE)
#define defer auto DEFER_OP_NAME(__LINE__) = DeferOp{} *[&]()  // NOLINT

struct JoshUpscaleFilter {
	static ::obs_source_info *getSourceInfo();

private:
	JoshUpscaleFilter(::obs_data_t *settings, ::obs_source_t *source);
	~JoshUpscaleFilter();

	static const char *getName(void *typeData) noexcept;

	static void *create(
	    ::obs_data_t *settings, ::obs_source_t *source) noexcept;
	void destroy() noexcept;
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

	::obs_source_t *m_Source;
	std::unique_ptr<core::Runtime> m_Runtime = nullptr;
	::gs_texrender_t *m_RenderTarget = nullptr;
	::gs_texrender_t *m_RenderInput = nullptr;
	::gs_texture_t *m_TargetTexture = nullptr;
	::gs_texture_t *m_OutputTexture = nullptr;
	std::unique_ptr<core::GraphicsResourceImage> m_InputImage = nullptr;
	std::unique_ptr<core::GraphicsResourceImage> m_OutputImage = nullptr;
	std::uint32_t m_LastWidth = 0;
	std::uint32_t m_LastHeight = 0;
	::vec2 m_Dimension = {};
	::vec2 m_DimensionInv = {};
	::gs_effect_t *m_ScaleEffect = nullptr;
	::gs_eparam_t *m_ScaleImgParam = nullptr;
	::gs_effect_t *m_OutputEffect = nullptr;
	::gs_eparam_t *m_OutputImgParam = nullptr;
	float m_FrameDuration = 0.0F;
	bool m_FrameReady = false;
};

}  // namespace obs

}  // namespace JoshUpscale
