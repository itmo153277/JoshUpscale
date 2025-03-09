// Copyright 2025 Viktor Ivanov

#include "JoshUpscale/obs/filter.h"

namespace JoshUpscale {

namespace obs {

::obs_source_info *JoshUpscaleFilter::getSourceInfo() {
	struct Eval {
		consteval static ::obs_source_info createSourceInfo() {
			::obs_source_info info = {};
#define CALLBACK_DEF(name) (Callback<&JoshUpscaleFilter::name>::getPtr())
			info.id = "joshupscale";
			info.type = OBS_SOURCE_TYPE_FILTER;
			info.output_flags = OBS_SOURCE_VIDEO;
			info.get_name = CALLBACK_DEF(getName);
			info.create = CALLBACK_DEF(create);
			info.destroy = CALLBACK_DEF(destroy);
			info.video_render = CALLBACK_DEF(render);
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
}

JoshUpscaleFilter::~JoshUpscaleFilter() {
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
		return nullptr;
	}
}

void JoshUpscaleFilter::destroy(void *data) noexcept {
	delete reinterpret_cast<JoshUpscaleFilter *>(data);
}

void JoshUpscaleFilter::render(
    [[maybe_unused]] ::gs_effect_t *effect) noexcept {
	::obs_source_skip_video_filter(m_Source);
}

}  // namespace obs

}  // namespace JoshUpscale
