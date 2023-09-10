// Copyright 2023 Viktor Ivanov

#include "JoshUpscale/obs/filter.h"

#include <JoshUpscale/core.h>

#include <cstddef>

namespace JoshUpscale {

namespace obs {

::obs_source_info *JoshUpscaleFilter::getSourceInfo() {
	static struct Data {
		::obs_source_info info = {};
		Data() {
			info.id = "joshupscale";
			info.type = OBS_SOURCE_TYPE_FILTER;
			info.output_flags = OBS_SOURCE_ASYNC_VIDEO;
			info.get_name = Callback<&JoshUpscaleFilter::getName>::getPtr();
			info.create = &JoshUpscaleFilter::create;
			info.destroy = &JoshUpscaleFilter::destroy;
			info.filter_video =
			    Callback<&JoshUpscaleFilter::filterVideo>::getPtr();
		}
	} data;
	return &data.info;
}

JoshUpscaleFilter::JoshUpscaleFilter(::obs_source_t *source)
    : m_Source(source)
    , m_InputBuffer(core::INPUT_WIDTH * core::OUTPUT_HEIGHT * 3)
    , m_OutputFrame(core::OUTPUT_WIDTH, core::OUTPUT_HEIGHT) {
	// TODO(me): Ability to select model
	m_Runtime.reset(core::createRuntime(0, obs_module_file("model.yaml")));
}

JoshUpscaleFilter::~JoshUpscaleFilter() {
}

::obs_source_frame *JoshUpscaleFilter::filterVideo(::obs_source_frame *frame) {
	::obs_source_t *parent = obs_filter_get_parent(m_Source);
	m_ScaleContext.scale(frame, m_InputBuffer.get());
	::obs_source_release_frame(parent, frame);
	try {
		core::Image inputImage = {
		    .ptr = m_InputBuffer.get(),
		    .stride = static_cast<std::ptrdiff_t>(core::INPUT_WIDTH * 3),
		    .width = core::INPUT_WIDTH,
		    .height = core::INPUT_HEIGHT,
		};
		core::Image outputImage = {
		    .ptr = m_OutputFrame->data[0],
		    .stride = static_cast<std::ptrdiff_t>(m_OutputFrame->linesize[0]),
		    .width = core::INPUT_WIDTH,
		    .height = core::INPUT_HEIGHT,
		};
		// TODO(me): async processing
		m_Runtime->processImage(inputImage, outputImage);
		return m_OutputFrame.get();
	} catch (...) {
		return nullptr;
	}
}

}  // namespace obs

}  // namespace JoshUpscale
