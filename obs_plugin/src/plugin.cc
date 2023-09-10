// Copyright 2023 Viktor Ivanov

#include "JoshUpscale/obs/plugin.h"

#include "JoshUpscale/obs/filter.h"

obs_source_info *getJoshUpscaleSourceInfo() {
	return JoshUpscale::obs::JoshUpscaleFilter::getSourceInfo();
}
