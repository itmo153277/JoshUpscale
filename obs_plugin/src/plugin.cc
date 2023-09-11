// Copyright 2023 Viktor Ivanov

#include "JoshUpscale/obs/plugin.h"

#include "JoshUpscale/obs/filter.h"

const char *PLUGIN_NAME = "obs-joshupscale";
const char *PLUGIN_VERSION = "1.0.0";

obs_source_info *getJoshUpscaleSourceInfo() {
	return JoshUpscale::obs::JoshUpscaleFilter::getSourceInfo();
}
