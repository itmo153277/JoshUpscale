// Copyright 2023 Viktor Ivanov

#include <obs-module.h>

#include "JoshUpscale/obs/plugin.h"

OBS_DECLARE_MODULE()

OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

bool obs_module_load(void) {
	preloadLibraries();
	obs_register_source(getJoshUpscaleSourceInfo());
	return true;
}

void obs_module_unload(void) {
}
