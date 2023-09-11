// Copyright 2023 Viktor Ivanov

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <obs-module.h>

extern const char *PLUGIN_NAME;
extern const char *PLUGIN_VERSION;

struct obs_source_info *getJoshUpscaleSourceInfo();

#ifdef __cplusplus
}
#endif
