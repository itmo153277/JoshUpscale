// Copyright 2021 Ivanov Viktor

#pragma once

#include <functional>
#include <string>
#include <vector>

namespace processor {

enum class DXVA { AUTO, FORCED, OFF };

struct SDevListItem {
	std::string deviceId;
	std::wstring deviceName;
};

using DeviceList = std::vector<SDevListItem>;

using PresentCallback = std::function<void()>;

void init();

void processAndShowVideo(const char *filename, const char *videoIn,
    const char *audioIn, const char *sourceOptions, const char *audioOut,
    DXVA dxva, bool showDebugInfo, PresentCallback cb);
DeviceList getVideoInDevices();
DeviceList getAudioInDevices();
DeviceList getAudioOutDevices();

}  // namespace processor
