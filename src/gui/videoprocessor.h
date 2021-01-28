#pragma once
#include <vector>
#include <string>
#include <functional>

namespace processor {

enum class DXVA { AUTO, FORCED, OFF };

struct SDevListItem {
	std::string deviceId;
	std::wstring deviceName;
};

using DeviceList = std::vector<SDevListItem>;

using PresentCallback = std::function<void()>;

void init();

void processAndShowVideo(const char *videoIn, const char *audioIn,
    const char *audioOut, DXVA dxva, PresentCallback cb);
DeviceList getVideoInDevices();
DeviceList getAudioInDevices();
DeviceList getAudioOutDevices();

}  // namespace processor
