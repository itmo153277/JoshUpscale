// Copyright 2021 Ivanov Viktor

#include "videoprocessor.h"

#define WIN32_LEAN_AND_MEAN

#include <dshow.h>
#include <windows.h>

#include <cassert>
#include <codecvt>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

#include "ffmpeg_wrappers.h"
#include "player.h"
#include "sdl_wrappers.h"
#include "upscaler.h"

std::wstring convertString(const std::string &str) {
	return std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t>{}
	    .from_bytes(str);
}

std::string convertString(const std::wstring &str) {
	return std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t>{}
	    .to_bytes(str);
}

void throwSystemError(HRESULT hr) {
	throw std::system_error(hr, std::system_category());
}

template <typename F, typename... Params>
void callOrThrow(F f, Params &&... params) {
	HRESULT hr = f(std::forward<Params>(params)...);
	if (FAILED(hr)) {
		throwSystemError(hr);
	}
}

template <typename T>
struct AutoRelease {
	AutoRelease() = default;
	AutoRelease(const AutoRelease &) = delete;
	AutoRelease(AutoRelease &&s) noexcept {
		value = s.value;
		s.value = nullptr;
	}
	~AutoRelease() {
		if (value) {
			value->Release();
		}
	}

	T *value = nullptr;

	T *operator->() {
		return value;
	}
	operator bool() {
		return value;
	}
	T *&get() {
		return value;
	}
};

struct AutoVariant : VARIANT {
	AutoVariant() {
		VariantInit(this);
	}
	AutoVariant(const AutoVariant &) = delete;
	AutoVariant(AutoVariant &&) = delete;
	~AutoVariant() {
		VariantClear(this);
	}
};

template <typename T>
struct AutoFree {
	AutoFree() = default;
	AutoFree(const AutoFree &) = delete;
	AutoFree(AutoFree &&s) noexcept {
		value = s.value;
		s.value = nullptr;
	}
	~AutoFree() {
		if (value) {
			CoTaskMemFree(value);
		}
	}

	T value = nullptr;

	T operator->() {
		return value;
	}
	operator bool() {
		return value;
	}
	T &get() {
		return value;
	}
};

HRESULT EnumerateDevices(REFGUID category, IEnumMoniker **ppEnum) {
	ICreateDevEnum *pDevEnum;
	HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL,
	    CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pDevEnum));

	if (SUCCEEDED(hr)) {
		hr = pDevEnum->CreateClassEnumerator(category, ppEnum, 0);
		if (hr == S_FALSE) {
			hr = VFW_E_NOT_FOUND;
		}
		pDevEnum->Release();
	}
	return hr;
}

processor::DeviceList getDShowDeviceList(REFGUID category) {
	AutoRelease<IEnumMoniker> pEnum;
	AutoRelease<IMoniker> pMoniker;
	processor::DeviceList list;

	callOrThrow(EnumerateDevices, category, &pEnum.get());
	while (pEnum->Next(1, &pMoniker.get(), NULL) == S_OK) {
		processor::SDevListItem device;
		AutoRelease<IPropertyBag> pPropBag;
		HRESULT hr =
		    pMoniker->BindToStorage(0, 0, IID_PPV_ARGS(&pPropBag.get()));
		if (FAILED(hr)) {
			continue;
		}
		AutoVariant var;
		hr = pPropBag->Read(L"FriendlyName", &var, nullptr);
		if (FAILED(hr)) {
			continue;
		}
		device.deviceName = var.bstrVal;

		AutoFree<LPOLESTR> oleStr;
		AutoRelease<IBindCtx> pBindCtx;
		callOrThrow(CreateBindCtx, 0, &pBindCtx.get());
		hr = pMoniker->GetDisplayName(pBindCtx.get(), nullptr, &oleStr.get());
		if (FAILED(hr)) {
			continue;
		}
		std::string displayName = convertString(oleStr.get());
		for (auto &c : displayName) {
			if (c == ':') {
				c = '_';
			}
		}
		device.deviceId = displayName;
		list.push_back(device);
	}
	return list;
}

std::string getSourceString(const char *videoIn, const char *audioIn) {
	std::stringstream ss;

	ss << "video=" << videoIn;
	if (audioIn) {
		ss << ":audio=" << audioIn;
	}
	return ss.str();
}

void processor::init() {
	ffmpeg::init();
	sdl::init();
}

void processor::processAndShowVideo(const char *filename, const char *videoIn,
    const char *audioIn, const char *audioOut, DXVA dxva, PresentCallback cb) {
	assert(filename != nullptr || videoIn != nullptr);
	std::string source =
	    filename != nullptr ? filename : getSourceString(videoIn, audioIn);
	upscaler::SUpscaler upscaler{"model.pb"};
	player::SPlayer player{upscaler::INPUT_WIDTH, upscaler::INPUT_HEIGHT,
	    upscaler::OUTPUT_WIDTH, upscaler::OUTPUT_HEIGHT, source.c_str(),
	    filename == nullptr ? "dshow" : nullptr,
	    static_cast<ffmpeg::DXVA>(dxva), audioOut,
	    [&upscaler](void *buf, std::size_t stride) {
		    upscaler.upscaleFrame(
		        reinterpret_cast<uint8_t *>(buf), static_cast<int>(stride));
	    },
	    [&upscaler](void *buf, std::size_t stride) {
		    upscaler.writeOutput(
		        reinterpret_cast<uint8_t *>(buf), static_cast<int>(stride));
	    }};
	player.play([cb] { cb(); });
}

processor::DeviceList processor::getVideoInDevices() {
	return getDShowDeviceList(CLSID_VideoInputDeviceCategory);
}

processor::DeviceList processor::getAudioInDevices() {
	return getDShowDeviceList(CLSID_AudioInputDeviceCategory);
}

processor::DeviceList processor::getAudioOutDevices() {
	processor::DeviceList deviceList(::SDL_GetNumAudioDevices(false));
	for (std::size_t i = 0; i < deviceList.size(); ++i) {
		const char *deviceName =
		    ::SDL_GetAudioDeviceName(static_cast<int>(i), false);
		if (!deviceName) {
			throw sdl::SDLException();
		}
		deviceList[i].deviceId = deviceName;
		deviceList[i].deviceName = convertString(deviceName);
	}
	return deviceList;
}
