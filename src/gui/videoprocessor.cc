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
void callOrThrow(F f, Params &&...params) {
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
	T *&releaseAndGet() {
		if (value != nullptr) {
			value->Release();
			value = nullptr;
		}
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
	T &freeAndGet() {
		if (value) {
			CoTaskMemFree(value);
			value = nullptr;
		}
		return value;
	}
};

HRESULT EnumerateDevices(REFGUID category, IEnumMoniker **ppEnum) {
	ICreateDevEnum *pDevEnum;
	HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, nullptr,
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
void getDShowDeviceList(
    processor::DeviceList *list, REFGUID category, REFGUID media) {
	AutoRelease<IEnumMoniker> pEnum;
	AutoRelease<IMoniker> pMoniker;

	callOrThrow(EnumerateDevices, category, &pEnum.get());
	while (pEnum->Next(1, &pMoniker.releaseAndGet(), nullptr) == S_OK) {
		AutoRelease<IBaseFilter> pFilter;
		HRESULT hr = pMoniker->BindToObject(nullptr, nullptr, IID_IBaseFilter,
		    reinterpret_cast<void **>(&pFilter.get()));
		if (FAILED(hr)) {
			continue;
		}
		AutoRelease<IEnumPins> pEnumPins;
		hr = pFilter->EnumPins(&pEnumPins.get());
		if (FAILED(hr)) {
			continue;
		}
		AutoRelease<IPin> pPin;
		bool found = false;
		while (!found &&
		       pEnumPins->Next(1, &pPin.releaseAndGet(), nullptr) == S_OK) {
			PIN_DIRECTION pinDirection;
			hr = pPin->QueryDirection(&pinDirection);
			if (FAILED(hr) || pinDirection != PINDIR_OUTPUT) {
				continue;
			}
			AutoFree<AM_MEDIA_TYPE *> pMediaType;
			AutoRelease<IEnumMediaTypes> pEnumMediaTypes;
			hr = pPin->EnumMediaTypes(&pEnumMediaTypes.get());
			if (FAILED(hr)) {
				continue;
			}
			while (pEnumMediaTypes->Next(
			           1, &pMediaType.freeAndGet(), nullptr) == S_OK) {
				if (pMediaType->majortype == media) {
					found = true;
					break;
				}
			}
		}
		if (!found) {
			continue;
		}

		processor::SDevListItem device;
		AutoRelease<IPropertyBag> pPropBag;
		hr = pMoniker->BindToStorage(0, 0, IID_PPV_ARGS(&pPropBag.get()));
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
		list->push_back(device);
	}
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
    const char *audioIn, const char *sourceOptions, const char *audioOut,
    DXVA dxva, bool showDebugInfo, PresentCallback cb) {
	assert(filename != nullptr || videoIn != nullptr);
	static const uint8_t emptyBuffer[upscaler::INPUT_WIDTH * 3] = {};
	std::string source =
	    filename != nullptr ? filename : getSourceString(videoIn, audioIn);
	upscaler::SUpscaler upscaler{"model.pb"};
	upscaler.upscaleFrame(emptyBuffer, 0);
	player::SPlayer player{upscaler::INPUT_WIDTH, upscaler::INPUT_HEIGHT,
	    upscaler::OUTPUT_WIDTH, upscaler::OUTPUT_HEIGHT, source.c_str(),
	    filename == nullptr ? "dshow" : nullptr, sourceOptions,
	    static_cast<ffmpeg::DXVA>(dxva), audioOut, showDebugInfo,
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
	processor::DeviceList list;
	getDShowDeviceList(&list, CLSID_VideoInputDeviceCategory, MEDIATYPE_Video);
	return list;
}

processor::DeviceList processor::getAudioInDevices() {
	processor::DeviceList list;
	getDShowDeviceList(&list, CLSID_AudioInputDeviceCategory, MEDIATYPE_Audio);
	getDShowDeviceList(&list, CLSID_VideoInputDeviceCategory, MEDIATYPE_Audio);
	return list;
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
