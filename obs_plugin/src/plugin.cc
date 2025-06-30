// Copyright 2025 Viktor Ivanov

#include "JoshUpscale/obs/plugin.h"

#include <JoshUpscale/core.h>

#include <exception>
#include <string>

#include "JoshUpscale/obs/filter.h"
#include "JoshUpscale/obs/logging.h"

const char *PLUGIN_NAME = "obs-joshupscale";
const char *PLUGIN_VERSION = "2.0.0";

obs_source_info *getJoshUpscaleSourceInfo() {
	return JoshUpscale::obs::JoshUpscaleFilter::getSourceInfo();
}

#ifdef _WIN32
#include <Windows.h>
#ifdef ERROR
#undef ERROR
#endif

#include <filesystem>
#include <stdexcept>
#include <system_error>

#include "JoshUpscale/obs/utils.h"

namespace {

[[maybe_unused]] std::filesystem::path findNvVfx() {
	WCHAR buffer[MAX_PATH];
	DWORD ret =
	    GetEnvironmentVariableW(L"NV_VIDEO_EFFECTS_PATH", buffer, MAX_PATH);
	if (ret != 0 && ret < MAX_PATH) {
		return buffer;
	}
	ret = GetEnvironmentVariableW(L"ProgramFiles", buffer, MAX_PATH);
	if (ret == 0 || ret >= MAX_PATH) {
		return "C:\\Program Files\\NVIDIA Corporation\\NVIDIA Video Effects";
	}
	return std::filesystem::path(buffer) / "NVIDIA Corporation" /
	       "NVIDIA Video Effects";
}

void preloadLibrariesWindows() {
#ifdef JOSHUPSCALE_NVVFX
	DLL_DIRECTORY_COOKIE nvVfxPtr =
	    AddDllDirectory(std::filesystem::absolute(findNvVfx()).c_str());
	if (nvVfxPtr == NULL) {
		throw std::system_error(static_cast<int>(GetLastError()),
		    std::system_category(), "Failed to set NVVFX path");
	}
	defer {
		RemoveDllDirectory(nvVfxPtr);
	};
#endif
	auto mainLibObsPath =
	    JoshUpscale::obs::OBSPtr(obs_module_file("JoshUpscale.dll"));
	if (!mainLibObsPath) {
		throw std::runtime_error("Failed to find main library");
	}
	std::filesystem::path mainLibPath =
	    std::filesystem::absolute(mainLibObsPath.get()).concat(".");
	HMODULE handle = LoadLibraryExW(mainLibPath.c_str(), NULL,
	    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR);
	if (handle == NULL) {
		throw std::system_error(static_cast<int>(GetLastError()),
		    std::system_category(), "Failed to load main library");
	}
}

}  // namespace

#endif

bool preloadLibraries() {
	try {
#ifdef _WIN32
		preloadLibrariesWindows();
#endif
		return true;
	} catch (std::exception &e) {
		JoshUpscale::obs::log(
		    JoshUpscale::core::LogLevel::ERROR, "Failed to load: %s", e.what());
		return false;
	}
}

void setupLogging() {
	struct LogSink : JoshUpscale::core::LogSink {
		void operator()(const char *tag, JoshUpscale::core::LogLevel logLevel,
		    const std::string &message) override {
			if (logLevel == JoshUpscale::core::LogLevel::ERROR ||
			    logLevel == JoshUpscale::core::LogLevel::WARNING) {
				JoshUpscale::obs::log(logLevel, "%s: %s", tag, message.c_str());
			}
		}
	};

	static LogSink logSink;
	JoshUpscale::core::setLogSink(&logSink);
}
