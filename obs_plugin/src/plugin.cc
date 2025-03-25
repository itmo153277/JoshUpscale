// Copyright 2025 Viktor Ivanov

#include "JoshUpscale/obs/plugin.h"

#include <JoshUpscale/core.h>

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
#include <cstddef>
#include <filesystem>

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

[[maybe_unused]] void setDllDir(std::nullptr_t) {
	SetDllDirectoryW(NULL);
}

[[maybe_unused]] void setDllDir(const wchar_t *dir) {
	SetDllDirectoryW(dir);
}

[[maybe_unused]] void setDllDir(const std::filesystem::path &dir) {
	setDllDir(dir.c_str());
}

void preloadLibrariesWindows() {
#ifdef JOSHUPSCALE_NVVFX
	setDllDir(findNvVfx());
#endif
	auto mainLibPath =
	    JoshUpscale::obs::OBSPtr(obs_module_file("JoshUpscale.dll"));
	std::filesystem::path mainLib = mainLibPath.get();
	DLL_DIRECTORY_COOKIE ptr = AddDllDirectory(
	    std::filesystem::absolute(mainLib.parent_path()).c_str());
	LoadLibraryExA("JoshUpscale.dll", NULL, LOAD_LIBRARY_SEARCH_USER_DIRS);
	RemoveDllDirectory(ptr);
	setDllDir(nullptr);
}

}  // namespace

#endif

void preloadLibraries() {
#ifdef _WIN32
	preloadLibrariesWindows();
#endif
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
