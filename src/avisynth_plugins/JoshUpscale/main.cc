// Copyright 2021 Ivanov Viktor

#define WIN32_LEAN_AND_MEAN

#include <avisynth.h>
#include <windows.h>

#include <memory>
#include <sstream>
#include <string>

#include "..\..\gui\upscaler.h"

class JoshUpscale : public GenericVideoFilter {
	upscaler::SUpscaler upscaler;

	static std::string getModelPath();

public:
	JoshUpscale(PClip _child, IScriptEnvironment *env);
	~JoshUpscale();
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env);
};

std::string JoshUpscale::getModelPath() {
	static const char defaultPath[] = "model.pb";

	HMODULE joshUpscaleModule = GetModuleHandle(L"JoshUpscale");
	if (joshUpscaleModule == NULL) {
		return defaultPath;
	}
	char path[MAX_PATH + 1];
	DWORD size = GetModuleFileNameA(joshUpscaleModule, path, MAX_PATH);
	while (size != 0) {
		size--;
		if (path[size] == '\\') {
			break;
		}
	}
	if (size == 0) {
		return defaultPath;
	}
	path[size] = 0;
	std::stringstream ss;
	ss << path << "\\model.pb";
	return ss.str();
}

JoshUpscale::JoshUpscale(PClip _child, IScriptEnvironment *env)
    : GenericVideoFilter(_child), upscaler{getModelPath().c_str()} {
	if (!vi.IsRGB24()) {
		env->ThrowError("JoshUpscale: only RGB24 format is supported");
	}
	if (vi.width != upscaler::INPUT_WIDTH ||
	    vi.height != upscaler::INPUT_HEIGHT) {
		env->ThrowError("JoshUpscale: unsupported video size");
	}
	vi.width = upscaler::OUTPUT_WIDTH;
	vi.height = upscaler::OUTPUT_HEIGHT;
	child->SetCacheHints(CACHE_ACCESS_SEQ1, 0);
}

JoshUpscale::~JoshUpscale() {
}

PVideoFrame __stdcall JoshUpscale::GetFrame(int n, IScriptEnvironment *env) {
	PVideoFrame src = child->GetFrame(n, env);
	int firstRow = (src->GetHeight() - 1) * src->GetPitch();
	upscaler.upscaleFrame(src->GetReadPtr() + firstRow, -src->GetPitch());
	PVideoFrame dst = env->NewVideoFrame(vi);
	env->MakeWritable(&dst);
	firstRow = (dst->GetHeight() - 1) * dst->GetPitch();
	upscaler.writeOutput(dst->GetWritePtr() + firstRow, -dst->GetPitch());
	return dst;
}

AVSValue __cdecl Create_JoshUpscale(
    AVSValue args, void *user_data, IScriptEnvironment *env) {
	return new JoshUpscale(args[0].AsClip(), env);
}

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char *__stdcall AvisynthPluginInit3(
    IScriptEnvironment *env, const AVS_Linkage *const vectors) {
	AVS_linkage = vectors;
	env->AddFunction("JoshUpscale", "c", Create_JoshUpscale, 0);
	return "JoshUpscale plugin";
}
