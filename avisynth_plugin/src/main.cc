// Copyright 2022 Ivanov Viktor

#define WIN32_LEAN_AND_MEAN

#include <JoshUpscale/core.h>
#include <avisynth.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <string>

class JoshUpscalePlugin : public GenericVideoFilter {
public:
	JoshUpscalePlugin(
	    PClip _child, IScriptEnvironment *env, const char *modelPath);
	~JoshUpscalePlugin();
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env);

private:
	std::unique_ptr<JoshUpscale::core::Runtime> m_Runtime;
};

JoshUpscalePlugin::JoshUpscalePlugin(PClip _child,  // NOLINT
    IScriptEnvironment *env, const char *modelPath)
    : GenericVideoFilter(_child) {
	if (!vi.IsRGB24()) {
		env->ThrowError("JoshUpscale: only RGB24 format is supported");
	}
	if (vi.width != JoshUpscale::core::INPUT_WIDTH ||
	    vi.height != JoshUpscale::core::INPUT_HEIGHT) {
		env->ThrowError("JoshUpscale: unsupported video size");
	}
	vi.width = JoshUpscale::core::OUTPUT_WIDTH;
	vi.height = JoshUpscale::core::OUTPUT_HEIGHT;
	child->SetCacheHints(CACHE_ACCESS_SEQ1, 0);
	m_Runtime.reset(JoshUpscale::core::createRuntime(0, modelPath));
}

JoshUpscalePlugin::~JoshUpscalePlugin() {
}

PVideoFrame __stdcall JoshUpscalePlugin::GetFrame(
    int n, IScriptEnvironment *env) {
	PVideoFrame src = child->GetFrame(n, env);
	PVideoFrame dst = env->NewVideoFrame(vi);
	env->MakeWritable(&dst);
	JoshUpscale::core::Image inputImage;
	inputImage.width = 480;
	inputImage.height = 270;
	inputImage.stride = -src->GetPitch();
	inputImage.ptr = reinterpret_cast<std::byte *>(const_cast<BYTE *>(
	    src->GetReadPtr() +
	    (static_cast<std::ptrdiff_t>(src->GetHeight()) - 1) * src->GetPitch()));
	JoshUpscale::core::Image outputImage;
	outputImage.width = 1920;
	outputImage.height = 1080;
	outputImage.stride = -dst->GetPitch();
	outputImage.ptr = reinterpret_cast<std::byte *>(
	    dst->GetWritePtr() +
	    (static_cast<std::ptrdiff_t>(dst->GetHeight()) - 1) * dst->GetPitch());
	m_Runtime->processImage(inputImage, outputImage);
	return dst;
}

AVSValue __cdecl Create_JoshUpscale(AVSValue args,  // NOLINT
    [[maybe_unused]] void *user_data, IScriptEnvironment *env) {
	return new JoshUpscalePlugin(args[0].AsClip(), env, args[1].AsString());
}

const AVS_Linkage *AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char *__stdcall AvisynthPluginInit3(
    IScriptEnvironment *env, const AVS_Linkage *const vectors) {
	AVS_linkage = vectors;
	env->AddFunction(
	    "JoshUpscale", "[clip]c[modelPath]s", Create_JoshUpscale, 0);
	return "JoshUpscale plugin";
}
