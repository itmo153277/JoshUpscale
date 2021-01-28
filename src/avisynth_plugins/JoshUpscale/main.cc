#define WIN32_LEAN_AND_MEAN

#include <memory>
#include <windows.h>
#include <avisynth.h>
#include "..\..\gui\upscaler.h"

class JoshUpscale : public GenericVideoFilter {
	upscaler::SUpscaler upscaler;

public:
	JoshUpscale(PClip _child, IScriptEnvironment *env);
	~JoshUpscale();
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env);
};

JoshUpscale::JoshUpscale(PClip _child, IScriptEnvironment *env)
    : GenericVideoFilter(_child) {
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
