// Copyright 2022 Ivanov Viktor

#define WIN32_LEAN_AND_MEAN

#include <JoshUpscale/core.h>
#include <avisynth.h>

#include <cassert>
#include <cstddef>
#include <memory>

namespace JoshUpscale {

namespace avisynth {

namespace {

class JoshUpscaleFilter : public GenericVideoFilter {
public:
	JoshUpscaleFilter(PClip _child, IScriptEnvironment *env,
	    const char *modelPath, core::Quantization quantization);
	~JoshUpscaleFilter();
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env);

private:
	std::unique_ptr<core::Runtime> m_Runtime;
};

JoshUpscaleFilter::JoshUpscaleFilter(PClip _child,  // NOLINT
    IScriptEnvironment *env, const char *modelPath,
    core::Quantization quantization)
    : GenericVideoFilter(_child) {
	if (!vi.IsRGB24()) {
		env->ThrowError("JoshUpscale: only RGB24 format is supported");
	}
	if (vi.width != core::INPUT_WIDTH || vi.height != core::INPUT_HEIGHT) {
		env->ThrowError("JoshUpscale: unsupported video size");
	}
	vi.width = core::OUTPUT_WIDTH;
	vi.height = core::OUTPUT_HEIGHT;
	child->SetCacheHints(CACHE_ACCESS_SEQ1, 0);
	m_Runtime.reset(core::createRuntime(0, modelPath, quantization));
}

JoshUpscaleFilter::~JoshUpscaleFilter() {
}

PVideoFrame __stdcall JoshUpscaleFilter::GetFrame(
    int n, IScriptEnvironment *env) {
	PVideoFrame src = child->GetFrame(n, env);
	PVideoFrame dst = env->NewVideoFrame(vi);
	env->MakeWritable(&dst);
	core::Image inputImage;
	inputImage.width = 480;
	inputImage.height = 270;
	inputImage.stride = -src->GetPitch();
	inputImage.ptr = const_cast<BYTE *>(
	    src->GetReadPtr() +
	    (static_cast<std::ptrdiff_t>(src->GetHeight()) - 1) * src->GetPitch());
	core::Image outputImage;
	outputImage.width = 1920;
	outputImage.height = 1080;
	outputImage.stride = -dst->GetPitch();
	outputImage.ptr =
	    dst->GetWritePtr() +
	    (static_cast<std::ptrdiff_t>(dst->GetHeight()) - 1) * dst->GetPitch();
	m_Runtime->processImage(inputImage, outputImage);
	return dst;
}

AVSValue __cdecl CreateFilter(AVSValue args,  // NOLINT
    [[maybe_unused]] void *user_data, IScriptEnvironment *env) {
	PClip clip = args[0].AsClip();
	const char *model = args[1].AsString();
	int quant = args[2].AsInt(static_cast<int>(core::Quantization::FP16));
	if (quant != static_cast<int>(core::Quantization::NONE) &&
	    quant != static_cast<int>(core::Quantization::FP16) &&
	    quant != static_cast<int>(core::Quantization::INT8)) {
		env->ThrowError("JoshUpscale: Invalid quantization value: %d", quant);
	}
	return new JoshUpscaleFilter(
	    clip, env, model, static_cast<core::Quantization>(quant));
}

}  // namespace

}  // namespace avisynth

}  // namespace JoshUpscale

const AVS_Linkage *AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char *__stdcall AvisynthPluginInit3(
    IScriptEnvironment *env, const AVS_Linkage *const vectors) {
	AVS_linkage = vectors;
	env->AddFunction("JoshUpscale", "[clip]c[modelPath]s[quant]i",
	    &JoshUpscale::avisynth::CreateFilter, nullptr);
	return "JoshUpscale plugin";
}
