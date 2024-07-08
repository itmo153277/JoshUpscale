// Copyright 2022 Ivanov Viktor

#define WIN32_LEAN_AND_MEAN

#include <JoshUpscale/core.h>
#include <avisynth.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

namespace JoshUpscale {

namespace avisynth {

namespace {

constexpr int MAX_BACKTRACK_SIZE = 16;
constexpr std::size_t CACHE_SIZE = 32;

class JoshUpscaleFilter : public GenericVideoFilter {
public:
	JoshUpscaleFilter(PClip _child, IScriptEnvironment *env,
	    const char *modelPath, core::Quantization quantization);
	~JoshUpscaleFilter();
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override;
	int __stdcall SetCacheHints(int cacheHints, int frameRange) override;

private:
	std::unique_ptr<core::Runtime> m_Runtime;
	int m_NextFrame = -MAX_BACKTRACK_SIZE;
	bool m_StopBacktrackWarning = false;
	std::vector<PVideoFrame> m_Cache;
	std::size_t m_CacheShift = 0;
	std::size_t m_DontCache = MAX_BACKTRACK_SIZE;
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
	env->CheckVersion(8);
	vi.width = core::OUTPUT_WIDTH;
	vi.height = core::OUTPUT_HEIGHT;
	m_Runtime.reset(core::createRuntime(0, modelPath, quantization));
	m_Cache.reserve(CACHE_SIZE);
}

JoshUpscaleFilter::~JoshUpscaleFilter() {
}

PVideoFrame __stdcall JoshUpscaleFilter::GetFrame(
    int n, IScriptEnvironment *env) {
	if (n < m_NextFrame) {
		auto offset = static_cast<std::size_t>(m_NextFrame - n);
		if (offset <= m_Cache.size()) {
			return m_Cache[(m_Cache.size() - offset + m_CacheShift) %
			               CACHE_SIZE];
		}
		std::clog << "[JoshUpscaleAvisynth] WARNING: Resetting stream"
		          << std::endl;
		// Redo all frames
		m_NextFrame = n - MAX_BACKTRACK_SIZE;
		m_Cache.clear();
		m_CacheShift = 0;
		m_DontCache = MAX_BACKTRACK_SIZE;
	}
	if (n > m_NextFrame) {
		if (m_NextFrame + MAX_BACKTRACK_SIZE < n) {
			m_NextFrame = n - MAX_BACKTRACK_SIZE;
		}
		if (!m_StopBacktrackWarning) {
			std::clog
			    << "[JoshUpscaleAvisynth] INFO: Backtracking stream from "
			    << m_NextFrame << " to " << n << std::endl;
			m_StopBacktrackWarning = true;
		}
		// Backtrack to our frame
		GetFrame(n - 1, env);
	}
	if (n != m_NextFrame) {
		env->ThrowError(
		    "JoshUpscale: expected frame %d, got %d", m_NextFrame, n);
	}
	m_StopBacktrackWarning = false;
	PVideoFrame src = child->GetFrame(n >= 0 ? n : -n, env);
	PVideoFrame dst = env->NewVideoFrameP(vi, &src);
	core::Image inputImage = {
	    .ptr = const_cast<BYTE *>(
	        src->GetReadPtr() +
	        (static_cast<std::ptrdiff_t>(core::INPUT_HEIGHT) - 1) *
	            src->GetPitch()),
	    .stride = -src->GetPitch(),
	    .width = core::INPUT_WIDTH,
	    .height = core::INPUT_HEIGHT,
	};
	core::Image outputImage = {
	    .ptr = dst->GetWritePtr() +
	           (static_cast<std::ptrdiff_t>(core::OUTPUT_HEIGHT) - 1) *
	               dst->GetPitch(),
	    .stride = -dst->GetPitch(),
	    .width = core::OUTPUT_WIDTH,
	    .height = core::OUTPUT_HEIGHT,
	};
	m_Runtime->processImage(inputImage, outputImage);
	m_NextFrame = n + 1;
	if (m_DontCache > 0) {
		--m_DontCache;
	} else if (m_Cache.size() == CACHE_SIZE) {
		m_Cache[m_CacheShift++] = dst;
		m_CacheShift %= CACHE_SIZE;
	} else {
		m_Cache.push_back(dst);
	}
	return dst;
}

int __stdcall JoshUpscaleFilter::SetCacheHints(
    int cacheHints, [[maybe_unused]] int frameRange) {
	switch (cacheHints) {
	// Deprecated hints
	case CACHE_GETCHILD_COST:
		return CACHE_COST_HI;
	case CACHE_GETCHILD_THREAD_MODE:
		return CACHE_THREAD_CLASS;
	case CACHE_GETCHILD_ACCESS_COST:
		return CACHE_ACCESS_SEQ1;
	// We run on CUDA, but accept and output for CPU
	case CACHE_GET_DEV_TYPE:
	case CACHE_GET_CHILD_DEV_TYPE:
		return DEV_TYPE_CPU;
	// Can't run in parallel
	case CACHE_GET_MTMODE:
		return MT_SERIALIZED;
	default:
		return 0;
	}
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
