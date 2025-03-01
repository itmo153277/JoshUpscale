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
constexpr std::size_t CACHE_SIZE = 16;

class JoshUpscaleFilter : public GenericVideoFilter {
public:
	JoshUpscaleFilter(
	    PClip _child, IScriptEnvironment *env, const char *modelPath);
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

	void resetStream(int n);
};

JoshUpscaleFilter::JoshUpscaleFilter(PClip _child,  // NOLINT
    IScriptEnvironment *env, const char *modelPath)
    : GenericVideoFilter(_child) {
	if (!vi.IsRGB24()) {
		env->ThrowError("JoshUpscale: only RGB24 format is supported");
	}
	try {
		m_Runtime.reset(core::createRuntime(0, modelPath));
	} catch (...) {
		auto exception = core::getExceptionString();
		env->ThrowError("JoshUpscale: %s", exception.c_str());
	}
	if (vi.width != static_cast<int>(m_Runtime->getInputWidth()) ||
	    vi.height != static_cast<int>(m_Runtime->getInputHeight())) {
		env->ThrowError("JoshUpscale: unsupported video size");
	}
	env->CheckVersion(8);
	vi.width = static_cast<int>(m_Runtime->getOuputWidth());
	vi.height = static_cast<int>(m_Runtime->getOuputHeight());
	m_Cache.reserve(CACHE_SIZE);
}

JoshUpscaleFilter::~JoshUpscaleFilter() {
}

void JoshUpscaleFilter::resetStream(int n) {
	std::clog << "[JoshUpscaleAvisynth] WARNING: Resetting stream\n";
	m_NextFrame = n - MAX_BACKTRACK_SIZE;
	m_Cache.clear();
	m_CacheShift = 0;
	m_DontCache = MAX_BACKTRACK_SIZE;
}

PVideoFrame __stdcall JoshUpscaleFilter::GetFrame(
    int n, IScriptEnvironment *env) {
	if (n < m_NextFrame) {
		auto offset = static_cast<std::size_t>(m_NextFrame - n);
		if (offset <= m_Cache.size()) {
			return m_Cache[(m_Cache.size() - offset + m_CacheShift) %
			               CACHE_SIZE];
		}
		resetStream(n);
	}
	if (n > m_NextFrame) {
		if (m_NextFrame + MAX_BACKTRACK_SIZE < n) {
			resetStream(n);
		}
		if (!m_StopBacktrackWarning) {
			std::clog << "[JoshUpscaleAvisynth] INFO: Backtracking stream from "
			          << m_NextFrame << " to " << n << '\n';
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
	core::DataLocation location{};
	switch (src->GetDevice().GetType()) {
	case DEV_TYPE_CPU:
		location = core::DataLocation::CPU;
		break;
	case DEV_TYPE_CUDA:
		location = core::DataLocation::CUDA;
		break;
	default:
		env->ThrowError("JoshUpscale: unsupported device");
	}

	PVideoFrame dst = env->NewVideoFrameP(vi, &src);
	const auto &srcVi = child->GetVideoInfo();
	core::Image inputImage = {
	    .ptr = const_cast<BYTE *>(
	        src->GetReadPtr() +
	        ((static_cast<std::ptrdiff_t>(srcVi.height) - 1) *
	            src->GetPitch())),
	    .location = location,
	    .stride = -src->GetPitch(),
	    .width = static_cast<std::size_t>(srcVi.width),
	    .height = static_cast<std::size_t>(srcVi.height),
	};
	core::Image outputImage = {
	    .ptr = dst->GetWritePtr() +
	           ((static_cast<std::ptrdiff_t>(vi.height) - 1) * dst->GetPitch()),
	    .location = location,
	    .stride = -dst->GetPitch(),
	    .width = static_cast<std::size_t>(vi.width),
	    .height = static_cast<std::size_t>(vi.height),
	};
	try {
		m_Runtime->processImage(inputImage, outputImage);
	} catch (...) {
		auto exception = core::getExceptionString();
		env->ThrowError("JoshUpscale: %s", exception.c_str());
	}
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
	case CACHE_GET_DEV_TYPE:
	case CACHE_GET_CHILD_DEV_TYPE:
		return DEV_TYPE_CPU | DEV_TYPE_CUDA;
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
	return new JoshUpscaleFilter(clip, env, model);
}

}  // namespace

}  // namespace avisynth

}  // namespace JoshUpscale

const AVS_Linkage *AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char *__stdcall AvisynthPluginInit3(
    IScriptEnvironment *env, const AVS_Linkage *const vectors) {
	AVS_linkage = vectors;
	env->AddFunction("JoshUpscale", "[clip]c[model_path]s",
	    &JoshUpscale::avisynth::CreateFilter, nullptr);
	return "JoshUpscale plugin";
}
