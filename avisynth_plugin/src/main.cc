// Copyright 2025 Ivanov Viktor

#include <JoshUpscale/core.h>
#include <avisynth.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace JoshUpscale {

namespace avisynth {

namespace {

using Microsoft::WRL::ComPtr;

ComPtr<ID3D12Device> createD3D12Device() {
	ComPtr<IDXGIFactory4> dxgiFactory;
	if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(dxgiFactory.GetAddressOf())))) {
		throw std::runtime_error("Failed to create DXGI factory");
	}
	ComPtr<IDXGIAdapter> adapter;
	if (FAILED(dxgiFactory->EnumAdapters(0, adapter.GetAddressOf()))) {
		throw std::runtime_error("Failed to get DXGI adapter");
	}
	ComPtr<ID3D12Device> device;
	if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
	        IID_PPV_ARGS(device.GetAddressOf())))) {
		throw std::runtime_error("Failed to d3d12 device");
	}
	return device;
}

constexpr int MAX_BACKTRACK_SIZE = 16;
constexpr std::size_t CACHE_SIZE = 16;

int getDeviceTypes(const PClip &child) {
	if (child->GetVersion() < 5) {
		return DEV_TYPE_CPU;
	}
	int types = child->SetCacheHints(CACHE_GET_DEV_TYPE, 0);
	if (types == 0) {
		return DEV_TYPE_CPU;
	}
	return types;
}

class JoshUpscaleFilter : public GenericVideoFilter {
public:
	JoshUpscaleFilter(
	    PClip _child, IScriptEnvironment *env, const char *modelPath);
	~JoshUpscaleFilter();
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env) override;
	int __stdcall SetCacheHints(int cacheHints, int frameRange) override;

private:
	ComPtr<ID3D12Device> m_D3D12Device;
	std::unique_ptr<core::Runtime> m_Runtime;
	int m_NextFrame = -MAX_BACKTRACK_SIZE;
	bool m_StopBacktrackWarning = false;
	std::vector<PVideoFrame> m_Cache;
	std::size_t m_CacheShift = 0;
	std::size_t m_DontCache = MAX_BACKTRACK_SIZE;
	std::vector<double> m_Timings;

	void resetStream(int n);
	void printStats();
};

JoshUpscaleFilter::JoshUpscaleFilter(PClip _child,  // NOLINT
    IScriptEnvironment *env, const char *modelPath)
    : GenericVideoFilter(_child) {
	if (!vi.IsRGB32()) {
		env->ThrowError("JoshUpscale: only RGB32 format is supported");
	}
	m_D3D12Device = createD3D12Device();
	if (FAILED(m_D3D12Device->SetStablePowerState(true))) {
		env->ThrowError(
		    "Failed to set stable power state. Is developer mode enabled?");
	}
	std::clog
	    << "[JoshUpscaleAvisynth] WARNING: GPU is locked to base clocks\n";
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
	vi.width = static_cast<int>(m_Runtime->getOutputWidth());
	vi.height = static_cast<int>(m_Runtime->getOutputHeight());
	m_Cache.reserve(CACHE_SIZE);
}

JoshUpscaleFilter::~JoshUpscaleFilter() {
	if (m_Timings.size() % 300 != 0) {
		printStats();
	}
	std::clog << "[JoshUpscaleAvisynth] INFO: Count: " << m_Timings.size()
	          << '\n';
}

void JoshUpscaleFilter::resetStream(int n) {
	std::clog << "[JoshUpscaleAvisynth] WARNING: Resetting stream\n";
	m_NextFrame = n - MAX_BACKTRACK_SIZE;
	m_Cache.clear();
	m_CacheShift = 0;
	m_DontCache = MAX_BACKTRACK_SIZE;
}

void JoshUpscaleFilter::printStats() {
	if (m_Timings.empty()) {
		return;
	}
	std::size_t p95i =
	    static_cast<std::size_t>(static_cast<double>(m_Timings.size()) * 0.05);
	std::size_t p99i =
	    static_cast<std::size_t>(static_cast<double>(m_Timings.size()) * 0.01);
	std::partial_sort(m_Timings.begin(),
	    m_Timings.begin() + static_cast<std::ptrdiff_t>(p95i + 1),
	    m_Timings.end(), std::greater{});

	auto avg = std::accumulate(m_Timings.begin(), m_Timings.end(), .0) /
	           m_Timings.size();
	auto p95 = m_Timings[p95i];
	auto p99 = m_Timings[p99i];

	std::clog << std::format(
	    "avg {:.6f} (FPS {:.3f})\tp95 {:.6f} (FPS {:.3f})\tp99 {:.6f} (FPS "
	    "{:.3f})\n",
	    avg, 1 / avg, p95, 1 / p95, p99, 1 / p99);
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
	auto startTimestamp = std::chrono::high_resolution_clock::now();
	try {
		m_Runtime->processImage(inputImage, outputImage);
	} catch (...) {
		auto exception = core::getExceptionString();
		env->ThrowError("JoshUpscale: %s", exception.c_str());
	}
	m_Timings.push_back(std::chrono::duration<double>(
	    std::chrono::high_resolution_clock::now() - startTimestamp)
	        .count());
	if (m_Timings.size() % 300 == 0) {
		printStats();
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
	// Support CPU & CUDA
	case CACHE_GET_DEV_TYPE:
	case CACHE_GET_CHILD_DEV_TYPE:
		return getDeviceTypes(child) & (DEV_TYPE_CPU | DEV_TYPE_CUDA);
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
