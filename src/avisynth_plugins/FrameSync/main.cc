// Copyright 2021 Ivanov Viktor

#define WIN32_LEAN_AND_MEAN

#include <avisynth.h>
#include <windows.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

constexpr int SCALE_FACTOR = 2;

class FrameSync : public GenericVideoFilter {
private:
	PClip clip1;
	PClip clip2;
	std::vector<int> dropped1;
	std::vector<int> dropped2;
	VideoInfo vi1;
	VideoInfo vi2;
	int offset1;
	int offset2;
	std::string out;
	std::string norm_out;
	std::vector<float> norms;

	float getNorm(PVideoFrame frame1, PVideoFrame frame2);

public:
	FrameSync(PClip _child, PClip _clip2, const char *_out, const char *_out2,
	    IScriptEnvironment *env);
	~FrameSync();
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env);
};

FrameSync::FrameSync(PClip _child, PClip _clip2, const char *_out,
    const char *_out2, IScriptEnvironment *env)
    : GenericVideoFilter(_child)
    , clip1(_child)
    , clip2(_clip2)
    , dropped1()
    , dropped2()
    , vi1(clip1->GetVideoInfo())
    , vi2(clip2->GetVideoInfo())
    , offset1()
    , offset2()
    , out(_out)
    , norm_out(_out2)
    , norms() {
	if (!vi1.IsRGB24() || !vi2.IsRGB24()) {
		env->ThrowError("FrameSync: only RGB24 format is supported");
	}
	if (vi1.width != vi2.width || vi1.height != vi2.height) {
		env->ThrowError("FrameSync: clips should have same exact sizes");
	}
	clip1->SetCacheHints(CACHE_ACCESS_SEQ1, 0);
	clip2->SetCacheHints(CACHE_ACCESS_SEQ1, 0);
	vi.num_frames = min(vi1.num_frames, vi2.num_frames);
}

FrameSync::~FrameSync() {
	try {
		std::ofstream f(out.c_str());
		f.exceptions(std::ofstream::badbit | std::ofstream::failbit);
		for (auto frame = dropped1.rbegin(); frame != dropped1.rend();
		     frame++) {
			f << "v1=v1.DeleteFrame(" << *frame << ")" << std::endl;
		}
		for (auto frame = dropped2.rbegin(); frame != dropped2.rend();
		     frame++) {
			f << "v2=v2.DeleteFrame(" << *frame << ")" << std::endl;
		}
		f.close();
	} catch (...) {
	}
	try {
		std::ofstream f(norm_out.c_str());
		f.exceptions(std::ofstream::badbit | std::ofstream::failbit);
		for (auto norm : norms) {
			f << norm << std::endl;
		}
		f.close();
	} catch (...) {
	}
}

PVideoFrame __stdcall FrameSync::GetFrame(int n, IScriptEnvironment *env) {
	int frame_n1 = n + offset1;
	int frame_n2 = n + offset2;
	PVideoFrame frame1 = nullptr;
	PVideoFrame frame2 = nullptr;
	PVideoFrame frame1_1 = nullptr;
	PVideoFrame frame2_1 = nullptr;
	if (frame_n1 < vi1.num_frames) {
		frame1 = clip1->GetFrame(frame_n1, env);
	}
	if (frame_n2 < vi2.num_frames) {
		frame2 = clip2->GetFrame(frame_n2, env);
	}
	if (frame_n1 + 1 < vi1.num_frames) {
		frame1_1 = clip1->GetFrame(frame_n1 + 1, env);
	}
	if (frame_n2 + 1 < vi2.num_frames) {
		frame2_1 = clip2->GetFrame(frame_n2 + 1, env);
	}
	if (frame2 == nullptr || frame1 == nullptr) {
		if (frame1 != nullptr) {
			dropped1.push_back(frame_n1);
			return frame1;
		}
		if (frame2 != nullptr) {
			dropped2.push_back(frame_n2);
			return frame2;
		}
		return env->NewVideoFrame(vi);
	}
	float norm = getNorm(frame1, frame2);
	int drop = 0;
	if (frame1_1 != nullptr) {
		float norm10 = getNorm(frame1_1, frame2);
		if (norm > norm10) {
			drop = 1;
			norm = norm10;
		}
	}
	if (frame2_1 != nullptr) {
		float norm01 = getNorm(frame1, frame2_1);
		if (norm > norm01) {
			drop = 2;
		}
	}
	if (drop == 1) {
		dropped1.push_back(frame_n1);
		offset1++;
	}
	if (drop == 2) {
		dropped2.push_back(frame_n2);
		offset2++;
	}
	if (drop == 0) {
		norms.push_back(norm);
		return frame1;
	}
	return GetFrame(n, env);
}

float FrameSync::getNorm(PVideoFrame frame1, PVideoFrame frame2) {
	double fullNorm = 0;
	std::size_t pitch1 = frame1->GetPitch();
	std::size_t pitch2 = frame2->GetPitch();
	std::size_t height = frame1->GetHeight();
	std::size_t width = frame1->GetRowSize();

	auto readPtr1 = frame1->GetReadPtr();
	auto readPtr2 = frame2->GetReadPtr();

	for (std::size_t h = 0; h < height; ++h) {
		for (std::size_t w = 0; w < width; ++w) {
			double f1 = readPtr1[w];
			double f2 = readPtr2[w];
			double diff = std::abs(f1 - f2);
			fullNorm += diff / 3;
		}
		readPtr1 += pitch1;
		readPtr2 += pitch2;
	}
	return static_cast<float>(fullNorm);
}

AVSValue __cdecl Create_FrameSync(
    AVSValue args, void *user_data, IScriptEnvironment *env) {
	return new FrameSync(args[0].AsClip(), args[1].AsClip(), args[2].AsString(),
	    args[3].AsString(), env);
}

const AVS_Linkage *AVS_linkage = 0;

extern "C" __declspec(dllexport) const char *__stdcall AvisynthPluginInit3(
    IScriptEnvironment *env, const AVS_Linkage *const vectors) {
	AVS_linkage = vectors;
	env->AddFunction("FrameSync", "[clip1]c[clip2]c[out_file]s[out_norm]s",
	    Create_FrameSync, 0);
	return "FrameSync plugin";
}
