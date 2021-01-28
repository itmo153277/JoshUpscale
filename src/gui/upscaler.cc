#include "upscaler.h"
#include <cstring>

static const int64_t inputFrameDim[4] = {
    1, upscaler::INPUT_HEIGHT, upscaler::INPUT_WIDTH, 3};
static const int64_t outputFrameDim[4] = {
    1, upscaler::OUTPUT_HEIGHT, upscaler::OUTPUT_WIDTH, 3};
static const unsigned char config[] = {50, 2, 32, 1};
static const smart::TF_SessionOptionsProto sessionOptions = {
    config, sizeof(config) / sizeof(*config)};

upscaler::SUpscaler::SUpscaler()
    : m_Graph{tf::readGraph("model.pb")}
    , m_InputOp{{{::TF_GraphOperationByName(m_Graph.get(), "cur_frame"), 0},
          {::TF_GraphOperationByName(m_Graph.get(), "last_frame"), 0},
          {::TF_GraphOperationByName(m_Graph.get(), "pre_gen"), 0}}}
    , m_OutputOp{TF_GraphOperationByName(m_Graph.get(), "output"), 0}
    , m_Input{inputFrameDim, 4}
    , m_LastFrame{inputFrameDim, 4}
    , m_PreGenTensor{outputFrameDim, 4}
    , m_Session{m_Graph, &sessionOptions, true} {
}

void upscaler::SUpscaler::upscaleFrame(const std::uint8_t *buf, int stride) {
	float *tensorDataPtr = static_cast<float *>(::TF_TensorData(m_Input.get()));
	const std::uint8_t *frameDataPtr = buf;
	for (std::size_t y = 0; y < INPUT_HEIGHT; ++y, frameDataPtr += stride) {
		for (std::size_t x = 0; x < INPUT_WIDTH * 3; ++x, ++tensorDataPtr) {
			*tensorDataPtr = static_cast<float>(frameDataPtr[x]) / 255.f;
		}
	}
	m_PreGenTensor = m_Session.run(m_InputOp,
	    {m_Input.get(), m_LastFrame.get(), m_PreGenTensor.get()}, m_OutputOp);
	memcpy(::TF_TensorData(m_LastFrame.get()), ::TF_TensorData(m_Input.get()),
	    ::TF_TensorByteSize(m_LastFrame.get()));
}

void upscaler::SUpscaler::writeOutput(std::uint8_t *buf, int stride) {
	float *tensorDataPtr =
	    static_cast<float *>(::TF_TensorData(m_PreGenTensor.get()));
	std::uint8_t *frameDataPtr = buf;
	for (std::size_t y = 0; y < OUTPUT_HEIGHT; ++y, frameDataPtr += stride) {
		for (std::size_t x = 0; x < OUTPUT_WIDTH * 3; ++x, ++tensorDataPtr) {
			frameDataPtr[x] = static_cast<std::uint8_t>(
			    std::min(std::max(*tensorDataPtr * 255.f, 0.f), 255.f));
		}
	}
}
