// Copyright 2021 Ivanov Viktor

#include "tf_wrappers.h"

#include <fstream>

smart::TF_Graph tf::readGraph(const std::string &fileName) {
	std::ifstream file(fileName.c_str(), std::ios::binary | std::ios::ate);
	file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	std::size_t size = file.tellg();
	file.seekg(0, std::ios::beg);
	smart::TF_Buffer buffer(size);
	file.read(buffer.getData(), size);
	smart::TF_Graph graph;
	smart::TF_Status status;
	smart::TF_ImportGraphDefOptions importOptions;
	::TF_ImportGraphDefOptionsSetPrefix(importOptions.get(), "");
	::TF_GraphImportGraphDef(
	    graph.get(), buffer.get(), importOptions.get(), status.get());
	if (::TF_GetCode(status.get()) != TF_OK) {
		throw new tf::TFException(status.get());
	}
	return graph;
}

smart::TF_Session::TF_Session(
    const TF_Graph &graph, const TF_SessionOptionsProto *options, bool xla) {
	smart::TF_SessionOptions sessionOptions;
	if (options != nullptr) {
		smart::TF_Status status;
		::TF_SetConfig(
		    sessionOptions.get(), options->proto, options->size, status.get());
		if (::TF_GetCode(status.get()) != TF_OK) {
			throw new tf::TFException(status.get());
		}
	}
	if (xla) {
		::TF_EnableXLACompilation(sessionOptions.get(), true);
	}
	smart::TF_Status status;
	m_Session =
	    ::TF_NewSession(graph.get(), sessionOptions.get(), status.get());
	if (::TF_GetCode(status.get()) != TF_OK) {
		throw new tf::TFException(status.get());
	}
}

smart::TF_Tensor<float> smart::TF_Session::run(
    const std::vector<::TF_Output> &inputOp,
    const std::vector<::TF_Tensor *> &inputValue, const TF_Output &outputOp) {
	smart::TF_Status status;
	::TF_Tensor *outputValue = nullptr;
	::TF_SessionRun(m_Session, nullptr, inputOp.data(), inputValue.data(),
	    static_cast<int>(inputValue.size()), &outputOp, &outputValue, 1,
	    nullptr, 0, nullptr, status.get());
	if (::TF_GetCode(status.get()) != TF_OK) {
		throw new tf::TFException(status.get());
	}
	return smart::TF_Tensor<float>(outputValue);
}
