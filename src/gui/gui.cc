#include "gui.h"
#include <cassert>

void CMainFrame::onVideoSelected(wxCommandEvent &event) {
	m_GoBtn->Enable(m_VideoInChoice->GetSelection() > 0);
}

void CMainFrame::onGoClicked(wxCommandEvent &event) {
	static const processor::DXVA dxvaValues[] = {
	    processor::DXVA::AUTO,
	    processor::DXVA::FORCED,
	    processor::DXVA::OFF,
	};
	assert(
	    m_DxvaChoice->GetSelection() >= 0 && m_DxvaChoice->GetSelection() <= 2);
	processor::DXVA dxva = dxvaValues[m_DxvaChoice->GetSelection()];
	assert(m_VideoInChoice->GetSelection() > 0 &&
	       m_VideoInChoice->GetSelection() <= m_VideoInDeviceList.size());
	const char *videoIn =
	    m_VideoInDeviceList[static_cast<std::size_t>(
	                            m_VideoInChoice->GetSelection()) -
	                        1]
	        .deviceId.c_str();
	const char *audioIn = nullptr;
	if (m_AudioInChoice->GetSelection() > 0) {
		assert(m_AudioInChoice->GetSelection() > 0 &&
		       m_AudioInChoice->GetSelection() <= m_AudioInDeviceList.size());
		audioIn = m_AudioInDeviceList[static_cast<std::size_t>(
		                                  m_AudioInChoice->GetSelection()) -
		                              1]
		              .deviceId.c_str();
	}
	const char *audioOut = nullptr;
	if (m_AudioOutChoice->GetSelection() > 0) {
		assert(m_AudioOutChoice->GetSelection() > 0 &&
		       m_AudioOutChoice->GetSelection() <= m_AudioOutDeviceList.size());
		audioOut = m_AudioOutDeviceList[static_cast<std::size_t>(
		                                    m_AudioOutChoice->GetSelection()) -
		                                1]
		               .deviceId.c_str();
	}
	Show(false);
	processor::processAndShowVideo(
	    videoIn, audioIn, audioOut, dxva, [this] { wxYield(); });
	Close();
}

CMainFrame::CMainFrame() : generated::MainFrame(nullptr) {
	SetIcon(wxICON(MainIcon));

	m_VideoInDeviceList = processor::getVideoInDevices();
	for (auto &device : m_VideoInDeviceList) {
		m_VideoInChoice->AppendString(device.deviceName);
	}
	m_AudioInDeviceList = processor::getAudioInDevices();
	for (auto &device : m_AudioInDeviceList) {
		m_AudioInChoice->AppendString(device.deviceName);
	}
	m_AudioOutDeviceList = processor::getAudioOutDevices();
	for (auto &device : m_AudioOutDeviceList) {
		m_AudioOutChoice->AppendString(device.deviceName);
	}
}

CMainFrame::~CMainFrame() {
}
