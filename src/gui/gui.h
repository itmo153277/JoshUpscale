#pragma once

#include "generated_ui.h"
#include "videoprocessor.h"

class CMainFrame : public generated::MainFrame {
private:
	processor::DeviceList m_VideoInDeviceList;
	processor::DeviceList m_AudioInDeviceList;
	processor::DeviceList m_AudioOutDeviceList;

protected:
	void onVideoSelected(wxCommandEvent &event) wxOVERRIDE;
	void onGoClicked(wxCommandEvent &event) wxOVERRIDE;

public:
	CMainFrame();
	~CMainFrame();
};
