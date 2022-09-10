// Copyright 2022 Ivanov Viktor

#ifdef _MSC_VER
#pragma warning(disable : 26439)
#pragma warning(disable : 26451)
#endif

#include <wx/app.h>
#include <wx/button.h>
#include <wx/frame.h>
#include <wx/listbook.h>
#include <wx/panel.h>
#include <wx/statline.h>
#include <wx/xrc/xmlres.h>
#include <wxrc_resource.h>

class CMainFrame : public CMainFrame_Base {
public:
	CMainFrame() : CMainFrame_Base(nullptr) {
		SetIcon(wxICON(WXICON_AAA));
	}

private:
	// NOLINTNEXTLINE
	void onStartClicked([[maybe_unused]] wxCommandEvent &event) {
		Close();
	}
	wxDECLARE_EVENT_TABLE();
};

// clang-format off
wxBEGIN_EVENT_TABLE(CMainFrame, CMainFrame_Base)
    EVT_BUTTON(XRCID("m_StartButton"), CMainFrame::onStartClicked)
wxEND_EVENT_TABLE();
// clang-format on

class CApp : public wxApp {
public:
	bool OnInit() wxOVERRIDE;
	int OnExit() wxOVERRIDE;
	void OnUnhandledException() wxOVERRIDE;
	bool OnExceptionInMainLoop() wxOVERRIDE;
};

wxIMPLEMENT_APP(CApp);

bool CApp::OnInit() {
	SetAppName(wxT("JoshUpscale"));
	SetAppDisplayName(wxT("JoshUpscale"));
	wxXmlResource::Get()->InitAllHandlers();
	InitXmlResource();
	auto *frame = new CMainFrame();
	frame->Show(true);
	SetTopWindow(frame);
	return true;
}

int CApp::OnExit() {
	return 0;
}

void CApp::OnUnhandledException() {
	try {
		throw;
	} catch (const std::exception &e) {
		wxLogError(e.what());
	}
}

bool CApp::OnExceptionInMainLoop() {
	try {
		throw;
	} catch (const std::exception &e) {
		wxLogError(e.what());
	}
	return false;
}
