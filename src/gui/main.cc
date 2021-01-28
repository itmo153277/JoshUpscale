#include <wx/wxprec.h>

#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

#define SDL_MAIN_HANDLED
#include <SDL.h>

#include "videoprocessor.h"
#include "gui.h"

#include <wx/snglinst.h>
#include <memory>
#include <exception>

class CApp : public wxApp {
private:
	std::unique_ptr<wxSingleInstanceChecker> m_SingleInstanceChecker;

public:
	bool OnInit() wxOVERRIDE;
	int OnExit() wxOVERRIDE;
	void OnUnhandledException() wxOVERRIDE;
	bool OnExceptionInMainLoop() wxOVERRIDE;
};

#ifdef NDEBUG
wxIMPLEMENT_APP(CApp);
#else
wxIMPLEMENT_APP_NO_MAIN(CApp);
#endif

bool CApp::OnInit() {
	SetAppName(wxT("JoshUpscale"));
	SetAppDisplayName(wxT("JoshUpscale"));
	m_SingleInstanceChecker = std::make_unique<wxSingleInstanceChecker>();
	if (m_SingleInstanceChecker->IsAnotherRunning()) {
		m_SingleInstanceChecker.reset();
		wxLogError(wxT("Already running"));
		return false;
	}
	::SDL_SetMainReady();
	processor::init();
	auto frame = new CMainFrame();
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
	} catch (std::exception &e) {
		wxLogError(e.what());
	}
}

bool CApp::OnExceptionInMainLoop() {
	try {
		throw;
	} catch (std::exception &e) {
		wxLogError(e.what());
	}
	return false;
}

#ifndef NDEBUG
int wmain(int argc, wchar_t *argv[]) {
	return wxEntry(argc, argv);
}
#endif
