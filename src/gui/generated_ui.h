///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Oct 26 2018)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#pragma once

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
#include <wx/string.h>
#include <wx/stattext.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/choice.h>
#include <wx/checkbox.h>
#include <wx/sizer.h>
#include <wx/statbox.h>
#include <wx/statline.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/button.h>
#include <wx/panel.h>
#include <wx/frame.h>

///////////////////////////////////////////////////////////////////////////

namespace generated
{

	///////////////////////////////////////////////////////////////////////////////
	/// Class MainFrame
	///////////////////////////////////////////////////////////////////////////////
	class MainFrame : public wxFrame
	{
		DECLARE_EVENT_TABLE()
		private:

			// Private event handlers
			void _wxFB_onVideoSelected( wxCommandEvent& event ){ onVideoSelected( event ); }
			void _wxFB_onGoClicked( wxCommandEvent& event ){ onGoClicked( event ); }


		protected:
			wxChoice* m_VideoInChoice;
			wxChoice* m_AudioInChoice;
			wxChoice* m_AudioOutChoice;
			wxChoice* m_DxvaChoice;
			wxCheckBox* m_DebugEnable;
			wxButton* m_GoBtn;

			// Virtual event handlers, overide them in your derived class
			virtual void onVideoSelected( wxCommandEvent& event ) { event.Skip(); }
			virtual void onGoClicked( wxCommandEvent& event ) { event.Skip(); }


		public:

			MainFrame( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("JoshUpscale"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 520,310 ), long style = wxCAPTION|wxCLOSE_BOX|wxMINIMIZE_BOX|wxSYSTEM_MENU|wxTAB_TRAVERSAL );

			~MainFrame();

	};

} // namespace generated

