///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Oct 26 2018)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "generated_ui.h"

///////////////////////////////////////////////////////////////////////////
using namespace generated;

BEGIN_EVENT_TABLE( MainFrame, wxFrame )
	EVT_CHOICE( wxID_ANY, MainFrame::_wxFB_onVideoSelected )
	EVT_BUTTON( wxID_ANY, MainFrame::_wxFB_onGoClicked )
END_EVENT_TABLE()

MainFrame::MainFrame( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	this->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_BTNFACE ) );

	wxBoxSizer* bSizer1;
	bSizer1 = new wxBoxSizer( wxVERTICAL );

	wxPanel* m_panel1;
	m_panel1 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer2;
	bSizer2 = new wxBoxSizer( wxVERTICAL );

	wxStaticBoxSizer* sbSizer1;
	sbSizer1 = new wxStaticBoxSizer( new wxStaticBox( m_panel1, wxID_ANY, wxT("Configuration") ), wxVERTICAL );

	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->AddGrowableCol( 1 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );

	wxStaticText* m_staticText1;
	m_staticText1 = new wxStaticText( sbSizer1->GetStaticBox(), wxID_ANY, wxT("Video Source Device"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText1->Wrap( -1 );
	fgSizer1->Add( m_staticText1, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	wxString m_VideoInChoiceChoices[] = { wxT("<--- Select Videe Device -->"), wxT("Open file...") };
	int m_VideoInChoiceNChoices = sizeof( m_VideoInChoiceChoices ) / sizeof( wxString );
	m_VideoInChoice = new wxChoice( sbSizer1->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, m_VideoInChoiceNChoices, m_VideoInChoiceChoices, 0 );
	m_VideoInChoice->SetSelection( 0 );
	fgSizer1->Add( m_VideoInChoice, 1, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );

	wxStaticText* m_staticText2;
	m_staticText2 = new wxStaticText( sbSizer1->GetStaticBox(), wxID_ANY, wxT("Audio Source Device"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText2->Wrap( -1 );
	fgSizer1->Add( m_staticText2, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	wxString m_AudioInChoiceChoices[] = { wxT("No Input Audio") };
	int m_AudioInChoiceNChoices = sizeof( m_AudioInChoiceChoices ) / sizeof( wxString );
	m_AudioInChoice = new wxChoice( sbSizer1->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, m_AudioInChoiceNChoices, m_AudioInChoiceChoices, 0 );
	m_AudioInChoice->SetSelection( 0 );
	m_AudioInChoice->Enable( false );

	fgSizer1->Add( m_AudioInChoice, 1, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );

	wxStaticText* m_staticText4;
	m_staticText4 = new wxStaticText( sbSizer1->GetStaticBox(), wxID_ANY, wxT("Audio Output Device"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText4->Wrap( -1 );
	fgSizer1->Add( m_staticText4, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	wxString m_AudioOutChoiceChoices[] = { wxT("System Default") };
	int m_AudioOutChoiceNChoices = sizeof( m_AudioOutChoiceChoices ) / sizeof( wxString );
	m_AudioOutChoice = new wxChoice( sbSizer1->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, m_AudioOutChoiceNChoices, m_AudioOutChoiceChoices, 0 );
	m_AudioOutChoice->SetSelection( 0 );
	fgSizer1->Add( m_AudioOutChoice, 1, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );

	wxStaticText* m_staticText3;
	m_staticText3 = new wxStaticText( sbSizer1->GetStaticBox(), wxID_ANY, wxT("DXVA Decoding"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText3->Wrap( -1 );
	fgSizer1->Add( m_staticText3, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	wxString m_DxvaChoiceChoices[] = { wxT("Auto"), wxT("Forced"), wxT("Off") };
	int m_DxvaChoiceNChoices = sizeof( m_DxvaChoiceChoices ) / sizeof( wxString );
	m_DxvaChoice = new wxChoice( sbSizer1->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, m_DxvaChoiceNChoices, m_DxvaChoiceChoices, 0 );
	m_DxvaChoice->SetSelection( 0 );
	fgSizer1->Add( m_DxvaChoice, 1, wxALL|wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );

	wxStaticText* m_staticText5;
	m_staticText5 = new wxStaticText( sbSizer1->GetStaticBox(), wxID_ANY, wxT("Model"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText5->Wrap( -1 );
	fgSizer1->Add( m_staticText5, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	wxChoice* m_ModelChoice;
	wxString m_ModelChoiceChoices[] = { wxT("Slim") };
	int m_ModelChoiceNChoices = sizeof( m_ModelChoiceChoices ) / sizeof( wxString );
	m_ModelChoice = new wxChoice( sbSizer1->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize, m_ModelChoiceNChoices, m_ModelChoiceChoices, 0 );
	m_ModelChoice->SetSelection( 0 );
	m_ModelChoice->Enable( false );

	fgSizer1->Add( m_ModelChoice, 1, wxALL|wxEXPAND|wxALIGN_CENTER_VERTICAL, 5 );

	wxStaticText* m_staticText6;
	m_staticText6 = new wxStaticText( sbSizer1->GetStaticBox(), wxID_ANY, wxT("Debug Info"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText6->Wrap( -1 );
	fgSizer1->Add( m_staticText6, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

	m_DebugEnable = new wxCheckBox( sbSizer1->GetStaticBox(), wxID_ANY, wxT("Enable"), wxDefaultPosition, wxDefaultSize, 0 );
	m_DebugEnable->SetValue(true);
	fgSizer1->Add( m_DebugEnable, 1, wxALL|wxEXPAND|wxALIGN_CENTER_VERTICAL, 5 );


	sbSizer1->Add( fgSizer1, 1, wxEXPAND|wxALL, 5 );


	bSizer2->Add( sbSizer1, 0, wxEXPAND|wxALL, 5 );

	wxBoxSizer* bSizer3;
	bSizer3 = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer* bSizer4;
	bSizer4 = new wxBoxSizer( wxVERTICAL );

	wxStaticLine* m_staticline1;
	m_staticline1 = new wxStaticLine( m_panel1, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer4->Add( m_staticline1, 0, wxEXPAND, 5 );

	m_GoBtn = new wxButton( m_panel1, wxID_ANY, wxT("Go"), wxDefaultPosition, wxDefaultSize, 0 );

	m_GoBtn->SetDefault();
	m_GoBtn->Enable( false );

	bSizer4->Add( m_GoBtn, 0, wxALL|wxALIGN_RIGHT, 5 );


	bSizer3->Add( bSizer4, 1, wxALIGN_BOTTOM, 5 );


	bSizer2->Add( bSizer3, 1, wxEXPAND, 5 );


	m_panel1->SetSizer( bSizer2 );
	m_panel1->Layout();
	bSizer2->Fit( m_panel1 );
	bSizer1->Add( m_panel1, 1, wxEXPAND, 5 );


	this->SetSizer( bSizer1 );
	this->Layout();

	this->Centre( wxBOTH );
}

MainFrame::~MainFrame()
{
}
