// -*- C++ -*-
//
// generated by wxGlade
//
// Example for compiling a single file project under Linux using g++:
//  g++ MyApp.cpp $(wx-config --libs) $(wx-config --cxxflags) -o MyApp
//
// Example for compiling a multi file project under Linux using g++:
//  g++ main.cpp $(wx-config --libs) $(wx-config --cxxflags) -o MyApp Dialog1.cpp Frame1.cpp
//

#include "ComplexExample.h"

// begin wxGlade: ::extracode
// end wxGlade



PyOgg2_MyFrame::PyOgg2_MyFrame(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style):
    wxFrame(parent, id, title, pos, size, wxDEFAULT_FRAME_STYLE)
{
    // begin wxGlade: PyOgg2_MyFrame::PyOgg2_MyFrame
    SetSize(wxSize(600, 500));
    SetTitle(_("mp3 2 ogg"));
    Mp3_To_Ogg_menubar = new wxMenuBar();
    wxMenu *wxglade_tmp_menu;
    wxglade_tmp_menu = new wxMenu();
    wxglade_tmp_menu->Append(wxID_OPEN, _("&Open"), wxEmptyString);
    Connect(wxID_OPEN, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler(PyOgg2_MyFrame::OnOpen));
    wxglade_tmp_menu->Append(wxID_EXIT, _("&Quit"), wxEmptyString);
    Connect(wxID_EXIT, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler(PyOgg2_MyFrame::OnClose));
    Mp3_To_Ogg_menubar->Append(wxglade_tmp_menu, _("&File"));
    wxglade_tmp_menu = new wxMenu();
    wxglade_tmp_menu->Append(wxID_ABOUT, _("&About"), _("About dialog"));
    Connect(wxID_ABOUT, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler(PyOgg2_MyFrame::OnAboutDialog));
    Mp3_To_Ogg_menubar->Append(wxglade_tmp_menu, _("&Help"));
    SetMenuBar(Mp3_To_Ogg_menubar);
    Mp3_To_Ogg_statusbar = CreateStatusBar(2);
    int Mp3_To_Ogg_statusbar_widths[] = { -2, -1 };
    Mp3_To_Ogg_statusbar->SetStatusWidths(2, Mp3_To_Ogg_statusbar_widths);
    
    // statusbar fields
    const wxString Mp3_To_Ogg_statusbar_fields[] = {
        _("Mp3_To_Ogg_statusbar"),
        wxEmptyString,
    };
    for(int i = 0; i < Mp3_To_Ogg_statusbar->GetFieldsCount(); ++i) {
        Mp3_To_Ogg_statusbar->SetStatusText(Mp3_To_Ogg_statusbar_fields[i], i);
    }
    Mp3_To_Ogg_toolbar = new wxToolBar(this, -1, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL|wxTB_TEXT);
    SetToolBar(Mp3_To_Ogg_toolbar);
    Mp3_To_Ogg_toolbar->AddTool(wxID_OPEN, _("&Open"), wxNullBitmap, wxNullBitmap, wxITEM_NORMAL, _("Open a file"), _("Open a MP3 file to convert into OGG format"));
    Connect(wxID_OPEN, wxEVT_TOOL, PyOgg2_MyFrame::OnOpen);
    Mp3_To_Ogg_toolbar->Realize();
    wxFlexGridSizer* sizer_1 = new wxFlexGridSizer(3, 1, 0, 0);
    notebook_1 = new wxNotebook(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxNB_BOTTOM);
    sizer_1->Add(notebook_1, 1, wxEXPAND, 0);
    notebook_1_pane_1 = new wxPanel(notebook_1, wxID_ANY);
    notebook_1->AddPage(notebook_1_pane_1, _("Input File"));
    wxFlexGridSizer* _gszr_pane1 = new wxFlexGridSizer(1, 3, 0, 0);
    wxStaticText* _lbl_input_filename = new wxStaticText(notebook_1_pane_1, wxID_ANY, _("File name:"));
    _gszr_pane1->Add(_lbl_input_filename, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
    text_ctrl_1 = new wxTextCtrl(notebook_1_pane_1, wxID_ANY, wxEmptyString);
    text_ctrl_1->SetBackgroundColour(wxNullColour);
    _gszr_pane1->Add(text_ctrl_1, 1, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5);
    button_3 = new wxButton(notebook_1_pane_1, wxID_OPEN, wxEmptyString);
    _gszr_pane1->Add(button_3, 0, wxALL, 5);
    notebook_1_pane_2 = new wxPanel(notebook_1, wxID_ANY);
    notebook_1->AddPage(notebook_1_pane_2, _("Converting Options"));
    wxBoxSizer* sizer_4 = new wxBoxSizer(wxHORIZONTAL);
    const wxString rbx_sampling_rate_choices[] = {
        _("44 kbit"),
        _("128 kbit"),
    };
    rbx_sampling_rate = new wxRadioBox(notebook_1_pane_2, wxID_ANY, _("Sampling Rate"), wxDefaultPosition, wxDefaultSize, 2, rbx_sampling_rate_choices, 0, wxRA_SPECIFY_ROWS);
    rbx_sampling_rate->SetSelection(0);
    sizer_4->Add(rbx_sampling_rate, 1, wxALL|wxEXPAND, 5);
    wxStaticBoxSizer* sizer_3 = new wxStaticBoxSizer(new wxStaticBox(notebook_1_pane_2, wxID_ANY, _("Misc")), wxHORIZONTAL);
    sizer_4->Add(sizer_3, 1, wxALL|wxEXPAND, 5);
    cbx_love = new wxCheckBox(notebook_1_pane_2, wxID_ANY, _("♥ Love this song"));
    cbx_love->SetToolTip(_("Yes!\nWe ♥ it!"));
    cbx_love->SetValue(1);
    sizer_3->Add(cbx_love, 0, wxALL|wxSHAPED, 5);
    notebook_1_pane_3 = new wxPanel(notebook_1, wxID_ANY);
    notebook_1->AddPage(notebook_1_pane_3, _("Converting Progress"));
    wxBoxSizer* _szr_pane3 = new wxBoxSizer(wxHORIZONTAL);
    text_ctrl_2 = new wxTextCtrl(notebook_1_pane_3, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE);
    _szr_pane3->Add(text_ctrl_2, 1, wxALL|wxEXPAND, 5);
    notebook_1_pane_4 = new wxPanel(notebook_1, wxID_ANY);
    notebook_1->AddPage(notebook_1_pane_4, _("Output File"));
    wxFlexGridSizer* _gszr_pane4 = new wxFlexGridSizer(2, 3, 0, 0);
    _lbl_output_filename = new wxStaticText(notebook_1_pane_4, wxID_ANY, _("File name:"));
    _gszr_pane4->Add(_lbl_output_filename, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
    text_ctrl_3 = new wxTextCtrl(notebook_1_pane_4, wxID_ANY, wxEmptyString);
    text_ctrl_3->SetToolTip(_("File name of the output file\nAn existing file will be overwritten without futher information!"));
    _gszr_pane4->Add(text_ctrl_3, 0, wxALL|wxEXPAND, 5);
    button_4 = new wxButton(notebook_1_pane_4, wxID_OPEN, wxEmptyString);
    _gszr_pane4->Add(button_4, 0, wxALL, 5);
    _gszr_pane4->Add(20, 20, 0, 0, 0);
    checkbox_1 = new wxCheckBox(notebook_1_pane_4, wxID_ANY, _("Overwrite existing file"));
    checkbox_1->SetToolTip(_("Overwrite an existing file"));
    checkbox_1->SetValue(1);
    _gszr_pane4->Add(checkbox_1, 0, wxALL|wxEXPAND, 5);
    _gszr_pane4->Add(20, 20, 0, 0, 0);
    notebook_1_pane_5 = new wxPanel(notebook_1, wxID_ANY);
    notebook_1->AddPage(notebook_1_pane_5, _("Some Text"));
    wxBoxSizer* sizer_5 = new wxBoxSizer(wxHORIZONTAL);
    label_1 = new wxStaticText(notebook_1_pane_5, wxID_ANY, _("Please check the format of those lines manually:\n\nSingle line without any special characters.\n\na line break between new and line: new\nline\na tab character between new and line: new\tline\ntwo backslash characters: \\\\ \nthree backslash characters: \\\\\\ \na double quote: \"\nan escaped new line sequence: \\n"));
    sizer_5->Add(label_1, 1, wxALL|wxEXPAND, 5);
    static_line_1 = new wxStaticLine(this, wxID_ANY);
    sizer_1->Add(static_line_1, 0, wxALL|wxEXPAND, 5);
    wxFlexGridSizer* sizer_2 = new wxFlexGridSizer(1, 3, 0, 0);
    sizer_1->Add(sizer_2, 0, wxALIGN_RIGHT, 0);
    button_5 = new wxButton(this, wxID_CLOSE, wxEmptyString);
    sizer_2->Add(button_5, 0, wxALIGN_RIGHT|wxALL, 5);
    button_2 = new wxButton(this, wxID_CANCEL, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxBU_TOP);
    sizer_2->Add(button_2, 0, wxALIGN_RIGHT|wxALL, 5);
    button_1 = new wxButton(this, wxID_OK, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxBU_TOP);
    sizer_2->Add(button_1, 0, wxALIGN_RIGHT|wxALL, 5);
    
    notebook_1_pane_5->SetSizer(sizer_5);
    _gszr_pane4->AddGrowableCol(1);
    notebook_1_pane_4->SetSizer(_gszr_pane4);
    notebook_1_pane_3->SetSizer(_szr_pane3);
    notebook_1_pane_2->SetSizer(sizer_4);
    _gszr_pane1->AddGrowableCol(1);
    notebook_1_pane_1->SetSizer(_gszr_pane1);
    sizer_1->AddGrowableRow(0);
    sizer_1->AddGrowableCol(0);
    SetSizer(sizer_1);
    sizer_1->SetSizeHints(this);
    Layout();
    Centre();
    // end wxGlade
}


BEGIN_EVENT_TABLE(PyOgg2_MyFrame, wxFrame)
    // begin wxGlade: PyOgg2_MyFrame::event_table
    EVT_BUTTON(wxID_OK, PyOgg2_MyFrame::startConverting)
    // end wxGlade
END_EVENT_TABLE();


void PyOgg2_MyFrame::OnOpen(wxCommandEvent &event)  // wxGlade: PyOgg2_MyFrame.<event_handler>
{
    event.Skip();
    // notify the user that he hasn't implemented the event handler yet
    wxLogDebug(wxT("Event handler (PyOgg2_MyFrame::OnOpen) not implemented yet"));
}

void PyOgg2_MyFrame::OnClose(wxCommandEvent &event)  // wxGlade: PyOgg2_MyFrame.<event_handler>
{
    event.Skip();
    // notify the user that he hasn't implemented the event handler yet
    wxLogDebug(wxT("Event handler (PyOgg2_MyFrame::OnClose) not implemented yet"));
}

void PyOgg2_MyFrame::OnAboutDialog(wxCommandEvent &event)  // wxGlade: PyOgg2_MyFrame.<event_handler>
{
    event.Skip();
    // notify the user that he hasn't implemented the event handler yet
    wxLogDebug(wxT("Event handler (PyOgg2_MyFrame::OnAboutDialog) not implemented yet"));
}

void PyOgg2_MyFrame::startConverting(wxCommandEvent &event)  // wxGlade: PyOgg2_MyFrame.<event_handler>
{
    event.Skip();
    // notify the user that he hasn't implemented the event handler yet
    wxLogDebug(wxT("Event handler (PyOgg2_MyFrame::startConverting) not implemented yet"));
}


// wxGlade: add PyOgg2_MyFrame event handlers


MyFrameGrid::MyFrameGrid(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style):
    wxFrame(parent, id, title, pos, size, wxDEFAULT_FRAME_STYLE)
{
    // begin wxGlade: MyFrameGrid::MyFrameGrid
    SetSize(wxSize(492, 300));
    SetTitle(_("FrameOggCompressionDetails"));
    _szr_frame = new wxBoxSizer(wxVERTICAL);
    grid_sizer = new wxFlexGridSizer(3, 1, 0, 0);
    _szr_frame->Add(grid_sizer, 1, wxEXPAND, 0);
    grid = new wxGrid(this, wxID_ANY);
    grid->CreateGrid(8, 3);
    grid_sizer->Add(grid, 1, wxEXPAND, 0);
    static_line = new wxStaticLine(this, wxID_ANY);
    grid_sizer->Add(static_line, 0, wxALL|wxEXPAND, 5);
    button = new wxButton(this, wxID_CLOSE, wxEmptyString);
    button->SetFocus();
    button->SetDefault();
    grid_sizer->Add(button, 0, wxALIGN_RIGHT|wxALL, 5);
    
    grid_sizer->AddGrowableRow(0);
    grid_sizer->AddGrowableCol(0);
    SetSizer(_szr_frame);
    _szr_frame->SetSizeHints(this);
    Layout();
    // end wxGlade
}


class MyApp: public wxApp {
public:
    bool OnInit();
protected:
    wxLocale m_locale;  // locale we'll be using
};

IMPLEMENT_APP(MyApp)

bool MyApp::OnInit()
{
    m_locale.Init();
#ifdef APP_LOCALE_DIR
    m_locale.AddCatalogLookupPathPrefix(wxT(APP_LOCALE_DIR));
#endif
    m_locale.AddCatalog(wxT(APP_CATALOG));

    wxInitAllImageHandlers();
    MyFrameGrid* FrameGrid = new MyFrameGrid(NULL, wxID_ANY, wxEmptyString);
    SetTopWindow(FrameGrid);
    FrameGrid->Show();
    return true;
}