#include "app.h"

#include <geotiff_handler.h>
#include <layer_builder.h>
#include <modeller/modeller_set.h>
#include <renderer.h>
#include <blur/blur.h>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <unordered_map>
#include <utility>

#include <fstream>
#include <sstream>

// ss
#include <wx/wx.h>
#include <wx/glcanvas.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkSmartPointer.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkInteractorStyleTrackballCamera.h>

class MyVTKCanvas : public wxGLCanvas {
public:
    MyVTKCanvas(wxWindow* parent)
        : wxGLCanvas(parent, wxID_ANY, nullptr, wxDefaultPosition, wxDefaultSize, wxFULL_REPAINT_ON_RESIZE) {
        context = new wxGLContext(this);
        SetCurrent(*context);

        vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
        renderWindow->SetMultiSamples(0);

        vtkNew<vtkRenderer> renderer;
        renderer->SetBackground(0.1, 0.2, 0.4);

        vtkNew<vtkTextActor> textActor;
        textActor->SetInput("Hello, VTK!");
        textActor->GetTextProperty()->SetFontSize(24);
        textActor->GetTextProperty()->SetColor(1.0, 0.0, 0.0);
        renderer->AddActor(textActor);

        renderWindow->AddRenderer(renderer);
        interactor->SetRenderWindow(renderWindow);
        interactor->SetInteractorStyle(vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New());

        this->renderWindow = renderWindow;
        this->renderer = renderer;
        this->interactor = interactor;

        Bind(wxEVT_PAINT, &MyVTKCanvas::OnPaint, this);
        Bind(wxEVT_SIZE, &MyVTKCanvas::OnResize, this);

        // Render once to initialize
        this->Initialize();
    }

    ~MyVTKCanvas() {
        delete context;
    }

    void OnPaint(wxPaintEvent& event) {
        SetCurrent(*context);
        renderWindow->Render();
    }

    void OnResize(wxSizeEvent& event) {
        int w, h;
        GetClientSize(&w, &h);
        if (renderWindow) {
            renderWindow->SetSize(w, h);
        }
        Refresh();
    }

    void Initialize() {
        SetCurrent(*context);
        renderWindow->Render();
        interactor->Initialize();
    }

private:
    wxGLContext* context;
    vtkSmartPointer<vtkGenericOpenGLRenderWindow> renderWindow;
    vtkSmartPointer<vtkRenderer> renderer;
    vtkSmartPointer<vtkRenderWindowInteractor> interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
};

class MyFrame : public wxFrame {
public:
    MyFrame() : wxFrame(NULL, wxID_ANY, "VTK and wxWidgets Integration") {
        wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);
        MyVTKCanvas* vtkCanvas = new MyVTKCanvas(this);
        sizer->Add(vtkCanvas, 1, wxEXPAND);
        SetSizer(sizer);
        SetSize(800, 600);
    }
};

class MyApp : public wxApp {
public:
    virtual bool OnInit() {
        MyFrame* frame = new MyFrame();
        frame->Show(true);
        return true;
    }
};

wxIMPLEMENT_APP(MyApp);
// int main(int argc, char* argv[])
// {
//     // Create and initialize the wxApp object
//     MyApp app;
//     wxApp::SetInstance(&app);
//     wxEntryStart(argc, argv);
//     int result = wxEntry(argc, argv);
//     wxEntryCleanup();
//     return result;
// }
// int main(int argc, char* argv[])
// {

//     // LayerBuilder layerBuilder("region");

//     // ModellerSet modeller(layerBuilder);
//     // modeller.createMeshes();

//     // Renderer renderer{};
//     // renderer.addMeshes(modeller.getMeshes());

//     // // Describe what you want to be rendered
//     // renderer.prepareEdges();
//     // renderer.prepareSurfaces();
//     // renderer.prepareLayerBody();

//     // // Render
//     // renderer.render();
//     return 0;
// }
