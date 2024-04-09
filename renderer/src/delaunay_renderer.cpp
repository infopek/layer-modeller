#include <delaunay_renderer.h>

#include <cgal_to_vtk_converter.h>

DelaunayRenderer::DelaunayRenderer(const Triangulation& triangulation)
    : m_triangulation{ triangulation }
{
    m_renderer = vtkSmartPointer<vtkRenderer>::New();
    m_renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    m_renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();

    init();
}

DelaunayRenderer::~DelaunayRenderer()
{
}

void DelaunayRenderer::init()
{
    // Renderer
    m_renderer->SetBackground(render::black.r, render::black.g, render::black.b);

    // Render window
    m_renderWindow->SetSize(render::windowWidth, render::windowHeight);
    m_renderWindow->AddRenderer(m_renderer);

    // Interactor
    m_renderWindowInteractor->SetRenderWindow(m_renderWindow);

    // Process triangulation
    m_triangulationGrid = CGALToVTKConverter::gridifyTriangulation(m_triangulation);
}

void DelaunayRenderer::addTriangulation()
{
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(m_triangulationGrid);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    m_renderer->AddActor(actor);
}

void DelaunayRenderer::addEdges()
{
    vtkSmartPointer<vtkExtractEdges> extractEdges = vtkSmartPointer<vtkExtractEdges>::New();
    extractEdges->SetInputData(m_triangulationGrid);
    extractEdges->Update();

    vtkSmartPointer<vtkPolyData> edges = extractEdges->GetOutput();
    vtkSmartPointer<vtkDataSetMapper> edgeMapper = vtkSmartPointer<vtkDataSetMapper>::New();
    edgeMapper->SetInputData(edges);

    vtkSmartPointer<vtkActor> edgeActor = vtkSmartPointer<vtkActor>::New();
    edgeActor->SetMapper(edgeMapper);
    edgeActor->GetProperty()->SetColor(render::red.r, render::red.g, render::red.b);

    m_renderer->AddActor(edgeActor);
}

void DelaunayRenderer::addPoints()
{
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();

    vtkSmartPointer<vtkPoints> pts = vtkSmartPointer<vtkPoints>::New();

    const auto& points = m_triangulation.getPoints();
    for (const auto& point : points)
    {
        pts->InsertNextPoint(point.x, point.y, point.z);
    }

    polyData->SetPoints(pts);

    vtkSmartPointer<vtkSphereSource> sphereSource = vtkSmartPointer<vtkSphereSource>::New();
    sphereSource->SetRadius(0.5);

    vtkSmartPointer<vtkGlyph3D> glyphFilter = vtkSmartPointer<vtkGlyph3D>::New();
    glyphFilter->SetInputData(polyData);
    glyphFilter->SetSourceConnection(sphereSource->GetOutputPort());

    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(glyphFilter->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(render::green.r, render::green.g, render::green.b);

    m_renderer->AddActor(actor);
}

void DelaunayRenderer::addCoordinateSystem()
{
    vtkSmartPointer<vtkLineSource> xLineSource = vtkSmartPointer<vtkLineSource>::New();
    xLineSource->SetPoint1(0, 0, 0);
    xLineSource->SetPoint2(10, 0, 0);

    vtkSmartPointer<vtkPolyDataMapper> xMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    xMapper->SetInputConnection(xLineSource->GetOutputPort());

    vtkSmartPointer<vtkActor> xActor = vtkSmartPointer<vtkActor>::New();
    xActor->SetMapper(xMapper);
    xActor->GetProperty()->SetColor(render::red.r, render::red.g, render::red.b);

    vtkSmartPointer<vtkLineSource> yLineSource = vtkSmartPointer<vtkLineSource>::New();
    yLineSource->SetPoint1(0, 0, 0);
    yLineSource->SetPoint2(0, 10, 0);

    vtkSmartPointer<vtkPolyDataMapper> yMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    yMapper->SetInputConnection(yLineSource->GetOutputPort());

    vtkSmartPointer<vtkActor> yActor = vtkSmartPointer<vtkActor>::New();
    yActor->SetMapper(yMapper);
    yActor->GetProperty()->SetColor(render::green.r, render::green.g, render::green.b);

    vtkSmartPointer<vtkLineSource> zLineSource = vtkSmartPointer<vtkLineSource>::New();
    zLineSource->SetPoint1(0, 0, 0);
    zLineSource->SetPoint2(0, 0, 10);

    vtkSmartPointer<vtkPolyDataMapper> zMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    zMapper->SetInputConnection(zLineSource->GetOutputPort());

    vtkSmartPointer<vtkActor> zActor = vtkSmartPointer<vtkActor>::New();
    zActor->SetMapper(zMapper);
    zActor->GetProperty()->SetColor(render::blue.r, render::blue.g, render::blue.b);

    m_renderer->AddActor(xActor);
    m_renderer->AddActor(yActor);
    m_renderer->AddActor(zActor);
}

void DelaunayRenderer::render()
{
    m_renderWindow->Render();
    m_renderWindowInteractor->Start();
}
