#include "triangulation_renderer.h"

#include "../cgal_to_vtk_converter.h"

TriangulationRenderer::TriangulationRenderer(const DT2& triangulation, const std::vector<Point>& points3d)
    : m_triangulation{ triangulation },
    m_points{ points3d }
{
    m_renderer = vtkSmartPointer<vtkRenderer>::New();
    m_renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    m_renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();

    init();
}

TriangulationRenderer::~TriangulationRenderer()
{
}

void TriangulationRenderer::init()
{
    // Renderer
    m_renderer->SetBackground(render::black.r, render::black.g, render::black.b);

    // Render window
    m_renderWindow->SetSize(render::windowWidth, render::windowHeight);
    m_renderWindow->AddRenderer(m_renderer);

    m_renderWindowInteractor->SetRenderWindow(m_renderWindow);

    // Process triangulation
    m_triangulationGrid = CGALToVTKConverter::gridifyTriangulation(m_triangulation, m_points);
}

void TriangulationRenderer::addTriangulation()
{
    // Render window setup
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(m_triangulationGrid);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // Edge setup
    vtkSmartPointer<vtkExtractEdges> extractEdges = vtkSmartPointer<vtkExtractEdges>::New();
    extractEdges->SetInputData(m_triangulationGrid);
    extractEdges->Update();

    vtkSmartPointer<vtkPolyData> edges = extractEdges->GetOutput();
    vtkSmartPointer<vtkDataSetMapper> edgeMapper = vtkSmartPointer<vtkDataSetMapper>::New();
    edgeMapper->SetInputData(edges);

    vtkSmartPointer<vtkActor> edgeActor = vtkSmartPointer<vtkActor>::New();
    edgeActor->SetMapper(edgeMapper);
    edgeActor->GetProperty()->SetColor(render::red.r, render::red.g, render::red.b);

    m_renderer->AddActor(actor);
    m_renderer->AddActor(edgeActor);
}

void TriangulationRenderer::addPoints()
{
    // Create a VTK PolyData object
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();

    // Create a VTK Points object to hold the point coordinates
    vtkSmartPointer<vtkPoints> pts = vtkSmartPointer<vtkPoints>::New();

    // Add the points to the VTK Points object
    for (const auto& point : m_points)
    {
        pts->InsertNextPoint(point.x, point.y, point.z);
    }

    // Set the points in the PolyData
    polyData->SetPoints(pts);

    // Create a sphere source for visualization
    vtkSmartPointer<vtkSphereSource> sphereSource = vtkSmartPointer<vtkSphereSource>::New();
    sphereSource->SetRadius(0.5);

    // Create a glyph filter to render spheres at each point
    vtkSmartPointer<vtkGlyph3D> glyphFilter = vtkSmartPointer<vtkGlyph3D>::New();
    glyphFilter->SetInputData(polyData);
    glyphFilter->SetSourceConnection(sphereSource->GetOutputPort());

    // Create a mapper
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(glyphFilter->GetOutputPort());

    // Create an actor
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(0.1, 1.0, 0.1); // Set color to green

    m_renderer->AddActor(actor);
}

void TriangulationRenderer::addCoordinateSystem()
{
    // Create a line source for x-axis
    vtkSmartPointer<vtkLineSource> xLineSource = vtkSmartPointer<vtkLineSource>::New();
    xLineSource->SetPoint1(0, 0, 0);
    xLineSource->SetPoint2(10, 0, 0);

    // Create a line mapper for x-axis
    vtkSmartPointer<vtkPolyDataMapper> xMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    xMapper->SetInputConnection(xLineSource->GetOutputPort());

    // Create an actor for x-axis
    vtkSmartPointer<vtkActor> xActor = vtkSmartPointer<vtkActor>::New();
    xActor->SetMapper(xMapper);
    xActor->GetProperty()->SetColor(1.0, 0.0, 0.0); // Red color for x-axis

    // Create a line source for y-axis
    vtkSmartPointer<vtkLineSource> yLineSource = vtkSmartPointer<vtkLineSource>::New();
    yLineSource->SetPoint1(0, 0, 0);
    yLineSource->SetPoint2(0, 10, 0);

    // Create a line mapper for y-axis
    vtkSmartPointer<vtkPolyDataMapper> yMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    yMapper->SetInputConnection(yLineSource->GetOutputPort());

    // Create an actor for y-axis
    vtkSmartPointer<vtkActor> yActor = vtkSmartPointer<vtkActor>::New();
    yActor->SetMapper(yMapper);
    yActor->GetProperty()->SetColor(0.0, 1.0, 0.0); // Green color for y-axis

    // Create a line source for z-axis
    vtkSmartPointer<vtkLineSource> zLineSource = vtkSmartPointer<vtkLineSource>::New();
    zLineSource->SetPoint1(0, 0, 0);
    zLineSource->SetPoint2(0, 0, 10);

    // Create a line mapper for z-axis
    vtkSmartPointer<vtkPolyDataMapper> zMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    zMapper->SetInputConnection(zLineSource->GetOutputPort());

    // Create an actor for z-axis
    vtkSmartPointer<vtkActor> zActor = vtkSmartPointer<vtkActor>::New();
    zActor->SetMapper(zMapper);
    zActor->GetProperty()->SetColor(0.0, 0.0, 1.0); // Blue color for z-axis

    // Create a renderer
    m_renderer->AddActor(xActor);
    m_renderer->AddActor(yActor);
    m_renderer->AddActor(zActor);
}

void TriangulationRenderer::render()
{
    m_renderWindow->Render();
    m_renderWindowInteractor->Start();
}
