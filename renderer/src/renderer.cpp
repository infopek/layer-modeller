#include <renderer.h>

#include <cgal_to_vtk_converter.h>

Renderer::Renderer()
{
    m_renderer = vtkSmartPointer<vtkRenderer>::New();
    m_renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    m_renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();

    init();
}

Renderer::~Renderer()
{
}

void Renderer::init()
{
    // Renderer
    m_renderer->SetBackground(render::black.rgb);

    // Render window
    m_renderWindow->SetSize(render::windowWidth, render::windowHeight);
    m_renderWindow->AddRenderer(m_renderer);

    // Interactor
    m_renderWindowInteractor->SetRenderWindow(m_renderWindow);
}

void Renderer::prepareConnectionMeshes()
{
    for (size_t i = 0; i < m_triangulations.size() - 1; ++i)
    {
        vtkSmartPointer<vtkPolyData> connectionMesh = generateConnectionMesh(m_triangulations[i], m_triangulations[i + 1]);

        // Add connection mesh to renderer
        vtkSmartPointer<vtkPolyDataMapper> connectionMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        connectionMapper->SetInputData(connectionMesh);

        vtkSmartPointer<vtkActor> connectionActor = vtkSmartPointer<vtkActor>::New();
        connectionActor->SetMapper(connectionMapper);
        connectionActor->GetProperty()->SetColor(render::brown.rgb);

        m_renderer->AddActor(connectionActor);
    }
}

vtkSmartPointer<vtkPolyData> Renderer::generateConnectionMesh(const Triangulation& layer1, const Triangulation& layer2)
{
    const auto& points1 = layer1.getPoints();
    const auto& points2 = layer2.getPoints();

    vtkSmartPointer<vtkPoints> connectionPoints = vtkSmartPointer<vtkPoints>::New();
    for (const auto& point : points1)
        connectionPoints->InsertNextPoint(point.x, point.y, point.z);

    for (const auto& point : points2)
        connectionPoints->InsertNextPoint(point.x, point.y, point.z);

    vtkSmartPointer<vtkCellArray> connectionCells = vtkSmartPointer<vtkCellArray>::New();

    for (vtkIdType i = 0; i < points1.size() - 1; ++i)
    {
        vtkSmartPointer<vtkTriangle> triangle1 = vtkSmartPointer<vtkTriangle>::New();
        triangle1->GetPointIds()->SetId(0, i);
        triangle1->GetPointIds()->SetId(1, i + 1);
        triangle1->GetPointIds()->SetId(2, i + points1.size());

        vtkSmartPointer<vtkTriangle> triangle2 = vtkSmartPointer<vtkTriangle>::New();
        triangle2->GetPointIds()->SetId(0, i + 1);
        triangle2->GetPointIds()->SetId(1, i + points1.size() + 1);
        triangle2->GetPointIds()->SetId(2, i + points1.size());

        connectionCells->InsertNextCell(triangle1);
        connectionCells->InsertNextCell(triangle2);
    }

    vtkSmartPointer<vtkPolyData> connectionPolyData = vtkSmartPointer<vtkPolyData>::New();
    connectionPolyData->SetPoints(connectionPoints);
    connectionPolyData->SetPolys(connectionCells);

    return connectionPolyData;
}

void Renderer::addLayers(const std::vector<Triangulation>& layers)
{
    const size_t numLayers = layers.size();
    m_triangulations.reserve(numLayers);
    m_triangulationGrids.reserve(numLayers);
    for (const auto& layer : layers)
    {
        auto grid = CGALToVTKConverter::gridifyTriangulation(layer);
        m_triangulations.push_back(layer);
        m_triangulationGrids.push_back(grid);
    }
}

void Renderer::prepareTriangulations()
{
    vtkSmartPointer<vtkAppendFilter> appendFilter = vtkSmartPointer<vtkAppendFilter>::New();
    for (const auto& grid : m_triangulationGrids)
        appendFilter->AddInputData(grid);
    appendFilter->Update();

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(appendFilter->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    m_renderer->AddActor(actor);
}

void Renderer::prepareEdges()
{
    for (const auto grid : m_triangulationGrids)
    {
        vtkSmartPointer<vtkExtractEdges> extractEdges = vtkSmartPointer<vtkExtractEdges>::New();
        extractEdges->SetInputData(grid);
        extractEdges->Update();

        vtkSmartPointer<vtkPolyData> edges = extractEdges->GetOutput();
        vtkSmartPointer<vtkDataSetMapper> edgeMapper = vtkSmartPointer<vtkDataSetMapper>::New();
        edgeMapper->SetInputData(edges);

        vtkSmartPointer<vtkActor> edgeActor = vtkSmartPointer<vtkActor>::New();
        edgeActor->SetMapper(edgeMapper);
        edgeActor->GetProperty()->SetColor(render::red.rgb);

        m_renderer->AddActor(edgeActor);
    }
}

void Renderer::preparePoints()
{
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();

    vtkSmartPointer<vtkPoints> pts = vtkSmartPointer<vtkPoints>::New();

    const auto& points1 = m_triangulations[1].getPoints();
    for (const auto& point : points1)
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
    actor->GetProperty()->SetColor(render::green.rgb);

    m_renderer->AddActor(actor);
}

void Renderer::prepareCoordinateSystem()
{
    vtkSmartPointer<vtkLineSource> xLineSource = vtkSmartPointer<vtkLineSource>::New();
    xLineSource->SetPoint1(0, 0, 0);
    xLineSource->SetPoint2(10, 0, 0);

    vtkSmartPointer<vtkPolyDataMapper> xMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    xMapper->SetInputConnection(xLineSource->GetOutputPort());

    vtkSmartPointer<vtkActor> xActor = vtkSmartPointer<vtkActor>::New();
    xActor->SetMapper(xMapper);
    xActor->GetProperty()->SetColor(render::red.rgb);

    vtkSmartPointer<vtkLineSource> yLineSource = vtkSmartPointer<vtkLineSource>::New();
    yLineSource->SetPoint1(0, 0, 0);
    yLineSource->SetPoint2(0, 10, 0);

    vtkSmartPointer<vtkPolyDataMapper> yMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    yMapper->SetInputConnection(yLineSource->GetOutputPort());

    vtkSmartPointer<vtkActor> yActor = vtkSmartPointer<vtkActor>::New();
    yActor->SetMapper(yMapper);
    yActor->GetProperty()->SetColor(render::green.rgb);

    vtkSmartPointer<vtkLineSource> zLineSource = vtkSmartPointer<vtkLineSource>::New();
    zLineSource->SetPoint1(0, 0, 0);
    zLineSource->SetPoint2(0, 0, 10);

    vtkSmartPointer<vtkPolyDataMapper> zMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    zMapper->SetInputConnection(zLineSource->GetOutputPort());

    vtkSmartPointer<vtkActor> zActor = vtkSmartPointer<vtkActor>::New();
    zActor->SetMapper(zMapper);
    zActor->GetProperty()->SetColor(render::blue.rgb);

    m_renderer->AddActor(xActor);
    m_renderer->AddActor(yActor);
    m_renderer->AddActor(zActor);
}

void Renderer::render()
{
    m_renderWindow->Render();
    m_renderWindowInteractor->Start();
}
