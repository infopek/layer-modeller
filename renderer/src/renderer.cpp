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

void Renderer::addMeshes(const std::vector<Mesh>& meshes)
{
    std::unordered_map<std::string, render::Color> colorMap{
        {"comp1", render::red},
        {"comp2", render::blue},
        {"comp3", render::brown},
        {"comp4", render::green},
        {"comp5", render::purple}, 
        {"comp6", render::orange},
        {"comp7", render::yellow}, 
        {"comp8", render::cyan},    
        {"comp9", render::magenta}, 
        {"comp10", render::lime},   
    };

    m_meshes = meshes;

    const size_t numMeshes = meshes.size();
    m_surfaceMeshPolyData.resize(numMeshes);
    m_layerBodyPolyData.resize(numMeshes);
    m_colors.resize(numMeshes);

    for (size_t i = 0; i < numMeshes; i++)
    {
        const auto& mesh = m_meshes[i];

        auto surfacePolyData = CGALToVTKConverter::convertMeshToVTK(mesh.surfaceMesh);
        auto layerBodyPolyData = CGALToVTKConverter::convertMeshToVTK(mesh.layerBody);

        m_surfaceMeshPolyData[i] = surfacePolyData;
        m_layerBodyPolyData[i] = layerBodyPolyData;
        m_colors[i] = colorMap[mesh.layer.composition];
    }

}

void Renderer::prepare(const std::vector<vtkSmartPointer<vtkPolyData>>& polyData)
{
    for (size_t i = 0; i < polyData.size(); ++i)
    {
        vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
        mapper->SetInputData(polyData[i]);

        vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);

        auto color = m_colors[i];
        actor->GetProperty()->SetColor(color.rgb);

        m_renderer->AddActor(actor);
    }
}

void Renderer::prepareSurfaces()
{
    prepare(m_surfaceMeshPolyData);
}

void Renderer::prepareLayerBody()
{
    prepare(m_layerBodyPolyData);
}

void Renderer::prepareEdges()
{
    for (const auto polyData : m_surfaceMeshPolyData)
    {
        vtkSmartPointer<vtkExtractEdges> extractEdges = vtkSmartPointer<vtkExtractEdges>::New();
        extractEdges->SetInputData(polyData);
        extractEdges->Update();

        vtkSmartPointer<vtkPolyData> edges = extractEdges->GetOutput();
        vtkSmartPointer<vtkDataSetMapper> edgeMapper = vtkSmartPointer<vtkDataSetMapper>::New();
        edgeMapper->SetInputData(edges);

        vtkSmartPointer<vtkActor> edgeActor = vtkSmartPointer<vtkActor>::New();
        edgeActor->SetMapper(edgeMapper);
        edgeActor->GetProperty()->SetColor(render::black.rgb);

        m_renderer->AddActor(edgeActor);
    }
}

void Renderer::preparePoints()
{
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

    for (const auto& point : m_meshes[0].layer.points)
        points->InsertNextPoint(point.x, point.y, point.z);
    polyData->SetPoints(points);

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
