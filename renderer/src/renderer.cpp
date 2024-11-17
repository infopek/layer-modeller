#include <renderer.h>

#include <converters/cgal_to_vtk_converter.h>
#include <vtkCubeSource.h>

std::string Renderer::s_logPrefix = "[RENDERER] --";

Renderer::Renderer(vtkSmartPointer<vtkRenderer> renderer)
    : m_renderer{ renderer }
{
}

Renderer::~Renderer()
{

}

void Renderer::addMeshes(const std::vector<Mesh>& meshes)
{
    std::unordered_map<std::string, render::Color> colorMap{
        {"comp0", render::red},
        {"comp1", render::blue},
        {"comp2", render::brown},
        {"comp3", render::green},
        {"comp4", render::purple},
        {"comp5", render::orange},
        {"comp6", render::yellow},
        {"comp7", render::cyan},
        {"comp8", render::magenta},
        {"comp9", render::lime},
    };

    m_meshes = meshes;

    const size_t numMeshes = meshes.size();
    m_surfaceMeshPolyData.resize(numMeshes);
    m_layerBodyPolyData.resize(numMeshes);
    m_colors.resize(numMeshes);

    Logger::log(LogLevel::INFO, Renderer::s_logPrefix + " Adding " + std::to_string(numMeshes) + " meshes...");
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
        actor->GetProperty()->SetOpacity(1.0);

        m_renderer->AddActor(actor);
    }
}

void Renderer::prepareSurfaces()
{
    Logger::log(LogLevel::INFO, Renderer::s_logPrefix + " Preparing surfaces of meshes...");
    prepare(m_surfaceMeshPolyData);
}

void Renderer::prepareMeshes()
{
    Logger::log(LogLevel::INFO, Renderer::s_logPrefix + " Preparing 3D meshes...");
    prepare(m_layerBodyPolyData);
}

void Renderer::prepareEdges()
{
    Logger::log(LogLevel::INFO, Renderer::s_logPrefix + " Preparing edges for meshes...");
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
        edgeActor->GetProperty()->SetOpacity(1.0);

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

void Renderer::clear()
{
    clearMeshes();
    clearPolyData();
    clearColors();
}

void Renderer::clearMeshes()
{
    m_meshes.clear();
}

void Renderer::clearPolyData()
{
    m_surfaceMeshPolyData.clear();
    m_layerBodyPolyData.clear();
}

void Renderer::clearColors()
{
    m_colors.clear();
}