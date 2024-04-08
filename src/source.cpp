#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/property_map.h>

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkGlyph3D.h>
#include <vtkLineSource.h>
#include <vtkProperty.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSphereSource.h>
#include <vtkExtractEdges.h>

#include <perlin-noise/perlin_noise.h>

#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <unordered_map>
#include <utility>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using DT2 = CGAL::Delaunay_triangulation_2<K>;
using Point2 = K::Point_2;
using Point3 = K::Point_3;

namespace render
{
    struct Color
    {
        double r;
        double g;
        double b;
        double a;
    };

    inline constexpr int windowWidth = 1920;
    inline constexpr int windowHeight = 1080;

    inline constexpr Color black{ 0.0, 0.0, 0.0, 1.0 };
    inline constexpr Color red{ 1.0, 0.2, 0.1, 0.4 };
}

static std::vector<Point3> generatePoints(int width, int height)
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<double> distribution(1.0, 10.0);

    const siv::PerlinNoise::seed_type seed = 123456u;
    const siv::PerlinNoise perlin{ seed };

    int numPoints = width * height;
    std::vector<Point3> points;
    points.reserve(numPoints);

    for (int y = -height / 2; y <= height / 2; y++)
    {
        for (int x = -width / 2; x <= width / 2; x++)
        {
            double z = perlin.octave2D_01((x), (y), 4) * 5.0;
            // double z = sin(x) + cos(y);
        // double z = (x * x - y * y) / 2.0;
            points.emplace_back(x * 7.0, y * 7.0, z * distribution(rng));
        }
    }

    return points;
}

static DT2 triangulate(const std::vector<Point3>& points)
{
    // Triangulate in 2d first
    std::vector<Point2> points2d;
    points2d.reserve(points.size());
    for (const auto& p : points)
        points2d.emplace_back(p.x(), p.y());

    DT2 dt;
    dt.insert(points2d.begin(), points2d.end());
    return dt;
}

static vtkSmartPointer<vtkUnstructuredGrid> gridifyTriangulation(const DT2& triangulation, const std::vector<Point3>& originalPoints)
{
    vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

    std::unordered_map<Point2, vtkIdType> pointMap{};

    for (const auto& p : originalPoints)
    {
        vtkIdType vtkId = points->InsertNextPoint(p.x(), p.y(), p.z());
        pointMap[Point2(p.x(), p.y())] = vtkId;
    }

    for (auto it = triangulation.finite_faces_begin(); it != triangulation.finite_faces_end(); ++it)
    {
        std::vector<vtkIdType> cellIds;
        for (int i = 0; i < 3; ++i)
        {
            Point2 p = it->vertex(i)->point();
            vtkIdType vtkId = pointMap[p];
            cellIds.push_back(vtkId);
        }
        cells->InsertNextCell(3, cellIds.data());
    }

    unstructuredGrid->SetPoints(points);
    unstructuredGrid->SetCells(VTK_TRIANGLE, cells);

    return unstructuredGrid;
}

vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();

static void renderTriangulationWithEdges(
    vtkSmartPointer<vtkRenderer> renderer,
    vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid
)
{
    // Render window setup
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(unstructuredGrid);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // Edge setup
    vtkSmartPointer<vtkExtractEdges> extractEdges = vtkSmartPointer<vtkExtractEdges>::New();
    extractEdges->SetInputData(unstructuredGrid);
    extractEdges->Update();

    vtkSmartPointer<vtkPolyData> edges = extractEdges->GetOutput();
    vtkSmartPointer<vtkDataSetMapper> edgeMapper = vtkSmartPointer<vtkDataSetMapper>::New();
    edgeMapper->SetInputData(edges);

    vtkSmartPointer<vtkActor> edgeActor = vtkSmartPointer<vtkActor>::New();
    edgeActor->SetMapper(edgeMapper);
    edgeActor->GetProperty()->SetColor(render::red.r, render::red.g, render::red.b);

    renderer->AddActor(actor);
    renderer->AddActor(edgeActor);
}

void renderPoints(
    vtkSmartPointer<vtkRenderer> renderer,
    const std::vector<Point3>& points
)
{
    // Create a VTK PolyData object
    vtkSmartPointer<vtkPolyData> polyData = vtkSmartPointer<vtkPolyData>::New();

    // Create a VTK Points object to hold the point coordinates
    vtkSmartPointer<vtkPoints> pts = vtkSmartPointer<vtkPoints>::New();

    // Add the points to the VTK Points object
    for (const auto& point : points)
    {
        pts->InsertNextPoint(point.x(), point.y(), point.z());
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

    renderer->AddActor(actor);
}

void renderCoordinateSystem(vtkSmartPointer<vtkRenderer> renderer)
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
    renderer->AddActor(xActor);
    renderer->AddActor(yActor);
    renderer->AddActor(zActor);
}

int main(int argc, char* argv[])
{
    auto points = generatePoints(160, 160);

    DT2 dt = triangulate(points);

    auto unstructuredGrid = gridifyTriangulation(dt, points);

    // To be moved into some class
    renderer->SetBackground(render::black.r, render::black.g, render::black.b);

    renderWindow->SetSize(render::windowWidth, render::windowHeight);
    renderWindow->AddRenderer(renderer);

    renderWindowInteractor->SetRenderWindow(renderWindow);

    renderCoordinateSystem(renderer);
    renderPoints(renderer, points);
    renderTriangulationWithEdges(renderer, unstructuredGrid);

    renderWindow->Render();
    renderWindowInteractor->Start();

    return 0;
}
