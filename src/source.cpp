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
#include <vtkProperty.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSphereSource.h>
#include <vtkExtractEdges.h>

#include <perlin_noise.h>

#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <unordered_map>
#include <utility>
#include <vtkSphereSource.h>

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
    };

    inline constexpr int windowWidth = 1080;
    inline constexpr int windowHeight = 720;

    inline constexpr Color black{ 0.0, 0.0, 0.0 };
    inline constexpr Color red{ 1.0, 0.2, 0.1 };
}

static std::vector<Point3> generatePoints(int width, int height)
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<double> distribution(0.0, 7.0);

    const siv::PerlinNoise::seed_type seed = 123456u;
    const siv::PerlinNoise perlin{ seed };

    std::vector<Point3> points;
    points.reserve(width * height);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            double xCoord = distribution(rng);
            double yCoord = distribution(rng);
            double elevation = perlin.octave2D_01((x * 0.01), (y * 0.01), 4);
            points.emplace_back(xCoord, yCoord, elevation * 20.0);
        }
    }

    return points;
}

static DT2 triangulate(const std::vector<Point3>& points)
{
    std::vector<Point2> points2d;
    for (const auto& p : points)
        points2d.emplace_back(p.x(), p.y());

    DT2 dt;
    dt.insert(points2d.begin(), points2d.end());
    return dt;
}

static double interpolateElevation(const Point2& p, const DT2& triangulation, const std::vector<Point3>& originalPoints)
{
    double elevation = 0.0;
    double totalWeight = 0.0;
    int numElevations = static_cast<int>(originalPoints.size());

    for (auto it = triangulation.finite_vertices_begin(); it != triangulation.finite_vertices_end(); ++it)
    {
        Point2 vertexPoint = it->point();
        int index = std::distance(triangulation.finite_vertices_begin(), it);
        if (index < 0 || index >= numElevations) continue;

        double distance = sqrt(pow(vertexPoint.x() - p.x(), 2) + pow(vertexPoint.y() - p.y(), 2));

        // Avoid division by zero and handle very small distances
        if (distance < 1e-6) {
            // If distance is very small, assume the points are coincident and assign a weight of 1.0
            elevation += originalPoints[index].z();
            totalWeight += 1.0;
        }
        else {
            double weight = 1.0 / distance;
            elevation += originalPoints[index].z() * weight;
            totalWeight += weight;
        }
    }

    // Normalize elevation by total weight to avoid skewing the result
    if (totalWeight > 0.0) {
        elevation /= totalWeight;
    }

    return elevation;
}

static vtkSmartPointer<vtkUnstructuredGrid> gridifyTriangulation(const DT2& triangulation, const std::vector<Point3>& originalPoints)
{
    vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

    std::map<Point2, vtkIdType> pointMap{};
    vtkIdType vtkId = 0;

    int numElevations = static_cast<int>(originalPoints.size());
    for (auto it = triangulation.finite_faces_begin(); it != triangulation.finite_faces_end(); ++it)
    {
        std::vector<vtkIdType> cellIds;
        for (int i = 0; i < 3; ++i)
        {
            Point2 p = it->vertex(i)->point();
            auto pointIter = pointMap.find(p);
            if (pointIter == pointMap.end())
            {
                int index = std::distance(triangulation.finite_faces_begin(), it);
                double elevation{};
                if (index >= 0 && index < numElevations)
                    elevation = originalPoints[index].z();
                else
                    elevation = interpolateElevation(p, triangulation, originalPoints);

                vtkId = points->InsertNextPoint(p.x(), p.y(), elevation);
                pointMap[p] = vtkId;
            }
            else
            {
                vtkId = pointIter->second;
            }
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
    vtkSmartPointer<vtkRenderWindow> renderWindow,
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor,
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
    vtkSmartPointer<vtkRenderWindow> renderWindow,
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor,
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
    sphereSource->SetRadius(0.1); // Adjust the radius of the spheres as needed

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

int main(int argc, char* argv[])
{
    auto points = generatePoints(6, 6);

    DT2 dt = triangulate(points);

    auto unstructuredGrid = gridifyTriangulation(dt, points);

    // To be moved into some class
    renderer->SetBackground(render::black.r, render::black.g, render::black.b);

    renderWindow->SetSize(render::windowWidth, render::windowHeight);
    renderWindow->AddRenderer(renderer);

    renderWindowInteractor->SetRenderWindow(renderWindow);

    renderPoints(renderer, renderWindow, renderWindowInteractor, points);
    renderTriangulationWithEdges(renderer, renderWindow, renderWindowInteractor, unstructuredGrid);

    renderWindow->Render();
    renderWindowInteractor->Start();

    return 0;
}
