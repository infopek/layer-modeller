/*
* CGAL, GDAL includes
*/
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <gdal_priv.h>
#include <gdal.h>

/*
* VTK includes
*/
#include <vtkActor.h>
#include <vtkActor2D.h>
#include <vtkCamera.h>
#include <vtkCellData.h>
#include <vtkCellIterator.h>
#include <vtkCleanPolyData.h>
#include <vtkDataSetMapper.h>
#include <vtkDelaunay3D.h>
#include <vtkExtractEdges.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkTextMapper.h>
#include <vtkTextProperty.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLPolyDataReader.h>

/*
* STL includes
*/
#include <sstream>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using DT = CGAL::Delaunay_triangulation_3<K>;
using Point = K::Point_3;

// static std::vector<Point> readTiff(const std::string& filepath)
// {
//     GDALAllRegister();

//     // Open the raster dataset (TIF file)
//     GDALDataset* dataset = (GDALDataset*)GDALOpen(filepath.c_str(), GA_ReadOnly);

//     // Get the raster band (assuming there's only one band)
//     GDALRasterBand* band = dataset->GetRasterBand(1);

//     // Get the dimensions of the raster
//     int width = band->GetXSize();
//     int height = band->GetYSize();

//     // Read the elevation data into a buffer
//     float* data = new float[width * height] {};
//     band->RasterIO(GF_Read, 0, 0, width, height, data, width, height, GDT_Float32, 0, 0);

//     // Create a vector to hold elevation points in 3D
//     std::vector<Point> points{};
//     points.reserve(static_cast<std::vector<Point, std::allocator<Point>>::size_type>(width) * height);

//     const int sampleDistance = 12;

//     // Populate points with elevation data from the raster in 3D
//     for (int y = 0; y < height; y += sampleDistance)
//     {
//         for (int x = 0; x < width; x += sampleDistance)
//         {
//             points.emplace_back(x, y, data[y * width + x]);
//         }
//     }

//     delete[] data;

//     return points;
// }
static std::vector<Point> getPoints()
{
    std::vector<Point> points;

    // Manually add a few points in 3D space
    points.emplace_back(0, 0, 0);
    points.emplace_back(1, 0, 0);
    points.emplace_back(0, 1, 0);
    points.emplace_back(0, 0, 1);

    return points;
}

static DT triangulate(const std::vector<Point>& points)
{
    // Create a 3D Delaunay triangulation from the points
    DT dt{};
    dt.insert(points.begin(), points.end());

    return dt;
}

static vtkSmartPointer<vtkUnstructuredGrid> ConvertCGALTriangulationToVTK(const DT& triangulation)
{
    vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

    // Populate points and cells with CGAL triangulation information
    std::map<Point, vtkIdType> pointMap{};
    vtkIdType vtkId = 0;

    for (auto it = triangulation.finite_cells_begin(); it != triangulation.finite_cells_end(); ++it)
    {
        std::vector<vtkIdType> cellIds;
        for (int i = 0; i < 4; ++i)
        {
            Point p = it->vertex(i)->point();
            auto pointIter = pointMap.find(p);
            if (pointIter == pointMap.end())
            {
                vtkId = points->InsertNextPoint(p.x(), p.y(), p.z());
                pointMap[p] = vtkId;
            }
            else
            {
                vtkId = pointIter->second;
            }
            cellIds.push_back(vtkId);
        }
        cells->InsertNextCell(4, cellIds.data());
    }

    // Set points and cells in the vtkUnstructuredGrid
    unstructuredGrid->SetPoints(points);
    unstructuredGrid->SetCells(VTK_TETRA, cells);

    return unstructuredGrid;
}

static void render(const vtkSmartPointer<vtkUnstructuredGrid>& unstructuredGrid)
{
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(unstructuredGrid);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(actor);

    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    renderWindow->Render();
    renderWindowInteractor->Start();
}

static void renderWithEdges(const vtkSmartPointer<vtkUnstructuredGrid>& unstructuredGrid)
{
    // Create a mapper for the original grid
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(unstructuredGrid);

    // Create an actor for the original grid
    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    // Create an extract edges filter
    vtkSmartPointer<vtkExtractEdges> extractEdges = vtkSmartPointer<vtkExtractEdges>::New();
    extractEdges->SetInputData(unstructuredGrid);
    extractEdges->Update();

    // Get the edges
    vtkSmartPointer<vtkPolyData> edges = extractEdges->GetOutput();

    // Create a mapper for the edges
    vtkSmartPointer<vtkDataSetMapper> edgeMapper = vtkSmartPointer<vtkDataSetMapper>::New();
    edgeMapper->SetInputData(edges);

    // Create an actor for the edges
    vtkSmartPointer<vtkActor> edgeActor = vtkSmartPointer<vtkActor>::New();
    edgeActor->SetMapper(edgeMapper);
    edgeActor->GetProperty()->SetColor(1.0, 0.0, 0.0); // Set color to red for example

    // Create a renderer, render window, and interactor
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    // Add actors to the renderer
    renderer->AddActor(actor);      // Add the original grid
    renderer->AddActor(edgeActor);  // Add the edges

    // Set background color and start rendering
    renderer->SetBackground(0.0, 0.0, 0.0);
    renderWindow->Render();
    renderWindowInteractor->Start();
}

int main(int argc, char* argv[])
{
    // const std::string filepath = "../../res/sample2_dem.tif";
    auto points = getPoints();

    DT dt = triangulate(points);

    auto unstructuredGrid = ConvertCGALTriangulationToVTK(dt);

    renderWithEdges(unstructuredGrid);
}
