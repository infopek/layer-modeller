#include <vector>
#include <iostream>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkExtractEdges.h>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using DT = CGAL::Delaunay_triangulation_3<K>;
using Point = K::Point_3;

static std::vector<Point> getPoints()
{
    std::vector<Point> points;
    points.emplace_back(0, 0, 0);
    points.emplace_back(1, 0, 0);
    points.emplace_back(0, 1, 0);
    points.emplace_back(0, 0, 1);

    return points;
}

static DT triangulate(const std::vector<Point>& points)
{
    DT dt;
    dt.insert(points.begin(), points.end());
    return dt;
}

static vtkSmartPointer<vtkUnstructuredGrid> gridifyTriangulation(const DT& triangulation)
{
    vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

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
    renderWindow->SetSize(860, 540); // Set the size of the render window
    renderWindow->AddRenderer(renderer);

    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    renderer->SetBackground(0.0, 0.0, 0.0);

    renderWindow->Render();
    renderWindowInteractor->Start();
}

static void renderWithEdges(const vtkSmartPointer<vtkUnstructuredGrid>& unstructuredGrid)
{
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputData(unstructuredGrid);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(actor);

    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(860, 540); // Set the size of the render window
    renderWindow->AddRenderer(renderer);

    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    vtkSmartPointer<vtkExtractEdges> extractEdges = vtkSmartPointer<vtkExtractEdges>::New();
    extractEdges->SetInputData(unstructuredGrid);
    extractEdges->Update();

    vtkSmartPointer<vtkPolyData> edges = extractEdges->GetOutput();

    vtkSmartPointer<vtkDataSetMapper> edgeMapper = vtkSmartPointer<vtkDataSetMapper>::New();
    edgeMapper->SetInputData(edges);

    vtkSmartPointer<vtkActor> edgeActor = vtkSmartPointer<vtkActor>::New();
    edgeActor->SetMapper(edgeMapper);
    edgeActor->GetProperty()->SetColor(1.0, 0.0, 0.0);

    renderer->AddActor(actor);
    renderer->AddActor(edgeActor);
    renderer->SetBackground(0.0, 0.0, 0.0);

    renderWindow->Render();
    renderWindowInteractor->Start();
}

int main(int argc, char* argv[])
{
    auto points = getPoints();

    DT dt = triangulate(points);

    vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid = gridifyTriangulation(dt);

    renderWithEdges(unstructuredGrid);

    return 0;
}
