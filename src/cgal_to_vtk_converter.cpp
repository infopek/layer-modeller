#include "./cgal_to_vtk_converter.h"

vtkSmartPointer<vtkUnstructuredGrid> CGALToVTKConverter::gridifyTriangulation(const DT2& triangulation, const std::vector<Point>& points3d)
{
    vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

    std::unordered_map<Point2, vtkIdType> pointMap{};

    for (const auto& p : points3d)
    {
        vtkIdType vtkId = points->InsertNextPoint(p.x, p.y, p.z);
        pointMap[Point2(p.x, p.y)] = vtkId;
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
