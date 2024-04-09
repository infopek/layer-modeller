#include <cgal_to_vtk_converter.h>

vtkSmartPointer<vtkUnstructuredGrid> CGALToVTKConverter::gridifyTriangulation(const Triangulation& triangulation)
{
    vtkSmartPointer<vtkUnstructuredGrid> unstructuredGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();

    std::unordered_map<Point2, vtkIdType> pointMap{};

    const auto& points3d = triangulation.getPoints();
    for (const auto& p : points3d)
    {
        vtkIdType vtkId = points->InsertNextPoint(p.x, p.y, p.z);
        pointMap[Point2(p.x, p.y)] = vtkId;
    }

    const auto& dt = triangulation.getTriangulation();
    for (auto it = dt.finite_faces_begin(); it != dt.finite_faces_end(); ++it)
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
