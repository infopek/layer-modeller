#include <cgal_to_vtk_converter.h>

vtkSmartPointer<vtkPolyData> CGALToVTKConverter::convertMeshToVTK(const Polyhedron& mesh)
{
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> polygons = vtkSmartPointer<vtkCellArray>::New();

    for (auto v = mesh.vertices_begin(); v != mesh.vertices_end(); ++v)
    {
        auto point = v->point();
        points->InsertNextPoint(point.x(), point.y(), point.z());
    }

    for (auto f = mesh.facets_begin(); f != mesh.facets_end(); ++f)
    {
        auto halfedge = f->halfedge();
        vtkIdType triangle[3];
        int i = 0;
        do
        {
            triangle[i++] = std::distance(mesh.vertices_begin(), halfedge->vertex());
            halfedge = halfedge->next();
        } while (halfedge != f->halfedge());
        polygons->InsertNextCell(3, triangle);
    }

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);
    polydata->SetPolys(polygons);

    return polydata;
}
