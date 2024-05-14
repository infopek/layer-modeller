#include <cgal_to_vtk_converter.h>

vtkSmartPointer<vtkPolyData> CGALToVTKConverter::convertMeshToVTK(const SurfaceMesh& mesh)
{
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
    vtkSmartPointer<vtkCellArray> triangles = vtkSmartPointer<vtkCellArray>::New();

    // Add vertices to points
    for (auto v : mesh.vertices()) {
        Point3 p = mesh.point(v);
        points->InsertNextPoint(p.x(), p.y(), p.z());
    }

    // Add faces to triangles
    for (auto f : mesh.faces())
    {
        std::vector<SurfaceMesh::Vertex_index> face_vertices;
        for (auto v : mesh.vertices_around_face(mesh.halfedge(f)))
            face_vertices.push_back(v);

        // Ensure the face is a triangle
        if (face_vertices.size() == 3)
        {
            vtkIdType triangle[3];
            for (int i = 0; i < 3; ++i)
                triangle[i] = face_vertices[i];

            triangles->InsertNextCell(3, triangle);
        }
    }

    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);
    polydata->SetPolys(triangles);

    return polydata;
}
