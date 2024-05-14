#pragma once

#include <models/point.h>
#include <models/mesh.h>

#include <common-includes/cgal.h>
#include <common-includes/vtk.h>

class CGALToVTKConverter
{
public:
    static vtkSmartPointer<vtkPolyData> convertMeshToVTK(const Polyhedron& mesh);
};
