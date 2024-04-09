#pragma once

#include <models/point.h>
#include <triangulation/delaunay.h>

#include <common-includes/cgal.h>
#include <common-includes/vtk.h>

class CGALToVTKConverter
{
public:
    static vtkSmartPointer<vtkUnstructuredGrid> gridifyTriangulation(const Triangulation& triangulation);
};
