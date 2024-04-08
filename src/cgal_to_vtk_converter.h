#pragma once

#include "./point/point.h"

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

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using DT2 = CGAL::Delaunay_triangulation_2<K>;
using Point2 = K::Point_2;
using Point3 = K::Point_3;

class CGALToVTKConverter
{
public:
    static vtkSmartPointer<vtkUnstructuredGrid> gridifyTriangulation(const DT2& triangulation, const std::vector<Point>& points3d);
};
