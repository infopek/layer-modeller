#pragma once

#include "../point/point.h"
#include "render_settings.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>

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

class TriangulationRenderer
{
public:
    TriangulationRenderer(const DT2& triangulation, const std::vector<Point>& points3d);
    ~TriangulationRenderer();

    void init();

    void addTriangulation();
    void addPoints();
    void addCoordinateSystem();

    void render();

private:
    vtkSmartPointer<vtkRenderer> m_renderer;
    vtkSmartPointer<vtkRenderWindow> m_renderWindow;
    vtkSmartPointer<vtkRenderWindowInteractor> m_renderWindowInteractor;

    std::vector<Point> m_points{};
    DT2 m_triangulation;
    vtkSmartPointer<vtkUnstructuredGrid> m_triangulationGrid;
};
