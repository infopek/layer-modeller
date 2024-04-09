#pragma once

#include <models/point.h>
#include <common-includes/vtk.h>

#include <triangulation/delaunay.h>

#include <render_settings.h>

class DelaunayRenderer
{
public:
    DelaunayRenderer(const Triangulation& triangulation);
    ~DelaunayRenderer();

    void init();

    void addTriangulation();
    void addEdges();
    void addPoints();
    void addCoordinateSystem();

    void render();

private:
    vtkSmartPointer<vtkRenderer> m_renderer;
    vtkSmartPointer<vtkRenderWindow> m_renderWindow;
    vtkSmartPointer<vtkRenderWindowInteractor> m_renderWindowInteractor;

    Triangulation m_triangulation;
    vtkSmartPointer<vtkUnstructuredGrid> m_triangulationGrid;
};
