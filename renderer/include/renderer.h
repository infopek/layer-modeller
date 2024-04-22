#pragma once

#include <models/point.h>
#include <common-includes/vtk.h>

#include <triangulation/delaunay.h>

#include <render_settings.h>

#include <vector>

class Renderer
{
public:
    Renderer();
    ~Renderer();

    // TODO: make Layer class that contains triangulation with textures, etc.
    void addLayers(const std::vector<Triangulation>& layers);

    void prepareTriangulations();
    void prepareConnectionMeshes();
    void prepareEdges();
    void preparePoints();
    void prepareCoordinateSystem();

    void render();

private:
    void init();

    vtkSmartPointer<vtkPolyData> generateConnectionMesh(const Triangulation& layer1, const Triangulation& layer2);

private:
    vtkSmartPointer<vtkRenderer> m_renderer;
    vtkSmartPointer<vtkRenderWindow> m_renderWindow;
    vtkSmartPointer<vtkRenderWindowInteractor> m_renderWindowInteractor;

    std::vector<Triangulation> m_triangulations{};
    std::vector<vtkSmartPointer<vtkUnstructuredGrid>> m_triangulationGrids{};
};
