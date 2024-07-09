#pragma once

#include <models/point.h>
#include <models/mesh.h>

#include <common-includes/vtk.h>

#include <render_settings.h>

#include <vector>

class Renderer
{
public:
    Renderer(vtkSmartPointer<vtkRenderer> renderer);
    ~Renderer();

    void addMeshes(const std::vector<Mesh>& meshes);

    void prepareSurfaces();
    void prepareLayerBody();
    void prepareEdges();
    void preparePoints();
    void prepareCoordinateSystem();

    void test();

private:
    void init();

    void prepare(const std::vector<vtkSmartPointer<vtkPolyData>>& polyData);

private:
    vtkSmartPointer<vtkRenderer> m_renderer;

    std::vector<Mesh> m_meshes{};
    std::vector<vtkSmartPointer<vtkPolyData>> m_surfaceMeshPolyData{};
    std::vector<vtkSmartPointer<vtkPolyData>> m_layerBodyPolyData{};
    std::vector<render::Color> m_colors{};
};
