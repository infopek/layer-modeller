#pragma once

#include <models/point.h>
#include <models/mesh.h>

#include <common-includes/vtk.h>

#include <string>

class ShapefileGenerator
{
public:
    ShapefileGenerator();
    ~ShapefileGenerator();

    void addMeshes(const std::vector<Mesh>& meshes);
    void writeVTKFile(const std::string& path);

private:
    std::vector<Mesh> m_meshes{};
    std::vector<vtkSmartPointer<vtkPolyData>> m_layerBodyPolyData{};
};