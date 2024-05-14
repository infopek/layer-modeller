#pragma once

#include <models/layer.h>
#include <models/mesh.h>

#include <common-includes/cgal.h>

#include <vector>

class ModellerSet
{
public:
    ModellerSet(const std::vector<Layer>& layers);
    ~ModellerSet();

    void createMeshes();

    inline const std::vector<Mesh>& getMeshes() const { return m_meshes; }

private:

private:
    std::vector<Mesh> m_meshes{};
};
