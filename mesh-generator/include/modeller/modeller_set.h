#pragma once

#include <models/mesh.h>
#include <layer_builder.h>

#include <common-includes/cgal.h>

#include <vector>

class ModellerSet
{
public:
    ModellerSet(const LayerBuilder& layerBuilder);
    ~ModellerSet();

    void createMeshes();

    inline const std::vector<Mesh>& getMeshes() const { return m_meshes; }

private:
    void init();

private:
    LayerBuilder m_layerBuilder;
    std::vector<Mesh> m_meshes{};
};
