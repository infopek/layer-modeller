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
    Polyhedron extrudeTriangle(const Point3& p0, const Point3& p1, const Point3& p2, double extrusionHeight);

private:
    std::vector<Mesh> m_meshes{};
};
