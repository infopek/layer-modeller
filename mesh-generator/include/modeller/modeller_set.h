#pragma once

#include <models/mesh.h>
#include <layer_builder.h>
#include <logging.h>

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

    void triangulate(int index);
    void extrude(float lowestZ, int index);
    void takeDifference(int idx1, int idx2);

    static void repair(SurfaceMesh& mesh);

    static void convertToPolygonSoup(const SurfaceMesh& mesh, std::vector<Point3>& points, std::vector<std::vector<std::size_t>>& polygons);
    static void convertToSurfaceMesh(const std::vector<Point3>& points, const std::vector<std::vector<std::size_t>>& polygons, SurfaceMesh& mesh);

    static float getMinimumZ(const std::vector<Point>& layerPoints);

private:
    LayerBuilder m_layerBuilder;
    std::vector<Mesh> m_meshes{};
    std::vector<SurfaceMesh> m_extrudedMeshes{};

    static std::string s_logPrefix;
};
