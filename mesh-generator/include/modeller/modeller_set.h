#pragma once

#include <models/mesh.h>
#include <layer_builder.h>
#include <logging.h>

#include <common-includes/cgal.h>

#include <unordered_map>
#include <vector>

struct PairHash
{
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const
    {
        auto hash1 = std::hash<T1>{}(pair.first);
        auto hash2 = std::hash<T2>{}(pair.second);
        return hash1 ^ hash2;
    }
};

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

    void flattenBottomSurface(SurfaceMesh& mesh, float zVal) const;

    std::vector<Point2> transformTo2D(const std::vector<Point>& points, std::unordered_map<std::pair<double, double>, double, PairHash>& elevations) const;
    void construct3DSurfaceMesh(CDT2& dt, SurfaceMesh& surfaceMesh, const std::unordered_map<std::pair<double, double>, double, PairHash>& elevations) const;

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
