#include <modeller/modeller_set.h>

#include <algorithm>
#include <unordered_map>

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

ModellerSet::ModellerSet(const std::vector<Layer>& layers)
{
    m_meshes.resize(layers.size());
    for (size_t i = 0; i < layers.size(); i++)
        m_meshes[i].layer = layers[i];
}

ModellerSet::~ModellerSet()
{
}

void ModellerSet::createMeshes()
{
    for (size_t i = 0; i < m_meshes.size(); ++i)
    {
        // References
        const auto& points3d = m_meshes[i].layer.points;
        auto& dt = m_meshes[i].dt;
        auto& surfaceMesh = m_meshes[i].surfaceMesh;
        auto& layerBody = m_meshes[i].layerBody;

        // Transform 3D points to 2D by ignoring z coordinate
        std::vector<Point2> points2d;
        std::unordered_map<std::pair<double, double>, double, PairHash> elevations{};
        points2d.reserve(points3d.size());
        for (const auto& p : points3d)
        {
            points2d.emplace_back(p.x, p.y);
            elevations[std::make_pair(p.x, p.y)] = p.z;
        }

        // Create 2D triangulation
        dt.insert(points2d.begin(), points2d.end());

        // Create surface meshes by adjusting the z-coordinates of triangle vertices
        layerBody.reserve(dt.number_of_faces());
        for (FaceIterator f = dt.finite_faces_begin(); f != dt.finite_faces_end(); ++f)
        {
            VertexHandle v0 = f->vertex(0);
            VertexHandle v1 = f->vertex(1);
            VertexHandle v2 = f->vertex(2);

            double z0 = elevations[std::make_pair(v0->point().x(), v0->point().y())];
            double z1 = elevations[std::make_pair(v1->point().x(), v1->point().y())];
            double z2 = elevations[std::make_pair(v2->point().x(), v2->point().y())];

            Point3 p0(v0->point().x(), v0->point().y(), z0);
            Point3 p1(v1->point().x(), v1->point().y(), z1);
            Point3 p2(v2->point().x(), v2->point().y(), z2);

            surfaceMesh.make_triangle(p0, p1, p2);

            // Compute convex hull of current triangle
            double extrusionHeight = 50.0;
            Polyhedron convexHull = extrudeTriangle(p0, p1, p2, extrusionHeight);

            layerBody.push_back(convexHull);
        }
    }
}

Polyhedron ModellerSet::extrudeTriangle(const Point3& p0, const Point3& p1, const Point3& p2, double extrusionHeight)
{
    std::vector<Point3> topPoints{
        p0,
        p1,
        p2
    };

    std::vector<Point3> bottomPoints{
        Point3(p0.x(), p0.y(), p0.z() - extrusionHeight),
        Point3(p1.x(), p1.y(), p1.z() - extrusionHeight),
        Point3(p2.x(), p2.y(), p2.z() - extrusionHeight)
    };

    std::vector<Point3> points{};
    points.insert(points.end(), bottomPoints.begin(), bottomPoints.end());
    points.insert(points.end(), topPoints.begin(), topPoints.end());

    Polyhedron convexHull{};
    CGAL::convex_hull_3(points.begin(), points.end(), convexHull);
    return convexHull;
}

Polyhedron getExtrudedPolygon(const Polyhedron& inputPolyhedron, double distance)
{
    Polyhedron result{};



    return result;
}

Polyhedron getPolyhedronBetweenLayers(const Polyhedron& topSurface, const Polyhedron& bottomSurface)
{

    double topExtrusionDist = 150.0;
    double bottomExtrusionDist = 50.0;

    auto topSolid = getExtrudedPolygon(topSurface, topExtrusionDist);
    auto bottomSolid = getExtrudedPolygon(bottomSurface, bottomExtrusionDist);

    Polyhedron result{};
    CGAL::Polygon_mesh_processing::corefine_and_compute_union(topSolid, bottomSolid, result);

    return result;
}

