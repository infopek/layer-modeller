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

std::string ModellerSet::s_logPrefix = "[MODELLER_SET] --";

ModellerSet::ModellerSet(const LayerBuilder& layerBuilder)
    : m_layerBuilder{ layerBuilder }
{
    init();
}

ModellerSet::~ModellerSet()
{
}

void ModellerSet::init()
{
    m_layerBuilder.buildLayers();

    const auto& layers = m_layerBuilder.getLayers();

    m_meshes.resize(layers.size());
    for (size_t i = 0; i < layers.size(); i++)
        m_meshes[i].layer = layers[i];
}

void ModellerSet::convertToPolygonSoup(const SurfaceMesh& mesh, std::vector<Point3>& points, std::vector<std::vector<size_t>>& polygons)
{
    std::map<SurfaceMesh::Vertex_index, size_t> vertexIndexMap{};
    size_t index = 0;
    for (auto v : mesh.vertices())
    {
        points.push_back(mesh.point(v));
        vertexIndexMap[v] = index++;
    }
    for (auto f : mesh.faces())
    {
        std::vector<size_t> polygon{};
        for (auto v : CGAL::vertices_around_face(mesh.halfedge(f), mesh))
            polygon.push_back(vertexIndexMap[v]);
        polygons.push_back(polygon);
    }
}

void ModellerSet::convertToSurfaceMesh(const std::vector<Point3>& points, const std::vector<std::vector<size_t>>& polygons, SurfaceMesh& mesh)
{
    std::vector<SurfaceMesh::Vertex_index> vertices{};
    vertices.reserve(points.size());
    for (const auto& point : points)
        vertices.push_back(mesh.add_vertex(point));

    for (const auto& polygon : polygons)
    {
        std::vector<SurfaceMesh::Vertex_index> face{};
        face.reserve(3);
        for (const auto& idx : polygon)
            face.push_back(vertices[idx]);
        mesh.add_face(face);
    }
}

void ModellerSet::processMesh(SurfaceMesh& mesh)
{
    // Convert to poly soup
    std::vector<Point3> points{};
    std::vector<std::vector<size_t>> polygons{};
    points.reserve(static_cast<size_t>(mesh.num_vertices()));
    polygons.reserve(static_cast<size_t>(mesh.num_faces()));
    convertToPolygonSoup(mesh, points, polygons);

    // Convert repaired mesh back to SurfaceMesh
    PMP::repair_polygon_soup(points, polygons);
    SurfaceMesh repairedMesh{};
    repairedMesh.reserve(points.size(), 0, polygons.size());
    convertToSurfaceMesh(points, polygons, repairedMesh);

    // Other repairs
    PMP::stitch_borders(repairedMesh);
    PMP::triangulate_faces(repairedMesh);
    if (!PMP::is_outward_oriented(repairedMesh))
        PMP::reverse_face_orientations(repairedMesh);
    PMP::remove_degenerate_faces(repairedMesh);

    assert(CGAL::is_valid_polygon_mesh(repairedMesh));

    // Our mesh is now repaired
    mesh = std::move(repairedMesh);
}

void ModellerSet::createMeshes()
{
    float lowestZ = getMinimumZ(m_meshes[0].layer.points);
    Logger::log(LogLevel::INFO, ModellerSet::s_logPrefix + " Lowest point: " + std::to_string(lowestZ));

    Logger::log(LogLevel::INFO, ModellerSet::s_logPrefix + " Creating " + std::to_string(m_meshes.size()) + " meshes...");
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
        Logger::log(LogLevel::INFO, ModellerSet::s_logPrefix + " (Layer " + std::to_string(i + 1) + "): " + "Flattening...");
        for (const auto& p : points3d)
        {
            points2d.emplace_back(p.x, p.y);
            elevations[std::make_pair(p.x, p.y)] = p.z;
        }

        // Create 2D triangulation
        Logger::log(LogLevel::INFO, ModellerSet::s_logPrefix + " (Layer " + std::to_string(i + 1) + "): " + "Creating 2D triangulation...");
        dt.insert(points2d.begin(), points2d.end());
        surfaceMesh.reserve(dt.number_of_faces() * 3, 0, dt.number_of_faces());
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

            auto vi0 = surfaceMesh.add_vertex(p0);
            auto vi1 = surfaceMesh.add_vertex(p1);
            auto vi2 = surfaceMesh.add_vertex(p2);

            surfaceMesh.add_face(vi0, vi1, vi2);
        }

        // Extrude surface
        Logger::log(LogLevel::INFO, ModellerSet::s_logPrefix + " (Layer " + std::to_string(i + 1) + "): " + "Extruding triangulated surface...");
        Vector3 extrudeVector(0.0, 0.0, -450.0);
        CGAL::Polygon_mesh_processing::extrude_mesh(surfaceMesh, layerBody, extrudeVector);

        // Repair mesh
        Logger::log(LogLevel::INFO, ModellerSet::s_logPrefix + " (Layer " + std::to_string(i + 1) + "): " + "Repairing extruded surface...");
        processMesh(layerBody);
        m_extrudedMeshes.push_back(layerBody);

        // Perform set operations on meshes
        if (i > 0)
        {
            SurfaceMesh result{};
            Logger::log(LogLevel::INFO, ModellerSet::s_logPrefix + " (Layer " + std::to_string(i + 1) + "): " + "Computing geometric difference with layer " + std::to_string(i) + " ...");
            bool validDifference = PMP::corefine_and_compute_difference(layerBody, m_extrudedMeshes[i - 1], result);
            layerBody = std::move(result);
            if (validDifference)
            {
                Logger::log(LogLevel::INFO, ModellerSet::s_logPrefix + " (Layer " + std::to_string(i + 1) + "): " + "The difference is valid.");
            }
            else
            {
                Logger::log(LogLevel::WARN, ModellerSet::s_logPrefix + " (Layer " + std::to_string(i + 1) + "): " + "The difference is invalid.");
            }
        }
    }
}

float ModellerSet::getMinimumZ(const std::vector<Point>& layerPoints)
{
    auto minPoint = std::min_element(layerPoints.begin(), layerPoints.end(),
        [](const Point& p1, const Point& p2)
        {
            return p1.z < p2.z;
        });

    return minPoint->z;
}

