#include <modeller/modeller_set.h>

#include <algorithm>
#include <execution>
#include <map>
#include <omp.h>
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

std::string ModellerSet::s_logPrefix = "[MODELLER_SET] -- ";

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
    m_extrudedMeshes.resize(layers.size());
    for (size_t i = 0; i < layers.size(); i++)
        m_meshes[i].layer = layers[i];
}

void ModellerSet::convertToPolygonSoup(const SurfaceMesh& mesh, std::vector<Point3>& points, std::vector<std::vector<size_t>>& polygons)
{
    std::unordered_map<SurfaceMesh::Vertex_index, size_t> vertexIndexMap{};
    size_t index = 0;

    std::vector<SurfaceMesh::Vertex_index> vertices;
    vertices.reserve(mesh.num_vertices());
    for (auto v : mesh.vertices())
        vertices.push_back(v);

    // Parallelize vertex processing
    points.resize(vertices.size());
#pragma omp parallel for
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        auto v = vertices[i];
        points[i] = mesh.point(v);

        // Use a critical section for thread-safe access to shared data
#pragma omp critical
        {
            vertexIndexMap[v] = i;
        }
    }

    for (auto f : mesh.faces())
    {
        std::vector<size_t> polygon{};
        polygon.reserve(CGAL::vertices_around_face(mesh.halfedge(f), mesh).size());

        for (auto v : CGAL::vertices_around_face(mesh.halfedge(f), mesh))
            polygon.push_back(vertexIndexMap[v]);

        polygons.push_back(std::move(polygon));
    }

}

void ModellerSet::convertToSurfaceMesh(const std::vector<Point3>& points, const std::vector<std::vector<size_t>>& polygons, SurfaceMesh& mesh)
{
    std::vector<SurfaceMesh::Vertex_index> vertices(points.size());

#pragma omp parallel for
    for (size_t i = 0; i < points.size(); ++i)
        vertices[i] = mesh.add_vertex(points[i]);

    for (const auto& polygon : polygons)
    {
        std::vector<SurfaceMesh::Vertex_index> face{};
        face.reserve(3);
        for (const auto& idx : polygon)
            face.push_back(vertices[idx]);

        mesh.add_face(std::move(face));
    }
}

void ModellerSet::repair(SurfaceMesh& mesh)
{
    // Convert to poly soup
    std::vector<Point3> points{};
    std::vector<std::vector<size_t>> polygons{};
    points.reserve(static_cast<size_t>(mesh.num_vertices()));
    polygons.reserve(static_cast<size_t>(mesh.num_faces()));
    convertToPolygonSoup(mesh, points, polygons);

    PMP::repair_polygon_soup(points, polygons);

    // Convert repaired mesh back to SurfaceMesh
    SurfaceMesh repairedMesh{};
    repairedMesh.reserve(points.size(), 0, polygons.size());
    convertToSurfaceMesh(points, polygons, repairedMesh);

    // Other repairs
    // PMP::stitch_borders(repairedMesh);
    // PMP::triangulate_faces(repairedMesh);
    // if (!PMP::is_outward_oriented(repairedMesh))
    //     PMP::reverse_face_orientations(repairedMesh);
    // PMP::remove_degenerate_faces(repairedMesh);

    assert(CGAL::is_valid_polygon_mesh(repairedMesh));

    // Our mesh is now repaired
    mesh = std::move(repairedMesh);
}

void ModellerSet::triangulate(int index)
{
    Logger::log(LogLevel::INFO, ModellerSet::s_logPrefix + "Triangulating surface for layer " + std::to_string(index) + "...");
    const auto& points = m_meshes[index].layer.points;
    auto& surfaceMesh = m_meshes[index].surfaceMesh;

    std::vector<Point2> points2d{};
    points2d.reserve(points.size());
    std::unordered_map<std::pair<double, double>, double, PairHash> elevations{};
    for (const auto& p : points)
    {
        points2d.emplace_back(p.x, p.y);    // transform to 2d by ignoring z coord
        elevations[std::make_pair(p.x, p.y)] = p.z;
    }

    CDT2 dt{};
    dt.insert(points2d.begin(), points2d.end());
    surfaceMesh.reserve(dt.number_of_faces() * 3, 0, dt.number_of_faces());  // 3 vertices for 1 triangle
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
}

void ModellerSet::extrude(float lowestZ, int index)
{
    Logger::log(LogLevel::INFO, ModellerSet::s_logPrefix + "Extruding surface for layer " + std::to_string(index) + "...");

    const auto& points = m_meshes[index].layer.points;
    const auto& surfaceMesh = m_meshes[index].surfaceMesh;
    auto& layerBody = m_meshes[index].layerBody;

    float currLowestZ = getMinimumZ(points);
    Vector3 extrudeVector(0.0, 0.0, -(currLowestZ - lowestZ));  // extrude vector should point downwards
    CGAL::Polygon_mesh_processing::extrude_mesh(surfaceMesh, layerBody, extrudeVector);

    repair(layerBody);

    m_extrudedMeshes[index] = layerBody;
}

void ModellerSet::takeDifference(int idx1, int idx2)
{
    Logger::log(LogLevel::INFO, ModellerSet::s_logPrefix + "Computing geometric difference with layer " + std::to_string(idx1) + "...");

    auto& layerBody = m_meshes[idx1].layerBody;
    SurfaceMesh result{};
    bool validDifference = PMP::corefine_and_compute_difference(layerBody, m_extrudedMeshes[idx2], result);
    if (validDifference)
    {
        layerBody = std::move(result);
    }
    else
    {
        Logger::log(LogLevel::WARN, ModellerSet::s_logPrefix + "The difference is invalid.");
    }
}

void ModellerSet::createMeshes()
{
    Logger::log(LogLevel::INFO, ModellerSet::s_logPrefix + " Creating " + std::to_string(m_meshes.size()) + " meshes...");

    const size_t numMeshes = m_meshes.size();
    const float lowestZ = getMinimumZ(m_meshes[numMeshes - 1].layer.points) - 200.0f;   // below the lowest point of bottom layer

    auto makeRange = [](int start, int end) -> std::vector<int> {
        std::vector<int> range{};
        for (int i = start; i >= end; --i)
            range.push_back(i);
        return range;
        };

    std::vector<int> range1 = makeRange(numMeshes - 1, 0);
    std::for_each(std::execution::par, range1.begin(), range1.end(), [&](int i) {
        triangulate(i);
        extrude(lowestZ, i);
        });

    std::vector<int> range2 = makeRange(numMeshes - 2, 0);
    std::for_each(std::execution::par, range2.begin(), range2.end(), [&](int i) {
        takeDifference(i, i + 1);
        });

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

