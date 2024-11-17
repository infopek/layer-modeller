#include <modeller/modeller_set.h>

#include <algorithm>
#include <execution>
#include <omp.h>

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

    // Get mesh vertices
    std::vector<SurfaceMesh::Vertex_index> vertices{};
    vertices.reserve(mesh.num_vertices());
    for (auto v : mesh.vertices())
        vertices.push_back(v);

    // Vertex processing
    points.resize(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        auto v = vertices[i];
        points[i] = mesh.point(v);
        vertexIndexMap[v] = i;
    }

    // Fill poly with vertices of mesh
    for (auto f : mesh.faces())
    {
        std::vector<size_t> polygon{};
        auto vertsIt = CGAL::vertices_around_face(mesh.halfedge(f), mesh);
        polygon.reserve(vertsIt.size());
        for (auto v : vertsIt)
            polygon.push_back(vertexIndexMap[v]);

        polygons.push_back(std::move(polygon));
    }

}

void ModellerSet::convertToSurfaceMesh(const std::vector<Point3>& points, const std::vector<std::vector<size_t>>& polygons, SurfaceMesh& mesh)
{
    std::vector<SurfaceMesh::Vertex_index> vertices(points.size());

    // Add vertices to mesh
    for (size_t i = 0; i < points.size(); ++i)
        vertices[i] = mesh.add_vertex(points[i]);

    // Fill mesh face-by-face
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

    mesh = std::move(repairedMesh);
}

std::vector<Point2> ModellerSet::transformTo2D(const std::vector<Point>& points,
    std::unordered_map<std::pair<double, double>, double, PairHash>& elevations) const
{
    std::vector<Point2> points2d{};
    points2d.reserve(points.size());
    for (const auto& p : points)
    {
        points2d.emplace_back(p.x, p.y);
        elevations[std::make_pair(p.x, p.y)] = p.z;
    }
    return points2d;
}

void ModellerSet::construct3DSurfaceMesh(CDT2& dt, SurfaceMesh& surfaceMesh,
    const std::unordered_map<std::pair<double, double>, double, PairHash>& elevations) const
{
    for (auto f = dt.finite_faces_begin(); f != dt.finite_faces_end(); ++f)
    {
        auto get3DPoint = [&](VertexHandle vh) -> Point3 {
            double z = elevations.at(std::make_pair(vh->point().x(), vh->point().y()));
            return Point3(vh->point().x(), vh->point().y(), z);
            };

        Point3 p0 = get3DPoint(f->vertex(0));
        Point3 p1 = get3DPoint(f->vertex(1));
        Point3 p2 = get3DPoint(f->vertex(2));

        auto vi0 = surfaceMesh.add_vertex(p0);
        auto vi1 = surfaceMesh.add_vertex(p1);
        auto vi2 = surfaceMesh.add_vertex(p2);

        surfaceMesh.add_face(vi0, vi1, vi2);
    }
}

void ModellerSet::triangulate(int index)
{
    Logger::log(LogLevel::INFO, ModellerSet::s_logPrefix + "Triangulating surface for layer " + std::to_string(index) + "...");

    // Transform points to 2D and store z-elevations
    std::unordered_map<std::pair<double, double>, double, PairHash> elevations{};
    std::vector<Point2> points2d = transformTo2D(m_meshes[index].layer.points, elevations);

    // Perform 2D triangulation
    CDT2 dt{};
    dt.insert(points2d.begin(), points2d.end());

    auto& surfaceMesh = m_meshes[index].surfaceMesh;
    surfaceMesh.reserve(dt.number_of_faces() * 3, 0, dt.number_of_faces());

    // Convert 2D triangulation to 3D surface mesh
    construct3DSurfaceMesh(dt, surfaceMesh, elevations);
}

void ModellerSet::extrude(float lowestZ, int index)
{
    Logger::log(LogLevel::INFO, ModellerSet::s_logPrefix + "Extruding surface for layer " + std::to_string(index) + "...");

    auto& layerBody = m_meshes[index].layerBody;

    const float currLowestZ = getMinimumZ(m_meshes[index].layer.points);
    Vector3 extrudeVector(0.0, 0.0, -(currLowestZ - lowestZ));  // extrude vector should point downwards
    CGAL::Polygon_mesh_processing::extrude_mesh(m_meshes[index].surfaceMesh, layerBody, extrudeVector);

    flattenBottomSurface(layerBody, lowestZ);

    repair(layerBody);

    m_extrudedMeshes[index] = layerBody;
}

void ModellerSet::flattenBottomSurface(SurfaceMesh& mesh, float zVal) const
{
    for (auto f : faces(mesh))
    {
        auto normal = CGAL::Polygon_mesh_processing::compute_face_normal(f, mesh);
        if (normal.z() < 0) // normal is pointing downwards -> bottom face
        {
            for (auto v : vertices_around_face(mesh.halfedge(f), mesh))
            {
                auto& point = mesh.point(v);
                point = Point3(point.x(), point.y(), zVal);
            }
        }
    }
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
    const float lowestZ = getMinimumZ(m_meshes[numMeshes - 1].layer.points) - s_underLowestZ;   // below the lowest point of bottom layer

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

