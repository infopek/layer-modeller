#include <triangulation/delaunay.h>

Triangulation::Triangulation(const std::vector<Point>& points)
    : m_points{ points }
{
    if (points.size() < 1)
        std::cerr << "Points array is empty!\n";

    triangulate();
}

Triangulation::~Triangulation()
{

}


void Triangulation::triangulate()
{
    std::vector<Point2> points2d;
    points2d.reserve(m_points.size());
    for (const auto& p : m_points)
        points2d.emplace_back(p.x, p.y);

    m_dt.insert(points2d.begin(), points2d.end());
}