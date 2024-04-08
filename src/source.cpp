#include "./triangulation-renderer/triangulation_renderer.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/property_map.h>

#include <perlin-noise/perlin_noise.h>

#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <unordered_map>
#include <utility>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using DT2 = CGAL::Delaunay_triangulation_2<K>;
using Point2 = K::Point_2;
using Point3 = K::Point_3;

static std::vector<Point> generatePoints(int width, int height)
{
    std::mt19937 rng(0);
    std::uniform_real_distribution<double> distribution(1.0, 10.0);

    const siv::PerlinNoise::seed_type seed = 123456u;
    const siv::PerlinNoise perlin{ seed };

    int numPoints = width * height;
    std::vector<Point> points;
    points.reserve(numPoints);

    for (int y = -height / 2; y <= height / 2; y++)
    {
        for (int x = -width / 2; x <= width / 2; x++)
        {
            double z = perlin.octave2D_01((x), (y), 4) * 5.0;
            // double z = (x * x - y * y) / 2.0;
            points.emplace_back(x * 7.0, y * 7.0, z * distribution(rng));
        }
    }

    return points;
}

static DT2 triangulate(const std::vector<Point>& points)
{
    // Triangulate in 2d first
    std::vector<Point2> points2d;
    points2d.reserve(points.size());
    for (const auto& p : points)
        points2d.emplace_back(p.x, p.y);

    DT2 dt;
    dt.insert(points2d.begin(), points2d.end());
    return dt;
}

int main(int argc, char* argv[])
{
    auto points = generatePoints(90, 90);

    DT2 dt = triangulate(points);

    TriangulationRenderer renderer(dt, points);
    renderer.addCoordinateSystem();
    renderer.addPoints();
    renderer.addTriangulation();

    renderer.render();

    return 0;
}
