#include <triangulation/delaunay.h>
#include <delaunay_renderer.h>

#include <perlin-noise/perlin_noise.h>

#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <unordered_map>
#include <utility>

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
            double z = perlin.octave2D_01(x, y, 4) * 5.0;
            // double z = (x * x - y * y) / 2.0;
            points.emplace_back(x * 7.0 + distribution(rng), y * 7.0 + distribution(rng), z * distribution(rng));
        }
    }

    return points;
}

int main(int argc, char* argv[])
{
    std::vector<Point> points = generatePoints(20, 20);

    Triangulation triangulation(points);
    triangulation.triangulate();

    DelaunayRenderer renderer(triangulation);
    renderer.addPoints();
    renderer.addTriangulation();
    renderer.render();

    return 0;
}