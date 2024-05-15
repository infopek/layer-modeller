#include <layer_builder.h>
#include <modeller/modeller_set.h>
#include <renderer.h>
// #include <kriging_cpu.h>
// #include <blur/blur.h>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <unordered_map>
#include <utility>

#include <fstream>
#include <sstream>

// static std::vector<Point> generatePoints(int width, int height)
// {
//     std::mt19937 rng(0);
//     std::uniform_real_distribution<double> distribution(1.0, 10.0);

//     const siv::PerlinNoise::seed_type seed = 123456u;
//     const siv::PerlinNoise perlin{ seed };

//     int numPoints = width * height;
//     std::vector<Point> points;
//     points.reserve(numPoints);

//     for (int y = -height / 2; y <= height / 2; y++)
//     {
//         for (int x = -width / 2; x <= width / 2; x++)
//         {
//             double z = perlin.octave2D_01(x, y, 4) * 5.0;
//             // double z = (x * x - y * y) / 2.0;
//             points.emplace_back(x * 7.0 + distribution(rng), y * 7.0 + distribution(rng), z * distribution(rng));
//         }
//     }

//     return points;
// }

static std::vector<Point> generatePoints(const std::string& filename)
{
    std::vector<Point> points{};
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file " << filename << std::endl;
        return points;
    }

    std::string line{};
    const int skip = 20;
    int i = 0;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        Point p;
        if (!(iss >> p.x >> p.y >> p.z))
        {
            std::cerr << "Error parsing line: " << line << std::endl;
            continue;
        }

        if (i % skip == 0)
            points.push_back(p);

        ++i;
    }
    file.close();

    return points;
}

// static void tiffToImage(const std::string& filepath)
// {
//     cv::Mat tiffImage = cv::imread(filepath, cv::IMREAD_UNCHANGED);

//     std::string outputFile = "../../../res/intermediate/tiff_png.png";
//     cv::imwrite(outputFile, tiffImage);
// }

// static void processTiff(const std::string& tiffImagePath)
// {
//     cv::Mat blurred = cv::imread(tiffImagePath, cv::IMREAD_UNCHANGED);
//     Blur::medianFilter(blurred.data, blurred.data, blurred.cols, blurred.rows, 25);
//     Blur::gaussFilter(blurred.data, blurred.data, blurred.cols, blurred.rows, 25, 3.4f);
//     cv::imwrite("../../../res/blurred/blurred_output.png", blurred);
// }

int main(int argc, char* argv[])
{
    // tiffToImage("../../../res/tiff/earthdata_1.tif");
    // processTiff("../../../res/intermediate/tiff_png.png");

    // Get points(from interpolator ? )
    std::vector<Point> points = generatePoints("../../../res/points/interpolated_points.txt");

    // std::vector<Triangulation> triangulations{};
    // triangulations.reserve(6);
    // for (size_t offset = 0; offset < 6; offset++)
    // {
    //     // To be removed
    //     for (auto& point : points)
    //     {
    //         point.z -= offset * 5.0;
    //     }

    //     // Create 3D mesh
    //     Triangulation triangulation(points);
    //     triangulations.push_back(triangulation);
    // }

    LayerBuilder layerBuilder(points);

    ModellerSet modeller(layerBuilder);
    modeller.createMeshes();

    Renderer renderer{};
    renderer.addMeshes(modeller.getMeshes());

    // Describe what you want to be rendered
    renderer.prepareEdges();
    renderer.prepareSurfaces();
    renderer.prepareLayerBody();

    // Render
    renderer.render();

    return 0;
}
