#include <geotiff_handler.h>
#include <layer_builder.h>
#include <modeller/modeller_set.h>
#include <renderer.h>
#include <blur/blur.h>
#include "mainwindow.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <unordered_map>
#include <utility>

#include <fstream>
#include <sstream>

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

static void tiffToImage(const std::string& inputPath, const std::string& outputPath)
{
    cv::Mat tiffImage = cv::imread(inputPath, cv::IMREAD_UNCHANGED);
    cv::imwrite(outputPath, tiffImage);
}

static std::vector<Point> processTiff(const std::string& tiffImagePath)
{
    cv::Mat blurred = cv::imread(tiffImagePath, cv::IMREAD_UNCHANGED);
    Blur::medianFilter(blurred.data, blurred.data, blurred.cols, blurred.rows, 25);
    Blur::gaussFilter(blurred.data, blurred.data, blurred.cols, blurred.rows, 25, 3.4f);

    std::vector<Point> points{};
    points.reserve(blurred.rows * blurred.cols);
    for (int y = 0; y < blurred.rows; ++y)
    {
        for (int x = 0; x < blurred.cols; ++x)
        {
            uchar z = blurred.at<uchar>(y, x);

            points.emplace_back(x, y, z);
        }
    }

    cv::imwrite("../../../res/blurred/blurred_output.png", blurred);
    return points;
}

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    MainWindow window;
    window.show();
    return app.exec();
    // tiffToImage("../../../res/tiff/earthdata_1.tif", "../../../res/intermediate/tiff_png.png");
    // auto points = processTiff("../../../res/intermediate/tiff_png.png");

    // LayerBuilder layerBuilder(points);
    // const std::string region = "region";
    // const std::string observationDataPath = "../../../res/boreholes/borehole_kovago.json";
    // const std::string tiffPath = "../../../res/tiff/pecs.tif";

    // LayerBuilder layerBuilder(region, observationDataPath, tiffPath);

    // ModellerSet modeller(layerBuilder);
    // modeller.createMeshes();

    // Renderer renderer{};
    // renderer.addMeshes(modeller.getMeshes());

    // // Describe what you want to be rendered
    // renderer.prepareEdges();
    // renderer.prepareSurfaces();
    // renderer.prepareLayerBody();

    // // Render
    // renderer.render();


    return 0;
}
