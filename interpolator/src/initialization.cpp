#include "initialization.h"
#include <set>

std::vector<Point> generatePoints(int numPoints, double mapSize, double deviationScale, std::vector<Point> testLayer)
{
    std::vector<Point> points;
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> uniformDist(0.0, mapSize);

    std::normal_distribution<> gaussianDist(mapSize / 2.0, mapSize * deviationScale);

    for (int i = 0; i < numPoints; ++i)
    {
        double x, y, z;
        if (deviationScale <= 0.0)
        {
            x = uniformDist(gen);
            y = uniformDist(gen);
        }
        else
        {
            x = gaussianDist(gen);
            y = gaussianDist(gen);
        }
        x = std::min(std::max(x, 0.0), static_cast<double>(mapSize - 1));
        y = std::min(std::max(y, 0.0), static_cast<double>(mapSize - 1));

        int xIndex = static_cast<int>(x);
        int yIndex = static_cast<int>(y);

        auto point = testLayer[xIndex * mapSize + yIndex];
        points.push_back(point);
    }
    return points;
}

void generateTestData(std::vector<std::pair<std::string, LithologyData>>& lithologyVector, std::vector<Point>& testLayer)
{
    const int numLayersX = 21;    // Number of layers in X direction
    const int numLayersY = 21;    // Number of layers in Y direction
    const int minPoints = 10;     // Minimum number of points per layer
    const int maxPoints = 30;     // Maximum number of points per layer
    const int pointStep = 1;      // Step size for number of points
    const double mapSize = 100.0; // Map size (100x100)
    const double elevationMin = 30.0;
    const double elevationMax = 100.0;
    const double noiseFrequency = 0.01;

    siv::PerlinNoise perlin(1000);
    for (double x = 0; x < mapSize; x++)
    {
        for (double y = 0; y < mapSize; y++)
        {
            double noiseValue = perlin.noise2D(x * noiseFrequency, y * noiseFrequency);
            noiseValue = (noiseValue + 1.0) / 2.0;
            double z = elevationMin + noiseValue * (elevationMax - elevationMin);      
            testLayer.push_back({x, y, z});
        }
    }

    // Generate layers
    for (int i = 0; i < numLayersX; ++i)
    {
        for (int j = 0; j < numLayersY; ++j)
        {
            int numPoints = minPoints + i * pointStep; // % ((maxPoints - minPoints) / pointStep + 1) * pointStep;
            double deviationScale = 0.1 + 0.01 * j;

            // Create layer name
            std::string layerName = std::to_string(numPoints) + "_" + std::format("{:.3f}", deviationScale) + "_test";

            LithologyData newData;
            newData.stratumName = layerName;
            newData.points = generatePoints(numPoints, mapSize, deviationScale,testLayer);
            lithologyVector.emplace_back(layerName, newData);
        }
    }
}
void readObservationDataFromJson(std::vector<std::pair<std::string, LithologyData>> &lithologyVector, const std::string &path)
{
    std::ifstream f(path);
    json dataJson = json::parse(f);

    std::unordered_map<std::string, std::vector<Point>> boreholePositions;
    for (const auto &entry : dataJson)
    {
        Point point;
        std::string lithoType = entry["reteg$lito$geo"];
        point.x = static_cast<double>(entry["eovX"]);
        point.y = static_cast<double>(entry["eovY"]);
        point.z = entry["reteg$mig"].get<double>();
        std::string jelszam = entry["jelszam"].get<std::string>();

        boreholePositions[jelszam].push_back(point);

        auto it = std::find_if(lithologyVector.begin(), lithologyVector.end(), [&lithoType](const auto &pair)
                               { return pair.first == lithoType; });

        if (it != lithologyVector.end())
        {
            it->second.points.push_back(point);
        }
        else
        {
            LithologyData newData;
            newData.stratumName = entry["reteg$lito$nev"];
            newData.points.push_back(point);
            lithologyVector.push_back({lithoType, newData});
        }
    }

    std::set<std::pair<double, double>> commonCoordinates;
    filterMaxElements(boreholePositions);
    for (auto position : boreholePositions)
    {
        auto point = position.second[0];
        commonCoordinates.insert({point.x, point.y});
    }

    for (auto &[lithoType, lithologyData] : lithologyVector)
    {
        auto &points = lithologyData.points;

        std::vector<Point> filteredPoints;
        for (const auto &point : points)
            if (commonCoordinates.count({point.x, point.y}))
                filteredPoints.push_back(point);

        if (!filteredPoints.empty())
        {
            lithologyData.averageDepth =
                std::accumulate(
                    filteredPoints.begin(),
                    filteredPoints.end(),
                    0.0,
                    [](double sum, const Point &p)
                    {
                        return sum + p.z;
                    }) /
                filteredPoints.size();
        }
        else
        {
            lithologyData.averageDepth = 0.0;
        }
    }

    std::sort(lithologyVector.begin(), lithologyVector.end(), [](const auto &a, const auto &b)
              { return a.second.averageDepth < b.second.averageDepth; });
}
void filterMaxElements(std::unordered_map<std::string, std::vector<Point>> &map)
{
    size_t maxSize = 0;
    for (const auto &pair : map)
        maxSize = std::max(maxSize, pair.second.size());
    for (auto it = map.begin(); it != map.end();)
    {
        if (it->second.size() < maxSize)
            it = map.erase(it);
        else
            ++it;
    }
}