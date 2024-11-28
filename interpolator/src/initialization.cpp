#include "initialization.h"
#include <set>

// Function to generate random points with elevation
std::vector<Point> generatePoints(int numPoints, double mapSize, double deviationScale, double elevationMin, double elevationMax, siv::PerlinNoise& perlin, double noiseFrequency) {
    std::vector<Point> points;
    std::random_device rd;  // Seed for the random number generator
    std::mt19937 gen(rd());
    
    // Uniform distribution for fully random points
    std::uniform_real_distribution<> uniformDist(0.0, mapSize);
    
    // Gaussian distribution for clustered points
    std::normal_distribution<> gaussianDist(mapSize / 2.0, mapSize * deviationScale);

    for (int i = 0; i < numPoints; ++i) {
        double x, y, z;
        if (deviationScale <= 0.0) {
            // Fully random (uniform distribution)
            x = uniformDist(gen);
            y = uniformDist(gen);
        } else {
            // Gaussian distribution
            x = gaussianDist(gen);
            y = gaussianDist(gen);
            std::cout<<x<<"x y"<<y<<std::endl;
            // Ensure points are within the map bounds
            x = std::max(0.0, std::min(mapSize, x));
            y = std::max(0.0, std::min(mapSize, y));
        }
        
        // Generate elevation (z) using Perlin noise
        double noiseValue = perlin.noise2D(x * noiseFrequency, y * noiseFrequency); // Get Perlin noise value for (x, y)
        noiseValue = (noiseValue + 1.0) / 2.0; // Normalize noise to [0, 1]
        z = elevationMin + noiseValue * (elevationMax - elevationMin); // Scale to elevation range
        
        points.push_back({x, y, z});
    }
    
    return points;
}

void generateTestData(std::vector<std::pair<std::string, LithologyData>>& lithologyVector){
    const int numLayersX = 10;     // Number of layers in X direction
    const int numLayersY = 10;     // Number of layers in Y direction
    const int minPoints = 10;      // Minimum number of points per layer
    const int maxPoints = 30;      // Maximum number of points per layer
    const int pointStep = 2;       // Step size for number of points
    const double mapSize = 100.0;  // Map size (100x100)
    const double elevationMin = 0.0;
    const double elevationMax = 100.0;
    const double noiseFrequency = 0.05; 

    std::random_device rd;
    siv::PerlinNoise perlin(rd());

    // Generate layers
    for (int i = 0; i < numLayersX; ++i) {
        for (int j = 0; j < numLayersY; ++j) {
            int numPoints = minPoints + (i * numLayersY + j) % ((maxPoints - minPoints) / pointStep + 1) * pointStep;
            double deviationScale = fmod(0.1 + 0.02 * (i * numLayersY + j), 5); // Example deviation scale variation

            // Create layer name
            std::string layerName = std::to_string(numPoints) + "_" + std::to_string(deviationScale) + "_test";

            LithologyData newData;
            newData.stratumName = layerName;
            newData.points= generatePoints(numPoints, mapSize, deviationScale, elevationMin, elevationMax, perlin, noiseFrequency);
            lithologyVector.emplace_back(layerName, newData);
        }
    }

}
void readObservationDataFromJson(std::vector<std::pair<std::string, LithologyData>>& lithologyVector, const std::string& path)
{
    std::ifstream f(path);
    json dataJson = json::parse(f);

    std::unordered_map<std::string, std::vector<Point>> boreholePositions;
    for (const auto& entry : dataJson)
    {
        Point point;
        std::string lithoType = entry["reteg$lito$geo"];
        point.x = static_cast<double>(entry["eovX"]);
        point.y = static_cast<double>(entry["eovY"]);
        point.z = entry["reteg$mig"].get<double>();
        std::string jelszam = entry["jelszam"].get<std::string>();

        boreholePositions[jelszam].push_back(point);

        auto it = std::find_if(lithologyVector.begin(), lithologyVector.end(), [&lithoType](const auto& pair) {
            return pair.first == lithoType;
        });

        if (it != lithologyVector.end())
        {
            it->second.points.push_back(point);
        }
        else
        {
            LithologyData newData;
            newData.stratumName = entry["reteg$lito$nev"];
            newData.points.push_back(point);
            lithologyVector.push_back({ lithoType, newData });
        }
    }

    std::set<std::pair<double, double>> commonCoordinates;
    filterMaxElements(boreholePositions);
    for(auto position:boreholePositions){
        auto point = position.second[0];
        commonCoordinates.insert({ point.x, point.y });
    }

    for (auto& [lithoType, lithologyData] : lithologyVector)
    {
        auto& points = lithologyData.points;

        std::vector<Point> filteredPoints;
        for (const auto& point : points)
            if (commonCoordinates.count({ point.x, point.y }))
                filteredPoints.push_back(point);

        if (!filteredPoints.empty())
        {
            lithologyData.averageDepth =
            std::accumulate(
                filteredPoints.begin(),
                filteredPoints.end(),
                0.0,
                [](double sum, const Point& p) {
                    return sum + p.z;    
                }
            )/ filteredPoints.size();
        }
        else
        {
            lithologyData.averageDepth = 0.0;
        }
    }

    std::sort(lithologyVector.begin(), lithologyVector.end(), [](const auto& a, const auto& b) {
        return a.second.averageDepth < b.second.averageDepth;
    });
}
void filterMaxElements(std::unordered_map<std::string, std::vector<Point>>& map) {
    size_t maxSize = 0;
    for (const auto& pair : map)
        maxSize = std::max(maxSize, pair.second.size());
    for (auto it = map.begin(); it != map.end();) {
        if (it->second.size() < maxSize) it = map.erase(it);
        else ++it;
    }
}