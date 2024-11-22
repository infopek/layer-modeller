#include "initialization.h"

// void readObservationDataFromJson(std::vector<std::pair<std::string, LithologyData>>& lithologyVector, const std::string& path)
// {
//     std::ifstream f(path);
//     json dataJson = json::parse(f);
//     for (const auto& entry : dataJson)
//     {
//         std::cout << entry;
//         Point point;
//         std::string lithoType = entry["reteg$lito$geo"];
//         point.x = static_cast<double>(entry["eovX"]);
//         point.y = static_cast<double>(entry["eovY"]);
//         point.z = entry["reteg$mig"].get<double>();
//         auto it = std::find_if(lithologyVector.begin(), lithologyVector.end(), [&lithoType](const auto &pair) {
//             return pair.first == lithoType;
//             });

//         if (it != lithologyVector.end())
//         {
//             it->second.points.push_back(point);
//         }
//         else
//         {
//             LithologyData newData;
//             newData.stratumName = entry["reteg$lito$nev"];
//             newData.points.push_back(point);
//             lithologyVector.push_back({ lithoType, newData });
//         }
//     }

//     std::sort(lithologyVector.begin(), lithologyVector.end(), [](const auto& a, const auto& b) {
//         const auto& pointsA = a.second.points;
//         const auto& pointsB = b.second.points;

//         double avgMigA = std::accumulate(pointsA.begin(), pointsA.end(), 0.0,
//             [](double sum, const Point& p) { return sum + p.z; }) / pointsA.size();

//         double avgMigB = std::accumulate(pointsB.begin(), pointsB.end(), 0.0,
//             [](double sum, const Point& p) { return sum + p.z; }) / pointsB.size();

//         return avgMigA < avgMigB;
//         });


// }
#include <set>
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