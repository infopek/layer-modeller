#include "initialization.h"

void readObservationDataFromJson(std::vector<std::pair<std::string, LithologyData>> &lithologyVector, const std::string &path)
{
    std::ifstream f(path);
    json dataJson = json::parse(f);
    for (const auto &entry : dataJson)
    {
        Point point;
        std::string lithoType = entry["reteg_lito_geo"];
        point.x = static_cast<double>(entry["x"]);
        point.y = static_cast<double>(entry["y"]);
        point.z = entry["reteg_mig"].get<double>();
        auto it = std::find_if(lithologyVector.begin(), lithologyVector.end(), [&lithoType](const auto &pair) {
            return pair.first == lithoType;
        });

        if (it != lithologyVector.end())
        {
            it->second.points.push_back(point);
        }
        else
        {
            LithologyData newData;
            newData.stratumName = entry["reteg_lito_nev"];
            newData.points.push_back(point);
            lithologyVector.push_back({lithoType, newData});
        }
    }
    std::sort(lithologyVector.begin(), lithologyVector.end(), [](const auto &a, const auto &b) {
        const auto &pointsA = a.second.points;
        const auto &pointsB = b.second.points;

        double avgMigA = std::accumulate(pointsA.begin(), pointsA.end(), 0.0, 
            [](double sum, const Point &p) { return sum + p.z; }) / pointsA.size();

        double avgMigB = std::accumulate(pointsB.begin(), pointsB.end(), 0.0, 
            [](double sum, const Point &p) { return sum + p.z; }) / pointsB.size();

        return avgMigA < avgMigB; 
    });

    std::vector<std::string> reteglitonev;
    for (const auto &entry : lithologyVector)
    {
        reteglitonev.push_back(entry.second.stratumName);
    }

}
